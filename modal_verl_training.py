"""
Modal deployment script for veRL multi-turn RL training on 8xH100 GPUs.

This script:
1. Deploys the FastAPI backend as a Modal web endpoint
2. Runs veRL PPO training on 8xH100 GPUs
3. Trains a politician to debate effectively and win votes
"""

import modal
import os
from pathlib import Path

app = modal.App("townhall-verl-training")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0",
        "aiohttp==3.9.1",
        "python-dotenv==1.0.0",
        "pydantic==2.5.0",
        "httpx==0.25.2",
    )
    .add_local_dir(".", remote_path="/root")
)

verl_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        "ray==2.9.3",
        "transformers==4.45.0",
        "accelerate==0.25.0",
        "datasets==2.15.0",
        "peft==0.7.0",
        "bitsandbytes==0.43.0",
        "scipy==1.11.4",
        "sentencepiece==0.1.99",
        "protobuf==4.25.1",
        "httpx==0.25.2",
        "pyyaml==6.0.1",
        "hydra-core==1.3.2",
        "trl==0.7.4",
        "sglang",
    )
    .run_commands(
        "pip install git+https://github.com/volcengine/verl.git@main",
    )
    .add_local_dir(".", remote_path="/root")
)

backend_volume = modal.Volume.from_name("townhall-backend-data", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("hf-cache", create_if_missing=True)

@app.function(
    image=image,
    gpu=None,
    scaledown_window=300,
    volumes={
        "/data": backend_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def backend_asgi():
    """
    ASGI app for the FastAPI backend.
    This serves the RL environment that veRL will interact with.
    """
    import sys
    sys.path.insert(0, "/root")
    
    from main import app as fastapi_app
    return fastapi_app


@app.function(
    image=verl_image,
    gpu="H100:8",
    timeout=3600 * 4,
    volumes={
        "/data": backend_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
    env={"BNB_CUDA_VERSION": "121"},
)
def run_verl_training(
    backend_url: str,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    num_episodes: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    topic: str = "immigration",
):
    """
    Run veRL PPO training on 8xH100 GPUs.
    
    Args:
        backend_url: URL of the deployed FastAPI backend
        model_name: HuggingFace model to train
        num_episodes: Number of training episodes
        batch_size: Batch size for training
        learning_rate: Learning rate for PPO
        topic: Debate topic (immigration, budget, etc.)
    """
    import torch
    import yaml
    import json
    import sys
    from pathlib import Path
    
    print(f"Starting veRL training on 8xH100 GPUs")
    print(f"Backend URL: {backend_url}")
    print(f"Model: {model_name}")
    print(f"Episodes: {num_episodes}")
    print(f"Topic: {topic}")
    
    work_dir = Path("/data/verl_training")
    work_dir.mkdir(parents=True, exist_ok=True)
    
    config_dir = work_dir / "config"
    config_dir.mkdir(exist_ok=True)
    
    datasets_dir = work_dir / "datasets"
    datasets_dir.mkdir(exist_ok=True)
    
    output_dir = work_dir / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    interaction_config = {
        "interaction": [
            {
                "name": "debate_townhall",
                "class_name": "verl_integration.debate_interaction.DebateTownHallInteraction",
                "config": {
                    "base_url": backend_url,
                    "topic": topic,
                    "num_rounds": 1,
                    "max_assistant_turns": 6,
                    "reward_type": "margin_normalized",
                    "timeout_s": 60.0,
                }
            }
        ]
    }
    
    with open(config_dir / "interaction.yaml", "w") as f:
        yaml.dump(interaction_config, f)
    
    batch_size = max(num_episodes * 32, 32)
    dataset_samples = [
        {
            "prompt": "You are a political candidate. Be persuasive and concise.",
        }
        for _ in range(batch_size)
    ]
    
    import pandas as pd
    df = pd.DataFrame(dataset_samples)
    dataset_file = datasets_dir / "debate_samples.parquet"
    df.to_parquet(dataset_file, index=False)
    
    print(f"Created interaction config and dataset in {config_dir}")
    
    print("\n" + "="*80)
    print("veRL Training Configuration Summary")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Backend: {backend_url}")
    print(f"Topic: {topic}")
    print(f"Episodes: {num_episodes}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Dataset: {dataset_file}")
    print(f"Output Dir: {output_dir}")
    
    import torch
    gpu_count = torch.cuda.device_count()
    print(f"GPUs Available: {gpu_count}")
    
    print("\nVerifying PyTorch DTensor support...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    try:
        from torch.distributed.tensor import DTensor
        print("✓ DTensor import successful")
    except ImportError as e:
        print(f"✗ DTensor import failed: {e}")
        print("ERROR: PyTorch version does not support DTensor in torch.distributed.tensor")
        print("This is required by veRL. Please upgrade PyTorch to 2.5.1 or later.")
        return {
            "status": "environment_error",
            "error": "DTensor not available",
            "message": f"PyTorch {torch.__version__} does not expose DTensor in torch.distributed.tensor"
        }
    print("="*80 + "\n")
    
    print("Probing backend URL to verify connectivity...")
    import httpx
    try:
        response = httpx.post(
            f"{backend_url}/rl/townhall/start",
            json={"topic": topic},
            timeout=30.0
        )
        if response.status_code == 200:
            print(f"✓ Backend is reachable and responding")
        else:
            print(f"⚠ Backend returned status {response.status_code}")
    except Exception as e:
        print(f"⚠ Warning: Could not reach backend: {e}")
        print("Continuing anyway - backend may start during training...")
    
    print("\n" + "="*80)
    print("Starting veRL PPO Training")
    print("="*80 + "\n")
    
    import subprocess
    
    training_args = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        f"actor_rollout_ref.rollout.name=sglang",
        f"actor_rollout_ref.rollout.mode=sync",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4",
        f"actor_rollout_ref.rollout.multi_turn.enable=true",
        f"actor_rollout_ref.rollout.multi_turn.interaction_config_path={config_dir/'interaction.yaml'}",
        f"actor_rollout_ref.rollout.multi_turn.max_assistant_turns=6",
        f"actor_rollout_ref.model.path={model_name}",
        f"actor_rollout_ref.model.trust_remote_code=true",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={batch_size}",
        f"critic.model.path={model_name}",
        f"critic.model.trust_remote_code=true",
        f"critic.ppo_micro_batch_size_per_gpu=4",
        f"critic.ppo_mini_batch_size={batch_size}",
        f"data.train_files={dataset_file}",
        f"data.val_files={dataset_file}",
        f"data.train_batch_size={batch_size}",
        f"trainer.total_training_steps={num_episodes}",
        f"trainer.default_local_dir={output_dir}",
        f"trainer.val_before_train=false",
        "trainer.logger=[console]",
    ]
    
    print(f"Running command: {' '.join(training_args)}")
    print("\n" + "-"*80 + "\n")
    
    try:
        result = subprocess.run(
            training_args,
            cwd=str(work_dir),
            check=True,
            capture_output=False,  # Stream to Modal logs
            text=True
        )
        
        print("\n" + "-"*80)
        print("✓ Training completed successfully!")
        print("-"*80 + "\n")
        
        return {
            "status": "training_completed",
            "exit_code": result.returncode,
            "config_dir": str(config_dir),
            "backend_url": backend_url,
            "model": model_name,
            "num_episodes": num_episodes,
            "message": "veRL PPO training completed successfully"
        }
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed with exit code {e.returncode}")
        print(f"Error: {e}")
        return {
            "status": "training_failed",
            "exit_code": e.returncode,
            "config_dir": str(config_dir),
            "backend_url": backend_url,
            "model": model_name,
            "num_episodes": num_episodes,
            "message": f"Training failed with exit code {e.returncode}"
        }
    except Exception as e:
        print(f"\n✗ Unexpected error during training: {e}")
        return {
            "status": "training_error",
            "error": str(e),
            "config_dir": str(config_dir),
            "backend_url": backend_url,
            "model": model_name,
            "num_episodes": num_episodes,
            "message": f"Unexpected error: {str(e)}"
        }


@app.function(
    image=verl_image,
    gpu=None,
    timeout=300,
)
def check_verl_config():
    """
    Check veRL's Hydra config structure to understand available groups and keys.
    """
    import subprocess
    import sys
    import os
    
    print("="*80)
    print("Checking veRL installation and config structure")
    print("="*80 + "\n")
    
    try:
        import verl
        verl_dir = os.path.dirname(verl.__file__)
        print(f"veRL installed at: {verl_dir}\n")
        
        config_path = os.path.join(verl_dir, "config")
        if os.path.exists(config_path):
            print(f"Config directory found at: {config_path}")
            print("Config files:")
            for root, dirs, files in os.walk(config_path):
                for file in files:
                    if file.endswith('.yaml'):
                        filepath = os.path.join(root, file)
                        print(f"  - {filepath}")
            print()
    except Exception as e:
        print(f"Error finding veRL: {e}\n")
    
    print("="*80)
    print("Running: python -m verl.trainer.main_ppo --help")
    print("="*80 + "\n")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "verl.trainer.main_ppo", "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except Exception as e:
        print(f"Error running --help: {e}")
    
    print("\n" + "="*80)
    print("Running: python -m verl.trainer.main_ppo --cfg job")
    print("="*80 + "\n")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "verl.trainer.main_ppo", "--cfg", "job"],
            capture_output=True,
            text=True,
            timeout=30
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except Exception as e:
        print(f"Error running --cfg job: {e}")
    
    return {"status": "complete"}


@app.local_entrypoint()
def main(
    num_episodes: int = 5,
    topic: str = "immigration",
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
):
    """
    Main entrypoint for running veRL training on Modal.
    
    Usage:
        modal run modal_verl_training.py --num-episodes 5 --topic immigration
    """
    import json
    
    backend_url = "https://anthonyindeepspace--townhall-verl-training-backend-asgi-dev.modal.run"
    print(f"Using FastAPI backend at: {backend_url}")
    
    print(f"\nStarting veRL training for {num_episodes} episodes on topic: {topic}")
    
    result = run_verl_training.remote(
        backend_url=backend_url,
        model_name=model_name,
        num_episodes=num_episodes,
        topic=topic,
    )
    
    print("\n" + "="*80)
    print("Training Result:")
    print("="*80)
    print(json.dumps(result, indent=2))
    print("="*80)
    
    return result
