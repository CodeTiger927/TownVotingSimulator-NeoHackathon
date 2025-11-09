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
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.0",
        "transformers==4.36.0",
        "accelerate==0.25.0",
        "datasets==2.15.0",
        "peft==0.7.0",
        "bitsandbytes==0.41.3",
        "scipy==1.11.4",
        "sentencepiece==0.1.99",
        "protobuf==4.25.1",
        "httpx==0.25.2",
        "pyyaml==6.0.1",
        "hydra-core==1.3.2",
    )
    .apt_install("git")
    .run_commands(
        "pip install git+https://github.com/volcengine/verl.git@main",
    )
    .add_local_dir(".", remote_path="/root")
)

backend_volume = modal.Volume.from_name("townhall-backend-data", create_if_missing=True)

@app.function(
    image=image,
    gpu=None,
    scaledown_window=300,
    volumes={"/data": backend_volume},
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
    volumes={"/data": backend_volume},
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
    
    rollout_config = {
        "actor_rollout_ref": {
            "rollout": {
                "name": "sglang",
                "mode": "sync",
                "multi_turn": True,
                "multi_turn_interaction_config_path": str(config_dir / "interaction.yaml"),
                "log_prob_micro_batch_size": 4,
                "temperature": 0.7,
                "top_p": 0.9,
                "max_new_tokens": 256,
            }
        }
    }
    
    with open(config_dir / "rollout.yaml", "w") as f:
        yaml.dump(rollout_config, f)
    
    train_config = {
        "model": {
            "model_name_or_path": model_name,
            "trust_remote_code": True,
        },
        "trainer": {
            "total_epochs": 1,
            "total_training_steps": num_episodes,
            "save_freq": max(num_episodes // 4, 1),
        },
        "algorithm": {
            "kl_ctrl": {
                "kl_coef": 0.05,
            },
            "adv_estimator": {
                "gamma": 0.99,
                "lam": 0.95,
            },
        },
        "ppo": {
            "num_mini_batches": 2,
            "ppo_mini_batch_size": batch_size,
            "ppo_epochs": 2,
            "clip_range": 0.2,
            "clip_range_value": 0.2,
            "learning_rate": learning_rate,
        },
    }
    
    with open(config_dir / "train_ppo.yaml", "w") as f:
        yaml.dump(train_config, f)
    
    print(f"Created config files in {config_dir}")
    print(f"Interaction config: {interaction_config}")
    print(f"Rollout config: {rollout_config}")
    print(f"Training config: {train_config}")
    
    print("\n" + "="*80)
    print("veRL Training Configuration Summary")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Backend: {backend_url}")
    print(f"Topic: {topic}")
    print(f"Episodes: {num_episodes}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"GPUs: 8xH100")
    print("="*80 + "\n")
    
    print("Note: Full veRL training integration requires:")
    print("1. veRL package installed with all dependencies")
    print("2. SGLang backend configured for multi-turn rollout")
    print("3. Proper model loading and distributed training setup")
    print("\nThis is a placeholder for the full training loop.")
    print("To run actual training, use: python -m verl.trainer.main_ppo --config-path=./config --config-name=train_ppo")
    
    return {
        "status": "training_configured",
        "config_dir": str(config_dir),
        "backend_url": backend_url,
        "model": model_name,
        "num_episodes": num_episodes,
        "message": "Training configuration created. Full training requires veRL trainer setup."
    }


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
    print("Deploying FastAPI backend...")
    backend_app = backend_asgi.spawn()
    
    import time
    time.sleep(5)
    
    backend_url = backend_app.web_url
    print(f"Backend deployed at: {backend_url}")
    
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
