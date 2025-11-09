# veRL Multi-turn RL Integration for Debate Town Halls

This document describes the veRL (Versatile Reinforcement Learning) integration for training political debate agents using multi-turn reinforcement learning.

## Overview

The veRL integration enables training a politician agent to debate effectively and win votes through multi-turn conversations with villagers. The system uses PPO (Proximal Policy Optimization) to optimize the politician's debate strategy based on voting outcomes.

## Architecture

### Components

1. **RL Endpoints** (`/rl/townhall/start` and `/rl/townhall/step` in `main.py`)
   - Session-based API for multi-turn RL training
   - Manages debate state, turn scheduling, and reward calculation
   - One politician is the RL actor (veRL policy), the other uses the normal LLM

2. **Debate Interaction** (`verl_integration/debate_interaction.py`)
   - veRL interaction class that wraps the RL endpoints
   - Implements the veRL multi-turn interface
   - Handles session management and reward aggregation

3. **Modal Deployment** (`modal_verl_training.py`)
   - Deploys FastAPI backend as a Modal web endpoint
   - Runs veRL PPO training on 8xH100 GPUs
   - Configures distributed training infrastructure

4. **Configuration Files** (`config/`)
   - `interaction.yaml`: Debate interaction configuration
   - `rollout.yaml`: SGLang rollout configuration for multi-turn
   - `train_ppo.yaml`: PPO training hyperparameters

## How It Works

### Turn Structure

Each debate episode follows this structure:

1. **Opening Phase**
   - Actor (RL politician) gives opening statement
   - Non-actor (normal LLM politician) gives opening statement
   - All villagers hear both opening statements

2. **Villager Interaction Phase** (repeated for each villager)
   - Villager speaks (asks question or makes comment)
   - Actor responds to the villager
   - Non-actor responds to the villager
   - Move to next villager

3. **Terminal Phase**
   - All villagers vote using LLM-based decision making
   - Reward is calculated based on voting outcomes
   - Episode ends

### Reward Calculation

The reward is calculated from villager votes (5 villagers total):

- **margin_normalized** (default): `(votes_for_actor - votes_for_non_actor) / 5.0` → Range: [-1.0, 1.0]
- **margin**: `votes_for_actor - votes_for_non_actor` → Range: [-5, 5]
- **votes_normalized**: `votes_for_actor / 5.0` → Range: [0.0, 1.0]
- **votes**: `votes_for_actor` → Range: [0, 5]

### Actor Selection

When a session starts, one politician is randomly selected as the actor (the RL policy being trained). The other politician always uses the normal LLM endpoint. This can be forced using the `force_actor` parameter.

## Usage

### Local Testing

1. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-verl.txt
```

2. Start the FastAPI backend:
```bash
python main.py
```

3. Test the RL endpoints:
```bash
# Start a session
curl -X POST http://localhost:8000/rl/townhall/start \
  -H "Content-Type: application/json" \
  -d '{"topic": "immigration", "max_assistant_turns": 6}'

# Step through the session (use session_id from start response)
curl -X POST http://localhost:8000/rl/townhall/step \
  -H "Content-Type: application/json" \
  -d '{"session_id": "YOUR_SESSION_ID", "actor_message": "I believe in fair immigration policies..."}'
```

### Modal Deployment and Training

1. Ensure Modal is configured:
```bash
modal token set
```

2. Deploy and run training on 8xH100:
```bash
modal run modal_verl_training.py --num-episodes 5 --topic immigration
```

This will:
- Deploy the FastAPI backend as a Modal web endpoint
- Run veRL PPO training on 8xH100 GPUs for 5 episodes
- Train the politician to optimize debate performance

### Configuration

#### Interaction Config (`config/interaction.yaml`)

```yaml
interaction:
  - name: "debate_townhall"
    class_name: "verl_integration.debate_interaction.DebateTownHallInteraction"
    config:
      base_url: "http://localhost:8000"  # Backend URL
      topic: "immigration"                # Debate topic
      num_rounds: 1                       # Number of rounds
      max_assistant_turns: 6              # Max actor turns (1 opening + 5 villager responses)
      reward_type: "margin_normalized"    # Reward calculation type
      timeout_s: 30.0                     # HTTP timeout
```

#### Rollout Config (`config/rollout.yaml`)

```yaml
actor_rollout_ref:
  rollout:
    name: sglang                          # Use SGLang backend for multi-turn
    mode: sync                            # Synchronous mode (more stable)
    multi_turn: true                      # Enable multi-turn rollout
    multi_turn_interaction_config_path: ./config/interaction.yaml
    temperature: 0.7                      # Sampling temperature
    top_p: 0.9                           # Nucleus sampling
    max_new_tokens: 256                  # Max tokens per response
```

#### Training Config (`config/train_ppo.yaml`)

```yaml
model:
  model_name_or_path: "Qwen/Qwen2.5-3B-Instruct"
  
trainer:
  total_training_steps: 10              # Number of training episodes
  save_freq: 5                          # Save checkpoint every N steps
  
ppo:
  learning_rate: 1.0e-5                 # PPO learning rate
  ppo_epochs: 2                         # PPO update epochs per batch
  clip_range: 0.2                       # PPO clip range
```

## API Reference

### POST /rl/townhall/start

Initialize a new RL town hall session.

**Request:**
```json
{
  "topic": "immigration",
  "num_rounds": 1,
  "max_assistant_turns": 6,
  "force_actor": null,
  "reward_type": "margin_normalized"
}
```

**Response:**
```json
{
  "session_id": "uuid",
  "actor_id": "politician_1",
  "non_actor_id": "politician_2",
  "next_message": "You are Alex, a politician...",
  "meta": {
    "session_id": "uuid",
    "actor_id": "politician_1",
    "phase": "opening",
    "assistant_turns_used": 0
  }
}
```

### POST /rl/townhall/step

Process one step of the RL town hall session.

**Request:**
```json
{
  "session_id": "uuid",
  "actor_message": "I support fair immigration policies..."
}
```

**Response:**
```json
{
  "terminal": false,
  "next_message": "The villager Sarah the Waitress just said: ...",
  "reward": 0.0,
  "meta": {
    "session_id": "uuid",
    "actor_id": "politician_1",
    "phase": "villager",
    "villager_idx": 1,
    "assistant_turns_used": 2
  }
}
```

When `terminal: true`, the response includes voting results:
```json
{
  "terminal": true,
  "next_message": "",
  "reward": 0.4,
  "meta": {
    "votes": {
      "politician_1": 4,
      "politician_2": 1,
      "undecided": 0
    },
    "voting_decisions": [...],
    "terminal_reason": "all_villagers_spoke"
  }
}
```

## Training Tips

1. **Start Small**: Begin with 5-10 episodes to verify the setup works
2. **Monitor Rewards**: Check that rewards are in the expected range [-1, 1] for margin_normalized
3. **Adjust Learning Rate**: If training is unstable, reduce learning_rate to 5e-6 or lower
4. **Topic Selection**: Different topics (immigration, budget) may require different strategies
5. **Actor Selection**: Use `force_actor` to train specific politicians deterministically

## Troubleshooting

### Session Not Found Error
- Ensure the backend is running and accessible
- Check that session_id matches the one from /rl/townhall/start

### Timeout Errors
- Increase `timeout_s` in interaction config
- Check backend logs for LLM call failures
- Verify Modal inference endpoint is responding

### Low Rewards
- Check voting_decisions in terminal meta to see why villagers voted
- Adjust debate topic or politician system prompts
- Increase max_assistant_turns to give more interaction time

### Training Crashes
- Reduce batch size if OOM on GPUs
- Check that all config paths are correct
- Verify veRL is installed: `pip list | grep verl`

## Files Created

- `main.py`: Added RL endpoints and session management
- `verl_integration/debate_interaction.py`: veRL interaction class
- `verl_integration/__init__.py`: Package init
- `config/interaction.yaml`: Interaction configuration
- `config/rollout.yaml`: Rollout configuration
- `config/train_ppo.yaml`: Training configuration
- `modal_verl_training.py`: Modal deployment script
- `requirements-verl.txt`: veRL dependencies
- `VERL_INTEGRATION.md`: This documentation

## Next Steps

1. Run initial training on Modal with 5-10 episodes
2. Analyze voting patterns and reward trends
3. Tune hyperparameters based on results
4. Scale up to longer training runs (100+ episodes)
5. Experiment with different debate topics and strategies
6. Compare performance of politician_1 vs politician_2 as actor

## References

- [veRL Documentation](https://verl.readthedocs.io/)
- [veRL Multi-turn Guide](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html)
- [SGLang Backend](https://verl.readthedocs.io/en/latest/workers/sglang_worker.html)
