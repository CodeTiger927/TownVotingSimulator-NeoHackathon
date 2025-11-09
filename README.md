# TownVotingSimulator-NeoHackathon

An election voting simulation featuring 5 AI-powered villager agents with distinct personalities, each simulated by Qwen3-8B language model running on Modal. Politicians can interact with villagers through dialogue and broadcasts to persuade them to vote.

## Overview

This backend enables interactions between 2 politicians and 5 villager agents in a town hall election simulation. Each villager has unique personality traits, values, and voting preferences that can be influenced through conversation.

## Architecture

- **Backend**: FastAPI server handling agent interactions and state management
- **Inference**: Modal-hosted vLLM server running Qwen3-8B-FP8 for agent responses
- **Storage**: In-memory agent state with conversation history

## The 5 Villager Agents

1. **Sarah the Waitress** - Working class, pro-welfare, liberal, anti-immigration (job concerns)
2. **Margaret the Librarian** - Education-focused, liberal, middle class, pro-health and education spending
3. **Brother Thomas (Monk)** - Spiritual, highly conservative, pro-health spending, selective immigration (religion-based)
4. **Officer James (Police)** - Pro-defense budget, slightly conservative, anti-immigration
5. **Emily the Mother** - Wealthy, liberal, pro-immigration, pro-school funding, anti-police funding

## Setup

### Prerequisites

- Python 3.12+
- Modal account with authentication configured
- pip or poetry for dependency management

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Honyant/TownVotingSimulator-NeoHackathon.git
cd TownVotingSimulator-NeoHackathon
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure Modal authentication:
```bash
modal token set
```

4. Deploy the Modal inference service:
```bash
modal deploy modal_inference.py
```

5. Create a `.env` file with your Modal endpoint:
```bash
cp .env.example .env
# Edit .env and set MODAL_INFERENCE_URL to your deployed Modal endpoint
```

6. Start the FastAPI server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Agent Information

- `GET /` - API information and available endpoints
- `GET /agents` - List all 5 villager agents
- `GET /agent/{agent_name}` - Get specific agent's state and memory

### Interactions

- `POST /talk` - One-on-one conversation with a specific agent
  ```json
  {
    "agent_name": "waitress",
    "politician_id": "politician_1",
    "message": "What are your thoughts on immigration?"
  }
  ```

- `POST /broadcast` - Broadcast message to all agents
  ```json
  {
    "politician_id": "politician_1",
    "message": "I propose increasing school funding by 20%",
    "topic": "budget"
  }
  ```

- `POST /townhall` - Town hall conversation with randomized agent responses
  ```json
  {
    "politician_1_message": "I support open immigration",
    "politician_2_message": "I support controlled immigration",
    "topic": "immigration"
  }
  ```

### Policy Management

- `GET /politicians` - Get current policies for both politicians
- `POST /politician/policy` - Update politician's policy positions
  ```json
  {
    "politician_id": "politician_1",
    "immigration_policy": "Open borders with background checks",
    "budget_policy": {
      "police": "decrease",
      "schools": "increase",
      "welfare": "increase",
      "health": "increase"
    }
  }
  ```

### Voting

- `POST /vote` - Cast a vote from an agent
  ```json
  {
    "agent_name": "waitress",
    "politician_id": "politician_1"
  }
  ```

- `GET /results` - Get current voting results

### Utility

- `POST /reset` - Reset all agent memories and politician policies

## Policy Topics

Politicians can address two main policy areas:

1. **Immigration**: Who should be allowed to join the village?
2. **Budget**: How to allocate spending across:
   - Police/Defense
   - Schools/Education
   - Welfare
   - Health
   - Government

## Agent Personalities

Each agent has a detailed system prompt that defines their:
- Background and occupation
- Political leanings (liberal/conservative)
- Key policy priorities
- Concerns and fears
- Speaking style
- Persuadability factors

See `agent_configs.py` for full personality definitions.

## Modal Inference

The backend uses Modal to host a vLLM server running Qwen3-8B-FP8 model. The inference service:
- Runs on H100 GPU for fast inference
- Uses OpenAI-compatible API format
- Supports streaming responses
- Auto-scales based on demand

The Modal service is defined in `modal_inference.py` and can be deployed independently.

## Development

### Testing Endpoints

Test the API using curl:

```bash
# List agents
curl http://localhost:8000/agents

# Talk to an agent
curl -X POST http://localhost:8000/talk \
  -H "Content-Type: application/json" \
  -d '{"agent_name": "waitress", "politician_id": "politician_1", "message": "Hello!"}'

# Broadcast to all agents
curl -X POST http://localhost:8000/broadcast \
  -H "Content-Type: application/json" \
  -d '{"politician_id": "politician_1", "message": "I support education!", "topic": "budget"}'

# Get voting results
curl http://localhost:8000/results
```

### Project Structure

```
.
├── main.py              # FastAPI backend with all endpoints
├── agent_configs.py     # Agent personality definitions and system prompts
├── modal_inference.py   # Modal vLLM inference service
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (Modal endpoint URL)
└── README.md           # This file
```

## Notes

- First inference request may take 1-2 minutes due to Modal cold start (GPU spin-up)
- Subsequent requests are much faster (~1-5 seconds)
- Agent memories are stored in-memory and will be lost on server restart
- The simulation supports 2 politicians competing for 5 votes

## License

MIT

## Credits

Built for NeoHackathon using Modal, FastAPI, and Qwen3-8B
