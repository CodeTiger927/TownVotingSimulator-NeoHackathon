"""
Main FastAPI backend for the Town Voting Simulator.
Handles interactions between politicians and 5 AI-powered villager agents.
"""

import os
import json
import random
from typing import Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import aiohttp
from dotenv import load_dotenv

from agent_configs import AGENT_CONFIGS, POLICY_TOPICS, get_initial_memory

load_dotenv()

app = FastAPI(title="Town Voting Simulator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent_memories: Dict[str, dict] = {}

politician_policies: Dict[str, dict] = {
    "politician_1": {
        "name": "Politician 1",
        "immigration_policy": "",
        "budget_policy": {}
    },
    "politician_2": {
        "name": "Politician 2",
        "immigration_policy": "",
        "budget_policy": {}
    }
}

MODAL_INFERENCE_URL = os.getenv("MODAL_INFERENCE_URL", "")

def initialize_agents():
    """Initialize all agent memories with default values."""
    for agent_key in AGENT_CONFIGS.keys():
        if agent_key not in agent_memories:
            agent_memories[agent_key] = get_initial_memory(agent_key)

initialize_agents()


class TalkRequest(BaseModel):
    agent_name: str
    politician_id: str  # "politician_1" or "politician_2"
    message: str


class BroadcastRequest(BaseModel):
    politician_id: str
    message: str
    topic: str  # "immigration" or "budget"


class PolicyUpdateRequest(BaseModel):
    politician_id: str
    immigration_policy: Optional[str] = None
    budget_policy: Optional[Dict[str, str]] = None


class VoteRequest(BaseModel):
    agent_name: str
    politician_id: str


class TownHallRequest(BaseModel):
    politician_1_message: str
    politician_2_message: str
    topic: str


async def call_llm(system_prompt: str, messages: List[dict]) -> str:
    """
    Call the Modal inference endpoint with the given system prompt and messages.
    Falls back to mock responses if Modal is not configured.
    """
    if not MODAL_INFERENCE_URL:
        return f"[Mock response] I understand your message. As an agent, I have my own views on this matter."
    
    try:
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        
        async with aiohttp.ClientSession() as session:
            payload = {
                "messages": full_messages,
                "model": "Qwen/Qwen3-8B-FP8",
                "stream": False,
                "max_tokens": 500
            }
            
            async with session.post(
                f"{MODAL_INFERENCE_URL}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as resp:
                if resp.status != 200:
                    raise HTTPException(status_code=500, detail=f"Modal inference failed: {resp.status}")
                
                result = await resp.json()
                return result["choices"][0]["message"]["content"]
    
    except Exception as e:
        return f"[Mock response due to error: {str(e)}] I understand your message."



@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Town Voting Simulator API",
        "version": "1.0.0",
        "endpoints": {
            "GET /agents": "List all agents",
            "GET /agent/{agent_name}": "Get agent state",
            "POST /talk": "Talk to a specific agent",
            "POST /broadcast": "Broadcast message to all agents",
            "POST /townhall": "Town hall conversation",
            "GET /politicians": "Get politician policies",
            "POST /politician/policy": "Update politician policy",
            "POST /vote": "Cast a vote",
            "GET /results": "Get voting results"
        }
    }


@app.get("/agents")
async def list_agents():
    """List all available agents."""
    return {
        "agents": [
            {
                "key": key,
                "name": config["name"],
                "full_name": config["full_name"]
            }
            for key, config in AGENT_CONFIGS.items()
        ]
    }


@app.get("/agent/{agent_name}")
async def get_agent_state(agent_name: str):
    """Get the current state and memory of a specific agent."""
    if agent_name not in AGENT_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    
    return {
        "agent": AGENT_CONFIGS[agent_name]["full_name"],
        "memory": agent_memories[agent_name]
    }


@app.post("/talk")
async def talk_to_agent(request: TalkRequest):
    """
    Have a one-on-one conversation with a specific agent.
    The agent will respond based on their personality and current memory.
    """
    if request.agent_name not in AGENT_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Agent '{request.agent_name}' not found")
    
    if request.politician_id not in politician_policies:
        raise HTTPException(status_code=404, detail=f"Politician '{request.politician_id}' not found")
    
    agent_config = AGENT_CONFIGS[request.agent_name]
    agent_memory = agent_memories[request.agent_name]
    
    conversation_messages = []
    for msg in agent_memory["conversation_history"][-10:]:  # Last 10 messages for context
        conversation_messages.append({"role": msg["role"], "content": msg["content"]})
    
    conversation_messages.append({"role": "user", "content": request.message})
    
    response = await call_llm(agent_config["system_prompt"], conversation_messages)
    
    agent_memory["conversation_history"].append({
        "role": "user",
        "content": request.message,
        "politician_id": request.politician_id,
        "timestamp": datetime.now().isoformat()
    })
    agent_memory["conversation_history"].append({
        "role": "assistant",
        "content": response,
        "timestamp": datetime.now().isoformat()
    })
    
    return {
        "agent": agent_config["full_name"],
        "response": response,
        "memory_updated": True
    }


@app.post("/broadcast")
async def broadcast_to_all(request: BroadcastRequest):
    """
    Broadcast a message to all agents. Each agent will update their memory
    based on the message and their personality.
    """
    if request.politician_id not in politician_policies:
        raise HTTPException(status_code=404, detail=f"Politician '{request.politician_id}' not found")
    
    responses = {}
    
    for agent_key, agent_config in AGENT_CONFIGS.items():
        agent_memory = agent_memories[agent_key]
        
        process_prompt = f"""A politician just made this announcement about {request.topic}:

"{request.message}"

How do you feel about this announcement? What are your thoughts? (Respond in character, briefly.)"""
        
        conversation_messages = [{"role": "user", "content": process_prompt}]
        response = await call_llm(agent_config["system_prompt"], conversation_messages)
        
        agent_memory["conversation_history"].append({
            "role": "broadcast",
            "content": request.message,
            "politician_id": request.politician_id,
            "topic": request.topic,
            "timestamp": datetime.now().isoformat()
        })
        agent_memory["conversation_history"].append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })
        
        responses[agent_key] = {
            "agent": agent_config["full_name"],
            "reaction": response
        }
    
    return {
        "broadcast_sent": True,
        "topic": request.topic,
        "responses": responses
    }


@app.post("/townhall")
async def town_hall_conversation(request: TownHallRequest):
    """
    Town hall style conversation where both politicians present on a topic,
    then agents respond in randomized order.
    """
    agent_keys = list(AGENT_CONFIGS.keys())
    random.shuffle(agent_keys)
    
    responses = []
    
    for agent_key in agent_keys:
        agent_config = AGENT_CONFIGS[agent_key]
        agent_memory = agent_memories[agent_key]
        
        town_hall_prompt = f"""This is a town hall meeting about {request.topic}.

Politician 1 says: "{request.politician_1_message}"

Politician 2 says: "{request.politician_2_message}"

What is your response or question to the politicians? (Respond in character, briefly.)"""
        
        conversation_messages = [{"role": "user", "content": town_hall_prompt}]
        response = await call_llm(agent_config["system_prompt"], conversation_messages)
        
        agent_memory["conversation_history"].append({
            "role": "townhall",
            "politician_1_message": request.politician_1_message,
            "politician_2_message": request.politician_2_message,
            "topic": request.topic,
            "timestamp": datetime.now().isoformat()
        })
        agent_memory["conversation_history"].append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })
        
        responses.append({
            "agent": agent_config["full_name"],
            "agent_key": agent_key,
            "response": response
        })
    
    return {
        "topic": request.topic,
        "agent_responses": responses,
        "order": [AGENT_CONFIGS[key]["full_name"] for key in agent_keys]
    }


@app.get("/politicians")
async def get_politicians():
    """Get current policies for both politicians."""
    return politician_policies


@app.post("/politician/policy")
async def update_politician_policy(request: PolicyUpdateRequest):
    """Update a politician's policy positions."""
    if request.politician_id not in politician_policies:
        raise HTTPException(status_code=404, detail=f"Politician '{request.politician_id}' not found")
    
    politician = politician_policies[request.politician_id]
    
    if request.immigration_policy is not None:
        politician["immigration_policy"] = request.immigration_policy
    
    if request.budget_policy is not None:
        politician["budget_policy"] = request.budget_policy
    
    return {
        "politician_id": request.politician_id,
        "updated": True,
        "policies": politician
    }


@app.post("/vote")
async def cast_vote(request: VoteRequest):
    """Cast a vote for a politician from a specific agent."""
    if request.agent_name not in AGENT_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Agent '{request.agent_name}' not found")
    
    if request.politician_id not in politician_policies:
        raise HTTPException(status_code=404, detail=f"Politician '{request.politician_id}' not found")
    
    agent_memory = agent_memories[request.agent_name]
    agent_memory["voting_preference"] = request.politician_id
    
    return {
        "agent": AGENT_CONFIGS[request.agent_name]["full_name"],
        "voted_for": request.politician_id,
        "vote_recorded": True
    }


@app.get("/results")
async def get_voting_results():
    """Get the current voting results."""
    votes = {
        "politician_1": 0,
        "politician_2": 0,
        "undecided": 0
    }
    
    vote_details = []
    
    for agent_key, agent_memory in agent_memories.items():
        vote = agent_memory.get("voting_preference")
        agent_name = AGENT_CONFIGS[agent_key]["full_name"]
        
        if vote == "politician_1":
            votes["politician_1"] += 1
            vote_details.append({"agent": agent_name, "vote": "Politician 1"})
        elif vote == "politician_2":
            votes["politician_2"] += 1
            vote_details.append({"agent": agent_name, "vote": "Politician 2"})
        else:
            votes["undecided"] += 1
            vote_details.append({"agent": agent_name, "vote": "Undecided"})
    
    return {
        "total_agents": len(AGENT_CONFIGS),
        "votes": votes,
        "vote_details": vote_details,
        "winner": "politician_1" if votes["politician_1"] > votes["politician_2"] else 
                 "politician_2" if votes["politician_2"] > votes["politician_1"] else "tie"
    }


@app.post("/reset")
async def reset_simulation():
    """Reset all agent memories and politician policies."""
    initialize_agents()
    
    politician_policies["politician_1"] = {
        "name": "Politician 1",
        "immigration_policy": "",
        "budget_policy": {}
    }
    politician_policies["politician_2"] = {
        "name": "Politician 2",
        "immigration_policy": "",
        "budget_policy": {}
    }
    
    return {"reset": True, "message": "All agent memories and policies have been reset"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
