"""
Main FastAPI backend for the Town Voting Simulator.
Handles interactions between politicians and 5 AI-powered villager agents.
"""

import os
import json
import random
import re
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

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

AGENT_POLICY_WEIGHTS = {
    "waitress": {
        "welfare": 3,
        "immigration_open": -3,
        "immigration_restrict": 2,
        "schools": 1,
        "health": 1,
        "police": 0
    },
    "librarian": {
        "schools": 3,
        "health": 2,
        "welfare": 1,
        "immigration_open": 1,
        "police": 0
    },
    "monk": {
        "health": 3,
        "immigration_restrict": 2,
        "immigration_religious": 3,
        "welfare": 1,
        "schools": 0,
        "police": 0
    },
    "police": {
        "police": 3,
        "immigration_restrict": 2,
        "immigration_open": -2,
        "welfare": 0,
        "schools": 0,
        "health": 0
    },
    "stay_at_home_mom": {
        "schools": 3,
        "immigration_open": 2,
        "welfare": 1,
        "health": 1,
        "police": -3
    }
}

def initialize_agents():
    """Initialize all agent memories with default values."""
    for agent_key in AGENT_CONFIGS.keys():
        if agent_key not in agent_memories:
            memory = get_initial_memory(agent_key)
            # Add persuasion tracking for each politician
            memory["persuasion"] = {
                "politician_1": 0,
                "politician_2": 0
            }
            agent_memories[agent_key] = memory

initialize_agents()


def strip_think_tags(text: str) -> str:
    """Remove <think> tags and their content from model output."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()


def calculate_persuasion_delta(agent_key: str, message: str, topic: str) -> int:
    """
    Calculate persuasion delta based on message content and agent preferences.
    Returns a value between -3 and +3.
    """
    message_lower = message.lower()
    weights = AGENT_POLICY_WEIGHTS.get(agent_key, {})
    delta = 0
    
    if "increase" in message_lower or "more" in message_lower or "expand" in message_lower:
        if "welfare" in message_lower:
            delta += weights.get("welfare", 0)
        if "school" in message_lower or "education" in message_lower:
            delta += weights.get("schools", 0)
        if "health" in message_lower or "healthcare" in message_lower:
            delta += weights.get("health", 0)
        if "police" in message_lower or "defense" in message_lower or "security" in message_lower:
            delta += weights.get("police", 0)
    
    if "decrease" in message_lower or "cut" in message_lower or "reduce" in message_lower:
        if "police" in message_lower or "defense" in message_lower:
            delta -= weights.get("police", 0)  # Negative of negative = positive for anti-police agents
    
    if "immigration" in message_lower or "immigrant" in message_lower:
        if "open" in message_lower or "welcome" in message_lower or "diversity" in message_lower:
            delta += weights.get("immigration_open", 0)
        if "restrict" in message_lower or "control" in message_lower or "limit" in message_lower:
            delta += weights.get("immigration_restrict", 0)
        if "religion" in message_lower or "faith" in message_lower:
            delta += weights.get("immigration_religious", 0)
    
    return max(-3, min(3, delta))


def calculate_policy_compatibility(agent_key: str, politician_id: str) -> int:
    """
    Calculate how compatible a politician's stated policies are with an agent's values.
    Returns a score between -10 and +10.
    """
    politician = politician_policies[politician_id]
    weights = AGENT_POLICY_WEIGHTS.get(agent_key, {})
    score = 0
    
    immigration_policy = politician.get("immigration_policy", "").lower()
    if immigration_policy:
        if "open" in immigration_policy or "welcome" in immigration_policy:
            score += weights.get("immigration_open", 0) * 2
        if "restrict" in immigration_policy or "control" in immigration_policy:
            score += weights.get("immigration_restrict", 0) * 2
        if "religion" in immigration_policy or "faith" in immigration_policy:
            score += weights.get("immigration_religious", 0) * 2
    
    budget_policy = politician.get("budget_policy", {})
    for category, stance in budget_policy.items():
        category_lower = category.lower()
        stance_lower = stance.lower()
        
        weight_key = category_lower
        if category_lower in ["police", "defense", "security"]:
            weight_key = "police"
        elif category_lower in ["school", "schools", "education"]:
            weight_key = "schools"
        elif category_lower in ["health", "healthcare"]:
            weight_key = "health"
        
        if "increase" in stance_lower or "more" in stance_lower:
            score += weights.get(weight_key, 0) * 2
        elif "decrease" in stance_lower or "less" in stance_lower:
            score -= weights.get(weight_key, 0) * 2
    
    return max(-10, min(10, score))


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
                response = result["choices"][0]["message"]["content"]
                return strip_think_tags(response)
    
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
    Updates persuasion score based on message content.
    """
    if request.agent_name not in AGENT_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Agent '{request.agent_name}' not found")
    
    if request.politician_id not in politician_policies:
        raise HTTPException(status_code=404, detail=f"Politician '{request.politician_id}' not found")
    
    agent_config = AGENT_CONFIGS[request.agent_name]
    agent_memory = agent_memories[request.agent_name]
    
    conversation_messages = []
    for msg in agent_memory["conversation_history"][-10:]:
        conversation_messages.append({"role": msg["role"], "content": msg["content"]})
    
    conversation_messages.append({"role": "user", "content": request.message})
    
    response = await call_llm(agent_config["system_prompt"], conversation_messages)
    
    persuasion_delta = calculate_persuasion_delta(request.agent_name, request.message, "general")
    agent_memory["persuasion"][request.politician_id] += persuasion_delta
    agent_memory["persuasion"][request.politician_id] = max(-10, min(10, agent_memory["persuasion"][request.politician_id]))
    
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
        "persuasion_delta": persuasion_delta,
        "current_persuasion": agent_memory["persuasion"][request.politician_id],
        "memory_updated": True
    }


@app.post("/broadcast")
async def broadcast_to_all(request: BroadcastRequest):
    """
    Broadcast a message to all agents. Each agent will update their memory
    based on the message and their personality.
    Updates persuasion scores based on message content.
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
        
        persuasion_delta = calculate_persuasion_delta(agent_key, request.message, request.topic)
        agent_memory["persuasion"][request.politician_id] += persuasion_delta
        agent_memory["persuasion"][request.politician_id] = max(-10, min(10, agent_memory["persuasion"][request.politician_id]))
        
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
            "reaction": response,
            "persuasion_delta": persuasion_delta,
            "current_persuasion": agent_memory["persuasion"][request.politician_id]
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
    Updates persuasion scores based on both politicians' messages.
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
        
        delta_1 = calculate_persuasion_delta(agent_key, request.politician_1_message, request.topic)
        delta_2 = calculate_persuasion_delta(agent_key, request.politician_2_message, request.topic)
        
        agent_memory["persuasion"]["politician_1"] += delta_1
        agent_memory["persuasion"]["politician_2"] += delta_2
        agent_memory["persuasion"]["politician_1"] = max(-10, min(10, agent_memory["persuasion"]["politician_1"]))
        agent_memory["persuasion"]["politician_2"] = max(-10, min(10, agent_memory["persuasion"]["politician_2"]))
        
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
            "response": response,
            "persuasion_deltas": {
                "politician_1": delta_1,
                "politician_2": delta_2
            },
            "current_persuasion": {
                "politician_1": agent_memory["persuasion"]["politician_1"],
                "politician_2": agent_memory["persuasion"]["politician_2"]
            }
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
    """Cast a manual vote for a politician from a specific agent."""
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


@app.post("/vote/auto")
async def auto_vote():
    """
    Automatically determine each agent's vote based on:
    1. Persuasion score accumulated from conversations
    2. Policy compatibility with each politician's stated positions
    
    Returns detailed breakdown of voting decisions.
    """
    voting_decisions = []
    
    for agent_key, agent_memory in agent_memories.items():
        agent_name = AGENT_CONFIGS[agent_key]["full_name"]
        
        # Calculate total scores for each politician
        scores = {}
        for politician_id in ["politician_1", "politician_2"]:
            persuasion = agent_memory["persuasion"].get(politician_id, 0)
            compatibility = calculate_policy_compatibility(agent_key, politician_id)
            total_score = persuasion + compatibility
            
            scores[politician_id] = {
                "persuasion": persuasion,
                "policy_compatibility": compatibility,
                "total_score": total_score
            }
        
        if scores["politician_1"]["total_score"] > scores["politician_2"]["total_score"]:
            vote = "politician_1"
        elif scores["politician_2"]["total_score"] > scores["politician_1"]["total_score"]:
            vote = "politician_2"
        else:
            vote = None  # Tie = undecided
        
        agent_memory["voting_preference"] = vote
        
        voting_decisions.append({
            "agent": agent_name,
            "agent_key": agent_key,
            "vote": vote,
            "scores": scores,
            "reasoning": f"Persuasion: P1={scores['politician_1']['persuasion']}, P2={scores['politician_2']['persuasion']}; "
                        f"Policy: P1={scores['politician_1']['policy_compatibility']}, P2={scores['politician_2']['policy_compatibility']}; "
                        f"Total: P1={scores['politician_1']['total_score']}, P2={scores['politician_2']['total_score']}"
        })
    
    # Calculate final results
    votes = {"politician_1": 0, "politician_2": 0, "undecided": 0}
    for decision in voting_decisions:
        if decision["vote"] == "politician_1":
            votes["politician_1"] += 1
        elif decision["vote"] == "politician_2":
            votes["politician_2"] += 1
        else:
            votes["undecided"] += 1
    
    winner = "politician_1" if votes["politician_1"] > votes["politician_2"] else \
             "politician_2" if votes["politician_2"] > votes["politician_1"] else "tie"
    
    return {
        "voting_complete": True,
        "method": "automatic",
        "voting_decisions": voting_decisions,
        "summary": {
            "total_agents": len(AGENT_CONFIGS),
            "votes": votes,
            "winner": winner
        }
    }


@app.get("/debug/agent/{agent_name}")
async def debug_agent(agent_name: str):
    """
    Debug endpoint to view detailed agent state including:
    - Memory summary
    - Persuasion scores
    - Policy compatibility scores
    - Computed vote
    """
    if agent_name not in AGENT_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    
    agent_memory = agent_memories[agent_name]
    agent_config = AGENT_CONFIGS[agent_name]
    
    # Calculate policy compatibility for both politicians
    compatibility_scores = {
        "politician_1": calculate_policy_compatibility(agent_name, "politician_1"),
        "politician_2": calculate_policy_compatibility(agent_name, "politician_2")
    }
    
    # Calculate total scores
    total_scores = {}
    for politician_id in ["politician_1", "politician_2"]:
        persuasion = agent_memory["persuasion"].get(politician_id, 0)
        compatibility = compatibility_scores[politician_id]
        total_scores[politician_id] = persuasion + compatibility
    
    if total_scores["politician_1"] > total_scores["politician_2"]:
        computed_vote = "politician_1"
    elif total_scores["politician_2"] > total_scores["politician_1"]:
        computed_vote = "politician_2"
    else:
        computed_vote = "undecided"
    
    return {
        "agent": agent_config["full_name"],
        "agent_key": agent_name,
        "persuasion_scores": agent_memory["persuasion"],
        "policy_compatibility": compatibility_scores,
        "total_scores": total_scores,
        "computed_vote": computed_vote,
        "current_vote": agent_memory.get("voting_preference"),
        "conversation_count": len(agent_memory["conversation_history"]),
        "memory_summary": {
            "current_stance": agent_memory.get("current_stance"),
            "key_concerns": agent_memory.get("key_concerns", [])
        }
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
