"""
Main FastAPI backend for the Town Voting Simulator.
Handles interactions between politicians and 5 AI-powered villager agents.
"""

import os
import json
import random
import re
import uuid
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

RL_SESSIONS: Dict[str, dict] = {}

politician_policies: Dict[str, dict] = {
    "politician_1": {
        "name": "Alex",
        "immigration_policy": "",
        "budget_policy": {}
    },
    "politician_2": {
        "name": "Anthony",
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
            memory["persuasion"] = {
                "politician_1": 0,
                "politician_2": 0
            }
            agent_config = AGENT_CONFIGS[agent_key]
            memory["summary"] = f"Core values: {agent_config['name']} with established personality and policy preferences."
            memory["llm_decision"] = None
            memory["last_llm_raw_response"] = None
            agent_memories[agent_key] = memory

initialize_agents()


def strip_think_tags(text: str) -> str:
    """Remove <think> tags and their content from model output."""
    # Remove <think>...</think> tags (case insensitive)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Remove <thinking>...</thinking> tags
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Remove any remaining unclosed think tags at the start
    text = re.sub(r'^<think>.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'^<thinking>.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


async def update_agent_summary(agent_key: str):
    """
    Update an agent's memory summary using the LLM.
    Creates a brief summary of their current values, concerns, and any shifts.
    """
    agent_config = AGENT_CONFIGS[agent_key]
    agent_memory = agent_memories[agent_key]
    
    recent_history = agent_memory["conversation_history"][-10:]
    
    if not recent_history:
        agent_memory["summary"] = f"No conversations yet. Core values: {agent_config['name']} with their established personality."
        return
    
    history_text = "\n".join([
        f"- {msg.get('role', 'unknown')}: {msg.get('content', '')[:200]}"
        for msg in recent_history
    ])
    
    summarization_prompt = f"""Based on the recent conversations below, write a brief 2-3 sentence summary of this person's current stance, key concerns, and any shifts in their thinking. Focus on what matters most to them regarding immigration and budget policies.

Recent conversations:
{history_text}

Output ONLY the summary, no other text."""
    
    messages = [{"role": "user", "content": summarization_prompt}]
    summary = await call_llm(agent_config["system_prompt"], messages)
    
    agent_memory["summary"] = summary[:500]  # Cap at 500 chars


async def get_llm_voting_decision(agent_key: str) -> dict:
    """
    Use the LLM to make a voting decision based on agent context and persona.
    Returns a dict with: vote, rationale, confidence
    """
    agent_config = AGENT_CONFIGS[agent_key]
    agent_memory = agent_memories[agent_key]
    
    p1_policy = politician_policies["politician_1"]
    p2_policy = politician_policies["politician_2"]
    
    p1_summary = f"Alex: Immigration - {p1_policy.get('immigration_policy', 'Not stated')}. "
    if p1_policy.get('budget_policy'):
        budget_items = [f"{k}: {v}" for k, v in list(p1_policy['budget_policy'].items())[:3]]
        p1_summary += f"Budget - {', '.join(budget_items)}"
    
    p2_summary = f"Anthony: Immigration - {p2_policy.get('immigration_policy', 'Not stated')}. "
    if p2_policy.get('budget_policy'):
        budget_items = [f"{k}: {v}" for k, v in list(p2_policy['budget_policy'].items())[:3]]
        p2_summary += f"Budget - {', '.join(budget_items)}"
    
    p1_highlights = []
    p2_highlights = []
    
    for msg in agent_memory["conversation_history"][-20:]:
        if msg.get("politician_id") == "politician_1":
            content = msg.get("content", "")[:150]
            if content and len(p1_highlights) < 3:
                p1_highlights.append(f"- {content}")
        elif msg.get("politician_id") == "politician_2":
            content = msg.get("content", "")[:150]
            if content and len(p2_highlights) < 3:
                p2_highlights.append(f"- {content}")
    
    p1_highlights_text = "\n".join(p1_highlights) if p1_highlights else "- No direct interactions"
    p2_highlights_text = "\n".join(p2_highlights) if p2_highlights else "- No direct interactions"
    
    summary_text = agent_memory.get('summary', '').strip()
    if not summary_text:
        summary_text = "No conversations yet. Making decision based on core personality values and politician policies."
    
    voting_prompt = f"""You are deciding who to vote for in an election. Here is the context:

YOUR CURRENT STANCE:
{summary_text}

POLITICIAN POLICIES:
{p1_summary}

{p2_summary}

YOUR RECENT INTERACTIONS:
With Alex:
{p1_highlights_text}

With Anthony:
{p2_highlights_text}

Based on your personality, values, and the context above, decide who you would vote for. Consider which politician's policies and messages align better with your core values and concerns.

Respond with a JSON object in this exact format:
{{"vote": "politician_1", "rationale": "brief 1-2 sentence explanation", "confidence": "high"}}

The vote field must be exactly one of: politician_1, politician_2, or undecided
The confidence field must be exactly one of: low, medium, or high"""
    # print("p1_summary: ", p1_summary)
    # print("p2_summary: ", p2_summary)
    # print("p1_highlights_text: ", p1_highlights_text)
    # print("p2_highlights_text: ", p2_highlights_text)
    # print("summary_text: ", summary_text)
    
    messages = [{"role": "user", "content": voting_prompt}]
    
    try:
        response = await call_llm(
            agent_config["system_prompt"], 
            messages, 
            temperature=0.3, 
            max_tokens=200,
            response_format={"type": "json_object"}
        )
        
        agent_memory["last_llm_raw_response"] = response[:500]
        
        response_clean = response.strip()
        
        try:
            decision = json.loads(response_clean)
        except json.JSONDecodeError:
            json_matches = list(re.finditer(r'\{[^{}]*"vote"[^{}]*\}', response_clean))
            if json_matches:
                last_match = json_matches[-1]
                try:
                    decision = json.loads(last_match.group(0))
                except json.JSONDecodeError:
                    decision = {
                        "vote": "undecided",
                        "rationale": f"Could not parse JSON from: {last_match.group(0)[:100]}",
                        "confidence": "low"
                    }
            else:
                decision = {
                    "vote": "undecided",
                    "rationale": f"No valid JSON found in response",
                    "confidence": "low"
                }
        
        if decision.get("vote") not in ["politician_1", "politician_2", "undecided"]:
            decision["vote"] = "undecided"
        
        if not decision.get("rationale"):
            decision["rationale"] = "No rationale provided"
        
        if decision.get("confidence") not in ["low", "medium", "high"]:
            decision["confidence"] = "medium"
        
        decision["decided_at"] = datetime.now().isoformat()
        
        return decision
        
    except Exception as e:
        return {
            "vote": "undecided",
            "rationale": f"Error during decision: {str(e)}",
            "confidence": "low",
            "decided_at": datetime.now().isoformat()
        }



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

    topic: str
    num_rounds: int = 1


class RLTownHallStartRequest(BaseModel):
    topic: str
    num_rounds: int = 1
    max_assistant_turns: int = 6
    force_actor: Optional[str] = None
    reward_type: str = "margin_normalized"


class RLTownHallStartResponse(BaseModel):
    session_id: str
    actor_id: str
    non_actor_id: str
    next_message: str
    meta: dict


class RLTownHallStepRequest(BaseModel):
    session_id: str
    actor_message: str


class RLTownHallStepResponse(BaseModel):
    terminal: bool
    next_message: str
    reward: float
    meta: dict


async def call_llm(system_prompt: str, messages: List[dict], temperature: float = 0.7, max_tokens: int = 500, response_format: dict = None) -> str:
    """
    Call the Modal inference endpoint with the given system prompt and messages.
    Falls back to mock responses if Modal is not configured.
    """
    print("calling llm")
    if not MODAL_INFERENCE_URL:
        return f"[Mock response] I understand your message. As an agent, I have my own views on this matter."
    
    try:
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        
        # Ensure URL doesn't have trailing slash
        base_url = MODAL_INFERENCE_URL.rstrip('/')
        endpoint = f"{base_url}/v1/chat/completions"
        
        async with aiohttp.ClientSession() as session:
            payload = {
                "messages": full_messages,
                "model": "Qwen/Qwen3-8B-FP8",
                "stream": False,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "enable_thinking": False,
                "top_p": 1.0
            }
            
            if response_format:
                payload["response_format"] = response_format
            
            async with session.post(
                endpoint,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as resp:
                if resp.status != 200:
                    # Try to get error details
                    error_text = await resp.text()
                    error_detail = f"Modal inference failed: {resp.status}"
                    try:
                        error_json = await resp.json()
                        if error_json:
                            error_detail += f" - {error_json}"
                    except:
                        if error_text:
                            error_detail += f" - {error_text[:200]}"
                    
                    print(f"ERROR: {error_detail}")
                    print(f"Endpoint: {endpoint}")
                    print(f"Payload keys: {list(payload.keys())}")
                    
                    # Return mock response instead of raising exception to allow simulation to continue
                    return f"[Mock response due to error: {error_detail}] I understand your message."
                
                result = await resp.json()
                response = result["choices"][0]["message"]["content"]
                return strip_think_tags(response)
    
    except aiohttp.ClientError as e:
        error_msg = f"Network error: {str(e)}"
        print(f"ERROR: {error_msg}")
        return f"[Mock response due to error: {error_msg}] I understand your message."
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"ERROR: {error_msg}")
        return f"[Mock response due to error: {error_msg}] I understand your message."



@app.get("/")
async def root():
    """Root endpoint with API information."""
    modal_status = "configured" if MODAL_INFERENCE_URL else "not configured"
    return {
        "message": "Town Voting Simulator API",
        "version": "1.0.0",
        "modal_inference": modal_status,
        "endpoints": {
            "GET /agents": "List all agents",
            "GET /agent/{agent_name}": "Get agent state",
            "POST /talk": "Talk to a specific agent",
            "POST /broadcast": "Broadcast message to all agents",
            "POST /townhall": "Town hall conversation",
            "GET /politicians": "Get politician policies",
            "POST /politician/policy": "Update politician policy",
            "POST /vote": "Cast a vote",
            "GET /results": "Get voting results",
            "GET /debug/modal": "Test Modal connection"
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
    
    await update_agent_summary(request.agent_name)
    
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
        
        await update_agent_summary(agent_key)
        
        responses[agent_key] = {
            "agent": agent_config["full_name"],
            "reaction": response,
        }
    
    return {
        "broadcast_sent": True,
        "topic": request.topic,
        "responses": responses
    }


@app.post("/townhall")
async def town_hall_conversation(request: TownHallRequest):
    """
    Town hall style conversation where all agents (including politicians) participate
    in a progressive discussion. Each agent sees the full conversation history.
    """
    # Get all agent keys (including politicians)
    all_agent_keys = list(AGENT_CONFIGS.keys())
    
    # Build conversation history that will be shared across all agents
    town_hall_history = []
    all_responses = []
    
    # Round 1: Politicians give opening statements
    politician_1_config = AGENT_CONFIGS["politician_1"]
    politician_2_config = AGENT_CONFIGS["politician_2"]
    
    # Generate or use provided opening statements for politicians
    # print("trying to get opening statements for politicians")
    p1_opening_prompt = f"""This is a town hall meeting about {request.topic}. 
You are giving your opening statement to the voters. What do you want to say? (Respond in character, 2-3 sentences.)"""
    p1_message = await call_llm(politician_1_config["system_prompt"], [{"role": "user", "content": p1_opening_prompt}])

    p2_opening_prompt = f"""This is a town hall meeting about {request.topic}. 
You are giving your opening statement to the voters. What do you want to say? (Respond in character, 2-3 sentences.)"""
    p2_message = await call_llm(politician_2_config["system_prompt"], [{"role": "user", "content": p2_opening_prompt}])

    # Add politician opening statements to town hall history
    town_hall_history.append({
        "speaker": "politician_1",
        "speaker_name": politician_1_config["full_name"],
        "content": p1_message,
        "timestamp": datetime.now().isoformat()
    })
    town_hall_history.append({
        "speaker": "politician_2",
        "speaker_name": politician_2_config["full_name"],
        "content": p2_message,
        "timestamp": datetime.now().isoformat()
    })
    
    # Update all agent memories with politicians' opening statements
    for agent_key in all_agent_keys:
        agent_memory = agent_memories[agent_key]
        # Villagers' memories - record both politicians' opening statements
        agent_memory["conversation_history"].append({
            "role": "townhall",
            "content": p1_message,
            "speaker": "politician_1",
            "speaker_name": politician_1_config["full_name"],
            "politician_id": "politician_1",
            "topic": request.topic,
            "timestamp": datetime.now().isoformat()
        })
        agent_memory["conversation_history"].append({
            "role": "townhall",
            "content": p2_message,
            "speaker": "politician_2",
            "speaker_name": politician_2_config["full_name"],
            "politician_id": "politician_2",
            "topic": request.topic,
            "timestamp": datetime.now().isoformat()
        })
    
    await update_agent_summary("politician_1")
    await update_agent_summary("politician_2")
    
    all_responses.append({
        "agent": politician_1_config["full_name"],
        "agent_key": "politician_1",
        "response": p1_message,
        "round": 1
    })
    all_responses.append({
        "agent": politician_2_config["full_name"],
        "agent_key": "politician_2",
        "response": p2_message,
        "round": 1
    })
    
    # Subsequent rounds: All agents participate in randomized order
    for round_num in range(1, request.num_rounds + 1):
        # Shuffle order for this round
        round_agent_keys = [k for k in all_agent_keys]
        random.shuffle(round_agent_keys)
        
        for agent_key in round_agent_keys:
            agent_config = AGENT_CONFIGS[agent_key]
            agent_memory = agent_memories[agent_key]
            
            # Build conversation context from town hall history
            conversation_context = f"This is a town hall meeting about {request.topic}.\n\n"
            conversation_context += "Here's what has been said so far:\n\n"
            
            for msg in town_hall_history:
                conversation_context += f"{msg['speaker_name']}: {msg['content']}\n\n"
            
            conversation_context += "\nIt's your turn to speak. What do you want to say? (Respond in character, briefly - 1-3 sentences.)"
            
            # Build messages for LLM with full conversation history
            conversation_messages = []
            
            # Add recent conversation history from agent's memory (last 5 messages)
            for msg in agent_memory["conversation_history"][-5:]:
                if msg.get("role") == "user":
                    conversation_messages.append({"role": "user", "content": msg.get("content", "")})
                elif msg.get("role") == "assistant":
                    conversation_messages.append({"role": "assistant", "content": msg.get("content", "")})
            
            # Add the town hall context
            conversation_messages.append({"role": "user", "content": conversation_context})
            
            # Get agent's response
            response = await call_llm(agent_config["system_prompt"], conversation_messages)
            
            # Add to town hall history
            town_hall_history.append({
                "speaker": agent_key,
                "speaker_name": agent_config["full_name"],
                "content": response,
                "timestamp": datetime.now().isoformat()
            })
            
            # Update all agents' memories with this new statement
            for other_agent_key in all_agent_keys:
                other_agent_memory = agent_memories[other_agent_key]
                
                if other_agent_key == agent_key:
                    # This is the speaker's own memory - add as assistant
                    other_agent_memory["conversation_history"].append({
                        "role": "assistant",
                        "content": response,
                        "speaker": agent_key,
                        "speaker_name": agent_config["full_name"],
                        "topic": request.topic,
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    # This is another agent hearing this statement - add as townhall with speaker info
                    other_agent_memory["conversation_history"].append({
                        "role": "townhall",
                        "content": response,
                        "speaker": agent_key,
                        "speaker_name": agent_config["full_name"],
                        "politician_id": agent_key if agent_key in ["politician_1", "politician_2"] else None,
                        "topic": request.topic,
                        "timestamp": datetime.now().isoformat()
                    })
            
            await update_agent_summary(agent_key)
            
            all_responses.append({
                "agent": agent_config["full_name"],
                "agent_key": agent_key,
                "response": response,
                "round": round_num + 1
            })
    
    return {
        "topic": request.topic,
        "agent_responses": all_responses,
        "conversation_history": town_hall_history
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
    Automatically determine each agent's vote using LLM-based decisions.
    Each agent makes a voting decision based on their persona, conversation context,
    and politician policies using the language model.
    
    Returns detailed breakdown of voting decisions with rationale and confidence.
    """
    voting_decisions = []
    
    for agent_key, agent_memory in agent_memories.items():
        agent_name = AGENT_CONFIGS[agent_key]["full_name"]
        
        llm_decision = await get_llm_voting_decision(agent_key)
        
        agent_memory["voting_preference"] = llm_decision["vote"]
        agent_memory["llm_decision"] = llm_decision
        
        voting_decisions.append({
            "agent": agent_name,
            "agent_key": agent_key,
            "vote": llm_decision["vote"],
            "rationale": llm_decision["rationale"],
            "confidence": llm_decision["confidence"],
            "decided_at": llm_decision["decided_at"]
        })
    
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
        "method": "llm_based",
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
    - LLM-based voting decision with rationale
    - Memory summary
    - Legacy numerical scores (for comparison)
    """
    if agent_name not in AGENT_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    
    agent_memory = agent_memories[agent_name]
    agent_config = AGENT_CONFIGS[agent_name]
    
    
    return {
        "agent": agent_config["full_name"],
        "agent_key": agent_name,
        "llm_decision": agent_memory.get("llm_decision"),
        "last_llm_raw_response": agent_memory.get("last_llm_raw_response"),
        "current_vote": agent_memory.get("voting_preference"),
        "memory_summary": agent_memory.get("summary", "No summary yet"),
        "conversation_count": len(agent_memory["conversation_history"]),
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
        if agent_key != "politician_1" and agent_key != "politician_2":
            vote = agent_memory.get("voting_preference")
            agent_name = AGENT_CONFIGS[agent_key]["full_name"]
            
            if vote == "politician_1":
                votes["politician_1"] += 1
                vote_details.append({"agent": agent_name, "vote": "Alex"})
            elif vote == "politician_2":
                votes["politician_2"] += 1  
                vote_details.append({"agent": agent_name, "vote": "Anthony"})
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
        "name": "Alex",
        "immigration_policy": "",
        "budget_policy": {}
    }
    politician_policies["politician_2"] = {
        "name": "Anthony",
        "immigration_policy": "",
        "budget_policy": {}
    }
    
    return {"reset": True, "message": "All agent memories and policies have been reset"}


@app.post("/rl/townhall/start")
async def rl_townhall_start(request: RLTownHallStartRequest):
    """
    Initialize a new RL town hall session for multi-turn training.
    Randomly selects one politician as the actor (veRL policy) and the other as non-actor (normal LLM).
    Returns the initial prompt for the actor's opening statement.
    """
    session_id = str(uuid.uuid4())
    
    if request.force_actor and request.force_actor in ["politician_1", "politician_2"]:
        actor_id = request.force_actor
    else:
        actor_id = random.choice(["politician_1", "politician_2"])
    
    non_actor_id = "politician_2" if actor_id == "politician_1" else "politician_1"
    
    villager_keys = [k for k in AGENT_CONFIGS.keys() if k not in ["politician_1", "politician_2"]]
    speaking_order = villager_keys.copy()
    random.shuffle(speaking_order)
    
    actor_config = AGENT_CONFIGS[actor_id]
    non_actor_config = AGENT_CONFIGS[non_actor_id]
    
    session_state = {
        "session_id": session_id,
        "actor_id": actor_id,
        "non_actor_id": non_actor_id,
        "topic": request.topic,
        "num_rounds": request.num_rounds,
        "max_assistant_turns": request.max_assistant_turns,
        "reward_type": request.reward_type,
        "assistant_turns_used": 0,
        "speaking_order": speaking_order,
        "villager_idx": 0,
        "phase": "opening",
        "town_hall_history": [],
        "created_at": datetime.now().isoformat()
    }
    
    RL_SESSIONS[session_id] = session_state
    
    next_message = f"""You are {actor_config['full_name']}, a politician running for office in a small village. This is a town hall meeting about {request.topic}.

You are giving your opening statement to the voters. What do you want to say? Respond in character as {actor_config['full_name']}, in 2-3 sentences."""
    
    meta = {
        "session_id": session_id,
        "actor_id": actor_id,
        "non_actor_id": non_actor_id,
        "phase": "opening",
        "assistant_turns_used": 0,
        "max_assistant_turns": request.max_assistant_turns
    }
    
    return RLTownHallStartResponse(
        session_id=session_id,
        actor_id=actor_id,
        non_actor_id=non_actor_id,
        next_message=next_message,
        meta=meta
    )


@app.post("/rl/townhall/step")
async def rl_townhall_step(request: RLTownHallStepRequest):
    """
    Process one step of the RL town hall session.
    Receives the actor's message, simulates the environment (non-actor + villagers),
    and returns the next message for the actor along with terminal flag and reward.
    """
    if request.session_id not in RL_SESSIONS:
        raise HTTPException(status_code=404, detail=f"Session '{request.session_id}' not found")
    
    session = RL_SESSIONS[request.session_id]
    actor_id = session["actor_id"]
    non_actor_id = session["non_actor_id"]
    actor_config = AGENT_CONFIGS[actor_id]
    non_actor_config = AGENT_CONFIGS[non_actor_id]
    
    session["assistant_turns_used"] += 1
    
    actor_memory = agent_memories[actor_id]
    actor_memory["conversation_history"].append({
        "role": "assistant",
        "content": request.actor_message,
        "speaker": actor_id,
        "speaker_name": actor_config["full_name"],
        "topic": session["topic"],
        "timestamp": datetime.now().isoformat()
    })
    
    session["town_hall_history"].append({
        "speaker": actor_id,
        "speaker_name": actor_config["full_name"],
        "content": request.actor_message,
        "timestamp": datetime.now().isoformat()
    })
    
    terminal = False
    reward = 0.0
    next_message = ""
    
    if session["phase"] == "opening":
        p1_opening_prompt = f"""This is a town hall meeting about {session['topic']}. 
You are giving your opening statement to the voters. What do you want to say? (Respond in character, 2-3 sentences.)"""
        non_actor_message = await call_llm(non_actor_config["system_prompt"], [{"role": "user", "content": p1_opening_prompt}])
        
        non_actor_memory = agent_memories[non_actor_id]
        non_actor_memory["conversation_history"].append({
            "role": "assistant",
            "content": non_actor_message,
            "speaker": non_actor_id,
            "speaker_name": non_actor_config["full_name"],
            "topic": session["topic"],
            "timestamp": datetime.now().isoformat()
        })
        
        session["town_hall_history"].append({
            "speaker": non_actor_id,
            "speaker_name": non_actor_config["full_name"],
            "content": non_actor_message,
            "timestamp": datetime.now().isoformat()
        })
        
        for agent_key in AGENT_CONFIGS.keys():
            agent_memory = agent_memories[agent_key]
            agent_memory["conversation_history"].append({
                "role": "townhall",
                "content": request.actor_message,
                "speaker": actor_id,
                "speaker_name": actor_config["full_name"],
                "politician_id": actor_id,
                "topic": session["topic"],
                "timestamp": datetime.now().isoformat()
            })
            agent_memory["conversation_history"].append({
                "role": "townhall",
                "content": non_actor_message,
                "speaker": non_actor_id,
                "speaker_name": non_actor_config["full_name"],
                "politician_id": non_actor_id,
                "topic": session["topic"],
                "timestamp": datetime.now().isoformat()
            })
        
        session["phase"] = "villager"
        
        first_villager_key = session["speaking_order"][0]
        first_villager_config = AGENT_CONFIGS[first_villager_key]
        
        conversation_context = f"This is a town hall meeting about {session['topic']}.\n\n"
        conversation_context += "Here's what has been said so far:\n\n"
        for msg in session["town_hall_history"]:
            conversation_context += f"{msg['speaker_name']}: {msg['content']}\n\n"
        conversation_context += "\nIt's your turn to speak. What do you want to say? (Respond in character, briefly - 1-3 sentences.)"
        
        villager_message = await call_llm(first_villager_config["system_prompt"], [{"role": "user", "content": conversation_context}])
        
        session["town_hall_history"].append({
            "speaker": first_villager_key,
            "speaker_name": first_villager_config["full_name"],
            "content": villager_message,
            "timestamp": datetime.now().isoformat()
        })
        
        for agent_key in AGENT_CONFIGS.keys():
            agent_memory = agent_memories[agent_key]
            if agent_key == first_villager_key:
                agent_memory["conversation_history"].append({
                    "role": "assistant",
                    "content": villager_message,
                    "speaker": first_villager_key,
                    "speaker_name": first_villager_config["full_name"],
                    "topic": session["topic"],
                    "timestamp": datetime.now().isoformat()
                })
            else:
                agent_memory["conversation_history"].append({
                    "role": "townhall",
                    "content": villager_message,
                    "speaker": first_villager_key,
                    "speaker_name": first_villager_config["full_name"],
                    "topic": session["topic"],
                    "timestamp": datetime.now().isoformat()
                })
        
        next_message = f"""The villager {first_villager_config['full_name']} just said:

"{villager_message}"

You are {actor_config['full_name']}. How do you respond? (Respond in character, briefly - 1-3 sentences.)"""
        
    elif session["phase"] == "villager":
        for agent_key in AGENT_CONFIGS.keys():
            agent_memory = agent_memories[agent_key]
            if agent_key == actor_id:
                pass
            else:
                agent_memory["conversation_history"].append({
                    "role": "townhall",
                    "content": request.actor_message,
                    "speaker": actor_id,
                    "speaker_name": actor_config["full_name"],
                    "politician_id": actor_id,
                    "topic": session["topic"],
                    "timestamp": datetime.now().isoformat()
                })
        
        conversation_context_non_actor = f"This is a town hall meeting about {session['topic']}.\n\n"
        conversation_context_non_actor += "Here's what has been said so far:\n\n"
        for msg in session["town_hall_history"][-5:]:
            conversation_context_non_actor += f"{msg['speaker_name']}: {msg['content']}\n\n"
        conversation_context_non_actor += f"\nIt's your turn to speak. What do you want to say in response? (Respond in character as {non_actor_config['full_name']}, briefly - 1-2 sentences.)"
        
        non_actor_response = await call_llm(non_actor_config["system_prompt"], [{"role": "user", "content": conversation_context_non_actor}])
        
        non_actor_memory = agent_memories[non_actor_id]
        non_actor_memory["conversation_history"].append({
            "role": "assistant",
            "content": non_actor_response,
            "speaker": non_actor_id,
            "speaker_name": non_actor_config["full_name"],
            "topic": session["topic"],
            "timestamp": datetime.now().isoformat()
        })
        
        session["town_hall_history"].append({
            "speaker": non_actor_id,
            "speaker_name": non_actor_config["full_name"],
            "content": non_actor_response,
            "timestamp": datetime.now().isoformat()
        })
        
        for agent_key in AGENT_CONFIGS.keys():
            if agent_key != non_actor_id:
                agent_memory = agent_memories[agent_key]
                agent_memory["conversation_history"].append({
                    "role": "townhall",
                    "content": non_actor_response,
                    "speaker": non_actor_id,
                    "speaker_name": non_actor_config["full_name"],
                    "politician_id": non_actor_id,
                    "topic": session["topic"],
                    "timestamp": datetime.now().isoformat()
                })
        
        session["villager_idx"] += 1
        
        if session["villager_idx"] >= len(session["speaking_order"]) or session["assistant_turns_used"] >= session["max_assistant_turns"]:
            terminal = True
            
            voting_decisions = []
            villager_keys = [k for k in AGENT_CONFIGS.keys() if k not in ["politician_1", "politician_2"]]
            
            for agent_key in villager_keys:
                llm_decision = await get_llm_voting_decision(agent_key)
                agent_memories[agent_key]["voting_preference"] = llm_decision["vote"]
                agent_memories[agent_key]["llm_decision"] = llm_decision
                voting_decisions.append({
                    "agent_key": agent_key,
                    "vote": llm_decision["vote"],
                    "rationale": llm_decision["rationale"],
                    "confidence": llm_decision["confidence"]
                })
            
            votes = {"politician_1": 0, "politician_2": 0, "undecided": 0}
            for decision in voting_decisions:
                if decision["vote"] == "politician_1":
                    votes["politician_1"] += 1
                elif decision["vote"] == "politician_2":
                    votes["politician_2"] += 1
                else:
                    votes["undecided"] += 1
            
            if session["reward_type"] == "margin_normalized":
                reward = (votes[actor_id] - votes[non_actor_id]) / 5.0
            elif session["reward_type"] == "margin":
                reward = votes[actor_id] - votes[non_actor_id]
            elif session["reward_type"] == "votes_normalized":
                reward = votes[actor_id] / 5.0
            else:
                reward = votes[actor_id]
            
            next_message = ""
            
            meta = {
                "session_id": request.session_id,
                "actor_id": actor_id,
                "non_actor_id": non_actor_id,
                "phase": "done",
                "assistant_turns_used": session["assistant_turns_used"],
                "max_assistant_turns": session["max_assistant_turns"],
                "votes": votes,
                "voting_decisions": voting_decisions,
                "reward_type": session["reward_type"],
                "terminal_reason": "max_turns" if session["assistant_turns_used"] >= session["max_assistant_turns"] else "all_villagers_spoke"
            }
        else:
            next_villager_key = session["speaking_order"][session["villager_idx"]]
            next_villager_config = AGENT_CONFIGS[next_villager_key]
            
            conversation_context = f"This is a town hall meeting about {session['topic']}.\n\n"
            conversation_context += "Here's what has been said so far:\n\n"
            for msg in session["town_hall_history"][-8:]:
                conversation_context += f"{msg['speaker_name']}: {msg['content']}\n\n"
            conversation_context += "\nIt's your turn to speak. What do you want to say? (Respond in character, briefly - 1-3 sentences.)"
            
            villager_message = await call_llm(next_villager_config["system_prompt"], [{"role": "user", "content": conversation_context}])
            
            session["town_hall_history"].append({
                "speaker": next_villager_key,
                "speaker_name": next_villager_config["full_name"],
                "content": villager_message,
                "timestamp": datetime.now().isoformat()
            })
            
            for agent_key in AGENT_CONFIGS.keys():
                agent_memory = agent_memories[agent_key]
                if agent_key == next_villager_key:
                    agent_memory["conversation_history"].append({
                        "role": "assistant",
                        "content": villager_message,
                        "speaker": next_villager_key,
                        "speaker_name": next_villager_config["full_name"],
                        "topic": session["topic"],
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    agent_memory["conversation_history"].append({
                        "role": "townhall",
                        "content": villager_message,
                        "speaker": next_villager_key,
                        "speaker_name": next_villager_config["full_name"],
                        "topic": session["topic"],
                        "timestamp": datetime.now().isoformat()
                    })
            
            next_message = f"""The villager {next_villager_config['full_name']} just said:

"{villager_message}"

You are {actor_config['full_name']}. How do you respond? (Respond in character, briefly - 1-3 sentences.)"""
            
            meta = {
                "session_id": request.session_id,
                "actor_id": actor_id,
                "non_actor_id": non_actor_id,
                "phase": "villager",
                "villager_idx": session["villager_idx"],
                "assistant_turns_used": session["assistant_turns_used"],
                "max_assistant_turns": session["max_assistant_turns"]
            }
    
    return RLTownHallStepResponse(
        terminal=terminal,
        next_message=next_message,
        reward=reward,
        meta=meta
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
