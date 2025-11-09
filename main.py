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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import aiohttp
from dotenv import load_dotenv

from agent_configs import AGENT_CONFIGS, get_initial_memory
from memory_manager import MemoryDatabase, canonicalize_content

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
            memory["persuasion"] = {
                "politician_1": 0,
                "politician_2": 0
            }
            agent_config = AGENT_CONFIGS[agent_key]
            memory["summary"] = (
                f"Core values: {agent_config['name']} with established personality "
                "and policy preferences."
            )
            memory["llm_decision"] = None
            memory["last_llm_raw_response"] = None
            memory["memory_db"] = MemoryDatabase()
            memory["interaction_count"] = 0
            agent_memories[agent_key] = memory


initialize_agents()


def strip_think_tags(text: str) -> str:
    """Remove <think> tags and their content from model output."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()


async def update_agent_summary(agent_key: str, force: bool = False):
    """
    Update an agent's memory summary using the LLM.
    Creates a brief summary of their current values, concerns, and any shifts.
    Only runs every 3 interactions unless forced.
    """
    agent_config = AGENT_CONFIGS[agent_key]
    agent_memory = agent_memories[agent_key]

    if not force and agent_memory.get("interaction_count", 0) % 3 != 0:
        return

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

    p1_summary = f"Politician 1: Immigration - {p1_policy.get('immigration_policy', 'Not stated')}. "
    if p1_policy.get('budget_policy'):
        budget_items = [f"{k}: {v}" for k, v in list(p1_policy['budget_policy'].items())[:3]]
        p1_summary += f"Budget - {', '.join(budget_items)}"

    p2_summary = f"Politician 2: Immigration - {p2_policy.get('immigration_policy', 'Not stated')}. "
    if p2_policy.get('budget_policy'):
        budget_items = [f"{k}: {v}" for k, v in list(p2_policy['budget_policy'].items())[:3]]
        p2_summary += f"Budget - {', '.join(budget_items)}"

    p1_highlights = []
    p2_highlights = []

    for msg in agent_memory["conversation_history"][-20:]:
        if msg.get("politician_id") == "politician_1" or msg.get("role") == "townhall":
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
With Politician 1:
{p1_highlights_text}

With Politician 2:
{p2_highlights_text}

Based on your personality, values, and the context above, decide who you would vote for. Consider which politician's policies and messages align better with your core values and concerns.

Respond with a JSON object in this exact format:
{{"vote": "politician_1", "rationale": "brief 1-2 sentence explanation", "confidence": "high"}}

The vote field must be exactly one of: politician_1, politician_2, or undecided
The confidence field must be exactly one of: low, medium, or high"""

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


async def call_llm(system_prompt: str, messages: List[dict], temperature: float = 0.7, max_tokens: int = 500, response_format: dict = None) -> str:
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
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 1.0
            }

            if response_format:
                payload["response_format"] = response_format

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
    Uses RAG to retrieve relevant memories.
    """
    if request.agent_name not in AGENT_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Agent '{request.agent_name}' not found")

    if request.politician_id not in politician_policies:
        raise HTTPException(status_code=404, detail=f"Politician '{request.politician_id}' not found")

    agent_config = AGENT_CONFIGS[request.agent_name]
    agent_memory = agent_memories[request.agent_name]
    memory_db = agent_memory["memory_db"]

    relevant_memories = memory_db.get_relevant_memories(request.message, k=8)
    memory_context = memory_db.format_memories_for_prompt(relevant_memories)

    conversation_messages = []

    if memory_context:
        conversation_messages.append({"role": "user", "content": memory_context})

    for msg in agent_memory["conversation_history"][-5:]:
        role = "user" if msg.get("role") in ["user", "townhall", "broadcast"] else "assistant"
        content = msg.get("content", "")
        if msg.get("role") == "townhall":
            content = f"[Town Hall] Politician 1: {msg.get('politician_1_message', '')} | Politician 2: {msg.get('politician_2_message', '')}"
        elif msg.get("role") == "broadcast":
            content = f"[Broadcast] {content}"
        conversation_messages.append({"role": role, "content": content})

    conversation_messages.append({"role": "user", "content": request.message})

    response = await call_llm(agent_config["system_prompt"], conversation_messages)

    persuasion_delta = calculate_persuasion_delta(request.agent_name, request.message, "general")
    agent_memory["persuasion"][request.politician_id] += persuasion_delta
    agent_memory["persuasion"][request.politician_id] = max(-10,
                                                            min(10, agent_memory["persuasion"][request.politician_id]))

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

    memory_db.add_memory(
        content=request.message,
        kind="talk_user",
        politician_id=request.politician_id
    )
    memory_db.add_memory(
        content=response,
        kind="talk_assistant",
        politician_id=request.politician_id
    )

    agent_memory["interaction_count"] = agent_memory.get("interaction_count", 0) + 1
    await update_agent_summary(request.agent_name)

    return {
        "agent": agent_config["full_name"],
        "response": response,
        "persuasion_delta": persuasion_delta,
        "current_persuasion": agent_memory["persuasion"][request.politician_id],
        "memory_updated": True,
        "rag_memories_used": len(relevant_memories)
    }


@app.post("/broadcast")
async def broadcast_to_all(request: BroadcastRequest):
    """
    Broadcast a message to all agents. Each agent will update their memory
    based on the message and their personality.
    Updates persuasion scores based on message content.
    Uses RAG to retrieve relevant memories.
    """
    if request.politician_id not in politician_policies:
        raise HTTPException(status_code=404, detail=f"Politician '{request.politician_id}' not found")

    responses = {}

    for agent_key, agent_config in AGENT_CONFIGS.items():
        agent_memory = agent_memories[agent_key]
        memory_db = agent_memory["memory_db"]

        broadcast_content = canonicalize_content(
            kind="broadcast",
            content=request.message,
            topic=request.topic
        )

        process_prompt = f"""A politician just made this announcement about {request.topic}:

"{request.message}"

How do you feel about this announcement? What are your thoughts? (Respond in character, briefly.)"""

        conversation_messages = [{"role": "user", "content": process_prompt}]
        response = await call_llm(agent_config["system_prompt"], conversation_messages)

        persuasion_delta = calculate_persuasion_delta(agent_key, request.message, request.topic)
        agent_memory["persuasion"][request.politician_id] += persuasion_delta
        agent_memory["persuasion"][request.politician_id] = max(-10,
                                                                min(10, agent_memory["persuasion"][request.politician_id]))

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

        memory_db.add_memory(
            content=broadcast_content,
            kind="broadcast",
            topic=request.topic,
            politician_id=request.politician_id
        )

        agent_memory["interaction_count"] = agent_memory.get("interaction_count", 0) + 1
        await update_agent_summary(agent_key)

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
    Uses RAG to retrieve relevant memories.
    """
    agent_keys = list(AGENT_CONFIGS.keys())
    random.shuffle(agent_keys)

    responses = []

    for agent_key in agent_keys:
        agent_config = AGENT_CONFIGS[agent_key]
        agent_memory = agent_memories[agent_key]
        memory_db = agent_memory["memory_db"]

        townhall_content = canonicalize_content(
            kind="townhall",
            topic=request.topic,
            politician_1_message=request.politician_1_message,
            politician_2_message=request.politician_2_message
        )

        relevant_memories = memory_db.get_relevant_memories(townhall_content, k=8)
        memory_context = memory_db.format_memories_for_prompt(relevant_memories)

        town_hall_prompt = f"""This is a town hall meeting about {request.topic}.

Politician 1 says: "{request.politician_1_message}"

Politician 2 says: "{request.politician_2_message}"

What is your response or question to the politicians? (Respond in character, briefly.)"""

        conversation_messages = []
        if memory_context:
            conversation_messages.append({"role": "user", "content": memory_context})

        conversation_messages.append({"role": "user", "content": town_hall_prompt})
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

        memory_db.add_memory(
            content=townhall_content,
            kind="townhall",
            topic=request.topic
        )

        agent_memory["interaction_count"] = agent_memory.get("interaction_count", 0) + 1
        await update_agent_summary(agent_key)

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

    compatibility_scores = {
        "politician_1": calculate_policy_compatibility(agent_name, "politician_1"),
        "politician_2": calculate_policy_compatibility(agent_name, "politician_2")
    }

    total_scores = {}
    for politician_id in ["politician_1", "politician_2"]:
        persuasion = agent_memory["persuasion"].get(politician_id, 0)
        compatibility = compatibility_scores[politician_id]
        total_scores[politician_id] = persuasion + compatibility

    return {
        "agent": agent_config["full_name"],
        "agent_key": agent_name,
        "llm_decision": agent_memory.get("llm_decision"),
        "last_llm_raw_response": agent_memory.get("last_llm_raw_response"),
        "current_vote": agent_memory.get("voting_preference"),
        "memory_summary": agent_memory.get("summary", "No summary yet"),
        "conversation_count": len(agent_memory["conversation_history"]),
        "legacy_scores": {
            "persuasion": agent_memory["persuasion"],
            "policy_compatibility": compatibility_scores,
            "total_scores": total_scores,
            "note": "These numerical scores are kept for debugging but not used for voting decisions"
        }
    }


@app.get("/debug/rag/{agent_name}")
async def debug_rag_memories(agent_name: str, query: str = "immigration policy"):
    """
    Debug endpoint to inspect RAG memory retrieval for an agent.
    Shows top-k relevant memories for a given query.
    """
    if agent_name not in AGENT_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    agent_memory = agent_memories[agent_name]
    memory_db = agent_memory["memory_db"]

    relevant_memories = memory_db.get_relevant_memories(query, k=8)

    memories_detail = []
    for item, score in relevant_memories:
        memories_detail.append({
            "id": item.id,
            "timestamp": item.timestamp,
            "kind": item.kind,
            "topic": item.topic,
            "politician_id": item.politician_id,
            "content": item.content[:200] + "..." if len(item.content) > 200 else item.content,
            "importance": item.importance,
            "relevance_score": round(score, 4)
        })

    return {
        "agent": AGENT_CONFIGS[agent_name]["full_name"],
        "query": query,
        "total_memories": len(memory_db.items),
        "retrieved_memories": len(relevant_memories),
        "memories": memories_detail,
        "formatted_context": memory_db.format_memories_for_prompt(relevant_memories)
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
