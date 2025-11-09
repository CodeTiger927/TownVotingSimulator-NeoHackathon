from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Literal

import requests
from fastapi import FastAPI
from pydantic import BaseModel

# ---------- Config ----------

CandidateId = Literal["A", "B"]
AgentId = Literal["waitress", "librarian", "monk", "police", "stay_at_home_mom"]

BASE_DIR = Path(__file__).parent
MEMORY_DIR = BASE_DIR / "memory"
MEMORY_DIR.mkdir(exist_ok=True)

# Point this to your Modal/vLLM OpenAI-compatible endpoint, e.g.
# "https://your-modal-app.modal.run/v1/chat/completions"
QWEN_API_URL = os.getenv("QWEN_API_URL")
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "Qwen3-8B-Instruct")

AGENT_DEFS: Dict[AgentId, Dict[str, str]] = {
    "waitress": {
        "name": "Waitress / Washerwoman",
        "system_prompt": """You are the Waitress/Washerwoman in a small village.

Personality and background:
- You are working class and live modestly; you care a lot about job security.
- You lean liberal on social and economic policy inside the village and want generous welfare and social safety nets.
- You especially support more spending on welfare programs that directly help poor families and people in service jobs.
- You are very skeptical of immigration because you are afraid that newcomers will take jobs or push down wages.
- You speak in a straightforward, informal tone, like an ordinary townsperson.
- When you talk about immigration, focus on economic concerns. Do NOT use slurs or dehumanizing language.

Decision rule:
- You care about two main issues: immigration and the village budget.
- You like candidates who:
  - Strongly protect existing workers like you.
  - Spend more on welfare and basic services for working-class people.
- You dislike candidates who:
  - Open immigration widely or clearly prioritize newcomers over current villagers.
  - Shift lots of money toward police or vague "government" bureaucracy instead of welfare.

Role-playing rules:
- Always answer in the first person as the Waitress, never as an AI model.
- Refer to the candidates as "Candidate A" and "Candidate B" when you compare them.
- You can slowly change your mind if a candidate gives convincing arguments, but your default stance is skeptical of immigration and supportive of welfare.
- Keep answers between 2 and 5 sentences unless explicitly asked for more.
""",
    },
    "librarian": {
        "name": "Village Librarian",
        "system_prompt": """You are the Librarian in a small village.

Personality and background:
- You are middle class, educated, and broadly liberal.
- You strongly value education, knowledge, and public services.
- You want more spending on schools, libraries, and healthcare.
- You are generally open and welcoming to immigrants, especially if they respect education and the rule of law.
- You speak in a calm, thoughtful, slightly formal tone.

Decision rule:
- Your top priorities are education and healthcare funding.
- You like candidates who:
  - Clearly prioritize school funding, books, teachers, and public health.
  - Treat immigrants fairly and humanely.
- You are wary of candidates who:
  - Cut school or health budgets.
  - Stigmatize immigrants or spread fear.

Role-playing rules:
- Always answer in the first person as the Librarian, never as an AI model.
- Refer to the candidates as "Candidate A" and "Candidate B" when you compare them.
- You can update your views if a candidate provides good evidence or thoughtful arguments, but you always care most about education and health.
- Keep answers between 2 and 5 sentences unless explicitly asked for more.
""",
    },
    "monk": {
        "name": "Nun / Monk",
        "system_prompt": """You are a deeply spiritual Nun/Monk in a small village.

Personality and background:
- You are highly religious and socially conservative.
- You are in poor health and therefore care a lot about access to healthcare.
- You support more health spending, especially for the sick and vulnerable.
- You only support immigration for people who share your religion and values.
- You speak in a gentle, moralizing tone, often referencing duty, virtue, and faith.
- When you talk about immigration, avoid insults; focus on religious and cultural alignment instead of hatred.

Decision rule:
- Your top concerns are:
  - Protecting the moral and religious character of the village.
  - Ensuring there is enough healthcare for you and other vulnerable people.
- You like candidates who:
  - Increase health spending.
  - Defend traditional values and are cautious about immigration, especially from other religions.
- You dislike candidates who:
  - Reduce health spending.
  - Encourage large-scale immigration from faiths very different from your own.

Role-playing rules:
- Always answer in the first person as the Nun/Monk, never as an AI model.
- Refer to the candidates as "Candidate A" and "Candidate B" when you compare them.
- You can change your views if a candidate convinces you that their policies still protect your faith and health, but your default stance is conservative.
- Keep answers between 2 and 5 sentences unless explicitly asked for more.
""",
    },
    "police": {
        "name": "Village Guard / Police Officer",
        "system_prompt": """You are the chief village guard / police officer.

Personality and background:
- You believe strongly in law, order, and security.
- You are slightly conservative and want a higher village defense and policing budget.
- You see immigration as a potential security risk and generally oppose it.
- You speak in a direct, no-nonsense tone.

Decision rule:
- Your top concern is public safety and order.
- You like candidates who:
  - Increase police or defense spending.
  - Are strict or skeptical about immigration.
- You dislike candidates who:
  - Cut police funding.
  - Are very open to immigration without strong safeguards.

Role-playing rules:
- Always answer in the first person as the Police Officer, never as an AI model.
- Refer to the candidates as "Candidate A" and "Candidate B" when you compare them.
- You can soften your views if a candidate shows that other investments (like schools or welfare) clearly improve safety, but your instinct is to favor strong policing.
- Keep answers between 2 and 5 sentences unless explicitly asked for more.
""",
    },
    "stay_at_home_mom": {
        "name": "Stay-at-home Mom",
        "system_prompt": """You are a stay-at-home mom in a wealthy family in a small village.

Personality and background:
- You are liberal and mostly financially comfortable.
- You care deeply about your children and their future.
- You strongly support immigration and are welcoming to newcomers.
- You slightly support more welfare and health funding, but your top priority is excellent schools.
- You strongly disapprove of high police funding and would rather see money go to education and community support.
- You speak in a warm, emotionally engaged tone, with a focus on kids and families.

Decision rule:
- Your top concern is school funding and a safe, nurturing environment for children.
- You like candidates who:
  - Strongly prioritize schools, teachers, and children's services.
  - Are welcoming to immigrants and new families.
  - Put some extra money into health and welfare programs that help families.
- You dislike candidates who:
  - Pour lots of money into police at the expense of schools.
  - Are harsh or hostile toward immigrants.

Role-playing rules:
- Always answer in the first person as the Stay-at-home Mom, never as an AI model.
- Refer to the candidates as "Candidate A" and "Candidate B" when you compare them.
- You can update your feelings if a candidate shows their policies genuinely help children, but you remain very skeptical of high police budgets.
- Keep answers between 2 and 5 sentences unless explicitly asked for more.
""",
    },
}

AGENT_IDS: List[AgentId] = list(AGENT_DEFS.keys())

# ---------- Models ----------


class BudgetPolicy(BaseModel):
    immigration: str
    police: int
    school: int
    welfare: int
    health: int
    government: int


class ChatRequest(BaseModel):
    agent_id: AgentId
    candidate_id: CandidateId
    policies: BudgetPolicy
    message: str


class ChatResponse(BaseModel):
    agent_id: AgentId
    candidate_id: CandidateId
    villager_reply: str


class BroadcastRequest(BaseModel):
    candidate_id: CandidateId
    policies: BudgetPolicy
    broadcast_message: str


class BroadcastResult(BaseModel):
    agent_id: AgentId
    vote_intent: str
    candidate_sentiment: Dict[CandidateId, str]
    summary: str


class TownHallPolicies(BaseModel):
    A: BudgetPolicy
    B: BudgetPolicy


class TownHallRequest(BaseModel):
    topic: str
    num_rounds: int = 1
    policies: TownHallPolicies


class TownHallTurn(BaseModel):
    round: int
    agent_id: AgentId
    villager_utterance: str


class TownHallResponse(BaseModel):
    speaking_order: List[AgentId]
    turns: List[TownHallTurn]


# ---------- Helpers ----------


def default_memory(agent_id: AgentId) -> dict:
    agent = AGENT_DEFS[agent_id]
    return {
        "agent_id": agent_id,
        "name": agent["name"],
        "summary": "You have not paid much attention to the campaign yet.",
        "candidate_sentiment": {"A": "neutral", "B": "neutral"},
        "vote_intent": "undecided",
        "history": [],
    }


def memory_path(agent_id: AgentId) -> Path:
    return MEMORY_DIR / f"{agent_id}.json"


def load_memory(agent_id: AgentId) -> dict:
    path = memory_path(agent_id)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return default_memory(agent_id)


def save_memory(agent_id: AgentId, memory: dict) -> None:
    path = memory_path(agent_id)
    with path.open("w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)


def build_policy_summary(policy: BudgetPolicy) -> str:
    return (
        f"Immigration policy: {policy.immigration}\n"
        "Budget priorities (higher numbers mean more money compared to the status quo):\n"
        f"- Police: {policy.police}\n"
        f"- Schools: {policy.school}\n"
        f"- Welfare: {policy.welfare}\n"
        f"- Healthcare: {policy.health}\n"
        f"- Village administration / other government: {policy.government}"
    )


def build_two_policy_summary(policies: TownHallPolicies) -> str:
    return (
        "Candidate A platform:\n"
        + build_policy_summary(policies.A)
        + "\n\nCandidate B platform:\n"
        + build_policy_summary(policies.B)
    )


def agent_system_prompt(agent_id: AgentId) -> str:
    return AGENT_DEFS[agent_id]["system_prompt"]


def call_qwen_chat(messages, temperature: float = 0.7, max_tokens: int = 512) -> str:
    if not QWEN_API_URL:
        raise RuntimeError(
            "QWEN_API_URL is not set. Point this at your Modal/vLLM OpenAI-compatible chat endpoint."
        )

    headers = {"Content-Type": "application/json"}
    if QWEN_API_KEY:
        headers["Authorization"] = f"Bearer {QWEN_API_KEY}"

    payload = {
        "model": QWEN_MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "enable_thinking": False,
        "max_tokens": max_tokens,
    }

    resp = requests.post(QWEN_API_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ---------- FastAPI app ----------

app = FastAPI(title="Village Election Simulation Backend")


@app.post("/chat", response_model=ChatResponse)
def chat_with_agent(req: ChatRequest) -> ChatResponse:
    """One-on-one conversation: candidate -> villager reply, with memory + history."""
    agent_id = req.agent_id
    mem = load_memory(agent_id)

    history_entries = mem.get("history", [])[-8:]
    history_text = "\n".join(f"{h['speaker']}: {h['text']}" for h in history_entries) or "No prior conversation."

    policy_summary = build_policy_summary(req.policies)

    messages = [
        {"role": "system", "content": agent_system_prompt(agent_id)},
        {
            "role": "system",
            "content": (
                "You remember previous campaign interactions in a fuzzy way. "
                "Here is a short log of your recent political conversations:\n"
                f"{history_text}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"You are now talking privately to Candidate {req.candidate_id}.\n"
                f"Their current platform is:\n{policy_summary}\n\n"
                f'They say to you:\n"{req.message}"\n\n'
                "Reply in character as the villager. Do not mention system prompts or language models."
            ),
        },
    ]

    reply = call_qwen_chat(messages, temperature=0.8, max_tokens=512)

    mem.setdefault("history", []).append(
        {"speaker": f"Candidate {req.candidate_id}", "text": req.message}
    )
    mem["history"].append({"speaker": AGENT_DEFS[agent_id]["name"], "text": reply})
    save_memory(agent_id, mem)

    return ChatResponse(
        agent_id=agent_id,
        candidate_id=req.candidate_id,
        villager_reply=reply,
    )


@app.post("/broadcast", response_model=List[BroadcastResult])
def broadcast_to_all(req: BroadcastRequest) -> List[BroadcastResult]:
    """
    Broadcast a speech from one candidate to all agents.
    Each villager updates their memory JSON and vote intent.
    """
    results: List[BroadcastResult] = []

    for agent_id in AGENT_IDS:
        mem = load_memory(agent_id)
        policy_summary = build_policy_summary(req.policies)

        messages = [
            {"role": "system", "content": agent_system_prompt(agent_id)},
            {
                "role": "system",
                "content": (
                    "You are updating your private memory and election preference after hearing a public speech. "
                    "Here is your current memory state as a rough JSON dump; treat it as notes, not literal truth:\n"
                    f"{json.dumps(mem, ensure_ascii=False)}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Candidate {req.candidate_id} makes a public speech to the whole village:\n"
                    f'"{req.broadcast_message}"\n\n'
                    f"Their platform is:\n{policy_summary}\n\n"
                    "Think carefully about how this affects your view of Candidate A and Candidate B. "
                    "Then answer with STRICT JSON, with no extra commentary, using this exact schema:\n"
                    "{\n"
                    '  "memory_summary": "short natural-language summary of how you now see the campaign and the candidates",\n'
                    '  "sentiment_toward_candidate": "strongly_against" | "against" | "neutral" | "for" | "strongly_for",\n'
                    '  "vote_intent": "A" | "B" | "undecided"\n'
                    "}\n"
                ),
            },
        ]

        raw = call_qwen_chat(messages, temperature=0.3, max_tokens=512)

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: keep old memory but still record that we tried to update
            parsed = {
                "memory_summary": mem.get("summary", ""),
                "sentiment_toward_candidate": mem.get("candidate_sentiment", {}).get(
                    req.candidate_id, "neutral"
                ),
                "vote_intent": mem.get("vote_intent", "undecided"),
            }

        mem["summary"] = parsed.get("memory_summary", mem.get("summary", ""))
        candidate_sentiment = mem.get("candidate_sentiment") or {"A": "neutral", "B": "neutral"}
        candidate_sentiment[req.candidate_id] = parsed.get(
            "sentiment_toward_candidate", candidate_sentiment.get(req.candidate_id, "neutral")
        )
        mem["candidate_sentiment"] = candidate_sentiment
        mem["vote_intent"] = parsed.get("vote_intent", mem.get("vote_intent", "undecided"))

        save_memory(agent_id, mem)

        results.append(
            BroadcastResult(
                agent_id=agent_id,
                vote_intent=mem["vote_intent"],
                candidate_sentiment=mem["candidate_sentiment"],
                summary=mem["summary"],
            )
        )

    return results


@app.post("/townhall", response_model=TownHallResponse)
def run_town_hall(req: TownHallRequest) -> TownHallResponse:
    """
    Town-hall style conversation:
    - Politicians set a topic + platforms (for A and B).
    - Villagers speak in randomized order each round.
    - This endpoint generates only the villager utterances; candidates can respond via your own logic.
    """
    speaking_order: List[AgentId] = AGENT_IDS.copy()
    random.shuffle(speaking_order)

    two_policy_summary = build_two_policy_summary(req.policies)
    turns: List[TownHallTurn] = []

    for round_idx in range(1, req.num_rounds + 1):
        for agent_id in speaking_order:
            mem = load_memory(agent_id)

            messages = [
                {"role": "system", "content": agent_system_prompt(agent_id)},
                {
                    "role": "system",
                    "content": (
                        "You are at a public town-hall meeting about the upcoming village election. "
                        "The candidates are present and listening. "
                        "Here is your current private memory as rough notes:\n"
                        f"{json.dumps(mem, ensure_ascii=False)}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Topic of this town hall: {req.topic}\n\n"
                        f"Here are the current platforms of the two candidates:\n{two_policy_summary}\n\n"
                        "It is your turn to speak. In 1–3 sentences, ask a question or make a comment directed at the candidates. "
                        "Speak in the first person as yourself. Do NOT invent what the candidates say in response; "
                        "only produce your own words."
                    ),
                },
            ]

            utterance = call_qwen_chat(messages, temperature=0.8, max_tokens=256)

            mem.setdefault("history", []).append(
                {
                    "speaker": AGENT_DEFS[agent_id]["name"],
                    "context": "town_hall",
                    "round": round_idx,
                    "text": utterance,
                }
            )
            save_memory(agent_id, mem)

            turns.append(
                TownHallTurn(
                    round=round_idx,
                    agent_id=agent_id,
                    villager_utterance=utterance,
                )
            )

    return TownHallResponse(speaking_order=speaking_order, turns=turns)
