"""
Trajectory management for town hall conversations and voting.
Handles creation and updating of trajectory.json files that track
the full conversation and voting history.
"""

import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


BASE_DIR = Path(__file__).parent
TRAJECTORIES_DIR = BASE_DIR / "memory" / "trajectories"
TRAJECTORIES_DIR.mkdir(parents=True, exist_ok=True)

_last_conversation_id: Optional[str] = None


def generate_conversation_id() -> str:
    """Generate a unique conversation ID for a new town hall."""
    return str(uuid.uuid4())


def get_last_conversation_id() -> Optional[str]:
    """Get the most recent conversation ID (fallback for clients that don't track it)."""
    return _last_conversation_id


def set_last_conversation_id(conversation_id: str):
    """Set the most recent conversation ID."""
    global _last_conversation_id
    _last_conversation_id = conversation_id


def trajectory_path(conversation_id: str) -> Path:
    """Get the file path for a trajectory file."""
    return TRAJECTORIES_DIR / f"trajectory_{conversation_id}.json"


def build_trajectory_payload(
    conversation_id: str,
    topic: str,
    num_rounds: int,
    start_time: datetime,
    duration_seconds: float,
    characters: List[Dict],
    speeches: List[Dict],
    final_state: Dict
) -> Dict:
    """
    Build the complete trajectory payload structure.
    
    Args:
        conversation_id: Unique identifier for this conversation
        topic: The topic of discussion (e.g., "Healthcare Reform", "immigration")
        num_rounds: Number of rounds in the town hall
        start_time: When the town hall started
        duration_seconds: How long the town hall took
        characters: List of character definitions with id, name, role, sprite, initial_position
        speeches: List of speech entries with round, character_id, character_name, message, etc.
        final_state: Final state with character_interests (summaries and vote_intent)
    
    Returns:
        Complete trajectory dictionary ready to be written to JSON
    """
    return {
        "metadata": {
            "conversation_id": conversation_id,
            "topic": topic,
            "num_rounds": num_rounds,
            "timestamp": start_time.isoformat(),
            "duration_seconds": duration_seconds
        },
        "characters": characters,
        "speeches": speeches,
        "final_state": final_state
    }


def write_trajectory(conversation_id: str, payload: Dict):
    """
    Write a trajectory to disk.
    
    Args:
        conversation_id: Unique identifier for this conversation
        payload: Complete trajectory dictionary
    """
    path = trajectory_path(conversation_id)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    
    set_last_conversation_id(conversation_id)
    print(f"Trajectory written to {path}")


def load_trajectory(conversation_id: str) -> Optional[Dict]:
    """
    Load a trajectory from disk.
    
    Args:
        conversation_id: Unique identifier for the conversation
    
    Returns:
        Trajectory dictionary or None if not found
    """
    path = trajectory_path(conversation_id)
    if not path.exists():
        return None
    
    with open(path, 'r') as f:
        return json.load(f)


def update_vote_intent(conversation_id: str, vote_intents: Dict[str, str]):
    """
    Update the vote_intent field for characters in an existing trajectory.
    
    Args:
        conversation_id: Unique identifier for the conversation
        vote_intents: Dictionary mapping character_id to vote_intent (e.g., {"waitress": "politician_1"})
    """
    trajectory = load_trajectory(conversation_id)
    if not trajectory:
        print(f"Warning: Trajectory {conversation_id} not found, cannot update vote intent")
        return
    
    if "final_state" in trajectory and "character_interests" in trajectory["final_state"]:
        for character in trajectory["final_state"]["character_interests"]:
            character_id = character.get("character_id")
            if character_id in vote_intents:
                character["vote_intent"] = vote_intents[character_id]
    
    write_trajectory(conversation_id, trajectory)
    print(f"Updated vote intent for conversation {conversation_id}")


def generate_emoji_summary(message: str) -> str:
    """
    Generate a simple emoji summary for a message.
    Uses basic heuristics rather than LLM to avoid API costs.
    
    Args:
        message: The speech message
    
    Returns:
        Emoji string (e.g., "🏥💪")
    """
    message_lower = message.lower()
    emojis = []
    
    if any(word in message_lower for word in ["healthcare", "health", "medical", "care", "hospital"]):
        emojis.append("🏥")
    
    if any(word in message_lower for word in ["support", "strong", "together", "fight", "commit"]):
        emojis.append("💪")
    
    if any(word in message_lower for word in ["concern", "worry", "fear", "scared", "afraid"]):
        emojis.append("😟")
    
    if any(word in message_lower for word in ["agree", "thank", "appreciate", "grateful"]):
        emojis.append("😊")
    
    if any(word in message_lower for word in ["faith", "prayer", "sacred", "spiritual", "church"]):
        emojis.append("🙏")
    
    if any(word in message_lower for word in ["education", "school", "learn", "library", "study"]):
        emojis.append("📚")
    
    if any(word in message_lower for word in ["family", "children", "kids", "mother", "parent"]):
        emojis.append("👨‍👩‍👧‍👦")
    
    if not emojis:
        emojis = ["🗣️", "💭"]
    
    return "".join(emojis[:2])
