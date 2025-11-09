"""
Debate Town Hall Interaction for veRL multi-turn RL training.
This interaction wraps the FastAPI backend's RL endpoints to enable
multi-turn reinforcement learning for political debate scenarios.
"""

from typing import Dict, Any, List, Tuple, Optional
import asyncio
import httpx
from uuid import uuid4


class DebateTownHallInteraction:
    """
    veRL interaction for debate town hall training.
    
    This class implements the veRL interaction interface to enable multi-turn
    RL training where one politician (the actor) learns to debate effectively
    against a fixed opponent and persuade villager voters.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the debate interaction.
        
        Args:
            config: Configuration dict with keys:
                - base_url: URL of the FastAPI backend (e.g., "http://localhost:8000")
                - topic: Debate topic (e.g., "immigration", "budget")
                - num_rounds: Number of debate rounds (default: 1)
                - max_assistant_turns: Max turns for the actor (default: 6)
                - force_actor: Force specific politician as actor (optional)
                - reward_type: Type of reward calculation (default: "margin_normalized")
                - timeout_s: HTTP request timeout in seconds (default: 30.0)
        """
        self._base_url = config.get("base_url", "http://localhost:8000").rstrip("/")
        self._topic = config.get("topic", "immigration")
        self._num_rounds = int(config.get("num_rounds", 1))
        self._max_assistant_turns = int(config.get("max_assistant_turns", 6))
        self._force_actor = config.get("force_actor")
        self._reward_type = config.get("reward_type", "margin_normalized")
        self._timeout = float(config.get("timeout_s", 30.0))
        
        self._client = httpx.AsyncClient(timeout=self._timeout)
        self._sessions: Dict[str, Dict[str, Any]] = {}
    
    async def start_interaction(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """
        Start a new debate town hall interaction session.
        
        Args:
            instance_id: Optional instance ID (will be generated if not provided)
            **kwargs: Additional keyword arguments (can override config values)
        
        Returns:
            instance_id: Unique identifier for this interaction session
        """
        if instance_id is None:
            instance_id = str(uuid4())
        
        topic = kwargs.get("topic", self._topic)
        num_rounds = kwargs.get("num_rounds", self._num_rounds)
        max_assistant_turns = kwargs.get("max_assistant_turns", self._max_assistant_turns)
        force_actor = kwargs.get("force_actor", self._force_actor)
        reward_type = kwargs.get("reward_type", self._reward_type)
        
        try:
            response = await self._client.post(
                f"{self._base_url}/rl/townhall/start",
                json={
                    "topic": topic,
                    "num_rounds": num_rounds,
                    "max_assistant_turns": max_assistant_turns,
                    "force_actor": force_actor,
                    "reward_type": reward_type
                }
            )
            response.raise_for_status()
            data = response.json()
            
            self._sessions[instance_id] = {
                "session_id": data["session_id"],
                "actor_id": data["actor_id"],
                "non_actor_id": data["non_actor_id"],
                "topic": topic,
                "reward_type": reward_type,
                "total_reward": 0.0,
                "turns": 0
            }
            
            return instance_id
            
        except Exception as e:
            raise RuntimeError(f"Failed to start debate interaction: {str(e)}")
    
    async def generate_response(
        self,
        instance_id: str,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> Tuple[bool, str, float, Dict[str, Any]]:
        """
        Generate the next environment response for the debate interaction.
        
        This method:
        1. Extracts the actor's last message from the messages list
        2. Sends it to the backend's /rl/townhall/step endpoint
        3. Receives the environment's next message (villager utterance)
        4. Returns terminal flag, next message, cumulative reward, and metadata
        
        Args:
            instance_id: Unique identifier for this interaction session
            messages: List of conversation messages (veRL format)
            **kwargs: Additional keyword arguments
        
        Returns:
            Tuple of (terminal, next_message, reward, meta):
                - terminal: Boolean indicating if the episode is complete
                - next_message: The next user message for the actor to respond to
                - reward: Cumulative reward for this episode
                - meta: Metadata dict with session info, votes, etc.
        """
        if instance_id not in self._sessions:
            raise ValueError(f"Instance {instance_id} not found. Call start_interaction first.")
        
        session_info = self._sessions[instance_id]
        backend_session_id = session_info["session_id"]
        
        last_assistant_message = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                last_assistant_message = msg.get("content", "")
                break
        
        if last_assistant_message is None:
            raise ValueError("No assistant message found in messages list")
        
        try:
            response = await self._client.post(
                f"{self._base_url}/rl/townhall/step",
                json={
                    "session_id": backend_session_id,
                    "actor_message": last_assistant_message
                }
            )
            response.raise_for_status()
            data = response.json()
            
            terminal = data["terminal"]
            next_message = data["next_message"]
            step_reward = data["reward"]
            step_meta = data["meta"]
            
            session_info["turns"] += 1
            
            if terminal:
                session_info["total_reward"] = step_reward
                reward = step_reward
            else:
                reward = 0.0
            
            meta = {
                "instance_id": instance_id,
                "backend_session_id": backend_session_id,
                "actor_id": session_info["actor_id"],
                "non_actor_id": session_info["non_actor_id"],
                "topic": session_info["topic"],
                "reward_type": session_info["reward_type"],
                "turns": session_info["turns"],
                "terminal": terminal,
                **step_meta
            }
            
            return terminal, next_message, reward, meta
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate response for debate interaction: {str(e)}")
    
    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            asyncio.get_event_loop().run_until_complete(self.close())
        except:
            pass
