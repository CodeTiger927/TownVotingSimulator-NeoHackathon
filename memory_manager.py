"""
Memory management system with RAG-based retrieval for AI agents.
Implements embeddings, importance scoring, and relevancy-based memory retrieval.
"""

import uuid
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer


class EmbeddingsManager:
    """Manages text embeddings using sentence-transformers."""

    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            print("Loading embedding model (all-MiniLM-L6-v2)...")
            self._model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Embedding model loaded successfully")
        return self._model

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a text string."""
        model = self.get_model()
        return model.encode(text, convert_to_numpy=True)


class MemoryItem:
    """Represents a single piece of information in agent memory."""

    def __init__(
        self,
        content: str,
        kind: str,
        timestamp: Optional[str] = None,
        topic: Optional[str] = None,
        politician_id: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
        importance: Optional[float] = None
    ):
        self.id = str(uuid.uuid4())
        self.timestamp = timestamp or datetime.now().isoformat()
        self.kind = kind
        self.topic = topic
        self.politician_id = politician_id
        self.content = content
        self.embedding = embedding
        self.importance = importance or self._calculate_importance()

    def _calculate_importance(self) -> float:
        """Calculate importance score based on kind and content."""
        base_scores = {
            "townhall": 0.8,
            "broadcast": 0.6,
            "talk_user": 0.5,
            "talk_assistant": 0.4
        }

        score = base_scores.get(self.kind, 0.5)

        policy_keywords = [
            "immigration", "immigr", "border",
            "school", "education",
            "welfare", "social",
            "health", "healthcare",
            "police", "security", "defense"
        ]

        content_lower = self.content.lower()
        if any(keyword in content_lower for keyword in policy_keywords):
            score += 0.1

        return min(1.0, score)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "kind": self.kind,
            "topic": self.topic,
            "politician_id": self.politician_id,
            "content": self.content,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "importance": self.importance
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryItem':
        """Create MemoryItem from dictionary."""
        item = cls(
            content=data["content"],
            kind=data["kind"],
            timestamp=data["timestamp"],
            topic=data.get("topic"),
            politician_id=data.get("politician_id"),
            importance=data.get("importance")
        )
        item.id = data["id"]
        if data.get("embedding"):
            item.embedding = np.array(data["embedding"])
        return item


class MemoryDatabase:
    """Manages a collection of memory items with RAG retrieval."""

    def __init__(self):
        self.items: List[MemoryItem] = []
        self.embeddings_manager = EmbeddingsManager()

    def add_memory(
        self,
        content: str,
        kind: str,
        topic: Optional[str] = None,
        politician_id: Optional[str] = None
    ) -> MemoryItem:
        """Add a new memory item with automatic embedding generation."""
        embedding = self.embeddings_manager.embed(content)

        item = MemoryItem(
            content=content,
            kind=kind,
            topic=topic,
            politician_id=politician_id,
            embedding=embedding
        )

        self.items.append(item)
        return item

    def get_relevant_memories(
        self,
        query: str,
        k: int = 8,
        w_similarity: float = 0.4,
        w_importance: float = 0.3,
        w_recency: float = 0.3,
        recency_tau_hours: float = 24.0
    ) -> List[Tuple[MemoryItem, float]]:
        """
        Retrieve top-k most relevant memories using RAG.

        Args:
            query: Query text to find relevant memories
            k: Number of memories to retrieve
            w_similarity: Weight for similarity score
            w_importance: Weight for importance score
            w_recency: Weight for recency score
            recency_tau_hours: Time decay constant in hours

        Returns:
            List of (MemoryItem, score) tuples sorted by relevance
        """
        if not self.items:
            return []

        query_embedding = self.embeddings_manager.embed(query)

        scored_items = []
        now = datetime.now()

        for item in self.items:
            if item.embedding is None:
                continue

            cosine_sim = np.dot(query_embedding, item.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(item.embedding)
            )
            similarity = (cosine_sim + 1) / 2  # Map from [-1, 1] to [0, 1]

            item_time = datetime.fromisoformat(item.timestamp)
            time_diff_hours = (now - item_time).total_seconds() / 3600
            recency = np.exp(-time_diff_hours / recency_tau_hours)

            score = (
                w_similarity * similarity +
                w_importance * item.importance +
                w_recency * recency
            )

            scored_items.append((item, score))

        scored_items.sort(key=lambda x: x[1], reverse=True)
        return scored_items[:k]

    def format_memories_for_prompt(
        self,
        memories: List[Tuple[MemoryItem, float]]
    ) -> str:
        """Format retrieved memories for inclusion in LLM prompt."""
        if not memories:
            return ""

        lines = ["Relevant memory (most similar/recent/important):"]
        for item, score in memories:
            dt = datetime.fromisoformat(item.timestamp)
            date_str = dt.strftime("%Y-%m-%d %H:%M")

            context = f"[{item.kind}"
            if item.topic:
                context += f":{item.topic}"
            context += "]"

            content = item.content
            if len(content) > 200:
                content = content[:197] + "..."

            lines.append(f"- {date_str} {context} {content}")

        return "\n".join(lines)

    def to_dict(self) -> List[Dict]:
        """Convert all memories to dictionary for serialization."""
        return [item.to_dict() for item in self.items]

    def from_dict(self, data: List[Dict]):
        """Load memories from dictionary."""
        self.items = [MemoryItem.from_dict(item_data) for item_data in data]


def canonicalize_content(
    kind: str,
    content: str = "",
    topic: str = "",
    politician_1_message: str = "",
    politician_2_message: str = ""
) -> str:
    """
    Canonicalize content for consistent embedding generation.

    For townhalls, combines both politicians' messages.
    For broadcasts, includes topic tag.
    For talks, uses content as-is.
    """
    if kind == "townhall":
        return f"[townhall:{topic}] Politician 1: {politician_1_message} | Politician 2: {politician_2_message}"
    elif kind == "broadcast":
        return f"[broadcast:{topic}] {content}"
    else:
        return content
