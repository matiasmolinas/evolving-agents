# evolving_agents/core/smart_context.py
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone

@dataclass
class Message:
    """Represents a message within the context history."""
    sender_id: str
    receiver_id: Optional[str] # Can be None for broadcasts or initial prompts
    content: Any
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: int = 1

@dataclass
class ContextEntry:
    """Holds the value and the embedding associated with a semantic key's description."""
    value: Any
    key_embedding: Optional[List[float]] = None # Embedding of the *description* of the key
    semantic_key: str # The human-readable key itself
    source_key: Optional[str] = None # Optional: The original key from global context if different

@dataclass
class SmartContext:
    """
    Refined context payload where data entries include key embeddings.
    Passed optionally via kwargs to agents.
    """
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Info like task_id, source_agent_id, target_agent_id."""

    # Keys are human-readable strings (semantic_key). Values are ContextEntry objects.
    data: Dict[str, ContextEntry] = field(default_factory=dict)
    """Semantically relevant data, including key embeddings."""

    messages: List[Message] = field(default_factory=list)
    """Filtered list of relevant messages."""

    # --- Convenience Methods for Accessing Data ---
    def get_value(self, key: str, default: Optional[Any] = None) -> Any:
        """Gets the value associated with a semantic key."""
        entry = self.data.get(key)
        return entry.value if entry else default

    def get_embedding(self, key: str) -> Optional[List[float]]:
        """Gets the key embedding associated with a semantic key."""
        entry = self.data.get(key)
        return entry.key_embedding if entry else None

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def __len__(self) -> int:
        return len(self.data)

    def items(self):
        """Iterate over (key, value) pairs."""
        return ((k, v.value) for k, v in self.data.items())

    def entries(self):
         """Iterate over (key, ContextEntry) pairs."""
         return self.data.items()