import time
from enum import Enum
from typing import Any, Dict, Optional


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message:
    """Standardized representation of a conversational message across all providers."""

    def __init__(
        self,
        role: MessageRole | str,
        content: str,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        images: Optional[List[str]] = None,
    ):
        if isinstance(role, str):
            try:
                self.role = MessageRole(role.lower())
            except ValueError:
                self.role = MessageRole.USER
        else:
            self.role = role

        self.content = str(content) if content is not None else ""
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.metadata = metadata or {}
        self.images = list(images) if images else []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to standard dictionary payload."""
        data = {
            "role": self.role.value,
            "content": self.content,
        }
        if self.images:
            data["images"] = list(self.images)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Instantiate a Message from a dictionary."""
        return cls(
            role=data.get("role", MessageRole.USER),
            content=data.get("content", ""),
            timestamp=data.get("timestamp"),
            metadata=data.get("metadata"),
            images=data.get("images"),
        )

    # Provide dictionary-like access (e.g. msg["role"], msg["content"]) for backwards compatibility
    def __getitem__(self, item: str) -> Any:
        if item == "role":
            return self.role.value
        elif item == "content":
            return self.content
        elif item == "timestamp":
            return self.timestamp
        elif item == "metadata":
            return self.metadata
        elif item == "images":
            return self.images
        raise KeyError(f"Invalid Message key: {item}")

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "role":
            self.role = MessageRole(value) if isinstance(value, str) else value
        elif key == "content":
            self.content = str(value)
        elif key == "timestamp":
            self.timestamp = float(value)
        elif key == "metadata":
            self.metadata = dict(value)
        elif key == "images":
            self.images = list(value) if value else []
        else:
            raise KeyError(f"Invalid Message key: {key}")

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __repr__(self) -> str:
        return f"Message(role={self.role.value!r}, content={self.content[:30]!r}...)"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Message):
            return self.role == other.role and self.content == other.content
        if isinstance(other, dict):
            return self.role.value == other.get("role") and self.content == other.get("content")
        return False

