from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Union

from src.core.events import StreamChunk
from src.core.message import Message, MessageRole
from src.providers.models import ModelInfo


class AIProvider(ABC):
    """Abstract Base Class for all Relay AI model providers."""

    provider_id: str = "base"
    display_name: str = "Base Provider"
    is_local: bool = False
    supports_streaming: bool = True

    @abstractmethod
    def generate(self, messages: List[Union[Message, Dict[str, Any]]], model: str, **kwargs) -> str:
        """Synchronously generate a full response from the model."""
        pass

    @abstractmethod
    def stream(self, messages: List[Union[Message, Dict[str, Any]]], model: str, **kwargs) -> Iterator[StreamChunk]:
        """Stream response chunks in real time from the provider."""
        pass

    @abstractmethod
    def validate_connection(self) -> bool:
        """Validate connectivity and authentication with the provider."""
        pass

    @abstractmethod
    def get_available_models(self) -> List[ModelInfo]:
        """Fetch available models from the provider."""
        pass

    def normalize_messages(self, messages: List[Union[Message, Dict[str, Any]]]) -> List[Message]:
        """Helper to ensure all items in the messages list are standardized Message instances."""
        normalized: List[Message] = []
        for m in messages:
            if isinstance(m, Message):
                normalized.append(m)
            elif isinstance(m, dict):
                normalized.append(Message.from_dict(m))
            else:
                normalized.append(Message(role=MessageRole.USER, content=str(m)))
        return normalized

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.provider_id!r} local={self.is_local}>"

