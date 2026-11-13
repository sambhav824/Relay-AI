from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ResponseMetadata:
    """Metadata describing the completion or streaming response."""
    model: str
    provider_id: str
    total_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamChunk:
    """Standardized streaming chunk emitted during token generation."""
    delta: str
    is_final: bool = False
    metadata: Optional[ResponseMetadata] = None

    @property
    def content(self) -> str:
        """Alias for delta for convenient consumption."""
        return self.delta

