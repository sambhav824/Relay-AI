from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelInfo:
    """Standardized descriptor of an AI model available through a provider."""
    id: str
    name: str
    provider_id: str
    is_local: bool = False
    context_window: Optional[int] = None
    description: str = ""

    def __str__(self) -> str:
        return f"{self.name} ({self.provider_id})"

