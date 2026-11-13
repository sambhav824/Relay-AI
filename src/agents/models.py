from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Dict, Optional

from src.agents.exceptions import AgentConfigurationError


class AgentRole(str, Enum):
    """Specialized functional role assigned to an AI agent in a collaboration team."""
    PLANNER = "planner"          # Breaks down objectives into structured milestones and requirements
    RESEARCHER = "researcher"    # Context gathering, document inspection, factual retrieval
    CREATOR = "creator"          # Generates substantive draft code, architecture, or content
    ARCHITECT = "architect"      # Designs schemas, interfaces, data pipelines, and system structure
    CRITIC = "critic"            # Evaluates drafts, identifies flaws, edge cases, and contradictions
    FINALIZER = "finalizer"      # Synthesizes inputs into a refined, formatted user-facing answer
    CUSTOM = "custom"            # Extensible user-defined specialized role


class AgentStatus(str, Enum):
    """Lifecycle execution status of an AI agent."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentConfig:
    """Configuration parameters specifying provider assignment and execution constraints."""
    provider_id: str
    model_id: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    timeout_sec: float = 60.0
    max_retries: int = 1
    system_instruction: str = ""
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider_id or not isinstance(self.provider_id, str):
            raise AgentConfigurationError("AgentConfig requires a non-empty string for 'provider_id'.")
        if not self.model_id or not isinstance(self.model_id, str):
            raise AgentConfigurationError("AgentConfig requires a non-empty string for 'model_id'.")
        if self.timeout_sec <= 0:
            raise AgentConfigurationError("AgentConfig 'timeout_sec' must be strictly positive.")
        if self.max_retries < 0:
            raise AgentConfigurationError("AgentConfig 'max_retries' cannot be negative.")
        if not (0.0 <= self.temperature <= 2.0):
            raise AgentConfigurationError("AgentConfig 'temperature' must be between 0.0 and 2.0.")


@dataclass
class AgentExecutionMetadata:
    """Execution metrics and audit metadata captured during an agent turn."""
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    duration_sec: float = 0.0
    provider_id: str = ""
    model_id: str = ""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost_estimate: Optional[float] = None
    attempt_count: int = 1
    critic_review: Optional[Any] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Outcome of an agent execution turn."""
    agent_id: str
    agent_role: AgentRole
    task_id: str
    status: AgentStatus
    output: str = ""
    error: Optional[str] = None
    metadata: AgentExecutionMetadata = field(default_factory=AgentExecutionMetadata)

    @property
    def is_success(self) -> bool:
        """Convenience property indicating clean completion."""
        return self.status == AgentStatus.COMPLETED

