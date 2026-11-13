from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Dict, List, Optional

from src.agents.models import AgentResult, AgentRole, AgentStatus
from src.workflow.exceptions import WorkflowConfigurationError


class WorkflowStatus(str, Enum):
    """Lifecycle states of a multi-agent workflow."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepFailurePolicy(str, Enum):
    """Action taken when an individual step execution fails."""
    ABORT = "abort"                    # Stop entire workflow immediately
    RETRY_THEN_ABORT = "retry_then_abort"  # Retry up to max_retries; abort if still failing
    SKIP_STEP = "skip_step"            # Record failure, skip step, and continue to next
    FALLBACK_STEP = "fallback_step"    # Execute explicit fallback agent if configured


@dataclass
class CriticReviewResult:
    """Structured evaluation produced by a Critic agent determining revision needs."""
    revision_required: bool = False
    feedback: str = ""
    target_step_id: Optional[str] = None
    reason: Optional[str] = None


@dataclass
class WorkflowStepConfig:
    """Configuration for an individual workflow execution step."""
    step_id: str
    agent_id: str
    step_name: str = ""
    instructions: str = ""
    failure_policy: StepFailurePolicy = StepFailurePolicy.ABORT
    fallback_agent_id: Optional[str] = None
    max_retries: int = 1
    dependencies: List[str] = field(default_factory=list)
    is_critic_step: bool = False
    allowed_revision_targets: List[str] = field(default_factory=list)
    required_inputs: Optional[List[str]] = None
    is_optional: bool = False

    def __post_init__(self) -> None:
        if not self.step_id or not isinstance(self.step_id, str):
            raise WorkflowConfigurationError("WorkflowStepConfig requires a non-empty string for 'step_id'.")
        if not self.agent_id or not isinstance(self.agent_id, str):
            raise WorkflowConfigurationError("WorkflowStepConfig requires a non-empty string for 'agent_id'.")
        if not self.step_name:
            self.step_name = self.step_id
        if self.max_retries < 0:
            raise WorkflowConfigurationError("WorkflowStepConfig 'max_retries' cannot be negative.")
        if self.failure_policy == StepFailurePolicy.FALLBACK_STEP and not self.fallback_agent_id:
            raise WorkflowConfigurationError(
                f"Step '{self.step_id}' specifies FALLBACK_STEP policy but no 'fallback_agent_id' is defined."
            )


@dataclass
class StepExecutionRecord:
    """Detailed audit record of an individual step execution turn."""
    step_id: str
    agent_id: str
    agent_role: AgentRole
    status: AgentStatus
    started_at: float
    finished_at: Optional[float] = None
    duration_sec: float = 0.0
    result: Optional[AgentResult] = None
    attempt_count: int = 1
    revision_iteration: int = 0
    critic_review: Optional[CriticReviewResult] = None
    error: Optional[str] = None


@dataclass
class WorkflowConfig:
    """Global configuration governing a multi-agent workflow execution."""
    workflow_id: str
    name: str
    description: str = ""
    steps: List[WorkflowStepConfig] = field(default_factory=list)
    max_steps: int = 15
    max_revision_loops: int = 3
    timeout_sec: float = 300.0
    token_budget: Optional[int] = None
    cost_budget: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.workflow_id:
            raise WorkflowConfigurationError("WorkflowConfig requires a non-empty 'workflow_id'.")
        if not self.name:
            raise WorkflowConfigurationError("WorkflowConfig requires a non-empty 'name'.")
        if self.max_steps <= 0:
            raise WorkflowConfigurationError("WorkflowConfig 'max_steps' must be strictly positive.")
        if self.max_revision_loops < 0:
            raise WorkflowConfigurationError("WorkflowConfig 'max_revision_loops' cannot be negative.")
        if self.timeout_sec <= 0:
            raise WorkflowConfigurationError("WorkflowConfig 'timeout_sec' must be strictly positive.")
        if self.token_budget is not None and self.token_budget <= 0:
            raise WorkflowConfigurationError("WorkflowConfig 'token_budget' must be strictly positive.")
        if self.cost_budget is not None and self.cost_budget <= 0.0:
            raise WorkflowConfigurationError("WorkflowConfig 'cost_budget' must be strictly positive.")


@dataclass
class WorkflowResult:
    """Consolidated outcome of a complete multi-agent workflow run."""
    workflow_id: str
    task_id: str
    status: WorkflowStatus
    final_output: str = ""
    step_records: List[StepExecutionRecord] = field(default_factory=list)
    total_duration_sec: float = 0.0
    total_prompt_tokens: Optional[int] = None
    total_completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    estimated_cost: Optional[float] = None
    failed_steps: List[str] = field(default_factory=list)
    skipped_steps: List[str] = field(default_factory=list)
    error_summary: Optional[str] = None

    @property
    def is_success(self) -> bool:
        """Indicates whether the workflow reached an acceptable completion state."""
        return self.status in (WorkflowStatus.COMPLETED, WorkflowStatus.PARTIAL_SUCCESS)
