from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.agents.models import AgentRole
from src.teams.exceptions import TeamConfigurationError
from src.workflow.models import (
    StepExecutionRecord,
    StepFailurePolicy,
    WorkflowResult,
    WorkflowStatus,
)


@dataclass
class TeamMemberConfig:
    """Configuration for an individual specialized agent within an AI Team."""
    role: AgentRole
    provider_id: str
    model_id: str
    agent_id: str = ""
    enabled: bool = True
    temperature: Optional[float] = None
    timeout_sec: float = 60.0
    max_retries: int = 1
    system_instruction: Optional[str] = None
    failure_policy: StepFailurePolicy = StepFailurePolicy.ABORT
    fallback_agent_id: Optional[str] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.role, AgentRole):
            raise TeamConfigurationError(f"TeamMemberConfig 'role' must be an AgentRole enum, got {type(self.role)}.")
        if self.enabled:
            if not self.provider_id or not isinstance(self.provider_id, str):
                raise TeamConfigurationError(f"Enabled member ({self.role.value}) requires a non-empty string for 'provider_id'.")
            if not self.model_id or not isinstance(self.model_id, str):
                raise TeamConfigurationError(f"Enabled member ({self.role.value}) requires a non-empty string for 'model_id'.")
        if self.timeout_sec <= 0:
            raise TeamConfigurationError(f"Member ({self.role.value}) 'timeout_sec' must be strictly positive.")
        if self.max_retries < 0:
            raise TeamConfigurationError(f"Member ({self.role.value}) 'max_retries' cannot be negative.")


@dataclass
class TeamConfig:
    """Specification defining an AI collaboration team, its member roles, and execution limits."""
    team_id: str
    name: str
    description: str = ""
    members: Dict[AgentRole, TeamMemberConfig] = field(default_factory=dict)
    timeout_sec: float = 300.0
    max_revision_loops: int = 2
    max_steps: int = 15
    token_budget: Optional[int] = None
    cost_budget: Optional[float] = None
    default_failure_policy: StepFailurePolicy = StepFailurePolicy.ABORT

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate team configuration integrity, membership, and enable/disable constraints."""
        if not self.team_id or not isinstance(self.team_id, str) or not self.team_id.strip():
            raise TeamConfigurationError("TeamConfig requires a non-empty string for 'team_id'.")
        if not self.name or not isinstance(self.name, str) or not self.name.strip():
            raise TeamConfigurationError("TeamConfig requires a non-empty string for 'name'.", team_id=self.team_id)
        if self.timeout_sec <= 0:
            raise TeamConfigurationError("TeamConfig 'timeout_sec' must be strictly positive.", team_id=self.team_id)
        if self.max_revision_loops < 0:
            raise TeamConfigurationError("TeamConfig 'max_revision_loops' cannot be negative.", team_id=self.team_id)
        if self.max_steps <= 0:
            raise TeamConfigurationError("TeamConfig 'max_steps' must be strictly positive.", team_id=self.team_id)
        if self.token_budget is not None and self.token_budget <= 0:
            raise TeamConfigurationError("TeamConfig 'token_budget' must be strictly positive.", team_id=self.team_id)
        if self.cost_budget is not None and self.cost_budget <= 0.0:
            raise TeamConfigurationError("TeamConfig 'cost_budget' must be strictly positive.", team_id=self.team_id)

        # Creator is strictly mandatory
        creator_member = self.members.get(AgentRole.CREATOR)
        if not creator_member or not creator_member.enabled:
            raise TeamConfigurationError(
                "AI Team configuration invalid: 'CREATOR' agent is mandatory and must be present and enabled.",
                team_id=self.team_id,
            )

        # If Critic is enabled, Creator must be enabled (already guaranteed above, but checked explicitly)
        critic_member = self.members.get(AgentRole.CRITIC)
        if critic_member and critic_member.enabled and not (creator_member and creator_member.enabled):
            raise TeamConfigurationError(
                "Critic cannot be enabled without an enabled Creator agent as its revision target.",
                team_id=self.team_id,
            )


@dataclass
class TeamExecutionResult:
    """Consolidated outcome of an AI Team execution turn."""
    team_id: str
    task_id: str
    status: WorkflowStatus
    final_deliverable: str = ""
    step_records: List[StepExecutionRecord] = field(default_factory=list)
    step_outputs: Dict[str, str] = field(default_factory=dict)
    total_duration_sec: float = 0.0
    total_prompt_tokens: Optional[int] = None
    total_completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    revision_count: int = 0
    failed_steps: List[str] = field(default_factory=list)
    skipped_steps: List[str] = field(default_factory=list)
    error_summary: Optional[str] = None

    @property
    def is_success(self) -> bool:
        return self.status in (WorkflowStatus.COMPLETED, WorkflowStatus.PARTIAL_SUCCESS)

    @property
    def is_timeout(self) -> bool:
        """Indicates whether the team execution failed specifically due to a global timeout."""
        return (
            self.status == WorkflowStatus.FAILED
            and bool(self.error_summary and "workflow exceeded global timeout" in self.error_summary.lower())
        )

    @classmethod
    def from_workflow_result(cls, wf_result: WorkflowResult, team_id: str) -> "TeamExecutionResult":
        """Translate lower-level WorkflowResult into domain TeamExecutionResult."""
        # Calculate total revision attempts from step records
        rev_count = sum(1 for r in wf_result.step_records if r.revision_iteration > 0)

        return cls(
            team_id=team_id,
            task_id=wf_result.task_id,
            status=wf_result.status,
            final_deliverable=wf_result.final_output,
            step_records=list(wf_result.step_records),
            step_outputs=dict(wf_result.step_outputs),
            total_duration_sec=wf_result.total_duration_sec,
            total_prompt_tokens=wf_result.total_prompt_tokens,
            total_completion_tokens=wf_result.total_completion_tokens,
            total_tokens=wf_result.total_tokens,
            revision_count=rev_count,
            failed_steps=list(wf_result.failed_steps),
            skipped_steps=list(wf_result.skipped_steps),
            error_summary=wf_result.error_summary,
        )

