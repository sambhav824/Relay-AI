from abc import ABC
from dataclasses import dataclass
from typing import Optional

from src.agents.models import AgentRole, AgentStatus
from src.teams.models import TeamExecutionResult
from src.workflow.models import StepExecutionRecord


@dataclass(frozen=True)
class TeamExecutionSnapshot:
    """Immutable real-time telemetry snapshot of an agent step execution turn."""
    step_id: str
    agent_id: str
    role: AgentRole
    status: AgentStatus
    started_at: float
    finished_at: Optional[float] = None
    duration_sec: float = 0.0
    provider_id: str = ""
    model_id: str = ""
    verified_prompt_tokens: Optional[int] = None
    verified_completion_tokens: Optional[int] = None
    verified_total_tokens: Optional[int] = None
    revision_iteration: int = 0
    error: Optional[str] = None

    @classmethod
    def from_step_record(cls, record: StepExecutionRecord) -> "TeamExecutionSnapshot":
        """Construct an immutable snapshot directly from a StepExecutionRecord without exposing secrets."""
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        provider_id = ""
        model_id = ""

        if record.result and record.result.metadata:
            meta = record.result.metadata
            prompt_tokens = meta.prompt_tokens
            completion_tokens = meta.completion_tokens
            total_tokens = meta.total_tokens
            provider_id = meta.provider_id
            model_id = meta.model_id

        return cls(
            step_id=record.step_id,
            agent_id=record.agent_id,
            role=record.agent_role,
            status=record.status,
            started_at=record.started_at,
            finished_at=record.finished_at,
            duration_sec=record.duration_sec,
            provider_id=provider_id,
            model_id=model_id,
            verified_prompt_tokens=prompt_tokens,
            verified_completion_tokens=completion_tokens,
            verified_total_tokens=total_tokens,
            revision_iteration=record.revision_iteration,
            error=record.error,
        )


class TeamExecutionObserver(ABC):
    """Abstract listener interface for real-time AI Team execution observability."""

    def on_team_started(self, team_id: str, task_id: str) -> None:
        """Invoked when team workflow execution commences."""
        pass

    def on_step_started(self, snapshot: TeamExecutionSnapshot) -> None:
        """Invoked when an individual agent begins executing its step."""
        pass

    def on_step_completed(self, snapshot: TeamExecutionSnapshot) -> None:
        """Invoked when an individual agent completes its execution turn."""
        pass

    def on_revision_requested(
        self,
        target_step_id: str,
        reason: str,
        feedback: str,
        loop_count: int,
    ) -> None:
        """Invoked when the Critic triggers a revision loop targeting a previous agent."""
        pass

    def on_team_completed(self, result: TeamExecutionResult) -> None:
        """Invoked exactly once when the team workflow terminates (COMPLETED, FAILED, or CANCELLED)."""
        pass

