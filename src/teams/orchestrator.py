import logging
import threading
import time
from typing import Callable, List, Optional, Set, Union

from src.agents import (
    AgentRegistry,
    AgentRole,
    AgentStatus,
    TaskContext,
)
from src.providers.manager import ProviderManager, get_default_provider_manager
from src.teams.exceptions import TeamExecutionError
from src.teams.factory import TeamFactory
from src.teams.models import TeamConfig, TeamExecutionResult
from src.teams.observer import TeamExecutionObserver, TeamExecutionSnapshot
from src.workflow.engine import WorkflowEngine
from src.workflow.models import StepExecutionRecord, WorkflowResult, WorkflowStatus

logger = logging.getLogger("RelayAI.TeamOrchestrator")


class TeamOrchestrator:
    """Coordinates and executes an AI Team, handling lifecycle state, observability, and cleanup."""

    def __init__(
        self,
        config: TeamConfig,
        factory: Optional[TeamFactory] = None,
        provider_manager: Optional[ProviderManager] = None,
        registry: Optional[AgentRegistry] = None,
    ):
        self.config = config
        self.provider_manager = provider_manager or get_default_provider_manager()
        self.registry = registry or AgentRegistry()
        self.factory = factory or TeamFactory(self.provider_manager, self.registry)

        self._observers: List[TeamExecutionObserver] = []
        self._observers_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()

        self._is_running = False
        self._active_engine: Optional[WorkflowEngine] = None
        self._registered_agent_ids: Set[str] = set()

    def add_observer(self, observer: TeamExecutionObserver) -> None:
        """Register an observer for execution lifecycle notifications."""
        with self._observers_lock:
            if observer not in self._observers:
                self._observers.append(observer)

    def remove_observer(self, observer: TeamExecutionObserver) -> None:
        """Unregister a lifecycle observer."""
        with self._observers_lock:
            if observer in self._observers:
                self._observers.remove(observer)

    def _notify_observers(self, event_fn: Callable[[TeamExecutionObserver], None]) -> None:
        """Safely notify all registered observers. Observer exceptions are captured and never abort execution."""
        with self._observers_lock:
            observers_copy = list(self._observers)

        for observer in observers_copy:
            try:
                event_fn(observer)
            except Exception as exc:
                logger.error(
                    "Observer '%s' raised exception during event notification: %s",
                    type(observer).__name__,
                    exc,
                )

    def cancel(self) -> None:
        """Thread-safely request cooperative cancellation of the active team execution."""
        with self._lifecycle_lock:
            if self._active_engine is not None:
                self._active_engine.cancel()

    def cleanup(self) -> None:
        """Unregister ONLY agents owned by this team from the registry."""
        with self._lifecycle_lock:
            for agent_id in list(self._registered_agent_ids):
                try:
                    self.registry.unregister(agent_id)
                except Exception as exc:
                    logger.warning("Failed to unregister team agent '%s': %s", agent_id, exc)
            self._registered_agent_ids.clear()

    def execute(self, task_or_prompt: Union[str, TaskContext]) -> TeamExecutionResult:
        """Execute the configured team pipeline on a user task.
        
        Guarantees:
        1. Single-task concurrency lock on the same orchestrator instance.
        2. Terminal on_team_completed observer event fires exactly once for every outcome.
        3. Observer failures are captured and never halt workflow.
        """
        # 1. Concurrency guard
        with self._lifecycle_lock:
            if self._is_running:
                raise TeamExecutionError(
                    f"Team '{self.config.team_id}' is already executing a task. "
                    f"Concurrent execution on the same orchestrator instance is prohibited.",
                    team_id=self.config.team_id,
                )
            self._is_running = True

        terminal_dispatched = False
        result: Optional[TeamExecutionResult] = None
        task_id = "unknown"
        started_event_dispatched = False

        try:
            # 2. Standardize context
            if isinstance(task_or_prompt, str):
                context = TaskContext(user_prompt=task_or_prompt)
            elif isinstance(task_or_prompt, TaskContext):
                context = task_or_prompt
            else:
                raise TeamExecutionError(
                    f"Expected str or TaskContext, got {type(task_or_prompt)}.",
                    team_id=self.config.team_id,
                )
            task_id = context.task_id

            # 3. Instantiate and register only enabled agents
            registered = self.factory.instantiate_and_register_agents(self.config, overwrite=True)
            self._registered_agent_ids.update(a.agent_id for a in registered)

            # 4. Compile dynamic workflow configuration
            wf_config = self.factory.compile_workflow_config(self.config)

            # 5. Build callbacks for real-time observability
            def on_step_start(step_id: str, agent_id: str, role: AgentRole, revision_iter: int):
                snap = TeamExecutionSnapshot(
                    step_id=step_id,
                    agent_id=agent_id,
                    role=role,
                    status=AgentStatus.RUNNING,
                    started_at=time.time(),
                    revision_iteration=revision_iter,
                )
                self._notify_observers(lambda o: o.on_step_started(snap))

            def on_step_finish(record: StepExecutionRecord):
                snap = TeamExecutionSnapshot.from_step_record(record)
                self._notify_observers(lambda o: o.on_step_completed(snap))

            def on_revision(target: str, reason: str, feedback: str, loop_count: int):
                self._notify_observers(lambda o: o.on_revision_requested(target, reason, feedback, loop_count))

            engine = WorkflowEngine(
                config=wf_config,
                registry=self.registry,
                on_step_start=on_step_start,
                on_step_finish=on_step_finish,
                on_revision=on_revision,
            )

            with self._lifecycle_lock:
                self._active_engine = engine

            self._notify_observers(lambda o: o.on_team_started(self.config.team_id, context.task_id))
            started_event_dispatched = True

            wf_result: WorkflowResult = engine.execute(context)
            result = TeamExecutionResult.from_workflow_result(wf_result, self.config.team_id)
            return result
        except Exception as exc:
            if started_event_dispatched:
                result = TeamExecutionResult(
                    team_id=self.config.team_id,
                    task_id=task_id,
                    status=WorkflowStatus.FAILED,
                    error_summary=str(exc),
                )
            raise
        finally:
            with self._lifecycle_lock:
                self._is_running = False
                self._active_engine = None

            if started_event_dispatched and not terminal_dispatched and result is not None:
                terminal_dispatched = True
                self._notify_observers(lambda o, res=result: o.on_team_completed(res))

