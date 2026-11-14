import logging
from typing import Any, Dict, List, Optional, Set, Union

from PySide6.QtCore import QObject, QThread, Signal

from src.agents.models import AgentRole, AgentStatus
from src.providers.manager import ProviderManager, get_default_provider_manager
from src.providers.models import ModelInfo
from src.teams.models import TeamExecutionResult
from src.teams.observer import TeamExecutionObserver, TeamExecutionSnapshot
from src.teams.orchestrator import TeamOrchestrator
from src.workflow.models import WorkflowStatus

logger = logging.getLogger("RelayAI.TeamsWorker")

# Retains strong Python references to running workers during window closing/application shutdown
# until their background QThread terminates, preventing native Qt memory access violation crashes.
_ORPHANED_TEAM_WORKERS: Set[QThread] = set()


def safely_anchor_worker(worker: QThread) -> None:
    """Retains strong Python reference to a running worker during shutdown so C++ never crashes."""
    if worker.isRunning():
        _ORPHANED_TEAM_WORKERS.add(worker)
        try:
            worker.finished.connect(lambda w=worker: _ORPHANED_TEAM_WORKERS.discard(w))
            worker.finished.connect(worker.deleteLater)
        except Exception:
            pass
    else:
        try:
            worker.deleteLater()
        except Exception:
            pass


class QtTeamObserverMeta(type(QObject), type(TeamExecutionObserver)):
    """Metaclass resolving Shiboken.ObjectType and abc.ABCMeta conflict."""
    pass


class QtTeamObserver(QObject, TeamExecutionObserver, metaclass=QtTeamObserverMeta):
    """Thread-safe Qt adapter for TeamExecutionObserver emitting queued signals across threads."""

    team_started = Signal(str, str)  # (team_id, task_id)
    step_started = Signal(object)  # (snapshot: TeamExecutionSnapshot)
    step_completed = Signal(object)  # (snapshot: TeamExecutionSnapshot)
    revision_requested = Signal(str, str, str, int)  # (target_step_id, reason, feedback, loop_count)
    team_completed = Signal(object)  # (result: TeamExecutionResult)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)

    def on_team_started(self, team_id: str, task_id: str) -> None:
        self.team_started.emit(team_id, task_id)

    def on_step_started(self, snapshot: TeamExecutionSnapshot) -> None:
        self.step_started.emit(snapshot)

    def on_step_completed(self, snapshot: TeamExecutionSnapshot) -> None:
        self.step_completed.emit(snapshot)

    def on_revision_requested(
        self, target_step_id: str, reason: str, feedback: str, loop_count: int
    ) -> None:
        self.revision_requested.emit(target_step_id, reason, feedback, loop_count)

    def on_team_completed(self, result: TeamExecutionResult) -> None:
        self.team_completed.emit(result)


class TeamExecutionWorker(QThread):
    """Background QThread executing TeamOrchestrator without blocking the Qt GUI thread."""

    # Forwarded lifecycle signals
    team_started = Signal(str, str)
    step_started = Signal(object)
    step_completed = Signal(object)
    revision_requested = Signal(str, str, str, int)
    team_completed = Signal(object)
    execution_error = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        orchestrator: TeamOrchestrator,
        task: Union[str, Any],
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.orchestrator = orchestrator
        self.task = task
        self._is_cancelled = False

        # Build and attach Qt-enabled observer
        self.observer = QtTeamObserver(parent=self)
        self.orchestrator.add_observer(self.observer)

        # Connect observer signals directly to worker signals
        self.observer.team_started.connect(self.team_started)
        self.observer.step_started.connect(self.step_started)
        self.observer.step_completed.connect(self.step_completed)
        self.observer.revision_requested.connect(self.revision_requested)
        self.observer.team_completed.connect(self._on_observer_team_completed)

    def _on_observer_team_completed(self, result: TeamExecutionResult) -> None:
        """Handle completion from observer, ensuring UI receives final result before any thread teardown."""
        self.team_completed.emit(result)
        if result.status == WorkflowStatus.CANCELLED:
            self.cancelled.emit()

    def cancel(self, wait_ms: int = 0) -> None:
        """Cooperatively request cancellation.
        
        Important: Completion signals are NOT disconnected here. The terminal
        execution result (with status=CANCELLED) must be delivered to the UI.
        """
        self._is_cancelled = True
        try:
            self.orchestrator.cancel()
        except Exception as exc:
            logger.warning("Failed to propagate cancellation to team orchestrator: %s", exc)

        if wait_ms > 0 and self.isRunning():
            self.wait(wait_ms)

    def disconnect_signals(self) -> None:
        """Safely disconnect all outgoing signals during widget teardown to prevent calling destroyed slots."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for sig in [
                self.team_started,
                self.step_started,
                self.step_completed,
                self.revision_requested,
                self.team_completed,
                self.execution_error,
                self.cancelled,
            ]:
                try:
                    sig.disconnect()
                except Exception:
                    pass

    def run(self) -> None:
        """Executes the orchestrator pipeline off the GUI thread."""
        try:
            self.orchestrator.execute(self.task)
            # Terminal result dispatch (COMPLETED, FAILED, CANCELLED, TIMEOUT)
            # is guaranteed to fire through on_team_completed via the orchestrator's finally block.
        except Exception as exc:
            logger.exception("Unexpected exception inside TeamExecutionWorker.run: %s", exc)
            self.execution_error.emit(str(exc))
        finally:
            try:
                self.orchestrator.remove_observer(self.observer)
            except Exception:
                pass


class ModelDiscoveryWorker(QThread):
    """Background QThread discovering available models for a provider without blocking the GUI thread."""

    discovery_started = Signal(str)  # (provider_id)
    models_loaded = Signal(str, list)  # (provider_id, List[ModelInfo])
    discovery_failed = Signal(str, str, list)  # (provider_id, error_message, List[ModelInfo] fallbacks)

    def __init__(
        self,
        provider_id: str,
        provider_manager: Optional[ProviderManager] = None,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.provider_id = provider_id
        self.provider_manager = provider_manager or get_default_provider_manager()
        self._is_cancelled = False

    def cancel(self, wait_ms: int = 0) -> None:
        """Cooperatively request cancellation of model discovery."""
        self._is_cancelled = True
        if wait_ms > 0 and self.isRunning():
            self.wait(wait_ms)

    def _get_fallback_models(self) -> List[ModelInfo]:
        """Return provider-specific fallback ModelInfo if discovery fails or models list is empty."""
        pid = self.provider_id.lower()
        if pid == "ollama":
            return [ModelInfo(id="llama3.2:3b", name="Llama 3.2 3B (Fallback)", provider_id="ollama")]
        elif pid == "groq":
            return [ModelInfo(id="openai/gpt-oss-120b", name="GPT-OSS 120B (Fallback)", provider_id="groq")]
        elif pid == "openai":
            return [ModelInfo(id="gpt-4o-mini", name="GPT-4o Mini (Fallback)", provider_id="openai")]
        elif pid == "gemini":
            return [ModelInfo(id="gemini-3.5-flash", name="Gemini 3.5 Flash (Fallback)", provider_id="gemini")]
        return [ModelInfo(id="default-model", name=f"{self.provider_id.title()} Default", provider_id=self.provider_id)]

    def run(self) -> None:
        """Fetches models off the main GUI thread."""
        self.discovery_started.emit(self.provider_id)

        if self._is_cancelled:
            return

        provider = self.provider_manager.get_provider(self.provider_id)
        if not provider:
            fallbacks = self._get_fallback_models()
            self.discovery_failed.emit(
                self.provider_id,
                f"Provider '{self.provider_id}' is not registered.",
                fallbacks,
            )
            return

        try:
            models = provider.get_available_models()
            if self._is_cancelled:
                return

            if models:
                self.models_loaded.emit(self.provider_id, list(models))
            else:
                fallbacks = self._get_fallback_models()
                self.discovery_failed.emit(
                    self.provider_id,
                    f"Provider '{self.provider_id}' returned 0 models.",
                    fallbacks,
                )
        except Exception as exc:
            if self._is_cancelled:
                return
            logger.warning("Model discovery failed for provider '%s': %s", self.provider_id, exc)
            fallbacks = self._get_fallback_models()
            self.discovery_failed.emit(
                self.provider_id,
                f"Discovery failed: {exc}",
                fallbacks,
            )
