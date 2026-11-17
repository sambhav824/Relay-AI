import logging
import time
import unittest
from typing import List, Optional
from unittest.mock import MagicMock

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

from src.agents.models import AgentRole, AgentStatus
from src.core.events import ResponseMetadata
from src.providers.base import AIProvider
from src.providers.manager import ProviderManager
from src.providers.models import ModelInfo
from src.teams.models import TeamExecutionResult
from src.teams.observer import TeamExecutionSnapshot
from src.teams.orchestrator import TeamOrchestrator
from src.ui.teams.worker import (
    ModelDiscoveryWorker,
    QtTeamObserver,
    TeamExecutionWorker,
    _ORPHANED_TEAM_WORKERS,
    safely_anchor_worker,
)
from src.workflow.models import CriticReviewResult, StepExecutionRecord, WorkflowStatus
from PySide6.QtGui import QGuiApplication
from src.agents.models import AgentExecutionMetadata, AgentResult
from src.ui.teams.agent_card import AgentStepCard, CardStatus, normalize_card_status
from src.ui.teams.config_panel import TeamConfigPanel
from src.ui.teams.deliverable_panel import (
    DeliverableState,
    FinalDeliverablePanel,
    render_markdown_safe,
)
from src.ui.teams.teams_view import AITeamsView, TeamRunHistoryItem
from src.ui.main_window import MainWindow

# Ensure a QApplication instance exists for QObject and signal/slot testing
_app = QApplication.instance() or QApplication([])


class TestQtTeamObserver(unittest.TestCase):
    """Test QtTeamObserver signal emission and contract forwarding."""

    def test_qt_team_observer_emits_all_signals(self):
        observer = QtTeamObserver()

        received_events = []

        observer.team_started.connect(lambda team_id, task_id: received_events.append(("started", team_id, task_id)))
        observer.step_started.connect(lambda snap: received_events.append(("step_started", snap.step_id)))
        observer.step_completed.connect(lambda snap: received_events.append(("step_completed", snap.step_id, snap.status)))
        observer.revision_requested.connect(
            lambda target, reason, fb, loop: received_events.append(("revision", target, reason, fb, loop))
        )
        observer.team_completed.connect(lambda res: received_events.append(("completed", res.team_id, res.status)))

        # 1. on_team_started
        observer.on_team_started("team_1", "task_abc")
        self.assertEqual(len(received_events), 1)
        self.assertEqual(received_events[-1], ("started", "team_1", "task_abc"))

        # 2. on_step_started
        snap_start = TeamExecutionSnapshot(
            step_id="step_planner",
            agent_id="ag_plan",
            role=AgentRole.PLANNER,
            status=AgentStatus.RUNNING,
            started_at=100.0,
        )
        observer.on_step_started(snap_start)
        self.assertEqual(len(received_events), 2)
        self.assertEqual(received_events[-1], ("step_started", "step_planner"))

        # 3. on_step_completed
        snap_comp = TeamExecutionSnapshot(
            step_id="step_planner",
            agent_id="ag_plan",
            role=AgentRole.PLANNER,
            status=AgentStatus.COMPLETED,
            started_at=100.0,
            finished_at=101.5,
            duration_sec=1.5,
            verified_total_tokens=150,
        )
        observer.on_step_completed(snap_comp)
        self.assertEqual(len(received_events), 3)
        self.assertEqual(received_events[-1], ("step_completed", "step_planner", AgentStatus.COMPLETED))

        # 4. on_revision_requested
        observer.on_revision_requested("step_creator", "Needs tests", "Add unit tests", 1)
        self.assertEqual(len(received_events), 4)
        self.assertEqual(received_events[-1], ("revision", "step_creator", "Needs tests", "Add unit tests", 1))

        # 5. on_team_completed
        final_res = TeamExecutionResult(
            team_id="team_1",
            task_id="task_abc",
            status=WorkflowStatus.COMPLETED,
            final_deliverable="Final code",
        )
        observer.on_team_completed(final_res)
        self.assertEqual(len(received_events), 5)
        self.assertEqual(received_events[-1], ("completed", "team_1", WorkflowStatus.COMPLETED))


class TestTeamExecutionWorker(unittest.TestCase):
    """Test TeamExecutionWorker background thread execution, signals, and cancellation."""

    def test_worker_successful_execution(self):
        """Worker executes orchestrator in background and forwards all observer events to signals."""
        mock_orch = MagicMock(spec=TeamOrchestrator)
        result_payload = TeamExecutionResult(
            team_id="t_test",
            task_id="task_1",
            status=WorkflowStatus.COMPLETED,
            final_deliverable="Success deliverable",
            total_duration_sec=2.0,
        )

        def fake_execute(task_prompt):
            for obs in mock_orch.add_observer.call_args_list:
                observer_inst = obs[0][0]
                observer_inst.on_team_started("t_test", "task_1")
                observer_inst.on_step_started(
                    TeamExecutionSnapshot(
                        step_id="s1",
                        agent_id="a1",
                        role=AgentRole.CREATOR,
                        status=AgentStatus.RUNNING,
                        started_at=10.0,
                    )
                )
                observer_inst.on_step_completed(
                    TeamExecutionSnapshot(
                        step_id="s1",
                        agent_id="a1",
                        role=AgentRole.CREATOR,
                        status=AgentStatus.COMPLETED,
                        started_at=10.0,
                        finished_at=11.0,
                        duration_sec=1.0,
                        verified_total_tokens=42,
                    )
                )
                observer_inst.on_team_completed(result_payload)
            return result_payload

        mock_orch.execute.side_effect = fake_execute

        worker = TeamExecutionWorker(orchestrator=mock_orch, task="Do test work")

        received = {}

        worker.team_started.connect(lambda t_id, task_id: received.setdefault("started", []).append((t_id, task_id)))
        worker.step_started.connect(lambda snap: received.setdefault("step_started", []).append(snap.step_id))
        worker.step_completed.connect(lambda snap: received.setdefault("step_completed", []).append(snap.step_id))
        worker.team_completed.connect(lambda res: received.setdefault("completed", []).append(res))

        worker.start()
        self.assertTrue(worker.wait(3000))
        self.assertFalse(worker.isRunning())
        QCoreApplication.processEvents()

        self.assertEqual(received["started"], [("t_test", "task_1")])
        self.assertEqual(received["step_started"], ["s1"])
        self.assertEqual(received["step_completed"], ["s1"])
        self.assertEqual(len(received["completed"]), 1)
        self.assertEqual(received["completed"][0].status, WorkflowStatus.COMPLETED)
        self.assertEqual(received["completed"][0].final_deliverable, "Success deliverable")

        mock_orch.remove_observer.assert_called_once()

    def test_worker_cooperative_cancellation_delivers_terminal_result(self):
        """Cancelling worker cooperatively propagates to orchestrator and delivers terminal CANCELLED result."""
        mock_orch = MagicMock(spec=TeamOrchestrator)

        cancelled_result = TeamExecutionResult(
            team_id="t_canc",
            task_id="task_canc",
            status=WorkflowStatus.CANCELLED,
            error_summary="Workflow execution cancelled by user.",
        )

        def slow_execute(task_prompt):
            for _ in range(15):
                time.sleep(0.02)
            for obs in mock_orch.add_observer.call_args_list:
                obs[0][0].on_team_completed(cancelled_result)
            return cancelled_result

        mock_orch.execute.side_effect = slow_execute

        worker = TeamExecutionWorker(orchestrator=mock_orch, task="Task to cancel")

        received = {}
        worker.team_completed.connect(lambda res: received.setdefault("completed", []).append(res))
        worker.cancelled.connect(lambda: received.setdefault("cancelled", True))

        worker.start()
        time.sleep(0.04)

        # Trigger cancellation
        worker.cancel(wait_ms=3000)
        self.assertFalse(worker.isRunning())
        QCoreApplication.processEvents()

        mock_orch.cancel.assert_called()

        self.assertEqual(len(received.get("completed", [])), 1)
        self.assertEqual(received["completed"][0].status, WorkflowStatus.CANCELLED)
        self.assertTrue(received.get("cancelled", False))

    def test_worker_timeout_result_delivery(self):
        """Worker delivers timeout failure result with is_timeout=True without fabricating metrics."""
        mock_orch = MagicMock(spec=TeamOrchestrator)

        timeout_result = TeamExecutionResult(
            team_id="t_to",
            task_id="task_to",
            status=WorkflowStatus.FAILED,
            error_summary="Workflow exceeded global timeout limit of 5.0s.",
            total_duration_sec=5.01,
        )

        def timeout_execute(task_prompt):
            for obs in mock_orch.add_observer.call_args_list:
                obs[0][0].on_team_completed(timeout_result)
            return timeout_result

        mock_orch.execute.side_effect = timeout_execute

        worker = TeamExecutionWorker(orchestrator=mock_orch, task="Timeout task")
        completed_results = []
        worker.team_completed.connect(completed_results.append)

        worker.start()
        self.assertTrue(worker.wait(3000))
        QCoreApplication.processEvents()

        self.assertEqual(len(completed_results), 1)
        res = completed_results[0]
        self.assertEqual(res.status, WorkflowStatus.FAILED)
        self.assertTrue(res.is_timeout)
        self.assertIn("timeout", res.error_summary.lower())

    def test_worker_unhandled_exception_emits_execution_error(self):
        """Worker catches unhandled orchestrator exception and emits execution_error signal."""
        mock_orch = MagicMock(spec=TeamOrchestrator)
        mock_orch.execute.side_effect = RuntimeError("Fatal orchestrator crash")

        worker = TeamExecutionWorker(orchestrator=mock_orch, task="Crash task")
        errors = []
        worker.execution_error.connect(errors.append)

        worker.start()
        self.assertTrue(worker.wait(3000))
        QCoreApplication.processEvents()

        self.assertEqual(len(errors), 1)
        self.assertIn("Fatal orchestrator crash", errors[0])

    def test_worker_disconnect_signals_detaches_slots(self):
        """Calling disconnect_signals detaches connected slots to prevent calls on destroyed objects."""
        mock_orch = MagicMock(spec=TeamOrchestrator)
        worker = TeamExecutionWorker(orchestrator=mock_orch, task="Disconnect task")

        received = []
        worker.team_started.connect(lambda *args: received.append("started"))
        worker.team_completed.connect(lambda *args: received.append("completed"))

        # Explicitly disconnect
        worker.disconnect_signals()

        # Emit directly on observer
        worker.observer.team_started.emit("t", "task")
        worker.observer.team_completed.emit(TeamExecutionResult(team_id="t", task_id="task", status=WorkflowStatus.COMPLETED))
        QCoreApplication.processEvents()

        self.assertEqual(received, [])


class MockDiscoveryProvider(AIProvider):
    """Mock provider for testing ModelDiscoveryWorker."""

    def __init__(self, provider_id: str, models: Optional[List[ModelInfo]] = None, exception: Optional[Exception] = None):
        self.provider_id = provider_id
        self.display_name = provider_id.title()
        self.is_local = (provider_id == "ollama")
        self.models = models if models is not None else []
        self.exception = exception

    def get_available_models(self) -> List[ModelInfo]:
        if self.exception:
            raise self.exception
        return self.models

    def generate(self, messages, model, **kwargs):
        return ""

    def stream(self, messages, model, **kwargs):
        yield from []

    def validate_connection(self):
        return True


class TestModelDiscoveryWorker(unittest.TestCase):
    """Test ModelDiscoveryWorker asynchronous model fetching, fallbacks, and cancellation."""

    def setUp(self):
        self.pm = ProviderManager()
        self.pm._providers.clear()

    def test_model_discovery_success(self):
        """Successfully discovers models and emits models_loaded off the GUI thread."""
        models = [
            ModelInfo(id="m1", name="Model 1", provider_id="test_prov"),
            ModelInfo(id="m2", name="Model 2", provider_id="test_prov"),
        ]
        prov = MockDiscoveryProvider("test_prov", models=models)
        self.pm.register_provider(prov)

        worker = ModelDiscoveryWorker(provider_id="test_prov", provider_manager=self.pm)

        started = []
        loaded = []
        failed = []

        worker.discovery_started.connect(started.append)
        worker.models_loaded.connect(lambda pid, m: loaded.append((pid, m)))
        worker.discovery_failed.connect(lambda pid, err, fb: failed.append((pid, err, fb)))

        worker.start()
        self.assertTrue(worker.wait(3000))
        self.assertFalse(worker.isRunning())
        QCoreApplication.processEvents()

        self.assertEqual(started, ["test_prov"])
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0][0], "test_prov")
        self.assertEqual(len(loaded[0][1]), 2)
        self.assertEqual(failed, [])

    def test_model_discovery_provider_not_registered_fallback(self):
        """Unregistered provider cleanly emits discovery_failed with provider fallback model."""
        worker = ModelDiscoveryWorker(provider_id="ollama", provider_manager=self.pm)

        failed = []
        worker.discovery_failed.connect(lambda pid, err, fb: failed.append((pid, err, fb)))

        worker.start()
        self.assertTrue(worker.wait(3000))
        QCoreApplication.processEvents()

        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0][0], "ollama")
        self.assertIn("not registered", failed[0][1])
        # Fallback for ollama must be llama3.2:3b
        self.assertEqual(len(failed[0][2]), 1)
        self.assertEqual(failed[0][2][0].id, "llama3.2:3b")

    def test_model_discovery_provider_exception_fallback(self):
        """Provider raising error during get_available_models emits discovery_failed with fallback models."""
        prov = MockDiscoveryProvider("gemini", exception=ConnectionError("Google API unreachable"))
        self.pm.register_provider(prov)

        worker = ModelDiscoveryWorker(provider_id="gemini", provider_manager=self.pm)

        failed = []
        worker.discovery_failed.connect(lambda pid, err, fb: failed.append((pid, err, fb)))

        worker.start()
        self.assertTrue(worker.wait(3000))
        QCoreApplication.processEvents()

        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0][0], "gemini")
        self.assertIn("Google API unreachable", failed[0][1])
        # Fallback for gemini is gemini-3.5-flash
        self.assertEqual(failed[0][2][0].id, "gemini-3.5-flash")

    def test_model_discovery_cancellation(self):
        """Cancelled discovery exits cleanly without hanging or emitting after cancel."""
        prov = MockDiscoveryProvider("test_p", models=[ModelInfo("m1", "M1", "test_p")])
        self.pm.register_provider(prov)

        worker = ModelDiscoveryWorker(provider_id="test_p", provider_manager=self.pm)
        worker.cancel()  # Cancel prior to run
        worker.start()
        self.assertTrue(worker.wait(2000))
        self.assertFalse(worker.isRunning())


class TestSafelyAnchorWorker(unittest.TestCase):
    """Test anchoring running workers during application shutdown."""

    def test_safely_anchor_running_and_terminating_worker(self):
        class LongRunningWorker(TeamExecutionWorker):
            def __init__(self):
                super().__init__(orchestrator=MagicMock(), task="dummy")

            def run(self):
                for _ in range(5):
                    time.sleep(0.02)

        w = LongRunningWorker()
        w.start()
        self.assertTrue(w.isRunning())

        # Anchor it
        safely_anchor_worker(w)
        self.assertIn(w, _ORPHANED_TEAM_WORKERS)

        # Wait for worker thread to naturally complete
        self.assertTrue(w.wait(2000))
        QCoreApplication.processEvents()

        self.assertNotIn(w, _ORPHANED_TEAM_WORKERS)


class TestAgentStepCard(unittest.TestCase):
    """Test AgentStepCard component behavior, status badges, metrics, revisions, and themes."""

    def setUp(self):
        self.card = AgentStepCard(
            step_id="step_creator",
            role=AgentRole.CREATOR,
            order=2,
            provider_id="openai",
            model_id="gpt-4o",
            dark=True,
        )
        self.card.show()
        QCoreApplication.processEvents()

    def tearDown(self):
        self.card.close()

    def test_initial_state_and_labels(self):
        """Card initializes with execution order, role, provider/model, and pending status."""
        self.assertEqual(self.card.order_label.text(), "#2")
        self.assertEqual(self.card.role_label.text(), "Creator")
        self.assertIn("openai / gpt-4o", self.card.provider_model_label.text())
        self.assertEqual(self.card.current_status, CardStatus.PENDING)
        self.assertEqual(self.card.status_badge.text(), "Pending")
        self.assertEqual(self.card.duration_label.text(), "--")
        self.assertEqual(self.card.tokens_label.text(), "--")
        self.assertTrue(self.card.is_expanded())

    def test_status_transitions(self):
        """Card handles status transitions across Pending, Running, Completed, Failed, Cancelled, and Skipped."""
        # Running
        self.card.set_status(AgentStatus.RUNNING)
        self.assertEqual(self.card.current_status, CardStatus.RUNNING)
        self.assertEqual(self.card.status_badge.text(), "Running")

        # Completed
        self.card.set_status(AgentStatus.COMPLETED)
        self.assertEqual(self.card.current_status, CardStatus.COMPLETED)
        self.assertEqual(self.card.status_badge.text(), "Completed")

        # Failed
        self.card.set_status(AgentStatus.FAILED)
        self.assertEqual(self.card.current_status, CardStatus.FAILED)
        self.assertEqual(self.card.status_badge.text(), "Failed")

        # Cancelled
        self.card.set_status(WorkflowStatus.CANCELLED)
        self.assertEqual(self.card.current_status, CardStatus.CANCELLED)
        self.assertEqual(self.card.status_badge.text(), "Cancelled")

        # Skipped
        self.card.set_status("skipped")
        self.assertEqual(self.card.current_status, CardStatus.SKIPPED)
        self.assertEqual(self.card.status_badge.text(), "Skipped")

        # Back to Pending
        self.card.set_status(AgentStatus.IDLE)
        self.assertEqual(self.card.current_status, CardStatus.PENDING)
        self.assertEqual(self.card.status_badge.text(), "Pending")

    def test_provider_model_updates(self):
        """Card updates provider and model info dynamically."""
        self.card.set_provider_model("anthropic", "claude-3-5-sonnet")
        self.assertEqual(self.card.provider_model_label.text(), "anthropic / claude-3-5-sonnet")

        # Model only
        self.card.provider_id = ""
        self.card.set_provider_model("", "llama3-70b")
        self.assertEqual(self.card.provider_model_label.text(), "llama3-70b")

    def test_verified_and_unavailable_metrics(self):
        """Card displays formatted duration and verified tokens, and '--' when unavailable."""
        # Duration formatting
        self.card.set_duration(3.14159)
        self.assertEqual(self.card.duration_label.text(), "3.14s")

        self.card.set_duration(None)
        self.assertEqual(self.card.duration_label.text(), "--")

        # Verified tokens (never fabricate)
        self.card.set_tokens(None)
        self.assertEqual(self.card.tokens_label.text(), "--")

        self.card.set_tokens(total_tokens=2500, prompt_tokens=1000, completion_tokens=1500)
        self.assertEqual(self.card.tokens_label.text(), "2,500 tok")
        self.assertIn("Prompt: 1,000", self.card.tokens_label.toolTip())
        self.assertIn("Completion: 1,500", self.card.tokens_label.toolTip())

    def test_snapshot_telemetry_updates_without_fabricating_output(self):
        """TeamExecutionSnapshot updates telemetry metadata but does not alter textual output."""
        self.card.set_output("Original text")
        snap = TeamExecutionSnapshot(
            step_id="step_creator",
            agent_id="ag_creator",
            role=AgentRole.CREATOR,
            status=AgentStatus.RUNNING,
            started_at=10.0,
            duration_sec=1.5,
            provider_id="groq",
            model_id="llama3-8b",
            verified_total_tokens=512,
        )
        self.card.update_from_snapshot(snap)

        self.assertEqual(self.card.current_status, CardStatus.RUNNING)
        self.assertEqual(self.card.duration_label.text(), "1.50s")
        self.assertEqual(self.card.tokens_label.text(), "512 tok")
        self.assertEqual(self.card.provider_model_label.text(), "groq / llama3-8b")
        # Text output remains unchanged by telemetry
        self.assertEqual(self.card.output_viewer.toPlainText(), "Original text")

    def test_output_display_and_expand_collapse(self):
        """Output text is displayed in read-only viewer and can be collapsed/expanded."""
        self.card.set_output("def generated_code():\n    return 42\n")
        self.assertIn("def generated_code():", self.card.output_viewer.toPlainText())
        self.assertTrue(self.card.output_viewer.isReadOnly())

        # Collapse
        self.card.toggle_expanded()
        self.assertFalse(self.card.is_expanded())
        self.assertFalse(self.card.body_container.isVisible())
        self.assertEqual(self.card.expand_btn.text(), "▶")

        # Expand
        self.card.set_expanded(True)
        self.assertTrue(self.card.is_expanded())
        self.assertTrue(self.card.body_container.isVisible())
        self.assertEqual(self.card.expand_btn.text(), "▼")

    def test_historical_revision_selection(self):
        """Inspection of historical revisions updates displayed output, duration, tokens, and feedback."""
        rec0 = StepExecutionRecord(
            step_id="step_creator",
            agent_id="ag_creator",
            agent_role=AgentRole.CREATOR,
            status=AgentStatus.COMPLETED,
            started_at=100.0,
            finished_at=101.0,
            duration_sec=1.0,
            revision_iteration=0,
            result=AgentResult(
                agent_id="ag_creator",
                agent_role=AgentRole.CREATOR,
                task_id="t1",
                status=AgentStatus.COMPLETED,
                output="First draft version 0",
                metadata=AgentExecutionMetadata(
                    duration_sec=1.0,
                    provider_id="openai",
                    model_id="gpt-4o",
                    total_tokens=400,
                ),
            ),
            critic_review=CriticReviewResult(
                revision_required=True,
                reason="Lacks type annotations",
                feedback="Please add type hints to all functions",
            ),
        )

        rec1 = StepExecutionRecord(
            step_id="step_creator",
            agent_id="ag_creator",
            agent_role=AgentRole.CREATOR,
            status=AgentStatus.COMPLETED,
            started_at=102.0,
            finished_at=103.5,
            duration_sec=1.5,
            revision_iteration=1,
            result=AgentResult(
                agent_id="ag_creator",
                agent_role=AgentRole.CREATOR,
                task_id="t1",
                status=AgentStatus.COMPLETED,
                output="Second revised version 1",
                metadata=AgentExecutionMetadata(
                    duration_sec=1.5,
                    provider_id="openai",
                    model_id="gpt-4o",
                    total_tokens=650,
                ),
            ),
            critic_review=None,
        )

        selected_revisions = []
        self.card.revision_selected.connect(lambda idx: selected_revisions.append(idx))

        # Set records with latest output
        self.card.set_records([rec0, rec1], latest_output="Second revised version 1 (from step_outputs)")

        # Defaults to latest revision (index 1)
        self.assertTrue(self.card.revision_bar.isVisible())
        self.assertEqual(self.card.revision_combo.count(), 2)
        self.assertEqual(self.card.revision_combo.currentIndex(), 1)
        self.assertEqual(self.card.output_viewer.toPlainText(), "Second revised version 1 (from step_outputs)")
        self.assertEqual(self.card.duration_label.text(), "1.50s")
        self.assertEqual(self.card.tokens_label.text(), "650 tok")
        self.assertFalse(self.card.critic_box.isVisible())

        # Select earlier revision 0
        self.card.revision_combo.setCurrentIndex(0)
        self.assertEqual(selected_revisions, [0])
        self.assertEqual(self.card.output_viewer.toPlainText(), "First draft version 0")
        self.assertEqual(self.card.duration_label.text(), "1.00s")
        self.assertEqual(self.card.tokens_label.text(), "400 tok")
        self.assertTrue(self.card.critic_box.isVisible())
        self.assertIn("Lacks type annotations", self.card.critic_title.text())
        self.assertIn("Please add type hints", self.card.critic_text.text())

    def test_critic_feedback_presentation(self):
        """Critic feedback is displayed prominently when present and hidden when cleared."""
        self.card.set_critic_feedback(
            feedback="Fix division by zero on empty input.",
            reason="Unchecked ZeroDivisionError",
        )
        self.assertTrue(self.card.critic_box.isVisible())
        self.assertIn("Unchecked ZeroDivisionError", self.card.critic_title.text())
        self.assertIn("Fix division by zero", self.card.critic_text.text())

        # Clear feedback
        self.card.set_critic_feedback(None)
        self.assertFalse(self.card.critic_box.isVisible())

    def test_failure_error_presentation(self):
        """Execution failure displays error box with detailed error message."""
        self.card.set_error("Provider rate limit exceeded. Retry after 20s.")
        self.assertTrue(self.card.error_box.isVisible())
        self.assertIn("rate limit exceeded", self.card.error_text.text())

        self.card.set_error(None)
        self.assertFalse(self.card.error_box.isVisible())

    def test_dark_light_theme_switching(self):
        """Theme can be switched between dark and light modes cleanly without errors."""
        self.card.set_dark_theme(False)
        self.assertFalse(self.card.dark)

        self.card.set_dark_theme(True)
        self.assertTrue(self.card.dark)


class TestFinalDeliverablePanel(unittest.TestCase):
    """Test FinalDeliverablePanel Markdown rendering, attribution, states, metrics, and clipboard."""

    def setUp(self):
        self.panel = FinalDeliverablePanel(dark=True)
        self.panel.show()
        QCoreApplication.processEvents()

    def tearDown(self):
        self.panel.close()

    def test_empty_state_initialization(self):
        """Panel initializes in clean EMPTY state with copy button disabled."""
        self.assertEqual(self.panel.current_state, DeliverableState.EMPTY)
        self.assertEqual(self.panel.duration_label.text(), "--")
        self.assertEqual(self.panel.tokens_label.text(), "--")
        self.assertFalse(self.panel.copy_btn.isEnabled())
        self.assertFalse(self.panel.attribution_badge.isVisible())

    def test_loading_state_transition(self):
        """Panel transitions to LOADING state with custom progress message."""
        self.panel.set_loading("Finalizer agent is synthesizing response...")
        self.assertEqual(self.panel.current_state, DeliverableState.LOADING)
        self.assertIn("Finalizer agent is synthesizing", self.panel.loading_label.text())
        self.assertFalse(self.panel.copy_btn.isEnabled())

    def test_error_state_transition(self):
        """Panel transitions to ERROR state displaying execution failure summary."""
        self.panel.set_error("Team execution timed out after 300 seconds.")
        self.assertEqual(self.panel.current_state, DeliverableState.ERROR)
        self.assertIn("timed out after 300 seconds", self.panel.error_label.text())
        self.assertFalse(self.panel.copy_btn.isEnabled())

    def test_deliverable_markdown_rendering(self):
        """Panel renders Markdown content into rich HTML, displays metrics and attribution."""
        md_text = "# Executive Summary\n\nAI Team **completed** the mission.\n- Point 1\n- Point 2"
        self.panel.set_deliverable(
            content=md_text,
            attribution="Finalizer Agent",
            duration=4.82,
            tokens=1850,
        )

        self.assertEqual(self.panel.current_state, DeliverableState.READY)
        self.assertEqual(self.panel.raw_markdown, md_text)
        self.assertEqual(self.panel.duration_label.text(), "⏱ 4.82s")
        self.assertEqual(self.panel.tokens_label.text(), "🪙 1,850 tok")
        self.assertTrue(self.panel.attribution_badge.isVisible())
        self.assertIn("Finalizer Agent", self.panel.attribution_badge.text())
        self.assertTrue(self.panel.copy_btn.isEnabled())

        # Verify rendered HTML contains styled header and bold elements
        rendered_html = self.panel.markdown_viewer.toHtml()
        self.assertIn("Executive Summary", rendered_html)
        self.assertIn("completed", rendered_html)

    def test_update_from_result_attribution_finalizer(self):
        """update_from_result attributes final output to Finalizer Agent when present."""
        result = TeamExecutionResult(
            team_id="t_main",
            task_id="task_1",
            status=WorkflowStatus.COMPLETED,
            final_deliverable="# Final Output\nAll tasks done.",
            total_duration_sec=5.2,
            total_tokens=2200,
            step_records=[
                StepExecutionRecord(
                    step_id="step_c",
                    agent_id="ag_c",
                    agent_role=AgentRole.CREATOR,
                    status=AgentStatus.COMPLETED,
                    started_at=10.0,
                    duration_sec=2.0,
                ),
                StepExecutionRecord(
                    step_id="step_f",
                    agent_id="ag_f",
                    agent_role=AgentRole.FINALIZER,
                    status=AgentStatus.COMPLETED,
                    started_at=12.0,
                    duration_sec=3.2,
                ),
            ],
        )

        self.panel.update_from_result(result)
        self.assertEqual(self.panel.current_state, DeliverableState.READY)
        self.assertEqual(self.panel.attribution, "Finalizer Agent")
        self.assertEqual(self.panel.duration_label.text(), "⏱ 5.20s")
        self.assertEqual(self.panel.tokens_label.text(), "🪙 2,200 tok")

    def test_update_from_result_attribution_creator(self):
        """update_from_result attributes final output to Creator Agent when Finalizer was not run."""
        result = TeamExecutionResult(
            team_id="t_main",
            task_id="task_2",
            status=WorkflowStatus.COMPLETED,
            final_deliverable="# Creator Output\nDirect delivery.",
            total_duration_sec=2.1,
            total_tokens=900,
            step_records=[
                StepExecutionRecord(
                    step_id="step_c",
                    agent_id="ag_c",
                    agent_role=AgentRole.CREATOR,
                    status=AgentStatus.COMPLETED,
                    started_at=10.0,
                    duration_sec=2.1,
                ),
            ],
        )

        self.panel.update_from_result(result)
        self.assertEqual(self.panel.current_state, DeliverableState.READY)
        self.assertEqual(self.panel.attribution, "Creator Agent")

    def test_update_from_result_failure_and_empty(self):
        """update_from_result transitions to error on failed team execution without deliverable."""
        result = TeamExecutionResult(
            team_id="t_main",
            task_id="task_3",
            status=WorkflowStatus.FAILED,
            error_summary="All steps failed.",
            final_deliverable="",
        )
        self.panel.update_from_result(result)
        self.assertEqual(self.panel.current_state, DeliverableState.ERROR)
        self.assertIn("All steps failed", self.panel.error_label.text())

    def test_verified_and_unavailable_metrics(self):
        """Verified tokens display strictly verified counts and '--' when unavailable."""
        self.panel.set_deliverable(
            content="Simple text",
            attribution="Creator Agent",
            duration=None,
            tokens=None,
        )
        self.assertEqual(self.panel.duration_label.text(), "--")
        self.assertEqual(self.panel.tokens_label.text(), "--")

    def test_copy_to_clipboard(self):
        """Clicking copy copies raw deliverable Markdown to system clipboard."""
        raw_text = "### Important Instructions\n\n1. Step one\n2. Step two"
        self.panel.set_deliverable(content=raw_text, attribution="Creator Agent")

        self.panel.copy_to_clipboard()
        clipboard_text = QGuiApplication.clipboard().text()
        self.assertEqual(clipboard_text, raw_text)
        self.assertEqual(self.panel.copy_btn.text(), "✓ Copied!")

        # Restore
        self.panel._restore_copy_btn()
        self.assertEqual(self.panel.copy_btn.text(), "Copy Deliverable")

    def test_dark_light_theme_switching(self):
        """Switching theme dynamically re-renders Markdown in appropriate theme colors."""
        self.panel.set_deliverable(
            content="```python\ndef test(): pass\n```",
            attribution="Creator Agent",
        )
        # Switch to light
        self.panel.set_dark_theme(False)
        self.assertFalse(self.panel.dark)
        html_light = self.panel.markdown_viewer.toHtml()
        self.assertTrue(len(html_light) > 0)

        # Switch to dark
        self.panel.set_dark_theme(True)
        self.assertTrue(self.panel.dark)
        html_dark = self.panel.markdown_viewer.toHtml()
        self.assertTrue(len(html_dark) > 0)


class TestTeamConfigPanel(unittest.TestCase):
    """Test TeamConfigPanel agent controls, model caching, validation, and presets."""

    def setUp(self):
        self.mock_pm = MagicMock(spec=ProviderManager)
        # Setup mock providers
        mock_ollama = MagicMock()
        mock_ollama.provider_id = "ollama"
        mock_groq = MagicMock()
        mock_groq.provider_id = "groq"
        self.mock_pm.list_providers.return_value = [mock_ollama, mock_groq]

        self.panel = TeamConfigPanel(provider_manager=self.mock_pm, dark=True)
        self.panel.show()
        QCoreApplication.processEvents()

    def tearDown(self):
        self.panel.close()

    def test_default_configuration_and_mandatory_creator(self):
        """Panel loads default presets and locks mandatory Creator agent as enabled."""
        creator_row = self.panel.agent_rows[AgentRole.CREATOR]
        self.assertTrue(creator_row.is_agent_enabled())
        self.assertTrue(creator_row.enable_chk.isChecked())
        # Creator checkbox must be locked
        self.assertFalse(creator_row.enable_chk.isEnabled())

        # Other roles are enabled by default and unlockable
        for role in [AgentRole.PLANNER, AgentRole.RESEARCHER, AgentRole.CRITIC, AgentRole.FINALIZER]:
            row = self.panel.agent_rows[role]
            self.assertTrue(row.is_agent_enabled())
            self.assertTrue(row.enable_chk.isEnabled())

    def test_optional_agent_enable_disable_behavior(self):
        """Optional agents can be enabled and disabled, toggling their input controls."""
        critic_row = self.panel.agent_rows[AgentRole.CRITIC]
        self.assertTrue(critic_row.is_agent_enabled())

        # Disable Critic
        critic_row.set_agent_enabled(False)
        self.assertFalse(critic_row.is_agent_enabled())
        self.assertFalse(critic_row.provider_combo.isEnabled())
        self.assertFalse(critic_row.model_combo.isEnabled())
        self.assertFalse(critic_row.temp_spin.isEnabled())

        config = self.panel.get_team_config()
        self.assertFalse(config.members[AgentRole.CRITIC].enabled)

        # Re-enable Critic
        critic_row.set_agent_enabled(True)
        self.assertTrue(critic_row.is_agent_enabled())
        self.assertTrue(critic_row.provider_combo.isEnabled())

        config2 = self.panel.get_team_config()
        self.assertTrue(config2.members[AgentRole.CRITIC].enabled)

    def test_provider_and_model_configuration(self):
        """Modifying provider, model, temperature, timeout, and retries reflects in TeamConfig."""
        creator_row = self.panel.agent_rows[AgentRole.CREATOR]
        creator_row.provider_combo.setCurrentText("groq")
        creator_row.model_combo.addItem("llama3-70b")
        creator_row.model_combo.setCurrentText("llama3-70b")
        creator_row.temp_spin.setValue(0.85)
        creator_row.timeout_spin.setValue(120)
        creator_row.retries_spin.setValue(2)

        config = self.panel.get_team_config()
        creator_member = config.members[AgentRole.CREATOR]
        self.assertEqual(creator_member.provider_id, "groq")
        self.assertEqual(creator_member.model_id, "llama3-70b")
        self.assertAlmostEqual(creator_member.temperature, 0.85)
        self.assertEqual(creator_member.timeout_sec, 120.0)
        self.assertEqual(creator_member.max_retries, 2)

    def test_model_discovery_caching_and_non_blocking(self):
        """Cached models are populated immediately without launching background discovery."""
        self.panel._model_cache["groq"] = [
            ModelInfo(id="groq/model-a", name="Model A", provider_id="groq"),
            ModelInfo(id="groq/model-b", name="Model B", provider_id="groq"),
        ]

        creator_row = self.panel.agent_rows[AgentRole.CREATOR]
        self.panel._on_role_provider_changed(AgentRole.CREATOR, "groq")

        self.assertEqual(creator_row.model_combo.count(), 2)
        self.assertEqual(creator_row.model_combo.itemText(0), "groq/model-a")
        # No worker launched because cache exists
        self.assertNotIn("groq", self.panel._discovery_workers)

    def test_configuration_validation(self):
        """Validation passes for sound config and returns descriptive errors for invalid setups."""
        # Default config is valid
        self.assertIsNone(self.panel.validate_config())
        self.assertIn("✓ Configuration valid", self.panel.validation_status_lbl.text())

        # Set empty model for mandatory Creator
        creator_row = self.panel.agent_rows[AgentRole.CREATOR]
        creator_row.model_combo.clear()

        err = self.panel.validate_config()
        self.assertIsNotNone(err)
        self.assertIn("creator", err.lower())

    def test_workflow_settings_and_reset_to_defaults(self):
        """Workflow ceiling limits configure properly and reset-to-defaults restores baseline."""
        self.panel.global_timeout_spin.setValue(600)
        self.panel.max_revision_spin.setValue(4)
        self.panel.max_steps_spin.setValue(25)
        self.panel.token_budget_spin.setValue(50000)

        cfg = self.panel.get_team_config()
        self.assertEqual(cfg.timeout_sec, 600.0)
        self.assertEqual(cfg.max_revision_loops, 4)
        self.assertEqual(cfg.max_steps, 25)
        self.assertEqual(cfg.token_budget, 50000)

        # Reset to defaults
        self.panel.reset_to_defaults()
        reset_cfg = self.panel.get_team_config()
        self.assertEqual(reset_cfg.timeout_sec, 300.0)
        self.assertEqual(reset_cfg.max_revision_loops, 2)
        self.assertEqual(reset_cfg.max_steps, 15)
        self.assertIsNone(reset_cfg.token_budget)

    def test_dark_light_theme_switching(self):
        """Theme switching applies cleanly across the config panel."""
        self.panel.set_dark_theme(False)
        self.assertFalse(self.panel.dark)
        self.panel.set_dark_theme(True)
        self.assertTrue(self.panel.dark)


class TestAITeamsView(unittest.TestCase):
    """Test AITeamsView master workspace, worker execution lifecycle, history, and theme."""

    def setUp(self):
        self.mock_pm = MagicMock(spec=ProviderManager)
        mock_ollama = MagicMock()
        mock_ollama.provider_id = "ollama"
        mock_ollama.is_local = True
        self.mock_pm.list_providers.return_value = [mock_ollama]
        self.mock_pm.get_provider.return_value = mock_ollama

        self.view = AITeamsView(provider_manager=self.mock_pm, dark=True)
        self.view.show()
        QCoreApplication.processEvents()

    def tearDown(self):
        self.view.close()

    def test_initial_view_state(self):
        """View initializes with correct controls, empty task input, and idle status."""
        self.assertTrue(self.view.run_btn.isEnabled())
        self.assertFalse(self.view.cancel_btn.isEnabled())
        self.assertEqual(self.view.status_label.text(), "Idle")
        self.assertEqual(self.view.task_input.toPlainText(), "")
        self.assertEqual(self.view.history_list.count(), 0)

    def test_empty_task_input_validation(self):
        """Attempting to run with empty task input fails validation and does not launch worker."""
        self.view.task_input.setPlainText("   ")
        self.view.run_team()

        self.assertIn("cannot be empty", self.view.status_label.text())
        self.assertFalse(self.view._is_running)
        self.assertTrue(self.view.run_btn.isEnabled())

    def test_preflight_validation_failure(self):
        """Preflight readiness failures halt execution and show error in deliverable panel."""
        self.view.task_input.setPlainText("Write an API client")
        # Force preflight failure by clearing providers
        self.mock_pm.get_provider.return_value = None

        self.view.run_team()
        self.assertIn("Preflight", self.view.status_label.text())
        self.assertFalse(self.view._is_running)
        self.assertTrue(self.view.run_btn.isEnabled())
        self.assertEqual(self.view.deliverable_panel.current_state, DeliverableState.ERROR)

    def test_duplicate_execution_prevention(self):
        """While a run is active, secondary run_team calls are ignored."""
        self.view._is_running = True
        # Setting task text; run_team must return early without error
        self.view.task_input.setPlainText("Another task")
        self.view.run_team()
        # Remains in active state without crashing
        self.assertTrue(self.view._is_running)

    def test_live_observer_updates_and_revision_banner(self):
        """Live step updates and revision requests update cards and revision banner."""
        run_id = "test_run_123"
        self.view._current_run_id = run_id
        self.view._is_running = True

        # Initialize cards
        cfg = self.view.config_panel.get_team_config()
        self.view._initialize_step_cards(cfg)

        # 1. Step started
        snap_start = TeamExecutionSnapshot(
            step_id="custom_team_creator",
            agent_id="ag_c",
            role=AgentRole.CREATOR,
            status=AgentStatus.RUNNING,
            started_at=10.0,
        )
        self.view._on_step_started(snap_start, run_id=run_id)
        card = self.view._step_cards[AgentRole.CREATOR]
        self.assertEqual(card.current_status, CardStatus.RUNNING)
        self.assertIn("Creator Agent running", self.view.status_label.text())

        # 2. Step completed
        snap_comp = TeamExecutionSnapshot(
            step_id="custom_team_creator",
            agent_id="ag_c",
            role=AgentRole.CREATOR,
            status=AgentStatus.COMPLETED,
            started_at=10.0,
            finished_at=12.5,
            duration_sec=2.5,
            verified_total_tokens=320,
        )
        self.view._on_step_completed(snap_comp, run_id=run_id)
        self.assertEqual(card.current_status, CardStatus.COMPLETED)
        self.assertEqual(card.duration_label.text(), "2.50s")

        # 3. Revision requested
        self.assertFalse(self.view.revision_banner.isVisible())
        self.view._on_revision_requested(
            target_step_id="custom_team_creator",
            reason="Missing error handling",
            feedback="Add try/except blocks",
            loop_count=1,
            run_id=run_id,
        )
        self.assertTrue(self.view.revision_banner.isVisible())
        self.assertIn("Loop #1", self.view.revision_banner_title.text())
        self.assertIn("Missing error handling", self.view.revision_banner_text.text())

    def test_stale_session_event_handling(self):
        """Events carrying a stale or mismatched run_id are ignored."""
        self.view._current_run_id = "active_run_id"
        self.view._is_running = True

        cfg = self.view.config_panel.get_team_config()
        self.view._initialize_step_cards(cfg)

        card = self.view._step_cards[AgentRole.CREATOR]
        initial_status = card.current_status

        # Stale snapshot
        stale_snap = TeamExecutionSnapshot(
            step_id="custom_team_creator",
            agent_id="ag_c",
            role=AgentRole.CREATOR,
            status=AgentStatus.RUNNING,
            started_at=1.0,
        )
        self.view._on_step_started(stale_snap, run_id="old_stale_run")
        # Status must not change
        self.assertEqual(card.current_status, initial_status)

    def test_team_completed_and_run_history_storage(self):
        """Terminal result dispatches deliverable, updates cards, and stores run in history."""
        run_id = "run_comp_1"
        self.view._current_run_id = run_id
        self.view._is_running = True
        self.view.task_input.setPlainText("Generate a microservice")

        cfg = self.view.config_panel.get_team_config()
        self.view._initialize_step_cards(cfg)

        result = TeamExecutionResult(
            team_id="custom_team",
            task_id="task_microservice",
            status=WorkflowStatus.COMPLETED,
            final_deliverable="# Complete Architecture\nCode ready.",
            total_duration_sec=3.8,
            total_tokens=1400,
            revision_count=1,
            step_records=[
                StepExecutionRecord(
                    step_id="custom_team_creator",
                    agent_id="ag_c",
                    agent_role=AgentRole.CREATOR,
                    status=AgentStatus.COMPLETED,
                    started_at=10.0,
                    duration_sec=3.8,
                )
            ],
            step_outputs={"custom_team_creator": "Creator output code"},
        )

        self.view._on_team_completed(result, run_id=run_id)

        # UI restored
        self.assertFalse(self.view._is_running)
        self.assertTrue(self.view.run_btn.isEnabled())
        self.assertFalse(self.view.cancel_btn.isEnabled())
        self.assertIn("Completed", self.view.status_label.text())

        # Deliverable rendered
        self.assertEqual(self.view.deliverable_panel.current_state, DeliverableState.READY)
        self.assertEqual(self.view.deliverable_panel.raw_markdown, "# Complete Architecture\nCode ready.")

        # History updated
        self.assertEqual(len(self.view._run_history), 1)
        self.assertEqual(self.view.history_list.count(), 1)
        self.assertEqual(self.view._run_history[0].run_id, run_id)

    def test_history_item_selection(self):
        """Clicking an item in the history list displays previous deliverable without altering inputs."""
        self.view.task_input.setPlainText("Active current prompt")
        old_item = TeamRunHistoryItem(
            run_id="historical_run_99",
            task="Previous task objective",
            start_time="12:00:00",
            status="COMPLETED",
            duration_sec=4.2,
            final_deliverable="# Historical Output\nFrom previous run.",
            total_tokens=1100,
            attribution="Finalizer Agent",
        )
        self.view._run_history = [old_item]
        self.view._refresh_history_list()

        # Click the item
        list_item = self.view.history_list.item(0)
        self.view._on_history_item_clicked(list_item)

        # Previous deliverable shown
        self.assertEqual(self.view.deliverable_panel.raw_markdown, "# Historical Output\nFrom previous run.")
        self.assertEqual(self.view.deliverable_panel.attribution, "Finalizer Agent")
        # Current task input text was NOT overwritten
        self.assertEqual(self.view.task_input.toPlainText(), "Active current prompt")

    def test_cancellation_flow(self):
        """Cancellation requests cooperatively cancel worker and deliver CANCELLED state."""
        run_id = "run_cancel_1"
        self.view._current_run_id = run_id
        self.view._is_running = True
        self.view.cancel_btn.setEnabled(True)

        mock_worker = MagicMock(spec=TeamExecutionWorker)
        self.view._active_worker = mock_worker

        self.view.cancel_team()
        mock_worker.cancel.assert_called_once()
        self.assertEqual(self.view.status_label.text(), "Canceling...")

        # Worker signals cancelled
        self.view._on_worker_cancelled(run_id=run_id)
        self.assertEqual(self.view.status_label.text(), "Cancelled")

    def test_dark_light_theme_switching(self):
        """Theme switching toggles styles on main view and child panels."""
        self.view.set_dark_theme(False)
        self.assertFalse(self.view.dark)
        self.assertFalse(self.view.config_panel.dark)
        self.assertFalse(self.view.deliverable_panel.dark)

        self.view.set_dark_theme(True)
        self.assertTrue(self.view.dark)
        self.assertTrue(self.view.config_panel.dark)
        self.assertTrue(self.view.deliverable_panel.dark)

    def test_close_event_safe_worker_anchoring(self):
        """Closing view while worker is active safely anchors and cancels it."""
        mock_worker = MagicMock(spec=TeamExecutionWorker)
        mock_worker.isRunning.return_value = True
        self.view._active_worker = mock_worker

        self.view.close()
        mock_worker.cancel.assert_called_once()
        self.assertIn(mock_worker, _ORPHANED_TEAM_WORKERS)


class TestMainWindowTeamsIntegration(unittest.TestCase):
    """Test MainWindow integration of AITeamsView, navigation, shared provider manager, and theme."""

    def setUp(self):
        self.mock_pm = MagicMock(spec=ProviderManager)
        mock_ollama = MagicMock()
        mock_ollama.provider_id = "ollama"
        mock_ollama.display_name = "Ollama"
        mock_ollama.is_local = True
        mock_ollama.validate_connection.return_value = True
        mock_ollama.get_available_models.return_value = []
        self.mock_pm.list_providers.return_value = [mock_ollama]
        self.mock_pm.get_provider.return_value = mock_ollama
        self.mock_pm.get_active_provider.return_value = mock_ollama
        self.mock_pm.credential_store = MagicMock()
        self.mock_pm.credential_store.has_api_key.return_value = False

        self.window = MainWindow(provider_manager=self.mock_pm, auto_discover=False)
        self.window.show()
        QCoreApplication.processEvents()

    def tearDown(self):
        self.window.close()
        QCoreApplication.processEvents()

    def test_navigation_registration_and_page_selection(self):
        """AI Teams is registered as a first-class navigation item at index 2."""
        self.assertIsNotNone(self.window.teams_view)
        # Check nav buttons
        self.assertEqual(len(self.window.nav_buttons), len(self.window.nav_buttons))
        teams_btn = self.window.nav_buttons[2]
        self.assertIn("AI Teams", teams_btn.text())

        # Click AI Teams nav button
        teams_btn.click()
        QCoreApplication.processEvents()
        self.assertEqual(self.window.stack.currentIndex(), 2)
        self.assertEqual(self.window.stack.currentWidget(), self.window.teams_view.parentWidget())

    def test_shared_provider_manager_instance(self):
        """MainWindow shares its ProviderManager instance directly with AITeamsView."""
        self.assertIs(self.window.teams_view.provider_manager, self.window.provider_manager)
        self.assertIs(self.window.teams_view.config_panel.provider_manager, self.window.provider_manager)

    def test_existing_navigation_preservation(self):
        """All existing navigation pages remain functional and reachable."""
        # 0: Dashboard
        self.window.nav_buttons[0].click()
        self.assertEqual(self.window.stack.currentIndex(), 0)

        # 1: AI Chat
        self.window.nav_buttons[1].click()
        self.assertEqual(self.window.stack.currentIndex(), 1)

        # 2: AI Teams
        self.window.nav_buttons[2].click()
        self.assertEqual(self.window.stack.currentIndex(), 2)

        # 3: Model Manager
        self.window.nav_buttons[3].click()
        self.assertEqual(self.window.stack.currentIndex(), 3)

        # 4: Settings (last button)
        last_idx = len(self.window.nav_buttons) - 1
        self.window.nav_buttons[last_idx].click()
        self.assertEqual(self.window.stack.currentIndex(), last_idx)

    def test_repeated_page_switching_preserves_state(self):
        """Navigating away from AI Teams and returning preserves task input and configuration."""
        self.window.nav_buttons[2].click()
        self.window.teams_view.task_input.setPlainText("Preserved objective text")

        # Navigate away to AI Chat, then Model Manager
        self.window.nav_buttons[1].click()
        self.assertEqual(self.window.stack.currentIndex(), 1)
        self.window.nav_buttons[3].click()
        self.assertEqual(self.window.stack.currentIndex(), 3)

        # Return to AI Teams
        self.window.nav_buttons[2].click()
        self.assertEqual(self.window.stack.currentIndex(), 2)
        self.assertEqual(self.window.teams_view.task_input.toPlainText(), "Preserved objective text")

    def test_theme_propagation_to_teams_view(self):
        """Theme toggling in MainWindow propagates to AITeamsView and child components."""
        self.assertTrue(self.window.dark)
        self.assertTrue(self.window.teams_view.dark)

        # Toggle to Light
        self.window.toggle_theme()
        self.assertFalse(self.window.dark)
        self.assertFalse(self.window.teams_view.dark)
        self.assertFalse(self.window.teams_view.config_panel.dark)
        self.assertFalse(self.window.teams_view.deliverable_panel.dark)

        # Toggle back to Dark
        self.window.toggle_theme()
        self.assertTrue(self.window.dark)
        self.assertTrue(self.window.teams_view.dark)
        self.assertTrue(self.window.teams_view.config_panel.dark)
        self.assertTrue(self.window.teams_view.deliverable_panel.dark)

    def test_close_event_cooperative_worker_shutdown(self):
        """MainWindow closeEvent safely cancels active team worker and anchors it."""
        mock_worker = MagicMock(spec=TeamExecutionWorker)
        mock_worker.isRunning.return_value = True
        self.window.teams_view._active_worker = mock_worker

        self.window.close()
        mock_worker.cancel.assert_called_once()
        self.assertIn(mock_worker, _ORPHANED_TEAM_WORKERS)


if __name__ == "__main__":
    unittest.main()



