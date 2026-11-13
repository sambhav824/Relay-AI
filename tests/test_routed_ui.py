import os
import unittest
from unittest.mock import MagicMock, patch

# Run Qt offscreen
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication

from src.core.events import StreamChunk
from src.core.message import Message, MessageRole
from src.core.security import CredentialStore
from src.providers.base import AIProvider
from src.providers.manager import ProviderManager
from src.providers.models import ModelInfo
from src.routing.types import (
    ClassificationResult,
    NoEligibleModelsError,
    RoutingCandidate,
    RoutingDecision,
    RoutingExplanation,
    RoutingStrategy,
    TaskType,
)
from src.ui.main_window import MainWindow
from src.ui.routing_pill import RoutingStatusPill

app = QApplication.instance() or QApplication([])


class MockStreamProvider(AIProvider):
    """Mock AIProvider for testing UI integration without network requests."""

    def __init__(
        self,
        provider_id: str,
        stream_chunks=None,
        exception_to_raise=None,
    ):
        self.provider_id = provider_id
        self.display_name = provider_id.title()
        self.is_local = (provider_id == "ollama")
        self.stream_chunks = stream_chunks or []
        self.exception_to_raise = exception_to_raise
        self.recorded_calls = []

    def stream(self, messages, model, **kwargs):
        self.recorded_calls.append({
            "messages": [Message(m.role, m.content) for m in messages],
            "model": model,
        })
        if self.exception_to_raise:
            raise self.exception_to_raise
        for chunk in self.stream_chunks:
            yield chunk

    def generate(self, messages, model, **kwargs):
        return "".join(c.delta for c in self.stream_chunks)

    def validate_connection(self):
        return True

    def get_available_models(self):
        return [ModelInfo(f"{self.provider_id}-model", f"{self.display_name} Model", self.provider_id)]


def make_dummy_decision(
    primary_prov="groq",
    primary_model="openai/gpt-oss-120b",
    strategy=RoutingStrategy.BALANCED,
    task_type=TaskType.CODING,
):
    classification = ClassificationResult(
        task_type=task_type,
        confidence=0.92,
        matched_cues=["def", "python", "quicksort"],
    )
    primary = RoutingCandidate(
        provider_id=primary_prov,
        model_id=primary_model,
        display_name=f"{primary_model} ({primary_prov.title()})",
        composite_score=0.95,
        task_alignment_score=9.5,
    )
    explanation = RoutingExplanation(
        selected_model_id=primary_model,
        selected_provider_id=primary_prov,
        detected_task=task_type,
        confidence=0.92,
        strategy=strategy,
        reason="Optimal composite score",
        pricing_summary="Input: $0.15/M, Output: $0.60/M",
        fallback_candidates=["OpenAI (gpt-4.1)"],
    )
    return RoutingDecision(
        primary=primary,
        fallback_chain=[],
        classification=classification,
        strategy=strategy,
        explanation=explanation,
    )


class TestRoutedUI(unittest.TestCase):
    def setUp(self):
        self.mock_cred_store = MagicMock(spec=CredentialStore)
        self.provider_mgr = ProviderManager(credential_store=self.mock_cred_store)
        self.provider_mgr._providers.clear()

        self.p_groq = MockStreamProvider("groq", [StreamChunk("Routed "), StreamChunk("response.")])
        self.p_ollama = MockStreamProvider("ollama", [StreamChunk("Local "), StreamChunk("response.")])
        self.provider_mgr.register_provider(self.p_groq)
        self.provider_mgr.register_provider(self.p_ollama)

        self.window = MainWindow(provider_manager=self.provider_mgr, auto_discover=False)

    def tearDown(self):
        self.window.close()

    def test_initial_mode_is_single(self):
        self.assertEqual(self.window.chat_mode, "single")
        self.assertFalse(self.window.single_controls_widget.isHidden())
        self.assertTrue(self.window.routed_controls_widget.isHidden())
        self.assertFalse(self.window.transcript.isHidden())
        self.assertTrue(self.window.compare_view.isHidden())

    def test_switch_to_routed_mode(self):
        self.window.set_chat_mode("routed")
        self.assertEqual(self.window.chat_mode, "routed")
        self.assertTrue(self.window.single_controls_widget.isHidden())
        self.assertFalse(self.window.routed_controls_widget.isHidden())
        self.assertFalse(self.window.transcript.isHidden())
        self.assertTrue(self.window.compare_view.isHidden())

    def test_switch_to_compare_mode(self):
        self.window.set_chat_mode("compare")
        self.assertEqual(self.window.chat_mode, "compare")
        self.assertTrue(self.window.single_controls_widget.isHidden())
        self.assertTrue(self.window.routed_controls_widget.isHidden())
        self.assertTrue(self.window.transcript.isHidden())
        self.assertFalse(self.window.compare_view.isHidden())

    def test_strategy_combo_options(self):
        strategies_in_combo = [
            self.window.strategy_combo.itemData(i)
            for i in range(self.window.strategy_combo.count())
        ]
        self.assertIn(RoutingStrategy.BALANCED, strategies_in_combo)
        self.assertIn(RoutingStrategy.QUALITY_FIRST, strategies_in_combo)
        self.assertIn(RoutingStrategy.COST_EFFICIENT, strategies_in_combo)
        self.assertIn(RoutingStrategy.LOCAL_FIRST, strategies_in_combo)

    def test_routing_pill_states(self):
        pill = self.window.routing_pill
        self.assertEqual(pill.icon_label.text(), "✦")
        self.assertIn("Auto-Route Ready", pill.text_label.text())

        # Set decision
        decision = make_dummy_decision()
        pill.set_decision(decision)
        self.assertIn("Coding", pill.text_label.text())
        self.assertIn("Groq", pill.text_label.text())
        self.assertIn("92%", pill.text_label.text())
        self.assertIn("Pricing:", pill.toolTip())

        # Set fallback
        pill.set_fallback("groq", "openai/gpt-oss-120b", "openai", "gpt-4.1", 1)
        self.assertEqual(pill.icon_label.text(), "⚠️")
        self.assertIn("Fallback 1", pill.text_label.text())
        self.assertIn("gpt-4.1", pill.text_label.text())

        # Set error
        pill.set_error("No eligible models found")
        self.assertEqual(pill.icon_label.text(), "✕")
        self.assertIn("Routing Error", pill.text_label.text())

        # Reset
        pill.reset()
        self.assertEqual(pill.icon_label.text(), "✦")
        self.assertIn("Auto-Route Ready", pill.text_label.text())

    def test_routed_prompt_non_duplication(self):
        self.window.set_chat_mode("routed")
        dummy_decision = make_dummy_decision(primary_prov="groq", primary_model="groq-model")

        with patch.object(self.window.model_router, "route", return_value=dummy_decision):
            self.window.prompt_input.setText("Write a quicksort in Python")
            self.window.send_prompt()

            # Wait for worker thread to finish
            if self.window.routed_worker:
                self.window.routed_worker.wait(2000)
            app.processEvents()

        # Check messages: exactly 1 user and 1 assistant
        self.assertEqual(len(self.window.messages), 2)
        self.assertEqual(self.window.messages[0].role, MessageRole.USER)
        self.assertEqual(self.window.messages[0].content, "Write a quicksort in Python")
        self.assertEqual(self.window.messages[1].role, MessageRole.ASSISTANT)
        self.assertEqual(self.window.messages[1].content, "Routed response.")
        self.assertEqual(self.window.messages[1].metadata.get("provider_id"), "groq")
        self.assertEqual(self.window.messages[1].metadata.get("model"), "groq-model")

    def test_routed_failure_retains_user_prompt(self):
        self.window.set_chat_mode("routed")
        dummy_decision = make_dummy_decision(primary_prov="groq", primary_model="groq-model")

        # Provider fails
        self.p_groq.exception_to_raise = RuntimeError("Provider down")

        with patch.object(self.window.model_router, "route", return_value=dummy_decision):
            self.window.prompt_input.setText("Will fail gracefully")
            self.window.send_prompt()

            if self.window.routed_worker:
                self.window.routed_worker.wait(2000)
            app.processEvents()

        # User prompt MUST be preserved, and no assistant message committed
        self.assertEqual(len(self.window.messages), 1)
        self.assertEqual(self.window.messages[0].role, MessageRole.USER)
        self.assertEqual(self.window.messages[0].content, "Will fail gracefully")
        self.assertIn("Error:", self.window.transcript.toHtml())

    def test_preflight_no_eligible_models_error(self):
        self.window.set_chat_mode("routed")

        with patch.object(self.window.model_router, "route", side_effect=NoEligibleModelsError("No models available")):
            self.window.prompt_input.setText("Preflight test")
            self.window.send_prompt()
            app.processEvents()

        # User prompt retained, preflight error rendered
        self.assertEqual(len(self.window.messages), 1)
        self.assertEqual(self.window.messages[0].content, "Preflight test")
        self.assertIn("No models available", self.window.transcript.toHtml())
        self.assertEqual(self.window.send_button.text(), "Send")
        self.assertTrue(self.window.mode_frame.isEnabled())

    def test_mode_switching_locked_during_generation(self):
        self.window.set_chat_mode("routed")
        with patch.object(self.window, "_is_generation_active", return_value=True):
            self.window.set_chat_mode("single")
            # Mode switch rejected
            self.assertEqual(self.window.chat_mode, "routed")

    def test_stale_signal_rejection(self):
        self.window.set_chat_mode("routed")
        initial_sid = self.window.current_session_id

        # Deliver a chunk with a stale session id
        self.window._on_chunk("Stale token", sid=initial_sid - 1)
        self.assertEqual(self.window.assistant_buffer, "")

        # Deliver success with stale session id
        self.window._on_routed_final_success("groq", "groq-model", "Stale answer", sid=initial_sid - 1)
        self.assertEqual(len(self.window.messages), 0)

    def test_local_only_checkbox_passed_to_router(self):
        self.window.set_chat_mode("routed")
        dummy_decision = make_dummy_decision(primary_prov="ollama", primary_model="ollama-model")
        captured_kwargs = {}

        def mock_route(**kwargs):
            captured_kwargs.update(kwargs)
            return dummy_decision

        self.window.local_only_checkbox.setChecked(True)
        with patch.object(self.window.model_router, "route", side_effect=mock_route):
            self.window.prompt_input.setText("Test local-only routing")
            self.window.send_prompt()
            if self.window.routed_worker:
                self.window.routed_worker.wait(2000)
            app.processEvents()

        self.assertTrue(captured_kwargs.get("local_only"))

    def test_strategy_selection_passed_to_router(self):
        self.window.set_chat_mode("routed")
        dummy_decision = make_dummy_decision()
        captured_kwargs = {}

        def mock_route(**kwargs):
            captured_kwargs.update(kwargs)
            return dummy_decision

        idx = self.window.strategy_combo.findData(RoutingStrategy.QUALITY_FIRST)
        self.window.strategy_combo.setCurrentIndex(idx)

        with patch.object(self.window.model_router, "route", side_effect=mock_route):
            self.window.prompt_input.setText("Test quality first")
            self.window.send_prompt()
            if self.window.routed_worker:
                self.window.routed_worker.wait(2000)
            app.processEvents()

        self.assertEqual(captured_kwargs.get("strategy"), RoutingStrategy.QUALITY_FIRST)

    def test_routed_cancellation_retains_user_prompt(self):
        self.window.set_chat_mode("routed")
        dummy_decision = make_dummy_decision()

        # Simulate cancellation during stream
        with patch.object(self.window.model_router, "route", return_value=dummy_decision):
            self.window.prompt_input.setText("Cancel me")
            self.window.send_prompt()
            # Immediately trigger cancellation
            self.window.cancel_generation()
            if self.window.routed_worker:
                self.window.routed_worker.wait(2000)
            app.processEvents()

        # User prompt retained, no assistant message
        self.assertEqual(len(self.window.messages), 1)
        self.assertEqual(self.window.messages[0].content, "Cancel me")
        self.assertEqual(self.window.send_button.text(), "Send")

    def test_close_event_cooperative_shutdown(self):
        self.window.set_chat_mode("routed")
        dummy_decision = make_dummy_decision()

        with patch.object(self.window.model_router, "route", return_value=dummy_decision):
            self.window.prompt_input.setText("Running task")
            self.window.send_prompt()
            # Window close while worker is active
            mock_event = MagicMock()
            self.window.closeEvent(mock_event)
            mock_event.accept.assert_called_once()
            self.assertTrue(self.window._is_shutting_down)

    def test_close_event_anchors_still_running_worker(self):
        from src.ui.main_window import _ORPHANED_WORKERS
        mock_worker = MagicMock(spec=QThread)
        mock_worker.isRunning.return_value = True

        self.window._active_workers.add(mock_worker)
        mock_event = MagicMock()
        self.window.closeEvent(mock_event)

        mock_event.accept.assert_called_once()
        self.assertIn(mock_worker, _ORPHANED_WORKERS)
        # Clean up mock from set
        _ORPHANED_WORKERS.discard(mock_worker)

    def test_no_qthread_terminate_in_source(self):
        main_window_path = os.path.join(os.path.dirname(__file__), "..", "src", "ui", "main_window.py")
        worker_path = os.path.join(os.path.dirname(__file__), "..", "src", "routing", "worker.py")

        for path in [main_window_path, worker_path]:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertNotIn(".terminate()", content, f"Found .terminate() call in {path}")


if __name__ == "__main__":
    unittest.main()
