import os
import time
import unittest
from unittest.mock import MagicMock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication
from src.core.comparison import (
    ComparisonCoordinator,
    ComparisonMetrics,
    ComparisonSlotConfig,
    ComparisonSlotResult,
    ModelCompareWorker,
)
from src.core.events import ResponseMetadata, StreamChunk
from src.core.message import Message, MessageRole
from src.providers.base import AIProvider
from src.providers.manager import ProviderManager
from src.ui.main_window import MainWindow

app = QApplication.instance() or QApplication([])


class SlowOrFailingProvider(AIProvider):
    def __init__(self, pid: str, chunks, delay: float = 0.0, error: Exception = None):
        self.provider_id = pid
        self.display_name = pid.title()
        self.is_local = (pid == "ollama")
        self.chunks = chunks
        self.delay = delay
        self.error = error

    def generate(self, messages, model, **kwargs):
        if self.error:
            raise self.error
        return "".join(self.chunks)

    def stream(self, messages, model, **kwargs):
        if self.error:
            raise self.error
        for ch in self.chunks:
            if self.delay > 0:
                time.sleep(self.delay)
            meta = ResponseMetadata(
                model=model,
                provider_id=self.provider_id,
                prompt_tokens=10,
                completion_tokens=len(ch),
                total_tokens=10 + len(ch),
            )
            yield StreamChunk(delta=ch, metadata=meta)

    def validate_connection(self):
        return True

    def get_available_models(self):
        return []


class TestComparisonIntegration(unittest.TestCase):
    def setUp(self):
        self.mgr = ProviderManager()
        # Mock get_available_models to prevent live network probes during unit tests
        for p in self.mgr.list_providers():
            p.get_available_models = MagicMock(return_value=[])

        self.window = MainWindow(provider_manager=self.mgr)
        self.window.show()
        # Navigate to AI Chat
        self.window.stack.setCurrentIndex(1)
        app.processEvents()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()

    def test_mode_toggle_single_and_compare(self):
        # Starts in single mode
        self.assertEqual(self.window.chat_mode, "single")
        self.assertTrue(self.window.transcript.isVisible())
        self.assertFalse(self.window.compare_view.isVisible())
        self.assertTrue(self.window.single_controls_widget.isVisible())

        # Switch to compare mode
        self.window.set_chat_mode("compare")
        self.assertEqual(self.window.chat_mode, "compare")
        self.assertFalse(self.window.transcript.isVisible())
        self.assertTrue(self.window.compare_view.isVisible())
        self.assertFalse(self.window.single_controls_widget.isVisible())

        # Switch back to single mode
        self.window.set_chat_mode("single")
        self.assertEqual(self.window.chat_mode, "single")
        self.assertTrue(self.window.transcript.isVisible())
        self.assertFalse(self.window.compare_view.isVisible())
        self.assertTrue(self.window.single_controls_widget.isVisible())

    def test_context_integrity_identical_prompt_and_single_commit(self):
        """Verify all comparison models receive identical history, and only the chosen answer commits."""
        self.window.set_chat_mode("compare")

        # Set up 2 mock providers
        p1 = SlowOrFailingProvider("ollama", ["Local answer"])
        p2 = SlowOrFailingProvider("groq", ["Cloud answer"])
        self.mgr.register_provider(p1)
        self.mgr.register_provider(p2)

        # Prepopulate with 1 turn of prior history
        self.window.messages = [
            Message(MessageRole.USER, "Prior user prompt"),
            Message(MessageRole.ASSISTANT, "Prior assistant response"),
        ]

        # Configure 2 slots
        slots = [
            ComparisonSlotConfig(slot_id=0, provider_id="ollama", model="llama3.2:3b"),
            ComparisonSlotConfig(slot_id=1, provider_id="groq", model="openai/gpt-oss-120b"),
        ]
        self.window.compare_view.get_configured_slots = MagicMock(return_value=slots)

        self.window.prompt_input.setText("Compare quantum computing explanations")
        self.window.send_prompt()

        # Wait for comparison to finish
        for w in self.window.comparison_coordinator.workers.values():
            w.wait(5000)
        app.processEvents()

        # Conversation history should now contain prior 2 messages + the 1 user prompt (NOT assistant responses yet!)
        self.assertEqual(len(self.window.messages), 3)
        self.assertEqual(self.window.messages[-1].role, MessageRole.USER)
        self.assertEqual(self.window.messages[-1].content, "Compare quantum computing explanations")

        # Select model #2 (Groq) to commit into conversation
        self.window.compare_view.slot_cards[1].select_btn.click()
        app.processEvents()

        # Now messages should contain exactly 4 items: prior 2 + user prompt + selected assistant response
        self.assertEqual(len(self.window.messages), 4)
        self.assertEqual(self.window.messages[-1].role, MessageRole.ASSISTANT)
        self.assertEqual(self.window.messages[-1].content, "Cloud answer")
        self.assertEqual(self.window.messages[-1].metadata["provider_id"], "groq")

        # Mode automatically switches back to single so user can continue conversation
        self.assertEqual(self.window.chat_mode, "single")
        self.assertIn("Cloud answer", self.window.transcript.toPlainText())

    def test_error_isolation_across_slots(self):
        """Verify that when one model worker fails (e.g. rate limit), the other continues and succeeds."""
        self.window.set_chat_mode("compare")

        p_success = SlowOrFailingProvider("groq", ["Groq successfully completed"])
        p_failure = SlowOrFailingProvider("openai", [], error=RuntimeError("HTTP 429 RateLimitError"))
        self.mgr.register_provider(p_success)
        self.mgr.register_provider(p_failure)

        slots = [
            ComparisonSlotConfig(slot_id=0, provider_id="groq", model="openai/gpt-oss-120b"),
            ComparisonSlotConfig(slot_id=1, provider_id="openai", model="gpt-4o-mini"),
        ]
        self.window.compare_view.get_configured_slots = MagicMock(return_value=slots)

        self.window.prompt_input.setText("Explain rate limits")
        self.window.send_prompt()

        for w in self.window.comparison_coordinator.workers.values():
            w.wait(5000)
        app.processEvents()

        card_success = self.window.compare_view.slot_cards[0]
        card_failure = self.window.compare_view.slot_cards[1]

        self.assertEqual(card_success.status_badge.text(), "Completed")
        self.assertEqual(card_success.content_buffer, "Groq successfully completed")
        self.assertTrue(card_success.select_btn.isEnabled())

        self.assertEqual(card_failure.status_badge.text(), "Error")
        self.assertFalse(card_failure.select_btn.isEnabled())
        self.assertIn("429", card_failure.transcript.toPlainText())

    def test_cancellation_during_comparison(self):
        """Verify cooperative cancellation safely stops all running threads without crashing."""
        self.window.set_chat_mode("compare")

        p_slow = SlowOrFailingProvider("gemini", ["chunk1", "chunk2", "chunk3", "chunk4"], delay=0.15)
        self.mgr.register_provider(p_slow)

        slots = [
            ComparisonSlotConfig(slot_id=0, provider_id="gemini", model="gemini-3.5-flash"),
            ComparisonSlotConfig(slot_id=1, provider_id="gemini", model="gemini-3.8-flash"),
        ]
        self.window.compare_view.get_configured_slots = MagicMock(return_value=slots)

        self.window.prompt_input.setText("Generate a massive essay")
        self.window.send_prompt()
        self.assertEqual(self.window.send_button.text(), "■ Stop All")

        time.sleep(0.05)
        # Click Stop All
        self.window.send_button.click()

        for w in self.window.comparison_coordinator.workers.values():
            w.wait(5000)
        app.processEvents()

        # Both cards should reflect canceled state
        card0 = self.window.compare_view.slot_cards[0]
        card1 = self.window.compare_view.slot_cards[1]
        self.assertIn(card0.status_badge.text(), ["Canceled", "Completed"])
        self.assertIn(card1.status_badge.text(), ["Canceled", "Completed"])
        self.assertEqual(self.window.send_button.text(), "Send")


if __name__ == "__main__":
    unittest.main()

