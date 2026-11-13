import os
import unittest
from unittest.mock import MagicMock

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication
from src.core.comparison import ComparisonMetrics, ComparisonSlotResult
from src.providers.manager import ProviderManager
from src.ui.compare_view import ComparisonView, ModelSlotCard

app = QApplication.instance() or QApplication([])


class TestCompareView(unittest.TestCase):
    def setUp(self):
        self.mgr = ProviderManager()
        # Mock get_available_models to avoid live network calls during UI unit tests
        for p in self.mgr.list_providers():
            p.get_available_models = MagicMock(return_value=[])
        self.view = ComparisonView(provider_manager=self.mgr, dark=True)

    def tearDown(self):
        self.view.deleteLater()

    def test_initial_two_slots(self):
        self.assertEqual(len(self.view.slot_cards), 2)
        self.assertEqual(len(self.view.slot_selectors), 2)
        configs = self.view.get_configured_slots()
        self.assertEqual(len(configs), 2)
        # Cannot delete when count is 2
        for info in self.view.slot_selectors.values():
            self.assertFalse(info["del_btn"].isEnabled())

    def test_add_and_remove_slots_responsive_grid(self):
        # Add 3rd slot
        self.view.add_model_btn.click()
        self.assertEqual(len(self.view.slot_cards), 3)
        # Now delete should be enabled
        for info in self.view.slot_selectors.values():
            self.assertTrue(info["del_btn"].isEnabled())

        # Add 4th slot
        self.view.add_model_btn.click()
        self.assertEqual(len(self.view.slot_cards), 4)
        self.assertFalse(self.view.add_model_btn.isEnabled(), "Should not allow adding > 4 models")

        # Remove 4th slot
        keys = list(self.view.slot_selectors.keys())
        self.view._remove_slot(keys[-1])
        self.assertEqual(len(self.view.slot_cards), 3)
        self.assertTrue(self.view.add_model_btn.isEnabled())

    def test_slot_card_streaming_and_completed(self):
        card = self.view.slot_cards[0]
        card.set_model_info("groq", "openai/gpt-oss-120b")
        card.reset_for_stream()
        self.assertEqual(card.status_badge.text(), "Thinking…")

        card.append_chunk("Hello ")
        card.append_chunk("world!")
        self.assertEqual(card.content_buffer, "Hello world!")
        self.assertEqual(card.status_badge.text(), "Streaming…")

        result = ComparisonSlotResult(
            slot_id=0,
            provider_id="groq",
            model="openai/gpt-oss-120b",
            content="Hello world!",
            is_success=True,
            metrics=ComparisonMetrics(
                elapsed_sec=0.85,
                total_tokens=25,
                tokens_per_sec=29.4,
                estimated_cost_usd=0.00004,
            ),
        )
        card.set_completed(result)
        self.assertEqual(card.status_badge.text(), "Completed")
        self.assertTrue(card.select_btn.isEnabled())
        self.assertTrue(card.copy_btn.isEnabled())
        self.assertIn("0.85s", card.metrics_bar.text())
        self.assertIn("29.4 t/s", card.metrics_bar.text())

    def test_slot_card_error_state(self):
        card = self.view.slot_cards[1]
        result = ComparisonSlotResult(
            slot_id=1,
            provider_id="openai",
            model="gpt-4o-mini",
            is_success=False,
            error_message="Insufficient credits remaining (429)",
            metrics=ComparisonMetrics(elapsed_sec=0.4),
        )
        card.set_completed(result)
        self.assertEqual(card.status_badge.text(), "Error")
        self.assertFalse(card.select_btn.isEnabled())
        self.assertIn("Insufficient credits", card.transcript.toPlainText())

    def test_comparison_summary(self):
        r1 = ComparisonSlotResult(
            slot_id=0,
            provider_id="ollama",
            model="llama3.2:3b",
            content="Fast local reply",
            is_success=True,
            metrics=ComparisonMetrics(elapsed_sec=1.1, total_tokens=20, tokens_per_sec=18.2),
        )
        r2 = ComparisonSlotResult(
            slot_id=1,
            provider_id="groq",
            model="openai/gpt-oss-120b",
            content="Fast cloud reply",
            is_success=True,
            metrics=ComparisonMetrics(elapsed_sec=0.55, total_tokens=25, tokens_per_sec=45.5),
        )
        self.view.show_comparison_summary([r1, r2])
        self.assertFalse(self.view.summary_banner.isHidden())
        self.assertIn("Fastest model:", self.view.summary_text.text())
        self.assertIn("Groq", self.view.summary_text.text())


if __name__ == "__main__":
    unittest.main()
