import os
import unittest
from unittest.mock import patch, MagicMock

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication
from src.core.message import Message, MessageRole
from src.core.events import StreamChunk
from src.providers.base import AIProvider
from src.providers.manager import ProviderManager
from src.providers.models import ModelInfo
from src.providers.exceptions import ProviderConnectionError
from src.ui.main_window import MainWindow

app = QApplication.instance() or QApplication([])


class MockCloudProvider(AIProvider):
    provider_id = "mockcloud"
    display_name = "Mock Cloud Provider"
    is_local = False

    def generate(self, messages, model, **kwargs):
        return "Generated cloud text"

    def stream(self, messages, model, **kwargs):
        yield StreamChunk(delta="Cloud ")
        yield StreamChunk(delta="stream ", is_final=False)
        yield StreamChunk(delta="done!", is_final=True)

    def validate_connection(self):
        return True

    def get_available_models(self):
        return [
            ModelInfo(id="cloud-fast", name="Cloud Fast", provider_id="mockcloud"),
            ModelInfo(id="cloud-smart", name="Cloud Smart", provider_id="mockcloud"),
        ]


class TestManagerIntegration(unittest.TestCase):
    def setUp(self):
        self.mgr = ProviderManager()
        self.mock_provider = MockCloudProvider()
        self.mgr.register_provider(self.mock_provider)
        self.mgr.set_active_provider("mockcloud")
        self.window = MainWindow(provider_manager=self.mgr)
        self.window.show()
        self.window.stack.setCurrentIndex(1)
        app.processEvents()

    def tearDown(self):
        self.window.close()

    def test_dynamic_models_populated(self):
        # Combo box should populate with active provider's models
        items = [self.window.model_combo.itemText(i) for i in range(self.window.model_combo.count())]
        self.assertIn("cloud-fast", items)
        self.assertIn("cloud-smart", items)

    def test_streaming_through_provider_manager(self):
        self.window.prompt_input.setText("Tell me something")
        self.window.send_prompt()

        if self.window.worker:
            self.window.worker.wait(5000)
        app.processEvents()

        # Check messages stored as Message instances
        self.assertEqual(len(self.window.messages), 2)
        self.assertIsInstance(self.window.messages[0], Message)
        self.assertIsInstance(self.window.messages[1], Message)
        self.assertEqual(self.window.messages[0].role, MessageRole.USER)
        self.assertEqual(self.window.messages[1].role, MessageRole.ASSISTANT)
        self.assertEqual(self.window.messages[1].content, "Cloud stream done!")

    def test_provider_error_handling(self):
        failing_provider = MockCloudProvider()
        failing_provider.provider_id = "failing"
        failing_provider.stream = MagicMock(side_effect=ProviderConnectionError("API server down", provider_id="failing"))

        self.mgr.register_provider(failing_provider)
        self.mgr.set_active_provider("failing")

        self.window.prompt_input.setText("Will fail")
        self.window.send_prompt()

        if self.window.worker:
            self.window.worker.wait(5000)
        app.processEvents()

        self.assertEqual(len(self.window.messages), 0)
        self.assertIn("API server down", self.window.transcript.toPlainText())
        self.assertTrue(self.window.send_button.isEnabled())


if __name__ == "__main__":
    unittest.main()

