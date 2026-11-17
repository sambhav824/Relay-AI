import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication
from src.core.message import MessageRole
from src.providers.manager import ProviderManager
from src.ui.main_window import MainWindow
from src.ui.attachment_menu import AttachmentMenu, CameraCaptureDialog

app = QApplication.instance() or QApplication([])


class TestChatAttachmentsAndVoiceUI(unittest.TestCase):
    def setUp(self):
        self.mgr = ProviderManager()
        for p in self.mgr.list_providers():
            p.get_available_models = MagicMock(return_value=[])
            p.validate_connection = MagicMock(return_value=True)

        self.window = MainWindow(provider_manager=self.mgr, auto_discover=False)

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()

    def test_ui_components_exist(self):
        """Verify modern capsule controls match user specification."""
        self.assertIsNotNone(self.window.btn_attach)
        self.assertEqual(self.window.btn_attach.text(), "+")

        self.assertIsNotNone(self.window.prompt_input)

        # Think section removed as requested
        self.assertIsNone(self.window.btn_think)

        # Dedicated mic button for speech-to-text dictation only
        self.assertIsNotNone(self.window.btn_mic)
        self.assertFalse(self.window.btn_mic.icon().isNull())
        self.assertIn("Speech-to-Text", self.window.btn_mic.toolTip())

        # Conversational two-way voice button removed as requested
        self.assertIsNone(self.window.btn_voice)

        self.assertIsNotNone(self.window.send_button)
        self.assertIsNotNone(self.window.disclaimer_label)
        self.assertIn("Relay AI can make mistakes", self.window.disclaimer_label.text())

    def test_disclaimer_cycling(self):
        """Verify clicking disclaimer cycles through alternative wordings."""
        initial_text = self.window.disclaimer_label.text()
        self.window._cycle_disclaimer()
        second_text = self.window.disclaimer_label.text()
        self.assertNotEqual(initial_text, second_text)

    def test_attachment_menu_popup(self):
        """Verify clicking '+' shows AttachmentMenu with Camera, Photos, Files."""
        self.window.btn_attach.click()
        self.assertIsNotNone(self.window._attachment_menu)
        menu = self.window._attachment_menu
        self.assertIsInstance(menu, AttachmentMenu)
        self.assertEqual(menu.item_camera.label.text(), "Camera")
        self.assertEqual(menu.item_photos.label.text(), "Photos")
        self.assertEqual(menu.item_files.label.text(), "Files")

    def test_attachment_lifecycle(self):
        """Verify adding and removing attachments."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as f:
            f.write("Test document content for Relay AI RAG and context.")
            temp_path = f.name

        try:
            self.assertEqual(len(self.window.attachments), 0)
            self.assertFalse(self.window.attachment_tray.isVisible())

            # Add attachment
            self.window.add_attachment(temp_path)
            self.assertEqual(len(self.window.attachments), 1)
            self.assertFalse(self.window.attachment_tray.isHidden())
            self.assertEqual(self.window.attachments[0]["name"], Path(temp_path).name)
            self.assertIn("Test document content", self.window.attachments[0]["extracted_text"])

            # Adding duplicate should be ignored
            self.window.add_attachment(temp_path)
            self.assertEqual(len(self.window.attachments), 1)

            # Remove attachment
            self.window.remove_attachment(0)
            self.assertEqual(len(self.window.attachments), 0)
            self.assertTrue(self.window.attachment_tray.isHidden())

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_send_prompt_with_attachments(self):
        """Verify send_prompt incorporates attached documents and extracts text."""
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w", encoding="utf-8") as f:
            f.write("# Specification\nKey metric: 99.9% uptime.")
            temp_path = f.name

        try:
            self.window.add_attachment(temp_path)
            self.window.prompt_input.setText("Review attached specs.")

            # Mock ChatWorker to capture submitted Message
            with patch("src.ui.main_window.ChatWorker") as MockWorker:
                mock_inst = MagicMock()
                mock_inst.isRunning.return_value = False
                MockWorker.return_value = mock_inst

                self.window.send_prompt()

                # User message in history
                self.assertEqual(len(self.window.messages), 1)
                user_msg = self.window.messages[0]
                self.assertEqual(user_msg.role, MessageRole.USER)
                self.assertIn("Review attached specs", user_msg.content)
                self.assertIn(Path(temp_path).name, user_msg.content)

                # Prompt sent to worker must contain document content
                args, _ = MockWorker.call_args
                history_sent = args[1]
                prompt_sent = history_sent[-1].content
                self.assertIn("99.9% uptime", prompt_sent)

                # Attachment tray must be cleared after sending
                self.assertEqual(len(self.window.attachments), 0)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_send_prompt_with_image_attachment(self):
        """Verify image attachments attach local file paths to Message and route to vision."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82")
            temp_path = f.name

        try:
            self.window.add_attachment(temp_path)
            self.window.prompt_input.setText("Rate this photo.")

            with patch("src.ui.main_window.ChatWorker") as MockWorker:
                mock_inst = MagicMock()
                mock_inst.isRunning.return_value = False
                MockWorker.return_value = mock_inst

                self.window.send_prompt()

                self.assertEqual(len(self.window.messages), 1)
                user_msg = self.window.messages[0]
                self.assertEqual(user_msg.images, [str(Path(temp_path).resolve())])

                # ChatWorker message payload must preserve image paths
                args, _ = MockWorker.call_args
                history_sent = args[1]
                last_msg = history_sent[-1]
                self.assertEqual(last_msg.images, [str(Path(temp_path).resolve())])
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_smart_router_vision_capability(self):
        """Verify router selects vision multimodal model when image is present."""
        from src.routing.router import ModelRouter
        from src.routing.registry import ModelCapabilityRegistry
        from src.routing.types import TaskType

        registry = ModelCapabilityRegistry()
        router = ModelRouter(registry=registry)
        decision = router.route("Rate this person", has_images=True)
        self.assertEqual(decision.classification.task_type, TaskType.VISION)
        self.assertTrue(decision.primary.profile.supports_vision)
        self.assertIn(decision.primary.provider_id, ("gemini", "openai"))

    def test_speech_to_text_dictation(self):
        """Verify microphone dictation appends transcribed speech to prompt input."""
        self.window.prompt_input.setText("Hello")
        self.window._on_voice_transcription("world from speech dictation")
        self.assertEqual(self.window.prompt_input.text(), "Hello world from speech dictation")

    def test_attachment_actions_wiring(self):
        """Verify attachment sub-handlers invoke proper dialogs."""
        with patch("PySide6.QtWidgets.QFileDialog.getOpenFileNames") as mock_files:
            mock_files.return_value = (["/fake/photo.png"], "Images (*.png)")
            self.window._on_attach_photos()
            self.assertTrue(mock_files.called)

        with patch("PySide6.QtWidgets.QFileDialog.getOpenFileNames") as mock_docs:
            mock_docs.return_value = (["/fake/doc.pdf"], "Documents (*.pdf)")
            self.window._on_attach_files()
            self.assertTrue(mock_docs.called)


if __name__ == "__main__":
    unittest.main()
