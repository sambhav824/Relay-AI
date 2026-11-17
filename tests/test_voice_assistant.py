import os
import sys
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication
from src.voice.assistant import (
    clean_markdown_for_speech,
    VoiceState,
    SpeechSynthesizer,
    VoiceAssistantController,
    WaveformWidget,
    VoiceOverlayPanel,
)

app = QApplication.instance() or QApplication([])


class TestVoiceAssistant(unittest.TestCase):
    def test_clean_markdown_for_speech(self):
        raw = """# Introduction
Here is **bold** text and *italic* text.
Also inline `variable_name` and a code block:
```python
def foo():
    return 42
```
Check this link: [OpenAI](https://openai.com) and raw url: https://google.com.
- Bullet point 1
- Bullet point 2
"""
        cleaned = clean_markdown_for_speech(raw)
        self.assertNotIn("```", cleaned)
        self.assertNotIn("def foo():", cleaned)
        self.assertNotIn("https://", cleaned)
        self.assertNotIn("**", cleaned)
        self.assertNotIn("`", cleaned)
        self.assertIn("Introduction", cleaned)
        self.assertIn("bold text and italic text", cleaned)
        self.assertIn("Bullet point 1", cleaned)

    def test_clean_markdown_empty_or_none(self):
        self.assertEqual(clean_markdown_for_speech(""), "")
        self.assertEqual(clean_markdown_for_speech(None), "")

    def test_speech_synthesizer_mock(self):
        synth = SpeechSynthesizer()
        self.assertFalse(synth.is_speaking())
        synth.stop()
        self.assertFalse(synth.is_speaking())

    def test_waveform_widget_states(self):
        wave = WaveformWidget(bars_count=5, dark=True)
        self.assertFalse(wave.is_active)
        wave.set_active(True)
        self.assertTrue(wave.is_active)
        wave._update_wave()
        wave.set_active(False)
        self.assertFalse(wave.is_active)

    def test_voice_overlay_panel_states(self):
        panel = VoiceOverlayPanel(dark=True)
        panel.set_state(VoiceState.LISTENING, "Test listening")
        self.assertIn("Listening", panel.status_label.text())

        panel.set_state(VoiceState.THINKING, "Test thinking")
        self.assertIn("Thinking", panel.status_label.text())

        panel.set_state(VoiceState.SPEAKING, "Test speaking")
        self.assertIn("Speaking", panel.status_label.text())

        panel.set_state(VoiceState.IDLE)
        self.assertIn("Ready", panel.status_label.text())

    def test_voice_assistant_controller_lifecycle(self):
        ctrl = VoiceAssistantController()
        ctrl.recognizer.start = MagicMock()
        self.assertEqual(ctrl.state, VoiceState.IDLE)
        self.assertFalse(ctrl.is_voice_mode_active)

        # Toggle on
        is_active = ctrl.toggle_voice_mode()
        self.assertTrue(is_active)
        self.assertTrue(ctrl.is_voice_mode_active)
        self.assertEqual(ctrl.state, VoiceState.LISTENING)

        # Notify prompt sent
        ctrl.on_prompt_sent()
        self.assertEqual(ctrl.state, VoiceState.THINKING)

        # Mock synthesizer to avoid live audio device requirement during unit testing
        ctrl.synthesizer.speak = MagicMock()
        ctrl.on_response_ready("Hello from Relay AI!")
        self.assertEqual(ctrl.state, VoiceState.SPEAKING)
        ctrl.synthesizer.speak.assert_called_once_with("Hello from Relay AI!")

        # Stop all
        ctrl.stop_all()
        self.assertEqual(ctrl.state, VoiceState.IDLE)
        self.assertFalse(ctrl.is_voice_mode_active)


if __name__ == "__main__":
    unittest.main()
