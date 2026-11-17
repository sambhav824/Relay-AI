import os
import unittest
from unittest.mock import patch

# Run Qt offscreen
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication
from src.ui import main_window
from src.ui.main_window import MainWindow

app = QApplication.instance()
if not app:
    app = QApplication([])


class TestRelayAIChatV2(unittest.TestCase):
    def setUp(self):
        self.window = MainWindow()

    def tearDown(self):
        self.window.close()

    def test_window_initialization(self):
        self.assertEqual(self.window.windowTitle(), "Relay AI")
        self.assertTrue(self.window.dark)
        self.assertEqual(len(self.window.messages), 0)
        self.assertIsNotNone(self.window.clear_button)
        self.assertIsNotNone(self.window.model_combo)
        self.assertIsNotNone(self.window.transcript)
        self.assertIsNotNone(self.window.prompt_input)
        self.assertIsNotNone(self.window.send_button)

    def test_theme_toggle(self):
        self.assertTrue(self.window.dark)
        self.window.toggle_theme()
        self.assertFalse(self.window.dark)
        self.window.toggle_theme()
        self.assertTrue(self.window.dark)

    def test_navigation_to_chat(self):
        self.assertEqual(self.window.stack.currentIndex(), 0)
        self.window.stack.setCurrentIndex(1)
        self.assertEqual(self.window.stack.currentIndex(), 1)

    def test_markdown_rendering_headings(self):
        md = "# Heading 1\n## Heading 2\n### Heading 3"
        html_out = MainWindow.markdown_to_html(md, dark=True)
        self.assertIn("<h1", html_out)
        self.assertIn("Heading 1</h1>", html_out)
        self.assertIn("<h2", html_out)
        self.assertIn("Heading 2</h2>", html_out)
        self.assertIn("<h3", html_out)
        self.assertIn("Heading 3</h3>", html_out)

    def test_markdown_rendering_code_blocks(self):
        md = "Here is code:\n```python\nprint('hello')\n```\nDone."
        html_out = MainWindow.markdown_to_html(md, dark=True)
        self.assertIn("<pre", html_out)
        self.assertTrue("hello" in html_out)

    def test_markdown_rendering_unclosed_code_blocks_streaming(self):
        md = "Streaming code:\n```python\ndef test():\n    return True"
        html_out = MainWindow.markdown_to_html(md, dark=True)
        self.assertIn("<pre", html_out)
        self.assertIn("def test():", html_out)

    def test_markdown_rendering_inline_code_and_bold(self):
        md = "This has `variable_name` and **important** info and *emphasis*."
        html_out = MainWindow.markdown_to_html(md, dark=True)
        self.assertIn("<code", html_out)
        self.assertIn("variable_name</code>", html_out)
        self.assertIn("<b>important</b>", html_out)
        self.assertIn("<i>emphasis</i>", html_out)

    def test_markdown_rendering_lists(self):
        md = "- First item\n- Second item\n1. Numbered 1\n2. Numbered 2"
        html_out = MainWindow.markdown_to_html(md, dark=True)
        self.assertIn("<ul", html_out)
        self.assertIn("<li>First item</li>", html_out)
        self.assertIn("<li>Second item</li>", html_out)
        self.assertIn("<ol", html_out)
        self.assertIn("<li>Numbered 1</li>", html_out)
        self.assertIn("<li>Numbered 2</li>", html_out)

    def test_html_escaping_for_security(self):
        md = "<script>alert('xss')</script>"
        html_out = MainWindow.markdown_to_html(md, dark=True)
        self.assertNotIn("<script>", html_out)
        self.assertIn("&lt;script&gt;", html_out)

    def test_chat_context_multi_turn(self):
        self.window.prompt_input.setText("Turn 1 prompt")
        recorded_calls = []

        def mock_stream(messages, model):
            recorded_calls.append(list(messages))
            yield "Turn 1 answer"

        with patch.object(main_window, "stream_chat", side_effect=mock_stream):
            self.window.send_prompt()
            if self.window.worker:
                self.window.worker.wait(5000)
            app.processEvents()

            self.window.prompt_input.setText("Turn 2 prompt")
            self.window.send_prompt()
            if self.window.worker:
                self.window.worker.wait(5000)
            app.processEvents()

        self.assertEqual(len(recorded_calls), 2)
        def to_dict_list(msgs):
            out = []
            for m in msgs:
                role = getattr(m, "role", m.get("role") if isinstance(m, dict) else None)
                role_str = role.value if hasattr(role, "value") else str(role).lower()
                if role_str == "system":
                    continue
                content = getattr(m, "content", m.get("content") if isinstance(m, dict) else None)
                out.append({"role": role_str, "content": content})
            return out

        self.assertEqual(to_dict_list(recorded_calls[0]), [{"role": "user", "content": "Turn 1 prompt"}])
        self.assertEqual(to_dict_list(recorded_calls[1]), [
            {"role": "user", "content": "Turn 1 prompt"},
            {"role": "assistant", "content": "Turn 1 answer"},
            {"role": "user", "content": "Turn 2 prompt"}
        ])
        self.assertEqual(len(self.window.messages), 4)

    def test_clear_chat(self):
        self.window.messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"}
        ]
        self.window.render_transcript()
        self.assertNotEqual(self.window.transcript.toPlainText().strip(), "")

        self.window.clear_chat()
        self.assertEqual(len(self.window.messages), 0)
        self.assertEqual(self.window.assistant_buffer, "")
        self.assertEqual(self.window.transcript.toPlainText().strip(), "")
        self.assertEqual(self.window.status_badge.text(), "")

    def test_thinking_indicator_removed_on_completed(self):
        self.window.prompt_input.setText("Hello")
        with patch.object(main_window, "stream_chat", return_value=["First chunk", " second chunk"]):
            self.window.send_prompt()
            # Before worker runs, thinking is set
            self.assertTrue(self.window.is_thinking)
            if self.window.worker:
                self.window.worker.wait(5000)
            app.processEvents()

        # After completed: is_thinking must be False and status badge cleared
        self.assertFalse(self.window.is_thinking)
        self.assertEqual(self.window.status_badge.text(), "")
        self.assertEqual(self.window.send_button.text(), "Send")
        # Transcript should NOT contain "thinking"
        transcript_text = self.window.transcript.toPlainText()
        self.assertNotIn("thinking", transcript_text.lower())
        self.assertIn("First chunk second chunk", transcript_text)

    def test_thinking_indicator_removed_on_error(self):
        self.window.prompt_input.setText("Failing prompt")
        with patch.object(main_window, "stream_chat", side_effect=RuntimeError("Cannot connect to Ollama")):
            self.window.send_prompt()
            if self.window.worker:
                self.window.worker.wait(5000)
            app.processEvents()

        self.assertFalse(self.window.is_thinking)
        self.assertEqual(len(self.window.messages), 0)
        self.assertEqual(self.window.send_button.text(), "Send")
        transcript_text = self.window.transcript.toPlainText()
        self.assertNotIn("thinking", transcript_text.lower())
        self.assertIn("Cannot connect to Ollama", transcript_text)
        self.assertTrue(self.window.send_button.isEnabled())

    def test_layout_adaptation_and_fixed_input(self):
        self.window.show()
        self.window.stack.setCurrentIndex(1)
        app.processEvents()

        for w, h in [(880, 560), (1280, 800), (1600, 900)]:
            self.window.resize(w, h)
            app.processEvents()

            # Ensure transcript is sized and visible
            self.assertGreater(self.window.transcript.height(), 150)
            # Ensure input and send button are within visible window bounds
            input_pos_y = self.window.prompt_input.mapTo(self.window, self.window.prompt_input.rect().topLeft()).y()
            send_pos_y = self.window.send_button.mapTo(self.window, self.window.send_button.rect().topLeft()).y()
            self.assertLess(input_pos_y + self.window.prompt_input.height(), h)
            self.assertLess(send_pos_y + self.window.send_button.height(), h)
            self.assertGreater(input_pos_y, 0)


if __name__ == "__main__":
    unittest.main()
