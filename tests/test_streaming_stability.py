import os
import unittest
from unittest.mock import patch, MagicMock

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication
from src.ui import main_window
from src.ui.main_window import MainWindow

app = QApplication.instance() or QApplication([])


class TestStreamingStability(unittest.TestCase):
    def setUp(self):
        self.window = MainWindow()
        self.window.show()
        self.window.stack.setCurrentIndex(1)
        app.processEvents()

    def tearDown(self):
        self.window.close()

    def test_backslash_line_endings_eliminated(self):
        sample = (
            "Line with backslash at end\\\n"
            "Another line with space and backslash \\  \n"
            "Normal line\n"
            "1. Step one\\\n"
            "2. Step two\\\n"
        )
        html_out = MainWindow.markdown_to_html(sample, dark=True)
        # Should NOT contain literal backslashes
        self.assertNotIn("\\<br>", html_out)
        self.assertNotIn("\\</li>", html_out)
        self.assertNotIn("\\\n", html_out)

    def test_long_response_with_code_lists_headings(self):
        long_markdown = (
            "# Advanced Python Guide\n\n"
            "This is the first paragraph describing the architecture.\\\n"
            "This continues the paragraph on a new line.\\\n\n"
            "## Key Features\n"
            "- First feature item\n"
            "- Second feature item\n\n"
            "### Execution Steps\n"
            "1. Clone the repository\\\n"
            "2. Install dependencies\\\n"
            "3. Run the service\\\n\n"
            "```python\n"
            "def calculate_metrics(values: list[float]) -> dict:\n"
            "    total = sum(values)\n"
            "    avg = total / len(values) if values else 0.0\n"
            "    return {'total': total, 'average': avg}\n"
            "```\n\n"
            "Final conclusion and summary paragraph.\n"
        )

        tokens = [long_markdown[i:i+15] for i in range(0, len(long_markdown), 15)]

        self.window.prompt_input.setText("Explain Python metrics calculation")
        with patch.object(main_window, "stream_chat", return_value=tokens):
            with patch.object(self.window.transcript, "setHtml", wraps=self.window.transcript.setHtml) as mock_setHtml:
                self.window.send_prompt()

                # Simulate timer ticks during streaming
                for _ in range(len(tokens) + 5):
                    app.processEvents()
                    self.window._on_stream_timer_tick()

                if self.window.worker:
                    self.window.worker.wait(5000)
                app.processEvents()

                # setHtml should NOT have been called during token stream updates
                # (it is only used on full reload / theme change, not during streaming)
                self.assertEqual(mock_setHtml.call_count, 0)

        # Check final rendered content
        transcript_text = self.window.transcript.toPlainText()
        self.assertIn("Advanced Python Guide", transcript_text)
        self.assertIn("def calculate_metrics", transcript_text)
        self.assertIn("Clone the repository", transcript_text)
        self.assertIn("Final conclusion", transcript_text)
        # Verify no backslashes appeared at line endings
        self.assertNotIn("architecture.\\", transcript_text)
        self.assertNotIn("repository\\", transcript_text)

    def test_scroll_position_preserved_when_user_scrolls_up(self):
        scrollbar = self.window.transcript.verticalScrollBar()

        # Seed transcript with historical messages to allow scrolling
        for i in range(50):
            cursor = self.window.transcript.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            cursor.insertHtml(f"<p>Previous conversation message {i}</p>")
            cursor.insertBlock()
        app.processEvents()

        # Scroll to an upward reading position (e.g. 80)
        self.assertGreater(scrollbar.maximum(), 200)
        scrollbar.setValue(80)
        user_reading_pos = scrollbar.value()

        # Stream an assistant message
        self.window.current_assistant_start = self.window.transcript.textCursor().position()
        cursor = self.window.transcript.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertHtml("<p>Initial token</p>")
        self.window.current_assistant_end = cursor.position()

        # User is scrolled up (not near bottom)
        self.assertGreater(scrollbar.maximum() - scrollbar.value(), 50)

        # Update streaming content
        self.window.assistant_buffer = "A long chunk that generates multiple lines\n" * 10
        self.window._update_current_assistant_block()
        app.processEvents()

        # User's reading scroll position must be preserved, NOT forced to maximum!
        self.assertEqual(scrollbar.value(), user_reading_pos)
        self.assertNotEqual(scrollbar.value(), scrollbar.maximum())

    def test_auto_scroll_when_user_is_at_bottom(self):
        scrollbar = self.window.transcript.verticalScrollBar()

        # Scroll to absolute bottom
        scrollbar.setValue(scrollbar.maximum())
        app.processEvents()

        # Create active assistant block
        cursor = self.window.transcript.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.window.current_assistant_start = cursor.position()
        cursor.insertHtml("<p>Starting</p>")
        self.window.current_assistant_end = cursor.position()

        # Add new streaming content
        self.window.assistant_buffer = "New lines appearing at bottom\n" * 20
        self.window._update_current_assistant_block()
        app.processEvents()

        # Should auto-scroll to the new maximum
        self.assertEqual(scrollbar.value(), scrollbar.maximum())


if __name__ == "__main__":
    unittest.main()

