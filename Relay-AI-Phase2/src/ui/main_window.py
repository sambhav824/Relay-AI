import html
import re
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton,
    QLabel, QFrame, QStackedWidget, QLineEdit, QTextBrowser, QComboBox,
    QGridLayout, QSizePolicy
)
from src.providers.ollama import stream_chat, generate

DARK = """
QMainWindow, QWidget { background: #111318; color: #E8EAF0; font-family: 'Segoe UI', sans-serif; }
QFrame#Sidebar { background: #191C23; border: 1px solid #292E39; border-radius: 14px; }
QLabel#Brand { font-size: 22px; font-weight: 700; color: #FFFFFF; }
QLabel#Muted { color: #9AA3B2; }
QPushButton { background: #242936; color: #E8EAF0; border: 1px solid #343B4A; border-radius: 9px; padding: 10px 13px; text-align: left; }
QPushButton:hover { background: #30394A; }
QPushButton:disabled { color: #737B89; background: #1C2029; border-color: #272D3B; }
QPushButton#Primary { background: #6658E8; border: 0; color: #FFFFFF; font-weight: 600; text-align: center; }
QPushButton#Primary:hover { background: #786CF0; }
QPushButton#Primary:disabled { background: #453E8A; color: #8F8CAE; }
QPushButton#Secondary { background: #242936; color: #E8EAF0; border: 1px solid #343B4A; border-radius: 9px; padding: 7px 14px; text-align: center; }
QPushButton#Secondary:hover { background: #30394A; }
QPushButton#Secondary:disabled { color: #737B89; background: #1C2029; border-color: #272D3B; }
QFrame#Panel { background: #191C23; border: 1px solid #292E39; border-radius: 14px; }
QFrame#InputContainer { background: #191C23; border: 1px solid #292E39; border-radius: 12px; }
QLineEdit, QComboBox { background: #202530; color: #F2F4F8; border: 1px solid #353D4D; border-radius: 8px; padding: 8px 12px; }
QLineEdit:focus, QComboBox:focus { border: 1px solid #6658E8; }
QTextBrowser { background: #14171E; color: #E8EAF0; border: 1px solid #282E3B; border-radius: 12px; padding: 14px; }
QScrollBar:vertical { background: #14171E; width: 8px; margin: 4px 2px 4px 0; border-radius: 4px; }
QScrollBar::handle:vertical { background: #2D3340; min-height: 24px; border-radius: 4px; }
QScrollBar::handle:vertical:hover { background: #40485A; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QLabel#PageTitle { font-size: 24px; font-weight: 700; }
"""

LIGHT = """
QMainWindow, QWidget { background: #F5F6FA; color: #202432; font-family: 'Segoe UI', sans-serif; }
QFrame#Sidebar { background: #FFFFFF; border: 1px solid #E1E4EC; border-radius: 14px; }
QLabel#Brand { font-size: 22px; font-weight: 700; color: #202432; }
QLabel#Muted { color: #687184; }
QPushButton { background: #FFFFFF; color: #202432; border: 1px solid #E1E4EC; border-radius: 9px; padding: 10px 13px; text-align: left; }
QPushButton:hover { background: #EEF0F8; }
QPushButton:disabled { color: #A0A6B2; background: #F8F9FC; border-color: #E6E9F0; }
QPushButton#Primary { background: #6658E8; border: 0; color: #FFFFFF; font-weight: 600; text-align: center; }
QPushButton#Primary:hover { background: #786CF0; }
QPushButton#Primary:disabled { background: #A29BE8; color: #EFEFFB; }
QPushButton#Secondary { background: #FFFFFF; color: #202432; border: 1px solid #E1E4EC; border-radius: 9px; padding: 7px 14px; text-align: center; }
QPushButton#Secondary:hover { background: #EEF0F8; }
QPushButton#Secondary:disabled { color: #A0A6B2; background: #F8F9FC; border-color: #E6E9F0; }
QFrame#Panel { background: #FFFFFF; border: 1px solid #E1E4EC; border-radius: 14px; }
QFrame#InputContainer { background: #FFFFFF; border: 1px solid #E1E4EC; border-radius: 12px; }
QLineEdit, QComboBox { background: #FFFFFF; color: #202432; border: 1px solid #D9DEEA; border-radius: 8px; padding: 8px 12px; }
QLineEdit:focus, QComboBox:focus { border: 1px solid #6658E8; }
QTextBrowser { background: #FFFFFF; color: #202432; border: 1px solid #E1E4EC; border-radius: 12px; padding: 14px; }
QScrollBar:vertical { background: #FFFFFF; width: 8px; margin: 4px 2px 4px 0; border-radius: 4px; }
QScrollBar::handle:vertical { background: #CBD2E0; min-height: 24px; border-radius: 4px; }
QScrollBar::handle:vertical:hover { background: #A4B0C4; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QLabel#PageTitle { font-size: 24px; font-weight: 700; }
"""


class ChatWorker(QThread):
    chunk_received = Signal(str)
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, messages, model):
        super().__init__()
        self.messages = messages
        self.model = model

    def run(self):
        parts = []
        try:
            for chunk in stream_chat(self.messages, self.model):
                parts.append(chunk)
                self.chunk_received.emit(chunk)
            self.completed.emit("".join(parts))
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.dark = True
        self.worker = None
        self.messages = []
        self.assistant_buffer = ""
        self.is_thinking = False

        # Active streaming cursor positions to update only the current message
        self.current_assistant_start = None
        self.current_assistant_end = None
        self.has_new_tokens = False

        # Controlled batch timer for smooth 60ms UI updates
        self.stream_timer = QTimer(self)
        self.stream_timer.setInterval(60)
        self.stream_timer.timeout.connect(self._on_stream_timer_tick)

        self.setWindowTitle("Relay AI")
        self.resize(1180, 760)
        self.setMinimumSize(880, 560)

        shell = QWidget()
        outer = QHBoxLayout(shell)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(14)

        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(220)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(14, 18, 14, 14)
        side.setSpacing(8)

        brand = QLabel("✦  Relay AI")
        brand.setObjectName("Brand")
        side.addWidget(brand)
        tagline = QLabel("Your multi-model workspace")
        tagline.setObjectName("Muted")
        side.addWidget(tagline)
        side.addSpacing(16)

        self.stack = QStackedWidget()
        pages = [
            ("⌂   Dashboard", self.dashboard_page()),
            ("✧   AI Chat", self.chat_page()),
            ("▦   Model Manager", self.simple_page("Model Manager", "Ollama local model: llama3.2:3b. Cloud providers are planned next.")),
            ("◷   Chat History", self.simple_page("Chat History", "Persistent SQLite conversation history is planned next.")),
            ("⚙   Settings", self.settings_page()),
        ]
        for label, page in pages:
            button = QPushButton(label)
            button.clicked.connect(lambda checked=False, p=page: self.stack.setCurrentWidget(p))
            side.addWidget(button)
        side.addStretch()

        theme_btn = QPushButton("☼   Toggle appearance")
        theme_btn.clicked.connect(self.toggle_theme)
        side.addWidget(theme_btn)

        content = QVBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.addWidget(self.stack)

        outer.addWidget(sidebar)
        outer.addLayout(content, 1)
        self.setCentralWidget(shell)
        self.apply_theme()

    def page_base(self, title, subtitle):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(10, 6, 10, 8)
        layout.setSpacing(12)
        title_label = QLabel(title)
        title_label.setObjectName("PageTitle")
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("Muted")
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        return page, layout

    def panel(self, title, description):
        frame = QFrame()
        frame.setObjectName("Panel")
        box = QVBoxLayout(frame)
        box.setContentsMargins(16, 14, 16, 14)
        heading = QLabel(title)
        heading.setStyleSheet("font-size: 15px; font-weight: 650;")
        desc = QLabel(description)
        desc.setObjectName("Muted")
        desc.setWordWrap(True)
        box.addWidget(heading)
        box.addWidget(desc)
        return frame, box

    def dashboard_page(self):
        page, layout = self.page_base("Good to see you.", "One workspace for all your AI models.")
        grid = QGridLayout()
        cards = [
            ("Connected providers", "Ollama — local"),
            ("Available local models", "llama3.2:3b"),
            ("Cloud providers", "Not configured yet"),
        ]
        for i, (title, desc) in enumerate(cards):
            card, _ = self.panel(title, desc)
            grid.addWidget(card, 0, i)
        layout.addLayout(grid)

        welcome, box = self.panel(
            "Your local AI is ready",
            "Llama 3.2 3B is connected through Ollama. Chat now with real-time streaming and session context."
        )
        action = QPushButton("Open AI Chat")
        action.setObjectName("Primary")
        action.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        box.addWidget(action)
        layout.addWidget(welcome)
        layout.addStretch()
        self.stack.addWidget(page)
        return page

    def chat_page(self):
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(8, 4, 8, 8)
        page_layout.setSpacing(10)

        # Header area: Title, Subtitle, Model Selector, Status Badge, Clear Conversation
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(4, 2, 4, 4)
        header_layout.setSpacing(10)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title_label = QLabel("AI Chat")
        title_label.setObjectName("PageTitle")
        subtitle_label = QLabel("Streaming responses · Session context · Markdown rendering")
        subtitle_label.setObjectName("Muted")
        title_box.addWidget(title_label)
        title_box.addWidget(subtitle_label)
        header_layout.addLayout(title_box)

        header_layout.addStretch()

        controls = QHBoxLayout()
        controls.setSpacing(8)

        lbl = QLabel("Model:")
        lbl.setObjectName("Muted")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["llama3.2:3b"])
        self.model_combo.setFixedWidth(140)

        self.status_badge = QLabel("")
        self.status_badge.setStyleSheet("font-size: 12px; font-weight: 600; padding: 2px 6px;")

        self.clear_button = QPushButton("Clear conversation")
        self.clear_button.setObjectName("Secondary")
        self.clear_button.clicked.connect(self.clear_chat)

        controls.addWidget(lbl)
        controls.addWidget(self.model_combo)
        controls.addWidget(self.status_badge)
        controls.addWidget(self.clear_button)
        header_layout.addLayout(controls)

        page_layout.addWidget(header, 0)

        # Chat transcript: Scrollable, takes available vertical space
        self.transcript = QTextBrowser()
        self.transcript.setOpenExternalLinks(True)
        self.transcript.setPlaceholderText("Start a conversation with Llama 3.2. Type your message below and press Enter.")
        self.transcript.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        page_layout.addWidget(self.transcript, 1)

        # Fixed bottom input bar: Fixed height, pinned to bottom
        input_container = QFrame()
        input_container.setObjectName("InputContainer")
        input_layout = QHBoxLayout(input_container)
        input_layout.setContentsMargins(10, 8, 10, 8)
        input_layout.setSpacing(10)

        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Message Relay AI… (Press Enter to send)")
        self.prompt_input.setFixedHeight(40)
        self.prompt_input.returnPressed.connect(self.send_prompt)

        self.send_button = QPushButton("Send")
        self.send_button.setObjectName("Primary")
        self.send_button.setFixedHeight(40)
        self.send_button.setFixedWidth(90)
        self.send_button.clicked.connect(self.send_prompt)

        input_layout.addWidget(self.prompt_input, 1)
        input_layout.addWidget(self.send_button, 0)

        page_layout.addWidget(input_container, 0)

        self.stack.addWidget(page)
        return page

    def send_prompt(self):
        prompt = self.prompt_input.text().strip()
        if not prompt or (self.worker and self.worker.isRunning()):
            return

        model = self.model_combo.currentText()
        self.messages.append({"role": "user", "content": prompt})
        self.prompt_input.clear()
        self.assistant_buffer = ""
        self.is_thinking = True
        self.has_new_tokens = False

        self.status_badge.setText("Thinking…")
        self.status_badge.setStyleSheet("color: #F59E0B; font-size: 12px; font-weight: 600;")
        self.send_button.setEnabled(False)
        self.send_button.setText("Thinking…")
        self.clear_button.setEnabled(False)
        self.prompt_input.setEnabled(False)

        # 1. Append User message to transcript
        cursor = self.transcript.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertHtml(self._format_user_card(prompt))
        cursor.insertBlock()

        # 2. Append Assistant placeholder to transcript and record cursor bounds
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.current_assistant_start = cursor.position()
        cursor.insertHtml(self._format_assistant_card("", is_thinking=True))
        self.current_assistant_end = cursor.position()

        # Scroll to bottom for user's new message
        self._scroll_if_near_bottom(force=True)

        # Start 60ms batch update timer
        self.stream_timer.start(60)

        # Start background worker
        self.worker = ChatWorker(list(self.messages), model)
        self.worker.chunk_received.connect(self.on_chunk)
        self.worker.completed.connect(self.on_completed)
        self.worker.failed.connect(self.on_error)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_chunk(self, chunk):
        # Buffer token chunks; UI is updated in controlled 60ms batches
        self.is_thinking = False
        self.assistant_buffer += chunk
        self.has_new_tokens = True

    def _on_stream_timer_tick(self):
        if self.has_new_tokens:
            self.has_new_tokens = False
            self.status_badge.setText("Streaming…")
            self.status_badge.setStyleSheet("color: #10B981; font-size: 12px; font-weight: 600;")
            self.send_button.setText("Responding…")
            self._update_current_assistant_block()

    def _update_current_assistant_block(self, final=False):
        if self.current_assistant_start is None:
            return

        scrollbar = self.transcript.verticalScrollBar()
        # Check if user is near the bottom (within 45px)
        was_near_bottom = (scrollbar.maximum() - scrollbar.value()) <= 45
        saved_scroll = scrollbar.value()

        # Update only the current assistant message range using QTextCursor
        cursor = self.transcript.textCursor()
        cursor.setPosition(self.current_assistant_start)
        cursor.setPosition(self.current_assistant_end, QTextCursor.MoveMode.KeepAnchor)
        cursor.insertHtml(self._format_assistant_card(self.assistant_buffer, is_thinking=self.is_thinking))
        self.current_assistant_end = cursor.position()

        # Preserve scroll position if user scrolled up; only auto-scroll if near bottom
        if was_near_bottom:
            scrollbar.setValue(scrollbar.maximum())
        else:
            scrollbar.setValue(saved_scroll)

    def on_completed(self, answer):
        self.stream_timer.stop()
        self.is_thinking = False
        self.has_new_tokens = False
        self.assistant_buffer = answer

        # Final flush update
        self._update_current_assistant_block(final=True)
        self.messages.append({"role": "assistant", "content": answer})

        self.assistant_buffer = ""
        self.current_assistant_start = None
        self.current_assistant_end = None
        self.status_badge.setText("")

    def on_error(self, message):
        self.stream_timer.stop()
        self.is_thinking = False
        self.has_new_tokens = False

        if self.messages and self.messages[-1]["role"] == "user":
            self.messages.pop()

        # Replace active assistant block with error banner
        if self.current_assistant_start is not None:
            cursor = self.transcript.textCursor()
            cursor.setPosition(self.current_assistant_start)
            cursor.setPosition(self.current_assistant_end, QTextCursor.MoveMode.KeepAnchor)
            cursor.insertHtml(self._format_error_card(message))
            self.current_assistant_start = None
            self.current_assistant_end = None

        self.assistant_buffer = ""
        self.status_badge.setText("Error")
        self.status_badge.setStyleSheet("color: #EF4444; font-size: 12px; font-weight: 600;")
        self._scroll_if_near_bottom(force=True)

    def on_finished(self):
        self.stream_timer.stop()
        self.is_thinking = False
        self.status_badge.setText("")
        self.send_button.setEnabled(True)
        self.send_button.setText("Send")
        self.clear_button.setEnabled(True)
        self.prompt_input.setEnabled(True)
        self.prompt_input.setFocus()

    def _scroll_if_near_bottom(self, force=False):
        scrollbar = self.transcript.verticalScrollBar()
        if force or (scrollbar.maximum() - scrollbar.value() <= 45):
            scrollbar.setValue(scrollbar.maximum())

    def _format_user_card(self, content):
        user_header_color = "#818CF8" if self.dark else "#4F46E5"
        user_bg = "#1B202B" if self.dark else "#F1F4F9"
        user_border = "#2B3344" if self.dark else "#DCE2ED"
        rendered = self.markdown_to_html(content, self.dark)
        return (
            f'<div style="margin-bottom: 16px;">'
            f'  <div style="font-weight: 700; font-size: 13px; color: {user_header_color}; margin-bottom: 6px; padding-left: 2px;">You</div>'
            f'  <div style="background: {user_bg}; border: 1px solid {user_border}; border-radius: 12px; padding: 12px 16px; font-size: 14px; line-height: 1.5;">'
            f'    {rendered}'
            f'  </div>'
            f'</div>'
        )

    def _format_assistant_card(self, content, is_thinking=False):
        bot_header_color = "#34D399" if self.dark else "#059669"
        bot_bg = "#151820" if self.dark else "#FFFFFF"
        bot_border = "#262D3B" if self.dark else "#E5E7EB"

        if is_thinking and not content:
            body = '<div style="color: #9AA3B2; font-style: italic;">Thinking…</div>'
        else:
            body = self.markdown_to_html(content, self.dark)

        return (
            f'<div style="margin-bottom: 16px;">'
            f'  <div style="font-weight: 700; font-size: 13px; color: {bot_header_color}; margin-bottom: 6px; padding-left: 2px;">✦ Relay AI · Llama</div>'
            f'  <div style="background: {bot_bg}; border: 1px solid {bot_border}; border-radius: 12px; padding: 12px 16px; font-size: 14px; line-height: 1.5;">'
            f'    {body}'
            f'  </div>'
            f'</div>'
        )

    def _format_error_card(self, error_message):
        error_bg = "#2D1517" if self.dark else "#FEE2E2"
        error_border = "#7F1D1D" if self.dark else "#FCA5A5"
        error_color = "#FCA5A5" if self.dark else "#991B1B"
        escaped_err = html.escape(error_message)
        return (
            f'<div style="margin-bottom: 16px; padding: 12px 16px; background: {error_bg}; border: 1px solid {error_border}; border-radius: 12px; color: {error_color};">'
            f'  <b>Connection Error:</b> {escaped_err}'
            f'</div>'
        )

    def render_transcript(self, error_message=None):
        """Full transcript render used for initial load or theme changes."""
        parts = []
        for msg in self.messages:
            if msg["role"] == "user":
                parts.append(self._format_user_card(msg["content"]))
            else:
                parts.append(self._format_assistant_card(msg["content"]))

        if self.assistant_buffer:
            parts.append(self._format_assistant_card(self.assistant_buffer))
        elif self.is_thinking:
            parts.append(self._format_assistant_card("", is_thinking=True))

        if error_message:
            parts.append(self._format_error_card(error_message))

        self.transcript.setHtml("".join(parts))

    @staticmethod
    def markdown_to_html(text, dark=True):
        if not text:
            return ""

        escaped = html.escape(text)
        code_blocks = []

        pre_bg = "#11141A" if dark else "#F8FAFC"
        pre_color = "#E2E8F0" if dark else "#1E293B"
        pre_border = "#2A3140" if dark else "#E2E8F0"
        code_bg = "#232936" if dark else "#E9EEF5"
        code_color = "#93C5FD" if dark else "#1D4ED8"

        def replace_code(match):
            code = match.group(1)
            code_blocks.append(
                f'<pre style="background: {pre_bg}; color: {pre_color}; border: 1px solid {pre_border}; '
                f'padding: 10px 14px; border-radius: 8px; font-family: Consolas, \'Courier New\', monospace; '
                f'font-size: 13px; white-space: pre-wrap; margin: 8px 0;">{code}</pre>'
            )
            return f"@@CODE_BLOCK_{len(code_blocks)-1}@@"

        # Match closed code blocks
        escaped = re.sub(r"```(?:[a-zA-Z0-9_+-]+)?\s*\n(.*?)```", replace_code, escaped, flags=re.S)
        # Match unclosed code blocks during real-time streaming
        escaped = re.sub(r"```(?:[a-zA-Z0-9_+-]+)?\s*\n(.*)$", replace_code, escaped, flags=re.S)

        # Headings
        escaped = re.sub(r"^### (.+)$", r'<h3 style="margin: 8px 0 4px 0; font-size: 15px;">\1</h3>', escaped, flags=re.M)
        escaped = re.sub(r"^## (.+)$", r'<h2 style="margin: 10px 0 6px 0; font-size: 17px;">\1</h2>', escaped, flags=re.M)
        escaped = re.sub(r"^# (.+)$", r'<h1 style="margin: 12px 0 8px 0; font-size: 19px;">\1</h1>', escaped, flags=re.M)

        # Bold & Italic
        escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
        escaped = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", escaped)

        # Inline code
        escaped = re.sub(
            r"`([^`]+)`",
            f'<code style="background: {code_bg}; color: {code_color}; padding: 2px 6px; border-radius: 4px; font-family: Consolas, monospace; font-size: 13px;">\\1</code>',
            escaped
        )

        # Lists & line breaks
        lines = escaped.splitlines()
        rendered = []
        in_ul = False
        in_ol = False

        for line in lines:
            # Strip trailing backslash characters emitted at line endings for markdown hard line breaks
            line = re.sub(r"\\\s*$", "", line)

            ul_match = re.match(r"^\s*[-*]\s+(.+)$", line)
            ol_match = re.match(r"^\s*\d+\.\s+(.+)$", line)

            if ul_match:
                if in_ol:
                    rendered.append("</ol>")
                    in_ol = False
                if not in_ul:
                    rendered.append('<ul style="margin: 4px 0; padding-left: 20px;">')
                    in_ul = True
                rendered.append(f"<li>{ul_match.group(1)}</li>")
            elif ol_match:
                if in_ul:
                    rendered.append("</ul>")
                    in_ul = False
                if not in_ol:
                    rendered.append('<ol style="margin: 4px 0; padding-left: 20px;">')
                    in_ol = True
                rendered.append(f"<li>{ol_match.group(1)}</li>")
            else:
                if in_ul:
                    rendered.append("</ul>")
                    in_ul = False
                if in_ol:
                    rendered.append("</ol>")
                    in_ol = False

                is_block_element = line.startswith("<h") or line.startswith("<pre") or "@@CODE_BLOCK_" in line
                rendered.append(line if is_block_element else (line + "<br>" if line else "<br>"))

        if in_ul:
            rendered.append("</ul>")
        if in_ol:
            rendered.append("</ol>")

        result = "\n".join(rendered)
        for i, block in enumerate(code_blocks):
            result = result.replace(f"@@CODE_BLOCK_{i}@@", block)

        return result

    def clear_chat(self):
        if self.worker and self.worker.isRunning():
            return
        self.stream_timer.stop()
        self.messages.clear()
        self.assistant_buffer = ""
        self.is_thinking = False
        self.current_assistant_start = None
        self.current_assistant_end = None
        self.status_badge.setText("")
        self.transcript.clear()

    def simple_page(self, title, description):
        page, layout = self.page_base(title, description)
        card, _ = self.panel("Planned milestone", description)
        layout.addWidget(card)
        layout.addStretch()
        self.stack.addWidget(page)
        return page

    def settings_page(self):
        page, layout = self.page_base("Settings", "Configure how Relay AI looks and behaves.")
        card, _ = self.panel("Appearance", "Use the button at the bottom of the sidebar to switch between dark and light themes.")
        layout.addWidget(card)
        layout.addStretch()
        self.stack.addWidget(page)
        return page

    def toggle_theme(self):
        self.dark = not self.dark
        self.apply_theme()
        if self.messages or self.assistant_buffer:
            self.render_transcript()

    def apply_theme(self):
        self.setStyleSheet(DARK if self.dark else LIGHT)
