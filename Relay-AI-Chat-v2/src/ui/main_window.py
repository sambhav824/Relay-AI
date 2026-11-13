import html
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton,
    QLabel, QFrame, QStackedWidget, QLineEdit, QTextBrowser, QComboBox,
    QGridLayout, QMessageBox
)
from src.providers.ollama import stream_chat

DARK = """
QMainWindow, QWidget { background: #111318; color: #E8EAF0; font-family: Segoe UI; }
QFrame#Sidebar { background: #191C23; border: 1px solid #292E39; border-radius: 14px; }
QLabel#Brand { font-size: 22px; font-weight: 700; color: #FFFFFF; }
QLabel#Muted { color: #9AA3B2; }
QPushButton { background: #242936; color: #E8EAF0; border: 1px solid #343B4A; border-radius: 9px; padding: 10px 13px; text-align: left; }
QPushButton:hover { background: #30394A; }
QPushButton:disabled { color: #737B89; }
QPushButton#Primary { background: #6658E8; border: 0; color: white; font-weight: 600; }
QFrame#Panel { background: #191C23; border: 1px solid #292E39; border-radius: 14px; }
QLineEdit, QComboBox { background: #202530; color: #F2F4F8; border: 1px solid #353D4D; border-radius: 9px; padding: 10px; }
QTextBrowser { background: #171B23; color: #E8EAF0; border: 1px solid #353D4D; border-radius: 12px; padding: 12px; }
QLabel#PageTitle { font-size: 26px; font-weight: 700; }
"""
LIGHT = """
QMainWindow, QWidget { background: #F5F6FA; color: #202432; font-family: Segoe UI; }
QFrame#Sidebar { background: #FFFFFF; border: 1px solid #E1E4EC; border-radius: 14px; }
QLabel#Brand { font-size: 22px; font-weight: 700; color: #202432; }
QLabel#Muted { color: #687184; }
QPushButton { background: #FFFFFF; color: #202432; border: 1px solid #E1E4EC; border-radius: 9px; padding: 10px 13px; text-align: left; }
QPushButton:hover { background: #EEF0F8; }
QPushButton:disabled { color: #A0A6B2; }
QPushButton#Primary { background: #6658E8; border: 0; color: white; font-weight: 600; }
QFrame#Panel { background: #FFFFFF; border: 1px solid #E1E4EC; border-radius: 14px; }
QLineEdit, QComboBox { background: #FFFFFF; color: #202432; border: 1px solid #D9DEEA; border-radius: 9px; padding: 10px; }
QTextBrowser { background: #FFFFFF; color: #202432; border: 1px solid #D9DEEA; border-radius: 12px; padding: 12px; }
QLabel#PageTitle { font-size: 26px; font-weight: 700; }
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
        self.setWindowTitle("Relay AI")
        self.resize(1240, 800)
        self.setMinimumSize(980, 640)

        shell = QWidget()
        outer = QHBoxLayout(shell)
        outer.setContentsMargins(18, 18, 18, 18)
        outer.setSpacing(16)

        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(225)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(16, 20, 16, 16)
        side.setSpacing(10)
        brand = QLabel("✦  Relay AI")
        brand.setObjectName("Brand")
        side.addWidget(brand)
        tagline = QLabel("Your multi-model workspace")
        tagline.setObjectName("Muted")
        side.addWidget(tagline)
        side.addSpacing(20)

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
        content.addWidget(self.stack)
        outer.addWidget(sidebar)
        outer.addLayout(content, 1)
        self.setCentralWidget(shell)
        self.apply_theme()

    def page_base(self, title, subtitle):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 8, 14, 8)
        layout.setSpacing(14)
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
        box.setContentsMargins(18, 16, 18, 16)
        heading = QLabel(title)
        heading.setStyleSheet("font-size: 16px; font-weight: 650;")
        desc = QLabel(description)
        desc.setObjectName("Muted")
        desc.setWordWrap(True)
        box.addWidget(heading)
        box.addWidget(desc)
        return frame, box

    def dashboard_page(self):
        page, layout = self.page_base("Good to see you.", "One workspace for all your AI models.")
        grid = QGridLayout()
        for i, (title, desc) in enumerate([
            ("Connected providers", "Ollama — local"),
            ("Available local models", "llama3.2:3b"),
            ("Cloud providers", "Not configured yet"),
        ]):
            card, _ = self.panel(title, desc)
            grid.addWidget(card, 0, i)
        layout.addLayout(grid)
        welcome, box = self.panel("Your local AI is ready", "Llama 3.2 3B is connected through Ollama. Chat now with streaming and session context.")
        action = QPushButton("Open AI Chat")
        action.setObjectName("Primary")
        action.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        box.addWidget(action)
        layout.addWidget(welcome)
        layout.addStretch()
        self.stack.addWidget(page)
        return page

    def chat_page(self):
        page, layout = self.page_base("AI Chat", "Streaming responses · Context within this session · Markdown display")
        bar = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(["llama3.2:3b"])
        bar.addWidget(QLabel("Model"))
        bar.addWidget(self.model_combo, 1)
        self.clear_button = QPushButton("Clear chat")
        self.clear_button.clicked.connect(self.clear_chat)
        bar.addWidget(self.clear_button)
        layout.addLayout(bar)

        self.transcript = QTextBrowser()
        self.transcript.setOpenExternalLinks(True)
        self.transcript.setPlaceholderText("Start a conversation with Llama 3.2.")
        layout.addWidget(self.transcript, 1)

        row = QHBoxLayout()
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Message Relay AI…")
        self.prompt_input.returnPressed.connect(self.send_prompt)
        self.send_button = QPushButton("Send")
        self.send_button.setObjectName("Primary")
        self.send_button.clicked.connect(self.send_prompt)
        row.addWidget(self.prompt_input, 1)
        row.addWidget(self.send_button)
        layout.addLayout(row)
        self.stack.addWidget(page)
        return page

    def send_prompt(self):
        prompt = self.prompt_input.text().strip()
        if not prompt or (self.worker and self.worker.isRunning()):
            return
        model = self.model_combo.currentText()
        self.messages.append({"role": "user", "content": prompt})
        self.transcript.append(f'<p><b>You</b><br>{html.escape(prompt)}</p>')
        self.prompt_input.clear()
        self.assistant_buffer = ""
        self.transcript.append("<p><i>Relay AI is thinking…</i></p>")
        self.send_button.setEnabled(False)
        self.worker = ChatWorker(list(self.messages), model)
        self.worker.chunk_received.connect(self.on_chunk)
        self.worker.completed.connect(self.on_completed)
        self.worker.failed.connect(self.on_error)
        self.worker.finished.connect(lambda: self.send_button.setEnabled(True))
        self.worker.start()

    def on_chunk(self, chunk):
        self.assistant_buffer += chunk
        # Re-render the current assistant message as safe HTML with basic Markdown support.
        self.render_transcript()

    def on_completed(self, answer):
        self.messages.append({"role": "assistant", "content": answer})
        self.assistant_buffer = ""
        self.render_transcript()

    def on_error(self, message):
        self.transcript.append(f"<p><b>Connection error:</b> {html.escape(message)}</p>")
        if self.messages and self.messages[-1]["role"] == "user":
            self.messages.pop()

    def render_transcript(self):
        parts = []
        for msg in self.messages:
            role = "You" if msg["role"] == "user" else "Relay AI · Llama"
            parts.append(f"<p><b>{role}</b><br>{self.markdown_to_html(msg['content'])}</p>")
        if self.assistant_buffer:
            parts.append(f"<p><b>Relay AI · Llama</b><br>{self.markdown_to_html(self.assistant_buffer)}</p>")
        self.transcript.setHtml("".join(parts))
        cursor = self.transcript.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.transcript.setTextCursor(cursor)
        self.transcript.ensureCursorVisible()

    @staticmethod
    def markdown_to_html(text):
        # Minimal safe Markdown renderer: fenced code, headings, bullets, bold and inline code.
        escaped = html.escape(text)
        code_blocks = []
        import re
        def replace_code(match):
            code = match.group(1)
            code_blocks.append(f'<pre style="background:#202530;padding:10px;border-radius:8px;white-space:pre-wrap;">{code}</pre>')
            return f"@@CODE{len(code_blocks)-1}@@"
        escaped = re.sub(r"```(?:[a-zA-Z0-9_+-]+)?\\s*\\n(.*?)```", replace_code, escaped, flags=re.S)
        escaped = re.sub(r"^### (.+)$", r"<h3>\\1</h3>", escaped, flags=re.M)
        escaped = re.sub(r"^## (.+)$", r"<h2>\\1</h2>", escaped, flags=re.M)
        escaped = re.sub(r"^# (.+)$", r"<h1>\\1</h1>", escaped, flags=re.M)
        escaped = re.sub(r"\\*\\*(.+?)\\*\\*", r"<b>\\1</b>", escaped)
        escaped = re.sub(r"`([^`]+)`", r"<code>\\1</code>", escaped)
        lines = escaped.splitlines()
        rendered = []
        in_list = False
        for line in lines:
            if re.match(r"^\\s*[-*] ", line):
                if not in_list:
                    rendered.append("<ul>")
                    in_list = True
                rendered.append("<li>" + re.sub(r"^\\s*[-*] ", "", line) + "</li>")
            else:
                if in_list:
                    rendered.append("</ul>")
                    in_list = False
                rendered.append(line + ("<br>" if line else ""))
        if in_list:
            rendered.append("</ul>")
        result = "\\n".join(rendered)
        for i, block in enumerate(code_blocks):
            result = result.replace(f"@@CODE{i}@@", block)
        return result

    def clear_chat(self):
        if self.worker and self.worker.isRunning():
            return
        self.messages.clear()
        self.assistant_buffer = ""
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

    def apply_theme(self):
        self.setStyleSheet(DARK if self.dark else LIGHT)
