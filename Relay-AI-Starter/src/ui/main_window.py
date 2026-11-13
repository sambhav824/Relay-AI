from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton,
    QLabel, QFrame, QStackedWidget, QLineEdit, QTextEdit, QComboBox,
    QGridLayout
)

DARK = """
QMainWindow, QWidget { background: #111318; color: #E8EAF0; font-family: Segoe UI; }
QFrame#Sidebar { background: #191C23; border: 1px solid #292E39; border-radius: 14px; }
QLabel#Brand { font-size: 22px; font-weight: 700; color: #FFFFFF; }
QLabel#Muted { color: #9AA3B2; }
QPushButton { background: #242936; color: #E8EAF0; border: 1px solid #343B4A; border-radius: 9px; padding: 10px 13px; text-align: left; }
QPushButton:hover { background: #30394A; }
QPushButton#Primary { background: #6658E8; border: 0; color: white; font-weight: 600; }
QPushButton#Primary:hover { background: #786CF0; }
QFrame#Panel { background: #191C23; border: 1px solid #292E39; border-radius: 14px; }
QLineEdit, QTextEdit, QComboBox { background: #202530; color: #F2F4F8; border: 1px solid #353D4D; border-radius: 9px; padding: 10px; }
QLabel#PageTitle { font-size: 26px; font-weight: 700; }
"""
LIGHT = """
QMainWindow, QWidget { background: #F5F6FA; color: #202432; font-family: Segoe UI; }
QFrame#Sidebar { background: #FFFFFF; border: 1px solid #E1E4EC; border-radius: 14px; }
QLabel#Brand { font-size: 22px; font-weight: 700; color: #202432; }
QLabel#Muted { color: #687184; }
QPushButton { background: #FFFFFF; color: #202432; border: 1px solid #E1E4EC; border-radius: 9px; padding: 10px 13px; text-align: left; }
QPushButton:hover { background: #EEF0F8; }
QPushButton#Primary { background: #6658E8; border: 0; color: white; font-weight: 600; }
QFrame#Panel { background: #FFFFFF; border: 1px solid #E1E4EC; border-radius: 14px; }
QLineEdit, QTextEdit, QComboBox { background: #FFFFFF; color: #202432; border: 1px solid #D9DEEA; border-radius: 9px; padding: 10px; }
QLabel#PageTitle { font-size: 26px; font-weight: 700; }
"""

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.dark = True
        self.setWindowTitle("Relay AI")
        self.resize(1240, 780)
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
            ("▦   Model Manager", self.simple_page("Model Manager", "Connect cloud providers and discover local Ollama models.")),
            ("◷   Chat History", self.simple_page("Chat History", "Your saved conversations will appear here.")),
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
        content.setContentsMargins(4, 4, 4, 4)
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
        cards = [
            ("Connected providers", "Not connected yet"),
            ("Available local models", "Connect Ollama"),
            ("Recent conversations", "Your chats will appear here"),
        ]
        for i, (title, desc) in enumerate(cards):
            card, _ = self.panel(title, desc)
            grid.addWidget(card, 0, i)
        layout.addLayout(grid)
        welcome, box = self.panel("Welcome to Relay AI", "Connect your first AI provider or configure Ollama to start chatting. Provider setup is the next development milestone.")
        action = QPushButton("Open AI Chat")
        action.setObjectName("Primary")
        action.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        box.addWidget(action)
        layout.addWidget(welcome)
        layout.addStretch()
        self.stack.addWidget(page)
        return page

    def chat_page(self):
        page, layout = self.page_base("AI Chat", "Choose a model and start a conversation.")
        bar = QHBoxLayout()
        provider = QComboBox()
        provider.addItems(["Select a model (not connected)", "Groq", "Gemini", "OpenRouter", "Ollama"])
        bar.addWidget(QLabel("Provider"))
        bar.addWidget(provider, 1)
        layout.addLayout(bar)
        transcript = QTextEdit()
        transcript.setReadOnly(True)
        transcript.setPlaceholderText("Your conversation will appear here once a provider is connected.")
        layout.addWidget(transcript, 1)
        row = QHBoxLayout()
        prompt = QLineEdit()
        prompt.setPlaceholderText("Message Relay AI…")
        send = QPushButton("Send")
        send.setObjectName("Primary")
        send.clicked.connect(lambda: transcript.append("Provider integration will be added in Phase 2."))
        row.addWidget(prompt, 1)
        row.addWidget(send)
        layout.addLayout(row)
        self.stack.addWidget(page)
        return page

    def simple_page(self, title, description):
        page, layout = self.page_base(title, description)
        card, _ = self.panel("Coming in the next phase", description)
        layout.addWidget(card)
        layout.addStretch()
        self.stack.addWidget(page)
        return page

    def settings_page(self):
        page, layout = self.page_base("Settings", "Configure how Relay AI looks and behaves.")
        card, box = self.panel("Appearance", "Use the button at the bottom of the sidebar to switch between dark and light themes.")
        layout.addWidget(card)
        layout.addStretch()
        self.stack.addWidget(page)
        return page

    def toggle_theme(self):
        self.dark = not self.dark
        self.apply_theme()

    def apply_theme(self):
        self.setStyleSheet(DARK if self.dark else LIGHT)
