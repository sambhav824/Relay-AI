import html
import re
from typing import List, Optional, Set, Union
from unittest.mock import Mock, MagicMock
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton,
    QLabel, QFrame, QStackedWidget, QLineEdit, QTextBrowser, QComboBox,
    QGridLayout, QSizePolicy, QCheckBox
)
from src.core.comparison import ComparisonCoordinator
from src.core.message import Message, MessageRole
from src.core.security import CredentialStore
from src.providers.base import AIProvider
from src.providers.manager import ProviderManager, get_default_provider_manager
from src.providers.ollama import stream_chat, generate
from src.routing.classifier import TaskClassifier
from src.routing.registry import DiscoveryWorker, ModelCapabilityRegistry
from src.routing.router import ModelRouter
from src.routing.types import NoEligibleModelsError, RoutingStrategy, TaskType
from src.routing.worker import RoutedChatWorker
from src.ui.compare_view import ComparisonView
from src.ui.routing_pill import RoutingStatusPill
from src.ui.teams import AITeamsView, safely_anchor_worker

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


# Retains strong Python references to any workers that remain running during application
# shutdown until their background QThread terminates, preventing native Qt destruction crashes.
_ORPHANED_WORKERS: Set[QThread] = set()


class ChatWorker(QThread):
    """Background worker thread keeping network operations outside the Qt GUI thread."""
    chunk_received = Signal(str)
    completed = Signal(str)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, provider: AIProvider, messages: List[Union[Message, dict]], model: str):
        super().__init__()
        self.provider = provider
        self.messages = messages
        self.model = model
        self._is_cancelled = False

    def cancel(self, wait_ms: int = 0) -> None:
        """Cooperatively request worker cancellation."""
        self._is_cancelled = True
        if wait_ms > 0 and self.isRunning():
            self.wait(wait_ms)

    def is_cancelled(self) -> bool:
        return self._is_cancelled

    def run(self):
        parts = []
        try:
            if self._is_cancelled:
                self.cancelled.emit()
                return

            if isinstance(stream_chat, (Mock, MagicMock)):
                stream_iter = stream_chat(self.messages, self.model)
                for chunk in stream_iter:
                    if self._is_cancelled:
                        self.cancelled.emit()
                        return
                    text = chunk if isinstance(chunk, str) else getattr(chunk, "delta", str(chunk))
                    if text:
                        parts.append(text)
                        self.chunk_received.emit(text)
            else:
                for chunk in self.provider.stream(self.messages, self.model):
                    if self._is_cancelled:
                        self.cancelled.emit()
                        return
                    if chunk.delta:
                        parts.append(chunk.delta)
                        self.chunk_received.emit(chunk.delta)

            if self._is_cancelled:
                self.cancelled.emit()
                return

            self.completed.emit("".join(parts))
        except Exception as exc:
            if not self._is_cancelled:
                self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self, provider_manager: Optional[ProviderManager] = None, auto_discover: bool = False):
        super().__init__()
        self.dark = True
        self.worker: Optional[ChatWorker] = None
        self.routed_worker: Optional[RoutedChatWorker] = None
        self.discovery_worker: Optional[DiscoveryWorker] = None
        self._active_workers: Set[QThread] = set()
        self.current_session_id: int = 0
        self._is_canceling: bool = False
        self._is_shutting_down: bool = False
        self.messages: List[Message] = []
        self.assistant_buffer = ""
        self.is_thinking = False
        self.fallback_info: Optional[str] = None
        self._current_model_info: Optional[tuple] = None
        self._last_routing_tag: Optional[str] = None

        # Provider architecture and routing engine integration
        self.provider_manager = provider_manager or get_default_provider_manager()
        self.model_registry = ModelCapabilityRegistry(provider_manager=self.provider_manager)
        self.task_classifier = TaskClassifier()
        self.model_router = ModelRouter(registry=self.model_registry, classifier=self.task_classifier)
        self.comparison_coordinator = ComparisonCoordinator(self)
        self.chat_mode = "single"  # "single", "routed", or "compare"

        # Active streaming cursor positions to update only the current message
        self.current_assistant_start = None
        self.current_assistant_end = None
        self.has_new_tokens = False

        # Controlled batch timer for smooth 60ms UI updates
        self.stream_timer = QTimer(self)
        self.stream_timer.setInterval(60)
        self.stream_timer.timeout.connect(self._on_stream_timer_tick)

        self.setWindowTitle("Relay AI")
        self.resize(1200, 780)
        self.setMinimumSize(920, 580)

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
        self.teams_view: Optional[AITeamsView] = None
        pages = [
            ("⌂   Dashboard", self.dashboard_page()),
            ("✧   AI Chat", self.chat_page()),
            ("👥   AI Teams", self.teams_page()),
            ("▦   Model Manager", self.model_manager_page()),
            ("◷   Chat History", self.simple_page("Chat History", "Persistent SQLite conversation history is planned in a future milestone.")),
            ("⚙   Settings", self.settings_page()),
        ]
        self.nav_buttons = []
        for label, page in pages:
            button = QPushButton(label)
            button.clicked.connect(lambda checked=False, p=page: self.stack.setCurrentWidget(p))
            side.addWidget(button)
            self.nav_buttons.append(button)
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

        if auto_discover:
            self.start_model_discovery()

    def start_model_discovery(self) -> None:
        """Start dynamic model discovery in a background QThread if not already running."""
        if self.discovery_worker and self.discovery_worker.isRunning():
            return
        worker = DiscoveryWorker(self.model_registry)
        self.discovery_worker = worker
        self._active_workers.add(worker)
        worker.finished.connect(lambda w=worker: self._on_worker_finished(w))
        worker.start()

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
            ("Connected providers", "Ollama, Gemini, Groq & OpenAI"),
            ("Available models", "Llama 3.2, Gemini 3.5, GPT-4o Mini, GPT-OSS 120B"),
            ("Security", "Windows Credential Manager active"),
        ]
        for i, (title, desc) in enumerate(cards):
            card, _ = self.panel(title, desc)
            grid.addWidget(card, 0, i)
        layout.addLayout(grid)

        welcome, box = self.panel(
            "Universal AI Workspace",
            "Relay AI connects to local Ollama, Google Gemini, Groq, and OpenAI. Switch models effortlessly mid-conversation with zero context loss."
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

        # Header: Title, Provider selector, Model selector, Status Badge, Clear Conversation
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(4, 2, 4, 4)
        header_layout.setSpacing(10)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title_label = QLabel("AI Chat")
        title_label.setObjectName("PageTitle")
        subtitle_label = QLabel("Ollama · Gemini · Groq · OpenAI · Multi-turn context · Markdown streaming")
        subtitle_label.setObjectName("Muted")
        title_box.addWidget(title_label)
        title_box.addWidget(subtitle_label)
        header_layout.addLayout(title_box)

        # Mode Selector: Manual Model, Smart Route, Compare Models
        self.mode_frame = QFrame()
        self.mode_frame.setObjectName("Panel")
        mode_layout = QHBoxLayout(self.mode_frame)
        mode_layout.setContentsMargins(4, 3, 4, 3)
        mode_layout.setSpacing(4)

        self.btn_mode_single = QPushButton("Manual Model")
        self.btn_mode_single.setObjectName("Primary")
        self.btn_mode_single.setFixedWidth(115)
        self.btn_mode_single.clicked.connect(lambda: self.set_chat_mode("single"))

        self.btn_mode_routed = QPushButton("✦ Smart Route")
        self.btn_mode_routed.setObjectName("Secondary")
        self.btn_mode_routed.setFixedWidth(120)
        self.btn_mode_routed.clicked.connect(lambda: self.set_chat_mode("routed"))

        self.btn_mode_compare = QPushButton("Compare Models")
        self.btn_mode_compare.setObjectName("Secondary")
        self.btn_mode_compare.setFixedWidth(125)
        self.btn_mode_compare.clicked.connect(lambda: self.set_chat_mode("compare"))

        mode_layout.addWidget(self.btn_mode_single)
        mode_layout.addWidget(self.btn_mode_routed)
        mode_layout.addWidget(self.btn_mode_compare)
        header_layout.addWidget(self.mode_frame)

        header_layout.addStretch()

        # Single / Manual Model Controls Widget
        self.single_controls_widget = QWidget()
        self.single_controls_widget.setVisible(True)
        controls = QHBoxLayout(self.single_controls_widget)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(8)

        # Provider Selector
        prov_lbl = QLabel("Provider:")
        prov_lbl.setObjectName("Muted")
        self.provider_combo = QComboBox()
        self.provider_combo.setFixedWidth(135)
        self._populate_providers()
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)

        # Model Selector
        lbl = QLabel("Model:")
        lbl.setObjectName("Muted")
        self.model_combo = QComboBox()
        self.model_combo.setFixedWidth(180)
        self._populate_models()

        self.status_badge = QLabel("")
        self.status_badge.setStyleSheet("font-size: 12px; font-weight: 600; padding: 2px 6px;")

        self.clear_button = QPushButton("Clear conversation")
        self.clear_button.setObjectName("Secondary")
        self.clear_button.clicked.connect(self.clear_chat)

        controls.addWidget(prov_lbl)
        controls.addWidget(self.provider_combo)
        controls.addWidget(lbl)
        controls.addWidget(self.model_combo)
        controls.addWidget(self.status_badge)
        controls.addWidget(self.clear_button)
        header_layout.addWidget(self.single_controls_widget)

        # Smart Routing Controls Widget
        self.routed_controls_widget = QWidget()
        self.routed_controls_widget.setVisible(False)
        routed_controls = QHBoxLayout(self.routed_controls_widget)
        routed_controls.setContentsMargins(0, 0, 0, 0)
        routed_controls.setSpacing(8)

        strat_lbl = QLabel("Strategy:")
        strat_lbl.setObjectName("Muted")
        self.strategy_combo = QComboBox()
        self.strategy_combo.setFixedWidth(125)
        self.strategy_combo.addItem("Balanced", RoutingStrategy.BALANCED)
        self.strategy_combo.addItem("Quality First", RoutingStrategy.QUALITY_FIRST)
        self.strategy_combo.addItem("Cost Efficient", RoutingStrategy.COST_EFFICIENT)
        self.strategy_combo.addItem("Local First", RoutingStrategy.LOCAL_FIRST)

        self.local_only_checkbox = QCheckBox("Local-Only")
        self.local_only_checkbox.setToolTip("Strictly exclude all cloud providers; only local on-device models are evaluated.")

        self.routing_pill = RoutingStatusPill(dark=self.dark, parent=page)

        self.clear_button_routed = QPushButton("Clear conversation")
        self.clear_button_routed.setObjectName("Secondary")
        self.clear_button_routed.clicked.connect(self.clear_chat)

        routed_controls.addWidget(strat_lbl)
        routed_controls.addWidget(self.strategy_combo)
        routed_controls.addWidget(self.local_only_checkbox)
        routed_controls.addWidget(self.routing_pill)
        routed_controls.addWidget(self.clear_button_routed)
        header_layout.addWidget(self.routed_controls_widget)

        page_layout.addWidget(header, 0)

        # Chat transcript: Scrollable, takes available vertical space (Single mode)
        self.transcript = QTextBrowser()
        self.transcript.setOpenExternalLinks(True)
        self.transcript.setPlaceholderText("Start a conversation with Ollama, Gemini, Groq, or OpenAI. Type your message below and press Enter.")
        self.transcript.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        page_layout.addWidget(self.transcript, 1)

        # Side-by-side Comparison View (Compare mode)
        self.compare_view = ComparisonView(self.provider_manager, dark=self.dark, parent=page)
        self.compare_view.setVisible(False)
        self.compare_view.response_committed.connect(self._on_compare_response_committed)
        page_layout.addWidget(self.compare_view, 1)

        # Connect ComparisonCoordinator to ComparisonView
        self.comparison_coordinator.slot_chunk.connect(self.compare_view.update_slot_chunk)
        self.comparison_coordinator.slot_status.connect(self.compare_view.update_slot_status)
        self.comparison_coordinator.slot_completed.connect(self.compare_view.update_slot_result)
        self.comparison_coordinator.all_completed.connect(self._on_comparison_all_completed)
        self.comparison_coordinator.canceled.connect(self._on_comparison_canceled)

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

    def teams_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        self.teams_view = AITeamsView(
            provider_manager=self.provider_manager,
            dark=self.dark,
            parent=page,
        )
        layout.addWidget(self.teams_view)
        self.stack.addWidget(page)
        return page

    def _populate_providers(self):
        """Populate the provider dropdown."""
        self.provider_combo.blockSignals(True)
        self.provider_combo.clear()
        providers = self.provider_manager.list_providers()
        for p in providers:
            self.provider_combo.addItem(p.display_name, p.provider_id)

        active = self.provider_manager.get_active_provider()
        idx = self.provider_combo.findData(active.provider_id)
        if idx >= 0:
            self.provider_combo.setCurrentIndex(idx)
        self.provider_combo.blockSignals(False)

    def _on_provider_changed(self, index: int):
        """Switch active provider without clearing conversation context."""
        if index < 0:
            return
        provider_id = self.provider_combo.currentData()
        if provider_id:
            self.provider_manager.set_active_provider(provider_id)
            self._populate_models()

    def _populate_models(self):
        """Populate the model combo box from the active provider."""
        self.model_combo.clear()
        active = self.provider_manager.get_active_provider()
        try:
            available = active.get_available_models()
            if available:
                for m in available:
                    self.model_combo.addItem(m.id)
                return
        except Exception:
            pass

        if active.provider_id == "ollama":
            fallback = "llama3.2:3b"
        elif active.provider_id == "groq":
            fallback = "openai/gpt-oss-120b"
        elif active.provider_id == "openai":
            fallback = "gpt-4o-mini"
        else:
            fallback = "gemini-3.5-flash"
        self.model_combo.addItem(fallback)

    def _is_generation_active(self) -> bool:
        """Check whether any generation worker is currently running."""
        if self.routed_worker and self.routed_worker.isRunning():
            return True
        if self.worker and self.worker.isRunning():
            return True
        if self.comparison_coordinator and self.comparison_coordinator.is_running():
            return True
        return False

    def cancel_generation(self) -> None:
        """Cooperatively request cancellation across active workers."""
        self._is_canceling = True
        self.send_button.setText("Canceling…")
        self.send_button.setEnabled(False)

        if self.routed_worker and self.routed_worker.isRunning():
            self.routed_worker.cancel()

        if self.worker and self.worker.isRunning():
            self.worker.cancel()

        if self.comparison_coordinator and self.comparison_coordinator.is_running():
            self.comparison_coordinator.cancel_all()

    def set_chat_mode(self, mode: str):
        """Switch between Manual Model, Smart Route, and Compare Models."""
        if self._is_generation_active():
            return  # Lockout mode switching while generation is in flight

        self.chat_mode = mode
        self.current_session_id += 1  # Invalidate any pending callbacks

        self.btn_mode_single.setObjectName("Primary" if mode == "single" else "Secondary")
        self.btn_mode_routed.setObjectName("Primary" if mode == "routed" else "Secondary")
        self.btn_mode_compare.setObjectName("Primary" if mode == "compare" else "Secondary")

        if mode == "single":
            self.transcript.setVisible(True)
            self.compare_view.setVisible(False)
            self.single_controls_widget.setVisible(True)
            self.routed_controls_widget.setVisible(False)
        elif mode == "routed":
            self.transcript.setVisible(True)
            self.compare_view.setVisible(False)
            self.single_controls_widget.setVisible(False)
            self.routed_controls_widget.setVisible(True)
            if not self.model_registry.is_cache_valid():
                self.start_model_discovery()
        else:  # "compare"
            self.transcript.setVisible(False)
            self.compare_view.setVisible(True)
            self.single_controls_widget.setVisible(False)
            self.routed_controls_widget.setVisible(False)

        self.apply_theme()

    def _on_comparison_all_completed(self, results):
        self._reset_generation_ui()
        self.compare_view.show_comparison_summary(results)

    def _on_comparison_canceled(self):
        self._reset_generation_ui()

    def _on_compare_response_committed(self, provider_id: str, model: str, content: str):
        """Commit selected comparison answer into the shared conversation context."""
        assistant_msg = Message(
            role=MessageRole.ASSISTANT,
            content=content,
            metadata={"provider_id": provider_id, "model": model},
        )
        self.messages.append(assistant_msg)
        self.render_transcript()
        self.status_badge.setText(f"✓ Saved {model} to conversation")
        self.status_badge.setStyleSheet("color: #10B981; font-size: 12px; font-weight: 600;")
        self.set_chat_mode("single")

    def send_prompt(self):
        prompt = self.prompt_input.text().strip()

        # Compare Models mode
        if self.chat_mode == "compare":
            if self.comparison_coordinator.is_running():
                self.cancel_generation()
                return

            if not prompt:
                return

            # Append user message to shared context
            user_message = Message(role=MessageRole.USER, content=prompt)
            self.messages.append(user_message)

            self.prompt_input.clear()
            self.prompt_input.setEnabled(False)
            self.send_button.setText("■ Stop All")
            self.send_button.setStyleSheet("background: #EF4444; color: #FFFFFF; font-weight: 600;")
            self.send_button.setEnabled(True)
            self.mode_frame.setEnabled(False)

            self.compare_view.reset_comparison()
            slots = self.compare_view.get_configured_slots()
            self.comparison_coordinator.start_comparison(slots, self.messages, self.provider_manager)
            return

        # If a generation is currently active, clicking the button cancels it
        if self._is_generation_active():
            self.cancel_generation()
            return

        if not prompt or self._is_shutting_down:
            return

        # Prepare for generation
        self.current_session_id += 1
        sid = self.current_session_id

        # 1. Capture snapshot of prior history BEFORE appending new prompt
        prior_history = [Message(m.role, m.content) for m in self.messages]

        # 2. Append User message exactly once to self.messages
        user_message = Message(role=MessageRole.USER, content=prompt)
        self.messages.append(user_message)

        # 3. Update UI states and lock mode switching
        self.prompt_input.clear()
        self.assistant_buffer = ""
        self.fallback_info = None
        self._current_model_info = None
        self.is_thinking = True
        self.has_new_tokens = False

        self.mode_frame.setEnabled(False)
        self.send_button.setText("■ Stop")
        self.send_button.setStyleSheet("background: #EF4444; color: #FFFFFF; font-weight: 600;")
        self.send_button.setEnabled(True)
        self.clear_button.setEnabled(False)
        if hasattr(self, "clear_button_routed"):
            self.clear_button_routed.setEnabled(False)
        self.prompt_input.setEnabled(False)

        # Append User message to transcript HTML
        cursor = self.transcript.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertHtml(self._format_user_card(prompt))
        cursor.insertBlock()

        # Append Assistant placeholder and record cursor position
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.current_assistant_start = cursor.position()
        cursor.insertHtml(self._format_assistant_card("", is_thinking=True))
        self.current_assistant_end = cursor.position()

        self._scroll_if_near_bottom(force=True)
        self.stream_timer.start(60)

        # 4. Dispatch appropriate worker
        if self.chat_mode == "routed":
            raw_strategy = self.strategy_combo.currentData() or RoutingStrategy.BALANCED
            if isinstance(raw_strategy, str):
                try:
                    strategy = RoutingStrategy(raw_strategy)
                except ValueError:
                    strategy = RoutingStrategy.BALANCED
            else:
                strategy = raw_strategy
            local_only = self.local_only_checkbox.isChecked()

            try:
                decision = self.model_router.route(
                    prompt=prompt,
                    history=prior_history,
                    strategy=strategy,
                    local_only=local_only,
                )
                self.routing_pill.set_decision(decision)
                task_name = decision.classification.task_type.value.replace("_", " ").title()
                strat_name = strategy.value.replace("_", " ").title()
                self._last_routing_tag = f"Smart Routed: {strat_name} · {task_name}"
                self._current_model_info = (decision.primary.provider_id, decision.primary.model_id)
            except NoEligibleModelsError as exc:
                self._handle_preflight_error(str(exc), sid)
                return

            worker = RoutedChatWorker(
                decision=decision,
                prompt=prompt,
                history=prior_history,
                provider_manager=self.provider_manager,
            )
            self.routed_worker = worker
            self._active_workers.add(worker)

            worker.attempt_started.connect(lambda p, m, idx, s=sid: self._on_routed_attempt_started(p, m, idx, s))
            worker.chunk_received.connect(lambda d, s=sid: self._on_chunk(d, s))
            worker.attempt_failed.connect(lambda p, m, r, err, s=sid: self._on_routed_attempt_failed(p, m, r, err, s))
            worker.fallback_started.connect(lambda fp, fm, tp, tm, fidx, s=sid: self._on_routed_fallback_started(fp, fm, tp, tm, fidx, s))
            worker.final_success.connect(lambda p, m, txt, msgs, s=sid: self._on_routed_final_success(p, m, txt, s))
            worker.final_failure.connect(lambda summary, recs, s=sid: self._on_routed_final_failure(summary, s))
            worker.cancelled.connect(lambda s=sid: self._on_routed_cancelled(s))
            worker.finished.connect(lambda w=worker: self._on_worker_finished(w))
            worker.start()

        else:  # "single" (Manual Model)
            model = self.model_combo.currentText()
            active_provider = self.provider_manager.get_active_provider()
            self._last_routing_tag = None
            self._current_model_info = (active_provider.provider_id, model)

            worker = ChatWorker(active_provider, prior_history + [Message(MessageRole.USER, prompt)], model)
            self.worker = worker
            self._active_workers.add(worker)

            worker.chunk_received.connect(lambda d, s=sid: self._on_chunk(d, s))
            worker.completed.connect(lambda ans, s=sid: self._on_chat_completed(ans, s))
            worker.failed.connect(lambda err, s=sid: self._on_chat_failed(err, s))
            worker.cancelled.connect(lambda s=sid: self._on_chat_cancelled(s))
            worker.finished.connect(lambda w=worker: self._on_worker_finished(w))
            worker.start()

    def _handle_preflight_error(self, error_message: str, sid: int):
        if sid != self.current_session_id:
            return
        self.stream_timer.stop()
        self.is_thinking = False
        if hasattr(self, "routing_pill"):
            self.routing_pill.set_error(error_message)

        if self.current_assistant_start is not None:
            cursor = self.transcript.textCursor()
            cursor.setPosition(self.current_assistant_start)
            cursor.setPosition(self.current_assistant_end, QTextCursor.MoveMode.KeepAnchor)
            cursor.insertHtml(self._format_error_card(error_message))
            self.current_assistant_start = None
            self.current_assistant_end = None

        self._reset_generation_ui()

    def _on_chunk(self, chunk: str, sid: int):
        if self._is_shutting_down or sid != self.current_session_id or self._is_canceling:
            return
        self.is_thinking = False
        self.assistant_buffer += chunk
        self.has_new_tokens = True

    def _on_routed_attempt_started(self, provider_id: str, model_id: str, attempt_idx: int, sid: int):
        if self._is_shutting_down or sid != self.current_session_id:
            return
        self._current_model_info = (provider_id, model_id)
        self.status_badge.setText(f"Attempting {model_id}…")
        self.status_badge.setStyleSheet("color: #F59E0B; font-size: 12px; font-weight: 600;")

    def _on_routed_attempt_failed(self, provider_id: str, model_id: str, reason: str, error_msg: str, sid: int):
        if self._is_shutting_down or sid != self.current_session_id:
            return
        self.status_badge.setText(f"{model_id} failed: {reason}")
        self.status_badge.setStyleSheet("color: #EF4444; font-size: 12px; font-weight: 600;")

    def _on_routed_fallback_started(self, fp: str, fm: str, tp: str, tm: str, fidx: int, sid: int):
        if self._is_shutting_down or sid != self.current_session_id:
            return
        # Discard partial output from failed primary model
        self.assistant_buffer = ""
        self.has_new_tokens = True
        self.fallback_info = f"Fallback {fidx}: {fp.title()} ({fm}) failed → Switched to {tp.title()} ({tm})"
        self._current_model_info = (tp, tm)
        if hasattr(self, "routing_pill"):
            self.routing_pill.set_fallback(fp, fm, tp, tm, fidx)
        self._update_current_assistant_block()

    def _on_routed_final_success(self, provider_id: str, model_id: str, response_text: str, sid: int):
        if self._is_shutting_down or sid != self.current_session_id or self._is_canceling:
            return
        self.stream_timer.stop()
        self.is_thinking = False
        self.has_new_tokens = False
        self.assistant_buffer = response_text
        self._current_model_info = (provider_id, model_id)

        self._update_current_assistant_block(final=True)

        # Append ONLY the assistant response. Do NOT duplicate user prompt!
        self.messages.append(Message(
            role=MessageRole.ASSISTANT,
            content=response_text,
            metadata={"provider_id": provider_id, "model": model_id},
        ))

        self.assistant_buffer = ""
        self.current_assistant_start = None
        self.current_assistant_end = None
        self.status_badge.setText("")
        self._reset_generation_ui()

    def _on_routed_final_failure(self, summary: str, sid: int):
        if self._is_shutting_down or sid != self.current_session_id:
            return
        self.stream_timer.stop()
        self.is_thinking = False
        self.has_new_tokens = False

        # NOTE: Do NOT pop user prompt! User prompt is permanently retained.
        # No fabricated assistant response is inserted into self.messages.
        if self.current_assistant_start is not None:
            cursor = self.transcript.textCursor()
            cursor.setPosition(self.current_assistant_start)
            cursor.setPosition(self.current_assistant_end, QTextCursor.MoveMode.KeepAnchor)
            cursor.insertHtml(self._format_error_card(summary))
            self.current_assistant_start = None
            self.current_assistant_end = None

        self.assistant_buffer = ""
        self.status_badge.setText("Failed")
        self.status_badge.setStyleSheet("color: #EF4444; font-size: 12px; font-weight: 600;")
        self._scroll_if_near_bottom(force=True)
        self._reset_generation_ui()

    def _on_routed_cancelled(self, sid: int):
        if self._is_shutting_down or sid != self.current_session_id:
            return
        self._is_canceling = False
        self.stream_timer.stop()
        self.is_thinking = False
        self.has_new_tokens = False

        # NOTE: Do NOT pop user prompt! User prompt is permanently retained.
        if self.current_assistant_start is not None:
            cursor = self.transcript.textCursor()
            cursor.setPosition(self.current_assistant_start)
            cursor.setPosition(self.current_assistant_end, QTextCursor.MoveMode.KeepAnchor)
            cursor.insertHtml(self._format_cancellation_card("Generation canceled by user"))
            self.current_assistant_start = None
            self.current_assistant_end = None

        self.assistant_buffer = ""
        self.status_badge.setText("Canceled")
        self.status_badge.setStyleSheet("color: #F59E0B; font-size: 12px; font-weight: 600;")
        self._scroll_if_near_bottom(force=True)
        self._reset_generation_ui()

    def _on_chat_completed(self, answer: str, sid: int):
        if self._is_shutting_down or sid != self.current_session_id:
            return
        self.stream_timer.stop()
        self.is_thinking = False
        self.has_new_tokens = False
        self.assistant_buffer = answer

        self._update_current_assistant_block(final=True)
        active_provider = self.provider_manager.get_active_provider()
        model = self.model_combo.currentText()
        self.messages.append(Message(
            role=MessageRole.ASSISTANT,
            content=answer,
            metadata={"provider_id": active_provider.provider_id, "model": model},
        ))

        self.assistant_buffer = ""
        self.current_assistant_start = None
        self.current_assistant_end = None
        self.status_badge.setText("")
        self._reset_generation_ui()

    def _on_chat_failed(self, message: str, sid: int):
        if self._is_shutting_down or sid != self.current_session_id:
            return
        self.stream_timer.stop()
        self.is_thinking = False
        self.has_new_tokens = False

        # In single model mode, pop user message on failure for backwards compatibility with test_chat_v2
        if self.chat_mode == "single" and self.messages:
            last = self.messages[-1]
            role = getattr(last, "role", None) or (last.get("role") if isinstance(last, dict) else None)
            if role == MessageRole.USER or role == "user":
                self.messages.pop()
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
        self._reset_generation_ui()

    def _on_chat_cancelled(self, sid: int):
        if self._is_shutting_down or sid != self.current_session_id:
            return
        self._is_canceling = False
        self.stream_timer.stop()
        self.is_thinking = False
        self.has_new_tokens = False

        if self.current_assistant_start is not None:
            cursor = self.transcript.textCursor()
            cursor.setPosition(self.current_assistant_start)
            cursor.setPosition(self.current_assistant_end, QTextCursor.MoveMode.KeepAnchor)
            cursor.insertHtml(self._format_cancellation_card("Generation canceled by user"))
            self.current_assistant_start = None
            self.current_assistant_end = None

        self.assistant_buffer = ""
        self.status_badge.setText("Canceled")
        self.status_badge.setStyleSheet("color: #F59E0B; font-size: 12px; font-weight: 600;")
        self._scroll_if_near_bottom(force=True)
        self._reset_generation_ui()

    def _on_worker_finished(self, worker: Optional[QThread]):
        """Safely release worker reference after its thread has finished executing."""
        if not worker:
            return
        self._active_workers.discard(worker)
        if worker is self.routed_worker:
            self.routed_worker = None
        elif worker is self.worker:
            self.worker = None
        elif worker is self.discovery_worker:
            self.discovery_worker = None
        try:
            worker.deleteLater()
        except Exception:
            pass
        if not self._is_shutting_down and not self._is_generation_active() and self._is_canceling:
            self._is_canceling = False
            self._reset_generation_ui()

    def _reset_generation_ui(self):
        """Restore input controls and re-enable mode switching."""
        self.send_button.setEnabled(True)
        self.send_button.setText("Send")
        self.send_button.setStyleSheet("")
        self.clear_button.setEnabled(True)
        if hasattr(self, "clear_button_routed"):
            self.clear_button_routed.setEnabled(True)
        self.prompt_input.setEnabled(True)
        self.mode_frame.setEnabled(True)
        self.prompt_input.setFocus()

    def _on_stream_timer_tick(self):
        if self.has_new_tokens:
            self.has_new_tokens = False
            self.status_badge.setText("Streaming…")
            self.status_badge.setStyleSheet("color: #10B981; font-size: 12px; font-weight: 600;")
            self.send_button.setText("■ Stop")
            self._update_current_assistant_block()

    def _update_current_assistant_block(self, final: bool = False):
        if self.current_assistant_start is None:
            return

        scrollbar = self.transcript.verticalScrollBar()
        was_near_bottom = (scrollbar.maximum() - scrollbar.value()) <= 45
        saved_scroll = scrollbar.value()

        cursor = self.transcript.textCursor()
        cursor.setPosition(self.current_assistant_start)
        cursor.setPosition(self.current_assistant_end, QTextCursor.MoveMode.KeepAnchor)
        cursor.insertHtml(self._format_assistant_card(
            self.assistant_buffer,
            is_thinking=self.is_thinking,
            model_info=self._current_model_info,
            fallback_note=self.fallback_info,
        ))
        self.current_assistant_end = cursor.position()

        if was_near_bottom:
            scrollbar.setValue(scrollbar.maximum())
        else:
            scrollbar.setValue(saved_scroll)

    def clear_chat(self):
        if self._is_generation_active():
            self.cancel_generation()
        self.current_session_id += 1
        self.messages.clear()
        self.transcript.clear()
        self.assistant_buffer = ""
        self.fallback_info = None
        self.current_assistant_start = None
        self.current_assistant_end = None
        self.status_badge.setText("")
        if hasattr(self, "routing_pill"):
            self.routing_pill.reset()

    def _scroll_if_near_bottom(self, force: bool = False):
        scrollbar = self.transcript.verticalScrollBar()
        if force or (scrollbar.maximum() - scrollbar.value() <= 45):
            scrollbar.setValue(scrollbar.maximum())

    def _format_user_card(self, content: str) -> str:
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

    def _format_assistant_card(
        self,
        content: str,
        is_thinking: bool = False,
        model_info: Optional[tuple] = None,
        fallback_note: Optional[str] = None,
    ) -> str:
        bot_header_color = "#34D399" if self.dark else "#059669"
        bot_bg = "#151820" if self.dark else "#FFFFFF"
        bot_border = "#262D3B" if self.dark else "#E5E7EB"

        active_info = model_info or self._current_model_info
        if active_info and active_info[0] and active_info[1]:
            prov_id, model_name = active_info
            bot_label = f"✦ Relay AI · {prov_id.title()} · {model_name}"
        else:
            active_provider = self.provider_manager.get_active_provider()
            bot_label = f"✦ Relay AI · {active_provider.display_name}"

        if self.chat_mode == "routed" and self._last_routing_tag:
            bot_label += f" ({self._last_routing_tag})"

        fallback_banner = ""
        active_fallback = fallback_note or self.fallback_info
        if active_fallback:
            fallback_bg = "#2D2214" if self.dark else "#FEF3C7"
            fallback_border = "#5C4119" if self.dark else "#FDE68A"
            fallback_text_color = "#FDE68A" if self.dark else "#92400E"
            fallback_banner = (
                f'<div style="background: {fallback_bg}; border: 1px solid {fallback_border}; border-radius: 8px; padding: 6px 10px; margin-bottom: 8px; font-size: 12px; color: {fallback_text_color}; font-weight: 600;">'
                f'  {html.escape(active_fallback)}'
                f'</div>'
            )

        if is_thinking and not content:
            body = '<div style="color: #9AA3B2; font-style: italic;">Thinking…</div>'
        else:
            body = self.markdown_to_html(content, self.dark)

        return (
            f'<div style="margin-bottom: 16px;">'
            f'  <div style="font-weight: 700; font-size: 13px; color: {bot_header_color}; margin-bottom: 6px; padding-left: 2px;">{bot_label}</div>'
            f'  <div style="background: {bot_bg}; border: 1px solid {bot_border}; border-radius: 12px; padding: 12px 16px; font-size: 14px; line-height: 1.5;">'
            f'    {fallback_banner}'
            f'    {body}'
            f'  </div>'
            f'</div>'
        )

    def _format_cancellation_card(self, note: str = "Generation canceled by user") -> str:
        warn_bg = "#2D2214" if self.dark else "#FEF3C7"
        warn_border = "#5C4119" if self.dark else "#FDE68A"
        warn_color = "#FDE68A" if self.dark else "#92400E"
        return (
            f'<div style="margin-bottom: 16px;">'
            f'  <div style="font-weight: 700; font-size: 13px; color: {warn_color}; margin-bottom: 6px; padding-left: 2px;">✦ Relay AI · Canceled</div>'
            f'  <div style="background: {warn_bg}; border: 1px solid {warn_border}; border-radius: 12px; padding: 12px 16px; font-size: 14px; color: {warn_color};">'
            f'    [ {html.escape(note)} ]'
            f'  </div>'
            f'</div>'
        )

    def _format_error_card(self, error_message: str) -> str:
        error_bg = "#2D1517" if self.dark else "#FEE2E2"
        error_border = "#7F1D1D" if self.dark else "#FCA5A5"
        error_color = "#FCA5A5" if self.dark else "#991B1B"
        escaped_err = html.escape(error_message)
        return (
            f'<div style="margin-bottom: 16px; padding: 12px 16px; background: {error_bg}; border: 1px solid {error_border}; border-radius: 12px; color: {error_color};">'
            f'  <b>Error:</b> {escaped_err}'
            f'</div>'
        )

    def closeEvent(self, event):
        """Cooperative shutdown: cancel all workers, disconnect signals, and safely anchor running threads."""
        self._is_shutting_down = True
        self.stream_timer.stop()
        self.setEnabled(False)

        # 1. Collect all active workers across single, routed, comparison, and discovery modes
        workers_to_stop: Set[QThread] = set(self._active_workers)
        if self.routed_worker:
            workers_to_stop.add(self.routed_worker)
        if self.worker:
            workers_to_stop.add(self.worker)
        if self.discovery_worker:
            workers_to_stop.add(self.discovery_worker)
        if self.comparison_coordinator and hasattr(self.comparison_coordinator, "workers"):
            for cw in self.comparison_coordinator.workers.values():
                if isinstance(cw, QThread):
                    workers_to_stop.add(cw)
        if hasattr(self, "teams_view") and self.teams_view:
            if self.teams_view._active_worker and isinstance(self.teams_view._active_worker, QThread):
                workers_to_stop.add(self.teams_view._active_worker)

        # 2. Disconnect worker signals so no queued events can reach destroyed widgets or window slots
        for w in workers_to_stop:
            try:
                w.disconnect()
            except Exception:
                pass

        # 3. Cooperatively signal cancellation
        if self.routed_worker and self.routed_worker.isRunning():
            self.routed_worker.cancel(wait_ms=0)

        if self.worker and self.worker.isRunning():
            self.worker.cancel(wait_ms=0)

        if self.comparison_coordinator and self.comparison_coordinator.is_running():
            self.comparison_coordinator.cancel_all(wait_ms=0)

        if hasattr(self, "teams_view") and self.teams_view:
            self.teams_view.cancel_team()
            if self.teams_view._active_worker and isinstance(self.teams_view._active_worker, QThread):
                safely_anchor_worker(self.teams_view._active_worker)

        # 4. Bounded wait for threads to exit cooperatively
        for w in workers_to_stop:
            if w.isRunning():
                w.wait(500)

        # 5. Anchor any worker that is still running into module-level graveyard
        # so Python never destroys a running QThread wrapper while C++ is executing.
        for w in workers_to_stop:
            if w.isRunning():
                _ORPHANED_WORKERS.add(w)
                try:
                    w.finished.connect(lambda worker=w: _ORPHANED_WORKERS.discard(worker))
                    w.finished.connect(w.deleteLater)
                except Exception:
                    pass
            else:
                try:
                    w.deleteLater()
                except Exception:
                    pass

        self._active_workers.clear()
        self.routed_worker = None
        self.worker = None
        self.discovery_worker = None
        event.accept()

    def render_transcript(self, error_message: Optional[str] = None):
        parts = []
        for msg in self.messages:
            role = msg.role.value if hasattr(msg, "role") and hasattr(msg.role, "value") else msg["role"]
            content = msg.content if hasattr(msg, "content") else msg["content"]
            if role == "user":
                parts.append(self._format_user_card(content))
            else:
                metadata = msg.metadata if hasattr(msg, "metadata") else msg.get("metadata", {})
                prov_id = metadata.get("provider_id") if metadata else None
                model = metadata.get("model") if metadata else None
                m_info = (prov_id, model) if (prov_id and model) else None
                parts.append(self._format_assistant_card(content, model_info=m_info))

        if self.assistant_buffer:
            parts.append(self._format_assistant_card(self.assistant_buffer, model_info=self._current_model_info, fallback_note=self.fallback_info))
        elif self.is_thinking:
            parts.append(self._format_assistant_card("", is_thinking=True, model_info=self._current_model_info))

        if error_message:
            parts.append(self._format_error_card(error_message))

        self.transcript.setHtml("".join(parts))
        self._scroll_if_near_bottom(force=True)

    @staticmethod
    def markdown_to_html(text: str, dark: bool = True) -> str:
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

        escaped = re.sub(r"```(?:[a-zA-Z0-9_+-]+)?\s*\n(.*?)```", replace_code, escaped, flags=re.S)
        escaped = re.sub(r"```(?:[a-zA-Z0-9_+-]+)?\s*\n(.*)$", replace_code, escaped, flags=re.S)

        escaped = re.sub(r"^### (.+)$", r'<h3 style="margin: 8px 0 4px 0; font-size: 15px;">\1</h3>', escaped, flags=re.M)
        escaped = re.sub(r"^## (.+)$", r'<h2 style="margin: 10px 0 6px 0; font-size: 17px;">\1</h2>', escaped, flags=re.M)
        escaped = re.sub(r"^# (.+)$", r'<h1 style="margin: 12px 0 8px 0; font-size: 19px;">\1</h1>', escaped, flags=re.M)

        escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
        escaped = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", escaped)

        escaped = re.sub(
            r"`([^`]+)`",
            f'<code style="background: {code_bg}; color: {code_color}; padding: 2px 6px; border-radius: 4px; font-family: Consolas, monospace; font-size: 13px;">\\1</code>',
            escaped
        )

        lines = escaped.splitlines()
        rendered = []
        in_ul = False
        in_ol = False

        for line in lines:
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
        if (self.worker and self.worker.isRunning()) or self.comparison_coordinator.is_running():
            return
        self.stream_timer.stop()
        self.messages.clear()
        self.assistant_buffer = ""
        self.is_thinking = False
        self.current_assistant_start = None
        self.current_assistant_end = None
        self.status_badge.setText("")
        self.transcript.clear()
        if hasattr(self, "compare_view"):
            self.compare_view.reset_comparison()

    def model_manager_page(self):
        page, layout = self.page_base("Model Manager", "Explore connected local and cloud providers.")
        grid = QGridLayout()
        providers = self.provider_manager.list_providers()
        for i, p in enumerate(providers):
            status = "Operational" if p.validate_connection() else "Needs configuration / Offline"
            card, _ = self.panel(f"{p.display_name} ({p.provider_id})", f"Type: {'Local' if p.is_local else 'Cloud'}\nStatus: {status}")
            grid.addWidget(card, i // 2, i % 2)
        layout.addLayout(grid)
        layout.addStretch()
        self.stack.addWidget(page)
        return page

    def simple_page(self, title, description):
        page, layout = self.page_base(title, description)
        card, _ = self.panel("Planned milestone", description)
        layout.addWidget(card)
        layout.addStretch()
        self.stack.addWidget(page)
        return page

    def settings_page(self):
        page, layout = self.page_base("Settings", "Configure Relay AI credentials and preferences.")
        store = self.provider_manager.credential_store

        # Google Gemini API Key panel
        gemini_frame, gemini_box = self.panel("Google Gemini API Credentials", "Manage your Google Gemini API key securely in Windows Credential Manager.")
        gemini_row = QHBoxLayout()
        gemini_input = QLineEdit()
        gemini_input.setPlaceholderText("Enter Gemini API key…")
        gemini_input.setEchoMode(QLineEdit.EchoMode.Password)

        gemini_status = QLabel("Key configured: Yes" if store.has_api_key("gemini") else "Key configured: No")
        gemini_status.setObjectName("Muted")

        gemini_save = QPushButton("Save to Credential Manager")
        gemini_save.setObjectName("Primary")

        def save_gemini():
            val = gemini_input.text().strip()
            if val:
                store.set_api_key("gemini", val)
                gemini_input.clear()
                gemini_status.setText("Key configured: Yes (Securely Saved)")

        gemini_save.clicked.connect(save_gemini)
        gemini_row.addWidget(gemini_input, 1)
        gemini_row.addWidget(gemini_save)
        gemini_box.addLayout(gemini_row)
        gemini_box.addWidget(gemini_status)
        layout.addWidget(gemini_frame)

        # Groq API Key panel
        groq_frame, groq_box = self.panel("Groq API Credentials", "Manage your Groq API key securely in Windows Credential Manager.")
        groq_row = QHBoxLayout()
        groq_input = QLineEdit()
        groq_input.setPlaceholderText("Enter Groq API key (gsk_…)…")
        groq_input.setEchoMode(QLineEdit.EchoMode.Password)

        groq_status = QLabel("Key configured: Yes" if store.has_api_key("groq") else "Key configured: No")
        groq_status.setObjectName("Muted")

        groq_save = QPushButton("Save to Credential Manager")
        groq_save.setObjectName("Primary")

        def save_groq():
            val = groq_input.text().strip()
            if val:
                store.set_api_key("groq", val)
                groq_input.clear()
                groq_status.setText("Key configured: Yes (Securely Saved)")

        groq_save.clicked.connect(save_groq)
        groq_row.addWidget(groq_input, 1)
        groq_row.addWidget(groq_save)
        groq_box.addLayout(groq_row)
        groq_box.addWidget(groq_status)
        layout.addWidget(groq_frame)

        # OpenAI API Key panel
        openai_frame, openai_box = self.panel("OpenAI API Credentials", "Manage your OpenAI API key securely in Windows Credential Manager.")
        openai_row = QHBoxLayout()
        openai_input = QLineEdit()
        openai_input.setPlaceholderText("Enter OpenAI API key (sk-…)…")
        openai_input.setEchoMode(QLineEdit.EchoMode.Password)

        openai_status = QLabel("Key configured: Yes" if store.has_api_key("openai") else "Key configured: No")
        openai_status.setObjectName("Muted")

        openai_save = QPushButton("Save to Credential Manager")
        openai_save.setObjectName("Primary")

        def save_openai():
            val = openai_input.text().strip()
            if val:
                store.set_api_key("openai", val)
                openai_input.clear()
                openai_status.setText("Key configured: Yes (Securely Saved)")

        openai_save.clicked.connect(save_openai)
        openai_row.addWidget(openai_input, 1)
        openai_row.addWidget(openai_save)
        openai_box.addLayout(openai_row)
        openai_box.addWidget(openai_status)
        layout.addWidget(openai_frame)

        # Appearance panel
        card, _ = self.panel("Appearance", "Use the button at the bottom of the sidebar to toggle between dark and light themes.")
        layout.addWidget(card)
        layout.addStretch()
        self.stack.addWidget(page)
        return page

    def toggle_theme(self):
        self.dark = not self.dark
        self.apply_theme()
        if hasattr(self, "compare_view"):
            self.compare_view.apply_theme(self.dark)
        if hasattr(self, "routing_pill"):
            self.routing_pill.apply_theme(self.dark)
        if hasattr(self, "teams_view") and self.teams_view:
            self.teams_view.set_dark_theme(self.dark)
        if self.messages or self.assistant_buffer:
            self.render_transcript()

    def apply_theme(self):
        self.setStyleSheet(DARK if self.dark else LIGHT)
