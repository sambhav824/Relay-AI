from __future__ import annotations

import html
import json
import re
from typing import Any, List, Optional, Set, Union
from unittest.mock import Mock, MagicMock
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSettings, QSize
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton,
    QLabel, QFrame, QStackedWidget, QLineEdit, QTextBrowser, QComboBox,
    QGridLayout, QSizePolicy, QCheckBox
)
from PySide6.QtWidgets import QDoubleSpinBox, QFormLayout, QSpinBox, QScrollArea
from src.core.comparison import ComparisonCoordinator
from src.core.message import Message, MessageRole
from src.core.security import CredentialStore
from src.providers.base import AIProvider
from src.providers.manager import ProviderManager, get_default_provider_manager
from src.providers.ollama import stream_chat, generate
from src.providers.cost_protection import UsageLimits, UsageTracker, estimate_input_tokens
from src.routing.classifier import TaskClassifier
from src.routing.registry import DiscoveryWorker, ModelCapabilityRegistry
from src.routing.router import ModelRouter
from src.routing.types import NoEligibleModelsError, RoutingStrategy, TaskType
from src.routing.worker import RoutedChatWorker
from src.ui.compare_view import ComparisonView
from src.ui.routing_pill import RoutingStatusPill
from src.ui.teams import AITeamsView, safely_anchor_worker
from src.voice import (
    VoiceAssistantController,
    VoiceOverlayPanel,
    SpeechSynthesizer,
)
from src.voice.assistant import VoiceState
from src.ui.theme import (
    get_stylesheet,
    get_theme,
    apply_theme,
    badge_style,
    RAVEN,
    CHARCOAL,
    WHITE,
    GRAY,
)

DARK = get_stylesheet(dark=True)
LIGHT = get_stylesheet(dark=False)


# Retains strong Python references to any workers that remain running during application
# shutdown until their background QThread terminates, preventing native Qt destruction crashes.
_ORPHANED_WORKERS: Set[QThread] = set()


class ChatWorker(QThread):
    """Background worker thread keeping network operations outside the Qt GUI thread."""
    chunk_received = Signal(str)
    completed = Signal(str)
    failed = Signal(str)
    cancelled = Signal()
    fallback_started = Signal(str, str, str)
    usage_received = Signal(object)
    context_retrieved = Signal(list)

    def __init__(
        self,
        provider: AIProvider,
        messages: List[Union[Message, dict]],
        model: str,
        retriever: Optional[Any] = None,
        prefer_offline: bool = False,
    ):
        super().__init__()
        self.provider = provider
        self.messages = list(messages)
        self.model = model
        self.retriever = retriever
        self.prefer_offline = prefer_offline
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

            # Perform RAG document retrieval if hybrid retriever is configured
            if self.retriever is not None and self.messages:
                user_prompt = ""
                for msg in reversed(self.messages):
                    role = msg.role.value if isinstance(msg, Message) else msg.get("role")
                    if role == "user":
                        user_prompt = msg.content if isinstance(msg, Message) else str(msg.get("content", ""))
                        break
                if user_prompt:
                    try:
                        chunks = self.retriever.search(
                            user_prompt,
                            match_count=5,
                            prefer_offline=self.prefer_offline,
                        )
                        if chunks:
                            self.context_retrieved.emit(chunks)
                            from src.context.retrieval import format_rag_context
                            rag_text = format_rag_context(chunks)
                            if rag_text:
                                has_system = False
                                for i, msg in enumerate(self.messages):
                                    role = msg.role.value if isinstance(msg, Message) else msg.get("role")
                                    if role == "system":
                                        content = msg.content if isinstance(msg, Message) else str(msg.get("content", ""))
                                        new_content = f"{content}\n\n{rag_text}"
                                        self.messages[i] = (
                                            Message(MessageRole.SYSTEM, new_content)
                                            if isinstance(msg, Message)
                                            else {"role": "system", "content": new_content}
                                        )
                                        has_system = True
                                        break
                                if not has_system:
                                    self.messages.insert(0, Message(MessageRole.SYSTEM, rag_text))
                    except Exception as ret_err:
                        import logging
                        logging.getLogger("RelayAI.ChatWorker").warning("ChatWorker RAG retrieval failed: %s", ret_err)

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
                    metadata = getattr(chunk, "metadata", None)
                    extra = getattr(metadata, "extra", {}) if metadata else {}
                    if extra and extra.get("automatic_fallback_from"):
                        parts.clear()
                        self.fallback_started.emit(
                            extra["automatic_fallback_from"],
                            metadata.provider_id,
                            metadata.model,
                        )
                    if metadata and (
                        metadata.prompt_tokens is not None
                        or metadata.completion_tokens is not None
                    ):
                        self.usage_received.emit(metadata)
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
    def __init__(
        self,
        provider_manager: Optional[ProviderManager] = None,
        auto_discover: bool = False,
        sync_worker: Optional[Any] = None,
        auth_manager: Optional[Any] = None,
        sync_queue: Optional[Any] = None,
        context_repo: Optional[Any] = None,
        document_manager: Optional[Any] = None,
        hybrid_retriever: Optional[Any] = None,
    ):
        super().__init__()
        self.dark = True
        self.worker: Optional[ChatWorker] = None
        self.routed_worker: Optional[RoutedChatWorker] = None
        self.discovery_worker: Optional[DiscoveryWorker] = None
        self.sync_worker = sync_worker
        self.auth_manager = auth_manager
        self.sync_queue = sync_queue
        self.context_repo = context_repo
        self.document_manager = document_manager
        self.hybrid_retriever = hybrid_retriever

        # Auto-initialize DocumentManager and HybridRetriever if context_repo is provided
        if self.document_manager is None and self.context_repo is not None:
            try:
                from src.context.documents.manager import DocumentManager
                user_id = self.auth_manager.user.id if (self.auth_manager and self.auth_manager.user) else "local"
                self.document_manager = DocumentManager(
                    connection=self.context_repo.conn,
                    user_id=user_id,
                    sync_queue=self.sync_queue,
                )
            except Exception as doc_init_err:
                import logging
                logging.getLogger("RelayAI.MainWindow").warning("Could not auto-initialize DocumentManager: %s", doc_init_err)

        if self.hybrid_retriever is None and self.context_repo is not None:
            try:
                from src.context.retrieval.hybrid import HybridRetriever
                from src.context.retrieval.sparse_client import SparseRetrievalClient
                sparse_client = SparseRetrievalClient(
                    sqlite_connection=self.context_repo.conn,
                    auth_manager=self.auth_manager,
                )
                self.hybrid_retriever = HybridRetriever(sparse_client=sparse_client)
            except Exception as ret_init_err:
                import logging
                logging.getLogger("RelayAI.MainWindow").warning("Could not auto-initialize HybridRetriever: %s", ret_init_err)

        from src.ui.documents_view import DocumentsView
        self.documents_view = DocumentsView(document_manager=self.document_manager)
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

        # Voice Assistant & File Attachments
        self.attachments: List[dict] = []
        self.voice_controller = VoiceAssistantController(self)
        self.voice_controller.transcription_ready.connect(self._on_voice_transcription)
        self.voice_controller.state_changed.connect(self._on_voice_state_changed)

        # Provider architecture and routing engine integration
        self.provider_manager = provider_manager or get_default_provider_manager()
        self.app_settings = QSettings("RelayAI", "RelayAI")
        self.usage_limits = UsageLimits.from_mapping({
            name: self.app_settings.value(f"cost/{name}", getattr(UsageLimits(), name))
            for name in vars(UsageLimits())
        })
        self.automatic_fallback_enabled = self.app_settings.value("provider/automatic_fallback", True, type=bool)
        preferred_provider = self.app_settings.value("provider/manual_selection", "", type=str)
        if preferred_provider and self.provider_manager.get_provider(preferred_provider):
            self.provider_manager.set_active_provider(preferred_provider)
        self.provider_manager.set_automatic_fallback(self.automatic_fallback_enabled)
        self.provider_manager.configure_usage_protection(self.usage_limits, self._usage_user_id)
        active_openai = self.provider_manager.get_provider("openai")
        self.usage_tracker = getattr(active_openai, "usage_tracker", None) or UsageTracker()
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
        shell.setObjectName("CentralWidget")
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
            ("⚙   Settings", self.settings_page()),
        ]
        self.nav_buttons = []
        for label, page in pages:
            button = QPushButton(label)
            button.clicked.connect(lambda checked=False, p=page: self.stack.setCurrentWidget(p))
            side.addWidget(button)
            self.nav_buttons.append(button)
        side.addStretch()

        self.sync_status_btn = QPushButton("☁️ Synced")
        self.sync_status_btn.setObjectName("SyncStatusButton")
        self.sync_status_btn.setToolTip("Cloud Sync: Up to date. Click to sync now.")
        self.sync_status_btn.clicked.connect(self.trigger_sync_now)
        side.addWidget(self.sync_status_btn)

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

        if self.sync_worker:
            self._connect_sync_worker(self.sync_worker)

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

    def _connect_sync_worker(self, worker: Any) -> None:
        """Attach signal handlers to the active sync worker."""
        self.sync_worker = worker
        self._active_workers.add(worker)
        if hasattr(worker, "queue_changed"):
            worker.queue_changed.connect(self._on_sync_queue_changed)
        if hasattr(worker, "item_synced"):
            worker.item_synced.connect(self._on_sync_item_synced)
        if hasattr(worker, "item_failed"):
            worker.item_failed.connect(self._on_sync_item_failed)
        if hasattr(worker, "sync_error"):
            worker.sync_error.connect(self._on_sync_error)
        if hasattr(worker, "finished"):
            worker.finished.connect(lambda w=worker: self._on_worker_finished(w))
        if hasattr(self.sync_queue, "pending_count"):
            try:
                self._update_sync_ui(status="synced", pending=self.sync_queue.pending_count())
            except Exception:
                pass

    def trigger_sync_now(self) -> None:
        """Manually trigger immediate sync and retry failed queue items."""
        if self.sync_queue and hasattr(self.sync_queue, "retry_failed"):
            try:
                self.sync_queue.retry_failed()
            except Exception:
                pass
        if self.sync_worker:
            if hasattr(self.sync_worker, "trigger_sync"):
                self.sync_worker.trigger_sync()
            elif hasattr(self.sync_worker, "run_once"):
                try:
                    self.sync_worker.run_once()
                except Exception:
                    pass
            self._update_sync_ui(status="syncing")

    def _on_sync_queue_changed(self, pending_count: int) -> None:
        if pending_count > 0:
            self._update_sync_ui(status="syncing", pending=pending_count)
        else:
            self._update_sync_ui(status="synced", pending=0)

    def _on_sync_item_synced(self, queue_id: str) -> None:
        if self.sync_queue and hasattr(self.sync_queue, "pending_count"):
            try:
                count = self.sync_queue.pending_count()
                if count == 0:
                    self._update_sync_ui(status="synced", pending=0)
            except Exception:
                pass

    def _on_sync_item_failed(self, queue_id: str, error: str) -> None:
        self._update_sync_ui(status="error", error_msg=error)

    def _on_sync_error(self, error: str) -> None:
        self._update_sync_ui(status="error", error_msg=error)

    def _update_sync_ui(self, status: str, pending: int = 0, error_msg: Optional[str] = None) -> None:
        if not hasattr(self, "sync_status_btn") or not self.sync_status_btn:
            return
        if status == "syncing":
            count_str = f" ({pending})" if pending > 0 else ""
            self.sync_status_btn.setText(f"🔄 Syncing{count_str}")
            self.sync_status_btn.setToolTip(f"Syncing with cloud... {pending} pending changes.")
        elif status == "error":
            self.sync_status_btn.setText("⚠️ Sync Error")
            self.sync_status_btn.setToolTip(f"Sync error: {error_msg or 'Connection issue'}. Click to retry.")
        else:
            count_str = f" ({pending} pending)" if pending > 0 else ""
            self.sync_status_btn.setText(f"☁️ Synced{count_str}")
            self.sync_status_btn.setToolTip("Cloud Sync: All changes up to date. Click to sync now.")

    def page_base(self, title, subtitle):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(10, 6, 10, 8)
        layout.setSpacing(12)
        title_label = QLabel(title)
        title_label.setObjectName("PageTitle")
        title_label.setStyleSheet("background-color: transparent;")
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("Muted")
        subtitle_label.setStyleSheet("background-color: transparent;")
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        return page, layout

    def panel(self, title, description):
        frame = QFrame()
        frame.setObjectName("Panel")
        box = QVBoxLayout(frame)
        box.setContentsMargins(16, 14, 16, 14)
        heading = QLabel(title)
        heading.setObjectName("PanelTitle")
        heading.setStyleSheet("font-size: 15px; font-weight: 650; background-color: transparent;")
        desc = QLabel(description)
        desc.setObjectName("Muted")
        desc.setStyleSheet("background-color: transparent;")
        desc.setWordWrap(True)
        box.addWidget(heading)
        box.addWidget(desc)
        return frame, box

    def dashboard_page(self):
        page, layout = self.page_base("Good to see you.", "One workspace for all your AI models.")
        grid = QGridLayout()
        cards = [
            ("Connected AI Providers", "Local Ollama, Google Gemini, Groq & OpenAI ready to use."),
            ("AI Models Ready", "Smart models for general chat, fast coding, vision, and deep reasoning."),
            ("Privacy & Security", "Encrypted local key storage with zero telemetry or data sharing."),
        ]
        for i, (title, desc) in enumerate(cards):
            card, _ = self.panel(title, desc)
            grid.addWidget(card, 0, i)
        layout.addLayout(grid)

        welcome, box = self.panel(
            "Welcome to Relay AI",
            "Chat with top AI models, automatically route questions to the smartest engine, compare answers side-by-side, or collaborate with autonomous AI teams."
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

        # Header: Clean two-tier layout (Row 1: Title & Mode Switcher; Row 2: Sub-toolbar)
        header = QWidget()
        header_vbox = QVBoxLayout(header)
        header_vbox.setContentsMargins(4, 2, 4, 4)
        header_vbox.setSpacing(8)

        # Row 1: Title and Segmented Mode Switcher
        top_row = QWidget()
        top_row_layout = QHBoxLayout(top_row)
        top_row_layout.setContentsMargins(0, 0, 0, 0)
        top_row_layout.setSpacing(10)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title_label = QLabel("AI Chat")
        title_label.setObjectName("PageTitle")
        subtitle_label = QLabel("Ollama · Gemini · Groq · OpenAI · Multi-turn context · Markdown streaming")
        subtitle_label.setObjectName("Muted")
        title_box.addWidget(title_label)
        title_box.addWidget(subtitle_label)
        top_row_layout.addLayout(title_box)

        top_row_layout.addStretch()

        # Mode Selector: Direct Chat, Smart Route, Compare Models
        self.mode_frame = QFrame()
        self.mode_frame.setObjectName("Panel")
        mode_layout = QHBoxLayout(self.mode_frame)
        mode_layout.setContentsMargins(4, 3, 4, 3)
        mode_layout.setSpacing(4)

        self.btn_mode_single = QPushButton("Direct Chat")
        self.btn_mode_single.setObjectName("Primary")
        self.btn_mode_single.setMinimumWidth(100)
        self.btn_mode_single.setToolTip("Chat directly with a selected model")
        self.btn_mode_single.clicked.connect(lambda: self.set_chat_mode("single"))

        self.btn_mode_routed = QPushButton("✦ Smart Route")
        self.btn_mode_routed.setObjectName("Secondary")
        self.btn_mode_routed.setMinimumWidth(110)
        self.btn_mode_routed.setToolTip("Automatically route prompts to the best model")
        self.btn_mode_routed.clicked.connect(lambda: self.set_chat_mode("routed"))

        self.btn_mode_compare = QPushButton("Compare Models")
        self.btn_mode_compare.setObjectName("Secondary")
        self.btn_mode_compare.setMinimumWidth(130)
        self.btn_mode_compare.setToolTip("Compare model outputs side-by-side")
        self.btn_mode_compare.clicked.connect(lambda: self.set_chat_mode("compare"))

        mode_layout.addWidget(self.btn_mode_single)
        mode_layout.addWidget(self.btn_mode_routed)
        mode_layout.addWidget(self.btn_mode_compare)
        top_row_layout.addWidget(self.mode_frame)
        header_vbox.addWidget(top_row)

        # Row 2: Sub-toolbar controls
        # Single / Direct Model Controls Widget
        self.single_controls_widget = QWidget()
        self.single_controls_widget.setVisible(True)
        controls = QHBoxLayout(self.single_controls_widget)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(10)

        # Provider Selector
        prov_lbl = QLabel("Provider:")
        prov_lbl.setObjectName("Muted")
        self.provider_combo = QComboBox()
        self.provider_combo.setMinimumWidth(140)
        self._populate_providers()
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)

        # Model Selector
        lbl = QLabel("Model:")
        lbl.setObjectName("Muted")
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(190)
        self._populate_models()

        self.status_badge = QLabel("")
        self.status_badge.setStyleSheet("font-size: 12px; font-weight: 600; padding: 2px 6px;")

        self.clear_button = QPushButton("Clear chat")
        self.clear_button.setObjectName("Secondary")
        self.clear_button.setToolTip("Clear conversation history")
        self.clear_button.clicked.connect(self.clear_chat)

        controls.addWidget(prov_lbl)
        controls.addWidget(self.provider_combo)
        controls.addWidget(lbl)
        controls.addWidget(self.model_combo)
        controls.addStretch()
        controls.addWidget(self.status_badge)
        controls.addWidget(self.clear_button)
        header_vbox.addWidget(self.single_controls_widget)

        # Smart Routing Controls Widget
        self.routed_controls_widget = QWidget()
        self.routed_controls_widget.setVisible(False)
        routed_controls = QHBoxLayout(self.routed_controls_widget)
        routed_controls.setContentsMargins(0, 0, 0, 0)
        routed_controls.setSpacing(10)

        strat_lbl = QLabel("Routing Strategy:")
        strat_lbl.setObjectName("Muted")
        self.strategy_combo = QComboBox()
        self.strategy_combo.setMinimumWidth(140)
        self.strategy_combo.addItem("Balanced", RoutingStrategy.BALANCED)
        self.strategy_combo.addItem("Quality First", RoutingStrategy.QUALITY_FIRST)
        self.strategy_combo.addItem("Cost Efficient", RoutingStrategy.COST_EFFICIENT)
        self.strategy_combo.addItem("Local First", RoutingStrategy.LOCAL_FIRST)

        self.local_only_checkbox = QCheckBox("Local-Only")
        self.local_only_checkbox.setToolTip("Strictly exclude all cloud providers; only local on-device models are evaluated.")

        self.routing_pill = RoutingStatusPill(dark=self.dark, parent=page)

        self.clear_button_routed = QPushButton("Clear chat")
        self.clear_button_routed.setObjectName("Secondary")
        self.clear_button_routed.setToolTip("Clear conversation history")
        self.clear_button_routed.clicked.connect(self.clear_chat)

        routed_controls.addWidget(strat_lbl)
        routed_controls.addWidget(self.strategy_combo)
        routed_controls.addWidget(self.local_only_checkbox)
        routed_controls.addStretch()
        routed_controls.addWidget(self.routing_pill)
        routed_controls.addWidget(self.clear_button_routed)
        header_vbox.addWidget(self.routed_controls_widget)

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

        # Speech & Dictation Overlay (preserved as None for STT-only mode)
        self.voice_overlay = None

        # Attachment Preview Tray (Initially hidden; appears when user uploads documents/images)
        self.attachment_tray = QFrame()
        self.attachment_tray.setObjectName("AttachmentTray")
        self.attachment_tray_layout = QHBoxLayout(self.attachment_tray)
        self.attachment_tray_layout.setContentsMargins(4, 2, 4, 2)
        self.attachment_tray_layout.setSpacing(6)
        self.attachment_tray.setVisible(False)
        page_layout.addWidget(self.attachment_tray, 0)

        # Fixed bottom input bar: Modern capsule matching reference design
        input_container = QFrame()
        input_container.setObjectName("InputContainer")
        input_layout = QHBoxLayout(input_container)
        input_layout.setContentsMargins(6, 4, 6, 4)
        input_layout.setSpacing(6)

        # 1. Attachment Button '+' (Opens Camera / Photos / Files popup menu)
        self.btn_attach = QPushButton("+")
        self.btn_attach.setObjectName("InputAddBtn")
        self.btn_attach.setFixedSize(32, 32)
        self.btn_attach.setToolTip("Attach photos, camera capture, or documents")
        self.btn_attach.clicked.connect(self._on_attach_clicked)
        input_layout.addWidget(self.btn_attach, 0)

        # 2. Text Input LineEdit
        self.prompt_input = QLineEdit()
        self.prompt_input.setObjectName("PromptInput")
        self.prompt_input.setPlaceholderText("Ask Relay AI… (Press Enter to send)")
        self.prompt_input.setFixedHeight(38)
        self.prompt_input.returnPressed.connect(self.send_prompt)
        input_layout.addWidget(self.prompt_input, 1)

        # 3. Dedicated Speech-to-Text Microphone Button
        from src.ui.attachment_menu import render_svg_icon, MIC_SVG
        self.btn_mic = QPushButton()
        self.btn_mic.setObjectName("MicBtn")
        self.btn_mic.setFixedSize(34, 34)
        mic_col = "#FFFFFF" if self.dark else "#202124"
        self.btn_mic.setIcon(render_svg_icon(MIC_SVG, color=mic_col, size=18))
        self.btn_mic.setIconSize(QSize(18, 18))
        self.btn_mic.setToolTip("Voice Dictation (Speech-to-Text)")
        self.btn_mic.clicked.connect(self._on_mic_clicked)
        input_layout.addWidget(self.btn_mic, 0)

        # 4. Send Button
        self.send_button = QPushButton("Send")
        self.send_button.setObjectName("Primary")
        self.send_button.setFixedHeight(34)
        self.send_button.setFixedWidth(75)
        self.send_button.clicked.connect(self.send_prompt)
        input_layout.addWidget(self.send_button, 0)

        # Preserved references for backwards compatibility
        self.btn_think = None
        self.btn_voice = None

        page_layout.addWidget(input_container, 0)

        # 7. Disclaimer Label with rotatable wording
        self.disclaimer_texts = [
            "Relay AI can make mistakes. Verify important info.",
            "Relay AI can make mistakes. Check important facts.",
            "AI models may produce inaccurate information. Please verify critical details.",
            "Review important information — Relay AI models can occasionally produce errors."
        ]
        self._disclaimer_idx = 0
        self.disclaimer_label = QLabel(self.disclaimer_texts[0])
        self.disclaimer_label.setObjectName("DisclaimerText")
        self.disclaimer_label.setAlignment(Qt.AlignCenter)
        self.disclaimer_label.setToolTip("Click to cycle disclaimer wordings")
        self.disclaimer_label.mousePressEvent = self._cycle_disclaimer
        page_layout.addWidget(self.disclaimer_label, 0)

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
            self.app_settings.setValue("provider/manual_selection", provider_id)
            if hasattr(self, "settings_provider_combo"):
                self.settings_provider_combo.blockSignals(True)
                self.settings_provider_combo.setCurrentIndex(
                    self.settings_provider_combo.findData(provider_id)
                )
                self.settings_provider_combo.blockSignals(False)
            self._populate_models()

    def _usage_user_id(self) -> str:
        user = getattr(self.auth_manager, "user", None) if self.auth_manager else None
        user_id = getattr(user, "id", None) if user else None
        return str(user_id) if user_id else "local"

    def _on_settings_provider_changed(self, index: int) -> None:
        if index < 0:
            return
        provider_id = self.settings_provider_combo.currentData()
        if not provider_id or not self.provider_manager.set_active_provider(provider_id):
            return
        self.app_settings.setValue("provider/manual_selection", provider_id)
        if hasattr(self, "provider_combo"):
            self.provider_combo.blockSignals(True)
            self.provider_combo.setCurrentIndex(self.provider_combo.findData(provider_id))
            self.provider_combo.blockSignals(False)
            self._populate_models()

    def _on_automatic_fallback_changed(self, enabled: bool) -> None:
        self.automatic_fallback_enabled = bool(enabled)
        self.app_settings.setValue("provider/automatic_fallback", self.automatic_fallback_enabled)
        self.provider_manager.set_automatic_fallback(self.automatic_fallback_enabled)

    def _save_usage_limits(self) -> None:
        self.usage_limits = UsageLimits.from_mapping({
            "max_input_tokens": self.max_input_tokens_spin.value(),
            "max_output_tokens": self.max_output_tokens_spin.value(),
            "max_request_cost_usd": self.max_request_budget_spin.value(),
            "daily_budget_usd": self.daily_budget_spin.value(),
            "monthly_budget_usd": self.monthly_budget_spin.value(),
            "max_requests_per_day": self.requests_per_day_spin.value(),
            "input_cost_per_million": self.input_rate_spin.value(),
            "output_cost_per_million": self.output_rate_spin.value(),
            "warning_fraction": 0.8,
        })
        for name, value in vars(self.usage_limits).items():
            self.app_settings.setValue(f"cost/{name}", value)
        self.provider_manager.configure_usage_protection(self.usage_limits, self._usage_user_id)
        self._refresh_usage_summary()
        self.usage_settings_status.setText("Usage limits saved.")

    def _refresh_usage_summary(self) -> None:
        summary = self.usage_tracker.summary(self._usage_user_id(), self.usage_limits)
        if not isinstance(summary, dict):
            return
        try:
            daily_cost = float(summary.get("daily_cost", 0.0) or 0.0)
            daily_cost_str = f"${daily_cost:.4f}"
        except (ValueError, TypeError):
            daily_cost_str = "$0.0000"

        try:
            monthly_cost = float(summary.get("monthly_cost", 0.0) or 0.0)
            monthly_cost_str = f"${monthly_cost:.4f}"
        except (ValueError, TypeError):
            monthly_cost_str = "$0.0000"

        daily_reqs = summary.get("daily_requests", 0)
        estimate_text = (
            f"Daily requests: {daily_reqs} · "
            f"Estimated spend: {daily_cost_str} today / {monthly_cost_str} this month"
        )
        if summary.get("daily_actual_input_tokens") or summary.get("daily_actual_output_tokens"):
            estimate_text += (
                f" · Actual tokens today: {summary.get('daily_actual_input_tokens', 0)} in / "
                f"{summary.get('daily_actual_output_tokens', 0)} out"
            )
        if summary.get("warning"):
            estimate_text = "Warning: configured budget/request threshold is near. " + estimate_text
        if hasattr(self, "usage_summary_label"):
            self.usage_summary_label.setText(estimate_text)

    def _on_provider_fallback(self, from_provider: str, to_provider: str, model: str, sid: int) -> None:
        if self._is_shutting_down or sid != self.current_session_id:
            return
        self.assistant_buffer = ""
        self.has_new_tokens = True
        self._current_model_info = (to_provider, model)
        self.fallback_info = (
            f"Automatic fallback: {from_provider.title()} quota/billing limit → "
            f"{to_provider.title()} ({model})"
        )
        self.status_badge.setText(f"Using {to_provider.title()} fallback")
        self.status_badge.setStyleSheet(badge_style("warning", self.dark))
        self._update_current_assistant_block()

    def _on_chat_usage(self, metadata: Any) -> None:
        prompt_tokens = getattr(metadata, "prompt_tokens", None)
        completion_tokens = getattr(metadata, "completion_tokens", None)
        if prompt_tokens is None and completion_tokens is None:
            return
        prompt_text = str(prompt_tokens) if prompt_tokens is not None else "estimated"
        completion_text = str(completion_tokens) if completion_tokens is not None else "estimated"
        self.status_badge.setText(f"Usage: {prompt_text} in / {completion_text} out")
        self.status_badge.setToolTip("Provider-reported token counts where available; otherwise counts are estimates.")
        self._latest_usage_text = self.status_badge.text()
        self._refresh_usage_summary()

    def _populate_models(self):
        """Populate the model combo box from the active provider."""
        self.model_combo.clear()
        active = self.provider_manager.get_active_provider()
        try:
            available = active.get_available_models()
            if available:
                chat_models = [
                    m for m in available
                    if not any(emb in m.id.lower() for emb in ("embed", "embedding", "bge-", "all-minilm", "arctic-embed"))
                ]
                for m in (chat_models if chat_models else available):
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
        self.status_badge.setStyleSheet(badge_style("completed", self.dark))
        self.set_chat_mode("single")

    # -----------------------------------------------------------------
    # Attachment Management (Camera, Photos, Files matching reference design)
    # -----------------------------------------------------------------
    def _on_attach_clicked(self) -> None:
        """Show attachment popup menu (Camera, Photos, Files) matching reference design."""
        if not hasattr(self, "_attachment_menu") or self._attachment_menu is None:
            from src.ui.attachment_menu import AttachmentMenu
            self._attachment_menu = AttachmentMenu(dark=self.dark, parent=self)
            self._attachment_menu.camera_selected.connect(self._on_attach_camera)
            self._attachment_menu.photos_selected.connect(self._on_attach_photos)
            self._attachment_menu.files_selected.connect(self._on_attach_files)
        self._attachment_menu.show_above(self.btn_attach)

    def _on_attach_camera(self) -> None:
        """Open camera capture dialog to take a snapshot photo."""
        from src.ui.attachment_menu import CameraCaptureDialog
        dialog = CameraCaptureDialog(dark=self.dark, parent=self)
        dialog.photo_captured.connect(self.add_attachment)
        dialog.exec()

    def _on_attach_photos(self) -> None:
        """Open file dialog specifically filtered for photos and images."""
        from PySide6.QtWidgets import QFileDialog
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Photos to Attach",
            "",
            "Photos / Images (*.png *.jpg *.jpeg *.webp *.bmp *.gif);;All Files (*.*)"
        )
        if files:
            for f in files:
                self.add_attachment(f)

    def _on_attach_files(self) -> None:
        """Open file dialog filtered for documents, PDFs, DOCX, and other files."""
        from PySide6.QtWidgets import QFileDialog
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Attach Files to Relay AI",
            "",
            "Documents (*.pdf *.docx *.txt *.md *.csv *.json);;All Files (*.*)"
        )
        if files:
            for f in files:
                self.add_attachment(f)

    def add_attachment(self, file_path: str) -> None:
        from pathlib import Path
        import mimetypes
        p = Path(file_path)
        if not p.is_file():
            return
        if any(att["path"] == str(p.resolve()) for att in self.attachments):
            return

        size_kb = max(1, p.stat().st_size // 1024)
        suffix = p.suffix.lower()
        is_image = suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")
        mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"

        extracted_text = ""
        if not is_image:
            try:
                from src.context.documents.extractors import DocumentExtractor
                ext = DocumentExtractor()
                res = ext.extract(p)
                extracted_text = res.text
            except Exception as e:
                extracted_text = f"[Text extraction note: {e}]"

        att_record = {
            "path": str(p.resolve()),
            "name": p.name,
            "size_kb": size_kb,
            "is_image": is_image,
            "mime": mime,
            "extracted_text": extracted_text,
        }
        self.attachments.append(att_record)
        self._refresh_attachment_tray()

    def remove_attachment(self, index: int) -> None:
        if 0 <= index < len(self.attachments):
            self.attachments.pop(index)
            self._refresh_attachment_tray()

    def _refresh_attachment_tray(self) -> None:
        if not hasattr(self, "attachment_tray") or not hasattr(self, "attachment_tray_layout"):
            return
        while self.attachment_tray_layout.count() > 0:
            item = self.attachment_tray_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self.attachments:
            self.attachment_tray.setVisible(False)
            return

        self.attachment_tray.setVisible(True)
        for idx, att in enumerate(self.attachments):
            chip = QFrame()
            chip.setObjectName("AttachmentChip")
            c_layout = QHBoxLayout(chip)
            c_layout.setContentsMargins(6, 2, 6, 2)
            c_layout.setSpacing(6)

            icon = "🖼" if att["is_image"] else ("📄" if att["name"].endswith(".pdf") else "📝")
            lbl = QLabel(f"{icon} {att['name']} ({att['size_kb']} KB)")
            lbl.setStyleSheet("font-size: 11px;")

            del_btn = QPushButton("✕")
            del_btn.setFixedSize(16, 16)
            del_btn.setObjectName("Secondary")
            del_btn.setStyleSheet("font-size: 9px; padding: 0px; border: none; background: transparent;")
            del_btn.clicked.connect(lambda chk=False, i=idx: self.remove_attachment(i))

            c_layout.addWidget(lbl)
            c_layout.addWidget(del_btn)
            self.attachment_tray_layout.addWidget(chip)

        self.attachment_tray_layout.addStretch()

    # -----------------------------------------------------------------
    # Voice Assistant & Speech Handlers
    # -----------------------------------------------------------------
    def _on_mic_clicked(self) -> None:
        """Trigger speech-to-text dictation into the prompt input box."""
        if self.voice_controller.state == VoiceState.LISTENING:
            self.voice_controller.stop_listening()
        else:
            self.voice_controller.start_listening("Speak your prompt…")

    def _on_voice_mode_clicked(self) -> None:
        """Toggle live conversational voice assistant mode."""
        active = self.voice_controller.toggle_voice_mode()
        if hasattr(self, "voice_overlay"):
            self.voice_overlay.setVisible(active)

    def _on_voice_transcription(self, text: str) -> None:
        """Handle recognized speech text from microphone (Speech-to-Text)."""
        if not text:
            return
        current_text = self.prompt_input.text().strip()
        new_text = f"{current_text} {text}".strip() if current_text else text
        self.prompt_input.setText(new_text)
        self.prompt_input.setFocus()

    def _on_voice_state_changed(self, state_name: str, detail: str) -> None:
        """Update voice assistant UI elements according to active state."""
        if hasattr(self, "voice_overlay") and self.voice_overlay:
            try:
                self.voice_overlay.set_state(VoiceState(state_name), detail)
            except Exception:
                pass
        if hasattr(self, "btn_mic") and self.btn_mic:
            from src.ui.attachment_menu import render_svg_icon, MIC_SVG, MIC_RECORDING_SVG
            t = get_theme(self.dark)
            if state_name == "listening":
                self.btn_mic.setIcon(render_svg_icon(MIC_RECORDING_SVG, color=t.text_inverted, size=18))
                self.btn_mic.setStyleSheet(f"background-color: {t.btn_primary_bg}; border-radius: 17px;")
                self.btn_mic.setToolTip("Listening… Click to stop dictation")
            else:
                icon_col = t.text_primary
                self.btn_mic.setIcon(render_svg_icon(MIC_SVG, color=icon_col, size=18))
                self.btn_mic.setStyleSheet("")
                self.btn_mic.setToolTip("Voice Dictation (Speech-to-Text)")

    def _cycle_disclaimer(self, event=None) -> None:
        """Cycle through alternative disclaimer wordings when clicked."""
        if hasattr(self, "disclaimer_texts") and hasattr(self, "disclaimer_label"):
            self._disclaimer_idx = (self._disclaimer_idx + 1) % len(self.disclaimer_texts)
            self.disclaimer_label.setText(self.disclaimer_texts[self._disclaimer_idx])

    def send_prompt(self):
        prompt = self.prompt_input.text().strip()
        t = get_theme(self.dark)

        # Check attachments and thinking mode
        active_attachments = list(self.attachments)
        self.attachments.clear()
        self._refresh_attachment_tray()

        if not prompt and active_attachments:
            prompt = "Please analyze the attached document(s) and information."

        prompt_to_send = prompt
        user_display_content = prompt
        image_paths: List[str] = []

        if active_attachments:
            att_snippets = []
            display_chips = []
            for att in active_attachments:
                icon = "🖼" if att["is_image"] else ("📄" if att["name"].endswith(".pdf") else "📝")
                display_chips.append(f"{icon} {att['name']} ({att['size_kb']} KB)")
                if att.get("extracted_text"):
                    snippet = att["extracted_text"]
                    if len(snippet) > 8000:
                        snippet = snippet[:8000] + "\n...[document text truncated]"
                    att_snippets.append(f"\n--- Attached Document ({att['name']}) ---\n{snippet}\n---")
                elif att["is_image"] and att.get("path"):
                    image_paths.append(att["path"])

            if att_snippets:
                prompt_to_send = (prompt + "\n" + "\n".join(att_snippets)).strip()
            if display_chips:
                user_display_content = f"{prompt}\n\n*📎 Attached: {', '.join(display_chips)}*".strip()

        if hasattr(self, "btn_think") and self.btn_think and self.btn_think.isChecked():
            prompt_to_send = f"[Deep Thinking Mode: Analyze thoroughly and reason step-by-step before answering]\n\n{prompt_to_send}"
            user_display_content = f"🧠 *Thinking Mode Active*\n\n{user_display_content}"

        if hasattr(self, "voice_controller"):
            self.voice_controller.on_prompt_sent()

        # Compare Models mode
        if self.chat_mode == "compare":
            if self.comparison_coordinator.is_running():
                self.cancel_generation()
                return

            if not prompt:
                return

            # Append user message to shared context
            user_message = Message(role=MessageRole.USER, content=user_display_content, images=image_paths)
            self.messages.append(user_message)

            self.prompt_input.clear()
            self.prompt_input.setEnabled(False)
            self.send_button.setText("■ Stop All")
            self.send_button.setStyleSheet(f"background: {t.bg_card}; color: {t.text_primary}; border: 2px solid {t.border_focus}; font-weight: 700;")
            self.send_button.setEnabled(True)
            self.mode_frame.setEnabled(False)

            self.compare_view.summary_banner.setVisible(False)
            for card in self.compare_view.slot_cards.values():
                card.reset_for_stream()
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
        prior_history = [Message(m.role, m.content, images=getattr(m, "images", None)) for m in self.messages]

        # 2. Append User message exactly once to self.messages
        user_message = Message(role=MessageRole.USER, content=user_display_content, images=image_paths)
        self.messages.append(user_message)

        # 3. Update UI states and lock mode switching
        self.prompt_input.clear()
        self.assistant_buffer = ""
        self.fallback_info = None
        self._current_model_info = None
        self._latest_usage_text = ""
        self.is_thinking = True
        self.has_new_tokens = False

        self.mode_frame.setEnabled(False)
        self.send_button.setText("■ Stop")
        self.send_button.setStyleSheet(f"background: {t.bg_card}; color: {t.text_primary}; border: 2px solid {t.border_focus}; font-weight: 700;")
        self.send_button.setEnabled(True)
        self.clear_button.setEnabled(False)
        if hasattr(self, "clear_button_routed"):
            self.clear_button_routed.setEnabled(False)
        self.prompt_input.setEnabled(False)

        # Append User message to transcript HTML
        cursor = self.transcript.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertHtml(self._format_user_card(user_display_content))
        cursor.insertBlock()

        # Append Assistant placeholder and record cursor position
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.current_assistant_start = cursor.position()
        cursor.insertHtml(self._format_assistant_card("", is_thinking=True))
        self.current_assistant_end = cursor.position()

        self._scroll_if_near_bottom(force=True)
        self.stream_timer.start(60)

        # 4. Dispatch appropriate worker
        local_only = self.local_only_checkbox.isChecked() if hasattr(self, "local_only_checkbox") else False
        if self.chat_mode == "routed":
            raw_strategy = self.strategy_combo.currentData() or RoutingStrategy.BALANCED
            if isinstance(raw_strategy, str):
                try:
                    strategy = RoutingStrategy(raw_strategy)
                except ValueError:
                    strategy = RoutingStrategy.BALANCED
            else:
                strategy = raw_strategy

            try:
                decision = self.model_router.route(
                    prompt=prompt_to_send,
                    history=prior_history,
                    strategy=strategy,
                    local_only=local_only,
                    has_images=bool(image_paths),
                )
                self.routing_pill.set_decision(decision)
                task_name = decision.classification.task_type.value.replace("_", " ").title()
                strat_name = strategy.value.replace("_", " ").title()
                self._last_routing_tag = f"Smart Routed: {strat_name} · {task_name}"
                self._current_model_info = (decision.primary.provider_id, decision.primary.model_id)
            except NoEligibleModelsError as exc:
                self._handle_preflight_error(str(exc), sid)
                return

            is_cand_local = False
            if decision.primary:
                if hasattr(decision.primary, "is_local"):
                    is_cand_local = bool(decision.primary.is_local)
                elif getattr(decision.primary, "profile", None) is not None:
                    is_cand_local = bool(getattr(decision.primary.profile, "is_local", False))
                elif getattr(decision.primary, "provider_id", "") == "ollama":
                    is_cand_local = True

            prefer_offline_routed = local_only or is_cand_local
            worker = RoutedChatWorker(
                decision=decision,
                prompt=prompt_to_send,
                history=prior_history,
                provider_manager=self.provider_manager,
                retriever=self.hybrid_retriever,
                prefer_offline=prefer_offline_routed,
                images=image_paths,
            )
            self.routed_worker = worker
            self._active_workers.add(worker)

            worker.attempt_started.connect(lambda p, m, idx, s=sid: self._on_routed_attempt_started(p, m, idx, s))
            worker.chunk_received.connect(lambda d, s=sid: self._on_chunk(d, s))
            worker.context_retrieved.connect(lambda chunks, s=sid: self._on_context_retrieved(chunks, s))
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
            is_local_provider = getattr(active_provider, "is_local", False)

            worker = ChatWorker(
                active_provider,
                prior_history + [Message(MessageRole.USER, prompt_to_send, images=image_paths)],
                model,
                retriever=self.hybrid_retriever,
                prefer_offline=local_only or is_local_provider,
            )
            self.worker = worker
            self._active_workers.add(worker)

            worker.chunk_received.connect(lambda d, s=sid: self._on_chunk(d, s))
            worker.context_retrieved.connect(lambda chunks, s=sid: self._on_context_retrieved(chunks, s))
            worker.fallback_started.connect(
                lambda fp, tp, model, s=sid: self._on_provider_fallback(fp, tp, model, s)
            )
            worker.usage_received.connect(lambda metadata, s=sid: self._on_chat_usage(metadata))
            worker.completed.connect(lambda ans, s=sid: self._on_chat_completed(ans, s))
            worker.failed.connect(lambda err, s=sid: self._on_chat_failed(err, s))
            worker.cancelled.connect(lambda s=sid: self._on_chat_cancelled(s))
            worker.finished.connect(lambda w=worker: self._on_worker_finished(w))
            worker.start()

    def _on_context_retrieved(self, chunks: list, sid: int) -> None:
        """Handle RAG context attribution notification from worker thread."""
        if sid != self.current_session_id or not chunks:
            return
        sources = []
        for c in chunks:
            meta = getattr(c, "metadata", {}) or {}
            src = meta.get("source_file") or meta.get("title") or getattr(c, "document_id", "Document")
            if src not in sources:
                sources.append(src)
        if sources:
            source_summary = ", ".join(sources[:3])
            if len(sources) > 3:
                source_summary += f" +{len(sources) - 3} more"
            if hasattr(self, "routing_pill") and hasattr(self.routing_pill, "setToolTip"):
                self.routing_pill.setToolTip(f"RAG active: {len(chunks)} chunk(s) from {source_summary}")

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
        self.status_badge.setStyleSheet(badge_style("running", self.dark))

    def _on_routed_attempt_failed(self, provider_id: str, model_id: str, reason: str, error_msg: str, sid: int):
        if self._is_shutting_down or sid != self.current_session_id:
            return
        self.status_badge.setText(f"{model_id} failed: {reason}")
        self.status_badge.setStyleSheet(badge_style("error", self.dark))

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
        self.status_badge.setText(self._latest_usage_text)
        self._reset_generation_ui()
        if hasattr(self, "voice_controller"):
            self.voice_controller.on_response_ready(response_text)

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
        self.status_badge.setStyleSheet(badge_style("error", self.dark))
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
        self.status_badge.setStyleSheet(badge_style("warning", self.dark))
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
        provider_id, model = self._current_model_info or (
            active_provider.provider_id,
            self.model_combo.currentText(),
        )
        self.messages.append(Message(
            role=MessageRole.ASSISTANT,
            content=answer,
            metadata={"provider_id": provider_id, "model": model},
        ))

        self.assistant_buffer = ""
        self.current_assistant_start = None
        self.current_assistant_end = None
        self.status_badge.setText("")
        self._reset_generation_ui()
        if hasattr(self, "voice_controller"):
            self.voice_controller.on_response_ready(answer)

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
        self.status_badge.setStyleSheet(badge_style("error", self.dark))
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
        self.status_badge.setStyleSheet(badge_style("warning", self.dark))
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
            self.status_badge.setStyleSheet(badge_style("running", self.dark))
            self.send_button.setText("■ Stop")
            t = get_theme(self.dark)
            self.send_button.setStyleSheet(f"background: {t.bg_card}; color: {t.text_primary}; border: 2px solid {t.border_focus}; font-weight: 700;")
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
        t = get_theme(self.dark)
        user_header_color = t.text_primary
        user_bg = t.bg_card if self.dark else t.bg_sunken
        user_border = t.border_strong
        rendered = self.markdown_to_html(content, self.dark)
        return (
            f'<div style="margin-bottom: 16px;">'
            f'  <div style="font-weight: 700; font-size: 13px; color: {user_header_color}; margin-bottom: 6px; padding-left: 2px;">You</div>'
            f'  <div style="background: {user_bg}; border: 1px solid {user_border}; border-radius: 10px; padding: 12px 16px; font-size: 14px; line-height: 1.5; color: {t.text_primary};">'
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
        t = get_theme(self.dark)
        bot_header_color = t.text_primary
        bot_bg = t.bg_sunken if self.dark else t.bg_card
        bot_border = t.border_subtle

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
            fallback_bg = t.bg_card
            fallback_border = t.border_focus
            fallback_banner = (
                f'<div style="background: {fallback_bg}; border: 1px dashed {fallback_border}; border-radius: 6px; padding: 6px 10px; margin-bottom: 8px; font-size: 12px; color: {t.text_primary}; font-weight: 600;">'
                f'  {html.escape(active_fallback)}'
                f'</div>'
            )

        if is_thinking and not content:
            body = f'<div style="color: {t.text_muted}; font-style: italic;">Thinking…</div>'
        else:
            body = self.markdown_to_html(content, self.dark)

        return (
            f'<div style="margin-bottom: 16px;">'
            f'  <div style="font-weight: 700; font-size: 13px; color: {bot_header_color}; margin-bottom: 6px; padding-left: 2px;">{bot_label}</div>'
            f'  <div style="background: {bot_bg}; border: 1px solid {bot_border}; border-radius: 10px; padding: 12px 16px; font-size: 14px; line-height: 1.5; color: {t.text_primary};">'
            f'    {fallback_banner}'
            f'    {body}'
            f'  </div>'
            f'</div>'
        )

    def _format_cancellation_card(self, note: str = "Generation canceled by user") -> str:
        t = get_theme(self.dark)
        return (
            f'<div style="margin-bottom: 16px;">'
            f'  <div style="font-weight: 700; font-size: 13px; color: {t.text_muted}; margin-bottom: 6px; padding-left: 2px;">✦ Relay AI · Canceled</div>'
            f'  <div style="background: {t.bg_card}; border: 1px dashed {t.border_strong}; border-radius: 10px; padding: 12px 16px; font-size: 14px; color: {t.text_secondary};">'
            f'    [ {html.escape(note)} ]'
            f'  </div>'
            f'</div>'
        )

    def _format_error_card(self, error_message: str) -> str:
        t = get_theme(self.dark)
        escaped_err = html.escape(error_message)
        return (
            f'<div style="margin-bottom: 16px; padding: 12px 16px; background: {t.bg_sunken}; border: 2px solid {t.border_focus}; border-radius: 10px; color: {t.text_primary};">'
            f'  <b>Error:</b> {escaped_err}'
            f'</div>'
        )

    def closeEvent(self, event):
        """Cooperative shutdown: cancel all workers, disconnect signals, and safely anchor running threads."""
        self._is_shutting_down = True
        self.stream_timer.stop()
        if hasattr(self, "voice_controller"):
            self.voice_controller.stop_all()
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
        if hasattr(self, "sync_worker") and self.sync_worker and isinstance(self.sync_worker, QThread):
            workers_to_stop.add(self.sync_worker)

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

        if hasattr(self, "sync_worker") and self.sync_worker:
            try:
                self.sync_worker.close()
            except Exception:
                pass

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

        t = get_theme(dark)
        pre_bg = t.bg_sunken
        pre_color = t.text_primary
        pre_border = t.border_strong
        code_bg = t.bg_card if dark else t.bg_sunken
        code_color = t.text_primary

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

    def documents_page(self):
        self.stack.addWidget(self.documents_view)
        return self.documents_view

    def simple_page(self, title, description):
        page, layout = self.page_base(title, description)
        card, _ = self.panel("Planned milestone", description)
        layout.addWidget(card)
        layout.addStretch()
        self.stack.addWidget(page)
        return page

    def settings_page(self):
        page = QWidget()
        outer_layout = QVBoxLayout(page)
        outer_layout.setContentsMargins(10, 6, 10, 8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        title = QLabel("Settings")
        title.setObjectName("PageTitle")
        subtitle = QLabel("Configure Relay AI credentials, provider selection, and usage guardrails.")
        subtitle.setObjectName("Muted")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        scroll.setWidget(content)
        outer_layout.addWidget(scroll)
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

        protection_frame, protection_box = self.panel(
            "Provider & Cost Protection",
            "OpenAI is preferred when configured. Quota/billing fallback uses only the local Ollama llama3.2:3b model. "
            "USD estimates use the editable rates below; zero budgets mean unlimited.",
        )
        controls_form = QFormLayout()
        self.settings_provider_combo = QComboBox()
        for provider in self.provider_manager.list_providers():
            self.settings_provider_combo.addItem(provider.display_name, provider.provider_id)
        self.settings_provider_combo.setCurrentIndex(
            max(0, self.settings_provider_combo.findData(self.provider_manager.get_active_provider().provider_id))
        )
        self.settings_provider_combo.currentIndexChanged.connect(self._on_settings_provider_changed)
        controls_form.addRow("Manual provider", self.settings_provider_combo)

        self.automatic_fallback_checkbox = QCheckBox("Automatically fall back to local Ollama on OpenAI quota/billing errors")
        self.automatic_fallback_checkbox.setChecked(self.automatic_fallback_enabled)
        self.automatic_fallback_checkbox.toggled.connect(self._on_automatic_fallback_changed)
        controls_form.addRow("Automatic fallback", self.automatic_fallback_checkbox)

        def token_spin(value: int) -> QSpinBox:
            control = QSpinBox()
            control.setRange(1, 1_000_000)
            control.setValue(value)
            return control

        def usd_spin(value: float, decimals: int = 4) -> QDoubleSpinBox:
            control = QDoubleSpinBox()
            control.setRange(0.0, 1_000_000_000.0)
            control.setDecimals(decimals)
            control.setSingleStep(0.01 if decimals <= 4 else 0.1)
            control.setValue(value)
            return control

        self.max_input_tokens_spin = token_spin(self.usage_limits.max_input_tokens)
        self.max_output_tokens_spin = token_spin(self.usage_limits.max_output_tokens)
        self.max_request_budget_spin = usd_spin(self.usage_limits.max_request_cost_usd)
        self.daily_budget_spin = usd_spin(self.usage_limits.daily_budget_usd)
        self.monthly_budget_spin = usd_spin(self.usage_limits.monthly_budget_usd)
        self.requests_per_day_spin = QSpinBox()
        self.requests_per_day_spin.setRange(0, 1_000_000)
        self.requests_per_day_spin.setSpecialValueText("Unlimited")
        self.requests_per_day_spin.setValue(self.usage_limits.max_requests_per_day)
        self.input_rate_spin = usd_spin(self.usage_limits.input_cost_per_million, 6)
        self.output_rate_spin = usd_spin(self.usage_limits.output_cost_per_million, 6)

        controls_form.addRow("Max input tokens / request", self.max_input_tokens_spin)
        controls_form.addRow("Max output tokens / request", self.max_output_tokens_spin)
        controls_form.addRow("Max estimated cost / request (USD; 0 = unlimited)", self.max_request_budget_spin)
        controls_form.addRow("Daily budget (USD; 0 = unlimited)", self.daily_budget_spin)
        controls_form.addRow("Monthly budget (USD; 0 = unlimited)", self.monthly_budget_spin)
        controls_form.addRow("Requests per user per day (0 = unlimited)", self.requests_per_day_spin)
        controls_form.addRow("Input price per 1M tokens (USD)", self.input_rate_spin)
        controls_form.addRow("Output price per 1M tokens (USD)", self.output_rate_spin)
        protection_box.addLayout(controls_form)
        self.usage_summary_label = QLabel()
        self.usage_summary_label.setObjectName("Muted")
        self.usage_summary_label.setWordWrap(True)
        protection_box.addWidget(self.usage_summary_label)
        self.usage_settings_status = QLabel("")
        self.usage_settings_status.setObjectName("Muted")
        protection_box.addWidget(self.usage_settings_status)
        save_limits = QPushButton("Save provider and usage settings")
        save_limits.setObjectName("Primary")
        save_limits.clicked.connect(self._save_usage_limits)
        protection_box.addWidget(save_limits)
        layout.addWidget(protection_frame)
        self._refresh_usage_summary()

        # Appearance panel
        card, _ = self.panel("Appearance", "Use the button at the bottom of the sidebar to toggle between dark and light themes.")
        layout.addWidget(card)
        layout.addStretch()
        self.stack.addWidget(page)
        return page

    def toggle_theme(self):
        self.dark = not self.dark
        self.apply_theme()
        if hasattr(self, "_attachment_menu"):
            self._attachment_menu = None
        if hasattr(self, "btn_mic") and self.btn_mic:
            from src.ui.attachment_menu import render_svg_icon, MIC_SVG
            mic_col = "#FFFFFF" if self.dark else "#202124"
            self.btn_mic.setIcon(render_svg_icon(MIC_SVG, color=mic_col, size=18))
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
