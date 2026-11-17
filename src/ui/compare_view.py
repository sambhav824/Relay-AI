import html
from typing import Dict, List, Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication, QTextCursor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QComboBox, QTextBrowser, QFrame, QSizePolicy
)

from src.core.comparison import (
    ComparisonMetrics,
    ComparisonSlotConfig,
    ComparisonSlotResult,
)
from src.providers.manager import ProviderManager
from src.ui.theme import get_theme, badge_style, RAVEN, CHARCOAL, WHITE, GRAY


class ModelSlotCard(QFrame):
    """An independent visual card representing one model stream in comparison mode."""

    select_response = Signal(int, str)  # (slot_id, content)

    def __init__(self, slot_id: int, dark: bool = True, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.slot_id = slot_id
        self.dark = dark
        self.provider_id = ""
        self.model_name = ""
        self.content_buffer = ""
        self.is_thinking = False
        self.result: Optional[ComparisonSlotResult] = None

        self.setObjectName("Panel")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._build_ui()
        self.apply_theme(self.dark)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        # Card Header: Model title + Status badge
        header_layout = QHBoxLayout()
        header_layout.setSpacing(6)

        self.title_label = QLabel(f"Model #{self.slot_id + 1}")
        self.title_label.setStyleSheet("font-size: 13px; font-weight: 700;")
        header_layout.addWidget(self.title_label, 1)

        self.status_badge = QLabel("Ready")
        self.status_badge.setStyleSheet("font-size: 11px; font-weight: 600; padding: 2px 6px; border-radius: 4px;")
        header_layout.addWidget(self.status_badge)

        layout.addLayout(header_layout)

        # Response transcript area
        self.transcript = QTextBrowser()
        self.transcript.setOpenExternalLinks(True)
        self.transcript.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.transcript, 1)

        # Performance & Metrics bar
        self.metrics_bar = QLabel("Time: --  ·  Tokens: --  ·  Speed: --  ·  Cost: --")
        self.metrics_bar.setObjectName("Muted")
        self.metrics_bar.setStyleSheet("font-size: 11px; padding: 2px 0;")
        layout.addWidget(self.metrics_bar)

        # Action Footer: Copy + Select as Active Response
        footer_layout = QHBoxLayout()
        footer_layout.setSpacing(8)

        self.copy_btn = QPushButton("📋 Copy")
        self.copy_btn.setObjectName("Secondary")
        self.copy_btn.setToolTip("Copy response text to clipboard")
        self.copy_btn.clicked.connect(self._copy_to_clipboard)
        self.copy_btn.setEnabled(False)
        footer_layout.addWidget(self.copy_btn)

        self.select_btn = QPushButton("✓ Use this response")
        self.select_btn.setObjectName("Primary")
        self.select_btn.setToolTip("Use this model's response in the conversation")
        self.select_btn.clicked.connect(self._on_select_clicked)
        self.select_btn.setEnabled(False)
        footer_layout.addWidget(self.select_btn, 1)

        layout.addLayout(footer_layout)

    def apply_theme(self, dark: bool):
        self.dark = dark
        t = get_theme(dark)
        bg = t.bg_card
        border = t.border_strong
        tb_bg = t.bg_sunken
        tb_color = t.text_primary
        tb_border = t.border_subtle

        self.setStyleSheet(f"QFrame#Panel {{ background: {bg}; border: 1px solid {border}; border-radius: 8px; }}")
        self.transcript.setStyleSheet(
            f"QTextBrowser {{ background: {tb_bg}; color: {tb_color}; border: 1px solid {tb_border}; border-radius: 6px; padding: 10px; }}"
        )
        self._refresh_content()

    def set_model_info(self, provider_id: str, model_name: str):
        self.provider_id = provider_id
        self.model_name = model_name
        self.title_label.setText(f"✦ {provider_id.title()} · {model_name}")

    def reset_view(self):
        """Reset slot to clean ready state without showing thinking indicators."""
        self.content_buffer = ""
        self.result = None
        self.is_thinking = False
        self.transcript.clear()
        self.metrics_bar.setText("Time: --  ·  Tokens: --  ·  Speed: --  ·  Cost: --")
        self.set_status("Ready", "#757575")
        self.copy_btn.setEnabled(False)
        self.select_btn.setEnabled(False)

    def reset_for_stream(self):
        self.content_buffer = ""
        self.result = None
        self.is_thinking = True
        self.transcript.clear()
        self.metrics_bar.setText("Time: Streaming…  ·  Tokens: --  ·  Speed: --  ·  Cost: --")
        self.set_status("Thinking…", "#F59E0B")
        self.copy_btn.setEnabled(False)
        self.select_btn.setEnabled(False)

    def append_chunk(self, delta: str):
        self.is_thinking = False
        self.content_buffer += delta
        self.set_status("Streaming…", "#10B981")
        self._refresh_content()

    def set_status(self, text: str, color_hex: str):
        self.status_badge.setText(text)
        self.status_badge.setStyleSheet(
            f"color: {color_hex}; font-size: 11px; font-weight: 600; padding: 2px 6px;"
        )

    def set_completed(self, result: ComparisonSlotResult):
        self.result = result
        self.is_thinking = False
        self.content_buffer = result.content
        if result.provider_id:
            self.provider_id = result.provider_id
        if result.model:
            self.model_name = result.model

        if result.is_canceled:
            self.set_status("Canceled", "#9AA3B2")
        elif result.is_success:
            self.set_status("Completed", "#10B981")
            self.copy_btn.setEnabled(bool(self.content_buffer.strip()))
            self.select_btn.setEnabled(bool(self.content_buffer.strip()))
        else:
            self.set_status("Error", "#EF4444")

        # Format metrics bar
        m = result.metrics
        is_local = (result.provider_id.lower() == "ollama")
        metrics_text = (
            f"Time: {m.formatted_elapsed()}  ·  "
            f"Tokens: {m.formatted_tokens()}  ·  "
            f"Speed: {m.formatted_speed()}  ·  "
            f"Cost: {m.formatted_cost(is_local)}"
        )
        self.metrics_bar.setText(metrics_text)
        self._refresh_content(error_message=result.error_message)

    def _refresh_content(self, error_message: Optional[str] = None):
        from src.ui.main_window import MainWindow

        if error_message:
            escaped = html.escape(error_message)
            err_bg = "#2D1517" if self.dark else "#FEE2E2"
            err_border = "#7F1D1D" if self.dark else "#FCA5A5"
            err_color = "#FCA5A5" if self.dark else "#991B1B"
            html_content = (
                f'<div style="background: {err_bg}; border: 1px solid {err_border}; border-radius: 8px; padding: 10px; color: {err_color};">'
                f'<b>Error:</b> {escaped}'
                f'</div>'
            )
            self.transcript.setHtml(html_content)
        elif self.content_buffer:
            html_content = MainWindow.markdown_to_html(self.content_buffer, self.dark)
            self.transcript.setHtml(html_content)
        elif self.is_thinking:
            self.transcript.setHtml('<div style="color: #9AA3B2; font-style: italic;">Connecting & thinking…</div>')
        else:
            self.transcript.clear()

        # Scroll to bottom if near bottom
        sb = self.transcript.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _copy_to_clipboard(self):
        if self.content_buffer:
            QGuiApplication.clipboard().setText(self.content_buffer)
            self.copy_btn.setText("✓ Copied!")
            from PySide6.QtCore import QTimer
            QTimer.singleShot(1500, lambda: self.copy_btn.setText("📋 Copy"))

    def _on_select_clicked(self):
        if self.content_buffer:
            self.select_response.emit(self.slot_id, self.content_buffer)


class ComparisonView(QWidget):
    """Container view hosting slot selectors, responsive side-by-side model cards, and comparative summary."""

    response_committed = Signal(str, str, str)  # (provider_id, model, content)

    def __init__(self, provider_manager: ProviderManager, dark: bool = True, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.provider_manager = provider_manager
        self.dark = dark

        self.slot_cards: Dict[int, ModelSlotCard] = {}
        self.slot_selectors: Dict[int, Dict[str, QWidget]] = {}
        self.current_slots_count: int = 2

        self._build_ui()
        self.apply_theme(self.dark)

    def _build_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(10)

        # 1. Slot Configuration Bar
        self.config_bar = QFrame()
        self.config_bar.setObjectName("Panel")
        self.config_layout = QHBoxLayout(self.config_bar)
        self.config_layout.setContentsMargins(12, 8, 12, 8)
        self.config_layout.setSpacing(10)

        self.slots_box = QHBoxLayout()
        self.slots_box.setSpacing(12)
        self.config_layout.addLayout(self.slots_box, 1)

        # Timeout Selector
        timeout_lbl = QLabel("Timeout:")
        timeout_lbl.setObjectName("Muted")
        self.timeout_combo = QComboBox()
        self.timeout_combo.addItem("30s", 30.0)
        self.timeout_combo.addItem("60s", 60.0)
        self.timeout_combo.addItem("120s", 120.0)
        self.timeout_combo.setCurrentIndex(1)
        self.timeout_combo.setFixedWidth(80)

        self.config_layout.addWidget(timeout_lbl)
        self.config_layout.addWidget(self.timeout_combo)

        # Add Model Button
        self.add_model_btn = QPushButton("+ Add Model")
        self.add_model_btn.setObjectName("Secondary")
        self.add_model_btn.clicked.connect(self._on_add_model_clicked)
        self.config_layout.addWidget(self.add_model_btn)

        root_layout.addWidget(self.config_bar)

        # 2. Side-by-side Response Grid
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(10)
        root_layout.addWidget(self.grid_container, 1)

        # 3. Comparative Summary Banner
        self.summary_banner = QFrame()
        self.summary_banner.setObjectName("Panel")
        self.summary_banner.setVisible(False)
        self.summary_layout = QVBoxLayout(self.summary_banner)
        self.summary_layout.setContentsMargins(12, 8, 12, 8)
        self.summary_layout.setSpacing(4)

        self.summary_title = QLabel("✦ Comparison Summary")
        self.summary_title.setStyleSheet("font-size: 13px; font-weight: 700; color: #818CF8;")
        self.summary_text = QLabel("")
        self.summary_text.setWordWrap(True)
        self.summary_text.setStyleSheet("font-size: 12px; line-height: 1.4;")
        self.summary_layout.addWidget(self.summary_title)
        self.summary_layout.addWidget(self.summary_text)

        root_layout.addWidget(self.summary_banner)

        # Initialize default 2 slots
        self._init_slots()

    def _init_slots(self):
        # Create initial 2 slots with default providers
        self._create_slot(slot_id=0, default_provider="ollama")
        self._create_slot(slot_id=1, default_provider="gemini")
        self._rearrange_grid()

    def _create_slot(self, slot_id: int, default_provider: Optional[str] = None):
        # Selector widgets
        slot_frame = QWidget()
        row = QHBoxLayout(slot_frame)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        num_lbl = QLabel(f"#{slot_id + 1}:")
        num_lbl.setStyleSheet("font-weight: 700;")
        row.addWidget(num_lbl)

        prov_combo = QComboBox()
        prov_combo.setMinimumWidth(100)
        prov_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        providers = self.provider_manager.list_providers()
        for p in providers:
            prov_combo.addItem(p.display_name, p.provider_id)

        model_combo = QComboBox()
        model_combo.setMinimumWidth(140)
        model_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)

        def populate_slot_models():
            model_combo.clear()
            pid = prov_combo.currentData()
            prov = self.provider_manager.get_provider(pid)
            if prov:
                try:
                    models = prov.get_available_models()
                    if models:
                        chat_models = [
                            m for m in models
                            if not any(emb in m.id.lower() for emb in ("embed", "embedding", "bge-", "all-minilm", "arctic-embed"))
                        ]
                        for m in (chat_models if chat_models else models):
                            model_combo.addItem(m.id)
                        return
                except Exception:
                    pass
            # Fallbacks
            if pid == "ollama":
                model_combo.addItem("llama3.2:3b")
            elif pid == "groq":
                model_combo.addItem("openai/gpt-oss-120b")
            elif pid == "openai":
                model_combo.addItem("gpt-4o-mini")
            else:
                model_combo.addItem("gemini-3.5-flash")

        prov_combo.currentIndexChanged.connect(lambda idx: (populate_slot_models(), self._update_card_title(slot_id)))
        model_combo.currentIndexChanged.connect(lambda idx: self._update_card_title(slot_id))

        # Set default provider if given
        if default_provider:
            idx = prov_combo.findData(default_provider)
            if idx >= 0:
                prov_combo.setCurrentIndex(idx)
        populate_slot_models()

        del_btn = QPushButton("✕")
        del_btn.setFixedSize(26, 26)
        del_btn.setObjectName("Secondary")
        del_btn.setToolTip("Remove this model slot")
        del_btn.clicked.connect(lambda: self._remove_slot(slot_id))

        row.addWidget(prov_combo)
        row.addWidget(model_combo)
        row.addWidget(del_btn)

        self.slots_box.addWidget(slot_frame)
        self.slot_selectors[slot_id] = {
            "frame": slot_frame,
            "provider_combo": prov_combo,
            "model_combo": model_combo,
            "del_btn": del_btn,
        }

        # Create card widget
        card = ModelSlotCard(slot_id=slot_id, dark=self.dark)
        card.select_response.connect(self._on_response_committed)
        self.slot_cards[slot_id] = card
        self._update_card_title(slot_id)

    def _update_card_title(self, slot_id: int):
        if slot_id in self.slot_selectors and slot_id in self.slot_cards:
            p_id = self.slot_selectors[slot_id]["provider_combo"].currentData() or ""
            m_id = self.slot_selectors[slot_id]["model_combo"].currentText() or ""
            self.slot_cards[slot_id].set_model_info(p_id, m_id)

    def _on_add_model_clicked(self):
        if len(self.slot_cards) >= 4:
            return
        new_id = 0
        while new_id in self.slot_cards:
            new_id += 1

        defaults = ["ollama", "gemini", "groq", "openai"]
        chosen_default = defaults[new_id % len(defaults)]
        self._create_slot(new_id, chosen_default)
        self._rearrange_grid()

    def _remove_slot(self, slot_id: int):
        if len(self.slot_cards) <= 2:
            return  # Minimum 2 slots

        if slot_id in self.slot_selectors:
            info = self.slot_selectors.pop(slot_id)
            self.slots_box.removeWidget(info["frame"])
            info["frame"].deleteLater()

        if slot_id in self.slot_cards:
            card = self.slot_cards.pop(slot_id)
            self.grid_layout.removeWidget(card)
            card.deleteLater()

        self._rearrange_grid()

    def _rearrange_grid(self):
        """Implement responsive layout: 2 cols for 2 models, 3 cols for 3 models, 2x2 grid for 4 models."""
        # Clear existing items from grid
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            # Do not delete widget, just remove from grid layout

        count = len(self.slot_cards)
        self.add_model_btn.setEnabled(count < 4)

        # Update delete buttons: disable if count <= 2
        for info in self.slot_selectors.values():
            info["del_btn"].setEnabled(count > 2)

        sorted_ids = sorted(self.slot_cards.keys())

        if count == 2:
            # 2 columns side by side
            for col, sid in enumerate(sorted_ids):
                self.grid_layout.addWidget(self.slot_cards[sid], 0, col)
                self.grid_layout.setColumnStretch(col, 1)
        elif count == 3:
            # 3 columns side by side
            for col, sid in enumerate(sorted_ids):
                self.grid_layout.addWidget(self.slot_cards[sid], 0, col)
                self.grid_layout.setColumnStretch(col, 1)
        elif count == 4:
            # 2x2 grid for compact readability
            coords = [(0, 0), (0, 1), (1, 0), (1, 1)]
            for idx, sid in enumerate(sorted_ids):
                r, c = coords[idx]
                self.grid_layout.addWidget(self.slot_cards[sid], r, c)
                self.grid_layout.setColumnStretch(c, 1)

    def get_configured_slots(self) -> List[ComparisonSlotConfig]:
        configs = []
        timeout = float(self.timeout_combo.currentData() or 60.0)
        for sid in sorted(self.slot_selectors.keys()):
            sel = self.slot_selectors[sid]
            pid = sel["provider_combo"].currentData()
            model = sel["model_combo"].currentText()
            configs.append(ComparisonSlotConfig(slot_id=sid, provider_id=pid, model=model, timeout_sec=timeout))
        return configs

    def reset_comparison(self):
        self.summary_banner.setVisible(False)
        for card in self.slot_cards.values():
            card.reset_view()

    def update_slot_chunk(self, slot_id: int, delta: str):
        if slot_id in self.slot_cards:
            self.slot_cards[slot_id].append_chunk(delta)

    def update_slot_status(self, slot_id: int, status: str):
        if slot_id in self.slot_cards:
            color = "#10B981" if status == "Completed" else ("#EF4444" if status == "Error" else "#F59E0B")
            self.slot_cards[slot_id].set_status(status, color)

    def update_slot_result(self, slot_id: int, result: ComparisonSlotResult):
        if slot_id in self.slot_cards:
            self.slot_cards[slot_id].set_completed(result)

    def show_comparison_summary(self, results: List[ComparisonSlotResult]):
        successful = [r for r in results if r.is_success]
        if not successful:
            self.summary_banner.setVisible(False)
            return

        fastest = min(successful, key=lambda r: r.metrics.elapsed_sec)
        summary_lines = [
            f"<b>Fastest model:</b> {fastest.provider_id.title()} · {fastest.model} ({fastest.metrics.formatted_elapsed()})",
        ]

        metrics_items = []
        for r in results:
            if r.is_success:
                metrics_items.append(
                    f"• <b>{r.model}</b>: {r.metrics.formatted_elapsed()} | {r.metrics.formatted_speed()} | {r.metrics.formatted_tokens()}"
                )
            elif r.is_canceled:
                metrics_items.append(f"• <b>{r.model}</b>: <i>Canceled</i>")
            else:
                metrics_items.append(f"• <b>{r.model}</b>: <span style='color: #EF4444;'>Failed</span>")

        summary_lines.append("<br>".join(metrics_items))
        self.summary_text.setText("<br>".join(summary_lines))
        self.summary_banner.setVisible(True)

    def _on_response_committed(self, slot_id: int, content: str):
        if slot_id in self.slot_cards:
            card = self.slot_cards[slot_id]
            self.response_committed.emit(card.provider_id, card.model_name, content)

    def apply_theme(self, dark: bool):
        self.dark = dark
        t = get_theme(dark)
        bg = t.bg_card
        border = t.border_strong
        self.config_bar.setStyleSheet(f"QFrame#Panel {{ background: {bg}; border: 1px solid {border}; border-radius: 8px; }}")
        self.summary_banner.setStyleSheet(f"QFrame#Panel {{ background: {bg}; border: 1px solid {border}; border-radius: 8px; }}")

        for card in self.slot_cards.values():
            card.apply_theme(dark)
