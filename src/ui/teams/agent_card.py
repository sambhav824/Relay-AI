"""Relay AI - Agent Step Card Component.

Displays individual AI agent execution results, telemetry metrics, status badges,
expandable output, error details, and revision history with critic feedback.
"""

from enum import Enum
import html
from typing import Any, Dict, List, Optional, Union

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from src.agents.models import AgentRole, AgentStatus
from src.teams.observer import TeamExecutionSnapshot
from src.workflow.models import StepExecutionRecord, WorkflowStatus


class CardStatus(str, Enum):
    """Normalized status values for display badges."""
    PENDING = "Pending"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"
    SKIPPED = "Skipped"


def normalize_card_status(
    status: Union[AgentStatus, WorkflowStatus, CardStatus, str]
) -> CardStatus:
    """Normalize any domain or string status into a CardStatus enum."""
    if isinstance(status, CardStatus):
        return status

    raw = status.value if hasattr(status, "value") else str(status)
    s = raw.strip().lower()

    if s in ("pending", "idle", "queued", "initialized"):
        return CardStatus.PENDING
    elif s in ("running", "active", "in_progress", "busy"):
        return CardStatus.RUNNING
    elif s in ("completed", "done", "success", "partial_success"):
        return CardStatus.COMPLETED
    elif s in ("failed", "error", "aborted", "timeout"):
        return CardStatus.FAILED
    elif s in ("cancelled", "canceled"):
        return CardStatus.CANCELLED
    elif s in ("skipped", "skip"):
        return CardStatus.SKIPPED

    return CardStatus.PENDING


class AgentStepCard(QFrame):
    """Reusable visual card displaying execution state, output, and revision history for one agent step."""

    revision_selected = Signal(int)  # Emitted when a revision index is chosen by the user

    def __init__(
        self,
        step_id: str = "",
        role: Union[AgentRole, str] = AgentRole.CREATOR,
        order: int = 1,
        provider_id: str = "",
        model_id: str = "",
        dark: bool = True,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.step_id = step_id
        self.role = role if isinstance(role, AgentRole) else AgentRole(str(role).lower())
        self.order = order
        self.provider_id = provider_id
        self.model_id = model_id
        self.dark = dark

        self._status: CardStatus = CardStatus.PENDING
        self._duration_sec: Optional[float] = None
        self._total_tokens: Optional[int] = None
        self._prompt_tokens: Optional[int] = None
        self._completion_tokens: Optional[int] = None
        self._current_output: str = ""
        self._latest_output: Optional[str] = None
        self._error_message: Optional[str] = None
        self._critic_feedback: Optional[str] = None
        self._critic_reason: Optional[str] = None
        self._records: List[StepExecutionRecord] = []
        self._is_expanded: bool = True

        self._init_ui()
        self._apply_theme()
        self._refresh_header()
        self._refresh_metrics()

    def _init_ui(self) -> None:
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 10, 12, 10)
        main_layout.setSpacing(8)

        # -------------------------------------------------------------
        # 1. Header Bar: Order, Role, Provider/Model, Metrics, Status, Expand
        # -------------------------------------------------------------
        self.header_widget = QWidget(self)
        header_layout = QHBoxLayout(self.header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(10)

        # Step Order badge
        self.order_label = QLabel(f"#{self.order}", self.header_widget)
        self.order_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        header_layout.addWidget(self.order_label)

        # Agent Role badge
        role_display = self.role.value.capitalize() if hasattr(self.role, "value") else str(self.role).capitalize()
        self.role_label = QLabel(role_display, self.header_widget)
        self.role_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        header_layout.addWidget(self.role_label)

        # Provider and Model info
        self.provider_model_label = QLabel("", self.header_widget)
        self.provider_model_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        header_layout.addWidget(self.provider_model_label)

        # Execution duration
        self.duration_label = QLabel("--", self.header_widget)
        self.duration_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        header_layout.addWidget(self.duration_label)

        # Verified tokens
        self.tokens_label = QLabel("--", self.header_widget)
        self.tokens_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        header_layout.addWidget(self.tokens_label)

        # Status badge
        self.status_badge = QLabel(self._status.value, self.header_widget)
        self.status_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_badge.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        header_layout.addWidget(self.status_badge)

        # Expand / Collapse toggle button
        self.expand_btn = QPushButton("▼", self.header_widget)
        self.expand_btn.setFixedSize(26, 26)
        self.expand_btn.setToolTip("Toggle agent output")
        self.expand_btn.clicked.connect(self.toggle_expanded)
        header_layout.addWidget(self.expand_btn)

        main_layout.addWidget(self.header_widget)

        # -------------------------------------------------------------
        # 2. Expandable Body Container
        # -------------------------------------------------------------
        self.body_container = QWidget(self)
        self.body_layout = QVBoxLayout(self.body_container)
        self.body_layout.setContentsMargins(0, 4, 0, 0)
        self.body_layout.setSpacing(6)

        # Revision Selector Bar (hidden until revisions are available)
        self.revision_bar = QWidget(self.body_container)
        rev_layout = QHBoxLayout(self.revision_bar)
        rev_layout.setContentsMargins(0, 0, 0, 0)
        rev_layout.setSpacing(8)

        self.revision_title = QLabel("Revision History:", self.revision_bar)
        self.revision_title.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        rev_layout.addWidget(self.revision_title)

        self.revision_combo = QComboBox(self.revision_bar)
        self.revision_combo.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.revision_combo.currentIndexChanged.connect(self._on_revision_changed)
        rev_layout.addWidget(self.revision_combo)

        rev_layout.addStretch(1)
        self.body_layout.addWidget(self.revision_bar)
        self.revision_bar.setVisible(False)

        # Critic Feedback Box (hidden until feedback exists)
        self.critic_box = QFrame(self.body_container)
        self.critic_box.setFrameShape(QFrame.Shape.StyledPanel)
        critic_layout = QVBoxLayout(self.critic_box)
        critic_layout.setContentsMargins(10, 8, 10, 8)
        critic_layout.setSpacing(4)

        self.critic_title = QLabel("Critic Feedback", self.critic_box)
        critic_layout.addWidget(self.critic_title)

        self.critic_text = QLabel("", self.critic_box)
        self.critic_text.setWordWrap(True)
        self.critic_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        critic_layout.addWidget(self.critic_text)

        self.body_layout.addWidget(self.critic_box)
        self.critic_box.setVisible(False)

        # Error Details Box (hidden until error exists)
        self.error_box = QFrame(self.body_container)
        self.error_box.setFrameShape(QFrame.Shape.StyledPanel)
        err_layout = QVBoxLayout(self.error_box)
        err_layout.setContentsMargins(10, 8, 10, 8)
        err_layout.setSpacing(4)

        self.error_title = QLabel("Execution Error", self.error_box)
        err_layout.addWidget(self.error_title)

        self.error_text = QLabel("", self.error_box)
        self.error_text.setWordWrap(True)
        self.error_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        err_layout.addWidget(self.error_text)

        self.body_layout.addWidget(self.error_box)
        self.error_box.setVisible(False)

        # Agent Output Viewer
        self.output_viewer = QTextBrowser(self.body_container)
        self.output_viewer.setReadOnly(True)
        self.output_viewer.setMinimumHeight(70)
        self.output_viewer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.body_layout.addWidget(self.output_viewer)

        main_layout.addWidget(self.body_container)

    # -----------------------------------------------------------------
    # Public Update API
    # -----------------------------------------------------------------

    def set_status(self, status: Union[AgentStatus, WorkflowStatus, CardStatus, str]) -> None:
        """Update card execution status and refresh badge styling."""
        self._status = normalize_card_status(status)
        self.status_badge.setText(self._status.value)
        self._apply_badge_style()

    @property
    def current_status(self) -> CardStatus:
        """Returns the current CardStatus."""
        return self._status

    def set_provider_model(self, provider_id: str, model_id: str) -> None:
        """Update provider and model information."""
        self.provider_id = provider_id or self.provider_id
        self.model_id = model_id or self.model_id
        self._refresh_header()

    def set_duration(self, duration_sec: Optional[float]) -> None:
        """Update execution duration. Shows '--' when unavailable."""
        self._duration_sec = duration_sec
        self._refresh_metrics()

    def set_tokens(
        self,
        total_tokens: Optional[int],
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
    ) -> None:
        """Update verified token counts. Never fabricates values if None."""
        self._total_tokens = total_tokens
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self._refresh_metrics()

    def set_output(self, text: str) -> None:
        """Set displayed agent output text."""
        self._current_output = text or ""
        self.output_viewer.setPlainText(self._current_output)
        if self._current_output and not self._is_expanded:
            self.set_expanded(True)

    def set_error(self, error_message: Optional[str]) -> None:
        """Display or clear error details."""
        self._error_message = error_message
        if error_message:
            self.error_text.setText(str(error_message))
            self.error_box.setVisible(True)
        else:
            self.error_text.clear()
            self.error_box.setVisible(False)

    def set_critic_feedback(
        self, feedback: Optional[str], reason: Optional[str] = None
    ) -> None:
        """Display or clear associated Critic review feedback."""
        self._critic_feedback = feedback
        self._critic_reason = reason
        if feedback or reason:
            title = f"Critic Review: {reason}" if reason else "Critic Review Feedback"
            self.critic_title.setText(title)
            self.critic_text.setText(feedback or "")
            self.critic_box.setVisible(True)
        else:
            self.critic_title.setText("Critic Feedback")
            self.critic_text.clear()
            self.critic_box.setVisible(False)

    def update_from_snapshot(self, snapshot: TeamExecutionSnapshot) -> None:
        """Update live telemetry metrics and status from a TeamExecutionSnapshot.

        Snapshot is telemetry only and does not supply full textual output.
        """
        self.set_status(snapshot.status)
        if snapshot.provider_id or snapshot.model_id:
            self.set_provider_model(
                snapshot.provider_id or self.provider_id,
                snapshot.model_id or self.model_id,
            )
        if snapshot.duration_sec > 0 or snapshot.finished_at is not None:
            self.set_duration(snapshot.duration_sec)
        if snapshot.verified_total_tokens is not None or snapshot.verified_prompt_tokens is not None:
            self.set_tokens(
                total_tokens=snapshot.verified_total_tokens,
                prompt_tokens=snapshot.verified_prompt_tokens,
                completion_tokens=snapshot.verified_completion_tokens,
            )
        if snapshot.error:
            self.set_error(snapshot.error)

    def set_records(
        self,
        records: List[StepExecutionRecord],
        latest_output: Optional[str] = None,
    ) -> None:
        """Populate historical revision records and latest valid output.

        Defaults to displaying the latest valid revision.
        """
        self._records = list(records)
        self._latest_output = latest_output

        if not self._records:
            if latest_output is not None:
                self.set_output(latest_output)
            self.revision_bar.setVisible(False)
            return

        # Populate revision dropdown
        self.revision_combo.blockSignals(True)
        self.revision_combo.clear()

        for idx, rec in enumerate(self._records):
            is_latest = (idx == len(self._records) - 1)
            iter_num = rec.revision_iteration
            label = f"Revision {iter_num}" if iter_num > 0 else "Initial Draft"
            if is_latest:
                label += " (Latest)"
            self.revision_combo.addItem(label, userData=idx)

        # Display latest revision by default
        latest_idx = len(self._records) - 1
        self.revision_combo.setCurrentIndex(latest_idx)
        self.revision_combo.blockSignals(False)

        # Show revision bar when multiple revisions or non-zero revision iteration exist
        self.revision_bar.setVisible(len(self._records) > 1 or self._records[0].revision_iteration > 0)

        # Apply latest revision content
        self._display_record_at_index(latest_idx)

    def _on_revision_changed(self, index: int) -> None:
        """Handle revision selection from version dropdown."""
        if index < 0 or index >= len(self._records):
            return
        self._display_record_at_index(index)
        self.revision_selected.emit(index)

    def _display_record_at_index(self, index: int) -> None:
        """Render a specific historical record turn."""
        if index < 0 or index >= len(self._records):
            return

        record = self._records[index]
        is_latest = (index == len(self._records) - 1)

        # 1. Output content
        if is_latest and self._latest_output is not None:
            output_text = self._latest_output
        elif record.result and record.result.output:
            output_text = record.result.output
        else:
            output_text = ""
        self.set_output(output_text)

        # 2. Duration
        self.set_duration(record.duration_sec)

        # 3. Tokens (strictly verified, never fabricated)
        if record.result and record.result.metadata:
            meta = record.result.metadata
            self.set_tokens(
                total_tokens=meta.total_tokens,
                prompt_tokens=meta.prompt_tokens,
                completion_tokens=meta.completion_tokens,
            )
            p_id = meta.provider_id or self.provider_id
            m_id = meta.model_id or self.model_id
            self.set_provider_model(p_id, m_id)
        else:
            self.set_tokens(None)

        # 4. Status
        self.set_status(record.status)

        # 5. Critic review feedback
        if record.critic_review and (record.critic_review.feedback or record.critic_review.reason):
            self.set_critic_feedback(record.critic_review.feedback, record.critic_review.reason)
        else:
            self.set_critic_feedback(None)

        # 6. Error details
        err = record.error or (record.result.error if record.result else None)
        self.set_error(err)

    # -----------------------------------------------------------------
    # Expand / Collapse & Theme
    # -----------------------------------------------------------------

    def toggle_expanded(self) -> None:
        """Toggle output viewer visibility."""
        self.set_expanded(not self._is_expanded)

    def set_expanded(self, expanded: bool) -> None:
        """Set output viewer expanded state."""
        self._is_expanded = expanded
        self.body_container.setVisible(expanded)
        self.expand_btn.setText("▼" if expanded else "▶")

    def is_expanded(self) -> bool:
        """Returns True if body container is expanded."""
        return self._is_expanded

    def set_dark_theme(self, dark: bool) -> None:
        """Update dark/light theme state and restyle component."""
        if self.dark == dark:
            return
        self.dark = dark
        self._apply_theme()
        self._refresh_header()
        self._refresh_metrics()
        self._apply_badge_style()

    # -----------------------------------------------------------------
    # Internal Styling & Refresh
    # -----------------------------------------------------------------

    def _refresh_header(self) -> None:
        # Provider & Model
        if self.provider_id and self.model_id:
            info = f"{self.provider_id} / {self.model_id}"
        elif self.model_id:
            info = self.model_id
        else:
            info = "--"
        self.provider_model_label.setText(info)

    def _refresh_metrics(self) -> None:
        # Duration
        if self._duration_sec is not None:
            self.duration_label.setText(f"{self._duration_sec:.2f}s")
        else:
            self.duration_label.setText("--")

        # Tokens
        if self._total_tokens is not None:
            self.tokens_label.setText(f"{self._total_tokens:,} tok")
            tooltip = f"Total: {self._total_tokens:,}"
            if self._prompt_tokens is not None and self._completion_tokens is not None:
                tooltip += f"\nPrompt: {self._prompt_tokens:,}\nCompletion: {self._completion_tokens:,}"
            self.tokens_label.setToolTip(tooltip)
        else:
            self.tokens_label.setText("--")
            self.tokens_label.setToolTip("Tokens unavailable")

    def _apply_theme(self) -> None:
        """Apply comprehensive styling according to active theme."""
        if self.dark:
            card_bg = "#1A1F2C"
            card_border = "#2B3545"
            text_pri = "#F1F5F9"
            text_sec = "#94A3B8"
            text_muted = "#64748B"
            order_bg = "#263042"
            order_color = "#93C5FD"
            role_bg = "#1E293B"
            role_color = "#38BDF8"
            box_bg = "#11141A"
            box_border = "#2A3140"
            box_color = "#E2E8F0"
            critic_bg = "#2B2113"
            critic_border = "#92400E"
            critic_title = "#FDE68A"
            critic_text = "#FEF3C7"
            err_bg = "#2B1416"
            err_border = "#991B1B"
            err_title = "#FCA5A5"
            err_text = "#FECACA"
            combo_bg = "#1E2430"
            combo_border = "#334155"
        else:
            card_bg = "#FFFFFF"
            card_border = "#E2E8F0"
            text_pri = "#0F172A"
            text_sec = "#475569"
            text_muted = "#94A3B8"
            order_bg = "#EFF6FF"
            order_color = "#1D4ED8"
            role_bg = "#F0F9FF"
            role_color = "#0284C7"
            box_bg = "#F8FAFC"
            box_border = "#CBD5E1"
            box_color = "#1E293B"
            critic_bg = "#FFFBEB"
            critic_border = "#F59E0B"
            critic_title = "#92400E"
            critic_text = "#78350F"
            err_bg = "#FEF2F2"
            err_border = "#EF4444"
            err_title = "#991B1B"
            err_text = "#7F1D1D"
            combo_bg = "#FFFFFF"
            combo_border = "#CBD5E1"

        self.setStyleSheet(
            f"AgentStepCard {{"
            f"  background-color: {card_bg};"
            f"  border: 1px solid {card_border};"
            f"  border-radius: 8px;"
            f"}}"
        )

        self.order_label.setStyleSheet(
            f"background-color: {order_bg}; color: {order_color};"
            f"font-weight: bold; font-size: 11px; padding: 3px 7px; border-radius: 4px;"
        )

        self.role_label.setStyleSheet(
            f"background-color: {role_bg}; color: {role_color};"
            f"font-weight: 600; font-size: 12px; padding: 3px 8px; border-radius: 4px;"
        )

        self.provider_model_label.setStyleSheet(
            f"color: {text_sec}; font-size: 12px; font-family: Consolas, monospace;"
        )

        self.duration_label.setStyleSheet(
            f"color: {text_muted}; font-size: 11px; font-family: Consolas, monospace;"
        )

        self.tokens_label.setStyleSheet(
            f"color: {text_muted}; font-size: 11px; font-family: Consolas, monospace;"
        )

        self.expand_btn.setStyleSheet(
            f"QPushButton {{"
            f"  background-color: transparent;"
            f"  color: {text_sec};"
            f"  border: 1px solid {combo_border};"
            f"  border-radius: 4px;"
            f"  font-size: 10px;"
            f"}}"
            f"QPushButton:hover {{"
            f"  background-color: {order_bg};"
            f"  color: {order_color};"
            f"}}"
        )

        self.revision_title.setStyleSheet(f"color: {text_sec}; font-size: 12px; font-weight: 600;")
        self.revision_combo.setStyleSheet(
            f"QComboBox {{"
            f"  background-color: {combo_bg};"
            f"  color: {text_pri};"
            f"  border: 1px solid {combo_border};"
            f"  border-radius: 4px;"
            f"  padding: 2px 8px;"
            f"  font-size: 12px;"
            f"}}"
            f"QComboBox::drop-down {{"
            f"  border: none;"
            f"}}"
        )

        self.critic_box.setStyleSheet(
            f"QFrame {{"
            f"  background-color: {critic_bg};"
            f"  border: 1px solid {critic_border};"
            f"  border-radius: 6px;"
            f"}}"
        )
        self.critic_title.setStyleSheet(f"color: {critic_title}; font-weight: bold; font-size: 11px;")
        self.critic_text.setStyleSheet(f"color: {critic_text}; font-size: 12px;")

        self.error_box.setStyleSheet(
            f"QFrame {{"
            f"  background-color: {err_bg};"
            f"  border: 1px solid {err_border};"
            f"  border-radius: 6px;"
            f"}}"
        )
        self.error_title.setStyleSheet(f"color: {err_title}; font-weight: bold; font-size: 11px;")
        self.error_text.setStyleSheet(f"color: {err_text}; font-size: 12px;")

        self.output_viewer.setStyleSheet(
            f"QTextBrowser {{"
            f"  background-color: {box_bg};"
            f"  color: {box_color};"
            f"  border: 1px solid {box_border};"
            f"  border-radius: 6px;"
            f"  padding: 8px;"
            f"  font-size: 12px;"
            f"  line-height: 1.4;"
            f"}}"
        )

        self._apply_badge_style()

    def _apply_badge_style(self) -> None:
        """Update status badge color based on current status and theme."""
        st = self._status
        if self.dark:
            if st == CardStatus.RUNNING:
                bg, fg, border = "#1E3A8A", "#93C5FD", "#3B82F6"
            elif st == CardStatus.COMPLETED:
                bg, fg, border = "#064E3B", "#6EE7B7", "#10B981"
            elif st == CardStatus.FAILED:
                bg, fg, border = "#7F1D1D", "#FCA5A5", "#EF4444"
            elif st == CardStatus.CANCELLED:
                bg, fg, border = "#78350F", "#FCD34D", "#F59E0B"
            elif st == CardStatus.SKIPPED:
                bg, fg, border = "#334155", "#CBD5E1", "#64748B"
            else:  # PENDING
                bg, fg, border = "#1E293B", "#94A3B8", "#475569"
        else:
            if st == CardStatus.RUNNING:
                bg, fg, border = "#DBEAFE", "#1D4ED8", "#93C5FD"
            elif st == CardStatus.COMPLETED:
                bg, fg, border = "#DCFCE7", "#15803D", "#86EFAC"
            elif st == CardStatus.FAILED:
                bg, fg, border = "#FEE2E2", "#B91C1C", "#FCA5A5"
            elif st == CardStatus.CANCELLED:
                bg, fg, border = "#FEF3C7", "#B45309", "#FCD34D"
            elif st == CardStatus.SKIPPED:
                bg, fg, border = "#F1F5F9", "#475569", "#CBD5E1"
            else:  # PENDING
                bg, fg, border = "#F8FAFC", "#64748B", "#E2E8F0"

        self.status_badge.setStyleSheet(
            f"background-color: {bg}; color: {fg}; border: 1px solid {border};"
            f"font-weight: bold; font-size: 11px; padding: 2px 10px; border-radius: 10px;"
        )
