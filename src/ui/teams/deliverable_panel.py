"""Relay AI - Final Deliverable Panel Component.

Prominently displays the synthesized final deliverable from AI Team collaboration,
with rich Markdown rendering, agent attribution, verified execution metrics,
copy-to-clipboard functionality, clear lifecycle states, and theme switching.
"""

from enum import Enum
import html
from typing import Optional, Union

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from src.agents.models import AgentRole, AgentStatus
from src.teams.models import TeamExecutionResult
from src.workflow.models import WorkflowStatus


class DeliverableState(str, Enum):
    """Lifecycle states of the Final Deliverable Panel."""
    EMPTY = "empty"
    LOADING = "loading"
    ERROR = "error"
    READY = "ready"


def render_markdown_safe(text: str, dark: bool = True) -> str:
    """Safely render Markdown to HTML reusing MainWindow.markdown_to_html without circular imports."""
    if not text:
        return ""
    try:
        from src.ui.main_window import MainWindow
        return MainWindow.markdown_to_html(text, dark=dark)
    except Exception:
        escaped = html.escape(text)
        return f"<div style='font-family: sans-serif; white-space: pre-wrap;'>{escaped}</div>"


class FinalDeliverablePanel(QFrame):
    """Prominent native PySide6 widget rendering the synthesized final deliverable."""

    def __init__(
        self,
        dark: bool = True,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.dark = dark
        self._state: DeliverableState = DeliverableState.EMPTY
        self._raw_markdown: str = ""
        self._attribution: str = ""
        self._duration_sec: Optional[float] = None
        self._total_tokens: Optional[int] = None
        self._prompt_tokens: Optional[int] = None
        self._completion_tokens: Optional[int] = None
        self._error_message: Optional[str] = None

        self._init_ui()
        self._apply_theme()
        self.set_empty()

    def _init_ui(self) -> None:
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(14, 12, 14, 12)
        main_layout.setSpacing(10)

        # -------------------------------------------------------------
        # 1. Header Toolbar
        # -------------------------------------------------------------
        self.header_widget = QWidget(self)
        header_layout = QHBoxLayout(self.header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(10)

        # Title
        self.title_label = QLabel("Final Deliverable", self.header_widget)
        self.title_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        header_layout.addWidget(self.title_label)

        # Source Attribution badge
        self.attribution_badge = QLabel("", self.header_widget)
        self.attribution_badge.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        header_layout.addWidget(self.attribution_badge)
        self.attribution_badge.setVisible(False)

        header_layout.addStretch(1)

        # Total Execution Duration
        self.duration_label = QLabel("--", self.header_widget)
        self.duration_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        header_layout.addWidget(self.duration_label)

        # Verified Tokens
        self.tokens_label = QLabel("--", self.header_widget)
        self.tokens_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        header_layout.addWidget(self.tokens_label)

        # Copy to Clipboard Button
        self.copy_btn = QPushButton("Copy Deliverable", self.header_widget)
        self.copy_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.copy_btn.clicked.connect(self.copy_to_clipboard)
        header_layout.addWidget(self.copy_btn)

        main_layout.addWidget(self.header_widget)

        # -------------------------------------------------------------
        # 2. Stacked Content Area (Empty, Loading, Error, Ready)
        # -------------------------------------------------------------
        self.stack = QStackedWidget(self)
        self.stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Index 0: Empty State
        self.empty_widget = QWidget(self.stack)
        empty_layout = QVBoxLayout(self.empty_widget)
        empty_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.setSpacing(6)
        self.empty_icon = QLabel("📋", self.empty_widget)
        self.empty_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_icon.setStyleSheet("font-size: 24px;")
        empty_layout.addWidget(self.empty_icon)
        self.empty_label = QLabel("No final deliverable available yet.", self.empty_widget)
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(self.empty_label)
        self.stack.addWidget(self.empty_widget)

        # Index 1: Loading State
        self.loading_widget = QWidget(self.stack)
        loading_layout = QVBoxLayout(self.loading_widget)
        loading_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        loading_layout.setSpacing(6)
        self.loading_icon = QLabel("⏳", self.loading_widget)
        self.loading_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_icon.setStyleSheet("font-size: 24px;")
        loading_layout.addWidget(self.loading_icon)
        self.loading_label = QLabel("AI Team is executing and synthesizing deliverable...", self.loading_widget)
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        loading_layout.addWidget(self.loading_label)
        self.stack.addWidget(self.loading_widget)

        # Index 2: Error State
        self.error_widget = QWidget(self.stack)
        err_layout = QVBoxLayout(self.error_widget)
        err_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        err_layout.setSpacing(8)
        self.error_title = QLabel("Execution Failed", self.error_widget)
        self.error_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        err_layout.addWidget(self.error_title)
        self.error_label = QLabel("", self.error_widget)
        self.error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.error_label.setWordWrap(True)
        self.error_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        err_layout.addWidget(self.error_label)
        self.stack.addWidget(self.error_widget)

        # Index 3: Ready / Markdown Viewer
        self.markdown_viewer = QTextBrowser(self.stack)
        self.markdown_viewer.setReadOnly(True)
        self.markdown_viewer.setOpenExternalLinks(True)
        self.markdown_viewer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.stack.addWidget(self.markdown_viewer)

        main_layout.addWidget(self.stack, 1)

    # -----------------------------------------------------------------
    # State Transitions & Display API
    # -----------------------------------------------------------------

    @property
    def current_state(self) -> DeliverableState:
        """Returns the current state of the panel."""
        return self._state

    @property
    def raw_markdown(self) -> str:
        """Returns the raw deliverable markdown text."""
        return self._raw_markdown

    @property
    def attribution(self) -> str:
        """Returns the current attribution text."""
        return self._attribution

    def set_empty(self) -> None:
        """Transition to clean Empty state."""
        self._state = DeliverableState.EMPTY
        self._raw_markdown = ""
        self._attribution = ""
        self._duration_sec = None
        self._total_tokens = None
        self._error_message = None

        self.attribution_badge.setVisible(False)
        self.duration_label.setText("--")
        self.tokens_label.setText("--")
        self.copy_btn.setEnabled(False)
        self.stack.setCurrentWidget(self.empty_widget)

    def set_loading(self, message: Optional[str] = None) -> None:
        """Transition to Loading state with an optional custom progress message."""
        self._state = DeliverableState.LOADING
        self.loading_label.setText(message or "AI Team is executing and synthesizing deliverable...")
        self.attribution_badge.setVisible(False)
        self.copy_btn.setEnabled(False)
        self.stack.setCurrentWidget(self.loading_widget)

    def set_error(self, error_message: str) -> None:
        """Transition to Error state displaying execution failure details."""
        self._state = DeliverableState.ERROR
        self._error_message = error_message or "An unspecified error occurred during team execution."
        self.error_label.setText(self._error_message)
        self.attribution_badge.setVisible(False)
        self.copy_btn.setEnabled(False)
        self.stack.setCurrentWidget(self.error_widget)

    def set_deliverable(
        self,
        content: str,
        attribution: Optional[str] = None,
        duration: Optional[float] = None,
        tokens: Optional[int] = None,
    ) -> None:
        """Transition to Ready state rendering the deliverable with full attribution and metrics."""
        self._state = DeliverableState.READY
        self._raw_markdown = content or ""
        self._attribution = attribution or ""
        self._duration_sec = duration
        self._total_tokens = tokens

        # Attribution badge
        if self._attribution:
            self.attribution_badge.setText(f"Source: {self._attribution}")
            self.attribution_badge.setVisible(True)
        else:
            self.attribution_badge.setVisible(False)

        # Duration
        if self._duration_sec is not None:
            self.duration_label.setText(f"⏱ {self._duration_sec:.2f}s")
        else:
            self.duration_label.setText("--")

        # Verified tokens (strictly verified, never fabricated)
        if self._total_tokens is not None:
            self.tokens_label.setText(f"🪙 {self._total_tokens:,} tok")
            self.tokens_label.setToolTip(f"Verified Total Tokens: {self._total_tokens:,}")
        else:
            self.tokens_label.setText("--")
            self.tokens_label.setToolTip("Token metrics unavailable")

        # Render Markdown to HTML
        html_content = render_markdown_safe(self._raw_markdown, dark=self.dark)
        self.markdown_viewer.setHtml(html_content)

        self.copy_btn.setEnabled(bool(self._raw_markdown.strip()))
        self.stack.setCurrentWidget(self.markdown_viewer)

    def update_from_result(self, result: TeamExecutionResult) -> None:
        """Update deliverable panel state and attribution directly from a TeamExecutionResult."""
        if not result:
            self.set_empty()
            return

        if result.final_deliverable and result.final_deliverable.strip():
            # Determine source attribution from step records
            attribution = self._determine_attribution(result)
            self.set_deliverable(
                content=result.final_deliverable,
                attribution=attribution,
                duration=result.total_duration_sec,
                tokens=result.total_tokens,
            )
        elif result.status in (WorkflowStatus.FAILED, WorkflowStatus.CANCELLED):
            err = result.error_summary or f"Team execution terminated with status: {result.status.value}."
            self.set_error(err)
        elif result.status == WorkflowStatus.RUNNING:
            self.set_loading()
        else:
            self.set_empty()

    def _determine_attribution(self, result: TeamExecutionResult) -> str:
        """Determine whether Finalizer Agent or Creator Agent (or other) authored the final deliverable."""
        if not result.step_records:
            return "AI Team"

        # Check if Finalizer executed cleanly
        for rec in reversed(result.step_records):
            if rec.agent_role == AgentRole.FINALIZER and rec.status == AgentStatus.COMPLETED:
                return "Finalizer Agent"

        # Otherwise check if Creator executed cleanly
        for rec in reversed(result.step_records):
            if rec.agent_role == AgentRole.CREATOR and rec.status == AgentStatus.COMPLETED:
                return "Creator Agent"

        # Fallback to the last completed step role
        for rec in reversed(result.step_records):
            if rec.status == AgentStatus.COMPLETED:
                role_val = rec.agent_role.value if hasattr(rec.agent_role, "value") else str(rec.agent_role)
                return f"{role_val.capitalize()} Agent"

        return "AI Team"

    # -----------------------------------------------------------------
    # Copy to Clipboard
    # -----------------------------------------------------------------

    def copy_to_clipboard(self) -> None:
        """Copy the raw deliverable Markdown text to the system clipboard with temporary visual feedback."""
        if not self._raw_markdown:
            return

        clipboard = QGuiApplication.clipboard()
        if clipboard:
            clipboard.setText(self._raw_markdown)

        self.copy_btn.setText("✓ Copied!")
        QTimer.singleShot(1500, self._restore_copy_btn)

    def _restore_copy_btn(self) -> None:
        """Restore default copy button text."""
        self.copy_btn.setText("Copy Deliverable")

    # -----------------------------------------------------------------
    # Theme Support
    # -----------------------------------------------------------------

    def set_dark_theme(self, dark: bool) -> None:
        """Switch dark/light theme and update styling and markdown rendering."""
        if self.dark == dark:
            return
        self.dark = dark
        self._apply_theme()

        # Re-render markdown if currently showing content
        if self._state == DeliverableState.READY and self._raw_markdown:
            html_content = render_markdown_safe(self._raw_markdown, dark=self.dark)
            self.markdown_viewer.setHtml(html_content)

    def _apply_theme(self) -> None:
        if self.dark:
            panel_bg = "#161B22"
            panel_border = "#30363D"
            title_color = "#F0F6FC"
            attr_bg = "#1F2937"
            attr_color = "#60A5FA"
            attr_border = "#3B82F6"
            metrics_color = "#8B949E"
            btn_bg = "#21262D"
            btn_border = "#363B42"
            btn_color = "#C9D1D9"
            btn_hover = "#30363D"
            viewer_bg = "#0D1117"
            viewer_border = "#21262D"
            empty_color = "#8B949E"
            err_title_color = "#F85149"
            err_box_bg = "#211516"
            err_border = "#671B1D"
            err_text = "#FF7B72"
        else:
            panel_bg = "#FFFFFF"
            panel_border = "#E2E8F0"
            title_color = "#0F172A"
            attr_bg = "#EFF6FF"
            attr_color = "#1D4ED8"
            attr_border = "#93C5FD"
            metrics_color = "#64748B"
            btn_bg = "#F8FAFC"
            btn_border = "#CBD5E1"
            btn_color = "#334155"
            btn_hover = "#E2E8F0"
            viewer_bg = "#F8FAFC"
            viewer_border = "#E2E8F0"
            empty_color = "#94A3B8"
            err_title_color = "#DC2626"
            err_box_bg = "#FEF2F2"
            err_border = "#FCA5A5"
            err_text = "#991B1B"

        self.setStyleSheet(
            f"FinalDeliverablePanel {{"
            f"  background-color: {panel_bg};"
            f"  border: 1px solid {panel_border};"
            f"  border-radius: 8px;"
            f"}}"
        )

        self.title_label.setStyleSheet(
            f"color: {title_color}; font-size: 14px; font-weight: bold;"
        )

        self.attribution_badge.setStyleSheet(
            f"background-color: {attr_bg}; color: {attr_color}; border: 1px solid {attr_border};"
            f"font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 4px;"
        )

        self.duration_label.setStyleSheet(
            f"color: {metrics_color}; font-size: 11px; font-family: Consolas, monospace;"
        )

        self.tokens_label.setStyleSheet(
            f"color: {metrics_color}; font-size: 11px; font-family: Consolas, monospace;"
        )

        self.copy_btn.setStyleSheet(
            f"QPushButton {{"
            f"  background-color: {btn_bg};"
            f"  color: {btn_color};"
            f"  border: 1px solid {btn_border};"
            f"  border-radius: 5px;"
            f"  padding: 4px 12px;"
            f"  font-size: 12px;"
            f"  font-weight: 500;"
            f"}}"
            f"QPushButton:hover {{"
            f"  background-color: {btn_hover};"
            f"}}"
            f"QPushButton:disabled {{"
            f"  opacity: 0.5;"
            f"  color: {metrics_color};"
            f"}}"
        )

        self.empty_label.setStyleSheet(f"color: {empty_color}; font-size: 13px;")
        self.loading_label.setStyleSheet(f"color: {empty_color}; font-size: 13px; font-style: italic;")

        self.error_widget.setStyleSheet(
            f"background-color: {err_box_bg}; border: 1px solid {err_border}; border-radius: 6px; padding: 16px;"
        )
        self.error_title.setStyleSheet(f"color: {err_title_color}; font-size: 14px; font-weight: bold;")
        self.error_label.setStyleSheet(f"color: {err_text}; font-size: 12px;")

        self.markdown_viewer.setStyleSheet(
            f"QTextBrowser {{"
            f"  background-color: {viewer_bg};"
            f"  border: 1px solid {viewer_border};"
            f"  border-radius: 6px;"
            f"  padding: 12px;"
            f"  font-size: 13px;"
            f"  line-height: 1.5;"
            f"}}"
        )
