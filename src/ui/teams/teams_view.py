"""Relay AI - AI Teams Master View.

Provides a unified PySide6 workspace for configuring, executing, monitoring,
and inspecting AI Team collaboration workflows with real-time agent timelines,
deliverable rendering, and in-memory run history.
"""

from dataclasses import dataclass, field
from datetime import datetime
import logging
import time
from typing import Dict, List, Optional
import uuid

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.agents.models import AgentRole, AgentStatus
from src.providers.manager import ProviderManager, get_default_provider_manager
from src.teams.factory import TeamFactory
from src.teams.models import TeamConfig, TeamExecutionResult
from src.teams.observer import TeamExecutionSnapshot
from src.teams.orchestrator import TeamOrchestrator
from src.ui.teams.agent_card import AgentStepCard, CardStatus
from src.ui.teams.config_panel import TeamConfigPanel
from src.ui.teams.deliverable_panel import FinalDeliverablePanel
from src.ui.teams.worker import TeamExecutionWorker, safely_anchor_worker
from src.workflow.models import StepExecutionRecord, WorkflowStatus

logger = logging.getLogger("RelayAI.AITeamsView")


@dataclass
class TeamRunHistoryItem:
    """In-memory record of an individual AI Team execution run."""
    run_id: str
    task: str
    start_time: str
    status: str
    duration_sec: float = 0.0
    final_deliverable: str = ""
    step_outputs: Dict[str, str] = field(default_factory=dict)
    step_records: List[StepExecutionRecord] = field(default_factory=list)
    revision_count: int = 0
    total_tokens: Optional[int] = None
    error_summary: Optional[str] = None
    attribution: str = ""


class AITeamsView(QWidget):
    """Master View orchestrating the entire AI Teams collaborative workflow."""

    def __init__(
        self,
        provider_manager: Optional[ProviderManager] = None,
        dark: bool = True,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.provider_manager = provider_manager or get_default_provider_manager()
        self.dark = dark

        self._active_worker: Optional[TeamExecutionWorker] = None
        self._is_running: bool = False
        self._current_run_id: str = ""
        self._run_history: List[TeamRunHistoryItem] = []
        self._step_cards: Dict[AgentRole, AgentStepCard] = {}

        self._init_ui()
        self._apply_theme()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(14, 12, 14, 12)
        main_layout.setSpacing(10)

        # -------------------------------------------------------------
        # 1. Header Bar
        # -------------------------------------------------------------
        self.header_widget = QWidget(self)
        header_layout = QVBoxLayout(self.header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(2)

        self.title_label = QLabel("AI Teams Collaboration", self.header_widget)
        self.title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        header_layout.addWidget(self.title_label)

        self.subtitle_label = QLabel(
            "Orchestrate multi-agent collaboration with specialized Planner, Researcher, Creator, Critic, and Finalizer agents.",
            self.header_widget,
        )
        self.subtitle_label.setStyleSheet("font-size: 12px; color: #64748B;")
        header_layout.addWidget(self.subtitle_label)

        main_layout.addWidget(self.header_widget)

        # -------------------------------------------------------------
        # 2. Main Content Splitter (Left: Config & History; Right: Task & Results)
        # -------------------------------------------------------------
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # Left Container: Configuration Panel + Run History
        self.left_pane = QWidget(self.main_splitter)
        left_layout = QVBoxLayout(self.left_pane)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(10)

        # Configuration Panel (inside scroll area)
        self.config_scroll = QScrollArea(self.left_pane)
        self.config_scroll.setWidgetResizable(True)
        self.config_panel = TeamConfigPanel(
            provider_manager=self.provider_manager,
            dark=self.dark,
            parent=self.config_scroll,
        )
        self.config_scroll.setWidget(self.config_panel)
        left_layout.addWidget(self.config_scroll, 3)

        # Recent History Container
        self.history_group = QFrame(self.left_pane)
        self.history_group.setFrameShape(QFrame.Shape.StyledPanel)
        hist_layout = QVBoxLayout(self.history_group)
        hist_layout.setContentsMargins(8, 8, 8, 8)
        hist_layout.setSpacing(6)

        hist_title = QLabel("Recent Team Runs", self.history_group)
        hist_title.setStyleSheet("font-weight: bold; font-size: 12px;")
        hist_layout.addWidget(hist_title)

        self.history_list = QListWidget(self.history_group)
        self.history_list.setMinimumHeight(120)
        self.history_list.itemClicked.connect(self._on_history_item_clicked)
        hist_layout.addWidget(self.history_list)

        left_layout.addWidget(self.history_group, 1)

        self.main_splitter.addWidget(self.left_pane)

        # Right Container: Task Input, Controls, Timeline, Deliverable
        self.right_pane = QWidget(self.main_splitter)
        right_layout = QVBoxLayout(self.right_pane)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.setSpacing(10)

        # Task Input Area
        task_box = QWidget(self.right_pane)
        task_layout = QVBoxLayout(task_box)
        task_layout.setContentsMargins(0, 0, 0, 0)
        task_layout.setSpacing(4)

        task_header = QLabel("Task Objective / Prompt:", task_box)
        task_header.setStyleSheet("font-weight: 600; font-size: 12px;")
        task_layout.addWidget(task_header)

        self.task_input = QPlainTextEdit(task_box)
        self.task_input.setPlaceholderText(
            "Enter the goal, requirements, or prompt for the AI Team to execute..."
        )
        self.task_input.setMinimumHeight(70)
        self.task_input.setMaximumHeight(130)
        task_layout.addWidget(self.task_input)

        right_layout.addWidget(task_box)

        # Control Action Bar (Run, Cancel, Status Label)
        controls_bar = QWidget(self.right_pane)
        ctl_layout = QHBoxLayout(controls_bar)
        ctl_layout.setContentsMargins(0, 0, 0, 0)
        ctl_layout.setSpacing(10)

        self.run_btn = QPushButton("▶ Run AI Team", controls_bar)
        self.run_btn.setStyleSheet("font-weight: bold; padding: 6px 16px;")
        self.run_btn.clicked.connect(self.run_team)
        ctl_layout.addWidget(self.run_btn)

        self.cancel_btn = QPushButton("⏹ Cancel", controls_bar)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_team)
        ctl_layout.addWidget(self.cancel_btn)

        self.status_label = QLabel("Idle", controls_bar)
        self.status_label.setStyleSheet("font-weight: 600; font-size: 12px; padding-left: 8px;")
        ctl_layout.addWidget(self.status_label)

        ctl_layout.addStretch(1)

        self.summary_metrics_label = QLabel("", controls_bar)
        self.summary_metrics_label.setStyleSheet("font-size: 11px; font-family: Consolas, monospace;")
        ctl_layout.addWidget(self.summary_metrics_label)

        right_layout.addWidget(controls_bar)

        # Revision Feedback / Status Callout Banner (hidden by default)
        self.revision_banner = QFrame(self.right_pane)
        self.revision_banner.setFrameShape(QFrame.Shape.StyledPanel)
        rev_b_layout = QVBoxLayout(self.revision_banner)
        rev_b_layout.setContentsMargins(10, 8, 10, 8)
        rev_b_layout.setSpacing(4)
        self.revision_banner_title = QLabel("Critic Revision Loop", self.revision_banner)
        self.revision_banner_title.setStyleSheet("font-weight: bold; font-size: 12px; color: #F59E0B;")
        rev_b_layout.addWidget(self.revision_banner_title)
        self.revision_banner_text = QLabel("", self.revision_banner)
        self.revision_banner_text.setWordWrap(True)
        rev_b_layout.addWidget(self.revision_banner_text)
        self.revision_banner.setVisible(False)
        right_layout.addWidget(self.revision_banner)

        # Agent Execution Timeline (Scroll area of AgentStepCards)
        self.timeline_scroll = QScrollArea(self.right_pane)
        self.timeline_scroll.setWidgetResizable(True)
        self.timeline_scroll.setMinimumHeight(180)
        self.timeline_container = QWidget(self.timeline_scroll)
        self.timeline_layout = QVBoxLayout(self.timeline_container)
        self.timeline_layout.setContentsMargins(4, 4, 4, 4)
        self.timeline_layout.setSpacing(8)
        self.timeline_layout.addStretch(1)
        self.timeline_scroll.setWidget(self.timeline_container)
        right_layout.addWidget(self.timeline_scroll, 2)

        # Final Deliverable Panel
        self.deliverable_panel = FinalDeliverablePanel(
            dark=self.dark,
            parent=self.right_pane,
        )
        self.deliverable_panel.setMinimumHeight(200)
        right_layout.addWidget(self.deliverable_panel, 3)

        self.main_splitter.addWidget(self.right_pane)

        # Set sensible default splitter proportion (40% left, 60% right)
        self.main_splitter.setSizes([380, 620])
        main_layout.addWidget(self.main_splitter, 1)

    # -----------------------------------------------------------------
    # Workflow Execution & Worker Lifecycle
    # -----------------------------------------------------------------

    def run_team(self) -> None:
        """Start team execution with preflight validation and safe worker lifecycle."""
        if self._is_running:
            logger.warning("run_team called while a run is already in progress.")
            return

        task_prompt = self.task_input.toPlainText().strip()
        if not task_prompt:
            self.status_label.setText("Error: Task objective cannot be empty.")
            self.status_label.setStyleSheet("color: #EF4444; font-weight: bold;")
            return

        # 1. Configuration Validation
        config_err = self.config_panel.validate_config()
        if config_err:
            self.status_label.setText(f"Config Error: {config_err}")
            self.status_label.setStyleSheet("color: #EF4444; font-weight: bold;")
            self.deliverable_panel.set_error(f"Configuration validation failed: {config_err}")
            return

        team_config = self.config_panel.get_team_config()

        # 2. Preflight Provider Readiness Check
        factory = TeamFactory(self.provider_manager)
        try:
            factory.validate_team_readiness(team_config)
        except Exception as exc:
            err_msg = str(exc)
            logger.error("Preflight team readiness check failed: %s", err_msg)
            self.status_label.setText(f"Preflight Failed: {err_msg}")
            self.status_label.setStyleSheet("color: #EF4444; font-weight: bold;")
            self.deliverable_panel.set_error(f"Provider readiness validation failed: {err_msg}")
            return

        # 3. Setup Session / Run ID
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        self._current_run_id = run_id
        self._is_running = True

        # Update UI state
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.config_panel.setEnabled(False)
        self.status_label.setText(f"Running ({run_id})...")
        self.status_label.setStyleSheet("color: #3B82F6; font-weight: bold;")
        self.summary_metrics_label.setText("")
        self.revision_banner.setVisible(False)
        self.deliverable_panel.set_loading("AI Team is executing specialized agent turns...")

        # Initialize Agent Step Cards for all enabled roles
        self._initialize_step_cards(team_config)

        # 4. Instantiate Orchestrator and Worker
        try:
            orchestrator = TeamOrchestrator(
                config=team_config,
                factory=factory,
                provider_manager=self.provider_manager,
            )
            worker = TeamExecutionWorker(
                orchestrator=orchestrator,
                task=task_prompt,
                parent=self,
            )
            self._active_worker = worker
            safely_anchor_worker(worker)

            # Connect lifecycle signals with session token guard
            worker.step_started.connect(
                lambda snap, rid=run_id: self._on_step_started(snap, rid)
            )
            worker.step_completed.connect(
                lambda snap, rid=run_id: self._on_step_completed(snap, rid)
            )
            worker.revision_requested.connect(
                lambda target, reason, fb, loop, rid=run_id: self._on_revision_requested(
                    target, reason, fb, loop, rid
                )
            )
            worker.team_completed.connect(
                lambda result, rid=run_id: self._on_team_completed(result, rid)
            )
            worker.execution_error.connect(
                lambda err, rid=run_id: self._on_execution_error(err, rid)
            )
            worker.cancelled.connect(
                lambda rid=run_id: self._on_worker_cancelled(rid)
            )

            # Start execution off GUI thread
            worker.start()
        except Exception as exc:
            self._is_running = False
            self.run_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)
            self.config_panel.setEnabled(True)
            self.status_label.setText(f"Launch Error: {exc}")
            self.status_label.setStyleSheet("color: #EF4444; font-weight: bold;")
            self.deliverable_panel.set_error(f"Failed to launch team execution worker: {exc}")

    def cancel_team(self) -> None:
        """Cooperatively request cancellation without disconnecting completion signals."""
        if not self._active_worker:
            return

        self.cancel_btn.setEnabled(False)
        self.status_label.setText("Canceling...")
        self.status_label.setStyleSheet("color: #F59E0B; font-weight: bold;")

        try:
            self._active_worker.cancel()
        except Exception as exc:
            logger.warning("Error signaling cancellation to worker: %s", exc)

    # -----------------------------------------------------------------
    # Signal Handlers (Session Token Protected)
    # -----------------------------------------------------------------

    def _on_step_started(self, snapshot: TeamExecutionSnapshot, run_id: str) -> None:
        """Handle agent step commencement."""
        if run_id != self._current_run_id:
            return  # Stale event

        card = self._step_cards.get(snapshot.role)
        if card:
            card.update_from_snapshot(snapshot)

        role_name = snapshot.role.value.capitalize()
        self.status_label.setText(f"Active: {role_name} Agent running...")

    def _on_step_completed(self, snapshot: TeamExecutionSnapshot, run_id: str) -> None:
        """Handle agent step completion telemetry."""
        if run_id != self._current_run_id:
            return  # Stale event

        card = self._step_cards.get(snapshot.role)
        if card:
            card.update_from_snapshot(snapshot)

    def _on_revision_requested(
        self,
        target_step_id: str,
        reason: str,
        feedback: str,
        loop_count: int,
        run_id: str,
    ) -> None:
        """Display revision callout banner when Critic requests revisions."""
        if run_id != self._current_run_id:
            return  # Stale event

        self.revision_banner_title.setText(
            f"Critic Revision Loop #{loop_count} (Target: {target_step_id})"
        )
        self.revision_banner_text.setText(f"Reason: {reason}\nFeedback: {feedback}")
        self.revision_banner.setVisible(True)

    def _on_team_completed(self, result: TeamExecutionResult, run_id: str) -> None:
        """Terminal result handler updating deliverable, step cards, and history."""
        if run_id != self._current_run_id:
            return  # Stale event

        self._is_running = False
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.config_panel.setEnabled(True)

        # 1. Update Deliverable Panel
        self.deliverable_panel.update_from_result(result)

        # 2. Update Step Cards with full outputs and records
        self._update_cards_from_result(result)

        # 3. Status label & metrics summary
        st_text = result.status.value.capitalize()
        if result.is_success:
            self.status_label.setText(f"Completed ({st_text})")
            self.status_label.setStyleSheet("color: #10B981; font-weight: bold;")
        elif result.status == WorkflowStatus.CANCELLED:
            self.status_label.setText("Cancelled")
            self.status_label.setStyleSheet("color: #F59E0B; font-weight: bold;")
        else:
            self.status_label.setText(f"Failed ({st_text})")
            self.status_label.setStyleSheet("color: #EF4444; font-weight: bold;")

        tok_str = f"{result.total_tokens:,} tok" if result.total_tokens is not None else "-- tok"
        self.summary_metrics_label.setText(
            f"⏱ {result.total_duration_sec:.2f}s | 🪙 {tok_str} | 🔄 {result.revision_count} revs"
        )

        # 4. Record to in-memory Run History
        self._record_run_history(result)

    def _on_execution_error(self, error_message: str, run_id: str) -> None:
        """Handle fatal unhandled worker error."""
        if run_id != self._current_run_id:
            return

        self._is_running = False
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.config_panel.setEnabled(True)

        self.status_label.setText(f"Execution Error")
        self.status_label.setStyleSheet("color: #EF4444; font-weight: bold;")
        self.deliverable_panel.set_error(error_message)

    def _on_worker_cancelled(self, run_id: str) -> None:
        """Worker cooperative cancellation signal received."""
        if run_id != self._current_run_id:
            return
        self.status_label.setText("Cancelled")
        self.status_label.setStyleSheet("color: #F59E0B; font-weight: bold;")

    # -----------------------------------------------------------------
    # Step Card Timeline Setup & Updates
    # -----------------------------------------------------------------

    def _initialize_step_cards(self, team_config: TeamConfig) -> None:
        """Clear timeline and construct fresh step cards for enabled team members."""
        # Clear existing cards
        while self.timeline_layout.count() > 1:
            item = self.timeline_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._step_cards.clear()

        # Build cards in logical execution order
        ordered_roles = [
            AgentRole.PLANNER,
            AgentRole.RESEARCHER,
            AgentRole.CREATOR,
            AgentRole.CRITIC,
            AgentRole.FINALIZER,
        ]

        order = 1
        for role in ordered_roles:
            member = team_config.members.get(role)
            if member and member.enabled:
                card = AgentStepCard(
                    step_id=member.agent_id or f"step_{role.value}",
                    role=role,
                    order=order,
                    provider_id=member.provider_id,
                    model_id=member.model_id,
                    dark=self.dark,
                    parent=self.timeline_container,
                )
                self._step_cards[role] = card
                self.timeline_layout.insertWidget(order - 1, card)
                order += 1

    def _update_cards_from_result(self, result: TeamExecutionResult) -> None:
        """Distribute execution records and latest outputs to matching step cards."""
        for role, card in self._step_cards.items():
            matching_records = [
                rec for rec in result.step_records if rec.agent_role == role
            ]
            latest_out = result.step_outputs.get(card.step_id)

            card.set_records(matching_records, latest_output=latest_out)

    # -----------------------------------------------------------------
    # In-Memory Run History (Latest 20 runs)
    # -----------------------------------------------------------------

    def _record_run_history(self, result: TeamExecutionResult) -> None:
        """Add completed run to in-memory session history and update list widget."""
        start_time_str = datetime.now().strftime("%H:%M:%S")
        task_snippet = self.task_input.toPlainText().strip()
        attribution = self.deliverable_panel.attribution or "AI Team"

        history_item = TeamRunHistoryItem(
            run_id=self._current_run_id,
            task=task_snippet,
            start_time=start_time_str,
            status=result.status.value.upper(),
            duration_sec=result.total_duration_sec,
            final_deliverable=result.final_deliverable,
            step_outputs=dict(result.step_outputs),
            step_records=list(result.step_records),
            revision_count=result.revision_count,
            total_tokens=result.total_tokens,
            error_summary=result.error_summary,
            attribution=attribution,
        )

        # Prepend to list and enforce max 20 ceiling
        self._run_history.insert(0, history_item)
        self._run_history = self._run_history[:20]

        self._refresh_history_list()

    def _refresh_history_list(self) -> None:
        """Populate the history list widget."""
        self.history_list.blockSignals(True)
        self.history_list.clear()

        for item in self._run_history:
            tok_snippet = f"{item.total_tokens:,} tok" if item.total_tokens is not None else "--"
            label = f"[{item.status}] {item.start_time} - {item.duration_sec:.1f}s ({tok_snippet})\n{item.task[:40]}..."
            list_item = QListWidgetItem(label)
            list_item.setData(Qt.ItemDataRole.UserRole, item.run_id)
            self.history_list.addItem(list_item)

        self.history_list.blockSignals(False)

    def _on_history_item_clicked(self, item: QListWidgetItem) -> None:
        """Display previous run deliverable and outputs without mutating active inputs."""
        run_id = item.data(Qt.ItemDataRole.UserRole)
        matched = next((h for h in self._run_history if h.run_id == run_id), None)
        if not matched:
            return

        # Render deliverable
        if matched.final_deliverable:
            self.deliverable_panel.set_deliverable(
                content=matched.final_deliverable,
                attribution=matched.attribution,
                duration=matched.duration_sec,
                tokens=matched.total_tokens,
            )
        elif matched.error_summary:
            self.deliverable_panel.set_error(matched.error_summary)
        else:
            self.deliverable_panel.set_empty()

        # Update step cards if records exist
        for role, card in self._step_cards.items():
            matching_recs = [r for r in matched.step_records if r.agent_role == role]
            card.set_records(matching_recs, latest_output=matched.step_outputs.get(card.step_id))

        self.summary_metrics_label.setText(
            f"(Inspecting Run {matched.run_id}) ⏱ {matched.duration_sec:.2f}s | 🔄 {matched.revision_count} revs"
        )

    # -----------------------------------------------------------------
    # Theme Support & Teardown
    # -----------------------------------------------------------------

    def set_dark_theme(self, dark: bool) -> None:
        """Switch dark/light theme dynamically across all panels."""
        if self.dark == dark:
            return
        self.dark = dark
        self._apply_theme()
        self.config_panel.set_dark_theme(dark)
        self.deliverable_panel.set_dark_theme(dark)
        for card in self._step_cards.values():
            card.set_dark_theme(dark)

    def _apply_theme(self) -> None:
        if self.dark:
            bg_color = "#0D1117"
            panel_bg = "#161B22"
            border_color = "#30363D"
            text_pri = "#F0F6FC"
            text_sec = "#8B949E"
            btn_run_bg = "#238636"
            btn_run_hover = "#2EA043"
            btn_run_text = "#FFFFFF"
            btn_cancel_bg = "#851A1D"
            btn_cancel_hover = "#A42327"
            btn_cancel_text = "#FCA5A5"
            input_bg = "#0D1117"
            list_bg = "#0D1117"
            rev_banner_bg = "#271B0B"
            rev_banner_border = "#92400E"
        else:
            bg_color = "#F8FAFC"
            panel_bg = "#FFFFFF"
            border_color = "#E2E8F0"
            text_pri = "#0F172A"
            text_sec = "#475569"
            btn_run_bg = "#16A34A"
            btn_run_hover = "#15803D"
            btn_run_text = "#FFFFFF"
            btn_cancel_bg = "#FEE2E2"
            btn_cancel_hover = "#FCA5A5"
            btn_cancel_text = "#B91C1C"
            input_bg = "#FFFFFF"
            list_bg = "#FFFFFF"
            rev_banner_bg = "#FEF3C7"
            rev_banner_border = "#F59E0B"

        self.title_label.setStyleSheet(f"color: {text_pri}; font-size: 18px; font-weight: bold;")
        self.subtitle_label.setStyleSheet(f"color: {text_sec}; font-size: 12px;")

        self.task_input.setStyleSheet(
            f"QPlainTextEdit {{"
            f"  background-color: {input_bg};"
            f"  color: {text_pri};"
            f"  border: 1px solid {border_color};"
            f"  border-radius: 6px;"
            f"  padding: 6px;"
            f"  font-size: 12px;"
            f"}}"
        )

        self.run_btn.setStyleSheet(
            f"QPushButton {{"
            f"  background-color: {btn_run_bg};"
            f"  color: {btn_run_text};"
            f"  border: none;"
            f"  border-radius: 6px;"
            f"  padding: 6px 16px;"
            f"  font-weight: bold;"
            f"}}"
            f"QPushButton:hover {{"
            f"  background-color: {btn_run_hover};"
            f"}}"
            f"QPushButton:disabled {{"
            f"  opacity: 0.5;"
            f"  background-color: #64748B;"
            f"}}"
        )

        self.cancel_btn.setStyleSheet(
            f"QPushButton {{"
            f"  background-color: {btn_cancel_bg};"
            f"  color: {btn_cancel_text};"
            f"  border: 1px solid {border_color};"
            f"  border-radius: 6px;"
            f"  padding: 6px 14px;"
            f"  font-weight: 600;"
            f"}}"
            f"QPushButton:hover {{"
            f"  background-color: {btn_cancel_hover};"
            f"}}"
            f"QPushButton:disabled {{"
            f"  opacity: 0.5;"
            f"}}"
        )

        self.history_group.setStyleSheet(
            f"QFrame {{"
            f"  background-color: {panel_bg};"
            f"  border: 1px solid {border_color};"
            f"  border-radius: 6px;"
            f"}}"
        )

        self.history_list.setStyleSheet(
            f"QListWidget {{"
            f"  background-color: {list_bg};"
            f"  color: {text_pri};"
            f"  border: 1px solid {border_color};"
            f"  border-radius: 4px;"
            f"  font-size: 11px;"
            f"}}"
            f"QListWidget::item {{"
            f"  padding: 4px;"
            f"  border-bottom: 1px solid {border_color};"
            f"}}"
            f"QListWidget::item:selected {{"
            f"  background-color: #2563EB;"
            f"  color: #FFFFFF;"
            f"}}"
        )

        self.revision_banner.setStyleSheet(
            f"QFrame {{"
            f"  background-color: {rev_banner_bg};"
            f"  border: 1px solid {rev_banner_border};"
            f"  border-radius: 6px;"
            f"}}"
        )

    def closeEvent(self, event) -> None:
        """Safely anchor running worker on view close to avoid native destruction crashes."""
        if self._active_worker and self._active_worker.isRunning():
            safely_anchor_worker(self._active_worker)
            self._active_worker.cancel()
        super().closeEvent(event)
