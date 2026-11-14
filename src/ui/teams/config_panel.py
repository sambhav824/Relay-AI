"""Relay AI - AI Teams Configuration Panel.

Allows configuring specialized agents (Planner, Researcher, Creator, Critic, Finalizer),
provider and model selection with asynchronous discovery, temperature, timeouts,
workflow ceilings, and preset management.
"""

import logging
from typing import Any, Dict, List, Optional, Set

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.agents.models import AgentRole
from src.providers.manager import ProviderManager, get_default_provider_manager
from src.providers.models import ModelInfo
from src.teams.factory import create_default_team_config
from src.teams.models import TeamConfig, TeamMemberConfig
from src.ui.teams.worker import ModelDiscoveryWorker, safely_anchor_worker

logger = logging.getLogger("RelayAI.TeamConfigPanel")

# Standard agent roles supported in AI Teams pipeline
CONFIGURABLE_ROLES: List[AgentRole] = [
    AgentRole.PLANNER,
    AgentRole.RESEARCHER,
    AgentRole.CREATOR,
    AgentRole.CRITIC,
    AgentRole.FINALIZER,
]


class AgentRowWidget(QWidget):
    """UI control row for configuring an individual agent role."""

    provider_changed = Signal(str)  # Emits new provider_id when user switches provider

    def __init__(
        self,
        role: AgentRole,
        is_creator: bool = False,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.role = role
        self.is_creator = is_creator

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # 1. Enable Checkbox
        role_title = self.role.value.capitalize()
        self.enable_chk = QCheckBox(role_title, self)
        self.enable_chk.setMinimumWidth(110)
        if self.is_creator:
            self.enable_chk.setChecked(True)
            self.enable_chk.setEnabled(False)  # Creator is strictly mandatory and locked
            self.enable_chk.setToolTip("Creator is the mandatory core generation agent and cannot be disabled.")
        else:
            self.enable_chk.setChecked(True)
            self.enable_chk.toggled.connect(self._on_toggled)

        layout.addWidget(self.enable_chk)

        # 2. Provider ComboBox
        self.provider_combo = QComboBox(self)
        self.provider_combo.setMinimumWidth(100)
        self.provider_combo.currentTextChanged.connect(self._on_provider_changed)
        layout.addWidget(self.provider_combo)

        # 3. Model ComboBox
        self.model_combo = QComboBox(self)
        self.model_combo.setMinimumWidth(160)
        self.model_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.model_combo)

        # 4. Temperature SpinBox
        temp_lbl = QLabel("Temp:", self)
        layout.addWidget(temp_lbl)
        self.temp_spin = QDoubleSpinBox(self)
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setSingleStep(0.05)
        self.temp_spin.setDecimals(2)
        self.temp_spin.setValue(0.7)
        self.temp_spin.setFixedWidth(65)
        layout.addWidget(self.temp_spin)

        # 5. Timeout SpinBox
        timeout_lbl = QLabel("Timeout:", self)
        layout.addWidget(timeout_lbl)
        self.timeout_spin = QSpinBox(self)
        self.timeout_spin.setRange(5, 600)
        self.timeout_spin.setSingleStep(5)
        self.timeout_spin.setValue(60)
        self.timeout_spin.setSuffix("s")
        self.timeout_spin.setFixedWidth(70)
        layout.addWidget(self.timeout_spin)

        # 6. Max Retries SpinBox
        retries_lbl = QLabel("Retries:", self)
        layout.addWidget(retries_lbl)
        self.retries_spin = QSpinBox(self)
        self.retries_spin.setRange(0, 5)
        self.retries_spin.setValue(1)
        self.retries_spin.setFixedWidth(50)
        layout.addWidget(self.retries_spin)

    def _on_toggled(self, enabled: bool) -> None:
        """Dim and disable inputs when agent is disabled."""
        self.provider_combo.setEnabled(enabled)
        self.model_combo.setEnabled(enabled)
        self.temp_spin.setEnabled(enabled)
        self.timeout_spin.setEnabled(enabled)
        self.retries_spin.setEnabled(enabled)

    def _on_provider_changed(self, text: str) -> None:
        if text:
            self.provider_changed.emit(text.strip().lower())

    def is_agent_enabled(self) -> bool:
        if self.is_creator:
            return True
        return self.enable_chk.isChecked()

    def set_agent_enabled(self, enabled: bool) -> None:
        if not self.is_creator:
            self.enable_chk.setChecked(enabled)
            self._on_toggled(enabled)


class TeamConfigPanel(QWidget):
    """Panel allowing interactive configuration of AI Team agents and workflow parameters."""

    config_changed = Signal()  # Emitted when configuration settings change

    def __init__(
        self,
        provider_manager: Optional[ProviderManager] = None,
        dark: bool = True,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.provider_manager = provider_manager or get_default_provider_manager()
        self.dark = dark

        # In-memory model discovery cache: provider_id -> List[ModelInfo]
        self._model_cache: Dict[str, List[ModelInfo]] = {}
        # Active discovery workers to avoid duplicates and track background operations
        self._discovery_workers: Dict[str, ModelDiscoveryWorker] = {}

        self.agent_rows: Dict[AgentRole, AgentRowWidget] = {}

        self._init_ui()
        self._populate_providers()
        self.reset_to_defaults()
        self._apply_theme()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(12)

        # -------------------------------------------------------------
        # 1. Agents Configuration Section
        # -------------------------------------------------------------
        self.agents_group = QGroupBox("Specialized Agent Team Members", self)
        agents_layout = QVBoxLayout(self.agents_group)
        agents_layout.setContentsMargins(10, 12, 10, 10)
        agents_layout.setSpacing(8)

        # Column headers guide
        guide_widget = QWidget(self.agents_group)
        guide_layout = QHBoxLayout(guide_widget)
        guide_layout.setContentsMargins(8, 0, 8, 2)
        guide_layout.setSpacing(8)

        lbl_role = QLabel("Role", guide_widget)
        lbl_role.setMinimumWidth(110)
        lbl_role.setStyleSheet("font-weight: bold; font-size: 11px;")
        guide_layout.addWidget(lbl_role)

        lbl_prov = QLabel("Provider", guide_widget)
        lbl_prov.setMinimumWidth(100)
        lbl_prov.setStyleSheet("font-weight: bold; font-size: 11px;")
        guide_layout.addWidget(lbl_prov)

        lbl_model = QLabel("Model", guide_widget)
        lbl_model.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        lbl_model.setStyleSheet("font-weight: bold; font-size: 11px;")
        guide_layout.addWidget(lbl_model)

        lbl_params = QLabel("Parameters (Temp / Timeout / Retries)", guide_widget)
        lbl_params.setFixedWidth(240)
        lbl_params.setStyleSheet("font-weight: bold; font-size: 11px;")
        guide_layout.addWidget(lbl_params)

        agents_layout.addWidget(guide_widget)

        # Create row widget for each role
        for role in CONFIGURABLE_ROLES:
            is_creator = (role == AgentRole.CREATOR)
            row_widget = AgentRowWidget(role, is_creator=is_creator, parent=self.agents_group)
            row_widget.provider_changed.connect(
                lambda pid, r=role: self._on_role_provider_changed(r, pid)
            )
            # Connect change notifications
            row_widget.enable_chk.toggled.connect(self.config_changed)
            row_widget.provider_combo.currentIndexChanged.connect(self.config_changed)
            row_widget.model_combo.currentIndexChanged.connect(self.config_changed)
            row_widget.temp_spin.valueChanged.connect(self.config_changed)
            row_widget.timeout_spin.valueChanged.connect(self.config_changed)
            row_widget.retries_spin.valueChanged.connect(self.config_changed)

            self.agent_rows[role] = row_widget
            agents_layout.addWidget(row_widget)

        main_layout.addWidget(self.agents_group)

        # -------------------------------------------------------------
        # 2. Workflow Parameters Section
        # -------------------------------------------------------------
        self.workflow_group = QGroupBox("Workflow Limits & Revision Policy", self)
        wf_layout = QGridLayout(self.workflow_group)
        wf_layout.setContentsMargins(12, 12, 12, 10)
        wf_layout.setHorizontalSpacing(16)
        wf_layout.setVerticalSpacing(8)

        # Global Timeout
        wf_layout.addWidget(QLabel("Global Timeout:", self.workflow_group), 0, 0)
        self.global_timeout_spin = QSpinBox(self.workflow_group)
        self.global_timeout_spin.setRange(10, 3600)
        self.global_timeout_spin.setValue(300)
        self.global_timeout_spin.setSuffix("s")
        self.global_timeout_spin.setFixedWidth(100)
        self.global_timeout_spin.valueChanged.connect(self.config_changed)
        wf_layout.addWidget(self.global_timeout_spin, 0, 1)

        # Max Revision Loops
        wf_layout.addWidget(QLabel("Max Revision Loops:", self.workflow_group), 0, 2)
        self.max_revision_spin = QSpinBox(self.workflow_group)
        self.max_revision_spin.setRange(0, 10)
        self.max_revision_spin.setValue(2)
        self.max_revision_spin.setFixedWidth(80)
        self.max_revision_spin.valueChanged.connect(self.config_changed)
        wf_layout.addWidget(self.max_revision_spin, 0, 3)

        # Max Workflow Steps
        wf_layout.addWidget(QLabel("Max Steps Ceiling:", self.workflow_group), 1, 0)
        self.max_steps_spin = QSpinBox(self.workflow_group)
        self.max_steps_spin.setRange(1, 50)
        self.max_steps_spin.setValue(15)
        self.max_steps_spin.setFixedWidth(100)
        self.max_steps_spin.valueChanged.connect(self.config_changed)
        wf_layout.addWidget(self.max_steps_spin, 1, 1)

        # Token Budget (Optional)
        wf_layout.addWidget(QLabel("Token Budget (0 = None):", self.workflow_group), 1, 2)
        self.token_budget_spin = QSpinBox(self.workflow_group)
        self.token_budget_spin.setRange(0, 1000000)
        self.token_budget_spin.setSingleStep(5000)
        self.token_budget_spin.setValue(0)
        self.token_budget_spin.setSpecialValueText("Unlimited")
        self.token_budget_spin.setFixedWidth(120)
        self.token_budget_spin.valueChanged.connect(self.config_changed)
        wf_layout.addWidget(self.token_budget_spin, 1, 3)

        main_layout.addWidget(self.workflow_group)

        # -------------------------------------------------------------
        # 3. Actions Toolbar
        # -------------------------------------------------------------
        actions_layout = QHBoxLayout()
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(10)

        self.reset_btn = QPushButton("Reset to Default Configuration", self)
        self.reset_btn.clicked.connect(self.reset_to_defaults)
        actions_layout.addWidget(self.reset_btn)

        actions_layout.addStretch(1)

        self.validation_status_lbl = QLabel("", self)
        self.validation_status_lbl.setStyleSheet("font-size: 11px;")
        actions_layout.addWidget(self.validation_status_lbl)

        main_layout.addLayout(actions_layout)

    # -----------------------------------------------------------------
    # Provider & Model Discovery Handling (Asynchronous)
    # -----------------------------------------------------------------

    def _populate_providers(self) -> None:
        """Populate provider dropdowns for each agent row with registered providers."""
        providers = self.provider_manager.list_providers()
        prov_ids = [p.provider_id.lower() for p in providers] if providers else ["ollama"]

        for row in self.agent_rows.values():
            row.provider_combo.blockSignals(True)
            row.provider_combo.clear()
            for pid in prov_ids:
                row.provider_combo.addItem(pid)
            row.provider_combo.blockSignals(False)

    def _on_role_provider_changed(self, role: AgentRole, provider_id: str) -> None:
        """Triggered when an individual agent's provider selection changes."""
        self._ensure_models_for_provider(provider_id)
        self._update_models_for_role(role, provider_id)

    def _ensure_models_for_provider(self, provider_id: str) -> None:
        """Ensure models for provider_id are fetched asynchronously off the GUI thread."""
        pid = provider_id.lower()
        if pid in self._model_cache:
            return  # Already cached

        if pid in self._discovery_workers and self._discovery_workers[pid].isRunning():
            return  # Discovery currently active

        # Launch asynchronous discovery worker
        worker = ModelDiscoveryWorker(
            provider_id=pid,
            provider_manager=self.provider_manager,
            parent=self,
        )
        self._discovery_workers[pid] = worker
        safely_anchor_worker(worker)

        worker.models_loaded.connect(self._on_models_loaded)
        worker.discovery_failed.connect(self._on_discovery_failed)
        worker.start()

    def _on_models_loaded(self, provider_id: str, models: List[ModelInfo]) -> None:
        """Slot invoked when background model discovery successfully finishes."""
        pid = provider_id.lower()
        self._model_cache[pid] = list(models)
        self._refresh_model_combos_for_provider(pid)

    def _on_discovery_failed(
        self, provider_id: str, error_msg: str, fallbacks: List[ModelInfo]
    ) -> None:
        """Slot invoked when background model discovery fails; populates fallbacks."""
        pid = provider_id.lower()
        self._model_cache[pid] = list(fallbacks)
        self._refresh_model_combos_for_provider(pid)
        logger.debug("Model discovery fallback applied for %s: %s", pid, error_msg)

    def _refresh_model_combos_for_provider(self, provider_id: str) -> None:
        """Update model comboboxes for all agent rows configured with provider_id."""
        models = self._model_cache.get(provider_id, [])
        model_names = [m.id for m in models if hasattr(m, "id")]

        for row in self.agent_rows.values():
            if row.provider_combo.currentText().strip().lower() == provider_id:
                current_selected = row.model_combo.currentText().strip()
                row.model_combo.blockSignals(True)
                row.model_combo.clear()
                for m_id in model_names:
                    row.model_combo.addItem(m_id)
                # Restore previous model selection or keep it
                if current_selected:
                    idx = row.model_combo.findText(current_selected)
                    if idx >= 0:
                        row.model_combo.setCurrentIndex(idx)
                    else:
                        row.model_combo.addItem(current_selected)
                        row.model_combo.setCurrentIndex(row.model_combo.count() - 1)
                elif row.model_combo.count() > 0:
                    row.model_combo.setCurrentIndex(0)
                row.model_combo.blockSignals(False)

    def _update_models_for_role(self, role: AgentRole, provider_id: str) -> None:
        """Populate model combobox for a single role immediately from cache if available."""
        row = self.agent_rows.get(role)
        if not row:
            return

        models = self._model_cache.get(provider_id, [])
        if models:
            row.model_combo.blockSignals(True)
            row.model_combo.clear()
            for m in models:
                row.model_combo.addItem(m.id)
            row.model_combo.blockSignals(False)

    # -----------------------------------------------------------------
    # Configuration Getters & Setters
    # -----------------------------------------------------------------

    def reset_to_defaults(self) -> None:
        """Reset all agent and workflow settings to standard default presets."""
        default_config = create_default_team_config()
        self.set_from_team_config(default_config)

    def set_from_team_config(self, config: TeamConfig) -> None:
        """Populate UI widgets from an existing TeamConfig object."""
        self.global_timeout_spin.setValue(int(config.timeout_sec))
        self.max_revision_spin.setValue(int(config.max_revision_loops))
        self.max_steps_spin.setValue(int(config.max_steps))
        self.token_budget_spin.setValue(int(config.token_budget) if config.token_budget else 0)

        for role, row in self.agent_rows.items():
            member = config.members.get(role)
            if member:
                row.set_agent_enabled(member.enabled)

                # Set provider
                idx = row.provider_combo.findText(member.provider_id.lower())
                if idx >= 0:
                    row.provider_combo.setCurrentIndex(idx)
                else:
                    row.provider_combo.addItem(member.provider_id.lower())
                    row.provider_combo.setCurrentIndex(row.provider_combo.count() - 1)

                # Fetch/ensure models for provider
                self._ensure_models_for_provider(member.provider_id.lower())

                # Set model
                m_idx = row.model_combo.findText(member.model_id)
                if m_idx >= 0:
                    row.model_combo.setCurrentIndex(m_idx)
                else:
                    row.model_combo.addItem(member.model_id)
                    row.model_combo.setCurrentIndex(row.model_combo.count() - 1)

                # Temperatures & Timeouts
                if member.temperature is not None:
                    row.temp_spin.setValue(float(member.temperature))
                row.timeout_spin.setValue(int(member.timeout_sec))
                row.retries_spin.setValue(int(member.max_retries))
            else:
                row.set_agent_enabled(False)

        self.config_changed.emit()

    def get_team_config(
        self,
        team_id: str = "custom_team",
        name: str = "AI Collaboration Team",
    ) -> TeamConfig:
        """Construct a validated TeamConfig dataclass instance reflecting the UI state."""
        members: Dict[AgentRole, TeamMemberConfig] = {}

        for role, row in self.agent_rows.items():
            enabled = row.is_agent_enabled()
            prov_id = row.provider_combo.currentText().strip().lower()
            m_id = row.model_combo.currentText().strip()
            temp = row.temp_spin.value()
            timeout = float(row.timeout_spin.value())
            retries = row.retries_spin.value()

            members[role] = TeamMemberConfig(
                role=role,
                provider_id=prov_id,
                model_id=m_id,
                agent_id=f"{team_id}_{role.value}",
                enabled=enabled,
                temperature=temp,
                timeout_sec=timeout,
                max_retries=retries,
            )

        budget = self.token_budget_spin.value()
        token_budget = budget if budget > 0 else None

        return TeamConfig(
            team_id=team_id,
            name=name,
            members=members,
            timeout_sec=float(self.global_timeout_spin.value()),
            max_revision_loops=int(self.max_revision_spin.value()),
            max_steps=int(self.max_steps_spin.value()),
            token_budget=token_budget,
        )

    def validate_config(self) -> Optional[str]:
        """Verify the configuration meets all orchestration rules.

        Returns None if valid, or an error description string if invalid.
        """
        try:
            config = self.get_team_config()
            config.validate()

            # Ensure all enabled agents have non-empty model_ids
            for role, member in config.members.items():
                if member.enabled and (not member.model_id or not member.model_id.strip()):
                    return f"Enabled agent '{role.value.capitalize()}' requires a valid model."

            self.validation_status_lbl.setText("✓ Configuration valid")
            self.validation_status_lbl.setStyleSheet("color: #10B981; font-size: 11px;")
            return None
        except Exception as exc:
            msg = str(exc)
            self.validation_status_lbl.setText(f"⚠ {msg}")
            self.validation_status_lbl.setStyleSheet("color: #EF4444; font-size: 11px;")
            return msg

    # -----------------------------------------------------------------
    # Theme Support
    # -----------------------------------------------------------------

    def set_dark_theme(self, dark: bool) -> None:
        """Switch dark/light theme dynamically."""
        if self.dark == dark:
            return
        self.dark = dark
        self._apply_theme()

    def _apply_theme(self) -> None:
        if self.dark:
            gb_border = "#2A3140"
            gb_title = "#93C5FD"
            panel_bg = "#161B22"
            text_pri = "#F0F6FC"
            combo_bg = "#1F2937"
            combo_border = "#374151"
            btn_bg = "#21262D"
            btn_border = "#363B42"
            btn_text = "#C9D1D9"
            btn_hover = "#30363D"
        else:
            gb_border = "#CBD5E1"
            gb_title = "#1E40AF"
            panel_bg = "#FFFFFF"
            text_pri = "#0F172A"
            combo_bg = "#FFFFFF"
            combo_border = "#CBD5E1"
            btn_bg = "#F8FAFC"
            btn_border = "#CBD5E1"
            btn_text = "#334155"
            btn_hover = "#E2E8F0"

        style = f"""
        QGroupBox {{
            font-weight: bold;
            font-size: 12px;
            color: {gb_title};
            border: 1px solid {gb_border};
            border-radius: 6px;
            margin-top: 10px;
            padding-top: 8px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 6px;
        }}
        QComboBox, QSpinBox, QDoubleSpinBox {{
            background-color: {combo_bg};
            color: {text_pri};
            border: 1px solid {combo_border};
            border-radius: 4px;
            padding: 2px 6px;
            font-size: 12px;
        }}
        QPushButton {{
            background-color: {btn_bg};
            color: {btn_text};
            border: 1px solid {btn_border};
            border-radius: 5px;
            padding: 5px 12px;
            font-size: 12px;
            font-weight: 500;
        }}
        QPushButton:hover {{
            background-color: {btn_hover};
        }}
        """
        self.setStyleSheet(style)
