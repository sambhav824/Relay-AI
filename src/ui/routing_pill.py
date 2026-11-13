import html
from typing import Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel

from src.routing.types import RoutingDecision


class RoutingStatusPill(QFrame):
    """Compact interactive status badge displaying current routing decision, strategy, and fallbacks."""

    def __init__(self, dark: bool = True, parent=None):
        super().__init__(parent)
        self.dark = dark
        self.setObjectName("RoutingStatusPill")
        self.setFixedHeight(30)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(6)

        self.icon_label = QLabel("✦")
        self.text_label = QLabel("Auto-Route Ready")
        self.text_label.setStyleSheet("font-size: 12px; font-weight: 600;")

        layout.addWidget(self.icon_label)
        layout.addWidget(self.text_label)

        self.apply_theme(self.dark)
        self.reset()

    def reset(self) -> None:
        """Reset the pill to default ready state."""
        self.icon_label.setText("✦")
        self.text_label.setText("Auto-Route Ready")
        self.setToolTip("Smart Routing evaluates your query and selects the optimal AI model.")
        self._set_pill_style("idle")

    def set_decision(self, decision: RoutingDecision) -> None:
        """Update pill to reflect a live RoutingDecision."""
        if not decision or not decision.primary:
            self.reset()
            return

        task_val = getattr(decision.classification.task_type, "value", str(decision.classification.task_type))
        task_name = str(task_val).replace("_", " ").title()
        conf_pct = int(decision.classification.confidence * 100)
        prov_name = decision.primary.provider_id.title()
        model_name = decision.primary.display_name or decision.primary.model_id
        strat_val = getattr(decision.strategy, "value", str(decision.strategy))
        strategy_name = str(strat_val).replace("_", " ").title()

        self.icon_label.setText("✦")
        self.text_label.setText(f"{task_name} ({conf_pct}%) → {model_name} ({prov_name})")

        # Build comprehensive explanatory tooltip
        tooltip_parts = [
            f"<b>Detected Task:</b> {task_name} (Confidence: {conf_pct}%)",
            f"<b>Strategy:</b> {strategy_name}",
            f"<b>Primary Model:</b> {model_name} [{prov_name}]",
            f"<b>Composite Score:</b> {decision.primary.composite_score:.3f} (Alignment: {decision.primary.task_alignment_score:.1f}/10)",
        ]

        if decision.explanation:
            if decision.explanation.pricing_summary:
                tooltip_parts.append(f"<b>Pricing:</b> {decision.explanation.pricing_summary}")
            if decision.explanation.fallback_candidates:
                tooltip_parts.append(f"<b>Fallback Queue:</b> {', '.join(decision.explanation.fallback_candidates)}")
            if decision.explanation.exclusions_summary:
                tooltip_parts.append(f"<b>Exclusions:</b> {decision.explanation.exclusions_summary}")

        self.setToolTip("<br>".join(tooltip_parts))
        self._set_pill_style("active")

    def set_fallback(
        self,
        prev_provider: str,
        prev_model: str,
        next_provider: str,
        next_model: str,
        fallback_index: int,
    ) -> None:
        """Display live fallback status."""
        self.icon_label.setText("⚠️")
        self.text_label.setText(
            f"Fallback {fallback_index}: {prev_provider.title()} failed → {next_model} ({next_provider.title()})"
        )
        self.setToolTip(
            f"Primary candidate {prev_provider.title()} ({prev_model}) encountered an issue.<br>"
            f"Automatically routed to fallback candidate {next_provider.title()} ({next_model})."
        )
        self._set_pill_style("warning")

    def set_error(self, message: str) -> None:
        """Display routing error."""
        self.icon_label.setText("✕")
        self.text_label.setText(f"Routing Error: {message[:35]}…" if len(message) > 35 else f"Routing Error: {message}")
        self.setToolTip(message)
        self._set_pill_style("error")

    def _set_pill_style(self, state: str) -> None:
        if self.dark:
            if state == "idle":
                bg = "#1B202B"
                border = "#2B3344"
                text_color = "#9AA3B2"
                icon_color = "#818CF8"
            elif state == "active":
                bg = "#1C2438"
                border = "#3B4868"
                text_color = "#E0E7FF"
                icon_color = "#6658E8"
            elif state == "warning":
                bg = "#2D2214"
                border = "#5C4119"
                text_color = "#FDE68A"
                icon_color = "#F59E0B"
            else:  # error
                bg = "#2D1517"
                border = "#631E21"
                text_color = "#FCA5A5"
                icon_color = "#EF4444"
        else:
            if state == "idle":
                bg = "#F3F4F6"
                border = "#E5E7EB"
                text_color = "#6B7280"
                icon_color = "#4F46E5"
            elif state == "active":
                bg = "#EEF2FF"
                border = "#C7D2FE"
                text_color = "#3730A3"
                icon_color = "#4F46E5"
            elif state == "warning":
                bg = "#FEF3C7"
                border = "#FDE68A"
                text_color = "#92400E"
                icon_color = "#D97706"
            else:  # error
                bg = "#FEE2E2"
                border = "#FECACA"
                text_color = "#991B1B"
                icon_color = "#DC2626"

        self.setStyleSheet(
            f"QFrame#RoutingStatusPill {{ background: {bg}; border: 1px solid {border}; border-radius: 8px; }}"
        )
        self.text_label.setStyleSheet(f"color: {text_color}; font-size: 12px; font-weight: 600;")
        self.icon_label.setStyleSheet(f"color: {icon_color}; font-size: 13px; font-weight: 700;")

    def apply_theme(self, dark: bool) -> None:
        self.dark = dark
        self._set_pill_style("idle")

