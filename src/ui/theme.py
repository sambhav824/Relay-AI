"""Relay AI - Centralized Monochromatic Theme Engine.

Centralizes:
- Raven (#202124)
- Charcoal (#36454F)
- Pure White (#FFFFFF)
- Mid-tone Gray (#757575)
- Strict dark and neutral light theme tokens
- Full application Qt Style Sheets (QSS)
- Typography, spacing, borders, radii, and micro-metrics
- Component style helpers (buttons, cards, badges, inputs, scrollbars)
"""

from dataclasses import dataclass
from typing import Optional
from PySide6.QtWidgets import QApplication, QWidget


# -----------------------------------------------------------------------------
# 1. CORE BRAND TOKENS (Strict Monochromatic)
# -----------------------------------------------------------------------------
RAVEN = "#202124"      # Primary Dark
CHARCOAL = "#36454F"   # Secondary Dark / Elevated surface
WHITE = "#FFFFFF"      # Accent / Pure White
GRAY = "#757575"       # Sub-accent / Mid-tone Gray

# Extended Neutrals
SURFACE_BASE_DARK = "#202124"
SURFACE_CARD_DARK = "#36454F"
SURFACE_CARD_HOVER_DARK = "#45545E"
SURFACE_ELEVATED_DARK = "#282A2D"
SURFACE_SUNKEN_DARK = "#17181A"
BORDER_SUBTLE_DARK = "#303236"
BORDER_STRONG_DARK = "#4D585E"
TEXT_PRIMARY_DARK = "#FFFFFF"
TEXT_SECONDARY_DARK = "#B0B0B0"
TEXT_MUTED_DARK = "#757575"
TEXT_INVERTED_DARK = "#202124"

# Neutral Light Theme Matrix
SURFACE_BASE_LIGHT = "#F8F9FA"
SURFACE_CARD_LIGHT = "#FFFFFF"
SURFACE_CARD_HOVER_LIGHT = "#F1F3F5"
SURFACE_ELEVATED_LIGHT = "#FFFFFF"
SURFACE_SUNKEN_LIGHT = "#EBECEF"
BORDER_SUBTLE_LIGHT = "#E0E3E6"
BORDER_STRONG_LIGHT = "#CBD0D5"
TEXT_PRIMARY_LIGHT = "#202124"
TEXT_SECONDARY_LIGHT = "#5F6368"
TEXT_MUTED_LIGHT = "#757575"
TEXT_INVERTED_LIGHT = "#FFFFFF"


@dataclass(frozen=True)
class ThemeTokens:
    """Design token specification for the active theme."""
    is_dark: bool
    bg_base: str
    bg_sidebar: str
    bg_card: str
    bg_card_hover: str
    bg_elevated: str
    bg_sunken: str
    bg_selected_nav: str
    border_subtle: str
    border_strong: str
    border_focus: str
    text_primary: str
    text_secondary: str
    text_muted: str
    text_inverted: str
    btn_primary_bg: str
    btn_primary_fg: str
    btn_primary_hover: str
    btn_secondary_bg: str
    btn_secondary_fg: str
    btn_secondary_hover: str
    badge_bg: str
    badge_fg: str
    badge_border: str


DARK_TOKENS = ThemeTokens(
    is_dark=True,
    bg_base=SURFACE_BASE_DARK,
    bg_sidebar=SURFACE_BASE_DARK,
    bg_card=SURFACE_CARD_DARK,
    bg_card_hover=SURFACE_CARD_HOVER_DARK,
    bg_elevated=SURFACE_ELEVATED_DARK,
    bg_sunken=SURFACE_SUNKEN_DARK,
    bg_selected_nav=SURFACE_CARD_DARK,
    border_subtle=BORDER_SUBTLE_DARK,
    border_strong=BORDER_STRONG_DARK,
    border_focus=WHITE,
    text_primary=TEXT_PRIMARY_DARK,
    text_secondary=TEXT_SECONDARY_DARK,
    text_muted=TEXT_MUTED_DARK,
    text_inverted=TEXT_INVERTED_DARK,
    btn_primary_bg=WHITE,
    btn_primary_fg=RAVEN,
    btn_primary_hover="#E0E0E0",
    btn_secondary_bg=SURFACE_CARD_DARK,
    btn_secondary_fg=WHITE,
    btn_secondary_hover=SURFACE_CARD_HOVER_DARK,
    badge_bg=SURFACE_CARD_DARK,
    badge_fg=WHITE,
    badge_border=BORDER_STRONG_DARK,
)

LIGHT_TOKENS = ThemeTokens(
    is_dark=False,
    bg_base=SURFACE_BASE_LIGHT,
    bg_sidebar=SURFACE_BASE_LIGHT,
    bg_card=SURFACE_CARD_LIGHT,
    bg_card_hover=SURFACE_CARD_HOVER_LIGHT,
    bg_elevated=SURFACE_ELEVATED_LIGHT,
    bg_sunken=SURFACE_SUNKEN_LIGHT,
    bg_selected_nav="#E2E6EA",
    border_subtle=BORDER_SUBTLE_LIGHT,
    border_strong=BORDER_STRONG_LIGHT,
    border_focus=RAVEN,
    text_primary=TEXT_PRIMARY_LIGHT,
    text_secondary=TEXT_SECONDARY_LIGHT,
    text_muted=TEXT_MUTED_LIGHT,
    text_inverted=TEXT_INVERTED_LIGHT,
    btn_primary_bg=RAVEN,
    btn_primary_fg=WHITE,
    btn_primary_hover="#36454F",
    btn_secondary_bg=SURFACE_CARD_LIGHT,
    btn_secondary_fg=RAVEN,
    btn_secondary_hover="#EAECEF",
    badge_bg="#EBECEF",
    badge_fg=RAVEN,
    badge_border=BORDER_SUBTLE_LIGHT,
)


def get_theme(dark: bool = True) -> ThemeTokens:
    """Return the active ThemeTokens instance."""
    return DARK_TOKENS if dark else LIGHT_TOKENS


# -----------------------------------------------------------------------------
# 2. FULL APPLICATION QT STYLE SHEET (QSS)
# -----------------------------------------------------------------------------
def get_stylesheet(dark: bool = True) -> str:
    """Generate the comprehensive monochromatic QSS for Relay AI."""
    t = get_theme(dark)

    return f"""
/* ==========================================================================
   RELAY AI - MONOCHROMATIC DESIGN SYSTEM QSS
   Mode: {'Dark (Raven & Charcoal)' if dark else 'Light (Neutral)'}
   ========================================================================== */

/* Global Reset */
QMainWindow, QStackedWidget {{
    background-color: {t.bg_base};
}}

QWidget {{
    color: {t.text_primary};
    font-family: "Segoe UI Variable", "Segoe UI", "Inter", -apple-system, sans-serif;
    font-size: 13px;
    selection-background-color: {t.bg_card};
    selection-color: {t.text_primary};
}}

QWidget#CentralWidget {{
    background-color: {t.bg_base};
}}

QDialog {{
    background-color: {t.bg_base};
    border: 1px solid {t.border_strong};
}}

/* Sidebar */
QFrame#Sidebar {{
    background-color: {t.bg_sidebar};
    border-right: 1px solid {t.border_subtle};
    border-radius: 0px;
}}

/* Typography Classes - Strict transparent backgrounds to prevent black boxes behind words */
QLabel {{
    background-color: transparent;
    color: {t.text_primary};
}}

QLabel#Brand {{
    font-size: 20px;
    font-weight: 700;
    color: {t.text_primary};
    letter-spacing: -0.01em;
    background-color: transparent;
}}

QLabel#PageTitle {{
    font-size: 22px;
    font-weight: 700;
    color: {t.text_primary};
    background-color: transparent;
}}

QLabel#PanelTitle {{
    font-size: 15px;
    font-weight: 650;
    color: {t.text_primary};
    background-color: transparent;
}}

QLabel#Muted {{
    color: {t.text_secondary};
    font-size: 12px;
    background-color: transparent;
}}

QLabel#Subtle {{
    color: {t.text_muted};
    font-size: 11px;
    background-color: transparent;
}}

/* Ensure all descendant labels of cards, panels, inputs and sidebars are completely transparent */
QFrame#Panel QLabel,
QFrame#Card QLabel,
QFrame#Sidebar QLabel,
QFrame#InputContainer QLabel,
QFrame#TeamStepperFrame QLabel,
QGroupBox QLabel {{
    background-color: transparent;
}}

/* Controls */
QCheckBox, QRadioButton {{
    background-color: transparent;
    color: {t.text_primary};
}}

/* Sidebar Navigation Buttons */
QPushButton#NavButton {{
    background-color: transparent;
    color: {t.text_secondary};
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 9px 14px;
    text-align: left;
    font-size: 13px;
    font-weight: 500;
}}

QPushButton#NavButton:hover {{
    background-color: {t.bg_elevated};
    color: {t.text_primary};
    border-color: {t.border_subtle};
}}

QPushButton#NavButton:checked, QPushButton#NavButton[selected="true"] {{
    background-color: {t.bg_selected_nav};
    color: {t.text_primary};
    font-weight: 600;
    border-left: 3px solid {t.border_focus};
}}

/* Standard Push Buttons */
QPushButton {{
    background-color: {t.btn_secondary_bg};
    color: {t.btn_secondary_fg};
    border: 1px solid {t.border_strong};
    border-radius: 6px;
    padding: 8px 14px;
    font-weight: 500;
    font-size: 12px;
}}

QPushButton:hover {{
    background-color: {t.btn_secondary_hover};
    border-color: {t.border_focus};
}}

QPushButton:pressed {{
    background-color: {t.bg_elevated};
}}

QPushButton:disabled {{
    background-color: {t.bg_base};
    color: {t.text_muted};
    border-color: {t.border_subtle};
}}

/* Primary High-Contrast Button */
QPushButton#Primary {{
    background-color: {t.btn_primary_bg};
    color: {t.btn_primary_fg};
    border: 1px solid {t.btn_primary_bg};
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 600;
    font-size: 13px;
    text-align: center;
}}

QPushButton#Primary:hover {{
    background-color: {t.btn_primary_hover};
    border-color: {t.btn_primary_hover};
}}

QPushButton#Primary:disabled {{
    background-color: {t.bg_card};
    color: {t.text_muted};
    border-color: {t.border_subtle};
}}

/* Secondary Action Button */
QPushButton#Secondary {{
    background-color: {t.btn_secondary_bg};
    color: {t.btn_secondary_fg};
    border: 1px solid {t.border_strong};
    border-radius: 6px;
    padding: 7px 14px;
    text-align: center;
    font-weight: 500;
}}

QPushButton#Secondary:hover {{
    background-color: {t.btn_secondary_hover};
    border-color: {t.border_focus};
}}

/* Panels and Cards */
QFrame#Panel {{
    background-color: {t.bg_card};
    border: 1px solid {t.border_strong};
    border-radius: 8px;
}}

QFrame#Card {{
    background-color: {t.bg_card};
    border: 1px solid {t.border_strong};
    border-radius: 8px;
    padding: 12px;
}}

QFrame#Card:hover {{
    background-color: {t.bg_card_hover};
    border-color: {t.border_focus};
}}

QFrame#InputContainer {{
    background-color: {t.bg_sunken};
    border: 1px solid {t.border_strong};
    border-radius: 22px;
    padding: 3px 8px;
}}

QFrame#InputContainer:focus-within {{
    border-color: {t.border_focus};
}}

QLineEdit#PromptInput {{
    background-color: transparent;
    border: none;
    padding: 6px 8px;
    font-size: 13px;
    color: {t.text_primary};
}}

QLineEdit#PromptInput:focus {{
    border: none;
    background-color: transparent;
}}

/* Chat Input Capsule Buttons (Reference Matching) */
QPushButton#InputAddBtn {{
    background-color: transparent;
    color: {t.text_primary};
    border: 1px solid {t.border_subtle};
    border-radius: 15px;
    font-size: 16px;
    font-weight: 700;
}}

QPushButton#InputAddBtn:hover {{
    background-color: {t.bg_card_hover};
    border-color: {t.border_focus};
}}

QPushButton#ThinkBtn {{
    background-color: transparent;
    color: {t.text_secondary};
    border: 1px solid {t.border_subtle};
    border-radius: 14px;
    padding: 4px 10px;
    font-size: 12px;
    font-weight: 600;
}}

QPushButton#ThinkBtn:hover {{
    background-color: {t.bg_card_hover};
    color: {t.text_primary};
}}

QPushButton#ThinkBtn[checked="true"] {{
    background-color: {t.bg_card};
    color: {t.text_primary};
    border: 1px solid {t.border_focus};
}}

QPushButton#MicBtn {{
    background-color: transparent;
    color: {t.text_primary};
    border: 1px solid {t.border_subtle};
    border-radius: 17px;
    padding: 0px;
}}

QPushButton#MicBtn:hover {{
    background-color: {t.bg_card_hover};
    border-color: {t.border_focus};
}}

QFrame#AttachmentChip {{
    background-color: {t.bg_card};
    border: 1px solid {t.border_strong};
    border-radius: 6px;
    padding: 2px 6px;
}}

QLabel#DisclaimerText {{
    color: {t.text_muted};
    font-size: 11px;
    background-color: transparent;
}}

/* Form Controls & Inputs */
QLineEdit, QTextEdit, QPlainTextEdit {{
    background-color: {t.bg_sunken};
    color: {t.text_primary};
    border: 1px solid {t.border_strong};
    border-radius: 6px;
    padding: 8px 10px;
    font-size: 13px;
}}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
    border: 1px solid {t.border_focus};
    background-color: {t.bg_sunken};
}}

QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled {{
    background-color: {t.bg_base};
    color: {t.text_muted};
    border-color: {t.border_subtle};
}}

/* Dropdowns / ComboBoxes */
QComboBox {{
    background-color: {t.bg_card};
    color: {t.text_primary};
    border: 1px solid {t.border_strong};
    border-radius: 6px;
    padding: 6px 26px 6px 12px;
    min-height: 22px;
}}

QComboBox:hover {{
    background-color: {t.bg_card_hover};
    border-color: {t.border_focus};
}}

QComboBox:focus {{
    border: 1px solid {t.border_focus};
}}

QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 22px;
    border: none;
}}

QComboBox QAbstractItemView {{
    background-color: {t.bg_base};
    color: {t.text_primary};
    border: 1px solid {t.border_strong};
    selection-background-color: {t.bg_card};
    selection-color: {t.text_primary};
    padding: 4px;
}}

/* Text Viewers (Transcript, Logs, Deliverables) */
QTextBrowser {{
    background-color: {t.bg_sunken};
    color: {t.text_primary};
    border: 1px solid {t.border_subtle};
    border-radius: 8px;
    padding: 12px;
    font-size: 13px;
    line-height: 1.5;
}}

/* Scrollbars */
QScrollBar:vertical {{
    background-color: {t.bg_base};
    width: 8px;
    margin: 2px 0px;
    border-radius: 4px;
}}

QScrollBar::handle:vertical {{
    background-color: {t.border_strong};
    min-height: 24px;
    border-radius: 4px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {t.text_muted};
}}

QScrollBar::handle:vertical:pressed {{
    background-color: {t.border_focus};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

QScrollBar:horizontal {{
    background-color: {t.bg_base};
    height: 8px;
    margin: 0px 2px;
    border-radius: 4px;
}}

QScrollBar::handle:horizontal {{
    background-color: {t.border_strong};
    min-width: 24px;
    border-radius: 4px;
}}

/* Spinboxes & Form Inputs */
QSpinBox, QDoubleSpinBox {{
    background-color: {t.bg_sunken};
    color: {t.text_primary};
    border: 1px solid {t.border_strong};
    border-radius: 6px;
    padding: 4px 8px;
}}

QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 1px solid {t.border_focus};
}}

/* Checkboxes */
QCheckBox {{
    color: {t.text_primary};
    spacing: 8px;
}}

QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {t.border_strong};
    border-radius: 4px;
    background-color: {t.bg_sunken};
}}

QCheckBox::indicator:hover {{
    border-color: {t.border_focus};
}}

QCheckBox::indicator:checked {{
    background-color: {t.btn_primary_bg};
    border-color: {t.btn_primary_bg};
}}

/* List Widgets (Recent Runs, History) */
QListWidget {{
    background-color: {t.bg_sunken};
    color: {t.text_primary};
    border: 1px solid {t.border_subtle};
    border-radius: 6px;
    padding: 4px;
}}

QListWidget::item {{
    padding: 6px 10px;
    border-radius: 4px;
    border-bottom: 1px solid {t.border_subtle};
}}

QListWidget::item:hover {{
    background-color: {t.bg_elevated};
}}

QListWidget::item:selected {{
    background-color: {t.bg_card};
    color: {t.text_primary};
    border-left: 2px solid {t.border_focus};
}}

/* Progress Bars */
QProgressBar {{
    background-color: {t.bg_sunken};
    border: 1px solid {t.border_subtle};
    border-radius: 4px;
    text-align: center;
    color: {t.text_primary};
    font-size: 11px;
    font-weight: 600;
}}

QProgressBar::chunk {{
    background-color: {t.border_focus};
    border-radius: 3px;
}}

/* Splitters */
QSplitter::handle {{
    background-color: {t.border_subtle};
}}

QSplitter::handle:hover {{
    background-color: {t.border_focus};
}}

/* Tooltips */
QToolTip {{
    background-color: {t.bg_card};
    color: {t.text_primary};
    border: 1px solid {t.border_focus};
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 11px;
}}
"""


# -----------------------------------------------------------------------------
# 3. REUSABLE COMPONENT STYLE HELPERS (Strict Monochromatic)
# -----------------------------------------------------------------------------
def badge_style(status_text: str, dark: bool = True) -> str:
    """Return monochromatic badge styling depending on state."""
    t = get_theme(dark)
    st = status_text.strip().lower()

    if st in ("running", "active", "syncing"):
        # High-contrast active state: White fill on dark / Raven on light
        return (
            f"background-color: {t.btn_primary_bg}; color: {t.btn_primary_fg}; "
            f"border: 1px solid {t.btn_primary_bg}; border-radius: 4px; "
            f"padding: 2px 8px; font-weight: 700; font-size: 11px;"
        )
    elif st in ("completed", "synced", "ready", "online", "success"):
        # Stable success state: Solid Charcoal with crisp border
        return (
            f"background-color: {t.bg_card}; color: {t.text_primary}; "
            f"border: 1px solid {t.border_focus}; border-radius: 4px; "
            f"padding: 2px 8px; font-weight: 600; font-size: 11px;"
        )
    elif st in ("failed", "error", "aborted", "timeout"):
        # Error state: Monochromatic hatched/thick high-contrast border
        return (
            f"background-color: {t.bg_sunken}; color: {t.text_primary}; "
            f"border: 2px solid {t.border_focus}; border-radius: 4px; "
            f"padding: 2px 7px; font-weight: 700; font-size: 11px;"
        )
    elif st in ("warning", "revision", "loop"):
        # Revision state: Dashed border treatment
        return (
            f"background-color: {t.bg_elevated}; color: {t.text_primary}; "
            f"border: 1px dashed {t.border_focus}; border-radius: 4px; "
            f"padding: 2px 8px; font-weight: 600; font-size: 11px;"
        )
    else:  # Pending, Idle, Queued, Skipped
        return (
            f"background-color: {t.bg_elevated}; color: {t.text_muted}; "
            f"border: 1px solid {t.border_subtle}; border-radius: 4px; "
            f"padding: 2px 8px; font-weight: 500; font-size: 11px;"
        )


def apply_theme(target: QWidget, dark: bool = True) -> None:
    """Apply the centralized monochromatic stylesheet to a window or widget."""
    sheet = get_stylesheet(dark)
    target.setStyleSheet(sheet)
