"""app/theme/style.py — Application-wide QSS stylesheet.

Call ``build_stylesheet()`` once at startup and pass the result to
``QApplication.setStyleSheet()``.

Design language: BLUEPRINT §2.  Flat, borderless, accent = deep indigo.
"""

from __future__ import annotations

from .colors import (
    ACCENT, ACCENT_HOVER, ACCENT_MUTED,
    BACKGROUND, SURFACE, SURFACE_ALT,
    BORDER, BORDER_STRONG,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, TEXT_ON_ACCENT,
    WARNING_BG, WARNING_BORDER, WARNING_TEXT,
    ERROR_BG, ERROR_BORDER, ERROR_TEXT,
    SUCCESS,
    ADVISORY_INFO_BG, ADVISORY_INFO_BORDER,
    ADVISORY_WARNING_BG, ADVISORY_WARNING_BORDER,
    ADVISORY_ERROR_BG, ADVISORY_ERROR_BORDER,
    ADVISORY_ACTION_BG, ADVISORY_ACTION_BORDER,
)


def build_stylesheet() -> str:
    """Return the full application QSS string."""
    return f"""
/* ================================================================
   Base
   ================================================================ */
QWidget {{
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    font-size: 13px;
    color: {TEXT_PRIMARY};
    background-color: {BACKGROUND};
}}

/* ================================================================
   Main window / scrollable pipeline area
   ================================================================ */
QMainWindow,
QScrollArea,
QScrollArea > QWidget,
QScrollArea > QWidget > QWidget {{
    background-color: {BACKGROUND};
    border: none;
}}

QScrollBar:vertical {{
    background: {SURFACE_ALT};
    width: 8px;
    margin: 0;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER_STRONG};
    border-radius: 4px;
    min-height: 20px;
}}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: {SURFACE_ALT};
    height: 8px;
    margin: 0;
    border-radius: 4px;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER_STRONG};
    border-radius: 4px;
    min-width: 20px;
}}
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ================================================================
   Buttons  — BLUEPRINT §2.3
   ================================================================ */

/* Primary: accent bg, white text */
QPushButton[class="primary"],
QPushButton[class="primary"]:default {{
    background-color: {ACCENT};
    color: {TEXT_ON_ACCENT};
    border: none;
    border-radius: 2px;
    padding: 5px 14px;
    font-weight: 600;
}}
QPushButton[class="primary"]:hover {{
    background-color: {ACCENT_HOVER};
}}
QPushButton[class="primary"]:pressed {{
    background-color: {ACCENT_HOVER};
    padding: 6px 14px 4px 14px;
}}
QPushButton[class="primary"]:disabled {{
    background-color: {SURFACE_ALT};
    color: {TEXT_MUTED};
}}

/* Secondary: transparent, accent border and text */
QPushButton[class="secondary"],
QPushButton {{
    background-color: transparent;
    color: {ACCENT};
    border: 1px solid {ACCENT};
    border-radius: 2px;
    padding: 4px 12px;
    font-weight: 500;
}}
QPushButton[class="secondary"]:hover,
QPushButton:hover {{
    background-color: {ACCENT_MUTED};
}}
QPushButton[class="secondary"]:pressed,
QPushButton:pressed {{
    background-color: {ACCENT_MUTED};
}}
QPushButton[class="secondary"]:disabled,
QPushButton:disabled {{
    color: {TEXT_MUTED};
    border-color: {BORDER};
}}

/* Tertiary: no border, secondary text, underline on hover */
QPushButton[class="tertiary"] {{
    background-color: transparent;
    color: {TEXT_SECONDARY};
    border: none;
    border-radius: 2px;
    padding: 2px 6px;
    text-decoration: none;
}}
QPushButton[class="tertiary"]:hover {{
    color: {TEXT_PRIMARY};
    text-decoration: underline;
}}
QPushButton[class="tertiary"]:disabled {{
    color: {TEXT_MUTED};
}}

/* ================================================================
   Inputs
   ================================================================ */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {SURFACE};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 2px;
    padding: 3px 7px;
    selection-background-color: {ACCENT_MUTED};
    selection-color: {TEXT_PRIMARY};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1px solid {ACCENT};
    outline: none;
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
QComboBox:disabled {{
    background-color: {SURFACE_ALT};
    color: {TEXT_MUTED};
    border-color: {BORDER};
}}
QLineEdit[invalid="true"] {{
    border-color: {ERROR_BORDER};
    background-color: {ERROR_BG};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox QAbstractItemView {{
    background-color: {SURFACE};
    border: 1px solid {BORDER_STRONG};
    selection-background-color: {ACCENT_MUTED};
    selection-color: {TEXT_PRIMARY};
    outline: none;
}}

/* SpinBox arrows */
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    border: none;
    width: 16px;
    background: transparent;
}}

/* ================================================================
   Check / Radio
   ================================================================ */
QCheckBox, QRadioButton {{
    spacing: 6px;
    color: {TEXT_PRIMARY};
    background: transparent;
}}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {BORDER_STRONG};
    border-radius: 2px;
    background-color: {SURFACE};
}}
QRadioButton::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {BORDER_STRONG};
    border-radius: 2px;
    background-color: {SURFACE};
}}
QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
    image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='10' viewBox='0 0 10 10'%3E%3Cpolyline points='1,5 4,8 9,2' stroke='white' stroke-width='1.5' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");
}}
QRadioButton::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
    image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='10' viewBox='0 0 10 10'%3E%3Cpolyline points='1,5 4,8 9,2' stroke='white' stroke-width='1.5' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");
}}
QCheckBox::indicator:hover,
QRadioButton::indicator:hover {{
    border-color: {ACCENT};
}}
QCheckBox:disabled,
QRadioButton:disabled {{
    color: {TEXT_MUTED};
}}
QCheckBox::indicator:disabled,
QRadioButton::indicator:disabled {{
    background-color: {SURFACE_ALT};
    border-color: {BORDER};
}}

/* ================================================================
   Labels
   ================================================================ */
QLabel {{
    background: transparent;
    color: {TEXT_PRIMARY};
}}
QLabel[class="muted"] {{
    color: {TEXT_MUTED};
    font-size: 11px;
}}
QLabel[class="secondary"] {{
    color: {TEXT_SECONDARY};
}}
QLabel[class="section-header"] {{
    font-weight: 600;
    font-size: 12px;
    color: {TEXT_SECONDARY};
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}

/* ================================================================
   Tables  — BLUEPRINT §2.8
   ================================================================ */
QTableWidget, QTableView {{
    background-color: {SURFACE};
    gridline-color: {BORDER};
    border: none;
    selection-background-color: {ACCENT_MUTED};
    selection-color: {TEXT_PRIMARY};
    alternate-background-color: {SURFACE_ALT};
}}
QTableWidget::item, QTableView::item {{
    padding: 4px 8px;
    border: none;
}}
QTableWidget::item:selected, QTableView::item:selected {{
    background-color: {ACCENT_MUTED};
    color: {TEXT_PRIMARY};
}}
QHeaderView::section {{
    background-color: {SURFACE_ALT};
    color: {TEXT_SECONDARY};
    border: none;
    border-bottom: 1px solid {BORDER_STRONG};
    padding: 4px 8px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
}}

/* ================================================================
   Progress bar
   ================================================================ */
QProgressBar {{
    background-color: {SURFACE_ALT};
    border: 1px solid {BORDER};
    border-radius: 3px;
    height: 8px;
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 3px;
}}

/* ================================================================
   CollapsibleCard  — BLUEPRINT §2.6
   ================================================================ */

/* Outer frame — 4 px left border as top accent line via left border trick */
CollapsibleCard,
QFrame[class="pipeline-card"] {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-top: 3px solid {ACCENT};
    border-radius: 2px;
    margin: 4px 0;
}}

/* Card header row */
QFrame[class="card-header"] {{
    background-color: {SURFACE};
    border: none;
    padding: 8px 12px;
}}

/* Card body content area */
QFrame[class="card-body"] {{
    background-color: {SURFACE};
    border: none;
    padding: 0 12px 12px 12px;
}}

/* Status badges  (BLUEPRINT §2.4) */
QLabel[class="badge-running"] {{
    color: {ACCENT};
    font-size: 11px;
    font-weight: 600;
}}
QLabel[class="badge-done"] {{
    color: {SUCCESS};
    font-size: 11px;
    font-weight: 600;
}}
QLabel[class="badge-warning"] {{
    color: {WARNING_BORDER};
    font-size: 11px;
    font-weight: 600;
}}
QLabel[class="badge-error"] {{
    color: {ERROR_BORDER};
    font-size: 11px;
    font-weight: 600;
}}
QLabel[class="badge-locked"],
QLabel[class="badge-ready"],
QLabel[class="badge-stale"] {{
    color: {TEXT_MUTED};
    font-size: 11px;
}}
QLabel[class="badge-stale"] {{
    font-style: italic;
}}

/* Card index circle: ①②③④ */
QLabel[class="card-index"] {{
    color: {ACCENT};
    font-weight: 700;
    font-size: 15px;
    background: transparent;
}}
QLabel[class="card-title"] {{
    font-weight: 600;
    font-size: 13px;
    color: {TEXT_PRIMARY};
    background: transparent;
}}
QLabel[class="card-summary"] {{
    color: {TEXT_SECONDARY};
    font-size: 12px;
    background: transparent;
}}

/* Expand / collapse button */
QPushButton[class="card-toggle"] {{
    background: transparent;
    border: none;
    color: {TEXT_SECONDARY};
    font-size: 14px;
    padding: 0 4px;
    min-width: 20px;
    max-width: 20px;
}}
QPushButton[class="card-toggle"]:hover {{
    color: {TEXT_PRIMARY};
}}

/* ================================================================
   AdvisoryBanner  — BLUEPRINT §2.5
   ================================================================ */
QFrame[class="advisory-info"] {{
    background-color: {ADVISORY_INFO_BG};
    border: none;
    border-left: 4px solid {ADVISORY_INFO_BORDER};
    padding: 8px 12px;
    margin: 4px 0;
}}
QFrame[class="advisory-warning"] {{
    background-color: {ADVISORY_WARNING_BG};
    border: none;
    border-left: 4px solid {ADVISORY_WARNING_BORDER};
    padding: 8px 12px;
    margin: 4px 0;
}}
QFrame[class="advisory-error"] {{
    background-color: {ADVISORY_ERROR_BG};
    border: none;
    border-left: 4px solid {ADVISORY_ERROR_BORDER};
    padding: 8px 12px;
    margin: 4px 0;
}}
QFrame[class="advisory-action"] {{
    background-color: {ADVISORY_ACTION_BG};
    border: none;
    border-left: 4px solid {ADVISORY_ACTION_BORDER};
    padding: 8px 12px;
    margin: 4px 0;
}}

QLabel[class="advisory-level-tag"] {{
    font-size: 10px;
    font-variant: small-caps;
    color: {TEXT_MUTED};
    background: transparent;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}}
QLabel[class="advisory-title"] {{
    font-weight: 600;
    font-size: 13px;
    color: {TEXT_PRIMARY};
    background: transparent;
}}
QLabel[class="advisory-body"] {{
    color: {TEXT_SECONDARY};
    font-size: 12px;
    background: transparent;
}}

/* ================================================================
   StepperBar  — BLUEPRINT §2.7
   ================================================================ */
QFrame[class="stepper-bar"] {{
    background-color: {BACKGROUND};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 8px 16px;
}}

/* ================================================================
   Tab widget (Data Hub internal sub-tabs)
   ================================================================ */
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-top: none;
    background-color: {SURFACE};
}}
QTabBar::tab {{
    background-color: {SURFACE_ALT};
    color: {TEXT_SECONDARY};
    border: none;
    border-bottom: 2px solid transparent;
    padding: 6px 16px;
    margin-right: 2px;
    font-size: 12px;
}}
QTabBar::tab:selected {{
    color: {ACCENT};
    background-color: {SURFACE};
    border-bottom: 2px solid {ACCENT};
    font-weight: 600;
}}
QTabBar::tab:hover:!selected {{
    color: {TEXT_PRIMARY};
    background-color: {SURFACE};
}}

/* ================================================================
   QGroupBox
   ================================================================ */
QGroupBox {{
    font-size: 11px;
    font-weight: 600;
    color: {TEXT_SECONDARY};
    border: 1px solid {BORDER};
    border-radius: 2px;
    margin-top: 8px;
    padding-top: 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 4px;
    color: {TEXT_SECONDARY};
    background-color: {BACKGROUND};
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}

/* ================================================================
   Progress bar (Data Hub generate)
   ================================================================ */
QProgressBar {{
    background-color: {SURFACE_ALT};
    border: none;
    border-radius: 3px;
    text-align: center;
    font-size: 11px;
    color: {TEXT_SECONDARY};
    height: 8px;
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 3px;
}}

/* ================================================================
   SlopeInput validation highlight
   ================================================================ */
QFrame[class="slope-input-invalid"] {{
    border: 1px solid {ERROR_BORDER};
    border-radius: 2px;
    background-color: {ERROR_BG};
}}
QFrame[class="slope-input-valid"] {{
    border: 1px solid {BORDER};
    border-radius: 2px;
    background-color: transparent;
}}

/* ================================================================
   Tooltips
   ================================================================ */
QToolTip {{
    background-color: {TEXT_PRIMARY};
    color: {TEXT_ON_ACCENT};
    border: none;
    padding: 4px 8px;
    font-size: 12px;
    border-radius: 2px;
}}
"""
