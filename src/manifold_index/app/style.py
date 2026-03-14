"""
app/style.py — Global stylesheet and palette helpers.

All styling goes through this module so the GUI respects system light/dark
mode automatically.  No hardcoded hex colours anywhere else.
"""

from __future__ import annotations

from PySide6.QtGui import QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication


# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------

def monospace_font(size: int = 12) -> QFont:
    f = QFontDatabase.systemFont(QFontDatabase.FixedFont)
    f.setPointSize(size)
    return f


def heading_font(size: int = 16, bold: bool = True) -> QFont:
    f = QFont()
    f.setPointSize(size)
    f.setBold(bold)
    return f


# ---------------------------------------------------------------------------
# Semantic colour helpers (resolve at call-time from current palette)
# ---------------------------------------------------------------------------

def palette() -> QPalette:
    return QApplication.instance().palette()


def success_style() -> str:
    """Green-ish text for success messages; works on light & dark."""
    return "color: #2ea043;"


def error_style() -> str:
    """Red-ish text for errors; visible on both themes."""
    return "color: #d1242f;"


def warning_style() -> str:
    """Amber text for warnings."""
    return "color: #d4880a;"


def muted_style() -> str:
    """Subdued text (captions, secondary info)."""
    return "color: palette(mid);"


# ---------------------------------------------------------------------------
# Application-level stylesheet
# ---------------------------------------------------------------------------

APP_STYLESHEET = """
/* ── Global ────────────────────────────────────────────── */
QMainWindow {
    font-size: 13px;
}

/* ── Sidebar ───────────────────────────────────────────── */
#Sidebar {
    background: palette(window);
    border-right: 1px solid palette(mid);
    min-width: 180px;
    max-width: 180px;
}
#Sidebar QPushButton {
    text-align: left;
    padding: 10px 16px;
    border: none;
    border-radius: 4px;
    font-size: 13px;
    background: transparent;
}
#Sidebar QPushButton:hover {
    background: palette(midlight);
}
#Sidebar QPushButton:checked {
    background: palette(highlight);
    color: palette(highlighted-text);
    font-weight: bold;
}
#Sidebar QPushButton:disabled {
    color: palette(mid);
}

/* ── Primary action buttons ────────────────────────────── */
QPushButton#primary {
    font-size: 14px;
    font-weight: bold;
    padding: 8px 24px;
    border-radius: 5px;
    background: palette(highlight);
    color: palette(highlighted-text);
    border: none;
}
QPushButton#primary:hover {
    background: palette(dark);
    color: palette(highlighted-text);
}
QPushButton#primary:disabled {
    background: palette(midlight);
    color: palette(mid);
}

/* ── Secondary buttons ─────────────────────────────────── */
QPushButton#secondary {
    padding: 6px 16px;
    border-radius: 4px;
    border: 1px solid palette(mid);
    background: palette(button);
}
QPushButton#secondary:hover {
    background: palette(midlight);
}

/* ── Group boxes ───────────────────────────────────────── */
QGroupBox {
    font-weight: bold;
    padding-top: 20px;
    margin-top: 8px;
    border: 1px solid palette(mid);
    border-radius: 4px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 8px;
}

/* ── Text edit (results display) ───────────────────────── */
QTextEdit#series {
    border: 1px solid palette(mid);
    border-radius: 4px;
    padding: 8px;
}

/* ── Progress bar ──────────────────────────────────────── */
QProgressBar {
    border: 1px solid palette(mid);
    border-radius: 3px;
    text-align: center;
    height: 16px;
}
QProgressBar::chunk {
    background: palette(highlight);
    border-radius: 2px;
}

/* ── Tabs ──────────────────────────────────────────────── */
QTabWidget::pane {
    border: 1px solid palette(mid);
    border-radius: 4px;
    padding: 4px;
}
QTabBar::tab {
    padding: 8px 20px;
    margin-right: 2px;
    border: 1px solid palette(mid);
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected {
    background: palette(window);
    font-weight: bold;
}
QTabBar::tab:!selected {
    background: palette(midlight);
}

/* ── Spin boxes ────────────────────────────────────────── */
QSpinBox {
    padding: 4px 8px;
    border: 1px solid palette(mid);
    border-radius: 3px;
}

/* ── Line edits ────────────────────────────────────────── */
QLineEdit {
    padding: 6px 10px;
    border: 1px solid palette(mid);
    border-radius: 4px;
}
"""
