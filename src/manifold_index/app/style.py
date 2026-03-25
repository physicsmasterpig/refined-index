"""
app/style.py — Qt stylesheet for the v0.3.0 GUI.
"""

APP_STYLESHEET = """
QMainWindow {
    font-size: 13px;
}

/* Panel frames */
QFrame#panel {
    border: 1px solid palette(mid);
    border-radius: 6px;
    background: palette(base);
}

/* Panel headers */
QLabel#panelTitle {
    font-size: 16px;
    font-weight: bold;
    padding: 4px 0;
}
QLabel#panelSubtitle {
    font-size: 11px;
    color: palette(mid);
    padding-bottom: 4px;
}

/* Section headers */
QLabel#sectionTitle {
    font-size: 13px;
    font-weight: bold;
    color: palette(text);
    padding: 6px 0 2px 0;
    border-bottom: 1px solid palette(midlight);
    margin-bottom: 4px;
}

/* Input area */
QLineEdit {
    padding: 6px 10px;
    border: 1px solid palette(mid);
    border-radius: 4px;
    font-size: 13px;
}

QSpinBox {
    padding: 4px 8px;
    border: 1px solid palette(mid);
    border-radius: 3px;
}

/* Buttons */
QPushButton#primary {
    font-size: 13px;
    font-weight: bold;
    padding: 8px 20px;
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
QPushButton#secondary {
    padding: 6px 14px;
    border-radius: 4px;
    border: 1px solid palette(mid);
    background: palette(button);
    font-size: 12px;
}
QPushButton#secondary:hover {
    background: palette(midlight);
}

/* Progress */
QProgressBar {
    border: 1px solid palette(mid);
    border-radius: 3px;
    text-align: center;
    height: 14px;
}
QProgressBar::chunk {
    background: palette(highlight);
    border-radius: 2px;
}

/* Checkboxes */
QCheckBox {
    spacing: 6px;
    font-size: 12px;
}

/* Scroll area */
QScrollArea {
    border: none;
}
"""
