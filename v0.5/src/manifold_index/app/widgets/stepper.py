"""app/widgets/stepper.py — Horizontal pipeline progress stepper bar.

See BLUEPRINT §9.2 and §2.7 for the visual layout.

Visual layout::

    ① Load  ─────  ② Index  ─────  ③ Fill  ─────  ④ Export
    Done ✓          Done ✓          Running          Locked

- Numbered circles (filled accent = done, outline = future).
- Clicking a completed step emits ``step_clicked(card_index)``.
- Active step label in accent colour; locked step in muted colour.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QBrush, QFont
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
    QVBoxLayout, QWidget,
)

from manifold_index.viewmodels.advisory import CardStatus
from manifold_index.app.theme import colors as C
from manifold_index.app.theme import icons


# Status → sub-label text
_STATUS_LABEL: dict[CardStatus, str] = {
    CardStatus.LOCKED:  "Locked",
    CardStatus.READY:   "Ready",
    CardStatus.RUNNING: "Running",
    CardStatus.DONE:    f"Done {icons.DONE}",
    CardStatus.WARNING: f"{icons.WARNING} Warning",
    CardStatus.ERROR:   f"{icons.ERROR} Error",
    CardStatus.STALE:   f"{icons.STALE} Stale",
}

# Numeric circle characters ①–⑨
_CIRCLE_DIGIT = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨"]


def _circle(n: int) -> str:
    if 1 <= n <= len(_CIRCLE_DIGIT):
        return _CIRCLE_DIGIT[n - 1]
    return str(n)


class _StepWidget(QWidget):
    """One step button: circle + title + status sub-label."""

    clicked = Signal(int)  # card_index

    def __init__(
        self,
        card_index: int,
        title: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._card_index = card_index
        self._status = CardStatus.LOCKED

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        # Circle + title on one row
        row = QHBoxLayout()
        row.setSpacing(6)
        row.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        self._circle_btn = QPushButton(_circle(card_index))
        self._circle_btn.setFlat(True)
        self._circle_btn.setFixedSize(24, 24)
        self._circle_btn.setCursor(Qt.CursorShape.ArrowCursor)
        self._circle_btn.clicked.connect(self._on_click)

        self._title_label = QLabel(title)
        self._title_label.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )

        row.addWidget(self._circle_btn)
        row.addWidget(self._title_label)
        layout.addLayout(row)

        # Status sub-label
        self._status_label = QLabel("Locked")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._status_label.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )
        layout.addWidget(self._status_label)

        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._refresh_style()

    def set_status(self, status: CardStatus) -> None:
        self._status = status
        self._refresh_style()

    def _on_click(self) -> None:
        if self._status in (
            CardStatus.DONE, CardStatus.WARNING, CardStatus.STALE, CardStatus.RUNNING
        ):
            self.clicked.emit(self._card_index)

    def _refresh_style(self) -> None:
        s = self._status

        # Circle button style
        if s == CardStatus.DONE:
            circle_style = (
                f"QPushButton {{ background-color: {C.ACCENT}; color: {C.TEXT_ON_ACCENT};"
                f" border-radius: 12px; font-weight: 700; font-size: 14px; border: none; }}"
                f"QPushButton:hover {{ background-color: {C.ACCENT_HOVER}; cursor: pointer; }}"
            )
            self._circle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        elif s == CardStatus.RUNNING:
            circle_style = (
                f"QPushButton {{ background-color: {C.ACCENT_MUTED}; color: {C.ACCENT};"
                f" border-radius: 12px; font-weight: 700; font-size: 14px;"
                f" border: 2px solid {C.ACCENT}; }}"
            )
            self._circle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        elif s == CardStatus.WARNING:
            circle_style = (
                f"QPushButton {{ background-color: {C.WARNING_BG}; color: {C.WARNING_BORDER};"
                f" border-radius: 12px; font-weight: 700; font-size: 14px;"
                f" border: 2px solid {C.WARNING_BORDER}; }}"
            )
            self._circle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        elif s == CardStatus.ERROR:
            circle_style = (
                f"QPushButton {{ background-color: {C.ERROR_BG}; color: {C.ERROR_BORDER};"
                f" border-radius: 12px; font-weight: 700; font-size: 14px;"
                f" border: 2px solid {C.ERROR_BORDER}; }}"
            )
            self._circle_btn.setCursor(Qt.CursorShape.ArrowCursor)
        else:  # LOCKED, READY, STALE
            circle_style = (
                f"QPushButton {{ background-color: {C.SURFACE_ALT}; color: {C.TEXT_MUTED};"
                f" border-radius: 12px; font-weight: 400; font-size: 14px;"
                f" border: 1px solid {C.BORDER}; }}"
            )
            self._circle_btn.setCursor(Qt.CursorShape.ArrowCursor)

        self._circle_btn.setStyleSheet(circle_style)

        # Title label
        if s in (CardStatus.RUNNING,):
            title_style = f"font-weight: 600; color: {C.ACCENT}; background: transparent;"
        elif s == CardStatus.DONE:
            title_style = f"font-weight: 600; color: {C.TEXT_PRIMARY}; background: transparent;"
        elif s == CardStatus.LOCKED:
            title_style = f"color: {C.TEXT_MUTED}; background: transparent;"
        else:
            title_style = f"color: {C.TEXT_SECONDARY}; background: transparent;"
        self._title_label.setStyleSheet(title_style)

        # Status sub-label
        sub = _STATUS_LABEL.get(s, s.value)
        self._status_label.setText(sub)
        if s == CardStatus.LOCKED:
            sub_style = f"font-size: 11px; color: {C.TEXT_MUTED}; background: transparent;"
        elif s == CardStatus.DONE:
            sub_style = f"font-size: 11px; color: {C.SUCCESS}; background: transparent;"
        elif s == CardStatus.RUNNING:
            sub_style = f"font-size: 11px; color: {C.ACCENT}; background: transparent;"
        elif s == CardStatus.WARNING:
            sub_style = f"font-size: 11px; color: {C.WARNING_BORDER}; background: transparent;"
        elif s == CardStatus.ERROR:
            sub_style = f"font-size: 11px; color: {C.ERROR_BORDER}; background: transparent;"
        else:
            sub_style = f"font-size: 11px; color: {C.TEXT_MUTED}; background: transparent;"
        self._status_label.setStyleSheet(sub_style)


class _ConnectorLine(QWidget):
    """A short horizontal line connecting two step widgets."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(2)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def paintEvent(self, event):  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(C.BORDER_STRONG), 1, Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        mid_y = self.height() // 2
        painter.drawLine(0, mid_y, self.width(), mid_y)


class StepperBar(QFrame):
    """Horizontal pipeline progress stepper bar.

    Parameters
    ----------
    steps : list[str]
        Ordered list of step titles, e.g. ``["Load", "Index", "Fill", "Export"]``.

    Signals
    -------
    step_clicked(int)  — card index (1-based) when a completed step is clicked.
    """

    step_clicked = Signal(int)

    def __init__(
        self,
        steps: list[str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.setProperty("class", "stepper-bar")
        self.setFrameShape(QFrame.Shape.NoFrame)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(0)

        self._step_widgets: list[_StepWidget] = []

        for i, title in enumerate(steps, start=1):
            step = _StepWidget(i, title, parent=self)
            step.clicked.connect(self.step_clicked)
            self._step_widgets.append(step)
            layout.addWidget(step)

            if i < len(steps):
                layout.addWidget(_ConnectorLine(self), 1)

    def set_step_status(self, index: int, status: CardStatus) -> None:
        """Update the visual status of step ``index`` (1-based)."""
        if 1 <= index <= len(self._step_widgets):
            self._step_widgets[index - 1].set_status(status)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

    app = QApplication(sys.argv)
    from manifold_index.app.theme.style import build_stylesheet
    app.setStyleSheet(build_stylesheet())

    win = QWidget()
    win.setWindowTitle("StepperBar smoke test")
    layout = QVBoxLayout(win)
    layout.setContentsMargins(24, 24, 24, 24)

    bar = StepperBar(["Load", "Index", "Fill", "Export"])
    bar.set_step_status(1, CardStatus.DONE)
    bar.set_step_status(2, CardStatus.DONE)
    bar.set_step_status(3, CardStatus.RUNNING)
    bar.set_step_status(4, CardStatus.LOCKED)

    bar.step_clicked.connect(lambda idx: print(f"Step {idx} clicked"))

    layout.addWidget(bar)
    layout.addStretch(1)

    win.resize(760, 120)
    win.show()
    sys.exit(app.exec())
