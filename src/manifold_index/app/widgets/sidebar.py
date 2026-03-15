"""
widgets/sidebar.py — Step-based sidebar navigation widget.
"""

from __future__ import annotations

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class Sidebar(QWidget):
    """Vertical navigation sidebar with numbered step buttons.

    Emits ``page_requested(index)`` when the user clicks a step.
    Steps can be enabled/disabled individually.
    """

    page_requested = Signal(int)

    _STEPS = [
        ("1", "Setup"),
        ("2", "Overview"),
        ("3", "Dehn Filling"),
        ("4", "Export"),
    ]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Sidebar")
        self._buttons: list[QPushButton] = []
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 16, 8, 16)
        layout.setSpacing(4)

        # App branding
        brand = QLabel("Refined 3D\nIndex")
        brand.setAlignment(Qt.AlignCenter)
        brand.setStyleSheet("font-size: 15px; font-weight: bold; padding: 8px 0 12px 0;")
        layout.addWidget(brand)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep)
        layout.addSpacing(8)

        for idx, (num, label) in enumerate(self._STEPS):
            btn = QPushButton(f"  {num}.  {label}")
            btn.setCheckable(True)
            btn.setObjectName("SidebarStep")
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setFixedHeight(40)
            btn.clicked.connect(lambda checked, i=idx: self.page_requested.emit(i))
            self._group.addButton(btn, idx)
            self._buttons.append(btn)
            layout.addWidget(btn)
            # Only first step enabled initially
            if idx > 0:
                btn.setEnabled(False)

        layout.addStretch()

        # Version label at bottom
        ver = QLabel("v1.0")
        ver.setAlignment(Qt.AlignCenter)
        ver.setStyleSheet("color: palette(mid); font-size: 10px; padding-top: 8px;")
        layout.addWidget(ver)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_active(self, index: int) -> None:
        """Visually select a step (does not emit signal)."""
        if 0 <= index < len(self._buttons):
            self._buttons[index].setChecked(True)

    def enable_up_to(self, index: int) -> None:
        """Enable steps 0 … index, disable the rest."""
        for i, btn in enumerate(self._buttons):
            btn.setEnabled(i <= index)

    def enable_step(self, index: int, enabled: bool = True) -> None:
        """Enable or disable a single step."""
        if 0 <= index < len(self._buttons):
            self._buttons[index].setEnabled(enabled)
