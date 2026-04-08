"""app/widgets/slope_input.py — (P, Q) integer-pair input with gcd validation.

See BLUEPRINT §9.6.

Usage::

    slope = SlopeInput(label="Cusp 0:", require_coprime=True)
    slope.slope_changed.connect(lambda p, q: print(p, q))
    slope.set_slope(2, 1)
    p, q = slope.get_slope()

Validation: when ``require_coprime=True`` (default), emits ``slope_changed``
and sets valid state only when gcd(|P|, |Q|) == 1.  The border turns red on
invalid input.
"""

from __future__ import annotations

import math

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QSizePolicy, QSpinBox, QWidget,
)

from manifold_index.app.theme import colors as C


class SlopeInput(QWidget):
    """(P, Q) integer-pair input with optional coprimality validation.

    Parameters
    ----------
    label : str
        Optional label shown to the left of the spin boxes.
    require_coprime : bool
        If True, gcd(|P|, |Q|) must equal 1 for the input to be valid.

    Signals
    -------
    slope_changed(int, int)  — emitted whenever both values change to a
                               valid (coprime if required) pair.
    """

    slope_changed = Signal(int, int)

    def __init__(
        self,
        label: str = "",
        require_coprime: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._require_coprime = require_coprime
        self._valid = True

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        if label:
            lbl = QLabel(label)
            lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            outer.addWidget(lbl)

        # Container frame (border changes colour on validation)
        self._container = QFrame()
        self._container.setFrameShape(QFrame.Shape.StyledPanel)
        self._container.setProperty("class", "slope-input-valid")
        container_layout = QHBoxLayout(self._container)
        container_layout.setContentsMargins(4, 2, 4, 2)
        container_layout.setSpacing(4)

        p_label = QLabel("P =")
        p_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        container_layout.addWidget(p_label)

        self._p_spin = QSpinBox()
        self._p_spin.setRange(-9999, 9999)
        self._p_spin.setValue(1)
        self._p_spin.setFixedWidth(70)
        self._p_spin.setAlignment(__import__("PySide6.QtCore", fromlist=["Qt"]).Qt.AlignmentFlag.AlignRight)
        container_layout.addWidget(self._p_spin)

        sep = QLabel("  Q =")
        sep.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        container_layout.addWidget(sep)

        self._q_spin = QSpinBox()
        self._q_spin.setRange(-9999, 9999)
        self._q_spin.setValue(0)
        self._q_spin.setFixedWidth(70)
        self._q_spin.setAlignment(__import__("PySide6.QtCore", fromlist=["Qt"]).Qt.AlignmentFlag.AlignRight)
        container_layout.addWidget(self._q_spin)

        outer.addWidget(self._container)
        outer.addStretch(1)

        # Validation label
        self._err_label = QLabel()
        self._err_label.setStyleSheet(
            f"font-size: 11px; color: {C.ERROR_TEXT}; background: transparent;"
        )
        self._err_label.setVisible(False)
        outer.addWidget(self._err_label)

        self._p_spin.valueChanged.connect(self._on_value_changed)
        self._q_spin.valueChanged.connect(self._on_value_changed)

        # Initial validation
        self._on_value_changed()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_slope(self, P: int, Q: int) -> None:
        """Programmatically set (P, Q) without emitting slope_changed."""
        self._p_spin.blockSignals(True)
        self._q_spin.blockSignals(True)
        self._p_spin.setValue(P)
        self._q_spin.setValue(Q)
        self._p_spin.blockSignals(False)
        self._q_spin.blockSignals(False)
        self._on_value_changed()

    def get_slope(self) -> tuple[int, int]:
        """Return the current (P, Q) values regardless of validity."""
        return self._p_spin.value(), self._q_spin.value()

    def set_valid(self, valid: bool) -> None:
        """Manually override the visual valid/invalid state."""
        self._apply_valid(valid)

    def is_valid(self) -> bool:
        """Return True if the current input satisfies the validation rule."""
        return self._valid

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_value_changed(self) -> None:
        P = self._p_spin.value()
        Q = self._q_spin.value()

        if self._require_coprime:
            if P == 0 and Q == 0:
                valid = False
                err = "P and Q cannot both be 0"
            else:
                g = math.gcd(abs(P), abs(Q))
                valid = (g == 1)
                err = f"gcd(|P|,|Q|) = {g} ≠ 1" if not valid else ""
        else:
            # Only disallow (0, 0)
            valid = not (P == 0 and Q == 0)
            err = "P and Q cannot both be 0" if not valid else ""

        self._valid = valid
        self._err_label.setText(err)
        self._err_label.setVisible(bool(err))
        self._apply_valid(valid)

        if valid:
            self.slope_changed.emit(P, Q)

    def _apply_valid(self, valid: bool) -> None:
        cls = "slope-input-valid" if valid else "slope-input-invalid"
        self._container.setProperty("class", cls)
        style = self._container.style()
        style.unpolish(self._container)
        style.polish(self._container)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget, QLabel

    app = QApplication(sys.argv)
    from manifold_index.app.theme.style import build_stylesheet
    app.setStyleSheet(build_stylesheet())

    win = QWidget()
    win.setWindowTitle("SlopeInput smoke test")
    layout = QVBoxLayout(win)
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(12)

    layout.addWidget(QLabel("Coprime required (default):"))
    s1 = SlopeInput(label="Slope:", require_coprime=True)
    s1.slope_changed.connect(lambda p, q: print(f"Slope: ({p}, {q})"))
    layout.addWidget(s1)

    layout.addWidget(QLabel("Already invalid (P=2, Q=4):"))
    s2 = SlopeInput(label="Slope:", require_coprime=True)
    s2.set_slope(2, 4)
    layout.addWidget(s2)

    layout.addWidget(QLabel("Coprime not required:"))
    s3 = SlopeInput(label="Range P:", require_coprime=False)
    s3.set_slope(0, 5)
    layout.addWidget(s3)

    layout.addStretch(1)
    win.resize(500, 280)
    win.show()
    sys.exit(app.exec())
