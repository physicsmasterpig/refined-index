"""
pages/setup_page.py — Page 1: Manifold input and computation parameters.
"""

from __future__ import annotations

from PySide6.QtCore import Signal, Slot, Qt, QTimer
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


def _has_snapy() -> bool:
    try:
        import snappy  # noqa: F401
        return True
    except ImportError:
        return False


class SetupPage(QWidget):
    """Page 1: manifold name, validate, parameters, Compute."""

    run_requested = Signal(str, int)
    # (name, q_order_half)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(600)
        self._debounce.timeout.connect(self._on_validate)
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(20)
        root.setContentsMargins(40, 32, 40, 32)

        # Header
        title = QLabel("Setup")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        root.addWidget(title)

        subtitle = QLabel(
            "Enter a cusped hyperbolic 3-manifold name and set computation parameters."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: palette(mid); font-size: 12px; margin-bottom: 8px;")
        root.addWidget(subtitle)

        # ── Manifold ──────────────────────────────────────────────
        mf_group = QGroupBox("Manifold")
        mf_form = QFormLayout(mf_group)
        mf_form.setSpacing(10)

        name_row = QHBoxLayout()
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g.  m004   4_1   L5a1   m036 …")
        self._name_edit.setFixedHeight(32)
        self._name_edit.textChanged.connect(self._on_name_changed)
        self._name_edit.returnPressed.connect(self._on_validate)

        self._validate_btn = QPushButton("Validate")
        self._validate_btn.setObjectName("secondary")
        self._validate_btn.setFixedWidth(90)
        self._validate_btn.clicked.connect(self._on_validate)

        name_row.addWidget(self._name_edit, 1)
        name_row.addWidget(self._validate_btn)
        mf_form.addRow("Name:", name_row)

        self._valid_label = QLabel("")
        self._valid_label.setWordWrap(True)
        mf_form.addRow("", self._valid_label)

        root.addWidget(mf_group)

        # ── Parameters ────────────────────────────────────────────
        param_group = QGroupBox("Parameters")
        param_form = QFormLayout(param_group)
        param_form.setSpacing(10)

        self._q_spin = QSpinBox()
        self._q_spin.setRange(4, 60)
        self._q_spin.setValue(10)
        self._q_spin.setToolTip(
            "Truncation order Nmax: series computed up to q^(Nmax/2)."
        )
        param_form.addRow("Nmax (q_order_half):", self._q_spin)

        root.addWidget(param_group)

        # ── Run button ────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._run_btn = QPushButton("Compute  ▶")
        self._run_btn.setObjectName("primary")
        self._run_btn.setFixedHeight(44)
        self._run_btn.setFixedWidth(200)
        self._run_btn.clicked.connect(self._on_run)
        btn_row.addWidget(self._run_btn)
        root.addLayout(btn_row)

        root.addStretch()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_name_changed(self, _text: str) -> None:
        """Auto-validate after a short delay."""
        self._debounce.start()

    @Slot()
    def _on_validate(self) -> None:
        self._debounce.stop()
        name = self._name_edit.text().strip()
        if not name:
            self._valid_label.setText("")
            return
        if not _has_snapy():
            self._valid_label.setText("⚠  SnaPy not installed.")
            self._valid_label.setStyleSheet("color: #d4880a;")
            return
        try:
            import snappy
            M = snappy.Manifold(name)
            n = M.num_tetrahedra()
            r = M.num_cusps()
            self._valid_label.setText(
                f"✓  {name}  —  {n} tetrahedra,  {r} cusp(s)"
            )
            self._valid_label.setStyleSheet("color: #2ea043;")
        except Exception as exc:
            self._valid_label.setText(f"✗  Not recognised: {exc}")
            self._valid_label.setStyleSheet("color: #d1242f;")

    @Slot()
    def _on_run(self) -> None:
        name = self._name_edit.text().strip()
        if not name:
            self._valid_label.setText("⚠  Enter a manifold name first.")
            self._valid_label.setStyleSheet("color: #d4880a;")
            return
        self.run_requested.emit(
            name,
            self._q_spin.value(),
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def manifold_name(self) -> str:
        return self._name_edit.text().strip()
