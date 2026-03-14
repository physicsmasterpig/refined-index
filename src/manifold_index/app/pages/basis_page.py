"""
pages/basis_page.py — Page 2: Pipeline progress + per-cusp cycle selection.
"""

from __future__ import annotations

from fractions import Fraction

from PySide6.QtCore import Signal, Slot, Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from manifold_index.app.workers import PipelineResult


class BasisPage(QWidget):
    """Page 2: pipeline progress + per-cusp cycle selection + eval grid."""

    compute_requested = Signal(object, object, int, int)
    # (PipelineResult, pq_choices, m_max, e_max)
    back_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pipeline_result: PipelineResult | None = None
        self._button_groups: list[QButtonGroup] = []
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(16)
        root.setContentsMargins(40, 32, 40, 32)

        title = QLabel("Basis Selection")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        root.addWidget(title)

        # ── Progress ──────────────────────────────────────────────
        prog_frame = QWidget()
        prog_vbox = QVBoxLayout(prog_frame)
        prog_vbox.setContentsMargins(0, 0, 0, 0)
        prog_vbox.setSpacing(6)

        self._status_label = QLabel("Waiting for pipeline …")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: palette(mid); font-size: 12px;")
        prog_vbox.addWidget(self._status_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self._progress_bar.setFixedHeight(14)
        prog_vbox.addWidget(self._progress_bar)

        root.addWidget(prog_frame)

        # ── Cusp selection (scrollable) ───────────────────────────
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._cusp_container = QWidget()
        self._cusp_layout = QVBoxLayout(self._cusp_container)
        self._cusp_layout.setSpacing(12)
        self._scroll.setWidget(self._cusp_container)
        root.addWidget(self._scroll, 1)

        # ── Evaluation grid ──────────────────────────────────────
        grid_group = QGroupBox("Evaluation Grid")
        grid_form = QHBoxLayout(grid_group)
        grid_form.setSpacing(12)

        grid_form.addWidget(QLabel("| m | ≤"))
        self._m_max_spin = QSpinBox()
        self._m_max_spin.setRange(0, 20)
        self._m_max_spin.setValue(4)
        self._m_max_spin.setToolTip("Meridian range: m ∈ {−m_max, …, +m_max}")
        self._m_max_spin.valueChanged.connect(self._update_grid_info)
        grid_form.addWidget(self._m_max_spin)

        grid_form.addSpacing(16)
        grid_form.addWidget(QLabel("| e | ≤"))
        self._e_max_spin = QSpinBox()
        self._e_max_spin.setRange(0, 10)
        self._e_max_spin.setValue(2)
        self._e_max_spin.setToolTip("Longitude range: e ∈ {−e_max, …, +e_max}  (step ½)")
        self._e_max_spin.valueChanged.connect(self._update_grid_info)
        grid_form.addWidget(self._e_max_spin)

        grid_form.addSpacing(16)
        self._grid_info = QLabel("")
        self._grid_info.setStyleSheet("color: palette(mid); font-size: 11px;")
        grid_form.addWidget(self._grid_info)
        grid_form.addStretch()

        root.addWidget(grid_group)
        self._update_grid_info()

        # ── Buttons ───────────────────────────────────────────────
        btn_row = QHBoxLayout()
        back_btn = QPushButton("← Back")
        back_btn.setObjectName("secondary")
        back_btn.setFixedHeight(32)
        back_btn.clicked.connect(self.back_requested.emit)
        btn_row.addWidget(back_btn)
        btn_row.addStretch()

        self._compute_btn = QPushButton("Compute Refined Index  ▶")
        self._compute_btn.setObjectName("primary")
        self._compute_btn.setFixedHeight(40)
        self._compute_btn.setEnabled(False)
        self._compute_btn.clicked.connect(self._on_compute)
        btn_row.addWidget(self._compute_btn)
        root.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear previous computation state."""
        self._pipeline_result = None
        self._button_groups.clear()
        self._status_label.setText("Waiting for pipeline …")
        self._status_label.setStyleSheet("color: palette(mid); font-size: 12px;")
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self._compute_btn.setEnabled(False)
        while self._cusp_layout.count():
            item = self._cusp_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    @Slot(str)
    def update_status(self, msg: str) -> None:
        self._status_label.setText(msg)
        self._status_label.setStyleSheet("color: palette(text); font-size: 12px;")

    @Slot(int, int, int)
    def update_slope_progress(self, cusp_idx: int, done: int, total: int) -> None:
        self._progress_bar.setRange(0, total)
        self._progress_bar.setValue(done)
        self._status_label.setText(
            f"Cusp {cusp_idx}: testing slope {done}/{total} …"
        )

    @Slot(object)
    def pipeline_finished(self, result: PipelineResult) -> None:
        self._pipeline_result = result
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(1)
        n_nc = sum(len(r.cycles) for r in result.cycle_results)
        self._status_label.setText(
            f"✓  Pipeline complete — {n_nc} non-closable cycle(s) found."
        )
        self._status_label.setStyleSheet("color: #2ea043; font-size: 12px;")
        self._build_cusp_groups(result)
        self._compute_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_cusp_groups(self, result: PipelineResult) -> None:
        """Build radio-button groups for each cusp."""
        for cusp_idx, res in enumerate(result.cycle_results):
            group_box = QGroupBox(f"Cusp {cusp_idx}")
            vbox = QVBoxLayout(group_box)
            vbox.setSpacing(6)
            btn_group = QButtonGroup(self)
            row_idx = 0
            found_pq = {(cyc.P, cyc.Q) for cyc in res.cycles}

            for cyc in res.cycles:
                label = self._cycle_label(cyc.P, cyc.Q, non_closable=True)
                rb = QRadioButton(label)
                rb.setProperty("P", cyc.P)
                rb.setProperty("Q", cyc.Q)
                btn_group.addButton(rb, row_idx)
                vbox.addWidget(rb)
                if row_idx == 0:
                    rb.setChecked(True)
                row_idx += 1

            if not res.cycles:
                note = QLabel(
                    "⚠  No non-closable cycles found in the given slope range.\n"
                    "    Using default Meridian."
                )
                note.setStyleSheet("color: #d4880a; font-size: 11px;")
                note.setWordWrap(True)
                vbox.addWidget(note)

            # Add fallback Meridian / Longitude if not already present
            if (1, 0) not in found_pq:
                rb_m = QRadioButton(self._cycle_label(1, 0, non_closable=False))
                rb_m.setProperty("P", 1)
                rb_m.setProperty("Q", 0)
                btn_group.addButton(rb_m, row_idx)
                vbox.addWidget(rb_m)
                if not res.cycles:
                    rb_m.setChecked(True)
                row_idx += 1

            if (0, 1) not in found_pq:
                rb_l = QRadioButton(self._cycle_label(0, 1, non_closable=False))
                rb_l.setProperty("P", 0)
                rb_l.setProperty("Q", 1)
                btn_group.addButton(rb_l, row_idx)
                vbox.addWidget(rb_l)

            self._button_groups.append(btn_group)
            self._cusp_layout.addWidget(group_box)

        self._cusp_layout.addStretch()

    @staticmethod
    def _cycle_label(P: int, Q: int, non_closable: bool) -> str:
        """Build a clean label for a cycle radio button.

        No wrong '→ m = P, e = Q/2' text.  Just the cycle identity.
        """
        prefix = "Non-closable  " if non_closable else ""
        if P == 1 and Q == 0:
            return f"{prefix}Meridian μ  (slope 1, 0)"
        if P == 0 and Q == 1:
            return f"{prefix}Longitude λ  (slope 0, 1)"
        return f"{prefix}Slope ({P}, {Q})"

    def _get_choices(self) -> list[tuple[int, int]]:
        """Read the user's selection from all button groups."""
        choices = []
        for bg in self._button_groups:
            checked = bg.checkedButton()
            if checked is None:
                choices.append((1, 0))
            else:
                choices.append((checked.property("P"), checked.property("Q")))
        return choices

    def _update_grid_info(self) -> None:
        m_max = self._m_max_spin.value()
        e_max = self._e_max_spin.value()
        n_m = 2 * m_max + 1
        n_e = 4 * e_max + 1  # step ½
        n_cusps = len(self._button_groups) if self._button_groups else 1
        if n_cusps == 1:
            total = n_m * n_e
        else:
            total = (n_m * n_e) ** n_cusps
        self._grid_info.setText(f"→ {total} evaluation point(s)")

    @Slot()
    def _on_compute(self) -> None:
        if self._pipeline_result is None:
            return
        choices = self._get_choices()
        self.compute_requested.emit(
            self._pipeline_result,
            choices,
            self._m_max_spin.value(),
            self._e_max_spin.value(),
        )
