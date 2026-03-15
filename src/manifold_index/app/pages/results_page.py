"""
pages/results_page.py — Page 3: Tabbed results display.

Tabs:
  1. Series  — formatted refined index with Weyl-manifest toggle.
  2. Weyl & Dehn Filling — a/b vectors, compatibility check, Dehn fill button.
  3. Raw Data — JSON-style tree of all evaluation results.
"""

from __future__ import annotations

from fractions import Fraction

from PySide6.QtCore import Signal, Slot, Qt
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from manifold_index.app.style import monospace_font
from manifold_index.app.formatters import (
    _fmt_frac,
    _fmt_charge,
    format_weyl_manifest_text,
)
from manifold_index.core.refined_index import format_multi_point_index
from manifold_index.core.weyl_check import run_weyl_checks, strip_weyl_monomial


class ResultsPage(QWidget):
    """Page 3: tabbed results (Series, Weyl & Dehn, Raw Data)."""

    dehn_fill_requested = Signal(object, int, int, int, int, int)
    # (nz_data, cusp_idx, P, Q, q_order_half, eta_order)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pipeline_result = None
        self._basis = None
        self._nz_changed = None
        self._refined_result: list | None = None
        self._weyl_result = None
        self._filled_refined_result = None   # FilledRefinedResult from Dehn filling
        self._dehn_slope_rows: list[dict] = []
        self._weyl_a: list[Fraction] = []
        self._weyl_b: list[Fraction] = []
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(24, 24, 24, 24)

        # Header
        hdr = QHBoxLayout()
        self._title_label = QLabel("Results")
        self._title_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        hdr.addWidget(self._title_label)
        hdr.addStretch()
        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: palette(mid); font-size: 11px;")
        self._info_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        hdr.addWidget(self._info_label)
        root.addLayout(hdr)

        # Progress (visible during computation)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self._progress_bar.setFixedHeight(14)
        self._progress_bar.hide()
        root.addWidget(self._progress_bar)

        # Status
        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("font-size: 11px; color: palette(mid);")
        root.addWidget(self._status_label)

        # Tabs
        self._tabs = QTabWidget()
        root.addWidget(self._tabs, 1)

        # ── Tab 1: Series ─────────────────────────────────────────
        series_widget = QWidget()
        series_layout = QVBoxLayout(series_widget)
        series_layout.setContentsMargins(8, 12, 8, 8)

        # Copy button
        copy_row = QHBoxLayout()
        copy_row.addStretch()
        copy_btn = QPushButton("⎘  Copy to Clipboard")
        copy_btn.setObjectName("secondary")
        copy_btn.setFixedHeight(28)
        copy_btn.clicked.connect(self._on_copy)
        copy_row.addWidget(copy_btn)
        series_layout.addLayout(copy_row)

        self._series_edit = QTextEdit()
        self._series_edit.setReadOnly(True)
        self._series_edit.setFont(monospace_font())
        self._series_edit.setObjectName("series")
        self._series_edit.setMinimumHeight(200)
        series_layout.addWidget(self._series_edit, 1)

        self._tabs.addTab(series_widget, "Series")

        # ── Tab 2: Weyl & Dehn Filling ────────────────────────────
        weyl_widget = QWidget()
        weyl_layout = QVBoxLayout(weyl_widget)
        weyl_layout.setContentsMargins(8, 12, 8, 8)
        weyl_layout.setSpacing(12)

        # Weyl symmetry info
        weyl_group = QGroupBox("Weyl Symmetry")
        weyl_vbox = QVBoxLayout(weyl_group)

        self._weyl_convention = QLabel(
            "Convention:  f(η) = η^{b·m + a·e} · I(m,e),   f(η) = f(η⁻¹)"
        )
        self._weyl_convention.setFont(monospace_font(11))
        self._weyl_convention.setWordWrap(True)
        weyl_vbox.addWidget(self._weyl_convention)

        self._weyl_ab_label = QLabel("(not yet computed)")
        self._weyl_ab_label.setFont(monospace_font(11))
        self._weyl_ab_label.setWordWrap(True)
        weyl_vbox.addWidget(self._weyl_ab_label)

        self._weyl_compat_label = QLabel("")
        weyl_vbox.addWidget(self._weyl_compat_label)

        weyl_layout.addWidget(weyl_group)

        # Dehn filling controls
        dehn_group = QGroupBox("Refined Dehn Filling")
        self._dehn_layout = QVBoxLayout(dehn_group)
        self._dehn_layout.setSpacing(8)

        self._dehn_placeholder = QLabel("(available after computation)")
        self._dehn_placeholder.setStyleSheet("color: palette(mid); font-size: 11px;")
        self._dehn_layout.addWidget(self._dehn_placeholder)

        weyl_layout.addWidget(dehn_group)

        # Dehn filling result display
        self._dehn_result_group = QGroupBox("Dehn Filling Result")
        dehn_result_vbox = QVBoxLayout(self._dehn_result_group)
        self._dehn_result_edit = QTextEdit()
        self._dehn_result_edit.setReadOnly(True)
        self._dehn_result_edit.setFont(monospace_font())
        self._dehn_result_edit.setMaximumHeight(200)
        dehn_result_vbox.addWidget(self._dehn_result_edit)
        self._dehn_result_group.hide()
        weyl_layout.addWidget(self._dehn_result_group)

        weyl_layout.addStretch()
        self._tabs.addTab(weyl_widget, "Weyl && Dehn")

        # ── Tab 3: Raw Data ───────────────────────────────────────
        raw_widget = QWidget()
        raw_layout = QVBoxLayout(raw_widget)
        raw_layout.setContentsMargins(8, 12, 8, 8)
        self._raw_edit = QTextEdit()
        self._raw_edit.setReadOnly(True)
        self._raw_edit.setFont(monospace_font(10))
        raw_layout.addWidget(self._raw_edit, 1)
        self._tabs.addTab(raw_widget, "Raw Data")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, pipeline_result, basis, nz_changed=None) -> None:
        self._pipeline_result = pipeline_result
        self._basis = basis
        self._nz_changed = nz_changed
        self._refined_result = None
        self._weyl_result = None
        self._filled_refined_result = None
        self._dehn_slope_rows.clear()

        name = pipeline_result.name
        nz = nz_changed if nz_changed is not None else pipeline_result.nz_data
        q = pipeline_result.q_order_half
        self._title_label.setText(f"Results — {name}")
        self._info_label.setText(
            f"{nz.n} tet  ·  {nz.r} cusp(s)  ·  "
            f"{nz.num_hard} hard edge(s)  ·  Nmax = {q}"
        )
        self._series_edit.setPlainText("Computing …")
        self._raw_edit.setPlainText("")
        self._status_label.setText("Computing refined index …")
        self._status_label.setStyleSheet("font-size: 11px; color: palette(mid);")
        self._weyl_ab_label.setText("(computing …)")
        self._weyl_compat_label.setText("")
        self._dehn_result_group.hide()
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self._progress_bar.show()

        # Clear Dehn filling rows
        self._clear_dehn_rows()

    def show_progress(self) -> None:
        self._progress_bar.show()

    @Slot(str)
    def update_status(self, msg: str) -> None:
        self._status_label.setText(msg)

    @Slot(int, int)
    def update_progress(self, done: int, total: int) -> None:
        self._progress_bar.setRange(0, total)
        self._progress_bar.setValue(done)

    @Slot(object)
    def computation_finished(self, result: list) -> None:
        self._refined_result = result
        self._progress_bar.hide()

        if self._pipeline_result is None:
            return

        nz = self._nz_changed if self._nz_changed is not None \
            else self._pipeline_result.nz_data
        num_hard = nz.num_hard

        # Weyl check
        try:
            self._weyl_result = run_weyl_checks(result, num_hard)
        except Exception:
            self._weyl_result = None

        # Format series (tab 1)
        q_var = "q"
        eta_vars = [f"η^(2·v_{a})" for a in range(num_hard)]

        if self._weyl_result is not None and self._weyl_result.ab_valid:
            series_str = format_weyl_manifest_text(
                result, num_hard, self._weyl_result.ab, eta_vars, q_var
            )
        else:
            series_str = format_multi_point_index(
                result, num_hard, q_var, eta_vars, show_zero=False
            )
        self._series_edit.setPlainText(series_str)

        # Raw data (tab 3)
        self._populate_raw_data(result, num_hard)

        # Weyl panel (tab 2)
        self._populate_weyl_panel(num_hard)

        # Auto-save
        try:
            from manifold_index.app.formatters import auto_save_nb
            name = self._pipeline_result.name
            q_ord = self._pipeline_result.q_order_half
            basis_summary = self._basis.summary() if self._basis else ""
            saved_path = auto_save_nb(result, name, q_ord, num_hard, basis_summary)
            self._status_label.setText(
                f"✓  {len(result)} evaluation(s) complete.   Auto-saved → {saved_path}"
            )
            self._status_label.setStyleSheet("color: #2ea043; font-size: 11px;")
        except Exception as exc:
            self._status_label.setText(
                f"✓  {len(result)} evaluation(s) complete.   (Auto-save failed: {exc})"
            )
            self._status_label.setStyleSheet("color: #d4880a; font-size: 11px;")

    @Slot(object)
    def dehn_filling_finished(self, filled_result) -> None:
        """Display the Dehn filling result."""
        self._filled_refined_result = filled_result
        self._dehn_result_group.show()
        if filled_result.is_zero:
            self._dehn_result_edit.setPlainText(
                f"I^ref_{{{filled_result.P}/{filled_result.Q}}}(η)  =  0"
            )
        else:
            text = filled_result.as_q_eta_string(
                q_var="q", eta_var="η", half_pow=True
            )
            self._dehn_result_edit.setPlainText(
                f"I^ref_{{{filled_result.P}/{filled_result.Q}}}(η)  =  {text}"
            )
        self._status_label.setText(
            f"✓  Dehn filling at slope ({filled_result.P}, {filled_result.Q}) complete."
        )
        self._status_label.setStyleSheet("color: #2ea043; font-size: 11px;")
        # Re-check compatibility (only re-enables buttons that are actually compatible)
        for i in range(len(self._dehn_slope_rows)):
            self._check_slope_compat(i)

    # ------------------------------------------------------------------
    # Weyl panel helpers
    # ------------------------------------------------------------------

    def _clear_dehn_rows(self) -> None:
        while self._dehn_layout.count() > 0:
            item = self._dehn_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._dehn_slope_rows.clear()

    def _populate_weyl_panel(self, num_hard: int) -> None:
        weyl = self._weyl_result
        if weyl is not None and weyl.ab is not None:
            a_new = [Fraction(1, 2) * av for av in weyl.ab.a]
            b_new = list(weyl.ab.b)
            a_str = "  ".join(f"a_{i} = {_fmt_frac(v)}" for i, v in enumerate(a_new))
            b_str = "  ".join(f"b_{i} = {_fmt_frac(v)}" for i, v in enumerate(b_new))
            self._weyl_ab_label.setText(f"{a_str}\n{b_str}")
            compat = all(v.denominator == 1 for v in a_new)
            if compat:
                self._weyl_compat_label.setText("✓  a ∈ ℤ — Dehn-filling compatible")
                self._weyl_compat_label.setStyleSheet("color: #2ea043; font-size: 12px;")
            else:
                self._weyl_compat_label.setText(
                    "✗  a ∉ ℤ — not directly compatible (slope Q must be even)"
                )
                self._weyl_compat_label.setStyleSheet("color: #d1242f; font-size: 12px;")
            self._weyl_a = a_new
            self._weyl_b = b_new
        else:
            a_new = [Fraction(0)] * num_hard
            b_new = [Fraction(0)] * num_hard
            self._weyl_ab_label.setText("(a, b) could not be determined — insufficient data")
            self._weyl_compat_label.setText("")
            self._weyl_a = a_new
            self._weyl_b = b_new

        # Build per-cusp Dehn filling rows
        self._clear_dehn_rows()

        nz = self._nz_changed if self._nz_changed is not None \
            else (self._pipeline_result.nz_data if self._pipeline_result else None)
        num_cusps = nz.r if nz is not None else 0

        if num_cusps == 0:
            note = QLabel("No cusps — Dehn filling not applicable.")
            note.setStyleSheet("color: palette(mid); font-size: 11px;")
            self._dehn_layout.addWidget(note)
            return

        for cusp_i in range(num_cusps):
            row_widget = QWidget()
            row_hbox = QHBoxLayout(row_widget)
            row_hbox.setContentsMargins(0, 4, 0, 4)
            row_hbox.setSpacing(8)

            lbl = QLabel(f"Cusp {cusp_i}")
            lbl.setFont(monospace_font(11))
            lbl.setFixedWidth(80)
            row_hbox.addWidget(lbl)

            row_hbox.addWidget(QLabel("P:"))
            p_spin = QSpinBox()
            p_spin.setRange(-999, 999)
            p_spin.setValue(1)
            p_spin.setFixedWidth(64)
            row_hbox.addWidget(p_spin)

            row_hbox.addWidget(QLabel("Q:"))
            q_spin = QSpinBox()
            q_spin.setRange(-999, 999)
            q_spin.setValue(0)
            q_spin.setFixedWidth(64)
            row_hbox.addWidget(q_spin)

            compat_label = QLabel("")
            compat_label.setMinimumWidth(200)
            row_hbox.addWidget(compat_label)

            row_hbox.addStretch()

            fill_btn = QPushButton("Dehn Fill  ▶")
            fill_btn.setObjectName("secondary")
            fill_btn.setFixedHeight(28)
            fill_btn.clicked.connect(lambda _, j=cusp_i: self._on_dehn_fill(j))
            row_hbox.addWidget(fill_btn)

            self._dehn_layout.addWidget(row_widget)
            row_data = {
                "p_spin": p_spin,
                "q_spin": q_spin,
                "compat_label": compat_label,
                "fill_btn": fill_btn,
                "cusp_idx": cusp_i,
            }
            self._dehn_slope_rows.append(row_data)

            p_spin.valueChanged.connect(lambda _, j=cusp_i: self._check_slope_compat(j))
            q_spin.valueChanged.connect(lambda _, j=cusp_i: self._check_slope_compat(j))
            self._check_slope_compat(cusp_i)

    def _check_slope_compat(self, cusp_idx: int) -> None:
        """Recompute and display Dehn-filling compatibility for one cusp.

        The compatibility check verifies ALL hard edges simultaneously:
        for each hard edge i, a_i must be integer, 2b_i must be integer,
        and 2b_i·P + a_i·Q must be an integer.
        """
        if cusp_idx >= len(self._dehn_slope_rows):
            return
        row = self._dehn_slope_rows[cusp_idx]
        P = row["p_spin"].value()
        Q = row["q_spin"].value()
        lbl: QLabel = row["compat_label"]
        btn = row["fill_btn"]

        if P == 0 and Q == 0:
            lbl.setText("")
            btn.setEnabled(False)
            return

        a_list = getattr(self, "_weyl_a", [])
        b_list = getattr(self, "_weyl_b", [])
        num_hard = len(a_list)

        if num_hard == 0:
            # No hard edges → always compatible
            lbl.setText("✓ compatible (no hard edges)")
            lbl.setStyleSheet("color: #2ea043; font-size: 11px;")
            btn.setEnabled(True)
            return

        # Check each hard edge
        for i in range(num_hard):
            a = a_list[i]
            b = b_list[i]
            if a.denominator != 1:
                lbl.setText(f"✗ incompatible  (a_{i} = {_fmt_frac(a)} ∉ ℤ)")
                lbl.setStyleSheet("color: #d1242f; font-size: 11px;")
                btn.setEnabled(False)
                return
            if (2 * b).denominator != 1:
                lbl.setText(f"✗ incompatible  (2b_{i} = {_fmt_frac(2 * b)} ∉ ℤ)")
                lbl.setStyleSheet("color: #d1242f; font-size: 11px;")
                btn.setEnabled(False)
                return
            mu2 = 2 * b * P + a * Q
            if mu2.denominator != 1:
                lbl.setText(
                    f"✗ incompatible  (edge {i}: 2μ = {mu2} ∉ ℤ)"
                )
                lbl.setStyleSheet("color: #d1242f; font-size: 11px;")
                btn.setEnabled(False)
                return

        lbl.setText("✓ compatible")
        lbl.setStyleSheet("color: #2ea043; font-size: 11px;")
        btn.setEnabled(True)

    def _on_dehn_fill(self, cusp_idx: int) -> None:
        """Launch refined Dehn filling for one cusp."""
        if cusp_idx >= len(self._dehn_slope_rows):
            return
        row = self._dehn_slope_rows[cusp_idx]
        P = row["p_spin"].value()
        Q = row["q_spin"].value()
        if P == 0 and Q == 0:
            return

        # Disable button during computation
        row["fill_btn"].setEnabled(False)
        self._status_label.setText(
            f"Computing refined Dehn filling at ({P}, {Q}) …"
        )
        self._status_label.setStyleSheet("font-size: 11px; color: palette(mid);")

        nz = self._nz_changed if self._nz_changed is not None \
            else self._pipeline_result.nz_data
        q_ord = self._pipeline_result.q_order_half

        self.dehn_fill_requested.emit(nz, cusp_idx, P, Q, q_ord, 5)

    # ------------------------------------------------------------------
    # Raw data
    # ------------------------------------------------------------------

    def _populate_raw_data(self, result: list, num_hard: int) -> None:
        """Populate the Raw Data tab with a compact JSON-like view."""
        lines: list[str] = []
        for m_ext, e_ext, refined in result:
            if not refined:
                continue
            charge = _fmt_charge(m_ext, e_ext)
            lines.append(f"I({charge}):")
            for key in sorted(refined.keys()):
                val = refined[key]
                q_pow = key[0]
                eta_exps = key[1:]
                parts = [f"q^({q_pow}/2)"]
                for a, e2 in enumerate(eta_exps):
                    if e2 != 0:
                        parts.append(f"η_{a}^({e2}/2)")
                lines.append(f"    {' · '.join(parts)} :  {val}")
            lines.append("")
        self._raw_edit.setPlainText("\n".join(lines) if lines else "(no data)")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_copy(self) -> None:
        txt = self._series_edit.toPlainText()
        QApplication.clipboard().setText(txt)

    # ------------------------------------------------------------------
    # Accessors for export page
    # ------------------------------------------------------------------

    @property
    def refined_result(self):
        return self._refined_result

    @property
    def weyl_result(self):
        return self._weyl_result

    @property
    def filled_refined_result(self):
        return self._filled_refined_result

    @property
    def pipeline_result(self):
        return self._pipeline_result

    @property
    def basis(self):
        return self._basis

    @property
    def nz_changed(self):
        return self._nz_changed
