"""
pages/dehn_filling_page.py — Page 3: Dehn filling workflow.

Workflow:
  1. User specifies a Dehn filling slope (P, Q) for one cusp
  2. Auto-search non-closable cycles in a small slope range
  3. For each non-closable cycle: basis change → transform slope → compute
  4. Display the Dehn-filled refined index
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd

from PySide6.QtCore import Signal, Slot, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from manifold_index.app.style import monospace_font
from manifold_index.app.formatters import _fmt_frac


class DehnFillingPage(QWidget):
    """Page 3: Dehn filling slope input → non-closable search → results."""

    compute_requested = Signal(object, int, int, int, int, object, object)
    # (nz_data, cusp_idx, P_user, Q_user, q_order_half, p_range, q_range)
    back_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._nz_data = None
        self._manifold_data = None
        self._easy_result = None
        self._weyl_result = None
        self._q_order_half: int = 10
        self._cusp_rows: list[dict] = []
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(24, 24, 24, 24)

        # Header
        hdr = QHBoxLayout()
        self._title_label = QLabel("Dehn Filling")
        self._title_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        hdr.addWidget(self._title_label)
        hdr.addStretch()
        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: palette(mid); font-size: 11px;")
        self._info_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        hdr.addWidget(self._info_label)
        root.addLayout(hdr)

        subtitle = QLabel(
            "Specify a Dehn filling slope (P, Q) at one cusp.  "
            "Non-closable cycles will be searched automatically."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: palette(mid); font-size: 12px; margin-bottom: 4px;")
        root.addWidget(subtitle)

        # ── Manifold info panel ───────────────────────────────────
        self._info_group = QGroupBox("Manifold & Edge Data")
        info_vbox = QVBoxLayout(self._info_group)
        self._info_label = QLabel("(not loaded)")
        self._info_label.setFont(monospace_font(11))
        self._info_label.setWordWrap(True)
        self._info_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        info_vbox.addWidget(self._info_label)
        self._info_group.hide()
        root.addWidget(self._info_group)

        # ── Per-cusp slope inputs ─────────────────────────────────
        self._slopes_group = QGroupBox("Dehn Filling Slopes")
        self._slopes_layout = QVBoxLayout(self._slopes_group)
        self._slopes_layout.setSpacing(8)
        root.addWidget(self._slopes_group)

        # ── Non-closable cycle search range ───────────────────────
        range_group = QGroupBox("Non-closable Cycle Search Range")
        range_hbox = QHBoxLayout(range_group)
        range_hbox.setSpacing(12)

        range_hbox.addWidget(QLabel("P ∈ ["))
        self._nc_p_min = QSpinBox()
        self._nc_p_min.setRange(-20, 0)
        self._nc_p_min.setValue(-2)
        range_hbox.addWidget(self._nc_p_min)
        range_hbox.addWidget(QLabel(","))
        self._nc_p_max = QSpinBox()
        self._nc_p_max.setRange(0, 20)
        self._nc_p_max.setValue(2)
        range_hbox.addWidget(self._nc_p_max)
        range_hbox.addWidget(QLabel("]"))

        range_hbox.addSpacing(16)

        range_hbox.addWidget(QLabel("Q ∈ ["))
        self._nc_q_min = QSpinBox()
        self._nc_q_min.setRange(-20, 0)
        self._nc_q_min.setValue(-2)
        range_hbox.addWidget(self._nc_q_min)
        range_hbox.addWidget(QLabel(","))
        self._nc_q_max = QSpinBox()
        self._nc_q_max.setRange(0, 20)
        self._nc_q_max.setValue(2)
        range_hbox.addWidget(self._nc_q_max)
        range_hbox.addWidget(QLabel("]"))

        range_hbox.addStretch()
        root.addWidget(range_group)

        # ── Compute button ────────────────────────────────────────
        compute_row = QHBoxLayout()
        compute_row.addStretch()
        self._compute_btn = QPushButton("Compute Dehn Filling  ▶")
        self._compute_btn.setObjectName("primary")
        self._compute_btn.setFixedHeight(40)
        self._compute_btn.setEnabled(False)
        self._compute_btn.clicked.connect(self._on_compute)
        compute_row.addWidget(self._compute_btn)
        root.addLayout(compute_row)

        # ── Progress ──────────────────────────────────────────────
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self._progress_bar.setFixedHeight(14)
        self._progress_bar.hide()
        root.addWidget(self._progress_bar)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("font-size: 11px; color: palette(mid);")
        root.addWidget(self._status_label)

        # ── Results ───────────────────────────────────────────────
        self._results_group = QGroupBox("Dehn Filling Results")
        results_vbox = QVBoxLayout(self._results_group)

        self._results_edit = QTextEdit()
        self._results_edit.setReadOnly(True)
        self._results_edit.setFont(monospace_font())
        self._results_edit.setMinimumHeight(160)
        results_vbox.addWidget(self._results_edit, 1)

        self._results_group.hide()
        root.addWidget(self._results_group, 1)

        # ── Back button ──────────────────────────────────────────
        btn_row = QHBoxLayout()
        back_btn = QPushButton("← Back")
        back_btn.setObjectName("secondary")
        back_btn.setFixedHeight(32)
        back_btn.clicked.connect(self.back_requested.emit)
        btn_row.addWidget(back_btn)
        btn_row.addStretch()
        root.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, name: str, nz_data, q_order_half: int,
              manifold_data=None, easy_result=None, weyl_result=None) -> None:
        """Initialise the Dehn filling page for a given manifold."""
        self._nz_data = nz_data
        self._manifold_data = manifold_data
        self._easy_result = easy_result
        self._weyl_result = weyl_result
        self._q_order_half = q_order_half
        self._title_label.setText(f"Dehn Filling — {name}")
        self._info_label.setText(
            f"{nz_data.r} cusp(s)  ·  Nmax = {q_order_half}"
        )
        self._status_label.setText("")
        self._progress_bar.hide()
        self._results_group.hide()
        self._results_edit.clear()
        self._build_cusp_rows(nz_data.r)
        self._compute_btn.setEnabled(nz_data.r > 0)
        self._populate_manifold_info(name, nz_data, manifold_data, easy_result, weyl_result)

    @Slot(str)
    def update_status(self, msg: str) -> None:
        self._status_label.setText(msg)
        self._status_label.setStyleSheet("font-size: 11px; color: palette(mid);")

    @Slot(int, int)
    def update_progress(self, done: int, total: int) -> None:
        self._progress_bar.setRange(0, total)
        self._progress_bar.setValue(done)
        self._progress_bar.show()

    @Slot(object)
    def dehn_filling_finished(self, result_info: dict) -> None:
        """Called when the full Dehn filling pipeline completes.

        Parameters
        ----------
        result_info : dict with keys:
            'cusp_idx' : int
            'P_user', 'Q_user' : int  (user's slope)
            'non_closable_cycles' : list of (P_nc, Q_nc)
            'results' : list of dicts with keys:
                'P_nc', 'Q_nc' : int (non-closable cycle)
                'a', 'b' : int (Bézout coefficients)
                'P_new', 'Q_new' : int (transformed slope)
                'filled_result' : FilledRefinedResult
        """
        self._progress_bar.hide()
        self._compute_btn.setEnabled(True)

        cusp_idx = result_info["cusp_idx"]
        P_user = result_info["P_user"]
        Q_user = result_info["Q_user"]
        nc_cycles = result_info["non_closable_cycles"]
        results = result_info["results"]

        # Import HJ-CF for display
        try:
            from manifold_index.core.refined_dehn_filling import hj_continued_fraction
        except ImportError:
            hj_continued_fraction = None

        lines: list[str] = []
        lines.append(f"Cusp {cusp_idx}: Dehn filling with slope ({P_user}, {Q_user})")
        lines.append(f"  = {P_user}·α_{cusp_idx} + {Q_user}·β_{cusp_idx}")

        # Show HJ-CF of the user's slope
        if hj_continued_fraction is not None and Q_user != 0:
            try:
                hj = hj_continued_fraction(P_user, Q_user)
                lines.append(f"  HJ continued fraction of {P_user}/{Q_user}: {hj}  (ℓ = {len(hj)})")
            except Exception:
                pass
        lines.append("")

        # Weyl shift status
        if self._weyl_result is not None and self._weyl_result.ab is not None and self._weyl_result.ab.is_valid:
            lines.append("✓ Weyl shift η^{b·m + a·e} applied BEFORE Dehn filling summation")
        else:
            lines.append("⚠ No valid Weyl (a,b) — Dehn filling WITHOUT Weyl shift")
        lines.append("")

        if not nc_cycles:
            lines.append("⚠  No non-closable cycles found in the given range.")
            lines.append("   Try increasing the search range.")
            self._status_label.setText("No non-closable cycles found.")
            self._status_label.setStyleSheet("color: #d4880a; font-size: 11px;")
        else:
            lines.append(f"Non-closable cycles found: {len(nc_cycles)}")
            for P_nc, Q_nc in nc_cycles:
                lines.append(f"  ({P_nc}, {Q_nc})")
            lines.append("")

            for i, res in enumerate(results):
                P_nc = res["P_nc"]
                Q_nc = res["Q_nc"]
                a_coeff = res["a"]
                b_coeff = res["b"]
                P_new = res["P_new"]
                Q_new = res["Q_new"]
                filled = res["filled_result"]

                lines.append(f"── Non-closable cycle ({P_nc}, {Q_nc}) ──")
                lines.append(f"  New meridian M' = {P_nc}·α + {Q_nc}·β")
                lines.append(f"  New longitude L' = {2*a_coeff}·α + {b_coeff}·β")
                lines.append(f"  Bézout: P·b − 2Q·a = {P_nc}·{b_coeff} − 2·{Q_nc}·{a_coeff} = {P_nc*b_coeff - 2*Q_nc*a_coeff}")
                lines.append(f"  Transformed slope: ({P_user}, {Q_user}) → ({P_new}, {Q_new})")

                # Show HJ-CF of the transformed slope
                if hj_continued_fraction is not None and Q_new != 0:
                    try:
                        hj_new = hj_continued_fraction(P_new, Q_new)
                        lines.append(f"  HJ-CF of {P_new}/{Q_new}: {hj_new}  (ℓ = {len(hj_new)})")
                    except Exception:
                        pass
                lines.append("")

                if filled.is_zero:
                    lines.append(f"  I^ref_{{({P_new},{Q_new})}}(η)  =  0")
                else:
                    text = filled.as_q_eta_string(
                        q_var="q", eta_var="η", half_pow=True
                    )
                    lines.append(f"  I^ref_{{({P_new},{Q_new})}}(η)  =  {text}")
                lines.append("")

            self._status_label.setText(
                f"✓  Dehn filling at ({P_user}, {Q_user}) complete — "
                f"{len(nc_cycles)} non-closable cycle(s)."
            )
            self._status_label.setStyleSheet("color: #2ea043; font-size: 11px;")

        self._results_edit.setPlainText("\n".join(lines))
        self._results_group.show()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _populate_manifold_info(self, name, nz_data, manifold_data, easy_result, weyl_result):
        """Build and display the manifold/edge info panel."""
        if nz_data is None:
            self._info_group.hide()
            return

        n = nz_data.n
        r = nz_data.r
        num_hard = nz_data.num_hard
        num_easy = nz_data.num_easy

        lines = []
        lines.append(f"Manifold: {name}")
        lines.append(f"Tetrahedra: {n}   Cusps: {r}")
        lines.append(f"Internal edges: {n - r}  (hard: {num_hard}, easy: {num_easy})")
        lines.append("")

        # ── SnaPy edge data ───────────────────────────────────────
        if manifold_data is not None:
            lines.append("═══ SnaPy Edge Equations ═══")
            edge_eqs = manifold_data.edge_equations  # shape (n, 3n)
            for i in range(n):
                row = edge_eqs[i]
                # Format as compact triplets: (Z, Z', Z'') per tet
                parts = []
                for t in range(n):
                    triplet = row[3*t : 3*t + 3]
                    parts.append(f"({triplet[0]},{triplet[1]},{triplet[2]})")
                lines.append(f"  Edge {i}: {' '.join(parts)}")
            lines.append("")

        # ── Easy / hard edge classification ───────────────────────
        if easy_result is not None:
            lines.append("═══ Easy Edges (all found) ═══")
            for idx, edge_vec in enumerate(easy_result.all_easy):
                marker = ""
                if idx in easy_result.independent_easy_indices:
                    # Find position in basis
                    basis_pos = (
                        r + num_hard
                        + easy_result.independent_easy_indices.index(idx)
                    )
                    marker = f"  ← basis row {basis_pos} (independent easy)"
                # Compact display of 3n-vector as triplets
                parts = []
                for t in range(n):
                    triplet = edge_vec[3*t : 3*t + 3]
                    parts.append(f"({triplet[0]},{triplet[1]},{triplet[2]})")
                lines.append(f"  E{idx}: {' '.join(parts)}{marker}")

            lines.append("")
            lines.append(f"Independent easy: {len(easy_result.independent_easy_indices)}")
            lines.append(f"  Indices: {easy_result.independent_easy_indices}")
            lines.append("")

            if easy_result.hard_padding:
                lines.append("═══ Hard Edges (SnaPy padding) ═══")
                for h_idx, hard_vec in enumerate(easy_result.hard_padding):
                    basis_pos = r + h_idx
                    parts = []
                    for t in range(n):
                        triplet = hard_vec[3*t : 3*t + 3]
                        parts.append(f"({triplet[0]},{triplet[1]},{triplet[2]})")
                    lines.append(f"  H{h_idx}: {' '.join(parts)}  ← basis row {basis_pos}")
                lines.append("")
            else:
                lines.append("No hard edges — all internal edges are easy.")
                lines.append("")

        # ── Weyl (a, b) vectors ───────────────────────────────────
        if weyl_result is not None and weyl_result.ab is not None:
            lines.append("═══ Weyl Symmetry (a, b) Vectors ═══")
            ab = weyl_result.ab
            for j in range(ab.num_hard):
                a_actual = Fraction(1, 2) * ab.a[j]
                b_val = ab.b[j]
                lines.append(
                    f"  Hard edge {j}: a = {_fmt_frac(a_actual)}, "
                    f"b = {_fmt_frac(b_val)}  "
                    f"(η^{{b·m + a·e}} shift applied before Dehn filling)"
                )
            if ab.is_valid:
                lines.append("  ✓ All (a, b) valid for Dehn filling")
            else:
                lines.append("  ✗ Some (a, b) values invalid — filling may not be well-defined")
            lines.append("")
        elif weyl_result is not None:
            lines.append("═══ Weyl Symmetry ═══")
            lines.append("  (a, b) vectors could not be determined.")
            lines.append("")

        self._info_label.setText("\n".join(lines))
        self._info_group.show()

    def _build_cusp_rows(self, num_cusps: int) -> None:
        """Build per-cusp slope input rows."""
        # Clear old rows
        self._cusp_rows.clear()
        while self._slopes_layout.count():
            item = self._slopes_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if num_cusps == 0:
            note = QLabel("Closed manifold — no Dehn filling applicable.")
            note.setStyleSheet("color: palette(mid); font-size: 11px;")
            self._slopes_layout.addWidget(note)
            return

        for cusp_i in range(num_cusps):
            row_widget = QWidget()
            row_hbox = QHBoxLayout(row_widget)
            row_hbox.setContentsMargins(0, 4, 0, 4)
            row_hbox.setSpacing(8)

            lbl = QLabel(f"Cusp {cusp_i}:")
            lbl.setFont(monospace_font(11))
            lbl.setFixedWidth(70)
            row_hbox.addWidget(lbl)

            unfilled_chk = QCheckBox("*  (unfilled)")
            unfilled_chk.setChecked(True)
            row_hbox.addWidget(unfilled_chk)

            row_hbox.addSpacing(8)
            row_hbox.addWidget(QLabel("P:"))
            p_spin = QSpinBox()
            p_spin.setRange(-999, 999)
            p_spin.setValue(3)
            p_spin.setFixedWidth(64)
            p_spin.setEnabled(False)
            row_hbox.addWidget(p_spin)

            row_hbox.addWidget(QLabel("Q:"))
            q_spin = QSpinBox()
            q_spin.setRange(-999, 999)
            q_spin.setValue(1)
            q_spin.setFixedWidth(64)
            q_spin.setEnabled(False)
            row_hbox.addWidget(q_spin)

            row_hbox.addStretch()

            # Toggle P/Q spinners based on unfilled checkbox
            unfilled_chk.toggled.connect(
                lambda checked, ps=p_spin, qs=q_spin: (
                    ps.setEnabled(not checked), qs.setEnabled(not checked)
                )
            )

            self._slopes_layout.addWidget(row_widget)
            self._cusp_rows.append({
                "cusp_idx": cusp_i,
                "unfilled_chk": unfilled_chk,
                "p_spin": p_spin,
                "q_spin": q_spin,
            })

    def _get_filled_cusps(self) -> list[tuple[int, int, int]]:
        """Return list of (cusp_idx, P, Q) for filled cusps."""
        filled = []
        for row in self._cusp_rows:
            if not row["unfilled_chk"].isChecked():
                cusp_i = row["cusp_idx"]
                P = row["p_spin"].value()
                Q = row["q_spin"].value()
                if P == 0 and Q == 0:
                    continue
                filled.append((cusp_i, P, Q))
        return filled

    @Slot()
    def _on_compute(self) -> None:
        if self._nz_data is None:
            return

        filled = self._get_filled_cusps()
        if not filled:
            self._status_label.setText("⚠  No cusps marked for filling.")
            self._status_label.setStyleSheet("color: #d4880a; font-size: 11px;")
            return

        if len(filled) > 1:
            self._status_label.setText(
                "⚠  Multi-cusp simultaneous filling is not yet supported.  "
                "Please fill one cusp at a time."
            )
            self._status_label.setStyleSheet("color: #d4880a; font-size: 11px;")
            return

        cusp_idx, P_user, Q_user = filled[0]
        if gcd(abs(P_user), abs(Q_user)) != 1:
            self._status_label.setText(
                f"⚠  Slope ({P_user}, {Q_user}) is not primitive "
                f"(gcd = {gcd(abs(P_user), abs(Q_user))})."
            )
            self._status_label.setStyleSheet("color: #d4880a; font-size: 11px;")
            return

        p_range = range(self._nc_p_min.value(), self._nc_p_max.value() + 1)
        q_range = range(self._nc_q_min.value(), self._nc_q_max.value() + 1)

        self._compute_btn.setEnabled(False)
        self._results_group.hide()
        self._status_label.setText(
            f"Searching non-closable cycles at cusp {cusp_idx} …"
        )
        self._status_label.setStyleSheet("font-size: 11px; color: palette(mid);")
        self._progress_bar.show()

        self.compute_requested.emit(
            self._nz_data,
            cusp_idx,
            P_user,
            Q_user,
            self._q_order_half,
            p_range,
            q_range,
        )

    # ------------------------------------------------------------------
    # Accessors (for export page)
    # ------------------------------------------------------------------

    @property
    def nz_data(self):
        return self._nz_data
