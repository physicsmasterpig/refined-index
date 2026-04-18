"""app/pipeline/filling_card.py — Card ③: Dehn Filling.

BLUEPRINT §10.4.

Phase A: Find NC cycles (NCSearchWorker per cusp).
Phase B: Compute filled index (FillWorker per query).

Layout
------
  • NC source row: p/q range spinboxes + "Use Cache" checkbox
  • "Find NC Cycles" button → NCSearchWorker
  • NC cycle table (MathView)
  • Filled query form: NC cycle selector + filling slope + "Compute Filling" button
  • SeriesTable for filling results
"""

from __future__ import annotations

import logging
from fractions import Fraction

from PySide6.QtCore import QCoreApplication, QTimer, Signal
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QGroupBox,
    QHBoxLayout, QLabel, QProgressBar, QPushButton, QRadioButton,
    QStackedWidget, QVBoxLayout, QWidget,
)

from manifold_index.app.widgets.no_scroll_spin import (
    NoScrollDoubleSpinBox as QDoubleSpinBox,
    NoScrollSpinBox as QSpinBox,
)

from manifold_index.services.session import (
    FillQuery, MultiFillQuery, NCCycleSet, PipelineStage, Session,
)
from manifold_index.services.compute_service import ComputeService
from manifold_index.viewmodels.advisory import CardStatus
from manifold_index.viewmodels.filling_vm import (
    build_nc_cycle_vm, NCCycleViewModel,
)
from manifold_index.formatters.filling_fmt_v2 import (
    format_slope_latex, format_filled_series_latex,
    format_nc_cycle_table_html, format_unrefined_series_latex,
    format_fill_result_detailed, frac_to_latex,
    format_filled_index_table_html, format_fill_result_as_index_row,
    format_fill_info_html, format_multi_fill_row_label, format_charge_as_alphabeta,
    build_fill_row_cells, build_fill_placeholder_cells,
    build_multi_fill_row_cells, build_multi_fill_placeholder_cells,
)
from manifold_index.app.widgets.collapsible_card import CollapsibleCard
from manifold_index.app.widgets.series_table import SeriesTable
from manifold_index.app.widgets.math_view import MathView
from manifold_index.app.widgets.slope_input import SlopeInput
from manifold_index.app.workers.nc_search_worker import NCSearchWorker
from manifold_index.app.workers.fill_worker import (
    FillWorker, MultiFillWorker, UnrefinedFillWorker, UnrefinedKernelFillWorker,
)
from manifold_index.app.workers.weyl_worker import WeylWorker, NcCompatWorker, MultiCuspNcCompatWorker
from manifold_index.viewmodels.index_vm import build_weyl_vm
from manifold_index.formatters.weyl_fmt import format_weyl_html


def _compute_extended_incompat_edges(ab, refined_adj_pass) -> list[int]:
    """Return edge indices where W_j must be turned off.

    W_j is compatible with Dehn filling iff ALL three hold:
      1. a[j] ∈ ℤ
      2. 2·b[j] ∈ ℤ
      3. refined q^1 projection = -1  (adj_pass is True)

    If condition 3 fails, ALL edges are incompatible regardless of a/b.
    """
    if ab is None:
        return []
    base = _compute_incompat_edges(ab)
    if refined_adj_pass is not True:
        # q^1 fails globally → all W_j are incompatible
        return list(range(len(ab.a)))
    return base


def _compute_incompat_edges(ab) -> list[int]:
    """Return list of edge indices j where a[j] ∉ ℤ or 2*b[j] ∉ ℤ.

    Parameters
    ----------
    ab : Weyl result object with .a and .b attributes

    Returns
    -------
    list of edge indices to zero out
    """
    incompat = []
    if ab is None:
        return incompat
    for j in range(len(ab.a)):
        a_val = ab.a[j]
        b_val = ab.b[j]
        # Check if a[j] is an integer
        if not isinstance(a_val, int) and a_val != int(a_val):
            incompat.append(j)
        # Check if 2*b[j] is an integer
        elif not isinstance(b_val, (int, float)):
            try:
                if (2 * b_val) != int(2 * b_val):
                    incompat.append(j)
            except Exception:
                incompat.append(j)
        elif isinstance(b_val, float):
            if (2 * b_val) != int(2 * b_val):
                incompat.append(j)
    return incompat


class FillingCard(QWidget):
    """Card ③: NC-cycle search + filled refined index.

    Signals
    -------
    session_updated(Session)
    """

    session_updated = Signal(object)

    def __init__(self, session: Session, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._session = session
        self._nc_workers: list[NCSearchWorker] = []
        self._fill_workers: list[FillWorker] = []
        self._nc_cycle_vms: list[NCCycleViewModel] = []
        self._fill_edge_checkboxes: list[QCheckBox] = []
        self._fill_cusp_checkboxes: list[QCheckBox] = []
        self._session_gen: int = 0   # incremented on each unlock(); guards stale signals
        self._auto_weyl_worker: WeylWorker | None = None  # Weyl check before fill
        self._nc_weyl_workers: dict[tuple[int, int], WeylWorker] = {}  # Weyl per NC cycle (P,Q)
        self._nc_weyl_results: dict[tuple[int, int], dict] = {}  # Results per cycle
        self._joint_adjoint_worker: "MultiCuspNcCompatWorker | None" = None  # joint multi-cusp check
        self._last_fill_info_specs: list = []        # cusp_specs last passed to fill-info view
        self._last_fill_info_result = None           # last fill result
        self._last_adjoint_per_cusp: "list | None" = None  # latest per-cusp adjoint results
        self._last_fill_info_ab = None               # ABVectors for per-edge a/b display
        self._last_fill_info_adj_pass: "bool | None" = None  # latest joint adj_pass gate
        self._nc_worker_progress: dict[int, tuple[int, int]] = {}
        self._fill_current_row: int | None = None
        self._fill_grid_total: int = 0
        self._fill_grid_done: int = 0
        self._cusp_fill_rows: list[dict] = []  # multi-cusp: list of {cusp_idx, nc_combo, slope}
        self._cusp_fill_container: QWidget | None = None  # multi-cusp container
        self._auto_weyl_status: QLabel | None = None  # inline status for auto-Weyl

        self._card = CollapsibleCard(3, "Dehn Filling", parent=self)
        self._card.set_status(CardStatus.LOCKED)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._card)

        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(10)

        # ── NC search parameters ──────────────────────────────────────
        search_box = QGroupBox("NC Cycle Search")
        search_layout = QVBoxLayout(search_box)

        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("P range ±"))
        self._p_range_spin = QSpinBox()
        self._p_range_spin.setRange(1, 50)
        self._p_range_spin.setValue(1)
        self._p_range_spin.setFixedWidth(55)
        range_row.addWidget(self._p_range_spin)
        range_row.addWidget(QLabel("  Q max"))
        self._q_range_spin = QSpinBox()
        self._q_range_spin.setRange(0, 50)
        self._q_range_spin.setValue(1)
        self._q_range_spin.setFixedWidth(55)
        range_row.addWidget(self._q_range_spin)
        self._cache_chk = QCheckBox("Use cache")
        range_row.addWidget(self._cache_chk)
        range_row.addStretch(1)
        search_layout.addLayout(range_row)

        nc_btn_row = QHBoxLayout()
        self._nc_search_btn = QPushButton("Find NC Cycles")
        self._nc_search_btn.setProperty("class", "primary")
        self._nc_search_btn.clicked.connect(self._on_find_nc_clicked)
        nc_btn_row.addWidget(self._nc_search_btn)
        self._nc_stop_btn = QPushButton("Stop")
        self._nc_stop_btn.setProperty("class", "secondary")
        self._nc_stop_btn.setEnabled(False)
        self._nc_stop_btn.clicked.connect(self._on_nc_stop_clicked)
        nc_btn_row.addWidget(self._nc_stop_btn)
        nc_btn_row.addStretch(1)
        search_layout.addLayout(nc_btn_row)

        self._nc_progress = QProgressBar()
        self._nc_progress.setRange(0, 0)
        self._nc_progress.setVisible(False)
        search_layout.addWidget(self._nc_progress)

        self._nc_status = QLabel()
        self._nc_status.setProperty("class", "muted")
        self._nc_status.setVisible(False)
        search_layout.addWidget(self._nc_status)

        bl.addWidget(search_box)

        # ── NC cycle tables (one per cusp, rebuilt on unlock) ─────────
        self._nc_table_box = QGroupBox("NC Cycles")
        self._nc_table_layout = QVBoxLayout(self._nc_table_box)
        self._nc_table_views: list[MathView] = []   # rebuilt in _rebuild_nc_tables
        self._nc_table_labels: list[QLabel] = []
        bl.addWidget(self._nc_table_box)

        # ── Filled query form ─────────────────────────────────────────
        fill_box = QGroupBox("Compute Filled Index")
        fill_layout = QVBoxLayout(fill_box)

        cusp_row = QHBoxLayout()
        cusp_row.addWidget(QLabel("Cusp:"))
        self._cusp_combo = QComboBox()
        cusp_row.addWidget(self._cusp_combo)
        cusp_row.addWidget(QLabel("  NC cycle:"))
        self._nc_combo = QComboBox()
        self._nc_combo.setMinimumWidth(120)
        self._nc_combo.currentIndexChanged.connect(self._on_nc_selected)
        cusp_row.addWidget(self._nc_combo)
        cusp_row.addStretch(1)
        fill_layout.addLayout(cusp_row)

        self._fill_slope = SlopeInput(label="Filling slope:", require_coprime=True)
        fill_layout.addWidget(self._fill_slope)

        # ── Unfilled cusp charges (multi-cusp only) ───────────────────
        self._other_box = QGroupBox("Unfilled Cusp Charges")
        other_layout = QVBoxLayout(self._other_box)

        mode_row = QHBoxLayout()
        self._other_point = QRadioButton("Point")
        self._other_grid  = QRadioButton("Grid")
        self._other_grid.setChecked(True)
        other_mode_grp = QButtonGroup(self._other_box)
        other_mode_grp.addButton(self._other_point)
        other_mode_grp.addButton(self._other_grid)
        self._other_point.toggled.connect(self._on_other_mode_changed)
        mode_row.addWidget(self._other_point)
        mode_row.addWidget(self._other_grid)
        mode_row.addStretch(1)
        other_layout.addLayout(mode_row)

        # Point mode: per-cusp m/e are set in the cusp rows above
        self._other_point_row = QWidget()
        pr = QHBoxLayout(self._other_point_row)
        pr.setContentsMargins(0, 0, 0, 0)
        _hint = QLabel("Set m and e per unfilled cusp in the rows above.")
        _hint.setProperty("class", "muted")
        pr.addWidget(_hint)
        pr.addStretch(1)
        self._other_point_row.setVisible(False)  # grid is default
        other_layout.addWidget(self._other_point_row)

        # Grid mode: m and e ranges
        self._other_grid_row = QWidget()
        gr = QHBoxLayout(self._other_grid_row)
        gr.setContentsMargins(0, 0, 0, 0)
        gr.addWidget(QLabel("m ∈ ["))
        self._other_m_min = QSpinBox()
        self._other_m_min.setRange(-99, 99)
        self._other_m_min.setValue(0)
        self._other_m_min.setFixedWidth(55)
        gr.addWidget(self._other_m_min)
        gr.addWidget(QLabel(","))
        self._other_m_max = QSpinBox()
        self._other_m_max.setRange(-99, 99)
        self._other_m_max.setValue(1)
        self._other_m_max.setFixedWidth(55)
        gr.addWidget(self._other_m_max)
        gr.addWidget(QLabel("]  e ∈ ["))
        self._other_e_min = QDoubleSpinBox()
        self._other_e_min.setRange(-49.5, 49.5)
        self._other_e_min.setSingleStep(0.5)
        self._other_e_min.setDecimals(1)
        self._other_e_min.setValue(0.0)
        self._other_e_min.setFixedWidth(65)
        gr.addWidget(self._other_e_min)
        gr.addWidget(QLabel(","))
        self._other_e_max = QDoubleSpinBox()
        self._other_e_max.setRange(-49.5, 49.5)
        self._other_e_max.setSingleStep(0.5)
        self._other_e_max.setDecimals(1)
        self._other_e_max.setValue(0.5)
        self._other_e_max.setFixedWidth(65)
        gr.addWidget(self._other_e_max)
        gr.addWidget(QLabel("]"))
        gr.addStretch(1)
        other_layout.addWidget(self._other_grid_row)
        self._other_grid_row.setVisible(True)  # grid is default

        self._other_box.setVisible(False)   # hidden for single-cusp manifolds
        fill_layout.addWidget(self._other_box)

        # ── Multi-cusp: per-cusp NC cycle + slope inputs ──────────────────────
        self._cusp_fill_container = QWidget()
        self._cusp_fill_layout = QVBoxLayout(self._cusp_fill_container)
        self._cusp_fill_layout.setContentsMargins(0, 0, 0, 0)
        self._cusp_fill_layout.setSpacing(8)
        self._cusp_fill_container.setVisible(False)
        fill_layout.addWidget(self._cusp_fill_container)

        fill_btn_row = QHBoxLayout()
        self._fill_btn = QPushButton("Compute Filling")
        self._fill_btn.setProperty("class", "primary")
        self._fill_btn.setEnabled(False)
        self._fill_btn.clicked.connect(self._on_fill_clicked)
        fill_btn_row.addWidget(self._fill_btn)
        self._fill_stop_btn = QPushButton("Stop")
        self._fill_stop_btn.setProperty("class", "secondary")
        self._fill_stop_btn.setEnabled(False)
        self._fill_stop_btn.clicked.connect(self._on_fill_stop_clicked)
        fill_btn_row.addWidget(self._fill_stop_btn)
        fill_btn_row.addStretch(1)
        fill_layout.addLayout(fill_btn_row)

        self._fill_progress = QProgressBar()
        self._fill_progress.setRange(0, 0)   # indeterminate
        self._fill_progress.setVisible(False)
        fill_layout.addWidget(self._fill_progress)

        self._fill_grid_bar = QProgressBar()
        self._fill_grid_bar.setTextVisible(True)
        self._fill_grid_bar.setFixedHeight(8)
        self._fill_grid_bar.setVisible(False)
        fill_layout.addWidget(self._fill_grid_bar)

        # Inline auto-Weyl status label (after fill progress bar)
        self._auto_weyl_status = QLabel()
        self._auto_weyl_status.setProperty("class", "muted")
        self._auto_weyl_status.setVisible(False)
        fill_layout.addWidget(self._auto_weyl_status)

        self._fill_status = QLabel()
        self._fill_status.setProperty("class", "muted")
        self._fill_status.setVisible(False)
        fill_layout.addWidget(self._fill_status)


        bl.addWidget(fill_box)

        # ── Filling results table ─────────────────────────────────────
        result_header = QHBoxLayout()
        result_label = QLabel("Results")
        result_label.setProperty("class", "secondary")
        result_header.addWidget(result_label)
        result_header.addStretch(1)
        self._clear_results_btn = QPushButton("Clear")
        self._clear_results_btn.setProperty("class", "secondary")
        self._clear_results_btn.clicked.connect(self._on_clear_results)
        result_header.addWidget(self._clear_results_btn)
        bl.addLayout(result_header)

        # ── Refinement toggle for filled results ──────────────────────
        self._fill_ref_box = QGroupBox("Refinement")
        fill_ref_layout = QVBoxLayout(self._fill_ref_box)
        self._fill_edge_container = QWidget()
        fill_edge_layout = QHBoxLayout(self._fill_edge_container)
        fill_edge_layout.setContentsMargins(0, 0, 0, 0)
        self._fill_edge_layout = fill_edge_layout
        fill_ref_layout.addWidget(self._fill_edge_container)
        self._fill_cusp_container = QWidget()
        fill_cusp_layout = QHBoxLayout(self._fill_cusp_container)
        fill_cusp_layout.setContentsMargins(0, 0, 0, 0)
        self._fill_cusp_layout = fill_cusp_layout
        self._fill_cusp_container.setVisible(False)
        fill_ref_layout.addWidget(self._fill_cusp_container)
        self._fill_ref_box.setVisible(False)
        bl.addWidget(self._fill_ref_box)

        self._fill_refresh_timer = QTimer(self)
        self._fill_refresh_timer.setSingleShot(True)
        self._fill_refresh_timer.setInterval(300)
        self._fill_refresh_timer.timeout.connect(self._do_refresh_fill_display)

        # ── Fill info panel (NC cycle, transformed slope, HJ k-vector) ─────
        self._fill_info_view = MathView(min_h=200, auto_fit=True)
        self._fill_info_view.setVisible(False)
        bl.addWidget(self._fill_info_view)

        self._fill_table = SeriesTable()
        self._fill_table.setMinimumHeight(100)
        bl.addWidget(self._fill_table)

        self._card.set_body(body)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def unlock(self, session: Session) -> None:
        self._session = session
        self._session_gen += 1
        self._card.set_status(CardStatus.READY)
        self._card.expand()
        # Populate the cusp combo for Phase B
        n_cusps = session.nz_data.r if session.nz_data is not None else 1

        if n_cusps > 1:
            # Multi-cusp: hide single-cusp UI, show multi-cusp container
            self._cusp_combo.setVisible(False)
            self._nc_combo.setVisible(False)
            self._fill_slope.setVisible(False)
            self._other_box.setVisible(True)
            # Only rebuild rows when the cusp count changes (not on every unlock)
            current_n = len(self._cusp_fill_rows)
            if current_n != n_cusps:
                self._rebuild_fill_cusp_rows(n_cusps)
            self._cusp_fill_container.setVisible(True)
        else:
            # Single-cusp: show single-cusp UI
            self._cusp_combo.setVisible(True)
            self._nc_combo.setVisible(True)
            self._fill_slope.setVisible(True)
            self._other_box.setVisible(False)
            self._cusp_fill_container.setVisible(False)
            # Populate cusp combo
            self._cusp_combo.clear()
            if n_cusps == 1:
                self._cusp_combo.addItem("C0", 0)
        # Rebuild NC tables
        self._rebuild_nc_tables(n_cusps)
        self._cache_chk.setChecked(
            session.cache_status.get("nc", {}).get("available", False)
        )
        # Rebuild edge-toggle checkboxes for filled results
        n_hard = session.num_hard()
        self._fill_ref_box.setVisible(n_hard > 0)
        self._rebuild_fill_edge_toggles(n_hard)

    def lock(self) -> None:
        # Disconnect and abandon any running workers so their stale signals
        # cannot corrupt the next session's state.
        self._abandon_nc_workers()
        self._abandon_fill_workers()
        # Abandon auto-Weyl workers (per-cycle NC and pre-fill)
        for worker in self._nc_weyl_workers.values():
            try:
                worker.finished.disconnect()
                worker.error.disconnect()
            except RuntimeError:
                pass
        self._nc_weyl_workers.clear()
        self._nc_weyl_results.clear()

        if self._auto_weyl_worker is not None:
            try:
                self._auto_weyl_worker.finished.disconnect()
                self._auto_weyl_worker.error.disconnect()
            except RuntimeError:
                pass
            self._auto_weyl_worker = None
        self._card.set_status(CardStatus.LOCKED)
        self._fill_table.clear_rows()
        for mv in self._nc_table_views:
            mv.setVisible(False)
        for lbl in self._nc_table_labels:
            lbl.setVisible(False)
        self._nc_cycle_vms.clear()
        # Reset multi-cusp fill rows
        for row in self._cusp_fill_rows:
            if row.get("widget"):
                row["widget"].hide()
                row["widget"].setParent(None)  # type: ignore[arg-type]
        self._cusp_fill_rows.clear()
        self._fill_btn.setEnabled(False)
        # Reset refinement toggles
        self._fill_refresh_timer.stop()
        self._fill_ref_box.setVisible(False)
        for cb in self._fill_edge_checkboxes:
            cb.setParent(None)  # type: ignore[arg-type]
        self._fill_edge_checkboxes.clear()
        for cb in self._fill_cusp_checkboxes:
            cb.setParent(None)  # type: ignore[arg-type]
        self._fill_cusp_checkboxes.clear()
        self._fill_cusp_container.setVisible(False)
        # Reset auto-Weyl UI
        if self._auto_weyl_status is not None:
            self._auto_weyl_status.setVisible(False)
        # Reset NC UI
        self._nc_search_btn.setEnabled(True)
        self._nc_stop_btn.setEnabled(False)
        self._nc_progress.setVisible(False)
        self._nc_status.setVisible(False)
        # Reset Fill UI
        self._fill_info_view.setVisible(False)
        self._fill_stop_btn.setEnabled(False)
        self._fill_progress.setVisible(False)
        self._fill_grid_bar.setVisible(False)
        self._fill_status.setVisible(False)
        # Reset tracking state
        self._nc_worker_progress.clear()
        self._fill_current_row = None
        self._fill_grid_total = 0
        self._fill_grid_done = 0

    def refresh(self, session: Session) -> None:
        self._session = session
        if session.nc_cycles:
            self._rebuild_nc_vms_from_session()
            self._render_nc_tables()
            self._rebuild_nc_combo()
            self._fill_btn.setEnabled(True)
        # Rebuild V_i toggles if any existing fill results have cusp eta
        all_results = (
            [fq.result for fq in session.fill_queries if fq.result is not None]
            + [mfq.result for mfq in session.multi_fill_queries if mfq.result is not None]
        )
        for r in all_results:
            if r.has_cusp_eta and r.num_cusp_eta > 0:
                if len(self._fill_cusp_checkboxes) != r.num_cusp_eta:
                    self._rebuild_cusp_eta_toggles(r.num_cusp_eta)
                break
        for fq in session.fill_queries:
            self._show_fill_query(fq)
        for mfq in session.multi_fill_queries:
            self._show_multi_fill_query(mfq)

    def trigger_find_nc(self) -> None:
        """Public: programmatically trigger NC cycle search (used by Run All).

        Always tries the disk cache first so Run All is as fast as a manual
        run where the user has already ticked "Use cache".
        """
        # Temporarily force the cache checkbox on so _on_find_nc_clicked
        # picks it up — it will be restored to the session-driven value the
        # next time unlock() is called.
        was_checked = self._cache_chk.isChecked()
        self._cache_chk.setChecked(True)
        self._on_find_nc_clicked()
        # Don't restore yet — let the async worker run; unlock() will reset it
        # when the next manifold is loaded.

    # ------------------------------------------------------------------
    # Internal — NC cycle search
    # ------------------------------------------------------------------

    def _abandon_nc_workers(self) -> None:
        """Signal all running NC workers to stop."""
        for w in self._nc_workers:
            try:
                w.finished.disconnect()
                w.error.disconnect()
                w.progress.disconnect()
            except RuntimeError:
                pass
            w.requestInterruption()
        self._nc_workers.clear()

    def _abandon_fill_workers(self) -> None:
        """Signal all running fill workers to stop."""
        for w in self._fill_workers:
            try:
                w.finished.disconnect()
                w.error.disconnect()
            except RuntimeError:
                pass
            w.requestInterruption()
        self._fill_workers.clear()

    def _rebuild_nc_vms_from_session(self) -> None:
        """Reconstruct NC cycle ViewModels from session data."""
        self._nc_cycle_vms.clear()
        s = self._session
        for ncs in s.nc_cycles:
            for cyc in ncs.cycles:
                P = int(cyc.P) if hasattr(cyc, "P") else 0
                Q = int(cyc.Q) if hasattr(cyc, "Q") else 0
                try:
                    sl = format_slope_latex(P, Q)
                except Exception:
                    sl = f"({P},{Q})"
                vm = build_nc_cycle_vm(
                    cusp_idx=ncs.cusp_idx,
                    P=P,
                    Q=Q,
                    weyl_compatible=None if s.weyl_result is None else True,
                    adjoint_proj_pass=s.weyl_adjoint_pass,
                    source=ncs.source,
                    slope_latex=sl,
                )
                self._nc_cycle_vms.append(vm)

    def _rebuild_nc_tables(self, n_cusps: int) -> None:
        """Rebuild NC cycle tables (one per cusp)."""
        # Clear old views
        for mv in self._nc_table_views:
            mv.setParent(None)  # type: ignore[arg-type]
        for lbl in self._nc_table_labels:
            lbl.setParent(None)  # type: ignore[arg-type]
        self._nc_table_views.clear()
        self._nc_table_labels.clear()

        # Clear layout
        while self._nc_table_layout.count():
            item = self._nc_table_layout.takeAt(0)
            if item and item.widget():
                item.widget().setParent(None)  # type: ignore[arg-type]

    def _render_nc_tables(self) -> None:
        """Render NC cycles (grouped by cusp) in the NC table views."""
        # Clear existing views before re-rendering to avoid duplicates
        for mv in self._nc_table_views:
            mv.setParent(None)  # type: ignore[arg-type]
        for lbl in self._nc_table_labels:
            lbl.setParent(None)  # type: ignore[arg-type]
        self._nc_table_views.clear()
        self._nc_table_labels.clear()
        while self._nc_table_layout.count():
            item = self._nc_table_layout.takeAt(0)
            if item and item.widget():
                item.widget().setParent(None)  # type: ignore[arg-type]

        s = self._session
        print(f"[NC-RENDER] Rendering {len(self._nc_cycle_vms)} total cycles")
        # Group cycles by cusp
        by_cusp: dict[int, list] = {}
        for vm in self._nc_cycle_vms:
            if vm.cusp_idx not in by_cusp:
                by_cusp[vm.cusp_idx] = []
            by_cusp[vm.cusp_idx].append(vm)
        print(f"[NC-RENDER] Found cycles for cusps: {sorted(by_cusp.keys())}")

        # Render one table per cusp
        # (per-cycle Weyl vectors are now in each VM, not global)
        for cusp_idx in sorted(by_cusp.keys()):
            vms = by_cusp[cusp_idx]
            html = format_nc_cycle_table_html(vms)

            lbl = QLabel(f"Cusp C{cusp_idx}")
            lbl.setProperty("class", "secondary")
            self._nc_table_layout.addWidget(lbl)
            self._nc_table_labels.append(lbl)

            mv = MathView(min_h=80, auto_fit=True)
            mv.set_html(html)
            self._nc_table_layout.addWidget(mv)
            self._nc_table_views.append(mv)

    def _rebuild_nc_combo(self) -> None:
        """Rebuild NC cycle combo with all loaded cycles."""
        self._nc_combo.blockSignals(True)
        self._nc_combo.clear()
        for i, vm in enumerate(self._nc_cycle_vms):
            slope_label = f"({vm.P},{vm.Q})"
            self._nc_combo.addItem(slope_label, i)
        self._nc_combo.blockSignals(False)
        # Also populate per-cusp combos for multi-cusp UI
        self._rebuild_fill_cusp_nc_combos()

    def _rebuild_fill_cusp_nc_combos(self) -> None:
        """Populate per-cusp NC cycle combos with cycles for that cusp."""
        # Group cycles by cusp
        by_cusp: dict[int, list] = {}
        for i, vm in enumerate(self._nc_cycle_vms):
            if vm.cusp_idx not in by_cusp:
                by_cusp[vm.cusp_idx] = []
            by_cusp[vm.cusp_idx].append((i, vm))

        # Populate each per-cusp combo and suggest an initial fill slope
        for row in self._cusp_fill_rows:
            cusp_idx = row["cusp_idx"]
            combo = row["nc_combo"]
            slope_widget = row["slope"]
            combo.blockSignals(True)
            combo.clear()
            first_vm = None
            if cusp_idx in by_cusp:
                for idx, vm in by_cusp[cusp_idx]:
                    slope_label = f"({vm.P},{vm.Q})"
                    combo.addItem(slope_label, idx)
                    if first_vm is None:
                        first_vm = vm
            combo.blockSignals(False)
            # Connect NC selection change to suggest a fill slope ≠ NC cycle
            try:
                combo.currentIndexChanged.disconnect()
            except RuntimeError:
                pass
            combo.currentIndexChanged.connect(
                lambda _idx, c=combo, sw=slope_widget: self._on_cusp_nc_selected(c, sw)
            )
            # Suggest fill slope for the initially selected NC cycle
            if first_vm is not None:
                self._suggest_fill_slope_for_widget(slope_widget, first_vm)

    def _on_find_nc_clicked(self) -> None:
        s = self._session
        if s.nz_data is None:
            return
        n_cusps = s.nz_data.r if s.nz_data is not None else 1
        print(f"[NC-SEARCH] Starting NC search for manifold with {n_cusps} cusps")

        p_half = self._p_range_spin.value()
        q_half = self._q_range_spin.value()
        p_range = (-p_half, p_half)
        q_range = (0, q_half)

        self._nc_search_btn.setEnabled(False)
        self._nc_stop_btn.setEnabled(True)
        self._nc_progress.setRange(0, 0)   # indeterminate
        self._nc_progress.setVisible(True)
        self._nc_status.setText("Searching for NC cycles…")
        self._nc_status.setVisible(True)
        self._card.set_status(CardStatus.RUNNING)

        # Clear old results before starting new search
        self._nc_cycle_vms.clear()
        self._rebuild_nc_tables(n_cusps)  # Properly remove old table widgets from layout
        self._nc_combo.clear()  # Clear the combo box
        self._fill_btn.setEnabled(False)  # Disable fill button until new results arrive

        gen = self._session_gen

        for i in range(n_cusps):
            print(f"[NC-SEARCH] Creating worker for cusp {i}")
            worker = NCSearchWorker(
                nz_data      = s.nz_data,
                cusp_idx     = i,
                p_range      = p_range,
                q_range      = q_range,
                use_cache    = self._cache_chk.isChecked(),
                q_order_half = s.q_order_half,
                manifold_name = s.manifold_name,
                parent       = self,
            )
            worker.finished.connect(
                lambda p, ci=i, g=gen: self._on_nc_finished(p, ci, g)
            )
            worker.progress.connect(
                lambda d, t, ci=i: self._on_nc_progress(d, t, ci)
            )
            worker.error.connect(
                lambda e, w=worker, ci=i, g=gen: self._on_nc_error(e, ci, g, w)
            )
            self._nc_workers.append(worker)
            worker.start()
            print(f"[NC-SEARCH] Worker {i} started")

    def _on_nc_finished(self, payload: dict, cusp_idx: int, gen: int, worker: object = None) -> None:
        print(f"[NC-SEARCH] Finished for cusp {cusp_idx}: {len(payload.get('cycles', []))} cycles found")
        if gen != self._session_gen:
            return   # stale: a new manifold was loaded since this worker launched
        # Remove the completed worker from the tracking list
        w = worker if worker is not None else self.sender()
        if w in self._nc_workers:
            self._nc_workers.remove(w)

        cycles = payload["cycles"]
        s = self._session
        print(f"[NC-SEARCH] Processing {len(cycles)} cycles for cusp {cusp_idx}")

        ncs = NCCycleSet(
            cusp_idx       = cusp_idx,
            search_p_range = (-self._p_range_spin.value(), self._p_range_spin.value()),
            search_q_range = (0, self._q_range_spin.value()),
            q_order_half   = s.q_order_half,
            cycles         = cycles,
            source         = "cache" if payload.get("nc_result") is None else "computed",
        )
        s.nc_cycles = [nc for nc in s.nc_cycles if nc.cusp_idx != cusp_idx]
        s.nc_cycles.append(ncs)

        for cyc in cycles:
            P = int(cyc.P) if hasattr(cyc, "P") else 0
            Q = int(cyc.Q) if hasattr(cyc, "Q") else 0
            try:
                sl = format_slope_latex(P, Q)
            except Exception:
                sl = f"({P},{Q})"
            vm = build_nc_cycle_vm(
                cusp_idx=cusp_idx, P=P, Q=Q,
                weyl_compatible=None if s.weyl_result is None else True,
                adjoint_proj_pass=s.weyl_adjoint_pass,
                source=ncs.source,
                slope_latex=sl,
            )
            self._nc_cycle_vms.append(vm)

        running = [w for w in self._nc_workers if w.isRunning()]
        if not running:
            self._nc_search_btn.setEnabled(True)
            self._nc_stop_btn.setEnabled(False)
            self._nc_progress.setVisible(False)
            self._nc_status.setVisible(False)
            self._nc_workers.clear()
            self._rebuild_nc_combo()
            # Always enable fill button, even if no NC cycles found
            # (meridian basis (1,0) fallback will be used)
            self._fill_btn.setEnabled(True)
            if self._nc_cycle_vms:
                self._card.set_status(CardStatus.DONE)
            else:
                self._card.set_status(CardStatus.READY)
            self._update_summary()
            # Render tables immediately so results are visible right away.
            # Weyl check runs in background and refreshes tables with
            # compatibility info when complete.
            self._render_nc_tables()
            self.session_updated.emit(s)
            self._launch_weyl_for_nc_compatibility()

    def _on_nc_error(self, msg: str, cusp_idx: int, gen: int, worker: object = None) -> None:
        if gen != self._session_gen:
            return
        w = worker if worker is not None else self.sender()
        if w in self._nc_workers:
            self._nc_workers.remove(w)
        # Only re-enable controls if no other NC workers are still running
        if not any(w.isRunning() for w in self._nc_workers):
            self._nc_search_btn.setEnabled(True)
            self._nc_stop_btn.setEnabled(False)
            self._nc_progress.setVisible(False)
            self._nc_status.setVisible(False)
            self._nc_workers.clear()
            self._rebuild_nc_combo()
            if self._nc_cycle_vms:
                self._fill_btn.setEnabled(True)
                self._update_summary()
                self._render_nc_tables()
                self.session_updated.emit(self._session)
                self._launch_weyl_for_nc_compatibility()
        logging.warning("NCSearchWorker cusp %d error: %s", cusp_idx, msg)
        self._card.set_status(CardStatus.ERROR)

    def _on_nc_stop_clicked(self) -> None:
        """Abandon all running NC search workers."""
        self._abandon_nc_workers()
        self._nc_worker_progress.clear()
        self._nc_search_btn.setEnabled(True)
        self._nc_stop_btn.setEnabled(False)
        self._nc_progress.setVisible(False)
        self._nc_status.setText("Stopped.")
        self._nc_status.setVisible(True)
        self._card.set_status(CardStatus.READY)

    def _on_nc_progress(self, done: int, total: int, cusp_idx: int) -> None:
        """Aggregate progress from all cusp workers into one bar."""
        self._nc_worker_progress[cusp_idx] = (done, total)
        agg_done  = sum(d for d, _ in self._nc_worker_progress.values())
        agg_total = sum(t for _, t in self._nc_worker_progress.values())
        if agg_total > 0:
            self._nc_progress.setRange(0, agg_total)
            self._nc_progress.setValue(agg_done)
        self._nc_status.setText(f"Tested {agg_done} / {agg_total} slopes…")

    def _launch_weyl_for_nc_compatibility(self) -> None:
        """After NC search, run Weyl check for EACH NC cycle with basis change.

        NcCompatWorker always computes its own (a,b) probe grid after the
        basis change, so it does not require pre-existing index queries.
        Any available user-grid entries are passed as supplementary data.
        """
        s = self._session
        if not self._nc_cycle_vms:
            return

        # Clear previous results
        self._nc_weyl_results.clear()
        self._nc_weyl_workers.clear()

        print(f"[WEYL-NC] Launching Weyl checks for {len(self._nc_cycle_vms)} NC cycles")

        gen = self._session_gen

        # Launch one NcCompatWorker per NC cycle — basis change, index
        # recomputation and Weyl check all happen off the main thread.
        for vm in self._nc_cycle_vms:
            P, Q     = vm.P, vm.Q
            cusp_idx = vm.cusp_idx
            cycle_key = (P, Q)
            worker = NcCompatWorker(
                nz_data      = s.nz_data,
                P            = P,
                Q            = Q,
                cusp_idx     = cusp_idx,
                index_queries = list(s.index_queries),
                num_hard     = s.num_hard(),
                q_order_half = s.q_order_half,
                parent       = self,
            )
            worker.finished.connect(
                lambda p, ck=cycle_key, g=gen: self._on_weyl_for_nc_cycle_done(p, ck, g)
            )
            worker.error.connect(
                lambda e, ck=cycle_key, g=gen: self._on_weyl_for_nc_cycle_error(e, ck, g)
            )
            self._nc_weyl_workers[cycle_key] = worker
            worker.start()

    def _on_weyl_for_nc_cycle_done(self, payload: dict, cycle_key: tuple, gen: int) -> None:
        """Handle Weyl check completion for one NC cycle."""
        if gen != self._session_gen:
            print(f"[WEYL-NC] Ignoring stale generation for cycle {cycle_key}")
            return

        P, Q = cycle_key
        if cycle_key in self._nc_weyl_workers:
            del self._nc_weyl_workers[cycle_key]

        ab           = payload["ab_vectors"]
        adj_pass     = payload.get("adjoint_is_pass")
        adj_value    = payload.get("adjoint_value")
        is_marginal  = payload.get("is_marginal")
        unrefined_q1_proj = payload.get("unrefined_q1_proj")

        print(
            f"[WEYL-NC] Cycle ({P},{Q}) done: ab={'present' if ab else 'None'}, "
            f"adj_pass={adj_pass}, adj_value={adj_value}, "
            f"is_marginal={is_marginal}, unrefined_q1_proj={unrefined_q1_proj}"
        )

        # Store per-cycle result
        self._nc_weyl_results[cycle_key] = {
            "ab":                   ab,
            "adj_pass":             adj_pass,
            "adj_value":            adj_value,
            "is_marginal":          is_marginal,
            "unrefined_q1_proj": unrefined_q1_proj,
        }

        # Check if all cycles are done
        if not self._nc_weyl_workers:
            print("[WEYL-NC] All Weyl checks complete, updating NC table")
            self._rebuild_nc_vm_compatibility()
            self._render_nc_tables()
            self.session_updated.emit(self._session)

    def _on_weyl_for_nc_cycle_error(self, msg: str, cycle_key: tuple, gen: int) -> None:
        """Handle Weyl check error for one NC cycle."""
        if gen != self._session_gen:
            return

        P, Q = cycle_key
        if cycle_key in self._nc_weyl_workers:
            del self._nc_weyl_workers[cycle_key]

        print(f"[WEYL-NC] Cycle ({P},{Q}) error: {msg}")

        # Store error result
        self._nc_weyl_results[cycle_key] = {
            "error": msg,
            "ab": None,
            "adj_pass": None,
        }

        # Check if all cycles are done
        if not self._nc_weyl_workers:
            print("[WEYL-NC] All Weyl checks complete, updating NC table")
            self._rebuild_nc_vm_compatibility()
            self._render_nc_tables()
            self.session_updated.emit(self._session)

    def _rebuild_nc_vm_compatibility(self) -> None:
        """Update NC cycle VMs with per-cycle Weyl symmetry results.

        The q¹ adjoint projection is NOT computed here — it requires the full
        Dehn filling configuration (all cusps + chosen NC cycles) and is only
        available after the user clicks "Compute Filling".
        """
        for vm in self._nc_cycle_vms:
            cycle_key = (vm.P, vm.Q)
            if cycle_key in self._nc_weyl_results:
                result = self._nc_weyl_results[cycle_key]
                if "error" in result:
                    vm.weyl_compatible      = None
                    vm.adjoint_proj_pass    = None
                    vm.adjoint_proj_value   = None
                    vm.weyl_a               = None
                    vm.weyl_b               = None
                    vm.is_marginal          = None
                    vm.unrefined_q1_proj = None
                else:
                    ab = result.get("ab")
                    vm.weyl_compatible    = None if ab is None else True
                    # Refined q^1 — controls W_i compatibility.
                    vm.adjoint_proj_pass  = result.get("adj_pass")
                    vm.adjoint_proj_value = result.get("adj_value")
                    if ab is not None:
                        vm.weyl_a = list(ab.a) if hasattr(ab, 'a') else None
                        vm.weyl_b = list(ab.b) if hasattr(ab, 'b') else None
                    else:
                        vm.weyl_a = None
                        vm.weyl_b = None
                    # Strongly-NC (unrefined q^1) — controls kernel choice.
                    vm.is_marginal          = result.get("is_marginal")
                    vm.unrefined_q1_proj = result.get("unrefined_q1_proj")
            else:
                vm.weyl_compatible      = None
                vm.adjoint_proj_pass    = None
                vm.adjoint_proj_value   = None
                vm.weyl_a               = None
                vm.weyl_b               = None
                vm.is_marginal          = None
                vm.unrefined_q1_proj = None

    # ------------------------------------------------------------------
    # Internal — Fill stop
    # ------------------------------------------------------------------

    def _on_fill_stop_clicked(self) -> None:
        """Abandon all running fill workers."""
        if self._fill_current_row is not None:
            self._fill_table.set_row_result(self._fill_current_row, "Stopped", "—")
            self._fill_current_row = None
        self._abandon_fill_workers()
        self._fill_btn.setEnabled(True)
        self._fill_stop_btn.setEnabled(False)
        self._fill_progress.setVisible(False)
        self._fill_grid_bar.setVisible(False)
        self._fill_status.setVisible(False)
        self._auto_weyl_status.setVisible(False)
        self._fill_grid_total = 0
        self._fill_grid_done = 0
        self._card.set_status(CardStatus.READY)

    def _on_clear_results(self) -> None:
        """Clear all rows from the filling results table and session fill queries."""
        self._fill_table.clear_rows()
        if self._session is not None:
            self._session.fill_queries.clear()
            self._update_summary()

    def _rebuild_fill_cusp_rows(self, n_cusps: int) -> None:
        """Create per-cusp rows; each row uses a QStackedWidget to swap between
        fill mode (page 0: NC combo + slope) and charge mode (page 1: m/e)."""
        for row in self._cusp_fill_rows:
            if row.get("widget"):
                row["widget"].hide()                       # prevent top-level flash
                row["widget"].setParent(None)  # type: ignore[arg-type]
        self._cusp_fill_rows.clear()

        for cusp_idx in range(n_cusps):
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)

            fill_chk = QCheckBox("Fill")
            fill_chk.setChecked(True)
            fill_chk.setFixedWidth(50)
            row_layout.addWidget(fill_chk)

            row_layout.addWidget(QLabel(f"C{cusp_idx}"))

            # Stack: page 0 = fill panel, page 1 = charge panel
            stack = QStackedWidget()

            # Page 0 — fill: NC combo + slope
            fill_panel = QWidget()
            fp = QHBoxLayout(fill_panel)
            fp.setContentsMargins(0, 0, 0, 0)
            fp.setSpacing(8)
            nc_combo = QComboBox()
            nc_combo.setMinimumWidth(100)
            fp.addWidget(nc_combo)
            slope = SlopeInput(label="→", require_coprime=True)
            fp.addWidget(slope)
            fp.addStretch(1)
            stack.addWidget(fill_panel)   # index 0

            # Page 1 — charge: per-cusp m and e
            charge_panel = QWidget()
            cp = QHBoxLayout(charge_panel)
            cp.setContentsMargins(0, 0, 0, 0)
            cp.setSpacing(4)
            cp.addWidget(QLabel("m ="))
            m_spin = QSpinBox()
            m_spin.setRange(-99, 99)
            m_spin.setValue(0)
            m_spin.setFixedWidth(60)
            cp.addWidget(m_spin)
            cp.addWidget(QLabel("  e ="))
            e_spin = QDoubleSpinBox()
            e_spin.setRange(-49.5, 49.5)
            e_spin.setSingleStep(0.5)
            e_spin.setDecimals(1)
            e_spin.setValue(0.0)
            e_spin.setFixedWidth(70)
            cp.addWidget(e_spin)
            cp.addStretch(1)
            stack.addWidget(charge_panel)  # index 1

            row_layout.addWidget(stack)
            self._cusp_fill_layout.addWidget(row_widget)

            row_dict = {
                "widget":   row_widget,
                "cusp_idx": cusp_idx,
                "fill_chk": fill_chk,
                "stack":    stack,
                "nc_combo": nc_combo,
                "slope":    slope,
                "m_spin":   m_spin,
                "e_spin":   e_spin,
            }
            self._cusp_fill_rows.append(row_dict)

            fill_chk.toggled.connect(
                lambda checked, rd=row_dict: self._update_cusp_row_stack(rd)
            )
            self._update_cusp_row_stack(row_dict)

    def _rebuild_fill_edge_toggles(self, n_hard: int) -> None:
        """Create per-edge W checkboxes; called from unlock()."""
        for cb in self._fill_edge_checkboxes:
            cb.setParent(None)  # type: ignore[arg-type]
        self._fill_edge_checkboxes.clear()
        for j in range(n_hard):
            sub = chr(0x2080 + j) if j <= 9 else f"_{j}"
            cb = QCheckBox(f"W{sub}")
            cb.setChecked(True)
            cb.stateChanged.connect(self._on_fill_edge_toggle)
            self._fill_edge_layout.addWidget(cb)
            self._fill_edge_checkboxes.append(cb)

    def _reset_fill_edge_toggles(self) -> None:
        """Re-enable and check all W checkboxes (called before each new fill)."""
        for cb in self._fill_edge_checkboxes:
            cb.blockSignals(True)
            cb.setChecked(True)
            cb.setEnabled(True)
            cb.setToolTip("")
            cb.blockSignals(False)

    def _rebuild_cusp_eta_toggles(self, n_cusp_eta: int) -> None:
        """Create per-cusp V checkboxes; called when first cusp-eta result arrives."""
        for cb in self._fill_cusp_checkboxes:
            cb.setParent(None)  # type: ignore[arg-type]
        self._fill_cusp_checkboxes.clear()
        for ci in range(n_cusp_eta):
            sub = chr(0x2080 + ci) if ci <= 9 else f"_{ci}"
            cb = QCheckBox(f"V{sub}")
            cb.setChecked(True)
            cb.stateChanged.connect(self._on_fill_cusp_toggle)
            self._fill_cusp_layout.addWidget(cb)
            self._fill_cusp_checkboxes.append(cb)
        self._fill_cusp_container.setVisible(n_cusp_eta > 0)
        if n_cusp_eta > 0:
            self._fill_ref_box.setVisible(True)

    def _fill_active_edges(self) -> list[bool]:
        return [cb.isChecked() for cb in self._fill_edge_checkboxes]

    def _fill_active_cusp_etas(self) -> list[bool]:
        return [cb.isChecked() for cb in self._fill_cusp_checkboxes]

    def _has_any_fill_results(self) -> bool:
        return bool(
            self._session
            and (self._session.fill_queries or self._session.multi_fill_queries)
        )

    def _on_fill_edge_toggle(self) -> None:
        """Called when user manually clicks an individual W checkbox."""
        if not self._has_any_fill_results():
            return
        self._fill_refresh_timer.start()

    def _on_fill_cusp_toggle(self) -> None:
        """Called when user manually clicks an individual V checkbox."""
        if not self._has_any_fill_results():
            return
        self._fill_refresh_timer.start()

    def _do_refresh_fill_display(self) -> None:
        """Re-render all fill rows with the current edge-toggle projection.

        Row order matches the session-loading order: fill_queries first,
        then multi_fill_queries (see unlock()).
        """
        if not self._session:
            return
        inactive_w = [j for j, a in enumerate(self._fill_active_edges()) if not a]
        inactive_v = [ci for ci, a in enumerate(self._fill_active_cusp_etas()) if not a]
        seq = 0  # fallback row for queries without stored row_index

        for fq in self._session.fill_queries:
            row = fq.row_index if fq.row_index >= 0 else seq
            if fq.result is not None:
                try:
                    projected = fq.result.collapse_eta_edges(inactive_w)
                    projected = projected.collapse_cusp_etas(inactive_v)
                    if fq.unrefined_fallback:
                        latex = format_unrefined_series_latex(projected.series)
                    else:
                        latex = format_filled_series_latex(
                            projected.series, projected.num_hard,
                            projected.has_cusp_eta, projected.num_cusp_eta,
                        )
                except Exception:
                    latex = "—"
                self._fill_table.set_row_result(row, latex, fq.source)
                if (seq + 1) % 50 == 0:
                    QCoreApplication.processEvents()
            seq += 1

        for mfq in self._session.multi_fill_queries:
            row = mfq.row_index if mfq.row_index >= 0 else seq
            if mfq.result is not None:
                try:
                    projected = mfq.result.collapse_eta_edges(inactive_w)
                    projected = projected.collapse_cusp_etas(inactive_v)
                    latex = format_filled_series_latex(
                        projected.series, projected.num_hard,
                        projected.has_cusp_eta, projected.num_cusp_eta,
                    )
                except Exception:
                    latex = "—"
                self._fill_table.set_row_result(row, latex, mfq.source)
                if (seq + 1) % 50 == 0:
                    QCoreApplication.processEvents()
            seq += 1

    def _update_summary(self) -> None:
        n_nc   = len(self._nc_cycle_vms)
        n_fill = len(self._session.fill_queries)
        weyl   = "✓" if self._session.weyl_checked else "not run"
        self._card.set_summary(
            f"Weyl: {weyl}  ·  {n_nc} NC cycle{'s' if n_nc != 1 else ''}  ·  "
            f"{n_fill} filled quer{'ies' if n_fill != 1 else 'y'}"
        )

    def _suggest_fill_slope(self, vm: "NCCycleViewModel") -> None:
        """Set the fill-slope widget to the nearest primitive slope ≠ NC cycle.

        Strategy: try (nc_P, nc_Q + 1), then (nc_P + 1, nc_Q), then
        (nc_P + 1, nc_Q + 1), stepping until a coprime pair is found that
        is not equal to (nc_P, nc_Q).
        """
        from math import gcd as _gcd
        nc_P, nc_Q = vm.P, vm.Q
        candidates = [
            (nc_P,     nc_Q + 1),
            (nc_P + 1, nc_Q),
            (nc_P + 1, nc_Q + 1),
            (nc_P - 1, nc_Q),
            (nc_P,     nc_Q - 1),
        ]
        for p, q in candidates:
            if (p, q) != (nc_P, nc_Q) and _gcd(abs(p), abs(q)) == 1:
                self._fill_slope.set_slope(p, q)
                return
        # Fallback: just nudge P by 2
        self._fill_slope.set_slope(nc_P + 2, nc_Q + 1)

    def _suggest_fill_slope_for_widget(self, slope_widget: "SlopeInput", vm: "NCCycleViewModel") -> None:
        """Suggest a fill slope for a given SlopeInput widget."""
        from math import gcd as _gcd
        nc_P, nc_Q = vm.P, vm.Q
        candidates = [
            (nc_P,     nc_Q + 1),
            (nc_P + 1, nc_Q),
            (nc_P + 1, nc_Q + 1),
            (nc_P - 1, nc_Q),
            (nc_P,     nc_Q - 1),
        ]
        for p, q in candidates:
            if (p, q) != (nc_P, nc_Q) and _gcd(abs(p), abs(q)) == 1:
                slope_widget.set_slope(p, q)
                return
        # Fallback: just nudge P by 2
        slope_widget.set_slope(nc_P + 2, nc_Q + 1)

    def _on_nc_selected(self, idx: int) -> None:
        if 0 <= idx < len(self._nc_cycle_vms):
            vm = self._nc_cycle_vms[idx]
            # Sync the cusp combo to the selected NC cycle's cusp.
            cusp_i = self._cusp_combo.findData(vm.cusp_idx)
            if cusp_i >= 0:
                self._cusp_combo.setCurrentIndex(cusp_i)
            # Do NOT auto-suggest a fill slope — let the user keep their input.

    def _on_cusp_nc_selected(self, combo: "QComboBox", slope_widget: "SlopeInput") -> None:
        """Called when a per-cusp NC combo selection changes."""
        pass  # Do NOT auto-suggest a fill slope

    def _update_cusp_row_stack(self, row_dict: dict) -> None:
        """Switch stack to fill panel (0) or charge panel (1); enable/disable m/e."""
        filled = row_dict["fill_chk"].isChecked()
        if filled:
            row_dict["stack"].setCurrentIndex(0)
        else:
            row_dict["stack"].setCurrentIndex(1)
            is_point = self._other_point.isChecked()
            row_dict["m_spin"].setEnabled(is_point)
            row_dict["e_spin"].setEnabled(is_point)

    def _on_other_mode_changed(self, point_checked: bool) -> None:
        self._other_point_row.setVisible(point_checked)
        self._other_grid_row.setVisible(not point_checked)
        for rd in self._cusp_fill_rows:
            self._update_cusp_row_stack(rd)

    # ------------------------------------------------------------------
    # Internal — fill computation with auto-Weyl dispatch
    # ------------------------------------------------------------------

    def _other_charge_points(self, n_cusps: int) -> list[tuple[list[int], list[Fraction]]]:
        """Return the list of (m_other, e_other) pairs to compute.

        Single-cusp: always [([], [])].
        Multi-cusp point mode: one pair from the spinboxes.
        Multi-cusp grid mode: all (m, e) on the Cartesian grid.
        """
        if n_cusps == 1:
            return [([], [])]

        if self._other_point.isChecked():
            # Multi-cusp point mode: charges are collected per-cusp in
            # _on_multi_fill_clicked; this function is only called for
            # single-cusp (n_cusps == 1) which returns [] above.
            return [([], [])]
        else:
            m_lo = self._other_m_min.value()
            m_hi = self._other_m_max.value()
            e_lo = Fraction(int(self._other_e_min.value() * 2), 2)
            e_hi = Fraction(int(self._other_e_max.value() * 2), 2)
            e_step = Fraction(1, 2)
            points = []
            m = m_lo
            while m <= m_hi:
                e = e_lo
                while e <= e_hi:
                    points.append(([m] * n_cusps, [e] * n_cusps))
                    e += e_step
                m += 1
            return points if points else [([0] * n_cusps, [Fraction(0)] * n_cusps)]

    def _on_fill_clicked(self) -> None:
        """Entry point for fill computation.

        Weyl vectors come from the per-cycle Weyl check run after NC search.
        They are stored in the selected NC cycle's ViewModel (vm.weyl_a/b).
        """
        s = self._session
        if s.nz_data is None:
            return

        # Check if we're in multi-cusp mode
        n_cusps = s.nz_data.r if s.nz_data is not None else 1
        if n_cusps > 1:
            self._on_multi_fill_clicked()
            return

        # Read Weyl vectors from the selected NC cycle's ViewModel.
        # These are computed in the NC basis (basis-changed NZ) by
        # _launch_weyl_for_nc_compatibility — exactly what filling needs.
        nc_idx = self._nc_combo.currentIndex()
        ab          = None
        adj_pass    = None
        is_marginal = None
        if 0 <= nc_idx < len(self._nc_cycle_vms):
            vm = self._nc_cycle_vms[nc_idx]
            # Always read adj_pass / is_marginal — they are valid even when
            # AB vectors are None (u-independent indices like torus knots).
            adj_pass    = vm.adjoint_proj_pass     # refined q^1 → W_i compat
            is_marginal = vm.is_marginal           # unrefined q^1 → kernel choice
            if vm.weyl_a is not None and vm.weyl_b is not None:
                ab = type('obj', (object,), {'a': vm.weyl_a, 'b': vm.weyl_b})()

        print(
            f"[FILL] NC idx={nc_idx}, ab={'present' if ab else 'None'}, "
            f"adj_pass={adj_pass}, is_marginal={is_marginal}"
        )

        # Launch joint adjoint check (d=1) — needs full filling config
        cusp_idx = self._cusp_combo.currentData() or 0
        if 0 <= nc_idx < len(self._nc_cycle_vms) and s.index_queries:
            nc_vm = self._nc_cycle_vms[nc_idx]
            cusp_specs_1 = [{"cusp_idx": cusp_idx, "nc_P": nc_vm.P, "nc_Q": nc_vm.Q}]
            gen = self._session_gen
            if self._joint_adjoint_worker is not None:
                try:
                    self._joint_adjoint_worker.finished.disconnect()
                    self._joint_adjoint_worker.error.disconnect()
                except RuntimeError:
                    pass
            self._joint_adjoint_worker = MultiCuspNcCompatWorker(
                nz_data      = s.nz_data,
                cusp_specs   = cusp_specs_1,
                index_queries = list(s.index_queries),
                num_hard     = s.num_hard(),
                q_order_half = s.q_order_half,
                parent       = self,
            )
            self._joint_adjoint_worker.finished.connect(
                lambda p, g=gen: self._on_joint_adjoint_done(p, g)
            )
            self._joint_adjoint_worker.error.connect(
                lambda e, g=gen: self._on_joint_adjoint_error(e, g)
            )
            self._joint_adjoint_worker.start()

        self._launch_fill_path(ab=ab, adj_pass=adj_pass, is_marginal=is_marginal)

    def _launch_auto_weyl_then_fill(self) -> None:
        """Launch WeylWorker to compute Weyl vectors, then proceed to fill."""
        s = self._session
        entries = [
            (q.m_ext, q.e_ext, q.result)
            for q in s.index_queries
            if q.result is not None
        ]

        print(f"[WEYL] Launching: found {len(entries)} entries with results (out of {len(s.index_queries)} total)")

        self._fill_btn.setEnabled(False)
        self._fill_stop_btn.setEnabled(True)
        self._fill_progress.setRange(0, 0)   # indeterminate
        self._fill_progress.setVisible(True)
        self._auto_weyl_status.setText("Computing Weyl vectors…")
        self._auto_weyl_status.setVisible(True)
        self._card.set_status(CardStatus.RUNNING)

        gen = self._session_gen
        self._auto_weyl_worker = WeylWorker(
            entries      = entries,
            num_hard     = s.num_hard(),
            q_order_half = s.q_order_half,
            parent       = self,
        )
        self._auto_weyl_worker.finished.connect(
            lambda p, g=gen: self._on_auto_weyl_done(p, g)
        )
        self._auto_weyl_worker.error.connect(
            lambda e, g=gen: self._on_auto_weyl_error(e, g)
        )
        self._auto_weyl_worker.start()

    def _on_auto_weyl_done(self, payload: dict, gen: int) -> None:
        """Handle successful auto-Weyl computation, then launch fill."""
        if gen != self._session_gen:
            print("[WEYL] Ignoring stale generation")
            return   # stale

        self._auto_weyl_worker = None
        s = self._session

        ab = payload["ab_vectors"]
        adj_pass = payload.get("adjoint_is_pass")
        adj_value = payload.get("adjoint_value")

        print(f"[WEYL] Done: ab={'present' if ab else 'None'}, adj_pass={adj_pass}")

        # Save Weyl result
        s.weyl_result = ab
        s.weyl_adjoint_pass = adj_pass
        s.weyl_checked = True

        # Build and display Weyl result with AB vectors and projection info
        if ab is not None:
            vm = build_weyl_vm(
                ab, s.num_hard(),
                adjoint_value=float(adj_value) if adj_value is not None else None,
                adjoint_passed=adj_pass,
            )
            self._card.set_advisories(vm.advisories)

            # Format AB vectors display
            a_str = ", ".join(f"{a}" for a in ab.a[:min(3, len(ab.a))])
            b_str = ", ".join(f"{b}" for b in ab.b[:min(3, len(ab.b))])
            adj_str = "✓ Pass" if adj_pass else "✗ Fail" if adj_pass is False else "—"

            status_msg = (
                f"Weyl: a=[{a_str}{'…' if len(ab.a) > 3 else ''}] "
                f"b=[{b_str}{'…' if len(ab.b) > 3 else ''}] | "
                f"q¹ projection: {adj_str}"
            )
            self._auto_weyl_status.setText(status_msg)
            self._auto_weyl_status.setVisible(True)
        else:
            self._auto_weyl_status.setText("Weyl check: AB vectors could not be extracted")
            self._auto_weyl_status.setVisible(True)

        # Proceed to fill path with Weyl vectors.
        # is_marginal not computed on this path — defaults to None (unrefined kernel).
        self._launch_fill_path(ab=ab, adj_pass=adj_pass, is_marginal=None)

    def _on_auto_weyl_error(self, msg: str, gen: int) -> None:
        """Handle auto-Weyl error; fall back to unrefined fill."""
        if gen != self._session_gen:
            print("[WEYL] Ignoring stale generation (error)")
            return   # stale

        self._auto_weyl_worker = None
        print(f"[WEYL] ERROR: {msg}")

        # Show warning but continue with unrefined fill
        self._auto_weyl_status.setText(f"Weyl computation failed: {msg}. Using unrefined fill.")
        self._auto_weyl_status.setVisible(True)

        # Proceed to fill path without Weyl vectors (unrefined)
        self._launch_fill_path(ab=None, adj_pass=None)

    def _refresh_fill_info_view(self) -> None:
        """Regenerate the fill-info MathView with the latest fill data and adjoint results."""
        if not self._last_fill_info_specs:
            return
        try:
            info_html = format_fill_info_html(
                self._last_fill_info_specs,
                self._last_fill_info_result,
                adjoint_per_cusp=self._last_adjoint_per_cusp,
                ab_vectors=self._last_fill_info_ab,
                adj_pass=self._last_fill_info_adj_pass,
            )
            if info_html:
                self._fill_info_view.set_html(info_html)
                self._fill_info_view.setVisible(True)
        except Exception:
            pass

    def _on_joint_adjoint_done(self, payload: dict, gen: int) -> None:
        """Store per-cusp q¹ adjoint projection results and refresh the fill-info panel."""
        if gen != self._session_gen:
            return
        self._joint_adjoint_worker = None
        per_cusp = payload.get("per_cusp_adjoint", [])
        adj_pass = payload.get("adjoint_is_pass")
        print(f"[JOINT-ADJ] done: per_cusp={per_cusp}, overall_pass={adj_pass}", flush=True)
        self._last_adjoint_per_cusp = per_cusp if per_cusp else None
        self._last_fill_info_ab = payload.get("ab_vectors")
        self._last_fill_info_adj_pass = adj_pass
        self._refresh_fill_info_view()

    def _on_joint_adjoint_error(self, msg: str, gen: int) -> None:
        """Handle joint adjoint check error — log and show in fill-info view."""
        if gen != self._session_gen:
            return
        self._joint_adjoint_worker = None
        print(f"[JOINT-ADJ] ERROR: {msg}", flush=True)
        # Show error as a degenerate adjoint row
        self._last_adjoint_per_cusp = [{"cusp_idx": "?", "value": None, "is_pass": False,
                                         "_error": msg}]
        self._refresh_fill_info_view()

    def _launch_fill_path(self, ab, adj_pass, is_marginal=None) -> None:
        """Route to the correct filling path.

        Three paths:
          1. ab is None
               → fully unrefined (I_{3D} + K unrefined)
          2. ab present AND is_marginal is False  (unrefined q^1 proj ≤ -1)
               → refined (I^ref + K^ref IS chain)
               incompat_edges = edges where a[j]∉ℤ or 2b[j]∉ℤ or refined q^1 > -1
          3. ab present AND is_marginal is True   (unrefined q^1 proj ≥ 0, marginal)
               → half-refined (I^ref + K(P,Q) unrefined kernel)
               incompat_edges = same rule as path 2

        Kernel: refined K^ref iff NOT marginal (is_marginal is False).
        W_j compatibility: a[j]∈ℤ AND 2b[j]∈ℤ AND refined q^1 ≤ -1.
        """
        s = self._session
        if s.nz_data is None:
            return

        n_cusps = s.nz_data.r if s.nz_data is not None else 1

        # Determine basis cycle (P, Q)
        nc_idx = self._nc_combo.currentIndex()
        if nc_idx < 0 or nc_idx >= len(self._nc_cycle_vms):
            # Fallback to meridian (1, 0)
            nc_P, nc_Q = 1, 0
            basis_label = "Meridian (1,0)"
        else:
            nc_vm = self._nc_cycle_vms[nc_idx]
            nc_P, nc_Q = nc_vm.P, nc_vm.Q
            basis_label = f"NC ({nc_P},{nc_Q})"

        user_P, user_Q = self._fill_slope.get_slope()
        if not self._fill_slope.is_valid():
            return

        cusp_idx = self._cusp_combo.currentData() or 0
        charge_points = self._other_charge_points(n_cusps)

        # u-independent indices (torus knots): AB extraction returns None
        # because I(m=0,e≠0) = 0, but adj_pass is overridden to True.
        # Synthesise all-zero AB vectors so that all edges are compatible.
        if ab is None and adj_pass is True:
            num_hard = s.num_hard()
            ab = type('obj', (object,), {
                'a': [0] * num_hard,
                'b': [0] * num_hard,
            })()

        use_refined_index  = ab is not None
        # Kernel: refined K^ref iff NOT marginal (unrefined q^1 proj ≤ -1).
        # is_marginal=True  → marginal cycle → unrefined K(P,Q)
        # is_marginal=False → non-marginal   → refined K^ref
        # is_marginal=None  → undetermined   → conservative: unrefined
        use_refined_kernel = use_refined_index and (is_marginal is False)

        print(
            f"[FILL] Deciding path: ab={'present' if ab else 'None'}, "
            f"is_marginal={is_marginal}, adj_pass={adj_pass}, "
            f"refined_index={use_refined_index}, refined_kernel={use_refined_kernel}"
        )

        if not use_refined_index:
            print("[FILL] → Path 1: fully UNREFINED (I_3D + K unrefined)")
            self._launch_unrefined_fill_workers(
                nc_P, nc_Q, user_P, user_Q, cusp_idx, charge_points, basis_label
            )
        elif use_refined_kernel:
            print("[FILL] → Path 2: REFINED (I^ref + K^ref IS chain)")
            weyl_a = list(ab.a)
            weyl_b = list(ab.b)
            self._launch_refined_fill_workers(
                nc_P, nc_Q, user_P, user_Q, cusp_idx, charge_points,
                weyl_a, weyl_b, adj_pass, basis_label, ab=ab,
            )
        else:
            print("[FILL] → Path 3: HALF-REFINED (I^ref + K(P,Q) unrefined kernel)")
            weyl_a = list(ab.a)
            weyl_b = list(ab.b)
            self._launch_unrefined_kernel_fill_workers(
                nc_P, nc_Q, user_P, user_Q, cusp_idx, charge_points,
                weyl_a, weyl_b, adj_pass, basis_label, ab=ab,
            )

    def _launch_refined_fill_workers(
        self, nc_P: int, nc_Q: int, user_P: int, user_Q: int, cusp_idx: int,
        charge_points: list, weyl_a, weyl_b, adj_pass, basis_label: str,
        ab=None,
    ) -> None:
        """Launch FillWorker for each charge point (path 2: refined kernel)."""
        s = self._session
        n_cusps = s.nz_data.r if s.nz_data is not None else 1

        # Clear old results before starting new computation
        self._fill_table.clear_rows()
        s.fill_queries.clear()
        self._reset_fill_edge_toggles()
        self._last_fill_info_specs  = []
        self._last_fill_info_result = None
        self._last_adjoint_per_cusp = None
        self._last_fill_info_adj_pass = None
        self._last_fill_info_ab = None
        self._fill_info_view.setVisible(False)

        # W_j compatible iff a[j]∈ℤ AND 2b[j]∈ℤ AND refined q^1 = -1.
        # _compute_extended_incompat_edges turns off ALL edges when adj_pass
        # is not True, and only the a/b-incompat ones when adj_pass is True.
        incompat_edges: list[int] = []
        if weyl_a is not None:
            ab_obj = type('obj', (object,), {'a': weyl_a, 'b': weyl_b})()
            incompat_edges = _compute_extended_incompat_edges(ab_obj, adj_pass)

        self._fill_btn.setEnabled(False)
        self._fill_stop_btn.setEnabled(True)
        self._fill_grid_total = len(charge_points)
        self._fill_grid_done = 0
        if self._fill_grid_total > 1:
            self._fill_progress.setVisible(False)
            self._fill_grid_bar.setRange(0, self._fill_grid_total)
            self._fill_grid_bar.setValue(0)
            self._fill_grid_bar.setVisible(True)
        else:
            self._fill_progress.setRange(0, 0)   # indeterminate
            self._fill_progress.setVisible(True)
            self._fill_grid_bar.setVisible(False)
        self._fill_status.setText(
            f"Computing {self._fill_grid_total} filling(s)…"
            if self._fill_grid_total > 1 else "Computing…"
        )
        self._fill_status.setVisible(True)
        self._card.set_status(CardStatus.RUNNING)

        gen = self._session_gen
        for m_other, e_other in charge_points:
            m_cells, eq_cell = build_fill_placeholder_cells(
                list(m_other), list(e_other),
                manifold_name=s.manifold_name or "",
            )
            row = self._fill_table.add_row(m_cells, eq_cell, "", "—")
            self._fill_table.set_row_computing(row)

            worker = FillWorker(
                nz_data        = s.nz_data,
                cusp_idx       = cusp_idx,
                nc_P           = nc_P,
                nc_Q           = nc_Q,
                user_P         = user_P,
                user_Q         = user_Q,
                m_other        = list(m_other),
                e_other        = list(e_other),
                q_order_half   = s.q_order_half,
                weyl_a         = weyl_a,
                weyl_b         = weyl_b,
                weyl_ab        = ab,
                incompat_edges = incompat_edges or None,
                manifold_name  = s.manifold_name if s.manifold_name else "unknown",
                parent         = self,
            )
            worker.finished.connect(
                lambda p, w=worker, r=row, nP=nc_P, nQ=nc_Q, uP=user_P, uQ=user_Q,
                       ci=cusp_idx, mo=list(m_other), eo=list(e_other), g=gen, ie=incompat_edges:
                    self._on_fill_finished(p, r, nP, nQ, uP, uQ, ci, mo, eo, g, w, ie)
            )
            worker.error.connect(lambda e, w=worker, r=row, g=gen: self._on_fill_error(e, r, g, w))
            self._fill_workers.append(worker)
            worker.start()

    def _launch_unrefined_kernel_fill_workers(
        self, nc_P: int, nc_Q: int, user_P: int, user_Q: int, cusp_idx: int,
        charge_points: list, weyl_a, weyl_b, adj_pass, basis_label: str,
        ab=None,
    ) -> None:
        """Launch UnrefinedKernelFillWorker for each charge point (path 3).

        NC cycle not strongly NC → unrefined K(P,Q) kernel.  I^ref is still
        used.  Same W_j compatibility rule: all off when refined q^1 ≠ -1.
        """
        s = self._session

        self._fill_table.clear_rows()
        s.fill_queries.clear()
        self._reset_fill_edge_toggles()
        self._last_fill_info_specs  = []
        self._last_fill_info_result = None
        self._last_adjoint_per_cusp = None
        self._last_fill_info_adj_pass = None
        self._last_fill_info_ab = None
        self._fill_info_view.setVisible(False)

        incompat_edges: list[int] = []
        if weyl_a is not None:
            ab_obj = type('obj', (object,), {'a': weyl_a, 'b': weyl_b})()
            incompat_edges = _compute_extended_incompat_edges(ab_obj, adj_pass)

        self._fill_btn.setEnabled(False)
        self._fill_stop_btn.setEnabled(True)
        self._fill_grid_total = len(charge_points)
        self._fill_grid_done = 0
        if self._fill_grid_total > 1:
            self._fill_progress.setVisible(False)
            self._fill_grid_bar.setRange(0, self._fill_grid_total)
            self._fill_grid_bar.setValue(0)
            self._fill_grid_bar.setVisible(True)
        else:
            self._fill_progress.setRange(0, 0)
            self._fill_progress.setVisible(True)
            self._fill_grid_bar.setVisible(False)
        self._fill_status.setText(
            f"Computing {self._fill_grid_total} filling(s)…"
            if self._fill_grid_total > 1 else "Computing…"
        )
        self._fill_status.setVisible(True)
        self._card.set_status(CardStatus.RUNNING)

        gen = self._session_gen
        for m_other, e_other in charge_points:
            m_cells, eq_cell = build_fill_placeholder_cells(
                list(m_other), list(e_other),
                manifold_name=s.manifold_name or "",
            )
            row = self._fill_table.add_row(m_cells, eq_cell, "", "—")
            self._fill_table.set_row_computing(row)

            worker = UnrefinedKernelFillWorker(
                nz_data        = s.nz_data,
                cusp_idx       = cusp_idx,
                nc_P           = nc_P,
                nc_Q           = nc_Q,
                user_P         = user_P,
                user_Q         = user_Q,
                m_other        = list(m_other),
                e_other        = list(e_other),
                q_order_half   = s.q_order_half,
                weyl_a         = weyl_a,
                weyl_b         = weyl_b,
                weyl_ab        = ab,
                incompat_edges = incompat_edges or None,
                manifold_name  = s.manifold_name if s.manifold_name else "unknown",
                parent         = self,
            )
            worker.finished.connect(
                lambda p, w=worker, r=row, nP=nc_P, nQ=nc_Q, uP=user_P, uQ=user_Q,
                       ci=cusp_idx, mo=list(m_other), eo=list(e_other), g=gen, ie=incompat_edges:
                    self._on_fill_finished(p, r, nP, nQ, uP, uQ, ci, mo, eo, g, w, ie)
            )
            worker.error.connect(lambda e, w=worker, r=row, g=gen: self._on_fill_error(e, r, g, w))
            self._fill_workers.append(worker)
            worker.start()

    def _launch_unrefined_fill_workers(
        self, nc_P: int, nc_Q: int, user_P: int, user_Q: int, cusp_idx: int,
        charge_points: list, basis_label: str
    ) -> None:
        """Launch UnrefinedFillWorker for each charge point."""
        s = self._session
        n_cusps = s.nz_data.r if s.nz_data is not None else 1

        # Clear old results before starting new computation
        self._fill_table.clear_rows()
        s.fill_queries.clear()
        self._reset_fill_edge_toggles()

        self._fill_btn.setEnabled(False)
        self._fill_stop_btn.setEnabled(True)
        self._fill_grid_total = len(charge_points)
        self._fill_grid_done = 0
        if self._fill_grid_total > 1:
            self._fill_progress.setVisible(False)
            self._fill_grid_bar.setRange(0, self._fill_grid_total)
            self._fill_grid_bar.setValue(0)
            self._fill_grid_bar.setVisible(True)
        else:
            self._fill_progress.setRange(0, 0)   # indeterminate
            self._fill_progress.setVisible(True)
            self._fill_grid_bar.setVisible(False)
        self._fill_status.setText(
            f"Computing {self._fill_grid_total} filling(s)…"
            if self._fill_grid_total > 1 else "Computing…"
        )
        self._fill_status.setVisible(True)
        self._card.set_status(CardStatus.RUNNING)

        gen = self._session_gen
        for m_other, e_other in charge_points:
            m_cells, eq_cell = build_fill_placeholder_cells(
                list(m_other), list(e_other),
                manifold_name=s.manifold_name or "",
            )
            row = self._fill_table.add_row(m_cells, eq_cell, "", "—")
            self._fill_table.set_row_computing(row)

            worker = UnrefinedFillWorker(
                nz_data      = s.nz_data,
                cusp_idx     = cusp_idx,
                user_P       = user_P,
                user_Q       = user_Q,
                m_other      = list(m_other),
                e_other      = list(e_other),
                q_order_half = s.q_order_half,
                manifold_name= s.manifold_name if s.manifold_name else "unknown",
                parent       = self,
            )
            worker.finished.connect(
                lambda p, w=worker, r=row, nP=nc_P, nQ=nc_Q, uP=user_P, uQ=user_Q,
                       ci=cusp_idx, mo=list(m_other), eo=list(e_other), g=gen:
                    self._on_unrefined_fill_finished(p, r, nP, nQ, uP, uQ, ci, mo, eo, g, w)
            )
            worker.error.connect(lambda e, w=worker, r=row, g=gen: self._on_fill_error(e, r, g, w))
            self._fill_workers.append(worker)
            worker.start()

    def _nc_vm_lookup(self, nc_P: int, nc_Q: int, cusp_idx: int):
        """Return the NCCycleViewModel for (nc_P, nc_Q, cusp_idx), or None."""
        for vm in self._nc_cycle_vms:
            if vm.P == nc_P and vm.Q == nc_Q and vm.cusp_idx == cusp_idx:
                return vm
        return None

    def _adjoint_from_nc_vm(
        self, nc_P: int, nc_Q: int, cusp_idx: int
    ) -> "list[dict] | None":
        """Return a preliminary adjoint_per_cusp list from the stored NC cycle VM."""
        vm = self._nc_vm_lookup(nc_P, nc_Q, cusp_idx)
        if vm is None:
            return None
        val = vm.adjoint_proj_value
        is_pass = vm.adjoint_proj_pass
        if val is not None or is_pass is not None:
            return [{"cusp_idx": cusp_idx, "value": val, "is_pass": is_pass}]
        return None

    def _on_fill_finished(
        self, payload: dict, row: int,
        nc_P: int, nc_Q: int, user_P: int, user_Q: int,
        cusp_idx: int, m_other: list, e_other: list, gen: int,
        worker: object = None, incompat_edges: list | None = None,
    ) -> None:
        w = worker if worker is not None else self.sender()
        if w in self._fill_workers:
            self._fill_workers.remove(w)

        if gen != self._session_gen:
            return   # stale result — manifold was reloaded

        # Grid progress tracking
        self._fill_grid_done += 1
        all_done = not self._fill_workers   # list already had sender removed above
        if self._fill_grid_total > 1:
            self._fill_grid_bar.setValue(self._fill_grid_done)
            self._fill_status.setText(
                f"{self._fill_grid_done} / {self._fill_grid_total} done…"
            )
        if all_done:
            self._fill_btn.setEnabled(True)
            self._fill_stop_btn.setEnabled(False)
            self._fill_progress.setVisible(False)
            self._fill_grid_bar.setVisible(False)
            self._fill_status.setVisible(False)
            self._auto_weyl_status.setVisible(False)
            self._fill_current_row = None

        result = payload["result"]
        p      = payload["p"]
        q      = payload["q"]
        s      = self._session

        try:
            if result is None:
                series_latex = "—"
            else:
                series_latex = format_filled_series_latex(
                    result.series,
                    result.num_hard,
                    result.has_cusp_eta,
                    result.num_cusp_eta,
                )
        except Exception:
            series_latex = "—"

        # ── Update fill-info panel (NC cycle, transformed slope, HJ) ──
        if all_done:
            try:
                vm = self._nc_vm_lookup(nc_P, nc_Q, cusp_idx)
                fake_spec = {
                    "cusp_idx": cusp_idx, "nc_P": nc_P, "nc_Q": nc_Q, "p": p, "q": q,
                    "weyl_a": vm.weyl_a if vm else None,
                    "weyl_b": vm.weyl_b if vm else None,
                }
                self._last_fill_info_specs  = [fake_spec]
                self._last_fill_info_result = result
                self._last_adjoint_per_cusp = self._adjoint_from_nc_vm(nc_P, nc_Q, cusp_idx)
                # Store full ABVectors for per-edge display
                cycle_key = (nc_P, nc_Q)
                nc_weyl = self._nc_weyl_results.get(cycle_key, {})
                self._last_fill_info_ab = nc_weyl.get("ab")
                self._last_fill_info_adj_pass = nc_weyl.get("adj_pass") \
                    if nc_weyl else None
                self._refresh_fill_info_view()
            except Exception:
                pass

        m_cells, eq_cell = build_fill_row_cells(
            user_P, user_Q, m_other, e_other,
            slope_a=r"\alpha", slope_b=r"\beta",
            manifold_name=s.manifold_name or "",
        )
        self._fill_table.update_row_metadata(row, m_cells, eq_cell)
        self._fill_table.set_row_result(row, series_latex, "computed")

        fq = FillQuery(
            cusp_idx     = cusp_idx,      # the cusp actually used in computation
            nc_P         = nc_P,
            nc_Q         = nc_Q,
            user_P       = user_P,
            user_Q       = user_Q,
            p            = p,
            q            = q,
            m_other      = m_other,       # charges actually passed to worker
            e_other      = e_other,
            q_order_half = s.q_order_half,
            result       = result,
            weyl_a       = list(s.weyl_result.a) if s.weyl_result else None,
            weyl_b       = list(s.weyl_result.b) if s.weyl_result else None,
            incompat_edges = incompat_edges or [],
            source       = "computed",
            row_index    = row,
        )
        s.fill_queries.append(fq)
        if s.stage < PipelineStage.FILLED:
            s.stage = PipelineStage.FILLED

        # Rebuild V_i toggles on first cusp-eta result
        if result is not None and result.has_cusp_eta:
            n_ce = result.num_cusp_eta
            if len(self._fill_cusp_checkboxes) != n_ce:
                self._rebuild_cusp_eta_toggles(n_ce)

        # Disable and uncheck W checkboxes for incompatible edges
        if incompat_edges:
            for j in incompat_edges:
                if 0 <= j < len(self._fill_edge_checkboxes):
                    cb = self._fill_edge_checkboxes[j]
                    cb.setChecked(False)
                    cb.setEnabled(False)
                    cb.setToolTip(
                        f"W{chr(0x2080 + j)} disabled: incompatible with Dehn filling "
                        f"(a[{j}]∉ℤ or 2b[{j}]∉ℤ)"
                    )

        self._card.set_status(CardStatus.DONE)
        self._update_summary()
        self.session_updated.emit(s)

    def _on_unrefined_fill_finished(
        self, payload: dict, row: int,
        nc_P: int, nc_Q: int, user_P: int, user_Q: int,
        cusp_idx: int, m_other: list, e_other: list, gen: int,
        worker: object = None,
    ) -> None:
        """Handle unrefined fill completion."""
        w = worker if worker is not None else self.sender()
        if w in self._fill_workers:
            self._fill_workers.remove(w)

        if gen != self._session_gen:
            return   # stale result — manifold was reloaded

        # Grid progress tracking
        self._fill_grid_done += 1
        all_done = not self._fill_workers
        if self._fill_grid_total > 1:
            self._fill_grid_bar.setValue(self._fill_grid_done)
            self._fill_status.setText(
                f"{self._fill_grid_done} / {self._fill_grid_total} done…"
            )
        if all_done:
            self._fill_btn.setEnabled(True)
            self._fill_stop_btn.setEnabled(False)
            self._fill_progress.setVisible(False)
            self._fill_grid_bar.setVisible(False)
            self._fill_status.setVisible(False)
            self._auto_weyl_status.setVisible(False)
            self._fill_current_row = None

        result = payload["result"]
        s = self._session
        try:
            if result is None:
                series_latex = "—"
            else:
                # Format unrefined series
                series_latex = format_unrefined_series_latex(result.series)
        except Exception:
            series_latex = "—"

        # ── Update fill-info panel with NC cycle / slope info ─────────
        if all_done:
            try:
                vm = self._nc_vm_lookup(nc_P, nc_Q, cusp_idx)
                fake_spec = {
                    "cusp_idx": cusp_idx, "nc_P": nc_P, "nc_Q": nc_Q, "p": 0, "q": 0,
                    "weyl_a": vm.weyl_a if vm else None,
                    "weyl_b": vm.weyl_b if vm else None,
                }
                self._last_fill_info_specs  = [fake_spec]
                self._last_fill_info_result = result
                self._last_adjoint_per_cusp = self._adjoint_from_nc_vm(nc_P, nc_Q, cusp_idx)
                cycle_key = (nc_P, nc_Q)
                nc_weyl = self._nc_weyl_results.get(cycle_key, {})
                self._last_fill_info_ab = nc_weyl.get("ab")
                self._last_fill_info_adj_pass = nc_weyl.get("adj_pass")
                info_html = format_fill_info_html([fake_spec], result,
                                                  adjoint_per_cusp=self._last_adjoint_per_cusp,
                                                  ab_vectors=self._last_fill_info_ab,
                                                  adj_pass=self._last_fill_info_adj_pass)
                if info_html:
                    self._fill_info_view.set_html(info_html)
                    self._fill_info_view.setVisible(True)
            except Exception:
                pass

        m_cells, eq_cell = build_fill_row_cells(
            user_P, user_Q, m_other, e_other,
            slope_a=r"\alpha", slope_b=r"\beta",
            manifold_name=s.manifold_name or "",
        )
        self._fill_table.update_row_metadata(row, m_cells, eq_cell)
        self._fill_table.set_row_result(row, series_latex, "computed")

        fq = FillQuery(
            cusp_idx     = cusp_idx,
            nc_P         = nc_P,
            nc_Q         = nc_Q,
            user_P       = user_P,
            user_Q       = user_Q,
            p            = 0,
            q            = 0,
            m_other      = m_other,
            e_other      = e_other,
            q_order_half = s.q_order_half,
            result       = result,
            weyl_a       = list(s.weyl_result.a) if s.weyl_result else None,
            weyl_b       = list(s.weyl_result.b) if s.weyl_result else None,
            incompat_edges = [],
            source       = "computed",
            unrefined_fallback = True,
            row_index    = row,
        )
        s.fill_queries.append(fq)
        if s.stage < PipelineStage.FILLED:
            s.stage = PipelineStage.FILLED

        self._card.set_status(CardStatus.DONE)
        self._update_summary()
        self.session_updated.emit(s)

    def _on_fill_error(self, msg: str, row: int, gen: int, worker: object = None) -> None:
        w = worker if worker is not None else self.sender()
        if w in self._fill_workers:
            self._fill_workers.remove(w)
        if gen != self._session_gen:
            return
        self._fill_grid_done += 1
        if not self._fill_workers:   # all done (some may have errored)
            self._fill_btn.setEnabled(True)
            self._fill_stop_btn.setEnabled(False)
            self._fill_progress.setVisible(False)
            self._fill_grid_bar.setVisible(False)
            self._fill_status.setVisible(False)
            self._auto_weyl_status.setVisible(False)
            self._fill_current_row = None
        self._fill_table.set_row_result(row, f"Error: {msg}", "—")
        logging.warning("FillWorker error: %s", msg)
        self._card.set_status(CardStatus.ERROR)

    def _on_multi_fill_clicked(self) -> None:
        """Handle multi-cusp filling (selected cusps simultaneously)."""
        s = self._session
        if s.nz_data is None:
            return

        # Collect cusp_specs from the per-cusp rows (only checked cusps)
        cusp_specs = []
        for row in self._cusp_fill_rows:
            fill_chk = row["fill_chk"]
            cusp_idx = row["cusp_idx"]

            if not fill_chk.isChecked():
                continue  # Skip unchecked cusps

            nc_combo = row["nc_combo"]
            slope = row["slope"]

            nc_vm_idx = nc_combo.currentData()
            if nc_vm_idx is None or nc_vm_idx < 0 or nc_vm_idx >= len(self._nc_cycle_vms):
                from manifold_index.viewmodels.advisory import Advisories
                self._card.set_advisories([
                    Advisories.warning(
                        "Incomplete selection",
                        f"Please select an NC cycle for Cusp {cusp_idx}."
                    )
                ])
                return

            nc_vm = self._nc_cycle_vms[nc_vm_idx]
            user_P, user_Q = slope.get_slope()
            if not slope.is_valid():
                from manifold_index.viewmodels.advisory import Advisories
                self._card.set_advisories([
                    Advisories.warning(
                        "Invalid slope",
                        f"Slope for Cusp {cusp_idx} is not a valid primitive pair."
                    )
                ])
                return

            # Use per-cycle Weyl vectors from this NC cycle's ViewModel
            weyl_a = list(nc_vm.weyl_a) if nc_vm.weyl_a is not None else None
            weyl_b = list(nc_vm.weyl_b) if nc_vm.weyl_b is not None else None
            # Full ABVectors (with cusp_columns) from the per-cycle Weyl-check
            # result — needed for multi-cusp Weyl shift that sums over ALL cusps.
            weyl_ab_full = None
            nc_res = self._nc_weyl_results.get((nc_vm.P, nc_vm.Q))
            if nc_res is not None:
                weyl_ab_full = nc_res.get("ab")

            cusp_specs.append({
                "cusp_idx": cusp_idx,
                "nc_P": nc_vm.P,
                "nc_Q": nc_vm.Q,
                "user_P": user_P,
                "user_Q": user_Q,
                "weyl_a": weyl_a,
                "weyl_b": weyl_b,
                "weyl_ab": weyl_ab_full,
            })

        # ── Compute joint incompat_edges across ALL cusps ─────────────
        # Edge j is incompatible if ANY cusp has a_I[j] ∉ ℤ or 2·b_I[j] ∉ ℤ.
        num_hard = s.num_hard()
        joint_incompat: set[int] = set()
        for spec in cusp_specs:
            wa = spec.get("weyl_a")
            wb = spec.get("weyl_b")
            if wa is not None and wb is not None:
                for j in range(min(len(wa), len(wb))):
                    a_val = wa[j]
                    b_val = wb[j]
                    try:
                        a_int = (a_val == int(a_val))
                    except (TypeError, ValueError, OverflowError):
                        a_int = False
                    try:
                        b2_int = (2 * b_val == int(2 * b_val))
                    except (TypeError, ValueError, OverflowError):
                        b2_int = False
                    if not a_int or not b2_int:
                        joint_incompat.add(j)
        joint_incompat_list = sorted(joint_incompat) if joint_incompat else None

        # Attach incompat_edges to each cusp_spec so FillingService can zero them
        if joint_incompat_list:
            for spec in cusp_specs:
                spec["incompat_edges"] = joint_incompat_list

        # Verify at least one cusp is selected for filling
        if not cusp_specs:
            from manifold_index.viewmodels.advisory import Advisories
            self._card.set_advisories([
                Advisories.warning(
                    "No cusps selected",
                    "Please check at least one cusp to fill."
                )
            ])
            return

        # ── Launch joint adjoint check (multi-cusp, d ≥ 1) ───────────
        # The per-cycle Weyl checks run at NC search time only vary one
        # cusp's e-charge while setting all others to zero — which is
        # incorrect when d > 1 cusps are filled simultaneously.  The
        # correct check integrates over ALL d filled-cusp fugacities and
        # must be done here, after the user has chosen the full filling
        # setup (which cusps to fill and which NC cycles to use).
        if len(cusp_specs) >= 1:
            gen = self._session_gen
            if self._joint_adjoint_worker is not None:
                try:
                    self._joint_adjoint_worker.finished.disconnect()
                    self._joint_adjoint_worker.error.disconnect()
                except RuntimeError:
                    pass
            self._joint_adjoint_worker = MultiCuspNcCompatWorker(
                nz_data      = s.nz_data,
                cusp_specs   = cusp_specs,
                index_queries = list(s.index_queries),
                num_hard     = s.num_hard(),
                q_order_half = s.q_order_half,
                parent       = self,
            )
            self._joint_adjoint_worker.finished.connect(
                lambda p, g=gen: self._on_joint_adjoint_done(p, g)
            )
            self._joint_adjoint_worker.error.connect(
                lambda e, g=gen: self._on_joint_adjoint_error(e, g)
            )
            self._joint_adjoint_worker.start()

        # ── Build list of charge configs (one per grid point) ─────────
        n_cusps = s.nz_data.r if s.nz_data is not None else 1
        filled_set = {sp["cusp_idx"] for sp in cusp_specs}
        unfilled_idxs = [i for i in range(n_cusps) if i not in filled_set]

        # charge_configs: list of (m_unfilled, e_unfilled, unfilled_charges)
        if not unfilled_idxs:
            charge_configs: list[tuple] = [(None, None, [])]
        elif self._other_point.isChecked():
            row_by_cusp = {rd["cusp_idx"]: rd for rd in self._cusp_fill_rows}
            m_u: list = []
            e_u: list = []
            for ci in unfilled_idxs:
                rd = row_by_cusp.get(ci)
                m_u.append(rd["m_spin"].value() if rd else 0)
                e_u.append(float(rd["e_spin"].value()) if rd else 0.0)
            charge_configs = [(m_u, e_u, list(zip(m_u, e_u)))]
        else:
            # Grid mode: iterate all (m, e) combinations
            m_lo = self._other_m_min.value()
            m_hi = self._other_m_max.value()
            e_lo = Fraction(int(self._other_e_min.value() * 2), 2)
            e_hi = Fraction(int(self._other_e_max.value() * 2), 2)
            e_step = Fraction(1, 2)
            charge_configs = []
            m = m_lo
            while m <= m_hi:
                e = e_lo
                while e <= e_hi:
                    mu = [m] * len(unfilled_idxs)
                    eu = [e] * len(unfilled_idxs)
                    charge_configs.append((mu, eu, [(m, e)] * len(unfilled_idxs)))
                    e += e_step
                m += 1
            if not charge_configs:
                m0 = self._other_m_min.value()
                e0 = Fraction(int(self._other_e_min.value() * 2), 2)
                mu = [m0] * len(unfilled_idxs)
                eu = [e0] * len(unfilled_idxs)
                charge_configs = [(mu, eu, [(m0, e0)] * len(unfilled_idxs))]

        # ── Set up progress and spawn one worker per charge config ─────
        self._fill_table.clear_rows()
        s.multi_fill_queries.clear()
        self._last_fill_info_specs  = []
        self._last_fill_info_result = None
        self._last_adjoint_per_cusp = None
        self._last_fill_info_adj_pass = None
        self._last_fill_info_ab = None
        self._fill_info_view.setVisible(False)
        self._fill_btn.setEnabled(False)
        self._fill_stop_btn.setEnabled(True)
        self._fill_grid_total = len(charge_configs)
        self._fill_grid_done = 0
        if self._fill_grid_total > 1:
            self._fill_progress.setVisible(False)
            self._fill_grid_bar.setRange(0, self._fill_grid_total)
            self._fill_grid_bar.setValue(0)
            self._fill_grid_bar.setVisible(True)
        else:
            self._fill_progress.setRange(0, 0)
            self._fill_progress.setVisible(True)
            self._fill_grid_bar.setVisible(False)
        self._fill_status.setText(
            f"Computing {self._fill_grid_total} filling(s)…"
            if self._fill_grid_total > 1 else "Computing multi-cusp filling…"
        )
        self._fill_status.setVisible(True)
        self._card.set_status(CardStatus.RUNNING)

        gen = self._session_gen
        for m_unfilled, e_unfilled, unfilled_charges in charge_configs:
            uc_indexed = [
                (idx, m, e)
                for idx, (m, e) in zip(unfilled_idxs, unfilled_charges)
            ]
            m_cells, eq_cell = build_multi_fill_placeholder_cells(cusp_specs, uc_indexed, manifold_name=s.manifold_name or "")
            row = self._fill_table.add_row(m_cells, eq_cell, "", "—")
            self._fill_table.set_row_computing(row)

            worker = MultiFillWorker(
                nz_data        = s.nz_data,
                cusp_specs     = cusp_specs,
                q_order_half   = s.q_order_half,
                manifold_name  = s.manifold_name if s.manifold_name else "unknown",
                m_unfilled     = m_unfilled,
                e_unfilled     = e_unfilled,
                parent         = self,
            )
            worker.finished.connect(
                lambda p, w=worker, r=row, g=gen, uc=unfilled_charges, nc=n_cusps:
                    self._on_multi_fill_finished(p, r, g, w, uc, nc)
            )
            worker.error.connect(lambda e, w=worker, r=row, g=gen: self._on_fill_error(e, r, g, w))
            self._fill_workers.append(worker)
            worker.start()

    def _on_multi_fill_finished(
        self, payload: dict, row: int, gen: int, worker: object = None,
        unfilled_charges: list | None = None, n_cusps: int = 1,
    ) -> None:
        """Handle completion of multi-cusp filling computation."""
        w = worker if worker is not None else self.sender()
        if w in self._fill_workers:
            self._fill_workers.remove(w)

        if gen != self._session_gen:
            return   # stale

        result = payload["result"]
        s = self._session
        cusp_specs = payload.get("cusp_specs", [])

        try:
            if result is None:
                series_latex = "—"
            else:
                series_latex = format_filled_series_latex(
                    result.series,
                    result.num_hard,
                    result.has_cusp_eta,
                    result.num_cusp_eta,
                )
        except Exception:
            series_latex = "—"

        # ── Update fill-info panel above table ────────────────────────
        try:
            self._last_fill_info_specs  = list(cusp_specs)
            self._last_fill_info_result = result
            self._refresh_fill_info_view()
        except Exception:
            pass

        # ── Update result row with final label and series ─────────────
        uc = unfilled_charges or []
        filled_set_f = {sp["cusp_idx"] for sp in cusp_specs}
        unfilled_idxs_f = [i for i in range(n_cusps) if i not in filled_set_f]
        uc_indexed = [(idx, m, e) for idx, (m, e) in zip(unfilled_idxs_f, uc)]
        try:
            m_cells, eq_cell = build_multi_fill_row_cells(cusp_specs, uc_indexed, manifold_name=s.manifold_name or "")
            self._fill_table.update_row_metadata(row, m_cells, eq_cell)
        except Exception:
            pass

        self._fill_table.set_row_result(row, series_latex, "computed")

        # Create MultiFillQuery
        mfq = MultiFillQuery(
            cusp_specs       = cusp_specs,
            q_order_half     = s.q_order_half,
            result           = result,
            unfilled_charges = uc_indexed,   # [(cusp_idx, m, e), …]
            source           = "computed",
            row_index        = row,
        )
        s.multi_fill_queries.append(mfq)

        if s.stage < PipelineStage.FILLED:
            s.stage = PipelineStage.FILLED

        # Disable and uncheck W checkboxes for incompatible edges (multi-cusp)
        joint_incompat: set[int] = set()
        for spec in cusp_specs:
            ie = spec.get("incompat_edges")
            if ie:
                joint_incompat.update(ie)
        if joint_incompat:
            for j in sorted(joint_incompat):
                if 0 <= j < len(self._fill_edge_checkboxes):
                    cb = self._fill_edge_checkboxes[j]
                    cb.setChecked(False)
                    cb.setEnabled(False)
                    cb.setToolTip(
                        f"W{chr(0x2080 + j)} disabled: incompatible with Dehn filling "
                        f"(a[{j}]∉ℤ or 2b[{j}]∉ℤ for some cusp)"
                    )

        # Rebuild V_i toggles on first cusp-eta result
        if result is not None and result.has_cusp_eta:
            n_ce = result.num_cusp_eta
            if len(self._fill_cusp_checkboxes) != n_ce:
                self._rebuild_cusp_eta_toggles(n_ce)

        self._fill_grid_done += 1
        if self._fill_grid_total > 1:
            self._fill_grid_bar.setValue(self._fill_grid_done)
            if self._fill_grid_done < self._fill_grid_total:
                self._fill_status.setText(
                    f"{self._fill_grid_done} / {self._fill_grid_total} done…"
                )
                self._update_summary()
                self.session_updated.emit(s)
                return

        self._fill_btn.setEnabled(True)
        self._fill_stop_btn.setEnabled(False)
        self._fill_progress.setVisible(False)
        self._fill_grid_bar.setVisible(False)
        self._fill_status.setVisible(False)
        self._card.set_status(CardStatus.DONE)
        self._update_summary()
        self.session_updated.emit(s)

    def _show_fill_query(self, fq: FillQuery) -> None:
        try:
            if fq.result is None:
                series_latex = "—"
            else:
                if fq.unrefined_fallback:
                    # For unrefined, format directly
                    series_latex = format_unrefined_series_latex(fq.result.series)
                else:
                    inactive_w = [j for j, a in enumerate(self._fill_active_edges()) if not a]
                    inactive_v = [ci for ci, a in enumerate(self._fill_active_cusp_etas()) if not a]
                    projected = fq.result.collapse_eta_edges(inactive_w)
                    projected = projected.collapse_cusp_etas(inactive_v)
                    series_latex = format_filled_series_latex(
                        projected.series,
                        projected.num_hard,
                        projected.has_cusp_eta,
                        projected.num_cusp_eta,
                    )
        except Exception:
            series_latex = "$0$" if (fq.result and fq.result.is_zero) else "—"

        if fq.unrefined_fallback:
            m_cells, eq_cell = build_fill_row_cells(
                fq.user_P, fq.user_Q, fq.m_other, fq.e_other,
                slope_a=r"\alpha", slope_b=r"\beta",
                manifold_name=self._session.manifold_name or "",
            )
        else:
            m_cells, eq_cell = build_fill_row_cells(
                fq.user_P, fq.user_Q, fq.m_other, fq.e_other,
                slope_a=r"\alpha", slope_b=r"\beta",
                manifold_name=self._session.manifold_name or "",
            )

        self._fill_table.add_row(m_cells, eq_cell, series_latex, fq.source)

    def _show_multi_fill_query(self, mfq: MultiFillQuery) -> None:
        """Display a multi-fill query result in the results table."""
        try:
            if mfq.result is None:
                series_latex = "—"
            else:
                inactive_w = [j for j, a in enumerate(self._fill_active_edges()) if not a]
                inactive_v = [ci for ci, a in enumerate(self._fill_active_cusp_etas()) if not a]
                projected = mfq.result.collapse_eta_edges(inactive_w)
                projected = projected.collapse_cusp_etas(inactive_v)
                series_latex = format_filled_series_latex(
                    projected.series,
                    projected.num_hard,
                    projected.has_cusp_eta,
                    projected.num_cusp_eta,
                )
        except Exception:
            series_latex = "$0$" if (mfq.result and mfq.result.is_zero) else "—"

        m_cells, eq_cell = build_multi_fill_row_cells(
            mfq.cusp_specs, mfq.unfilled_charges,   # [(cusp_idx, m, e), …]
            manifold_name=self._session.manifold_name or "",
        )
        self._fill_table.add_row(m_cells, eq_cell, series_latex, mfq.source)
