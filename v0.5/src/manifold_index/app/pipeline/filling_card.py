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
    QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QGroupBox,
    QHBoxLayout, QLabel, QProgressBar, QPushButton, QRadioButton,
    QSpinBox, QVBoxLayout, QWidget,
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
)
from manifold_index.app.widgets.collapsible_card import CollapsibleCard
from manifold_index.app.widgets.series_table import SeriesTable
from manifold_index.app.widgets.math_view import MathView
from manifold_index.app.widgets.slope_input import SlopeInput
from manifold_index.app.workers.nc_search_worker import NCSearchWorker
from manifold_index.app.workers.fill_worker import (
    FillWorker, MultiFillWorker, UnrefinedFillWorker
)
from manifold_index.app.workers.weyl_worker import WeylWorker
from manifold_index.viewmodels.index_vm import build_weyl_vm
from manifold_index.formatters.weyl_fmt import format_weyl_html


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
        self._session_gen: int = 0   # incremented on each unlock(); guards stale signals
        self._auto_weyl_worker: WeylWorker | None = None  # Weyl check before fill
        self._nc_weyl_workers: dict[tuple[int, int], WeylWorker] = {}  # Weyl per NC cycle (P,Q)
        self._nc_weyl_results: dict[tuple[int, int], dict] = {}  # Results per cycle
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
        range_row.addWidget(QLabel("  Q range ±"))
        self._q_range_spin = QSpinBox()
        self._q_range_spin.setRange(0, 50)
        self._q_range_spin.setValue(0)
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
        self._other_point.setChecked(True)
        other_mode_grp = QButtonGroup(self._other_box)
        other_mode_grp.addButton(self._other_point)
        other_mode_grp.addButton(self._other_grid)
        self._other_point.toggled.connect(self._on_other_mode_changed)
        mode_row.addWidget(self._other_point)
        mode_row.addWidget(self._other_grid)
        mode_row.addStretch(1)
        other_layout.addLayout(mode_row)

        # Point mode: single (m, e) for unfilled cusps
        self._other_point_row = QWidget()
        pr = QHBoxLayout(self._other_point_row)
        pr.setContentsMargins(0, 0, 0, 0)
        pr.addWidget(QLabel("m ="))
        self._other_m = QSpinBox()
        self._other_m.setRange(-99, 99)
        self._other_m.setValue(0)
        self._other_m.setFixedWidth(60)
        pr.addWidget(self._other_m)
        pr.addWidget(QLabel("  e ="))
        self._other_e = QDoubleSpinBox()
        self._other_e.setRange(-49.5, 49.5)
        self._other_e.setSingleStep(0.5)
        self._other_e.setDecimals(1)
        self._other_e.setValue(0.0)
        self._other_e.setFixedWidth(70)
        pr.addWidget(self._other_e)
        pr.addStretch(1)
        other_layout.addWidget(self._other_point_row)

        # Grid mode: m and e ranges
        self._other_grid_row = QWidget()
        gr = QHBoxLayout(self._other_grid_row)
        gr.setContentsMargins(0, 0, 0, 0)
        gr.addWidget(QLabel("m ∈ ["))
        self._other_m_min = QSpinBox()
        self._other_m_min.setRange(-99, 99)
        self._other_m_min.setValue(-2)
        self._other_m_min.setFixedWidth(55)
        gr.addWidget(self._other_m_min)
        gr.addWidget(QLabel(","))
        self._other_m_max = QSpinBox()
        self._other_m_max.setRange(-99, 99)
        self._other_m_max.setValue(2)
        self._other_m_max.setFixedWidth(55)
        gr.addWidget(self._other_m_max)
        gr.addWidget(QLabel("]  e ∈ ["))
        self._other_e_min = QDoubleSpinBox()
        self._other_e_min.setRange(-49.5, 49.5)
        self._other_e_min.setSingleStep(0.5)
        self._other_e_min.setDecimals(1)
        self._other_e_min.setValue(-2.0)
        self._other_e_min.setFixedWidth(65)
        gr.addWidget(self._other_e_min)
        gr.addWidget(QLabel(","))
        self._other_e_max = QDoubleSpinBox()
        self._other_e_max.setRange(-49.5, 49.5)
        self._other_e_max.setSingleStep(0.5)
        self._other_e_max.setDecimals(1)
        self._other_e_max.setValue(2.0)
        self._other_e_max.setFixedWidth(65)
        gr.addWidget(self._other_e_max)
        gr.addWidget(QLabel("]"))
        gr.addStretch(1)
        other_layout.addWidget(self._other_grid_row)
        self._other_grid_row.setVisible(False)

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
        fill_preset_row = QHBoxLayout()
        fill_preset_row.addWidget(QLabel("Preset:"))
        self._fill_preset_combo = QComboBox()
        self._fill_preset_combo.addItems(["Full Refined", "Unrefined", "Custom"])
        self._fill_preset_combo.currentTextChanged.connect(self._on_fill_preset_changed)
        fill_preset_row.addWidget(self._fill_preset_combo)
        fill_preset_row.addStretch(1)
        fill_ref_layout.addLayout(fill_preset_row)
        self._fill_edge_container = QWidget()
        fill_edge_layout = QHBoxLayout(self._fill_edge_container)
        fill_edge_layout.setContentsMargins(0, 0, 0, 0)
        self._fill_edge_layout = fill_edge_layout
        fill_ref_layout.addWidget(self._fill_edge_container)
        self._fill_ref_box.setVisible(False)
        bl.addWidget(self._fill_ref_box)

        self._fill_refresh_timer = QTimer(self)
        self._fill_refresh_timer.setSingleShot(True)
        self._fill_refresh_timer.setInterval(300)
        self._fill_refresh_timer.timeout.connect(self._do_refresh_fill_display)

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
        n_cusps = session.manifold_data.num_cusps if session.manifold_data else 1

        if n_cusps > 1:
            # Multi-cusp: hide single-cusp UI, show multi-cusp container
            self._cusp_combo.setVisible(False)
            self._nc_combo.setVisible(False)
            self._fill_slope.setVisible(False)
            self._other_box.setVisible(True)
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
        n_hard = session.num_hard() if session.manifold_data else 0
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
                row["widget"].setParent(None)  # type: ignore[arg-type]
        self._cusp_fill_rows.clear()
        self._fill_btn.setEnabled(False)
        # Reset refinement toggles
        self._fill_refresh_timer.stop()
        self._fill_ref_box.setVisible(False)
        for cb in self._fill_edge_checkboxes:
            cb.setParent(None)  # type: ignore[arg-type]
        self._fill_edge_checkboxes.clear()
        self._fill_preset_combo.blockSignals(True)
        self._fill_preset_combo.setCurrentIndex(0)
        self._fill_preset_combo.blockSignals(False)
        # Reset auto-Weyl UI
        if self._auto_weyl_status is not None:
            self._auto_weyl_status.setVisible(False)
        # Reset NC UI
        self._nc_search_btn.setEnabled(True)
        self._nc_stop_btn.setEnabled(False)
        self._nc_progress.setVisible(False)
        self._nc_status.setVisible(False)
        # Reset Fill UI
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
        s = self._session
        # Group cycles by cusp
        by_cusp: dict[int, list] = {}
        for vm in self._nc_cycle_vms:
            if vm.cusp_idx not in by_cusp:
                by_cusp[vm.cusp_idx] = []
            by_cusp[vm.cusp_idx].append(vm)

        # Render one table per cusp
        # (per-cycle Weyl vectors are now in each VM, not global)
        for cusp_idx in sorted(by_cusp.keys()):
            vms = by_cusp[cusp_idx]
            html = format_nc_cycle_table_html(vms)

            lbl = QLabel(f"Cusp C{cusp_idx}")
            lbl.setProperty("class", "secondary")
            self._nc_table_layout.addWidget(lbl)
            self._nc_table_labels.append(lbl)

            mv = MathView(min_h=80)
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

    def _on_find_nc_clicked(self) -> None:
        s = self._session
        if s.nz_data is None:
            return
        n_cusps = s.manifold_data.num_cusps if s.manifold_data else 1

        p_half = self._p_range_spin.value()
        q_half = self._q_range_spin.value()
        p_range = (-p_half, p_half)
        q_range = (-q_half, q_half)

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

    def _on_nc_finished(self, payload: dict, cusp_idx: int, gen: int, worker: object = None) -> None:
        if gen != self._session_gen:
            return   # stale: a new manifold was loaded since this worker launched
        # Remove the completed worker from the tracking list
        w = worker if worker is not None else self.sender()
        if w in self._nc_workers:
            self._nc_workers.remove(w)

        cycles = payload["cycles"]
        s = self._session

        ncs = NCCycleSet(
            cusp_idx       = cusp_idx,
            search_p_range = (-self._p_range_spin.value(), self._p_range_spin.value()),
            search_q_range = (-self._q_range_spin.value(), self._q_range_spin.value()),
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
            # Don't render tables yet - wait for Weyl check to complete
            # (tables will be rendered after Weyl with compatibility info)
            self._rebuild_nc_combo()
            # Always enable fill button, even if no NC cycles found
            # (meridian basis (1,0) fallback will be used)
            self._fill_btn.setEnabled(True)
            if self._nc_cycle_vms:
                self._card.set_status(CardStatus.DONE)
            else:
                self._card.set_status(CardStatus.READY)
            self._update_summary()
            self.session_updated.emit(s)

            # Launch Weyl check automatically after NC search completes
            # (this will render the tables with compatibility info)
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
        """After NC search, run Weyl check for EACH NC cycle with basis change."""
        s = self._session
        # Check if we have index queries (needed for Weyl check)
        if not s.index_queries:
            print("[WEYL-NC] Skipping Weyl checks: no index queries yet")
            self._render_nc_tables()
            return

        # Build entries from index queries that have results
        entries = [
            (q.m_ext, q.e_ext, q.result)
            for q in s.index_queries
            if q.result is not None
        ]

        if not entries:
            print("[WEYL-NC] Skipping Weyl checks: no index query results")
            self._render_nc_tables()
            return

        # Clear previous results
        self._nc_weyl_results.clear()
        self._nc_weyl_workers.clear()

        # Import here to avoid circular dependencies
        from manifold_index.core import dehn_filling as _df_mod, neumann_zagier as _nz_mod

        print(f"[WEYL-NC] Launching Weyl checks for {len(self._nc_cycle_vms)} NC cycles")

        gen = self._session_gen
        cusp_idx = 0  # Assuming single cusp for now

        # Launch a WeylWorker for each NC cycle with basis change applied
        for vm in self._nc_cycle_vms:
            P, Q = vm.P, vm.Q
            print(f"[WEYL-NC] Cycle ({P},{Q}): applying basis change and recomputing index")

            try:
                # Apply basis change using this cycle as basis
                R, S = _df_mod.find_rs(P, Q)
                nz_nc = _nz_mod.apply_general_cusp_basis_change(
                    s.nz_data, cusp_idx, a=P, b=Q, c=-R, d=-S
                )

                # RECOMPUTE all index entries in the basis-changed NZ structure
                entries_nc = []
                for q in s.index_queries:
                    if q.result is not None:
                        # Recompute refined index on basis-changed NZ data
                        result_nc = ComputeService.compute_refined_index(
                            nz_nc, q.m_ext, q.e_ext, s.q_order_half
                        )
                        if result_nc is not None:
                            entries_nc.append((q.m_ext, q.e_ext, result_nc))

                print(f"[WEYL-NC] Cycle ({P},{Q}): recomputed {len(entries_nc)} entries, launching Weyl check")

                # Run Weyl check on basis-changed entries
                cycle_key = (P, Q)
                worker = WeylWorker(
                    entries      = entries_nc,  # Use basis-changed entries!
                    num_hard     = s.num_hard(),
                    q_order_half = s.q_order_half,
                    parent       = self,
                )
                # Store metadata: which cycle this worker is for
                worker._cycle_key = cycle_key  # type: ignore

                worker.finished.connect(
                    lambda p, ck=cycle_key, g=gen: self._on_weyl_for_nc_cycle_done(p, ck, g)
                )
                worker.error.connect(
                    lambda e, ck=cycle_key, g=gen: self._on_weyl_for_nc_cycle_error(e, ck, g)
                )
                self._nc_weyl_workers[cycle_key] = worker
                worker.start()
            except Exception as e:
                print(f"[WEYL-NC] Error setting up worker for cycle ({P},{Q}): {e}")
                self._nc_weyl_results[(P, Q)] = {"error": str(e)}

    def _on_weyl_for_nc_cycle_done(self, payload: dict, cycle_key: tuple, gen: int) -> None:
        """Handle Weyl check completion for one NC cycle."""
        if gen != self._session_gen:
            print(f"[WEYL-NC] Ignoring stale generation for cycle {cycle_key}")
            return

        P, Q = cycle_key
        if cycle_key in self._nc_weyl_workers:
            del self._nc_weyl_workers[cycle_key]

        ab = payload["ab_vectors"]
        adj_pass = payload.get("adjoint_is_pass")

        print(f"[WEYL-NC] Cycle ({P},{Q}) done: ab={'present' if ab else 'None'}, adj_pass={adj_pass}")

        # Store per-cycle result
        self._nc_weyl_results[cycle_key] = {
            "ab": ab,
            "adj_pass": adj_pass,
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
        """Update NC cycle VMs with per-cycle Weyl results."""
        for vm in self._nc_cycle_vms:
            cycle_key = (vm.P, vm.Q)
            if cycle_key in self._nc_weyl_results:
                result = self._nc_weyl_results[cycle_key]
                if "error" in result:
                    # Error occurred
                    vm.weyl_compatible = None
                    vm.adjoint_proj_pass = None
                    vm.weyl_a = None
                    vm.weyl_b = None
                else:
                    # Success - check both Weyl and q¹
                    ab = result.get("ab")
                    adj_pass = result.get("adj_pass")
                    vm.weyl_compatible = None if ab is None else True
                    vm.adjoint_proj_pass = adj_pass
                    # Store Weyl vectors
                    if ab is not None:
                        vm.weyl_a = list(ab.a) if hasattr(ab, 'a') else None
                        vm.weyl_b = list(ab.b) if hasattr(ab, 'b') else None
                    else:
                        vm.weyl_a = None
                        vm.weyl_b = None
            else:
                # Not yet computed
                vm.weyl_compatible = None
                vm.adjoint_proj_pass = None
                vm.weyl_a = None
                vm.weyl_b = None

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
        """Create per-cusp checkbox + NC cycle + slope input rows for multi-cusp filling."""
        # Clear existing rows
        for row in self._cusp_fill_rows:
            if row.get("widget"):
                row["widget"].setParent(None)  # type: ignore[arg-type]
        self._cusp_fill_rows.clear()

        # Build one row per cusp
        for cusp_idx in range(n_cusps):
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)

            # "Fill" checkbox
            fill_chk = QCheckBox("Fill")
            fill_chk.setChecked(True)  # Default: fill all cusps
            fill_chk.setFixedWidth(50)
            row_layout.addWidget(fill_chk)

            # Cusp label
            row_layout.addWidget(QLabel(f"C{cusp_idx}"))

            # NC cycle combo
            nc_combo = QComboBox()
            nc_combo.setMinimumWidth(100)
            row_layout.addWidget(nc_combo)

            # Filling slope
            slope = SlopeInput(label="→", require_coprime=True)
            row_layout.addWidget(slope)
            row_layout.addStretch(1)

            self._cusp_fill_layout.addWidget(row_widget)
            self._cusp_fill_rows.append({
                "widget": row_widget,
                "cusp_idx": cusp_idx,
                "fill_chk": fill_chk,
                "nc_combo": nc_combo,
                "slope": slope,
            })

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

    def _fill_active_edges(self) -> list[bool]:
        return [cb.isChecked() for cb in self._fill_edge_checkboxes]

    def _on_fill_preset_changed(self, text: str) -> None:
        """Apply preset and refresh display (no debounce — instant)."""
        for cb in self._fill_edge_checkboxes:
            cb.blockSignals(True)
            if text == "Full Refined":
                cb.setChecked(True)
                cb.setEnabled(False)
            elif text == "Unrefined":
                cb.setChecked(False)
                cb.setEnabled(False)
            else:  # Custom
                cb.setEnabled(True)
            cb.blockSignals(False)
        if text != "Custom":
            self._do_refresh_fill_display()

    def _on_fill_edge_toggle(self) -> None:
        """Called when user manually clicks an individual W checkbox."""
        if not self._session or not self._session.fill_queries:
            return
        # Switch preset label to "Custom"
        self._fill_preset_combo.blockSignals(True)
        self._fill_preset_combo.setCurrentText("Custom")
        self._fill_preset_combo.blockSignals(False)
        self._fill_refresh_timer.start()

    def _do_refresh_fill_display(self) -> None:
        """Re-render all fill rows with the current edge-toggle projection."""
        if not self._session:
            return
        inactive = [j for j, a in enumerate(self._fill_active_edges()) if not a]
        for idx, fq in enumerate(self._session.fill_queries):
            if fq.result is None:
                continue
            try:
                if fq.unrefined_fallback:
                    # For unrefined results, apply edge-toggle projection directly
                    projected = fq.result.collapse_eta_edges(inactive)
                    latex = format_filled_series_latex(
                        projected.series,
                        projected.num_hard,
                        projected.has_cusp_eta,
                        projected.num_cusp_eta,
                    )
                else:
                    projected = fq.result.collapse_eta_edges(inactive)
                    latex = format_filled_series_latex(
                        projected.series,
                        projected.num_hard,
                        projected.has_cusp_eta,
                        projected.num_cusp_eta,
                    )
            except Exception:
                latex = "—"
            self._fill_table.set_row_result(idx, latex, fq.source)
            # Yield to event loop every 50 rows to keep UI responsive
            # when re-projecting filled results after edge toggle.
            if (idx + 1) % 50 == 0:
                QCoreApplication.processEvents()

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
            # Suggest a fill slope ≠ NC cycle (filling at the NC cycle always
            # gives series={} by definition).
            self._suggest_fill_slope(vm)

    def _on_other_mode_changed(self, point_checked: bool) -> None:
        self._other_point_row.setVisible(point_checked)
        self._other_grid_row.setVisible(not point_checked)

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
            m_val = self._other_m.value()
            e_val = Fraction(int(self._other_e.value() * 2), 2)
            return [([m_val], [e_val])]
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

        Weyl check is run automatically after NC search.
        This method just determines refined vs unrefined based on saved Weyl results.
        """
        s = self._session
        if s.nz_data is None:
            return

        # Check if we're in multi-cusp mode
        n_cusps = s.manifold_data.num_cusps if s.manifold_data else 1
        if n_cusps > 1:
            self._on_multi_fill_clicked()
            return

        # Use saved Weyl results from NC search
        print(f"[FILL] weyl_checked={s.weyl_checked}, weyl_result={'present' if s.weyl_result else 'None'}")

        # Go directly to fill path with Weyl results
        self._launch_fill_path(ab=s.weyl_result, adj_pass=s.weyl_adjoint_pass)

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

        # Proceed to fill path with Weyl vectors
        self._launch_fill_path(ab=ab, adj_pass=adj_pass)

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

    def _launch_fill_path(self, ab, adj_pass) -> None:
        """Decide whether to use refined or unrefined fill.

        If ab is not None and adj_pass is True: use refined with incompat edges
        Otherwise: use unrefined
        """
        s = self._session
        if s.nz_data is None:
            return

        n_cusps = s.manifold_data.num_cusps if s.manifold_data else 1

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

        # Determine refined vs unrefined
        q1_passes = (adj_pass is True)
        use_refined = ab is not None and q1_passes

        print(f"[FILL] Deciding path: ab={'present' if ab else 'None'}, adj_pass={adj_pass}, use_refined={use_refined}")

        # Determine refined vs unrefined and launch appropriate workers
        # (Weyl status label was already set in _on_auto_weyl_done and will remain visible)
        if use_refined:
            print("[FILL] → Using REFINED fill")
            weyl_a = list(ab.a) if ab is not None else None
            weyl_b = list(ab.b) if ab is not None else None
            self._launch_refined_fill_workers(
                nc_P, nc_Q, user_P, user_Q, cusp_idx, charge_points,
                weyl_a, weyl_b, basis_label
            )
        else:
            print("[FILL] → Using UNREFINED fill")
            self._launch_unrefined_fill_workers(
                nc_P, nc_Q, user_P, user_Q, cusp_idx, charge_points, basis_label
            )

    def _launch_refined_fill_workers(
        self, nc_P: int, nc_Q: int, user_P: int, user_Q: int, cusp_idx: int,
        charge_points: list, weyl_a, weyl_b, basis_label: str
    ) -> None:
        """Launch FillWorker for each charge point (refined path)."""
        s = self._session
        n_cusps = s.manifold_data.num_cusps if s.manifold_data else 1

        # Clear old results before starting new computation
        self._fill_table.clear_rows()

        # Compute incompat edges if weyl_a is provided
        incompat_edges = []
        if weyl_a is not None:
            ab = type('obj', (object,), {'a': weyl_a, 'b': weyl_b})()
            incompat_edges = _compute_incompat_edges(ab)

        # Zero out incompat edges in weyl_a
        if incompat_edges and weyl_a is not None:
            weyl_a = list(weyl_a)
            for j in incompat_edges:
                if j < len(weyl_a):
                    weyl_a[j] = 0

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
            if n_cusps > 1:
                m_lbl = f"m={m_other[0]}" if m_other else ""
                e_val = float(e_other[0]) if e_other else 0.0
                e_lbl = f"e={e_val:g}"
                row_m = f"{m_lbl} {e_lbl}  {basis_label}"
                row_e = f"→({user_P},{user_Q})"
            else:
                row_m = basis_label
                row_e = f"slope ({user_P},{user_Q})"

            row = self._fill_table.add_row(row_m, row_e, "", "—")
            self._fill_table.set_row_computing(row)

            worker = FillWorker(
                nz_data      = s.nz_data,
                cusp_idx     = cusp_idx,
                nc_P         = nc_P,
                nc_Q         = nc_Q,
                user_P       = user_P,
                user_Q       = user_Q,
                m_other      = list(m_other),
                e_other      = list(e_other),
                q_order_half = s.q_order_half,
                weyl_a       = weyl_a,
                weyl_b       = weyl_b,
                manifold_name= s.manifold_name if s.manifold_name else "unknown",
                parent       = self,
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
        n_cusps = s.manifold_data.num_cusps if s.manifold_data else 1

        # Clear old results before starting new computation
        self._fill_table.clear_rows()

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
            if n_cusps > 1:
                m_lbl = f"m={m_other[0]}" if m_other else ""
                e_val = float(e_other[0]) if e_other else 0.0
                e_lbl = f"e={e_val:g}"
                row_m = f"{m_lbl} {e_lbl}  {basis_label}"
                row_e = f"→({user_P},{user_Q})"
            else:
                row_m = basis_label
                row_e = f"slope ({user_P},{user_Q})"

            row = self._fill_table.add_row(row_m, row_e, "", "—")
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
            series_latex = "$0$" if (result and result.is_zero) else str(result.series if result else "—")

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
        )
        s.fill_queries.append(fq)
        if s.stage < PipelineStage.FILLED:
            s.stage = PipelineStage.FILLED

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

            nc_idx = nc_combo.currentIndex()
            if nc_idx < 0 or nc_idx >= len(self._nc_cycle_vms):
                from manifold_index.viewmodels.advisory import Advisories
                self._card.set_advisories([
                    Advisories.warning(
                        "Incomplete selection",
                        f"Please select an NC cycle for Cusp {cusp_idx}."
                    )
                ])
                return

            nc_vm = self._nc_cycle_vms[nc_idx]
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

            weyl_a = list(s.weyl_result.a) if s.weyl_result is not None else None
            weyl_b = list(s.weyl_result.b) if s.weyl_result is not None else None

            cusp_specs.append({
                "cusp_idx": cusp_idx,
                "nc_P": nc_vm.P,
                "nc_Q": nc_vm.Q,
                "user_P": user_P,
                "user_Q": user_Q,
                "weyl_a": weyl_a,
                "weyl_b": weyl_b,
            })

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

        # All specs collected; launch worker
        cusp_strs = ", ".join(f"C{s['cusp_idx']}" for s in cusp_specs)
        slope_strs = ", ".join(f"({s['user_P']},{s['user_Q']})" for s in cusp_specs)
        row = self._fill_table.add_row(
            cusp_strs,
            slope_strs,
            "",
            "—"
        )
        self._fill_table.set_row_computing(row)

        self._fill_btn.setEnabled(False)
        self._fill_stop_btn.setEnabled(True)
        self._fill_progress.setRange(0, 0)  # indeterminate
        self._fill_progress.setVisible(True)
        self._fill_status.setText("Computing multi-cusp filling…")
        self._fill_status.setVisible(True)
        self._card.set_status(CardStatus.RUNNING)

        gen = self._session_gen
        worker = MultiFillWorker(
            nz_data        = s.nz_data,
            cusp_specs     = cusp_specs,
            q_order_half   = s.q_order_half,
            manifold_name  = s.manifold_name if s.manifold_name else "unknown",
            parent         = self,
        )
        worker.finished.connect(
            lambda p, w=worker, r=row, g=gen: self._on_multi_fill_finished(p, r, g, w)
        )
        worker.error.connect(lambda e, w=worker, r=row, g=gen: self._on_fill_error(e, r, g, w))
        self._fill_workers.append(worker)
        worker.start()

    def _on_multi_fill_finished(
        self, payload: dict, row: int, gen: int, worker: object = None
    ) -> None:
        """Handle completion of multi-cusp filling computation."""
        w = worker if worker is not None else self.sender()
        if w in self._fill_workers:
            self._fill_workers.remove(w)

        if gen != self._session_gen:
            return   # stale

        result = payload["result"]
        s = self._session
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
            series_latex = "$0$" if (result and result.is_zero) else "—"

        self._fill_table.set_row_result(row, series_latex, "computed")

        # Extract cusp_specs from payload
        cusp_specs = payload.get("cusp_specs", [])

        # Create MultiFillQuery
        mfq = MultiFillQuery(
            cusp_specs   = cusp_specs,
            q_order_half = s.q_order_half,
            result       = result,
            source       = "computed",
        )
        s.multi_fill_queries.append(mfq)

        if s.stage < PipelineStage.FILLED:
            s.stage = PipelineStage.FILLED

        self._fill_btn.setEnabled(True)
        self._fill_stop_btn.setEnabled(False)
        self._fill_progress.setVisible(False)
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
                    inactive = [j for j, a in enumerate(self._fill_active_edges()) if not a]
                    projected = fq.result.collapse_eta_edges(inactive)
                    series_latex = format_filled_series_latex(
                        projected.series,
                        projected.num_hard,
                        projected.has_cusp_eta,
                        projected.num_cusp_eta,
                    )
        except Exception:
            series_latex = "$0$" if (fq.result and fq.result.is_zero) else "—"

        # Format NC basis using v0.4 style: γ = P·γ + Q·δ
        nc_slope = format_slope_latex(fq.nc_P, fq.nc_Q, a=r"\gamma", b=r"\delta")
        nc_label = f"$\\gamma = {nc_slope}$"

        # Format user slope using v0.4 style: A·α + B·β
        user_slope = format_slope_latex(fq.user_P, fq.user_Q, a=r"\alpha", b=r"\beta")
        slope_label = f"${user_slope}$"

        # Format Weyl info string with improved notation
        weyl_info = ""
        if fq.weyl_a is not None and fq.weyl_b is not None:
            a_str = ", ".join(frac_to_latex(a) for a in fq.weyl_a)
            b_str = ", ".join(frac_to_latex(b) for b in fq.weyl_b)
            weyl_info = f"$a=({a_str}), b=({b_str})$"

        # Combine query metadata into a single label for better layout
        # Format: γ = ... / slope = ... / weyl info
        metadata = f"{nc_label} — {slope_label}"
        if weyl_info:
            metadata = f"{metadata}<br>{weyl_info}"

        row = self._fill_table.add_row(
            metadata,
            "",
            series_latex,
            fq.source,
        )
        self._fill_table.set_row_result(row, series_latex, fq.source)

    def _show_multi_fill_query(self, mfq: MultiFillQuery) -> None:
        """Display a multi-fill query result in the results table."""
        try:
            if mfq.result is None:
                series_latex = "—"
            else:
                inactive = [j for j, a in enumerate(self._fill_active_edges()) if not a]
                projected = mfq.result.collapse_eta_edges(inactive)
                series_latex = format_filled_series_latex(
                    projected.series,
                    projected.num_hard,
                    projected.has_cusp_eta,
                    projected.num_cusp_eta,
                )
        except Exception:
            series_latex = "$0$" if (mfq.result and mfq.result.is_zero) else "—"

        # Format row labels with improved LaTeX notation
        # Show all cusps being filled with their slopes in v0.4 style
        slope_parts = []
        for s in mfq.cusp_specs:
            cusp_label = f"C{s['cusp_idx']}"
            slope = format_slope_latex(s['user_P'], s['user_Q'], a=r"\alpha", b=r"\beta")
            slope_parts.append(f"${cusp_label}: {slope}$")

        metadata = "<br>".join(slope_parts)

        row = self._fill_table.add_row(
            metadata,
            "",
            series_latex,
            mfq.source,
        )
        self._fill_table.set_row_result(row, series_latex, mfq.source)
