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

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QGroupBox,
    QHBoxLayout, QLabel, QProgressBar, QPushButton, QRadioButton,
    QSpinBox, QVBoxLayout, QWidget,
)

from manifold_index.services.session import (
    FillQuery, NCCycleSet, PipelineStage, Session,
)
from manifold_index.viewmodels.advisory import CardStatus
from manifold_index.viewmodels.filling_vm import (
    build_nc_cycle_vm, NCCycleViewModel,
)
from manifold_index.formatters.filling_fmt import (
    format_slope_latex, format_filled_series_latex,
    format_nc_cycle_table_html,
)
from manifold_index.app.widgets.collapsible_card import CollapsibleCard
from manifold_index.app.widgets.series_table import SeriesTable
from manifold_index.app.widgets.math_view import MathView
from manifold_index.app.widgets.slope_input import SlopeInput
from manifold_index.app.workers.nc_search_worker import NCSearchWorker
from manifold_index.app.workers.fill_worker import FillWorker
from manifold_index.app.workers.weyl_worker import WeylWorker
from manifold_index.viewmodels.index_vm import build_weyl_vm
from manifold_index.formatters.weyl_fmt import format_weyl_html


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
        self._weyl_worker: WeylWorker | None = None
        self._nc_worker_progress: dict[int, tuple[int, int]] = {}
        self._fill_current_row: int | None = None
        self._fill_grid_total: int = 0
        self._fill_grid_done: int = 0

        self._card = CollapsibleCard(3, "Dehn Filling", parent=self)
        self._card.set_status(CardStatus.LOCKED)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._card)

        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(10)

        # ── Weyl Check ───────────────────────────────────────────────
        weyl_box = QGroupBox("Weyl Check")
        weyl_layout = QVBoxLayout(weyl_box)

        weyl_btn_row = QHBoxLayout()
        self._weyl_btn = QPushButton("Run Weyl Check")
        self._weyl_btn.setProperty("class", "primary")
        self._weyl_btn.clicked.connect(self._on_weyl_clicked)
        weyl_btn_row.addWidget(self._weyl_btn)
        self._weyl_stop_btn = QPushButton("Stop")
        self._weyl_stop_btn.setProperty("class", "secondary")
        self._weyl_stop_btn.setEnabled(False)
        self._weyl_stop_btn.clicked.connect(self._on_weyl_stop_clicked)
        weyl_btn_row.addWidget(self._weyl_stop_btn)
        weyl_btn_row.addStretch(1)
        weyl_layout.addLayout(weyl_btn_row)

        self._weyl_progress = QProgressBar()
        self._weyl_progress.setRange(0, 0)   # indeterminate
        self._weyl_progress.setVisible(False)
        weyl_layout.addWidget(self._weyl_progress)

        self._weyl_status = QLabel()
        self._weyl_status.setProperty("class", "muted")
        self._weyl_status.setVisible(False)
        weyl_layout.addWidget(self._weyl_status)

        self._weyl_view = MathView(min_h=80)
        self._weyl_view.setVisible(False)
        weyl_layout.addWidget(self._weyl_view)

        bl.addWidget(weyl_box)

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

        # Manual basis cycle row (shown when user wants unrefined filling)
        manual_chk_row = QHBoxLayout()
        self._manual_basis_chk = QCheckBox("Manual basis cycle (unrefined)")
        self._manual_basis_chk.setToolTip(
            "When no NC cycles are found, or to override, enter any (P,Q) as\n"
            "the basis cycle and compute Dehn filling without NC-cycle restriction."
        )
        self._manual_basis_chk.toggled.connect(self._on_manual_basis_toggled)
        manual_chk_row.addWidget(self._manual_basis_chk)
        manual_chk_row.addStretch(1)
        fill_layout.addLayout(manual_chk_row)

        self._manual_basis_widget = QWidget()
        mbw_layout = QHBoxLayout(self._manual_basis_widget)
        mbw_layout.setContentsMargins(0, 0, 0, 0)
        mbw_layout.addWidget(QLabel("  Basis cycle (P, Q):"))
        self._manual_nc_P = QSpinBox()
        self._manual_nc_P.setRange(-999, 999)
        self._manual_nc_P.setValue(1)
        self._manual_nc_P.setFixedWidth(65)
        mbw_layout.addWidget(self._manual_nc_P)
        mbw_layout.addWidget(QLabel(","))
        self._manual_nc_Q = QSpinBox()
        self._manual_nc_Q.setRange(-999, 999)
        self._manual_nc_Q.setValue(0)
        self._manual_nc_Q.setFixedWidth(65)
        mbw_layout.addWidget(self._manual_nc_Q)
        mbw_layout.addStretch(1)
        self._manual_basis_widget.setVisible(False)
        fill_layout.addWidget(self._manual_basis_widget)

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
        self._cusp_combo.blockSignals(True)
        self._cusp_combo.clear()
        for i in range(n_cusps):
            self._cusp_combo.addItem(f"Cusp {i}", i)
        self._cusp_combo.blockSignals(False)
        self._rebuild_nc_tables(n_cusps)
        self._cache_chk.setChecked(
            session.cache_status.get("nc", {}).get("available", False)
        )
        # Show unfilled-cusp charge controls only for multi-cusp manifolds
        self._other_box.setVisible(n_cusps > 1)
        # Rebuild edge-toggle checkboxes for filled results
        n_hard = session.num_hard() if session.manifold_data else 0
        self._fill_ref_box.setVisible(n_hard > 0)
        self._rebuild_fill_edge_toggles(n_hard)

    def lock(self) -> None:
        # Disconnect and abandon any running workers so their stale signals
        # cannot corrupt the next session's state.
        self._abandon_nc_workers()
        self._abandon_fill_workers()
        # Abandon Weyl worker
        if self._weyl_worker is not None:
            try:
                self._weyl_worker.finished.disconnect()
                self._weyl_worker.error.disconnect()
            except RuntimeError:
                pass
            self._weyl_worker = None
        self._card.set_status(CardStatus.LOCKED)
        self._card.collapse()
        self._fill_table.clear_rows()
        for mv in self._nc_table_views:
            mv.setVisible(False)
        for lbl in self._nc_table_labels:
            lbl.setVisible(False)
        self._nc_cycle_vms.clear()
        self._fill_btn.setEnabled(False)
        # Reset manual-basis mode
        self._manual_basis_chk.setChecked(False)
        self._manual_basis_widget.setVisible(False)
        # Reset refinement toggles
        self._fill_refresh_timer.stop()
        self._fill_ref_box.setVisible(False)
        for cb in self._fill_edge_checkboxes:
            cb.setParent(None)  # type: ignore[arg-type]
        self._fill_edge_checkboxes.clear()
        self._fill_preset_combo.blockSignals(True)
        self._fill_preset_combo.setCurrentIndex(0)
        self._fill_preset_combo.blockSignals(False)
        # Reset Weyl UI
        self._weyl_btn.setEnabled(True)
        self._weyl_stop_btn.setEnabled(False)
        self._weyl_progress.setVisible(False)
        self._weyl_status.setVisible(False)
        self._weyl_view.setVisible(False)
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
        # Restore Weyl result if available
        if session.weyl_checked and session.weyl_result is not None:
            vm = build_weyl_vm(
                session.weyl_result,
                session.num_hard(),
                adjoint_value=None,
                adjoint_passed=session.weyl_adjoint_pass,
            )
            html = "<h3>Weyl Check</h3>\n" + format_weyl_html(vm)
            self._weyl_view.set_html(html)
            self._weyl_view.setVisible(True)
        if session.nc_cycles:
            self._rebuild_nc_vms_from_session()
            self._render_nc_tables()
            self._rebuild_nc_combo()
            self._fill_btn.setEnabled(True)
        for fq in session.fill_queries:
            self._show_fill_query(fq)

    def trigger_find_nc(self) -> None:
        """Public: programmatically trigger NC cycle search (used by Run All)."""
        self._on_find_nc_clicked()

    def trigger_stop_nc(self) -> None:
        """Public: programmatically trigger NC stop (used by Run All Stop)."""
        self._on_nc_stop_clicked()

    # ------------------------------------------------------------------
    # Internal — worker lifecycle helpers
    # ------------------------------------------------------------------

    def _abandon_nc_workers(self) -> None:
        """Disconnect signals from all NC workers and let them finish silently."""
        for w in self._nc_workers:
            try:
                w.finished.disconnect()
                w.error.disconnect()
                w.status.disconnect()
            except RuntimeError:
                pass  # already disconnected
        self._nc_workers.clear()

    def _abandon_fill_workers(self) -> None:
        """Disconnect signals from all fill workers and let them finish silently."""
        for w in self._fill_workers:
            try:
                w.finished.disconnect()
                w.error.disconnect()
            except RuntimeError:
                pass
        self._fill_workers.clear()

    # ------------------------------------------------------------------
    # Internal — NC cycle search
    # ------------------------------------------------------------------

    def _on_find_nc_clicked(self) -> None:
        s = self._session
        if s.stage < PipelineStage.LOADED or s.nz_data is None:
            return

        p_half  = self._p_range_spin.value()
        q_half  = self._q_range_spin.value()
        p_range = (-p_half, p_half)
        q_range = (-q_half, q_half)
        use_cache = self._cache_chk.isChecked()

        n_cusps = (
            self._session.manifold_data.num_cusps
            if self._session.manifold_data else 1
        )
        self._nc_search_btn.setEnabled(False)
        self._nc_stop_btn.setEnabled(True)
        self._nc_worker_progress.clear()
        self._nc_progress.setRange(0, 0)   # indeterminate until first progress signal
        self._nc_progress.setVisible(True)
        self._nc_status.setText(f"Searching NC cycles for {n_cusps} cusp(s)…")
        self._nc_status.setVisible(True)
        self._card.set_status(CardStatus.RUNNING)
        self._nc_cycle_vms.clear()

        gen = self._session_gen          # capture generation at launch time
        for i in range(n_cusps):
            worker = NCSearchWorker(
                nz_data       = s.nz_data,
                cusp_idx      = i,
                p_range       = p_range,
                q_range       = q_range,
                q_order_half  = s.q_order_half,
                manifold_name = s.manifold_name,
                use_cache     = use_cache,
                parent        = self,
            )
            worker.status.connect(self._nc_status.setText)
            worker.progress.connect(
                lambda d, t, ci=i: self._on_nc_progress(d, t, ci)
            )
            worker.finished.connect(
                lambda p, w=worker, ci=i, g=gen: self._on_nc_finished(p, ci, g, w)
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
            self._render_nc_tables()
            self._rebuild_nc_combo()
            if self._nc_cycle_vms:
                self._card.set_status(CardStatus.DONE)
                self._fill_btn.setEnabled(True)
            else:
                self._card.set_status(CardStatus.ERROR)
                from manifold_index.viewmodels.advisory import Advisories
                self._card.set_advisories([Advisories.C1()])
            self._update_summary()
            self.session_updated.emit(s)

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

    # ------------------------------------------------------------------
    # Internal — Weyl check
    # ------------------------------------------------------------------

    def _on_weyl_clicked(self) -> None:
        s = self._session
        if not s.index_queries:
            self._weyl_status.setText("No index queries to check. Run index computation first.")
            self._weyl_status.setVisible(True)
            return
        entries = [
            (q.m_ext, q.e_ext, q.result)
            for q in s.index_queries
            if q.result is not None
        ]
        self._weyl_btn.setEnabled(False)
        self._weyl_stop_btn.setEnabled(True)
        self._weyl_progress.setRange(0, 0)   # indeterminate
        self._weyl_progress.setVisible(True)
        self._weyl_status.setText("Running Weyl check…")
        self._weyl_status.setVisible(True)
        self._weyl_view.setVisible(False)
        self._card.set_status(CardStatus.RUNNING)

        self._weyl_worker = WeylWorker(
            entries      = entries,
            num_hard     = s.num_hard(),
            q_order_half = s.q_order_half,
            parent       = self,
        )
        gen = self._session_gen
        self._weyl_worker.finished.connect(lambda p, g=gen: self._on_weyl_finished(p, g))
        self._weyl_worker.error.connect(self._on_weyl_error)
        self._weyl_worker.start()

    def _on_weyl_stop_clicked(self) -> None:
        """Abandon the running Weyl worker."""
        if self._weyl_worker is not None:
            try:
                self._weyl_worker.finished.disconnect()
                self._weyl_worker.error.disconnect()
            except RuntimeError:
                pass
            self._weyl_worker = None
        self._weyl_btn.setEnabled(True)
        self._weyl_stop_btn.setEnabled(False)
        self._weyl_progress.setVisible(False)
        self._weyl_status.setText("Stopped.")
        self._weyl_status.setVisible(True)
        self._card.set_status(CardStatus.READY)

    def _on_weyl_finished(self, payload: dict, gen: int) -> None:
        if gen != self._session_gen:
            return   # stale
        self._weyl_worker = None
        self._weyl_btn.setEnabled(True)
        self._weyl_stop_btn.setEnabled(False)
        self._weyl_progress.setVisible(False)
        self._weyl_status.setVisible(False)

        ab        = payload["ab_vectors"]
        adj_pass  = payload.get("adjoint_is_pass")
        adj_value = payload.get("adjoint_value")
        s = self._session
        s.weyl_result       = ab
        s.weyl_adjoint_pass = adj_pass
        s.weyl_checked      = True

        vm = build_weyl_vm(
            ab, s.num_hard(),
            adjoint_value=float(adj_value) if adj_value is not None else None,
            adjoint_passed=adj_pass,
        )
        self._card.set_advisories(vm.advisories)

        if ab is not None:
            html = "<h3>Weyl Check</h3>\n" + format_weyl_html(vm)
        else:
            html = "<h3>Weyl Check</h3><p class='warn'>⚠ Could not extract Weyl vectors</p>"
        self._weyl_view.set_loading(False)
        self._weyl_view.set_html(html)
        self._weyl_view.setVisible(True)
        self._card.set_status(CardStatus.DONE)
        self._update_summary()
        self.session_updated.emit(s)

    def _on_weyl_error(self, msg: str) -> None:
        self._weyl_worker = None
        self._weyl_btn.setEnabled(True)
        self._weyl_stop_btn.setEnabled(False)
        self._weyl_progress.setVisible(False)
        self._weyl_status.setVisible(False)
        self._weyl_view.set_loading(False)
        self._weyl_view.set_html(f"<p class='warn'>Weyl error: {msg}</p>")
        self._weyl_view.setVisible(True)
        self._card.set_status(CardStatus.ERROR)

    # ------------------------------------------------------------------
    # Internal — NC stop and progress
    # ------------------------------------------------------------------

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
        self._fill_grid_total = 0
        self._fill_grid_done = 0
        self._card.set_status(CardStatus.READY)

    def _on_clear_results(self) -> None:
        """Clear all rows from the filling results table and session fill queries."""
        self._fill_table.clear_rows()
        if self._session is not None:
            self._session.fill_queries.clear()
            self._update_summary()

    def _rebuild_nc_tables(self, n_cusps: int) -> None:
        """Create/recreate one MathView per cusp inside the NC Cycles group box."""
        for lbl in self._nc_table_labels:
            lbl.setParent(None)  # type: ignore[arg-type]
        self._nc_table_labels.clear()
        for mv in self._nc_table_views:
            mv.setParent(None)  # type: ignore[arg-type]
        self._nc_table_views.clear()
        for i in range(n_cusps):
            label = QLabel(f"Cusp {i}")
            label.setProperty("class", "secondary")
            self._nc_table_layout.addWidget(label)
            self._nc_table_labels.append(label)
            mv = MathView(min_h=60)
            mv.setVisible(False)
            self._nc_table_layout.addWidget(mv)
            self._nc_table_views.append(mv)

    def _render_nc_tables(self) -> None:
        """Re-render each per-cusp MathView from the current NC cycle VMs."""
        # Group VMs by cusp index
        by_cusp: dict[int, list] = {}
        for vm in self._nc_cycle_vms:
            by_cusp.setdefault(vm.cusp_idx, []).append(vm)

        for i, mv in enumerate(self._nc_table_views):
            cusp_vms = by_cusp.get(i, [])
            if cusp_vms:
                html = format_nc_cycle_table_html(cusp_vms)
                mv.set_html(html)
                mv.setVisible(True)
            else:
                mv.set_html('<p class="muted">No non-closable cycles found.</p>')
                mv.setVisible(True)

    def _rebuild_nc_vms_from_session(self) -> None:
        self._nc_cycle_vms.clear()
        for ncs in self._session.nc_cycles:
            for cyc in ncs.cycles:
                P = int(cyc.P) if hasattr(cyc, "P") else 0
                Q = int(cyc.Q) if hasattr(cyc, "Q") else 0
                try:
                    sl = format_slope_latex(P, Q)
                except Exception:
                    sl = f"({P},{Q})"
                vm = build_nc_cycle_vm(
                    cusp_idx=ncs.cusp_idx, P=P, Q=Q,
                    slope_latex=sl, source=ncs.source,
                )
                self._nc_cycle_vms.append(vm)

    def _rebuild_nc_combo(self) -> None:
        self._nc_combo.blockSignals(True)
        self._nc_combo.clear()
        for i, vm in enumerate(self._nc_cycle_vms):
            self._nc_combo.addItem(f"C{vm.cusp_idx}: ({vm.P},{vm.Q})", i)
        self._nc_combo.blockSignals(False)
        # After (re)populating, suggest a fill slope that differs from the first
        # NC cycle so the user doesn't accidentally fill at the NC cycle itself.
        if self._nc_cycle_vms:
            self._suggest_fill_slope(self._nc_cycle_vms[0])

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

    # ------------------------------------------------------------------
    # Internal — manual basis cycle toggle
    # ------------------------------------------------------------------

    def _on_manual_basis_toggled(self, checked: bool) -> None:
        """Show/hide the manual (P,Q) spinboxes and update fill-button state."""
        self._manual_basis_widget.setVisible(checked)
        if checked:
            # Manual mode: always allow filling (basis is whatever the user enters)
            self._fill_btn.setEnabled(True)
        else:
            # Back to NC mode: only enable fill if NC cycles are loaded
            has_nc = bool(self._nc_cycle_vms) and self._nc_combo.currentIndex() >= 0
            self._fill_btn.setEnabled(has_nc)

    # ------------------------------------------------------------------
    # Internal — other-cusp mode toggle
    # ------------------------------------------------------------------

    def _on_other_mode_changed(self, point_checked: bool) -> None:
        self._other_point_row.setVisible(point_checked)
        self._other_grid_row.setVisible(not point_checked)

    # ------------------------------------------------------------------
    # Internal — fill computation
    # ------------------------------------------------------------------

    def _other_charge_points(self, n_cusps: int) -> list[tuple[list[int], list[Fraction]]]:
        """Return the list of (m_other, e_other) pairs to compute.

        Single-cusp: always [([], [])].
        Multi-cusp point mode: one pair from the spinboxes.
        Multi-cusp grid mode: all (m, e) on the Cartesian grid.
        """
        n_other = max(0, n_cusps - 1)
        if n_other == 0:
            return [([], [])]

        if not self._other_box.isVisible() or self._other_point.isChecked():
            m_val = self._other_m.value()
            e_val = Fraction(self._other_e.value()).limit_denominator(2)
            # For now, apply the same (m, e) to all other cusps
            return [([m_val] * n_other, [e_val] * n_other)]

        # Grid mode
        m_lo = self._other_m_min.value()
        m_hi = self._other_m_max.value()
        e_lo = Fraction(self._other_e_min.value()).limit_denominator(2)
        e_hi = Fraction(self._other_e_max.value()).limit_denominator(2)
        e_step = Fraction(1, 2)
        points = []
        m = m_lo
        while m <= m_hi:
            e = e_lo
            while e <= e_hi:
                points.append(([m] * n_other, [e] * n_other))
                e += e_step
            m += 1
        return points if points else [([0] * n_other, [Fraction(0)] * n_other)]

    def _on_fill_clicked(self) -> None:
        s = self._session
        if s.nz_data is None:
            return

        # Determine basis cycle (P, Q) — either from the NC combo or from the
        # manual spinboxes when the user has no NC cycles (or wants to override).
        if self._manual_basis_chk.isChecked():
            nc_P = self._manual_nc_P.value()
            nc_Q = self._manual_nc_Q.value()
            basis_label = f"Manual ({nc_P},{nc_Q})"
        else:
            nc_idx = self._nc_combo.currentIndex()
            if nc_idx < 0 or nc_idx >= len(self._nc_cycle_vms):
                return
            nc_vm = self._nc_cycle_vms[nc_idx]
            nc_P, nc_Q = nc_vm.P, nc_vm.Q
            basis_label = f"NC ({nc_P},{nc_Q})"

        user_P, user_Q = self._fill_slope.get_slope()
        if not self._fill_slope.is_valid():
            return

        cusp_idx = self._cusp_combo.currentData() or 0
        n_cusps  = s.manifold_data.num_cusps if s.manifold_data else 1

        charge_points = self._other_charge_points(n_cusps)
        weyl_a = list(s.weyl_result.a) if s.weyl_result is not None else None
        weyl_b = list(s.weyl_result.b) if s.weyl_result is not None else None

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
            # Row label: for multi-cusp show (m,e) alongside basis; for single
            # cusp show basis label / filling slope.
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
                       ci=cusp_idx, mo=list(m_other), eo=list(e_other), g=gen:
                    self._on_fill_finished(p, r, nP, nQ, uP, uQ, ci, mo, eo, g, w)
            )
            worker.error.connect(lambda e, w=worker, r=row, g=gen: self._on_fill_error(e, r, g, w))
            self._fill_workers.append(worker)
            worker.start()

    def _on_fill_finished(
        self, payload: dict, row: int,
        nc_P: int, nc_Q: int, user_P: int, user_Q: int,
        cusp_idx: int, m_other: list, e_other: list, gen: int,
        worker: object = None,
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
            incompat_edges = [],
            source       = "computed",
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
            self._fill_current_row = None
        self._fill_table.set_row_result(row, f"Error: {msg}", "—")
        logging.warning("FillWorker error: %s", msg)
        self._card.set_status(CardStatus.ERROR)

    def _show_fill_query(self, fq: FillQuery) -> None:
        try:
            if fq.result is None:
                series_latex = "—"
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
        row = self._fill_table.add_row(
            f"NC ({fq.nc_P},{fq.nc_Q})",
            f"slope ({fq.user_P},{fq.user_Q})",
            "",
            fq.source,
        )
        self._fill_table.set_row_result(row, series_latex, fq.source)

    # ------------------------------------------------------------------
    # Internal — filled results edge toggles
    # ------------------------------------------------------------------

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
        for i, fq in enumerate(self._session.fill_queries):
            if fq.result is None:
                continue
            try:
                projected = fq.result.collapse_eta_edges(inactive)
                latex = format_filled_series_latex(
                    projected.series,
                    projected.num_hard,
                    projected.has_cusp_eta,
                    projected.num_cusp_eta,
                )
            except Exception:
                latex = "—"
            self._fill_table.set_row_result(i, latex, fq.source)

    def _update_summary(self) -> None:
        n_nc   = len(self._nc_cycle_vms)
        n_fill = len(self._session.fill_queries)
        weyl   = "✓" if self._session.weyl_checked else "not run"
        self._card.set_summary(
            f"Weyl: {weyl}  ·  {n_nc} NC cycle{'s' if n_nc != 1 else ''}  ·  "
            f"{n_fill} filled quer{'ies' if n_fill != 1 else 'y'}"
        )
