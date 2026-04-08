"""app/pipeline/filling_card.py — Card ③: Dehn Filling.

BLUEPRINT §10.4.

Phase A: Find NC cycles (NCSearchWorker per cusp).
Phase B: Compute filled index (FillWorker per query).

Layout
------
  • NC source row: p/q range spinboxes + "Use Cache" checkbox
  • Cusp rows: one SlopeInput per cusp (user Dehn-filling slope)
  • "Find NC Cycles" button → NCSearchWorker
  • NC cycle table (MathView)
  • Filled query form: NC cycle selector + "Compute Filling" button
  • SeriesTable for filling results
"""

from __future__ import annotations

import logging
from fractions import Fraction

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QLabel,
    QPushButton, QSpinBox, QVBoxLayout, QWidget,
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
        self._session_gen: int = 0   # incremented on each unlock(); guards stale signals

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
        self._p_range_spin.setValue(5)
        self._p_range_spin.setFixedWidth(55)
        range_row.addWidget(self._p_range_spin)
        range_row.addWidget(QLabel("  Q range ±"))
        self._q_range_spin = QSpinBox()
        self._q_range_spin.setRange(0, 50)
        self._q_range_spin.setValue(5)
        self._q_range_spin.setFixedWidth(55)
        range_row.addWidget(self._q_range_spin)
        self._cache_chk = QCheckBox("Use cache")
        range_row.addWidget(self._cache_chk)
        range_row.addStretch(1)
        search_layout.addLayout(range_row)

        # Cusp slope inputs (rebuilt on unlock)
        self._slope_inputs: list[SlopeInput] = []
        self._cusp_input_container = QWidget()
        self._cusp_input_layout = QVBoxLayout(self._cusp_input_container)
        self._cusp_input_layout.setContentsMargins(0, 0, 0, 0)
        search_layout.addWidget(self._cusp_input_container)

        self._nc_search_btn = QPushButton("Find NC Cycles")
        self._nc_search_btn.setProperty("class", "primary")
        self._nc_search_btn.clicked.connect(self._on_find_nc_clicked)
        search_layout.addWidget(self._nc_search_btn)

        self._nc_status = QLabel()
        self._nc_status.setProperty("class", "muted")
        self._nc_status.setVisible(False)
        search_layout.addWidget(self._nc_status)

        bl.addWidget(search_box)

        # ── NC cycle table ────────────────────────────────────────────
        nc_table_box = QGroupBox("NC Cycles")
        nc_table_layout = QVBoxLayout(nc_table_box)
        self._nc_table_view = MathView(min_h=80)
        self._nc_table_view.setVisible(False)
        nc_table_layout.addWidget(self._nc_table_view)
        bl.addWidget(nc_table_box)

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

        self._fill_btn = QPushButton("Compute Filling")
        self._fill_btn.setProperty("class", "primary")
        self._fill_btn.setEnabled(False)
        self._fill_btn.clicked.connect(self._on_fill_clicked)
        fill_layout.addWidget(self._fill_btn)

        self._fill_status = QLabel()
        self._fill_status.setProperty("class", "muted")
        self._fill_status.setVisible(False)
        fill_layout.addWidget(self._fill_status)

        bl.addWidget(fill_box)

        # ── Filling results table ─────────────────────────────────────
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
        self._rebuild_cusp_inputs()
        self._cache_chk.setChecked(
            session.cache_status.get("nc", {}).get("available", False)
        )

    def lock(self) -> None:
        # Disconnect and abandon any running workers so their stale signals
        # cannot corrupt the next session's state.
        self._abandon_nc_workers()
        self._abandon_fill_workers()
        self._card.set_status(CardStatus.LOCKED)
        self._card.collapse()
        self._fill_table.clear_rows()
        self._nc_table_view.setVisible(False)
        self._nc_cycle_vms.clear()
        self._fill_btn.setEnabled(False)

    def refresh(self, session: Session) -> None:
        self._session = session
        if session.nc_cycles:
            self._rebuild_nc_vms_from_session()
            self._render_nc_table()
            self._rebuild_nc_combo()
            self._fill_btn.setEnabled(True)
        for fq in session.fill_queries:
            self._show_fill_query(fq)

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

    def _rebuild_cusp_inputs(self) -> None:
        for si in self._slope_inputs:
            si.setParent(None)  # type: ignore[arg-type]
        self._slope_inputs.clear()

        n_cusps = (
            self._session.manifold_data.num_cusps
            if self._session.manifold_data else 1
        )
        for i in range(n_cusps):
            si = SlopeInput(label=f"Cusp {i} slope:")
            si.set_slope(1, 0)
            self._cusp_input_layout.addWidget(si)
            self._slope_inputs.append(si)

        self._cusp_combo.blockSignals(True)
        self._cusp_combo.clear()
        for i in range(n_cusps):
            self._cusp_combo.addItem(f"Cusp {i}", i)
        self._cusp_combo.blockSignals(False)

    def _on_find_nc_clicked(self) -> None:
        s = self._session
        if s.stage < PipelineStage.LOADED or s.nz_data is None:
            return

        p_half  = self._p_range_spin.value()
        q_half  = self._q_range_spin.value()
        p_range = (-p_half, p_half)
        q_range = (-q_half, q_half)
        use_cache = self._cache_chk.isChecked()

        n_cusps = len(self._slope_inputs)
        self._nc_search_btn.setEnabled(False)
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
            worker.finished.connect(
                lambda p, ci=i, g=gen: self._on_nc_finished(p, ci, g)
            )
            worker.error.connect(
                lambda e, ci=i, g=gen: self._on_nc_error(e, ci, g)
            )
            self._nc_workers.append(worker)
            worker.start()

    def _on_nc_finished(self, payload: dict, cusp_idx: int, gen: int) -> None:
        if gen != self._session_gen:
            return   # stale: a new manifold was loaded since this worker launched
        # Remove the completed worker from the tracking list
        sender = self.sender()
        if sender in self._nc_workers:
            self._nc_workers.remove(sender)

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
                source=ncs.source,
                slope_latex=sl,
            )
            self._nc_cycle_vms.append(vm)

        running = [w for w in self._nc_workers if w.isRunning()]
        if not running:
            self._nc_search_btn.setEnabled(True)
            self._nc_status.setVisible(False)
            self._nc_workers.clear()
            self._render_nc_table()
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

    def _on_nc_error(self, msg: str, cusp_idx: int, gen: int) -> None:
        if gen != self._session_gen:
            return
        sender = self.sender()
        if sender in self._nc_workers:
            self._nc_workers.remove(sender)
        # Only re-enable controls if no other NC workers are still running
        if not any(w.isRunning() for w in self._nc_workers):
            self._nc_search_btn.setEnabled(True)
            self._nc_status.setVisible(False)
            self._nc_workers.clear()
        logging.warning("NCSearchWorker cusp %d error: %s", cusp_idx, msg)
        self._card.set_status(CardStatus.ERROR)

    def _render_nc_table(self) -> None:
        if not self._nc_cycle_vms:
            self._nc_table_view.setVisible(False)
            return
        html = format_nc_cycle_table_html(self._nc_cycle_vms)
        self._nc_table_view.set_html(html)
        self._nc_table_view.setVisible(True)

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

    def _on_nc_selected(self, idx: int) -> None:
        if 0 <= idx < len(self._nc_cycle_vms):
            vm = self._nc_cycle_vms[idx]
            self._fill_slope.set_slope(vm.P, vm.Q)

    # ------------------------------------------------------------------
    # Internal — fill computation
    # ------------------------------------------------------------------

    def _on_fill_clicked(self) -> None:
        s = self._session
        if s.nz_data is None:
            return

        nc_idx = self._nc_combo.currentIndex()
        if nc_idx < 0 or nc_idx >= len(self._nc_cycle_vms):
            return

        nc_vm  = self._nc_cycle_vms[nc_idx]
        user_P, user_Q = self._fill_slope.get_slope()
        if not self._fill_slope.is_valid():
            return

        cusp_idx = self._cusp_combo.currentData() or 0
        n_cusps  = s.manifold_data.num_cusps if s.manifold_data else 1
        m_other  = [0] * max(0, n_cusps - 1)
        e_other  = [Fraction(0)] * max(0, n_cusps - 1)

        weyl_a = list(s.weyl_result.a) if s.weyl_result is not None else None
        weyl_b = list(s.weyl_result.b) if s.weyl_result is not None else None

        self._fill_btn.setEnabled(False)
        self._fill_status.setText("Computing…")
        self._fill_status.setVisible(True)
        self._card.set_status(CardStatus.RUNNING)

        row = self._fill_table.add_row(
            f"NC ({nc_vm.P},{nc_vm.Q})",
            f"slope ({user_P},{user_Q})",
            "",
            "—",
        )
        self._fill_table.set_row_computing(row)

        gen = self._session_gen          # capture for stale-signal guard
        worker = FillWorker(
            nz_data      = s.nz_data,
            cusp_idx     = cusp_idx,
            nc_P         = nc_vm.P,
            nc_Q         = nc_vm.Q,
            user_P       = user_P,
            user_Q       = user_Q,
            m_other      = m_other,
            e_other      = e_other,
            q_order_half = s.q_order_half,
            weyl_a       = weyl_a,
            weyl_b       = weyl_b,
            parent       = self,
        )
        worker.finished.connect(
            lambda p, r=row, nv=nc_vm, uP=user_P, uQ=user_Q, ci=cusp_idx,
                   mo=m_other, eo=e_other, g=gen:
                self._on_fill_finished(p, r, nv, uP, uQ, ci, mo, eo, g)
        )
        worker.error.connect(lambda e, r=row, g=gen: self._on_fill_error(e, r, g))
        self._fill_workers.append(worker)
        worker.start()

    def _on_fill_finished(
        self, payload: dict, row: int,
        nc_vm: NCCycleViewModel, user_P: int, user_Q: int,
        cusp_idx: int, m_other: list, e_other: list, gen: int,
    ) -> None:
        sender = self.sender()
        if sender in self._fill_workers:
            self._fill_workers.remove(sender)

        if gen != self._session_gen:
            return   # stale result — manifold was reloaded

        self._fill_btn.setEnabled(True)
        self._fill_status.setVisible(False)

        result = payload["result"]
        p      = payload["p"]
        q      = payload["q"]
        s      = self._session

        try:
            series_latex = format_filled_series_latex(result, s.num_hard(), False, s.q_order_half)
        except Exception:
            series_latex = str(result) if result else "0"

        self._fill_table.set_row_result(row, series_latex, "computed")

        fq = FillQuery(
            cusp_idx     = cusp_idx,      # the cusp actually used in computation
            nc_P         = nc_vm.P,
            nc_Q         = nc_vm.Q,
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

    def _on_fill_error(self, msg: str, row: int, gen: int) -> None:
        sender = self.sender()
        if sender in self._fill_workers:
            self._fill_workers.remove(sender)
        if gen != self._session_gen:
            return
        self._fill_btn.setEnabled(True)
        self._fill_status.setVisible(False)
        self._fill_table.set_row_result(row, f"Error: {msg}", "—")
        logging.warning("FillWorker error: %s", msg)
        self._card.set_status(CardStatus.ERROR)

    def _show_fill_query(self, fq: FillQuery) -> None:
        try:
            series_latex = format_filled_series_latex(
                fq.result, self._session.num_hard(), False, fq.q_order_half
            )
        except Exception:
            series_latex = "—"
        row = self._fill_table.add_row(
            f"NC ({fq.nc_P},{fq.nc_Q})",
            f"slope ({fq.user_P},{fq.user_Q})",
            "",
            fq.source,
        )
        self._fill_table.set_row_result(row, series_latex, fq.source)

    def _update_summary(self) -> None:
        n_nc   = len(self._nc_cycle_vms)
        n_fill = len(self._session.fill_queries)
        self._card.set_summary(
            f"{n_nc} NC cycle{'s' if n_nc != 1 else ''}  ·  "
            f"{n_fill} filled quer{'ies' if n_fill != 1 else 'y'}"
        )
