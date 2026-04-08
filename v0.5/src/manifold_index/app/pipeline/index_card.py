"""app/pipeline/index_card.py — Card ②: Refined Index (Query / Grid / Cache).

BLUEPRINT §10.3.

Three modes: Query (one (m,e)), Grid (batch), From Cache.
Optional refinement section (hidden when num_hard=0).
SeriesTable shows accumulated results.
WeylWorker run after Grid to extract A/B vectors.
"""

from __future__ import annotations

import logging
from fractions import Fraction
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox,
    QFrame, QGroupBox, QHBoxLayout, QLabel, QPushButton,
    QRadioButton, QSizePolicy, QSpinBox, QVBoxLayout, QWidget,
)

from manifold_index.services.session import IndexQuery, PipelineStage, Session
from manifold_index.viewmodels.advisory import Advisory, AdvisoryLevel, CardStatus
from manifold_index.viewmodels.index_vm import (
    build_index_query_vm, build_index_vm, build_weyl_vm,
)
from manifold_index.formatters.index_fmt import format_series_latex
from manifold_index.app.widgets.collapsible_card import CollapsibleCard
from manifold_index.app.widgets.series_table import SeriesTable
from manifold_index.app.widgets.math_view import MathView
from manifold_index.app.workers.index_worker import IndexWorker
from manifold_index.app.workers.weyl_worker import WeylWorker


class IndexCard(QWidget):
    """Card ②: compute or load I^ref(m,e).

    Signals
    -------
    session_updated(Session)
    """

    session_updated = Signal(object)

    def __init__(self, session: Session, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._session   = session
        self._workers: list[IndexWorker] = []
        self._weyl_worker: WeylWorker | None = None
        self._session_gen: int = 0   # incremented on unlock(); guards stale signals

        self._card = CollapsibleCard(2, "Refined Index", parent=self)
        self._card.set_status(CardStatus.LOCKED)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._card)

        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(10)

        # ── Mode selector ─────────────────────────────────────────────
        mode_box = QGroupBox("Mode")
        mode_layout = QHBoxLayout(mode_box)
        self._mode_query  = QRadioButton("Query")
        self._mode_grid   = QRadioButton("Grid")
        self._mode_cache  = QRadioButton("From Cache")
        self._mode_query.setChecked(True)
        self._mode_group  = QButtonGroup(self)
        for rb in (self._mode_query, self._mode_grid, self._mode_cache):
            self._mode_group.addButton(rb)
            mode_layout.addWidget(rb)
        mode_layout.addStretch(1)
        bl.addWidget(mode_box)

        # ── Refinement section (hidden when num_hard=0) ───────────────
        self._refinement_box = QGroupBox("Refinement")
        ref_layout = QVBoxLayout(self._refinement_box)
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self._preset_combo = QComboBox()
        self._preset_combo.addItems(["Full Refined", "Unrefined", "Custom"])
        self._preset_combo.currentTextChanged.connect(self._on_preset_changed)
        preset_row.addWidget(self._preset_combo)
        preset_row.addStretch(1)
        ref_layout.addLayout(preset_row)

        self._edge_checkboxes: list[QCheckBox] = []
        self._edge_check_container = QWidget()
        self._edge_check_layout = QHBoxLayout(self._edge_check_container)
        self._edge_check_layout.setContentsMargins(0, 0, 0, 0)
        ref_layout.addWidget(self._edge_check_container)

        self._refinement_box.setVisible(False)
        bl.addWidget(self._refinement_box)

        # ── Charge input (Query mode) ─────────────────────────────────
        self._charge_box = QGroupBox("Charges")
        charge_layout = QHBoxLayout(self._charge_box)
        charge_layout.addWidget(QLabel("m:"))
        self._m_spin = QSpinBox()
        self._m_spin.setRange(-99, 99)
        self._m_spin.setValue(0)
        self._m_spin.setFixedWidth(60)
        charge_layout.addWidget(self._m_spin)
        charge_layout.addWidget(QLabel("  e:"))
        self._e_spin = QDoubleSpinBox()
        self._e_spin.setRange(-49.5, 49.5)
        self._e_spin.setSingleStep(0.5)
        self._e_spin.setDecimals(1)
        self._e_spin.setValue(0.0)
        self._e_spin.setFixedWidth(70)
        charge_layout.addWidget(self._e_spin)
        charge_layout.addStretch(1)
        bl.addWidget(self._charge_box)

        # ── Compute button row ────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._compute_btn = QPushButton("Compute")
        self._compute_btn.setProperty("class", "primary")
        self._compute_btn.clicked.connect(self._on_compute_clicked)
        btn_row.addWidget(self._compute_btn)

        self._weyl_btn = QPushButton("Run Weyl Check")
        self._weyl_btn.setProperty("class", "secondary")
        self._weyl_btn.setVisible(False)
        self._weyl_btn.clicked.connect(self._on_weyl_clicked)
        btn_row.addWidget(self._weyl_btn)
        btn_row.addStretch(1)
        bl.addLayout(btn_row)

        # Status label
        self._status_label = QLabel()
        self._status_label.setProperty("class", "muted")
        self._status_label.setVisible(False)
        bl.addWidget(self._status_label)

        # ── Results table ─────────────────────────────────────────────
        self._results_table = SeriesTable()
        self._results_table.setMinimumHeight(100)
        self._results_table.copy_latex_requested.connect(self._on_copy_latex)
        self._results_table.row_removed.connect(self._on_row_removed)
        bl.addWidget(self._results_table)

        # ── Weyl status view ──────────────────────────────────────────
        self._weyl_view = MathView(min_h=80)
        self._weyl_view.setVisible(False)
        bl.addWidget(self._weyl_view)

        self._card.set_body(body)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def unlock(self, session: Session) -> None:
        """Called by PipelineView when Card ① succeeds."""
        self._session = session
        self._session_gen += 1
        self._card.set_status(CardStatus.READY)
        self._card.expand()
        self._rebuild_edge_toggles()
        self._rebuild_mode_availability()

    def lock(self) -> None:
        """Called by PipelineView when session is invalidated."""
        # Disconnect signals from all in-flight workers so stale results
        # cannot corrupt the next session's state.
        for w in self._workers:
            try:
                w.finished.disconnect()
                w.error.disconnect()
            except RuntimeError:
                pass
        self._workers.clear()
        if self._weyl_worker is not None:
            try:
                self._weyl_worker.finished.disconnect()
                self._weyl_worker.error.disconnect()
            except RuntimeError:
                pass
            self._weyl_worker = None
        self._card.set_status(CardStatus.LOCKED)
        self._card.collapse()
        self._results_table.clear_rows()
        self._weyl_view.setVisible(False)
        self._weyl_btn.setVisible(False)

    def refresh(self, session: Session) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _rebuild_edge_toggles(self) -> None:
        """Recreate per-edge checkboxes based on session.num_hard()."""
        # Clear existing
        for cb in self._edge_checkboxes:
            cb.setParent(None)
        self._edge_checkboxes.clear()

        n = self._session.num_hard()
        self._refinement_box.setVisible(n > 0)
        if n == 0:
            return

        for j in range(n):
            cb = QCheckBox(f"η_{j}")
            cb.setChecked(True)
            self._edge_check_layout.addWidget(cb)
            self._edge_checkboxes.append(cb)

        self._preset_combo.setCurrentText("Full Refined")

    def _rebuild_mode_availability(self) -> None:
        iref_avail = self._session.cache_status.get("iref", {}).get("available", False)
        self._mode_cache.setEnabled(iref_avail)
        if not iref_avail and self._mode_cache.isChecked():
            self._mode_query.setChecked(True)

    def _on_preset_changed(self, text: str) -> None:
        custom = (text == "Custom")
        for cb in self._edge_checkboxes:
            cb.setEnabled(custom)
            if text == "Full Refined":
                cb.setChecked(True)
            elif text == "Unrefined":
                cb.setChecked(False)

    def _active_edges(self) -> list[bool]:
        if not self._edge_checkboxes:
            return []
        return [cb.isChecked() for cb in self._edge_checkboxes]

    def _on_compute_clicked(self) -> None:
        s = self._session
        if s.stage < PipelineStage.LOADED or s.nz_data is None:
            return

        r = int(s.nz_data.r)
        m_ext = [self._m_spin.value()] * r
        e_val = Fraction(self._e_spin.value()).limit_denominator(2)
        e_ext = [e_val] * r
        use_cache = self._mode_cache.isChecked()

        self._status_label.setText("Computing…")
        self._status_label.setVisible(True)
        self._card.set_status(CardStatus.RUNNING)
        self._compute_btn.setEnabled(False)

        row = self._results_table.add_row(
            m_ext[0] if r == 1 else str(m_ext),
            str(e_ext[0]) if r == 1 else str(e_ext),
            "",
            "—",
        )
        self._results_table.set_row_computing(row)

        worker = IndexWorker(
            nz_data       = s.nz_data,
            m_ext         = m_ext,
            e_ext         = e_ext,
            q_order_half  = s.q_order_half,
            manifold_name = s.manifold_name,
            use_cache     = use_cache,
            parent        = self,
        )
        gen = self._session_gen          # capture for stale-signal guard
        worker.finished.connect(lambda p, r=row, g=gen: self._on_index_finished(p, r, g))
        worker.error.connect(lambda e, r=row, g=gen: self._on_index_error(e, r, g))
        self._workers.append(worker)
        worker.start()

    def _on_index_finished(self, payload: dict, row: int, gen: int) -> None:
        sender = self.sender()
        if sender in self._workers:
            self._workers.remove(sender)

        if gen != self._session_gen:
            return   # stale: manifold was reloaded since this worker launched

        self._compute_btn.setEnabled(True)
        self._status_label.setVisible(False)

        m_ext   = payload["m_ext"]
        e_ext   = payload["e_ext"]
        result  = payload["result"]
        source  = "cache" if payload["from_cache"] else "computed"

        try:
            series_latex = format_series_latex(result, self._session.num_hard(), self._session.q_order_half)
        except Exception:
            series_latex = str(result) if result else "0"

        self._results_table.set_row_result(row, series_latex, source)

        # Persist to session
        s = self._session
        iq = IndexQuery(
            m_ext         = m_ext,
            e_ext         = e_ext,
            q_order_half  = s.q_order_half,
            result        = result,
            projected_result = None,
            active_edges  = self._active_edges(),
            source        = source,
        )
        s.index_queries.append(iq)
        if s.stage < PipelineStage.INDEXED:
            s.stage = PipelineStage.INDEXED

        self._card.set_status(CardStatus.DONE)
        self._weyl_btn.setVisible(True)
        self._update_summary()
        self.session_updated.emit(s)

    def _on_index_error(self, msg: str, row: int, gen: int) -> None:
        sender = self.sender()
        if sender in self._workers:
            self._workers.remove(sender)
        if gen != self._session_gen:
            return
        self._compute_btn.setEnabled(True)
        self._status_label.setVisible(False)
        self._results_table.set_row_result(row, f"Error: {msg}", "—")
        logging.warning("IndexWorker error: %s", msg)
        self._card.set_status(CardStatus.ERROR)

    def _on_weyl_clicked(self) -> None:
        s = self._session
        if not s.index_queries:
            return
        entries = [
            (q.m_ext, q.e_ext, q.result)
            for q in s.index_queries
            if q.result is not None
        ]
        self._weyl_btn.setEnabled(False)
        self._weyl_view.set_loading(True)
        self._weyl_view.setVisible(True)

        self._weyl_worker = WeylWorker(
            entries      = entries,
            num_hard     = s.num_hard(),
            q_order_half = s.q_order_half,
            parent       = self,
        )
        self._weyl_worker.finished.connect(self._on_weyl_finished)
        self._weyl_worker.error.connect(self._on_weyl_error)
        self._weyl_worker.start()

    def _on_weyl_finished(self, payload: dict) -> None:
        self._weyl_btn.setEnabled(True)
        ab = payload["ab_vectors"]
        self._session.weyl_result  = ab
        self._session.weyl_checked = True

        vm = build_weyl_vm(ab, self._session.num_hard())
        self._card.set_advisories(vm.advisories)

        if ab is not None:
            html = (
                f"<h3>Weyl Check</h3>"
                f"<p class='success'>✓ Compatible</p>"
                f"<p><b>a:</b> {ab.a}</p>"
                f"<p><b>b:</b> {ab.b}</p>"
            )
        else:
            html = "<h3>Weyl Check</h3><p class='warn'>⚠ Could not extract Weyl vectors</p>"
        self._weyl_view.set_loading(False)
        self._weyl_view.set_html(html)
        self.session_updated.emit(self._session)

    def _on_weyl_error(self, msg: str) -> None:
        self._weyl_btn.setEnabled(True)
        self._weyl_view.set_loading(False)
        self._weyl_view.set_html(f"<p class='warn'>Weyl error: {msg}</p>")

    def _on_copy_latex(self, row: int) -> None:
        from PySide6.QtWidgets import QApplication
        queries = [q for q in self._session.index_queries if q.result is not None]
        if row < len(queries):
            try:
                latex = format_series_latex(
                    queries[row].result,
                    self._session.num_hard(),
                    queries[row].q_order_half,
                )
                QApplication.clipboard().setText(latex)
            except Exception:
                pass

    def _on_row_removed(self, row: int) -> None:
        queries = [q for q in self._session.index_queries if q.result is not None]
        if row < len(queries):
            self._session.index_queries.remove(queries[row])
        self.session_updated.emit(self._session)

    def _update_summary(self) -> None:
        n = self._session.index_query_count()
        n_hard = self._session.num_hard()
        mode = "full refined" if all(self._active_edges()) else "custom η" if n_hard else "3D index"
        weyl = "✓" if self._session.weyl_checked else "not run"
        self._card.set_summary(f"{n} quer{'y' if n == 1 else 'ies'}  ·  {mode}  ·  Weyl: {weyl}")

