"""app/pipeline/index_card.py — Card ②: Refined Index (Query / Grid / Cache).

BLUEPRINT §10.3.

Three modes: Query (one (m,e)), Grid (batch), From Cache.
Optional refinement section (hidden when num_hard=0).
SeriesTable shows accumulated results.
Weyl check has moved to FillingCard (Card ③).
"""

from __future__ import annotations

import itertools
import logging
from fractions import Fraction

from PySide6.QtCore import Qt, QCoreApplication, QTimer, Signal
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox,
    QGroupBox, QHBoxLayout, QLabel, QProgressBar, QPushButton,
    QRadioButton, QSpinBox, QVBoxLayout, QWidget,
)

from manifold_index.services.session import IndexQuery, PipelineStage, Session
from manifold_index.services.compute_service import ComputeService
from manifold_index.viewmodels.advisory import CardStatus
from manifold_index.viewmodels.index_vm import build_index_query_vm, build_index_vm
from manifold_index.formatters.index_fmt import format_series_latex
from manifold_index.app.widgets.collapsible_card import CollapsibleCard
from manifold_index.app.widgets.series_table import SeriesTable
from manifold_index.app.workers.index_worker import IndexWorker


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _fmt_charges(vals: list) -> str:
    """Format a list of m/e charge values as a parenthesized tuple string.

    Converts ``[0, Fraction(1, 2)]`` → ``"(0, 1/2)"`` for display.
    """
    parts = []
    for v in vals:
        f = Fraction(v).limit_denominator(1000)
        parts.append(str(int(f)) if f.denominator == 1 else str(f))
    return "(" + ", ".join(parts) + ")"


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
        self._pending_grid: list[tuple[list, list, int]] = []  # (m_ext, e_ext, row) queue for serialised grid
        self._session_gen: int = 0
        self._grid_total: int = 0
        self._grid_done: int  = 0

        # Debounce timer for edge-toggle projection refresh.
        # Fires 300 ms after the last checkbox state change.
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(300)
        self._refresh_timer.timeout.connect(self._do_refresh_series_display)

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
        self._mode_grid.setChecked(True)
        self._mode_group  = QButtonGroup(self)
        for rb in (self._mode_query, self._mode_grid, self._mode_cache):
            self._mode_group.addButton(rb)
            mode_layout.addWidget(rb)
        mode_layout.addStretch(1)
        self._mode_group.buttonToggled.connect(self._on_mode_changed)
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
        lbl_m = QLabel("<i>m</i>:")
        lbl_m.setTextFormat(Qt.TextFormat.RichText)
        charge_layout.addWidget(lbl_m)
        self._m_spin = QSpinBox()
        self._m_spin.setRange(-99, 99)
        self._m_spin.setValue(0)
        self._m_spin.setFixedWidth(60)
        charge_layout.addWidget(self._m_spin)
        lbl_e = QLabel("  <i>e</i>:")
        lbl_e.setTextFormat(Qt.TextFormat.RichText)
        charge_layout.addWidget(lbl_e)
        self._e_spin = QDoubleSpinBox()
        self._e_spin.setRange(-49.5, 49.5)
        self._e_spin.setSingleStep(0.5)
        self._e_spin.setDecimals(1)
        self._e_spin.setValue(0.0)
        self._e_spin.setFixedWidth(70)
        charge_layout.addWidget(self._e_spin)
        charge_layout.addStretch(1)
        bl.addWidget(self._charge_box)

        # ── Grid range input (Grid mode) ──────────────────────────────
        self._grid_box = QGroupBox("Grid Range")
        grid_layout = QHBoxLayout(self._grid_box)
        lbl_gm = QLabel("<i>m</i>:")
        lbl_gm.setTextFormat(Qt.TextFormat.RichText)
        grid_layout.addWidget(lbl_gm)
        self._m_min_spin = QSpinBox()
        self._m_min_spin.setRange(-99, 99)
        self._m_min_spin.setValue(-2)
        self._m_min_spin.setFixedWidth(60)
        grid_layout.addWidget(self._m_min_spin)
        grid_layout.addWidget(QLabel("to"))
        self._m_max_spin = QSpinBox()
        self._m_max_spin.setRange(-99, 99)
        self._m_max_spin.setValue(2)
        self._m_max_spin.setFixedWidth(60)
        grid_layout.addWidget(self._m_max_spin)
        lbl_ge = QLabel("   <i>e</i>:")
        lbl_ge.setTextFormat(Qt.TextFormat.RichText)
        grid_layout.addWidget(lbl_ge)
        self._e_min_spin = QDoubleSpinBox()
        self._e_min_spin.setRange(-49.5, 49.5)
        self._e_min_spin.setSingleStep(0.5)
        self._e_min_spin.setDecimals(1)
        self._e_min_spin.setValue(-2.0)
        self._e_min_spin.setFixedWidth(70)
        grid_layout.addWidget(self._e_min_spin)
        grid_layout.addWidget(QLabel("to"))
        self._e_max_spin = QDoubleSpinBox()
        self._e_max_spin.setRange(-49.5, 49.5)
        self._e_max_spin.setSingleStep(0.5)
        self._e_max_spin.setDecimals(1)
        self._e_max_spin.setValue(2.0)
        self._e_max_spin.setFixedWidth(70)
        grid_layout.addWidget(self._e_max_spin)
        grid_layout.addStretch(1)
        self._grid_box.setVisible(False)
        bl.addWidget(self._grid_box)

        # ── Compute / Stop button row ─────────────────────────────────
        btn_row = QHBoxLayout()
        self._compute_btn = QPushButton("Compute")
        self._compute_btn.setProperty("class", "primary")
        self._compute_btn.clicked.connect(self._on_compute_clicked)
        btn_row.addWidget(self._compute_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setProperty("class", "secondary")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        btn_row.addWidget(self._stop_btn)

        btn_row.addStretch(1)
        bl.addLayout(btn_row)

        # Status label
        self._status_label = QLabel()
        self._status_label.setProperty("class", "muted")
        self._status_label.setVisible(False)
        bl.addWidget(self._status_label)

        # Progress bar (grid mode only)
        self._progress_bar = QProgressBar()
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setFixedHeight(8)
        self._progress_bar.setVisible(False)
        bl.addWidget(self._progress_bar)

        # ── Results table ─────────────────────────────────────────────
        self._results_table = SeriesTable()
        self._results_table.setMinimumHeight(100)
        self._results_table.copy_latex_requested.connect(self._on_copy_latex)
        self._results_table.row_removed.connect(self._on_row_removed)
        bl.addWidget(self._results_table)

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
        for w in self._workers:
            try:
                w.finished.disconnect()
                w.error.disconnect()
            except RuntimeError:
                pass
        self._workers.clear()
        self._pending_grid.clear()   # discard any unstarted grid items
        self._card.set_status(CardStatus.LOCKED)
        self._card.collapse()
        self._results_table.clear_rows()
        self._progress_bar.setVisible(False)
        self._refresh_timer.stop()
        self._status_label.setVisible(False)
        self._stop_btn.setEnabled(False)
        self._grid_total = 0
        self._grid_done  = 0

    def refresh(self, session: Session) -> None:
        self._session = session

    def trigger_compute(self) -> None:
        """Public: programmatically trigger a Compute (used by Run All)."""
        self._on_compute_clicked()

    def trigger_stop(self) -> None:
        """Public: programmatically trigger the Stop button (used by Run All)."""
        self._on_stop_clicked()

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
            # Use Unicode subscript digits for a LaTeX-style look: W₀, W₁, …
            sub = chr(0x2080 + j) if j <= 9 else f"_{j}"
            cb = QCheckBox(f"W{sub}")
            cb.setChecked(True)
            cb.stateChanged.connect(self._on_edge_toggle)
            self._edge_check_layout.addWidget(cb)
            self._edge_checkboxes.append(cb)

        self._preset_combo.setCurrentText("Full Refined")

    def _rebuild_mode_availability(self) -> None:
        iref_avail = self._session.cache_status.get("iref", {}).get("available", False)
        self._mode_cache.setEnabled(iref_avail)
        if not iref_avail and self._mode_cache.isChecked():
            self._mode_query.setChecked(True)
        self._on_mode_changed()

    def _on_mode_changed(self, *_: object) -> None:
        """Show/hide input boxes and update button label based on mode."""
        is_query = self._mode_query.isChecked()
        is_grid  = self._mode_grid.isChecked()
        is_cache = self._mode_cache.isChecked()
        self._charge_box.setVisible(is_query)
        self._grid_box.setVisible(is_grid)
        if is_cache:
            self._compute_btn.setText("Load from Cache")
        elif is_grid:
            self._compute_btn.setText("Compute Grid")
        else:
            self._compute_btn.setText("Compute")

        if hasattr(self, "_session") and self._session is not None:
            self._results_table.clear_rows()
            self._session.index_queries.clear()
            self._status_label.setVisible(False)
            self._progress_bar.setVisible(False)
            self._stop_btn.setEnabled(False)
            self._compute_btn.setEnabled(True)
            self._card.set_status(CardStatus.READY)

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

        # Reset grid counters for this computation run
        self._grid_total = 0
        self._grid_done  = 0
        self._progress_bar.setVisible(False)

        # ── From Cache: bulk-load all cached (m,e) entries ───────────
        if self._mode_cache.isChecked():
            self._compute_btn.setEnabled(False)
            self._status_label.setText("Loading from cache…")
            self._status_label.setVisible(True)
            self._card.set_status(CardStatus.RUNNING)

            entries = ComputeService.enumerate_iref_cache(
                s.manifold_name, s.nz_data, s.q_order_half
            )
            if not entries:
                self._status_label.setText("No cache entries found for this manifold.")
                self._compute_btn.setEnabled(True)
                self._card.set_status(CardStatus.ERROR)
                return

            for m_ext, e_ext, result in entries:
                r_int = int(s.nz_data.r)
                m_disp = m_ext[0] if r_int == 1 else _fmt_charges(m_ext)
                e_disp = str(e_ext[0]) if r_int == 1 else _fmt_charges(e_ext)
                try:
                    series_latex = format_series_latex(result, s.num_hard(), s.q_order_half)
                except Exception:
                    series_latex = str(result) if result else "0"
                row = self._results_table.add_row(m_disp, e_disp, series_latex, "cache")
                iq = IndexQuery(
                    m_ext            = m_ext,
                    e_ext            = e_ext,
                    q_order_half     = s.q_order_half,
                    result           = result,
                    projected_result = None,
                    active_edges     = self._active_edges(),
                    source           = "cache",
                )
                s.index_queries.append(iq)

            if s.stage < PipelineStage.INDEXED:
                s.stage = PipelineStage.INDEXED
            self._card.set_status(CardStatus.DONE)
            self._status_label.setVisible(False)
            self._compute_btn.setEnabled(True)
            self._update_summary()
            self.session_updated.emit(s)
            return

        # ── Grid: spawn one worker per (m,e) point ────────────────────
        if self._mode_grid.isChecked():
            m_min = self._m_min_spin.value()
            m_max = self._m_max_spin.value()
            if m_min > m_max:
                m_min, m_max = m_max, m_min
            e_min_half = round(self._e_min_spin.value() * 2)
            e_max_half = round(self._e_max_spin.value() * 2)
            if e_min_half > e_max_half:
                e_min_half, e_max_half = e_max_half, e_min_half

            r_int = int(s.nz_data.r)

            # Build the grid as the Cartesian product of per-cusp (m, e) pairs.
            # For single-cusp this is just the flat grid; for multi-cusp every
            # cusp independently ranges over [m_min..m_max] × [e_min..e_max],
            # yielding (m_count × e_count)^r_int total evaluation points.
            per_cusp_pairs = [
                (m_val, Fraction(e_half, 2))
                for m_val in range(m_min, m_max + 1)
                for e_half in range(e_min_half, e_max_half + 1)
            ]
            if r_int == 1:
                grid_points: list[tuple[list, list]] = [
                    ([m], [e]) for m, e in per_cusp_pairs
                ]
            else:
                grid_points = [
                    ([c[0] for c in combo], [c[1] for c in combo])
                    for combo in itertools.product(per_cusp_pairs, repeat=r_int)
                ]
            n_points = len(grid_points)
            if n_points == 0:
                return

            self._grid_total = n_points
            self._grid_done  = 0
            self._compute_btn.setEnabled(False)
            self._stop_btn.setEnabled(True)
            self._status_label.setText(f"Computing grid: 0 / {n_points}")
            self._status_label.setVisible(True)
            self._progress_bar.setMaximum(n_points)
            self._progress_bar.setValue(0)
            self._progress_bar.setVisible(True)
            self._card.set_status(CardStatus.RUNNING)

            gen = self._session_gen
            # Pre-create all table rows so the user sees placeholders immediately,
            # then queue (m_ext, e_ext, row) for serialised execution — only one
            # IndexWorker runs at a time to avoid GIL contention and shared-state
            # fights that make the computation slower and the UI freeze.
            self._pending_grid.clear()
            first_item: tuple | None = None
            for idx, (m_ext, e_ext) in enumerate(grid_points):
                m_disp = m_ext[0] if r_int == 1 else _fmt_charges(m_ext)
                e_disp = str(e_ext[0]) if r_int == 1 else _fmt_charges(e_ext)
                row = self._results_table.add_row(m_disp, e_disp, "", "—")
                self._results_table.set_row_computing(row)
                if first_item is None:
                    first_item = (m_ext, e_ext, row, gen)
                else:
                    self._pending_grid.append((m_ext, e_ext, row, gen))
                # Yield to event loop every 100 rows to keep UI responsive during
                # grid setup for heavy manifolds with many cusps/ranges.
                if (idx + 1) % 100 == 0:
                    QCoreApplication.processEvents()

            # Kick off only the first worker; _on_index_finished/_on_index_error
            # will drain _pending_grid one entry at a time.
            if first_item is not None:
                m0, e0, row0, g0 = first_item
                self._start_grid_worker(s, m0, e0, row0, g0)
            return

        # ── Query (single point) ──────────────────────────────────────
        r_int = int(s.nz_data.r)
        m_ext = [self._m_spin.value()] * r_int
        e_val = Fraction(self._e_spin.value()).limit_denominator(2)
        e_ext = [e_val] * r_int

        self._status_label.setText("Computing…")
        self._status_label.setVisible(True)
        self._card.set_status(CardStatus.RUNNING)
        self._compute_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)

        # Reuse existing row if (m, e) was already queried; avoids duplicate rows.
        existing_row: int | None = None
        for i, iq_ex in enumerate(s.index_queries):
            if iq_ex.m_ext == m_ext and iq_ex.e_ext == e_ext:
                existing_row = i
                break

        if existing_row is not None:
            row = existing_row
            self._results_table.set_row_computing(row)
        else:
            row = self._results_table.add_row(
                m_ext[0] if r_int == 1 else _fmt_charges(m_ext),
                str(e_ext[0]) if r_int == 1 else _fmt_charges(e_ext),
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
            use_cache     = False,
            parent        = self,
        )
        gen = self._session_gen
        worker.finished.connect(lambda p, r=row, g=gen, w=worker: self._on_index_finished(p, r, g, w))
        worker.error.connect(lambda e, r=row, g=gen, w=worker: self._on_index_error(e, r, g, w))
        self._workers.append(worker)
        worker.start()

    def _start_grid_worker(self, s: object, m_ext: list, e_ext: list, row: int, gen: int) -> None:
        """Spawn exactly one IndexWorker for a grid point and track it."""
        worker = IndexWorker(
            nz_data       = s.nz_data,
            m_ext         = m_ext,
            e_ext         = e_ext,
            q_order_half  = s.q_order_half,
            manifold_name = s.manifold_name,
            use_cache     = False,
            parent        = self,
        )
        worker.finished.connect(
            lambda p, r=row, g=gen, w=worker: self._on_index_finished(p, r, g, w)
        )
        worker.error.connect(
            lambda e, r=row, g=gen, w=worker: self._on_index_error(e, r, g, w)
        )
        self._workers.append(worker)
        worker.start()

    def _drain_pending_grid(self) -> None:
        """Start the next queued grid worker, if any remain."""
        if self._pending_grid and self._session_gen == self._session_gen:
            s = self._session
            m_ext, e_ext, row, gen = self._pending_grid.pop(0)
            if gen == self._session_gen:
                self._start_grid_worker(s, m_ext, e_ext, row, gen)

    def _on_index_finished(self, payload: dict, row: int, gen: int, worker: object) -> None:
        if worker in self._workers:
            self._workers.remove(worker)

        if gen != self._session_gen:
            # Stale: manifold was reloaded; clear any pending queue too
            self._pending_grid.clear()
            return

        # Update progress bar for grid runs
        if self._grid_total > 0:
            self._grid_done += 1
            self._progress_bar.setValue(self._grid_done)
            self._status_label.setText(
                f"Computing grid: {self._grid_done} / {self._grid_total}"
            )

        # Start the next queued grid worker (serialised execution).
        # Do this before the "all workers finished?" check so the new worker
        # is in self._workers before we test for emptiness.
        if self._pending_grid:
            self._drain_pending_grid()

        # Only re-enable controls when all workers have finished and
        # the pending queue is also empty.
        all_done = not self._workers and not self._pending_grid
        if all_done:
            self._compute_btn.setEnabled(True)
            self._status_label.setVisible(False)
            self._progress_bar.setVisible(False)

        m_ext   = payload["m_ext"]
        e_ext   = payload["e_ext"]
        result  = payload["result"]
        source  = "cache" if payload["from_cache"] else "computed"

        # Apply η projection according to current checkbox state
        active = self._active_edges()
        try:
            projected = ComputeService.project_refined_index(result, active) if active else result
            series_latex = format_series_latex(projected, self._session.num_hard(), self._session.q_order_half)
        except Exception:
            series_latex = str(result) if result else "0"

        self._results_table.set_row_result(row, series_latex, source)

        # Persist to session — replace any existing entry for this (m, e)
        s = self._session
        s.index_queries = [q for q in s.index_queries
                           if not (q.m_ext == m_ext and q.e_ext == e_ext)]
        iq = IndexQuery(
            m_ext         = m_ext,
            e_ext         = e_ext,
            q_order_half  = s.q_order_half,
            result        = result,
            projected_result = None,
            active_edges  = active,
            source        = source,
        )
        s.index_queries.append(iq)
        if s.stage < PipelineStage.INDEXED:
            s.stage = PipelineStage.INDEXED

        # In query mode, sort the accumulated results by (m, e) and rebuild
        # the table so rows stay in a consistent order.
        if self._grid_total == 0:
            self._rebuild_sorted_table()

        all_done = not self._workers and not self._pending_grid
        if all_done:
            self._card.set_status(CardStatus.DONE)
            self._stop_btn.setEnabled(False)
            self._update_summary()
        self.session_updated.emit(s)

    def _on_index_error(self, msg: str, row: int, gen: int, worker: object) -> None:
        if worker in self._workers:
            self._workers.remove(worker)
        if gen != self._session_gen:
            self._pending_grid.clear()
            return
        if self._grid_total > 0:
            self._grid_done += 1
            self._progress_bar.setValue(self._grid_done)
            self._status_label.setText(
                f"Computing grid: {self._grid_done} / {self._grid_total}"
            )
        # Even on error, drain to the next pending grid item so the rest complete.
        if self._pending_grid:
            self._drain_pending_grid()
        all_done = not self._workers and not self._pending_grid
        if all_done:
            self._compute_btn.setEnabled(True)
            self._status_label.setVisible(False)
            self._progress_bar.setVisible(False)
        self._results_table.set_row_result(row, f"Error: {msg}", "—")
        logging.warning("IndexWorker error: %s", msg)
        if all_done:
            self._card.set_status(CardStatus.ERROR)
            self._update_summary()

    # ------------------------------------------------------------------
    # Display helpers: projection refresh and sorted-table rebuild
    # ------------------------------------------------------------------

    def _project_latex(self, result: object) -> str:
        """Format *result* with current active-edge projection applied."""
        active = self._active_edges()
        try:
            projected = ComputeService.project_refined_index(result, active) if active else result
            return format_series_latex(projected, self._session.num_hard(), self._session.q_order_half)
        except Exception:
            return str(result) if result else "0"

    def _rebuild_sorted_table(self) -> None:
        """Clear the table and re-populate from session.index_queries sorted by (m, e)."""
        s = self._session
        s.index_queries.sort(
            key=lambda q: (q.m_ext[0] if q.m_ext else 0, float(q.e_ext[0]) if q.e_ext else 0.0)
        )
        r_int = int(s.nz_data.r) if s.nz_data else 1
        self._results_table.clear_rows()
        for iq in s.index_queries:
            if iq.result is None:
                continue
            m_disp = iq.m_ext[0] if r_int == 1 else _fmt_charges(iq.m_ext)
            e_disp = str(iq.e_ext[0]) if r_int == 1 else _fmt_charges(iq.e_ext)
            latex  = self._project_latex(iq.result)
            self._results_table.add_row(m_disp, e_disp, latex, iq.source)

    def _refresh_series_display(self) -> None:
        """Re-project and re-render every table row after an active-edge toggle."""
        if not hasattr(self, "_session") or self._session is None:
            return
        active = self._active_edges()
        for i, iq in enumerate(self._session.index_queries):
            if iq is None or iq.result is None:
                continue
            try:
                projected = ComputeService.project_refined_index(iq.result, active)
                latex = format_series_latex(projected, self._session.num_hard(),
                                            self._session.q_order_half)
            except Exception:
                latex = str(iq.result) if iq.result else "0"
            self._results_table.set_row_result(i, latex, iq.source)

    def _on_edge_toggle(self) -> None:
        """Called when any W_j checkbox is toggled.

        Shows a brief "Updating…" indicator and defers the actual re-projection
        via a 300 ms single-shot timer so that rapid multi-checkbox changes (or
        preset switches on large manifolds) only trigger one repaint.
        """
        if not hasattr(self, "_session") or self._session is None:
            return
        if not self._session.index_queries:
            return
        # Show a transient status message while the timer is pending
        self._status_label.setText("Updating…")
        self._status_label.setVisible(True)
        # Restart the debounce timer (kills any pending fire)
        self._refresh_timer.start()

    def _do_refresh_series_display(self) -> None:
        """Actual projection refresh, invoked by the debounce timer."""
        self._refresh_series_display()
        # Hide the status label only if we're not in the middle of a grid run
        if self._grid_total == 0 or self._grid_done >= self._grid_total:
            self._status_label.setVisible(False)

    def _on_stop_clicked(self) -> None:
        """Abandon all running index/grid workers and clear the pending queue."""
        for w in self._workers:
            try:
                w.finished.disconnect()
                w.error.disconnect()
            except RuntimeError:
                pass
        self._workers.clear()
        self._pending_grid.clear()   # discard any unstarted grid items
        self._grid_total = 0
        self._grid_done  = 0
        self._compute_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._progress_bar.setVisible(False)
        self._status_label.setText("Stopped.")
        self._status_label.setVisible(True)
        self._card.set_status(CardStatus.READY)

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
        if n_hard > 0 and all(self._active_edges()):
            mode = "full refined"
        elif n_hard > 0:
            mode = "custom η"
        else:
            mode = "3D index"
        weyl = "✓" if self._session.weyl_checked else "not run"
        self._card.set_summary(f"{n} quer{'y' if n == 1 else 'ies'}  ·  {mode}")

