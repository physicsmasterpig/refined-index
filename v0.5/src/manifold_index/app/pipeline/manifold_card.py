"""app/pipeline/manifold_card.py — Card ①: Load manifold + NZ + cache probe.

BLUEPRINT §10.2.

Input:  manifold name QLineEdit + Nmax QSpinBox + "Load" QPushButton.
Worker: LoadWorker → ComputeService.load_manifold + probe_cache.

On success: populates session, builds ManifoldViewModel, sets summary,
collapses card, emits session_updated.
On failure: ERROR advisory shown, card stays expanded.
"""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QSizePolicy, QSpinBox, QVBoxLayout, QWidget,
)

from manifold_index.services.session import PipelineStage, Session
from manifold_index.viewmodels.advisory import Advisory, AdvisoryLevel, CardStatus
from manifold_index.viewmodels.manifold_vm import build_manifold_vm, build_manifold_vm_error
from manifold_index.formatters.manifold_fmt import (
    format_nz_latex, format_gluing_table_html,
    format_easy_edges_html, format_hard_edges_html,
)
from manifold_index.app.widgets.collapsible_card import CollapsibleCard
from manifold_index.app.widgets.math_view import MathView, build_katex_html, sys_colors
from manifold_index.app.workers.load_worker import LoadWorker

if TYPE_CHECKING:
    pass

# ── Common advisory helpers ───────────────────────────────────────────────────

def _cache_advisory(cache_info: dict) -> Advisory | None:
    """Return an INFO advisory summarising cache hits, or None."""
    parts = []
    if cache_info.get("iref", {}).get("available"):
        qq = cache_info["iref"].get("qq_order")
        parts.append(f"I^ref cached (qq={qq})")
    if cache_info.get("nc", {}).get("available"):
        parts.append("NC cached")
    kcount = cache_info.get("kernels", {}).get("count", 0)
    if kcount:
        parts.append(f"{kcount} kernels cached")
    if not parts:
        return None
    return Advisory(
        advisory_id="A0-cache",
        level=AdvisoryLevel.INFO,
        title="Local cache",
        body="  ·  ".join(parts),
    )


class ManifoldCard(QWidget):
    """Card ①: load a manifold by name.

    Signals
    -------
    session_updated(Session) — emitted after a successful load so
                               PipelineView can refresh downstream cards.
    load_failed()            — emitted on worker error (downstream stays locked).
    """

    session_updated = Signal(object)   # Session
    load_failed     = Signal()

    def __init__(self, session: Session, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._session = session
        self._worker: LoadWorker | None = None

        # ── CollapsibleCard shell ────────────────────────────────────
        self._card = CollapsibleCard(1, "Manifold", parent=self)
        self._card.set_status(CardStatus.READY)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._card)

        # ── Body widget ──────────────────────────────────────────────
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(8)

        # Input row
        input_row = QHBoxLayout()
        input_row.setSpacing(8)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Manifold name (e.g. m004, 4_1)")
        self._name_edit.returnPressed.connect(self._on_load_clicked)
        input_row.addWidget(self._name_edit, 3)
        self._attach_autocomplete()

        nmax_label = QLabel("Nmax:")
        nmax_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        input_row.addWidget(nmax_label)

        self._nmax_spin = QSpinBox()
        self._nmax_spin.setRange(4, 100)
        self._nmax_spin.setValue(10)
        self._nmax_spin.setFixedWidth(60)
        self._nmax_spin.setToolTip(
            "Series truncation order Nmax (q_order_half = 2×Nmax)"
        )
        input_row.addWidget(self._nmax_spin)

        self._load_btn = QPushButton("Load")
        self._load_btn.setProperty("class", "primary")
        self._load_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._load_btn.clicked.connect(self._on_load_clicked)
        input_row.addWidget(self._load_btn)

        body_layout.addLayout(input_row)

        # Status label (shown during loading)
        self._status_label = QLabel()
        self._status_label.setProperty("class", "muted")
        self._status_label.setVisible(False)
        body_layout.addWidget(self._status_label)

        # MathView for NZ matrix + gluing table
        self._math_view = MathView(min_h=120)
        body_layout.addWidget(self._math_view)

        self._card.set_body(body)
        self._card.expand()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def refresh(self, session: Session) -> None:
        """Re-render from session (called by PipelineView on restore)."""
        self._session = session
        if session.stage >= PipelineStage.LOADED and session.nz_data is not None:
            self._name_edit.setText(session.manifold_name)
            self._nmax_spin.setValue(session.q_order_half // 2)
            self._render_loaded(session)

    def trigger_load(self) -> None:
        """Public: programmatically trigger a Load (used by Run All)."""
        self._on_load_clicked()

    # ------------------------------------------------------------------
    # Autocomplete
    # ------------------------------------------------------------------

    def _attach_autocomplete(self) -> None:
        """Attach SnaPy census names as inline autocomplete (best-effort).

        Reads manifold names directly from SnaPy's SQLite files using a
        short-lived connection that is opened and closed here, on the main
        thread, and never shared with worker threads.  This avoids the
        "SQLite object created in thread X used in thread Y" error that occurs
        when SnaPy's own ORM connection is opened on the main thread.
        """
        try:
            import snappy_manifolds.sqlite_files as sf  # type: ignore[import]
            import sqlite3, os

            db_path = os.path.join(sf.__path__[0], "manifolds.sqlite")
            if not os.path.isfile(db_path):
                return

            # Open a dedicated read-only connection — never reused by SnaPy.
            con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True,
                                  check_same_thread=False)
            names: list[str] = []
            try:
                # Cusped censuses have a plain 'name' column.
                for tbl in ("orientable_cusped_census",
                            "nonorientable_cusped_census",
                            "census_knots"):
                    try:
                        rows = con.execute(f"SELECT DISTINCT name FROM {tbl}").fetchall()
                        names.extend(r[0] for r in rows if r[0])
                    except sqlite3.OperationalError:
                        pass

                # Closed censuses store the Dehn-filled name as cusped(m,l).
                for tbl in ("orientable_closed_census",
                            "nonorientable_closed_census"):
                    try:
                        rows = con.execute(
                            f"SELECT DISTINCT cusped, m, l FROM {tbl}"
                        ).fetchall()
                        names.extend(
                            f"{r[0]}({r[1]},{r[2]})" for r in rows
                        )
                    except sqlite3.OperationalError:
                        pass
            finally:
                con.close()

            if not names:
                return

            from PySide6.QtWidgets import QCompleter
            from PySide6.QtCore import Qt
            completer = QCompleter(sorted(set(names)), self)
            completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
            completer.setFilterMode(Qt.MatchFlag.MatchContains)
            self._name_edit.setCompleter(completer)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_load_clicked(self) -> None:
        name = self._name_edit.text().strip()
        if not name:
            return
        if self._worker and self._worker.isRunning():
            return

        # Invalidate downstream if already loaded
        if self._session.stage >= PipelineStage.LOADED:
            self._session.invalidate_from(PipelineStage.LOADED)
            self.session_updated.emit(self._session)

        self._session.q_order_half = self._nmax_spin.value() * 2
        self._card.set_status(CardStatus.RUNNING)
        self._load_btn.setEnabled(False)
        self._status_label.setText("Loading…")
        self._status_label.setVisible(True)
        self._math_view.set_loading(True)

        # Load manifold in main thread (SnapPy SQLite is thread-local)
        try:
            from manifold_index.services.compute_service import ComputeService
            manifold_data, easy_result, nz_data = ComputeService.load_manifold(name)

            # Start worker to probe cache (thread-safe operation)
            self._worker = LoadWorker(
                name,
                manifold_data=manifold_data,
                easy_result=easy_result,
                nz_data=nz_data,
                parent=self
            )
            self._worker.status.connect(self._on_status)
            self._worker.finished.connect(self._on_finished)
            self._worker.error.connect(self._on_error)
            self._worker.start()
        except Exception as exc:
            self._on_error(str(exc))

    def _on_status(self, msg: str) -> None:
        self._status_label.setText(msg)

    def _on_finished(self, payload: dict) -> None:
        self._load_btn.setEnabled(True)
        self._status_label.setVisible(False)
        self._math_view.set_loading(False)

        manifold_data = payload["manifold_data"]
        easy_result   = payload["easy_result"]
        nz_data       = payload["nz_data"]
        cache_info    = payload["cache_info"]

        # ── Populate session ──────────────────────────────────────────
        s = self._session
        s.manifold_name = manifold_data.name
        s.manifold_data = manifold_data
        s.nz_data       = nz_data
        s.cache_status  = cache_info
        s.active_edges  = [True] * int(nz_data.num_hard)
        s.stage         = PipelineStage.LOADED

        self._render_loaded(s, easy_result=easy_result)
        self.session_updated.emit(s)

    def _on_error(self, msg: str) -> None:
        self._load_btn.setEnabled(True)
        self._status_label.setVisible(False)
        self._math_view.set_loading(False)

        name = self._name_edit.text().strip()
        vm = build_manifold_vm_error(name, ValueError(msg))
        self._card.set_status(CardStatus.ERROR)
        self._card.set_advisories(vm.advisories)
        self._math_view.set_html(
            f"<p class='warn'>Error: {msg}</p>"
            f"<p class='muted'>Try: m003, m004, 4_1, 5_2, …</p>"
        )
        self.load_failed.emit()

    # ------------------------------------------------------------------
    # Render helpers
    # ------------------------------------------------------------------

    def _render_loaded(self, session: Session, easy_result=None) -> None:
        md = session.manifold_data
        nz = session.nz_data

        # Build ViewModel (formatters fill HTML)
        try:
            nz_latex   = format_nz_latex(nz)
            gl_html    = format_gluing_table_html(md)
            easy_html  = format_easy_edges_html(easy_result) if easy_result is not None else ""
            hard_html  = format_hard_edges_html(easy_result) if easy_result is not None else ""
        except Exception:
            nz_latex = gl_html = easy_html = hard_html = ""

        vm = build_manifold_vm(
            md, None, nz, session.cache_status,
            nz_latex=nz_latex,
            gluing_table_html=gl_html,
            easy_edges_html=easy_html,
            hard_edges_html=hard_html,
        )

        # Advisories
        advisories = list(vm.advisories)
        cache_adv = _cache_advisory(session.cache_status)
        if cache_adv:
            advisories.insert(0, cache_adv)
        self._card.set_advisories(advisories)

        # Summary
        n_hard = vm.num_hard
        cache_parts = []
        if session.cache_status.get("iref", {}).get("available"):
            cache_parts.append("I^ref ✓")
        if session.cache_status.get("nc", {}).get("available"):
            cache_parts.append("NC ✓")
        cache_str = "  ·  cache: " + "  ".join(cache_parts) if cache_parts else ""
        summary = (
            f"{session.manifold_name}  ·  {vm.n_tetrahedra} tet  ·  "
            f"{vm.n_cusps} cusp  ·  {n_hard} hard{cache_str}"
        )
        self._card.set_summary(summary)
        self._card.set_status(CardStatus.DONE)

        # Math view body
        colors = sys_colors()
        body_html = ""
        if nz_latex:
            body_html += f"<h3>NZ Matrix</h3><p>{nz_latex}</p>"
        if gl_html:
            body_html += f"<h3>Gluing Equations</h3>{gl_html}"
        if hard_html:
            body_html += f"<h3>Hard Edges</h3>{hard_html}"
        if not body_html:
            body_html = (
                f"<p><b>{session.manifold_name}</b> loaded — "
                f"{vm.n_tetrahedra} tet, {vm.n_cusps} cusp, {n_hard} hard edge(s)</p>"
            )
        self._math_view.set_html(body_html)
        self._card.collapse()

