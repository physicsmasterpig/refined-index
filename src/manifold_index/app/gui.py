"""
app/gui.py — PySide6 GUI for the Refined 3D Index Calculator.

Architecture
────────────
Sidebar (fixed 180 px)  |  QStackedWidget (4 pages)
                        |
  1. Setup              |  SetupPage       — manifold name, parameters
  2. Basis              |  BasisPage       — pipeline progress + cycle selection
  3. Results            |  ResultsPage     — tabbed: Series · Weyl & Dehn · Raw
  4. Export             |  ExportPage      — batch export to txt/tex/json/nb

Entry point:
    from manifold_index.app.gui import launch_gui
    launch_gui()
"""

from __future__ import annotations

import traceback
from fractions import Fraction
from itertools import product as itertools_product

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QWidget,
)

from manifold_index.app.style import APP_STYLESHEET
from manifold_index.app.widgets.sidebar import Sidebar
from manifold_index.app.pages.setup_page import SetupPage
from manifold_index.app.pages.basis_page import BasisPage
from manifold_index.app.pages.results_page import ResultsPage
from manifold_index.app.pages.export_page import ExportPage
from manifold_index.app.workers import (
    PipelineResult,
    PipelineWorker,
    RefinedIndexWorker,
    DehnFillingWorker,
)
from manifold_index.core.basis_selection import (
    BasisSelection,
    apply_basis_changes,
    make_basis_selection,
)


# ---------------------------------------------------------------------------
# Evaluation grid builder (moved from old gui.py, cleaned up)
# ---------------------------------------------------------------------------

def _build_eval_grid(
    bs: BasisSelection,
    nz_changed,
    m_max: int,
    e_max: int,
) -> list[tuple[list[int], list]]:
    """Build the list of (m_ext, e_ext) evaluation points.

    For each cusp:
        m ∈ {−m_max, …, +m_max}           (step 1, integers)
        e ∈ {−e_max, −e_max+½, …, +e_max} (step ½, half-integers)

    Then Cartesian product across cusps.
    """
    per_cusp: list[list[tuple[int, Fraction]]] = []
    for _ch in bs.choices:
        m_vals = list(range(-m_max, m_max + 1))
        e_vals = [Fraction(k, 2) for k in range(-2 * e_max, 2 * e_max + 1)]
        per_cusp.append([(m, e) for m in m_vals for e in e_vals])

    eval_points: list[tuple[list[int], list]] = []
    for combo in itertools_product(*per_cusp):
        m_list = [pair[0] for pair in combo]
        e_list = [pair[1] for pair in combo]
        eval_points.append((m_list, e_list))
    return eval_points


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """Two-panel main window: sidebar + stacked pages."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Refined 3D Index Calculator")
        self.setMinimumSize(880, 640)
        self.resize(1000, 720)

        # ── Central widget ────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        hlayout = QHBoxLayout(central)
        hlayout.setContentsMargins(0, 0, 0, 0)
        hlayout.setSpacing(0)

        # Sidebar
        self._sidebar = Sidebar()
        hlayout.addWidget(self._sidebar)

        # Pages
        self._stack = QStackedWidget()
        hlayout.addWidget(self._stack, 1)

        self._page_setup = SetupPage()
        self._page_basis = BasisPage()
        self._page_results = ResultsPage()
        self._page_export = ExportPage()
        self._page_export.set_results_page(self._page_results)

        self._stack.addWidget(self._page_setup)    # 0
        self._stack.addWidget(self._page_basis)    # 1
        self._stack.addWidget(self._page_results)  # 2
        self._stack.addWidget(self._page_export)   # 3

        # ── Workers ───────────────────────────────────────────────
        self._pipeline_worker: PipelineWorker | None = None
        self._refined_worker: RefinedIndexWorker | None = None
        self._dehn_worker: DehnFillingWorker | None = None

        # ── Connections ───────────────────────────────────────────
        self._sidebar.page_requested.connect(self._go_to_page)
        self._page_setup.run_requested.connect(self._start_pipeline)
        self._page_basis.compute_requested.connect(self._start_refined_index)
        self._page_basis.back_requested.connect(lambda: self._go_to_page(0))
        self._page_results.dehn_fill_requested.connect(self._start_dehn_filling)

        # ── Keyboard shortcuts ────────────────────────────────────
        QShortcut(QKeySequence("Ctrl+1"), self, lambda: self._go_to_page(0))
        QShortcut(QKeySequence("Ctrl+2"), self, lambda: self._go_to_page(1))
        QShortcut(QKeySequence("Ctrl+3"), self, lambda: self._go_to_page(2))
        QShortcut(QKeySequence("Ctrl+4"), self, lambda: self._go_to_page(3))
        QShortcut(QKeySequence("Ctrl+E"), self, lambda: self._go_to_page(3))

        # Status bar
        self.statusBar().showMessage("Ready")

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    @Slot(int)
    def _go_to_page(self, index: int) -> None:
        if index < 0 or index >= self._stack.count():
            return
        btn = self._sidebar._buttons[index]  # noqa: access for enabled check
        if not btn.isEnabled():
            return
        self._stack.setCurrentIndex(index)
        self._sidebar.set_active(index)
        if index == 3:
            # Prepare export page with current manifold name
            name = self._page_setup.manifold_name()
            if name:
                self._page_export.prepare(name)

    # ------------------------------------------------------------------
    # Pipeline launch  (Step 4-5)
    # ------------------------------------------------------------------

    @Slot(str, int, object, object)
    def _start_pipeline(
        self,
        name: str,
        q_order_half: int,
        p_range: range,
        q_range: range,
    ) -> None:
        # Steps 1-3 run in the main thread (SnaPy SQLite thread-safety)
        from manifold_index.core.manifold import load_manifold
        from manifold_index.core.phase_space import find_easy_edges
        from manifold_index.core.neumann_zagier import build_neumann_zagier

        try:
            data = load_manifold(name)
            easy = find_easy_edges(data)
            nz = build_neumann_zagier(data, easy)
        except Exception as exc:
            tb = traceback.format_exc()
            QMessageBox.critical(
                self, "Pipeline error",
                f"Failed to load '{name}':\n{exc}\n\n{tb}",
            )
            return

        self.statusBar().showMessage(f"Loaded '{name}' — searching slopes …")

        # Keep raw data for the full report export
        self._manifold_data = data
        self._easy_result = easy

        # Switch to basis page
        self._sidebar.enable_up_to(1)
        self._go_to_page(1)
        self._page_basis.reset()
        self._page_basis.update_status(
            f"Loaded '{name}' ({nz.r} cusp(s)).  Searching slopes …"
        )

        worker = PipelineWorker(
            name=name,
            nz_data=nz,
            q_order_half=q_order_half,
            p_range=p_range,
            q_range=q_range,
        )
        worker.status.connect(self._page_basis.update_status)
        worker.slope_progress.connect(self._page_basis.update_slope_progress)
        worker.finished.connect(self._attach_raw_data)
        worker.finished.connect(self._page_basis.pipeline_finished)
        worker.finished.connect(
            lambda _: self.statusBar().showMessage(f"'{name}' pipeline complete.")
        )
        worker.error.connect(self._on_pipeline_error)

        self._pipeline_worker = worker
        worker.start()

    @Slot(str)
    def _on_pipeline_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Pipeline error", msg)
        self._go_to_page(0)

    @Slot(object)
    def _attach_raw_data(self, result: PipelineResult) -> None:
        """Attach ManifoldData and EasyEdgeResult to the pipeline result."""
        result.manifold_data = getattr(self, "_manifold_data", None)
        result.easy_result = getattr(self, "_easy_result", None)

    # ------------------------------------------------------------------
    # Refined index launch  (Step 8)
    # ------------------------------------------------------------------

    @Slot(object, object, int, int)
    def _start_refined_index(
        self,
        pipeline_result: PipelineResult,
        pq_choices: list,
        m_max: int,
        e_max: int,
    ) -> None:
        nz = pipeline_result.nz_data

        try:
            bs = make_basis_selection(
                nz, pipeline_result.cycle_results, pq_choices, strict=False,
            )
        except ValueError as exc:
            QMessageBox.critical(self, "Basis selection error", str(exc))
            return

        try:
            nz_changed = apply_basis_changes(nz, bs)
        except Exception as exc:
            QMessageBox.critical(self, "Basis change error", str(exc))
            return

        eval_points = _build_eval_grid(bs, nz_changed, m_max, e_max)

        # Switch to results page
        self._sidebar.enable_up_to(2)
        self._go_to_page(2)
        self._page_results.reset(pipeline_result, bs, nz_changed)
        self.statusBar().showMessage(
            f"Computing refined index — {len(eval_points)} evaluation point(s) …"
        )

        worker = RefinedIndexWorker(
            nz_data=nz_changed,
            eval_points=eval_points,
            q_order_half=pipeline_result.q_order_half,
        )
        worker.status.connect(self._page_results.update_status)
        worker.progress.connect(self._page_results.update_progress)
        worker.finished.connect(self._on_refined_finished)
        worker.error.connect(self._on_refined_error)

        self._refined_worker = worker
        worker.start()

    @Slot(object)
    def _on_refined_finished(self, result: list) -> None:
        self._page_results.computation_finished(result)
        self._sidebar.enable_up_to(3)  # unlock export page
        self.statusBar().showMessage("Computation complete.")

    @Slot(str)
    def _on_refined_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Refined index error", msg)

    # ------------------------------------------------------------------
    # Dehn filling launch
    # ------------------------------------------------------------------

    @Slot(object, int, int, int, int, int)
    def _start_dehn_filling(
        self,
        nz_data,
        cusp_idx: int,
        P: int,
        Q: int,
        q_order_half: int,
        eta_order: int,
    ) -> None:
        self.statusBar().showMessage(
            f"Computing refined Dehn filling at slope ({P}, {Q}) …"
        )
        worker = DehnFillingWorker(
            nz_data=nz_data,
            cusp_idx=cusp_idx,
            P=P,
            Q=Q,
            q_order_half=q_order_half,
            eta_order=eta_order,
        )
        worker.status.connect(self._page_results.update_status)
        worker.finished.connect(self._page_results.dehn_filling_finished)
        worker.finished.connect(
            lambda _: self.statusBar().showMessage("Dehn filling complete.")
        )
        worker.error.connect(self._on_dehn_error)

        self._dehn_worker = worker
        worker.start()

    @Slot(str)
    def _on_dehn_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Dehn filling error", msg)
        self.statusBar().showMessage("Dehn filling failed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def launch_gui() -> None:
    """Create the QApplication and show the main window."""
    import sys

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Refined 3D Index Calculator")
    app.setOrganizationName("RefinedIndex")
    app.setStyleSheet(APP_STYLESHEET)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
