"""
app/gui.py — PySide6 GUI for the Refined 3D Index Calculator.

Architecture
────────────
Sidebar (fixed 180 px)  |  QStackedWidget (4 pages)
                        |
  1. Setup              |  SetupPage          — manifold name, Nmax
  2. Overview           |  OverviewPage       — H₁(∂N,ℤ/2) sectors + refined index
  3. Dehn Filling       |  DehnFillingPage    — slope input → auto-search → results
  4. Export             |  ExportPage         — batch export to txt/tex/json/nb

Workflow:
  SetupPage  → compute NZ + refined index for all ℤ/2 sectors
  OverviewPage  → display sectors + Weyl check
  DehnFillingPage  → user inputs slope → auto non-closable search
                     → basis change + slope transform → filled refined index

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
from manifold_index.app.pages.overview_page import OverviewPage
from manifold_index.app.pages.dehn_filling_page import DehnFillingPage
from manifold_index.app.pages.export_page import ExportPage
from manifold_index.app.workers import (
    RefinedIndexWorker,
    DehnFillingPipelineWorker,
)


# ---------------------------------------------------------------------------
# Evaluation grid builder
# ---------------------------------------------------------------------------

def _build_eval_grid(r: int) -> list[tuple[list[int], list]]:
    """Build the 25^r evaluation grid for refined index + Weyl extraction.

    For each cusp:
        m ∈ {-2, -1, 0, 1, 2}
        e ∈ {-1, -1/2, 0, 1/2, 1}

    The extended range (including negative values) is required so that
    ``compute_ab_vectors`` can find conjugate-charge pairs:

      • (m, 0) / (−m, 0)   →  determines **b**
      • (0, e) / (0, −e)   →  determines **a**

    Returns list of (m_ext, e_ext) tuples.
    """
    per_cusp = [
        (m, Fraction(k, 2))
        for m in (-2, -1, 0, 1, 2)
        for k in (-2, -1, 0, 1, 2)
    ]
    eval_points: list[tuple[list[int], list]] = []
    for combo in itertools_product(*([per_cusp] * r)):
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
        self._page_overview = OverviewPage()
        self._page_dehn = DehnFillingPage()
        self._page_export = ExportPage()
        self._page_export.set_results_page(self._page_overview)

        self._stack.addWidget(self._page_setup)      # 0
        self._stack.addWidget(self._page_overview)    # 1
        self._stack.addWidget(self._page_dehn)        # 2
        self._stack.addWidget(self._page_export)      # 3

        # ── State ─────────────────────────────────────────────────
        self._nz_data = None
        self._manifold_name: str = ""
        self._q_order_half: int = 10

        # ── Workers ───────────────────────────────────────────────
        self._refined_worker: RefinedIndexWorker | None = None
        self._dehn_pipeline_worker: DehnFillingPipelineWorker | None = None

        # ── Connections ───────────────────────────────────────────
        self._sidebar.page_requested.connect(self._go_to_page)
        self._page_setup.run_requested.connect(self._start_compute)
        self._page_overview.continue_requested.connect(lambda: self._go_to_page(2))
        self._page_overview.back_requested.connect(lambda: self._go_to_page(0))
        self._page_dehn.compute_requested.connect(self._start_dehn_pipeline)
        self._page_dehn.back_requested.connect(lambda: self._go_to_page(1))

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
            name = self._page_setup.manifold_name()
            if name:
                self._page_export.prepare(name)

    # ------------------------------------------------------------------
    # Initial compute  (NZ + refined index for ℤ/2 sectors)
    # ------------------------------------------------------------------

    @Slot(str, int)
    def _start_compute(self, name: str, q_order_half: int) -> None:
        """Load manifold, build NZ data, compute refined index for H₁ sectors."""
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

        self._manifold_name = name
        self._q_order_half = q_order_half
        self._nz_data = nz
        self._manifold_data = data
        self._easy_result = easy

        self.statusBar().showMessage(
            f"Loaded '{name}' — computing refined index for ℤ/2 sectors …"
        )

        # Build evaluation grid  (25^r points for Weyl extraction)
        eval_points = _build_eval_grid(nz.r)

        # Switch to overview page
        self._sidebar.enable_up_to(1)
        self._go_to_page(1)
        self._page_overview.reset(name, nz, q_order_half)

        # Launch refined index worker
        worker = RefinedIndexWorker(
            nz_data=nz,
            eval_points=eval_points,
            q_order_half=q_order_half,
        )
        worker.status.connect(self._page_overview.update_status)
        worker.progress.connect(self._page_overview.update_progress)
        worker.finished.connect(self._on_refined_finished)
        worker.error.connect(self._on_refined_error)

        self._refined_worker = worker
        worker.start()

    @Slot(object)
    def _on_refined_finished(self, result: list) -> None:
        self._page_overview.computation_finished(result)
        # Unlock Dehn filling and export pages
        self._sidebar.enable_up_to(3)
        # Prepare the Dehn filling page with manifold data and Weyl info
        weyl_result = self._page_overview.weyl_result
        self._page_dehn.reset(
            self._manifold_name, self._nz_data, self._q_order_half,
            manifold_data=getattr(self, '_manifold_data', None),
            easy_result=getattr(self, '_easy_result', None),
            weyl_result=weyl_result,
        )
        self.statusBar().showMessage("Refined index computation complete.")

    @Slot(str)
    def _on_refined_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Refined index error", msg)

    # ------------------------------------------------------------------
    # Dehn filling pipeline
    # ------------------------------------------------------------------

    @Slot(object, int, int, int, int, object, object)
    def _start_dehn_pipeline(
        self,
        nz_data,
        cusp_idx: int,
        P_user: int,
        Q_user: int,
        q_order_half: int,
        p_range: range,
        q_range: range,
    ) -> None:
        """Launch the full Dehn filling pipeline."""
        self.statusBar().showMessage(
            f"Dehn filling cusp {cusp_idx} at ({P_user}, {Q_user}) …"
        )

        # Extract Weyl (a, b) vectors from the overview page's Weyl check
        weyl_result = self._page_overview.weyl_result
        weyl_a = None
        weyl_b = None
        if weyl_result is not None and weyl_result.ab is not None and weyl_result.ab.is_valid:
            weyl_a = list(weyl_result.ab.a)
            weyl_b = list(weyl_result.ab.b)

        worker = DehnFillingPipelineWorker(
            nz_data=nz_data,
            cusp_idx=cusp_idx,
            P_user=P_user,
            Q_user=Q_user,
            q_order_half=q_order_half,
            p_range=p_range,
            q_range=q_range,
            weyl_a=weyl_a,
            weyl_b=weyl_b,
        )
        worker.status.connect(self._page_dehn.update_status)
        worker.progress.connect(self._page_dehn.update_progress)
        worker.finished.connect(self._page_dehn.dehn_filling_finished)
        worker.finished.connect(
            lambda _: self.statusBar().showMessage("Dehn filling complete.")
        )
        worker.error.connect(self._on_dehn_error)

        self._dehn_pipeline_worker = worker
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
