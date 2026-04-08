"""
app/window.py — Main window for the v0.4.2 tabbed GUI.

Tabs:
  1. Calculator — three-panel (Manifold | Dehn Filling | Export)
  2. Kernel Builder — precompute filling kernels for selected slopes
  3. Data Packs — browse and download pre-computed data packs

Run with:
    python -m manifold_index.app
"""

from __future__ import annotations

import sys
import traceback

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QTabWidget,
    QWidget,
    QHBoxLayout,
)

from manifold_index.app.style import APP_STYLESHEET
from manifold_index.app.panels.manifold_panel import ManifoldPanel
from manifold_index.app.panels.filling_panel import FillingPanel
from manifold_index.app.panels.export_panel import ExportPanel
from manifold_index.app.panels.kernel_panel import KernelPanel
from manifold_index.app.panels.data_panel import DataPanel
from manifold_index.app.workers import (
    RefinedIndexWorker,
    DehnFillingWorker,
    build_eval_grid,
)


class MainWindow(QMainWindow):
    """Tabbed main window: Calculator | Kernel Builder."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Refined 3D Index Calculator — v0.4.2")
        self.setMinimumSize(1200, 700)
        self.resize(1500, 850)

        # ── Central tab widget ────────────────────────────────
        self._tabs = QTabWidget()
        self.setCentralWidget(self._tabs)

        # ── Tab 1: Calculator (three-panel splitter) ──────────
        calc_page = QWidget()
        calc_layout = QHBoxLayout(calc_page)
        calc_layout.setContentsMargins(8, 8, 8, 8)
        calc_layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(6)

        self._panel1 = ManifoldPanel()
        self._panel2 = FillingPanel()
        self._panel3 = ExportPanel()

        splitter.addWidget(self._panel1)
        splitter.addWidget(self._panel2)
        splitter.addWidget(self._panel3)

        # Initial sizes: ~45% / ~35% / ~20%
        splitter.setSizes([540, 420, 240])

        calc_layout.addWidget(splitter)
        self._tabs.addTab(calc_page, "🧮  Calculator")

        # ── Tab 2: Kernel Builder ─────────────────────────────
        self._kernel_panel = KernelPanel()
        self._kernel_panel.build_finished.connect(
            lambda msg: self.statusBar().showMessage(msg)
        )
        self._tabs.addTab(self._kernel_panel, "🗄  Kernel Builder")

        # ── Tab 3: Data Packs ─────────────────────────────────
        self._data_panel = DataPanel()
        self._tabs.addTab(self._data_panel, "📦  Data Packs")

        # ── State ─────────────────────────────────────────────
        self._nz_data = None
        self._q_order_half: int = 10
        self._refined_worker: RefinedIndexWorker | None = None
        self._dehn_worker: DehnFillingWorker | None = None

        # ── Connections ───────────────────────────────────────
        self._panel1.compute_requested.connect(self._start_compute)
        self._panel1.data_ready.connect(self._on_panel1_ready)
        self._panel2.fill_requested.connect(self._start_dehn_filling)

        self.statusBar().showMessage("Ready — enter a manifold name and click Compute.")

    # ==================================================================
    # Panel 1 → Compute pipeline
    # ==================================================================

    @Slot(str, int)
    def _start_compute(self, name: str, q_order_half: int) -> None:
        """Load manifold on main thread, then launch refined index worker."""
        from manifold_index.core.manifold import load_manifold
        from manifold_index.core.phase_space import find_phase_space_basis
        from manifold_index.core.neumann_zagier import build_neumann_zagier

        self._panel1.set_loading(name)
        self.statusBar().showMessage(f"Loading '{name}'…")

        try:
            md = load_manifold(name)
            ps = find_phase_space_basis(md)
            nz = build_neumann_zagier(md, ps)
        except Exception as exc:
            tb = traceback.format_exc()
            self._panel1.set_error(str(exc))
            QMessageBox.critical(
                self, "Pipeline error",
                f"Failed to load '{name}':\n{exc}\n\n{tb}",
            )
            return

        self._nz_data = nz
        self._q_order_half = q_order_half

        # Show NZ data immediately
        self._panel1.show_nz_data(md, ps, nz)
        self.statusBar().showMessage(
            f"Loaded '{name}' — computing refined index ({5 ** nz.r} display + {25 ** nz.r} Weyl)…"
        )

        # Build 25^r evaluation grid
        eval_points = build_eval_grid(nz.r)

        # Launch worker
        worker = RefinedIndexWorker(nz, eval_points, q_order_half)
        worker.status.connect(self._panel1.update_status)
        worker.progress.connect(self._panel1.update_progress)
        worker.finished.connect(self._on_refined_finished)
        worker.error.connect(self._on_refined_error)

        self._refined_worker = worker
        worker.start()

    @Slot(object)
    def _on_refined_finished(self, results: list) -> None:
        """Refined index computation done — run Weyl checks and update UI."""
        from manifold_index.core.weyl_check import run_weyl_checks

        nz = self._nz_data
        entries = results  # list of (m_ext, e_ext, result)

        # Run Weyl checks
        try:
            weyl_result = run_weyl_checks(
                entries, nz.num_hard,
                q_order_half=self._q_order_half,
            )
        except Exception:
            weyl_result = None

        self._panel1.computation_finished(entries, weyl_result)
        self.statusBar().showMessage(
            f"✓  Refined index complete — {len(entries)} sectors computed."
        )

    @Slot(str)
    def _on_refined_error(self, msg: str) -> None:
        self._panel1.set_error(msg)
        QMessageBox.critical(self, "Refined index error", msg)

    # ==================================================================
    # Panel 1 data ready → unlock Panels 2 & 3
    # ==================================================================

    @Slot(object)
    def _on_panel1_ready(self, data: dict) -> None:
        """Panel 1 has all results — configure Panels 2 and 3."""
        self._panel2.reset(data)
        self._panel3.set_data(data)

    # ==================================================================
    # Panel 2 → Dehn filling pipeline
    # ==================================================================

    @Slot(object)
    def _start_dehn_filling(self, payload: dict) -> None:
        """Launch the unified Dehn filling worker (fill → NC search → NC fill)."""
        self._panel2.set_loading()
        self.statusBar().showMessage("Dehn filling…")

        nz = payload["nz_data"]
        cusp_configs = payload["cusp_configs"]
        q_order_half = payload["q_order_half"]
        p_range = payload["p_range"]
        q_range = payload["q_range"]
        manifold_name = payload.get("manifold_name", "unknown")

        worker = DehnFillingWorker(
            nz_data=nz,
            cusp_configs=cusp_configs,
            q_order_half=q_order_half,
            p_range=p_range,
            q_range=q_range,
            manifold_name=manifold_name,
        )
        worker.status.connect(self._panel2.update_status)
        worker.progress.connect(self._panel2.update_progress)
        worker.nc_found.connect(self._panel2.nc_search_done)
        worker.finished.connect(self._panel2.filling_finished)
        worker.finished.connect(self._panel3.set_dehn_data)
        worker.error.connect(self._on_dehn_error)
        worker.finished.connect(
            lambda _: self.statusBar().showMessage("✓  Dehn filling complete.")
        )

        self._dehn_worker = worker
        worker.start()

    @Slot(str)
    def _on_dehn_error(self, msg: str) -> None:
        self._panel2.set_error(msg)
        QMessageBox.critical(self, "Dehn filling error", msg)

    # ==================================================================
    # Cleanup — prevent macOS PySide6 malloc double-free on shutdown
    # ==================================================================

    def closeEvent(self, event) -> None:  # noqa: N802
        """Ensure background workers are stopped before Qt tears down."""
        for w in (self._refined_worker, self._dehn_worker):
            if w is not None and w.isRunning():
                w.quit()
                w.wait(2000)
        # Stop any kernel builder running inside the kernel panel
        if hasattr(self._kernel_panel, '_worker'):
            kw = self._kernel_panel._worker
            if kw is not None and kw.isRunning():
                kw.cancel_requested = True
                kw.quit()
                kw.wait(2000)
        # Stop any data pack download in progress
        if hasattr(self._data_panel, '_worker'):
            dw = self._data_panel._worker
            if dw is not None and dw.isRunning():
                dw.quit()
                dw.wait(2000)
        super().closeEvent(event)


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def launch_gui() -> None:
    """Create the QApplication and show the main window."""
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Refined 3D Index Calculator v0.4.2")
    app.setOrganizationName("RefinedIndex")
    app.setStyleSheet(APP_STYLESHEET)

    window = MainWindow()
    window.show()

    # Run the event loop, then clean up *before* sys.exit so that
    # Qt C++ destructors run while the Python objects are still alive.
    # This prevents the macOS "pointer being freed was not allocated"
    # crash in PySide6 ≤ 6.10.
    ret = app.exec()
    del window
    sys.exit(ret)


if __name__ == "__main__":
    launch_gui()
