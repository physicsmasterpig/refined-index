"""
app/panels/manifold_panel.py — Panel 1: Manifold Analysis.

Input manifold name + Nmax → load + compute NZ + refined index → display.
"""

from __future__ import annotations

import traceback

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from manifold_index.app.katex import make_math_view, update_math_view
from manifold_index.app.formatters import format_panel1_html


# ═══════════════════════════════════════════════════════════════════════════

def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("sectionTitle")
    return lbl


class ManifoldPanel(QFrame):
    """Panel 1: manifold analysis.

    Signals
    -------
    compute_requested(str, int)
        Emitted when the user clicks Compute.  Args: (manifold_name, q_order_half).
    data_ready(object)
        Emitted after NZ data + entries are fully available, carrying a dict:
        {manifold_data, easy_result, nz_data, entries, weyl_result, q_order_half}.
    """

    compute_requested = Signal(str, int)
    data_ready = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        self._math_view: QWidget | None = None
        self._entries = None
        self._nz_data = None
        self._md = None
        self._ps = None
        self._weyl = None
        self._q_order_half = 10
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 12, 16, 12)
        outer.setSpacing(8)

        # Title
        t = QLabel("① Manifold Analysis")
        t.setObjectName("panelTitle")
        outer.addWidget(t)

        sub = QLabel("Input a manifold name to load SnaPy data and compute refined index.")
        sub.setObjectName("panelSubtitle")
        sub.setWordWrap(True)
        outer.addWidget(sub)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        outer.addWidget(sep)

        # ── Input row ─────────────────────────────────────────
        input_box = QWidget()
        input_row = QHBoxLayout(input_box)
        input_row.setContentsMargins(0, 0, 0, 0)
        input_row.setSpacing(8)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g.  m003  m004  4_1  L5a1  …")
        self._name_edit.setFixedHeight(32)
        input_row.addWidget(self._name_edit, 1)

        input_row.addWidget(QLabel("Nmax:"))
        self._nmax_spin = QSpinBox()
        self._nmax_spin.setRange(4, 60)
        self._nmax_spin.setValue(10)
        self._nmax_spin.setFixedWidth(60)
        input_row.addWidget(self._nmax_spin)

        self._compute_btn = QPushButton("Compute")
        self._compute_btn.setObjectName("primary")
        self._compute_btn.setFixedHeight(32)
        input_row.addWidget(self._compute_btn)

        outer.addWidget(input_box)

        # ── Progress ──────────────────────────────────────────
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFixedHeight(12)
        self._progress.hide()
        outer.addWidget(self._progress)

        self._status = QLabel("")
        self._status.setStyleSheet("font-size: 11px;")
        self._status.hide()
        outer.addWidget(self._status)

        # ── Scrollable content area ───────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll_widget = QWidget()
        self._content = QVBoxLayout(scroll_widget)
        self._content.setContentsMargins(0, 4, 0, 4)
        self._content.setSpacing(6)
        scroll.setWidget(scroll_widget)
        outer.addWidget(scroll, 1)

        # ── Connections ───────────────────────────────────────
        self._compute_btn.clicked.connect(self._on_compute_clicked)
        self._name_edit.returnPressed.connect(self._on_compute_clicked)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    @Slot()
    def _on_compute_clicked(self) -> None:
        name = self._name_edit.text().strip()
        if not name:
            return
        q_order_half = self._nmax_spin.value() * 2
        self.compute_requested.emit(name, q_order_half)

    # ------------------------------------------------------------------
    # Public methods called by the MainWindow
    # ------------------------------------------------------------------

    def set_loading(self, name: str) -> None:
        """Show loading state while NZ data is being built."""
        self._progress.show()
        self._progress.setRange(0, 0)  # indeterminate
        self._status.show()
        self._status.setText(f"Loading {name}…")
        self._status.setStyleSheet("font-size: 11px; color: palette(text);")
        self._compute_btn.setEnabled(False)

    def show_nz_data(
        self,
        md,
        ps,
        nz,
    ) -> None:
        """Display the manifold info + NZ matrix (before refined index is done)."""
        self._md = md
        self._ps = ps
        self._nz_data = nz

        html = format_panel1_html(md, ps, nz)
        self._set_math_content(html)

        # Switch to indeterminate until the worker's first progress signal
        self._progress.setRange(0, 0)
        self._status.setText("Computing refined index…")

    def update_progress(self, done: int, total: int) -> None:
        """Update progress bar during refined index computation."""
        self._progress.setRange(0, total)
        self._progress.setValue(done)
        self._status.setText(f"Computing sector {done}/{total}…")

    def update_status(self, msg: str) -> None:
        """Update status label."""
        self._status.setText(msg)

    def computation_finished(
        self,
        entries: list,
        weyl_result=None,
    ) -> None:
        """Called when all refined index computations are done."""
        self._entries = entries
        self._weyl = weyl_result

        total = len(entries)
        n_nonzero = sum(1 for _, _, r in entries if r)

        html = format_panel1_html(
            self._md, self._ps, self._nz_data,
            entries=entries,
            weyl_result=weyl_result,
            max_q_terms=self._nmax_spin.value(),
        )
        self._set_math_content(html)

        self._progress.setRange(0, total)
        self._progress.setValue(total)
        self._status.setText(f"✓  {total} sectors computed  ·  {self._md.name}")
        self._status.setStyleSheet("color: #2ea043; font-size: 11px;")
        self._compute_btn.setEnabled(True)

        # Notify other panels
        self.data_ready.emit({
            "manifold_data": self._md,
            "easy_result": self._ps,
            "nz_data": self._nz_data,
            "entries": entries,
            "weyl_result": weyl_result,
            "q_order_half": self._nmax_spin.value() * 2,
        })

    def set_error(self, msg: str) -> None:
        """Show an error message."""
        self._progress.hide()
        self._status.show()
        self._status.setText(f"✗  {msg}")
        self._status.setStyleSheet("color: #cf222e; font-size: 11px;")
        self._compute_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def manifold_name(self) -> str:
        return self._name_edit.text().strip()

    @property
    def nz_data(self):
        return self._nz_data

    @property
    def entries(self):
        return self._entries

    @property
    def weyl_result(self):
        return self._weyl

    @property
    def q_order_half(self) -> int:
        return self._nmax_spin.value() * 2

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _set_math_content(self, html_body: str) -> None:
        """Create or update the KaTeX view."""
        if self._math_view is not None:
            update_math_view(self._math_view, html_body)
        else:
            self._math_view = make_math_view(html_body, min_h=400)
            self._content.addWidget(self._math_view, 1)
