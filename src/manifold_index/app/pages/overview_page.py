"""
pages/overview_page.py — Page 2: Refined index overview for H₁(∂N, ℤ/2) sectors.

Displays:
  • Computation progress bar
  • H₁(∂N, ℤ/2) basis description
  • Refined index for each evaluation point
  • Weyl symmetry information
"""

from __future__ import annotations

from fractions import Fraction

from PySide6.QtCore import Signal, Slot, Qt
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from manifold_index.app.style import monospace_font
from manifold_index.app.formatters import _fmt_frac


class OverviewPage(QWidget):
    """Page 2: H₁(∂N, ℤ/2) sectors and refined index overview."""

    continue_requested = Signal()
    back_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._nz_data = None
        self._refined_results: list | None = None
        self._weyl_result = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(24, 24, 24, 24)

        # Header
        hdr = QHBoxLayout()
        self._title_label = QLabel("Overview")
        self._title_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        hdr.addWidget(self._title_label)
        hdr.addStretch()
        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: palette(mid); font-size: 11px;")
        self._info_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        hdr.addWidget(self._info_label)
        root.addLayout(hdr)

        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self._progress_bar.setFixedHeight(14)
        root.addWidget(self._progress_bar)

        # Status
        self._status_label = QLabel("Waiting for computation …")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("font-size: 11px; color: palette(mid);")
        root.addWidget(self._status_label)

        # H₁ basis info
        self._h1_group = QGroupBox("H₁(∂N, ℤ/2)")
        h1_vbox = QVBoxLayout(self._h1_group)
        self._h1_label = QLabel("(not yet computed)")
        self._h1_label.setFont(monospace_font(11))
        self._h1_label.setWordWrap(True)
        h1_vbox.addWidget(self._h1_label)
        self._h1_group.hide()
        root.addWidget(self._h1_group)

        # Weyl info
        self._weyl_group = QGroupBox("Weyl Symmetry")
        weyl_vbox = QVBoxLayout(self._weyl_group)
        self._weyl_label = QLabel("(not yet computed)")
        self._weyl_label.setFont(monospace_font(11))
        self._weyl_label.setWordWrap(True)
        weyl_vbox.addWidget(self._weyl_label)
        self._weyl_group.hide()
        root.addWidget(self._weyl_group)

        # Refined index per sector (scrollable)
        self._series_group = QGroupBox("Refined Index per Sector")
        series_vbox = QVBoxLayout(self._series_group)

        copy_row = QHBoxLayout()
        copy_row.addStretch()
        copy_btn = QPushButton("⎘  Copy to Clipboard")
        copy_btn.setObjectName("secondary")
        copy_btn.setFixedHeight(28)
        copy_btn.clicked.connect(self._on_copy)
        copy_row.addWidget(copy_btn)
        series_vbox.addLayout(copy_row)

        self._series_edit = QTextEdit()
        self._series_edit.setReadOnly(True)
        self._series_edit.setFont(monospace_font())
        self._series_edit.setMinimumHeight(180)
        series_vbox.addWidget(self._series_edit, 1)

        self._series_group.hide()
        root.addWidget(self._series_group, 1)

        # Buttons
        btn_row = QHBoxLayout()
        back_btn = QPushButton("← Back")
        back_btn.setObjectName("secondary")
        back_btn.setFixedHeight(32)
        back_btn.clicked.connect(self.back_requested.emit)
        btn_row.addWidget(back_btn)
        btn_row.addStretch()

        self._continue_btn = QPushButton("Continue to Dehn Filling  ▶")
        self._continue_btn.setObjectName("primary")
        self._continue_btn.setFixedHeight(40)
        self._continue_btn.setEnabled(False)
        self._continue_btn.clicked.connect(self.continue_requested.emit)
        btn_row.addWidget(self._continue_btn)
        root.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, name: str, nz_data, q_order_half: int) -> None:
        """Prepare page for a new computation."""
        self._nz_data = nz_data
        self._refined_results = None
        self._weyl_result = None
        self._title_label.setText(f"Overview — {name}")
        self._info_label.setText(
            f"{nz_data.n} tet  ·  {nz_data.r} cusp(s)  ·  "
            f"{nz_data.num_hard} hard edge(s)  ·  Nmax = {q_order_half}"
        )
        self._status_label.setText("Computing refined index for ℤ/2 sectors …")
        self._status_label.setStyleSheet("font-size: 11px; color: palette(mid);")
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self._progress_bar.show()
        self._h1_group.hide()
        self._weyl_group.hide()
        self._series_group.hide()
        self._series_edit.clear()
        self._continue_btn.setEnabled(False)

        # Show H₁ basis right away
        self._populate_h1_basis(nz_data)

    @Slot(str)
    def update_status(self, msg: str) -> None:
        self._status_label.setText(msg)

    @Slot(int, int)
    def update_progress(self, done: int, total: int) -> None:
        self._progress_bar.setRange(0, total)
        self._progress_bar.setValue(done)

    @Slot(object)
    def computation_finished(self, results: list) -> None:
        """Called when all H₁(∂N, ℤ/2) sector computations are done.

        Parameters
        ----------
        results : list of (m_ext, e_ext, RefinedIndexResult)
        """
        self._refined_results = results
        self._progress_bar.hide()

        nz = self._nz_data
        if nz is None:
            return

        # Weyl check
        try:
            from manifold_index.core.weyl_check import run_weyl_checks
            self._weyl_result = run_weyl_checks(results, nz.num_hard)
        except Exception:
            self._weyl_result = None

        self._populate_weyl_info()
        self._populate_series(results, nz.num_hard)
        self._continue_btn.setEnabled(True)

        self._status_label.setText(
            f"✓  {len(results)} sector(s) computed."
        )
        self._status_label.setStyleSheet("color: #2ea043; font-size: 11px;")

    # ------------------------------------------------------------------
    # Populate sections
    # ------------------------------------------------------------------

    def _populate_h1_basis(self, nz) -> None:
        """Show H₁(∂N, ℤ/2) = Span{α₁,β₁, …, αᵣ,βᵣ}."""
        r = nz.r
        if r == 0:
            self._h1_label.setText("Closed manifold — no boundary.")
            self._h1_group.show()
            return

        parts = []
        for i in range(r):
            parts.append(f"α_{i} (meridian)")
            parts.append(f"β_{i} (longitude)")
        basis_str = ",  ".join(parts)

        lines = [
            f"H₁(∂N, ℤ/2) = (ℤ/2)^{{{2*r}}}",
            f"Basis: {{ {basis_str} }}",
            "",
            f"Evaluation grid per cusp:",
            f"  m_i ∈ {{-2, -1, 0, 1, 2}},  e_i ∈ {{-1, -1/2, 0, 1/2, 1}}",
            f"Total evaluation points: 25^{r} = {25**r}",
        ]
        self._h1_label.setText("\n".join(lines))
        self._h1_group.show()

    def _populate_weyl_info(self) -> None:
        weyl = self._weyl_result
        if weyl is None or weyl.ab is None:
            self._weyl_label.setText("Weyl parameters could not be determined.")
            self._weyl_group.show()
            return

        a_half = [Fraction(1, 2) * av for av in weyl.ab.a]
        b_vals = list(weyl.ab.b)
        a_str = "  ".join(f"a_{i} = {_fmt_frac(v)}" for i, v in enumerate(a_half))
        b_str = "  ".join(f"b_{i} = {_fmt_frac(v)}" for i, v in enumerate(b_vals))

        compat = all(v.denominator == 1 for v in a_half)
        if compat:
            compat_str = "✓  a ∈ ℤ — Dehn-filling compatible"
            color = "#2ea043"
        else:
            compat_str = "✗  a ∉ ℤ — not directly compatible (slope Q must be even)"
            color = "#d1242f"

        text = (
            f"Convention: f(η) = η^{{b·m + a·e}} · I(m,e),  f(η) = f(η⁻¹)\n"
            f"{a_str}\n{b_str}\n\n{compat_str}"
        )
        self._weyl_label.setText(text)
        self._weyl_group.show()

    def _populate_series(self, results: list, num_hard: int) -> None:
        """Format and display the refined index for each ℤ/2 sector."""
        from manifold_index.core.refined_index import format_multi_point_index
        from manifold_index.app.formatters import format_weyl_manifest_text

        q_var = "q"
        eta_vars = [f"η^(2·v_{a})" for a in range(num_hard)]

        if self._weyl_result is not None and self._weyl_result.ab_valid:
            series_str = format_weyl_manifest_text(
                results, num_hard, self._weyl_result.ab, eta_vars, q_var
            )
        else:
            series_str = format_multi_point_index(
                results, num_hard, q_var, eta_vars, show_zero=False
            )

        self._series_edit.setPlainText(series_str)
        self._series_group.show()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_copy(self) -> None:
        txt = self._series_edit.toPlainText()
        QApplication.clipboard().setText(txt)

    # ------------------------------------------------------------------
    # Accessors (for export page)
    # ------------------------------------------------------------------

    @property
    def refined_results(self):
        return self._refined_results

    @property
    def weyl_result(self):
        return self._weyl_result

    @property
    def nz_data(self):
        return self._nz_data
