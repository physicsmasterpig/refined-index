"""
pages/overview_page.py — Page 2: Refined index overview for H₁(∂N, ℤ/2) sectors.

Displays:
  • Computation progress bar
  • Manifold & edge data (SnaPy edges, easy/hard classification)
  • H₁(∂N, ℤ/2) basis description
  • Weyl symmetry information
  • Refined index per sector (KaTeX-rendered)
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
from manifold_index.app.widgets.math_display import MathDisplay


class OverviewPage(QWidget):
    """Page 2: H₁(∂N, ℤ/2) sectors and refined index overview."""

    continue_requested = Signal()
    back_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._nz_data = None
        self._manifold_data = None
        self._easy_result = None
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

        # ── Scrollable area for all content ───────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll_widget = QWidget()
        self._content_layout = QVBoxLayout(scroll_widget)
        self._content_layout.setSpacing(12)
        self._content_layout.setContentsMargins(0, 0, 12, 0)
        scroll.setWidget(scroll_widget)
        root.addWidget(scroll, 1)

        # ── Manifold & Edge Data (rendered in MathDisplay) ────────
        self._manifold_group = QGroupBox("Manifold & Edge Data")
        manifold_vbox = QVBoxLayout(self._manifold_group)
        self._manifold_display = MathDisplay(
            font_size=13, mono_size=11, label_size=11, heading_size=13, min_height=120
        )
        manifold_vbox.addWidget(self._manifold_display, 1)
        self._manifold_group.hide()
        self._content_layout.addWidget(self._manifold_group)

        # ── H₁ basis info ─────────────────────────────────────────
        self._h1_group = QGroupBox("H₁(∂N, ℤ/2)")
        h1_vbox = QVBoxLayout(self._h1_group)
        self._h1_display = MathDisplay(
            font_size=13, mono_size=11, label_size=11, heading_size=13, min_height=80
        )
        h1_vbox.addWidget(self._h1_display, 1)
        self._h1_group.hide()
        self._content_layout.addWidget(self._h1_group)

        # ── Weyl info ─────────────────────────────────────────────
        self._weyl_group = QGroupBox("Weyl Symmetry")
        weyl_vbox = QVBoxLayout(self._weyl_group)
        self._weyl_display = MathDisplay(
            font_size=13, mono_size=11, label_size=11, heading_size=13, min_height=80
        )
        weyl_vbox.addWidget(self._weyl_display, 1)
        self._weyl_group.hide()
        self._content_layout.addWidget(self._weyl_group)

        # ── Refined index per sector (KaTeX-rendered) ─────────────
        self._series_group = QGroupBox("Refined Index per Sector")
        series_vbox = QVBoxLayout(self._series_group)

        copy_row = QHBoxLayout()
        copy_row.addStretch()

        copy_text_btn = QPushButton("⎘  Copy Plain Text")
        copy_text_btn.setObjectName("secondary")
        copy_text_btn.setFixedHeight(28)
        copy_text_btn.clicked.connect(self._on_copy_text)
        copy_row.addWidget(copy_text_btn)

        copy_latex_btn = QPushButton("⎘  Copy LaTeX")
        copy_latex_btn.setObjectName("secondary")
        copy_latex_btn.setFixedHeight(28)
        copy_latex_btn.clicked.connect(self._on_copy_latex)
        copy_row.addWidget(copy_latex_btn)

        series_vbox.addLayout(copy_row)

        self._series_display = MathDisplay(min_height=200)
        series_vbox.addWidget(self._series_display, 1)

        # Caches for clipboard copy
        self._series_text_cache = ""
        self._series_latex_cache = ""

        self._series_group.hide()
        self._content_layout.addWidget(self._series_group, 1)

        self._content_layout.addStretch()

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

    def reset(
        self,
        name: str,
        nz_data,
        q_order_half: int,
        manifold_data=None,
        easy_result=None,
    ) -> None:
        """Prepare page for a new computation."""
        self._nz_data = nz_data
        self._manifold_data = manifold_data
        self._easy_result = easy_result
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
        self._manifold_group.hide()
        self._h1_group.hide()
        self._weyl_group.hide()
        self._series_group.hide()
        self._series_display.clear()
        self._series_text_cache = ""
        self._series_latex_cache = ""
        self._continue_btn.setEnabled(False)

        # Show manifold info and H₁ basis right away
        self._populate_manifold_info(name, nz_data, manifold_data, easy_result)
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

    def _populate_manifold_info(self, name, nz_data, manifold_data, easy_result) -> None:
        """Show manifold & edge data using the MathDisplay widget."""
        if nz_data is None:
            self._manifold_group.hide()
            return

        from manifold_index.app.formatters import format_manifold_info_html
        html = format_manifold_info_html(name, nz_data, manifold_data, easy_result)
        self._manifold_display.set_content(html)
        self._manifold_group.show()

    def _populate_h1_basis(self, nz) -> None:
        """Show H₁(∂N, ℤ/2) = Span{α₁,β₁, …, αᵣ,βᵣ} using KaTeX."""
        r = nz.r
        if r == 0:
            self._h1_display.set_content("<p>Closed manifold — no boundary.</p>")
            self._h1_group.show()
            return

        basis_parts = []
        for i in range(r):
            basis_parts.append(rf"\alpha_{{{i}}}\;\text{{(meridian)}}")
            basis_parts.append(rf"\beta_{{{i}}}\;\text{{(longitude)}}")
        basis_str = ",\\;".join(basis_parts)

        html = (
            f'$$H_1(\\partial N,\\, \\mathbb{{Z}}/2) '
            f'= (\\mathbb{{Z}}/2)^{{{2*r}}}$$\n'
            f'<p>Basis: $\\{{ {basis_str} \\}}$</p>\n'
            f'<p>Evaluation grid per cusp: '
            f'$m_i \\in \\{{-2, -1, 0, 1, 2\\}}$, &nbsp; '
            f'$e_i \\in \\{{-1, -\\tfrac{{1}}{{2}}, 0, \\tfrac{{1}}{{2}}, 1\\}}$</p>\n'
            f'<p>Total evaluation points: $25^{{{r}}} = {25**r}$</p>'
        )
        self._h1_display.set_content(html)
        self._h1_group.show()

    def _populate_weyl_info(self) -> None:
        weyl = self._weyl_result
        if weyl is None or weyl.ab is None:
            self._weyl_display.set_content(
                '<p class="muted">Weyl parameters could not be determined.</p>'
            )
            self._weyl_group.show()
            return

        a_half = [Fraction(1, 2) * av for av in weyl.ab.a]
        b_vals = list(weyl.ab.b)

        a_parts = [rf"a_{{{i}}} = {_fmt_frac(v)}" for i, v in enumerate(a_half)]
        b_parts = [rf"b_{{{i}}} = {_fmt_frac(v)}" for i, v in enumerate(b_vals)]

        compat = all(v.denominator == 1 for v in a_half)
        if compat:
            compat_html = '<p class="success">✓ &nbsp; $a \\in \\mathbb{Z}$ — Dehn-filling compatible</p>'
        else:
            compat_html = '<p class="error">✗ &nbsp; $a \\notin \\mathbb{Z}$ — not directly compatible (slope Q must be even)</p>'

        html = (
            f'<p>Convention: $f(\\eta) = \\eta^{{b \\cdot m + a \\cdot e}} \\cdot I(m,e)$, '
            f'&nbsp; $f(\\eta) = f(\\eta^{{-1}})$</p>\n'
            f'<p>${",\\quad ".join(a_parts)}$</p>\n'
            f'<p>${",\\quad ".join(b_parts)}$</p>\n'
            f'{compat_html}'
        )
        self._weyl_display.set_content(html)
        self._weyl_group.show()

    def _populate_series(self, results: list, num_hard: int) -> None:
        """Format and display the refined index for each ℤ/2 sector using KaTeX."""
        from manifold_index.core.refined_index import format_multi_point_index
        from manifold_index.app.formatters import (
            format_weyl_manifest_text,
            format_series_katex_html,
        )

        q_var = "q"
        eta_vars = [f"η^(2·v_{a})" for a in range(num_hard)]

        # KaTeX HTML rendering
        katex_html = format_series_katex_html(results, num_hard, self._weyl_result)
        self._series_display.set_content(katex_html)

        # Cache plain text for copy button
        if self._weyl_result is not None and self._weyl_result.ab_valid:
            self._series_text_cache = format_weyl_manifest_text(
                results, num_hard, self._weyl_result.ab, eta_vars, q_var
            )
        else:
            self._series_text_cache = format_multi_point_index(
                results, num_hard, q_var, eta_vars, show_zero=False
            )

        # Cache LaTeX for copy button
        self._series_latex_cache = self._build_latex_cache(results, num_hard)

        self._series_group.show()

    def _build_latex_cache(self, results: list, num_hard: int) -> str:
        """Build LaTeX string for clipboard copy."""
        from manifold_index.app.formatters import series_to_latex, centre_to_latex, _fmt_charge
        from manifold_index.core.weyl_check import strip_weyl_monomial

        ab_valid = self._weyl_result is not None and self._weyl_result.ab_valid
        lines: list[str] = []
        lines.append(r"\begin{align*}")

        for m_ext, e_ext, result in results:
            if not result:
                continue
            charge = _fmt_charge(m_ext, e_ext)
            if ab_valid:
                centre, stripped = strip_weyl_monomial(
                    result, m_ext, e_ext, self._weyl_result.ab, num_hard
                )
                prefix = centre_to_latex(centre, num_hard)
                body = series_to_latex(stripped, num_hard)
                if prefix == "1":
                    lines.append(rf"I({charge}) &= {body} \\")
                else:
                    lines.append(rf"I({charge}) &= {prefix} \left( {body} \right) \\")
            else:
                body = series_to_latex(result, num_hard)
                lines.append(rf"I({charge}) &= {body} \\")

        lines.append(r"\end{align*}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_copy_text(self) -> None:
        QApplication.clipboard().setText(self._series_text_cache)

    def _on_copy_latex(self) -> None:
        QApplication.clipboard().setText(self._series_latex_cache)

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
