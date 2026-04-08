"""
app/panels/filling_panel.py — Panel 2: Dehn Filling.

Per-cusp slope inputs → NC cycle search → filled refined index.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from manifold_index.app.katex import make_math_view, update_math_view
from manifold_index.app.formatters import (
    format_panel2_html,
    format_multi_cusp_fill_results,
)


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("sectionTitle")
    return lbl


class FillingPanel(QFrame):
    """Panel 2: Dehn filling.

    Signals
    -------
    fill_requested(object)
        Emitted when the user clicks "Dehn Fill".
        Payload is a dict with cusp configs, NC search range, etc.
    """

    fill_requested = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        self._math_view: QWidget | None = None
        self._nz_data = None
        self._weyl_result = None
        self._q_order_half = 20
        self._cusp_widgets: list[dict] = []
        self._nc_results = None
        self._transformed_results = None
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 12, 16, 12)
        outer.setSpacing(8)

        t = QLabel("② Dehn Filling")
        t.setObjectName("panelTitle")
        outer.addWidget(t)

        sub = QLabel("Set slopes → Dehn Fill searches NC cycles and computes filled index.")
        sub.setObjectName("panelSubtitle")
        sub.setWordWrap(True)
        outer.addWidget(sub)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        outer.addWidget(sep)

        # ── Dynamic cusp input area (rebuilt per manifold) ────
        self._cusp_container = QWidget()
        self._cusp_layout = QVBoxLayout(self._cusp_container)
        self._cusp_layout.setContentsMargins(0, 0, 0, 0)
        self._cusp_layout.setSpacing(6)
        outer.addWidget(self._cusp_container)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setFrameShadow(QFrame.Shadow.Sunken)
        outer.addWidget(sep2)

        # ── NC search range ───────────────────────────────────
        outer.addWidget(_section_label("NC Cycle Search Range"))

        range_row = QWidget()
        range_h = QHBoxLayout(range_row)
        range_h.setContentsMargins(0, 0, 0, 0)
        range_h.setSpacing(6)
        range_h.addWidget(QLabel("P ∈"))
        self._p_min = QSpinBox()
        self._p_min.setRange(-50, 0)
        self._p_min.setValue(-2)
        self._p_min.setFixedWidth(50)
        range_h.addWidget(self._p_min)
        range_h.addWidget(QLabel("to"))
        self._p_max = QSpinBox()
        self._p_max.setRange(0, 50)
        self._p_max.setValue(2)
        self._p_max.setFixedWidth(50)
        range_h.addWidget(self._p_max)
        range_h.addSpacing(8)
        range_h.addWidget(QLabel("Q ∈"))
        self._q_min = QSpinBox()
        self._q_min.setRange(-20, 0)
        self._q_min.setValue(0)
        self._q_min.setFixedWidth(50)
        range_h.addWidget(self._q_min)
        range_h.addWidget(QLabel("to"))
        self._q_max = QSpinBox()
        self._q_max.setRange(0, 20)
        self._q_max.setValue(2)
        self._q_max.setFixedWidth(50)
        range_h.addWidget(self._q_max)
        range_h.addStretch()
        outer.addWidget(range_row)

        # ── Single Dehn Fill button ───────────────────────────
        btn_row = QWidget()
        btn_h = QHBoxLayout(btn_row)
        btn_h.setContentsMargins(0, 0, 0, 0)
        btn_h.setSpacing(8)
        self._fill_btn = QPushButton("Dehn Fill  ▶")
        self._fill_btn.setObjectName("primary")
        self._fill_btn.setFixedHeight(36)
        self._fill_btn.setEnabled(False)
        btn_h.addStretch()
        btn_h.addWidget(self._fill_btn)
        outer.addWidget(btn_row)

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

        # ── Scrollable results area ───────────────────────────
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
        self._fill_btn.clicked.connect(self._on_fill_clicked)

    # ------------------------------------------------------------------
    # Public: called by MainWindow when Panel 1 data is ready
    # ------------------------------------------------------------------

    def reset(self, data: dict) -> None:
        """Rebuild cusp inputs based on new manifold data."""
        self._nz_data = data["nz_data"]
        self._weyl_result = data.get("weyl_result")
        self._q_order_half = data.get("q_order_half", 20)
        md = data.get("manifold_data")
        self._manifold_name: str = md.name if md is not None else "unknown"
        self._nc_results = None
        self._transformed_results = None

        r = self._nz_data.r

        # Clear old cusp widgets
        for w in self._cusp_widgets:
            for widget in w.values():
                if isinstance(widget, QWidget):
                    widget.setParent(None)
        self._cusp_widgets.clear()

        # Clear cusp layout
        while self._cusp_layout.count():
            item = self._cusp_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

        # Build cusp input rows
        for k in range(r):
            lbl = _section_label(f"Cusp {k}")
            self._cusp_layout.addWidget(lbl)

            row = QWidget()
            row_h = QHBoxLayout(row)
            row_h.setContentsMargins(0, 0, 0, 0)
            row_h.setSpacing(8)

            chk = QCheckBox(f"Fill cusp {k}")
            chk.setChecked(True)
            row_h.addWidget(chk)
            row_h.addSpacing(8)

            row_h.addWidget(QLabel("P:"))
            p_spin = QSpinBox()
            p_spin.setRange(-999, 999)
            p_spin.setValue(1)
            p_spin.setFixedWidth(60)
            row_h.addWidget(p_spin)

            row_h.addWidget(QLabel("Q:"))
            q_spin = QSpinBox()
            q_spin.setRange(-999, 999)
            q_spin.setValue(0)
            q_spin.setFixedWidth(60)
            row_h.addWidget(q_spin)

            row_h.addStretch()
            self._cusp_layout.addWidget(row)

            self._cusp_widgets.append({
                "label": lbl,
                "row": row,
                "checkbox": chk,
                "p_spin": p_spin,
                "q_spin": q_spin,
            })

        self._fill_btn.setEnabled(True)
        self._progress.hide()
        self._status.hide()

        # Clear results view — show compatibility summary from Weyl check
        self._set_math_content(format_panel2_html(weyl=self._weyl_result))

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    @Slot()
    def _on_fill_clicked(self) -> None:
        """Collect slopes + NC range and emit fill_requested."""
        cusp_configs = []
        for k, w in enumerate(self._cusp_widgets):
            cusp_configs.append({
                "cusp_idx": k,
                "fill": w["checkbox"].isChecked(),
                "P": w["p_spin"].value(),
                "Q": w["q_spin"].value(),
            })

        p_range = range(self._p_min.value(), self._p_max.value() + 1)
        q_range = range(self._q_min.value(), self._q_max.value() + 1)

        self.fill_requested.emit({
            "cusp_configs": cusp_configs,
            "p_range": p_range,
            "q_range": q_range,
            "nz_data": self._nz_data,
            "q_order_half": self._q_order_half,
            "weyl_result": self._weyl_result,
            "manifold_name": self._manifold_name,
        })

    # ------------------------------------------------------------------
    # Progress / status updates from worker
    # ------------------------------------------------------------------

    def set_loading(self) -> None:
        self._fill_btn.setEnabled(False)
        self._nc_results = None
        self._transformed_results = None
        self._progress.show()
        self._progress.setRange(0, 0)
        self._status.show()
        self._status.setText("Searching non-closable cycles…")
        self._status.setStyleSheet("font-size: 11px; color: palette(text);")

    def update_progress(self, done: int, total: int) -> None:
        self._progress.setRange(0, total)
        self._progress.setValue(done)

    def update_status(self, msg: str) -> None:
        self._status.setText(msg)

    def nc_search_done(self, nc_results: list) -> None:
        """Called when NC cycle search is complete (step 1)."""
        self._nc_results = nc_results
        html = format_panel2_html(
            nc_results=nc_results,
            nz=self._nz_data,
            weyl=self._weyl_result,
        )
        self._set_math_content(html)
        # Reset progress to indeterminate while step 2 loads kernels
        self._progress.setRange(0, 0)
        self._status.setText("Computing filled index…")

    def filling_finished(self, results: list) -> None:
        """Called when all filling computations are complete (step 2).

        *results* is either ``list[TransformedFillResult]`` (single-cusp
        filling), ``list[MultiCuspFillResult]`` (all cusps filled
        simultaneously), or ``list[UnrefinedFillResult]`` (fallback when
        no NC cycles are found).
        """
        from manifold_index.app.workers import (
            MultiCuspFillResult,
            UnrefinedFillResult,
        )

        self._transformed_results = results

        # Detect result type
        is_multi = (
            len(results) > 0
            and isinstance(results[0], MultiCuspFillResult)
        )
        is_unrefined = (
            len(results) > 0
            and isinstance(results[0], UnrefinedFillResult)
        )

        # Nmax = q_order_half / 2 → show that many q-terms
        nmax = self._q_order_half // 2

        if is_unrefined:
            html = format_panel2_html(
                nc_results=self._nc_results,
                unrefined_results=results,
                nz=self._nz_data,
                weyl=self._weyl_result,
                max_q_terms=nmax,
            )
            total_evals = sum(len(ur.fill_results) for ur in results)
            self._progress.setRange(0, 1)
            self._progress.setValue(1)
            self._status.setText(
                f"✓  0 NC cycle(s) · "
                f"{total_evals} unrefined filling evaluation(s)"
            )
        elif is_multi:
            html = format_panel2_html(
                nc_results=self._nc_results,
                multi_cusp_results=results,
                nz=self._nz_data,
                weyl=self._weyl_result,
                max_q_terms=nmax,
            )
            total_nc = sum(len(nc.cycles) for nc in (self._nc_results or []))
            self._progress.setRange(0, 1)
            self._progress.setValue(1)
            self._status.setText(
                f"✓  {total_nc} NC cycle(s) · "
                f"{len(results)} combination(s) filled sequentially"
            )
        else:
            html = format_panel2_html(
                nc_results=self._nc_results,
                transformed_results=results,
                nz=self._nz_data,
                weyl=self._weyl_result,
                max_q_terms=nmax,
            )
            total_nc = sum(len(nc.cycles) for nc in (self._nc_results or []))
            total_evals = sum(
                len(tr.fill_results) for tr in results
            )
            self._progress.setRange(0, 1)
            self._progress.setValue(1)
            self._status.setText(
                f"✓  {total_nc} NC cycle(s) · "
                f"{total_evals} filled index evaluation(s)"
            )

        self._set_math_content(html)
        self._status.setStyleSheet("color: #2ea043; font-size: 11px;")
        self._fill_btn.setEnabled(True)

    def set_error(self, msg: str) -> None:
        self._progress.hide()
        self._status.show()
        self._status.setText(f"✗  {msg}")
        self._status.setStyleSheet("color: #cf222e; font-size: 11px;")
        self._fill_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _set_math_content(self, html_body: str) -> None:
        if self._math_view is not None:
            update_math_view(self._math_view, html_body)
        else:
            self._math_view = make_math_view(html_body, min_h=200)
            self._content.addWidget(self._math_view, 1)
