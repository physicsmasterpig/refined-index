"""app/widgets/series_table.py — q-series result table with KaTeX rendering.

Single QWebEngineView showing a table with per-row Copy / Remove actions
delivered via a QWebChannel bridge.  No QTableWidget — the KaTeX table is
the only display widget.

See BLUEPRINT §9.5 and §2.8.
"""

from __future__ import annotations

import json as _json

from PySide6.QtCore import QFile, QObject, Qt, QTimer, QUrl, Signal, Slot
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from manifold_index.app.theme import colors as C
from manifold_index.app.widgets.math_view import build_katex_html, sys_colors
from manifold_index.app.widgets.katex_assets import katex_base_url

# ---------------------------------------------------------------------------
# WebEngine / WebChannel availability guard
# ---------------------------------------------------------------------------
_HAS_WEBENGINE = False
try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWebChannel import QWebChannel
    _HAS_WEBENGINE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Bridge: JavaScript → Python communication
# ---------------------------------------------------------------------------

class _SeriesTableBridge(QObject):
    """Exposes ``copyRow`` / ``removeRow`` slots to page JavaScript."""

    copy_row   = Signal(int)
    remove_row = Signal(int)

    @Slot(int)
    def copyRow(self, row: int) -> None:      # noqa: N802
        self.copy_row.emit(row)

    @Slot(int)
    def removeRow(self, row: int) -> None:    # noqa: N802
        self.remove_row.emit(row)


class SeriesTable(QWidget):
    """Widget showing accumulated query results with KaTeX-rendered series.

    Rendering strategy
    ------------------
    KaTeX is loaded once via ``setHtml()`` when the first row is added.
    All subsequent updates inject new table content via ``runJavaScript()``
    and call ``renderMathInElement`` on the table element directly — there
    is no full page reload after the initial load, which eliminates both
    the render flicker and the LaTeX flash-of-unstyled-content during
    rapid computation updates.

    A 50 ms debounce timer coalesces bursts of calls (e.g. progress
    updates during a long computation) into a single DOM update.

    Signals
    -------
    row_removed(int)            — row index that was removed.
    copy_latex_requested(int)   — row index whose LaTeX was requested.
    """

    row_removed          = Signal(int)
    copy_latex_requested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # Internal data model: list of {m, e, series_latex, source, computing}
        self._rows: list[dict] = []

        # Page lifecycle flags
        self._page_ready = False   # True once loadFinished(ok=True) has fired
        self._loading    = False   # True between setHtml() and loadFinished

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Debounce timer — coalesces rapid _rebuild_html() calls into one update
        self._rebuild_timer = QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.setInterval(50)   # 50 ms
        self._rebuild_timer.timeout.connect(self._do_rebuild)

        if _HAS_WEBENGINE:
            self._view: QWebEngineView | None = QWebEngineView()
            self._view.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
            self._view.setMinimumHeight(80)
            self._view.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )

            # Bridge and channel
            self._bridge: _SeriesTableBridge | None = _SeriesTableBridge(self)
            self._bridge.copy_row.connect(self.copy_latex_requested)
            self._bridge.remove_row.connect(self._remove_row)

            self._channel: QWebChannel | None = QWebChannel(self)
            self._channel.registerObject("bridge", self._bridge)
            self._view.page().setWebChannel(self._channel)

            # Read Qt's bundled qwebchannel.js for post-load injection
            self._qwc_js: str = _load_qwebchannel_js()

            self._view.loadFinished.connect(self._on_load_finished)
            self._view.setVisible(False)
            layout.addWidget(self._view)
        else:
            from PySide6.QtWidgets import QLabel  # noqa: PLC0415
            self._view = None
            self._bridge = None
            self._channel = None
            self._qwc_js = ""
            layout.addWidget(QLabel("WebEngine not available"))

    # ------------------------------------------------------------------
    # Internal: page lifecycle
    # ------------------------------------------------------------------

    def _on_load_finished(self, ok: bool) -> None:
        """Called when setHtml() finishes loading the KaTeX page."""
        self._loading = False
        self._page_ready = ok
        if ok and self._view is not None and self._qwc_js:
            js = (
                self._qwc_js
                + ";new QWebChannel(qt.webChannelTransport,"
                  " function(ch){window.bridge=ch.objects.bridge;});"
            )
            self._view.page().runJavaScript(js)
        if ok:
            # Flush any data changes that arrived while the page was loading
            self._do_rebuild()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_row(
        self,
        m: int | str,
        e: int | str,
        series_latex: str,
        source: str,
    ) -> int:
        """Append a row and return its index."""
        m_str = str(m)
        idx = len(self._rows)
        self._rows.append({
            "m": m_str,
            "e": str(e),
            "series_latex": series_latex,
            "source": source,
            "computing": False,
            "n_label": m_str.count("<td"),
        })
        self._rebuild_html()
        return idx

    def set_row_computing(self, row: int) -> None:
        """Mark a row as currently computing (shows placeholder)."""
        if 0 <= row < len(self._rows):
            self._rows[row]["computing"] = True
            self._rows[row]["series_latex"] = ""
            self._rebuild_html()

    def set_row_result(self, row: int, series_latex: str, source: str) -> None:
        """Update a row with the computed result."""
        if 0 <= row < len(self._rows):
            self._rows[row]["computing"] = False
            self._rows[row]["series_latex"] = series_latex
            self._rows[row]["source"] = source
            self._rebuild_html()

    def update_row_metadata(self, row: int, m: str, e: str = "") -> None:
        """Update the m/e label columns of an existing row (e.g. after result is known)."""
        if 0 <= row < len(self._rows):
            m_str = str(m)
            self._rows[row]["m"] = m_str
            self._rows[row]["e"] = str(e)
            self._rows[row]["n_label"] = m_str.count("<td")
            self._rebuild_html()

    def clear_rows(self) -> None:
        """Remove all rows."""
        self._rows.clear()
        self._rebuild_html()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _remove_row(self, row: int) -> None:
        if 0 <= row < len(self._rows):
            self._rows.pop(row)
        self.row_removed.emit(row)
        self._rebuild_html()

    def _rebuild_html(self) -> None:
        """Schedule a debounced rebuild (coalesces rapid calls)."""
        if not self._rebuild_timer.isActive():
            self._rebuild_timer.start()

    def _build_table_parts(self) -> tuple[str, str, str]:
        """Return (css, thead_html, tbody_html) from current _rows data."""
        ac = C.ACCENT
        am = C.ACCENT_MUTED
        tm = C.TEXT_MUTED
        ts = C.TEXT_SECONDARY
        bd = C.BORDER
        bs = C.BORDER_STRONG
        er = C.ERROR_BORDER

        css = (
            f"<style>"
            f".st{{border-collapse:collapse;width:100%;font-size:11px}}"
            f".st th{{font-weight:600;text-transform:uppercase;"
            f"color:{ts};border-bottom:1px solid {bs};"
            f"padding:2px 4px;white-space:nowrap}}"
            f".st td{{border-bottom:1px solid {bd};vertical-align:baseline;white-space:nowrap}}"
            f".st tr:hover td{{background:rgba(59,59,154,0.05)}}"
            f".ic{{color:{tm};font-size:11px;width:1px;text-align:right;padding:3px 8px 3px 0}}"
            f".i{{width:1px;text-align:right;padding:3px 0}}"
            f".al{{width:1px;text-align:right;padding:3px 0}}"
            f".bl{{width:1px;text-align:left;padding:3px 0}}"
            f".sym{{width:1px;text-align:left;padding:3px 6px 3px 0}}"
            f".cp{{width:1px;text-align:left;padding:3px 4px 3px 0}}"
            f".eq{{width:1px;text-align:center;padding:3px 4px}}"
            f".nc{{text-align:left;padding:3px 4px}}"
            f".sr{{text-align:left;padding:3px 4px}}"
            f".vc{{font-size:11px;color:{tm};white-space:nowrap;padding:3px 4px}}"
            f".ac{{white-space:nowrap;width:60px;text-align:right;padding:3px 4px}}"
            f"button.a{{background:transparent;border:none;cursor:pointer;"
            f"color:{ts};font-size:13px;padding:1px 4px;border-radius:2px}}"
            f"button.a:hover{{background:{am};color:{ac}}}"
            f"button.a.r:hover{{background:#FFEEEE;color:{er}}}"
            f"</style>"
        )

        has_html_notation = any(row['m'].strip().startswith('<td') for row in self._rows)
        if has_html_notation:
            max_n_label = max(
                (row.get("n_label", 4) for row in self._rows
                 if row["m"].strip().startswith("<td")),
                default=4,
            )
            # +1 for the eq cell that lives in row['e']
            colspan = max_n_label + 1
            thead_html = f"<th>#</th><th colspan='{colspan}'></th><th>Series</th><th>Source</th><th></th>"
        else:
            max_n_label = 0
            thead_html = "<th>#</th><th>$m$</th><th>$e$</th><th>Series</th><th>Source</th><th></th>"

        rows_html: list[str] = []
        for i, row in enumerate(self._rows):
            if row["computing"]:
                sc = f"<span style='color:{tm}'>[computing…]</span>"
            elif not row["series_latex"]:
                sc = f"<span style='color:{tm}'>—</span>"
            else:
                sc = row["series_latex"]

            m_val = row['m']
            is_html_notation = m_val.strip().startswith('<td')

            if is_html_notation:
                eq_val = row['e']
                if eq_val.strip().startswith('<td'):
                    eq_html = eq_val
                else:
                    eq_html = f"<td class='eq'>{eq_val}</td>" if eq_val.strip() else ""

                # Pad shorter rows so every html row spans the same columns
                row_n = row.get("n_label", 4)
                pad = "<td></td>" * (max_n_label - row_n)

                row_content = (
                    f"<tr>"
                    f"<td class='ic'>{i}</td>"
                    f"{m_val}"
                    f"{pad}"
                    f"{eq_html}"
                    f"<td class='sr'>{sc}</td>"
                    f"<td class='vc'>{row['source']}</td>"
                    f"<td class='ac'>"
                    f"<button class='a'"
                    f" onclick=\"if(window.bridge)window.bridge.copyRow({i})\""
                    f" title='Copy LaTeX'>⧉</button>"
                    f"<button class='a r'"
                    f" onclick=\"if(window.bridge)window.bridge.removeRow({i})\""
                    f" title='Remove'>✕</button>"
                    f"</td>"
                    f"</tr>"
                )
            else:
                row_content = (
                    f"<tr>"
                    f"<td class='ic'>{i}</td>"
                    f"<td class='nc'><i>{m_val}</i></td>"
                    f"<td class='nc'><i>{row['e']}</i></td>"
                    f"<td class='sc'>{sc}</td>"
                    f"<td class='vc'>{row['source']}</td>"
                    f"<td class='ac'>"
                    f"<button class='a'"
                    f" onclick=\"if(window.bridge)window.bridge.copyRow({i})\""
                    f" title='Copy LaTeX'>⧉</button>"
                    f"<button class='a r'"
                    f" onclick=\"if(window.bridge)window.bridge.removeRow({i})\""
                    f" title='Remove'>✕</button>"
                    f"</td>"
                    f"</tr>"
                )
            rows_html.append(row_content)

        return css, thead_html, "".join(rows_html)

    def _do_rebuild(self) -> None:
        """Perform the actual HTML update — called by the debounce timer."""
        if self._view is None:
            return
        if not self._rows:
            self._view.setVisible(False)
            return

        css, thead_html, tbody_html = self._build_table_parts()

        if self._page_ready:
            # Fast path: update DOM in place via JS — no page reload, no KaTeX flash.
            # renderMathInElement re-renders only the table element.
            js = (
                "(function(){"
                "var h=document.getElementById('st-head');"
                "var b=document.getElementById('st-body');"
                "if(!h||!b)return;"
                f"h.innerHTML={_json.dumps(thead_html)};"
                f"b.innerHTML={_json.dumps(tbody_html)};"
                "var t=document.getElementById('st-tbl');"
                "if(t&&typeof renderMathInElement!=='undefined'){"
                "renderMathInElement(t,{"
                "delimiters:["
                "{left:'$$',right:'$$',display:true},"
                "{left:'$',right:'$',display:false}"
                "],"
                "throwOnError:false"
                "});}"
                "})();"
            )
            self._view.page().runJavaScript(js)

        elif not self._loading:
            # Initial load: build the full KaTeX page with the table skeleton.
            # After loadFinished, all further updates go through the JS path above.
            body = (
                css
                + "<table class='st' id='st-tbl'>"
                + "<thead><tr id='st-head'>" + thead_html + "</tr></thead>"
                + "<tbody id='st-body'>" + tbody_html + "</tbody>"
                + "</table>"
            )
            colors = sys_colors()
            full_html = build_katex_html(body, **colors)
            self._loading = True
            self._page_ready = False
            self._view.setHtml(full_html, katex_base_url())

        # else: page is currently loading — _on_load_finished will call _do_rebuild()

        self._view.setVisible(True)


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _load_qwebchannel_js() -> str:
    """Read Qt's bundled ``qwebchannel.js`` from Qt resources."""
    f = QFile(":/qtwebchannel/qwebchannel.js")
    if f.open(QFile.OpenModeFlag.ReadOnly):
        content = bytes(f.readAll()).decode("utf-8")
        f.close()
        return content
    return ""


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

    app = QApplication(sys.argv)
    from manifold_index.app.theme.style import build_stylesheet
    app.setStyleSheet(build_stylesheet())

    win = QWidget()
    win.setWindowTitle("SeriesTable smoke test")
    layout = QVBoxLayout(win)
    layout.setContentsMargins(16, 16, 16, 16)

    tbl = SeriesTable()
    tbl.add_row(0, 0, r"$1 - q^{1/2} + q - q^{3/2} + \cdots$", "computed")
    tbl.add_row(1, 0, r"$q^{1/2} - 2q + 3q^{3/2} - \cdots$", "computed")
    row_idx = tbl.add_row(0, r"\tfrac{1}{2}", "", "—")
    tbl.set_row_computing(row_idx)

    tbl.copy_latex_requested.connect(lambda r: print(f"Copy LaTeX row {r}"))
    tbl.row_removed.connect(lambda r: print(f"Removed row {r}"))

    layout.addWidget(tbl)
    win.resize(800, 400)
    win.show()
    sys.exit(app.exec())
