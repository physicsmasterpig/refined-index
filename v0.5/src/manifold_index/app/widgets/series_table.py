"""app/widgets/series_table.py — q-series result table with KaTeX rendering.

Single QWebEngineView showing a table with per-row Copy / Remove actions
delivered via a QWebChannel bridge.  No QTableWidget — the KaTeX table is
the only display widget.

See BLUEPRINT §9.5 and §2.8.
"""

from __future__ import annotations

from PySide6.QtCore import QFile, QObject, Qt, QUrl, Signal, Slot
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from manifold_index.app.theme import colors as C
from manifold_index.app.widgets.math_view import build_katex_html, sys_colors

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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

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
    # Internal: re-initialise QWebChannel JS after every page reload
    # ------------------------------------------------------------------

    def _on_load_finished(self, ok: bool) -> None:
        if ok and self._view is not None and self._qwc_js:
            js = (
                self._qwc_js
                + ";new QWebChannel(qt.webChannelTransport,"
                  " function(ch){window.bridge=ch.objects.bridge;});"
            )
            self._view.page().runJavaScript(js)

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
        idx = len(self._rows)
        self._rows.append({
            "m": str(m),
            "e": str(e),
            "series_latex": series_latex,
            "source": source,
            "computing": False,
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
        if self._view is None:
            return
        if not self._rows:
            self._view.setVisible(False)
            return

        # Colour aliases for f-string readability
        ac = C.ACCENT
        am = C.ACCENT_MUTED
        tm = C.TEXT_MUTED
        ts = C.TEXT_SECONDARY
        bd = C.BORDER
        bs = C.BORDER_STRONG
        er = C.ERROR_BORDER

        # Inline CSS — all f-strings: {{ / }} → literal { / }
        css = (
            f"<style>"
            f".st{{border-collapse:collapse;width:100%}}"
            f".st th{{font-size:11px;font-weight:600;text-transform:uppercase;"
            f"color:{ts};border-bottom:1px solid {bs};"
            f"padding:3px 8px;white-space:nowrap}}"
            f".st td{{padding:4px 8px;border-bottom:1px solid {bd};"
            f"vertical-align:middle}}"
            f".st tr:hover td{{background:rgba(59,59,154,0.05)}}"
            f".ic{{color:{tm};font-size:11px;width:24px;text-align:right}}"
            f".nc{{text-align:center;width:40px}}"
            f".sc{{word-break:break-all}}"
            f".vc{{font-size:11px;color:{tm};white-space:nowrap}}"
            f".ac{{white-space:nowrap;width:64px;text-align:right}}"
            f"button.a{{background:transparent;border:none;cursor:pointer;"
            f"color:{ts};font-size:13px;padding:1px 4px;border-radius:2px}}"
            f"button.a:hover{{background:{am};color:{ac}}}"
            f"button.a.r:hover{{background:#FFEEEE;color:{er}}}"
            f"</style>"
        )

        rows_html: list[str] = []
        for i, row in enumerate(self._rows):
            if row["computing"]:
                sc = f"<span style='color:{tm}'>[computing…]</span>"
            elif not row["series_latex"]:
                sc = f"<span style='color:{tm}'>—</span>"
            else:
                # series_latex already carries $…$ delimiters from the formatter
                sc = row["series_latex"]

            rows_html.append(
                f"<tr>"
                f"<td class='ic'>{i}</td>"
                f"<td class='nc'><i>{row['m']}</i></td>"
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

        body = (
            css
            + "<table class='st'>"
            "<thead><tr>"
            "<th>#</th><th>$m$</th><th>$e$</th>"
            "<th>Series</th><th>Source</th><th></th>"
            "</tr></thead>"
            "<tbody>"
            + "".join(rows_html)
            + "</tbody></table>"
        )

        colors = sys_colors()
        full_html = build_katex_html(body, **colors)
        self._view.setHtml(full_html, QUrl("https://cdn.jsdelivr.net/"))
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
