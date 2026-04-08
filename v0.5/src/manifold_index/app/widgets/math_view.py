"""app/widgets/math_view.py — KaTeX WebEngine wrapper.

Port of v0.4 ``katex.py`` with additions required for v0.5:
  - ``MathView(QWidget)`` — class wrapping QWebEngineView (or text fallback)
  - ``scroll_to_bottom()`` — auto-scroll when new queries are appended
  - ``set_loading(bool)`` — show / hide a "Computing…" overlay

Module-level helpers are preserved verbatim for use by formatters:
  - ``build_katex_html(body, bg, fg, border, muted, accent) -> str``
  - ``sys_colors() -> dict``
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import (
    QApplication, QLabel, QSizePolicy, QStackedWidget,
    QTextEdit, QVBoxLayout, QWidget,
)

# ---------------------------------------------------------------------------
# WebEngine availability guard
# ---------------------------------------------------------------------------
_HAS_WEBENGINE = False
try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    _HAS_WEBENGINE = True
except ImportError:
    pass

KATEX_VERSION = "0.16.21"


# ---------------------------------------------------------------------------
# CSS fragment shared by all KaTeX pages  (verbatim from v0.4 katex.py)
# ---------------------------------------------------------------------------

_PAGE_CSS = """
:root {{ color-scheme: light dark; }}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 13px;
    line-height: 1.6;
    padding: 12px 14px;
    background: {bg};
    color: {fg};
}}
h3 {{
    font-size: 13px;
    font-weight: 600;
    margin: 14px 0 4px 0;
    padding-bottom: 3px;
    border-bottom: 1px solid {border};
    color: {fg};
}}
h3:first-child {{ margin-top: 0; }}
p {{ margin: 3px 0; }}
.muted {{ color: {muted}; font-size: 12px; }}
.success {{ color: #2ea043; }}
.warn {{ color: #d4880a; }}
.mono {{
    font-family: "SF Mono", Menlo, Monaco, Consolas, monospace;
    font-size: 11px;
}}
table {{
    font-family: "SF Mono", Menlo, Monaco, Consolas, monospace;
    font-size: 11px;
    border-collapse: collapse;
    margin: 4px 0;
    width: 100%;
}}
td, th {{
    padding: 2px 6px;
    text-align: left;
    white-space: nowrap;
}}
th {{
    font-weight: 600;
    border-bottom: 1px solid {border};
}}
tr:nth-child(even) {{ background: rgba(128,128,128,0.04); }}
table.idx {{
    border-collapse: collapse;
    margin: 6px 0;
    font-size: 13px;
    width: max-content;
}}
table.idx td {{
    padding: 3px 0;
    vertical-align: baseline;
    white-space: nowrap;
}}
table.idx td.i  {{ text-align: right; padding-right: 0; }}
table.idx td.al {{ text-align: right; }}
table.idx td.bl {{ text-align: left; }}
table.idx td.cp {{ text-align: left; padding-left: 0; }}
table.idx td.eq {{ text-align: center; padding: 3px 4px; }}
table.idx td.sr {{ text-align: left; padding-left: 4px; }}
table.nc {{
    border-collapse: collapse;
    margin: 4px 0;
    font-size: 12px;
    width: max-content;
}}
table.nc th {{
    font-weight: 600;
    border-bottom: 1px solid {border};
    padding: 2px 4px;
    text-align: center;
}}
table.nc td {{
    padding: 2px 8px;
    vertical-align: baseline;
    white-space: nowrap;
}}
table.nc td.r {{ text-align: right; padding-right: 0; }}
table.nc td.l {{ text-align: left;  padding-left: 2px; }}
table.nc td.sp {{ width: 10px; border-left: 1px solid {border}; }}
.katex-display {{ margin: 6px 0 !important; }}
.katex {{ font-size: 1.0em !important; }}
.sector {{
    margin: 4px 0;
    padding: 4px 0 4px 8px;
    border-bottom: 1px solid {border};
    text-align: left;
}}
.sector:last-child {{ border-bottom: none; }}
.sector-label {{ color: {muted}; font-size: 11px; margin-bottom: 1px; }}
hr {{ border: none; border-top: 1px solid {border}; margin: 10px 0; }}
"""


# ---------------------------------------------------------------------------
# Module-level helpers  (also used by formatters)
# ---------------------------------------------------------------------------

def build_katex_html(
    body: str,
    bg: str = "#ffffff",
    fg: str = "#000000",
    border: str = "#dddddd",
    muted: str = "#888888",
    accent: str = "#0969da",
) -> str:
    """Wrap *body* in a complete KaTeX-enabled HTML page."""
    css = _PAGE_CSS.format(bg=bg, fg=fg, border=border, muted=muted, accent=accent)
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/katex@{KATEX_VERSION}/dist/katex.min.css"
      crossorigin="anonymous">
<script defer
        src="https://cdn.jsdelivr.net/npm/katex@{KATEX_VERSION}/dist/katex.min.js"
        crossorigin="anonymous"></script>
<script defer
        src="https://cdn.jsdelivr.net/npm/katex@{KATEX_VERSION}/dist/contrib/auto-render.min.js"
        crossorigin="anonymous"></script>
<style>
{css}
</style>
</head>
<body>
{body}
<script>
document.addEventListener("DOMContentLoaded", function() {{
    renderMathInElement(document.body, {{
        delimiters: [
            {{left: "$$", right: "$$", display: true}},
            {{left: "$",  right: "$",  display: false}}
        ],
        throwOnError: false
    }});
}});
</script>
</body>
</html>"""


def sys_colors() -> dict[str, str]:
    """Read current Qt palette and return colour dict for build_katex_html."""
    pal = QApplication.instance().palette()
    return {
        "bg":     pal.color(pal.ColorRole.Base).name(),
        "fg":     pal.color(pal.ColorRole.Text).name(),
        "border": pal.color(pal.ColorRole.Mid).name(),
        "muted":  pal.color(pal.ColorRole.Mid).name(),
        "accent": pal.color(pal.ColorRole.Highlight).name(),
    }


# ---------------------------------------------------------------------------
# MathView widget
# ---------------------------------------------------------------------------

class MathView(QWidget):
    """QWidget wrapping a QWebEngineView (or QTextEdit fallback) for KaTeX.

    Usage::

        view = MathView(min_h=200)
        view.set_html("<p>$E = mc^2$</p>")
        # later, append more content:
        view.update_html("<p>$\\\\pi \\\\approx 3.14$</p>")
        # while computing:
        view.set_loading(True)
        # scroll after new rows:
        view.scroll_to_bottom()
    """

    def __init__(self, min_h: int = 100, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._min_h = min_h
        self._current_body: str = ""
        self._loading: bool = False

        # Stack: page 0 = math view, page 1 = loading overlay
        self._stack = QStackedWidget(self)

        # --- Math view (WebEngine or plain text fallback) ---
        if _HAS_WEBENGINE:
            self._view: QWidget = QWebEngineView()
            self._view.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        else:
            self._view = QTextEdit()
            self._view.setReadOnly(True)  # type: ignore[attr-defined]

        self._view.setMinimumHeight(min_h)
        self._view.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._stack.addWidget(self._view)   # index 0

        # --- Loading overlay ---
        self._loading_label = QLabel("Computing…")
        self._loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._loading_label.setStyleSheet(
            "color: #9A9A9A; font-size: 13px; background: #F9F9F8;"
        )
        self._loading_label.setMinimumHeight(min_h)
        self._stack.addWidget(self._loading_label)  # index 1

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._stack)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_html(self, body: str) -> None:
        """Replace the entire math view content with *body* (HTML fragment)."""
        self._current_body = body
        self._render(body)

    def update_html(self, body: str) -> None:
        """Replace content — synonym of set_html for v0.4 API compat."""
        self.set_html(body)

    def scroll_to_bottom(self) -> None:
        """Scroll the web view to the bottom of the page."""
        if _HAS_WEBENGINE and isinstance(self._view, QWebEngineView):
            self._view.page().runJavaScript(
                "window.scrollTo(0, document.body.scrollHeight);"
            )
        elif isinstance(self._view, QTextEdit):
            sb = self._view.verticalScrollBar()
            if sb is not None:
                sb.setValue(sb.maximum())

    def set_loading(self, loading: bool) -> None:
        """Show/hide the 'Computing…' overlay in place of the math view."""
        self._loading = loading
        self._stack.setCurrentIndex(1 if loading else 0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _render(self, body: str) -> None:
        colors = sys_colors()
        full_html = build_katex_html(body, **colors)
        if _HAS_WEBENGINE and isinstance(self._view, QWebEngineView):
            self._view.setHtml(full_html, QUrl("https://cdn.jsdelivr.net/"))
        elif isinstance(self._view, QTextEdit):
            # Fallback: strip HTML to plain text for readability
            self._view.setHtml(full_html)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication, QVBoxLayout, QPushButton, QWidget

    app = QApplication(sys.argv)
    win = QWidget()
    win.setWindowTitle("MathView smoke test")
    layout = QVBoxLayout(win)

    mv = MathView(min_h=300)
    mv.set_html(
        "<h3>Refined Index</h3>"
        "<p>$I^{\\mathrm{ref}}(0,0) = 1 - q^{1/2} + q - q^{3/2} + \\cdots$</p>"
        "<hr><p class='muted'>From kernel cache</p>"
    )
    layout.addWidget(mv)

    btn_loading = QPushButton("Toggle Loading")
    btn_loading.clicked.connect(lambda: mv.set_loading(not mv._loading))
    layout.addWidget(btn_loading)

    btn_scroll = QPushButton("Scroll to Bottom")
    btn_scroll.clicked.connect(mv.scroll_to_bottom)
    layout.addWidget(btn_scroll)

    win.resize(600, 450)
    win.show()
    sys.exit(app.exec())
