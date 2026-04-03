"""
app/katex.py — KaTeX HTML wrapper and QWebEngineView factory.

Provides:
  - build_katex_html(body, **colors) → full HTML page with KaTeX CDN
  - sys_colors() → dict of current palette colors
  - make_math_view(html_body, min_h) → QWidget (QWebEngineView or fallback)
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import QApplication, QTextEdit, QWidget

# ---------------------------------------------------------------------------
# WebEngine availability
# ---------------------------------------------------------------------------
_HAS_WEBENGINE = False
try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    _HAS_WEBENGINE = True
except ImportError:
    pass

KATEX_VERSION = "0.16.21"


# ---------------------------------------------------------------------------
# CSS fragment shared by all KaTeX pages
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
/* Index alignment table */
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
/* NC table — sub-column alignment */
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


def build_katex_html(
    body: str,
    bg: str = "#ffffff",
    fg: str = "#000000",
    border: str = "#dddddd",
    muted: str = "#888888",
    accent: str = "#0969da",
) -> str:
    """Wrap *body* content in a full KaTeX-enabled HTML page."""
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
            {{left: "$", right: "$", display: false}}
        ],
        throwOnError: false
    }});
}});
</script>
</body>
</html>"""


def sys_colors() -> dict[str, str]:
    """Get current system palette colors."""
    pal = QApplication.instance().palette()
    return {
        "bg": pal.color(pal.ColorRole.Base).name(),
        "fg": pal.color(pal.ColorRole.Text).name(),
        "border": pal.color(pal.ColorRole.Mid).name(),
        "muted": pal.color(pal.ColorRole.Mid).name(),
        "accent": pal.color(pal.ColorRole.Highlight).name(),
    }


def make_math_view(html_body: str, min_h: int = 100) -> QWidget:
    """Create a QWebEngineView showing KaTeX-rendered content.

    Falls back to a plain QTextEdit if WebEngine is unavailable.
    """
    if _HAS_WEBENGINE:
        view = QWebEngineView()
        view.setMinimumHeight(min_h)
        view.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        colors = sys_colors()
        full = build_katex_html(html_body, **colors)
        view.setHtml(full, QUrl("https://cdn.jsdelivr.net/"))
        return view
    else:
        te = QTextEdit()
        te.setReadOnly(True)
        te.setMinimumHeight(min_h)
        te.setPlainText(html_body)
        return te


def update_math_view(view: QWidget, html_body: str) -> None:
    """Update an existing math view with new HTML body content."""
    if _HAS_WEBENGINE and isinstance(view, QWebEngineView):
        colors = sys_colors()
        full = build_katex_html(html_body, **colors)
        view.setHtml(full, QUrl("https://cdn.jsdelivr.net/"))
    elif isinstance(view, QTextEdit):
        view.setPlainText(html_body)
