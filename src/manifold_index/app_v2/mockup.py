"""
app_v2/mockup.py — Visual mockup of the v0.2.0 three-panel GUI.

Run with:
    .venv/bin/python -m manifold_index.app_v2.mockup

All content is hardcoded placeholder data (m125) to show the layout.
"""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# Try to import QWebEngineView for KaTeX rendering
# ---------------------------------------------------------------------------
_HAS_WEBENGINE = False
try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    _HAS_WEBENGINE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KATEX_VERSION = "0.16.21"

# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------

STYLESHEET = """
QMainWindow {
    font-size: 13px;
}

/* Panel frames */
QFrame#panel {
    border: 1px solid palette(mid);
    border-radius: 6px;
    background: palette(base);
}

/* Panel headers */
QLabel#panelTitle {
    font-size: 16px;
    font-weight: bold;
    padding: 4px 0;
}
QLabel#panelSubtitle {
    font-size: 11px;
    color: palette(mid);
    padding-bottom: 4px;
}

/* Section headers */
QLabel#sectionTitle {
    font-size: 13px;
    font-weight: bold;
    color: palette(text);
    padding: 6px 0 2px 0;
    border-bottom: 1px solid palette(midlight);
    margin-bottom: 4px;
}

/* Input area */
QLineEdit {
    padding: 6px 10px;
    border: 1px solid palette(mid);
    border-radius: 4px;
    font-size: 13px;
}

QSpinBox {
    padding: 4px 8px;
    border: 1px solid palette(mid);
    border-radius: 3px;
}

/* Buttons */
QPushButton#primary {
    font-size: 13px;
    font-weight: bold;
    padding: 8px 20px;
    border-radius: 5px;
    background: palette(highlight);
    color: palette(highlighted-text);
    border: none;
}
QPushButton#primary:hover {
    background: palette(dark);
    color: palette(highlighted-text);
}
QPushButton#primary:disabled {
    background: palette(midlight);
    color: palette(mid);
}
QPushButton#secondary {
    padding: 6px 14px;
    border-radius: 4px;
    border: 1px solid palette(mid);
    background: palette(button);
    font-size: 12px;
}
QPushButton#secondary:hover {
    background: palette(midlight);
}

/* Progress */
QProgressBar {
    border: 1px solid palette(mid);
    border-radius: 3px;
    text-align: center;
    height: 14px;
}
QProgressBar::chunk {
    background: palette(highlight);
    border-radius: 2px;
}

/* Checkboxes */
QCheckBox {
    spacing: 6px;
    font-size: 12px;
}

/* Scroll area */
QScrollArea {
    border: none;
}
"""


# ---------------------------------------------------------------------------
# KaTeX HTML builder
# ---------------------------------------------------------------------------

def _build_katex_html(body: str, bg: str = "#ffffff", fg: str = "#000000",
                      border: str = "#dddddd", muted: str = "#888888",
                      accent: str = "#0969da") -> str:
    """Wrap body content in a full KaTeX-enabled HTML page."""
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
    padding: 2px 1px;
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


def _sys_colors() -> dict:
    """Get current system palette colors."""
    pal = QApplication.instance().palette()
    return {
        "bg": pal.color(pal.ColorRole.Base).name(),
        "fg": pal.color(pal.ColorRole.Text).name(),
        "border": pal.color(pal.ColorRole.Mid).name(),
        "muted": pal.color(pal.ColorRole.Mid).name(),
        "accent": pal.color(pal.ColorRole.Highlight).name(),
    }


# ---------------------------------------------------------------------------
# Helper: create a KaTeX-rendering QWebEngineView
# ---------------------------------------------------------------------------

def _make_math_view(html_body: str, min_h: int = 100) -> QWidget:
    """Create a QWebEngineView showing KaTeX-rendered content."""
    if _HAS_WEBENGINE:
        view = QWebEngineView()
        view.setMinimumHeight(min_h)
        view.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        colors = _sys_colors()
        full = _build_katex_html(html_body, **colors)
        view.setHtml(full, QUrl("https://cdn.jsdelivr.net/"))
        return view
    else:
        # Fallback: plain text
        te = QTextEdit()
        te.setReadOnly(True)
        te.setMinimumHeight(min_h)
        te.setPlainText(html_body)
        return te


# ---------------------------------------------------------------------------
# Hardcoded m125 placeholder data for the mockup
# ---------------------------------------------------------------------------

PANEL1_MANIFOLD_HTML = """
<h3>Manifold</h3>
<p><b>m125</b></p>
<p>Tetrahedra: <b>4</b> &nbsp;&bull;&nbsp; Cusps: <b>2</b></p>
<p>Internal edges: <b>2</b> &nbsp;(easy: <b>1</b>, hard: <b>1</b>)</p>

<h3>Gluing Equations (SnaPy)</h3>
<table>
<tr><th>Edge</th><th>$(Z_1, Z_1', Z_1'')$</th><th>$(Z_2, Z_2', Z_2'')$</th>
    <th>$(Z_3, Z_3', Z_3'')$</th><th>$(Z_4, Z_4', Z_4'')$</th></tr>
<tr><td><b>0</b></td><td>$(2, 0, 0)$</td><td>$(0, 0, 2)$</td>
    <td>$(0, 0, 0)$</td><td>$(0, 0, 0)$</td></tr>
<tr><td><b>1</b></td><td>$(0, 0, 2)$</td><td>$(0, 2, 0)$</td>
    <td>$(0, 2, 0)$</td><td>$(0, 2, 0)$</td></tr>
<tr><td><b>2</b></td><td>$(0, 2, 0)$</td><td>$(2, 0, 0)$</td>
    <td>$(2, 0, 0)$</td><td>$(2, 0, 0)$</td></tr>
<tr><td><b>3</b></td><td>$(0, 0, 0)$</td><td>$(0, 0, 0)$</td>
    <td>$(0, 0, 2)$</td><td>$(0, 0, 2)$</td></tr>
</table>

<h3>Edge Classification</h3>
<table>
<tr><th></th><th>Triplets</th><th>Origin</th><th>Role</th></tr>
<tr><td><b>E0</b> <span class="muted">(easy)</span></td>
    <td>$(0,0,1)\\;(0,1,0)\\;(0,1,0)\\;(0,1,0)$</td>
    <td>edge 1</td><td>basis row 1</td></tr>
<tr><td><b>H0</b> <span class="muted">(hard)</span></td>
    <td>$(0,1,0)\\;(0,1,1)\\;(1,0,0)\\;(0,1,1)$</td>
    <td>—</td><td>basis row 2</td></tr>
</table>

<h3>Neumann–Zagier Data</h3>
<p>$g_{\\text{NZ}} \\in \\mathrm{Sp}(2r,\\,\\mathbb{Q})$, &nbsp;
$r = 4$ tetrahedra &nbsp;→&nbsp; $8 \\times 8$ matrix</p>
$$g_{\\text{NZ}} = \\begin{pmatrix}
  -2   & -1   &  0   & -2   & -1   &  0   &  0   & -1  \\\\
   1   &  0   &  0   & -1   &  1   &  0   & -1   & -1  \\\\
  -1   & -1   &  1   & -1   & -1   &  0   &  0   &  0  \\\\
   0   & -1   & -1   & -1   &  1   & -1   & -1   & -1  \\\\
   0   & -\\tfrac{1}{2} & -\\tfrac{1}{2} & 0 & 0 & 0 & -\\tfrac{1}{2} & -\\tfrac{1}{2}  \\\\
   0   & -\\tfrac{1}{2} &  0   &  0   & \\tfrac{1}{2} &  0   &  0   & -\\tfrac{1}{2}  \\\\
  -1   & -\\tfrac{1}{2} &  2   & -1   & -\\tfrac{3}{2} &  0   &  1   & \\tfrac{1}{2}  \\\\
   0   &  1   &  0   &  0   &  0   &  0   &  0   &  0
\\end{pmatrix}$$
<p>Affine shifts: &nbsp;
$\\nu_x = (2,\\, 0,\\, 1,\\, 1)$, &nbsp;
$\\nu_p = (\\tfrac{1}{2},\\, 0,\\, 0,\\, 0)$</p>

<h3>Weyl Symmetry</h3>
<p>Convention: $f(\\eta) = \\eta^{b \\cdot m + a \\cdot e} \\cdot I(m,e)$, &nbsp;
$f(\\eta) = f(\\eta^{-1})$</p>
<p>$a_0 = -\\tfrac{2}{3}, \\quad b_0 = 0$</p>
<p class="warn">⚠ &nbsp; $a \\notin \\mathbb{Z}$ — Dehn filling <b>not</b> compatible</p>

<h3>Refined Index</h3>
<p class="muted">$N_{\\max} = 10$, &nbsp; $q$-order up to $q^5$. &nbsp;m00
Charges per cusp: $0,\\, \\pm\\tfrac{1}{2},\\, \\pm 1$ &nbsp;→&nbsp; $5^2 = 25$ sectors.
Label: $I(m_i, e_i) \\to I(-e_i\\,\\alpha_i + \\tfrac{m_i}{2}\\,\\beta_i)$.</p>

<table class="idx">
<tr>
  <td class="i">$I($</td>
  <td class="al">$0\\,\\alpha_1 +\\; 0\\,\\alpha_2$</td>
  <td class="bl">$+\\; 0\\,\\beta_1 +\\; 0\\,\\beta_2$</td>
  <td class="cp">$)$</td>
  <td class="eq">$=$</td>
  <td class="sr">$1 - 3q + (\\eta^{-2v_0} - 1 + \\eta^{2v_0})q^2 + \\cdots$</td>
</tr>
<tr>
  <td class="i">$I($</td>
  <td class="al">$0\\,\\alpha_1 +\\; 0\\,\\alpha_2$</td>
  <td class="bl">$+\\; 0\\,\\beta_1 +\\; \\tfrac{1}{2}\\,\\beta_2$</td>
  <td class="cp">$)$</td>
  <td class="eq">$=$</td>
  <td class="sr">$0$</td>
</tr>
<tr>
  <td class="i">$I($</td>
  <td class="al">$0\\,\\alpha_1 -\\; \\tfrac{1}{2}\\,\\alpha_2$</td>
  <td class="bl">$+\\; 0\\,\\beta_1 +\\; 0\\,\\beta_2$</td>
  <td class="cp">$)$</td>
  <td class="eq">$=$</td>
  <td class="sr">$0$</td>
</tr>
<tr>
  <td class="i">$I($</td>
  <td class="al">$0\\,\\alpha_1 +\\; 0\\,\\alpha_2$</td>
  <td class="bl">$+\\; 0\\,\\beta_1 +\\; 1\\,\\beta_2$</td>
  <td class="cp">$)$</td>
  <td class="eq">$=$</td>
  <td class="sr">$-q + (\\eta^{-2v_0}+1)q^2 + (2\\eta^{-2v_0}+6+\\eta^{2v_0})q^3 + \\cdots$</td>
</tr>
<tr>
  <td class="i">$I($</td>
  <td class="al">$0\\,\\alpha_1 -\\; 1\\,\\alpha_2$</td>
  <td class="bl">$+\\; 0\\,\\beta_1 +\\; 0\\,\\beta_2$</td>
  <td class="cp">$)$</td>
  <td class="eq">$=$</td>
  <td class="sr">$-q + (1+\\eta^{2v_0})q^2 + (\\eta^{-2v_0}+6+2\\eta^{2v_0})q^3 + \\cdots$</td>
</tr>
<tr>
  <td class="i">$I($</td>
  <td class="al">$0\\,\\alpha_1 +\\; 0\\,\\alpha_2$</td>
  <td class="bl">$+\\; \\tfrac{1}{2}\\,\\beta_1 +\\; 0\\,\\beta_2$</td>
  <td class="cp">$)$</td>
  <td class="eq">$=$</td>
  <td class="sr">$0$</td>
</tr>
<tr>
  <td class="i">$I($</td>
  <td class="al">$0\\,\\alpha_1 +\\; 0\\,\\alpha_2$</td>
  <td class="bl">$+\\; \\tfrac{1}{2}\\,\\beta_1 +\\; \\tfrac{1}{2}\\,\\beta_2$</td>
  <td class="cp">$)$</td>
  <td class="eq">$=$</td>
  <td class="sr">$0$</td>
</tr>
<tr>
  <td class="i">$I($</td>
  <td class="al">$0\\,\\alpha_1 -\\; \\tfrac{1}{2}\\,\\alpha_2$</td>
  <td class="bl">$+\\; \\tfrac{1}{2}\\,\\beta_1 +\\; 0\\,\\beta_2$</td>
  <td class="cp">$)$</td>
  <td class="eq">$=$</td>
  <td class="sr">$0$</td>
</tr>
<tr>
  <td class="i">$I($</td>
  <td class="al">$0\\,\\alpha_1 +\\; 0\\,\\alpha_2$</td>
  <td class="bl">$+\\; \\tfrac{1}{2}\\,\\beta_1 +\\; 1\\,\\beta_2$</td>
  <td class="cp">$)$</td>
  <td class="eq">$=$</td>
  <td class="sr">$0$</td>
</tr>
<tr>
  <td class="i">$I($</td>
  <td class="al">$0\\,\\alpha_1 -\\; 1\\,\\alpha_2$</td>
  <td class="bl">$+\\; \\tfrac{1}{2}\\,\\beta_1 +\\; 0\\,\\beta_2$</td>
  <td class="cp">$)$</td>
  <td class="eq">$=$</td>
  <td class="sr">$0$</td>
</tr>
<tr>
  <td class="i">$I($</td>
  <td class="al">$-\\tfrac{1}{2}\\,\\alpha_1 +\\; 0\\,\\alpha_2$</td>
  <td class="bl">$+\\; 0\\,\\beta_1 +\\; 0\\,\\beta_2$</td>
  <td class="cp">$)$</td>
  <td class="eq">$=$</td>
  <td class="sr">$0$</td>
</tr>
<tr>
  <td class="i">$I($</td>
  <td class="al">$-\\tfrac{1}{2}\\,\\alpha_1 +\\; 0\\,\\alpha_2$</td>
  <td class="bl">$+\\; 0\\,\\beta_1 +\\; \\tfrac{1}{2}\\,\\beta_2$</td>
  <td class="cp">$)$</td>
  <td class="eq">$=$</td>
  <td class="sr">$-(\\eta^{-2v_0}+1)q + (\\eta^{-2v_0}+1)q^2 + \\cdots$</td>
</tr>
<tr>
  <td class="i">$I($</td>
  <td class="al">$-\\tfrac{1}{2}\\,\\alpha_1 -\\; \\tfrac{1}{2}\\,\\alpha_2$</td>
  <td class="bl">$+\\; 0\\,\\beta_1 +\\; 0\\,\\beta_2$</td>
  <td class="cp">$)$</td>
  <td class="eq">$=$</td>
  <td class="sr">$0$</td>
</tr>
<tr>
  <td class="i">$I($</td>
  <td class="al">$-\\tfrac{1}{2}\\,\\alpha_1 +\\; 0\\,\\alpha_2$</td>
  <td class="bl">$+\\; 0\\,\\beta_1 +\\; 1\\,\\beta_2$</td>
  <td class="cp">$)$</td>
  <td class="eq">$=$</td>
  <td class="sr">$0$</td>
</tr>
<tr>
  <td class="i">$I($</td>
  <td class="al">$-\\tfrac{1}{2}\\,\\alpha_1 -\\; 1\\,\\alpha_2$</td>
  <td class="bl">$+\\; 0\\,\\beta_1 +\\; 0\\,\\beta_2$</td>
  <td class="cp">$)$</td>
  <td class="eq">$=$</td>
  <td class="sr">$0$</td>
</tr>
<tr>
  <td class="i">$I($</td>
  <td class="al">$0\\,\\alpha_1 +\\; 0\\,\\alpha_2$</td>
  <td class="bl">$+\\; 1\\,\\beta_1 +\\; 0\\,\\beta_2$</td>
  <td class="cp">$)$</td>
  <td class="eq">$=$</td>
  <td class="sr">$-q + (\\eta^{-2v_0}+1)q^2 + (2\\eta^{-2v_0}+6+\\eta^{2v_0})q^3 + \\cdots$</td>
</tr>
<tr>
  <td class="i">$I($</td>
  <td class="al">$0\\,\\alpha_1 +\\; 0\\,\\alpha_2$</td>
  <td class="bl">$+\\; 1\\,\\beta_1 +\\; \\tfrac{1}{2}\\,\\beta_2$</td>
  <td class="cp">$)$</td>
  <td class="eq">$=$</td>
  <td class="sr">$0$</td>
</tr>
<tr>
  <td class="i">$I($</td>
  <td class="al">$0\\,\\alpha_1 -\\; \\tfrac{1}{2}\\,\\alpha_2$</td>
  <td class="bl">$+\\; 1\\,\\beta_1 +\\; 0\\,\\beta_2$</td>
  <td class="cp">$)$</td>
  <td class="eq">$=$</td>
  <td class="sr">$0$</td>
</tr>
<tr>
  <td class="i">$I($</td>
  <td class="al">$0\\,\\alpha_1 +\\; 0\\,\\alpha_2$</td>
  <td class="bl">$+\\; 1\\,\\beta_1 +\\; 1\\,\\beta_2$</td>
  <td class="cp">$)$</td>
  <td class="eq">$=$</td>
  <td class="sr">$(\\eta^{-2v_0}+1)q^2 + (\\eta^{-2v_0}+3)q^3 + \\cdots$</td>
</tr>
<tr>
  <td class="i">$I($</td>
  <td class="al">$0\\,\\alpha_1 -\\; 1\\,\\alpha_2$</td>
  <td class="bl">$+\\; 1\\,\\beta_1 +\\; 0\\,\\beta_2$</td>
  <td class="cp">$)$</td>
  <td class="eq">$=$</td>
  <td class="sr">$q^2 + (\\eta^{-2v_0}+3+\\eta^{2v_0})q^3 + \\cdots$</td>
</tr>
<tr>
  <td class="i">$I($</td>
  <td class="al">$-1\\,\\alpha_1 +\\; 0\\,\\alpha_2$</td>
  <td class="bl">$+\\; 0\\,\\beta_1 +\\; 0\\,\\beta_2$</td>
  <td class="cp">$)$</td>
  <td class="eq">$=$</td>
  <td class="sr">$-\\eta^{-2v_0}q + (\\eta^{-2v_0}+1)q^2 + (\\eta^{-4v_0}+6\\eta^{-2v_0}+2)q^3 + \\cdots$</td>
</tr>
<tr>
  <td class="i">$I($</td>
  <td class="al">$-1\\,\\alpha_1 +\\; 0\\,\\alpha_2$</td>
  <td class="bl">$+\\; 0\\,\\beta_1 +\\; \\tfrac{1}{2}\\,\\beta_2$</td>
  <td class="cp">$)$</td>
  <td class="eq">$=$</td>
  <td class="sr">$0$</td>
</tr>
<tr>
  <td class="i">$I($</td>
  <td class="al">$-1\\,\\alpha_1 -\\; \\tfrac{1}{2}\\,\\alpha_2$</td>
  <td class="bl">$+\\; 0\\,\\beta_1 +\\; 0\\,\\beta_2$</td>
  <td class="cp">$)$</td>
  <td class="eq">$=$</td>
  <td class="sr">$0$</td>
</tr>
<tr>
  <td class="i">$I($</td>
  <td class="al">$-1\\,\\alpha_1 +\\; 0\\,\\alpha_2$</td>
  <td class="bl">$+\\; 0\\,\\beta_1 +\\; 1\\,\\beta_2$</td>
  <td class="cp">$)$</td>
  <td class="eq">$=$</td>
  <td class="sr">$\\eta^{-2v_0}q^2 + (\\eta^{-4v_0}+3\\eta^{-2v_0}+1)q^3 + \\cdots$</td>
</tr>
<tr>
  <td class="i">$I($</td>
  <td class="al">$-1\\,\\alpha_1 -\\; 1\\,\\alpha_2$</td>
  <td class="bl">$+\\; 0\\,\\beta_1 +\\; 0\\,\\beta_2$</td>
  <td class="cp">$)$</td>
  <td class="eq">$=$</td>
  <td class="sr">$(\\eta^{-2v_0}+1)q^2 + (3\\eta^{-2v_0}+1)q^3 + \\cdots$</td>
</tr>
</table>
"""


PANEL2_FILLING_HTML_BEFORE = """
<h3>Dehn Filling Setup</h3>
<p class="muted">Select cusps and input slopes below, then click <b>Dehn Fill</b>.</p>
"""

PANEL2_FILLING_HTML_AFTER = """
<h3>Non-closable Cycles</h3>
<p>Found <b>3</b> non-closable cycles at cusp 0:</p>
<table class="nc">
<tr>
  <th colspan="2">$\\gamma_0$</th>
  <th class="sp"></th>
  <th colspan="2">$\\delta_0$</th>
  <th class="sp"></th>
  <th colspan="2">Dehn filling slope</th>
</tr>
<tr>
  <td class="r">$1\\,\\alpha_0$</td><td class="l">$+\\; 0\\,\\beta_0$</td>
  <td class="sp"></td>
  <td class="r">$0\\,\\alpha_0$</td><td class="l">$+\\; 1\\,\\beta_0$</td>
  <td class="sp"></td>
  <td class="r">$3\\,\\gamma_0$</td><td class="l">$+\\; 1\\,\\delta_0$</td>
</tr>
<tr>
  <td class="r">$-1\\,\\alpha_0$</td><td class="l">$+\\; 0\\,\\beta_0$</td>
  <td class="sp"></td>
  <td class="r">$0\\,\\alpha_0$</td><td class="l">$-\\; 1\\,\\beta_0$</td>
  <td class="sp"></td>
  <td class="r">$-3\\,\\gamma_0$</td><td class="l">$-\\; 1\\,\\delta_0$</td>
</tr>
<tr>
  <td class="r">$1\\,\\alpha_0$</td><td class="l">$+\\; 1\\,\\beta_0$</td>
  <td class="sp"></td>
  <td class="r">$-1\\,\\alpha_0$</td><td class="l">$+\\; 1\\,\\beta_0$</td>
  <td class="sp"></td>
  <td class="r">$1\\,\\gamma_0$</td><td class="l">$-\\; 2\\,\\delta_0$</td>
</tr>
</table>

<p>Found <b>2</b> non-closable cycles at cusp 1:</p>
<table class="nc">
<tr>
  <th colspan="2">$\\gamma_1$</th>
  <th class="sp"></th>
  <th colspan="2">$\\delta_1$</th>
  <th class="sp"></th>
  <th colspan="2">Dehn filling slope</th>
</tr>
<tr>
  <td class="r">$1\\,\\alpha_1$</td><td class="l">$+\\; 0\\,\\beta_1$</td>
  <td class="sp"></td>
  <td class="r">$0\\,\\alpha_1$</td><td class="l">$+\\; 1\\,\\beta_1$</td>
  <td class="sp"></td>
  <td class="r">$2\\,\\gamma_1$</td><td class="l">$+\\; 1\\,\\delta_1$</td>
</tr>
<tr>
  <td class="r">$-1\\,\\alpha_1$</td><td class="l">$+\\; 0\\,\\beta_1$</td>
  <td class="sp"></td>
  <td class="r">$0\\,\\alpha_1$</td><td class="l">$-\\; 1\\,\\beta_1$</td>
  <td class="sp"></td>
  <td class="r">$-2\\,\\gamma_1$</td><td class="l">$-\\; 1\\,\\delta_1$</td>
</tr>
</table>

<h3>Filled Refined Index</h3>
<p class="warn">⚠ &nbsp; $a \\notin \\mathbb{Z}$: Dehn filling results may be unreliable.</p>
<table class="idx">
<tr>
  <td class="i" colspan="4" style="text-align:right; padding-right:6px;">
    <span class="muted">cusp 0, $\\gamma_0 = \\alpha_0$:</span> &nbsp;
    $I^{\\mathrm{ref}}_{3\\gamma_0+\\delta_0}$</td>
  <td class="eq">$=$</td>
  <td class="sr">$1 + q + q^{3/2} + 2q^2 + \\cdots$</td>
</tr>
<tr>
  <td class="i" colspan="4" style="text-align:right; padding-right:6px;">
    <span class="muted">cusp 0, $\\gamma_0 = \\alpha_0+\\beta_0$:</span> &nbsp;
    $I^{\\mathrm{ref}}_{\\gamma_0-2\\delta_0}$</td>
  <td class="eq">$=$</td>
  <td class="sr">$1 + q + q^{3/2} + 2q^2 + \\cdots$</td>
</tr>
<tr>
  <td class="i" colspan="4" style="text-align:right; padding-right:6px;">
    <span class="muted">cusp 1, $\\gamma_1 = \\alpha_1$:</span> &nbsp;
    $I^{\\mathrm{ref}}_{2\\gamma_1+\\delta_1}$</td>
  <td class="eq">$=$</td>
  <td class="sr">$1 + q + q^{3/2} + 2q^2 + \\cdots$</td>
</tr>
</table>

<p class="success">✓ &nbsp; Results agree across non-closable cycles (both cusps).</p>
"""


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------

def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("sectionTitle")
    return lbl


def _build_panel_frame(title: str, subtitle: str = "") -> tuple[QFrame, QVBoxLayout]:
    """Build a styled panel frame with a title header. Returns (frame, content_layout)."""
    frame = QFrame()
    frame.setObjectName("panel")
    outer = QVBoxLayout(frame)
    outer.setContentsMargins(16, 12, 16, 12)
    outer.setSpacing(8)

    # Title
    t = QLabel(title)
    t.setObjectName("panelTitle")
    outer.addWidget(t)

    if subtitle:
        s = QLabel(subtitle)
        s.setObjectName("panelSubtitle")
        s.setWordWrap(True)
        outer.addWidget(s)

    # Separator
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.HLine)
    sep.setFrameShadow(QFrame.Shadow.Sunken)
    outer.addWidget(sep)

    # Scrollable content
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll_widget = QWidget()
    content = QVBoxLayout(scroll_widget)
    content.setContentsMargins(0, 4, 0, 4)
    content.setSpacing(6)
    scroll.setWidget(scroll_widget)
    outer.addWidget(scroll, 1)

    return frame, content


# ---------------------------------------------------------------------------
# Panel 1: Manifold Analysis
# ---------------------------------------------------------------------------

def build_panel_manifold() -> QFrame:
    frame, layout = _build_panel_frame(
        "① Manifold Analysis",
        "Input a manifold name to load SnaPy data and compute refined index."
    )

    # ── Input row ─────────────────────────────────────────────
    input_box = QWidget()
    input_row = QHBoxLayout(input_box)
    input_row.setContentsMargins(0, 0, 0, 0)
    input_row.setSpacing(8)

    name_edit = QLineEdit()
    name_edit.setPlaceholderText("e.g.  m003  m004  4_1  L5a1  …")
    name_edit.setText("m125")
    name_edit.setFixedHeight(32)
    input_row.addWidget(name_edit, 1)

    input_row.addWidget(QLabel("Nmax:"))
    nmax_spin = QSpinBox()
    nmax_spin.setRange(4, 60)
    nmax_spin.setValue(10)
    nmax_spin.setFixedWidth(60)
    input_row.addWidget(nmax_spin)

    compute_btn = QPushButton("Compute")
    compute_btn.setObjectName("primary")
    compute_btn.setFixedHeight(32)
    input_row.addWidget(compute_btn)

    layout.addWidget(input_box)

    # ── Progress ──────────────────────────────────────────────
    progress = QProgressBar()
    progress.setRange(0, 100)
    progress.setValue(100)
    progress.setFixedHeight(12)
    layout.addWidget(progress)

    status = QLabel("✓  1129 sectors computed  ·  m125  ·  4 tet, 2 cusps, 1 easy, 1 hard")
    status.setStyleSheet("color: #2ea043; font-size: 11px;")
    layout.addWidget(status)

    # ── Main data display (KaTeX) ─────────────────────────────
    math_view = _make_math_view(PANEL1_MANIFOLD_HTML, min_h=400)
    layout.addWidget(math_view, 1)

    return frame


# ---------------------------------------------------------------------------
# Panel 2: Dehn Filling
# ---------------------------------------------------------------------------

def build_panel_filling() -> QFrame:
    frame, layout = _build_panel_frame(
        "② Dehn Filling",
        "Select cusps to fill, input slopes, search non-closable cycles."
    )

    # ── Cusp selection ────────────────────────────────────────
    cusp_box = QWidget()
    cusp_layout = QVBoxLayout(cusp_box)
    cusp_layout.setContentsMargins(0, 0, 0, 0)
    cusp_layout.setSpacing(6)

    # --- Cusp 0 ---
    lbl0 = _section_label("Cusp 0")
    cusp_layout.addWidget(lbl0)

    row0 = QWidget()
    row0_h = QHBoxLayout(row0)
    row0_h.setContentsMargins(0, 0, 0, 0)
    row0_h.setSpacing(8)

    chk0 = QCheckBox("Fill cusp 0")
    chk0.setChecked(True)
    row0_h.addWidget(chk0)
    row0_h.addSpacing(8)

    row0_h.addWidget(QLabel("P:"))
    p0_spin = QSpinBox()
    p0_spin.setRange(-999, 999)
    p0_spin.setValue(3)
    p0_spin.setFixedWidth(60)
    row0_h.addWidget(p0_spin)

    row0_h.addWidget(QLabel("Q:"))
    q0_spin = QSpinBox()
    q0_spin.setRange(-999, 999)
    q0_spin.setValue(1)
    q0_spin.setFixedWidth(60)
    row0_h.addWidget(q0_spin)

    row0_h.addStretch()
    cusp_layout.addWidget(row0)

    # --- Cusp 1 ---
    lbl1 = _section_label("Cusp 1")
    cusp_layout.addWidget(lbl1)

    row1 = QWidget()
    row1_h = QHBoxLayout(row1)
    row1_h.setContentsMargins(0, 0, 0, 0)
    row1_h.setSpacing(8)

    chk1 = QCheckBox("Fill cusp 1")
    chk1.setChecked(True)
    row1_h.addWidget(chk1)
    row1_h.addSpacing(8)

    row1_h.addWidget(QLabel("P:"))
    p1_spin = QSpinBox()
    p1_spin.setRange(-999, 999)
    p1_spin.setValue(2)
    p1_spin.setFixedWidth(60)
    row1_h.addWidget(p1_spin)

    row1_h.addWidget(QLabel("Q:"))
    q1_spin = QSpinBox()
    q1_spin.setRange(-999, 999)
    q1_spin.setValue(1)
    q1_spin.setFixedWidth(60)
    row1_h.addWidget(q1_spin)

    row1_h.addStretch()
    cusp_layout.addWidget(row1)

    # NC search range
    range_lbl = QLabel("NC search:  P ∈ [−2, 2],  Q ∈ [−2, 2]")
    range_lbl.setStyleSheet("font-size: 11px; color: palette(mid);")
    cusp_layout.addWidget(range_lbl)

    layout.addWidget(cusp_box)

    # ── Dehn fill button ──────────────────────────────────────
    btn_row = QWidget()
    btn_h = QHBoxLayout(btn_row)
    btn_h.setContentsMargins(0, 0, 0, 0)
    fill_btn = QPushButton("Dehn Fill  ▶")
    fill_btn.setObjectName("primary")
    fill_btn.setFixedHeight(36)
    btn_h.addStretch()
    btn_h.addWidget(fill_btn)
    layout.addWidget(btn_row)

    # ── Progress ──────────────────────────────────────────────
    progress = QProgressBar()
    progress.setRange(0, 5)
    progress.setValue(5)
    progress.setFixedHeight(12)
    layout.addWidget(progress)

    status = QLabel("✓  5 non-closable cycles found · filling complete")
    status.setStyleSheet("color: #2ea043; font-size: 11px;")
    layout.addWidget(status)

    # ── Results (KaTeX) ───────────────────────────────────────
    results_view = _make_math_view(
        PANEL2_FILLING_HTML_BEFORE + PANEL2_FILLING_HTML_AFTER,
        min_h=300
    )
    layout.addWidget(results_view, 1)

    return frame


# ---------------------------------------------------------------------------
# Panel 3: Export
# ---------------------------------------------------------------------------

def build_panel_export() -> QFrame:
    frame, layout = _build_panel_frame(
        "③ Export",
        "Save results to various formats."
    )

    # ── Format section ────────────────────────────────────────
    layout.addWidget(_section_label("Formats"))

    chk_tex = QCheckBox("LaTeX  (.tex)")
    chk_tex.setChecked(True)
    layout.addWidget(chk_tex)

    chk_report = QCheckBox("Full Report  (.tex)")
    chk_report.setChecked(True)
    layout.addWidget(chk_report)

    chk_nb = QCheckBox("Mathematica  (.nb)")
    chk_nb.setChecked(True)
    layout.addWidget(chk_nb)

    chk_txt = QCheckBox("Plain text  (.txt)")
    chk_txt.setChecked(False)
    layout.addWidget(chk_txt)

    chk_json = QCheckBox("JSON  (.json)")
    chk_json.setChecked(False)
    layout.addWidget(chk_json)

    layout.addSpacing(8)

    # ── Options ───────────────────────────────────────────────
    layout.addWidget(_section_label("Options"))

    chk_weyl = QCheckBox("Weyl-manifest form")
    chk_weyl.setChecked(True)
    layout.addWidget(chk_weyl)

    chk_dehn = QCheckBox("Include Dehn filling results")
    chk_dehn.setChecked(True)
    layout.addWidget(chk_dehn)

    layout.addSpacing(8)

    # ── Output path ───────────────────────────────────────────
    layout.addWidget(_section_label("Output"))

    dir_row = QWidget()
    dir_h = QHBoxLayout(dir_row)
    dir_h.setContentsMargins(0, 0, 0, 0)
    dir_h.setSpacing(4)
    dir_edit = QLineEdit()
    dir_edit.setText("~/results/")
    dir_edit.setFixedHeight(28)
    dir_h.addWidget(dir_edit, 1)
    browse = QPushButton("…")
    browse.setObjectName("secondary")
    browse.setFixedWidth(28)
    browse.setFixedHeight(28)
    dir_h.addWidget(browse)
    layout.addWidget(dir_row)

    prefix_row = QWidget()
    prefix_h = QHBoxLayout(prefix_row)
    prefix_h.setContentsMargins(0, 0, 0, 0)
    prefix_h.setSpacing(4)
    prefix_h.addWidget(QLabel("Prefix:"))
    prefix_edit = QLineEdit()
    prefix_edit.setText("m125_index")
    prefix_edit.setFixedHeight(28)
    prefix_h.addWidget(prefix_edit, 1)
    layout.addWidget(prefix_row)

    layout.addSpacing(12)

    # ── Export buttons ────────────────────────────────────────
    export_btn = QPushButton("Export All  ▶")
    export_btn.setObjectName("primary")
    export_btn.setFixedHeight(36)
    layout.addWidget(export_btn)

    layout.addSpacing(4)

    copy_btn = QPushButton("⎘  Copy LaTeX to Clipboard")
    copy_btn.setObjectName("secondary")
    copy_btn.setFixedHeight(30)
    layout.addWidget(copy_btn)

    copy_txt_btn = QPushButton("⎘  Copy Plain Text")
    copy_txt_btn.setObjectName("secondary")
    copy_txt_btn.setFixedHeight(30)
    layout.addWidget(copy_txt_btn)

    layout.addSpacing(8)

    # ── Status ────────────────────────────────────────────────
    status = QLabel("✓  Exported 3 files to ~/results/")
    status.setStyleSheet("color: #2ea043; font-size: 11px;")
    status.setWordWrap(True)
    layout.addWidget(status)

    layout.addStretch()

    return frame


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

class MockupWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Refined 3D Index Calculator — v0.2.0 Mockup")
        self.setMinimumSize(1200, 700)
        self.resize(1500, 850)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(0)

        # Use QSplitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(6)

        panel1 = build_panel_manifold()
        panel2 = build_panel_filling()
        panel3 = build_panel_export()

        splitter.addWidget(panel1)
        splitter.addWidget(panel2)
        splitter.addWidget(panel3)

        # Set initial sizes: ~45% / ~35% / ~20%
        splitter.setSizes([540, 420, 240])

        root.addWidget(splitter)

        self.statusBar().showMessage(
            "v0.2.0 MOCKUP  ·  Layout preview — no functionality connected"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Refined 3D Index — Mockup")
    app.setStyleSheet(STYLESHEET)

    win = MockupWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
