"""app/widgets/math_label.py — Inline KaTeX label for mathematical notation.

Lightweight wrapper rendering a single LaTeX expression via KaTeX.
Usage: label = MathLabel("N_{\\max}:")
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import QSizePolicy, QWidget, QVBoxLayout

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    _HAS_WEBENGINE = True
except ImportError:
    from PySide6.QtWidgets import QLabel  # fallback
    _HAS_WEBENGINE = False

from manifold_index.app.widgets.math_view import build_katex_html, sys_colors

KATEX_VERSION = "0.16.21"


class MathLabel(QWidget):
    """Inline mathematical label using KaTeX.

    Renders a single LaTeX expression inline (e.g., N_{\\max}, \\mathcal{I}, etc.)

    Usage::

        label = MathLabel("N_{\\max}:")
        layout.addWidget(label)
    """

    def __init__(self, latex: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._latex = latex

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        if _HAS_WEBENGINE:
            self._view = QWebEngineView()
            self._view.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
            self._view.setMinimumHeight(20)
            self._view.setMaximumHeight(22)
            self._view.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
            # Set transparent background
            self._view.setStyleSheet("background-color: transparent; border: none;")
            layout.addWidget(self._view)
            self._render()
        else:
            # Fallback to plain text if WebEngine unavailable
            from PySide6.QtWidgets import QLabel
            label = QLabel(latex)
            label.setMaximumHeight(20)
            layout.addWidget(label)

    def _render(self) -> None:
        """Render the LaTeX expression via KaTeX."""
        if not _HAS_WEBENGINE:
            return

        # Create minimal HTML with the LaTeX expression
        body = f"<span>${self._latex}$</span>"

        # Build full KaTeX HTML page with minimal styling
        colors = sys_colors()
        css = f"""
        :root {{ color-scheme: light dark; }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        html, body {{
            margin: 0;
            padding: 0;
            height: 100%;
            background: transparent;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            font-size: 13px;
            background: transparent;
            color: {colors['fg']};
            display: flex;
            align-items: center;
            justify-content: flex-start;
        }}
        span {{
            white-space: nowrap;
            padding: 0;
            margin: 0;
        }}
        .katex {{
            font-size: 0.95em !important;
            margin: 0 !important;
        }}
        .katex-html {{
            padding: 0 !important;
        }}
        """

        full_html = f"""<!DOCTYPE html>
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
<style>{css}</style>
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

        self._view.setHtml(full_html, QUrl("https://cdn.jsdelivr.net/"))

    def set_latex(self, latex: str) -> None:
        """Update the LaTeX expression."""
        self._latex = latex
        if _HAS_WEBENGINE:
            self._render()
