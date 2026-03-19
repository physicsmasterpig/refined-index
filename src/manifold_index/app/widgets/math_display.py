"""
widgets/math_display.py — KaTeX-powered math display widget.

Uses QWebEngineView to render LaTeX via KaTeX (loaded from CDN).
Falls back to a styled QTextBrowser if QtWebEngine is unavailable.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import QVBoxLayout, QWidget

# ---------------------------------------------------------------------------
# Try to import QWebEngineView (may not be installed)
# ---------------------------------------------------------------------------

_HAS_WEBENGINE = False
try:
    from PySide6.QtWebEngineWidgets import QWebEngineView  # type: ignore[attr-defined]
    from PySide6.QtWebEngineCore import QWebEnginePage      # type: ignore[attr-defined]
    _HAS_WEBENGINE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# KaTeX HTML template
# ---------------------------------------------------------------------------

_KATEX_VERSION = "0.16.21"

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/katex@{version}/dist/katex.min.css"
      crossorigin="anonymous">
<script defer
        src="https://cdn.jsdelivr.net/npm/katex@{version}/dist/katex.min.js"
        crossorigin="anonymous"></script>
<script defer
        src="https://cdn.jsdelivr.net/npm/katex@{version}/dist/contrib/auto-render.min.js"
        crossorigin="anonymous"></script>
<style>
:root {{
    color-scheme: light dark;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: {font_size}px;
    line-height: 1.7;
    padding: 14px 18px;
    background: {bg};
    color: {fg};
    overflow-x: auto;
}}
.sector {{
    margin-bottom: 10px;
    padding: 6px 0;
    border-bottom: 1px solid {border};
}}
.sector:last-child {{
    border-bottom: none;
    margin-bottom: 0;
}}
.sector-label {{
    font-size: {label_size}px;
    color: {muted};
    margin-bottom: 2px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}}
.katex-display {{
    margin: 4px 0 !important;
    overflow-x: auto;
    overflow-y: hidden;
    padding: 2px 0;
}}
.katex {{
    font-size: 1.05em !important;
}}
.info-block {{
    font-family: "SF Mono", "Menlo", "Monaco", "Consolas", monospace;
    font-size: {mono_size}px;
    line-height: 1.5;
    white-space: pre-wrap;
    word-break: break-all;
}}
h3 {{
    font-size: {heading_size}px;
    margin: 12px 0 6px 0;
    padding-bottom: 4px;
    border-bottom: 2px solid {accent};
    color: {fg};
}}
h3:first-child {{ margin-top: 0; }}
.edge-table {{
    font-family: "SF Mono", "Menlo", "Monaco", "Consolas", monospace;
    font-size: {mono_size}px;
    border-collapse: collapse;
    margin: 4px 0;
}}
.edge-table td, .edge-table th {{
    padding: 2px 8px;
    text-align: left;
    white-space: nowrap;
}}
.edge-table th {{
    font-weight: 600;
    border-bottom: 1px solid {border};
}}
.success {{ color: #2ea043; }}
.error   {{ color: #d1242f; }}
.muted   {{ color: {muted}; }}
</style>
</head>
<body>
{content}
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


def _system_colors() -> dict[str, str]:
    """Resolve current palette colours for the HTML template."""
    from PySide6.QtWidgets import QApplication
    pal = QApplication.instance().palette()
    bg = pal.color(pal.ColorRole.Base)
    fg = pal.color(pal.ColorRole.Text)
    mid = pal.color(pal.ColorRole.Mid)
    accent = pal.color(pal.ColorRole.Highlight)
    border = pal.color(pal.ColorRole.Mid)
    return {
        "bg": bg.name(),
        "fg": fg.name(),
        "muted": mid.name(),
        "accent": accent.name(),
        "border": border.name(),
    }


# ---------------------------------------------------------------------------
# MathDisplay widget
# ---------------------------------------------------------------------------

class MathDisplay(QWidget):
    """Widget that renders LaTeX + HTML content via KaTeX.

    Usage::

        md = MathDisplay()
        md.set_content("<h3>Result</h3>$$x^2 + y^2 = z^2$$")
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        font_size: int = 14,
        mono_size: int = 12,
        label_size: int = 11,
        heading_size: int = 14,
        min_height: int = 180,
    ) -> None:
        super().__init__(parent)
        self._font_size = font_size
        self._mono_size = mono_size
        self._label_size = label_size
        self._heading_size = heading_size
        self._pending_html: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if _HAS_WEBENGINE:
            self._view = QWebEngineView()
            self._view.setMinimumHeight(min_height)
            self._view.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
            layout.addWidget(self._view, 1)
            self._fallback = None
        else:
            from PySide6.QtWidgets import QTextBrowser
            self._view = None
            self._fallback = QTextBrowser()
            self._fallback.setReadOnly(True)
            self._fallback.setOpenExternalLinks(False)
            self._fallback.setMinimumHeight(min_height)
            layout.addWidget(self._fallback, 1)

    # ---------------------------------------------------------------

    def set_content(self, html_body: str) -> None:
        """Set the inner HTML content (may include $…$ and $$…$$ for KaTeX)."""
        if self._view is not None:
            colors = _system_colors()
            full_html = _HTML_TEMPLATE.format(
                version=_KATEX_VERSION,
                font_size=self._font_size,
                mono_size=self._mono_size,
                label_size=self._label_size,
                heading_size=self._heading_size,
                content=html_body,
                **colors,
            )
            self._view.setHtml(full_html, QUrl("https://cdn.jsdelivr.net/"))
        elif self._fallback is not None:
            # Fallback: just show as HTML (no LaTeX rendering)
            self._fallback.setHtml(html_body)

    def clear(self) -> None:
        if self._view is not None:
            self._view.setHtml("")
        elif self._fallback is not None:
            self._fallback.clear()

    def set_plain_text(self, text: str) -> None:
        """Convenience: wrap plain text in a <pre> block."""
        escaped = (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        self.set_content(f'<div class="info-block">{escaped}</div>')
