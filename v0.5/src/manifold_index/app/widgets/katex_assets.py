"""Resolve local KaTeX asset paths for offline math rendering.

KaTeX CSS, JS, and font files are bundled under
``manifold_index/data/katex/``.  This module exposes the base URL
and the three ``<link>``/``<script>`` tags that widgets embed into
their HTML pages.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QUrl

# ── Locate the katex directory ─────────────────────────────────────
# Works both for dev (pip install -e .) and PyInstaller bundles.
if getattr(sys, "frozen", False):
    _BASE = Path(sys._MEIPASS) / "manifold_index" / "data" / "katex"
else:
    _BASE = Path(__file__).resolve().parents[2] / "data" / "katex"


def katex_base_url() -> QUrl:
    """``QUrl`` pointing to the local katex directory (for ``setHtml()``)."""
    return QUrl.fromLocalFile(str(_BASE) + "/")


def katex_head_tags() -> str:
    """``<link>`` + ``<script>`` tags loading KaTeX from local files."""
    css_path = (_BASE / "katex.min.css").as_uri()
    js_path = (_BASE / "katex.min.js").as_uri()
    ar_path = (_BASE / "auto-render.min.js").as_uri()
    return (
        f'<link rel="stylesheet" href="{css_path}">\n'
        f'<script defer src="{js_path}"></script>\n'
        f'<script defer src="{ar_path}"></script>'
    )
