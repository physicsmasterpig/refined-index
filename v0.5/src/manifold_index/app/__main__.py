"""Entry point for the Refined Index Calculator GUI.

Run with::

    python -m manifold_index.app

or via the ``manifold-index`` console script defined in pyproject.toml.
"""

from __future__ import annotations

import sys

# ── SQLite thread-safety patch ────────────────────────────────────────────────
# SnaPy opens ~20 SQLite connections at import time, all bound to whichever
# thread first imports snappy (the main thread).  Python 3.12+ tightened the
# default check_same_thread=True enforcement, so worker QThreads that call
# snappy.Manifold() raise "SQLite object created in thread X used in thread Y".
#
# Fix: patch sqlite3.connect so every connection is created with
# check_same_thread=False.  This is safe here because snappy only ever *reads*
# its census databases; no concurrent writes occur.
#
# Must be done BEFORE any import that could transitively import snappy.
import sqlite3 as _sqlite3

_sqlite3_connect_orig = _sqlite3.connect


def _sqlite3_connect_nothreadcheck(*args, **kwargs):
    kwargs.setdefault("check_same_thread", False)
    return _sqlite3_connect_orig(*args, **kwargs)


_sqlite3.connect = _sqlite3_connect_nothreadcheck  # type: ignore[assignment]
# ─────────────────────────────────────────────────────────────────────────────


def launch_gui() -> None:
    """Create the QApplication and show the MainWindow."""
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt

    # High-DPI / Retina support (must be set before QApplication)
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("ManifoldIndex")
    app.setApplicationVersion("0.5.0")
    app.setOrganizationName("physicsmasterpig")

    from manifold_index.app.window import MainWindow
    win = MainWindow()
    win.show()

    sys.exit(app.exec())


# Allow `python -m manifold_index.app`
if __name__ == "__main__":
    launch_gui()

