"""PyInstaller entry point — launches the Refined Index Calculator GUI."""
import multiprocessing
multiprocessing.freeze_support()

# SQLite thread-safety patch — must come before any snappy import.
# See manifold_index/app/__main__.py for full explanation.
import sqlite3 as _sq
_sq_orig = _sq.connect
def _sq_nothreadcheck(*a, **kw):
    kw.setdefault("check_same_thread", False)
    return _sq_orig(*a, **kw)
_sq.connect = _sq_nothreadcheck  # type: ignore[assignment]

from manifold_index.app import launch_gui
launch_gui()
