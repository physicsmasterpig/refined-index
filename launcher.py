"""PyInstaller entry point — launches the Refined Index Calculator GUI."""
import multiprocessing
multiprocessing.freeze_support()

# SQLite thread-safety patch. In the frozen app the authoritative copy
# lives in rthook_snappy.py (which performs the first snappy import,
# before this script runs); this one covers non-frozen paths and is a
# harmless no-op re-patch otherwise.
import sqlite3 as _sq
_sq_orig = _sq.connect
def _sq_nothreadcheck(*a, **kw):
    kw.setdefault("check_same_thread", False)
    return _sq_orig(*a, **kw)
_sq.connect = _sq_nothreadcheck  # type: ignore[assignment]

# Crash trail before the heavy app import — an ImportError inside the
# bundle would otherwise vanish (console=False on Windows).
from manifold_index import __version__ as _app_version
from manifold_index.app import crash_log
crash_log.install(app_version=_app_version)

from manifold_index.app import launch_gui
launch_gui()
