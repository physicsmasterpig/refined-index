"""
PyInstaller runtime hook for snappy + snappy_manifolds.

Ensures data-file paths (SQLite databases, .gz censuses) resolve
correctly inside the frozen bundle.

Strategy
--------
*Eager import + direct patch* — no meta-path finder, no deprecated
``find_module`` / ``load_module`` APIs.

1.  Import ``snappy_manifolds.sqlite_files`` first and, if necessary,
    fix its ``__path__`` so it points to the real on-disk directory
    that contains the ``.sqlite`` files.
2.  Import ``snappy_manifolds.database`` *after* the patch.  Because
    ``database.py`` reads ``from .sqlite_files import __path__`` at
    module level, it will pick up the corrected paths.
3.  Ensure the ``snappy`` package's ``__path__`` includes the on-disk
    directory that holds ``manifolds/HTWKnots/*.gz`` so the C
    extension can locate them.
4.  Write a short diagnostic to ``stderr`` for debugging when the
    app is launched from a terminal.
"""

import os
import sys


def _patch_snappy_paths() -> None:
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass is None:
        return  # not frozen — nothing to do

    # On macOS .app bundles produced by PyInstaller >= 6,
    # sys._MEIPASS -> Contents/Frameworks  (binaries + PYZ)
    # data files   -> Contents/Resources   (datas)
    #
    # Both locations may contain package sub-directories; we check
    # both so the hook works regardless of BUNDLE vs plain COLLECT.
    resources = os.path.join(os.path.dirname(meipass), "Resources")
    search_roots = [meipass]
    if os.path.isdir(resources) and resources != meipass:
        search_roots.append(resources)

    def _find_dir(*parts: str) -> str | None:
        """Return the first existing directory for *parts* under the
        search roots, or ``None``."""
        for root in search_roots:
            d = os.path.join(root, *parts)
            if os.path.isdir(d):
                return d
        return None

    # ── 1. snappy_manifolds.sqlite_files ──────────────────────────
    sqlite_dir = _find_dir("snappy_manifolds", "sqlite_files")
    if sqlite_dir:
        try:
            import snappy_manifolds.sqlite_files
            snappy_manifolds.sqlite_files.__path__ = [sqlite_dir]
        except Exception:
            pass  # will be diagnosed below

    # ── 2. snappy_manifolds.database ──────────────────────────────
    if sqlite_dir:
        try:
            import snappy_manifolds.database as _db

            _expected = {
                "database_path": "manifolds.sqlite",
                "alt_database_path": "more_manifolds.sqlite",
                "platonic_database_path": "platonic_manifolds.sqlite",
                "ribbon_database_path": "ribbon_links.sqlite",
            }
            for attr, filename in _expected.items():
                current = getattr(_db, attr, "")
                if not os.path.isfile(current):
                    fixed = os.path.join(sqlite_dir, filename)
                    if os.path.isfile(fixed):
                        setattr(_db, attr, fixed)
        except Exception:
            pass

    # ── 3. snappy package path ────────────────────────────────────
    snappy_dir = _find_dir("snappy")
    if snappy_dir:
        try:
            import snappy
            if snappy_dir not in snappy.__path__:
                snappy.__path__.insert(0, snappy_dir)
        except Exception:
            pass

    # ── 4. Diagnostics (visible when run from terminal) ───────────
    print("[rthook_snappy] frozen =", getattr(sys, "frozen", False),
          file=sys.stderr)
    print("[rthook_snappy] _MEIPASS =", meipass, file=sys.stderr)
    if sqlite_dir:
        for fn in ("manifolds.sqlite", "more_manifolds.sqlite"):
            full = os.path.join(sqlite_dir, fn)
            print(f"[rthook_snappy] {fn}: exists={os.path.isfile(full)}",
                  file=sys.stderr)
    else:
        print("[rthook_snappy] WARNING: sqlite_files dir not found!",
              file=sys.stderr)

    # Verify snappy manifold data dirs
    for sub in ("manifolds/HTWKnots",):
        for root in search_roots:
            d = os.path.join(root, "snappy", sub)
            if os.path.isdir(d):
                files = os.listdir(d)
                print(f"[rthook_snappy] snappy/{sub} ({root}): {files}",
                      file=sys.stderr)


_patch_snappy_paths()
