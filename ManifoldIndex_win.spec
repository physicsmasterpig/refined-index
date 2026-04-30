# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Refined Index Calculator v0.5 — Windows build.

Works in both pip (venv / GitHub Actions) and conda environments.
No hard-coded paths — all DLL resolution is auto-detected at build time.

Usage:
    pyinstaller ManifoldIndex_win.spec --noconfirm [--clean]

Output:
    dist/ManifoldIndex/ManifoldIndex.exe
"""

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_submodules,
)

# ── Paths ──────────────────────────────────────────────────────────
PROJECT = Path(SPECPATH)
SRC = PROJECT / "src"
APP_VERSION = "1.1.3"

# ── conda vs pip auto-detection ────────────────────────────────────
# conda environments have a Library/bin/ directory under sys.prefix
# that stores system DLLs (sqlite3, ssl, lzma, ...).
# PyInstaller's DLL walker does NOT search this path by default.
# pip/venv installs on GitHub Actions or standard Python don't have
# this directory — the DLLs are already on the system PATH.
_conda_lib_bin = Path(sys.prefix) / "Library" / "bin"
CONDA_LIB_BIN = _conda_lib_bin if _conda_lib_bin.exists() else None

if CONDA_LIB_BIN:
    # Prepend before Analysis() so the DLL dependency walker can
    # resolve all .pyd → .dll edges automatically.
    os.environ["PATH"] = str(CONDA_LIB_BIN) + os.pathsep + os.environ.get("PATH", "")

# ── collect_all for snappy ecosystem ──────────────────────────────
_snappy_datas, _snappy_bins, _snappy_hidden = collect_all("snappy")
_sm_datas,     _sm_bins,     _sm_hidden     = collect_all("snappy_manifolds")
_sph_datas,    _sph_bins,    _sph_hidden    = collect_all("spherogram")
_pl_datas,     _pl_bins,     _pl_hidden     = collect_all("plink")
_cy_datas,     _cy_bins,     _cy_hidden     = collect_all("cypari")
_fx_datas,     _fx_bins,     _fx_hidden     = collect_all("FXrays")
_kfh_datas,    _kfh_bins,    _kfh_hidden    = collect_all("knot_floer_homology")
_li_datas,     _li_bins,     _li_hidden     = collect_all("low_index")

# ── Hidden imports ─────────────────────────────────────────────────
hidden = collect_submodules("manifold_index")

hidden += _snappy_hidden + _sm_hidden + _sph_hidden + _pl_hidden
hidden += _cy_hidden + _fx_hidden + _kfh_hidden + _li_hidden

hidden += [
    "low_index",
    "networkx",
    "png",
    # stdlib C extensions (need explicit DLLs on conda)
    "_ssl", "ssl",
    "_sqlite3", "sqlite3",
    "_lzma", "_bz2",
    # scipy
    "scipy.special._ufuncs",
    "scipy.linalg",
    "scipy.linalg.lapack",
    "scipy.linalg.blas",
    "scipy.sparse",
    "scipy.sparse.linalg",
    # numpy
    "numpy._core._methods",
    "numpy.linalg._umath_linalg",
    # PySide6
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "PySide6.QtNetwork",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebChannel",
    # stdlib
    "fractions",
    "importlib.resources",
    "importlib.metadata",
    "threading",
    "json",
    "pathlib",
    # multiprocessing — Windows only (no popen_fork / popen_spawn_posix)
    "multiprocessing",
    "multiprocessing.pool",
    "multiprocessing.resource_tracker",
    "multiprocessing.spawn",
    "multiprocessing.popen_spawn_win32",
]

# ── Data files ─────────────────────────────────────────────────────
datas = []
datas += _snappy_datas + _sm_datas + _sph_datas + _pl_datas
datas += _cy_datas + _fx_datas + _kfh_datas + _li_datas
datas += collect_data_files("scipy")
datas += collect_data_files("manifold_index", subdir="data")
datas += [(str(PROJECT / "src" / "manifold_index" / "data" / "katex"),
           "manifold_index/data/katex")]

# ── Binaries ───────────────────────────────────────────────────────
binaries = []
binaries += _snappy_bins + _sm_bins + _sph_bins + _pl_bins
binaries += _cy_bins + _fx_bins + _kfh_bins + _li_bins

# conda only: glob all substantive DLLs from Library/bin
# pip/GitHub Actions: DLLs are already on PATH, PyInstaller finds them
if CONDA_LIB_BIN:
    # Exclude Windows API forwarding stubs (part of the OS, never bundle)
    # Exclude tcl/tk (tkinter is not used in this build)
    _EXCL_PREFIX = ("api-ms-win-",)
    _EXCL_NAME   = {"tcl86t.dll", "tk86t.dll"}
    for _dll in sorted(CONDA_LIB_BIN.glob("*.dll")):
        _n = _dll.name.lower()
        if _n in _EXCL_NAME:
            continue
        if any(_n.startswith(p) for p in _EXCL_PREFIX):
            continue
        binaries.append((str(_dll), "."))

# ── Analysis ───────────────────────────────────────────────────────
a = Analysis(
    [str(PROJECT / "launcher.py")],
    pathex=[str(SRC)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(PROJECT / "rthook_snappy.py")],
    excludes=[
        "tkinter",
        "matplotlib",
        "IPython",
        "jupyter",
        "notebook",
        "sphinx",
        "docutils",
        "sage",
        "pytest",
        "mypy",
        "ruff",
        "_pytest",
    ],
    noarchive=False,
    optimize=0,
)

# ── PYZ ────────────────────────────────────────────────────────────
pyz = PYZ(a.pure, cipher=None)

# ── EXE (onefile) ──────────────────────────────────────────────────
# All binaries and data are bundled into a single ManifoldIndex.exe.
# Output: dist/ManifoldIndex.exe
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    exclude_binaries=False,
    name="ManifoldIndex",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(PROJECT / "assets" / "ManifoldIndex.ico"),
)
