# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Refined Index Calculator v0.5.

Usage:
    pyinstaller ManifoldIndex.spec --noconfirm

Output:
    dist/ManifoldIndex.app   — standalone macOS application bundle
    dist/ManifoldIndex.zip   — distributable zip (created by build_app.sh)

Changes from v0.4:
  - APP_VERSION bumped to 0.5.0
  - CFBundleDisplayName updated to "Refined Index Calculator"
  - Entry point: launcher.py → manifold_index.app:launch_gui  (same file)
  - Added PySide6.QtWebEngineWidgets to hidden imports (math_view.py)
  - Added manifold_index.services.*, manifold_index.viewmodels.*,
    manifold_index.formatters.*, manifold_index.app.* sub-packages
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
PKG = SRC / "manifold_index"

# ── Version ────────────────────────────────────────────────────────
APP_VERSION = "0.5.4"

# ── collect_all for snappy ecosystem and UI framework ────────────
_snappy_datas, _snappy_bins, _snappy_hidden = collect_all("snappy")
_sm_datas,     _sm_bins,     _sm_hidden     = collect_all("snappy_manifolds")
_sph_datas,    _sph_bins,    _sph_hidden    = collect_all("spherogram")
_pl_datas,     _pl_bins,     _pl_hidden     = collect_all("plink")
_cy_datas,     _cy_bins,     _cy_hidden     = collect_all("cypari")
_fx_datas,     _fx_bins,     _fx_hidden     = collect_all("FXrays")
_kfh_datas,    _kfh_bins,    _kfh_hidden    = collect_all("knot_floer_homology")
_li_datas,     _li_bins,     _li_hidden     = collect_all("low_index")
_pyside_datas, _pyside_bins, _pyside_hidden = collect_all("PySide6")

# ── Hidden imports ─────────────────────────────────────────────────
hidden = collect_submodules("manifold_index")

# Snappy ecosystem
hidden += _snappy_hidden + _sm_hidden + _sph_hidden + _pl_hidden
hidden += _cy_hidden + _fx_hidden + _kfh_hidden + _li_hidden

# PySide6 ecosystem
hidden += _pyside_hidden

# Additional runtime deps
hidden += [
    "low_index",
    "networkx",
    "png",
]

# stdlib / third-party that auto-discovery misses
hidden += [
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
    # PySide6 — collected via collect_all above, but listed for clarity
    "PySide6",
    # stdlib
    "fractions",
    "importlib.resources",
    "importlib.metadata",
    "threading",
    "json",
    "pathlib",
    # multiprocessing
    "multiprocessing",
    "multiprocessing.pool",
    "multiprocessing.resource_tracker",
    "multiprocessing.spawn",
    "multiprocessing.forkserver",
    "multiprocessing.popen_spawn_posix",
    "multiprocessing.popen_fork",
]

# ── Data files ─────────────────────────────────────────────────────
datas = []
datas += _snappy_datas + _sm_datas + _sph_datas + _pl_datas
datas += _cy_datas + _fx_datas + _kfh_datas + _li_datas
datas += collect_data_files("scipy")
datas += _pyside_datas  # PySide6 plugins, libraries, etc.

# Our package data (data_packs.json, kernel caches, etc.)
datas += collect_data_files("manifold_index", subdir="data")

# ── Binaries ───────────────────────────────────────────────────────
binaries = []

# Snappy ecosystem binaries
binaries += _snappy_bins + _sm_bins + _sph_bins + _pl_bins
binaries += _cy_bins + _fx_bins + _kfh_bins + _li_bins

# PySide6 binaries (Qt frameworks, libraries)
binaries += _pyside_bins

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

# ── EXE ────────────────────────────────────────────────────────────
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ManifoldIndex",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=True,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# ── COLLECT ────────────────────────────────────────────────────────
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="ManifoldIndex",
)

# ── macOS .app BUNDLE ─────────────────────────────────────────────
app = BUNDLE(
    coll,
    name="ManifoldIndex.app",
    icon="assets/ManifoldIndex.icns",
    bundle_identifier="com.manifold-index.calculator",
    info_plist={
        "CFBundleName": "Refined Index",
        "CFBundleDisplayName": "Refined Index Calculator",
        "CFBundleShortVersionString": APP_VERSION,
        "CFBundleVersion": APP_VERSION,
        "NSHighResolutionCapable": True,
        "NSRequiresAquaSystemAppearance": False,
        "LSMinimumSystemVersion": "11.0",
    },
)
