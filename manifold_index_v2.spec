# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Manifold Index Calculator v2.

Builds a macOS .app bundle from the app_v2 three-panel GUI with:
  - PySide6 + QtWebEngine (KaTeX math rendering)
  - snappy (SnapPea kernel + manifold databases)
  - numpy / scipy
  - C extension (_c_tet_index)

Usage:
    pyinstaller manifold_index_v2.spec --noconfirm
"""

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_submodules,
    collect_dynamic_libs,
)

# ── Paths ──────────────────────────────────────────────────────────
PROJECT = Path(SPECPATH)
SRC = PROJECT / "src"
PKG = SRC / "manifold_index"

# ── Version (single source of truth) ──────────────────────────────
APP_VERSION = "0.2.3"

# ── collect_all for snappy ecosystem ──────────────────────────────
_snappy_datas, _snappy_bins, _snappy_hidden = collect_all("snappy")
_sm_datas, _sm_bins, _sm_hidden = collect_all("snappy_manifolds")
_sph_datas, _sph_bins, _sph_hidden = collect_all("spherogram")
_pl_datas, _pl_bins, _pl_hidden = collect_all("plink")
_cy_datas, _cy_bins, _cy_hidden = collect_all("cypari")
_fx_datas, _fx_bins, _fx_hidden = collect_all("FXrays")
_kfh_datas, _kfh_bins, _kfh_hidden = collect_all("knot_floer_homology")
_li_datas, _li_bins, _li_hidden = collect_all("low_index")

# ── Hidden imports ─────────────────────────────────────────────────
hidden = collect_submodules("manifold_index")

# Snappy ecosystem
hidden += _snappy_hidden + _sm_hidden + _sph_hidden + _pl_hidden
hidden += _cy_hidden + _fx_hidden + _kfh_hidden + _li_hidden

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
    # PySide6 — core + WebEngine for KaTeX
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineCore",
    "PySide6.QtNetwork",
    # stdlib
    "fractions",
    "importlib.resources",
    "importlib.metadata",
]

# ── Data files ─────────────────────────────────────────────────────
datas = []
datas += _snappy_datas + _sm_datas + _sph_datas + _pl_datas
datas += _cy_datas + _fx_datas + _kfh_datas + _li_datas
datas += collect_data_files("scipy")

# PySide6 WebEngine resources (needed for KaTeX rendering)
try:
    datas += collect_data_files("PySide6", subdir="Qt/lib/QtWebEngineCore.framework")
except Exception:
    pass
try:
    datas += collect_data_files("PySide6", subdir="resources")
except Exception:
    pass

# ── Binaries ───────────────────────────────────────────────────────
binaries = []

# Our custom C extension
c_ext = PKG / "core" / "_c_tet_index.cpython-314-darwin.so"
if c_ext.exists():
    binaries.append(
        (str(c_ext), os.path.join("manifold_index", "core"))
    )

# Snappy ecosystem binaries
binaries += _snappy_bins + _sm_bins + _sph_bins + _pl_bins
binaries += _cy_bins + _fx_bins + _kfh_bins + _li_bins

# ── Analysis ───────────────────────────────────────────────────────
# Entry point: app_v2/__main__.py  (the three-panel v2 GUI)
a = Analysis(
    [str(PKG / "app_v2" / "__main__.py")],
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
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

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
    icon=None,
    bundle_identifier="com.manifold-index.calculator",
    info_plist={
        "CFBundleName": "Manifold Index",
        "CFBundleDisplayName": "Manifold Index Calculator",
        "CFBundleShortVersionString": APP_VERSION,
        "CFBundleVersion": APP_VERSION,
        "NSHighResolutionCapable": True,
        "NSRequiresAquaSystemAppearance": False,
        "LSMinimumSystemVersion": "11.0",
    },
)
