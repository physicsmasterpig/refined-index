# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Manifold Index Calculator.

Builds a macOS .app bundle with all dependencies:
  - PySide6 GUI
  - snappy (SnapPea kernel + manifold databases)
  - numpy / scipy
  - C extension (_c_tet_index)

Usage:
    pyinstaller manifold_index.spec --noconfirm
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

# ── collect_all for snappy ecosystem ──────────────────────────────
# collect_all returns (datas, binaries, hiddenimports) — this is the
# most thorough collection method and avoids missing any data files,
# submodules, or shared libraries.

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

# Snappy ecosystem (from collect_all)
hidden += _snappy_hidden
hidden += _sm_hidden
hidden += _sph_hidden
hidden += _pl_hidden
hidden += _cy_hidden
hidden += _fx_hidden
hidden += _kfh_hidden
hidden += _li_hidden

# Additional snappy runtime dependencies discovered via import tracing
hidden += [
    "low_index",
    "networkx",
    "png",
]

# Extras that auto-discovery sometimes misses
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
    # PySide6
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    # stdlib
    "fractions",
    "importlib.resources",
    "importlib.metadata",
]

# ── Data files ─────────────────────────────────────────────────────
datas = []

# 1. Snappy ecosystem data files (from collect_all)
datas += _snappy_datas
datas += _sm_datas
datas += _sph_datas
datas += _pl_datas
datas += _cy_datas
datas += _fx_datas
datas += _kfh_datas
datas += _li_datas

# 4. scipy data
datas += collect_data_files("scipy")

# ── Binaries ───────────────────────────────────────────────────────
binaries = []

# Our custom C extension
c_ext = PKG / "core" / "_c_tet_index.cpython-314-darwin.so"
if c_ext.exists():
    binaries.append(
        (str(c_ext), os.path.join("manifold_index", "core"))
    )

# Snappy ecosystem binaries (from collect_all)
binaries += _snappy_bins
binaries += _sm_bins
binaries += _sph_bins
binaries += _pl_bins
binaries += _cy_bins
binaries += _fx_bins
binaries += _kfh_bins
binaries += _li_bins

# ── Analysis ───────────────────────────────────────────────────────
a = Analysis(
    [str(PKG / "app" / "main.py")],
    pathex=[str(SRC)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(PROJECT / "rthook_snappy.py")],
    excludes=[
        # Exclude unnecessary large packages
        "tkinter",
        "matplotlib",
        "IPython",
        "jupyter",
        "notebook",
        "sphinx",
        "docutils",
        "sage",
        # Exclude test frameworks from bundle
        "pytest",
        "mypy",
        "ruff",
        "_pytest",
    ],
    noarchive=False,
    optimize=0,
)

# ── PYZ (compressed Python archive) ───────────────────────────────
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# ── EXE ────────────────────────────────────────────────────────────
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,  # onedir mode
    name="ManifoldIndex",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,              # UPX disabled — can corrupt .so on macOS
    console=False,          # Windowed app, no terminal
    disable_windowed_traceback=False,
    argv_emulation=True,    # macOS: accept file drops
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# ── COLLECT (onedir) ──────────────────────────────────────────────
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,              # UPX disabled
    upx_exclude=[],
    name="ManifoldIndex",
)

# ── macOS .app BUNDLE ─────────────────────────────────────────────
app = BUNDLE(
    coll,
    name="ManifoldIndex.app",
    icon=None,  # TODO: add an .icns file if desired
    bundle_identifier="com.manifold-index.calculator",
    info_plist={
        "CFBundleName": "Manifold Index",
        "CFBundleDisplayName": "Manifold Index Calculator",
        "CFBundleShortVersionString": "0.1.0",
        "CFBundleVersion": "0.1.0",
        "NSHighResolutionCapable": True,
        "NSRequiresAquaSystemAppearance": False,  # support Dark Mode
        "LSMinimumSystemVersion": "11.0",
    },
)
