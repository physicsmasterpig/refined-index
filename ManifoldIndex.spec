# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for ManifoldIndex.app (macOS arm64)."""

import os
import site

block_cipher = None

sp = site.getsitepackages()[0]

a = Analysis(
    ['launcher.py'],
    pathex=[os.path.join(os.getcwd(), 'src')],
    binaries=[],
    datas=[
        (os.path.join('src', 'manifold_index', 'data'), os.path.join('manifold_index', 'data')),
        (os.path.join(sp, 'snappy'), 'snappy'),
        (os.path.join(sp, 'snappy_manifolds'), 'snappy_manifolds'),
        (os.path.join(sp, 'cypari'), 'cypari'),
        (os.path.join(sp, 'plink'), 'plink'),
    ],
    hiddenimports=[
        'manifold_index', 'manifold_index.app', 'manifold_index.app.window',
        'manifold_index.app.panels', 'manifold_index.app.panels.index_panel',
        'manifold_index.app.panels.refined_panel', 'manifold_index.app.panels.dehn_panel',
        'manifold_index.app.panels.export_panel', 'manifold_index.app.panels.kernel_panel',
        'manifold_index.core', 'manifold_index.core.manifold',
        'manifold_index.core.gluing_equations', 'manifold_index.core.phase_space',
        'manifold_index.core.neumann_zagier', 'manifold_index.core.index_3d',
        'manifold_index.core.refined_index', 'manifold_index.core.dehn_filling',
        'manifold_index.core.basis_selection',
        'manifold_index.utils', 'manifold_index.utils.exporters',
        'snappy', 'snappy.SnapPy', 'snappy.SnapPyHP',
        'snappy.snap', 'snappy.snap.t3mlite', 'snappy.manifolds', 'snappy.database',
        'plink', 'cypari', 'cypari._pari',
        'numpy', 'numpy._core', 'numpy._core._methods',
        'scipy', 'scipy.sparse', 'scipy.linalg',
        'PySide6', 'PySide6.QtCore', 'PySide6.QtWidgets', 'PySide6.QtGui',
        'fractions',
        'multiprocessing', 'multiprocessing.pool',
        'multiprocessing.resource_tracker', 'multiprocessing.spawn',
        'multiprocessing.forkserver', 'multiprocessing.popen_spawn_posix',
        'multiprocessing.popen_fork',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', '_tkinter', 'matplotlib', 'IPython', 'jupyter', 'test', 'tests'],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name='ManifoldIndex',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    target_arch='arm64',
)

coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=False, upx=False,
    name='ManifoldIndex',
)

app = BUNDLE(
    coll,
    name='ManifoldIndex.app',
    icon=None,
    bundle_identifier='com.physicsmasterpig.manifoldindex',
    info_plist={
        'CFBundleShortVersionString': '0.3.5',
        'CFBundleVersion': '0.3.5',
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '11.0',
    },
)
