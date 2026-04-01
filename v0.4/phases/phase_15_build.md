# Phase 15 — Build & Packaging

> **Files:**
> - `pyproject.toml`
> - `src/manifold_index/__init__.py`
> - `ManifoldIndex.spec` (optional PyInstaller spec)
> - `build_app.sh` (optional macOS app bundle script)
> - `launcher.py` (optional PyInstaller entry point)
> - `rthook_snappy.py` (optional PyInstaller runtime hook)
>
> **Depends on:** all phases

---

## 0. Purpose

Package the manifold-index project for installation via pip, editable
development installs, and optionally as a standalone macOS app bundle
via PyInstaller.

---

## 1. `pyproject.toml`

### 1.1 Build System

```toml
[build-system]
requires = ["setuptools>=68", "setuptools-scm"]
build-backend = "setuptools.build_meta"
```

### 1.2 Project Metadata

```toml
[project]
name = "manifold-index"
version = "0.4.0"
description = "3-Manifold index calculator: 3D index, Dehn filling, non-closable cycles, and refined index."
readme = "README.md"
requires-python = ">=3.10"
license = { text = "GPL-2.0-or-later" }
classifiers = [
    "Development Status :: 4 - Beta",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Python :: 3.14",
    "Programming Language :: C",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Operating System :: OS Independent",
]
```

### 1.3 Dependencies

```toml
dependencies = [
    "snappy",
    "numpy",
    "scipy",
]

[project.optional-dependencies]
gui = ["PySide6"]
dev = ["pytest", "pytest-cov", "pytest-timeout", "ruff", "mypy"]
```

### 1.4 Entry Points

```toml
[project.scripts]
manifold-index = "manifold_index.app:main"
```

### 1.5 Package Discovery

```toml
[tool.setuptools.packages.find]
where = ["src"]
```

### 1.6 Package Data

```toml
[tool.setuptools.package-data]
manifold_index = [
    "mathematica/*.wl",
    "data/tet_index/**",
    "data/kernel_cache/*.pkl.gz",
    "data/data_packs.json",
]
```

### 1.7 C Extension (Phase 12)

The C extension must be declared for setuptools to compile it.
Either via `pyproject.toml` extension table or a `setup.py` shim:

```python
# setup.py (if needed for C extension)
from setuptools import setup, Extension
setup(
    ext_modules=[
        Extension(
            "manifold_index.core._c_kernel._c_tet_index",
            sources=["src/manifold_index/core/_c_kernel/tet_index.c"],
        ),
    ],
)
```

### 1.8 Tool Configuration

```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.mypy]
python_version = "3.10"
strict = false
ignore_missing_imports = true
```

---

## 2. Package Layout

```
src/
  manifold_index/
    __init__.py            # version, top-level imports
    core/
      __init__.py
      manifold.py          # Phase 1
      gluing_equations.py  # Phase 2
      basis_selection.py   # Phase 3
      neumann_zagier.py    # Phase 4
      phase_space.py       # Phase 5
      index_3d.py          # Phase 6
      refined_index.py     # Phase 7
      weyl_check.py        # Phase 8
      dehn_filling.py      # Phase 9
      refined_dehn_filling.py  # Phase 10
      kernel_cache.py      # Phase 11
      _c_kernel/
        __init__.py
        tet_index.c        # Phase 12
    utils/
      __init__.py
      exporters.py         # Phase 13
    app/
      __init__.py
      __main__.py
      window.py            # Phase 14
      workers.py
      formatters.py
      katex.py
      style.py
      panels/
        __init__.py
        manifold_panel.py
        filling_panel.py
        export_panel.py
        kernel_panel.py
        data_panel.py
    data/
      kernel_cache/        # bundled pre-computed kernels
      data_packs.json      # registry of downloadable packs
    mathematica/
      *.wl                 # Mathematica helper functions
```

---

## 3. `__init__.py`

```python
"""manifold-index — Refined 3D index calculator for cusped hyperbolic 3-manifolds."""
__version__ = "0.4.0"
```

---

## 4. Installation Commands

### 4.1 Editable Development Install

```bash
pip install -e ".[dev,gui]"
```

### 4.2 Production Install

```bash
pip install .
# or with GUI:
pip install ".[gui]"
```

### 4.3 Running

```bash
# CLI (when GUI deps installed)
manifold-index

# Module
python -m manifold_index.app

# Programmatic
from manifold_index.core.manifold import load_manifold
```

---

## 5. PyInstaller Bundle (Optional)

### 5.1 `ManifoldIndex.spec`

PyInstaller spec file for creating a macOS `.app` bundle.
Key considerations:
- Collect snappy data files (topology databases)
- Include bundled kernel cache `.pkl.gz` files
- Runtime hook for snappy's native libraries (`rthook_snappy.py`)
- PySide6 plugins and Qt platform files

### 5.2 `build_app.sh`

Shell script that:
1. Activates the correct conda/venv environment
2. Runs `pyinstaller ManifoldIndex.spec`
3. Creates the `.app` bundle in `dist/`

### 5.3 `launcher.py`

Minimal entry point for PyInstaller:
```python
from manifold_index.app import main
main()
```

### 5.4 `rthook_snappy.py`

Runtime hook that sets up snappy's data paths when running from
a frozen PyInstaller bundle.

---

## 6. Testing

### 6.1 Test Runner

```bash
pytest tests/ -v
```

### 6.2 Coverage

```bash
pytest tests/ --cov=manifold_index --cov-report=html
```

### 6.3 Linting

```bash
ruff check src/ tests/
```

### 6.4 Type Checking

```bash
mypy src/manifold_index/
```

---

## 7. Tests

### T15.1 — Package Importable

```python
import manifold_index
assert hasattr(manifold_index, "__version__")
```

### T15.2 — Core Importable Without GUI

```python
# Should work without PySide6
from manifold_index.core.manifold import load_manifold
from manifold_index.core.refined_index import compute_refined_index
```

### T15.3 — Entry Point Exists

```bash
pip install -e .
manifold-index --help  # or check it at least starts
```

### T15.4 — C Extension Builds

```bash
pip install -e .
python -c "from manifold_index.core._c_kernel._c_tet_index import tet_index_series; print('OK')"
```

### T15.5 — Package Data Included

```python
from importlib.resources import files
data_dir = files("manifold_index") / "data"
assert (data_dir / "data_packs.json").is_file()
```

---

## 8. Acceptance Criteria

- [ ] `pip install -e ".[dev,gui]"` succeeds on macOS and Linux
- [ ] `pip install .` succeeds without GUI dependencies
- [ ] C extension compiles automatically during install
- [ ] All `pytest` tests pass after fresh install
- [ ] `ruff check` clean
- [ ] `mypy` passes (with `ignore_missing_imports`)
- [ ] Package data files are accessible at runtime
- [ ] Entry point `manifold-index` works
