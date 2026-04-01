# Phase 0: Project Skeleton

## Goal
Create an installable Python package with the correct directory structure.
After this phase, `pip install -e ".[dev]"` and `import manifold_index` work.

## Files to Create

### `pyproject.toml`
```toml
[build-system]
requires = ["setuptools>=68", "setuptools-scm"]
build-backend = "setuptools.build_meta"

[project]
name = "manifold-index"
version = "0.4.0"
description = "3-Manifold index calculator: 3D index, Dehn filling, non-closable cycles, and refined index."
readme = "README.md"
requires-python = ">=3.10"
license = { text = "GPL-2.0-or-later" }
dependencies = ["snappy", "numpy", "scipy"]

[project.optional-dependencies]
gui = ["PySide6"]
dev = ["pytest", "pytest-cov", "pytest-timeout", "ruff", "mypy"]

[project.scripts]
manifold-index = "manifold_index.app:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
manifold_index = [
    "data/kernel_cache/*.pkl.gz",
    "data/data_packs.json",
]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.mypy]
python_version = "3.10"
strict = false
ignore_missing_imports = true
```

### `README.md`
A short placeholder — will be expanded later.

### `src/manifold_index/__init__.py`
```python
"""
manifold_index — 3-Manifold Index Calculator
"""
```

### `src/manifold_index/core/__init__.py`
```python
"""core package — mathematical pipeline modules"""
```

### `src/manifold_index/utils/__init__.py`
```python
"""utils package — shared helper functions"""
```

### `src/manifold_index/app/__init__.py`
```python
"""app package — PySide6 GUI (optional)"""
```

### Empty directories (create with `.gitkeep` or `__init__.py`):
- `src/manifold_index/app/panels/`
- `src/manifold_index/data/kernel_cache/`

## Acceptance Criteria

```bash
cd v0.4
pip install -e ".[dev]"
python -c "import manifold_index; print('OK')"
pytest --co  # collection succeeds (0 tests found is fine)
```

All three commands must succeed without error.

---

*Phase 0 complete → proceed to Phase 1.*
