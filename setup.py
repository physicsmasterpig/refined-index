"""setup.py — C extension build for manifold-index.

This file exists alongside pyproject.toml to declare the C extension module.
All other project metadata lives in pyproject.toml.

Usage:
    pip install -e .          # editable install (compiles C extension)
    pip install .             # normal install
    python setup.py build_ext --inplace   # build C extension in-place (dev)
"""

from setuptools import setup, Extension

ext_modules = [
    Extension(
        "manifold_index.core._c_tet_index",
        sources=["src/manifold_index/core/_c_kernel/tet_index.c"],
        extra_compile_args=["-O2", "-Wall"],
    ),
]

setup(ext_modules=ext_modules)
