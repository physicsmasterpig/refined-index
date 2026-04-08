from setuptools import setup, Extension

setup(
    ext_modules=[
        Extension(
            "manifold_index.core._c_kernel._c_tet_index",
            sources=["src/manifold_index/core/_c_kernel/tet_index.c"],
        ),
    ],
)
