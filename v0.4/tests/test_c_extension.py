"""Tests for C extension (optional — skipped if not compiled)."""
import pytest


@pytest.fixture(scope="module")
def c_ext():
    try:
        from manifold_index.core._c_kernel._c_tet_index import (
            tet_index_series,
            tet_degree_x2,
        )
        return tet_index_series, tet_degree_x2
    except ImportError:
        pytest.skip("C extension not compiled")


def test_tet_degree_x2_agreement(c_ext):
    from manifold_index.core.index_3d import _tet_degree_x2 as py_deg
    c_tet_index_series, c_tet_deg = c_ext
    for m in range(-10, 11):
        for e in range(-10, 11):
            assert c_tet_deg(m, e) == py_deg(m, e), f"Mismatch at m={m}, e={e}"


def test_tet_index_series_agreement(c_ext):
    from manifold_index.core.index_3d import _tet_index_series_python as py_fn
    c_fn, _ = c_ext
    for m in range(-5, 6):
        for e in range(-5, 6):
            assert c_fn(m, e, 10) == py_fn(m, e, 10), f"Mismatch at m={m}, e={e}"


def test_fallback_when_no_c():
    """The pure-Python fallback must always work."""
    from manifold_index.core.index_3d import _tet_index_series_python
    s = _tet_index_series_python(0, 0, 10)
    assert 0 in s
