"""Tests for 3D index computation."""
from __future__ import annotations

from fractions import Fraction
import pytest
from manifold_index.core.index_3d import (
    tet_degree, _tet_degree_x2, _tet_index_series_python,
    enumerate_summation_terms, compute_index_3d_python,
)


def test_tet_degree_known_values():
    assert tet_degree(1, 0) == Fraction(1, 1)   # exact: m(m+e+1)/2 = 1*2/2 = 1
    assert tet_degree(0, 0) == 0


def test_tet_degree_symmetry():
    """δ(m,e) == δ(-e,-m)"""
    for m in range(-3, 4):
        for e in range(-3, 4):
            assert tet_degree(m, e) == tet_degree(-e, -m)


def test_tet_degree_nonneg():
    for m in range(-3, 4):
        for e in range(-3, 4):
            assert tet_degree(m, e) >= 0


def test_tet_degree_x2_matches():
    for m in range(-3, 4):
        for e in range(-3, 4):
            assert _tet_degree_x2(m, e) == int(2 * tet_degree(m, e))


def test_tet_index_series_basic():
    """I_Δ(0, 0) should have nonzero constant term."""
    s = _tet_index_series_python(0, 0, 10)
    assert 0 in s and s[0] != 0


def test_enumerate_summation_terms(nz_m004):
    ext = [0] * nz_m004.r
    terms = enumerate_summation_terms(nz_m004, ext, ext, q_order_half=10)
    assert len(terms) > 1
    assert all("phase_exp" in t and "tet_args" in t for t in terms)


def test_compute_index_basic(nz_m004):
    pytest.importorskip("snappy")
    result = compute_index_3d_python(
        nz_m004, [1], [0], q_order_half=8
    )
    assert len(result.coeffs) > 0
    assert result.n_terms > 0
