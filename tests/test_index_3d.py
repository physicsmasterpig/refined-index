"""Tests for 3D index computation."""

from fractions import Fraction
from manifold_index.core.index_3d import tet_degree, enumerate_summation_terms


def test_tet_degree():
    assert tet_degree(1, 0) == Fraction(3, 2)
    for m in range(-3, 4):
        for e in range(-3, 4):
            assert tet_degree(m, e) >= 0


def test_enumerate_summation_terms(nz_m004):
    ext = [0] * nz_m004.r
    terms = enumerate_summation_terms(nz_m004, ext, ext, q_order_half=10)
    assert len(terms) > 1
    assert all("phase_exp" in t and "tet_args" in t for t in terms)
