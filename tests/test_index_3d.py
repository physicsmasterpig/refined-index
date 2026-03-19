"""tests/test_index_3d.py — Smoke tests for 3D index computation."""

from fractions import Fraction
import pytest
from manifold_index.core.index_3d import (
    tet_degree, build_kappa, enumerate_summation_terms,
)


def test_tet_degree_one_zero():
    assert tet_degree(1, 0) == Fraction(3, 2)


def test_tet_degree_nonnegative():
    for m in range(-3, 4):
        for e in range(-3, 4):
            assert tet_degree(m, e) >= 0


def test_build_kappa_shape():
    kappa = build_kappa([1], [Fraction(3, 2)], [], n=1, r=1)
    assert kappa.shape == (2,)
    assert kappa[0] == 1
    assert kappa[1] == Fraction(3, 2)


def test_enumerate_summation_terms_m004():
    pytest.importorskip("snappy")
    from manifold_index.core.manifold import load_manifold
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import build_neumann_zagier
    data = load_manifold("m004")
    easy = find_easy_edges(data)
    nz = build_neumann_zagier(data, easy)
    ext = [0] * nz.r
    result = enumerate_summation_terms(nz, ext, ext, q_order_half=10)
    assert isinstance(result, list)
    assert len(result) > 1
    for term in result:
        assert "phase_exp" in term
        assert "tet_args" in term
        assert isinstance(term["phase_exp"], int)
