"""tests/test_weyl_check.py — Smoke tests for Weyl symmetry checks."""

from fractions import Fraction
import pytest

from manifold_index.core.weyl_check import (
    ABVectors, check_adjoint_character, check_weyl_symmetry,
    compute_ab_vectors, extract_leading_eta_exponents, run_weyl_checks,
)
from manifold_index.core.refined_index import RefinedIndexResult


def test_extract_leading_single_term():
    result = {(4, 2): 3}
    eta = extract_leading_eta_exponents(result, 1)
    assert eta == [Fraction(1)]


def test_compute_ab_combined():
    """b=-1/2, a=-1 from meridian+longitude pairs."""
    def _e(m, e, res):
        return ([m], [Fraction(e)], res)

    entries = [
        _e(+2, 0, {(0, 2): 1}),
        _e(-2, 0, {(0, -2): 1}),
        _e(0, +1, {(0, 1): 1}),
        _e(0, -1, {(0, -1): 1}),
    ]
    ab = compute_ab_vectors(entries, 1)
    assert ab is not None
    assert ab.a == [Fraction(-1)]
    assert ab.b == [Fraction(-1, 2)]
    assert ab.is_valid


def test_weyl_symmetry_adjoint():
    """η^{-1} + 1 + η at q^0 is Weyl-symmetric."""
    result = {(0, -2): 1, (0, 0): 1, (0, 2): 1}
    ab = ABVectors(a=[Fraction(0)], b=[Fraction(0)], num_hard=1)
    entries = [([0], [Fraction(0)], result)]
    res = check_weyl_symmetry(entries, 1, ab)
    assert res[((0,), (Fraction(0),))] is True


def test_adjoint_character_exact():
    result = {(2, -2): 1, (2, 0): 1, (2, 2): 1}
    leading = [Fraction(0)]
    assert check_adjoint_character(result, leading, 1, 0) is True


def test_m004_ab_vectors():
    """m004 integration: (a, b) should be valid."""
    pytest.importorskip("snappy")
    from manifold_index.core.manifold import load_manifold
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import build_neumann_zagier
    from manifold_index.core.refined_index import compute_refined_index

    data = load_manifold("m004")
    easy = find_easy_edges(data)
    nz = build_neumann_zagier(data, easy)
    num_hard = nz.num_hard
    assert num_hard >= 1

    entries = []
    for m in [-2, -1, 0, 1, 2]:
        for e in [-1, 0, 1]:
            result = compute_refined_index(nz, [m], [Fraction(e)], q_order_half=20)
            entries.append(([m], [Fraction(e)], result))

    ab = compute_ab_vectors(entries, num_hard)
    assert ab is not None
    assert ab.is_valid
    assert ab.b_is_half_integer == [True] * num_hard
    assert ab.a_is_integer == [True] * num_hard
