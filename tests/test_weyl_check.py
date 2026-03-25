"""Tests for Weyl symmetry checks."""

from fractions import Fraction
from manifold_index.core.weyl_check import (
    check_adjoint_character, compute_ab_vectors_for_cusp,
    extract_leading_eta_exponents,
)


def test_extract_leading():
    assert extract_leading_eta_exponents({(4, 2): 3}, 1) == [Fraction(1)]


def test_adjoint_character():
    result = {(2, -2): 1, (2, 0): 1, (2, 2): 1}
    assert check_adjoint_character(result, [Fraction(0)], 1, 0) is True


def test_m004_ab_vectors(nz_m004):
    ab = compute_ab_vectors_for_cusp(nz_m004, cusp_idx=0, q_order_half=20)
    assert ab is not None
    assert ab.a == [Fraction(1, 2)]
    assert ab.b == [Fraction(-1, 2)]
    assert ab.is_valid is False
