"""Tests for Weyl symmetry checks."""
from fractions import Fraction
import pytest
from manifold_index.core.weyl_check import (
    ABVectors, _eta_center_at_leading_q, check_adjoint_projection,
    check_adjoint_with_w_vector, scan_w_vectors,
)


def test_eta_center_at_leading_q():
    # key (4, 2) → qq=4, η_0 doubled-exp=2 → true exp=1; coeff=3; centre=1
    assert _eta_center_at_leading_q({(4, 2): 3}, 1) == [Fraction(1)]


def test_adjoint_projection_pass():
    """1/2(c_{-1} + c_{+1} - c_{-2} - c_{+2}) = -1."""
    entries = [
        ([0], [Fraction(-2)], {(2, 0): 2}),
        ([0], [Fraction(-1)], {(2, 0): 1}),
        ([0], [Fraction(0)],  {(2, 0): 99}),
        ([0], [Fraction(1)],  {(2, 0): 1}),
        ([0], [Fraction(2)],  {(2, 0): 2}),
    ]
    result = check_adjoint_projection(entries, num_hard=1, cusp_idx=0)
    assert result.is_pass is True
    assert result.projected_value == -1


def test_adjoint_projection_fail():
    entries = [
        ([0], [Fraction(-2)], {(2, 0): 0}),
        ([0], [Fraction(-1)], {(2, 0): 1}),
        ([0], [Fraction(0)],  {(2, 0): 0}),
        ([0], [Fraction(1)],  {(2, 0): 1}),
        ([0], [Fraction(2)],  {(2, 0): 0}),
    ]
    result = check_adjoint_projection(entries, num_hard=1, cusp_idx=0)
    assert result.is_pass is False
    assert result.projected_value == 1


def test_m004_ab_vectors(nz_m004):
    from manifold_index.core.weyl_check import compute_ab_vectors_for_cusp
    ab = compute_ab_vectors_for_cusp(nz_m004, cusp_idx=0, q_order_half=20)
    assert ab is not None
    assert ab.a == [Fraction(1, 2)]
    assert ab.b == [Fraction(-1, 2)]
