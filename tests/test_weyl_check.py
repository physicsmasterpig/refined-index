"""Tests for Weyl symmetry checks."""

from fractions import Fraction
from manifold_index.core.weyl_check import (
    ABVectors,
    check_adjoint_projection, compute_ab_vectors_for_cusp,
    extract_leading_eta_exponents,
)


def test_extract_leading():
    assert extract_leading_eta_exponents({(4, 2): 3}, 1) == [Fraction(1)]


def test_adjoint_projection_pass():
    """Synthetic 1-cusp test: ½(c_{-1} + c_{+1} − c_{-2} − c_{+2}) = −1.

    Set c_{-1} = c_{+1} = 1, c_{-2} = c_{+2} = 2, so ½(1+1−2−2) = −1. ✓
    No Weyl shift (a=0, b=0) so raw η⁰ coefficients are used directly.
    """
    # Each entry: (m_ext=[0], e_ext=[e], result_dict)
    # result keys: (q_half_pow, 2*η_0_exp), num_hard=1
    entries = [
        ([0], [Fraction(-2)], {(2, 0): 2}),   # c_{-2} = 2
        ([0], [Fraction(-1)], {(2, 0): 1}),   # c_{-1} = 1
        ([0], [Fraction(0)],  {(2, 0): 99}),  # c_0 = doesn't matter
        ([0], [Fraction(1)],  {(2, 0): 1}),   # c_{+1} = 1
        ([0], [Fraction(2)],  {(2, 0): 2}),   # c_{+2} = 2
    ]
    result = check_adjoint_projection(entries, num_hard=1, cusp_idx=0)
    assert result.is_pass is True
    assert result.projected_value == -1


def test_adjoint_projection_with_weyl_shift():
    """Test with Weyl shift a=1.

    With a=[1], at m=0 the Weyl shift is η^{a·e} = η^e.
    c_e is the η⁰ coeff of η^e · I(0,e), which equals the η^{-e}
    coeff of I(0,e).

    Set up so that I(0, e=-1) has η¹ → -1 (shift picks η⁰ → c_{-1}=-1)
    and I(0, e=+1) has η⁻¹ → -1 (shift picks η⁰ → c_{+1}=-1),
    with c_{±2} = 0.  Then ½(-1 + -1 - 0 - 0) = -1. ✓
    """
    ab = ABVectors(a=[Fraction(1)], b=[Fraction(0)], num_hard=1)
    entries = [
        ([0], [Fraction(-2)], {}),                    # c_{-2} = 0
        ([0], [Fraction(-1)], {(2, 2): -1}),          # η_x2=2 → after shift by -2, η_x2=0 → c_{-1}=-1
        ([0], [Fraction(0)],  {(2, 0): -2}),          # no shift → c_0=-2
        ([0], [Fraction(1)],  {(2, -2): -1}),         # η_x2=-2 → after shift by +2, η_x2=0 → c_{+1}=-1
        ([0], [Fraction(2)],  {}),                    # c_{+2} = 0
    ]
    result = check_adjoint_projection(entries, num_hard=1, ab=ab, cusp_idx=0)
    assert result.is_pass is True
    assert result.projected_value == -1


def test_adjoint_projection_fail():
    """Synthetic test with wrong projected value."""
    entries = [
        ([0], [Fraction(-2)], {(2, 0): 0}),
        ([0], [Fraction(-1)], {(2, 0): 1}),
        ([0], [Fraction(0)],  {(2, 0): 0}),
        ([0], [Fraction(1)],  {(2, 0): 1}),
        ([0], [Fraction(2)],  {(2, 0): 0}),
    ]
    result = check_adjoint_projection(entries, num_hard=1, cusp_idx=0)
    # ½(1 + 1 − 0 − 0) = 1, not −1
    assert result.is_pass is False
    assert result.projected_value == 1


def test_adjoint_projection_missing_entries():
    """Test that missing e=±2 entries are flagged."""
    entries = [
        ([0], [Fraction(-1)], {(2, 0): 1}),
        ([0], [Fraction(0)],  {(2, 0): 0}),
        ([0], [Fraction(1)],  {(2, 0): 1}),
    ]
    result = check_adjoint_projection(entries, num_hard=1, cusp_idx=0)
    assert result.is_pass is False
    assert result.projected_value is None
    assert len(result.missing_e) > 0


def test_m004_ab_vectors(nz_m004):
    ab = compute_ab_vectors_for_cusp(nz_m004, cusp_idx=0, q_order_half=20)
    assert ab is not None
    assert ab.a == [Fraction(1, 2)]
    assert ab.b == [Fraction(-1, 2)]
    assert ab.is_valid is False
