"""Tests for Weyl symmetry checks."""

from fractions import Fraction
from manifold_index.core.weyl_check import (
    ABVectors,
    WScanEntry,
    check_adjoint_projection,
    check_adjoint_with_w_vector,
    compute_ab_vectors_for_cusp,
    extract_leading_eta_exponents,
    scan_w_vectors,
)


def test_extract_leading():
    assert extract_leading_eta_exponents({(4, 2): 3}, 1) == [Fraction(1)]


def test_adjoint_projection_pass():
    """Synthetic 1-cusp test: 1/2(c_{-1} + c_{+1} - c_{-2} - c_{+2}) = -1.

    Set c_{-1} = c_{+1} = 1, c_{-2} = c_{+2} = 2, so 1/2(1+1-2-2) = -1.
    No Weyl shift (a=0, b=0) so raw eta-0 coefficients are used directly.
    """
    # Each entry: (m_ext=[0], e_ext=[e], result_dict)
    # result keys: (q_half_pow, 2*eta_0_exp), num_hard=1
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

    With a=[1], at m=0 the Weyl shift is eta^{a*e} = eta^e.
    c_e is the eta-0 coeff of eta^e * I(0,e), which equals the eta^{-e}
    coeff of I(0,e).

    Set up so that I(0, e=-1) has eta^1 -> -1 (shift picks eta^0 -> c_{-1}=-1)
    and I(0, e=+1) has eta^{-1} -> -1 (shift picks eta^0 -> c_{+1}=-1),
    with c_{+-2} = 0.  Then 1/2(-1 + -1 - 0 - 0) = -1.
    """
    ab = ABVectors(a=[Fraction(1)], b=[Fraction(0)], num_hard=1)
    entries = [
        ([0], [Fraction(-2)], {}),                    # c_{-2} = 0
        ([0], [Fraction(-1)], {(2, 2): -1}),          # eta_x2=2 -> after shift by -2, eta_x2=0 -> c_{-1}=-1
        ([0], [Fraction(0)],  {(2, 0): -2}),          # no shift -> c_0=-2
        ([0], [Fraction(1)],  {(2, -2): -1}),         # eta_x2=-2 -> after shift by +2, eta_x2=0 -> c_{+1}=-1
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
    # 1/2(1 + 1 - 0 - 0) = 1, not -1
    assert result.is_pass is False
    assert result.projected_value == 1


def test_adjoint_projection_missing_entries():
    """Test that missing e=+-2 entries are flagged."""
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


# ---------------------------------------------------------------------------
# W-vector framework tests
# ---------------------------------------------------------------------------

def test_w_vector_trivial_single_edge():
    """w=(1,) with a=0 is equivalent to the per-edge adjoint check."""
    ab = ABVectors(a=[Fraction(0)], b=[Fraction(0)], num_hard=1)
    entries = [
        ([0], [Fraction(-2)], {(2, 0): 2}),
        ([0], [Fraction(-1)], {(2, 0): 1}),
        ([0], [Fraction(0)],  {(2, 0): 99}),
        ([0], [Fraction(1)],  {(2, 0): 1}),
        ([0], [Fraction(2)],  {(2, 0): 2}),
    ]
    result = check_adjoint_with_w_vector(entries, 1, ab, [1], cusp_idx=0)
    assert result.is_pass is True
    assert result.projected_value == -1


def test_w_vector_with_weyl_shift():
    """w=(1,) with a=1 shifts eta; matches the per-edge shifted check."""
    ab = ABVectors(a=[Fraction(1)], b=[Fraction(0)], num_hard=1)
    entries = [
        ([0], [Fraction(-2)], {}),
        ([0], [Fraction(-1)], {(2, 2): -1}),
        ([0], [Fraction(0)],  {(2, 0): -2}),
        ([0], [Fraction(1)],  {(2, -2): -1}),
        ([0], [Fraction(2)],  {}),
    ]
    result = check_adjoint_with_w_vector(entries, 1, ab, [1], cusp_idx=0)
    assert result.is_pass is True
    assert result.projected_value == -1


def test_w_vector_two_edges_projection():
    """Two hard edges, v=(1,1): project onto combined variable."""
    ab = ABVectors(
        a=[Fraction(0), Fraction(0)],
        b=[Fraction(0), Fraction(0)],
        num_hard=2,
    )
    entries = [
        ([0], [Fraction(-2)], {(2, 0, 0): 2}),
        ([0], [Fraction(-1)], {(2, 2, -2): 1, (2, 0, 0): 5}),
        ([0], [Fraction(0)],  {(2, 0, 0): 99}),
        ([0], [Fraction(1)],  {(2, -2, 2): 1, (2, 0, 0): 5}),
        ([0], [Fraction(2)],  {(2, 0, 0): 2}),
    ]
    result = check_adjoint_with_w_vector(entries, 2, ab, [1, 1], cusp_idx=0)
    # combined_x2 = key[1]+key[2].
    # e=-1: (2,2,-2)->0 match, (2,0,0)->0 match -> 1+5=6.  c_{-1}=6
    # e=+1: (2,-2,2)->0 match, (2,0,0)->0 match -> 1+5=6.  c_{+1}=6
    # e=+-2: (2,0,0)->0 match -> 2.  c_{+-2}=2
    # 1/2(6+6-2-2)=4
    assert result.is_pass is False
    assert result.projected_value == 4


def test_w_vector_with_shift_two_edges():
    """w=(1,0): turns off edge 1, uses edge 0 only."""
    ab = ABVectors(
        a=[Fraction(1), Fraction(-1, 2)],
        b=[Fraction(0), Fraction(0)],
        num_hard=2,
    )
    # v=(1,0): combined_x2 = key[1], target_x2 = -2*1*e
    entries = [
        ([0], [Fraction(-2)], {(2, 4, 0): 3, (2, 4, 5): 7}),
        ([0], [Fraction(-1)], {(2, 2, 0): -1, (2, 2, 3): 4}),
        ([0], [Fraction(0)],  {(2, 0, 0): 99}),
        ([0], [Fraction(1)],  {(2, -2, 0): 3}),
        ([0], [Fraction(2)],  {(2, -4, 0): 10}),
    ]
    result = check_adjoint_with_w_vector(entries, 2, ab, [1, 0], cusp_idx=0)
    # c_{-2}=3+7=10, c_{-1}=-1+4=3, c_{+1}=3, c_{+2}=10
    # 1/2(3+3-10-10) = -7
    assert result.is_pass is False
    assert result.projected_value == -7


def test_scan_v_vectors_single_edge():
    """Scan with 1 hard edge: canonical v in {(1,), (2,), (3,)}."""
    ab = ABVectors(a=[Fraction(0)], b=[Fraction(0)], num_hard=1)
    entries = [
        ([0], [Fraction(-2)], {(2, 0): 2}),
        ([0], [Fraction(-1)], {(2, 0): 1}),
        ([0], [Fraction(0)],  {(2, 0): 99}),
        ([0], [Fraction(1)],  {(2, 0): 1}),
        ([0], [Fraction(2)],  {(2, 0): 2}),
    ]
    result = scan_w_vectors(entries, 1, ab, max_coeff=3)
    assert len(result.entries) == 3
    ws = [e.w for e in result.entries]
    assert (1,) in ws
    assert (2,) in ws
    assert (3,) in ws
    # All have a_eff=0 -> target=0. All keys have eta_x2=0, so all match
    assert len(result.passing) == 3


def test_scan_v_vectors_canonicalisation():
    """Negative v is excluded by sign canonicalisation."""
    ab = ABVectors(
        a=[Fraction(0), Fraction(0)],
        b=[Fraction(0), Fraction(0)],
        num_hard=2,
    )
    entries = [
        ([0], [Fraction(-2)], {(2, 0, 0): 2}),
        ([0], [Fraction(-1)], {(2, 0, 0): 1}),
        ([0], [Fraction(0)],  {(2, 0, 0): 99}),
        ([0], [Fraction(1)],  {(2, 0, 0): 1}),
        ([0], [Fraction(2)],  {(2, 0, 0): 2}),
    ]
    result = scan_w_vectors(entries, 2, ab, max_coeff=1)
    ws = [e.w for e in result.entries]
    # (1,-1), (1,0), (1,1), (0,1) -> 4 canonical vectors
    assert len(ws) == 4
    assert (1, 0) in ws
    assert (0, 1) in ws
    assert (1, 1) in ws
    assert (1, -1) in ws
    assert (-1, 0) not in ws
    assert (0, -1) not in ws


def test_scan_v_vectors_skip_incompatible():
    """skip_incompatible=True sets adjoint=None for non-integer a_eff."""
    ab = ABVectors(a=[Fraction(1, 2)], b=[Fraction(0)], num_hard=1)
    entries = [
        ([0], [Fraction(-2)], {(2, 0): 2}),
        ([0], [Fraction(-1)], {(2, 0): 1}),
        ([0], [Fraction(0)],  {(2, 0): 99}),
        ([0], [Fraction(1)],  {(2, 0): 1}),
        ([0], [Fraction(2)],  {(2, 0): 2}),
    ]
    result = scan_w_vectors(
        entries, 1, ab, max_coeff=2, skip_incompatible=True,
    )
    v1 = next(e for e in result.entries if e.w == (1,))
    v2 = next(e for e in result.entries if e.w == (2,))
    assert v1.a_eff_is_integer is False
    assert v1.adjoint is None
    assert v2.a_eff_is_integer is True
    assert v2.adjoint is not None


def test_scan_zero_hard_edges():
    """With 0 hard edges, scan returns empty."""
    ab = ABVectors(a=[], b=[], num_hard=0)
    result = scan_w_vectors([], 0, ab, max_coeff=2)
    assert result.entries == []
    assert result.passing == []
