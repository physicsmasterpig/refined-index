"""
tests/test_weyl_check.py — Tests for core/weyl_check.py.

Test classes
------------
TestExtractLeadingEtaExponents
    Unit tests for the leading η-exponent extractor.  No SnaPy required.

TestComputeAbVectors
    Unit tests for (a, b) vector computation from synthetic data.
    No SnaPy required.

TestWeylSymmetry
    Unit tests for check_weyl_symmetry.  No SnaPy required.

TestAdjointCharacter
    Unit tests for check_adjoint_character.  No SnaPy required.

TestRunWeylChecksSnaPy
    Integration tests on real manifolds that require SnaPy.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from manifold_index.core.weyl_check import (
    ABVectors,
    WeylCheckResult,
    check_adjoint_character,
    check_weyl_symmetry,
    compute_ab_vectors,
    extract_leading_eta_exponents,
    run_weyl_checks,
    strip_weyl_monomial,
)
from manifold_index.core.refined_index import RefinedIndexResult


# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

def _has_snapy() -> bool:
    try:
        import snappy  # noqa: F401
        return True
    except ImportError:
        return False


skip_no_snapy = pytest.mark.skipif(
    not _has_snapy(), reason="SnaPy not installed"
)


# ===========================================================================
# TestExtractLeadingEtaExponents
# ===========================================================================

class TestExtractLeadingEtaExponents:
    """extract_leading_eta_exponents: lowest q-level, component-wise min η."""

    def test_empty(self):
        assert extract_leading_eta_exponents({}, 1) is None

    def test_all_zero_coefficients(self):
        result: RefinedIndexResult = {(0, 0): 0, (2, 2): 0}
        assert extract_leading_eta_exponents(result, 1) is None

    def test_single_term(self):
        # key=(q_half=4, 2*η_0=2): q^2 * η_0^1
        result: RefinedIndexResult = {(4, 2): 3}
        eta = extract_leading_eta_exponents(result, 1)
        assert eta == [Fraction(1)]

    def test_picks_min_q_level(self):
        # Terms at q^0 and q^1 — should pick q^0 term's η-exponent
        result: RefinedIndexResult = {
            (0, 0): 1,    # q^0 * η^0
            (2, 4): 2,    # q^1 * η^2
        }
        eta = extract_leading_eta_exponents(result, 1)
        assert eta == [Fraction(0)]

    def test_componentwise_min_at_leading_q(self):
        # Two terms at q^0: η^{+1} and η^{-1}; minimum is -1/2 = -1 in doubled
        result: RefinedIndexResult = {
            (0, 2): 1,    # q^0 * η^1
            (0, -2): 1,   # q^0 * η^{-1}
        }
        eta = extract_leading_eta_exponents(result, 1)
        # component-wise min = -1
        assert eta == [Fraction(-1)]

    def test_two_hard_edges(self):
        # keys: (q_half, 2*η_0, 2*η_1)
        result: RefinedIndexResult = {
            (0, 2, -4): 1,   # η_0^1 η_1^{-2}
            (0, -2, 0): 1,   # η_0^{-1} η_1^0
        }
        eta = extract_leading_eta_exponents(result, 2)
        assert eta == [Fraction(-1), Fraction(-2)]

    def test_half_integer_exponent(self):
        # 2*η_exp = 3 → actual exp = 3/2
        result: RefinedIndexResult = {(2, 3): 1}
        eta = extract_leading_eta_exponents(result, 1)
        assert eta == [Fraction(3, 2)]

    def test_zero_eta_at_leading_q(self):
        result: RefinedIndexResult = {(0, 0): 1, (0, 2): -1}
        eta = extract_leading_eta_exponents(result, 1)
        assert eta == [Fraction(0)]  # min(-0, 1) = 0 in doubled → 0


# ===========================================================================
# TestComputeAbVectors
# ===========================================================================

class TestComputeAbVectors:
    """compute_ab_vectors on synthetic refined index data."""

    def _make_entry(self, m: int, e: int | Fraction, result: RefinedIndexResult):
        """Helper: single-cusp entry."""
        return ([m], [Fraction(e)], result)

    # -----------------------------------------------------------------------
    # Trivial / degenerate cases
    # -----------------------------------------------------------------------

    def test_no_hard_edges(self):
        entries = [self._make_entry(1, 0, {(2,): 1})]
        ab = compute_ab_vectors(entries, 0)
        assert ab is not None
        assert ab.a == []
        assert ab.b == []
        assert ab.num_hard == 0
        assert ab.is_valid

    def test_none_if_no_pairs(self):
        # Only (1, 0) entry but no (-1, 0) — cannot determine b
        entries = [self._make_entry(1, 0, {(0, 0): 1})]
        ab = compute_ab_vectors(entries, 1)
        # Result may be None or have warnings about missing data
        if ab is not None:
            assert ab.warnings  # should warn about missing data

    # -----------------------------------------------------------------------
    # b extraction from meridian pairs
    # -----------------------------------------------------------------------

    def test_b_from_meridian_pair_zero(self):
        """b = 0 when I(+m,0) and I(-m,0) have the same leading η-power."""
        entries = [
            self._make_entry(+2, 0, {(0, 0): 1}),   # g(+2,0) = 0
            self._make_entry(-2, 0, {(0, 0): 1}),   # g(-2,0) = 0
        ]
        ab = compute_ab_vectors(entries, 1)
        assert ab is not None
        assert ab.b == [Fraction(0)]
        assert ab.b_is_half_integer == [True]

    def test_b_from_meridian_pair_half(self):
        """b = -1/2: multiply I(+2,0) by η^{b·m} = η^{-1} to get the Weyl-manifest f."""
        # centre(+2,0) = +1, centre(-2,0) = -1 → diff = 2
        # b = −diff / (2*m) = −2 / 4 = −1/2
        entries = [
            self._make_entry(+2, 0, {(0, 2): 1}),   # η^1
            self._make_entry(-2, 0, {(0, -2): 1}),  # η^{-1}
        ]
        ab = compute_ab_vectors(entries, 1)
        assert ab is not None
        assert ab.b == [Fraction(-1, 2)]
        assert ab.b_is_half_integer == [True]

    def test_b_from_meridian_pair_integer(self):
        """b = -1: multiply I(+2,0) by η^{b·m} = η^{-2} to get f."""
        # centre(+2,0) = +2, centre(-2,0) = -2 → diff = 4
        # b = −4 / (2*2) = −1
        entries = [
            self._make_entry(+2, 0, {(0, 4): 1}),   # η^2
            self._make_entry(-2, 0, {(0, -4): 1}),  # η^{-2}
        ]
        ab = compute_ab_vectors(entries, 1)
        assert ab is not None
        assert ab.b == [Fraction(-1)]
        assert ab.b_is_half_integer == [True]
        assert ab.a_is_integer == [True]   # a defaulted to 0 → integer

    def test_b_negative(self):
        """b = +1/2 (centre(+2,0) = -1, so multiplier is +1/2)."""
        entries = [
            self._make_entry(+2, 0, {(0, -2): 1}),  # η^{-1}
            self._make_entry(-2, 0, {(0, 2): 1}),   # η^{+1}
        ]
        ab = compute_ab_vectors(entries, 1)
        assert ab is not None
        assert ab.b == [Fraction(1, 2)]

    def test_b_from_m1_pair(self):
        """b = -1/2 using m=±1 pair: multiply I(+1,0) by η^{b·1} = η^{-1/2} to get f."""
        # centre(+1,0) = 1/2 (key=1), centre(-1,0) = -1/2 → diff = 1
        # b = −1 / (2*1) = −1/2
        entries = [
            self._make_entry(+1, 0, {(0, 1): 1}),   # η^{1/2} — half-integer
            self._make_entry(-1, 0, {(0, -1): 1}),  # η^{-1/2}
        ]
        ab = compute_ab_vectors(entries, 1)
        assert ab is not None
        assert ab.b == [Fraction(-1, 2)]

    # -----------------------------------------------------------------------
    # a extraction from longitude pairs
    # -----------------------------------------------------------------------

    def test_a_from_longitude_pair_zero(self):
        """a = 0 when I(0,+e) and I(0,-e) have the same leading η-power."""
        entries = [
            self._make_entry(0, +1, {(0, 0): 1}),
            self._make_entry(0, -1, {(0, 0): 1}),
        ]
        ab = compute_ab_vectors(entries, 1)
        assert ab is not None
        assert ab.a == [Fraction(0)]
        assert ab.a_is_integer == [True]

    def test_a_from_longitude_pair_integer(self):
        """a = -1: multiply I(0,+1) by η^{a·e} = η^{-1} to get the Weyl-manifest f."""
        # centre(0,+1) = 1/2  (doubled key=1 → η^{1/2})
        # centre(0,-1) = -1/2
        # a = −(1/2 − (−1/2)) / 1 = −1
        entries = [
            self._make_entry(0, +1, {(0, 1): 1}),   # 2·η_exp=1 → η^{1/2}, centre=1/2
            self._make_entry(0, -1, {(0, -1): 1}),  # 2·η_exp=-1 → η^{-1/2}, centre=-1/2
        ]
        ab = compute_ab_vectors(entries, 1)
        assert ab is not None
        assert ab.a == [Fraction(-1)]
        assert ab.a_is_integer == [True]

    def test_a_from_longitude_pair_half_fails_validity(self):
        """a = -1/2 violates the integer constraint → is_valid False."""
        # centre(0,+2) = 1/2 (from key (0,1)), centre(0,-2) = -1/2
        # a = −(1/2 − (−1/2)) / 2 = −1/2  ← non-integer
        entries = [
            self._make_entry(0, +2, {(0, 1): 1}),   # 2·η_exp=1 → η^{1/2}, centre=1/2
            self._make_entry(0, -2, {(0, -1): 1}),  # 2·η_exp=-1 → η^{-1/2}, centre=-1/2
        ]
        ab = compute_ab_vectors(entries, 1)
        assert ab is not None
        assert ab.a == [Fraction(-1, 2)]
        assert ab.a_is_integer == [False]
        assert not ab.is_valid

    # -----------------------------------------------------------------------
    # Combined (a, b)
    # -----------------------------------------------------------------------

    def test_combined_ab(self):
        """b=-1/2, a=-1 from a minimal set of 4 entries."""
        entries = [
            self._make_entry(+2, 0, {(0, 2): 1}),   # centre_b(+2,0) = 1
            self._make_entry(-2, 0, {(0, -2): 1}),  # centre_b(-2,0) = -1; diff=2 → b=-1/2
            self._make_entry(0, +1, {(0, 1): 1}),   # centre_a(0,+1) = 1/2
            self._make_entry(0, -1, {(0, -1): 1}),  # centre_a(0,-1) = -1/2; diff=1 → a=-1
        ]
        ab = compute_ab_vectors(entries, 1)
        assert ab is not None
        assert ab.a == [Fraction(-1)]
        assert ab.b == [Fraction(-1, 2)]
        assert ab.is_valid

    def test_is_valid_str(self):
        """ABVectors.__str__ runs without error."""
        ab = ABVectors(a=[Fraction(1)], b=[Fraction(1, 2)], num_hard=1)
        s = str(ab)
        assert "a = (1)" in s
        assert "b = (1/2)" in s
        assert "✓" in s

    def test_is_valid_false_str(self):
        """ABVectors with non-integer a shows ✗."""
        ab = ABVectors(a=[Fraction(1, 3)], b=[Fraction(1, 2)], num_hard=1)
        assert not ab.is_valid
        assert "✗" in str(ab)


# ===========================================================================
# TestWeylSymmetry
# ===========================================================================

class TestWeylSymmetry:
    """check_weyl_symmetry on synthetic data."""

    def _entry(self, m, e, result):
        return ([m], [Fraction(e)], result)

    def test_symmetric_trivial(self):
        """Single term at η^0 is trivially symmetric."""
        ab = ABVectors(a=[Fraction(0)], b=[Fraction(0)], num_hard=1)
        entries = [self._entry(0, 0, {(0, 0): 1})]
        res = check_weyl_symmetry(entries, 1, ab)
        assert res[((0,), (Fraction(0),))] is True

    def test_adjoint_at_q0_is_symmetric(self):
        """η^{-1} + 1 + η at q^0 is Weyl-symmetric."""
        result: RefinedIndexResult = {(0, -2): 1, (0, 0): 1, (0, 2): 1}
        ab = ABVectors(a=[Fraction(0)], b=[Fraction(0)], num_hard=1)
        entries = [self._entry(0, 0, result)]
        res = check_weyl_symmetry(entries, 1, ab)
        assert res[((0,), (Fraction(0),))] is True

    def test_asymmetric_fails(self):
        """η^1 only (no η^{-1}) is not symmetric."""
        result: RefinedIndexResult = {(0, 2): 1}
        ab = ABVectors(a=[Fraction(0)], b=[Fraction(0)], num_hard=1)
        entries = [self._entry(0, 0, result)]
        res = check_weyl_symmetry(entries, 1, ab)
        assert res[((0,), (Fraction(0),))] is False

    def test_with_nontrivial_shift(self):
        """I(2,0) with b=1/2: after stripping η^1, should be Weyl-symmetric."""
        # I(2,0) = η^1 * (η^{-1} + 1 + η) = η^0 + η^1 + η^2
        # Keys (doubled): (0,-2), (0,0), (0,2) → shifted by b*m = 1/2*2 = 1 → x2=2
        # After shift: (0,-2-2), (0,0-2), (0,2-2) = (0,-4),(0,-2),(0,0)
        # Wait, let me recalculate:
        # b=-1/2, m=2; shift_x2 = 2*(-1/2)*2 = -2; new_key = key + (-2) = key - 2
        # I(2,0) has keys: (0, 0), (0, 2), (0, 4) = η^0, η^1, η^2
        # After adding shift of -2: keys become (0,-2),(0,0),(0,2) → symmetric ✓
        result: RefinedIndexResult = {(0, 0): 1, (0, 2): 1, (0, 4): 1}
        ab = ABVectors(a=[Fraction(0)], b=[Fraction(-1, 2)], num_hard=1)
        entries = [([2], [Fraction(0)], result)]
        res = check_weyl_symmetry(entries, 1, ab)
        assert res[((2,), (Fraction(0),))] is True


# ===========================================================================
# TestStripWeylMonomial
# ===========================================================================

class TestStripWeylMonomial:
    """strip_weyl_monomial: factor out the Weyl η-monomial from a single entry."""

    def test_trivial_shift_identity(self):
        """b=0, a=0 → centre=[0], stripped == original."""
        result: RefinedIndexResult = {(0, -2): 1, (0, 0): 1, (0, 2): 1}
        ab = ABVectors(a=[Fraction(0)], b=[Fraction(0)], num_hard=1)
        centre, stripped = strip_weyl_monomial(result, [0], [Fraction(0)], ab, 1)
        assert centre == [Fraction(0)]
        assert stripped == result

    def test_shift_by_integer(self):
        """b=-1/2, m=2 → centre=[1]; keys shift by -2 (doubled)."""
        # I(2,0) = η^0 + η^1 + η^2  (doubled keys: 0,2,4)
        result: RefinedIndexResult = {(0, 0): 1, (0, 2): 1, (0, 4): 1}
        ab = ABVectors(a=[Fraction(0)], b=[Fraction(-1, 2)], num_hard=1)
        centre, stripped = strip_weyl_monomial(result, [2], [Fraction(0)], ab, 1)
        assert centre == [Fraction(1)]
        # After multiplying by η^1: η^{-1} + η^0 + η^1 (doubled: -2, 0, 2)
        assert stripped == {(0, -2): 1, (0, 0): 1, (0, 2): 1}

    def test_stripped_is_weyl_symmetric(self):
        """Stripped series has the reflection property coeff(k) == coeff(-k)."""
        result: RefinedIndexResult = {(0, 0): 1, (0, 2): 1, (0, 4): 1}
        ab = ABVectors(a=[Fraction(0)], b=[Fraction(-1, 2)], num_hard=1)
        _, stripped = strip_weyl_monomial(result, [2], [Fraction(0)], ab, 1)
        for key, coeff in stripped.items():
            reflect = (key[0],) + tuple(-key[1 + j] for j in range(1))
            assert stripped.get(reflect, 0) == coeff

    def test_shift_by_a_vector(self):
        """a=-2, e=1 → centre=[1]; same shift as above but via e-channel."""
        # a=-2, e=1 → shift_x2 = a*e = -2*1 = -2; new_key = key + (-2) → centre = 1
        result: RefinedIndexResult = {(2, 0): 1, (2, 2): 1, (2, 4): 1}
        ab = ABVectors(a=[Fraction(-2)], b=[Fraction(0)], num_hard=1)
        centre, stripped = strip_weyl_monomial(result, [0], [Fraction(1)], ab, 1)
        assert centre == [Fraction(1)]
        assert stripped == {(2, -2): 1, (2, 0): 1, (2, 2): 1}

    def test_zero_result(self):
        """Empty series → empty stripped, centre = [0] (or computed shift)."""
        result: RefinedIndexResult = {}
        ab = ABVectors(a=[Fraction(0)], b=[Fraction(-1, 2)], num_hard=1)
        centre, stripped = strip_weyl_monomial(result, [2], [Fraction(0)], ab, 1)
        assert centre == [Fraction(1)]
        assert stripped == {}

    def test_half_integer_centre(self):
        """b=-1/2, m=1 → centre=[1/2]; shift_x2=-1."""
        result: RefinedIndexResult = {(0, 1): 3}
        ab = ABVectors(a=[Fraction(0)], b=[Fraction(-1, 2)], num_hard=1)
        centre, stripped = strip_weyl_monomial(result, [1], [Fraction(0)], ab, 1)
        assert centre == [Fraction(1, 2)]
        assert stripped == {(0, 0): 3}

    def test_two_hard_edges(self):
        """num_hard=2: each edge shifted independently."""
        # b=[-1/2, 0], a=[0, -2], m=[2], e=[1]
        # shift_x2[0] = 2*(-1/2)*2 + 0*1 = -2, shift_x2[1] = 2*0*2 + (-2)*1 = -2
        # new_key = (0, 4+(-2), 2+(-2)) = (0, 2, 0); centre=[1, 1]
        result: RefinedIndexResult = {(0, 4, 2): 5}
        ab = ABVectors(a=[Fraction(0), Fraction(-2)], b=[Fraction(-1, 2), Fraction(0)], num_hard=2)
        centre, stripped = strip_weyl_monomial(result, [2], [Fraction(1)], ab, 2)
        assert centre == [Fraction(1), Fraction(1)]
        assert stripped == {(0, 2, 0): 5}


# ===========================================================================
# TestAdjointCharacter
# ===========================================================================

class TestAdjointCharacter:
    """check_adjoint_character on synthetic data."""

    def test_exact_adjoint(self):
        """q^1 coefficient is exactly η^{-1} + 1 + η → True."""
        result: RefinedIndexResult = {
            (2, -2): 1,   # q^1 * η^{-1}
            (2, 0): 1,    # q^1 * η^0
            (2, 2): 1,    # q^1 * η^1
        }
        leading = [Fraction(0)]
        assert check_adjoint_character(result, leading, 1, 0) is True

    def test_exact_adjoint_with_q0_term(self):
        """Other q-powers don't affect the check."""
        result: RefinedIndexResult = {
            (0, 0): 1,    # q^0 term (ignored)
            (2, -2): 1,
            (2, 0): 1,
            (2, 2): 1,
        }
        leading = [Fraction(0)]
        assert check_adjoint_character(result, leading, 1, 0) is True

    def test_adjoint_missing_middle(self):
        """η^{-1} + η (no singlet) → False."""
        result: RefinedIndexResult = {
            (2, -2): 1,
            (2, 2): 1,
        }
        leading = [Fraction(0)]
        assert check_adjoint_character(result, leading, 1, 0) is False

    def test_adjoint_wrong_coefficients(self):
        """η^{-1} + 2*η^0 + η is not adjoint-proportional → False."""
        result: RefinedIndexResult = {
            (2, -2): 1,
            (2, 0): 2,
            (2, 2): 1,
        }
        leading = [Fraction(0)]
        assert check_adjoint_character(result, leading, 1, 0) is False

    def test_adjoint_scaled(self):
        """2*(η^{-1} + 1 + η) is still adjoint-proportional → True."""
        result: RefinedIndexResult = {
            (2, -2): 2,
            (2, 0): 2,
            (2, 2): 2,
        }
        leading = [Fraction(0)]
        assert check_adjoint_character(result, leading, 1, 0) is True

    def test_adjoint_no_q1_terms(self):
        """No q^1 terms → False."""
        result: RefinedIndexResult = {(0, 0): 1}
        leading = [Fraction(0)]
        assert check_adjoint_character(result, leading, 1, 0) is False

    def test_adjoint_with_shift(self):
        """After stripping leading η^1: keys at q^1 are η^0, η^1, η^2 → adjoint."""
        result: RefinedIndexResult = {
            (2, 0): 1,    # stripped: η^0-1=-1 → η^{-1}
            (2, 2): 1,    # stripped: η^1-1= 0 → η^0
            (2, 4): 1,    # stripped: η^2-1= 1 → η^1
        }
        leading = [Fraction(1)]  # leading η-power = 1 (doubled: 2)
        assert check_adjoint_character(result, leading, 1, 0) is True

    def test_empty_result(self):
        assert check_adjoint_character({}, [Fraction(0)], 1, 0) is False

    def test_none_leading(self):
        assert check_adjoint_character({(0, 0): 1}, None, 1, 0) is False  # type: ignore[arg-type]


# ===========================================================================
# TestRunWeylChecksSnaPy — integration tests on real manifolds
# ===========================================================================

@skip_no_snapy
class TestRunWeylChecksSnaPy:
    """Integration tests using real manifold data from SnaPy."""

    def _compute_entries(self, name: str, m_range, e_range, q_order: int = 20):
        """Compute multi-point refined index entries for a manifold."""
        from manifold_index.core.manifold import load_manifold
        from manifold_index.core.neumann_zagier import build_neumann_zagier
        from manifold_index.core.phase_space import find_easy_edges
        from manifold_index.core.refined_index import compute_refined_index

        data = load_manifold(name)
        easy = find_easy_edges(data)
        nz = build_neumann_zagier(data, easy)
        num_hard = nz.num_hard

        entries = []
        for m in m_range:
            for e in e_range:
                result = compute_refined_index(nz, [m], [Fraction(e)], q_order)
                entries.append(([m], [Fraction(e)], result))

        return entries, num_hard

    # -----------------------------------------------------------------------
    # m004 — figure-eight knot complement, 1 hard edge
    # -----------------------------------------------------------------------

    def test_m004_ab_vectors(self):
        """m004 should have well-defined (a, b) with correct integrality."""
        entries, num_hard = self._compute_entries(
            "m004",
            m_range=[-2, -1, 0, 1, 2],
            e_range=[-1, 0, 1],
        )
        assert num_hard >= 1

        ab = compute_ab_vectors(entries, num_hard)
        assert ab is not None, "Should be able to compute (a, b) for m004"
        # b must be half-integer, a must be integer
        assert ab.b_is_half_integer == [True] * num_hard, f"b not half-integer: {ab.b}"
        assert ab.a_is_integer == [True] * num_hard, f"a not integer: {ab.a}"
        assert ab.is_valid

    def test_m004_weyl_check_result(self):
        """run_weyl_checks should return a WeylCheckResult for m004."""
        entries, num_hard = self._compute_entries(
            "m004",
            m_range=[-2, 0, 2],
            e_range=[-1, 0, 1],
        )
        wcr = run_weyl_checks(entries, num_hard)
        assert isinstance(wcr, WeylCheckResult)
        assert wcr.ab is not None
        assert wcr.ab_valid

    def test_m004_ab_consistency_multiple_pairs(self):
        """b computed from m=±1 and m=±2 pairs should agree."""
        entries, num_hard = self._compute_entries(
            "m004",
            m_range=[-4, -2, -1, 0, 1, 2, 4],
            e_range=[0],
        )
        ab = compute_ab_vectors(entries, num_hard)
        assert ab is not None
        # No warnings about inconsistency between pairs
        inconsistency_warnings = [w for w in ab.warnings if "inconsistent" in w]
        assert inconsistency_warnings == [], f"Inconsistent b estimates: {inconsistency_warnings}"
