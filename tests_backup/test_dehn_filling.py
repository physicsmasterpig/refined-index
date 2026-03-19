"""
tests/test_dehn_filling.py — Tests for Dehn filling kernel and non-closable
cycle search (Step 5).

Test classes:
  TestExtGcd               — pure arithmetic (_ext_gcd)
  TestFindRS               — R·Q − P·S = 1 (find_rs)
  TestParticularSolution   — P·m0 + 2Q·e0 = c (_particular_solution)
  TestQSeriesArithmetic    — dict-based q-series helpers
  TestApplyKernel          — kernel * series product
  TestFilledIndexResult    — dataclass properties (no SnaPy / Mathematica)
  TestEnumerateKernelTerms — needs SnaPy; verifies structural properties
  TestComputeFilledIndex   — needs SnaPy + Mathematica
  TestFindNonClosableCycles — needs SnaPy + Mathematica

NZ basis reminder
─────────────────
  position  = M (meridian)
  momentum  = Λ = L/2 (half-longitude)
  slope P/Q: physical cycle P·M + Q·L
  R·Q − P·S = 1
  kernel index: c = P·m + 2Q·e  ∈ {−2, 0, 2}
"""

from __future__ import annotations

import subprocess
from fractions import Fraction
from math import gcd

import pytest

from manifold_index.core.dehn_filling import (
    FilledIndexResult,
    KernelTerm,
    NonClosableCycle,
    NonClosableCycleResult,
    QSeries,
    _apply_kernel,
    _candidate_slopes,
    _ext_gcd,
    _particular_solution,
    _qseries_add,
    _qseries_from_result,
    _qseries_scale,
    _qseries_shift,
    _qseries_truncate,
    compute_filled_index,
    enumerate_kernel_terms,
    find_non_closable_cycles,
    find_rs,
)
from manifold_index.core.index_3d import Index3DResult


# ---------------------------------------------------------------------------
# Skip markers (same pattern as test_index_3d.py)
# ---------------------------------------------------------------------------

def _has_snapy() -> bool:
    try:
        import snappy  # noqa: F401
        return True
    except ImportError:
        return False


def _has_mathematica() -> bool:
    for cmd in (
        "wolframscript",
        "/usr/local/bin/wolframscript",
        "/Applications/Wolfram.app/Contents/MacOS/wolframscript",
        "/Applications/Mathematica.app/Contents/MacOS/wolframscript",
        "/Applications/Wolfram Engine.app/Contents/MacOS/wolframscript",
    ):
        try:
            r = subprocess.run([cmd, "-version"], capture_output=True, timeout=10)
            if r.returncode == 0:
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return False


skip_no_snapy = pytest.mark.skipif(
    not _has_snapy(), reason="SnaPy not installed"
)
skip_no_math = pytest.mark.skipif(
    not _has_mathematica(), reason="Mathematica / WolframScript not available"
)

# ---------------------------------------------------------------------------
# Shared SnaPy fixture (m004)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def nz_m004():
    from manifold_index.core.manifold import load_manifold
    from manifold_index.core.gluing_equations import reduce_gluing_equations
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import build_neumann_zagier

    data = load_manifold("m004")
    easy = find_easy_edges(data)
    return build_neumann_zagier(data, easy)


# ===========================================================================
# TestExtGcd
# ===========================================================================

class TestExtGcd:
    """_ext_gcd returns (g, x, y) with a·x + b·y = g = gcd(|a|, |b|)."""

    def _check(self, a: int, b: int) -> None:
        g, x, y = _ext_gcd(a, b)
        assert a * x + b * y == g
        assert g == gcd(abs(a), abs(b)) if b != 0 else abs(a)

    def test_3_5(self):
        self._check(3, 5)

    def test_7_3(self):
        self._check(7, 3)

    def test_negative_a(self):
        g, x, y = _ext_gcd(-6, 9)
        assert -6 * x + 9 * y == g

    def test_b_zero(self):
        g, x, y = _ext_gcd(4, 0)
        assert g == 4
        assert 4 * x + 0 * y == 4

    def test_one_one(self):
        self._check(1, 1)

    def test_large(self):
        self._check(123, 456)


# ===========================================================================
# TestFindRS
# ===========================================================================

class TestFindRS:
    """find_rs returns (R, S) with R·Q − P·S = 1."""

    def _check(self, P: int, Q: int) -> None:
        R, S = find_rs(P, Q)
        assert R * Q - P * S == 1, (
            f"find_rs({P}, {Q}) = ({R}, {S}) but R·Q−P·S = {R*Q-P*S} ≠ 1"
        )

    def test_slope_1_1(self):
        self._check(1, 1)

    def test_slope_2_1(self):
        self._check(2, 1)

    def test_slope_1_2(self):
        self._check(1, 2)

    def test_slope_minus1_1(self):
        self._check(-1, 1)

    def test_slope_3_2(self):
        self._check(3, 2)

    def test_slope_5_3(self):
        self._check(5, 3)

    def test_slope_1_0(self):
        # Q=0: meridian surgery; R·0 − 1·S = −S = 1 → S = −1, R arbitrary
        R, S = find_rs(1, 0)
        assert R * 0 - 1 * S == 1

    def test_slope_0_1(self):
        # P=0: longitudinal surgery; R·1 − 0·S = R = 1
        R, S = find_rs(0, 1)
        assert R * 1 - 0 * S == 1

    def test_slope_minus3_2(self):
        self._check(-3, 2)

    def test_invalid_not_coprime_raises(self):
        with pytest.raises(ValueError, match="gcd"):
            find_rs(2, 4)

    def test_invalid_both_zero_raises(self):
        with pytest.raises(ValueError):
            find_rs(0, 0)


# ===========================================================================
# TestParticularSolution
# ===========================================================================

class TestParticularSolution:
    """_particular_solution(P, Q, c) returns (m0, e0) with P·m0 + 2Q·e0 = c."""

    def _check(self, P: int, Q: int, c: int) -> None:
        m0, e0 = _particular_solution(P, Q, c)
        lhs = P * m0 + 2 * Q * e0
        assert lhs == c, (
            f"P·m0+2Q·e0 = {lhs} ≠ {c} for P={P},Q={Q},c={c}"
        )
        # e0 must be integer or half-integer
        assert (2 * e0).denominator == 1, f"e0={e0} not in (½)ℤ"

    def test_c0_trivial(self):
        # gcd(P,Q)=1, c=0: particular solution should give 0
        m0, e0 = _particular_solution(1, 1, 0)
        assert 1 * m0 + 2 * 1 * e0 == 0

    def test_c2_slope_1_1(self):
        self._check(1, 1, 2)

    def test_c_minus2_slope_1_1(self):
        self._check(1, 1, -2)

    def test_c2_slope_2_1(self):
        self._check(2, 1, 2)

    def test_c0_slope_3_2(self):
        self._check(3, 2, 0)

    def test_c2_slope_3_2(self):
        self._check(3, 2, 2)

    def test_c_minus2_slope_5_3(self):
        self._check(5, 3, -2)

    def test_c2_slope_1_0(self):
        # P=1, Q=0: 1·m0 + 0 = 2 → m0=2
        m0, e0 = _particular_solution(1, 0, 2)
        assert m0 == 2

    def test_c2_slope_0_1(self):
        # P=0, Q=1: 0·m0 + 2·e0 = 2 → e0=1
        m0, e0 = _particular_solution(0, 1, 2)
        assert e0 == 1


# ===========================================================================
# TestQSeriesArithmetic
# ===========================================================================

class TestQSeriesArithmetic:
    """Unit tests for dict-based q^{1/2}-series helpers (no SnaPy needed)."""

    def _s(self, d: dict[int, int]) -> QSeries:
        return {k: Fraction(v) for k, v in d.items()}

    # ---- _qseries_shift ---------------------------------------------------

    def test_shift_zero(self):
        s = self._s({0: 1, 2: 3})
        assert _qseries_shift(s, 0) == s

    def test_shift_positive(self):
        # shift by 1: keys increase by 1
        s = self._s({0: 1, 4: 2})
        result = _qseries_shift(s, 1)
        assert result == self._s({1: 1, 5: 2})

    def test_shift_negative(self):
        s = self._s({3: 5})
        result = _qseries_shift(s, -3)
        assert result == self._s({0: 5})

    # ---- _qseries_scale ---------------------------------------------------

    def test_scale_by_one(self):
        s = self._s({0: 3, 2: 7})
        assert _qseries_scale(s, Fraction(1)) == s

    def test_scale_by_half(self):
        s = self._s({0: 2, 2: 4})
        result = _qseries_scale(s, Fraction(1, 2))
        assert result[0] == Fraction(1)
        assert result[2] == Fraction(2)

    def test_scale_by_zero(self):
        s = self._s({0: 5, 1: 3})
        assert _qseries_scale(s, Fraction(0)) == {}

    def test_scale_by_minus_half(self):
        s = self._s({0: 4})
        result = _qseries_scale(s, Fraction(-1, 2))
        assert result[0] == Fraction(-2)

    def test_scale_drops_zero_coeffs(self):
        # Fraction arithmetic: 0 × anything = 0 → should be omitted
        s = {0: Fraction(0), 2: Fraction(3)}
        result = _qseries_scale(s, Fraction(2))
        assert 0 not in result
        assert result[2] == 6

    # ---- _qseries_add -----------------------------------------------------

    def test_add_empty(self):
        s = self._s({0: 1, 2: 3})
        assert _qseries_add(s, {}) == s
        assert _qseries_add({}, s) == s

    def test_add_cancels(self):
        a = self._s({0: 1, 2: 3})
        b = {0: Fraction(-1), 2: Fraction(-3)}
        result = _qseries_add(a, b)
        assert result == {}

    def test_add_disjoint(self):
        a = self._s({0: 1})
        b = self._s({1: 2})
        result = _qseries_add(a, b)
        assert result == self._s({0: 1, 1: 2})

    def test_add_overlapping(self):
        a = self._s({0: 1, 2: 3})
        b = self._s({2: 1, 4: 5})
        result = _qseries_add(a, b)
        assert result[0] == 1
        assert result[2] == 4
        assert result[4] == 5

    # ---- _qseries_truncate ------------------------------------------------

    def test_truncate_keeps_boundary(self):
        s = self._s({-2: 1, 0: 2, 2: 3, 4: 4})
        result = _qseries_truncate(s, 2)
        assert 4 not in result
        assert result[2] == 3

    def test_truncate_empty(self):
        assert _qseries_truncate({}, 10) == {}

    def test_truncate_all_below(self):
        s = self._s({0: 1, 1: 2})
        result = _qseries_truncate(s, 5)
        assert result == s

    # ---- _qseries_from_result ---------------------------------------------

    def test_from_result_basic(self):
        """Index3DResult → QSeries dict."""
        r = Index3DResult(
            min_power=0, coeffs=[1, 0, 2], n_terms=3,
            q_order_half=10, m_ext=[0], e_ext=[0],
        )
        s = _qseries_from_result(r)
        assert s[0] == 1
        assert 1 not in s   # zero coefficient omitted
        assert s[2] == 2

    def test_from_result_negative_min_power(self):
        r = Index3DResult(
            min_power=-2, coeffs=[3, 1], n_terms=2,
            q_order_half=10, m_ext=[0], e_ext=[0],
        )
        s = _qseries_from_result(r)
        assert s[-2] == 3
        assert s[-1] == 1


# ===========================================================================
# TestApplyKernel
# ===========================================================================

class TestApplyKernel:
    """_apply_kernel(term, series) — pure arithmetic, no SnaPy needed."""

    def _s(self, d: dict[int, int]) -> QSeries:
        return {k: Fraction(v) for k, v in d.items()}

    def test_c0_phase0_trivial(self):
        """c=0, phase=0: kernel = (½)(+1)(q^0·I + q^0·I) = I."""
        term = KernelTerm(m=0, e=Fraction(0), c=0, phase=0)
        # (½)·(1)·(shift(I,0) + shift(I,0)) = (½)·2·I = I
        s = self._s({0: 1, 2: 3})
        result = _apply_kernel(term, s)
        assert result == s

    def test_c0_phase1_positive_sign(self):
        """c=0, phase=1: kernel = (½)(−1)^1 · (q^{1/2}·I + q^{-1/2}·I).
        With I = {0: 1} (just q^0):
          = (½)(−1) · ({1: 1} + {-1: 1})
          = (−½) · {-1: 1, 1: 1}
          = {-1: −½, 1: −½}
        """
        term = KernelTerm(m=0, e=Fraction(0), c=0, phase=1)
        s = self._s({0: 1})
        result = _apply_kernel(term, s)
        assert result == {-1: Fraction(-1, 2), 1: Fraction(-1, 2)}

    def test_c0_phase2(self):
        """c=0, phase=2: kernel = (½)(+1) · (q·I + q^{−1}·I).
        With I = {0: 1}:
          = (½)({2: 1} + {-2: 1}) = {-2: ½, 2: ½}
        """
        term = KernelTerm(m=0, e=Fraction(0), c=0, phase=2)
        s = self._s({0: 1})
        result = _apply_kernel(term, s)
        assert result == {-2: Fraction(1, 2), 2: Fraction(1, 2)}

    def test_c2_phase0(self):
        """c=2, phase=0: kernel = −(½)(+1) · I = −(½)·I."""
        term = KernelTerm(m=0, e=Fraction(0), c=2, phase=0)
        s = self._s({0: 4, 2: 6})
        result = _apply_kernel(term, s)
        assert result == {0: Fraction(-2), 2: Fraction(-3)}

    def test_c_minus2_phase1(self):
        """c=−2, phase=1: kernel = −(½)(−1)^1 · I = (½)·I."""
        term = KernelTerm(m=1, e=Fraction(1, 2), c=-2, phase=1)
        s = self._s({0: 2})
        result = _apply_kernel(term, s)
        assert result == {0: Fraction(1)}

    def test_c2_phase3(self):
        """c=2, phase=3 (odd): kernel = −(½)(−1)^3 · I = (½)·I."""
        term = KernelTerm(m=0, e=Fraction(0), c=2, phase=3)
        s = self._s({0: 6})
        result = _apply_kernel(term, s)
        assert result == {0: Fraction(3)}

    def test_empty_series(self):
        term = KernelTerm(m=0, e=Fraction(0), c=0, phase=0)
        assert _apply_kernel(term, {}) == {}

    def test_c0_phase0_larger_series(self):
        """c=0 phase=0: should just return I (identity kernel)."""
        s = {k: Fraction(k + 1) for k in range(5)}
        result = _apply_kernel(KernelTerm(m=0, e=Fraction(0), c=0, phase=0), s)
        assert result == s


# ===========================================================================
# TestFilledIndexResult
# ===========================================================================

class TestFilledIndexResult:
    """FilledIndexResult dataclass properties — no SnaPy / Mathematica needed."""

    def _make(self, series: QSeries) -> FilledIndexResult:
        return FilledIndexResult(
            P=2, Q=1, cusp_idx=0,
            series=series,
            q_order_half=20,
            n_kernel_terms=5,
        )

    def test_is_zero_empty(self):
        assert self._make({}).is_zero is True

    def test_is_zero_all_zero_values(self):
        # _qseries_add removes zero values, but defensive test:
        # Direct construction with zero values should still work.
        # is_zero checks len == 0; a dict with a zero-valued key is non-empty
        # so is_zero returns False (key 0 still present even though value = 0).
        # The QSeries helpers always remove zero-value keys, so in practice
        # this can only happen if someone manually constructs the result.
        s: QSeries = {0: Fraction(0)}
        r = self._make(s)
        assert r.is_zero is False  # key 0 still present

    def test_is_zero_nonzero(self):
        s = {0: Fraction(1), 2: Fraction(3)}
        assert self._make(s).is_zero is False

    # ---- is_stably_zero ---------------------------------------------------

    def test_is_stably_zero_empty(self):
        """Empty series is stably zero."""
        assert self._make({}).is_stably_zero() is True

    def test_is_stably_zero_stable_term_below_cutoff(self):
        """Non-zero term well below q_order_half is a stable term → not zero."""
        # q_order_half=20, buffer=max(5,20//2)=10, cutoff=10.  k=0 < 10.
        s = {0: Fraction(1)}
        assert self._make(s).is_stably_zero() is False

    def test_is_stably_zero_boundary_artifact_only(self):
        """Non-zero term exactly at q_order_half is a boundary artifact → zero."""
        # q_order_half=20, buffer=max(5,20//2)=10, cutoff=10.  k=20 > 10 → artifact.
        s = {20: Fraction(2)}
        assert self._make(s).is_stably_zero() is True

    def test_is_stably_zero_near_boundary_only(self):
        """Non-zero terms only in the last few powers → stably zero."""
        # q_order_half=20, buffer=max(5,20//2)=10, cutoff=10. Terms at 16,18,20 are all > 10.
        s = {16: Fraction(-1), 18: Fraction(3), 20: Fraction(7)}
        assert self._make(s).is_stably_zero() is True

    def test_is_stably_zero_custom_buffer(self):
        """Custom buffer shifts the stable/artifact boundary."""
        # buffer=2, cutoff=18.  k=17 ≤ 18 → stable → not zero.
        s = {17: Fraction(5)}
        assert self._make(s).is_stably_zero(buffer=2) is False
        # buffer=5, cutoff=15.  k=17 > 15 → artifact → zero.
        assert self._make(s).is_stably_zero(buffer=5) is True

    def test_is_stably_zero_mixed_stable_and_artifact(self):
        """Series with both stable and boundary terms is not stably zero."""
        # Stable term at k=2 overrules boundary artifact at k=20.
        s = {2: Fraction(1), 20: Fraction(-3)}
        assert self._make(s).is_stably_zero() is False

    def test_is_stably_zero_small_q_order_half_no_negative_cutoff(self):
        """Bug regression: at small q_order_half the buffer must be clamped so
        cutoff ≥ 1.  Without clamping, buffer=max(5,2)=5 at q_order_half=4
        gives cutoff=−1, making every non-empty series appear stably zero."""
        # q_order_half=4: buffer=min(max(5,4//2),4-1)=min(5,3)=3, cutoff=1.
        # A series with a term at k=0 has a stable term ≤ 1 → NOT stably zero.
        r_low = FilledIndexResult(
            P=0, Q=1, cusp_idx=0,
            series={0: Fraction(1), 2: Fraction(-2), 4: Fraction(3)},
            q_order_half=4,
            n_kernel_terms=1,
        )
        assert r_low.is_stably_zero() is False  # k=0 ≤ cutoff=1 → not zero

        # A series whose first term is at k=2 (above cutoff=1) IS stably zero.
        r_meridian_like = FilledIndexResult(
            P=1, Q=0, cusp_idx=0,
            series={2: Fraction(2), 4: Fraction(7)},
            q_order_half=4,
            n_kernel_terms=1,
        )
        assert r_meridian_like.is_stably_zero() is True  # no term ≤ cutoff=1

    def test_as_polynomial_string_zero(self):
        assert self._make({}).as_polynomial_string() == "0"

    def test_as_polynomial_string_nonzero(self):
        s = {0: Fraction(1), 2: Fraction(3)}
        ps = self._make(s).as_polynomial_string()
        assert "q" in ps or "1" in ps

    def test_fields_stored(self):
        r = self._make({4: Fraction(7)})
        assert r.P == 2
        assert r.Q == 1
        assert r.cusp_idx == 0
        assert r.q_order_half == 20
        assert r.n_kernel_terms == 5


# ===========================================================================
# TestCandidateSlopes
# ===========================================================================

class TestCandidateSlopes:
    """_candidate_slopes — pure Python, no SnaPy."""

    def test_excludes_zero_zero(self):
        slopes = _candidate_slopes(range(-1, 2), range(0, 2))
        assert (0, 0) not in slopes

    def test_all_coprime(self):
        slopes = _candidate_slopes(range(-3, 4), range(0, 5))
        for P, Q in slopes:
            assert gcd(abs(P), abs(Q)) == 1, f"({P},{Q}) not coprime"

    def test_includes_meridian(self):
        # (1, 0) is primitive
        slopes = _candidate_slopes(range(0, 2), range(0, 2))
        assert (1, 0) in slopes

    def test_includes_longitude(self):
        slopes = _candidate_slopes(range(0, 2), range(0, 2))
        assert (0, 1) in slopes

    def test_excludes_non_primitive(self):
        slopes = _candidate_slopes(range(0, 5), range(0, 5))
        assert (2, 4) not in slopes
        assert (3, 6) not in slopes

    # --- canonical_only tests ---

    def test_canonical_only_first_nonzero_positive(self):
        """Every slope returned by canonical_only=True has Q > 0, or Q=0 and P > 0."""
        slopes = _candidate_slopes(range(-3, 4), range(-3, 4), canonical_only=True)
        for P, Q in slopes:
            if Q != 0:
                assert Q > 0, f"({P},{Q}) has Q <= 0"
            else:
                assert P > 0, f"({P},{Q}) has Q=0 and P <= 0"

    def test_canonical_only_no_antipodal_pairs(self):
        """canonical_only=True: no slope AND its negation both appear."""
        slopes = _candidate_slopes(range(-2, 3), range(-2, 3), canonical_only=True)
        slope_set = set(slopes)
        for P, Q in slopes:
            assert (-P, -Q) not in slope_set, (
                f"Both ({P},{Q}) and ({-P},{-Q}) appear in canonical list"
            )

    def test_canonical_only_covers_full_range(self):
        """Every slope in full list has its canonical rep in the canonical_only list."""
        full = set(_candidate_slopes(range(-2, 3), range(-2, 3)))
        canon = set(_candidate_slopes(range(-2, 3), range(-2, 3), canonical_only=True))
        for P, Q in full:
            assert (P, Q) in canon or (-P, -Q) in canon, (
                f"Neither ({P},{Q}) nor ({-P},{-Q}) found in canonical list"
            )

    def test_canonical_only_half_size_symmetric_range(self):
        """canonical_only halves the Q≠0 slopes; Q=0 slopes are already normalised
        to P>0 so they appear once in both full and canon."""
        full = _candidate_slopes(range(-3, 4), range(-3, 4))
        canon = _candidate_slopes(range(-3, 4), range(-3, 4), canonical_only=True)
        # Q=0 slopes appear once in full (P>0 only) and once in canon.
        # Q≠0 slopes come in antipodal pairs in full; canon keeps one per pair.
        n_meridional = sum(1 for _, Q in full if Q == 0)
        assert len(canon) == (len(full) + n_meridional) // 2

    def test_q0_always_normalised_to_positive_p(self):
        """When Q=0, only P>0 is returned regardless of canonical_only."""
        for canonical in (False, True):
            slopes = _candidate_slopes(range(-3, 4), range(-1, 2), canonical_only=canonical)
            q0_slopes = [(P, Q) for P, Q in slopes if Q == 0]
            assert all(P > 0 for P, Q in q0_slopes), (
                f"canonical_only={canonical}: got Q=0 slope with P≤0: {q0_slopes}"
            )
            # Only (1,0) is primitive with Q=0 in this range
            assert q0_slopes == [(1, 0)]

    def test_canonical_only_false_is_default(self):
        """canonical_only=False (default) returns same list as omitting the arg."""
        slopes_default = _candidate_slopes(range(-2, 3), range(-2, 3))
        slopes_explicit = _candidate_slopes(range(-2, 3), range(-2, 3), canonical_only=False)
        assert slopes_default == slopes_explicit

    def test_canonical_only_subset_of_full(self):
        """canonical_only=True is a subset of the full slope list."""
        full = set(_candidate_slopes(range(-2, 3), range(-2, 3)))
        canon = set(_candidate_slopes(range(-2, 3), range(-2, 3), canonical_only=True))
        assert canon.issubset(full)


# ===========================================================================
# TestEnumerateKernelTerms  (requires SnaPy)
# ===========================================================================

@skip_no_snapy
class TestEnumerateKernelTerms:
    """Structural tests for enumerate_kernel_terms — needs SnaPy for NZ data."""

    def test_returns_list(self, nz_m004):
        R, S = find_rs(2, 1)
        terms = enumerate_kernel_terms(2, 1, R, S, nz_m004, 0, [], [], 10)
        assert isinstance(terms, list)

    def test_all_terms_satisfy_c_constraint(self, nz_m004):
        """Every returned term must have P·m + 2Q·e ∈ {−2, 0, 2}."""
        P, Q = 2, 1
        R, S = find_rs(P, Q)
        terms = enumerate_kernel_terms(P, Q, R, S, nz_m004, 0, [], [], 10)
        for kt in terms:
            c_val = P * kt.m + 2 * Q * kt.e
            assert c_val in (-2, 0, 2), (
                f"c={c_val} not in {{-2,0,2}} for m={kt.m}, e={kt.e}"
            )
            assert kt.c == c_val

    def test_all_phases_are_integer(self, nz_m004):
        """phase = R·m + 2S·e must be integer."""
        P, Q = 3, 2
        R, S = find_rs(P, Q)
        terms = enumerate_kernel_terms(P, Q, R, S, nz_m004, 0, [], [], 8)
        for kt in terms:
            assert isinstance(kt.phase, int), (
                f"phase={kt.phase} is not int for m={kt.m},e={kt.e}"
            )

    def test_no_duplicate_me_pairs(self, nz_m004):
        """Each (m, e) pair should appear at most once."""
        P, Q = 1, 1
        R, S = find_rs(P, Q)
        terms = enumerate_kernel_terms(P, Q, R, S, nz_m004, 0, [], [], 10)
        keys = [(kt.m, kt.e) for kt in terms]
        assert len(keys) == len(set(keys)), "Duplicate (m, e) pairs found"

    def test_e_is_fraction(self, nz_m004):
        """e must be a Fraction (or integer coercible to Fraction)."""
        P, Q = 2, 1
        R, S = find_rs(P, Q)
        terms = enumerate_kernel_terms(P, Q, R, S, nz_m004, 0, [], [], 8)
        for kt in terms:
            assert isinstance(kt.e, Fraction), f"e={kt.e!r} is not Fraction"

    def test_c_field_consistent(self, nz_m004):
        """kt.c attribute must equal P·m + 2Q·e."""
        P, Q = 1, 2
        R, S = find_rs(P, Q)
        terms = enumerate_kernel_terms(P, Q, R, S, nz_m004, 0, [], [], 8)
        for kt in terms:
            expected_c = P * kt.m + 2 * Q * kt.e
            assert kt.c == expected_c


# ===========================================================================
# TestComputeFilledIndex  (requires SnaPy + Mathematica)
# ===========================================================================

@skip_no_snapy
@skip_no_math
class TestComputeFilledIndex:
    """Integration tests for compute_filled_index."""

    def test_returns_filled_index_result(self, nz_m004):
        result = compute_filled_index(
            nz_m004, cusp_idx=0, P=2, Q=1, q_order_half=8
        )
        assert isinstance(result, FilledIndexResult)

    def test_n_kernel_terms_nonnegative(self, nz_m004):
        result = compute_filled_index(
            nz_m004, cusp_idx=0, P=1, Q=1, q_order_half=6
        )
        assert result.n_kernel_terms >= 0

    def test_pq_stored_correctly(self, nz_m004):
        result = compute_filled_index(
            nz_m004, cusp_idx=0, P=3, Q=2, q_order_half=6
        )
        assert result.P == 3
        assert result.Q == 2
        assert result.cusp_idx == 0

    def test_series_keys_within_order(self, nz_m004):
        """All series keys must be ≤ q_order_half."""
        q = 8
        result = compute_filled_index(
            nz_m004, cusp_idx=0, P=1, Q=1, q_order_half=q
        )
        for k in result.series:
            assert k <= q, f"key {k} exceeds q_order_half={q}"

    def test_series_values_are_fractions(self, nz_m004):
        result = compute_filled_index(
            nz_m004, cusp_idx=0, P=2, Q=1, q_order_half=8
        )
        for v in result.series.values():
            assert isinstance(v, Fraction)

    def test_integer_slope_fills_cusp(self, nz_m004):
        """Smoke test: meridian filling (1, 0) should produce a finite series."""
        result = compute_filled_index(
            nz_m004, cusp_idx=0, P=1, Q=0, q_order_half=6
        )
        assert isinstance(result, FilledIndexResult)


# ===========================================================================
# TestFindNonClosableCycles  (requires SnaPy + Mathematica)
# ===========================================================================

@skip_no_snapy
@skip_no_math
class TestFindNonClosableCycles:
    """Integration tests for find_non_closable_cycles."""

    def test_returns_result_type(self, nz_m004):
        result = find_non_closable_cycles(
            nz_m004, cusp_idx=0,
            p_range=range(0, 2), q_range=range(0, 2),
            q_order_half=6,
        )
        assert isinstance(result, NonClosableCycleResult)

    def test_slopes_tested_nonempty(self, nz_m004):
        result = find_non_closable_cycles(
            nz_m004, cusp_idx=0,
            p_range=range(0, 2), q_range=range(0, 2),
            q_order_half=6,
        )
        assert len(result.slopes_tested) > 0

    def test_slopes_tested_all_primitive(self, nz_m004):
        result = find_non_closable_cycles(
            nz_m004, cusp_idx=0,
            p_range=range(-2, 3), q_range=range(0, 3),
            q_order_half=6,
        )
        for P, Q in result.slopes_tested:
            assert gcd(abs(P), abs(Q)) == 1

    def test_cycles_are_subset_of_slopes_tested(self, nz_m004):
        result = find_non_closable_cycles(
            nz_m004, cusp_idx=0,
            p_range=range(-1, 2), q_range=range(0, 2),
            q_order_half=6,
        )
        tested_set = set(result.slopes_tested)
        for nc in result.cycles:
            assert (nc.P, nc.Q) in tested_set

    def test_non_closable_cycle_str(self):
        nc = NonClosableCycle(cusp_idx=0, P=2, Q=1)
        s = str(nc)
        assert "2" in s and "1" in s

    def test_cusp_idx_stored(self, nz_m004):
        result = find_non_closable_cycles(
            nz_m004, cusp_idx=0,
            p_range=range(0, 2), q_range=range(0, 2),
            q_order_half=6,
        )
        assert result.cusp_idx == 0
        for nc in result.cycles:
            assert nc.cusp_idx == 0

    def test_slopes_tested_full_regardless_of_symmetry(self, nz_m004):
        """slopes_tested always contains all primitive slopes, even with use_symmetry=True."""
        r_sym = find_non_closable_cycles(
            nz_m004, cusp_idx=0,
            p_range=range(-1, 2), q_range=range(-1, 2),
            q_order_half=6, use_symmetry=True,
        )
        r_full = find_non_closable_cycles(
            nz_m004, cusp_idx=0,
            p_range=range(-1, 2), q_range=range(-1, 2),
            q_order_half=6, use_symmetry=False,
        )
        assert set(r_sym.slopes_tested) == set(r_full.slopes_tested)

    def test_use_symmetry_antipodal_pair_both_in_cycles(self, nz_m004):
        """If (P,Q) is non-closable, use_symmetry=True also marks (-P,-Q)."""
        p_rng, q_rng = range(-2, 3), range(-2, 3)
        result = find_non_closable_cycles(
            nz_m004, cusp_idx=0,
            p_range=p_rng, q_range=q_rng,
            q_order_half=6, use_symmetry=True,
        )
        all_slopes = set(_candidate_slopes(p_rng, q_rng))
        cycle_set = {(nc.P, nc.Q) for nc in result.cycles}
        for P, Q in cycle_set:
            # Antipodal partner must also be a cycle (if it's in the search range)
            if (-P, -Q) in all_slopes:
                assert (-P, -Q) in cycle_set, (
                    f"({P},{Q}) is non-closable but ({-P},{-Q}) is not in cycles"
                )


# ---------------------------------------------------------------------------
# v1683 regression: is_stably_zero buffer-clamp bug
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def nz_v1683():
    from manifold_index.core.manifold import load_manifold
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import build_neumann_zagier

    data = load_manifold("v1683")
    easy = find_easy_edges(data)
    return build_neumann_zagier(data, easy)


@skip_no_snapy
@skip_no_math
class TestNonClosableCyclesV1683:
    """Regression tests for v1683: only the meridian (1,0) is non-closable.

    Prior bug: at q_order_half=4 the default buffer formula
    max(5, q_order_half//2) = 5 produced cutoff = 4 − 5 = −1, a negative
    value.  Since no series key satisfies k ≤ −1, every slope appeared
    stably zero, yielding 8 spurious non-closable cycles.

    Fix: buffer is now clamped to q_order_half−1, guaranteeing cutoff ≥ 1.
    """

    @pytest.mark.slow
    def test_meridian_only_at_q_order_half_8(self, nz_v1683):
        """q_order_half=8: exactly one non-closable cycle, the meridian (1,0)."""
        result = find_non_closable_cycles(
            nz_v1683, cusp_idx=0,
            p_range=range(-2, 3), q_range=range(0, 3),
            q_order_half=8,
        )
        cycle_set = {(nc.P, nc.Q) for nc in result.cycles}
        assert cycle_set == {(1, 0)}, (
            f"Expected {{(1,0)}}, got {cycle_set}"
        )

    def test_meridian_only_at_q_order_half_4(self, nz_v1683):
        """q_order_half=4: still correctly identifies only the meridian (1,0).

        Regression for the negative-cutoff bug: before the fix, every slope
        was incorrectly reported as non-closable at this small q_order_half.
        """
        result = find_non_closable_cycles(
            nz_v1683, cusp_idx=0,
            p_range=range(-2, 3), q_range=range(0, 3),
            q_order_half=4,
        )
        cycle_set = {(nc.P, nc.Q) for nc in result.cycles}
        assert cycle_set == {(1, 0)}, (
            f"Expected {{(1,0)}}, got {cycle_set}"
        )
