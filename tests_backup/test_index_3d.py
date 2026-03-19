"""
tests/test_index_3d.py — Tests for 3D index computation (Step 4).

Test classes:
  TestTetDegree               — pure formula, no SnaPy
  TestBuildKappa              — κ assembly with synthetic data
  TestPhaseExponent           — phase dot-product with synthetic data
  TestValidHalfIntegerPatterns — integrality filter with synthetic NZ-inv matrices
  TestEnumerateSummationTerms — needs SnaPy (m004); skipped if unavailable

Precomputed δ(m, e) values (from Lemma 3.6):
  (0, 0) → 0
  (1, 0) → 3/2
  (0, 1) → 0
  (-1, 0) → 0
  (0, -1) → 3/2
  (1, -1) → 1
  (-1, 1) → 1/2
  (1, 1) → 2
  (2, -3) → 9/2
  (-2, 3) → 3
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from manifold_index.core.index_3d import (
    Index3DResult,
    build_kappa,
    enumerate_summation_terms,
    phase_exponent,
    tet_degree,
    valid_half_integer_patterns,
)


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


# ---------------------------------------------------------------------------
# Fixtures (require SnaPy)
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
# TestTetDegree
# ===========================================================================

class TestTetDegree:
    """Pure formula tests — no SnaPy, no Mathematica."""

    def test_zero_zero(self):
        assert tet_degree(0, 0) == Fraction(0)

    def test_one_zero(self):
        # m₊(m+e)₊ = 1·1 = 1; (-m)₊ e₊ = 0; (-e)₊(-e-m)₊ = 0
        # half = 1/2; max{0,1,-0} = 1  → total = 3/2
        assert tet_degree(1, 0) == Fraction(3, 2)

    def test_zero_one(self):
        # m₊(m+e)₊=0; (-m)₊e₊=0; (-e)₊(-e-m)₊=0; max{0,0,-1}=0 → 0
        assert tet_degree(0, 1) == Fraction(0)

    def test_minus_one_zero(self):
        # m₊(m+e)₊=0; (-m)₊e₊=1·0=0; (-e)₊(-e-m)₊=0; max{0,-1,0}=0 → 0
        assert tet_degree(-1, 0) == Fraction(0)

    def test_zero_minus_one(self):
        # m₊(m+e)₊=0; (-m)₊e₊=0; (-e)₊(-e-m)₊=1·1=1; max{0,0,1}=1 → 1/2+1=3/2
        assert tet_degree(0, -1) == Fraction(3, 2)

    def test_one_minus_one(self):
        # m=1,e=-1: m₊(m+e)₊=1·0=0; (-m)₊e₊=0; (-e)₊(-e-m)₊=1·0=0
        # max{0,1,1}=1 → 0 + 1 = 1
        assert tet_degree(1, -1) == Fraction(1)

    def test_minus_one_one(self):
        # m=-1,e=1: m₊=0; (-m)₊e₊=1·1=1; (-e)₊=0; max{0,-1,-1}=0 → 1/2
        assert tet_degree(-1, 1) == Fraction(1, 2)

    def test_one_one(self):
        # m=1,e=1: m₊(m+e)₊=1·2=2; (-m)₊=0; (-e)₊=0; max{0,1,-1}=1 → 1+1=2
        assert tet_degree(1, 1) == Fraction(2)

    def test_two_minus_three(self):
        # m=2,e=-3: m₊(m+e)₊=2·0=0; (-m)₊=0; (-e)₊(-e-m)₊=3·1=3
        # max{0,2,3}=3 → 3/2+3=9/2
        assert tet_degree(2, -3) == Fraction(9, 2)

    def test_minus_two_three(self):
        # m=-2,e=3: m₊=0; (-m)₊e₊=2·3=6; (-e)₊=0; max{0,-2,-3}=0 → 3
        assert tet_degree(-2, 3) == Fraction(3)

    def test_nonnegative(self):
        for m in range(-4, 5):
            for e in range(-4, 5):
                assert tet_degree(m, e) >= 0, f"negative degree at ({m},{e})"

    def test_returns_fraction(self):
        result = tet_degree(1, 0)
        assert isinstance(result, Fraction)


# ===========================================================================
# TestBuildKappa
# ===========================================================================

class TestBuildKappa:
    """κ vector assembly with synthetic shape parameters."""

    def test_cusp_only_no_internal(self):
        # n=2, r=2, n_int=0: no internal edges
        # m_ext=[3,5], e_ext=[7,11], e_int=[]
        kappa = build_kappa([3, 5], [7, 11], [], n=2, r=2)
        assert kappa.shape == (4,)
        assert kappa[0] == 3
        assert kappa[1] == 5
        assert kappa[2] == Fraction(7)
        assert kappa[3] == Fraction(11)

    def test_with_internal_edge(self):
        # n=3, r=1, n_int=2: 2 internal edges (was: 1 hard + 1 easy)
        # m_ext=[1] (cusp only), e_ext=[3], e_int=[4, Fraction(1,2)]
        kappa = build_kappa(
            [1], [3], [4, Fraction(1, 2)],
            n=3, r=1
        )
        assert kappa.shape == (6,)
        assert kappa[1] == 0           # internal edge m forced to 0
        assert kappa[2] == 0           # internal edge m forced to 0
        assert kappa[3] == Fraction(3) # e_cusp
        assert kappa[4] == Fraction(4) # e_int[0]
        assert kappa[5] == Fraction(1, 2)  # e_int[1]

    def test_internal_m_forced_zero(self):
        # The internal-edge block of kappa[:n] must be zero
        kappa = build_kappa(
            [10], [20], [3],
            n=2, r=1
        )
        assert kappa[1] == 0  # internal edge m = 0

    def test_e_ext_as_fraction(self):
        kappa = build_kappa([0], [0], [], n=1, r=1)
        assert isinstance(kappa[1], Fraction)

    def test_shape_2n(self):
        for n in range(1, 5):
            kappa = build_kappa(
                [0] * n, [0] * n, [],
                n=n, r=n
            )
            assert kappa.shape == (2 * n,)


# ===========================================================================
# TestPhaseExponent
# ===========================================================================

class TestPhaseExponent:
    """Phase dot-product with synthetic ν_x, ν_p vectors."""

    def _make_kappa(self, m_vals, e_vals):
        n = len(m_vals)
        kappa = np.empty(2 * n, dtype=object)
        for i, v in enumerate(m_vals):
            kappa[i] = v
        for i, v in enumerate(e_vals):
            kappa[n + i] = Fraction(v)
        return kappa

    def test_zero_phase(self):
        kappa = self._make_kappa([0, 0], [0, 0])
        nu_x = np.array([1, 2])
        nu_p = np.array([3, 4])
        assert phase_exponent(kappa, nu_x, nu_p, n=2, r=2, num_hard=0) == 0

    def test_only_m_contribution(self):
        # m_full · nu_p with e=0
        kappa = self._make_kappa([1, 2], [0, 0])
        nu_x = np.array([10, 10])
        nu_p = np.array([3, 5])
        # 1*3 + 2*5 = 13
        assert phase_exponent(kappa, nu_x, nu_p, n=2, r=2, num_hard=0) == 13

    def test_only_e_contribution(self):
        # -e_full · nu_x with m=0
        kappa = self._make_kappa([0, 0], [1, 2])
        nu_x = np.array([4, 6])
        nu_p = np.array([0, 0])
        # -(1*4 + 2*6) = -16
        assert phase_exponent(kappa, nu_x, nu_p, n=2, r=2, num_hard=0) == -16

    def test_combined(self):
        kappa = self._make_kappa([1, 0], [2, 3])
        nu_x = np.array([1, 1])
        nu_p = np.array([7, 0])
        # m·nu_p = 1*7 + 0*0 = 7
        # e·nu_x = 2*1 + 3*1 = 5
        # phase = 7 - 5 = 2
        assert phase_exponent(kappa, nu_x, nu_p, n=2, r=2, num_hard=0) == 2

    def test_returns_fraction(self):
        kappa = self._make_kappa([1], [0])
        nu_x = np.array([1])
        nu_p = np.array([1])
        result = phase_exponent(kappa, nu_x, nu_p, n=1, r=1, num_hard=0)
        assert isinstance(result, Fraction)


# ===========================================================================
# TestValidHalfIntegerPatterns
# ===========================================================================

class TestValidHalfIntegerPatterns:
    """Integrality filter for internal-edge half-integer patterns."""

    def test_no_internal_edges(self):
        # n=2, r=2 → n_int=0: no internal edges at all
        g_inv = np.eye(4, dtype=int)
        result = valid_half_integer_patterns(g_inv, n=2, r=2)
        assert len(result) == 1
        assert result[0].shape == (0,)

    def test_one_internal_odd_column(self):
        # n=2, r=1 → n_int=1: one internal edge
        # g_inv = I_4; int_cols = g_inv[:, 3:4] = column 3 = [0,0,0,1]^T
        # delta=0 → [0,0,0,0]: all even ✓; delta=1 → [0,0,0,1]: odd → excluded
        g_inv = np.eye(4, dtype=int)
        result = valid_half_integer_patterns(g_inv, n=2, r=1)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], [0])

    def test_even_column_allows_half(self):
        # If int_col is all-even, both delta=0 and delta=1 are valid
        g_inv = np.array([
            [1, 0, 0, 2],
            [0, 1, 0, 4],
            [0, 0, 1, 0],
            [0, 0, 0, 2],
        ], dtype=int)
        result = valid_half_integer_patterns(g_inv, n=2, r=1)
        assert len(result) == 2  # both delta=0 and delta=1

    def test_two_internal_edges(self):
        # n=3, r=1 → n_int=2: 2 internal edges
        # g_inv = I_6 → int_cols = columns 4..5 = identity columns (odd) → only delta=[0,0]
        g_inv = np.zeros((6, 6), dtype=int)
        np.fill_diagonal(g_inv, 1)
        result = valid_half_integer_patterns(g_inv, n=3, r=1)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], [0, 0])

    def test_all_zeros_g_inv(self):
        # Degenerate: all int columns zero → all 2^k patterns valid
        n_int = 3
        n = n_int + 1
        g_inv = np.zeros((2 * n, 2 * n), dtype=int)
        result = valid_half_integer_patterns(g_inv, n=n, r=1)
        assert len(result) == 2 ** n_int


# ===========================================================================
# TestEnumerateSummationTerms  (requires SnaPy)
# ===========================================================================

@skip_no_snapy
class TestEnumerateSummationTerms:
    """Integration tests for the Python-side summation enumeration."""

    def test_returns_list(self, nz_m004):
        ext = [0] * nz_m004.r
        result = enumerate_summation_terms(nz_m004, ext, ext, q_order_half=10)
        assert isinstance(result, list)

    def test_term_keys(self, nz_m004):
        ext = [0] * nz_m004.r
        result = enumerate_summation_terms(nz_m004, ext, ext, q_order_half=10)
        for term in result:
            assert "phase_exp" in term
            assert "tet_args" in term
            assert "min_degree" in term

    def test_phase_exp_is_int(self, nz_m004):
        ext = [0] * nz_m004.r
        result = enumerate_summation_terms(nz_m004, ext, ext, q_order_half=10)
        for term in result:
            assert isinstance(term["phase_exp"], int), (
                f"phase_exp should be int, got {type(term['phase_exp'])}: {term['phase_exp']}"
            )

    def test_tet_args_integer(self, nz_m004):
        ext = [0] * nz_m004.r
        result = enumerate_summation_terms(nz_m004, ext, ext, q_order_half=10)
        for term in result:
            for ta, tb in term["tet_args"]:
                assert isinstance(ta, int)
                assert isinstance(tb, int)

    def test_degree_cutoff(self, nz_m004):
        cutoff = 6
        ext = [0] * nz_m004.r
        result = enumerate_summation_terms(nz_m004, ext, ext, q_order_half=cutoff)
        for term in result:
            assert term["min_degree"] <= cutoff

    def test_multiple_terms_summed(self, nz_m004):
        # With ALL internal edges summed, m004 should have > 1 term at q_order_half=10
        ext = [0] * nz_m004.r
        result = enumerate_summation_terms(nz_m004, ext, ext, q_order_half=10)
        assert len(result) > 1, (
            f"Expected multiple summation terms (all internal edges summed), "
            f"got {len(result)}"
        )


# ===========================================================================
# TestIndex3DResult
# ===========================================================================

class TestIndex3DResult:
    """Tests for the Index3DResult dataclass."""

    def test_as_polynomial_zero(self):
        r = Index3DResult(
            coeffs=[0, 0, 0], min_power=0, q_order_half=2,
            m_ext=[0], e_ext=[0], n_terms=0
        )
        assert r.as_polynomial_string() == "0"

    def test_as_polynomial_constant(self):
        r = Index3DResult(
            coeffs=[5, 0, 0], min_power=0, q_order_half=2,
            m_ext=[0], e_ext=[0], n_terms=1
        )
        assert r.as_polynomial_string() == "5"

    def test_as_polynomial_q_term(self):
        r = Index3DResult(
            coeffs=[0, 0, 1], min_power=0, q_order_half=2,
            m_ext=[0], e_ext=[0], n_terms=1
        )
        assert "q" in r.as_polynomial_string()
