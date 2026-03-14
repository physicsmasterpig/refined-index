"""
tests/test_refined_index.py — Tests for the refined index (Step 8).

Test classes
------------
TestProjectTo3DIndex
    Pure unit tests of ``project_to_3d_index``.  No SnaPy required.

TestComputeRefinedIndexAllEasy
    For manifolds whose triangulation has NO hard edges (num_hard == 0),
    ``compute_refined_index`` must produce keys of the form ``(q_half_power,)``
    and ``project_to_3d_index`` must exactly match ``compute_index_3d_python``.
    Requires SnaPy.

TestComputeRefinedIndexHardEdges
    For manifolds with hard edges, the η=1 projection must equal
    ``compute_index_3d_python``.  Requires SnaPy.

TestFormatRefinedIndex
    Smoke tests for ``format_refined_index``.  No SnaPy required.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from manifold_index.core.refined_index import (
    RefinedIndexResult,
    compute_refined_index,
    format_refined_index,
    project_to_3d_index,
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
# Helpers shared by several tests
# ---------------------------------------------------------------------------

def _load_nz(name: str):
    """Return (nz_data, num_cusps) for a named manifold."""
    from manifold_index.core.manifold import load_manifold
    from manifold_index.core.neumann_zagier import build_neumann_zagier
    from manifold_index.core.phase_space import find_easy_edges

    data = load_manifold(name)
    easy = find_easy_edges(data)
    nz = build_neumann_zagier(data, easy)
    return nz


# ===========================================================================
# TestProjectTo3DIndex — pure unit tests, no SnaPy
# ===========================================================================

class TestProjectTo3DIndex:
    """project_to_3d_index sums fugacity monomials → ordinary q-series."""

    def test_empty(self):
        assert project_to_3d_index({}) == {}

    def test_no_hard_edges(self):
        """With k=0 keys are (q_pow,); projection is the identity."""
        refined: RefinedIndexResult = {(0,): 1, (2,): -3, (4,): 2}
        assert project_to_3d_index(refined) == {0: 1, 2: -3, 4: 2}

    def test_single_hard_edge_identity(self):
        """When η_0 exponent is 0 for every term, projection is identity."""
        refined: RefinedIndexResult = {(0, 0): 5, (2, 0): -1}
        assert project_to_3d_index(refined) == {0: 5, 2: -1}

    def test_sums_fugacity_monomials(self):
        """Coefficients sharing the same q-power are summed."""
        refined: RefinedIndexResult = {
            (2, 2): 1,    # q * η_0
            (2, 0): 1,    # q
            (2, -2): 1,   # q * η_0^{-1}
        }
        # η=1: all three contribute to q^1 coefficient
        assert project_to_3d_index(refined) == {2: 3}

    def test_cancellation(self):
        """Coefficients that cancel after projection give zero → dropped."""
        refined: RefinedIndexResult = {
            (4, 2): 1,
            (4, -2): -1,
        }
        # sum = 0 → key 4 dropped
        assert project_to_3d_index(refined) == {}

    def test_two_hard_edges(self):
        """Three fugacity variables; only q-power matters for projection."""
        refined: RefinedIndexResult = {
            (0, 2, 0): 3,
            (0, -2, 4): -1,
            (0, 0, 0): 2,
        }
        assert project_to_3d_index(refined) == {0: 4}


# ===========================================================================
# TestComputeRefinedIndexAllEasy — requires SnaPy
# ===========================================================================

class TestComputeRefinedIndexAllEasy:
    """Key structure and η=1 projection correctness checks."""

    @skip_no_snapy
    @pytest.mark.parametrize("manifold,m_ext,e_ext", [
        ("4_1", [0], [0]),
        ("4_1", [1], [0]),
        ("4_1", [1], [1]),
        ("m004", [0], [0]),
        ("m004", [1], [0]),
    ])
    def test_key_length_equals_1_plus_num_hard(self, manifold, m_ext, e_ext):
        """Every key must have length 1 + num_hard."""
        nz = _load_nz(manifold)
        k = nz.num_hard
        refined = compute_refined_index(nz, m_ext, e_ext, q_order_half=10)
        for key in refined:
            assert len(key) == 1 + k, (
                f"{manifold}: key {key} has length {len(key)}, expected {1+k}"
            )

    @skip_no_snapy
    @pytest.mark.parametrize("manifold,m_ext,e_ext", [
        ("4_1", [0], [0]),
        ("4_1", [1], [0]),
        ("4_1", [1], [1]),
        ("m004", [0], [0]),
        ("m004", [1], [0]),
    ])
    def test_projection_matches_3d_index(self, manifold, m_ext, e_ext):
        """project_to_3d_index(refined) must equal compute_index_3d_python."""
        from manifold_index.core.index_3d import compute_index_3d_python

        nz = _load_nz(manifold)
        q_ord = 12
        refined = compute_refined_index(nz, m_ext, e_ext, q_order_half=q_ord)
        projected = project_to_3d_index(refined)

        res3d = compute_index_3d_python(nz, m_ext, e_ext, q_order_half=q_ord)
        # Build q_half_power → coeff dict from Index3DResult
        expected = {
            res3d.min_power + k: c
            for k, c in enumerate(res3d.coeffs)
            if c != 0
        }

        assert projected == expected, (
            f"{manifold} m={m_ext} e={e_ext}: "
            f"projected={dict(sorted(projected.items()))}, "
            f"expected={dict(sorted(expected.items()))}"
        )


# ===========================================================================
# TestComputeRefinedIndexHardEdges — requires SnaPy
# ===========================================================================

class TestComputeRefinedIndexHardEdges:
    """For manifolds with hard edges, η=1 projection must match 3D index."""

    @skip_no_snapy
    def _find_manifold_with_hard_edges(self):
        """Return a (name, nz) pair for a manifold with num_hard > 0, or None."""
        for name in ["m003", "m009", "m015", "m017", "m023"]:
            try:
                nz = _load_nz(name)
                if nz.num_hard > 0:
                    return name, nz
            except Exception:
                continue
        return None

    @skip_no_snapy
    def test_hard_edge_key_length(self):
        """Each key must have length 1 + num_hard."""
        pair = self._find_manifold_with_hard_edges()
        if pair is None:
            pytest.skip("No manifold with hard edges found in test set")
        name, nz = pair
        k = nz.num_hard
        r = nz.r
        m_ext = [0] * r
        e_ext = [0] * r
        refined = compute_refined_index(nz, m_ext, e_ext, q_order_half=8)
        for key in refined:
            assert len(key) == 1 + k, (
                f"{name}: key {key} has length {len(key)}, expected {1+k}"
            )

    @skip_no_snapy
    def test_projection_matches_3d_index_hard(self):
        """η=1 projection of refined index equals compute_index_3d_python."""
        from manifold_index.core.index_3d import compute_index_3d_python

        pair = self._find_manifold_with_hard_edges()
        if pair is None:
            pytest.skip("No manifold with hard edges found in test set")
        name, nz = pair
        r = nz.r
        m_ext = [0] * r
        e_ext = [0] * r
        q_ord = 10

        refined = compute_refined_index(nz, m_ext, e_ext, q_order_half=q_ord)
        projected = project_to_3d_index(refined)

        res3d = compute_index_3d_python(nz, m_ext, e_ext, q_order_half=q_ord)
        expected = {
            res3d.min_power + k: c
            for k, c in enumerate(res3d.coeffs)
            if c != 0
        }

        assert projected == expected, (
            f"{name} m={m_ext} e={e_ext}: "
            f"projected={dict(sorted(projected.items()))}, "
            f"expected={dict(sorted(expected.items()))}"
        )

    @skip_no_snapy
    def test_fugacity_nontrivial(self):
        """With hard edges, not all keys should have zero fugacity exponents."""
        pair = self._find_manifold_with_hard_edges()
        if pair is None:
            pytest.skip("No manifold with hard edges found in test set")
        name, nz = pair
        r = nz.r
        m_ext = [0] * r
        e_ext = [0] * r
        refined = compute_refined_index(nz, m_ext, e_ext, q_order_half=10)
        # At least one term must have a nonzero fugacity exponent
        has_nonzero_eta = any(any(e != 0 for e in key[1:]) for key in refined)
        assert has_nonzero_eta, (
            f"{name}: all fugacity exponents are zero — "
            "refined index is trivially the same as 3D index"
        )


# ===========================================================================
# TestFormatRefinedIndex — no SnaPy required
# ===========================================================================

class TestFormatRefinedIndex:
    """format_refined_index produces human-readable strings."""

    def test_empty(self):
        assert format_refined_index({}, num_hard=1) == "0"

    def test_constant_term(self):
        refined: RefinedIndexResult = {(0, 0): 3}
        s = format_refined_index(refined, num_hard=1)
        assert "3" in s

    def test_no_hard_edges(self):
        """num_hard=0: keys are (q_pow,); only q factors appear."""
        refined: RefinedIndexResult = {(0,): 1, (2,): -2}
        s = format_refined_index(refined, num_hard=0)
        assert "η" not in s

    def test_single_hard_edge(self):
        """q * η_0 term."""
        refined: RefinedIndexResult = {(2, 2): 1}
        s = format_refined_index(refined, num_hard=1)
        assert "η" in s or "q" in s  # something meaningful rendered

    def test_inverse_fugacity(self):
        """Negative η exponent term should render without crash."""
        refined: RefinedIndexResult = {(4, -2): 1}
        s = format_refined_index(refined, num_hard=1)
        assert "-2" in s  # η^(-2·v_0)

    def test_custom_eta_vars(self):
        """eta_vars is ignored in the current convention; output uses v_a labels."""
        refined: RefinedIndexResult = {(2, 2): 1}
        s = format_refined_index(refined, num_hard=1, eta_vars=["x"])
        assert "v_0" in s  # new convention always uses v_a

    def test_two_hard_edges(self):
        """Multiple fugacity variables render without crash."""
        refined: RefinedIndexResult = {(2, 2, -4): 1}
        s = format_refined_index(refined, num_hard=2)
        assert "v_0" in s or "v_1" in s or "q" in s
