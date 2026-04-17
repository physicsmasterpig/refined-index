"""
Phase 2 tests — ComputeService.

Key encoding (from refined_index.py):
    Keys are  (q_half_power, 2*η_0_exp, 2*η_1_exp, ...)
    For m004 (num_hard=1): length-2 tuples.
    Example: (0, 0) → q^0 η^0,  (1, 2) → q^{1/2} η,  (2, 0) → q η^0

Requires snappy.  All oracle values verified against v0.4 passing tests.
"""
from __future__ import annotations

import pytest
from fractions import Fraction

snappy = pytest.importorskip("snappy")   # skip entire module if snappy absent

from manifold_index.services.compute_service import ComputeService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def m004_triple():
    """(ManifoldData, EasyEdgeResult, NeumannZagierData) for m004."""
    return ComputeService.load_manifold("m004")


@pytest.fixture(scope="module")
def m004_nz(m004_triple):
    return m004_triple[2]


@pytest.fixture(scope="module")
def m004_index_result(m004_nz):
    """I^ref(0,0) for m004 at qq=20  (m_ext=[0], e_ext=[0])."""
    return ComputeService.compute_refined_index(m004_nz, [0], [Fraction(0)], 20)


# ---------------------------------------------------------------------------
# load_manifold
# ---------------------------------------------------------------------------

class TestLoadManifold:
    def test_returns_three_objects(self, m004_triple):
        manifold_data, easy_result, nz_data = m004_triple
        assert manifold_data is not None
        assert easy_result is not None
        assert nz_data is not None

    def test_manifold_name(self, m004_triple):
        manifold_data = m004_triple[0]
        assert manifold_data.name == "m004"

    def test_nz_tetrahedra_count(self, m004_triple):
        """m004 has 2 tetrahedra → nz.n == 2."""
        _, _, nz_data = m004_triple
        assert nz_data.n == 2

    def test_nz_cusp_count(self, m004_triple):
        """m004 has 1 cusp → nz.r == 1."""
        _, _, nz_data = m004_triple
        assert nz_data.r == 1

    def test_num_hard_edges(self, m004_triple):
        """m004 has exactly 1 hard edge (n - r = 2 - 1 = 1)."""
        _, _, nz_data = m004_triple
        assert nz_data.num_hard == 1

    def test_easy_result_attributes(self, m004_triple):
        _, easy_result, _ = m004_triple
        assert hasattr(easy_result, "all_easy")
        assert hasattr(easy_result, "hard_padding")

    def test_nonexistent_raises_value_error(self):
        with pytest.raises(ValueError):
            ComputeService.load_manifold("no_such_manifold_zzz")


# ---------------------------------------------------------------------------
# probe_cache
# ---------------------------------------------------------------------------

class TestProbeCache:
    def test_returns_dict_with_three_keys(self, m004_nz):
        result = ComputeService.probe_cache("m004", m004_nz)
        assert isinstance(result, dict)
        assert {"iref", "nc", "kernels"} == set(result.keys())

    def test_iref_section_structure(self, m004_nz):
        iref = ComputeService.probe_cache("m004", m004_nz)["iref"]
        assert "available" in iref
        assert "qq_order" in iref
        assert "m_range" in iref
        assert isinstance(iref["available"], bool)

    def test_nc_section_structure(self, m004_nz):
        nc = ComputeService.probe_cache("m004", m004_nz)["nc"]
        assert "available" in nc
        assert "qq_order" in nc
        assert "p_range" in nc
        assert isinstance(nc["available"], bool)

    def test_kernels_section_structure(self, m004_nz):
        kernels = ComputeService.probe_cache("m004", m004_nz)["kernels"]
        assert "available" in kernels
        assert "count" in kernels
        assert "qq_orders" in kernels
        assert isinstance(kernels["count"], int)
        assert isinstance(kernels["qq_orders"], list)
        assert kernels["count"] >= 0


# ---------------------------------------------------------------------------
# compute_refined_index
# ---------------------------------------------------------------------------

class TestComputeRefinedIndex:
    def test_returns_dict(self, m004_index_result):
        assert isinstance(m004_index_result, dict)

    def test_keys_are_length_two_tuples(self, m004_index_result):
        """m004: num_hard=1 → every key is (q_half_pow, 2*eta_exp)."""
        for key in m004_index_result.keys():
            assert isinstance(key, tuple)
            assert len(key) == 2, f"Expected length-2 key, got {key!r}"

    def test_constant_term_is_one(self, m004_index_result):
        """Leading term q^0 η^0 → key (0, 0) with coefficient 1."""
        assert m004_index_result.get((0, 0)) == 1

    def test_no_zero_coefficients(self, m004_index_result):
        """Result must be compact (no stored zeros)."""
        assert all(v != 0 for v in m004_index_result.values())

    def test_q_first_order_eta_zero(self, m004_index_result):
        """
        Oracle (q^1, η^0 term): coefficient −2  (m004 I^ref(0,0) at q^1 is −2q).
          key (2, 0) → −2
        """
        assert m004_index_result.get((2, 0)) == -2

    def test_eta_terms_appear_at_q3(self, m004_index_result):
        """
        Oracle: first η-nonzero terms appear at q^3 (q_half_pow=6).
          key (6,  2) → 1    [q^3 η ]
          key (6, -2) → 1    [q^3 η⁻¹]
        No η-charged terms exist below q^3 at (m=0, e=0).
        """
        r = m004_index_result
        assert r.get((6, 2)) == 1
        assert r.get((6, -2)) == 1
        # Confirm no spurious half-integer q_half_pow keys
        assert (1, 2) not in r
        assert (1, -2) not in r

    def test_truncation_respects_q_order_half(self, m004_nz):
        """No key should exceed the requested q_order_half truncation."""
        for qq in (8, 12, 20):
            result = ComputeService.compute_refined_index(m004_nz, [0], [Fraction(0)], qq)
            for key in result:
                assert key[0] <= qq, f"Key {key!r} exceeds q_order_half={qq}"

    def test_result_is_reproducible(self, m004_nz):
        """Calling compute twice with the same args must return equal dicts."""
        r1 = ComputeService.compute_refined_index(m004_nz, [0], [Fraction(0)], 12)
        r2 = ComputeService.compute_refined_index(m004_nz, [0], [Fraction(0)], 12)
        assert r1 == r2


# ---------------------------------------------------------------------------
# project_refined_index
# ---------------------------------------------------------------------------

class TestProjectRefinedIndex:
    def test_all_active_returns_copy(self, m004_index_result):
        """[True] → shallow copy with identical content."""
        projected = ComputeService.project_refined_index(m004_index_result, [True])
        assert projected == m004_index_result
        assert projected is not m004_index_result

    def test_all_active_does_not_mutate_input(self, m004_index_result):
        original_len = len(m004_index_result)
        ComputeService.project_refined_index(m004_index_result, [True])
        assert len(m004_index_result) == original_len

    def test_all_inactive_collapses_eta_exponent_to_zero(self, m004_index_result):
        """[False] → every key must have eta_exp part == 0."""
        projected = ComputeService.project_refined_index(m004_index_result, [False])
        assert all(k[1] == 0 for k in projected.keys()), (
            "After projecting with [False], all eta exponents should be 0"
        )

    def test_all_inactive_sums_q3_eta(self, m004_index_result):
        """
        q^3 terms (q_half_pow=6): (6,2)→1 and (6,-2)→1 → projected (6,0)→2.
        """
        projected = ComputeService.project_refined_index(m004_index_result, [False])
        assert projected.get((6, 0)) == 2

    def test_all_inactive_q1_unchanged(self, m004_index_result):
        """
        q^1 term (2,0)→-2 already has η=0; projection leaves it unchanged.
        """
        projected = ComputeService.project_refined_index(m004_index_result, [False])
        assert projected.get((2, 0)) == -2

    def test_all_inactive_constant_term_unchanged(self, m004_index_result):
        """Constant term: (0,0)→1 unaffected by summing η."""
        projected = ComputeService.project_refined_index(m004_index_result, [False])
        assert projected.get((0, 0)) == 1

    def test_all_inactive_does_not_mutate_input(self, m004_index_result):
        original = dict(m004_index_result)
        ComputeService.project_refined_index(m004_index_result, [False])
        assert m004_index_result == original

    def test_cancellation_drops_zero_terms(self):
        """Terms that cancel to 0 after projection must be absent in result."""
        result = {(2, 2): 1, (2, -2): -1}
        projected = ComputeService.project_refined_index(result, [False])
        assert (2, 0) not in projected

    def test_partial_project_two_hard_edges(self):
        """Partial projection: keep first η, collapse second η."""
        result = {
            (0, 2, 2): 1,
            (0, 2, -2): 3,
            (0, -2, 2): 5,
        }
        # active_edges = [True, False]: keep η_0, collapse η_1
        projected = ComputeService.project_refined_index(result, [True, False])
        # (0,2,2)+(0,2,-2) → (0,2,0) with coeff 4; (0,-2,2) → (0,-2,0) with coeff 5
        assert projected == {(0, 2, 0): 4, (0, -2, 0): 5}

    def test_all_inactive_matches_project_to_3d_index(self, m004_nz):
        """project_refined_index([False]) must match project_to_3d_index from core."""
        from manifold_index.core.refined_index import project_to_3d_index
        result = ComputeService.compute_refined_index(m004_nz, [0], [Fraction(0)], 12)
        our_proj = ComputeService.project_refined_index(result, [False])
        core_3d = project_to_3d_index(result)  # dict[q_half_pow → coeff]
        # our_proj keys: (q_half_pow, 0) → coeff; core_3d keys: q_half_pow → coeff
        our_flat = {k[0]: v for k, v in our_proj.items()}
        assert our_flat == core_3d


# ---------------------------------------------------------------------------
# run_weyl_check  (light tests — not asserting exact a/b values at low qq)
# ---------------------------------------------------------------------------

class TestRunWeylCheck:
    @pytest.fixture(scope="class")
    def weyl_entries(self, m004_nz):
        """Build a small set of entries at qq=12 for m004."""
        qq = 12
        return [
            ([m], [Fraction(e, 2)],
             ComputeService.compute_refined_index(m004_nz, [m], [Fraction(e, 2)], qq))
            for m in range(-3, 4)
            for e in range(-6, 7)
        ]

    def test_returns_ab_or_none(self, weyl_entries):
        ab, adj_pass, adj_value = ComputeService.run_weyl_check(weyl_entries, num_hard=1, q_order_half=12)
        assert ab is None or (hasattr(ab, "a") and hasattr(ab, "b"))
        assert adj_pass is None or isinstance(adj_pass, bool)
        assert adj_value is None or isinstance(adj_value, int)

    def test_ab_has_correct_length(self, weyl_entries):
        ab, _adj, _adjv = ComputeService.run_weyl_check(weyl_entries, num_hard=1, q_order_half=12)
        if ab is not None:
            assert len(ab.a) == 1  # num_hard = 1
            assert len(ab.b) == 1

    def test_ab_values_are_rational(self, weyl_entries):
        ab, _adj, _adjv = ComputeService.run_weyl_check(weyl_entries, num_hard=1, q_order_half=12)
        if ab is not None:
            for av in ab.a:
                Fraction(av)   # must be rational
            for bv in ab.b:
                Fraction(bv)
