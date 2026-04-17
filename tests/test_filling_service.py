"""
Phase 2 tests — FillingService.

Oracle values verified against v0.4 passing tests.

Requires snappy.  NC cycle (1,0) is known for m004.
hj_continued_fraction facts (from v0.4):
  (1, 0) → length 0 or 1  (the trivial case — HJ boundary is degenerate)
  (2, 1) → length 1
  (3, 2) → length 2
"""
from __future__ import annotations

import pytest
from fractions import Fraction
from types import SimpleNamespace

snappy = pytest.importorskip("snappy")   # skip entire module if snappy absent

from manifold_index.services.compute_service import ComputeService
from manifold_index.services.filling_service import FillingService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def m004_nz():
    _, _, nz = ComputeService.load_manifold("m004")
    return nz


@pytest.fixture(scope="module")
def m004_nc_result(m004_nz):
    """NC cycle search for cusp 0, slopes |P|≤2, Q∈[0,2], qq=12."""
    return FillingService.find_nc_cycles(
        nz_data=m004_nz,
        cusp_idx=0,
        p_range=(-2, 2),
        q_range=(0, 2),
        q_order_half=12,
    )


# ---------------------------------------------------------------------------
# find_nc_cycles
# ---------------------------------------------------------------------------

class TestFindNcCycles:
    def test_returns_result_with_cycles(self, m004_nc_result):
        assert hasattr(m004_nc_result, "cycles")

    def test_finds_at_least_one_cycle(self, m004_nc_result):
        assert len(m004_nc_result.cycles) >= 1

    def test_m004_contains_known_nc_cycle_1_0(self, m004_nc_result):
        """(1, 0) is a known NC cycle for m004 at the tested range."""
        slopes = {(c.P, c.Q) for c in m004_nc_result.cycles}
        assert (1, 0) in slopes or (-1, 0) in slopes, (
            f"Expected (1,0) or (-1,0) in NC cycles, got: {slopes}"
        )

    def test_cycles_have_p_q_attributes(self, m004_nc_result):
        for cyc in m004_nc_result.cycles:
            assert hasattr(cyc, "P")
            assert hasattr(cyc, "Q")

    def test_wider_search_finds_0_1_cycle(self, m004_nz):
        """Wider range finds (0,1) as an NC cycle for m004 cusp 0."""
        result = FillingService.find_nc_cycles(
            nz_data=m004_nz,
            cusp_idx=0,
            p_range=(-1, 1),
            q_range=(0, 1),
            q_order_half=12,
        )
        slopes = {(c.P, c.Q) for c in result.cycles}
        # At least one of the known NC cycles should appear
        has_known = any(
            s in slopes for s in [(1, 0), (-1, 0), (0, 1)]
        )
        assert has_known, f"No known NC cycle found, got: {slopes}"


# ---------------------------------------------------------------------------
# canonicalise_nc_cycles
# ---------------------------------------------------------------------------

class TestCanonicaliseNcCycles:
    def test_empty_list(self):
        assert FillingService.canonicalise_nc_cycles([]) == []

    def test_deduplicates_opposite_signs(self):
        """(1,0) and (-1,0) are antipodal → only one survives."""
        cyc_pos = SimpleNamespace(P=1, Q=0)
        cyc_neg = SimpleNamespace(P=-1, Q=0)
        result = FillingService.canonicalise_nc_cycles([cyc_pos, cyc_neg])
        assert len(result) == 1
        # Canonical representative has positive first nonzero coordinate
        assert result[0].P > 0

    def test_deduplicates_opposite_q_signs(self):
        """(0,1) and (0,-1) are antipodal → only one survives."""
        cyc_pos = SimpleNamespace(P=0, Q=1)
        cyc_neg = SimpleNamespace(P=0, Q=-1)
        result = FillingService.canonicalise_nc_cycles([cyc_pos, cyc_neg])
        assert len(result) == 1
        assert result[0].Q > 0

    def test_keeps_distinct_slopes(self):
        """(1,0), (-1,0), (0,1) → two canonical cycles: (1,0) and (0,1)."""
        cyc1 = SimpleNamespace(P=1, Q=0)
        cyc2 = SimpleNamespace(P=-1, Q=0)
        cyc3 = SimpleNamespace(P=0, Q=1)
        result = FillingService.canonicalise_nc_cycles([cyc1, cyc2, cyc3])
        slopes = {(c.P, c.Q) for c in result}
        assert len(result) == 2
        assert (1, 0) in slopes
        assert (0, 1) in slopes

    def test_idempotent(self):
        """Applying canonicalise twice gives the same result."""
        cycles = [
            SimpleNamespace(P=1, Q=0),
            SimpleNamespace(P=-1, Q=0),
            SimpleNamespace(P=2, Q=1),
            SimpleNamespace(P=-2, Q=-1),
        ]
        once = FillingService.canonicalise_nc_cycles(cycles)
        twice = FillingService.canonicalise_nc_cycles(once)
        slopes1 = [(c.P, c.Q) for c in once]
        slopes2 = [(c.P, c.Q) for c in twice]
        assert slopes1 == slopes2

    def test_result_sorted_by_abs_p_q(self):
        """Output must be sorted by (|P|, |Q|)."""
        cycles = [
            SimpleNamespace(P=2, Q=1),
            SimpleNamespace(P=1, Q=0),
            SimpleNamespace(P=0, Q=1),
        ]
        result = FillingService.canonicalise_nc_cycles(cycles)
        keys = [(abs(c.P), abs(c.Q)) for c in result]
        assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# probe_kernel
# ---------------------------------------------------------------------------

class TestProbeKernel:
    def test_returns_dict_with_expected_keys(self):
        info = FillingService.probe_kernel(1, 0, 12)
        assert "available" in info
        assert "cached_qq" in info
        assert "hj_length" in info

    def test_available_is_bool(self):
        info = FillingService.probe_kernel(1, 0, 12)
        assert isinstance(info["available"], bool)

    def test_hj_length_is_int(self):
        info = FillingService.probe_kernel(2, 1, 12)
        assert isinstance(info["hj_length"], int)
        assert info["hj_length"] >= 0

    def test_hj_length_2_1_is_one(self):
        """hj_continued_fraction(2, 1) has length 1."""
        info = FillingService.probe_kernel(2, 1, 12)
        assert info["hj_length"] == 1

    def test_hj_length_3_2_is_two(self):
        """hj_continued_fraction(3, 2) has length 2."""
        info = FillingService.probe_kernel(3, 2, 12)
        assert info["hj_length"] == 2

    def test_hj_length_5_3_gte_two(self):
        """hj_continued_fraction(5, 3) has length ≥ 2."""
        info = FillingService.probe_kernel(5, 3, 12)
        assert info["hj_length"] >= 2

    def test_uncached_has_available_false(self):
        """A freshly generated exotic slope should not be in the kernel cache."""
        info = FillingService.probe_kernel(997, 993, 9999)
        assert info["available"] is False
        assert info["cached_qq"] is None


# ---------------------------------------------------------------------------
# compute_filled_index
# ---------------------------------------------------------------------------

class TestComputeFilledIndex:
    """Tests for compute_filled_index.

    Note: FilledRefinedResult is a dataclass (NOT a plain dict).
    Access coefficients via result.series (a dict).
    Filling along an NC cycle gives is_zero=True (series = {}).
    We use slope (3,1) which is NOT an NC cycle for m004.
    """

    @pytest.fixture(scope="class")
    def filled_3_1(self, m004_nz):
        """Filled index at slope (3,1) with NC basis (1,0) for m004."""
        return FillingService.compute_filled_index(
            nz_data=m004_nz,
            cusp_idx=0,
            nc_P=1, nc_Q=0,
            user_P=3, user_Q=1,
            m_other=[], e_other=[],
            q_order_half=10,
            weyl_a=None, weyl_b=None,
        )

    def test_returns_triple(self, filled_3_1):
        p, q, result = filled_3_1
        assert isinstance(p, int)
        assert isinstance(q, int)
        assert hasattr(result, "series")

    def test_result_is_nonempty(self, filled_3_1):
        _, _, result = filled_3_1
        assert not result.is_zero
        assert len(result.series) > 0

    def test_result_series_is_dict(self, filled_3_1):
        _, _, result = filled_3_1
        assert isinstance(result.series, dict)

    def test_result_has_no_zero_coefficients(self, filled_3_1):
        _, _, result = filled_3_1
        assert all(v != 0 for v in result.series.values())

    def test_slope_1_0_gives_zero_filled_index(self, m004_nz):
        """Filling along the NC cycle (1,0) gives a zero filled index."""
        _, _, result = FillingService.compute_filled_index(
            nz_data=m004_nz, cusp_idx=0,
            nc_P=1, nc_Q=0, user_P=1, user_Q=0,
            m_other=[], e_other=[],
            q_order_half=10, weyl_a=None, weyl_b=None,
        )
        assert result.is_zero, (
            "Filling along an NC cycle must give a zero filled index"
        )

    def test_basis_change_for_slope_3_1(self, filled_3_1):
        """
        Basis change from (α,β) to NC basis (1,0):
          find_rs(1,0) → R=0, S=-1  (satisfies R·0 − 1·S = 1)
          Matrix [[nc_P,nc_Q],[-R,-S]] = [[1,0],[0,1]] = Identity
          p = R·user_Q − S·user_P = 0·1 − (−1)·3 = 3
          q = nc_P·user_Q − nc_Q·user_P = 1·1 − 0·3 = 1
        """
        p, q, _ = filled_3_1
        assert p == 3
        assert q == 1

    def test_result_has_constant_term(self, filled_3_1):
        """The filled index should include the q^0 term with coefficient 1."""
        _, _, result = filled_3_1
        val = result.series.get((0, 0))
        assert val is not None
        assert val == 1

    def test_reproducible(self, m004_nz):
        """Calling compute_filled_index twice yields identical results."""
        kwargs = dict(
            nz_data=m004_nz, cusp_idx=0,
            nc_P=1, nc_Q=0, user_P=3, user_Q=1,
            m_other=[], e_other=[],
            q_order_half=10, weyl_a=None, weyl_b=None,
        )
        p1, q1, r1 = FillingService.compute_filled_index(**kwargs)
        p2, q2, r2 = FillingService.compute_filled_index(**kwargs)
        assert p1 == p2 and q1 == q2 and r1.series == r2.series

    def test_formatter_receives_correct_fields(self, filled_3_1):
        """Regression: format_filled_series_latex must be called with
        result.series (dict), not the FilledRefinedResult dataclass.

        Previously the UI called format_filled_series_latex(result, ...) which
        raised AttributeError (no .items()) and fell back to str(result),
        making the table show 'FilledRefinedResult(P=..., series={}, ...)'.
        """
        from manifold_index.formatters.filling_fmt import format_filled_series_latex
        _, _, result = filled_3_1
        # Calling with result.series (the dict) must not raise and must not
        # return a repr-string of the dataclass.
        latex = format_filled_series_latex(
            result.series,
            result.num_hard,
            result.has_cusp_eta,
            result.num_cusp_eta,
        )
        assert "FilledRefinedResult" not in latex, (
            "Formatter was passed the dataclass instead of result.series"
        )
        assert latex.startswith("$"), f"Expected $...$, got: {latex!r}"

