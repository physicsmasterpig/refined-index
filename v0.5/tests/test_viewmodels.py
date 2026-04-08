"""
Phase 3 tests — ViewModel builders.

Tests for ManifoldViewModel, IndexViewModel, WeylViewModel,
FillingViewModel, and ExportViewModel.

SnaPy-dependent tests are marked with ``@pytest.mark.skipif`` / importorskip.
Pure-data tests run without snappy.
"""
from __future__ import annotations

import pytest
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import MagicMock

from manifold_index.viewmodels.advisory import Advisory, AdvisoryLevel, Advisories
from manifold_index.viewmodels.manifold_vm import (
    ManifoldViewModel,
    build_manifold_vm,
    build_manifold_vm_error,
)
from manifold_index.viewmodels.index_vm import (
    IndexQueryViewModel,
    WeylViewModel,
    IndexViewModel,
    build_index_query_vm,
    build_weyl_vm,
    build_index_vm,
)
from manifold_index.viewmodels.filling_vm import (
    NCCycleViewModel,
    FillQueryViewModel,
    FillingViewModel,
    build_nc_cycle_vm,
    build_fill_query_vm,
    build_filling_vm,
)
from manifold_index.viewmodels.export_vm import (
    ExportAvailability,
    ExportFormatSelection,
    ExportViewModel,
    build_export_vm,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_nz(n=2, r=1, num_hard=1):
    """Minimal NeumannZagierData-like stub."""
    nz = SimpleNamespace(n=n, r=r, num_hard=num_hard)
    return nz


def _make_manifold_data(name="m004"):
    return SimpleNamespace(name=name)


def _make_easy_result(all_easy=False, hard_padding=0):
    return SimpleNamespace(all_easy=all_easy, hard_padding=hard_padding)


EMPTY_CACHE_STATUS = {
    "iref": {"available": False, "qq_order": None, "m_range": None},
    "nc":   {"available": False, "qq_order": None, "p_range": None},
    "kernels": {"available": False, "count": 0, "qq_orders": []},
}


# ---------------------------------------------------------------------------
# ManifoldViewModel
# ---------------------------------------------------------------------------

class TestManifoldViewModel:
    def test_num_hard_one_no_A1(self):
        """num_hard=1 → has_hard_edges=True, no A1 advisory."""
        vm = build_manifold_vm(
            _make_manifold_data("m004"),
            _make_easy_result(),
            _make_nz(n=2, r=1, num_hard=1),
            EMPTY_CACHE_STATUS,
        )
        assert vm.has_hard_edges is True
        assert vm.index_title == "Refined Index"
        ids = [a.advisory_id for a in vm.advisories]
        assert "A1" not in ids

    def test_num_hard_zero_has_A1(self):
        """num_hard=0 → has_hard_edges=False, advisory A1 present."""
        vm = build_manifold_vm(
            _make_manifold_data("m003"),
            _make_easy_result(all_easy=True),
            _make_nz(n=2, r=1, num_hard=0),
            EMPTY_CACHE_STATUS,
        )
        assert vm.has_hard_edges is False
        assert vm.index_title == "3D Index"
        ids = [a.advisory_id for a in vm.advisories]
        assert "A1" in ids

    def test_A1_is_info_level(self):
        vm = build_manifold_vm(
            _make_manifold_data("m003"),
            _make_easy_result(all_easy=True),
            _make_nz(n=2, r=1, num_hard=0),
            EMPTY_CACHE_STATUS,
        )
        a1 = next(a for a in vm.advisories if a.advisory_id == "A1")
        assert a1.level == AdvisoryLevel.INFO

    def test_name_stored(self):
        vm = build_manifold_vm(
            _make_manifold_data("m125"),
            _make_easy_result(),
            _make_nz(n=5, r=1, num_hard=2),
            EMPTY_CACHE_STATUS,
        )
        assert vm.name == "m125"

    def test_n_tetrahedra_and_cusps(self):
        vm = build_manifold_vm(
            _make_manifold_data("m004"),
            _make_easy_result(),
            _make_nz(n=2, r=1, num_hard=1),
            EMPTY_CACHE_STATUS,
        )
        assert vm.n_tetrahedra == 2
        assert vm.n_cusps == 1

    def test_cache_status_stored(self):
        vm = build_manifold_vm(
            _make_manifold_data("m004"),
            _make_easy_result(),
            _make_nz(),
            EMPTY_CACHE_STATUS,
        )
        assert vm.cache_status is EMPTY_CACHE_STATUS

    def test_error_vm_has_A2(self):
        vm = build_manifold_vm_error("nonexistent_xyz", ValueError("not found"))
        ids = [a.advisory_id for a in vm.advisories]
        assert "A2" in ids

    def test_error_vm_A2_contains_name(self):
        vm = build_manifold_vm_error("badname", ValueError())
        a2 = next(a for a in vm.advisories if a.advisory_id == "A2")
        assert "badname" in a2.body

    def test_error_vm_is_error_level(self):
        vm = build_manifold_vm_error("x", Exception())
        a2 = next(a for a in vm.advisories if a.advisory_id == "A2")
        assert a2.level == AdvisoryLevel.ERROR

    def test_error_vm_zero_tetrahedra(self):
        vm = build_manifold_vm_error("x", Exception())
        assert vm.n_tetrahedra == 0


# ---------------------------------------------------------------------------
# IndexQueryViewModel
# ---------------------------------------------------------------------------

class TestIndexQueryViewModel:
    def test_nonzero_result_no_B1(self):
        result = {(0, 0): 1, (2, 0): -2}
        vm = build_index_query_vm([0], [Fraction(0)], 20, [True], result)
        assert vm.is_zero is False
        assert not any(a.advisory_id == "B1" for a in vm.advisories)

    def test_zero_result_has_B1(self):
        result = {}
        vm = build_index_query_vm([0], [Fraction(0)], 20, [True], result)
        assert vm.is_zero is True
        ids = [a.advisory_id for a in vm.advisories]
        assert "B1" in ids

    def test_B1_contains_qq(self):
        vm = build_index_query_vm([0], [Fraction(0)], 30, [True], {})
        b1 = next(a for a in vm.advisories if a.advisory_id == "B1")
        assert "30" in b1.body

    def test_m_ext_stored(self):
        vm = build_index_query_vm([1, -1], [Fraction(0), Fraction(0)], 12, [True, False], {(0, 0): 1})
        assert vm.m_ext == [1, -1]

    def test_active_edges_stored(self):
        vm = build_index_query_vm([0], [Fraction(0)], 12, [False], {(0, 0): 1})
        assert vm.active_edges == [False]

    def test_source_stored(self):
        vm = build_index_query_vm([0], [Fraction(0)], 12, [True], {(0, 0): 1},
                                  source="cache")
        assert vm.source == "cache"

    def test_timestamp_is_float(self):
        vm = build_index_query_vm([0], [Fraction(0)], 12, [True], {(0, 0): 1})
        assert isinstance(vm.timestamp, float)


# ---------------------------------------------------------------------------
# WeylViewModel
# ---------------------------------------------------------------------------

class TestWeylViewModel:
    def test_none_result_has_B3(self):
        """ab_result=None → advisory B3 present."""
        vm = build_weyl_vm(None, num_hard=1)
        assert vm.checked is True
        ids = [a.advisory_id for a in vm.advisories]
        assert "B3" in ids

    def test_none_result_not_compatible(self):
        vm = build_weyl_vm(None, num_hard=1)
        assert vm.is_fully_compatible is False
        assert vm.a_vectors == []
        assert vm.b_vectors == []

    def test_valid_result_no_B3(self):
        """Valid ABVectors → no B3."""
        ab = SimpleNamespace(a=[Fraction(2)], b=[Fraction(1, 2)])
        vm = build_weyl_vm(ab, num_hard=1)
        assert not any(a.advisory_id == "B3" for a in vm.advisories)

    def test_valid_result_fully_compatible(self):
        """b=[1/2] is half-integer → compatible."""
        ab = SimpleNamespace(a=[Fraction(2)], b=[Fraction(1, 2)])
        vm = build_weyl_vm(ab, num_hard=1)
        assert vm.is_fully_compatible is True
        assert vm.edge_compatible == [True]

    def test_incompatible_edge_has_B4(self):
        """b=[1/3] is not half-integer → B4 advisory."""
        ab = SimpleNamespace(a=[Fraction(2)], b=[Fraction(1, 3)])
        vm = build_weyl_vm(ab, num_hard=1)
        ids = [a.advisory_id for a in vm.advisories]
        assert "B4" in ids
        assert vm.is_fully_compatible is False

    def test_B4_lists_incompatible_index(self):
        ab = SimpleNamespace(a=[Fraction(2), Fraction(2)],
                             b=[Fraction(1, 2), Fraction(1, 3)])
        vm = build_weyl_vm(ab, num_hard=2)
        b4 = next(a for a in vm.advisories if a.advisory_id == "B4")
        assert "1" in b4.body   # edge index 1 is listed

    def test_a_b_vectors_stored(self):
        ab = SimpleNamespace(a=[Fraction(2)], b=[Fraction(1, 2)])
        vm = build_weyl_vm(ab, num_hard=1)
        assert vm.a_vectors == [Fraction(2)]
        assert vm.b_vectors == [Fraction(1, 2)]

    def test_adjoint_failure_has_B5(self):
        ab = SimpleNamespace(a=[Fraction(2)], b=[Fraction(1, 2)])
        vm = build_weyl_vm(ab, num_hard=1, adjoint_value=-0.5, adjoint_passed=False)
        ids = [a.advisory_id for a in vm.advisories]
        assert "B5" in ids

    def test_adjoint_pass_no_B5(self):
        ab = SimpleNamespace(a=[Fraction(2)], b=[Fraction(1, 2)])
        vm = build_weyl_vm(ab, num_hard=1, adjoint_value=-1.0, adjoint_passed=True)
        assert not any(a.advisory_id == "B5" for a in vm.advisories)


# ---------------------------------------------------------------------------
# IndexViewModel (top-level)
# ---------------------------------------------------------------------------

class TestIndexViewModel:
    def test_no_queries_no_advisories(self):
        vm = build_index_vm([], None)
        assert vm.queries == []
        assert vm.weyl_status is None
        assert vm.advisories == []

    def test_queries_forwarded(self):
        q = build_index_query_vm([0], [Fraction(0)], 12, [True], {(0, 0): 1})
        vm = build_index_vm([q], None)
        assert len(vm.queries) == 1
        assert vm.queries[0] is q


# ---------------------------------------------------------------------------
# NCCycleViewModel
# ---------------------------------------------------------------------------

class TestNCCycleViewModel:
    def test_basic_fields(self):
        vm = build_nc_cycle_vm(cusp_idx=0, P=1, Q=0)
        assert vm.cusp_idx == 0
        assert vm.P == 1
        assert vm.Q == 0
        assert vm.source == "computed"

    def test_slope_latex_provided(self):
        vm = build_nc_cycle_vm(0, 1, 0, slope_latex=r"$\alpha$")
        assert vm.slope_latex == r"$\alpha$"

    def test_slope_latex_fallback_generated(self):
        """When no slope_latex provided, a non-empty fallback is generated."""
        vm = build_nc_cycle_vm(0, 1, 0)
        assert vm.slope_latex != ""

    def test_weyl_compatible_stored(self):
        vm = build_nc_cycle_vm(0, 1, 0, weyl_compatible=True)
        assert vm.weyl_compatible is True

    def test_weyl_compatible_none_default(self):
        vm = build_nc_cycle_vm(0, 1, 0)
        assert vm.weyl_compatible is None


# ---------------------------------------------------------------------------
# FillQueryViewModel
# ---------------------------------------------------------------------------

class TestFillQueryViewModel:
    def _make_result(self, is_zero=False):
        r = SimpleNamespace(is_zero=is_zero, series={} if is_zero else {(0, 0): 1})
        return r

    def test_nonempty_result_not_zero(self):
        vm = build_fill_query_vm(1, 0, 3, 1, p=-3, q=1,
                                 m_other=[], e_other=[],
                                 result=self._make_result(is_zero=False))
        assert vm.is_zero is False

    def test_empty_result_is_zero(self):
        vm = build_fill_query_vm(1, 0, 1, 0, p=-1, q=0,
                                 m_other=[], e_other=[],
                                 result=self._make_result(is_zero=True))
        assert vm.is_zero is True

    def test_p_zero_has_C2(self):
        """p=0 in NC basis → advisory C2."""
        vm = build_fill_query_vm(1, 0, 0, 1, p=0, q=1,
                                 m_other=[], e_other=[],
                                 result=self._make_result())
        ids = [a.advisory_id for a in vm.advisories]
        assert "C2" in ids

    def test_p_nonzero_no_C2(self):
        vm = build_fill_query_vm(1, 0, 3, 1, p=-3, q=1,
                                 m_other=[], e_other=[],
                                 result=self._make_result())
        assert not any(a.advisory_id == "C2" for a in vm.advisories)

    def test_p_q_stored(self):
        vm = build_fill_query_vm(1, 0, 3, 1, p=-3, q=1,
                                 m_other=[], e_other=[],
                                 result=self._make_result())
        assert vm.p == -3
        assert vm.q == 1

    def test_weyl_latex_none_when_not_given(self):
        vm = build_fill_query_vm(1, 0, 3, 1, p=-3, q=1,
                                 m_other=[], e_other=[],
                                 result=self._make_result())
        assert vm.weyl_a_latex is None
        assert vm.weyl_b_latex is None

    def test_weyl_latex_formatted_when_given(self):
        vm = build_fill_query_vm(1, 0, 3, 1, p=-3, q=1,
                                 m_other=[], e_other=[],
                                 result=self._make_result(),
                                 weyl_a=[Fraction(2)],
                                 weyl_b=[Fraction(1, 2)])
        assert vm.weyl_a_latex is not None
        assert vm.weyl_b_latex is not None

    def test_timestamp_is_float(self):
        vm = build_fill_query_vm(1, 0, 3, 1, p=-3, q=1,
                                 m_other=[], e_other=[],
                                 result=self._make_result())
        assert isinstance(vm.timestamp, float)


# ---------------------------------------------------------------------------
# FillingViewModel (top-level)
# ---------------------------------------------------------------------------

class TestFillingViewModel:
    def test_empty_nc_cycles_has_C1(self):
        """No NC cycles → card advisory C1 present."""
        vm = build_filling_vm(nc_cycles=[], fill_queries=[])
        ids = [a.advisory_id for a in vm.advisories]
        assert "C1" in ids

    def test_nonempty_nc_cycles_no_C1(self):
        nc = build_nc_cycle_vm(0, 1, 0)
        vm = build_filling_vm(nc_cycles=[nc], fill_queries=[])
        assert not any(a.advisory_id == "C1" for a in vm.advisories)

    def test_nc_cycles_forwarded(self):
        nc = build_nc_cycle_vm(0, 1, 0)
        vm = build_filling_vm([nc], [])
        assert len(vm.nc_cycles) == 1

    def test_fill_queries_forwarded(self):
        r = SimpleNamespace(is_zero=False, series={(0, 0): 1})
        fq = build_fill_query_vm(1, 0, 3, 1, -3, 1, [], [], r)
        nc = build_nc_cycle_vm(0, 1, 0)
        vm = build_filling_vm([nc], [fq])
        assert len(vm.fill_queries) == 1


# ---------------------------------------------------------------------------
# ExportViewModel
# ---------------------------------------------------------------------------

class TestExportViewModel:
    def test_unlocked_when_manifold_present(self):
        vm = build_export_vm({"manifold": True, "index_queries": 0,
                              "weyl": False, "nc_cycles": 0, "fill_queries": 0})
        assert vm.is_unlocked is True

    def test_locked_when_nothing(self):
        vm = build_export_vm({"manifold": False, "index_queries": 0,
                              "weyl": False, "nc_cycles": 0, "fill_queries": 0})
        assert vm.is_unlocked is False

    def test_availability_index_count(self):
        vm = build_export_vm({"manifold": True, "index_queries": 3,
                              "weyl": False, "nc_cycles": 0, "fill_queries": 0})
        assert vm.availability.n_index_queries == 3
        assert vm.availability.has_index is True

    def test_availability_no_index(self):
        vm = build_export_vm({"manifold": True, "index_queries": 0,
                              "weyl": False, "nc_cycles": 0, "fill_queries": 0})
        assert vm.availability.has_index is False

    def test_availability_has_filling(self):
        vm = build_export_vm({"manifold": True, "index_queries": 1,
                              "weyl": False, "nc_cycles": 2, "fill_queries": 1})
        assert vm.availability.has_filling is True

    def test_default_format_selection(self):
        vm = build_export_vm({"manifold": True, "index_queries": 1,
                              "weyl": False, "nc_cycles": 0, "fill_queries": 0})
        assert vm.format_selection.latex is True
        assert vm.format_selection.mathematica is True
        assert vm.format_selection.full_report is False
        assert vm.format_selection.json is False

    def test_custom_format_selection(self):
        fmt = ExportFormatSelection(latex=False, mathematica=False,
                                    full_report=True, json=True)
        vm = build_export_vm({"manifold": True, "index_queries": 1,
                              "weyl": False, "nc_cycles": 0, "fill_queries": 0},
                             format_selection=fmt)
        assert vm.format_selection.full_report is True
        assert vm.format_selection.latex is False

    def test_no_advisories_by_default(self):
        vm = build_export_vm({"manifold": True, "index_queries": 1,
                              "weyl": False, "nc_cycles": 0, "fill_queries": 0})
        assert vm.advisories == []

    def test_output_path_stored(self):
        vm = build_export_vm({"manifold": True, "index_queries": 1,
                              "weyl": False, "nc_cycles": 0, "fill_queries": 0},
                             output_path="/tmp/out")
        assert vm.output_path == "/tmp/out"

    def test_has_any_true_with_index(self):
        vm = build_export_vm({"manifold": False, "index_queries": 2,
                              "weyl": False, "nc_cycles": 0, "fill_queries": 0})
        assert vm.availability.has_any is True

    def test_has_any_false_when_empty(self):
        vm = build_export_vm({"manifold": False, "index_queries": 0,
                              "weyl": False, "nc_cycles": 0, "fill_queries": 0})
        assert vm.availability.has_any is False
