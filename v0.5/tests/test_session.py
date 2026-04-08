"""
tests.test_session
==================
Unit tests for services.session — Phase 1.

All tests use plain Python data only; no core/, PySide6, or snappy.
"""

from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from manifold_index.services.session import (
    FillQuery,
    IndexQuery,
    NCCycleSet,
    PipelineStage,
    Session,
    session_from_dict,
    session_to_dict,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_index_query(result=None) -> IndexQuery:
    return IndexQuery(
        m_ext=[1, 0],
        e_ext=[Fraction(1, 2), Fraction(0)],
        q_order_half=20,
        result=result,
        projected_result=None,
        active_edges=[True, True],
        timestamp=0.0,
        source="computed",
    )


def _make_fill_query(result=None) -> FillQuery:
    return FillQuery(
        cusp_idx=0,
        nc_P=3,
        nc_Q=1,
        user_P=2,
        user_Q=1,
        p=1,
        q=0,
        m_other=[],
        e_other=[],
        q_order_half=20,
        result=result,
        weyl_a=None,
        weyl_b=None,
        incompat_edges=[],
        timestamp=0.0,
        source="computed",
    )


def _make_ncs(cusp_idx: int = 0) -> NCCycleSet:
    return NCCycleSet(
        cusp_idx=cusp_idx,
        search_p_range=(-5, 5),
        search_q_range=(-5, 5),
        q_order_half=20,
        cycles=[object(), object()],   # two dummy cycles
        source="computed",
    )


def _loaded_session() -> Session:
    """Session in LOADED state with mock manifold data."""
    s = Session(manifold_name="m004", stage=PipelineStage.LOADED)
    s.manifold_data = MagicMock()
    s.nz_data = SimpleNamespace(num_hard=2)
    s.active_edges = [True, True]
    return s


# ── 1. Construction ───────────────────────────────────────────────────────────

class TestSessionDefaults:
    def test_default_stage_is_empty(self):
        s = Session()
        assert s.stage == PipelineStage.EMPTY

    def test_default_manifold_name_empty(self):
        s = Session()
        assert s.manifold_name == ""

    def test_default_generation_zero(self):
        s = Session()
        assert s.generation == 0

    def test_default_lists_empty(self):
        s = Session()
        assert s.index_queries == []
        assert s.fill_queries == []
        assert s.nc_cycles == []
        assert s.active_edges == []

    def test_default_q_order_half(self):
        s = Session()
        assert s.q_order_half == 20

    def test_default_manifold_data_none(self):
        s = Session()
        assert s.manifold_data is None
        assert s.nz_data is None
        assert s.weyl_result is None

    def test_default_cache_status_dict(self):
        s = Session()
        assert isinstance(s.cache_status, dict)

    def test_independent_lists_across_instances(self):
        """Dataclass mutable defaults must not be shared."""
        s1, s2 = Session(), Session()
        s1.index_queries.append(_make_index_query())
        assert s2.index_queries == []


# ── 2. has_any_results ────────────────────────────────────────────────────────

class TestHasAnyResults:
    def test_empty(self):
        assert not Session().has_any_results()

    def test_loaded(self):
        s = Session(stage=PipelineStage.LOADED)
        assert s.has_any_results()

    def test_indexed(self):
        s = Session(stage=PipelineStage.INDEXED)
        assert s.has_any_results()

    def test_filled(self):
        s = Session(stage=PipelineStage.FILLED)
        assert s.has_any_results()


# ── 3. invalidate_from(LOADED) ────────────────────────────────────────────────

class TestInvalidateFromLoaded:
    def _filled_session(self) -> Session:
        s = _loaded_session()
        s.stage = PipelineStage.FILLED
        s.index_queries.append(_make_index_query(result={(0,): 1}))
        s.fill_queries.append(_make_fill_query(result={(0,): 1}))
        s.nc_cycles.append(_make_ncs())
        s.weyl_result = MagicMock()
        s.weyl_checked = True
        return s

    def test_clears_manifold_data(self):
        s = self._filled_session()
        s.invalidate_from(PipelineStage.LOADED)
        assert s.manifold_data is None

    def test_clears_nz_data(self):
        s = self._filled_session()
        s.invalidate_from(PipelineStage.LOADED)
        assert s.nz_data is None

    def test_clears_all_queries(self):
        s = self._filled_session()
        s.invalidate_from(PipelineStage.LOADED)
        assert s.index_queries == []
        assert s.fill_queries == []
        assert s.nc_cycles == []

    def test_clears_weyl(self):
        s = self._filled_session()
        s.invalidate_from(PipelineStage.LOADED)
        assert s.weyl_result is None
        assert s.weyl_checked is False

    def test_clears_active_edges(self):
        s = self._filled_session()
        s.invalidate_from(PipelineStage.LOADED)
        assert s.active_edges == []

    def test_clears_cache_status(self):
        s = self._filled_session()
        s.cache_status = {"iref": {"available": True}}
        s.invalidate_from(PipelineStage.LOADED)
        assert s.cache_status == {}

    def test_increments_generation(self):
        s = self._filled_session()
        gen_before = s.generation
        s.invalidate_from(PipelineStage.LOADED)
        assert s.generation == gen_before + 1

    def test_sets_stage_empty(self):
        s = self._filled_session()
        s.invalidate_from(PipelineStage.LOADED)
        assert s.stage == PipelineStage.EMPTY


# ── 4. invalidate_from(INDEXED) ───────────────────────────────────────────────

class TestInvalidateFromIndexed:
    def _filled_session(self) -> Session:
        s = _loaded_session()
        s.stage = PipelineStage.FILLED
        s.index_queries.append(_make_index_query(result={(0,): 1}))
        s.fill_queries.append(_make_fill_query(result={(0,): 1}))
        s.nc_cycles.append(_make_ncs())
        s.weyl_result = MagicMock()
        s.weyl_checked = True
        return s

    def test_preserves_manifold_data(self):
        s = self._filled_session()
        md = s.manifold_data
        s.invalidate_from(PipelineStage.INDEXED)
        assert s.manifold_data is md

    def test_preserves_nz_data(self):
        s = self._filled_session()
        nz = s.nz_data
        s.invalidate_from(PipelineStage.INDEXED)
        assert s.nz_data is nz

    def test_preserves_generation(self):
        s = self._filled_session()
        gen = s.generation
        s.invalidate_from(PipelineStage.INDEXED)
        assert s.generation == gen

    def test_clears_index_queries(self):
        s = self._filled_session()
        s.invalidate_from(PipelineStage.INDEXED)
        assert s.index_queries == []

    def test_clears_fill_queries(self):
        s = self._filled_session()
        s.invalidate_from(PipelineStage.INDEXED)
        assert s.fill_queries == []

    def test_clears_nc_cycles(self):
        s = self._filled_session()
        s.invalidate_from(PipelineStage.INDEXED)
        assert s.nc_cycles == []

    def test_clears_weyl(self):
        s = self._filled_session()
        s.invalidate_from(PipelineStage.INDEXED)
        assert s.weyl_result is None
        assert s.weyl_checked is False

    def test_sets_stage_loaded(self):
        s = self._filled_session()
        s.invalidate_from(PipelineStage.INDEXED)
        assert s.stage == PipelineStage.LOADED


# ── 5. invalidate_from(FILLED) ────────────────────────────────────────────────

class TestInvalidateFromFilled:
    def _filled_session(self) -> Session:
        s = _loaded_session()
        s.stage = PipelineStage.FILLED
        s.index_queries.append(_make_index_query(result={(0,): 1}))
        s.fill_queries.append(_make_fill_query(result={(0,): 1}))
        s.nc_cycles.append(_make_ncs())
        s.weyl_result = MagicMock()
        s.weyl_checked = True
        return s

    def test_preserves_index_queries(self):
        s = self._filled_session()
        count = len(s.index_queries)
        s.invalidate_from(PipelineStage.FILLED)
        assert len(s.index_queries) == count

    def test_preserves_nc_cycles(self):
        s = self._filled_session()
        count = len(s.nc_cycles)
        s.invalidate_from(PipelineStage.FILLED)
        assert len(s.nc_cycles) == count

    def test_preserves_weyl(self):
        s = self._filled_session()
        wr = s.weyl_result
        s.invalidate_from(PipelineStage.FILLED)
        assert s.weyl_result is wr
        assert s.weyl_checked is True

    def test_clears_fill_queries(self):
        s = self._filled_session()
        s.invalidate_from(PipelineStage.FILLED)
        assert s.fill_queries == []

    def test_sets_stage_indexed(self):
        s = self._filled_session()
        s.invalidate_from(PipelineStage.FILLED)
        assert s.stage == PipelineStage.INDEXED


# ── 6. num_hard ───────────────────────────────────────────────────────────────

class TestNumHard:
    def test_none_nz_data_returns_zero(self):
        assert Session().num_hard() == 0

    def test_mock_nz_data(self):
        s = Session()
        s.nz_data = SimpleNamespace(num_hard=3)
        assert s.num_hard() == 3

    def test_num_hard_zero(self):
        s = Session()
        s.nz_data = SimpleNamespace(num_hard=0)
        assert s.num_hard() == 0

    def test_no_hard_edges_true(self):
        s = Session()
        s.nz_data = SimpleNamespace(num_hard=0)
        assert s.no_hard_edges() is True

    def test_no_hard_edges_false(self):
        s = Session()
        s.nz_data = SimpleNamespace(num_hard=2)
        assert s.no_hard_edges() is False


# ── 7. all_edges_active ───────────────────────────────────────────────────────

class TestAllEdgesActive:
    def test_empty_list_returns_true(self):
        assert Session().all_edges_active() is True

    def test_all_true(self):
        s = Session()
        s.active_edges = [True, True, True]
        assert s.all_edges_active() is True

    def test_one_false(self):
        s = Session()
        s.active_edges = [True, False, True]
        assert s.all_edges_active() is False

    def test_all_false(self):
        s = Session()
        s.active_edges = [False, False]
        assert s.all_edges_active() is False


# ── 8. nc_cycles_for ─────────────────────────────────────────────────────────

class TestNCCyclesFor:
    def test_not_found_returns_none(self):
        s = Session()
        assert s.nc_cycles_for(0) is None

    def test_found_correct_cusp(self):
        s = Session()
        ncs0 = _make_ncs(cusp_idx=0)
        ncs1 = _make_ncs(cusp_idx=1)
        s.nc_cycles = [ncs0, ncs1]
        assert s.nc_cycles_for(0) is ncs0
        assert s.nc_cycles_for(1) is ncs1

    def test_missing_cusp_returns_none(self):
        s = Session()
        s.nc_cycles = [_make_ncs(cusp_idx=0)]
        assert s.nc_cycles_for(2) is None


# ── 9. query counts ───────────────────────────────────────────────────────────

class TestQueryCounts:
    def test_index_count_empty(self):
        assert Session().index_query_count() == 0

    def test_index_count_with_none_result(self):
        s = Session()
        s.index_queries.append(_make_index_query(result=None))
        assert s.index_query_count() == 0

    def test_index_count_with_result(self):
        s = Session()
        s.index_queries.append(_make_index_query(result={(0,): 1}))
        s.index_queries.append(_make_index_query(result=None))
        assert s.index_query_count() == 1

    def test_fill_count_with_result(self):
        s = Session()
        s.fill_queries.append(_make_fill_query(result={(0,): 2}))
        assert s.fill_query_count() == 1


# ── 10. Serialisation roundtrip ───────────────────────────────────────────────

class TestSerialisation:
    def _rich_session(self) -> Session:
        s = Session(
            manifold_name="m125",
            generation=3,
            stage=PipelineStage.FILLED,
            q_order_half=50,
            active_edges=[True, False],
            weyl_checked=True,
            export_path="/tmp/out",
        )
        s.cache_status = {"iref": {"available": True, "qq_order": 50}}
        # index query with result
        iq = _make_index_query()
        iq.result = {(0, 1): 3, (2, -1): -2}
        iq.projected_result = {(0,): 1}
        s.index_queries.append(iq)
        # NC cycle metadata
        ncs = _make_ncs(cusp_idx=0)
        s.nc_cycles.append(ncs)
        # fill query with result
        fq = _make_fill_query()
        fq.result = {(4,): 7}
        fq.weyl_a = [Fraction(1, 2), Fraction(-1, 3)]
        fq.weyl_b = [Fraction(0), Fraction(1)]
        fq.incompat_edges = [1]
        s.fill_queries.append(fq)
        return s

    def test_roundtrip_basic_fields(self):
        s = self._rich_session()
        d = session_to_dict(s)
        r = session_from_dict(d)
        assert r.manifold_name == s.manifold_name
        assert r.generation == s.generation
        assert r.stage == s.stage
        assert r.q_order_half == s.q_order_half
        assert r.active_edges == s.active_edges
        assert r.weyl_checked == s.weyl_checked
        assert r.export_path == s.export_path

    def test_roundtrip_cache_status(self):
        s = self._rich_session()
        r = session_from_dict(session_to_dict(s))
        assert r.cache_status == s.cache_status

    def test_roundtrip_index_query_fractions(self):
        s = self._rich_session()
        r = session_from_dict(session_to_dict(s))
        assert len(r.index_queries) == 1
        iq = r.index_queries[0]
        assert iq.e_ext == s.index_queries[0].e_ext
        assert all(isinstance(e, Fraction) for e in iq.e_ext)

    def test_roundtrip_index_query_result(self):
        s = self._rich_session()
        r = session_from_dict(session_to_dict(s))
        iq = r.index_queries[0]
        assert iq.result == {(0, 1): 3, (2, -1): -2}
        assert iq.projected_result == {(0,): 1}

    def test_roundtrip_fill_query_result(self):
        s = self._rich_session()
        r = session_from_dict(session_to_dict(s))
        fq = r.fill_queries[0]
        assert fq.result == {(4,): 7}

    def test_roundtrip_fill_query_weyl(self):
        s = self._rich_session()
        r = session_from_dict(session_to_dict(s))
        fq = r.fill_queries[0]
        assert fq.weyl_a == [Fraction(1, 2), Fraction(-1, 3)]
        assert fq.weyl_b == [Fraction(0), Fraction(1)]
        assert all(isinstance(a, Fraction) for a in fq.weyl_a)

    def test_roundtrip_nc_cycles_metadata(self):
        s = self._rich_session()
        r = session_from_dict(session_to_dict(s))
        assert len(r.nc_cycles) == 1
        ncs = r.nc_cycles[0]
        assert ncs.cusp_idx == 0
        assert ncs.search_p_range == (-5, 5)
        assert ncs.search_q_range == (-5, 5)
        # cycles list is empty after restore
        assert ncs.cycles == []

    def test_roundtrip_non_serialisable_fields_are_none(self):
        s = self._rich_session()
        s.manifold_data = MagicMock()
        s.nz_data = MagicMock()
        s.weyl_result = MagicMock()
        r = session_from_dict(session_to_dict(s))
        assert r.manifold_data is None
        assert r.nz_data is None
        assert r.weyl_result is None

    def test_roundtrip_empty_session(self):
        s = Session()
        r = session_from_dict(session_to_dict(s))
        assert r.stage == PipelineStage.EMPTY
        assert r.index_queries == []
        assert r.fill_queries == []
        assert r.nc_cycles == []

    def test_version_field_present(self):
        d = session_to_dict(Session())
        assert d["version"] == "0.5"
