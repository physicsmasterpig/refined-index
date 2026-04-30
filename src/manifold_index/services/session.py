"""
services.session
================
Single source of truth for one manifold calculation session.

All pipeline cards read from and write to a shared Session instance.
No card stores results in instance variables.

Rule: this module imports only the Python standard library.
      It must not import from `app/` or `PySide6`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from fractions import Fraction
from typing import Any


# ── Pipeline stage ────────────────────────────────────────────────────────────

class PipelineStage(IntEnum):
    """Minimum stage reached in the current session.

    Stages are ordered: EMPTY < LOADED < INDEXED < FILLED.
    """
    EMPTY   = 0   # no manifold loaded
    LOADED  = 1   # manifold + NZ matrix built
    INDEXED = 2   # at least one refined index query computed
    FILLED  = 3   # at least one filled index query computed


# ── Per-query records ─────────────────────────────────────────────────────────

@dataclass
class IndexQuery:
    """One (m_ext, e_ext) refined index query and its result.

    *result* is a RefinedIndexResult — a dict mapping tuples of ints
    (q_half_power, 2*η₀_exp, …) to integer coefficients.

    *projected_result* is the same dict after applying *active_edges*
    (setting inactive η variables to 1 by summing over them).
    It equals *result* when all edges are active, and equals the 3D index
    dict when all edges are inactive.  None means no projection has been
    computed yet (i.e. full refined output is current).
    """
    m_ext: list[int]
    e_ext: list[Fraction]
    q_order_half: int
    result: Any                      # RefinedIndexResult | None
    projected_result: Any            # projected dict | None
    active_edges: list[bool]         # snapshot of toggles at query time
    timestamp: float = field(default_factory=time.time)
    source: str = "computed"         # "computed" | "cache"


@dataclass
class FillQuery:
    """One NC cycle + user slope + (optional) other-cusp charges, and result.

    *nc_P*, *nc_Q* — the NC cycle expressed in the (α, β) cusp basis.
    *user_P*, *user_Q* — the user's Dehn filling slope in the same basis.
    *p*, *q* — the user slope expressed in the NC basis (γ, δ).
    *m_other*, *e_other* — external charges on unfilled cusps (multi-cusp).
    *incompat_edges* — indices of η variables forced to 1 for Weyl compat.
    *unrefined_fallback* — True when NC cycles were absent and we fell back
                            to the unrefined filling formula.
    """
    cusp_idx: int
    nc_P: int
    nc_Q: int
    user_P: int
    user_Q: int
    p: int                           # user slope in NC basis
    q: int
    m_other: list[int]
    e_other: list[Fraction]
    q_order_half: int
    result: Any                      # FilledRefinedResult | None
    weyl_a: list[Fraction] | None
    weyl_b: list[Fraction] | None
    incompat_edges: list[int]
    timestamp: float = field(default_factory=time.time)
    source: str = "computed"         # "computed" | "cache"
    unrefined_fallback: bool = False
    row_index: int = -1              # table row this query owns (for stable re-render)


@dataclass
class MultiFillQuery:
    """Multi-cusp simultaneous Dehn filling result.

    *cusp_specs* — per-cusp list of dicts with keys:
        cusp_idx, nc_P, nc_Q, user_P, user_Q, p, q
    *result* — FilledRefinedResult from compute_multi_cusp_filled_refined_index
    *unfilled_charges* — list of (m, e) tuples for cusps not being filled
    """
    cusp_specs: list[dict]               # one dict per filled cusp
    q_order_half: int
    result: Any                          # FilledRefinedResult | None
    unfilled_charges: list = field(default_factory=list)  # [(cusp_idx, m, e), …]
    timestamp: float = field(default_factory=time.time)
    source: str = "computed"             # "computed" | "cache"
    row_index: int = -1                  # table row this query owns (for stable re-render)


@dataclass
class NCCycleSet:
    """NC cycle search results for one cusp."""
    cusp_idx: int
    search_p_range: tuple[int, int]
    search_q_range: tuple[int, int]
    q_order_half: int
    cycles: list[Any]                # list[NonClosableCycle]
    source: str = "computed"         # "computed" | "cache"


# ── Session ───────────────────────────────────────────────────────────────────

@dataclass
class Session:
    """Complete state for one manifold calculation session.

    Pipeline cards read from this; workers write to this via signals that
    PipelineView intercepts and applies.

    Fields that hold rich Python objects from core/ (manifold_data,
    nz_data, weyl_result) are not serialised — they are re-built on
    session restore from manifold_name + settings alone.
    """

    # ── Identity ──────────────────────────────────────────────────────
    manifold_name: str = ""
    generation: int = 0              # increments each time the manifold changes

    # ── Stage ─────────────────────────────────────────────────────────
    stage: PipelineStage = PipelineStage.EMPTY

    # ── Card ① results ────────────────────────────────────────────────
    manifold_data: Any = None        # snappy.Manifold | None  (not serialised)
    nz_data: Any = None              # NeumannZagierData | None (not serialised)
    easy_result: Any = None          # EasyEdgeResult | None    (not serialised)
    cache_status: dict[str, Any] = field(default_factory=dict)
    # cache_status keys: "iref", "nc", "kernels"
    # values: {"available": bool, "qq_order": int|None, ...}

    # ── Card ② settings ───────────────────────────────────────────────
    q_order_half: int = 20           # shared qq setting (applies everywhere)
    nc_q_order_half: int = 10        # NC-cycle search truncation (= q^5 default)
    nc_search_p_range: int = 1       # NC search: |p| ≤ this
    nc_search_q_range: int = 1       # NC search: 0 ≤ q ≤ this
    active_edges: list[bool] = field(default_factory=list)
    # active_edges[j] = True  → η_j is kept in output
    # active_edges[j] = False → η_j is projected out (set to 1)
    index_queries: list[IndexQuery] = field(default_factory=list)

    # ── Weyl bridge (②→③) ─────────────────────────────────────────────
    weyl_result: Any = None          # ABVectors | None  (not serialised)
    weyl_adjoint_pass: "bool | None" = None  # q¹ adjoint projection pass (not serialised)
    weyl_checked: bool = False

    # ── Card ③ results ────────────────────────────────────────────────
    nc_cycles: list[NCCycleSet] = field(default_factory=list)
    fill_queries: list[FillQuery] = field(default_factory=list)
    multi_fill_queries: list[MultiFillQuery] = field(default_factory=list)

    # ── Card ④ settings ───────────────────────────────────────────────
    export_path: str = ""

    # ── Queries ───────────────────────────────────────────────────────

    def nc_cycles_for(self, cusp_idx: int) -> NCCycleSet | None:
        """Return the NCCycleSet for *cusp_idx*, or None if not searched."""
        for ncs in self.nc_cycles:
            if ncs.cusp_idx == cusp_idx:
                return ncs
        return None

    def index_query_count(self) -> int:
        """Number of completed (non-None result) index queries."""
        return sum(1 for q in self.index_queries if q.result is not None)

    def fill_query_count(self) -> int:
        """Number of completed (non-None result) fill queries."""
        return sum(1 for q in self.fill_queries if q.result is not None)

    # ── Stage helpers ─────────────────────────────────────────────────

    def has_any_results(self) -> bool:
        """True once a manifold has been loaded (stage ≥ LOADED)."""
        return self.stage >= PipelineStage.LOADED

    def num_hard(self) -> int:
        """Number of hard edges (η variables) for the loaded manifold."""
        if self.nz_data is None:
            return 0
        return int(self.nz_data.num_hard)

    def all_edges_active(self) -> bool:
        """True when every η variable is active (full refined output)."""
        return all(self.active_edges) if self.active_edges else True

    def no_hard_edges(self) -> bool:
        """True when num_hard == 0 (output is the plain 3D index)."""
        return self.num_hard() == 0

    # ── Mutation ──────────────────────────────────────────────────────

    def invalidate_from(self, stage: PipelineStage) -> None:
        """Clear all results at and beyond *stage*.

        This is the single place where downstream state is wiped.
        Callers (PipelineView) use this whenever upstream input changes:

            session.invalidate_from(PipelineStage.LOADED)   # new manifold
            session.invalidate_from(PipelineStage.INDEXED)  # qq changed
            session.invalidate_from(PipelineStage.FILLED)   # NC re-searched

        *generation* is incremented only when the LOADED stage is cleared,
        because that is the only case where all card outputs are invalidated.
        """
        if stage <= PipelineStage.LOADED:
            self.manifold_data = None
            self.nz_data = None
            self.easy_result = None
            self.cache_status = {}
            self.active_edges = []
            self.index_queries = []
            self.weyl_result = None
            self.weyl_adjoint_pass = None
            self.weyl_checked = False
            self.nc_cycles = []
            self.fill_queries = []
            self.generation += 1
            self.stage = PipelineStage.EMPTY

        elif stage <= PipelineStage.INDEXED:
            self.index_queries = []
            self.weyl_result = None
            self.weyl_adjoint_pass = None
            self.weyl_checked = False
            self.nc_cycles = []
            self.fill_queries = []
            self.stage = PipelineStage.LOADED

        elif stage <= PipelineStage.FILLED:
            self.fill_queries = []
            # Keep nc_cycles: they belong to INDEXED stage.
            if self.stage > PipelineStage.INDEXED:
                self.stage = PipelineStage.INDEXED


# ── Serialisation (Phase 10) ──────────────────────────────────────────────────
# Rich core objects (manifold_data, nz_data, weyl_result) are intentionally
# excluded.  On restore the caller must re-run ComputeService.load_manifold
# and, if desired, re-run Weyl check.

def _fraction_to_list(f: Fraction) -> list[int]:
    return [f.numerator, f.denominator]


def _list_to_fraction(lst: list[int]) -> Fraction:
    return Fraction(lst[0], lst[1])


def _result_to_json(result: Any) -> list | None:
    """Serialise a RefinedIndexResult / FilledRefinedResult dict."""
    if result is None:
        return None
    return [[list(k), v] for k, v in result.items()]


def _json_to_result(data: list | None) -> dict | None:
    """Restore a result dict from its JSON form."""
    if data is None:
        return None
    return {tuple(pair[0]): pair[1] for pair in data}


def session_to_dict(session: Session) -> dict:
    """Serialise a Session to a JSON-safe dict.

    Non-serialisable fields (manifold_data, nz_data, weyl_result) are
    omitted.  The caller must reload them on restore.
    """
    def _iq(q: IndexQuery) -> dict:
        return {
            "m_ext": q.m_ext,
            "e_ext": [_fraction_to_list(e) for e in q.e_ext],
            "q_order_half": q.q_order_half,
            "result": _result_to_json(q.result),
            "projected_result": _result_to_json(q.projected_result),
            "active_edges": q.active_edges,
            "timestamp": q.timestamp,
            "source": q.source,
        }

    def _fq(q: FillQuery) -> dict:
        return {
            "cusp_idx": q.cusp_idx,
            "nc_P": q.nc_P,
            "nc_Q": q.nc_Q,
            "user_P": q.user_P,
            "user_Q": q.user_Q,
            "p": q.p,
            "q": q.q,
            "m_other": q.m_other,
            "e_other": [_fraction_to_list(e) for e in q.e_other],
            "q_order_half": q.q_order_half,
            "result": _result_to_json(q.result),
            "weyl_a": [_fraction_to_list(a) for a in q.weyl_a] if q.weyl_a else None,
            "weyl_b": [_fraction_to_list(b) for b in q.weyl_b] if q.weyl_b else None,
            "incompat_edges": q.incompat_edges,
            "timestamp": q.timestamp,
            "source": q.source,
            "unrefined_fallback": q.unrefined_fallback,
        }

    def _ncs(ncs: NCCycleSet) -> dict:
        # NonClosableCycle objects are not serialised; only metadata.
        return {
            "cusp_idx": ncs.cusp_idx,
            "search_p_range": list(ncs.search_p_range),
            "search_q_range": list(ncs.search_q_range),
            "q_order_half": ncs.q_order_half,
            "n_cycles": len(ncs.cycles),
            "source": ncs.source,
        }

    return {
        "version": "0.5",
        "manifold_name": session.manifold_name,
        "generation": session.generation,
        "stage": session.stage.value,
        "cache_status": session.cache_status,
        "q_order_half": session.q_order_half,
        "nc_q_order_half": session.nc_q_order_half,
        "nc_search_p_range": session.nc_search_p_range,
        "nc_search_q_range": session.nc_search_q_range,
        "active_edges": session.active_edges,
        "index_queries": [_iq(q) for q in session.index_queries],
        "weyl_checked": session.weyl_checked,
        "nc_cycles_meta": [_ncs(ncs) for ncs in session.nc_cycles],
        "fill_queries": [_fq(q) for q in session.fill_queries],
        "export_path": session.export_path,
    }


def session_from_dict(data: dict) -> Session:
    """Restore a Session from a saved dict (produced by session_to_dict).

    *manifold_data*, *nz_data*, *weyl_result*, and the actual
    NonClosableCycle objects inside *nc_cycles* are NOT restored —
    callers must re-run LoadWorker and (if desired) NCSearchWorker.
    """
    def _iq(d: dict) -> IndexQuery:
        return IndexQuery(
            m_ext=d["m_ext"],
            e_ext=[_list_to_fraction(e) for e in d["e_ext"]],
            q_order_half=d["q_order_half"],
            result=_json_to_result(d["result"]),
            projected_result=_json_to_result(d["projected_result"]),
            active_edges=d["active_edges"],
            timestamp=d["timestamp"],
            source=d["source"],
        )

    def _fq(d: dict) -> FillQuery:
        return FillQuery(
            cusp_idx=d["cusp_idx"],
            nc_P=d["nc_P"],
            nc_Q=d["nc_Q"],
            user_P=d["user_P"],
            user_Q=d["user_Q"],
            p=d["p"],
            q=d["q"],
            m_other=d["m_other"],
            e_other=[_list_to_fraction(e) for e in d["e_other"]],
            q_order_half=d["q_order_half"],
            result=_json_to_result(d["result"]),
            weyl_a=[_list_to_fraction(a) for a in d["weyl_a"]] if d["weyl_a"] else None,
            weyl_b=[_list_to_fraction(b) for b in d["weyl_b"]] if d["weyl_b"] else None,
            incompat_edges=d["incompat_edges"],
            timestamp=d["timestamp"],
            source=d["source"],
            unrefined_fallback=d.get("unrefined_fallback", False),
        )

    s = Session(
        manifold_name=data["manifold_name"],
        generation=data["generation"],
        stage=PipelineStage(data["stage"]),
        cache_status=data.get("cache_status", {}),
        q_order_half=data["q_order_half"],
        nc_q_order_half=data.get("nc_q_order_half", 10),
        nc_search_p_range=data.get("nc_search_p_range", 1),
        nc_search_q_range=data.get("nc_search_q_range", 1),
        active_edges=data["active_edges"],
        index_queries=[_iq(q) for q in data.get("index_queries", [])],
        weyl_checked=data.get("weyl_checked", False),
        fill_queries=[_fq(q) for q in data.get("fill_queries", [])],
        export_path=data.get("export_path", ""),
    )
    # nc_cycles: restore metadata-only stubs (cycles list is empty)
    for ncs_d in data.get("nc_cycles_meta", []):
        s.nc_cycles.append(NCCycleSet(
            cusp_idx=ncs_d["cusp_idx"],
            search_p_range=tuple(ncs_d["search_p_range"]),
            search_q_range=tuple(ncs_d["search_q_range"]),
            q_order_half=ncs_d["q_order_half"],
            cycles=[],           # must be re-populated by NCSearchWorker
            source=ncs_d["source"],
        ))
    return s
