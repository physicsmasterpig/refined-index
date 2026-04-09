"""
viewmodels.filling_vm
=====================
ViewModels for the Dehn Filling (Card ③) panel.

Classes
-------
NCCycleViewModel   — one non-closable cycle entry
FillQueryViewModel — one filled-index computation
FillingViewModel   — full card display state

BLUEPRINT references
--------------------
§8.4  FillingViewModel, NCCycleViewModel, FillQueryViewModel
§12.3 Category C advisories
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from fractions import Fraction
from math import gcd
from typing import Any

from manifold_index.viewmodels.advisory import Advisory, Advisories


# ---------------------------------------------------------------------------
# NCCycleViewModel
# ---------------------------------------------------------------------------

@dataclass
class NCCycleViewModel:
    """Display data for a single non-closable cycle.

    Attributes
    ----------
    cusp_idx : int
        Which cusp this cycle belongs to.
    P, Q : int
        Slope in the (γ, δ) basis (meridian = γ, longitude = δ).
    slope_latex : str
        KaTeX-ready slope string, e.g. ``r"$\gamma$"`` for (1,0).
    weyl_compatible : bool | None
        Whether the slope is compatible with the Weyl symmetry vectors.
        ``None`` if Weyl check has not been run.
    adjoint_proj_pass : bool | None
        Whether the q¹ adjoint su(2) projection equals −1.
        ``None`` if the adjoint check has not been run.
    source : str
        ``"computed"`` or ``"cache"``.
    """
    cusp_idx: int
    P: int
    Q: int
    slope_latex: str
    weyl_compatible: "bool | None"
    adjoint_proj_pass: "bool | None"
    source: str


# ---------------------------------------------------------------------------
# FillQueryViewModel
# ---------------------------------------------------------------------------

@dataclass
class FillQueryViewModel:
    """Display data for one filled-index computation.

    Attributes
    ----------
    nc_slope_latex : str
        KaTeX NC cycle slope (the basis).
    user_slope_latex : str
        KaTeX user-requested filling slope.
    p, q : int
        Slope in the NC basis after basis change.
    m_other : list[int]
        Meridian charges on unfilled cusps.
    e_other : list
        Longitude charges on unfilled cusps.
    result_latex : str
        KaTeX-ready filled series string.
    is_zero : bool
        True if the filled index is identically zero.
    incompat_edges : list[int]
        Hard-edge indices incompatible with the filling slope.
    weyl_a_latex : str | None
        KaTeX Weyl *a* string (None if not available).
    weyl_b_latex : str | None
        KaTeX Weyl *b* string (None if not available).
    source : str
        ``"computed"`` or ``"cache"``.
    timestamp : float
        Unix time when the result was obtained.
    advisories : list[Advisory]
        Per-query advisories.
    """
    nc_slope_latex: str
    user_slope_latex: str
    p: int
    q: int
    m_other: list[int]
    e_other: list
    result_latex: str
    is_zero: bool
    incompat_edges: list[int]
    weyl_a_latex: "str | None"
    weyl_b_latex: "str | None"
    source: str
    timestamp: float
    advisories: list[Advisory] = field(default_factory=list)


# ---------------------------------------------------------------------------
# FillingViewModel
# ---------------------------------------------------------------------------

@dataclass
class FillingViewModel:
    """Display-ready data for the full Filling card (Card ③).

    Attributes
    ----------
    nc_cycles : list[NCCycleViewModel]
        All canonicalised NC cycles across all cusps.
    fill_queries : list[FillQueryViewModel]
        All completed filling computations.
    advisories : list[Advisory]
        Card-level advisories (e.g. C1 if no NC cycles found).
    """
    nc_cycles: list[NCCycleViewModel]
    fill_queries: list[FillQueryViewModel]
    advisories: list[Advisory] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_nc_cycle_vm(
    cusp_idx: int,
    P: int,
    Q: int,
    weyl_compatible: "bool | None" = None,
    adjoint_proj_pass: "bool | None" = None,
    source: str = "computed",
    *,
    slope_latex: str = "",
) -> NCCycleViewModel:
    """Construct an ``NCCycleViewModel``.

    The *slope_latex* string is filled in by the Phase 4 formatter.
    Provide a placeholder here when formatters are not yet available.
    """
    if not slope_latex:
        # Minimal fallback: P γ + Q δ notation in plain LaTeX
        slope_latex = _simple_slope_latex(P, Q)

    return NCCycleViewModel(
        cusp_idx=cusp_idx,
        P=P,
        Q=Q,
        slope_latex=slope_latex,
        weyl_compatible=weyl_compatible,
        adjoint_proj_pass=adjoint_proj_pass,
        source=source,
    )


def build_fill_query_vm(
    nc_P: int,
    nc_Q: int,
    user_P: int,
    user_Q: int,
    p: int,
    q: int,
    m_other: list[int],
    e_other: list,
    result: Any,   # FilledRefinedResult
    weyl_a: "list | None" = None,
    weyl_b: "list | None" = None,
    source: str = "computed",
    *,
    result_latex: str = "",
    nc_slope_latex: str = "",
    user_slope_latex: str = "",
    timestamp: float | None = None,
) -> FillQueryViewModel:
    """Construct a ``FillQueryViewModel`` from a ``FilledRefinedResult``."""
    is_zero = result.is_zero if hasattr(result, "is_zero") else not bool(result)

    advisories: list[Advisory] = []

    # C2: trivial slope (p=0) in NC basis
    if p == 0:
        advisories.append(Advisories.C2())

    # Fallback slope strings
    if not nc_slope_latex:
        nc_slope_latex = _simple_slope_latex(nc_P, nc_Q)
    if not user_slope_latex:
        user_slope_latex = _simple_slope_latex(user_P, user_Q)

    weyl_a_latex = _vec_latex(weyl_a) if weyl_a is not None else None
    weyl_b_latex = _vec_latex(weyl_b) if weyl_b is not None else None

    return FillQueryViewModel(
        nc_slope_latex=nc_slope_latex,
        user_slope_latex=user_slope_latex,
        p=p,
        q=q,
        m_other=list(m_other),
        e_other=list(e_other),
        result_latex=result_latex,
        is_zero=is_zero,
        incompat_edges=[],
        weyl_a_latex=weyl_a_latex,
        weyl_b_latex=weyl_b_latex,
        source=source,
        timestamp=timestamp if timestamp is not None else time.time(),
        advisories=advisories,
    )


def build_filling_vm(
    nc_cycles: list[NCCycleViewModel],
    fill_queries: list[FillQueryViewModel],
) -> FillingViewModel:
    """Construct the top-level ``FillingViewModel``."""
    advisories: list[Advisory] = []

    if not nc_cycles:
        advisories.append(Advisories.C1())

    return FillingViewModel(
        nc_cycles=nc_cycles,
        fill_queries=fill_queries,
        advisories=advisories,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _simple_slope_latex(P: int, Q: int) -> str:
    """Minimal KaTeX slope string for fallback use (meridian=γ, longitude=δ)."""
    if Q == 0:
        return rf"${P}\gamma$" if P != 1 else r"$\gamma$"
    if P == 0:
        return rf"${Q}\delta$" if Q != 1 else r"$\delta$"
    q_str = str(Q) if Q != 1 else ""
    return rf"${P}\gamma + {q_str}\delta$"


def _vec_latex(vec: list) -> str:
    """Format a list of Fraction values as a KaTeX row vector."""
    parts = [str(Fraction(v)) for v in vec]
    return r"$\left(" + r",\,".join(parts) + r"\right)$"
