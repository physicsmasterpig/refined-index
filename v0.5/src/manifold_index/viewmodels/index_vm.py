"""
viewmodels.index_vm
===================
ViewModels for the Index (Card ②) panel.

Classes
-------
IndexQueryViewModel  — one computed I^ref(m, e) entry
WeylViewModel        — Weyl symmetry check result
IndexViewModel       — the full card's display state

BLUEPRINT references
--------------------
§8.2  IndexQueryViewModel
§8.3  WeylViewModel
§12.2 Category B advisories
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from manifold_index.viewmodels.advisory import Advisory, Advisories


# ---------------------------------------------------------------------------
# IndexQueryViewModel
# ---------------------------------------------------------------------------

@dataclass
class IndexQueryViewModel:
    """Display-ready data for one I^ref(m, e) computation.

    Attributes
    ----------
    m_ext : list[int]
        Meridian external charge values (one per cusp).
    e_ext : list[Fraction]
        Longitude external charge values (one per cusp).
    q_order_half : int
        Truncation order used (power of q^{1/2}).
    active_edges : list[bool]
        Which hard edges have η active after projection.
    result_latex : str
        KaTeX-ready full refined index series string.
    projected_latex : str
        KaTeX-ready projected (η=1 for inactive edges) string.
    is_zero : bool
        True if the result dict is empty.
    source : str
        ``"computed"`` or ``"cache"``.
    timestamp : float
        Unix time when the result was obtained.
    advisories : list[Advisory]
        Per-query advisories (e.g. B1 for zero result).
    """
    m_ext: list[int]
    e_ext: list
    q_order_half: int
    active_edges: list[bool]
    result_latex: str
    projected_latex: str
    is_zero: bool
    source: str
    timestamp: float
    advisories: list[Advisory] = field(default_factory=list)


# ---------------------------------------------------------------------------
# WeylViewModel
# ---------------------------------------------------------------------------

@dataclass
class WeylViewModel:
    """Display-ready data for the Weyl symmetry check.

    Attributes
    ----------
    checked : bool
        True if ``run_weyl_check`` has been called.
    a_vectors : list[Fraction]
        Weyl *a* vectors (one per hard edge).
    b_vectors : list[Fraction]
        Weyl *b* vectors (one per hard edge).
    edge_compatible : list[bool]
        Per-edge half-integer e compatibility flags.
    is_fully_compatible : bool
        True iff all edges are compatible.
    adjoint_value : float | None
        su(2) adjoint check value; ``None`` if not computed.
    adjoint_passed : bool | None
        True iff ``adjoint_value ≈ −1``.
    warnings : list[str]
        Human-readable warning strings for minor issues.
    advisories : list[Advisory]
        Advisories generated for this check.
    """
    checked: bool
    a_vectors: list
    b_vectors: list
    edge_compatible: list[bool]
    is_fully_compatible: bool
    adjoint_value: float | None
    adjoint_passed: bool | None
    warnings: list[str]
    advisories: list[Advisory] = field(default_factory=list)


# ---------------------------------------------------------------------------
# IndexViewModel
# ---------------------------------------------------------------------------

@dataclass
class IndexViewModel:
    """Display-ready data for the full Index card (Card ②).

    Attributes
    ----------
    queries : list[IndexQueryViewModel]
        All computed index entries in the current session.
    weyl_status : WeylViewModel | None
        None if Weyl check has not been run.
    advisories : list[Advisory]
        Card-level advisories (e.g. B3 if Weyl returned None).
    """
    queries: list[IndexQueryViewModel]
    weyl_status: "WeylViewModel | None"
    advisories: list[Advisory] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_index_query_vm(
    m_ext: list[int],
    e_ext: list,
    q_order_half: int,
    active_edges: list[bool],
    result: Any,           # RefinedIndexResult dict
    source: str = "computed",
    *,
    result_latex: str = "",
    projected_latex: str = "",
    timestamp: float | None = None,
) -> IndexQueryViewModel:
    """Construct an ``IndexQueryViewModel`` from a computed result dict.

    Parameters
    ----------
    result : RefinedIndexResult
        Plain ``dict[tuple[int,...], int]`` from ``ComputeService``.
    result_latex / projected_latex
        Pre-formatted KaTeX strings; filled in by Phase 4 formatters.
    """
    is_zero = not bool(result)
    advisories: list[Advisory] = []
    if is_zero:
        advisories.append(Advisories.B1(q_order_half))

    return IndexQueryViewModel(
        m_ext=list(m_ext),
        e_ext=list(e_ext),
        q_order_half=q_order_half,
        active_edges=list(active_edges),
        result_latex=result_latex,
        projected_latex=projected_latex,
        is_zero=is_zero,
        source=source,
        timestamp=timestamp if timestamp is not None else time.time(),
        advisories=advisories,
    )


def build_weyl_vm(
    ab_result: Any,   # ABVectors from ComputeService.run_weyl_check, or None
    num_hard: int,
    *,
    adjoint_value: float | None = None,
    adjoint_passed: bool | None = None,
) -> WeylViewModel:
    """Construct a ``WeylViewModel`` from the Weyl-check return value.

    Parameters
    ----------
    ab_result : ABVectors | None
        Return of ``ComputeService.run_weyl_check``.  ``None`` when check failed.
    num_hard : int
        Number of hard edges.
    """
    advisories: list[Advisory] = []

    if ab_result is None:
        return WeylViewModel(
            checked=True,
            a_vectors=[],
            b_vectors=[],
            edge_compatible=[False] * num_hard,
            is_fully_compatible=False,
            adjoint_value=None,
            adjoint_passed=None,
            warnings=[],
            advisories=[Advisories.B3()],
        )

    a_vecs = list(ab_result.a)
    b_vecs = list(ab_result.b)

    # Edge compatibility: b_j should be a half-integer (denominator 1 or 2)
    edge_compat: list[bool] = []
    incompat_edges: list[int] = []
    for j, bv in enumerate(b_vecs):
        frac = Fraction(bv)
        compat = (frac.denominator in (1, 2))
        edge_compat.append(compat)
        if not compat:
            incompat_edges.append(j)

    if incompat_edges:
        advisories.append(Advisories.B4(incompat_edges))

    if adjoint_value is not None and adjoint_passed is False:
        advisories.append(Advisories.B5(adjoint_value))

    return WeylViewModel(
        checked=True,
        a_vectors=a_vecs,
        b_vectors=b_vecs,
        edge_compatible=edge_compat,
        is_fully_compatible=len(incompat_edges) == 0,
        adjoint_value=adjoint_value,
        adjoint_passed=adjoint_passed,
        warnings=[],
        advisories=advisories,
    )


def build_index_vm(
    queries: list[IndexQueryViewModel],
    weyl_status: "WeylViewModel | None",
) -> IndexViewModel:
    """Construct the top-level ``IndexViewModel``."""
    card_advisories: list[Advisory] = []
    if weyl_status is not None and not weyl_status.checked:
        pass  # Weyl not run — no card advisory
    # Card-level advisory B3 is surfaced via weyl_status.advisories, not duplicated here
    return IndexViewModel(
        queries=queries,
        weyl_status=weyl_status,
        advisories=card_advisories,
    )
