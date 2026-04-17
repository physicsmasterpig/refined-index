"""
viewmodels.manifold_vm
======================
ManifoldViewModel — display-ready data for the Manifold (Card ①) panel.

No Qt dependency.  Constructed by the card or a dedicated builder function.

BLUEPRINT references
--------------------
§8.1  ManifoldViewModel dataclass
§12.1 Category A advisories
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from manifold_index.viewmodels.advisory import Advisory, Advisories


# ---------------------------------------------------------------------------
# ManifoldViewModel
# ---------------------------------------------------------------------------

@dataclass
class ManifoldViewModel:
    """Display-ready data for the manifold load step.

    Attributes
    ----------
    name : str
        Manifold name as given to SnaPy (e.g. ``"m004"``).
    n_tetrahedra : int
        Number of ideal tetrahedra (*n* in NZ notation).
    n_cusps : int
        Number of cusps (*r*).
    num_hard : int
        Number of hard edges (η-fugacity variables).
    num_easy : int
        Number of easy edges (no fugacity weight).
    has_hard_edges : bool
        ``num_hard > 0``.
    index_title : str
        ``"Refined Index"`` when ``has_hard_edges``, else ``"3D Index"``.
    nz_latex : str
        KaTeX-ready string for the NZ gluing matrix (formatted by
        ``formatters.manifold_fmt``; empty string if not yet formatted).
    gluing_table_html : str
        HTML ``<table>`` of gluing equations (one row per tetrahedron).
    easy_edges_html : str
        HTML fragment listing easy edges and their positions.
    hard_edges_html : str
        HTML fragment listing hard edges and their η labels.
    is_symplectic : bool
        Whether the NZ matrix satisfies the symplectic constraint.
    cache_status : dict
        Raw output of ``ComputeService.probe_cache``.
    advisories : list[Advisory]
        Advisories generated for this card.
    """

    name: str
    n_tetrahedra: int
    n_cusps: int
    num_hard: int
    num_easy: int
    has_hard_edges: bool
    index_title: str
    nz_latex: str
    gluing_table_html: str
    easy_edges_html: str
    hard_edges_html: str
    is_symplectic: bool
    cache_status: dict
    advisories: list[Advisory] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_manifold_vm(
    manifold_data: Any,
    easy_result: Any,
    nz_data: Any,
    cache_status: dict,
    *,
    nz_latex: str = "",
    gluing_table_html: str = "",
    easy_edges_html: str = "",
    hard_edges_html: str = "",
    is_symplectic: bool = True,
) -> ManifoldViewModel:
    """Construct a ``ManifoldViewModel`` from service-layer objects.

    Parameters
    ----------
    manifold_data : ManifoldData
    easy_result : EasyEdgeResult
    nz_data : NeumannZagierData
    cache_status : dict
        Return value of ``ComputeService.probe_cache``.
    nz_latex / gluing_table_html / easy_edges_html / hard_edges_html
        Pre-formatted strings from ``formatters.manifold_fmt``.
        Pass empty strings here; the formatters layer fills them in Phase 4.
    is_symplectic : bool
        Supplied by ``formatters.manifold_fmt`` (Phase 4).

    Returns
    -------
    ManifoldViewModel
    """
    n_hard: int = nz_data.num_hard
    n_easy: int = nz_data.n - nz_data.r - n_hard  # internal edges minus hard

    has_hard = n_hard > 0
    index_title = "Refined Index" if has_hard else "3D Index"

    # ── Generate advisories ───────────────────────────────────────────
    advisories: list[Advisory] = []

    if not has_hard:
        advisories.append(Advisories.A1())

    return ManifoldViewModel(
        name=manifold_data.name,
        n_tetrahedra=nz_data.n,
        n_cusps=nz_data.r,
        num_hard=n_hard,
        num_easy=n_easy,
        has_hard_edges=has_hard,
        index_title=index_title,
        nz_latex=nz_latex,
        gluing_table_html=gluing_table_html,
        easy_edges_html=easy_edges_html,
        hard_edges_html=hard_edges_html,
        is_symplectic=is_symplectic,
        cache_status=cache_status,
        advisories=advisories,
    )


def build_manifold_vm_error(name: str, error: Exception) -> ManifoldViewModel:
    """Construct a failed ManifoldViewModel with an A2 advisory.

    Used when ``ComputeService.load_manifold`` raises ``ValueError``.
    """
    return ManifoldViewModel(
        name=name,
        n_tetrahedra=0,
        n_cusps=0,
        num_hard=0,
        num_easy=0,
        has_hard_edges=False,
        index_title="3D Index",
        nz_latex="",
        gluing_table_html="",
        easy_edges_html="",
        hard_edges_html="",
        is_symplectic=False,
        cache_status={},
        advisories=[Advisories.A2(name)],
    )
