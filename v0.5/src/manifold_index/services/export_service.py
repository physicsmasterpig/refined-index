"""
services.export_service
=======================
ExportService — write session results to files.

All methods accept a Session and optional output path.  They delegate
entirely to utils/exporters.py and utils/cache_export.py; no formatting
logic lives here.

Rule: no Qt, no app imports.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from manifold_index.utils import exporters as _exp_mod

if TYPE_CHECKING:
    from manifold_index.services.session import Session


class ExportService:
    """Write session results to files."""

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @staticmethod
    def available_data(session: "Session") -> dict:
        """Inspect session and return what can be exported.

        Returns
        -------
        dict with keys:
          "manifold"      : bool — manifold_data and nz_data are present
          "index_queries" : int  — number of completed index query results
          "weyl"          : bool — weyl_result is available
          "nc_cycles"     : int  — total NC cycles found (all cusps)
          "fill_queries"  : int  — number of completed fill query results
        """
        n_iqs = sum(1 for q in session.index_queries if q.result is not None)
        n_nc = sum(
            len(ncs.cycles)
            for ncs in session.nc_cycles
        )
        n_fqs = sum(1 for q in session.fill_queries if q.result is not None)
        return {
            "manifold": session.manifold_data is not None and session.nz_data is not None,
            "index_queries": n_iqs,
            "weyl": session.weyl_result is not None,
            "nc_cycles": n_nc,
            "fill_queries": n_fqs,
        }

    # ------------------------------------------------------------------
    # File writers
    # ------------------------------------------------------------------

    @staticmethod
    def write_latex(
        session: "Session",
        output_path,
        include_filling: bool = True,
    ) -> None:
        """Write a compact LaTeX .tex file for all index query results.

        Uses the legacy ``write_latex`` wrapper in exporters.py, which
        produces a minimal compilable document suitable for quick export.
        For the comprehensive multi-section report, use ``write_full_report``.
        """
        if session.nz_data is None:
            raise ValueError("No NZ data in session — load a manifold first")
        entries = [
            (q.m_ext, q.e_ext, q.result)
            for q in session.index_queries
            if q.result is not None
        ]
        dehn_results = _ExportService_build_dehn_results(session) if include_filling else None
        _exp_mod.write_latex(
            path=output_path,
            manifold_name=session.manifold_name,
            nz=session.nz_data,
            entries=entries,
            weyl=session.weyl_result,
            dehn_results=dehn_results,
            include_dehn=include_filling and bool(dehn_results),
        )

    @staticmethod
    def write_mathematica(
        session: "Session",
        output_path,
        include_filling: bool = True,
    ) -> None:
        """Write a Mathematica .m data file."""
        if session.nz_data is None:
            raise ValueError("No NZ data in session — load a manifold first")
        entries = [
            (q.m_ext, q.e_ext, q.result)
            for q in session.index_queries
            if q.result is not None
        ]
        dehn_results = _ExportService_build_dehn_results(session) if include_filling else None
        _exp_mod.write_mathematica(
            path=output_path,
            manifold_data=_ExportService_manifold_data(session),
            nz_data=session.nz_data,
            entries=entries,
            weyl_result=session.weyl_result,
            dehn_results=dehn_results,
            q_order_half=session.q_order_half,
        )

    @staticmethod
    def write_json(
        session: "Session",
        output_path,
        include_filling: bool = True,
    ) -> None:
        """Write a structured JSON file."""
        if session.nz_data is None:
            raise ValueError("No NZ data in session — load a manifold first")
        entries = [
            (q.m_ext, q.e_ext, q.result)
            for q in session.index_queries
            if q.result is not None
        ]
        dehn_results = _ExportService_build_dehn_results(session) if include_filling else None
        _exp_mod.write_json(
            path=output_path,
            manifold_data=_ExportService_manifold_data(session),
            easy_result=None,          # easy_result not stored in session
            nz_data=session.nz_data,
            entries=entries,
            weyl_result=session.weyl_result,
            dehn_results=dehn_results,
            q_order_half=session.q_order_half,
        )

    @staticmethod
    def write_full_report(
        session: "Session",
        output_path,
        include_filling: bool = True,
    ) -> None:
        """Write a compilable LaTeX report with all sections."""
        if session.nz_data is None:
            raise ValueError("No NZ data in session — load a manifold first")
        entries = [
            (q.m_ext, q.e_ext, q.result)
            for q in session.index_queries
            if q.result is not None
        ]
        dehn_results = _ExportService_build_dehn_results(session) if include_filling else None
        _exp_mod.write_full_report(
            path=output_path,
            manifold_data=_ExportService_manifold_data(session),
            easy_result=None,          # easy_result not stored in session
            nz_data=session.nz_data,
            entries=entries,
            weyl_result=session.weyl_result,
            dehn_results=dehn_results or None,
            q_order_half=session.q_order_half,
        )

    # ------------------------------------------------------------------
    # Clipboard helpers
    # ------------------------------------------------------------------

    @staticmethod
    def clipboard_latex(result: Any, num_hard: int) -> str:
        """Return a LaTeX string for *one* RefinedIndexResult.

        Suitable for pasting into a LaTeX document.  Does not produce a
        full ``align*`` environment — just the series expression.

        Parameters
        ----------
        result : RefinedIndexResult  (a dict {tuple: int})
        num_hard : int               number of η variables
        """
        return _exp_mod.to_latex_series(result, num_hard)

    @staticmethod
    def clipboard_plain(result: Any, num_hard: int) -> str:
        """Return a plain-text representation for *one* RefinedIndexResult.

        Uses the internal ``_plain_series`` helper from exporters.py.
        """
        return _exp_mod._plain_series(result, num_hard)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _ManifoldDataStub:
    """Minimal stand-in for ManifoldData when the live object is not available
    (e.g. after a session restore from disk, where manifold_data is not
    serialised).  Exporters only use .name and .gluing_matrix."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.gluing_matrix = None


def _ExportService_manifold_data(session: "Session") -> Any:
    """Return session.manifold_data, falling back to a lightweight stub."""
    if session.manifold_data is not None:
        return session.manifold_data
    return _ManifoldDataStub(session.manifold_name or "unknown")


def _ExportService_build_dehn_results(session: "Session") -> list:
    """Build a dehn_results list from FillQuery records.

    The exporters.py functions check ``isinstance(res, TransformedFillResult)``
    which requires objects from ``app/workers``.  Those worker result types are
    only fully defined once Phase 7 (Pipeline Cards) is implemented.

    Until then, this helper attempts a lazy import and returns [] on failure
    so that calls to write_* gracefully omit filling data rather than crashing.
    This is correct behaviour in Phase 2; the export_card.py (Phase 7) will
    supply pre-built result objects directly.

    Returns
    -------
    list
        A list of TransformedFillResult-compatible objects, or [] if the
        app workers module is not yet available.
    """
    queries = [q for q in session.fill_queries if q.result is not None]
    if not queries:
        return []

    # Try to import the app-layer result types.
    # Graceful fallback on ImportError so early phases don't crash.
    try:
        from manifold_index.app.workers import (  # type: ignore[attr-defined]
            TransformedFillResult,
            UnrefinedFillResult,
        )
    except (ImportError, AttributeError):
        return []

    shims: list[Any] = []
    for fq in queries:
        if fq.unrefined_fallback:
            shim = UnrefinedFillResult.__new__(UnrefinedFillResult)
            shim.cusp_idx = fq.cusp_idx
            shim.P_user = fq.user_P
            shim.Q_user = fq.user_Q
            shim.fill_results = [(fq.m_other, fq.e_other, fq.result)]
            shims.append(shim)
            continue

        try:
            from manifold_index.core.dehn_filling import find_rs
            R, S = find_rs(fq.nc_P, fq.nc_Q)
        except Exception:
            R, S = 0, 1

        shim = TransformedFillResult.__new__(TransformedFillResult)
        shim.cusp_idx = fq.cusp_idx
        shim.P_user = fq.user_P
        shim.Q_user = fq.user_Q
        shim.P_nc = fq.nc_P
        shim.Q_nc = fq.nc_Q
        shim.R = R
        shim.S = S
        shim.p = fq.p
        shim.q = fq.q
        shim.weyl_a_phys = fq.weyl_a
        shim.weyl_b_phys = fq.weyl_b
        # fill_results: list of (m_other, e_other, FilledRefinedResult)
        shim.fill_results = [(fq.m_other, fq.e_other, fq.result)]
        shims.append(shim)
    return shims

