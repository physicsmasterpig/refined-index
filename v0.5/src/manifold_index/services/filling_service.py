"""
services.filling_service
========================
Orchestration layer for NC cycle search, basis change, and filled index
computation.

Rule: no PySide6, no app/ imports.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Any, Callable

from manifold_index.core import (
    dehn_filling as _df_mod,
    refined_dehn_filling as _rdf_mod,
    kernel_cache as _kc_mod,
    neumann_zagier as _nz_mod,
    basis_selection as _bs_mod,
)


class FillingService:
    """Orchestrates NC search → basis change → kernel → filled index."""

    # ── NC cycle search ───────────────────────────────────────────────

    @staticmethod
    def find_nc_cycles(
        nz_data: Any,
        cusp_idx: int,
        p_range: tuple[int, int],
        q_range: tuple[int, int],
        q_order_half: int,
        progress_fn: Callable[[int, int], None] | None = None,
    ) -> Any:
        """Run ``find_non_closable_cycles`` for *cusp_idx* over the given ranges.

        Parameters
        ----------
        nz_data : NeumannZagierData
        cusp_idx : int
        p_range : (lo, hi) inclusive — searched as ``range(lo, hi+1)``
        q_range : (lo, hi) inclusive — searched as ``range(lo, hi+1)``
        q_order_half : int
        progress_fn : optional callback ``(done, total) -> None``

        Returns
        -------
        NonClosableCycleResult
        """
        p_lo, p_hi = p_range
        q_lo, q_hi = q_range
        return _df_mod.find_non_closable_cycles(
            nz_data,
            cusp_idx=cusp_idx,
            p_range=range(p_lo, p_hi + 1),
            q_range=range(q_lo, q_hi + 1),
            q_order_half=q_order_half,
            progress_fn=progress_fn,
        )

    @staticmethod
    def load_nc_from_cache(
        name: str,
        nz_data: Any,
        cusp_idx: int,
        q_order_half: int,
    ) -> list[Any] | None:
        """Load NC cycles for *cusp_idx* from disk.

        Returns ``list[NonClosableCycle]`` or ``None`` if not cached.
        """
        try:
            result = _kc_mod.load_nc_cycle_cache(
                nz_data,
                manifold_name=name,
                q_order_half=q_order_half,
            )
            if result is None:
                return None
            # result is a list[NonClosableCycleResult], one per cusp
            for ncr in result:
                if hasattr(ncr, "cusp_idx") and ncr.cusp_idx == cusp_idx:
                    return list(ncr.cycles)
                # dict payload from older cache format
                if isinstance(ncr, dict) and ncr.get("cusp_idx") == cusp_idx:
                    return ncr.get("cycles", [])
        except Exception:
            pass
        return None

    # ── Kernel probing ────────────────────────────────────────────────

    @staticmethod
    def probe_kernel(P: int, Q: int, q_order_half: int) -> dict:
        """Check kernel cache for the slope (P, Q) at *q_order_half*.

        Returns
        -------
        dict:
            ``{"available": bool, "cached_qq": int | None, "hj_length": int}``
        """
        from manifold_index.core.refined_dehn_filling import hj_continued_fraction

        hj_ks = hj_continued_fraction(P, Q)
        hj_len = len(hj_ks)

        cached_kernels = _kc_mod.list_cached_kernels()
        cached_qq: int | None = None
        for kP, kQ, kqq in cached_kernels:
            if kP == P and kQ == Q and kqq == q_order_half:
                cached_qq = kqq
                break

        return {
            "available": cached_qq is not None,
            "cached_qq": cached_qq,
            "hj_length": hj_len,
        }

    # ── Filled index ──────────────────────────────────────────────────

    @staticmethod
    def compute_filled_index(
        nz_data: Any,
        cusp_idx: int,
        nc_P: int,
        nc_Q: int,
        user_P: int,
        user_Q: int,
        m_other: list[int] | None,
        e_other: list[Fraction] | None,
        q_order_half: int,
        weyl_a: list[Fraction] | None,
        weyl_b: list[Fraction] | None,
        auto_precompute: bool = True,
        progress_fn: Callable | None = None,
    ) -> tuple[int, int, Any]:
        """Apply basis change and compute the filled refined index.

        The NC cycle (nc_P, nc_Q) defines the filling basis for *cusp_idx*.
        The user's slope (user_P, user_Q) is expressed in the (α, β) basis
        (before basis change).  After the basis change to the NC basis (γ, δ),
        the user slope becomes (p, q).

        Parameters
        ----------
        nz_data : NeumannZagierData  (original, before basis change)
        cusp_idx : int
        nc_P, nc_Q : int  — NC cycle in (α, β) basis
        user_P, user_Q : int  — user slope in (α, β) basis
        m_other, e_other : charges for unfilled cusps (None → all zeros)
        q_order_half : int
        weyl_a, weyl_b : Weyl vectors (None → no Weyl shift)
        auto_precompute : bool — automatically build kernel if not cached
        progress_fn : optional progress callback

        Returns
        -------
        (p, q, FilledRefinedResult)
        where p, q are the slope in the NC basis and
        FilledRefinedResult = ``dict[tuple[int, ...], int]``.
        """
        # ── Basis change: (α,β) → (γ,δ) using NC cycle (nc_P, nc_Q) ──
        # The NC basis change applies apply_cusp_basis_change(nz, cusp, nc_P, nc_Q)
        # which makes γ = nc_P·α + nc_Q·β the new position variable.
        # In the new basis the user slope (user_P, user_Q) becomes (p, q)
        # via the inverse of the SL(2,Z) matrix [[nc_P, nc_Q], [R, S]]
        # where nc_P·S - nc_Q·R = 1:
        #   [p, q] = [[S, -nc_Q], [-R, nc_P]] @ [user_P, user_Q]

        # Find R, S: nc_P·S - nc_Q·R = 1
        R, S = _df_mod.find_rs(nc_P, nc_Q)
        p = S * user_P - nc_Q * user_Q
        q = -R * user_P + nc_P * user_Q

        # Apply the basis change to nz_data (only if nc_P is odd)
        if nc_P % 2 != 0:
            nz_nc = _nz_mod.apply_cusp_basis_change(nz_data, cusp_idx, nc_P, nc_Q)
        else:
            # Even nc_P: basis change not strictly required; evaluate at (p, q) directly
            nz_nc = nz_data
            p, q = user_P, user_Q

        # ── Compute filled refined index ──────────────────────────────
        # After the basis change, the filling slope is (p, 1) in the NZ kernel
        # convention, i.e. we fill along p·γ + q·δ = p·pos + 2q·mom.
        result = _rdf_mod.compute_filled_refined_index(
            nz_nc,
            cusp_idx=cusp_idx,
            P=p,
            Q=q,
            m_other=m_other,
            e_other=e_other,
            q_order_half=q_order_half,
            weyl_a=weyl_a,
            weyl_b=weyl_b,
            auto_precompute=auto_precompute,
        )

        return p, q, result

    # ── Utilities ─────────────────────────────────────────────────────

    @staticmethod
    def canonicalise_nc_cycles(cycles: list[Any]) -> list[Any]:
        """Deduplicate: keep one from each ``{(P,Q), (-P,-Q)}`` pair.

        The canonical representative is the one whose first nonzero
        coordinate is positive.

        Parameters
        ----------
        cycles : list[NonClosableCycle]

        Returns
        -------
        list[NonClosableCycle] — deduplicated, sorted by (|P|, |Q|).
        """
        seen: set[tuple[int, int]] = set()
        result: list[Any] = []
        for nc in cycles:
            P, Q = nc.P, nc.Q
            # Canonical: first nonzero coordinate is positive
            first_nz = P if P != 0 else Q
            if first_nz < 0:
                P, Q = -P, -Q
            key = (P, Q)
            if key not in seen:
                seen.add(key)
                result.append(nc)
        result.sort(key=lambda nc: (abs(nc.P), abs(nc.Q)))
        return result