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
        cancel_check: Callable[[], None] | None = None,
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
            cancel_check=cancel_check,
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
        incompat_edges: list[int] | None = None,
        auto_precompute: bool = True,
        progress_fn: Callable | None = None,
        manifold_name: str = "unknown",
        weyl_ab: Any = None,
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
        manifold_name : str — used to name the I^ref disk-cache file

        Returns
        -------
        (p, q, FilledRefinedResult)
        where p, q are the slope in the NC basis and
        FilledRefinedResult = ``dict[tuple[int, ...], int]``.
        """
        # ── Basis change: (α,β) → (γ,δ) using NC cycle (nc_P, nc_Q) ──
        # find_rs(P, Q) returns (R, S) with R·Q − P·S = 1.
        # We apply the SL(2,ℤ) basis change with matrix [[nc_P, nc_Q], [-R, -S]]:
        #
        #   new_μ   = nc_P·μ + nc_Q·λ          (new meridian  = NC cycle)
        #   new_λ   =  -R·μ  −  S·λ            (new longitude, Bézout conjugate)
        #
        # In the NZ (position, momentum) = (M, L/2) convention this becomes:
        #   new_pos = nc_P·old_pos + 2·nc_Q·old_mom
        #   new_mom = (−R/2)·old_pos + (−S)·old_mom
        #
        # Inverting: old_pos = −S·new_pos − 2·nc_Q·new_mom
        #            old_mom = (R/2)·new_pos + nc_P·new_mom
        #
        # Substituting into user_P·old_pos + 2·user_Q·old_mom = c gives the
        # user slope in the new basis:
        #   p = R·user_Q − S·user_P
        #   q = nc_P·user_Q − nc_Q·user_P

        R, S = _df_mod.find_rs(nc_P, nc_Q)   # R·nc_Q − nc_P·S = 1
        p = R * user_Q - S * user_P
        q = nc_P * user_Q - nc_Q * user_P

        # apply_general_cusp_basis_change works for any parity of nc_P.
        nz_nc = _nz_mod.apply_general_cusp_basis_change(
            nz_data, cusp_idx, a=nc_P, b=nc_Q, c=-R, d=-S
        )

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
            weyl_ab=weyl_ab,
            incompat_edges=incompat_edges,
            auto_precompute=auto_precompute,
        )

        # Persist newly computed I^ref entries to the disk cache so they are
        # reloaded automatically in future sessions.
        try:
            _kc_mod.save_iref_cache(nz_nc, manifold_name=manifold_name)
        except Exception:
            pass  # best-effort — never let a cache write break the result

        return p, q, result

    @staticmethod
    def compute_unrefined_kernel_fill(
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
        weyl_ab: Any = None,
        incompat_edges: list[int] | None = None,
        manifold_name: str = "unknown",
    ) -> tuple[int, int, Any]:
        """Apply basis change and compute filling with unrefined K(P,Q) kernel + I^ref.

        Used when the q^1 SU(2)-adjoint projection fails: the refined IS-chain
        kernel K^ref is replaced by the ordinary unrefined kernel K(P,Q;m,e),
        but I^ref (with η variables) is still used so hard-edge fugacities are
        preserved.  Incompat η edges are projected to 1 pre-kernel.

        Returns
        -------
        (p, q, FilledRefinedResult)  — same shape as compute_filled_index.
        """
        R, S = _df_mod.find_rs(nc_P, nc_Q)
        p = R * user_Q - S * user_P
        q = nc_P * user_Q - nc_Q * user_P

        nz_nc = _nz_mod.apply_general_cusp_basis_change(
            nz_data, cusp_idx, a=nc_P, b=nc_Q, c=-R, d=-S
        )

        result = _rdf_mod.compute_unrefined_kernel_refined_index(
            nz_nc,
            cusp_idx=cusp_idx,
            P=p,
            Q=q,
            m_other=m_other,
            e_other=e_other,
            q_order_half=q_order_half,
            incompat_edges=incompat_edges,
            weyl_a=weyl_a,
            weyl_b=weyl_b,
            weyl_ab=weyl_ab,
        )

        try:
            _kc_mod.save_iref_cache(nz_nc, manifold_name=manifold_name)
        except Exception:
            pass

        return p, q, result

    @staticmethod
    def compute_multi_cusp_filled_index(
        nz_data: Any,
        cusp_specs: list[dict],
        q_order_half: int,
        auto_precompute: bool = True,
        progress_fn: Callable | None = None,
        manifold_name: str = "unknown",
        m_unfilled: list | None = None,
        e_unfilled: list | None = None,
    ) -> tuple[list[dict], Any]:
        """Sequentially fill multiple cusps of a manifold (supports any number).

        For each cusp in cusp_specs:
        1. Apply NC basis change (apply_general_cusp_basis_change)
        2. Compute slope transform to get (p, q) in NC basis
        3. Build MultiCuspFillSpec

        Then call compute_multi_cusp_filled_refined_index sequentially for 2-cusp groups.
        For >2 cusps, chains multiple 2-cusp fillings together.

        Parameters
        ----------
        nz_data : NeumannZagierData
        cusp_specs : list[dict]
            Each dict has keys:
            - cusp_idx : int
            - nc_P, nc_Q : int (NC cycle in (α, β) basis)
            - user_P, user_Q : int (user slope in (α, β) basis)
            - weyl_a, weyl_b (optional) : list[Fraction] | None
            - incompat_edges (optional) : list[int] | None
        q_order_half : int
        auto_precompute : bool
        progress_fn : callable | None
        manifold_name : str

        Returns
        -------
        (augmented_cusp_specs, FilledRefinedResult)
            where augmented_cusp_specs has p, q added to each dict.
        """
        nz_transformed = nz_data
        fill_specs = []
        augmented = []

        for spec in cusp_specs:
            cusp_idx = spec["cusp_idx"]
            nc_P, nc_Q = spec["nc_P"], spec["nc_Q"]
            user_P, user_Q = spec["user_P"], spec["user_Q"]

            R, S = _df_mod.find_rs(nc_P, nc_Q)  # R·nc_Q − nc_P·S = 1
            p = R * user_Q - S * user_P
            q = nc_P * user_Q - nc_Q * user_P

            nz_transformed = _nz_mod.apply_general_cusp_basis_change(
                nz_transformed, cusp_idx, a=nc_P, b=nc_Q, c=-R, d=-S
            )

            weyl_a = spec.get("weyl_a")
            weyl_b = spec.get("weyl_b")
            weyl_ab = spec.get("weyl_ab")
            incompat_edges = spec.get("incompat_edges")

            fill_specs.append(_rdf_mod.MultiCuspFillSpec(
                cusp_idx=cusp_idx, P=p, Q=q,
                weyl_a=weyl_a, weyl_b=weyl_b,
                weyl_ab=weyl_ab,
                incompat_edges=incompat_edges,
            ))
            augmented.append({**spec, "p": p, "q": q})

        # Call core multi-cusp filling (now supports any number of cusps via sequential algorithm)
        result = _rdf_mod.compute_multi_cusp_filled_refined_index(
            nz_transformed,
            fill_specs=fill_specs,
            q_order_half=q_order_half,
            auto_precompute=auto_precompute,
            progress_callback=progress_fn,
            m_unfilled=m_unfilled,
            e_unfilled=e_unfilled,
        )

        # Best-effort I^ref cache save
        try:
            _kc_mod.save_iref_cache(nz_transformed, manifold_name=manifold_name)
        except Exception:
            pass

        return augmented, result

    @staticmethod
    def compute_unrefined_filled_index(
        nz_data: Any,
        cusp_idx: int,
        user_P: int,
        user_Q: int,
        m_other: list[int] | None = None,
        e_other: list[Fraction] | None = None,
        q_order_half: int = 10,
    ) -> Any:
        """Unrefined (3D) Dehn filling — no basis change, no η variables.

        Parameters
        ----------
        nz_data : NeumannZagierData  (original basis, unchanged)
        cusp_idx : int
        user_P, user_Q : int  — filling slope in original (α, β) basis
        m_other, e_other : charges for unfilled cusps (None → all zeros)
        q_order_half : int

        Returns
        -------
        FilledIndexResult
        """
        return _df_mod.compute_filled_index(
            nz_data,
            cusp_idx=cusp_idx,
            P=user_P,
            Q=user_Q,
            m_other=m_other,
            e_other=e_other,
            q_order_half=q_order_half,
        )

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