"""
app/workers.py — Background QThread workers for computation.

Workers
-------
RefinedIndexWorker  — Compute refined index at every (m_ext, e_ext) point.
DehnFillingWorker   — Find NC cycles → transform user slope → compute filled index.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product as itertools_product
from math import gcd

from PySide6.QtCore import QThread, Signal


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation grid
# ═══════════════════════════════════════════════════════════════════════════

def build_eval_grid(r: int) -> list[tuple[list[int], list[Fraction]]]:
    """Build the 25^r evaluation grid for refined index + Weyl extraction.

    For each cusp:
        m ∈ {-2, -1, 0, 1, 2}
        e ∈ {-1, -1/2, 0, 1/2, 1}

    Returns list of (m_ext, e_ext) tuples.
    """
    per_cusp = [
        (m, Fraction(k, 2))
        for m in (-2, -1, 0, 1, 2)
        for k in (-2, -1, 0, 1, 2)
    ]
    eval_points: list[tuple[list[int], list[Fraction]]] = []
    for combo in itertools_product(*([per_cusp] * r)):
        m_list = [pair[0] for pair in combo]
        e_list = [pair[1] for pair in combo]
        eval_points.append((m_list, e_list))
    return eval_points


# ═══════════════════════════════════════════════════════════════════════════
# Worker 1: Refined Index computation
# ═══════════════════════════════════════════════════════════════════════════

class RefinedIndexWorker(QThread):
    """Compute refined index for every (m_ext, e_ext) in the evaluation grid.

    Emits:
        status(str)        — human-readable stage description
        progress(int, int) — (done, total)
        finished(list)     — list of (m_ext, e_ext, RefinedIndexResult)
        error(str)
    """

    status = Signal(str)
    progress = Signal(int, int)
    finished = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        nz_data,
        eval_points: list[tuple[list[int], list[Fraction]]],
        q_order_half: int,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._nz_data = nz_data
        self._eval_points = eval_points
        self._q_order_half = q_order_half

    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:
            import traceback
            self.error.emit(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}")

    def _run(self) -> None:
        from manifold_index.core.refined_index import compute_refined_index

        total = len(self._eval_points)
        results = []
        for idx, (m_ext, e_ext) in enumerate(self._eval_points):
            self.status.emit(f"Computing sector {idx + 1}/{total}…")
            self.progress.emit(idx + 1, total)
            result = compute_refined_index(
                self._nz_data, m_ext, e_ext, self._q_order_half,
            )
            results.append((m_ext, e_ext, result))

        self.status.emit(f"Done — {total} sectors computed.")
        self.finished.emit(results)


# ═══════════════════════════════════════════════════════════════════════════
# Worker 2: Dehn filling (NC search → slope transform → compute)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TransformedFillResult:
    """Result of filling at the user's slope, transformed into an NC cycle's basis.

    Given NC cycle γ = P_nc·α + Q_nc·β with complement δ = R·α + S·β
    such that det [[P_nc, R], [Q_nc, S]] = +1 (SL(2,ℤ) convention),
    the user's slope P_user·α + Q_user·β is expressed as p·γ + q·δ.
    The filled refined index is computed at slope (p, q) for each external
    charge configuration on the unfilled cusps.

    Attributes
    ----------
    fill_results : list[tuple[list[int], list, FilledRefinedResult]]
        Each entry is ``(m_other, e_other, fr)`` where ``m_other`` and
        ``e_other`` are the charges on the *unfilled* cusps.
        For a 1-cusp manifold this is a single entry with empty lists.
    weyl_a_phys : list[Fraction] | None
        Physical Weyl vector *a* (per hard edge).  Compatible iff ``a[j] ∈ ℤ``.
        None if unavailable.
    weyl_b_phys : list[Fraction] | None
        Physical Weyl vector *b* (per hard edge).  Compatible iff ``2·b[j] ∈ ℤ``.
        None if unavailable.
    """
    cusp_idx: int
    P_nc: int       # NC cycle in (α, β) basis
    Q_nc: int
    R: int          # complement δ in (α, β) basis (R·Q_nc − P_nc·S = 1)
    S: int
    p: int          # user's slope in (γ, δ) basis
    q: int
    P_user: int     # user's original slope in (α, β) basis
    Q_user: int
    fill_results: list  # list[tuple[list[int], list, FilledRefinedResult]]
    weyl_a_phys: list | None = None  # a_physical per hard edge
    weyl_b_phys: list | None = None  # b_physical per hard edge


@dataclass
class CuspNCInfo:
    """Per-cusp NC cycle and basis info for multi-cusp filling."""
    cusp_idx: int
    P_nc: int
    Q_nc: int
    R: int
    S: int
    p: int          # user's slope in (γ, δ) basis
    q: int
    P_user: int
    Q_user: int
    weyl_a_phys: list | None = None
    weyl_b_phys: list | None = None


@dataclass
class MultiCuspFillResult:
    """Result of simultaneously filling multiple cusps.

    Each entry in ``cusp_info`` describes one cusp's NC cycle and basis.
    ``fill_result`` is the single combined filled refined index.
    """
    cusp_info: list[CuspNCInfo]
    fill_result: object  # FilledRefinedResult


def _canonicalize_nc_cycles(cycles: list) -> list:
    """Deduplicate NC cycles: keep one from each {(P,Q), (−P,−Q)} pair.

    Canonical representative: Q > 0, or (Q = 0 and P > 0).
    """
    seen: set[tuple[int, int]] = set()
    canonical = []
    for cyc in cycles:
        P, Q = cyc.P, cyc.Q
        # Determine canonical form
        if Q > 0 or (Q == 0 and P > 0):
            key = (P, Q)
        else:
            key = (-P, -Q)
        if key not in seen:
            seen.add(key)
            # Always store the canonical version
            from manifold_index.core.dehn_filling import NonClosableCycle
            canonical.append(NonClosableCycle(
                cusp_idx=cyc.cusp_idx, P=key[0], Q=key[1],
            ))
    return canonical


class DehnFillingWorker(QThread):
    """Dehn filling worker: NC search → slope transform → compute.

    Workflow:
      1. Search non-closable cycles over the given (P,Q) range.
      2. Deduplicate: keep one from each {γ, −γ} pair.
      3. For each NC cycle γ_i, transform the user's slope into the
         {γ_i, δ_i} basis and compute the filled refined index at the
         transformed coordinates.

    Emits:
        status(str)
        progress(int, int)
        nc_found(object)       — list[NonClosableCycleResult] (deduplicated)
        finished(object)       — list[TransformedFillResult]
        error(str)
    """

    status = Signal(str)
    progress = Signal(int, int)
    nc_found = Signal(object)
    finished = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        nz_data,
        cusp_configs: list[dict],
        q_order_half: int,
        p_range: range,
        q_range: range,
        manifold_name: str = "unknown",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._nz_data = nz_data
        self._cusp_configs = cusp_configs
        self._q_order_half = q_order_half
        self._p_range = p_range
        self._q_range = q_range
        self._manifold_name = manifold_name

    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:
            import traceback
            self.error.emit(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}")

    def _run(self) -> None:
        from manifold_index.core.dehn_filling import (
            NonClosableCycleResult,
            NonClosableCycle,
            _candidate_slopes,
            compute_filled_index,
            find_rs,
        )
        from manifold_index.core.refined_dehn_filling import (
            compute_filled_refined_index,
            compute_multi_cusp_filled_refined_index,
            MultiCuspFillSpec,
        )
        from manifold_index.core.neumann_zagier import (
            apply_general_cusp_basis_change,
        )
        from manifold_index.core.weyl_check import compute_ab_vectors_for_cusp

        nz = self._nz_data
        r = nz.r
        q_order_half = self._q_order_half
        active = [c for c in self._cusp_configs if c.get("fill", False)]
        n_filling = len(active)
        all_cusps_filled = (n_filling == r)

        # ── Step 1: Search non-closable cycles per cusp ───────
        nc_results = []
        for cfg in active:
            cusp_idx = cfg["cusp_idx"]
            self.status.emit(
                f"Step 1/2 — Searching NC cycles at cusp {cusp_idx}…"
            )

            m_other = [0] * (r - 1)
            e_other = [0] * (r - 1)

            compute_slopes = _candidate_slopes(
                self._p_range, self._q_range, canonical_only=True,
            )
            all_slopes = _candidate_slopes(
                self._p_range, self._q_range, canonical_only=False,
            )
            total = len(compute_slopes)

            result = NonClosableCycleResult(cusp_idx=cusp_idx)
            result.slopes_tested = all_slopes[:]
            computed: dict[tuple[int, int], bool] = {}

            for done_idx, (P, Q) in enumerate(compute_slopes):
                filled = compute_filled_index(
                    nz, cusp_idx=cusp_idx, P=P, Q=Q,
                    m_other=list(m_other), e_other=list(e_other),
                    q_order_half=q_order_half,
                )
                non_closable = filled.is_stably_zero()
                computed[(P, Q)] = non_closable

                if non_closable:
                    result.cycles.append(NonClosableCycle(
                        cusp_idx=cusp_idx, P=P, Q=Q,
                    ))

                neg = (-P, -Q)
                if neg in set(all_slopes) and neg not in computed:
                    computed[neg] = non_closable
                    if non_closable:
                        result.cycles.append(NonClosableCycle(
                            cusp_idx=cusp_idx, P=-P, Q=-Q,
                        ))

                self.progress.emit(done_idx + 1, total)

            result.cycles = _canonicalize_nc_cycles(result.cycles)
            nc_results.append(result)
            n_nc = len(result.cycles)
            self.status.emit(f"Cusp {cusp_idx}: {n_nc} NC cycle(s) (deduplicated)")

        self.nc_found.emit(nc_results)

        # ── Step 2: compute filled index ──────────────────────
        # Build lookup: cusp_idx → user's (P, Q)
        user_slopes: dict[int, tuple[int, int]] = {}
        for cfg in active:
            user_slopes[cfg["cusp_idx"]] = (cfg["P"], cfg["Q"])

        if all_cusps_filled and n_filling >= 2:
            self._run_multi_cusp(
                nz, nc_results, user_slopes, q_order_half,
            )
        else:
            self._run_single_cusp(
                nz, nc_results, user_slopes, q_order_half,
            )

    # ------------------------------------------------------------------
    # Multi-cusp filling: all cusps filled simultaneously
    # ------------------------------------------------------------------

    def _run_multi_cusp(
        self,
        nz,
        nc_results: list,
        user_slopes: dict[int, tuple[int, int]],
        q_order_half: int,
    ) -> None:
        """Fill all cusps simultaneously via sequential kernel application."""
        from manifold_index.core.dehn_filling import find_rs
        from manifold_index.core.refined_dehn_filling import (
            compute_multi_cusp_filled_refined_index,
            MultiCuspFillSpec,
        )
        from manifold_index.core.neumann_zagier import (
            apply_general_cusp_basis_change,
        )
        from manifold_index.core.weyl_check import compute_ab_vectors_for_cusp

        r = nz.r

        # Collect NC cycles per cusp
        nc_by_cusp: dict[int, list] = {}
        for nc_res in nc_results:
            nc_by_cusp[nc_res.cusp_idx] = nc_res.cycles

        # Build all NC cycle combinations (one per filled cusp)
        filled_cusps = sorted(user_slopes.keys())
        nc_lists = [nc_by_cusp.get(ci, []) for ci in filled_cusps]

        # Cartesian product of NC cycles across cusps
        from itertools import product as itertools_product
        nc_combos = list(itertools_product(*nc_lists))

        if not nc_combos:
            self.status.emit("No NC cycle combinations found.")
            self.finished.emit([])
            return

        multi_results: list[MultiCuspFillResult] = []
        total_combos = len(nc_combos)

        for combo_idx, nc_combo in enumerate(nc_combos):
            self.status.emit(
                f"Step 2/2 — NC combination {combo_idx + 1}/{total_combos}…"
            )
            self.progress.emit(combo_idx, total_combos)

            # For each cusp in this combo: basis change + slope transform
            nz_transformed = nz
            cusp_info_list: list[CuspNCInfo] = []
            fill_specs: list[MultiCuspFillSpec] = []

            for cusp_order, cyc in enumerate(nc_combo):
                cusp_idx = filled_cusps[cusp_order]
                P_nc, Q_nc = cyc.P, cyc.Q
                P_user, Q_user = user_slopes[cusp_idx]

                R0, S0 = find_rs(P_nc, Q_nc)
                R_comp, S_comp = -R0, -S0

                p = S_comp * P_user - R_comp * Q_user
                q = -Q_nc * P_user + P_nc * Q_user

                # Apply cusp basis change
                nz_transformed = apply_general_cusp_basis_change(
                    nz_transformed, cusp_idx,
                    a=P_nc, b=Q_nc, c=R_comp, d=S_comp,
                )

                # Extract Weyl vectors for this cusp
                self.status.emit(
                    f"  Cusp {cusp_idx}, NC ({P_nc},{Q_nc}): "
                    f"extracting Weyl vectors…"
                )
                ab_nc = compute_ab_vectors_for_cusp(
                    nz_transformed, cusp_idx, q_order_half=q_order_half,
                )

                weyl_a_phys = list(ab_nc.a) if ab_nc else None
                weyl_b_phys = list(ab_nc.b) if ab_nc else None

                incompat_edges = []
                weyl_a_fill = None
                weyl_b_fill = None
                if ab_nc is not None:
                    incompat_edges = [
                        j for j, ok in enumerate(ab_nc.edge_compatible)
                        if not ok
                    ]
                    ab_compat = ab_nc.make_filling_compatible()
                    weyl_a_fill = list(ab_compat.a)
                    weyl_b_fill = list(ab_compat.b)
                    if all(
                        a == 0 and b == 0
                        for a, b in zip(weyl_a_fill, weyl_b_fill)
                    ):
                        weyl_a_fill = None
                        weyl_b_fill = None

                cusp_info_list.append(CuspNCInfo(
                    cusp_idx=cusp_idx,
                    P_nc=P_nc, Q_nc=Q_nc,
                    R=R_comp, S=S_comp,
                    p=p, q=q,
                    P_user=P_user, Q_user=Q_user,
                    weyl_a_phys=weyl_a_phys,
                    weyl_b_phys=weyl_b_phys,
                ))
                fill_specs.append(MultiCuspFillSpec(
                    cusp_idx=cusp_idx,
                    P=p, Q=q,
                    weyl_a=weyl_a_fill,
                    weyl_b=weyl_b_fill,
                    incompat_edges=incompat_edges if incompat_edges else None,
                ))

            # Compute the combined filling
            self.status.emit(
                f"  Computing sequential filling for combination "
                f"{combo_idx + 1}/{total_combos}…"
            )

            try:
                combined = compute_multi_cusp_filled_refined_index(
                    nz_transformed,
                    fill_specs=fill_specs,
                    q_order_half=q_order_half,
                    verbose=False,
                    progress_callback=lambda msg: self.status.emit(f"  {msg}"),
                    auto_precompute=True,
                    cache_iref=True,
                    manifold_name=self._manifold_name,
                )
                multi_results.append(MultiCuspFillResult(
                    cusp_info=cusp_info_list,
                    fill_result=combined,
                ))
            except Exception as exc:
                self.status.emit(
                    f"  Error in combination {combo_idx + 1}: {exc}"
                )

            self.progress.emit(combo_idx + 1, total_combos)

        self.status.emit(
            f"Done — {total_combos} NC combination(s), "
            f"{len(multi_results)} result(s)."
        )
        self.finished.emit(multi_results)

    # ------------------------------------------------------------------
    # Single-cusp filling (original per-cusp logic)
    # ------------------------------------------------------------------

    def _run_single_cusp(
        self,
        nz,
        nc_results: list,
        user_slopes: dict[int, tuple[int, int]],
        q_order_half: int,
    ) -> None:
        """Fill cusps independently (when not all cusps are filled)."""
        from manifold_index.core.dehn_filling import find_rs
        from manifold_index.core.refined_dehn_filling import (
            compute_filled_refined_index,
        )
        from manifold_index.core.neumann_zagier import (
            apply_general_cusp_basis_change,
        )
        from manifold_index.core.weyl_check import compute_ab_vectors_for_cusp

        r = nz.r

        _FILL_DISPLAY_CHARGES = [
            (0, Fraction(0)),
            (1, Fraction(0)),
            (0, Fraction(1, 2)),
            (2, Fraction(0)),
            (0, Fraction(1)),
        ]

        all_nc = [
            (nc.cusp_idx, cyc) for nc in nc_results for cyc in nc.cycles
        ]

        transformed_results: list[TransformedFillResult] = []
        total_jobs = 0
        for cusp_idx, _cyc in all_nc:
            n_unfilled = r - 1
            n_combos = len(_FILL_DISPLAY_CHARGES) ** n_unfilled if n_unfilled > 0 else 1
            total_jobs += n_combos
        done_jobs = 0

        for cusp_idx, cyc in all_nc:
            P_nc, Q_nc = cyc.P, cyc.Q
            P_user, Q_user = user_slopes[cusp_idx]

            R0, S0 = find_rs(P_nc, Q_nc)
            R, S = -R0, -S0

            p = S * P_user - R * Q_user
            q = -Q_nc * P_user + P_nc * Q_user

            nz_nc = apply_general_cusp_basis_change(
                nz, cusp_idx, a=P_nc, b=Q_nc, c=R, d=S,
            )

            self.status.emit(
                f"Step 2/2 — Cusp {cusp_idx}, NC ({P_nc},{Q_nc}): "
                f"extracting Weyl vectors…"
            )
            ab_nc = compute_ab_vectors_for_cusp(
                nz_nc, cusp_idx, q_order_half=q_order_half,
            )
            weyl_a_phys = list(ab_nc.a) if ab_nc else None
            weyl_b_phys = list(ab_nc.b) if ab_nc else None

            incompat_edges = []
            weyl_a_nc = None
            weyl_b_nc = None
            if ab_nc is not None:
                incompat_edges = [
                    j for j, ok in enumerate(ab_nc.edge_compatible) if not ok
                ]
                ab_compat = ab_nc.make_filling_compatible()
                weyl_a_nc = list(ab_compat.a)
                weyl_b_nc = list(ab_compat.b)
                if all(a == 0 and b == 0 for a, b in zip(weyl_a_nc, weyl_b_nc)):
                    weyl_a_nc = None
                    weyl_b_nc = None

            n_unfilled = r - 1
            if n_unfilled == 0:
                ext_combos: list[tuple[list[int], list[Fraction]]] = [([], [])]
            else:
                ext_combos = []
                for combo in itertools_product(
                    _FILL_DISPLAY_CHARGES, repeat=n_unfilled,
                ):
                    m_o = [pair[0] for pair in combo]
                    e_o = [pair[1] for pair in combo]
                    ext_combos.append((m_o, e_o))

            fill_results: list[tuple[list[int], list, object]] = []
            for m_o, e_o in ext_combos:
                self.status.emit(
                    f"Step 2/2 — Cusp {cusp_idx}, "
                    f"NC ({P_nc},{Q_nc}) → ({p},{q}), "
                    f"ext ({list(m_o)},{list(e_o)})…"
                )
                done_jobs += 1
                self.progress.emit(done_jobs, total_jobs)

                try:
                    fr = compute_filled_refined_index(
                        nz_data=nz_nc,
                        cusp_idx=cusp_idx,
                        P=p, Q=q,
                        m_other=list(m_o) if m_o else None,
                        e_other=list(e_o) if e_o else None,
                        q_order_half=q_order_half,
                        weyl_a=weyl_a_nc,
                        weyl_b=weyl_b_nc,
                        auto_precompute=True,
                        cache_iref=True,
                        manifold_name=self._manifold_name,
                    )
                    if incompat_edges:
                        fr = fr.collapse_eta_edges(incompat_edges)
                    fill_results.append((list(m_o), list(e_o), fr))
                except Exception as exc:
                    self.status.emit(
                        f"Error at NC ({P_nc},{Q_nc}) → ({p},{q}) "
                        f"ext ({list(m_o)},{list(e_o)}): {exc}"
                    )

            transformed_results.append(TransformedFillResult(
                cusp_idx=cusp_idx,
                P_nc=P_nc, Q_nc=Q_nc,
                R=R, S=S,
                p=p, q=q,
                P_user=P_user, Q_user=Q_user,
                fill_results=fill_results,
                weyl_a_phys=weyl_a_phys,
                weyl_b_phys=weyl_b_phys,
            ))

        n_nc_total = len(all_nc)
        n_evals = sum(len(tr.fill_results) for tr in transformed_results)
        self.status.emit(
            f"Done — {n_nc_total} NC cycle(s), {n_evals} filled index evaluation(s)."
        )
        self.finished.emit(transformed_results)


# ═══════════════════════════════════════════════════════════════════════════
# Kernel Builder Worker
# ═══════════════════════════════════════════════════════════════════════════

class KernelBuilderWorker(QThread):
    """Background worker that pre-computes and saves a single filling kernel."""

    status = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(
        self,
        P: int,
        Q: int,
        qq_order: int,
        n_workers: int | None = None,
    ) -> None:
        super().__init__()
        self.P = P
        self.Q = Q
        self.qq_order = qq_order
        self.n_workers = n_workers
        self.cancel_requested = False

    def run(self) -> None:
        try:
            from manifold_index.core.kernel_cache import (
                precompute_filling_kernel,
                save_kernel_table,
            )

            def _progress(msg: str) -> None:
                if self.cancel_requested:
                    raise InterruptedError("Cancelled")
                self.status.emit(msg)

            self.status.emit(
                f"Computing kernel P={self.P}, Q={self.Q}, qq={self.qq_order}…"
            )
            kt = precompute_filling_kernel(
                P=self.P,
                Q=self.Q,
                qq_order=self.qq_order,
                verbose=False,
                progress_callback=_progress,
                n_workers=self.n_workers,
            )

            self.status.emit("Saving to cache…")
            save_kernel_table(kt)

            self.status.emit(
                f"✓ Kernel P={self.P}, Q={self.Q}, qq={self.qq_order} saved."
            )
            self.finished.emit()

        except InterruptedError:
            self.status.emit("Cancelled.")
        except Exception as exc:
            self.error.emit(f"P={self.P}, Q={self.Q}: {exc}")
