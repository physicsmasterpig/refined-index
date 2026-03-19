"""
app/workers.py — Background QThread workers for computation.

Workers
-------
PipelineWorker                — Steps 4-5: non-closable cycle search per cusp.
RefinedIndexWorker            — Step 8: refined index at every (m_ext, e_ext) point.
DehnFillingWorker             — Refined Dehn filling via compute_filled_refined_index.
DehnFillingPipelineWorker     — Full Dehn-filling pipeline: search non-closable
                                cycles, basis change, slope transform, compute
                                filled refined index.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Sequence

from PySide6.QtCore import QThread, Signal


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class NonClosableCycle:
    """A single non-closable cycle found at one cusp."""
    cusp_idx: int
    P: int
    Q: int

    def __str__(self) -> str:
        return f"({self.P}, {self.Q})"


@dataclass
class CuspResult:
    """Non-closable cycles found at one cusp."""
    cusp_idx: int
    cycles: list[NonClosableCycle] = field(default_factory=list)
    filled_indices: dict = field(default_factory=dict)
    # filled_indices: {(P, Q): FilledIndexResult} — all slopes tested


@dataclass
class PipelineResult:
    """Value object collecting everything produced by the pipeline worker."""
    name: str
    nz_data: object           # NeumannZagierData
    q_order_half: int
    cycle_results: list[CuspResult] = field(default_factory=list)
    manifold_data: object = field(default=None)   # ManifoldData (set by gui.py)
    easy_result: object = field(default=None)      # EasyEdgeResult (set by gui.py)


# ---------------------------------------------------------------------------
# PipelineWorker — non-closable cycle search
# ---------------------------------------------------------------------------

class PipelineWorker(QThread):
    """Run Steps 4-5 (Dehn filling per cusp) in the background.

    Emits progress signals so the UI can update a progress bar.
    """

    status = Signal(str)
    slope_progress = Signal(int, int, int)   # (cusp_idx, done, total)
    finished = Signal(object)                # PipelineResult
    error = Signal(str)

    def __init__(
        self,
        name: str,
        nz_data,
        q_order_half: int,
        p_range: range,
        q_range: range,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._name = name
        self._nz_data = nz_data
        self._q_order_half = q_order_half
        self._p_range = p_range
        self._q_range = q_range

    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:
            import traceback
            self.error.emit(
                f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
            )

    def _run(self) -> None:
        result = PipelineResult(
            name=self._name,
            nz_data=self._nz_data,
            q_order_half=self._q_order_half,
        )
        nz = self._nz_data
        for cusp_idx in range(nz.r):
            self.status.emit(f"Cusp {cusp_idx}: searching non-closable cycles …")
            cusp_res = _find_non_closable_with_progress(
                nz, cusp_idx,
                self._q_order_half,
                self._p_range, self._q_range,
                on_status=self.status.emit,
                on_slope=lambda d, t, ci=cusp_idx: self.slope_progress.emit(ci, d, t),
            )
            result.cycle_results.append(cusp_res)
        self.finished.emit(result)


def _find_non_closable_with_progress(
    nz_data,
    cusp_idx: int,
    q_order_half: int,
    p_range: range,
    q_range: range,
    on_status=None,
    on_slope=None,
) -> CuspResult:
    """Search for non-closable cycles at one cusp, reporting progress."""
    from math import gcd
    from manifold_index.core.dehn_filling import compute_filled_index

    result = CuspResult(cusp_idx=cusp_idx)
    slopes: list[tuple[int, int]] = []
    for P in p_range:
        for Q in q_range:
            if P == 0 and Q == 0:
                continue
            if gcd(abs(P), abs(Q)) != 1:
                continue
            slopes.append((P, Q))

    all_slopes = set(slopes)
    computed: dict[tuple[int, int], bool] = {}
    total = len(slopes)

    r = nz_data.r
    m_other = [0] * (r - 1)
    e_other = [Fraction(0)] * (r - 1)

    for done_idx, (P, Q) in enumerate(slopes):
        if (P, Q) in computed:
            if on_slope:
                on_slope(done_idx + 1, total)
            continue

        filled = compute_filled_index(
            nz_data,
            cusp_idx=cusp_idx,
            P=P,
            Q=Q,
            m_other=list(m_other),
            e_other=list(e_other),
            q_order_half=q_order_half,
        )
        non_closable = filled.is_stably_zero()
        computed[(P, Q)] = non_closable
        result.filled_indices[(P, Q)] = filled

        if non_closable:
            result.cycles.append(
                NonClosableCycle(cusp_idx=cusp_idx, P=P, Q=Q)
            )

        neg = (-P, -Q)
        if neg in all_slopes and neg not in computed:
            computed[neg] = non_closable
            # Build the negated result by mirroring (same series)
            result.filled_indices[neg] = filled
            if non_closable:
                result.cycles.append(
                    NonClosableCycle(cusp_idx=cusp_idx, P=-P, Q=-Q)
                )

        if on_slope:
            on_slope(done_idx + 1, total)

    return result


# ---------------------------------------------------------------------------
# RefinedIndexWorker
# ---------------------------------------------------------------------------

class RefinedIndexWorker(QThread):
    """Run Step 8 (compute_refined_index) in the background.

    Evaluates the refined index at every (m_ext, e_ext) point and emits
    a list of (m_ext, e_ext, result) triples.
    """

    status = Signal(str)
    progress = Signal(int, int)    # (done, total)
    finished = Signal(object)      # list of (m_ext, e_ext, RefinedIndexResult)
    error = Signal(str)

    def __init__(
        self,
        nz_data,
        eval_points: list,
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
            self.error.emit(
                f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
            )

    def _run(self) -> None:
        from manifold_index.core.refined_index import compute_refined_index

        total = len(self._eval_points)
        results = []
        for idx, (m_ext, e_ext) in enumerate(self._eval_points):
            self.status.emit(
                f"Computing I({m_ext}, {e_ext}) … ({idx + 1}/{total})"
            )
            self.progress.emit(idx + 1, total)
            result = compute_refined_index(
                self._nz_data, m_ext, e_ext, self._q_order_half,
            )
            results.append((m_ext, e_ext, result))

        self.status.emit("Done.")
        self.finished.emit(results)


# ---------------------------------------------------------------------------
# DehnFillingWorker
# ---------------------------------------------------------------------------

class DehnFillingWorker(QThread):
    """Compute the refined Dehn-filled index in the background.

    Calls ``compute_filled_refined_index`` and emits the result.
    """

    status = Signal(str)
    finished = Signal(object)    # FilledRefinedResult
    error = Signal(str)

    def __init__(
        self,
        nz_data,
        cusp_idx: int,
        P: int,
        Q: int,
        q_order_half: int,
        eta_order: int = 5,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._nz_data = nz_data
        self._cusp_idx = cusp_idx
        self._P = P
        self._Q = Q
        self._q_order_half = q_order_half
        self._eta_order = eta_order

    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:
            import traceback
            self.error.emit(
                f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
            )

    def _run(self) -> None:
        from manifold_index.core.refined_dehn_filling import (
            compute_filled_refined_index,
        )

        self.status.emit(
            f"Computing refined Dehn filling at slope ({self._P}, {self._Q}) …"
        )
        result = compute_filled_refined_index(
            self._nz_data,
            cusp_idx=self._cusp_idx,
            P=self._P,
            Q=self._Q,
            q_order_half=self._q_order_half,
            eta_order=self._eta_order,
            verbose=False,
        )
        self.status.emit("Dehn filling complete.")
        self.finished.emit(result)


# ---------------------------------------------------------------------------
# DehnFillingPipelineWorker — full pipeline: search + transform + fill
# ---------------------------------------------------------------------------

def _ext_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Extended Euclidean algorithm: returns (g, x, y) with a*x + b*y = g."""
    if b == 0:
        return a, 1, 0
    g, x1, y1 = _ext_gcd(b, a % b)
    return g, y1, x1 - (a // b) * y1


def _transform_slope(
    P_user: int, Q_user: int,
    P_nc: int, Q_nc: int,
) -> tuple[int, int, int, int]:
    """Transform the user's Dehn filling slope into the new basis.

    The non-closable cycle (P_nc, Q_nc) becomes the new meridian:
        new_M = P_nc·M + Q_nc·L

    The new longitude is determined by the Bézout identity:
        P_nc·b − 2·Q_nc·a = 1
    so:
        new_L = 2a·M + b·L

    The user's slope (P_user, Q_user) in the original (M, L) basis
    is re-expressed as (P_new, Q_new) in the new (M', L') basis:
        P_new = b · P_user − 2a · Q_user
        Q_new = −Q_nc · P_user + P_nc · Q_user

    Returns (P_new, Q_new, a, b).
    """
    g, b, a = _ext_gcd(P_nc, -2 * Q_nc)
    # _ext_gcd(P_nc, -2*Q_nc) → P_nc*b + (-2*Q_nc)*a = g
    # i.e. P_nc*b − 2*Q_nc*a = g (should be 1)
    assert g == 1, (
        f"gcd({P_nc}, {-2*Q_nc}) = {g} ≠ 1; "
        f"P_nc must be odd for integer Bézout solution"
    )
    P_new = b * P_user - 2 * a * Q_user
    Q_new = -Q_nc * P_user + P_nc * Q_user
    return P_new, Q_new, a, b


class DehnFillingPipelineWorker(QThread):
    """Full Dehn filling pipeline: search non-closable cycles → basis
    change → slope transform → compute filled refined index.

    Emits a dict with all results when done.
    """

    status = Signal(str)
    progress = Signal(int, int)   # (done, total)
    finished = Signal(object)     # dict result_info
    error = Signal(str)

    def __init__(
        self,
        nz_data,
        cusp_idx: int,
        P_user: int,
        Q_user: int,
        q_order_half: int,
        p_range: range,
        q_range: range,
        eta_order: int = 5,
        weyl_a: list | None = None,
        weyl_b: list | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._nz_data = nz_data
        self._cusp_idx = cusp_idx
        self._P_user = P_user
        self._Q_user = Q_user
        self._q_order_half = q_order_half
        self._p_range = p_range
        self._q_range = q_range
        self._eta_order = eta_order
        self._weyl_a = weyl_a
        self._weyl_b = weyl_b

    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:
            import traceback
            self.error.emit(
                f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
            )

    def _run(self) -> None:
        from math import gcd as _gcd
        from manifold_index.core.dehn_filling import compute_filled_index
        from manifold_index.core.neumann_zagier import apply_cusp_basis_change
        from manifold_index.core.refined_dehn_filling import (
            compute_filled_refined_index,
        )

        nz = self._nz_data
        cusp_idx = self._cusp_idx
        q_order_half = self._q_order_half

        # ── Step 1: search non-closable cycles ────────────────────
        self.status.emit(
            f"Cusp {cusp_idx}: searching non-closable cycles …"
        )

        slopes: list[tuple[int, int]] = []
        for P in self._p_range:
            for Q in self._q_range:
                if P == 0 and Q == 0:
                    continue
                if _gcd(abs(P), abs(Q)) != 1:
                    continue
                # Need P odd for basis change to work
                if P % 2 == 0:
                    continue
                slopes.append((P, Q))

        # Deduplicate by removing (P,Q) if (-P,-Q) already present
        seen: set[tuple[int, int]] = set()
        unique_slopes = []
        for P, Q in slopes:
            if (-P, -Q) not in seen:
                unique_slopes.append((P, Q))
                seen.add((P, Q))
        slopes = unique_slopes

        total = len(slopes)
        non_closable: list[tuple[int, int]] = []

        r = nz.r
        m_other = [0] * (r - 1)
        e_other = [Fraction(0)] * (r - 1)

        for done_idx, (P_nc, Q_nc) in enumerate(slopes):
            self.progress.emit(done_idx + 1, total)
            self.status.emit(
                f"Cusp {cusp_idx}: testing slope ({P_nc}, {Q_nc}) "
                f"({done_idx + 1}/{total}) …"
            )

            filled = compute_filled_index(
                nz, cusp_idx=cusp_idx, P=P_nc, Q=Q_nc,
                m_other=list(m_other), e_other=list(e_other),
                q_order_half=q_order_half,
            )
            if filled.is_stably_zero():
                non_closable.append((P_nc, Q_nc))
                # Also add the negation
                non_closable.append((-P_nc, -Q_nc))

        # ── Step 2: for each non-closable cycle, transform + fill ──
        results: list[dict] = []

        if non_closable:
            # Use just the first non-closable cycle for now (they should
            # give the same result).  But record all for display.
            # Actually, compute for ALL to verify agreement.
            total_nc = len(non_closable)
            for nc_idx, (P_nc, Q_nc) in enumerate(non_closable):
                self.status.emit(
                    f"Non-closable ({P_nc}, {Q_nc}): "
                    f"computing filled refined index ({nc_idx + 1}/{total_nc}) …"
                )
                self.progress.emit(nc_idx + 1, total_nc)

                try:
                    P_new, Q_new, a_coeff, b_coeff = _transform_slope(
                        self._P_user, self._Q_user, P_nc, Q_nc,
                    )
                except AssertionError as e:
                    # Skip this cycle (P_nc even — shouldn't happen)
                    continue

                # Apply basis change
                nz_changed = apply_cusp_basis_change(nz, cusp_idx, P_nc, Q_nc)

                # Compute filled refined index with transformed slope
                filled_result = compute_filled_refined_index(
                    nz_changed,
                    cusp_idx=cusp_idx,
                    P=P_new,
                    Q=Q_new,
                    q_order_half=q_order_half,
                    eta_order=self._eta_order,
                    weyl_a=self._weyl_a,
                    weyl_b=self._weyl_b,
                    verbose=False,
                )

                results.append({
                    "P_nc": P_nc,
                    "Q_nc": Q_nc,
                    "a": a_coeff,
                    "b": b_coeff,
                    "P_new": P_new,
                    "Q_new": Q_new,
                    "filled_result": filled_result,
                })

        result_info = {
            "cusp_idx": cusp_idx,
            "P_user": self._P_user,
            "Q_user": self._Q_user,
            "non_closable_cycles": non_closable,
            "results": results,
        }
        self.status.emit("Dehn filling pipeline complete.")
        self.finished.emit(result_info)

