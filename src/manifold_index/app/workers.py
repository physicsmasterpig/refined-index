"""
app/workers.py — Background QThread workers for computation.

Workers
-------
PipelineWorker       — Steps 4-5: non-closable cycle search per cusp.
RefinedIndexWorker   — Step 8: refined index at every (m_ext, e_ext) point.
DehnFillingWorker    — Refined Dehn filling via compute_filled_refined_index.
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
