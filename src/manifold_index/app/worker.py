"""
app/worker.py — Background computation workers for the GUI.

Two workers:

PipelineWorker
    Runs Steps 1–5 in a QThread:
      1. load_manifold
      2. find_easy_edges
      3. build_neumann_zagier
      4/5. find_non_closable_cycles for each cusp (slow)

    Emits:
      status(str)         — human-readable stage description
      slope_progress(int, int, int)
                          — (cusp_idx, done, total) during Dehn filling
      finished(object)    — PipelineResult on success
      error(str)          — error message on failure

RefinedIndexWorker
    Runs Step 8 (compute_refined_index) in a QThread.

    Emits:
      status(str)
      finished(object)    — RefinedIndexResult (dict)
      error(str)
"""

from __future__ import annotations

from fractions import Fraction

from PySide6.QtCore import QThread, Signal


# ---------------------------------------------------------------------------
# PipelineResult — value object passed from PipelineWorker to the GUI
# ---------------------------------------------------------------------------

class PipelineResult:
    """All data produced by the automatic pipeline (Steps 1–5)."""

    def __init__(
        self,
        name: str,
        nz_data,
        q_order_half: int,
        cycle_results: list,    # list[NonClosableCycleResult]
    ) -> None:
        self.name = name
        self.nz_data = nz_data
        self.q_order_half = q_order_half
        self.cycle_results = cycle_results  # one per cusp


# ---------------------------------------------------------------------------
# PipelineWorker
# ---------------------------------------------------------------------------

class PipelineWorker(QThread):
    """Run Steps 4–5 (Dehn filling per cusp) in the background.

    Steps 1–3 (load_manifold, find_easy_edges, build_neumann_zagier) are
    intentionally performed in the main thread before this worker is created,
    because SnaPy uses a SQLite connection that is bound to the thread in which
    the module is first imported.  Passing the resulting NeumannZagierData
    (a pure Python dataclass) to the worker avoids the thread-safety issue.
    """

    status = Signal(str)
    slope_progress = Signal(int, int, int)   # cusp_idx, done, total
    finished = Signal(object)                # PipelineResult
    error = Signal(str)

    def __init__(
        self,
        name: str,
        nz_data,                 # NeumannZagierData — pre-computed in main thread
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
            self.error.emit(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}")

    def _run(self) -> None:
        name = self._name
        nz = self._nz_data
        q_order_half = self._q_order_half

        r = nz.r

        # --- Steps 4/5: Dehn filling per cusp ---
        cycle_results = []
        for cusp_idx in range(r):
            all_slopes = _count_slopes(self._p_range, self._q_range)
            self.status.emit(
                f"Dehn filling — cusp {cusp_idx}/{r-1}: "
                f"searching {all_slopes} slope(s) …"
            )

            def _on_slope(done: int, total: int, ci: int = cusp_idx) -> None:
                self.slope_progress.emit(ci, done, total)

            res = _find_non_closable_with_progress(
                nz,
                cusp_idx=cusp_idx,
                p_range=self._p_range,
                q_range=self._q_range,
                q_order_half=q_order_half,
                on_slope=_on_slope,
            )
            cycle_results.append(res)

            nc = len(res.cycles)
            self.status.emit(
                f"Cusp {cusp_idx}: {nc} non-closable cycle(s) found."
            )

        result = PipelineResult(
            name=name,
            nz_data=nz,
            q_order_half=q_order_half,
            cycle_results=cycle_results,
        )
        self.finished.emit(result)


def _count_slopes(p_range: range, q_range: range) -> int:
    """Quick estimate of how many primitive slopes fit in the range."""
    from math import gcd
    count = 0
    for P in p_range:
        for Q in q_range:
            if P == 0 and Q == 0:
                continue
            if gcd(abs(P), abs(Q)) == 1:
                count += 1
    return count


def _find_non_closable_with_progress(
    nz_data,
    cusp_idx: int,
    p_range: range,
    q_range: range,
    q_order_half: int,
    on_slope,
):
    """
    Wrapper around find_non_closable_cycles that calls on_slope(done, total)
    after each slope is evaluated.

    Because find_non_closable_cycles doesn't expose a per-slope callback, we
    re-implement the same logic with an inline progress hook.
    """
    from math import gcd
    from manifold_index.core.dehn_filling import (
        NonClosableCycleResult,
        NonClosableCycle,
        _candidate_slopes,
        compute_filled_index,
    )

    r = nz_data.r
    m_other = [0] * (r - 1)
    e_other = [0] * (r - 1)

    all_slopes = _candidate_slopes(p_range, q_range, canonical_only=False)
    compute_slopes = _candidate_slopes(p_range, q_range, canonical_only=True)
    total = len(compute_slopes)

    result = NonClosableCycleResult(cusp_idx=cusp_idx)
    result.slopes_tested = all_slopes[:]

    computed: dict[tuple[int, int], bool] = {}

    for done_idx, (P, Q) in enumerate(compute_slopes):
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

        if non_closable:
            result.cycles.append(NonClosableCycle(cusp_idx=cusp_idx, P=P, Q=Q))

        neg = (-P, -Q)
        if neg in set(all_slopes) and neg not in computed:
            computed[neg] = non_closable
            if non_closable:
                result.cycles.append(NonClosableCycle(cusp_idx=cusp_idx, P=-P, Q=-Q))

        on_slope(done_idx + 1, total)

    return result


# ---------------------------------------------------------------------------
# RefinedIndexWorker
# ---------------------------------------------------------------------------

class RefinedIndexWorker(QThread):
    """Run Step 8 (compute_refined_index) in the background.

    Evaluates the refined index at every ``(m_ext, e_ext)`` point in
    *eval_points* and emits a list of ``(m_ext, e_ext, result)`` triples.
    """

    status = Signal(str)
    finished = Signal(object)    # list of (m_ext, e_ext, RefinedIndexResult)
    error = Signal(str)

    def __init__(
        self,
        nz_data,
        eval_points: list,   # list of (m_ext: list[int], e_ext: list)
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
            self.status.emit(
                f"Computing I({m_ext}, {e_ext}) … ({idx + 1}/{total})"
            )
            result = compute_refined_index(
                self._nz_data,
                m_ext,
                e_ext,
                self._q_order_half,
            )
            results.append((m_ext, e_ext, result))

        self.status.emit("Done.")
        self.finished.emit(results)

