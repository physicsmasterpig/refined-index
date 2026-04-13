"""app/workers/weyl_worker.py — QThread workers for Weyl-symmetry checks.

``WeylWorker``
    Runs ``ComputeService.run_weyl_check`` on pre-computed entries.

``NcCompatWorker``
    Combined worker for per-NC-cycle compatibility checks: applies the
    basis change, recomputes index entries, then runs the Weyl check —
    all off the main thread so the UI stays responsive.

Finished payload (both workers)::

    {
        "ab_vectors": ABVectors | None,
        "adjoint_is_pass": bool | None,
        "adjoint_value": float | None,
    }
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from PySide6.QtCore import QThread, Signal

from manifold_index.services.compute_service import ComputeService


class WeylWorker(QThread):
    """Run the Weyl-symmetry check on a collection of index entries."""

    status   = Signal(str)
    progress = Signal(int, int)
    finished = Signal(object)
    error    = Signal(str)

    def __init__(
        self,
        entries: list[tuple[list[int], list[Fraction], Any]],
        num_hard: int,
        q_order_half: int,
        cusp_idx: int = 0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._entries      = entries
        self._num_hard     = num_hard
        self._q_order_half = q_order_half
        self._cusp_idx     = cusp_idx

    def run(self) -> None:
        try:
            self.status.emit("Running Weyl symmetry check…")
            ab_vectors, adjoint_is_pass, adjoint_value = ComputeService.run_weyl_check(
                self._entries,
                self._num_hard,
                self._q_order_half,
                self._cusp_idx,
            )
            self.finished.emit({
                "ab_vectors": ab_vectors,
                "adjoint_is_pass": adjoint_is_pass,
                "adjoint_value": adjoint_value,
            })
        except Exception as exc:
            self.error.emit(str(exc))


class NcCompatWorker(QThread):
    """Basis-change + index recomputation + Weyl check for one NC cycle.

    Runs entirely off the main thread so the UI stays responsive during
    per-cycle compatibility checks after NC search.
    """

    finished = Signal(object)
    error    = Signal(str)

    def __init__(
        self,
        nz_data: Any,
        P: int,
        Q: int,
        cusp_idx: int,
        index_queries: list,
        num_hard: int,
        q_order_half: int,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._nz_data      = nz_data
        self._P            = P
        self._Q            = Q
        self._cusp_idx     = cusp_idx
        self._index_queries = index_queries
        self._num_hard     = num_hard
        self._q_order_half = q_order_half

    def run(self) -> None:
        try:
            from manifold_index.core import (          # noqa: PLC0415
                dehn_filling as _df,
                neumann_zagier as _nz,
            )
            R, S = _df.find_rs(self._P, self._Q)
            nz_nc = _nz.apply_general_cusp_basis_change(
                self._nz_data, self._cusp_idx,
                a=self._P, b=self._Q, c=-R, d=-S,
            )
            from fractions import Fraction as _Frac  # noqa: PLC0415

            # ── User-grid entries (basis-changed) ────────────────────────
            seen: set[tuple] = set()
            entries_nc: list = []
            for q in self._index_queries:
                if q.result is not None:
                    result_nc = ComputeService.compute_refined_index(
                        nz_nc, q.m_ext, q.e_ext, self._q_order_half
                    )
                    if result_nc is not None:
                        key = (tuple(q.m_ext), tuple(q.e_ext))
                        seen.add(key)
                        entries_nc.append((q.m_ext, q.e_ext, result_nc))

            # ── Required adjoint-check points: m=0, e ∈ {−2,−1,+1,+2} ──
            # check_adjoint_projection needs all four e values; compute any
            # that aren't already covered by the user's grid.
            n_cusps = len(self._nz_data.e_ext_size) if hasattr(self._nz_data, 'e_ext_size') else 1
            try:
                n_cusps = self._nz_data.r
            except AttributeError:
                n_cusps = 1
            for e_val in (_Frac(-2), _Frac(-1), _Frac(1), _Frac(2)):
                m_ext = [0] * n_cusps
                e_ext = [_Frac(0)] * n_cusps
                e_ext[0] = e_val
                key = (tuple(m_ext), tuple(e_ext))
                if key not in seen:
                    result_nc = ComputeService.compute_refined_index(
                        nz_nc, m_ext, e_ext, self._q_order_half
                    )
                    if result_nc is not None:
                        seen.add(key)
                        entries_nc.append((m_ext, e_ext, result_nc))

            if not entries_nc:
                self.finished.emit({
                    "ab_vectors": None,
                    "adjoint_is_pass": None,
                    "adjoint_value": None,
                })
                return

            ab_vectors, adjoint_is_pass, adjoint_value = ComputeService.run_weyl_check(
                entries_nc, self._num_hard, self._q_order_half, 0,
            )
            self.finished.emit({
                "ab_vectors": ab_vectors,
                "adjoint_is_pass": adjoint_is_pass,
                "adjoint_value": adjoint_value,
            })
        except Exception as exc:
            self.error.emit(str(exc))

