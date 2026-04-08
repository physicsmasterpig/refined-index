"""app/workers/weyl_worker.py — QThread for Weyl-symmetry check.

Calls ``ComputeService.run_weyl_check(entries, num_hard, q_order_half)``.

Finished payload::

    {
        "ab_vectors": ABVectors | None,
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
            ab_vectors = ComputeService.run_weyl_check(
                self._entries,
                self._num_hard,
                self._q_order_half,
                self._cusp_idx,
            )
            self.finished.emit({"ab_vectors": ab_vectors})
        except Exception as exc:
            self.error.emit(str(exc))

