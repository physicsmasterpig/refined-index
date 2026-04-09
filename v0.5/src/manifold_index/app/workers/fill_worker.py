"""app/workers/fill_worker.py — QThread for filled refined index computation.

Calls ``FillingService.compute_filled_index(…)``.

Finished payload::

    {
        "cusp_idx":  int,
        "nc_P":      int,
        "nc_Q":      int,
        "user_P":    int,
        "user_Q":    int,
        "p":         int,     # slope in NC basis
        "q":         int,
        "result":    FilledRefinedResult,
    }
"""

from __future__ import annotations

import time
from fractions import Fraction
from typing import Any

from PySide6.QtCore import QThread, Signal

from manifold_index.services.filling_service import FillingService


class FillWorker(QThread):
    """Apply basis change and compute the filled refined index."""

    status   = Signal(str)
    progress = Signal(int, int)
    finished = Signal(object)
    error    = Signal(str)

    def __init__(
        self,
        nz_data: Any,
        cusp_idx: int,
        nc_P: int,
        nc_Q: int,
        user_P: int,
        user_Q: int,
        m_other: list[int] | None,
        e_other: list[Fraction] | None,
        q_order_half: int,
        weyl_a: list[Fraction] | None = None,
        weyl_b: list[Fraction] | None = None,
        auto_precompute: bool = True,
        manifold_name: str = "unknown",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._nz_data         = nz_data
        self._cusp_idx        = cusp_idx
        self._nc_P            = nc_P
        self._nc_Q            = nc_Q
        self._user_P          = user_P
        self._user_Q          = user_Q
        self._m_other         = m_other
        self._e_other         = e_other
        self._q_order_half    = q_order_half
        self._weyl_a          = weyl_a
        self._weyl_b          = weyl_b
        self._auto_precompute = auto_precompute
        self._manifold_name   = manifold_name

    def run(self) -> None:
        # Lower this thread's priority so the main (UI) thread stays responsive.
        self.setPriority(QThread.Priority.LowPriority)
        try:
            self.status.emit(
                f"Computing filled index (NC=({self._nc_P},{self._nc_Q}), "
                f"slope=({self._user_P},{self._user_Q}))…"
            )

            def _prog(done: int, total: int) -> None:
                # time.sleep() releases the Python GIL so the main thread can
                # process Qt/Cocoa events between kernel-term iterations.
                time.sleep(0.005)
                self.progress.emit(done, total)

            p, q, result = FillingService.compute_filled_index(
                nz_data        = self._nz_data,
                cusp_idx       = self._cusp_idx,
                nc_P           = self._nc_P,
                nc_Q           = self._nc_Q,
                user_P         = self._user_P,
                user_Q         = self._user_Q,
                m_other        = self._m_other,
                e_other        = self._e_other,
                q_order_half   = self._q_order_half,
                weyl_a         = self._weyl_a,
                weyl_b         = self._weyl_b,
                auto_precompute= self._auto_precompute,
                progress_fn    = _prog,
                manifold_name  = self._manifold_name,
            )
            self.finished.emit(
                {
                    "cusp_idx": self._cusp_idx,
                    "nc_P":     self._nc_P,
                    "nc_Q":     self._nc_Q,
                    "user_P":   self._user_P,
                    "user_Q":   self._user_Q,
                    "p":        p,
                    "q":        q,
                    "result":   result,
                }
            )
        except Exception as exc:
            self.error.emit(str(exc))

