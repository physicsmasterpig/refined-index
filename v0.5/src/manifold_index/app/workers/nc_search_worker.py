"""app/workers/nc_search_worker.py — QThread for NC-cycle search.

Calls ``FillingService.find_nc_cycles(nz_data, cusp_idx, p_range, q_range,
q_order_half, progress_fn)``.

Finished payload::

    {
        "cusp_idx":  int,
        "nc_result": NonClosableCycleResult,
        "cycles":    list[NonClosableCycle],   # canonicalised
    }

Progress is forwarded from the inner ``progress_fn`` callback so the card
can show a running count of tested slopes.
"""

from __future__ import annotations

import time
from typing import Any

from PySide6.QtCore import QThread, Signal

from manifold_index.services.filling_service import FillingService


class NCSearchWorker(QThread):
    """Search for non-closable cycles over a slope range."""

    status   = Signal(str)
    progress = Signal(int, int)   # (tested, total)
    finished = Signal(object)
    error    = Signal(str)

    def __init__(
        self,
        nz_data: Any,
        cusp_idx: int,
        p_range: tuple[int, int],
        q_range: tuple[int, int],
        q_order_half: int,
        manifold_name: str = "",
        use_cache: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._nz_data      = nz_data
        self._cusp_idx     = cusp_idx
        self._p_range      = p_range
        self._q_range      = q_range
        self._q_order_half = q_order_half
        self._name         = manifold_name
        self._use_cache    = use_cache
        self._tested       = 0
        self._total        = 0

    def run(self) -> None:
        # Lower this thread's priority so macOS gives the main (UI) thread
        # more CPU time and the event loop stays responsive.
        self.setPriority(QThread.Priority.LowPriority)
        try:
            # Try cache first
            if self._use_cache and self._name:
                self.status.emit("Loading NC cycles from cache…")
                cached = FillingService.load_nc_from_cache(
                    self._name, self._nz_data,
                    self._cusp_idx, self._q_order_half,
                )
                if cached is not None:
                    canonical = FillingService.canonicalise_nc_cycles(cached)
                    self.finished.emit(
                        {
                            "cusp_idx":  self._cusp_idx,
                            "nc_result": None,
                            "cycles":    canonical,
                        }
                    )
                    return

            p_lo, p_hi = self._p_range
            q_lo, q_hi = self._q_range
            # Estimate total (exclude (0,0))
            self._total = (p_hi - p_lo + 1) * (q_hi - q_lo + 1) - 1
            self._tested = 0

            self.status.emit(
                f"Searching NC cycles (cusp {self._cusp_idx}, "
                f"P∈[{p_lo},{p_hi}], Q∈[{q_lo},{q_hi}])…"
            )

            def _prog(done: int, total: int) -> None:
                # time.sleep() is guaranteed to release the Python GIL so the
                # main thread can process Qt/Cocoa events between slopes.
                # msleep() is a C++ call that may NOT release the GIL.
                time.sleep(0.005)
                self.progress.emit(done, total)

            nc_result = FillingService.find_nc_cycles(
                self._nz_data,
                self._cusp_idx,
                self._p_range,
                self._q_range,
                self._q_order_half,
                progress_fn=_prog,
            )
            cycles = FillingService.canonicalise_nc_cycles(
                list(nc_result.cycles) if hasattr(nc_result, "cycles") else []
            )
            self.finished.emit(
                {
                    "cusp_idx":  self._cusp_idx,
                    "nc_result": nc_result,
                    "cycles":    cycles,
                }
            )
        except Exception as exc:
            self.error.emit(str(exc))

