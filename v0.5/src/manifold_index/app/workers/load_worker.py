"""app/workers/load_worker.py — QThread wrapper for manifest load + cache probe.

Calls:
  1. ``ComputeService.load_manifold(name)``
  2. ``ComputeService.probe_cache(name, nz_data)``

Emits a single ``finished`` signal carrying a dict::

    {
        "manifold_data": ManifoldData,
        "easy_result":   EasyEdgeResult,
        "nz_data":       NeumannZagierData,
        "cache_info":    dict,   # keys: "iref", "nc", "kernels"
    }

BLUEPRINT §13 Phase 6: "run() calls exactly one service method, ≤ 20 lines".
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QThread, Signal

from manifold_index.services.compute_service import ComputeService


class LoadWorker(QThread):
    """Load a manifold by name and probe the local cache."""

    status   = Signal(str)        # human-readable progress message
    progress = Signal(int, int)   # (done, total) — not used by this worker
    finished = Signal(object)     # dict payload on success
    error    = Signal(str)        # error message on failure

    def __init__(
        self,
        name: str,
        manifold_data: Any = None,
        easy_result: Any = None,
        nz_data: Any = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._name = name
        self._manifold_data = manifold_data
        self._easy_result = easy_result
        self._nz_data = nz_data

    def run(self) -> None:
        try:
            # NOTE: load_manifold accesses SnapPy's SQLite database, which is
            # thread-local. We load it in the main thread and pass it to the worker.
            # This worker only probes the local cache (which is thread-safe).
            self.status.emit("Probing cache…")
            cache_info = ComputeService.probe_cache(self._name, self._nz_data)
            self.finished.emit(
                {
                    "manifold_data": self._manifold_data,
                    "easy_result":   self._easy_result,
                    "nz_data":       self._nz_data,
                    "cache_info":    cache_info,
                }
            )
        except Exception as exc:
            self.error.emit(str(exc))

