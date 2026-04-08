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

from PySide6.QtCore import QThread, Signal

from manifold_index.services.compute_service import ComputeService


class LoadWorker(QThread):
    """Load a manifold by name and probe the local cache."""

    status   = Signal(str)        # human-readable progress message
    progress = Signal(int, int)   # (done, total) — not used by this worker
    finished = Signal(object)     # dict payload on success
    error    = Signal(str)        # error message on failure

    def __init__(self, name: str, parent=None) -> None:
        super().__init__(parent)
        self._name = name

    def run(self) -> None:
        try:
            self.status.emit(f"Loading {self._name}…")
            manifold_data, easy_result, nz_data = ComputeService.load_manifold(
                self._name
            )
            self.status.emit("Probing cache…")
            cache_info = ComputeService.probe_cache(self._name, nz_data)
            self.finished.emit(
                {
                    "manifold_data": manifold_data,
                    "easy_result":   easy_result,
                    "nz_data":       nz_data,
                    "cache_info":    cache_info,
                }
            )
        except Exception as exc:
            self.error.emit(str(exc))

