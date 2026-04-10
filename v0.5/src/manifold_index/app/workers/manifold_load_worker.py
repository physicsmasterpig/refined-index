"""app/workers/manifold_load_worker.py — QThread for manifold loading.

Loads manifold, finds phase space basis, builds NZ matrix in a separate thread.
SnaPy's SQLite is thread-local, but this worker creates its own thread-local
connection by importing snappy in the worker thread (not shared with main thread).

Returns payload with (manifold_data, easy_result, nz_data).
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QThread, Signal


class ManifoldLoadWorker(QThread):
    """Load manifold in a separate thread with thread-local SnaPy connection."""

    status = Signal(str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        name: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._name = name

    def run(self) -> None:
        """Load manifold using thread-local SnaPy connection."""
        self.setPriority(QThread.Priority.LowPriority)
        try:
            self.status.emit(f"Loading {self._name}…")

            # Import ComputeService in worker thread so SnaPy's SQLite
            # connection is thread-local and not shared with main thread.
            from manifold_index.services.compute_service import ComputeService

            manifold_data, easy_result, nz_data = ComputeService.load_manifold(
                self._name
            )

            self.finished.emit(
                {
                    "manifold_data": manifold_data,
                    "easy_result": easy_result,
                    "nz_data": nz_data,
                }
            )
        except Exception as exc:
            self.error.emit(str(exc))
