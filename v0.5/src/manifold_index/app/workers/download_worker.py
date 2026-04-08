"""app/workers/download_worker.py — QThread for data-pack download.

Calls ``DataHubService.download_pack(registry, pack, progress_fn, status_fn)``.

Progress is forwarded from the inner ``progress_fn`` as ``(received, total)``
bytes.  The card can use this to drive a ``QProgressBar``.

Finished payload::

    {
        "pack_name":   str,
        "files_count": int,
    }
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QThread, Signal

from manifold_index.services.datahub_service import DataHubService


class DownloadWorker(QThread):
    """Download and install a remote data pack."""

    status   = Signal(str)
    progress = Signal(int, int)   # (bytes_received, bytes_total)
    finished = Signal(object)
    error    = Signal(str)

    def __init__(
        self,
        registry: Any,
        pack: Any,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._registry = registry
        self._pack     = pack

    def run(self) -> None:
        try:
            pack_name = getattr(self._pack, "name", str(self._pack))
            self.status.emit(f"Downloading {pack_name}…")

            def _progress(received: int, total: int) -> None:
                self.progress.emit(received, total)

            def _status(msg: str) -> None:
                self.status.emit(msg)

            n_files = DataHubService.download_pack(
                registry    = self._registry,
                pack        = self._pack,
                progress_fn = _progress,
                status_fn   = _status,
            )
            self.finished.emit(
                {
                    "pack_name":   pack_name,
                    "files_count": n_files,
                }
            )
        except Exception as exc:
            self.error.emit(str(exc))

