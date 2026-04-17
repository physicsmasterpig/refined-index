"""app/workers/generate_worker.py — QThread for Data Hub generate operations.

Handles three distinct task types, selected at construction time:

``task="kernels"``
    Calls ``DataHubService.build_kernels(slopes, qq, n_workers,
    skip_existing, progress_fn, status_fn, cancel_fn)``.

``task="iref"``
    Calls ``DataHubService.build_iref_cache(manifold_names, qq, m_max,
    e_max, n_workers, skip_existing, progress_fn, status_fn)``.

``task="nc"``
    Calls ``DataHubService.build_nc_cache(manifold_names, qq, p_max,
    q_max, n_workers, skip_existing, progress_fn, status_fn)``.

Finished payload: the ``list[...]`` returned by the respective service call.

``cancel()`` sets an internal flag that is polled by the ``cancel_fn``
passed to ``build_kernels`` (the other two tasks are serial and cannot be
interrupted mid-manifold).
"""

from __future__ import annotations

import threading
from typing import Any

from PySide6.QtCore import QThread, Signal

from manifold_index.services.datahub_service import DataHubService


class GenerateWorker(QThread):
    """Run a Data Hub generation task in a background thread."""

    status   = Signal(str)
    progress = Signal(int, int)   # (done, total) where meaningful
    finished = Signal(object)     # list of (key, status_str) results
    error    = Signal(str)

    def __init__(
        self,
        task: str,                # "kernels" | "iref" | "nc"
        params: dict[str, Any],
        pause_event: "threading.Event | None" = None,
        parent=None,
    ) -> None:
        """
        Parameters
        ----------
        task : str
            One of ``"kernels"``, ``"iref"``, ``"nc"``.
        params : dict
            For ``"kernels"``:
                slopes (list[(P,Q)]), qq (int), n_workers (int),
                skip_existing (bool)
            For ``"iref"``:
                manifold_names (list[str]), qq (int), m_max (int),
                e_max (int), n_workers (int), skip_existing (bool)
            For ``"nc"``:
                manifold_names (list[str]), qq (int), p_max (int),
                q_max (int), n_workers (int), skip_existing (bool)
        pause_event : threading.Event or None
            When set, the worker pauses between tasks until cleared.
        """
        super().__init__(parent)
        self._task        = task
        self._params      = params
        self._cancel      = False
        self._pause_event = pause_event

    def cancel(self) -> None:
        """Request cancellation (honoured between slopes/manifolds)."""
        self._cancel = True

    def run(self) -> None:
        try:
            p = self._params

            def _status(msg: str) -> None:
                self.status.emit(msg)

            def _progress_int(done: int, total: int) -> None:
                self.progress.emit(done, total)

            def _cancel_fn() -> bool:
                # Honour pause: block until event is cleared
                if self._pause_event is not None:
                    while self._pause_event.is_set():
                        if self._cancel:
                            return True
                        import time
                        time.sleep(0.1)
                return self._cancel

            if self._task == "kernels":
                results = DataHubService.build_kernels(
                    slopes        = p["slopes"],
                    qq            = p["qq"],
                    n_workers     = p.get("n_workers", 1),
                    skip_existing = p.get("skip_existing", True),
                    progress_fn   = _progress_int,
                    status_fn     = _status,
                    cancel_fn     = _cancel_fn,
                )

            elif self._task == "iref":
                results = DataHubService.build_iref_cache(
                    manifold_names = p["manifold_names"],
                    qq             = p["qq"],
                    m_max          = p.get("m_max", 1),
                    e_max          = p.get("e_max", 0),
                    n_workers      = p.get("n_workers", 1),
                    skip_existing  = p.get("skip_existing", True),
                    progress_fn    = _progress_int,
                    status_fn      = _status,
                    cancel_fn      = _cancel_fn,
                )

            elif self._task == "nc":
                results = DataHubService.build_nc_cache(
                    manifold_names = p["manifold_names"],
                    qq             = p["qq"],
                    p_max          = p.get("p_max", 5),
                    q_max          = p.get("q_max", 5),
                    n_workers      = p.get("n_workers", 1),
                    skip_existing  = p.get("skip_existing", True),
                    progress_fn    = _progress_int,
                    status_fn      = _status,
                    cancel_fn      = _cancel_fn,
                )

            else:
                raise ValueError(f"Unknown generate task: {self._task!r}")

            self.finished.emit(results)

        except Exception as exc:
            self.error.emit(str(exc))

