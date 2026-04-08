"""app/workers/index_worker.py — QThread for refined index computation.

Calls ``ComputeService.compute_refined_index(nz_data, m_ext, e_ext, q_order_half)``.

Finished payload::

    {
        "m_ext":        list[int],
        "e_ext":        list[Fraction],
        "result":       RefinedIndexResult,   # dict[tuple[int,...], int]
        "from_cache":   bool,
    }
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from PySide6.QtCore import QThread, Signal

from manifold_index.services.compute_service import ComputeService


class IndexWorker(QThread):
    """Compute I^ref(m_ext, e_ext) at the given truncation order."""

    status   = Signal(str)
    progress = Signal(int, int)
    finished = Signal(object)
    error    = Signal(str)

    def __init__(
        self,
        nz_data: Any,
        m_ext: list[int],
        e_ext: list[Fraction],
        q_order_half: int,
        manifold_name: str = "",
        use_cache: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._nz_data      = nz_data
        self._m_ext        = m_ext
        self._e_ext        = e_ext
        self._q_order_half = q_order_half
        self._name         = manifold_name
        self._use_cache    = use_cache

    def run(self) -> None:
        try:
            self.status.emit(
                f"Computing I^ref({self._m_ext}, {self._e_ext})…"
            )
            from_cache = False

            if self._use_cache and self._name:
                result = ComputeService.load_refined_index_from_cache(
                    self._name, self._nz_data,
                    self._m_ext, self._e_ext, self._q_order_half,
                )
                if result is not None:
                    from_cache = True
                else:
                    result = ComputeService.compute_refined_index(
                        self._nz_data, self._m_ext,
                        self._e_ext, self._q_order_half,
                    )
            else:
                result = ComputeService.compute_refined_index(
                    self._nz_data, self._m_ext,
                    self._e_ext, self._q_order_half,
                )

            self.finished.emit(
                {
                    "m_ext":      self._m_ext,
                    "e_ext":      self._e_ext,
                    "result":     result,
                    "from_cache": from_cache,
                }
            )
        except Exception as exc:
            self.error.emit(str(exc))

