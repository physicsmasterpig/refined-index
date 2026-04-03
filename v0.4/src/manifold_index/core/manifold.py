"""Manifold loading via SnaPy."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ManifoldData:
    name: str
    num_tetrahedra: int
    num_cusps: int
    gluing_matrix: np.ndarray   # shape (n + 2r, 3n), dtype int
    raw: Any = field(repr=False)

    @property
    def edge_equations(self) -> np.ndarray:
        """Rows 0…n-1, shape (n, 3n)."""
        return self.gluing_matrix[:self.num_tetrahedra]

    @property
    def meridian_equations(self) -> np.ndarray:
        """Rows n, n+2, n+4, …, shape (r, 3n)."""
        n, r = self.num_tetrahedra, self.num_cusps
        return self.gluing_matrix[n::2][:r]

    @property
    def longitude_equations(self) -> np.ndarray:
        """Rows n+1, n+3, n+5, …, shape (r, 3n)."""
        n, r = self.num_tetrahedra, self.num_cusps
        return self.gluing_matrix[n + 1::2][:r]

    def cusp_equations(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """(meridian_row, longitude_row) for cusp k, each shape (3n,)."""
        n = self.num_tetrahedra
        return (self.gluing_matrix[n + 2 * k], self.gluing_matrix[n + 2 * k + 1])


def load_manifold(name: str) -> ManifoldData:
    """
    Load a manifold by name using SnaPy.

    Raises
    ------
    ImportError  — if snappy not installed
    ValueError   — if name not recognized by SnaPy
    """
    try:
        import snappy
    except ImportError as exc:
        raise ImportError(f"Failed to import snappy: {exc}") from exc

    try:
        M = snappy.Manifold(name)
    except Exception as exc:
        raise ValueError(f"SnaPy could not load manifold '{name}': {exc}") from exc

    n = M.num_tetrahedra()
    r = M.num_cusps()
    raw_eqs = M.gluing_equations()
    rows, cols = raw_eqs.shape
    gluing_matrix = np.array(raw_eqs.list(), dtype=int).reshape(rows, cols)

    if rows != n + 2 * r or cols != 3 * n:
        raise ValueError(
            f"Unexpected gluing matrix shape {(rows, cols)}, "
            f"expected ({n + 2 * r}, {3 * n}) for n={n}, r={r}"
        )

    return ManifoldData(name=name, num_tetrahedra=n, num_cusps=r,
                        gluing_matrix=gluing_matrix, raw=M)
