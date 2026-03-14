"""
core/manifold.py — Manifold loading and SnaPy interface (Step 2).

Responsibilities:
  - Accept a manifold name string
  - Use SnaPy to extract:
      · number of tetrahedra  (n)
      · number of cusps       (r)
      · gluing equations as a numpy integer matrix of shape (n + 2r, 3n)
  - Return a ManifoldData dataclass

Gluing equation matrix layout (see SPEC.md §Step 2):
  Rows 0       … n-1      : edge equations               (n rows)
  Row  n + 2k             : meridian equation for cusp k  (k = 0…r-1)
  Row  n + 2k+1           : longitude equation for cusp k (k = 0…r-1)

  Cusp rows are INTERLEAVED:  μ₀, λ₀, μ₁, λ₁, …, μᵣ₋₁, λᵣ₋₁

  Columns ordered as:  Z_1, Z_1', Z_1'',  Z_2, Z_2', Z_2'',  …,  Z_n, Z_n', Z_n''

Each row encodes:  ∑_i ( a_i·Z_i + b_i·Z_i' + c_i·Z_i'' ) = 2πi · RHS
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ManifoldData:
    """
    Raw data extracted from SnaPy for a given manifold.

    Attributes
    ----------
    name : str
        The SnaPy manifold name string.
    num_tetrahedra : int
        Number of tetrahedra (n).
    num_cusps : int
        Number of cusps (r).
    gluing_matrix : np.ndarray, shape (n + 2r, 3n), dtype int
        Integer coefficient matrix of the gluing equations.
        Row layout: [edge eqs (n) | meridian eqs (r) | longitude eqs (r)]
        Col layout: [Z_1, Z_1', Z_1'', Z_2, Z_2', Z_2'', ..., Z_n, Z_n', Z_n'']
    raw : Any
        The raw SnaPy Manifold object, kept for any further queries.
    """

    name: str
    num_tetrahedra: int
    num_cusps: int
    gluing_matrix: np.ndarray = field(default=None)
    raw: Any = field(default=None, repr=False)

    # Convenience slices into gluing_matrix
    @property
    def edge_equations(self) -> np.ndarray:
        """Rows 0…n-1: edge equations, shape (n, 3n)."""
        n = self.num_tetrahedra
        return self.gluing_matrix[:n]

    @property
    def meridian_equations(self) -> np.ndarray:
        """
        Rows n, n+2, n+4, …: meridian (μ) equations, shape (r, 3n).
        Cusp rows are interleaved μ₀,λ₀,μ₁,λ₁,…, so meridians are even-offset rows.
        """
        n, r = self.num_tetrahedra, self.num_cusps
        return self.gluing_matrix[n::2][:r]

    @property
    def longitude_equations(self) -> np.ndarray:
        """
        Rows n+1, n+3, n+5, …: longitude (λ) equations, shape (r, 3n).
        Cusp rows are interleaved μ₀,λ₀,μ₁,λ₁,…, so longitudes are odd-offset rows.
        """
        n, r = self.num_tetrahedra, self.num_cusps
        return self.gluing_matrix[n + 1::2][:r]

    def cusp_equations(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (meridian, longitude) equation rows for cusp k.

        Parameters
        ----------
        k : int
            Cusp index (0-based).

        Returns
        -------
        (meridian_row, longitude_row) each of shape (3n,)
        """
        n = self.num_tetrahedra
        return (
            self.gluing_matrix[n + 2 * k],
            self.gluing_matrix[n + 2 * k + 1],
        )


def load_manifold(name: str) -> ManifoldData:
    """
    Load a manifold by name using SnaPy and extract relevant data.

    Parameters
    ----------
    name : str
        A SnaPy-recognizable manifold name (e.g. 'm004', '4_1').

    Returns
    -------
    ManifoldData

    Raises
    ------
    ImportError
        If SnaPy is not installed.
    ValueError
        If SnaPy does not recognize the manifold name.
    """
    try:
        import snappy  # type: ignore[import]
    except ImportError as exc:
        # In a frozen (PyInstaller) app the real error may be a missing
        # sub-dependency, not snappy itself.  Propagate the original
        # message so the user (and developer) can diagnose it.
        raise ImportError(
            f"Failed to import snappy: {exc}"
        ) from exc

    try:
        M = snappy.Manifold(name)
    except Exception as exc:
        raise ValueError(f"SnaPy could not load manifold '{name}': {exc}") from exc

    n = M.num_tetrahedra()
    r = M.num_cusps()

    # Convert SnaPy's SimpleMatrix to a plain numpy integer array.
    # SimpleMatrix.shape is a tuple (rows, cols); .list() gives a flat list.
    raw_eqs = M.gluing_equations()
    rows, cols = raw_eqs.shape
    gluing_matrix = np.array(raw_eqs.list(), dtype=int).reshape(rows, cols)

    # Sanity check against SPEC.md §Step 2 layout
    assert rows == n + 2 * r, f"Unexpected gluing eq rows: {rows} (expected {n + 2*r})"
    assert cols == 3 * n,     f"Unexpected gluing eq cols: {cols} (expected {3*n})"

    return ManifoldData(
        name=name,
        num_tetrahedra=n,
        num_cusps=r,
        gluing_matrix=gluing_matrix,
        raw=M,
    )
