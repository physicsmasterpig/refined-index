"""
core/gluing_equations.py — Reduced gluing equations and independent edge basis (Step 2 post-processing).

Implements:

1. **Variable reduction** (substituting Z_i' = 1 - Z_i - Z_i''):
   - Reduces each gluing equation row from 3n variables to 2n variables
     {Z_1, Z_1'', Z_2, Z_2'', ..., Z_n, Z_n''} plus a constant.
   - Reduced coefficient of Z_i:   f_i - g_i
   - Reduced coefficient of Z_i'': h_i - g_i
   - Constant contribution of tet i: g_i

2. **Independent edge basis**:
   - The n SnaPy edge equations have rank n - r in the reduced basis.
   - We pick a maximal set of n - r linearly independent rows from the
     SnaPy edge equations via column-pivoted QR on the coefficient matrix.

3. **Symplectic pairing**:
   - [Z_i, Z_i''] = 1, cross-tet variables commute.
   - For two linear combos A, B in the reduced basis:
       [A, B] = ∑_i ( a_i · b_i'' - a_i'' · b_i )
   - This is precomputed as a (2n x 2n) symplectic matrix Ω.

See SPEC.md §Step 2 for full mathematical specification.
"""

from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import qr

from manifold_index.core.manifold import ManifoldData


# ---------------------------------------------------------------------------
# Reduced equation representation
# ---------------------------------------------------------------------------

@dataclass
class ReducedGluingData:
    """
    The gluing equations expressed in the reduced variable basis
    {Z_1, Z_1'', Z_2, Z_2'', ..., Z_n, Z_n''} after substituting
    Z_i' = 1 - Z_i - Z_i'' for every tetrahedron i.

    Attributes
    ----------
    n : int
        Number of tetrahedra.
    r : int
        Number of cusps.
    edge_coeffs : np.ndarray, shape (n, 2n), dtype int
        Coefficient matrix of the n SnaPy edge equations in the reduced basis.
    edge_consts : np.ndarray, shape (n,), dtype int
        Constant terms of the n SnaPy edge equations after reduction.
        Each edge equation reads:  edge_consts[i] + edge_coeffs[i] · v = 2
        where v = [Z_1, Z_1'', ..., Z_n, Z_n''].
    cusp_coeffs : np.ndarray, shape (2r, 2n), dtype int
        Meridian and longitude equations in the reduced basis (interleaved:
        μ_0, λ_0, μ_1, λ_1, ...).
    cusp_consts : np.ndarray, shape (2r,), dtype int
        Constant terms of the cusp equations after reduction.
    independent_edge_indices : list[int]
        Indices (into the n SnaPy edge rows) of a maximal linearly independent
        subset of edge equations.  Length = n - r.
    symplectic_matrix : np.ndarray, shape (2n, 2n), dtype int
        The symplectic form Ω on the reduced variable space.
        Ω[2i, 2i+1] = +1  (from [Z_i, Z_i''] = 1)
        Ω[2i+1, 2i] = -1
        All other entries = 0.
    """

    n: int
    r: int
    edge_coeffs: np.ndarray
    edge_consts: np.ndarray
    cusp_coeffs: np.ndarray
    cusp_consts: np.ndarray
    independent_edge_indices: list[int]
    symplectic_matrix: np.ndarray

    @property
    def independent_edge_coeffs(self) -> np.ndarray:
        """Coefficient rows for the independent edge basis, shape (n-r, 2n)."""
        return self.edge_coeffs[self.independent_edge_indices]

    @property
    def independent_edge_consts(self) -> np.ndarray:
        """Constant terms for the independent edge basis, shape (n-r,)."""
        return self.edge_consts[self.independent_edge_indices]

    def meridian_coeffs(self, k: int) -> np.ndarray:
        """Reduced coefficient vector for meridian of cusp k, shape (2n,)."""
        return self.cusp_coeffs[2 * k]

    def longitude_coeffs(self, k: int) -> np.ndarray:
        """Reduced coefficient vector for longitude of cusp k, shape (2n,)."""
        return self.cusp_coeffs[2 * k + 1]

    def commutator(self, a: np.ndarray, b: np.ndarray) -> int:
        """
        Compute the symplectic pairing [A, B] = a^T Ω b for two vectors
        a, b in the reduced variable space (length 2n).
        """
        return int(a @ self.symplectic_matrix @ b)


# ---------------------------------------------------------------------------
# Reduction helpers
# ---------------------------------------------------------------------------

def _reduce_row(row: np.ndarray, n: int) -> tuple[int, np.ndarray]:
    """
    Substitute Z_i' = 1 - Z_i - Z_i'' into one row of the gluing matrix.

    Parameters
    ----------
    row : np.ndarray, shape (3n,)
        Original coefficient row  [f_1, g_1, h_1,  f_2, g_2, h_2, ...,  f_n, g_n, h_n].
    n : int
        Number of tetrahedra.

    Returns
    -------
    const : int
        Sum of g_i coefficients (= constant term after substitution).
    coeff : np.ndarray, shape (2n,), dtype int
        Coefficients [f_1-g_1, h_1-g_1,  f_2-g_2, h_2-g_2, ...] of
        [Z_1, Z_1'', Z_2, Z_2'', ...].
    """
    const = 0
    coeff = np.zeros(2 * n, dtype=int)
    for i in range(n):
        f, g, h = int(row[3*i]), int(row[3*i+1]), int(row[3*i+2])
        const += g
        coeff[2*i]   = f - g   # Z_i
        coeff[2*i+1] = h - g   # Z_i''
    return const, coeff


def _build_symplectic_matrix(n: int) -> np.ndarray:
    """
    Build the (2n x 2n) symplectic matrix Ω where
        Ω[2i, 2i+1] = +1   ([Z_i, Z_i''] = 1)
        Ω[2i+1, 2i] = -1
    and all other entries are zero.
    """
    omega = np.zeros((2 * n, 2 * n), dtype=int)
    for i in range(n):
        omega[2*i,   2*i+1] = +1
        omega[2*i+1, 2*i  ] = -1
    return omega


def _independent_row_indices(coeff_matrix: np.ndarray, expected_rank: int) -> list[int]:
    """
    Find a maximal set of linearly independent row indices from coeff_matrix
    using column-pivoted QR on the transpose.

    Returns
    -------
    list[int]  length = expected_rank, sorted ascending.
    """
    _, _, piv = qr(coeff_matrix.astype(float).T, pivoting=True)
    return sorted(piv[:expected_rank].tolist())


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def reduce_gluing_equations(data: ManifoldData) -> ReducedGluingData:
    """
    Reduce the gluing equations by substituting Z_i' = 1 - Z_i - Z_i'' and
    extract a linearly independent basis of edge equations.

    Parameters
    ----------
    data : ManifoldData

    Returns
    -------
    ReducedGluingData
    """
    n, r = data.num_tetrahedra, data.num_cusps

    # --- reduce every row ---
    def reduce_block(rows):
        consts, coeffs = [], []
        for row in rows:
            c, v = _reduce_row(row, n)
            consts.append(c)
            coeffs.append(v)
        return np.array(consts, dtype=int), np.array(coeffs, dtype=int)

    edge_consts, edge_coeffs   = reduce_block(data.edge_equations)

    # cusp rows in original matrix: rows n, n+1, n+2, ..., n+2r-1
    cusp_rows = data.gluing_matrix[n : n + 2*r]
    cusp_consts, cusp_coeffs = reduce_block(cusp_rows)

    # --- independent edge indices ---
    expected_rank = n - r
    ind_indices = _independent_row_indices(edge_coeffs, expected_rank)

    # --- symplectic matrix ---
    omega = _build_symplectic_matrix(n)

    return ReducedGluingData(
        n=n,
        r=r,
        edge_coeffs=edge_coeffs,
        edge_consts=edge_consts,
        cusp_coeffs=cusp_coeffs,
        cusp_consts=cusp_consts,
        independent_edge_indices=ind_indices,
        symplectic_matrix=omega,
    )
