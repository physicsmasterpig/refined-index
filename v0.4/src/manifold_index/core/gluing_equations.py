"""Gluing equation reduction: Z_i' elimination and independent edge selection."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import qr

from manifold_index.core.manifold import ManifoldData


@dataclass
class ReducedGluingData:
    n: int                              # tetrahedra
    r: int                              # cusps
    edge_coeffs: np.ndarray             # (n, 2n), int
    edge_consts: np.ndarray             # (n,), int
    cusp_coeffs: np.ndarray             # (2r, 2n), int
    cusp_consts: np.ndarray             # (2r,), int
    independent_edge_indices: list[int]  # length n-r, sorted
    symplectic_matrix: np.ndarray       # (2n, 2n), int — Ω

    @property
    def independent_edge_coeffs(self) -> np.ndarray:
        """shape (n-r, 2n)"""
        return self.edge_coeffs[self.independent_edge_indices]

    @property
    def independent_edge_consts(self) -> np.ndarray:
        """shape (n-r,)"""
        return self.edge_consts[self.independent_edge_indices]

    def meridian_coeffs(self, k: int) -> np.ndarray:
        """Reduced meridian for cusp k, shape (2n,)."""
        return self.cusp_coeffs[2 * k]

    def longitude_coeffs(self, k: int) -> np.ndarray:
        """Reduced longitude for cusp k, shape (2n,)."""
        return self.cusp_coeffs[2 * k + 1]

    def commutator(self, a: np.ndarray, b: np.ndarray) -> int:
        """Symplectic pairing [A, B] = a^T Ω b."""
        return int(a @ self.symplectic_matrix @ b)


def _reduce_row(row_3n: np.ndarray, n: int) -> tuple[int, np.ndarray]:
    """Substitute Z_i' = 1 - Z_i - Z_i'' into one row of the gluing matrix.

    Returns (const, coeff_2n) in interleaved (Z_1, Z_1'', Z_2, Z_2'', …) ordering.
    """
    const = 0
    coeff = np.zeros(2 * n, dtype=int)
    for i in range(n):
        f = int(row_3n[3 * i])      # coeff of Z_i
        g = int(row_3n[3 * i + 1])  # coeff of Z_i'
        h = int(row_3n[3 * i + 2])  # coeff of Z_i''
        const += g
        coeff[2 * i]     = f - g
        coeff[2 * i + 1] = h - g
    return const, coeff


def _build_symplectic_matrix(n: int) -> np.ndarray:
    """Build the 2n×2n symplectic matrix Ω in interleaved ordering."""
    omega = np.zeros((2 * n, 2 * n), dtype=int)
    for i in range(n):
        omega[2 * i,     2 * i + 1] = +1
        omega[2 * i + 1, 2 * i    ] = -1
    return omega


def _independent_row_indices(coeff_matrix: np.ndarray, expected_rank: int) -> list[int]:
    """Find `expected_rank` linearly independent rows via column-pivoted QR on transpose."""
    if expected_rank == 0:
        return []
    _, _, piv = qr(coeff_matrix.astype(float).T, pivoting=True)
    return sorted(piv[:expected_rank].tolist())


def reduce_gluing_equations(data: ManifoldData) -> ReducedGluingData:
    """Reduce all gluing equations by substituting Z_i' = 1 - Z_i - Z_i''."""
    n, r = data.num_tetrahedra, data.num_cusps

    edge_consts_list, edge_coeffs_list = [], []
    for row in data.edge_equations:
        c, v = _reduce_row(row, n)
        edge_consts_list.append(c)
        edge_coeffs_list.append(v)
    edge_consts = np.array(edge_consts_list, dtype=int)
    edge_coeffs = np.array(edge_coeffs_list, dtype=int)

    cusp_rows = data.gluing_matrix[n: n + 2 * r]
    cusp_consts_list, cusp_coeffs_list = [], []
    for row in cusp_rows:
        c, v = _reduce_row(row, n)
        cusp_consts_list.append(c)
        cusp_coeffs_list.append(v)
    cusp_consts = np.array(cusp_consts_list, dtype=int)
    cusp_coeffs = np.array(cusp_coeffs_list, dtype=int)

    independent_edge_indices = _independent_row_indices(edge_coeffs, n - r)
    omega = _build_symplectic_matrix(n)

    return ReducedGluingData(
        n=n, r=r,
        edge_coeffs=edge_coeffs,
        edge_consts=edge_consts,
        cusp_coeffs=cusp_coeffs,
        cusp_consts=cusp_consts,
        independent_edge_indices=independent_edge_indices,
        symplectic_matrix=omega,
    )
