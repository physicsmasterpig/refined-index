"""Phase space basis: easy-edge detection and hard-edge padding."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from fractions import Fraction
from itertools import product as _product

import numpy as np
from scipy.linalg import lstsq, qr

from manifold_index.core.manifold import ManifoldData
from manifold_index.core.gluing_equations import _reduce_row, reduce_gluing_equations

_Z   = 0  # Z_i   → column 3i
_ZP  = 1  # Z_i'  → column 3i+1
_ZPP = 2  # Z_i'' → column 3i+2
_OFF = 3  # tet contributes zero

MAX_COEFF = 3


@dataclass
class EasyEdgeResult:
    all_easy: list[np.ndarray]            # all discovered easy edges (3n-vectors)
    independent_easy_indices: list[int]   # indices into all_easy
    hard_padding: list[np.ndarray]        # hard edges to reach n-r total
    n: int
    r: int

    @property
    def num_independent_easy(self) -> int:
        return len(self.independent_easy_indices)

    @property
    def basis_edges(self) -> list[np.ndarray]:
        """[independent_easy... | hard_padding...], length n-r."""
        easy = [self.all_easy[i] for i in self.independent_easy_indices]
        return easy + self.hard_padding


def _is_easy(edge_3n: np.ndarray, n: int) -> bool:
    """Return True iff at most one of (Z_i, Z_i', Z_i'') is nonzero per tet."""
    for i in range(n):
        triplet = edge_3n[3 * i: 3 * i + 3]
        if np.count_nonzero(triplet) > 1:
            return False
    return True


def _build_constraint_matrix(
    pattern: tuple[int, ...],
    edge_rows: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build constraint system M @ a = rhs for the given slot pattern.

    For tet j with active slot d and inactive slots s1, s2:
        constraint: sum_i a_i * (col_s1 - col_s2) = 0  (1 row)
        ref col: col_s1
    For OFF tet: col_f-col_g = 0 and col_g-col_h = 0  (2 rows)
        ref col: col_f

    Final normalization row: a @ (2·ones - sum_ref) = 2
    """
    rows_M: list[np.ndarray] = []
    rhs_list: list[float] = []
    sum_ref = np.zeros(n, dtype=float)

    for j, slot in enumerate(pattern):
        col_f = edge_rows[:, 3 * j]
        col_g = edge_rows[:, 3 * j + 1]
        col_h = edge_rows[:, 3 * j + 2]
        slot_cols = {_Z: col_f, _ZP: col_g, _ZPP: col_h}

        if slot == _OFF:
            rows_M.append(col_f - col_g)
            rows_M.append(col_g - col_h)
            rhs_list.extend([0.0, 0.0])
            sum_ref += col_f
        else:
            inactive = [s for s in (_Z, _ZP, _ZPP) if s != slot]
            s1, s2 = inactive
            rows_M.append(slot_cols[s1] - slot_cols[s2])
            rhs_list.append(0.0)
            sum_ref += slot_cols[s1]

    # Normalization: a @ (2·ones − sum_ref) = 2
    rows_M.append(2.0 * np.ones(n, dtype=float) - sum_ref)
    rhs_list.append(2.0)

    return np.array(rows_M, dtype=float), np.array(rhs_list, dtype=float)


def _compute_b(a: np.ndarray, pattern: tuple[int, ...],
               edge_rows: np.ndarray, n: int) -> np.ndarray:
    """Recover b_j for each tet j given solution a."""
    b = np.zeros(n, dtype=float)
    for j, slot in enumerate(pattern):
        if slot == _OFF:
            ref_col = edge_rows[:, 3 * j]
        else:
            inactive = [s for s in (_Z, _ZP, _ZPP) if s != slot][0]
            ref_col = edge_rows[:, 3 * j + inactive]
        b[j] = -float(a @ ref_col)
    return b


def _solve_integer_system(M_float: np.ndarray, rhs_float: np.ndarray) -> list[np.ndarray]:
    """Exact Fraction-based RREF solver; returns all integer solutions with |coeff| <= MAX_COEFF.

    Emits a warning if the system is consistent but MAX_COEFF may be too small.
    """
    m, ncols = M_float.shape
    aug = [[Fraction(int(round(M_float[i, j]))) for j in range(ncols)]
           + [Fraction(int(round(rhs_float[i])))]
           for i in range(m)]

    pivot_col: list[int] = []
    row_idx = 0
    for col in range(ncols):
        found = None
        for r in range(row_idx, m):
            if aug[r][col] != 0:
                found = r
                break
        if found is None:
            continue
        aug[row_idx], aug[found] = aug[found], aug[row_idx]
        piv = aug[row_idx][col]
        aug[row_idx] = [v / piv for v in aug[row_idx]]
        for r in range(m):
            if r == row_idx:
                continue
            factor = aug[r][col]
            if factor != 0:
                aug[r] = [aug[r][j] - factor * aug[row_idx][j] for j in range(ncols + 1)]
        pivot_col.append(col)
        row_idx += 1

    for r in range(row_idx, m):
        if aug[r][ncols] != 0:
            return []

    free_cols = [c for c in range(ncols) if c not in pivot_col]

    if not free_cols:
        a: list[Fraction] = [Fraction(0)] * ncols
        for k, pcol in enumerate(pivot_col):
            val = aug[k][ncols]
            if val.denominator != 1:
                return []
            a[pcol] = val
        if any(abs(v) > MAX_COEFF for v in a):
            return []
        return [np.array([int(v) for v in a], dtype=int)]

    solutions: list[np.ndarray] = []
    search_range = range(-MAX_COEFF, MAX_COEFF + 1)
    for free_vals in _product(search_range, repeat=len(free_cols)):
        a = [Fraction(0)] * ncols
        for idx, fc in enumerate(free_cols):
            a[fc] = Fraction(free_vals[idx])
        ok = True
        for k, pcol in enumerate(pivot_col):
            val = aug[k][ncols]
            for fc_idx, fc in enumerate(free_cols):
                val -= aug[k][fc] * a[fc]
            if val.denominator != 1:
                ok = False
                break
            if abs(val) > MAX_COEFF:
                ok = False
                break
            a[pcol] = val
        if ok:
            solutions.append(np.array([int(v) for v in a], dtype=int))
    return solutions


def find_easy_edges(data: ManifoldData, tol: float = 1e-8) -> EasyEdgeResult:
    """Find all easy internal edges and pad with hard edges to reach n-r total."""
    n, r = data.num_tetrahedra, data.num_cusps
    target_rank = n - r
    edge_rows = data.edge_equations.astype(float)
    reduced = reduce_gluing_equations(data)

    seen: set[tuple] = set()
    all_easy: list[np.ndarray] = []
    all_easy_reduced: list[np.ndarray] = []

    # --- Stage 0: Fast check of raw SnaPy edge rows ---
    for row_idx in range(data.edge_equations.shape[0]):
        E = data.edge_equations[row_idx].copy()
        if np.any(E < 0):
            continue
        if not _is_easy(E, n):
            continue
        if int(np.sum(E)) != 2:
            continue
        key = tuple(E.tolist())
        if key in seen:
            continue
        seen.add(key)
        all_easy.append(E)
        _, rv = _reduce_row(E, n)
        all_easy_reduced.append(rv)

    # --- Stage 1: Pattern-first enumeration ---
    # Short-circuit: skip if Stage 0 already found a complete independent set.
    stage0_rank = (int(np.linalg.matrix_rank(np.array(all_easy_reduced, dtype=float)))
                   if all_easy_reduced else 0)
    if stage0_rank < target_rank:
        for pattern in _product((_Z, _ZP, _ZPP, _OFF), repeat=n):
            M, rhs = _build_constraint_matrix(pattern, edge_rows, n)
            if M.shape[0] == 0:
                continue

            a_candidates: list[np.ndarray] = []
            a_sol, residual, rank_M, _ = lstsq(M, rhs)

            lstsq_ok = True
            if residual.size > 0 and float(np.sum(residual ** 2)) > tol:
                lstsq_ok = False
            if lstsq_ok and np.linalg.norm(M @ a_sol - rhs) > tol:
                lstsq_ok = False

            if lstsq_ok:
                a_int = np.round(a_sol).astype(int)
                if np.linalg.norm(M @ a_int.astype(float) - rhs) < tol:
                    a_candidates.append(a_int)
                else:
                    a_candidates = _solve_integer_system(M, rhs)

            for a_int in a_candidates:
                b = _compute_b(a_int.astype(float), pattern, edge_rows, n)
                b_int = np.round(b).astype(int)
                if np.linalg.norm(b - b_int) > tol:
                    continue
                if 2 * int(np.sum(a_int)) + int(np.sum(b_int)) != 2:
                    continue

                T_matrix = np.zeros((n, 3 * n), dtype=int)
                for j in range(n):
                    T_matrix[j, 3 * j: 3 * j + 3] = 1
                E = a_int @ data.edge_equations.astype(int) + b_int @ T_matrix

                if np.any(E < 0):
                    continue
                if not _is_easy(E, n):
                    continue

                key = tuple(E.tolist())
                if key in seen:
                    continue
                seen.add(key)
                all_easy.append(E)
                _, rv = _reduce_row(E, n)
                all_easy_reduced.append(rv)

    # --- Stage 2: Select maximal independent subset ---
    independent_easy_indices: list[int] = []
    if all_easy_reduced:
        R = np.array(all_easy_reduced, dtype=float)
        sv = np.linalg.svd(R, compute_uv=False)
        easy_rank = min(int(np.sum(sv > tol)), target_rank)
        if easy_rank > 0:
            _, _, piv = qr(R.T, pivoting=True)
            independent_easy_indices = sorted(piv[:easy_rank].tolist())

    # --- Stage 3: Pad with hard edges ---
    hard_padding: list[np.ndarray] = []
    num_easy_ind = len(independent_easy_indices)
    if num_easy_ind < target_rank:
        current_reduced = [all_easy_reduced[i] for i in independent_easy_indices]
        for idx in reduced.independent_edge_indices:
            if len(independent_easy_indices) + len(hard_padding) >= target_rank:
                break
            snappy_row = data.edge_equations[idx]
            _, rv = _reduce_row(snappy_row, n)
            test_mat = np.array(current_reduced + [rv.astype(float)])
            sv = np.linalg.svd(test_mat, compute_uv=False)
            new_rank = int(np.sum(sv > tol))
            if new_rank > len(current_reduced):
                current_reduced.append(rv.astype(float))
                hard_padding.append(snappy_row)

    return EasyEdgeResult(
        all_easy=all_easy,
        independent_easy_indices=independent_easy_indices,
        hard_padding=hard_padding,
        n=n,
        r=r,
    )


find_phase_space_basis = find_easy_edges
