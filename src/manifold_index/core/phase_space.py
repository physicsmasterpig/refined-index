"""
core/phase_space.py — Easy edges and phase space basis construction (Step 6).

Responsibilities:
  - Identify *easy edges* from the gluing equations.
  - Select a maximal linearly independent subset of easy edges.
  - Construct a phase space basis: max easy edges first, padded with hard edges
    to reach n - r total independent internal edges.

Algorithm (pattern-first, Algorithm B from SPEC.md §Step 6):
  Every easy edge has a fixed *pattern* — for each tet j exactly one of
  {Z, Z', Z'', 0} is active.  Enumerating all 4^n patterns and, for each,
  solving the resulting overdetermined linear system on the (a_i) coefficients
  is far faster than brute-force (a, b) enumeration.

See SPEC.md §Step 6 for full mathematical specification.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product

import numpy as np
from scipy.linalg import lstsq, qr

from manifold_index.core.manifold import ManifoldData
from manifold_index.core.gluing_equations import ReducedGluingData, reduce_gluing_equations, _reduce_row


# Slot index within a tet triplet
_Z   = 0   # Z_i   → column 3i
_ZP  = 1   # Z_i'  → column 3i+1
_ZPP = 2   # Z_i'' → column 3i+2
_OFF = 3   # tet contributes zero


@dataclass
class EasyEdgeResult:
    """
    Result of the easy-edge search.

    Attributes
    ----------
    all_easy : list[np.ndarray]
        All distinct easy internal edges found (as 3n-vectors, non-negative int).
        Each evaluates to 2 under the normalization convention.
    independent_easy_indices : list[int]
        Indices into ``all_easy`` forming a maximal linearly independent subset
        (in the reduced 2n-variable space).
    hard_padding : list[np.ndarray]
        Hard edges (from SnaPy rows not already captured as easy) appended so
        that the combined basis has exactly n - r independent edges total.
    n : int
        Number of tetrahedra.
    r : int
        Number of cusps.
    """

    all_easy: list[np.ndarray]
    independent_easy_indices: list[int]
    hard_padding: list[np.ndarray]
    n: int
    r: int

    @property
    def num_independent_easy(self) -> int:
        return len(self.independent_easy_indices)

    @property
    def basis_edges(self) -> list[np.ndarray]:
        """
        Full ordered basis of n - r independent internal edges:
        independent easy edges first, then hard padding.
        """
        easy_part = [self.all_easy[i] for i in self.independent_easy_indices]
        return easy_part + self.hard_padding


def _is_easy(edge_3n: np.ndarray, n: int) -> bool:
    """Return True iff at most one of (Z_i, Z_i', Z_i'') is non-zero for every tet i."""
    for i in range(n):
        triplet = edge_3n[3*i : 3*i + 3]
        if np.count_nonzero(triplet) > 1:
            return False
    return True


def _build_constraint_matrix(pattern: tuple[int, ...], gluing_matrix_edge_rows: np.ndarray,
                              n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    For a fixed easy pattern (one of {_Z, _ZP, _ZPP, _OFF} per tet), build the
    overdetermined linear system  M @ a = rhs  where a = (a_1, ..., a_n).

    Rows 0..k-1 — pattern constraints (rhs = 0):
        For tet j with active slot d and inactive slots s1, s2:
            ∑_i aᵢ·col_s1 + bⱼ = 0 }  → difference:
            ∑_i aᵢ·col_s2 + bⱼ = 0 }    ∑_i aᵢ·(col_s1 − col_s2) = 0  (1 row)
        For _OFF tet: col_f−col_g = 0 and col_g−col_h = 0  (2 rows)

    Last row — normalization constraint (rhs = 2):
        bⱼ = −(a @ ref_col_j)  where ref_col_j is the first inactive col (or col_f).
        2∑aᵢ + ∑bⱼ = 2  →  a @ (2·ones − ∑_j ref_col_j) = 2

    Including normalization prevents lstsq from returning the trivial a = 0.

    Returns
    -------
    M   : np.ndarray, shape (num_constraints, n)
    rhs : np.ndarray, shape (num_constraints,)
    """
    rows_M: list[np.ndarray] = []
    rhs_list: list[float] = []
    sum_ref = np.zeros(n, dtype=float)  # ∑_j ref_col_j for normalization row

    for j, slot in enumerate(pattern):
        col_f = gluing_matrix_edge_rows[:, 3*j]
        col_g = gluing_matrix_edge_rows[:, 3*j + 1]
        col_h = gluing_matrix_edge_rows[:, 3*j + 2]
        slot_cols = {_Z: col_f, _ZP: col_g, _ZPP: col_h}

        if slot == _OFF:
            rows_M.append(col_f - col_g)
            rows_M.append(col_g - col_h)
            rhs_list.extend([0.0, 0.0])
            sum_ref += col_f          # b_j = -(a @ col_f)
        else:
            inactive = [s for s in (_Z, _ZP, _ZPP) if s != slot]
            s1, s2 = inactive
            rows_M.append(slot_cols[s1] - slot_cols[s2])
            rhs_list.append(0.0)
            sum_ref += slot_cols[s1]  # b_j = -(a @ col_s1)

    # Normalization: a @ (2·ones − sum_ref) = 2
    rows_M.append(2.0 * np.ones(n, dtype=float) - sum_ref)
    rhs_list.append(2.0)

    return np.array(rows_M, dtype=float), np.array(rhs_list, dtype=float)


def _compute_b(a: np.ndarray, pattern: tuple[int, ...],
               gluing_matrix_edge_rows: np.ndarray, n: int) -> np.ndarray:
    """
    Given a solution a, recover b_j for each tet j.

    For the active slot at tet j:
        b_j = -(∑_i a_i * inactive_col_{ij})
    using the first inactive slot (either choice gives the same b_j when the
    constraint is satisfied).

    For _OFF tets: b_j = -(∑_i a_i * col_f_{ij}).
    """
    b = np.zeros(n, dtype=float)
    for j, slot in enumerate(pattern):
        if slot == _OFF:
            ref_col = gluing_matrix_edge_rows[:, 3*j]
        else:
            inactive = [s for s in (_Z, _ZP, _ZPP) if s != slot][0]
            ref_col = gluing_matrix_edge_rows[:, 3*j + inactive]
        b[j] = -float(a @ ref_col)
    return b


# ---------------------------------------------------------------------------
# Exact rational solver for integer systems
# ---------------------------------------------------------------------------

def _solve_integer_system(M_float: np.ndarray, rhs_float: np.ndarray
                          ) -> list[np.ndarray]:
    """
    Find all small-coefficient integer solutions to  M @ a = rhs.

    Uses exact Fraction arithmetic to row-reduce the augmented matrix,
    then enumerates integer assignments for the free variables within
    a bounded range and back-substitutes to obtain the pivot variables.

    Parameters
    ----------
    M_float : np.ndarray, shape (m, n)
        Integer constraint matrix (entries must be integral).
    rhs_float : np.ndarray, shape (m,)
        Integer right-hand side (entries must be integral).

    Returns
    -------
    list[np.ndarray]
        All integer solution vectors ``a`` of shape ``(n,)`` with
        components in ``[-MAX_COEFF, MAX_COEFF]``.  May be empty if
        no integer solution exists.
    """
    MAX_COEFF = 3           # easy-edge coefficients are always small

    m, ncols = M_float.shape
    # Build augmented matrix [M | rhs] over Fraction
    aug = [[Fraction(int(round(M_float[i, j]))) for j in range(ncols)]
           + [Fraction(int(round(rhs_float[i])))]
           for i in range(m)]

    # --- Row echelon form (partial pivoting over Q) ---
    pivot_col: list[int] = []       # pivot_col[k] = column index of pivot in row k
    row_idx = 0
    for col in range(ncols):
        # Find a nonzero entry in column `col` at or below row_idx
        found = None
        for r in range(row_idx, m):
            if aug[r][col] != 0:
                found = r
                break
        if found is None:
            continue        # free variable
        # Swap into position
        aug[row_idx], aug[found] = aug[found], aug[row_idx]
        # Scale pivot row
        piv = aug[row_idx][col]
        aug[row_idx] = [v / piv for v in aug[row_idx]]
        # Eliminate below AND above (reduced row echelon)
        for r in range(m):
            if r == row_idx:
                continue
            factor = aug[r][col]
            if factor != 0:
                aug[r] = [aug[r][j] - factor * aug[row_idx][j]
                          for j in range(ncols + 1)]
        pivot_col.append(col)
        row_idx += 1

    rank = len(pivot_col)

    # Check consistency: any row with all-zero LHS but nonzero RHS?
    for r in range(rank, m):
        if aug[r][ncols] != 0:
            return []       # inconsistent system

    free_cols = [c for c in range(ncols) if c not in pivot_col]

    # --- No free variables: unique solution ---
    if not free_cols:
        a = [Fraction(0)] * ncols
        for k, pcol in enumerate(pivot_col):
            val = aug[k][ncols]
            if val.denominator != 1:
                return []   # no integer solution
            a[pcol] = val
        if any(abs(v) > MAX_COEFF for v in a):
            return []
        return [np.array([int(v) for v in a], dtype=int)]

    # --- Enumerate free-variable assignments ---
    from itertools import product as _product

    search_range = range(-MAX_COEFF, MAX_COEFF + 1)
    solutions: list[np.ndarray] = []

    for free_vals in _product(search_range, repeat=len(free_cols)):
        a = [Fraction(0)] * ncols
        # Assign free variables
        for idx, col in enumerate(free_cols):
            a[col] = Fraction(free_vals[idx])

        # Back-substitute pivot variables
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

        if not ok:
            continue

        solutions.append(np.array([int(v) for v in a], dtype=int))

    return solutions


def find_easy_edges(data: ManifoldData,
                    tol: float = 1e-8) -> EasyEdgeResult:
    """
    Find all easy internal edges via the pattern-first algorithm (Algorithm B).

    Parameters
    ----------
    data : ManifoldData
        Raw manifold data (gluing matrix etc.).
    tol : float
        Tolerance for treating a floating-point solution as integer.

    Returns
    -------
    EasyEdgeResult
    """
    n = data.num_tetrahedra
    r = data.num_cusps
    target_rank = n - r

    edge_rows = data.edge_equations.astype(float)  # shape (n, 3n)

    # We also need the reduced equations for linear-independence checks.
    reduced = reduce_gluing_equations(data)

    # --- Stage 0: direct check of SnaPy edge rows for easiness ---
    # Some easy edges are exactly SnaPy edge equation rows; checking them
    # directly is both fast and immune to the numerical issues that can
    # plague the pattern-first algorithm on underdetermined systems.
    seen: set[tuple[int, ...]] = set()          # deduplicate via tuple of edge vector
    all_easy: list[np.ndarray] = []
    all_easy_reduced: list[np.ndarray] = []     # for independence check

    for row_idx in range(data.edge_equations.shape[0]):
        E = data.edge_equations[row_idx].copy()
        if np.any(E < 0):
            continue
        if not _is_easy(E, n):
            continue
        # Verify normalization: sum of entries = 2 (since ∑coeffs = 2 for
        # any internal edge satisfying 2∑a + ∑b = 2).
        if int(np.sum(E)) != 2:
            continue
        key = tuple(E.tolist())
        if key in seen:
            continue
        seen.add(key)
        all_easy.append(E)
        _, reduced_vec = _reduce_row(E, n)
        all_easy_reduced.append(reduced_vec)

    # --- Stage 1: pattern-first search ---
    # For each pattern in {_Z, _ZP, _ZPP, _OFF}^n:
    #   build constraint system M @ a = rhs
    #   Fast path: lstsq + round (works for determined/overdetermined systems)
    #   Slow path: exact rational solver (for underdetermined systems where
    #              lstsq's min-norm solution rounds incorrectly)

    for pattern in product((_Z, _ZP, _ZPP, _OFF), repeat=n):
        M, rhs = _build_constraint_matrix(pattern, edge_rows, n)

        if M.shape[0] == 0:
            continue

        # --- Fast path: lstsq + round ---
        a_candidates: list[np.ndarray] = []

        a_sol, residual, rank_M, _ = lstsq(M, rhs)

        # Check if lstsq found a consistent solution
        lstsq_ok = True
        if residual.size > 0 and np.sum(residual**2) > tol:
            lstsq_ok = False
        if lstsq_ok and np.linalg.norm(M @ a_sol - rhs) > tol:
            lstsq_ok = False

        if lstsq_ok:
            a_int = np.round(a_sol).astype(int)
            if np.linalg.norm(M @ a_int.astype(float) - rhs) < tol:
                # Rounded solution is exact — use it
                a_candidates.append(a_int)
            else:
                # Rounded solution fails (underdetermined system) — exact fallback
                a_candidates = _solve_integer_system(M, rhs)
        # else: system inconsistent, skip pattern

        for a_int in a_candidates:
            # Compute b_j
            b = _compute_b(a_int.astype(float), pattern, edge_rows, n)
            b_int = np.round(b).astype(int)
            if np.linalg.norm(b - b_int) > tol:
                continue

            # Check normalization: 2∑aᵢ + ∑bⱼ = 2
            norm_val = 2 * int(np.sum(a_int)) + int(np.sum(b_int))
            if norm_val != 2:
                continue

            # Reconstruct 3n-vector: E = ∑ aᵢ Cᵢ + ∑ bⱼ Tⱼ
            # T_j has (1,1,1) at tet j and 0 elsewhere
            T_matrix = np.zeros((n, 3*n), dtype=int)
            for j in range(n):
                T_matrix[j, 3*j : 3*j+3] = 1

            E = a_int @ edge_rows.astype(int) + b_int @ T_matrix  # shape (3n,)

            # Check non-negativity
            if np.any(E < 0):
                continue

            # Check easiness (double-check; should be guaranteed by construction)
            if not _is_easy(E, n):
                continue

            # Deduplicate
            key = tuple(E.tolist())
            if key in seen:
                continue
            seen.add(key)

            all_easy.append(E)

            # Compute the reduced-space representation for independence checking
            _, reduced_vec = _reduce_row(E, n)
            all_easy_reduced.append(reduced_vec)

    # --- Stage 2: select maximal independent subset of easy edges ---
    independent_easy_indices: list[int] = []
    if all_easy_reduced:
        R_mat = np.array(all_easy_reduced, dtype=float)  # shape (k, 2n)
        _, _, piv = qr(R_mat.T, pivoting=True)
        # How many easy edges are linearly independent?
        # Use SVD rank check
        sv = np.linalg.svd(R_mat, compute_uv=False)
        easy_rank = int(np.sum(sv > tol))
        easy_rank = min(easy_rank, target_rank)
        independent_easy_indices = sorted(piv[:easy_rank].tolist())

    # --- Stage 3: pad with hard SnaPy edges to reach target_rank ---
    hard_padding: list[np.ndarray] = []
    num_easy_ind = len(independent_easy_indices)
    if num_easy_ind < target_rank:
        # Use the SnaPy independent edge basis (already computed)
        # Build current basis in reduced space
        current_reduced = [all_easy_reduced[i] for i in independent_easy_indices]

        for idx in reduced.independent_edge_indices:
            snappy_row = data.edge_equations[idx]  # 3n-vector
            _, r_vec = _reduce_row(snappy_row, n)

            # Check if adding this row increases the rank
            test_mat = np.array(current_reduced + [r_vec], dtype=float)
            sv = np.linalg.svd(test_mat, compute_uv=False)
            new_rank = int(np.sum(sv > tol))

            if new_rank > len(current_reduced):
                current_reduced.append(r_vec)
                hard_padding.append(snappy_row)

            if len(independent_easy_indices) + len(hard_padding) == target_rank:
                break

    return EasyEdgeResult(
        all_easy=all_easy,
        independent_easy_indices=independent_easy_indices,
        hard_padding=hard_padding,
        n=n,
        r=r,
    )


# Keep old stub name as alias for now
def find_phase_space_basis(manifold: ManifoldData) -> EasyEdgeResult:
    """Alias for find_easy_edges (forward-compatible name)."""
    return find_easy_edges(manifold)
