"""
core/neumann_zagier.py — Neumann-Zagier matrix and affine shift (Step 3).

Constructs the symplectic matrix ``g_NZ ∈ Sp(2n, Q)`` and the affine shift
vectors ``ν_x``, ``ν_p`` from the gluing data and the easy-edge basis.

──────────────────────────────────────────────────────────────────────────────
Mathematical background
──────────────────────────────────────────────────────────────────────────────

Variables
---------
After substituting  Z_i' = 1 - Z_i - Z_i''  the reduction convention,
each gluing equation is linear in the 2n variables

    v = (Z_1, Z_2, …, Z_n,  Z_1'', Z_2'', …, Z_n'')      [block ordering]

Column ordering in g_NZ follows block ordering; the permutation that converts
our internal interleaved ordering (Z_1, Z_1'', Z_2, Z_2'', …) is applied
automatically inside this module.

Row structure of g_NZ  (size 2n × 2n)
--------------------------------------
Top n rows — "position" variables:
    Rows 0 … r-1           : meridian equations  (one per cusp)
    Rows r … r+d_hard-1    : hard internal edges
    Rows r+d_hard … n-1    : easy internal edges

Bottom n rows — "momentum" variables:
    Rows n … n+r-1         : longitude / 2  (one per cusp)
    Rows n+r … 2n-1        : Γ vectors  (momentum conjugate to each internal edge)

Reduction convention
--------------------
With Z_i' = 1 - Z_i - Z_i'' (same as Step 2), a row [f_i, g_i, h_i]
at tet i reduces to:

    g_i  +  (f_i - g_i)·Z_i  +  (h_i - g_i)·Z_i''

The constant term is  ∑_i g_i  (returned by ``_reduce_row``).
The linear coefficients are  (f_i - g_i, h_i - g_i).

Symplectic form
---------------
In block ordering:

    Ω_block = [[0_n,  I_n],
               [-I_n, 0_n]]

Pairing:  [u, v] = u_Z · v_{Z''} - u_{Z''} · v_Z  =  u @ Ω_block @ v.

The g_NZ matrix is symplectic: g_NZ Ω g_NZ^T = Ω.

Γ construction
--------------
Given the n × 2n position block P and the first r momentum rows (longitudes/2),
the remaining n-r Γ rows are found by solving the integer linear system:

    [P ; Q_long]  Ω  Γ^T  =  RHS

where RHS has shape (n+r) × (n-r) with RHS[r:n, :] = I_{n-r}, rest = 0.
A float solution via least-squares is rounded to integers and validated.

Affine shift ν
--------------
The affine shift satisfies  g_NZ_row · v + ν = RHS.  Since the reduced
equation is  c + coeff · v = RHS, we get  ν = c − RHS  where  c = ∑_i g_i
is the constant from ``_reduce_row`` (using Z_i' = 1 - Z_i - Z_i'').

RHS values by row type:
- Meridians (cusp equations):      RHS = 0  →  ν_x = c
- Internal edges (edge equations): RHS = 2  →  ν_x = c − 2
- Longitude/2:                     ν_p = c_long / 2  (halved because the
                                   momentum row stores L/2, not L)

For Γ rows (constructed, not from gluing data), ν_p = 0.

See SPEC.md §Step 3 for the full specification.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from manifold_index.core.manifold import ManifoldData
from manifold_index.core.gluing_equations import ReducedGluingData, _reduce_row
from manifold_index.core.phase_space import EasyEdgeResult


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class NeumannZagierData:
    """
    The Neumann-Zagier symplectic matrix and affine shift for a manifold.

    g_NZ lives in Sp(2n, Q): the longitude/2 rows may have half-integer
    entries when the SnapPy longitude has odd coefficients.  nu_p is
    correspondingly a float array; nu_x and the position rows remain integer.

    Attributes
    ----------
    g_NZ : np.ndarray, shape (2n, 2n), dtype float
        The symplectic matrix.  Columns correspond to the block-ordered
        variables (Z_1, …, Z_n, Z_1'', …, Z_n'').
        Rows are ordered:
            0 … r-1           meridians
            r … r+d_hard-1    hard edges
            r+d_hard … n-1    easy edges
            n … n+r-1         longitudes / 2
            n+r … 2n-1        Γ vectors (momenta of internal edges)
    nu_x : np.ndarray, shape (n,), dtype int
        Affine shift for the top-n (position) rows.
    nu_p : np.ndarray, shape (n,), dtype float
        Affine shift for the bottom-n (momentum) rows.
    n : int
        Number of tetrahedra.
    r : int
        Number of cusps.
    num_hard : int
        Number of hard internal edges in the basis.
    num_easy : int
        Number of (independent) easy internal edges in the basis.
    """

    g_NZ: np.ndarray
    nu_x: np.ndarray
    nu_p: np.ndarray
    n: int
    r: int
    num_hard: int
    num_easy: int

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def symplectic_form(self) -> np.ndarray:
        """Standard block symplectic form Ω_block, shape (2n, 2n)."""
        n = self.n
        omega = np.zeros((2 * n, 2 * n), dtype=int)
        omega[:n, n:] = np.eye(n, dtype=int)
        omega[n:, :n] = -np.eye(n, dtype=int)
        return omega

    def is_symplectic(self, tol: float = 1e-9) -> bool:
        """Check that g_NZ Ω g_NZ^T = Ω."""
        omega = self.symplectic_form
        lhs = self.g_NZ @ omega @ self.g_NZ.T
        return bool(np.allclose(lhs, omega, atol=tol))

    def g_NZ_inv(self) -> np.ndarray:
        """
        Compute g_NZ^{-1} exactly using the symplectic identity:
            g^{-1} = [[D^T, -B^T], [-C^T, A^T]]
        where g = [[A, B], [C, D]]  (n × n blocks).

        Returns an object array of :class:`fractions.Fraction` so that
        downstream callers get exact rational arithmetic.  The longitude/2
        rows of g_NZ may have half-integer entries, which float→int rounding
        would silently corrupt.
        """
        n = self.n
        A = self.g_NZ[:n, :n]
        B = self.g_NZ[:n, n:]
        C = self.g_NZ[n:, :n]
        D = self.g_NZ[n:, n:]
        top = np.hstack([D.T, -B.T])
        bot = np.hstack([-C.T, A.T])
        result_float = np.vstack([top, bot])
        # Convert to exact Fraction.  Entries are always rational (the
        # symplectic formula only uses transpose and negation, so the
        # denominator structure of g_NZ is preserved exactly).
        # limit_denominator(1000) recovers the exact fraction from the
        # float without any rounding.
        return np.array(
            [[Fraction(v).limit_denominator(1000) for v in row]
             for row in result_float],
            dtype=object,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _interleaved_to_block(coeff_interleaved: np.ndarray, n: int) -> np.ndarray:
    """
    Convert a 2n coefficient vector from interleaved ordering
        (Z_1, Z_1'', Z_2, Z_2'', …, Z_n, Z_n'')
    to block ordering
        (Z_1, Z_2, …, Z_n,  Z_1'', Z_2'', …, Z_n'').

    Parameters
    ----------
    coeff_interleaved : np.ndarray, shape (2n,)
    n : int

    Returns
    -------
    np.ndarray, shape (2n,)
    """
    perm = np.array([2 * i for i in range(n)] + [2 * i + 1 for i in range(n)])
    return coeff_interleaved[perm]


def _reduce_to_block(row_3n: np.ndarray, n: int) -> tuple[int, np.ndarray]:
    """
    Reduce a 3n gluing-equation row to the 2n (block-ordered) representation.

    Uses the Z_i' = 1 - Z_i - Z_i'' convention.

    Returns
    -------
    const : int
        Constant term under the Z_i' = 1 - Z_i - Z_i'' convention.
    coeff_block : np.ndarray, shape (2n,)
        Linear coefficients in block ordering (Z_1, …, Z_n, Z_1'', …, Z_n'').
    """
    const, coeff_interleaved = _reduce_row(row_3n, n)
    coeff_block = _interleaved_to_block(coeff_interleaved, n)
    return const, coeff_block


def _build_omega_block(n: int) -> np.ndarray:
    """Return Ω_block = [[0, I_n], [-I_n, 0]], shape (2n, 2n), dtype int."""
    omega = np.zeros((2 * n, 2 * n), dtype=int)
    omega[:n, n:] = np.eye(n, dtype=int)
    omega[n:, :n] = -np.eye(n, dtype=int)
    return omega


def _int_right_inverse(A_int: np.ndarray) -> np.ndarray:
    """
    Given an n × 2n integer matrix A whose Smith Normal Form has all invariant
    factors equal to 1, return a 2n × n integer matrix Q_T such that
    A @ Q_T = I_n.

    Algorithm
    ---------
    Euclidean column reduction (tracks transformation matrix V) produces
    A @ V = [H | 0] where H is n × n lower-triangular with 1s on the diagonal.
    The right inverse is  Q_T = V[:, :n] @ H_inv, where H_inv is the integer
    inverse of H (also lower-triangular with 1s on the diagonal).
    """
    n, m = A_int.shape  # m = 2n

    # Work with Python lists of lists for exact integer arithmetic
    A = [list(map(int, row)) for row in A_int]
    V = [[int(i == j) for j in range(m)] for i in range(m)]  # identity 2n×2n

    def swap_cols(c1: int, c2: int) -> None:
        for row in A:
            row[c1], row[c2] = row[c2], row[c1]
        for row in V:
            row[c1], row[c2] = row[c2], row[c1]

    def add_col(src: int, dst: int, factor: int) -> None:
        """col[dst] += factor * col[src]"""
        for row in A:
            row[dst] += factor * row[src]
        for row in V:
            row[dst] += factor * row[src]

    def negate_col(c: int) -> None:
        for row in A:
            row[c] = -row[c]
        for row in V:
            row[c] = -row[c]

    # Forward pass: for each pivot row i, reduce row i to have a single
    # nonzero at column i (value = gcd = 1).
    for pivot_row in range(n):
        col_start = pivot_row

        # Euclidean reduction loop: bring gcd of A[pivot_row][col_start:] to col_start
        while True:
            nonzero_cols = [c for c in range(col_start, m)
                            if A[pivot_row][c] != 0]
            if not nonzero_cols:
                raise RuntimeError(
                    f"Row {pivot_row} is all zeros — matrix does not have "
                    "all invariant factors = 1."
                )
            # Move the smallest-absolute-value column to col_start
            best = min(nonzero_cols, key=lambda c: abs(A[pivot_row][c]))
            if best != col_start:
                swap_cols(col_start, best)

            # Attempt to zero out all entries to the right of col_start in pivot_row
            changed = False
            for c in range(col_start + 1, m):
                if A[pivot_row][c] != 0:
                    q = A[pivot_row][c] // A[pivot_row][col_start]
                    if q != 0:
                        add_col(col_start, c, -q)
                        changed = True

            # Check termination: all entries right of pivot_row are zero
            if all(A[pivot_row][c] == 0 for c in range(col_start + 1, m)):
                break

            # If no progress with floor division, try swapping and one step
            if not changed:
                remaining = [c for c in range(col_start + 1, m)
                             if A[pivot_row][c] != 0]
                if remaining:
                    c2 = remaining[0]
                    swap_cols(col_start, c2)
                    # Now col_start has the formerly smaller element — retry
                    # (the outer while-True will re-examine)

        # Ensure pivot is positive
        if A[pivot_row][col_start] < 0:
            negate_col(col_start)

        # After processing this pivot_row, the invariant is:
        #   A[pivot_row][j] = 0  for all j > pivot_row
        #   A[pivot_row][pivot_row] = 1
        # (sub-diagonal entries A[pivot_row][j] for j < pivot_row may be nonzero)

    # ----------------------------------------------------------------
    # V[:, :n] satisfies A @ V[:, :n] = H  where H is lower-triangular
    # with 1s on the diagonal.  Compute H and its integer inverse.
    # ----------------------------------------------------------------
    V_arr = np.array(V, dtype=int)          # (2n, 2n)
    A_arr = np.array(A, dtype=int)          # (n, 2n)  — current reduced form

    # H = A_arr[:, :n]  (should be lower-triangular with diag = 1)
    H = A_arr[:, :n].copy()

    # Integer inverse of lower-triangular H with unit diagonal via forward substitution:
    # If H @ X = I_n then X is also lower-triangular with unit diagonal.
    H_inv = np.eye(n, dtype=int)
    for i in range(n):
        for j in range(i):
            H_inv[i] -= H[i, j] * H_inv[j]

    Q_T = V_arr[:, :n] @ H_inv            # (2n, n)
    return Q_T


def _make_isotropic(
    Q_T: np.ndarray,
    P: np.ndarray,
    omega: np.ndarray,
) -> np.ndarray:
    """
    Given a 2n × n integer right-inverse Q_T of P @ omega (so P @ omega @ Q_T = I_n),
    return a corrected Q_T' with the same right-inverse property AND
    Q_T'^T @ omega @ Q_T' = 0 (isotropic).

    The correction adds integer multiples of the null-space vectors (columns of P^T)
    to the columns of Q_T.  Setting C = strictly-lower-triangular part of
    S = Q_T^T @ omega @ Q_T (anti-symmetric integer matrix), the adjustment is:

        Q_T' = Q_T + P^T @ C

    which zeroes S without changing P @ omega @ Q_T' = I_n.
    """
    n = P.shape[0]
    # Anti-symmetric pairing matrix S[i,j] = [col_i(Q_T), col_j(Q_T)]
    S = Q_T.T @ omega @ Q_T  # (n, n), integer, anti-symmetric

    # C = strictly lower-triangular part of S
    C = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i):  # j < i  →  lower triangle
            C[i, j] = S[i, j]

    # Adjust: Q_T' = Q_T + P^T @ C
    Q_T_new = Q_T + P.T @ C   # (2n, n)
    return Q_T_new


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _ext_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Return ``(g, x, y)`` with ``g = gcd(|a|, |b|) ≥ 0`` and ``a*x + b*y = g``."""
    if b == 0:
        if a >= 0:
            return a, 1, 0
        return -a, -1, 0
    g, x, y = _ext_gcd(b, a % b)
    return g, y, x - (a // b) * y


def apply_cusp_basis_change(
    nz_data: NeumannZagierData,
    cusp_idx: int,
    P: int,
    Q: int,
) -> NeumannZagierData:
    """Apply a symplectic basis change at one cusp.

    Replaces the cusp basis ``(M_k, L_k/2)`` by the new basis

        new_position  =  P · M_k  +  2Q · (L_k/2)   =  P·M + Q·L
        new_momentum  =  a · M_k  +  b  · (L_k/2)

    where the integers ``(a, b)`` satisfy the Bézout identity ``P·b − 2Q·a = 1``
    (found via the extended Euclidean algorithm).  This ensures that the pair
    ``(new_position, new_momentum)`` is again a symplectic basis:

        { new_position, new_momentum } = Pb − 2Qa = 1.

    The affine shifts are updated consistently:

        nu_x_new[k]  =  P · nu_x[k]  +  2Q · nu_p[k]   (always an integer)
        nu_p_new[k]  =  a · nu_x[k]  +  b  · nu_p[k]   (may be half-integer)

    Parameters
    ----------
    nz_data : NeumannZagierData
    cusp_idx : int
        Zero-based cusp index ``k`` (must satisfy ``0 ≤ k < nz_data.r``).
    P, Q : int
        Slope defining the new position cycle ``P·M + Q·L``.  The pair must
        be primitive (``gcd(|P|, |Q|) = 1``) and ``P`` must be **odd**
        (equivalently, ``gcd(P, 2Q) = 1``) so that an integer Bézout solution
        exists.

    Returns
    -------
    NeumannZagierData
        A new ``NeumannZagierData`` with rows ``k`` and ``n+k`` of ``g_NZ``
        replaced and ``nu_x[k]``, ``nu_p[k]`` updated accordingly.  All other
        rows and shifts are unchanged.

    Raises
    ------
    ValueError
        If ``P`` is even (no integer Bézout solution) or ``cusp_idx`` is out
        of range.
    """
    from math import gcd as _gcd

    n = nz_data.n
    k = cusp_idx

    if not (0 <= k < nz_data.r):
        raise ValueError(
            f"apply_cusp_basis_change: cusp_idx={k} out of range [0, {nz_data.r})"
        )
    if P % 2 == 0:
        raise ValueError(
            f"apply_cusp_basis_change: P={P} is even; "
            "no integer symplectic conjugate exists (need P odd)"
        )

    # Solve P·b − 2Q·a = 1  ↔  P·b + (−2Q)·a = 1.
    # For primitive (P, Q) with P odd: gcd(P, 2Q) = 1, so a solution exists.
    g, b, a = _ext_gcd(P, -2 * Q)
    if g != 1:
        raise ValueError(
            f"apply_cusp_basis_change: gcd({P}, {-2 * Q}) = {g} ≠ 1; "
            "no integer solution for P·b − 2Q·a = 1"
        )
    # _ext_gcd(P, -2Q) returns (g, x, y) with P·x + (-2Q)·y = g,
    # i.e. P·b - 2Q·a = 1  with  b = x, a = y.

    # ---- build new g_NZ ----
    g_NZ_new = nz_data.g_NZ.copy()

    old_pos = nz_data.g_NZ[k].copy()       # meridian row (integer entries)
    old_mom = nz_data.g_NZ[n + k].copy()   # longitude/2 row (may be half-integer)

    # new position row: P·M + 2Q·(L/2)
    g_NZ_new[k] = P * old_pos + 2 * Q * old_mom

    # new momentum row: a·M + b·(L/2)
    g_NZ_new[n + k] = a * old_pos + b * old_mom

    # ---- update affine shifts ----
    # nu_x_new[k] = P·nu_x[k] + 2Q·nu_p[k]
    # This is mathematically an integer: P·(int) + 2Q·(half-int) = int.
    nu_x_float = nz_data.nu_x.astype(float)
    nu_x_float[k] = P * float(nz_data.nu_x[k]) + 2.0 * Q * float(nz_data.nu_p[k])
    nu_x_new = np.round(nu_x_float).astype(int)

    nu_p_new = nz_data.nu_p.copy()
    nu_p_new[k] = a * float(nz_data.nu_x[k]) + b * float(nz_data.nu_p[k])

    return NeumannZagierData(
        g_NZ=g_NZ_new,
        nu_x=nu_x_new,
        nu_p=nu_p_new,
        n=nz_data.n,
        r=nz_data.r,
        num_hard=nz_data.num_hard,
        num_easy=nz_data.num_easy,
    )


def apply_general_cusp_basis_change(
    nz_data: NeumannZagierData,
    cusp_idx: int,
    a: int,
    b: int,
    c: int,
    d: int,
) -> NeumannZagierData:
    """Apply a *general* SL(2,ℤ) basis change at one cusp.

    The SL(2,ℤ) matrix ``[[a, b], [c, d]]`` (with ``a·d − b·c = 1``)
    acts on the peripheral curves ``(μ, λ)`` as

        new_μ = a·μ + b·λ
        new_λ = c·μ + d·λ

    In the NZ convention ``(M, L/2)`` where ``M ↔ μ`` and ``L ↔ λ``,
    the transformed rows are:

        new_M   =  a · M  +  2b · (L/2)
        new_L/2 = (c/2) · M  +  d · (L/2)

    The symplectic pairing is preserved:

        {new_M, new_L/2} = a·d − b·c = 1.

    Unlike :func:`apply_cusp_basis_change`, this function does **not**
    require ``a`` to be odd; the resulting ``new_L/2`` row may acquire
    half-integer entries (which the NZ data already supports).

    Parameters
    ----------
    nz_data : NeumannZagierData
    cusp_idx : int
        Zero-based cusp index.
    a, b, c, d : int
        Entries of the SL(2,ℤ) matrix ``[[a, b], [c, d]]``, satisfying
        ``a·d − b·c = 1``.

    Returns
    -------
    NeumannZagierData
        New NZ data with the cusp-k rows and affine shifts updated.

    Raises
    ------
    ValueError
        If the matrix determinant is not 1 or cusp_idx is out of range.
    """
    n = nz_data.n
    k = cusp_idx

    if not (0 <= k < nz_data.r):
        raise ValueError(
            f"apply_general_cusp_basis_change: cusp_idx={k} "
            f"out of range [0, {nz_data.r})"
        )

    det = a * d - b * c
    if det != 1:
        raise ValueError(
            f"apply_general_cusp_basis_change: "
            f"det [[{a},{b}],[{c},{d}]] = {det} ≠ 1"
        )

    # ---- build new g_NZ ----
    g_NZ_new = nz_data.g_NZ.copy()

    old_pos = nz_data.g_NZ[k].copy()       # M_k  (integer entries)
    old_mom = nz_data.g_NZ[n + k].copy()   # L_k/2 (may be half-integer)

    # new_M   =  a · M  +  2b · (L/2)     → always integer
    g_NZ_new[k] = a * old_pos + 2 * b * old_mom

    # new_L/2 = (c/2) · M  +  d · (L/2)   → entries in Z/2
    g_NZ_new[n + k] = (c / 2) * old_pos + d * old_mom

    # ---- update affine shifts ----
    nu_x_float = nz_data.nu_x.astype(float)
    nu_x_float[k] = a * float(nz_data.nu_x[k]) + 2 * b * float(nz_data.nu_p[k])
    nu_x_new = np.round(nu_x_float).astype(int)

    nu_p_new = nz_data.nu_p.copy()
    nu_p_new[k] = (c / 2) * float(nz_data.nu_x[k]) + d * float(nz_data.nu_p[k])

    return NeumannZagierData(
        g_NZ=g_NZ_new,
        nu_x=nu_x_new,
        nu_p=nu_p_new,
        n=nz_data.n,
        r=nz_data.r,
        num_hard=nz_data.num_hard,
        num_easy=nz_data.num_easy,
    )


def build_neumann_zagier(
    data: ManifoldData,
    easy_result: EasyEdgeResult,
    reduced: ReducedGluingData | None = None,
) -> NeumannZagierData:
    """
    Construct the Neumann-Zagier symplectic matrix and affine shift.

    Parameters
    ----------
    data : ManifoldData
        Manifold data (from ``load_manifold``).
    easy_result : EasyEdgeResult
        Easy-edge search result (from ``find_easy_edges``).
    reduced : ReducedGluingData, optional
        Pre-computed reduced gluing data.  If None it is recomputed.

    Returns
    -------
    NeumannZagierData
    """
    from manifold_index.core.gluing_equations import reduce_gluing_equations

    n = data.num_tetrahedra
    r = data.num_cusps

    if reduced is None:
        reduced = reduce_gluing_equations(data)

    omega = _build_omega_block(n)

    # ------------------------------------------------------------------
    # 1. Build the n × 2n position block P
    #    Rows:  [meridians (r) | hard edges (d_hard) | easy edges (d_easy)]
    # ------------------------------------------------------------------

    # --- Meridian rows ---
    P_meridian = np.zeros((r, 2 * n), dtype=int)
    nu_meridian = np.zeros(r, dtype=int)
    for k in range(r):
        merid_row = data.gluing_matrix[n + 2 * k]
        c, cb = _reduce_to_block(merid_row, n)
        P_meridian[k] = cb
        nu_meridian[k] = c

    # --- Internal edge rows  (hard first, then easy, to match EasyEdgeResult.basis_edges) ---
    # EasyEdgeResult.basis_edges = [easy … | hard …]   (easy first per phase_space.py)
    # We want ordering:  hard edges first, then easy edges in g_NZ.
    basis = easy_result.basis_edges   # length n-r:  [easy..., hard...]
    num_easy = easy_result.num_independent_easy
    num_hard = len(easy_result.hard_padding)

    # Reorder: hard first, easy last (to match the row description in the NZ paper)
    hard_edges = easy_result.hard_padding                         # num_hard 3n-vectors
    easy_edges = [easy_result.all_easy[i]
                  for i in easy_result.independent_easy_indices]  # num_easy 3n-vectors
    internal_edges_ordered = hard_edges + easy_edges              # total n-r

    P_internal = np.zeros((n - r, 2 * n), dtype=int)
    nu_internal = np.zeros(n - r, dtype=int)
    for j, edge_3n in enumerate(internal_edges_ordered):
        c, cb = _reduce_to_block(edge_3n, n)
        P_internal[j] = cb
        # Internal edge equations have RHS = 2 (in this project's units where 2πi → 2).
        # The affine shift ν satisfies: g_NZ_row · v + ν = RHS, so ν = c - RHS = c - 2.
        # (Meridian rows have RHS = 0, so their ν = c.)
        nu_internal[j] = c - 2

    # Position block
    P = np.vstack([P_meridian, P_internal])   # (n, 2n)
    nu_x = np.concatenate([nu_meridian, nu_internal])  # (n,)

    # ------------------------------------------------------------------
    # 2. Build the n × 2n momentum block Q
    #    Rows 0..r-1   : longitude/2 rows (taken directly from gluing data)
    #    Rows r..n-1   : Γ rows (right-inverse of P @ ω, then corrected to be
    #                    symplectically orthogonal to the actual longitude rows)
    # ------------------------------------------------------------------

    A = P @ omega                          # (n, 2n)
    Q_T = _int_right_inverse(A)            # (2n, n)  — satisfies A @ Q_T = I_n
    Q_T = _make_isotropic(Q_T, P, omega)   # still A @ Q_T = I_n, now Q isotropic
    Q = Q_T.T                              # (n, 2n)

    # ------------------------------------------------------------------
    # 2a. Build longitude/2 rows directly from gluing data and compute ν_p.
    #
    #     g_NZ lives in Sp(2n, Q): the longitude/2 rows may have half-integer
    #     entries whenever the SnapPy longitude has odd coefficients.
    #     Q_lon and nu_p are therefore stored as float arrays.
    #
    #     For Γ rows (k ≥ r): ν_p[k] = 0 by construction.
    # ------------------------------------------------------------------
    Q_lon = np.zeros((r, 2 * n), dtype=float)
    nu_p = np.zeros(n, dtype=float)
    for k in range(r):
        long_row = data.gluing_matrix[n + 2 * k + 1]
        c_long, lb = _reduce_to_block(long_row, n)
        Q_lon[k] = lb / 2
        nu_p[k] = c_long / 2
    # nu_p[r:] remain 0 (Γ rows have zero affine shift by construction)

    # Correct each Γ row (Q[r:n]) so that it is symplectically orthogonal to
    # every actual longitude/2 row.  The right-inverse Γ rows are orthogonal
    # to the *computed* longitude columns of Q_T, but those may differ from
    # the actual L_k/2.  Adding a multiple of M_k (= P[k], the k-th meridian
    # row) to Γ_j adjusts [Γ_j, L_k/2] without changing [E_i, Γ_j] = δ_ij
    # (since [E_i, M_k] = 0) or the mutual isotropicity of the Γ rows
    # (since [Γ_j, M_k] = 0 from the existing symplectic structure).
    Q = Q.astype(float)
    for j in range(r, n):
        for k in range(r):
            pairing = Q[j] @ omega @ Q_lon[k]    # [Γ_j, L_k/2]
            Q[j] -= pairing * P[k]               # subtract pairing × M_k

    # Replace the first r rows of Q with the actual longitude/2 rows.
    Q[0:r] = Q_lon

    # ------------------------------------------------------------------
    # 3. Assemble g_NZ
    # ------------------------------------------------------------------
    g_NZ = np.vstack([P, Q])

    return NeumannZagierData(
        g_NZ=g_NZ,
        nu_x=nu_x,
        nu_p=nu_p,
        n=n,
        r=r,
        num_hard=num_hard,
        num_easy=num_easy,
    )
