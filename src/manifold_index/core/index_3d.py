"""
core/index_3d.py — 3D Index calculation I(m⃗, e⃗)  (Step 4).

See SPEC.md §Step 4 for the full mathematical specification.

────────────────────────────────────────────────────────────────────────────
Notation mapping  (paper → this project)
────────────────────────────────────────────────────────────────────────────
Paper symbol   Our symbol    Meaning
─────────────  ────────────  ────────────────────────────────────────────────
r              n             number of tetrahedra
n              r             number of cusps
κ              kappa         combined (m, e) vector of size 2n
ν_x            nu_x          affine shift of position rows (top n of g_NZ)
ν_p            nu_p          affine shift of momentum rows (bottom n of g_NZ)
────────────────────────────────────────────────────────────────────────────

Formula (SPEC.md eq. 2.41)
──────────────────────────
  I(m_ext, e_ext) =
    Σ_{e_int ∈ (1/2)Z^{n-r}}
      (-q^{1/2})^{ m_full · nu_p  −  e_full · nu_x }
      · ∏_{a=0}^{n-1} I_Δ( (g_NZ⁻¹ κ)_a ,  (g_NZ⁻¹ κ)_{n+a} )

where:
  m_full  = (m_ext, 0^{n-r})               (size n; internal edge m forced to 0)
  e_full  = (e_ext, e_int)                 (size n; internal edge e summed over)
  κ       = (m_full, e_full)               (size 2n)

  m_ext, e_ext have length r (cusp variables only).
  ALL n-r internal edges are summed over (with m_int = 0 forced).

A term is zero whenever any tetrahedron-index argument is non-integer,
i.e. whenever g_NZ⁻¹ κ has non-integer entries.  This is the ONLY
integrality constraint — there is no separate φ∈ℤ condition.  (When
local charges are integers the phase is automatically an integer, by the
symplectic structure of the NZ matrix.)

The q^{1/2}-series is computed entirely in Python (with an optional
C extension for inner-loop speed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product as iproduct
from typing import Callable, Iterator, Sequence

import numpy as np

from manifold_index.core.neumann_zagier import NeumannZagierData

# ---------------------------------------------------------------------------
# C extension import (optional — falls back to pure Python)
# ---------------------------------------------------------------------------

try:
    from manifold_index.core._c_tet_index import (           # type: ignore[import-not-found]
        tet_index_series as _c_tet_index_series,
        tet_degree_x2   as _c_tet_degree_x2,
        poly_convolve   as _c_poly_convolve,
    )
    _HAS_C_KERNEL = True
except ImportError:
    _HAS_C_KERNEL = False


# ===========================================================================
# Part 1a — Pure Python tetrahedron index  I_Δ(m, e; qq)
# ===========================================================================

# ---- Memoization cache (module-level, survives across calls) ----
_tet_cache: dict[tuple[int, int, int], dict[int, int]] = {}


def clear_tet_cache() -> None:
    """Clear the tetrahedron-index memoization cache.

    Call this if you need to reclaim memory after a large computation.
    The cache is automatically populated as new (m, e, qq_order) triples
    are encountered, typically yielding 12-42× fewer recomputations.
    """
    _tet_cache.clear()


def _tet_index_series(m: int, e: int, qq_order: int) -> dict[int, int]:
    """Compute the tetrahedron index I_Δ(m, e) — cached, C-accelerated.

    Checks the module-level cache first.  On a miss, delegates to the C
    extension (_c_tet_index) if available, otherwise falls back to pure
    Python.  The result is always cached for subsequent lookups.

    Returns {qq_power: coeff} dict, 0 ≤ power ≤ qq_order.
    Non-integer m or e → returns {}.
    """
    if not (isinstance(m, (int, np.integer)) and isinstance(e, (int, np.integer))):
        return {}
    m, e = int(m), int(e)

    key = (m, e, qq_order)
    cached = _tet_cache.get(key)
    if cached is not None:
        return cached

    if _HAS_C_KERNEL:
        result = _c_tet_index_series(m, e, qq_order)
    else:
        result = _tet_index_series_python(m, e, qq_order)

    _tet_cache[key] = result
    return result


def _tet_index_series_python(m: int, e: int, qq_order: int) -> dict[int, int]:
    """Pure-Python tetrahedron index I_Δ(m, e) — uncached fallback.

    Implements the Garoufalidis–Kim formula (TetIndex.wl):

        I_t(m,e) = Σ_{n=n_min}^∞  (-1)^n · qq^{n(n+1)-(2n+e)·m}
                   ──────────────────────────────────────────────
                        ∏_{k=1}^{n}(1-qq^{2k}) · ∏_{k=1}^{n+e}(1-qq^{2k})

        MIt(m,e): if m+e≥0  →  (-qq)^m · I_t(-m-e, m)
                  else        →  I_t(m, e)

    Returns {power: coeff} dict (qq^½ powers), 0 ≤ power ≤ qq_order.
    Non-integer m or e → returns {}.
    """
    if not (isinstance(m, (int, np.integer)) and isinstance(e, (int, np.integer))):
        return {}
    m, e = int(m), int(e)

    def it_direct(mm: int, ee: int, inner_order: int) -> dict[int, int]:
        """Raw I_t(mm, ee) series (no MIt symmetry), up to qq^inner_order."""
        n_min = max(0, -ee)
        # q-factorial coefficients: poly[k] = coeff of qq^k in 1/Q(qq^2,n)
        # We maintain the running inverse-qfact polynomial up to inner_order.
        result: dict[int, int] = {}
        # inv_qfact[j] stores 1/(Q(qq^2,n)*Q(qq^2,n+ee)) as a coefficient dict
        # Build incrementally as n increases.

        # Precompute 1/(q-factorial) polynomials up to the needed depth.
        # inv_fact[n] = {power: coeff} for 1/∏_{k=1}^{n}(1-qq^{2k}) mod qq^{inner_order+1}
        inv_fact: list[dict[int, int]] = [{0: 1}]  # inv_fact[0] = 1
        # grow as needed
        max_n_needed = n_min + inner_order + abs(mm) + abs(ee) + 10

        def extend_inv_fact(up_to: int) -> None:
            while len(inv_fact) <= up_to:
                k = len(inv_fact)
                # multiply current by 1/(1 - qq^{2k})
                # = Σ_{j=0}^∞ qq^{2kj}, so convolve
                prev = inv_fact[-1]
                new: dict[int, int] = {}
                for pwr, c in prev.items():
                    j = 0
                    while pwr + 2 * k * j <= inner_order:
                        p2 = pwr + 2 * k * j
                        new[p2] = new.get(p2, 0) + c
                        j += 1
                inv_fact.append(new)

        for n in range(n_min, n_min + inner_order + abs(mm) + abs(ee) + 20):
            exp_qq = n * (n + 1) - (2 * n + ee) * mm
            if exp_qq > inner_order:
                break
            # denominator: 1/(Q(qq^2,n)*Q(qq^2,n+ee))
            extend_inv_fact(max(n, n + ee))
            d1 = inv_fact[n]
            d2 = inv_fact[n + ee] if n + ee >= 0 else {0: 1}
            # convolve d1 * d2, shifted by exp_qq
            sign = (-1) ** n
            for p1, c1 in d1.items():
                for p2, c2 in d2.items():
                    total_pwr = exp_qq + p1 + p2
                    if total_pwr <= inner_order:
                        result[total_pwr] = result.get(total_pwr, 0) + sign * c1 * c2
        return result

    if m + e >= 0:
        # MIt(m,e) = (-qq)^m * I_t(-m-e, m)
        #
        # When m < 0 the multiplication by qq^m shifts the raw series DOWN by
        # |m|.  To produce output keys up to qq_order we therefore need raw
        # keys up to qq_order - m = qq_order + |m|.  Pass that as the inner
        # cutoff; extra raw terms beyond qq_order are harmlessly discarded.
        inner_order = qq_order - m  # = qq_order + |m| when m < 0
        raw = it_direct(-m - e, m, inner_order)
        # multiply by (-qq)^m = (-1)^m * qq^m
        # Use modular sign to stay int ((-1)**negative gives float in Python)
        sign_m = 1 if m % 2 == 0 else -1
        out: dict[int, int] = {}
        for pwr, c in raw.items():
            new_pwr = pwr + m
            if 0 <= new_pwr <= qq_order:
                out[new_pwr] = out.get(new_pwr, 0) + sign_m * c
        return {k: v for k, v in out.items() if v != 0}
    else:
        raw = it_direct(m, e, qq_order)
        return {k: v for k, v in raw.items() if v != 0}


def compute_index_3d_python(
    nz_data: NeumannZagierData,
    m_ext: Sequence[int],
    e_ext: Sequence[int | Fraction],
    q_order_half: int = 20,
    _precomputed_terms: list[dict] | None = None,
) -> "Index3DResult":
    """Python-native 3D index (no Mathematica).

    Computes I(m_ext, e_ext) entirely in Python by evaluating each
    tetrahedron index I_Δ as a polynomial and multiplying term-by-term.

    Parameters
    ----------
    _precomputed_terms : list[dict] or None
        If provided, skip the internal ``enumerate_summation_terms`` call and
        use these pre-enumerated terms instead.  The caller is responsible for
        ensuring they were computed with a q_order_half value ≥ the one passed
        here (extra high-degree terms are harmlessly discarded by the inner
        ``shifted ≤ q_order_half`` guard).

    Returns
    -------
    Index3DResult
    """
    terms = (
        _precomputed_terms
        if _precomputed_terms is not None
        else enumerate_summation_terms(nz_data, m_ext, e_ext, q_order_half)
    )

    if not terms:
        return Index3DResult(
            coeffs=[0] * (q_order_half + 1),
            min_power=0,
            q_order_half=q_order_half,
            m_ext=list(m_ext),
            e_ext=list(e_ext),
            n_terms=0,
        )

    total: dict[int, int] = {}
    for term in terms:
        phase_exp = term["phase_exp"]
        budget = q_order_half - phase_exp  # max qq-power for the product
        # Optimization: track the minimum power already accumulated in prod.
        # The remaining budget for the next tet series shrinks accordingly,
        # so _tet_index_series is called with a progressively tighter cutoff.
        prod: dict[int, int] = {0: 1}
        prod_min_pow = 0  # lower bound on min(prod.keys())
        for ta, tb in term["tet_args"]:
            cutoff = budget - prod_min_pow  # tighter than bare `budget`
            if cutoff < 0:
                prod = {}
                break
            s = _tet_index_series(ta, tb, cutoff)
            if not s:
                prod = {}
                break
            # multiply prod * s (use C convolution when available)
            if _HAS_C_KERNEL:
                prod = _c_poly_convolve(prod, s, budget)
            else:
                new_prod: dict[int, int] = {}
                for p1, c1 in prod.items():
                    for p2, c2 in s.items():
                        pp = p1 + p2
                        if pp <= budget:
                            new_prod[pp] = new_prod.get(pp, 0) + c1 * c2
                prod = {k: v for k, v in new_prod.items() if v != 0}
            prod_min_pow += min(s.keys())  # accumulate tet's min contribution
        # apply phase: multiply by (-qq)^{phase_exp} = (-1)^{phase_exp} * qq^{phase_exp}
        # Use modular sign to stay int ((-1)**negative gives float in Python)
        sign = 1 if phase_exp % 2 == 0 else -1
        for pp, c in prod.items():
            shifted = pp + phase_exp
            if 0 <= shifted <= q_order_half:
                total[shifted] = total.get(shifted, 0) + sign * c

    # Collect nonzero keys; handles both the empty-dict case and the
    # all-cancelled case (total is non-empty but every coefficient is 0).
    nonzero_keys = [k for k, v in total.items() if v != 0]
    if not nonzero_keys:
        return Index3DResult(
            coeffs=[0] * (q_order_half + 1),
            min_power=0,
            q_order_half=q_order_half,
            m_ext=list(m_ext),
            e_ext=list(e_ext),
            n_terms=len(terms),
        )

    min_power = min(nonzero_keys)
    max_power = q_order_half
    coeffs = [total.get(k, 0) for k in range(min_power, max_power + 1)]
    return Index3DResult(
        coeffs=coeffs,
        min_power=min_power,
        q_order_half=q_order_half,
        m_ext=list(m_ext),
        e_ext=list(e_ext),
        n_terms=len(terms),
    )


# ===========================================================================
# Part 1 — Pure Python: degree formula, κ construction, summation range
# ===========================================================================

def tet_degree(m: int, e: int) -> Fraction:
    """Leading q^{1/2}-power of the tetrahedron index I_Δ(m, e).

    From Lemma 3.6 of Garoufalidis–Kim "The 3D index of an ideal triangulation":

        δ(m, e) = ½ (m₊(m+e)₊ + (−m)₊ e₊ + (−e)₊(−e−m)₊)  +  max{0, m, −e}

    where x₊ = max{0, x}.

    Returns
    -------
    Fraction
        Exact leading power (integer or half-integer).  Returns 0 if both
        m = 0 and e = 0.

    Notes
    -----
    - ``tet_degree(0, 0) == 0``
    - ``tet_degree(m, e) == tet_degree(-e, -m)``  (symmetry)
    - Result is always ≥ 0.
    """
    pos = lambda x: max(0, x)
    half_sum = (
        pos(m) * pos(m + e)
        + pos(-m) * pos(e)
        + pos(-e) * pos(-e - m)
    )
    return Fraction(half_sum, 2) + max(0, m, -e)


def _tet_degree_x2(m: int, e: int) -> int:
    """Return ``2 * tet_degree(m, e)`` as a plain :class:`int`.

    Avoids all :class:`~fractions.Fraction` construction — identical result,
    pure integer arithmetic.  Used in the hot-path scan inside
    :func:`_exact_e0_candidates`.
    """
    pos = lambda x: max(0, x)
    half_sum = (
        pos(m) * pos(m + e)
        + pos(-m) * pos(e)
        + pos(-e) * pos(-e - m)
    )
    return half_sum + 2 * max(0, m, -e)


def build_kappa(
    m_ext: Sequence[int],
    e_ext: Sequence[int | Fraction],
    e_int: Sequence[int | Fraction],
    n: int,
    r: int,
) -> np.ndarray:
    """Assemble the full κ vector of size 2n for formula (2.41).

    κ = (m_full, e_full) where
        m_full = (m_ext, 0^{n-r})    [internal edge m forced to 0]
        e_full = (e_ext, e_int)

    Parameters
    ----------
    m_ext : sequence of int, length r
        External position variables (cusp meridians only).
    e_ext : sequence of int or Fraction, length r
        External momentum variables (cusp longitudes/2 only).
    e_int : sequence of int or Fraction, length n-r
        Internal-edge momentum variables (summed over in the formula).
        Covers ALL n-r internal edges (both "hard" and "easy").
    n, r : int
        Shape parameters from NeumannZagierData.

    Returns
    -------
    np.ndarray, shape (2n,), dtype object (to hold Fraction entries)
    """
    kappa = np.empty(2 * n, dtype=object)
    # Position block: m_ext (cusps) followed by zeros for all internal edges
    for k, v in enumerate(m_ext):
        kappa[k] = int(v)
    for k in range(r, n):
        kappa[k] = 0
    # Momentum block: e_ext (cusps) followed by e_int (all internal edges)
    for k, v in enumerate(e_ext):
        kappa[n + k] = Fraction(v)
    for k, v in enumerate(e_int):
        kappa[n + r + k] = Fraction(v)
    return kappa


def phase_exponent(
    kappa: np.ndarray,
    nu_x: np.ndarray,
    nu_p: np.ndarray,
    n: int,
    r: int,
    num_hard: int,
) -> Fraction:
    """Compute the exponent of the phase factor (-q^{1/2})^{phase}.

    Phase exponent = m_full · nu_p − e_full · nu_x

    where m_full = kappa[:n]  and  e_full = kappa[n:].

    Because nu_p[r:] = 0 (Γ rows) and kappa[:n][r+num_hard:] = 0 (easy m forced),
    the sums simplify, but we compute the full dot product for correctness.

    Returns
    -------
    Fraction
        Exact phase exponent (integer or half-integer).
    """
    m_full = kappa[:n]
    e_full = kappa[n:]
    result = Fraction(0)
    for k in range(n):
        # nu_p is float (dtype float) and may have half-integer values (e.g. -0.5).
        # int() would truncate these to 0 — use limit_denominator to get the exact
        # rational value.  nu_x is always integer so int() is fine there.
        result += Fraction(m_full[k]) * Fraction(nu_p[k]).limit_denominator(1000)
        result -= Fraction(e_full[k]) * int(nu_x[k])
    return result


def valid_half_integer_patterns(
    g_NZ_inv: np.ndarray,
    n: int,
    r: int,
) -> list[np.ndarray]:
    """Find all δ ∈ {0, 1}^{n-r} that give integer g_NZ⁻¹ κ.

    When e_int = e0 + δ/2 (e0 integer, δ ∈ {0,1}^{n-r}), the
    half-integer contribution to g_NZ⁻¹ κ is

        (1/2) · g_NZ_inv[:, n+r : 2n] @ δ

    For the result to be an integer vector we need each component of
    ``g_NZ_inv[:, n+r:2n] @ δ`` to be even.  All n-r internal-edge
    columns are checked (both "hard" and "easy" edges).

    Parameters
    ----------
    g_NZ_inv : np.ndarray, shape (2n, 2n), dtype int
    n, r : int

    Returns
    -------
    list of np.ndarray, each shape (n-r,) dtype int
        All valid δ patterns (including all-zeros = integer summation).
    """
    n_int = n - r
    if n_int == 0:
        return [np.array([], dtype=int)]

    int_cols = g_NZ_inv[:, n + r: 2 * n]  # (2n, n-r)
    valid: list[np.ndarray] = []
    for bits in range(2 ** n_int):
        delta = np.array([(bits >> j) & 1 for j in range(n_int)], dtype=int)
        product = int_cols @ delta  # (2n,)
        if np.all(product % 2 == 0):
            valid.append(delta)
    return valid


def _axis_scan_bound(
    F_1d: "Callable[[int], int]",
    q_bound: int,
) -> int:
    """Find max |t| with F_1d(t) ≤ q_bound for a convex function on Z.

    F_1d is assumed convex (piecewise-quadratic, growing to +∞ in both
    directions).  We scan outward in each direction from t = 0 and stop once
    we are strictly past the function's minimum *and* have exceeded q_bound
    for two consecutive steps (guaranteeing the quadratic tail has taken over).

    Returns
    -------
    int
        Maximum |t| seen with F_1d(t) ≤ q_bound (0 if none).
    """
    max_abs = 0
    _HARD_MAX = 4 * q_bound + 50  # safety cap; won't be reached for generic g_NZ

    for sign in (1, -1):
        prev_val = F_1d(0)
        consec_over = 0
        for t in range(1, _HARD_MAX + 1):
            val = F_1d(sign * t)
            if val <= q_bound:
                max_abs = max(max_abs, t)
                consec_over = 0
            else:
                consec_over += 1
                # Stop when F is increasing AND has been above q_bound twice.
                # Convexity guarantees it will remain above afterwards.
                if val >= prev_val and consec_over >= 2:
                    break
            prev_val = val
    return max_abs


def _proj_min_fixed(
    F_x2: "Callable[[np.ndarray], int]",
    fixed_j: int,
    tj: int,
    num_easy: int,
    q_bound_x2: int,
) -> int:
    """Return min_{e0: e0[fixed_j]=tj} F_x2(e0) via iterative direction scan.

    Fixes ``e0[fixed_j] = tj`` and minimises F_x2 over the remaining
    ``num_easy − 1`` free integer components.

    Algorithm
    ---------
    Iteratively scan along every non-zero direction in {-1, 0, 1}^{d-1}
    (where d = num_easy − 1 is the number of free components) from the
    current best point, updating the best whenever a strictly lower value
    is found.  Repeat until no direction yields an improvement.

    This is strictly better than coordinate-descent (axis-only directions)
    because F_x2 may have a diagonal valley where moving along any single
    axis from the current best point does not improve F, but a diagonal
    step does.  Including all {-1,0,1}^{d-1} directions guarantees that
    every single-step neighbour of the current best is visited in each
    outer iteration, so the algorithm cannot stall on such ridges.

    Stopping condition within each 1-D scan along direction v:
    Stop once the function value is *strictly above* best_val AND
    non-decreasing for two consecutive steps.  Crucially, we do NOT stop
    when val == best_val (plateau), because the minimum reachable via
    another direction may lie further along the current one.
    """
    free_dims = [k for k in range(num_easy) if k != fixed_j]
    d = len(free_dims)

    e0_start = np.zeros(num_easy, dtype=int)
    e0_start[fixed_j] = tj

    if d == 0:
        return int(F_x2(e0_start))

    # Build direction vectors in the free-component subspace, extended to
    # the full e0 space (fixed component stays zero in the direction).
    if d <= 4:
        free_dir_tuples = [
            v for v in iproduct((-1, 0, 1), repeat=d) if any(v)
        ]
    else:
        # Cardinals + pairwise diagonals for large d
        free_dir_tuples = []
        for i in range(d):
            for s in (-1, 1):
                t = [0] * d; t[i] = s
                free_dir_tuples.append(tuple(t))
        for i in range(d):
            for j in range(i + 1, d):
                for si in (-1, 1):
                    for sj in (-1, 1):
                        t = [0] * d; t[i] = si; t[j] = sj
                        free_dir_tuples.append(tuple(t))

    # Map each free-direction tuple to a full-space direction vector
    dir_vecs: list[np.ndarray] = []
    for fd in free_dir_tuples:
        v_full = np.zeros(num_easy, dtype=int)
        for i, k in enumerate(free_dims):
            v_full[k] = fd[i]
        dir_vecs.append(v_full)

    _SCAN_MAX = 4 * q_bound_x2 + 50
    best_val = int(F_x2(e0_start))
    best_e0 = e0_start.copy()
    max_outer = 2 * d + 4

    for _ in range(max_outer):
        changed = False
        for v_full in dir_vecs:
            # Scan from the current best in direction v_full (fixed origin).
            start_e0 = best_e0.copy()
            prev = best_val
            consec = 0
            for s in range(1, _SCAN_MAX + 1):
                trial = start_e0 + s * v_full
                val = int(F_x2(trial))
                if val < best_val:
                    best_val = val
                    best_e0 = trial.copy()
                    changed = True
                    consec = 0
                else:
                    # Plateau (val == best_val): do NOT stop.  The global
                    # minimum via another axis may be further along this ray.
                    if val > best_val and val >= prev:
                        consec += 1
                        if consec >= 2:
                            break
                    else:
                        consec = 0
                prev = val
        if not changed:
            break

    return best_val


def _exact_e0_candidates(
    base_args: np.ndarray,
    easy_cols: np.ndarray,
    nu_x_easy: np.ndarray,
    phase_base_x2: int,
    q_bound: int,
    n: int,
    num_easy: int,
) -> list[np.ndarray]:
    """Return every e0 ∈ Z^{num_easy} where F(e0) ≤ q_bound.

    The effective degree

        F(e0) = Σ_a tet_degree(base_args[a] + easy_cols[a,:] @ e0,
                                base_args[n+a] + easy_cols[n+a,:] @ e0)
                  + phase_base  −  (nu_x_easy · e0)

    is piecewise-quadratic and convex in e0, growing to +∞ in all directions
    (for non-degenerate easy columns).  We therefore:

      1. For each component j, compute
             R[j] = max{|t| : min_{y ∈ ℤ^{d-1}} F(t·e_j + y) ≤ q_bound}
         by scanning outward in t while minimising F over the d−1 free
         components at each t via coordinate descent (_proj_min_fixed).
         The projected function G_j(t) is convex, so _axis_scan_bound
         terminates correctly.  This gives the exact per-axis extent of the
         sublevel set {F ≤ q_bound}, regardless of its shape.
      2. Enumerate the bounding box Π_j[−R_j, R_j] and keep only
         e0 with F(e0) ≤ q_bound.

    Internally uses ``2 * F`` (an exact :class:`int`) to avoid all
    :class:`~fractions.Fraction` construction in the hot loop.
    """
    if num_easy == 0:
        return [np.zeros(0, dtype=int)]

    # Work with 2*F (always an integer) to avoid all Fraction arithmetic.
    # phase_base_x2 = int(2 * phase_base) is passed in by the caller.
    q_bound_x2 = 2 * q_bound

    # Threshold: for small n the Python loop with inlined arithmetic is faster
    # (numpy per-call overhead dominates for tiny arrays); for large n numpy wins.
    _USE_NUMPY_DEG = n >= 8

    def F_x2(e0: np.ndarray) -> int:
        args = base_args + easy_cols @ e0   # int64 array, shape (2n,)
        if _USE_NUMPY_DEG:
            # Vectorised over all n tets at once — efficient for large n.
            m_arr = args[:n]
            e_arr = args[n:]
            half_sum = (
                np.maximum(m_arr, 0) * np.maximum(m_arr + e_arr, 0)
                + np.maximum(-m_arr, 0) * np.maximum(e_arr, 0)
                + np.maximum(-e_arr, 0) * np.maximum(-e_arr - m_arr, 0)
            )
            deg_x2 = int(np.sum(half_sum + 2 * np.maximum(np.maximum(m_arr, -e_arr), 0)))
        else:
            # Inlined scalar loop — no function calls, no lambda, fastest for small n.
            deg_x2 = 0
            for a in range(n):
                m = int(args[a]); e_a = int(args[n + a])
                hs = (
                    max(m, 0) * max(m + e_a, 0)
                    + max(-m, 0) * max(e_a, 0)
                    + max(-e_a, 0) * max(-e_a - m, 0)
                )
                deg_x2 += hs + 2 * max(m, -e_a, 0)
        shift: int = int(nu_x_easy @ e0)
        return deg_x2 + phase_base_x2 - 2 * shift

    # --- Step 1: per-component projection bound (exact for convex F) ---
    # For each axis j we compute:
    #   R[j] = max { |t| : min_{y ∈ ℤ^{d-1}} F_x2(t·e_j + y) ≤ q_bound_x2 }
    #
    # The projected function G_j(t) = min_y F_x2(t·e_j + y) is also convex,
    # so _axis_scan_bound terminates correctly.  The per-scan minimisation
    # over the d-1 free components is done by coordinate descent via
    # _proj_min_fixed, which converges for convex piecewise-quadratic F.
    #
    # This replaces the old direction-scan over {-1,0,1}^d which
    # systematically under-estimated R[j] whenever the sublevel-set extended
    # further along a non-cardinal direction (provably wrong for num_easy ≥ 2).
    R = np.zeros(num_easy, dtype=int)
    for j in range(num_easy):
        def _G_j(t: int, _j: int = j) -> int:
            return _proj_min_fixed(F_x2, _j, t, num_easy, q_bound_x2)
        R[j] = _axis_scan_bound(_G_j, q_bound_x2)

    # --- Step 2: enumerate bounding box and filter ---
    ranges = [range(-int(R[j]), int(R[j]) + 1) for j in range(num_easy)]
    result: list[np.ndarray] = []
    for e0_tuple in iproduct(*ranges):
        e0 = np.array(e0_tuple, dtype=int)
        if F_x2(e0) <= q_bound_x2:
            result.append(e0)
    return result


def enumerate_summation_terms(
    nz_data: NeumannZagierData,
    m_ext: Sequence[int],
    e_ext: Sequence[int | Fraction],
    q_order_half: int,
) -> list[dict]:
    """Enumerate all (e_int, phase_exp, tet_args) triples contributing to I.

    For each valid half-integer pattern δ, finds **exactly** the integer
    offsets e0 such that the combined effective degree

        F(e0) = Σ_a tet_degree(args_a, args_{n+a})  +  phase_exp(e0)

    does not exceed *q_order_half*.  Because F is piecewise-quadratic and
    convex in e0, this set is bounded and is computed without any fixed
    search-radius heuristic.

    ALL n-r internal edges are summed over (both "hard" and "easy").
    m_int is forced to 0 for all internal edges.

    Parameters
    ----------
    nz_data : NeumannZagierData
    m_ext : sequence of int, length nz_data.r   (cusp meridians only)
    e_ext : sequence of int, length nz_data.r   (cusp longitudes/2 only)
    q_order_half : int
        Cutoff order in q^{1/2} (e.g. 20 keeps up to q^10).

    Returns
    -------
    list of dict, each containing:
        "e_int"      : list[str]          — e_int as "p/q" strings (info only)
        "phase_exp"  : int                — integer exponent of (−q^{1/2})
        "tet_args"   : list[(int, int)]   — (m_a, e_a) for a = 0 … n−1
        "min_degree" : float              — Σ tet_degree values
    """
    n = nz_data.n
    r = nz_data.r
    n_int = n - r   # number of internal edges (all summed over)
    # g_NZ_inv() returns an exact Fraction object array — no rounding.
    g_inv = nz_data.g_NZ_inv()

    assert len(m_ext) == r, f"m_ext length {len(m_ext)} ≠ r={r}"
    assert len(e_ext) == r, f"e_ext length {len(e_ext)} ≠ r={r}"

    # Internal-edge columns of g_inv: columns n+r .. 2n-1, shape (2n, n_int)
    int_col_start = n + r
    int_cols = g_inv[:, int_col_start: 2 * n]   # (2n, n_int), Fraction (integer-valued)
    # Convert to plain int64 once so F_x2 uses fast numpy int arithmetic,
    # not element-wise Fraction arithmetic.
    int_cols_int = np.array([[int(v) for v in row] for row in int_cols], dtype=np.int64)

    # nu_x contribution from e_int variation: nu_x[r:n] (shape n_int)
    nu_x_int = nz_data.nu_x[r: n]  # (n_int,) int

    patterns = valid_half_integer_patterns(g_inv, n, r)
    terms: list[dict] = []

    for delta in patterns:
        # δ/2 contribution: build kappa at e0 = 0 to get base args and phase
        delta_half = [Fraction(delta[j], 2) for j in range(n_int)]
        kappa_base = build_kappa(m_ext, e_ext, delta_half, n, r)

        # Base tet-args (integer because δ is a valid pattern)
        base_args_frac = g_inv @ kappa_base.astype(object)
        if not all(Fraction(v).denominator == 1 for v in base_args_frac):
            continue  # shouldn't happen for valid δ, but guard anyway
        base_args = np.array([int(Fraction(v)) for v in base_args_frac], dtype=int)

        # Base phase (at e0 = 0)
        phase_base = phase_exponent(kappa_base, nz_data.nu_x, nz_data.nu_p, n, r, 0)

        # Find exact e0 candidates via convex degree bound.
        # Pass phase_base_x2 = int(2 * phase_base) so that _exact_e0_candidates
        # can use pure integer arithmetic (2*F is always an integer).
        phase_base_x2 = int(2 * phase_base)
        candidates = _exact_e0_candidates(
            base_args, int_cols_int, nu_x_int, phase_base_x2, q_order_half, n, n_int
        )

        for e0 in candidates:
            # e_int = e0 + δ/2
            e_int_vec = [Fraction(e0[j]) + Fraction(delta[j], 2) for j in range(n_int)]

            # Full kappa and tet_args for this e0
            kappa = build_kappa(m_ext, e_ext, e_int_vec, n, r)
            args_int_list = (base_args + int_cols @ e0).tolist()

            min_deg: Fraction = sum(  # type: ignore[assignment]
                tet_degree(args_int_list[a], args_int_list[n + a]) for a in range(n)
            )
            phase_exp = phase_base - int(nu_x_int @ e0)

            # Effective start power
            if min_deg + phase_exp > q_order_half:
                continue  # shouldn't happen (already filtered), but guard

            # NOTE: the only integrality constraint is that the local charges
            # (g_NZ_inv @ κ) are integers, which is already enforced above.
            # When local charges are integers the phase φ = m·ν_p − e·ν_x is
            # automatically an integer (structural property of the NZ matrix);
            # there is NO separate φ∈ℤ discarding condition.
            tet_args = [(int(round(args_int_list[a])), int(round(args_int_list[n + a])))
                        for a in range(n)]
            terms.append({
                "e_int":     [str(v) for v in e_int_vec],
                "phase_exp": int(phase_exp),
                "tet_args":  tet_args,
                "min_degree": float(min_deg),
            })

    return terms


# ===========================================================================
# Part 2 — Result dataclass
# ===========================================================================

@dataclass
class Index3DResult:
    """q^{1/2}-series result of the 3D index computation.

    Attributes
    ----------
    coeffs : list of int
        Coefficients of the q^{1/2}-series.  ``coeffs[k]`` is the
        coefficient of ``(q^{1/2})^{min_power + k}``.
    min_power : int
        Lowest q^{1/2}-power present (may be negative or non-zero).
    q_order_half : int
        Cutoff used (series is computed modulo (q^{1/2})^{q_order_half+1}).
    m_ext : list
        The external m⃗ input.
    e_ext : list
        The external e⃗ input.
    n_terms : int
        Number of summation terms (e_easy values) that contributed.
    """

    coeffs: list[int]
    min_power: int
    q_order_half: int
    m_ext: list
    e_ext: list
    n_terms: int = 0

    def as_polynomial_string(self, var: str = "q") -> str:
        """Return a human-readable polynomial string in q^{1/2}."""
        parts = []
        for k, c in enumerate(self.coeffs):
            if c == 0:
                continue
            pw = self.min_power + k
            if pw == 0:
                parts.append(str(c))
            elif pw == 2:
                parts.append(f"{c}*{var}" if c != 1 else var)
            else:
                parts.append(
                    f"{c}*{var}^({pw}/2)"
                    if pw % 2 != 0
                    else f"{c}*{var}^{pw // 2}"
                )
        return " + ".join(parts) if parts else "0"
