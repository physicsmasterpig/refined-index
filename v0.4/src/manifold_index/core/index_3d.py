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

import os
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
    return Fraction(half_sum + max(0, m, -e), 2)  # max is INSIDE the ½ per eq.(12)


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
    return half_sum + max(0, m, -e)  # max is INSIDE the ½ → 2δ = half_sum + max


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
    """Enumerate ALL δ ∈ {0, 1}^{n-r} candidate half-integer patterns.

    When e_int = e0 + δ/2 (e0 integer, δ ∈ {0,1}^{n-r}), the integrality
    of g_NZ⁻¹ κ depends on the FULL κ vector — including cusp charges
    (m_ext, e_ext) — not just the internal-edge columns.  In particular,
    when g_NZ⁻¹ has half-integer entries in cusp columns (common when the
    SnapPy longitude has odd coefficients), a half-integer e_ext can flip
    the parity so that a δ pattern rejected by the internal-only check
    becomes valid (the cusp and internal half-integer parts cancel).

    Therefore this function returns ALL 2^{n-r} patterns.  The runtime
    integrality check in ``_enumerate_with_state`` (``base_args_x2S % S2``)
    performs the exact combined check for each (m_ext, e_ext, δ) triple.

    Parameters
    ----------
    g_NZ_inv : np.ndarray, shape (2n, 2n), dtype int or Fraction
    n, r : int

    Returns
    -------
    list of np.ndarray, each shape (n-r,) dtype int
        All 2^{n-r} δ patterns (including all-zeros = integer summation).
    """
    n_int = n - r
    if n_int == 0:
        return [np.array([], dtype=int)]

    patterns: list[np.ndarray] = []
    for bits in range(2 ** n_int):
        delta = np.array([(bits >> j) & 1 for j in range(n_int)], dtype=int)
        patterns.append(delta)
    return patterns


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
    _HARD_MAX = 4 * q_bound + 50  # safety cap

    for sign in (1, -1):
        prev_val = F_1d(0)
        consec_over = 0
        ever_hit = False  # did F ever reach at or below q_bound?
        for t in range(1, _HARD_MAX + 1):
            val = F_1d(sign * t)
            if val <= q_bound:
                max_abs = max(max_abs, t)
                consec_over = 0
                ever_hit = True
            else:
                consec_over += 1
                if ever_hit:
                    # Past minimum, climbing back up: stop after 2 non-decreasing steps.
                    if val >= prev_val and consec_over >= 2:
                        break
                else:
                    # Never hit q_bound yet. Two sub-cases:
                    # (a) F is strictly increasing: minimum is behind us, min > q_bound.
                    #     One step of strict increase suffices to confirm (convexity).
                    if val > prev_val:
                        break
                    # (b) F is still decreasing toward a potential minimum.
                    #     Use a linear extrapolation to estimate whether the minimum
                    #     can possibly reach q_bound within HARD_MAX steps.
                    # If the step size is so small that we'd need > HARD_MAX steps
                    # to reach q_bound, skip immediately — min > q_bound.
                    step = prev_val - val  # amount of decrease per step (>= 0)
                    if step > 0:
                        steps_to_q = (val - q_bound + step - 1) // step  # ceiling
                        if steps_to_q > _HARD_MAX - t:
                            break  # min definitely > q_bound within scan range
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
            deg_x2 = int(np.sum(half_sum + np.maximum(np.maximum(m_arr, -e_arr), 0)))
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
                deg_x2 += hs + max(m, -e_a, 0)
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
    # Hard cap on total box volume to avoid pathological manifolds where
    # degenerate NZ columns produce R[j] >> q_bound (e.g. s900 with R[3]=210).
    # If the bounding box would exceed _MAX_BOX_SIZE, tighten each R[j]
    # proportionally.  Candidates outside the tightened box are discarded —
    # this is safe because F_x2 grows quadratically away from its minimum,
    # so extremely large e0 values cannot satisfy F_x2 ≤ q_bound in practice.
    _MAX_BOX_SIZE = 50_000_000  # 50M: ~30 MB per batch, traversable in < 30s
    R = np.zeros(num_easy, dtype=int)
    for j in range(num_easy):
        def _G_j(t: int, _j: int = j) -> int:
            return _proj_min_fixed(F_x2, _j, t, num_easy, q_bound_x2)
        R[j] = _axis_scan_bound(_G_j, q_bound_x2)

    # Clamp R so that box_size ≤ _MAX_BOX_SIZE: reduce the largest R[j] first.
    box_size_raw = int(np.prod(2 * R + 1))
    if box_size_raw > _MAX_BOX_SIZE:
        # Reduce each R[j] to R_cap = floor((_MAX_BOX_SIZE^(1/d) - 1) / 2)
        R_cap = max(1, int(_MAX_BOX_SIZE ** (1.0 / num_easy) - 1) // 2)
        R = np.minimum(R, R_cap)

    # --- Step 2: enumerate bounding box and filter ---
    # Flat-index batching: iterate over the box in fixed-size chunks of
    # _NP_BOX_BATCH rows.  Each chunk is generated from a contiguous slice
    # of the flat (C-order) index [0, box_size), decoded into per-dimension
    # coordinates using modulo/divide — no meshgrid, no large temporaries.
    #
    # Peak allocation per batch:
    #   e0_batch  : B × num_easy × 8 bytes
    #   args_b    : 2n × B × 8 bytes  (from easy_cols @ e0_batch.T)
    #   ~3 intermediate (2n × B) arrays
    # = B × (num_easy + 5·2n) × 8  ≤  200k × (5 + 60) × 8  ≈  104 MB   (n=6)
    # This bound holds regardless of box dimensionality or R magnitude.
    _NP_BOX_THRESH = 64
    _NP_BOX_BATCH  = 200_000

    box_size = 1
    for j in range(num_easy):
        box_size *= 2 * int(R[j]) + 1

    ranges_np = [np.arange(-int(R[j]), int(R[j]) + 1, dtype=np.int64)
                 for j in range(num_easy)]

    # Pre-compute per-dimension sizes and C-order strides for flat→coord decoding
    sizes = np.array([len(r) for r in ranges_np], dtype=np.int64)

    def _eval_batch(e0_batch: np.ndarray) -> list[np.ndarray]:
        """Evaluate F_x2 for a (B, num_easy) int64 batch, return valid rows."""
        # RSS guard: macOS ignores RLIMIT_AS for mmap allocations, so we poll
        # resource usage directly.  Raises MemoryError → caught by worker,
        # reported as FAILED rather than killing the whole system.
        _rss_limit = os.environ.get("_IREF_RSS_LIMIT_GB", "0")
        if _rss_limit and _rss_limit != "0":
            import resource as _res
            _rss = _res.getrusage(_res.RUSAGE_SELF).ru_maxrss
            # macOS returns bytes in ru_maxrss
            _rss_gb = _rss / (1024 ** 3)
            if _rss_gb > float(_rss_limit):
                raise MemoryError(
                    f"Worker RSS {_rss_gb:.1f} GB exceeds limit "
                    f"{_rss_limit} GB — aborting chunk"
                )
        args_b = base_args[:, np.newaxis] + easy_cols @ e0_batch.T  # (2n, B)
        m_b = args_b[:n]; e_b = args_b[n:]; me_b = m_b + e_b
        half_sums = (
            np.maximum(m_b, 0) * np.maximum(me_b, 0)
            + np.maximum(-m_b, 0) * np.maximum(e_b, 0)
            + np.maximum(-e_b, 0) * np.maximum(-me_b, 0)
        )
        deg_x2 = (half_sums + np.maximum(np.maximum(m_b, -e_b), 0)).sum(axis=0)
        phase_shift = (-2 * (nu_x_easy @ e0_batch.T)).astype(np.int64)
        F_b = deg_x2 + phase_base_x2 + phase_shift
        valid = e0_batch[F_b <= q_bound_x2]
        return [valid[i] for i in range(len(valid))]

    def _flat_to_e0(flat: np.ndarray) -> np.ndarray:
        """Decode flat C-order indices into (B, num_easy) coordinate array."""
        e0 = np.empty((len(flat), num_easy), dtype=np.int64)
        rem = flat.copy()
        for j in range(num_easy - 1, -1, -1):
            e0[:, j] = ranges_np[j][rem % sizes[j]]
            rem //= sizes[j]
        return e0

    if box_size <= _NP_BOX_THRESH:
        # Tiny box — Python loop is fastest (no numpy call overhead).
        result: list[np.ndarray] = []
        for e0_tuple in iproduct(*ranges_np):
            e0 = np.array(e0_tuple, dtype=np.int64)
            if F_x2(e0) <= q_bound_x2:
                result.append(e0)
        return result

    # All other sizes: flat-index batches, capped at _NP_BOX_BATCH rows each.
    # box_size can be arbitrarily large (e.g. s900 has ~4.7 billion); the loop
    # processes it in ~200k-row slices without ever allocating the full box.
    result = []
    for start in range(0, box_size, _NP_BOX_BATCH):
        flat = np.arange(start, min(start + _NP_BOX_BATCH, box_size), dtype=np.int64)
        result.extend(_eval_batch(_flat_to_e0(flat)))
    return result


def has_valid_summation_terms(
    nz_data: NeumannZagierData,
    m_ext: Sequence[int],
    e_ext: Sequence[int | Fraction],
) -> bool:
    """Fast O(1) check: does I(m_ext, e_ext) have *any* non-zero terms?

    Returns False iff all half-integer patterns δ fail the integrality check,
    i.e. the NZ-matrix constraints produce non-integer internal charges for every
    pattern.  This is a pure linear-algebra check — no enumeration, no degree
    computation, ~1 μs per call.

    Use this as a cheap pre-filter before calling enumerate_summation_terms, to
    distinguish integrality zeros (structurally zero index) from degree zeros
    (index non-zero but above budget).
    """
    state = _get_enum_state(nz_data)
    m_arr = np.array(m_ext, dtype=np.int64)
    e_arr = np.array([int(Fraction(v) * 2) for v in e_ext], dtype=np.int64)
    me_contrib = (state.cusp_m_cols_xS @ (2 * m_arr)
                  + state.cusp_e_cols_xS @ e_arr)
    S2 = 2 * state.S
    for pat_idx in range(len(state.patterns)):
        base_x2S = me_contrib + state.delta_contrib_x2S[pat_idx]
        if np.all(base_x2S % S2 == 0):
            return True
    return False


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
        "e_int"      : list[Fraction]      — e_int as Fraction objects
        "phase_exp"  : int                — integer exponent of (−q^{1/2})
        "tet_args"   : list[(int, int)]   — (m_a, e_a) for a = 0 … n−1
        "min_degree" : Fraction           — Σ tet_degree values
    """
    state = _get_enum_state(nz_data)
    return _enumerate_with_state(state, m_ext, e_ext, q_order_half)


# ---------------------------------------------------------------------------
# Cached enumeration state — computed once per manifold, reused across
# all (m_ext, e_ext) evaluations.
# ---------------------------------------------------------------------------

@dataclass
class _EnumerationState:
    """Pre-computed manifold-dependent state for enumerate_summation_terms.

    Everything here depends only on the NZ data (manifold + basis choice),
    NOT on the per-evaluation (m_ext, e_ext) values.  Creating this once
    and reusing it across thousands of evaluations eliminates repeated
    Fraction construction, matrix inversion, and pattern enumeration.
    """
    n: int
    r: int
    n_int: int  # = n - r
    S: int                      # LCD of g_NZ_inv entries (inv_denom)
    # g_NZ_inv × S as int64 — avoids ALL Fraction arithmetic
    g_inv_xS: np.ndarray        # (2n, 2n) int64
    # Internal-edge columns of g_inv (integer-valued, exact)
    int_cols_int: np.ndarray    # (2n, n_int) int64
    # Affine shifts
    nu_x_int: np.ndarray        # (n_int,) int64 — nu_x[r:n] (easy edges)
    nu_x_full: np.ndarray       # (n,) int64 — full nu_x
    nu_p_x2: np.ndarray         # (n,) int64 — 2 * nu_p (exact)
    # Valid half-integer patterns
    patterns: list[np.ndarray]  # each shape (n_int,) int
    # Per-pattern pre-computed data:
    #   delta_contrib_x2S[i] = g_inv_xS @ kappa_delta_x2
    #     Scale: 2S × (g_inv @ kappa_delta)
    delta_contrib_x2S: list[np.ndarray]  # each (2n,) int64
    #   delta_phase_x2[i] = -delta @ nu_x[r:n]   (2 × phase_delta)
    delta_phase_x2: list[int]
    # Cusp columns of g_inv_xS for quick (m,e)-dependent matmul
    cusp_m_cols_xS: np.ndarray  # (2n, r) int64 — g_inv_xS[:, :r]
    cusp_e_cols_xS: np.ndarray  # (2n, r) int64 — g_inv_xS[:, n:n+r]


# Module-level cache: one _EnumerationState per NeumannZagierData content.
# We use a content-based key (matrix bytes) rather than ``id(nz_data)``
# because Python may reuse the same memory address for a different object
# after the original is garbage-collected (e.g. across NC cycles in Dehn
# filling, each of which creates a fresh basis-changed NZ object).
_enum_state_cache: dict[tuple, _EnumerationState] = {}


def clear_enum_state_cache() -> None:
    """Clear the cached enumeration states (call when switching manifolds)."""
    _enum_state_cache.clear()


def _nz_content_key(nz_data: NeumannZagierData) -> tuple:
    """Return a hashable, content-based fingerprint of *nz_data*."""
    return (
        nz_data.g_NZ.data.tobytes(),
        nz_data.nu_x.data.tobytes(),
        nz_data.nu_p.data.tobytes(),
    )


def _get_enum_state(nz_data: NeumannZagierData) -> _EnumerationState:
    """Get or create the cached enumeration state for *nz_data*."""
    key = _nz_content_key(nz_data)
    cached = _enum_state_cache.get(key)
    if cached is not None:
        return cached

    n, r = nz_data.n, nz_data.r
    n_int = n - r

    S, g_inv_xS = nz_data.g_NZ_inv_scaled()  # (int, (2n,2n) int64)

    # Internal-edge columns of g_inv — these are ALWAYS integer-valued.
    int_cols_xS = g_inv_xS[:, n + r: 2 * n]  # (2n, n_int) int64
    # Verify all entries are divisible by S (i.e. the original was integer)
    assert np.all(int_cols_xS % S == 0), \
        f"Internal-edge columns of g_inv are not integer; xS/S lossy (S={S})"
    int_cols_int = int_cols_xS // S  # (2n, n_int) int64, exact

    # Affine shifts
    nu_x_int = nz_data.nu_x[r: n].astype(np.int64)  # easy-edge part
    nu_x_full = nz_data.nu_x.astype(np.int64)
    nu_p_x2 = np.round(2.0 * nz_data.nu_p).astype(np.int64)

    # Valid half-integer patterns — uses the Fraction g_inv (cached on nz_data)
    g_inv_frac = nz_data.g_NZ_inv()
    patterns = valid_half_integer_patterns(g_inv_frac, n, r)

    # Per-pattern pre-computation
    delta_contrib_x2S_list: list[np.ndarray] = []
    delta_phase_x2_list: list[int] = []
    for delta in patterns:
        # kappa_delta_x2: only positions n+r..2n-1 have delta values, rest 0
        kappa_delta_x2 = np.zeros(2 * n, dtype=np.int64)
        kappa_delta_x2[n + r: 2 * n] = delta
        contrib_x2S = g_inv_xS @ kappa_delta_x2  # (2n,) int64, scale = 2S
        delta_contrib_x2S_list.append(contrib_x2S)
        # Phase from delta:  phase_delta = -(delta/2) · nu_x[r:n]
        # 2 * phase_delta = -delta · nu_x[r:n]
        delta_phase_x2_list.append(-int(delta @ nu_x_int))

    state = _EnumerationState(
        n=n, r=r, n_int=n_int,
        S=S,
        g_inv_xS=g_inv_xS,
        int_cols_int=int_cols_int,
        nu_x_int=nu_x_int,
        nu_x_full=nu_x_full,
        nu_p_x2=nu_p_x2,
        patterns=patterns,
        delta_contrib_x2S=delta_contrib_x2S_list,
        delta_phase_x2=delta_phase_x2_list,
        cusp_m_cols_xS=g_inv_xS[:, :r].copy(),
        cusp_e_cols_xS=g_inv_xS[:, n: n + r].copy(),
    )
    _enum_state_cache[key] = state
    return state


def _enumerate_with_state(
    state: _EnumerationState,
    m_ext: Sequence[int],
    e_ext: Sequence[int | Fraction],
    q_order_half: int,
) -> list[dict]:
    """Core enumeration using pre-computed state — no Fraction arithmetic."""
    n, r, n_int = state.n, state.r, state.n_int

    assert len(m_ext) == r, f"m_ext length {len(m_ext)} ≠ r={r}"
    assert len(e_ext) == r, f"e_ext length {len(e_ext)} ≠ r={r}"

    # Build the (m,e)-dependent contribution to base_args (×2S representation).
    # kappa_me_x2 has: positions 0..r-1 = 2*m_ext, n..n+r-1 = 2*e_ext, rest 0
    m_arr = np.array(m_ext, dtype=np.int64)
    e_arr = np.array([int(Fraction(v) * 2) for v in e_ext], dtype=np.int64)  # 2*e_ext
    S2 = 2 * state.S  # combined scale factor
    # me_contrib_x2S = g_inv_xS @ kappa_me_x2
    #                = sum_k cusp_m_cols_xS[:, k] * (2*m_ext[k])
    #                + sum_k cusp_e_cols_xS[:, k] * (2*e_ext[k])
    me_contrib_x2S = (state.cusp_m_cols_xS @ (2 * m_arr)
                      + state.cusp_e_cols_xS @ e_arr)

    # Phase from (m,e): phase_me = m_ext · nu_p[:r] - e_ext · nu_x[:r]
    # 2 * phase_me = m_ext · nu_p_x2[:r] - (2*e_ext) · nu_x[:r]
    phase_me_x2 = int(m_arr @ state.nu_p_x2[:r] - e_arr @ state.nu_x_full[:r])

    terms: list[dict] = []

    for pat_idx, delta in enumerate(state.patterns):
        # base_args_x2S = me_contrib_x2S + delta_contrib_x2S
        base_args_x2S = me_contrib_x2S + state.delta_contrib_x2S[pat_idx]

        # Integrality check: base_args = base_args_x2S / (2S) must be integer
        if np.any(base_args_x2S % S2 != 0):
            continue
        base_args = base_args_x2S // S2  # (2n,) int64

        # Phase base (×2): phase_base_x2 = phase_me_x2 + delta_phase_x2
        phase_base_x2 = phase_me_x2 + state.delta_phase_x2[pat_idx]

        # Find exact e0 candidates via convex degree bound.
        candidates = _exact_e0_candidates(
            base_args, state.int_cols_int, state.nu_x_int,
            phase_base_x2, q_order_half, n, n_int
        )

        # phase_base = phase_base_x2 / 2 (must be integer)
        assert phase_base_x2 % 2 == 0, \
            f"phase_base_x2={phase_base_x2} is odd; structural assumption violated"
        phase_base = phase_base_x2 // 2

        for e0 in candidates:
            args = base_args + state.int_cols_int @ e0  # (2n,) int64
            args_list = args.tolist()

            # Compute effective degree using int arithmetic (×2)
            min_deg_x2 = 0
            for a in range(n):
                min_deg_x2 += _tet_degree_x2(args_list[a], args_list[n + a])
            min_deg = Fraction(min_deg_x2, 2)

            phase_exp = phase_base - int(state.nu_x_int @ e0)

            if min_deg + phase_exp > q_order_half:
                continue  # shouldn't happen (already filtered), but guard

            tet_args = [(args_list[a], args_list[n + a]) for a in range(n)]

            # Build e_int as Fraction objects
            e_int_fracs: list[Fraction] = []
            for j in range(n_int):
                val = Fraction(int(e0[j])) + delta[j] * Fraction(1, 2)
                e_int_fracs.append(val)

            terms.append({
                "e_int":     e_int_fracs,
                "phase_exp": int(phase_exp),
                "tet_args":  tet_args,
                "min_degree": min_deg,
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
