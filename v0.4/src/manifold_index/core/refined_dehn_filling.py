"""
core/refined_dehn_filling.py — Refined Dehn filling kernel K^ref(P,Q;m,e;η).

See Appendix A of "Refined 3D index" (Chung-Gang-Kim, arXiv) for the full
mathematical specification.

────────────────────────────────────────────────────────────────────────────
Overview
────────────────────────────────────────────────────────────────────────────
The refined Dehn filling kernel is defined via a Hirzebruch-Jung continued
fraction (HJ-CF) expansion of the slope P/Q:

    P/Q = k_1 − 1/(k_2 − 1/(… − 1/k_ℓ))     (HJ-CF, eq. 2.35)

Special cases:
    Q = 0, P = ±1  →  ℓ = 2, k = [0, 0]
    |Q| = 1        →  ℓ = 1, k = [P/Q]  (unrefined K suffices)

The kernel chain (eq. A.7):

    K^ref(P,Q; m,e; η) =
        Σ_{m_1,e_1} … Σ_{m_{ℓ-1},e_{ℓ-1}}
            I_S(m,  −e  − k_1/2·m,   m_1, e_1)
          · I_S(m_1, −e_1 − k_2/2·m_1, m_2, e_2)
          · …
          · K(k_ℓ, 1; m_{ℓ-1}, e_{ℓ-1})

where K(·,1;·,·) is the *unrefined* Dehn filling kernel (see dehn_filling.py)
and I_S is the "symplectic kernel" defined below.

────────────────────────────────────────────────────────────────────────────
I_S kernel (eq. A.5, DFK.nb `is[]`)
────────────────────────────────────────────────────────────────────────────

    I_S(m1, e1, m2, e2; η) =
        (1/2)·(−1)^{m1}·(q^{m1/2} + q^{−m1/2}) · ẽI_S(m1, e1, m2, e2)
        − (1/2)·(−1)^{m1} · ẽI_S(m1, e1−1, m2, e2)
        − (1/2)·(−1)^{m1} · ẽI_S(m1, e1+1, m2, e2)

────────────────────────────────────────────────────────────────────────────
ẽI_S inner function (DFK.nb `expr8[]`)
────────────────────────────────────────────────────────────────────────────

    ẽI_S(m1, e1, m2, e2; η) =
        Σ_{e ∈ Z, t ∈ Z}  η^e
        · I_Δ(−e1 − m2/2,   −e/2 + e1 + m1/2 + t)
        · I_Δ( e1 + m2/2,   −e/2 + e2 − m2/2 + t)
        · I_Δ(−e2 − m1/2,    e2 + m1/2 + t)
        · I_Δ( e2 + m1/2,    e1 − m2/2 + t)
        · (−q^{1/2})^{−e + e1 + e2 + m1/2 − m2/2 + 2t}

Convention: qq = q^{1/2}; I_Δ returns a {qq_power: int} dict.
Non-integer first or second arguments → I_Δ returns {}, so the term is 0.

The sum over e is implicitly filtered to integer parity = (m1+m2) mod 2,
which is the only value for which all four I_Δ first-arguments are integers.

────────────────────────────────────────────────────────────────────────────
Output format
────────────────────────────────────────────────────────────────────────────
The filled refined index is a dict

    key = (q_half_power, eta_exp)  →  int coefficient

where q_half_power is the power of q^{1/2} (= qq power) and eta_exp is
the integer power of η.

Because I_S produces only even qq powers (= integer q powers) and the
ordinary 3D index also has definite qq-parity for each (m,e) pair, the
final sum may still be in q^{1/2}.  For closed manifolds it reduces to
pure integer q powers.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from manifold_index.core.dehn_filling import (
    KernelTerm,
    _ext_gcd,
    _particular_solution,
    _qseries_from_result,
    compute_filled_index,
    enumerate_kernel_terms,
    find_rs,
)
import numpy as np

from manifold_index.core.index_3d import (
    _tet_index_series,
    compute_index_3d_python,
    enumerate_summation_terms,
)
from manifold_index.core.neumann_zagier import NeumannZagierData
from manifold_index.core.refined_index import (
    RefinedIndexResult,
    compute_refined_index,
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# QEtaSeries: a q^{1/2}-series with η-Laurent polynomial coefficients.
# key = (qq_power, eta_exp)  →  Fraction coefficient
# qq_power: power of q^{1/2}
# eta_exp: integer power of η (can be negative)
QEtaSeries = dict[tuple[int, int], Fraction]

# MultiEtaSeries: a q^{1/2}-series with multiple fugacity dimensions.
# key = (qq_power, dim_1, dim_2, ...)  →  Fraction coefficient
# For ℓ=1 Dehn filling: key = (qq_power, 2W_0, ..., 2W_{k-1})
#   Same shape as RefinedIndexResult but with Fraction values.
# For ℓ≥2 Dehn filling: key = (qq_power, 2W_0, ..., 2W_{k-1}, 2V_0)
#   Appends one additional IS kernel η-variable (cusp refinement).
MultiEtaSeries = dict[tuple[int, ...], Fraction]

# ---------------------------------------------------------------------------
# Dense numpy helpers for hot-path polynomial arithmetic
# ---------------------------------------------------------------------------

# _tet_arr_cache: (m, e, qq_order) → int64 numpy array of length qq_order+1
# Dense version of _tet_index_series, cached for reuse in _etilde_is_numpy.
_tet_arr_cache: dict[tuple[int, int, int], np.ndarray] = {}

# Shared empty array (immutable sentinel) — avoids repeated allocations.
_EMPTY_ARR: np.ndarray = np.empty(0, dtype=np.int64)


def _tet_index_array(m: int, e: int, qq_order: int) -> np.ndarray:
    """Dense int64 array version of ``_tet_index_series``.

    Returns a numpy array *a* of length ``qq_order + 1`` where ``a[k]`` is
    the coefficient of ``qq^k``.  Returns the shared ``_EMPTY_ARR`` sentinel
    (length 0) when the series is identically zero within the truncation.

    The result is cached in ``_tet_arr_cache``.
    """
    key = (m, e, qq_order)
    cached = _tet_arr_cache.get(key)
    if cached is not None:
        return cached

    sparse = _tet_index_series(m, e, qq_order)
    if not sparse:
        _tet_arr_cache[key] = _EMPTY_ARR
        return _EMPTY_ARR

    arr = np.zeros(qq_order + 1, dtype=np.int64)
    for p, c in sparse.items():
        if 0 <= p <= qq_order:
            arr[p] = c
    _tet_arr_cache[key] = arr
    return arr


def _clear_tet_arr_cache() -> int:
    """Clear the dense tetrahedron-index cache.  Returns evicted count."""
    n = len(_tet_arr_cache)
    _tet_arr_cache.clear()
    return n


# ---------------------------------------------------------------------------
# Module-level caches
# ---------------------------------------------------------------------------
# _is_kernel and _etilde_is are pure functions of their (hashable) arguments.
# They are cached via @functools.lru_cache on the functions themselves.
#
# compute_refined_index depends on a NeumannZagierData object (not directly
# hashable), so we wrap it in a manual dict cache keyed by a content-based
# fingerprint of nz_data and hashable representations of the remaining args.
# This gives cross-P/Q memoisation: the I^ref grid scan for a second filling
# of the same manifold reuses previously computed values.
#
# NOTE: We previously used ``id(nz_data)`` as the cache key, but that is
# unsafe because Python may reuse the same memory address for a *different*
# NeumannZagierData object after the original is garbage-collected.
# This caused wrong results when Dehn filling with multiple NC cycles
# (each creates a distinct basis-changed NZ object in a loop).

_CACHE_MISS = object()

_iref_cache: dict[tuple, dict] = {}


def _nz_content_key(nz_data: "NeumannZagierData") -> tuple:
    """Return a hashable, content-based fingerprint of *nz_data*.

    For typical manifolds (n ≤ 10 tetrahedra) this is a fast bytes
    comparison (≤ 3200 bytes for the 2n × 2n matrix + two length-n vectors).
    """
    return (
        nz_data.g_NZ.data.tobytes(),
        nz_data.nu_x.data.tobytes(),
        nz_data.nu_p.data.tobytes(),
    )


def _cached_compute_refined_index(
    nz_data: "NeumannZagierData",
    m_ext: list[int],
    e_ext: Sequence[int | Fraction],
    q_order_half: int,
) -> "RefinedIndexResult":
    """Wrapper around ``compute_refined_index`` with memoisation.

    The cache key uses a content-based fingerprint of *nz_data* (matrix
    bytes + shift bytes) so that different basis changes are never confused.
    """
    key = (
        _nz_content_key(nz_data),
        tuple(m_ext),
        tuple(Fraction(e) for e in e_ext),
        q_order_half,
    )
    cached = _iref_cache.get(key, _CACHE_MISS)
    if cached is not _CACHE_MISS:
        return cached
    result = compute_refined_index(nz_data, m_ext, e_ext, q_order_half)
    _iref_cache[key] = result
    return result


def clear_filling_caches() -> dict[str, int]:
    """Clear all module-level filling caches.

    Call this when switching to a different manifold or to reclaim memory.

    Returns
    -------
    dict  mapping cache name → number of entries that were evicted.
    """
    stats: dict[str, int] = {}

    # _etilde_is and _is_kernel are @lru_cache — use .cache_clear()
    info_etilde = _etilde_is.cache_info()
    _etilde_is.cache_clear()
    stats["_etilde_is"] = info_etilde.currsize

    info_is = _is_kernel.cache_info()
    _is_kernel.cache_clear()
    stats["_is_kernel"] = info_is.currsize

    stats["_iref_cache"] = len(_iref_cache)
    _iref_cache.clear()

    # Dense tetrahedron-index array cache
    stats["_tet_arr_cache"] = _clear_tet_arr_cache()

    # Also clear the in-memory kernel table cache
    from manifold_index.core.kernel_cache import clear_kernel_cache
    stats["_kernel_mem_cache"] = clear_kernel_cache()

    return stats


def clear_computation_caches() -> None:
    """Clear only the hot-path computation caches.

    This flushes ``_tet_arr_cache``, ``_etilde_is``, and ``_is_kernel``
    without touching ``_iref_cache`` or the in-memory kernel table cache.
    Used by the parallel-decision pilot to ensure cold-cache timing.
    """
    _clear_tet_arr_cache()
    _etilde_is.cache_clear()
    _is_kernel.cache_clear()


# ---------------------------------------------------------------------------
# Part 1 — Hirzebruch-Jung continued fraction
# ---------------------------------------------------------------------------


def hj_continued_fraction(P: int, Q: int) -> list[int]:
    """Hirzebruch-Jung continued fraction for P/Q (shortest form).

    Returns [k_1, …, k_ℓ] such that
        P/Q = k_1 − 1/(k_2 − 1/(… − 1/k_ℓ))

    Unlike the classical HJ-CF (which forces k_i ≥ 2 via ceiling and can
    produce chains of length O(Q)), this implementation finds a shorter
    decomposition by:

    1. Trying length-2 decompositions via divisor search.
    2. Using nearest-integer rounding (instead of ceiling) for the
       general case, giving O(log Q) chain length.
    3. Falling back to the classical ceiling algorithm and returning
       whichever is shortest.

    Special cases
    -------------
    Q = 0, P ∈ {±1}  →  [0, 0]   (longitude / meridian special case)
    gcd(|P|, |Q|) must equal 1.

    Examples
    --------
    >>> hj_continued_fraction(1, 3)
    [0, -3]
    >>> hj_continued_fraction(1, 4)
    [0, -4]
    >>> hj_continued_fraction(4, 3)
    [1, -3]
    >>> hj_continued_fraction(5, 2)
    [3, 2]
    >>> hj_continued_fraction(1, 1)
    [1]
    """
    if Q == 0:
        assert abs(P) == 1, f"Q=0 but |P|={abs(P)} ≠ 1"
        return [0, 0]

    # Normalise to Q > 0
    if Q < 0:
        P, Q = -P, -Q

    # Length 1: P/Q is an integer
    if Q == 1:
        return [P]

    # Try length-2: P/Q = k1 - 1/k2, i.e. k2 = Q/(k1*Q - P).
    # Need d = k1*Q - P to be a nonzero divisor of Q, with k1 = (P+d)/Q integer.
    # Equivalently: d | Q and Q | (P + d).
    best_len2: list[int] | None = None
    best_len2_cost = float("inf")
    absQ = abs(Q)
    for i in range(1, absQ + 1):
        if absQ % i != 0:
            continue
        for d in (i, -i):
            if (P + d) % Q == 0:
                k1 = (P + d) // Q
                k2 = Q // d
                cost = abs(k1) + abs(k2)
                if cost < best_len2_cost:
                    best_len2_cost = cost
                    best_len2 = [k1, k2]

    if best_len2 is not None:
        return best_len2

    # General case: compute both ceiling-based and nearest-integer CFs,
    # return the shorter one.
    ks_ceil = _hj_cf_ceil(P, Q)
    ks_round = _hj_cf_round(P, Q)

    return ks_round if len(ks_round) < len(ks_ceil) else ks_ceil


def _hj_cf_ceil(P: int, Q: int) -> list[int]:
    """Classical HJ-CF using ceiling (k_i ≥ 2 except terminal)."""
    x = Fraction(P, Q)
    ks: list[int] = []
    while True:
        k = math.ceil(x)
        ks.append(k)
        remainder = k - x
        if remainder == 0:
            break
        x = Fraction(1, remainder)
    return ks


def _hj_cf_round(P: int, Q: int) -> list[int]:
    """HJ-CF using nearest-integer rounding (shorter chains, O(log Q))."""
    x = Fraction(P, Q)
    ks: list[int] = []
    while True:
        # Round half away from zero
        k = int(x + Fraction(1, 2)) if x >= 0 else -int(-x + Fraction(1, 2))
        ks.append(k)
        remainder = k - x
        if remainder == 0:
            break
        x = Fraction(1, remainder)
    return ks


# ---------------------------------------------------------------------------
# Part 2 — QEtaSeries arithmetic helpers
# ---------------------------------------------------------------------------


def _qeta_add(a: QEtaSeries, b: QEtaSeries) -> QEtaSeries:
    """Add two QEtaSeries (non-destructive)."""
    result: QEtaSeries = dict(a)
    for key, val in b.items():
        new_val = result.get(key, Fraction(0)) + val
        if new_val == 0:
            result.pop(key, None)
        else:
            result[key] = new_val
    return result


def _qeta_scale(s: QEtaSeries, scalar: Fraction) -> QEtaSeries:
    """Multiply all coefficients by scalar."""
    if scalar == 0:
        return {}
    return {k: v * scalar for k, v in s.items() if v * scalar != 0}


def _qeta_shift_qq(s: QEtaSeries, qq_shift: int) -> QEtaSeries:
    """Multiply by q^{qq_shift/2} (shift all qq_power keys by qq_shift)."""
    return {(qq + qq_shift, eta): v for (qq, eta), v in s.items()}


def _qeta_truncate(s: QEtaSeries, qq_order: int) -> QEtaSeries:
    """Keep only entries with qq_power ≤ qq_order."""
    return {k: v for k, v in s.items() if k[0] <= qq_order}


def _qeta_convolve(a: QEtaSeries, b: QEtaSeries, qq_order: int | None = None) -> QEtaSeries:
    """Multiply two QEtaSeries (convolve qq-powers, add η-exponents)."""
    result: QEtaSeries = {}
    for (qq1, eta1), c1 in a.items():
        for (qq2, eta2), c2 in b.items():
            new_qq = qq1 + qq2
            if qq_order is not None and new_qq > qq_order:
                continue
            key = (new_qq, eta1 + eta2)
            new_val = result.get(key, Fraction(0)) + c1 * c2
            if new_val == 0:
                result.pop(key, None)
            else:
                result[key] = new_val
    return result


def _tet_series_to_qeta(s: dict[int, int], eta_exp: int = 0) -> QEtaSeries:
    """Convert a plain qq-series dict[int,int] to QEtaSeries at fixed η power."""
    return {(qq, eta_exp): Fraction(c) for qq, c in s.items() if c != 0}


def _int_qqseries_convolve(
    a: dict[int, int], b: dict[int, int], qq_order: int | None = None
) -> dict[int, int]:
    """Convolve two plain integer qq-series dicts."""
    result: dict[int, int] = {}
    for p1, c1 in a.items():
        for p2, c2 in b.items():
            pp = p1 + p2
            if qq_order is not None and pp > qq_order:
                continue
            result[pp] = result.get(pp, 0) + c1 * c2
    return result


# ---------------------------------------------------------------------------
# Part 3 — ẽI_S kernel  (expr8 in DFK.nb)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _etilde_is(
    m1: int,
    e1: Fraction,
    m2: int,
    e2: Fraction,
    qq_order: int,
    eta_order: int,
) -> dict[tuple[int, int], int]:
    """Compute ẽI_S(m1, e1, m2, e2; η) = expr8[m1, e1, m2, e2] in DFK.nb.

    Returns a dict[(qq_power, eta_exp) → int].
    Values are always integers (verified empirically for all tested manifolds).

    Parameters
    ----------
    m1, m2 : int
        Integer cusp-meridian variables.
    e1, e2 : Fraction
        Half-integer cusp-momentum variables (in (1/2)Z).
    qq_order : int
        Truncate the series at qq^{qq_order}.
    eta_order : int
        Sum η exponent over range [−eta_order, eta_order].

    Returns
    -------
    QEtaSeries  (may be empty if integrality conditions fail)

    Notes
    -----
    The formula involves four tetrahedron indices:
        tind1 = I_Δ(−e1 − m2/2,  −e/2 + e1 + m1/2 + t)
        tind2 = I_Δ( e1 + m2/2,  −e/2 + e2 − m2/2 + t)
        tind3 = I_Δ(−e2 − m1/2,   e2 + m1/2 + t)        ← e-var-independent
        tind4 = I_Δ( e2 + m1/2,   e1 − m2/2 + t)        ← e-var-independent

    The outer integrality filters:
        m_a1 = −e1 − m2/2  must be integer   (first arg of tind1)
        m_a3 = −e2 − m1/2  must be integer   (first arg of tind3)

    When these hold, the e-sum variable must have parity (m1+m2) % 2
    for the tind1/tind2 second arguments to be integers.
    """
    # ------------------------------------------------------------------
    # Outer integrality check
    # ------------------------------------------------------------------
    m_a1_frac = -e1 - Fraction(m2, 2)   # first arg of tind1; also -m_a2
    m_a3_frac = -e2 - Fraction(m1, 2)   # first arg of tind3; also -m_a4

    if m_a1_frac.denominator != 1 or m_a3_frac.denominator != 1:
        return {}

    m_a1 = int(m_a1_frac)    # = −e1 − m2/2
    m_a2 = -m_a1              # =  e1 + m2/2
    m_a3 = int(m_a3_frac)    # = −e2 − m1/2
    m_a4 = -m_a3              # =  e2 + m1/2

    # Base e-arguments for tind3 and tind4 (before adding t):
    #   tind3 second arg = e2 + m1/2 + t  → base = m_a4
    #   tind4 second arg = e1 − m2/2 + t  → base = e1 − m2/2
    e3_base_frac = e2 + Fraction(m1, 2)    # = m_a4
    e4_base_frac = e1 - Fraction(m2, 2)
    if e3_base_frac.denominator != 1 or e4_base_frac.denominator != 1:
        return {}  # sanity (should follow from above checks)
    e3_base = int(e3_base_frac)
    e4_base = int(e4_base_frac)

    # Phase constant B = e1 + e2 + m1/2 − m2/2  (must be integer)
    B_frac = e1 + e2 + Fraction(m1, 2) - Fraction(m2, 2)
    if B_frac.denominator != 1:
        return {}
    B = int(B_frac)

    # e-var parity: for tind1/tind2 second args to be integers,
    # e_var ≡ (m1 + m2) (mod 2).
    e_var_parity = (m1 + m2) % 2

    # Pre-compute the "base" for tind1/tind2 e-arguments after factoring
    # out the n_eta (where e_var = 2*n_eta + e_var_parity):
    #   tind1 second arg = −e_var/2 + e1 + m1/2 + t
    #                    = −n_eta − p/2 + e1 + m1/2 + t
    #                    = t − n_eta + (e1 + m1/2 − p/2)
    # The parenthesised term must be integer when the parity is correct.
    e_arg1_base_frac = e1 + Fraction(m1, 2) - Fraction(e_var_parity, 2)
    e_arg2_base_frac = e2 - Fraction(m2, 2) - Fraction(e_var_parity, 2)
    if e_arg1_base_frac.denominator != 1 or e_arg2_base_frac.denominator != 1:
        return {}
    e_arg1_base = int(e_arg1_base_frac)
    e_arg2_base = int(e_arg2_base_frac)

    # ------------------------------------------------------------------
    # Main double sum: t ∈ Z,  n_eta ∈ [−eta_order, eta_order]
    # ------------------------------------------------------------------
    # Bound on t: tind3/tind4 have minimum qq-degree that grows with |t|;
    # once the minimum qq-degree of the tind3·tind4 product exceeds
    # qq_order + max X (where X = B + 2t), no further contributions land
    # within [0, qq_order].  We use a generous scan and rely on early
    # termination via empty s3/s4.
    t_range = qq_order + abs(B) + 10

    # NOTE: _etilde_is values are always integers (verified empirically).
    #
    # === NUMPY FFT-BATCHED PATH ===
    # Three levels of vectorisation over the naïve Python dict approach:
    #   1. Tetrahedron indices stored as dense int64 arrays (_tet_index_array)
    #   2. s3·s4 and s1·s2 polynomial multiplies use np.convolve (C-level)
    #   3. For each t the n_eta loop is BATCHED:
    #      - all valid s12 arrays are stacked into a matrix
    #      - a single FFT-based batch convolution with s34 produces all
    #        n_eta convolution results simultaneously
    #      This eliminates per-call Python overhead and leverages
    #      vectorised FFT (O(n log n) per row) instead of direct
    #      convolution (O(n²) per call).
    n_eta_bins = 2 * eta_order + 1
    result_2d = np.zeros((qq_order + 1, n_eta_bins), dtype=np.int64)

    # FFT length for batched convolution (power-of-2 for efficiency).
    # Maximum conv length = (qq_order+1) + (qq_order+1) - 1 = 2*qq_order+1.
    fft_len = 1
    while fft_len < 2 * qq_order + 1:
        fft_len <<= 1

    for t in range(-t_range, t_range + 1):
        e3 = e3_base + t
        e4 = e4_base + t

        # tind3 and tind4 are independent of the η sum variable
        a3 = _tet_index_array(m_a3, e3, qq_order)
        if len(a3) == 0:
            continue
        a4 = _tet_index_array(m_a4, e4, qq_order)
        if len(a4) == 0:
            continue

        # Convolve s3 · s4 using numpy
        a34_full = np.convolve(a3, a4)
        if len(a34_full) == 0:
            continue
        a34 = a34_full[: qq_order + 1]
        L34 = len(a34)

        # ── Collect all valid (n_eta, s12) for this t ──
        # s12 depends only on u = t − n_eta (through e_a1, e_a2).
        # Cache s12 by u to avoid recomputing for different (t, n_eta)
        # pairs that share the same u.
        batch_n_eta: list[int] = []
        batch_X: list[int] = []
        batch_s12: list[np.ndarray] = []
        s12_by_u: dict[int, np.ndarray | None] = {}

        for n_eta in range(-eta_order, eta_order + 1):
            u = t - n_eta
            # Check u-cache first
            if u in s12_by_u:
                a12 = s12_by_u[u]
            else:
                e_a1 = u + e_arg1_base
                e_a2 = u + e_arg2_base
                a1 = _tet_index_array(m_a1, e_a1, qq_order)
                if len(a1) == 0:
                    s12_by_u[u] = None
                    continue
                a2 = _tet_index_array(m_a2, e_a2, qq_order)
                if len(a2) == 0:
                    s12_by_u[u] = None
                    continue
                a12_full = np.convolve(a1, a2)
                a12 = a12_full[: qq_order + 1] if len(a12_full) > 0 else None
                s12_by_u[u] = a12

            if a12 is None:
                continue

            e_var = 2 * n_eta + e_var_parity
            X = -e_var + B + 2 * t  # = 2*u - e_var_parity + B

            batch_n_eta.append(n_eta)
            batch_X.append(X)
            batch_s12.append(a12)

        if not batch_s12:
            continue

        N_batch = len(batch_s12)

        # ── Batched convolution: all s12[i] ⊗ s34 in one FFT pass ──
        # For small batches, individual np.convolve is faster than FFT overhead.
        if not (N_batch >= 4 or qq_order >= 50):
            # Scalar path — direct convolution (faster for small N or small qq)
            for i in range(N_batch):
                conv = np.convolve(batch_s12[i], a34)
                X = batch_X[i]
                sign = 1 if X % 2 == 0 else -1
                eta_idx = batch_n_eta[i] + eta_order

                src_lo = max(0, -X)
                src_hi = min(len(conv), qq_order + 1 - X)
                if src_lo >= src_hi:
                    continue
                dst_lo = src_lo + X
                dst_hi = src_hi + X

                if sign == 1:
                    result_2d[dst_lo:dst_hi, eta_idx] += conv[src_lo:src_hi]
                else:
                    result_2d[dst_lo:dst_hi, eta_idx] -= conv[src_lo:src_hi]
        else:
            # FFT-batched path — stack s12's into a matrix, single FFT pass
            L_max = max(len(a) for a in batch_s12)
            conv_len = L_max + L34 - 1
            fft_n = 1
            while fft_n < conv_len:
                fft_n <<= 1

            # Stack s12 arrays (zero-padded to fft_n)
            s12_matrix = np.zeros((N_batch, fft_n), dtype=np.int64)
            for i, a12 in enumerate(batch_s12):
                s12_matrix[i, : len(a12)] = a12

            # Pad s34 to fft_n
            s34_padded = np.zeros(fft_n, dtype=np.int64)
            s34_padded[: L34] = a34

            # Batched FFT convolution (float64 intermediate)
            S12_fft = np.fft.rfft(s12_matrix.astype(np.float64), n=fft_n, axis=1)
            s34_fft = np.fft.rfft(s34_padded.astype(np.float64), n=fft_n)
            conv_matrix_f = np.fft.irfft(S12_fft * s34_fft[None, :], n=fft_n, axis=1)

            # Round to nearest integer (all true values are exact integers)
            conv_matrix = np.rint(conv_matrix_f[:, :conv_len]).astype(np.int64)

            # Scatter-add each row into result_2d
            for i in range(N_batch):
                X = batch_X[i]
                sign = 1 if X % 2 == 0 else -1
                eta_idx = batch_n_eta[i] + eta_order

                src_lo = max(0, -X)
                src_hi = min(conv_len, qq_order + 1 - X)
                if src_lo >= src_hi:
                    continue
                dst_lo = src_lo + X
                dst_hi = src_hi + X

                if sign == 1:
                    result_2d[dst_lo:dst_hi, eta_idx] += conv_matrix[i, src_lo:src_hi]
                else:
                    result_2d[dst_lo:dst_hi, eta_idx] -= conv_matrix[i, src_lo:src_hi]

    # Convert dense 2D array → sparse dict[(qq, eta) → int]
    result: dict[tuple[int, int], int] = {}
    nz_qq, nz_eta = np.nonzero(result_2d)
    for idx in range(len(nz_qq)):
        qq_p = int(nz_qq[idx])
        eta_idx = int(nz_eta[idx])
        e_var = 2 * (eta_idx - eta_order) + e_var_parity
        result[(qq_p, e_var)] = int(result_2d[qq_p, eta_idx])

    return result


# ---------------------------------------------------------------------------
# Part 4 — I_S kernel  (is[] in DFK.nb)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _is_kernel(
    m1: int,
    e1: Fraction,
    m2: int,
    e2: Fraction,
    qq_order: int,
    eta_order: int,
) -> dict[tuple[int, int], int]:
    """Compute 2 × I_S(m1, e1, m2, e2; η) — the symplectic IS kernel, ×2 scaled.

    Formula (DFK.nb `is[]`):
        I_S = (1/2)·(−1)^{m1}·(qq^{m1} + qq^{−m1}) · ẽI_S(m1, e1,   m2, e2)
            − (1/2)·(−1)^{m1} · ẽI_S(m1, e1−1, m2, e2)
            − (1/2)·(−1)^{m1} · ẽI_S(m1, e1+1, m2, e2)

    Returns **2 × I_S** as integer-valued dict to avoid Fraction arithmetic.
    Since ẽI_S returns integers and the formula has a (1/2) prefactor that
    yields half-integer values, multiplying by 2 makes all results integral.
    The accumulated LCD (least common denominator) is tracked by callers.

    Returns
    -------
    dict[(qq_power, eta_exp) → int]
        All values are 2 × the true I_S coefficient.
        Returns {} if the integrality conditions for ẽI_S fail.
    """
    ei_center = _etilde_is(m1, e1,     m2, e2, qq_order, eta_order)
    ei_minus  = _etilde_is(m1, e1 - 1, m2, e2, qq_order, eta_order)
    ei_plus   = _etilde_is(m1, e1 + 1, m2, e2, qq_order, eta_order)

    if not ei_center and not ei_minus and not ei_plus:
        return {}

    sign_m1 = 1 if m1 % 2 == 0 else -1

    # === NUMPY-ACCELERATED PATH ===
    # Build a 2D accumulator result_2d[qq, eta_idx] to replace Python dict ops.
    # Collect all eta values across the three _etilde_is results to map them
    # to column indices.
    all_etas: set[int] = set()
    for d in (ei_center, ei_minus, ei_plus):
        for (_, eta) in d:
            all_etas.add(eta)
    if not all_etas:
        return {}

    eta_list = sorted(all_etas)
    eta_to_idx = {e: i for i, e in enumerate(eta_list)}
    n_eta = len(eta_list)

    # Dense 2D accumulator: rows = qq powers [0, qq_order], cols = eta
    result_2d = np.zeros((qq_order + 1, n_eta), dtype=np.int64)

    # Term A+B: (−1)^{m1}·(qq^{m1} + qq^{−m1}) · ẽI_S(e1)   [no ½ — ×2 absorbed]
    for (qq_p, eta), c in ei_center.items():
        scaled = c * sign_m1
        if scaled == 0:
            continue
        col = eta_to_idx[eta]
        # Shift by +m1
        new_qq = qq_p + m1
        if 0 <= new_qq <= qq_order:
            result_2d[new_qq, col] += scaled
        # Shift by −m1
        new_qq = qq_p - m1
        if 0 <= new_qq <= qq_order:
            result_2d[new_qq, col] += scaled

    # Terms C+D: −(−1)^{m1} · ẽI_S(e1±1)   [no ½ — ×2 absorbed]
    neg_sign = -sign_m1
    for src_series in (ei_minus, ei_plus):
        for (qq_p, eta), c in src_series.items():
            scaled = c * neg_sign
            if scaled == 0 or not (0 <= qq_p <= qq_order):
                continue
            result_2d[qq_p, eta_to_idx[eta]] += scaled

    # Convert to sparse dict
    result: dict[tuple[int, int], int] = {}
    nz_r, nz_c = np.nonzero(result_2d)
    for idx in range(len(nz_r)):
        result[(int(nz_r[idx]), eta_list[int(nz_c[idx])])] = int(
            result_2d[nz_r[idx], nz_c[idx]]
        )

    return result


def _is_kernel_frac(
    m1: int,
    e1: Fraction,
    m2: int,
    e2: Fraction,
    qq_order: int,
    eta_order: int,
) -> QEtaSeries:
    """Fraction-valued I_S kernel wrapper, for the multi-cusp Fraction path.

    Calls ``_is_kernel`` (×2 int) and converts each value to Fraction(v, 2).
    """
    raw = _is_kernel(m1, e1, m2, e2, qq_order, eta_order)
    if not raw:
        return {}
    return {k: Fraction(v, 2) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Part 5 — Intermediate-kernel enumeration  (K(k, 1; m, e) support)
# ---------------------------------------------------------------------------


def _enumerate_slope1_terms(
    k: int,
    t_range: int,
) -> list[tuple[int, Fraction, int, int]]:
    """Enumerate (m, e, c, phase) for the unrefined kernel K(k, 1; m, e).

    Only c ∈ {0, 2} are returned (c=-2 is handled by multiplicity=2 for c=2,
    except for c=0 which uses multiplicity=2 for t≠0).

    The general families for Q=1:
        c=0:  m_t =  Q·t = t,  e_t = −k·t/2,   phase_t = t        (t ∈ Z)
        c=2:  m_t = m_c + t,   e_t = e_c − k·t/2,  phase_t = phase_c + t

    For K(k, 1; ·): R=1, S=0 always (R·Q − k·S = 1·1 − k·0 = 1).
    Phase = R·m + 2S·e = m.

    Parameters
    ----------
    k : int
        The integer slope value (unrefined kernel K(k, 1; ·)).
    t_range : int
        Scan |t| ≤ t_range.

    Returns
    -------
    list of (m, e, c, phase) tuples
        All (m, e) pairs with |t| ≤ t_range satisfying k·m + 2e ∈ {0, 2}.
    """
    terms: list[tuple[int, Fraction, int, int]] = []
    seen: set[tuple[int, Fraction]] = set()

    for c in (0, 2):
        m_c, e_c = _particular_solution(k, 1, c)
        # Phase = m (since R=1, S=0)
        phase_c0 = m_c  # = R*m_c + 2*S*e_c = 1*m_c + 0

        signs = (1,) if c == 0 else (1, -1)
        for sign in signs:
            for t_abs in range(0, t_range + 1):
                t = sign * t_abs
                if sign == -1 and t_abs == 0:
                    continue

                m_t = m_c + 1 * t   # Q=1
                e_t = e_c - Fraction(k * t, 2)
                phase_t = phase_c0 + t

                key = (m_t, e_t)
                if key not in seen:
                    seen.add(key)
                    terms.append((m_t, e_t, c, phase_t))

    return terms


def _enumerate_slope1_all_halfshift(
    k: int,
    t_range: int,
) -> list[tuple[int, Fraction, int, int]]:
    """Enumerate half-integer-shifted (m, e) pairs for IS chain intermediates.

    When k is even, :func:`_enumerate_slope1_all` produces only integer-e
    targets.  But the ẽI_S integrality condition B (``−e_target − m_src/2 ∈ ℤ``)
    requires half-integer e_target when the source m is odd.

    This function produces the same m values as ``_enumerate_slope1_all(k, t_range)``
    but with all e values shifted by +1/2.  The c and phase fields are set to
    dummy values (0, 0) since they are not used by the IS convolution step.

    These targets are only needed for intermediate IS chain steps (not the
    final K-factor application), and only when the state contains entries
    with odd m.
    """
    terms: list[tuple[int, Fraction, int, int]] = []
    seen: set[tuple[int, Fraction]] = set()

    for c in (0, 2, -2):
        m_c, e_c = _particular_solution(k, 1, c)
        e_c_shifted = e_c + Fraction(1, 2)

        for t in range(-t_range, t_range + 1):
            m_t = m_c + t
            e_t = e_c_shifted - Fraction(k * t, 2)

            key = (m_t, e_t)
            if key not in seen:
                seen.add(key)
                # c and phase are dummies — not used by _apply_is_step's inner loop
                terms.append((m_t, e_t, 0, 0))

    return terms


def _enumerate_is_full(
    m1_range: int,
    e1_range: int,
) -> list[tuple[int, Fraction, int, int]]:
    """Enumerate the full (½)ℤ² lattice for intermediate IS-chain steps.

    Unlike :func:`_enumerate_slope1_all`, this does NOT restrict to the
    K(k, 1) support (c ∈ {0, ±2}).  It returns every (m₁, e₁) with
    |m₁| ≤ *m1_range* and |e₁| ≤ *e1_range*, where e₁ ∈ (½)ℤ.

    This is required for intermediate IS steps (all steps except the last
    before the final K-factor), because the IS kernel maps sources to
    targets on a lattice determined by the integrality conditions—not
    by the K-support of the next HJ entry.

    The c and phase slots (3rd and 4th elements) are set to 0 since
    they are unused by :func:`_apply_is_step`.
    """
    terms: list[tuple[int, Fraction, int, int]] = []
    for m1 in range(-m1_range, m1_range + 1):
        for f1 in range(-2 * e1_range, 2 * e1_range + 1):
            terms.append((m1, Fraction(f1, 2), 0, 0))
    return terms


def _enumerate_slope1_all(
    k: int,
    t_range: int,
) -> list[tuple[int, Fraction, int, int]]:
    """Enumerate ALL (m, e, c, phase) for K(k, 1; m, e) — no symmetry shortcuts.

    Unlike :func:`_enumerate_slope1_terms`, this enumerates:

    - **c ∈ {−2, 0, 2}** (includes c = −2 explicitly).
    - **t ∈ [−t_range, t_range]** for *all* c values (no positive-only
      shortcut for c = 0).

    All returned terms have implicit multiplicity = 1.  This is required
    for the ℓ ≥ 2 IS-kernel chain, where the (m, e) → (−m, −e) symmetry
    of the 3D index payload no longer holds for the intermediate IS state,
    so the doubling trick used in the ℓ = 1 path is invalid.
    """
    terms: list[tuple[int, Fraction, int, int]] = []
    seen: set[tuple[int, Fraction]] = set()

    for c in (0, 2, -2):
        m_c, e_c = _particular_solution(k, 1, c)
        phase_c0 = m_c  # R·m_c + 2·S·e_c = 1·m_c + 0  (R=1, S=0 for Q=1)

        for t in range(-t_range, t_range + 1):
            m_t = m_c + t   # Q = 1
            e_t = e_c - Fraction(k * t, 2)
            phase_t = phase_c0 + t   # = m_t

            key = (m_t, e_t)
            if key not in seen:
                seen.add(key)
                terms.append((m_t, e_t, c, phase_t))

    return terms


# ---------------------------------------------------------------------------
# Part 6 — Apply unrefined K(k, 1; m1, e1) factor to a QEtaSeries
# ---------------------------------------------------------------------------


def _apply_k1_factor(
    is_series: QEtaSeries,
    m1: int,
    e1: Fraction,
    c: int,
    phase: int,
    multiplicity: int,
    qq_order: int,
) -> QEtaSeries:
    """Apply the unrefined K(k, 1; m1, e1) factor to a QEtaSeries.

    K(k, 1; m1, e1) factor (R=1, S=0):
        c=0:  (1/2)·(−1)^{phase}·(qq^{phase} + qq^{−phase})
        c=±2: −(1/2)·(−1)^{phase}

    Multiplies *is_series* by this factor, scales by *multiplicity*, and
    truncates to qq_order.
    """
    sign = Fraction(1 if phase % 2 == 0 else -1)
    half = Fraction(1, 2)
    mult = Fraction(multiplicity)

    if c == 0:
        result: QEtaSeries = {}
        scalar = half * sign * mult
        for (qq_p, eta), c_val in is_series.items():
            scaled = c_val * scalar
            if scaled == 0:
                continue
            # qq^{+phase} shift
            new_qq_a = qq_p + phase
            if new_qq_a <= qq_order:
                key = (new_qq_a, eta)
                v = result.get(key, Fraction(0)) + scaled
                if v == 0:
                    result.pop(key, None)
                else:
                    result[key] = v
            # qq^{−phase} shift
            new_qq_b = qq_p - phase
            if new_qq_b <= qq_order:
                key = (new_qq_b, eta)
                v = result.get(key, Fraction(0)) + scaled
                if v == 0:
                    result.pop(key, None)
                else:
                    result[key] = v
        return result
    else:
        # c = ±2: constant factor, no q-shift
        scalar = -half * sign * mult
        return _qeta_scale(is_series, scalar)


# ---------------------------------------------------------------------------
# Part 6b — MultiEtaSeries helpers
# ---------------------------------------------------------------------------


def _multi_add(a: MultiEtaSeries, b: MultiEtaSeries) -> MultiEtaSeries:
    """Add two MultiEtaSeries (non-destructive).

    Works with both Fraction and int values (polymorphic).
    """
    result: MultiEtaSeries = dict(a)
    for key, val in b.items():
        new_val = result.get(key, 0) + val
        if new_val == 0:
            result.pop(key, None)
        else:
            result[key] = new_val
    return result


def _multi_convolve_is(
    is_series: QEtaSeries,
    multi_series: MultiEtaSeries,
    qq_order: int | None = None,
) -> MultiEtaSeries:
    """Convolve a QEtaSeries (IS kernel) with a MultiEtaSeries.

    The IS kernel's η (cusp η) is mapped to the LAST dimension of the
    multi-key.  The qq powers are summed; inner η dimensions (hard-edge
    fugacities) are untouched.

    Parameters
    ----------
    is_series : QEtaSeries
        Keys: ``(qq_power, 2V)``
    multi_series : MultiEtaSeries
        Keys: ``(qq_power, 2W_0, …, 2W_{k-1}, 2V)``
    qq_order : int or None
        Truncation cutoff.

    Returns
    -------
    MultiEtaSeries with the same key structure as *multi_series*.
    """
    # ── Small-product fast path: plain nested loop ──
    n_is = len(is_series)
    n_multi = len(multi_series)
    if n_is * n_multi < 500:
        result: MultiEtaSeries = {}
        for (qq_is, eta_is), c_is in is_series.items():
            for multi_key, c_multi in multi_series.items():
                new_qq = qq_is + multi_key[0]
                if qq_order is not None and new_qq > qq_order:
                    continue
                new_key = (new_qq,) + multi_key[1:-1] + (multi_key[-1] + eta_is,)
                new_val = result.get(new_key, 0) + c_is * c_multi
                if new_val == 0:
                    result.pop(new_key, None)
                else:
                    result[new_key] = new_val
        return result

    # ── Grouped path: batch by rest-key to avoid redundant tuple ops ──
    # Group multi_series entries by everything EXCEPT qq_power.
    # Within each group the hard-η + cusp-η dimensions are identical,
    # so the output rest-key is computed once per (group, IS entry).
    from collections import defaultdict
    groups: dict[tuple, list[tuple[int, int | Fraction]]] = defaultdict(list)
    for multi_key, c_multi in multi_series.items():
        groups[multi_key[1:]].append((multi_key[0], c_multi))

    result = {}
    _get = result.get  # local lookup — marginal but free
    for (qq_is, eta_is), c_is in is_series.items():
        for rest, members in groups.items():
            out_rest = rest[:-1] + (rest[-1] + eta_is,)
            for qq_multi, c_multi in members:
                new_qq = qq_is + qq_multi
                if qq_order is not None and new_qq > qq_order:
                    continue
                new_key = (new_qq,) + out_rest
                new_val = _get(new_key, 0) + c_is * c_multi
                if new_val == 0:
                    result.pop(new_key, None)
                else:
                    result[new_key] = new_val
    return result


def _apply_k1_factor_multi(
    series: MultiEtaSeries,
    c: int,
    phase: int,
    multiplicity: int,
    qq_order: int,
    truncate: bool = True,
    int_mode: bool = False,
) -> MultiEtaSeries:
    """Apply unrefined K(k, 1; m, e) factor to a MultiEtaSeries.

    Identical logic to ``_apply_k1_factor`` but operates on multi-
    dimensional keys.  Only the qq_power (first element) is shifted;
    all η dimensions are untouched.

    Parameters
    ----------
    truncate : bool
        If True (default), keep only terms with ``new_qq ≤ qq_order``.
        If False, skip the bounds check — used when building
        manifold-independent kernel tables so that the deferred
        truncation happens after convolution with I^ref.
    int_mode : bool
        If True, use pure int arithmetic (×2 scaling: the ½ factor in the
        K-factor is absorbed).  Series values must be int.  The output LCD
        is 2× the input LCD.  Used in the ℓ≥2 IS chain hot path.
    """
    if int_mode:
        # ×2 mode: sign and mult are plain ints, no Fraction(½)
        sign = 1 if phase % 2 == 0 else -1
        mult = multiplicity
        if c == 0:
            scalar = sign * mult
            result: MultiEtaSeries = {}
            for key, c_val in series.items():
                scaled = c_val * scalar
                if scaled == 0:
                    continue
                qq_p = key[0]
                rest = key[1:]
                for new_qq in (qq_p + phase, qq_p - phase):
                    if not truncate or new_qq <= qq_order:
                        new_key = (new_qq,) + rest
                        v = result.get(new_key, 0) + scaled
                        if v == 0:
                            result.pop(new_key, None)
                        else:
                            result[new_key] = v
            return result
        else:
            scalar = -sign * mult
            if scalar == 0:
                return {}
            return {k: v * scalar for k, v in series.items() if v * scalar != 0}

    # Original Fraction-based path (ℓ=1, kernel_cache, etc.)
    sign = Fraction(1 if phase % 2 == 0 else -1)
    half = Fraction(1, 2)
    mult = Fraction(multiplicity)

    if c == 0:
        result: MultiEtaSeries = {}
        scalar = half * sign * mult
        for key, c_val in series.items():
            scaled = c_val * scalar
            if scaled == 0:
                continue
            qq_p = key[0]
            rest = key[1:]
            # +phase shift
            new_qq_a = qq_p + phase
            if not truncate or new_qq_a <= qq_order:
                new_key = (new_qq_a,) + rest
                v = result.get(new_key, Fraction(0)) + scaled
                if v == 0:
                    result.pop(new_key, None)
                else:
                    result[new_key] = v
            # -phase shift
            new_qq_b = qq_p - phase
            if not truncate or new_qq_b <= qq_order:
                new_key = (new_qq_b,) + rest
                v = result.get(new_key, Fraction(0)) + scaled
                if v == 0:
                    result.pop(new_key, None)
                else:
                    result[new_key] = v
        return result
    else:
        # c = ±2: constant factor, no q-shift
        scalar = -half * sign * mult
        if scalar == 0:
            return {}
        return {k: v * scalar for k, v in series.items() if v * scalar != 0}


def _apply_weyl_shift(
    refined: RefinedIndexResult,
    m_ext: list[int],
    e_ext: Sequence[int | Fraction],
    weyl_a: list[Fraction],
    weyl_b: list[Fraction],
    num_hard: int,
    cusp_idx: int = 0,
) -> RefinedIndexResult:
    """Multiply a refined index by the Weyl monomial η^{a·e_I + b·m_I}.

    This transforms I^ref(m, e; η) → f(m, e; η) = η^{a·e_I + b·m_I} · I^ref
    so that Dehn filling operates on the Weyl-manifest form.

    In the matrix Weyl model, each filled cusp *I* has its own column of
    Weyl vectors (a^{(I)}, b^{(I)}).  The shift uses **only the charges of
    the filled cusp** (not a sum over all cusps).

    The shift in the 2×-encoded key convention is:

        shift_x2[j] = 2·(a[j]·e_I + b[j]·m_I)

    where ``a[j]`` and ``b[j]`` are the physical Weyl vectors, and
    ``m_I = m_ext[cusp_idx]``, ``e_I = e_ext[cusp_idx]``.

    Parameters
    ----------
    refined : RefinedIndexResult
        Keys ``(qq_power, 2η_0, …, 2η_{k-1})``.
    m_ext, e_ext : sequences, length r (number of cusps)
        Meridian / longitude charges for this evaluation.
    weyl_a, weyl_b : list[Fraction], length num_hard
        Physical Weyl vectors for the cusp being filled.
    num_hard : int
        Number of hard edges.
    cusp_idx : int
        Index of the cusp being filled (0-based).

    Returns
    -------
    RefinedIndexResult
        Shifted copy.
    """
    m_I = m_ext[cusp_idx]
    e_I = Fraction(e_ext[cusp_idx])
    shift_x2 = [
        int(2 * (weyl_a[j] * e_I + weyl_b[j] * m_I))
        for j in range(num_hard)
    ]
    # If all shifts are zero, return original unmodified
    if all(s == 0 for s in shift_x2):
        return refined

    result: RefinedIndexResult = {}
    for key, coeff in refined.items():
        if coeff == 0:
            continue
        new_key = (key[0],) + tuple(
            key[1 + j] + shift_x2[j] for j in range(num_hard)
        )
        result[new_key] = result.get(new_key, 0) + coeff
    return result


def _refined_to_multi(
    refined: RefinedIndexResult,
    append_cusp_eta: bool = False,
    use_int: bool = False,
) -> MultiEtaSeries:
    """Convert a RefinedIndexResult to MultiEtaSeries.

    Parameters
    ----------
    refined : RefinedIndexResult
        Keys: ``(qq_power, 2W_0, …, 2W_{k-1})``
    append_cusp_eta : bool
        If True, append a ``cusp_eta = 0`` dimension to every key
        (needed for ℓ ≥ 2 before IS convolution steps).
    use_int : bool
        If True, keep values as int (from RefinedIndexResult which has int
        values).  Used in the ℓ≥2 int-mode hot path.

    Returns
    -------
    MultiEtaSeries with Fraction or int values.
    """
    result: MultiEtaSeries = {}
    for key, coeff in refined.items():
        if coeff == 0:
            continue
        new_key = key + (0,) if append_cusp_eta else key
        result[new_key] = coeff if use_int else Fraction(coeff)
    return result


# ---------------------------------------------------------------------------
# Part 7 — Single-step IS convolution
# ---------------------------------------------------------------------------


def _process_is_chunk(
    chunk: list[tuple[tuple[int, Fraction], MultiEtaSeries]],
    k_current: int,
    is_last_step: bool,
    use_int: bool,
    qq_order: int,
    eta_order: int,
    parity_data: dict,
) -> dict[tuple[int, Fraction], MultiEtaSeries]:
    """Process a chunk of source entries for ``_apply_is_step``.

    This is a **module-level** function so that it can be pickled and
    dispatched to worker processes via ``ProcessPoolExecutor``.
    It is algebraically identical to the inner loop of ``_apply_is_step``:
    each source (m, e, src_series) is independent.
    """
    _kernel_fn = _is_kernel if use_int else _is_kernel_frac
    local: dict[tuple[int, Fraction], MultiEtaSeries] = {}

    for (m, e), src_series in chunk:
        if not src_series:
            continue

        e_in = -e - Fraction(k_current * m, 2)
        e_half = int(2 * e)
        p = -(e_half + k_current * m)

        if not is_last_step:
            m_is_even = (m % 2 == 0)
            if p % 2 == 0:
                compatible_m1 = (
                    parity_data["even_eint"]
                    if m_is_even
                    else parity_data["even_ehalf"]
                )
            else:
                compatible_m1 = (
                    parity_data["odd_eint"]
                    if m_is_even
                    else parity_data["odd_ehalf"]
                )
        else:
            compatible_m1 = (
                parity_data["even"] if (p % 2 == 0) else parity_data["odd"]
            )

        for m1, e1, _, _ in compatible_m1:
            is_val = _kernel_fn(m, e_in, m1, e1, qq_order, eta_order)
            if not is_val:
                continue
            product = _multi_convolve_is(is_val, src_series, qq_order)
            if not product:
                continue
            key = (m1, e1)
            if key in local:
                local[key] = _multi_add(local[key], product)
            else:
                local[key] = product

    return local


def _apply_is_step(
    state: dict[tuple[int, Fraction], MultiEtaSeries],
    k_current: int,
    k_next: int,
    qq_order: int,
    eta_order: int,
    m1_range: int,
    use_int: bool = False,
    is_last_step: bool = True,
    n_workers: int = 1,
) -> dict[tuple[int, Fraction], MultiEtaSeries]:
    """Apply one IS convolution step to the state.

    Maps  state[(m, e)]  →  new_state[(m1, e1)]  via

        new_state[(m1, e1)] +=
            I_S(m, −e − k_current/2·m,  m1, e1; η) · state[(m, e)]

    The IS kernel's η variable maps to the LAST dimension of the
    MultiEtaSeries keys.  Hard-edge η dimensions are carried through
    unchanged.

    Parameters
    ----------
    state : dict[(int, Fraction) → MultiEtaSeries]
        Current state (source variables).
    k_current : int
        k_i from the HJ-CF (used to compute the e-transform −e − k_i/2·m).
    k_next : int
        k_{i+1} (the NEXT continued-fraction entry, i.e. the slope of the
        next kernel step); used to enumerate which (m1, e1) are relevant
        when *is_last_step* is True.
    qq_order : int
    eta_order : int
    m1_range : int
        Scan |m1| ≤ m1_range for intermediate variables.
    use_int : bool
        If True, use ``_is_kernel`` which returns ×2 int values
        for maximum performance.  Callers must track the accumulated LCD
        (×2 per IS step).  State values should be int.
        If False (default), use ``_is_kernel_frac`` which returns exact
        Fraction values.  Safe for Fraction-valued state (kernel_cache,
        multi-cusp path, etc.).
    is_last_step : bool
        If True (default), enumerate targets from the K(k_next, 1) support.
        This is correct for the last IS step before the final K-factor,
        because only those targets yield non-zero K-factor contributions.
        If False, enumerate the full (½)ℤ² lattice.  This is required for
        intermediate IS steps (ℓ ≥ 3 chains, all steps except the last),
        because the IS kernel maps to targets whose e₁ can be half-integer
        when the source m is odd—outside the K(k, 1) support for even k.
    n_workers : int
        Number of parallel processes for the IS step.  When > 1 and the
        state dict is large enough, source entries are dispatched to
        ``ProcessPoolExecutor`` workers.  Default 1 (sequential).

    Returns
    -------
    new_state : dict[(int, Fraction) → MultiEtaSeries]
    """
    new_state: dict[tuple[int, Fraction], MultiEtaSeries] = {}

    # Target enumeration depends on whether this is the last IS step.
    if is_last_step:
        # Last IS step: restrict to K(k_next, 1) support, since only
        # these targets yield a non-zero final K-factor.
        m1_terms = _enumerate_slope1_all(k_next, m1_range)
    else:
        # Intermediate IS step: enumerate the full (½)ℤ² lattice.
        # The e₁ range is bounded by the qq truncation order plus a
        # margin for the source m range.
        e1_range = qq_order + m1_range // 2
        m1_terms = _enumerate_is_full(m1_range, e1_range)

    # ── Parity pre-filter ──
    # The _etilde_is integrality check A requires:
    #   (e_in + m1_target / 2) ∈ ℤ  ⟺  (2·e_in + m1_target) is even
    # Since e_in = −e − k_current·m/2 and e = e_half/2:
    #   2·e_in = −(e_half + k_current·m)  =:  p
    # So the check reduces to (p + m1_target) % 2 == 0, i.e. m1_target
    # must have the SAME parity as p.
    #
    # For intermediate IS steps (full lattice), we additionally exploit
    # integrality check B: −e1 − m/2 ∈ ℤ  ⟹
    #   source m even  → e1 must be integer
    #   source m odd   → e1 must be half-integer
    # This gives 4-way partitioning: (m1_parity) × (e1_integrality).
    #
    # For the last IS step (K-support), e1 integrality is already
    # implicit in the K(k, 1) enumeration for even k.
    if not is_last_step:
        # 4-way partition: (m1_even/odd) × (e1_int/half)
        m1_even_eint: list[tuple[int, Fraction, int, int]] = []
        m1_even_ehalf: list[tuple[int, Fraction, int, int]] = []
        m1_odd_eint: list[tuple[int, Fraction, int, int]] = []
        m1_odd_ehalf: list[tuple[int, Fraction, int, int]] = []
        for entry in m1_terms:
            m1_val, e1_val = entry[0], entry[1]
            is_m1_even = (m1_val % 2 == 0)
            is_e1_int = (e1_val.denominator == 1)
            if is_m1_even:
                if is_e1_int:
                    m1_even_eint.append(entry)
                else:
                    m1_even_ehalf.append(entry)
            else:
                if is_e1_int:
                    m1_odd_eint.append(entry)
                else:
                    m1_odd_ehalf.append(entry)
    else:
        # 2-way partition by m1 parity only (K-support already constrains e1)
        m1_even: list[tuple[int, Fraction, int, int]] = []
        m1_odd: list[tuple[int, Fraction, int, int]] = []
        for entry in m1_terms:
            if entry[0] % 2 == 0:
                m1_even.append(entry)
            else:
                m1_odd.append(entry)

    _kernel_fn = _is_kernel if use_int else _is_kernel_frac

    # ── Parallel dispatch (ProcessPoolExecutor) ──
    # Each source (m, e) → src_series is independent; only the merge
    # into new_state requires coordination.  We dispatch chunks of
    # source entries to separate processes when the state is large
    # enough to amortise the pickling overhead.
    _MIN_ENTRIES_PER_WORKER = 4
    _use_parallel = (
        n_workers > 1
        and len(state) >= n_workers * _MIN_ENTRIES_PER_WORKER
    )
    if _use_parallel:
        if not is_last_step:
            parity_data = {
                "even_eint": m1_even_eint,
                "even_ehalf": m1_even_ehalf,
                "odd_eint": m1_odd_eint,
                "odd_ehalf": m1_odd_ehalf,
            }
        else:
            parity_data = {"even": m1_even, "odd": m1_odd}

        items = list(state.items())
        chunk_size = max(1, len(items) // n_workers)
        chunks = [
            items[i : i + chunk_size]
            for i in range(0, len(items), chunk_size)
        ]
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [
                pool.submit(
                    _process_is_chunk,
                    chunk,
                    k_current,
                    is_last_step,
                    use_int,
                    qq_order,
                    eta_order,
                    parity_data,
                )
                for chunk in chunks
            ]
            for fut in futures:
                local = fut.result()
                for key, series in local.items():
                    if key in new_state:
                        new_state[key] = _multi_add(new_state[key], series)
                    else:
                        new_state[key] = series
        return new_state

    # ── Sequential path ──
    for (m, e), src_series in state.items():
        if not src_series:
            continue

        # e-transform: first argument transforms as −e − k_current/2·m
        e_in = -e - Fraction(k_current * m, 2)

        # Select the parity-compatible m1 partition.
        # p = 2·e_in = −(2e + k_current·m).  Since e is Fraction(e_half, 2):
        # p = −(e_half + k_current·m).  We need m1 with same parity as p.
        e_half = int(2 * e)  # e is always in (1/2)Z
        p = -(e_half + k_current * m)

        if not is_last_step:
            # 4-way: select (m1_parity) × (e1_integrality) partition
            m_is_even = (m % 2 == 0)
            if p % 2 == 0:
                compatible_m1 = m1_even_eint if m_is_even else m1_even_ehalf
            else:
                compatible_m1 = m1_odd_eint if m_is_even else m1_odd_ehalf
        else:
            compatible_m1 = m1_even if (p % 2 == 0) else m1_odd

        for m1, e1, _, _ in compatible_m1:
            is_val = _kernel_fn(m, e_in, m1, e1, qq_order, eta_order)
            if not is_val:
                continue

            # Multiply I_S (QEtaSeries) · state[(m, e)] (MultiEtaSeries)
            product = _multi_convolve_is(is_val, src_series, qq_order)
            if not product:
                continue

            key = (m1, e1)
            if key in new_state:
                new_state[key] = _multi_add(new_state[key], product)
            else:
                new_state[key] = product

    return new_state


# ---------------------------------------------------------------------------
# Part 8 — Main computation: compute_filled_refined_index
# ---------------------------------------------------------------------------


@dataclass
class FilledRefinedResult:
    """Result of refined Dehn filling I^ref_{P/Q}(η^{2W}, [η^{2V}, …]).

    The series carries:
    - For ℓ=1 (|Q|=1, no IS kernel):
        key = (qq_power, 2W_0, …, 2W_{k-1})
        Only hard-edge fugacities; no cusp η.
    - For ℓ≥2 (IS kernel chain):
        key = (qq_power, 2W_0, …, 2W_{k-1}, 2V_0)
        Hard-edge fugacities + one cusp η from the IS chain.
    - For multi-cusp sequential filling (two fillings with ℓ≥2):
        key = (qq, 2W_0, …, 2W_{k-1}, 2V_0, 2V_1)
        Hard-edge fugacities + one cusp η per filling step.

    Attributes
    ----------
    P, Q : int
        Slope (physical cycle P·M + Q·L).
    cusp_idx : int
        Index of the filled cusp (or -1 for multi-cusp).
    series : MultiEtaSeries
        Multi-dimensional η-polynomial in q^{1/2}.
    qq_order : int
        Series is truncated at qq^{qq_order}.
    eta_order : int
        Maximum |cusp η exponent| to retain (0 for ℓ=1).
    hj_ks : list[int]
        Hirzebruch-Jung continued fraction coefficients [k_1, …, k_ℓ].
    n_kernel_terms : int
        Number of outer (m, e) pairs evaluated.
    num_hard : int
        Number of hard-edge η dimensions.
    has_cusp_eta : bool
        True if any cusp η dimension is present.
    num_cusp_eta : int
        Number of cusp-η dimensions (0 for ℓ=1, 1 for single ℓ≥2,
        2 for two sequential ℓ≥2 fillings, etc.).
    """

    P: int
    Q: int
    cusp_idx: int
    series: MultiEtaSeries
    qq_order: int
    eta_order: int
    hj_ks: list[int]
    n_kernel_terms: int
    num_hard: int
    has_cusp_eta: bool
    num_cusp_eta: int = 0

    @property
    def is_zero(self) -> bool:
        """True if no non-zero coefficients."""
        return len(self.series) == 0

    def collapse_eta_edges(self, edges: list[int]) -> "FilledRefinedResult":
        """Set η_j = 1 (W_j = 0) for the given hard-edge indices.

        For each edge j in *edges*, the doubled-exponent at position 1+j in
        every key is set to 0 and coefficients with identical collapsed keys
        are summed.  Returns a new ``FilledRefinedResult``.

        This is used when edge *j* is incompatible with Dehn filling
        (``a[j] ∉ ℤ``), so the refinement for that edge must be turned off.
        """
        if not edges:
            return self
        positions = [1 + j for j in edges]  # key indices to zero out
        new_series: MultiEtaSeries = {}
        for key, coeff in self.series.items():
            if coeff == 0:
                continue
            new_key = list(key)
            for pos in positions:
                new_key[pos] = 0
            new_key = tuple(new_key)
            new_series[new_key] = new_series.get(new_key, 0) + coeff
        # Remove zeros
        new_series = {k: v for k, v in new_series.items() if v != 0}
        return FilledRefinedResult(
            P=self.P, Q=self.Q, cusp_idx=self.cusp_idx,
            series=new_series, qq_order=self.qq_order,
            eta_order=self.eta_order, hj_ks=self.hj_ks,
            n_kernel_terms=self.n_kernel_terms,
            num_hard=self.num_hard, has_cusp_eta=self.has_cusp_eta,
            num_cusp_eta=self.num_cusp_eta,
        )

    def eta1_series(self) -> dict[int, Fraction]:
        """Set all η variables to 1: pure qq-series."""
        result: dict[int, Fraction] = {}
        for key, c in self.series.items():
            qq_p = key[0]
            new = result.get(qq_p, Fraction(0)) + c
            if new == 0:
                result.pop(qq_p, None)
            else:
                result[qq_p] = new
        return result

    def q_series_at_eta(self, eta_val: int = 1) -> dict[int, Fraction]:
        """Evaluate all η at a specific integer value: pure qq-series.

        For the hard-edge η's (stored as 2×exponent), the actual exponent
        is key[i]/2.  For the cusp η (if present), the exponent is key[-1].
        """
        result: dict[int, Fraction] = {}
        for key, c in self.series.items():
            qq_p = key[0]
            # Hard-edge η: exponents in key[1:1+num_hard], stored as 2×exp
            contrib = c
            for i in range(1, 1 + self.num_hard):
                half_exp = key[i]  # = 2 * η_a_exp
                # η_val^(half_exp/2) — only works for η_val = ±1
                if eta_val == 1 or half_exp == 0:
                    pass
                else:
                    contrib *= Fraction(eta_val ** (half_exp // 2))
            # Cusp η's (one per filling step)
            for ci in range(self.num_cusp_eta):
                pos = 1 + self.num_hard + ci
                if pos >= len(key):
                    break
                cusp_exp = key[pos]
                if cusp_exp != 0 and eta_val != 1:
                    contrib *= Fraction(eta_val ** cusp_exp)
            new = result.get(qq_p, Fraction(0)) + contrib
            if new == 0:
                result.pop(qq_p, None)
            else:
                result[qq_p] = new
        return result

    def as_q_eta_string(
        self,
        q_var: str = "q",
        eta_var: str = "η",
        half_pow: bool = False,
    ) -> str:
        """Produce a human-readable string representation.

        Parameters
        ----------
        q_var, eta_var : str
        half_pow : bool
            If True, write q^{k/2} for qq_powers.
        """
        if not self.series:
            return "0"
        parts: list[str] = []
        for key in sorted(self.series.keys()):
            c = self.series[key]
            if c == 0:
                continue
            qq_p = key[0]
            # Build q factor string
            if half_pow:
                q_str = (
                    "" if qq_p == 0 else
                    f"{q_var}" if qq_p == 2 else
                    f"{q_var}^({qq_p}/2)" if qq_p % 2 != 0 else
                    f"{q_var}^{qq_p // 2}"
                )
            else:
                qp = qq_p // 2
                q_str = "" if qp == 0 else f"{q_var}" if qp == 1 else f"{q_var}^{qp}"
            # Build η factor strings
            eta_parts: list[str] = []
            # Hard-edge η's
            for a in range(self.num_hard):
                exp2 = key[1 + a]  # = 2 * η_a_exp
                if exp2 == 0:
                    continue
                if exp2 == 2:
                    eta_parts.append(f"{eta_var}_{a}")
                elif exp2 == -2:
                    eta_parts.append(f"{eta_var}_{a}^(-1)")
                elif exp2 % 2 == 0:
                    eta_parts.append(f"{eta_var}_{a}^{exp2 // 2}")
                else:
                    eta_parts.append(f"{eta_var}_{a}^({exp2}/2)")
            # Cusp η's (one per filling step, after hard-edge η's)
            n_ce = self.num_cusp_eta
            for ci in range(n_ce):
                pos = 1 + self.num_hard + ci
                if pos >= len(key):
                    break
                cusp_exp = key[pos]
                label = f"{eta_var}_c" if n_ce == 1 else f"{eta_var}_{{c{ci}}}"
                if cusp_exp == 1:
                    eta_parts.append(label)
                elif cusp_exp == -1:
                    eta_parts.append(f"{label}^(-1)")
                elif cusp_exp != 0:
                    eta_parts.append(f"{label}^{cusp_exp}")
            h_str = "·".join(eta_parts)
            monomial = (q_str + ("·" + h_str if h_str else "")) or "1"
            parts.append(f"{c}*{monomial}" if c != 1 else monomial)
        return " + ".join(parts) if parts else "0"


def compute_filled_refined_index(
    nz_data: NeumannZagierData,
    cusp_idx: int,
    P: int,
    Q: int,
    m_other: Sequence[int] | None = None,
    e_other: Sequence[int | Fraction] | None = None,
    q_order_half: int = 10,
    eta_order: int | None = None,
    m1_range: int | None = None,
    weyl_a: list[Fraction] | None = None,
    weyl_b: list[Fraction] | None = None,
    verbose: bool = False,
    n_workers: int = 1,
    auto_precompute: bool = False,
    cache_iref: bool = False,
    manifold_name: str = "unknown",
) -> FilledRefinedResult:
    """Compute the refined Dehn-filled index I^ref_{P/Q}(η^{2W}, η^{2V}).

    Applies the refined Dehn filling kernel K^ref(P,Q; m,e; η^{2V}) to the
    refined 3D index I^ref(m,e; η^{2W}) summed over contributing (m,e) pairs.

    Algorithm
    ---------
    1. Compute the HJ-CF k = [k_1, …, k_ℓ] for P/Q.
    2. If ℓ = 1: K^ref = K(k_1, 1; ·) (unrefined kernel, no IS chain).
       Sum K(k_1,1; m,e) · I^ref(m,e; η^{2W}) over kernel support.
       Result has only η^{2W} (hard-edge) variables.
    3. If ℓ ≥ 2:
       a. Scan ALL (m,e) with non-zero I^ref; initialise state with
          I^ref(m,e; η^{2W}) ⊗ η^{2V·0}.
       b. Apply ℓ−1 IS convolution steps (IS kernel multiplies into cusp-η).
       c. Apply the final unrefined K(k_ℓ, 1; ·) to the last state.
       Result has η^{2W} (hard-edge) + η^{2V} (cusp) variables.
    4. Return FilledRefinedResult.

    Parameters
    ----------
    nz_data : NeumannZagierData
    cusp_idx : int
        Which cusp to fill (0-based).
    P, Q : int
        Coprime integers defining the slope **in the original (α, β) basis**
        of nz_data.  The filling cycle is P·α + Q·β.
    m_other, e_other : sequences of length r−1, optional
        Values for the remaining cusps. Defaults to all zeros.
    q_order_half : int
        Series cutoff in q^{1/2} powers (= qq_order).
    eta_order : int or None
        Maximum |cusp η exponent| to retain in IS kernels.
        Default (None): auto-set to ``q_order_half`` so that the
        η summation is bounded only by the q-order truncation.
    m1_range : int or None
        Scan range for intermediate (m_1, e_1) variables.
        Default: 2 * q_order_half.
    weyl_a, weyl_b : list[Fraction] or None, optional
        Physical Weyl-symmetry vectors from :class:`ABVectors`.  When
        provided, each I^ref(m, e) is multiplied by η^{a·e + b·m} *before*
        the Dehn filling kernel is applied, so that the filling operates on
        the Weyl-manifest form.
    verbose : bool
        Print progress to stdout.
    n_workers : int
        Number of worker processes for parallel I^ref computation
        (passed to ``apply_precomputed_kernel``).  Default 1 = sequential.
    auto_precompute : bool
        If True, automatically precompute and cache the manifold-independent
        filling kernel K^ref(P/Q) when an ℓ ≥ 2 filling is needed and no
        cached kernel exists.  This is *slower* for the first call (~22s vs
        ~8s at qq=20) but makes all subsequent calls instant (~0.6s) for
        ANY manifold at the same (P, Q, qq_order).  Recommended for
        interactive/batch workflows where the same NC-transformed slopes
        recur across manifolds.
    cache_iref : bool
        If True, persist I^ref(m,e) results to disk and reload them on
        subsequent calls.  Since I^ref is slope-independent, this makes
        the 2nd through Nth slope computations for the same manifold
        dramatically faster.
    manifold_name : str
        Human-readable label for the I^ref cache file (e.g. ``"m003"``).
        Only used when *cache_iref* is True.

    Returns
    -------
    FilledRefinedResult
    """
    r = nz_data.r
    num_hard = nz_data.num_hard
    if m_other is None:
        m_other = [0] * (r - 1)
    if e_other is None:
        e_other = [0] * (r - 1)
    assert len(m_other) == r - 1
    assert len(e_other) == r - 1

    if m1_range is None:
        m1_range = 2 * q_order_half

    # Auto-compute eta_order: the diamond truncation rule
    # (qq_power + |cusp_eta| ≤ qq_order) ensures that only terms with
    # |cusp_eta| ≤ qq_order survive in the output.  However, the IS
    # kernel computes at qq_internal (= qq_order + buffer), so the
    # internal η sum can be tighter than qq_internal.  We use qq_order
    # itself as the η budget: this is exact for the diamond rule and
    # avoids wasting time on high-η terms that will be discarded.
    if eta_order is None:
        eta_order = q_order_half

    qq_order = q_order_half

    # ------------------------------------------------------------------
    # Step 1: HJ continued fraction
    # ------------------------------------------------------------------
    hj_ks = hj_continued_fraction(P, Q)
    ell = len(hj_ks)
    if verbose:
        print(f"[refined_filling] P={P}, Q={Q}, HJ-CF={hj_ks}, ℓ={ell}")

    # ------------------------------------------------------------------
    # Helper: build full (m_ext, e_ext) from cusp charge (m_i, e_i)
    # ------------------------------------------------------------------
    def _make_ext(
        m_i: int, e_i: Fraction | int,
    ) -> tuple[list[int], list[int | Fraction]]:
        m_ext: list[int] = []
        e_ext: list[int | Fraction] = []
        other_m_iter = iter(m_other)
        other_e_iter = iter(e_other)
        for k_idx in range(r):
            if k_idx == cusp_idx:
                m_ext.append(m_i)
                e_ext.append(e_i)
            else:
                m_ext.append(next(other_m_iter))
                e_ext.append(next(other_e_iter))
        return m_ext, e_ext

    # ------------------------------------------------------------------
    # Step 2: ℓ = 1 special case (no IS kernel, no cusp η)
    # ------------------------------------------------------------------
    if ell == 1:
        k1 = hj_ks[0]
        if verbose:
            print(f"[refined_filling] ℓ=1, k={k1}: refined K(k,1) filling")

        # Enumerate (m, e) from unrefined K(k1, 1) support
        slope1_terms = _enumerate_slope1_terms(k1, m1_range)

        total_series: MultiEtaSeries = {}
        n_terms = 0

        for m_t, e_t, c_val, phase_t in slope1_terms:
            m_ext, e_ext = _make_ext(m_t, e_t)
            # Extra qq budget for c=0 terms that shift by ±phase
            extra_q = abs(phase_t) if c_val == 0 else 0
            refined = _cached_compute_refined_index(
                nz_data, m_ext, e_ext, q_order_half=qq_order + extra_q
            )
            if not refined:
                continue
            n_terms += 1

            # Apply Weyl shift η^{b·m_I + a·e_I} before filling
            if weyl_a is not None and weyl_b is not None:
                refined = _apply_weyl_shift(
                    refined, m_ext, e_ext, weyl_a, weyl_b, num_hard,
                    cusp_idx=cusp_idx,
                )

            # Convert to MultiEtaSeries (no cusp η dimension)
            multi = _refined_to_multi(refined, append_cusp_eta=False)

            # Determine multiplicity
            m0, _ = _particular_solution(k1, 1, c_val)
            t_abs = abs(m_t - m0)
            mult = 2 if (c_val == 2 or (c_val == 0 and t_abs > 0)) else 1

            contribution = _apply_k1_factor_multi(
                multi, c_val, phase_t, mult, qq_order
            )
            total_series = _multi_add(total_series, contribution)

        if verbose:
            print(
                f"[refined_filling] ℓ=1 done: {n_terms} terms, "
                f"{len(total_series)} non-zero entries"
            )

        return FilledRefinedResult(
            P=P, Q=Q, cusp_idx=cusp_idx,
            series=total_series,
            qq_order=qq_order,
            eta_order=0,
            hj_ks=hj_ks,
            n_kernel_terms=n_terms,
            num_hard=num_hard,
            has_cusp_eta=False,
            num_cusp_eta=0,
        )

    # ------------------------------------------------------------------
    # Step 3: ℓ ≥ 2 — Check for pre-computed kernel (fast path)
    # ------------------------------------------------------------------
    from manifold_index.core.kernel_cache import (
        apply_precomputed_kernel,
        load_kernel_table,
    )

    cached_kernel = load_kernel_table(P, Q, qq_order)
    if cached_kernel is not None:
        if verbose:
            extra = ""
            if cached_kernel.qq_order > qq_order:
                extra = f", stored at qq={cached_kernel.qq_order}"
            print(
                f"[refined_filling] ℓ={ell}: using pre-computed kernel "
                f"({len(cached_kernel.table)} entries{extra})"
            )
        total_series_fast = apply_precomputed_kernel(
            cached_kernel,
            nz_data,
            cusp_idx=cusp_idx,
            m_other=m_other,
            e_other=e_other,
            weyl_a=weyl_a,
            weyl_b=weyl_b,
            qq_order=qq_order,
            verbose=verbose,
            n_workers=n_workers,
            cache_iref=cache_iref,
            manifold_name=manifold_name,
        )
        # Apply diamond truncation: qq + |cusp_eta| ≤ qq_order
        truncated: MultiEtaSeries = {
            k: v for k, v in total_series_fast.items()
            if k[0] + abs(k[-1]) <= qq_order
        }
        return FilledRefinedResult(
            P=P, Q=Q, cusp_idx=cusp_idx,
            series=truncated,
            qq_order=qq_order,
            eta_order=eta_order,
            hj_ks=hj_ks,
            n_kernel_terms=len(cached_kernel.table),
            num_hard=num_hard,
            has_cusp_eta=True,
            num_cusp_eta=1,
        )

    # ------------------------------------------------------------------
    # Step 3b: ℓ ≥ 2 — Auto-precompute kernel if missing (parallel)
    # ------------------------------------------------------------------
    # The pre-computed kernel is manifold-independent: it only depends on
    # the slope (P, Q) and the qq_order.  Precomputing it once (using all
    # available CPU cores) and saving to disk is slower for the *first*
    # call but pays for itself on every subsequent call (with ANY manifold
    # at the same slope).  Since NC-transformed slopes recur across
    # manifolds, this is almost always beneficial.
    if auto_precompute:
        from manifold_index.core.kernel_cache import (
            precompute_filling_kernel,
            save_kernel_table,
        )
        import os as _os

        _n_auto_workers = max(1, (_os.cpu_count() or 4) - 2)
        if verbose:
            print(
                f"[refined_filling] ℓ={ell}: no cached kernel for "
                f"({P}/{Q}) at qq≥{qq_order}. "
                f"Auto-precomputing with {_n_auto_workers} workers…"
            )

        auto_kernel = precompute_filling_kernel(
            P, Q, qq_order=qq_order,
            verbose=verbose,
            n_workers=_n_auto_workers,
        )
        save_kernel_table(auto_kernel)
        if verbose:
            print(
                f"[refined_filling] Kernel saved: "
                f"{len(auto_kernel.table)} entries, "
                f"{auto_kernel.compute_time_s:.1f}s"
            )

        # Now apply the freshly computed kernel (fast path)
        total_series_auto = apply_precomputed_kernel(
            auto_kernel,
            nz_data,
            cusp_idx=cusp_idx,
            m_other=m_other,
            e_other=e_other,
            weyl_a=weyl_a,
            weyl_b=weyl_b,
            qq_order=qq_order,
            verbose=verbose,
            n_workers=n_workers,
            cache_iref=cache_iref,
            manifold_name=manifold_name,
        )
        truncated_auto: MultiEtaSeries = {
            k: v for k, v in total_series_auto.items()
            if k[0] + abs(k[-1]) <= qq_order
        }
        return FilledRefinedResult(
            P=P, Q=Q, cusp_idx=cusp_idx,
            series=truncated_auto,
            qq_order=qq_order,
            eta_order=eta_order,
            hj_ks=hj_ks,
            n_kernel_terms=len(auto_kernel.table),
            num_hard=num_hard,
            has_cusp_eta=True,
            num_cusp_eta=1,
        )

    # ------------------------------------------------------------------
    # Step 3c: ℓ ≥ 2 — Grid scan + IS chain (fallback)
    # ------------------------------------------------------------------
    # Reached when auto_precompute=False or the slope has no cached
    # kernel yet.  This is the original direct-computation path.
    #
    # Buffer rationale: The IS convolution produces cusp-η terms whose
    # reliability degrades near the qq truncation boundary.  For a term
    # at qq_power=a, cusp_eta=e to be correct after diamond truncation
    # (a + |e| ≤ qq_order), the internal computation must carry
    # qq_internal ≥ qq_order + |e|_max ≈ 2·qq_order.  The old buffer
    # (qq_order//2 + 4) was insufficient: at qq_order=20 it gave
    # qq_internal=34, producing spurious terms at (qq, |cusp_eta|) near
    # the diamond boundary (e.g. 5_1 knot, 1/0 filling).
    _is_buffer = qq_order + 4
    qq_internal = qq_order + _is_buffer
    m1_range = max(m1_range, 2 * qq_internal)
    if verbose:
        print(
            f"[refined_filling] ℓ={ell}: qq_order={qq_order}, "
            f"qq_internal={qq_internal} (buffer={_is_buffer}), "
            f"eta_order={eta_order}"
        )
        print(f"[refined_filling] scanning (m,e) grid for I^ref ≠ 0")

    # Scan bounds: m ∈ [-m_scan, m_scan], e ∈ [-e_scan, e_scan] step 1/2
    m_scan = 2 * qq_internal
    e_scan = qq_internal  # in half-integer units, covers ±qq_internal/2

    state: dict[tuple[int, Fraction], MultiEtaSeries] = {}
    n_grid_terms = 0

    for m_i in range(-m_scan, m_scan + 1):
        for e_half in range(-2 * e_scan, 2 * e_scan + 1):
            e_i = Fraction(e_half, 2)
            m_ext, e_ext = _make_ext(m_i, e_i)

            refined = _cached_compute_refined_index(
                nz_data, m_ext, e_ext, q_order_half=qq_internal
            )
            if not refined:
                continue
            n_grid_terms += 1

            # Apply Weyl shift η^{b·m_I + a·e_I} before IS convolutions
            if weyl_a is not None and weyl_b is not None:
                refined = _apply_weyl_shift(
                    refined, m_ext, e_ext, weyl_a, weyl_b, num_hard,
                    cusp_idx=cusp_idx,
                )

            # Convert to MultiEtaSeries with cusp_eta=0 appended
            # use_int=True: int arithmetic in the IS chain (3-5× faster)
            multi = _refined_to_multi(refined, append_cusp_eta=True, use_int=True)
            existing = state.get((m_i, e_i))
            state[(m_i, e_i)] = (
                _multi_add(existing, multi) if existing else multi
            )

    if verbose:
        print(
            f"[refined_filling] Grid scan: {n_grid_terms} non-zero (m,e) "
            f"pairs → {len(state)} state entries"
        )

    # ------------------------------------------------------------------
    # Step 4: Apply ℓ−1 IS convolution steps
    # ------------------------------------------------------------------
    for step_i in range(ell - 1):
        k_current = hj_ks[step_i]
        k_next = hj_ks[step_i + 1]
        if verbose:
            print(
                f"[refined_filling] IS step {step_i+1}/{ell-1}: "
                f"k_current={k_current}, k_next={k_next}, "
                f"|state|={len(state)}"
            )
        state = _apply_is_step(
            state,
            k_current=k_current,
            k_next=k_next,
            qq_order=qq_internal,
            eta_order=eta_order,
            m1_range=m1_range,
            use_int=True,  # ×2 int IS kernel for performance
            is_last_step=(step_i == ell - 2),
            n_workers=n_workers,
        )
        if verbose:
            print(f"            → new |state|={len(state)}")

    # ------------------------------------------------------------------
    # Step 5: Apply final unrefined K(k_ℓ, 1; m_{ℓ-1}, e_{ℓ-1})
    # ------------------------------------------------------------------
    k_final = hj_ks[-1]
    # Use full enumeration (no ±t / ±c shortcuts) — see _enumerate_slope1_all.
    final_terms = _enumerate_slope1_all(k_final, m1_range)
    # Build lookup: (m, e) → (c, phase, multiplicity=1)
    final_term_info: dict[tuple[int, Fraction], tuple[int, int, int]] = {}
    seen_final: set[tuple[int, Fraction]] = set()
    for m1, e1, c_final, phase_final in final_terms:
        key = (m1, e1)
        if key in seen_final:
            continue
        seen_final.add(key)
        # Multiplicity is always 1 — every (m,e) is enumerated explicitly.
        final_term_info[key] = (c_final, phase_final, 1)

    total_series_ell2: MultiEtaSeries = {}
    for (m1, e1), src_series in state.items():
        if not src_series:
            continue
        info = final_term_info.get((m1, e1))
        if info is None:
            continue
        c_final, phase_final, mult_final = info
        contribution = _apply_k1_factor_multi(
            src_series, c_final, phase_final, mult_final, qq_internal,
            int_mode=True,
        )
        total_series_ell2 = _multi_add(total_series_ell2, contribution)

    # ------------------------------------------------------------------
    # Convert int-scaled values back to Fraction
    # ------------------------------------------------------------------
    # The LCD (least common denominator) accumulated through the chain:
    #   - Each IS step introduces ×2 (from _is_kernel absorbing ½)
    #   - The final K-factor introduces another ×2 (from int_mode ½ absorbed)
    #   Total LCD = 2^ℓ  where ℓ = len(hj_ks)
    # Use Python int (arbitrary precision) — NOT numpy int64 which overflows at ell >= 63
    lcd = 1 << ell  # 2^ℓ

    # ------------------------------------------------------------------
    # Step 6: Truncate to user-requested qq_order  +  diamond cutoff
    # ------------------------------------------------------------------
    # Standard truncation: qq_power ≤ qq_order.
    # Diamond truncation on cusp η:  qq_power + |cusp_eta| ≤ qq_order.
    #
    # The IS convolution produces cusp-η terms whose reliability is
    # limited by distance from the qq truncation boundary.  A term
    # q^a · η_c^e requires ~|e| extra qq-budget to be fully resolved;
    # terms near the boundary (a + |e| > qq_order) are artifacts of the
    # finite series truncation and are not stable under changes in
    # qq_order.  The diamond rule removes exactly these artifacts.
    #
    # The cusp η is always the LAST key dimension (appended in step 3).
    truncated: MultiEtaSeries = {}
    for k, v in total_series_ell2.items():
        if k[0] + abs(k[-1]) <= qq_order:
            frac_v = Fraction(v, lcd)
            if frac_v != 0:
                truncated[k] = frac_v

    if verbose:
        n_raw = sum(1 for k, v in total_series_ell2.items() if k[0] <= qq_order and v != 0)
        print(
            f"[refined_filling] Done: {len(truncated)} "
            f"non-zero multi-η entries (diamond from {n_raw} raw, "
            f"pre-truncation {len(total_series_ell2)})"
        )

    return FilledRefinedResult(
        P=P, Q=Q, cusp_idx=cusp_idx,
        series=truncated,
        qq_order=qq_order,
        eta_order=eta_order,
        hj_ks=hj_ks,
        n_kernel_terms=n_grid_terms,
        num_hard=num_hard,
        has_cusp_eta=True,
        num_cusp_eta=1,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Part 9 — Multi-cusp sequential filling
# ═══════════════════════════════════════════════════════════════════════════


def _apply_filling_kernel_to_intermediate(
    intermediate: dict[tuple[int, Fraction], MultiEtaSeries],
    P: int,
    Q: int,
    qq_order: int,
    eta_order: int | None = None,
    m1_range: int | None = None,
    num_hard: int = 0,
    num_cusp_eta_in: int = 0,
    verbose: bool = False,
) -> FilledRefinedResult:
    """Apply the refined Dehn filling kernel to precomputed intermediate series.

    This is the multi-cusp analogue of ``compute_filled_refined_index``.
    Instead of computing I^ref from NZ data, it looks up results from
    *intermediate*, which maps ``(m, e)`` charges (for the cusp being
    filled) to MultiEtaSeries from previous filling steps.

    The maths are identical to ``compute_filled_refined_index``; only the
    input source differs.

    Parameters
    ----------
    intermediate : dict[(int, Fraction), MultiEtaSeries]
        Precomputed series from a previous filling step, keyed by the
        (m, e) charges of the cusp now being filled.
    P, Q : int
        Coprime slope for this cusp filling.
    qq_order : int
        Series truncation in q^{1/2} powers.
    eta_order : int or None
        Maximum |cusp η exponent|.  Default: qq_order.
    m1_range : int or None
        Scan range for intermediate variables.  Default: 2 * qq_order.
    num_hard : int
        Number of hard-edge η dimensions (for metadata).
    num_cusp_eta_in : int
        Number of cusp-η dimensions already present in the intermediate
        series (from previous filling steps).
    verbose : bool

    Returns
    -------
    FilledRefinedResult
    """
    if eta_order is None:
        eta_order = qq_order
    if m1_range is None:
        m1_range = 2 * qq_order

    hj_ks = hj_continued_fraction(P, Q)
    ell = len(hj_ks)

    if verbose:
        print(
            f"[multi_fill] P={P}, Q={Q}, HJ-CF={hj_ks}, ℓ={ell}, "
            f"|intermediate|={len(intermediate)}, "
            f"num_cusp_eta_in={num_cusp_eta_in}"
        )

    # ------------------------------------------------------------------
    # ℓ = 1: direct K(k1, 1) application — no new cusp η added
    # ------------------------------------------------------------------
    # The ℓ=1 kernel is unrefined, so it does NOT introduce a new cusp η.
    # However the intermediate series may already carry cusp η's from
    # previous filling steps; those are preserved through _apply_k1_factor_multi.
    if ell == 1:
        k1 = hj_ks[0]
        slope1_terms = _enumerate_slope1_terms(k1, m1_range)

        total_series: MultiEtaSeries = {}
        n_terms = 0

        for m_t, e_t, c_val, phase_t in slope1_terms:
            multi = intermediate.get((m_t, Fraction(e_t)))
            if multi is None:
                multi = intermediate.get((m_t, e_t))
            if not multi:
                continue
            n_terms += 1

            m0, _ = _particular_solution(k1, 1, c_val)
            t_abs = abs(m_t - m0)
            mult = 2 if (c_val == 2 or (c_val == 0 and t_abs > 0)) else 1

            contribution = _apply_k1_factor_multi(
                multi, c_val, phase_t, mult, qq_order
            )
            total_series = _multi_add(total_series, contribution)

        if verbose:
            print(
                f"[multi_fill] ℓ=1 done: {n_terms} terms, "
                f"{len(total_series)} entries"
            )

        # ℓ=1 does NOT add a new cusp η → num_cusp_eta stays the same
        num_cusp_eta_out = num_cusp_eta_in
        return FilledRefinedResult(
            P=P, Q=Q, cusp_idx=-1,
            series=total_series,
            qq_order=qq_order,
            eta_order=0,
            hj_ks=hj_ks,
            n_kernel_terms=n_terms,
            num_hard=num_hard,
            has_cusp_eta=(num_cusp_eta_out > 0),
            num_cusp_eta=num_cusp_eta_out,
        )

    # ------------------------------------------------------------------
    # ℓ ≥ 2: IS convolution chain
    # ------------------------------------------------------------------
    _is_buffer = qq_order + 4
    qq_internal = qq_order + _is_buffer
    m1_range = max(m1_range, 2 * qq_internal)

    if verbose:
        print(
            f"[multi_fill] ℓ={ell}: qq_internal={qq_internal}, "
            f"eta_order={eta_order}"
        )

    # Build state from intermediate — extend each series with cusp_eta=0
    # for THIS filling step.  The intermediate may already have cusp_eta
    # dimensions from previous filling steps; those are preserved and the
    # new cusp_eta is appended as the last dimension.
    m_scan = 2 * qq_internal
    e_scan = qq_internal
    state: dict[tuple[int, Fraction], MultiEtaSeries] = {}
    n_grid_terms = 0

    for m_i in range(-m_scan, m_scan + 1):
        for e_half in range(-2 * e_scan, 2 * e_scan + 1):
            e_i = Fraction(e_half, 2)
            multi = intermediate.get((m_i, e_i))
            if not multi:
                continue
            n_grid_terms += 1

            # Extend with a new cusp_eta=0 dimension for THIS filling
            extended: MultiEtaSeries = {}
            for k, v in multi.items():
                extended[k + (0,)] = v
            state[(m_i, e_i)] = extended

    if verbose:
        print(f"[multi_fill] grid: {n_grid_terms} non-zero entries")

    # IS convolution steps
    for step_i in range(ell - 1):
        k_current = hj_ks[step_i]
        k_next = hj_ks[step_i + 1]
        if verbose:
            print(
                f"[multi_fill] IS step {step_i+1}/{ell-1}: "
                f"|state|={len(state)}"
            )
        state = _apply_is_step(
            state,
            k_current=k_current,
            k_next=k_next,
            qq_order=qq_internal,
            eta_order=eta_order,
            m1_range=m1_range,
            # use_int=False (default): Fraction state from intermediate
            is_last_step=(step_i == ell - 2),
        )

    # Final K(k_ℓ, 1) application
    k_final = hj_ks[-1]
    final_terms = _enumerate_slope1_all(k_final, m1_range)
    final_term_info: dict[tuple[int, Fraction], tuple[int, int, int]] = {}
    seen_final: set[tuple[int, Fraction]] = set()
    for m1, e1, c_final, phase_final in final_terms:
        key = (m1, e1)
        if key in seen_final:
            continue
        seen_final.add(key)
        final_term_info[key] = (c_final, phase_final, 1)

    total_series_ell2: MultiEtaSeries = {}
    for (m1, e1), src_series in state.items():
        if not src_series:
            continue
        info = final_term_info.get((m1, e1))
        if info is None:
            continue
        c_final, phase_final, mult_final = info
        contribution = _apply_k1_factor_multi(
            src_series, c_final, phase_final, mult_final, qq_internal
        )
        total_series_ell2 = _multi_add(total_series_ell2, contribution)

    # Diamond truncation — generalized for multiple cusp η's.
    # Key structure: (qq, 2W_0, …, 2W_{H-1}, 2V_0, …, 2V_{C})
    # where C = num_cusp_eta_in (from previous fillings) + 1 (this filling).
    # Rule: qq_power + Σ|cusp_eta_i| ≤ qq_order
    num_cusp_eta_out = num_cusp_eta_in + 1
    cusp_start = 1 + num_hard  # index of first cusp_eta in key tuple

    truncated: MultiEtaSeries = {}
    for k, v in total_series_ell2.items():
        cusp_eta_sum = sum(
            abs(k[cusp_start + i])
            for i in range(num_cusp_eta_out)
            if cusp_start + i < len(k)
        )
        if k[0] + cusp_eta_sum <= qq_order:
            truncated[k] = v

    if verbose:
        print(
            f"[multi_fill] ℓ≥2 done: {len(truncated)} entries "
            f"(from {len(total_series_ell2)} raw), "
            f"num_cusp_eta={num_cusp_eta_out}"
        )

    return FilledRefinedResult(
        P=P, Q=Q, cusp_idx=-1,
        series=truncated,
        qq_order=qq_order,
        eta_order=eta_order,
        hj_ks=hj_ks,
        n_kernel_terms=n_grid_terms,
        num_hard=num_hard,
        has_cusp_eta=True,
        num_cusp_eta=num_cusp_eta_out,
    )


def _refined_to_multi_with_spectators(
    refined: RefinedIndexResult,
    m_tag: int,
    e_x2_tag: int,
    append_cusp_eta: bool = False,
) -> MultiEtaSeries:
    """Convert a RefinedIndexResult to MultiEtaSeries with spectator dims.

    Like ``_refined_to_multi`` but inserts spectator dimensions
    ``(m_tag, e_x2_tag)`` after the hard-edge η's and before the optional
    cusp-η dimension.

    Key structure:
        ``(qq, 2W_0, …, 2W_{H-1}, m_tag, e_x2_tag [, 2V=0])``

    Parameters
    ----------
    refined : RefinedIndexResult
    m_tag : int
        Spectator tag for the next cusp's meridian charge.
    e_x2_tag : int
        Spectator tag for the next cusp's longitude charge (``2 * e``).
    append_cusp_eta : bool
        Whether to append ``cusp_eta = 0`` at the end.

    Returns
    -------
    MultiEtaSeries
    """
    result: MultiEtaSeries = {}
    for key, coeff in refined.items():
        if coeff == 0:
            continue
        new_key = key + (m_tag, e_x2_tag)
        if append_cusp_eta:
            new_key = new_key + (0,)
        result[new_key] = Fraction(coeff)
    return result


def _needed_spectator_charges(
    spec: "MultiCuspFillSpec",
    qq_order: int,
) -> set[tuple[int, Fraction]]:
    """Determine which (m, e) charge pairs the given filling needs.

    For an ℓ=1 filling, only the K(k,1) support matters (finite set).
    For an ℓ≥2 filling, we return the full grid — the actual reduction
    happens inside ``_batched_first_filling`` via probe-based filtering.

    Returns
    -------
    set of (int, Fraction) pairs
    """
    hj_ks = hj_continued_fraction(spec.P, spec.Q)
    ell = len(hj_ks)

    if ell == 1:
        # Only the K(k1, 1) support — use full enumeration
        k1 = hj_ks[0]
        m1_range = 2 * qq_order
        terms = _enumerate_slope1_all(k1, m1_range)
        return set((m, e) for m, e, _, _ in terms)
    else:
        # ℓ ≥ 2: return the full grid.  The _batched_first_filling
        # function will probe to discover the actual sparse support
        # and skip the vast majority of zero entries.
        _is_buffer = qq_order + 4  # was qq_order // 2 + 4 — insufficient for ℓ≥3
        qq_internal = qq_order + _is_buffer
        m_scan = 2 * qq_internal
        e_scan = qq_internal
        result = set()
        for m in range(-m_scan, m_scan + 1):
            for e_half in range(-2 * e_scan, 2 * e_scan + 1):
                result.add((m, Fraction(e_half, 2)))
        return result


def _batched_first_filling(
    nz_data: "NeumannZagierData",
    first_spec: "MultiCuspFillSpec",
    next_cusp_idx: int,
    needed_me: set[tuple[int, Fraction]],
    qq_order: int,
    verbose: bool = False,
    progress_callback=None,
    auto_precompute: bool = False,
    cache_iref: bool = False,
    manifold_name: str = "unknown",
) -> tuple[dict[tuple[int, Fraction], MultiEtaSeries], int]:
    """Compute the first filling with batched spectator dimensions.

    This is the performance-critical optimisation for multi-cusp filling.
    Instead of calling ``compute_filled_refined_index`` once per
    ``(m_next, e_next)`` charge pair (which repeats the expensive IS chain
    each time), this function:

    1. Pre-computes I^ref for all needed ``(m0, e0, m_next, e_next)``
       combinations in a two-pass scan.
    2. Embeds ``(m_next, e_next)`` as *spectator* dimensions in the
       MultiEtaSeries key tuples.
    3. Runs the HJ continued-fraction kernel chain (IS steps + final K)
       exactly ONCE on the combined state.
    4. Extracts per-``(m_next, e_next)`` intermediate series from the
       batched result.

    The spectator dimensions are transparent to the IS convolution
    (which operates on the *last* key dim) and the K-factor (which shifts
    only the *first* key dim), so no changes to the core chain functions
    are needed.

    Parameters
    ----------
    nz_data : NeumannZagierData
    first_spec : MultiCuspFillSpec
        Filling specification for the first cusp.
    next_cusp_idx : int
        Cusp index of the *next* cusp to be filled (the spectator).
    needed_me : set of (int, Fraction)
        Set of (m_next, e_next) pairs that the second filling needs.
    qq_order : int
    verbose : bool
    progress_callback : callable or None

    Returns
    -------
    intermediate : dict[(int, Fraction) → MultiEtaSeries]
        Keyed by ``(m_next, e_next)``.  Each value is a MultiEtaSeries
        with keys ``(qq, 2W_0, …, 2W_{H-1} [, 2V])``.
    num_cusp_eta : int
        0 if ℓ=1 (no cusp η), 1 if ℓ≥2.
    """
    r = nz_data.r
    num_hard = nz_data.num_hard
    cusp_idx = first_spec.cusp_idx
    P, Q = first_spec.P, first_spec.Q
    weyl_a = first_spec.weyl_a
    weyl_b = first_spec.weyl_b

    def _status(msg: str):
        if progress_callback:
            progress_callback(msg)
        if verbose:
            print(msg)

    # HJ continued fraction
    hj_ks = hj_continued_fraction(P, Q)
    ell = len(hj_ks)

    # ------------------------------------------------------------------
    # ℓ=1 path: per-spectator delegation (no IS chain to amortise)
    # ------------------------------------------------------------------
    # For ℓ=1 the filling is a single K-factor pass over the K-support
    # lines — there is no IS chain.  Batching all spectators into one
    # combined state provides *zero* performance benefit (the bottleneck
    # is the per-point compute_refined_index calls, which are the same
    # either way) and introduces subtle extra_q budget and multiplicity
    # complications.  Instead, simply call compute_filled_refined_index
    # for each spectator charge, which is known-correct and efficient:
    # ~243 K-support points × ~243 spectators × 0.0014 s ≈ 83 s at
    # qq_order = 20.
    if ell == 1:
        _status(
            f"[batched] ℓ=1: per-spectator filling cusp {cusp_idx} "
            f"({P}/{Q}), {len(needed_me)} spectator charges…"
        )

        intermediate: dict[tuple[int, Fraction], MultiEtaSeries] = {}
        n_done = 0

        for m_next, e_next in sorted(needed_me):
            # Build m_other / e_other for this spectator value
            m_other: list[int] = []
            e_other: list[int | Fraction] = []
            for j in range(r):
                if j == cusp_idx:
                    continue
                if j == next_cusp_idx:
                    m_other.append(m_next)
                    e_other.append(e_next)
                else:
                    m_other.append(0)
                    e_other.append(Fraction(0))

            result = compute_filled_refined_index(
                nz_data,
                cusp_idx=cusp_idx,
                P=P, Q=Q,
                m_other=m_other,
                e_other=e_other,
                q_order_half=qq_order,
                weyl_a=weyl_a,
                weyl_b=weyl_b,
                verbose=False,
                auto_precompute=auto_precompute,
                cache_iref=cache_iref,
                manifold_name=manifold_name,
            )

            if result.series:
                series = result.series
                if first_spec.incompat_edges:
                    collapsed = result.collapse_eta_edges(
                        first_spec.incompat_edges
                    )
                    series = collapsed.series
                if series:
                    intermediate[(m_next, e_next)] = series

            n_done += 1
            if n_done % 50 == 0:
                _status(
                    f"[batched] ℓ=1: {n_done}/{len(needed_me)} spectators, "
                    f"{len(intermediate)} non-zero"
                )

        _status(
            f"[batched] ℓ=1 done: {len(intermediate)} non-zero "
            f"intermediates from {len(needed_me)} spectators"
        )
        return intermediate, 0

    # ------------------------------------------------------------------
    # ℓ ≥ 2 path: per-spectator delegation with probe-based filtering
    # ------------------------------------------------------------------
    # For ℓ≥2 the filling involves an IS convolution chain that is
    # expensive (~40-60 s per call at qq_order=20).  Embedding all
    # spectators into a single batched state makes the IS chain's
    # per-entry convolution cost proportional to the number of
    # spectators, giving no net speedup.
    #
    # Instead, we use the same per-spectator delegation as ℓ=1:
    #   1. Probe to discover which spectators produce non-zero I^ref
    #      (~30 s for ~18,000 probe calls).
    #   2. Call compute_filled_refined_index once per active spectator.
    #      Typical: ~50 active spectators × ~66 s ≈ 55 min.
    #
    # This is simple, correct (reuses the known-good single-cusp path),
    # and feasible.
    # ------------------------------------------------------------------
    _is_buffer = qq_order + 4  # was qq_order // 2 + 4 — insufficient for ℓ≥3
    qq_internal = qq_order + _is_buffer

    m_scan = 2 * qq_internal
    e_scan = qq_internal

    # ── Probe: discover active spectator charges ─────────────────────
    # Fix (m0=0, e0=0) for the filling cusp and scan all candidate
    # spectator (m_next, e_next) values.  Non-zero I^ref entries identify
    # the active spectator support.  The (0,0) probe point captures the
    # widest support; we also sample a few extreme grid points.
    full_grid_size = (2 * m_scan + 1) * (4 * e_scan + 1)
    _status(
        f"[batched] ℓ={ell}: probing {full_grid_size} spectator "
        f"charges at (m0=0, e0=0)…"
    )

    def _make_ext_probe(m_fill, e_fill, m_sp, e_sp):
        m_ext, e_ext = [], []
        for j in range(r):
            if j == cusp_idx:
                m_ext.append(m_fill); e_ext.append(e_fill)
            elif j == next_cusp_idx:
                m_ext.append(m_sp); e_ext.append(e_sp)
            else:
                m_ext.append(0); e_ext.append(Fraction(0))
        return m_ext, e_ext

    probe_spectators: set[tuple[int, Fraction]] = set()
    for m_next in range(-m_scan, m_scan + 1):
        for e_half_next in range(-2 * e_scan, 2 * e_scan + 1):
            e_next = Fraction(e_half_next, 2)
            m_ext, e_ext = _make_ext_probe(0, Fraction(0), m_next, e_next)
            refined = _cached_compute_refined_index(
                nz_data, m_ext, e_ext, q_order_half=qq_internal
            )
            if refined:
                probe_spectators.add((m_next, e_next))

    _status(
        f"[batched] probe done: {len(probe_spectators)} active "
        f"spectators of {full_grid_size}"
    )

    # Intersect with what the second filling actually needs
    active_me = needed_me & probe_spectators
    if not active_me:
        # Fall back: use all probed spectators
        active_me = probe_spectators

    _status(
        f"[batched] ℓ={ell}: per-spectator filling cusp {cusp_idx} "
        f"({P}/{Q}), {len(active_me)} active spectator charges…"
    )

    # ── Per-spectator delegation ─────────────────────────────────────
    intermediate: dict[tuple[int, Fraction], MultiEtaSeries] = {}
    n_done = 0

    for m_next, e_next in sorted(active_me):
        n_done += 1

        # Build m_other / e_other for this spectator value
        m_other: list[int] = []
        e_other: list[int | Fraction] = []
        for j in range(r):
            if j == cusp_idx:
                continue
            if j == next_cusp_idx:
                m_other.append(m_next)
                e_other.append(e_next)
            else:
                m_other.append(0)
                e_other.append(Fraction(0))

        result = compute_filled_refined_index(
            nz_data,
            cusp_idx=cusp_idx,
            P=P, Q=Q,
            m_other=m_other,
            e_other=e_other,
            q_order_half=qq_order,
            weyl_a=weyl_a,
            weyl_b=weyl_b,
            verbose=False,
            auto_precompute=auto_precompute,
            cache_iref=cache_iref,
            manifold_name=manifold_name,
        )

        if result.series:
            series = result.series
            if first_spec.incompat_edges:
                collapsed = result.collapse_eta_edges(
                    first_spec.incompat_edges
                )
                series = collapsed.series
            if series:
                intermediate[(m_next, e_next)] = series

        if n_done % 5 == 0 or n_done == len(active_me):
            _status(
                f"[batched] ℓ={ell}: {n_done}/{len(active_me)} "
                f"spectators, {len(intermediate)} non-zero"
            )

    _status(
        f"[batched] ℓ={ell} done: {len(intermediate)} non-zero "
        f"intermediates from {len(active_me)} spectators"
    )
    num_cusp_eta_out = 1  # ℓ ≥ 2 always adds one cusp η
    return intermediate, num_cusp_eta_out


@dataclass
class MultiCuspFillSpec:
    """Specification for filling one cusp in a multi-cusp filling."""
    cusp_idx: int
    P: int          # slope in NC-transformed basis
    Q: int
    weyl_a: list[Fraction] | None = None
    weyl_b: list[Fraction] | None = None
    incompat_edges: list[int] | None = None


def compute_multi_cusp_filled_refined_index(
    nz_data: NeumannZagierData,
    fill_specs: list[MultiCuspFillSpec],
    q_order_half: int = 10,
    verbose: bool = False,
    progress_callback=None,
    auto_precompute: bool = False,
    cache_iref: bool = False,
    manifold_name: str = "unknown",
) -> FilledRefinedResult:
    """Sequentially fill multiple cusps of a manifold.

    Fills cusps one at a time.  After filling cusp *j*, the intermediate
    result is a function of the remaining unfilled cusp charges.  The next
    filling operates on these intermediate results.

    Algorithm
    ---------
    For a manifold with r cusps, filling cusps j_1, j_2, …, j_k:

    1. Fill cusp j_1:
       For each (m, e) grid point of the REMAINING cusps, compute
       ``compute_filled_refined_index(nz, cusp_idx=j_1, P_1, Q_1,
       m_other=[…], e_other=[…])``.
       Store results keyed by the next cusp's (m, e).

    2. Fill cusp j_2:
       Apply ``_apply_filling_kernel_to_intermediate(intermediate, P_2, Q_2)``
       where intermediate maps (m_{j_2}, e_{j_2}) → MultiEtaSeries.

    3. Repeat for remaining cusps.

    Parameters
    ----------
    nz_data : NeumannZagierData
    fill_specs : list[MultiCuspFillSpec]
        One spec per cusp to fill, in the order they should be processed.
        The NZ data should already have cusp basis changes applied.
    q_order_half : int
    verbose : bool
    progress_callback : callable or None
        Called as ``progress_callback(msg: str)`` for status updates.

    Returns
    -------
    FilledRefinedResult
        Single combined result with all specified cusps filled.
    """
    r = nz_data.r
    num_hard = nz_data.num_hard
    n_fills = len(fill_specs)
    qq_order = q_order_half

    if n_fills == 0:
        raise ValueError("No fill specs provided")

    def _status(msg: str):
        if progress_callback:
            progress_callback(msg)
        if verbose:
            print(msg)

    if n_fills == 1:
        # Single-cusp case: delegate directly
        spec = fill_specs[0]
        _status(f"Filling cusp {spec.cusp_idx} with ({spec.P}, {spec.Q})…")
        result = compute_filled_refined_index(
            nz_data,
            cusp_idx=spec.cusp_idx,
            P=spec.P, Q=spec.Q,
            m_other=[0] * (r - 1),
            e_other=[0] * (r - 1),
            q_order_half=q_order_half,
            weyl_a=spec.weyl_a,
            weyl_b=spec.weyl_b,
            verbose=verbose,
            auto_precompute=auto_precompute,
            cache_iref=cache_iref,
            manifold_name=manifold_name,
        )
        if spec.incompat_edges:
            result = result.collapse_eta_edges(spec.incompat_edges)
        return result

    # ------------------------------------------------------------------
    # Multi-cusp (n_fills == 2): batched first filling + second filling
    # ------------------------------------------------------------------
    if n_fills > 2:
        raise NotImplementedError(
            "Sequential filling of >2 cusps not yet implemented. "
            f"Have {n_fills} fill specs."
        )

    first = fill_specs[0]
    second = fill_specs[1]
    next_cusp = second.cusp_idx

    # Step 1: determine what (m, e) pairs the second filling needs
    needed_me = _needed_spectator_charges(second, qq_order)
    _status(
        f"Step 1/{n_fills}: Batched filling cusp {first.cusp_idx} "
        f"with ({first.P}, {first.Q}), "
        f"{len(needed_me)} spectator charges for cusp {next_cusp}…"
    )

    # Step 2: batched first filling
    intermediate, num_cusp_eta_accum = _batched_first_filling(
        nz_data,
        first_spec=first,
        next_cusp_idx=next_cusp,
        needed_me=needed_me,
        qq_order=qq_order,
        verbose=verbose,
        progress_callback=progress_callback,
        auto_precompute=auto_precompute,
        cache_iref=cache_iref,
        manifold_name=manifold_name,
    )

    _status(
        f"Step 1/{n_fills} done: {len(intermediate)} non-zero "
        f"intermediate entries, num_cusp_eta={num_cusp_eta_accum}"
    )

    # Step 3: second filling
    _status(
        f"Step 2/{n_fills}: Filling cusp {second.cusp_idx} "
        f"with ({second.P}, {second.Q}), "
        f"|intermediate|={len(intermediate)}, "
        f"num_cusp_eta_so_far={num_cusp_eta_accum}…"
    )

    fill_result = _apply_filling_kernel_to_intermediate(
        intermediate,
        P=second.P, Q=second.Q,
        qq_order=qq_order,
        num_hard=num_hard,
        num_cusp_eta_in=num_cusp_eta_accum,
        verbose=verbose,
    )
    num_cusp_eta_accum = fill_result.num_cusp_eta

    if second.incompat_edges:
        fill_result = fill_result.collapse_eta_edges(second.incompat_edges)

    _status(
        f"Step 2/{n_fills} done: "
        f"{len(fill_result.series)} entries in final result, "
        f"num_cusp_eta={fill_result.num_cusp_eta}"
    )

    return fill_result