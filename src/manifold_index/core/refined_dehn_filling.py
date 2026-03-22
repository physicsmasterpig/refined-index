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
# For ℓ=1 Dehn filling: key = (qq_power, 2*η_0, ..., 2*η_{k-1})
#   Same shape as RefinedIndexResult but with Fraction values.
# For ℓ≥2 Dehn filling: key = (qq_power, 2*η_0, ..., 2*η_{k-1}, cusp_eta)
#   Appends one additional IS kernel η-variable (integer exponent).
MultiEtaSeries = dict[tuple[int, ...], Fraction]

# ---------------------------------------------------------------------------
# Module-level caches
# ---------------------------------------------------------------------------
# _is_kernel and _etilde_is are pure functions of their (hashable) arguments.
# They are cached via @functools.lru_cache on the functions themselves.
#
# compute_refined_index depends on a NeumannZagierData object (not directly
# hashable), so we wrap it in a manual dict cache keyed by id(nz_data) and
# hashable representations of the remaining args.  This gives cross-P/Q
# memoisation: the I^ref grid scan for a second filling of the same manifold
# reuses previously computed values.

_CACHE_MISS = object()

_iref_cache: dict[tuple, dict] = {}


def _cached_compute_refined_index(
    nz_data: "NeumannZagierData",
    m_ext: list[int],
    e_ext: Sequence[int | Fraction],
    q_order_half: int,
) -> "RefinedIndexResult":
    """Wrapper around ``compute_refined_index`` with memoisation.

    The cache key uses ``id(nz_data)`` (valid as long as the same Python
    object is alive) and tuple-ified charge vectors.
    """
    key = (
        id(nz_data),
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

    return stats


# ---------------------------------------------------------------------------
# Part 1 — Hirzebruch-Jung continued fraction
# ---------------------------------------------------------------------------


def hj_continued_fraction(P: int, Q: int) -> list[int]:
    """Hirzebruch-Jung continued fraction for P/Q.

    Returns [k_1, …, k_ℓ] such that
        P/Q = k_1 − 1/(k_2 − 1/(… − 1/k_ℓ))
    where every k_i = ⌈remaining value⌉ ≥ 2 except possibly the
    terminal entry.

    Special cases
    -------------
    Q = 0, P ∈ {±1}  →  [0, 0]   (longitude / meridian special case)
    gcd(|P|, |Q|) must equal 1.

    Examples
    --------
    >>> hj_continued_fraction(1, 2)
    [1, 2]
    >>> hj_continued_fraction(5, 2)
    [3, 2]
    >>> hj_continued_fraction(1, 1)
    [1]
    >>> hj_continued_fraction(-1, 2)
    [0, 2]
    """
    if Q == 0:
        assert abs(P) == 1, f"Q=0 but |P|={abs(P)} ≠ 1"
        return [0, 0]

    # Normalise to Q > 0
    if Q < 0:
        P, Q = -P, -Q

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
) -> QEtaSeries:
    """Compute ẽI_S(m1, e1, m2, e2; η) = expr8[m1, e1, m2, e2] in DFK.nb.

    Returns a QEtaSeries dict[(qq_power, eta_exp) → Fraction].

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
    result: QEtaSeries = {}

    for t in range(-t_range, t_range + 1):
        e3 = e3_base + t
        e4 = e4_base + t

        # tind3 and tind4 are independent of the η sum variable
        s3 = _tet_index_series(m_a3, e3, qq_order)
        if not s3:
            continue
        s4 = _tet_index_series(m_a4, e4, qq_order)
        if not s4:
            continue

        # Convolve s3 · s4  (integer qq-series)
        s34 = _int_qqseries_convolve(s3, s4, qq_order)
        if not s34:
            continue

        # Sum over η exponent with correct parity
        for n_eta in range(-eta_order, eta_order + 1):
            e_var = 2 * n_eta + e_var_parity

            # tind1 and tind2 second args (integer by parity choice)
            e_a1 = t - n_eta + e_arg1_base
            e_a2 = t - n_eta + e_arg2_base

            s1 = _tet_index_series(m_a1, e_a1, qq_order)
            if not s1:
                continue
            s2 = _tet_index_series(m_a2, e_a2, qq_order)
            if not s2:
                continue

            # Phase factor: (−qq)^X  where  X = −e_var + B + 2t
            X = -e_var + B + 2 * t
            sign = 1 if X % 2 == 0 else -1

            # Convolve s1 · s2
            s12 = _int_qqseries_convolve(s1, s2, qq_order)
            if not s12:
                continue

            # Combine s12 · s34 · (−qq)^X
            for p12, c12 in s12.items():
                for p34, c34 in s34.items():
                    total_qq = p12 + p34 + X
                    if total_qq < 0 or total_qq > qq_order:
                        continue
                    key = (total_qq, e_var)
                    new_val = result.get(key, Fraction(0)) + Fraction(sign * c12 * c34)
                    if new_val == 0:
                        result.pop(key, None)
                    else:
                        result[key] = new_val

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
) -> QEtaSeries:
    """Compute I_S(m1, e1, m2, e2; η) — the symplectic IS kernel.

    Formula (DFK.nb `is[]`):
        I_S = (1/2)·(−1)^{m1}·(qq^{m1} + qq^{−m1}) · ẽI_S(m1, e1,   m2, e2)
            − (1/2)·(−1)^{m1} · ẽI_S(m1, e1−1, m2, e2)
            − (1/2)·(−1)^{m1} · ẽI_S(m1, e1+1, m2, e2)

    The (1/2) prefactor: empirically the combination always yields integer
    coefficients.  We use Fraction arithmetic and assert integrality at
    the end (cheap sanity check).

    Returns
    -------
    QEtaSeries : dict[(qq_power, eta_exp) → Fraction]
        All Fraction values have denominator 1 (i.e. are integers).
        Returns {} if the integrality conditions for ẽI_S fail.
    """
    ei_center = _etilde_is(m1, e1,     m2, e2, qq_order, eta_order)
    ei_minus  = _etilde_is(m1, e1 - 1, m2, e2, qq_order, eta_order)
    ei_plus   = _etilde_is(m1, e1 + 1, m2, e2, qq_order, eta_order)

    sign_m1 = Fraction(1 if m1 % 2 == 0 else -1)
    half = Fraction(1, 2)

    result: QEtaSeries = {}

    # Term A: (1/2)·(−1)^{m1}·qq^{m1}  · ẽI_S(e1)
    # Term B: (1/2)·(−1)^{m1}·qq^{−m1} · ẽI_S(e1)
    factor_ab = half * sign_m1
    for (qq_p, eta), c in ei_center.items():
        scaled = c * factor_ab
        if scaled == 0:
            continue
        # Term A: shift by +m1
        new_qq_a = qq_p + m1
        if 0 <= new_qq_a <= qq_order:
            key = (new_qq_a, eta)
            v = result.get(key, Fraction(0)) + scaled
            if v == 0:
                result.pop(key, None)
            else:
                result[key] = v
        # Term B: shift by −m1
        new_qq_b = qq_p - m1
        if 0 <= new_qq_b <= qq_order:
            key = (new_qq_b, eta)
            v = result.get(key, Fraction(0)) + scaled
            if v == 0:
                result.pop(key, None)
            else:
                result[key] = v

    # Terms C and D: −(1/2)·(−1)^{m1} · ẽI_S(e1±1)
    factor_cd = -half * sign_m1
    for src_series in (ei_minus, ei_plus):
        for (qq_p, eta), c in src_series.items():
            scaled = c * factor_cd
            if scaled == 0:
                continue
            if not (0 <= qq_p <= qq_order):
                continue
            key = (qq_p, eta)
            v = result.get(key, Fraction(0)) + scaled
            if v == 0:
                result.pop(key, None)
            else:
                result[key] = v

    return result


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
            if 0 <= new_qq_a <= qq_order:
                key = (new_qq_a, eta)
                v = result.get(key, Fraction(0)) + scaled
                if v == 0:
                    result.pop(key, None)
                else:
                    result[key] = v
            # qq^{−phase} shift
            new_qq_b = qq_p - phase
            if 0 <= new_qq_b <= qq_order:
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
    """Add two MultiEtaSeries (non-destructive)."""
    result: MultiEtaSeries = dict(a)
    for key, val in b.items():
        new_val = result.get(key, Fraction(0)) + val
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
        Keys: ``(qq_power, cusp_eta_exp)``
    multi_series : MultiEtaSeries
        Keys: ``(qq_power, 2η_0, …, 2η_{k-1}, cusp_eta_exp)``
    qq_order : int or None
        Truncation cutoff.

    Returns
    -------
    MultiEtaSeries with the same key structure as *multi_series*.
    """
    result: MultiEtaSeries = {}
    for (qq_is, eta_is), c_is in is_series.items():
        for multi_key, c_multi in multi_series.items():
            new_qq = qq_is + multi_key[0]
            if qq_order is not None and new_qq > qq_order:
                continue
            # Keep hard-η dims unchanged, add cusp-η exponents
            new_key = (new_qq,) + multi_key[1:-1] + (multi_key[-1] + eta_is,)
            new_val = result.get(new_key, Fraction(0)) + c_is * c_multi
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
) -> MultiEtaSeries:
    """Apply unrefined K(k, 1; m, e) factor to a MultiEtaSeries.

    Identical logic to ``_apply_k1_factor`` but operates on multi-
    dimensional keys.  Only the qq_power (first element) is shifted;
    all η dimensions are untouched.

    Parameters
    ----------
    truncate : bool
        If True (default), keep only terms with ``0 ≤ new_qq ≤ qq_order``.
        If False, skip the bounds check — used when building
        manifold-independent kernel tables so that the deferred
        truncation happens after convolution with I^ref.
    """
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
            if not truncate or 0 <= new_qq_a <= qq_order:
                new_key = (new_qq_a,) + rest
                v = result.get(new_key, Fraction(0)) + scaled
                if v == 0:
                    result.pop(new_key, None)
                else:
                    result[new_key] = v
            # -phase shift
            new_qq_b = qq_p - phase
            if not truncate or 0 <= new_qq_b <= qq_order:
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
) -> MultiEtaSeries:
    """Convert a RefinedIndexResult to MultiEtaSeries.

    Parameters
    ----------
    refined : RefinedIndexResult
        Keys: ``(qq_power, 2η_0, …, 2η_{k-1})``
    append_cusp_eta : bool
        If True, append a ``cusp_eta = 0`` dimension to every key
        (needed for ℓ ≥ 2 before IS convolution steps).

    Returns
    -------
    MultiEtaSeries with Fraction values.
    """
    result: MultiEtaSeries = {}
    for key, coeff in refined.items():
        if coeff == 0:
            continue
        new_key = key + (0,) if append_cusp_eta else key
        result[new_key] = Fraction(coeff)
    return result


# ---------------------------------------------------------------------------
# Part 7 — Single-step IS convolution
# ---------------------------------------------------------------------------


def _apply_is_step(
    state: dict[tuple[int, Fraction], MultiEtaSeries],
    k_current: int,
    k_next: int,
    qq_order: int,
    eta_order: int,
    m1_range: int,
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
        next kernel step); used to enumerate which (m1, e1) are relevant.
    qq_order : int
    eta_order : int
    m1_range : int
        Scan |m1| ≤ m1_range for intermediate variables.

    Returns
    -------
    new_state : dict[(int, Fraction) → MultiEtaSeries]
    """
    new_state: dict[tuple[int, Fraction], MultiEtaSeries] = {}

    # Enumerate ALL candidate (m1, e1) pairs from the K(k_next, 1) support,
    # including both ±t for c=0 and c=−2.  The full enumeration is required
    # because the IS kernel breaks the (m,e)→(−m,−e) symmetry that the
    # ℓ=1 path's multiplicity trick relies on.
    m1_terms = _enumerate_slope1_all(k_next, m1_range)

    for (m, e), src_series in state.items():
        if not src_series:
            continue

        # e-transform: first argument transforms as −e − k_current/2·m
        e_in = -e - Fraction(k_current * m, 2)

        for m1, e1, _, _ in m1_terms:
            is_val = _is_kernel(m, e_in, m1, e1, qq_order, eta_order)
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
    """Result of refined Dehn filling I^ref_{P/Q}(η_hard, [η_cusp, …]).

    The series carries:
    - For ℓ=1 (|Q|=1, no IS kernel):
        key = (qq_power, 2*η_0_exp, …, 2*η_{k-1}_exp)
        Only hard-edge fugacities; no cusp η.
    - For ℓ≥2 (IS kernel chain):
        key = (qq_power, 2*η_0_exp, …, 2*η_{k-1}_exp, cusp_eta_exp)
        Hard-edge fugacities + one cusp η from the IS chain.
    - For multi-cusp sequential filling (two fillings with ℓ≥2):
        key = (qq, 2*η_0, …, 2*η_{k-1}, cusp_eta_0, cusp_eta_1)
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
        """Set η_j = 1 (v_j = 0) for the given hard-edge indices.

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
) -> FilledRefinedResult:
    """Compute the refined Dehn-filled index I^ref_{P/Q}(η_hard, η_cusp).

    Applies the refined Dehn filling kernel K^ref(P,Q; m,e; η_cusp) to the
    refined 3D index I^ref(m,e; η_hard) summed over contributing (m,e) pairs.

    Algorithm
    ---------
    1. Compute the HJ-CF k = [k_1, …, k_ℓ] for P/Q.
    2. If ℓ = 1: K^ref = K(k_1, 1; ·) (unrefined kernel, no IS chain).
       Sum K(k_1,1; m,e) · I^ref(m,e; η_hard) over kernel support.
       Result has only hard-edge η's.
    3. If ℓ ≥ 2:
       a. Scan ALL (m,e) with non-zero I^ref; initialise state with
          I^ref(m,e; η_hard) ⊗ η_cusp^0.
       b. Apply ℓ−1 IS convolution steps (IS kernel multiplies into cusp-η).
       c. Apply the final unrefined K(k_ℓ, 1; ·) to the last state.
       Result has hard-edge η's + cusp η.
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
    # Step 3b: ℓ ≥ 2 — Grid scan of (m, e) with non-zero I^ref
    # ------------------------------------------------------------------
    # The IS kernel's n_eta summation (bounded by eta_order) must stay
    # well below qq_internal; otherwise tetrahedron-index truncation
    # at the qq_internal boundary produces spurious high-η entries
    # that K-factor phase shifts move into the user range.  We inflate
    # qq_internal beyond qq_order so that the IS convolution boundary
    # artifacts live above qq_order and are discarded at final truncation.
    _is_buffer = qq_order // 2 + 4
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
            multi = _refined_to_multi(refined, append_cusp_eta=True)
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
            src_series, c_final, phase_final, mult_final, qq_internal
        )
        total_series_ell2 = _multi_add(total_series_ell2, contribution)

        

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
    truncated: MultiEtaSeries = {
        k: v for k, v in total_series_ell2.items()
        if k[0] + abs(k[-1]) <= qq_order
    }

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
    _is_buffer = qq_order // 2 + 4
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
    # Key structure: (qq, 2*η_0, …, 2*η_{H-1}, cusp_eta_0, …, cusp_eta_{C})
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
        ``(qq, 2η_0, …, 2η_{H-1}, m_tag, e_x2_tag [, cusp_eta=0])``

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
        _is_buffer = qq_order // 2 + 4
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
        with keys ``(qq, 2η_0, …, 2η_{H-1} [, cusp_eta])``.
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
    _is_buffer = qq_order // 2 + 4
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