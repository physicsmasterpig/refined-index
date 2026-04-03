"""
core/dehn_filling.py — Dehn filling kernel and non-closable cycle search (Step 5).

See SPEC.md §Step 5 for the full mathematical specification.

────────────────────────────────────────────────────────────────────────────
Setup
────────────────────────────────────────────────────────────────────────────
For each cusp i the Neumann-Zagier basis uses:
  position variable : M_i   (meridian)
  momentum variable : Λ_i = L_i/2  (half-longitude)

Dehn filling kernel — slope P/Q (surgery along P·M + Q·L = P·pos + 2Q·mom):

  K(P, Q; m, e) = ½ (−1)^{Rm+2Se} ·
    [ δ_{Pm+2Qe, 0} · (q^{(Rm+2Se)/2} + q^{−(Rm+2Se)/2})
      − δ_{Pm+2Qe, −2}
      − δ_{Pm+2Qe, 2} ]

where R, S ∈ ℤ satisfy R·Q − P·S = 1, m ∈ ℤ, e ∈ (½)ℤ.

Filled 3D index:
  I_{P/Q}^{(i)}(m_other, e_other)
    = Σ_{m_i ∈ ℤ, e_i ∈ (½)ℤ}  K(P, Q; m_i, e_i) · I(m_all, e_all)

Non-closable cycle:
  The cycle P·M + Q·L at cusp i is *non-closable* if
  I_{P/Q}^{(i)} = 0  (identically, for all other-cusp variable values).
  In practice we test with m_other = e_other = 0 for each other cusp.

Summation bounds:
  For the c=0 family:  m_i = Q·t, e_i = −P·t/2, phase = t.
    Kernel contributes q^{t/2} and q^{−t/2}.
    Terms with min_degree(I_{3D}) ≤ q_order_half + |t|/2 can contribute.
  For the c=±2 families: kernel is constant (no q-power).
    Terms with min_degree(I_{3D}) ≤ q_order_half can contribute.

  Because min_degree(I_{3D}) grows at least quadratically in |t|,
  the contributing set is finite for any given q_order_half.
  We scan t until enumerate_summation_terms returns no terms for two
  consecutive |t| values (convex degree bound guarantees termination).
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from fractions import Fraction
from math import gcd
from typing import Sequence

from manifold_index.core.index_3d import (
    Index3DResult,
    compute_index_3d_python,
    enumerate_summation_terms,
    has_valid_summation_terms,
)
from manifold_index.core.neumann_zagier import NeumannZagierData

# Module-level summation term cache: keyed by content so it persists across
# slope evaluations for the same manifold.  Saves ~20× time in cycle searches.
_summation_term_cache: dict = {}


def clear_summation_cache() -> int:
    """Clear module-level summation term cache. Returns number of entries removed."""
    n = len(_summation_term_cache)
    _summation_term_cache.clear()
    return n


# ===========================================================================
# Part 1 — Extended Euclidean and R, S finder
# ===========================================================================

def _ext_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Return (g, x, y) with a·x + b·y = g = gcd(|a|, |b|)."""
    if b == 0:
        return a, 1, 0
    g, x1, y1 = _ext_gcd(b, a % b)
    return g, y1, x1 - (a // b) * y1


def find_rs(P: int, Q: int) -> tuple[int, int]:
    """Find R, S ∈ ℤ with R·Q − P·S = 1.

    Parameters
    ----------
    P, Q : int
        Coprime integers defining the slope P/Q.

    Returns
    -------
    (R, S) : tuple[int, int]
        Any solution.  Not unique; callers may reduce modulo (P, Q).

    Raises
    ------
    ValueError
        If gcd(|P|, |Q|) ≠ 1.
    """
    if gcd(abs(P), abs(Q)) != 1:
        raise ValueError(f"gcd(P={P}, Q={Q}) ≠ 1; slope must be primitive")

    # Strategy: solve |Q|·x₀ + |P|·y₀ = 1 with non-negative inputs
    # (so _ext_gcd's signed-% behaviour is never triggered), then correct
    # for signs: Q·(sign(Q)·x₀) + P·(sign(P)·y₀) = |Q|·x₀ + |P|·y₀ = 1.
    # Setting R = x, S = −y gives R·Q − P·S = Q·x + P·y = 1.
    _, x, y = _ext_gcd(abs(Q), abs(P))
    if Q < 0:
        x = -x
    if P < 0:
        y = -y
    return x, -y  # (R, S)


# ===========================================================================
# Part 2 — Kernel term enumeration
# ===========================================================================

@dataclass(frozen=True)
class KernelTerm:
    """One (m, e) summand in the Dehn filling kernel for a fixed slope P/Q.

    Attributes
    ----------
    m : int
        Meridian variable value (integer).
    e : Fraction
        Half-longitude variable value (integer or half-integer).
    c : int
        Value of P·m + 2Q·e  ∈ {−2, 0, 2}.
    phase : int
        R·m + 2S·e  (always an integer due to the integrality of the kernel).

    Kernel factor applied to I_{3D}(m, e)
    ──────────────────────────────────────
    c = 0 :  (½)·(−1)^phase · (q^{phase/2} + q^{−phase/2})
    c = ±2:  −(½)·(−1)^phase   (constant, no q-shift)
    """

    m: int
    e: Fraction
    c: int    # ∈ {-2, 0, 2}
    phase: int
    multiplicity: int = 1  # 2 for c=0, phase>0 (antipodal symmetry: t and -t contribute identically)


def _particular_solution(P: int, Q: int, c: int) -> tuple[int, Fraction]:
    """Find (m0, e0) with P·m0 + 2Q·e0 = c, m0 ∈ ℤ, e0 ∈ (½)ℤ.

    Sets f0 = 2·e0 (integer) and solves P·m0 + Q·f0 = c via extended GCD.
    Requires gcd(P, Q) = 1.
    """
    # P·x + Q·y = 1  →  P·(c·x) + Q·(c·y) = c
    _, x, y = _ext_gcd(P, Q)
    return c * x, Fraction(c * y, 2)


def enumerate_kernel_terms(
    P: int,
    Q: int,
    R: int,
    S: int,
    nz_data: NeumannZagierData,
    cusp_idx: int,
    m_other: Sequence[int],
    e_other: Sequence[int | Fraction],
    q_order_half: int,
    _summation_cache: dict | None = None,
) -> list[KernelTerm]:
    """Enumerate all kernel terms (m, e) that could contribute to I_{P/Q}.

    For each c ∈ {−2, 0, 2}:
      - particular solution (m_c, e_c) with P·m_c + 2Q·e_c = c
      - general family: m_t = m_c + Q·t,  e_t = e_c − P·t/2
      - phase_t = R·m_c + 2S·e_c + t

    Inclusion criterion (degree filter using pure-Python enumerate_summation_terms):
      - For c = 0:  min_degree(I_{3D}(m_t, e_t)) ≤ q_order_half + |phase_t|/2
      - For c = ±2: min_degree(I_{3D}(m_t, e_t)) ≤ q_order_half

    Scanning stops when two consecutive |t| steps (in each direction) produce
    no contributing terms, exploiting the convexity / growth of min_degree.

    Parameters
    ----------
    P, Q, R, S : int
        Slope and its auxiliary R, S with R·Q − P·S = 1.
    nz_data : NeumannZagierData
    cusp_idx : int
        Index of the cusp being filled (0-based).
    m_other, e_other : sequences of length r−1
        Values for the remaining cusps (used in degree filter).
    q_order_half : int
    _summation_cache : dict or None
        Optional output-parameter dict.  When provided, the summation terms
        computed during the degree filter are stored here under the key
        ``(m, e, adjusted_q)`` for every accepted kernel term.  The caller
        can then pass these to ``compute_index_3d_python`` to avoid a second
        ``enumerate_summation_terms`` call.

    Returns
    -------
    list of KernelTerm (deduplicated by (m, e) key)
    """
    r = nz_data.r
    assert 0 <= cusp_idx < r, f"cusp_idx={cusp_idx} out of range [0, {r})"

    def _make_ext(
        m_i: int, e_i: Fraction
    ) -> tuple[list[int], list[int | Fraction]]:
        """Build full m_ext / e_ext with cusp_idx slot set to (m_i, e_i)."""
        m_ext: list[int] = []
        e_ext: list[int | Fraction] = []
        other_m_iter = iter(m_other)
        other_e_iter = iter(e_other)
        for k in range(r):
            if k == cusp_idx:
                m_ext.append(m_i)
                e_ext.append(e_i)
            else:
                m_ext.append(next(other_m_iter))
                e_ext.append(next(other_e_iter))
        return m_ext, e_ext

    def _min_degree(
        m_i: int, e_i: Fraction, adjusted_q: int
    ) -> tuple[float, list[dict]]:
        """Min total tet-degree for I_{3D}; returns (inf, []) if no terms exist.

        Uses *adjusted_q* (no +50 buffer): any summation term with
        min_deg + phase_exp > adjusted_q contributes to I_{3D} at powers
        above adjusted_q, and after the kernel shift lands above q_order_half
        — so it is safely discarded by the final truncation.

        Fast path: uses has_valid_summation_terms() (~1 μs) to check integrality
        before running the full enumeration (~50–100 ms).  This avoids expensive
        enumeration for structurally-zero sectors.
        """
        m_ext, e_ext = _make_ext(m_i, e_i)
        if not has_valid_summation_terms(nz_data, m_ext, e_ext):
            return float("inf"), []
        terms = enumerate_summation_terms(nz_data, m_ext, e_ext, adjusted_q)
        if not terms:
            return float("inf"), terms
        return min(t["min_degree"] for t in terms), terms

    seen: set[tuple[int, Fraction]] = set()
    result: list[KernelTerm] = []
    _CONSEC_EMPTY_STOP = 2  # stop scanning after this many consecutive misses

    # We enumerate c ∈ {0, 2} only.
    #
    # c=0 antipodal symmetry (already applied):
    #   I_{3D}(m_t, e_t) = I_{3D}(m_{-t}, e_{-t}) and the kernel factor
    #   is identical for phase=+t and phase=-t (same sign, same q-shift sum),
    #   so we only enumerate the positive direction and account for the
    #   negative direction via multiplicity=2.
    #
    # c=±2 antipodal symmetry (new):
    #   _particular_solution(P,Q,-2) = -(m_{c2}, e_{c2}), so the c=-2 term at
    #   index t' is the antipodal (-m, -e) of the c=2 term at t = -t'.
    #   Since I_{3D}(-m,-e) = I_{3D}(m,e) (verified) and (-1)^{-phase}
    #   = (-1)^{phase}, every c=-2 contribution equals its paired c=2
    #   contribution exactly.  We therefore skip c=-2 entirely and set
    #   multiplicity=2 for all c=2 terms (the c=2 bijection covers ALL t,
    #   including t=0, whose partner is the distinct c=-2,t=0 term).
    for c in (0, 2):
        m_c, e_c = _particular_solution(P, Q, c)
        phase_c0_frac = R * m_c + 2 * S * e_c
        assert Fraction(phase_c0_frac).denominator == 1, (
            f"phase_c0 = {phase_c0_frac} not integer for P={P},Q={Q},c={c}"
        )
        phase_c0 = int(phase_c0_frac)

        signs = (1,) if c == 0 else (1, -1)
        for sign in signs:
            consec_degree_miss = 0    # consecutive degree misses (terms exist, too high)
            consec_integ_miss  = 0    # consecutive integrality misses (no valid charges)
            # Integrality misses are structural zeros (the NZ matrix constraints rule
            # out this (m_t, e_t)). They must NOT stop the scan early — the cancelling
            # term may lie beyond a run of them (e.g. m=2,e=10 for 5_1/1/0 requires
            # scanning past e=1/2,1,...,5/2 which are all integrality zeros).
            # However, an unbounded run of integrality misses means the whole family
            # is structurally zero; cap at q_order_half consecutive integrality misses.
            _INTEG_MISS_CAP = q_order_half
            t_abs = 0
            while t_abs <= 4 * q_order_half + 50:
                t = sign * t_abs
                # avoid double-counting t=0 when sign=-1
                if sign == -1 and t_abs == 0:
                    t_abs += 1
                    continue

                m_t = m_c + Q * t
                e_t = e_c - Fraction(P * t, 2)
                phase_t = phase_c0 + t

                # Adjusted degree bound: kernel shifts the series by ±phase/2
                adjusted_q = q_order_half + (abs(phase_t) if c == 0 else 0)

                md, sum_terms = _min_degree(m_t, e_t, adjusted_q)
                if md <= adjusted_q:
                    key = (m_t, e_t)
                    if key not in seen:
                        seen.add(key)
                        if _summation_cache is not None:
                            _summation_cache[(m_t, e_t, adjusted_q)] = sum_terms
                        result.append(
                            KernelTerm(
                                m=m_t, e=e_t, c=c, phase=phase_t,
                                multiplicity=2 if (c == 2 or (c == 0 and t_abs > 0)) else 1,
                            )
                        )
                    consec_degree_miss = 0
                    consec_integ_miss  = 0
                elif md < float("inf"):
                    # Degree miss: terms exist but all have degree > budget.
                    consec_degree_miss += 1
                    consec_integ_miss   = 0
                    if consec_degree_miss >= _CONSEC_EMPTY_STOP:
                        break
                else:
                    # Integrality miss: no valid NZ charges for this (m_t, e_t).
                    # Do NOT increment degree-miss counter. Use a separate cap.
                    consec_integ_miss += 1
                    if consec_integ_miss >= _INTEG_MISS_CAP:
                        break

                t_abs += 1

    return result


# ===========================================================================
# Part 3 — q-series arithmetic (dict-based, half-integer powers)
# ===========================================================================

# A q^{1/2}-series is a dict[int, Fraction]:
#   key k  →  coefficient of q^{k/2}
# Fraction coefficients handle the ½ factors in the kernel exactly.
# After all cancellations the filled index should have integer coefficients.

QSeries = dict[int, Fraction]


def _qseries_from_result(result: Index3DResult) -> QSeries:
    """Convert Index3DResult (list of int coeffs) to QSeries dict."""
    s: QSeries = {}
    for k, c in enumerate(result.coeffs):
        if c != 0:
            s[result.min_power + k] = Fraction(c)
    return s


def _qseries_shift(s: QSeries, power_shift: int) -> QSeries:
    """Multiply q-series by q^{power_shift/2} (add power_shift to all keys)."""
    return {k + power_shift: v for k, v in s.items()}


def _qseries_scale(s: QSeries, scalar: Fraction) -> QSeries:
    """Multiply all coefficients by scalar."""
    if scalar == 0:
        return {}
    return {k: v * scalar for k, v in s.items() if v * scalar != 0}


def _qseries_add(a: QSeries, b: QSeries) -> QSeries:
    """Add two q-series (in-place style, non-destructive)."""
    result: QSeries = dict(a)
    for k, v in b.items():
        new_val = result.get(k, Fraction(0)) + v
        if new_val == 0:
            result.pop(k, None)
        else:
            result[k] = new_val
    return result


def _qseries_truncate(s: QSeries, q_order_half: int) -> QSeries:
    """Keep only keys k ≤ q_order_half (i.e., powers up to q^{q_order_half/2})."""
    return {k: v for k, v in s.items() if k <= q_order_half}


def _apply_kernel(
    term: KernelTerm,
    index_series: QSeries,
    q_order_half: int | None = None,
) -> QSeries:
    """Compute K(P,Q; m, e) · I_{3D}(m, e) as a QSeries.

    c = 0 :  (½)·(−1)^phase · (q^{phase/2}·I + q^{−phase/2}·I)
    c = ±2:  −(½)·(−1)^phase · I

    When *q_order_half* is provided, the upward shift q^{+phase/2}·I is
    skipped for |phase| > q_order_half: every resulting key would exceed
    q_order_half and be discarded on the final truncation anyway.
    """
    # Use modular sign to stay integer — (-1)**negative gives float in Python.
    sign = Fraction(1 if term.phase % 2 == 0 else -1)
    half = Fraction(1, 2)

    if term.c == 0:
        b = _qseries_shift(index_series, -term.phase)
        if q_order_half is None or abs(term.phase) <= q_order_half:
            a = _qseries_shift(index_series, +term.phase)
            return _qseries_scale(_qseries_add(a, b), half * sign)
        # shift(+phase) lands entirely above q_order_half → skip it
        return _qseries_scale(b, half * sign)
    else:
        return _qseries_scale(index_series, -half * sign)


# ===========================================================================
# Part 4 — Filled index computation
# ===========================================================================

@dataclass
class FilledIndexResult:
    """Result of Dehn filling I_{P/Q}^{(cusp_idx)}.

    Attributes
    ----------
    P, Q : int
        Slope (physical cycle P·M + Q·L).
    cusp_idx : int
        Index of the filled cusp.
    series : QSeries
        q^{1/2}-series; key k → coefficient of q^{k/2}.
    q_order_half : int
        Series truncated at q^{q_order_half/2}.
    n_kernel_terms : int
        Number of distinct (m, e) kernel summands evaluated.
    """

    P: int
    Q: int
    cusp_idx: int
    series: QSeries
    q_order_half: int
    n_kernel_terms: int

    @property
    def is_zero(self) -> bool:
        """True if the truncated series is identically zero (no non-zero keys)."""
        return len(self.series) == 0

    def is_stably_zero(self, buffer: int | None = None) -> bool:
        """True if no stable (non-boundary) non-zero terms exist.

        For a truncated q-series, high-phase kernel terms produce
        downward-shifted contributions that land near ``q_order_half`` but
        whose cancellation partners (upward shifts) are above the cutoff and
        get truncated.  These *boundary artifacts* appear only in the top
        ``buffer`` powers.

        A term at power ``k`` is considered *stable* (reliable) only if
        ``k ≤ q_order_half − buffer``.  The series is declared stably zero
        if it has no stable non-zero term.

        Parameters
        ----------
        buffer : int, optional
            Number of powers near the cutoff to ignore.
            Default: ``min(max(5, q_order_half // 2), q_order_half - 1)``.
            The upper clamp ensures ``cutoff ≥ 1`` even when ``q_order_half``
            is very small (e.g. 4), preventing a negative cutoff that would
            cause every series to appear stably zero.

        Returns
        -------
        bool
        """
        if buffer is None:
            buffer = min(max(5, self.q_order_half // 2), self.q_order_half - 1)
        cutoff = self.q_order_half - buffer
        return not any(v != 0 for k, v in self.series.items() if k <= cutoff)

    def as_polynomial_string(self, var: str = "q") -> str:
        """Human-readable q^{1/2}-series string."""
        if not self.series:
            return "0"
        parts = []
        for k in sorted(self.series):
            c = self.series[k]
            if c == 0:
                continue
            if k == 0:
                parts.append(str(c))
            elif k % 2 == 0:
                pw = k // 2
                label = f"{var}^{pw}" if pw != 1 else var
                parts.append(f"{c}*{label}" if c != 1 else label)
            else:
                parts.append(f"{c}*{var}^({k}/2)")
        return " + ".join(parts) if parts else "0"


def compute_filled_index(
    nz_data: NeumannZagierData,
    cusp_idx: int,
    P: int,
    Q: int,
    m_other: Sequence[int] | None = None,
    e_other: Sequence[int | Fraction] | None = None,
    q_order_half: int = 20,
    verbose: bool = False,
    _t0: float | None = None,
) -> FilledIndexResult:
    """Compute the Dehn-filled 3D index I_{P/Q}^{(cusp_idx)}.

    Fills cusp ``cusp_idx`` with slope P/Q (physical cycle P·M + Q·L) by
    evaluating the Dehn filling kernel and calling compute_index_3d_python
    for each contributing (m_i, e_i) pair.

    Parameters
    ----------
    nz_data : NeumannZagierData
    cusp_idx : int
        Which cusp to fill (0-based).
    P, Q : int
        Coprime integers defining the slope.  Physical cycle: P·M + Q·L.
    m_other, e_other : sequences of length r−1, optional
        Values for the remaining cusps.  Defaults to all zeros.
    q_order_half : int
        Series cutoff (default 20).
    verbose : bool
        If True, print timestamped progress to stdout.
    _t0 : float, optional
        Reference timestamp for verbose output (uses time.time() if None).

    Returns
    -------
    FilledIndexResult
    """
    if _t0 is None:
        _t0 = time.time()

    def _log(msg: str) -> None:
        if verbose:
            print(f"  [{time.time() - _t0:8.3f}s] {msg}", flush=True)

    r = nz_data.r
    if m_other is None:
        m_other = [0] * (r - 1)
    if e_other is None:
        e_other = [0] * (r - 1)
    assert len(m_other) == r - 1, f"m_other length {len(m_other)} ≠ r-1={r-1}"
    assert len(e_other) == r - 1, f"e_other length {len(e_other)} ≠ r-1={r-1}"

    R, S = find_rs(P, Q)

    # ------------------------------------------------------------------
    # Step A: enumerate kernel terms.  The summation terms computed
    #         during the degree filter are cached in _summation_cache so
    #         Step B can reuse them (one enumerate_summation_terms call
    #         per kernel term total, instead of two).
    # ------------------------------------------------------------------
    _log(f"Step A: enumerate_kernel_terms(P={P}, Q={Q}, q={q_order_half}) …")
    t_a = time.time()
    _summation_cache: dict = _summation_term_cache
    kernel_terms = enumerate_kernel_terms(
        P, Q, R, S, nz_data, cusp_idx, m_other, e_other, q_order_half,
        _summation_cache=_summation_cache,
    )
    _log(f"Step A done: {len(kernel_terms)} kernel terms  ({time.time()-t_a:.3f}s)")

    def _make_ext(
        m_i: int, e_i: Fraction
    ) -> tuple[list[int], list[int | Fraction]]:
        m_ext: list[int] = []
        e_ext: list[int | Fraction] = []
        other_m_iter = iter(m_other)
        other_e_iter = iter(e_other)
        for k in range(r):
            if k == cusp_idx:
                m_ext.append(m_i)
                e_ext.append(e_i)
            else:
                m_ext.append(next(other_m_iter))
                e_ext.append(next(other_e_iter))
        return m_ext, e_ext

    total_series: QSeries = {}

    # ------------------------------------------------------------------
    # Step B: for each kernel term, compute I_{3D}(m, e) then apply
    #         the kernel factor and accumulate into the total series.
    #
    #   Sub-steps per term:
    #     B1  enumerate_summation_terms  (enumerate valid (e_int, tet_args))
    #     B2  _tet_index_series          (build each tet's qq-polynomial)
    #     B3  multiply tet polynomials   (convolve)
    #     B4  _apply_kernel              (shift / scale by kernel factor)
    #     B5  _qseries_add               (accumulate)
    # ------------------------------------------------------------------
    _log(f"Step B: computing I_{{3D}} for each of {len(kernel_terms)} kernel terms …")
    for i, kt in enumerate(kernel_terms):
        m_ext, e_ext = _make_ext(kt.m, kt.e)

        # For c=0 the kernel shifts the series by ±phase/2, so we need
        # extra headroom so that after shifting the terms still land within
        # [*, q_order_half].  For c=±2 the kernel is constant (no q-shift).
        index_q_order = q_order_half + (abs(kt.phase) if kt.c == 0 else 0)

        t_b = time.time()
        index_result = compute_index_3d_python(
            nz_data,
            m_ext=m_ext,
            e_ext=e_ext,
            q_order_half=index_q_order,
            _precomputed_terms=_summation_cache.get((kt.m, kt.e, index_q_order)),
        )
        index_series = _qseries_from_result(index_result)
        contribution = _apply_kernel(kt, index_series, q_order_half)
        if kt.multiplicity != 1:
            contribution = _qseries_scale(contribution, Fraction(kt.multiplicity))
        total_series = _qseries_add(total_series, contribution)

        _log(
            f"  term {i+1:3d}/{len(kernel_terms)}"
            f"  m={kt.m} e={kt.e} c={kt.c:+d} phase={kt.phase:+d}"
            f"  idx_q={index_q_order}"
            f"  I={dict(sorted(index_series.items())) if index_series else '{}'}"
            f"  ({time.time()-t_b:.3f}s)"
        )

    # Truncate to requested order
    total_series = _qseries_truncate(total_series, q_order_half)

    _log(f"Step B done. series={dict(sorted(total_series.items()))}")

    return FilledIndexResult(
        P=P,
        Q=Q,
        cusp_idx=cusp_idx,
        series=total_series,
        q_order_half=q_order_half,
        n_kernel_terms=len(kernel_terms),
    )


# ===========================================================================
# Part 5 — Non-closable cycle search
# ===========================================================================

@dataclass
class NonClosableCycle:
    """A non-closable cycle at a given cusp.

    The cycle is P·M + Q·L in the physical meridian/longitude basis,
    equivalently slope P/Q in the Dehn filling kernel
    (NZ position = M, momentum = L/2).

    Attributes
    ----------
    cusp_idx : int
    P, Q : int
        Physical cycle: P·M + Q·L.
    """

    cusp_idx: int
    P: int
    Q: int

    def __str__(self) -> str:
        return f"cusp {self.cusp_idx}: {self.P}·M + {self.Q}·L  (slope {self.P}/{self.Q})"


@dataclass
class NonClosableCycleResult:
    """Per-cusp non-closable cycle search results.

    Attributes
    ----------
    cusp_idx : int
    cycles : list[NonClosableCycle]
        Non-closable cycles found within the search range.
        Empty if none found.
    slopes_tested : list[tuple[int,int]]
        All (P, Q) slopes evaluated.
    """

    cusp_idx: int
    cycles: list[NonClosableCycle] = field(default_factory=list)
    slopes_tested: list[tuple[int, int]] = field(default_factory=list)


def _candidate_slopes(
    p_range: range,
    q_range: range,
    canonical_only: bool = False,
) -> list[tuple[int, int]]:
    """Return all primitive (P, Q) with P ∈ p_range, Q ∈ q_range, gcd=1.

    When Q = 0, only P > 0 is returned (since (P, 0) and (−P, 0) represent
    the same geometric slope up to orientation reversal; the convention is
    to use P > 0, i.e. 1/0 not −1/0).

    Parameters
    ----------
    p_range, q_range : range
    canonical_only : bool
        If True, return only one representative from each antipodal pair
        ``{(P, Q), (−P, −Q)}``, choosing the one where Q > 0 (or, when
        Q = 0, P > 0).  This halves the search space because
        I_{P/Q} = I_{−P/−Q} (the filled index is invariant under orientation
        reversal of the slope).
    """
    slopes = []
    seen: set[tuple[int, int]] = set()
    for P in p_range:
        for Q in q_range:
            if P == 0 and Q == 0:
                continue
            if gcd(abs(P), abs(Q)) != 1:
                continue
            # When Q = 0, (P, 0) and (−P, 0) are the same geometric slope
            # (opposite orientations of the meridian).  Always keep only P > 0
            # to avoid listing both 1/0 and −1/0.
            if Q == 0 and P < 0:
                continue
            if canonical_only:
                # Canonical representative: Q > 0, or (Q == 0 and P > 0).
                # The Q == 0 case is already enforced above; only need Q > 0 here.
                if Q < 0:
                    continue
            key = (P, Q)
            if key not in seen:
                seen.add(key)
                slopes.append((P, Q))
    return slopes


def find_non_closable_cycles(
    nz_data: NeumannZagierData,
    cusp_idx: int,
    p_range: range | None = None,
    q_range: range | None = None,
    m_other: Sequence[int] | None = None,
    e_other: Sequence[int | Fraction] | None = None,
    q_order_half: int = 20,
    use_symmetry: bool = True,
    verbose: bool = False,
) -> NonClosableCycleResult:
    """Search for non-closable cycles at ``cusp_idx`` over a range of slopes.

    For each primitive slope (P, Q) in p_range × q_range, computes the
    Dehn-filled index and checks if it is identically zero (within the
    q-series truncation order).  If so, P·M + Q·L is non-closable.

    Antipodal symmetry
    ------------------
    ``I_{P/Q} = I_{−P/−Q}`` because the filled index is invariant under
    orientation reversal of the slope.  When ``use_symmetry=True`` (default),
    only the *canonical* representative of each antipodal pair is computed
    (the one with first nonzero coordinate positive), and the result is
    automatically mirrored to its partner ``(−P, −Q)``.  This halves the
    number of expensive index evaluations.

    Parameters
    ----------
    nz_data : NeumannZagierData
    cusp_idx : int
    p_range : range, optional
        Range of P values.  Default: range(-3, 4).
    q_range : range, optional
        Range of Q values.  Default: range(0, 4).
    m_other, e_other : sequences, optional
        Other-cusp values.  Default: all zeros.
    q_order_half : int
        Truncation order (default 20).
    use_symmetry : bool
        Exploit (P,Q) ↔ (−P,−Q) symmetry to halve the search (default True).
    verbose : bool
        Print timestamped per-slope progress to stdout.

    Returns
    -------
    NonClosableCycleResult
    """
    if p_range is None:
        p_range = range(-3, 4)
    if q_range is None:
        q_range = range(0, 4)

    r = nz_data.r
    if m_other is None:
        m_other = [0] * (r - 1)
    if e_other is None:
        e_other = [0] * (r - 1)

    # All primitive slopes in the requested rectangle (full list for bookkeeping)
    all_slopes = _candidate_slopes(p_range, q_range, canonical_only=False)
    # Slopes we actually compute (canonical half or full list)
    compute_slopes = _candidate_slopes(p_range, q_range, canonical_only=use_symmetry)

    result = NonClosableCycleResult(cusp_idx=cusp_idx)
    result.slopes_tested = all_slopes[:]

    t0 = time.time()

    if verbose:
        sym_note = " (with antipodal symmetry)" if use_symmetry else ""
        print(
            f"[{0:8.3f}s] find_non_closable_cycles: "
            f"{len(all_slopes)} slopes in search range, "
            f"{len(compute_slopes)} to compute{sym_note}",
            flush=True,
        )

    # Map from canonical slope → is_non_closable bool
    computed: dict[tuple[int, int], bool] = {}

    for idx, (P, Q) in enumerate(compute_slopes):
        t_slope = time.time()
        if verbose:
            print(
                f"[{t_slope - t0:8.3f}s] slope {idx+1}/{len(compute_slopes)}"
                f"  ({P:+d},{Q:+d})  …",
                flush=True,
            )

        filled = compute_filled_index(
            nz_data,
            cusp_idx=cusp_idx,
            P=P,
            Q=Q,
            m_other=list(m_other),
            e_other=list(e_other),
            q_order_half=q_order_half,
            verbose=verbose,
            _t0=t0,
        )
        non_closable = filled.is_stably_zero()
        computed[(P, Q)] = non_closable

        if verbose:
            verdict = "NON-CLOSABLE" if non_closable else "closable"
            print(
                f"[{time.time() - t0:8.3f}s] slope ({P:+d},{Q:+d})"
                f"  {verdict}"
                f"  n_kernel={filled.n_kernel_terms}"
                f"  ({time.time()-t_slope:.3f}s this slope)",
                flush=True,
            )

        if non_closable:
            result.cycles.append(NonClosableCycle(cusp_idx=cusp_idx, P=P, Q=Q))

        # Mirror result to antipodal partner (−P, −Q) if it is in the
        # search range and was not already computed directly.
        if use_symmetry:
            neg = (-P, -Q)
            if neg in set(all_slopes) and neg not in computed:
                computed[neg] = non_closable
                if non_closable:
                    result.cycles.append(
                        NonClosableCycle(cusp_idx=cusp_idx, P=-P, Q=-Q)
                    )
                if verbose:
                    verdict = "NON-CLOSABLE" if non_closable else "closable"
                    print(
                        f"[{time.time() - t0:8.3f}s] slope ({-P:+d},{-Q:+d})"
                        f"  {verdict}  [by (P,Q)↔(−P,−Q) symmetry]",
                        flush=True,
                    )

    if verbose:
        print(
            f"[{time.time() - t0:8.3f}s] Done."
            f"  {len(result.cycles)} non-closable cycle(s) found.",
            flush=True,
        )

    return result
