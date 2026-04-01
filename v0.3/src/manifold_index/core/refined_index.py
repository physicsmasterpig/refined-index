"""
core/refined_index.py — Refined Index computation  (Step 8).

See SPEC.md §Step 8 for the full mathematical specification.

────────────────────────────────────────────────────────────────────────────
What the refined index is
────────────────────────────────────────────────────────────────────────────

The refined index is the 3D index with one formal fugacity variable η_a
attached to each of the ``num_hard`` hard internal edges.  Hard edges occupy
rows  r … r+num_hard-1  of the position block of g_NZ, and therefore their
internal charges occupy positions  0 … num_hard-1  inside the ``e_int``
vector (which has total length n-r).

Formula (SPEC.md §Step 8):

    I^ref(q; η_0, …, η_{k-1}) =
        Σ_{e_int ∈ (½)Z^{n-r}}
            [ ∏_{a=0}^{k-1}  η_a^{e_{r+a}} ]
            · (−q^{½})^{ m · ν_p  −  e · ν_x }
            · ∏_{j=0}^{n-1} I_Δ( (g_NZ⁻¹ κ)_j ,  (g_NZ⁻¹ κ)_{n+j} )

where k = num_hard, and the κ / tet_arg assembly is identical to Step 4.

Output key convention
---------------------
    key  = (q_half_power,  2*η_0_exp,  2*η_1_exp,  …,  2*η_{k-1}_exp)
    value = integer coefficient

All fugacity exponents are half-integers, so multiplying by 2 gives integers
and the key is a plain tuple of ints.  Setting all η = 1 (projecting by
summing over all keys with the same q_half_power) exactly recovers the output
of ``compute_index_3d_python``.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence

from manifold_index.core.index_3d import (
    _tet_index_series,
    enumerate_summation_terms,
    _get_enum_state,
    _enumerate_with_state,
)
from manifold_index.core.neumann_zagier import NeumannZagierData

# Try to import C poly_convolve for faster polynomial multiplication
try:
    from manifold_index.core._c_tet_index import (       # type: ignore[import-not-found]
        poly_convolve as _c_poly_convolve,
    )
    _HAS_C_POLY = True
except ImportError:
    _HAS_C_POLY = False

# ---------------------------------------------------------------------------
# Public type alias
# ---------------------------------------------------------------------------

# key = (q_half_power, 2*η_0_exp, 2*η_1_exp, ..., 2*η_{k-1}_exp)
# value = integer coefficient
RefinedIndexResult = dict[tuple[int, ...], int]


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_refined_index(
    nz_data: NeumannZagierData,
    m_ext: Sequence[int],
    e_ext: Sequence[int | Fraction],
    q_order_half: int = 20,
) -> RefinedIndexResult:
    """Compute the refined index I^ref(q; η_0, …, η_{k-1}).

    This is a direct extension of ``compute_index_3d_python``.  For each
    summation term the first ``num_hard`` entries of ``e_int`` contribute
    fugacity weights  η_a^{e_{r+a}};  the remaining easy-edge charges have no
    fugacity and contribute to the q-series weight as usual.

    Parameters
    ----------
    nz_data : NeumannZagierData
        Output of ``build_neumann_zagier``.
    m_ext : sequence of int, length ``nz_data.r``
        Cusp meridian variables.
    e_ext : sequence of int or Fraction, length ``nz_data.r``
        Cusp longitude/2 variables.
    q_order_half : int
        Cutoff order in q^{1/2} (default 20 → up to q^10).

    Returns
    -------
    RefinedIndexResult
        ``dict`` mapping

            (q_half_power, 2*η_0_exp, 2*η_1_exp, …, 2*η_{k-1}_exp)
            →  integer coefficient

        The first entry of the key is the power of q^{1/2}.
        The next ``k = num_hard`` entries are ``2 × (fugacity exponent)``
        for each hard edge (always integers because every half-integer × 2
        is an integer).

        When ``num_hard = 0`` (all easy edges) the keys are length-1 tuples
        ``(q_half_power,)`` — the ordinary 3D index series.

    Notes
    -----
    - To recover the ordinary 3D index: sum all coefficients sharing the
      same ``q_half_power`` (equivalent to setting every η_a = 1).
      See :func:`project_to_3d_index`.
    - The series is truncated at ``q_order_half``; higher-degree terms are
      discarded.
    """
    k = nz_data.num_hard   # number of hard edges = number of fugacity variables
    terms = enumerate_summation_terms(nz_data, m_ext, e_ext, q_order_half)

    if not terms:
        return {}

    result: RefinedIndexResult = {}

    for term in terms:
        phase_exp: int = term["phase_exp"]
        tet_args: list[tuple[int, int]] = term["tet_args"]
        e_int_strs: list[str] = term["e_int"]

        # --- Extract hard-edge fugacity exponents (first k entries of e_int) ---
        # Each e_int entry is stored as a "p/q" Fraction string.
        # We store  2 * η_a_exponent  so the key stays a tuple of plain ints.
        eta_exps_x2 = tuple(
            int(Fraction(e_int_strs[a]) * 2)
            for a in range(k)
        )

        # --- Compute the q-series contribution from this term ---
        # Tighter per-tet cutoffs: each _tet_index_series call only needs to
        # reach (budget - min power already accumulated in prod).
        budget = q_order_half - phase_exp
        prod: dict[int, int] = {0: 1}
        prod_min_pow = 0  # lower bound on min(prod.keys())
        for ta, tb in tet_args:
            cutoff = budget - prod_min_pow
            if cutoff < 0:
                prod = {}
                break
            s = _tet_index_series(ta, tb, cutoff)
            if not s:
                prod = {}
                break
            # Use C poly_convolve when available (same as compute_index_3d_python)
            if _HAS_C_POLY:
                prod = _c_poly_convolve(prod, s, budget)
            else:
                new_prod: dict[int, int] = {}
                for p1, c1 in prod.items():
                    for p2, c2 in s.items():
                        pp = p1 + p2
                        if pp <= budget:
                            new_prod[pp] = new_prod.get(pp, 0) + c1 * c2
                prod = {kk: vv for kk, vv in new_prod.items() if vv != 0}
            prod_min_pow += min(s.keys())  # accumulate tet's min contribution

        # Apply phase factor (-q^{1/2})^{phase_exp}
        sign = 1 if phase_exp % 2 == 0 else -1
        for pp, c in prod.items():
            shifted = pp + phase_exp
            if 0 <= shifted <= q_order_half:
                key = (shifted,) + eta_exps_x2
                result[key] = result.get(key, 0) + sign * c

    # Remove zero entries
    return {key: val for key, val in result.items() if val != 0}


# ---------------------------------------------------------------------------
# Batch computation — pre-compute manifold state once, reuse across entries
# ---------------------------------------------------------------------------

def compute_refined_index_batch(
    nz_data: NeumannZagierData,
    entries: list[tuple[Sequence[int], Sequence[int | Fraction]]],
    q_order_half: int = 20,
) -> list[RefinedIndexResult]:
    """Compute I^ref for multiple (m_ext, e_ext) pairs, sharing manifold setup.

    This is equivalent to calling :func:`compute_refined_index` for each
    entry individually, but **much** faster because the manifold-dependent
    pre-computation (g_NZ inverse, valid half-integer patterns, internal-edge
    columns, etc.) is done once and reused for all entries.

    Parameters
    ----------
    nz_data : NeumannZagierData
    entries : list of (m_ext, e_ext) tuples
    q_order_half : int
        Cutoff order in q^{1/2}.

    Returns
    -------
    list of RefinedIndexResult
        One result per entry, in the same order.
    """
    # Pre-compute enumeration state once
    state = _get_enum_state(nz_data)
    k = nz_data.num_hard

    results: list[RefinedIndexResult] = []
    for m_ext, e_ext in entries:
        # Enumerate using pre-computed state (fast path)
        terms = _enumerate_with_state(state, m_ext, e_ext, q_order_half)
        if not terms:
            results.append({})
            continue

        result: RefinedIndexResult = {}
        for term in terms:
            phase_exp: int = term["phase_exp"]
            tet_args: list[tuple[int, int]] = term["tet_args"]
            e_int_strs: list[str] = term["e_int"]

            eta_exps_x2 = tuple(
                int(Fraction(e_int_strs[a]) * 2) for a in range(k)
            )

            budget = q_order_half - phase_exp
            prod: dict[int, int] = {0: 1}
            prod_min_pow = 0
            for ta, tb in tet_args:
                cutoff = budget - prod_min_pow
                if cutoff < 0:
                    prod = {}
                    break
                s = _tet_index_series(ta, tb, cutoff)
                if not s:
                    prod = {}
                    break
                if _HAS_C_POLY:
                    prod = _c_poly_convolve(prod, s, budget)
                else:
                    new_prod: dict[int, int] = {}
                    for p1, c1 in prod.items():
                        for p2, c2 in s.items():
                            pp = p1 + p2
                            if pp <= budget:
                                new_prod[pp] = new_prod.get(pp, 0) + c1 * c2
                    prod = {kk: vv for kk, vv in new_prod.items() if vv != 0}
                prod_min_pow += min(s.keys())

            sign = 1 if phase_exp % 2 == 0 else -1
            for pp, c in prod.items():
                shifted = pp + phase_exp
                if 0 <= shifted <= q_order_half:
                    key = (shifted,) + eta_exps_x2
                    result[key] = result.get(key, 0) + sign * c

        results.append({key: val for key, val in result.items() if val != 0})

    return results


# ---------------------------------------------------------------------------
# Utility: project onto ordinary 3D index (set all η = 1)
# ---------------------------------------------------------------------------

def project_to_3d_index(refined: RefinedIndexResult) -> dict[int, int]:
    """Sum all fugacity monomials for each q-power (η_a = 1 for all a).

    Equivalent to substituting all η_a = 1 in the refined index.
    Returns a plain ``dict[q_half_power → coefficient]``.
    """
    out: dict[int, int] = {}
    for key, coeff in refined.items():
        q_pow = key[0]
        out[q_pow] = out.get(q_pow, 0) + coeff
    return {k: v for k, v in out.items() if v != 0}


# ---------------------------------------------------------------------------
# Utility: pretty-print as a Laurent series
# ---------------------------------------------------------------------------

def format_refined_index(
    refined: RefinedIndexResult,
    num_hard: int,
    q_var: str = "q",
    eta_vars: list[str] | None = None,
) -> str:
    """Return a human-readable string of the refined index series.

    Parameters
    ----------
    refined : RefinedIndexResult
    num_hard : int
        Number of hard edges (= number of fugacity variables).
    q_var : str
        Symbol to use for q (default ``"q"``).
    eta_vars : list of str, optional
        Ignored in the current convention (single variable η with
        charge labels ``v_0, v_1, …``).  Kept for API compatibility.

    Returns
    -------
    str
        Sum of monomials, e.g.
        ``"-q^(1/2)*η^(2·W_0)  +  3*q  +  q^(3/2)*η^(-2·W_0)"``

    Notes
    -----
    Fugacity convention:  ``η_a = η^{2W_a}``.  The stored ``exp_x2``
    equals ``2 × true_exponent``, so the factor is ``η^(exp_x2·W_a)``.
    """
    def _monomial(key: tuple[int, ...], coeff: int) -> str:
        q_pow = key[0]
        eta_pows_x2 = key[1:]

        factors: list[str] = []

        # q factor — q_pow is stored doubled, actual power = q_pow/2
        if q_pow != 0:
            if q_pow % 2 == 0:
                n = q_pow // 2
                if n == 1:
                    factors.append(q_var)
                else:
                    factors.append(f"{q_var}^{n}")
            else:
                factors.append(f"{q_var}^({q_pow}/2)")

        # η factors: η^(exp_x2·W_a)
        for a, exp_x2 in enumerate(eta_pows_x2):
            if exp_x2 == 0:
                continue
            if exp_x2 == 1:
                factors.append(f"η^(W_{a})")
            elif exp_x2 == -1:
                factors.append(f"η^(-W_{a})")
            else:
                factors.append(f"η^({exp_x2}·W_{a})")

        body = "*".join(factors) if factors else ""
        if not body:                      # constant term
            return str(coeff)
        if coeff == 1:
            return body
        if coeff == -1:
            return f"-{body}"
        return f"{coeff}*{body}"

    # Sort by q_power then fugacity powers for a stable display
    sorted_items = sorted(refined.items(), key=lambda kv: kv[0])
    parts: list[str] = []
    for k, v in sorted_items:
        term = _monomial(k, v)
        if not parts:
            parts.append(term)
        elif term.startswith("-"):
            parts.append(f"  -  {term[1:]}")
        else:
            parts.append(f"  +  {term}")
    return "".join(parts) if parts else "0"


# ---------------------------------------------------------------------------
# Utility: format a grid of evaluations (like the .nb output)
# ---------------------------------------------------------------------------

# Each entry: (m_ext values, e_ext values, RefinedIndexResult)
MultiPointEntry = tuple[list[int], list, RefinedIndexResult]


def format_multi_point_index(
    entries: list[MultiPointEntry],
    num_hard: int,
    q_var: str = "q",
    eta_vars: list[str] | None = None,
    *,
    show_zero: bool = False,
) -> str:
    """Format a list of (m_ext, e_ext, result) triples as a multi-section string.

    The output resembles the Mathematica notebook style::

        I(+0, +0)  =  1
        I(+2, +0)  =  -q^(1/2) + 3*q  + …
        I(-2, +0)  =  …
        I(+0, +1)  =  …

    Parameters
    ----------
    entries : list of (m_ext, e_ext, RefinedIndexResult)
        Sorted list of evaluation points and their refined-index results.
    num_hard : int
        Number of hard edges (= number of fugacity variables η).
    q_var : str
        Symbol used for the q variable (default ``"q^(1/2)"``).
    eta_vars : list[str], optional
        Ignored in the current convention (single variable η with charge
        labels v_0, v_1, …).  Kept for API compatibility.
    show_zero : bool
        If *True*, include entries whose result is the empty dict (= 0).
        Default *False*.

    Returns
    -------
    str
        Multi-line string with one ``I(…) = …`` line per evaluation point.
    """
    if eta_vars is None:
        eta_vars = [f"η^(2·W_{a})" for a in range(num_hard)]

    lines: list[str] = []
    for m_ext, e_ext, result in entries:
        if not result and not show_zero:
            continue

        # Build the charge label, e.g. "(+2, -1)"
        charge_parts = []
        for m in m_ext:
            charge_parts.append(f"{m:+d}")
        for e in e_ext:
            frac = Fraction(e).limit_denominator(1000)
            if frac.denominator == 1:
                charge_parts.append(f"{int(frac):+d}")
            else:
                sign = "+" if frac > 0 else "-"
                charge_parts.append(f"{sign}{abs(frac)}")
        charge_label = "(" + ", ".join(charge_parts) + ")"

        series_str = format_refined_index(result, num_hard, q_var, eta_vars)
        lines.append(f"I{charge_label}  =  {series_str}")

    return "\n".join(lines) if lines else "(all evaluations are zero)"

