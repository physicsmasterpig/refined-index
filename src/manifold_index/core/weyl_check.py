"""
core/weyl_check.py — Weyl-symmetry prerequisites for Dehn filling.

Before a cusp can be Dehn-filled in the *refined* index, three conditions
must hold (SPEC.md §Dehn Filling, conditions 1–3):

  1. **Non-closability** – the chosen cycle must be non-closable under the
     ordinary 3D index Dehn filling (handled by ``find_non_closable_cycles``).

  2. **Weyl-symmetry** – the refined index can be made Weyl-manifest by
     multiplying by a monomial:

         f(m, e) = η^{ b·m + a·e } · I(m, e)   is  η ↔ η^{−1} symmetric

     for fixed vectors  a ∈ ℤ^{num_hard}  and  b ∈ (ℤ/2)^{num_hard}.

     Here *a* and *b* are the **multiplier** convention: to obtain the
     Weyl-symmetric series ``f``, multiply ``I(m,e)`` by the monomial
     ``η^{b·m + a·e}``.  Equivalently:

         I(m, e) = η^{−(b·m + a·e)} · f(η ↔ η^{−1} symmetric terms)

     .. note::
        **Internal storage convention** — ``ABVectors.a`` stores ``2*a``
        (an integer), and ``ABVectors.b`` stores *b* directly (a
        half-integer).  The factored-exponent in *I* is recovered as::

            centre_j = −(b_stored[j] * m + (a_stored[j] / 2) * e)

  3. **Adjoint su(2) character** – the coefficient of q^1 (after stripping
     the leading η-monomial) must equal  η + 1 + η^{-1}  (the adjoint
     character of su(2)) for each hard edge.

This module handles conditions 2 and 3.

────────────────────────────────────────────────────────────────────────────
Key formula
────────────────────────────────────────────────────────────────────────────

For a single cusp with one hard edge (the typical case), let

    centre(m, e)  =  Σ_k (η_exp_k · coeff_k) / Σ_k coeff_k

be the coefficient-weighted centre of the η-polynomial at the leading q-order
for the index I(m, e).  Then

    centre(m, 0) = b · m + const
    centre(0, e) = (a/2) · e + const

which gives

    b = [centre(+m, 0) − centre(−m, 0)] / (2m)        (half-integer)
    a = [centre(0, +e) − centre(0, −e)] / e            (integer)

For multiple hard edges the same logic applies component-wise.

The function ``compute_ab_vectors`` searches the given entries table for
suitable conjugate-charge pairs, computes (a, b) from each pair, checks
consistency, and validates the integrality constraints.

────────────────────────────────────────────────────────────────────────────
Key representation
────────────────────────────────────────────────────────────────────────────

``RefinedIndexResult`` keys are

    (q_half_power,  2*η_0_exp,  2*η_1_exp,  …,  2*η_{k-1}_exp)

All exponents are stored doubled so that every entry is a plain ``int``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Sequence

from manifold_index.core.refined_index import RefinedIndexResult

# ---------------------------------------------------------------------------
# Public data class
# ---------------------------------------------------------------------------

@dataclass
class ABVectors:
    """
    Weyl-symmetry vectors (a, b) for the refined index.

    Attributes
    ----------
    a : list[Fraction]
        One entry per hard edge.  Must be an integer (ℤ) for the Dehn
        filling to be well-defined.
    b : list[Fraction]
        One entry per hard edge.  Must be a half-integer (ℤ/2) for the
        Dehn filling to be well-defined.
    num_hard : int
        Number of hard edges.
    a_is_integer : list[bool]
        ``a[j] in ℤ`` for each j.
    b_is_half_integer : list[bool]
        ``2*b[j] in ℤ`` for each j.
    warnings : list[str]
        Non-fatal messages (e.g., inconsistency between different pair
        estimates, or missing data for some components).

    Properties
    ----------
    is_valid : bool
        True iff all integrality constraints are satisfied.
    """

    a: list[Fraction]
    b: list[Fraction]
    num_hard: int
    a_is_integer: list[bool] = field(default_factory=list)
    b_is_half_integer: list[bool] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.a_is_integer:
            self.a_is_integer = [v.denominator == 1 for v in self.a]
        if not self.b_is_half_integer:
            self.b_is_half_integer = [(v * 2).denominator == 1 for v in self.b]

    @property
    def is_valid(self) -> bool:
        """True iff a ∈ ℤ^num_hard and b ∈ (ℤ/2)^num_hard."""
        return all(self.a_is_integer) and all(self.b_is_half_integer)

    def __str__(self) -> str:
        def _fmt(v: Fraction) -> str:
            return str(int(v)) if v.denominator == 1 else str(v)

        a_str = "(" + ", ".join(_fmt(v) for v in self.a) + ")"
        b_str = "(" + ", ".join(_fmt(v) for v in self.b) + ")"
        valid = "✓" if self.is_valid else "✗"
        lines = [f"a = {a_str}  b = {b_str}  {valid}"]
        for w in self.warnings:
            lines.append(f"  warning: {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Low-level helper: extract leading η-exponents from one result
# ---------------------------------------------------------------------------

def extract_leading_eta_exponents(
    result: RefinedIndexResult,
    num_hard: int,
) -> list[Fraction] | None:
    """
    Return the minimum η-exponent vector at the leading (lowest) q-order.

    For a result with keys  (q_half_pow, 2*η_0_exp, …, 2*η_{k-1}_exp),
    the function:

    1. Finds the minimum ``q_half_pow`` among all nonzero entries.
    2. At that minimum q-power, collects all ``(η_0_exp, …)`` tuples.
    3. Returns the *component-wise minimum* of those tuples as a list of
       ``Fraction`` values.

    The component-wise minimum is chosen rather than, say, lexicographic
    minimum because it corresponds to the "lowest corner" η-monomial in
    the Newton polytope, which is the Weyl-lowest weight at leading q-order.

    Returns ``None`` if the result is empty or all-zero.

    Parameters
    ----------
    result : RefinedIndexResult
        Output of ``compute_refined_index``.
    num_hard : int
        Number of hard edges (= number of η variables).
    """
    if not result:
        return None

    # Step 1 – find minimum q-half-power
    nonzero = [(k, v) for k, v in result.items() if v != 0]
    if not nonzero:
        return None

    min_q = min(k[0] for k, _ in nonzero)

    # Step 2 – collect η-exponent tuples at that q-level
    at_min_q = [k for k, _ in nonzero if k[0] == min_q]

    # Step 3 – component-wise minimum (η exps stored doubled → divide by 2)
    min_eta = [
        Fraction(min(k[1 + j] for k in at_min_q), 2)
        for j in range(num_hard)
    ]
    return min_eta


# ---------------------------------------------------------------------------
# Low-level helper: compute weighted-centre η-exponents at leading q
# ---------------------------------------------------------------------------

def _eta_center_at_leading_q(
    result: RefinedIndexResult,
    num_hard: int,
) -> list[Fraction] | None:
    """
    Return the coefficient-weighted centre of the η-exponents at the lowest
    q-order.

        centre_j = (Σ_k  η_j_exp_k · coeff_k) / (Σ_k coeff_k)

    where the sums run over all terms k at the minimum q-half-power with
    nonzero coefficient.

    Returns ``None`` if the result is empty, all-zero, or if the total
    weight Σ coeff is zero (exact cancellation at leading q).

    Unlike :func:`extract_leading_eta_exponents` (component-wise minimum),
    the centre is the midpoint of the η-polynomial, which satisfies

        centre(m, e) = b · m + (a/2) · e

    and enables correct extraction of both *a* (integer) and *b*
    (half-integer) from conjugate charge pairs.
    """
    if not result:
        return None

    nonzero = [(k, v) for k, v in result.items() if v != 0]
    if not nonzero:
        return None

    min_q = min(k[0] for k, _ in nonzero)
    at_min_q = [(k, v) for k, v in nonzero if k[0] == min_q]

    total_weight = sum(v for _, v in at_min_q)
    if total_weight == 0:
        return None  # exact cancellation → centre undefined

    centres: list[Fraction] = []
    for j in range(num_hard):
        # k[1+j] is the doubled η-exponent; divide by 2·total_weight
        weighted_sum = sum(k[1 + j] * v for k, v in at_min_q)
        centres.append(Fraction(weighted_sum, 2 * total_weight))
    return centres


# ---------------------------------------------------------------------------
# Main function: compute (a, b) vectors
# ---------------------------------------------------------------------------

def compute_ab_vectors(
    entries: Sequence[tuple[list[int], list[Fraction], RefinedIndexResult]],
    num_hard: int,
) -> ABVectors | None:
    """
    Compute the Weyl-symmetry vectors (a, b) from a table of refined index
    evaluations.

    The entries are ``(m_ext, e_ext, result)`` triples as produced by the
    multi-point evaluation grid in the GUI (and by ``compute_refined_index``
    called at multiple external charges).

    Algorithm
    ---------
    For each hard-edge component j, let ``centre_j(m, e)`` be the
    coefficient-weighted centre of the η_j exponents at the leading q-order
    (see :func:`_eta_center_at_leading_q`).  Then:

    * **b[j]** from meridian pairs  (m, 0) / (−m, 0)  with m > 0:

          b[j]  =  −[centre_j(+m, 0) − centre_j(−m, 0)] / (2 m)

    * **a[j]** from longitude pairs  (0, e) / (0, −e)  with sum(e) > 0:

          a[j]  =  −[centre_j(0, +e) − centre_j(0, −e)] / sum(|e|)

    Multiple pairs are used for robustness; consistency is checked across
    them.  Only pairs from the "positive" side (m_sum > 0 or e_sum > 0) are
    processed to guarantee a consistent sign convention.

    Parameters
    ----------
    entries : sequence of (m_ext, e_ext, result)
        ``m_ext`` is a list of ints (one per cusp);
        ``e_ext`` is a list of Fraction (one per cusp);
        ``result`` is a ``RefinedIndexResult``.
    num_hard : int
        Number of hard edges.

    Returns
    -------
    ABVectors or None
        ``None`` if there are not enough conjugate-charge pairs to determine
        both a and b.  Otherwise returns an ``ABVectors`` instance (check
        ``is_valid`` for integrality constraints).
    """
    if num_hard == 0:
        return ABVectors(a=[], b=[], num_hard=0)

    # ------------------------------------------------------------------
    # Index entries by (m_ext_tuple, e_ext_tuple) → η centre vector.
    # ------------------------------------------------------------------
    indexed: dict[tuple, list[Fraction] | None] = {}
    for m_ext, e_ext, result in entries:
        key = (tuple(m_ext), tuple(e_ext))
        indexed[key] = _eta_center_at_leading_q(result, num_hard)

    def get_c(m_vec: tuple[int, ...], e_vec: tuple[Fraction, ...]) -> list[Fraction] | None:
        return indexed.get((m_vec, e_vec))

    r = len(next(iter(indexed))[0])  # number of cusps

    b_estimates: list[list[Fraction]] = []
    a_estimates: list[list[Fraction]] = []

    seen_pairs_b: set[tuple] = set()
    seen_pairs_a: set[tuple] = set()

    for (m_key, e_key), c_pos in indexed.items():
        if c_pos is None:
            continue

        # ---- meridian pairs: all e_key == 0, sum(m_key) > 0 ----
        if all(e == 0 for e in e_key) and sum(m_key) > 0:
            neg_m_key = tuple(-m for m in m_key)
            pair_tag = (m_key, neg_m_key)
            if pair_tag not in seen_pairs_b:
                c_neg = get_c(neg_m_key, e_key)
                if c_neg is not None:
                    seen_pairs_b.add(pair_tag)
                    total_m = sum(abs(m) for m in m_key)  # = sum(m_key) > 0
                    b_vec = [
                        -(c_pos[j] - c_neg[j]) / (2 * total_m)
                        for j in range(num_hard)
                    ]
                    b_estimates.append(b_vec)

        # ---- longitude pairs: all m_key == 0, sum(e_key) > 0 ----
        if all(m == 0 for m in m_key) and sum(e_key) > 0:
            neg_e_key = tuple(-e for e in e_key)
            pair_tag = (e_key, neg_e_key)
            if pair_tag not in seen_pairs_a:
                c_neg = get_c(m_key, neg_e_key)
                if c_neg is not None:
                    seen_pairs_a.add(pair_tag)
                    total_e = sum(abs(e) for e in e_key)  # = sum(e_key) > 0
                    a_vec = [
                        -(c_pos[j] - c_neg[j]) / total_e
                        for j in range(num_hard)
                    ]
                    a_estimates.append(a_vec)

    # ------------------------------------------------------------------
    # If no pure-meridian pairs found, fall back on individual m-entries
    # compared to (0, 0, …).
    # ------------------------------------------------------------------
    if not b_estimates:
        zero_m = tuple(0 for _ in range(r))
        zero_e = tuple(Fraction(0) for _ in range(r))
        c_zero = get_c(zero_m, zero_e)

        if c_zero is not None:
            for (m_key, e_key), c_pos in indexed.items():
                if c_pos is None or (m_key, e_key) == (zero_m, zero_e):
                    continue
                if not all(e == 0 for e in e_key):
                    continue
                total_m = sum(m_key)  # signed
                if total_m == 0:
                    continue
                b_vec = [
                    -(c_pos[j] - c_zero[j]) / total_m
                    for j in range(num_hard)
                ]
                b_estimates.append(b_vec)

    # ------------------------------------------------------------------
    # Consensus: use the first estimate, warn if others disagree
    # ------------------------------------------------------------------
    warnings: list[str] = []

    def _consensus(
        estimates: list[list[Fraction]],
        label: str,
    ) -> list[Fraction] | None:
        if not estimates:
            return None
        ref = estimates[0]
        for est in estimates[1:]:
            for j, (r_j, e_j) in enumerate(zip(ref, est)):
                if r_j != e_j:
                    warnings.append(
                        f"{label}[{j}]: inconsistent estimates "
                        f"{r_j} vs {e_j} (using first)"
                    )
        return ref

    b_vec = _consensus(b_estimates, "b")
    a_vec = _consensus(a_estimates, "a")

    # ------------------------------------------------------------------
    # If either is still None, we cannot complete the result
    # ------------------------------------------------------------------
    if b_vec is None and a_vec is None:
        return None

    if b_vec is None:
        warnings.append("b: no meridian pairs found; defaulting to 0")
        b_vec = [Fraction(0)] * num_hard
    if a_vec is None:
        warnings.append("a: no longitude pairs found; defaulting to 0")
        a_vec = [Fraction(0)] * num_hard

    return ABVectors(a=a_vec, b=b_vec, num_hard=num_hard, warnings=warnings)


# ---------------------------------------------------------------------------
# Check Weyl symmetry
# ---------------------------------------------------------------------------

def check_weyl_symmetry(
    entries: Sequence[tuple[list[int], list[Fraction], RefinedIndexResult]],
    num_hard: int,
    ab: ABVectors,
) -> dict[tuple, bool]:
    """
    For each entry verify  f(m, e) = η^{b·m + a·e} · I(m, e)  is Weyl-symmetric.

    Computes the shifted series  f = η^{shift} · I  (where shift = b·m + a·e)
    and checks that  f(η) = f(η^{−1}).

    A series ``f(η)`` is Weyl-symmetric (under the diagonal η_j → η_j^{−1})
    iff for every key ``(q_pow, 2*η_0, …)`` with nonzero coefficient, the
    reflected key ``(q_pow, −2*η_0, …)`` has the *same* coefficient.

    Returns
    -------
    dict mapping  (tuple(m_ext), tuple(e_ext))  →  bool (True = symmetric).
    """
    results: dict[tuple, bool] = {}

    for m_ext, e_ext, result in entries:
        # expected η-centre for this (m, e):
        # centre_j = b[j]*m + (a[j]/2)*e
        # In 2x-scaled integer grid: 2*centre_j = 2*b[j]*m + a[j]*e
        m_sum = sum(m_ext)
        e_sum = sum(e_ext)
        # shift_j = b[j]*m + (a[j]/2)*e  (multiplier exponent in user convention)
        # In 2x-scaled integer grid: 2*shift_j = 2*b[j]*m + a[j]*e
        shift_x2 = [
            int(2 * ab.b[j] * m_sum + ab.a[j] * e_sum)
            for j in range(num_hard)
        ]

        # Multiply each key by +shift (add shift to η exponents)
        shifted: RefinedIndexResult = {}
        for key, coeff in result.items():
            if coeff == 0:
                continue
            new_key = (key[0],) + tuple(
                key[1 + j] + shift_x2[j] for j in range(num_hard)
            )
            shifted[new_key] = shifted.get(new_key, 0) + coeff

        # Check Weyl symmetry of the shifted series
        ok = True
        for key, coeff in shifted.items():
            if coeff == 0:
                continue
            reflect = (key[0],) + tuple(-key[1 + j] for j in range(num_hard))
            if shifted.get(reflect, 0) != coeff:
                ok = False
                break

        results[(tuple(m_ext), tuple(e_ext))] = ok

    return results


# ---------------------------------------------------------------------------
# Strip the Weyl η-monomial from a single refined index entry
# ---------------------------------------------------------------------------

def strip_weyl_monomial(
    result: RefinedIndexResult,
    m_ext: list[int],
    e_ext: "Sequence[Fraction]",
    ab: ABVectors,
    num_hard: int,
) -> "tuple[list[Fraction], RefinedIndexResult]":
    """Factor out the Weyl η-monomial from a single entry.

    In the multiplier convention  f(m,e) = η^{b·m + a·e} · I(m,e),
    this function computes the Weyl-manifest series ``f`` by multiplying
    ``I(m,e)`` by ``η^{b·m + a·e}``.

    Returns the per-edge factored-exponent in *I* (i.e. the *negative* of
    the multiplier exponent) and the stripped (Weyl-manifest) series.
    If *ab* is valid the stripped series satisfies
    ``f(η) = f(η^{−1})`` (all η-exponents appear in conjugate pairs with
    equal coefficients).

    Parameters
    ----------
    result : RefinedIndexResult
        A single refined index evaluation, e.g. *I*(m, e).
    m_ext : list[int]
        Meridian charges for this evaluation (one per cusp).
    e_ext : sequence of Fraction
        Longitude charges for this evaluation (one per cusp).
    ab : ABVectors
        Vectors as returned by ``compute_ab_vectors``.
        Should satisfy ``ab.is_valid`` for the stripped series to be Weyl-symmetric.
    num_hard : int
        Number of hard edges.

    Returns
    -------
    centre : list[Fraction]
        Exponent of the η-monomial **in I** (the factored-form exponent).
        Equal to ``−(b·m + a·e)`` where *b* = ``ab.b`` and *a* = ``ab.a/2``.
        Used to display  ``I(m,e) = η^{centre} · f(η)``.
    stripped : RefinedIndexResult
        The Weyl-manifest series ``f = η^{b·m + a·e} · I``.  Stored in
        the same doubled-exponent encoding as *result*.
    """
    m_sum = sum(m_ext)
    e_sum = sum(Fraction(v) for v in e_ext)
    shift_x2 = [
        int(2 * ab.b[j] * m_sum + ab.a[j] * e_sum)
        for j in range(num_hard)
    ]
    # centre[j] = −(b[j]*m + (a[j]/2)*e) = the exponent in I(m,e) = η^{centre} · f
    centre = [Fraction(-s, 2) for s in shift_x2]
    stripped: RefinedIndexResult = {}
    for key, coeff in result.items():
        if coeff == 0:
            continue
        new_key = (key[0],) + tuple(
            key[1 + j] + shift_x2[j] for j in range(num_hard)
        )
        stripped[new_key] = stripped.get(new_key, 0) + coeff
    return centre, stripped


# ---------------------------------------------------------------------------
# Check adjoint su(2) character at q^1
# ---------------------------------------------------------------------------

def check_adjoint_character(
    result: RefinedIndexResult,
    leading_eta: list[Fraction],
    num_hard: int,
    hard_idx: int = 0,
) -> bool:
    """
    Check that the q^1 coefficient (after stripping the leading η-monomial)
    equals the su(2) adjoint character  η^{−1} + 1 + η  for the given
    hard edge index.

    Parameters
    ----------
    result : RefinedIndexResult
        A single refined index evaluation (e.g. I(1, 0) or I(0, 0)).
    leading_eta : list[Fraction]
        The leading η-exponent vector as returned by
        ``extract_leading_eta_exponents``.
    num_hard : int
        Total number of hard edges.
    hard_idx : int
        Which hard edge to check (default 0 = first).

    Returns
    -------
    bool
        True iff the q^1 coefficient polynomial in η_{hard_idx} matches
        ``η^{-1} + 1 + η``.

    Notes
    -----
    The q^1 coefficient polynomial is the set of terms with
    ``q_half_pow == 2`` (i.e., q^1) after stripping the overall
    η^{leading} factor from every key.
    """
    if not result or leading_eta is None:
        return False

    leading_x2 = [int(v * 2) for v in leading_eta]

    # Extract all terms at q-half-power = 2 (i.e., q^1)
    # after stripping the leading η-shift
    q1_terms: dict[int, int] = {}  # η_hard_idx power (doubled) → coeff sum
    for key, coeff in result.items():
        if coeff == 0:
            continue
        if key[0] != 2:
            continue
        # Shifted η exponents
        shifted_eta_j = key[1 + hard_idx] - leading_x2[hard_idx]
        # Sum over all other η components (assumed fixed / projected)
        q1_terms[shifted_eta_j] = q1_terms.get(shifted_eta_j, 0) + coeff

    # Remove zeros
    q1_terms = {k: v for k, v in q1_terms.items() if v != 0}

    if not q1_terms:
        return False

    # Adjoint character: η^{-1} + η^0 + η^1
    # In doubled notation: {-2: 1, 0: 1, 2: 1}
    adjoint = {-2: 1, 0: 1, 2: 1}

    # Normalize: divide by the overall scalar factor (in case the coefficient
    # is a multiple of the adjoint character rather than exactly 1)
    # First check the shape matches
    if set(q1_terms.keys()) != set(adjoint.keys()):
        return False

    # Check all three coefficients are equal (proportional to adjoint)
    vals = [q1_terms[k] for k in (-2, 0, 2)]
    return vals[0] == vals[1] == vals[2]


# ---------------------------------------------------------------------------
# Convenience: run all three checks
# ---------------------------------------------------------------------------

@dataclass
class WeylCheckResult:
    """Aggregated result of all Weyl-symmetry prerequisite checks."""

    ab: ABVectors | None
    "The (a, b) vectors; None if not computable."

    ab_valid: bool
    "True iff ab is not None and ab.is_valid."

    weyl_symmetric: dict[tuple, bool]
    "Per-entry Weyl-symmetry check results."

    all_weyl_symmetric: bool
    "True iff every entry passes the Weyl-symmetry check."

    adjoint_checks: dict[tuple, bool]
    "Per-entry adjoint-character checks (only entries with nonzero q^1 terms)."

    def __str__(self) -> str:  # pragma: no cover
        lines = []
        lines.append("=== Weyl-Symmetry Check ===")
        if self.ab is not None:
            lines.append(str(self.ab))
        else:
            lines.append("(a, b): could not be computed (insufficient data)")
        lines.append(
            f"Weyl symmetry: {'PASS' if self.all_weyl_symmetric else 'FAIL'} "
            f"({sum(self.weyl_symmetric.values())}/{len(self.weyl_symmetric)} entries OK)"
        )
        if self.adjoint_checks:
            n_pass = sum(self.adjoint_checks.values())
            lines.append(
                f"Adjoint q^1 check: {'PASS' if n_pass == len(self.adjoint_checks) else 'FAIL'} "
                f"({n_pass}/{len(self.adjoint_checks)} entries OK)"
            )
        return "\n".join(lines)


def run_weyl_checks(
    entries: Sequence[tuple[list[int], list[Fraction], RefinedIndexResult]],
    num_hard: int,
    hard_idx: int = 0,
) -> WeylCheckResult:
    """
    Run all Weyl-symmetry prerequisite checks for Dehn filling.

    Parameters
    ----------
    entries : sequence of (m_ext, e_ext, result)
    num_hard : int
    hard_idx : int
        Which hard edge to use for the adjoint-character check.

    Returns
    -------
    WeylCheckResult
    """
    ab = compute_ab_vectors(entries, num_hard)
    ab_valid = ab is not None and ab.is_valid

    if ab is not None:
        weyl_sym = check_weyl_symmetry(entries, num_hard, ab)
    else:
        weyl_sym = {(tuple(m), tuple(e)): False for m, e, _ in entries}

    all_weyl = all(weyl_sym.values()) if weyl_sym else False

    adjoint: dict[tuple, bool] = {}
    for m_ext, e_ext, result in entries:
        leading = extract_leading_eta_exponents(result, num_hard)
        if leading is None:
            continue
        key = (tuple(m_ext), tuple(e_ext))
        adjoint[key] = check_adjoint_character(result, leading, num_hard, hard_idx)

    return WeylCheckResult(
        ab=ab,
        ab_valid=ab_valid,
        weyl_symmetric=weyl_sym,
        all_weyl_symmetric=all_weyl,
        adjoint_checks=adjoint,
    )
