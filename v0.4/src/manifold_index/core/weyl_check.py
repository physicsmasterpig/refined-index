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

  3. **Adjoint su(2) character** – define 𝒥_{q¹}(η, ũ; a, b) as the q¹
     coefficient of the generating function (2.60) at m̃ = 0.  The adjoint
     projection (2.61) integrates out η (extract η⁰), integrates unfilled
     cusp fugacities (extract u_i⁰), and integrates each filled cusp
     fugacity u_{n+J} against the SU(2) Haar measure weighted by the adjoint
     character.  The result must equal exactly −1 for each filled cusp I:

         𝒥_{q¹}|_{(adj su(2)_I)} = −1,    I = 1, …, d

     For a single cusp (d = 1, n = 0), this reduces to:

         ½(c_{−1} + c_{+1} − c_{−2} − c_{+2}) = −1

     where c_e = coeff(q¹, η⁰) of I^ref(m=0, e).

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

    The physical Weyl monomial is

    .. math::

        \\eta_j^{\\sum_I (a_I[j] \\cdot e_I + b_I[j] \\cdot m_I)}

    For **1-cusp** manifolds the coupling is scalar:
    ``shift_j = a[j]·e + b[j]·m``, and ``a[j]``, ``b[j]`` are plain
    ``Fraction`` values.

    For **multi-cusp** manifolds each cusp *I* contributes independently.
    The per-cusp columns are stored in :attr:`cusp_columns`; the flat
    ``a`` and ``b`` fields hold the **cusp-0 column** for display
    backward-compatibility.  All shift computations must use
    :meth:`shift_x2` which handles both cases correctly.

    An edge *j* is **compatible** with Dehn filling iff, for every
    cusp *I*, ``a_I[j] ∈ ℤ`` and ``2·b_I[j] ∈ ℤ``.

    Attributes
    ----------
    a : list[Fraction]
        One entry per hard edge.  For 1-cusp this is the full coupling;
        for multi-cusp it is the cusp-0 column (see :attr:`cusp_columns`).
    b : list[Fraction]
        One entry per hard edge (same convention as *a*).
    num_hard : int
        Number of hard edges.
    num_cusps : int
        Number of cusps (default 1 for backward compatibility).
    cusp_columns : list[ABVectors] or None
        For multi-cusp manifolds: one single-cusp ``ABVectors`` per cusp.
        ``None`` for single-cusp manifolds (where ``a``, ``b`` suffice).
    warnings : list[str]
        Non-fatal messages (e.g., inconsistency between different pair
        estimates, or missing data for some components).
    """

    a: list[Fraction]
    b: list[Fraction]
    num_hard: int
    num_cusps: int = 1
    cusp_columns: "list[ABVectors] | None" = field(default=None, repr=False)
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Shift computation (correct for any number of cusps)
    # ------------------------------------------------------------------

    def shift_x2(
        self,
        m_ext: "Sequence[int]",
        e_ext: "Sequence[Fraction]",
    ) -> list[int]:
        """Doubled Weyl shift for each hard edge.

        Returns ``[2·Σ_I (a_I[j]·e_I + b_I[j]·m_I)  for j in 0..num_hard-1]``.

        For 1-cusp manifolds (``cusp_columns is None``) this falls back to
        the simple ``2·(a[j]·Σe + b[j]·Σm)`` formula, which is identical
        since there is only one cusp.
        """
        if self.cusp_columns is not None:
            out: list[int] = []
            for j in range(self.num_hard):
                s = sum(
                    self.cusp_columns[I].a[j] * Fraction(e_ext[I])
                    + self.cusp_columns[I].b[j] * m_ext[I]
                    for I in range(self.num_cusps)
                )
                out.append(int(2 * s))
            return out
        # 1-cusp fast path
        e_sum = sum(Fraction(v) for v in e_ext)
        m_sum = sum(m_ext)
        return [
            int(2 * (self.a[j] * e_sum + self.b[j] * m_sum))
            for j in range(self.num_hard)
        ]

    # ------------------------------------------------------------------
    # Compatibility checks
    # ------------------------------------------------------------------

    @property
    def a_is_integer(self) -> list[bool]:
        """``a_I[j] ∈ ℤ`` for each j (across all cusps)."""
        if self.cusp_columns is not None:
            return [
                all(col.a[j].denominator == 1 for col in self.cusp_columns)
                for j in range(self.num_hard)
            ]
        return [v.denominator == 1 for v in self.a]

    @property
    def b_is_half_integer(self) -> list[bool]:
        """``2·b_I[j] ∈ ℤ`` for each j (across all cusps)."""
        if self.cusp_columns is not None:
            return [
                all((col.b[j] * 2).denominator == 1 for col in self.cusp_columns)
                for j in range(self.num_hard)
            ]
        return [(v * 2).denominator == 1 for v in self.b]

    @property
    def is_valid(self) -> bool:
        """True iff every edge is compatible: a ∈ ℤ and 2b ∈ ℤ."""
        return all(self.a_is_integer) and all(self.b_is_half_integer)

    @property
    def edge_compatible(self) -> list[bool]:
        """Per-edge compatibility with Dehn filling.

        Edge *j* is compatible iff ``a_I[j] ∈ ℤ`` and ``2·b_I[j] ∈ ℤ``
        for *every* cusp *I*.  Incompatible edges must be turned off:
        η_j = 1 (W_j = 0).
        """
        return [
            a_ok and b_ok
            for a_ok, b_ok in zip(self.a_is_integer, self.b_is_half_integer)
        ]

    def make_filling_compatible(self) -> "ABVectors":
        """Return a copy with incompatible edges zeroed out.

        For each hard edge j where :attr:`edge_compatible` is False,
        sets ``a_I[j] = 0`` and ``b_I[j] = 0`` for *every* cusp *I*
        (i.e., η_j = 1 during filling).  Compatible edges are kept
        unchanged.
        """
        mask = self.edge_compatible
        a_new = [
            self.a[j] if mask[j] else Fraction(0)
            for j in range(self.num_hard)
        ]
        b_new = [
            self.b[j] if mask[j] else Fraction(0)
            for j in range(self.num_hard)
        ]
        zeroed = [j for j in range(self.num_hard) if not mask[j]]
        new_warnings = list(self.warnings)
        if zeroed:
            new_warnings.append(
                f"Edges {zeroed} incompatible with half-integer e; "
                f"set η_j=1 (W_j=0) for filling"
            )
        # Zero out cusp columns too
        new_columns = None
        if self.cusp_columns is not None:
            new_columns = []
            for col in self.cusp_columns:
                new_columns.append(ABVectors(
                    a=[col.a[j] if mask[j] else Fraction(0)
                       for j in range(self.num_hard)],
                    b=[col.b[j] if mask[j] else Fraction(0)
                       for j in range(self.num_hard)],
                    num_hard=self.num_hard,
                ))
        return ABVectors(
            a=a_new, b=b_new, num_hard=self.num_hard,
            num_cusps=self.num_cusps, cusp_columns=new_columns,
            warnings=new_warnings,
        )

    # ------------------------------------------------------------------
    # Per-cusp accessors
    # ------------------------------------------------------------------

    def a_for_cusp(self, cusp_idx: int) -> list[Fraction]:
        """Column of a-vectors for one cusp: ``[a_I[0], a_I[1], …]``."""
        if self.cusp_columns is not None:
            return list(self.cusp_columns[cusp_idx].a)
        return list(self.a)

    def b_for_cusp(self, cusp_idx: int) -> list[Fraction]:
        """Column of b-vectors for one cusp."""
        if self.cusp_columns is not None:
            return list(self.cusp_columns[cusp_idx].b)
        return list(self.b)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        def _fmt(v: Fraction) -> str:
            return str(int(v)) if v.denominator == 1 else str(v)

        valid = "✓" if self.is_valid else "✗"

        if self.cusp_columns is not None:
            # Multi-cusp: show matrix
            lines = []
            for j in range(self.num_hard):
                a_parts = ", ".join(
                    _fmt(col.a[j]) for col in self.cusp_columns
                )
                b_parts = ", ".join(
                    _fmt(col.b[j]) for col in self.cusp_columns
                )
                lines.append(f"  edge {j}: a = ({a_parts})  b = ({b_parts})")
            header = f"Weyl vectors ({self.num_cusps} cusps, {self.num_hard} hard edges)  {valid}"
            return "\n".join([header] + lines + [f"  warning: {w}" for w in self.warnings])

        a_str = "(" + ", ".join(_fmt(v) for v in self.a) + ")"
        b_str = "(" + ", ".join(_fmt(v) for v in self.b) + ")"
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

        centre(m, e) = −(b · m + (a/2) · e)

    where (a, b) are the stored Weyl vectors.  The negative sign arises
    because the Weyl monomial  f = η^{b·m + (a/2)·e} · I^ref  cancels
    the centre shift, making f Weyl-symmetric.

    This enables correct extraction of both *a* and *b* from conjugate
    charge pairs.
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
    evaluations — **per-cusp** matrix model.

    The entries are ``(m_ext, e_ext, result)`` triples as produced by the
    multi-point evaluation grid in the GUI (and by ``compute_refined_index``
    called at multiple external charges).

    Algorithm
    ---------
    For each cusp *I*, filter the entries to those where **only cusp I**
    has nonzero charges (all other cusps at m = 0, e = 0).  From those
    "pure cusp I" entries, extract the per-cusp Weyl column
    ``(a_I, b_I)`` using the η-centre method (see
    :func:`_extract_cusp_ab_from_entries`).

    The full Weyl shift is then
    ``shift_j = Σ_I (a_I[j]·e_I + b_I[j]·m_I)``.

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
        both a and b for at least cusp 0.  Otherwise returns an ``ABVectors``
        instance with ``cusp_columns`` populated for multi-cusp manifolds.
    """
    if num_hard == 0:
        return ABVectors(a=[], b=[], num_hard=0)

    if not entries:
        return None

    r = len(entries[0][0])  # number of cusps

    if r == 1:
        # Single cusp — use the original scalar algorithm (fast path)
        return _compute_ab_vectors_scalar(entries, num_hard)

    # ------------------------------------------------------------------
    # Multi-cusp: extract per-cusp columns
    # ------------------------------------------------------------------
    cusp_cols: list[ABVectors | None] = []
    all_warnings: list[str] = []

    for cusp_idx in range(r):
        col = _extract_cusp_ab_from_entries(entries, num_hard, cusp_idx, r)
        cusp_cols.append(col)
        if col is None:
            all_warnings.append(
                f"Cusp {cusp_idx}: could not determine (a, b) — "
                f"insufficient pure-cusp entries"
            )

    # If ALL cusps failed, return None
    if all(c is None for c in cusp_cols):
        return None

    # Fill missing cusps with zeros
    columns: list[ABVectors] = []
    for I, col in enumerate(cusp_cols):
        if col is not None:
            columns.append(col)
        else:
            columns.append(ABVectors(
                a=[Fraction(0)] * num_hard,
                b=[Fraction(0)] * num_hard,
                num_hard=num_hard,
                warnings=[f"Cusp {I}: defaulted to a=0, b=0"],
            ))

    # Merge warnings from all columns
    for col in columns:
        for w in col.warnings:
            all_warnings.append(w)

    # a, b = cusp 0 column (for backward compat / display)
    return ABVectors(
        a=list(columns[0].a),
        b=list(columns[0].b),
        num_hard=num_hard,
        num_cusps=r,
        cusp_columns=columns,
        warnings=all_warnings,
    )


def _extract_cusp_ab_from_entries(
    entries: Sequence[tuple[list[int], list[Fraction], RefinedIndexResult]],
    num_hard: int,
    cusp_idx: int,
    num_cusps: int,
) -> ABVectors | None:
    """Extract (a, b) for a single cusp from pre-computed entries.

    Filters *entries* to "pure cusp *cusp_idx*" entries (all other cusps
    at m = 0, e = 0) and runs the η-centre extraction algorithm.
    """
    # Build 1-D indexed table: (m_I, e_I) → centre
    indexed: dict[tuple[int, Fraction], list[Fraction] | None] = {}

    for m_ext, e_ext, result in entries:
        # Check: all OTHER cusps must be at (0, 0)
        pure = True
        for I in range(num_cusps):
            if I == cusp_idx:
                continue
            if m_ext[I] != 0 or e_ext[I] != 0:
                pure = False
                break
        if not pure:
            continue

        m_i = m_ext[cusp_idx]
        e_i = Fraction(e_ext[cusp_idx])
        indexed[(m_i, e_i)] = _eta_center_at_leading_q(result, num_hard)

    if not indexed:
        return None

    # --- Extract a and b using the centre-based algorithm ---
    def get_c(m_i: int, e_i: Fraction) -> list[Fraction] | None:
        return indexed.get((m_i, e_i))

    b_estimates: list[list[Fraction]] = []
    a_estimates: list[list[Fraction]] = []

    # meridian pairs: e_i = 0, m_i > 0
    m_vals = sorted(set(m for m, e in indexed if e == 0 and m > 0))
    for m_i in m_vals:
        c_pos = get_c(m_i, Fraction(0))
        c_neg = get_c(-m_i, Fraction(0))
        if c_pos is not None and c_neg is not None:
            b_vec = [
                -(c_pos[j] - c_neg[j]) / (2 * m_i)
                for j in range(num_hard)
            ]
            b_estimates.append(b_vec)

    # longitude pairs: m_i = 0, e_i > 0
    e_vals = sorted(set(e for m, e in indexed if m == 0 and e > 0))
    for e_i in e_vals:
        c_pos = get_c(0, e_i)
        c_neg = get_c(0, -e_i)
        if c_pos is not None and c_neg is not None:
            a_vec = [
                -(c_pos[j] - c_neg[j]) / (2 * abs(e_i))
                for j in range(num_hard)
            ]
            a_estimates.append(a_vec)

    # Fallback for b: compare to zero-charge centre
    if not b_estimates:
        c_zero = get_c(0, Fraction(0))
        if c_zero is not None:
            for m_i in m_vals:
                c_pos = get_c(m_i, Fraction(0))
                if c_pos is not None:
                    b_vec = [
                        -(c_pos[j] - c_zero[j]) / m_i
                        for j in range(num_hard)
                    ]
                    b_estimates.append(b_vec)

    # Consensus
    warnings: list[str] = []

    def _consensus(
        estimates: list[list[Fraction]], label: str,
    ) -> list[Fraction] | None:
        if not estimates:
            return None
        ref = estimates[0]
        for est in estimates[1:]:
            for j, (r_j, e_j) in enumerate(zip(ref, est)):
                if r_j != e_j:
                    warnings.append(
                        f"cusp {cusp_idx} {label}[{j}]: inconsistent "
                        f"estimates {r_j} vs {e_j} (using first)"
                    )
        return ref

    b_vec = _consensus(b_estimates, "b")
    a_vec = _consensus(a_estimates, "a")

    if b_vec is None and a_vec is None:
        return None

    if b_vec is None:
        warnings.append(f"cusp {cusp_idx} b: no meridian pairs; defaulting to 0")
        b_vec = [Fraction(0)] * num_hard
    if a_vec is None:
        warnings.append(f"cusp {cusp_idx} a: no longitude pairs; defaulting to 0")
        a_vec = [Fraction(0)] * num_hard

    return ABVectors(a=a_vec, b=b_vec, num_hard=num_hard, warnings=warnings)


def _compute_ab_vectors_scalar(
    entries: Sequence[tuple[list[int], list[Fraction], RefinedIndexResult]],
    num_hard: int,
) -> ABVectors | None:
    """Original single-cusp algorithm (preserved for 1-cusp fast path).

    Uses ``sum(e)`` and ``sum(m)`` as coupling variables, which is
    correct when there is only one cusp.
    """
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
                        -(c_pos[j] - c_neg[j]) / (2 * total_e)
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
# Per-cusp Weyl vector extraction
# ---------------------------------------------------------------------------

def compute_ab_vectors_for_cusp(
    nz_data,  # NeumannZagierData — late import to avoid circular deps
    cusp_idx: int,
    q_order_half: int = 20,
) -> ABVectors | None:
    r"""
    Compute Weyl vectors (a, b) for a *single* cusp by numerical probing.

    Evaluates :func:`~manifold_index.core.refined_index.compute_refined_index`
    at a small set of charge configurations where **only** cusp *cusp_idx*
    has nonzero charges (all other cusps are at ``m = 0, e = 0``), then
    extracts the per-cusp Weyl column ``(a^{(I)}, b^{(I)})`` from the
    η-centre shift.

    This is the correct approach for the matrix Weyl model:

    .. math::

        f = \prod_{i=1}^{\text{num\_hard}}
            \eta_i^{\sum_{I=1}^{d}(a^{(i)}_I e_{n+I} + b^{(i)}_I m_{n+I})}
            \cdot \mathcal{I}^{\text{ref}}

    where *I* indexes filled cusps.  Each column
    ``(a^{(\cdot)}_I, b^{(\cdot)}_I)`` is extracted independently by varying
    only cusp *I*'s charges.

    The function is designed to be called **after** the NC basis change
    (i.e. on the rebased ``nz_nc``), so that the returned (a, b) are in
    the correct cusp basis for filling.

    Parameters
    ----------
    nz_data : NeumannZagierData
        Neumann–Zagier data (potentially after cusp basis change).
    cusp_idx : int
        Which cusp to extract Weyl vectors for (0-based).
    q_order_half : int
        Series cutoff for the refined index evaluations.

    Returns
    -------
    ABVectors or None
        Weyl vectors of length ``num_hard``.
        ``a[j]`` is the physical longitude Weyl coupling (may be half-integer).
        ``b[j]`` is the physical meridian Weyl coupling (may be half-integer).
        An edge is compatible with filling iff ``a[j] ∈ ℤ`` and ``2·b[j] ∈ ℤ``.
        Returns ``None`` if extraction fails.
    """
    from manifold_index.core.refined_index import compute_refined_index

    r = nz_data.r
    num_hard = nz_data.num_hard

    if num_hard == 0:
        return ABVectors(a=[], b=[], num_hard=0)

    # Evaluation grid: vary only cusp_idx, all others at (0, 0)
    _m_vals = [-2, -1, 0, 1, 2]
    _e_halves = [-2, -1, 0, 1, 2]  # e = k/2

    entries: list[tuple[list[int], list[Fraction], RefinedIndexResult]] = []
    for m_i in _m_vals:
        for e_half in _e_halves:
            e_i = Fraction(e_half, 2)
            # Build full (m_ext, e_ext): only cusp_idx is nonzero
            m_ext = [0] * r
            e_ext: list[Fraction] = [Fraction(0)] * r
            m_ext[cusp_idx] = m_i
            e_ext[cusp_idx] = e_i
            result = compute_refined_index(
                nz_data, m_ext, e_ext, q_order_half=q_order_half,
            )
            entries.append((m_ext, e_ext, result))

    # ------------------------------------------------------------------
    # Extract centres and compute (a, b) using per-cusp charges
    # ------------------------------------------------------------------
    indexed: dict[tuple[int, Fraction], list[Fraction] | None] = {}
    for m_ext, e_ext, result in entries:
        m_i = m_ext[cusp_idx]
        e_i = e_ext[cusp_idx]
        indexed[(m_i, e_i)] = _eta_center_at_leading_q(result, num_hard)

    def get_c(m_i: int, e_i: Fraction) -> list[Fraction] | None:
        return indexed.get((m_i, e_i))

    b_estimates: list[list[Fraction]] = []
    a_estimates: list[list[Fraction]] = []

    # ---- meridian pairs: e_i = 0, m_i > 0 ----
    for m_i in [1, 2]:
        c_pos = get_c(m_i, Fraction(0))
        c_neg = get_c(-m_i, Fraction(0))
        if c_pos is not None and c_neg is not None:
            b_vec = [
                -(c_pos[j] - c_neg[j]) / (2 * m_i)
                for j in range(num_hard)
            ]
            b_estimates.append(b_vec)

    # ---- longitude pairs: m_i = 0, e_i > 0 ----
    for e_i in [Fraction(1, 2), Fraction(1)]:
        c_pos = get_c(0, e_i)
        c_neg = get_c(0, -e_i)
        if c_pos is not None and c_neg is not None:
            a_vec = [
                -(c_pos[j] - c_neg[j]) / (2 * abs(e_i))
                for j in range(num_hard)
            ]
            a_estimates.append(a_vec)

    # Fallback: compare to zero-charge centre
    if not b_estimates:
        c_zero = get_c(0, Fraction(0))
        if c_zero is not None:
            for m_i in [1, 2]:
                c_pos = get_c(m_i, Fraction(0))
                if c_pos is not None:
                    b_vec = [
                        -(c_pos[j] - c_zero[j]) / m_i
                        for j in range(num_hard)
                    ]
                    b_estimates.append(b_vec)

    # ------------------------------------------------------------------
    # Consensus
    # ------------------------------------------------------------------
    warnings: list[str] = []

    def _consensus(
        estimates: list[list[Fraction]], label: str,
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
    Check Weyl symmetry  f(m, e) = f(−m, −e)  across sector pairs.

    For each sector (m, e), define the Weyl-shifted series

        f(m, e) = η^{a·e + b·m} · I(m, e)

    Weyl symmetry demands  f(m, e) = f(−m, −e)  as formal power series.
    This is verified by computing the shifted series for every available
    sector and comparing the (m, e) series against its partner (−m, −e).

    If the partner sector is not present in *entries* the check is marked
    as failed (the pair cannot be verified).

    Returns
    -------
    dict mapping  (tuple(m_ext), tuple(e_ext))  →  bool (True = symmetric).
    """
    # --- First pass: compute all shifted series --------------------------
    shifted_lookup: dict[tuple, RefinedIndexResult] = {}

    for m_ext, e_ext, result in entries:
        # Per-cusp shift: 2·Σ_I (a_I[j]·e_I + b_I[j]·m_I)
        shift_x2 = ab.shift_x2(m_ext, e_ext)

        shifted: RefinedIndexResult = {}
        for key, coeff in result.items():
            if coeff == 0:
                continue
            new_key = (key[0],) + tuple(
                key[1 + j] + shift_x2[j] for j in range(num_hard)
            )
            shifted[new_key] = shifted.get(new_key, 0) + coeff

        # Drop any residual zeros
        shifted = {k: v for k, v in shifted.items() if v != 0}
        shifted_lookup[(tuple(m_ext), tuple(e_ext))] = shifted

    # --- Second pass: compare f(m, e) vs f(−m, −e) ----------------------
    results: dict[tuple, bool] = {}

    for (m_key, e_key), f_me in shifted_lookup.items():
        neg_m = tuple(-x for x in m_key)
        neg_e = tuple(-x for x in e_key)
        partner_key = (neg_m, neg_e)

        if partner_key not in shifted_lookup:
            # Partner sector missing — cannot verify
            results[(m_key, e_key)] = False
            continue

        f_neg = shifted_lookup[partner_key]
        results[(m_key, e_key)] = (f_me == f_neg)

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

    In the multiplier convention  f(m,e) = η^{a·e + b·m} · I(m,e),
    this function computes the Weyl-manifest series ``f`` by multiplying
    ``I(m,e)`` by ``η^{a·e + b·m}``.

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
        Equal to ``−(a·e + b·m)`` where *a* = ``ab.a`` and *b* = ``ab.b``.
        Used to display  ``I(m,e) = η^{centre} · f(η)``.
    stripped : RefinedIndexResult
        The Weyl-manifest series ``f = η^{a·e + b·m} · I``.  Stored in
        the same doubled-exponent encoding as *result*.
    """
    shift_x2 = ab.shift_x2(m_ext, e_ext)
    # centre[j] = −(a[j]·e + b[j]·m)
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
# Check adjoint su(2) character at q^1 — eq (2.59) condition 2
# ---------------------------------------------------------------------------

@dataclass
class AdjointProjectionResult:
    """Result of the adjoint su(2) projection check (eq 2.59–2.61).

    The projection integrates out η (extract η⁰) and all cusp fugacities
    against the SU(2) Haar measure weighted by the adjoint character.
    The result must equal exactly −1 for each filled cusp.

    Attributes
    ----------
    projected_value : int | None
        The computed value of the adjoint projection.  ``None`` if the
        required entries are missing (e.g. e = ±1, ±2 not available).
    is_pass : bool
        True iff ``projected_value == -1``.
    c_e : dict[Fraction, int]
        The intermediate c_e values: for each e, the (q¹, η⁰) coefficient
        of I^ref(m=0, e).  Useful for diagnostics.
    missing_e : list[Fraction]
        List of e-values needed but not found among the entries.
    """

    projected_value: int | None
    is_pass: bool
    c_e: dict[Fraction, int] = field(default_factory=dict)
    missing_e: list[Fraction] = field(default_factory=list)


def _extract_q1_eta0_coeff_shifted(
    result: RefinedIndexResult,
    num_hard: int,
    shift_x2: list[int],
) -> int:
    """Extract the (q¹, all-η⁰) coefficient from a Weyl-shifted result.

    After applying the Weyl shift  η^{a·e + b·m}  to the raw index, the
    η⁰ component of the shifted series corresponds to the coefficient at
    ``η_x2[j] == -shift_x2[j]`` in the *unshifted* result.

    Parameters
    ----------
    result : RefinedIndexResult
        Raw (unshifted) refined index.
    num_hard : int
        Number of hard edges.
    shift_x2 : list[int]
        Doubled Weyl shift: ``2 * (a[j] * e + b[j] * m)`` for each
        hard edge j.  For m = 0 this simplifies to ``2 * a[j] * e``.
    """
    coeff = 0
    for key, val in result.items():
        if val == 0:
            continue
        if key[0] != 2:          # not q¹
            continue
        # After shifting by +shift_x2, the new η = old η + shift_x2.
        # We want new η = 0, so need old η = -shift_x2.
        if all(key[1 + h] == -shift_x2[h] for h in range(num_hard)):
            coeff += val
    return coeff


def check_adjoint_projection(
    entries: Sequence[tuple[list[int], list[Fraction], RefinedIndexResult]],
    num_hard: int,
    ab: ABVectors | None = None,
    cusp_idx: int = 0,
) -> AdjointProjectionResult:
    r"""Check condition 2 of (2.59): adjoint-projected q¹ coefficient = −1.

    Implements equations (2.60)–(2.61).  The notation
    ``I^{ref}_N(v; \tilde m, \tilde e; a, b)`` in (2.60) means the
    **Weyl-manifest** form  ``f = η^{a·e + b·m} · I(m,e)``.  So c_e is
    the (q¹, η⁰) coefficient of the Weyl-shifted series at m = 0.

    For a single cusp being filled (d = 1, n = 0):

    1. Collect all entries at m_ext = 0.
    2. For each, extract c_e := coeff(q¹, η⁰) from
       ``η^{a·e} · I^ref(m=0, e)``  (the Weyl-manifest form at m = 0).
    3. Apply the SU(2) Haar × adjoint projection:

       .. math::

           \text{proj} = \frac{1}{2}\bigl(
               c_{-1} + c_{+1} - c_{-2} - c_{+2}
           \bigr)

    4. Check ``proj == −1``.

    Parameters
    ----------
    entries : sequence of (m_ext, e_ext, result)
        The full evaluation grid (all m, e points).
    num_hard : int
        Number of hard edges (η variables).
    ab : ABVectors or None
        Weyl vectors.  When provided the Weyl shift η^{a·e} is applied
        before extracting η⁰.  When ``None`` no shift is applied (equivalent
        to a = 0, b = 0).
    cusp_idx : int
        Which cusp is being checked for filling compatibility.

    Returns
    -------
    AdjointProjectionResult
    """
    # ---- Step 1: collect m=0 entries, extract c_e (Weyl-shifted) ----
    c_e: dict[Fraction, int] = {}
    for m_ext, e_ext, result in entries:
        # All cusps must have m = 0
        if any(m != 0 for m in m_ext):
            continue
        # For multi-cusp: all *other* cusps must have e = 0 too
        # (the ∮ duᵢ/(2πi uᵢ) projection extracts uᵢ⁰ for unfilled cusps)
        skip = False
        for i, e_val in enumerate(e_ext):
            if i != cusp_idx and e_val != 0:
                skip = True
                break
        if skip:
            continue

        e_val = Fraction(e_ext[cusp_idx])

        # Compute the Weyl shift at m=0 using per-cusp model
        if ab is not None and num_hard > 0:
            shift_x2 = ab.shift_x2(m_ext, e_ext)
        else:
            shift_x2 = [0] * num_hard

        coeff = _extract_q1_eta0_coeff_shifted(result, num_hard, shift_x2)
        c_e[e_val] = c_e.get(e_val, 0) + coeff

    # ---- Step 2: check we have the needed e-values ----
    needed = [Fraction(-2), Fraction(-1), Fraction(1), Fraction(2)]
    missing = [e for e in needed if e not in c_e]

    if missing:
        return AdjointProjectionResult(
            projected_value=None, is_pass=False,
            c_e=c_e, missing_e=missing,
        )

    # ---- Step 3: Haar × adjoint projection ----
    # Kernel: (1/2)(u² + u⁻² − u⁴ − u⁻⁴)
    # After f(u²) = Σ c_e u^{2e}, extracting [u⁰]:
    #   (1/2)(c_{-1} + c_{+1} − c_{-2} − c_{+2})
    numerator = c_e[Fraction(-1)] + c_e[Fraction(1)] \
        - c_e[Fraction(-2)] - c_e[Fraction(2)]

    # numerator must be even for an integer result
    if numerator % 2 != 0:
        return AdjointProjectionResult(
            projected_value=None, is_pass=False,
            c_e=c_e, missing_e=[],
        )

    projected = numerator // 2

    return AdjointProjectionResult(
        projected_value=projected,
        is_pass=(projected == -1),
        c_e=c_e,
        missing_e=[],
    )




# ---------------------------------------------------------------------------
# W-vector linear combination framework
# ---------------------------------------------------------------------------

def _extract_q1_projected_coeff(
    result,  # RefinedIndexResult
    num_hard: int,
    w,       # Sequence[int]
    target_x2: int,
) -> int:
    r"""Extract the q^1 coefficient at a given projected eta-exponent.

    Projects the multi-eta polynomial onto a single combined variable
    via the W-vector:

        combined_x2 = sum_j W_j * key[1+j]

    and returns the sum of coefficients where ``key[0] == 2`` (q^1) and
    ``combined_x2 == target_x2``.
    """
    coeff = 0
    for key, val in result.items():
        if val == 0:
            continue
        if key[0] != 2:  # not q^1
            continue
        combined_x2 = sum(w[j] * key[1 + j] for j in range(num_hard))
        if combined_x2 == target_x2:
            coeff += val
    return coeff


@dataclass
class WScanEntry:
    r"""Result of the adjoint projection check for one W-vector.

    Attributes
    ----------
    w : tuple[int, ...]
        The W-vector: eta_j = eta^{2*W_j}.
    a_eff : Fraction
        Effective Weyl a-coefficient: W . a.
    b_eff : Fraction
        Effective Weyl b-coefficient: W . b.
    a_eff_is_integer : bool
        Whether a_eff is an integer (filling compatibility condition).
    adjoint : AdjointProjectionResult | None
        Adjoint projection result.  None if the check was skipped.
    """

    w: tuple[int, ...]
    a_eff: Fraction
    b_eff: Fraction
    a_eff_is_integer: bool
    adjoint: AdjointProjectionResult | None


@dataclass
class WScanResult:
    r"""Aggregated results of scanning W-vectors.

    Attributes
    ----------
    ab : ABVectors
        The raw Weyl vectors used for the scan.
    entries : list[WScanEntry]
        All scanned W-vectors with their results.
    passing : list[WScanEntry]
        Subset with adjoint.is_pass == True.
    """

    ab: ABVectors
    entries: list[WScanEntry]
    passing: list[WScanEntry]


def check_adjoint_with_w_vector(
    entries,     # Sequence[tuple[list[int], list[Fraction], RefinedIndexResult]]
    num_hard: int,
    ab,          # ABVectors
    w,           # Sequence[int]
    cusp_idx: int = 0,
) -> AdjointProjectionResult:
    r"""Check the adjoint projection for a given W-vector.

    Instead of requiring eta^0 for each hard-edge eta_j independently, this
    projects the multi-eta polynomial onto a single combined variable via

        eta_j = eta^{2*W_j}

    and checks the adjoint projection condition on the combined variable.

    The Weyl shift becomes  eta^{a_eff * e + b_eff * m}  where
    a_eff = W . a_I  and  b_eff = W . b_I  (for the cusp *I* being
    checked).

    Algorithm
    ---------
    For each entry at m = 0:

    1. Compute target_x2 = -2 * a_eff * e_I -- the doubled combined
       eta-exponent that maps to eta^0 after the Weyl shift.
    2. Sum all q^1 monomials whose projected exponent
       sum(W_j * key[1+j]) == target_x2.
    3. Collect c_e for e in {-2, -1, +1, +2} and apply the Haar x adjoint
       projection: 1/2(c_{-1} + c_{+1} - c_{-2} - c_{+2}) = -1.
    """
    a_col = ab.a_for_cusp(cusp_idx)
    b_col = ab.b_for_cusp(cusp_idx)
    a_eff = sum(Fraction(w[j]) * a_col[j] for j in range(num_hard))

    c_e: dict[Fraction, int] = {}
    for m_ext, e_ext, result in entries:
        if any(m != 0 for m in m_ext):
            continue
        skip = False
        for i, e_val in enumerate(e_ext):
            if i != cusp_idx and e_val != 0:
                skip = True
                break
        if skip:
            continue

        e_val = Fraction(e_ext[cusp_idx])

        target_raw = -2 * a_eff * e_val
        if target_raw.denominator != 1:
            c_e[e_val] = c_e.get(e_val, 0)
            continue
        target_x2 = int(target_raw)

        coeff = _extract_q1_projected_coeff(result, num_hard, w, target_x2)
        c_e[e_val] = c_e.get(e_val, 0) + coeff

    needed = [Fraction(-2), Fraction(-1), Fraction(1), Fraction(2)]
    missing = [e for e in needed if e not in c_e]

    if missing:
        return AdjointProjectionResult(
            projected_value=None, is_pass=False,
            c_e=c_e, missing_e=missing,
        )

    numerator = c_e[Fraction(-1)] + c_e[Fraction(1)] \
        - c_e[Fraction(-2)] - c_e[Fraction(2)]

    if numerator % 2 != 0:
        return AdjointProjectionResult(
            projected_value=None, is_pass=False,
            c_e=c_e, missing_e=[],
        )

    projected = numerator // 2
    return AdjointProjectionResult(
        projected_value=projected,
        is_pass=(projected == -1),
        c_e=c_e,
        missing_e=[],
    )


def scan_w_vectors(
    entries,     # Sequence[tuple[list[int], list[Fraction], RefinedIndexResult]]
    num_hard: int,
    ab,          # ABVectors
    cusp_idx: int = 0,
    max_coeff: int = 3,
    *,
    skip_incompatible: bool = False,
) -> WScanResult:
    r"""Scan W-vectors and check the adjoint projection for each.

    Enumerates all integer W-vectors with |W_j| <= max_coeff (excluding
    the zero vector).  Canonicalises by sign: only W-vectors whose first
    nonzero entry is positive are tested, since v and -v give the same
    adjoint projection (the Haar x adjoint kernel is even in eta).

    For each W-vector, computes a_eff = W . a_I, b_eff = W . b_I
    (for the cusp being checked), checks whether a_eff is an integer,
    and (unless skip_incompatible is set and a_eff is not integer) runs
    the adjoint projection check.
    """
    from itertools import product

    if num_hard == 0:
        return WScanResult(ab=ab, entries=[], passing=[])

    a_col = ab.a_for_cusp(cusp_idx)
    b_col = ab.b_for_cusp(cusp_idx)

    all_entries: list[WScanEntry] = []
    passing: list[WScanEntry] = []

    rng = range(-max_coeff, max_coeff + 1)
    for combo in product(rng, repeat=num_hard):
        if all(c == 0 for c in combo):
            continue
        first_nz = next(c for c in combo if c != 0)
        if first_nz < 0:
            continue

        w = combo
        a_eff = sum(Fraction(w[j]) * a_col[j] for j in range(num_hard))
        b_eff = sum(Fraction(w[j]) * b_col[j] for j in range(num_hard))
        a_int = a_eff.denominator == 1

        if skip_incompatible and not a_int:
            adj = None
        else:
            adj = check_adjoint_with_w_vector(
                entries, num_hard, ab, w, cusp_idx,
            )

        entry = WScanEntry(
            w=w,
            a_eff=a_eff,
            b_eff=b_eff,
            a_eff_is_integer=a_int,
            adjoint=adj,
        )
        all_entries.append(entry)
        if adj is not None and adj.is_pass:
            passing.append(entry)

    return WScanResult(ab=ab, entries=all_entries, passing=passing)


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

    adjoint: AdjointProjectionResult | None
    "Adjoint su(2) projection check result (eq 2.59–2.61)."

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
        if self.adjoint is not None:
            if self.adjoint.missing_e:
                lines.append(
                    f"Adjoint q¹ projection: INCOMPLETE "
                    f"(missing e = {self.adjoint.missing_e})"
                )
            elif self.adjoint.projected_value is not None:
                status = "PASS" if self.adjoint.is_pass else "FAIL"
                lines.append(
                    f"Adjoint q¹ projection: {status} "
                    f"(projected = {self.adjoint.projected_value}, expected −1)"
                )
        return "\n".join(lines)


def run_weyl_checks(
    entries: Sequence[tuple[list[int], list[Fraction], RefinedIndexResult]],
    num_hard: int,
    cusp_idx: int = 0,
) -> WeylCheckResult:
    """
    Run all Weyl-symmetry prerequisite checks for Dehn filling.

    Parameters
    ----------
    entries : sequence of (m_ext, e_ext, result)
    num_hard : int
    cusp_idx : int
        Which cusp to check for the adjoint projection.

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

    # Adjoint projection check (eq 2.59–2.61)
    try:
        adjoint_result = check_adjoint_projection(
            entries, num_hard, ab=ab, cusp_idx=cusp_idx,
        )
    except Exception:
        adjoint_result = None

    return WeylCheckResult(
        ab=ab,
        ab_valid=ab_valid,
        weyl_symmetric=weyl_sym,
        all_weyl_symmetric=all_weyl,
        adjoint=adjoint_result,
    )
