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

    The physical Weyl monomial is  ``η^{a·e + b·m}``, so:

    * ``a[j]`` couples to the longitude charge *e*
    * ``b[j]`` couples to the meridian charge *m*

    An edge *j* is **compatible** with Dehn filling iff ``a[j] ∈ ℤ``
    and ``2·b[j] ∈ ℤ``.  (This ensures the doubled-exponent shift
    ``2(a·e + b·m)`` is always an integer for half-integer *e* and
    integer *m*.)  Incompatible edges must have their refinement
    turned off (η_j = 1, v_j = 0).

    Attributes
    ----------
    a : list[Fraction]
        One entry per hard edge.  Compatible iff ``a[j] ∈ ℤ``.
    b : list[Fraction]
        One entry per hard edge.  Compatible iff ``2·b[j] ∈ ℤ``.
    num_hard : int
        Number of hard edges.
    warnings : list[str]
        Non-fatal messages (e.g., inconsistency between different pair
        estimates, or missing data for some components).
    """

    a: list[Fraction]
    b: list[Fraction]
    num_hard: int
    warnings: list[str] = field(default_factory=list)

    @property
    def a_is_integer(self) -> list[bool]:
        """``a[j] ∈ ℤ`` for each j."""
        return [v.denominator == 1 for v in self.a]

    @property
    def b_is_half_integer(self) -> list[bool]:
        """``2·b[j] ∈ ℤ`` for each j."""
        return [(v * 2).denominator == 1 for v in self.b]

    @property
    def is_valid(self) -> bool:
        """True iff every edge is compatible: a ∈ ℤ and 2b ∈ ℤ."""
        return all(self.a_is_integer) and all(self.b_is_half_integer)

    @property
    def edge_compatible(self) -> list[bool]:
        """Per-edge compatibility with Dehn filling.

        Edge *j* is compatible iff ``a[j] ∈ ℤ`` and ``2·b[j] ∈ ℤ``.
        Incompatible edges must be turned off: η_j = 1 (v_j = 0).
        """
        return [
            a_ok and b_ok
            for a_ok, b_ok in zip(self.a_is_integer, self.b_is_half_integer)
        ]

    def make_filling_compatible(self) -> "ABVectors":
        """Return a copy with incompatible edges zeroed out.

        For each hard edge j where :attr:`edge_compatible` is False,
        sets ``a[j] = 0`` and ``b[j] = 0`` (i.e., η_j = 1 during
        filling).  Compatible edges are kept unchanged.
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
                f"set η_j=1 (v_j=0) for filling"
            )
        return ABVectors(
            a=a_new, b=b_new, num_hard=self.num_hard,
            warnings=new_warnings,
        )

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

          a[j]  =  −[centre_j(0, +e) − centre_j(0, −e)] / (2 · sum(|e|))

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
    For each entry verify  f(m, e) = η^{a·e + b·m} · I(m, e)  is Weyl-symmetric.

    Computes the shifted series  f = η^{shift} · I  (where shift = a·e + b·m)
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
        m_sum = sum(m_ext)
        e_sum = sum(e_ext)
        # shift_x2 = 2·(a·e + b·m) in the doubled-exponent encoding
        shift_x2 = [
            int(2 * (ab.a[j] * e_sum + ab.b[j] * m_sum))
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
    m_sum = sum(m_ext)
    e_sum = sum(Fraction(v) for v in e_ext)
    shift_x2 = [
        int(2 * (ab.a[j] * e_sum + ab.b[j] * m_sum))
        for j in range(num_hard)
    ]
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

        # Compute the Weyl shift at m=0: 2·(a[j]·e + b[j]·0) = 2·a[j]·e
        if ab is not None and num_hard > 0:
            e_sum = sum(Fraction(v) for v in e_ext)
            shift_x2 = [int(2 * ab.a[j] * e_sum) for j in range(num_hard)]
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
