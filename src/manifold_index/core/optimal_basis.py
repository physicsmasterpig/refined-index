"""Hard-edge basis optimisation for refined Dehn filling.

Given a manifold and an NC cycle (P, Q), search integer-unimodular
changes-of-basis ``G`` on the hard-edge subspace to maximise the
refinement count

    refinement = #{j : a[j] ∈ ℤ  AND  2·b[j] ∈ ℤ}

subject to the adjoint q¹ projection passing with an integer value.

Math used in the prefilter
--------------------------
Under a unimodular ``G`` acting on the hard rows, the Weyl vectors
transform contragrediently:

    (a', b') = (G⁻ᵀ · a, G⁻ᵀ · b)

So we can predict the refinement count for any candidate ``G`` directly
from the default-basis ``(a, b)`` — no need to rebuild the NZ data or
recompute ``I^ref``.  This makes the search cheap.

The adjoint q¹ projection, by contrast, depends on the full refined
index in the new basis, so each promising candidate is verified with a
full evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product as _iproduct
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from manifold_index.core.manifold import ManifoldData
    from manifold_index.core.phase_space import EasyEdgeResult


# ---------------------------------------------------------------------------

@dataclass
class OptimalBasisResult:
    """Outcome of a hard-edge basis search."""

    G: list[list[int]]
    """Unimodular ``num_hard × num_hard`` matrix selected (rows are the
    integer-combination coefficients applied to the original hard rows)."""

    refinement: int
    """Refinement count under the selected basis (counting compatible edges)."""

    default_refinement: int
    """Refinement count under the original (unmodified) hard-edge basis."""

    new_easy_result: "EasyEdgeResult"
    """``EasyEdgeResult`` rebuilt with the optimised hard rows + correct RHS."""

    ab_a: list[Fraction]
    """New ``a`` vector from a fresh Weyl check on the optimised basis."""

    ab_b: list[Fraction]
    """New ``b`` vector likewise."""

    adj_pass: bool
    adj_val: int

    n_candidates_searched: int = 0
    """Diagnostic — how many unimodular ``G`` matrices were stage-A inspected."""

    n_candidates_verified: int = 0
    """Diagnostic — how many candidates required a full Weyl re-evaluation."""


# ---------------------------------------------------------------------------
# Linear-algebra helpers
# ---------------------------------------------------------------------------

def _int_det(G: np.ndarray) -> int:
    return int(round(np.linalg.det(G)))


def _adjugate(G: np.ndarray) -> np.ndarray:
    """Integer adjugate (transpose-of-cofactor matrix)."""
    n = G.shape[0]
    cof = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            minor = np.delete(np.delete(G, i, 0), j, 1)
            cof[i, j] = ((-1) ** (i + j)) * int(round(np.linalg.det(minor)))
    return cof.T


def _G_inv_T_int(G: np.ndarray) -> np.ndarray:
    """``(G⁻ᵀ)`` for unimodular ``G``, kept exact via the integer adjugate."""
    det = _int_det(G)
    if det not in (-1, 1):
        raise ValueError(f"G must be unimodular (got det = {det})")
    return (_adjugate(G) // det).T


def _apply_G_inv_T_to_vec(G: np.ndarray, v: list[Fraction]) -> list[Fraction]:
    GiT = _G_inv_T_int(G)
    n = len(v)
    return [
        sum(Fraction(int(GiT[i, k])) * v[k] for k in range(n))
        for i in range(n)
    ]


def _refinement_count(a_vec, b_vec) -> int:
    return sum(
        1 for j in range(len(a_vec))
        if Fraction(a_vec[j]).denominator == 1
        and (Fraction(b_vec[j]) * 2).denominator == 1
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def find_optimal_hard_basis(
    data: "ManifoldData",
    easy_result: "EasyEdgeResult",
    cusp_idx: int,
    nc_P: int,
    nc_Q: int,
    *,
    q_order_half: int = 8,
    coeff_range: int = 1,
    require_integer_adj: bool = True,
    require_adj_pass: bool = True,
) -> "OptimalBasisResult | None":
    """Search hard-edge basis space for max refinement on NC cycle ``(P, Q)``.

    Returns ``OptimalBasisResult`` if a strictly-better basis is found and
    verifies adjoint conditions; otherwise ``None``.

    The search enumerates all unimodular ``num_hard × num_hard`` matrices
    with entries in ``[-coeff_range, coeff_range]``.  Default
    ``coeff_range = 1`` finishes in ~1–2 s on ``num_hard = 3``; raising to
    2 takes ~20 s.  Verification (full Weyl check) is run only on the few
    candidates whose predicted refinement exceeds the default.
    """
    # Local imports to avoid a circular dependency at module-load time.
    from manifold_index.core.phase_space import EasyEdgeResult, _is_easy
    from manifold_index.core.neumann_zagier import (
        build_neumann_zagier, apply_general_cusp_basis_change,
    )
    from manifold_index.core import dehn_filling as _df
    from manifold_index.services.compute_service import ComputeService

    # ── Step 0: helper that runs the Weyl probe + adjoint check ────────────
    def _weyl_probe(local_easy: "EasyEdgeResult") -> "tuple | None":
        nz = build_neumann_zagier(data, local_easy)
        R, S = _df.find_rs(nc_P, nc_Q)
        nz_nc = apply_general_cusp_basis_change(nz, cusp_idx,
                                                a=nc_P, b=nc_Q, c=-R, d=-S)
        n_cusps = nz.r
        entries = []
        seen = set()
        for e_val in (Fraction(-2), Fraction(-1), Fraction(-1, 2),
                      Fraction(1, 2), Fraction(1), Fraction(2)):
            m_e = [0] * n_cusps
            e_e = [Fraction(0)] * n_cusps
            e_e[cusp_idx] = e_val
            key = (tuple(m_e), tuple(e_e))
            if key in seen:
                continue
            seen.add(key)
            res = ComputeService.compute_refined_index(nz_nc, m_e, e_e, q_order_half)
            if res is not None:
                entries.append((m_e, e_e, res))
        for m_val in (-2, -1, 1, 2):
            m_e = [0] * n_cusps
            e_e = [Fraction(0)] * n_cusps
            m_e[cusp_idx] = m_val
            key = (tuple(m_e), tuple(e_e))
            if key in seen:
                continue
            seen.add(key)
            res = ComputeService.compute_refined_index(nz_nc, m_e, e_e, q_order_half)
            if res is not None:
                entries.append((m_e, e_e, res))
        m_e = [0] * n_cusps; e_e = [Fraction(0)] * n_cusps
        if (tuple(m_e), tuple(e_e)) not in seen:
            res = ComputeService.compute_refined_index(nz_nc, m_e, e_e, q_order_half)
            if res is not None:
                entries.append((m_e, e_e, res))

        if not entries:
            return None
        ab, ap, av = ComputeService.run_weyl_check(
            entries, num_hard=len(local_easy.hard_padding),
            q_order_half=q_order_half, cusp_idx=cusp_idx,
        )
        return ab, ap, av

    # ── Step 1: default basis Weyl check ──────────────────────────────────
    default_probe = _weyl_probe(easy_result)
    if default_probe is None:
        return None
    default_ab, default_ap, default_av = default_probe
    if default_ab is None:
        return None
    default_a = [Fraction(x) for x in default_ab.a]
    default_b = [Fraction(x) for x in default_ab.b]
    default_refine = _refinement_count(default_a, default_b)

    # ── Step 2: analytic prefilter over unimodular G ───────────────────────
    num_hard = len(easy_result.hard_padding)
    if num_hard == 0:
        return None  # nothing to optimise

    coeff_set = tuple(range(-coeff_range, coeff_range + 1))
    candidates: list[tuple[int, np.ndarray]] = []  # (predicted_refine, G)
    n_searched = 0
    seen_signatures: set = set()
    for g_flat in _iproduct(*([coeff_set] * (num_hard * num_hard))):
        n_searched += 1
        G = np.array(g_flat, dtype=int).reshape(num_hard, num_hard)
        if _int_det(G) not in (-1, 1):
            continue
        try:
            a_new = _apply_G_inv_T_to_vec(G, default_a)
            b_new = _apply_G_inv_T_to_vec(G, default_b)
        except ValueError:
            continue
        rc = _refinement_count(a_new, b_new)
        if rc <= default_refine:
            continue
        # Skip duplicates that produce the same (a', b').  Different G's can
        # map to the same hard-row span; the resulting Weyl vectors are
        # identical, so the verification step will give the same answer.
        sig = (tuple(a_new), tuple(b_new))
        if sig in seen_signatures:
            continue
        seen_signatures.add(sig)
        candidates.append((rc, G))

    candidates.sort(key=lambda x: -x[0])

    # ── Step 3: verify each candidate (best predicted first) ──────────────
    n_verified = 0
    for predicted_rc, G in candidates:
        # Build the new hard rows and RHS.  The RHS scales linearly with G
        # because each raw hard-edge equation has a definite RHS, and a
        # linear combination's RHS is the same linear combination of RHSes.
        new_hard: list[np.ndarray] = []
        new_rhs: list[int] = []
        for i in range(num_hard):
            row = np.zeros_like(easy_result.hard_padding[0])
            rhs = 0
            for k in range(num_hard):
                row = row + int(G[i, k]) * easy_result.hard_padding[k]
                rhs += int(G[i, k]) * int(easy_result.hard_padding_rhs[k])
            new_hard.append(row)
            new_rhs.append(rhs)
        # Skip if any new row is "easy" (would not be a valid hard row).
        if any(_is_easy(r, easy_result.n) for r in new_hard):
            continue

        new_easy = EasyEdgeResult(
            all_easy=easy_result.all_easy,
            independent_easy_indices=easy_result.independent_easy_indices,
            hard_padding=new_hard,
            n=easy_result.n,
            r=easy_result.r,
            hard_padding_rhs=new_rhs,
            all_easy_rhs=list(easy_result.all_easy_rhs),
        )
        try:
            probe = _weyl_probe(new_easy)
        except Exception:
            continue
        if probe is None:
            continue
        ab, ap, av = probe
        if ab is None:
            continue
        n_verified += 1
        a_v = [Fraction(x) for x in ab.a]
        b_v = [Fraction(x) for x in ab.b]
        actual_rc = _refinement_count(a_v, b_v)
        if actual_rc <= default_refine:
            continue
        if require_adj_pass and ap is not True:
            continue
        if require_integer_adj and not isinstance(av, int):
            continue
        # Found it — first acceptance wins (sorted by predicted refinement).
        return OptimalBasisResult(
            G=[[int(G[i, j]) for j in range(num_hard)] for i in range(num_hard)],
            refinement=actual_rc,
            default_refinement=default_refine,
            new_easy_result=new_easy,
            ab_a=a_v,
            ab_b=b_v,
            adj_pass=bool(ap),
            adj_val=int(av) if av is not None else 0,
            n_candidates_searched=n_searched,
            n_candidates_verified=n_verified,
        )

    return None


# ---------------------------------------------------------------------------
# Multi-cusp variant
# ---------------------------------------------------------------------------

def find_optimal_hard_basis_multi(
    data: "ManifoldData",
    easy_result: "EasyEdgeResult",
    cusp_specs: list,
    *,
    q_order_half: int = 8,
    coeff_range: int = 1,
    require_integer_adj: bool = True,
    require_adj_pass: bool = True,
) -> "OptimalBasisResult | None":
    """Multi-cusp variant of :func:`find_optimal_hard_basis`.

    ``cusp_specs`` is a list of dicts ``{"cusp_idx", "nc_P", "nc_Q"}``,
    one per simultaneously-filled cusp.  Optimises the hard basis for
    the JOINT multi-cusp adjoint check: ``ab.cusp_columns`` carries one
    Weyl column per cusp and refinement = #{j : every column has
    ``a_I[j] ∈ ℤ ∧ 2·b_I[j] ∈ ℤ``}.

    The contragredient transform applies independently per column, so
    Stage A is still cheap rational arithmetic.
    """
    from itertools import product as _product
    from manifold_index.core.phase_space import EasyEdgeResult, _is_easy
    from manifold_index.core.neumann_zagier import (
        build_neumann_zagier, apply_general_cusp_basis_change,
    )
    from manifold_index.core import dehn_filling as _df
    from manifold_index.core.weyl_check import run_weyl_checks
    from manifold_index.services.compute_service import ComputeService

    filled_cusp_indices = [int(s["cusp_idx"]) for s in cusp_specs]
    d = len(filled_cusp_indices)
    if d == 0:
        return None

    def _multi_basis_changed(nz):
        for spec in cusp_specs:
            P, Q = int(spec["nc_P"]), int(spec["nc_Q"])
            ci = int(spec["cusp_idx"])
            R, S = _df.find_rs(P, Q)
            nz = apply_general_cusp_basis_change(nz, ci, a=P, b=Q, c=-R, d=-S)
        return nz

    def _multi_probe(local_easy):
        nz = build_neumann_zagier(data, local_easy)
        nz_nc = _multi_basis_changed(nz)
        n_cusps = nz.r
        entries: list = []
        seen: set = set()
        a_vals = (Fraction(-2), Fraction(-1), Fraction(-1, 2),
                  Fraction(1, 2), Fraction(1), Fraction(2))
        b_vals = (-2, -1, 1, 2)
        for ci in filled_cusp_indices:
            for e_val in a_vals:
                m_e = [0] * n_cusps; e_e = [Fraction(0)] * n_cusps
                e_e[ci] = e_val
                key = (tuple(m_e), tuple(e_e))
                if key in seen: continue
                seen.add(key)
                res = ComputeService.compute_refined_index(nz_nc, m_e, e_e, q_order_half)
                if res is not None:
                    entries.append((m_e, e_e, res))
            for m_val in b_vals:
                m_e = [0] * n_cusps; e_e = [Fraction(0)] * n_cusps
                m_e[ci] = m_val
                key = (tuple(m_e), tuple(e_e))
                if key in seen: continue
                seen.add(key)
                res = ComputeService.compute_refined_index(nz_nc, m_e, e_e, q_order_half)
                if res is not None:
                    entries.append((m_e, e_e, res))
        e_target = (Fraction(-2), Fraction(-1), Fraction(1), Fraction(2))
        e_other = (Fraction(-1), Fraction(0), Fraction(1))
        for target_idx in range(d):
            for e_t in e_target:
                for e_o_combo in _product(e_other, repeat=d - 1):
                    m_e = [0] * n_cusps; e_e = [Fraction(0)] * n_cusps
                    o = 0
                    for j, ci in enumerate(filled_cusp_indices):
                        if j == target_idx:
                            e_e[ci] = e_t
                        else:
                            e_e[ci] = e_o_combo[o]; o += 1
                    key = (tuple(m_e), tuple(e_e))
                    if key in seen: continue
                    seen.add(key)
                    res = ComputeService.compute_refined_index(nz_nc, m_e, e_e, q_order_half)
                    if res is not None:
                        entries.append((m_e, e_e, res))
        if not entries:
            return None
        wc = run_weyl_checks(
            entries, num_hard=len(local_easy.hard_padding),
            filled_cusp_indices=filled_cusp_indices,
            q_order_half=q_order_half,
        )
        return wc.ab, wc.multi_cusp_adjoint

    def _multi_refinement(ab) -> int:
        if ab is None: return 0
        nh = ab.num_hard
        if ab.cusp_columns is None:
            return sum(1 for j in range(nh)
                       if Fraction(ab.a[j]).denominator == 1
                       and (Fraction(ab.b[j]) * 2).denominator == 1)
        out = 0
        for j in range(nh):
            ok = True
            for col in ab.cusp_columns:
                if Fraction(col.a[j]).denominator != 1:
                    ok = False; break
                if (Fraction(col.b[j]) * 2).denominator != 1:
                    ok = False; break
            if ok: out += 1
        return out

    def _predict_refine(GiT, ab) -> int:
        nh = ab.num_hard
        cols = ab.cusp_columns or [ab]
        out = 0
        for j in range(nh):
            ok = True
            for col in cols:
                a_j = sum(Fraction(int(GiT[j, k])) * Fraction(col.a[k])
                          for k in range(nh))
                b_j = sum(Fraction(int(GiT[j, k])) * Fraction(col.b[k])
                          for k in range(nh))
                if a_j.denominator != 1 or (b_j * 2).denominator != 1:
                    ok = False; break
            if ok: out += 1
        return out

    default = _multi_probe(easy_result)
    if default is None:
        return None
    default_ab, default_mc = default
    if default_ab is None:
        return None
    default_refine = _multi_refinement(default_ab)

    num_hard = len(easy_result.hard_padding)
    if num_hard == 0:
        return None
    coeff_set = tuple(range(-coeff_range, coeff_range + 1))
    candidates: list = []
    n_searched = 0
    for g_flat in _iproduct(*([coeff_set] * (num_hard * num_hard))):
        n_searched += 1
        G = np.array(g_flat, dtype=int).reshape(num_hard, num_hard)
        if _int_det(G) not in (-1, 1):
            continue
        try:
            GiT = _G_inv_T_int(G)
        except ValueError:
            continue
        rc = _predict_refine(GiT, default_ab)
        if rc <= default_refine:
            continue
        candidates.append((rc, G))
    candidates.sort(key=lambda x: -x[0])

    n_verified = 0
    for predicted_rc, G in candidates:
        new_hard: list[np.ndarray] = []
        new_rhs: list[int] = []
        for i in range(num_hard):
            row = np.zeros_like(easy_result.hard_padding[0])
            rhs = 0
            for k in range(num_hard):
                row = row + int(G[i, k]) * easy_result.hard_padding[k]
                rhs += int(G[i, k]) * int(easy_result.hard_padding_rhs[k])
            new_hard.append(row); new_rhs.append(rhs)
        if any(_is_easy(r, easy_result.n) for r in new_hard):
            continue
        new_easy = EasyEdgeResult(
            all_easy=easy_result.all_easy,
            independent_easy_indices=easy_result.independent_easy_indices,
            hard_padding=new_hard, n=easy_result.n, r=easy_result.r,
            hard_padding_rhs=new_rhs,
            all_easy_rhs=list(easy_result.all_easy_rhs),
        )
        try:
            probe = _multi_probe(new_easy)
        except Exception:
            continue
        if probe is None:
            continue
        ab, mc = probe
        if ab is None:
            continue
        n_verified += 1
        actual_rc = _multi_refinement(ab)
        if actual_rc <= default_refine:
            continue
        if require_adj_pass:
            if mc is None or not getattr(mc, 'all_pass', False):
                continue
        if require_integer_adj and mc is not None and mc.results:
            if any(r.projected_value is None for r in mc.results):
                continue
        a_v = [Fraction(x) for x in ab.a]
        b_v = [Fraction(x) for x in ab.b]
        adj_pass = bool(mc and mc.all_pass)
        adj_val = 0
        if mc and mc.results and mc.results[0].projected_value is not None:
            adj_val = int(mc.results[0].projected_value)
        return OptimalBasisResult(
            G=[[int(G[i, j]) for j in range(num_hard)] for i in range(num_hard)],
            refinement=actual_rc,
            default_refinement=default_refine,
            new_easy_result=new_easy,
            ab_a=a_v, ab_b=b_v,
            adj_pass=adj_pass, adj_val=adj_val,
            n_candidates_searched=n_searched,
            n_candidates_verified=n_verified,
        )

    return None
