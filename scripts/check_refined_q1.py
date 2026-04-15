#!/usr/bin/env -S python3 -u
"""
check_refined_q1.py
===================
For each NC cycle found in configurable census manifold ranges (P in {-1,0,1},
Q in {0,1}), compute the REFINED q^1 SU(2)-adjoint projection:

    proj_refined = (1/2)(c_{-1} + c_{+1} - c_{-2} - c_{+2})

where c_e is the (q^1, η^0) coefficient of the Weyl-shifted refined index
I^ref(m=0, e) in the NC basis, extracted AFTER applying the Weyl shift η^{a·e}.

This tests W_j compatibility: the NC cycle supports refined Dehn filling iff
proj_refined ≤ -1.  Also shows the character decomposition of the Weyl-shifted
q^1 coefficient and the Weyl vectors (a, b).

Contrast with check_strongly_nc.py which uses the UNREFINED index (η=1).

Output
------
  - Per-manifold table with NC/refined-proj status
  - Character decomposition of the Weyl-shifted q^1 term
  - Summary: cycles with proj ≤ -1 vs proj > -1

Usage
-----
  python check_refined_q1.py

Configure MANIFOLD_RANGES and Q_ORDER_HALF at the top.
"""

from __future__ import annotations

import sys
import os
from fractions import Fraction
from math import gcd
from dataclasses import dataclass, field
from collections import defaultdict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MANIFOLD_RANGES = [
    ("m", 3, 9),
    # ("s", 0, 5),
]

CUSP_IDX = 0
REQUIRE_SINGLE_CUSP = True
Q_ORDER_HALF = 12

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_V05_SRC = os.path.join(_SCRIPT_DIR, "..", "v0.5", "src")
if _V05_SRC not in sys.path:
    sys.path.insert(0, _V05_SRC)

from manifold_index.services.compute_service import ComputeService
from manifold_index.core import dehn_filling as _df
from manifold_index.core import neumann_zagier as _nz
from manifold_index.core.weyl_check import run_weyl_checks, _extract_q1_eta0_coeff_shifted

# ---------------------------------------------------------------------------
# Slopes
# ---------------------------------------------------------------------------
_ALL_SLOPES: list[tuple[int, int]] = [
    (P, Q)
    for P in range(-1, 2)
    for Q in range(0, 2)
    if not (P == 0 and Q == 0) and gcd(abs(P), abs(Q)) == 1
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class SlopeResult:
    manifold: str
    P: int
    Q: int
    is_nc: bool
    weyl_a: list | None = None
    weyl_b: list | None = None
    incompat_edges: list[int] = field(default_factory=list)
    refined_proj: int | None = None      # Weyl-shifted η^0 adjoint projection
    refined_pass: bool | None = None     # proj ≤ -1
    decomp: dict[int, int] = field(default_factory=dict)  # {j: multiplicity}

    def slope_str(self) -> str:
        return f"({self.P:+d},{self.Q:+d})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_nc(nz_data, P, Q, cusp_idx):
    filled = _df.compute_filled_index(
        nz_data, cusp_idx=cusp_idx, P=P, Q=Q,
        m_other=[0]*(nz_data.r-1), e_other=[0]*(nz_data.r-1),
        q_order_half=Q_ORDER_HALF,
    )
    return filled.is_stably_zero()


def _refined_check(
    nz_data, P: int, Q: int, cusp_idx: int
) -> tuple[list | None, list | None, list[int], int | None, dict[int, int]]:
    """
    Returns (weyl_a, weyl_b, incompat_edges, proj, char_decomp).

    Computes Weyl vectors (a, b) via run_weyl_checks on the NC-basis refined
    index, then extracts the Weyl-shifted (q^1, η^0) coefficient at each
    e ∈ {-2,-1,+1,+2} and applies the adjoint projection formula.

    The character decomposition is of the Weyl-shifted c_e polynomial in u.
    """
    R, S = _df.find_rs(P, Q)
    nz_nc = _nz.apply_general_cusp_basis_change(
        nz_data, cusp_idx, a=P, b=Q, c=-R, d=-S
    )
    n_cusps = nz_nc.r
    num_hard = nz_nc.num_hard

    # Build entries at m=0, e ∈ {-2,-1,-1/2,+1/2,+1,+2} plus b-probe
    entries: list = []
    seen: set = set()
    for e_val in (Fraction(-2), Fraction(-1), Fraction(-1,2),
                  Fraction(1,2), Fraction(1), Fraction(2)):
        m_ext = [0]*n_cusps; e_ext = [Fraction(0)]*n_cusps
        e_ext[cusp_idx] = e_val
        key = (tuple(m_ext), tuple(e_ext))
        if key not in seen:
            result = ComputeService.compute_refined_index(nz_nc, m_ext, e_ext, Q_ORDER_HALF)
            if result is not None:
                seen.add(key); entries.append((m_ext, e_ext, result))
    for m_val in (-2, -1, 1, 2):
        m_ext = [0]*n_cusps; e_ext = [Fraction(0)]*n_cusps
        m_ext[cusp_idx] = m_val
        key = (tuple(m_ext), tuple(e_ext))
        if key not in seen:
            result = ComputeService.compute_refined_index(nz_nc, m_ext, e_ext, Q_ORDER_HALF)
            if result is not None:
                seen.add(key); entries.append((m_ext, e_ext, result))

    if not entries:
        return None, None, [], None, {}

    # Run full Weyl check to get (a, b) vectors and refined adjoint proj
    wr = run_weyl_checks(entries, num_hard, cusp_idx=cusp_idx, q_order_half=Q_ORDER_HALF)
    ab = wr.ab
    adj = wr.adjoint

    weyl_a = list(ab.a) if ab else None
    weyl_b = list(ab.b) if ab else None

    # Incompat edges: a[j] ∉ ℤ or 2b[j] ∉ ℤ
    incompat = []
    if ab:
        for j in range(len(ab.a)):
            a_val = ab.a[j]; b_val = ab.b[j]
            if a_val != int(a_val):
                incompat.append(j)
            elif hasattr(b_val, 'denominator') and (2*b_val).denominator != 1:
                incompat.append(j)
            elif isinstance(b_val, float) and (2*b_val) != int(2*b_val):
                incompat.append(j)

    proj = adj.projected_value if adj else None

    # Build character decomposition of the Weyl-shifted c_e polynomial
    decomp: dict[int, int] = {}
    if adj and ab and proj is not None:
        # c_e values from adj.c_e (Fraction keys)
        c_e = adj.c_e  # {Fraction(e): coeff}
        # Also get c_0 and c_±3, c_±4 for full decomposition
        for e_val in (Fraction(0), Fraction(-3), Fraction(3), Fraction(-4), Fraction(4)):
            m_ext = [0]*n_cusps; e_ext = [Fraction(0)]*n_cusps
            e_ext[cusp_idx] = e_val
            result = ComputeService.compute_refined_index(nz_nc, m_ext, e_ext, Q_ORDER_HALF)
            if result is not None:
                if ab:
                    shift_x2 = ab.shift_x2(m_ext, e_ext)
                else:
                    shift_x2 = [0]*num_hard
                coeff = _extract_q1_eta0_coeff_shifted(result, num_hard, shift_x2)
                if coeff != 0:
                    c_e = dict(c_e)
                    c_e[e_val] = coeff

        # Build {u_power: coeff} and decompose into chi_j
        u_poly: dict[int, int] = {int(e*2): c for e, c in c_e.items() if c != 0}
        remaining = dict(u_poly)
        max_j = max((abs(k)//2 for k in remaining), default=0)
        for j in range(max_j, -1, -1):
            c_j = remaining.get(2*j, 0)
            if c_j != 0:
                decomp[j] = c_j
                for e in range(-j, j+1):
                    remaining[2*e] = remaining.get(2*e, 0) - c_j
        remaining = {k:v for k,v in remaining.items() if v != 0}
        if remaining:
            decomp[-1] = remaining  # carry residue

    return weyl_a, weyl_b, incompat, proj, decomp


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

def _expand_range(spec):
    if spec[0] == "list": return list(spec[1])
    prefix, lo, hi = spec[0], spec[1], spec[2]
    pad = 4 if prefix == "v" else 3
    return [f"{prefix}{i:0{pad}d}" for i in range(lo, hi+1)]


def scan_manifold(name: str) -> list[SlopeResult]:
    try:
        _, _easy, nz = ComputeService.load_manifold(name)
    except Exception as exc:
        print(f"  [SKIP] {name}: {exc}")
        return []
    if REQUIRE_SINGLE_CUSP and nz.r != 1:
        print(f"  [SKIP] {name}: {nz.r} cusps")
        return []

    results = []
    for P, Q in _ALL_SLOPES:
        try:
            is_nc = _is_nc(nz, P, Q, CUSP_IDX)
        except Exception as exc:
            print(f"  [WARN] {name} ({P:+d},{Q:+d}) NC check: {exc}")
            continue

        res = SlopeResult(manifold=name, P=P, Q=Q, is_nc=is_nc)

        if is_nc:
            try:
                wa, wb, incompat, proj, decomp = _refined_check(nz, P, Q, CUSP_IDX)
                res.weyl_a = wa; res.weyl_b = wb
                res.incompat_edges = incompat
                res.refined_proj = proj
                res.refined_pass = (proj <= -1) if proj is not None else None
                res.decomp = decomp
            except Exception as exc:
                print(f"  [WARN] {name} ({P:+d},{Q:+d}) refined check: {exc}")

        results.append(res)
    return results


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _vec_str(v):
    if v is None: return "—"
    return "(" + ", ".join(str(x) for x in v) + ")"


def _decomp_str(decomp):
    if not decomp: return "—"
    parts = []
    for j in sorted(k for k in decomp if k >= 0):
        m = decomp[j]
        if m == -1: parts.append(f"-χ_{j}")
        elif m == 1: parts.append(f"χ_{j}")
        else: parts.append(f"{m}χ_{j}")
    if -1 in decomp:
        parts.append(f"(+residue)")
    return " + ".join(parts).replace("+ -", "- ")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    names = []
    for spec in MANIFOLD_RANGES:
        names.extend(_expand_range(spec))

    print(f"Scanning {len(names)} manifold(s), {len(_ALL_SLOPES)} slopes each")
    print(f"Slopes: {_ALL_SLOPES}  |  q_order_half = {Q_ORDER_HALF}")
    print()

    all_results: list[SlopeResult] = []
    for name in names:
        print(f"[{name}]")
        results = scan_manifold(name)
        all_results.extend(results)
        for r in results:
            if not r.is_nc:
                print(f"  slope {r.slope_str()}  —")
                continue
            incompat_str = f"  incompat={r.incompat_edges}" if r.incompat_edges else ""
            if r.refined_proj is None:
                detail = f"  refined proj=?{incompat_str}"
            else:
                tag = "PASS" if r.refined_pass else "FAIL"
                detail = (f"  [{tag}] proj={r.refined_proj}"
                          f"  a={_vec_str(r.weyl_a)}  b={_vec_str(r.weyl_b)}"
                          f"{incompat_str}"
                          f"  [{_decomp_str(r.decomp)}]")
            print(f"  slope {r.slope_str()}  NC{detail}")

    # -----------------------------------------------------------------------
    nc_all    = [r for r in all_results if r.is_nc]
    pass_list = [r for r in nc_all if r.refined_pass is True]
    fail_list = [r for r in nc_all if r.refined_pass is False]
    unk_list  = [r for r in nc_all if r.refined_pass is None]

    print()
    print("=" * 70)
    print("REFINED PROJ ≤ -1  (W-compatible NC cycles)")
    print("=" * 70)
    if pass_list:
        for r in pass_list:
            print(f"  {r.manifold}  {r.slope_str()}  proj={r.refined_proj}"
                  f"  a={_vec_str(r.weyl_a)}  [{_decomp_str(r.decomp)}]")
    else:
        print("  (none)")

    print()
    print("=" * 70)
    print("REFINED PROJ > -1  (NOT W-compatible)")
    print("=" * 70)
    if fail_list:
        for r in fail_list:
            incompat_str = f"  incompat={r.incompat_edges}" if r.incompat_edges else ""
            print(f"  {r.manifold}  {r.slope_str()}  proj={r.refined_proj}"
                  f"  a={_vec_str(r.weyl_a)}{incompat_str}  [{_decomp_str(r.decomp)}]")
    else:
        print("  (none)")

    print()
    print("=" * 70)
    print("FULL TABLE")
    print("=" * 70)
    hdr = f"{'Manifold':<12} {'Slope':>8} {'NC':>5} {'Ref.Pass':>9} {'proj':>5}  a / b / decomp"
    print(hdr); print("-"*len(hdr))
    for r in all_results:
        nc_s = "True " if r.is_nc else "False"
        if not r.is_nc:
            print(f"{r.manifold:<12} {r.slope_str():>8} {nc_s:>5}"); continue
        rp_s = ("True " if r.refined_pass else "False") if r.refined_pass is not None else "  —  "
        pv   = str(r.refined_proj) if r.refined_proj is not None else "—"
        inc  = f" incompat={r.incompat_edges}" if r.incompat_edges else ""
        print(f"{r.manifold:<12} {r.slope_str():>8} {nc_s:>5} {rp_s:>9} {pv:>5}"
              f"  a={_vec_str(r.weyl_a)} b={_vec_str(r.weyl_b)}{inc}"
              f"  [{_decomp_str(r.decomp)}]")

    print()
    print(f"Total NC: {len(nc_all)}  |  Refined PASS (≤-1): {len(pass_list)}"
          f"  |  FAIL: {len(fail_list)}  |  Unknown: {len(unk_list)}")


if __name__ == "__main__":
    main()
