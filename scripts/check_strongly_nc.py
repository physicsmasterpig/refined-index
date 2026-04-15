#!/usr/bin/env -S python3 -u
"""
check_strongly_nc.py
====================
For each census manifold in configurable ranges, test every primitive slope
(P, Q) with P in {-1, 0, 1} and Q in {0, 1}:

  1. Is P·α + Q·β an NC cycle?    (filled 3D index = 0)
  2. Is it MARGINAL?               (unrefined q^1 adjoint projection ≥ 0)

The check uses I^{3D}(m=0; u, q) in the NC basis.  Setting η=1 gives the
3D index directly.  The adjoint projection formula extracts the multiplicity
of χ_1 in the q^1 coefficient:

    proj = (1/2)(c_{-1} + c_{+1} - c_{-2} - c_{+2})

where c_e = [q^1] I^{3D}(m=0, e).

  proj ≤ -1  →  non-marginal  →  refined kernel K^ref may be used
  proj ≥  0  →  MARGINAL      →  unrefined kernel K(P,Q) is used

Also shows the χ_j character decomposition of the q^1 coefficient.

Usage
-----
  python check_strongly_nc.py

Configure MANIFOLD_RANGES and Q_ORDER_HALF at the top.
"""

from __future__ import annotations

import sys
import os
from fractions import Fraction
from math import gcd
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MANIFOLD_RANGES = [
    ("m", 3, 50),
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
from manifold_index.core.index_3d import compute_index_3d_python

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
    cusp: int
    P: int
    Q: int
    is_nc: bool
    is_marginal: bool | None = None   # True = proj ≥ 0 (marginal)
    unrefined_q1_proj: int | None = None
    decomp: dict[int, int] = field(default_factory=dict)

    def slope_str(self) -> str:
        return f"({self.P:+d},{self.Q:+d})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expand_range(spec: tuple) -> list[str]:
    if spec[0] == "list":
        return list(spec[1])
    prefix, lo, hi = spec[0], spec[1], spec[2]
    pad = 4 if prefix in ("v",) else 3
    return [f"{prefix}{i:0{pad}d}" for i in range(lo, hi + 1)]


def _is_nc(nz_data, P: int, Q: int, cusp_idx: int) -> bool:
    filled = _df.compute_filled_index(
        nz_data, cusp_idx=cusp_idx, P=P, Q=Q,
        m_other=[0] * (nz_data.r - 1),
        e_other=[0] * (nz_data.r - 1),
        q_order_half=Q_ORDER_HALF,
    )
    return filled.is_stably_zero()


def _unrefined_q1_check(
    nz_data, P: int, Q: int, cusp_idx: int
) -> tuple[int | None, dict[int, int]]:
    """
    Returns (proj, char_decomp).
      proj        = (1/2)(c_{-1}+c_{+1}-c_{-2}-c_{+2}), or None if incomplete.
      char_decomp = {j: multiplicity of χ_j in f_{q^1}(u)}

    is_marginal iff proj ≥ 0.
    """
    R, S = _df.find_rs(P, Q)
    nz_nc = _nz.apply_general_cusp_basis_change(
        nz_data, cusp_idx, a=P, b=Q, c=-R, d=-S
    )
    n = nz_nc.r

    needed = [Fraction(-2), Fraction(-1), Fraction(1), Fraction(2)]
    c_e: dict[Fraction, int] = {}
    for e_val in needed:
        m_ext = [0]*n; e_ext = [Fraction(0)]*n
        e_ext[cusp_idx] = e_val
        r3d = compute_index_3d_python(nz_nc, m_ext, e_ext, Q_ORDER_HALF)
        idx = 2 - r3d.min_power
        c_e[e_val] = r3d.coeffs[idx] if 0 <= idx < len(r3d.coeffs) else 0

    num = (c_e[Fraction(-1)] + c_e[Fraction(1)]
           - c_e[Fraction(-2)] - c_e[Fraction(2)])
    if num % 2 != 0:
        return None, {}
    proj = num // 2

    # Extra e-values for full character decomposition
    for e_val in [Fraction(0), Fraction(-3), Fraction(3), Fraction(-4), Fraction(4)]:
        m_ext = [0]*n; e_ext = [Fraction(0)]*n
        e_ext[cusp_idx] = e_val
        r3d = compute_index_3d_python(nz_nc, m_ext, e_ext, Q_ORDER_HALF)
        idx = 2 - r3d.min_power
        q1 = r3d.coeffs[idx] if 0 <= idx < len(r3d.coeffs) else 0
        if q1 != 0:
            c_e[e_val] = q1

    # Decompose f_{q^1}(u) = Σ_e c_e u^{2e} into SU(2) characters χ_j
    u_poly: dict[int, int] = {int(e*2): c for e, c in c_e.items() if c != 0}
    remaining = dict(u_poly)
    max_j = max((abs(k)//2 for k in remaining), default=0)
    decomp: dict[int, int] = {}
    for j in range(max_j, -1, -1):
        c_j = remaining.get(2*j, 0)
        if c_j != 0:
            decomp[j] = c_j
            for e in range(-j, j+1):
                remaining[2*e] = remaining.get(2*e, 0) - c_j
    remaining = {k: v for k, v in remaining.items() if v != 0}
    if remaining:
        decomp[-1] = remaining   # residue

    return proj, decomp


# ---------------------------------------------------------------------------
# Per-manifold scan
# ---------------------------------------------------------------------------

def scan_manifold(name: str) -> list[SlopeResult]:
    try:
        _, _easy, nz = ComputeService.load_manifold(name)
    except Exception as exc:
        print(f"  [SKIP] {name}: {exc}")
        return []
    if REQUIRE_SINGLE_CUSP and nz.r != 1:
        print(f"  [SKIP] {name}: {nz.r} cusps")
        return []

    results: list[SlopeResult] = []
    for P, Q in _ALL_SLOPES:
        try:
            is_nc = _is_nc(nz, P, Q, CUSP_IDX)
        except Exception as exc:
            print(f"  [WARN] {name} ({P:+d},{Q:+d}) NC check: {exc}")
            continue

        res = SlopeResult(manifold=name, cusp=CUSP_IDX, P=P, Q=Q, is_nc=is_nc)

        if is_nc:
            try:
                proj, decomp = _unrefined_q1_check(nz, P, Q, CUSP_IDX)
                res.unrefined_q1_proj = proj
                res.is_marginal = (proj >= 0) if proj is not None else None
                res.decomp = decomp
            except Exception as exc:
                print(f"  [WARN] {name} ({P:+d},{Q:+d}) unref. q^1 check: {exc}")

        results.append(res)
    return results


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _decomp_str(decomp: dict) -> str:
    if not decomp:
        return "—"
    parts = []
    for j in sorted(k for k in decomp if k >= 0):
        m = decomp[j]
        if m == -1:   parts.append(f"-χ_{j}")
        elif m == 1:  parts.append(f"χ_{j}")
        else:         parts.append(f"{m}χ_{j}")
    if -1 in decomp:
        parts.append("(residue)")
    return " + ".join(parts).replace("+ -", "- ")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    names: list[str] = []
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
                continue
            if r.unrefined_q1_proj is None:
                detail = "  unref. q^1=?"
            else:
                tag = "MARGINAL" if r.is_marginal else "non-marginal"
                detail = (f"  [{tag}] proj={r.unrefined_q1_proj}"
                          f"  [{_decomp_str(r.decomp)}]")
            print(f"  slope {r.slope_str()}  NC  {detail}")

    # -----------------------------------------------------------------------
    nc_all       = [r for r in all_results if r.is_nc]
    marginal     = [r for r in nc_all if r.is_marginal is True]
    non_marginal = [r for r in nc_all if r.is_marginal is False]
    undecided    = [r for r in nc_all if r.is_marginal is None]

    print()
    print("=" * 70)
    print("MARGINAL NC CYCLES  (proj ≥ 0 — unrefined kernel)")
    print("=" * 70)
    if marginal:
        for r in marginal:
            print(f"  {r.manifold}  {r.slope_str()}  proj={r.unrefined_q1_proj}"
                  f"  [{_decomp_str(r.decomp)}]")
    else:
        print("  (none)")

    print()
    print("=" * 70)
    print("NON-MARGINAL NC CYCLES  (proj ≤ -1 — refined kernel may be used)")
    print("=" * 70)
    if non_marginal:
        for r in non_marginal:
            print(f"  {r.manifold}  {r.slope_str()}  proj={r.unrefined_q1_proj}"
                  f"  [{_decomp_str(r.decomp)}]")
    else:
        print("  (none)")

    print()
    print("=" * 70)
    print("FULL TABLE")
    print("=" * 70)
    hdr = (f"{'Manifold':<12}  {'Slope':>8}  {'NC':>5}  {'Marginal':>9}"
           f"  {'proj':>5}  f_{{q^1}} decomp")
    print(hdr); print("-" * len(hdr))
    for r in all_results:
        nc_s  = "True " if r.is_nc else "False"
        marg_s = ("True " if r.is_marginal else "False") if r.is_marginal is not None else "  —  "
        pv     = str(r.unrefined_q1_proj) if r.unrefined_q1_proj is not None else "—"
        dec    = _decomp_str(r.decomp) if r.is_nc else ""
        print(f"{r.manifold:<12}  {r.slope_str():>8}  {nc_s:>5}  {marg_s:>9}"
              f"  {pv:>5}  {dec}")

    print()
    print(f"Total NC: {len(nc_all)}  |  Marginal (proj≥0): {len(marginal)}"
          f"  |  Non-marginal (proj≤-1): {len(non_marginal)}"
          f"  |  Undetermined: {len(undecided)}")


if __name__ == "__main__":
    main()
