#!/usr/bin/env python3
"""
Consistency check: m003, NC=(1,0) standard, slope 3/2 surgery.

3/2 is non-integer → Q=2, HJ-CF of 3/2 = [2,2], ℓ=2 → IS chain.

Compare:
  (A) Unrefined Dehn filling at 3/2
  (B) Refined Dehn filling at 3/2, then set all η=1

Expected: (A) == (B)
"""

from fractions import Fraction
from manifold_index.core.manifold import load_manifold
from manifold_index.core.neumann_zagier import build_neumann_zagier
from manifold_index.core.phase_space import find_easy_edges
from manifold_index.core.dehn_filling import compute_filled_index
from manifold_index.core.refined_dehn_filling import (
    compute_filled_refined_index,
    hj_continued_fraction,
)

data = load_manifold("m003")
easy = find_easy_edges(data)
nz = build_neumann_zagier(data, easy)

P, Q = 3, 2
qq_order = 10

print(f"m003, NC=(1,0), slope = {P}/{Q}")
print(f"HJ-CF({P}/{Q}) = {hj_continued_fraction(P, Q)}")
print()

# =====================================================================
# (A) Unrefined Dehn filling
# =====================================================================
print("=" * 60)
print("(A) UNREFINED Dehn filling at 3/2")
print("=" * 60)

unrefined = compute_filled_index(
    nz, cusp_idx=0, P=P, Q=Q,
    m_other=[], e_other=[],
    q_order_half=qq_order,
)
print(f"Unrefined result:")
unref_series = unrefined.series
for qq in sorted(unref_series.keys()):
    if unref_series[qq] != 0:
        print(f"  qq^{qq}: {unref_series[qq]}")

# =====================================================================
# (B) Refined Dehn filling, then η=1
# =====================================================================
print()
print("=" * 60)
print("(B) REFINED Dehn filling at 3/2, then set all η=1")
print("=" * 60)

for eta_ord in [3, 5, 8, 12]:
    print(f"\n--- eta_order = {eta_ord} ---")
    
    refined = compute_filled_refined_index(
        nz, cusp_idx=0, P=P, Q=Q,
        q_order_half=qq_order, eta_order=eta_ord,
        verbose=True,
    )
    print(f"  has_cusp_eta = {refined.has_cusp_eta}")
    print(f"  HJ = {refined.hj_ks}")
    print(f"  Non-zero entries: {len(refined.series)}")
    
    # Set all η = 1
    projected = refined.eta1_series()
    
    print(f"\n  Comparison (η=1 projection vs unrefined):")
    all_qq = sorted(set(list(unref_series.keys()) + list(projected.keys())))
    all_match = True
    for qq in all_qq:
        v_unref = unref_series.get(qq, 0)
        v_proj = projected.get(qq, Fraction(0))
        status = "✓" if v_unref == v_proj else "✗"
        if v_unref != v_proj:
            all_match = False
        if v_unref != 0 or v_proj != 0:
            print(f"    qq^{qq}: unrefined={v_unref}, refined(η=1)={v_proj} {status}")
    
    if all_match:
        print("  >>> ALL MATCH ✓")
    else:
        print("  >>> MISMATCH ✗")
    
    # Show a few η_cusp breakdown entries
    print(f"\n  Sample refined entries (first 20):")
    for key in sorted(refined.series.keys())[:20]:
        c = refined.series[key]
        if c != 0:
            print(f"    {key}: {c}")
