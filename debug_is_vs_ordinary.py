#!/usr/bin/env python3
"""
Consistency check: For m003 slope (3,1), compare:
  Basis 1 (NC=(1,0)): ℓ=1, ordinary kernel K(3,1) → result R₁(q, η_hard)
  Basis 2 (NC=(-1,1)): ℓ=2, IS chain K^ref(3,4) → result R₂(q, η_hard, η_cusp)

Expected: R₂(q, η_hard, η_cusp=1) = R₁(q, η_hard)
(up to possible η_hard → −η_hard from basis coupling)

This is the user's consistency requirement:
"turning off the Dehn filling refinement [η_cusp=1] in one slope
 should reproduce the integer one"
"""

from fractions import Fraction
from manifold_index.core.manifold import load_manifold
from manifold_index.core.neumann_zagier import build_neumann_zagier, apply_cusp_basis_change
from manifold_index.core.phase_space import find_easy_edges
from manifold_index.core.refined_dehn_filling import compute_filled_refined_index

data = load_manifold("m003")
easy = find_easy_edges(data)

# =====================================================================
# Basis 1: NC = (1,0) — default, slope (3,1) → ℓ=1
# =====================================================================
nz1 = build_neumann_zagier(data, easy)
qq_order = 10

print("=" * 70)
print("Basis 1: default (NC=(1,0)), slope (3,1) → ℓ=1, ordinary kernel")
print("=" * 70)

result1 = compute_filled_refined_index(
    nz1, cusp_idx=0, P=3, Q=1,
    q_order_half=qq_order, eta_order=5, verbose=True,
)
print(f"\nBasis 1 has_cusp_eta = {result1.has_cusp_eta}")
print(f"Basis 1 HJ = {result1.hj_ks}")
print(f"Non-zero entries: {len(result1.series)}")

# Collect basis 1 as {(qq, η_hard_exp): coeff}
basis1_coeffs = {}
for key, c in sorted(result1.series.items()):
    if c != 0:
        basis1_coeffs[key] = c
        print(f"  {key}: {c}")

# =====================================================================
# Basis 2: NC = (-1,1), slope (3,1) → transformed to (3,4), ℓ=2
# =====================================================================
nz2 = build_neumann_zagier(data, easy)
apply_cusp_basis_change(nz2, 0, -1, 1)

print("\n" + "=" * 70)
print("Basis 2: NC=(-1,1), slope (3,4) → ℓ=2, IS chain")
print("=" * 70)

# Try several eta_order values to see convergence
for eta_ord in [3, 5, 8, 12]:
    print(f"\n--- eta_order = {eta_ord} ---")
    result2 = compute_filled_refined_index(
        nz2, cusp_idx=0, P=3, Q=4,
        q_order_half=qq_order, eta_order=eta_ord, verbose=True,
    )
    print(f"Basis 2 has_cusp_eta = {result2.has_cusp_eta}")
    print(f"Basis 2 HJ = {result2.hj_ks}")
    print(f"Non-zero entries: {len(result2.series)}")

    # Project to η_cusp = 1: sum over all cusp_eta values
    # key = (qq, 2*η_hard, cusp_eta) → sum coeff over cusp_eta
    projected = {}
    for key, c in result2.series.items():
        if c == 0:
            continue
        proj_key = key[:-1]  # drop cusp_eta dimension
        projected[proj_key] = projected.get(proj_key, Fraction(0)) + c

    # Clean zeros
    projected = {k: v for k, v in projected.items() if v != 0}

    # Compare with basis 1
    all_keys = sorted(set(list(basis1_coeffs.keys()) + list(projected.keys())))
    match = True
    match_with_eta_flip = True
    for key in all_keys[:15]:  # show first 15
        b1 = basis1_coeffs.get(key, Fraction(0))
        b2 = projected.get(key, Fraction(0))
        # Also check η_hard → −η_hard
        flipped_key = (key[0],) + tuple(-x for x in key[1:])
        b2_flip = projected.get(flipped_key, Fraction(0))
        
        status = "✓" if b1 == b2 else "✗"
        flip_status = "✓" if b1 == b2_flip else "✗"
        if b1 != b2:
            match = False
        if b1 != b2_flip:
            match_with_eta_flip = False
        print(f"  {key}: basis1={b1}, IS(η_cusp=1)={b2} {status}  (η_flip={b2_flip} {flip_status})")

    if match:
        print("  >>> MATCH: IS chain η_cusp=1 reproduces ordinary kernel ✓")
    elif match_with_eta_flip:
        print("  >>> MATCH with η_hard→−η_hard ✓")
    else:
        print("  >>> MISMATCH ✗")

    # Also show η_cusp breakdown for a few qq powers
    print(f"\n  η_cusp breakdown at qq=0:")
    for key, c in sorted(result2.series.items()):
        if key[0] == 0 and c != 0:
            print(f"    {key}: {c}")
    
    print(f"\n  η_cusp breakdown at qq=2:")
    for key, c in sorted(result2.series.items()):
        if key[0] == 2 and c != 0:
            print(f"    {key}: {c}")

print("\n" + "=" * 70)
print("FULLY UNREFINED comparison (set ALL η = 1)")
print("=" * 70)
unref1 = result1.eta1_series()
# For basis 2, use the last eta_order result
unref2 = result2.eta1_series()
for qq in sorted(set(list(unref1.keys()) + list(unref2.keys()))):
    v1 = unref1.get(qq, Fraction(0))
    v2 = unref2.get(qq, Fraction(0))
    status = "✓" if v1 == v2 else "✗"
    print(f"  qq^{qq}: basis1={v1}, basis2={v2} {status}")
