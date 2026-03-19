#!/usr/bin/env python3
"""Quick diagnostic: compare ℓ=2 refined at η=1 with unrefined."""
from fractions import Fraction
from manifold_index.core.manifold import load_manifold
from manifold_index.core.phase_space import find_easy_edges
from manifold_index.core.neumann_zagier import build_neumann_zagier, apply_cusp_basis_change
from manifold_index.core.dehn_filling import compute_filled_index
from manifold_index.core.refined_dehn_filling import (
    compute_filled_refined_index,
    hj_continued_fraction,
)

data = load_manifold("m003")
easy = find_easy_edges(data)
nz = build_neumann_zagier(data, easy)

# Test 1: Slope 1/0 via HJ-CF [0,0] — should give zero
print("=" * 60)
print("Test 1: I^ref_{1/0} should be zero")
print("=" * 60)
ks = hj_continued_fraction(1, 0)
print(f"  HJ-CF(1,0) = {ks}, ℓ={len(ks)}")
ref = compute_filled_refined_index(nz, 0, 1, 0, q_order_half=8, eta_order=5)
eta1 = ref.eta1_series()
print(f"  I^ref_{{1/0}} at η=1: {dict(sorted(eta1.items())[:8])}")
print(f"  is_zero = {ref.is_zero}")

# Test 2: Slope 3/1 via direct (ℓ=1) and via NC (-1,1) → ℓ=2
print("\n" + "=" * 60)
print("Test 2: Slope 3/1 — direct vs NC(-1,1)")
print("=" * 60)

# Direct: slope 3/1, HJ-CF=[3], ℓ=1
ref_direct = compute_filled_refined_index(nz, 0, 3, 1, q_order_half=10, eta_order=5)
eta1_direct = ref_direct.eta1_series()
print(f"  Direct 3/1 (ℓ=1): {[eta1_direct.get(i, 0) for i in range(12)]}")

# Via NC(-1,1): basis change transforms slope
from manifold_index.core.neumann_zagier import apply_cusp_basis_change
nz_bc = apply_cusp_basis_change(nz, 0, -1, 1)

# Transform slope: user (3,1) in original → new basis with NC=(-1,1)
# NC = (P_nc, Q_nc) = (-1, 1)
# If M' = -M + L, L' = ? (basis change M' = P_nc*M + Q_nc*L)
# Need to find how (3,1) transforms.
# The basis change applied to nz_data is (P_nc, Q_nc) = (-1,1).
# apply_cusp_basis_change does M_new = P*M + Q*L, L_new chosen to complete SL(2,Z).
# For user slope p*M + q*L, we need to express in new basis:
#   p*M + q*L = p*(a*M_new + b*L_new) + q*(c*M_new + d*L_new)
# where [[P,Q],[R,S]] = [[-1,1],[R,S]] with -1*S - 1*R = 1 → -S - R = 1
# One solution: R=0, S=-1 → check: -1*(-1) - 1*0 = 1 ✓
# Inverse: [[S,-Q],[-R,P]] = [[-1,-1],[0,-1]]
# (p_new, q_new) = p*S + q*(-R), p*(-Q) + q*P = p*(-1) + 0, p*(-1) + q*(-1) = (-p, -p-q)
# For (p,q) = (3,1): (-3, -3-1) = (-3, -4)
P_new, Q_new = -3, -4
# Normalize sign if needed
if P_new < 0:
    P_new, Q_new = -P_new, -Q_new
P_new, Q_new = 3, 4

ks_new = hj_continued_fraction(P_new, Q_new)
print(f"  NC(-1,1) → slope ({P_new},{Q_new}), HJ-CF={ks_new}, ℓ={len(ks_new)}")

ref_nc = compute_filled_refined_index(nz_bc, 0, P_new, Q_new, q_order_half=10, eta_order=5)
eta1_nc = ref_nc.eta1_series()
print(f"  NC(-1,1) ℓ={len(ks_new)}: {[eta1_nc.get(i, 0) for i in range(12)]}")

# Unrefined for comparison
unr = compute_filled_index(nz, 0, 3, 1, q_order_half=10)
print(f"  Unrefined 3/1: {[unr.series.get(i, 0) for i in range(12)]}")

# Test 3: Slope 1/2 (the actual test case)
print("\n" + "=" * 60)
print("Test 3: Slope 1/2 — ℓ=2 vs unrefined")
print("=" * 60)
ks_12 = hj_continued_fraction(1, 2)
print(f"  HJ-CF(1,2) = {ks_12}")

unr_12 = compute_filled_index(nz, 0, 1, 2, q_order_half=8)
ref_12 = compute_filled_refined_index(nz, 0, 1, 2, q_order_half=8, eta_order=6)
eta1_12 = ref_12.eta1_series()
print(f"  Unrefined 1/2: {[unr_12.series.get(i, 0) for i in range(10)]}")
print(f"  Refined η=1:   {[eta1_12.get(i, 0) for i in range(10)]}")

# Test 4: Slope 5/1 via NC(-1,1)
print("\n" + "=" * 60)
print("Test 4: Slope 5/1 — direct vs NC(-1,1)")
print("=" * 60)
ref_51 = compute_filled_refined_index(nz, 0, 5, 1, q_order_half=10, eta_order=5)
eta1_51 = ref_51.eta1_series()
print(f"  Direct 5/1 (ℓ=1): {[eta1_51.get(i, 0) for i in range(12)]}")

# NC(-1,1) → slope (5,6)
P5, Q5 = 5, 6
ks_56 = hj_continued_fraction(P5, Q5)
print(f"  NC(-1,1) → slope ({P5},{Q5}), HJ-CF={ks_56}, ℓ={len(ks_56)}")

ref_nc51 = compute_filled_refined_index(nz_bc, 0, P5, Q5, q_order_half=10, eta_order=5)
eta1_nc51 = ref_nc51.eta1_series()
print(f"  NC(-1,1) ℓ={len(ks_56)}: {[eta1_nc51.get(i, 0) for i in range(12)]}")

unr_51 = compute_filled_index(nz, 0, 5, 1, q_order_half=10)
print(f"  Unrefined 5/1: {[unr_51.series.get(i, 0) for i in range(12)]}")
