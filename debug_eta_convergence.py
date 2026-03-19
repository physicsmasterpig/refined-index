#!/usr/bin/env python3
"""Test: does increasing eta_order make ℓ≥2 converge to unrefined?"""
from fractions import Fraction
from manifold_index.core.manifold import load_manifold
from manifold_index.core.phase_space import find_easy_edges
from manifold_index.core.neumann_zagier import build_neumann_zagier, apply_cusp_basis_change
from manifold_index.core.dehn_filling import compute_filled_index
from manifold_index.core.refined_dehn_filling import compute_filled_refined_index

data = load_manifold("m003")
easy = find_easy_edges(data)
nz = build_neumann_zagier(data, easy)

# Test with slope 1/2 (HJ-CF [1,2], ℓ=2)
P, Q = 1, 2
qq_order = 8

unr = compute_filled_index(nz, 0, P, Q, q_order_half=qq_order)
unr_list = [unr.series.get(i, Fraction(0)) for i in range(10)]
print(f"Unrefined {P}/{Q}: {unr_list}")

for eta_ord in [3, 5, 8, 12, 16, 20]:
    ref = compute_filled_refined_index(nz, 0, P, Q, q_order_half=qq_order, eta_order=eta_ord)
    eta1 = ref.eta1_series()
    eta1_list = [eta1.get(i, Fraction(0)) for i in range(10)]
    stable = qq_order - 2 * eta_ord
    print(f"  η_order={eta_ord:2d} (stable≤{stable:3d}): {eta1_list}")

# Test with slope 3/4 via NC(-1,1) 
print()
nz_bc = apply_cusp_basis_change(nz, 0, -1, 1)
P2, Q2 = 3, 4

unr2 = compute_filled_index(nz, 0, 3, 1, q_order_half=10)
unr2_list = [unr2.series.get(i, Fraction(0)) for i in range(12)]
print(f"Unrefined 3/1 (= 3/4 in original basis): {unr2_list}")

for eta_ord in [3, 5, 8, 12, 16]:
    ref = compute_filled_refined_index(nz_bc, 0, P2, Q2, q_order_half=10, eta_order=eta_ord)
    eta1 = ref.eta1_series()
    eta1_list = [eta1.get(i, Fraction(0)) for i in range(12)]
    stable = 10 - 2 * eta_ord
    print(f"  η_order={eta_ord:2d} (stable≤{stable:3d}): {eta1_list}")
