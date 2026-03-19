"""
Debug: 3/2 surgery with inflated qq_order for IS kernel.

Key insight from debug_is_chain_kernel2.py:
  The IS chain kernel at η=1 MATCHES the ordinary K(3,2) only when
  qq_order >> eta_order (stable region ≈ qq ≤ qq_order - 2*eta_order).

This script tests whether:
1. Using inflated qq_order in the IS kernel fixes the η=1 mismatch
2. Individual η^k coefficients remain stable regardless

For m003, slope 3/2, HJ-CF = [2, 2].
"""

import sys
from fractions import Fraction

# Add the source directory
sys.path.insert(0, "src")

from manifold_index.core.manifold import load_manifold
from manifold_index.core.phase_space import find_easy_edges
from manifold_index.core.neumann_zagier import build_neumann_zagier, apply_cusp_basis_change
from manifold_index.core.dehn_filling import compute_filled_index
from manifold_index.core.refined_dehn_filling import (
    compute_filled_refined_index,
    hj_continued_fraction,
)


# =====================================================================
# Setup m003
# =====================================================================
mdata = load_manifold("m003")
easy = find_easy_edges(mdata)
nz = build_neumann_zagier(mdata, easy)

print(f"m003: n={nz.n}, r={nz.r}, num_hard={nz.num_hard}")

# Use NC = (1, 0) as meridian standard
NC_P, NC_Q = 1, 0
nz_nc = apply_cusp_basis_change(nz, cusp_idx=0, P=NC_P, Q=NC_Q)
print(f"NC cycle: ({NC_P}, {NC_Q})")

# Slope 3/2 surgery
P, Q = 3, 2
ks = hj_continued_fraction(P, Q)
print(f"Slope: {P}/{Q}, HJ-CF = {ks}")

# =====================================================================
# Unrefined filling (baseline)
# =====================================================================
q_order = 10
unref = compute_filled_index(
    nz_nc, cusp_idx=0, P=P, Q=Q,
    m_other=[], e_other=[],
    q_order_half=q_order,
)

print(f"\n{'='*60}")
print(f"Unrefined filling (baseline), q_order_half={q_order}")
print(f"{'='*60}")
for qq_p in sorted(unref.series.keys()):
    if abs(unref.series[qq_p]) > 0:
        print(f"  qq^{qq_p} = {unref.series[qq_p]}")

# =====================================================================
# Refined filling with various eta_order and inflated qq_order
# =====================================================================
print(f"\n{'='*60}")
print(f"Refined filling (IS chain), inflated qq_order")
print(f"{'='*60}")

eta_orders = [3, 5, 8, 12]

for eta_ord in eta_orders:
    # Use inflated qq_order = q_order + 2*eta_order for IS stability
    inflated_qq = q_order + 2 * eta_ord
    
    filled = compute_filled_refined_index(
        nz_nc, cusp_idx=0, P=P, Q=Q,
        m_other=[], e_other=[],
        q_order_half=inflated_qq,
        eta_order=eta_ord,
    )
    
    # η=1 projection (but only look at qq ≤ q_order)
    eta1 = filled.eta1_series()
    eta1_truncated = {k: v for k, v in eta1.items() if k <= q_order}
    
    print(f"\n  eta_order={eta_ord}, internal_qq={inflated_qq}")
    print(f"  η=1 projection (qq ≤ {q_order}):")
    
    for qq_p in range(0, q_order + 1, 2):
        v_ref = eta1_truncated.get(qq_p, Fraction(0))
        v_unref = unref.series.get(qq_p, Fraction(0))
        match = "✓" if v_ref == v_unref else f"✗ (expected {v_unref})"
        if v_ref != 0 or v_unref != 0:
            print(f"    qq^{qq_p}: {v_ref} {match}")

# =====================================================================
# Also check individual η coefficients stability
# =====================================================================
print(f"\n{'='*60}")
print(f"Individual η^k coefficient stability (inflated qq_order)")
print(f"{'='*60}")

# Collect coefficients at fixed (qq, eta_cusp) across eta_orders
target_keys_by_eta = {}  # (qq, eta_cusp) -> {eta_order: value}

for eta_ord in eta_orders:
    inflated_qq = q_order + 2 * eta_ord
    filled = compute_filled_refined_index(
        nz_nc, cusp_idx=0, P=P, Q=Q,
        m_other=[], e_other=[],
        q_order_half=inflated_qq,
        eta_order=eta_ord,
    )
    
    for key, val in filled.series.items():
        if abs(val) > 0 and key[0] <= q_order:
            # key = (qq_power, cusp_eta)  (num_hard=1 for m003 but the
            # hard η dimension is also present)
            # For simplicity, collect by the full key but only qq ≤ q_order
            qp = key[0]
            rest = key[1:]
            tkey = (qp, rest)
            if tkey not in target_keys_by_eta:
                target_keys_by_eta[tkey] = {}
            target_keys_by_eta[tkey][eta_ord] = val

# Show stability for first few keys
print(f"\nKey = (qq_power, η_dims) → value at each eta_order")
sorted_keys = sorted(target_keys_by_eta.keys())[:30]
for tkey in sorted_keys:
    vals = target_keys_by_eta[tkey]
    unique_vals = set(vals.values())
    stable = "STABLE" if len(unique_vals) == 1 else "VARIES"
    val_str = ", ".join(f"η_ord={eo}:{v}" for eo, v in sorted(vals.items()))
    print(f"  ({tkey[0]}, {tkey[1]}): {stable}  [{val_str}]")
