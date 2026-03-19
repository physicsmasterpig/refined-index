"""
Diagnostic: Check whether individual η_cusp^k coefficients stabilize
as eta_order increases.

Hypothesis: The ℓ≥2 algorithm IS correct for individual η terms.
The divergence at η=1 is expected because IS(η=1)=δ requires infinite
eta_order.  If individual coefficients stabilize, the code is correct.
"""
import sys
sys.path.insert(0, "src")

from fractions import Fraction
from manifold_index.core.manifold import load_manifold
from manifold_index.core.neumann_zagier import build_neumann_zagier
from manifold_index.core.phase_space import find_easy_edges
from manifold_index.core.refined_dehn_filling import compute_filled_refined_index

data = load_manifold("m003")
easy = find_easy_edges(data)
nz = build_neumann_zagier(data, easy)

# Use HJ=[0,0] (slope 1/0, meridian) for ℓ=2 test.
# The unrefined I_{1/0}(m003) = 0 (meridian is NC).
# So at η_cusp=1, each qq coefficient should sum to 0.

qq_order = 8

print("=" * 70)
print("Individual η_cusp^k coefficients of I^ref_{1/0}(m003) via ℓ=2 [0,0]")
print("=" * 70)

for eta_order in [3, 5, 8, 12]:
    result = compute_filled_refined_index(
        nz, cusp_idx=0, P=1, Q=0,
        q_order_half=qq_order, eta_order=eta_order,
    )
    
    # Group by (qq_power, cusp_eta), sum over hard-eta dims
    # Since num_hard=1 for m003, key = (qq, 2*η_hard, cusp_eta)
    # For simplicity, also set η_hard=1 (sum over hard dim)
    
    by_cusp_eta: dict[int, dict[int, Fraction]] = {}  # cusp_eta → {qq → coeff}
    for key, c in result.series.items():
        qq_p = key[0]
        cusp_eta = key[-1]  # last dim is cusp_eta
        if cusp_eta not in by_cusp_eta:
            by_cusp_eta[cusp_eta] = {}
        by_cusp_eta[cusp_eta][qq_p] = by_cusp_eta[cusp_eta].get(qq_p, Fraction(0)) + c
    
    print(f"\neta_order={eta_order:2d}:")
    # Show cusp_eta=0, ±1, ±2 coefficients (qq^0 through qq^4)
    for ce in sorted(by_cusp_eta.keys()):
        if abs(ce) > 5:
            continue
        coeffs = by_cusp_eta[ce]
        vals = [str(coeffs.get(qq, 0)) for qq in range(0, min(6, qq_order+1))]
        print(f"  η_cusp^{ce:+d}: qq^[0..5] = [{', '.join(vals)}]")

print("\n" + "=" * 70)
print("Now test slope (3,1) with NC=(-1,1), HJ=[1,4], ℓ=2")
print("Compare with NC=(1,0), HJ=[3], ℓ=1")
print("=" * 70)

# ℓ=1 reference (NC=(1,0), slope P'=3, Q'=1)
from manifold_index.core.basis_selection import make_basis_selection
from manifold_index.core.dehn_filling import find_non_closable_cycles

# Direct: use slope (3,1) with ℓ=1 path
ref_l1 = compute_filled_refined_index(
    nz, cusp_idx=0, P=3, Q=1,
    q_order_half=qq_order, eta_order=5,
)
print(f"\nℓ=1 (P=3,Q=1): has_cusp_eta={ref_l1.has_cusp_eta}, HJ={ref_l1.hj_ks}")

# Show ℓ=1 result (η_hard only, no cusp_eta)
l1_eta1 = ref_l1.eta1_series()
l1_vals = [str(l1_eta1.get(qq, 0)) for qq in range(0, min(10, qq_order+1))]
print(f"  η=1: qq^[0..9] = [{', '.join(l1_vals)}]")

# ℓ=2 (need basis change from NC=(-1,1))
# The basis change transforms slope (3,1) to some (P', Q') with |Q'|≥2.
# Using NC=(-1,1): Q' = P_nc*Q_user - Q_nc*P_user = (-1)*1 - 1*3 = -4
# So P'/Q' has HJ-CF expansion.
# Actually, let me compute P' properly.
# Basis matrix [[P_nc, a], [Q_nc, b]] = [[-1, a], [1, b]] with -b - a = 1
# Choose a=0, b=-1 (or a=-1, b=0)
# Inverse: [[b, -a], [-Q_nc, P_nc]] = [[-1, 0], [-1, -1]]
# P' = b*P_user - a*Q_user = -1*3 - 0*1 = -3
# Q' = -Q_nc*P_user + P_nc*Q_user = -1*3 + (-1)*1 = -4
# So slope = -3/-4 = 3/4.  HJ-CF of 3/4?
# But that doesn't match [1,4]... let me check.

# Actually the slope transform depends on the exact basis change matrix.
# Let me just use the direct call with (P=3,Q=1) and see the HJ-CF.
# The code computes HJ-CF of P/Q = 3/1 = [3] → ℓ=1.
# To get ℓ=2, we need to use the basis-changed slope.

# The NC cycle (-1,1) means new meridian = -M + L.
# Extended R,S: det = P_nc*S_nc - Q_nc*R_nc = 1
# [-1, R_nc; 1, S_nc] with -S_nc - R_nc = 1
# E.g., R_nc=0, S_nc=-1: [[-1,0],[1,-1]], det=-1*(-1)-0*1=1 ✓
# Inverse: [[-1,0],[-1,-1]]
# User slope (3,1) → (P',Q') = inverse * (3,1)^T
# P' = -1*3 + 0*1 = -3
# Q' = -1*3 + (-1)*1 = -4
# Slope = P'/Q' = -3/-4 = 3/4
# HJ-CF of 3/4: ceil(3/4)=1, remainder=1-3/4=1/4, reciprocal=4. So [1,4] ✓

# To test ℓ=2 for the SAME physical slope (3,1), we'd need the basis-changed NZ data.
# But compute_filled_refined_index takes the slope directly — it doesn't do basis change.
# We'd need to apply the basis change to the NZ data first.
# This is complex, so let's just check coefficient stability for the 1/0 case.

print("\n" + "=" * 70)
print("Coefficient stability check for 1/0 (HJ=[0,0])")
print("Each η_cusp^k coeff should stabilize once eta_order ≥ |k|")
print("=" * 70)

prev_data = {}
for eta_order in [3, 5, 8, 12, 16]:
    result = compute_filled_refined_index(
        nz, cusp_idx=0, P=1, Q=0,
        q_order_half=qq_order, eta_order=eta_order,
    )
    
    by_cusp_eta: dict[int, dict[int, Fraction]] = {}
    for key, c in result.series.items():
        qq_p = key[0]
        cusp_eta = key[-1]
        if cusp_eta not in by_cusp_eta:
            by_cusp_eta[cusp_eta] = {}
        by_cusp_eta[cusp_eta][qq_p] = by_cusp_eta[cusp_eta].get(qq_p, Fraction(0)) + c
    
    # Check stability: compare cusp_eta=0 and cusp_eta=1 with previous
    for ce in [0, 1, 2]:
        key = (ce, eta_order)
        coeffs = by_cusp_eta.get(ce, {})
        vals = tuple(coeffs.get(qq, Fraction(0)) for qq in range(6))
        
        prev_key = (ce, "prev")
        if prev_key in prev_data:
            old_vals = prev_data[prev_key]
            changed = any(vals[i] != old_vals[i] for i in range(len(vals)))
            status = "CHANGED!" if changed else "stable ✓"
        else:
            status = "(first)"
        
        prev_data[prev_key] = vals
        vals_str = [str(v) for v in vals]
        print(f"  eta_order={eta_order:2d}, η_cusp^{ce}: [{', '.join(vals_str)}] {status}")
