"""
m003: Refined index I^ref(m, e; η) in both NC-cycle bases.

Basis 1 (default): NC cycle = (1, 0)  — SnaPy meridian
Basis 2:           NC cycle = (-1, 1) — new meridian = -M + L

For each basis, show I^ref at small (m, e) values.
"""
import sys
sys.path.insert(0, "src")

from fractions import Fraction
from manifold_index.core.manifold import load_manifold
from manifold_index.core.neumann_zagier import build_neumann_zagier, apply_cusp_basis_change
from manifold_index.core.phase_space import find_easy_edges
from manifold_index.core.refined_index import compute_refined_index

data = load_manifold("m003")
easy = find_easy_edges(data)
nz_default = build_neumann_zagier(data, easy)

print(f"m003: n={nz_default.n}, r={nz_default.r}, num_hard={nz_default.num_hard}")
print()

qq_order = 10  # q^{1/2} cutoff

def format_refined(result, num_hard):
    """Format a refined index result as η-polynomial coefficients at each qq power."""
    if not result:
        return "  0"
    
    # Group by qq_power
    by_qq = {}
    for key, c in result.items():
        qq_p = key[0]
        eta_key = key[1:]  # (2*η_0, ...)
        if qq_p not in by_qq:
            by_qq[qq_p] = {}
        by_qq[qq_p][eta_key] = c
    
    lines = []
    for qq_p in sorted(by_qq.keys()):
        terms = by_qq[qq_p]
        if num_hard == 1:
            # Format as polynomial in η (stored as 2*exp)
            parts = []
            for (eta2,), c in sorted(terms.items()):
                if c == 0:
                    continue
                exp = Fraction(eta2, 2)
                if exp == 0:
                    parts.append(f"{c}")
                elif exp == 1:
                    parts.append(f"{c}·η" if c != 1 else "η")
                elif exp == -1:
                    parts.append(f"{c}·η⁻¹" if c != 1 else "η⁻¹")
                else:
                    parts.append(f"{c}·η^{exp}")
            eta_str = " + ".join(parts) if parts else "0"
        else:
            # Just show raw terms
            eta_str = str(terms)
        
        if qq_p % 2 == 0:
            q_str = f"q^{qq_p//2}" if qq_p != 0 else "q^0"
        else:
            q_str = f"q^({qq_p}/2)"
        lines.append(f"    {q_str}: ({eta_str})")
    
    return "\n".join(lines)

# =====================================================================
# Basis 1: Default (NC = (1,0), meridian)
# =====================================================================
print("=" * 70)
print("BASIS 1: Default SnaPy basis (NC cycle = (1,0) = meridian)")
print("  m = meridian charge, e = longitude/2 charge")
print("=" * 70)

for m in range(0, 4):
    for e_half in range(0, 5):
        e = Fraction(e_half, 2)
        result = compute_refined_index(nz_default, [m], [e], q_order_half=qq_order)
        if result:
            print(f"\nI^ref(m={m}, e={e}; η):")
            print(format_refined(result, nz_default.num_hard))

# =====================================================================
# Basis 2: NC cycle = (-1, 1), new meridian = -M + L
# =====================================================================
print()
print("=" * 70)
print("BASIS 2: NC cycle = (-1, 1), new meridian = -M + L")
print("  m' = (-M+L) charge, e' = conjugate momentum charge")
print("=" * 70)

nz_basis2 = apply_cusp_basis_change(nz_default, cusp_idx=0, P=-1, Q=1)
print(f"  After basis change: num_hard={nz_basis2.num_hard}")

for m in range(0, 4):
    for e_half in range(0, 5):
        e = Fraction(e_half, 2)
        result = compute_refined_index(nz_basis2, [m], [e], q_order_half=qq_order)
        if result:
            print(f"\nI^ref(m'={m}, e'={e}; η):")
            print(format_refined(result, nz_basis2.num_hard))
