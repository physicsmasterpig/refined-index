"""
KEY TEST: Can we just use the ORDINARY kernel K(P,Q) for refined Dehn filling
at ANY slope — not just |Q|=1?

If YES: the ℓ=1 and ℓ=2 results should match, because:
  Basis 1: Σ_{m,e} K(3,1; m,e) · I^ref_basis1(m,e; η)      [slope 3/1]
  Basis 2: Σ_{m',e'} K(3,4; m',e') · I^ref_basis2(m',e'; η) [slope 3/4]
Both use the same ORDINARY kernel, just in different coordinate systems.
"""
import sys
sys.path.insert(0, "src")

from fractions import Fraction
from manifold_index.core.manifold import load_manifold
from manifold_index.core.neumann_zagier import build_neumann_zagier, apply_cusp_basis_change
from manifold_index.core.phase_space import find_easy_edges
from manifold_index.core.refined_index import compute_refined_index
from manifold_index.core.dehn_filling import enumerate_kernel_terms, find_rs

data = load_manifold("m003")
easy = find_easy_edges(data)
nz_default = build_neumann_zagier(data, easy)

qq_order = 10

# =====================================================================
# Basis 1: slope (3, 1), using ordinary kernel K(3, 1)
# =====================================================================
P1, Q1 = 3, 1
R1, S1 = find_rs(P1, Q1)

print(f"Basis 1: P={P1}, Q={Q1}, R={R1}, S={S1}")

terms1 = enumerate_kernel_terms(
    P1, Q1, R1, S1, nz_default, cusp_idx=0,
    m_other=[], e_other=[], q_order_half=qq_order,
)
print(f"  {len(terms1)} kernel terms")

# Sum K(P,Q; m,e) · I^ref(m,e; η) using the ordinary kernel
result1 = {}  # MultiEtaSeries: (qq, 2*η_hard) → Fraction

for kt in terms1:
    m_i, e_i = kt.m, kt.e
    c_val, phase = kt.c, kt.phase
    mult = kt.multiplicity

    # Extra q budget for c=0 terms
    extra_q = abs(phase) if c_val == 0 else 0
    refined = compute_refined_index(nz_default, [m_i], [e_i], q_order_half=qq_order + extra_q)
    if not refined:
        continue

    # Apply kernel factor K(P,Q; m,e) to each refined term
    sign = Fraction(1 if phase % 2 == 0 else -1)
    half = Fraction(1, 2)

    for key, coeff in refined.items():
        if coeff == 0:
            continue
        qq_p = key[0]
        eta_rest = key[1:]  # (2*η_hard,)
        scaled = Fraction(coeff) * half * sign * Fraction(mult)

        if c_val == 0:
            # Term A: +phase shift
            new_qq = qq_p + phase
            if 0 <= new_qq <= qq_order:
                rk = (new_qq,) + eta_rest
                result1[rk] = result1.get(rk, Fraction(0)) + scaled
                if result1[rk] == 0:
                    del result1[rk]
            # Term B: -phase shift
            new_qq = qq_p - phase
            if 0 <= new_qq <= qq_order:
                rk = (new_qq,) + eta_rest
                result1[rk] = result1.get(rk, Fraction(0)) + scaled
                if result1[rk] == 0:
                    del result1[rk]
        else:
            # c = ±2: no q-shift, negative sign
            if 0 <= qq_p <= qq_order:
                rk = (qq_p,) + eta_rest
                result1[rk] = result1.get(rk, Fraction(0)) - scaled
                if result1[rk] == 0:
                    del result1[rk]

print("\nBasis 1 filled refined (P=3,Q=1):")
for key in sorted(result1.keys()):
    print(f"  {key}: {result1[key]}")

# =====================================================================
# Basis 2: slope (3, 4), using ordinary kernel K(3, 4)
# After basis change with NC=(-1, 1)
# =====================================================================
nz_basis2 = apply_cusp_basis_change(nz_default, cusp_idx=0, P=-1, Q=1)

# The physical cycle 3M + L in the new basis becomes slope (3, 4)
# Derivation: M = -M' - L', L = -L'  ⟹  3M + L = -3M' - 4L'  ⟹  (P',Q') = (3, 4)
P2, Q2 = 3, 4
R2, S2 = find_rs(P2, Q2)

print(f"\nBasis 2: P={P2}, Q={Q2}, R={R2}, S={S2}")

terms2 = enumerate_kernel_terms(
    P2, Q2, R2, S2, nz_basis2, cusp_idx=0,
    m_other=[], e_other=[], q_order_half=qq_order,
)
print(f"  {len(terms2)} kernel terms")

result2 = {}

for kt in terms2:
    m_i, e_i = kt.m, kt.e
    c_val, phase = kt.c, kt.phase
    mult = kt.multiplicity

    extra_q = abs(phase) if c_val == 0 else 0
    refined = compute_refined_index(nz_basis2, [m_i], [e_i], q_order_half=qq_order + extra_q)
    if not refined:
        continue

    sign = Fraction(1 if phase % 2 == 0 else -1)
    half = Fraction(1, 2)

    for key, coeff in refined.items():
        if coeff == 0:
            continue
        qq_p = key[0]
        eta_rest = key[1:]
        scaled = Fraction(coeff) * half * sign * Fraction(mult)

        if c_val == 0:
            new_qq = qq_p + phase
            if 0 <= new_qq <= qq_order:
                rk = (new_qq,) + eta_rest
                result2[rk] = result2.get(rk, Fraction(0)) + scaled
                if result2[rk] == 0:
                    del result2[rk]
            new_qq = qq_p - phase
            if 0 <= new_qq <= qq_order:
                rk = (new_qq,) + eta_rest
                result2[rk] = result2.get(rk, Fraction(0)) + scaled
                if result2[rk] == 0:
                    del result2[rk]
        else:
            if 0 <= qq_p <= qq_order:
                rk = (qq_p,) + eta_rest
                result2[rk] = result2.get(rk, Fraction(0)) - scaled
                if result2[rk] == 0:
                    del result2[rk]

print("\nBasis 2 filled refined (P=3,Q=4 in changed basis):")
for key in sorted(result2.keys()):
    print(f"  {key}: {result2[key]}")

# =====================================================================
# Compare
# =====================================================================
print("\n" + "=" * 70)
print("COMPARISON: Do both bases give the same filled refined index?")
print("=" * 70)

all_keys = sorted(set(result1.keys()) | set(result2.keys()))
match = True
for key in all_keys:
    v1 = result1.get(key, Fraction(0))
    v2 = result2.get(key, Fraction(0))
    status = "✓" if v1 == v2 else "✗ MISMATCH"
    if v1 != v2:
        match = False
    print(f"  {key}: basis1={v1}, basis2={v2}  {status}")

print(f"\nOverall: {'MATCH ✓' if match else 'MISMATCH ✗'}")

# Also show η=1 projection
print("\nη=1 projection:")
eta1_1 = {}
for key, c in result1.items():
    eta1_1[key[0]] = eta1_1.get(key[0], Fraction(0)) + c
eta1_2 = {}
for key, c in result2.items():
    eta1_2[key[0]] = eta1_2.get(key[0], Fraction(0)) + c

for qq in sorted(set(eta1_1.keys()) | set(eta1_2.keys())):
    v1 = eta1_1.get(qq, Fraction(0))
    v2 = eta1_2.get(qq, Fraction(0))
    status = "✓" if v1 == v2 else "✗"
    print(f"  qq^{qq}: basis1={v1}, basis2={v2}  {status}")
