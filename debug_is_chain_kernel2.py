"""
Debug: IS chain kernel test with proper qq_order >> eta_order.

Focus: Test whether the IS chain at η=1 reproduces K(3,2) when we use
qq_order = 40 and eta_order = 10 (stable region ≈ qq ≤ 20).
"""

from fractions import Fraction
from manifold_index.core.refined_dehn_filling import (
    _is_kernel,
    _etilde_is,
    _enumerate_slope1_all,
    _particular_solution,
    hj_continued_fraction,
)
from manifold_index.core.dehn_filling import find_rs


def ordinary_kernel_value(P, Q, m, e, qq_order=40):
    """K(P,Q; m,e) as qq-series."""
    R, S = find_rs(P, Q)
    c_frac = Fraction(P * m) + 2 * Fraction(Q) * Fraction(e)
    if c_frac.denominator != 1:
        return {}
    c_val = int(c_frac)
    phase = int(Fraction(R * m) + 2 * Fraction(S) * Fraction(e))
    sign = Fraction(1 if phase % 2 == 0 else -1)
    half = Fraction(1, 2)

    result = {}
    if c_val == 0:
        factor = half * sign
        for shift in (phase, -phase):
            if 0 <= shift <= qq_order:
                result[shift] = result.get(shift, Fraction(0)) + factor
    elif abs(c_val) == 2:
        factor = -half * sign
        result[0] = result.get(0, Fraction(0)) + factor
    return {k: v for k, v in result.items() if v != 0}


def is_chain_kernel(P, Q, m, e, qq_order=40, eta_order=10):
    """IS chain at η=1 for ℓ=2."""
    ks = hj_continued_fraction(P, Q)
    assert len(ks) == 2
    k1, k2 = ks
    e_in = -Fraction(e) - Fraction(k1 * m, 2)
    k2_terms = _enumerate_slope1_all(k2, qq_order)

    result = {}
    n_terms = 0
    for m1, e1, c_final, phase_final in k2_terms:
        is_val = _is_kernel(m, e_in, m1, e1, qq_order, eta_order)
        if not is_val:
            continue
        n_terms += 1

        # η=1 projection
        is_eta1 = {}
        for (qq_p, eta_exp), coeff in is_val.items():
            is_eta1[qq_p] = is_eta1.get(qq_p, Fraction(0)) + coeff
        is_eta1 = {k: v for k, v in is_eta1.items() if v != 0}
        if not is_eta1:
            continue

        # K(k2,1) factor
        sign_final = Fraction(1 if phase_final % 2 == 0 else -1)
        half = Fraction(1, 2)
        k_factor = {}
        if c_final == 0:
            factor = half * sign_final
            for shift in (phase_final, -phase_final):
                if 0 <= shift <= qq_order:
                    k_factor[shift] = k_factor.get(shift, Fraction(0)) + factor
        elif abs(c_final) == 2:
            factor = -half * sign_final
            k_factor[0] = k_factor.get(0, Fraction(0)) + factor
        k_factor = {k: v for k, v in k_factor.items() if v != 0}
        if not k_factor:
            continue

        for p1, c1 in is_eta1.items():
            for p2, c2 in k_factor.items():
                pp = p1 + p2
                if pp < 0 or pp > qq_order:
                    continue
                result[pp] = result.get(pp, Fraction(0)) + c1 * c2

    return {k: v for k, v in result.items() if v != 0}, n_terms


# =====================================================================
P, Q = 3, 2
qq_order = 40
eta_order = 10
stable_cutoff = qq_order - 2 * eta_order  # = 20

print(f"P/Q = {P}/{Q}, HJ-CF = {hj_continued_fraction(P, Q)}")
print(f"qq_order = {qq_order}, eta_order = {eta_order}")
print(f"Stable region: qq ≤ {stable_cutoff}")
print()

# Pick test points
test_points = []
for m in range(-6, 7):
    for e_half in range(-12, 13):
        e = Fraction(e_half, 2)
        kv = ordinary_kernel_value(P, Q, m, e, qq_order)
        if kv:
            test_points.append((m, e))

print(f"Total non-zero K(3,2; m,e) points: {len(test_points)}")
print()

# Compare
n_match = 0
n_mismatch = 0

for m, e in test_points:
    kv_ord = ordinary_kernel_value(P, Q, m, e, qq_order)
    kv_chain, n_terms = is_chain_kernel(P, Q, m, e, qq_order, eta_order)

    # Compare only in stable region
    all_qq = set(list(kv_ord.keys()) + list(kv_chain.keys()))
    all_qq = [p for p in all_qq if p <= stable_cutoff]

    mismatches = []
    for qq_p in sorted(all_qq):
        v_ord = kv_ord.get(qq_p, Fraction(0))
        v_chain = kv_chain.get(qq_p, Fraction(0))
        if v_ord != v_chain:
            mismatches.append(f"qq^{qq_p}: chain={v_chain} vs ord={v_ord}")

    if mismatches:
        n_mismatch += 1
        print(f"MISMATCH (m={m}, e={e}) [n_IS_terms={n_terms}]:")
        print(f"  Ordinary: {dict(sorted(kv_ord.items()))}")
        for mm in mismatches:
            print(f"  {mm}")
    else:
        n_match += 1

print(f"\nSummary: {n_match} MATCH, {n_mismatch} MISMATCH (in stable region qq ≤ {stable_cutoff})")

if n_mismatch > 0:
    print("\n*** The IS chain at η=1 does NOT reproduce the ordinary kernel ***")
    print("*** This indicates a formula or structural issue ***")
else:
    print("\n*** IS chain at η=1 MATCHES the ordinary kernel in stable region ***")
    print("*** The earlier mismatches were truncation artifacts ***")
