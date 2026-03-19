"""
Debug: Verify IS chain reproduces ordinary K(P,Q) kernel at η=1.

For m003, slope 3/2:
  HJ-CF = [2, 2], ℓ = 2
  K^ref(3,2; m,e; η) = Σ_{m1,e1} I_S(m, -e-m, m1, e1; η) · K(2,1; m1,e1)

At η=1, this should reproduce K(3,2; m,e) from the ordinary kernel.

Strategy:
  1. Pick a specific (m, e) pair with non-zero K(3,2; m, e).
  2. Compute K(3,2; m,e) directly via enumerate_kernel_terms.
  3. Compute the IS chain: Σ_{m1,e1} I_S(m, -e-m, m1, e1; η=1) · K(2,1; m1,e1)
  4. Compare.

This isolates the IS formula from any payload issues (no 3D index involved).
"""

from fractions import Fraction
from manifold_index.core.refined_dehn_filling import (
    _is_kernel,
    _etilde_is,
    _enumerate_slope1_all,
    _particular_solution,
    hj_continued_fraction,
)
from manifold_index.core.dehn_filling import (
    enumerate_kernel_terms,
    find_rs,
)


def ordinary_kernel_value(P, Q, m, e, qq_order=20):
    """Compute K(P,Q; m, e) as a qq-series dict[int, Fraction].

    K(P,Q; m,e) = (1/2)·(-1)^{Rm+2Se} · [
        δ_{Pm+2Qe,0}·(qq^{Rm+2Se} + qq^{-(Rm+2Se)})
        - δ_{Pm+2Qe,2}
        - δ_{Pm+2Qe,-2}
    ]
    """
    R, S = find_rs(P, Q)
    c_val = P * m + 2 * Q * int(e)  # Pm + 2Qe  (e might be Fraction)
    if isinstance(e, Fraction):
        c_val = P * m + 2 * Q * e
        c_val = int(c_val) if c_val.denominator == 1 else None
        if c_val is None:
            return {}

    phase = R * m + 2 * S * int(e) if isinstance(e, int) else int(R * m + 2 * S * e)
    sign = Fraction(1 if phase % 2 == 0 else -1)
    half = Fraction(1, 2)

    result = {}
    if c_val == 0:
        # (1/2)·(-1)^phase·(qq^phase + qq^{-phase})
        factor = half * sign
        if 0 <= phase <= qq_order:
            result[phase] = result.get(phase, Fraction(0)) + factor
        if 0 <= -phase <= qq_order:
            result[-phase] = result.get(-phase, Fraction(0)) + factor
    elif abs(c_val) == 2:
        # -(1/2)·(-1)^phase
        factor = -half * sign
        if 0 <= 0 <= qq_order:
            result[0] = result.get(0, Fraction(0)) + factor
    # else: c_val not in {0, ±2}, kernel is 0

    # Clean zeros
    return {k: v for k, v in result.items() if v != 0}


def is_chain_kernel_value(P, Q, m, e, qq_order=20, eta_order=15):
    """Compute K^ref(P,Q; m,e; η=1) via the IS chain.

    For HJ-CF = [k1, k2, ..., kℓ]:
    K^ref = Σ_{m1,e1} I_S(m, -e-k1/2·m, m1, e1; η) · K(k2,1; m1,e1)
    Then set η=1.

    For ℓ=2 only (simplification).
    """
    ks = hj_continued_fraction(P, Q)
    assert len(ks) == 2, f"This function only handles ℓ=2, got ks={ks}"
    k1, k2 = ks

    # e-transform
    e_in = -e - Fraction(k1 * m, 2)

    # Enumerate K(k2, 1; m1, e1) support
    k2_terms = _enumerate_slope1_all(k2, qq_order)

    result = {}  # qq_power -> Fraction

    for m1, e1, c_final, phase_final in k2_terms:
        # Compute I_S(m, e_in, m1, e1; η)
        is_val = _is_kernel(m, e_in, m1, e1, qq_order, eta_order)
        if not is_val:
            continue

        # Set η=1: sum over eta dimension
        is_eta1 = {}  # qq_power -> Fraction
        for (qq_p, eta_exp), coeff in is_val.items():
            is_eta1[qq_p] = is_eta1.get(qq_p, Fraction(0)) + coeff

        # Remove zeros
        is_eta1 = {k: v for k, v in is_eta1.items() if v != 0}
        if not is_eta1:
            continue

        # Compute K(k2, 1; m1, e1) factor (qq-series)
        sign_final = Fraction(1 if phase_final % 2 == 0 else -1)
        half = Fraction(1, 2)

        k_factor = {}  # qq_power -> Fraction
        if c_final == 0:
            factor = half * sign_final
            if abs(phase_final) <= qq_order:
                k_factor[phase_final] = k_factor.get(phase_final, Fraction(0)) + factor
                k_factor[-phase_final] = k_factor.get(-phase_final, Fraction(0)) + factor
        elif abs(c_final) == 2:
            factor = -half * sign_final
            k_factor[0] = k_factor.get(0, Fraction(0)) + factor

        k_factor = {k: v for k, v in k_factor.items() if v != 0}
        if not k_factor:
            continue

        # Convolve is_eta1 · k_factor
        for p1, c1 in is_eta1.items():
            for p2, c2 in k_factor.items():
                pp = p1 + p2
                if pp < 0 or pp > qq_order:
                    continue
                result[pp] = result.get(pp, Fraction(0)) + c1 * c2

    return {k: v for k, v in result.items() if v != 0}


# =====================================================================
# Main comparison
# =====================================================================
print("=" * 70)
print("Test: IS chain at η=1 vs ordinary K(P,Q)")
print("Slope 3/2, HJ-CF = [2, 2]")
print("=" * 70)

P, Q = 3, 2
ks = hj_continued_fraction(P, Q)
print(f"HJ-CF({P}/{Q}) = {ks}")

R, S = find_rs(P, Q)
print(f"R={R}, S={S} (R·Q - P·S = {R*Q - P*S})")

# Find some (m, e) pairs where K(3,2; m,e) ≠ 0
print("\n--- Ordinary K(3,2; m, e) for various (m, e) ---")
test_points = []
for m in range(-6, 7):
    for e_half in range(-12, 13):
        e = Fraction(e_half, 2)
        kv = ordinary_kernel_value(P, Q, m, e, qq_order=10)
        if kv:
            test_points.append((m, e))
            if len(test_points) <= 10:
                print(f"  K(3,2; {m}, {e}) = {dict(sorted(kv.items()))}")

print(f"\nTotal non-zero (m,e) pairs: {len(test_points)}")

# Now compare IS chain vs ordinary for each test point
print("\n--- IS chain comparison ---")
eta_orders = [5, 10, 15, 20]
mismatches = 0

for m, e in test_points[:10]:  # test first 10
    kv_ordinary = ordinary_kernel_value(P, Q, m, e, qq_order=10)

    print(f"\n(m={m}, e={e})")
    print(f"  Ordinary K(3,2): {dict(sorted(kv_ordinary.items()))}")

    for eta_ord in eta_orders:
        kv_chain = is_chain_kernel_value(P, Q, m, e, qq_order=10, eta_order=eta_ord)
        # Compare low qq powers
        match_str = ""
        for qq_p in sorted(set(list(kv_ordinary.keys()) + list(kv_chain.keys()))):
            if qq_p > 6:
                continue
            v_ord = kv_ordinary.get(qq_p, Fraction(0))
            v_chain = kv_chain.get(qq_p, Fraction(0))
            if v_ord != v_chain:
                match_str += f" qq^{qq_p}:{v_chain}≠{v_ord}"
        if match_str:
            print(f"  IS chain (η_ord={eta_ord}): MISMATCH{match_str}")
            mismatches += 1
        else:
            stable_qq = max(k for k in kv_ordinary.keys()) if kv_ordinary else 0
            print(f"  IS chain (η_ord={eta_ord}): MATCH up to qq^6")

print(f"\n{'='*70}")
if mismatches == 0:
    print("ALL MATCH — IS kernel formula is correct")
else:
    print(f"{mismatches} MISMATCHES — IS kernel formula has issues")
    print("\nNow testing with MODIFIED formula (no 1/2 on last two terms)...")


# =====================================================================
# Test variant: I_S without the (1/2)·(-1)^m1 factor on last two terms
# =====================================================================
print("\n" + "=" * 70)
print("Variant B: I_S with factor_cd = -1 (no half, no sign)")
print("=" * 70)


def _is_kernel_variant_B(m1, e1, m2, e2, qq_order, eta_order):
    """IS kernel with terms C,D having factor = -1 (not -half·sign_m1)."""
    ei_center = _etilde_is(m1, e1,     m2, e2, qq_order, eta_order)
    ei_minus  = _etilde_is(m1, e1 - 1, m2, e2, qq_order, eta_order)
    ei_plus   = _etilde_is(m1, e1 + 1, m2, e2, qq_order, eta_order)

    sign_m1 = Fraction(1 if m1 % 2 == 0 else -1)
    half = Fraction(1, 2)

    result = {}

    # Term A,B: (1/2)·(-1)^{m1}·(qq^{m1} + qq^{-m1}) · ẽI_S(e1)
    factor_ab = half * sign_m1
    for (qq_p, eta), c in ei_center.items():
        scaled = c * factor_ab
        if scaled == 0:
            continue
        for shift in (m1, -m1):
            new_qq = qq_p + shift
            if 0 <= new_qq <= qq_order:
                key = (new_qq, eta)
                v = result.get(key, Fraction(0)) + scaled
                if v == 0:
                    result.pop(key, None)
                else:
                    result[key] = v

    # Terms C,D: -ẽI_S(e1±1)  (NO half, NO sign_m1)
    factor_cd = Fraction(-1)
    for src in (ei_minus, ei_plus):
        for (qq_p, eta), c in src.items():
            scaled = c * factor_cd
            if scaled == 0:
                continue
            if not (0 <= qq_p <= qq_order):
                continue
            key = (qq_p, eta)
            v = result.get(key, Fraction(0)) + scaled
            if v == 0:
                result.pop(key, None)
            else:
                result[key] = v

    return result


def is_chain_variant_B(P, Q, m, e, qq_order=20, eta_order=15):
    """IS chain using variant B kernel."""
    ks = hj_continued_fraction(P, Q)
    assert len(ks) == 2
    k1, k2 = ks
    e_in = -e - Fraction(k1 * m, 2)
    k2_terms = _enumerate_slope1_all(k2, qq_order)

    result = {}
    for m1, e1, c_final, phase_final in k2_terms:
        is_val = _is_kernel_variant_B(m, e_in, m1, e1, qq_order, eta_order)
        if not is_val:
            continue

        is_eta1 = {}
        for (qq_p, eta_exp), coeff in is_val.items():
            is_eta1[qq_p] = is_eta1.get(qq_p, Fraction(0)) + coeff
        is_eta1 = {k: v for k, v in is_eta1.items() if v != 0}
        if not is_eta1:
            continue

        sign_final = Fraction(1 if phase_final % 2 == 0 else -1)
        half = Fraction(1, 2)
        k_factor = {}
        if c_final == 0:
            factor = half * sign_final
            if abs(phase_final) <= qq_order:
                k_factor[phase_final] = k_factor.get(phase_final, Fraction(0)) + factor
                k_factor[-phase_final] = k_factor.get(-phase_final, Fraction(0)) + factor
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

    return {k: v for k, v in result.items() if v != 0}


for m, e in test_points[:10]:
    kv_ordinary = ordinary_kernel_value(P, Q, m, e, qq_order=10)
    kv_B = is_chain_variant_B(P, Q, m, e, qq_order=10, eta_order=15)

    match_str = ""
    for qq_p in sorted(set(list(kv_ordinary.keys()) + list(kv_B.keys()))):
        if qq_p > 6:
            continue
        v_ord = kv_ordinary.get(qq_p, Fraction(0))
        v_B = kv_B.get(qq_p, Fraction(0))
        if v_ord != v_B:
            match_str += f" qq^{qq_p}:{v_B}≠{v_ord}"
    status = "MISMATCH" + match_str if match_str else "MATCH up to qq^6"
    print(f"  (m={m}, e={e}): {status}")
