"""
Test: does the refined Dehn filling give the same answer for different
HJ-CF representations of the same slope?

Standard HJ-CF (k_i ≥ 2 except terminal) vs shorter alternatives
with unrestricted k_i values.

Examples:
  1/3:  standard [1,2,2] (ℓ=3)  vs  [0,-3] (ℓ=2)
  2/5:  standard [1,3,2] (ℓ=3)  vs  [1,-5] ... hmm, 1 - 1/(-5) = 1+1/5 = 6/5 ≠ 2/5
        actually: need k1 - 1/k2 = 2/5 → k2 such that k1 - 1/k2 = 2/5
        if k1=0: -1/k2 = 2/5 → k2 = -5/2 (not integer)
        if k1=1: 1 - 1/k2 = 2/5 → 1/k2 = 3/5 → k2 = 5/3 (not integer)
        So 2/5 has no length-2 representation.  But could have shorter than [1,3,2].
        Actually [0, -3, 2]: 0 - 1/(-3 - 1/2) = 0 - 1/(-7/2) = 2/7 ≠ 2/5.
        Let's just stick to cases that work.

  1/3:  [1,2,2] vs [0,-3]
  3/5:  standard [2,2,2,2] (ℓ=4) vs ...
        Actually: [1,2,3] → 1 - 1/(2 - 1/3) = 1 - 1/(5/3) = 1 - 3/5 = 2/5. No.

  Let's compute alternatives programmatically.
"""

import sys
import time
from fractions import Fraction
from unittest.mock import patch

sys.path.insert(0, "src")

from manifold_index.core.manifold import load_manifold
from manifold_index.core.phase_space import find_easy_edges
from manifold_index.core.neumann_zagier import build_neumann_zagier
from manifold_index.core.refined_dehn_filling import (
    hj_continued_fraction,
    compute_filled_refined_index,
)


def load_nz(name: str):
    data = load_manifold(name)
    easy = find_easy_edges(data)
    return build_neumann_zagier(data, easy)


def recover(ks: list[int]) -> Fraction:
    """Recover P/Q from HJ-CF coefficients."""
    x = Fraction(ks[-1])
    for k in reversed(ks[:-1]):
        x = k - Fraction(1, x)
    return x


def find_short_hjcf(P: int, Q: int, max_len: int = 3) -> list[list[int]]:
    """Find all HJ-CF representations of P/Q with length ≤ max_len.

    Brute-force search over k_i ∈ [-10, 10].
    """
    target = Fraction(P, Q)
    results = []

    # Length 1
    if target.denominator == 1:
        results.append([int(target)])

    # Length 2
    for k1 in range(-10, 11):
        for k2 in range(-10, 11):
            if k2 == 0:
                continue
            val = Fraction(k1) - Fraction(1, k2)
            if val == target:
                results.append([k1, k2])

    # Length 3
    if max_len >= 3:
        for k1 in range(-10, 11):
            for k2 in range(-10, 11):
                for k3 in range(-10, 11):
                    if k3 == 0:
                        continue
                    inner = Fraction(k2) - Fraction(1, k3)
                    if inner == 0:
                        continue
                    val = Fraction(k1) - Fraction(1, inner)
                    if val == target:
                        results.append([k1, k2, k3])

    return results


def test_slope(manifold_name: str, P: int, Q: int, qq: int = 6):
    """Test all short HJ-CF representations give the same eta1_series."""
    nz = load_nz(manifold_name)

    standard_ks = hj_continued_fraction(P, Q)
    print(f"\n{'='*60}")
    print(f"Manifold: {manifold_name}, slope {P}/{Q}")
    print(f"Standard HJ-CF: {standard_ks} (ℓ={len(standard_ks)})")

    # Find alternatives
    alternatives = find_short_hjcf(P, Q)
    # Filter out the standard one and sort by length
    alternatives = [a for a in alternatives if a != standard_ks]
    alternatives.sort(key=len)

    print(f"Alternative representations: {len(alternatives)}")
    for alt in alternatives:
        print(f"  {alt} (ℓ={len(alt)}) → {recover(alt)}")

    if not alternatives:
        print("No alternatives found, skipping.")
        return True

    # Pick the shortest alternative
    best_alt = min(alternatives, key=len)
    print(f"\nComparing standard {standard_ks} vs shortest {best_alt}")

    # Compute with standard HJ-CF
    t0 = time.time()
    res_std = compute_filled_refined_index(
        nz, 0, P, Q, q_order_half=qq,
        eta_order=qq, auto_precompute=False, verbose=False,
    )
    t_std = time.time() - t0

    # Compute with alternative HJ-CF by monkey-patching
    t0 = time.time()
    with patch(
        "manifold_index.core.refined_dehn_filling.hj_continued_fraction",
        return_value=best_alt,
    ):
        res_alt = compute_filled_refined_index(
            nz, 0, P, Q, q_order_half=qq,
            eta_order=qq, auto_precompute=False, verbose=False,
        )
    t_alt = time.time() - t0

    # Compare eta1_series (η → 1)
    eta1_std = res_std.eta1_series()
    eta1_alt = res_alt.eta1_series()

    print(f"Standard (ℓ={len(standard_ks)}): {t_std:.2f}s, {len(res_std.series)} terms")
    print(f"Alternative (ℓ={len(best_alt)}): {t_alt:.2f}s, {len(res_alt.series)} terms")
    print(f"Speedup: {t_std/t_alt:.1f}×" if t_alt > 0 else "")

    # Compare eta1 (should match for correct computation)
    match_eta1 = (eta1_std == eta1_alt)
    print(f"eta1_series match: {'PASS' if match_eta1 else 'FAIL'}")

    if not match_eta1:
        # Show diffs
        all_keys = sorted(set(eta1_std) | set(eta1_alt))
        for k in all_keys:
            v1 = eta1_std.get(k, Fraction(0))
            v2 = eta1_alt.get(k, Fraction(0))
            if v1 != v2:
                print(f"  qq={k}: std={v1}, alt={v2}, diff={v1-v2}")

    # Also compare full series if both have cusp_eta
    if res_std.has_cusp_eta and res_alt.has_cusp_eta:
        match_full = (res_std.series == res_alt.series)
        print(f"Full series match: {'PASS' if match_full else 'FAIL'}")
        if not match_full:
            all_keys = sorted(set(res_std.series) | set(res_alt.series))
            n_diff = 0
            for k in all_keys:
                v1 = res_std.series.get(k, Fraction(0))
                v2 = res_alt.series.get(k, Fraction(0))
                if v1 != v2:
                    n_diff += 1
                    if n_diff <= 10:
                        print(f"  {k}: std={v1}, alt={v2}")
            if n_diff > 10:
                print(f"  ... ({n_diff} total diffs)")

    return match_eta1


if __name__ == "__main__":
    QQ = 6

    # First, show available alternatives for common slopes
    print("Surveying HJ-CF alternatives:")
    test_slopes = [
        (1, 3),   # standard [1,2,2] ℓ=3
        (1, 4),   # standard ?
        (2, 3),   # standard ?
        (3, 5),   # standard ?
        (4, 3),   # standard ?
        # (1, 0),   # longitude [0,0] ℓ=2 — skip (Q=0)
    ]

    for P, Q in test_slopes:
        std = hj_continued_fraction(P, Q)
        alts = find_short_hjcf(P, Q)
        shorter = [a for a in alts if len(a) < len(std)]
        print(f"  {P}/{Q}: std={std} (ℓ={len(std)}), "
              f"shorter alternatives: {shorter if shorter else 'none'}")

    # Now test with m003
    print("\n" + "="*60)
    print(f"Testing with m003, QQ={QQ}")
    print("="*60)

    all_pass = True
    for P, Q in test_slopes:
        if Q == 0:
            continue  # skip longitude for now
        try:
            ok = test_slope("m003", P, Q, qq=QQ)
            all_pass = all_pass and ok
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            all_pass = False

    print(f"\n{'='*60}")
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAIL'}")
