#!/usr/bin/env python3
"""Targeted diagnostic: compare unrefined and refined filling for m003
with different NC cycle choices."""

from fractions import Fraction

from manifold_index.core.dehn_filling import (
    _ext_gcd,
    compute_filled_index,
)
from manifold_index.core.manifold import load_manifold
from manifold_index.core.neumann_zagier import (
    apply_cusp_basis_change,
    build_neumann_zagier,
)
from manifold_index.core.phase_space import find_easy_edges
from manifold_index.core.refined_dehn_filling import (
    compute_filled_refined_index,
    hj_continued_fraction,
)


def _transform_slope(P_user, Q_user, P_nc, Q_nc):
    g, b, a = _ext_gcd(P_nc, -2 * Q_nc)
    if g < 0:
        g, b, a = -g, -b, -a
    assert g == 1
    P_new = b * P_user - 2 * a * Q_user
    Q_new = -Q_nc * P_user + P_nc * Q_user
    return P_new, Q_new, a, b


def main():
    q_order_half = 10
    data = load_manifold("m003")
    easy = find_easy_edges(data)
    nz = build_neumann_zagier(data, easy)

    nc_cycles = [(-1, 1), (1, 0)]
    user_slopes = [(3, 1), (5, 1)]

    print("=" * 70)
    print("DIAGNOSTIC 1: Unrefined filling — should be independent of NC cycle")
    print("=" * 70)

    for P_user, Q_user in user_slopes:
        print(f"\n--- User slope ({P_user}, {Q_user}) ---")

        # Direct unrefined filling (no basis change)
        direct = compute_filled_index(
            nz, cusp_idx=0, P=P_user, Q=Q_user, q_order_half=q_order_half
        )
        print(f"  Direct I_{{{P_user}/{Q_user}}} (unrefined):")
        for k in sorted(direct.series.keys())[:8]:
            print(f"    q^({k}/2): {direct.series[k]}")

        for P_nc, Q_nc in nc_cycles:
            if P_nc % 2 == 0:
                continue
            P_new, Q_new, a, b = _transform_slope(P_user, Q_user, P_nc, Q_nc)
            nz_changed = apply_cusp_basis_change(nz, 0, P_nc, Q_nc)

            filled = compute_filled_index(
                nz_changed, cusp_idx=0, P=P_new, Q=Q_new,
                q_order_half=q_order_half
            )
            print(f"  NC ({P_nc},{Q_nc}) → I_{{{P_new}/{Q_new}}} (unrefined):")
            for k in sorted(filled.series.keys())[:8]:
                print(f"    q^({k}/2): {filled.series[k]}")

    print()
    print("=" * 70)
    print("DIAGNOSTIC 2: Refined filling (eta=1) — should agree at eta=1")
    print("=" * 70)

    for P_user, Q_user in user_slopes:
        print(f"\n--- User slope ({P_user}, {Q_user}) ---")

        for P_nc, Q_nc in nc_cycles:
            if P_nc % 2 == 0:
                continue
            P_new, Q_new, a, b = _transform_slope(P_user, Q_user, P_nc, Q_nc)
            hj_ks = hj_continued_fraction(P_new, Q_new)
            ell = len(hj_ks)
            nz_changed = apply_cusp_basis_change(nz, 0, P_nc, Q_nc)

            filled = compute_filled_refined_index(
                nz_changed, cusp_idx=0, P=P_new, Q=Q_new,
                q_order_half=q_order_half, eta_order=5,
            )
            eta1 = filled.eta1_series()
            print(f"  NC ({P_nc},{Q_nc}) → ({P_new},{Q_new}), "
                  f"HJ-CF={hj_ks}, ℓ={ell}, "
                  f"#entries={len(filled.series)}")
            print(f"    eta=1 series:")
            for k in sorted(eta1.keys())[:8]:
                print(f"      q^({k}/2): {eta1[k]}")

    print()
    print("=" * 70)
    print("DIAGNOSTIC 3: Direct sanity check — I^ref for ℓ=2 slope (1/0)")
    print("            This should be zero (it's the meridian = non-closable)")
    print("=" * 70)

    # Fill slope (1,0) with NO basis change — uses ℓ=2 path since |Q|≠1
    # Actually HJ-CF(1,0) = [0,0] → ℓ=2
    hj_ks = hj_continued_fraction(1, 0)
    print(f"\n  HJ-CF(1, 0) = {hj_ks}")
    filled_10 = compute_filled_refined_index(
        nz, cusp_idx=0, P=1, Q=0,
        q_order_half=q_order_half, eta_order=5,
    )
    eta1_10 = filled_10.eta1_series()
    print(f"  I^ref_{{1/0}} eta=1 series (should be ~0):")
    for k in sorted(eta1_10.keys())[:8]:
        print(f"    q^({k}/2): {eta1_10[k]}")
    if not eta1_10:
        print(f"    (empty — good!)")

    # Also check unrefined I_{1/0}
    direct_10 = compute_filled_index(
        nz, cusp_idx=0, P=1, Q=0, q_order_half=q_order_half,
    )
    print(f"\n  I_{{1/0}} unrefined series (should be ~0):")
    for k in sorted(direct_10.series.keys())[:8]:
        print(f"    q^({k}/2): {direct_10.series[k]}")
    if not direct_10.series:
        print(f"    (empty — good, confirming non-closable)")

    print()
    print("=" * 70)
    print("DIAGNOSTIC 4: Fill slope (-1/1) directly with ℓ=1 (no basis change)")
    print("              This is another non-closable cycle")
    print("=" * 70)
    hj_ks_m11 = hj_continued_fraction(-1, 1)
    print(f"\n  HJ-CF(-1, 1) = {hj_ks_m11}")
    filled_m11 = compute_filled_refined_index(
        nz, cusp_idx=0, P=-1, Q=1,
        q_order_half=q_order_half, eta_order=5,
    )
    eta1_m11 = filled_m11.eta1_series()
    print(f"  I^ref_{{-1/1}} eta=1 series (should be ~0):")
    for k in sorted(eta1_m11.keys())[:8]:
        print(f"    q^({k}/2): {eta1_m11[k]}")
    if not eta1_m11:
        print(f"    (empty — good, confirming non-closable)")

    # Full series for inspection
    print(f"  I^ref_{{-1/1}} full series ({len(filled_m11.series)} entries):")
    for i, (key, c) in enumerate(sorted(filled_m11.series.items())):
        if i >= 15:
            break
        print(f"    {key}: {c}")


if __name__ == "__main__":
    main()
