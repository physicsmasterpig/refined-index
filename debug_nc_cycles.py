#!/usr/bin/env python3
"""Debug: compare refined Dehn filling for m003's two non-closable cycles."""

from fractions import Fraction
from math import gcd

from manifold_index.core.dehn_filling import (
    _ext_gcd,
    compute_filled_index,
    find_non_closable_cycles,
)
from manifold_index.core.manifold import load_manifold
from manifold_index.core.neumann_zagier import (
    NeumannZagierData,
    apply_cusp_basis_change,
    build_neumann_zagier,
)
from manifold_index.core.phase_space import find_easy_edges
from manifold_index.core.refined_dehn_filling import (
    compute_filled_refined_index,
)
from manifold_index.core.refined_index import compute_refined_index
from manifold_index.core.weyl_check import compute_ab_vectors, run_weyl_checks


def _transform_slope(P_user, Q_user, P_nc, Q_nc):
    g, b, a = _ext_gcd(P_nc, -2 * Q_nc)
    # Normalize to positive gcd
    if g < 0:
        g, b, a = -g, -b, -a
    assert g == 1, f"gcd({P_nc}, {-2*Q_nc}) = {g}"
    P_new = b * P_user - 2 * a * Q_user
    Q_new = -Q_nc * P_user + P_nc * Q_user
    return P_new, Q_new, a, b


def main():
    q_order_half = 10
    name = "m003"

    print(f"=== {name} ===")
    data = load_manifold(name)
    easy = find_easy_edges(data)
    nz = build_neumann_zagier(data, easy)
    print(f"  n={nz.n}, r={nz.r}, num_hard={nz.num_hard}")

    # Step 1: Find non-closable cycles
    print("\n--- Non-closable cycle search ---")
    nc_result = find_non_closable_cycles(
        nz, cusp_idx=0,
        p_range=range(-5, 6),
        q_range=range(0, 4),
        q_order_half=q_order_half,
        verbose=True,
    )
    nc_cycles = [(nc.P, nc.Q) for nc in nc_result.cycles]
    print(f"\nNon-closable cycles: {nc_cycles}")

    if len(nc_cycles) < 2:
        print("Less than 2 non-closable cycles found. Expand search range?")
        return

    # De-duplicate: keep only one from each ±(P,Q) pair
    unique_nc = []
    seen = set()
    for P, Q in nc_cycles:
        if (-P, -Q) not in seen:
            unique_nc.append((P, Q))
            seen.add((P, Q))
    print(f"Unique (up to sign): {unique_nc}")

    # Step 2: For a test user slope, compute refined filling for each NC cycle
    test_slopes = [(1, 0), (3, 1), (5, 1)]

    for P_user, Q_user in test_slopes:
        print(f"\n{'='*60}")
        print(f"User slope: ({P_user}, {Q_user})")
        print(f"{'='*60}")

        for P_nc, Q_nc in unique_nc:
            if P_nc % 2 == 0:
                print(f"  Skipping ({P_nc}, {Q_nc}): P_nc even")
                continue

            P_new, Q_new, a_coeff, b_coeff = _transform_slope(
                P_user, Q_user, P_nc, Q_nc
            )
            print(f"\n  NC cycle ({P_nc}, {Q_nc}): "
                  f"Bézout a={a_coeff}, b={b_coeff}")
            print(f"    Transformed slope: ({P_new}, {Q_new})")

            nz_changed = apply_cusp_basis_change(nz, 0, P_nc, Q_nc)

            filled = compute_filled_refined_index(
                nz_changed,
                cusp_idx=0,
                P=P_new,
                Q=Q_new,
                q_order_half=q_order_half,
                eta_order=5,
                verbose=False,
            )
            print(f"    HJ-CF: {filled.hj_ks}")
            print(f"    #terms: {filled.n_kernel_terms}")
            print(f"    has_cusp_eta: {filled.has_cusp_eta}")
            print(f"    #entries: {len(filled.series)}")

            # Show eta=1 series
            eta1 = filled.eta1_series()
            print(f"    eta=1 series (q^(1/2) powers):")
            for k in sorted(eta1.keys())[:15]:
                print(f"      q^({k}/2): {eta1[k]}")

            # Show the full series (first few terms)
            print(f"    Full series (first 10 entries):")
            for i, (key, c) in enumerate(sorted(filled.series.items())):
                if i >= 10:
                    print(f"      ... ({len(filled.series)} total)")
                    break
                print(f"      {key}: {c}")

    # Step 3: Also check Weyl parameters
    print(f"\n{'='*60}")
    print(f"Weyl parameter check")
    print(f"{'='*60}")

    # Build evaluation grid
    entries = []
    for m in range(-2, 3):
        for e_half in range(-2, 3):
            e = Fraction(e_half, 2)
            m_ext = [m]
            e_ext = [e]
            result = compute_refined_index(nz, m_ext, e_ext, q_order_half=q_order_half)
            if result:
                entries.append((m_ext, e_ext, result))

    print(f"  {len(entries)} non-zero grid points")

    ab = compute_ab_vectors(entries, nz.num_hard)
    if ab is None:
        print("  Could not determine Weyl parameters!")
    else:
        print(f"  a = {[str(v) for v in ab.a]}")
        print(f"  b = {[str(v) for v in ab.b]}")
        print(f"  valid = {ab.is_valid}")

    # Step 4: Compare WITH Weyl shift
    if ab is not None and ab.is_valid:
        print(f"\n{'='*60}")
        print(f"With Weyl shift η^{{b·m + a·e}}")
        print(f"{'='*60}")

        for P_user, Q_user in test_slopes:
            print(f"\nUser slope: ({P_user}, {Q_user})")

            for P_nc, Q_nc in unique_nc:
                if P_nc % 2 == 0:
                    continue

                P_new, Q_new, a_coeff, b_coeff = _transform_slope(
                    P_user, Q_user, P_nc, Q_nc
                )
                nz_changed = apply_cusp_basis_change(nz, 0, P_nc, Q_nc)

                filled_w = compute_filled_refined_index(
                    nz_changed,
                    cusp_idx=0,
                    P=P_new,
                    Q=Q_new,
                    q_order_half=q_order_half,
                    eta_order=5,
                    weyl_a=ab.a,
                    weyl_b=ab.b,
                    verbose=False,
                )
                eta1_w = filled_w.eta1_series()
                print(f"  NC ({P_nc}, {Q_nc}) → ({P_new}, {Q_new}): "
                      f"#entries={len(filled_w.series)}")
                print(f"    eta=1 (Weyl-shifted):")
                for k in sorted(eta1_w.keys())[:10]:
                    print(f"      q^({k}/2): {eta1_w[k]}")


if __name__ == "__main__":
    main()
