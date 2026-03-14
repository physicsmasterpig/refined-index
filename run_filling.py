#!/usr/bin/env python3
"""Quick script to compute I_{P/Q}(4_1) for various slopes."""
import sys
import traceback
from fractions import Fraction

from manifold_index.core.manifold import load_manifold
from manifold_index.core.phase_space import find_easy_edges
from manifold_index.core.neumann_zagier import build_neumann_zagier
from manifold_index.core.dehn_filling import compute_filled_index

def run(P, Q, q_order_half=10, cache_dir=None):
    print(f"\n=== I_{{{P}/{Q}}}(4_1), q_order_half={q_order_half} ===")
    try:
        data = load_manifold('4_1')
        easy = find_easy_edges(data)
        nz = build_neumann_zagier(data, easy)
        result = compute_filled_index(nz, cusp_idx=0, P=P, Q=Q,
                                      q_order_half=q_order_half,
                                      cache_dir=cache_dir)
        print(f"n_kernel_terms = {result.n_kernel_terms}")
        print(f"is_zero = {result.is_zero}")
        series = dict(sorted(result.series.items()))
        print(f"series = {series}")
        # Pretty-print as q-series
        terms = []
        for k2, c in sorted(series.items()):
            if c == 0:
                continue
            if k2 == 0:
                terms.append(str(c))
            elif k2 % 2 == 0:
                terms.append(f"{c}*q^{k2//2}")
            else:
                terms.append(f"{c}*q^({k2}/2)")
        print(f"  = {' + '.join(terms) if terms else '0'}")
    except Exception as e:
        traceback.print_exc()

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('P', type=int, nargs='?', default=1)
    p.add_argument('Q', type=int, nargs='?', default=0)
    p.add_argument('--q-order', type=int, default=10)
    p.add_argument('--cache-dir', default=None)
    args = p.parse_args()
    run(args.P, args.Q, args.q_order, args.cache_dir)
