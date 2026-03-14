"""Debug script for m035 performance issue."""
import sys
import time
sys.path.insert(0, 'src')

from fractions import Fraction
from manifold_index.core.manifold import load_manifold
from manifold_index.core.phase_space import find_easy_edges
from manifold_index.core.neumann_zagier import build_neumann_zagier
from manifold_index.core.index_3d import (
    enumerate_summation_terms,
    valid_half_integer_patterns,
    _exact_e0_candidates,
    _tet_index_series,
    build_kappa,
    phase_exponent,
)
import numpy as np

data = load_manifold('m035')
easy = find_easy_edges(data)
nz = build_neumann_zagier(data, easy)
print(f"m035: n={nz.n}, r={nz.r}, n_int={nz.n - nz.r}")
g_inv = nz.g_NZ_inv()

t0 = time.time()
patterns = valid_half_integer_patterns(g_inv, nz.n, nz.r)
print(f"valid patterns: {len(patterns)} ({time.time()-t0:.3f}s)")
for p in patterns:
    print(f"  delta={p}")

n = nz.n
r = nz.r
n_int = n - r

int_cols = g_inv[:, n + r: 2 * n]
int_cols_int = np.array([[int(v) for v in row] for row in int_cols], dtype=np.int64)
nu_x_int = nz.nu_x[r:n]

print("\nFor m_ext=[0], e_ext=[0]:")
for delta in patterns:
    print(f"  delta={delta}")
    delta_half = [Fraction(delta[j], 2) for j in range(n_int)]
    kappa_base = build_kappa([0], [Fraction(0)], delta_half, n, r)
    base_args_frac = g_inv @ kappa_base.astype(object)
    base_args = np.array([int(Fraction(v)) for v in base_args_frac], dtype=int)
    print(f"    base_args={base_args}")
    phase_base = phase_exponent(kappa_base, nz.nu_x, nz.nu_p, n, r, 0)
    print(f"    phase_base={phase_base}")
    phase_base_x2 = int(2 * phase_base)

    t0 = time.time()
    candidates = _exact_e0_candidates(
        base_args, int_cols_int, nu_x_int, phase_base_x2, 20, n, n_int
    )
    print(f"    n_candidates={len(candidates)} ({time.time()-t0:.3f}s)")

    # Check how many pass the F_x2 filter
    valid = []
    from manifold_index.core.index_3d import tet_degree
    for e0 in candidates:
        args_full = base_args + int_cols_int @ e0
        md = sum(tet_degree(int(args_full[a]), int(args_full[n+a])) for a in range(n))
        pe = phase_base - int(nu_x_int @ e0)
        if md + pe <= 20:
            valid.append(e0)
    print(f"    n_valid_after_degree_filter={len(valid)}")

    # Time _tet_index_series for a few typical cases
    if valid:
        t0 = time.time()
        count = 0
        for e0 in valid[:50]:
            args_full = base_args + int_cols_int @ e0
            pe = int(phase_base - nu_x_int @ e0)
            budget = 20 - pe
            for a in range(n):
                s = _tet_index_series(int(args_full[a]), int(args_full[n+a]), max(0, budget))
                count += 1
        print(f"    {count} _tet_index_series calls for first 50 valid: {time.time()-t0:.3f}s")

print("\nNow calling enumerate_summation_terms...")
t0 = time.time()
terms = enumerate_summation_terms(nz, [0], [Fraction(0)], 20)
print(f"  {len(terms)} terms in {time.time()-t0:.3f}s")

if terms:
    print("  First few terms:")
    for t in terms[:3]:
        print(f"    phase={t['phase_exp']}, min_deg={t['min_degree']:.1f}, tet_args={t['tet_args']}")
    if len(terms) > 3:
        print(f"  ... and {len(terms)-3} more")
