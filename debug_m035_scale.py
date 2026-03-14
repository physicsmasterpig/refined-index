"""Test m035 at different q values to find scaling."""
import sys
import time
sys.path.insert(0, 'src')

from fractions import Fraction
from manifold_index.core.manifold import load_manifold
from manifold_index.core.phase_space import find_easy_edges
from manifold_index.core.neumann_zagier import build_neumann_zagier
from manifold_index.core.dehn_filling import compute_filled_index

data = load_manifold('m035')
easy = find_easy_edges(data)
nz = build_neumann_zagier(data, easy)

print("Testing m035 slope (3,20) at different q values:")
for q in [20, 30, 40, 50, 60]:
    t0 = time.time()
    result = compute_filled_index(nz, cusp_idx=0, P=3, Q=20, q_order_half=q)
    dt = time.time() - t0
    print(f"  q={q}: {dt:.3f}s, n_kernel={result.n_kernel_terms}, is_stably_zero={result.is_stably_zero()}")

print()
print("Testing various slopes at q=20:")
for P, Q in [(1,1),(1,2),(1,3),(2,3),(3,4),(1,5),(2,5),(3,5),(4,5),(1,10),(3,10),(1,20),(3,20)]:
    t0 = time.time()
    result = compute_filled_index(nz, cusp_idx=0, P=P, Q=Q, q_order_half=20)
    dt = time.time() - t0
    print(f"  ({P},{Q}): {dt:.3f}s, n_kernel={result.n_kernel_terms}, is_stably_zero={result.is_stably_zero()}")
