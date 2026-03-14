"""Debug script for m035 compute_filled_index performance."""
import sys
import time
sys.path.insert(0, 'src')

from fractions import Fraction
from manifold_index.core.manifold import load_manifold
from manifold_index.core.phase_space import find_easy_edges
from manifold_index.core.neumann_zagier import build_neumann_zagier
from manifold_index.core.index_3d import compute_index_3d_python
from manifold_index.core.dehn_filling import compute_filled_index, find_rs

data = load_manifold('m035')
easy = find_easy_edges(data)
nz = build_neumann_zagier(data, easy)

print("Timing compute_index_3d_python(m035, m=0, e=0, q=20)...")
t0 = time.time()
result = compute_index_3d_python(nz, m_ext=[0], e_ext=[Fraction(0)], q_order_half=20)
dt = time.time() - t0
print(f"  Done in {dt:.3f}s, n_terms={result.n_terms}")
print(f"  nonzero coeffs: {[k+result.min_power for k,v in enumerate(result.coeffs) if v != 0][:10]}")

print()
print("Timing compute_filled_index(m035, slope=(3,20), q=20) verbose...")
t0 = time.time()
result2 = compute_filled_index(nz, cusp_idx=0, P=3, Q=20, q_order_half=20, verbose=True)
dt = time.time() - t0
print(f"  Done in {dt:.3f}s")
print(f"  series: {dict(sorted(result2.series.items()))}")
print(f"  is_stably_zero: {result2.is_stably_zero()}")
