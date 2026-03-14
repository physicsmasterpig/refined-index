"""Profile all m035 slopes to find slow ones."""
import sys, time
sys.path.insert(0, 'src')
from manifold_index.core.manifold import load_manifold
from manifold_index.core.phase_space import find_easy_edges
from manifold_index.core.neumann_zagier import build_neumann_zagier
from manifold_index.core.dehn_filling import compute_filled_index, _candidate_slopes

data = load_manifold('m035')
easy = find_easy_edges(data)
nz = build_neumann_zagier(data, easy)

slopes = _candidate_slopes(range(-3,4), range(0,21), canonical_only=True)
print(f'Total canonical slopes: {len(slopes)}')

slow_threshold = 0.5
total = 0.0
all_times = []
for P, Q in slopes:
    t0 = time.time()
    result = compute_filled_index(nz, cusp_idx=0, P=P, Q=Q, q_order_half=20)
    dt = time.time() - t0
    total += dt
    all_times.append((P, Q, dt, result.n_kernel_terms))
    
print(f'Total time: {total:.2f}s for {len(slopes)} slopes')
print(f'\nAll slopes sorted by time (slowest first):')
for P, Q, dt, nk in sorted(all_times, key=lambda x: -x[2])[:30]:
    print(f'  ({P:+d},{Q:+d}): {dt:.3f}s, n_kernel={nk}')
