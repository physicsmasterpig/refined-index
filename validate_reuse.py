#!/usr/bin/env python3
"""Validate that a kernel pre-computed at qq_order=50 correctly serves
requests at qq_order=8 (comparing against the slow path)."""
import sys
import time
from fractions import Fraction

sys.path.insert(0, "src")

from manifold_index.core.manifold import load_manifold
from manifold_index.core.neumann_zagier import build_neumann_zagier
from manifold_index.core.phase_space import find_easy_edges
from manifold_index.core.kernel_cache import load_kernel_table, apply_precomputed_kernel
from manifold_index.core.refined_dehn_filling import (
    compute_filled_refined_index, clear_filling_caches,
)

# Parameters
manifold_name = "m003"
P, Q = 1, 2

data = load_manifold(manifold_name)
easy = find_easy_edges(data)
nz = build_neumann_zagier(data, easy)

print("=" * 60)
print(f"Validating kernel reuse: {manifold_name}, slope {P}/{Q}")
print("=" * 60)

# Check what kernel is on disk
kt = load_kernel_table(P, Q, 8)
if kt is None:
    print("ERROR: No kernel found for qq_order=8 (even via fallback)")
    sys.exit(1)
print(f"\nLoaded kernel: stored at qq_order={kt.qq_order}, "
      f"qq_internal={kt.qq_internal}, {len(kt.table)} entries")

for req_qq in [8, 10, 20, 30, 50]:
    print(f"\n--- Testing qq_order={req_qq} ---")

    # Fast path: use the qq_order=50 kernel for this request
    clear_filling_caches()
    t0 = time.perf_counter()
    fast_series = apply_precomputed_kernel(
        kt, nz, cusp_idx=0, qq_order=req_qq,
    )
    fast_trunc = {
        k: v for k, v in fast_series.items()
        if k[0] + abs(k[-1]) <= req_qq
    }
    t_fast = time.perf_counter() - t0

    # Slow path: full IS chain (no kernel cache)
    # Temporarily hide the kernel file
    import os
    cache_dir = "data/kernel_cache"
    files = [f for f in os.listdir(cache_dir) if f.endswith(".pkl.gz")]
    for f in files:
        os.rename(os.path.join(cache_dir, f), os.path.join(cache_dir, f + ".bak"))

    clear_filling_caches()
    t0 = time.perf_counter()
    slow_result = compute_filled_refined_index(
        nz, cusp_idx=0, P=P, Q=Q, q_order_half=req_qq,
    )
    t_slow = time.perf_counter() - t0

    # Restore kernel files
    for f in files:
        os.rename(os.path.join(cache_dir, f + ".bak"), os.path.join(cache_dir, f))

    slow_series = slow_result.series

    # Compare
    all_keys = set(fast_trunc.keys()) | set(slow_series.keys())
    mismatches = 0
    for k in sorted(all_keys):
        v_fast = fast_trunc.get(k, Fraction(0))
        v_slow = slow_series.get(k, Fraction(0))
        if v_fast != v_slow:
            mismatches += 1
            if mismatches <= 5:
                print(f"  MISMATCH: {k}: fast={v_fast}, slow={v_slow}")

    status = "✅ MATCH" if mismatches == 0 else f"❌ {mismatches} mismatches"
    print(f"  {status} ({len(all_keys)} entries, "
          f"fast={t_fast:.3f}s, slow={t_slow:.1f}s)")

print("\n" + "=" * 60)
