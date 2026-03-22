#!/usr/bin/env python3
"""Pre-compute Dehn filling kernel for slope 1/2 at qq_order=50."""
import sys, time
sys.path.insert(0, "src")

from manifold_index.core.kernel_cache import precompute_filling_kernel, save_kernel_table

print("Starting kernel pre-computation: P=1, Q=2, qq_order=50")
print("This may take several minutes...")
t0 = time.perf_counter()

kt = precompute_filling_kernel(P=1, Q=2, qq_order=50, verbose=True)

path = save_kernel_table(kt)
total = time.perf_counter() - t0
print(f"\nSaved to {path}")
print(f"Entries: {len(kt.table)}, qq_internal={kt.qq_internal}")
print(f"Total time: {total:.1f}s ({total/60:.1f}min)")

neg = sum(1 for entry in kt.table.values() for (qq, _), _ in entry.items() if qq < 0)
print(f"Negative-qq kernel terms: {neg}")
