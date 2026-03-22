#!/usr/bin/env python3
"""Pre-compute Dehn filling kernel for slope 1/2 at qq_order=50 (parallel)."""
import sys
import time

sys.path.insert(0, "src")

from manifold_index.core.kernel_cache import precompute_filling_kernel, save_kernel_table

print("=" * 60)
print("Pre-computing K^ref(1/2) at qq_order=50")
print("=" * 60)

t0 = time.perf_counter()
kt = precompute_filling_kernel(P=1, Q=2, qq_order=50, verbose=True, n_workers=12)
t_total = time.perf_counter() - t0

path = save_kernel_table(kt)
print(f"\nSaved to {path}")
print(f"Entries: {len(kt.table)}, qq_internal={kt.qq_internal}")
print(f"Total time: {t_total:.1f}s ({t_total / 60:.1f}min)")

max_m = max(abs(m) for (m, e) in kt.table)
max_e = max(abs(e) for (m, e) in kt.table)
print(f"max|m|={max_m} ({max_m / kt.qq_internal:.2f}×qq_int)")
print(f"max|e|={max_e} ({float(max_e) / kt.qq_internal:.2f}×qq_int)")

neg = sum(1 for entry in kt.table.values() for (qq, _), _ in entry.items() if qq < 0)
print(f"Negative-qq kernel terms: {neg}")
