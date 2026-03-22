#!/usr/bin/env python3
"""Pre-compute Dehn filling kernels for slope 1/2 at qq_order=30 and 50."""
import sys
import time

sys.path.insert(0, "src")

from manifold_index.core.kernel_cache import precompute_filling_kernel, save_kernel_table

# First: qq=30 (fast, ~60s)
print("=" * 60)
print("Pre-computing K^ref(1/2) at qq_order=30")
print("=" * 60)

t0 = time.perf_counter()
kt30 = precompute_filling_kernel(P=1, Q=2, qq_order=30, verbose=True, n_workers=12)
path30 = save_kernel_table(kt30)
t30 = time.perf_counter() - t0
print(f"\n✅ Saved to {path30}")
print(f"   Entries: {len(kt30.table)}, time: {t30:.1f}s ({t30/60:.1f}min)")
sys.stdout.flush()

# Second: qq=50 (the target, ~10-20 min)
print("\n" + "=" * 60)
print("Pre-computing K^ref(1/2) at qq_order=50")
print("=" * 60)

t0 = time.perf_counter()
kt50 = precompute_filling_kernel(P=1, Q=2, qq_order=50, verbose=True, n_workers=12)
path50 = save_kernel_table(kt50)
t50 = time.perf_counter() - t0

print(f"\n✅ Saved to {path50}")
print(f"   Entries: {len(kt50.table)}, time: {t50:.1f}s ({t50/60:.1f}min)")

max_m = max(abs(m) for (m, e) in kt50.table)
max_e = max(abs(e) for (m, e) in kt50.table)
print(f"   max|m|={max_m} ({max_m/kt50.qq_internal:.2f}×qq_int)")
print(f"   max|e|={max_e} ({float(max_e)/kt50.qq_internal:.2f}×qq_int)")

neg = sum(1 for entry in kt50.table.values() for (qq, _), _ in entry.items() if qq < 0)
print(f"   Negative-qq kernel terms: {neg}")

print("\n✅ All done!")
