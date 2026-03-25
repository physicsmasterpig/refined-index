#!/usr/bin/env python3
"""Rebuild all cached IS kernels using the new degree-bound analysis.

Compares each rebuilt kernel against the original to verify identical
table contents, then overwrites the old file.
"""

import gzip
import pickle
import sys
import time
from pathlib import Path

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from manifold_index.core.kernel_cache import (
    KernelTable,
    list_cached_kernels,
    precompute_filling_kernel,
    save_kernel_table,
)

CACHE_DIR = Path(__file__).parent / "data" / "kernel_cache"


def load_raw(path: Path) -> KernelTable:
    """Load a KernelTable from disk without the memory cache."""
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def compare_tables(old: KernelTable, new: KernelTable) -> tuple[int, int, int]:
    """Compare two kernel tables.

    Returns (n_common, n_old_only, n_new_only).
    Checks that all common entries have identical coefficients.
    """
    old_keys = set(old.table.keys())
    new_keys = set(new.table.keys())
    common = old_keys & new_keys
    old_only = old_keys - new_keys
    new_only = new_keys - old_keys

    # Verify common entries are identical
    mismatches = 0
    for key in common:
        if old.table[key] != new.table[key]:
            mismatches += 1
            print(f"  ⚠ MISMATCH at {key}!")

    if mismatches:
        print(f"  ⚠ {mismatches} coefficient mismatches!")
    return len(common), len(old_only), len(new_only)


def main():
    kernels = list_cached_kernels(CACHE_DIR)
    print(f"Found {len(kernels)} cached kernels: {kernels}\n")

    # Sort: small qq first (fast), large qq last
    kernels.sort(key=lambda x: (x[2], x[0], x[1]))

    total_t0 = time.perf_counter()

    for P, Q, qq in kernels:
        fname = f"kernel_P{P}_Q{Q}_qq{qq}.pkl.gz"
        path = CACHE_DIR / fname

        print(f"{'='*60}")
        print(f"Rebuilding {fname} ...")

        # Load old kernel for comparison
        old_kt = load_raw(path)
        print(f"  Old: {len(old_kt.table)} entries, "
              f"compute_time={old_kt.compute_time_s:.1f}s")

        # Rebuild
        t0 = time.perf_counter()
        new_kt = precompute_filling_kernel(
            P, Q, qq,
            eta_order=old_kt.eta_order,
            verbose=True,
            n_workers=0,  # serial for reproducibility
        )
        rebuild_time = time.perf_counter() - t0

        print(f"  New: {len(new_kt.table)} entries, "
              f"compute_time={new_kt.compute_time_s:.1f}s "
              f"(wall={rebuild_time:.1f}s)")

        # Compare
        n_common, n_old_only, n_new_only = compare_tables(old_kt, new_kt)
        print(f"  Compare: {n_common} common, "
              f"{n_old_only} old-only, {n_new_only} new-only")

        if n_old_only > 0:
            print(f"  ⚠ WARNING: {n_old_only} entries in old kernel "
                  f"not found in new! This should not happen.")
        if n_new_only > 0:
            print(f"  ℹ {n_new_only} new entries not in old "
                  f"(wider degree-bound coverage)")

        # Save (overwrite)
        save_path = save_kernel_table(new_kt, CACHE_DIR)
        print(f"  ✓ Saved to {save_path}")
        print()

    total_time = time.perf_counter() - total_t0
    print(f"{'='*60}")
    print(f"All {len(kernels)} kernels rebuilt in {total_time:.1f}s "
          f"({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
