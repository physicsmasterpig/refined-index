#!/usr/bin/env python3
"""
scripts/rebuild_kernels.py — Rebuild all filling kernels from scratch.

Enumerates coprime (P, Q) slopes with P ∈ [-5, 5], Q ∈ [0, 5],
skips ℓ < 2 (where precomputation is not needed), and pre-computes
the refined Dehn filling kernel at qq_order = 25.

Usage:
    python scripts/rebuild_kernels.py
    python scripts/rebuild_kernels.py --qq 30   # override qq_order
    python scripts/rebuild_kernels.py --dry-run  # list slopes only
"""

from __future__ import annotations

import argparse
import sys
import time
from math import gcd
from pathlib import Path

# Ensure package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from manifold_index.core.kernel_cache import (
    precompute_filling_kernel,
    save_kernel_table,
)
from manifold_index.core.refined_dehn_filling import hj_continued_fraction


def coprime_slopes(p_range: tuple[int, int], q_range: tuple[int, int]):
    """Enumerate coprime (P, Q) with ℓ ≥ 2."""
    slopes = []
    for Q in range(q_range[0], q_range[1] + 1):
        for P in range(p_range[0], p_range[1] + 1):
            if Q == 0:
                # P/0 = ∞: only |P| = 1 is valid (coprime convention)
                if abs(P) != 1:
                    continue
            elif P == 0:
                # 0/Q: valid only if Q = ±1 → ℓ = 1, skip
                continue
            else:
                if gcd(abs(P), abs(Q)) != 1:
                    continue

            # Check ℓ (HJ continued fraction length)
            try:
                hj = hj_continued_fraction(P, Q)
            except Exception:
                continue
            if len(hj) < 2:
                continue

            slopes.append((P, Q, hj))
    return slopes


def main():
    parser = argparse.ArgumentParser(description="Rebuild filling kernels")
    parser.add_argument("--qq", type=int, default=25, help="qq_order (default 25)")
    parser.add_argument("--p-min", type=int, default=-5)
    parser.add_argument("--p-max", type=int, default=5)
    parser.add_argument("--q-min", type=int, default=0)
    parser.add_argument("--q-max", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true", help="List slopes only")
    parser.add_argument("--workers", type=int, default=None,
                        help="Worker processes per kernel (None = auto)")
    args = parser.parse_args()

    qq_order = args.qq
    slopes = coprime_slopes((args.p_min, args.p_max), (args.q_min, args.q_max))

    print(f"Slopes to compute: {len(slopes)}")
    print(f"qq_order = {qq_order}")
    print()

    for i, (P, Q, hj) in enumerate(slopes, 1):
        tag = f"{P}/{Q}" if Q != 0 else f"{P}/0 (∞)"
        print(f"  [{i:2d}] slope {tag:>8s}   HJ = {hj}   ℓ = {len(hj)}")

    if args.dry_run:
        print("\n[dry-run] No computation performed.")
        return

    print(f"\n{'='*60}")
    print(f"Starting kernel precomputation  ({len(slopes)} slopes, qq={qq_order})")
    print(f"{'='*60}\n")

    t_total = time.perf_counter()
    results = []

    for i, (P, Q, hj) in enumerate(slopes, 1):
        tag = f"{P}/{Q}" if Q != 0 else f"{P}/0"
        print(f"[{i}/{len(slopes)}] Computing kernel for slope {tag}  (ℓ={len(hj)}) ...")
        t0 = time.perf_counter()

        try:
            kt = precompute_filling_kernel(
                P, Q, qq_order,
                verbose=True,
                n_workers=args.workers,
            )
            path = save_kernel_table(kt)
            dt = time.perf_counter() - t0
            n_entries = len(kt.table)
            print(f"  ✓ {n_entries} entries, saved → {path.name}  ({dt:.1f}s)\n")
            results.append((P, Q, n_entries, dt, True))
        except Exception as e:
            dt = time.perf_counter() - t0
            print(f"  ✗ FAILED: {e}  ({dt:.1f}s)\n")
            results.append((P, Q, 0, dt, False))

    dt_total = time.perf_counter() - t_total

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary  ({len(slopes)} slopes, qq={qq_order})")
    print(f"{'='*60}")
    ok = sum(1 for *_, s in results if s)
    fail = len(results) - ok
    print(f"  Success: {ok}    Failed: {fail}    Total time: {dt_total:.1f}s")
    if fail:
        print("\nFailed slopes:")
        for P, Q, _, _, s in results:
            if not s:
                print(f"  {P}/{Q}")
    print()


if __name__ == "__main__":
    main()
