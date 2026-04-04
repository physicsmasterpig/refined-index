#!/usr/bin/env python3
"""
scripts/rebuild_kernels.py — Rebuild all filling kernels from scratch.

Enumerates coprime (P, Q) slopes with P in [-5, 5], Q in [0, 5],
skips l < 2 (where precomputation is not needed), and pre-computes
the refined Dehn filling kernel at qq_order = 25.

After all kernels finish, optionally runs two follow-up phases
automatically (unless --no-iref / --no-nc are passed):

  Phase 2 — build_iref_cache.py   Pre-compute I^ref for census manifolds.
  Phase 3 — build_nc_cache.py     Search NC cycles for census manifolds.

Usage:
    python scripts/rebuild_kernels.py
    python scripts/rebuild_kernels.py --qq 30          # override qq_order
    python scripts/rebuild_kernels.py --dry-run        # list slopes only
    python scripts/rebuild_kernels.py --force          # recompute even if cached
    python scripts/rebuild_kernels.py --qq 50 --q-min 0 --q-max 4 --workers 8
    python scripts/rebuild_kernels.py --no-iref --no-nc   # kernels only
    python scripts/rebuild_kernels.py --census m003-m050  # limit follow-up range
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from math import gcd
from pathlib import Path

# Ensure package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from manifold_index.core.kernel_cache import (
    load_kernel_table,
    precompute_filling_kernel,
    save_kernel_table,
)
from manifold_index.core.refined_dehn_filling import hj_continued_fraction


def coprime_slopes(p_range: tuple[int, int], q_range: tuple[int, int]):
    """Enumerate coprime (P, Q) with l >= 2."""
    slopes = []
    for Q in range(q_range[0], q_range[1] + 1):
        for P in range(p_range[0], p_range[1] + 1):
            if Q == 0:
                # P/0 = infinity: only |P| = 1 is valid (coprime convention)
                if abs(P) != 1:
                    continue
            elif P == 0:
                # 0/Q: valid only if Q = +/-1 -> l = 1, skip
                continue
            else:
                if gcd(abs(P), abs(Q)) != 1:
                    continue

            # Check l (HJ continued fraction length)
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
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if a kernel is already cached "
                             "(default: skip cached slopes)")
    # ---- Follow-up phase flags ----
    parser.add_argument("--no-iref", action="store_true",
                        help="Skip Phase 2: do NOT run build_iref_cache.py after kernels.")
    parser.add_argument("--no-nc", action="store_true",
                        help="Skip Phase 3: do NOT run build_nc_cache.py after kernels.")
    parser.add_argument("--census", default="m003-m412", metavar="RANGE",
                        help="Census range passed to follow-up scripts "
                             "(default: m003-m412).")
    parser.add_argument("--nc-qq", type=int, default=20,
                        help="q_order_half for NC cycle search (default: 20).")
    parser.add_argument("--nc-p-max", type=int, default=5,
                        help="NC cycle slope |P| ≤ nc_p_max (default: 5).")
    parser.add_argument("--nc-q-max", type=int, default=5,
                        help="NC cycle slope Q ≤ nc_q_max (default: 5).")
    args = parser.parse_args()

    qq_order = args.qq
    skip_existing = not args.force
    slopes = coprime_slopes((args.p_min, args.p_max), (args.q_min, args.q_max))

    print(f"Slopes to compute: {len(slopes)}")
    print(f"qq_order        = {qq_order}")
    print(f"skip_existing   = {skip_existing}  (use --force to recompute)")
    print()

    for i, (P, Q, hj) in enumerate(slopes, 1):
        tag = f"{P}/{Q}" if Q != 0 else f"{P}/0 (inf)"
        print(f"  [{i:2d}] slope {tag:>8s}   HJ = {hj}   l = {len(hj)}")

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
        print(f"[{i}/{len(slopes)}] Computing kernel for slope {tag}  (l={len(hj)}) ...")
        t0 = time.perf_counter()

        # ------------------------------------------------------------------
        # Resume support: skip if an adequate kernel is already on disk.
        # ------------------------------------------------------------------
        if skip_existing:
            existing = load_kernel_table(P, Q, qq_order)
            if existing is not None and existing.qq_order >= qq_order:
                dt = time.perf_counter() - t0
                n_entries = len(existing.table)
                print(f"  Skip: already cached at qq={existing.qq_order} "
                      f"({n_entries} entries)\n")
                results.append((P, Q, n_entries, dt, True))
                continue

        try:
            kt = precompute_filling_kernel(
                P, Q, qq_order,
                verbose=True,
                n_workers=args.workers,
            )
            path = save_kernel_table(kt)
            dt = time.perf_counter() - t0
            n_entries = len(kt.table)
            print(f"  OK {n_entries} entries, saved -> {path.name}  ({dt:.1f}s)\n")
            results.append((P, Q, n_entries, dt, True))
        except Exception as e:
            dt = time.perf_counter() - t0
            print(f"  FAILED: {e}  ({dt:.1f}s)\n")
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

    # ------------------------------------------------------------------
    # Phase 2: I^ref cache
    # ------------------------------------------------------------------
    scripts_dir = Path(__file__).resolve().parent

    if not args.no_iref:
        print(f"\n{'='*60}")
        print("Phase 2: Building I^ref cache")
        print(f"  census : {args.census}")
        print(f"  qq     : {qq_order}")
        print(f"{'='*60}\n")
        cmd = [
            sys.executable,
            str(scripts_dir / "build_iref_cache.py"),
            "--qq", str(qq_order),
            "--census", args.census,
            "--skip-existing",
        ]
        print(f"Running: {' '.join(cmd)}\n")
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"\n[WARNING] build_iref_cache.py exited with code {ret.returncode}")
    else:
        print("\n[Phase 2 skipped: --no-iref]")

    # ------------------------------------------------------------------
    # Phase 3: NC cycle cache
    # ------------------------------------------------------------------
    if not args.no_nc:
        print(f"\n{'='*60}")
        print("Phase 3: Building NC cycle cache")
        print(f"  census : {args.census}")
        print(f"  qq     : {args.nc_qq}")
        print(f"  P range: ±{args.nc_p_max},  Q range: 0…{args.nc_q_max}")
        print(f"{'='*60}\n")
        cmd = [
            sys.executable,
            str(scripts_dir / "build_nc_cache.py"),
            "--qq", str(args.nc_qq),
            "--p-max", str(args.nc_p_max),
            "--q-max", str(args.nc_q_max),
            "--census", args.census,
            "--skip-existing",
        ]
        print(f"Running: {' '.join(cmd)}\n")
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"\n[WARNING] build_nc_cache.py exited with code {ret.returncode}")
    else:
        print("\n[Phase 3 skipped: --no-nc]")

    print(f"\n{'='*60}")
    print("All phases complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
