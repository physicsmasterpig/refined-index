#!/usr/bin/env python3
"""
scripts/build_iref_cache.py — Pre-compute and cache I^ref for census manifolds.

Parallel across manifolds.  Each worker process loads all kernel tables once
at startup (pool initializer), then processes many manifolds in sequence.
File writes are per-manifold so there are no inter-process conflicts.

Usage
-----
  # All m003-m412, qq=50 kernels, 8 workers:
  python scripts/build_iref_cache.py --qq 50 --workers 8

  # Skip already-done (safe to re-run after interruption):
  python scripts/build_iref_cache.py --qq 50 --workers 8 --skip-existing

  # Mac 1 of 2 — split census:
  python scripts/build_iref_cache.py --qq 50 --workers 8 --census m003-m207
  # Mac 2 of 2:
  python scripts/build_iref_cache.py --qq 50 --workers 8 --census m208-m412
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

_M_SERIES_MIN = 3
_M_SERIES_MAX = 412


def _default_census_names() -> list[str]:
    return [f"m{i:03d}" for i in range(_M_SERIES_MIN, _M_SERIES_MAX + 1)]


def _parse_manifold_range(spec: str) -> list[str]:
    m = re.fullmatch(r"([a-zA-Z]+)(\d+)-[a-zA-Z]*(\d+)", spec)
    if m:
        prefix, start, end = m.group(1), int(m.group(2)), int(m.group(3))
        width = len(m.group(2))
        return [f"{prefix}{i:0{width}d}" for i in range(start, end + 1)]
    return [spec.strip()]


# ---------------------------------------------------------------------------
# Worker pool initializer — load all kernel tables ONCE per worker process
# ---------------------------------------------------------------------------

_WORKER_KERNELS: list = []


def _init_worker(kernel_specs: list[tuple[int, int, int]], cache_dir: str) -> None:
    global _WORKER_KERNELS
    if cache_dir:
        os.environ["MANIFOLD_INDEX_CACHE_DIR"] = cache_dir
    from manifold_index.core.kernel_cache import load_kernel_table
    _WORKER_KERNELS = []
    for P, Q, qq in kernel_specs:
        kt = load_kernel_table(P, Q, qq)
        if kt is not None:
            _WORKER_KERNELS.append(kt)


# ---------------------------------------------------------------------------
# Per-manifold worker (top-level for pickling)
# ---------------------------------------------------------------------------

def _process_manifold(args: tuple[str, bool]) -> dict:
    name, skip_existing = args
    res: dict = {"name": name, "status": "ok", "n_iref": 0,
                 "n_tet": "?", "n_cusps": 1, "elapsed": 0.0, "error": ""}
    t0 = time.perf_counter()
    try:
        from manifold_index.core.manifold import load_manifold
        from manifold_index.core.phase_space import find_easy_edges
        from manifold_index.core.neumann_zagier import build_neumann_zagier
        from manifold_index.core.kernel_cache import (
            apply_precomputed_kernel,
            _DEFAULT_IREF_DIR,
            _iref_filename,
        )
        from manifold_index.core.refined_dehn_filling import clear_filling_caches

        md = load_manifold(name)
        easy = find_easy_edges(md)
        nz = build_neumann_zagier(md, easy)
        res["n_tet"] = md.num_tetrahedra
        res["n_cusps"] = md.num_cusps

        if skip_existing:
            cache_path = _DEFAULT_IREF_DIR / _iref_filename(name, nz)
            if cache_path.exists():
                res["status"] = "skipped"
                res["elapsed"] = time.perf_counter() - t0
                return res

        for kt in _WORKER_KERNELS:
            apply_precomputed_kernel(
                kt, nz, cusp_idx=0,
                cache_iref=True, manifold_name=name, verbose=False,
            )

        cache_path = _DEFAULT_IREF_DIR / _iref_filename(name, nz)
        if cache_path.exists():
            import gzip, pickle
            with gzip.open(cache_path, "rb") as f:
                payload = pickle.load(f)
            res["n_iref"] = len(payload.get("entries", {}))

        clear_filling_caches()

    except Exception as e:
        res["status"] = "failed"
        res["error"] = str(e)

    res["elapsed"] = time.perf_counter() - t0
    return res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute I^ref cache for census manifolds (parallel).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--qq", type=int, default=50,
                        help="qq_order of the kernels to use (default: 50).")
    parser.add_argument("--census", metavar="RANGE",
                        help="Manifold range, e.g. 'm003-m412'. Default: all.")
    parser.add_argument("--manifolds", nargs="+", metavar="NAME")
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 4) - 1),
                        help="Parallel worker processes (default: cpu_count-1).")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.manifolds:
        names = args.manifolds
    elif args.census:
        names: list[str] = []
        for spec in args.census.split(","):
            names.extend(_parse_manifold_range(spec.strip()))
    else:
        names = _default_census_names()

    qq_order = args.qq

    from manifold_index.core.kernel_cache import list_cached_kernels
    from manifold_index.core.refined_dehn_filling import hj_continued_fraction

    all_kernels = list_cached_kernels()
    kernels_at_qq = sorted(
        [(P, Q, qq) for P, Q, qq in all_kernels if qq == qq_order],
        key=lambda t: len(hj_continued_fraction(t[0], t[1])),
    )

    if not kernels_at_qq:
        print(f"[SKIP] No kernels found at qq={qq_order}.")
        print(f"       Run: bin/kernel_build_start.sh --qq {qq_order} first.")
        sys.exit(0)

    print("=" * 62)
    print("  I^ref Cache Builder — parallel")
    print(f"  Host      : {os.uname().nodename}")
    print(f"  Date      : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 62)
    print(f"  Manifolds : {len(names)}  ({names[0]} … {names[-1]})")
    print(f"  qq_order  : {qq_order}")
    print(f"  Kernels   : {len(kernels_at_qq)}")
    print(f"  Workers   : {args.workers}")
    print(f"  Skip exist: {args.skip_existing}")
    print()

    if args.dry_run:
        for n in names[:15]:
            print(f"  {n}")
        if len(names) > 15:
            print(f"  … and {len(names)-15} more")
        print("\n[dry-run] No computation performed.")
        return

    tasks = [(name, args.skip_existing) for name in names]
    cache_dir = os.environ.get("MANIFOLD_INDEX_CACHE_DIR", "")

    t_global = time.perf_counter()
    n_ok = n_skip = n_fail = 0
    total = len(tasks)

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(kernels_at_qq, cache_dir),
    ) as pool:
        for idx, res in enumerate(pool.imap_unordered(_process_manifold, tasks), 1):
            st = res["status"]
            elapsed = res["elapsed"]
            name = res["name"]
            if st == "skipped":
                n_skip += 1
                print(f"[{idx:4d}/{total}] {name:8s}  SKIPPED  ({elapsed:.1f}s)", flush=True)
            elif st == "failed":
                n_fail += 1
                print(f"[{idx:4d}/{total}] {name:8s}  FAILED   ({elapsed:.1f}s)  {res['error']}", flush=True)
            else:
                n_ok += 1
                print(
                    f"[{idx:4d}/{total}] {name:8s}  ok  "
                    f"{res['n_iref']:6d} entries  "
                    f"({res['n_tet']} tet)  {elapsed:.1f}s",
                    flush=True,
                )

    dt = time.perf_counter() - t_global
    print(f"\n{'='*62}")
    print(f"Done.  ok={n_ok}  skipped={n_skip}  failed={n_fail}")
    print(f"Wall time: {dt:.1f}s  ({dt/60:.1f} min)")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
