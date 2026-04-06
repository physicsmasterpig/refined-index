#!/usr/bin/env python3
"""
scripts/build_nc_cache.py — Search non-closable cycles for census manifolds.

Parallel across manifolds using multiprocessing.Pool.  Each manifold is
fully independent so there are no inter-process conflicts.

The cache file is range-agnostic: re-running with a wider P/Q range merges
new slopes into the existing file without losing previously computed results.
--skip-existing checks coverage at the slope level, not the file level.

Usage
-----
  # All m003-m412, qq=20, 8 workers:
  python scripts/build_nc_cache.py --qq 20 --workers 8

  # Skip slopes already cached (entry-level, safe after range extension):
  python scripts/build_nc_cache.py --qq 20 --workers 8 --skip-existing

  # Mac 1 of 2 — split census:
  python scripts/build_nc_cache.py --qq 20 --workers 8 --census m003-m207
  # Mac 2 of 2:
  python scripts/build_nc_cache.py --qq 20 --workers 8 --census m208-m412
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
# Per-manifold worker (top-level for pickling)
# ---------------------------------------------------------------------------

def _process_manifold(args: tuple) -> dict:
    name, qq, p_max, q_max, skip_existing = args
    res: dict = {"name": name, "status": "ok", "n_nc": 0,
                 "n_tet": "?", "n_cusps": 1, "elapsed": 0.0, "error": ""}
    t0 = time.perf_counter()
    try:
        from manifold_index.core.manifold import load_manifold
        from manifold_index.core.phase_space import find_easy_edges
        from manifold_index.core.neumann_zagier import build_neumann_zagier
        from manifold_index.core.dehn_filling import find_non_closable_cycles
        from manifold_index.core.kernel_cache import (
            save_nc_cycle_cache,
            load_nc_cycle_cache,
        )

        p_range = (-p_max, p_max)
        q_range = (0, q_max)

        md = load_manifold(name)
        easy = find_easy_edges(md)
        nz = build_neumann_zagier(md, easy)
        res["n_tet"] = md.num_tetrahedra
        res["n_cusps"] = md.num_cusps

        # Entry-level skip: check if all requested slopes are already covered
        if skip_existing:
            cached = load_nc_cycle_cache(
                nz, name, qq,
                p_range=p_range,
                q_range=q_range,
            )
            if cached is not None:
                res["status"] = "skipped"
                res["n_nc"] = sum(len(r.cycles) for r in cached)
                res["elapsed"] = time.perf_counter() - t0
                return res

        nc_results = []
        for cusp_idx in range(md.num_cusps):
            nc = find_non_closable_cycles(
                nz,
                cusp_idx=cusp_idx,
                p_range=range(p_range[0], p_range[1] + 1),
                q_range=range(q_range[0], q_range[1] + 1),
                q_order_half=qq,
                use_symmetry=True,
                verbose=False,
            )
            nc_results.append(nc)

        save_nc_cycle_cache(nz, name, nc_results, q_order_half=qq)
        res["n_nc"] = sum(len(r.cycles) for r in nc_results)

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
        description="Pre-compute NC cycle cache for census manifolds (parallel).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--qq", type=int, default=20,
                        help="q_order_half for NC search (default: 20).")
    parser.add_argument("--p-max", type=int, default=5,
                        help="Search |P| <= p_max (default: 5).")
    parser.add_argument("--q-max", type=int, default=5,
                        help="Search Q in [0, q_max] (default: 5).")
    parser.add_argument("--census", metavar="RANGE",
                        help="Manifold range, e.g. 'm003-m412'. Default: all.")
    parser.add_argument("--manifolds", nargs="+", metavar="NAME")
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 4) - 1),
                        help="Parallel worker processes (default: cpu_count-1).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip manifolds whose cache already covers all "
                             "requested slopes (entry-level, safe after range "
                             "extension).")
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

    print("=" * 62)
    print("  NC Cycle Cache Builder — parallel")
    print(f"  Host      : {os.uname().nodename}")
    print(f"  Date      : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 62)
    print(f"  Manifolds : {len(names)}  ({names[0]} … {names[-1]})")
    print(f"  qq        : {args.qq}")
    print(f"  P range   : [-{args.p_max}, +{args.p_max}]")
    print(f"  Q range   : [0, {args.q_max}]")
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

    tasks = [
        (name, args.qq, args.p_max, args.q_max, args.skip_existing)
        for name in names
    ]
    cache_dir = os.environ.get("MANIFOLD_INDEX_CACHE_DIR", "")
    if cache_dir:
        os.environ["MANIFOLD_INDEX_CACHE_DIR"] = cache_dir

    t_global = time.perf_counter()
    n_ok = n_skip = n_fail = 0
    total = len(tasks)

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
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
                nc_str = f"{res['n_nc']} NC" if res["n_nc"] else "0 NC"
                print(
                    f"[{idx:4d}/{total}] {name:8s}  ok  "
                    f"{nc_str:8s}  "
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
