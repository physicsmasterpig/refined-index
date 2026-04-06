#!/usr/bin/env python3
"""
scripts/build_iref_cache.py — Pre-compute and cache I^ref for census manifolds.

Computes I^ref(m, e; η^{2W}) directly over a uniform integer grid:
    m ∈ [-m_max, m_max],  e ∈ [-e_max, e_max]

No kernel tables are needed.  I^ref is a pure manifold property; the kernel
only enters later when doing Dehn filling.  This script is fully decoupled.

Parallel across manifolds.  Each worker uses compute_refined_index_batch
which builds the NZ state once per manifold and reuses it for all grid points.

Usage
-----
  # All m003-m412, qq=20, m/e grid ±20, 8 workers:
  python scripts/build_iref_cache.py --qq 20 --workers 8

  # Skip already-done entries (entry-level, safe to extend grid incrementally):
  python scripts/build_iref_cache.py --qq 20 --workers 8 --skip-existing

  # Mac 1 of 2 — split census:
  python scripts/build_iref_cache.py --qq 20 --workers 8 --census m003-m207
  # Mac 2 of 2:
  python scripts/build_iref_cache.py --qq 20 --workers 8 --census m208-m412
"""

from __future__ import annotations

import argparse
import gzip
import multiprocessing
import os
import pickle
import re
import sys
import time
from fractions import Fraction
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

def _process_manifold(args: tuple[str, int, int, int, bool]) -> dict:
    name, qq_order, m_max, e_max, skip_existing = args
    res: dict = {"name": name, "status": "ok", "n_iref": 0,
                 "n_tet": "?", "elapsed": 0.0, "error": ""}
    t0 = time.perf_counter()
    try:
        from manifold_index.core.manifold import load_manifold
        from manifold_index.core.phase_space import find_easy_edges
        from manifold_index.core.neumann_zagier import build_neumann_zagier
        from manifold_index.core.refined_index import compute_refined_index_batch
        from manifold_index.core.kernel_cache import (
            _DEFAULT_IREF_DIR,
            _iref_filename,
            save_iref_cache,
            load_iref_cache,
        )
        from manifold_index.core.refined_dehn_filling import (
            _iref_cache,
            _nz_content_key,
            clear_filling_caches,
        )

        md = load_manifold(name)
        easy = find_easy_edges(md)
        nz = build_neumann_zagier(md, easy)
        res["n_tet"] = md.num_tetrahedra
        r = nz.r

        grid_params = {"m_max": m_max, "e_max": e_max, "qq_order": qq_order}

        # Fast file-level skip: if the stored grid_params covers the requested
        # grid (same or wider m/e range, same qq), every point has already been
        # evaluated (including zero-result ones that are not stored as entries).
        # This avoids re-iterating ~3000 zero-result points on re-runs.
        if skip_existing:
            cache_path = _DEFAULT_IREF_DIR / _iref_filename(name, nz)
            if cache_path.exists():
                try:
                    with gzip.open(cache_path, "rb") as f:
                        stored = pickle.load(f)
                    gp = stored.get("grid_params", {})
                    if (gp.get("m_max", -1) >= m_max
                            and gp.get("e_max", -1) >= e_max
                            and gp.get("qq_order", -1) >= qq_order):
                        res["status"] = "skipped"
                        res["n_iref"] = len(stored.get("entries", {}))
                        res["elapsed"] = time.perf_counter() - t0
                        return res
                except Exception:
                    pass  # corrupted or missing grid_params → fall through

        # Build uniform (m, e) grid.
        # m is an integer (meridian); e is a half-integer (longitude/2),
        # stepping by 1/2 to cover both integer and half-integer values.
        # m_ext / e_ext have length r (one entry per cusp).
        # Cusp 0 is the filling cusp; all others are fixed at 0.
        e_values = [Fraction(k, 2) for k in range(-2 * e_max, 2 * e_max + 1)]
        all_entries: list[tuple] = []
        for m_i in range(-m_max, m_max + 1):
            for e_i in e_values:
                m_ext = [0] * r
                e_ext: list[int | Fraction] = [Fraction(0)] * r
                m_ext[0] = m_i
                e_ext[0] = e_i
                all_entries.append((m_ext, e_ext))

        # Entry-level skip: load existing non-zero entries into _iref_cache, then
        # filter all_entries to only those not yet cached.  Used when grid_params
        # is absent or the stored grid is narrower than requested (grid extension).
        nz_key = _nz_content_key(nz)
        n_existing = 0
        if skip_existing:
            n_existing = load_iref_cache(nz, manifold_name=name, qq_filter=qq_order)
            entries = [
                (m_ext, e_ext) for m_ext, e_ext in all_entries
                if (nz_key, tuple(m_ext), tuple(Fraction(e) for e in e_ext), qq_order)
                   not in _iref_cache
            ]
            if not entries:
                res["status"] = "skipped"
                res["n_iref"] = n_existing
                res["elapsed"] = time.perf_counter() - t0
                return res
        else:
            entries = all_entries

        # compute_refined_index_batch builds NZ state ONCE, reuses for all points
        results = compute_refined_index_batch(nz, entries, q_order_half=qq_order)

        # Insert non-empty results into the shared _iref_cache
        n_new = 0
        for (m_ext, e_ext), result in zip(entries, results):
            if not result:
                continue
            key = (
                nz_key,
                tuple(m_ext),
                tuple(Fraction(e) for e in e_ext),
                qq_order,
            )
            if key not in _iref_cache:
                _iref_cache[key] = result
                n_new += 1

        # Flush to disk, recording grid_params so future skip-existing runs are instant
        save_iref_cache(nz, manifold_name=name, grid_params=grid_params)

        res["n_iref"] = n_existing + n_new
        clear_filling_caches()

    except Exception as e:
        import traceback
        res["status"] = "failed"
        res["error"] = f"{e}\n{traceback.format_exc()}"

    res["elapsed"] = time.perf_counter() - t0
    return res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute I^ref cache for census manifolds (parallel, kernel-free).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--qq", type=int, default=20,
                        help="q_order_half for I^ref (default: 20).")
    parser.add_argument("--m-max", type=int, default=20,
                        help="Grid range: m ∈ [-m_max, m_max] (default: 20).")
    parser.add_argument("--e-max", type=int, default=20,
                        help="Grid range: e ∈ [-e_max, e_max] (default: 20).")
    parser.add_argument("--census", metavar="RANGE",
                        help="Manifold range, e.g. 'm003-m412'. Default: all.")
    parser.add_argument("--manifolds", nargs="+", metavar="NAME")
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 4) - 1),
                        help="Parallel worker processes (default: cpu_count-1).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip individual (m,e) entries already on disk "
                             "(entry-level; safe to re-run after grid extension).")
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
    m_max = args.m_max
    e_max = args.e_max
    n_grid = (2 * m_max + 1) * (4 * e_max + 1)  # e steps by 1/2

    print("=" * 62)
    print("  I^ref Cache Builder — parallel, kernel-free")
    print(f"  Host      : {os.uname().nodename}")
    print(f"  Date      : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 62)
    print(f"  Manifolds : {len(names)}  ({names[0]} … {names[-1]})")
    print(f"  qq_order  : {qq_order}")
    print(f"  Grid      : m ∈ [-{m_max},{m_max}] (step 1),  e ∈ [-{e_max},{e_max}] (step 1/2)  ({n_grid} points)")
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

    tasks = [(name, qq_order, m_max, e_max, args.skip_existing) for name in names]
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
                print(f"[{idx:4d}/{total}] {name:8s}  FAILED   ({elapsed:.1f}s)  {res['error'][:200]}", flush=True)
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
