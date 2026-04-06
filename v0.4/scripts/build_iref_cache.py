#!/usr/bin/env python3
"""
scripts/build_iref_cache.py — Pre-compute and cache I^ref for census manifolds.

Computes I^ref(m, e; η^{2W}) directly over a uniform integer grid:
    m ∈ [-m_max, m_max],  e ∈ [-e_max, e_max]  (e in half-integer steps)

Parallel at two levels:
  • Manifold level: each manifold is an independent worker task.
  • Grid level: manifolds with n_tet ≥ --chunk-tet are split into multiple
    tasks (grid chunks) that run concurrently.  Each chunk writes its results
    atomically (fcntl-locked read-merge-write), so all workers can safely
    share the same output file.

Usage
-----
  # All m003-m412, qq=20, m/e grid ±20, 13 workers:
  python scripts/build_iref_cache.py --qq 20 --workers 13

  # Skip already-done entries (entry-level for partial grids,
  # file-level when grid_params is present):
  python scripts/build_iref_cache.py --qq 20 --workers 13 --skip-existing

  # Mac 1 of 2 — split census:
  python scripts/build_iref_cache.py --qq 20 --workers 13 --census m003-m207
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


def _get_n_tet(name: str) -> int:
    """Return the tetrahedron count for a census manifold (fast, no NZ build)."""
    try:
        import snappy
        return snappy.Manifold(name).num_tetrahedra()
    except Exception:
        return 1  # conservative fallback: no chunking


# ---------------------------------------------------------------------------
# Per-chunk worker (top-level for pickling)
# ---------------------------------------------------------------------------

def _process_chunk(args: tuple) -> dict:
    """Compute I^ref for one chunk of the (m, e) grid for a single manifold.

    args = (name, qq_order, m_max, e_max, skip_existing, chunk_idx, n_chunks)
    """
    name, qq_order, m_max, e_max, skip_existing, chunk_idx, n_chunks = args
    res: dict = {"name": name, "chunk": chunk_idx, "n_chunks": n_chunks,
                 "status": "ok", "n_new": 0, "n_existing": 0,
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

        # Fast file-level skip: if grid_params on disk covers this request,
        # every chunk can independently detect and skip without loading the
        # full entry list.  Multi-chunk manifolds rely on this since they
        # don't write grid_params (avoids the race of "which chunk finishes last").
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
                        res["n_existing"] = len(stored.get("entries", {}))
                        res["elapsed"] = time.perf_counter() - t0
                        return res
                except Exception:
                    pass

        # Build the full (m, e) grid for this manifold.
        e_values = [Fraction(k, 2) for k in range(-2 * e_max, 2 * e_max + 1)]
        all_entries: list[tuple] = []
        for m_i in range(-m_max, m_max + 1):
            for e_i in e_values:
                m_ext = [0] * r
                e_ext: list[int | Fraction] = [Fraction(0)] * r
                m_ext[0] = m_i
                e_ext[0] = e_i
                all_entries.append((m_ext, e_ext))

        # This chunk owns every n_chunks-th entry starting at chunk_idx.
        chunk_entries = all_entries[chunk_idx::n_chunks]

        # Entry-level skip: load existing non-zero entries, filter this chunk.
        nz_key = _nz_content_key(nz)
        n_existing = 0
        if skip_existing:
            n_existing = load_iref_cache(nz, manifold_name=name, qq_filter=qq_order)
            chunk_entries = [
                (m_ext, e_ext) for m_ext, e_ext in chunk_entries
                if (nz_key, tuple(m_ext), tuple(Fraction(e) for e in e_ext), qq_order)
                   not in _iref_cache
            ]
            if not chunk_entries:
                res["status"] = "skipped"
                res["n_existing"] = n_existing
                res["elapsed"] = time.perf_counter() - t0
                return res

        results = compute_refined_index_batch(nz, chunk_entries, q_order_half=qq_order)

        n_new = 0
        for (m_ext, e_ext), result in zip(chunk_entries, results):
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

        # Write grid_params only for single-chunk runs: we know the whole grid
        # was evaluated in one pass so the flag is safe.  Multi-chunk runs skip
        # this to avoid the race of "which chunk finishes last" — their subsequent
        # re-runs use entry-level skip instead (all entries are already cached).
        gp_to_write = grid_params if n_chunks == 1 else None
        save_iref_cache(nz, manifold_name=name, grid_params=gp_to_write)

        res["n_new"] = n_new
        res["n_existing"] = n_existing
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
        description="Pre-compute I^ref cache for census manifolds.",
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
    parser.add_argument("--chunk-tet", type=int, default=4,
                        help="Split manifolds with n_tet >= this value into "
                             "multiple grid chunks (default: 4).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip entries already on disk.")
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
    n_grid = (2 * m_max + 1) * (4 * e_max + 1)

    print("=" * 62)
    print("  I^ref Cache Builder — parallel, kernel-free")
    print(f"  Host      : {os.uname().nodename}")
    print(f"  Date      : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 62)
    print(f"  Manifolds : {len(names)}  ({names[0]} … {names[-1]})")
    print(f"  qq_order  : {qq_order}")
    print(f"  Grid      : m ∈ [-{m_max},{m_max}] (step 1),"
          f"  e ∈ [-{e_max},{e_max}] (step 1/2)  ({n_grid} points)")
    print(f"  Workers   : {args.workers}")
    print(f"  Chunk tet : {args.chunk_tet}+ tet → grid chunks")
    print(f"  Skip exist: {args.skip_existing}")
    print()

    if args.dry_run:
        for n in names[:15]:
            print(f"  {n}")
        if len(names) > 15:
            print(f"  … and {len(names)-15} more")
        print("\n[dry-run] No computation performed.")
        return

    # Build task list: large manifolds get multiple chunk tasks.
    # Each chunk owns entries[chunk_idx::n_chunks] of the full grid.
    tasks: list[tuple] = []
    n_chunks_for: dict[str, int] = {}
    for name in names:
        n_tet = _get_n_tet(name)
        # Number of chunks: cap at args.workers to avoid over-subscription.
        if n_tet >= args.chunk_tet:
            n_ch = min(n_tet - 1, args.workers)
        else:
            n_ch = 1
        n_chunks_for[name] = n_ch
        for ci in range(n_ch):
            tasks.append((name, qq_order, m_max, e_max, args.skip_existing, ci, n_ch))

    cache_dir = os.environ.get("MANIFOLD_INDEX_CACHE_DIR", "")
    if cache_dir:
        os.environ["MANIFOLD_INDEX_CACHE_DIR"] = cache_dir

    t_global = time.perf_counter()
    # Accumulate per-manifold totals across chunks
    manifold_totals: dict[str, dict] = {}
    n_ok = n_skip = n_fail = 0
    total_manifolds = len(names)

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
        for res in pool.imap_unordered(_process_chunk, tasks):
            name = res["name"]
            n_ch = n_chunks_for[name]
            st = res["status"]

            if name not in manifold_totals:
                manifold_totals[name] = {
                    "chunks_done": 0, "n_new": 0, "n_existing": 0,
                    "n_tet": res["n_tet"], "status": "ok",
                    "elapsed_max": 0.0, "failed": False, "skipped_all": True,
                }
            mt = manifold_totals[name]
            mt["chunks_done"] += 1
            mt["n_new"] += res.get("n_new", 0)
            mt["n_existing"] = max(mt["n_existing"], res.get("n_existing", 0))
            mt["elapsed_max"] = max(mt["elapsed_max"], res["elapsed"])
            if st == "failed":
                mt["failed"] = True
            if st != "skipped":
                mt["skipped_all"] = False

            # Print when the last chunk for this manifold finishes
            if mt["chunks_done"] == n_ch:
                manifold_totals[name + "_done"] = True  # marker
                elapsed = mt["elapsed_max"]
                n_iref = mt["n_existing"] + mt["n_new"]
                n_tet_str = mt["n_tet"]
                done_so_far = sum(
                    1 for nm in names
                    if manifold_totals.get(nm, {}).get("chunks_done", 0)
                    == n_chunks_for.get(nm, 1)
                )
                if mt["failed"]:
                    n_fail += 1
                    print(f"[{done_so_far:4d}/{total_manifolds}] {name:8s}"
                          f"  FAILED  ({elapsed:.1f}s)", flush=True)
                elif mt["skipped_all"]:
                    n_skip += 1
                    print(f"[{done_so_far:4d}/{total_manifolds}] {name:8s}"
                          f"  SKIPPED ({elapsed:.1f}s)", flush=True)
                else:
                    n_ok += 1
                    chunk_note = f"  [{n_ch}ch]" if n_ch > 1 else ""
                    print(
                        f"[{done_so_far:4d}/{total_manifolds}] {name:8s}  ok"
                        f"{chunk_note}"
                        f"  {n_iref:6d} entries"
                        f"  ({n_tet_str} tet)  {elapsed:.1f}s",
                        flush=True,
                    )

    dt = time.perf_counter() - t_global
    print(f"\n{'='*62}")
    print(f"Done.  ok={n_ok}  skipped={n_skip}  failed={n_fail}")
    print(f"Wall time: {dt:.1f}s  ({dt/60:.1f} min)")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
