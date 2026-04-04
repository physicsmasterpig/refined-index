#!/usr/bin/env python3
"""
scripts/build_nc_cache.py — Search non-closable cycles for census manifolds.

Iterates over a range of census manifolds, runs find_non_closable_cycles for
each cusp, and persists the results to disk via save_nc_cycle_cache().

Why cache NC cycles?
--------------------
Finding non-closable cycles requires evaluating the (unrefined) Dehn-filling
index at every slope in the search range and checking whether the result is
stably zero.  For a typical 1-cusp manifold with p_range = (-5, 5) and
q_range = (0, 5), this means ~30 slope evaluations per manifold.  At qq=20
(the default for NC search) each evaluation takes ~1-10 s, so total
precomputation is ~1-5 min per manifold.  Storing the result on disk means the
GUI can show NC cycles instantly.

Usage
-----
  # All m-series manifolds, default search range, qq=20:
  python scripts/build_nc_cache.py

  # Custom range and qq:
  python scripts/build_nc_cache.py --census m003-m050 --qq 20

  # Wider slope search (more thorough):
  python scripts/build_nc_cache.py --census m003-m050 --p-max 7 --q-max 6

  # Explicit manifolds:
  python scripts/build_nc_cache.py --manifolds m003 m004 s000

  # Skip already-cached manifolds:
  python scripts/build_nc_cache.py --census m003-m412 --skip-existing

  # Dry run:
  python scripts/build_nc_cache.py --census m003-m050 --dry-run
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Make package importable when running from repo root or scripts/
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _build_pipeline(name: str):
    from manifold_index.core.manifold import load_manifold
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import build_neumann_zagier

    md = load_manifold(name)
    easy = find_easy_edges(md)
    nz = build_neumann_zagier(md, easy)
    return md, nz


def _cache_exists(nz_data, name, q_order_half, p_range, q_range) -> bool:
    from manifold_index.core.kernel_cache import (
        _DEFAULT_NC_DIR, _nc_cycle_filename,
    )
    path = _DEFAULT_NC_DIR / _nc_cycle_filename(
        name, nz_data, q_order_half, p_range, q_range,
    )
    return path.exists()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute NC cycle cache for census manifolds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--qq", type=int, default=20,
        help="q_order_half for the NC search (default: 20).",
    )
    parser.add_argument(
        "--p-max", type=int, default=5,
        help="Search |P| ≤ p_max (default: 5); range is [-p_max, +p_max].",
    )
    parser.add_argument(
        "--q-max", type=int, default=5,
        help="Search Q ∈ [0, q_max] (default: 5).",
    )
    parser.add_argument(
        "--census", metavar="RANGE",
        help="Manifold range, e.g. 'm003-m412'.  Default: m003-m412.",
    )
    parser.add_argument(
        "--manifolds", nargs="+", metavar="NAME",
        help="Explicit manifold names (overrides --census).",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip manifolds that already have an NC cycle cache file.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-slope progress from find_non_closable_cycles.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List what would be computed without actually running.",
    )
    args = parser.parse_args()

    # ----- Determine manifold list -----
    if args.manifolds:
        names = args.manifolds
    elif args.census:
        names: list[str] = []
        for spec in args.census.split(","):
            names.extend(_parse_manifold_range(spec.strip()))
    else:
        names = _default_census_names()

    q_order_half = args.qq
    p_range = (-args.p_max, args.p_max)
    q_range = (0, args.q_max)

    print(f"Manifolds      : {len(names)}")
    print(f"q_order_half   : {q_order_half}")
    print(f"slope P range  : {p_range[0]} … {p_range[1]}")
    print(f"slope Q range  : {q_range[0]} … {q_range[1]}")
    print(f"skip_existing  : {args.skip_existing}")
    print()

    if args.dry_run:
        print(f"Manifolds ({len(names)} total):")
        for n in names[:20]:
            print(f"  {n}")
        if len(names) > 20:
            print(f"  … and {len(names) - 20} more")
        print("\n[dry-run] No computation performed.")
        return

    from manifold_index.core.dehn_filling import find_non_closable_cycles
    from manifold_index.core.kernel_cache import save_nc_cycle_cache

    t_global = time.perf_counter()
    n_skipped = 0
    n_ok = 0
    n_fail = 0

    for idx, name in enumerate(names, 1):
        print(f"[{idx}/{len(names)}] {name} ", end="", flush=True)
        t_m = time.perf_counter()

        # --- Build NZ pipeline ---
        try:
            md, nz = _build_pipeline(name)
        except Exception as e:
            print(f"  PIPELINE ERROR: {e}")
            n_fail += 1
            continue

        n_cusps = md.num_cusps
        print(f"(n={md.num_tetrahedra}, r={n_cusps})", end=" ", flush=True)

        # --- Skip if already cached ---
        if args.skip_existing and _cache_exists(nz, name, q_order_half, p_range, q_range):
            print("→ skipped (cache exists)")
            n_skipped += 1
            continue

        # --- Search NC cycles for each cusp ---
        try:
            nc_results = []
            for cusp_idx in range(n_cusps):
                nc = find_non_closable_cycles(
                    nz,
                    cusp_idx=cusp_idx,
                    p_range=range(p_range[0], p_range[1] + 1),
                    q_range=range(q_range[0], q_range[1] + 1),
                    q_order_half=q_order_half,
                    use_symmetry=True,
                    verbose=args.verbose,
                )
                nc_results.append(nc)

            path = save_nc_cycle_cache(
                nz, name, nc_results,
                q_order_half=q_order_half,
                p_range=p_range,
                q_range=q_range,
            )

            total_nc = sum(len(r.cycles) for r in nc_results)
            dt = time.perf_counter() - t_m
            print(f"→ {total_nc} NC cycles found  ({dt:.1f}s)  → {path.name}")
            n_ok += 1

        except Exception as e:
            dt = time.perf_counter() - t_m
            print(f"  COMPUTE ERROR ({dt:.1f}s): {e}")
            n_fail += 1

    dt_total = time.perf_counter() - t_global
    print(f"\n{'='*60}")
    print(f"Done.  ok={n_ok}  skipped={n_skipped}  failed={n_fail}")
    print(f"Total time: {dt_total:.1f}s  ({dt_total/60:.1f} min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
