#!/usr/bin/env python3
"""
scripts/build_iref_cache.py — Pre-compute and cache I^ref for census manifolds.

After the Dehn-filling kernels are ready (rebuild_kernels.py), run this script
to populate the on-disk I^ref cache so that the GUI's Dehn-filling calculations
become near-instant lookups instead of multi-minute computations.

How it works
------------
For each census manifold:
  1. Run the standard pipeline:  load_manifold → find_easy_edges → build_neumann_zagier
  2. For each pre-computed kernel at the requested qq_order, call
     apply_precomputed_kernel(..., cache_iref=True).  This evaluates
     I^ref(m,e) for every (m,e) grid point in that kernel's table and saves
     the results to disk.
  3. Subsequent kernels for the same manifold find the I^ref entries already
     cached — only truly new grid points are computed.

After this script finishes, opening any of the processed manifolds in the GUI
and running Dehn filling is essentially free (sub-second).

Usage
-----
  # All 1-cusp "m" manifolds, using kernels computed at qq=25:
  python scripts/build_iref_cache.py --qq 25

  # A custom range:
  python scripts/build_iref_cache.py --qq 25 --census m003-m050

  # Specific manifolds:
  python scripts/build_iref_cache.py --qq 25 --manifolds m003 m004 s000

  # Only re-run manifolds that don't have a cache file yet:
  python scripts/build_iref_cache.py --qq 25 --census m003-m412 --skip-existing

  # Dry run (list manifolds + kernels, no computation):
  python scripts/build_iref_cache.py --qq 25 --dry-run
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
_M_SERIES_MAX = 412  # snappy's OrientableCuspedCensus "m" series ends around here


def _default_census_names() -> list[str]:
    """Return the full m003…m412 census range."""
    return [f"m{i:03d}" for i in range(_M_SERIES_MIN, _M_SERIES_MAX + 1)]


def _parse_manifold_range(spec: str) -> list[str]:
    """Parse a range spec like 'm003-m050' or a single name like 'm003'."""
    m = re.fullmatch(r"([a-zA-Z]+)(\d+)-[a-zA-Z]*(\d+)", spec)
    if m:
        prefix, start, end = m.group(1), int(m.group(2)), int(m.group(3))
        width = len(m.group(2))
        return [f"{prefix}{i:0{width}d}" for i in range(start, end + 1)]
    # Single name
    return [spec.strip()]


def _build_pipeline(name: str):
    """Load manifold and build NeumannZagierData.

    Returns (ManifoldData, NeumannZagierData) or raises on failure.
    """
    from manifold_index.core.manifold import load_manifold
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import build_neumann_zagier

    md = load_manifold(name)
    easy = find_easy_edges(md)
    nz = build_neumann_zagier(md, easy)
    return md, nz


def _iref_cache_path(nz_data, name: str) -> Path:
    """Return the expected iref cache file path for this manifold."""
    from manifold_index.core.kernel_cache import _DEFAULT_IREF_DIR, _iref_filename
    return _DEFAULT_IREF_DIR / _iref_filename(name, nz_data)


def _iref_cache_exists(nz_data, name: str) -> bool:
    return _iref_cache_path(nz_data, name).exists()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute I^ref cache for census manifolds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--qq", type=int, default=25,
        help="qq_order of the kernels to use (default: 25).",
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
        help="Skip manifolds that already have an iref cache file.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without computing anything.",
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

    qq_order = args.qq

    # ----- Find available kernels at the requested qq -----
    from manifold_index.core.kernel_cache import (
        list_cached_kernels,
        load_kernel_table,
        apply_precomputed_kernel,
    )

    all_kernels = list_cached_kernels()
    kernels_at_qq = [(P, Q, qq) for P, Q, qq in all_kernels if qq == qq_order]

    if not kernels_at_qq:
        print(f"[SKIP] No kernels found at qq={qq_order} — nothing to cache.")
        print(f"       Run rebuild_kernels.py --qq {qq_order} first.")
        sys.exit(0)  # Not an error: kernels simply not computed yet

    print(f"Manifolds      : {len(names)}")
    print(f"qq_order       : {qq_order}")
    print(f"Kernels at qq  : {len(kernels_at_qq)}")
    print(f"skip_existing  : {args.skip_existing}")
    print()

    # Sort kernels by HJ chain length (shortest/cheapest first)
    from manifold_index.core.refined_dehn_filling import hj_continued_fraction
    kernels_at_qq.sort(key=lambda t: len(hj_continued_fraction(t[0], t[1])))

    if args.dry_run:
        print("Kernels (sorted by ℓ):")
        for P, Q, qq in kernels_at_qq:
            hj = hj_continued_fraction(P, Q)
            print(f"  K^ref({P:+d}/{Q})  HJ={hj}  ℓ={len(hj)}")
        print(f"\nManifolds ({len(names)} total):")
        for n in names[:20]:
            print(f"  {n}")
        if len(names) > 20:
            print(f"  … and {len(names) - 20} more")
        print("\n[dry-run] No computation performed.")
        return

    # ----- Main loop -----
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
        if args.skip_existing and _iref_cache_exists(nz, name):
            print("→ skipped (cache exists)")
            n_skipped += 1
            continue

        # --- Populate iref cache via each kernel ---
        n_iref_total = 0
        try:
            for k_idx, (P, Q, qq) in enumerate(kernels_at_qq):
                kt = load_kernel_table(P, Q, qq)
                if kt is None:
                    continue

                # For multi-cusp manifolds, use cusp 0; all cusps share
                # the same iref grid so one pass suffices per manifold.
                apply_precomputed_kernel(
                    kt,
                    nz,
                    cusp_idx=0,
                    cache_iref=True,
                    manifold_name=name,
                    verbose=False,
                )

            # Count entries in the saved file
            p = _iref_cache_path(nz, name)
            if p.exists():
                import gzip, pickle
                with gzip.open(p, "rb") as f:
                    payload = pickle.load(f)
                n_iref_total = len(payload.get("entries", {}))

        except Exception as e:
            print(f"  COMPUTE ERROR: {e}")
            n_fail += 1
            # Clear in-memory iref cache to avoid cross-manifold pollution
            from manifold_index.core.refined_dehn_filling import clear_filling_caches
            clear_filling_caches()
            continue

        dt = time.perf_counter() - t_m
        print(f"→ {n_iref_total} I^ref entries cached  ({dt:.1f}s)")
        n_ok += 1

        # Clear in-memory iref cache before moving to next manifold
        # to avoid unbounded memory growth.
        from manifold_index.core.refined_dehn_filling import clear_filling_caches
        clear_filling_caches()

    dt_total = time.perf_counter() - t_global
    print(f"\n{'='*60}")
    print(f"Done.  ok={n_ok}  skipped={n_skipped}  failed={n_fail}")
    print(f"Total time: {dt_total:.1f}s  ({dt_total/60:.1f} min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
