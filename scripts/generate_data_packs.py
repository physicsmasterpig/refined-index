#!/usr/bin/env python3
"""
scripts/generate_data_packs.py — Batch-generate data packs for distribution.

Computes filling kernels and/or refined indices for census manifolds,
packages them into .tar.gz archives, and updates data_packs.json with
sizes and SHA-256 hashes.

Usage
-----
    # Generate kernels for coprime (P,Q) with |P|,|Q| <= 10 at qq=50:
    python scripts/generate_data_packs.py --pack kernels-qq50

    # Generate all packs:
    python scripts/generate_data_packs.py --pack all

    # Just package existing cache files (skip computation):
    python scripts/generate_data_packs.py --pack kernels-qq50 --package-only

    # Dry run — show what would be computed:
    python scripts/generate_data_packs.py --pack kernels-qq50 --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import time
from math import gcd
from pathlib import Path

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from manifold_index.core.kernel_cache import (
    _DEFAULT_CACHE_DIR,
    _DEFAULT_IREF_DIR,
    _user_cache_dir,
    precompute_filling_kernel,
    save_kernel_table,
)

# ── Registry path ────────────────────────────────────────────────────
_REGISTRY_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "manifold_index" / "data" / "data_packs.json"
)
_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "dist" / "data_packs"


# ── Slope enumeration ────────────────────────────────────────────────

def coprime_slopes(max_pq: int) -> list[tuple[int, int]]:
    """Enumerate coprime (P, Q) with Q >= 2 and |P| <= max_pq, Q <= max_pq.

    Returns slopes in a canonical form: Q > 0, sorted by Q then P.
    Includes both positive and negative P (the kernel is symmetric
    in (m,e) ↔ (-m,-e) but not in P ↔ -P for general Q).
    """
    slopes = []
    for Q in range(2, max_pq + 1):
        for P in range(-max_pq, max_pq + 1):
            if gcd(abs(P), Q) == 1:
                slopes.append((P, Q))
    slopes.sort(key=lambda s: (s[1], s[0]))
    return slopes


# ── Pack definitions ─────────────────────────────────────────────────

PACK_CONFIGS = {
    "kernels-qq50": {
        "type": "kernels",
        "qq": 50,
        "max_pq": 10,
        "target_subdir": "kernel_cache",
    },
    "kernels-qq100": {
        "type": "kernels",
        "qq": 100,
        "max_pq": 10,
        "target_subdir": "kernel_cache",
    },
    "iref-census-qq20": {
        "type": "iref",
        "qq": 20,
        "target_subdir": "iref_cache",
    },
    "iref-census-qq50": {
        "type": "iref",
        "qq": 50,
        "target_subdir": "iref_cache",
    },
}


# ── Kernel generation ────────────────────────────────────────────────

def generate_kernels(qq: int, max_pq: int, dry_run: bool = False) -> list[Path]:
    """Compute and save kernel tables for all coprime slopes.

    Returns list of output file paths.
    """
    slopes = coprime_slopes(max_pq)
    print(f"Kernel generation: qq={qq}, max |P|,|Q|={max_pq}")
    print(f"  {len(slopes)} slopes to compute")

    if dry_run:
        for P, Q in slopes:
            print(f"  Would compute kernel P={P}, Q={Q}, qq={qq}")
        return []

    files = []
    total = len(slopes)
    for i, (P, Q) in enumerate(slopes, 1):
        print(f"\n[{i}/{total}] Computing kernel P={P}, Q={Q}, qq={qq}…")
        t0 = time.time()
        try:
            kt = precompute_filling_kernel(
                P, Q, qq, verbose=True,
            )
            path = save_kernel_table(kt)
            dt = time.time() - t0
            print(f"  → saved {path.name}  ({dt:.1f}s)")
            files.append(path)
        except Exception as exc:
            print(f"  ✗ FAILED: {exc}")

    print(f"\nKernel generation complete: {len(files)}/{total} succeeded")
    return files


# ── I^ref generation ─────────────────────────────────────────────────

def generate_iref(qq: int, dry_run: bool = False) -> list[Path]:
    """Compute refined indices for all SnapPy census manifolds.

    Returns list of output file paths.
    """
    try:
        import snappy
    except ImportError:
        print("ERROR: snappy not installed; cannot generate I^ref census data.")
        return []

    from manifold_index.core.manifold import load_manifold
    from manifold_index.core.phase_space import find_phase_space_basis
    from manifold_index.core.neumann_zagier import build_neumann_zagier
    from manifold_index.core.index_3d import compute_refined_index
    from manifold_index.core.kernel_cache import save_iref_cache
    from manifold_index.app.workers import build_eval_grid

    # Get all census manifolds
    census_names = []
    for table in [snappy.OrientableCuspedCensus]:
        for M in table:
            census_names.append(M.name())

    print(f"I^ref generation: qq={qq}")
    print(f"  {len(census_names)} census manifolds")

    if dry_run:
        for name in census_names[:20]:
            print(f"  Would compute I^ref for {name}, qq={qq}")
        if len(census_names) > 20:
            print(f"  … and {len(census_names) - 20} more")
        return []

    files = []
    total = len(census_names)
    for i, name in enumerate(census_names, 1):
        print(f"\n[{i}/{total}] Computing I^ref for {name}, qq={qq}…")
        t0 = time.time()
        try:
            md = load_manifold(name)
            ps = find_phase_space_basis(md)
            nz = build_neumann_zagier(md, ps)
            eval_points = build_eval_grid(nz.r)
            results = compute_refined_index(nz, eval_points, qq)
            save_path = save_iref_cache(nz, manifold_name=name)
            dt = time.time() - t0
            print(f"  → saved ({dt:.1f}s)")
            if save_path:
                files.append(save_path)
        except Exception as exc:
            print(f"  ✗ FAILED: {exc}")

    print(f"\nI^ref generation complete: {len(files)}/{total} succeeded")
    return files


# ── Packaging ────────────────────────────────────────────────────────

def package_files(
    pack_id: str,
    target_subdir: str,
    pattern: str = "*.pkl.gz",
) -> tuple[Path, int, str]:
    """Collect files from cache and create a .tar.gz archive.

    Returns (archive_path, size_bytes, sha256_hex).
    """
    cache_dir = _user_cache_dir()
    source_dir = cache_dir / target_subdir

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    files = sorted(source_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No {pattern} files in {source_dir}")

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    archive_name = f"{pack_id}.tar.gz"
    archive_path = _OUTPUT_DIR / archive_name

    print(f"\nPackaging {len(files)} files into {archive_name}…")

    with tarfile.open(archive_path, "w:gz") as tar:
        for f in files:
            # Archive path: target_subdir/filename
            arcname = f"{target_subdir}/{f.name}"
            tar.add(str(f), arcname=arcname)
            print(f"  + {arcname}")

    size = archive_path.stat().st_size
    sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()

    print(f"  Archive: {archive_path}")
    print(f"  Size: {size:,} bytes ({size / 1024**2:.1f} MB)")
    print(f"  SHA-256: {sha}")

    return archive_path, size, sha


# ── Registry update ──────────────────────────────────────────────────

def update_registry(pack_id: str, size_bytes: int, sha256: str, filename: str) -> None:
    """Update data_packs.json with actual size and hash."""
    with open(_REGISTRY_PATH) as f:
        data = json.load(f)

    for entry in data.get("packs", []):
        if entry["id"] == pack_id:
            entry["size_bytes"] = size_bytes
            entry["sha256"] = sha256
            entry["filename"] = filename
            break
    else:
        print(f"WARNING: pack '{pack_id}' not found in registry!")
        return

    with open(_REGISTRY_PATH, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

    print(f"  Updated {_REGISTRY_PATH.name}: size={size_bytes:,}, sha256={sha256[:16]}…")


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate data packs for the Refined 3D Index Calculator.",
    )
    parser.add_argument(
        "--pack",
        required=True,
        choices=list(PACK_CONFIGS.keys()) + ["all"],
        help="Which pack to generate.",
    )
    parser.add_argument(
        "--package-only",
        action="store_true",
        help="Skip computation; only package existing cache files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be computed without running.",
    )
    args = parser.parse_args()

    packs = list(PACK_CONFIGS.keys()) if args.pack == "all" else [args.pack]

    for pack_id in packs:
        cfg = PACK_CONFIGS[pack_id]
        print(f"\n{'='*60}")
        print(f"  Pack: {pack_id}")
        print(f"{'='*60}")

        # Step 1: Compute (unless --package-only)
        if not args.package_only:
            if cfg["type"] == "kernels":
                generate_kernels(cfg["qq"], cfg["max_pq"], dry_run=args.dry_run)
            elif cfg["type"] == "iref":
                generate_iref(cfg["qq"], dry_run=args.dry_run)

        if args.dry_run:
            continue

        # Step 2: Package
        try:
            archive_name = f"{pack_id}.tar.gz"
            archive_path, size, sha = package_files(
                pack_id, cfg["target_subdir"],
            )
        except FileNotFoundError as e:
            print(f"  SKIP packaging: {e}")
            continue

        # Step 3: Update registry
        update_registry(pack_id, size, sha, archive_name)

    print(f"\n{'='*60}")
    print("Done!")
    if not args.dry_run:
        print(f"Archives written to: {_OUTPUT_DIR}")
        print("Upload them as GitHub release assets under the 'data-v1' tag.")


if __name__ == "__main__":
    main()
