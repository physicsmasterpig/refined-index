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

    # Generate I^ref for census manifolds (parallelized):
    python scripts/generate_data_packs.py --pack iref-census-qq20 --resume
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing
import os
import pickle
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from itertools import product as itertools_product
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

def generate_kernels(qq: int, max_pq: int, dry_run: bool = False, resume: bool = False) -> list[Path]:
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
    skipped = 0
    for i, (P, Q) in enumerate(slopes, 1):
        # Skip existing kernels in resume mode
        if resume:
            expected = _DEFAULT_CACHE_DIR / f"kernel_P{P}_Q{Q}_qq{qq}.pkl.gz"
            if expected.exists():
                skipped += 1
                print(f"[{i}/{total}] SKIP P={P}, Q={Q}, qq={qq} (already exists)")
                files.append(expected)
                continue

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

    computed = len(files) - skipped
    print(f"\nKernel generation complete: {computed} computed, {skipped} skipped, {len(files)}/{total} total")
    return files


# ── I^ref generation ─────────────────────────────────────────────────

# ── I^ref generation ─────────────────────────────────────────────────

def _iref_eval_grid(r: int) -> list[tuple[list[int], list[Fraction]]]:
    """Build evaluation grid for refined index: m ∈ {-2..2}, e ∈ {-2..2} step 1/2."""
    per_cusp = [
        (m, Fraction(k, 2))
        for m in (-2, -1, 0, 1, 2)
        for k in (-4, -3, -2, -1, 0, 1, 2, 3, 4)
    ]
    pts: list[tuple[list[int], list[Fraction]]] = []
    for combo in itertools_product(*([per_cusp] * r)):
        pts.append(([p[0] for p in combo], [p[1] for p in combo]))
    return pts


def _worker_compute_iref_one(
    name: str,
    qq: int,
    cache_dir: str,
) -> tuple[str, str | None, float, int]:
    """Compute I^ref for a single census manifold and save to disk.

    Designed to run inside a :class:`ProcessPoolExecutor` worker.
    Each worker is fully self-contained: loads the manifold, computes
    the batch refined index for every (m, e) in the evaluation grid,
    and writes the result to a gzipped pickle.

    Returns ``(name, path_or_None, elapsed_seconds, n_nonzero)``.
    """
    t0 = time.time()
    try:
        from manifold_index.core.manifold import load_manifold
        from manifold_index.core.phase_space import find_phase_space_basis
        from manifold_index.core.neumann_zagier import build_neumann_zagier
        from manifold_index.core.refined_index import compute_refined_index_batch
        from manifold_index.core.kernel_cache import _nz_hash, _iref_filename

        md = load_manifold(name)
        ps = find_phase_space_basis(md)
        nz = build_neumann_zagier(md, ps)

        eval_points = _iref_eval_grid(nz.r)
        results = compute_refined_index_batch(nz, eval_points, qq)

        # Build entries dict matching the save_iref_cache payload format
        entries: dict[tuple, dict] = {}
        n_nonzero = 0
        for (m_ext, e_ext), result in zip(eval_points, results):
            short_key = (tuple(m_ext), tuple(Fraction(e) for e in e_ext), qq)
            entries[short_key] = result
            if result:
                n_nonzero += 1

        # Write directly (bypass _iref_cache global to avoid cross-process issues)
        d = Path(cache_dir)
        d.mkdir(parents=True, exist_ok=True)
        path = d / _iref_filename(name, nz)

        payload = {
            "nz_hash": _nz_hash(nz),
            "manifold_name": name,
            "n_tetrahedra": int(nz.n),
            "n_cusps": int(nz.r),
            "num_hard": int(nz.num_hard),
            "entries": entries,
        }
        with gzip.open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        return (name, str(path), time.time() - t0, n_nonzero)

    except Exception as exc:
        return (name, None, time.time() - t0, -1)


def generate_iref(
    qq: int,
    dry_run: bool = False,
    resume: bool = False,
    max_cusps: int = 1,
) -> list[Path]:
    """Compute refined indices for SnapPy census manifolds (parallelized).

    Parameters
    ----------
    qq : int
        q-order (q_order_half).
    dry_run : bool
        Print what would be done without computing.
    resume : bool
        Skip manifolds whose iref cache file already exists.
    max_cusps : int
        Only process manifolds with at most this many cusps.
        Default 1 (1-cusp only; grid = 45 pts).
        Set to 2 for 2-cusp (grid = 2025 pts per manifold — much slower).

    Returns list of output file paths.
    """
    try:
        import snappy
    except ImportError:
        print("ERROR: snappy not installed; cannot generate I^ref census data.")
        return []

    # Pre-import so fork() copies loaded modules to workers
    from manifold_index.core.manifold import load_manifold          # noqa: F401
    from manifold_index.core.phase_space import find_phase_space_basis  # noqa: F401
    from manifold_index.core.neumann_zagier import build_neumann_zagier  # noqa: F401
    from manifold_index.core.refined_index import compute_refined_index_batch  # noqa: F401
    from manifold_index.core.kernel_cache import _iref_filename, _nz_hash  # noqa: F401

    # Enumerate census manifolds
    census_names: list[str] = []
    for M in snappy.OrientableCuspedCensus:
        if M.num_cusps() <= max_cusps:
            census_names.append(M.name())

    total = len(census_names)
    grid_size = sum(45 ** min(max_cusps, c) for c in range(1, max_cusps + 1))
    print(f"I^ref generation: qq={qq}, max_cusps={max_cusps}")
    print(f"  {total} census manifolds (≤ {max_cusps} cusps)")
    print(f"  Eval grid: 45^r pts per manifold")

    if dry_run:
        for name in census_names[:20]:
            print(f"  Would compute I^ref for {name}, qq={qq}")
        if total > 20:
            print(f"  ... and {total - 20} more")
        return []

    # Resume: check which manifolds already have cache files
    cache_dir = _DEFAULT_IREF_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    existing_files = set(f.name for f in cache_dir.glob("iref_*.pkl.gz"))

    if resume:
        todo_names: list[str] = []
        skipped = 0
        for name in census_names:
            # We can't know the exact filename without NZ data, so check
            # by prefix: iref_{safe_name}_
            safe_name = name.replace("/", "_").replace(" ", "_")
            prefix = f"iref_{safe_name}_"
            if any(f.startswith(prefix) for f in existing_files):
                skipped += 1
            else:
                todo_names.append(name)
        print(f"  Resume: {skipped} already cached, {len(todo_names)} remaining")
    else:
        todo_names = census_names
        skipped = 0

    if not todo_names:
        print("  Nothing to compute!")
        return list(cache_dir.glob("iref_*.pkl.gz"))

    # Parallel dispatch: one task per manifold, natural work-stealing
    n_workers = max(1, os.cpu_count() - 2)
    print(f"  Dispatching {len(todo_names)} manifolds to {n_workers} workers...")

    cache_dir_str = str(cache_dir)
    files: list[Path] = []
    failed: list[str] = []
    done_count = 0
    t0_all = time.time()
    log_interval = max(1, len(todo_names) // 200)  # ~0.5% increments

    ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        futures = {
            pool.submit(_worker_compute_iref_one, name, qq, cache_dir_str): name
            for name in todo_names
        }

        for future in as_completed(futures):
            name_key = futures[future]
            try:
                name, path_str, dt, n_nz = future.result()
            except Exception as exc:
                failed.append(name_key)
                done_count += 1
                continue

            done_count += 1
            if path_str is None or n_nz < 0:
                failed.append(name)
            else:
                files.append(Path(path_str))

            if done_count % log_interval == 0 or done_count == len(todo_names):
                elapsed = time.time() - t0_all
                rate = done_count / elapsed if elapsed > 0 else 0
                eta_s = (len(todo_names) - done_count) / rate if rate > 0 else 0
                pct = done_count / len(todo_names) * 100
                print(
                    f"  [{done_count}/{len(todo_names)}] ({pct:.1f}%) "
                    f"{elapsed:.0f}s elapsed, ~{eta_s:.0f}s remaining, "
                    f"{len(files)} saved, {len(failed)} failed"
                )

    total_elapsed = time.time() - t0_all
    print(
        f"\nI^ref generation complete: "
        f"{len(files)} saved, {len(failed)} failed, "
        f"{skipped} previously cached, "
        f"{total_elapsed:.0f}s total"
    )
    if failed:
        print(f"  Failed manifolds: {failed[:20]}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")

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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip slopes/manifolds that already have cached results.",
    )
    parser.add_argument(
        "--max-cusps",
        type=int,
        default=1,
        help="For iref packs: only process manifolds with at most this many cusps (default 1).",
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
                generate_kernels(cfg["qq"], cfg["max_pq"], dry_run=args.dry_run, resume=args.resume)
            elif cfg["type"] == "iref":
                generate_iref(cfg["qq"], dry_run=args.dry_run,
                              resume=args.resume, max_cusps=args.max_cusps)

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
