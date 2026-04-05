#!/usr/bin/env python3
"""
scripts/publish_datapack.py — Package cache files into a release tarball and
update data_packs.json ready for a GitHub Release upload.

Usage
-----
  # Pack all qq=50 kernels for |P|≤5, Q=1–5 into a tarball:
  python scripts/publish_datapack.py \\
      --type kernels --qq 50 --p-min -5 --p-max 5 --q-min 1 --q-max 5 \\
      --tag data-v1 --out dist/

  # Pack iref cache for the same census:
  python scripts/publish_datapack.py \\
      --type iref --qq 50 --tag data-v1 --out dist/

  # Pack NC cycle cache:
  python scripts/publish_datapack.py \\
      --type nc --qq 20 --tag data-v1 --out dist/

  # Dry run (list files that would be packed, no output written):
  python scripts/publish_datapack.py --type kernels --qq 50 --dry-run

After running, upload the .tar.gz file(s) to a GitHub Release as assets,
then commit the updated data_packs.json.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import sys
import tarfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).resolve().parent
V04_DIR     = SCRIPT_DIR.parent
SRC_DIR     = V04_DIR / "src"
CACHE_DIR   = V04_DIR / "cache"
DIST_DIR    = V04_DIR / "dist"
REGISTRY    = V04_DIR / "src" / "manifold_index" / "data" / "data_packs.json"

sys.path.insert(0, str(SRC_DIR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


def hj_ell(P: int, Q: int) -> int:
    from manifold_index.core.refined_dehn_filling import hj_continued_fraction
    return len(hj_continued_fraction(P, Q))


# ---------------------------------------------------------------------------
# File selection
# ---------------------------------------------------------------------------

def select_kernel_files(qq: int, p_min: int, p_max: int, q_min: int, q_max: int) -> list[Path]:
    """Return kernel .pkl.gz files matching the slope range at given qq."""
    files = []
    pat = re.compile(r"kernel_P(-?\d+)_Q(\d+)_qq(\d+)\.pkl\.gz")
    for f in sorted((CACHE_DIR / "kernel_cache").glob("kernel_P*_qq*.pkl.gz")):
        m = pat.match(f.name)
        if not m:
            continue
        P, Q, fqq = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if fqq != qq:
            continue
        if not (p_min <= P <= p_max and q_min <= Q <= q_max):
            continue
        if math.gcd(abs(P), Q) != 1:
            continue
        if hj_ell(P, Q) < 2:
            continue
        files.append(f)
    return files


def select_iref_files(qq: int) -> list[Path]:
    """Return all iref cache files (not qq-filtered — they cover all kernels)."""
    return sorted((CACHE_DIR / "iref_cache").glob("iref_*.pkl.gz"))


def select_nc_files(qq: int) -> list[Path]:
    """Return all NC cycle cache files."""
    return sorted((CACHE_DIR / "nc_cycle_cache").glob("nc_*.pkl.gz"))


# ---------------------------------------------------------------------------
# Pack
# ---------------------------------------------------------------------------

def make_tarball(
    files: list[Path],
    subdir: str,
    out_path: Path,
    dry_run: bool = False,
) -> tuple[int, str]:
    """Create a .tar.gz of the given files, storing them under subdir/.

    Returns (size_bytes, sha256).
    """
    if dry_run:
        total = sum(f.stat().st_size for f in files)
        print(f"  [dry-run] Would pack {len(files)} files → {out_path.name}")
        print(f"  [dry-run] Estimated compressed size: {human_size(total)} (uncompressed)")
        return 0, ""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Writing {out_path} ...", flush=True)
    t0 = time.perf_counter()
    with tarfile.open(out_path, "w:gz") as tar:
        for f in files:
            arcname = f"{subdir}/{f.name}"
            tar.add(f, arcname=arcname)
    elapsed = time.perf_counter() - t0
    size = out_path.stat().st_size
    digest = sha256_file(out_path)
    print(f"  Done: {human_size(size)}, sha256={digest[:16]}…  ({elapsed:.1f}s)")
    return size, digest


# ---------------------------------------------------------------------------
# Registry update
# ---------------------------------------------------------------------------

def load_registry() -> dict:
    if REGISTRY.exists():
        txt = REGISTRY.read_text().strip()
        if txt and txt != "{}":
            return json.loads(txt)
    return {
        "registry_version": 1,
        "base_url": "https://github.com/physicsmasterpig/refined-index/releases/download",
        "packs": [],
    }


def update_registry(
    pack_id: str,
    name: str,
    description: str,
    filename: str,
    release_tag: str,
    size_bytes: int,
    sha256: str,
    target_subdir: str,
    category: str,
    dry_run: bool = False,
) -> None:
    reg = load_registry()

    # Remove existing entry with same id
    reg["packs"] = [p for p in reg["packs"] if p["id"] != pack_id]

    entry = {
        "id": pack_id,
        "name": name,
        "description": description,
        "filename": filename,
        "release_tag": release_tag,
        "size_bytes": size_bytes,
        "sha256": sha256,
        "target_subdir": target_subdir,
        "category": category,
    }
    reg["packs"].append(entry)

    # Sort by category then id
    reg["packs"].sort(key=lambda x: (x["category"], x["id"]))

    if dry_run:
        print(f"\n  [dry-run] Would add to data_packs.json:")
        print(f"    {json.dumps(entry, indent=6)}")
        return

    REGISTRY.write_text(json.dumps(reg, indent=2) + "\n")
    print(f"\n  Updated {REGISTRY.relative_to(V04_DIR)}")
    print(f"  Pack id: {pack_id}")
    print(f"  Total packs in registry: {len(reg['packs'])}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package cache files into a data pack tarball for GitHub Release."
    )
    parser.add_argument("--type", choices=["kernels", "iref", "nc"], required=True,
                        help="Type of cache to pack.")
    parser.add_argument("--qq",   type=int, required=True,
                        help="qq_order the cache was computed at.")
    parser.add_argument("--p-min", type=int, default=-5)
    parser.add_argument("--p-max", type=int, default=5)
    parser.add_argument("--q-min", type=int, default=1)
    parser.add_argument("--q-max", type=int, default=5)
    parser.add_argument("--tag",  default="data-v1",
                        help="GitHub Release tag (e.g. data-v1).")
    parser.add_argument("--out",  type=Path, default=DIST_DIR,
                        help="Output directory for the tarball.")
    parser.add_argument("--dry-run", action="store_true",
                        help="List files without writing anything.")
    args = parser.parse_args()

    pack_type = args.type
    qq = args.qq

    print(f"{'='*60}")
    print(f"  Manifold Index — Data Pack Publisher")
    print(f"  Type    : {pack_type}")
    print(f"  qq      : {qq}")
    print(f"  Tag     : {args.tag}")
    print(f"  Out     : {args.out}")
    print(f"{'='*60}\n")

    if pack_type == "kernels":
        files = select_kernel_files(qq, args.p_min, args.p_max, args.q_min, args.q_max)
        subdir = "kernel_cache"
        filename = f"kernels_qq{qq}_P{args.p_min}to{args.p_max}_Q{args.q_min}to{args.q_max}.tar.gz"
        pack_id  = f"kernels_qq{qq}"
        name     = f"Filling Kernels qq={qq}"
        desc     = (f"Pre-computed K^ref Dehn filling kernels at qq={qq}, "
                    f"slopes P={args.p_min}…{args.p_max}, Q={args.q_min}…{args.q_max}.")
        category = "kernels"

    elif pack_type == "iref":
        files = select_iref_files(qq)
        subdir = "iref_cache"
        filename = f"iref_cache_qq{qq}.tar.gz"
        pack_id  = f"iref_qq{qq}"
        name     = f"I^ref Cache qq={qq}"
        desc     = f"Pre-computed I^ref(m,e) refined index cache for census manifolds at qq={qq}."
        category = "iref"

    else:  # nc
        files = select_nc_files(qq)
        subdir = "nc_cycle_cache"
        filename = f"nc_cycle_cache_qq{qq}.tar.gz"
        pack_id  = f"nc_qq{qq}"
        name     = f"NC Cycle Cache qq={qq}"
        desc     = f"Non-closable cycle search results for census manifolds at qq={qq}."
        category = "nc"

    print(f"Files selected: {len(files)}")
    for f in files[:10]:
        print(f"  {f.name}")
    if len(files) > 10:
        print(f"  … and {len(files)-10} more")

    if not files:
        print("\nERROR: No files found. Have you run the computation yet?")
        sys.exit(1)

    out_path = args.out / filename
    size_bytes, digest = make_tarball(files, subdir, out_path, dry_run=args.dry_run)

    update_registry(
        pack_id=pack_id,
        name=name,
        description=desc,
        filename=filename,
        release_tag=args.tag,
        size_bytes=size_bytes,
        sha256=digest,
        target_subdir=subdir,
        category=category,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        print(f"\n{'='*60}")
        print(f"  Next steps:")
        print(f"  1. Go to: https://github.com/physicsmasterpig/refined-index/releases/new")
        print(f"  2. Create tag '{args.tag}' and upload:")
        print(f"       {out_path}")
        print(f"  3. Commit data_packs.json (already updated):")
        print(f"       git add src/manifold_index/data/data_packs.json")
        print(f"       git commit -m 'data: publish {pack_id} pack ({args.tag})'")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
