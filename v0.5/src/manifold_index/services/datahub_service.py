"""
services.datahub_service
========================
DataHubService — orchestrates data pack download, generation, and
cache export/publishing.

Rule: no Qt, no app imports.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from manifold_index.core import (
    data_packs as _dp_mod,
    kernel_cache as _kc_mod,
)


class DataHubService:
    """Orchestrates data pack download, generation, and publishing."""

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    @staticmethod
    def load_registry(use_remote: bool = False) -> Any:
        """Load the pack registry (bundled or remote).

        Parameters
        ----------
        use_remote : bool
            If True, try fetching the latest registry from GitHub first;
            falls back to the bundled registry on network failure.

        Returns
        -------
        PackRegistry
        """
        registry = _dp_mod.load_registry(use_remote=use_remote)
        _dp_mod.check_installed(registry)
        return registry

    @staticmethod
    def download_pack(
        registry: Any,
        pack: Any,
        progress_fn: Callable[[int, int], None] | None = None,
        status_fn: Callable[[str], None] | None = None,
    ) -> int:
        """Download and install a data pack.

        Parameters
        ----------
        registry : PackRegistry
        pack : PackInfo
        progress_fn : callable(received_bytes, total_bytes) or None
        status_fn : callable(message) or None

        Returns
        -------
        int
            Number of files extracted.

        Raises
        ------
        Exception
            On network or extraction errors.
        """
        return _dp_mod.download_and_install(
            registry=registry,
            pack=pack,
            progress_fn=progress_fn,
            status_fn=status_fn,
        )

    @staticmethod
    def remove_pack(pack: Any) -> int:
        """Remove installed files for a data pack.

        Parameters
        ----------
        pack : PackInfo

        Returns
        -------
        int
            Number of files removed.
        """
        return _dp_mod.uninstall_pack(pack)

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    @staticmethod
    def build_kernels(
        slopes: list[tuple[int, int]],
        qq: int,
        n_workers: int,
        skip_existing: bool,
        progress_fn: Callable[[str], None] | None = None,
        status_fn: Callable[[str], None] | None = None,
        cancel_fn: Callable[[], bool] | None = None,
    ) -> list[tuple[int, int, str]]:
        """Pre-compute filling kernels for the given slopes.

        For each (P, Q) slope, checks whether a kernel at the requested
        *qq* order (or higher) is already cached.  Skips if *skip_existing*
        is True and a sufficient kernel is found.

        Parameters
        ----------
        slopes : list[(P, Q)]
        qq : int
            Series truncation order.
        n_workers : int
            Parallel worker count (passed to ``precompute_filling_kernel``).
        skip_existing : bool
            If True, skip slopes that are already cached at ≥ *qq*.
        progress_fn, status_fn : callable(str) or None
        cancel_fn : callable() → bool or None
            Return True to abort remaining slopes.

        Returns
        -------
        list[(P, Q, status_str)]
            Status is one of: "done", "skipped", "error:<msg>", "cancelled".
        """
        results: list[tuple[int, int, str]] = []
        cached = {(p, q, o) for p, q, o in _kc_mod.list_cached_kernels()}

        for P, Q in slopes:
            if cancel_fn and cancel_fn():
                results.append((P, Q, "cancelled"))
                break

            # Check skip condition — any cached entry at ≥ qq suffices
            if skip_existing:
                already = any(
                    cp == P and cq == Q and co >= qq
                    for cp, cq, co in cached
                )
                if already:
                    results.append((P, Q, "skipped"))
                    if status_fn:
                        status_fn(f"Kernel ({P},{Q}) already cached at qq≥{qq} — skipped")
                    continue

            try:
                if status_fn:
                    status_fn(f"Building kernel ({P},{Q}) at qq={qq}…")
                kt = _kc_mod.precompute_filling_kernel(
                    P=P,
                    Q=Q,
                    qq_order=qq,
                    n_workers=n_workers,
                    progress_callback=progress_fn,
                )
                # Save to disk
                _kc_mod.save_kernel_table(kt)
                results.append((P, Q, "done"))
                if status_fn:
                    status_fn(f"Kernel ({P},{Q}) done")
                # Update local cached set
                cached.add((P, Q, qq))
            except Exception as exc:
                results.append((P, Q, f"error:{exc}"))
                if status_fn:
                    status_fn(f"Kernel ({P},{Q}) error: {exc}")

        return results

    @staticmethod
    def build_iref_cache(
        manifold_names: list[str],
        qq: int,
        m_max: int,
        e_max: int,
        n_workers: int,
        skip_existing: bool,
        progress_fn: Callable[[str], None] | None = None,
        status_fn: Callable[[str], None] | None = None,
    ) -> list[tuple[str, str]]:
        """Build I^ref disk cache for a list of manifolds.

        For each manifold, loads the manifold, builds the NZ matrix, then
        evaluates I^ref on the (m, e) grid and saves to disk.

        Parameters
        ----------
        manifold_names : list[str]
        qq, m_max, e_max : int
        n_workers : int         (reserved; computation is serial per manifold)
        skip_existing : bool    Skip manifolds that already have a cache file.

        Returns
        -------
        list[(name, status_str)]
        """
        from fractions import Fraction
        from manifold_index.core import manifold as _manifold_mod
        from manifold_index.core import phase_space as _ps_mod
        from manifold_index.core import neumann_zagier as _nz_mod
        from manifold_index.core import refined_index as _ri_mod

        results: list[tuple[str, str]] = []
        existing = {d["manifold_name"] for d in _kc_mod.list_iref_caches()}

        for name in manifold_names:
            if skip_existing and name in existing:
                results.append((name, "skipped"))
                if status_fn:
                    status_fn(f"I^ref cache for {name} already exists — skipped")
                continue

            try:
                if status_fn:
                    status_fn(f"Building I^ref cache for {name}…")

                # Load manifold
                manifold_data = _manifold_mod.load_manifold(name)
                easy_result = _ps_mod.find_easy_edges(manifold_data)
                nz_data = _nz_mod.build_neumann_zagier(manifold_data, easy_result)

                # Enumerate grid
                r = int(nz_data.r)
                m_values = list(range(-m_max, m_max + 1))
                e_halves = list(range(-2 * e_max, 2 * e_max + 1))
                e_values = [Fraction(e2, 2) for e2 in e_halves]
                m_ext_choices = _grid_product(m_values, r)
                e_ext_choices = _grid_product(e_values, r)

                n_total = len(m_ext_choices) * len(e_ext_choices)
                n_done = 0
                for m_ext in m_ext_choices:
                    for e_ext in e_ext_choices:
                        _ri_mod.compute_refined_index(nz_data, m_ext, e_ext, qq)
                        n_done += 1
                        if progress_fn and n_done % 10 == 0:
                            progress_fn(f"{name}: {n_done}/{n_total}")

                # Persist
                path = _kc_mod.save_iref_cache(
                    nz_data=nz_data,
                    manifold_name=name,
                    grid_params={"m_max": m_max, "e_max": e_max, "qq_order": qq},
                )
                results.append((name, "done" if path else "no-entries"))
                if status_fn:
                    status_fn(f"I^ref cache for {name} saved ({n_done} entries)")

            except Exception as exc:
                results.append((name, f"error:{exc}"))
                if status_fn:
                    status_fn(f"I^ref cache for {name} error: {exc}")

        return results

    @staticmethod
    def build_nc_cache(
        manifold_names: list[str],
        qq: int,
        p_max: int,
        q_max: int,
        n_workers: int,
        skip_existing: bool,
        progress_fn: Callable[[str], None] | None = None,
        status_fn: Callable[[str], None] | None = None,
    ) -> list[tuple[str, str]]:
        """Build NC-cycle disk cache for a list of manifolds.

        Parameters
        ----------
        manifold_names : list[str]
        qq : int                Series truncation order.
        p_max, q_max : int      Search range: |P| ≤ p_max, 0 ≤ Q ≤ q_max.
        n_workers : int         (reserved; serial per manifold for now)
        skip_existing : bool

        Returns
        -------
        list[(name, status_str)]
        """
        from manifold_index.core import manifold as _manifold_mod
        from manifold_index.core import phase_space as _ps_mod
        from manifold_index.core import neumann_zagier as _nz_mod
        from manifold_index.core import dehn_filling as _df_mod

        results: list[tuple[str, str]] = []
        existing_nc = {d["manifold_name"] for d in _kc_mod.list_nc_cycle_caches()}

        for name in manifold_names:
            if skip_existing and name in existing_nc:
                results.append((name, "skipped"))
                if status_fn:
                    status_fn(f"NC cache for {name} already exists — skipped")
                continue

            try:
                if status_fn:
                    status_fn(f"Building NC cache for {name}…")

                manifold_data = _manifold_mod.load_manifold(name)
                easy_result = _ps_mod.find_easy_edges(manifold_data)
                nz_data = _nz_mod.build_neumann_zagier(manifold_data, easy_result)

                r = int(nz_data.r)
                nc_results = []
                for cusp_idx in range(r):
                    nc_result = _df_mod.find_non_closable_cycles(
                        nz_data=nz_data,
                        cusp_idx=cusp_idx,
                        p_range=range(-p_max, p_max + 1),
                        q_range=range(0, q_max + 1),
                        q_order_half=qq,
                    )
                    nc_results.append(nc_result)

                _kc_mod.save_nc_cycle_cache(
                    nz_data=nz_data,
                    manifold_name=name,
                    q_order_half=qq,
                    nc_results=nc_results,
                )
                n_cycles = sum(len(nc.cycles) for nc in nc_results)
                results.append((name, f"done:{n_cycles}"))
                if status_fn:
                    status_fn(f"NC cache for {name}: {n_cycles} cycles found")

            except Exception as exc:
                results.append((name, f"error:{exc}"))
                if status_fn:
                    status_fn(f"NC cache for {name} error: {exc}")

        return results

    # ------------------------------------------------------------------
    # Local cache introspection
    # ------------------------------------------------------------------

    @staticmethod
    def list_local_cache() -> dict:
        """Aggregate summary of all local cache contents.

        Returns
        -------
        dict with keys "kernels", "iref", "nc", each containing:
          {"count": int, "size_bytes": int, "entries": list[dict]}
        """
        kernel_entries = _kc_mod.list_cached_kernels()
        iref_entries = _kc_mod.list_iref_caches()
        nc_entries = _kc_mod.list_nc_cycle_caches()

        # Compute sizes for kernel entries (list of (P,Q,qq) tuples)
        kernel_size = 0
        kernel_dicts: list[dict] = []
        for P, Q, qq in kernel_entries:
            kernel_dicts.append({"P": P, "Q": Q, "qq_order": qq})

        # Compute sizes for iref entries
        iref_size = 0
        for entry in iref_entries:
            p = Path(entry["path"])
            if p.exists():
                iref_size += p.stat().st_size

        # Compute sizes for nc entries
        nc_size = 0
        for entry in nc_entries:
            p = Path(entry["path"])
            if p.exists():
                nc_size += p.stat().st_size

        return {
            "kernels": {
                "count": len(kernel_entries),
                "size_bytes": kernel_size,
                "entries": kernel_dicts,
            },
            "iref": {
                "count": len(iref_entries),
                "size_bytes": iref_size,
                "entries": iref_entries,
            },
            "nc": {
                "count": len(nc_entries),
                "size_bytes": nc_size,
                "entries": nc_entries,
            },
        }

    @staticmethod
    def list_cache_files(type_filter: str | None = None) -> list[dict]:
        """List individual cache files.

        Parameters
        ----------
        type_filter : str or None
            One of "kernels", "iref", "nc", or None for all.

        Returns
        -------
        list[dict]  each dict has keys: "type", "path", and type-specific metadata.
        """
        files: list[dict] = []

        if type_filter in (None, "kernels"):
            for P, Q, qq in _kc_mod.list_cached_kernels():
                files.append({"type": "kernels", "P": P, "Q": Q, "qq_order": qq})

        if type_filter in (None, "iref"):
            for entry in _kc_mod.list_iref_caches():
                row = dict(entry)
                row["type"] = "iref"
                p = Path(entry["path"])
                row["size_bytes"] = p.stat().st_size if p.exists() else 0
                files.append(row)

        if type_filter in (None, "nc"):
            for entry in _kc_mod.list_nc_cycle_caches():
                row = dict(entry)
                row["type"] = "nc"
                p = Path(entry["path"])
                row["size_bytes"] = p.stat().st_size if p.exists() else 0
                files.append(row)

        return files

    # ------------------------------------------------------------------
    # Export & Share
    # ------------------------------------------------------------------

    @staticmethod
    def export_cache_files(
        file_paths: list,
        formats: list[str],
        output_dir,
    ) -> list:
        """Convert pkl.gz cache files to the requested formats.

        Parameters
        ----------
        file_paths : list[str | Path]
            Paths to .pkl.gz files (kernels or iref).
        formats : list[str]
            Subset of ["mathematica", "json"].
        output_dir : str | Path

        Returns
        -------
        list[Path]
            Paths of files written.
        """
        from manifold_index.utils import cache_export as _ce

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []

        for raw_path in file_paths:
            path = Path(raw_path)
            stem = path.stem.replace(".pkl", "")

            # Determine type by filename prefix
            if path.name.startswith("iref_"):
                try:
                    data = _ce.load_iref_file(path)
                except Exception:
                    continue
                if "mathematica" in formats:
                    out = output_dir / f"{stem}.m"
                    _ce.export_iref_mathematica(data, out)
                    written.append(out)
                if "json" in formats:
                    out = output_dir / f"{stem}.json"
                    _ce.export_iref_json(data, out)
                    written.append(out)

            elif path.name.startswith("kernel_"):
                try:
                    kt = _ce.load_kernel_file(path)
                except Exception:
                    continue
                if "mathematica" in formats:
                    out = output_dir / f"{stem}.m"
                    _ce.export_kernel_mathematica(kt, out)
                    written.append(out)
                if "json" in formats:
                    out = output_dir / f"{stem}.json"
                    _ce.export_kernel_json(kt, out)
                    written.append(out)

        return written

    @staticmethod
    def create_tarball(
        file_paths: list,
        pack_name: str,
        release_tag: str,
        output_dir,
        update_registry: bool = True,
    ) -> dict:
        """Package cache files into a .tar.gz data pack.

        Parameters
        ----------
        file_paths : list[str | Path]
            Paths of cache files to bundle.
        pack_name : str
            Short identifier (e.g. "kernels_qq50").
        release_tag : str
            GitHub release tag (e.g. "data-v2").
        output_dir : str | Path
        update_registry : bool
            If True, update data_packs.json with the new pack entry.

        Returns
        -------
        dict with keys: "path", "sha256", "size_bytes", "n_files".
        """
        import hashlib
        import tarfile as _tarfile

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        archive_name = f"{pack_name}.tar.gz"
        archive_path = output_dir / archive_name

        n_files = 0
        with _tarfile.open(archive_path, "w:gz") as tar:
            for raw_path in file_paths:
                path = Path(raw_path)
                if not path.exists():
                    continue
                # Determine the archive member name: preserve the subdir
                # structure relative to the user cache dir
                from manifold_index.core.kernel_cache import _user_cache_dir
                cache_root = _user_cache_dir()
                try:
                    arcname = str(path.relative_to(cache_root))
                except ValueError:
                    arcname = path.name
                tar.add(path, arcname=arcname)
                n_files += 1

        # SHA-256
        h = hashlib.sha256()
        with open(archive_path, "rb") as f:
            for chunk in iter(lambda: f.read(256 * 1024), b""):
                h.update(chunk)
        sha256 = h.hexdigest()
        size_bytes = archive_path.stat().st_size

        result = {
            "path": archive_path,
            "sha256": sha256,
            "size_bytes": size_bytes,
            "n_files": n_files,
            "pack_name": pack_name,
            "release_tag": release_tag,
        }

        if update_registry:
            _update_registry_entry(
                pack_name=pack_name,
                release_tag=release_tag,
                filename=archive_name,
                sha256=sha256,
                size_bytes=size_bytes,
            )
            result["registry_updated"] = True

        return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _grid_product(values: list, r: int) -> list[tuple]:
    """Cartesian product of *values* repeated *r* times.

    For r=1 returns [(v,) for v in values].
    For r=2 returns [(a, b) for a in values for b in values], etc.
    """
    if r == 0:
        return [()]
    if r == 1:
        return [(v,) for v in values]
    sub = _grid_product(values, r - 1)
    return [(v,) + s for v in values for s in sub]


def _update_registry_entry(
    pack_name: str,
    release_tag: str,
    filename: str,
    sha256: str,
    size_bytes: int,
) -> None:
    """Add or update an entry in the bundled data_packs.json."""
    import json
    from manifold_index.core.data_packs import _bundled_registry_path

    registry_path = _bundled_registry_path()
    if not registry_path.exists():
        return

    try:
        with open(registry_path, "r") as f:
            data = json.load(f)
    except Exception:
        return

    packs = data.get("packs", [])
    # Find existing entry with same id or filename
    existing_idx = None
    for i, p in enumerate(packs):
        if p.get("id") == pack_name or p.get("filename") == filename:
            existing_idx = i
            break

    new_entry = {
        "id": pack_name,
        "name": pack_name,
        "description": f"Auto-generated pack: {pack_name}",
        "filename": filename,
        "release_tag": release_tag,
        "size_bytes": size_bytes,
        "sha256": sha256,
        "target_subdir": _infer_target_subdir(filename),
        "category": _infer_category(filename),
    }

    if existing_idx is not None:
        packs[existing_idx] = new_entry
    else:
        packs.append(new_entry)
    data["packs"] = packs

    try:
        with open(registry_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass  # registry update is best-effort


def _infer_target_subdir(filename: str) -> str:
    if "kernel" in filename.lower():
        return "kernel_cache"
    if "iref" in filename.lower():
        return "iref_cache"
    if "nc" in filename.lower():
        return "nc_cycle_cache"
    return "kernel_cache"


def _infer_category(filename: str) -> str:
    if "kernel" in filename.lower():
        return "kernels"
    if "iref" in filename.lower():
        return "iref"
    if "nc" in filename.lower():
        return "nc"
    return "other"
