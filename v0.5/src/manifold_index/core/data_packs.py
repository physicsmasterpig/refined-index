"""
core/data_packs.py — Data pack registry, download, and extraction.

Manages optional downloadable data packs (pre-computed kernels, refined
indices) that accelerate the app.  Packs are hosted as GitHub release
assets and extracted into the user cache directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import ssl
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from manifold_index.core.kernel_cache import _user_cache_dir


# ── SSL context (macOS PyInstaller bundles lack default CA certs) ────
def _ssl_context() -> ssl.SSLContext:
    """Return an SSL context that works in frozen macOS apps.

    Tries (in order):
      1. Default context with system cert store
      2. certifi bundle (if installed)
      3. Unverified context (last resort — still SHA-256 verified)
    """
    # Try default first
    try:
        ctx = ssl.create_default_context()
        # Quick test: can we load the default certs?
        if ctx.get_ca_certs():
            return ctx
    except Exception:
        pass

    # Try certifi
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        pass

    # Try macOS system certs via subprocess
    try:
        import subprocess
        import tempfile
        result = subprocess.run(
            ["security", "find-certificate", "-a", "-p",
             "/System/Library/Keychains/SystemRootCertificates.keychain"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and "BEGIN CERTIFICATE" in result.stdout:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".pem", delete=False) as f:
                f.write(result.stdout)
                pem_path = f.name
            ctx = ssl.create_default_context(cafile=pem_path)
            os.unlink(pem_path)
            return ctx
    except Exception:
        pass

    # Last resort: unverified (download is still SHA-256 checked)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


# ── Registry location ────────────────────────────────────────────────
def _bundled_registry_path() -> Path:
    """Locate data_packs.json in both dev and frozen (PyInstaller) environments."""
    import sys
    if getattr(sys, "frozen", False):
        # PyInstaller bundle: data is under _MEIPASS/manifold_index/data/
        base = Path(sys._MEIPASS) / "manifold_index" / "data"
    else:
        base = Path(__file__).resolve().parent.parent / "data"
    return base / "data_packs.json"

_REMOTE_REGISTRY = (
    "https://raw.githubusercontent.com/physicsmasterpig/refined-index/"
    "master/v0.4/src/manifold_index/data/data_packs.json"
)


# ── Data classes ─────────────────────────────────────────────────────

@dataclass
class PackInfo:
    """Metadata for a single data pack."""

    id: str
    name: str
    description: str
    filename: str
    release_tag: str
    size_bytes: int
    sha256: str
    target_subdir: str          # e.g. "kernel_cache" or "iref_cache"
    category: str               # "kernels" or "iref"

    # Runtime state (not serialised)
    installed: bool = field(default=False, repr=False)
    installed_files: int = field(default=0, repr=False)

    @property
    def size_human(self) -> str:
        """Human-readable file size."""
        if self.size_bytes == 0:
            return "—"
        if self.size_bytes < 1024:
            return f"{self.size_bytes} B"
        if self.size_bytes < 1024 ** 2:
            return f"{self.size_bytes / 1024:.0f} KB"
        if self.size_bytes < 1024 ** 3:
            return f"{self.size_bytes / 1024**2:.1f} MB"
        return f"{self.size_bytes / 1024**3:.2f} GB"


@dataclass
class PackRegistry:
    """Collection of available data packs."""

    version: int
    base_url: str
    packs: list[PackInfo]

    def get(self, pack_id: str) -> PackInfo | None:
        for p in self.packs:
            if p.id == pack_id:
                return p
        return None


# ── Load registry ────────────────────────────────────────────────────

def _parse_registry(data: dict) -> PackRegistry:
    """Parse a registry dict into typed objects."""
    packs = []
    for entry in data.get("packs", []):
        packs.append(PackInfo(
            id=entry["id"],
            name=entry["name"],
            description=entry["description"],
            filename=entry["filename"],
            release_tag=entry["release_tag"],
            size_bytes=entry.get("size_bytes", 0),
            sha256=entry.get("sha256", ""),
            target_subdir=entry["target_subdir"],
            category=entry.get("category", ""),
        ))
    return PackRegistry(
        version=data.get("registry_version", 1),
        base_url=data.get("base_url", ""),
        packs=packs,
    )


def load_registry(use_remote: bool = False) -> PackRegistry:
    """Load the pack registry (bundled or remote).

    Parameters
    ----------
    use_remote : bool
        If True, try fetching the latest registry from GitHub first.
        Falls back to bundled on failure.
    """
    if use_remote:
        try:
            req = urllib.request.Request(_REMOTE_REGISTRY, headers={
                "User-Agent": "manifold-index",
            })
            with urllib.request.urlopen(req, timeout=10, context=_ssl_context()) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return _parse_registry(data)
        except Exception:
            pass  # fall through to bundled

    with open(_bundled_registry_path(), "r") as f:
        data = json.load(f)
    return _parse_registry(data)


# ── Installation check ───────────────────────────────────────────────

def _marker_path(pack_id: str) -> Path:
    """Path to the marker file that records a pack is installed."""
    return _user_cache_dir() / ".packs" / f"{pack_id}.installed"


def check_installed(registry: PackRegistry) -> None:
    """Update ``installed`` flags on all packs by checking marker files."""
    for pack in registry.packs:
        marker = _marker_path(pack.id)
        pack.installed = marker.exists()
        if pack.installed:
            try:
                meta = json.loads(marker.read_text())
                pack.installed_files = meta.get("n_files", 0)
            except Exception:
                pack.installed_files = 0


def _write_marker(pack: PackInfo, n_files: int, file_list: list[str] | None = None) -> None:
    """Write an installation marker after successful extraction."""
    marker = _marker_path(pack.id)
    marker.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "id": pack.id,
        "filename": pack.filename,
        "sha256": pack.sha256,
        "n_files": n_files,
        "files": file_list or [],
    }
    marker.write_text(json.dumps(meta, indent=2))


# ── Download URL ─────────────────────────────────────────────────────

def download_url(registry: PackRegistry, pack: PackInfo) -> str:
    """Construct the full download URL for a pack."""
    return f"{registry.base_url}/{pack.release_tag}/{pack.filename}"


# ── Download + extract ───────────────────────────────────────────────

def download_and_install(
    registry: PackRegistry,
    pack: PackInfo,
    progress_fn: Callable[[int, int], None] | None = None,
    status_fn: Callable[[str], None] | None = None,
) -> int:
    """Download a data pack and extract into the user cache.

    Parameters
    ----------
    registry : PackRegistry
    pack : PackInfo
    progress_fn : callable(received_bytes, total_bytes)
        Called periodically during download.
    status_fn : callable(message)
        Called with status strings.

    Returns
    -------
    int
        Number of files extracted.

    Raises
    ------
    Exception
        On network or extraction errors.
    """
    cache_dir = _user_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    url = download_url(registry, pack)
    if status_fn:
        status_fn(f"Downloading {pack.name}…")

    # Download to a temp file
    tmp_path = cache_dir / f".{pack.filename}.tmp"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "manifold-index"})
        resp = urllib.request.urlopen(req, timeout=300, context=_ssl_context())
        total = int(resp.headers.get("Content-Length", 0))

        received = 0
        hasher = hashlib.sha256()
        with open(tmp_path, "wb") as f:
            while True:
                chunk = resp.read(256 * 1024)  # 256 KB chunks
                if not chunk:
                    break
                f.write(chunk)
                hasher.update(chunk)
                received += len(chunk)
                if progress_fn:
                    progress_fn(received, total)

        # Verify SHA-256 if provided
        if pack.sha256:
            actual = hasher.hexdigest()
            if actual != pack.sha256:
                raise ValueError(
                    f"SHA-256 mismatch for {pack.filename}: "
                    f"expected {pack.sha256[:16]}…, got {actual[:16]}…"
                )

        # Extract
        if status_fn:
            status_fn(f"Extracting {pack.name}…")

        n_files = 0
        extracted_files: list[str] = []
        with tarfile.open(tmp_path, "r:gz") as tar:
            # Security: only extract into expected subdirectory
            for member in tar.getmembers():
                # Normalise and check for path traversal
                name = os.path.normpath(member.name)
                if name.startswith("..") or name.startswith("/"):
                    continue
                # Only extract files (skip directories — we mkdir ourselves)
                if not member.isfile():
                    continue
                # Extract into cache_dir (tarball paths like "kernel_cache/file.pkl.gz")
                dest = cache_dir / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src:
                    dest.write_bytes(src.read())
                extracted_files.append(name)
                n_files += 1

        # Write marker (with file list for clean uninstall)
        _write_marker(pack, n_files, extracted_files)
        pack.installed = True
        pack.installed_files = n_files

        if status_fn:
            status_fn(f"✓ {pack.name} installed ({n_files} files)")
        return n_files

    finally:
        # Clean up temp file
        if tmp_path.exists():
            tmp_path.unlink()


# ── Uninstall ────────────────────────────────────────────────────────

def uninstall_pack(pack: PackInfo) -> int:
    """Remove installed files for a pack.

    Uses the file list stored in the installation marker for precise
    removal.  Only removes files that were extracted by this pack.

    Returns number of files removed.
    """
    cache_dir = _user_cache_dir()
    removed = 0

    # Read the marker to get the exact list of installed files
    marker = _marker_path(pack.id)
    file_list: list[str] = []
    if marker.exists():
        try:
            meta = json.loads(marker.read_text())
            file_list = meta.get("files", [])
        except Exception:
            pass

    if file_list:
        # Precise removal — only files belonging to this pack
        for rel_path in file_list:
            f = cache_dir / rel_path
            if f.exists():
                f.unlink()
                removed += 1
    else:
        # Fallback: no file list in marker (old marker format).
        # Remove all .pkl.gz in target_subdir as best effort.
        target = cache_dir / pack.target_subdir
        if target.exists():
            for f in target.glob("*.pkl.gz"):
                f.unlink()
                removed += 1

    # Remove marker
    if marker.exists():
        marker.unlink()

    pack.installed = False
    pack.installed_files = 0
    return removed
