"""
utils/cache_export.py — Export iref and kernel cache data packs to external formats.

Provides functions to read `.pkl.gz` cache files and write them as
Mathematica `.m` or JSON `.json` files for use outside the app.
"""

from __future__ import annotations

import gzip
import json
import os
import pickle
import sys
import zipfile
from fractions import Fraction
from pathlib import Path


# ── Cache directory helpers ───────────────────────────────────────────

def default_cache_dir() -> Path:
    """Return the default ManifoldIndex cache directory."""
    env = os.environ.get("MANIFOLD_INDEX_CACHE_DIR")
    if env:
        return Path(env)
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "manifold-index"
    elif sys.platform == "win32":
        base = Path(os.environ.get(
            "LOCALAPPDATA",
            str(Path.home() / "AppData" / "Local"),
        ))
        return base / "manifold-index"
    else:
        base = Path(os.environ.get(
            "XDG_CACHE_HOME",
            str(Path.home() / ".cache"),
        ))
        return base / "manifold-index"


def iref_cache_dir() -> Path:
    return default_cache_dir() / "iref_cache"


def kernel_cache_dir() -> Path:
    return default_cache_dir() / "kernel_cache"


# ── Formatting helpers ────────────────────────────────────────────────

def _frac_to_str(v) -> str:
    f = Fraction(v).limit_denominator(10000)
    if f.denominator == 1:
        return str(int(f))
    return f"{f.numerator}/{f.denominator}"


def _frac_to_mathematica(v) -> str:
    f = Fraction(v).limit_denominator(10000)
    if f.denominator == 1:
        return str(int(f))
    return f"({f.numerator}/{f.denominator})"


# ── File loaders ──────────────────────────────────────────────────────

def load_iref_file(path: Path) -> dict:
    """Load an iref cache `.pkl.gz` file and return its payload dict."""
    with gzip.open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or "entries" not in data:
        raise ValueError(f"Not a valid iref cache file: {path}")
    return data


def load_kernel_file(path: Path):
    """Load a kernel cache `.pkl.gz` file and return the KernelTable."""
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def find_iref_file(manifold_name: str, cache_dir: Path | None = None) -> Path | None:
    """Find the iref cache file for a named manifold (if it exists)."""
    d = cache_dir or iref_cache_dir()
    if not d.exists():
        return None
    safe = manifold_name.replace("/", "_").replace(" ", "_")
    for f in d.glob(f"iref_{safe}_*.pkl.gz"):
        return f  # first match
    return None


def list_iref_files(cache_dir: Path | None = None) -> list[Path]:
    """Return all iref cache files in the given directory."""
    d = cache_dir or iref_cache_dir()
    if not d.exists():
        return []
    return sorted(d.glob("iref_*.pkl.gz"))


def list_kernel_files(cache_dir: Path | None = None) -> list[Path]:
    """Return all kernel cache files in the given directory."""
    d = cache_dir or kernel_cache_dir()
    if not d.exists():
        return []
    return sorted(d.glob("kernel_*.pkl.gz"))


# ── Iref → Mathematica ───────────────────────────────────────────────

def _iref_entry_to_mathematica(
    key: tuple,
    result: dict,
    num_hard: int,
) -> str:
    """Format one I^ref entry as a Mathematica expression."""
    if not result:
        return "0"

    terms: list[str] = []
    for k in sorted(result.keys()):
        c = result[k]
        if c == 0:
            continue
        qq = k[0]
        etas = k[1:]

        if qq == 0:
            q_part = ""
        elif qq == 2:
            q_part = "q"
        elif qq % 2 == 0:
            q_part = f"q^{qq // 2}"
        else:
            q_part = f"q^({qq}/2)"

        eta_parts: list[str] = []
        for a in range(min(num_hard, len(etas))):
            eta_2 = etas[a]
            if eta_2 != 0:
                if eta_2 == 2:
                    eta_parts.append(f"eta[{a}]")
                elif eta_2 == -2:
                    eta_parts.append(f"eta[{a}]^(-1)")
                elif eta_2 % 2 == 0:
                    eta_parts.append(f"eta[{a}]^{eta_2 // 2}")
                else:
                    eta_parts.append(f"eta[{a}]^({eta_2}/2)")

        parts_str = " ".join([p for p in [q_part] + eta_parts if p])
        if not parts_str:
            parts_str = "1"

        c_frac = Fraction(c).limit_denominator(10000)
        if c_frac.denominator == 1:
            c_str = str(int(c_frac))
        else:
            c_str = f"({c_frac.numerator}/{c_frac.denominator})"

        if c_str == "1" and parts_str != "1":
            terms.append(parts_str)
        elif c_str == "-1" and parts_str != "1":
            terms.append(f"-{parts_str}")
        elif parts_str == "1":
            terms.append(c_str)
        else:
            terms.append(f"{c_str}*{parts_str}")

    if not terms:
        return "0"
    result_str = terms[0]
    for t in terms[1:]:
        if t.startswith("-"):
            result_str += f" {t}"
        else:
            result_str += f" + {t}"
    return result_str


def iref_mathematica_content(data: dict) -> tuple[str, int]:
    """Generate Mathematica source for an iref cache.

    Returns ``(content_string, entry_count)``.
    """
    name = data.get("manifold_name", "unknown")
    n_tet = data.get("n_tetrahedra", "?")
    n_cusps = data.get("n_cusps", "?")
    num_hard = data.get("num_hard", 0)
    nz_hash = data.get("nz_hash", "")
    entries = data.get("entries", {})
    grid = data.get("grid_params")

    L: list[str] = []
    L.append(f'(* Iref cache export for {name} *)')
    L.append(f'(* Tetrahedra: {n_tet}, Cusps: {n_cusps}, '
             f'Hard edges: {num_hard}, NZ hash: {nz_hash} *)')
    if grid:
        L.append(f'(* Grid: m_max={grid.get("m_max")}, '
                 f'e_max={grid.get("e_max")}, '
                 f'qq_order={grid.get("qq_order")} *)')
    L.append(f'(* Entries: {len(entries)} *)')
    L.append("")
    L.append(f'manifoldName = "{name}";')
    L.append(f'nTetrahedra = {n_tet};')
    L.append(f'nCusps = {n_cusps};')
    L.append(f'numHard = {num_hard};')
    L.append("")

    by_qq: dict[int, list] = {}
    for key, result in entries.items():
        m_tuple = key[0]
        e_tuple = key[1]
        qq = key[2]
        by_qq.setdefault(qq, []).append((m_tuple, e_tuple, result))

    for qq in sorted(by_qq):
        L.append(f'(* qq_order_half = {qq} *)')
        for m_tuple, e_tuple, result in by_qq[qq]:
            m_str = "{" + ", ".join(str(m) for m in m_tuple) + "}"
            e_str = "{" + ", ".join(_frac_to_mathematica(e) for e in e_tuple) + "}"
            expr = _iref_entry_to_mathematica(
                (m_tuple, e_tuple, qq), result, num_hard,
            )
            L.append(f'Iref["{name}", {m_str}, {e_str}, {qq}] = {expr};')
        L.append("")

    return "\n".join(L), len(entries)


def export_iref_mathematica(data: dict, out_path: Path) -> int:
    """Write an iref cache to a Mathematica `.m` file.

    Returns the number of entries written.
    """
    content, n = iref_mathematica_content(data)
    out_path.write_text(content, encoding="utf-8")
    return n


# ── Iref → JSON ──────────────────────────────────────────────────────

def iref_json_content(data: dict) -> tuple[str, int]:
    """Generate JSON string for an iref cache.

    Returns ``(json_string, entry_count)``.
    """
    name = data.get("manifold_name", "unknown")
    entries = data.get("entries", {})

    json_entries = []
    for key, result in entries.items():
        m_tuple = key[0]
        e_tuple = key[1]
        qq = key[2]

        series = {}
        for k, v in result.items():
            key_str = ",".join(str(x) for x in k)
            v_frac = Fraction(v).limit_denominator(10000)
            series[key_str] = (
                int(v_frac) if v_frac.denominator == 1
                else {"n": v_frac.numerator, "d": v_frac.denominator}
            )

        json_entries.append({
            "m": list(m_tuple),
            "e": [_frac_to_str(e) for e in e_tuple],
            "qq": qq,
            "series": series,
        })

    out = {
        "manifold_name": name,
        "n_tetrahedra": data.get("n_tetrahedra"),
        "n_cusps": data.get("n_cusps"),
        "num_hard": data.get("num_hard"),
        "nz_hash": data.get("nz_hash"),
        "grid_params": data.get("grid_params"),
        "entries": json_entries,
    }

    return json.dumps(out, indent=2), len(entries)


def export_iref_json(data: dict, out_path: Path) -> int:
    """Write an iref cache to a JSON file.  Returns entry count."""
    content, n = iref_json_content(data)
    out_path.write_text(content, encoding="utf-8")
    return n


# ── Kernel → Mathematica ─────────────────────────────────────────────

def kernel_mathematica_content(kt) -> tuple[str, int]:
    """Generate Mathematica source for a KernelTable.

    Returns ``(content_string, entry_count)``.
    """
    L: list[str] = []
    L.append(f'(* Dehn filling kernel: P={kt.P}, Q={kt.Q}, '
             f'qq_order={kt.qq_order} *)')
    L.append(f'(* HJ: [{", ".join(str(k) for k in kt.hj_ks)}] *)')
    L.append(f'(* m_scan={kt.m_scan}, e_scan={kt.e_scan}, '
             f'eta_order={kt.eta_order} *)')
    L.append(f'(* Compute time: {kt.compute_time_s:.1f}s *)')
    L.append("")
    L.append(f'kernelP = {kt.P};')
    L.append(f'kernelQ = {kt.Q};')
    L.append(f'qqOrder = {kt.qq_order};')
    L.append(f'hjCF = {{{", ".join(str(k) for k in kt.hj_ks)}}};')
    L.append("")

    n_entries = 0
    for (m, e), series in sorted(kt.table.items()):
        if not series:
            continue
        n_entries += 1
        e_str = _frac_to_mathematica(e)
        terms: list[str] = []
        for (qq_k, eta_cusp), c in sorted(series.items()):
            if c == 0:
                continue
            c_frac = Fraction(c).limit_denominator(10000)
            if c_frac.denominator == 1:
                c_str = str(int(c_frac))
            else:
                c_str = f"({c_frac.numerator}/{c_frac.denominator})"

            parts: list[str] = []
            if qq_k != 0:
                if qq_k == 2:
                    parts.append("q")
                elif qq_k % 2 == 0:
                    parts.append(f"q^{qq_k // 2}")
                else:
                    parts.append(f"q^({qq_k}/2)")
            if eta_cusp != 0:
                if eta_cusp == 1:
                    parts.append("v")
                elif eta_cusp == -1:
                    parts.append("v^(-1)")
                else:
                    parts.append(f"v^{eta_cusp}")

            base = " ".join(parts) if parts else "1"
            if c_str == "1" and base != "1":
                terms.append(base)
            elif c_str == "-1" and base != "1":
                terms.append(f"-{base}")
            elif base == "1":
                terms.append(c_str)
            else:
                terms.append(f"{c_str}*{base}")

        if not terms:
            continue
        expr = terms[0]
        for t in terms[1:]:
            if t.startswith("-"):
                expr += f" {t}"
            else:
                expr += f" + {t}"
        L.append(f'K[{m}, {e_str}] = {expr};')

    L.append("")
    L.append(f'(* Total non-zero entries: {n_entries} *)')

    return "\n".join(L), n_entries


def export_kernel_mathematica(kt, out_path: Path) -> int:
    """Write a KernelTable to a Mathematica `.m` file.  Returns entry count."""
    content, n = kernel_mathematica_content(kt)
    out_path.write_text(content, encoding="utf-8")
    return n


# ── Kernel → JSON ────────────────────────────────────────────────────

def kernel_json_content(kt) -> tuple[str, int]:
    """Generate JSON string for a KernelTable.

    Returns ``(json_string, entry_count)``.
    """
    entries = []
    for (m, e), series in sorted(kt.table.items()):
        if not series:
            continue
        terms = {}
        for (qq_k, eta_cusp), c in sorted(series.items()):
            if c == 0:
                continue
            c_frac = Fraction(c).limit_denominator(10000)
            key_str = f"{qq_k},{eta_cusp}"
            terms[key_str] = (
                int(c_frac) if c_frac.denominator == 1
                else {"n": c_frac.numerator, "d": c_frac.denominator}
            )
        entries.append({
            "m": m,
            "e": _frac_to_str(e),
            "series": terms,
        })

    out = {
        "P": kt.P,
        "Q": kt.Q,
        "qq_order": kt.qq_order,
        "hj_ks": kt.hj_ks,
        "m_scan": kt.m_scan,
        "e_scan": kt.e_scan,
        "eta_order": kt.eta_order,
        "entries": entries,
    }

    return json.dumps(out, indent=2), len(entries)


def export_kernel_json(kt, out_path: Path) -> int:
    """Write a KernelTable to a JSON file.  Returns entry count."""
    content, n = kernel_json_content(kt)
    out_path.write_text(content, encoding="utf-8")
    return n


# ── Zip helpers ───────────────────────────────────────────────────────

def export_iref_zip(
    data: dict, zip_path: Path, fmt: str = "mathematica",
) -> int:
    """Write an iref cache into a ``.zip`` archive.

    *fmt* is ``"mathematica"`` or ``"json"``.
    Returns the number of entries written.
    """
    name = data.get("manifold_name", "unknown")
    safe = name.replace("/", "_").replace(" ", "_")

    if fmt == "json":
        content, n = iref_json_content(data)
        arcname = f"iref_{safe}.json"
    else:
        content, n = iref_mathematica_content(data)
        arcname = f"iref_{safe}.m"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(arcname, content)
    return n


def export_kernels_zip(
    kernel_files: list[Path],
    zip_path: Path,
    fmt: str = "mathematica",
) -> tuple[int, int]:
    """Write multiple kernel caches into a single ``.zip`` archive.

    Returns ``(exported_count, error_count)``.
    """
    exported = 0
    errors = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in kernel_files:
            try:
                kt = load_kernel_file(f)
                stem = f.stem.replace(".pkl", "")
                if fmt == "json":
                    content, _ = kernel_json_content(kt)
                    arcname = f"{stem}.json"
                else:
                    content, _ = kernel_mathematica_content(kt)
                    arcname = f"{stem}.m"
                zf.writestr(arcname, content)
                exported += 1
            except Exception:
                errors += 1
    return exported, errors
