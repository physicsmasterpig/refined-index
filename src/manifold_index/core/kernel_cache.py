"""
core/kernel_cache.py — Pre-computed Dehn filling kernel tables.

The refined Dehn filling kernel K^ref(P/Q; m, e; η_cusp) is **manifold-
independent**: it depends only on the slope (P, Q) and the charge
variables (m, e).  The manifold enters only through the refined 3D index
I^ref(m, e; η_hard), which is then convolved with the kernel:

    Î^ref_{P/Q}(η_hard, η_cusp) = Σ_{m,e}  I^ref(m,e; η_hard) · K^ref(P/Q; m,e; η_cusp)

Pre-computing the kernel for a slope and storing it on disk turns a
~10 minute IS-chain computation into a sub-second lookup + summation.

Storage
-------
Tables are stored as compressed pickle files in ``data/kernel_cache/``:

    kernel_P{P}_Q{Q}_qq{qq_order}.pkl.gz

Each file contains a ``KernelTable`` dict with metadata and the sparse
table mapping ``(m, e) → QEtaSeries``.
"""

from __future__ import annotations

import gzip
import hashlib
import multiprocessing
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from fractions import Fraction
from math import gcd as _gcd
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Type aliases (must match refined_dehn_filling.py)
# ---------------------------------------------------------------------------
QEtaSeries = dict[tuple[int, int], Fraction]
MultiEtaSeries = dict[tuple[int, ...], Fraction]

# Default cache directories (sibling to src/)
_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "kernel_cache"
_DEFAULT_IREF_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "iref_cache"

# Sentinel for infeasible degree bounds (any value > practical qq_limit)
_INF_DEG = 10**9


# ---------------------------------------------------------------------------
# Degree-bound helpers  (numpy-vectorised, pure integer arithmetic)
# ---------------------------------------------------------------------------

def _tet_degree_x2(m: int, e: int) -> int:
    """Return ``2 × tet_degree(m, e)`` as pure int (scalar version)."""
    p_m = max(0, m)
    p_me = max(0, m + e)
    p_nm = max(0, -m)
    p_e = max(0, e)
    p_ne = max(0, -e)
    p_nem = max(0, -e - m)
    return p_m * p_me + p_nm * p_e + p_ne * p_nem + 2 * max(0, m, -e)


def _tdeg_arr(m: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Fully-vectorised ``2 × tet_degree`` – both *m* and *e* arrays."""
    me = m + e
    ne = -e
    nem = ne - m
    return (
        np.maximum(0, m) * np.maximum(0, me)
        + np.maximum(0, -m) * np.maximum(0, e)
        + np.maximum(0, ne) * np.maximum(0, nem)
        + 2 * np.maximum(np.maximum(0, m), ne)
    )


def _is_kernel_min_degree_x2(
    m_src: int, e_in_half: int, m_tgt: int, e_tgt_half: int,
) -> int:
    """Lower bound on 2× minimum qq-power of IS kernel output (scalar).

    Used only for the ℓ ≥ 3 backward-reachability pass where the number
    of evaluations is small.
    """
    best = _INF_DEG
    for shift_half in (-2, 0, 2):
        eih = e_in_half + shift_half
        # Integrality checks
        if (eih + m_tgt) % 2 != 0:
            continue
        if (e_tgt_half + m_src) % 2 != 0:
            continue
        if (eih - m_tgt) % 2 != 0:
            continue
        B_num = eih + e_tgt_half + m_src - m_tgt
        if B_num % 2 != 0:
            continue
        p = (m_src + m_tgt) % 2
        if (eih + m_src - p) % 2 != 0:
            continue
        if (e_tgt_half - m_tgt - p) % 2 != 0:
            continue

        m_a1 = -(eih + m_tgt) // 2
        m_a3 = -(e_tgt_half + m_src) // 2
        e3b = -m_a3
        e4b = (eih - m_tgt) // 2
        B = B_num // 2
        ea1 = (eih + m_src - p) // 2
        ea2 = (e_tgt_half - m_tgt - p) // 2

        # min d34(t) near t=0 and t=t4_opt
        t4 = m_a3 - e4b
        d34 = _INF_DEG
        for t in range(min(0, t4) - 4, max(0, t4) + 5):
            v = _tet_degree_x2(m_a3, e3b + t) + _tet_degree_x2(-m_a3, e4b + t)
            if v < d34:
                d34 = v

        # min g(u) near u1_opt and u2_opt
        u1 = -m_a1 - ea1
        u2 = m_a1 - ea2
        gmin = _INF_DEG
        for u in range(min(u1, u2) - 6, max(u1, u2) + 5):
            v = _tet_degree_x2(m_a1, ea1 + u) + _tet_degree_x2(-m_a1, ea2 + u) + 4 * u
            if v < gmin:
                gmin = v

        total = d34 + gmin - 2 * p + 2 * B
        if total < best:
            best = total
    return best


def _degree_feasible_row(
    m0: int,
    k1: int,
    e_half_arr: np.ndarray,
    mt: np.ndarray,
    et: np.ndarray,
    qq_limit_x2: int,
) -> np.ndarray:
    """Return boolean array (len(e_half_arr),) of feasibility per e_half.

    Fully numpy-vectorised over *all* (e_half, target) pairs at once.
    Shape of intermediates: (E, T) where E = len(e_half_arr), T = len(mt).
    Uses int32 arithmetic — all values comfortably fit (|val| < 2×10⁹).
    """
    _I32 = np.int32
    E = len(e_half_arr)
    T = len(mt)

    # e_in for each e_half: shape (E,)
    e_in = -(e_half_arr + _I32(k1) * _I32(m0))
    # Broadcast shapes: eih (E,1), mt2 (1,T), et2 (1,T)
    mt2 = mt[np.newaxis, :]                   # (1, T)
    et2 = et[np.newaxis, :]                   # (1, T)

    # Track per-e_half feasibility across shifts (avoids keeping full
    # (E, T) array when most e_half values become feasible early).
    feasible = np.zeros(E, dtype=bool)
    m0_32 = _I32(m0)
    _INF = _I32(_INF_DEG)

    for shift in _I32([-2, 0, 2]):
        # Skip shift if all e_half values are already feasible
        if feasible.all():
            break

        eih = e_in[:, np.newaxis] + shift      # (E, 1) broadcast → (E, T)

        # ---- Integrality masks ----
        valid = (eih + mt2) % 2 == 0
        eih_minus_m = eih - mt2
        valid &= eih_minus_m % 2 == 0
        B_num = eih + et2 + m0_32 - mt2
        valid &= B_num % 2 == 0
        p = (m0_32 + mt2) % 2                           # (1, T)
        valid &= (eih + m0_32 - p) % 2 == 0
        valid &= (et2 - mt2 - p) % 2 == 0
        # (et2 + m0) % 2 == 0 already guaranteed by caller parity filter

        if not np.any(valid):
            continue

        # ---- Parameters (compute everywhere; mask applied later) ----
        m_a1 = -(eih + mt2) // 2                         # (E, T)
        m_a3 = -(et2 + m0_32) // 2                      # (1, T)
        e3b  = -m_a3                                      # (1, T)
        e4b  = eih_minus_m // 2                           # (E, T)
        B    = B_num // 2                                 # (E, T)
        ea1  = (eih + m0_32 - p) // 2                    # (E, T)
        ea2  = (et2 - mt2 - p) // 2                       # (1, T)

        # ---- min d34(t): scan near t=0 and near t4_opt ----
        t4_opt = m_a3 - e4b                               # (E, T)
        min_d34 = np.full((E, T), _INF, dtype=_I32)
        for t_off in range(-4, 5):
            # near t = 0
            d = _tdeg_arr(m_a3, e3b + t_off) + _tdeg_arr(-m_a3, e4b + t_off)
            np.minimum(min_d34, d, out=min_d34)
            # near t = t4_opt
            t = t4_opt + t_off
            d = _tdeg_arr(m_a3, e3b + t) + _tdeg_arr(-m_a3, e4b + t)
            np.minimum(min_d34, d, out=min_d34)

        # ---- min g(u): scan near u1_opt and u2_opt ----
        u1_opt = -m_a1 - ea1                              # (E, T)
        u2_opt =  m_a1 - ea2                              # (E, T)
        min_g = np.full((E, T), _INF, dtype=_I32)
        for u_off in range(-6, 5):
            u = u1_opt + u_off
            g = _tdeg_arr(m_a1, ea1 + u) + _tdeg_arr(-m_a1, ea2 + u) + 4 * u
            np.minimum(min_g, g, out=min_g)
            u = u2_opt + u_off
            g = _tdeg_arr(m_a1, ea1 + u) + _tdeg_arr(-m_a1, ea2 + u) + 4 * u
            np.minimum(min_g, g, out=min_g)

        # ---- Combine ----
        total = min_d34 + min_g - 2 * p + 2 * B
        total = np.where(valid, total, _INF)
        # Update feasibility: for each e_half, any target ≤ limit?
        feasible |= np.any(total <= qq_limit_x2, axis=1)

    return feasible


def _compute_degree_bounds(
    hj_ks: list[int],
    qq_internal: int,
    m_scan: int,
    e_scan: int,
    m_step: int,
    m_start: int,
    m1_range: int,
    final_term_info: dict,
    status_fn=None,
) -> tuple[dict[int, tuple[int, int]], list[int]]:
    """Compute per-m₀ e-bounds via degree analysis.

    Returns ``(e_bounds, target_m_values)`` in the same format as the
    probe-and-scale Phase 2 output, but **provably correct**: every
    (m₀, e₀) with a non-zero kernel entry is guaranteed to be inside
    the returned bounds.

    Uses numpy-vectorised degree evaluation over all (e_half, target)
    pairs per m₀ row.  Backward reachability prunes the target set
    for ℓ ≥ 3 IS-chain steps.
    """
    from manifold_index.core.refined_dehn_filling import _enumerate_slope1_all

    ell = len(hj_ks)
    qq_limit_x2 = 2 * qq_internal

    # --- Build the final target set as (m, e_half) pairs ---
    final_targets: set[tuple[int, int]] = set()
    for (m_t, e_t) in final_term_info:
        e_t_half = int(2 * e_t)
        final_targets.add((m_t, e_t_half))

    # --- Backward reachability for ℓ ≥ 3 ---
    reachable: set[tuple[int, int]] = final_targets

    for step_i in range(ell - 2, 0, -1):
        k_curr = hj_ks[step_i]
        k_next = hj_ks[step_i + 1]
        src_terms = _enumerate_slope1_all(k_next, m1_range)
        new_reachable: set[tuple[int, int]] = set()

        for (m1, e1, _, _) in src_terms:
            e1_half = int(2 * e1)
            e_in_half = -(e1_half + k_curr * m1)
            for (m_tgt, e_tgt_half) in reachable:
                d = _is_kernel_min_degree_x2(m1, e_in_half, m_tgt, e_tgt_half)
                if d <= qq_limit_x2:
                    new_reachable.add((m1, e1_half))
                    break
        if status_fn:
            status_fn(f"[degree]   Backward step {step_i}: "
                       f"{len(new_reachable)} reachable from {len(reachable)} targets")
        reachable = new_reachable

    # --- Forward pass: vectorised per m₀ row ---
    k1 = hj_ks[0]
    target_list = sorted(reachable)

    # Partition targets by e_tgt_half parity
    m_even = np.array([mt for (mt, et) in target_list if et % 2 == 0], dtype=np.int32)
    e_even = np.array([et for (mt, et) in target_list if et % 2 == 0], dtype=np.int32)
    m_odd  = np.array([mt for (mt, et) in target_list if et % 2 != 0], dtype=np.int32)
    e_odd  = np.array([et for (mt, et) in target_list if et % 2 != 0], dtype=np.int32)

    e_half_all = np.arange(-2 * e_scan, 2 * e_scan + 1, dtype=np.int32)
    e_bounds: dict[int, tuple[int, int]] = {}

    for m0 in range(m_start, m_scan + 1, m_step):
        # Select parity-compatible targets: (e_tgt_half + m0) even
        if m0 % 2 == 0:
            mt, et = m_even, e_even
        else:
            mt, et = m_odd, e_odd
        if len(mt) == 0:
            continue

        feasible = _degree_feasible_row(m0, k1, e_half_all, mt, et, qq_limit_x2)
        if not np.any(feasible):
            continue

        idx = np.where(feasible)[0]
        e_bounds[m0] = (int(e_half_all[idx[0]]), int(e_half_all[idx[-1]]))

    target_m_values = sorted(e_bounds.keys())
    return e_bounds, target_m_values


# ---------------------------------------------------------------------------
# KernelTable dataclass
# ---------------------------------------------------------------------------

@dataclass
class KernelTable:
    """A pre-computed Dehn filling kernel for a single slope.

    Attributes
    ----------
    P, Q : int
        Slope (coprime).
    qq_order : int
        The user-facing qq truncation order.
    qq_internal : int
        The inflated internal qq used during pre-computation
        (``qq_order + buffer``).
    eta_order : int
        Maximum |η_cusp| exponent retained.
    hj_ks : list[int]
        Hirzebruch-Jung continued-fraction coefficients.
    table : dict[(int, Fraction), QEtaSeries]
        Sparse kernel.  ``table[(m, e)]`` gives the QEtaSeries
        ``{(qq_shift, eta_cusp): Fraction}`` for that grid point.
        Only non-zero entries are stored.
    m_scan : int
        Grid range: ``m ∈ [-m_scan, m_scan]``.
    e_scan : int
        Grid range: ``e ∈ [-e_scan, e_scan]`` in half-integer steps.
    """
    P: int
    Q: int
    qq_order: int
    qq_internal: int
    eta_order: int
    hj_ks: list[int]
    table: dict[tuple[int, Fraction], QEtaSeries]
    m_scan: int
    e_scan: int
    compute_time_s: float = 0.0

    # Persistent fast representation (survives pickling)
    # lcd: least common denominator of all Fraction coefficients
    # int_grouped: (m,e) → {η_cusp: (min_qq, np.ndarray[int64])}
    #   where the array is a dense kernel vector scaled by lcd,
    #   offset so that array[i] corresponds to qq = i + min_qq.
    _fast_lcd: int = field(default=0, repr=False)
    _fast_grouped: dict | None = field(default=None, repr=False)

    def get_int_grouped(
        self,
    ) -> tuple[int, dict[tuple[int, Fraction], dict[int, tuple[int, np.ndarray]]]]:
        """Return ``(lcd, grouped)`` for numpy-accelerated convolution.

        ``grouped[(m,e)][eta_cusp] = (min_qq, arr)`` where *arr* is a
        dense ``int64`` numpy array with ``arr[i]`` = kernel coefficient
        at ``qq = i + min_qq``, scaled by *lcd*.

        The representation is **built once** (lazily on first call or
        eagerly at save-time via :meth:`ensure_fast_repr`) and cached
        on the instance.  It survives pickling so subsequent loads skip
        the conversion entirely.
        """
        if self._fast_grouped is not None:
            return self._fast_lcd, self._fast_grouped
        self.ensure_fast_repr()
        return self._fast_lcd, self._fast_grouped

    def ensure_fast_repr(self) -> None:
        """Build the fast (lcd, int-grouped-numpy) representation.

        Idempotent — does nothing if already built.
        Called automatically by :meth:`get_int_grouped` and
        explicitly by :func:`save_kernel_table` so the representation
        is persisted to disk.
        """
        if self._fast_grouped is not None:
            return

        # Compute LCD of all Fraction denominators
        denoms: set[int] = set()
        for entry in self.table.values():
            for c in entry.values():
                denoms.add(c.denominator)
        lcd = 1
        for d in denoms:
            lcd = lcd * d // _gcd(lcd, d)

        # Build int-grouped representation with pre-built numpy arrays
        grouped: dict[tuple[int, Fraction], dict[int, tuple[int, np.ndarray]]] = {}
        for me, entry in self.table.items():
            # First pass: group by eta_cusp → {qq_k: int_coeff}
            by_eta_dict: dict[int, dict[int, int]] = {}
            for (qq_k, eta_cusp), c in entry.items():
                by_eta_dict.setdefault(eta_cusp, {})[qq_k] = int(lcd * c)
            # Second pass: convert each eta_cusp group to (offset, dense array)
            by_eta: dict[int, tuple[int, np.ndarray]] = {}
            for eta_cusp, qq_map in by_eta_dict.items():
                min_qk = min(qq_map)
                max_qk = max(qq_map)
                arr = np.zeros(max_qk - min_qk + 1, dtype=np.int64)
                for qq_k, c_k in qq_map.items():
                    arr[qq_k - min_qk] = c_k
                by_eta[eta_cusp] = (min_qk, arr)
            grouped[me] = by_eta

        self._fast_lcd = lcd
        self._fast_grouped = grouped


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def _kernel_filename(P: int, Q: int, qq_order: int) -> str:
    """Canonical filename for a kernel table."""
    return f"kernel_P{P}_Q{Q}_qq{qq_order}.pkl.gz"


def save_kernel_table(
    kt: KernelTable,
    cache_dir: str | Path | None = None,
) -> Path:
    """Save a KernelTable to disk (gzipped pickle).

    Eagerly builds the fast int-grouped representation so it is
    persisted and subsequent loads skip the Fraction→int conversion.

    Returns the path written.
    """
    kt.ensure_fast_repr()
    d = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / _kernel_filename(kt.P, kt.Q, kt.qq_order)
    with gzip.open(path, "wb") as f:
        pickle.dump(kt, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


# In-memory cache: (P, Q, qq_order, cache_dir) → KernelTable | None
# Avoids repeated gzip decompression + pickle loads for the same kernel.
_kernel_mem_cache: dict[tuple, KernelTable | None] = {}


def load_kernel_table(
    P: int,
    Q: int,
    qq_order: int,
    cache_dir: str | Path | None = None,
) -> KernelTable | None:
    """Load a KernelTable from disk, or return None if not found.

    Results are cached in memory so repeated calls for the same
    (P, Q, qq_order) avoid disk I/O entirely.

    First tries an exact match on *qq_order*.  If none exists, falls
    back to the **smallest** cached kernel whose ``stored_qq ≥ qq_order``
    for the same (P, Q).  A higher-order kernel is a mathematical
    superset — extra terms are harmless because the caller's diamond
    truncation discards anything above the requested qq_order.
    """
    cache_key = (P, Q, qq_order, cache_dir)
    cached = _kernel_mem_cache.get(cache_key)
    if cached is not None:
        return cached

    d = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR

    # 1. Exact match (fast path)
    path = d / _kernel_filename(P, Q, qq_order)
    if path.exists():
        with gzip.open(path, "rb") as f:
            kt = pickle.load(f)
        if isinstance(kt, KernelTable):
            _kernel_mem_cache[cache_key] = kt
            return kt

    # 2. Fallback: smallest stored_qq ≥ qq_order for same (P, Q)
    if not d.exists():
        return None
    best: KernelTable | None = None
    for cached_path in sorted(d.glob(f"kernel_P{P}_Q{Q}_qq*.pkl.gz")):
        parts = cached_path.stem.replace(".pkl", "").split("_")
        try:
            stored_qq = int(parts[3][2:])
        except (IndexError, ValueError):
            continue
        if stored_qq < qq_order:
            continue
        with gzip.open(cached_path, "rb") as f:
            candidate = pickle.load(f)
        if not isinstance(candidate, KernelTable):
            continue
        if best is None or candidate.qq_order < best.qq_order:
            best = candidate
    if best is not None:
        _kernel_mem_cache[cache_key] = best
    return best


def clear_kernel_cache() -> int:
    """Clear the in-memory kernel cache.  Returns evicted count."""
    n = len(_kernel_mem_cache)
    _kernel_mem_cache.clear()
    return n


def list_cached_kernels(
    cache_dir: str | Path | None = None,
) -> list[tuple[int, int, int]]:
    """List all cached (P, Q, qq_order) tuples."""
    d = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
    if not d.exists():
        return []
    result = []
    for path in sorted(d.glob("kernel_P*_Q*_qq*.pkl.gz")):
        parts = path.stem.replace(".pkl", "").split("_")
        try:
            p = int(parts[1][1:])
            q = int(parts[2][1:])
            qq = int(parts[3][2:])
            result.append((p, q, qq))
        except (IndexError, ValueError):
            continue
    return result


# ---------------------------------------------------------------------------
# I^ref disk cache — persists refined 3D indices per manifold
# ---------------------------------------------------------------------------
# The refined index I^ref(m, e; η) is manifold-dependent but
# slope-independent.  When computing Dehn fillings for multiple slopes
# of the same manifold, the I^ref grid is identical for all slopes —
# only the kernel convolution changes.
#
# This disk cache stores I^ref results keyed by:
#   (m_ext, e_ext, q_order_half)
# grouped by a content-hash of the NeumannZagierData.  Loading the cache
# populates the in-memory _iref_cache in refined_dehn_filling.py so that
# subsequent calls to _cached_compute_refined_index find them instantly.
#
# Storage: ``data/iref_cache/<name>_<hash16>.pkl.gz``
# ---------------------------------------------------------------------------

def _nz_hash(nz_data: Any) -> str:
    """16-char hex hash of a NeumannZagierData's content."""
    h = hashlib.sha256()
    h.update(nz_data.g_NZ.data.tobytes())
    h.update(nz_data.nu_x.data.tobytes())
    h.update(nz_data.nu_p.data.tobytes())
    return h.hexdigest()[:16]


def _iref_filename(manifold_name: str, nz_data: Any) -> str:
    """Canonical filename for an I^ref cache file."""
    h = _nz_hash(nz_data)
    safe_name = manifold_name.replace("/", "_").replace(" ", "_")
    return f"iref_{safe_name}_{h}.pkl.gz"


def save_iref_cache(
    nz_data: Any,
    manifold_name: str = "unknown",
    cache_dir: str | Path | None = None,
) -> Path | None:
    """Save I^ref entries for *nz_data* from the in-memory cache to disk.

    Extracts all ``_iref_cache`` entries whose nz-content-key matches
    *nz_data* and writes them to a gzipped pickle in *cache_dir*.

    Merges with any existing file so that entries from previous sessions
    (possibly at different qq_orders) are preserved.

    Returns the path written, or ``None`` if there were no entries.
    """
    from manifold_index.core.refined_dehn_filling import (
        _iref_cache,
        _nz_content_key,
    )

    nz_key = _nz_content_key(nz_data)

    # Extract entries belonging to this manifold
    entries: dict[tuple, dict] = {}
    for full_key, value in _iref_cache.items():
        if full_key[0] == nz_key:
            # Strip nz_key from the stored key → (m_ext, e_ext, qq)
            entries[full_key[1:]] = value

    if not entries:
        return None

    d = Path(cache_dir) if cache_dir else _DEFAULT_IREF_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / _iref_filename(manifold_name, nz_data)

    # Merge with existing file (preserve entries at other qq_orders)
    if path.exists():
        try:
            with gzip.open(path, "rb") as f:
                old_data = pickle.load(f)
            if isinstance(old_data, dict) and old_data.get("nz_hash") == _nz_hash(nz_data):
                old_entries = old_data.get("entries", {})
                old_entries.update(entries)
                entries = old_entries
        except Exception:
            pass  # corrupted file — overwrite

    payload = {
        "nz_hash": _nz_hash(nz_data),
        "manifold_name": manifold_name,
        "n_tetrahedra": int(nz_data.n),
        "n_cusps": int(nz_data.r),
        "num_hard": int(nz_data.num_hard),
        "entries": entries,
    }

    with gzip.open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_iref_cache(
    nz_data: Any,
    manifold_name: str = "unknown",
    cache_dir: str | Path | None = None,
    qq_filter: int | None = None,
) -> int:
    """Load I^ref entries from disk into the in-memory ``_iref_cache``.

    Searches for a cache file matching *nz_data*'s content hash.
    Loaded entries are inserted into the module-level ``_iref_cache``
    in ``refined_dehn_filling.py`` so that subsequent calls to
    ``_cached_compute_refined_index`` find them as cache hits.

    Parameters
    ----------
    nz_data : NeumannZagierData
    manifold_name : str
        Used to locate the file (must match save-time name).
    cache_dir : str or Path or None
        Override cache directory.
    qq_filter : int or None
        If given, only load entries with matching ``q_order_half``.
        This avoids filling memory with entries at other qq_orders.

    Returns
    -------
    int
        Number of entries loaded (0 if no file found or empty).
    """
    from manifold_index.core.refined_dehn_filling import (
        _iref_cache,
        _nz_content_key,
    )

    d = Path(cache_dir) if cache_dir else _DEFAULT_IREF_DIR
    path = d / _iref_filename(manifold_name, nz_data)

    if not path.exists():
        return 0

    try:
        with gzip.open(path, "rb") as f:
            payload = pickle.load(f)
    except Exception:
        return 0

    if not isinstance(payload, dict):
        return 0
    if payload.get("nz_hash") != _nz_hash(nz_data):
        return 0

    entries = payload.get("entries", {})
    nz_key = _nz_content_key(nz_data)

    loaded = 0
    for short_key, value in entries.items():
        # short_key = (m_ext_tuple, e_ext_tuple, q_order_half)
        if qq_filter is not None and short_key[-1] != qq_filter:
            continue
        full_key = (nz_key,) + short_key
        if full_key not in _iref_cache:
            _iref_cache[full_key] = value
            loaded += 1
    return loaded


def list_iref_caches(
    cache_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """List all I^ref cache files with their metadata.

    Returns a list of dicts with keys: ``path``, ``manifold_name``,
    ``nz_hash``, ``n_entries``, ``n_tetrahedra``, ``n_cusps``.
    """
    d = Path(cache_dir) if cache_dir else _DEFAULT_IREF_DIR
    if not d.exists():
        return []
    result = []
    for path in sorted(d.glob("iref_*.pkl.gz")):
        try:
            with gzip.open(path, "rb") as f:
                payload = pickle.load(f)
            if isinstance(payload, dict):
                result.append({
                    "path": str(path),
                    "manifold_name": payload.get("manifold_name", "?"),
                    "nz_hash": payload.get("nz_hash", "?"),
                    "n_entries": len(payload.get("entries", {})),
                    "n_tetrahedra": payload.get("n_tetrahedra", "?"),
                    "n_cusps": payload.get("n_cusps", "?"),
                })
        except Exception:
            continue
    return result


# ---------------------------------------------------------------------------
# Pre-computation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Worker function for multiprocessing  (module-level for pickle)
# ---------------------------------------------------------------------------

def _compute_one_kernel_entry(
    m0: int,
    e0: Fraction,
    hj_ks: list[int],
    qq_internal: int,
    eta_order: int,
    m1_range: int,
    final_term_info: dict[tuple[int, Fraction], tuple[int, int, int]],
) -> QEtaSeries | None:
    """Compute a single kernel table entry K^ref(m0, e0).

    Returns the QEtaSeries if non-zero, else None.
    Factored out so that both serial/parallel and adaptive paths share
    exactly the same computation.
    """
    from manifold_index.core.refined_dehn_filling import (
        _apply_is_step,
        _apply_k1_factor_multi,
        _multi_add,
    )

    ell = len(hj_ks)
    lcd = 1 << ell  # 2^ℓ
    unit: MultiEtaSeries = {(0, 0): 1}
    state: dict[tuple[int, Fraction], MultiEtaSeries] = {(m0, e0): unit}

    for step_i in range(ell - 1):
        k_curr = hj_ks[step_i]
        k_next = hj_ks[step_i + 1]
        state = _apply_is_step(
            state, k_curr, k_next,
            qq_internal, eta_order, m1_range,
            use_int=True,
        )

    entry: QEtaSeries = {}
    for (m1, e1), src_series in state.items():
        if not src_series:
            continue
        info = final_term_info.get((m1, e1))
        if info is None:
            continue
        c_f, ph_f, mult_f = info
        contribution = _apply_k1_factor_multi(
            src_series, c_f, ph_f, mult_f, qq_internal,
            truncate=False,
            int_mode=True,
        )
        if contribution:
            entry = _multi_add(entry, contribution) if entry else dict(contribution)

    if entry:
        return {k: Fraction(v, lcd) for k, v in entry.items() if v != 0}
    return None


def _worker_compute_row(
    m_values: list[int],
    e_bounds: dict[int, tuple[int, int]],
    hj_ks: list[int],
    qq_internal: int,
    eta_order: int,
    m1_range: int,
    final_term_info: dict[tuple[int, Fraction], tuple[int, int, int]],
) -> tuple[dict[tuple[int, Fraction], QEtaSeries], int]:
    """Compute kernel entries for complete m-rows (one worker).

    Each m-row scans e within the bounds given by *e_bounds[m]*.
    Row-based dispatch maximises LRU-cache reuse inside each worker
    because all entries in one row share the same m value.

    Returns (partial_table, n_computed).
    """
    result: dict[tuple[int, Fraction], QEtaSeries] = {}
    n_computed = 0
    for m0 in m_values:
        e_lo, e_hi = e_bounds.get(m0, (-100, 100))
        for e_half in range(e_lo, e_hi + 1):
            e0 = Fraction(e_half, 2)
            n_computed += 1
            entry = _compute_one_kernel_entry(
                m0, e0, hj_ks, qq_internal, eta_order,
                m1_range, final_term_info,
            )
            if entry is not None:
                result[(m0, e0)] = entry
    return result, n_computed


def precompute_filling_kernel(
    P: int,
    Q: int,
    qq_order: int,
    eta_order: int | None = None,
    verbose: bool = False,
    progress_callback: Any = None,
    n_workers: int | None = None,
) -> KernelTable:
    """Pre-compute the full Dehn filling kernel K^ref(P/Q; m, e; η).

    For each (m, e) in the relevant grid, runs the IS convolution chain
    on a unit-delta state and applies the final K-factor.

    Optimisations (v3)
    ------------------
    1. **Symmetry**: K^ref(m,e) = K^ref(−m,−e) always holds.  Only
       m ≥ 0 is computed; results are mirrored.  (2× speedup.)
    2. **Parity auto-detection**: Probes m=0..3 to determine if only
       one m-parity produces non-zero entries.  (Up to 2× more.)
    3. **Degree-bound support analysis**: Exact tetrahedron-index degree
       formula to prune the grid to only feasible points (14-19% of full
       grid).  Provably correct — zero false negatives.
    4. **Pilot-gated parallelism**: Flushes computation caches, then
       computes a small stratified sample of grid points to measure
       *cold-cache* per-point cost.  Enables multiprocessing only when
       estimated serial time exceeds 60 s.  Row-based dispatch with
       greedy load-balancing across workers.

    Parameters
    ----------
    P, Q : int
        Coprime slope.
    qq_order : int
        User-facing series truncation order.
    eta_order : int or None
        Max |η_cusp|.  Default: ``qq_order``.
    verbose : bool
        Print progress.
    progress_callback : callable or None
        Called as ``progress_callback(msg: str)``.
    n_workers : int or None
        Number of worker processes.  ``None`` → ``max(1, cpu_count - 2)``.
        Set to ``0`` or ``1`` to disable multiprocessing.

    Returns
    -------
    KernelTable
    """
    # Lazy imports to avoid circular dependency
    from manifold_index.core.refined_dehn_filling import (
        _enumerate_slope1_all,
        hj_continued_fraction,
    )

    if eta_order is None:
        eta_order = qq_order

    hj_ks = hj_continued_fraction(P, Q)
    ell = len(hj_ks)

    if ell < 2:
        raise ValueError(
            f"Slope {P}/{Q} has ℓ={ell} (HJ-CF={hj_ks}). "
            "Pre-computation is only needed for ℓ ≥ 2."
        )

    _is_buffer = qq_order // 2 + 4
    qq_internal = qq_order + _is_buffer

    # Scan-range bounds (generous, used as absolute cap)
    m_scan = int(1.25 * qq_internal) + 2
    e_scan = int(0.90 * qq_internal) + 2
    m1_range = int(1.10 * qq_internal) + 2

    def _status(msg: str):
        if progress_callback:
            progress_callback(msg)
        if verbose:
            print(msg, flush=True)

    # Build final K-factor lookup
    k_final = hj_ks[-1]
    final_terms = _enumerate_slope1_all(k_final, m1_range)
    final_term_info: dict[tuple[int, Fraction], tuple[int, int, int]] = {}
    for m1, e1, c_f, ph_f in final_terms:
        key = (m1, e1)
        if key not in final_term_info:
            final_term_info[key] = (c_f, ph_f, 1)

    # Decide parallelism
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 4) - 2)

    _status(f"[kernel] Pre-computing K^ref({P}/{Q}) at qq={qq_order}: "
            f"HJ={hj_ks}, ℓ={ell}, qq_internal={qq_internal}")

    # ------------------------------------------------------------------
    # Phase 1: Parity auto-detection
    # ------------------------------------------------------------------
    _status("[kernel] Phase 1: parity detection ...")

    has_even = False
    has_odd = False
    _probe_es = [Fraction(0), Fraction(1, 2), Fraction(1), Fraction(-1, 2)]
    for m_probe in range(4):
        if has_even and has_odd:
            break
        for e_probe in _probe_es:
            entry = _compute_one_kernel_entry(
                m_probe, e_probe, hj_ks, qq_internal, eta_order,
                m1_range, final_term_info,
            )
            if entry is not None:
                if m_probe % 2 == 0:
                    has_even = True
                else:
                    has_odd = True
                break

    if has_even and has_odd:
        m_step = 1
        parity_desc = "both parities"
    elif has_even:
        m_step = 2
        parity_desc = "even-m only"
    elif has_odd:
        m_step = 2
        parity_desc = "odd-m only"
    else:
        m_step = 1
        parity_desc = "no hits in probe (full scan)"

    m_start = 0 if has_even else 1
    _status(f"[kernel]   → {parity_desc}, m_step={m_step}")

    # ------------------------------------------------------------------
    # Phase 2: Degree-bound support analysis
    # ------------------------------------------------------------------
    # Use the exact tetrahedron-index degree formula to determine which
    # (m₀, e₀) grid points CAN produce non-zero kernel entries.  This
    # is provably correct (no false negatives) and replaces the previous
    # probe-and-scale heuristic.
    _status("[kernel] Phase 2: degree-bound analysis ...")

    e_bounds, target_m_values = _compute_degree_bounds(
        hj_ks, qq_internal, m_scan, e_scan, m_step, m_start,
        m1_range, final_term_info,
        status_fn=_status,
    )
    total_pts = sum(hi - lo + 1 for lo, hi in e_bounds.values())
    full_pts = len(target_m_values) * (4 * e_scan + 1) if target_m_values else 1
    _status(
        f"[kernel]   → {len(target_m_values)} non-empty rows, "
        f"target grid: {total_pts} pts "
        f"(vs {full_pts} full = {total_pts/full_pts*100:.0f}%)"
    )

    # ------------------------------------------------------------------
    # Phase 3: Compute kernel entries (m ≥ 0 only, symmetry)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    kernel_table: dict[tuple[int, Fraction], QEtaSeries] = {}

    # ── Pilot timing: stratified sample to estimate per-point cost ──
    # Always flush computation caches before the pilot so we measure
    # *cold-cache* cost — which is what parallel workers experience.
    # If the pilot picks serial, the serial path re-warms within the
    # first row (~1-2 s overhead, negligible).
    _PARALLEL_THRESHOLD_S = 60   # use parallel if est. serial time > 60s
    _PILOT_ROWS = 20             # sample rows (spread across full grid)

    if n_workers >= 2 and total_pts > _PILOT_ROWS:
        # Flush computation caches to ensure cold-cache measurement
        from manifold_index.core.refined_dehn_filling import (
            clear_computation_caches,
        )
        clear_computation_caches()

        # Stratified sample: 1 point per evenly-spaced row
        n_sample_rows = min(_PILOT_ROWS, len(target_m_values))
        stride = max(1, len(target_m_values) // n_sample_rows)
        pilot_pts_list: list[tuple[int, Fraction]] = []
        for idx in range(0, len(target_m_values), stride):
            m0 = target_m_values[idx]
            e_lo, e_hi = e_bounds[m0]
            e_mid = (e_lo + e_hi) // 2
            pilot_pts_list.append((m0, Fraction(e_mid, 2)))

        t_pilot = time.perf_counter()
        for m0_p, e0_p in pilot_pts_list:
            entry = _compute_one_kernel_entry(
                m0_p, e0_p, hj_ks, qq_internal, eta_order,
                m1_range, final_term_info,
            )
            if entry is not None:
                kernel_table[(m0_p, e0_p)] = entry
        pilot_elapsed = time.perf_counter() - t_pilot
        cost_per_pt = pilot_elapsed / len(pilot_pts_list)
        est_serial_s = cost_per_pt * total_pts
        use_parallel = est_serial_s > _PARALLEL_THRESHOLD_S

        _status(
            f"[kernel]   Pilot: {len(pilot_pts_list)} pts in "
            f"{pilot_elapsed:.2f}s ({cost_per_pt*1000:.1f}ms/pt) → "
            f"est. serial {est_serial_s:.0f}s → "
            f"{'PARALLEL ×' + str(n_workers) if use_parallel else 'serial'}"
        )
    else:
        use_parallel = False

    _status(f"[kernel] Phase 3: computing {total_pts} grid points (m ≥ 0), "
            f"workers={'parallel ×' + str(n_workers) if use_parallel else 'serial'}")

    if use_parallel:
        # Row-based dispatch: assign complete m-rows to workers
        # Greedy load balancing: sort rows by width (descending), then
        # assign each row to the worker with the smallest current load.
        row_sizes = [(e_bounds[m0][1] - e_bounds[m0][0] + 1, m0)
                     for m0 in target_m_values]
        row_sizes.sort(reverse=True)  # heaviest rows first
        worker_loads = [0] * n_workers
        worker_rows: list[list[int]] = [[] for _ in range(n_workers)]
        for size, m0 in row_sizes:
            lightest = min(range(n_workers), key=lambda w: worker_loads[w])
            worker_rows[lightest].append(m0)
            worker_loads[lightest] += size

        ctx = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            futures = {
                pool.submit(
                    _worker_compute_row,
                    rows, e_bounds, hj_ks, qq_internal, eta_order,
                    m1_range, final_term_info,
                ): i
                for i, rows in enumerate(worker_rows)
                if rows
            }
            done_count = 0
            for future in as_completed(futures):
                worker_id = futures[future]
                partial, n_computed = future.result()
                kernel_table.update(partial)
                done_count += n_computed
                elapsed = time.perf_counter() - t0
                pct = done_count / total_pts * 100 if total_pts else 100
                eta_s = (elapsed / done_count * total_pts - elapsed) if done_count else 0
                _status(
                    f"[kernel]   Worker {worker_id} done: "
                    f"{done_count}/{total_pts} ({pct:.0f}%), "
                    f"{elapsed:.0f}s elapsed, ~{eta_s:.0f}s remaining, "
                    f"{len(kernel_table)} non-zero so far"
                )
    else:
        # Serial path (skip points already computed by pilot)
        computed = len(kernel_table)  # pilot entries count towards total
        for m0 in target_m_values:
            e_lo, e_hi = e_bounds[m0]
            for e_half in range(e_lo, e_hi + 1):
                e0 = Fraction(e_half, 2)
                if (m0, e0) in kernel_table:
                    computed += 1
                    continue
                entry = _compute_one_kernel_entry(
                    m0, e0, hj_ks, qq_internal, eta_order,
                    m1_range, final_term_info,
                )
                if entry is not None:
                    kernel_table[(m0, e0)] = entry
                computed += 1

            if computed % max(1, total_pts // 20) < (e_hi - e_lo + 1):
                elapsed = time.perf_counter() - t0
                eta_s = elapsed / computed * total_pts - elapsed if computed else 0
                _status(
                    f"[kernel]   {computed}/{total_pts} "
                    f"({computed/total_pts*100:.0f}%): "
                    f"{elapsed:.0f}s elapsed, ~{eta_s:.0f}s remaining, "
                    f"{len(kernel_table)} non-zero"
                )

    # ------------------------------------------------------------------
    # Phase 4: Mirror symmetry  K(m,e) → K(−m,−e)
    # ------------------------------------------------------------------
    mirror_entries: dict[tuple[int, Fraction], QEtaSeries] = {}
    for (m, e), entry in kernel_table.items():
        if m == 0 and e == 0:
            continue
        mirror_key = (-m, -e)
        if mirror_key not in kernel_table:
            mirror_entries[mirror_key] = entry
    kernel_table.update(mirror_entries)

    compute_time = time.perf_counter() - t0
    _status(
        f"[kernel] Done: {len(kernel_table)} non-zero entries "
        f"(mirrored {len(mirror_entries)}) "
        f"in {compute_time:.1f}s ({compute_time/60:.1f}min)"
    )

    return KernelTable(
        P=P, Q=Q,
        qq_order=qq_order,
        qq_internal=qq_internal,
        eta_order=eta_order,
        hj_ks=hj_ks,
        table=kernel_table,
        m_scan=m_scan,
        e_scan=e_scan,
        compute_time_s=compute_time,
    )



# ---------------------------------------------------------------------------
# Fast application: use pre-computed kernel instead of IS chain
# ---------------------------------------------------------------------------

# -- Worker helpers for parallel I^ref computation -------------------------

_worker_nz_data: Any = None


def _iref_worker_init(nz_data: Any) -> None:
    """Initialiser for each worker process — store NZ data globally."""
    global _worker_nz_data
    _worker_nz_data = nz_data


def _iref_worker_fn(
    args: tuple[list[int], list, int],
) -> tuple[tuple[int, ...], tuple, dict]:
    """Compute a single I^ref in a worker process.

    Returns (m_ext_tuple, e_ext_tuple, result_dict).
    """
    from manifold_index.core.refined_index import compute_refined_index

    m_ext, e_ext, qq_internal = args
    result = compute_refined_index(_worker_nz_data, m_ext, e_ext, qq_internal)
    return (tuple(m_ext), tuple(e_ext), result)


def apply_precomputed_kernel(
    kernel: KernelTable,
    nz_data: Any,
    cusp_idx: int,
    m_other: Sequence[int] | None = None,
    e_other: Sequence[int | Fraction] | None = None,
    weyl_a: list[Fraction] | None = None,
    weyl_b: list[Fraction] | None = None,
    qq_order: int | None = None,
    verbose: bool = False,
    n_workers: int = 1,
    cache_iref: bool = False,
    manifold_name: str = "unknown",
) -> MultiEtaSeries:
    """Apply a pre-computed kernel to a manifold — the fast path.

    Replaces the grid-scan + IS-chain + K-factor steps of
    ``compute_filled_refined_index`` with a simple summation:

        result = Σ_{m,e}  I^ref(m,e; η_hard) ⊗ K^ref[(m,e)]

    Returns the raw MultiEtaSeries (before diamond truncation).

    Parameters
    ----------
    kernel : KernelTable
    nz_data : NeumannZagierData
    cusp_idx : int
    m_other, e_other : sequences, optional
    weyl_a, weyl_b : Weyl vectors, optional
    qq_order : int or None
        The *user-requested* truncation order.  When the loaded kernel
        was pre-computed at a higher qq_order, this ensures I^ref is
        computed at the appropriate (smaller) depth and the convolution
        bounds are tight.  If *None*, falls back to the kernel's own
        qq_internal.
    verbose : bool
    n_workers : int
        Number of worker processes for parallel I^ref computation.
        1 (default) = sequential.  Values > 1 use a
        ``ProcessPoolExecutor`` to compute I^ref for each (m, e)
        point in parallel.  Beneficial for large grids / high qq;
        the process-spawning overhead may negate gains for small jobs.
    cache_iref : bool
        If True, load I^ref entries from disk before computing and
        save new entries back after.  Dramatically speeds up multi-slope
        workflows for the same manifold.
    manifold_name : str
        Human-readable name for the cache file (e.g. ``"m003"``).
        Required when *cache_iref* is True.

    Returns
    -------
    MultiEtaSeries
        Keys: ``(qq, 2η_0, …, 2η_{H-1}, η_cusp)``
    """
    from manifold_index.core.refined_dehn_filling import (
        _apply_weyl_shift,
        _cached_compute_refined_index,
    )

    r = nz_data.r
    num_hard = nz_data.num_hard

    # Use the request's qq_internal when a lower qq_order is requested,
    # falling back to the kernel's own qq_internal.
    if qq_order is not None:
        _is_buffer = qq_order // 2 + 4
        qq_internal = qq_order + _is_buffer
    else:
        qq_internal = kernel.qq_internal

    if m_other is None:
        m_other = [0] * (r - 1)
    if e_other is None:
        e_other = [0] * (r - 1)

    # Helper to build m_ext, e_ext
    def _make_ext(m_i, e_i):
        m_ext, e_ext = [], []
        other_m_iter = iter(m_other)
        other_e_iter = iter(e_other)
        for k_idx in range(r):
            if k_idx == cusp_idx:
                m_ext.append(m_i)
                e_ext.append(e_i)
            else:
                m_ext.append(next(other_m_iter))
                e_ext.append(next(other_e_iter))
        return m_ext, e_ext

    # ----- Numpy-accelerated integer convolution -----
    # Kernel coefficients are pre-scaled to integers (×LCD) and stored as
    # dense numpy arrays (one per η_cusp, keyed by offset).  The convolution
    # accumulates into dense per-suffix numpy arrays — no Python-level
    # per-element dict ops in the inner loop.
    lcd, k_grouped = kernel.get_int_grouped()

    # ----- Load I^ref disk cache if requested -----
    n_loaded = 0
    if cache_iref:
        n_loaded = load_iref_cache(
            nz_data, manifold_name=manifold_name, qq_filter=qq_internal,
        )
        if verbose and n_loaded:
            print(f"[kernel] Loaded {n_loaded} I^ref entries from disk cache")

    # ----- Pre-compute all I^ref results (sequential or parallel) -----
    # Build the full (m_ext, e_ext) list for every kernel entry.
    all_me_pairs: list[tuple[tuple[int, int], list[int], list]] = []
    for (m_i, e_i) in k_grouped:
        m_ext, e_ext = _make_ext(m_i, e_i)
        all_me_pairs.append(((m_i, e_i), m_ext, e_ext))

    # iref_results: (m_i, e_i) → RefinedIndexResult (dict or empty dict)
    iref_results: dict[tuple[int, int], dict] = {}

    if n_workers > 1 and len(all_me_pairs) > 1:
        # --- Parallel path: ProcessPoolExecutor with initialiser ---
        n_workers = min(n_workers, len(all_me_pairs), os.cpu_count() or 1)
        if verbose:
            print(
                f"[kernel] Parallel I^ref: {len(all_me_pairs)} (m,e) pairs "
                f"across {n_workers} workers"
            )
        tasks = [
            (m_ext, e_ext, qq_internal) for (_, m_ext, e_ext) in all_me_pairs
        ]
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_iref_worker_init,
            initargs=(nz_data,),
        ) as executor:
            futures = {
                executor.submit(_iref_worker_fn, t): me_key
                for t, (me_key, _, _) in zip(tasks, all_me_pairs)
            }
            for future in as_completed(futures):
                _m_ext_t, _e_ext_t, result = future.result()
                me_key = futures[future]
                iref_results[me_key] = result
    else:
        # --- Sequential path: reuse the module-level _iref_cache ---
        for (me_key, m_ext, e_ext) in all_me_pairs:
            iref_results[me_key] = _cached_compute_refined_index(
                nz_data, m_ext, e_ext, q_order_half=qq_internal,
            )

    # ----- Save I^ref disk cache if requested -----
    if cache_iref:
        save_path = save_iref_cache(nz_data, manifold_name=manifold_name)
        if verbose and save_path:
            from manifold_index.core.refined_dehn_filling import (
                _iref_cache,
                _nz_content_key,
            )
            nz_key = _nz_content_key(nz_data)
            n_total = sum(1 for k in _iref_cache if k[0] == nz_key)
            n_new = n_total - n_loaded
            print(
                f"[kernel] Saved I^ref cache: {n_total} entries "
                f"({n_new} new) → {save_path.name}"
            )

    # Dense accumulators: suffix → int64 array of length qq_internal+1
    # where suffix = (2η_0, …, 2η_{H-1}, η_cusp).
    accum_arrays: dict[tuple[int, ...], np.ndarray] = {}
    n_hits = 0

    for (m_i, e_i), by_eta_k in k_grouped.items():
        refined = iref_results.get((m_i, e_i), {})
        if not refined:
            continue

        # Apply Weyl shift (manifold-dependent, hard-η only)
        if weyl_a is not None and weyl_b is not None:
            m_ext, e_ext = _make_ext(m_i, e_i)
            refined = _apply_weyl_shift(
                refined, m_ext, e_ext, weyl_a, weyl_b, num_hard,
                cusp_idx=cusp_idx,
            )

        if not refined:
            continue
        n_hits += 1

        # Group I^ref by η_hard pattern → {η_hard: dense int64 array}
        iref_by_eta: dict[tuple[int, ...], np.ndarray] = {}
        for key, c in refined.items():
            eta_h = key[1:]
            qq_r = key[0]
            arr = iref_by_eta.get(eta_h)
            if arr is None:
                arr = np.zeros(qq_internal + 1, dtype=np.int64)
                iref_by_eta[eta_h] = arr
            if qq_r <= qq_internal:
                arr[qq_r] = c

        # For each (η_hard, η_cusp) pair, convolve and accumulate via numpy.
        # Batched approach: instead of N_ec separate np.convolve calls per
        # η_h group, stack all kern_arr into a matrix and accumulate using
        # L_i vectorised 2-D numpy operations (one per nonzero iref position).
        # This eliminates the Python call overhead of each np.convolve and
        # uses cache-friendly 2-D slice additions instead.
        #
        # Threshold: for very small N_ec the overhead of building kern_matrix
        # and conv_results exceeds the savings; fall back to individual calls.
        _BATCH_THRESH = 4  # use batched path when N_ec >= this

        for eta_h, iref_arr in iref_by_eta.items():
            # Trim trailing zeros for smaller convolution
            nz_pos = np.flatnonzero(iref_arr)
            if len(nz_pos) == 0:
                continue
            iref_trimmed = iref_arr[: int(nz_pos[-1]) + 1]
            L_i = len(iref_trimmed)

            eta_c_items = list(by_eta_k.items())  # [(eta_c, (min_qk, kern_arr)), ...]
            N_ec = len(eta_c_items)

            if N_ec >= _BATCH_THRESH:
                # --- Batched path ---
                # Stack kern arrays into one matrix (pad to common length).
                L_k_max = max(len(v[1]) for _, v in eta_c_items)
                L_out = L_i + L_k_max - 1

                kern_matrix = np.zeros((N_ec, L_k_max), dtype=np.int64)
                for idx, (_, (_, karr)) in enumerate(eta_c_items):
                    kern_matrix[idx, : len(karr)] = karr

                # For each nonzero position j in iref_trimmed, spread
                # iref_trimmed[j] * kern_matrix into the result slice.
                conv_results = np.zeros((N_ec, L_out), dtype=np.int64)
                for j in range(L_i):
                    v = int(iref_trimmed[j])
                    if v == 0:
                        continue
                    conv_results[:, j: j + L_k_max] += v * kern_matrix

                # Write each row to its suffix accumulator.
                for idx, (eta_c, (min_qk, _)) in enumerate(eta_c_items):
                    suffix = eta_h + (eta_c,)
                    conv = conv_results[idx]

                    src_lo = max(0, -min_qk)
                    src_hi = min(L_out, qq_internal + 1 - min_qk)
                    if src_lo >= src_hi:
                        continue
                    dst_lo = src_lo + min_qk
                    dst_hi = src_hi + min_qk

                    acc = accum_arrays.get(suffix)
                    if acc is None:
                        acc = np.zeros(qq_internal + 1, dtype=np.int64)
                        accum_arrays[suffix] = acc
                    acc[dst_lo:dst_hi] += conv[src_lo:src_hi]

            else:
                # --- Scalar path (small N_ec) ---
                for eta_c, (min_qk, kern_arr) in eta_c_items:
                    suffix = eta_h + (eta_c,)

                    conv = np.convolve(iref_trimmed, kern_arr)

                    src_lo = max(0, -min_qk)
                    src_hi = min(len(conv), qq_internal + 1 - min_qk)
                    if src_lo >= src_hi:
                        continue

                    dst_lo = src_lo + min_qk
                    dst_hi = src_hi + min_qk

                    acc = accum_arrays.get(suffix)
                    if acc is None:
                        acc = np.zeros(qq_internal + 1, dtype=np.int64)
                        accum_arrays[suffix] = acc
                    acc[dst_lo:dst_hi] += conv[src_lo:src_hi]

    # Convert dense int64 arrays → sparse Fraction dict, drop zeros
    total_series: MultiEtaSeries = {}
    for suffix, arr in accum_arrays.items():
        nz_idx = np.flatnonzero(arr)
        for qi in nz_idx:
            total_series[(int(qi),) + suffix] = Fraction(int(arr[qi]), lcd)

    if verbose:
        print(
            f"[kernel] Applied pre-computed K^ref({kernel.P}/{kernel.Q}): "
            f"{n_hits} active (m,e) points, "
            f"{len(total_series)} result entries (lcd={lcd})"
        )

    return total_series
