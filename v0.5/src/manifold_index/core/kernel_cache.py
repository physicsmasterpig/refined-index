"""
core/kernel_cache.py — Pre-computed Dehn filling kernel tables.

The refined Dehn filling kernel K^ref(P/Q; m, e; η^{2V}) is **manifold-
independent**: it depends only on the slope (P, Q) and the charge
variables (m, e).  The manifold enters only through the refined 3D index
I^ref(m, e; η^{2W}), which is then convolved with the kernel:

    Î^ref_{P/Q}(η^{2W}, η^{2V}) = Σ_{m,e}  I^ref(m,e; η^{2W}) · K^ref(P/Q; m,e; η^{2V})

Pre-computing the kernel for a slope and storing it on disk turns a
~10 minute IS-chain computation into a sub-second lookup + summation.

Storage
-------
**Bundled kernels** (read-only, shipped with the package):

    src/manifold_index/data/kernel_cache/kernel_P{P}_Q{Q}_qq{qq}.pkl.gz

**User cache** (writable, runtime-generated kernels & I^ref):

    ~/Library/Caches/manifold-index/   (macOS)
    ~/.cache/manifold-index/           (Linux)
    %LOCALAPPDATA%/manifold-index/     (Windows)

Lookup order: user cache → bundled.  New kernels are always saved to the
user cache.  Each file contains a ``KernelTable`` with metadata and the
sparse table mapping ``(m, e) → QEtaSeries``.
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


def _user_cache_dir() -> Path:
    """Return a platform-appropriate writable cache directory.

    Override by setting the environment variable MANIFOLD_INDEX_CACHE_DIR
    to any absolute path (e.g. a repo-tracked ``cache/`` folder).

    macOS:   ~/Library/Caches/manifold-index/
    Linux:   $XDG_CACHE_HOME/manifold-index/ (default ~/.cache/)
    Windows: %LOCALAPPDATA%/manifold-index/
    """
    import sys

    env_override = os.environ.get("MANIFOLD_INDEX_CACHE_DIR")
    if env_override:
        return Path(env_override)

    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    return base / "manifold-index"


# Bundled (read-only) kernels shipped with the package
_BUNDLED_KERNEL_DIR = Path(__file__).resolve().parent.parent / "data" / "kernel_cache"

# User-writable cache for runtime-generated kernels and I^ref entries
_DEFAULT_CACHE_DIR = _user_cache_dir() / "kernel_cache"
_DEFAULT_IREF_DIR = _user_cache_dir() / "iref_cache"

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
    override_reachable: set[tuple[int, int]] | None = None,
) -> tuple[dict[int, tuple[int, int]], list[int]]:
    """Compute per-m₀ e-bounds via degree analysis.

    Returns ``(e_bounds, target_m_values)`` in the same format as the
    probe-and-scale Phase 2 output, but **provably correct**: every
    (m₀, e₀) with a non-zero kernel entry is guaranteed to be inside
    the returned bounds.

    Uses numpy-vectorised degree evaluation over all (e_half, target)
    pairs per m₀ row.  Backward reachability prunes the target set
    for ℓ ≥ 3 IS-chain steps.

    Parameters
    ----------
    override_reachable : set of (int, int) or None
        When provided, skip the backward reachability pass entirely and
        use this set of (m, e_half) pairs as the target set for the
        forward pass.  Used by the V-map path where the reachable
        intermediates have already been degree-filtered during V-map
        precomputation.
    """
    from manifold_index.core.refined_dehn_filling import (
        _enumerate_is_full,
        _enumerate_slope1_all,
    )

    ell = len(hj_ks)
    qq_limit_x2 = 2 * qq_internal

    # --- Build the final target set as (m, e_half) pairs ---
    final_targets: set[tuple[int, int]] = set()
    for (m_t, e_t) in final_term_info:
        e_t_half = int(2 * e_t)
        final_targets.add((m_t, e_t_half))

    # --- Backward reachability for ℓ ≥ 3 ---
    if override_reachable is not None:
        reachable: set[tuple[int, int]] = override_reachable
        if status_fn:
            status_fn(
                f"[degree]   Override reachable: {len(reachable)} V-map targets "
                f"(skipping backward pass)"
            )
    else:
        reachable = final_targets

        for step_i in range(ell - 2, 0, -1):
            k_curr = hj_ks[step_i]
            k_next = hj_ks[step_i + 1]
            # For the last IS step (step_i == ℓ-2) the output feeds into
            # the final K-factor, so candidates are on K(k_ℓ, 1) support.
            # For intermediate steps, the IS kernel can map to the full
            # (½)ℤ² lattice — NOT restricted to any K-support.
            is_last = (step_i == ell - 2)
            if is_last:
                src_terms = _enumerate_slope1_all(k_next, m1_range)
            else:
                e1_range = qq_internal + m1_range // 2
                src_terms = _enumerate_is_full(m1_range, e1_range)
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
        Maximum |cusp V| exponent retained.
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
    # int_grouped: (m,e) → {V_exp: (min_qq, np.ndarray[int64])}
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

    Search order:
      1. User cache directory (runtime-generated kernels)
      2. Bundled kernels shipped with the package

    Within each directory, first tries an exact match on *qq_order*.
    If none exists, falls back to the **smallest** cached kernel whose
    ``stored_qq ≥ qq_order`` for the same (P, Q).  A higher-order
    kernel is a mathematical superset — extra terms are harmless because
    the caller's diamond truncation discards anything above the
    requested qq_order.
    """
    cache_key = (P, Q, qq_order, cache_dir)
    cached = _kernel_mem_cache.get(cache_key)
    if cached is not None:
        return cached

    # Directories to search (user cache first, then bundled)
    dirs: list[Path] = []
    if cache_dir is not None:
        dirs.append(Path(cache_dir))
    else:
        dirs.append(_DEFAULT_CACHE_DIR)
        dirs.append(_BUNDLED_KERNEL_DIR)

    for d in dirs:
        result = _load_kernel_from_dir(d, P, Q, qq_order)
        if result is not None:
            _kernel_mem_cache[cache_key] = result
            return result

    return None


def _load_kernel_from_dir(
    d: Path, P: int, Q: int, qq_order: int,
) -> KernelTable | None:
    """Search a single directory for a matching kernel file."""
    if not d.exists():
        return None

    # 1. Exact match (fast path)
    path = d / _kernel_filename(P, Q, qq_order)
    if path.exists():
        with gzip.open(path, "rb") as f:
            kt = pickle.load(f)
        if isinstance(kt, KernelTable):
            return kt

    # 2. Fallback: smallest stored_qq ≥ qq_order for same (P, Q)
    #    Parse qq from filenames first, sort by qq, then load only the
    #    smallest valid one — avoids deserialising every candidate file
    #    (which can be very slow for large kernels like qq=100).
    candidates: list[tuple[int, Path]] = []
    for cached_path in d.glob(f"kernel_P{P}_Q{Q}_qq*.pkl.gz"):
        parts = cached_path.stem.replace(".pkl", "").split("_")
        try:
            stored_qq = int(parts[3][2:])
        except (IndexError, ValueError):
            continue
        if stored_qq >= qq_order:
            candidates.append((stored_qq, cached_path))

    # Sort by qq ascending so we can load the smallest first
    candidates.sort()

    for _stored_qq, cached_path in candidates:
        with gzip.open(cached_path, "rb") as f:
            candidate = pickle.load(f)
        if isinstance(candidate, KernelTable):
            return candidate

    return None


def clear_kernel_cache() -> int:
    """Clear the in-memory kernel cache.  Returns evicted count."""
    n = len(_kernel_mem_cache)
    _kernel_mem_cache.clear()
    return n


def list_cached_kernels(
    cache_dir: str | Path | None = None,
) -> list[tuple[int, int, int]]:
    """List all cached (P, Q, qq_order) tuples.

    Returns kernels from both the user cache and the bundled package data.
    """
    dirs: list[Path] = []
    if cache_dir is not None:
        dirs.append(Path(cache_dir))
    else:
        dirs.append(_DEFAULT_CACHE_DIR)
        dirs.append(_BUNDLED_KERNEL_DIR)

    seen: set[tuple[int, int, int]] = set()
    result: list[tuple[int, int, int]] = []
    for d in dirs:
        if not d.exists():
            continue
        for path in sorted(d.glob("kernel_P*_Q*_qq*.pkl.gz")):
            parts = path.stem.replace(".pkl", "").split("_")
            try:
                p = int(parts[1][1:])
                q = int(parts[2][1:])
                qq = int(parts[3][2:])
            except (IndexError, ValueError):
                continue
            key = (p, q, qq)
            if key not in seen:
                seen.add(key)
                result.append(key)
    return sorted(result)


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
    grid_params: dict | None = None,
) -> Path | None:
    """Save I^ref entries for *nz_data* from the in-memory cache to disk.

    Extracts all ``_iref_cache`` entries whose nz-content-key matches
    *nz_data* and writes them to a gzipped pickle in *cache_dir*.

    Merges with any existing file so that entries from previous sessions
    (possibly at different qq_orders) are preserved.  The entire
    read-merge-write is serialised with an exclusive ``fcntl`` lock so that
    multiple worker processes can safely write the same file concurrently
    (as happens when a single manifold is split into grid chunks).

    Parameters
    ----------
    grid_params : dict or None
        Optional ``{"m_max": int, "e_max": int, "qq_order": int}`` describing
        the grid that was fully evaluated.  Stored in the file so that
        ``--skip-existing`` can do an instant file-level skip on re-runs
        rather than re-evaluating all zero-result points.  If a wider grid
        is already recorded in the file, the existing record is kept.

    Returns the path written, or ``None`` if there were no entries.
    """
    import sys
    from manifold_index.core.refined_dehn_filling import (
        _iref_cache,
        _nz_content_key,
    )

    nz_key = _nz_content_key(nz_data)

    # Extract entries belonging to this manifold
    entries: dict[tuple, dict] = {}
    for full_key, value in _iref_cache.items():
        if full_key[0] == nz_key:
            entries[full_key[1:]] = value

    if not entries:
        return None

    d = Path(cache_dir) if cache_dir else _DEFAULT_IREF_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / _iref_filename(manifold_name, nz_data)

    # Acquire an exclusive lock on a companion .lock file so that concurrent
    # workers writing the same manifold file do not corrupt the output.
    lock_path = path.with_suffix(".lock")
    lock_file = open(lock_path, "w")
    try:
        if sys.platform != "win32":
            import fcntl
            fcntl.flock(lock_file, fcntl.LOCK_EX)

        # Merge with existing file (preserve entries at other qq_orders)
        old_grid_params: dict | None = None
        if path.exists():
            try:
                with gzip.open(path, "rb") as f:
                    old_data = pickle.load(f)
                if isinstance(old_data, dict) and old_data.get("nz_hash") == _nz_hash(nz_data):
                    old_entries = old_data.get("entries", {})
                    old_grid_params = old_data.get("grid_params")
                    new_gp_wider = grid_params and (
                        old_grid_params is None
                        or grid_params.get("m_max", 0) > old_grid_params.get("m_max", 0)
                        or grid_params.get("e_max", 0) > old_grid_params.get("e_max", 0)
                    )
                    if all(k in old_entries for k in entries) and not new_gp_wider:
                        return path
                    old_entries.update(entries)
                    entries = old_entries
            except Exception:
                pass  # corrupted file — overwrite

        # Determine the grid_params to store: keep the wider of old vs new
        stored_gp: dict | None = None
        if grid_params and old_grid_params:
            stored_gp = {
                "m_max": max(grid_params.get("m_max", 0), old_grid_params.get("m_max", 0)),
                "e_max": max(grid_params.get("e_max", 0), old_grid_params.get("e_max", 0)),
                "qq_order": max(grid_params.get("qq_order", 0), old_grid_params.get("qq_order", 0)),
            }
        else:
            stored_gp = grid_params or old_grid_params

        payload: dict = {
            "nz_hash": _nz_hash(nz_data),
            "manifold_name": manifold_name,
            "n_tetrahedra": int(nz_data.n),
            "n_cusps": int(nz_data.r),
            "num_hard": int(nz_data.num_hard),
            "entries": entries,
        }
        if stored_gp:
            payload["grid_params"] = stored_gp

        # Write atomically: temp file + rename so a crashed writer never
        # leaves a half-written file visible to readers.
        tmp_path = path.with_suffix(".tmp")
        with gzip.open(tmp_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)

    finally:
        if sys.platform != "win32":
            import fcntl
            fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass  # best-effort cleanup; stale .lock files are harmless

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


def enumerate_iref_entries(
    nz_data: Any,
    manifold_name: str,
    qq_filter: int | None = None,
    cache_dir: str | Path | None = None,
) -> list[tuple[list, list, Any]]:
    """Read every cached I^ref entry for *manifold_name* from disk.

    Does **not** load anything into the in-memory cache.

    Parameters
    ----------
    nz_data : NeumannZagierData
        Used to locate the file and verify the NZ hash.
    manifold_name : str
        Must match the name used at save time.
    qq_filter : int or None
        If given, only return entries whose q_order_half matches.
    cache_dir : path or None
        Override cache directory.

    Returns
    -------
    list of ``(m_ext, e_ext, result)`` where:
        - ``m_ext``  is ``list[int]``
        - ``e_ext``  is ``list[Fraction]``
        - ``result`` is the ``RefinedIndexResult`` (``dict``)

    Returns an empty list if no matching cache file is found.
    """
    from fractions import Fraction as _Fraction

    d = Path(cache_dir) if cache_dir else _DEFAULT_IREF_DIR
    path = d / _iref_filename(manifold_name, nz_data)

    if not path.exists():
        return []

    try:
        with gzip.open(path, "rb") as f:
            payload = pickle.load(f)
    except Exception:
        return []

    if not isinstance(payload, dict):
        return []
    if payload.get("nz_hash") != _nz_hash(nz_data):
        return []

    entries = payload.get("entries", {})
    out: list[tuple[list, list, Any]] = []
    for short_key, value in entries.items():
        # short_key = (m_ext_tuple, e_ext_tuple, q_order_half)
        if len(short_key) < 3:
            continue
        m_ext_tuple, e_ext_tuple, qq = short_key
        if qq_filter is not None and qq != qq_filter:
            continue
        out.append((
            list(m_ext_tuple),
            [_Fraction(e) for e in e_ext_tuple],
            value,
        ))
    return out


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
# Non-closable cycle disk cache
# ---------------------------------------------------------------------------
# Stores NonClosableCycleResult objects (one per cusp) for a manifold.
# Results are keyed by the NZ content-hash so they are invalidated
# automatically if the manifold's gluing data changes.
#
# Storage: ~/Library/Caches/manifold-index/nc_cycle_cache/
#          nc_cycle_{name}_{hash16}_qq{qq}.pkl.gz
#
# The filename intentionally omits the slope search range.  The file is
# range-agnostic: it accumulates all slopes ever tested for this manifold
# at this qq_order and merges incrementally on each save.  The caller
# supplies its required range to load_nc_cycle_cache, which checks whether
# those slopes are already covered before returning cached data.
# ---------------------------------------------------------------------------

_DEFAULT_NC_DIR = _user_cache_dir() / "nc_cycle_cache"


def _nc_cycle_filename(
    manifold_name: str,
    nz_data: Any,
    q_order_half: int,
) -> str:
    """Canonical (range-agnostic) filename for an NC-cycle cache file."""
    h = _nz_hash(nz_data)
    safe = manifold_name.replace("/", "_").replace(" ", "_")
    return f"nc_cycle_{safe}_{h}_qq{q_order_half}.pkl.gz"


def save_nc_cycle_cache(
    nz_data: Any,
    manifold_name: str,
    nc_results: list,
    q_order_half: int,
    cache_dir: str | Path | None = None,
) -> Path:
    """Save non-closable cycle results for a manifold, merging with existing data.

    Parameters
    ----------
    nz_data : NeumannZagierData
    manifold_name : str
    nc_results : list[NonClosableCycleResult]
        One entry per cusp (from ``find_non_closable_cycles``).  Each entry
        must have ``slopes_tested``, ``cycles``, and (optionally) ``series_data``
        populated.
    q_order_half : int
        The ``q_order_half`` value used during the NC search.
    cache_dir : Path or None

    Returns
    -------
    Path  — path of the written file.

    Notes
    -----
    The file is **merged** with any existing cache at the same path.
    New slopes extend the ``slopes_tested`` set; existing entries are
    preserved.  This means saves are always cumulative — you can extend the
    search range and re-save without losing previously computed slopes.
    """
    d = Path(cache_dir) if cache_dir else _DEFAULT_NC_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / _nc_cycle_filename(manifold_name, nz_data, q_order_half)

    # Load existing file for merging (if present and hash-compatible)
    old_by_cusp: dict[int, dict] = {}
    if path.exists():
        try:
            with gzip.open(path, "rb") as f:
                old_payload = pickle.load(f)
            if (isinstance(old_payload, dict) and
                    old_payload.get("nz_hash") == _nz_hash(nz_data)):
                for old_entry in old_payload.get("results", []):
                    old_by_cusp[old_entry["cusp_idx"]] = old_entry
        except Exception:
            pass  # corrupted — overwrite

    merged_results = []
    for nc in nc_results:
        old = old_by_cusp.get(nc.cusp_idx, {})

        # slopes_tested: union (old first, then new not already present)
        old_slopes_set = {tuple(s) for s in old.get("slopes_tested", [])}
        new_slopes = [tuple(s) for s in nc.slopes_tested]
        merged_slopes = list(old.get("slopes_tested", [])) + [
            s for s in new_slopes if s not in old_slopes_set
        ]

        # cycles: union by (P, Q) key
        old_cycles_set = {(c["P"], c["Q"]) for c in old.get("cycles", [])}
        new_cycle_dicts = [
            {"cusp_idx": c.cusp_idx, "P": c.P, "Q": c.Q}
            for c in nc.cycles
        ]
        merged_cycles = list(old.get("cycles", [])) + [
            c for c in new_cycle_dicts if (c["P"], c["Q"]) not in old_cycles_set
        ]

        # series_data: new entries override old for the same slope
        merged_series: dict = {**old.get("series_data", {}), **dict(nc.series_data)}

        merged_results.append({
            "cusp_idx": nc.cusp_idx,
            "slopes_tested": merged_slopes,
            "cycles": merged_cycles,
            "series_data": merged_series,
        })

    payload: dict[str, Any] = {
        "manifold_name": manifold_name,
        "nz_hash": _nz_hash(nz_data),
        "n_tetrahedra": int(nz_data.n),
        "n_cusps": int(nz_data.r),
        "q_order_half": q_order_half,
        "results": merged_results,
    }

    with gzip.open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_nc_cycle_cache(
    nz_data: Any,
    manifold_name: str,
    q_order_half: int,
    p_range: tuple[int, int] | None = None,
    q_range: tuple[int, int] | None = None,
    cache_dir: str | Path | None = None,
) -> list | None:
    """Load cached non-closable cycle results from disk.

    Parameters
    ----------
    p_range, q_range : tuple[int, int] or None
        If provided, the function checks that every slope in
        ``_candidate_slopes(p_range, q_range)`` is present in the cached
        ``slopes_tested``.  Returns ``None`` if any slope is missing
        (triggering a fresh search for the extended range).

    Returns
    -------
    list[NonClosableCycleResult]  or  None if not found / hash mismatch /
    coverage incomplete.
    """
    from manifold_index.core.dehn_filling import (
        NonClosableCycle,
        NonClosableCycleResult,
        _candidate_slopes,
    )

    d = Path(cache_dir) if cache_dir else _DEFAULT_NC_DIR
    path = d / _nc_cycle_filename(manifold_name, nz_data, q_order_half)
    if not path.exists():
        return None

    try:
        with gzip.open(path, "rb") as f:
            payload = pickle.load(f)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("nz_hash") != _nz_hash(nz_data):
        return None

    # Coverage check: verify all requested slopes are already in slopes_tested
    if p_range is not None and q_range is not None:
        required = set(_candidate_slopes(
            range(p_range[0], p_range[1] + 1),
            range(q_range[0], q_range[1] + 1),
            canonical_only=False,
        ))
        for entry in payload.get("results", []):
            covered = {tuple(s) for s in entry.get("slopes_tested", [])}
            if not required.issubset(covered):
                return None  # cache miss — caller should extend the search

    results = []
    for entry in payload.get("results", []):
        nc = NonClosableCycleResult(cusp_idx=entry["cusp_idx"])
        nc.cycles = [
            NonClosableCycle(cusp_idx=c["cusp_idx"], P=c["P"], Q=c["Q"])
            for c in entry.get("cycles", [])
        ]
        nc.slopes_tested = [tuple(s) for s in entry.get("slopes_tested", [])]
        nc.series_data = {
            tuple(k): v for k, v in entry.get("series_data", {}).items()
        }
        results.append(nc)
    return results


def list_nc_cycle_caches(
    cache_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """List all NC-cycle cache files with their metadata.

    Returns a list of dicts with keys: ``path``, ``manifold_name``,
    ``nz_hash``, ``n_cusps``, ``q_order_half``, ``n_slopes_tested``,
    ``n_nc_cycles`` (total across all cusps).
    """
    d = Path(cache_dir) if cache_dir else _DEFAULT_NC_DIR
    if not d.exists():
        return []
    result = []
    for path in sorted(d.glob("nc_cycle_*.pkl.gz")):
        try:
            with gzip.open(path, "rb") as f:
                payload = pickle.load(f)
            if isinstance(payload, dict):
                n_nc = sum(
                    len(r.get("cycles", []))
                    for r in payload.get("results", [])
                )
                n_slopes = sum(
                    len(r.get("slopes_tested", []))
                    for r in payload.get("results", [])
                )
                result.append({
                    "path": str(path),
                    "manifold_name": payload.get("manifold_name", "?"),
                    "nz_hash": payload.get("nz_hash", "?"),
                    "n_cusps": payload.get("n_cusps", "?"),
                    "q_order_half": payload.get("q_order_half", "?"),
                    "n_slopes_tested": n_slopes,
                    "n_nc_cycles": n_nc,
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
    tail_map: dict[tuple[int, Fraction], dict[tuple[int, int], int]] | None = None,
    _return_raw_int: bool = False,
) -> QEtaSeries | dict[tuple[int, int], int] | None:
    """Compute a single kernel table entry K^ref(m0, e0).

    Returns the QEtaSeries if non-zero, else None.
    Factored out so that both serial/parallel and adaptive paths share
    exactly the same computation.

    Parameters
    ----------
    tail_map : dict or None
        Optional precomputed IS-chain tail map for ℓ≥3 (see
        ``_precompute_tail_map``).  When provided, IS steps 1..ℓ-1 and
        the final K-factor are replaced by a single O(1) lookup plus a
        series convolution — O(|intermediate_state| × |F_entry|) — instead
        of O(|intermediate_state| × |K-support|) per grid point.
        Ignored when ℓ<3.
    _return_raw_int : bool
        When True, return the raw integer dict (values are true ×2^ℓ
        coefficients) instead of converting to Fractions.  Used by
        ``_precompute_v_map`` when building the V-map base case.
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

    if tail_map is not None and ell >= 3:
        # ── Fast path for ℓ≥3: one IS step + precomputed tail lookup ──
        #
        # IS step 0 only (intermediate step, is_last_step=False).
        # After this step, state[(m1, e1)] holds ×2 int series — each entry
        # is the IS-kernel contribution from (m0, e0) to that (m1, e1).
        state = _apply_is_step(
            state, hj_ks[0], hj_ks[1],
            qq_internal, eta_order, m1_range,
            use_int=True, is_last_step=False,
        )
        # Contract intermediate state with precomputed tail map.
        #   state[(m1,e1)]:    ×2^1 ints  (one IS step from unit source)
        #   tail_map[(m1,e1)]: ×2^(ℓ-1) ints  (tail sub-chain + K-factor)
        #   product:           ×2^ℓ ints  → Fraction(v, 2^ℓ) = Fraction(v, lcd)
        int_entry: dict[tuple[int, int], int] = {}
        for (m1, e1), src_series in state.items():
            if not src_series:
                continue
            F_int = tail_map.get((m1, e1))
            if F_int is None:
                continue
            for (qq_s, eta_s), v_s in src_series.items():
                for (qq_f, eta_f), v_f in F_int.items():
                    new_qq = qq_s + qq_f
                    if new_qq > qq_internal:
                        continue
                    key = (new_qq, eta_s + eta_f)
                    new_val = int_entry.get(key, 0) + v_s * v_f
                    if new_val == 0:
                        int_entry.pop(key, None)
                    else:
                        int_entry[key] = new_val
        if int_entry:
            if _return_raw_int:
                return {k: v for k, v in int_entry.items() if v != 0}
            return {k: Fraction(v, lcd) for k, v in int_entry.items() if v != 0}
        return None
    for step_i in range(ell - 1):
        k_curr = hj_ks[step_i]
        k_next = hj_ks[step_i + 1]
        state = _apply_is_step(
            state, k_curr, k_next,
            qq_internal, eta_order, m1_range,
            use_int=True,
            is_last_step=(step_i == ell - 2),
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
            if entry:
                # In-place accumulation — avoids copying the full dict on each step
                for key, val in contribution.items():
                    new_val = entry.get(key, 0) + val
                    if new_val == 0:
                        entry.pop(key, None)
                    else:
                        entry[key] = new_val
            else:
                entry = dict(contribution)

    if entry:
        if _return_raw_int:
            return {k: v for k, v in entry.items() if v != 0}
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

    .. deprecated:: 0.3.7
        Superseded by :func:`_worker_compute_chunk` for better load
        balancing.  Kept for backward compatibility.
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


def _worker_compute_chunk(
    points: list[tuple[int, int]],
    hj_ks: list[int],
    qq_internal: int,
    eta_order: int,
    m1_range: int,
    final_term_info: dict[tuple[int, Fraction], tuple[int, int, int]],
) -> tuple[dict[tuple[int, Fraction], QEtaSeries], int]:
    """Compute kernel entries for a small chunk of (m, e_half) points.

    Chunks are cut from contiguous row segments so that neighbouring
    points share the same *m* value, preserving LRU-cache locality.

    The caller submits many small chunks to a :class:`ProcessPoolExecutor`;
    as each chunk completes the freed worker picks up the next pending
    chunk, achieving natural **work-stealing** load balancing.

    Returns (partial_table, n_computed).
    """
    result: dict[tuple[int, Fraction], QEtaSeries] = {}
    n_computed = 0
    for m0, e_half in points:
        e0 = Fraction(e_half, 2)
        n_computed += 1
        entry = _compute_one_kernel_entry(
            m0, e0, hj_ks, qq_internal, eta_order,
            m1_range, final_term_info,
        )
        if entry is not None:
            result[(m0, e0)] = entry
    return result, n_computed


def _worker_compute_chunk_with_map(
    points: list[tuple[int, int]],
    hj_ks: list[int],
    qq_internal: int,
    eta_order: int,
    m1_range: int,
    final_term_info: dict[tuple[int, Fraction], tuple[int, int, int]],
    tail_map: dict[tuple[int, Fraction], dict[tuple[int, int], int]],
) -> tuple[dict[tuple[int, Fraction], QEtaSeries], int]:
    """Like ``_worker_compute_chunk`` but uses a precomputed tail map.

    For ℓ≥3 kernels, the tail_map lets each point skip the inner IS steps
    (steps 1..ℓ-1) and the K-factor, replacing them with a lookup and a
    cheap series contraction.
    """
    result: dict[tuple[int, Fraction], QEtaSeries] = {}
    n_computed = 0
    for m0, e_half in points:
        e0 = Fraction(e_half, 2)
        n_computed += 1
        entry = _compute_one_kernel_entry(
            m0, e0, hj_ks, qq_internal, eta_order,
            m1_range, final_term_info,
            tail_map=tail_map,
        )
        if entry is not None:
            result[(m0, e0)] = entry
    return result, n_computed


def _worker_compute_tail_chunk(
    m1_e1_halves: list[tuple[int, int]],
    sub_ks: list[int],
    qq_internal: int,
    eta_order: int,
    m1_range: int,
    final_term_info: dict[tuple[int, Fraction], tuple[int, int, int]],
    sub_tail_map: dict[tuple[int, Fraction], dict[tuple[int, int], int]] | None,
) -> dict[tuple[int, Fraction], dict[tuple[int, int], int]]:
    """Worker for parallel tail-map precomputation.

    For each (m1, e1) in *m1_e1_halves* (encoded as (m1, 2*e1) ints),
    computes the sub-chain kernel entry and returns the raw ×(2^ℓ') int
    version for use by ``_precompute_tail_map``.
    """
    sub_lcd = 1 << len(sub_ks)
    chunk_result: dict[tuple[int, Fraction], dict[tuple[int, int], int]] = {}
    for m1, e1_half in m1_e1_halves:
        e1 = Fraction(e1_half, 2)
        F_frac = _compute_one_kernel_entry(
            m1, e1, sub_ks, qq_internal, eta_order,
            m1_range, final_term_info,
            tail_map=sub_tail_map,
        )
        if F_frac is not None:
            # Convert Fraction(v, sub_lcd) → int v.  All values have
            # denominator dividing sub_lcd by construction, so v*sub_lcd
            # is always an exact integer.
            F_int = {k: int(v * sub_lcd) for k, v in F_frac.items()}
            if F_int:
                chunk_result[(m1, e1)] = F_int
    return chunk_result


def _precompute_tail_map(
    hj_ks: list[int],
    qq_internal: int,
    eta_order: int,
    m1_range: int,
    final_term_info: dict[tuple[int, Fraction], tuple[int, int, int]],
    n_workers: int = 1,
    status_fn: Any = None,
) -> dict[tuple[int, Fraction], dict[tuple[int, int], int]]:
    """Precompute the IS-chain tail map for ℓ≥3 kernel computation.

    For a kernel with HJ-CF chain hj_ks = [k₁, …, kₗ] (ℓ≥3), precomputes:

        tail_map[(m₁, e₁)] = raw ×(2^(ℓ-1)) int series
                             = IS steps [k₂,…,kₗ] + K-factor applied to
                               unit source at (m₁, e₁)

    This allows ``_compute_one_kernel_entry`` to skip the expensive inner IS
    steps for every grid point: instead of re-running steps 1..ℓ-1 + K-factor
    for each of N_grid source points, those steps are done once here for the
    ~38K possible intermediate (m₁, e₁) targets.

    For ℓ=3 this is a one-time cost of O(N_intermediate × |K-support|) IS
    kernel evaluations, after which each grid point only needs step 0
    (O(N_intermediate) evals) plus an O(|sparse_state|) contraction.

    Recursive for ℓ≥4: the sub-chain [k₂,…,kₗ] tail map is itself built
    with this function, so deep chains are handled by a bottom-up sequence of
    fast sub-chain computations.

    Parameters
    ----------
    hj_ks : list[int]
        Full HJ-CF chain [k₁, …, kₗ] with ℓ ≥ 3.
    qq_internal : int
        Internal qq truncation order.
    eta_order : int
        Maximum |cusp V| exponent retained.
    m1_range : int
        Intermediate lattice bound |m₁| ≤ m1_range.
    final_term_info : dict
        K-factor lookup (for the last HJ entry kₗ).
    n_workers : int
        Number of parallel worker processes for the precomputation.
    status_fn : callable or None
        Progress callback.

    Returns
    -------
    dict[(m₁, e₁) → {(qq, η) → int}]
        Only non-zero entries are included.  Values are ×(2^(ℓ-1)) ints.
    """
    from manifold_index.core.refined_dehn_filling import _enumerate_is_full

    ell = len(hj_ks)
    assert ell >= 3, f"_precompute_tail_map requires ℓ ≥ 3, got {ell}"

    sub_ks = list(hj_ks[1:])          # [k₂, …, kₗ], length ℓ-1
    sub_lcd = 1 << len(sub_ks)        # 2^(ℓ-1)

    # Recursively build a sub_tail_map for the sub-chain when ℓ'≥3.
    sub_tail_map: dict | None = None
    if len(sub_ks) >= 3:
        if status_fn:
            status_fn(
                f"[tail_map] Recursively building sub-tail-map for "
                f"sub-chain {sub_ks} (ℓ'={len(sub_ks)})"
            )
        sub_tail_map = _precompute_tail_map(
            sub_ks, qq_internal, eta_order, m1_range, final_term_info,
            n_workers=n_workers, status_fn=status_fn,
        )

    # Enumerate the full intermediate (½)ℤ² lattice.
    e1_range = qq_internal + m1_range // 2
    all_targets = _enumerate_is_full(m1_range, e1_range)
    # Encode as (m1, e1_half) int pairs for easy chunking / worker dispatch.
    all_m1_e1_halves: list[tuple[int, int]] = [
        (m1, int(2 * e1)) for m1, e1, _, _ in all_targets
    ]

    if status_fn:
        status_fn(
            f"[tail_map] Pre-computing tail map for ℓ={ell}, "
            f"sub-chain {sub_ks}: {len(all_m1_e1_halves)} intermediate targets"
        )

    tail_map: dict[tuple[int, Fraction], dict[tuple[int, int], int]] = {}

    if n_workers >= 2 and len(all_m1_e1_halves) > n_workers * 4:
        # Parallel precomputation: split intermediate targets into chunks.
        _TAIL_CHUNK = max(50, len(all_m1_e1_halves) // (n_workers * 8))
        chunks: list[list[tuple[int, int]]] = [
            all_m1_e1_halves[i : i + _TAIL_CHUNK]
            for i in range(0, len(all_m1_e1_halves), _TAIL_CHUNK)
        ]
        ctx = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            futures = [
                pool.submit(
                    _worker_compute_tail_chunk,
                    chunk, sub_ks, qq_internal, eta_order,
                    m1_range, final_term_info, sub_tail_map,
                )
                for chunk in chunks
            ]
            n_done = 0
            log_every = max(1, len(chunks) // 10)
            for i, fut in enumerate(as_completed(futures)):
                tail_map.update(fut.result())
                n_done += 1
                if n_done % log_every == 0 or n_done == len(chunks):
                    if status_fn:
                        status_fn(
                            f"[tail_map]   {n_done}/{len(chunks)} chunks, "
                            f"{len(tail_map)} non-zero so far"
                        )
    else:
        # Serial precomputation.
        for m1, e1_half in all_m1_e1_halves:
            e1 = Fraction(e1_half, 2)
            F_frac = _compute_one_kernel_entry(
                m1, e1, sub_ks, qq_internal, eta_order,
                m1_range, final_term_info,
                tail_map=sub_tail_map,
            )
            if F_frac is not None:
                F_int = {k: int(v * sub_lcd) for k, v in F_frac.items()}
                if F_int:
                    tail_map[(m1, e1)] = F_int

    if status_fn:
        status_fn(
            f"[tail_map]   → {len(tail_map)} non-zero entries "
            f"out of {len(all_m1_e1_halves)} targets "
            f"({len(tail_map)/len(all_m1_e1_halves)*100:.1f}%)"
        )
    return tail_map


# ---------------------------------------------------------------------------
# V-map optimisation for ℓ ≥ 3 kernels
# ---------------------------------------------------------------------------
# The existing _precompute_tail_map + _compute_one_kernel_entry(tail_map=…)
# path has two compounding performance problems at high qq (e.g. qq=50):
#
#  (a) Phase 1 probes call _compute_one_kernel_entry WITHOUT a tail_map →
#      IS step 0 uses _apply_is_step(is_last_step=False) which scans the
#      FULL 151K intermediate grid → ~91 s per probe → SIGALRM timeout.
#
#  (b) Even with tail_map, IS step 0 in _compute_one_kernel_entry still
#      calls _apply_is_step(is_last_step=False) → same 151K scan.
#
#  (c) _precompute_tail_map itself uses _enumerate_is_full (151K) as
#      candidate targets with no degree filter → O(151K × 699) IS-kernel
#      calls = ~105 M calls × 0.6 ms = ~63 000 s.
#
# The V-map approach solves all three:
#  1. _precompute_v_map builds V_raw[(m_i, e_i)] for DEGREE-FEASIBLE
#     intermediates only (typically a few hundred, vs 151K).
#  2. _compute_entry_from_v replaces IS step 0 with a direct scan of
#     V_raw.keys() → O(|V_raw|) IS-kernel calls per source point.
#  3. Phase 1 probes use _compute_entry_from_v → fast.
# ---------------------------------------------------------------------------

_v_parts_global: Any = None  # fork-inherited V-map partition (set before fork)


def _precompute_v_map(
    hj_ks_sub: list[int],
    qq_internal: int,
    eta_order: int,
    m1_range: int,
    final_term_info: dict[tuple[int, Fraction], tuple[int, int, int]],
    status_fn=None,
) -> dict[tuple[int, Fraction], dict[tuple[int, int], int]]:
    """Precompute degree-filtered V-map for the inner IS sub-chain.

    Returns ``V_raw``: a dict mapping ``(m_i, e_i)`` intermediate points
    to raw-integer series dicts (values scaled ×2^len(hj_ks_sub)).
    Only intermediates that can BOTH:
      (a) be reached via IS step 0 from some degree-feasible source, AND
      (b) reach the K-support final targets via the sub-chain
    are included.  Typically a few hundred entries vs the 151K full grid.

    Base case (len=2)
        Calls ``_compute_one_kernel_entry(..., _return_raw_int=True)``
        for each degree-feasible (m_i, e_i) using the K-support targets.

    Recursive case (len≥3)
        Recurses on ``hj_ks_sub[1:]`` to get ``V_inner``, then degree-
        filters using ``V_inner.keys()`` as the target set.
    """
    ell_sub = len(hj_ks_sub)
    if ell_sub < 2:
        raise ValueError(f"_precompute_v_map: sub-chain must have ≥2 steps, got {ell_sub}")

    k_sub0 = hj_ks_sub[0]
    e1_range = qq_internal + m1_range // 2
    qq_limit_x2 = 2 * qq_internal
    e_max_half = 2 * e1_range
    e_half_all = np.arange(-e_max_half, e_max_half + 1, dtype=np.int32)

    # ── Base case: ℓ_sub = 2 ─────────────────────────────────────────────
    if ell_sub == 2:
        # Degree-filter: find which (m_i, e_i) can reach K-support within qq
        target_list = sorted((m_t, int(2 * e_t)) for (m_t, e_t) in final_term_info)
        mt_all = np.array([mt for mt, _  in target_list], dtype=np.int32)
        et_all = np.array([et for _,  et in target_list], dtype=np.int32)
        mt_even = mt_all[et_all % 2 == 0];  et_even = et_all[et_all % 2 == 0]
        mt_odd  = mt_all[et_all % 2 != 0];  et_odd  = et_all[et_all % 2 != 0]

        V_raw: dict[tuple[int, Fraction], dict[tuple[int, int], int]] = {}
        n_cands = 0
        for m_i in range(-m1_range, m1_range + 1):
            mt = mt_even if m_i % 2 == 0 else mt_odd
            et = et_even if m_i % 2 == 0 else et_odd
            if len(mt) == 0:
                continue
            feasible = _degree_feasible_row(m_i, k_sub0, e_half_all, mt, et, qq_limit_x2)
            if not np.any(feasible):
                continue
            for e_half_i in e_half_all[feasible].tolist():
                n_cands += 1
                e_i = Fraction(e_half_i, 2)
                raw = _compute_one_kernel_entry(
                    m_i, e_i, hj_ks_sub, qq_internal, eta_order,
                    m1_range, final_term_info,
                    _return_raw_int=True,
                )
                if raw:
                    V_raw[(m_i, e_i)] = raw

        if status_fn:
            status_fn(
                f"[V-map] ℓ_sub={ell_sub}: {n_cands} candidates → "
                f"{len(V_raw)} non-zero V entries"
            )
        return V_raw

    # ── Recursive case: ℓ_sub ≥ 3 ────────────────────────────────────────
    V_inner = _precompute_v_map(
        hj_ks_sub[1:], qq_internal, eta_order, m1_range, final_term_info,
        status_fn=status_fn,
    )
    if not V_inner:
        return {}

    v_inner_parts = _partition_v_map(V_inner)

    # Degree-filter using V_inner.keys() as the reachable target set
    v_inner_keys = sorted((m_k, int(2 * e_k)) for (m_k, e_k) in V_inner)
    mt_inner = np.array([mt for mt, _  in v_inner_keys], dtype=np.int32)
    et_inner = np.array([et for _,  et in v_inner_keys], dtype=np.int32)
    mt_even_i = mt_inner[et_inner % 2 == 0];  et_even_i = et_inner[et_inner % 2 == 0]
    mt_odd_i  = mt_inner[et_inner % 2 != 0];  et_odd_i  = et_inner[et_inner % 2 != 0]

    V_raw = {}
    n_cands = 0
    for m_i in range(-m1_range, m1_range + 1):
        mt = mt_even_i if m_i % 2 == 0 else mt_odd_i
        et = et_even_i if m_i % 2 == 0 else et_odd_i
        if len(mt) == 0:
            continue
        feasible = _degree_feasible_row(m_i, k_sub0, e_half_all, mt, et, qq_limit_x2)
        if not np.any(feasible):
            continue
        for e_half_i in e_half_all[feasible].tolist():
            n_cands += 1
            e_i = Fraction(e_half_i, 2)
            raw = _compute_raw_v_entry(
                m_i, e_i, k_sub0, v_inner_parts, qq_internal, eta_order,
            )
            if raw:
                V_raw[(m_i, e_i)] = raw

    if status_fn:
        status_fn(
            f"[V-map] ℓ_sub={ell_sub}: {n_cands} candidates → "
            f"{len(V_raw)} non-zero V entries"
        )
    return V_raw


def _partition_v_map(
    V_raw: dict[tuple[int, Fraction], dict],
) -> tuple[list, list, list, list]:
    """Partition V_raw into 4 groups by (m_i parity) × (e_i integrality).

    Returns ``(v_even_eint, v_even_ehalf, v_odd_eint, v_odd_ehalf)``
    where each element is a list of ``((m_i, e_i), raw_dict)`` pairs.

    This matches the 4-way parity selection used by ``_apply_is_step``
    with ``is_last_step=False``:
      - ``p = -(e_half_src + k_curr * m_src)``
      - m_i parity must equal ``p % 2``
      - e_i integrality: integer iff m_src is even
    """
    v_even_eint: list = []
    v_even_ehalf: list = []
    v_odd_eint: list = []
    v_odd_ehalf: list = []
    for (m_i, e_i), raw in V_raw.items():
        if m_i % 2 == 0:
            if e_i.denominator == 1:
                v_even_eint.append(((m_i, e_i), raw))
            else:
                v_even_ehalf.append(((m_i, e_i), raw))
        else:
            if e_i.denominator == 1:
                v_odd_eint.append(((m_i, e_i), raw))
            else:
                v_odd_ehalf.append(((m_i, e_i), raw))
    return v_even_eint, v_even_ehalf, v_odd_eint, v_odd_ehalf


def _compute_raw_v_entry(
    m0: int,
    e0: Fraction,
    k0: int,
    v_inner_parts: tuple,
    qq_internal: int,
    eta_order: int,
) -> dict[tuple[int, int], int] | None:
    """Compute a raw-integer V-map entry from an inner V-map (recursive case).

    For each (m_i, e_i) in the inner V-map:
        result += IS_raw(m0, e_in0, m_i, e_i) ⊗ V_inner_raw[(m_i, e_i)]

    where IS_raw is the ×2-scaled IS kernel.

    LCD accounting:
        IS_raw values are ×2.
        V_inner_raw values are ×2^(ℓ_sub-1).
        Product is ×2^ℓ_sub = ×lcd_this (correct for the caller level).

    Used inside ``_precompute_v_map`` for ℓ_sub ≥ 3.
    """
    from manifold_index.core.refined_dehn_filling import _is_kernel as _isk

    e_in0 = -e0 - Fraction(k0 * m0, 2)
    p = -(int(2 * e0) + k0 * m0)
    v_ee, v_eh, v_oe, v_oh = v_inner_parts
    v_items = (v_ee if m0 % 2 == 0 else v_eh) if p % 2 == 0 else (v_oe if m0 % 2 == 0 else v_oh)
    if not v_items:
        return None

    entry: dict[tuple[int, int], int] = {}
    for (m_i, e_i), v_raw in v_items:
        is_val = _isk(m0, e_in0, m_i, e_i, qq_internal, eta_order)
        if not is_val:
            continue
        for (qq_is, eta_is), c_is in is_val.items():
            for (qq_v, eta_v), c_v in v_raw.items():
                new_qq = qq_is + qq_v
                if new_qq > qq_internal:
                    continue
                key = (new_qq, eta_is + eta_v)
                new_val = entry.get(key, 0) + c_is * c_v
                if new_val:
                    entry[key] = new_val
                else:
                    entry.pop(key, None)
    return entry if entry else None


def _compute_entry_from_v(
    m0: int,
    e0: Fraction,
    k0: int,
    v_parts: tuple,
    qq_internal: int,
    eta_order: int,
    lcd_full: int,
) -> QEtaSeries | None:
    """Compute K^ref(m0, e0) for ℓ≥3 using the precomputed V-map.

    Replaces the slow O(151K) IS-step-0 scan in ``_compute_one_kernel_entry``
    with a targeted O(|V_raw|) scan over precomputed feasible intermediates.

    LCD accounting:
        IS kernel (_is_kernel) returns ×2 raw ints.
        V_raw values are ×2^(ℓ-1) raw ints.
        Product: ×2 × ×2^(ℓ-1) = ×2^ℓ = ×lcd_full ✓

    Parameters
    ----------
    k0 : int
        ``hj_ks[0]`` — the first Hirzebruch-Jung coefficient.
    v_parts : tuple
        4-way partition from ``_partition_v_map(V_raw)``.
    lcd_full : int
        ``2^ℓ`` — the LCD for converting raw ints to Fractions.
    """
    from manifold_index.core.refined_dehn_filling import _is_kernel as _isk

    e_in0 = -e0 - Fraction(k0 * m0, 2)
    p = -(int(2 * e0) + k0 * m0)
    v_ee, v_eh, v_oe, v_oh = v_parts
    v_items = (v_ee if m0 % 2 == 0 else v_eh) if p % 2 == 0 else (v_oe if m0 % 2 == 0 else v_oh)
    if not v_items:
        return None

    entry: dict[tuple[int, int], int] = {}
    for (m_i, e_i), v_raw in v_items:
        is_val = _isk(m0, e_in0, m_i, e_i, qq_internal, eta_order)
        if not is_val:
            continue
        for (qq_is, eta_is), c_is in is_val.items():
            for (qq_v, eta_v), c_v in v_raw.items():
                new_qq = qq_is + qq_v
                if new_qq > qq_internal:
                    continue
                key = (new_qq, eta_is + eta_v)
                new_val = entry.get(key, 0) + c_is * c_v
                if new_val:
                    entry[key] = new_val
                else:
                    entry.pop(key, None)

    if entry:
        return {k: Fraction(v, lcd_full) for k, v in entry.items() if v != 0}
    return None


def _worker_compute_chunk_v_map(
    points: list[tuple[int, int]],
    k0: int,
    qq_internal: int,
    eta_order: int,
    lcd_full: int,
) -> tuple[dict[tuple[int, Fraction], QEtaSeries], int]:
    """Parallel worker for ℓ≥3 kernel entries using the fork-inherited V-map.

    Uses the module-level ``_v_parts_global`` (set in the parent process
    before the ``ProcessPoolExecutor`` is created) to avoid pickling the
    potentially large V-map data.

    Parameters
    ----------
    points : list of (m0, e_half) pairs
    k0 : int
        ``hj_ks[0]``.
    """
    result: dict[tuple[int, Fraction], QEtaSeries] = {}
    n_computed = 0
    for m0, e_half in points:
        e0 = Fraction(e_half, 2)
        n_computed += 1
        entry = _compute_entry_from_v(
            m0, e0, k0, _v_parts_global, qq_internal, eta_order, lcd_full,
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
    hj_ks_override: list[int] | None = None,
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
        Max |cusp V|.  Default: ``qq_order``.
    verbose : bool
        Print progress.
    progress_callback : callable or None
        Called as ``progress_callback(msg: str)``.
    n_workers : int or None
        Number of worker processes.  ``None`` → ``max(1, cpu_count - 2)``.
        Set to ``0`` or ``1`` to disable multiprocessing.
    hj_ks_override : list[int] or None
        If provided, use this HJ-CF list instead of the one computed from
        P/Q.  Useful for consistency checks: the same rational slope can
        have multiple valid HJ decompositions (e.g. 1/2 = [0,-2] at ℓ=2
        or [1,3,1] at ℓ=3); passing the alternate list forces the longer
        decomposition through the V-map path so the two results can be
        compared.  The list must evaluate to P/Q as an HJ-CF.

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
    if hj_ks_override is not None:
        hj_ks = list(hj_ks_override)
    ell = len(hj_ks)

    if ell < 2:
        raise ValueError(
            f"Slope {P}/{Q} has ℓ={ell} (HJ-CF={hj_ks}). "
            "Pre-computation is only needed for ℓ ≥ 2."
        )

    _is_buffer = qq_order + 4
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
    # ℓ ≥ 3: V-map fast path  (bypasses the slow 151K IS-step-0 scan)
    # ------------------------------------------------------------------
    # For ℓ ≥ 3 kernels the standard path is bottlenecked by IS step 0
    # which calls _apply_is_step(is_last_step=False) → scans 151K
    # intermediate targets (~91 s per probe at qq=50).
    #
    # The V-map path precomputes the inner sub-chain for degree-feasible
    # intermediates only (typically a few hundred entries), then computes
    # each (m0, e0) entry by scanning only those V-map keys — O(|V_raw|)
    # IS-kernel calls instead of O(151K).
    #
    # This block handles ALL of Phases 1-4 for ℓ ≥ 3 and returns early,
    # leaving the existing code below untouched for the ℓ = 2 case.
    # ------------------------------------------------------------------
    if ell >= 3:
        _status(
            f"[kernel] ℓ={ell}≥3: building V-map for sub-chain {hj_ks[1:]} ..."
        )
        t_vmap = time.perf_counter()
        V_raw = _precompute_v_map(
            hj_ks[1:], qq_internal, eta_order, m1_range, final_term_info,
            status_fn=_status,
        )
        vmap_elapsed = time.perf_counter() - t_vmap
        _status(
            f"[kernel]   V-map: {len(V_raw)} non-zero intermediate points "
            f"in {vmap_elapsed:.1f}s"
        )

        if not V_raw:
            _status("[kernel]   V-map empty → kernel is identically zero")
            return KernelTable(
                P=P, Q=Q,
                qq_order=qq_order,
                qq_internal=qq_internal,
                eta_order=eta_order,
                hj_ks=hj_ks,
                table={},
                m_scan=m_scan,
                e_scan=e_scan,
                compute_time_s=0.0,
            )

        v_parts = _partition_v_map(V_raw)
        lcd_full = 1 << ell  # 2^ℓ

        # Phase 1 (V-map): fast parity detection
        _status("[kernel] Phase 1: parity detection (V-map) ...")
        has_even = has_odd = False
        _probe_es_v = [Fraction(0), Fraction(1, 2), Fraction(1), Fraction(-1, 2)]
        for m_probe in range(4):
            if has_even and has_odd:
                break
            for e_probe in _probe_es_v:
                entry = _compute_entry_from_v(
                    m_probe, e_probe, hj_ks[0], v_parts,
                    qq_internal, eta_order, lcd_full,
                )
                if entry is not None:
                    if m_probe % 2 == 0:
                        has_even = True
                    else:
                        has_odd = True
                    break

        if has_even and has_odd:
            m_step_v, parity_desc_v = 1, "both parities"
        elif has_even:
            m_step_v, parity_desc_v = 2, "even-m only"
        elif has_odd:
            m_step_v, parity_desc_v = 2, "odd-m only"
        else:
            m_step_v, parity_desc_v = 1, "no hits in probe (full scan)"
        m_start_v = 0 if has_even else 1
        _status(f"[kernel]   → {parity_desc_v}, m_step={m_step_v}")

        # Phase 2 (V-map): degree-bound analysis
        _status("[kernel] Phase 2: degree-bound analysis (V-map targets) ...")
        override_reach = {(m_i, int(2 * e_i)) for (m_i, e_i) in V_raw}
        e_bounds_v, target_m_v = _compute_degree_bounds(
            hj_ks, qq_internal, m_scan, e_scan, m_step_v, m_start_v,
            m1_range, final_term_info,
            status_fn=_status,
            override_reachable=override_reach,
        )
        total_pts_v = sum(hi - lo + 1 for lo, hi in e_bounds_v.values())
        full_pts_v = len(target_m_v) * (4 * e_scan + 1) if target_m_v else 1
        _status(
            f"[kernel]   → {len(target_m_v)} non-empty rows, "
            f"target grid: {total_pts_v} pts "
            f"(vs {full_pts_v} full = {total_pts_v/full_pts_v*100:.0f}%)"
        )

        # Phase 3 (V-map): compute entries
        t0_v = time.perf_counter()
        kernel_table_v: dict[tuple[int, Fraction], QEtaSeries] = {}

        _PARALLEL_THRESHOLD_V = 60   # seconds
        _PILOT_ROWS_V = 20

        if n_workers >= 2 and total_pts_v > _PILOT_ROWS_V:
            from manifold_index.core.refined_dehn_filling import clear_computation_caches
            clear_computation_caches()

            n_samp = min(_PILOT_ROWS_V, len(target_m_v))
            stride_v = max(1, len(target_m_v) // n_samp)
            pilot_pts_v: list[tuple[int, Fraction]] = []
            for idx in range(0, len(target_m_v), stride_v):
                m0 = target_m_v[idx]
                e_lo, e_hi = e_bounds_v[m0]
                e_mid = (e_lo + e_hi) // 2
                pilot_pts_v.append((m0, Fraction(e_mid, 2)))

            t_pilot_v = time.perf_counter()
            for m0_p, e0_p in pilot_pts_v:
                ent = _compute_entry_from_v(
                    m0_p, e0_p, hj_ks[0], v_parts, qq_internal, eta_order, lcd_full,
                )
                if ent is not None:
                    kernel_table_v[(m0_p, e0_p)] = ent
            pilot_elapsed_v = time.perf_counter() - t_pilot_v
            cost_per_pt_v = pilot_elapsed_v / max(1, len(pilot_pts_v))
            est_serial_v = cost_per_pt_v * total_pts_v
            use_parallel_v = est_serial_v > _PARALLEL_THRESHOLD_V

            _status(
                f"[kernel]   Pilot (V-map): {len(pilot_pts_v)} pts in "
                f"{pilot_elapsed_v:.2f}s ({cost_per_pt_v*1000:.1f}ms/pt) → "
                f"est. serial {est_serial_v:.0f}s → "
                f"{'PARALLEL ×' + str(n_workers) if use_parallel_v else 'serial'}"
            )
        else:
            use_parallel_v = False
            pilot_pts_v = []

        _status(
            f"[kernel] Phase 3: computing {total_pts_v} grid pts (V-map path), "
            f"workers={'parallel ×' + str(n_workers) if use_parallel_v else 'serial'}"
        )

        if use_parallel_v:
            global _v_parts_global
            _v_parts_global = v_parts

            _CHUNK_SIZE_V = 50
            chunks_v: list[list[tuple[int, int]]] = []
            cur_chunk_v: list[tuple[int, int]] = []
            pilot_set_v = {(m0, e0) for m0, e0 in pilot_pts_v}
            for m0 in target_m_v:
                e_lo, e_hi = e_bounds_v[m0]
                for e_half in range(e_lo, e_hi + 1):
                    if (m0, Fraction(e_half, 2)) in pilot_set_v:
                        continue
                    cur_chunk_v.append((m0, e_half))
                    if len(cur_chunk_v) >= _CHUNK_SIZE_V:
                        chunks_v.append(cur_chunk_v)
                        cur_chunk_v = []
            if cur_chunk_v:
                chunks_v.append(cur_chunk_v)

            remaining_v = sum(len(c) for c in chunks_v)
            _status(
                f"[kernel]   Chunk dispatch: {len(chunks_v)} chunks "
                f"(~{_CHUNK_SIZE_V} pts each), {remaining_v} pts to compute"
            )

            ctx_v = multiprocessing.get_context("fork")
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx_v) as pool_v:
                futures_v = {
                    pool_v.submit(
                        _worker_compute_chunk_v_map,
                        chunk, hj_ks[0], qq_internal, eta_order, lcd_full,
                    ): i
                    for i, chunk in enumerate(chunks_v)
                }
                done_v = len(pilot_pts_v)
                chunks_done_v = 0
                log_iv = max(1, len(chunks_v) // 20)
                for fut_v in as_completed(futures_v):
                    partial_v, nc_v = fut_v.result()
                    kernel_table_v.update(partial_v)
                    done_v += nc_v
                    chunks_done_v += 1
                    if chunks_done_v % log_iv == 0 or chunks_done_v == len(chunks_v):
                        elapsed_v = time.perf_counter() - t0_v
                        pct_v = done_v / total_pts_v * 100 if total_pts_v else 100
                        eta_v = (
                            elapsed_v / done_v * total_pts_v - elapsed_v
                        ) if done_v else 0
                        _status(
                            f"[kernel]   {chunks_done_v}/{len(chunks_v)} chunks, "
                            f"{done_v}/{total_pts_v} pts ({pct_v:.0f}%), "
                            f"{elapsed_v:.0f}s elapsed, ~{eta_v:.0f}s remaining, "
                            f"{len(kernel_table_v)} non-zero so far"
                        )
        else:
            computed_v = len(kernel_table_v)
            for m0 in target_m_v:
                e_lo, e_hi = e_bounds_v[m0]
                for e_half in range(e_lo, e_hi + 1):
                    e0 = Fraction(e_half, 2)
                    if (m0, e0) in kernel_table_v:
                        computed_v += 1
                        continue
                    ent = _compute_entry_from_v(
                        m0, e0, hj_ks[0], v_parts, qq_internal, eta_order, lcd_full,
                    )
                    if ent is not None:
                        kernel_table_v[(m0, e0)] = ent
                    computed_v += 1

                if computed_v % max(1, total_pts_v // 20) < (e_hi - e_lo + 1):
                    elapsed_v = time.perf_counter() - t0_v
                    eta_v = (
                        elapsed_v / computed_v * total_pts_v - elapsed_v
                    ) if computed_v else 0
                    _status(
                        f"[kernel]   {computed_v}/{total_pts_v} "
                        f"({computed_v/total_pts_v*100:.0f}%): "
                        f"{elapsed_v:.0f}s elapsed, ~{eta_v:.0f}s remaining, "
                        f"{len(kernel_table_v)} non-zero"
                    )

        # Phase 4 (V-map): Mirror symmetry  K(m,e) → K(−m,−e)
        mirror_v: dict[tuple[int, Fraction], QEtaSeries] = {}
        for (m, e), ent in kernel_table_v.items():
            if m == 0 and e == Fraction(0):
                continue
            mirror_key = (-m, -e)
            if mirror_key not in kernel_table_v:
                mirror_v[mirror_key] = ent
        kernel_table_v.update(mirror_v)

        compute_time_v = time.perf_counter() - t0_v
        _status(
            f"[kernel] Done (V-map): {len(kernel_table_v)} non-zero entries "
            f"(mirrored {len(mirror_v)}) "
            f"in {compute_time_v:.1f}s ({compute_time_v/60:.1f}min)"
        )
        return KernelTable(
            P=P, Q=Q,
            qq_order=qq_order,
            qq_internal=qq_internal,
            eta_order=eta_order,
            hj_ks=hj_ks,
            table=kernel_table_v,
            m_scan=m_scan,
            e_scan=e_scan,
            compute_time_s=compute_time_v,
        )
    # ── end ℓ ≥ 3 V-map fast path ─────────────────────────────────────

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
    # Phase 2.5: Precompute IS-chain tail map (ℓ≥3 only)
    # ------------------------------------------------------------------
    # For ℓ≥3 kernels, the dominant per-point cost is the inner IS steps
    # (steps 1..ℓ-1) + K-factor, which touch O(|K-support|) IS-kernel
    # evaluations for every non-zero intermediate (m1, e1) produced by
    # IS step 0.  These inner evaluations depend only on (m1, e1), NOT on
    # the source (m0, e0), so they can be pre-computed once for all
    # possible intermediate targets and reused across the entire grid.
    #
    # The tail map is built with warm lru-caches (from degree analysis),
    # then caches are cleared so the pilot measures the true cold cost of
    # the now-fast per-point path.
    tail_map: dict | None = None
    t_tail = time.perf_counter()
    if ell >= 3:
        _status(f"[kernel] Phase 2.5: pre-computing IS tail map for ℓ={ell} ...")
        tail_map = _precompute_tail_map(
            hj_ks, qq_internal, eta_order, m1_range, final_term_info,
            n_workers=n_workers,
            status_fn=_status,
        )
        tail_elapsed = time.perf_counter() - t_tail
        _status(
            f"[kernel]   tail map ready: {len(tail_map)} entries "
            f"in {tail_elapsed:.1f}s"
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
                tail_map=tail_map,
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
        # Chunk-based work queue: split the grid into small chunks of
        # ~_CHUNK_SIZE points and submit them all to the executor.
        # Workers pull the next chunk as soon as they finish the current
        # one, achieving natural work-stealing load balancing.
        #
        # Chunks are cut from contiguous row segments so that points in
        # one chunk share the same m value, preserving LRU-cache reuse.
        _CHUNK_SIZE = 50

        chunks: list[list[tuple[int, int]]] = []
        current_chunk: list[tuple[int, int]] = []
        for m0 in target_m_values:
            e_lo, e_hi = e_bounds[m0]
            for e_half in range(e_lo, e_hi + 1):
                if (m0, Fraction(e_half, 2)) in kernel_table:
                    continue  # already computed by pilot
                current_chunk.append((m0, e_half))
                if len(current_chunk) >= _CHUNK_SIZE:
                    chunks.append(current_chunk)
                    current_chunk = []
        if current_chunk:
            chunks.append(current_chunk)

        remaining_pts = sum(len(c) for c in chunks)
        _status(
            f"[kernel]   Chunk dispatch: {len(chunks)} chunks "
            f"(~{_CHUNK_SIZE} pts each), {remaining_pts} pts to compute"
        )

        ctx = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            if tail_map is not None:
                # ℓ≥3 fast path: workers skip inner IS steps via tail_map.
                futures = {
                    pool.submit(
                        _worker_compute_chunk_with_map,
                        chunk, hj_ks, qq_internal, eta_order,
                        m1_range, final_term_info, tail_map,
                    ): i
                    for i, chunk in enumerate(chunks)
                }
            else:
                futures = {
                    pool.submit(
                        _worker_compute_chunk,
                        chunk, hj_ks, qq_internal, eta_order,
                        m1_range, final_term_info,
                    ): i
                    for i, chunk in enumerate(chunks)
                }
            done_count = len(pilot_pts_list)  # pilot computed this many points (incl. zeros)
            chunks_done = 0
            log_interval = max(1, len(chunks) // 20)  # ~5% increments
            for future in as_completed(futures):
                partial, n_computed = future.result()
                kernel_table.update(partial)
                done_count += n_computed
                chunks_done += 1
                if chunks_done % log_interval == 0 or chunks_done == len(chunks):
                    elapsed = time.perf_counter() - t0
                    pct = done_count / total_pts * 100 if total_pts else 100
                    eta_s = (elapsed / done_count * total_pts - elapsed) if done_count else 0
                    _status(
                        f"[kernel]   {chunks_done}/{len(chunks)} chunks, "
                        f"{done_count}/{total_pts} pts ({pct:.0f}%), "
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
                    tail_map=tail_map,
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

        result = Σ_{m,e}  I^ref(m,e; η^{2W}) ⊗ K^ref[(m,e)]

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
        Keys: ``(qq, 2W_0, …, 2W_{H-1}, 2V)``
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
        _is_buffer = qq_order + 4
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
    # dense numpy arrays (one per V_exp, keyed by offset).  The convolution
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
    # where suffix = (2W_0, …, 2W_{H-1}, 2V).
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

        # Group I^ref by η^{2W} pattern → {η_key: dense int64 array}
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

        # For each (η^{2W}, V_exp) pair, convolve and accumulate via numpy.
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
