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

# Default cache directory (sibling to src/)
_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "kernel_cache"


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
# Pre-computation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Worker function for multiprocessing  (module-level for pickle)
# ---------------------------------------------------------------------------

def _worker_compute_chunk(
    chunk: list[tuple[int, Fraction]],
    hj_ks: list[int],
    qq_internal: int,
    eta_order: int,
    m1_range: int,
    final_term_info: dict[tuple[int, Fraction], tuple[int, int, int]],
) -> dict[tuple[int, Fraction], QEtaSeries]:
    """Compute kernel entries for a chunk of grid points (one worker)."""
    from manifold_index.core.refined_dehn_filling import (
        _apply_is_step,
        _apply_k1_factor_multi,
        _multi_add,
    )

    ell = len(hj_ks)
    unit: MultiEtaSeries = {(0, 0): Fraction(1)}
    result: dict[tuple[int, Fraction], QEtaSeries] = {}

    for m0, e0 in chunk:
        state: dict[tuple[int, Fraction], MultiEtaSeries] = {(m0, e0): unit}

        for step_i in range(ell - 1):
            k_curr = hj_ks[step_i]
            k_next = hj_ks[step_i + 1]
            state = _apply_is_step(
                state, k_curr, k_next,
                qq_internal, eta_order, m1_range,
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
            )
            if contribution:
                entry = _multi_add(entry, contribution) if entry else dict(contribution)

        if entry:
            result[(m0, e0)] = entry

    return result


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
    on a unit-delta state and applies the final K-factor.  The LRU caches
    on ``_is_kernel`` and ``_etilde_is`` make successive grid points fast
    once the first few have warmed the cache.

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
        _apply_is_step,
        _apply_k1_factor_multi,
        _enumerate_slope1_all,
        _multi_add,
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

    # ----- Tighter bounds (empirically max|m| ≈ qq_internal, ------
    #        max|e| ≈ 0.8*qq_internal for the non-zero region)
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

    # Build grid
    grid_points: list[tuple[int, Fraction]] = []
    for m in range(-m_scan, m_scan + 1):
        for e_half in range(-2 * e_scan, 2 * e_scan + 1):
            grid_points.append((m, Fraction(e_half, 2)))

    total_pts = len(grid_points)

    # Decide parallelism
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 4) - 2)
    use_parallel = n_workers >= 2 and total_pts >= 500

    _status(
        f"[kernel] Pre-computing K^ref({P}/{Q}) at qq_order={qq_order}: "
        f"HJ-CF={hj_ks}, ℓ={ell}, grid={total_pts}, "
        f"qq_internal={qq_internal}, eta_order={eta_order}, "
        f"m_scan={m_scan}, e_scan={e_scan}, m1_range={m1_range}, "
        f"workers={'parallel ×' + str(n_workers) if use_parallel else 'serial'}"
    )

    t0 = time.perf_counter()

    if use_parallel:
        kernel_table = _precompute_parallel(
            grid_points, hj_ks, qq_internal, eta_order, m1_range,
            final_term_info, n_workers, _status,
        )
    else:
        kernel_table = _precompute_serial(
            grid_points, hj_ks, ell, qq_internal, eta_order, m1_range,
            final_term_info, _status, t0,
        )

    compute_time = time.perf_counter() - t0
    _status(
        f"[kernel] Done: {len(kernel_table)} non-zero entries "
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


def _precompute_serial(
    grid_points, hj_ks, ell, qq_internal, eta_order, m1_range,
    final_term_info, _status, t0,
) -> dict[tuple[int, Fraction], QEtaSeries]:
    """Serial (single-process) kernel pre-computation."""
    from manifold_index.core.refined_dehn_filling import (
        _apply_is_step,
        _apply_k1_factor_multi,
        _multi_add,
    )

    total_pts = len(grid_points)
    unit: MultiEtaSeries = {(0, 0): Fraction(1)}
    kernel_table: dict[tuple[int, Fraction], QEtaSeries] = {}

    for idx, (m0, e0) in enumerate(grid_points):
        state: dict[tuple[int, Fraction], MultiEtaSeries] = {(m0, e0): unit}

        for step_i in range(ell - 1):
            k_curr = hj_ks[step_i]
            k_next = hj_ks[step_i + 1]
            state = _apply_is_step(
                state, k_curr, k_next,
                qq_internal, eta_order, m1_range,
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
            )
            if contribution:
                entry = _multi_add(entry, contribution) if entry else dict(contribution)

        if entry:
            kernel_table[(m0, e0)] = entry

        if (idx + 1) % max(1, total_pts // 20) == 0:
            elapsed = time.perf_counter() - t0
            eta_s = elapsed / (idx + 1) * total_pts - elapsed
            _status(
                f"[kernel] {idx+1}/{total_pts} "
                f"({(idx+1)/total_pts*100:.0f}%): "
                f"{elapsed:.0f}s elapsed, ~{eta_s:.0f}s remaining, "
                f"{len(kernel_table)} non-zero"
            )

    return kernel_table


def _precompute_parallel(
    grid_points, hj_ks, qq_internal, eta_order, m1_range,
    final_term_info, n_workers, _status,
) -> dict[tuple[int, Fraction], QEtaSeries]:
    """Parallel kernel pre-computation using multiprocessing."""
    total_pts = len(grid_points)

    # Split grid into chunks — one per worker, interleaved for balance.
    # (Interleaving ensures each worker sees a mix of "easy" and "hard"
    # grid points, avoiding load imbalance.)
    chunks: list[list[tuple[int, Fraction]]] = [[] for _ in range(n_workers)]
    for i, pt in enumerate(grid_points):
        chunks[i % n_workers].append(pt)

    _status(
        f"[kernel] Dispatching {total_pts} grid points "
        f"to {n_workers} workers "
        f"({len(chunks[0])} pts/worker)"
    )

    kernel_table: dict[tuple[int, Fraction], QEtaSeries] = {}
    done_count = 0
    t0 = time.perf_counter()

    # Use 'fork' context on macOS/Linux for shared-nothing but fast start
    ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        futures = {
            pool.submit(
                _worker_compute_chunk,
                chunk, hj_ks, qq_internal, eta_order, m1_range,
                final_term_info,
            ): i
            for i, chunk in enumerate(chunks)
        }

        for future in as_completed(futures):
            worker_id = futures[future]
            partial = future.result()
            kernel_table.update(partial)
            done_count += len(chunks[worker_id])
            elapsed = time.perf_counter() - t0
            pct = done_count / total_pts * 100
            eta_s = (elapsed / done_count * total_pts - elapsed) if done_count else 0
            _status(
                f"[kernel] Worker {worker_id} done: "
                f"{done_count}/{total_pts} ({pct:.0f}%), "
                f"{elapsed:.0f}s elapsed, ~{eta_s:.0f}s remaining, "
                f"{len(kernel_table)} non-zero so far"
            )

    return kernel_table


# ---------------------------------------------------------------------------
# Fast application: use pre-computed kernel instead of IS chain
# ---------------------------------------------------------------------------

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

    # Dense accumulators: suffix → int64 array of length qq_internal+1
    # where suffix = (2η_0, …, 2η_{H-1}, η_cusp).
    accum_arrays: dict[tuple[int, ...], np.ndarray] = {}
    n_hits = 0

    for (m_i, e_i), by_eta_k in k_grouped.items():
        m_ext, e_ext = _make_ext(m_i, e_i)

        refined = _cached_compute_refined_index(
            nz_data, m_ext, e_ext, q_order_half=qq_internal,
        )
        if not refined:
            continue

        # Apply Weyl shift (manifold-dependent, hard-η only)
        if weyl_a is not None and weyl_b is not None:
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

        # For each (η_hard, η_cusp) pair, convolve and accumulate via numpy
        for eta_h, iref_arr in iref_by_eta.items():
            # Trim trailing zeros for smaller convolution
            nz_pos = np.flatnonzero(iref_arr)
            if len(nz_pos) == 0:
                continue
            iref_trimmed = iref_arr[: int(nz_pos[-1]) + 1]

            for eta_c, (min_qk, kern_arr) in by_eta_k.items():
                suffix = eta_h + (eta_c,)

                # 1-D convolution (C-level)
                conv = np.convolve(iref_trimmed, kern_arr)

                # conv[i] corresponds to qq = i + min_qk
                # We want qq ∈ [0, qq_internal], so:
                #   i + min_qk ≥ 0  →  i ≥ -min_qk
                #   i + min_qk ≤ qq_internal  →  i ≤ qq_internal - min_qk
                src_lo = max(0, -min_qk)
                src_hi = min(len(conv), qq_internal + 1 - min_qk)
                if src_lo >= src_hi:
                    continue

                dst_lo = src_lo + min_qk
                dst_hi = src_hi + min_qk

                # Accumulate directly into dense numpy array (no Python loop)
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
