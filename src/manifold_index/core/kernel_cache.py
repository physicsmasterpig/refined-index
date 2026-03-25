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
    3. **Probe-and-scale support prediction**: Runs a cheap low-qq
       probe to discover the non-zero support shape, then scales the
       per-m e-bounds for the target qq.  Eliminates ~80-90% of zero
       computations.
    4. **Row-based parallel dispatch**: Workers process complete m-rows
       (better LRU-cache locality than interleaved points).

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
    # Phase 2: Low-qq probe to discover support shape
    # ------------------------------------------------------------------
    # Run a cheap computation at low qq to find the (m, e) support
    # boundary.  Then scale the boundary for the target qq.
    _PROBE_QQ = 8  # fast, usually < 1.5 s; more data points than 6
    _WIDTH_MARGIN = 1.4  # safety factor on half-width only
    _MARGIN_ABS = 8  # absolute margin in half-integer e-steps

    do_probe = qq_order > _PROBE_QQ + 4  # only probe if target is much larger

    if do_probe:
        _status("[kernel] Phase 2: low-qq probe for support shape ...")
        from manifold_index.core.refined_dehn_filling import clear_filling_caches
        clear_filling_caches()

        probe_is_buffer = _PROBE_QQ // 2 + 4
        probe_qq_internal = _PROBE_QQ + probe_is_buffer
        probe_m_scan = int(1.25 * probe_qq_internal) + 2
        probe_e_scan = int(0.90 * probe_qq_internal) + 2
        probe_m1_range = int(1.10 * probe_qq_internal) + 2

        # Build probe final terms
        probe_final_terms = _enumerate_slope1_all(k_final, probe_m1_range)
        probe_fti: dict[tuple[int, Fraction], tuple[int, int, int]] = {}
        for m1, e1, c_f, ph_f in probe_final_terms:
            key = (m1, e1)
            if key not in probe_fti:
                probe_fti[key] = (c_f, ph_f, 1)

        # Scan the full grid at probe qq (very fast)
        from collections import defaultdict
        probe_e_bounds: dict[int, tuple[float, float]] = {}  # m → (e_min, e_max)
        probe_m_max = 0
        for m0 in range(m_start, probe_m_scan + 1, m_step):
            row_es: list[float] = []
            for e_half in range(-2 * probe_e_scan, 2 * probe_e_scan + 1):
                e0 = Fraction(e_half, 2)
                entry = _compute_one_kernel_entry(
                    m0, e0, hj_ks, probe_qq_internal, eta_order,
                    probe_m1_range, probe_fti,
                )
                if entry is not None:
                    row_es.append(e_half)
            if row_es:
                probe_e_bounds[m0] = (min(row_es), max(row_es))
                probe_m_max = max(probe_m_max, m0)

        clear_filling_caches()  # free probe caches

        # Scale probe bounds to target qq
        scale = qq_internal / probe_qq_internal if probe_qq_internal > 0 else 5.0
        scaled_m_max = int(probe_m_max * scale * _WIDTH_MARGIN) + 2

        # Build per-m e-bounds by interpolating the probe shape
        # Uses center + half-width scaling: the center shifts linearly
        # but the half-width gets a multiplicative safety margin.
        e_bounds: dict[int, tuple[int, int]] = {}
        probe_ms = sorted(probe_e_bounds.keys())
        for m0 in range(m_start, min(m_scan, scaled_m_max) + 1, m_step):
            # Find the nearest probe m (normalised coordinates)
            m_norm = m0 / scale
            # Binary search for nearest probe row
            best_lo, best_hi = None, None
            for pm in probe_ms:
                if pm <= m_norm:
                    best_lo = pm
                if pm >= m_norm and best_hi is None:
                    best_hi = pm
            if best_lo is None:
                best_lo = best_hi
            if best_hi is None:
                best_hi = best_lo
            if best_lo is None:
                # No probe data — use full range
                e_bounds[m0] = (-2 * e_scan, 2 * e_scan)
                continue

            # Interpolate e-bounds from nearest probe rows
            lo_emin, lo_emax = probe_e_bounds[best_lo]
            hi_emin, hi_emax = probe_e_bounds[best_hi]
            if best_hi != best_lo:
                t = (m_norm - best_lo) / (best_hi - best_lo)
            else:
                t = 0.0
            interp_emin = lo_emin * (1 - t) + hi_emin * t
            interp_emax = lo_emax * (1 - t) + hi_emax * t

            # Center + half-width scaling (tighter than uniform factor)
            center = (interp_emin + interp_emax) / 2.0
            half_w = (interp_emax - interp_emin) / 2.0
            scaled_center = center * scale
            scaled_hw = half_w * scale * _WIDTH_MARGIN + _MARGIN_ABS
            e_lo = int(scaled_center - scaled_hw) - 1
            e_hi = int(scaled_center + scaled_hw) + 1

            # Clamp to absolute bounds
            e_lo = max(e_lo, -2 * e_scan)
            e_hi = min(e_hi, 2 * e_scan)
            e_bounds[m0] = (e_lo, e_hi)

        target_m_values = sorted(e_bounds.keys())
        total_pts = sum(hi - lo + 1 for lo, hi in e_bounds.values())
        full_pts = len(target_m_values) * (4 * e_scan + 1)
        _status(
            f"[kernel]   → probe found {len(probe_e_bounds)} non-empty rows "
            f"(m_max={probe_m_max}), "
            f"scale={scale:.2f}, "
            f"target grid: {total_pts} pts "
            f"(vs {full_pts} full = {total_pts/full_pts*100:.0f}%)"
        )
    else:
        # Small qq — no probe, just full scan
        target_m_values = list(range(m_start, m_scan + 1, m_step))
        e_bounds = {m: (-2 * e_scan, 2 * e_scan) for m in target_m_values}
        total_pts = sum(hi - lo + 1 for lo, hi in e_bounds.values())

    # ------------------------------------------------------------------
    # Phase 3: Compute kernel entries (m ≥ 0 only, symmetry)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    kernel_table: dict[tuple[int, Fraction], QEtaSeries] = {}

    use_parallel = n_workers >= 2 and total_pts >= 500

    _status(f"[kernel] Phase 3: computing {total_pts} grid points (m ≥ 0), "
            f"workers={'parallel ×' + str(n_workers) if use_parallel else 'serial'}")

    if use_parallel:
        # Row-based dispatch: assign complete m-rows to workers
        # Round-robin by row for load balance
        worker_rows: list[list[int]] = [[] for _ in range(n_workers)]
        for i, m0 in enumerate(target_m_values):
            worker_rows[i % n_workers].append(m0)

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
        # Serial path
        computed = 0
        for m0 in target_m_values:
            e_lo, e_hi = e_bounds[m0]
            for e_half in range(e_lo, e_hi + 1):
                e0 = Fraction(e_half, 2)
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
