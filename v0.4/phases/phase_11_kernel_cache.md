# Phase 11 — Kernel Cache

> **File:** `src/manifold_index/core/kernel_cache.py`
> **Depends on:** phase 10 (refined_dehn_filling)

---

## 0. Purpose

Pre-compute the **manifold-independent** refined Dehn filling kernel
K^ref(P/Q; m,e; η^{2V}) and persist it to disk.  Applying a cached
kernel turns a ~10 min IS-chain computation into a sub-second lookup.

Also manages the I^ref disk cache (manifold-dependent, slope-independent).

---

## 1. Storage Layout

### 1.1 Kernel Files

**Bundled (read-only):**
```
src/manifold_index/data/kernel_cache/kernel_P{P}_Q{Q}_qq{qq}.pkl.gz
```

**User cache (writable, runtime-generated):**
```
~/Library/Caches/manifold-index/kernel_cache/   (macOS)
~/.cache/manifold-index/kernel_cache/           (Linux)
%LOCALAPPDATA%/manifold-index/kernel_cache/     (Windows)
```

Lookup order: user cache → bundled.  Higher-order kernels are valid
supersets (caller's diamond truncation discards excess terms).

### 1.2 I^ref Cache Files

```
~/Library/Caches/manifold-index/iref_cache/iref_{name}_{hash16}.pkl.gz
```

Content-hash based on `(g_NZ.tobytes(), nu_x.tobytes(), nu_p.tobytes())`.
Merges with existing file on save.

---

## 2. Data Structures

### 2.1 KernelTable

```python
@dataclass
class KernelTable:
    P: int
    Q: int
    qq_order: int           # user-facing truncation
    qq_internal: int        # inflated internal (qq_order + buffer)
    eta_order: int          # max |cusp V|
    hj_ks: list[int]       # HJ-CF coefficients
    table: dict[tuple[int, Fraction], QEtaSeries]
    m_scan: int
    e_scan: int
    compute_time_s: float = 0.0

    # Fast representation (survives pickling):
    _fast_lcd: int = 0
    _fast_grouped: dict | None = None
```

**`get_int_grouped()`** returns `(lcd, grouped)`:
- `grouped[(m,e)][eta_cusp] = (min_qq, np.ndarray[int64])`
- Dense arrays scaled by LCD, offset by min_qq
- Built lazily on first call, persisted via `ensure_fast_repr()`

> **Lazy loading note:** `get_int_grouped()` materialises **all** (m,e) entries
> as dense int64 arrays in one shot.  For large kernels (e.g. P=1, Q=7+) this
> can require hundreds of MB.  If `apply_precomputed_kernel` only needs a subset
> of entries (e.g. after degree-bound pruning), prefer to iterate `kernel.table`
> directly for that subset and convert on-demand, rather than calling
> `get_int_grouped()` upfront.  The fast path in `apply_precomputed_kernel` should
> call `get_int_grouped()` only when the caller has already confirmed the kernel
> is small enough (use `len(kernel.table)` as a proxy).

### 2.2 Degree-Bound Helpers

Pure-function degree analysis to prune the (m,e) grid:

| Function | Purpose |
|---|---|
| `_tet_degree_x2(m, e)` | 2× tetrahedron degree, scalar |
| `_tdeg_arr(m, e)` | Vectorised numpy version |
| `_is_kernel_min_degree_x2(…)` | Lower bound on IS kernel output degree |
| `_degree_feasible_row(m0, k1, …)` | Vectorised per-row feasibility |
| `_compute_degree_bounds(…)` | Full degree-bound analysis with backward reachability |

---

## 3. File I/O

```python
def save_kernel_table(kt: KernelTable, cache_dir=None) -> Path:
    # Eagerly builds fast repr, gzip-pickles to disk

def load_kernel_table(P, Q, qq_order, cache_dir=None) -> KernelTable | None:
    # In-memory cache → user cache → bundled; exact match then fallback

def clear_kernel_cache() -> int:
    # Clear in-memory _kernel_mem_cache

def list_cached_kernels(cache_dir=None) -> list[tuple[int,int,int]]:
    # List all (P, Q, qq_order) tuples from both user cache and bundled
```

In-memory cache: `_kernel_mem_cache: dict[tuple, KernelTable | None]`.

---

## 4. Pre-Computation: `precompute_filling_kernel`

```python
def precompute_filling_kernel(P, Q, qq_order, eta_order=None,
                              verbose=False, progress_callback=None,
                              n_workers=None) -> KernelTable
```

**4-Phase Algorithm:**

### Phase 1: Parity Auto-Detection
- Probe m=0..3 with several e values
- Determine if only even-m or odd-m produce non-zero entries
- Sets `m_step=2` if single parity → 2× speedup

### Phase 2: Degree-Bound Support Analysis
- Uses exact `_tet_degree_x2` formula to determine feasible (m₀, e₀) grid
- Backward reachability for ℓ ≥ 3 chains
- Typically retains 14–19% of full grid

### Phase 3: Compute Kernel Entries (m ≥ 0 only)

**Pilot-gated parallelism:**
1. Flush computation caches for cold-cache measurement
2. Stratified sample of ~20 grid points
3. Estimate serial time → enable parallel if > 60s

**Serial path:** iterate over `target_m_values`, `e_bounds[m]`

**Parallel path:** chunk-based work queue
- Chunks of ~50 points, contiguous within m-rows (LRU locality)
- `ProcessPoolExecutor` with fork context
- Work-stealing via `as_completed`

**Per-point computation** (`_compute_one_kernel_entry`):
1. Unit delta state: `{(m₀, e₀): {(0,0): 1}}`
2. ℓ−1 IS steps via `_apply_is_step(use_int=True)`
3. Final K-factor via `_apply_k1_factor_multi(int_mode=True, truncate=False)`
4. LCD = 2^ℓ, convert back to Fraction

### Phase 4: Mirror Symmetry
- K^ref(m,e) = K^ref(−m,−e) always holds
- Only m ≥ 0 computed; results mirrored → 2× speedup

---

## 5. Fast Application: `apply_precomputed_kernel`

```python
def apply_precomputed_kernel(kernel, nz_data, cusp_idx, m_other=None,
                             e_other=None, weyl_a=None, weyl_b=None,
                             qq_order=None, verbose=False, n_workers=1,
                             cache_iref=False, manifold_name="unknown"
                             ) -> MultiEtaSeries
```

**Algorithm:**
1. Get `(lcd, k_grouped)` from `kernel.get_int_grouped()`
2. Optionally load I^ref disk cache
3. Pre-compute all I^ref results (sequential or parallel workers)
4. For each kernel entry (m,e):
   - Look up I^ref, apply Weyl shift
   - Group I^ref by η^{2W} pattern → dense int64 arrays
   - Batched convolution (stack kern arrays for N_ec ≥ 4):
     - For each nonzero position j in iref: spread into conv_results
   - Scalar np.convolve for small N_ec
5. Convert dense accumulators → sparse Fraction dict (÷ lcd)
6. Optionally save I^ref disk cache

---

## 6. I^ref Disk Cache

```python
def save_iref_cache(nz_data, manifold_name, cache_dir=None) -> Path | None
def load_iref_cache(nz_data, manifold_name, cache_dir=None, qq_filter=None) -> int
def list_iref_caches(cache_dir=None) -> list[dict[str, Any]]
```

- Content-hash: SHA-256 of (g_NZ + nu_x + nu_p), first 16 hex chars
- Merges on save (preserves entries at other qq_orders)
- `qq_filter` avoids loading unnecessary entries

---

## 7. Worker Functions (module-level for pickle)

```python
def _compute_one_kernel_entry(m0, e0, hj_ks, …) -> QEtaSeries | None
def _worker_compute_chunk(points, hj_ks, …) -> (dict, int)
def _iref_worker_init(nz_data) -> None
def _iref_worker_fn(args) -> (tuple, tuple, dict)
```

All module-level for multiprocessing pickle compatibility.

---

## 8. Tests

### T11.1 — Save/Load Round-Trip

```python
kt = precompute_filling_kernel(1, 2, qq_order=8, n_workers=1)
path = save_kernel_table(kt)
loaded = load_kernel_table(1, 2, 8)
assert loaded is not None
assert loaded.P == 1 and loaded.Q == 2
assert len(loaded.table) == len(kt.table)
```

### T11.2 — Fast Repr Persistence

```python
kt.ensure_fast_repr()
lcd, grouped = kt.get_int_grouped()
assert lcd > 0
assert len(grouped) == len(kt.table)
# Verify all dense arrays are int64
for by_eta in grouped.values():
    for _, (_, arr) in by_eta.items():
        assert arr.dtype == np.int64
```

### T11.3 — Apply vs Direct Comparison

```python
# For m003 with P=1, Q=2, qq=8:
# Direct computation and kernel-application must agree
result_direct = compute_filled_refined_index(nz_bc, 0, 1, 2, q_order_half=8)
result_cached = apply_precomputed_kernel(kt, nz_bc, 0, qq_order=8)
# Compare after diamond truncation
# (exact match up to diamond boundary)
```

### T11.4 — Higher-Order Kernel Fallback

A kernel at qq=20 should be loadable when qq=8 is requested.

### T11.5 — Mirror Symmetry

```python
for (m, e), entry in kt.table.items():
    mirror = kt.table.get((-m, -e))
    if m != 0 or e != 0:
        assert mirror == entry
```

---

## 9. Acceptance Criteria

- [ ] `precompute_filling_kernel` produces non-empty KernelTable for P=1,Q=2
- [ ] Save/load round-trip preserves all entries
- [ ] `apply_precomputed_kernel` matches direct computation
- [ ] In-memory cache avoids repeated disk reads
- [ ] Platform-specific cache dirs work on macOS/Linux/Windows
- [ ] I^ref disk cache reduces multi-slope computation time
