# Phase 10 — Refined Dehn Filling

> **File:** `src/manifold_index/core/refined_dehn_filling.py`
> **Depends on:** phases 6 (index_3d), 7 (refined_index), 9 (dehn_filling)
> **Notation:** see `CONVENTIONS.md` §Doubled-exponent convention

---

## 0. Purpose

Compute the **refined Dehn filling kernel** K^ref(P,Q; m,e; η^{2V})
and apply it to the refined 3D index I^ref(m,e; η^{2W}) to produce
the filled refined index Î^ref_{P/Q}(η^{2W}, η^{2V}).

This is the largest and most complex module (~2900 lines in v0.3).

---

## 1. Mathematical Specification

### 1.1 Hirzebruch-Jung Continued Fraction (HJ-CF)

Given coprime P/Q, find the **shortest** representation:

    P/Q = k₁ − 1/(k₂ − 1/(… − 1/kₗ))

Special cases:
- Q = 0, P = ±1 → [0, 0]  (longitude/meridian)
- |Q| = 1       → [P/Q]   (length 1, unrefined kernel suffices)

> **Q=0 degenerate case (implementation note):** Although `hj_continued_fraction`
> returns `[0,0]` (ℓ=2) for Q=0, the IS-chain path is **incorrect** for this case.
> The IS-chain introduces a spurious intermediate cusp-η dimension and produces a
> non-zero refined filling even when the ordinary filling is zero (e.g. 5_1 at 1/0),
> because the intermediate (m₁,e₁) charges do not inherit the cancellation structure
> of I^ref(0,0) = I^ref(2,10).
> **Fix:** In `compute_filled_refined_index`, override `hj_ks = [P], ell = 1` when
> `Q = 0`, routing through the direct ℓ=1 kernel path.

Algorithm for shortest form:
1. Q = 0: return `[0, 0]`
2. Q = 1: return `[P]`
3. **Try length-2:** search integer divisors d of Q (including negatives: ±1, ±Q, ±factors)
   with `(P + d) % Q == 0`.  For each valid d, set `k₁ = (P + d) // Q` and `k₂ = Q // d`.
   Return `[k₁, k₂]` for the first valid d found (prefer smallest |k₂|).

   *Why this works:* P/Q = k₁ − 1/k₂ ⟺ k₂·P = k₁·k₂·Q − Q ⟺ P + Q/k₂ = k₁·Q.
   For k₂ integer we need k₂ | Q, and then k₁ = (P + Q/k₂) / Q = (P + d) / Q
   where d = Q/k₂ is a divisor of Q.

   Example: P=1, Q=3 → divisors of 3: ±1, ±3.
   d=−1: (1+(−1)) % 3 = 0 ✓ → k₁=(1−1)/3=0, k₂=3/(−1)=−3 → [0, −3] ✓.
4. General: compute both ceiling-based and nearest-integer-rounding CFs,
   return the shorter one.

### 1.2 Kernel Chain (eq. A.7)

    K^ref(P,Q; m,e; η) =
        Σ_{m₁,e₁} … Σ_{m_{ℓ-1},e_{ℓ-1}}
            I_S(m,  −e − k₁/2·m,   m₁, e₁)
          · I_S(m₁, −e₁ − k₂/2·m₁, m₂, e₂)
          · …
          · K(kₗ, 1; m_{ℓ-1}, e_{ℓ-1})

where K(·,1;·,·) is the unrefined kernel (phase 9) and I_S is the
symplectic kernel.

### 1.3 I_S Kernel (eq. A.5)

    I_S(m₁, e₁, m₂, e₂; η) =
        ½·(−1)^{m₁}·(q^{m₁/2} + q^{−m₁/2}) · ẽI_S(m₁, e₁,   m₂, e₂)
      − ½·(−1)^{m₁} · ẽI_S(m₁, e₁−1, m₂, e₂)
      − ½·(−1)^{m₁} · ẽI_S(m₁, e₁+1, m₂, e₂)

### 1.4 ẽI_S Inner Kernel (DFK.nb `expr8[]`)

    ẽI_S(m₁, e₁, m₂, e₂; η) =
        Σ_{e ∈ Z, t ∈ Z}  η^e
        · I_Δ(−e₁ − m₂/2,   −e/2 + e₁ + m₁/2 + t)
        · I_Δ( e₁ + m₂/2,   −e/2 + e₂ − m₂/2 + t)
        · I_Δ(−e₂ − m₁/2,    e₂ + m₁/2 + t)
        · I_Δ( e₂ + m₁/2,    e₁ − m₂/2 + t)
        · (−q^{1/2})^{−e + e₁ + e₂ + m₁/2 − m₂/2 + 2t}

Integrality filters:
- m_a1 = −e₁ − m₂/2 must be integer
- m_a3 = −e₂ − m₁/2 must be integer
- e_var parity = (m₁ + m₂) mod 2

---

## 2. Type Aliases

```python
QEtaSeries = dict[tuple[int, int], Fraction]
# key = (qq_power, eta_exp) → Fraction coefficient

MultiEtaSeries = dict[tuple[int, ...], Fraction]
# key = (qq, 2W_0, …, 2W_{H-1} [, 2V_0, …]) → Fraction
# For ℓ=1: no cusp η dimension
# For ℓ≥2: one cusp η appended as last dimension
```

---

## 3. Module Structure (9 parts)

### Part 1: `hj_continued_fraction`

```
hj_continued_fraction(P: int, Q: int) -> list[int]
```

- **Signature:** `(P, Q) → [k₁, …, kₗ]`
- **Helper `_hj_cf_ceil`:** classical ceiling algorithm, O(Q) length
- **Helper `_hj_cf_round`:** nearest-integer rounding, O(log Q) length

### Part 2: QEtaSeries Arithmetic

Small pure-function helpers:

| Function | Signature | Notes |
|---|---|---|
| `_qeta_add` | `(a, b) → QEtaSeries` | Non-destructive add |
| `_qeta_scale` | `(s, scalar) → QEtaSeries` | Multiply all coeffs |
| `_qeta_shift_qq` | `(s, shift) → QEtaSeries` | q^{shift/2} multiply |
| `_qeta_truncate` | `(s, qq_order) → QEtaSeries` | Keep qq ≤ limit |
| `_qeta_convolve` | `(a, b, qq_order) → QEtaSeries` | Polynomial multiply |
| `_tet_series_to_qeta` | `(s, eta_exp) → QEtaSeries` | int dict → QEta |
| `_int_qqseries_convolve` | `(a, b, qq_order) → dict[int,int]` | Pure int path |

### Part 3: ẽI_S Kernel — `_etilde_is`

```
@functools.lru_cache(maxsize=None)
_etilde_is(m1, e1, m2, e2, qq_order, eta_order) -> dict[(int,int), int]
```

**Algorithm:**
1. Outer integrality checks on m_a1, m_a3, parity
2. Precompute base args for tind3, tind4 (t-independent first args)
3. Double sum over t ∈ Z, n_eta ∈ [−eta_order, eta_order]
4. **Performance:** Uses dense numpy arrays via `_tet_index_array`:
   - Cache tetrahedron index as int64 arrays: `_tet_arr_cache`
   - Convolve s3·s4 with `np.convolve`
   - FFT-batched path for N_batch ≥ 4: stack s12 arrays, single `rfft` pass
   - Scalar path for small batches
5. Returns sparse dict[(qq, eta) → int]

   **Why the values are always integers** (not merely empirical):
   - Each I_Δ(m, e) series has integer coefficients (by construction in Phase 6).
   - The phase factor `(−q^{1/2})^{exponent}` contributes ±1 to every coefficient
     (the exponent is always an integer due to the integrality filters in step 1:
     m_a1 integer ⟹ −m₂/2 is a half-integer offset that cancels with e₁ integer;
     m_a3 integer gives the same guarantee for the other pair; parity filter ensures
     the e_var term is even).
   - The full product of four I_Δ series and the phase factor therefore has integer
     coefficients, and the η-exponent is always an integer.
   Add a debug-mode assertion `assert all(isinstance(v, int) for v in result.values())`
   when first porting this function.

**Dense cache:**
```python
_tet_arr_cache: dict[tuple[int, int, int], np.ndarray] = {}

def _tet_index_array(m, e, qq_order) -> np.ndarray:
    # Returns int64 array of length qq_order+1, or _EMPTY_ARR sentinel
```

### Part 4: I_S Kernel — `_is_kernel`

```
@functools.lru_cache(maxsize=None)
_is_kernel(m1, e1, m2, e2, qq_order, eta_order) -> dict[(int,int), int]
```

Returns **2 × I_S** to avoid Fraction arithmetic.  Combines three
`_etilde_is` calls with qq shifts and sign.  Uses numpy 2D accumulator.

**Fraction wrapper:**
```python
_is_kernel_frac(…) -> QEtaSeries  # Calls _is_kernel, divides by 2
```

### Part 5: K(k,1) Support Enumeration

| Function | Use case |
|---|---|
| `_enumerate_slope1_terms(k, t_range)` | ℓ=1 path: c ∈ {0,2}, ±t symmetry |
| `_enumerate_slope1_all(k, t_range)` | ℓ≥2 final K: c ∈ {−2,0,2}, all t |
| `_enumerate_slope1_all_halfshift(k, t_range)` | Half-integer e for odd-m sources |
| `_enumerate_is_full(m1_range, e1_range)` | Full (½)Z² lattice for intermediate IS steps |

All return `list[tuple[int, Fraction, int, int]]` = (m, e, c, phase).

### Part 6: K-Factor Application

```python
_apply_k1_factor(is_series, m1, e1, c, phase, multiplicity, qq_order) -> QEtaSeries
_apply_k1_factor_multi(series, c, phase, multiplicity, qq_order,
                       truncate=True, int_mode=False) -> MultiEtaSeries
```

K(k,1) factor for R=1, S=0:
- c=0: `½·(−1)^{phase}·(qq^{phase} + qq^{−phase})`
- c=±2: `−½·(−1)^{phase}`

`int_mode=True`: absorbs the ½ factor (caller tracks LCD = 2^ℓ).

### Part 6b: MultiEtaSeries Helpers

| Function | Purpose |
|---|---|
| `_multi_add(a, b)` | Polymorphic add (Fraction or int) |
| `_multi_convolve_is(is_series, multi_series, qq_order)` | IS η maps to last dim |
| `_apply_weyl_shift(refined, m_ext, e_ext, weyl_a, weyl_b, …)` | η^{a·e_I + b·m_I} |
| `_refined_to_multi(refined, append_cusp_eta, use_int)` | RefinedIndexResult → Multi |

### Part 7: Single IS Step — `_apply_is_step`

```python
_apply_is_step(state, k_current, k_next, qq_order, eta_order, m1_range,
               use_int=False, is_last_step=True)
    -> dict[(int, Fraction), MultiEtaSeries]
```

Maps state[(m,e)] → new_state[(m₁,e₁)] via I_S convolution.

**Key optimisations:**
- e-transform: `e_in = −e − k_current/2·m`
- Parity pre-filter: 2-way (last step) or 4-way (intermediate)
- `use_int=True`: uses `_is_kernel` (×2 int) for 3-5× speedup
- `is_last_step=True`: restricts targets to K(k_next,1) support
- `is_last_step=False`: full (½)Z² lattice for intermediate steps

### Part 8: Main Function — `compute_filled_refined_index`

```python
@dataclass
class FilledRefinedResult:
    P: int
    Q: int
    cusp_idx: int
    series: MultiEtaSeries
    qq_order: int
    eta_order: int
    hj_ks: list[int]
    n_kernel_terms: int
    num_hard: int
    has_cusp_eta: bool
    num_cusp_eta: int = 0

    # Methods: is_zero, collapse_eta_edges, eta1_series,
    #          q_series_at_eta, as_q_eta_string
```

**Algorithm in `compute_filled_refined_index`:**

1. **HJ-CF** → `hj_ks`, `ell`
2. **ℓ=1 path** (no IS chain):
   - Enumerate K(k₁,1) support via `_enumerate_slope1_terms`
   - For each (m,e): compute I^ref, apply Weyl shift, apply K-factor
   - Multiplicity: 2 for c=2 or c=0 with t≠0, else 1
   - Result has NO cusp η dimension
3. **ℓ≥2 fast path**: Check `load_kernel_table` → `apply_precomputed_kernel`
4. **ℓ≥2 auto-precompute**: If `auto_precompute=True`, build and save kernel
5. **ℓ≥2 fallback** (grid scan + IS chain):
   - `qq_internal = qq_order + qq_order//2 + 4` (buffer)
   - Grid scan: m ∈ [−2·qq_int, 2·qq_int], e ∈ half-integers
   - Apply ℓ−1 IS steps via `_apply_is_step`
   - Apply final K(kₗ,1) via `_apply_k1_factor_multi` with `int_mode=True`
   - LCD = 2^ℓ (divide back to Fraction at end).
     **Use Python `int` throughout** — do NOT use numpy int64 for the IS chain
     accumulator.  For ℓ = 20 (a long HJ-CF), LCD = 2^20 ≈ 1M; int64 overflows
     at ℓ ≥ 63 and silently wraps.  Python's arbitrary-precision int has no such
     limit.  Only convert to numpy arrays at the final dense-array convolution step.
   - **Diamond truncation**: qq + |cusp_eta| ≤ qq_order

**Cache infrastructure:**
```python
_iref_cache: dict[tuple, dict] = {}  # Content-keyed by _nz_content_key

def _nz_content_key(nz_data) -> tuple:
    return (g_NZ.data.tobytes(), nu_x.data.tobytes(), nu_p.data.tobytes())

def _cached_compute_refined_index(nz_data, m_ext, e_ext, q_order_half):
    # Memoised wrapper around compute_refined_index

def clear_filling_caches() -> dict[str, int]:
    # Clears _etilde_is, _is_kernel, _iref_cache, _tet_arr_cache, kernel_cache
```

### Part 9: Multi-Cusp Sequential Filling

```python
@dataclass
class MultiCuspFillSpec:
    cusp_idx: int
    P: int
    Q: int
    weyl_a: list[Fraction] | None = None
    weyl_b: list[Fraction] | None = None
    incompat_edges: list[int] | None = None
```

**`_apply_filling_kernel_to_intermediate`:**
- Applies filling kernel to pre-existing intermediate series
- Same math as `compute_filled_refined_index` but input is
  dict[(m,e) → MultiEtaSeries] instead of NZ data
- For ℓ=1: no new cusp η added
- For ℓ≥2: extends keys with new cusp η dimension
- Generalised diamond truncation: qq + Σ|cusp_eta_i| ≤ qq_order

**`compute_multi_cusp_filled_refined_index`:**
- Currently supports up to 2 cusps (raises for >2)
- Step 1: `_batched_first_filling` — computes I^ref for all spectator charges
- Step 2: `_apply_filling_kernel_to_intermediate` for second cusp

**`_batched_first_filling`:**
- For ℓ=1: per-spectator delegation (no IS chain to amortise)
- For ℓ≥2: probe-based filtering + per-spectator delegation
  - Probe: fix (m₀=0, e₀=0) for filling cusp, scan all spectator (m,e)
  - Intersect probed non-zero set with second filling's needed charges
  - Call `compute_filled_refined_index` once per active spectator

---

## 4. Implementation Order

1. HJ continued fraction + tests
2. QEtaSeries arithmetic helpers
3. `_tet_index_array` dense cache
4. `_etilde_is` (scalar path first, then numpy FFT-batched)
5. `_is_kernel` and `_is_kernel_frac`
6. K(k,1) enumeration functions
7. K-factor application + MultiEtaSeries helpers
8. `_apply_is_step`
9. `FilledRefinedResult` dataclass
10. `compute_filled_refined_index` (ℓ=1 first, then ℓ≥2 fallback)
11. Cache infrastructure (`_iref_cache`, `clear_filling_caches`)
12. Multi-cusp: `MultiCuspFillSpec`, `_apply_filling_kernel_to_intermediate`,
    `_batched_first_filling`, `compute_multi_cusp_filled_refined_index`

---

## 5. Tests

### T10.1 — HJ Continued Fraction

```python
assert hj_continued_fraction(1, 3) == [0, -3]
assert hj_continued_fraction(5, 2) == [2, -2]   # 5/2 = 2 - 1/(-2); shortest form
assert hj_continued_fraction(1, 1) == [1]
assert hj_continued_fraction(1, 0) == [0, 0]  # Q=0 special
assert hj_continued_fraction(-1, 0) == [0, 0]
assert hj_continued_fraction(4, 3) == [1, -3]
```

Verify roundtrip: for each test case, evaluate the CF and check it equals P/Q.

### T10.2 — ẽI_S Integrality

For m₁=0, e₁=0, m₂=0, e₂=0, qq_order=10, eta_order=10:
- All values in result must be integers
- Result should be non-empty

### T10.3 — I_S ×2 Scaling

For any (m₁,e₁,m₂,e₂), verify `_is_kernel` returns all-even or all-odd
values (confirming 2×I_S is integral).

### T10.4 — ℓ=1 Filling (m003, P=5, Q=2)

```python
nz = build_neumann_zagier(load_manifold("m003"))
nz_bc = apply_cusp_basis_change(nz, 0, 1, 0)  # NC cycle
result = compute_filled_refined_index(nz_bc, 0, 5, 2, q_order_half=8)
assert not result.is_zero
assert result.has_cusp_eta is False  # ℓ=1 → no cusp η
```

### T10.5 — ℓ≥2 Filling (m003, P=1, Q=2)

```python
result = compute_filled_refined_index(nz_bc, 0, 1, 2, q_order_half=8)
assert result.has_cusp_eta is True
assert result.num_cusp_eta == 1
# Diamond truncation: all keys satisfy k[0] + |k[-1]| ≤ 8
for k in result.series:
    assert k[0] + abs(k[-1]) <= 8
```

### T10.6 — LCD Consistency

For ℓ≥2, verify that LCD = 2^ℓ produces all-integer numerators
after multiplication.

### T10.7 — Cache Content-Key Safety

Create two NZ objects with different basis changes.  Verify
`_cached_compute_refined_index` returns different results for each.

---

## 6. Performance Notes

- `_etilde_is` and `_is_kernel` use `@lru_cache(maxsize=None)` — clear
  when switching manifolds via `clear_filling_caches()`
- `_tet_index_array` uses a module-level dict cache (`_tet_arr_cache`)
- FFT-batched convolution in `_etilde_is` when **N_batch ≥ 4 OR qq_order ≥ 50**
  (FFT beats O(n²) convolution even for N_batch=1 at large qq_order)
- `int_mode=True` in IS chain gives 3-5× speedup over Fraction path
- Diamond truncation prevents unreliable high-η artifacts

---

## 7. Acceptance Criteria

- [ ] `hj_continued_fraction` passes all T10.1 cases
- [ ] `_etilde_is` returns integer values for all tested inputs
- [ ] ℓ=1 and ℓ≥2 paths produce non-zero results for m003
- [ ] Diamond truncation enforced on all ℓ≥2 results
- [ ] `clear_filling_caches` evicts all caches and returns counts
- [ ] Multi-cusp filling works for 2-cusp manifold (e.g. 5_1^2)
