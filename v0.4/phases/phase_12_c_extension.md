# Phase 12 — C Extension

> **File:** `src/manifold_index/core/_c_kernel/tet_index.c`
> **Build:** setuptools C extension via `pyproject.toml`
> **Depends on:** phase 6 (index_3d) — provides the pure-Python reference

---

## 0. Purpose

Drop-in C replacement for the hot-path tetrahedron index functions.
5–50× faster than Python.  Results are **bit-identical** to the Python
implementation.  Falls back to pure Python automatically if the
extension is unavailable.

---

## 1. Module Name and Entry Point

```c
PyMODINIT_FUNC PyInit__c_tet_index(void)
```

Module name: `_c_tet_index`.

---

## 2. Exported Functions

### 2.1 `tet_index_series(m, e, qq_order) → dict[int, int]`

Full MIt(m,e) with symmetry.  Algorithm:

1. If `m + e ≥ 0`:
   - `raw = it_direct(-m-e, m, qq_order - m)`
   - Shift keys by +m, multiply coefficients by `(-1)^m`
2. Else:
   - `raw = it_direct(m, e, qq_order)`

Build Python dict from non-zero entries.

### 2.2 `tet_degree_x2(m, e) → int`

Returns `2 × tet_degree(m, e)` as plain integer.

```c
int half_sum = pos_m * pos_me + pos_nm * pos_e + pos_ne * pos_nem;
int mx = max(0, max(m, -e));   /* C max() is 2-arg; chain calls for 3-way */
return half_sum + 2 * mx;
```

### 2.3 `poly_convolve(poly1, poly2, budget) → dict[int, int]`

Multiply two sparse polynomials (Python dicts) with power ≤ budget.

Implementation:
1. Convert both dicts to dense `long long` arrays of length `budget+1`
2. Dense O(n²) convolution with budget cutoff
3. Build Python dict from non-zero results

---

## 3. Internal: `it_direct(mm, ee, inner_order)`

Raw I_t series computation.  Returns dense `long long` array.

### 3.1 inv_fact Table

`inv_fact[k] = 1 / prod_{j=1}^{k} (1 − qq^{2j})`

Stored as dense int64 arrays of length `inner_order + 1`.
Extended incrementally via `extend_inv_fact`:

```c
// inv_fact[k] = inv_fact[k-1] * (1 + qq^{2k} + qq^{4k} + ...)
for (int p = 0; p < poly_len; p++) {
    if (prev[p] == 0) continue;
    for (int q = p; q < poly_len; q += step)
        new_poly[q] += prev[p];
}
```

### 3.2 Main Loop

```c
for (int n = n_min; ; n++) {
    exp_qq = n*(n+1) - (2*n+ee)*mm;
    if (exp_qq > inner_order) break;

    // d1 = inv_fact[n], d2 = inv_fact[n+ee]
    // sign = (-1)^n
    // Convolve d1 * d2, shift by exp_qq, accumulate into result
}
```

---

## 4. Memory Management

- `inv_fact_table`: dynamically grown pointer array (`realloc`)
- Each `inv_fact[k]`: `calloc(poly_len, sizeof(long long))`
- `free_inv_fact(table, count)` releases everything
- Result arrays: caller must free

---

## 5. Python Integration

### 5.1 Fallback Loading (in `index_3d.py`)

```python
try:
    from manifold_index.core._c_kernel._c_tet_index import (
        tet_index_series as _c_tet_index_series,
        tet_degree_x2 as _c_tet_degree_x2,
    )
    _USE_C = True
except ImportError:
    _USE_C = False
```

When `_USE_C` is True, `_tet_index_series` dispatches to C.

### 5.2 Build Configuration

In `pyproject.toml` (or `setup.py`):

```toml
[tool.setuptools]
ext-modules = [
    {name = "manifold_index.core._c_kernel._c_tet_index",
     sources = ["src/manifold_index/core/_c_kernel/tet_index.c"]}
]
```

---

## 6. Tests

### T12.1 — Bit-Identical Output

For a grid of (m, e) ∈ [−5, 5]² and qq_order ∈ {10, 20, 50}:
```python
py_result = _tet_index_series_python(m, e, qq_order)
c_result = _c_tet_index_series(m, e, qq_order)
assert py_result == c_result
```

### T12.2 — tet_degree_x2 Agreement

```python
for m in range(-10, 11):
    for e in range(-10, 11):
        assert _c_tet_degree_x2(m, e) == _py_tet_degree_x2(m, e)
```

### T12.3 — poly_convolve

```python
a = {0: 1, 2: -1, 4: 1}
b = {0: 1, 1: 1}
result = _c_poly_convolve(a, b, 5)
# Check against Python dict convolution
```

### T12.4 — Fallback Works

When `_c_tet_index` is not importable, the pure-Python path
must still produce correct results.

### T12.5 — Performance Benchmark

```python
import timeit
# C version should be 5-50× faster than Python for qq_order=50
t_c = timeit.timeit(lambda: c_fn(0, 3, 50), number=1000)
t_py = timeit.timeit(lambda: py_fn(0, 3, 50), number=1000)
assert t_c < t_py / 3  # At least 3× faster
```

---

## 7. Acceptance Criteria

- [ ] C extension compiles on macOS, Linux (gcc/clang)
- [ ] All three functions are bit-identical to Python
- [ ] Fallback to pure Python works when extension unavailable
- [ ] No memory leaks (valgrind clean or AddressSanitizer)
- [ ] `poly_convolve` handles empty dicts gracefully
