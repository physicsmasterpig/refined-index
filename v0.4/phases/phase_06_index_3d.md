# Phase 6: 3D Index Kernel (I_Δ + Summation)

## Dependencies
- Phase 4: `NeumannZagierData`
- External: `numpy`, `fractions.Fraction`, `itertools.product`

## Files to Create
- `src/manifold_index/core/index_3d.py`
- `tests/test_index_3d.py`

---

## Notation Warning (paper ↔ code swap)

| Paper | Code | Meaning |
|-------|------|---------|
| r     | n    | number of tetrahedra |
| n     | r    | number of cusps |
| κ     | kappa| combined (m, e) vector of size 2n |

---

## Part A: Tetrahedron Index I_Δ(m, e; qq_order)

### Formula (Garoufalidis–Kim)

Raw series `I_t(m, e)`:
```
I_t(m, e) = Σ_{n=n_min}^∞  (-1)^n · qq^{n(n+1) - (2n+e)·m}
             ─────────────────────────────────────────────────
             ∏_{k=1}^{n}(1 - qq^{2k}) · ∏_{k=1}^{n+e}(1 - qq^{2k})
```
where `n_min = max(0, -e)`, `qq = q^{1/2}`.

Mirror symmetry `MIt(m, e)`:
```
if m + e ≥ 0:  MIt(m, e) = (-qq)^m · I_t(-m-e, m)
else:           MIt(m, e) = I_t(m, e)
```

### Implementation: `_tet_index_series_python(m, e, qq_order) → dict[int,int]`

- Non-integer m or e → return `{}`
- Build inverse q-factorial polynomials incrementally:
  `inv_fact[k] = 1/∏_{j=1}^{k}(1 - qq^{2j})` as `{power: coeff}` dict
- Extend: `inv_fact[k] = inv_fact[k-1] * Σ_{j≥0} qq^{2kj}` (geometric series,
  truncated at `qq_order`)
- For each summation index n: compute `exp_qq`, multiply `inv_fact[n] * inv_fact[n+e]`,
  shift by `exp_qq`, accumulate with sign `(-1)^n`
- Apply MIt symmetry: multiply by `(-1)^m · qq^m` (shift all powers by m)

### Caching: `_tet_cache: dict[(m, e, qq_order), dict[int,int]]`

Module-level memoization. `_tet_index_series(m, e, qq_order)` checks cache
first, delegates to C extension if available, else Python fallback.

Provide `clear_tet_cache()`.

### C Extension (optional, Phase 13)

When `_c_tet_index_series` is importable, use it. Also `_c_poly_convolve`
for polynomial multiplication. Fallback to pure Python is always available.

---

## Part B: Degree Formula

### `tet_degree(m, e) → Fraction`

Leading qq-power of I_Δ(m, e):
```
δ(m, e) = ½(m₊(m+e)₊ + (−m)₊ e₊ + (−e)₊(−e−m)₊) + max{0, m, −e}
```
where `x₊ = max(0, x)`.

Properties: `tet_degree(0,0) = 0`, always ≥ 0, symmetry `δ(m,e) = δ(-e,-m)`.

### `_tet_degree_x2(m, e) → int`

Returns `2 * tet_degree(m, e)` as plain int. Used in hot-path enumeration
to avoid Fraction construction.

---

## Part C: κ Construction & Phase

### `build_kappa(m_ext, e_ext, e_int, n, r) → np.ndarray`
```
κ = (m_full, e_full) where
    m_full = (m_ext[0..r-1], 0, …, 0)        # n entries; internal m forced to 0
    e_full = (e_ext[0..r-1], e_int[0..n-r-1]) # n entries; all internal edges
```
Returns shape (2n,) dtype object (holds Fraction entries).

### `phase_exponent(kappa, nu_x, nu_p, n, r, num_hard) → Fraction`
```
phase = m_full · nu_p − e_full · nu_x
```
nu_p entries may be half-integer → use `Fraction(v).limit_denominator(1000)`.

---

## Part D: Half-Integer Patterns

### `valid_half_integer_patterns(g_NZ_inv, n, r) → list[np.ndarray]`

When `e_int = e0 + δ/2` (e0 integer, δ ∈ {0,1}^{n-r}), the half-integer
contribution to `g_NZ⁻¹ κ` is `(1/2) · g_NZ_inv[:, n+r:2n] @ δ`.

For the result to be integer: each component of `g_NZ_inv[:, n+r:2n] @ δ`
must be even.

Enumerate all `2^{n-r}` bit patterns δ, keep those satisfying the
even-parity check. (For n-r=0, return `[np.array([], dtype=int)]`.)

---

## Part E: Enumeration Range (Convex Bounding Box)

### `_exact_e0_candidates(base_args, easy_cols, nu_x_easy, phase_base_x2, q_bound, n, num_easy)`

Find all `e0 ∈ Z^{num_easy}` where effective degree ≤ q_bound.

**Effective degree F(e0):**
```
args = base_args + easy_cols @ e0     # (2n,) int64
F(e0) = Σ_a tet_degree(args[a], args[n+a])  +  phase_base  −  nu_x_easy · e0
```

F is piecewise-quadratic, convex, grows to +∞ in all directions.

**Algorithm** (uses `2*F` to stay in exact int arithmetic):

1. **Per-axis projection bounds**: For each axis j, compute
   `R[j] = max{|t| : min_{y} F_x2(t·e_j + y) ≤ 2·q_bound}`
   using `_axis_scan_bound` on the projected function `G_j(t) = min_y F_x2(…)`.
   Minimisation over free variables uses `_proj_min_fixed` (coordinate descent
   with diagonal directions for d ≤ 4, cardinal + pairwise diagonals for d > 4).

2. **Enumerate bounding box** `Π_j[-R_j, R_j]`, keep `e0` with `F_x2(e0) ≤ 2·q_bound`.

### `_axis_scan_bound(F_1d, q_bound) → int`
Scan outward from 0 in both directions. Stop when F_1d(t) > q_bound and
non-decreasing for 2 consecutive steps (convexity guarantee).

### `_proj_min_fixed(F_x2, fixed_j, tj, num_easy, q_bound_x2) → int`
Fixes e0[j] = tj, minimizes F_x2 over remaining d-1 free components
via iterative direction scanning. Direction set:
- d ≤ 4: all {-1,0,1}^{d-1} (excludes all-zero)
- d > 4: cardinal + pairwise diagonal directions

Repeat outer loop until no direction improves; each 1-D scan stops when
val > best_val and non-decreasing for 2 consecutive steps (plateau-safe).

---

## Part F: Enumeration State (Performance Cache)

### `_EnumerationState` (dataclass)

Pre-computed per-manifold state, reused across all (m_ext, e_ext) evaluations:
```python
@dataclass
class _EnumerationState:
    n: int; r: int; n_int: int
    S: int                      # LCD of g_NZ_inv entries
    g_inv_xS: np.ndarray        # (2n,2n) int64 — S × g_NZ^{-1}
    int_cols_int: np.ndarray    # (2n, n_int) int64 — internal-edge columns of g_inv (exact int)
    nu_x_int: np.ndarray        # (n_int,) int64 — nu_x[r:n]
    nu_x_full: np.ndarray       # (n,) int64
    nu_p_x2: np.ndarray         # (n,) int64 — round(2 * nu_p)
    patterns: list[np.ndarray]  # valid δ patterns
    delta_contrib_x2S: list[np.ndarray]  # per-pattern g_inv contribution
    delta_phase_x2: list[int]            # per-pattern phase contribution
    cusp_m_cols_xS: np.ndarray  # (2n, r) int64
    cusp_e_cols_xS: np.ndarray  # (2n, r) int64
```

**Content-based cache key**: `(g_NZ.tobytes(), nu_x.tobytes(), nu_p.tobytes())`.

**Important**: Internal-edge columns of g_inv are always integer-valued.
Assert `int_cols_xS % S == 0` at construction time.

---

## Part G: `enumerate_summation_terms(nz_data, m_ext, e_ext, q_order_half)`

Main enumeration. Returns `list[dict]` where each dict has:
- `"e_int"`: list[str] — e_int as "p/q" strings
- `"phase_exp"`: int — exponent of (-q^{1/2})
- `"tet_args"`: list[(int,int)] — (m_a, e_a) per tet
- `"min_degree"`: float — sum of tet_degree values

**Algorithm** (using _EnumerationState):

For each δ pattern:
1. Compute `base_args_x2S = me_contrib_x2S + delta_contrib_x2S`
2. Check integrality: `base_args_x2S % (2S) == 0`; skip if not
3. `base_args = base_args_x2S // (2S)`
4. `phase_base_x2 = phase_me_x2 + delta_phase_x2`
5. Call `_exact_e0_candidates(…)` to get all valid e0
6. For each e0: compute `args = base_args + int_cols_int @ e0`, degree, phase, build dict

---

## Part H: `compute_index_3d_python(nz_data, m_ext, e_ext, q_order_half, _precomputed_terms=None)`

Computes I(m_ext, e_ext) as a q^{1/2}-series.

**Algorithm**:
1. Get terms from `enumerate_summation_terms` (or use _precomputed_terms)
2. For each term:
   a. `budget = q_order_half - phase_exp`
   b. Multiply tet index series `∏_a I_Δ(m_a, e_a)` as polynomial dicts:
      ```
      prod = {0: 1}
      for (m_a, e_a) in tet_args:
          s = _tet_index_series(m_a, e_a, cutoff)
          prod = convolve(prod, s, budget)
      ```
      Track `prod_min_pow` for tighter cutoffs.
   c. Apply phase: shift by `phase_exp`, multiply by `(-1)^{phase_exp}`
   d. Accumulate into total dict
3. Return `Index3DResult`

---

## Part I: `Index3DResult`

```python
@dataclass
class Index3DResult:
    coeffs: list[int]     # coeffs[k] = coeff of qq^{min_power+k}
    min_power: int
    q_order_half: int
    m_ext: list
    e_ext: list
    n_terms: int = 0

    def as_polynomial_string(self, var="q") -> str: ...
```

---

## Tests (`tests/test_index_3d.py`)

```python
"""Tests for 3D index computation."""
from fractions import Fraction
import pytest
from manifold_index.core.index_3d import (
    tet_degree, _tet_degree_x2, _tet_index_series_python,
    enumerate_summation_terms, compute_index_3d_python,
)


def test_tet_degree_known_values():
    assert tet_degree(1, 0) == Fraction(3, 2)
    assert tet_degree(0, 0) == 0


def test_tet_degree_symmetry():
    """δ(m,e) == δ(-e,-m)"""
    for m in range(-3, 4):
        for e in range(-3, 4):
            assert tet_degree(m, e) == tet_degree(-e, -m)


def test_tet_degree_nonneg():
    for m in range(-3, 4):
        for e in range(-3, 4):
            assert tet_degree(m, e) >= 0


def test_tet_degree_x2_matches():
    for m in range(-3, 4):
        for e in range(-3, 4):
            assert _tet_degree_x2(m, e) == int(2 * tet_degree(m, e))


def test_tet_index_series_basic():
    """I_Δ(0, 0) should have nonzero constant term."""
    s = _tet_index_series_python(0, 0, 10)
    assert 0 in s and s[0] != 0


def test_enumerate_summation_terms(nz_m004):
    ext = [0] * nz_m004.r
    terms = enumerate_summation_terms(nz_m004, ext, ext, q_order_half=10)
    assert len(terms) > 1
    assert all("phase_exp" in t and "tet_args" in t for t in terms)


def test_compute_index_basic(nz_m004):
    pytest.importorskip("snappy")
    result = compute_index_3d_python(
        nz_m004, [1], [0], q_order_half=8
    )
    assert len(result.coeffs) > 0
    assert result.n_terms > 0
```

---

*Phase 6 complete → proceed to Phase 7.*
