# Phase 4: Neumann-Zagier Matrix & Affine Shift

## Dependencies
- Phase 1: `ManifoldData`, `load_manifold()`
- Phase 2: `ReducedGluingData`, `reduce_gluing_equations()`, `_reduce_row()`
- Phase 3: `EasyEdgeResult`, `find_easy_edges()`
- External: `numpy`, `scipy`, `fractions.Fraction`

## Files to Create
- `src/manifold_index/core/neumann_zagier.py`
- `tests/test_neumann_zagier.py`
- Update `tests/conftest.py` (add session-scoped NZ fixtures)

---

## Mathematical Background

### Variables
After substituting Z_i' = 1 − Z_i − Z_i'' (the reduction convention from
Phase 2), each gluing equation is linear in the **2n** variables

    v = (Z_1, …, Z_n,  Z_1'', …, Z_n'')      [block ordering]

### Ordering conventions
- **Interleaved**: (Z_1, Z_1'', Z_2, Z_2'', …)  — output of `_reduce_row`
- **Block**: (Z_1, …, Z_n, Z_1'', …, Z_n'')  — columns of g_NZ

### Symplectic form (block ordering)
```
Ω_block = [[ 0_n,  I_n],
           [-I_n,  0_n]]
```
Pairing: [u, v] = u^T Ω v = u_Z · v_{Z''} − u_{Z''} · v_Z.

### g_NZ row structure (2n × 2n)
```
Top n rows (position):
  rows 0 … r−1              meridian equations (one per cusp)
  rows r … r+d_hard−1       hard internal edges
  rows r+d_hard … n−1       easy internal edges

Bottom n rows (momentum):
  rows n … n+r−1            longitude / 2 (one per cusp)
  rows n+r … 2n−1           Γ vectors (conjugate momenta of internal edges)
```

### Affine shift ν
The reduced equation is `c + coeff · v = RHS`, so `ν = c − RHS` where
c = Σ_i g_i is the constant from `_reduce_row`.

| Row type          | RHS | ν formula         |
|-------------------|-----|-------------------|
| Meridian          | 0   | ν_x = c           |
| Internal edge     | 2   | ν_x = c − 2       |
| Longitude/2       | 0   | ν_p = c_long / 2  |
| Γ (constructed)   | —   | ν_p = 0           |

---

## Public API

### `NeumannZagierData` (dataclass)

```python
@dataclass
class NeumannZagierData:
    g_NZ: np.ndarray        # (2n, 2n), float. Longitude rows may be half-int.
    nu_x: np.ndarray        # (n,), int. Position affine shift.
    nu_p: np.ndarray        # (n,), float. Momentum affine shift. Γ entries = 0.
    n: int
    r: int
    num_hard: int
    num_easy: int
    # Private caches (init to None):
    _g_inv_cache: np.ndarray | None
    _g_inv_scaled_cache: tuple[int, np.ndarray] | None
```

**Properties / methods:**

| Name                | Returns                         | Notes |
|---------------------|---------------------------------|-------|
| `symplectic_form`   | `np.ndarray (2n,2n)` int        | Ω_block |
| `is_symplectic(tol)`| `bool`                          | Checks g Ω g^T = Ω |
| `g_NZ_inv()`        | `np.ndarray (2n,2n)` Fraction   | Uses symplectic identity, cached |
| `g_NZ_inv_scaled()` | `(S, S·g^{-1})`                | S = LCD of all entries. Cached. |
| `inv_denom`         | `int`                           | S from above |

### Symplectic inverse formula (exact)
For g = [[A, B], [C, D]] in n×n blocks:
```
g^{-1} = [[ D^T, -B^T],
           [-C^T,  A^T]]
```
Convert float entries → `Fraction(v).limit_denominator(1000)`.

### `g_NZ_inv_scaled()`
Computes LCD S of all Fraction entries in g^{-1}, returns `(S, int_matrix)`
where `int_matrix = S * g^{-1}` as int64. Typically S = 2 (from L/2 rows).

---

## Internal Helpers

### `_interleaved_to_block(coeff_2n, n) → np.ndarray`
Permutation:
```python
perm = [2*i for i in range(n)] + [2*i+1 for i in range(n)]
return coeff_2n[perm]
```

### `_reduce_to_block(row_3n, n) → (int, np.ndarray)`
Calls `_reduce_row(row_3n, n)` → `(const, coeff_interleaved)`, then
`_interleaved_to_block(coeff_interleaved, n)` → `coeff_block`.

### `_build_omega_block(n) → np.ndarray`
Returns Ω_block as (2n, 2n) int array.

---

## Critical Algorithm: `_int_right_inverse(A_int) → np.ndarray`

Given an **n × 2n** integer matrix A of full row-rank, find a **2n × n**
rational matrix Q_T (entries are `Fraction`) such that `A @ Q_T = I_n`.

**Pre-condition (assert before proceeding):**
```python
rank = np.linalg.matrix_rank(A.astype(float))
if rank < n:
    raise ValueError(
        f"_int_right_inverse: A has rank {rank} < n={n}. "
        "The NZ position block P @ omega is rank-deficient — "
        "this indicates a degenerate or incorrectly assembled manifold."
    )
```

### Algorithm: Euclidean column reduction

Work with Python lists-of-lists for exact integer arithmetic.

1. Initialize V = I_{2n} (transformation matrix).
2. For each `pivot_row` in 0…n−1:
   a. **Euclidean loop** on row `pivot_row`, columns `col_start` … 2n−1:
      - Find column with smallest |entry| among nonzero columns → swap to `col_start`
      - For each column c > col_start: `col[c] -= (entry[c] // entry[col_start]) * col[col_start]`
      - Repeat until all entries right of col_start are zero
      - If no progress with floor division, swap pivot with a remaining column and retry
   b. If pivot value < 0, negate column.
   c. After loop: `A[pivot_row][col_start]` is the only nonzero in that row (= gcd, should be 1 for unit Smith factors).
   d. All column operations are mirrored on V.

3. Result: `A @ V = [H | 0]` where H is n×n lower-triangular with positive diagonal.
4. Compute `H^{-1}` via forward substitution in **exact Fraction arithmetic**:
   ```python
   H_inv = identity(n, Fraction)
   for i in range(n):
       for j in range(i):
           H_inv[i] -= Fraction(H[i,j]) * H_inv[j]
       H_inv[i] /= Fraction(H[i,i])   # division by diagonal
   ```
5. Return `Q_T = V[:, :n] @ H_inv` (2n × n, Fraction).

**Important**: When Smith invariant factors > 1 (e.g., v0901), the diagonal
of H has entries > 1, making H_inv have proper fractions. This is correct —
the right inverse will have half-integer entries.

---

## Algorithm: `_make_isotropic(Q_T, P, omega) → np.ndarray`

Given Q_T (2n × n, Fraction) with `P @ omega @ Q_T = I_n`, adjust Q_T so
that `Q_T^T @ omega @ Q_T = 0` (isotropic) while preserving the
right-inverse property.

1. Compute anti-symmetric pairing matrix `S = Q_T^T @ omega @ Q_T` (n × n).
2. Extract `C = strictly lower-triangular part of S` (C[i,j] = S[i,j] for j < i, else 0).
3. `Q_T' = Q_T + P^T @ C` (add null-space corrections).

**Why this works**: columns of P^T are in ker(A) = ker(P @ omega), so
`A @ (P^T @ C) = 0` → right-inverse property preserved. The correction
zeroes the anti-symmetric S because S' = S + C^T − C = 0.

---

## Main Function: `build_neumann_zagier(data, easy_result, reduced=None)`

### Stage 1: Build position block P (n × 2n, int) and ν_x (n,)

**Meridian rows** (k = 0…r−1):
```python
merid_row_3n = data.gluing_matrix[n + 2*k]      # SnaPy: row n+2k
const, coeff_block = _reduce_to_block(merid_row_3n, n)
P[k] = coeff_block
nu_x[k] = const                                  # RHS = 0
```

**Internal edge rows** (j = 0…n−r−1):
Ordering: **hard edges first, then easy edges** (opposite of EasyEdgeResult.basis_edges which is easy-first).
```python
hard_edges = easy_result.hard_padding                           # 3n-vecs
easy_edges = [easy_result.all_easy[i]
              for i in easy_result.independent_easy_indices]     # 3n-vecs
internal_ordered = hard_edges + easy_edges                      # total n-r
for j, edge_3n in enumerate(internal_ordered):
    const, coeff_block = _reduce_to_block(edge_3n, n)
    P[r + j] = coeff_block
    nu_x[r + j] = const - 2                                    # RHS = 2
```

### Stage 2: Build momentum block Q (n × 2n, float) and ν_p (n,)

**Step 2a**: Compute Γ rows via right-inverse:
```python
A = P @ omega                                  # (n, 2n) int
Q_T = _int_right_inverse(A)                    # (2n, n) Fraction
Q_T = _make_isotropic(Q_T, P, omega)           # still right-inverse, now isotropic
Q = Q_T.T                                      # (n, 2n) Fraction
```

**Step 2b**: Build longitude/2 rows from SnaPy:
```python
for k in range(r):
    long_row_3n = data.gluing_matrix[n + 2*k + 1]   # SnaPy: row n+2k+1
    const_long, coeff_block = _reduce_to_block(long_row_3n, n)
    Q_lon[k] = coeff_block / 2
    nu_p[k] = const_long / 2
# nu_p[r:] = 0   (Γ rows have zero shift)
```

**Step 2c**: Correct Γ rows for orthogonality to actual longitudes.
The right-inverse Γ rows are orthogonal to the *computed* longitude columns
of Q_T, which may differ from the actual L_k/2 from SnaPy. Fix:
```python
Q = Q.astype(float)
for j in range(r, n):           # each Γ row
    for k in range(r):          # each longitude/2 row
        pairing = Q[j] @ omega @ Q_lon[k]    # [Γ_j, L_k/2]
        Q[j] -= pairing * P[k]               # subtract pairing × M_k
```
**Why M_k**: [E_i, M_k] = 0 and [Γ_j, M_k] = 0 (from symplectic structure),
so adding multiples of M_k preserves all other pairings.

**Step 2d**: Replace first r rows of Q with actual longitude/2:
```python
Q[0:r] = Q_lon
```

### Stage 3: Assemble
```python
g_NZ = np.vstack([P, Q])     # (2n, 2n)
return NeumannZagierData(g_NZ, nu_x, nu_p, n, r, num_hard, num_easy)
```

---

## Cusp Basis Change Functions

### `_ext_gcd(a, b) → (g, x, y)` with `a*x + b*y = g`, g ≥ 0
Standard recursive extended GCD.

### `apply_cusp_basis_change(nz_data, cusp_idx, P, Q) → NeumannZagierData`

Replaces cusp k basis (M_k, L_k/2) with:
```
new_position  = P · M_k  + 2Q · (L_k/2)       [= P·M + Q·L]
new_momentum  = a · M_k  + b  · (L_k/2)
```
where `P·b − 2Q·a = 1` (Bézout), found via `_ext_gcd(P, −2Q)`.

**Requirements**: P must be **odd** (equivalently gcd(P, 2Q) = 1).

**Affine shift update**:
```
nu_x_new[k] = P * nu_x[k] + 2Q * nu_p[k]      (round to int)
nu_p_new[k] = a * nu_x[k] + b  * nu_p[k]
```

**Raises** ValueError if P is even or cusp_idx out of range.

### `apply_general_cusp_basis_change(nz_data, cusp_idx, a, b, c, d)`

General SL(2,ℤ) matrix [[a,b],[c,d]] (ad − bc = 1) acting on (μ, λ):
```
new_M   = a · M  + 2b · (L/2)          [always integer]
new_L/2 = (c/2) · M + d · (L/2)        [entries in Z/2]
```
No parity requirement on a. The resulting L/2 row may have half-int entries.

**Raises** ValueError if det ≠ 1 or cusp_idx out of range.

### Which function to use?

| Situation | Use |
|-----------|-----|
| Dehn filling at slope P/Q with P **odd** | `apply_cusp_basis_change(nz, k, P, Q)` |
| Dehn filling at slope P/Q with P **even** | `apply_general_cusp_basis_change(nz, k, P, Q//gcd, Q, -P//gcd)` — but note: even-P slopes cannot be handled directly because `apply_cusp_basis_change` requires P odd (gcd(P,2Q)=1). Instead, use the unrefined kernel K(P,Q;m,e) directly from Phase 9 without a basis change. |
| Non-closable cycle search or Weyl basis | `apply_cusp_basis_change` (NC basis always has P=1, which is odd) |
| General SL(2,ℤ) diagnostic/test | `apply_general_cusp_basis_change` |

> **Note on downstream half-integer rows:** When `apply_general_cusp_basis_change`
> is used, the resulting `nu_p[k]` may be non-integer (half-integer).  Phase 6's
> `phase_exponent` handles this correctly via `Fraction(v).limit_denominator(1000)`.
> `apply_cusp_basis_change` always produces integer `nu_p[k]` because P is odd.

---

## conftest.py Fixtures

```python
@pytest.fixture(scope="session")
def nz_m004():
    pytest.importorskip("snappy")
    from manifold_index.core.manifold import load_manifold
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import build_neumann_zagier
    data = load_manifold("m004")
    easy = find_easy_edges(data)
    return build_neumann_zagier(data, easy)

@pytest.fixture(scope="session")
def nz_m003():
    pytest.importorskip("snappy")
    from manifold_index.core.manifold import load_manifold
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import build_neumann_zagier
    data = load_manifold("m003")
    easy = find_easy_edges(data)
    return build_neumann_zagier(data, easy)

@pytest.fixture(scope="session")
def nz_v0901():
    """v0901 (7 tet, 1 cusp) — has non-unit Smith invariant factors."""
    pytest.importorskip("snappy")
    from manifold_index.core.manifold import load_manifold
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import build_neumann_zagier
    data = load_manifold("v0901")
    easy = find_easy_edges(data)
    return build_neumann_zagier(data, easy)
```

---

## Tests (`tests/test_neumann_zagier.py`)

```python
"""Tests for Neumann-Zagier matrix construction."""
import numpy as np
import pytest


def test_symplectic_and_inverse(nz_m004):
    nz = nz_m004
    assert nz.is_symplectic()
    product = nz.g_NZ @ nz.g_NZ_inv()
    np.testing.assert_array_almost_equal(product, np.eye(2 * nz.n), decimal=9)


def test_symplectic_nonunit_smith(nz_v0901):
    """v0901 has Smith invariant factors > 1 — g_NZ must still be symplectic."""
    nz = nz_v0901
    assert nz.is_symplectic()
    det = np.linalg.det(nz.g_NZ)
    assert abs(abs(det) - 1) < 1e-6
    product = nz.g_NZ @ nz.g_NZ_inv()
    np.testing.assert_array_almost_equal(product, np.eye(2 * nz.n), decimal=9)


def test_affine_shift_dimensions(nz_m004):
    nz = nz_m004
    assert nz.nu_x.shape == (nz.n,)
    assert nz.nu_p.shape == (nz.n,)
    assert nz.g_NZ.shape == (2 * nz.n, 2 * nz.n)


def test_inv_scaled_integrality(nz_m004):
    S, scaled = nz_m004.g_NZ_inv_scaled()
    assert isinstance(S, int) and S >= 1
    assert scaled.dtype == np.int64


def test_num_hard_easy(nz_m004):
    nz = nz_m004
    assert nz.num_hard + nz.num_easy == nz.n - nz.r
```

---

*Phase 4 complete → proceed to Phase 5.*
