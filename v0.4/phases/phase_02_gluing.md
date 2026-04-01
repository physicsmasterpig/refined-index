# Phase 2: Gluing Equation Reduction

## Dependencies
- Phase 1 complete: `ManifoldData`, `load_manifold()`
- External: `numpy`, `scipy.linalg.qr`

## Files to Create
- `src/manifold_index/core/gluing_equations.py`
- `tests/test_gluing_equations.py`

---

## Public API

```python
@dataclass
class ReducedGluingData:
    n: int                              # tetrahedra
    r: int                              # cusps
    edge_coeffs: np.ndarray             # (n, 2n), int
    edge_consts: np.ndarray             # (n,), int
    cusp_coeffs: np.ndarray             # (2r, 2n), int
    cusp_consts: np.ndarray             # (2r,), int
    independent_edge_indices: list[int]  # length n-r, sorted
    symplectic_matrix: np.ndarray       # (2n, 2n), int — Ω

    # Properties:
    @property
    def independent_edge_coeffs(self) -> np.ndarray:
        """shape (n-r, 2n)"""

    @property
    def independent_edge_consts(self) -> np.ndarray:
        """shape (n-r,)"""

    def meridian_coeffs(self, k: int) -> np.ndarray:
        """Reduced meridian for cusp k, shape (2n,)."""

    def longitude_coeffs(self, k: int) -> np.ndarray:
        """Reduced longitude for cusp k, shape (2n,)."""

    def commutator(self, a: np.ndarray, b: np.ndarray) -> int:
        """Symplectic pairing [A, B] = a^T Ω b."""


def reduce_gluing_equations(data: ManifoldData) -> ReducedGluingData
```

---

## Algorithm

### Step 1: `_reduce_row(row_3n, n) → (const, coeff_2n)`

Substitutes Z_i' = 1 - Z_i - Z_i'' into one row of the gluing matrix.

For each tetrahedron i = 0…n-1, extract the triple:
```
f = row[3*i]      # coefficient of Z_i
g = row[3*i + 1]  # coefficient of Z_i'
h = row[3*i + 2]  # coefficient of Z_i''
```

Then:
```
const += g                  # from the substitution g·(1 - Z - Z'')
coeff[2*i]   = f - g       # net coefficient of Z_i
coeff[2*i+1] = h - g       # net coefficient of Z_i''
```

Returns `(const, coeff)` where `coeff` has shape `(2n,)`, dtype int.

**Derivation:** The original row is Σ_i (f_i·Z_i + g_i·Z_i' + h_i·Z_i'').
Substituting Z_i' = 1 - Z_i - Z_i'':
= Σ_i (f_i·Z_i + g_i·(1 - Z_i - Z_i'') + h_i·Z_i'')
= Σ_i g_i + Σ_i (f_i - g_i)·Z_i + Σ_i (h_i - g_i)·Z_i''

### Step 2: `_build_symplectic_matrix(n) → np.ndarray`

Builds the 2n×2n symplectic matrix Ω in interleaved ordering:
```python
omega = np.zeros((2*n, 2*n), dtype=int)
for i in range(n):
    omega[2*i,   2*i+1] = +1   # [Z_i, Z_i''] = +1
    omega[2*i+1, 2*i  ] = -1
return omega
```

### Step 3: `_independent_row_indices(coeff_matrix, expected_rank) → list[int]`

Finds n-r linearly independent rows from the n edge equations.

Uses **column-pivoted QR** on the transpose:
```python
from scipy.linalg import qr
_, _, piv = qr(coeff_matrix.astype(float).T, pivoting=True)
return sorted(piv[:expected_rank].tolist())
```

This works because QR with column pivoting on A^T reveals which rows of A
form a maximal independent set — the first `rank` pivot indices correspond
to the rows that contributed the most linearly independent information.

### Step 4: `reduce_gluing_equations(data) → ReducedGluingData`

Main function:

```python
n, r = data.num_tetrahedra, data.num_cusps

# Reduce edge equations (n rows)
edge_consts, edge_coeffs = [], []
for row in data.edge_equations:
    c, v = _reduce_row(row, n)
    edge_consts.append(c)
    edge_coeffs.append(v)
edge_consts = np.array(edge_consts, dtype=int)
edge_coeffs = np.array(edge_coeffs, dtype=int)

# Reduce cusp equations (2r rows, interleaved μ₀,λ₀,μ₁,λ₁,…)
cusp_rows = data.gluing_matrix[n : n + 2*r]
cusp_consts, cusp_coeffs = [], []
for row in cusp_rows:
    c, v = _reduce_row(row, n)
    cusp_consts.append(c)
    cusp_coeffs.append(v)
cusp_consts = np.array(cusp_consts, dtype=int)
cusp_coeffs = np.array(cusp_coeffs, dtype=int)

# Find independent edges
independent_edge_indices = _independent_row_indices(edge_coeffs, n - r)

# Build symplectic matrix
omega = _build_symplectic_matrix(n)

return ReducedGluingData(n, r, edge_coeffs, edge_consts,
                         cusp_coeffs, cusp_consts,
                         independent_edge_indices, omega)
```

---

## Edge Cases

- If `n == r` (no internal edges): `independent_edge_indices` is empty,
  `expected_rank = 0`, and QR should handle this gracefully.
- Cusp rows include both meridian and longitude interleaved — indexing
  is `cusp_coeffs[2*k]` for meridian, `cusp_coeffs[2*k+1]` for longitude.

---

## Test Values

### m004 (n=2, r=1)

```python
data = load_manifold("m004")
rd = reduce_gluing_equations(data)

# Shapes
assert rd.edge_coeffs.shape == (2, 4)
assert rd.edge_consts.shape == (2,)
assert rd.cusp_coeffs.shape == (2, 4)
assert rd.cusp_consts.shape == (2,)

# Rank
assert np.linalg.matrix_rank(rd.edge_coeffs) == 1  # n-r = 2-1 = 1

# Independent edges
assert len(rd.independent_edge_indices) == 1

# Symplectic pairing: [μ, λ] = 2 for any cusp
mu = rd.meridian_coeffs(0)
lam = rd.longitude_coeffs(0)
assert rd.commutator(mu, lam) == 2

# Symplectic matrix structure
assert rd.symplectic_matrix.shape == (4, 4)
assert rd.symplectic_matrix[0, 1] == 1
assert rd.symplectic_matrix[1, 0] == -1
assert rd.symplectic_matrix[2, 3] == 1
assert rd.symplectic_matrix[3, 2] == -1
```

### Key identity: [μ_k, λ_k] = 2

For every cusped hyperbolic 3-manifold, the symplectic pairing of the
k-th meridian and longitude in the reduced basis equals exactly 2.
This is a fundamental topological fact.  Test it for m004 and m003.

---

## Tests to Write (`tests/test_gluing_equations.py`)

```python
"""Tests for reduced gluing equations."""
import pytest
import numpy as np
from manifold_index.core.manifold import load_manifold
from manifold_index.core.gluing_equations import reduce_gluing_equations


def test_shapes_m004():
    pytest.importorskip("snappy")
    data = load_manifold("m004")
    rd = reduce_gluing_equations(data)
    n, r = 2, 1
    assert rd.edge_coeffs.shape == (n, 2 * n)
    assert rd.edge_consts.shape == (n,)
    assert rd.cusp_coeffs.shape == (2 * r, 2 * n)
    assert rd.cusp_consts.shape == (2 * r,)


def test_edge_rank_m004():
    pytest.importorskip("snappy")
    data = load_manifold("m004")
    rd = reduce_gluing_equations(data)
    assert np.linalg.matrix_rank(rd.edge_coeffs) == data.num_tetrahedra - data.num_cusps


def test_independent_edge_count():
    pytest.importorskip("snappy")
    data = load_manifold("m004")
    rd = reduce_gluing_equations(data)
    assert len(rd.independent_edge_indices) == data.num_tetrahedra - data.num_cusps


def test_commutator_meridian_longitude():
    """[μ, λ] = 2 for every cusp."""
    pytest.importorskip("snappy")
    for name in ["m004", "m003"]:
        data = load_manifold(name)
        rd = reduce_gluing_equations(data)
        for k in range(data.num_cusps):
            mu = rd.meridian_coeffs(k)
            lam = rd.longitude_coeffs(k)
            assert rd.commutator(mu, lam) == 2, f"{name} cusp {k}: [μ,λ] ≠ 2"


def test_symplectic_matrix_structure():
    pytest.importorskip("snappy")
    data = load_manifold("m004")
    rd = reduce_gluing_equations(data)
    omega = rd.symplectic_matrix
    # Antisymmetric
    assert np.array_equal(omega, -omega.T)
    # Correct diagonal blocks
    for i in range(data.num_tetrahedra):
        assert omega[2*i, 2*i+1] == 1
        assert omega[2*i+1, 2*i] == -1
```

---

*Phase 2 complete → proceed to Phase 3.*
