# Phase 1: Manifold Loading

## Dependencies
- Phase 0 complete (package installable)
- External: `snappy`

## Files to Create
- `src/manifold_index/core/manifold.py`
- `tests/test_manifold.py`

---

## Public API

```python
@dataclass
class ManifoldData:
    name: str                     # SnaPy name, e.g. "m004"
    num_tetrahedra: int           # n
    num_cusps: int                # r
    gluing_matrix: np.ndarray     # shape (n + 2r, 3n), dtype int
    raw: Any                      # snappy.Manifold object (repr=False)

    @property
    def edge_equations(self) -> np.ndarray:
        """Rows 0…n-1, shape (n, 3n)."""

    @property
    def meridian_equations(self) -> np.ndarray:
        """Rows n, n+2, n+4, …, shape (r, 3n)."""

    @property
    def longitude_equations(self) -> np.ndarray:
        """Rows n+1, n+3, n+5, …, shape (r, 3n)."""

    def cusp_equations(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """(meridian_row, longitude_row) for cusp k, each shape (3n,)."""


def load_manifold(name: str) -> ManifoldData:
    """
    Load a manifold by name using SnaPy.

    Raises
    ------
    ImportError  — if snappy not installed
    ValueError   — if name not recognized by SnaPy
    """
```

---

## Algorithm

`load_manifold(name)`:

1. `import snappy` — catch `ImportError`, re-raise with message
   `f"Failed to import snappy: {exc}"`
2. `M = snappy.Manifold(name)` — catch any exception, raise
   `ValueError(f"SnaPy could not load manifold '{name}': {exc}")`
3. `n = M.num_tetrahedra()`
4. `r = M.num_cusps()`
5. `raw_eqs = M.gluing_equations()` — this is SnaPy's `SimpleMatrix`
6. `rows, cols = raw_eqs.shape`
7. `gluing_matrix = np.array(raw_eqs.list(), dtype=int).reshape(rows, cols)`
8. Assert `rows == n + 2*r` and `cols == 3*n`
9. Return `ManifoldData(name, n, r, gluing_matrix, M)`

### Property implementations

```python
@property
def edge_equations(self):
    return self.gluing_matrix[:self.num_tetrahedra]

@property
def meridian_equations(self):
    n, r = self.num_tetrahedra, self.num_cusps
    return self.gluing_matrix[n::2][:r]

@property
def longitude_equations(self):
    n, r = self.num_tetrahedra, self.num_cusps
    return self.gluing_matrix[n + 1::2][:r]

def cusp_equations(self, k):
    n = self.num_tetrahedra
    return (self.gluing_matrix[n + 2*k], self.gluing_matrix[n + 2*k + 1])
```

---

## Gotchas

- SnaPy's `gluing_equations()` (no argument) returns rect format by
  default.  Do NOT pass `"rect"` — just call `M.gluing_equations()`.
  The returned `SimpleMatrix` has shape `(n+2r, 3n)`.  Passing `"rect"`
  in some SnaPy versions returns `3n+1` columns (with a constant column
  appended) — we don't want that.  The no-argument call is always `3n` columns.
  Verify at runtime with the assert on step 8.

- The `raw` field stores the SnaPy `Manifold` object.  Use `repr=False`
  in the field definition to keep `repr()` clean.

---

## Test Values

### m004 (figure-eight knot complement)
```python
data = load_manifold("m004")
assert data.num_tetrahedra == 2
assert data.num_cusps == 1
assert data.gluing_matrix.shape == (4, 6)
assert data.edge_equations.shape == (2, 6)
assert data.meridian_equations.shape == (1, 6)
assert data.longitude_equations.shape == (1, 6)
mu, lam = data.cusp_equations(0)
assert mu.shape == (6,)
assert lam.shape == (6,)
```

### m003
```python
data = load_manifold("m003")
assert data.num_tetrahedra == 2
assert data.num_cusps == 1
assert data.gluing_matrix.shape == (4, 6)
```

### Error case
```python
with pytest.raises(ValueError):
    load_manifold("not_a_real_manifold_xyz")
```

---

## Tests to Write (`tests/test_manifold.py`)

```python
"""Tests for manifold loading."""
import pytest
from manifold_index.core.manifold import load_manifold, ManifoldData


def test_load_m004():
    pytest.importorskip("snappy")
    data = load_manifold("m004")
    assert isinstance(data, ManifoldData)
    assert data.num_tetrahedra == 2
    assert data.num_cusps == 1
    assert data.gluing_matrix.shape == (4, 6)


def test_load_m003():
    pytest.importorskip("snappy")
    data = load_manifold("m003")
    assert data.num_tetrahedra == 2
    assert data.num_cusps == 1


def test_edge_equations_shape():
    pytest.importorskip("snappy")
    data = load_manifold("m004")
    assert data.edge_equations.shape == (2, 6)


def test_cusp_equations_shape():
    pytest.importorskip("snappy")
    data = load_manifold("m004")
    mu, lam = data.cusp_equations(0)
    assert mu.shape == (6,)
    assert lam.shape == (6,)


def test_load_unknown_raises():
    pytest.importorskip("snappy")
    with pytest.raises(ValueError):
        load_manifold("not_a_real_manifold_xyz")
```

---

## Shared Fixture (`tests/conftest.py`)

Create the conftest stub — later phases will add more fixtures:

```python
"""Shared fixtures for the manifold-index test suite."""
import pytest
```

(No fixtures yet — they depend on later phases.)

---

*Phase 1 complete → proceed to Phase 2.*
