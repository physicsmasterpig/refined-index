# Phase 3: Phase Space Basis (Easy Edges)

## Dependencies
- Phase 1: `ManifoldData`, `load_manifold()`
- Phase 2: `ReducedGluingData`, `reduce_gluing_equations()`, `_reduce_row()`
- External: `numpy`, `scipy.linalg.lstsq`, `scipy.linalg.qr`, `fractions.Fraction`

## Files to Create
- `src/manifold_index/core/phase_space.py`
- `tests/test_phase_space.py`

---

## Public API

```python
@dataclass
class EasyEdgeResult:
    all_easy: list[np.ndarray]            # all discovered easy edges (3n-vectors)
    independent_easy_indices: list[int]   # indices into all_easy
    hard_padding: list[np.ndarray]        # hard edges to reach n-r total
    n: int
    r: int

    @property
    def num_independent_easy(self) -> int: ...
    @property
    def basis_edges(self) -> list[np.ndarray]:
        """[independent_easy... | hard_padding...], length n-r."""


def find_easy_edges(data: ManifoldData, tol: float = 1e-8) -> EasyEdgeResult
def find_phase_space_basis(data: ManifoldData) -> EasyEdgeResult  # alias
```

---

## Constants

```python
_Z   = 0   # Z_i   → column 3i
_ZP  = 1   # Z_i'  → column 3i+1
_ZPP = 2   # Z_i'' → column 3i+2
_OFF = 3   # tet contributes zero
```

---

## Key Helper: `_is_easy(edge_3n, n) → bool`

Returns True iff at most one of (Z_i, Z_i', Z_i'') is nonzero per tet i:
```python
for i in range(n):
    triplet = edge_3n[3*i : 3*i + 3]
    if np.count_nonzero(triplet) > 1:
        return False
return True
```

---

## Algorithm: `find_easy_edges(data, tol)`

### Stage 0: Fast check of raw SnaPy edge rows

Check each of the n edge equation rows directly:
1. Skip if any entry < 0
2. Skip if not `_is_easy(row, n)`
3. Skip if `sum(row) != 2`
4. Deduplicate via `tuple(row.tolist())`
5. Add passing rows to `all_easy`

This catches easy edges that are exact SnaPy rows — fast and
numerically safe.

### Stage 1: Pattern-first enumeration

> **Short-circuit:** Before starting this stage, check whether Stage 0 already
> yielded enough independent easy edges.  If `easy_rank = svd_rank(Stage0_edges)`
> already equals `n - r` (the target), skip Stage 1 entirely — 4^n iterations are
> not needed.  This is common for small well-triangulated manifolds (e.g. m003).
> Only proceed to Stage 1 when Stage 0 did NOT find a complete independent set.

For each pattern ∈ {_Z, _ZP, _ZPP, _OFF}^n (total 4^n patterns):

**Step 1a: Build constraint matrix** `_build_constraint_matrix(pattern, edge_rows, n)`

For each tet j with pattern slot:
- **Active slot** (Z/Z'/Z''): Let inactive = the other two slots. Add one
  row: `col_{inactive_1} - col_{inactive_2}` (rhs = 0). This enforces
  that the two inactive slots have equal net contribution (guaranteeing
  they can be absorbed into b_j).
  Record `ref_col_j = col_{inactive_1}` for normalization.
- **OFF slot**: Add two rows: `col_Z - col_Z'` and `col_Z' - col_Z''`
  (rhs = 0, 0). Record `ref_col_j = col_Z`.

Final normalization row: `a · (2·ones - Σ_j ref_col_j) = 2`

The constraint matrix M has shape (num_constraints, n) and rhs is a vector.

**Step 1b: Solve M·a = rhs**

*Fast path:* `scipy.linalg.lstsq(M, rhs)` → round to int → verify
`‖M·a_int - rhs‖ < tol`. Use this if it works.

*Slow path (underdetermined):* Exact Fraction-based RREF.  See
`_solve_integer_system` below.

**Step 1c: Recover b_j** via `_compute_b(a, pattern, edge_rows, n)`:
```python
for j, slot in enumerate(pattern):
    if slot == _OFF:
        ref_col = edge_rows[:, 3*j]
    else:
        inactive_slot = [s for s in (_Z, _ZP, _ZPP) if s != slot][0]
        ref_col = edge_rows[:, 3*j + inactive_slot]
    b[j] = -(a @ ref_col)
```

**Step 1d: Validate and reconstruct**
1. Check `2*sum(a) + sum(b) == 2`
2. Reconstruct: `E = a @ edge_rows + b @ T_matrix` where T_j has (1,1,1)
   at tet j
3. Check `E >= 0` (non-negativity)
4. Check `_is_easy(E, n)`
5. Deduplicate via `tuple(E.tolist())`

### Stage 2: Select maximal independent subset

Build matrix R of reduced representations of all easy edges.
Column-pivoted QR on R^T → first `easy_rank` pivots are independent.
`easy_rank = min(svd_rank(R), n-r)`.

### Stage 3: Pad with hard edges

If `num_independent_easy < n - r`:
- Iterate through `reduced.independent_edge_indices`
- For each SnaPy edge row, check if adding it increases SVD rank
- Add to `hard_padding` until `n - r` total

---

## Helper: `_solve_integer_system(M, rhs) → list[np.ndarray]`

Exact Fraction-based RREF solver for small integer systems.

```
MAX_COEFF = 3
```

> **Warning:** If a manifold's easy-edge solution requires coefficients with
> |a_j| > 3, `_solve_integer_system` will return `[]` and Stage 1 will find
> no easy edges from that pattern.  In practice all known SnaPy census manifolds
> have coefficients ≤ 3, but if you encounter a manifold where `num_hard` is
> unexpectedly large (all n-r edges end up in `hard_padding`), increase
> `MAX_COEFF` and re-run as a diagnostic step.  Emit a `warnings.warn` if
> Stage 1 finds zero solutions for any pattern where the system is consistent
> but the coefficient bounds clamp the answer.

**Algorithm:**
1. Build augmented matrix `[M | rhs]` over Fraction (round floats to int first)
2. **RREF** with partial pivoting over Q:
   - For each column, find nonzero entry at or below current row
   - Swap, scale pivot to 1, eliminate above AND below
   - Track `pivot_col` list
3. Check consistency: any all-zero-LHS row with nonzero RHS → return []
4. Identify `free_cols = [c for c not in pivot_col]`
5. If no free variables: read unique solution, check integrality and |coeff| ≤ MAX_COEFF
6. If free variables: enumerate all assignments in `[-MAX_COEFF, MAX_COEFF]`
   per free var, back-substitute pivot vars, check integrality and bounds

Returns list of all valid integer solution vectors.

---

## Edge Cases

- **n=1:** Only 4 patterns; trivial system.
- **All easy (num_hard=0):** e.g., m003. `hard_padding` is empty.
- **No easy edges:** `all_easy` is empty, `independent_easy_indices` is
  empty, all n-r edges come from `hard_padding`.
- **4^n can be large:** For n=7 (v0901), 4^7 = 16384 patterns — still
  fast. For n>10, may want optimisation (not needed for current census).

---

## Test Values

### m004 (n=2, r=1, target_rank=1)
```python
data = load_manifold("m004")
result = find_easy_edges(data)
assert len(result.basis_edges) == 1  # n-r = 1
# m004 has one hard edge, so basis_edges[0] is a hard edge
```

### m003 (n=2, r=1, target_rank=1)
```python
data = load_manifold("m003")
result = find_easy_edges(data)
assert len(result.basis_edges) == 1
# m003's single internal edge is [2,0,1,2,0,1] — two nonzero entries per tet,
# so it is NOT easy.  It lands in hard_padding.
assert len(result.hard_padding) == 1
assert len(result.all_easy) == 0
```

### _is_easy unit tests
```python
assert _is_easy(np.array([2, 0, 0, 0, 3, 0]), n=2) == True
assert _is_easy(np.array([2, 1, 0, 0, 0, 0]), n=2) == False
```

---

## Tests to Write (`tests/test_phase_space.py`)

```python
"""Tests for easy-edge detection."""
import numpy as np
import pytest
from manifold_index.core.phase_space import _is_easy, find_easy_edges, EasyEdgeResult
from manifold_index.core.manifold import load_manifold


def test_is_easy_true():
    assert _is_easy(np.array([2, 0, 0, 0, 3, 0]), n=2)


def test_is_easy_false():
    assert not _is_easy(np.array([2, 1, 0, 0, 0, 0]), n=2)


def test_find_easy_edges_m004():
    pytest.importorskip("snappy")
    data = load_manifold("m004")
    result = find_easy_edges(data)
    assert isinstance(result, EasyEdgeResult)
    assert len(result.basis_edges) == data.num_tetrahedra - data.num_cusps


def test_find_easy_edges_m003_all_easy():
    pytest.importorskip("snappy")
    data = load_manifold("m003")
    result = find_easy_edges(data)
    assert len(result.basis_edges) == data.num_tetrahedra - data.num_cusps
    assert len(result.hard_padding) == 0  # m003 has all easy edges


def test_easy_edges_are_easy():
    pytest.importorskip("snappy")
    data = load_manifold("m004")
    result = find_easy_edges(data)
    for edge in result.all_easy:
        assert _is_easy(edge, data.num_tetrahedra)


def test_basis_edges_independent():
    """Basis edges must be linearly independent in the reduced space."""
    pytest.importorskip("snappy")
    from manifold_index.core.gluing_equations import _reduce_row
    data = load_manifold("m004")
    result = find_easy_edges(data)
    n = data.num_tetrahedra
    reduced_vecs = []
    for edge in result.basis_edges:
        _, rv = _reduce_row(edge, n)
        reduced_vecs.append(rv)
    if reduced_vecs:
        mat = np.array(reduced_vecs, dtype=float)
        assert np.linalg.matrix_rank(mat) == len(reduced_vecs)
```

---

*Phase 3 complete → proceed to Phase 4.*
