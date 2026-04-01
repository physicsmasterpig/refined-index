# Phase 8: Weyl Checks

## Dependencies
- Phase 7: `RefinedIndexResult`, `compute_refined_index`
- Phase 4: `NeumannZagierData`
- External: `fractions.Fraction`

## Files to Create
- `src/manifold_index/core/weyl_check.py`
- `tests/test_weyl_check.py`

---

## Overview

Before a cusp can be Dehn-filled in the *refined* index, three conditions
must hold:

1. **Non-closability** — handled by Phase 9 (Dehn filling).
2. **Weyl symmetry** — the refined index can be made η ↔ η⁻¹ symmetric
   by multiplying by a monomial `η^{a·e + b·m}`.
3. **Adjoint su(2) character** — the q¹ coefficient under adjoint
   projection must equal exactly −1.

---

## Data Classes

### `ABVectors`

```python
@dataclass
class ABVectors:
    a: list[Fraction]        # one per hard edge; couples to longitude e
    b: list[Fraction]        # one per hard edge; couples to meridian m
    num_hard: int
    warnings: list[str] = field(default_factory=list)

    @property
    def a_is_integer(self) -> list[bool]: ...
    @property
    def b_is_half_integer(self) -> list[bool]: ...     # 2·b[j] ∈ ℤ
    @property
    def is_valid(self) -> bool: ...                     # all integer a, all half-int b
    @property
    def edge_compatible(self) -> list[bool]: ...        # per-edge: a ∈ ℤ AND 2b ∈ ℤ
    def make_filling_compatible(self) -> ABVectors: ... # zeroes incompatible edges
```

### `AdjointProjectionResult`

```python
@dataclass
class AdjointProjectionResult:
    projected_value: int | None   # None if missing entries
    is_pass: bool                 # True iff projected_value == -1
    c_e: dict[Fraction, int]     # intermediate c_e values per e
    missing_e: list[Fraction]    # e-values needed but not found
```

### `WScanEntry`, `WScanResult`

```python
@dataclass
class WScanEntry:
    w: tuple[int, ...]           # W-vector
    a_eff: Fraction              # W · a
    b_eff: Fraction              # W · b
    a_eff_is_integer: bool
    adjoint: AdjointProjectionResult | None

@dataclass
class WScanResult:
    ab: ABVectors
    entries: list[WScanEntry]
    passing: list[WScanEntry]    # entries with adjoint.is_pass == True
```

### `WeylCheckResult`

```python
@dataclass
class WeylCheckResult:
    ab: ABVectors | None
    ab_valid: bool
    weyl_symmetric: dict[tuple, bool]   # per-entry symmetry check
    all_weyl_symmetric: bool
    adjoint: AdjointProjectionResult | None
```

---

## Core Algorithms

### `_eta_center_at_leading_q(result, num_hard) → list[Fraction] | None`

Coefficient-weighted centre of η-exponents at minimum q-half-power:
```
centre_j = (Σ_k η_j_exp_k · coeff_k) / (Σ_k coeff_k)
```
where sums are over terms at the leading q-power.
Returns None if empty, all-zero, or total_weight = 0.

### `compute_ab_vectors(entries, num_hard) → ABVectors | None`

From a table of `(m_ext, e_ext, RefinedIndexResult)` triples:

**b extraction** (meridian pairs with all e = 0, Σm > 0):
```
b[j] = −[centre_j(+m, 0) − centre_j(−m, 0)] / (2·Σ|m|)
```

**a extraction** (longitude pairs with all m = 0, Σe > 0):
```
a[j] = −[centre_j(0, +e) − centre_j(0, −e)] / (2·Σ|e|)
```

Fallback: if no meridian pairs, compare to zero-charge centre.
Consensus: use first estimate, warn if others disagree.

### `compute_ab_vectors_for_cusp(nz_data, cusp_idx, q_order_half=20)`

Numerically probes a single cusp by evaluating `compute_refined_index`
at a grid of charges (m ∈ {−2,…,2}, e ∈ {−1,−1/2,0,1/2,1}), then
extracts (a, b) using per-cusp logic.

### `check_weyl_symmetry(entries, num_hard, ab) → dict[tuple, bool]`

For each sector (m, e), compute `f(m,e) = η^{a·e + b·m} · I(m,e)`,
check `f(m,e) == f(−m,−e)` as formal power series.

### `check_adjoint_projection(entries, num_hard, ab=None, cusp_idx=0)`

For single cusp (d=1, n=0):
1. Collect m=0 entries, extract `c_e = coeff(q¹, η⁰)` of Weyl-shifted
   series `η^{a·e} · I^ref(m=0, e)`
2. Need e ∈ {−2, −1, +1, +2}
3. `proj = ½(c_{−1} + c_{+1} − c_{−2} − c_{+2})`
4. Check `proj == −1`

### `check_adjoint_with_w_vector(entries, num_hard, ab, w, cusp_idx=0)`

Projects multi-η polynomial onto single variable via W-vector:
`combined_x2 = Σ_j W_j · key[1+j]`. Same adjoint check on projected variable.

### `scan_w_vectors(entries, num_hard, ab, cusp_idx=0, max_coeff=3)`

Enumerate all integer W-vectors with |W_j| ≤ max_coeff (canonical: first
nonzero entry positive). For each, check adjoint projection. Return
`WScanResult` with passing entries.

### `run_weyl_checks(entries, num_hard, cusp_idx=0) → WeylCheckResult`

Convenience: calls `compute_ab_vectors`, `check_weyl_symmetry`,
`check_adjoint_projection`, aggregates into `WeylCheckResult`.

---

## Tests (`tests/test_weyl_check.py`)

```python
"""Tests for Weyl symmetry checks."""
from fractions import Fraction
import pytest
from manifold_index.core.weyl_check import (
    ABVectors, extract_leading_eta_exponents, check_adjoint_projection,
    check_adjoint_with_w_vector, scan_w_vectors,
)


def test_extract_leading():
    assert extract_leading_eta_exponents({(4, 2): 3}, 1) == [Fraction(1)]


def test_adjoint_projection_pass():
    """1/2(c_{-1} + c_{+1} - c_{-2} - c_{+2}) = -1."""
    entries = [
        ([0], [Fraction(-2)], {(2, 0): 2}),
        ([0], [Fraction(-1)], {(2, 0): 1}),
        ([0], [Fraction(0)],  {(2, 0): 99}),
        ([0], [Fraction(1)],  {(2, 0): 1}),
        ([0], [Fraction(2)],  {(2, 0): 2}),
    ]
    result = check_adjoint_projection(entries, num_hard=1, cusp_idx=0)
    assert result.is_pass is True
    assert result.projected_value == -1


def test_adjoint_projection_fail():
    entries = [
        ([0], [Fraction(-2)], {(2, 0): 0}),
        ([0], [Fraction(-1)], {(2, 0): 1}),
        ([0], [Fraction(0)],  {(2, 0): 0}),
        ([0], [Fraction(1)],  {(2, 0): 1}),
        ([0], [Fraction(2)],  {(2, 0): 0}),
    ]
    result = check_adjoint_projection(entries, num_hard=1, cusp_idx=0)
    assert result.is_pass is False
    assert result.projected_value == 1


def test_m004_ab_vectors(nz_m004):
    from manifold_index.core.weyl_check import compute_ab_vectors_for_cusp
    ab = compute_ab_vectors_for_cusp(nz_m004, cusp_idx=0, q_order_half=20)
    assert ab is not None
    assert ab.a == [Fraction(1, 2)]
    assert ab.b == [Fraction(-1, 2)]
```

---

*Phase 8 complete → proceed to Phase 9.*
