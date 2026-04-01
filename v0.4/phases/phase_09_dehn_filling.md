# Phase 9: Dehn Filling

## Dependencies
- Phase 6: `Index3DResult`, `compute_index_3d_python`, `enumerate_summation_terms`
- Phase 4: `NeumannZagierData`
- External: `fractions.Fraction`, `math.gcd`, `time`

## Files to Create
- `src/manifold_index/core/dehn_filling.py`
- `tests/test_dehn_filling.py`

---

## Overview

For each cusp i, the Dehn filling kernel at slope P/Q implements:

```
K(P, Q; m, e) = ½ (−1)^{Rm+2Se} ·
  [ δ_{Pm+2Qe, 0} · (q^{(Rm+2Se)/2} + q^{−(Rm+2Se)/2})
    − δ_{Pm+2Qe, −2}
    − δ_{Pm+2Qe, 2} ]
```

where R·Q − P·S = 1, m ∈ ℤ, e ∈ (½)ℤ.

**Filled index**: `I_{P/Q}^{(i)} = Σ_{m_i, e_i} K(P,Q; m_i, e_i) · I(m, e)`

**Non-closable cycle**: P·M + Q·L is non-closable if I_{P/Q}^{(i)} = 0.

---

## Part 1: Extended GCD & R,S Finder

### `_ext_gcd(a, b) → (g, x, y)` with a·x + b·y = g

### `find_rs(P, Q) → (R, S)` with R·Q − P·S = 1
Solve via `_ext_gcd(|Q|, |P|)`, correct for signs. Raises ValueError
if gcd(|P|, |Q|) ≠ 1.

---

## Part 2: Kernel Term Enumeration

### `KernelTerm` (frozen dataclass)

```python
@dataclass(frozen=True)
class KernelTerm:
    m: int
    e: Fraction
    c: int              # ∈ {−2, 0, 2}: value of P·m + 2Q·e
    phase: int          # R·m + 2S·e
    multiplicity: int = 1  # 2 for c=0 with |t|>0 or c=2 (antipodal symmetry)
```

Kernel factor per term type:
- `c=0`: `½·(−1)^phase · (q^{phase/2} + q^{−phase/2})`
- `c=±2`: `−½·(−1)^phase` (constant)

### `_particular_solution(P, Q, c) → (m0, e0)`

Find (m0, e0) with P·m0 + 2Q·e0 = c via ext_gcd(P, Q).

### `enumerate_kernel_terms(...) → list[KernelTerm]`

For each c ∈ {0, 2} (c = −2 handled by c = 2 antipodal symmetry):
- Particular solution (m_c, e_c), then family m_t = m_c + Q·t, e_t = e_c − P·t/2
- phase_t = phase_c0 + t
- Inclusion: `min_degree(I_{3D}) ≤ adjusted_q` where:
  - c=0: adjusted_q = q_order_half + |phase_t|/2
  - c=±2: adjusted_q = q_order_half
- Scan t outward, stop after 2 consecutive misses (convexity)
- Antipodal: c=0 with |t|>0 gets multiplicity=2; c=2 always multiplicity=2
- Cache summation terms in `_summation_cache` dict for reuse

---

## Part 3: q-series Arithmetic

```python
QSeries = dict[int, Fraction]   # key k → coeff of q^{k/2}
```

Helpers:
- `_qseries_from_result(Index3DResult) → QSeries`
- `_qseries_shift(s, power_shift) → QSeries`
- `_qseries_scale(s, scalar) → QSeries`
- `_qseries_add(a, b) → QSeries`
- `_qseries_truncate(s, q_order_half) → QSeries`

### `_apply_kernel(term, index_series, q_order_half=None) → QSeries`

- c=0: `½·sign · (shift(+phase) + shift(−phase))` of index_series
  (skip upward shift if |phase| > q_order_half)
- c=±2: `−½·sign · index_series`

---

## Part 4: Filled Index Computation

### `FilledIndexResult`

```python
@dataclass
class FilledIndexResult:
    P: int; Q: int; cusp_idx: int
    series: QSeries
    q_order_half: int
    n_kernel_terms: int

    @property
    def is_zero(self) -> bool: ...
    def is_stably_zero(self, buffer=None) -> bool: ...
    def as_polynomial_string(self, var="q") -> str: ...
```

`is_stably_zero(buffer)`: ignores top `buffer` powers near truncation
(default: min(max(5, q//2), q−1)) — boundary artifacts from truncated
upward shifts.

### `compute_filled_index(nz_data, cusp_idx, P, Q, m_other, e_other, q_order_half, verbose)`

1. **Step A**: `enumerate_kernel_terms(…, _summation_cache={})`
2. **Step B**: For each kernel term:
   - `compute_index_3d_python(…, _precomputed_terms=cache)` → index_series
   - `_apply_kernel(term, index_series)` → contribution
   - Multiply by multiplicity, accumulate
3. Truncate, return `FilledIndexResult`

---

## Part 5: Non-Closable Cycle Search

### `NonClosableCycle`, `NonClosableCycleResult`

```python
@dataclass
class NonClosableCycle:
    cusp_idx: int; P: int; Q: int

@dataclass
class NonClosableCycleResult:
    cusp_idx: int
    cycles: list[NonClosableCycle]
    slopes_tested: list[tuple[int, int]]
```

### `_candidate_slopes(p_range, q_range, canonical_only=False)`

Return all primitive (P, Q) with P ∈ p_range, Q ∈ q_range.
When Q=0, only P>0 (avoid ±1/0 duplication). When `canonical_only`,
keep only Q>0 representative (exploit `I_{P/Q} = I_{−P/−Q}`).

### `find_non_closable_cycles(nz_data, cusp_idx, p_range, q_range, ...)`

For each primitive slope: `compute_filled_index` → `is_stably_zero()`?
Antipodal symmetry: compute canonical half, mirror result.
Default ranges: p ∈ [−3, 3], q ∈ [0, 3].

---

## Tests (`tests/test_dehn_filling.py`)

```python
"""Tests for Dehn filling."""
from fractions import Fraction
from manifold_index.core.dehn_filling import (
    KernelTerm, _apply_kernel, _particular_solution, find_rs,
)


def test_find_rs():
    R, S = find_rs(3, 2)
    assert R * 2 - 3 * S == 1


def test_particular_solution():
    m0, e0 = _particular_solution(3, 2, 2)
    assert 3 * m0 + 2 * 2 * e0 == 2


def test_apply_kernel_identity():
    """c=0, phase=0 → K = ½·1·(q⁰+q⁰) = I, so result = input series."""
    term = KernelTerm(m=0, e=Fraction(0), c=0, phase=0)
    s = {0: Fraction(1), 2: Fraction(3)}
    assert _apply_kernel(term, s) == s
```

---

*Phase 9 complete → proceed to Phase 10.*
