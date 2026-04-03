# Phase 5: Basis Selection

## Dependencies
- Phase 4: `NeumannZagierData`, `apply_cusp_basis_change()`
- External: `fractions.Fraction`, `math.gcd`

## Files to Create
- `src/manifold_index/core/basis_selection.py`
- `tests/test_basis_selection.py`

---

## Overview

After Dehn filling (Phase 9) identifies non-closable cycles at each cusp,
the user chooses one cycle per cusp. That cycle fixes the external cusp
variables `(m_ext, e_ext)` for the refined-index computation.

**Translation rule**:
```
Cycle P·M + Q·L at cusp k  →  m_ext[k] = P,  e_ext[k] = Q/2
```

**Defaults** (when no non-closable cycle is found):
| Curve      | Slope  | m   | e     |
|------------|--------|-----|-------|
| Meridian M | (1, 0) | 1   | 0     |
| Longitude L| (0, 1) | 0   | 1/2   |

---

## Data Classes

### `CycleChoice`

```python
@dataclass
class CycleChoice:
    cusp_idx: int
    P: int                  # primitive pair defining P·M + Q·L
    Q: int
    label: str = ""         # auto-generated if empty
    is_default: bool = False

    @property
    def m(self) -> int: return self.P
    @property
    def e(self) -> Fraction: return Fraction(self.Q, 2)
    @property
    def slope_str(self) -> str: return f"{self.P}/{self.Q}"
```

**Validation in `__post_init__`**:
- (P, Q) ≠ (0, 0)
- gcd(|P|, |Q|) == 1 (primitive)
- Auto-generate label if empty: "meridian M (1/0)", "longitude L (0/1)",
  or "slope P/Q"

### `BasisSelection`

```python
@dataclass
class BasisSelection:
    choices: list[CycleChoice]   # length r, choices[i].cusp_idx == i

    @property
    def r(self) -> int: ...
    @property
    def m_ext(self) -> list[int]: ...
    @property
    def e_ext(self) -> list[Fraction]: ...
    def summary(self) -> str: ...
```

**Validation in `__post_init__`**:
- choices non-empty
- choices[i].cusp_idx == i for all i

---

## Convenience Constructors

```python
def default_meridian_choice(cusp_idx: int) -> CycleChoice:
    return CycleChoice(cusp_idx, P=1, Q=0, label="meridian M (1/0)", is_default=True)

def default_longitude_choice(cusp_idx: int) -> CycleChoice:
    return CycleChoice(cusp_idx, P=0, Q=1, label="longitude L (0/1)", is_default=True)
```

---

## `make_basis_selection(nz_data, cycle_results, choices, *, default="M", strict=False)`

**Parameters**:
- `nz_data`: provides `r` (number of cusps)
- `cycle_results`: `list[NonClosableCycleResult]` (from Phase 9), may be empty
- `choices`: `list[tuple[int,int] | None]`, length r. None → use default.
- `default`: `"M"` or `"L"` — which default curve for None entries
- `strict`: if True, raise ValueError if chosen slope not in cycle_results

**Algorithm**:
1. Build lookup: `found_slopes[cusp_idx] = {(P,Q), ...}` from cycle_results
2. For each cusp i:
   - None → default_meridian_choice or default_longitude_choice
   - (P,Q) → validate primitive, optionally strict-check against found_slopes
   - Determine label and is_default flag
3. Return `BasisSelection(choices=cusp_choices)`

---

## `apply_basis_changes(nz_data, basis) → NeumannZagierData`

For each cusp k with slope (P_k, Q_k):
- If P_k is **odd**: call `apply_cusp_basis_change(nz_data, k, P_k, Q_k)`
- If P_k is **even** (including P=0 for longitude): skip this cusp — do NOT
  call `apply_cusp_basis_change`.

Returns a `NeumannZagierData` where only odd-P cusps have had their basis
changed.  Even-P cusps retain the original meridian/longitude basis in the
returned object.

> **What "skip" means for even-P cusps:** The caller evaluates the refined
> index at `(m_ext[k] = P_k, e_ext[k] = Q_k/2)` using the **unchanged**
> basis for that cusp.  There is no cusp basis transformation applied — the
> original (M, L/2) rows of g_NZ for that cusp are left intact.
> Example: longitude choice P=0, Q=1 → skip basis change, evaluate at m=0, e=1/2.

Apply the changes sequentially: `nz = nz_data` then for each odd-P cusp
`nz = apply_cusp_basis_change(nz, k, P_k, Q_k)`.  Return the final `nz`.

---

## Tests (`tests/test_basis_selection.py`)

```python
"""Tests for basis selection."""
from fractions import Fraction
import pytest
from manifold_index.core.basis_selection import (
    CycleChoice, BasisSelection, make_basis_selection,
    default_meridian_choice, default_longitude_choice,
)


def test_zero_zero_raises():
    with pytest.raises(ValueError):
        CycleChoice(cusp_idx=0, P=0, Q=0)


def test_non_primitive_raises():
    with pytest.raises(ValueError):
        CycleChoice(cusp_idx=0, P=2, Q=4)


def test_cycle_choice_properties():
    cc = CycleChoice(cusp_idx=0, P=3, Q=2)
    assert cc.m == 3
    assert cc.e == Fraction(1)  # 2/2
    assert cc.slope_str == "3/2"


def test_make_basis_selection_with_stub():
    class _Stub:
        r = 1
    bs = make_basis_selection(_Stub(), [], [(2, 3)])
    assert bs.m_ext == [2]
    assert bs.e_ext == [Fraction(3, 2)]


def test_default_choices():
    m_ch = default_meridian_choice(0)
    assert m_ch.m == 1 and m_ch.e == Fraction(0) and m_ch.is_default
    l_ch = default_longitude_choice(0)
    assert l_ch.m == 0 and l_ch.e == Fraction(1, 2) and l_ch.is_default


def test_basis_selection_ordering():
    """choices must have cusp_idx == i."""
    with pytest.raises(ValueError):
        BasisSelection(choices=[CycleChoice(cusp_idx=1, P=1, Q=0)])
```

---

*Phase 5 complete → proceed to Phase 6.*
