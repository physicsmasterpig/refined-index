# Phase 7: Refined Index

## Dependencies
- Phase 6: `_tet_index_series`, `enumerate_summation_terms`, `_get_enum_state`, `_enumerate_with_state`, `Index3DResult`
- Phase 4: `NeumannZagierData`
- External: `fractions.Fraction`

## Files to Create
- `src/manifold_index/core/refined_index.py`
- `tests/test_refined_index.py`

---

## Overview

The refined index attaches one formal fugacity variable η_a to each of the
`num_hard` hard internal edges. Hard edges occupy rows `r…r+num_hard−1`
of the position block of g_NZ, and their internal charges occupy positions
`0…num_hard−1` inside the `e_int` vector.

### Formula

```
I^ref(q; η_0, …, η_{k-1}) =
    Σ_{e_int ∈ (½)Z^{n-r}}
        [ ∏_{a=0}^{k-1}  η_a^{e_{r+a}} ]
        · (−q^½)^{ m · ν_p  −  e · ν_x }
        · ∏_{j=0}^{n-1} I_Δ( (g_NZ⁻¹ κ)_j , (g_NZ⁻¹ κ)_{n+j} )
```
where k = num_hard.

### Output Key Convention

```python
RefinedIndexResult = dict[tuple[int, ...], int]
# key = (q_half_power, 2*η_0_exp, 2*η_1_exp, …, 2*η_{k-1}_exp)
# value = integer coefficient
```

All fugacity exponents are half-integers; multiplying by 2 gives ints.
Setting all η = 1 (summing keys sharing same q_half_power) recovers
the ordinary 3D index.

When `num_hard = 0`, keys are `(q_half_power,)` — the ordinary 3D index.

---

## Public API

### `compute_refined_index(nz_data, m_ext, e_ext, q_order_half=20) → RefinedIndexResult`

Identical to `compute_index_3d_python` except:
1. Extract first `k = num_hard` entries of `e_int` as fugacity exponents
2. `eta_exps_x2 = tuple(int(term["e_int"][a] * 2) for a in range(k))`
   (`e_int` entries are `Fraction` objects — see Phase 6 Part G; no string parsing needed)
3. Key becomes `(shifted_qq_power,) + eta_exps_x2`

Algorithm:
- Call `enumerate_summation_terms(nz_data, m_ext, e_ext, q_order_half)`
- For each term:
  - Extract `eta_exps_x2` from first k entries of `e_int` (list[Fraction])
  - Multiply tet index series ∏ I_Δ(m_a, e_a) as polynomial dicts
  - Apply phase factor `(-1)^{phase_exp} · qq^{phase_exp}`
  - Accumulate into `result[(shifted,) + eta_exps_x2]`
- Remove zero entries

### `compute_refined_index_batch(nz_data, entries, q_order_half=20) → list[RefinedIndexResult]`

Computes I^ref for multiple (m_ext, e_ext) pairs. Pre-computes
`_get_enum_state(nz_data)` once, reuses across all entries via
`_enumerate_with_state`. Much faster than calling `compute_refined_index`
individually.

### `project_to_3d_index(refined) → dict[int, int]`

Sum all fugacity monomials: `out[q_pow] = Σ refined[(q_pow, ...)]`.

### `format_refined_index(refined, num_hard, q_var="q", eta_vars=None) → str`

Human-readable string. Fugacity convention: `η_a = η^{2W_a}`.
Stored `exp_x2 = 2 × true_exponent`, so display as `η^(exp_x2 · W_a)`.

### `format_multi_point_index(entries, num_hard, ...) → str`

Multi-line display of `I(m, e) = series` for a list of evaluation points.

---

## Tests (`tests/test_refined_index.py`)

```python
"""Tests for refined index."""
from manifold_index.core.refined_index import (
    compute_refined_index, project_to_3d_index,
)


def test_projection_arithmetic():
    assert project_to_3d_index({(2, 2): 1, (2, 0): 1, (2, -2): 1}) == {2: 3}
    assert project_to_3d_index({(4, 2): 1, (4, -2): -1}) == {}


def test_projection_matches_3d_index(nz_m004):
    from manifold_index.core.index_3d import compute_index_3d_python
    q_ord = 12
    refined = compute_refined_index(nz_m004, [0], [0], q_order_half=q_ord)
    projected = project_to_3d_index(refined)
    res3d = compute_index_3d_python(nz_m004, [0], [0], q_order_half=q_ord)
    expected = {res3d.min_power + k: c for k, c in enumerate(res3d.coeffs) if c != 0}
    assert projected == expected
```

---

*Phase 7 complete → proceed to Phase 8.*
