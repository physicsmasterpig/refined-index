# Shared Conventions for All Phases

> **Read this file before every phase.** It contains rules that apply
> everywhere and must not be violated.

---

## 1. Naming Conventions

| Code symbol | Meaning |
|:------------|:--------|
| `n` | Number of tetrahedra |
| `r` | Number of cusps |
| `num_hard` | Number of hard internal edges |
| `num_easy` | Number of easy internal edges (= n - r - num_hard) |
| `qq` / `qq_order` / `q_order_half` | Power of q^{1/2}; max half-power kept |

> **WARNING:** The paper uses r for tetrahedra and n for cusps — the
> **opposite** of the code.  The code convention (n=tet, r=cusps) is used
> consistently.  Never mix them.

---

## 2. Package Layout

```
src/manifold_index/
    __init__.py           → "manifold_index — 3-Manifold Index Calculator"
    core/
        __init__.py       → "core package — mathematical pipeline modules"
        manifold.py
        gluing_equations.py
        phase_space.py
        neumann_zagier.py
        basis_selection.py
        index_3d.py
        refined_index.py
        weyl_check.py
        dehn_filling.py
        refined_dehn_filling.py
        kernel_cache.py
        _c_kernel/
            tet_index.c
    utils/
        __init__.py       → "utils package — shared helper functions"
        exporters.py
        math_utils.py
        io_utils.py
    app/
        __init__.py
        __main__.py
        window.py
        workers.py
        formatters.py
        katex.py
        style.py
        panels/
            __init__.py
            manifold_panel.py
            filling_panel.py
            export_panel.py
            kernel_panel.py
            data_panel.py
    data/
        kernel_cache/     (bundled .pkl.gz)
```

---

## 3. Type Aliases (defined once used)

```python
# In index_3d.py or a shared types module:
from fractions import Fraction

QSeries            = dict[int, Fraction]             # qq_power → coeff
RefinedIndexResult = dict[tuple[int, ...], int]      # (qq, 2·e_int_0, …) → coeff
QEtaSeries         = dict[tuple[int, int], Fraction] # (qq_pow, η_exp) → coeff
MultiEtaSeries     = dict[tuple[int, ...], Fraction] # multi-dim key → coeff
```

---

## 4. Doubled-Exponent Convention

All η fugacity exponents are half-integers (e_int can be k/2).
**Always store `2 × exponent` as `int`** in dict keys.

A key `(qq_pow, 4, -2)` for a 2-hard-edge manifold means:
- q^{qq_pow/2}
- hard edge 0 exponent = 4/2 = 2
- hard edge 1 exponent = -2/2 = -1

In the article notation: η^{2W_0·2 + 2W_1·(-1)} = η^{4W_0 - 2W_1}.

---

## 5. Content-Based Cache Keys

**NEVER use `id(obj)` as a cache key.** Python GC reuses addresses.

For NeumannZagierData:
```python
def _nz_content_key(nz_data):
    return (
        nz_data.g_NZ.data.tobytes(),
        nz_data.nu_x.data.tobytes(),
        nz_data.nu_p.data.tobytes(),
    )
```

---

## 6. Fraction vs Float

- **Exact arithmetic:** Use `fractions.Fraction` (e_ext values, g_NZ_inv).
- **NumPy arrays:** `g_NZ` stores exact rationals as `float64`.
  The exact inverse `g_NZ_inv()` returns a Fraction array.
- **Hot-path integer trick:** `g_NZ_inv_scaled()` returns `(S, int64_array)`
  where `S` is the LCD and `int64_array = S · g_NZ_inv`, all entries integer.

---

## 7. Data Structures Are Frozen

All major containers (`ManifoldData`, `ReducedGluingData`,
`EasyEdgeResult`, `NeumannZagierData` fields, etc.) should be
**dataclasses**.  Use `frozen=True` where possible.

Exception: `NeumannZagierData` uses `@dataclass` (not frozen) because
cached properties (`g_NZ_inv`, `g_NZ_inv_scaled`) are set lazily.

---

## 8. Import Style

```python
from __future__ import annotations          # every file
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Sequence

import numpy as np
```

Intra-package imports use absolute paths:
```python
from manifold_index.core.manifold import ManifoldData
from manifold_index.core.gluing_equations import ReducedGluingData
```

---

## 9. Testing

- All tests require `snappy` → use `pytest.importorskip("snappy")`.
- Shared fixtures are in `tests/conftest.py` (session-scoped).
- Test file names match source: `core/foo.py` → `tests/test_foo.py`.

---

## 10. Symplectic Matrix (Interleaved Ω)

In the reduced 2n-variable basis (Z₁, Z₁'', Z₂, Z₂'', …):
```
Ω[2i, 2i+1] = +1
Ω[2i+1, 2i] = -1
all other entries = 0
```
This is the **interleaved** form.

In the block basis (Z₁,…,Z_n, Z₁'',…,Z_n''):
```
Ω_block = [[0, I_n], [-I_n, 0]]
```

Conversion: `_interleaved_to_block(coeff_2n, n)` permutation.

---

## 11. Column Orderings

| Context | Ordering | Example (n=2) |
|:--------|:---------|:--------------|
| Gluing matrix (3n cols) | Interleaved triples | Z₁, Z₁', Z₁'', Z₂, Z₂', Z₂'' |
| Reduced basis (2n cols) | Interleaved pairs | Z₁, Z₁'', Z₂, Z₂'' |
| g_NZ matrix (2n cols) | Block | Z₁, Z₂, Z₁'', Z₂'' |

---

## 12. Gluing Matrix Row Layout

```
Row 0     … n-1      : edge equations    (n rows)
Row n + 2k            : meridian μ_k     (k = 0…r-1)
Row n + 2k+1          : longitude λ_k    (k = 0…r-1)
```

Cusp rows are **interleaved**: μ₀, λ₀, μ₁, λ₁, …

---

*End of shared conventions.*
