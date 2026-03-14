# SPEC.md — 3-Manifold Index Calculator

> **Auto-generated from codebase review.**  
> Describes the implemented system as of 2025-07.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Mathematical Pipeline](#3-mathematical-pipeline)
   - [Step 1 — Manifold Input](#step-1--manifold-input)
   - [Step 2 — Gluing Equations](#step-2--gluing-equations)
   - [Step 3 — Neumann-Zagier Matrix](#step-3--neumann-zagier-matrix)
   - [Step 4 — 3D Index](#step-4--3d-index)
   - [Step 5 — Dehn Filling & Non-closable Cycles](#step-5--dehn-filling--non-closable-cycles)
   - [Step 6 — Easy Edges & Phase Space](#step-6--easy-edges--phase-space)
   - [Step 7 — Basis Selection](#step-7--basis-selection)
   - [Step 8 — Refined Index](#step-8--refined-index)
   - [Step 9 — Weyl Checks](#step-9--weyl-checks)
   - [Step 10 — Refined Dehn Filling](#step-10--refined-dehn-filling)
4. [Data Structures](#4-data-structures)
5. [Notation & Conventions](#5-notation--conventions)
6. [GUI Application](#6-gui-application)
7. [Output Formats](#7-output-formats)
8. [Mathematica Integration](#8-mathematica-integration)
9. [Tests](#9-tests)
10. [Implementation Status](#10-implementation-status)

---

## 1. Project Overview

A PySide6 desktop application (+ CLI entry point) that computes
the **3D index**, **Dehn-filled index**, **refined index**, and
**refined Dehn-filled index** of ideal triangulated 3-manifolds,
starting from a SnaPy manifold name.

| Item | Value |
|------|-------|
| Package | `manifold-index` |
| Entry point | `manifold-index` → `manifold_index.app.main:main` |
| Python | ≥ 3.10 |
| Dependencies | `snappy`, `numpy`, `scipy`, `PySide6` |
| Dev deps | `pytest`, `pytest-cov`, `ruff`, `mypy` |
| Mathematica | Optional: `wolframscript` for fast tetrahedron-index cache generation |

---

## 2. Architecture

```
src/manifold_index/
├── core/                        # Pure mathematics — no GUI
│   ├── manifold.py              # Step 1: SnaPy extraction
│   ├── gluing_equations.py      # Step 2: variable reduction, symplectic form
│   ├── phase_space.py           # Step 6: easy-edge search
│   ├── neumann_zagier.py        # Step 3: NZ matrix construction
│   ├── index_3d.py              # Step 4: I(m,e) computation (Python + Mathematica paths)
│   ├── dehn_filling.py          # Step 5: Dehn filling kernel, non-closable cycles
│   ├── basis_selection.py       # Step 7: per-cusp cycle choice → (m_ext, e_ext)
│   ├── refined_index.py         # Step 8: refined index with η fugacities
│   ├── weyl_check.py            # Step 9: Weyl symmetry, (a,b) vectors, adjoint check
│   └── refined_dehn_filling.py  # Step 10: HJ-CF chain, IS kernel, refined filling
├── app/                         # GUI layer
│   ├── main.py                  # Application entry point
│   ├── interface.py             # CLI interface (if any)
│   ├── gui.py                   # PySide6 three-screen GUI
│   └── worker.py                # QThread workers for background computation
├── mathematica/                 # Wolfram Language scripts
│   ├── TetIndex.wl              # Tetrahedron index I_Δ(m,e) and cache
│   ├── Index3D.wl               # Summation of terms
│   └── ComputeIndex3D.wl        # Entry-point script for subprocess calls
└── data/
    └── tet_index/               # .mx cache files for Mathematica
```

The **pipeline** runs in two phases:
1. **Background** (`PipelineWorker`): Steps 1–5 run in a `QThread`.
2. **On-demand** (`RefinedIndexWorker`): Step 8 (refined index) runs
   after the user selects a basis in the GUI.

---

## 3. Mathematical Pipeline

### Step 1 — Manifold Input

**Module:** `core/manifold.py`  
**Input:** SnaPy manifold name (e.g. `"m003"`, `"L5a1"`)  
**Output:** `ManifoldData` dataclass

Extracts the **gluing matrix** (shape `(n+2r) × 3n`, integer) from SnaPy,
where:
- `n` = number of tetrahedra
- `r` = number of cusps

The gluing matrix has three groups of rows:
- Rows `0 … n−1`: edge equations (one per SnaPy edge)
- Rows `n+2k`: meridian equation for cusp `k`
- Rows `n+2k+1`: longitude equation for cusp `k`

Each row has columns `[f₁, g₁, h₁, f₂, g₂, h₂, …]` corresponding to the
three shape parameters `(Z_i, Z_i', Z_i'')` of each tetrahedron.

### Step 2 — Gluing Equations

**Module:** `core/gluing_equations.py`  
**Output:** `ReducedGluingData`

**Variable reduction.** Substitute `Z_i' = 1 − Z_i − Z_i''` for every
tetrahedron `i`, reducing from `3n` to `2n` variables:

```
v = (Z₁, Z₁'', Z₂, Z₂'', …, Zₙ, Zₙ'')    [interleaved ordering]
```

For a row `[f_i, g_i, h_i]` at tetrahedron `i`:
- Constant contribution: `g_i`
- Coefficient of `Z_i`: `f_i − g_i`
- Coefficient of `Z_i''`: `h_i − g_i`

**Independent edge basis.** The `n` SnaPy edge equations have rank `n − r`
in the reduced basis. A maximal linearly independent subset is selected
via column-pivoted QR on the coefficient matrix.

**Symplectic form.** On the reduced `2n`-dimensional space:

```
Ω[2i, 2i+1] = +1     (from [Z_i, Z_i''] = 1)
Ω[2i+1, 2i] = −1
all other entries = 0
```

Commutator: `[A, B] = a^T Ω b`.

### Step 3 — Neumann-Zagier Matrix

**Module:** `core/neumann_zagier.py`  
**Output:** `NeumannZagierData`

Constructs the symplectic matrix `g_NZ ∈ Sp(2n, ℚ)` and affine shift
vectors `ν_x`, `ν_p`.

**Column ordering:** block ordering
```
(Z₁, Z₂, …, Zₙ,  Z₁'', Z₂'', …, Zₙ'')
```
A permutation converts from the interleaved ordering used in Step 2.

**Row structure of `g_NZ`** (size `2n × 2n`):

| Row range | Content |
|-----------|---------|
| `0 … r−1` | Meridian equations (one per cusp) |
| `r … r+d_hard−1` | Hard internal edges |
| `r+d_hard … n−1` | Easy internal edges |
| `n … n+r−1` | Longitude / 2 (one per cusp) |
| `n+r … 2n−1` | Γ vectors (momentum conjugate to each internal edge) |

**Top n rows** = "position" variables; **bottom n rows** = "momentum" variables.

**Reduction convention.** The same substitution `Z_i' = 1 − Z_i − Z_i''` is used
as in Step 2. The constant and linear coefficients are identical to those
produced by `_reduce_row`. There is no separate "NZ convention" — the whole
project uses `Z_i + Z_i' + Z_i'' = 1` throughout.

**Symplectic form in block ordering:**

```
Ω_block = [[0_n,  I_n],
            [−I_n, 0_n]]
```

Pairing: `[u, v] = u_Z · v_{Z''} − u_{Z''} · v_Z = u @ Ω_block @ v`.

**Γ construction.** Given the `n × 2n` position block `P` and the first `r`
momentum rows (longitudes/2), the remaining `n − r` Γ rows are found by solving:

```
[P ; Q_long] Ω Γ^T = RHS
```
where RHS has shape `(n+r) × (n−r)` with `RHS[r:n, :] = I_{n−r}`, rest = 0.

**Inverse.** `g_NZ⁻¹` is computed exactly using the symplectic identity:
```
g⁻¹ = [[D^T, −B^T], [−C^T, A^T]]
```
where `g = [[A, B], [C, D]]` in `n × n` blocks. Returned as a `Fraction`
array for exact rational arithmetic.

**Cusp basis change.** `apply_cusp_basis_change(nz, cusp_idx, P, Q)` applies
an `SL(2,ℤ)` transformation to cusp `k`, used by Dehn filling (Step 5).

**Affine shift ν.** The reduced equation reads `c + coeff · v = RHS` where
`c = Σ_i g_i` (from `_reduce_row`). The affine shift satisfies
`g_NZ_row · v + ν = RHS`, so `ν = c − RHS`:

- **Meridian rows** (RHS = 0): `ν_x[k] = c` = `Σ_i g_i`
- **Internal edge rows** (RHS = 2): `ν_x[k] = c − 2` = `Σ_i g_i − 2`
- **Longitude/2 rows**: `ν_p[k] = c_long / 2` (halved because the momentum
  row stores `L/2`, not `L`)
- **Γ rows**: `ν_p[k] = 0` by construction.

### Step 4 — 3D Index

**Module:** `core/index_3d.py`  
**Output:** `Index3DResult`

**Tetrahedron index** `I_Δ(m, e; q)` (Garoufalidis–Kim formula):

```
I_t(m, e) = Σ_{n=max(0,−e)}^∞  (−1)^n · q^{n(n+1)/2 − (n + e/2)·m}
            ─────────────────────────────────────────────────────────
                     Q(q, n) · Q(q, n+e)

Q(q, n) = ∏_{k=1}^{n} (1 − q^k)
```

Symmetry (`MIt`):
```
if m + e ≥ 0:   MIt(m, e) = (−q^{1/2})^m · I_t(−m−e, m)
else:            MIt(m, e) = I_t(m, e)
```

**Degree formula** `δ(m, e)` (Lemma 3.6):
```
δ(m, e) = ½(m₊(m+e)₊ + (−m)₊ e₊ + (−e)₊(−e−m)₊) + max{0, m, −e}
```
where `x₊ = max{0, x}`.

**Convention:** `qq = q^{1/2}`. All series are polynomials in `qq` with
integer coefficients. The series variable `qq` is shared between TetIndex.wl
and Index3D.wl.

**3D index formula** (eq. 2.41):

```
I(m_ext, e_ext) =
   Σ_{e_int ∈ (½)ℤ^{n−r}}
       (−q^{1/2})^{ m_full · ν_p  −  e_full · ν_x }
       · ∏_{a=0}^{n−1} I_Δ( (g_NZ⁻¹ κ)_a,  (g_NZ⁻¹ κ)_{n+a} )
```

where:
- `m_full = (m_ext, 0^{n−r})` (internal edge m = 0)
- `e_full = (e_ext, e_int)` (internal edge e summed over)
- `κ = (m_full, e_full)` (size 2n)
- `m_ext, e_ext` have length `r` (cusp variables only)

**Integrality filter.** A term is zero whenever `g_NZ⁻¹ κ` has non-integer
entries. This is the ONLY constraint — no separate `φ∈ℤ` condition. When
local charges are integers the phase is automatically an integer.

**Summation bounds.** `enumerate_summation_terms` enumerates valid
half-integer patterns for `e_int` using the valid-half-integer-patterns
routine, which checks `g_NZ⁻¹ κ` integrality for all `4^{n−r}` parity
patterns.

**Two computation paths:**
1. **Pure Python** (`compute_index_3d_python`): evaluates each `I_Δ` as a
   polynomial and multiplies term-by-term with tighter per-tet cutoffs.
2. **Mathematica** (`compute_index_3d`): calls `wolframscript` with
   pre-enumerated terms in JSON; uses binary `.mx` caches for speed.

### Step 5 — Dehn Filling & Non-closable Cycles

**Module:** `core/dehn_filling.py`  
**Output:** `FilledIndexResult`, `NonClosableCycleResult`

**Setup.** For cusp `i`, position = M_i (meridian), momentum = Λ_i = L_i/2
(half-longitude). Dehn filling along `P·M + Q·L`:

**Kernel** `K(P, Q; m, e)`:
```
K(P, Q; m, e) = ½ (−1)^{Rm+2Se} ·
   [ δ_{Pm+2Qe, 0} · (q^{(Rm+2Se)/2} + q^{−(Rm+2Se)/2})
     − δ_{Pm+2Qe, −2}
     − δ_{Pm+2Qe, 2} ]
```

where `R, S ∈ ℤ` satisfy `R·Q − P·S = 1`, `m ∈ ℤ`, `e ∈ (½)ℤ`.

**`find_rs(P, Q)`:** Extended GCD to find `(R, S)` with `RQ − PS = 1`.

**Filled 3D index:**
```
I_{P/Q}^{(i)}(m_other, e_other) =
   Σ_{m_i, e_i}  K(P, Q; m_i, e_i) · I(m_all, e_all)
```

**Kernel term enumeration** (`enumerate_kernel_terms`):
- `c = 0` family: `m_i = Q·t`, `e_i = −P·t/2`, phase = `R·m + 2S·e`
- `c = ±2` families: particular solutions shifted by `t`
- Scanning terminates when `enumerate_summation_terms` returns no terms for
  two consecutive `|t|` values.

**Non-closable cycle search** (`find_non_closable_cycles`):
- A cycle `P·M + Q·L` is *non-closable* if `I_{P/Q}^{(i)} = 0` identically.
- Tested with `m_other = e_other = 0` for each other cusp.
- Scans slopes `P/Q` in a given range.

### Step 6 — Easy Edges & Phase Space

**Module:** `core/phase_space.py`  
**Output:** `EasyEdgeResult`

An **easy edge** is a non-negative integer linear combination of the SnaPy
edge equations such that at most one of `{Z_i, Z_i', Z_i''}` is non-zero per
tetrahedron `i`.

**Algorithm (pattern-first):** Enumerate all `4^n` patterns (one of
`{Z, Z', Z'', 0}` per tet). For each pattern, solve an overdetermined
linear system `M @ a = rhs` via least-squares to find integer coefficients
`a_i`. Valid easy edges satisfy:
- `a_i ≥ 0` for all `i`
- `Σ a_i · edge_row_i` evaluates to 2 under the normalization convention
- At most one active slot per tetrahedron

**Independent easy-edge selection:** QR with column pivoting selects a maximal
linearly independent subset of easy edges (in the reduced 2n-variable space).

**Hard padding.** If the number of independent easy edges is less than `n − r`,
hard edges from SnaPy rows are appended to complete the basis.

The **basis order** is: independent easy edges first, then hard padding,
giving `n − r` total independent internal edges.

### Step 7 — Basis Selection

**Module:** `core/basis_selection.py`  
**Output:** `BasisSelection`

After Step 5 identifies non-closable cycles at each cusp, the user chooses
one cycle per cusp. That cycle determines the external cusp variables
`(m_i, e_i)` for the refined index computation.

**Slope convention:**

A cycle at cusp `i` is specified by a slope `(P, Q)`, meaning the
homology class `P·μ + Q·λ` where `μ` is the meridian and `λ` is
the longitude.

- Meridian μ = slope (1, 0)
- Longitude λ = slope (0, 1)
- General filling cycle = slope (P, Q),  i.e. `P·μ + Q·λ`

**Data classes:**
- `CycleChoice`: one cusp's chosen cycle (cusp_idx, P, Q, label, is_default).
  Properties: `slope_str = "P/Q"`.
- `BasisSelection`: list of `CycleChoice` objects; exposes `m_ext`, `e_ext`.

`apply_basis_changes(nz_data, bs)` applies all cusp basis changes to the NZ
data, producing a modified `NeumannZagierData` with transformed meridian and
longitude rows.

### Step 8 — Refined Index

**Module:** `core/refined_index.py`  
**Output:** `RefinedIndexResult = dict[tuple[int, ...], int]`

The refined index attaches one formal fugacity variable `η_a` to each of the
`num_hard` hard internal edges. Hard edges occupy rows `r … r+num_hard−1`
of the position block of `g_NZ`, so their internal charges occupy positions
`0 … num_hard−1` inside `e_int`.

**Formula:**

```
I^ref(q; η₀, …, η_{k−1}) =
   Σ_{e_int ∈ (½)ℤ^{n−r}}
       [ ∏_{a=0}^{k−1}  η_a^{e_{r+a}} ]
       · (−q^{½})^{ m · ν_p  −  e · ν_x }
       · ∏_{j=0}^{n−1} I_Δ( (g_NZ⁻¹ κ)_j,  (g_NZ⁻¹ κ)_{n+j} )
```

where `k = num_hard` and `κ` / tet_arg assembly is identical to Step 4.

**Output key convention:**

```
key   = (q_half_power,  2·η₀_exp,  2·η₁_exp, …, 2·η_{k−1}_exp)
value = integer coefficient
```

All fugacity exponents are half-integers, so multiplying by 2 gives integers.
Setting all η = 1 (summing coefficients sharing the same `q_half_power`)
exactly recovers the ordinary 3D index.

**Projection:** `project_to_3d_index(refined)` sums over all η exponents.

**Display convention.** The internal keys store `exp_x2 = 2 × true_exponent`.
In formatted output the fugacity of hard edge `a` is written as:

```
η^{exp_x2 · v_a}
```

where `v_a` labels the hard-edge variable. For example, the key entry
`2·η₀_exp = 3` displays as `η^{3·v₀}` (representing `η₀^{3/2}`).

**Formatters:**
- `format_refined_index(result, num_hard, q_var, eta_vars)` → text string
- `format_multi_point_index(entries, num_hard, ...)` → multi-point text

### Step 9 — Weyl Checks

**Module:** `core/weyl_check.py`  
**Output:** `WeylCheckResult`

Before a cusp can be Dehn-filled in the *refined* index, three conditions
must hold:

1. **Non-closability** — handled by Step 5.
2. **Weyl symmetry** — the refined index can be made Weyl-manifest by
   multiplying by a monomial:

   ```
   f(m, e) = η^{b·m + a·e} · I(m, e)    is  η ↔ η⁻¹ symmetric
   ```

   where `a ∈ ℤ^{num_hard}` and `b ∈ (ℤ/2)^{num_hard}`.

3. **Adjoint su(2) character** — the coefficient of `q¹` (after stripping the
   leading η-monomial) must equal `η + 1 + η⁻¹` for each hard edge.

**Key formula for (a, b) extraction.** Let

```
centre(m, e) = (Σ_k  η_exp_k · coeff_k) / (Σ_k coeff_k)
```

be the coefficient-weighted centre of the η-polynomial at the leading
q-order for `I(m, e)`. Then:

```
b = −[centre(+m, 0) − centre(−m, 0)] / (2m)        (half-integer)
a = −[centre(0, +e) − centre(0, −e)] / Σ|e|         (integer)
```

**Internal storage convention:**
- `ABVectors.a` stores `2·a` (an integer)
- `ABVectors.b` stores `b` directly (a half-integer via `Fraction`)

**Functions:**

| Function | Description |
|----------|-------------|
| `extract_leading_eta_exponents(result, num_hard)` | Component-wise minimum η at leading q |
| `_eta_center_at_leading_q(result, num_hard)` | Weighted centre at leading q |
| `compute_ab_vectors(entries, num_hard)` | Main (a, b) computation from conjugate pairs |
| `check_weyl_symmetry(entries, num_hard, ab)` | Per-entry η ↔ η⁻¹ check |
| `strip_weyl_monomial(result, m_ext, e_ext, ab, num_hard)` | Factor out η monomial → Weyl-manifest f |
| `check_adjoint_character(result, leading_eta, num_hard, hard_idx)` | q¹ = η+1+η⁻¹ check |
| `run_weyl_checks(entries, num_hard, hard_idx)` | Aggregate all checks → `WeylCheckResult` |

**`WeylCheckResult` fields:** `ab`, `ab_valid`, `weyl_symmetric` (per-entry),
`all_weyl_symmetric`, `adjoint_checks` (per-entry).

### Step 10 — Refined Dehn Filling

**Module:** `core/refined_dehn_filling.py`  
**Output:** `FilledRefinedResult`

Implements the refined Dehn filling kernel `K^ref(P, Q; m, e; η)` using a
Hirzebruch-Jung continued fraction (HJ-CF) expansion.

**HJ-CF** of `P/Q`:
```
P/Q = k₁ − 1/(k₂ − 1/(… − 1/k_ℓ))
```

Special cases:
- `Q = 0, P = ±1` → `[0, 0]`
- `|Q| = 1` → `ℓ = 1, k = [P/Q]` (unrefined K suffices)

**Kernel chain** (eq. A.7):
```
K^ref(P,Q; m,e; η) =
   Σ_{m₁,e₁} … Σ_{m_{ℓ-1},e_{ℓ-1}}
       I_S(m,  −e − k₁/2·m,   m₁, e₁)
     · I_S(m₁, −e₁ − k₂/2·m₁, m₂, e₂)
     · …
     · K(k_ℓ, 1; m_{ℓ-1}, e_{ℓ-1})
```

**ẽI_S inner function** (expr8 in DFK.nb):
```
ẽI_S(m₁, e₁, m₂, e₂; η) =
   Σ_{e,t ∈ ℤ}  η^e
   · I_Δ(−e₁ − m₂/2,   −e/2 + e₁ + m₁/2 + t)
   · I_Δ( e₁ + m₂/2,   −e/2 + e₂ − m₂/2 + t)
   · I_Δ(−e₂ − m₁/2,    e₂ + m₁/2 + t)
   · I_Δ( e₂ + m₁/2,    e₁ − m₂/2 + t)
   · (−q^{1/2})^{−e + e₁ + e₂ + m₁/2 − m₂/2 + 2t}
```

**I_S kernel** (is[] in DFK.nb):
```
I_S = ½·(−1)^{m₁}·(qq^{m₁} + qq^{−m₁}) · ẽI_S(m₁, e₁, m₂, e₂)
    − ½·(−1)^{m₁} · ẽI_S(m₁, e₁−1, m₂, e₂)
    − ½·(−1)^{m₁} · ẽI_S(m₁, e₁+1, m₂, e₂)
```

**Data type.** `QEtaSeries = dict[tuple[int, int], Fraction]` where the key
is `(qq_power, eta_exp)` and value is a `Fraction` coefficient.

**Arithmetic helpers:** `_qeta_add`, `_qeta_scale`, `_qeta_shift_qq`,
`_qeta_truncate`, `_qeta_convolve`.

**Algorithm (ℓ ≥ 2):**
1. Compute HJ-CF `k = [k₁, …, k_ℓ]`.
2. Enumerate outer kernel terms `(m, e)` from the unrefined kernel support.
3. For each `(m, e)`: initialise `state[(m, e)] = I_{3D}(m, e)` as QEtaSeries
   with η⁰.
4. Apply `ℓ−1` IS convolution steps (`_apply_is_step`).
5. Apply final unrefined `K(k_ℓ, 1; ·)` to the last state (`_apply_k1_factor`).

**`FilledRefinedResult`** fields: `P`, `Q`, `cusp_idx`, `series` (QEtaSeries),
`qq_order`, `eta_order`, `hj_ks`, `n_kernel_terms`.  
Properties: `is_zero`, `eta1_series()`, `q_series_at_eta(val)`, `as_q_eta_string()`.

---

## 4. Data Structures

| Dataclass | Module | Key Fields |
|-----------|--------|------------|
| `ManifoldData` | manifold.py | `name`, `num_tetrahedra` (n), `num_cusps` (r), `gluing_matrix` (n+2r × 3n) |
| `ReducedGluingData` | gluing_equations.py | `edge_coeffs` (n × 2n), `edge_consts` (n,), `cusp_coeffs` (2r × 2n), `cusp_consts` (2r,), `independent_edge_indices`, `symplectic_matrix` (2n × 2n) |
| `EasyEdgeResult` | phase_space.py | `all_easy`, `independent_easy_indices`, `hard_padding`, `n`, `r` |
| `NeumannZagierData` | neumann_zagier.py | `g_NZ` (2n × 2n), `nu_x` (n,), `nu_p` (n,), `n`, `r`, `num_hard`, `num_easy` |
| `Index3DResult` | index_3d.py | `coeffs`, `min_power`, `q_order_half`, `m_ext`, `e_ext`, `n_terms` |
| `KernelTerm` | dehn_filling.py | `m`, `e`, `c`, `phase`, `multiplicity` |
| `FilledIndexResult` | dehn_filling.py | `series` (dict[int, Fraction]), `P`, `Q`, `cusp_idx`, etc. |
| `NonClosableCycle` | dehn_filling.py | `P`, `Q`, `slope_str`, `filled_result` |
| `NonClosableCycleResult` | dehn_filling.py | `cusp_idx`, `non_closable`, `closable`, `tested_slopes` |
| `CycleChoice` | basis_selection.py | `cusp_idx`, `P`, `Q`, `label`, `is_default` |
| `BasisSelection` | basis_selection.py | `choices` |
| `ABVectors` | weyl_check.py | `a` (list[Fraction]), `b` (list[Fraction]), `num_hard`, `warnings` |
| `WeylCheckResult` | weyl_check.py | `ab`, `ab_valid`, `weyl_symmetric`, `all_weyl_symmetric`, `adjoint_checks` |
| `FilledRefinedResult` | refined_dehn_filling.py | `P`, `Q`, `cusp_idx`, `series` (QEtaSeries), `qq_order`, `eta_order`, `hj_ks`, `n_kernel_terms` |

**Type aliases:**

| Alias | Definition |
|-------|------------|
| `RefinedIndexResult` | `dict[tuple[int, ...], int]` — key: `(q_half_power, 2·η₀_exp, …)` |
| `QEtaSeries` | `dict[tuple[int, int], Fraction]` — key: `(qq_power, eta_exp)` |

---

## 5. Notation & Conventions

### Paper ↔ Code symbol mapping

| Paper | Code | Meaning |
|-------|------|---------|
| r | `n` | Number of tetrahedra |
| n | `r` | Number of cusps |
| κ | `kappa` | Combined (m, e) vector of size 2n |
| ν_x | `nu_x` | Affine shift of position rows |
| ν_p | `nu_p` | Affine shift of momentum rows |
| q^{1/2} | `qq` | Series variable |

### q-series variable

- **Internal:** `qq = q^{1/2}`. All series are polynomials in `qq`.
- **`q_order_half`**: truncation order in units of `qq` (= `q^{1/2}`).
  So `q_order_half = 20` means series up to `qq^{20} = q^{10}`.

### η fugacity convention

- **Internal keys** store `exp_x2 = 2 × (true exponent)`, so all keys are plain `int`.
- **Display (text/LaTeX):** hard edge `a` shows as `η^{exp_x2 · v_a}`.
  - Example: `exp_x2 = 3` → `η^{3·v_a}` (meaning `η_a^{3/2}`).
- **Mathematica (.nb):** exported as `η[a]^{exp_x2}` (doubled exponent).

### Weyl-symmetry convention (multiplier)

The Weyl-manifest form is obtained by *multiplying* `I(m, e)` by a monomial:
```
f(m, e) = η^{b·m + a·e} · I(m, e)
```
where `f` is symmetric under `η ↔ η⁻¹`.

### Cusp variable convention

| Variable | Symbol | Domain |
|----------|--------|--------|
| Meridian | m | ℤ |
| Half-longitude | e | (½)ℤ |

In the evaluation grid: `m ∈ ℤ`, `e ∈ ½ℤ` (i.e. `e` can be half-integer).

---

## 6. GUI Application

**Module:** `app/gui.py` (≈1600 lines)  
**Framework:** PySide6 with `QStackedWidget` (3 screens)

### Screen 1 — Manifold Input (`ManifoldInputScreen`)

- Manifold name text field (SnaPy format)
- `q_order_half` spin box
- Slope range controls for non-closable cycle search
- "Compute" button → launches `PipelineWorker` (Steps 1–5)

### Screen 2 — Basis Selection (`BasisSelectionScreen`)

- Progress bar during computation
- Per-cusp radio buttons for cycle selection:
  - Non-closable cycles found in Step 5
  - Default meridian (1, 0) and longitude (0, 1)
- Weyl-check result display (once refined index is computed)
- "Compute Refined Index" button → launches `RefinedIndexWorker` (Step 8)
- Evaluation grid: generates multi-point results for Weyl checks

### Screen 3 — Output (`OutputScreen`)

- Formatted series display (text, LaTeX, Mathematica)
- Export buttons: `.txt`, `.tex`, `.json`, `.nb` (Mathematica notebook)
- Evaluation grid controls (m ∈ ℤ, e ∈ ½ℤ)
- Weyl-manifest display (η factored form + symmetry status)
- "Refined Dehn Filling" button (currently disabled — backend exists but
  GUI integration is not wired up)

### Workers (`app/worker.py`)

| Worker | Thread | Steps |
|--------|--------|-------|
| `PipelineWorker` | Background | 1 → 2 → 3 → 5 → 6 (manifold → NZ → easy edges → non-closable) |
| `RefinedIndexWorker` | Background | 7 → 8 → 9 (basis selection → refined index → Weyl checks) |

`PipelineResult`: `name`, `nz_data`, `q_order_half`, `cycle_results`.

### Formatters in gui.py

| Function | Output |
|----------|--------|
| `_series_to_latex(result, num_hard)` | LaTeX `q`·`η` polynomial |
| `_series_to_mathematica(result, num_hard)` | Mathematica `q`·`\[Eta]` expression |
| `_centre_to_latex(centre)` | LaTeX for the Weyl η-monomial exponent |
| `_fmt_centre_text(centre)` | Plain text for the Weyl η-monomial exponent |
| `_format_weyl_manifest_text(entries, num_hard, ab)` | Full Weyl-manifest display |
| `_build_nb_content(name, m_ext, e_ext, result, num_hard, q_ord)` | Mathematica .nb JSON |
| `_build_eval_grid(r, m_range, e_range)` | Unified grid m ∈ ℤ, e ∈ ½ℤ |

---

## 7. Output Formats

### Plain Text (`.txt`)

```
I(m=[1], e=[0]) = 2  -  q  +  3·q^2·η^{2·v_0}  + …
```

- Constant terms display as their coefficient (no `·1`)
- `q^1` displays as `q` (not `q^1`)
- Signs are properly separated: `… - q …` (not `… + -q …`)
- η variables use `η^{exp_x2 · v_a}` notation

### LaTeX (`.tex`)

```latex
I(m=[1], e=[0]) = 2 - q + 3\,q^{2}\,\eta^{2\,v_0} + \cdots
```

### Mathematica (`.nb`)

Generated as a Mathematica notebook JSON structure with:
- Function definition `IndexTable[m_, e_]` returning the series
- `SeriesData` wrapper with correct `qOrder = ⌈q_ord/2⌉`
- Zero-result entries included as `0` (not skipped)
- Pattern `e_` (not `e_Integer`) to allow half-integer arguments
- Usage example with half-integer `e` call

### JSON (`.json`)

Raw export of the `RefinedIndexResult` dictionary (keys as strings).

---

## 8. Mathematica Integration

Three Wolfram Language scripts in `src/manifold_index/mathematica/`:

| Script | Purpose |
|--------|---------|
| `TetIndex.wl` | Tetrahedron index `I_Δ(m, e)` + binary `.mx` cache |
| `Index3D.wl` | Sum-of-terms computation `ComputeOneTerm`, `Index3DFromTerms` |
| `ComputeIndex3D.wl` | Entry-point script for `wolframscript` subprocess |

**Communication protocol:** Python writes a JSON input file, calls
`wolframscript -file ComputeIndex3D.wl input.json output.json`, and
reads the JSON output.

**Cache system:** `.mx` binary files in `data/cache/<manifold>/` store
precomputed `I_Δ(m, e)` values. Cache key format:
`cache_m{mMax}_e{eMax}_q{qqOrder}.mx`.

---

## 9. Tests

| Test File | Tests | Modules Covered |
|-----------|-------|-----------------|
| `test_manifold.py` | 5 | manifold.py |
| `test_gluing_equations.py` | 6 | gluing_equations.py |
| `test_neumann_zagier.py` | 18 | neumann_zagier.py |
| `test_phase_space.py` | 16 | phase_space.py |
| `test_index_3d.py` | 45 | index_3d.py |
| `test_dehn_filling.py` | 98 | dehn_filling.py |
| `test_basis_selection.py` | 56 | basis_selection.py |
| `test_refined_index.py` | 18 | refined_index.py |
| `test_refined_dehn_filling.py` | 38 | refined_dehn_filling.py |
| `test_weyl_check.py` | 44 | weyl_check.py |
| **Total** | **344** | |

**Markers:**
- Many `test_dehn_filling.py` and `test_index_3d.py` tests are `@pytest.mark.slow`
- `test_ell2_eta1_matches_unrefined` in `test_refined_dehn_filling.py` is `@pytest.mark.xfail`
  (refined at η=1 doesn't exactly match unrefined due to truncation artefacts)

---

## 10. Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Step 1 — Manifold Input | ✅ Complete | `manifold.py` |
| Step 2 — Gluing Equations | ✅ Complete | `gluing_equations.py` |
| Step 3 — Neumann-Zagier | ✅ Complete | `neumann_zagier.py` |
| Step 4 — 3D Index | ✅ Complete | `index_3d.py` (Python + Mathematica) |
| Step 5 — Dehn Filling | ✅ Complete | `dehn_filling.py` |
| Step 6 — Easy Edges | ✅ Complete | `phase_space.py` |
| Step 7 — Basis Selection | ✅ Complete | `basis_selection.py` |
| Step 8 — Refined Index | ✅ Complete | `refined_index.py` |
| Step 9 — Weyl Checks | ✅ Complete | `weyl_check.py` |
| Step 10 — Refined Dehn Filling (backend) | ✅ Complete | `refined_dehn_filling.py` |
| GUI — Screen 1 (Input) | ✅ Complete | |
| GUI — Screen 2 (Basis Selection) | ✅ Complete | |
| GUI — Screen 3 (Output) | ✅ Complete | |
| GUI — Refined Dehn Filling button | 🔴 Not wired | Backend exists; GUI button disabled |
| Mathematica cache generation | ✅ Complete | `.mx` caching |
| Export: txt/LaTeX/JSON/nb | ✅ Complete | |

### Known Issues

1. **`test_ell2_eta1_matches_unrefined`** — `xfail`: refined Dehn filling
   at η=1 doesn't exactly match unrefined due to truncation artefacts.
2. **Refined Dehn filling in GUI** — backend is complete but the GUI button
   is disabled (`"not yet implemented"`).
