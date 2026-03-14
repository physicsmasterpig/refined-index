# Project Specification — 3-Manifold Index Calculator

> **This file is the single source of truth for all mathematical definitions,
> conventions, algorithms, and decisions made during development.**
> It is updated continuously as new details are agreed upon.
> When in doubt, this file takes precedence over any assumption.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Pipeline Summary](#2-pipeline-summary)
3. [Step-by-Step Details](#3-step-by-step-details)
   - [Step 1 — Manifold Input](#step-1--manifold-input)
   - [Step 2 — SnaPy Data Extraction](#step-2--snappy-data-extraction)
   - [Step 3 — Neumann-Zagier Matrix & Affine Shift](#step-3--neumann-zagier-matrix--affine-shift)
   - [Step 4 — 3D Index Calculation](#step-4--3d-index-calculation)
   - [Step 5 — Dehn Filling & Non-Closable Cycles](#step-5--dehn-filling--non-closable-cycles)
   - [Step 6 — Basis Selection](#step-6--basis-selection)
   - [Step 7 — Easy Edges & Phase Space Basis](#step-7--easy-edges--phase-space-basis)
   - [Step 8 — Refined Index](#step-8--refined-index)
4. [Data Structures & Types](#4-data-structures--types)
5. [Input / Output Spec](#5-input--output-spec)
6. [Dependencies](#6-dependencies)
7. [Packaging & Distribution](#7-packaging--distribution)
8. [GUI Workflow](#8-gui-workflow)
9. [Open Questions](#9-open-questions)
10. [Changelog](#10-changelog)

---

## 1. Project Overview

| Field | Value |
|---|---|
| Project name | Refined Index Calculator |
| Language | Python 3.x |
| Primary libraries | SnaPy, PyQt6 (or PySide6), (others TBD) |
| Target platform | macOS (primary), GUI application |
| Status | In development |

**Goal:** Given a 3-manifold name, compute the *refined index* by walking through a
well-defined sequence of intermediate computations involving the 3D index,
Dehn filling, non-closable cycles, and a phase space basis derived from the
gluing equations.

---

## 2. Pipeline Summary

```
User input: manifold name
        │
        ▼
[Step 2] SnaPy extraction
         · number of cusps
         · number of tetrahedra
         · gluing equations
        │
        ▼
[Step 3] Neumann-Zagier matrix & affine shift
         · g_NZ ∈ Sp(2n, Z), affine shifts ν_x, ν_p
         · position block: meridians, hard edges, easy edges
         · momentum block: longitude/2, Γ vectors
        │
        ▼
[Step 4] 3D Index  I(m⃗, e⃗)
         · computed over a range of (m⃗, e⃗)
         · range: default or user-specified
         · ⚠️ algorithm TBD
        │
        ▼
[Step 5] Dehn filling on each cusp
         · range of slopes: default or user-specified
         · search for non-closable cycles
        │
        ▼
[Step 6] Basis selection (interactive)
         · display non-closable cycles per cusp
         · user selects one cycle per cusp as basis
         · if no non-closable cycle → use default curve
        │
        ▼
[Step 7] Easy edges from gluing equations
         · construct phase space basis
        │
        ▼
[Step 8] Refined index
         · output / export
```

---

## 3. Step-by-Step Details

### Step 1 — Manifold Input

- **Input method:** TBD (CLI argument / interactive prompt / GUI field)
- **Format:** SnaPy manifold name string (e.g. `"m004"`, `"4_1"`, etc.)
- **Validation:** confirm manifold is recognized by SnaPy before proceeding
- ⚠️ *Details to be filled in.*

---

### Step 2 — SnaPy Data Extraction

- **Library:** `snappy`
- **SnaPy API:**
  - `M = snappy.Manifold(name)`
  - `M.num_tetrahedra()` → `n`  (int)
  - `M.num_cusps()` → `r`  (int)
  - `M.gluing_equations()` → `SimpleMatrix`, shape `(n + 2r, 3n)`
    - Convert to numpy: `np.array(eqs.list(), dtype=int).reshape(n + 2*r, 3*n)`

#### Normalization convention

All `Z_i` variables are rescaled relative to standard hyperbolic geometry by a factor of `iπ`:

```
Z_i  (this project)  =  Z_i (SnaPy / log-shape)  /  iπ
```

Consequences:
- Tetrahedral constraint: `Z_i + Z_i' + Z_i'' = 1`  (standard hyperbolic: = `iπ`)
- Edge equation RHS: `= 2`                            (standard: = `2πi`)
- Cusp equation for complete structure: TBD

#### Gluing equation matrix layout

For a manifold with **n tetrahedra** and **r cusps**, the matrix is **(n + 2r) × 3n**:

| Rows | Count | Meaning |
|---|---|---|
| `0 … n-1` | n | **Edge equations** (one per internal edge) |
| `n + 2k` | 1 per cusp | **Meridian equation** for cusp k (k = 0…r-1) |
| `n + 2k+1` | 1 per cusp | **Longitude equation** for cusp k (k = 0…r-1) |

> ⚠️ **Cusp rows are interleaved:** the ordering is μ₀, λ₀, μ₁, λ₁, …, μᵣ₋₁, λᵣ₋₁  
> (NOT all meridians followed by all longitudes)

#### Column layout (3n columns)

The columns are ordered as:

```
Z_1, Z_1', Z_1'',   Z_2, Z_2', Z_2'',   …   Z_n, Z_n', Z_n''
```

where `Z_i`, `Z_i'`, `Z_i''` are the **logarithmic shape parameters** (rescaled) of tetrahedron `i`.

Each equation takes the form:

```
∑_{i=1}^{n}  ( f_i · Z_i  +  g_i · Z_i'  +  h_i · Z_i'' )  =  2
```

#### Variable reduction: eliminating Z_i'

**Constraint:** For each tetrahedron `i`:  `Z_i + Z_i' + Z_i'' = 1`

**Substitution:** `Z_i' = 1 - Z_i - Z_i''`

Reduced variable space (**2n variables**):
```
v = [ Z_1, Z_1'',  Z_2, Z_2'',  …,  Z_n, Z_n'' ]
```

A row `(f_1,g_1,h_1, …, f_n,g_n,h_n)` maps to:
```
const  =  ∑_i g_i
coeff of Z_i   =  f_i - g_i
coeff of Z_i'' =  h_i - g_i
```
So the equation reads: `const + coeff · v = 2`.

#### General internal edges

Any valid internal edge is of the form:

```
∑_{i=1}^n  a_i · C_i  +  b_i · T_i
```

where:
- `C_i` = i-th SnaPy edge equation (evaluates to 2)
- `T_i = Z_i + Z_i' + Z_i''` (evaluates to 1)
- Normalization: `∑_i (2a_i + b_i) = 2`
- All `(Z_i, Z_i', Z_i'')` coefficients in the result are **non-negative integers**

`T_i` commutes with everything (its symplectic pairing with any variable is 0).  
`C_i` also commutes with everything (verified computationally).

#### Rank of edge equations

The n SnaPy edge equations have **rank n − r** in the reduced 2n-dimensional variable space.

**Verified:**

| Manifold | n | r | rank | n−r |
|---|---|---|---|---|
| `m004` | 2 | 1 | 1 | 1 ✓ |
| `s776` | 6 | 3 | 3 | 3 ✓ |
| `t12047` | 8 | 4 | 4 | 4 ✓ |
| `m125` | 4 | 2 | 2 | 2 ✓ |
| `s000` | 6 | 1 | 5 | 5 ✓ |

A basis of **n − r linearly independent edge equations** is selected by column-pivoted QR on the
reduced edge coefficient matrix (always a subset of the raw SnaPy rows).

#### Commutation relations (symplectic structure)

The reduced variables carry the symplectic form **Ω**:

```
[Z_i, Z_i''] = 1   (for each i)
[Z_i, Z_j  ] = [Z_i, Z_j''] = [Z_i'', Z_j''] = 0   (i ≠ j)
```

The cyclic relations follow automatically:
```
[Z_i', Z_i]   = [1 - Z_i - Z_i'', Z_i]   = 1  ✓
[Z_i'', Z_i'] = [Z_i'', 1 - Z_i - Z_i''] = 1  ✓
```

For any two linear combinations `A = a·v` and `B = b·v`:
```
[A, B] = a^T Ω b  =  ∑_i ( a_{2i} · b_{2i+1} − a_{2i+1} · b_{2i} )
```

**Key commutation values (verified computationally across all test manifolds):**
- `[M_k, L_k] = 2` for each cusp k
- `[E_i, M_k] = [E_i, L_k] = 0` for every independent edge E_i and every cusp k
- `[E_i, E_j] = 0` for any two independent edges

#### Verified example — `m004` (figure-8 knot, n=2, r=1)

```
           Z1   Z1'  Z1''   Z2   Z2'  Z2''
edge 0:     2    1    0      1    0    2
edge 1:     0    1    2      1    2    0
merid 0:    1    0    0      0   -1    0    ← row n+0 = row 2
long  0:    0    0    0      0   -2    2    ← row n+1 = row 3
```

After reduction (basis = edge row 0 only, since rank = 1):
```
edge 0 reduced:  const=1,  [ Z1: 1,  Z1'': -1,  Z2: 1,  Z2'': 2 ]   → 1 + Z1 - Z1'' + Z2 + 2Z2'' = 2

---

### Step 3 — Neumann-Zagier Matrix & Affine Shift

- **Module:** `src/manifold_index/core/neumann_zagier.py`
- **Public API:**
  - `build_neumann_zagier(data, easy_result, reduced=None) → NeumannZagierData`
- **Status:** ✅ Complete (34 tests passing for m004, s776, v2408)

#### Output — `NeumannZagierData`

| Field | Shape | Type | Description |
|---|---|---|---|
| `g_NZ` | `(2n, 2n)` | `int` | Symplectic matrix (see row/column layout below) |
| `nu_x` | `(n,)` | `int` | Affine shift for position rows |
| `nu_p` | `(n,)` | `int` | Affine shift for momentum rows |
| `n` | scalar | `int` | Number of tetrahedra |
| `r` | scalar | `int` | Number of cusps |
| `num_hard` | scalar | `int` | Number of hard internal edges |
| `num_easy` | scalar | `int` | Number of independent easy internal edges |

Methods: `is_symplectic()`, `symplectic_form` (property), `g_NZ_inv()`.

#### Variable convention (NZ reduction)

The same substitution `Z_i' = 1 − Z_i − Z_i''` is used as in Step 2.
A row `(f_i, g_i, h_i)` at tet `i` reduces to:

```
const   =  ∑_i g_i
coeff Z_i   =  f_i − g_i
coeff Z_i'' =  h_i − g_i
```

The constant and linear coefficients are identical to those from `_reduce_row`.
There is no separate "NZ convention" — the whole project uses `Z_i + Z_i' + Z_i'' = 1`.

#### Column ordering in `g_NZ`

Columns correspond to the **block-ordered** variable vector:

```
v = (Z_1, Z_2, …, Z_n,  Z_1'', Z_2'', …, Z_n'')
```

(not the interleaved ordering used internally by `gluing_equations.py`).

#### Row ordering in `g_NZ`

```
Rows 0 … r-1              position:   meridian equations   (one per cusp)
Rows r … r+num_hard-1     position:   hard internal edges
Rows r+num_hard … n-1     position:   easy internal edges
Rows n … n+r-1            momentum:   longitude / 2        (one per cusp)
Rows n+r … 2n-1           momentum:   Γ vectors            (one per internal edge)
```

The **position block** P = `g_NZ[:n, :]` and the **momentum block** Q = `g_NZ[n:, :]`.

#### Symplectic form

In block ordering:

```
Ω_block = [[  0_n,  I_n ],
           [ −I_n,  0_n ]]
```

Pairing: `[u, v] = u_Z · v_{Z''} − u_{Z''} · v_Z = u @ Ω_block @ v`.

`g_NZ` satisfies `g_NZ @ Ω_block @ g_NZ^T = Ω_block`.

#### Affine shift

The reduced equation reads `c + coeff · v = RHS` where `c = ∑_i g_i`
(from `_reduce_row`).  The affine shift satisfies `g_NZ_row · v + ν = RHS`,
so `ν = c − RHS`.

The RHS differs by row type:
- **Meridian rows** (cusp equations): RHS = 0 (complete-structure holonomy is parabolic)
- **Internal edge rows** (edge equations): RHS = 2 (one full edge loop = 2πi → 2)

```
ν_x[k]  = c                 (for meridian position rows, k < r)
ν_x[k]  = c − 2             (for internal edge position rows, k ≥ r)
```

For momentum rows:
- **Longitude/2 rows:** `ν_p[k] = c_long / 2` where `c_long` is the longitude constant from `_reduce_row`. (Longitude RHS = 0, so ν_p = c_long / 2.)
- **Γ rows:** `ν_p[k] = 0` by construction.

#### Algorithm — integer right-inverse + isotropic correction

The momentum block Q is found by solving `P @ Ω @ Q^T = I_n` over **Z** and then making Q isotropic.

**Step 1 — Integer right-inverse** (`_int_right_inverse`)

Apply Euclidean column reduction to `A = P @ Ω` (n × 2n), tracking the unimodular
transformation `V`. After reduction, `A @ V = [H | 0]` where H is n × n lower-triangular
with unit diagonal. The right-inverse is:

```
Q_T = V[:, :n] @ H_inv
```

where `H_inv` is computed by exact forward-substitution (also lower-triangular, integer).

This works because `SNF(P @ Ω)` has all invariant factors = 1 (verified for all test manifolds).

**Step 2 — Isotropic correction** (`_make_isotropic`)

Given `Q_T` (satisfies `A @ Q_T = I_n` but Q may not be isotropic), compute:

```
S[i,j] = [Q_T[:,i], Q_T[:,j]]  (anti-symmetric integer matrix)
C = strictly lower-triangular part of S
Q_T ← Q_T + P^T @ C
```

The general integer null space of `A = P @ Ω` is `{P^T c : c ∈ Z^n}`.  Adding `P^T @ C` to
`Q_T` preserves `A @ Q_T = I_n` and zeroes out S, since `[Q_i', Q_j'] = S_{ij} − C[i,j] + C[j,i] = 0`.

#### Verified invariants (all test manifolds)

- `g_NZ @ Ω @ g_NZ^T = Ω` (symplectic) ✓
- `[μ_k, Λ_k] = 1` where `Λ_k = g_NZ[n+k]` (half-longitude row) ✓
- `[C_i, Γ_i] = 1` for each internal edge i, and `[C_i, Γ_j] = 0` for i ≠ j ✓
- All commutators between position and momentum rows follow the canonical pattern ✓
- `nu_p[k] = 0` for all Γ rows ✓

---

### Step 4 — 3D Index Calculation

- **Symbol:** `I(m⃗, e⃗)` — a formal power series in q^{1/2}
- **Inputs:** `NeumannZagierData` (Step 3)
- **Parameters:**
  - `m_ext` — integer vector of length `r` (cusp meridians only)
  - `e_ext` — integer vector of length `r` (cusp longitudes÷2 only)
  - `q_order_half` — truncation order in q^{1/2} (default 20)
- **Reference:** Garoufalidis–Kim, *"The 3D index of an ideal triangulation and angle
  structures"*, formula (2.41) and Lemma 3.6

#### Notation mapping (paper → project)

| Paper symbol | Our symbol | Meaning |
|---|---|---|
| `r` | `n` | number of tetrahedra |
| `n` | `r` | number of cusps |
| `κ` | `kappa` | combined (m, e) vector, size 2n |
| `ν_x` | `nu_x` | affine shift of position rows (top n of g_NZ) |
| `ν_p` | `nu_p` | affine shift of momentum rows (bottom n of g_NZ) |

> **Note:** The `num_hard`/`num_easy` distinction (from the NZ basis construction)
> does **not** control the summation.  All `n − r` internal edges are summed over;
> `num_hard`/`num_easy` are retained in `NeumannZagierData` only as bookkeeping metadata.

#### Formula (2.41)

```
I(m_ext, e_ext) =
    Σ_{e_int ∈ (½)Z^{n-r}}
        (−q^{½})^{ m · ν_p  −  e · ν_x }
        · ∏_{a=0}^{n-1} I_Δ( (g_NZ⁻¹ κ)_a ,  (g_NZ⁻¹ κ)_{n+a} )

where:
    κ = (m_ext, 0^{n-r}, e_ext, e_int)     [size 2n]
    m = (m_ext, 0^{n-r})                    [internal m's forced to 0]
    e = (e_ext, e_int)                      [internal e's summed over]
```

Terms with non-integer `g_NZ⁻¹ κ` arguments are zero (I_Δ(m, e) = 0 for non-integer m, e).

#### Example — 4_1 figure-eight knot (n=2, r=1)

```
κ(t) = [1, 0, 1, t]    (m_ext=[1], e_ext=[1], e_int=[t])

g_NZ⁻¹ κ(t) = [t, t, t+1, t-1]
             → tet 0: (m=t, e=t+1),  tet 1: (m=t, e=t-1)

I(1,1) = Σ_t  I_Δ(t, t+1) · I_Δ(t, t-1)
```

For `q_order_half=20`, 5 terms contribute (t ∈ {−2, −1, 0, 1, 2}).

#### Degree formula δ(m, e) — Lemma 3.6

```
δ(m, e) = ½ (m₊(m+e)₊ + (−m)₊ e₊ + (−e)₊(−e−m)₊)  +  max{0, m, −e}
```

where `x₊ = max{0, x}`. This is the leading power of q^{1/2} in I_Δ(m, e).

#### Summation strategy

1. **Integrality filter.** Python enumerates all `δ ∈ {0,1}^{n-r}` (half-integer
   offset patterns) and checks whether `g_NZ_inv[:, n+r : 2n] @ δ ≡ 0 (mod 2)`.
   Only valid δ yield non-zero terms (I_Δ vanishes for non-integer arguments).
2. **Exact degree bound.** For each valid δ, the effective degree

   ```
   F(e_int) = Σ_a δ(base_args_a + int_cols_a · e_int)  +  phase_base  −  ν_x_int · e_int
   ```

   is piecewise-quadratic and **convex** in e_int ∈ ℤ^{n-r}, growing to +∞ in
   all directions.  Python determines the exact set `{e_int : F(e_int) ≤ q_order_half}`
   by scanning along all {−1,0,1}^{n-r} directions (cardinal + diagonal), then
   enumerating the resulting tight bounding box.  No fixed search-radius heuristic
   is needed.
3. **Mathematica** performs all q-series arithmetic, using a precomputed
   tetrahedron-index cache (`.mx` format, stored in `data/tet_index/`).

#### Files

| File | Purpose |
|---|---|
| `core/index_3d.py` | Python: degree formula, κ assembly, summation enumeration, Mathematica bridge |
| `mathematica/TetIndex.wl` | Mathematica: I_Δ(m,e) formula (placeholder — fill in from paper), cache management |
| `mathematica/Index3D.wl` | Mathematica: q-series summation, phase × product assembly |
| `mathematica/ComputeIndex3D.wl` | Mathematica script entry-point (subprocess target) |
| `data/tet_index/` | Cache directory for `.mx` tetrahedron index tables |

#### Output

`Index3DResult` dataclass:
- `coeffs: list[int]` — q^{1/2}-series coefficients
- `min_power: int` — lowest power present
- `q_order_half: int` — cutoff used
- `n_terms: int` — number of contributing summation terms
- `m_ext: list[int]` — cusp m values used (length r)
- `e_ext: list[int]` — cusp e values used (length r)

#### Status: ✅ Implemented and verified (4_1 I(1,1) produces 5 terms; 159 tests passing)

---

### Step 5 — Dehn Filling & Non-Closable Cycles

- **Operation:** Dehn filling applied to each cusp
- **Slope range:**
  - Default range: TBD
  - User may override
- **Goal:** Find *non-closable cycles* for each cusp

#### Definition — Non-Closable Cycle

For a cusp `i` with meridian `M` and longitude `L`, the cycle `P·M + Q·L`
(slope `P/Q`, with `gcd(P,Q) = 1`) is **non-closable** if the Dehn-filled index
vanishes:

```
I_{P/Q}^{(i)} = 0   (identically as a q^{1/2}-series)
```

where `I_{P/Q}^{(i)}` is defined by the Dehn filling kernel sum:

```
I_{P/Q}^{(i)} = Σ_{m,e}  K(P,Q; m,e) · I(m_all, e_all)
```

with kernel:

```
K(P,Q; m,e) = ½ (−1)^{Rm+2Se} ·
  [ δ_{Pm+2Qe, 0} · (q^{(Rm+2Se)/2} + q^{−(Rm+2Se)/2})
    − δ_{Pm+2Qe, 2}
    − δ_{Pm+2Qe, −2} ]
```

where `R, S ∈ ℤ` satisfy `R·Q − P·S = 1`, `m ∈ ℤ`, `e ∈ (½)ℤ`, and
`m_all`, `e_all` are the full cusp variable vectors with cusp `i` set to
`(m, e)` and all other cusps set to their test values (default: 0).

#### Verified example — `4_1` (figure-eight knot complement)

| Slope P/Q | `I_{P/Q}(4_1)` | Non-closable? |
|---|---|---|
| **1/0** (meridian) | **0** | **Yes** ✓ |
| 0/1 (longitude) | `1` | No |
| ±1/1, ±2/1, ±3/1 | `1` | No |
| ±4/1 (exceptional) | `17 − 2q² − 2q³ + …` | No |
| ±5/1 | `1 − q − 2q² − q³ − q⁴ + q⁵ + …` | No |

The ±5/1 results are equal (reflecting the amphichirality of `4_1`). ✓

#### Files

| File | Purpose |
|---|---|
| `core/dehn_filling.py` | Kernel enumeration, q-series arithmetic, `compute_filled_index`, `find_non_closable_cycles` |
| `tests/test_dehn_filling.py` | 80 unit tests (all passing) |

#### Status: ✅ Implemented and verified

- **Output:** List of non-closable cycles per cusp

---

### Step 6 — Basis Selection

- **Module:** `src/manifold_index/core/basis_selection.py`
- **Status:** ✅ Complete (56 tests passing)
- **Interaction:** App displays list of non-closable cycles per cusp
- **User action:** Select one cycle per cusp as basis
- **Fallback:** If no non-closable cycle exists for a cusp → use *default curve*
- **Definition of default curve:** SnaPy meridian M (slope 1/0) or longitude L (slope 0/1); both offered as labelled options.

#### Slope convention

A cycle at cusp i is specified by slope (P, Q), meaning the homology class P·μ + Q·λ.

#### Public API

| Symbol | Description |
|---|---|
| `CycleChoice` | One cusp's chosen cycle: fields `cusp_idx, P, Q, label, is_default`; property `slope_str` |
| `BasisSelection` | Full per-cusp selection: field `choices`; properties `m_ext`, `e_ext`, `r` |
| `default_meridian_choice(cusp_idx)` | Slope (1, 0), cycle μ |
| `default_longitude_choice(cusp_idx)` | Slope (0, 1), cycle λ |
| `make_basis_selection(nz_data, cycle_results, choices, *, default="M", strict=False)` | Main entry point; builds `BasisSelection` from per-cusp `(P, Q)` choices or `None` (→ default) |

---

### Step 7 — Easy Edges & Phase Space Basis

- **Source:** Gluing equations (from Step 2); specifically `data.edge_equations` (shape `(n, 3n)`)
- **Goal:** Find *easy edges* to form a phase space basis of size `n − r`
- **Output:** `EasyEdgeResult` — maximal independent easy edges + hard padding

#### Definition — Easy Edge

A **valid internal edge** is a non-negative integer 3n-vector `E` expressible as

    E = ∑ᵢ aᵢ Cᵢ + ∑ⱼ bⱼ Tⱼ

where `Cᵢ = data.edge_equations[i]` (SnaPy edge equations), `Tⱼ[3j:3j+3] = (1,1,1)` and 0
elsewhere, `aᵢ, bⱼ ∈ ℤ`, and the **normalization constraint** holds:

    2 ∑ aᵢ + ∑ bⱼ = 2

An internal edge `E` is **easy** if for every tetrahedron `j`, at most one of
`(E[3j], E[3j+1], E[3j+2])` is non-zero.

We call `E[3j]`, `E[3j+1]`, `E[3j+2]` the *Z*, *Z′*, *Z″* slots of tet `j`, respectively.

#### Easy Pattern

Every easy edge induces a **pattern** `p ∈ {Z, Z′, Z″, OFF}ⁿ` where

- `p[j] = Z`  if `E[3j] ≠ 0` (only the Z slot is active)
- `p[j] = Z′` if `E[3j+1] ≠ 0`
- `p[j] = Z″` if `E[3j+2] ≠ 0`
- `p[j] = OFF` if all three slots are zero

#### Algorithm B — Pattern-First Search

**Stage 1 — Pattern enumeration (4ⁿ patterns)**

For each pattern `p ∈ {Z, Z′, Z″, OFF}ⁿ`:

1. **Build constraint matrix** `M` (shape `(num_rows, n)`) and rhs vector:

   - For each tet `j` with active slot `d ∈ {Z, Z′, Z″}` and inactive slots `s1, s2`:
     - Both inactive slots must equal zero: `∑ᵢ aᵢ col_s1 + bⱼ = 0` and `∑ᵢ aᵢ col_s2 + bⱼ = 0`
     - Taking the difference eliminates `bⱼ`: `∑ᵢ aᵢ (col_s1 − col_s2) = 0`  *(1 row, rhs=0)*
     - `bⱼ` is then recovered as `bⱼ = −(a @ col_s1)`

   - For each tet `j` with `p[j] = OFF` (all slots zero):
     - `∑ᵢ aᵢ (col_Z − col_Z′) = 0` and `∑ᵢ aᵢ (col_Z′ − col_Z″) = 0`  *(2 rows, rhs=0)*
     - `bⱼ = −(a @ col_Z)`

   - **Normalization row** (rhs = 2): `a @ (2·1 − ∑ⱼ ref_col_j) = 2`
     where `ref_col_j` is `col_s1` for active patterns and `col_Z` for OFF.
     This encodes `2∑aᵢ + ∑bⱼ = 2` and prevents `lstsq` returning the trivial `a = 0`.

   Here `col_d` at tet `j` = `edge_equations[:, 3j+d]` (a length-n vector of coefficients).

2. **Solve** `M @ a = rhs` via `scipy.linalg.lstsq` (minimum-norm least-squares).
   - If the system is inconsistent (`‖M a − rhs‖ > tol`), skip.

3. **Round** `a_sol` to nearest integer `a_int = round(a_sol)`.
   - Validate: `‖M a_int − rhs‖ ≤ tol`  (minimum-norm solution can differ from the integer
     solution, so we re-check the rounded result rather than checking `‖a_sol − a_int‖`).

4. **Recover** `bⱼ = −(a_int @ ref_col_j)` for each tet and round to integer `b_int`.
   - Validate: all `bⱼ` are integer and `2∑aᵢ + ∑bⱼ = 2`.

5. **Reconstruct** `E = a_int @ edge_equations + b_int @ T_matrix`.
   - Check `E ≥ 0` (non-negativity) and `_is_easy(E)`.
   - Deduplicate via a `set` of tuple-keys.

**Stage 2 — Maximal independent subset**

Use QR with column pivoting on the `(k × 2n)` matrix of *reduced* easy-edge vectors
(reduced = eliminate Z″ via `Z + Z′ + Z″ = 1` to get a 2n representation).
Select the leading `easy_rank = min(rank, n−r)` pivot columns.

**Stage 3 — Hard padding**

If `num_independent_easy < n − r`, iterate over `reduced.independent_edge_indices`
(the basis from Step 2) and greedily add SnaPy rows that increase the rank, until the
basis has exactly `n − r` edges.

**Final basis order:** independent easy edges first, then hard-padding edges.

#### Implementation

```
src/manifold_index/core/phase_space.py
```

Key symbols:
- `EasyEdgeResult` — dataclass holding `all_easy`, `independent_easy_indices`,
  `hard_padding`, `n`, `r`; properties `num_independent_easy`, `basis_edges`
- `find_easy_edges(data, tol=1e-8) → EasyEdgeResult`
- `_is_easy(edge_3n, n) → bool`
- `_build_constraint_matrix(pattern, edge_rows, n) → (M, rhs)`
- `_compute_b(a, pattern, edge_rows, n) → b`
- Slot constants: `_Z=0, _ZP=1, _ZPP=2, _OFF=3`

Tests: `tests/test_phase_space.py` (40 tests, all passing).

---

### Step 8 — Refined Index

- **Module:** `src/manifold_index/core/refined_index.py` *(to be created)*
- **Status:** ❌ Not yet implemented

#### Overview

The refined index is a generalisation of the 3D index (Step 4) that introduces one formal
fugacity variable per **hard edge**, promoting those charges from summation variables to
explicit parameters of the output.

#### Hard vs Easy Edges (recap)

From Step 7, the `n − r` internal edges are partitioned into:

- `num_easy` **easy edges** — found first by Algorithm B; these occupy rows
  `r + num_hard … n − 1` of `g_NZ`.
- `num_hard = (n − r) − num_easy` **hard edges** — the remaining edges padded in by
  Algorithm B Stage 3; these occupy rows `r … r + num_hard − 1` of `g_NZ`.

Algorithm B maximises `num_easy`. There is no "typical" number of hard edges — the split
depends entirely on the geometry of the manifold. When `num_hard = 0` the refined index
equals the ordinary 3D index exactly.

#### Fugacity Assignment

Each hard edge `C_a` (a = 0, …, num_hard − 1) carries one formal fugacity `η_a`.
When summing over the internal charge `e_{r+a}` corresponding to `C_a`, every term in the
sum is weighted by `η_a^{e_{r+a}}`.

The easy edge charges `e_{r+num_hard}, …, e_{n−1}` carry no fugacity; they are summed
over exactly as in the ordinary 3D index.

#### Formula

```
I^ref(q; η_0, …, η_{num_hard−1}) =
    Σ_{e_int ∈ (½)Z^{n-r}}
        [ ∏_{a=0}^{num_hard−1}  η_a^{e_{r+a}} ]
        · (−q^{½})^{ m · ν_p  −  e · ν_x }
        · ∏_{j=0}^{n-1} I_Δ( (g_NZ⁻¹ κ)_j ,  (g_NZ⁻¹ κ)_{n+j} )
```

where `κ`, `m`, `e` are assembled identically to Step 4 and `e_int` is split as:

```
e_int = ( e_{r}, …, e_{r+num_hard−1},  e_{r+num_hard}, …, e_{n−1} )
          ╰─── hard charges ───────────╯  ╰─── easy charges ───────╯
```

Setting all `η_a = 1` recovers `compute_index_3d` exactly.

#### Output Representation

```python
dict[tuple[int, ...], Fraction]
# key  : (q_half_power, η_0_power, …, η_{k−1}_power)   length = 1 + num_hard
# value: integer coefficient
```

The output is a Laurent polynomial in `(q^{1/2}, η_0, …, η_{k−1})` truncated at the
requested `q_order_half`.

#### Future — Dehn Filling of the Refined Index

Dehn filling of the refined index is **not implemented** and should not be attempted until
the following three conditions are verified for the chosen boundary cycle:

1. **Non-closability** — the cycle is non-closable under the ordinary 3D index (Step 5).
2. **Weyl symmetry** — the refined index satisfies the Weyl symmetry condition with
   respect to the fugacity associated to that cusp.
3. **Adjoint `su(2)` character** — the coefficient of `q^1` in the refined index, treated
   as a function of the relevant fugacity `η`, equals `η + 1 + η^{−1}` (the adjoint
   `su(2)` character).

Only when all three conditions hold is the Dehn filling well-defined. The kernel used is
substantially more complex than the ordinary kernel and is given by formula A.7 of the
reference paper (*Refined 3D Index*, appendix A). Implementation is deferred.

---

## 4. Data Structures & Types

> To be filled in as implementation progresses.

| Name | Type | Description |
|---|---|---|
| `ManifoldData` | dataclass | Raw data extracted from SnaPy |
| `ReducedGluingData` | dataclass | Step 2 output — reduced gluing equations & symplectic basis |
| `EasyEdgeResult` | dataclass | Step 7 output — easy/hard edge classification & phase space basis |
| `NeumannZagierData` | dataclass | Step 3 output — `g_NZ`, `nu_x`, `nu_p` |
| `GluingEquations` | TBD | Representation of gluing equations |
| `NonClosableCycles` | TBD | Per-cusp list of cycles |
| `RefinedIndex` | TBD | Final result |

---

## 5. Input / Output Spec

### Inputs
| Parameter | Type | Default | Description |
|---|---|---|---|
| Manifold name | `str` | — | SnaPy-recognizable name |
| `m⃗` range | range/list | TBD | Range for 3D index |
| `e⃗` range | range/list | TBD | Range for 3D index |
| Slope range | range/list | TBD | For Dehn filling |

### Outputs
| Output | Format | Description |
|---|---|---|
| Non-closable cycles | TBD | Per-cusp lists |
| Refined index | TBD | Final result |

---

## 6. Dependencies

| Package | Version | Purpose |
|---|---|---|
| `snappy` | latest | 3-manifold topology |
| `numpy` | latest | Numerical computation |
| TBD | | |

---

## 7. Packaging & Distribution

- **Target:** macOS `.app` bundle
- **Tooling:** TBD (`py2app` / `Briefcase` / `PyInstaller`)
- **Interface:** GUI (see Section 8)

---

## 8. GUI Workflow

> ⚠️ *This section captures the agreed high-level workflow. Detailed screen layouts,
> widget choices, and framework selection (tkinter / PyQt / web-based) are TBD.*

The application exposes the full pipeline as a **step-by-step GUI** with automatic
computation stages and one interactive pause (Step 6 — basis selection).

### Screen 1 — Manifold Input

| Element | Description |
|---|---|
| Text field | Manifold name (e.g. `m004`, `4_1`) |
| Validate button | Calls SnaPy to confirm the name is recognized |
| Advanced options | `q_order_half` (default 20), slope range for Dehn filling (default TBD) |
| Run button | Launches the full computation pipeline |

### Automatic Stages (Steps 2 – 5)

After the user hits **Run**, the following stages execute without further input, with a
progress indicator:

1. **SnaPy extraction** (Step 2) — fast
2. **NZ matrix** (Step 3) — fast
3. **Easy/hard edge classification** (Step 7) — fast
4. **3D index** (Step 4) — may take seconds to minutes depending on `q_order_half`
5. **Dehn filling search** (Step 5) — iterates over all slopes in the configured range;
   progress shown as `slope k / N`

At the end of Stage 5, the app displays the **list of non-closable cycles** found per
cusp and pauses for user input.

### Screen 2 — Basis Selection (Step 6, interactive)

For each cusp, the app shows:

- The list of non-closable cycles found (displayed as slopes `P/Q`)
- A selection control (radio buttons / dropdown) to pick **one cycle per cusp**
- If **no non-closable cycle** was found for a cusp → the app uses the *default curve*:
  the **SnaPy-provided meridian M and longitude L** for that cusp.  Both are offered
  as labelled options in the selection control.

The user confirms their selection and presses **Compute Refined Index**.

### Screen 3 — Refined Index Output (Step 8)

Displays the refined index as a formatted Laurent series in `q` and the fugacity
variables `η_0, …, η_{k−1}` (one per hard edge). Shows:

- The number of hard edges `k` and their labels
- The truncated series up to `q_order_half`
- Export options: **plain text, LaTeX, JSON, Mathematica `.mx`** (maximum options)

### GUI Design Decisions

| # | Decision |
|---|---|
| G1 | **Framework: PyQt6 (or PySide6).** Best macOS native look; packages cleanly as a `.app` via PyInstaller; rich built-in widgets. PySide6 preferred for LGPL licensing. |
| G2 | ✅ **All intermediate results are exportable** (3D index, non-closable cycle list, refined index) — maximum export options. |
| G3 | Batch mode (multiple manifolds, no interaction) is a **future feature**, not required for initial release. |
| G4 | ✅ **Slope range is per-cusp** — each cusp has its own configurable min/max slope. |
| G5 | ✅ **Default curve = SnaPy meridian M and longitude L** for that cusp (both offered as options). |

---

## 9. Open Questions

1. ~~What is the precise formula for `I(m⃗, e⃗)`?~~ ✅ Resolved — formula (2.41) implemented.
2. ~~What are the dimensions of `m⃗` and `e⃗`?~~ ✅ Resolved — both have length `r` (cusps only).
3. ~~What is the definition of a *non-closable cycle*?~~ ✅ Resolved — a cycle P/Q is
   non-closable if the Dehn-filled 3D index `I_{P/Q}` vanishes identically (Step 5).
4. ~~What is the *default curve* fallback when no non-closable cycle exists for a cusp?~~ ✅ Resolved —
   the SnaPy-provided meridian M and longitude L for that cusp (both offered as options in the GUI).
5. ~~What is the formula for the *refined index*?~~ ✅ Resolved — see Step 8.
6. ~~What output/export formats are needed?~~ ✅ Resolved — maximum options: plain text, LaTeX, JSON,
   Mathematica `.mx` (see Section 8).
7. ~~CLI only, or also a graphical interface?~~ ✅ Resolved — GUI using PyQt6/PySide6 (see Section 8).
8. ~~Project name?~~ ✅ Resolved — **Refined Index Calculator**.

---

## 10. Changelog

| Date | Change |
|---|---|
| 2026-03-06 | Initial scaffold created. Pipeline skeleton from user description. |
| 2026-03-06 | Step 2 implemented (`gluing_equations.py`): SnaPy extraction, interleaved cusp ordering, variable reduction, symplectic structure. 40 tests passing. |
| 2026-03-06 | Step 7 implemented (`phase_space.py`): easy edge definition, Algorithm B (pattern-first), `EasyEdgeResult`. 40 tests passing (80 total). |
| 2026-03-06 | Step 3 implemented (`neumann_zagier.py`): integer right-inverse + isotropic correction, `NeumannZagierData`. 34 tests passing (114 total). |
| 2026-03-06 | Pipeline renumbered: NZ matrix = Step 3 (new); 3D index placeholder inserted as Step 4; old Steps 4–7 → Steps 5–8. |
| 2026-03-07 | Step 4 implemented (`index_3d.py`, `TetIndex.wl`, `Index3D.wl`, `ComputeIndex3D.wl`): degree formula, κ assembly, integrality filter, Mathematica bridge, cache system. Tetrahedron index formula left as placeholder in `TetIndex.wl` (fill from paper). |
| 2026-03-07 | Step 4 corrected: ALL n−r internal edges are summed (not just "easy" ones). `m_ext`/`e_ext` now length `r` (cusp-only). κ = (m_cusp, 0^{n-r}, e_cusp, e_int). `build_kappa`, `valid_half_integer_patterns`, `enumerate_summation_terms` rewritten. Bug fixed: `nu_x_int = nu_x[r:n]` (was empty slice). Verified: 4_1 I(1,1) = Σ_t I_Δ(t,t+1)·I_Δ(t,t-1), 5 terms. 159 tests passing. |
| 2026-03-08 | Step 6 implemented (`basis_selection.py`): `CycleChoice`, `BasisSelection`, `make_basis_selection`. Cycle (P,Q) → m_ext[i]=P, e_ext[i]=Q/2. Default curves M (1/0) and L (0/1). 56 tests passing (391 total). GUI implemented (`app/gui.py`, `app/worker.py`): three-screen PySide6 application with QStackedWidget; `PipelineWorker` (Steps 1–5 in QThread with per-slope progress signal); `RefinedIndexWorker` (Step 8 in QThread); Screen 1 (manifold input + q_order_half + slope ranges); Screen 2 (progress bar + per-cusp radio-button cycle selection); Screen 3 (formatted series + export: plain text, LaTeX, JSON, Mathematica .mx). |
