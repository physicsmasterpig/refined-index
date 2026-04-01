# Manifold Index: Comprehensive Technical Guide

**Version:** 0.4.0
**Project:** `manifold-index` — 3D index, refined index, Dehn filling, and non-closable cycles for hyperbolic 3-manifolds

---

## Table of Contents

1. [Introduction and Mathematical Context](#1-introduction-and-mathematical-context)
2. [Notation Reference](#2-notation-reference)
3. [Installation and Entry Points](#3-installation-and-entry-points)
4. [End-to-End Pipeline](#4-end-to-end-pipeline)
   - [Step 1: Load Manifold](#step-1-load-manifold--manifestpy)
   - [Step 2: Reduce Gluing Equations](#step-2-reduce-gluing-equations--gluing_equationspy)
   - [Step 3: Phase Space Basis (Easy Edges)](#step-3-phase-space-basis--phase_spacepy)
   - [Step 4: Neumann-Zagier Matrix](#step-4-neumann-zagier-matrix--neumann_zagierpy)
   - [Step 5: Basis Selection](#step-5-basis-selection--basis_selectionpy)
   - [Step 6: 3D Index](#step-6-3d-index--index_3dpy)
   - [Step 7: Refined Index](#step-7-refined-index--refined_indexpy)
   - [Step 8: Weyl Checks](#step-8-weyl-checks--weyl_checkpy)
   - [Step 9: Dehn Filling and Non-Closable Cycles](#step-9-dehn-filling--dehn_fillingpy)
   - [Step 10: Refined Dehn Filling](#step-10-refined-dehn-filling--refined_dehn_fillingpy)
   - [Step 11: Kernel Caching](#step-11-kernel-caching--kernel_cachepy)
5. [GUI Architecture](#5-gui-architecture)
6. [Export Infrastructure](#6-export-infrastructure)
7. [Data Structures Reference](#7-data-structures-reference)
8. [Caching Architecture](#8-caching-architecture)
9. [Performance Guide](#9-performance-guide)
10. [Testing](#10-testing)
11. [Build and Packaging](#11-build-and-packaging)

---

## 1. Introduction and Mathematical Context

### What the 3D index computes

For a cusped hyperbolic 3-manifold M, the **3D index** is a topological invariant introduced by Garoufalidis–Kim–Zagier. It takes the form of a formal power series in q^{1/2} whose coefficients count certain triangulation-level configurations.

Given a cusped triangulation (SnaPy database), the program:

1. Extracts a symplectic system of gluing equations from the triangulation
2. Builds the **Neumann-Zagier matrix** g_NZ — a 2n×2n symplectic change-of-basis relating shape variables to canonical cusp/edge charges
3. Computes the **3D index** I(m, e) by summing products of tetrahedron indices I_Δ over internal edge charges
4. Extends to the **refined index** I^ref(q; η) by tracking hard-edge fugacities η_a separately
5. Performs **Dehn filling** via the kernel K(P,Q; m,e) to obtain the filled index at slope P/Q
6. Identifies **non-closable cycles** — slopes where the filled index vanishes identically

### Why each piece matters

- **Tetrahedron index** I_Δ(m,e): the quantum dilogarithm building block; each tetrahedron in the triangulation contributes one factor
- **Easy vs hard edges**: easy edges have a "diagonal" basis change (only one of Z, Z', Z'' active per tet); hard edges do not. Hard edges require fugacity variables η_a
- **Neumann-Zagier matrix**: the change from shape variables to the canonical (position/momentum) basis that makes the symplectic structure explicit
- **Non-closable cycles**: a slope P/Q at cusp i for which I_{P/Q}^{(i)} ≡ 0 is geometrically significant — it is a candidate for the "non-closable" curve relevant to the 3D-index machinery
- **Refined index / Dehn filling chain**: the main original result of the paper, extending the ordinary filling kernel to track η-fugacities through the Hirzebruch-Jung continued fraction chain

---

## 2. Notation Reference

This codebase systematically swaps the paper's r and n symbols:

| Paper | Code | Meaning |
|-------|------|---------|
| r | n (`num_tetrahedra`) | Number of tetrahedra |
| n | r (`num_cusps`) | Number of cusps |
| q^{1/2} | `qq` | The formal half-integer power variable |
| q^{1/2}-series | `{qq_power: coeff}` dict | Sparse polynomial in qq |
| η_a | `eta` (hard edges) | Fugacity for hard edge a |
| Λ_i = L_i/2 | `e_ext[i]` | Half-longitude (cusp momentum) |
| κ | `kappa` | Combined (m, e) size-2n vector |
| ν_x | `nu_x` | Affine shift of position rows |
| ν_p | `nu_p` | Affine shift of momentum rows |
| Z_i' | eliminated | Z_i' = 1 - Z_i - Z_i'' throughout |

**Column ordering conventions:**

- `gluing_matrix` columns: Z_1, Z_1', Z_1'', Z_2, Z_2', Z_2'', ... (interleaved)
- `g_NZ` columns: Z_1, ..., Z_n, Z_1'', ..., Z_n'' (block ordering)
- Conversion: `_interleaved_to_block(coeff, n)` in `neumann_zagier.py`

**Doubled exponents:** All η fugacity exponents are stored as `2 × exponent` in dict keys to keep everything a plain `int`. A key like `(qq_power, 4, -2)` means q^{qq_power/2} · η_0^2 · η_1^{-1}.

---

## 3. Installation and Entry Points

### Install

```bash
# Core (math pipeline only):
pip install -e .

# With GUI:
pip install -e ".[gui]"

# With dev tools (pytest, ruff, mypy):
pip install -e ".[dev]"
```

Requires Python ≥ 3.10. Compiles a C extension (`_c_tet_index.so`) at install time from `src/manifold_index/core/_c_kernel/tet_index.c`.

### Entry points

| Method | How |
|--------|-----|
| GUI app | `manifold-index` (CLI script) or `python -c "from manifold_index.app import main; main()"` |
| GUI (code) | `from manifold_index.app import launch_gui; launch_gui()` |
| Frozen macOS app | PyInstaller bundle via `build_app.sh` / `ManifoldIndex.spec` |
| Programmatic | Import directly from `manifold_index.core.*` |

### Minimal programmatic example

```python
from manifold_index.core.manifold import load_manifold
from manifold_index.core.phase_space import find_easy_edges
from manifold_index.core.neumann_zagier import build_neumann_zagier
from manifold_index.core.refined_index import compute_refined_index

data = load_manifold("m004")
easy = find_easy_edges(data)
nz = build_neumann_zagier(data, easy)
result = compute_refined_index(nz, m_ext=[1], e_ext=[0], q_order_half=20)
```

---

## 4. End-to-End Pipeline

The pipeline is a directed acyclic graph of pure-function transformations:

```
load_manifold
    ↓ ManifoldData
reduce_gluing_equations    find_easy_edges
    ↓                           ↓
ReducedGluingData         EasyEdgeResult
         \                /
          build_neumann_zagier
                ↓
          NeumannZagierData
         /        |        \
compute_    compute_     find_non_
index_3d    refined_     closable_
            index        cycles
                ↓
        compute_filled_
        refined_index
        (kernel chain)
```

---

### Step 1: Load Manifold — `manifold.py`

**Function:** `load_manifold(name: str) → ManifoldData`

Calls `snappy.Manifold(name)` and extracts the gluing equations as an integer matrix.

**ManifoldData fields:**

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | SnaPy name, e.g. `"m004"` |
| `num_tetrahedra` | `int` | n = number of tetrahedra |
| `num_cusps` | `int` | r = number of cusps |
| `gluing_matrix` | `ndarray (n+2r, 3n) int` | Integer coefficient matrix |
| `raw` | `snappy.Manifold` | Raw SnaPy object for further queries |

**Gluing matrix layout:**

```
Rows 0   … n-1     :  edge equations   (n rows)
Row  n + 2k        :  meridian μ_k     (k = 0…r-1)
Row  n + 2k+1      :  longitude λ_k
```

Cusp rows are **interleaved**: μ₀, λ₀, μ₁, λ₁, ...

Each row encodes: Σ_i (a_i·Z_i + b_i·Z_i' + c_i·Z_i'') = 2πi · RHS

**Convenience properties:** `edge_equations`, `meridian_equations`, `longitude_equations`, `cusp_equations(k)`

---

### Step 2: Reduce Gluing Equations — `gluing_equations.py`

**Function:** `reduce_gluing_equations(data: ManifoldData) → ReducedGluingData`

**Why this step:** The shape variables Z_i, Z_i', Z_i'' are not independent — they satisfy Z_i + Z_i' + Z_i'' = 1 (up to a 2πi constant). This step eliminates Z_i' via substitution Z_i' = 1 - Z_i - Z_i'', reducing 3n variables to 2n.

**Reduction formula** (applied per row, per tet i with coefficients f_i, g_i, h_i):

```
const += g_i
coeff[2i]   = f_i - g_i    # coefficient of Z_i
coeff[2i+1] = h_i - g_i    # coefficient of Z_i''
```

Implemented in `_reduce_row(row, n) → (const, coeff_2n)`.

**ReducedGluingData fields:**

| Field | Type | Description |
|-------|------|-------------|
| `n`, `r` | `int` | Dimensions |
| `edge_coeffs` | `ndarray (n, 2n) int` | Reduced edge equations |
| `edge_consts` | `ndarray (n,) int` | Constant terms; `edge_consts[i] + edge_coeffs[i]·v = 2` |
| `cusp_coeffs` | `ndarray (2r, 2n) int` | Meridian/longitude equations (interleaved) |
| `cusp_consts` | `ndarray (2r,) int` | Constant terms of cusp equations |
| `independent_edge_indices` | `list[int]` | n-r independent edge rows (QR pivoting) |
| `symplectic_matrix` | `ndarray (2n, 2n) int` | Ω[2i, 2i+1]=+1, Ω[2i+1, 2i]=-1 |

**Key fact:** The n edge equations have rank n-r in the reduced basis. `_independent_row_indices` selects n-r of them via column-pivoted QR on the transposed matrix.

**Symplectic pairing:** For vectors a, b in reduced space, [A,B] = a^T Ω b. This encodes [Z_i, Z_i''] = 1 for same i, and all cross-tet pairings are zero.

---

### Step 3: Phase Space Basis — `phase_space.py`

**Function:** `find_easy_edges(data: ManifoldData) → EasyEdgeResult`

**Why this step:** The Neumann-Zagier matrix requires a specific basis of n-r internal edges. Ideally these are **easy edges** — edges where at most one of (Z_i, Z_i', Z_i'') is nonzero per tetrahedron. Easy edges produce a simple (almost diagonal) NZ matrix. If not enough easy edges exist, hard edges are padded in.

**EasyEdgeResult fields:**

| Field | Type | Description |
|-------|------|-------------|
| `all_easy` | `list[ndarray (3n,)]` | All distinct easy edges found |
| `independent_easy_indices` | `list[int]` | Indices into `all_easy` forming max independent subset |
| `hard_padding` | `list[ndarray (3n,)]` | Hard SnaPy edges appended to reach n-r total |
| `n`, `r` | `int` | Dimensions |
| `basis_edges` (property) | `list[ndarray]` | Independent easy edges + hard padding |

**Algorithm B (Pattern-first):**

An easy edge has a **pattern** — for each tet j, exactly one of {Z, Z', Z'', OFF} is active (constants _Z=0, _ZP=1, _ZPP=2, _OFF=3).

**Stage 0:** Check SnaPy edge rows directly for easiness (fast path; non-negative entries, at most one nonzero per tet triplet, sum = 2).

**Stage 1:** Enumerate all 4^n patterns. For each pattern, build and solve the constraint system:

- For each tet j with active slot and two inactive slots s1, s2:
  ```
  Σ_i a_i · (col_s1 - col_s2) = 0
  ```
- For OFF tet: two difference rows = 0
- Normalization: `a · (2·ones - Σ ref_cols) = 2` (prevents trivial a=0)

Solve with:
1. **Fast path**: `scipy.linalg.lstsq` + round. Verify residual < tol.
2. **Slow path** (underdetermined): exact Fraction-based RREF + free-variable enumeration with |coefficients| ≤ `MAX_COEFF=3`.

Reconstruct 3n-vector: `E = Σ a_i · edge_row_i + Σ b_j · T_j` where `T_j` = (1,1,1) at tet j.

**Stage 2:** QR column-pivoting on all `all_easy_reduced` vectors to find maximal independent subset.

**Stage 3:** Pad with SnaPy independent edge rows (SVD rank check) until n-r independent edges total.

**Why MAX_COEFF=3:** Easy edges in practice have small integer coefficients a_i. This bounds the search space while being provably sufficient for all known manifolds.

---

### Step 4: Neumann-Zagier Matrix — `neumann_zagier.py`

**Function:** `build_neumann_zagier(data: ManifoldData, easy: EasyEdgeResult) → NeumannZagierData`

**Why this step:** The NZ matrix is the change of basis from shape variables to canonical (position/momentum) variables. It is symplectic: g_NZ Ω g_NZ^T = Ω. This structure is used pervasively — to compute tet-index arguments, to invert efficiently, and to verify integrality.

**NeumannZagierData fields:**

| Field | Type | Description |
|-------|------|-------------|
| `g_NZ` | `ndarray (2n, 2n) float` | The symplectic matrix; block-ordered columns |
| `nu_x` | `ndarray (n,) int` | Affine shift for position rows |
| `nu_p` | `ndarray (n,) float` | Affine shift for momentum rows |
| `n`, `r` | `int` | Dimensions |
| `num_hard` | `int` | Hard edges in basis |
| `num_easy` | `int` | Independent easy edges in basis |

**Row structure of g_NZ (2n rows total):**

```
Position block (top n rows):
  Rows 0   … r-1              : meridians (one per cusp)
  Rows r   … r+num_hard-1     : hard internal edges
  Rows r+num_hard … n-1       : easy internal edges

Momentum block (bottom n rows):
  Rows n   … n+r-1            : longitudes / 2
  Rows n+r … 2n-1             : Γ vectors (symplectic conjugates of internal edges)
```

**Column ordering:** Block ordering — columns 0..n-1 are position (Z_1,...,Z_n), columns n..2n-1 are momentum (Z_1'',...,Z_n''). This is different from the interleaved ordering of `gluing_matrix`.

**Affine shifts (ν):** Each row encodes `g_NZ_row · v + ν = RHS`. Since the reduced equation is `c + coeff · v = RHS`, we get `ν = c - RHS`:
- Meridians (RHS=0): `ν_x = c`
- Internal edges (RHS=2): `ν_x = c - 2`
- Longitudes/2: `ν_p = c_long / 2`
- Γ rows (constructed, not from data): `ν_p = 0`

**Γ row construction:** Given the position block P (n×2n) and r longitude/2 rows, the remaining n-r Γ rows are found by solving:

```
[P ; Q_long] · Ω_block · Γ^T = RHS
```

where RHS[r:n, :] = I_{n-r} (the n-r × n-r identity). The system is solved via `_int_right_inverse` — exact integer column reduction (Euclidean algorithm tracking a transformation matrix V) with `Fraction` arithmetic for Smith-factor denominators.

**Key methods on NeumannZagierData:**

- `is_symplectic(tol)`: checks g_NZ Ω g_NZ^T = Ω
- `g_NZ_inv() → Fraction[2n, 2n]`: exact inverse via the symplectic identity `g^{-1} = [[D^T, -B^T], [-C^T, A^T]]` where g = [[A,B],[C,D]] in n×n blocks. Returns a Fraction-dtype array. Cached.
- `g_NZ_inv_scaled() → (S, S·g_inv_int64)`: LCD S of all entries of g^{-1}; returns integer-scaled inverse. S=2 for most manifolds (from longitude/2 rows). Cached.
- `inv_denom` (property): just S from above

**Helper `_interleaved_to_block(coeff, n)`:** permutation (Z_1, Z_1'', Z_2, Z_2'',...) → (Z_1,...,Z_n, Z_1'',...,Z_n'').

---

### Step 5: Basis Selection — `basis_selection.py`

**Why this step:** Before computing the refined index, the user must choose which cycle (P·M + Q·L) to evaluate at each cusp. This sets the external (cusp) variables m_ext and e_ext. A non-closable cycle must first be identified (Step 9) before it can be chosen here.

**CycleChoice fields:**

| Field | Type | Description |
|-------|------|-------------|
| `cusp_idx` | `int` | 0-based cusp index |
| `P`, `Q` | `int` | Primitive integers for P·M + Q·L |
| `label` | `str` | Human-readable label |
| `is_default` | `bool` | True if fallback meridian/longitude |
| `m` (property) | `int` | P (meridian variable) |
| `e` (property) | `Fraction` | Q/2 (half-longitude variable) |

**BasisSelection:** Ordered list of CycleChoice (one per cusp). Properties: `m_ext`, `e_ext`, `r`.

**`make_basis_selection(nz_data, cycle_results, choices, *, default="M", strict=False)`:**
- `choices[i]` = `(P,Q)` or `None` (use default meridian/longitude)
- `strict=True`: raises if chosen slope was not found non-closable
- Validates primitivity (gcd = 1) for each choice

**`apply_basis_changes(nz_data, basis) → NeumannZagierData`:**
- For each cusp k with odd P_k: calls `apply_cusp_basis_change(nz_data, k, P_k, Q_k)` to perform an SL(2,Z) transformation on the cusp rows of g_NZ
- For even P_k (including longitude P=0): skips the change; caller must pass `m=P`, `e=Q/2` directly

**Default curves:**
- `default_meridian_choice(k)`: (P,Q)=(1,0), m=1, e=0
- `default_longitude_choice(k)`: (P,Q)=(0,1), m=0, e=1/2

---

### Step 6: 3D Index — `index_3d.py`

**Function:** `compute_index_3d_python(nz_data, m_ext, e_ext, q_order_half=20) → Index3DResult`

**The formula** (SPEC.md eq. 2.41):

```
I(m_ext, e_ext) = Σ_{e_int ∈ (½)Z^{n-r}}
    (-q^{1/2})^{ m_full · nu_p  −  e_full · nu_x }
    · ∏_{a=0}^{n-1} I_Δ( (g_NZ⁻¹ κ)_a ,  (g_NZ⁻¹ κ)_{n+a} )
```

where:
- `m_full = (m_ext[0],...,m_ext[r-1], 0,...,0)` — length n; internal edge m forced to 0
- `e_full = (e_ext[0],...,e_ext[r-1], e_int[0],...,e_int[n-r-1])` — length n
- `κ = (m_full, e_full)` — size 2n combined vector

**Integrality constraint:** A term is nonzero only if `g_NZ⁻¹ κ` has all-integer entries. This is checked via `g_NZ_inv_scaled()`: compute `S·g^{-1}·κ` in int64, check divisibility by S.

**Enumeration:** `enumerate_summation_terms(nz_data, m_ext, e_ext, q_order_half)` iterates over half-integer patterns of e_int ∈ {k/2 : k ∈ Z}^{n-r}, filtering by integrality and degree bound:

```python
for e_int in half_integer_grid:
    S, g_inv_S = nz_data.g_NZ_inv_scaled()
    kappa = build_kappa(m_full, e_full)  # size 2n
    local_charges = g_inv_S @ kappa      # int64; must be divisible by S
    if not all divisible by S: continue
    tet_args = local_charges / S         # tet (m_j, e_j) pairs
    ...
```

**Phase computation:** `phase_exp = m_full · nu_p − e_full · nu_x` — always integer by the symplectic structure of g_NZ.

**Term result:** For each valid e_int, multiply tet-index polynomials:

```python
prod = {0: 1}
for (ta, tb) in tet_args:
    s = _tet_index_series(ta, tb, budget)
    prod = convolve(prod, s)
total += (-1)^phase_exp · prod shifted by phase_exp
```

**Tetrahedron index** `_tet_index_series(m, e, qq_order) → dict[int, int]`:

The Garoufalidis–Kim formula (applied via `MIt` symmetry reduction):

```
MIt(m, e):
  if m+e ≥ 0:  (-qq)^m · I_t(-m-e, m)
  else:         I_t(m, e)

I_t(m, e) = Σ_{n=max(0,-e)}^∞  (-1)^n · qq^{n(n+1)-(2n+e)·m}
             ──────────────────────────────────────────────
             ∏_{k=1}^{n}(1-qq^{2k}) · ∏_{k=1}^{n+e}(1-qq^{2k})
```

Returns `{qq_power: coeff}` up to `qq_order`. Non-integer m or e → returns `{}`.

Cached in module-level `_tet_cache[(m, e, qq_order)]`. C-accelerated when `_c_tet_index` extension is available.

**Dynamic budget shrinking:** Each successive tet-index call in the product gets a tighter `cutoff = budget - prod_min_pow`, where `prod_min_pow` tracks the running minimum power of the accumulating product. This avoids computing high-power terms that will always get truncated.

**C extension:** `_c_tet_index_series(m, e, qq_order)` and `_c_poly_convolve(prod, s, budget)` from `_c_kernel/tet_index.c`. Used automatically when compiled; falls back to Python silently.

**Index3DResult fields:**

| Field | Type | Description |
|-------|------|-------------|
| `coeffs` | `list[int]` | Coefficients from `min_power` to `q_order_half` |
| `min_power` | `int` | Lowest nonzero qq-power |
| `q_order_half` | `int` | Cutoff used |
| `m_ext`, `e_ext` | `list` | Input variables |
| `n_terms` | `int` | Number of nonzero summation terms |

---

### Step 7: Refined Index — `refined_index.py`

**Function:** `compute_refined_index(nz_data, m_ext, e_ext, q_order_half=20) → RefinedIndexResult`

**What changes vs. Step 6:** The first `k = num_hard` entries of e_int correspond to hard edges. These are tracked as fugacity exponents η_a instead of being summed into the q-series.

**RefinedIndexResult type:** `dict[tuple[int,...], int]`

Key structure: `(q_half_power, 2·η_0_exp, 2·η_1_exp, ..., 2·η_{k-1}_exp)`

Why doubled: all fugacity exponents are half-integers; ×2 gives plain ints.

**Formula:**

```
I^ref(q; η_0,...,η_{k-1}) =
    Σ_{e_int}  [ ∏_{a=0}^{k-1} η_a^{e_{r+a}} ]
    · (-q^{1/2})^{...} · ∏_j I_Δ(...)
```

The hard-edge fugacity exponents are extracted from e_int:

```python
eta_exps_x2 = tuple(int(Fraction(e_int_strs[a]) * 2) for a in range(k))
key = (shifted_power,) + eta_exps_x2
result[key] += sign * coeff
```

When `num_hard = 0` (all easy edges), keys are length-1 tuples `(qq_power,)` — identical to the ordinary 3D index.

**Batch computation:** `compute_refined_index_batch(nz_data, entries, q_order_half)` pre-computes `_get_enum_state(nz_data)` once and reuses it for all (m_ext, e_ext) pairs. Typically 10-100× faster for grid evaluations.

**Utility functions:**
- `project_to_3d_index(refined) → dict[int, int]`: sum over all η monomials per q-power (equivalent to η_a = 1 for all a)
- `format_refined_index(refined, num_hard, q_var, eta_vars) → str`: human-readable Laurent series string

---

### Step 8: Weyl Checks — `weyl_check.py`

**Functions:** `compute_ab_vectors(nz_data, refined_grid, q_order) → ABVectors`, `run_weyl_checks(...) → WeylCheckResult`

**Why this step:** Before performing refined Dehn filling, the refined index must satisfy two prerequisite symmetry conditions.

**ABVectors fields:**

| Field | Type | Description |
|-------|------|-------------|
| `a` | `list[Fraction]` | Coupling to longitude e; must be integer for valid basis |
| `b` | `list[Fraction]` | Coupling to meridian m; must be half-integer |
| `num_hard` | `int` | k = number of hard edges |
| `warnings` | `list[str]` | Non-fatal issues found |
| `a_is_integer` (property) | `bool` | All a[j] ∈ Z |
| `b_is_half_integer` (property) | `bool` | All 2·b[j] ∈ Z |
| `is_valid` (property) | `bool` | Both conditions satisfied |

**Condition 2 — Weyl symmetry:** The refined index can be made η ↔ η^{-1} symmetric after multiplying by monomial η^{b·m + a·e}. The vectors (a, b) are extracted from the centre-of-mass of the η polynomial at leading q-order:

```
centre(m, e) = Σ_k (η_exp_k · coeff_k) / Σ_k coeff_k
b[j] = [centre(+m, 0) - centre(-m, 0)] / (2m)   (half-integer)
a[j] = [centre(0, +e) - centre(0, -e)] / e        (integer)
```

**Condition 3 — Adjoint su(2) character:** The η⁰ coefficient of q^1 in the generating function must integrate to −1 under the SU(2) Haar measure. For single-cusp case, this reduces to:

```
(1/2)(c_{-1} + c_{+1} - c_{-2} - c_{+2}) = -1
```

where c_j is the coefficient of η^j in the η-series at q^1.

---

### Step 9: Dehn Filling — `dehn_filling.py`

**Functions:** `find_non_closable_cycles(nz_data, q_order_half, max_slope) → ...`, `compute_filled_index(nz_data, P, Q, m_other, e_other, q_order_half) → Index3DResult`

**Setup:** For cusp i with NZ basis (M_i, Λ_i = L_i/2), Dehn filling along slope P/Q means surgery along P·M_i + 2Q·Λ_i = P·pos + 2Q·mom = c.

**Kernel formula:**

```
K(P, Q; m, e) = ½ · (-1)^{Rm+2Se} ·
  [ δ_{Pm+2Qe, 0} · (q^{(Rm+2Se)/2} + q^{-(Rm+2Se)/2})
    - δ_{Pm+2Qe, -2}
    - δ_{Pm+2Qe, +2} ]
```

where R, S ∈ Z satisfy R·Q − P·S = 1 (computed via `find_rs(P, Q)` using `_ext_gcd`).

**Filled index:**

```
I_{P/Q}^{(i)}(m_other, e_other) = Σ_{m_i, e_i} K(P, Q; m_i, e_i) · I(m_all, e_all)
```

**KernelTerm dataclass (frozen):**

| Field | Description |
|-------|-------------|
| `m`, `e` | Point (m_i, e_i) in the kernel sum |
| `c` | Value of P·m + 2Q·e ∈ {-2, 0, 2} |
| `phase` | R·m + 2S·e (always integer) |
| `multiplicity` | 2 for c=2 (antipodal) and c=0 with t≠0; 1 for c=0,t=0 |

**Kernel enumeration** `enumerate_kernel_terms(P, Q, R, S, nz_data, ...)`:

For each c ∈ {0, 2} (c = -2 covered by antipodal symmetry of c = +2):

1. Particular solution: `_particular_solution(P, Q, c)` solves P·m₀ + 2Q·e₀ = c via `_ext_gcd(P, Q)`
2. General family: `m_t = m_c + Q·t`, `e_t = e_c - P·t/2`, `phase_t = phase_c0 + t`
3. Degree filter: `adjusted_q = q_order_half + |phase_t|` for c=0; `= q_order_half` for c=2
4. Include term if `min_degree(I_{3D}(m_t, e_t)) ≤ adjusted_q`
5. Stop after 2 consecutive empty steps (convex degree growth guarantees termination)

**Antipodal symmetry:** `I_{3D}(-m, -e) = I_{3D}(m, e)` and the kernel factor at phase=-t equals that at phase=+t. So c=2 enumeration with multiplicity=2 covers all c=±2 terms; c=0 with t>0 uses multiplicity=2.

**Non-closable cycle:** A slope (P, Q) is non-closable at cusp i if `I_{P/Q}^{(i)} ≡ 0` for all (m_other, e_other). In practice, tested at m_other = e_other = 0 for each other cusp.

**Finding R, S:** `find_rs(P, Q)` uses `_ext_gcd(|Q|, |P|)` then corrects signs to ensure R·Q - P·S = 1.

---

### Step 10: Refined Dehn Filling — `refined_dehn_filling.py`

**Function:** `compute_filled_refined_index(nz_data, P, Q, ...)` — applies the full Hirzebruch-Jung kernel chain

**Overview:** The ordinary Dehn filling kernel extends to a refined kernel K^ref(P,Q; m,e; η) that tracks hard-edge charges through surgery. This requires the Hirzebruch-Jung continued fraction (HJ-CF) expansion of P/Q and a chain of I_S symplectic kernels.

**HJ-CF expansion:**

```
P/Q = k_1 − 1/(k_2 − 1/(... − 1/k_ℓ))
```

Special cases:
- Q=0, P=±1: ℓ=2, k=[0,0]
- |Q|=1: ℓ=1, k=[P/Q] — ordinary (unrefined) kernel K(k₁,1;·,·) suffices

**Kernel chain** (ℓ ≥ 2):

```
K^ref(P,Q; m,e; η) =
    Σ_{m_1,e_1} ... Σ_{m_{ℓ-1},e_{ℓ-1}}
        I_S(m, -e - k_1/2·m, m_1, e_1)
      · I_S(m_1, -e_1 - k_2/2·m_1, m_2, e_2)
      · ...
      · K(k_ℓ, 1; m_{ℓ-1}, e_{ℓ-1})
```

**I_S kernel** ("symplectic kernel", `_is_kernel`):

```
I_S(m1, e1, m2, e2; η) =
    (½)·(-1)^{m1}·(q^{m1/2} + q^{-m1/2}) · ẽI_S(m1, e1,   m2, e2)
  - (½)·(-1)^{m1}                         · ẽI_S(m1, e1-1, m2, e2)
  - (½)·(-1)^{m1}                         · ẽI_S(m1, e1+1, m2, e2)
```

**ẽI_S inner function** (`_etilde_is`, `@functools.lru_cache`):

```
ẽI_S(m1, e1, m2, e2; η) =
    Σ_{e ∈ Z, t ∈ Z}  η^e
    · I_Δ(-e1 - m2/2,  -e/2 + e1 + m1/2 + t)
    · I_Δ( e1 + m2/2,  -e/2 + e2 - m2/2 + t)
    · I_Δ(-e2 - m1/2,   e2 + m1/2 + t)
    · I_Δ( e2 + m1/2,   e1 - m2/2 + t)
    · (-q^{1/2})^{-e + e1 + e2 + m1/2 - m2/2 + 2t}
```

The sum over e is filtered to integer parity = (m1+m2) mod 2. Non-integer I_Δ arguments → series = {} → term is 0.

**Output type:** `QEtaSeries = dict[(qq_power, eta_exp), Fraction]`

**int-mode arithmetic:** The I_S chain uses integer arithmetic throughout, converting to Fraction only at the end. The LCD accumulates as 2^ℓ (one factor of 2 per kernel level). This avoids expensive Fraction operations in the hot loop.

**Diamond truncation:** After each I_S convolution step, the output is clipped to remove entries near the boundary of the truncation domain. These boundary terms are unreliable due to finite-order truncation of the tet-index series and would corrupt subsequent convolution steps.

**Caching:**

- `_etilde_is`: `@functools.lru_cache` (pure function of integer arguments)
- `_is_kernel`: `@functools.lru_cache`
- `_iref_cache`: manual dict, keyed by `_nz_content_key(nz_data)` + `(m_ext, e_ext, q)`. Content-based key (NOT `id(nz_data)`) to prevent stale hits when GC recycles memory addresses.
- `clear_filling_caches()`: clears all three. Returns evicted-count dict.

**Multi-cusp extension:** For r ≥ 2 cusps, `compute_multi_cusp_filled_refined_index` applies kernel chains sequentially, one cusp at a time, building up a `MultiEtaSeries` with one η-variable per cusp plus hard-edge fugacities.

**Current limitation:** r > 2 cusps raises `NotImplementedError`.

---

### Step 11: Kernel Caching — `kernel_cache.py`

**Why:** The I_S kernel chain is manifold-independent (it depends only on P,Q and the charge variables m,e). Pre-computing and storing it eliminates the ~10-minute HJ-CF computation on repeat evaluations of the same slope.

**Storage layout:**

```
Bundled (read-only, shipped):
  src/manifold_index/data/kernel_cache/kernel_P{P}_Q{Q}_qq{qq}.pkl.gz

User cache (writable, runtime-generated):
  ~/Library/Caches/manifold-index/kernel_cache/    (macOS)
  ~/.cache/manifold-index/kernel_cache/             (Linux)
  %LOCALAPPDATA%/manifold-index/kernel_cache/       (Windows)
```

Lookup order: user cache → bundled. New kernels always saved to user cache.

**KernelTable dataclass:** Stores `P`, `Q`, `qq_order`, and the sparse table `(m, e) → QEtaSeries`.

**`precompute_filling_kernel(P, Q, qq_order, ...)`:** Computes K^ref(P,Q; m,e; η) for all (m,e) pairs that could contribute and saves to cache.

**`apply_precomputed_kernel(kernel_table, nz_data, m_ext_other, e_ext_other, q_order_half)`:**
Applies a loaded kernel to the manifold's I^ref:

```
Î^ref_{P/Q}(W, V) = Σ_{m,e}  I^ref(m,e; η^{2W}) · K^ref(P,Q; m,e; η^{2V})
```

Uses dense `int64` numpy arrays for fast inner products. `ProcessPoolExecutor` for parallel I^ref batch computation when the grid is large.

**Degree-feasibility filtering (vectorised):**

`_degree_feasible_row(m0, k1, e_half_arr, mt, et, qq_limit_x2)` — fully numpy-vectorised over all (e_half, target) pairs simultaneously. Uses `int32` arithmetic (all values fit). Returns boolean array of feasible (e_half, target) pairs.

`_tet_degree_x2(m, e) → int`: returns `2 × min_degree(I_Δ(m,e))` as a pure integer (no floats). Scalar version.

`_tdeg_arr(m, e) → ndarray`: same but vectorised over numpy arrays.

These avoid computing tet-index series for (m,e) pairs where the degree is provably above the cutoff.

---

## 5. GUI Architecture

The GUI is a PySide6 application (`src/manifold_index/app/`). All long computations run on background QThreads; the main thread never blocks.

### MainWindow (`window.py`)

- `QMainWindow` with a `QTabWidget`
- **Tab 1 "Calculator":** `QSplitter` with 3 panels (Manifold | Filling | Export)
- **Tab 2 "Kernel Builder":** `KernelPanel` for precomputing and caching kernels

**Key state:**
- `_nz_data`: `NeumannZagierData` — shared across all panels after computation
- `_refined_worker`: `RefinedIndexWorker` thread
- `_dehn_worker`: `DehnFillingWorker` thread

**Key methods:**
- `_start_compute(name, q_order_half)`: loads manifold, builds NZ data, launches `RefinedIndexWorker`
- `_on_refined_finished(results)`: stores results, updates Panel 1 display, enables Panels 2/3
- `_start_dehn_filling(payload)`: launches `DehnFillingWorker` with cusp configs
- `closeEvent()`: stops workers before Qt cleanup

### Workers (`workers.py`)

**`RefinedIndexWorker(QThread)`:**
- Evaluates I^ref on a 45^r grid: m ∈ {-2,-1,0,1,2}, e ∈ {-2,-3/2,...,2} per cusp
- Signals: `status(str)`, `progress(int, int)`, `finished(list)`, `error(str)`
- Output: `list[(m_ext, e_ext, RefinedIndexResult)]`
- After grid completion: runs `run_weyl_checks()`, emits result

**`DehnFillingWorker(QThread)`:**
- **Step 1:** NC cycle search — `compute_filled_index()` for each candidate slope in [min_P,max_P]×[min_Q,max_Q], deduplicates via `_canonicalize_nc_cycles()`
- **Step 2:** Compute filled refined index
  - **Path A (multi-cusp):** `_run_multi_cusp()` — applies `apply_general_cusp_basis_change()` then `compute_multi_cusp_filled_refined_index()`
  - **Path B (single-cusp):** `_run_single_cusp()` — transforms slope to (γ,δ) basis, calls `compute_filled_refined_index()` for each NC cycle

**`KernelBuilderWorker(QThread)`:** calls `precompute_filling_kernel()` with user-specified slopes.

### Panels

**`ManifoldPanel` (`panels/manifold_panel.py`):**
- Input: manifold name + q_order_half spinner
- Output: KaTeX-rendered NZ matrix, refined index grid display, Weyl check results
- `computation_finished(entries, weyl_result)` emits `data_ready` to unlock Panels 2/3

**`FillingPanel` (`panels/filling_panel.py`):**
- Per-cusp input rows (P/Q spinners) rebuilt dynamically per manifold
- NC range controls; "Dehn Fill ▶" button
- Displays NC cycles found (Step 1), then filled index results (Step 2)

**`ExportPanel` (`panels/export_panel.py`):**
- Format checkboxes: LaTeX, Full Report (.tex), Mathematica (.m), Plain text (.txt), JSON
- Output directory selector, filename prefix
- "Export All", "Copy LaTeX", "Copy Plain Text" buttons
- Calls `manifold_index.utils.exporters.*` on button clicks

### KaTeX rendering (`formatters.py`, `katex.py`)

`formatters.py` generates HTML fragments with KaTeX markup for:
- `series_to_katex(result, num_hard, max_q_terms)` — refined index as rendered LaTeX
- `format_nz_matrix(nz_data)` — HTML table of the NZ matrix
- `format_panel1_html(...)` / `format_panel2_html(...)` — full panel HTML

`katex.py` bridges the Qt WebEngine to KaTeX's JavaScript rendering.

### Signal flow summary

```
User types manifold name → Panel 1 emits compute_requested(name, q)
→ MainWindow._start_compute() → RefinedIndexWorker.start()
→ Worker emits finished(results)
→ MainWindow._on_refined_finished() → stores _nz_data
→ Panel 1.computation_finished() → emits data_ready
→ Panels 2 and 3 unlock

User sets slopes → Panel 2 emits fill_requested(payload)
→ MainWindow._start_dehn_filling() → DehnFillingWorker.start()
→ Worker emits nc_found(cycles), then filling_finished(results)
→ Panel 2 displays; Panel 3 ready to export
```

---

## 6. Export Infrastructure

**Module:** `src/manifold_index/utils/exporters.py`

All exporters take the Panel 1 data dict (NZ data, refined index entries, Weyl results) and optional Panel 2 data (Dehn filling results).

### Format writers

| Function | Output | Description |
|----------|--------|-------------|
| `write_latex(path, data)` | `.tex` | LaTeX series, standalone fragment |
| `write_full_report(path, data, dehn_data)` | `.tex` | Full LaTeX document with header, NZ matrix, series, Weyl checks |
| `write_mathematica(path, data)` | `.m` | Mathematica list of rules `{key} -> coeff` |
| `write_plain_text(path, data)` | `.txt` | Human-readable ASCII series |
| `write_json(path, data, dehn_data)` | `.json` | Structured JSON with all results |
| `clipboard_latex(data)` | clipboard | LaTeX series only |
| `clipboard_plain_text(data)` | clipboard | Plain text series |

### Monomial formatting helpers

**LaTeX:**
- `_latex_q_factor(qq_pow)` → `"q"`, `"q^3"`, `"q^{7/2}"`, etc.
- `_latex_eta_factors_hard(key, num_hard)` → `r"\eta_a^{2W}"` factors
- `_latex_eta_factors_cusp(key, num_hard, num_cusp_eta)` → cusp η factors
- `_latex_monomial(key, coeff, num_hard, num_cusp_eta)` → full term with ± sign

**Mathematica:**
- `_math_q_factor()`, `_math_eta_hard()`, `_math_eta_cusp()`, `_math_monomial()` — equivalent for Mathematica list syntax

**Key design:** All monomial formatters take the raw tuple key from `RefinedIndexResult` or `MultiEtaSeries`. The formatting functions know how to interpret the doubled-exponent convention.

---

## 7. Data Structures Reference

All major data containers are frozen dataclasses (immutable) to ensure safe use as cache keys and across thread boundaries.

| Class | Module | Key Fields | Notes |
|-------|--------|-----------|-------|
| `ManifoldData` | manifold.py | `name`, `n`, `r`, `gluing_matrix (n+2r, 3n)` | Holds raw SnaPy object |
| `ReducedGluingData` | gluing_equations.py | `edge_coeffs (n,2n)`, `cusp_coeffs (2r,2n)`, `independent_edge_indices`, `symplectic_matrix` | After Z' substitution |
| `EasyEdgeResult` | phase_space.py | `all_easy`, `independent_easy_indices`, `hard_padding`, `basis_edges` (property) | Phase space basis |
| `NeumannZagierData` | neumann_zagier.py | `g_NZ (2n,2n)`, `nu_x (n,)`, `nu_p (n,)`, `num_hard`, `num_easy` | Core symplectic data; caches g^{-1} |
| `CycleChoice` | basis_selection.py | `cusp_idx`, `P`, `Q`, `m` (=P), `e` (=Q/2) | One cusp's chosen cycle |
| `BasisSelection` | basis_selection.py | `choices`, `m_ext`, `e_ext` | Per-cusp selection |
| `KernelTerm` | dehn_filling.py | `m`, `e`, `c`, `phase`, `multiplicity` | One term in Dehn filling sum |
| `Index3DResult` | index_3d.py | `coeffs`, `min_power`, `q_order_half`, `m_ext`, `e_ext`, `n_terms` | Dense q-series |
| `ABVectors` | weyl_check.py | `a`, `b`, `num_hard`, `is_valid` | Weyl symmetry vectors |
| `KernelTable` | kernel_cache.py | `P`, `Q`, `qq_order`, `table (m,e → QEtaSeries)` | Cached kernel |

**Type aliases:**

```python
QSeries          = dict[int, Fraction]               # q^{k/2} → coeff
RefinedIndexResult = dict[tuple[int,...], int]        # (qq, 2η_0,...) → coeff
QEtaSeries       = dict[tuple[int, int], Fraction]   # (qq_pow, η_exp) → coeff
MultiEtaSeries   = dict[tuple[int,...], Fraction]    # multi-cusp filling key
```

---

## 8. Caching Architecture

Six independent caches with different scopes and eviction strategies:

| Cache | Location | Scope | Key | Eviction |
|-------|----------|-------|-----|----------|
| `_tet_cache` | `index_3d.py` module | Process lifetime | `(m, e, qq_order)` | `clear_tet_cache()` |
| `_tet_arr_cache` | `refined_dehn_filling.py` module | Process lifetime | `(m, e, qq_order)` | `_clear_tet_arr_cache()` |
| `_enum_state_cache` | `index_3d.py` module | Process lifetime | NZ content bytes | `clear_enum_state_cache()` |
| `_iref_cache` | `refined_dehn_filling.py` module | Process lifetime | NZ content + (m,e,qq) | `clear_filling_caches()` |
| `_etilde_is` | `refined_dehn_filling.py` | Process (LRU) | `(m1,e1,m2,e2,qq,eta)` | `clear_filling_caches()` |
| `_is_kernel` | `refined_dehn_filling.py` | Process (LRU) | `(m1,e1,m2,e2,qq)` | `clear_filling_caches()` |
| `KernelTable` disk | `kernel_cache.py` | Cross-session | `P_Q_qq` filename | Manual deletion |
| `_kernel_mem_cache` | `kernel_cache.py` module | Process lifetime | `(P,Q,qq,dir)` | `clear_kernel_cache()` |

**Critical: content-based keys.** NeumannZagierData is not hashable. The caches in `refined_dehn_filling.py` use `_nz_content_key(nz_data)`:

```python
def _nz_content_key(nz_data):
    return (
        nz_data.g_NZ.data.tobytes(),
        nz_data.nu_x.data.tobytes(),
        nz_data.nu_p.data.tobytes(),
    )
```

**Why not `id()`:** Python may GC an NZ object and reuse the same memory address for a new one. Using `id()` would cause a cache hit returning the wrong result. This bug manifested when multiple basis-changed NZ objects were created in a loop (one per NC cycle).

---

## 9. Performance Guide

### C extension (`_c_kernel/tet_index.c`)

Three exported functions:
- `tet_index_series(m, e, qq_order) → dict` — the full tetrahedron index (≈12× faster than Python)
- `tet_degree_x2(m, e) → int` — degree lower bound, pure integer
- `poly_convolve(prod, s, budget) → dict` — sparse polynomial convolution with cutoff

Falls back to Python silently if not compiled (`_HAS_C_KERNEL = False`).

### Doubled exponents

All η exponents are stored as `2 × exp` (integer). This avoids `Fraction` in the hot inner loops of `compute_refined_index` and `compute_refined_index_batch`.

### Scaled inverse `g_NZ_inv_scaled()`

Returns `(S, S·g^{-1})` as `int64`. The scale S is the LCD of all entries of g^{-1} — typically 2. This converts the integrality check from `Fraction` operations to integer modular arithmetic:

```python
S, g_inv_S = nz_data.g_NZ_inv_scaled()
charges = g_inv_S @ kappa  # int64 matmul
if all(c % S == 0 for c in charges):
    tet_args = [(charges[j] // S, charges[n+j] // S) for j in range(n)]
```

### x2 degree arithmetic

`_tet_degree_x2(m, e)` returns `2 × min_degree(I_Δ(m,e))` as a plain integer. All degree-bound comparisons multiply thresholds by 2. This avoids any float operations in the feasibility filter.

### Dynamic budget shrinking

In `compute_index_3d_python` (and `compute_refined_index`), each tet-index series call gets a tighter cutoff:

```python
cutoff = budget - prod_min_pow
```

where `prod_min_pow` accumulates the minimum powers of all previously multiplied series. This eliminates computing high-power tet-index terms that will always get truncated after convolution.

### Vectorised degree feasibility

`_degree_feasible_row` in `kernel_cache.py` uses numpy int32 broadcasting over all (e_half, target) pairs simultaneously — O(E×T) array operations instead of O(E×T) Python loops.

### Batch refined index

`compute_refined_index_batch()` pre-computes the enumeration state (g^{-1}, valid patterns, etc.) once, then reuses it across all (m_ext, e_ext) evaluations. This is the dominant speedup for the 45^r grid scan in the GUI.

### Process pool for kernel application

`apply_precomputed_kernel` uses `ProcessPoolExecutor` to compute I^ref in parallel for large grids, bypassing Python's GIL.

---

## 10. Testing

### Test structure

```
tests/
  conftest.py          — shared fixtures (nz_m004, nz_m003, nz_v0901)
  test_manifold.py     — ManifoldData loading, matrix shape assertions
  test_gluing_equations.py — variable reduction, independent edge count
  test_phase_space.py  — easy edge discovery, pattern consistency
  test_neumann_zagier.py — g_NZ symplecticity, inverse correctness
  test_basis_selection.py — cycle choice, basis change roundtrip
  test_index_3d.py     — 3D index values vs. known results
  test_refined_index.py — refined index, η-projection consistency
  test_dehn_filling.py — filling kernel, NC cycle detection
  test_weyl_check.py   — Weyl symmetry verification
  test_refined_dehn_filling.py — HJ-CF, I_S chain, refined filling
  test_exporters.py    — export format correctness
```

### Fixtures (`conftest.py`, session scope)

All fixtures require `snappy` (skipped if not installed):

```python
@pytest.fixture(scope="session")
def nz_m004():
    """NZ data for m004 (2 tet, 1 cusp, 1 hard edge)."""
    data = load_manifold("m004")
    easy = find_easy_edges(data)
    return build_neumann_zagier(data, easy)

@pytest.fixture(scope="session")
def nz_m003():
    """NZ data for m003 (2 tet, 1 cusp, all easy edges)."""
    ...

@pytest.fixture(scope="session")
def nz_v0901():
    """NZ data for v0901 (7 tet, 1 cusp, non-unit Smith factors)."""
    ...
```

`v0901` is important for testing the `_int_right_inverse` with non-unit Smith invariant factors (S > 2).

### Running tests

```bash
pytest tests/                    # all tests
pytest tests/ -v                 # verbose
pytest tests/ --timeout=30       # with timeout (pytest-timeout)
pytest tests/ --cov=manifold_index  # with coverage
```

Total runtime: ~0.44 seconds for 21 core tests.

---

## 11. Build and Packaging

### Build system

- `pyproject.toml`: setuptools + setuptools-scm
- C extension compiled from `src/manifold_index/core/_c_kernel/tet_index.c` at `pip install` time
- Package data includes:
  - `data/kernel_cache/*.pkl.gz` (bundled pre-computed kernels)
  - `mathematica/*.wl` (reference Wolfram notebooks)

### PyInstaller (frozen macOS app)

```bash
./build_app.sh           # builds ManifoldIndex.app
```

`ManifoldIndex.spec`: pinned dependency list, runtime hooks for SnaPy/multiprocessing.
`launcher.py`: entry point for frozen app — calls `multiprocessing.freeze_support()` before launching GUI.

### Dependencies

| Package | Role |
|---------|------|
| `snappy` | Manifold database and triangulation |
| `numpy` | Matrix operations, vectorised arithmetic |
| `scipy` | QR decomposition (independent edge selection), `lstsq` |
| `PySide6` (optional) | GUI framework |

### Development dependencies (`[dev]`)

`pytest`, `pytest-cov`, `pytest-timeout`, `ruff` (linter), `mypy` (type checker)

### Ruff configuration

```toml
[tool.ruff]
line-length = 100
target-version = "py310"
```

---

*End of Comprehensive Guide*
