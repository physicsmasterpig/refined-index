# Manifold Index v0.4 — Complete Reconstruction Blueprint

> **Purpose of this document:** This is the single source of truth for
> reconstructing the `manifold-index` project from scratch. Every module,
> every data structure, every algorithm, every formula, and every design
> decision is recorded here so that an LLM agent (Sonnet) can implement
> each phase independently without needing to re-read the old codebase.
>
> **How to use:** Read this document fully. Then follow the Implementation
> Phases (§11) one at a time. Each phase is self-contained: it lists
> exactly which files to create, what they must contain, and how to
> test them.

---

## Table of Contents

1.  [Project Overview](#1-project-overview)
2.  [Mathematical Background](#2-mathematical-background)
3.  [Notation and Conventions](#3-notation-and-conventions)
4.  [Directory Layout](#4-directory-layout)
5.  [Module Dependency Graph](#5-module-dependency-graph)
6.  [Data Structures](#6-data-structures)
7.  [Core Pipeline — Module by Module](#7-core-pipeline)
    - 7.1  manifold.py
    - 7.2  gluing_equations.py
    - 7.3  phase_space.py
    - 7.4  neumann_zagier.py
    - 7.5  basis_selection.py
    - 7.6  index_3d.py
    - 7.7  refined_index.py
    - 7.8  weyl_check.py
    - 7.9  dehn_filling.py
    - 7.10 refined_dehn_filling.py
    - 7.11 kernel_cache.py
8.  [C Extension](#8-c-extension)
9.  [GUI Architecture](#9-gui-architecture)
10. [Export Infrastructure](#10-export-infrastructure)
11. [Implementation Phases](#11-implementation-phases)
12. [Testing Strategy & Known-Good Values](#12-testing-strategy)
13. [Performance-Critical Design Decisions](#13-performance-critical-decisions)
14. [Caching Architecture](#14-caching-architecture)
15. [Build & Packaging](#15-build-and-packaging)
16. [Gotchas & Lessons Learned](#16-gotchas-and-lessons-learned)

---

## 1. Project Overview

**manifold-index** computes topological invariants of cusped hyperbolic
3-manifolds. Given a manifold name from the SnaPy census (e.g. `"m004"`,
the figure-eight knot complement), it computes:

1. **Triangulation data** — tetrahedra count (n), cusp count (r), gluing equations
2. **3D index** I(m⃗, e⃗) — a q^{1/2}-series (Garoufalidis–Kim–Zagier)
3. **Refined index** I^ref(q; η, W) — tracks hard-edge fugacities via η^{2W_a e_{int,a}}
4. **Non-closable cycles** — surgery slopes where the filled index vanishes
5. **Dehn filling** — surgery along slope P/Q via filling kernel
6. **Refined Dehn filling** — via Hirzebruch-Jung continued fraction chain
7. **Weyl symmetry checks** — prerequisite validations for refined filling

The code is ~17,000 lines of Python + 410 lines of C, with a PySide6 GUI.

**Key dependencies:** snappy, numpy, scipy, PySide6 (optional for GUI).

**License:** GPL-2.0-or-later.

---

## 2. Mathematical Background

### 2.1 Cusped Hyperbolic 3-Manifolds

A cusped hyperbolic 3-manifold M is obtained from an ideal triangulation
of n tetrahedra with r cusps. SnaPy provides the triangulation data
including shape parameters Z_i, Z_i', Z_i'' for each tetrahedron i,
satisfying Z_i + Z_i' + Z_i'' = 1.

### 2.2 Gluing Equations

The triangulation imposes linear constraints:
- **n edge equations**: around each edge, the shape parameters sum to 2πi
- **2r cusp equations**: for each cusp, a meridian (μ) and longitude (λ) holonomy

These are encoded in an integer matrix of shape (n + 2r, 3n).

### 2.3 The 3D Index

The 3D index is:

```
I(m_ext, e_ext) = Σ_{e_int ∈ (½)Z^{n-r}}
    (-q^{1/2})^{phase}
    · ∏_{a=0}^{n-1} I_Δ(m_a, e_a)
```

where:
- m_a, e_a are derived from the external charges via g_NZ^{-1} · κ
- I_Δ is the tetrahedron index (quantum dilogarithm building block)
- The sum runs over half-integer internal edge charges
- Only terms where g_NZ^{-1} · κ has all-integer entries contribute

### 2.4 Tetrahedron Index

The Garoufalidis–Kim formula:

```
MIt(m, e):
  if m+e ≥ 0:  (-qq)^m · I_t(-m-e, m)
  else:         I_t(m, e)

I_t(m, e) = Σ_{k=max(0,-e)}^∞  (-1)^k · qq^{k(k+1)-(2k+e)·m}
             / [∏_{j=1}^{k}(1-qq^{2j}) · ∏_{j=1}^{k+e}(1-qq^{2j})]
```

Degree lower bound: δ(m,e) = ½[m₊(m+e)₊ + (-m)₊·e₊ + (-e)₊(-e-m)₊] + max{0,m,-e}
where x₊ = max(0, x).

### 2.5 Neumann-Zagier Matrix

g_NZ ∈ Sp(2n, ℚ) is the symplectic change-of-basis from shape variables
(Z_i, Z_i'') to canonical position/momentum variables (cusp charges,
internal edge charges). It satisfies g_NZ · Ω · g_NZ^T = Ω.

### 2.6 Easy vs Hard Edges

An edge is **easy** if, in its 3n-dimensional representation, at most
one of {Z_i, Z_i', Z_i''} is nonzero per tetrahedron. Easy edges produce
a simple (near-diagonal) NZ matrix. **Hard edges** require a fugacity
variable η with weights W_a to track their contributions.

### 2.7 Refined Index

Like the 3D index, but hard-edge charges are tracked as fugacity exponents
η^{2W_a e_{int,a}} instead of being absorbed into the q-series.  Here η
is a single formal fugacity variable and W_a are formal weight variables
(one per hard edge a = 0,…,k-1).  The product over hard edges gives

    ∏_a η^{2W_a e_{int,a}} = η^{Σ_a 2W_a e_{int,a}}

The per-edge exponents e_{int,a} are stored individually in the key; the
W_a parametrise the projection at evaluation time.  Setting all W_a = 0
(equivalently η = 1) recovers the ordinary 3D index.

### 2.8 Dehn Filling

Surgery along slope P/Q at a cusp uses the filling kernel:

```
K(P,Q; m,e) = ½·(-1)^{Rm+2Se}·
  [δ_{Pm+2Qe,0}·(q^{t/2} + q^{-t/2}) - δ_{Pm+2Qe,-2} - δ_{Pm+2Qe,+2}]
```

where t = Rm + 2Se and R·Q - P·S = 1.

### 2.9 Refined Dehn Filling (HJ-CF Chain)

For |Q| ≥ 2, the refined filling kernel uses the Hirzebruch-Jung
continued fraction P/Q = k₁ - 1/(k₂ - 1/(…)) of length ℓ,
and a chain of ℓ-1 symplectic kernels I_S plus a final K-factor.

### 2.10 Weyl Symmetry

Before refined Dehn filling, the refined index must satisfy:
1. The Weyl-shifted series
   f(m,e) = η^{Σ_a 2W_a(b_a m + a_a e)} · I^ref(m,e)
   is symmetric under flipping each per-edge exponent e_{int,a} → −e_{int,a}
   (equivalently η → η^{-1} for every choice of W_a, since W_a are formal)
2. The q^1 coefficient has adjoint su(2) character structure

---

## 3. Notation and Conventions

### 3.1 Symbol Map (Paper ↔ Code)

| Paper symbol | Code variable | Meaning |
|:-------------|:--------------|:--------|
| r | `n` / `num_tetrahedra` | Number of tetrahedra |
| n | `r` / `num_cusps` | Number of cusps |
| Z_i, Z'_i, Z''_i | columns of `gluing_matrix` | Shape parameters per tet |
| q^{1/2} | `qq` | Half-power variable |
| — | `qq_order` / `q_order_half` | Max power of qq kept |
| η | `eta` | Single fugacity variable |
| W_a | `W[a]` | Weight for hard edge a (formal variable) |
| M_i, L_i | meridian, longitude | Cusp basis cycles |
| Λ_i = L_i/2 | `e_ext[i]` | Cusp momentum (half-longitude) |
| P, Q | `P`, `Q` | Surgery slope: P·M + Q·L |
| g_NZ | `nz_data.g_NZ` | Symplectic (2n×2n) Neumann-Zagier matrix |
| κ | `kappa` | Combined (m_full, e_full) vector of size 2n |
| ν_x | `nu_x` | Affine shift of position rows |
| ν_p | `nu_p` | Affine shift of momentum rows |

### 3.2 Numeric Conventions

| Convention | Description |
|:-----------|:------------|
| `qq_order` = max power of q^{1/2} | Every truncation-order parameter |
| RefinedIndexResult key = `(qq_pow, 2*e_int_0, 2*e_int_1, …)` | Per-edge exponents stored doubled; η^{2W_a e_a} at eval time |
| `g_NZ_inv_scaled()` returns `(S, S·g^{-1})` as int64 | Avoids Fraction arithmetic in hot paths |
| `_is_kernel()` returns `2·I_S` as int | Absorbs ½ prefactor; LCD tracks as 2^ℓ |
| `e_ext` values can be half-integer (Fraction) | From cusp longitude/2 |
| Content-based NZ fingerprint for cache keys | NEVER use `id(nz_data)` |

### 3.3 Column Ordering

- **Gluing matrix** columns: Z₁, Z₁', Z₁'', Z₂, Z₂', Z₂'', … (interleaved triples)
- **g_NZ** columns: Z₁,…,Z_n, Z₁'',…,Z_n'' (block ordering)
- Conversion: `_interleaved_to_block(coeff, n)` permutation

### 3.4 Doubled Exponents Rationale

All η fugacity exponents are half-integers in general (e_int can be k/2).
Storing `2 × exponent` as plain `int` in dict keys avoids `Fraction`
in the innermost loops.  A key like `(qq_pow, 4, -2)` means
q^{qq_pow/2} · η^{2W_0 · 2} · η^{2W_1 · (-1)} = q^{qq_pow/2} · η^{4W_0 - 2W_1}.
Since W_a are formal variables, the key stores per-edge doubled exponents
(4 and -2), and the total η power Σ_a 2W_a · (key[1+a]/2) is computed
at evaluation time once specific W_a are chosen.

---

## 4. Directory Layout

```
v0.4/
├── BLUEPRINT.md              ← this file
├── pyproject.toml
├── README.md
├── src/
│   └── manifold_index/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── manifold.py           (~160 lines)
│       │   ├── gluing_equations.py   (~220 lines)
│       │   ├── phase_space.py        (~460 lines)
│       │   ├── neumann_zagier.py     (~780 lines)
│       │   ├── basis_selection.py    (~375 lines)
│       │   ├── index_3d.py           (~1010 lines)
│       │   ├── refined_index.py      (~440 lines)
│       │   ├── weyl_check.py         (~1260 lines)
│       │   ├── dehn_filling.py       (~840 lines)
│       │   ├── refined_dehn_filling.py (~2900 lines)
│       │   ├── kernel_cache.py       (~1600 lines)
│       │   ├── data_packs.py         (~380 lines)
│       │   └── _c_kernel/
│       │       └── tet_index.c       (~410 lines)
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── exporters.py          (~1600 lines)
│       │   ├── math_utils.py
│       │   └── io_utils.py
│       ├── app/
│       │   ├── __init__.py
│       │   ├── __main__.py
│       │   ├── window.py             (~276 lines)
│       │   ├── workers.py            (~700 lines)
│       │   ├── formatters.py         (~1170 lines)
│       │   ├── katex.py              (~214 lines)
│       │   ├── style.py              (~154 lines)
│       │   └── panels/
│       │       ├── __init__.py
│       │       ├── manifold_panel.py (~272 lines)
│       │       ├── filling_panel.py  (~383 lines)
│       │       ├── export_panel.py   (~323 lines)
│       │       ├── kernel_panel.py   (~530 lines)
│       │       └── data_panel.py     (~395 lines)
│       └── data/
│           └── kernel_cache/         (bundled .pkl.gz files)
├── tests/
│   ├── conftest.py
│   ├── test_manifold.py
│   ├── test_gluing_equations.py
│   ├── test_phase_space.py
│   ├── test_neumann_zagier.py
│   ├── test_basis_selection.py
│   ├── test_index_3d.py
│   ├── test_refined_index.py
│   ├── test_weyl_check.py
│   ├── test_dehn_filling.py
│   ├── test_refined_dehn_filling.py
│   └── test_exporters.py
├── data/
│   ├── kernel_cache/                 (user-generated caches)
│   └── iref_cache/
└── docs/
```

**Total:** ~17,000 lines Python + 410 lines C across ~35 source files.

---

## 5. Module Dependency Graph

Arrows mean "imports from". No circular dependencies.

```
manifold  →  gluing_equations  →  phase_space
                                       |
                                       v
                                  neumann_zagier  →  basis_selection
                                       |
                     +-----------------+
                     v                 v
                 index_3d  ←——  refined_index
                     |
          +----------+
          v          v
     dehn_filling  weyl_check  ←—  refined_index
          |
          v
  refined_dehn_filling  →  kernel_cache
```

- `app/workers.py` imports from every `core/` module
- `app/formatters.py` imports data classes from `core/`
- `utils/exporters.py` imports data classes from `core/`

---

## 6. Data Structures

All major containers are **frozen dataclasses** (immutable, thread-safe,
suitable as cache components).

### 6.1 ManifoldData

```python
@dataclass(frozen=True)
class ManifoldData:
    name: str                     # SnaPy name, e.g. "m004"
    num_tetrahedra: int           # n
    num_cusps: int                # r
    gluing_matrix: np.ndarray     # shape (n + 2r, 3n), int
    raw: Any                      # snappy.Manifold object

    # Properties:
    # edge_equations -> gluing_matrix[:n]
    # meridian_equations -> gluing_matrix[n::2]
    # longitude_equations -> gluing_matrix[n+1::2]
    # cusp_equations(k) -> (meridian_row, longitude_row)
```

**Gluing matrix row layout:**
- Rows 0…n-1: edge equations (n rows)
- Row n+2k: meridian μ_k (k=0…r-1)
- Row n+2k+1: longitude λ_k
- Columns grouped in triples per tet: Z₁, Z₁', Z₁'', Z₂, Z₂', Z₂'', …

### 6.2 ReducedGluingData

```python
@dataclass(frozen=True)
class ReducedGluingData:
    n: int                          # tetrahedra
    r: int                          # cusps
    edge_coeffs: np.ndarray         # (n, 2n), int
    edge_consts: np.ndarray         # (n,), int
    cusp_coeffs: np.ndarray         # (2r, 2n), int
    cusp_consts: np.ndarray         # (2r,), int
    independent_edge_indices: list   # length n-r
    symplectic_matrix: np.ndarray   # (2n, 2n), int — Ω
```

After eliminating Z' via Z' = 1 - Z - Z'', the 3n variables reduce to 2n.
The 2n variables are interleaved: (Z₁, Z₁'', Z₂, Z₂'', …).

### 6.3 EasyEdgeResult

```python
@dataclass(frozen=True)
class EasyEdgeResult:
    all_easy: list[np.ndarray]          # all discovered easy edges (3n-vectors)
    independent_easy_indices: list[int] # indices into all_easy
    hard_padding: list[np.ndarray]      # hard edges appended to reach n-r
    n: int
    r: int

    @property
    def basis_edges(self) -> list:
        """[independent_easy ... | hard_padding ...], length n-r."""
```

### 6.4 NeumannZagierData

```python
@dataclass
class NeumannZagierData:
    g_NZ: np.ndarray          # (2n, 2n), float64 (exact rationals as float)
    nu_x: np.ndarray          # (n,), int — position affine shifts
    nu_p: np.ndarray          # (n,), float — momentum affine shifts
    n: int
    r: int
    num_hard: int
    num_easy: int

    # Cached methods:
    # g_NZ_inv() -> Fraction array (exact via symplectic identity)
    # g_NZ_inv_scaled() -> (S: int, S·g^{-1}: int64 array)
    # is_symplectic(tol) -> bool
    # inv_denom -> int (just S from g_NZ_inv_scaled)
```

**g_NZ row structure (2n rows):**

| Row range | Content |
|:----------|:--------|
| 0…r-1 | Meridians (cusp positions) |
| r…r+num_hard-1 | Hard internal edges |
| r+num_hard…n-1 | Easy internal edges |
| n…n+r-1 | Longitude/2 (cusp momentum) |
| n+r…2n-1 | Γ vectors (isotropic complement) |

**Column ordering:** Block form — cols 0…n-1 = Z₁,…,Z_n; cols n…2n-1 = Z₁'',…,Z_n''

**Symplectic inverse identity:**
g^{-1} = [[D^T, -B^T], [-C^T, A^T]] where g = [[A,B],[C,D]] in n×n blocks.

### 6.5 CycleChoice / BasisSelection

```python
@dataclass(frozen=True)
class CycleChoice:
    cusp_idx: int
    P: int
    Q: int
    label: str = ""
    is_default: bool = False

    @property
    def m(self) -> int: return self.P
    @property
    def e(self) -> Fraction: return Fraction(self.Q, 2)

@dataclass(frozen=True)
class BasisSelection:
    choices: list[CycleChoice]  # one per cusp

    @property
    def m_ext(self) -> list[int]: ...
    @property
    def e_ext(self) -> list[Fraction]: ...
```

### 6.6 Index3DResult

```python
@dataclass(frozen=True)
class Index3DResult:
    coeffs: list[int]       # dense coefficients from min_power to q_order_half
    min_power: int           # lowest nonzero qq-power
    q_order_half: int        # cutoff used
    m_ext: list              # input charges
    e_ext: list
    n_terms: int             # number of nonzero summation terms
```

### 6.7 Type Aliases

```python
QSeries          = dict[int, Fraction]              # qq_power → coeff
RefinedIndexResult = dict[tuple[int, ...], int]     # (qq, 2·e_int_0, …) → coeff
QEtaSeries       = dict[tuple[int, int], Fraction]  # (qq_pow, η_exp) → coeff
MultiEtaSeries   = dict[tuple[int, ...], Fraction]  # multi-cusp key → coeff
```

### 6.8 KernelTerm

```python
@dataclass(frozen=True)
class KernelTerm:
    m: int
    e: Fraction
    c: int                  # P·m + 2Q·e ∈ {-2, 0, 2}
    phase: int              # R·m + 2S·e
    multiplicity: int = 1   # 2 for antipodal, 1 for origin
```

### 6.9 KernelTable

```python
@dataclass
class KernelTable:
    P: int
    Q: int
    qq_order: int
    qq_internal: int
    eta_order: int
    hj_ks: list[int]
    table: dict[tuple[int, Fraction], QEtaSeries]  # (m, e) → series
    m_scan: range
    e_scan: list[Fraction]
    # _fast_lcd, _fast_grouped — dense int64 numpy for fast application
```

### 6.10 ABVectors

```python
@dataclass
class ABVectors:
    a: list[Fraction]       # coupling to longitude e; must be integer
    b: list[Fraction]       # coupling to meridian m; must be half-integer
    num_hard: int
    warnings: list[str]

    @property
    def is_valid(self) -> bool: ...   # a ∈ Z and 2b ∈ Z for all entries
```

### 6.11 FilledRefinedResult

```python
@dataclass
class FilledRefinedResult:
    series: MultiEtaSeries
    num_hard: int
    has_cusp_eta: bool
    num_cusp_eta: int
    # Key structure (η^{2W_a e_{int,a}} notation):
    #   L=1: (qq, 2*e_int_0, …, 2*e_int_{H-1})
    #   L≥2: (qq, 2*e_int_0, …, 2*e_int_{H-1}, cusp_η)
```

---

## 7. Core Pipeline — Module by Module

### 7.1 manifold.py (~160 lines)

**Purpose:** Load a manifold from SnaPy, extract gluing equations.

**Entry point:**
```python
def load_manifold(name: str) -> ManifoldData
```

**Algorithm:**
1. Call `snappy.Manifold(name)`
2. Get `gluing_equations("rect")` → list of n+2r rows of 3n+1 values
3. Extract the integer coefficient matrix (first 3n columns)
4. Validate: n rows for edges, 2r rows for cusps, columns = 3n
5. Return frozen `ManifoldData`

**Properties on ManifoldData:**
- `edge_equations` → `gluing_matrix[:n]`
- `meridian_equations` → `gluing_matrix[n::2]`
- `longitude_equations` → `gluing_matrix[n+1::2]`
- `cusp_equations(k)` → `(gluing_matrix[n+2k], gluing_matrix[n+2k+1])`

---

### 7.2 gluing_equations.py (~220 lines)

**Purpose:** Reduce 3n variables to 2n by eliminating Z'.

**Entry point:**
```python
def reduce_gluing_equations(data: ManifoldData) -> ReducedGluingData
```

**Algorithm:**

**`_reduce_row(row_3n, n)`** — for each tet i with coefficients (f_i, g_i, h_i):
```
const += g_i
coeff[2i]   = f_i - g_i    # coefficient of Z_i
coeff[2i+1] = h_i - g_i    # coefficient of Z_i''
```
Returns `(const, coeff_2n)`.

**`_build_symplectic_matrix(n)`** — 2n×2n block-diagonal:
Ω[2i, 2i+1] = +1, Ω[2i+1, 2i] = -1. (Interleaved ordering.)

**`_independent_row_indices(coeff_matrix, expected_rank)`** — Column-pivoted
QR on transposed matrix. Returns n-r pivot indices = the n-r independent
edge equations.

**Main function:**
1. Apply `_reduce_row` to each of the n+2r rows
2. Split into edge_coeffs (n, 2n), cusp_coeffs (2r, 2n)
3. Find independent edges (rank = n-r)
4. Build Ω
5. Return `ReducedGluingData`

---

### 7.3 phase_space.py (~460 lines)

**Purpose:** Classify edges as easy/hard, build phase-space basis.

**Entry point:**
```python
def find_easy_edges(data: ManifoldData, tol=1e-10) -> EasyEdgeResult
```

**Key helper `_is_easy(edge_3n, n)`:** Returns True if in the 3n-vector,
at most one of (Z_i, Z_i', Z_i'') is nonzero per tetrahedron i.

**Algorithm B (pattern-first search):**

Constants: `_Z=0, _ZP=1, _ZPP=2, _OFF=3`

**Stage 0 (fast path):** Check raw SnaPy edge rows for easiness directly:
non-negative entries, at most one nonzero per tet triplet, sum=2.

**Stage 1 (pattern enumeration):** For each pattern ∈ {Z, Z', Z'', OFF}^n:
1. Build constraint matrix from gluing_matrix edge rows
2. For active tet j with slot s and two inactive slots s1, s2:
   add constraint rows Σ_i a_i · (col_{s1} - col_{s2}) = 0
3. For OFF tet j: two difference rows = 0
4. Add normalization: a · (2·ones - Σ ref_cols) = 2

**Solving:**
- **Fast path:** `scipy.linalg.lstsq` + round. Verify residual < tol.
- **Slow path (underdetermined):** Exact `Fraction`-based RREF +
  free-variable enumeration with |coefficients| ≤ MAX_COEFF=3.

**Reconstruction:** `E_3n = Σ a_i · edge_row_i + Σ b_j · T_j` where
T_j = (1,1,1) at tet j. `_compute_b()` solves for the per-tet constant b_j.

**Stage 2:** QR column-pivoting on all reduced easy edges → max
independent subset.

**Stage 3:** Pad with SnaPy independent edges (SVD rank check) until
n-r total basis edges.

**Convenience alias:**
```python
def find_phase_space_basis(manifold: ManifoldData) -> EasyEdgeResult
```

---

### 7.4 neumann_zagier.py (~780 lines)

**Purpose:** Build the symplectic NZ matrix g_NZ ∈ Sp(2n, ℚ).

**Entry point:**
```python
def build_neumann_zagier(data: ManifoldData, easy: EasyEdgeResult) -> NeumannZagierData
```

**Algorithm:**

1. **Position block P (n rows × 2n cols):**
   - Rows 0…r-1: Meridians (from reduced cusp coefficients)
   - Rows r…r+num_hard-1: Hard edges (from basis_edges, reduced to block form)
   - Rows r+num_hard…n-1: Easy edges (from basis_edges, reduced to block form)

2. **Momentum block Q_long (r rows × 2n cols):**
   - Rows 0…r-1: Longitude/2 (from reduced cusp coefficients)

3. **Γ construction (n-r rows × 2n cols):**
   Build the system `[P; Q_long] · Ω_block · Γ^T = RHS` where
   RHS[r:n, :] = I_{n-r}. Solve via `_int_right_inverse`.

**`_int_right_inverse(A_int)`:**
Exact integer column reduction (Euclidean algorithm) tracking
transformation matrix V. Uses `Fraction` for Smith-factor denominators.
Iteratively reduces columns via gcd operations, producing a right-inverse
A_right such that A · A_right = I.

**`_make_isotropic(gamma_raw, long_half_block)`:**
Correction: gamma → gamma - ½ gamma·Ω·gamma^T applied iteratively
so that the resulting Γ rows are mutually isotropic (Γ_i · Ω · Γ_j^T = 0).

**Affine shifts (ν):**
- Meridians (RHS=0): ν_x = const
- Internal edges (RHS=2): ν_x = const - 2
- Longitudes/2: ν_p = const_long / 2
- Γ rows: ν_p = 0

**Conversion helper:**
`_interleaved_to_block(coeff_2n, n)`: permutation
(Z₁,Z₁'',Z₂,Z₂'',…) → (Z₁,…,Z_n,Z₁'',…,Z_n'')

`_reduce_to_block(row_3n, n)`: compose _reduce_row + _interleaved_to_block.

**Ω in block form:**
`_build_omega_block(n)` → 2n×2n: [[0, I_n], [-I_n, 0]].

**Methods on NeumannZagierData:**

- `is_symplectic(tol)`: checks g_NZ · Ω · g_NZ^T = Ω
- `g_NZ_inv()`: symplectic identity → Fraction array. **Cached.**
- `g_NZ_inv_scaled()`: LCD S of all entries; returns (S, int64 S·g^{-1}). **Cached.**
- `inv_denom`: property, just S.

**Cusp basis changes:**

`apply_cusp_basis_change(nz, cusp_idx, P, Q)` — requires P odd.
Uses Bezout: P·b - 2Q·a = 1. Transforms meridian/longitude rows
of g_NZ by the SL(2,ℤ) matrix.

`apply_general_cusp_basis_change(nz, cusp_idx, a, b, c, d)` —
general SL(2,ℤ); allows half-integer entries in result.

**`_ext_gcd(a, b)`:** Extended Euclidean → (g, x, y) with a·x + b·y = g.

---

### 7.5 basis_selection.py (~375 lines)

**Purpose:** Per-cusp cycle selection for refined index computation.

**Entry points:**
```python
def make_basis_selection(nz_data, cycle_results, choices, *, default="M", strict=False) -> BasisSelection
def apply_basis_changes(nz_data, basis) -> NeumannZagierData
```

**Defaults:**
- `default_meridian_choice(k)`: (P,Q)=(1,0), m=1, e=0
- `default_longitude_choice(k)`: (P,Q)=(0,1), m=0, e=½

**Logic:**
- `choices[i]` can be `(P,Q)` or `None` (use default)
- Validates primitivity (gcd(P,Q)=1)
- `strict=True` checks that the chosen slope was found non-closable
- `apply_basis_changes`: for each cusp with odd P, calls `apply_cusp_basis_change`

---

### 7.6 index_3d.py (~1010 lines)

**Purpose:** Compute the 3D index I(m⃗, e⃗) as a q^{1/2}-series.

**Entry points:**
```python
def compute_index_3d_python(nz_data, m_ext, e_ext, q_order_half=20, *, _precomputed_terms=None) -> Index3DResult
def enumerate_summation_terms(nz_data, m_ext, e_ext, q_order_half) -> list[dict]
```

**Tetrahedron index (`_tet_index_series(m, e, qq_order) → dict[int, int]`):**
- Returns {qq_power: coefficient}
- C extension preferred; pure-Python fallback `_tet_index_series_python`
- Module-level memoization: `_tet_cache[(m, e, qq_order)]`
- Uses Garoufalidis-Kim `MIt` symmetry reduction

**Degree bound (`_tet_degree_x2(m, e) → int`):**
Returns `2 × δ(m,e)` as plain int (no Fraction, no float).

**Enumeration State (`_EnumerationState`, cached per NZ content):**
Pre-computes:
- `g_inv_x2`: scaled inverse × 2 (or more generally S times inverse)
- Valid half-integer patterns `delta ∈ {0,1}^{n-r}`
- Per-pattern contributions (delta_contrib_x4, delta_phase_x2)
- Cusp columns

**`_enumerate_with_state(state, m_ext, e_ext, q_order_half)`:**
1. For each valid delta pattern, compute base_args_x4
2. Check divisibility by 4 (integrality)
3. Find integer offsets e0 via `_exact_e0_candidates()`
4. For each valid e0, compute tet_args and phase_exp

**`_exact_e0_candidates(base, cols, budget_x4, delta_deg_x4)`:**
- Per-axis projection bounds via `_axis_scan_bound`
- `_proj_min_fixed`: fix all but one axis, find minimum degree
- Bounding-box scan over integer lattice
- Returns list of (e0_vec, tet_args_x4, phase_x2)

**`compute_index_3d_python`:**
1. Call `enumerate_summation_terms`
2. For each term, multiply n tet-index series via sparse dict convolution
3. Apply phase factor `(-qq)^{phase_exp}`
4. Dynamic budget shrinking: `cutoff = budget - prod_min_pow`
5. Aggregate all terms into final series

**C extension helpers:**
- `_c_tet_index_series(m, e, qq_order)` → dict
- `_c_poly_convolve(prod, s, budget)` → dict

**Cache clearing:**
- `clear_tet_cache()`
- `clear_enum_state_cache()`

---

### 7.7 refined_index.py (~440 lines)

**Purpose:** Refined index tracking hard-edge fugacities via η^{2W_a e_{int,a}}.

**Entry points:**
```python
def compute_refined_index(nz_data, m_ext, e_ext, q_order_half=20) -> RefinedIndexResult
def compute_refined_index_batch(nz_data, entries, q_order_half) -> list[RefinedIndexResult]
```

**Key difference from 3D index:** The first `k = num_hard` entries of e_int
are tracked as fugacity exponents instead of being summed into q-series.

**Key extraction:**
```python
eta_exps_x2 = tuple(int(Fraction(e_int_strs[a]) * 2) for a in range(k))
key = (shifted_power,) + eta_exps_x2
result[key] += sign * coeff
```

When `num_hard = 0`: keys are length-1 `(qq_power,)` — same as 3D index.

**Batch optimization:** `compute_refined_index_batch` pre-computes
`_get_enum_state(nz_data)` once, reuses for all (m_ext, e_ext) pairs.
Typically 10-100× faster for grid evaluations.

**Utilities:**
- `project_to_3d_index(refined) → dict[int, int]`: sum over all η monomials per q-power
- `format_refined_index(refined, num_hard, q_var, eta_vars) → str`
- `format_multi_point_index(entries, ...) → str`

---

### 7.8 weyl_check.py (~1260 lines)

**Purpose:** Verify Weyl symmetry prerequisites before Dehn filling.

**Entry points:**
```python
def compute_ab_vectors(entries, num_hard) -> ABVectors
def compute_ab_vectors_for_cusp(nz_data, cusp_idx, q_order_half) -> ABVectors
def run_weyl_checks(entries, num_hard) -> WeylCheckResult
def check_weyl_symmetry(entries, num_hard, ab) -> bool
def strip_weyl_monomial(result, m_ext, e_ext, ab, num_hard) -> RefinedIndexResult
def check_adjoint_projection(entries, num_hard, ab) -> AdjointProjectionResult
def scan_w_vectors(nz_data, cusp_idx, q_order_half) -> WScanResult
```

**ABVectors extraction:**
```
b[j] = -[centre(+m,0) - centre(-m,0)] / (2m)
a[j] = -[centre(0,+e) - centre(0,-e)] / (2e)
```
where centre = weighted average of per-edge exponents at leading q-order.

**Weyl symmetry check:** After stripping the Weyl monomial
η^{Σ_a 2W_a(a_a·e + b_a·m)}, the per-edge exponents of the refined index
must be symmetric under e_{int,a} → −e_{int,a}.

**Adjoint projection:** q^1 coefficient must satisfy:
```
(½)(c_{-1} + c_{+1} - c_{-2} - c_{+2}) = -1
```

**W-vector scan:** Search integer specialisations of (W_0,…,W_{k-1}) that
make the adjoint check pass when hard edges are not Dehn-filling compatible.

**`strip_weyl_monomial(result, m_ext, e_ext, ab, num_hard)`:**
Multiplies I^ref by η^{Σ_a 2W_a(a_a·e + b_a·m)} — shifts per-edge
exponents in each key by `a[j]*e + b[j]*m` for hard edge j.

---

### 7.9 dehn_filling.py (~840 lines)

**Purpose:** Ordinary (unrefined) Dehn filling and non-closable cycle search.

**Entry points:**
```python
def compute_filled_index(nz_data, cusp_idx, P, Q, q_order_half=20) -> FilledIndexResult
def find_non_closable_cycles(nz_data, cusp_idx, ...) -> NonClosableCycleResult
def enumerate_kernel_terms(P, Q, R, S, nz_data, ...) -> list[KernelTerm]
def find_rs(P, Q) -> tuple[int, int]
```

**Kernel formula:**
```
K(P,Q; m,e) = ½·(-1)^{Rm+2Se}·
  [δ_{Pm+2Qe,0}·(q^{t/2} + q^{-t/2}) - δ_{Pm+2Qe,-2} - δ_{Pm+2Qe,+2}]
```
where t = Rm + 2Se and R·Q - P·S = 1.

**Kernel term enumeration:**
- For each c ∈ {0, 2} (c=-2 handled by antipodal symmetry):
  1. Particular solution via `_ext_gcd`
  2. General family: m_t = m_c + Q·t, e_t = e_c - P·t/2
  3. Degree filter against adjusted budget
  4. Stop after 2 consecutive empty steps

**Antipodal symmetries:**
- c=0: t and -t contribute identically → multiplicity=2 for t>0
- c=+/-2: (m,e) and (-m,-e) are antipodal → skip c=-2, double c=2
- (P,Q) ↔ (-P,-Q): canonical half search

**Summation-cache optimization:**
`enumerate_kernel_terms` pre-computes `enumerate_summation_terms` during
the degree filter and stores results in `_summation_cache` so
`compute_index_3d_python` can reuse them via `_precomputed_terms`.

**QSeries arithmetic:**
`_qseries_shift`, `_qseries_scale`, `_qseries_add`, `_qseries_truncate`,
`_apply_kernel`

**`_apply_kernel(term, index_series)`:**
- c=0: multiply by `½(q^{t/2} + q^{-t/2})·(-1)^t`, i.e., average of +t and -t shifts
- c=2: multiply by `-½·(-1)^t`

**Non-closable cycles:**
- Scan slopes in p_range × q_range
- Fill each, check `is_stably_zero()` (ignores top `buffer` powers)
- `_candidate_slopes`: generates primitive (P,Q) pairs with |P| ≤ max, |Q| ≤ max

**Key dataclasses:**
- `FilledIndexResult`: series (QSeries), is_stably_zero flag
- `NonClosableCycle`: cusp_idx, P, Q, label
- `NonClosableCycleResult`: list of cycles, search parameters

---

### 7.10 refined_dehn_filling.py (~2900 lines)

**Purpose:** Refined Dehn filling via HJ continued fraction chain.

**Entry points:**
```python
def compute_filled_refined_index(nz_data, cusp_idx, P, Q, q_order_half, eta_order, *, ...) -> FilledRefinedResult
def compute_multi_cusp_filled_refined_index(nz_data, fill_specs, q_order_half, eta_order, *, ...) -> FilledRefinedResult
def hj_continued_fraction(P, Q) -> list[int]
def clear_filling_caches() -> dict[str, int]
def clear_computation_caches() -> None
```

**HJ-CF expansion:**
```
P/Q = k₁ - 1/(k₂ - 1/(… - 1/k_ℓ))
```
Special cases: Q=0,P=±1 → [0,0]; |Q|=1 → L=1.

**Three paths in `compute_filled_refined_index`:**
1. **L=1:** K(k₁,1) · I^ref — ordinary kernel, no IS chain
2. **L≥2, cached kernel exists:** `apply_precomputed_kernel()` (sub-second)
3. **L≥2, no cache:** Grid scan + IS chain convolution + final K-factor

**I_S kernel (`_is_kernel(m1, e1, m2, e2, qq_order, eta_order)`):**
Returns `2 · I_S` as int. LRU-cached. Uses `_etilde_is` inner function.

**`_etilde_is(m1, e1, m2, e2, qq_order, eta_order)`:**
4-fold tet-index product summed over (e, t) double loop:
```
ẽI_S(m1,e1,m2,e2;η) = Σ_{e,t} η^e
    · I_Δ(-e1 - m2/2,  -e/2 + e1 + m1/2 + t)
    · I_Δ( e1 + m2/2,  -e/2 + e2 - m2/2 + t)
    · I_Δ(-e2 - m1/2,   e2 + m1/2 + t)
    · I_Δ( e2 + m1/2,   e1 - m2/2 + t)
    · (-q^{1/2})^{-e + e1 + e2 + m1/2 - m2/2 + 2t}
```
Parity filter: e has parity (m1+m2) mod 2.

**IS chain convolution (`_apply_is_step`):**
- Input: MultiEtaSeries (accumulated state)
- For each (m_prev, e_prev) in state, convolve with
  I_S(m_prev, -e_prev - k_j/2·m_prev, m_next, e_next) over all (m_next, e_next)
- int-mode: LCD accumulates as 2^L
- Diamond truncation after each step: qq + |cusp_eta| ≤ qq_order

**Final K-factor (`_apply_k1_factor`):**
Ordinary Dehn filling kernel K(k_ℓ, 1; m, e) applied to the
last chain variable.

**Weyl shift (`_apply_weyl_shift`):**
Before the kernel chain, multiply each I^ref(m,e) by the Weyl monomial
η^{Σ_a 2W_a(a_a·e + b_a·m)}, i.e., shift per-edge exponent key[1+j]
by `2*(a[j]*e + b[j]*m)` for each hard edge j.

**Multi-cusp filling:**
- Step 1: `_batched_first_filling()` — fills cusp 0 with spectator dims
- Step 2: `_apply_filling_kernel_to_intermediate()` — fills cusp 1
- Currently supports ≤ 2 cusps

**Internal helpers:**
- `_qeta_add`, `_qeta_scale`, `_qeta_shift_qq`, `_qeta_truncate`, `_qeta_convolve`
- `_tet_series_to_qeta`, `_int_qqseries_convolve`
- `_refined_to_multi(refined, num_hard) → MultiEtaSeries`
- `_enumerate_slope1_terms`, `_enumerate_slope1_all_halfshift`
- `_enumerate_slope1_all`, `_enumerate_is_full`

**Caching:**
- `_tet_arr_cache`: numpy array versions of tet index
- `_iref_cache`: content-keyed refined index cache
- `_etilde_is.cache`, `_is_kernel.cache`: LRU caches

---

### 7.11 kernel_cache.py (~1600 lines)

**Purpose:** Pre-compute, save, load manifold-independent kernel tables.

**Entry points:**
```python
def precompute_filling_kernel(P, Q, qq_order, ...) -> KernelTable
def apply_precomputed_kernel(kernel, nz_data, ...) -> FilledRefinedResult
def save_kernel_table(table, directory=None) -> Path
def load_kernel_table(P, Q, qq_order, ...) -> KernelTable | None
def list_cached_kernels(directory=None) -> list
def clear_kernel_cache() -> int
```

**Also provides I^ref caching:**
```python
def save_iref_cache(manifold_name, nz_data, entries, q_order_half, directory=None) -> Path
def load_iref_cache(manifold_name, nz_data, q_order_half, directory=None) -> list | None
def list_iref_caches(directory=None) -> list
```

**Storage:** `data/kernel_cache/kernel_P{P}_Q{Q}_qq{qq}.pkl.gz`
- User cache: `~/Library/Caches/manifold-index/kernel_cache/` (macOS)
- Bundled cache: `src/manifold_index/data/kernel_cache/`
- Lookup order: user → bundled

**`precompute_filling_kernel` — four-phase algorithm (v3):**
1. **Parity auto-detection:** Probe m=0..3 to find which m-parity gives non-zero
2. **Probe-and-scale support prediction (PROBE_QQ=8):** Low-qq computation
   discovers non-zero support shape, then scales via center+half-width
   interpolation (WIDTH_MARGIN=1.4, MARGIN_ABS=8)
3. **Compute entries (m≥0 only):** Row-based parallel (ProcessPoolExecutor)
4. **Mirror symmetry:** K(m,e)=K(-m,-e) → copy m≥0 to -m,-e

**`apply_precomputed_kernel`:**
- Dense int64 numpy arrays from `KernelTable._fast_grouped`
- Batch I^ref computation via ProcessPoolExecutor
- np.convolve or batched 2-D matrix spread
- `_BATCH_THRESH = 4`

**Degree helpers:**
- `_tet_degree_x2(m, e)`: scalar, pure-int
- `_tdeg_arr(m, e)`: vectorised numpy version
- `_is_kernel_min_degree_x2(m1, e1, m2, e2, ks)`: IS chain degree lower bound
- `_degree_feasible_row(m0, k1, ...)`: vectorised feasibility filter

---

## 8. C Extension

**File:** `src/manifold_index/core/_c_kernel/tet_index.c` (~410 lines)

**Three exported functions:**

1. `tet_index_series(m, e, qq_order) → dict`
   - Full tetrahedron index computation (~12× faster than Python)
   - Same formula as `_tet_index_series_python`

2. `tet_degree_x2(m, e) → int`
   - Degree lower bound, pure integer
   - Same as `_tet_degree_x2` in Python

3. `poly_convolve(prod, s, budget) → dict`
   - Sparse polynomial convolution with cutoff
   - `dict[int,int] × dict[int,int] → dict[int,int]` truncated at budget

**Build:** Compiled at `pip install` time. Falls back silently to Python
if not available (`_HAS_C_KERNEL = False`).

**Detection pattern in index_3d.py:**
```python
try:
    from manifold_index.core._c_tet_index import (
        tet_index_series as _c_tet_index_series,
        poly_convolve as _c_poly_convolve,
    )
    _HAS_C_KERNEL = True
except ImportError:
    _HAS_C_KERNEL = False
```

---

## 9. GUI Architecture

### 9.1 Overview

PySide6 application. All long computations run on background QThreads.

### 9.2 Entry Points

- CLI: `manifold-index` script → `manifold_index.app:main`
- Code: `from manifold_index.app import main; main()`
- Frozen: PyInstaller bundle → `launcher.py`

### 9.3 MainWindow (`window.py`)

- `QMainWindow` with `QTabWidget`
- Tab 1 "Calculator": `QSplitter` with 3 panels (Manifold | Filling | Export)
- Tab 2 "Kernel Builder": `KernelPanel`

**Key state:** `_nz_data`, `_refined_worker`, `_dehn_worker`

**Signal flow:**
```
User types manifold → Panel1.compute_requested(name, q)
→ MainWindow._start_compute() → RefinedIndexWorker.start()
→ Worker.finished(results) → MainWindow._on_refined_finished()
→ stores _nz_data → Panel1.computation_finished()
→ Panel1.data_ready → Panels 2,3 unlock

User sets slopes → Panel2.fill_requested(payload)
→ MainWindow._start_dehn_filling() → DehnFillingWorker.start()
→ Worker.nc_found(cycles), filling_finished(results)
→ Panel2 displays; Panel3 ready to export
```

### 9.4 Workers (`workers.py`)

**`RefinedIndexWorker(QThread)`:**
- Evaluates I^ref on a 45^r grid: m ∈ {-2,-1,0,1,2}, e ∈ {-2,-3/2,…,2}
- Signals: `status(str)`, `progress(int,int)`, `finished(list)`, `error(str)`
- After grid: runs `run_weyl_checks()`, emits result

**`DehnFillingWorker(QThread)`:**
- Step 1: NC cycle search
- Step 2A (multi-cusp): `_run_multi_cusp()`
- Step 2B (single-cusp): `_run_single_cusp()`

**`KernelBuilderWorker(QThread)`:**
- Calls `precompute_filling_kernel()` with user-specified slopes

### 9.5 Panels

- `ManifoldPanel`: manifold name input, q-order spinner, KaTeX-rendered results
- `FillingPanel`: per-cusp slope controls, NC range, Dehn Fill button
- `ExportPanel`: format checkboxes (LaTeX, Report, Mathematica, JSON, Plain Text)
- `KernelPanel`: kernel precomputation UI
- `DataPanel`: data packs management

### 9.6 Formatters (`formatters.py`)

- `series_to_katex(result, num_hard, max_q_terms)` — refined index → KaTeX
- `format_nz_matrix(nz_data)` — HTML table of NZ matrix
- `format_panel1_html(...)`, `format_panel2_html(...)` — full panel HTML

### 9.7 KaTeX Bridge (`katex.py`)

Bridges Qt WebEngine to KaTeX JavaScript for math rendering.

---

## 10. Export Infrastructure

**Module:** `src/manifold_index/utils/exporters.py` (~1600 lines)

### Monomial Formatting (LaTeX)

- `_latex_q_factor(qq_pow)` → `"q"`, `"q^3"`, `"q^{7/2}"`
- `_latex_eta_factors_hard(key, num_hard)` → hard-edge η^{2W_a e_a} factors
- `_latex_eta_factors_cusp(key, num_hard, num_cusp_eta)` → cusp η factors
- `_latex_monomial(key, coeff, num_hard, num_cusp_eta)` → full term

### Monomial Formatting (Mathematica)

- `_math_q_factor`, `_math_eta_hard`, `_math_eta_cusp`, `_math_monomial`

### Writers

| Function | Output | Description |
|----------|--------|-------------|
| `write_latex(path, data)` | `.tex` | LaTeX series fragment |
| `write_full_report(path, data, dehn_data)` | `.tex` | Full LaTeX document |
| `write_mathematica(path, data)` | `.m` | Mathematica rules |
| `write_plain_text(path, data)` | `.txt` | ASCII series |
| `write_json(path, data, dehn_data)` | `.json` | Structured JSON |
| `clipboard_latex(data)` | clipboard | LaTeX series |
| `clipboard_plain_text(data)` | clipboard | Plain text |

### Helpers

- `_charge_label`, `_frac_tex`, `_int_or_frac_tex`, `_matrix_tex`
- `_row_label`, `_fmt_linear_combination`
- `_np_to_mathematica`, `_math_frac`
- `_tex_escape`, `_plain_series`
- `_append_single_cusp_filling`, `_append_multi_cusp_filling`
- `_append_gluing_table`, `_edge_triplets_tex`, `_edge_equation_tex`

---

## 11. Implementation Phases

### Phase 0: Project Skeleton
**Files to create:**
- `pyproject.toml`
- `README.md`
- `src/manifold_index/__init__.py`
- `src/manifold_index/core/__init__.py`
- `src/manifold_index/utils/__init__.py`

**Goal:** Installable package with `pip install -e ".[dev]"`.

**Acceptance:** `python -c "import manifold_index"` works.

---

### Phase 1: Manifold Loading
**Files:** `core/manifold.py`, `tests/conftest.py`, `tests/test_manifold.py`

**Test criteria:**
- `load_manifold("m004")` → n=2, r=1, gluing_matrix shape (4, 6)
- `load_manifold("m003")` → n=2, r=1
- `load_manifold("v0901")` → n=7, r=1
- Properties (edge_equations, cusp_equations) return correct slices

---

### Phase 2: Gluing Equation Reduction
**Files:** `core/gluing_equations.py`, `tests/test_gluing_equations.py`

**Test criteria:**
- edge_coeffs shape = (n, 2n)
- rank(edge_coeffs[independent]) = n-r
- symplectic_matrix is antisymmetric with correct structure
- edge_consts: internal edges have const values consistent with sum=2

---

### Phase 3: Phase Space Basis (Easy Edges)
**Files:** `core/phase_space.py`, `tests/test_phase_space.py`

**Test criteria:**
- m004: finds easy edges, basis_edges length = n-r = 1
- All easy edges satisfy `_is_easy()` predicate
- basis_edges are linearly independent (rank check)

---

### Phase 4: Neumann-Zagier Matrix
**Files:** `core/neumann_zagier.py`, `tests/test_neumann_zagier.py`

**Test criteria:**
- g_NZ is symplectic: g_NZ · Ω · g_NZ^T = Ω (to float tolerance)
- g_NZ_inv() · g_NZ = I (exact with Fraction)
- g_NZ_inv_scaled() returns (S, int64 array) consistent with g_NZ_inv()
- m004: 4×4 matrix, num_hard=1, num_easy=0
- v0901: 14×14 matrix, non-unit Smith factors (S > 2)

---

### Phase 5: Basis Selection
**Files:** `core/basis_selection.py`, `tests/test_basis_selection.py`

**Test criteria:**
- default_meridian_choice → m=1, e=0
- default_longitude_choice → m=0, e=1/2
- apply_basis_changes roundtrip preserves symplecticity

---

### Phase 6: 3D Index
**Files:** `core/index_3d.py`, `tests/test_index_3d.py`

**Test criteria:**
- `tet_degree(1, 0) == Fraction(3, 2)`
- `tet_degree(m, e) >= 0` for all m, e in [-3, 3]
- `enumerate_summation_terms(nz_m004, [0], [0], 10)` returns >1 terms
- All terms have "phase_exp" and "tet_args" keys

---

### Phase 7: Refined Index
**Files:** `core/refined_index.py`, `tests/test_refined_index.py`

**Test criteria:**
- `project_to_3d_index({(2,2):1, (2,0):1, (2,-2):1}) == {2: 3}`
- `project_to_3d_index({(4,2):1, (4,-2):-1}) == {}`
- For m004 at (m,e)=(0,0): `project(refined) == 3D index`

---

### Phase 8: Weyl Checks
**Files:** `core/weyl_check.py`, `tests/test_weyl_check.py`

**Test criteria:**
- ABVectors extracted from m004 have a ∈ ℤ, 2b ∈ ℤ
- Weyl symmetry check passes for m004
- strip_weyl_monomial produces η ↔ η^{-1} symmetric result

---

### Phase 9: Dehn Filling
**Files:** `core/dehn_filling.py`, `tests/test_dehn_filling.py`

**Test criteria:**
- `find_rs(3, 2)` → R·2 - 3·S = 1
- `_particular_solution(3, 2, 2)` → 3·m₀ + 4·e₀ = 2
- `_apply_kernel(KernelTerm(0, 0, 0, 0), series)` = series (identity)

---

### Phase 10: Refined Dehn Filling
**Files:** `core/refined_dehn_filling.py`, `tests/test_refined_dehn_filling.py`

**Test criteria:**
- HJ-CF recovery: for (P,Q) ∈ {(1,2),(5,2),(3,4),(7,5)}, reconstruct P/Q
- `_is_kernel_frac(0,0,0,0, qq=20, eta=10)`: sum at qq=0 equals 1
- L=1 case: matches unrefined filling (m003, slope 5/1, q_order=10)

---

### Phase 11: Kernel Cache
**Files:** `core/kernel_cache.py`

**Test criteria:**
- save/load roundtrip preserves kernel table
- `apply_precomputed_kernel` matches direct computation

---

### Phase 12: C Extension
**Files:** `core/_c_kernel/tet_index.c`, build configuration in `pyproject.toml`

**Test criteria:**
- C `tet_index_series` matches Python for all (m,e) in test range
- C `poly_convolve` matches Python convolution
- C `tet_degree_x2` matches Python

---

### Phase 13: Export Infrastructure
**Files:** `utils/exporters.py`, `tests/test_exporters.py`

**Test criteria:**
- LaTeX output is valid LaTeX
- Mathematica output parses correctly
- JSON roundtrip preserves data

---

### Phase 14: GUI
**Files:** entire `app/` directory

**Test criteria:**
- Window opens without crash
- Computation pipeline runs end-to-end
- Export buttons produce files

---

### Phase 15: Build & Packaging
**Files:** `build_app.sh`, `ManifoldIndex.spec`, `launcher.py`

---

## 12. Testing Strategy & Known-Good Values

### 12.1 Test Manifolds

| Manifold | n (tet) | r (cusps) | num_hard | Special |
|:---------|:--------|:----------|:---------|:--------|
| m004 | 2 | 1 | 1 | Figure-eight knot; canonical test case |
| m003 | 2 | 1 | 0 | All easy edges; no η variables |
| v0901 | 7 | 1 | ? | Non-unit Smith factors in g^{-1} (S > 2) |

### 12.2 Shared Fixtures (conftest.py)

```python
@pytest.fixture(scope="session")
def nz_m004():
    data = load_manifold("m004")
    easy = find_easy_edges(data)
    return build_neumann_zagier(data, easy)

# Similarly for nz_m003, nz_v0901
```

All fixtures require `snappy` — skip if not installed.

### 12.3 Known Values

- `tet_degree(1, 0) == 3/2`
- `find_rs(3, 2)`: R·2 - 3·S = 1
- `_particular_solution(3, 2, 2)`: 3·m₀ + 4·e₀ = 2
- `project_to_3d_index({(2,2):1,(2,0):1,(2,-2):1}) == {2:3}`
- `project_to_3d_index({(4,2):1,(4,-2):-1}) == {}`
- HJ-CF(1,2) recovers 1/2; HJ-CF(5,2) recovers 5/2; etc.
- IS kernel at (0,0,0,0): sum at qq=0 is 1
- L=1 refined filling matches unrefined (m003, slope 5/1)

---

## 13. Performance-Critical Design Decisions

### 13.1 Doubled Exponents
All η exponents stored as `2×exp` (int) to avoid Fraction in hot loops.

### 13.2 Scaled Inverse
`g_NZ_inv_scaled()` returns `(S, int64)` — avoids Fraction in integrality checks.

### 13.3 x2 Degree Arithmetic
`_tet_degree_x2(m,e)` returns `2×δ(m,e)` as int — no float in feasibility filter.

### 13.4 x2 IS Kernel
`_is_kernel` returns `2·I_S` as int — LCD = 2^ℓ applied once at end.

### 13.5 Dynamic Budget Shrinking
Each tet-index call in the product gets `cutoff = budget - prod_min_pow`.

### 13.6 C Extension
Tet-index and poly-convolve ~12× faster in C. Silent Python fallback.

### 13.7 Batch Refined Index
Pre-compute enumeration state once, reuse for all (m,e) evaluations.

### 13.8 Vectorised Degree Feasibility
`_degree_feasible_row` in kernel_cache uses numpy int32 broadcasting.

### 13.9 Dense Int64 Kernel Application
KernelTable stores dense numpy arrays for fast convolution.

### 13.10 Process Pool Parallelism
Kernel precomputation and application use ProcessPoolExecutor.

---

## 14. Caching Architecture

| Cache | Location | Key | Eviction |
|:------|:---------|:----|:---------|
| `_tet_cache` | index_3d.py | `(m, e, qq_order)` | `clear_tet_cache()` |
| `_tet_arr_cache` | refined_dehn_filling.py | `(m, e, qq_order)` | `_clear_tet_arr_cache()` |
| `_enum_state_cache` | index_3d.py | NZ content bytes | `clear_enum_state_cache()` |
| `_iref_cache` | refined_dehn_filling.py | NZ content + (m,e,qq) | `clear_filling_caches()` |
| `_etilde_is` | refined_dehn_filling.py (LRU) | `(m1,e1,m2,e2,qq,eta)` | `clear_filling_caches()` |
| `_is_kernel` | refined_dehn_filling.py (LRU) | `(m1,e1,m2,e2,qq)` | `clear_filling_caches()` |
| `_kernel_mem_cache` | kernel_cache.py | `(P,Q,qq,dir)` | `clear_kernel_cache()` |
| Disk kernels | kernel_cache.py | `P_Q_qq` filename | Manual deletion |
| `g_NZ_inv()`, `g_NZ_inv_scaled()` | NZ instance | — | GC |
| `KernelTable._fast_grouped` | KT instance | — | GC |

**CRITICAL: Content-based cache keys.**
```python
def _nz_content_key(nz_data):
    return (nz_data.g_NZ.data.tobytes(), nz_data.nu_x.data.tobytes(), nz_data.nu_p.data.tobytes())
```
NEVER use `id(nz_data)` — Python GC reuses addresses, causing wrong results.

---

## 15. Build and Packaging

### 15.1 pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68", "setuptools-scm"]
build-backend = "setuptools.build_meta"

[project]
name = "manifold-index"
version = "0.4.0"
requires-python = ">=3.10"
dependencies = ["snappy", "numpy", "scipy"]

[project.optional-dependencies]
gui = ["PySide6"]
dev = ["pytest", "pytest-cov", "pytest-timeout", "ruff", "mypy"]

[project.scripts]
manifold-index = "manifold_index.app:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
manifold_index = [
    "data/kernel_cache/*.pkl.gz",
    "data/data_packs.json",
]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.mypy]
python_version = "3.10"
strict = false
ignore_missing_imports = true
```

### 15.2 C Extension Build

The C extension `_c_kernel/tet_index.c` is compiled at pip install time.
Package data includes `data/kernel_cache/*.pkl.gz`.

### 15.3 PyInstaller (macOS)

- `build_app.sh` + `ManifoldIndex.spec`
- `launcher.py`: calls `multiprocessing.freeze_support()` before GUI
- Runtime hooks for SnaPy/multiprocessing

---

## 16. Gotchas & Lessons Learned

1. **Block vs interleaved ordering:** g_NZ uses block ordering for the
   symplectic inverse identity. The gluing matrix uses interleaved.
   `_interleaved_to_block()` converts.

2. **Content-based cache keys:** `id(nz_data)` is recycled by Python GC.
   This caused wrong results when multiple basis-changed NZ objects
   were created in a loop. Always use `.tobytes()` fingerprinting.

3. **Doubled eta exponents:** Mandatory for performance. Half-integer
   e_int → half-integer η exp → storing 2× as int avoids Fraction.

4. **x2 IS kernel scaling:** The ½ prefactor in I_S is absorbed into
   integer arithmetic. LCD = 2^ℓ applied once at output.

5. **Diamond truncation:** IS chain boundary artifacts removed by
   qq + |cusp_eta| ≤ qq_order rule.

6. **is_stably_zero() buffer:** Non-closable detection ignores top
   buffer powers of filled series to avoid truncation artifacts.

7. **Fraction vs float:** All exact arithmetic uses `fractions.Fraction`.
   `g_NZ` entries are exact rationals stored as float64 for numpy.
   The inverse `g_NZ_inv()` returns Fraction.

8. **Multi-cusp limit:** >2 cusps raises NotImplementedError.
   Extending requires a tree of intermediate results.

9. **C extension fallback:** Always provide pure-Python implementations.
   The C extension is optional for performance only.

10. **Gamma row isotropic correction:** The initial integer right-inverse
    may not produce isotropic Γ rows. `_make_isotropic` applies the
    correction iteratively.

11. **v0901 (non-unit Smith factors):** Some manifolds have g_NZ_inv
    entries with denominators > 2. The LCD S must be computed dynamically,
    not hardcoded to 2. This was a v0.3.5 bugfix.

12. **Kernel parity detection:** Some slopes have non-zero entries only
    for even m or only for odd m. The probe phase detects this to halve
    computation time.

13. **n vs r swapped from paper:** The code uses n = num_tetrahedra and
    r = num_cusps, which is the OPPOSITE of the paper's convention.
    This is deliberate and consistent throughout the codebase.

14. **SnaPy gluing_equations("rect"):** Returns n+2r rows of 3n+1 values.
    The +1 is the constant term (always 2 for edges, 0 for cusps).
    We extract only the first 3n columns as the integer coefficient matrix.

15. **Phase exponent integrality:** The phase m_full · ν_p - e_full · ν_x
    is always integer by the symplectic structure. If it's not, there's
    a bug in the NZ construction.

---

*End of Blueprint — Ready for Phase-by-Phase Implementation*
