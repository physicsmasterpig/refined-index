# Architecture -- manifold-index v0.3.0

> Ground-truth reference for the codebase.  Read this before changing code.

---

## 1  Directory Layout

```
src/manifold_index/
  __init__.py
  core/                 # Math pipeline (pure Python, no GUI deps)
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
  utils/                # Placeholder helpers (math_utils, io_utils)
  data/                 # Static data / Mathematica notebooks
  mathematica/          # Reference .nb files (not used at runtime)
  app/                  # PySide6 GUI
    __init__.py         # entry point main()
    __main__.py         # python -m manifold_index.app
    window.py           # MainWindow
    workers.py          # QThread workers
    formatters.py       # LaTeX / KaTeX formatters
    katex.py            # KaTeX rendering bridge
    style.py            # CSS stylesheet
    panels/
      manifold_panel.py   # Step 1: manifold input, info display
      filling_panel.py    # Step 2: Dehn filling controls and results
      export_panel.py     # Export to LaTeX / clipboard
data/
  cache/                # Per-manifold Mathematica .mx caches
  kernel_cache/         # Pre-computed kernel tables (pkl.gz)
tests/                  # pytest suite, 21 tests (~0.44 s)
```

---

## 2  Module Dependency Graph

Arrows mean "imports from".  No circular dependencies exist.

```
manifold  -->  gluing_equations  -->  phase_space
                                          |
                                          v
                                     neumann_zagier  -->  basis_selection
                                          |
                      +-------------------+
                      v                   v
                  index_3d  <------  refined_index
                      |
           +----------+
           v          v
      dehn_filling  weyl_check  <--  refined_index
           |
           v
   refined_dehn_filling  -->  kernel_cache
```

- `app/workers.py` imports from every `core/` module.
- `app/formatters.py` imports data classes from `core/`.
- `utils/` is currently placeholder (no imports from `core/`).

---

## 3  Notation and Symbol Map

The paper and the code use different letters.  This table is definitive.

| Paper symbol | Code variable | Meaning |
|:-------------|:--------------|:--------|
| r | `n` | Number of tetrahedra |
| n | `r` | Number of cusps |
| Z_i, Z'_i, Z''_i | columns of `gluing_matrix` | Shape parameters (triples per tet) |
| q | -- | Formal variable |
| q^{1/2} | `qq` | Half-power; `qq_order` = max power of qq |
| eta_a | `eta` | Fugacity for hard edge a |
| M_i, L_i | meridian, longitude | Cusp basis cycles |
| Lambda_i = L_i/2 | `e_ext[i]` | Cusp momentum (half-longitude) |
| P, Q | `P`, `Q` | Slope of filling cycle P*M + Q*L |
| g_NZ | `nz_data.g_NZ` | Symplectic (2n x 2n) Neumann-Zagier matrix |

### Key numeric conventions

| Convention | Where it matters |
|:-----------|:-----------------|
| `q_order_half` / `qq_order` = power of q^{1/2} | Every function that takes a truncation order |
| `RefinedIndexResult` key = `(qq_pow, 2*eta_0, 2*eta_1, ...)` | All eta exponents stored doubled as int |
| `g_NZ_inv_x2()` returns 2 * g_NZ^{-1} as int64 | Avoids all Fraction arithmetic in hot paths |
| `_is_kernel()` returns 2 * I_S as int | Same reason; LCD tracked via 2^L accumulator |
| `e_ext` values can be half-integer (Fraction) | Cusp longitudes/2; internal edges may also be half-int |
| Content-based NZ fingerprint = `(g_NZ.tobytes(), nu_x.tobytes(), nu_p.tobytes())` | Cache keys; NEVER use id(nz_data) |

---

## 4  Core Modules -- Detailed Reference

### 4.1  manifold.py (158 lines)

**Purpose:** Load a manifold from SnaPy and extract its gluing equations.

**Key types:**
- `ManifoldData` -- frozen dataclass: `name`, `num_tetrahedra` (= n),
  `num_cusps` (= r), `gluing_matrix` shape `(n + 2r, 3n)`, `raw` SnaPy object.

**Gluing matrix layout:**
- Rows 0 ... n-1: edge equations (n rows, one per independent edge).
- Rows n ... n+2r-1: cusp equations, interleaved mu_0, lambda_0, mu_1, lambda_1, ...
- Columns grouped in triples: Z_1, Z_1', Z_1'', Z_2, Z_2', Z_2'', ...

**Entry point:** `load_manifold(name: str) -> ManifoldData`.
Validates shapes with assertions.

---

### 4.2  gluing_equations.py (220 lines)

**Purpose:** Reduce 3n shape variables to 2n by substituting
Z_i' = 1 - Z_i - Z_i'', then find the independent edge basis.

**Key types:**
- `ReducedGluingData` -- n, r, edge_coeffs (n,2n), edge_consts (n,),
  cusp_coeffs (2r,2n), cusp_consts (2r,), independent_edge_indices,
  symplectic_matrix (2n,2n).

**Symplectic matrix Omega:**
- Interleaved ordering: Omega[2i, 2i+1] = +1, Omega[2i+1, 2i] = -1.
- Rows/cols indexed as (Z_1, Z_1'', Z_2, Z_2'', ...).

**Independent edges:** Column-pivoted QR factorization of edge_coeffs;
rank = n - r.  Returns indices of the n-r independent edge equations.

**Entry point:** `reduce_gluing_equations(data: ManifoldData) -> ReducedGluingData`.

---

### 4.3  phase_space.py (457 lines)

**Purpose:** Classify internal edges as "easy" or "hard" and construct
the phase-space basis.

**Key types:**
- `EasyEdgeResult` -- all_easy (list of 3n-vectors),
  independent_easy_indices, hard_padding, n, r.

**Easy edge definition:** An edge is easy if, in the 3n-dimensional
solution vector, at most one of {Z, Z', Z''} is nonzero per tetrahedron.

**Algorithm B (pattern-first search):**
1. Enumerate candidate patterns in {Z, Z', Z'', OFF}^n.
2. For each pattern, set up the (potentially underdetermined) linear system.
3. Solve exactly using fractions.Fraction (no floating-point).
4. Accept solutions that satisfy the "at most one nonzero per tet" rule.

**basis_edges property:** [independent easy edges ... | hard padding ...],
total length = n - r.

**Entry point:** `find_easy_edges(data: ReducedGluingData, tol) -> EasyEdgeResult`.

---

### 4.4  neumann_zagier.py (772 lines)

**Purpose:** Build the symplectic Neumann-Zagier matrix g_NZ in Sp(2n, Q)
and affine shift vectors (nu_x, nu_p).

**Key types:**
- `NeumannZagierData` -- g_NZ (2n,2n) float64, nu_x (n,),
  nu_p (n,), n, r, num_hard, num_easy.

**g_NZ row ordering (rows of the 2n x 2n matrix):**

| Row range | Content |
|:----------|:--------|
| 0 ... r-1 | Meridians (cusp position vars) |
| r ... r+num_hard-1 | Hard internal edges |
| r+num_hard ... n-1 | Easy internal edges |
| n ... n+r-1 | Longitude / 2 (cusp momentum) |
| n+r ... 2n-1 | Gamma vectors (isotropic complement) |

**g_NZ column ordering (block form, NOT interleaved):**
- Columns 0 ... n-1: Z_1, Z_2, ..., Z_n  (position block)
- Columns n ... 2n-1: Z_1'', Z_2'', ..., Z_n''  (momentum block)

> WARNING: The gluing matrix uses interleaved ordering (Z_1,Z_1'',Z_2,Z_2'',...)
> but g_NZ uses block ordering.  The internal helper
> `_interleaved_to_block()` performs the conversion.

**Affine shifts:**
- nu_x[i]: edge equation constant (= 2 for edges, 0 for cusps and Gamma).
- nu_p[i]: longitude constant / 2 for cusps; 0 for Gamma rows.

**Cached methods on NeumannZagierData:**
- `g_NZ_inv()` -> Fraction array (exact via symplectic identity
  g^{-1} = [[D^T, -B^T], [-C^T, A^T]]).
- `g_NZ_inv_x2()` -> int64 array = 2 * g_NZ^{-1} (all entries guaranteed
  to have denominator dividing 2).

**Gamma construction:** Integer right-inverse via Euclidean column reduction,
followed by isotropic correction (so the full matrix is symplectic).

**Cusp basis changes:**
- `apply_cusp_basis_change(nz, cusp_idx, P, Q)` -- requires P odd.
  Uses Bezout coefficients with P*b - 2Q*a = 1.
- `apply_general_cusp_basis_change(nz, cusp_idx, a, b, c, d)` --
  general SL(2,Z); allows half-integer entries in the result.

**Entry point:** `build_neumann_zagier(data, easy_result) -> NeumannZagierData`.

---

### 4.5  basis_selection.py (375 lines)

**Purpose:** Per-cusp cycle selection for refined index computation.

**Key types:**
- `CycleChoice(cusp_idx, P, Q)` -- with derived m = P, e = Fraction(Q, 2),
  slope_str.
- `BasisSelection(choices)` -- validates ordering, exposes m_ext, e_ext.

**Entry points:**
- `make_basis_selection(nz_data, cycle_results, choices, default="M", strict=False)`.
- `apply_basis_changes(nz_data, basis) -> NeumannZagierData` -- applies cusp
  basis changes for cusps with odd P.

---

### 4.6  index_3d.py (1015 lines)

**Purpose:** Compute the 3D index I(m_vec, e_vec) as a q^{1/2}-series.

**Key types:**
- `Index3DResult` -- coeffs (list[int]), min_power, q_order_half,
  m_ext, e_ext, n_terms.

**Tetrahedron index `_tet_index_series(m, e, qq_order)`:**
- Returns dict[int, int] mapping qq-power to coefficient.
- C extension (_c_tet_index.cpython-*.so) preferred; pure-Python fallback.
- Module-level memoization: _tet_cache[(m, e, qq_order)].
- Garoufalidis-Kim formula: MIt(m,e) = (-qq)^m * I_t(-m-e, m) when m+e >= 0.

**Degree formula `tet_degree(m, e)`:**
- delta(m,e) = 1/2 * (m_+ * (m+e)_+ + (-m)_+ * e_+ + (-e)_+ * (-e-m)_+) + max{0, m, -e}
- `_tet_degree_x2(m, e)` -> int = 2 * tet_degree (avoids Fraction in hot loops).

**Summation enumeration (the heart of the algorithm):**

`enumerate_summation_terms(nz_data, m_ext, e_ext, q_order_half)` returns
list of dicts, each containing:
```
  "e_int"      : list[str]          -- e_int as "p/q" strings (info only)
  "phase_exp"  : int                -- integer exponent of (-q^{1/2})
  "tet_args"   : list[(int, int)]   -- (m_a, e_a) for a = 0 ... n-1
  "min_degree" : float              -- sum of tet_degree values
```

**Algorithm outline:**
1. Pre-compute `_EnumerationState` (cached per NZ content):
   g_inv_x2, internal-edge columns, valid half-integer patterns delta in {0,1}^{n-r},
   per-pattern contributions (delta_contrib_x4, delta_phase_x2), cusp columns.
2. For each valid delta pattern, compute base_args_x4 from (m_ext, e_ext, delta).
   Skip if not divisible by 4 (integrality check).
3. Find integer offsets e0 via `_exact_e0_candidates()`:
   convex piecewise-quadratic degree bound -> exact sublevel-set enumeration
   via per-axis projection bounds + bounding-box scan.
4. For each valid e0, compute tet_args = g_NZ^{-1} * kappa (where kappa encodes
   m_ext, e_ext, e_int), phase_exp, and min_degree.

**`compute_index_3d_python(nz_data, m_ext, e_ext, q_order_half)`:**
- Calls enumerate_summation_terms, then for each term multiplies
  n tetrahedron index series together (dict convolution, with C extension
  for speed).
- Phase factor: (-qq)^{phase_exp}.
- Budget optimization: progressive tightening of per-tet cutoffs.
- Accepts `_precomputed_terms` to skip re-enumeration (used by Dehn filling).

---

### 4.7  refined_index.py (440 lines)

**Purpose:** Refined index with fugacity variables eta_a for hard edges.

**Key type:**
```
RefinedIndexResult = dict[tuple[int, ...], int]
  key = (q_half_power, 2*eta_0_exp, 2*eta_1_exp, ..., 2*eta_{k-1}_exp)
```
All eta exponents are stored doubled as plain int to avoid Fraction.

**`compute_refined_index(nz_data, m_ext, e_ext, q_order_half)`:**
- Like compute_index_3d_python but tracks eta contributions from each
  summation term's e_int values for hard edges.
- eta_a exponent = 2 * e_int[a] for hard edge a (stored in key).

**`compute_refined_index_batch(nz_data, entries, q_order_half)`:**
- Shares _EnumerationState across multiple (m_ext, e_ext) evaluations.
- Returns list[RefinedIndexResult] in the same order.

**Utilities:**
- `project_to_3d_index(refined) -> dict[int, int]` -- sets all eta = 1.
- `format_refined_index(refined, num_hard)` -- human-readable string.
- `format_multi_point_index(entries, ...)` -- multi-line I(charges) = series.

---

### 4.8  weyl_check.py (917 lines)

**Purpose:** Verify Weyl symmetry prerequisites before Dehn filling.

Three conditions checked:
1. Non-closability (handled by dehn_filling.py).
2. Weyl symmetry: f(m,e) = eta^{b*m + a*e} * I(m,e) is eta <-> eta^{-1} symmetric.
3. Adjoint character: q^1 coefficient = eta + 1 + eta^{-1}.

**Key types:**
- `ABVectors` -- a: list[Fraction], b: list[Fraction], per hard edge.
  - a[j] in Z and 2*b[j] in Z required for Dehn filling compatibility.
  - `edge_compatible` property; `make_filling_compatible()` zeros incompatible edges.

**`compute_ab_vectors(entries, num_hard)`:**
- Extracts (a, b) from conjugate-charge pairs using eta-centre shifts:
  b[j] = -[centre(+m,0) - centre(-m,0)] / (2m),
  a[j] = -[centre(0,+e) - centre(0,-e)] / (2e).

**`compute_ab_vectors_for_cusp(nz_data, cusp_idx, q_order_half)`:**
- Single-cusp version: evaluates I^ref at a 5x5 grid, extracts per-cusp
  Weyl column.

**`run_weyl_checks(entries, num_hard)` -> WeylCheckResult** with all three checks.

**`strip_weyl_monomial(result, m_ext, e_ext, ab, num_hard)`:**
- Multiplies I by eta^{a*e + b*m} and returns the Weyl-manifest series.

---

### 4.9  dehn_filling.py (844 lines)

**Purpose:** Ordinary (unrefined) Dehn filling kernel and non-closable cycle search.

**Kernel formula (slope P/Q, R*Q - P*S = 1):**
```
K(P,Q; m,e) = 1/2 * (-1)^{Rm+2Se} *
  [ delta_{Pm+2Qe,0} * (q^{(Rm+2Se)/2} + q^{-(Rm+2Se)/2})
    - delta_{Pm+2Qe,-2}
    - delta_{Pm+2Qe,2} ]
```

**Key types:**
- `KernelTerm(m, e, c, phase, multiplicity)` -- one (m,e) summand.
  c in {0, 2} (c = -2 handled by multiplicity = 2 on c = 2).
- `FilledIndexResult` -- series: QSeries, where
  QSeries = dict[int, Fraction] (key k -> coeff of q^{k/2}).
- `NonClosableCycle`, `NonClosableCycleResult`.

**Antipodal symmetries exploited:**
- c = 0: t and -t contribute identically -> multiplicity = 2 for t > 0.
- c = +/-2: (m,e) and (-m,-e) are antipodal -> skip c = -2, double c = 2.
- (P,Q) <-> (-P,-Q): I_{P/Q} = I_{-P/-Q} -> canonical half search.

**Summation-cache optimization:** enumerate_kernel_terms pre-computes
enumerate_summation_terms during the degree filter and stores results
in `_summation_cache` so compute_index_3d_python can reuse them (one
enum call per kernel term, not two).

**`compute_filled_index(nz_data, cusp_idx, P, Q, ...) -> FilledIndexResult`.**

**`find_non_closable_cycles(nz_data, cusp_idx, ...) -> NonClosableCycleResult`:**
- Scans slopes in p_range x q_range, fills each, checks is_stably_zero().
- Stability buffer: ignores high-power boundary artifacts near cutoff.

---

### 4.10  refined_dehn_filling.py (2498 lines)

**Purpose:** Refined Dehn filling via Hirzebruch-Jung continued fraction
and I_S convolution chain.

**Hirzebruch-Jung CF:** P/Q = k_1 - 1/(k_2 - 1/(... - 1/k_L)).
- Q = 0, P = +/-1 -> [0, 0].
- |Q| = 1 -> L = 1, unrefined K suffices.

**I_S kernel (symplectic kernel, eq. A.5):**
- `_etilde_is(m1,e1,m2,e2,qq_order,eta_order)` -- inner function, 4 tet-index
  products summed over (t, eta) double loop.  LRU-cached.  Returns
  dict[(qq,eta) -> int].
- `_is_kernel(m1,e1,m2,e2,qq_order,eta_order)` -- returns 2 * I_S as int.
  LRU-cached.  The x2 scaling absorbs the 1/2 prefactor.

**Key type aliases:**
- `QEtaSeries = dict[(qq_power, eta_exp), Fraction]` -- single cusp eta.
- `MultiEtaSeries = dict[tuple[int,...], Fraction]` -- multi-dimensional eta key:
  (qq, 2*eta_0, ..., 2*eta_{H-1}, cusp_eta_0, cusp_eta_1, ...).

**FilledRefinedResult:**
- series: MultiEtaSeries, num_hard, has_cusp_eta, num_cusp_eta.
- L = 1: key = (qq, 2*eta_0, ...) -- no cusp eta.
- L >= 2: key = (qq, 2*eta_0, ..., cusp_eta) -- one cusp eta per filling step.

**Main function `compute_filled_refined_index(...)`:**
1. Compute HJ-CF -> L.
2. L = 1: K(k_1,1) factor * I^ref(m,e) -- no IS chain.
3. L >= 2, cached kernel exists: `apply_precomputed_kernel()` (sub-second).
4. L >= 2, auto_precompute=True: compute kernel -> save -> apply.
5. L >= 2, fallback: grid scan + L-1 IS convolution steps + final K-factor.

**int-mode IS chain (L >= 2 performance):**
- State values are int, not Fraction.
- Each _is_kernel call returns x2 int -> LCD accumulates as 2^L.
- Conversion to Fraction happens once at the end: Fraction(v, 2^L).

**Diamond truncation:** qq + |cusp_eta| <= qq_order -- removes boundary
artifacts from finite-series truncation in the IS chain.

**Weyl shift:** `_apply_weyl_shift(refined, m_ext, e_ext, a, b, ...)` multiplies
each I^ref(m,e) by eta^{a*e_I + b*m_I} before the kernel acts on it.

**Multi-cusp filling (compute_multi_cusp_filled_refined_index):**
- Currently supports up to 2 cusps (raises NotImplementedError for >2).
- Step 1: `_batched_first_filling()` -- fills cusp 0 with spectator dims.
- Step 2: `_apply_filling_kernel_to_intermediate()` -- fills cusp 1 on
  intermediate results.
- Generalized diamond truncation: qq + sum|cusp_eta_i| <= qq_order.

---

### 4.11  kernel_cache.py (~1000 lines)

**Purpose:** Pre-compute, save, and load manifold-independent Dehn filling
kernel tables for L >= 2 slopes.

**Key type:**
- `KernelTable` -- P, Q, qq_order, qq_internal, eta_order, hj_ks,
  table: dict[(m,Fraction) -> QEtaSeries], m_scan, e_scan.
- Fast representation: _fast_lcd: int, _fast_grouped: dict --
  dense int64 numpy arrays per (m,e,eta_cusp), built lazily.

**Storage:** data/kernel_cache/kernel_P{P}_Q{Q}_qq{qq}.pkl.gz.
Gzipped pickle.  Fast repr is built before saving so loads skip conversion.

**Loading with fallback:**  load_kernel_table(P,Q,qq) first tries exact
match, then falls back to the smallest stored kernel with stored_qq >= qq.
In-memory cache avoids repeated disk I/O.

**precompute_filling_kernel(P, Q, qq_order, ...) — four-phase algorithm (v3):**
1. **Parity auto-detection**: Probes m=0..3 to determine if only one
   m-parity produces non-zero entries (m_step=1 or 2).
2. **Probe-and-scale support prediction** (PROBE_QQ=8): Runs a cheap
   low-qq computation to discover the non-zero support shape, then scales
   per-m e-bounds for the target qq via center+half-width interpolation
   (WIDTH_MARGIN=1.4, MARGIN_ABS=8).  Eliminates ~80-90% of zero grid points.
3. **Compute entries (m >= 0 only)**: Exploits K(m,e)=K(-m,-e) symmetry.
   Row-based parallel dispatch via ProcessPoolExecutor (fork context)
   for maximal LRU-cache locality.
4. **Mirror symmetry**: Copies m>=0 entries to -m,-e.

**apply_precomputed_kernel(kernel, nz_data, cusp_idx, ...):**
- Numpy-accelerated convolution: kernel arrays are dense int64, I^ref is
  grouped by eta-pattern -> np.convolve or batched matrix spread.
- _BATCH_THRESH = 4: uses batched 2-D numpy accumulation when the number
  of eta_cusp entries per (m,e) point exceeds this threshold.
- Parallel I^ref computation: ProcessPoolExecutor with initializer for nz_data.

---

## 5  GUI Architecture

### 5.1  Entry Point

pyproject.toml -> `manifold-index = "manifold_index.app:main"`.
app/__init__.py exports main() which calls launch_gui() in window.py.

### 5.2  Window and Panels

MainWindow(QMainWindow) in window.py:
- Contains ManifoldPanel, FillingPanel, ExportPanel.
- Signals/slots between panels and QThread workers.

ManifoldPanel -- text input for manifold name, compute button, info display.
FillingPanel -- NC cycle selection, slope controls, filling results.
ExportPanel -- LaTeX/clipboard export.

### 5.3  Workers

RefinedIndexWorker(QThread) -- runs the full pipeline:
load_manifold -> reduce -> find_easy -> build_NZ -> compute_refined_index_batch.

DehnFillingWorker(QThread) -- runs Dehn filling:
- _run_single_cusp() -- one cusp filling.
- _run_multi_cusp() -- multi-cusp sequential filling via
  compute_multi_cusp_filled_refined_index.

### 5.4  Formatters

formatters.py -- HTML + KaTeX strings for display:
- format_manifold_info, format_gluing_equations, format_edge_classification,
  format_nz_matrix, format_weyl_check, _series_to_katex.

---

## 6  Caching Strategy

| Cache | Scope | Key type | Eviction |
|:------|:------|:---------|:---------|
| _tet_cache | Module-level | (m, e, qq_order) | clear_tet_cache() |
| _enum_state_cache | Module-level | NZ content tuple (bytes) | clear_enum_state_cache() |
| _iref_cache | Module-level | NZ content + (m,e,qq) | clear_filling_caches() |
| _etilde_is.cache | LRU (unbounded) | (m1,e1,m2,e2,qq,eta) | clear_filling_caches() |
| _is_kernel.cache | LRU (unbounded) | same | clear_filling_caches() |
| _kernel_mem_cache | Module-level | (P,Q,qq,dir) | clear_kernel_cache() |
| g_NZ_inv(), g_NZ_inv_x2() | Instance | -- | Not evictable |
| KernelTable._fast_grouped | Instance | -- | GC |

> WARNING: Never use id(nz_data) as a cache key.  Python reuses memory
> addresses after GC, which caused wrong results in early versions.

---

## 7  Performance-Critical Paths

1. **Tetrahedron index series:** C extension _c_tet_index (compiled .so);
   Python fallback via _tet_index_series_python.  Memoized.

2. **Polynomial convolution:** C extension _c_poly_convolve; Python fallback
   (dict-based, truncated at budget).

3. **Summation enumeration:** _exact_e0_candidates uses _tet_degree_x2
   (pure-int, no Fraction) and per-axis projection bounds with coordinate
   descent.  Numpy vectorization for n >= 8.

4. **IS kernel chain (L >= 2):** int-mode (use_int=True) through all steps,
   converting to Fraction only at the final output.  LCD = 2^L.

5. **Kernel application:** Dense int64 numpy arrays; batched 2-D matrix
   spread for large eta_cusp counts.

---

## 8  Testing

21 tests in tests/, ~0.44 s total.  Run with `pytest`.

| File | Tests what |
|:-----|:-----------|
| test_manifold.py | ManifoldData loading, shape assertions |
| test_gluing_equations.py | Variable reduction, independent edge count |
| test_phase_space.py | Easy edge discovery, pattern consistency |
| test_neumann_zagier.py | g_NZ symplecticity, inverse correctness |
| test_basis_selection.py | Cycle choice, basis change roundtrip |
| test_index_3d.py | 3D index values against known results |
| test_refined_index.py | Refined index, eta-projection = 3D index |
| test_dehn_filling.py | Filling kernel, non-closable detection |

---

## 9  Build and Packaging

```
[build-system]
requires = ["setuptools>=68"]

[project]
name = "manifold-index"
version = "0.3.0"
requires-python = ">=3.10"
dependencies = ["snappy", "numpy", "scipy"]

[project.optional-dependencies]
gui = ["PySide6"]
dev = ["pytest", "pytest-cov", "pytest-timeout", "ruff", "mypy"]
```

C extensions are compiled from src/manifold_index/core/ C source files
at install time.  The .so is platform-specific and not committed.

---

## 10  Design Decisions and Gotchas

1. **Why block ordering for g_NZ?**  The symplectic identity
   g^{-1} = [[D^T,-B^T],[-C^T,A^T]] requires 2x2 block structure.
   Block ordering makes the n x n sub-blocks contiguous.

2. **Why doubled eta exponents?**  Half-integer e_int values produce
   half-integer eta exponents.  Storing 2x as int avoids Fraction
   in the innermost loops of refined_index and refined_dehn_filling.

3. **Why x2 scaling on _is_kernel?**  The I_S formula has a 1/2 prefactor.
   Returning 2*I_S as int avoids all Fraction arithmetic in the IS
   chain.  The accumulated denominator 2^L is applied once at output.

4. **Why content-based cache keys?**  Each NC cycle creates a fresh
   basis-changed NZ object.  id(nz) is recycled by Python GC, causing
   cross-cycle cache collisions.  Content fingerprinting (matrix bytes)
   is safe.

5. **Diamond truncation rule:**  The IS chain computes at qq_internal =
   qq_order + buffer, but high-|cusp_eta| terms are only partially
   resolved.  The diamond rule qq + |eta_cusp| <= qq_order removes
   exactly the unreliable boundary terms.

6. **is_stably_zero() buffer:**  Non-closable detection ignores the top
   buffer powers of the filled series, where incomplete cancellations
   from truncated kernel terms produce spurious nonzero coefficients.

7. **Fraction vs float:**  All exact arithmetic uses fractions.Fraction.
   Float (float64) appears only in g_NZ entries (which are exact
   rationals stored as float for numpy compatibility) and in tolerance
   comparisons.  The inverse g_NZ_inv() returns Fraction.

8. **Multi-cusp limit:**  compute_multi_cusp_filled_refined_index raises
   NotImplementedError for >2 cusps.  The batched spectator approach
   works for 2 cusps; extending to 3+ requires a tree of intermediate
   results.
