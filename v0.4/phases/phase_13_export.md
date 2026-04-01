# Phase 13 — Export Infrastructure

> **File:** `src/manifold_index/utils/exporters.py`
> **Depends on:** phases 4–10 (all result dataclasses)

---

## 0. Purpose

Converts computation results into human-readable and machine-readable
formats: LaTeX (full compilable report), JSON (structured data),
Mathematica (.m rules), clipboard (LaTeX & plain text).

---

## 1. Module Layout

```
utils/
  exporters.py      # All export logic
```

Single file, 9 logical sections.

---

## 2. Monomial Formatters (§1)

### 2.1 Key Layout

Keys follow the doubled-exponent convention from CONVENTIONS.md:

```
key = (qq, 2*e_hard_0, 2*e_hard_1, ..., cusp_eta_0, cusp_eta_1, ...)
       ^     ^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^
       |     hard-edge η exponents        cusp η exponents (if filled)
       doubled q-power
```

### 2.2 LaTeX Monomial Helpers

```python
_latex_q_factor(qq_pow: int) → str
    # "" if 0, "q" if 2, "q^{n}" if even, "q^{qq/2}" if odd

_latex_eta_factors_hard(key, num_hard) → str
    # \eta_a^{exp2/2} for each hard edge a

_latex_eta_factors_cusp(key, num_hard, num_cusp_eta) → str
    # \eta^{2*cusp_exp * V_ci} for cusp dimensions

_latex_monomial(key, coeff, num_hard, num_cusp_eta=0) → str
    # Combines coefficient + q + hard η + cusp η
```

### 2.3 Mathematica Monomial Helpers

Parallel set: `_math_q_factor`, `_math_eta_hard`, `_math_eta_cusp`,
`_math_monomial` — same logic, Mathematica syntax
(`q^n`, `eta[a]^p`, `etaCusp[ci]^exp`).

---

## 3. Series Formatters (§2)

```python
to_latex_series(result: dict, num_hard: int) → str
to_latex_filled_series(series, num_hard, num_cusp_eta=0, max_q_terms=None) → str
to_mathematica_series(result: dict, num_hard: int) → str
to_mathematica_filled_series(series, num_hard, num_cusp_eta=0) → str
```

All sort by key, skip zero coefficients, handle `+`/`−` sign joining.
`to_latex_filled_series` supports optional `max_q_terms` truncation
(appends `+ \cdots`).

---

## 4. LaTeX Utility Helpers (§3–4)

```python
_charge_label(m_ext, e_ext, latex=False) → str
    # "m=(0,1), e=(1/2,0)" or LaTeX equivalent

_frac_tex(v) → str           # Fraction → LaTeX \tfrac{n}{d}
_int_or_frac_tex(v) → str    # same, handles floats
_matrix_tex(mat, env="pmatrix") → str   # numpy array → LaTeX pmatrix
_row_label(i, n, r, num_hard) → str     # μ_i, H_j, E_k, λ_i/2, Γ_i
_fmt_linear_combination(terms) → str    # [(Fraction, var_str)] → LaTeX
_tex_escape(s) → str          # Escape _, &, %, #, {, }
```

---

## 5. Full LaTeX Report (§5)

### 5.1 `write_full_report`

```python
def write_full_report(
    path: Path | str,
    manifold_data: ManifoldData,
    easy_result: EasyEdgeResult,
    nz_data: NeumannZagierData,
    entries: list[tuple[m_ext, e_ext, result]],
    weyl_result: WeylCheckResult | None = None,
    dehn_results: list | None = None,
    q_order_half: int | None = None,
) → None
```

### 5.2 Report Sections (8 total)

| Section | Content |
|---------|---------|
| 1. Overview | Name, n, r, easy/hard counts, q-order, boundary curve ranges |
| 2. Triangulation | SnaPy gluing matrix as longtable of triplets (Z, Z', Z'') |
| 3. Edge Classification | Hard edges table + easy edges table with ✓/✗ independence |
| 4. Neumann–Zagier | g_NZ matrix with row labels, affine shifts ν_x, ν_p |
| 5. 3D Index Formula | Mathematical formula, explicit local charges per tetrahedron |
| 6. Refined Index Results | align* block of all I^ref(m,e) sectors |
| 7. Weyl Symmetry | a/b vectors, sector symmetry check, adjoint projection |
| 8. Dehn Filling | Per-slope: HJ-CF, basis change, physical Weyl, filled series |

### 5.3 Preamble

Custom LaTeX commands:
- `\Iref` → `I^{\mathrm{ref}}`
- `\Ifill` → `I^{\mathrm{ref,filled}}`
- `\IDelta` → `I_{\Delta}`
- `\checkmark` (green), `\crossmark` (red)
- `\passmark`, `\failmark`
- Colors: `pass` (#2ea043), `fail` (#cf222e), `warn` (#d4880a)

### 5.4 Dehn Filling Sub-Sections

```python
_append_single_cusp_filling(L, res: TransformedFillResult, nz, n, r, idx)
_append_multi_cusp_filling(L, res: MultiCuspFillResult, nz, n, r, idx)
```

Single-cusp: slope transform, HJ-CF, basis-changed g_NZ, physical Weyl
vectors, filled series per external-charge configuration.

Multi-cusp: per-cusp NC info + physical Weyl, chained g_NZ basis
changes, combined filled result.

### 5.5 Table Helpers

```python
_append_gluing_table(L, rows, n, prefix, row_range, labels=None)
_edge_triplets_tex(edge, n) → str
_edge_equation_tex(edge, n) → str
```

---

## 6. JSON Writer (§6)

```python
def write_json(
    path, manifold_data, easy_result, nz_data, entries,
    weyl_result=None, dehn_results=None, q_order_half=None,
) → None
```

Structure:
```json
{
  "manifold": "m003",
  "n": 2, "r": 1, "num_hard": 1, "num_easy": 1,
  "gluing_matrix": [[...]],
  "g_NZ": [[...]], "nu_x": [...], "nu_p": [...],
  "g_NZ_inv_scale": S, "g_NZ_inv_scaled": [[...]],
  "is_symplectic": true,
  "easy_edges": { "count": ..., "independent_count": ..., ... },
  "sectors": [{"m": [...], "e": [...], "coefficients": {...}}],
  "weyl": { "a": [...], "b": [...], "symmetric_sectors": {...}, ... },
  "dehn_filling": [...]
}
```

Uses `json.dumps(data, indent=2, default=str)`.

---

## 7. Mathematica Writer (§7)

```python
def write_mathematica(
    path, manifold_data, nz_data, entries,
    weyl_result=None, dehn_results=None, q_order_half=None,
) → None
```

Outputs `.m` file with:
- Manifold parameters as variables
- `gNZ`, `gNZInv`, `nuX`, `nuP` matrices
- `Iref["name", {m}, {e}] = ...` rules
- Weyl vectors
- `IrefFilled[...]` and `hjCF[...]` rules

Helpers:
```python
_np_to_mathematica(arr) → str    # numpy → "{...}" notation
_math_frac(v) → str              # Fraction → "n/d" or "n"
```

---

## 8. Clipboard Helpers (§8)

```python
clipboard_latex(manifold_name, entries, num_hard,
                dehn_results=None, include_dehn=False) → str
    # LaTeX align* block

clipboard_plain_text(manifold_name, entries, num_hard,
                     dehn_results=None, include_dehn=False) → str
    # Plain text with _plain_series helper

_plain_series(series, num_hard, num_cusp_eta=0) → str
    # "1 + q^1*eta_0 - 2*q^2"
```

---

## 9. Legacy Wrappers (§9)

```python
write_latex(path, manifold_name, nz, entries, ...)
    # Minimal LaTeX document (old API)

write_report(path, manifold_name, nz, easy_result, entries, ...)
    # Delegates to write_full_report via _Stub ManifoldData

write_plain_text(path, manifold_name, nz, entries, ...)
    # Delegates to clipboard_plain_text
```

These preserve backward compatibility for external callers.

---

## 10. Tests

### T13.1 — Monomial Formatting

```python
key = (4, 2, 0)  # q^2 * η_0
assert _latex_monomial(key, 1, 2) == r"q^{2} \, \eta_0"
assert _math_monomial(key, -1, 2) == "-q^2 eta[0]"
```

### T13.2 — Series Formatting

Build a small result dict, check `to_latex_series` produces valid LaTeX
with sorted terms and correct sign handling.

### T13.3 — Full Report Smoke Test

Given minimal ManifoldData / NeumannZagierData / EasyEdgeResult stubs,
`write_full_report` should produce a `.tex` file that contains all 8
`\section{...}` headers and is syntactically valid LaTeX.

### T13.4 — JSON Round-Trip

`write_json` → `json.loads(path.read_text())` → verify all expected
top-level keys present and types correct.

### T13.5 — Mathematica Writer

Check output contains `Iref[`, `gNZ`, `gNZInv`, proper `{...}` nesting.

### T13.6 — Clipboard Helpers

Verify `clipboard_latex` returns string containing `\begin{align*}`.

---

## 11. Acceptance Criteria

- [ ] All 4 writers produce valid output files
- [ ] LaTeX report compiles with pdflatex (manual check)
- [ ] JSON is valid JSON
- [ ] Mathematica syntax parseable by Wolfram kernel
- [ ] Legacy wrappers pass existing tests unchanged
