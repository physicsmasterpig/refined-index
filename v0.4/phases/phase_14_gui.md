# Phase 14 — GUI (PySide6)

> **Files:**
> - `src/manifold_index/app/__init__.py`
> - `src/manifold_index/app/__main__.py`
> - `src/manifold_index/app/window.py`
> - `src/manifold_index/app/workers.py`
> - `src/manifold_index/app/formatters.py`
> - `src/manifold_index/app/katex.py`
> - `src/manifold_index/app/style.py`
> - `src/manifold_index/app/panels/manifold_panel.py`
> - `src/manifold_index/app/panels/filling_panel.py`
> - `src/manifold_index/app/panels/export_panel.py`
> - `src/manifold_index/app/panels/kernel_panel.py`
> - `src/manifold_index/app/panels/data_panel.py`
>
> **Depends on:** all core phases (0–11), phase 13 (exporters)
> **Optional dependency:** PySide6

---

## 0. Purpose

Interactive desktop GUI for the manifold-index package.  Three tabs:
Calculator, Kernel Builder, Data Packs.  Renders mathematical output
with KaTeX in embedded QWebEngineView widgets.

---

## 1. Architecture Overview

```
app/
  __init__.py     # `def main()` entry point
  __main__.py     # `python -m manifold_index.app` → main()
  window.py       # MainWindow (QMainWindow, tabbed)
  workers.py      # QThread workers + result dataclasses
  formatters.py   # HTML/KaTeX formatting of results for display
  katex.py        # KaTeX HTML wrapper + QWebEngineView factory
  style.py        # APP_STYLESHEET (Qt CSS string)
  panels/
    __init__.py
    manifold_panel.py   # Tab 1: left panel
    filling_panel.py    # Tab 1: center panel
    export_panel.py     # Tab 1: right panel
    kernel_panel.py     # Tab 2
    data_panel.py       # Tab 3
```

---

## 2. Entry Points

### 2.1 `app/__init__.py`

```python
def main():
    from manifold_index.app.window import launch_gui
    launch_gui()
```

### 2.2 `app/__main__.py`

```python
from manifold_index.app import main
main()
```

### 2.3 `pyproject.toml` script entry

```toml
[project.scripts]
manifold-index = "manifold_index.app:main"
```

---

## 3. `window.py` — MainWindow

### 3.1 Class: `MainWindow(QMainWindow)`

- Title: "Refined 3D Index Calculator — v0.4.0"
- Minimum size: 1200 × 700, initial: 1500 × 850
- Central widget: `QTabWidget` with 3 tabs

### 3.2 Tab 1: Calculator

`QSplitter(Horizontal)` with 3 panels (45%/35%/20%):
- `ManifoldPanel` — manifold loading, NZ data display, refined index
- `FillingPanel` — Dehn filling configuration and results
- `ExportPanel` — file export and clipboard operations

### 3.3 Tab 2: Kernel Builder

Single `KernelPanel` widget for precomputing filling kernels.

### 3.4 Tab 3: Data Packs

Single `DataPanel` widget for browsing/downloading pre-computed data.

### 3.5 Compute Pipeline (signals/slots)

```
ManifoldPanel.compute_requested(name, q_order_half)
    → MainWindow._start_compute()
        → load_manifold(), find_phase_space_basis(), build_neumann_zagier()
        → show NZ data immediately
        → launch RefinedIndexWorker
    → MainWindow._on_refined_finished()
        → run_weyl_checks()
        → ManifoldPanel.computation_finished(entries, weyl_result)
    → ManifoldPanel.data_ready(data_dict)
        → FillingPanel.reset(data)
        → ExportPanel.set_data(data)
```

### 3.6 Dehn Filling Pipeline

```
FillingPanel.fill_requested(payload)
    → MainWindow._start_dehn_filling()
        → launch DehnFillingWorker
        → nc_found → FillingPanel.nc_search_done()
        → finished → FillingPanel.filling_finished()
                    → ExportPanel.set_dehn_data()
```

### 3.7 Shutdown

`closeEvent` stops all running workers (refined, dehn, kernel, data)
with `quit()` + `wait(2000)` to prevent macOS PySide6 malloc double-free.

### 3.8 `launch_gui()`

```python
def launch_gui():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(APP_STYLESHEET)
    window = MainWindow()
    window.show()
    ret = app.exec()
    del window      # prevent macOS PySide6 crash
    sys.exit(ret)
```

---

## 4. `workers.py` — QThread Workers

### 4.1 `build_eval_grid(r: int) → list[tuple[list[int], list[Fraction]]]`

Per cusp: `m ∈ {-2,-1,0,1,2}`, `e ∈ {-2,-3/2,...,3/2,2}` → 45 per cusp.
Grid size: `45^r`.  Extended e-range (up to ±2) needed for adjoint su(2)
projection check.

### 4.2 `RefinedIndexWorker(QThread)`

**Signals:** `status(str)`, `progress(int, int)`, `finished(object)`, `error(str)`

Iterates `eval_points`, calls `compute_refined_index()` for each,
emits progress and final list of `(m_ext, e_ext, result)`.

### 4.3 Result Dataclasses

```python
@dataclass
class TransformedFillResult:
    cusp_idx: int
    P_nc: int; Q_nc: int           # NC cycle in (α,β)
    R: int; S: int                 # SL(2,ℤ) complement
    p: int; q: int                 # user slope in (γ,δ)
    P_user: int; Q_user: int       # original user slope
    fill_results: list             # [(m_other, e_other, FilledRefinedResult)]
    weyl_a_phys: list | None = None
    weyl_b_phys: list | None = None

@dataclass
class CuspNCInfo:
    cusp_idx: int
    P_nc: int; Q_nc: int; R: int; S: int
    p: int; q: int; P_user: int; Q_user: int
    weyl_a_phys: list | None = None
    weyl_b_phys: list | None = None

@dataclass
class MultiCuspFillResult:
    cusp_info: list[CuspNCInfo]
    fill_result: object  # FilledRefinedResult
```

### 4.4 `_canonicalize_nc_cycles(cycles) → list`

Dedup: keep one from each `{(P,Q), (−P,−Q)}` pair.
Canonical: `Q > 0`, or `Q = 0 and P > 0`.

### 4.5 `DehnFillingWorker(QThread)`

**Signals:** `status(str)`, `progress(int, int)`, `nc_found(object)`,
`finished(object)`, `error(str)`

**Constructor args:** `nz_data`, `cusp_configs` (list of per-cusp
`{cusp_idx, P, Q}` dicts), `q_order_half`, `p_range`, `q_range`,
`manifold_name`.

**Pipeline (`_run`):**
1. For each cusp config, search NC cycles in `(p_range, q_range)` grid
2. Canonicalize NC cycles
3. Emit `nc_found` signal
4. Route to `_run_single_cusp` or `_run_multi_cusp` depending on count
5. Compute filled refined index for each (m_other, e_other) configuration
6. Emit `finished` with list of `TransformedFillResult`/`MultiCuspFillResult`

### 4.6 `KernelBuilderWorker(QThread)`

Wraps `precompute_filling_kernel()` from phase 11 with progress callback.
Used by `KernelPanel`.

---

## 5. `formatters.py` — HTML/KaTeX Formatting

### 5.1 Low-Level Helpers

```python
_frac_to_latex(v: Fraction | float) → str
_coeff_to_latex(c: int | float | Fraction) → str
_slope_latex(P, Q, a=r"\alpha", b=r"\beta") → str
```

### 5.2 Series to KaTeX

```python
_series_to_katex(result: dict, num_hard: int) → str
_filled_series_to_katex(series, num_hard, num_cusp_eta=0) → str
```

Convert coefficient dicts to KaTeX-renderable LaTeX strings.
Inner helpers: `_eta_part()`, `_q_factor()`.

### 5.3 Panel 1 Formatters

```python
format_manifold_info(md, ps) → str       # HTML block
format_gluing_equations(md) → str         # HTML table
format_edge_classification(ps) → str      # HTML with easy/hard edges
format_nz_matrix(nz) → str               # KaTeX matrix block
format_weyl_check(weyl, nz) → str        # HTML + KaTeX for Weyl results
format_refined_index_table(entries, nz, ...) → str  # Main results table
format_panel1_html(md, ps, nz, entries, weyl, ...) → str  # Combined
```

### 5.4 Panel 2 Formatters

```python
format_nc_cycles(cycles, cusp_idx) → str
format_transformed_fill_results(results) → str  # HTML + KaTeX
```

### 5.5 Helper

```python
_charge_to_me(alpha, beta) → (m, e)
_alpha_latex(coeff, cusp) → str
_beta_latex(coeff, cusp) → str
```

---

## 6. `katex.py` — KaTeX Integration

### 6.1 `build_katex_html(body, **colors) → str`

Wraps HTML body in a full page with:
- KaTeX CSS/JS from CDN (version 0.16.21)
- Auto-render extension for `$...$` and `$$...$$`
- System-matched colors (dark/light mode)
- Custom CSS for compact display

### 6.2 `sys_colors() → dict[str, str]`

Detects system dark/light mode, returns `{bg, fg, link}` color strings.

### 6.3 `make_math_view(html_body, min_h=100) → QWidget`

Factory: creates a QWebEngineView with the given KaTeX-wrapped content.

---

## 7. `style.py` — Qt Stylesheet

Exports `APP_STYLESHEET: str` — a Qt CSS string for consistent theming
across all panels (fonts, colors, margins, QFrame borders).

---

## 8. Panels

### 8.1 `ManifoldPanel(QFrame)`

**Signals:** `compute_requested(str, int)`, `data_ready(object)`

**UI:** Manifold name input + q-order spinbox + Compute button +
progress bar + scrollable KaTeX display area.

**Slots:**
- `set_loading(name)`, `show_nz_data(md, ps, nz)`
- `update_progress(done, total)`, `update_status(msg)`
- `computation_finished(entries, weyl_result)`
- `set_error(msg)`

**Properties:** `manifold_name`, `nz_data`, `entries`, `weyl_result`, `q_order_half`

### 8.2 `FillingPanel(QFrame)`

**Signal:** `fill_requested(object)`

**UI:** Per-cusp slope inputs (P, Q spinboxes), NC search range,
Fill button, scrollable results display.

**Slots:**
- `reset(data)` — configure UI from Panel 1 results
- `set_loading()`, `update_progress(done, total)`, `update_status(msg)`
- `nc_search_done(results)`, `filling_finished(results)`, `set_error(msg)`

### 8.3 `ExportPanel(QFrame)`

**UI:** Directory browser, format checkboxes (LaTeX report, JSON,
Mathematica, plain text), "Include Dehn filling" checkbox,
Export button, clipboard buttons (Copy LaTeX, Copy Text).

**Slots:**
- `set_data(data)` — receive Panel 1 data
- `set_dehn_data(results)` — receive Panel 2 data

**Methods:** `_on_export()`, `_copy_latex()`, `_copy_text()`

Calls `write_full_report`, `write_json`, `write_mathematica` from phase 13.

### 8.4 `KernelPanel(QWidget)`

**Signal:** `build_finished(str)`

**UI:** Mode selector (single slope / range), slope inputs, qq_order
spinbox, progress display, Build button, Cancel button.

**Methods:** `_collect_slopes()`, launches `KernelBuilderWorker`.

### 8.5 `DataPanel(QWidget)`

**UI:** Table of available data packs (name, size, status), Refresh
button, Download/Remove buttons per pack, progress bar.

**Inner class:** `_DownloadWorker(QThread)` — fetches + extracts data
packs with progress signals.

**Methods:** `_load_registry()`, `_refresh_remote()`, `_populate_table()`,
`_download_pack()`, `_remove_pack()`.

---

## 9. Tests

GUI is not unit-tested with automated tests (PySide6 requires display).
Instead:

### T14.1 — Worker Unit Tests (no GUI required)

```python
# Test build_eval_grid
grid = build_eval_grid(1)
assert len(grid) == 45  # 5 m × 9 e values

grid2 = build_eval_grid(2)
assert len(grid2) == 45 ** 2
```

### T14.2 — Formatter Unit Tests

```python
# Test _series_to_katex with known result dict
result = {(2, 2): 1, (4, 0): -1}
tex = _series_to_katex(result, num_hard=1)
assert r"\eta" in tex
assert "q" in tex
```

### T14.3 — KaTeX HTML Generation

```python
html = build_katex_html("<p>$x^2$</p>")
assert "katex" in html.lower()
assert "<p>$x^2$</p>" in html
```

### T14.4 — Dataclass Construction

```python
tfr = TransformedFillResult(
    cusp_idx=0, P_nc=1, Q_nc=0, R=0, S=1,
    p=1, q=3, P_user=1, Q_user=3,
    fill_results=[],
)
assert tfr.cusp_idx == 0
```

---

## 10. Acceptance Criteria

- [ ] `python -m manifold_index.app` launches without error (when PySide6 installed)
- [ ] All 3 tabs render correctly
- [ ] Compute pipeline: name → NZ display → refined index → Weyl check
- [ ] Dehn filling pipeline: NC search → filled index → display
- [ ] Export writes all 4 formats to selected directory
- [ ] Clipboard copy works for LaTeX and plain text
- [ ] Kernel builder completes and saves to cache directory
- [ ] No crash on window close (macOS PySide6 cleanup)
- [ ] Graceful degradation when PySide6 not installed (ImportError)
