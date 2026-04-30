# Changelog

All notable changes to Refined Index Calculator.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] — 2026-04-30

### Added
- **Hard-edge basis optimiser** (`core/optimal_basis.find_optimal_hard_basis`).
  Given a manifold and an NC cycle (P, Q), searches integer-unimodular
  changes-of-basis on the hard-edge subspace to maximise the refinement
  count `#{j : a[j] ∈ ℤ ∧ 2·b[j] ∈ ℤ}` subject to `adj_pass = True` and
  integer `adj_val`. Two-stage:
  1. *Analytic prefilter* using the contragredient transform
     `(a, b) → (G⁻ᵀ·a, G⁻ᵀ·b)` — pure rational arithmetic, cheap.
  2. *Verification* via a full Weyl + adjoint check on the top
     candidates only.

  Default `coeff_range = 1` finishes in ≈ 1.7 s on `num_hard = 3`;
  `coeff_range = 2` is more thorough at ≈ 19 s.

- **Auto-application in `NcCompatWorker`**. Per-cycle Weyl checks in
  Card ③ now silently optimise the hard basis when an improvement
  exists. The displayed `(a, b)`, refinement count, and `adj_val` come
  from the optimised basis. Optimisation failures fall back silently.

- **UI indicator**: green *opt K→K′* pill next to γᵢ in the NC table
  when the basis was optimised; tooltip shows G and the refinement gain.

- New `EasyEdgeResult.hard_padding_rhs` and `all_easy_rhs` fields
  (default `[2] * N`, backward-compatible).

- `Session.easy_result` field (not serialised).

### Fixed
- **`ν_internal` hard-coded RHS = 2** in `build_neumann_zagier`. For
  any integer combination of edge equations (e.g. `α·raw[i] + β·raw[j]`)
  the RHS is `2(α + β)`, not 2 — the wrong value introduced a q-power
  offset that broke `m, e ↔ −m, −e` Weyl symmetry of the unrefined
  index for non-raw bases. The default flow (raw + Stage-1 easy edges,
  all RHS = 2 by construction) is unaffected; v1.0.x behaviour is
  preserved.

### Verified
- 6_2 NC=(1,0): default `refinement = 1, adj_val = −2`; optimiser finds
  `refinement = 2, adj_val = −1` (clean integer) at coeff_range = 1
  in ~2 s end-to-end.
- m060 NC=(1,1): default basis already optimal; optimiser returns
  `None` in ~0.1 s, no overhead.
- All 287 unit tests pass.

### Known limitations
- `MultiCuspNcCompatWorker` does not yet run the optimiser.
- The kernel cache key does not include the basis G hash. In practice
  the optimiser is deterministic for a given (manifold, NC cycle) so
  cache collisions don't occur, but defensive cache-key hardening is
  planned for a follow-up.

## [1.0.9] — 2026-04-22

### Fixed
- **6_2 (1, 0) unrefined-fill regression** introduced by v1.0.8: the
  `_exact_e0_candidates` lattice enumeration was silently undercounting
  for high-`n_int` manifolds (e.g. 6_2, n_int=4) whenever the heuristic
  per-axis radius `R[j]` from `_axis_scan_bound`+`_proj_min_fixed`
  underestimated the true sublevel extent. The v1.0.8 floor padding
  raised `R[j]` to ≥40, but the resulting initial box exceeded the 1M
  isotropic cap → R was clamped back to ~15 per axis → adaptive shell
  growth was *skipped entirely* due to the `_was_clamped` early-exit →
  ~192 valid sublevel points (per kernel context) were dropped, leaving
  an uncancelled `{0: -1}` term in the truncated series. Three changes:
  1. Removed the `if _was_clamped: return result` early-exit so shell
     growth always runs.
  2. Removed the inner `_MAX_BOX_SIZE` check inside shell growth — the
     cap now applies only to the *initial* isotropic box; shell growth
     is free to extend each axis anisotropically until the sublevel set
     is exhausted.
  3. Raised `_MAX_SHELL_STEPS` from 32 → 80 with a `_EMPTY_SHELL_STOP=3`
     convergence heuristic (stop after 3 consecutive empty shells).

  6_2 (1, 0) is now correctly stably zero at q_half ∈ {4, 6, 8, 10}.
  m060 (1, 1) refined fix from v1.0.8 is preserved (q^0|η=0 = 0).

### Performance
- **Batched `F_x2` in `_proj_min_fixed`**: the per-axis projection
  minimiser now evaluates ray-scan trial points in 32-point NumPy
  chunks instead of single calls, and the inner `F_x2` always uses the
  vectorised path (the n<8 scalar fallback is removed). Saves the
  Python call overhead at ~35M F evaluations per slope. ~2× speedup
  on the projection step alone.

### Known regressions
- **m111 (1, 0) unrefined fill at q_half=4** is now ~1.5× slower
  (102 s → ~150 s) because the unbounded shell growth correctly walks
  out the (genuinely large) sublevel set instead of stopping at the
  isotropic cap. The cap-induced undercount happened to land on a
  topologically correct stably-zero verdict for this slope, so the
  result is unchanged. A principled fix that replaces `_proj_min_fixed`
  with an exact projection minimiser (eliminating the need for the
  floor padding and shell growth entirely) is planned for v1.1.0.

## [1.0.8] — 2026-04-20

### Fixed
- **Refined Dehn-filling basis-dependence bug** on m060 (1, 1) and similar
  single-cusp fillings: the `_exact_e0_candidates` enumerator was
  under-sizing its per-axis bounding box when the inner local-search
  minimiser (`_proj_min_fixed`) returned non-optimal values, silently
  dropping valid lattice points and making the filled-index result
  basis-dependent at η → 1. Two layers of fix:
  1. *Clip-to-floor* on the heuristic `R[j]`: pad suspiciously-small
     axes without inflating large ones.
  2. *Per-axis adaptive shell growth*: after the initial enumeration,
     grow any axis whose valid points touch the face by +1 and
     enumerate only the new slab (canonical-axis trick). Terminates on
     saturation.
- **Stop button** in the NC cycle search no longer raises
  `AttributeError: 'FillingCard' has no attribute 'trigger_stop_nc'`.
- **NC cycle search cancellation** is now responsive mid-slope: the
  `cancel_check` callback threads through `compute_filled_index` and
  `enumerate_kernel_terms`, so clicking Stop aborts within ~5 s even on
  slow manifolds (previously had to wait for the current slope to
  finish, potentially many minutes).

### Changed
- **`_MAX_BOX_SIZE` default lowered from 50M to 1M** (overridable via
  `_IREF_MAX_BOX_SIZE` env var). On manifolds with highly-elongated
  sublevel sets (e.g. m111 at slope (1, 0)), the default keeps
  per-slope compute time tractable. Trade-off: lattice points beyond
  the per-axis clamp are silently dropped; for the affected manifolds
  the NC-cycle classification of certain fillings may change from
  "closable" to "NC" at very small caps. Raise the env var to
  re-validate if precise results are needed on those cases.

## [1.0.0] — unreleased

First stable release. Data-pack formats, GUI layout, and the service-level
Python APIs consumed by the `manifold_index.services` layer are now considered
stable and will follow semver for future changes.

### Added
- **Data Hub** (3-tab workflow: Download, Generate, Export & Share).
  - Sequential multi-pack downloads with per-pack progress.
  - Generate queue with pause/resume/cancel for kernel, I^ref, and NC-cycle
    caches. Cancel status is now tracked and reflected in the queue row.
  - Publish flow: user-entered pack name/description flow through to the
    registry; category and `target_subdir` are derived from the actual bundled
    file types (with a homogeneity check rejecting mixed-type packs).
- **Offline KaTeX rendering** — math views no longer require network access.
- **Weyl compatibility diagnostic** with collapsed-edge support.
- **Multi-cusp adjoint projection** checks in the filling service.
- **v0.4-style α/β notation** throughout the UI, with γᵢ/δᵢ display for NC
  cycles and full `I(Aα + Bβ) = series` formatting in fill results.
- **Cross-platform builds** — macOS `.app` + Windows `.exe` from a unified
  `release.sh`, with GitHub Actions handling the Windows build on tag push.

### Changed
- **Repository layout flattened** — the `v0.5/` sub-tree was promoted to the
  repository root. `src/`, `tests/`, `assets/`, `pyproject.toml`, and the
  PyInstaller specs now live at the top level.
- **Project status** — `Development Status :: 3 - Alpha` → `5 - Production/Stable`.
- Kernel-cache size and path now populate correctly in the Export tab's
  browser (previously showed `size_bytes=0` and empty paths).
- I^ref and NC-cycle skip-existing logic is now coverage-aware (skips only
  when the cached grid actually covers the requested `qq`, `m_max`, `e_max`).
- Manifold loading moved to the main thread to avoid SnaPy SQLite threading
  errors.

### Fixed
- Kernel path lookup in `list_cache_files` / `list_local_cache`.
- `skip_existing` key mismatch between the Generate tab and the service.
- `progress` signal signature aligned between workers and the service
  (`(done, total)` ints).
- Bézout complement for negative `Q` slopes.
- Fill-result display showing the NC cycle in the α/β basis.
- Window title now shows the full version string.
- Version/platform selectors are independent.
- Live progress updates no longer block the event loop during large grid
  processing.

### Removed
- Legacy `v0.3/` and `v0.4/` development trees (still retrievable via the
  `pre-v1.0-restructure` git tag and commit history).
- Unused publish-target radio buttons in the Export tab (the service infers
  the target subdir from file types).
- Root-level scratch files (`COPILOT_REVIEW.md`, `t.txt`,
  `v0.5_ERROR_ANALYSIS.md`, `v0{4,5}_m006*.json`).

---

For changes prior to 1.0.0, see the `v0.5.*` git tags.
