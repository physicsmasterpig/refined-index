# Changelog

All notable changes to Refined Index Calculator.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
