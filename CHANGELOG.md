# Changelog

All notable changes to Refined Index Calculator.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
