# Refined Index Calculator v0.5 — Implementation Status

## Phase Progress

| Phase | Description | Status |
|---|---|---|
| 0 | Scaffold — directory tree, copy core/, __init__.py files, pyproject.toml | ✅ Done |
| 1 | Session model — Session, PipelineStage, IndexQuery, FillQuery, NCCycleSet | ✅ Done |
| 2 | Service layer — ComputeService, FillingService, ExportService, DataHubService | ✅ Done (117/117 tests passing) |
| 3 | Advisory & ViewModel layer — Advisory, CardStatus, all ViewModels | ✅ Done (89/89 tests passing) |
| 4 | Formatters — manifold_fmt, index_fmt, weyl_fmt, filling_fmt | ✅ Done (77/77 tests passing) |
| 5 | Theme & core widgets — style, colors, CollapsibleCard, StepperBar, AdvisoryBanner, MathView, SeriesTable, SlopeInput | ✅ Done (manual smoke test; 283/283 prior tests still passing) |
| 6 | Workers — LoadWorker, IndexWorker, WeylWorker, NCSearchWorker, FillWorker, GenerateWorker, DownloadWorker | ✅ Done (283/283 tests still passing) |
| 7 | Pipeline Cards — PipelineView, ManifoldCard, IndexCard, FillingCard, ExportCard | ✅ Done (283/283 tests still passing; manual smoke test pending) |
| 8 | Data Hub — DataHubView, DownloadTab, GenerateTab, ExportTab | ✅ Done (283/283 tests still passing; manual smoke test pending) |
| 9 | Main window — MainWindow, __main__.py, launch_gui() | ✅ Done |
| 10 | Polish — session save/restore, keyboard shortcuts, autocomplete | ✅ Done |

## Build & Packaging

| Artifact | Status |
|---|---|
| `ManifoldIndex.spec` | ✅ Written (v0.5.0, adds PySide6.QtWebEngine* hidden imports) |
| `build_app.sh` | ✅ Written (auto-detects venv, preflight check, ad-hoc sign, zip) |
| `rthook_snappy.py` | ✅ Copied verbatim from v0.4 |
| `assets/` | ✅ Icons copied from v0.4 (.icns, .png, .svg) |
| `launcher.py` | ✅ Already written (Phase 9) |

## Notes

- `core/` copied verbatim from v0.4. Do not modify.
- `utils/` copied verbatim from v0.4.
- Each stub file contains a comment with its target Phase and a BLUEPRINT reference.
- Implement phases in order; each phase is independently testable before starting the next.
