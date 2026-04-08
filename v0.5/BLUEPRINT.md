# Refined Index Calculator v0.5 — Complete Architecture Blueprint

> **Purpose:** Single source of truth for reconstructing and extending the
> `manifold-index` project as version 0.5.  This document covers the full
> GUI/workflow redesign; the mathematical core (`core/`) is carried over
> from v0.4 unchanged and is only referenced here, not re-specified.
>
> **How to use:** Read §1–§6 for the big picture.  Then implement each
> Phase in §14 sequentially.  Each phase is self-contained and testable.

---

## Table of Contents

1.  [Goals of v0.5](#1-goals-of-v05)
2.  [Design Language](#2-design-language)
3.  [What is Kept from v0.4](#3-what-is-kept-from-v04)
4.  [What Changed and Why](#4-what-changed-and-why)
5.  [Directory Layout](#5-directory-layout)
6.  [Dependency Graph](#6-dependency-graph)
7.  [Session Model](#7-session-model)
8.  [Service Layer](#8-service-layer)
9.  [ViewModel Layer](#9-viewmodel-layer)
10. [App Layer — Core Widgets](#10-app-layer--core-widgets)
11. [App Layer — Pipeline Cards](#11-app-layer--pipeline-cards)
12. [App Layer — Data Hub](#12-app-layer--data-hub)
13. [Edge Case & Advisory System](#13-edge-case--advisory-system)
14. [Implementation Phases](#14-implementation-phases)
15. [Testing Strategy](#15-testing-strategy)
16. [Build & Packaging](#16-build--packaging)
17. [Design Decisions Log](#17-design-decisions-log)

---

## 1. Goals of v0.5

### 1.1 Core Problems Fixed

The v0.4 GUI had the following structural problems (to be fully resolved):

| Problem | Root cause |
|---|---|
| `window.py` is a god controller (282 lines of cross-panel wiring) | No service layer; business logic in UI code |
| `workers.py` is monolithic (668 lines; one worker does 4 things) | No separation of async tasks |
| `formatters.py` is monolithic (1207 lines; all HTML in one file) | No split by domain |
| Forced linear workflow (Panel 2 locked until Panel 1 finishes) | State lives in panels, not a session |
| No targeted query — always computes all 225+ sectors | No "query" mode |
| No refinement control — always full refined | No η toggle |
| No cache awareness in UI — always computes from scratch | No service-layer cache probing |
| Silent fallbacks on edge cases (no NC cycles, Weyl fails, etc.) | No advisory system |
| 2 separate tabs (Kernel Builder + Data Packs) for related things | No unified Data Hub |
| CLI-only tools for I^ref cache, NC cache, pack publishing | Not exposed in GUI |

### 1.2 v0.5 Design Goals

1. **Layered architecture**: `core/` ← `services/` ← `viewmodels/` ← `app/`.  Each layer has zero upward dependency.
2. **Session as truth**: All computation state in one `Session` object.  Panels only read from / write to session.
3. **Query-based workflow**: Users ask specific questions (one `(m, e)` at a time) rather than compute-everything-show-everything.
4. **Targeted refinement**: Per-hard-edge η toggles.  Projection computed in post-processing, not by re-running core.
5. **Cache-aware UI**: Card ① probes all three caches at load time.  Cards ② ③ offer "From Cache" mode.
6. **Graceful degradation**: Every edge case (no NC cycles, Weyl fails, etc.) shows an advisory banner with actionable choices.  No silent fallbacks, no dead ends.
7. **Pipeline card layout**: Single scrollable page with collapsible cards.  No sidebar.  Previous results are always accessible via expand.
8. **Unified Data Hub**: Download / Generate / Export & Share in one page, replacing both the Kernel panel and Data panel.
9. **Thin workers**: One `QThread` per async task.  Workers call exactly one service method.
10. **Split formatters**: One formatter module per domain (`manifold_fmt`, `index_fmt`, `filling_fmt`, `weyl_fmt`).

---

## 2. Design Language

This is a **research instrument**, not a consumer app.  The visual language
reflects that: precise, quiet, and distraction-free.  The mathematics is
the content; the UI is the frame.

### 2.1 Principles

| Principle | Implication |
|---|---|
| **Minimal chrome** | Every pixel that isn't data is overhead |
| **One accent, one weight** | A single accent colour (deep indigo) on a neutral base; bold type only for labels, never for decoration |
| **No decorative icons** | Icons only where a glyph conveys information faster than a word — and nowhere else.  Status and mode are communicated through colour and text, not emoji or multi-colour icons |
| **Dense but breathable** | Research tables are dense; cards and panels give 12–16 px breathing room, not 48 px padding |
| **Monospace for math** | All q-series output, NZ matrix entries, and numeric results render in a monospace face.  UI labels are in a system sans-serif |
| **Flat, borderless** | Cards have a 1 px top border (accent colour) and a subtle background tint.  No drop shadows, no gradients, no rounded corners beyond 2 px |
| **Colour carries only state** | Colour differences are limited to: default / accent / warning-amber / error-red / muted.  Never used purely decoratively |

### 2.2 Palette

```
Background:        #F9F9F8      (warm off-white; eases long reading sessions)
Surface (card):    #FFFFFF
Surface (alt):     #F3F3F2      (alternating table rows, collapsed card fill)
Border:            #E2E2E0
Border (strong):   #C8C8C4

Accent:            #3B3B9A      (deep indigo — used for: card top border, focused
                                 inputs, active stepper step, primary button bg)
Accent hover:      #2E2E7A
Accent muted:      #EBEBF5      (light tint of accent; advisory info background)

Text primary:      #1A1A1A
Text secondary:    #5A5A5A
Text muted:        #9A9A9A
Text on accent:    #FFFFFF

Warning bg:        #FFFBF0
Warning border:    #D4860A
Warning text:      #7A4E00

Error bg:          #FFF5F5
Error border:      #C0392B
Error text:        #7D1E1E

Success:           #2E7D52      (used only for cache hit indicators)

Monospace font:    "JetBrains Mono", "Fira Code", "Menlo", monospace
UI font:           system-ui, -apple-system, "Segoe UI", sans-serif
```

### 2.3 Button Styles

Three button variants only:

```
Primary   — accent background, white text, 2 px radius
            "Run"   "Compute"   "Load"   "Export"

Secondary — transparent background, accent text, 1 px accent border
            "Add query"   "Browse"   "Cancel"   "Remove"

Tertiary  — transparent background, secondary text, no border
            "Copy LaTeX"   "Copy Math"   (hover shows underline)
```

No button has an icon.  If the action needs a visual cue, the button label
is unambiguous enough on its own.

**Exception:** Pause / Resume in the Data Hub Generate tab use ▶ and ▐▐ 
glyphs inline in the label text (not as icon widgets), e.g. `"▶  Resume"`.

### 2.4 Status Indicators

Status is shown by a small coloured pill badge (text only):

```
● Running   (accent colour)
✓ Done      (success green)     — on collapsed card header
  Ready     (muted, no dot)
  Locked    (muted, no dot)
⚠ Warning   (warning amber)
✕ Error     (error red)
∿ Stale     (muted, italic)
```

No multi-colour icons.  No emoji in any status badge.

### 2.5 Advisory Banners

Each advisory is a horizontal stripe inside the card:

```
┌────────────────────────────────────────────────────────────────────┐
│  Left border (4 px, colour matches level)                          │
│  Level tag (small caps, muted)      TITLE (medium weight)          │
│  Body text (secondary colour, regular weight)                      │
│                                        [Action A]   [Action B]     │
└────────────────────────────────────────────────────────────────────┘
```

Level → border colour mapping:

| Level | Border | Background |
|---|---|---|
| info | Accent | Accent muted (#EBEBF5) |
| warning | Warning border (#D4860A) | Warning bg (#FFFBF0) |
| error | Error border (#C0392B) | Error bg (#FFF5F5) |
| action | #6A3B9A (purple) | #F5F0FF |

### 2.6 Card Header

```
┌────────────────────────────────────────────────────────────────────┐ ← 4 px top border (accent)
│  ①  Manifold                             ● Running  [collapse ▲]  │
│     m004 · 2 tet · 1 cusp · 1 hard edge                           │  ← visible when collapsed
└────────────────────────────────────────────────────────────────────┘
```

- Card index number (①②③④) in a small circle, accent colour
- Title in medium weight
- Status badge right-aligned
- Collapse/expand: plain text `[▲]` / `[▼]` — no icon widget
- Summary line visible only when collapsed; hidden when expanded

### 2.7 Stepper Bar

```
─── ① Load ─────────────────── ② Index ──────────────────── ③ Fill ── ④ Export ───
    Done ✓                      Done ✓                      Running    Locked
```

- Horizontal rule connecting steps
- Numbered circles (16 px), filled accent = done, outline = future
- Step label below in small caps
- Clicking a completed step scrolls to that card

### 2.8 Table Design

```
  m      e      Series (q^½ terms)               Source
─────────────────────────────────────────────────────────────────────
  0      0      1  −  η₀ q^{1/2}  +  q  −  …    computed
  1      0      q^{1/2}  −  2q  +  3q^{3/2}  …  computed
─────────────────────────────────────────────────────────────────────
```

- No outer border on the table itself; rows separated by 1 px lines
- Action buttons ("Copy LaTeX", "Remove") appear on hover only, in a
  right-aligned tertiary style.  The table row is otherwise clean.
- Alternating row tint: every other row is Surface (alt) #F3F3F2

---

## 3. What is Kept from v0.4

The following are **copied verbatim** into `v0.5/src/manifold_index/`:

```
core/
    __init__.py
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
    data_packs.py
    _c_kernel/
        tet_index.c
        setup.py

utils/
    exporters.py
    cache_export.py

data/
    data_packs.json
    kernel_cache/        (user's local cache — not in repo)
```

**No changes to `core/` are required for v0.5.**  All new features are implemented in `services/`, `viewmodels/`, and `app/`.

---

## 3. What Changed and Why

### 3.1 GUI Architecture: Splitter → Pipeline Cards

**v0.4:** Three horizontal panels in a `QSplitter`.  Panel 2 is locked until Panel 1 finishes.  No scrolling; content cramped in 1/3 width.

**v0.5:** Single scrollable page with four collapsible `PipelineCard` widgets stacked vertically.

- **Completed cards auto-collapse** to a 1–2 line summary.
- **Active card is fully expanded** with all controls and results.
- **Future cards show as locked placeholders** with a tooltip explaining prerequisites.
- **Any card can be expanded** at any time to revisit its full results.
- A **Stepper bar** at the top provides click-to-scroll navigation and shows pipeline progress.

Reason: sidebar navigation hides previous results behind page switches, which breaks the research workflow (you need NZ data visible while interpreting the refined index, etc.).  The pipeline card approach gives full context when wanted and clean focus otherwise.

### 3.2 Single-Page Pipeline vs Separate Pages

**v0.4:** Tab widget with three tabs.  "Calculator" tab contains the three panels.

**v0.5:** Two **modes** (not tabs) toggled by a top-bar button:
- **Calculator** — the scrollable pipeline card view
- **Data Hub** — unified data management (sub-tabbed internally)

Reason: Data Hub is not part of the calculation pipeline.  It's a utility tool.  A top-bar toggle is less confusing than a tab that "accidentally" appears next to the workflow.

### 3.3 Workflow: Compute-All → Query-Based

**v0.4:** Card ① press → computes ALL 45^r sectors automatically.

**v0.5:** Card ② has three modes:
1. **Query** (default): User inputs `(m, e)` per cusp → one result
2. **Grid**: Configurable range, batch compute (like v0.4)
3. **From Cache**: Instant lookup from I^ref datapack

Reason: A physicist typically wants `I^ref(0, 0)` or `I^ref(1, 0)`, not all 225 sectors.  Forcing the full grid makes the app slow and the results overwhelming.

### 3.4 Refinement Controls

**v0.4:** Always full-refined.  `num_hard = 0` case silently produces 1-tuples.

**v0.5:**
- **Preset dropdown**: Full Refined / Unrefined / Custom
- **Per-edge toggles**: η_j on/off (Custom mode only)
- **Projection is post-processing**: Core always computes full refined; toggles project the result dict — no re-computation.
- **Auto-adapt**: When `num_hard = 0`, the refinement section is hidden entirely and the card title changes to "3D Index".

### 3.5 Dehn Filling: Monolithic → Two-Phase

**v0.4:** NC search + filling run as one monolithic `DehnFillingWorker`.

**v0.5:** Card ③ has two distinct phases with separate buttons:
1. **Phase A — Find NC Cycles**: Run `find_non_closable_cycles` or load from NC cache.
2. **Phase B — Compute Filled Index**: Pick one NC cycle + one `(m_other, e_other)` query → one result.

Reason: The NC search result is informative on its own (topology).  Forcing it to couple to filling means you can't inspect NC cycles without also specifying a user slope.

### 3.6 Data Hub Unification

**v0.4:** Kernel panel (GUI) + Data panel (GUI) + 5 CLI scripts + `cache_export.py` (not in GUI).

**v0.5:** Single Data Hub page with three sub-tabs:
- **Download**: Registry + remote packs (replaces Data panel)
- **Generate**: All data types — kernels, I^ref cache, NC cycles (replaces Kernel panel + 3 CLI scripts)
- **Export & Share**: Cache browser + format conversion + pack publishing (replaces `cache_export.py` + `publish_datapack.py`)

---

## 4. Directory Layout

```
v0.5/
├── BLUEPRINT.md                 ← this file
├── STATUS.md                    ← phase progress tracker
├── pyproject.toml
├── setup.py
├── launcher.py
├── build_app.sh
├── ManifoldIndex.spec
├── assets/
│   ├── ManifoldIndex.icns
│   ├── ManifoldIndex.svg
│   └── ManifoldIndex_1024.png
├── tests/
│   ├── conftest.py
│   ├── test_session.py
│   ├── test_compute_service.py
│   ├── test_filling_service.py
│   ├── test_datahub_service.py
│   ├── test_viewmodels.py
│   └── test_advisory.py
└── src/
    └── manifold_index/
        ├── __init__.py
        │
        ├── core/                        ← copied verbatim from v0.4
        │   ├── manifold.py
        │   ├── gluing_equations.py
        │   ├── phase_space.py
        │   ├── neumann_zagier.py
        │   ├── basis_selection.py
        │   ├── index_3d.py
        │   ├── refined_index.py
        │   ├── weyl_check.py
        │   ├── dehn_filling.py
        │   ├── refined_dehn_filling.py
        │   ├── kernel_cache.py
        │   ├── data_packs.py
        │   └── _c_kernel/
        │       ├── tet_index.c
        │       └── setup.py
        │
        ├── services/                    ← NEW: business logic, no Qt
        │   ├── __init__.py
        │   ├── session.py               ← Session model
        │   ├── compute_service.py       ← load → NZ → refined index
        │   ├── filling_service.py       ← NC search → kernel → fill
        │   ├── datahub_service.py       ← download, generate, export packs
        │   └── export_service.py        ← format + write result files
        │
        ├── viewmodels/                  ← NEW: display-ready data, no Qt
        │   ├── __init__.py
        │   ├── advisory.py              ← Advisory / CardStatus system
        │   ├── manifold_vm.py           ← ManifoldViewModel
        │   ├── index_vm.py              ← IndexViewModel
        │   ├── filling_vm.py            ← FillingViewModel
        │   └── export_vm.py             ← ExportViewModel
        │
        ├── app/                         ← Qt GUI
        │   ├── __init__.py
        │   ├── __main__.py
        │   ├── window.py                ← MainWindow: mode switcher
        │   │
        │   ├── theme/
        │   │   ├── __init__.py
        │   │   ├── style.py             ← QSS stylesheet
        │   │   ├── colors.py            ← palette constants
        │   │   └── icons.py             ← icon registry
        │   │
        │   ├── widgets/                 ← Reusable atomic widgets
        │   │   ├── __init__.py
        │   │   ├── collapsible_card.py  ← PipelineCard expand/collapse
        │   │   ├── stepper.py           ← Pipeline progress stepper bar
        │   │   ├── advisory_banner.py   ← Inline advisory/warning banners
        │   │   ├── math_view.py         ← KaTeX WebEngine wrapper
        │   │   ├── progress_card.py     ← Inline progress indicator
        │   │   ├── series_table.py      ← q-series result table
        │   │   └── slope_input.py       ← (P, Q) input row
        │   │
        │   ├── pipeline/                ← Calculator mode
        │   │   ├── __init__.py
        │   │   ├── pipeline_view.py     ← Scrollable column + stepper
        │   │   ├── manifold_card.py     ← Card ①: Load
        │   │   ├── index_card.py        ← Card ②: Refined Index
        │   │   ├── filling_card.py      ← Card ③: Dehn Filling
        │   │   └── export_card.py       ← Card ④: Export
        │   │
        │   ├── datahub/                 ← Data Hub mode
        │   │   ├── __init__.py
        │   │   ├── datahub_view.py      ← Sub-tab container
        │   │   ├── download_tab.py      ← Browse + download remote packs
        │   │   ├── generate_tab.py      ← Build kernels / iref / NC
        │   │   └── export_tab.py        ← Cache browser + publish
        │   │
        │   └── workers/                 ← One thin QThread per task
        │       ├── __init__.py
        │       ├── load_worker.py
        │       ├── index_worker.py
        │       ├── nc_search_worker.py
        │       ├── fill_worker.py
        │       ├── weyl_worker.py
        │       ├── generate_worker.py   ← kernels / iref / NC
        │       └── download_worker.py
        │
        ├── formatters/                  ← HTML generation, split by domain
        │   ├── __init__.py
        │   ├── manifold_fmt.py
        │   ├── index_fmt.py
        │   ├── filling_fmt.py
        │   └── weyl_fmt.py
        │
        ├── utils/                       ← copied verbatim from v0.4
        │   ├── exporters.py
        │   └── cache_export.py
        │
        └── data/
            ├── data_packs.json
            └── kernel_cache/            ← user local cache (not in repo)
```

---

## 5. Dependency Graph

```
core/         (no internal app deps; pure math)
    │
    ▼
services/     (imports core only; zero Qt)
    │
    ▼
viewmodels/   (imports services + core types; zero Qt)
    │
    ▼
formatters/   (imports viewmodels; zero Qt)
    │
    ▼
app/          (imports everything; Qt lives here only)
  ├── workers/   (imports services + Qt)
  ├── widgets/   (imports viewmodels + Qt)
  ├── pipeline/  (imports widgets + workers + viewmodels + Qt)
  └── datahub/   (imports widgets + workers + viewmodels + Qt)
```

**Golden rule:** Nothing in `core/`, `services/`, `viewmodels/`, or `formatters/` may import from `app/` or `PySide6`.

---

## 6. Session Model

**File:** `services/session.py`

The `Session` is the single source of truth for one manifold's calculation state.  All pipeline cards read from and write to a shared session instance.  No card stores results in instance variables.

```python
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from fractions import Fraction
from typing import Any


class PipelineStage(Enum):
    """Which stage of the pipeline has been reached (minimum requirement)."""
    EMPTY    = 0   # no manifold loaded
    LOADED   = 1   # manifold data + NZ built
    INDEXED  = 2   # at least one refined index query computed
    FILLED   = 3   # at least one filled index query computed


@dataclass
class IndexQuery:
    """One (m_ext, e_ext) refined index query and its result."""
    m_ext: list[int]
    e_ext: list[Fraction]
    q_order_half: int
    result: Any                    # RefinedIndexResult | None
    projected_result: Any          # after η projection; None if full refined
    active_edges: list[bool]       # per hard-edge refinement toggles
    timestamp: float = field(default_factory=time.time)
    source: str = "computed"       # "computed" | "cache"


@dataclass
class FillQuery:
    """One NC cycle + charge + filled index result."""
    cusp_idx: int
    nc_P: int                      # NC cycle in (α,β) basis
    nc_Q: int
    user_P: int                    # user's slope
    user_Q: int
    p: int                         # user slope in (γ,δ) basis
    q: int
    m_other: list[int]             # charges on unfilled cusps
    e_other: list[Fraction]
    q_order_half: int
    result: Any                    # FilledRefinedResult | None
    weyl_a: list[Fraction] | None  # Weyl vectors used
    weyl_b: list[Fraction] | None
    incompat_edges: list[int]      # edges forced to η=1 for Weyl compat
    timestamp: float = field(default_factory=time.time)
    source: str = "computed"


@dataclass
class NCCycleSet:
    """Result of an NC cycle search for one cusp."""
    cusp_idx: int
    search_p_range: tuple[int, int]
    search_q_range: tuple[int, int]
    q_order_half: int
    cycles: list[Any]              # list[NonClosableCycle]
    source: str = "computed"       # "computed" | "cache"


@dataclass
class Session:
    """
    Complete state for one manifold calculation session.

    Panels read from this; workers write to this via signals that
    MainWindow intercepts and applies.  The session is serialisable
    to JSON for save/restore.
    """

    # ── Identity ──────────────────────────────────────────────
    manifold_name: str = ""
    generation: int = 0            # increments each time manifold changes

    # ── Stage ─────────────────────────────────────────────────
    stage: PipelineStage = PipelineStage.EMPTY

    # ── Card ① results ────────────────────────────────────────
    manifold_data: Any = None      # ManifoldData | None
    nz_data: Any = None            # NeumannZagierData | None
    cache_status: dict = field(default_factory=dict)
    # cache_status keys: "iref", "nc", "kernels"
    # values: {"available": bool, "qq_order": int|None, "details": str}

    # ── Card ② settings ───────────────────────────────────────
    q_order_half: int = 20         # shared qq setting (applies to ①②③)
    active_edges: list[bool] = field(default_factory=list)
    # active_edges[j] = True → η_j is active in output; False → project out
    index_queries: list[IndexQuery] = field(default_factory=list)

    # ── Weyl bridge (②→③) ─────────────────────────────────────
    weyl_result: Any = None        # ABVectors | None
    weyl_checked: bool = False

    # ── Card ③ results ────────────────────────────────────────
    nc_cycles: list[NCCycleSet] = field(default_factory=list)
    fill_queries: list[FillQuery] = field(default_factory=list)

    # ── Card ④ settings ───────────────────────────────────────
    export_path: str = ""

    # ── Helpers ───────────────────────────────────────────────

    def invalidate_from(self, stage: PipelineStage) -> None:
        """Clear all results at and beyond *stage*.  Called when upstream changes."""
        if stage <= PipelineStage.LOADED:
            self.manifold_data = None
            self.nz_data = None
            self.cache_status = {}
            self.active_edges = []
            self.index_queries = []
            self.weyl_result = None
            self.weyl_checked = False
            self.nc_cycles = []
            self.fill_queries = []
            self.generation += 1
            self.stage = PipelineStage.EMPTY
        elif stage <= PipelineStage.INDEXED:
            self.index_queries = []
            self.weyl_result = None
            self.weyl_checked = False
            self.nc_cycles = []
            self.fill_queries = []
            self.stage = PipelineStage.LOADED
        elif stage <= PipelineStage.FILLED:
            self.fill_queries = []
            self.stage = PipelineStage.INDEXED

    def has_any_results(self) -> bool:
        return self.stage >= PipelineStage.LOADED

    def num_hard(self) -> int:
        if self.nz_data is None:
            return 0
        return self.nz_data.num_hard

    def all_edges_active(self) -> bool:
        return all(self.active_edges) if self.active_edges else True

    def no_hard_edges(self) -> bool:
        return self.num_hard() == 0
```

### 6.1 Session Serialisation

The session is serialisable to JSON (for save/restore and future session tabs):

```python
def session_to_dict(session: Session) -> dict:
    """Serialise session to a JSON-safe dict."""
    ...

def session_from_dict(data: dict) -> Session:
    """Restore a Session from a saved dict."""
    ...
```

Serialisation stores manifold name, settings, and all query inputs/outputs (converting `Fraction` → `[num, den]`, `RefinedIndexResult` → list of `[key, val]` pairs, etc.).

---

## 7. Service Layer

All services are **pure Python** — no Qt, no UI imports.  They take plain inputs, call core, and return typed results.  They are independently testable without a running application.

### 7.1 `ComputeService` (`services/compute_service.py`)

```python
class ComputeService:
    """Orchestrates the load → NZ → index pipeline."""

    @staticmethod
    def load_manifold(name: str) -> tuple[Any, Any, Any]:
        """
        Load manifold, find phase space basis, build NZ matrix.

        Returns (ManifoldData, EasyEdgeResult, NeumannZagierData).
        Raises ValueError if SnaPy cannot find the manifold.
        """

    @staticmethod
    def probe_cache(name: str, nz_data: Any) -> dict:
        """
        Check which data packs are available for this manifold.

        Returns dict:
          {
            "iref":    {"available": bool, "qq_order": int|None, "m_range": int|None},
            "nc":      {"available": bool, "qq_order": int|None, "p_range": int|None},
            "kernels": {"available": bool, "count": int, "qq_orders": list[int]},
          }
        """

    @staticmethod
    def compute_refined_index(
        nz_data: Any,
        m_ext: list[int],
        e_ext: list,
        q_order_half: int,
    ) -> Any:
        """Call core.refined_index.compute_refined_index. Returns RefinedIndexResult."""

    @staticmethod
    def load_refined_index_from_cache(
        name: str,
        m_ext: list[int],
        e_ext: list,
    ) -> Any | None:
        """Load I^ref(m,e) from iref cache file. Returns None if not cached."""

    @staticmethod
    def project_refined_index(
        result: Any,
        active_edges: list[bool],
    ) -> Any:
        """
        Project RefinedIndexResult by setting η_j = 1 for inactive edges.

        Implemented as: sum keys with same q_half_power and non-masked η positions.
        Returns a new RefinedIndexResult (never mutates input).
        """

    @staticmethod
    def run_weyl_check(
        entries: list,
        num_hard: int,
        q_order_half: int,
    ) -> Any | None:
        """
        Run Weyl symmetry check on a set of (m_ext, e_ext, result) entries.
        Returns ABVectors | None.
        """
```

### 7.2 `FillingService` (`services/filling_service.py`)

```python
class FillingService:
    """Orchestrates NC search → basis change → kernel → filled index."""

    @staticmethod
    def find_nc_cycles(
        nz_data: Any,
        cusp_idx: int,
        p_range: tuple[int, int],
        q_range: tuple[int, int],
        q_order_half: int,
        progress_fn=None,
    ) -> Any:
        """Run find_non_closable_cycles. Returns NonClosableCycleResult."""

    @staticmethod
    def load_nc_from_cache(name: str, cusp_idx: int) -> Any | None:
        """Load NC cycle list from nc cache file. Returns list[NonClosableCycle] | None."""

    @staticmethod
    def probe_kernel(P: int, Q: int, q_order_half: int) -> dict:
        """
        Check kernel cache for (P, Q, qq).

        Returns {"available": bool, "cached_qq": int|None, "hj_length": int}.
        """

    @staticmethod
    def compute_filled_index(
        nz_data: Any,
        cusp_idx: int,
        nc_P: int,
        nc_Q: int,
        user_P: int,
        user_Q: int,
        m_other: list[int] | None,
        e_other: list | None,
        q_order_half: int,
        weyl_a: list | None,
        weyl_b: list | None,
        auto_precompute: bool = True,
        progress_fn=None,
    ) -> tuple[int, int, Any]:
        """
        Apply basis change, compute filled refined index.

        Returns (p, q, FilledRefinedResult) where p, q are the slope in the
        NC basis.  Handles single-cusp and multi-cusp cases uniformly.
        """

    @staticmethod
    def canonicalise_nc_cycles(cycles: list) -> list:
        """Deduplicate: keep one from each {(P,Q), (-P,-Q)} pair."""
```

### 7.3 `DataHubService` (`services/datahub_service.py`)

```python
class DataHubService:
    """Orchestrates data pack download, generation, and publishing."""

    # ── Download ──
    @staticmethod
    def load_registry(use_remote: bool = False) -> Any:
        """Load pack registry (bundled or remote)."""

    @staticmethod
    def download_pack(registry: Any, pack: Any, progress_fn=None, status_fn=None) -> int:
        """Download and install a pack. Returns n_files."""

    @staticmethod
    def remove_pack(pack: Any) -> int:
        """Remove installed pack files. Returns n_removed."""

    # ── Generate ──
    @staticmethod
    def build_kernels(
        slopes: list[tuple[int, int]],
        qq: int,
        n_workers: int,
        skip_existing: bool,
        progress_fn=None,
        status_fn=None,
        cancel_fn=None,
    ) -> list[tuple[int, int, str]]:
        """Build filling kernels. Returns list of (P, Q, status_str)."""

    @staticmethod
    def build_iref_cache(
        manifold_names: list[str],
        qq: int,
        m_max: int,
        e_max: int,
        n_workers: int,
        skip_existing: bool,
        progress_fn=None,
        status_fn=None,
    ) -> list[tuple[str, str]]:
        """Build I^ref cache for each manifold. Returns list of (name, status)."""

    @staticmethod
    def build_nc_cache(
        manifold_names: list[str],
        qq: int,
        p_max: int,
        q_max: int,
        n_workers: int,
        skip_existing: bool,
        progress_fn=None,
        status_fn=None,
    ) -> list[tuple[str, str]]:
        """Build NC cycle cache. Returns list of (name, status)."""

    # ── Local cache introspection ──
    @staticmethod
    def list_local_cache() -> dict:
        """
        Aggregate summary of all local cache contents.

        Returns {
          "kernels": {"count": int, "size_bytes": int, "entries": list[...]},
          "iref":    {"count": int, "size_bytes": int, "entries": list[...]},
          "nc":      {"count": int, "size_bytes": int, "entries": list[...]},
        }
        """

    @staticmethod
    def list_cache_files(type_filter: str | None = None) -> list[dict]:
        """List individual cache files. type_filter: "kernels"|"iref"|"nc"|None."""

    # ── Export & Share ──
    @staticmethod
    def export_cache_files(
        file_paths: list,
        formats: list[str],
        output_dir,
    ) -> list:
        """Convert pkl.gz files to selected formats. Returns list[Path]."""

    @staticmethod
    def create_tarball(
        file_paths: list,
        pack_name: str,
        release_tag: str,
        output_dir,
        update_registry: bool = True,
    ) -> dict:
        """Package files into .tar.gz + update data_packs.json.
        Returns {"path": Path, "sha256": str, "size_bytes": int, ...}."""
```

### 7.4 `ExportService` (`services/export_service.py`)

```python
class ExportService:
    """Write session results to files."""

    @staticmethod
    def available_data(session: "Session") -> dict:
        """
        Inspect session and return what can be exported.

        Returns {
          "manifold": bool,
          "index_queries": int,          # number of computed query results
          "weyl": bool,
          "nc_cycles": int,
          "fill_queries": int,
        }
        """

    @staticmethod
    def write_latex(session: "Session", output_path, include_filling: bool = True) -> None:
        """Write a LaTeX .tex file for all session results."""

    @staticmethod
    def write_mathematica(session: "Session", output_path, include_filling: bool = True) -> None:
        """Write a Mathematica .m file."""

    @staticmethod
    def write_json(session: "Session", output_path) -> None:
        """Write a structured JSON file."""

    @staticmethod
    def write_full_report(session: "Session", output_path) -> None:
        """Write a compilable LaTeX report with all data."""

    @staticmethod
    def clipboard_latex(result: Any, num_hard: int) -> str:
        """Return LaTeX string for a single result (for clipboard copy)."""

    @staticmethod
    def clipboard_plain(result: Any, num_hard: int) -> str:
        """Return plain text string for a single result."""
```

---

## 8. ViewModel Layer

ViewModels transform service/core outputs into **display-ready data structures**.  They contain pre-formatted strings, status labels, and advisory lists.  They have **zero Qt dependency** and are independently testable.

### 8.1 `ManifoldViewModel` (`viewmodels/manifold_vm.py`)

```python
@dataclass
class ManifoldViewModel:
    name: str
    n_tetrahedra: int
    n_cusps: int
    num_hard: int
    num_easy: int
    has_hard_edges: bool           # num_hard > 0
    index_title: str               # "Refined Index" or "3D Index" (num_hard=0)
    nz_latex: str                  # KaTeX-ready matrix string
    gluing_table_html: str         # HTML table for gluing equations
    easy_edges_html: str
    hard_edges_html: str
    is_symplectic: bool
    cache_status: dict             # from ComputeService.probe_cache

    # Advisories attached to this card
    advisories: list["Advisory"]
```

### 8.2 `IndexViewModel` (`viewmodels/index_vm.py`)

```python
@dataclass
class IndexQueryViewModel:
    m_ext: list[int]
    e_ext: list               # list[Fraction]
    q_order_half: int
    active_edges: list[bool]
    result_latex: str          # KaTeX-ready series string (full refined)
    projected_latex: str       # KaTeX-ready string after η projection
    is_zero: bool
    source: str               # "computed" | "cache"
    timestamp: float

@dataclass
class IndexViewModel:
    queries: list[IndexQueryViewModel]
    weyl_status: "WeylViewModel | None"
    advisories: list["Advisory"]
```

### 8.3 `WeylViewModel` (`viewmodels/index_vm.py` — appended)

```python
@dataclass
class WeylViewModel:
    checked: bool
    a_vectors: list            # list[Fraction] per hard edge
    b_vectors: list
    edge_compatible: list[bool]
    is_fully_compatible: bool
    adjoint_value: float | None
    adjoint_passed: bool | None
    warnings: list[str]
    advisories: list["Advisory"]
```

### 8.4 `FillingViewModel` (`viewmodels/filling_vm.py`)

```python
@dataclass
class NCCycleViewModel:
    cusp_idx: int
    P: int
    Q: int
    slope_latex: str           # "$\alpha$" or "$\alpha + 2\beta$" etc.
    weyl_compatible: bool | None
    source: str               # "computed" | "cache"

@dataclass
class FillQueryViewModel:
    nc_slope_latex: str
    user_slope_latex: str
    p: int
    q: int
    m_other: list[int]
    e_other: list
    result_latex: str
    is_zero: bool
    incompat_edges: list[int]
    weyl_a_latex: str | None
    weyl_b_latex: str | None
    source: str
    timestamp: float

@dataclass
class FillingViewModel:
    nc_cycles: list[NCCycleViewModel]   # per cusp
    fill_queries: list[FillQueryViewModel]
    advisories: list["Advisory"]
```

---

## 9. App Layer — Core Widgets

### 9.1 `CollapsibleCard` (`app/widgets/collapsible_card.py`)

The foundational UI element.  Used for all four pipeline cards.

```
State machine:
  LOCKED → (prerequisite met) → READY
  READY  → (run) → RUNNING → DONE | WARNING | ERROR
  DONE   ← expand/collapse → DONE
  DONE | WARNING → (upstream change) → STALE
  STALE  → (re-run) → RUNNING → DONE | WARNING | ERROR
  ERROR  → (re-run) → RUNNING → DONE | WARNING | ERROR
```

**API:**
```python
class CardStatus(Enum):
    LOCKED  = "locked"
    READY   = "ready"
    RUNNING = "running"
    DONE    = "done"
    WARNING = "warning"
    ERROR   = "error"
    STALE   = "stale"

class CollapsibleCard(QFrame):
    """
    Expandable/collapsible pipeline section card.

    Signals:
        expand_requested(int)   — user clicked to expand (card index)
        collapse_requested(int) — user clicked to collapse (card index)
    """

    def __init__(self, card_index: int, title: str, parent=None): ...

    def set_status(self, status: CardStatus) -> None: ...
    def set_summary(self, html: str) -> None: ...       # collapsed view
    def set_body(self, widget: QWidget) -> None: ...    # expanded view
    def set_advisories(self, advisories: list) -> None: ...
    def is_expanded(self) -> bool: ...
    def expand(self) -> None: ...
    def collapse(self) -> None: ...
```

**Visual layout (expanded):**
```
┌─ ① Title ────────────────────────── [status badge] [▴ collapse] ─┐
│  ┌─ Advisory banner (if any) ──────────────────────────────────┐  │
│  │  [level icon]  Title                                        │  │
│  │  Body text                                  [Action btn]    │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  [body widget — full card content]                                 │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

**Visual layout (collapsed):**
```
┌─ ① Title ─── [summary one-liner] ─── [status badge] [▾ expand] ──┐
└────────────────────────────────────────────────────────────────────┘
```

**Visual layout (locked):**
```
┌─ ④ Export ─────────────────────────────── Locked ───────────────┐
│  Load a manifold first (step ①)                                   │
└────────────────────────────────────────────────────────────────────┘
```

### 9.2 `StepperBar` (`app/widgets/stepper.py`)

Horizontal progress indicator at the top of the pipeline view.

```
① Load  ─────────  ② Index  ─────────  ③ Fill  ─────────  ④ Export
Done ✓              Done ✓              Running             Locked
```

- Clicking a completed step scrolls to that card and expands it.
- Active step label rendered in accent colour.
- Locked step rendered in muted colour; tooltip: "Complete step ② first".

### 9.3 `AdvisoryBanner` (`app/widgets/advisory_banner.py`)

Inline notification widget rendered inside `CollapsibleCard`.

```python
class AdvisoryLevel(Enum):
    INFO    = "info"      # blue — informational, no action needed
    WARNING = "warning"   # amber — something is suboptimal
    ERROR   = "error"     # red — computation failed
    ACTION  = "action"    # purple — user decision required

@dataclass
class AdvisoryAction:
    label: str            # button text
    callback: Callable    # lambda called on click (no arguments)

@dataclass
class Advisory:
    level: AdvisoryLevel
    title: str
    body: str
    actions: list[AdvisoryAction] = field(default_factory=list)

class AdvisoryBanner(QFrame):
    """Renders one Advisory as a styled inline banner."""
```

### 9.4 `MathView` (`app/widgets/math_view.py`)

Carries over `katex.py` from v0.4 with these additions:
- `scroll_to_bottom()` — auto-scroll when new queries are appended
- `set_loading(bool)` — show spinner overlay while computing

### 9.5 `SeriesTable` (`app/widgets/series_table.py`)

A `QTableWidget`-based widget showing accumulated query results.

```
  m      e      Series                    Source
──────────────────────────────────────────────────────────────
  0      0      1 − η₀q^{1/2} + …        computed
  1      0      q^{1/2} − 2q + …         computed
  0      ½      [computing]               —
──────────────────────────────────────────────────────────────
```

Row actions ("Copy LaTeX", "Remove") appear on hover only, right-aligned, in tertiary button style.  The table is otherwise uncluttered.

### 9.6 `SlopeInput` (`app/widgets/slope_input.py`)

Reusable `(P, Q)` integer-pair input with validation:
- Input validation: gcd(|P|,|Q|) must be 1 (for Dehn filling slopes)
- Optional: allow non-coprime (for NC search range inputs)
- Emits `slope_changed(int, int)` signal

---

## 10. App Layer — Pipeline Cards

### 10.1 `PipelineView` (`app/pipeline/pipeline_view.py`)

```python
class PipelineView(QWidget):
    """
    The Calculator mode: scrollable column of four PipelineCards + StepperBar.

    Holds the shared Session instance.
    Coordinates signal/slot wiring between cards and workers.
    """

    def __init__(self, session: Session, parent=None): ...

    # Cards
    _manifold_card: ManifoldCard
    _index_card: IndexCard
    _filling_card: FillingCard
    _export_card: ExportCard
    _stepper: StepperBar

    # Session (shared)
    _session: Session
```

`PipelineView` is the **only class** that wires signals between cards and workers.  It replaces the god-controller `window.py` from v0.4.  It is ~150 lines (vs 282 lines in v0.4) because all computation logic lives in services and workers.

### 10.2 Card ① — `ManifoldCard` (`app/pipeline/manifold_card.py`)

**Purpose:** Load manifold, build NZ, probe cache.

**Input widgets:**
- `QLineEdit` — manifold name (with autocomplete suggestions: m003, m004, …)
- `QSpinBox` — Nmax (= q_order_half / 2), range 4–100, default 10
- `QPushButton` — "Load" (primary)

**Computation:** Runs `LoadWorker` (calls `ComputeService.load_manifold` then `probe_cache`).

**On success:**
1. Session `manifold_data`, `nz_data`, `cache_status` populated
2. Session `active_edges` initialised to `[True] * num_hard`
3. Session `stage` = `LOADED`
4. Advisories generated:
   - `num_hard = 0` → INFO: "All edges easy — showing 3D index"
   - Cache hit(s) → INFO: "I^ref cached at qq=20; NC cached; 75 kernels cached"
5. Card ① summary set to: `"m004 · 2 tet · 1 cusp · 1 hard  ·  cache: I^ref ✓  NC ✓"`
6. Card ① collapses, Card ② unlocks (status → READY)

**On failure:**
- Advisory ERROR: "SnaPy cannot find 'xyz'. Try: m003, m004, 4_1, …"
- All downstream cards remain LOCKED

**Stale handling:** If manifold name changes after Card ② has results, clicking "Load" triggers `session.invalidate_from(LOADED)` and shows a confirm dialog if queries exist.

### 10.3 Card ② — `IndexCard` (`app/pipeline/index_card.py`)

**Purpose:** Compute or load I^ref at user-specified (m,e).

**Sections:**

**A. Mode selector** (radio buttons)
- `● Query` — targeted single query (default)
- `○ Grid` — batch over a range
- `○ From Cache` — instant lookup (greyed if `cache_status["iref"]["available"]` is False)

**B. Refinement section** (hidden if `num_hard = 0`)
- `QComboBox` preset: Full Refined / Unrefined / Custom
- In Custom mode: one `QCheckBox` per hard edge (`η_j: ☑ active`)
- When Weyl check runs and finds incompatible edges, those checkboxes are auto-forced to ☐ and locked (visually different from user-set ☐)

**C. Charge input**
- *Query mode:* Per cusp — `QSpinBox` for m (integer), `QDoubleSpinBox` or custom half-integer spinbox for e
- *Grid mode:* Range spinboxes for m and e; coprime-only checkbox
- *From Cache:* Shows available cache range; user picks (m,e) from the same spinboxes

**D. Compute button**
- Query: "Compute" → launches `IndexWorker` for one (m,e)
- Grid: "Compute Grid" + "Run Weyl Check" (appears after grid done)
- Cache: "Load" (no computation)

**E. Results area** — `SeriesTable` widget
- Each row: (m,e), rendered series, source, action buttons
- Clicking a row's LaTeX button copies to clipboard
- Clicking trash removes the row (and the `IndexQuery` from session)

**F. Weyl status** (below results, in Grid or Cache mode)
- Shows `WeylViewModel` data (a, b vectors, edge_compatible, adjoint check)
- Only visible after `WeylWorker` completes

**Advisories produced:**
- B1 — I^ref = 0 → INFO: "This sector vanishes"
- B2 — cache qq mismatch → ACTION: offer "use cache" vs "compute fresh"
- B3 — Weyl extraction fails → WARNING with explanation
- B4 — edges incompatible → WARNING showing per-edge status, auto-locking toggles
- B5 — adjoint check fails → WARNING with "Proceed anyway" action

**Collapsed summary:**
> `"3 queries · full refined · Weyl: ✓"` or `"5 queries · custom η · Weyl: not run"`

### 10.4 Card ③ — `FillingCard` (`app/pipeline/filling_card.py`)

**Purpose:** Find NC cycles, compute filled refined index.

**Sections:**

**A. Source settings**
```
NC cycles:  (● Compute)  (○ From Cache)   [greyed if no NC cache]
Kernels:    (● Auto)     (○ Compute)      (○ Cache Only)
```
- "Auto" kernel mode: use cached kernel if available at ≥ current qq, else build kernel and cache it.
- "Cache Only": raise advisory if no cached kernel, never compute.

**B. Cusp configuration** (dynamic: r rows)

For each cusp k:
```
Cusp k:  [☑ Fill along]   P = [1]  Q = [0]     ← SlopeInput widget
```
NC search range (shown only for cusps marked as "Fill"):
```
NC search:  P ∈ [-2, 2]   Q ∈ [0, 2]
```

**C. "Find NC Cycles" button** (Phase A)
- Launches `NCSearchWorker` per filled cusp
- Progress shown inline per cusp
- Results populate the NC cycle table

**D. NC Cycles table**
```
Cusp   Cycle         Weyl compat    Source      Action
  0    γ = α         compatible     computed    [Use]
  0    γ = α + 2β    partial        computed    [Use]
```
Weyl compat uses `edge_compatible` from `ABVectors`.
Text values: "compatible" / "partial" / "—" (if Weyl not run).

**E. Filled Index Query** (Phase B, unlocked after Phase A)

```
NC cycle:   [γ = α          ▾]

(for multi-cusp r ≥ 2, unfilled cusps only:)
Cusp 1 (unfilled):   m  [0]     e  [0]

[Compute Filled Index]    [Add query]
```

**F. Results area** — `SeriesTable`
```
  NC cycle    Slope    m_other    e_other    Series                 Source
───────────────────────────────────────────────────────────────────────────
  γ = α       (1, 0)   —          —          1 + (η₀+η₀⁻¹)q + …  computed
  γ = α+2β    (1, 0)   —          —          [computing]           —
───────────────────────────────────────────────────────────────────────────
```

**Advisories produced:**
- C1 — No NC cycles → ACTION: "Widen range" / "Unrefined filling" / "Skip"
- C2 — User slope trivial in NC basis → WARNING
- C3 — Kernel not cached + ℓ ≥ 2 → ACTION with time estimate: "Build & cache" / "Unrefined"
- C4 — NC combo explosion → ACTION: "Compute all N" / "Pick combination"

**When "Unrefined filling" is chosen (C1):**
- Phase A shows: "NC cycles: 0 (unrefined fallback)"
- Phase B shows: Normal slope input, no NC cycle dropdown
- Worker calls `compute_filled_index_unrefined` directly
- Results table shows `[unrefined]` label in source column

**Collapsed summary:**
> `"2 NC cycles · 3 filled queries"` or `"0 NC (unrefined) · 1 filled query"`

### 10.5 Card ④ — `ExportCard` (`app/pipeline/export_card.py`)

**Purpose:** Export session results to files.

**Unlock condition:** `session.has_any_results()` — unlocks as soon as Card ① loads (stage ≥ LOADED).

**Sections:**

**A. Available data summary**
```
☑ Manifold data (m004 · NZ matrix · triangulation)
☑ Refined Index — 3 queries computed
☐ Weyl check — not run (use Grid mode in ②)
☐ Dehn Filling — not computed (step ③)
```
Checkboxes let user select what to include.

**B. Format**
```
☑ LaTeX (.tex)        ☑ Mathematica (.m)
☐ Full Report (.tex)  ☐ JSON (.json)
```

**C. Output path + buttons**
```
Output:  [~/Desktop/m004_export/              ]  [Browse]

[Export]         [Copy LaTeX]    [Copy Mathematica]
```

"Copy LaTeX" and "Copy Mathematica" are tertiary buttons — they copy
the most recent index query result to the clipboard without writing any
file.  All three copy buttons show a brief "Copied." confirmation in
muted text replacing the button label for 1.5 s.

---

## 11. App Layer — Data Hub

### 11.1 `DataHubView` (`app/datahub/datahub_view.py`)

A `QTabWidget` with three sub-tabs: Download | Generate | Export & Share.

### 11.2 Download Tab (`app/datahub/download_tab.py`)

Replaces v0.4's `data_panel.py` with these improvements:
- Shows pack categories (kernels / iref / nc) as section headers
- Displays installed pack details (coverage, qq order) not just "✅ N files"
- "Check for Updates" refreshes from GitHub

### 11.3 Generate Tab (`app/datahub/generate_tab.py`)

**Left pane — Task builder:**

```
Data Type: [Filling Kernels ▾]
           ├─ Filling Kernels
           ├─ I^ref Cache
           └─ NC Cycle Cache

┌─ Parameters (swaps by type) ──────────────────────────┐
│ [Kernels]:                                             │
│   Mode: (● Single slope) (○ Range)                    │
│   P = [1]  Q = [0]  /  P∈[-3,3]  Q∈[0,3]            │
│   qq order: [50]   Coprime only: ☑                    │
│   Skip cached ≥ this order: ☑                          │
│                                                        │
│ [I^ref Cache]:                                         │
│   Census: [m003–m412] (or custom range)                │
│   qq order: [20]   m range: ±[20]   e range: ±[20]   │
│   Skip existing: ☑                                     │
│                                                        │
│ [NC Cache]:                                            │
│   Census: [m003–m412]                                  │
│   qq order: [20]   |P| ≤ [10]   Q ≤ [10]              │
│   Skip existing: ☑                                     │
└────────────────────────────────────────────────────────┘

Workers: [10]   [+ Add to Queue]
```

**Right pane — Task queue + local cache summary:**

```
┌─ Task Queue ───────────────────────────────────────────────────────┐
│  1   Kernels  P∈[-3,3]  Q∈[0,3]  qq=50           running  ████░░  │
│  2   I^ref   m003–m050  qq=20                     queued           │
│  3   NC      m003–m412  qq=20                     queued           │
└────────────────────────────────────────────────────────────────────┘

[Start]   [▶  Resume]   [▐▐  Pause]   [Cancel]   [Clear completed]

┌─ Local Cache ──────────────────────────────────────────────────────┐
│  Type      Count    Size       Last built                          │
│  Kernels     75    354 MB    2026-04-01                            │
│  I^ref      410    128 MB    2026-03-28                            │
│  NC         410     12 MB    2026-03-28                            │
│                                                  [Refresh]         │
└────────────────────────────────────────────────────────────────────┘
```

Pause/Resume: the `GenerateWorker` checks a `threading.Event` between tasks.  Pause sets the event; Resume clears it.  ▶ and ▐▐ are inline text glyphs, not icon widgets.

### 11.4 Export & Share Tab (`app/datahub/export_tab.py`)

```
┌─ Cache Browser ────────────────────────────────────────────────────┐
│  Filter: [All ▾]    [Search                               ]        │
│                                                                     │
│  ☑  kernel_1_0_qq50.pkl.gz      Kernel  (1,0)  qq=50   2.1 MB    │
│  ☑  kernel_1_1_qq50.pkl.gz      Kernel  (1,1)  qq=50   2.1 MB    │
│  ☐  iref_m003_qq20.pkl.gz       I^ref   m003   qq=20   0.3 MB    │
│  ☐  nc_m003_qq20.pkl.gz         NC      m003   qq=20   0.1 MB    │
│                                        [Select all]  [Clear]       │
└─────────────────────────────────────────────────────────────────────┘

┌─ Export Selected ──────────────────────────────────────────────────┐
│  Format:  ☑ Mathematica (.m)   ☐ JSON (.json)   ☐ LaTeX (.tex)   │
│  Output:  [~/Desktop/manifold_export/           ]  [Browse]        │
│                                                  [Export selected] │
└─────────────────────────────────────────────────────────────────────┘

┌─ Publish as Data Pack ─────────────────────────────────────────────┐
│  Pack ID:      [kernels_qq50_custom                    ]           │
│  Pack name:    [Filling Kernels qq=50 (custom)         ]           │
│  Description:  [Pre-computed kernels at qq=50          ]           │
│  Release tag:  [data-v2                                ]           │
│  Target dir:   (● kernel_cache) (○ iref_cache) (○ nc_cache)      │
│                                                                     │
│  [Create archive]                                                  │
│  → dist/kernels_qq50_custom.tar.gz  (354 MB  ·  SHA-256 verified) │
│  → data_packs.json updated                                         │
│  Next: upload to GitHub Releases as a release asset.               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 12. Edge Case & Advisory System

Every exceptional mathematical or computational situation is handled via the `Advisory` system rather than modal dialogs or silent fallbacks.

### 12.1 Category A — Manifold Properties (Card ①)

| ID | Trigger | Advisory level | Title | Body | Actions |
|---|---|---|---|---|---|
| A1 | `num_hard == 0` | INFO | "All edges easy" | "Refined index equals the ordinary 3D index. No η variables." | — (informational only) |
| A2 | SnaPy raises `KeyError` / `RuntimeError` | ERROR | "Manifold not found" | "SnaPy cannot find '{name}'. Check spelling." | [Suggest examples] |
| A3 | SnaPy shapes degenerate | WARNING | "Degenerate triangulation" | "Shape parameters may be unreliable. Consider retriangulating." | [Proceed anyway] |

### 12.2 Category B — Refined Index (Card ②)

| ID | Trigger | Level | Title | Body | Actions |
|---|---|---|---|---|---|
| B1 | `result == {}` | INFO | "Sector vanishes" | "I^ref(m,e) = 0 at qq={qq}. May be exact or truncation artifact." | — |
| B2 | Cache qq < requested qq | ACTION | "Cache qq mismatch" | "Cache at qq={cache_qq}, requested qq={req_qq}." | [Use cache (qq={cache_qq})] [Compute fresh] |
| B3 | `run_weyl_check` returns `None` | WARNING | "Weyl extraction failed" | "Cannot extract (a,b) vectors. Filling proceeds without Weyl correction." | — |
| B4 | Some `edge_compatible[j] == False` | WARNING | "Partial Weyl incompatibility" | "Edges {list} incompatible with half-integer e. Will be projected out for filling." | — |
| B5 | Adjoint su(2) value ≠ −1 | WARNING | "Adjoint check failed" | "½(c₋₁+c₊₁−c₋₂−c₊₂) = {val} ≠ −1. Refined filling may be inconsistent." | [Proceed anyway] [Increase qq] |

### 12.3 Category C — Dehn Filling (Card ③)

| ID | Trigger | Level | Title | Body | Actions |
|---|---|---|---|---|---|
| C1 | `len(result.cycles) == 0` | ACTION | "No non-closable cycles found" | "Every slope in the search range is closable. Refined filling requires an NC cycle as basis." | [Widen search range] [Unrefined filling] [Skip filling] |
| C2 | Transformed slope `p = 0` | WARNING | "Trivial slope in NC basis" | "The user's slope maps to the NC cycle itself (p=0). Filling along it gives zero by definition." | — |
| C3 | HJ chain ℓ ≥ 2 + no cached kernel | ACTION | "Kernel not cached" | "Slope ({P},{Q}) → HJ chain of length {ℓ}. Estimated build time: ~{t}s." | [Build & cache kernel] [Use unrefined kernel] |
| C4 | `len(nc_combos) > 4` (multi-cusp explosion) | ACTION | "Many NC combinations" | "{r} cusps × {n} NC cycles each = {total} combinations. Each requires a filling computation." | [Compute all {total}] [Pick specific combination] |
| C5 | Slope `gcd(|P|,|Q|) ≠ 1` | ERROR | "Non-primitive slope" | "(P,Q) = ({P},{Q}) is not coprime. Dehn surgery requires a primitive cycle." | — |
| C6 | Kernel computation cancelled | INFO | "Build cancelled" | "Kernel build was cancelled. Previous cache state is unaffected." | [Retry] [Use unrefined] |

### 12.4 Category D — Cross-Card State

| ID | Trigger | Level | Title | Body | Actions |
|---|---|---|---|---|---|
| D1 | Manifold changed after ② has results | ACTION | "Manifold changed" | "All results for '{old}' will be cleared." | [Confirm] [Cancel] |
| D2 | qq order changed after ③ has filling results | WARNING | "qq order changed" | "Filling results were at qq={old}. May be insufficient at qq={new}." | [Keep results] [Recompute] |
| D3 | Session restored with lower qq than requested | WARNING | "Restored session" | "Loaded session computed at qq={saved}. Current setting is qq={now}." | [Keep saved] [Recompute] |

### 12.5 Advisory Rendering Rules

1. Advisories are **rendered in order** within their card, above the results area.
2. ERROR advisories **always show**, even when the card is collapsed (shown in the summary line as "✗ …").
3. ACTION advisories render their action buttons as `QPushButton` (secondary style) inline in the banner.
4. Clicking an action button fires the stored `callback` and dismisses the advisory.
5. Advisories are re-generated each time the ViewModel is rebuilt (stateless).

---

## 13. Implementation Phases

### Phase 0 — Project Skeleton

**Files to create:**
- `v0.5/pyproject.toml` (copy + update version to 0.5.0)
- `v0.5/setup.py`
- `v0.5/STATUS.md`
- All `__init__.py` files for `services/`, `viewmodels/`, `app/`, `app/theme/`, `app/widgets/`, `app/pipeline/`, `app/datahub/`, `app/workers/`, `formatters/`
- Copy `core/` verbatim from `v0.4/src/manifold_index/core/`
- Copy `utils/` verbatim
- Copy `data/` verbatim

**Test:** `python -c "from manifold_index.core.manifold import load_manifold; print('ok')`

---

### Phase 1 — Session Model

**Files:**
- `services/session.py` — full `Session`, `PipelineStage`, `IndexQuery`, `FillQuery`, `NCCycleSet`
- `services/__init__.py`
- `tests/test_session.py`

**Test cases:**
- Construct empty session
- `invalidate_from(LOADED)` clears all downstream data
- `invalidate_from(INDEXED)` leaves manifold data intact
- `has_any_results()` correct at each stage
- `session_to_dict` / `session_from_dict` roundtrip

---

### Phase 2 — Service Layer

**Files:**
- `services/compute_service.py`
- `services/filling_service.py`
- `services/export_service.py`
- `services/datahub_service.py`
- `tests/test_compute_service.py`
- `tests/test_filling_service.py`

**Test cases (compute_service):**
- `load_manifold("m004")` returns `(ManifoldData, EasyEdgeResult, NeumannZagierData)`
- `load_manifold("nonexistent")` raises `ValueError`
- `probe_cache("m004", nz)` returns correct dict structure
- `compute_refined_index(nz, [0], [0], 20)` matches v0.4 result
- `project_refined_index(result, [True, False])` correctly sums over η₁ dimension
- `project_refined_index(result, [True])` == identity
- `project_refined_index(result, [False])` == `project_to_3d_index(result)`

**Test cases (filling_service):**
- `find_nc_cycles` for m004 at default range returns known NC cycle (1,0) or (0,1)
- `canonicalise_nc_cycles` deduplicates opposite-sign pairs
- `probe_kernel` returns correct `hj_length` for (1,0), (2,1), (3,2)
- `compute_filled_index` for m004 (1,0) matches v0.4 result

---

### Phase 3 — Advisory & ViewModel Layer

**Files:**
- `viewmodels/advisory.py` — `Advisory`, `AdvisoryAction`, `AdvisoryLevel`, `CardStatus`
- `viewmodels/manifold_vm.py`
- `viewmodels/index_vm.py` (+ `WeylViewModel`)
- `viewmodels/filling_vm.py`
- `viewmodels/export_vm.py`
- `tests/test_viewmodels.py`
- `tests/test_advisory.py`

**Test cases:**
- `ManifoldViewModel` for num_hard=0 → `has_hard_edges=False`, `index_title="3D Index"`, advisory A1 present
- `ManifoldViewModel` for num_hard=1 → `has_hard_edges=True`, no A1 advisory
- Advisory A2 triggers when load fails
- `IndexQueryViewModel` for zero result → `is_zero=True`, advisory B1 present
- `WeylViewModel` for None weyl → advisory B3 present
- `WeylViewModel` with incompatible edge → advisory B4 present
- `FillingViewModel` for empty NC list → advisory C1 present

---

### Phase 4 — Formatters (split from v0.4 `formatters.py`)

**Files:**
- `formatters/manifold_fmt.py` — NZ matrix, gluing table, easy/hard edge HTML
- `formatters/index_fmt.py` — series HTML from `RefinedIndexResult` / projected result
- `formatters/weyl_fmt.py` — ABVectors display, edge compatibility table
- `formatters/filling_fmt.py` — NC cycle table, filled series HTML

**Test:** Each formatter produces valid HTML/LaTeX strings for known inputs.  No reference to Qt.

---

### Phase 5 — Theme & Core Widgets

**Files:**
- `app/theme/style.py` — new QSS (cleaner, HiDPI-aware, dark/light palette)
- `app/theme/colors.py`
- `app/theme/icons.py`
- `app/widgets/advisory_banner.py`
- `app/widgets/collapsible_card.py`
- `app/widgets/stepper.py`
- `app/widgets/math_view.py` (port from v0.4 `katex.py` + new methods)
- `app/widgets/series_table.py`
- `app/widgets/slope_input.py`

**Manual test:** Run a standalone `QApplication` that shows all widget states (locked, ready, running, done, warning, error, stale).

---

### Phase 6 — Workers

**Files:**
- `app/workers/load_worker.py` → calls `ComputeService.load_manifold` + `probe_cache`
- `app/workers/index_worker.py` → calls `ComputeService.compute_refined_index`
- `app/workers/weyl_worker.py` → calls `ComputeService.run_weyl_check`
- `app/workers/nc_search_worker.py` → calls `FillingService.find_nc_cycles`
- `app/workers/fill_worker.py` → calls `FillingService.compute_filled_index`
- `app/workers/generate_worker.py` → calls `DataHubService.build_kernels` / `build_iref_cache` / `build_nc_cache`
- `app/workers/download_worker.py` → calls `DataHubService.download_pack`

**Each worker:**
- Inherits `QThread`
- Emits: `status: Signal(str)`, `progress: Signal(int, int)`, `finished: Signal(object)`, `error: Signal(str)`
- `run()` calls exactly one service method, wraps in try/except, emits `error` on failure
- Body of `run()` is ≤ 20 lines

---

### Phase 7 — Pipeline Cards

**Files:**
- `app/pipeline/pipeline_view.py`
- `app/pipeline/manifold_card.py`
- `app/pipeline/index_card.py`
- `app/pipeline/filling_card.py`
- `app/pipeline/export_card.py`

**Implementation order:** manifold_card → index_card → filling_card → export_card → pipeline_view (wiring).

**Key rules:**
- Cards hold **no session state** in instance variables — always read from `Session`
- Cards emit signals upward; `PipelineView` intercepts and updates `Session`
- Cards read `Session` to rebuild their ViewModel on each update
- `PipelineView._on_session_updated()` triggers ViewModel rebuild → card display refresh

---

### Phase 8 — Data Hub

**Files:**
- `app/datahub/datahub_view.py`
- `app/datahub/download_tab.py`
- `app/datahub/generate_tab.py`
- `app/datahub/export_tab.py`

---

### Phase 9 — Main Window

**Files:**
- `app/window.py` — `MainWindow` with top bar (mode toggle) + `QStackedWidget` (Calculator / Data Hub)
- `app/__main__.py` — `launch_gui()`

**`MainWindow` is ≤ 80 lines** — it only:
1. Creates `QStackedWidget` with `PipelineView` and `DataHubView`
2. Creates top bar with mode toggle buttons
3. Passes a shared `Session` to `PipelineView`
4. Applies stylesheet

---

### Phase 10 — Polish

- Session save/restore to `~/.manifold_index_sessions/`
- Keyboard shortcuts: `Cmd+1`–`Cmd+4` to expand cards; `Cmd+Enter` to compute
- HiDPI / macOS Retina fixes for KaTeX math view
- Dark mode support (palette-based stylesheet from v0.4 extended)
- Manifold name autocomplete (census list from SnaPy)
- `closeEvent` cleanup (stop all workers, wait 2s each)

---

## 14. Testing Strategy

### 14.1 Unit Tests (pytest, no Qt)

All `services/` and `viewmodels/` code is tested without launching a `QApplication`.

```
tests/
    test_session.py          Phase 1
    test_compute_service.py  Phase 2
    test_filling_service.py  Phase 2
    test_viewmodels.py       Phase 3
    test_advisory.py         Phase 3
    test_formatters.py       Phase 4
```

### 14.2 Known-Good Values

Use v0.4 test suite as oracle.  For every `core/` function, the v0.5 service must produce bit-identical results.

Reference values (from v0.4 passing tests):
- `m004`, (m=0,e=0), qq=20: `I^ref = {(0,): 1, (2,-2): -1, (2,0): -1, (2,2): -1, ...}` (exact dict from v0.4)
- `m004` NC cycles at |P|≤2, Q∈[0,2]: `{(1,0), (0,1)}` (canonical)
- Weyl vectors for m004: `a=[2], b=[1/2]`

### 14.3 Widget Smoke Tests

Each widget module has a `if __name__ == "__main__":` block that runs a standalone `QApplication` showing the widget in all states.  These are run manually, not in CI.

---

## 15. Build & Packaging

Copy `v0.4/ManifoldIndex.spec`, `v0.4/build_app.sh`, `v0.4/rthook_snappy.py` and update:
- App name: `"Refined Index Calculator"`
- Version string in `pyproject.toml`: `0.5.0`
- Entry point: `manifold_index.app:launch_gui`

No changes needed to PyInstaller spec beyond the name/version update.

---

## 16. Design Decisions Log

| Decision | Rationale | Alternative considered |
|---|---|---|
| Pipeline cards (not sidebar) | Research workflow needs previous results visible; scrolling is natural | Sidebar hides context behind page switches |
| Cards are not pages | Each card can be expanded independently; no navigation | Wizard pages force linear progression |
| Two top-bar modes (not more tabs) | Calculator and Data Hub are truly orthogonal | Could have been 5 sidebar items |
| Query mode as default in Card ② | Most research queries target specific (m,e); full grid is expensive | Grid as default (v0.4 behaviour) |
| Projection as post-processing | No re-computation needed when toggling η; core always runs full | Separate `compute_index_3d_python` call for unrefined |
| Two-phase Card ③ (NC search ≠ filling) | NC cycles are informative independently; decoupling simplifies workers | Single monolithic operation (v0.4 approach) |
| Advisory banners (not modal dialogs) | Non-blocking; user can dismiss or act at their own pace | `QMessageBox.critical` blocks the UI |
| Service layer with zero Qt | Independently testable; can be used from CLI or future web API | Logic in workers (v0.4 approach) |
| Session as single source of truth | Eliminates cross-panel state, eliminates "stale results" bugs | State in each panel (v0.4 approach) |
| One QThread per async task | Easy to test, cancel, and reason about | Monolithic workers (v0.4 approach) |
| Formatters split by domain | Each formatter is ~200 lines, focussed, testable | One 1200-line formatters.py (v0.4 approach) |
| Data Hub replaces Kernel panel + Data panel | All data operations belong together; exposes CLI-only tools | Keep two separate tabs |
| `CollapsibleCard` as single reusable widget | All four pipeline cards share one codebase; consistent UX | Custom layout per card |

| No decorative icons or emoji | Research instrument aesthetic; icons only where glyph conveys more than a word | Icon-heavy toolbar (v0.4 approach) |
| Text-only buttons | Label is unambiguous; no icon-to-meaning mapping to memorise | Icon + label buttons |
| Colour carries only state | Every colour difference is meaningful; no decoration | Colour used decoratively |
| Single accent colour on neutral base | Visually quiet; the mathematics is the content | Multi-colour accent scheme |
