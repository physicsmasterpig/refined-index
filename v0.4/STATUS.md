# Manifold Index v0.4 — Implementation Status

## Spec Status

All 16 phase specification files are **complete**.
Each is self-contained: to implement phase N, an LLM needs only
`CONVENTIONS.md` + `phase_NN_*.md`.

## Progress Tracker

| Phase | Description | Spec | Code | Files |
|:------|:-----------|:-----|:-----|:------|
| 0 | Project Skeleton | ✅ | ⬜ | pyproject.toml, __init__.py files |
| 1 | Manifold Loading | ✅ | ⬜ | core/manifold.py |
| 2 | Gluing Equation Reduction | ✅ | ⬜ | core/gluing_equations.py |
| 3 | Phase Space Basis | ✅ | ⬜ | core/phase_space.py |
| 4 | Neumann-Zagier Matrix | ✅ | ⬜ | core/neumann_zagier.py |
| 5 | Basis Selection | ✅ | ⬜ | core/basis_selection.py |
| 6 | 3D Index | ✅ | ⬜ | core/index_3d.py |
| 7 | Refined Index | ✅ | ⬜ | core/refined_index.py |
| 8 | Weyl Checks | ✅ | ⬜ | core/weyl_check.py |
| 9 | Dehn Filling | ✅ | ⬜ | core/dehn_filling.py |
| 10 | Refined Dehn Filling | ✅ | ⬜ | core/refined_dehn_filling.py |
| 11 | Kernel Cache | ✅ | ⬜ | core/kernel_cache.py |
| 12 | C Extension | ✅ | ⬜ | core/_c_kernel/tet_index.c |
| 13 | Export Infrastructure | ✅ | ⬜ | utils/exporters.py |
| 14 | GUI | ✅ | ⬜ | app/ directory |
| 15 | Build & Packaging | ✅ | ⬜ | build scripts |

## Legend
- ⬜ Not started
- 🟡 In progress
- ✅ Complete
- ❌ Blocked

## Implementation Order

Phases should be implemented sequentially (0 → 15).  Each phase's tests
must pass before proceeding.  Dependencies:

```
0 (skeleton)
├─ 1 (manifold)
│  ├─ 2 (gluing)
│  │  ├─ 3 (phase space) ──┐
│  │  └─ 5 (basis sel.)    │
│  └───────────────────────┤
│                          ▼
│                    4 (NZ matrix)
│                    ├─ 6 (3D index)
│                    │  └─ 7 (refined index)
│                    │     ├─ 8 (Weyl check)
│                    │     └─ 9 (Dehn filling)
│                    │        └─ 10 (refined Dehn filling)
│                    │           └─ 11 (kernel cache)
│                    └─ 12 (C extension — optional, speeds up 6)
│
├─ 13 (export — after 4–10)
├─ 14 (GUI — after all core + 13)
└─ 15 (build — after all)
```

## Notes

### Design Decisions
- **Refined index notation:** Adopted article notation η^{2W_a e_{int,a}}
  (single η, formal weight variables W_a) instead of separate η_a per hard
  edge.  Data structure unchanged: keys store per-edge doubled exponents
  `(qq, 2*e_int_0, …, 2*e_int_{k-1})`.  W_a are evaluation-time parameters.

- Each phase should be implemented with its tests passing before moving on
- Per-phase specs in `phases/` are the primary reference (not BLUEPRINT.md)
- Use `pytest tests/test_<module>.py` to validate each phase

### Token Budget Guide
- CONVENTIONS.md: ~140 lines
- Phase specs: 80–600 lines each
- To implement phase N, load only: CONVENTIONS.md + phase_NN_*.md
- Estimated context per phase: 200–750 lines (vs ~1500 for full BLUEPRINT)
