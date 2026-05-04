# TileLang Blackhole Backend Workspace

这是 TileLang Tenstorrent Blackhole 后端的主开发工作区。

## Authoritative Entrypoints

- Overall design:
  `tasks/dev_design/final_blackhole_backend_redesign.md`
- Current execution status / blocker / next task:
  `tasks/progress.md`
- Active design index:
  `tasks/dev_design/README.md`
- Historical protocol audit:
  `tasks/dev_design/blackhole_first_principles_protocol_audit.md`
- Working norms:
  `AGENTS.md`
  / `CLAUDE.md`
  / `GEMINI.md`
- Stable experience and bug memory:
  `memory/general_dev.md`
  / `memory/bugs.md`

Do not use `tasks/dev_design/archive/`
as current design input.

## Status Policy

Current execution state is intentionally not duplicated here.
Read `tasks/progress.md`.

The durable architecture is:

```text
Normalized Tile TIR
  -> SpatialPlan
  -> TTProgram
  -> ExecutableSpec
```

Active task, blockers, next steps, and latest verification are maintained only
in `tasks/progress.md`.

## Recommended Reading Order

1. `tasks/dev_design/final_blackhole_backend_redesign.md`
2. `tasks/dev_design/task0_ir_layering_root_cause.md`
3. `tasks/dev_design/task1_spatial_plan_companion.md`
4. `tasks/dev_design/task2_ttprogram_companion_cutover.md`
5. `tasks/dev_design/task3_runtime_gate_and_workload_cutover.md`
6. `tasks/progress.md`
7. `tasks/dev_design/README.md`
8. Active design docs listed in `tasks/dev_design/README.md`
9. `memory/general_dev.md`
10. `memory/bugs.md`
11. Relevant code and tests

## Repository Layout

- `tilelang_repo/`:
  TileLang development checkout; Blackhole implementation lives mostly here.
- `tt_metal_repo/`:
  TT-Metal checkout; API, runtime, simulator, and examples reference.
- `tasks/`:
  design contracts, progress board, and archived task history.
- `memory/`:
  stable engineering lessons and reusable bug records.
- `scripts/`:
  environment setup and helper scripts.

## Documentation Rules

- Do not add a second overall design document.
- Keep current status only in `tasks/progress.md`.
- Keep design docs as contracts, not chronological notebooks.
- Put durable lessons in `memory/`.
- If docs and code diverge, update the relevant design/status first, then
  continue implementation.
