# Legacy Blackhole Runtime Architecture

## Status

- **Status**: archived / legacy
- **Authority**: not authoritative for new implementation work
- **Superseded by**: `tasks/dev_design/final_blackhole_backend_redesign.md`

This document preserves the previously mixed Blackhole runtime/compiler architecture narrative so that historical decisions remain readable after the main design was rewritten around the layered IR architecture.

## 1. What This Legacy Architecture Described

The earlier architecture centered on the already-working direct runtime path:

```text
TileLang DSL
  -> PrimFunc / TIR
  -> LowerTileOp / split-before planning
  -> SplitHostDevice
  -> SplitBlackholeKernel
  -> LowerBlackholeOps
  -> PlanBlackholeCB
  -> AssignBlackholeCores
  -> rt_mod_blackhole
  -> Extract ExecutableSpec
  -> BlackholeModule direct host path
```

Its main focus was:

1. make `ExecutableSpec` the main backend product
2. stabilize `blackhole.*` attrs and direct host materialization
3. bring up copy, GEMM, and multi-core direct runtime

That work remains historically important, and much of it is still retained as migration substrate.

## 2. What Was Valuable And Is Still Retained

The following outcomes from the legacy architecture remain real and are not discarded:

- `ExecutableSpec` as the final materialized runtime product
- `BlackholeModule` in-process direct host path as the official execution path
- `LowerBlackholeOps -> PlanBlackholeCB -> AssignBlackholeCores -> rt_mod_blackhole` as the current implementation chain
- copy / GEMM / multi-core direct path bring-up
- `blackhole.cb_requirements` and `blackhole.cb_configs` based planning protocol

These pieces are now treated as migration-era components that must be re-owned by cleaner layers.

## 3. Why The Legacy Architecture Was Superseded

The old architecture document mixed together four different things:

1. current runtime execution structure
2. near-term pass ownership
3. future compiler architecture
4. historical stage-by-stage bring-up notes

That was acceptable while the main task was getting copy/GEMM/direct runtime real. It became insufficient once flash-attn exposed a deeper compiler problem:

- algorithm semantics were being recovered too late
- spatial/dataflow program structure had no first-class representation
- TT target planning was mixed with semantic recovery

The current layered design was introduced because the old structure could not cleanly answer:

- what the algorithm means
- how the algorithm becomes a spatial program
- how that spatial program becomes a TT contract

## 4. The Old "Stateful Tiled IR" Direction

Before the layered rewrite, the next-stage direction was framed as a single compiler-internal `Stateful Tiled IR`.

That direction was superseded because one layer would have had to carry all of the following simultaneously:

- algorithm semantics
- task/channel/layout/sync structure
- target-sensitive carry and mapping decisions

The layered rewrite replaced it with:

```text
Stateful Semantic IR
  -> Spatial Program IR
  -> TT Target IR
```

The historical implementation-plan draft remains here:

- `tasks/dev_design/2026-04-02-stateful-tiled-ir-phase1-implementation-plan.md`

That file should now be read only as a historical Phase A semantic-layer draft, not as the top-level architecture.

## 5. How To Read This Legacy Document

Use this legacy note only when you need one of the following:

1. historical context for Stage 0-3 execution-path decisions
2. rationale behind current `ExecutableSpec` or direct runtime boundaries
3. context for how `LowerBlackholeOps`, `PlanBlackholeCB`, `AssignBlackholeCores`, and `rt_mod_blackhole` grew into their present form

Do not use this document to define new IR objects, new pass boundaries, or new implementation plans.
