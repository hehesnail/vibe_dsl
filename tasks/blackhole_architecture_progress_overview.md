# Blackhole Architecture And Progress Overview

## Role

This document is a compact natural-language overview of the current Blackhole
backend architecture and completion state.  It is intended for project
orientation and for generating an external architecture/progress illustration.

The authoritative design remains
`tasks/dev_design/final_blackhole_backend_redesign.md`.
The live execution board remains `tasks/progress.md`.

## Image Brief

Show the project as a compiler pipeline that has moved away from fragile late
matchers and toward an explicit four-layer IR contract.  The main horizontal
flow is:

`Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec -> Blackhole runtime/codegen`.

The visual should emphasize that semantics move forward through typed
representations and validators.  Runtime and codegen sit at the far right as
leaf consumers.  They consume the executable projection and do not look back
into source text, final TIR bodies, names, argument positions, or builtin
neighborhoods to recover missing planning facts.

## Architecture Summary

The root problem was that target execution semantics escaped the IR chain.
Earlier implementations could make individual cases pass by carrying hidden
truth through attrs, helper maps, payloads, naming conventions, body scans, or
runtime fallbacks.  That did not scale to complex frontend workloads.

The current architecture makes each stage own one level of meaning:

`Normalized Tile TIR` owns authored algorithmic structure, explicit tile ops,
buffer loads/stores, predicates, loops, access expressions, and leaf TT-Metal
compute normalization when the primitive can be expressed.

`SpatialPlan` owns target-independent virtual spatial and dataflow semantics:
execution units, dataflow/carry/reduction/broadcast/join relations, logical
live values, materialization boundaries, access-region evidence, and validated
hints.

`TTProgram` owns Blackhole target realization.  This is the stable target
execution contract: hardware model facts, placement, core groups, kernel
roles, buffer distributions, compute op plans, CB plans, semaphore and sync
plans, runtime/per-work ABI, launch ordering, resource pressure, exact-CB
lifecycle, and typed kernel queue events.

`ExecutableSpec` owns the leaf projection and backend admission contract.
Runtime and codegen consume this representation directly.  If the spec lacks a
required record, leaf readers fail closed with a typed reason instead of
reconstructing planner semantics.

## Contract Spine

The most important current contract is that `TTProgram` is the target-facing
execution contract.  Physical CB queue order is represented by
`TTKernel.queue_events`, then projected to `KernelSpec.queue_events`.
Exact-CB lifecycle is represented by typed virtual values, use events, live
intervals, allocations, and release events.  Remote synchronization endpoints
are represented by explicit remote core descriptor records.  Per-work dynamic
values are represented by generic `TTPerWorkArgSpec` records with
`value_source=value_expr` and optional `AccessRegion` evidence.

The retired surfaces are part of the story: generated source text, final-body
scans, `blackhole.segment_kind`, top-level payloads, facts bags, workload-shaped
schemas, runtime-arg name prefixes, CB-name suffixes, and fallback defaults are
not current protocol.

## Completion State

The foundation lanes are complete.  Buffer ABI, leaf compute/GEMM, tensor
placement and sharding, external accessors, topk value/index selection,
materialization, exact-CB lifecycle, and current non-workload direct-runtime
paths all use typed records or fail closed.

The P0 target execution contract hardening lane is complete.  Covered target
execution facts are now represented by `TTProgram` typed fields or objects and
projected once into `ExecutableSpec`.  Source/runtime/codegen recovery from
body, source, names, or argument positions is guarded against.

The T8 irregular/indexed access lane is complete.  Indexed, sparse, ragged,
paged, segmented, and grouped-feed access patterns use generic TIR-derived
`AccessRegion` plus `value_expr` evidence.  Workload-shaped fields such as
`index_table_*`, `row_start`, `row_count`, or `page_index` are not public
schema.

The active lane is P1 / T9 workload-first paths.  T9.1 through T9.5 are
admitted on the current bf16 direct-runtime surfaces: grouped GEMM, paged GQA,
paged MLA, sparse/ragged GQA, and chunk recurrence / scan.  The active boundary
is T9.6 multi-block flash decode, where split blocks must use explicit exact-CB
publish/consume and partial-combine contracts.

The queued lane is P2 / T10 distributed production.  Mesh placement,
collectives, NoC/multicast/global scheduling, distributed workload correctness,
and production partial-K reduction remain future target-realization work.

## Suggested Visual Emphasis

Use a clean layered compiler diagram.  The left side should feel like
high-level program meaning, the middle like typed planning, and the right side
like executable runtime admission.  The strongest visual contrast should be
between the bright main contract path and faded retired side channels.

The completed lanes can be shown as solid completed bands beneath the main
pipeline: Foundation, P0, and T8.  The active lane should be highlighted at
P1 / T9.6 multi-block flash decode.  The future lane should sit further right
or below as P2 / T10 distributed production.

If the illustration uses callouts, the key callouts should be:

- IR-first owner truth.
- Validators fail closed before source/runtime emission.
- `TTProgram` is the Blackhole target execution contract.
- `ExecutableSpec` is a leaf projection, not a recovery pass.
- Runtime/codegen do not scan source or final TIR to rebuild planner facts.

## Short Caption

TileLang Blackhole backend is converging on an explicit compiler contract:
Normalized Tile TIR captures algorithmic tile semantics, SpatialPlan captures
virtual dataflow, TTProgram captures Blackhole execution protocol, and
ExecutableSpec feeds runtime/codegen.  Completed foundation, P0, and T8 lanes
have removed major recovery paths; current work is T9.6 multi-block flash
decode, with distributed production variants queued for T10.
