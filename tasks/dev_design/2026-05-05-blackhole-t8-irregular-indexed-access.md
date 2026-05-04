# Blackhole T8 Irregular Work Domains And Indexed Access

## Role

This document defines the task-level design for T8 irregular work domains and
indexed access in the Blackhole backend.

It is not a second overall design document.
The durable chain remains:

```text
Normalized Tile TIR
  -> SpatialPlan
  -> TTProgram
  -> ExecutableSpec
```

Current execution status lives in `tasks/progress.md`.

## Problem

T1-T7.5 admitted regular tensor, sharded, page-indexed, topk, and exact-CB
surfaces.  T8 is the next boundary: workload witnesses such as routed/grouped
GEMM, paged decode, sparse/ragged attention, and indexed block traversal need
non-uniform work and address evidence.

The missing semantic object is not a workload registry.
Names such as `group_sizes`, `group_offsets`, `cache_seqlens`, or
`block_indices` are examples of operands that may appear in TIR.  They are not
protocol owners.

The backend must derive irregular work and indexed access from:

- TIR loop domains and launch axes;
- TIR predicates;
- `BufferLoad` / `BufferStore` index expressions;
- explicit runtime operands only when those operands participate in the above
  expressions.

If that structure is absent, the backend must reject.  It must not recover
semantics from workload names, buffer names, Python-side metadata, or generated
source text.

## Goal

Represent TIR-derived irregular/indexed evidence explicitly and make the active
backend consume it:

```text
Normalized Tile TIR access / predicate / loop evidence
  -> SpatialPlan AccessRegion indexed and guarded evidence
  -> TTProgram per-work / indexed-access descriptors
  -> ExecutableSpec source/runtime addressing records
  -> BlackholeModule direct runtime consumes those records
```

T8 is complete only when the admitted first surfaces use this evidence to drive
source/runtime addressing, and unsupported forms fail closed before source or
runtime guessing.

## Non-Goals

- No frontend `T.irregular`, `T.block_indices`, `T.ragged`, `T.grouped_gemm`,
  or workload-specific op.
- No workload metadata registry.
- No source-name, buffer-name, argument-position, or generated-source recovery.
- No claim of full MoE, paged decode, sparse attention, or distributed
  production correctness; those are T9/T10 workload lanes.
- No new long-lived IR layer outside `SpatialPlan`, `TTProgram`, and
  `ExecutableSpec`.

## Representation Contract

### Normalized Tile TIR

Owns the source evidence:

- loop and launch domains;
- predicates guarding reads/writes;
- index expressions on `BufferLoad` / `BufferStore`;
- explicit operands used inside those expressions.

It does not own TT runtime arg layout, physical worker assignment, or direct
runtime packetization.

### SpatialPlan

`AccessRegion` becomes the target-independent owner of indexed access evidence.

Required fields for indexed/guarded regions:

- `subject`: accessed buffer;
- `unit_name` / `unit_index`: producing or consuming execution unit;
- `access_kind`: read, write, or read_write;
- `loop_vars`: loop or launch variables that participate in the region;
- `index_exprs`: one expression per logical dimension, derived from the actual
  TIR access;
- `lower_bounds`, `extents`, and `strides`: conservative region extent;
- `coverage_kind`: `full`, `slice`, `row_slice`, or `grouped_slice`;
- `predicate_kind`: `unconditional`, `guarded`, or `unknown`.

`AccessRegion` must not be just a dump.  It must be validated and consumed by
TT planning for the admitted subset.

### TTProgram

TTProgram owns target realization of indexed work/addressing.

The first implementation reuses and tightens `TTPerWorkArgSpec` for simple
per-work tile descriptors, while adding an explicit reference back to the
SpatialPlan `AccessRegion` evidence when the descriptor is derived from an
indexed region.

Later T8 slices may add focused objects for:

- segmented ranges;
- ragged bounds;
- block-index table traversal.

Those objects must still point back to `AccessRegion` / predicate evidence
rather than to workload names.

### ExecutableSpec

ExecutableSpec projects the selected per-work and indexed descriptors.

Runtime/source may consume projected descriptors, but must not recompute:

- logical block axes from source text;
- tile starts from raw `work_linear_id` when a stronger descriptor exists;
- ragged bounds or index-table traversal from argument names.

## First Implementation Slice

The first T8 slice is deliberately narrow:

1. `AccessRegion` records concrete `BufferLoad` / `BufferStore` index
   expressions and whether the access is guarded.
2. `ValidateSpatialPlan` rejects malformed indexed regions.
3. `TTPerWorkArgSpec` / executable projection preserve a link to the source
   `AccessRegion` for per-work tile descriptors derived from indexed access.
4. Codegen/runtime continue to consume the projected per-work descriptors; the
   descriptor is no longer an untraceable source-emitter default.
5. A direct-runtime grid-indexed copy remains the positive gate because it is
   already a real multi-work item addressing path.

This slice is not the whole T8 task.  It establishes the evidence chain that
segmented, ragged, and table-indexed cases must use.

## Later T8 Slices

Segmented/grouped dispatch:

- derive non-uniform group starts/counts from TIR range and address
  expressions;
- operands such as `group_sizes` or `group_offsets` are evidence only if read
  by those expressions;
- source/runtime descriptors must carry the selected segment start/count.

The admitted first segmented/grouped slice is a row-segment staged copy:

- a one-dimensional `int32` table load used in the source address row
  expression becomes a per-work `segment_row_start` descriptor;
- a one-dimensional `int32` table load used in the guarded copy predicate
  becomes a per-work `segment_row_count` descriptor;
- the device source consumes runtime args derived from those descriptors and
  must not emit raw source reads from the tables;
- the copied output is a compact per-work tile-sized block, so invalid rows
  are zero-filled inside that block and cannot clobber a neighboring segment;
- this slice admits non-uniform segment starts/counts only for a single
  page-indexed bf16 row-copy surface.  Wider grouped GEMM dispatch is a later
  workload path, but must reuse the same descriptor/evidence contract.

Ragged bounds:

- derive valid row/token bounds from TIR predicates;
- operands such as `cache_seqlens` are evidence only if used in the predicate
  or index expression;
- invalid rows/tokens must be skipped in source/runtime.

Indexed block traversal:

- derive table-driven block traversal from `BufferLoad` / `BufferStore` index
  expressions;
- operands such as `block_indices` are evidence only if they are read and used
  to form a memory address;
- source/runtime must consume the projected table descriptor.

The admitted first table-indexed form is a per-work tile descriptor whose
tile start is read from a one-dimensional `int32` index table at
`work_linear_id`.  The table buffer and value scale are part of
`TTPerWorkArgSpec`; source code consumes the normal tile-start runtime arg and
direct runtime evaluates that arg from the projected table descriptor.  The
device source must not emit a raw `BufferLoad` from the index table to recover
the tile id.

## Validation Plan

Structure:

- `AccessRegion` tests for grid-indexed copy show non-empty `index_exprs`,
  participating loop/launch variables, `slice` coverage for per-work global
  tiles, and `guarded` predicate kind when a TIR predicate protects access.
- validator negative tests reject indexed regions whose `index_exprs` do not
  match rank or whose coverage/predicate fields are inconsistent.

Source/spec:

- per-work descriptors in TTProgram / ExecutableSpec point back to the
  `AccessRegion` used to derive them.
- removing that evidence fails validation or source build.

Runtime:

- existing grid-indexed copy direct runtime remains green and proves that
  projected per-work descriptors drive source/runtime addressing.
- segmented/ragged/table-indexed direct runtime cases are required before T8
  can be marked complete.

Unsupported diagnostics:

- missing TIR index/predicate evidence: `lowering_missing`;
- indexed evidence exists but no admitted TT descriptor exists:
  `backend_op_missing`;
- descriptor exists but runtime cannot execute the shape:
  `admission_blocked` or typed simulator boundary.

## 2026-05-05 First Slice Status

Implemented:

- `AccessRegion` records concrete `BufferLoad` / `BufferStore`
  `index_exprs`, participating loop/launch variables, and guarded vs
  unconditional predicate kind for the covered grid-indexed slice.
- `ValidateSpatialPlan` rejects slice / row-slice / grouped-slice regions
  whose `index_exprs` do not match logical rank.
- `TTPerWorkArgSpec`, executable projection, serialization, and direct runtime
  metadata carry `access_region` and `access_region_index` for per-work tile
  descriptors derived from indexed access.
- The grid-indexed direct-runtime copy path remains green through
  `BlackholeModule`.
- A larger flash-attn regression tightened the first-slice evidence boundary:
  rank-aligned constant accesses such as `[0, 0]` are not indexed evidence.
  `AccessRegion.index_exprs` is projected for the admitted indexed slice only
  when the TIR access expression contains an actual participating index
  variable.  Constant full-tile reads remain `full` coverage and must not force
  downstream loop-var evidence.

2026-05-05 table-indexed slice status:

- A minimal `BlockIndices[bx]` staged copy is admitted as the first
  table-backed per-work tile descriptor.
- `BuildSpatialPlan` substitutes active `LetStmt` bindings when recording
  `AccessRegion.index_exprs`, so the A read evidence contains the actual
  table-derived index expression rather than an unbound temporary.
- `TTPerWorkArgSpec` now admits `value_source=index_table` with
  `index_buffer` and `index_value_scale` fields.  Projection, executable
  serialization, segment metadata, Python helpers, and direct runtime consume
  those typed fields.
- Guarded `tir.if_then_else(load, zero)` copies are recognized as predicated
  copies for the admitted source rewrite.  Source consumes
  `runtime_arg_u32("a_tile_start_id")`; it must not emit a raw
  `BufferLoad(BlockIndices[...])`.
- The index table is materialized as a page-indexed DRAM input buffer with
  4-byte pages so direct runtime can evaluate the per-work arg from host-side
  table data without positional argument recovery.
- Direct runtime validates the computed tile start against the target buffer's
  typed materialization page count.  Out-of-range table entries fail closed
  instead of relying on the original TIR guard after source lowering.

2026-05-05 ragged row-bound slice status:

- A minimal `RowCounts[bx]` staged copy is admitted as the first
  predicate-derived ragged bound descriptor.
- The TIR predicate `i < valid_rows`, where `valid_rows` is bound by a real
  `BufferLoad` from the row-count table, is lowered to an `a_valid_rows`
  runtime arg with `TTPerWorkArgSpec` descriptor kind `valid_rows`,
  `value_source=index_table`, and `index_buffer=RowCounts`.
- The reader source consumes the projected per-work arg, reads only valid
  row pages from the input, and writes zero pages for invalid rows before
  publishing the CB.  The writer consumes the same 32 published row pages and
  writes them back, preserving the original TIR `if_then_else(load, 0)`
  semantics without relying on output-buffer initialization.
- Row-count tables are materialized as 4-byte page-indexed DRAM inputs.
  64-byte bf16 row/stick pages remain row-major host pages; only complete
  32x32 tile pages use nfaces host tilization in direct runtime transfer.
- The admitted direct-runtime gate proves `RowCounts=[32,17,0]` copies only
  valid rows and writes zeros for invalid rows through `BlackholeModule`.

2026-05-05 segmented row-segment slice status:

- A minimal non-uniform row segment copy is admitted as the first
  segmented/grouped dispatch surface.
- The row start table is used in the TIR source address expression and lowers
  to an `a_segment_row_start` runtime arg with descriptor kind
  `segment_row_start`, `value_source=index_table`, and
  `index_buffer=SegmentOffsets`.
- The row count table is used in the TIR guarded-copy predicate and lowers to
  an `a_segment_row_count` runtime arg with descriptor kind
  `segment_row_count`, `value_source=index_table`, and
  `index_buffer=SegmentCounts`.
- Source consumes those two projected per-work descriptors and must not emit
  raw source reads from `SegmentOffsets` / `SegmentCounts`.
- The segmented reader uses page-indexed row pages
  `segment_row_start + page_row`; the writer emits a compact per-work output
  block and writes explicit zero pages for invalid rows.  The old input
  `a_tile_start_id` / tile-count / tile-stride descriptors are not synthesized
  for this segmented input path.
- The admitted direct-runtime gate proves non-32-aligned starts and
  non-uniform counts, including a zero-count segment, through `BlackholeModule`.

Still open for T8:

- broader indexed block traversal beyond the first one-dimensional
  per-work tile-start table descriptor.
- broader ragged token/page forms beyond the first row-count predicate slice.

## Completion Criteria

T8 is implemented only when:

- segmented/grouped, ragged, and indexed-block evidence are derived from TIR
  structure, not workload names;
- every projected TT/indexed descriptor references explicit SpatialPlan
  evidence;
- validators reject missing or inconsistent indexed evidence;
- source/runtime addressing consumes the projected descriptors;
- at least one admitted positive path in each T8 family has direct-runtime
  correctness or a typed simulator capability boundary after source/spec
  admission;
- docs, progress, and memory reflect the boundary.
