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
  -> TTProgram per-work / indexed-access bindings
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
- `predicate_exprs`: the concrete boolean TIR predicate expressions guarding
  the region when `predicate_kind=guarded`.

`AccessRegion` must not be just a dump.  It must be validated and consumed by
TT planning for the admitted subset.

### TTProgram

TTProgram owns target realization of indexed work/addressing.

The first implementation reuses and tightens `TTPerWorkArgSpec` for simple
per-work tile bindings, while adding an explicit reference back to the
SpatialPlan `AccessRegion` evidence when the binding is derived from an
indexed region.

Later T8 slices may add focused objects for:

- segmented ranges;
- ragged bounds;
- block-index table traversal.

Those objects must still point back to `AccessRegion` / predicate evidence
rather than to workload names.

### ExecutableSpec

ExecutableSpec projects the selected per-work and indexed bindings.

Runtime/source may consume projected bindings, but must not recompute:

- logical block axes from source text;
- tile starts from raw `work_linear_id` when a stronger binding exists;
- ragged bounds or index-table traversal from argument names.

## First Implementation Slice

The first T8 slice is deliberately narrow:

1. `AccessRegion` records concrete `BufferLoad` / `BufferStore` index
   expressions and whether the access is guarded.
2. `ValidateSpatialPlan` rejects malformed indexed regions.
3. `TTPerWorkArgSpec` / executable projection preserve a link to the source
   `AccessRegion` for per-work tile bindings derived from indexed access.
4. Codegen/runtime continue to consume the projected per-work bindings; the
   binding is no longer an untraceable source-emitter default.
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
- source/runtime bindings must carry the selected segment start/count.

The admitted first segmented/grouped slice is a row-segment staged copy:

- a one-dimensional `int32` table load used in the source address row
  expression becomes a generic per-work value binding whose `value_expr`
  carries that TIR load;
- a one-dimensional `int32` table load used in the guarded copy predicate
  becomes another generic per-work value binding whose `predicate_exprs`
  connect it to the guarded `AccessRegion`;
- the device source consumes runtime args derived from those bindings and
  must not emit raw source reads from the tables;
- the copied output is a compact per-work tile-sized block, so invalid rows
  are zero-filled inside that block and cannot clobber a neighboring segment;
- this slice admits non-uniform segment starts/counts only for a single
  page-indexed bf16 row-copy surface.  Wider grouped GEMM dispatch is a later
  workload path, but must reuse the same binding/evidence contract.

The next segmented slice allows multiple independent row segments in one
logical work item.  Each `SegmentOffsets[...]` table load that drives a source
row expression gets its own generic value runtime arg identity, and each
matching guarded `SegmentCounts[...]` table load gets its own
generic value identity.  Row-page source rendering must use the runtime
arg present in the current TIR access/predicate, not a hardcoded singleton
`per_work_value` / `per_work_value_1` pair.  Because this path uses
64-byte row pages, the source page id is the TIR-derived row start plus the
local row; it must not reuse a full-tile `base_tile_index` or divide the row
start by the tile height.

Ragged bounds:

- derive valid row/token bounds from TIR predicates;
- operands such as `cache_seqlens` are evidence only if used in the predicate
  or index expression;
- invalid rows/tokens must be skipped in source/runtime.

The next ragged token/page slice admits a copy-shaped paged decode primitive:
`PageTable[bx, by]` selects the source page and `CacheSeqLens[bx]` is consumed
by a predicate of the form `by * page_rows + row < cache_len`.  This is still
ordinary TIR.  TTProgram must project both the page-table tile-start binding
and generic value bindings for the predicate bound / launch-axis value, so
source can evaluate row validity from typed per-work args instead of leaving
block-axis variables or cache-length table loads in device source.

A broader ragged page slice uses a per-page row-bound table rather than a
prefix sequence length: `PageTable[bx, by]` selects the source page and
`PageValidRows[bx, by]` guards rows with `row < valid_rows`.  Both table loads
must carry `[logical_block_x, logical_block_y]` addressing in their
bindings.  This remains a predicate-derived row-bound surface; it must not
be recovered from table names or argument positions.

Indexed block traversal:

- derive table-driven block traversal from `BufferLoad` / `BufferStore` index
  expressions;
- operands such as `block_indices` are evidence only if they are read and used
  to form a memory address;
- source/runtime must consume the projected table binding.

The admitted first table-indexed form is a per-work tile binding derived
from ordinary TIR indexed access.  The owner truth is the `AccessRegion`
`index_exprs` plus the typed per-work binding back to that region; source code
consumes the normal tile-start runtime arg and must not emit a raw `BufferLoad`
from the index table to recover the tile id.

The next indexed-block slice broadens that same binding to table address
expressions that are not equivalent to `work_linear_id`.  When the TIR address
uses a table load such as `BlockIndices[bx, by]`, `TTPerWorkArgSpec` must carry
the generic TIR `value_expr` for the runtime arg value.  Direct runtime
evaluates that expression under the current work context; it must not keep a
`work_linear_id` compatibility fallback or use any parallel `index_table_*`
projection field as a second evaluator.

Sparse traversal can require more than one table-derived tile start inside the
same logical work item.  `SpatialPlan` must preserve each structurally
distinct same-subject access as its own `AccessRegion`; `TTPerWorkArgSpec`
then references the region selected by structural `index_exprs` matching.
Current direct runtime evaluates the serialized `value_expr`; legacy table
projection fields are not durable semantic owners and must not be broadened
into per-case schema.

The schema boundary is intentionally narrow: do not add
`index_table_*` variants, `topk` fields, selection plans, or workload-shaped
records to patch a single admitted example.  If downstream execution needs a
fact that survives across stages, the fact must be represented either as
generic `AccessRegion` evidence (`index_exprs`, `predicate_exprs`, loop vars)
or as a generic lowered `ExecutableSpec` evaluator/input record derived from
that evidence.  Public per-work schema uses `value_source=value_expr` plus the
serialized TIR expression; `index_buffer`, `index_value_scale`,
`index_table_shape`, `index_table_index_sources`, and
`value_source=index_table` are deleted protocol surfaces.

`value_expr` must also stay IR-first when it depends on launch work state.
The projected expression must not contain naked work-variable names for
runtime to reinterpret.  ABI lowering normalizes block-axis variables to
explicit `tl.blackhole.runtime_arg_u32(...)` calls derived from the current
IR `thread_extent` binding, and direct runtime evaluates only that explicit
call form.  Compute-derived quantities such as GEMM K-tile counts or N-tile
strides are folded from typed GEMM/core-grid records into ordinary constant
`value_expr`s before projection; do not reintroduce `num_k_tiles`,
`logical_n_tiles`, or `bx/by/bz` name recovery in leaf readers.

The same rule applies to table-derived ragged bounds.  If one work item has
multiple independent guarded sparse reads, each bound table load gets its own
generic runtime arg identity, e.g. `per_work_value` /
`per_work_value_1`, with a distinct `value_expr` on the corresponding
`TTPerWorkArgSpec`.  A later row-page reader may use those args to decide
per-row zero-fill, but it must not collapse distinct guarded reads back into
one shared generic value.

The next sparse slice broadens this from exactly two entries to more than two
entries in the same logical work item.  The contract is not a special
three-entry object: every independent `BlockIndices[bx, k]` and
`ValidRows[bx, k]` TIR load must allocate or reuse the matching per-work arg
identity from structural `value_expr` equality plus the matched
`AccessRegion.index_exprs` / `predicate_exprs`, with literal `k` represented
as `constant:k`.

## Validation Plan

Structure:

- `AccessRegion` tests for grid-indexed copy show non-empty `index_exprs`,
  participating loop/launch variables, `slice` coverage for per-work global
  tiles, and `guarded` predicate kind when a TIR predicate protects access.
- validator negative tests reject indexed regions whose `index_exprs` do not
  match rank or whose coverage/predicate fields are inconsistent.
- validator negative tests reject `predicate_kind=guarded` regions that lack
  `predicate_exprs`, and reject predicate expressions outside guarded regions.

Source/spec:

- per-work bindings in TTProgram / ExecutableSpec point back to the
  `AccessRegion` used to derive them.
- removing that evidence fails validation or source build.

Runtime:

- existing grid-indexed copy direct runtime remains green and proves that
  projected per-work bindings drive source/runtime addressing.
- segmented/ragged/table-indexed direct runtime cases are required before T8
  can be marked complete.

Unsupported diagnostics:

- missing TIR index/predicate evidence: `lowering_missing`;
- indexed evidence exists but no admitted TT binding exists:
  `backend_op_missing`;
- binding exists but runtime cannot execute the shape:
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
  bindings derived from indexed access.
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
  table-backed per-work tile binding.
- `BuildSpatialPlan` substitutes active `LetStmt` bindings when recording
  `AccessRegion.index_exprs`, so the A read evidence contains the actual
  table-derived index expression rather than an unbound temporary.
- `TTPerWorkArgSpec` carries a generic TIR `value_expr` for the runtime value
  that a per-work binding must pass to source.  This is not an
  index-table-specific schema: the expression can contain the original
  `BufferLoad`, arithmetic, and launch-axis variables.  The old
  `value_source=index_table`, `index_buffer`, `index_value_scale`,
  `index_table_shape`, and `index_table_index_sources` projection surfaces
  have been deleted from TTProgram projection, ExecutableSpec metadata, and
  direct runtime metadata.
- A per-work value that is consumed as the associated buffer's tile origin
  carries the generic `value_usage=buffer_tile_origin` marker.  This marker is
  not an index-table, row-bound, or page-workload schema; it tells runtime
  which already-projected generic value needs validation against the target
  buffer materialization page count.  Predicate / row-bound values such as
  `per_work_value` do not carry this usage.
- Guarded `tir.if_then_else(load, zero)` copies are recognized as predicated
  copies for the admitted source rewrite.  Source consumes
  `runtime_arg_u32("a_tile_start_id")`; it must not emit a raw
  `BufferLoad(BlockIndices[...])`.
- The index table is materialized as a page-indexed DRAM input buffer with
  4-byte pages so direct runtime can evaluate the per-work arg from host-side
  table data without positional argument recovery.
- Direct runtime validates only `value_usage=buffer_tile_origin` values against
  the target buffer's typed materialization page count.  Out-of-range table
  entries fail closed instead of relying on the original TIR guard after
  source lowering, while row-bound `value_expr` bindings remain ordinary
  guarded-copy values.
- A two-dimensional `BlockIndices[bx, by]` staged copy is now admitted for the
  indexed-block traversal slice.  The A tile-start binding carries
  `value_source=value_expr`; the serialized TIR expression contains the
  `BlockIndices` `BufferLoad` and launch-axis variables.
- Direct runtime evaluates the binding's generic `value_expr` under the
  current logical work context, including integer `BufferLoad` reads from the
  materialized host-side table data.  It does not compute the runtime value
  from side metadata, and there is no `work_linear_id` compatibility fallback
  for missing value evidence.
- The two-dimensional case is covered by direct-runtime correctness and by a
  serialized-module round trip, so `BlackholeModule` save/load preserves the
  table addressing contract.
- A minimal multi-tile indexed block copy is admitted for contiguous block
  traversal.  When the TIR source row expression uses
  `BlockIndices[bx] * block_rows + row`, the A tile-start binding's
  `value_expr` includes the scale arithmetic, direct runtime passes the scaled
  tile start, and source lowering consumes that runtime arg as the base tile
  id for each subtile instead of multiplying it by the block scale again.
- A minimal sparse two-entry indexed traversal is admitted for one work item
  reading two independently indexed source tiles.  `BlockIndices[bx, 0]` and
  `BlockIndices[bx, 1]` lower to separate A tile-start runtime args
  `a_tile_start_id` and `a_tile_start_id_1`.  `SpatialPlan` now preserves the
  two A read regions separately and binding binding selects the matching
  region by structural `index_exprs`, so the second entry cannot silently bind
  to the first A read region.  Source consumes the two projected runtime args
  and emits no raw index-table read.
- The sparse two-entry surface also admits independent per-entry row bounds.
  `ValidRows[bx, 0]` and `ValidRows[bx, 1]` lower to generic A value
  bindings with identities `per_work_value` and `per_work_value_1`, carrying
  distinct `value_expr` evidence.  The row-page reader uses the matching
  runtime arg for each sparse tile and zero-fills invalid rows
  independently.
- The sparse ragged surface is no longer proven only by exactly two entries:
  a three-entry direct-runtime gate covers `BlockIndices[bx, 0/1/2]` and
  `ValidRows[bx, 0/1/2]`, with literal columns preserved inside each
  `value_expr` and
  per-entry runtime arg identities.  This remains the same
  `TTPerWorkArgSpec` contract, not a sparse-specific operator.
- Guarded `AccessRegion` now carries the actual TIR predicate expressions.
  For the admitted ragged row-bound shape, the A read region records
  `T.shift_right(tx, 2) < RowCounts[bx]`.  `ValidateSpatialPlan` rejects a
  guarded region without predicate expressions, so later TT planning cannot
  recover the row-bound predicate from runtime arg names or binding kinds.
- The old buffer-wide `index_buffer -> table addressing` side cache in
  TT lowering has been deleted.  Table addressing must be present on the
  concrete per-work binding produced from the TIR load / matching
  `AccessRegion`; a later ABI step can no longer fill missing table shape or
  index-source fields by looking up only the index-buffer name.
- The later pass-local `IndexTableAddressing` helper has also been deleted.
  Per-work runtime-arg identity and aliasing are deduplicated from structural
  `value_expr` equality plus the matched `AccessRegion.index_exprs`, not from
  a second table-shape/index-source object.
- Compute-segment admission for row-bound runtime args is carried by a
  pass-local control flag.  It must not be inferred from a `per_work_value`
  prefix or any other runtime-arg naming convention.
- Direct runtime no longer falls back to `work_linear_id` for table-backed
  bindings with missing value evidence.  `value_source=value_expr` requires
  a generic `value_expr` owner.  The old ABI synthesis branches that rebuilt
  row-count / row-start bindings from only an index-buffer name were
  removed; missing binding evidence must fail closed instead.

2026-05-05 ragged row-bound slice status:

- A minimal `RowCounts[bx]` staged copy is admitted as the first
  predicate-derived ragged bound binding.
- The TIR predicate `i < valid_rows`, where `valid_rows` is bound by a real
  `BufferLoad` from the bound table, is lowered to a generic `per_work_value`
  runtime arg with `value_source=value_expr`.
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

2026-05-05 paged ragged cache-length slice status:

- A copy-shaped paged decode primitive is admitted for the first
  `PageTable[bx, by]` plus `CacheSeqLens[bx]` shape.
- The page table participates in the TIR source address expression and lowers
  to the existing A `tile_start` binding with
  `value_source=value_expr`.
- The cache-length table participates in the TIR guarded-copy predicate and
  lowers to a generic A value binding with `value_source=value_expr`.
- The launch page axis is represented by a generic A value binding with
  `value_source=logical_block_y`.  Source evaluates row
  validity as `logical_block_y * page_rows + page_row < cache_len` from
  projected runtime args; it does not leave raw `PageTable` /
  `CacheSeqLens` loads or block-axis variables in the device source.
- Row-bound staged copy lowering treats this surface as separate row-page
  reader/writer transport rather than a fused full-tile shortcut, preserving
  invalid-row zero-fill semantics.
- The admitted direct-runtime gate proves nontrivial page order and
  non-page-aligned cache lengths through `BlackholeModule`.
- A broader page-local row-bound gate admits `PageValidRows[bx, by]` as the
  row predicate source.  Its generic A value binding carries
  `value_source=value_expr`.  Unlike the prefix cache-length form, this case
  does not need a separate launch-axis value binding because the TIR predicate
  is already local to the selected page.

2026-05-05 segmented row-segment slice status:

- A minimal non-uniform row segment copy is admitted as the first
  segmented/grouped dispatch surface.
- The row start table is used in the TIR source address expression and lowers
  to a generic `per_work_value` runtime arg with `value_source=value_expr`.
- The row count table is used in the TIR guarded-copy predicate and lowers to
  another generic `per_work_value*` runtime arg with
  `value_source=value_expr`.
- Source consumes those two projected per-work bindings and must not emit
  raw source reads from `SegmentOffsets` / `SegmentCounts`.
- The segmented reader uses page-indexed row pages
  `base_value + page_row`; the writer emits a compact per-work output
  block and writes explicit zero pages for invalid rows.  The old input
  `a_tile_start_id` / tile-count / tile-stride bindings are not synthesized
  for this segmented input path.
- The admitted direct-runtime gate proves non-32-aligned starts and
  non-uniform counts, including a zero-count segment, through `BlackholeModule`.

Still open for T8:

- broader indexed block/page traversal beyond the admitted launch-axis table
  addressing, contiguous scaled-block copy, two-entry sparse copy, and
  sparse+ragged two-entry slices.
- broader ragged token/page forms beyond the admitted row-count, copy-shaped
  paged cache-length predicate, and per-entry sparse row-bound slices.

## Completion Criteria

T8 is implemented only when:

- segmented/grouped, ragged, and indexed-block evidence are derived from TIR
  structure, not workload names;
- every projected TT/indexed binding references explicit SpatialPlan
  evidence;
- validators reject missing or inconsistent indexed evidence;
- source/runtime addressing consumes the projected bindings;
- at least one admitted positive path in each T8 family has direct-runtime
  correctness or a typed simulator capability boundary after source/spec
  admission;
- docs, progress, and memory reflect the boundary.
