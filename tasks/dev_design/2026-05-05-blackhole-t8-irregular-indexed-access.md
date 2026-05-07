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

T1-T7.5 admitted regular tensor, sharded, page-addressed, topk, and exact-CB
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

T8 extensions must continue to lower through generic indexed/guarded evidence:
`AccessRegion.index_exprs`, `AccessRegion.predicate_exprs`, loop/launch vars,
and generic per-work `value_expr` bindings.  A new public object is valid only
if it is generic across IR structures and changes legality, typed planning, or
runtime/source addressing.  Workload-shaped schema for segmented ranges,
ragged bounds, or block-index traversal is not allowed.

### ExecutableSpec

ExecutableSpec projects the selected per-work and indexed bindings.

Runtime/source may consume projected bindings, but must not recompute:

- logical block axes from source text;
- tile starts from raw `work_linear_id` when a stronger binding exists;
- ragged bounds or index-table traversal from argument names.

Fused-dataflow ABI completion may synthesize the regular default tile-origin
runtime args only when the projected segment lacks stronger value evidence.
The suppression test is the typed projected binding itself: existing
non-synthesized runtime args and `TTPerWorkArgSpec` records whose
`value_source=value_expr` is not a `buffer_tile_origin`.  It must not classify
runtime args by identities or names such as `per_work_value*`.

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

## Current Surface And Remaining Extensions

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
  page-addressed bf16 row-copy surface.  Wider grouped GEMM dispatch is a later
  workload path, but must reuse the same binding/evidence contract.

The admitted multi-segment extension allows multiple independent row segments
in one logical work item.  Each `SegmentOffsets[...]` table load that drives a
source row expression gets its own generic value runtime arg identity, and each
matching guarded `SegmentCounts[...]` table load gets its own
generic value identity.  Row-page source rendering must use the runtime
arg present in the current TIR access/predicate, not a hardcoded singleton
`per_work_value` / `per_work_value_1` pair.  Because this path uses
64-byte row pages, the source row-page address is the TIR-derived row start
plus the local row; it must not reuse a full-tile `base_tile_index` or divide
the row start by the tile height.

Ragged bounds:

- derive valid row/token bounds from TIR predicates;
- operands such as `cache_seqlens` are evidence only if used in the predicate
  or index expression;
- invalid rows/tokens must be skipped in source/runtime.

The admitted ragged token/page slice covers a copy-shaped paged decode
primitive: `PageTable[bx, by]` selects the source page and
`CacheSeqLens[bx]` is consumed by a predicate of the form
`by * page_rows + row < cache_len`.  This is still ordinary TIR.  TTProgram
must project both the page-table tile-start binding and generic value bindings
for the predicate bound / launch-axis value, so source can evaluate row
validity from typed per-work args instead of leaving block-axis variables or
cache-length table loads in device source.

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

The admitted indexed-block extension broadens that same binding to table
address expressions that are not equivalent to `work_linear_id`.  When the TIR
address uses a table load such as `BlockIndices[bx, by]`, `TTPerWorkArgSpec`
must carry the generic TIR `value_expr` for the runtime arg value.  Direct
runtime evaluates that expression under the current work context; it must not
keep a `work_linear_id` compatibility fallback or use any parallel
`index_table_*` projection field as a second evaluator.

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

The admitted sparse slice broadens this from exactly two entries to more than
two entries in the same logical work item.  The contract is not a special
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

## Current Status

T8 has completed the generic evidence chain for the current backend boundary.
The current owner truth is:

- `AccessRegion` records concrete access `index_exprs`, participating
  loop/launch variables, guarded predicate kind, and concrete
  `predicate_exprs` for guarded regions;
- `ValidateSpatialPlan` rejects malformed indexed/guarded regions before TT
  planning;
- `TTPerWorkArgSpec`, executable projection, serialization, and direct
  runtime metadata carry the source `AccessRegion` identity for per-work
  bindings derived from indexed access;
- `ValidateTTProgram` rejects buffer-bound per-work specs that lack
  `access_region`, have an invalid `access_region_index`, or point at a region
  whose subject does not match the bound buffer;
- `FindSpatialAccessRegionRef(subject, access_kind, index_exprs)` fails closed
  when indexed structural indices are present and no exact `AccessRegion`
  matches; it must not fall back to the first same-buffer/same-kind region;
- dynamic table and work-context values lower as generic
  `value_source=value_expr` records, with optional
  `value_usage=buffer_tile_origin` only when the value is consumed as a buffer
  tile origin;
- launch-axis dependencies are normalized to explicit runtime-arg calls before
  projection, so direct runtime evaluates serialized expressions instead of
  interpreting naked variable names;
- value-expression matching includes the referenced table-buffer identity and
  `value_usage`, so structurally similar loads such as
  `SegmentOffsets[bx, k]` and `SegmentCounts[bx, k]` do not collapse into one
  runtime arg.

Admitted positive surfaces include:

- grid-indexed direct-runtime copy;
- one-dimensional and two-dimensional table-indexed staged copy;
- contiguous scaled-block copy;
- sparse two-entry and three-entry indexed traversal;
- independent sparse row bounds through `ValidRows[bx, k]`;
- ragged row-bound copy through `RowCounts[bx]`;
- paged ragged copy through `PageTable[bx, by]` plus `CacheSeqLens[bx]`;
- page-local row-bound copy through `PageValidRows[bx, by]`;
- non-uniform segmented row copy through one, two, and three independent
  `SegmentOffsets` / `SegmentCounts` range pairs;
- the T9.1 grouped-GEMM segmented-A feed path.

Deleted or forbidden protocol surfaces remain deleted:

- `value_source=index_table`, `index_buffer`, `index_value_scale`,
  `index_table_shape`, and `index_table_index_sources`;
- buffer-wide table-addressing side caches and pass-local
  `IndexTableAddressing` helpers;
- `work_linear_id` compatibility fallback for table-backed bindings with
  missing value evidence;
- row-start / row-count binding synthesis from only an index-buffer name;
- runtime-arg-name classification such as `per_work_value*` prefixes,
  including fused-dataflow default tile-arg admission;
- ABI consumer-side access-region recovery helpers such as
  `inferred_access_kind_for_spec` and `attach_access_region_evidence`;
- indexed `AccessRegion` first-match fallbacks that ignore structural
  `index_exprs`.

The latest cleanup deleted the remaining fused-dataflow ABI consumer-side
`per_work_value*` prefix classifier.  Explicit generic value-expression
bindings now suppress fallback input tile defaults even when their runtime arg
kind is not a legacy prefix.  The same validation slice also caught a CB
lifecycle regression: writer-visible output CBs must not receive generic
retained-front pre-drain rewrites, because the capacity-aware reserve rewrite
already owns real pressure and premature output pops destroy FIFO order.

The final T8 cleanup also removed the remaining consumer-side access-region
reconstruction path.  ABI lowering now creates each buffer-bound per-work spec
with explicit read/write access context, `ValidateTTProgram` checks that
evidence, and indexed lookup fails closed when the current TIR-derived
structural indices do not match a SpatialPlan region.  The broadened runtime
gate includes three independent segmented ranges in one work item, plus the
existing sparse, ragged, paged, and indexed copy cases.

Known residual outside T8: the full copy-pipeline suite still exposes a flash
bridge granularity guard at `merge_fragment_tiles` destination `acc_o`.  That is
a T9 exact-CB / materialization follow-up and must not be used to reopen T8
copy-side fallback schema.

## Completion Criteria

T8 is implemented when:

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
