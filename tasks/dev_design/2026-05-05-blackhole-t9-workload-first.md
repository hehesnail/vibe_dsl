# Blackhole T9 Workload First Paths

## Role

This document defines the task-level design for T9 workload-first paths in the
Blackhole backend.

It is not a new overall design document.
The durable chain remains:

```text
Normalized Tile TIR
  -> SpatialPlan
  -> TTProgram
  -> ExecutableSpec
```

Current execution status lives in `tasks/progress.md`.

## Problem

T8 admitted irregular/indexed/ragged primitives as direct-runtime copy-shaped
surfaces.  T9 must prove that real workload shapes can be decomposed into
those primitive surfaces plus admitted compute/materialization paths.

The workload name is not the protocol owner.  Names such as grouped GEMM,
paged decode, sparse attention, recurrence, or multi-block flash are task
labels only.  The backend must still derive evidence from TIR access
expressions, predicates, loop domains, explicit tile compute, and typed
placement/lifecycle records.

## Goal

Each T9 checkpoint must run through `BlackholeModule` direct runtime and
compare against a host reference.  A checkpoint is admitted only when:

- workload evidence is present in ordinary TIR;
- the evidence lowers through `SpatialPlan`, `TTProgram`, and
  `ExecutableSpec` typed records;
- source/runtime consume projected records instead of names, argument
  positions, or generated source text;
- unsupported workload variants fail closed before source/runtime guessing.

## Current Status

- T9.1 pre-grouped MoE/routed grouped GEMM has bf16 direct-runtime correctness
  through ordinary TIR-derived segmented A bindings and typed
  materialization/lifecycle records.
- T9.3 dual-score MLA GEMM has bf16 direct-runtime correctness through
  explicit `Q_nope @ KV_latent^T` plus `Q_pe @ K_pe^T` score accumulation,
  with the accumulator kept on a compute-local live-form CB until the final
  output publish.
- T9.2 paged GQA has source/spec admission, page-addressed QK/AV micro
  runtime coverage, and full online-softmax bf16 direct-runtime correctness.
- T9.3 full paged MLA decode keeps source/spec admission through generic
  page-table/ragged `value_expr` bindings, explicit score GEMMs, retained
  latent-KV lifetime, and the existing flash partial-combine path, and now
  has full online-softmax bf16 direct-runtime correctness.
- T9.4 sparse/ragged attention, T9.5 recurrence/scan, and T9.6 multi-block
  flash decode are queued.

## Non-Goals

- No frontend workload op such as `T.grouped_gemm`, `T.paged_decode`, or
  `T.sparse_attention`.
- No workload metadata registry.
- No source-name, buffer-name, or argument-position recovery.
- No distributed or production collective claim; T10 owns those variants.
- No new long-lived representation layer outside the established chain.

## T9.1 Pre-Grouped GEMM

The first T9.1 slice admits a pre-grouped MoE/routed grouped GEMM surface:

- `GroupOffsets[g]` is used in the A source row expression;
- `GroupSizes[g]` is used in the guarded A copy predicate;
- each logical group is one work item that computes a compact
  `tile_m x tile_n` output block for that group;
- invalid A rows inside the group tile are zero-filled before GEMM so the
  output rows beyond `GroupSizes[g]` are deterministic zeros;
- B/W expert weights are selected by the same logical group axis through
  ordinary TIR indexing.

The first admitted shape is intentionally narrow:

- bf16 A and B/W inputs;
- fp32 output;
- static `tile_m=32`, `tile_n=32`, and `K=128`;
- one output N tile per group;
- no dynamic expert-id remapping beyond the current logical group index;
- no fused activation, topk routing, or all-to-all.

The required evidence chain is:

```text
TIR GroupOffsets/GroupSizes loads
  -> generic per-work value bindings for A base and bound values
  -> row-page A materialization with zero-fill
  -> compute-compatible A live form for matmul_tiles
  -> grouped GEMM direct runtime correctness
```

This checkpoint is not satisfied by a grouped row-copy alone.  The row-bound
materialization must feed the admitted GEMM path through typed live-form /
materialization records or fail closed with a typed reason.

The admitted implementation materializes A from row-major DRAM pages into a
compute-compatible tiled CB:

- the reader consumes generic A per-work value runtime args derived from
  `GroupOffsets` / `GroupSizes`; raw table loads are not present in source;
- the A accessor is page-addressed over 64-byte bf16 row tiles, so non-32-row
  group offsets are row-addressed instead of tile-addressed;
- each valid row tile is read into a scratch CB and copied into the nfaces
  tile layout expected by `matmul_tiles`;
- invalid rows zero the destination tile slices before publish;
- direct runtime treats segmented A as a raw row-major backing tensor and
  decodes grouped output as compact `groups * tile_m` rows by `tile_n`
  columns.

The scratch CB / `copy_cb_page` step is a data-movement implementation detail
needed to satisfy Blackhole NOC alignment while preserving the typed
binding contract.  It is not a workload-level side channel.

## T9.2 Paged GQA Decode

The first T9.2 slice admits a paged GQA decode tile through ordinary TIR:

- `PageTable[sequence, page]` selects the KV cache page for a statically
  known page step;
- `CacheSeqLens[sequence]` guards rows inside each page;
- multiple query heads share one KV head, so the kernel is GQA even though
  the admitted first shape uses one KV head;
- the attention update is the existing flash-attention tile sequence with
  `scores_max`, `scores_scale`, `logsum`, `acc_s_cast`, and `acc_o`
  partial combine.

The first admitted shape is intentionally narrow:

- bf16 Q, paged K cache, paged V cache, and output;
- fp32 accumulators;
- `batch=2`, `heads=4`, `groups=4`, one KV head;
- `block_M=32`, `block_N=32`, `dim=32`;
- exactly two KV pages per sequence, with ragged lengths such as 45 and 64
  tokens;
- page ids are non-contiguous and table-driven;
- the two page steps are static TIR statements, not a frontend decode op.

The required evidence chain is:

```text
TIR PageTable / CacheSeqLens loads
  -> per-page tile-start ABI bindings for K and V cache reads
  -> generic bound-value bindings for the same guarded row copies
  -> compute-compatible K/V live forms
  -> existing flash partial-combine compute/materialization path
  -> paged GQA direct runtime correctness
```

The source may consume runtime args projected from these bindings, but it
must not emit raw `PageTable` or `CacheSeqLens` reads to recover the page or
ragged-bound semantics.  If the page-table/ragged evidence cannot feed the
existing flash compute path, the backend must reject with a typed reason
before source/runtime guessing.

The admitted implementation keeps the workload surface in that chain:

- K cache and V cache page selection are ordinary TIR-derived per-work
  bindings with `value_source=value_expr`.  Page 0 and page 1 are separate
  static TIR statements that differ by their serialized value expression, not
  by a frontend decode op.
- `CacheSeqLens[sequence]` lowers to generic per-work value bindings paired
  with the page-local row predicate.  Source
  consumes the projected runtime args and does not reload the page table or
  cache-length buffers.
- Direct runtime materializes page-addressed K/V inputs from the executable
  materialization records.  Complete tile pages use tiled host transfer;
  row/stick-style page-addressed inputs remain raw row-major.  Indexed GEMM B
  buffers that use explicit tile-start bindings stay raw so the runtime
  binding remains the owner of page selection.
- The flash compute path reuses exact-CB partial combine.  Local intermediate
  stream CBs pop a stale front page before a later producer republishes to
  the same physical CB, and generated static pops are clamped to pages that
  were actually visible on that local intermediate front.
- Source codegen loads CB config metadata independently from runtime-arg
  binding, so compute segments with no runtime args can still render typed CB
  operations such as `untilize_cb_front_tile_fragment`.

## T9.3 Paged MLA Decode

The first T9.3 slice admits a paged MLA decode tile through ordinary TIR:

- `PageTable[sequence, page]` selects both latent-KV and K-PE cache pages for
  a statically known page step;
- `CacheSeqLens[sequence]` guards rows inside each page;
- score computation is the explicit sum of two leaf GEMM contributions:
  `Q_nope @ KV_latent^T` and `Q_pe @ K_pe^T`;
- value computation reuses the same latent KV page as V:
  `softmax(scores) @ KV_latent`;
- the online softmax and output update reuse the existing flash partial
  combine sequence.

The first admitted shape is intentionally narrow:

- bf16 Q-nope, Q-PE, paged latent KV cache, paged K-PE cache, and output;
- fp32 accumulators;
- `batch=2`, `heads=4`, one KV head / shared latent cache;
- `block_M=32`, `block_N=32`, `dv=32`, and `dpe=32`;
- exactly two cache pages per sequence, with ragged lengths such as 45 and 64
  tokens;
- page ids are non-contiguous and table-driven;
- both page steps are static TIR statements, not a frontend MLA or paged
  decode op.

The required evidence chain is:

```text
TIR PageTable / CacheSeqLens loads
  -> per-page tile-start ABI bindings for latent-KV and K-PE reads
  -> generic bound-value bindings for the guarded page copies
  -> compute-compatible latent-KV and K-PE live forms
  -> two explicit score GEMMs into acc_s
  -> latent-KV retained until the value GEMM in the same page step
  -> existing flash partial-combine compute/materialization path
  -> paged MLA direct runtime correctness
```

The source may consume runtime args projected from these bindings, but it
must not emit raw `PageTable` or `CacheSeqLens` reads to recover page or
ragged-bound semantics.  The retained latent-KV input lifetime is part of the
typed CB lifecycle contract; it must not be repaired by source-name matching
or by reloading the page through a separate workload path.

If the additive score chain needs fusion or producer grouping, that grouping
must be a generic typed compute-region / producer-chain lowering over explicit
IR dependencies, lifecycle intervals, and compatible tile domains.  It must not
be implemented as an adjacent-GEMM or MLA-specific source-shape matcher.

## Later T9 Checkpoints

- T9.4 sparse/ragged attention: indexed sparse-block traversal plus ragged
  valid lengths feeding attention compute.
- T9.5 chunk recurrence/scan: multi-chunk loop-carried device state.
- T9.6 multi-block flash decode: split blocks with exact-CB
  publish/consume and partial combine.

Each later checkpoint must define its own narrow admitted shape and direct
runtime correctness gate before broadening.

## Validation Plan

Structure/source:

- the grouped GEMM lowered executable contains A generic value bindings
  derived from `GroupOffsets[g]` and `GroupSizes[g]`;
- source contains no raw `GroupOffsets` / `GroupSizes` loads;
- source contains a real `matmul_tiles` compute path for the grouped GEMM;
- the paged GQA executable contains value-expr K/V tile-start ABI bindings
  and generic bound-value bindings for both static page steps;
- the paged MLA executable contains value-expr latent-KV and K-PE
  tile-start ABI bindings and generic bound-value bindings for both static page
  steps;
- source contains no raw `PageTable` / `CacheSeqLens` loads and no workload
  decode registry;
- source contains the existing flash partial-combine sequence rather than a
  separate paged-decode compute path;
- source contains two explicit score GEMM contributions for MLA and keeps
  latent-KV live until the value GEMM;
- no workload registry, grouped-GEMM frontend op, paged-decode frontend op, or
  MLA frontend op is introduced.

Runtime:

- direct runtime runs under the repository TT-Sim setup;
- group offsets and sizes are non-uniform, including at least one zero or
  partial group;
- host reference computes each group independently with zero-padded rows.
- paged GQA uses two static KV page steps, non-contiguous page ids, ragged
  sequence lengths, shared KV head semantics, and a host flash-attention
  reference;
- paged MLA uses two static latent/K-PE page steps, non-contiguous page ids,
  ragged sequence lengths, score accumulation from Q-nope and Q-PE, latent-KV
  value reuse, and a host MLA reference;
- page-addressed QK and AV micro-tests exercise both page 0 and page 1 so table
  constants and host materialization are covered independently from the full
  GQA tile.
- full GEMM/online-softmax flash paths that contain GEMM plus
  `reduce_tile` and `exp2_tile` / `recip_tile` must run as positive bf16
  direct-runtime correctness gates for the admitted T9.2/T9.3 shapes.
- a non-softmax MLA score-only slice remains a positive direct-runtime gate so
  the T9.3 additive GEMM chain is covered independently from the full
  online-softmax decode.

Unsupported diagnostics:

- if row-page A materialization cannot feed GEMM as a compute-compatible live
  form, the backend must reject with a typed admission reason rather than
  recovering from names or source text;
- if page-addressed K/V materialization, ragged row bounds, or exact-CB
  lifecycle cannot feed the existing flash path, the backend must reject with
  a typed admission reason rather than recovering from names, argument order,
  or generated source text;
- if page-addressed latent-KV / K-PE materialization, retained latent-KV input
  lifetime, or the additive score GEMM sequence cannot feed the existing flash
  path, the backend must reject with a typed admission reason rather than
  adding a workload-specific side path.
