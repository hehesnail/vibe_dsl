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
  -> segment_row_start / segment_row_count per-work descriptors for A
  -> row-page A materialization with zero-fill
  -> compute-compatible A live form for matmul_tiles
  -> grouped GEMM direct runtime correctness
```

This checkpoint is not satisfied by a grouped row-copy alone.  The row-bound
materialization must feed the admitted GEMM path through typed live-form /
materialization records or fail closed with a typed reason.

The admitted implementation materializes A from row-major DRAM pages into a
compute-compatible tiled CB:

- the reader consumes A `segment_row_start` / `segment_row_count`
  descriptors; raw `GroupOffsets` / `GroupSizes` loads are not present in
  source;
- the A accessor is page-indexed over 64-byte bf16 row tiles, so non-32-row
  group offsets are row-addressed instead of tile-addressed;
- each valid row tile is read into a scratch CB and copied into the nfaces
  tile layout expected by `matmul_tiles`;
- invalid rows zero the destination tile slices before publish;
- direct runtime treats segmented A as a raw row-major backing tensor and
  decodes grouped output as compact `groups * tile_m` rows by `tile_n`
  columns.

The scratch CB / `copy_cb_page` step is a data-movement implementation detail
needed to satisfy Blackhole NOC alignment while preserving the typed
descriptor contract.  It is not a workload-level side channel.

## Later T9 Checkpoints

- T9.2 paged GQA decode: page/block-table KV reads with ragged cache lengths
  and admitted partial combine.
- T9.3 paged MLA decode: paged latent/KV access through admitted page-table
  and ragged-bound records.
- T9.4 sparse/ragged attention: indexed sparse-block traversal plus ragged
  valid lengths feeding attention compute.
- T9.5 chunk recurrence/scan: multi-chunk loop-carried device state.
- T9.6 multi-block flash decode: split blocks with exact-CB
  publish/consume and partial combine.

Each later checkpoint must define its own narrow admitted shape and direct
runtime correctness gate before broadening.

## Validation Plan

Structure/source:

- the grouped GEMM lowered executable contains A `segment_row_start` and
  `segment_row_count` per-work descriptors derived from
  `GroupOffsets[g]` and `GroupSizes[g]`;
- source contains no raw `GroupOffsets` / `GroupSizes` loads;
- source contains a real `matmul_tiles` compute path for the grouped GEMM;
- no workload registry or grouped-GEMM frontend op is introduced.

Runtime:

- direct runtime runs under the repository TT-Sim setup;
- group offsets and sizes are non-uniform, including at least one zero or
  partial group;
- host reference computes each group independently with zero-padded rows.

Unsupported diagnostics:

- if row-page A materialization cannot feed GEMM as a compute-compatible live
  form, the backend must reject with a typed admission reason rather than
  recovering from names or source text.
