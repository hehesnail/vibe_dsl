# Blackhole T10 Partial-K Reducer Protocol

## Role

This document defines the T10.4 replacement for the current K-sharded GEMM
direct-runtime partial-sum path.

Overall architecture:
`final_blackhole_backend_redesign.md`.
Current status:
`tasks/progress.md`.

## Problem

The existing K-sharded GEMM runtime path proves numerical correctness by
running one `logical_grid_z` wave at a time.  Shard `0` writes final `C`;
later shards write a private partial-C buffer and the runtime issues a
separate device tile-add program to accumulate into final `C`.

That path is a useful correctness barrier, but the reducer protocol itself is
not explicit owner truth.  Runtime currently derives the reducer from GEMM
shape, output buffer, sharded accessor metadata, and `logical_grid_z`.  T10.4
requires that reducer ownership, scratch placement/lifetime, synchronization,
route, transport, accumulation order, and final-writer timing cross
`TTProgram -> ExecutableSpec` as typed records.

## Goal

Introduce a durable reducer contract for partial-K GEMM:

```text
TTProgram.reducer_plans -> ExecutableSpec.reducer_plans -> BlackholeModule
```

The direct runtime may execute the partial-K reducer only when it consumes an
admitted reducer plan.  It must not reconstruct reducer semantics from source,
buffer names, argument positions, neighboring builtins, generated kernel text,
or runtime observations.

## Representation Contract

`TTReducerPlan` is the owner record.

Required fields:

- `name`
- `reducer_kind`: currently `partial_k_sum`
- `compute_op_plan`
- `compute_op_plan_index`
- `target_buffer`
- `target_buffer_distribution`
- `target_buffer_distribution_index`
- `scratch_buffer`
- `scratch_scope`
- `scratch_layout`
- `scratch_memory_space`
- `scratch_lifetime`
- `producer_axis`
- `producer_count`
- `logical_grid`
- `tile_shape`
- `reduction_op`
- `transport_kind`
- `route_kind`
- `accumulation_order`
- `final_writer_timing`
- `final_writer_producer`
- `required_semaphore_plan_indices`
- `required_sync_plan_indices`
- `remote_core_descriptor_indices`
- `admission_status`
- `unsupported_reason`

For the current single-card direct-runtime admissible protocol:

- `producer_axis = logical_grid_z`
- `producer_count = logical_grid[2]`
- `scratch_scope = per_target_buffer`
- `scratch_layout` / `scratch_memory_space` match the target buffer
  distribution
- `scratch_lifetime = one_producer_wave`
- `transport_kind = device_tile_add`
- `route_kind = local_same_device_sharded_tile`
- `accumulation_order = ascending_producer_id`
- `final_writer_timing = producer_0_writes_final_then_later_producers_reduce`
- `final_writer_producer = 0`
- semaphore, sync, and remote route index arrays are explicit and may be
  empty only for this local host-sequenced direct-runtime protocol

The scratch buffer is runtime-private storage with the target buffer's
materialization and distribution.  It is not a new public tensor and does not
need a `SpatialPlan` layout spec.

## Validation Contract

`ValidateTTProgram` rejects:

- missing reducer names or unsupported reducer kinds;
- missing compute-op references or out-of-bounds compute-op indices;
- reducer records attached to non-GEMM compute ops;
- missing or invalid target buffer distribution references;
- producer counts smaller than two;
- `producer_axis != logical_grid_z` for `partial_k_sum`;
- non-positive or rank-3-mismatched `logical_grid`;
- non-positive or rank-2-mismatched `tile_shape`;
- unsupported reduction ops, transport kinds, route kinds, accumulation
  orders, or final-writer timing;
- admitted records that still carry `unsupported_reason`;
- unsupported records without `unsupported_reason`;
- semaphore or sync indices outside their owning TTProgram arrays.

## Runtime Contract

`ExecutableSpec.reducer_plans` is the canonical runtime input.

For an admitted `partial_k_sum` plan, direct runtime launches producer shards
in the plan's `accumulation_order`.  Producer `final_writer_producer` writes
final `C`.  Later producers write `scratch_buffer`; after each producer wave,
runtime invokes the typed `transport_kind` reducer to add scratch into final
`C`.  The scratch buffer is allocated according to the plan and is released
after the producer wave is reduced.

If a K-sharded GEMM reaches runtime without an admitted matching reducer plan,
the executable must fail closed with a typed unsupported reason before
launching.  Runtime must not fall back to the old implicit z-wave inference.

## Verification

This slice is verified by:

- structure tests proving K-sharded GEMM materializes exactly one
  `TTReducerPlan` and projects it to `ExecutableSpec.reducer_plans`;
- validator tests rejecting malformed reducer records;
- direct-runtime positive tests for the small and many-core bf16 K-sharded
  GEMM cases proving numerical correctness through `BlackholeModule`;
- source/runtime guards proving partial-K direct runtime consumes a reducer
  plan and fails closed when a K-sharded GEMM has no admitted reducer plan;
- compile gate: `cmake --build build -j32`.

## Completion Criteria

T10.4 is complete only when the current K-sharded GEMM correctness path is
owned by `TTReducerPlan -> ExecutableSpec.reducer_plans`, positive direct
runtime correctness still passes for bf16 partial-K GEMM, malformed or missing
reducer contracts fail closed, and the old implicit reducer inference is no
longer a public runtime protocol.
