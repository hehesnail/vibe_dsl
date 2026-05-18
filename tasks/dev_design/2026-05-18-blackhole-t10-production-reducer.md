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
- `route_kind = local_same_device_sharded_tile` for sharded L1 targets or
  `local_same_device_interleaved_tile` for interleaved DRAM targets
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
final `C`.  Later producers write `scratch_buffer`; after each producer
shard's output waves finish, runtime invokes the typed `transport_kind`
reducer for those waves to add scratch into final `C`.  The scratch buffer is
allocated according to the plan and reused only within this host-sequenced
producer-shard reduction window.

Current direct-runtime admission supports output grids that need multiple
temporal waves per producer.  The runtime keeps the same typed reducer plan:
producer `0` writes final `C`, later producers write the typed scratch buffer,
and each producer wave is reduced before the next producer is observed by the
host result path.

For logical output tiles covered by the physical launch core set, runtime uses
the `device_tile_add` transport.  For later temporal output waves that reuse
physical workers after the first `physical_cores.size()` logical tiles, the
direct runtime performs a host-mediated float32 page add from the typed
scratch buffer into the typed final buffer.  This is still a runtime
implementation of the admitted `partial_k_sum` reducer plan; it is not a
separate public protocol or source-derived fallback.

Large logical output grids must not automatically make the target sharded L1
grid equal to the logical tile grid.  The logical grid belongs to work
coverage; the resident sharded L1 grid belongs to physical memory placement
and must satisfy the core/L1-bank limit.  A larger per-shard `shard_shape`
may cover multiple logical output tiles per resident shard.  The verified
`20x20x4` direct-runtime case uses C resident grid `10x10` and
`shard_shape=(64,64)` so the `400` logical output tiles are covered by `100`
resident L1 shards plus temporal work/reduction.

Shape-general correctness must not depend on the full logical C tensor being
resident in L1.  When the full output shape cannot fit as an admitted sharded
L1 resident view under the active CB/L1 budget, the admitted typed path is an
interleaved DRAM output/scratch reducer:

- target buffer distribution is `interleaved` / `DRAM` with
  `interleaved_linear_page` indexing;
- scratch layout and memory space match the target distribution;
- route kind is `local_same_device_interleaved_tile`;
- producer shards still run in ascending z order, producer 0 writes final C,
  later producers write the typed scratch buffer, and runtime reduces scratch
  pages into final C using the admitted reducer plan.

This makes large MNK support a memory-placement decision rather than a new
workload-specific semantic path: L1 sharded resident outputs remain supported
when they fit; larger full-output tensors use DRAM as the full tensor owner
and L1 only as the kernel staging/CB working set.

Large MNK support also must not be modeled as "increase the logical output
grid until it is large."  For shapes beyond the one-output-tile-per-work-item
form, the logical/core grid stays within the available core budget and each
work item may cover multiple C tiles using explicit core-internal M/N serial
loops.  When a producer's K shard is larger than the working CB window, that
same work item also loops over K chunks before publishing its partial C
tiles.  The direct runtime reduces interleaved DRAM partial-K scratch by
adding the full output scratch buffer into final C after each nonzero
producer shard, because a single work item may have written many output
pages.

Compute-side CB lifetime must follow the explicit reader/compute events in
that core-internal loop form.  Repeated serial loops are not proof that input
CB pages are invariant: retaining A/B pages across a local output-tile loop
can replay stale tiles.  The admitted path therefore uses the ordinary
per-consume pop/reacquire protocol for compute input CB pages.

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
- temporal correctness guards proving `13x10x4` partial-K output grids with
  `130` logical output tiles over `110` physical launch cores and large
  `20x20x4` output grids with C resident grid `10x10` /
  `shard_shape=(64,64)` run through direct runtime and match the torch bf16
  reference;
- large MNK DRAM-output guard proving `M=N=512,K=2048,k_shards=4` runs on a
  bounded `4x4x4` logical/core grid, assigns `4x4` output tiles to each core,
  tiles each K shard as two `k_tile=256` chunks, runs through direct runtime
  with an interleaved DRAM output/scratch reducer, and matches the torch bf16
  reference;
- compile gate: `cmake --build build -j32`.

## Completion Criteria

T10.4 is complete only when the current K-sharded GEMM correctness path is
owned by `TTReducerPlan -> ExecutableSpec.reducer_plans`, positive direct
runtime correctness still passes for bf16 partial-K GEMM, temporal output
waves do not return wrong values, malformed or missing reducer contracts fail
closed, and the old implicit reducer inference is no longer a public runtime
protocol.  Fully device-side temporal sharded-L1 window ownership remains a
future resource-planning expansion, but the current single-card direct
runtime owns correctness for the admitted `13x10x4` temporal-wave subset, the
larger `20x20x4` logical-grid / capped-resident-grid subset, and larger MNK
core-tiled full-output tensors through the interleaved DRAM output/scratch
reducer path.
