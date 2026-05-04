# Blackhole T4/T5 Accessor And Sharded GEMM Design

## Role

This document defines the active task-level boundary for T4 external
accessor/runtime ABI expansion and the dependent T5 sharded GEMM/layout
variants.

It is not a second overall design document.  The durable chain remains:

```text
Normalized Tile TIR
  -> SpatialPlan
  -> TTProgram
  -> ExecutableSpec
```

Current execution status remains in `tasks/progress.md`.

## T4 Goal

T4 turns external accessor records from typed-but-not-admitted metadata into
real executable ABI records.

The owner truth is:

- `SpatialPlan.TensorPlacementIntent` for user/default tensor placement
- `TTBufferDistributionPlan` for low-level address layout
- `TTTensorMemoryConfigPlan` and `TTReshardPlan` for placement conversion
- `TTABIPlan` / `TTAccessorSpec` for kernel-visible accessor ABI
- `ExecutableSpec` for leaf projection and direct-runtime admission

The runtime and codegen may consume these records.  They must not infer
sharding, page metadata, buffer role, or accessor shape from source text,
names, argument positions, or accessor kind strings.

## T4 Accessor Contract

An external accessor is admitted only when the executable record states:

- exact buffer identity
- accessor kind:
  `interleaved_accessor_cta`, `sharded_accessor_cta`, or
  `page_indexed_accessor_cta`
- compile-time arg offset and count
- common-runtime arg offset and count
- TT-Metal `tensor_accessor::ArgConfig` bits
- transport page size when the kernel uses page transport
- layout and memory space
- for sharded forms, buffer distribution fields:
  shard grid, per-core shard shape, sharding strategy, shard orientation,
  logical-index mapping, and core-local address mapping

`compile_time_arg_count` must match the actual TT-Metal accessor ABI for the
selected `args_config_bits` and buffer distribution.  Multiple accessors in a
kernel must have non-overlapping offsets that match the TIR builtin
`accessor_slot` operands.
For static sharded L1 buffers, this count is the TT-Metal
`TensorAccessorArgs` count after `BufferDistributionSpec` converts tensor and
shard shapes into tile-page space and squeezes adjacent dimensions.  It is not
the raw shard-grid rank.

Unsupported cases fail from executable records before source/runtime guessing.
Initial direct-runtime support may admit a narrower static-metadata subset,
but unsupported dynamic/common-runtime or page-shape variants must carry typed
diagnostics instead of falling back to interleaved assumptions.

## T4 Non-Goals

- Do not add a second source emitter or legacy external runner path.
- Do not recover accessor metadata from generated C++ source.
- Do not treat `sharded_accessor_cta` as a string switch that bypasses
  `TTBufferDistributionPlan`.
- Do not claim DRAM-sharded production weights or N-D production sharding
  unless the matching executable accessor records and runtime admission exist.

## T4 Landed Surface

The admitted T4 subset is intentionally narrow and explicit:

- 64B page-indexed DRAM transport accessors are represented as
  `page_indexed_accessor_cta` with `layout = page_indexed`, positive
  `transport_page_size`, and matching `TTBufferDistributionPlan.page_size_bytes`.
- static external sharded L1 accessors are represented as
  `sharded_accessor_cta` with sharded buffer distribution fields sufficient for
  TT-Metal `TensorAccessorArgs`.
- codegen resolves TensorAccessor compile-time offsets from executable
  accessor records by buffer identity, not from stale TIR immediates.
- direct runtime constructs interleaved/page-indexed DRAM buffers and static
  sharded L1 MeshBuffers from executable buffer distribution records.
- unsupported dynamic/common-runtime accessor metadata and missing page/shard
  metadata fail closed from executable records.

The T4 implementation also keeps GEMM accumulator canonicalization separate
from accessor admission: a same-subject loop-carried SpatialPlan boundary is
not by itself evidence that an accumulator reload is required.  Fresh or
precleared GEMM fragments may still canonicalize to `clear_accum = true`,
while post-merge materialization and true live-in accumulator cases remain
typed lowering decisions.

## T5 Goal

T5 admits GEMM/layout variants that depend on real tensor placement.

The owner truth is:

- input/output placement intent in `SpatialPlan`
- physical memory and address ABI in `TTProgram`
- `TTOpShardingContract` and `TTPlacementResolutionPlan` for supported GEMM
  operand/output placements
- explicit `TTReshardPlan` or typed reject when a producer/consumer placement
  conflicts
- `ExecutableSpec` accessor and runtime records for the admitted layout

T5 starts only after the T4 accessor surface needed by a GEMM variant either
has source/spec/direct-runtime admission or a precise typed reject.

## T5 Non-Goals

- No implicit retile or work-coarsening.
- No sharded GEMM claim based only on metadata projection.
- No layout variant that changes logical work mapping unless the retile /
  work-coarsening plan and source-region/address mapping change together.
- No distributed mesh/CCL/NoC production claim.

## T5 Landed Surface

The admitted T5 subset is the first static external sharded-L1 GEMM layout:

- external `A`, `B`, and `C` tensors may carry explicit block-sharded L1
  placement intent when their shard grid is covered by the kernel work
  mapping;
- the selected GEMM consumes T4 `TTABIPlan` / `ExecutableSpec`
  `sharded_accessor_cta` records, including typed accessor offsets/counts,
  layout, memory space, and buffer distribution fields;
- direct runtime executes the admitted single-core bf16-input / fp32-output
  GEMM and the hardened multi-core GEMM through `BlackholeModule` using
  sharded L1 MeshBuffers.  The covered multi-core cases include a 2x2 smoke
  shape and an 11x10 many-core shape (`M=320`, `N=352`, `K=256`) that uses
  all 110 logical worker cores.  They cover `A` and `B` height-sharded L1
  inputs and a block-sharded L1 output, with both bf16-input / fp32-output
  and all external bf16 input/output tensors;
- all-external-bf16 output uses the explicit preclear plus
  `clear_accum = false` post-merge cast path so the materialized bf16 tile is
  published with `publication_protocol = pack_tile`, not an unadmitted
  compute-thread mailbox/tilize cast path;
- the first K-dimension sharded GEMM direct-runtime path represents A/B
  placement as width-sharded L1 over their K dimension, projects
  `logical_grid_z` from `T.Kernel(grid_x, grid_y, k_shards)`, uses
  per-work `bk` to read the correct K slice, rotates each logical-z wave across
  physical cores, and uses a device-side TT-Metal tile-add reduction program to
  merge later K-shard partial-C scratch tiles into the first wave's final `C`
  buffer before host readback.  This is a runtime-issued device reduction with
  blocking wave barriers, not a production single-launch semaphore/atomic
  reduce protocol;
- an external sharded L1 accessor whose `shard_grid_shape` is not covered by
  the attached `TTCoreGroup.work_packets` fails in `ValidateTTProgram` with a
  retile/work-coarsening diagnostic.

The work-mapping reject is intentionally attached to the accessor ABI
boundary, not to every device-local sharded distribution.  Internal
materialization records can have different shard-grid mechanics and are not
runtime-visible external accessor claims.

### Production Partial-K Follow-Up

The current K-dimension sharded GEMM path is not the final production
cross-core reduction protocol.  The production follow-up belongs to the
distributed / NoC scheduling lane and must be tracked explicitly, not hidden
under a generic runtime bucket.

That follow-up must add typed `TTProgram` / `ExecutableSpec` records for:

- reducer ownership for each output tile;
- partial-C scratch placement and lifetime;
- producer-to-reducer semaphore ids;
- remote worker NOC routes;
- NoC read/write or multicast transport choice;
- reducer accumulation order and final writer timing.

Only after those records are projected and validated may K-sharded GEMM claim a
single-launch or fused-launch semaphore/NoC partial-reduce implementation.
When that production protocol lands, the current runtime-issued blocking
z-wave tile-add reduction path must be deleted or folded into the typed
protocol implementation.  It must not remain as a parallel special-case
execution path.

## Validation

T4 validation must cover:

- `TTProgram` and executable projection tests for sharded/page-indexed
  accessor records
- codegen tests proving admitted source uses typed accessor offsets and rejects
  unsupported accessor ABI shapes
- direct-runtime tests proving admitted cases run through `BlackholeModule`
  and unsupported cases fail closed from executable records
- serialization tests for accessor records

T5 validation must cover:

- GEMM placement-contract tests for admitted input/output memory configs
- source/spec tests proving GEMM accessor records match placement decisions
- direct-runtime bf16 correctness for the first admitted sharded layout,
  including multi-core sharded execution, a many-core shape that uses the full
  110-worker logical grid, all external bf16 input/output coverage, and
  K-dimension sharded partial-sum correctness through `logical_grid_z`
- typed rejects for unsupported placement, conversion, page-shape, retile, or
  work-coarsening combinations
