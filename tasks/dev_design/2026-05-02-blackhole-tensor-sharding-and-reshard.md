# Blackhole Tensor Sharding And Reshard Design

## Role

This document defines the task-level design for real tensor/value sharding
and reshard support in the Blackhole backend.

It is not a second overall design document.
The overall chain remains:

```text
Normalized Tile TIR
  -> SpatialPlan
  -> TTProgram
  -> ExecutableSpec
```

Current implementation status still lives in `tasks/progress.md`.

## Problem

The current Blackhole implementation has a `TTBufferDistributionPlan`.
That object is useful, but it is not a complete sharding model.

The current implementation mostly derives low-level buffer distribution from:

- `SpatialPlan.LayoutSpec.scope`
- local / shared / global buffer class
- the selected `TTCoreGroup`
- CB-backed or staged-copy address requirements

That is enough for the current direct-runtime address ABI:

- interleaved DRAM runtime buffers
- staged-copy resident L1 / CB-backed views
- page-indexed address contracts

It is not enough for real tensor sharding.

Real sharding must answer all of these questions:

- which logical tensor or value is sharded
- which tensor dimensions are partitioned
- which mesh / core grid owns the shards
- what per-shard tensor shape is used
- what memory space and buffer type hold the shards
- which operation requires which input and output placements
- what happens when a producer placement conflicts with a consumer placement
- whether an explicit reshard / layout conversion is required

The current one-buffer-one-distribution plan cannot represent this whole
problem.

## TT-Metal Ground Truth

The design must match TT-Metal / TTNN behavior instead of inventing a
TileLang-only abstraction.

### MemoryConfig Is The Tensor Placement Contract

TTNN represents tensor memory placement through `MemoryConfig`.
The relevant fields are:

- `TensorMemoryLayout`
- `BufferType`
- optional `ShardSpec`
- optional `NdShardSpec`

For legacy 2D sharding, `ShardSpec` carries:

- `CoreRangeSet grid`
- per-core tensor `shape`
- `ShardOrientation`

For newer generalized sharding, `NdShardSpec` carries:

- N-D `shard_shape`
- `CoreRangeSet grid`
- `ShardOrientation`
- `ShardDistributionStrategy`

`height`, `width`, and `block` are tensor memory-layout strategies.
`row_major` and `col_major` are core traversal orientations.
They are separate concepts.

### Callers Set Target Sharding

In TTNN, sharding is normally set by the caller, model config, or op wrapper.
Examples:

- `ttnn.create_sharded_memory_config(...)` builds a sharded `MemoryConfig`
  from a shape, core grid, strategy, and orientation.
- Models directly construct
  `MemoryConfig(TensorMemoryLayout::WIDTH_SHARDED, BufferType::DRAM,
  ShardSpec(...))`
  for DRAM-sharded weights.
- Raw TT-Metal examples build `ShardSpecBuffer` and pass
  `BufferShardingArgs` into buffer creation.

The convenience helper's L1 note is not a system-wide restriction.
TTNN models and tests also construct DRAM-sharded tensors directly through
`MemoryConfig(..., BufferType::DRAM, ShardSpec(...))`.

The program factory does not recover sharding from names.
It consumes the tensor or output tensor's already-selected memory config.

### Conversion Is Explicit

TTNN uses explicit conversion operations:

- `to_memory_config`
- `interleaved_to_sharded`
- `sharded_to_interleaved`
- `reshard`

`to_memory_config` takes a target `MemoryConfig`.
If the current tensor and target config differ, it dispatches to the needed
conversion path.

This is the key rule TileLang must adopt:

> Sharding conflicts are not silently solved by runtime recovery.
> They are explicit placement-conversion edges or typed rejects.

### Program Factories Consume Sharding

Program factories and kernels use the already materialized tensor/buffer
sharding:

- they read `tensor.memory_config().memory_layout()`
- they read `tensor.shard_spec()`
- they read buffer `ShardSpecBuffer` / distribution specs
- they validate supported combinations
- they select runtime args, address generators, and compute grids

They do not decide the semantic tensor sharding from source text, buffer
names, or argument order.

### Repo Evidence

The design above is based on these local TT-Metal / TTNN surfaces:

- `ttnn/ttnn/core.py`:
  `create_sharded_memory_config` maps caller-provided shape, grid, strategy,
  and orientation to `MemoryConfig`.
- `tt_metal/api/tt-metalium/experimental/tensor/spec/memory_config/memory_config.hpp`:
  `MemoryConfig` owns memory layout, buffer type, `ShardSpec`, and
  `NdShardSpec`.
- `tt_metal/api/tt-metalium/buffer.hpp`:
  `ShardSpec`, `ShardSpecBuffer`, `ShardedBufferConfig`, and
  `BufferShardingArgs` are the low-level buffer materialization surface.
- `tt_metal/impl/tensor/spec/layout/tensor_layout.cpp`:
  `TensorLayout::compute_buffer_sharding_args` materializes a sharded
  `MemoryConfig` into buffer sharding args and distribution specs.
- `ttnn/cpp/ttnn/operations/core/to_memory_config/to_memory_config_op.hpp`:
  `to_memory_config` dispatches explicit target-placement conversions.
- `ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded`
  and `reshard`:
  device ops validate and consume caller-provided output memory configs.
- `ttnn/cpp/ttnn/operations/matmul/device/config/matmul_program_config.cpp`:
  matmul derives program config from already-sharded inputs / requested
  output placement.
- `models/tt_transformers/tt/model_config.py`,
  `models/tt_transformers/tt/prefetcher.py`,
  and related tests:
  production-style model code constructs and requires DRAM-sharded weights.
- `tt_metal/programming_examples/vecadd_sharding/vecadd_sharding.cpp`:
  raw TT-Metal constructs `ShardSpecBuffer` and `BufferShardingArgs`
  explicitly.

The corresponding current TileLang surfaces are:

- `tilelang/language/annotations.py`:
  `annotate_layout` already exposes a block-attribute user surface for
  buffer-index layout maps.
  This is local layout / index-transform evidence, not tensor sharding.
- `tilelang/language/ast/ir.py` and `tilelang/language/allocate.py`:
  `match_buffer`, `alloc_buffer`, `alloc_shared`, and `alloc_fragment`
  expose buffer subjects, shape, dtype, scope, strides, and buffer type.
  They do not currently expose a TTNN-style memory config.
- `tilelang/language/kernel.py`:
  `T.Kernel(...)` exposes the logical work-item grid.
  It is not a TT physical core grid or tensor shard grid.
- `src/transform/common/spatial_plan.h`:
  `LayoutSpec` carries target-independent subject / scope /
  distribution-kind evidence, not TT sharding.
- `src/transform/build_spatial_plan.cc`:
  current `distribution_kind` is derived mainly from buffer scope.
- `src/transform/common/tt_target_program.h`:
  `TTBufferDistributionPlan` carries low-level buffer distribution and
  address ABI fields.
- `src/transform/build_tt_program.cc`:
  current sharded L1 distribution is derived from `LayoutSpec`,
  `TTCoreGroup`, CB/materialization facts, and source-buffer evidence.
- `src/transform/validate_tt_program.cc`,
  `src/target/blackhole_module.cc`,
  and `src/target/rt_mod_blackhole.cc`:
  validators and runtime admission consume the projected buffer distribution
  address contract.

## Current TileLang State

### SpatialPlan

`LayoutSpec` currently carries:

- subject
- scope
- distribution kind
- logical and local tile layout fields when recoverable
- execution-unit association

It does not carry a real tensor sharding intent.
`distribution_kind = shared_visible` means a target-independent visibility
class.
It does not mean "this tensor is width-sharded" or "this operation requires a
specific shard shape".

### TTProgram

`TTBufferDistributionPlan` currently carries physical buffer placement and
address ABI fields:

- distribution kind
- layout
- memory space
- page size
- shard grid shape
- sharding strategy
- shard orientation
- per-core shard shape
- source buffer / source region binding
- logical-index mapping
- core-local address mapping

This is the right object for low-level buffer placement.
It is not the right owner for user intent, op-level placement requirements,
or producer/consumer sharding conflicts.

The current staged-copy path should be reinterpreted as:

```text
interleaved DRAM source tensor
  -> explicit materialization / conversion edge
  -> resident L1 sharded working view
```

It should not be described as proof that TileLang has complete tensor
sharding.

### ExecutableSpec / Runtime

`ExecutableSpec` currently projects buffer distribution fields as a runtime
address contract.

That is correct for leaf consumption.
The runtime must continue to consume final typed placement and conversion
records only.
It must not become the place that infers missing sharding.

## DSL / User Interface Design

The missing frontend piece is explicit user placement intent.
Without it, the backend can only derive local scratch placement from buffer
scope and access shape.
That is not enough for TTNN-style sharded inputs, outputs, weights, or
intermediate values.

### User-Facing Objects

TileLang should introduce a small, TT-Metal-aligned placement API on the
language surface:

```python
T.MemoryConfig(
    memory_layout="interleaved" | "height_sharded" | "width_sharded" |
                  "block_sharded" | "nd_sharded",
    buffer_type="dram" | "l1",
    shard=T.ShardSpec(
        grid=T.CoreGrid(x=..., y=...) | T.CoreRangeSet(...),
        shape=(...),
        orientation="row_major" | "col_major",
    ) | T.NDShardSpec(...),
)
```

The names intentionally mirror TTNN's `MemoryConfig`, `ShardSpec`, and
`NdShardSpec` concepts.
The Python implementation may use enums or typed constants instead of raw
strings, but the public meaning must stay the same:

- `memory_layout` is interleaved / height / width / block / N-D sharding
- `buffer_type` is the storage class, such as DRAM or L1
- `shard.grid` is the core or storage grid
- `shard.shape` is the per-shard tensor shape
- `shard.orientation` is row-major or column-major traversal

Convenience constructors can be added, but only as sugar for the same object:

```python
T.interleaved_dram()
T.sharded_l1(strategy="height", grid=T.CoreGrid(x=8, y=8), shard_shape=(..., ...))
T.sharded_dram(strategy="width", grid=T.CoreGrid(x=12, y=1), shard_shape=(..., ...))
```

These helpers must not introduce a second semantic model.
They must lower to `T.MemoryConfig`.

### Attachment Surface

The first durable attachment surface should be a block annotation, analogous
to existing `T.annotate_layout`:

```python
T.annotate_memory_config({
    A: T.interleaved_dram(),
    W: T.sharded_dram(
        strategy="width",
        grid=T.CoreGrid(x=dram_cores, y=1),
        shard_shape=(K, padded_N // dram_cores),
        orientation="row_major",
    ),
    C: T.interleaved_dram(),
})
...
```

This should lower to a single typed block attr, for example
`tl.memory_config_map`, whose keys are buffer data vars and whose values are
typed placement objects.
`BuildSpatialPlan` consumes that attr and emits
`TensorPlacementIntent(source="user", ...)`.

The annotation surface is the first implementation target because it matches
TileLang's existing attribute mechanism and avoids changing every buffer
constructor at the same time.
After the attr path is working, the same config can be accepted as sugar on
buffer constructors:

```python
A = T.match_buffer(a, (M, K), dtype, memory_config=T.interleaved_dram())
W = T.match_buffer(w, (K, N), dtype, memory_config=T.sharded_dram(...))
S = T.alloc_shared((tile_m, tile_n), dtype, memory_config=T.sharded_l1(...))
```

Constructor sugar must lower to the same placement intent.
It must not create a separate hidden side channel.

### What The User Can And Cannot Specify

The user can specify:

- external input, weight, and output tensor memory config
- explicit intermediate materialization target when a named buffer/value exists
- whether a consumer boundary may insert a reshard conversion
- preferred shard grid, shard shape, strategy, and orientation
- DRAM-sharded and L1-sharded placements as distinct configs

The user cannot specify sharding by:

- naming a buffer a certain way
- using `scope="shared"` or `scope="global"`
- using `T.annotate_layout`
- relying on `T.Kernel(grid_x, grid_y)` as a physical shard grid
- relying on backend work-packet mapping to change tensor placement

`scope` continues to describe buffer storage class for TIR allocation.
`T.annotate_layout` continues to describe local layout / index mapping.
`T.Kernel` continues to describe logical work items.
Those three concepts may be used as validation evidence, but none of them is
the user sharding API.

### Defaults

Default behavior must be boring and explicit:

- a global external buffer with no memory config defaults to interleaved DRAM
- an output with no memory config defaults to the current external ABI policy,
  currently interleaved DRAM for runtime-bound outputs
- `alloc_shared` defaults to per-worker, per-work-item L1 / CB-backed scratch
- `alloc_fragment` and local accumulators default to local live forms
- no default may silently transform a full tensor into a sharded tensor

If an operation contract prefers a different placement, placement resolution
either inserts an explicit conversion plan or rejects.

### Reshard Policy In The DSL

Reshard permission is a property of the placement request, not a runtime
guess.
The user-facing config should carry:

```python
T.MemoryConfig(..., allow_reshard=True | False)
```

or the equivalent field on the annotation entry.

Meaning:

- `allow_reshard=True` lets the planner insert a supported conversion when an
  op contract needs a different config
- `allow_reshard=False` means the requested config is a hard requirement
- if two consumers need incompatible configs and conversion is not allowed or
  not admitted, planning fails with a typed conflict diagnostic

This mirrors TTNN usage where model code explicitly calls
`to_memory_config`, `interleaved_to_sharded`, `sharded_to_interleaved`, or
`reshard`.
TileLang may synthesize the conversion only when the user/op contract allows
it and the conversion is represented as `TTReshardPlan`.

### Examples

External DRAM-sharded weight:

```python
T.annotate_memory_config({
    W: T.sharded_dram(
        strategy="width",
        grid=T.CoreGrid(x=dram_cores, y=1),
        shard_shape=(K, padded_N // dram_cores),
        orientation="row_major",
        allow_reshard=False,
    )
})
...
```

L1-sharded working view materialized from an interleaved source:

```python
A = T.match_buffer(a, (M, K), dtype)
A_tile = T.alloc_shared((tile_m, tile_k), dtype)

T.annotate_memory_config({
    A: T.interleaved_dram(),
    A_tile: T.sharded_l1(
        strategy="block",
        grid=T.CoreGrid(x=8, y=8),
        shard_shape=(tile_m_per_core, tile_k_per_core),
        orientation="row_major",
        allow_reshard=True,
    ),
})
T.copy(A[...], A_tile[...])
```

The second example does not mean `alloc_shared` is globally sharded by
default.
It means the user requested a concrete resident L1 materialized view.
Planning must connect it to a source value through `TTReshardPlan` or reject.

### DSL Validation

The frontend / `BuildSpatialPlan` boundary must reject:

- memory config entries whose keys are not live TIR buffers
- duplicate configs for the same value in one scope unless the newer config
  creates an explicit value version or conversion boundary
- shard dimensions or shard shapes inconsistent with buffer rank and shape
- `height`, `width`, `block` used as orientation
- `row_major`, `col_major` used as strategy
- L1-sharded external runtime inputs without an admitted materialization or
  direct accessor plan
- `T.annotate_layout` values used as sharding proof
- `T.Kernel` extents used directly as shard grid without an explicit
  placement config

After this validation, `SpatialPlan` receives typed
`TensorPlacementIntent`.
Downstream phases never read Python helper objects or raw attrs as owner
truth.

## Design Goals

- Represent user/model/op tensor sharding intent explicitly.
- Provide a concrete TileLang DSL / annotation surface for that intent.
- Keep tensor/value placement separate from low-level buffer distribution.
- Represent per-op sharding contracts before physical source/runtime
  emission.
- Detect producer/consumer placement conflicts before codegen/runtime.
- Insert explicit reshard/materialization plans when a supported conversion
  exists.
- Fail closed with typed diagnostics when a conversion or placement is not
  admitted.
- Preserve TT-Metal terminology and semantics.
- Allow DRAM-sharded tensors and L1-sharded tensors as separate placements.
- Keep runtime/codegen as consumers of `ExecutableSpec`, not planners.

## Non-Goals

- No runtime-side sharding inference.
- No buffer-name based sharding roles.
- No assumption that every sharded tensor is L1.
- No assumption that one buffer has one placement for the entire program.
- No silent retile or work coarsening to make a placement fit.
- No overloading `T.annotate_layout`, `scope`, or `T.Kernel` to mean tensor
  sharding.
- No claim of distributed production support before mesh / CCL / NoC plans
  exist.

## Representation Design

### SpatialPlan: TensorPlacementIntent

`SpatialPlan` needs a `TensorPlacementIntent` object for target-independent
logical placement intent.

Required fields:

- logical subject identity
- logical value/version identity when the placement is value-specific
- source of intent:
  `user`,
  `op_contract`,
  `derived_default`,
  or
  `materialization_requirement`
- DSL origin:
  `memory_config_map`,
  constructor sugar,
  operation output policy,
  or
  derived default
- optional user-provided config identity for diagnostics
- logical tensor rank and shape evidence
- partitioned tensor dimensions
- replicated dimensions
- virtual mesh axes or logical device axes
- allowed memory-space class:
  `DRAM`,
  `L1`,
  or
  `either`
- allowed strategy class:
  `interleaved`,
  `height_sharded`,
  `width_sharded`,
  `block_sharded`,
  or
  `nd_sharded`
- whether reshard is allowed at consumer boundaries
- whether this placement is a hard user requirement
- anchors / access-region evidence

This layer must not contain TT physical core coordinates.
It may carry validated user hints, but only after they are checked against
current TIR shape and access evidence.
When the user provides a concrete TT core or DRAM grid, `SpatialPlan` records
only a target-independent binding requirement.
`TTProgram` resolves that requirement against `TTHardwareModel`.

Default behavior must be explicit:

- external global tensors default to interleaved DRAM unless the user or op
  contract states otherwise
- local fragment / accumulator values default to local live forms
- shared / CB-backed views default to a local materialization requirement,
  not a global tensor sharding intent
- user configs override defaults only after validation

### TTProgram: TTTensorMemoryConfigPlan

`TTProgram` needs `TTTensorMemoryConfigPlan`, a concrete plan that mirrors
TT-Metal's `MemoryConfig` model.

Required fields:

- logical value or tensor identity
- logical shape
- dtype / layout evidence
- memory layout:
  `INTERLEAVED`,
  `HEIGHT_SHARDED`,
  `WIDTH_SHARDED`,
  `BLOCK_SHARDED`,
  or
  `ND_SHARDED`
- buffer type:
  `DRAM`,
  `L1`,
  or an explicitly admitted variant
- grid:
  concrete `TTCoreGroup` reference or DRAM grid reference
- shard shape
- shard orientation
- shard distribution strategy for N-D sharding
- page shape / tile layout evidence when needed by buffer materialization
- resolved source of any user-provided grid or mesh binding
- whether a matching runtime accessor exists or materialization is required
- origin:
  user intent,
  op requirement,
  default,
  or
  conversion result

This object is the TileLang analogue of TTNN `MemoryConfig + ShardSpec`.
`TTBufferDistributionPlan` should be derived from it where a concrete buffer
must be allocated or addressed.
For external runtime buffers, it also drives backend admission:
direct runtime may reject a concrete config, but it must reject by reading
this plan and the projected executable record.

### TTProgram: TTOpShardingContract

Every TT leaf or higher-level admitted op family that cares about placement
needs a `TTOpShardingContract`.

Required contract fields:

- operation identity
- operand role
- accepted input memory layouts
- accepted buffer types
- accepted shard grids or grid classes when the op is grid-sensitive
- required shard-rank / strategy / orientation constraints
- output placement policy:
  inherit input,
  caller-specified,
  op-selected,
  or
  interleaved default
- whether the op may request an input conversion
- whether the output can be produced directly in the requested placement
- whether the op can write directly to external DRAM, resident L1, or a
  materialized output buffer
- typed reject reasons for unsupported combinations

For example:

- a simple elementwise leaf may accept matching input placements and produce
  the same output placement
- a matmul variant may require `input_a` and output to share a layout in some
  sharded configurations
- a DRAM-streaming matmul may accept DRAM-sharded weights but not arbitrary
  L1 sharding
- a staged shared-memory tile load may require an interleaved DRAM source and
  produce a resident L1 sharded working view

The contract belongs in planning.
It must not be reconstructed in source hooks.
If an operation has no sharding-sensitive contract yet, it must be treated as
accepting only the default placement classes already admitted by its current
implementation.

### TTProgram: TTPlacementResolutionPlan

Planning needs a placement-resolution pass over a value/use graph.
Its durable output is `TTPlacementResolutionPlan`.

Inputs:

- `SpatialPlan.TensorPlacementIntent`
- access-region evidence
- live-value edges
- `TTOpShardingContract`
- hardware model facts
- current live-form / materialization plans

Outputs:

- `TTTensorMemoryConfigPlan`
- selected placement per logical value version and per op result
- selected placement per consumer use when a use requires conversion
- typed conflicts
- required conversion edges
- explicit default-placement records for values that had no user config

Conflict handling must be deterministic:

1. If producer and all consumers accept one placement, select it.
2. If a user-specified placement exists and is legal, preserve it.
3. If a consumer requires a different placement and conversion is admitted,
   insert a conversion edge.
4. If multiple consumers require incompatible placements, either clone through
   explicit conversions or choose a common materialized placement only when
   every consumer contract accepts it.
5. If none of those are possible, emit a typed reject naming the producer,
   consumer, source placement, target placement, and missing conversion.

The resolver must not collapse two different value versions into one buffer
distribution just because they share a TIR buffer name.
If a value changes placement, the changed placement is a new value/materialized
view connected by `TTReshardPlan`.

### TTProgram: TTReshardPlan

Resharding must be represented by `TTReshardPlan`.
It is not a side effect of buffer distribution.

Required fields:

- source logical value
- target logical value or materialized view
- source tensor-memory-config plan
- target tensor-memory-config plan
- conversion kind:
  `interleaved_to_sharded`,
  `sharded_to_interleaved`,
  `reshard`,
  `dram_sharded_to_l1_sharded`,
  or
  `unsupported`
- source and target access regions
- materialization protocol
- transport plan references
- CB / semaphore / sync requirements when required
- whether conversion is compile-time, load-time, or runtime
- whether the conversion was user-authored or planner-inserted under
  `allow_reshard`
- typed admission reason if not supported

The current DRAM-to-resident-L1 staged copy can become the first admitted
conversion class.
Later conversion classes can follow TTNN's split:

- interleaved to sharded
- sharded to interleaved
- sharded to sharded
- DRAM-sharded weight streaming

### TTBufferDistributionPlan: Low-Level Buffer Placement

`TTBufferDistributionPlan` remains useful, but its role must be narrowed:

- allocate or address a concrete buffer
- describe physical memory space and page size
- describe core-local layout for resident buffers
- carry source-region ABI only for materialized views
- project into `ExecutableSpec` as a leaf address contract

It must not be the owner of:

- user sharding intent
- op sharding requirements
- placement conflict resolution
- reshard insertion

For compatibility with TT-Metal, when it represents a sharded buffer it must
be derivable from a concrete tensor-memory-config plan:

```text
Tensor memory layout  -> sharding_strategy
ShardSpec.grid        -> shard_grid_shape / attached grid
ShardSpec.shape       -> shard_shape
ShardSpec.orientation -> shard_orientation
BufferType            -> memory_space / buffer type
```

### ExecutableSpec

`ExecutableSpec` must project:

- concrete tensor-memory-config records needed by leaf consumers
- buffer distribution records
- reshard / conversion records
- materialization records
- typed admission results

Direct runtime can reject a projected conversion kind that it cannot execute,
but it must reject from the explicit conversion record.
It must not infer the conversion from source text or buffer names.

## Validation Contract

### SpatialPlan Validation

Reject:

- sharding intent without subject or logical value identity
- user memory-config annotations whose subject is not a live TIR buffer or
  named logical value
- duplicate user configs for the same value without an explicit value-version
  or conversion boundary
- partition dimensions outside tensor rank
- shard / replicate dimensions that overlap illegally
- shard shape inconsistent with static shape evidence when the relevant
  dimensions are static
- user sharding hints that cannot be validated against shape/access evidence
- target-specific physical coordinates in `SpatialPlan`
- layout specs that imply TT-specific sharding through names or scopes
- `T.Kernel` logical grid extents used as a shard grid without an explicit
  memory config

### TTProgram Validation

Reject:

- sharded tensor-memory-config plan without grid, shard shape, strategy, and
  orientation
- strategy / orientation conflation
- unresolved user grid / mesh binding after hardware-model resolution
- op placement not accepted by its sharding contract
- producer/consumer placement conflict without an admitted conversion plan
- conversion plan whose source and target configs are identical
- conversion plan missing source/target value identity
- planner-inserted conversion when `allow_reshard=False`
- DRAM-sharded placement routed through L1-only assumptions
- low-level `TTBufferDistributionPlan` that cannot be traced to a resolved
  tensor-memory-config plan or an explicitly local scratch/materialization
  requirement

### ExecutableSpec Validation

Reject:

- missing concrete placement records needed by kernels or accessors
- conversion records with unsupported or incomplete source/target configs
- buffer distributions whose sharding fields disagree with resolved tensor
  memory configs
- runtime-bound buffers whose distribution is not admitted by direct runtime
- external L1-sharded buffers without a projected accessor or materialization
  record
- fallback source-region recovery from names, arguments, or source text

## Implementation Order

The implementation is ordered so each checkpoint leaves the active IR chain
more explicit.
No checkpoint by itself is a claim that full tensor sharding is complete.

### S1: Lock The Current Boundary

Clarify in docs and tests that current `TTBufferDistributionPlan` is a
buffer placement / address ABI object.
It is not the full tensor sharding model.

No behavior change.

### S2: Add The DSL Placement Surface

Add the Python-facing placement objects:

- `T.MemoryConfig`
- `T.ShardSpec`
- `T.NDShardSpec`
- `T.CoreGrid` / `T.CoreRangeSet` if no suitable public TileLang wrapper
  already exists
- convenience constructors such as `T.interleaved_dram`,
  `T.sharded_l1`, and `T.sharded_dram`

Add `T.annotate_memory_config({buffer: config})`.
This should emit one typed block attr.
It should be round-trippable in TIR printing / parsing and should reject
invalid Python object shapes before lowering.

The acceptance test for this checkpoint is structural:
a TileLang kernel with annotated input/output/weight buffers produces the
expected typed attr, and invalid strategy/orientation combinations reject.

### S3: Lower User Placement To SpatialPlan

Add `TensorPlacementIntent` to `SpatialPlan`.
`BuildSpatialPlan` consumes the typed DSL attr and emits validated placement
intent.

This checkpoint must prove:

- existing `T.annotate_layout` still produces only layout evidence
- `scope` and `T.Kernel` do not create sharding intent
- unannotated global buffers receive explicit default interleaved-DRAM
  placement intent
- invalid or duplicate user configs fail before TT planning

### S4: Add TTTensorMemoryConfigPlan

Add the TTProgram-level typed object that mirrors TT-Metal
`MemoryConfig + ShardSpec / NdShardSpec`.

The first producer path must cover every placement already present in the
current implementation:

- interleaved DRAM external buffers
- resident L1 staged-copy views
- device-local replicated local buffers

Validators must ensure existing `TTBufferDistributionPlan` records are
consistent with these new plans.
At this point, a buffer distribution that cannot be traced to either
`TTTensorMemoryConfigPlan` or an explicitly local scratch/materialization
requirement is invalid.

### S5: Add TTOpShardingContract

Add placement contracts for the op families that matter to the active queue:

- copy / staged load
- leaf elementwise
- reduction
- matmul / GEMM variants
- output stores / external ABI writes

Each contract states accepted input placements, output policy, conversion
permission, and typed reject reasons.

This is where different compute requirements become explicit.
For example, two consumers of the same producer may require different shard
dimensions; the planner must see that before source emission.

### S6: Add TTPlacementResolutionPlan And Conflict Rejects

Build the value/use placement-resolution pass.

The first resolver milestone must not silently insert a conversion.
It selects a common legal placement when one exists and otherwise emits a
typed conflict diagnostic naming:

- producer value
- consumer op/use
- source placement
- required target placement
- whether the missing edge is a conversion implementation, an op contract,
  or a user `allow_reshard=False` constraint

This immediately prevents silent wrong placement.

### S7: Add TTReshardPlan Conversion Paths

Admit explicit conversion plans in this order:

1. interleaved DRAM to resident L1 sharded view,
   matching the existing staged-copy path
2. sharded L1 view to interleaved DRAM output when needed by external ABI
3. L1 sharded to L1 sharded when source and target shard shapes differ
4. DRAM sharded to L1 sharded for weight / prefetcher-style paths
5. sharded DRAM to sharded DRAM when TT-Metal / TTNN evidence and runtime
   admission are available

Each conversion needs source/spec/direct-runtime or codegen admission.
Unsupported conversions must be typed rejects.

### S8: Project Placement And Conversion To ExecutableSpec

Extend `ExecutableSpec` projection with:

- tensor memory config records
- conversion / reshard records
- materialization records
- placement-derived direct-runtime admission reasons

Leaf readers, `BlackholeModule`, and codegen consume these records.
They may select a concrete leaf implementation or reject unsupported
conversion kinds.
They may not recover missing placement from lower-level source structure.

### S9: DRAM-Sharded And N-D Production Cases

After 2D L1 conversion paths and executable projection are stable, add:

- DRAM-sharded tensor configs
- DRAM-sharded matmul / weight streaming contracts
- `NdShardSpec`-style N-D sharding
- distributed mesh / CCL production variants

This belongs before claiming production distributed sharding support.

## Relation To Current Task Queue

T2 leaf compute / GEMM baseline work can continue for interleaved or
already-admitted placements.

This design is queued as T3.
It starts after the T2 current-placement baseline and before any sharded
GEMM/layout claim.
The external `sharded_accessor_cta` / `page_indexed_accessor_cta` runtime
ABI gap is queued as T4 after this design, because those accessors must
consume projected placement / conversion records rather than infer sharding
or page metadata.

Any task that claims sharded GEMM/layout support must depend on:

- DSL / user placement intent capture
- `TTTensorMemoryConfigPlan`
- `TTOpShardingContract`
- placement conflict validation
- `TTReshardPlan` or typed rejects

T10 distributed production variants depend on this design before they can
claim sharding support.
Mesh / CCL / NoC plans are not enough by themselves if tensor sharding and
reshard conflicts are still implicit.

The ordering relationship is:

```text
T2 interleaved / current admitted placements
  can proceed now

T3 tensor/value sharding and explicit reshard
  starts after T2 baseline

T4 external accessor / runtime ABI expansion
  consumes the projected placement and conversion records

T5 sharded GEMM / layout variants
  require S2-S8 of this design

T9 workload first paths that rely on sharded weights or activations
  require DSL intent, memory-config plans, op contracts, and conflict rejects;
  admitted correctness requires the matching reshard path

T10 production distributed variants
  require the whole sharding / reshard lane plus mesh / CCL / NoC plans
```

## Completion Criteria

This design is implemented only when:

- user or op sharding intent is explicit before TT planning
- TileLang exposes a concrete DSL / annotation surface for memory configs
- TTProgram contains `TTTensorMemoryConfigPlan`
- `TTOpShardingContract` drives placement decisions
- producer/consumer placement conflicts are either converted or typed
  rejected
- reshard/materialization is represented by `TTReshardPlan`
- `TTBufferDistributionPlan` is derived low-level placement, not sharding
  owner truth
- ExecutableSpec projects placement and conversion records
- runtime/codegen consume records or fail closed
- tests cover at least one incompatible producer/consumer sharding case
- tests cover at least one admitted explicit reshard/conversion case

## Runtime Correctness Hardening Gate

The first admitted T3 runtime conversion is still narrowly scoped:

```text
interleaved DRAM runtime tensor
  -> resident L1 sharded staged-copy view
  -> interleaved DRAM runtime tensor
```

The runtime gate for this surface must prove more than record projection.
It must prove that `BlackholeModule` consumes the projected
`TTTensorMemoryConfigPlan`, `TTReshardPlan`, and
`TTBufferDistributionPlan` records when executing or rejecting the path.

Required direct-runtime coverage:

- large-shape staged-copy correctness at `1024x1024`, `2048x4096`,
  `4096x2048`, and `4096x4096`;
- oversubscribed logical grids where `work_per_core > 1`;
- wide, tall, and square aspect ratios;
- resident-L1 staged-copy inputs followed by continuous elementwise compute
  chains over multiple tensor shapes and height / width / block sharding
  strategies;
- mixed elementwise-plus-reduce forms where the reduce result is consumed by a
  later elementwise leaf sequence and runtime correctness is checked on the
  final tensor output;
- an explicit user placement case that names the resident L1 view with a
  non-template buffer name and requires an `interleaved_to_sharded`
  conversion record;
- multiple independent reshard records in one executable;
- non-zero source and destination tile offsets over a large tensor region;
- typed rejects when executable placement or reshard records are removed,
  unsupported, or inconsistent with each other;
- serialization round-trip preservation for tensor memory config and reshard
  records.

These tests do not admit external `sharded_accessor_cta`,
`page_indexed_accessor_cta`, sharded GEMM/layout variants, or production
DRAM-sharded weights. Those remain T4/T5 surfaces.
