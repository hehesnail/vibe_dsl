# TileLang Blackhole Backend Redesign

## Basic Info

- **Document ID**: `final_blackhole_backend_redesign`
- **Date**: 2026-03-19 (created), 2026-04-02 (rewritten)
- **Status**: current sole authoritative architecture document
- **Scope**: `tilelang_repo` Blackhole compiler architecture, compiler-internal IR layering, TT target mapping, runtime materialization boundary
- **Supersedes**:
  - the previously mixed runtime-architecture narrative now archived in `tasks/dev_design/archive/legacy_blackhole_runtime_architecture.md`
  - the old single-layer `Stateful Tiled IR` direction as the next-stage top-level architecture

## 1. Problem Definition

Blackhole is no longer blocked by "how to emit some TT-Metal kernel string." The real problem is structural:

1. `PrimFunc / TIR` does not stably encode `stateful / routed / phased / segmented` algorithm semantics once TileLang tile ops are lowered and scalarized.
2. TT-Metal and TT-style spatial/dataflow hardware require an explicit program model:
   - task / kernel roles
   - communication channels
   - circular buffers
   - semaphores / multicast / synchronization edges
   - dst/register layout
   - core placement / work distribution
   - compile-time / runtime ABI
3. The current pipeline therefore mixes three different responsibilities:
   - recovering algorithm meaning from crushed TIR
   - inventing a spatial program structure
   - choosing TT-specific resources and ABI

That mix is the root cause of the current `blackhole.acc` ambiguity and of the broader flash-attn correctness gap.

**Design thesis**: the compiler must stop trying to solve semantic recovery, spatial organization, and TT target planning inside one layer. The next-stage architecture is a multi-level compiler-internal IR stack:

```text
Stateful Semantic IR
  -> Spatial Program IR
  -> TT Target IR
```

Each layer covers only its own semantics. Lower layers may consume higher-layer truth, but may not reverse-infer it.

## 2. Design Goals And Non-Goals

### Goals

1. Keep TileLang Python DSL largely stable.
2. End late target-specific semantic guessing.
3. Make state, phase, relation, layout, task, sync, and TT resource planning first-class in the correct layer.
4. Make the architecture TT-first for the target layer, while keeping upper layers generic enough to remain valuable beyond TT.
5. Reduce codegen/runtime to materialization and execution, rather than semantic reconstruction.

### Non-Goals

1. Do not create a TT-Metal-specific user DSL.
2. Do not expose `task / channel / CB / semaphore / runtime_args` as first-class Python programming concepts.
3. Do not push all complexity into one "super IR".
4. Do not replace the current direct host path with a second execution path.
5. Do not rewrite already working Stage 0-3 infrastructure unless it must move to a cleaner layer boundary.

## 3. Current Hard Constraints

The redesign starts from the current reality rather than from a greenfield compiler:

1. `BlackholeModule` in-process direct host path remains the only official execution path.
2. `ExecutableSpec` remains the final materialized product consumed by the runtime.
3. Stage 0-3 foundations are already real and are not thrown away:
   - `ExecutableSpec`
   - `rt_mod_blackhole`
   - `BlackholeModule`
   - copy / GEMM / multi-core direct host path
4. The existing recovery-oriented analysis passes remain the starting point for semantic recovery:
   - `AnalyzeBlackholeWorkDecomposition`
   - `AnalyzeBlackholeFragmentRegions`
   - `AnalyzeBlackholePipelineStages`
5. `PlanBlackholeCB`, `AssignBlackholeCores`, and `rt_mod_blackhole` survive during migration, but their long-term ownership moves under the new target-layer architecture.

## 4. Authoritative Architecture

### 4.1 Overall Flow

```text
TileLang DSL / Python
  -> PrimFunc / TIR
  -> Semantic Recovery
  -> Stateful Semantic IR
  -> Semantic Validation
  -> Spatialization
  -> Spatial Program IR
  -> Spatial Validation
  -> Hardware-Aware Mapping
  -> TT Target IR
  -> Target Validation
  -> MaterializeTTExecutableSpec
  -> Codegen / rt_mod_blackhole / BlackholeModule
```

### 4.2 Layer Summary

| Layer | Primary Question | Source Of Truth | Typical Output |
|------|------------------|-----------------|----------------|
| `PrimFunc / TIR` | What did the user write in the standard compiler pipeline? | generic TileLang / TVM IR | normalized TIR |
| `Stateful Semantic IR` | What algorithm is this program actually computing? | algorithm semantics | `SemanticProgram` |
| `Spatial Program IR` | How should that algorithm be organized as a spatial/dataflow program? | task/channel/layout/sync/work graph | `SpatialProgram` |
| `TT Target IR` | How does that spatial program become a legal TT-Metal contract? | TT resource and ABI contract | `TTProgram` |
| `ExecutableSpec / runtime` | How is the frozen TT contract materialized and executed? | materialized target schema | launchable `ExecutableSpec` and host objects |

### 4.3 Design Inputs

The layered architecture is informed by four classes of prior work:

- `T2S`: algorithm semantics and spatial mapping should be separated.
- `Dato`: `task / channel / layout` and `virtual -> physical mapping` deserve first-class representation.
- `TL`: hardware representation and mapping are compiler problems, not codegen afterthoughts.
- `SPADA`: routing and synchronization correctness need explicit validation.

The compiler should borrow the useful abstractions, but not copy any one paper wholesale.

## 5. Layer Design

### 5.1 `Stateful Semantic IR`

#### Why This Layer Exists

This layer exists to answer only one question: **what algorithm is being computed?**

Without it, downstream code must recover semantics from:

- scalarized `BufferLoad / BufferStore`
- fragment helper names
- target builtins
- runtime-side heuristics

That is exactly how mixed semantics like `blackhole.acc` arise.

#### Design Goals

1. Freeze algorithm semantics before any TT resource decision is made.
2. Represent carry, combine, domain constraints, and phase ordering explicitly.
3. Support flash-attn, online-softmax, Welford, routed/paged/select, and future recurrent workloads through the same semantic vocabulary.

#### Core Objects

| Object | Key Fields | Meaning |
|--------|------------|---------|
| `SemanticProgram` | `domains`, `states`, `relations`, `phases`, `regions` | container for semantic truth |
| `Domain` | `kind`, `iter_vars`, `bound_expr`, `predicate`, `index_remapping` | logical iteration domain |
| `State` | `kind`, `lifetime`, `shape`, `value_semantics` | mutable algorithm state |
| `Relation` | `kind`, `source`, `target`, `combine_function`, `mapping_expr` | semantic relationship between state/domain/region |
| `Phase` | `kind`, `live_in`, `live_out`, `regions` | algorithm phase boundary |
| `SemanticRegion` | `op_family`, `reads`, `writes`, `domain`, `phase` | single semantic responsibility region |

#### Object Design

**`Domain`**

- `kind`: `dense | segmented | routed | paged`
- must support:
  - `bound_expr` for causal/data-dependent bounds
  - `predicate` for masked/predicated domains
  - `index_remapping` for grouped/routed/paged access

**`State`**

- `kind`: `matrix_state | vector_state | scalar_state | index_state`
- `lifetime`: `ephemeral | carry | cross_phase`
- does not encode TT backing resources

**`Relation`**

- `kind`: `reduced_from | applies_to | indexes | scatters_to | carried_across`
- `combine_function` is not restricted to a tiny enum; it must support coupled update functions such as Welford or online-softmax rescale/update

**`Phase`**

- models `algorithm phase`
- does not encode reader/compute/writer or pipeline kernel roles

**`SemanticRegion`**

- semantic family examples:
  - `matmul`
  - `reduce`
  - `normalize`
  - `select`
  - `scatter`
  - `recurrence`

#### Inputs

- `PrimFunc / TIR`
- `AnalyzeBlackholeWorkDecomposition`
- `AnalyzeBlackholeFragmentRegions`
- `AnalyzeBlackholePipelineStages`
- minimal Python annotations only when semantic ambiguity cannot be resolved from IR

#### Outputs

- a frozen `SemanticProgram`

#### Validation Responsibilities

1. `state kind / lifetime / shape` consistency
2. `carried_across` vs `live_in / live_out` consistency
3. completeness of `bound_expr / predicate / index_remapping`
4. existence and connectivity of `combine_function`
5. prohibition on one object simultaneously serving as algorithm state and target scratch

#### Explicitly Not In This Layer

- `reader / compute / writer`
- `task / channel / placement`
- `CB / semaphore / dst offset / core group`
- compile-time or runtime ABI
- carry implementation strategy

### 5.2 `Spatial Program IR`

#### Why This Layer Exists

This layer exists to answer a different question: **how should the algorithm be organized as a spatial/dataflow program?**

`task / channel / layout / sync` are not algorithm truth, but they are also much higher level than TT-specific resources. If this layer is skipped, one of two bad outcomes happens:

1. semantic IR grows execution topology and stops being semantic, or
2. TT target lowering becomes a black hole that both invents tasks and plans TT resources.

#### Design Goals

1. Represent task graph, channel graph, layout, work partition, and sync explicitly.
2. Make routing/synchronization/layout compilation visible before target-specific planning.
3. Keep this layer target-neutral enough that TT-specific details do not leak upward.

#### Core Objects

| Object | Key Fields | Meaning |
|--------|------------|---------|
| `SpatialProgram` | `tasks`, `channels`, `layouts`, `work_partitions`, `placements`, `sync_edges`, `resource_intents` | spatial program container |
| `Task` | `kind`, `semantic_regions`, `input_ports`, `output_ports`, `execution_scope` | logical execution unit |
| `Channel` | `producer`, `consumer`, `payload_kind`, `transport_semantics`, `ordering` | task-to-task dataflow edge |
| `Layout` | `layout_kind`, `partition_axes`, `mapping_expr` | distributed data organization |
| `WorkPartition` | `partition_kind`, `partition_expr`, `load_balance_policy` | logical work decomposition |
| `Placement` | `placement_kind`, `constraints` | virtual placement relation |
| `SyncEdge` | `kind`, `scope`, `source_task`, `target_task` | synchronization requirement |
| `ResourceIntent` | `kind`, `capacity_hint`, `visibility`, `reuse_policy` | resource need without target binding |

#### Object Design

**`Task`**

- `kind` examples:
  - `load`
  - `compute`
  - `reduce`
  - `exchange`
  - `store`
- does not equal one TT kernel by definition

**`Channel`**

- `payload_kind` examples:
  - `tile`
  - `vector`
  - `scalar`
  - `index`
- `transport_semantics` examples:
  - `fifo`
  - `multicast`
  - `gather`
  - `scatter`

**`Layout`**

- examples:
  - `tile`
  - `shard`
  - `grouped`
  - `routed`
  - `paged`

**`WorkPartition`**

- examples:
  - `row`
  - `pair`
  - `split_k`
  - `expert`
  - `page`

**`Placement`**

- virtual only
- examples:
  - `colocated`
  - `adjacent`
  - `row_group`
  - `column_group`

**`SyncEdge`**

- examples:
  - `producer_consumer`
  - `barrier`
  - `completion`
  - `multicast_ready`

#### Inputs

- frozen `SemanticProgram`
- target-neutral spatialization policy:
  - task fusion / splitting policy
  - layout choice policy
  - work partition policy
  - sync construction policy

#### Outputs

- a frozen `SpatialProgram`

#### Validation Responsibilities

1. task/channel graph closure
2. correctness of producer-consumer and barrier structure
3. work-partition and layout consistency
4. correct carried-state movement across task boundaries
5. consistency of virtual placement constraints
6. obvious race / deadlock / routing inconsistency detection

#### Explicitly Not In This Layer

- `CBIndex`
- `semaphore_id`
- `dst offset`
- `CreateCircularBuffer / SetRuntimeArgs`
- TT physical core identifiers

### 5.3 `TT Target IR`

#### Why This Layer Exists

This layer exists to answer the TT-specific question: **how does the spatial program become a legal and stable TT-Metal contract?**

TT complexity is not "just codegen." It is an explicit program model involving:

- kernel roles
- circular buffers
- synchronization protocols
- dst/register planning
- core placement
- compile-time/runtime ABI

Those must become first-class compiler objects before runtime.

#### Design Goals

1. Capture TT-Metal program structure as target contract, not as scattered pass side effects.
2. Make `CB / semaphore / dst layout / kernel role / ABI / execution plan` explicit.
3. Reduce `rt_mod_blackhole` and runtime to materialization only.

#### Core Objects

| Object | Key Fields | Meaning |
|--------|------------|---------|
| `TTProgram` | `kernels`, `core_groups`, `cb_plan`, `semaphore_plan`, `dst_layout_plan`, `abi_plan`, `execution_plan` | target contract container |
| `TTKernel` | `role`, `task_subset`, `core_group`, `compile_time_args`, `runtime_args` | TT kernel contract |
| `TTCoreGroup` | `physical_cores`, `core_type`, `topology_role` | mapped physical execution group |
| `TTCBPlan` | `resource_class`, `capacity`, `producer`, `consumer`, `binding_scope` | final CB plan |
| `TTSemaphorePlan` | `kind`, `source_group`, `target_group`, `protocol` | final synchronization plan |
| `TTDstLayoutPlan` | `state_bindings`, `offset`, `tile_span`, `layout_role` | dst/register residency plan |
| `TTABIPlan` | `compile_time_arg_specs`, `runtime_arg_specs`, `accessor_specs`, `launch_specs` | ABI contract |
| `TTExecutionPlan` | `work_distribution`, `remote_core_descriptors`, `kernel_order` | execution and launch plan |

#### Object Design

**`TTKernel.role`**

- `reader`
- `compute`
- `writer`
- optional future roles such as `relay` or `reduction`

**`TTCBPlan.resource_class`**

- `transport`
- `tile_scratch`
- `scalar_scratch`
- `persistent_carry`
- `output`

**`TTSemaphorePlan.kind`**

- `local`
- `remote`
- `multicast`
- `barrier`

**`TTDstLayoutPlan`**

- binds semantic/target states to concrete dst offsets
- the plan is static and compile-time determined

#### Inputs

- frozen `SpatialProgram`
- TT hardware model:
  - topology
  - memory hierarchy
  - NoC / multicast / semaphore capabilities
  - dst/register capacity
  - core kinds and placement constraints

#### Outputs

- a frozen `TTProgram`
- materializable `ExecutableSpec`

#### Validation Responsibilities

1. L1 / CB / dst capacity legality
2. semaphore / multicast / routing legality
3. core placement legality
4. compile-time/runtime ABI completeness
5. sufficiency of information to materialize `ExecutableSpec` without guesswork

#### Explicitly Not In This Layer

- modification of semantic state / relation / phase
- modification of spatial task / channel / layout / sync structure
- runtime-side reconstruction of missing protocol

## 6. Layer Interfaces And Invariants

### 6.1 Source Of Truth Rules

1. Algorithm semantics live only in `Stateful Semantic IR`.
2. Spatial organization lives only in `Spatial Program IR`.
3. TT resources and ABI live only in `TT Target IR`.
4. `ExecutableSpec` is materialized from `TT Target IR`; it is not a second source of truth.

### 6.2 Handoff Contracts

| From | To | Required Contract | Allowed Decisions | Forbidden Behavior |
|------|----|-------------------|------------------|--------------------|
| `Semantic Recovery` | `Stateful Semantic IR` | recovered domain/state/relation/phase facts | objectization and freezing | leaking TT resource facts |
| `Stateful Semantic IR` | `Spatial Program IR` | frozen algorithm semantics | task/channel/layout/sync/work construction | changing semantic meaning |
| `Spatial Program IR` | `TT Target IR` | frozen spatial structure | TT mapping, resource planning, ABI definition | inventing a new task graph or semantic combine |
| `TT Target IR` | `ExecutableSpec / runtime` | frozen TT contract | API materialization and launch emission | semantic recovery or protocol patching |

### 6.3 Validation Layers

- `ValidateStatefulSemanticIR`
- `ValidateSpatialProgram`
- `ValidateTTTargetProgram`

Failures should surface in the earliest layer that owns the violated invariant.

### 6.4 No Reverse Inference

The following are explicitly forbidden:

1. using `CB / dst layout / runtime args` to infer state semantics
2. using TT kernel names to invent task graph structure
3. letting runtime patch missing synchronization or carry strategy
4. exposing `task / channel / semaphore` as Python DSL surface merely because the backend needs them

## 7. `flash-attn` Forward Example

### 7.1 Facts Recovered Before Semantic IR

From TIR and the current analysis passes, the compiler should recover at least:

- a tiled `Q x K^T` step
- row-wise `max / sum / rescale / accumulate`
- loop-carried states such as `acc_o`, `scores_max`, `scores_sum`
- optional `bound_expr` for causal attention
- optional `index_remapping` for GQA
- optional `predicate` or routed/page index facts for sparse or paged variants

### 7.2 `Stateful Semantic IR` Shape

- `Domain`
  - `q_tile_domain`
  - `kv_chunk_domain`
  - optional `causal_bound_expr`
  - optional `gqa_index_remapping`
- `State`
  - `acc_o : matrix_state(carry)`
  - `scores_max : vector_state(carry)`
  - `scores_sum : vector_state(carry)`
  - `logsum : scalar_state(cross_phase)` or epilogue-derived result
- `Relation`
  - `scores_max reduced_from qk_scores`
  - `scores_sum reduced_from exp_scores`
  - `acc_o / scores_max / scores_sum carried_across kv_chunk_phase`
- `Phase`
  - `kv_chunk_phase`
  - `epilogue_phase`
- `SemanticRegion`
  - `qk_matmul`
  - `row_max_reduce`
  - `row_sum_reduce`
  - `rescale_update`
  - `ov_matmul`
  - `epilogue_writeback`

### 7.3 `Spatial Program IR` Shape

- `Task`
  - `load_q_task`
  - `stream_k_task`
  - `stream_v_task`
  - `attention_step_task`
  - `store_out_task`
- `Channel`
  - `q_tiles`
  - `k_tiles`
  - `v_tiles`
  - `carry_state`
  - `out_tiles`
- `Layout`
  - `q_row_layout`
  - `kv_chunk_layout`
  - optional `grouped_layout` or `paged_layout`
- `WorkPartition`
  - `row_partition`
  - optional `causal_pair_partition`
- `SyncEdge`
  - load/stream tasks into `attention_step_task`
  - `attention_step_task -> store_out_task`
- `ResourceIntent`
  - transport buffers for `q/k/v`
  - persistent carry for `acc_o/max/sum`
  - output

### 7.4 `TT Target IR` Shape

- `TTKernel`
  - `reader_qkv`
  - `compute_attention`
  - `writer_out`
- `TTCBPlan`
  - transport CBs for `q/k/v`
  - temporary CBs for intermediate scores
  - optional carry CBs when carry is not register-resident
  - output CB
- `TTDstLayoutPlan`
  - offsets for `acc_o`
  - offsets for `scores_max`
  - offsets for `scores_sum`
  - offsets for temporary score tiles
- `TTSemaphorePlan`
  - reader-to-compute readiness
  - compute-to-writer completion
  - optional multicast/barrier edges for multi-core variants
- `TTABIPlan`
  - compile-time tile shape / transpose / pack config
  - runtime args for tile counts, row ids, bounds, and addresses
- `TTExecutionPlan`
  - virtual row partition to physical core groups

### 7.5 What This Example Proves

1. semantic recurrence must freeze before any CB or dst decision exists
2. task/channel/sync must freeze before TT kernel roles are chosen
3. TT runtime must never need to rediscover attention semantics

## 8. Codebase Mapping And Migration

### 8.1 Current Components Mapped To New Architecture

| Current Pass / Module | New Ownership | Long-Term Status |
|-----------------------|---------------|------------------|
| `AnalyzeBlackholeWorkDecomposition` | semantic recovery input producer | keep and generalize |
| `AnalyzeBlackholeFragmentRegions` | semantic recovery input producer | keep and generalize |
| `AnalyzeBlackholePipelineStages` | semantic recovery input producer | keep and tighten |
| `LowerBlackholeOps` | mixed legacy layer; future split between spatial lowering and TT target lowering | shrink and eventually disappear as a monolith |
| `PlanBlackholeCB` | TT target planner submodule | keep but demote |
| `AssignBlackholeCores` | TT target planner submodule | keep but narrow |
| `rt_mod_blackhole` | codegen/runtime materialization | keep and tighten |
| `ExecutableSpec` | TT target materialization result | keep |

### 8.2 Target Pass Chain

```text
SplitBlackholeKernel
  -> AnalyzeBlackholeWorkDecomposition
  -> AnalyzeBlackholeFragmentRegions
  -> AnalyzeBlackholePipelineStages
  -> LiftToStatefulSemanticIR
  -> ValidateStatefulSemanticIR
  -> LowerToSpatialProgram
  -> ValidateSpatialProgram
  -> LowerSpatialProgramToTTTarget
  -> ValidateTTTargetProgram
  -> MaterializeTTExecutableSpec
  -> rt_mod_blackhole
```

### 8.3 Migration Phases

**Phase A: Semantic IR**

1. add `Domain / State / Relation / Phase / SemanticRegion`
2. land `LiftToStatefulSemanticIR`
3. land `ValidateStatefulSemanticIR`
4. require existing target lowering to consume semantic truth instead of recovering the main semantics itself

**Phase B: Spatial Program IR**

1. add `Task / Channel / Layout / WorkPartition / Placement / SyncEdge / ResourceIntent`
2. land `LowerToSpatialProgram`
3. land `ValidateSpatialProgram`
4. pull task/channel/layout/sync/work-partition logic out of `LowerBlackholeOps`

**Phase C: TT Target IR**

1. add `TTKernel / TTCoreGroup / TTCBPlan / TTSemaphorePlan / TTDstLayoutPlan / TTABIPlan / TTExecutionPlan`
2. land `LowerSpatialProgramToTTTarget`
3. land `ValidateTTTargetProgram`
4. materialize `ExecutableSpec` from `TT Target IR`
5. reduce `PlanBlackholeCB`, `AssignBlackholeCores`, and `rt_mod_blackhole` to their proper target/runtime roles

## 9. TT-Specific Target Constraints

TT-specific facts belong to the target layer and must be designed around explicitly.

### 9.1 TT Program Model Is Program-Level Structure

TT-Metal programs are naturally built around:

- `reader / compute / writer`
- host-side `CreateCircularBuffer / CreateKernel / SetRuntimeArgs`
- per-core runtime arguments

Those facts belong in `TT Target IR`, not in semantic or spatial layers.

### 9.2 Dst/Register Layout Is A First-Class Target Decision

Flash-attn and similar kernels keep long-lived state in dst/register space across chunk loops. Therefore:

- `TTDstLayoutPlan` is mandatory
- carry strategy may be `register-resident` or `CB-round-trip`
- that choice is not semantic; it is target mapping

### 9.3 CB And Semaphore Are Target Resources

CB and semaphore planning must not be treated as generic buffer lowering side effects.

They are explicit manifestations of:

- transport channels
- scratch resources
- persistent carry realization
- synchronization protocol

### 9.4 Ground-Truth References

Current TT target design should continue to cross-check these references:

- `tt_metal_repo/tt-train/sources/ttml/metal/ops/sdpa_fw/device/kernels/compute/sdpa_fw_compute_kernel.cpp`
- `tt_metal_repo/models/demos/deepseek_v3_b1/kernel_includes/tt_metal/include/compute_kernel_api/sdpa.h`
- `tt_metal_repo/tt-train/sources/ttml/metal/ops/sdpa_fw/device/sdpa_fw_program_factory.cpp`

## 10. Rollout And Acceptance

### 10.1 Rollout Order

1. finish the architecture rewrite and retire mixed old/new structure
2. execute Phase A on top of existing direct runtime path
3. execute Phase B to remove task/channel/layout/sync from the monolithic lowering path
4. execute Phase C to make TT resource and ABI planning first-class

### 10.2 Acceptance Criteria

The redesign is only considered real when all of the following are true:

1. semantic truth is frozen before TT target planning
2. spatial structure is frozen before TT resource and ABI planning
3. runtime/codegen no longer infer missing semantics
4. copy/GEMM compile-path regression gates remain green
5. flash-attn forward migrates off mixed `blackhole.acc` semantics toward the new layered contract

## 11. Legacy Documents

The following documents are historical or implementation-history references only:

- `tasks/dev_design/archive/legacy_blackhole_runtime_architecture.md`
- `tasks/dev_design/archive/2026-04-02-stateful-tiled-ir-phase1-implementation-plan.md`

They are not authoritative for new implementation work. New implementation must follow this document.

## 12. References

These papers informed the layering, validation, and target-mapping direction in this document. They are design inputs, not protocol sources of truth for the TileLang Blackhole backend.

- `Dato: A Task-Based Programming Model for Dataflow Accelerators` (2025)
  https://arxiv.org/abs/2509.06794
  - referenced for `task / channel / layout` first-class representation and `virtual -> physical` mapping split

- `TL: Automatic End-to-End Compiler of Tile-Based Languages for Spatial Dataflow Architectures` (2025)
  https://arxiv.org/abs/2512.22168
  - referenced for explicit hardware representation and compiler-owned spatial mapping

- `SPADA: A Spatial Dataflow Architecture Programming Language` (2025)
  https://arxiv.org/abs/2511.09447
  - referenced for rigorous dataflow semantics and explicit routing / synchronization validation

- `Revet: A Language and Compiler for Dataflow Threads` (2023/2024)
  https://arxiv.org/abs/2302.06124
  - referenced for separating high-level threaded/dataflow semantics from backend realization

- `Programmatic Control of a Compiler for Generating High-performance Spatial Hardware` (`T2S`, 2017)
  https://arxiv.org/abs/1711.07606
  - referenced for separating algorithm semantics from spatial mapping

- `Spatial: A Language and Compiler for Application Accelerators` (PLDI 2018)
  https://pldi18.sigplan.org/event/pldi-2018-papers-spatial-a-language-and-compiler-for-application-accelerators
  - referenced as an earlier precedent for accelerator-oriented language/compiler layering
