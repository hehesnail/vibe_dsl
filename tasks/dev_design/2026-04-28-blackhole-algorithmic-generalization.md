# Blackhole Algorithmic Generalization Refactor

## Goal

把当前 Blackhole passes 从
“手写 visitor + 局部 matcher + pass-local map”
推进到可承载更复杂计算模式的算法化结构。

本设计只定义三件事：

1. `AccessRegion` / affine-lite access analysis
2. graph-backed `SpatialPlan` dependence model
3. `LiveValueSSA` / BufferSSA-style version and event model

它们不是新的长期 IR 层。
长期主链仍然只有：

```text
Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec
```

这三件事分别落在现有层里：

- `AccessRegion`
  是从 `Normalized Tile TIR`
  结构推导出来的规范化访问描述；
  如果下游需要跨阶段消费，
  就作为 `SpatialPlan` typed object 持久化。
- dependence graph
  是 `SpatialPlan`
  内部显式对象之间的图语义，
  不替代 `SpatialPlan`。
- `LiveValueSSA`
  是 `SpatialPlan.LiveValue`
  / `LiveValueEdge`
  / `MaterializationBoundary`
  的版本化强化，
  并由 `TTProgram.TTLiveFormPlan`
  / `TTMaterializationPlan`
  做 target realization。

## References

本设计借鉴这些经典实践，但不照搬框架：

- LLVM Dependence Graph:
  data/control dependence graph,
  SCC / pi-block
  用来表达 loop-carried recurrence。
- LLVM MemorySSA:
  `MemoryDef` / `MemoryUse` / `MemoryPhi`
  用版本化内存状态回答 reaching-def /
  clobber 查询。
- MLIR affine dialect:
  用受限 affine map /
  integer set
  让 access/dependence analysis
  可验证。
- MLIR sparse dataflow:
  lattice + transfer function + worklist
  作为 live-form propagation
  的算法骨架。

补充调研结论：

- LLVM GlobalISel
  把 generic MIR 放进固定主链：
  `IRTranslator -> Legalizer -> RegBankSelect -> InstructionSelect`。
  Legalizer 的结果不是旁观 dump，
  而是把当前 MIR 改写到 target selector
  后续一定能处理的合法形态；
  legality query 失败时必须走
  lower / custom / unsupported
  这类显式 action。
- LLVM MemorySSA
  是 virtual IR，
  但它被真正使用的原因是
  `MemoryDef / MemoryUse / MemoryPhi`
  能回答 reaching-def /
  clobber query。
  如果版本化结构不能回答 query，
  它只是日志结构。
- LLVM dependence graph
  的 SCC / pi-block
  不是为了记录“有环”，
  而是为了让 reordering /
  scheduling /
  vectorization
  知道哪些 nodes 不能被独立移动。
- MLIR dialect conversion /
  pattern rewriter
  把 legality target、
  rewrite pattern、
  type conversion、
  match failure diagnostic
  放进同一个 driver contract。
  pattern/debug 信息能解释为什么没选中，
  而不是让下游继续猜。
- Cranelift ISLE
  把 backend lowering rules
  写成纯 SSA rewrite；
  matcher/extractor
  不能有 side effect，
  真正的 mutation 只能在 rule
  已经 commit 后发生。
  这对应我们这里：
  query 和 legality 可以是 pass-local，
  但跨阶段选择结果必须进入 typed plan。
- LLVM VPlan
  的关键点是显式候选计划同时服务
  cost / legality 后的优化
  和最终 IR translation。
  它提醒我们：
  `SpatialPlan` /
  `TTProgram`
  不能只保存唯一已经手选好的路径；
  当后续存在多种 live-form /
  materialization /
  compute leaf 覆盖选择时，
  需要显式 plan object
  承载候选与选择结果，
  让 validator 和 projector
  能审计为什么选中或为什么拒绝。
- LLVM LoopAccessAnalysis
  把 memory-dependence legality
  与可生成的 runtime check /
  failure diagnostic
  连接起来。
  对应本设计，
  `AccessRegion`
  不只是记录 index 表达式，
  它必须回答 overlap /
  coverage /
  stride /
  axis legality query；
  query 不可证明时
  应 fail closed
  或进入显式 admission lane，
  不能让下游按名字补猜。
- MLIR affine dialect
  通过受限 affine map /
  integer set
  换取可分析性和闭包性。
  对应本设计，
  affine-lite 的限制是算法合同，
  不是实现偷懒：
  超出限制的访问要么扩表示，
  要么显式 unsupported。
- MLIR Linalg
  把 structured op
  的 indexing /
  iterator /
  reduction 语义放在 IR 接口里，
  使 tiling、
  fusion、
  library / special-instruction
  lowering 成为通用 transform。
  对应 Blackhole，
  tile compute preservation
  是 Algorithmic Generalization
  的前置条件；
  如果 tile leaf 语义先被 scalar lowering
  打碎，
  后续 DAG covering 只能退化成 matcher。

因此本设计的实现标准要收紧：

> Algorithmic structures are complete only when they become
> legality / query / action inputs on the active chain.
> Merely defining objects,
> dumping them,
> or validating that they are well-formed
> is only foundation work,
> not algorithmic generalization completion.

## Non-Goals

- 不引入 MLIR、LLVM IR、TableGen、egg
  作为依赖。
- 不新增第五个 IR 层。
- 不把 `softmax`、
  `exp2_affine`、
  `row_broadcast_exp2_affine`
  这类 composite helper
  重新写进生产协议。
- 不把 runtime/backend admission
  倒灌成上层 planner truth。
- 不以名字匹配恢复 buffer role、
  reduction role、
  event lifetime
  或 source-live-form。

## Current Problem

当前 repo HEAD 已经完成
tile-compute preservation
和 legacy protocol deletion，
但算法骨架仍然偏局部：

- `BuildSpatialPlan`
  已能构造 `ExecutionUnit`、
  `DataflowEdge`、
  `LiveValue`、
  `LiveValueEdge`
  和 `MaterializationBoundary`，
  但 live value
  还不是严格版本化 def-use model。
- access reasoning
  分散在局部 visitor、
  buffer metadata、
  `ProvenEqual`
  和 target helper 里。
- `lower_blackhole_*`
  的 live-form / exact-CB
  逻辑仍然大量依赖 pass-local
  mutable maps。
- multi-block flash-attn
  暴露出核心缺口：
  upstream live value、
  event lifetime、
  carry/reduction recurrence、
  exact-CB publish/consume
  需要统一模型，
  不能继续 workload-specific patch。

## End State

### 1. Affine-Lite `AccessRegion`

每个可参与空间规划 /
live-form planning
的读写访问都规范化成
`AccessRegion`。

建议新增 typed object：

```text
AccessRegion
  name
  subject
  unit_name
  unit_index
  access_kind            // read | write | read_write | reduce_read | reduce_write
  value_kind             // tensor | tile | fragment | accumulator | scalar
  logical_rank
  loop_vars
  index_exprs
  lower_bounds
  extents
  strides
  coverage_kind          // full | slice | row_slice | grouped_slice | scalar
  predicate_kind         // unconditional | guarded | unknown
  anchors
```

`AccessRegion`
的第一版只接受 affine-lite 子集：

- loop var linear combination
- constant stride
- static or symbolic positive extents
- normalized linearized index
- simple row/group slice

不接受时必须 fail-closed 或标记
`unsupported_access_region`，
不能退回名字匹配。

核心查询 API：

```text
SameSubject(lhs, rhs)
MayOverlap(lhs, rhs)
MustCover(lhs, rhs)
IsFullLogicalValue(region)
IsSliceCompatible(producer, consumer)
LinearizedIndex(region)
RegionElementCount(region)
```

这些查询应集中到
`spatial_access_region.{h,cc}`
或等价 helper，
不要分散在 lowering passes。

### 2. Graph-Backed `SpatialPlan`

`SpatialPlan`
继续是 target-independent
virtual spatial/dataflow program，
但其构造过程改为显式图算法。

输入：

- `ExecutionUnit`
- `AccessRegion`
- existing local value flow evidence
- phase/order evidence
- tile op `GetDataflowAccessInfo()`

输出：

- `DataflowEdge`
- `LiveValue`
- `LiveValueEdge`
- `MaterializationBoundary`
- optional `DependenceComponent`
  for SCC / recurrence diagnostics

建议新增 typed object：

```text
DependenceComponent
  name
  component_kind         // acyclic | recurrence | carry_cycle | reduction_cycle
  unit_indices
  edge_indices
  subjects
  anchors
```

算法：

1. Collect all `AccessRegion`s per unit.
2. Build per-subject read/write event order.
3. Add flow edges from reaching writes to reads.
4. Add output edges between writes when order matters.
5. Add anti edges only when they affect materialization /
   scheduling legality.
6. Add carry / reduction / broadcast / join edges
   from explicit tile op and loop structure evidence.
7. Run SCC detection over unit/edge graph.
8. Emit recurrence components for non-trivial SCCs.

`DataflowEdge.kind`
should cover:

```text
flow
carry
reduction
broadcast
join
materialize
output_order
anti_order
```

Validator rules:

- every edge references valid producer/consumer units
- every edge has a subject and region evidence
- `crosses_phase`
  matches `PhasePlan`
  membership
- recurrence components must be backed by SCC evidence
- `broadcast` /
  `reduction`
  edges must be backed by tile op dataflow info
  or explicit loop/dataflow structure,
  never by buffer names

### 3. `LiveValueSSA`

`LiveValue`
becomes the explicit logical-value definition
version, not just a producer/subject label.

The model mirrors MemorySSA at the level we need:

```text
LiveValueDef
  external input
  compute write
  materialization write
  phi / join
  host output

LiveValueUse
  compute consume
  materialization consume
  transport consume
  host output consume

LiveValuePhi
  carry merge
  reduction merge
  branch/join merge
```

Implementation can reuse existing object names,
but their semantics must become explicit:

- `LiveValue`
  is a versioned definition.
- `LiveValueEdge`
  is a def-use relation from one `LiveValue`
  to a consumer edge.
- `MaterializationBoundary`
  is a form/visibility transition
  between source and target live values.

Recommended field extensions:

```text
LiveValue
  version_index
  definition_kind
  defining_access_region_index
  defining_event_index

LiveValueEdge
  use_kind
  consumer_access_region_index
  source_version_index
  target_version_index

MaterializationBoundary
  source_access_region_index
  target_access_region_index
  event_lifetime_kind       // single_event | multi_event | loop_carried
  min_publish_pages
  max_consume_pages
```

The first implementation may store some fields as
typed arrays alongside existing objects
instead of changing every constructor at once,
but the end state must not be another map payload.

Core invariants:

- every compute write creates a new `LiveValue`
  version
- every compute read consumes exactly one reaching
  `LiveValue`
  or an explicit phi result
- read-write ops consume the old version before
  defining the new version
- every cross-phase consumer has either direct
  visibility proof or a `MaterializationBoundary`
- every exact-CB publish/consume event is owned by
  a live value version and has bounded lifetime
- stale synthetic fill fallback is illegal if a
  reaching live value exists

## TTProgram Projection

`TTProgram`
must not recompute semantic dataflow.
It should only realize validated `SpatialPlan`
objects into TT-specific resources.

Mapping:

```text
SpatialPlan.LiveValue
  -> TTLiveFormPlan

SpatialPlan.MaterializationBoundary
  -> TTMaterializationPlan

SpatialPlan.LiveValueEdge
  -> TTConsumerBindingPlan

SpatialPlan.AccessRegion
  -> TTBufferDistributionPlan / per-work descriptor evidence
```

The target planner may choose physical forms:

```text
fragment
accumulator
exact_tiled_cb
tiled_cb
dram_buffer
host_visible
```

but it must not invent producer/consumer semantics.
If the validated `SpatialPlan`
does not prove a reaching live value,
the planner must reject the shape or require an
earlier normalization extension.

### Active-Use Contract

每个算法结构必须有明确 consumer。
当前实现和后续 cutover
按下面的 owner truth 收束：

| Structure | Required active use | Current repo HEAD status | Required next use |
| --- | --- | --- | --- |
| `AccessRegion` | drive access overlap, coverage, axis legality, materialization shape, and compute DAG operands | already builds boundary seeds and validator coverage, but most regions are full/local and downstream still relies on helper facts | make dataflow/live/materialization legality query `AccessRegion` first; reject slice/full mismatch instead of trusting traits |
| `DependenceComponent` | drive recurrence scheduling, loop-carried lifetime, phi/live-form join, and no-reorder barriers | constructed and validated, but not yet a decisive input to target planning | make `event_lifetime_kind=loop_carried` and recurrence live-form joins require component evidence |
| `LiveValueSSA` | answer reaching-def, source-live-form, and consumer-binding queries | partially active: TT plans reference live values/edges/boundaries, but `PlanTTKernelABI` still indexes by subject string in places | replace subject-pair lookup with `LiveValueEdge` / `MaterializationBoundary` index queries |
| live-form solver | choose physical form, topology, materialization protocol, and consumer requirements from typed boundary evidence | active only for fragment/cast transfer and still mostly constant-return | expand to a worklist/lattice solver over all `SpatialPlan` live edges consumed by TT planning |
| tile compute legalizer / DAG covering | legalize and select compute leaf patterns using access/live/dependence evidence | design-complete, not implemented | start only after the above structures drive query/action decisions |

This table is normative.
If a field is not consumed by a validator,
legalizer,
solver,
or typed plan projection,
it is still foundation surface,
not completed algorithmic generalization.

### Anti-Overdesign Pay-Rent Rule

Algorithmic generalization is allowed only when the structure
solves an active compiler problem on the main chain.
It is not allowed as framework imitation.

A new algorithmic object,
field,
solver state,
or pattern abstraction
must pay rent in at least one of these ways:

- it changes a legality decision,
- it answers a query that previously required a side channel,
- it chooses a target action that is projected into typed plans,
- it produces a typed unsupported diagnostic earlier than source /
  runtime emission,
- or it deletes an old matcher,
  helper fact,
  subject-pair map,
  payload,
  or stale fallback.

The following are explicitly not sufficient:

- constructing a typed object and dumping it,
- adding a validator that only checks shape while downstream still ignores it,
- adding a general solver whose output is constant for all admitted cases,
- moving branch logic behind a more abstract class without deleting
  owner-truth side channels,
- creating a DAG / legalizer that re-discovers semantics from scalar loops
  or source-like idioms,
- adding a candidate-plan structure before any downstream consumer can be
  affected by the selected candidate.

This rule is the overdesign guardrail.
If a proposed structure cannot name its active consumer and the old
mechanic it removes or dominates,
it should stay out of the main design.
If a structure is needed only for future work,
it may be documented as a future candidate,
but it must not be reported as completed implementation.

Implementation discipline:

1. Prefer the smallest algorithm that answers the current query.
   Do not introduce full Presburger solving,
   equality saturation,
   ILP covering,
   or a generic multi-target selector
   until a concrete admitted workload requires it.
2. Every algorithmic cutover needs a counterfactual test:
   changing the evidence must change legality,
   solver output,
   typed plans,
   or unsupported diagnostics.
3. A local cache is acceptable only when it is recomputable from current IR.
   It cannot become owner truth for cross-stage semantics.
4. A phase is not complete while its new typed evidence coexists with
   an older side channel that can still override or silently bypass it.

### Coverage Model

这轮设计覆盖面不能被收窄成某一个 pass
或某一个 workload。
算法化必须同时覆盖四类 decision surfaces：

1. `SpatialPlan` legality surface
   - `AccessRegion`
     回答访问、覆盖、slice/full、
     axis 和 overlap 问题。
   - `DependenceComponent`
     回答 recurrence、
     loop-carried lifetime、
     no-reorder barrier
     和 phi/join 问题。
2. `LiveValueSSA` query surface
   - versioned live values
     回答 reaching-def、
     stale-source rejection、
     source-live-form ownership
     和 consumer binding 问题。
3. `TTProgram` action surface
   - target planner
     把 validated logical evidence
     变成 physical live form、
     materialization protocol、
     CB / ABI / sync binding
     和 typed compute plans。
   - action 结果必须进入
     `TTLiveFormPlan`,
     `TTMaterializationPlan`,
     `TTConsumerBindingPlan`,
     `TTComputeOpPlan`
     等 typed records。
4. `ExecutableSpec` admission surface
   - leaf reader /
     runtime /
     codegen
     只能消费 typed projection。
   - backend support subset
     通过 admission metadata
     fail closed，
     不能反向补 planner semantics。

因此 Phase E 不能理解成
“先把几个字段接到当前 fragment/cast 路径”。
那只是一条可验证切片。
完整 Phase E 的含义是：
每个 Foundation structure
至少被一个 active legality /
query /
action consumer 使用，
且旧 subject-pair map /
helper fact /
stale fallback
不能再作为 owner truth。

## Worklist Dataflow For Live Forms

The Blackhole target planner should use a small
lattice solver for physical live forms.

Suggested lattice:

```text
Bottom
Fragment
Accumulator
ExactCB(single_event)
ExactCB(multi_event)
TiledCB
DRAM
HostVisible
Conflict
Unsupported
```

Transfer functions:

- tile compute write:
  defines `Fragment` or `Accumulator`
- `copy_tile` / `pack_tile` publication:
  defines `ExactCB`
- materialization to output:
  defines `DRAM` or `HostVisible`
- cross-phase edge:
  requires form with visibility across the phase
- carry/reduction phi:
  joins incoming versions and checks compatibility
- consumer binding:
  requires a form accepted by the consumer op

Join rules:

- same form joins to itself
- compatible single-event exact CB joins only if
  lifetimes do not overlap incorrectly
- incompatible forms require a materialization boundary
  or become `Conflict`
- `Unsupported`
  fails closed with a typed unsupported reason

## Architecture Review

This section records the design review against
`final_blackhole_backend_redesign.md`
and compiler architecture practice.

Review scope:

- only the algorithmic foundation lane:
  `AccessRegion`,
  graph-backed `SpatialPlan` dependence,
  `LiveValueSSA`,
  and live-form propagation
- not runtime admission policy beyond the typed evidence required
  to admit or reject a runtime shape

Primary checks:

1. Representation ownership
   stays on the existing chain:

   ```text
   Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec
   ```

   `AccessRegion`
   is either builder-local analysis
   or a typed `SpatialPlan` object when a downstream phase needs it.
   dependence edges,
   live-value versions,
   and materialization boundaries
   are `SpatialPlan` semantics.
   physical forms are `TTProgram` realization.

2. Analysis does not become protocol.
   Access queries,
   SCC detection,
   reaching-def construction,
   and worklist propagation
   may use local caches,
   but any result consumed by a downstream phase
   must be reflected as typed IR fields.
   Parallel maps are allowed only as pass-local implementation state;
   they cannot be serialized,
   exposed through helper APIs,
   or read by downstream passes as owner truth.

3. Lowering does not recover semantics from names.
   Region overlap,
   broadcast,
   reduction,
   carry,
   and event lifetime
   are derived from TIR structure,
   tile op dataflow info,
   access regions,
   and explicit version edges.
   Buffer names may appear in diagnostics,
   but not in legality decisions.

4. Legality fails closed.
   Non-affine access,
   unproven reaching definitions,
   incompatible live-form joins,
   missing event lifetime,
   and unvalidated cross-phase visibility
   all produce typed unsupported diagnostics.
   They must not fall back to synthetic fill,
   runtime-side source patching,
   or leaf reader inference.

5. Stage responsibilities remain separated.
   `SpatialPlan`
   owns target-independent dependence and logical value versions.
   `TTProgram`
   chooses TT physical forms and materialization protocols.
   `ExecutableSpec`
   only projects validated leaf records.
   The live-form solver is a target-planning algorithm,
   not a new IR layer and not a runtime admission shortcut.

6. Compiler practice alignment:
   the design borrows
   affine-style access normalization,
   dependence graph SCCs,
   MemorySSA-like versioning,
   and sparse dataflow lattice solving
   because those algorithms match the queries this backend needs.
   It does not import their framework boundaries
   or add a new generic compiler layer.

The review conclusion is that this design follows the overall
IR-first architecture,
provided the implementation keeps the transition rule strict:
when a new typed field starts feeding a downstream phase,
the corresponding pass-local fallback must be deleted in the same
implementation phase.

## Global Task Order

This design owns the first four implementation units in the
post-review queue:

1. `AccessRegion` foundation
2. graph-backed `SpatialPlan` dependence construction
3. `LiveValueSSA`
4. TT live-form solver

The first four implementation units above are now classified as
foundation-complete,
not usage-complete.
Before the tile compute legalizer /
DAG covering lane starts,
the active chain must complete a decision-use cutover:

1. `SpatialPlan` decision-use cutover
   - `AccessRegion`
     must be used for full/slice compatibility,
     access overlap,
     and materialization boundary coverage.
   - `DependenceComponent`
     must drive recurrence /
     loop-carried lifetime legality.
2. `LiveValueSSA` query cutover
   - downstream planning must query by live value edge /
     materialization boundary indices,
     not by subject-pair maps.
   - stale source-live fallback remains illegal.
3. TT live-form solver cutover
   - solver input is the set of validated
     `LiveValueEdge`
     and
     `MaterializationBoundary`
     records.
   - output is typed
     `TTLiveFormPlan`,
     `TTMaterializationPlan`,
     and
     `TTConsumerBindingPlan`.
   - unsupported joins or event lifetimes fail closed
     with typed diagnostic.

Only after those decision-use gates are active should the
tile compute legalizer /
DAG covering lane rely on them for pattern legality,
fanout handling,
and materialization-aware covering.
The runtime admission lanes start only after target planning can
project that evidence into typed
`TTProgram -> ExecutableSpec`
records.

## Implementation Plan

### Phase A: AccessRegion Foundation

Status: completed in repo HEAD.

Files:

- create `tilelang_repo/src/transform/common/spatial_access_region.h`
- create `tilelang_repo/src/transform/common/spatial_access_region.cc`
- modify `tilelang_repo/src/transform/common/spatial_plan.h`
- modify `tilelang_repo/src/transform/common/spatial_plan.cc`
- modify `tilelang_repo/src/transform/build_spatial_plan.cc`
- modify `tilelang_repo/src/transform/validate_spatial_plan.cc`
- modify Python rebuild helpers and tests under
  `tilelang_repo/testing/python/transform/`

Work:

1. Add `AccessRegion` object and reflection.
2. Build access regions from
   `BufferLoad`,
   `BufferStore`,
   `RegionOp`,
   and tile op dataflow access info.
3. Add validator checks for subject,
   unit index,
   rank,
   positive extents,
   and supported coverage kind.
4. Add tests for full tile,
   row slice,
   grouped slice,
   scalar,
   and unsupported non-affine access.

Completion gate:

- `SpatialPlan.access_regions`
  covers every read/write recorded in execution units.
- existing Blackhole transform tests still pass.

### Phase B: Dependence Graph Builder

Status: completed in repo HEAD.

Files:

- create `tilelang_repo/src/transform/common/spatial_dependence_graph.h`
- create `tilelang_repo/src/transform/common/spatial_dependence_graph.cc`
- modify `build_spatial_plan.cc`
- modify `validate_spatial_plan.cc`
- modify tests in
  `test_blackhole_spatial_ir.py`

Work:

1. Build per-subject access event lists.
2. Construct flow/carry/reduction/broadcast/join edges
   using access overlap and tile op evidence.
3. Add SCC detection and optional
   `DependenceComponent`.
4. Replace ad hoc local value flow edge construction
   with graph builder output.
5. Keep compatibility projection tests green.

Implementation notes:

- `SpatialPlan` now owns typed `DependenceComponent`
  recurrence diagnostics.
- `spatial_dependence_graph.{h,cc}` builds closure compatibility
  boundaries from ordered `AccessRegion` read/write events and emits
  same-unit materialize edges from local value-flow evidence.
- `build_spatial_plan.cc` no longer decides flow/carry/join edge
  construction from private closure read/write vectors; it projects the
  typed access graph back to `ClosureBoundary` only for compatibility.
- Current Phase B emits the edge kinds already admitted by downstream
  `LiveValueEdge` and materialization planning
  (`flow`, `carry`, `join`, `materialize`).
  `reduction`, `broadcast`, `output_order`, and `anti_order`
  remain reserved design vocabulary for later legalization/scheduling
  admission, because the active downstream validators do not yet consume
  those edge kinds.

Completion gate:

- preserved reduce emits both consume and produce edges.
- flash-attn row max/sum/scale dataflow edges are derived
  from graph evidence.
- no DataflowEdge kind is decided by buffer name.

### Phase C: LiveValueSSA

Status: completed in repo HEAD.

Files:

- modify `spatial_plan.h/.cc`
- modify `build_spatial_plan.cc`
- modify `validate_spatial_plan.cc`
- modify `build_tt_program.cc`
- modify `lower_blackhole_state.cc`
- modify `lower_blackhole_materialization.cc`
- modify `validate_tt_program.cc`
- modify `tilelang_repo/src/target/tt_program_projection.h`
- update Python tests and typed rebuild helpers

Work:

1. Make `LiveValue`
   carry version / definition-kind evidence.
2. Add def-use construction from dependence graph edges.
3. Add phi records for carry/reduction/join SCCs.
4. Strengthen materialization boundary construction
   from source/target versions.
5. Update TT live-form planning to consume
   versioned `LiveValue`
   rather than pass-local source heuristics.
6. Keep old object names only as compatibility
   projection where needed;
   do not introduce payload fallback.

Implementation notes:

- `LiveValue` now carries
  `version_index`,
  `definition_kind`,
  `defining_access_region_index`,
  and `defining_event_index`.
- `LiveValueEdge` now carries
  `use_kind`,
  `consumer_access_region_index`,
  `source_version_index`,
  and `target_version_index`.
- `MaterializationBoundary` now carries
  source/target access-region indices,
  `event_lifetime_kind`,
  and publish/consume page bounds.
- The builder resolves local materialize edges to the reaching source
  live version instead of treating a consumer-side source as a new
  definition.  This keeps the fragment-fill -> cast -> publish path from
  producing stale synthetic source definitions.
- TT validation now requires version/use/lifetime evidence on
  `TTLiveFormPlan`, `TTMaterializationPlan`, and
  `TTConsumerBindingPlan` references back to `SpatialPlan`.

Completion gate:

- read-write accumulator tests prove old version is consumed
  before new version is defined.
- stale synthetic fill fallback is rejected when a reaching
  upstream live value exists.
- exact-CB republish events reference source live versions.

### Phase D: Live-Form Solver And Runtime Admission

Status: completed in repo HEAD for the current fragment/cast live-form
transfer surface.  Wider flash-attn runtime admission remains a later
runtime support lane, not part of this phase completion.

Files:

- create `tilelang_repo/src/transform/common/tt_live_form_solver.h`
- create `tilelang_repo/src/transform/common/tt_live_form_solver.cc`
- modify `lower_blackhole_state.cc`
- modify `lower_blackhole_materialization.cc`
- modify `validate_tt_program.cc`
- modify direct runtime unsupported-reason tests

Work:

1. Implement lattice and transfer functions.
2. Replace local exact-CB/live-form maps with solver output
   where the value must survive across events.
3. Project solver decisions into
   `TTLiveFormPlan`,
   `TTMaterializationPlan`,
   and `TTConsumerBindingPlan`.
4. Keep pass-local caches only as derived implementation state.
5. Admit multi-block flash-attn only after solver evidence
   is visible in typed `TTProgram -> ExecutableSpec`.

Implementation notes:

- Added `tt_live_form_solver.{h,cc}` with typed request/result structs for
  fragment/cast live-form transitions.
- `PlanTTKernelABI` now delegates physical form,
  topology,
  ownership,
  materialization protocol,
  publication protocol,
  and consumer full/slice requirements to the solver result before
  constructing `TTLiveFormPlan`,
  `TTMaterializationPlan`,
  and `TTConsumerBindingPlan`.
- Pass-local maps remain only as implementation caches for CB/order state.
  The solver-visible inputs come from typed `SpatialPlan`
  live values and materialization boundaries.
- Seq64 / multi-K-step flash-attn remains outside direct-runtime
  admission and still fails closed through the typed direct-runtime
  unsupported-reason metadata gate.

Completion gate:

- seq64 / multi-K-step flash-attn either passes TT-Sim bf16
  or fails closed with a typed missing contract reason.
- no runtime-only patch is needed to select source-live-form.

### Phase E: Decision-Use Cutover

Status: in progress.

This phase corrects the current gap where the algorithmic structures
exist but are not yet broadly decisive.
It is a broad main-chain cutover,
not a single workload-specific implementation.

Current implementation notes:

- `E0 Review`
  has been completed for the current scope and folded back into
  this document plus the tile-compute legalizer design.
- The first `E1 Spatial legality` slice is active:
  `AccessRegion`
  compatibility now gates distributed-slice consumer acceptance,
  and recurrence
  `DependenceComponent`
  evidence now gates loop-carried materialization lifetime.
- Validators now reject distributed-slice live edges without
  producer/consumer access-region evidence and loop-carried
  boundaries without recurrence component evidence.
- The first `E2 Indexed query` slice is active for
  fragment/cast materialization planning:
  `PlanTTKernelABI`
  now looks up `MaterializationBoundary`
  by source/target live-value indices rather than
  source/target subject pair.
  Subject lookup remains only as a derived entry point from the current
  TIR buffer to the latest `LiveValue`;
  the wider subject-map deletion remains pending.
- The first `E3 Solver` slice is active:
  `tt_live_form_solver`
  consumes boundary logical coverage and changes consumer
  full-tile vs distributed-slice requirements from typed
  `MaterializationBoundary`
  evidence.

Remaining work:

- remove remaining owner-truth dependence on subject-level live-value
  lookup in downstream TT planning,
- expand the solver into a graph worklist/lattice over all validated
  live edges,
- add broader TTProgram / ExecutableSpec projection validators for
  selected live edge /
  boundary /
  compute plan evidence,
- only then start production DAG covering migration.

Files:

- modify `build_spatial_plan.cc`
- modify `validate_spatial_plan.cc`
- modify `lower_blackhole_state.cc`
- modify `lower_blackhole_ops.h`
- modify `tt_live_form_solver.{h,cc}`
- update planner and runtime tests

Work:

0. Re-audit each existing algorithmic structure
   against the Coverage Model above before adding more code.
   A structure is allowed to remain pass-local only if no downstream
   legality /
   query /
   action decision depends on it after the pass returns.
1. Make `BuildMaterializationBoundaries`
   consume `DependenceComponent`
   evidence when choosing
   `event_lifetime_kind`.
2. Make `LiveValueEdge`
   full/slice consumer requirements
   depend on `AccessRegion`
   compatibility,
   not only local flow traits.
3. Add validator rules that reject:
   missing access-region evidence for distributed-slice consumers,
   loop-carried boundaries without recurrence component evidence,
   and TT plans that do not reference the exact live edge /
   boundary selected by the solver.
4. Replace
   `PlanTTKernelABI`
   subject-pair live-value maps with indexed
   `LiveValueEdge`
   / `MaterializationBoundary`
   queries.
5. Expand `tt_live_form_solver`
   from a fragment/cast helper into a small worklist solver
   over the validated live-value graph.
   The first supported lattice states are:
   `Fragment`,
   `Accumulator`,
   `ExactCB(single_event)`,
   `ExactCB(multi_event)`,
   `HostVisible`,
   `Conflict`,
   and
   `Unsupported`.
6. Keep pass-local maps only as caches derived from the current IR.
   A cache cannot be the owner truth for legality,
   source live form,
   or event lifetime.
7. Carry the same decision-use rule through
   `TTProgram -> ExecutableSpec`
   projection:
   leaf records must preserve the selected live edge /
   boundary /
   compute plan indices needed for validators
   and runtime admission diagnostics.

Implementation order:

1. `E0 Review`
   - compare `AccessRegion`,
     `DependenceComponent`,
     `LiveValueSSA`,
     solver,
     and tile-compute DAG design
     against the prior-art contracts above.
   - update docs before further code.
2. `E1 Spatial legality`
   - access-region slice/full compatibility,
     recurrence-backed lifetime,
     validator rejects.
3. `E2 Indexed query cutover`
   - downstream TT planning queries
     live edges /
     materialization boundaries
     by stable indices,
     with subject maps reduced to derived caches only.
4. `E3 Solver cutover`
   - worklist/lattice live-form propagation over validated edges.
5. `E4 Projection/admission audit`
   - ensure typed TT plans
     and executable records retain enough evidence
     for fail-closed runtime/codegen decisions.

Completion gate:

- tests prove that changing
  `AccessRegion`,
  `DependenceComponent`,
  or
  `MaterializationBoundary`
  evidence changes legality /
  solver output /
  typed TT plans.
- existing copy,
  GEMM,
  fragment-fill-cast-publish,
  and flash-attn metadata-gate tests remain green.
- no admitted path gets source-live-form,
  event lifetime,
  or consumer full/slice requirements
  from names or stale helper facts.

## Testing Strategy

### Structure tests

- `SpatialPlan.access_regions`
  exact count / kind / coverage for copy,
  GEMM,
  reduce,
  flash-attn.
- `DataflowEdge`
  edge kind and producer/consumer indices.
- SCC / recurrence component
  for carry/reduction loops.
- `LiveValue`
  version ordering and def-use links.
- `MaterializationBoundary`
  source/target version and event lifetime.

### Planner tests

- `TTLiveFormPlan`
  references spatial live value index.
- `TTMaterializationPlan`
  references boundary index and produced live form.
- `TTConsumerBindingPlan`
  references live value edge index.
- validator rejects missing version,
  stale source,
  or incompatible form join.

### Runtime tests

- existing copy/GEMM/flash-attn bf16 gates remain green.
- multi-block flash-attn becomes the first admission target.
- larger stage2/block64 remains gated until multi-page event
  lifetime is represented and verified.

## Migration Discipline

- Add new typed objects before deleting old mechanics.
- Keep compatibility projection only inside the current pass
  implementation boundary;
  do not expose new public payloads.
- Every new typed field must be reflected in:
  C++ object schema,
  producer pass,
  validator,
  TTProgram projection,
  Python rebuild helper,
  and tests.
- Once a consumer reads the new typed object,
  delete the corresponding pass-local fallback in the same change.
- Each phase must end with a commit whose tests prove both old admitted
  support and the new invariant.

## Expected Payoff

- New compute patterns become graph / access / live-form cases,
  not workload-specific matchers.
- Multi-block flash-attn uses the same live-value versioning
  as future fused reductions and matmul epilogues.
- Exact-CB event lifetime becomes a typed planner fact,
  not hidden source-generation state.
- Runtime admission stays a leaf decision;
  upstream representation remains target-contract clean.
