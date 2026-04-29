# Blackhole Tile Compute Legalizer And DAG Covering

## Goal

把 Blackhole tile compute selection
从手写 per-op branch
推进到可扩展的
legalizer + DAG covering
模型。

本设计服务于
`Normalized Tile TIR`
里的 TT-Metal leaf API 粒度 compute preservation：

```text
TIR scalar expression / generic tile op semantics
  -> explicit leaf tile-compute TIR normalization
  -> TileComputeDAG over explicit leaf nodes
  -> legalization
  -> target leaf pattern covering
  -> TTComputeOpPlan / leaf source sequence
```

The usefulness gate for this design is not that a DAG exists.
It is that DSL-authored tile compute lowers to real TT-Metal hardware code
more reliably or more efficiently:
the legalizer must either choose a typed leaf plan,
produce a typed unsupported diagnostic before source/runtime emission,
feed resource planning,
or delete an old per-op matcher /
payload /
fallback.
Diagnostic-only DAG construction,
metadata that source/runtime does not need to act on,
or source hooks that recover composite semantics
do not satisfy this gate.

它不是新的长期 IR 层。
`TileComputeDAG`
是显式 leaf
`Normalized Tile TIR`
到 `TTProgram.compute_op_plans`
之间的 pass-local selection model；
如果某个选择结果需要跨阶段保留，
必须落到 typed
`TTComputeOpPlan`
/
`TTCBPlan`
/
`TTLiveFormPlan`
/
`TTMaterializationPlan`
中。

`TileComputeDAG`
不负责把 composite TIR expression
展开为多个 target leaf op。
这类展开必须发生在
`Normalized Tile TIR`
normalization 边界，
输出为多个显式
`tl.tileop.blackhole_compute`
leaf statements
或等价的 generic tile-op leaf statements。
DAG covering
只能覆盖这些已经显式存在的 leaf nodes，
不能把一个 source node
解释成一个会在 source emission
阶段继续产生多条 semantic compute op
的 composite decision。

`2026-04-29` 收缩后再固定一条：
`TileComputeDAG`
不能只把 fanout /
materialization
结果送进 resource admission。
凡是 DAG covering 决定了 leaf pattern、
materialization policy
或 fanout policy，
这些决定必须写入 typed lower plan，
至少进入
`TTComputeOpPlan`
的 DAG node /
materialization /
fanout 字段，
并由 validator /
resource planning /
executable projection
在需要时继续携带。
source lowering 只能消费这份 pass-local DAG lower plan；
不能在 explicit tile-compute source path
重新按 operation name 做第二次 production selection。
如果实现为了 codegen
保留 source hook，
它只能是当前 semantic leaf
的投影策略，
不能授权 hook
展开成多个 semantic
`TTComputeOpPlan`
或多个不同 operation family。

当前实现状态：

- `BuildBlackholeTileComputeDAG`
  仍是 pass-local analysis，
  不作为跨阶段 IR 层保存。
- `PlanTTKernelABI`
  在 source lowering 前构造
  DAG lower plan，
  `LowerExplicitTileComputeCall`
  消费该 plan 中的 selected covering
  决定 leaf source hook。
- DAG-driven exact compute
  会把 source DAG node id、
  materialization policy、
  fanout use count
  和 fanout policy
  记录到
  `TTComputeOpPlan`；
  source hook 只能作为 leaf projection metadata，
  不能成为 runtime-facing semantic truth。
- `ValidateTTProgram`
  必须校验 DAG source node、
  selected covering、
  和
  `TTComputeOpPlan.operation_name`
  指向同一个 semantic leaf op。
  一个 DAG source node
  不能合法展开成多个不同
  `TTComputeOpPlan.operation_name`
  entry；
  如果需要多步 lowering，
  这些步骤必须已经在
  `Normalized Tile TIR`
  中显式拆成多个 leaf nodes。

Boundary correction status (`2026-04-29`):
the known implementation residues have been repaired.
Composite
`exp2(lhs * s0 - rhs * s1)`
is no longer packed behind an
`exp2_tile`
source payload;
the frontend normalizer emits explicit leaf tile-compute statements before
DAG construction.
Row-broadcast division is no longer packed behind
`mul_tiles_bcast_cols("div", ...)`;
it is represented as
`recip_tile`
plus
`mul_tiles_bcast_cols`.
`PlanTTKernelABI`
does not own a string-mode composite source dispatcher for these paths.
DAG-driven source hooks are constrained to one selected semantic leaf plan.

## References

本设计借鉴 LLVM codegen 的两个经典点：

- SelectionDAG:
  build DAG,
  optimize,
  legalize,
  optimize again,
  select target instructions,
  then schedule。
- TableGen / pattern selection:
  target instruction 的 operands、
  legality、
  pattern
  和 cost metadata
  不应该散在手写 if/else 中。

这里不引入 TableGen。
第一版用 C++ typed schema table
表达 pattern 和 legalization rule。

补充约束：

- LLVM GlobalISel 的 useful boundary
  不是“有一个 legalizer 类”，
  而是 legalizer action
  决定后续 selector
  能否处理当前 IR。
  对应本设计，
  `TileComputeDAG`
  不能只是 dump；
  legalizer 必须产生
  `Legal / Lower / Split / PromoteDType / Materialize / Reject`
  action，
  并且 action 必须影响
  `TTComputeOpPlan`
  或 typed unsupported diagnostic。
- MLIR pattern infrastructure
  把 cost model、
  pattern benefit、
  match failure reason
  放在 driver contract 中。
  对应本设计，
  每个 rejected leaf pattern
  必须能解释缺失的 access/live/dependence evidence。
- Cranelift ISLE 要求 lowering rules
  保持 pure SSA rewrite，
  side effect 在 rule commit 后发生。
  对应本设计，
  DAG matcher/legality predicate
  不分配 CB、
  不改写 TIR、
  不发 source；
  只有 selected pattern
  的 leaf source hook
  可以把已经选择好的一个 semantic leaf op
  投影成 source micro-sequence
  和对应 typed plan metadata。
  它不能承担 expression decomposition
  或 composite-to-leaf lowering。
- LLVM VPlan
  把多个 transformation candidates
  放进显式 plan，
  用 cost / legality /
  plan-to-plan transforms
  决定最终 IR translation。
  对应本设计，
  DAG covering
  不能只复刻当前 per-op branch；
  它需要保留候选 pattern、
  materialization choice、
  live-form cost
  和 reject reason，
  让复杂 epilogue /
  fanout /
  event-lifetime-sensitive
  选择可审计。
  复杂 epilogue
  如果包含多条 semantic compute op，
  必须先在
  `Normalized Tile TIR`
  中变成多条显式 leaf statements，
  再进入 DAG covering。
- LLVM LoopAccessAnalysis
  的经验是：
  access legality query
  要能解释 vectorization /
  transform
  失败原因，
  必要时可以生成显式 runtime check。
  Blackhole 第一版不新增 runtime check，
  但必须把 unsupported 原因落到 typed diagnostic，
  不能让 source emitter 隐式跳过。
- MLIR Linalg
  说明 structured compute semantics
  和 indexing maps
  是通用 tiling /
  fusion /
  library-special-instruction lowering
  的前提。
  对应本设计，
  `TileComputeDAG`
  的输入必须是 explicit preserved leaf tile compute semantics
  加 `AccessRegion` /
  `LiveValueSSA`
  证据，
  不能从 scalar loop idiom
  重新恢复。

## Non-Goals

- 不恢复 downstream scalar-loop matcher。
- 不引入 composite operation name：
  `softmax`,
  `exp2_affine`,
  `row_broadcast_exp2_affine`
  等不能进入生产 compute protocol。
- 不把 composite semantics
  伪装成 leaf operation payload：
  `exp2_tile(mode, lhs, rhs, scale, ...)`
  和
  `mul_tiles_bcast_cols("div", ...)`
  这类 leaf-looking composite source call
  与 composite operation name
  同样禁止。
- 不把
  `TileComputeDAG`
  或 source emitter
  当成 expression normalizer。
  TIR expression
  到多条 TT-Metal leaf op
  的分解必须发生在
  `Normalized Tile TIR`
  normalization 边界。
- 不把 DAG covering 结果作为新的 cross-pass payload。
- 不绕开
  `SpatialPlan`
  /
  `TTProgram`
  /
  `ExecutableSpec`
  validator。
- 不用 equality saturation 作为第一版主算法；
  e-graph 最多未来用于局部 algebraic exploration。

## Current Problem

当前 compute selection 已经比旧路径干净：

- tile compute semantics
  已在 `Normalized Tile TIR`
  中保留 / 规范化。
- old scalar matcher / generate family
  已删除。
- `TTComputeOpPlan.operation_name`
  已保持 TT-Metal leaf API 粒度。

但 selection 仍然偏手写：

- `lower_blackhole_tile_compute.cc`
  对 fill/copy/typecast/binary/broadcast/reduce/exp2
  分别写 match/emit 分支。
- legality check、
  operand role extraction、
  CB requirement、
  live-form source choice、
  source sequence emission
  仍混在同一类逻辑附近。
- 增加新 compute 类型时，
  容易复制一整套 branch，
  而不是添加一条 declarative-ish pattern。
- `2026-04-29`
  边界修复后，
  已知复合 TIR 表达式不再被压进 leaf-looking
  `tl.tileop.blackhole_compute`
  payload。
  如果后续发现新的 multi-step compute expression，
  必须同样在
  `Normalized Tile TIR`
  中显式 leaf-sequence 化，
  不能把展开推迟到
  `PlanTTKernelABI`
  source hook。

后续复杂模式需要更通用的选择机制：

- multi-step epilogue
- fused reduce + broadcast + binary
- multiple legal lowering choices
- exact-CB event lifetime-sensitive choice
- dtype / accumulator / fragment form choice
- pack / tilize / untilize placement

其中 multi-step epilogue
表示 DAG 可以在多个显式 leaf nodes
之间做 legality /
fanout /
materialization reasoning，
不表示一个 source leaf node
可以隐藏多步 semantic lowering。

## End State

### 1. `TileComputeDAG`

用 pass-local DAG 表达当前 compute selection
面对的 tile-level compute program。

建议数据结构：

```text
TileComputeNode
  id
  op_kind                 // generic_tile_op | blackhole_leaf_op | materialize | pack
  op_name                 // reduce, add, mul_tiles, exp2_tile, pack_tile, ...
  dtype
  accumulator_dtype
  logical_shape
  tile_shape
  access_region_indices
  input_edges
  output_edges
  side_effect_class       // none | tile_regs | cb | dst | memory
  token_input
  token_output
  anchors

TileComputeEdge
  id
  producer_node
  consumer_node
  value_role              // lhs | rhs | src | dst | scale | accumulator | output
  live_value_index
  access_region_index
  requires_materialization
```

Token edges are required for side-effecting
CB / tile-reg / pack operations.
They are not semantic payloads;
they are pass-local ordering mechanics.

Construction inputs:

- preserved `tl.tileop.*`
- `tl.tileop.blackhole_compute`
- `AccessRegion`
- `LiveValueSSA`
- existing `TTProgram` staged kernel/block context

### 2. Canonicalization

Before legalization and covering,
normalize the DAG into predictable form.

Required canonicalizations:

- commutative operands sorted by stable role
  and live value identity
- constants/scales on the right-hand side
- cast chains collapsed when dtype legality allows
- row/column broadcast axis normalized
  to explicit `broadcast_axis`
- reduce axis normalized against logical rank
- fill values normalized:
  `0`,
  `-inf`,
  scalar literal
- copy/typecast canonicalized as unary leaf candidates
- materialization nodes explicit,
  not hidden inside pattern emission

This pass may simplify representation,
but it must not invent compute semantics
not present in current TIR / typed graph evidence.

### 3. Legalization

Legalization classifies each node:

```text
Legal
Lower
Split
PromoteDType
Materialize
Reject
```

Rule examples:

- `reduce(sum|max, axis=1, tile=32x32)`
  -> legal row reduction leaf.
- unsupported reduce axis
  -> reject unless a generic decomposition exists
  in `Normalized Tile TIR`.
- binary op with exact-CB live inputs
  -> legal if both operands have compatible form.
- binary op with one fragment-only producer
  -> materialize source first if
  `LiveValueSSA`
  and event lifetime permit.
- multi-page publish/consume event
  -> reject until event lifetime contract is admitted.

Legalization output is still a DAG.
It does not emit final source.

### 4. Pattern Covering

A `TTLeafPattern`
describes one TT-Metal leaf API
or a small required init/body/pack sequence.

Suggested schema:

```text
TTLeafPattern
  name
  root_op_name
  result_kind
  operation_name          // TT-Metal leaf API name for TTComputeOpPlan
  operand_roles
  required_input_forms
  produced_form
  side_effect_class
  legality_predicate
  cost_model
  emit_plan
  emit_source_sequence
```

`operation_name`
must remain leaf API granularity:

```text
fill_tile
copy_tile
typecast_tile
binary_max_tile
mul_tiles
add_tiles
mul_tiles_bcast_cols
add_tiles_bcast_cols
exp2_tile
reduce_tile
pack_tile
recip_tile
matmul_tiles
```

Covering algorithm:

1. Walk DAG bottom-up.
2. For each node,
   enumerate legal patterns whose root matches.
3. Compute candidate cost:
   pattern cost +
   operand materialization cost +
   live-form transition cost +
   CB page pressure cost +
   event lifetime risk cost.
4. Keep best candidate per
   `(node, required_output_form)`.
5. For DAG fanout,
   share producer if lifetime permits;
   otherwise add explicit materialization/copy.
6. Emit selected pattern set in dependence order.

第一版可以实现成 greedy + local DP：

- trees use bottom-up DP
- DAG fanout uses conservative materialization
- no global ILP
- no e-graph saturation

如果后续 cost conflict 变复杂，
再考虑 global DP / shortest path /
ILP。

### 5. Emission

Pattern covering produces two outputs:

1. typed plans:
   - `TTComputeOpPlan`
   - `TTCBPlan`
   - `TTLiveFormPlan`
   - `TTMaterializationPlan`
   - `TTConsumerBindingPlan`
2. source sequence:
   - init calls
   - wait/reserve/pop/push
   - tile register acquire/commit/wait/release
   - leaf compute API calls
   - pack/publication calls

Source emission must be a leaf projection of selected
typed plans.
It must not run another semantic matcher.

## Integration With The Algorithmic Generalization Lane

This design depends on the three earlier foundations:

- `AccessRegion`
  supplies shape/coverage/axis legality.
- graph-backed `SpatialPlan`
  supplies producer/consumer dependence and recurrence.
- `LiveValueSSA`
  supplies source versions,
  consumer uses,
  and event lifetime evidence.

Without those,
DAG covering would merely relocate today’s branch logic.

Algorithmic Generalization has selected
`Phase E: Decision-Use Cutover`
on the active chain for admitted live-form /
materialization decisions.
That is sufficient for the compute legalizer to consume
`AccessRegion`,
`DependenceComponent`,
and
`LiveValueSSA`
as typed evidence, because those structures already drive:

- access full/slice compatibility,
- recurrence / loop-carried lifetime legality,
- source-live reaching-def queries,
- TT live-form solver output,
- and typed materialization / consumer binding plans.

It is not a claim that algorithmic generalization solved compute expression
lowering,
global lifecycle planning,
or hardware-aware resource allocation.
Those remain separate contracts under the hardware-codegen usefulness gate.

If future covering work bypasses this gate,
`TileComputeDAG`
would become another parallel matcher surface,
which violates the final backend redesign.

The gate is intentionally broader than a single solver call.
Before this lane selects production compute plans,
the following must remain true on the active chain:

- `AccessRegion`
  changes legality for slice/full,
  axis,
  and materialization coverage decisions.
- `DependenceComponent`
  changes legality for recurrence /
  loop-carried lifetime decisions.
- `LiveValueSSA`
  answers source-live reaching-def
  and consumer binding by explicit indices.
- the live-form solver
  changes physical form /
  consumer requirement /
  materialization protocol
  from typed boundary evidence.
- validators reject missing or inconsistent evidence
  before source emission.

With those facts active,
a DAG legalizer has stable facts to cover.
If a later implementation drops them,
the legalizer would be forced to rebuild semantics
with its own matcher,
which is exactly the architecture being deleted.

This lane also inherits the Algorithmic Generalization
anti-overdesign rule:
`TileComputeDAG`,
legalization,
and covering
are justified only when they change selected typed compute plans,
emit earlier typed reject diagnostics,
or delete current per-op branch mechanics.
A read-only DAG dump,
pattern table,
or generic covering class
is foundation work only.
It must not be reported as production completion until its decision
affects `TTComputeOpPlan`,
`TTLiveFormPlan`,
`TTMaterializationPlan`,
`TTConsumerBindingPlan`,
or a typed unsupported reason on the active chain.

It also inherits the problem-family generality rule.
The first migrated op family is only a witness that DAG covering is
on the active chain.
It must not become the covering contract.
The covering model must stay valid for the TT-Metal leaf compute
problem family:
unary,
binary,
broadcast,
reduce,
pack,
matmul,
fanout,
and materialization-aware selection.

If the admitted surface is too small for the DAG to make a real graph-level
decision,
the DAG remains foundation/debug infrastructure.
It becomes a production mechanism only when leaf-graph fanout,
materialization,
physical-form,
resource-demand,
or typed reject decisions change downstream behavior or delete old branches.

If a proposed pattern table or legalizer rule can explain only the
current workload shape,
it belongs in a narrower local helper,
not in the production DAG covering design.

## Architecture Review

This section records the design review against
`final_blackhole_backend_redesign.md`
and compiler architecture practice.

Review scope:

- only tile compute selection:
  `TileComputeDAG`,
  legalization,
  TT-Metal leaf pattern covering,
  and source/plan emission for selected leaf operations
- not broader transport,
  ABI,
  mesh,
  or runtime admission expansion

Primary checks:

1. No new long-lived IR layer is introduced.
   `TileComputeDAG`
   is a pass-local selection model.
   It may be dumped for diagnostics and tested structurally,
   but no downstream phase may consume it as owner truth.
   The only durable outputs are typed
   `TTComputeOpPlan`,
   `TTCBPlan`,
   `TTLiveFormPlan`,
   `TTMaterializationPlan`,
   and `TTConsumerBindingPlan`
   records.

2. Semantics enter from explicit IR,
   not from late scalar-loop recovery.
   DAG nodes are built from preserved tile compute calls,
   tile op dataflow evidence,
   `AccessRegion`,
   `LiveValueSSA`,
   and validated planning context.
   Operation family,
   operand role,
   broadcast axis,
   reduction axis,
   and live-form source
   cannot be inferred from buffer names,
   source text,
   or workload-specific helper names.

3. Legalization is separate from emission.
   The legalizer answers whether an operation is legal,
   needs lowering,
   needs materialization,
   needs dtype promotion,
   or must be rejected.
   It does not emit source and does not allocate runtime artifacts.
   Pattern covering chooses among legal leaf patterns.
   Source emission is a mechanical projection of the selected typed
   plans and pattern IDs.
   If legalization says a TIR expression needs lowering into multiple
   semantic leaf ops,
   that lowering target is explicit
   `Normalized Tile TIR`,
   not a source emitter side effect.

4. Leaf granularity is preserved.
   `TTComputeOpPlan.operation_name`
   remains TT-Metal API granularity:
   `mul_tiles`,
   `add_tiles`,
   `*_bcast_cols`,
   `exp2_tile`,
   `reduce_tile`,
   `pack_tile`,
   `matmul_tiles`,
   and similar leaf APIs.
   Composite helpers such as
   `softmax`,
   `exp2_affine`,
   and row-broadcast affine variants
   stay out of the production protocol.
   Leaf-looking source calls with composite payloads are also forbidden:
   operation name,
   operand role schema,
   and source hook
   must describe the same single TT-Metal semantic leaf.

5. Lowering is an explicit information trade.
   TIR normalization may freeze a target leaf sequence,
   explicit logical temps,
   and semantic copy/materialization requirements.
   DAG covering may then freeze target leaf pattern choices,
   operand forms,
   and materialization points,
   but it must preserve enough typed evidence for validators,
   source projection,
   and runtime admission to audit the decision.
   If event lifetime or live-form source cannot be proven,
   the candidate is admission-blocked,
   not merely expensive.
   This is distinct from semantic unsupported:
   before using an unsupported diagnostic for compute semantics,
   the backend must audit whether TT-Metal already has the primitive,
   whether the Blackhole wrapper/codegen has failed to expose it,
   or whether existing leaf ops can express the value through an explicit
   TIR sequence.

6. Compiler practice alignment:
   the design borrows SelectionDAG-style
   build/legalize/select structure
   and TableGen-style pattern metadata,
   but implements them as local C++ typed tables
   because the current backend needs bounded target selection,
   not a new multi-target instruction selection framework.

The review conclusion is that the design follows the overall
IR-first architecture
provided the implementation keeps one invariant non-negotiable:
covering output must be typed plans,
not a replacement matcher,
not a source-string protocol,
and not a pass-to-pass DAG payload.

### Resource-Planning Review Addendum (`2026-04-29`)

The resource-planning review narrows this design's production boundary.

`TileComputeDAG`
is not a global dataflow analysis,
resource allocator,
core placer,
NoC scheduler,
or lifecycle engine.
It is only the pass-local tile-compute selection model used to choose legal
TT-Metal leaf compute patterns.

The following responsibilities are outside this design and belong to the
resource-planning roadmap:

- CB live-interval allocation
- L1/SRAM pressure admission
- hardware-model-backed core placement
- buffer distribution / sharding decisions
- multicast / NoC / scheduling optimization

This means a production migration is acceptable only when it pays rent through
leaf-graph decisions,
not through composite source expansion.
Accepted outputs are:

- changes a typed compute / CB / live-form /
  materialization / consumer-binding plan,
- emits a typed unsupported reason earlier than source/runtime emission,
- or deletes an old per-op branch / fallback path.

A cursor-based hook chain,
`try_*` family,
or source-emitter registry that merely wraps the old branch logic is not enough
to claim production completion.
Such mechanics must either be simplified into pattern-owned selected emission
or deleted.

The next resource-aware work is therefore not to expand this DAG into a
planner-owned resource system.
The first typed-resource step is DAG-backed
`ResourceDemand`
/
`ResourcePressureReport`
that consumes pass-local fanout /
materialization decisions.
That first typed surface exists in repo HEAD,
but it is not enough by itself to prove the DAG should remain on the production
path.
With the known composite pseudo-leaf cleanup complete,
the DAG must still prove that its leaf-graph fanout /
materialization /
resource-demand decisions affect typed plans,
validators,
or admission diagnostics.
The remaining resource-planning work is to upgrade CB / L1 / core / buffer
planning in `TTProgram`
and `ExecutableSpec`
as described by
`2026-04-29-blackhole-resource-planning-roadmap.md`.

Implementation status:

- production code does not persist a function-level
  `TileComputeDAG`
  covering cache in
  `PlanTTKernelABI`
- `blackhole_tile_compute_covering.h`
  exposes leaf covering decisions and diagnostic FFI only;
  it does not expose a durable
  `BlackholeTileComputeDAGCovering`
  production object
- explicit source emission continues through selected leaf pattern
  source hooks
  and the known composite source payload violations have been removed:
  each admitted hook now projects one selected semantic leaf op instead of
  expanding a composite payload into multiple compute plans
- static tests guard these boundaries before wider resource-planning expands
- DAG-wide fanout /
  materialization /
  unsupported covering reasons are now merged into
  `ResourceDemand`
  /
  `ResourcePressureReport`:
  the full pre-selection DAG feeds typed demand,
  `ValidateTTProgram`
  consumes matching reports and typed unsupported reasons,
  and executable projection carries
  `resource_pressure_reports`.
  The DAG itself remains pass-local and is still not a durable planning layer.

The source-lowering repair is complete for the known active residues:
`exp2_tile`
composite payloads and
`mul_tiles_bcast_cols("div", ...)`
are deleted from active lowering and moved into explicit
`Normalized Tile TIR`
leaf normalization.
Any future source hook that records multiple semantic
`TTComputeOpPlan`
entries for one DAG source node is still a violation and must be deleted or
moved into explicit leaf normalization before admission.

## Global Task Order

This lane is sequenced after the selected algorithmic generalization
decision-use facts,
but before wider core / buffer / runtime expansion:

1. Keep or delete production DAG mechanics based on whether they drive
   leaf-graph fanout /
   materialization /
   physical-form /
   resource-demand /
   typed reject decisions.
2. Use the repaired explicit leaf graph while upgrading core /
   buffer placement and wider resource admission.
5. Resume core /
   buffer /
   runtime admission only after selected patterns,
   live-form solver,
   and resource plans produce typed
   `TTProgram -> ExecutableSpec`
   evidence.

## Implementation Plan

### Phase A: Pattern Schema And Read-Only DAG Dump

Status: completed in repo HEAD as foundation.
The diagnostic surface is pass-local /
test-visible only;
it is not a new IR layer and is not persisted as a cross-pass payload.

Files:

- create `tilelang_repo/src/transform/common/blackhole_tile_compute_dag.h`
- create `tilelang_repo/src/transform/common/blackhole_tile_compute_dag.cc`
- create `tilelang_repo/src/transform/common/blackhole_tile_compute_patterns.h`
- create `tilelang_repo/src/transform/common/blackhole_tile_compute_patterns.cc`
- modify `lower_blackhole_abi.cc`
- add tests under
  `tilelang_repo/testing/python/transform/`

Work:

0. Precondition:
   Algorithmic Generalization
   `Phase E: Decision-Use Cutover`
   has selected live-form /
   materialization decision-use on the active chain.
   This is typed evidence for legality and materialization;
   it is not a compute expression lowering completion claim.
1. Build `TileComputeDAG`
   from existing explicit tile compute calls
   without changing selection.
2. Add debug/test serialization for node/edge count,
   op names,
   operand roles,
   side-effect tokens.
3. Define pattern schema for current leaf ops.
4. Add static tests that every current production
   `operation_name`
   has a pattern entry.

Implementation notes:

- Added
  `blackhole_tile_compute_dag.{h,cc}`
  with
  `BuildBlackholeTileComputeDAGDiagnostic`
  for read-only node/edge diagnostics over explicit tile compute calls.
- Added
  `blackhole_tile_compute_patterns.{h,cc}`
  with a typed C++ leaf pattern table covering the current TT-Metal
  leaf operation names,
  including
  `matmul_tiles`.
- Tests assert reduce,
  GEMM,
  and flash-attn explicit tile compute calls are represented by the
  diagnostic DAG without changing source emission.

Completion gate:

- no emitted source changes
- existing Blackhole tests unchanged
- DAG dump proves current flash-attn/copy/GEMM
  compute ops are represented

### Phase B: Legalizer Scaffolding

Status: completed in repo HEAD as foundation.
The legalizer is active for current admitted
`TTComputeOpPlan`
validation and synthetic reject diagnostics,
and later phases added mechanics for typed plan recording,
source dispatch,
and
`ValidateTTProgram`
through covering selection.
Those mechanics became production-clean for the known admitted source surface
after the
`2026-04-29`
leaf-normalization repair removed composite pseudo-leaf payloads.
The remaining low-level source hooks are hook targets selected
from pattern metadata; they must be leaf projections only,
not composite lowering owners.
Any hook that records multiple semantic
`TTComputeOpPlan`
entries for one source DAG node is a violation of this phase's contract.

Files:

- create `tilelang_repo/src/transform/common/blackhole_tile_compute_legalizer.h`
- create `tilelang_repo/src/transform/common/blackhole_tile_compute_legalizer.cc`
- modify `lower_blackhole_abi.cc`
- modify `validate_tt_program.cc`
- modify frontend / transform tests

Work:

1. Add `LegalizationAction`.
2. Implement legality checks for:
   fill,
   copy,
   typecast,
   binary max,
   add/mul,
   bcast-cols,
   exp2,
   reduce,
   pack.
3. Return typed reject diagnostics for unsupported
   axis,
   dtype,
   shape,
   or event lifetime.
4. Keep legacy branch emission alive until covering
   owns the same ops.

Implementation notes:

- Added
  `BlackholeTileLegalizationAction`
  with
  `Legal`,
  `Lower`,
  `Split`,
  `PromoteDType`,
  `Materialize`,
  and
  `Reject`
  actions.
- Current scaffold returns
  `Legal`
  or typed
  `Reject`
  diagnostics for the admitted operation set.
  Lower/split/promote/materialize actions are reserved for later production
  migration when a pattern needs them.
- `RecordExactComputeOpPlan`
  and GEMM compute-plan construction call the legalizer before storing
  new compute plans.
- `ValidateTTProgram`
  calls the legalizer again so corrupted /
  synthetic unsupported
  `operation_name`
  values fail closed before projection.

Completion gate:

- legalizer validates all current admitted ops
- unsupported synthetic cases fail closed at selection time

### Phase C: Local DAG Covering For Current Ops

Status: complete for the current admitted explicit tile-compute source
surface after the
`2026-04-29`
boundary repair.
Covering selection gates typed compute-plan recording,
GEMM
`matmul_tiles`
plan construction,
explicit source dispatch,
and
`ValidateTTProgram`
before an operation is accepted.
Pattern metadata carries the selected
`source_emitter`
implementation hook and explicit source dispatch uses that hook.
This field is only a leaf projection hook name;
it is not a semantic owner.
The old operation-name dispatch chain and add/mul operation-name
builtin-selection branch have been deleted from the covered source path.
`TileComputeDAG`
now has a typed pass-local C++ builder,
and DAG covering emits selected pattern IDs,
leaf source hooks,
local-DP state keys,
and costs in dependence order.
Phase E cleanup is now complete for the admitted explicit source surface:
explicit source emission now dispatches through the selected
`source_emitter`
hook registry,
generic reduce source lowering enters the same covering path,
and unsupported standalone explicit-source patterns fail closed after
selection instead of falling through to an old branch-only emitter path.
The known composite payload expansions have been removed from selected hooks,
so the covered path now satisfies the one source node to one semantic leaf
invariant for admitted source calls.
The implementation now represents operation,
result kind,
operand role,
value form,
side-effect class,
and source-emitter kind with typed C++ enums.
`source_emitter`
is optional in the pattern schema,
so patterns without a standalone explicit source path do not register fake
unsupported emitters.
DAG construction and explicit source buffer-argument lookup consume the same
pattern-owned call operand layout.
The implementation keeps the typed schema compact:
enum/string conversion is driven by small lookup tables instead of one switch
per enum family,
and pattern call-operand vectors use direct aggregate initialization rather
than helper wrappers.
The boundary repair tightened the schema so pattern operand layouts describe
true leaf operands for the known active residues:
`exp2_tile`
uses unary
`input`
/
`output`
operands,
`mul_tiles_bcast_cols`
and
`add_tiles_bcast_cols`
use binary
`lhs`
/
`rhs`
/
`output`
operands,
and
`recip_tile`
is admitted as a unary leaf source projection.
Operation-changing
`mode`
/
`kind`
payloads remain forbidden.

Files:

- create `tilelang_repo/src/transform/common/blackhole_tile_compute_covering.h`
- create `tilelang_repo/src/transform/common/blackhole_tile_compute_covering.cc`
- modify `blackhole_tile_compute_patterns.cc`
- modify `lower_blackhole_tile_compute.cc`
- modify `lower_blackhole_abi.cc`
- modify planner tests

Work:

1. Implement bottom-up local DP for tree-shaped
   DAG pieces.
2. Use pattern selected result to build
   `TTComputeOpPlan`.
3. Reuse existing `ExactTileComputeEmitter`
   as low-level source emitter.
4. Migrate fill/copy/typecast first.
5. Migrate binary/broadcast/unary leaf ops.
6. Migrate reduce.

Implementation notes:

- Added
  `blackhole_tile_compute_covering.{h,cc}`
  with
  `SelectBlackholeTileComputeCovering`
  and a diagnostic FFI surface for tests.
- `RecordExactComputeOpPlan`
  and GEMM
  `matmul_tiles`
  plan construction now select a covering pattern before accepting a
  durable
  `TTComputeOpPlan`.
- `LowerExplicitTileComputeCall`
  now selects a covering pattern before dispatching to
  `EmitCoveredBlackholeTileCompute`.
- `BlackholeTileComputePattern`
  now includes
  `source_emitter`
  metadata.
  The covering decision carries that hook,
  and
  `EmitCoveredBlackholeTileCompute`
  dispatches by selected emitter hook instead of operation-name branches.
- The
  `add_tiles`
  and
  `mul_tiles`
  source paths have separate emitter hooks,
  so their TT-Metal builtin selection is no longer recovered from
  `operation_name`
  inside the low-level generator.
- The current selector is a local DAG DP over the Phase A-B pattern table.
  It reuses existing low-level source emitter functions only as named hook
  targets after selected-pattern dispatch;
  the separate inline source-emitter table is deleted by Phase E.
- The selector must not make a composite source decision.
  If a TIR expression requires
  `mul_tiles`
  /
  `add_tiles`
  /
  `exp2_tile`
  as a sequence,
  those must be separate leaf nodes before DAG construction.
- `materialization_policy`
  is selected per pattern and reported by DAG covering.
  Fanout policy is now computed from producer-use edges.

Completion gate:

- existing operation plans are byte-for-byte equivalent
  where practical,
  or structurally equivalent in tests
- no current workload regresses

### Phase D: Fanout And Materialization-Aware Covering

Status: mechanically present in repo HEAD for the admitted compute surface;
not a production-complete DAG claim.
The DAG builder connects producer-use edges using IR object identity
for buffer values;
textual value strings remain diagnostic output only.
The covering diagnostic reports fanout decisions and materialization
policy decisions.
For fanout,
fragment values may be shared,
while
`tile_regs`,
`dst`,
and
`pack`
producers conservatively require
`materialize_before_cross_event_use`
unless a later admitted path proves a stronger lifetime contract.
The stale fallback policy is explicitly
`reject`.

This does not admit multi-block flash-attn direct runtime correctness.
That support surface remains gated by typed unsupported-reason metadata
until the runtime lane proves the wider exact-CB event behavior under
TT-Sim
`bf16`.

Files:

- modify `blackhole_tile_compute_dag.cc`
- modify `blackhole_tile_compute_legalizer.cc`
- modify live-form solver files from the algorithmic
  generalization lane
- modify runtime admission tests

Work:

1. Detect shared producers.
2. Use `LiveValueSSA`
   and event lifetime to decide share vs materialize.
3. Add explicit materialization nodes when needed.
4. Make multi-page exact-CB event conflict reject typed
   until admitted.
5. Use cost model to avoid unnecessary CB republish.

Completion gate:

- multi-block flash-attn source-live-form choice
  is selected by graph/lifetime evidence
- stale fallback source cannot be selected

Repo HEAD partially satisfies this gate for compile/source/spec planning:
source-live-form choice already comes from
`LiveValueSSA`
/
TT live-form solver evidence,
and DAG covering now reports materialization/fanout policy without
selecting stale fallback sources.
This statement is limited to live-form / fanout evidence;
it does not by itself prove production DAG usefulness beyond the typed plans,
resource demand,
and branch deletion it actually drives.
Direct-runtime correctness for multi-block flash-attn remains a later
runtime admission task.

### Phase E: Delete Old Per-Op Selection Branches

Status: architecturally complete for the current admitted explicit source
surface after the
`2026-04-29`
boundary review.
Repo HEAD removed old operation-name dispatch mechanics and then removed the
known pseudo-leaf composite source paths instead of merely routing them through
a hook registry.
Pattern metadata is now the single source of truth for explicit tile-compute
source hook selection.
`EmitCoveredBlackholeTileCompute`
finds a registered
`source_emitter`
hook from the covering decision and invokes that hook directly;
it no longer owns a local vector of per-op lambdas or a
`std::find_if`
operation-name selector.
Generic
`tl.tileop.reduce`
source lowering now enters
`LowerExplicitTileComputeCall`,
selects the
`reduce_tile`
covering pattern,
and then emits through the selected reduce hook.
Pattern entries that are not admitted as standalone explicit
`tl.tileop.blackhole_compute`
source calls must have no production source hook.
Adding a pattern cannot silently bypass the selected-emitter gate,
and a source hook cannot be used to smuggle a composite expression
into production lowering.

Files:

- modify `lower_blackhole_tile_compute.cc`
- modify `lower_blackhole_ops.h`
- modify cleanup/static tests

Work:

1. Remove direct per-op match/emit branches
   that have pattern coverage.
2. Keep only:
   DAG construction,
   legalizer invocation,
   pattern covering,
   selected pattern emission.
3. Delete pseudo-leaf composite source hooks:
   `exp2_tile(mode, lhs, rhs, scale, ...)`,
   `mul_tiles_bcast_cols("div", ...)`,
   and any equivalent operation-changing payload.
4. Add static tests preventing
   duplicate manual selection branches.

Implementation notes:

- Added
  `PlanTTKernelABI::GetTileComputeSourceEmitterHooks`
  and
  `FindTileComputeSourceEmitterHook`
  as the selected source-emitter registry.
- Refactored
  `BlackholeTileComputePattern`
  from string metadata into typed C++ enums for operation,
  result kind,
  operand role,
  value form,
  side-effect class,
  and optional source-emitter kind.
- Replaced per-enum
  `ToString`
  switch boilerplate with compact typed lookup tables,
  and removed the call-operand vector wrapper helper in favor of direct
  aggregate initialization in the pattern table.
- Added pattern-owned call operand layouts for
  `tl.tileop.blackhole_compute`
  and generic tile-op source calls;
  the DAG builder and explicit source buffer-argument helpers now read those
  layouts instead of duplicating operation-name / argument-index branches.
- Moved the old inline explicit-source lambda bodies into named hook methods
  such as
  `EmitFillFragmentTileComputeSource`,
  `EmitCopyTileComputeSource`,
  `EmitTypecastTileComputeSource`,
  `EmitAddTilesComputeSource`,
  `EmitMulTilesComputeSource`,
  `EmitAddTilesBcastColsComputeSource`,
  `EmitExp2TileComputeSource`,
  `EmitRecipTileComputeSource`,
  and
  `EmitReduceTileComputeSource`.
- Follow-up boundary repair deleted the old
  `GetBlackholeTileComputeStringArg`
  helper,
  deleted the composite
  `GenerateExp2TileLeafDAGSequence`
  and string-mode
  `GenerateMulTilesBcastColsSequence`
  paths,
  made
  `EmitExp2TileComputeSource`
  a unary leaf projection,
  made
  `EmitMulTilesBcastColsComputeSource`
  a true broadcast binary leaf projection,
  and added one-to-one
  `add_tiles_bcast_cols`
  /
  `recip_tile`
  leaf hooks.
- Replaced the direct
  `MatchExplicitTileReduce`
  /
  `GenerateRowReductionSequence`
  branch in
  `lower_blackhole_ops.cc`
  with
  `LowerExplicitTileComputeCall`,
  so reduce source emission is covered by the same pattern-selection gate.
- Added static tests that require every pattern-table
  `source_emitter`
  to have a registered hook,
  require typed enum / optional-emitter schema fields,
  require compact enum/string lookup tables and no call-operand wrapper helper,
  require DAG construction to use pattern-owned operand layouts,
  forbid the old inline emitter table /
  `std::find_if`
  dispatch in
  `lower_blackhole_tile_compute.cc`,
  and forbid the direct explicit-reduce emission branch in
  `lower_blackhole_ops.cc`.

Completion gate:

- no old branch-only path remains for admitted ops
- no leaf-looking source call carries composite semantics
- one DAG source node records at most one semantic
  `TTComputeOpPlan.operation_name`,
  and that operation matches the selected leaf pattern
- expression decomposition happens before DAG construction as explicit
  `Normalized Tile TIR`
  leaf sequence
- adding a new leaf op requires adding:
  pattern schema,
  legality predicate,
  tests,
  and a leaf source hook when a standalone explicit source path is admitted

Repo HEAD satisfies this gate for the known admitted explicit source surface.
Future compute admission must keep the same gate:
if a source node needs several semantic leaf ops,
the frontend normalizer must emit several explicit leaf statements before DAG
construction.

## Cost Model

第一版 cost model 应保持简单：

```text
base_instruction_cost
+ cb_page_cost
+ materialization_cost
+ dtype_conversion_cost
+ event_lifetime_cost
+ unsupported_penalty
```

Rules:

- semantic unsupported is never selected
- missing source-live-form proof is
  `admission_blocked`,
  not high cost
- missing expression normalization is
  `lowering_missing`,
  not semantic unsupported
- missing TT-Metal wrapper/codegen coverage is
  `backend_op_missing`,
  not semantic unsupported
- exact-CB reuse is cheaper than republish only if
  lifetime proof exists
- fewer materialization boundaries wins
  only when validator accepts the same semantics

## Validator Requirements

- every selected pattern must produce a supported
  `TTComputeOpPlan.operation_name`
- every operand role in `TTComputeOpPlan`
  must come from a DAG edge
  or explicit region operand
- every materialization inserted by covering
  must have a corresponding
  `MaterializationBoundary`
  or TT-local proof that it does not cross stage
- every source sequence must be traceable to selected
  pattern IDs
- every DAG-driven source hook must project exactly one semantic leaf op;
  if a source node needs several leaf ops,
  those leaf ops must already be explicit TIR nodes before DAG construction
- source/codegen/runtime must not infer operation
  family from source text

## Testing Strategy

### Static tests

- pattern table covers all allowed leaf op names
- no composite helper names appear in
  `TTComputeOpPlan.operation_name`
- no composite helper semantics appear behind leaf-looking
  `tl.tileop.blackhole_compute`
  payloads
- admitted ops do not bypass legalizer
- removed per-op branches do not reappear

### Structure tests

- dump / inspect DAG for copy,
  GEMM,
  reduce,
  flash-attn.
- assert operand role edges for
  binary,
  bcast-cols,
  reduce,
  pack.
- assert selected pattern IDs and costs.

### Planner tests

- `TTComputeOpPlan`
  matches selected pattern operation.
- `TTCBPlan`
  requirements match pattern input/output forms.
- materialization-aware covering inserts typed
  materialization plans when needed.

### Runtime tests

- current bf16 copy/GEMM/flash-attn gates remain green.
- multi-block flash-attn is admitted only after
  source-live-form and event lifetime are selected
  through DAG/lifetime evidence.

## Payoff

- Adding a new TT-Metal tile compute leaf op becomes
  pattern + legality + tests,
  not a new branch family.
- Complex fused compute becomes DAG covering
  over explicit dependencies,
  not workload-specific matcher chains.
- Legalization explains why a shape is unsupported
  before source/runtime emission.
- Cost model gives a place to improve quality
  without weakening IR contracts.
