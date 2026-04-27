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
tile compute semantics
  -> TileComputeDAG
  -> legalization
  -> canonical DAG
  -> target leaf pattern covering
  -> TTComputeOpPlan / source sequence
```

它不是新的长期 IR 层。
`TileComputeDAG`
是 `Normalized Tile TIR`
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

## Non-Goals

- 不恢复 downstream scalar-loop matcher。
- 不引入 composite operation name：
  `softmax`,
  `exp2_affine`,
  `row_broadcast_exp2_affine`
  等不能进入生产 compute protocol。
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

后续复杂模式需要更通用的选择机制：

- multi-step epilogue
- fused reduce + broadcast + binary
- multiple legal lowering choices
- exact-CB event lifetime-sensitive choice
- dtype / accumulator / fragment form choice
- pack / tilize / untilize placement

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

5. Lowering is an explicit information trade.
   DAG covering may freeze a target leaf sequence,
   operand forms,
   and materialization points,
   but it must preserve enough typed evidence for validators,
   source projection,
   and runtime admission to audit the decision.
   If event lifetime or live-form source cannot be proven,
   the candidate is unsupported,
   not merely expensive.

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

## Implementation Plan

### Phase A: Pattern Schema And Read-Only DAG Dump

Files:

- create `tilelang_repo/src/transform/common/blackhole_tile_compute_dag.h`
- create `tilelang_repo/src/transform/common/blackhole_tile_compute_dag.cc`
- create `tilelang_repo/src/transform/common/blackhole_tile_compute_patterns.h`
- create `tilelang_repo/src/transform/common/blackhole_tile_compute_patterns.cc`
- modify `lower_blackhole_tile_compute.cc`
- add tests under
  `tilelang_repo/testing/python/transform/`

Work:

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

Completion gate:

- no emitted source changes
- existing Blackhole tests unchanged
- DAG dump proves current flash-attn/copy/GEMM
  compute ops are represented

### Phase B: Legalizer Scaffolding

Files:

- create `tilelang_repo/src/transform/common/blackhole_tile_compute_legalizer.h`
- create `tilelang_repo/src/transform/common/blackhole_tile_compute_legalizer.cc`
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

Completion gate:

- legalizer validates all current admitted ops
- unsupported synthetic cases fail closed at selection time

### Phase C: Local DAG Covering For Current Ops

Files:

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
5. Migrate binary/broadcast/exp2.
6. Migrate reduce.

Completion gate:

- existing operation plans are byte-for-byte equivalent
  where practical,
  or structurally equivalent in tests
- no current workload regresses

### Phase D: Fanout And Materialization-Aware Covering

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

### Phase E: Delete Old Per-Op Selection Branches

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
3. Add static tests preventing
   duplicate manual selection branches.

Completion gate:

- no old branch-only path remains for admitted ops
- adding a new leaf op requires adding:
  pattern schema,
  legality predicate,
  tests,
  and source emitter hook

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

- unsupported is never selected
- missing source-live-form proof is unsupported,
  not high cost
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
- source/codegen/runtime must not infer operation
  family from source text

## Testing Strategy

### Static tests

- pattern table covers all allowed leaf op names
- no composite helper names appear in
  `TTComputeOpPlan.operation_name`
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
