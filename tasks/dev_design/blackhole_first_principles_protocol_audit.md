# Blackhole First-Principles Protocol Audit

> 本文档不是新的总体设计。
>
> 它只做一件事：
> **基于第一性原理，对现存 surface 做 owner / non-owner / validator / cutover disposition。**
>
> 下表里出现的现存 surface 名，全部按当前仓库里的历史字面名列出，
> 目的只有一个：做删除与切换清单。
> 它们不是当前允许继续扩展的协议名。

## 1. 判定标准

对 spatial/dataflow target，
长期协议只能归到下面四层之一：

1. `Normalized Tile TIR`
2. `SpatialPlan`
3. `TTProgram`
4. `ExecutableSpec`

如果某个 surface 不能稳定归到这四层之一，
它就不是长期 owner truth。

## 2. Owner Disposition Table

| 现存 surface | 长期 owner | 非 owner 结论 | validator obligation | 去留 |
|---|---|---|---|---|
| `blackhole.copy_semantics` | `Normalized Tile TIR` access truth + `SpatialPlan.DataflowEdge` + `TTTransportPlan` | 不能继续当方向/角色真源 | `ValidateSpatialPlan` 检查 edge completeness；`ValidateTTProgram` 检查 transport realization | 过渡保留；仅 lowering-time internal marker |
| `blackhole.segment_kind` | `TTKernelPlan.role` | 不应再写回 TIR attr | `ValidateTTProgram` 检查 kernel role / ABI / transport 闭合 | 过渡保留；仅 lowering-time internal marker |
| `blackhole.work_decomposition` | `TTBlockPlan` + `TTExecutionPlan` | 不属于 `SpatialPlan` 公开协议 | `ValidateTTProgram` 检查 work ownership / placement / wave legality | 主链删除；debug analysis helper 保留 |
| `blackhole.compute_regions` | `Normalized Tile TIR` anchored sub-TIR + `SpatialPlan.ExecutionUnit` | 不能继续作为 compute truth bag | `ValidateSpatialPlan` 检查 execution-unit coverage | 主链删除；debug analysis helper 保留 |
| `blackhole.pipeline_stages` | `SpatialPlan.PhasePlan` + `TTSyncPlan` | 不能继续作为跨层 bag | `ValidateSpatialPlan` 检查 phase/order；`ValidateTTProgram` 检查 completion/materialization | 主链删除；debug analysis helper 保留 |
| `blackhole.lowering_requirements` | 拆回 analyses + `TTKernelPlan / TTTransportPlan / TTSyncPlan / TTABIPlan / TTExecutionPlan` | 不是长期公共协议 | 三层 validator 分别检查 completeness；leaf 禁止读取 | public attr 删除；internal builder/projection 保留 |
| `blackhole.resource_plan` | `TTTransportPlan` / `TTSyncPlan` / `TTExecutionPlan` | 是 canonicalization 时代的影子产物 | `ValidateTTProgram` 直接验证 typed target truth | 删除 |
| `tl.internal_tt_*` | `TTProgram` typed owner objects | 只能短期 bridge，不是正式协议 | `ValidateTTProgram` 只接受 typed owner object，不接受 internal attr bag | 删除 |
| `TTProgram.payload` 大袋子 | `TTProgram` typed owner objects | 只能保留 leaf projection 级别 payload | `ValidateTTProgram` 禁止 payload 反客为主 | 已收紧 |
| `ExecutableSpec` 中的 raw payload | `ExecutableSpec` leaf projection | 只能是投影结果，不能反向变成 planning source | `ValidateExecutableSpecProjection` | 已收紧 |

## 3. Repo HEAD 落地说明

`Legacy Protocol Deletion`
在 repo HEAD 的含义已经固定为：

- canonical `LowerToBlackholePhaseB`
  不再发布
  `blackhole.work_decomposition /
   blackhole.compute_regions /
   blackhole.pipeline_stages`
- `PlanTTBlocks -> PlanTTExecution`
  直接增量写 staged
  `tl.tt_program`；
  `BuildTTProgram`
  不再接受
  `tl.internal_tt_*`
  当过渡 owner bag
- leaf / resource path
  不再发布
  `blackhole.lowering_requirements`
  与
  `blackhole.resource_plan`
- 为了让 optimized/helper 入口
  还能把 pre-opt logical tile bridge spec
  对齐回 optimized device func，
  repo HEAD 允许一个窄 internal bridge attr：
  `tl.blackhole_logical_buffer_tile_bridge_specs`
  它只承接
  `buffer_tile_bridge_specs`
  这一个 leaf-local bridge surface，
  不重新引入整袋
  `blackhole.compute_regions`
- `AnalyzeBlackholeWorkDecomposition /
   AnalyzeBlackholeComputeRegions /
   AnalyzeBlackholePipelineStages`
  这些 pass wrapper
  当前仍保留为 debug / regression helper；
  但它们不再属于 canonical bundle，
  也不允许重新变成 pass-to-pass protocol

## 4. 长期保留的 truth

### `Normalized Tile TIR`

长期保留：

- tile op
- `BufferLoad / BufferStore`
- address expr
- region / predicate / loop/domain
- loop-carried / dataflow structure

### `SpatialPlan`

长期保留：

- `ExecutionUnit`
- `DataflowEdge`
- `LayoutSpec`
- `PhasePlan`
- `ValidatedHintSet`

### `TTProgram`

长期保留：

- `TTBlockPlan`
- `TTKernelPlan`
- `TTTransportPlan`
- `TTSyncPlan`
- `TTABIPlan`
- `TTExecutionPlan`

### `ExecutableSpec`

长期保留：

- 只读 leaf projection
- runtime/build/codegen 消费视图

## 5. 当前代码判断

当前代码的真实问题不是“旧协议太散”，
而是：

- 中间 spatial/dataflow owner layer 太薄
- 后段被迫补出 fake protocol
- leaf readers 仍在消费 fake protocol

因此 disposition 的执行顺序固定为：

1. 先把 `SpatialPlan`
   重写成 virtual spatial/dataflow program
2. 再把 `TTProgram`
   收回 target owner
3. 再切 leaf readers
4. 最后删 fake protocol
