# Blackhole First-Principles Protocol Audit

> 本文档不是新的总体设计。
>
> 它只做一件事：
> **基于第一性原理，对现存 surface 做 owner / non-owner / validator / cutover disposition。**

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
| `blackhole.copy_semantics` | `Normalized Tile TIR` access truth + `SpatialPlan.DataflowEdge` + `TTTransportPlan` | 不能继续当方向/角色真源 | `ValidateSpatialPlan` 检查 edge completeness；`ValidateTTProgram` 检查 transport realization | 删除 |
| `blackhole.segment_kind` | `TTKernelPlan.role` | 不应再写回 TIR attr | `ValidateTTProgram` 检查 kernel role / ABI / transport 闭合 | 删除 |
| `blackhole.work_decomposition` | `TTBlockPlan` + `TTExecutionPlan` | 不属于 `SpatialPlan` 公开协议 | `ValidateTTProgram` 检查 work ownership / placement / wave legality | 删除 |
| `blackhole.compute_regions` | `Normalized Tile TIR` anchored sub-TIR + `SpatialPlan.ExecutionUnit` | 不能继续作为 compute truth bag | `ValidateSpatialPlan` 检查 execution-unit coverage | 删除 |
| `blackhole.pipeline_stages` | `SpatialPlan.PhasePlan` + `TTSyncPlan` | 不能继续作为跨层 bag | `ValidateSpatialPlan` 检查 phase/order；`ValidateTTProgram` 检查 completion/materialization | 删除 |
| `blackhole.lowering_requirements` | 拆回 analyses + `TTKernelPlan / TTTransportPlan / TTSyncPlan / TTABIPlan / TTExecutionPlan` | 不是长期公共协议 | 三层 validator 分别检查 completeness；leaf 禁止读取 | 整体删除 |
| `blackhole.resource_plan` | `TTTransportPlan` / `TTSyncPlan` / `TTExecutionPlan` | 是 canonicalization 时代的影子产物 | `ValidateTTProgram` 直接验证 typed target truth | 删除 |
| `tl.internal_tt_*` | `TTProgram` typed owner objects | 只能短期 bridge，不是正式协议 | `ValidateTTProgram` 只接受 typed owner object，不接受 internal attr bag | 删除 |
| `TTProgram.payload` 大袋子 | `TTProgram` typed owner objects | 只能保留 leaf projection 级别 payload | `ValidateTTProgram` 禁止 payload 反客为主 | 收紧 |
| `ExecutableSpec` 中的 raw payload | `ExecutableSpec` leaf projection | 只能是投影结果，不能反向变成 planning source | `ValidateExecutableSpecProjection` | 收紧 |

## 3. 长期保留的 truth

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

## 4. 当前代码判断

当前代码的真实问题不是“旧协议太散”，
而是：

- 中间 virtual layer 太薄
- 后段被迫补出 fake protocol
- leaf readers 仍在消费 fake protocol

因此 disposition 的执行顺序固定为：

1. 先把 `SpatialPlan`
   重写成 virtual spatial/dataflow program
2. 再把 `TTProgram`
   收回 target owner
3. 再切 leaf readers
4. 最后删 fake protocol
