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
长期 truth 只能归到下面四层之一：

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
| `blackhole.segment_kind` | `TTKernelPlan.kind` / `TTKernel.kind` | 不应再写回 TIR attr | `ValidateTTProgram` 检查 reader/compute/writer kernel kind、ABI、transport 闭合 | 删除 |
| `blackhole.work_decomposition` | `TTBlockPlan` + `TTExecutionPlan` | 不属于 `SpatialPlan` 公开协议 | `ValidateTTProgram` 检查 work ownership / placement / wave legality | 删除 |
| `blackhole.compute_regions` | `Normalized Tile TIR` anchored sub-TIR + `SpatialPlan.ExecutionUnit` | 不能继续作为 compute truth bag | `ValidateSpatialPlan` 检查 execution-unit coverage | 删除 |
| `blackhole.pipeline_stages` | `SpatialPlan.PhasePlan` + `TTSyncPlan` | 不能继续作为跨层 bag | `ValidateSpatialPlan` 检查 phase/order；`ValidateTTProgram` 检查 completion/materialization | 删除 |
| `blackhole.lowering_requirements` | 拆回当前 pass 的直接 IR/owner 读取 + `TTKernelPlan / TTTransportPlan / TTSyncPlan / TTABIPlan / TTExecutionPlan / ExecutableSpec` | 不是长期公共协议，也不应继续保留 internal builder bag | 三层 validator 分别检查 completeness；leaf 禁止反向依赖 planning bag | 删除 |
| `blackhole.resource_plan` | `TTTransportPlan` / `TTSyncPlan` / `TTExecutionPlan` | 是 canonicalization 时代的影子产物 | `ValidateTTProgram` 直接验证 canonical target truth | 删除 |
| `tl.internal_tt_*` | `TTProgram` canonical owner objects | 只能短期 bridge，不是正式协议 | `ValidateTTProgram` 只接受 canonical owner objects，不接受 internal attr bag | 删除 |
| `TTProgram.payload` 大袋子 | `TTProgram` canonical owner objects + leaf projection payload | 只能保留 leaf 投影级 payload，不能反向充当 planning 真源 | `ValidateTTProgram` 禁止 payload 反客为主 | 已收紧 |
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
  只允许一个窄 internal bridge attr：
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
  与对应的 internal evidence bag helper
  都应从 active chain 删除，
  不能继续以 debug / regression helper 名义常驻

### 3.1 `2026-04-17` Task 0 落地补充

- `SelectBlackholeTTMetalBuiltins`
  现在已经位于
  `PlanTTBlocks`
  与
  `PlanTTCompute`
  之间，
  compute-side exact builtin 选择前移到 planner helper 路线之前
- `compute_epilogue_ops`
  不再属于 repo HEAD 的 active protocol：
  它已从
  `TTProgram.payload`、
  executable projection、
  codegen、
  runtime
  和测试基线移除
- 旧 helper/composite builtin 名称
  不再是 active IR surface；
  selector / validator
  会按 exact op 名 fail-closed 拒绝 residue
- 当前唯一和 Task 0 直接相关、仍暂时保留的过渡面是
  `tl.blackhole_lowering_requirements_seed`：
  它只承接
  `buffer_materialization_contracts`
  与
  `buffer_tile_bridge_specs`
  这两个稳定 seed，
  供 selector-forwarding 跨 rewrite 保持桥接事实，
  并在最终
  `TTProgram`
  物化前剥离；
  它不是新的 planning truth

## 4. 长期保留的 truth 与 pass 纪律

补充约束：

- 只允许 owner layer 定义长期协议面
- IR 不是 read-only 观察对象；pass 的职责是把当前 stage 的 IR/object 改写到下一个更具体 stage
- helper 只允许作为同一 `.cc` 内的局部 visitor / matcher / mutator mechanics
- 如果一个 pass 能从当前 IR 或当前 owner object 直接恢复所需信息，就必须直接恢复，不能先发明 bag 再读 bag
- helper 必须复用已有 owner enum / handle / object identity
- 不能在 helper 里重新发明一套 `kind / direction / role` 字符串或平行 enum
- 不能用 `Map<String, Any>` 充当跨 pass 语义协议

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

- leaf projection
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
