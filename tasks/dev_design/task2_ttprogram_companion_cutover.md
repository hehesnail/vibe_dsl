# Task 2: TTProgram Representation Cutover

## 基本信息

- **文档角色**: `TTProgram` 表示层合同文档
- **任务链位置**: `Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec`
- **唯一总体设计**: `tasks/dev_design/final_blackhole_backend_redesign.md`

说明：

- 文件名中的 `companion`
  只是历史文件名，
  不是新的 IR 层命名
- 本文档只定义目标合同和完成判据
- 当前 repo HEAD 状态统一只看 `tasks/progress.md`

## 1. 目标

`Task 2` 只负责一件事：

> **让 `TTProgram` 成为唯一的 target realization representation。**

它必须回答：

- target planning 允许读取哪些上游显式表示
- `TTProgram`
  长期保留哪些显式 slice
- `ExecutableSpec`
  与 build/codegen/runtime
  的 reader/writer 边界是什么

## 2. 合法输入与禁止输入

target planning 只允许读取：

- `SpatialPlan`
  的
  `ExecutionUnit / DataflowEdge / LayoutSpec / PhasePlan / ValidatedHintSet`
- anchored sub-TIR
- `TTHardwareModel`
- validate 后的 target hints

不允许回升为 target planning 输入的东西：

- 审计表列出的 legacy transition attrs / helper bridge
- runtime/codegen 自己恢复出来的结论
- leaf projection 或 leaf payload 倒灌回来的语义

## 3. `TTProgram` 的显式 slice

`TTProgram`
长期只保留下面这些显式 plan：

- `TTBlockPlan`
- `TTKernelPlan`
- `TTTransportPlan`
- `TTSyncPlan`
- `TTABIPlan`
- `TTExecutionPlan`

这组对象合起来定义 target realization。

当前代码中的：

- `TTKernel`
- `TTCoreGroup`
- `TTCBPlan`
- `TTSemaphorePlan`
- `TTComputeSyncPlan`
- `TTDstLayoutPlan`

只能作为：

- compatibility carrier
- realization detail
- leaf projection data

它们不能反向定义 `TTProgram`
的长期表示合同。

## 4. Construction 边界

`TTProgram`
必须由：

- 当前 `SpatialPlan`
- anchored sub-TIR
- `TTHardwareModel`

直接 lower / build 出来。

实现上可以把 target planning
拆成多个 planner passes，
例如：

- block planning
- compute planning
- transport planning
- sync planning
- ABI planning
- execution planning

但这些 pass 名字只是实现细节，
不是架构边界。

架构边界只有一条：

- target planning
  必须把当前输入表示
  直接写成
  `TTProgram`
  的显式 slice

如果实现上仍保留局部 collector，
它也只能留在同一个 planner `.cc`
里作为 mechanics，
不能长成新的 bridge bag。

补充说明：

- 如果当前实现仍保留
  `BuildTTProgram`
  这个入口，
  它的长期角色只能是聚合已有 slice，
  不能再承担 planning 语义

## 5. Validator 合同

`ValidateTTProgram`
必须检查：

1. `TTBlockPlan / TTKernelPlan / TTTransportPlan /
   TTSyncPlan / TTABIPlan / TTExecutionPlan`
   的 completeness / consistency
2. exact TT-Metal builtin / transport / sync legality
3. transport / sync / ABI / execution 闭合
4. payload / compatibility carrier
   没有反客为主，
   变成 planning 语义来源

validator 失败时必须 fail-closed。

## 6. Reader / Writer 边界

### `TTProgram`

是唯一 target realization 表示。

### `MaterializeBlackholeExecutable`

是唯一 leaf writer。

它只允许读取：

- `TTProgram`
- `TTHardwareModel`
- 必要的 target-local legalization result

### build / codegen / runtime / `BlackholeModule`

只允许读取：

- `tl.blackhole_executable`
- 或其内部 `ExecutableSpec` 投影

不允许再读取：

- 审计表列出的 legacy gate attrs / transition attrs / internal bridge payload
- 任何 target planning residue

## 7. Legacy bridge 的退场规则

审计表列出的
legacy helper bridge / runtime-arg bridge / internal payload
都不再是 `TTProgram`
的长期语义来源。

过渡期如果仍保留，
也只能是短命 bridge，
不能被 build/codegen/runtime 当主表示消费。

## 8. 当前执行重点

`Task 2` 当前重点固定为：

1. 把 target planning 的结果
   直接写入
   `TTProgram`
   显式 slice
2. 让 `BuildTTProgram`
   退成纯聚合器
3. 停止让 leaf readers
   依赖 legacy gate attrs
4. 让 `TTProgram`
   越来越靠近 TT-Metal host-facing target representation

补充约束：

- `Task 2`
  不得在 `Task 1`
  未站稳之前提前扩 workload payoff
- 如果 target planning 发现上游语义不够，
  结论只能是回到 `Task 1`
  补 `SpatialPlan`
  的显式对象或 validator，
  不能在 `TTProgram`
  再补一层 target-independent bridge
- 如果当前 target-side implementation
  和表示层边界冲突，
  优先改 target-side implementation；
  不允许先把现有 planner / codegen 习惯
  固化成上游必须适配的前提

## 9. Completion Contract

`Task 2`
只有在下面这些条件同时满足后才算完成：

1. target planning
   直接构造
   `TTBlockPlan / TTKernelPlan / TTTransportPlan /
    TTSyncPlan / TTABIPlan / TTExecutionPlan`
   这组显式 slice
2. `BuildTTProgram`
   已退成纯聚合器
3. `ValidateTTProgram`
   已对这组显式 slice 提供正式 gate
4. `TTProgram`
   已经能以这组显式 slice
   充当唯一 target realization representation
5. leaf writer / leaf readers
   不再反向依赖 legacy planning residue
