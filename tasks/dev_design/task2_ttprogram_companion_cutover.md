# Task 2: TTProgram Companion Cutover

## 基本信息

- **文档角色**: `Task 2` 的 target owner cutover 设计文档
- **当前状态**: `2026-04-16` 活动设计文档
- **任务链位置**: `Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec`
- **唯一总体设计**: `tasks/dev_design/final_blackhole_backend_redesign.md`

## 1. 目标

`Task 2` 只负责一件事：

> **让 `TTProgram` 成为唯一 physical realization truth。**

它必须回答：

- target planner 允许读取哪些上游 truth
- `TTProgram`
  长期保留哪些 owner object
- 哪些 pass 承担 target planning
- `ExecutableSpec`
  与 build/codegen/runtime
  的 reader/writer 边界是什么

## 2. 合法输入

`TT` planner 只允许读取：

- `SpatialPlan`
  的
  `ExecutionUnit / DataflowEdge / LayoutSpec / PhasePlan / ValidatedHintSet`
- anchored sub-TIR
- `TTHardwareModel`
- validate 后的 target hints

不允许回升为 target owner 输入的东西：

- 审计表列出的 legacy transition attrs / helper bridge
- runtime/codegen 自己恢复出来的结论

## 3. 长期 owner object set

`TTProgram` 长期 primary owner object
只保留下面这些 plan：

- `TTBlockPlan`
- `TTKernelPlan`
- `TTTransportPlan`
- `TTSyncPlan`
- `TTABIPlan`
- `TTExecutionPlan`

`TTProgram`
只是这组 object 的稳定聚合结果，
不是第二真源。

当前代码中的：

- `TTKernel`
- `TTCoreGroup`
- `TTCBPlan`
- `TTSemaphorePlan`
- `TTComputeSyncPlan`
- `TTDstLayoutPlan`

只能作为：

- 兼容载体
- realization detail
- leaf projection data

## 4. Pass owner 主链

长期 canonical pass chain 固定为：

```text
AnalyzeSpatialStructureFacts
  -> BuildSpatialPlanCompanion
  -> ValidateSpatialPlan
  -> PlanTTBlocks
  -> PlanTTCompute
  -> PlanTTTransport
  -> PlanTTSync
  -> PlanTTABI
  -> PlanTTExecution
  -> BuildTTProgram
  -> ValidateTTProgram
  -> MaterializeBlackholeExecutable
```

各 pass 职责：

- `PlanTTBlocks`
  - 决定 target-side work ownership / placement / slicing
- `PlanTTCompute`
  - 从 tile op / layout / region
    做 compute family 选择
- `PlanTTTransport`
  - 从 TIR access truth + `DataflowEdge + LayoutSpec + TTBlockPlan`
    做 transport / route / delivery realization
- `PlanTTSync`
  - 负责 completion / ordering / visibility
- `PlanTTABI`
  - 形成 compile/common/per-work ABI
- `PlanTTExecution`
  - 形成 launch order / waves / kernel-to-core binding
- `BuildTTProgram`
  - 只聚合上面的 typed truth
- `ValidateTTProgram`
  - 检查 typed owner object completeness / consistency

`BuildTTProgram`
不再允许承担任何 planning owner 责任。

## 5. Reader / Writer 边界

### `TTProgram`

是唯一 target truth。

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

## 6. Legacy bridge 的退场规则

审计表列出的
legacy helper bridge / runtime-arg bridge / internal payload
都不再保留 owner 身份。

过渡期如果仍保留，
也只能是短命 bridge，
不能被 build/codegen/runtime 当主真源消费。

## 7. 当前执行重点

`Task 2` 当前重点固定为：

1. 让 `BuildTTProgram`
   退成纯聚合器
2. 显式拉出
   `PlanTTSync / PlanTTABI / PlanTTExecution`
3. 停止让 leaf readers
   依赖 legacy gate attrs
4. 让 `TTProgram`
   越来越靠近 TT-Metal host truth
   (`ProgramDescriptor` 风格)
