# Task 2: TTProgram Companion Cutover

## 基本信息

- **文档角色**: `Task 2` 的 target owner cutover 设计文档
- **当前状态**: `2026-04-13` 活动设计文档；按 `tasks/progress.md`，
  当前 `Task 2` 仍未开始
- **任务链位置**:
  `Normalized Tile TIR -> SpatialPlan companion -> TTProgram companion ->
  ExecutableSpec` 中第二层 companion 的 owner 设计
- **目标**:
  让 `TTProgram companion` 成为唯一 target truth，
  让 `MaterializeBlackholeExecutable` 成为唯一 writer
- **非目标**:
  - 不在 `Task 2` 里兑现 runtime/correctness payoff
  - 不在 `Task 2` 里承接 workload family 回归
  - 不把 `TTProgram companion` 退化成 payload-backed attr bag
- **唯一总体设计**: `tasks/dev_design/final_blackhole_backend_redesign.md`
- **上游设计输入**:
  - `tasks/dev_design/task0_ir_layering_root_cause.md`
  - `tasks/dev_design/task1_spatial_plan_companion.md`

## 1. 作用域

`Task 2` 只负责一件事：

- 把 target owner 链从旧 recovery/materialization 主链切到
  `TTProgram companion` 主链

它必须回答：

- `TT` planner 允许读取哪些 owner truth
- target 侧长期只保留哪些 plan object
- 哪些 pass 承担 target planning owner
- `TTProgram` 与 `ExecutableSpec` 的 writer / reader 边界是什么
- 旧 `LowerBlackholeOps / PlanBlackholeCB / AssignBlackholeCores /
  LowerSpatialProgramToTTTarget / MaterializeTTExecutableSpec`
  分别如何退场或降级

它不负责：

- 重新定义 `SpatialPlan companion`
- 在 target 侧补 non-TT-specific semantic recovery
- 处理 `Task 3` 的 runtime gate、support surface 和 workload family 回归

## 2. 合法输入边界

`TT` planner 只允许消费下面四类输入：

- `SpatialPlan companion`
  的最小 planning truth：
  `ExecutionClosure / ClosureBoundary / ValidatedHintSet`
- closure 所锚定的 sub-TIR
- `ValidatedHintSet` 中已经验证成功的 target hints
- `TTHardwareModel`

不允许回升为 target owner 输入的东西：

- `SemanticProgram`
- 旧 `SpatialProgram` payload
- `tl.tt_*` seed attrs
- `LowerBlackholeOps` 的 matcher 结果
- `blackhole.*` bridge attrs
- runtime / codegen 自己再补出来的 planning 结论

如果仅靠合法输入还不能决定 target mapping，
结论只能是：

- 上游 `TIR/DSL` 还缺显式表达
- 更早的 analysis 还不够
- 当前 case 必须显式 `unsupported`

不允许第四种动作：

- 在 target 侧继续补 semantic recovery
- 再造一层 payload bag 或 seed attrs 充当主协议

## 3. `TTProgram companion` 的最小 owner object set

`Task 2` 完成后，长期 primary target owner object
只保留下面这些 plan：

- `TTBlockPlan`
- `TTKernelPlan`
- `TTTransportPlan`
- `TTSyncPlan`
- `TTABIPlan`
- `TTExecutionPlan`

`TTProgram` 是这组 owner object 的稳定聚合结果，
不是与它们并列的第二层真源。

对象边界：

- `TTBlockPlan`
  - owner：target-side block sizing、slice、buffering、
    decomposition truth
- `TTKernelPlan`
  - owner：kernel grouping、kernel role、core-type-facing execution view
- `TTTransportPlan`
  - owner：boundary 的 target transport realization
- `TTSyncPlan`
  - owner：transport / completion / ordering 的 target sync realization
- `TTABIPlan`
  - owner：compile-time / common-runtime / per-work ABI
- `TTExecutionPlan`
  - owner：launch order、wave、kernel-to-execution binding
- `TTProgram`
  - owner：以上 plan object 的稳定聚合结果

下面这些不再作为长期 primary owner object 单独存在：

- `TTCoreGroup`
- `TTCBPlan`
- `TTSemaphorePlan`
- `TTComputeSyncPlan`
- `TTDstLayoutPlan`
- `blackhole.segment_plan`
- `blackhole.runtime_args`
- `blackhole.common_runtime_args`

它们只能作为：

- plan realization detail
- leaf materialization result
- compatibility projection

## 4. Pass owner 主链

`Task 2` 的 canonical pass chain 固定为：

```text
AnalyzeSpatialStructureFacts
  -> BuildSpatialPlanCompanion
  -> PlanTTBlocks
  -> PlanTTTransport
  -> PlanTTSync
  -> PlanTTABI
  -> PlanTTExecution
  -> BuildTTProgram
  -> ValidateTTProgram
  -> MaterializeBlackholeExecutable
```

各 pass 的 owner 职责：

- `PlanTTBlocks`
  - 决定 target-side decomposition truth
  - 不允许 `work_linear_id`、case-local shape、
    payload bag 继续兼职 block owner
- `PlanTTTransport`
  - 根据 `ClosureBoundary + TTBlockPlan + anchored sub-TIR`
    做 transport realization
  - 不再从 builtin 排列或 matcher 恢复 transport intent
- `PlanTTSync`
  - 根据 dependency / boundary / transport result
    做 target sync realization
- `PlanTTABI`
  - 把 owner-side access truth 物化成 target ABI
  - 不再从 lowered loop、arg kind、runtime heuristics 反推
- `PlanTTExecution`
  - 形成 kernel grouping / launch order / wave scheduling
- `BuildTTProgram`
  - 聚合 target owner plan
  - 不重新恢复 target truth
- `ValidateTTProgram`
  - 检查 completeness / consistency
  - 禁止 legacy attrs、seed attrs、payload bag 回升为主协议
- `MaterializeBlackholeExecutable`
  - 只消费经过验证的 `TTProgram`
  - 物化最终 `ExecutableSpec`

## 5. `ExecutableSpec` 的物化边界

`ExecutableSpec` 的地位固定为：

- `TTProgram` 是唯一 target truth
- `ExecutableSpec` 只是面向执行侧的冻结物化结果
- runtime / codegen / `BlackholeModule`
  只消费 `ExecutableSpec`

唯一合法 writer：

- `MaterializeBlackholeExecutable`

它只允许读取：

- `TTProgram`
- `TTHardwareModel`
- 必要的 target-local legalization 结果

它不允许再读取：

- `SemanticProgram`
- 旧 `SpatialProgram`
- `tl.tt_*` seed attrs
- `blackhole.*` bridge attrs
- `LowerBlackholeOps` pass-local 副产物

## 6. 旧 pass 的退场与 compatibility projection

激进重写路线下，下面这些 pass 不再保留 owner 身份：

- `LowerBlackholeOps`
- `PlanBlackholeCB`
- `AssignBlackholeCores`
- `LowerSpatialProgramToTTTarget`
- `MaterializeTTExecutableSpec`

处理规则：

- `LowerBlackholeOps`
  最多只能退成 leaf lowering/materialization consumer
- `PlanBlackholeCB / AssignBlackholeCores /
  LowerSpatialProgramToTTTarget`
  不再出现在长期主链中
- `MaterializeTTExecutableSpec`
  必须被真实的 `MaterializeBlackholeExecutable` 取代

过渡期如果仍保留：

- `blackhole.*` attrs
- 部分 `tl.tt_*` bridge attrs

它们也只能是：

- 调试投影
- compatibility projection
- migration 期间的短命桥接物

不能被 runtime/codegen 当主真源消费。

## 7. 完成判定

`Task 2` 完成必须同时满足：

1. `TTProgram companion` 成为唯一 target truth owner
2. `MaterializeBlackholeExecutable` 成为唯一 writer
3. runtime / codegen / `BlackholeModule`
   只读 `ExecutableSpec`
4. `LowerBlackholeOps / PlanBlackholeCB / AssignBlackholeCores /
   LowerSpatialProgramToTTTarget / MaterializeTTExecutableSpec`
   不再承担 planning owner
5. 发现缺口时，只能通过
   `补 TIR/DSL / 补 analysis / explicit unsupported`
   解决

在这五条同时成立之前，
`TTProgram` 已进入主链
不等于 `Task 2` 已完成。
