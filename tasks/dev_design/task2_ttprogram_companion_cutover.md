# Task 2: TTProgram Companion Cutover

## 基本信息

- **文档角色**: `Task 2` 的 target owner cutover 设计文档
- **当前状态**: `2026-04-15` 活动设计文档；
  `Task 2` 的 active TT bundle / helper shell
  已切进主链，
  但 owner cutover 还没有完成；
  当前未收口项是
  build/codegen/executable extraction
  仍消费
  `blackhole.lowering_requirements`
- **任务链位置**:
  `Normalized Tile TIR -> SpatialPlan companion -> TTProgram companion ->
  ExecutableSpec` 中第二层 companion 的 owner 设计
- **目标**:
  让 `TTProgram companion` 成为唯一 target truth，
  让 executable writer/readers 的边界
  和代码现实重新一致
- **非目标**:
  - 不在 `Task 2` 里兑现 runtime/correctness payoff
  - 不在 `Task 2` 里承接 workload family 回归
  - 不把 `TTProgram companion` 退化成 payload-backed attr bag
  - 不把 non-Blackhole backend 一并收口到同一套 target contract
  - 不把 public Python `transform` API 改名当作 cutover 前置
- **唯一总体设计**: `tasks/dev_design/final_blackhole_backend_redesign.md`
- **上游设计输入**:
  - `tasks/dev_design/task0_ir_layering_root_cause.md`
  - `tasks/dev_design/task1_spatial_plan_companion.md`

读法说明：

- 本文档只定义当前有效的 `TTProgram` owner 边界
- 若文中出现旧 pass / 兼容 shell / bridge attr，
  默认按**迁移 inventory**理解，
  不是当前长期架构对象

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
- 统一 non-Blackhole backend 的 runtime / artifact contract
- 先行清理 public `tilelang.transform` 命名

## 2. 合法输入边界

`TT` planner 只允许消费下面四类输入：

- `SpatialPlan companion`
  的最小 planning truth：
  `ExecutionClosure / ClosureBoundary / ValidatedHintSet`
- closure 所锚定的 sub-TIR
- `ValidatedHintSet` 中已经验证成功的 target hints
- `TTHardwareModel`

补充说明：

- anchored sub-TIR 中仍然存活的
  tile-op、layout、`BufferLoad / BufferStore`、region、address expr
  都属于合法 planner 输入
- 这些输入必须在 target builtin 选择发生前被直接消费，
  不能等到 lowered loop / bridge attr / matcher 之后再恢复

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
- 引入 `row_*`、`broadcast_sources`、
  `index map / access pattern` 这类 side contract
  充当 target owner truth

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
  - owner：reader / compute / writer kernel 内的 target builtin family
- `TTTransportPlan`
  - owner：boundary 的 target transport realization
  - owner：`TensorAccessor / CB / NoC / semaphore / multicast`
    这组 data movement protocol
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
  -> PlanTTCompute
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
  - 同时承接 memory-access truth
    和 communication 里的 routing / multicast / remote endpoint truth
  - 不再从 builtin 排列或 matcher 恢复 transport intent
- `PlanTTCompute`
  - 根据 anchored sub-TIR 中仍然可见的
    tile-op、layout、operand/result region
    做 compute builtin family 选择
  - 不再从 lowered loop / `row_*` / side contract
    恢复 compute intent
- `PlanTTSync`
  - 根据 dependency / boundary / transport result
    做 communication 的 completion / visibility realization
- `PlanTTABI`
  - 把 owner-side access truth 物化成 target ABI
  - 不再从 lowered loop、arg kind、runtime heuristics 反推
- `PlanTTExecution`
  - 形成 kernel grouping / launch order / wave scheduling
  - 承接 communication 里的 core placement / execution order truth
- `BuildTTProgram`
  - 聚合 target owner plan
  - 不重新恢复 target truth

当前 `2026-04-15` active path 补充：

- 代码已经显式经过
  `PlanTTBlocks -> PlanTTCompute -> PlanTTTransport -> BuildTTProgram`
- 其中 `PlanTTTransport`
  当前消费 `PlanTTCompute`
  发布的 CB/accessor requirement schema
  来完成 CB allocation / transport materialization
- `BuildTTProgram`
  已退回聚合器；不再内部直接实例化
  `PlanTTKernelABI / PlanTTCBAlloc / PlanTTCoreGroups`
- `ValidateTTProgram`
  - 检查 completeness / consistency
  - 禁止 legacy attrs、seed attrs、payload bag 回升为主协议
- `MaterializeBlackholeExecutable`
  - 只消费经过验证的 `TTProgram`
  - 物化最终 `ExecutableSpec`

## 4.1 当前执行优先级

基于当前 review 结论，`Task 2` 内部继续拆成下面的顺序：

- active roadmap 统一使用 `R0 / R1 / ...`
- `Task 2` 内部顺序统一使用 `T2.x`

1. **T2.0: internal owner cutover**
   - 让 `Blackhole` active compile path
     先真实经过新的 owner chain
   - 重点是 pass owner、typed plan object、active reader 切换
   - 旧 public API 名字可暂时保留为 compatibility shell
2. **T2.1: target truth object set 收口**
   - 落实 `TTBlockPlan / TTKernelPlan / TTTransportPlan /
     TTSyncPlan / TTABIPlan / TTExecutionPlan`
   - 旧 `TTCoreGroup / TTCBPlan / ... / payload bag`
     退出 primary owner 角色
3. **T2.2: writer cutover**
   - `MaterializeBlackholeExecutable`
     成为唯一 writer
   - `MaterializeTTExecutableSpec`
     退为 compatibility bridge 或直接退出
4. **T2.3: phase bundle 固化**
   - 为编译、测试、probe 建立 canonical bundle/helper
   - 不再让测试手写长 pass 链充当事实标准

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

补充纪律：

- `Task 2` 的完成判定基于 internal owner chain 是否切进 active path
- 不要求 non-Blackhole backend 同期收口
- 不要求 public Python `transform` API 同期改名

## 7. 完成判定

`Task 2` 完成必须同时满足：

1. `TTProgram companion` 成为唯一 target truth owner
2. `MaterializeBlackholeExecutable` 成为唯一 writer
3. build/runtime side consumers
   只读 typed target truth
   （`TTProgram / ExecutableSpec`），
   不读 legacy attrs
4. `LowerBlackholeOps / PlanBlackholeCB / AssignBlackholeCores /
   LowerSpatialProgramToTTTarget / MaterializeTTExecutableSpec`
   不再承担 planning owner
5. 发现缺口时，只能通过
   `补 TIR/DSL / 补 analysis / explicit unsupported`
   解决

不额外要求：

- non-Blackhole backend 采用同一套 target/object 边界
- public Python `transform` API 已完成命名清理

在这五条同时成立之前，
`TTProgram` 已进入主链
不等于 `Task 2` 已完成。

## 8. 当前落地到的程度

`Task 2` 当前已经按下面方式切入主链：

- active Blackhole compile path 已固定使用 canonical TT bundle：
  `BuildTTProgram -> ValidateTTProgram -> MaterializeBlackholeExecutable`
- Python / engine 侧已经固化 canonical bundle helper：
  `LowerToBlackholePhaseB -> LowerToBlackholeTTProgram -> LowerToBlackholeExecutable`
- `TTProgram`
  已成为 active target-truth carrier；
  `tt_program_projection`
  当前也已成为 runtime/build 侧的正式读取入口
- `LowerSpatialProgramToTTTarget / ValidateTTTargetProgram /
  MaterializeTTExecutableSpec`
  继续保留为 compatibility shell，
  但不再作为当前入口命名
- 测试 helper 已切到 bundle helper，
  不再把长 pass 链手写成事实标准

但下面这些 gap 说明
`Task 2`
还不能写成已完成：

- `T2.4`
  已完成：
  `MaterializeBlackholeExecutable`
  不再是 no-op shell；
  当前显式把
  已验证 `TTProgram`
  物化成
  `tl.blackhole_executable`
  writer attr，
  build 缺失该 writer attr
  直接 fail-fast
- codegen / executable extraction
  仍消费
  `blackhole.lowering_requirements`
  这类过渡 attr，
  不能宣称已经只读
  `TTProgram / ExecutableSpec`
- `blackhole.*`
  过渡 analysis/projection attr
  仍在 active path / test surface 中可见

## 8.1 当前 closeout tasks

在当前审计口径下，
`Task 2`
还需要完成下面 2 项 closeout：

1. **`R1.1`（旧别名：`T2.5`）: 去掉 build/codegen/executable extraction 对 `blackhole.lowering_requirements` 的依赖**
   - unsupported-compute /
     bridge-spec /
     materialization gate
     收回 `TTProgram / ExecutableSpec`
     typed truth
   - 当前执行前置是
     `Task 3`
     的
     `R0.1-R0.3`：
     先把混在
     `blackhole.lowering_requirements`
     里的
     effect/use-role、
     liveness、
     materialization decision
     拆成独立 analysis/planner，
     再收掉这条过渡 attr 依赖
2. **`T2.6`: 收紧 `blackhole.*` 过渡 attr 的公开地位**
   - probe/debug 可保留，
     但不能继续被文档和 regression
     当作正式 target owner surface
