# Task 2: TTProgram Representation Cutover

## 基本信息

- **文档角色**: `TTProgram` 表示层合同文档
- **任务链位置**: `Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec`
- **唯一总体设计**: `tasks/dev_design/final_blackhole_backend_redesign.md`

说明：

- 文件名中的 `companion`
  只是历史文件名，
  不是新的 IR 层命名
- 本文档定义
  `TTProgram`
  的长期表示层合同
  和它与 cleanup task2/task3 的关系
- 当前 repo HEAD 状态统一只看 `tasks/progress.md`

### repo HEAD note (`2026-04-22`)

- `tl.blackhole_lowering_requirements_seed`
  和
  `blackhole.cb_requirements`
  已退出 active planning chain
- staged CB planning truth
  已收回
  `TTProgram.cb_plans`
  / typed `TTCBPlan`
- `BuildTTProgram`
  当前只承担 staged slice aggregation /
  finalize /
  cleanup
- `ValidateTTProgram`
  当前会对 unresolved
  `unsupported_compute_ops`
  fail-close
- repo HEAD
  仍保留的
  `TTProgram.payload`
  字段
  只按 task3
  leaf compatibility residue
  记录，
  不再表述成
  `TTProgram`
  owner truth

## 1. 目标

`TTProgram`
是长期主链里的第三层显式表示，
不是 staging attr 的汇总袋，
也不是给 leaf writer
兜底的一包 payload。

它的语义固定为：

> **唯一的 TT-specific target realization representation**

它负责回答：

- block / core placement
- kernel family / role / core type
- transport / routing / delivery
- sync / completion / ordering
- ABI / runtime-args / accessor / semaphore binding
- execution / launch order / waves

它不负责：

- target-independent 的
  virtual spatial/dataflow 语义
- compute-side exact builtin 选择本身
- leaf projection /
  executable materialization /
  runtime backend admission

补充说明：

- cleanup task2/task3
  只是把 repo HEAD
  收回到这个合同上的执行切片
- 它们不是新的表示层，
  也不能反向重写
  `TTProgram`
  的长期语义

## 2. 合法输入与禁止输入

target planning
只允许读取：

- validated `SpatialPlan`
  的显式对象
  - `ExecutionUnit`
  - `DataflowEdge`
  - `LayoutSpec`
  - `PhasePlan`
  - `ValidatedHintSet`
- anchored sub-TIR
- 当前层已经建立好的
  compute-side exact builtin-selected IR
- `TTHardwareModel`
- validate 后的 target hints

这里要特别写清楚：

- `TTProgram`
  可以依赖
  “exact TT-Metal builtin legality
   已经在当前 TIR 上成立”
  这个前置条件
- 但这个前置条件
  不是 `TTProgram`
  自己创建的
  target-independent 语义层

不允许回升为 target planning 输入的东西：

- public `AnalyzeBlackhole*` wrapper
- internal `*Evidence(...)` helper
- `blackhole.lowering_requirements`
  / `Map<String, Any>` broad bag
- `tl.blackhole_lowering_requirements_seed`
- `blackhole.cb_requirements`
- runtime / codegen /
  executable projection
  自己恢复出来的结论
- leaf payload / projection /
  compatibility reader
  倒灌回来的语义

如果当前输入表示
证据不足，
结论只能是：

- 回到 `SpatialPlan`
  或当前 TIR
  补显式表示
- 扩 validator
- 显式 reject / unsupported

不能再造 replacement bag /
replacement seed /
replacement helper layer。

## 3. `TTProgram` 的显式 slice

### 3.1 `TTBlockPlan`

表示 target-side 的
task grouping /
placement 粒度。

它应编码：

- block identity
- placement kind
- task membership
- block-level payload

它不负责：

- target-independent work semantics
- leaf-time core launch 恢复

### 3.2 `TTKernelPlan`

表示 target kernel
的显式 realization。

它应编码：

- kernel identity
- kernel kind / role
- core type
- 所属 `TTBlockPlan`
- 所属 `TTABIPlan`

它不负责：

- helper/composite builtin 的补洞
- leaf reader
  再猜一遍 kernel kind

### 3.3 `TTTransportPlan`

表示 target transport /
routing /
delivery 关系。

它应编码：

- source task
- target task
- payload kind
- delivery kind
- transport payload

### 3.4 `TTSyncPlan`

表示 target-visible 的
ordering /
completion /
handoff 关系。

它应编码：

- source task
- target task
- sync kind
- ordering kind
- completion kind

### 3.5 `TTABIPlan`

表示 kernel ABI /
runtime arg /
accessor /
semaphore binding 合同。

它应编码：

- kernel identity
- runtime args
- common runtime args
- compile-time arg specs
- accessors
- semaphore bindings

这里的 ABI
已经是 target-side
显式程序构造事实，
不是 leaf runtime
再从 payload 猜回来的约定。

### 3.6 `TTExecutionPlan`

表示 target-visible 的
execution /
launch grouping。

它应编码：

- execution identity
- ordered kernel set
- phase / wave grouping
- execution payload

### 3.7 兼容 / realization detail 视图

当前代码里如果仍保留：

- `TTKernel`
- `TTCoreGroup`
- `TTCBPlan`
- `TTSemaphorePlan`
- `TTComputeSyncPlan`
- `TTDstLayoutPlan`
- `TTProgram.payload`

它们也只能是：

- compatibility carrier
- realization detail
- leaf projection data
- 尚未删除的 cleanup residue

它们不能反向定义
`TTProgram`
的 owner truth。

长期 owner truth
只能写成：

- `TTBlockPlan`
- `TTKernelPlan`
- `TTTransportPlan`
- `TTSyncPlan`
- `TTABIPlan`
- `TTExecutionPlan`

## 4. Wrong-Now Residue 与 Cleanup Debt

下面这些东西
必须明确写成
**wrong now, delete later**
或 **transitional debt**，
不能写成 `TTProgram`
的中期协议层。

### 4.1 public/internal analysis bag 不是 `TTProgram` 边界

如果 repo HEAD
里仍存在：

- public `AnalyzeBlackhole*`
- internal `*Evidence(...)`
- `BuildBlackholeLoweringRequirements`
- `blackhole.lowering_requirements`
- `tl.blackhole_lowering_requirements_seed`

它们的正确口径只能是：

- cleanup task2
  尚未删完的 planning debt
- 仍被局部实现依赖的 side channel
- 必须继续删除的 forced residue

它们不是
`TTProgram`
的长期输入边界，
也不是可接受的
medium-term bridge layer。

### 4.2 `BuildTTProgram` 不是 planning owner

如果 repo HEAD
里仍保留
`BuildTTProgram`
这个入口，
它的长期角色只能是：

- staged slice aggregation
- cross-slice completeness check
- 中间 attrs / seeds 清理

它不能再承担：

- target planning owner truth
- public/internal bag reread
- 再发明一层中间协议

也就是说：

- `PlanTTBlocks`
- `PlanTTCompute`
- `PlanTTTransport`
- `PlanTTSync`
- `PlanTTABI`
- `PlanTTExecution`

这些名字如果仍存在，
也只是当前实现里
分别构造 slice 的手段；
真正的长期边界
还是 `TTProgram`
这层显式对象。

### 4.3 payload residue 不是 `TTProgram` 语义

如果 repo HEAD
里仍通过
`TTProgram.payload`
携带：

- `buffer_tile_bridge_specs`
- `unsupported_compute_ops`
- `compute_contract`
- `gemm_contract`
- `multi_compute_contracts`
- `multi_gemm_contracts`
- `direct_runtime_unsupported_reasons`
- 其他 leaf/runtime compatibility payload

这些东西也只能写成：

- leaf compatibility debt
- `ExecutableSpec` 投影 residue
- task3 继续删除的 forced carrier

它们不能被文档表述成：

- `TTProgram`
  必须长期拥有的语义字段
- `ValidateTTProgram`
  的长期 owner truth
- TT-Metal target model
  要求保留的 compiler-side contract

其中要单独写清楚：

- `buffer_tile_bridge_specs`
  仍然活着时，
  只是 cleanup task1/task3
  之间尚未删完的 narrow compatibility residue
- `compute_contract` /
  `gemm_contract` /
  `multi_*_contracts`
  仍然活着时，
  只是 task3
  的 leaf/runtime compatibility debt
- `ValidateTTProgram`
  对这些 payload family
  做 shape check
  只能表示
  debt containment，
  不能表示这些字段已经成为
  `TTProgram`
  的合法 owner truth
- required end-state
  是把仍需要的
  compute ABI /
  compute config /
  epilogue /
  materialization
  事实收进
  `TTKernelPlan`
  /
  `TTABIPlan`
  /
  `TTExecutionPlan`
  /
  `TTLiveFormPlan`
  /
  `TTMaterializationPlan`
  /
  `TTConsumerBindingPlan`
  等 typed slice，
  或等价的一等 typed
  `TTProgram`
  object；
  `TTProgram.payload`
  不再被
  `MaterializeBlackholeExecutable`
  复制成 leaf truth

## 5. Validator 合同

`ValidateTTProgram`
是主链对象，
不是补丁。

它必须成为：

- `TTProgram`
  进入 leaf writer
  之前的正式 hard gate
- exact TT-Metal
  target realization legality
  的 fail-closed verifier

当前合同至少包括：

1. `TTBlockPlan /
    TTKernelPlan /
    TTTransportPlan /
    TTSyncPlan /
    TTABIPlan /
    TTExecutionPlan`
   的 completeness / consistency
2. slice 之间的 index /
   reference /
   alignment
   必须闭合
3. exact TT-Metal builtin /
   transport /
   sync /
   ABI /
   execution legality
   必须已经成立
4. `TTKernel` /
   `TTCoreGroup` /
   `TTCBPlan` /
   `TTSemaphorePlan` /
   `TTComputeSyncPlan` /
   `TTDstLayoutPlan`
   如果暂时存在，
   只能作为
   compatibility / realization detail
   接受对齐检查，
   不能反客为主
5. `TTProgram.payload`
   如果暂时存在，
   validator
   只能检查其残留 shape
   不违反当前实现的 leaf compatibility 需要，
   不能把它升格成
   planning owner truth

补充要求：

- validator 成功后，
  leaf writer
  必须只消费
  已验证的 `TTProgram`
- runtime / codegen
  不允许绕过这个 gate
  再补 planner 语义

## 6. Construction / Lowering 边界

`TTProgram`
必须由当前：

- validated `SpatialPlan`
- exact builtin-selected current IR
- `TTHardwareModel`

直接 lower / build 出来。

允许的实现形态是：

- 按 slice 拆开的 planner passes
- 同一实现单元里的
  visitor / matcher / builder
- pass-local collector / helper

明确禁止：

- public analysis wrapper
- pass-to-pass facts bag
- broad `Map<String, Any>` planning carrier
- leaf writer /
  codegen /
  runtime
  去反推上游 planning 语义

换句话说：

- 如果当前实现
  需要 `PlanTT*` 这组 pass，
  它们也只是当前物理拆分方式
- 如果当前实现
  需要 `BuildTTProgram`，
  它也只能是
  staged slices
  的聚合 / finalize / cleanup 点

不能把这些 implementation detail
写成新的架构层。

## 7. Reader / Writer 边界

`TTProgram`
是唯一 target realization 表示。

`MaterializeBlackholeExecutable`
是唯一 leaf writer。

它只允许读取：

- validated `TTProgram`
- `TTHardwareModel`
- 必要的 leaf-local projection logic

build / codegen / runtime /
`BlackholeModule`
只允许读取：

- `tl.blackhole_executable`
- 或其内部
  `ExecutableSpec`
  projection

不允许再读取：

- public/internal legacy analysis bag
- planning seed / bridge seed
- 原始 planning residue
- 任何需要 target planner
  再解释一次的 side channel

这里要特别写清楚：

- `TTProgram`
  需要在 leaf materialization 之前
  就已经显式承接
  kernel / CB / semaphore /
  runtime-arg / execution grouping
  这类 target-side 程序构造事实
- 这些事实不能被推迟到
  runtime / codegen /
  leaf reader
  再恢复

## 8. 历史 Surface 的落点

具体显式名和 disposition
统一看协议审计表。

表示层落点固定为：

- work-decomposition-like 信息
  - 落到
    `TTBlockPlan`
    / `TTExecutionPlan`
- kernel kind / role /
  core type / ABI 绑定
  - 落到
    `TTKernelPlan`
    / `TTABIPlan`
- transport / delivery /
  routing-like 信息
  - 落到 `TTTransportPlan`
- sync / semaphore /
  completion-like 信息
  - 落到
    `TTSyncPlan`
    和必要的
    target-side realization detail
- runtime-arg /
  accessor /
  compile-time arg-like 信息
  - 落到 `TTABIPlan`
- launch grouping /
  phase / wave-like 信息
  - 落到 `TTExecutionPlan`
- `blackhole.lowering_requirements` /
  `tl.blackhole_lowering_requirements_seed` /
  `blackhole.cb_requirements`
  - 不属于 `TTProgram`
  - 只能删除，
    或收回 pass-local helper
- `buffer_tile_bridge_specs` /
  `compute_contract` /
  `gemm_contract` /
  `multi_*_contracts` /
  `direct_runtime_unsupported_reasons`
  - 不属于 `TTProgram`
    长期字段
  - 只允许作为 task3
    继续删除的
    leaf compatibility residue

## 9. Completion Contract

`Task 2`
只有在下面这些条件同时满足后
才算完成：

补充硬约束：

- `Task 2`
  默认按终态实现，
  不接受
  “先把 owner truth
   迁到 typed `TTProgram` slice，
   再把旧 analysis bag /
   lowering bag /
   seed /
   wrapper
   留到后面删除”
  的过渡式收口
- 除本文档明确标成
  task3
  继续删除对象的
  leaf compatibility residue 外，
  旧 `wrapper / bag / seed / payload`
  只要仍在
  `Task 2`
  边界里充当兼容壳，
  就算未完成

1. target planning
   直接构造
   `TTBlockPlan /
    TTKernelPlan /
    TTTransportPlan /
    TTSyncPlan /
    TTABIPlan /
    TTExecutionPlan`
   这组显式 slice
2. `BuildTTProgram`
   已退成
   staged slice aggregation /
   finalize /
   cleanup 点，
   不再承担 planning owner truth
3. `ValidateTTProgram`
   已对这组显式 slice
   提供正式 hard gate
4. public/internal
   legacy analysis bag /
   lowering bag /
   seed
   已经退出 active planning chain，
   不再定义
   `TTProgram`
   输入边界
5. `TTKernel` /
   `TTCoreGroup` /
   `TTCBPlan` /
   `TTSemaphorePlan` /
   `TTComputeSyncPlan` /
   `TTDstLayoutPlan` /
   `TTProgram.payload`
   即使仍暂时存在，
   也只剩 compatibility /
   realization detail /
   leaf residue 身份，
   不再承载 owner truth
6. leaf writer / leaf readers
   不再把 planning residue
   反向合法化成
   `TTProgram`
   语义边界
7. `compute_contract` /
   `gemm_contract` /
   `multi_*_contracts`
   如果仍存在，
   只允许作为 task3
   leaf compatibility debt；
   它们不能被
   `ValidateTTProgram`
   的 shape check
   或 executable projection
   重新解释成
   `TTProgram`
   完成态字段

## 10. 与 Cleanup Task 的关系

这份文档定义的是
长期 `TTProgram` 合同，
不是当前 cleanup 顺序说明。

和 cleanup 文档的关系固定为：

1. cleanup task2
   负责删除
   public/internal legacy analysis bag
   对 active planning chain
   的控制，
   把
   `SpatialPlan -> TTProgram`
   收回到
   current IR /
   current object /
   direct planner builder
2. cleanup task3
   负责继续删除
   `TTProgram.payload`
   /
   executable projection
   /
   codegen/runtime
   仍在使用的
   leaf compatibility residue
3. 如果 repo HEAD
   在 cleanup 期间
   仍然保留上述 debt，
   正确写法也只能是
   **wrong now, delete later**
   或
   **transitional debt**

当前实现依赖
不能削弱这个 verdict。
