# TileLang Blackhole Backend Redesign

## 基本信息

- **文档 ID**: `final_blackhole_backend_redesign`
- **日期**: `2026-04-15`
- **状态**: 当前唯一权威总体设计文档
- **定位**: 只保留长期架构、层间边界、真源规则和当前 rewrite 方向

## 1. 设计结论

Blackhole 当前的核心问题不是“还差几个 builtin”或者“还差一个 contract”，
而是 **target builtin mapping 边界放错了**。

一旦 tile op、layout、真实 `load/store` 关系被打碎，
后段就只能靠：

- matcher
- side contract
- bridge attr
- payload bag

去恢复“原来这是在做什么”。

这正是历史包袱反复长出来的根因。

因此长期路线固定为：

```text
Normalized Tile TIR
  -> SpatialPlan companion
  -> TTProgram companion
  -> ExecutableSpec / BlackholeModule
```

这里不是再造一层新的 semantic IR，
而是把每层只收回它本来该拥有的 truth。

## 2. 第一性原理

对 spatial target，一个算子最终只会落成三类事实：

1. **访存**
   - 从哪里读
   - 搬到哪里
   - 怎么写回
   - 是否跨 core / multicast / gather / remote write
2. **计算**
   - 在 tile 上执行什么 compute builtin
3. **通信 / 同步**
   - 谁等谁
   - 何时可见
   - 哪些 barrier / semaphore / completion relation 生效

所以我们的编译链也必须按这三类事实组织，
而不是按历史补丁名词组织。

## 3. 层间边界

### 3.1 `Normalized Tile TIR`

唯一语义本体。

继续持有：

- tile op
- loop/domain
- predicate
- `BufferLoad / BufferStore`
- address expr
- region/subscript
- tile-op 参数

只要信息还能由 TIR 稳定表达，
就不允许复制到 companion。

### 3.2 `SpatialPlan companion`

只回答空间切分问题：

- 哪些 anchored sub-TIR 构成一个局部执行闭包
- 闭包之间的稳定 boundary 是什么
- 哪些 frontier 可以切
- 哪些 hint 经过 validate 后可进入 planner

长期只保留：

- `ExecutionClosure`
- `ClosureBoundary`
- `ValidatedHintSet`

它不负责：

- access pattern
- index map
- target builtin family
- CB / semaphore / runtime arg

### 3.3 `TTProgram companion`

只回答 target realization：

- block
- kernel
- transport
- sync
- ABI
- execution

长期 primary owner object set：

- `TTBlockPlan`
- `TTKernelPlan`
- `TTTransportPlan`
- `TTSyncPlan`
- `TTABIPlan`
- `TTExecutionPlan`

### 3.4 `ExecutableSpec / runtime / codegen`

只负责：

- 冻结 `TTProgram` 结果
- 物化并执行

不再承担：

- target planning
- semantic recovery
- builtin guessing

## 4. TT builtin mapping 边界

这次重写最关键的约束只有一条：

> **TT builtin mapping 必须发生在 anchored sub-TIR 仍然保留 tile-op、layout、真实 load/store truth 的边界。**

具体拆成两类 planning：

### 4.1 `PlanTTTransport`

输入：

- `BufferLoad / BufferStore`
- address expr
- region
- `ClosureBoundary`
- `TTBlockPlan`

输出：

- `TensorAccessor / CB / NoC / semaphore / multicast`
  这组 data movement protocol

### 4.2 `PlanTTCompute`

输入：

- tile op
- layout
- operand/result region

输出：

- TT-Metal compute family
  - `matmul`
  - `eltwise`
  - `reduce`
  - `sfpu`
  - `copy / pack`

### 4.3 `PlanTTSync`

只负责：

- ordering
- completion
- barrier / semaphore relation

不再兼职恢复 compute 或 transport 语义。

补充：

- 这里只定义 sync 这第三类事实
  该由谁负责
- 第一性原理目标是否完成，
  还要同时看 mapping 边界、
  真源位置和后段 gate

## 5. 真源规则

1. 语义 body 只存在于 `Normalized Tile TIR`
2. `SpatialPlan companion` 只保存 planning 必须持久化、但 TIR 没对象化的 truth
3. target truth 只存在于 `TTProgram companion`
4. `ExecutableSpec` 只物化，不是第二真源
5. 缺信息时只能：
   - 扩 `TIR/DSL/schema`
   - 补更早层 analysis
   - `explicit unsupported`
6. 不允许回退到 late matcher / late recovery

### 5.1 第一性原理目标的完成判定

当前 rewrite 的第一性原理目标，
不是“跑通一个 workload”
或者“把某个 pass 名字切过去”，
而是下面 4 条必须同时成立：

1. **mapping 边界正确**
   - TT builtin mapping
     发生在 anchored sub-TIR
     仍保留
     `tile-op / layout / load-store truth`
     的边界
2. **三类事实各有 owner**
   - transport 由 `PlanTTTransport`
   - compute 由 `PlanTTCompute`
   - sync 由 `PlanTTSync`
3. **真源位置收紧**
   - 语义 body 只在 `Normalized Tile TIR`
   - `SpatialPlan companion`
     只保留 planning 必须持久化的 truth
   - target truth 只在 `TTProgram companion`
   - `ExecutableSpec`
     只负责物化，不是第二真源
4. **后段不再补语义**
   - runtime / codegen
     不再做 planning recovery
   - late matcher / side contract / payload bag
     不再承担 owner 职责

补充约束：

- `direct / indirect / paged / sharded`
  都不是独立 semantic layer，
  而是地址表达式与 transport realization 的问题
- `row / col / face`
  只允许作为 leaf builtin variant 出现，
  不能回流成主设计 noun

## 6. 明确删除方向

下面这些都不再是长期设计对象：

- `SemanticProgram`
- `SpatialProgram` 作为独立 execution-bearing IR
- `row_*`
- `broadcast_sources`
- `index map`
- `access pattern`
- `buffer_distribution_contract`
- 其它只为 late matcher 服务的 side contract

下面这些当前可能仍出现在代码里，
但只按**迁移残留**处理：

- `blackhole.work_decomposition`
- `blackhole.compute_regions`
- `blackhole.pipeline_stages`
- `BuildTTProgram` 内部
  `PlanTTKernelABI / PlanTTCBAlloc / PlanTTCoreGroups`

它们不能继续扩张，也不能再被文档写成长期架构。

`2026-04-15` 当前状态补充：

- `SpatialProgram` 已退出 active compile/runtime path
- `buffer_distribution_contract`
  已从 active lowering/codegen/probe surface 删除
- 最近完成的是 `Task 3B cleanup`
  （`T3B.0-T3B.4`）
  这批旧链清理，
  不是当前 roadmap `R0` 的完成信号
- 当前剩余架构债不再是旧 side-contract public surface，
  而是 `BuildTTProgram` 内部 helper bridge
  还没有完全拆成真实
  `PlanTTTransport + PlanTTCompute`

## 7. Canonical Pass Chain

长期主链固定为：

```text
BindTarget
  -> AddWrapperForSingleBufStore
  -> LegalizeNegativeIndex
  -> VerifyParallelLoop
  -> InjectAssumes
  -> Simplify
  -> AnalyzeSpatialStructureFacts
  -> BuildSpatialPlanCompanion
  -> PlanTTBlocks
  -> PlanTTTransport
  -> PlanTTCompute
  -> PlanTTSync
  -> PlanTTABI
  -> PlanTTExecution
  -> BuildTTProgram
  -> ValidateTTProgram
  -> MaterializeBlackholeExecutable
```

其中真正的 cutover 重点是：

- `PlanTTTransport`
- `PlanTTCompute`

但第一性原理目标的完成判定
不止这两项；
还要求：

- admitted scope 内的 `PlanTTSync`
  进入稳定 owner/runtime semantics
- runtime / codegen
  严格退回到
  `TTProgram / ExecutableSpec`
  reader 角色
- late matcher / side contract / helper bridge
  不再承担主协议职责

## 8. 当前执行重点

当前优先级固定为：

- 这里的 `R0 / R1 / ...`
  只表示当前 roadmap 总优先级
- task 内部或 cleanup batch
  统一用 `Tn.x`

1. **R0: 真实 `PlanTTTransport + PlanTTCompute` cut-in**
   - 删除 helper bridge
   - 删除 late matcher
   - 删除 side contract
2. **R1: runtime gate 收口**
   - runtime / codegen 不再补 target planning
   - admitted / unsupported 边界要和 owner truth 对齐
3. **R2: admitted-scope `PlanTTSync` 收口**
   - `ordering / completion / barrier / semaphore`
     进入稳定 owner/runtime semantics
   - 这是第一性原理目标的一部分，
     不是后置支持面
4. **R3: `flash-attn` payoff**
5. **R4: wider family cutover**
   - `topk / fusedmoe / paged decode / chunk recurrence`
6. **R5: wider support surface**
   - copy / data movement / wider sync

其中：

- `R0-R2`
  用来完成第一性原理目标本身
- `R3-R5`
  用来验证这套分层不是纸面方案，
  并把 admitted support surface 继续扩出去

## 9. 硬约束

- `BlackholeModule` direct host path
  仍是唯一正式执行路径
- `ExecutableSpec`
  仍是 runtime 消费的最终物化产物
- copy / GEMM / export 当前支持面不能回退
- 不引入第二条正式执行路径
- runtime / codegen 不再承担 planning recovery
- 当前收口范围只针对 `Blackhole` active compile/runtime path

## 10. Supporting Docs

- 根因诊断：
  `tasks/dev_design/task0_ir_layering_root_cause.md`
- `SpatialPlan` 边界：
  `tasks/dev_design/task1_spatial_plan_companion.md`
- `TTProgram` 边界：
  `tasks/dev_design/task2_ttprogram_companion_cutover.md`
- runtime gate / workload cutover：
  `tasks/dev_design/task3_runtime_gate_and_workload_cutover.md`
- 当前状态：
  `tasks/progress.md`
