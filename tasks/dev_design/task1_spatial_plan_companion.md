# Task 1: SpatialPlan 与 Virtual Layer 边界

## 基本信息

- **文档角色**: `Task 1` 的 `SpatialPlan` 设计文档
- **当前状态**: `2026-04-16` 活动设计文档
- **任务链位置**: `Normalized Tile TIR -> SpatialPlan -> TTProgram`
- **唯一总体设计**: `tasks/dev_design/final_blackhole_backend_redesign.md`

## 1. 角色重定义

`SpatialPlan` 不再表示“薄兼容层”，
而是表示：

> **target-independent 的 virtual spatial/dataflow program**

它负责承接：

- execution unit
- dataflow edge
- layout / sharding / distribution truth
- phase / ordering / materialization boundary
- validated planner hints

它不负责：

- TT builtin family
- CB / semaphore / runtime arg
- target block placement
- executable materialization

## 2. 合法输入

`SpatialPlan` builder 只允许读取：

- `Normalized Tile TIR`
- 结构分析得到的 execution-unit candidate / dependence / region facts
- validate 后的 hints

禁止把下面这些东西抬成 `SpatialPlan` 真源：

- 审计表列出的 legacy transition attrs / helper bridge
- 任何 TT-specific noun

如果 TIR 证据不足，
结论只能是：

- 补 TIR / DSL / schema
- 补更早的 analysis
- 显式 reject / unsupported

## 3. 长期 owner object set

### 3.1 `ExecutionUnit`

表示一个 anchored sub-TIR 执行单元。

owner：

- unit 名称和 anchors
- unit role
- unit 覆盖的稳定 TIR 区域
- 单元级别的 locality / carry / aggregation obligation

它不复制：

- tile op 本体
- address expr
- region 细节

这些仍留在 anchored sub-TIR。

### 3.2 `DataflowEdge`

表示 execution unit 之间的显式数据流关系。

owner：

- producer / consumer unit
- edge kind
  - `flow`
  - `carry`
  - `broadcast`
  - `reduction`
  - `join`
- subject
- 是否跨 phase
- edge anchors

它不直接决定：

- CB
- NoC route
- TT semaphore
- target delivery kind

这些属于 `TTTransportPlan / TTSyncPlan`。

### 3.3 `LayoutSpec`

表示 target-independent 的 virtual layout / sharding truth。

owner：

- logical tensor/buffer identity
- shard / replicate / distribute 关系
- virtual device axis 绑定
- planner 需要持久化的 layout identity

### 3.4 `PhasePlan`

表示 virtual ordering / materialization truth。

owner：

- phase identity
- unit membership
- inter-phase edge
- ordering / visibility obligation

### 3.5 `ValidatedHintSet`

表示经验证的 planner 输入。

owner：

- accepted hints
- rejected hints
- diagnostics

## 4. 兼容视图

当前代码里的：

- legacy compatibility projection

可以继续存在，
但只允许作为：

- 调试展示
- 兼容视图
- 迁移 projection

长期 primary owner truth
已经转到：

- `ExecutionUnit`
- `DataflowEdge`
- `LayoutSpec`
- `PhasePlan`

## 5. Validator 纪律

`ValidateSpatialPlan` 必须检查：

1. execution unit coverage 完整
2. 每条 `DataflowEdge`
   的 producer / consumer / subject 完整
3. phase membership 与 inter-phase edge 一致
4. `LayoutSpec`
   引用的对象都可回溯到 TIR / unit
5. `SpatialPlan`
   中不出现 TT noun
6. 兼容视图与主 truth 不冲突

validator 失败时必须 fail-closed。

## 6. Pass 责任

### `AnalyzeSpatialStructureFacts`

负责：

- 收集 closure 候选
- 收集 dependence / region / carry / boundary facts
- 形成 builder 需要的 analysis facts

### `BuildSpatialPlanCompanion`

负责：

- 从 analysis facts 构造
  `ExecutionUnit / DataflowEdge / LayoutSpec / PhasePlan / ValidatedHintSet`
- 同步生成 legacy compatibility projection

### `ValidateSpatialPlan`

负责：

- 保证 `SpatialPlan`
  是后续 planner 的稳定输入

### 下游 readers

只允许：

- `PlanTTBlocks`
- `PlanTTCompute`
- `PlanTTTransport`
- `PlanTTSync`
- `PlanTTABI`
- `PlanTTExecution`

消费 `SpatialPlan`。

不允许下游重新制造新的伪中间层 bag。

## 7. 现存 fake protocol 的归位

具体显式名和 disposition
统一看协议审计表。

在 owner 语义上固定为：

- compute-region-like 信息
  - 拆回 `ExecutionUnit` coverage
- pipeline-stage-like 信息
  - 拆回 `PhasePlan`
- copy/dataflow-like 信息
  - 拆回 `DataflowEdge` 和 TIR access truth
- work-decomposition-like 信息
  - 不属于 `SpatialPlan`；
    归 `TTBlockPlan / TTExecutionPlan`

## 8. 当前实现方向

`Task 1` 的代码方向固定为：

1. 扩 `SpatialPlan` object model
2. 保留 legacy compatibility projection
3. 新增 `ValidateSpatialPlan`
4. 逐步让 `BuildTTProgram` 和各 `PlanTT*`
   读新对象，不再读 fake protocol
