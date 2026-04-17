# Task 1: SpatialPlan Representation Cutover

## 基本信息

- **文档角色**: `SpatialPlan` 表示层合同文档
- **任务链位置**: `Normalized Tile TIR -> SpatialPlan -> TTProgram`
- **唯一总体设计**: `tasks/dev_design/final_blackhole_backend_redesign.md`

说明：

- 文件名中的 `companion`
  只是历史文件名，
  不是新的 IR 层命名
- 本文档只定义目标合同和完成判据
- 当前 repo HEAD 状态统一只看 `tasks/progress.md`

## 1. 目标

`SpatialPlan` 不再表示“薄兼容层”，
而是表示：

> **target-independent 的 virtual spatial/dataflow representation**

它负责承接：

- execution unit
- dataflow edge
- layout / sharding / distribution 关系
- phase / ordering / materialization boundary
- validated planner hints

它不负责：

- TT builtin family
- CB / semaphore / runtime arg
- target block placement
- executable materialization

## 2. 合法输入与禁止输入

`SpatialPlan` 的构造只允许读取：

- `Normalized Tile TIR`
- anchored sub-TIR 上可直接结构遍历得到的
  execution-unit / dependence / region evidence
- validate 后的 hints

禁止把下面这些东西抬成 `SpatialPlan` 的长期语义来源：

- 审计表列出的 legacy transition attrs / helper bridge
- 任何 TT-specific noun
- runtime/codegen/leaf path 恢复出来的结论

如果当前 TIR 证据不足，
结论只能是：

- 扩 TIR / DSL
- 扩当前构造逻辑
- 扩 validator
- 显式 reject / unsupported

## 3. `SpatialPlan` 的显式对象

### 3.1 `ExecutionUnit`

表示一个 anchored sub-TIR 执行单元。

它应编码：

- unit identity 和 anchors
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

它应编码：

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

这些属于 `TTProgram` 的 target slices。

### 3.3 `LayoutSpec`

表示 target-independent 的 virtual layout / sharding / distribution 关系。

它应编码：

- logical tensor / buffer identity
- shard / replicate / distribute 关系
- virtual device axis 绑定
- planner 需要持久化的 layout identity

### 3.4 `PhasePlan`

表示 virtual ordering / materialization boundary。

它应编码：

- phase identity
- unit membership
- inter-phase edge
- ordering / visibility obligation

### 3.5 `ValidatedHintSet`

表示经验证的 planner 输入。

它应编码：

- accepted hints
- rejected hints
- diagnostics

## 4. 兼容视图与迁移 residue

当前代码里的 legacy compatibility projection
可以继续存在，
但只允许作为：

- 调试展示
- 兼容视图
- 迁移 projection

它不是 `SpatialPlan` 的语义来源，
也不能反向定义下游 planner 合法性。

## 5. Validator 合同

`ValidateSpatialPlan` 必须检查：

1. execution-unit coverage 完整
2. 每条 `DataflowEdge`
   的 producer / consumer / subject 完整
3. phase membership 与 inter-phase edge 一致
4. `LayoutSpec`
   引用的对象都可回溯到 TIR / unit
5. `SpatialPlan`
   中不出现 TT noun
6. 兼容视图与显式对象不冲突

validator 失败时必须 fail-closed。

## 6. Construction / Lowering 边界

`SpatialPlan` 必须由当前 `Normalized Tile TIR`
直接构造。

当前实现可以使用：

- visitor
- matcher
- mutator
- builder

这些局部 mechanics。

如果实现上仍保留前置结构收集步骤，
它也只能留在同一个 `.cc`
里作为 pass-local helper，
不能形成：

- public analysis wrapper
- pass-to-pass facts bag
- 新的中间语义层

补充说明：

- 当前实现里如果仍存在
  `BuildSpatialPlanCompanion`
  这个名字，
  它也只是历史实现名，
  不是架构边界

## 7. 下游消费边界

下游 target planner
只允许消费：

- `SpatialPlan`
  的显式对象
- anchored sub-TIR

不允许：

- 从 fake protocol 恢复中间层语义
- 重新制造伪中间层 bag
- 用 leaf-time matcher 补回 `SpatialPlan` 本应承载的东西

## 8. 历史 surface 的迁移落点

具体显式名和 disposition
统一看协议审计表。

在表示层落点上固定为：

- compute-region-like 信息
  - 落到 `ExecutionUnit` coverage
- pipeline-stage-like 信息
  - 落到 `PhasePlan`
- copy/dataflow-like 信息
  - 落到 `DataflowEdge`
    与 TIR access semantics
- work-decomposition-like 信息
  - 不属于 `SpatialPlan`
  - 落到 `TTBlockPlan / TTExecutionPlan`

## 9. Completion Contract

`Task 1`
只有在下面这些条件同时满足后才算完成：

1. `SpatialPlan`
   已显式收成
   `ExecutionUnit / DataflowEdge / LayoutSpec / PhasePlan / ValidatedHintSet`
2. `ValidateSpatialPlan`
   已落地并成为下游 planner 的正式前置 gate
3. `work_decomposition / compute_regions / pipeline_stages`
   这类过渡 surface
   已经降成 projection / migration residue，
   不再承载中间层语义
4. 下游 target planner
   读取 `SpatialPlan` 显式对象，
   而不是继续从 fake protocol 恢复
   virtual spatial/dataflow 语义

## 10. 当前执行切片

`Task 1` 的代码方向固定为：

1. 扩 `SpatialPlan` object model
   - 先把
     `ExecutionClosure / ClosureBoundary`
     收成
     `ExecutionUnit / DataflowEdge`
2. 保留 legacy compatibility projection
   - 但只保留 projection 身份
3. 新增 `ValidateSpatialPlan`
4. 把
   `work_decomposition / compute_regions / pipeline_stages`
   拆回显式对象 /
   builder 直接恢复的结构语义
5. 再逐步让下游 planner
   读新对象，不再读 fake protocol
