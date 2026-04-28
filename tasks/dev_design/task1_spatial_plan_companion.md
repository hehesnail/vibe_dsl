# Task 1: SpatialPlan Representation Cutover

## 基本信息

- **文档角色**: `SpatialPlan` 表示层合同文档
- **任务链位置**: `Normalized Tile TIR -> SpatialPlan -> TTProgram`
- **唯一总体设计**: `tasks/dev_design/final_blackhole_backend_redesign.md`

说明：

- 文件名中的 `companion`
  只是历史文件名，
  不是新的 IR 层命名
- 本文档定义
  `SpatialPlan`
  的长期表示层合同
  以及它与 archived cleanup task1/task2
  的历史边界关系
- 当前 repo HEAD 状态统一只看 `tasks/progress.md`

## 1. 目标

`SpatialPlan`
是长期主链里的第二层显式表示，
不是“薄兼容层”，
也不是给后续 planner
准备的一包临时 facts。

它的语义固定为：

> **target-independent 的 virtual spatial/dataflow program**

它负责回答：

- 哪些 anchored sub-TIR
  构成稳定执行单元
- 单元之间有哪些显式
  dataflow / carry / reduction / broadcast / join 关系
- virtual layout / sharding / distribution 语义是什么
- virtual phase / ordering / materialization boundary 是什么
- logical live value
  的 producer / consumer /
  carry / reduction / broadcast
  关系是什么
- logical value
  在跨 execution unit /
  phase 被消费前
  需要什么 materialization boundary
- 哪些 hints
  经过 validate 后进入 planner

它不负责：

- TT builtin family
- CB / semaphore / runtime arg
- target block placement
- transport / sync / launch
- executable leaf materialization

补充说明：

- archived cleanup task1/task2
  只是把历史 repo state
  收回到这个合同上的执行记录
- 它们不是当前活动路线图，
  也不是新的表示层，
  也不能反向重写
  `SpatialPlan`
  的长期语义

## 2. 合法输入与禁止输入

`SpatialPlan`
的构造只允许读取：

- `Normalized Tile TIR`
- anchored sub-TIR
  上可直接结构遍历得到的
  execution-unit / dependence / region evidence
- validate 后的 hints

允许存在的实现 mechanics：

- visitor
- matcher
- mutator
- builder
- pass-local collector

但这些只能是
“从当前 TIR
 直接构造 `SpatialPlan`”
的局部实现手段，
不能升格成新的语义层。

禁止把下面这些东西
抬成 `SpatialPlan`
的长期语义来源：

- 审计表列出的
  legacy transition attrs / helper bridge
- public analysis wrapper
  或 pass-to-pass facts bag
- 任何 TT-specific noun
- runtime / codegen / leaf path
  恢复出来的结论

如果当前 TIR
证据不足，
结论只能是：

- 扩 TIR / DSL
- 扩当前构造逻辑
- 扩 validator
- 显式 reject / unsupported

不能靠 bag / payload /
helper alias / late matcher
补一层旁路语义。

## 3. `SpatialPlan` 的显式对象

### 3.1 `ExecutionUnit`

表示一个 anchored sub-TIR
执行单元。

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

表示 execution unit
之间的显式数据流关系。

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

这些属于
`TTProgram`
的 target slices。

### 3.3 `LayoutSpec`

表示 target-independent 的
virtual layout /
sharding /
distribution 关系。

它应编码：

- logical tensor / buffer identity
- shard / replicate / distribute 关系
- virtual device axis 绑定
- logical mesh axis
  与 tensor axis /
  phase
  的关系
- collective intent
  的 target-independent
  形态，
  例如
  replicate /
  shard /
  reduce /
  gather
  需要跨哪些 virtual axis
- planner 需要持久化的 layout identity

它不编码：

- TT-Metal `MeshDevice`
- `MeshWorkload`
- `MeshBuffer`
- fabric node id
- physical device coordinate

这些属于
`TTProgram`
的 target realization。

### 3.4 `PhasePlan`

表示 virtual ordering /
materialization boundary。

这里的 `materialization`
只指 virtual phase
上的可见性 /
顺序 / 边界收束，
不是 TT-Metal
`Program / MeshWorkload`
上的 compile /
configure / launch 物化。

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

### 3.6 `LiveValue` / `LiveValueEdge` / `MaterializationBoundary`

表示 support surface
扩展时必须保留的
logical live-value
关系。

它们应编码：

- logical value identity
- producer / consumer
  `ExecutionUnit`
- producer-consumer edge
  与 carry /
  reduction /
  broadcast /
  recurrence 分类
- consumer 是否需要
  full logical value
  或可消费 distributed slice
- materialization boundary
  的 visibility /
  coverage /
  phase relation

它们不编码：

- CB id
- semaphore id
- runtime arg slot
- TT core coordinate
- publication protocol

这些属于
`TTProgram`
或
`ExecutableSpec`
边界。

当前实现已提供这组三类对象的第一轮
schema：

- `live_values`
  记录 logical value identity、
  producer unit、
  subject、
  logical shape /
  dtype
  和 target-independent role
- `live_value_edges`
  把 logical live value
  绑定到
  `DataflowEdge`
  与 producer /
  consumer
  relation，
  并标出 consumer
  是否要求 full logical value
  或可消费 distributed slice
- `materialization_boundaries`
  描述 live edge
  在 same-phase /
  cross-phase
  场景下的 visibility、
  source live value、
  target live value、
  phase relation
  和 coverage

当前 P1 refinement 已把
same-unit local value flow
纳入 `SpatialPlan`：

- `BufferStore`
  的结构化 local-to-local
  source / target 关系会生成
  `materialize`
  `DataflowEdge` /
  `LiveValueEdge`
- `MaterializationBoundary`
  同时携带
  `source_live_value(_index)`
  与
  `target_live_value(_index)`
- distributed-slice consumer
  由 `LiveValueEdge.accepts_distributed_slice`
  和
  `MaterializationBoundary.logical_coverage`
  显式表达
- 下游 planner
  必须按
  source -> target
  boundary ref 消费，
  不能只按 source subject
  取第一条 live value /
  boundary

后续 support-surface
工作如果需要更细的 recurrence /
reduction /
broadcast /
live-in distinction，
应继续扩展这组三类对象和
validator，
不能由
`PlanTT*`
或 leaf reader
从 body order /
builtin 序列 /
buffer 名
恢复。

### 3.7 兼容视图

当前代码里如果仍保留：

- `ExecutionClosure`
- `ClosureBoundary`
- 其他 legacy compatibility projection

它们也只能是：

- 调试展示
- 兼容视图
- 迁移 projection

它们不是
`SpatialPlan`
的 owner truth，
也不能反向定义
下游 planner 合法性。

## 4. Wrong-Now Residue 与 Cleanup Exception

下面这些东西
必须明确写成
**wrong now, delete later**
或 **transitional debt**，
不能写成新的中期层。

### 4.1 public wrapper / facts object 必须退出主链与 public surface

下面这些 surface
不允许继续存在于
active chain
或 public API：

- `AnalyzeSpatialStructureFacts`
- `SpatialStructureFacts`
- `tl.spatial_structure_facts`
- `BuildSpatialPlanCompanion`

它们的唯一正确 disposition 是：

- 删除 public wrapper
- 删除 pass-to-pass facts attr
- 删除 facts object
- 把仍需要的局部结构分析
  收回
  `BuildSpatialPlan`
  同一实现单元内的
  pass-local mechanics

长期边界只能写成：

- `Normalized Tile TIR`
- `SpatialPlan`
- `TTProgram`

而不是某个 pass 名字
或 facts helper 名字。

### 4.2 legacy attrs 不是 `SpatialPlan` 语义

下面这些 surface
不能继续被描述成
`SpatialPlan`
的语义来源：

- `blackhole.work_decomposition`
- `blackhole.compute_regions`
- `blackhole.pipeline_stages`

如果它们未来重新出现在
active chain，
只能视为：

- regression
- 需要删除的 compatibility residue
- 需要迁回 typed `SpatialPlan`
  对象的临时 helper 输入

而不能再定义
当前 owner truth。

### 4.3 `tl.blackhole_logical_buffer_tile_bridge_specs`
已删除，不再是 cleanup exception

task1/task2
相关文档必须统一写清楚：

- `tl.blackhole_logical_buffer_tile_bridge_specs`
  曾经是 cleanup
  允许存在的窄 temporary handoff，
  但 repo HEAD
  active chain
  已不再发布或消费它
- 它不是
  `SpatialPlan`
  / `TTProgram`
  / `ExecutableSpec`
  的长期表示
- 它也不是
  TT-Metal program /
  runtime contract
- 它不是新的 medium-term bridge layer

后续如果发现 logical tile /
live-value /
materialization
信息不足，
只能扩
`SpatialPlan`
显式对象和 validator，
不能恢复该 attr
作为 debug helper
或临时 bridge。

## 5. Validator 合同

`ValidateSpatialPlan`
是主链对象，
不是补丁。

它必须成为
下游 target planner
之前的正式 hard gate，
并且 fail-closed。

当前合同至少包括：

1. phase identity / index 唯一，
   且每个 `ExecutionUnit`
   必须恰好属于一个 `PhasePlan`
2. 每条 `DataflowEdge`
   的 producer / consumer / subject 完整，
   且 `crosses_phase`
   与 phase membership 一致
3. 每个 `LayoutSpec`
   引用的 subject
   都可回溯到
   `ExecutionUnit`
   或 anchored sub-TIR
4. `ExecutionUnit` /
   `DataflowEdge` /
   `LayoutSpec`
   的公开枚举和值域中
   不允许 TT noun 泄漏
5. `ValidatedHintSet`
   中的 accepted / rejected /
   diagnostics 自洽
6. logical live-value
   对象存在时，
   producer /
   consumer /
   materialization boundary
   必须引用已存在的
   `ExecutionUnit`
   /
   `DataflowEdge`
   /
   `PhasePlan`
   且不得包含 TT noun
7. 如果 compatibility projection
   仍暂时存在，
   它们必须和显式对象对齐，
   不能各自漂移

补充要求：

- validator 成功后，
  下游必须显式要求
  validated marker
- `BuildTTProgram`
  或其他 target planner
  不能绕过这个 gate

## 6. Construction / Lowering 边界

`SpatialPlan`
必须由当前
`Normalized Tile TIR`
直接构造。

允许的实现形态是：

- 同一实现单元里的
  visitor / matcher / builder
- pass-local collector / helper

明确禁止：

- public analysis wrapper
- pass-to-pass facts bag
- facts attr
- 新的 helper protocol
- 让 leaf-time matcher
  重新恢复
  `SpatialPlan`
  本应承载的语义

换句话说：

- public 构造入口
  只能保留
  `BuildSpatialPlan`
- 如果仍需要前置结构收集，
  它也只能留在
  `BuildSpatialPlan`
  同一个实现单元里
  作为局部 mechanics

不能把这些 implementation detail
写成新的架构层。

## 7. 下游消费边界

下游 target planner
只允许消费：

- validated `SpatialPlan`
  的显式对象
- anchored sub-TIR

不允许：

- 从 fake protocol
  恢复中间层语义
- 重新制造伪中间层 bag
- 从 runtime / codegen /
  executable reader
  反推上游 planning 语义

`TTProgram`
是下一层 owner truth。

它负责承接：

- work decomposition
- block / core placement
- kernel family / role
- transport / routing
- sync / completion
- ABI / runtime args
- execution / launch order

这些语义都不属于
`SpatialPlan`。

它们后续会落到
显式的 target-side
program / workload
materialization 边界，
而不是反向倒灌回
virtual spatial/dataflow 层。

## 8. 历史 Surface 的落点

具体显式名和 disposition
统一看协议审计表。

表示层落点固定为：

- compute-region-like 信息
  - 落到
    `ExecutionUnit` coverage
    / `DataflowEdge`
    / anchored sub-TIR
- pipeline-stage-like 信息
  - 落到 `PhasePlan`
- copy/dataflow-like 信息
  - 落到 `DataflowEdge`
    与 TIR access semantics
- work-decomposition-like 信息
  - 不属于 `SpatialPlan`
  - 落到
    `TTBlockPlan`
    / `TTExecutionPlan`
- logical bridge spec-like 信息
  - 已经不能再以
    `tl.blackhole_logical_buffer_tile_bridge_specs`
    暂存
  - 它不是
    `SpatialPlan`
    的长期字段

## 9. Completion Contract

`Task 1`
只有在下面这些条件同时满足后
才算完成：

补充硬约束：

- `Task 1`
  默认按终态实现，
  不接受
  “先迁 `SpatialPlan` owner truth，
   后删旧 facts /
   wrapper /
   public surface”
  的过渡式收口
- 旧 `wrapper / facts / bag / public surface`
  只要仍在
  `Task 1`
  边界里活着，
  就算未完成

1. `SpatialPlan`
   已显式收成
   `ExecutionUnit /
    DataflowEdge /
    LayoutSpec /
    PhasePlan /
    ValidatedHintSet`
   这组 cleanup owner truth；
   后续 live-form /
   materialization
   support lane
   若需要跨阶段 logical live-value
   语义，
   还必须继续补齐
   `LiveValue /
    LiveValueEdge /
    MaterializationBoundary`
   一等对象
2. `ValidateSpatialPlan`
   已落地并成为
   下游 planner
   的正式前置 gate
3. public `BuildSpatialPlan`
   已成为
   `SpatialPlan`
   的唯一构造入口，
   且
   `AnalyzeSpatialStructureFacts` /
   `SpatialStructureFacts` /
   `tl.spatial_structure_facts` /
   `BuildSpatialPlanCompanion`
   已退出
   active chain
   和 public surface
4. `ExecutionClosure` /
   `ClosureBoundary`
   如果仍暂存，
   也只允许作为
   `SpatialPlan`
   内部 compatibility projection，
   不再承载 owner truth
5. `blackhole.work_decomposition` /
   `blackhole.compute_regions` /
   `blackhole.pipeline_stages`
   这类过渡 surface
   已经降成 migration residue，
   不能再定义
   `SpatialPlan`
   边界
6. 下游 target planner
   读取 validated
   `SpatialPlan`
   显式对象，
   而不是继续从
   fake protocol /
   bridge bag /
   leaf reader
   恢复
   virtual spatial/dataflow 语义

## 10. 与 Archived Cleanup Task 的关系

这份文档定义的是
长期 `SpatialPlan` 合同，
不是执行顺序说明。

archive 里的 cleanup task1/task2
只记录历史收口：

1. `SpatialPlan`
   已收成 direct builder；
   public wrapper /
   facts object /
   facts attr
   不能再控制 active chain
   或 public surface。
2. `SpatialPlan -> TTProgram`
   的边界必须保持在显式对象
   和 direct planner builder
   上；
   broad planning bag
   与 payload-style owner truth
   不能重新出现。
3. `tl.blackhole_logical_buffer_tile_bridge_specs`
   已从 active code path 删除；
   若未来重新出现，
   应视为 regression，
   不是 cleanup exception。

当前实现依赖或历史 archive 文档
不能削弱这个 verdict。
