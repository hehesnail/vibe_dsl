# TileLang Blackhole 后端重设计

## 基本信息

- **文档 ID**: `final_blackhole_backend_redesign`
- **日期**: `2026-04-13`
- **状态**: 当前唯一权威总体设计文档
- **定位**: 轻量总纲；只保留长期架构、层间边界、真源规则、任务链与 cutover invariant
- **详细设计入口**:
  - `tasks/dev_design/task0_ir_layering_root_cause.md`
  - `tasks/dev_design/task1_spatial_plan_companion.md`
  - `tasks/dev_design/task2_ttprogram_companion_cutover.md`
  - `tasks/dev_design/task3_runtime_gate_and_workload_cutover.md`

## 1. 问题定义

Blackhole 当前的核心问题不是“还差一个 kernel emitter”，
而是主链在错误层次上承载 truth：

- `destroy-then-recover`
  - tile 级计算在 `LowerTileOp` 前后被打散
  - `Fragment/layout` truth 在 `OptimizeForTarget / SplitHostDevice` 后丢失
  - 后段再靠 `semantic_manifest`、`fragment_layout_seeds`、
    `LowerBlackholeOps::Match*` 恢复语义
- `enum-of-forms`
  - `BufferMaterializationInfo`、`companion_base.h`、
    `SpatialProgram.payload` 这类 stringly-typed schema
    让每扩一种 family 就多一轮 kind 字符串、reader 特判和 matcher
- `GPU realization leakage`
  - `fragment/shared/layout_map/blackhole.acc/blackhole.cb`
    这类对象本质上是 TileLang 现有 GPU realization noun
  - 它们不应成为 spatial/dataflow 主链的长期 ontology

这些问题已经被同一批 workload family 共同暴露出来：

- `flash-attn / online softmax / attention_sink`
- `topk / selection`
- `fusedmoe / grouped dispatch`
- `paged decode / sparse decode`
- `chunk recurrence / scan`

因此长期主链固定为：

```text
Normalized Tile TIR
  -> SpatialPlan companion
  -> TTProgram companion
  -> ExecutableSpec / BlackholeModule
```

这里不是“再造两层独立 semantic IR”，而是：

- `Normalized Tile TIR` 作为唯一语义本体
- `SpatialPlan companion` 作为第一层 planning companion
- `TTProgram companion` 作为第二层 target companion
- `ExecutableSpec / runtime` 只做最终物化，不再承担语义恢复

## 2. 长期目标与硬约束

### 2.1 目标

1. 保持 TileLang Python DSL 主体写法基本稳定
2. 结束 late target-specific semantic guessing
3. 明确 `TIR body / analysis facts / persistent companions` 的边界，
   不重复编码已有 TIR 信息
4. 让 `Task / Channel` 继续存在，但退回 coarse execution view，
   不再是 primary owner
5. 让 `TTProgram companion` 只承接 target block/resource/ABI/execution truth
6. 让 codegen/runtime 回到 materialization 与 execution，
   不再承担 planning recovery

### 2.2 当前硬约束

- `BlackholeModule` 进程内 direct host path 仍是唯一正式执行路径
- `ExecutableSpec` 仍是 runtime 消费的最终物化产物
- copy / GEMM / export 当前支持面不能回退
- 不引入第二条正式执行路径
- 不允许名字匹配、位置猜测、单 case matcher 进入长期协议
- 当前重设计必须建立在现有 Blackhole 主链上完成，不是 greenfield compiler
- 当前收口范围只针对 `Blackhole` active compile/runtime path；
  non-Blackhole backend 的统一化不属于当前 blocker
- 当前 cutover 以 internal owner chain / active path 的真实切换为完成标准；
  不把 public Python `transform` API 改名当作前置目标

## 3. 权威架构

### 3.1 `TIR + companion` 主链

- `Normalized Tile TIR`
  - 唯一语义本体
  - 保留 loop、expr、predicate、indirection、tile-op 参数、
    region/access 细节
- `SpatialPlan companion`
  - 只回答：哪些 anchored sub-TIR 构成局部执行闭包
  - 只回答：闭包之间有哪些稳定 boundary，哪些 frontier 允许 cut
  - 只回答：哪些 hints 经验证后成为 planner input
- `TTProgram companion`
  - 只回答：closure/boundary 在当前 target 上如何变成
    block/resource/transport/sync/ABI/execution truth
- `ExecutableSpec / BlackholeModule`
  - 只回答：冻结后的 TT contract 如何被物化并执行

### 3.2 `SpatialPlan companion` 的 primary truth

`SpatialPlan companion` 只保留最小 planning truth：

- `ExecutionClosure`
- `ClosureBoundary`
- `ValidatedHintSet`

设计纪律：

- `fragment/shared/layout_map/blackhole.acc/blackhole.cb`
  只能作为 frontend hint 或 target realization，不能作为 ontology
- index expr、predicate、indirection、tile-op 参数、访问公式
  继续留在 TIR 里，不在 companion 里重复编码
- `Task / Channel` 不是 primary truth，而是 derived view

### 3.3 `Task / Channel` 的长期边界

- `Task`
  - 继续存在
  - 含义变成 closure 上的 coarse execution grouping
  - 在 `SpatialPlan companion` 层只是 `ExecutionClosure` 的展示视图
  - 在 `TTProgram companion` 层才物化成具体 block/task instance
- `Channel`
  - 继续存在
  - 含义变成 closure boundary 的 transport/materialization grouping
  - 不再承载 producer-consumer 语义真源

### 3.4 `TTProgram companion` 的 primary truth

`TTProgram companion` 的长期 primary owner object set
以 target owner 为准：

- `TTBlockPlan`
- `TTKernelPlan`
- `TTTransportPlan`
- `TTSyncPlan`
- `TTABIPlan`
- `TTExecutionPlan`

其中 `TTBlockPlan` 是必须补齐的新 owner：

- `work_packets / core grouping`
  不能继续在其它对象上兼职 decomposition owner
- `CB / semaphore / accessor / runtime args`
  不能再从 lowered TIR matcher 恢复
- `TTProgram companion`
  必须直接从 spatial closure/boundary truth 派生
- `TTProgram`
  则是这组 owner object 的稳定聚合结果，
  不是第二层并列真源

## 4. 真源规则

1. 语义 body 只存在于 `Normalized Tile TIR`
2. `SpatialPlan companion` 只保存 TIR 未对象化、但 planning 必须持久化的 truth
3. `Task / Channel` 只作为 derived execution view 存在
4. TT 资源与 ABI 只存在于 `TTProgram companion`
5. `ExecutableSpec` 只由 `TTProgram companion` 物化，不是第二真源
6. 若某类 truth 无法从现有结构稳定获得，
   要么扩 IR/schema，要么 explicit unsupported；
   不允许回退到 late recovery

## 5. 介入层次与 canonical 主链

### 5.1 介入层次

新的 spatial planning 主链介入点固定在：

```text
BindTarget
  -> AddWrapperForSingleBufStore
  -> LegalizeNegativeIndex
  -> VerifyParallelLoop
  -> InjectAssumes
  -> Simplify
  -> AnalyzeSpatialStructureFacts
  -> BuildSpatialPlanCompanion
```

也就是当前活跃主链里：

- 在 `LayoutReducer / LayoutInference` 之前
- 在 `SplitBlackholeKernel / device-side realization` 之前

原因：

- 到 `Simplify` 为止，tile 级计算、structured loop、region truth 仍然活着
- `LayoutReducer / LayoutInference` 已经开始往 GPU realization 收
- `LowerTileOp` 是第一个真正把 tile graph 打散的断点

### 5.2 Canonical pass chain

```text
Frontend Tile TIR
  -> NormalizeTileFrontend
  -> AnalyzeSpatialStructureFacts
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

这条 canonical chain 的语义是：

- 它是 `Blackhole` 的长期 internal owner chain
- 它不要求当前仓库同步完成 repo-wide frontend 统一
- 它也不要求 public pass API 先一步改名才能推进 cutover

详细 pass owner、旧 pass 归位与 supporting schema，
统一维护在 `tasks/dev_design/task1_spatial_plan_companion.md`。

## 6. 任务链

当前架构推进统一按下面 4 个任务组织：

### `Task 0`: 文档与 owner 链收口

- 统一总纲、进度、索引和 supporting design 的角色
- 不再让旧三层链和新 companion 链并列叙述
- 让后续工作都按同一条任务链推进

### `Task 1`: `SpatialPlan companion` cut-in

- 在 `Simplify` 后引入
  `AnalyzeSpatialStructureFacts -> BuildSpatialPlanCompanion`
- 建立 `ExecutionClosure / ClosureBoundary / ValidatedHintSet`
- 让 `Task / Channel` 退回 derived view
- 当前落地状态：
  `AnalyzeSpatialStructureFacts -> BuildSpatialPlanCompanion`
  已接入 active Blackhole compile path；
  `Task 2` cutover 与 `Task 3A` semantic-layer deletion
  均已完成，当前 active path 直接从
  `Normalized Tile TIR + SpatialPlan companion +
  blackhole.work_decomposition / blackhole.compute_regions /
  blackhole.pipeline_stages`
  进入 `SpatialProgram / TTProgram` owner 链

### `Task 2`: `TTProgram companion` cutover

- 建立
  `PlanTTBlocks -> PlanTTTransport -> PlanTTSync -> PlanTTABI ->
  PlanTTExecution -> BuildTTProgram -> ValidateTTProgram ->
  MaterializeBlackholeExecutable`
- 补齐 `TTBlockPlan`
- 让 `TTProgram companion` 成为唯一 target truth
- 让 `MaterializeBlackholeExecutable` 成为唯一 writer
- 让 target owner 从旧 recovery/materialization 路线切到 companion 主链

### `Task 3`: 旧 recovery 链退场与 workload 回归

- 在同一轮 cutover 中退场
  `semantic_manifest / SemanticProgram / 旧 SpatialProgram /
  matcher-driven LowerBlackholeOps` 主路线
- 用新主链重新承接
  `flash-attn / topk / fusedmoe / paged decode / chunk recurrence`
- 兑现 runtime gate、wider family 与更宽 copy/dataflow/sync 支持面

### 当前优先级

当前设计结论下，执行优先级固定为：

1. `Task 3A`: 删除 persistent
   `SemanticProgram / Stateful Semantic IR`
   这一层旧 companion
   - 已完成；active path 已直接从
     `Normalized Tile TIR + SpatialPlan companion + Blackhole analysis facts`
     进入 `SpatialProgram / TTProgram` owner 链
   - semantic pass / wrapper / 过期测试 / dead code
     已完成删除
2. `Task 3B`: 在新主链上收 runtime gate，
   再兑现 `flash-attn` admitted subset payoff
3. `Task 3C`: 再扩 wider family / support surface
   - `topk -> fusedmoe -> paged decode -> chunk recurrence`
   - wider copy/dataflow
   - wider sync 最后进入 admitted surface

## 7. Cutover invariant

- 合入形态必须一次性切到 `TIR + two companions` 主链
- 不保留长期双轨
- Blackhole active compile path 中不再出现 `tl.semantic_*` 主协议
- 不再依赖 `LowerTileOp` 后 matcher/recovery 恢复 task/transport/materialization
- `TTProgram companion` 必须拥有 `TTBlockPlan`
- 若 `AnalyzeSpatialStructureFacts` 不能安全总结某段结构，
  必须 explicit unsupported，不能回退到旧 recovery 路线

当前代码基线、当前 blocker、当前支持面与验证快照，
统一维护在 `tasks/progress.md`；
根因细节和 pass 细节分别维护在 supporting docs。

## 8. 覆盖要求

新的两层主链不能只用 copy / GEMM 证明自己。
compile-path 与后续 runtime/correctness 设计至少要覆盖：

- `flash-attn`
- `topk`
- `fusedmoe`
- `paged decode`
- `chunk recurrence`

同时必须保证 copy / GEMM / export 当前正式支持面不回退。
