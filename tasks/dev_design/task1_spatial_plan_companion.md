# Task 1: SpatialPlan Companion 与 Pass Owner 边界

## 基本信息

- **文档角色**: `Task 1` 的 `SpatialPlan companion` 与 pass owner 设计文档
- **当前状态**: `2026-04-13` 活动设计文档；`Task 1` 已落地，当前继续承接 `Task 2`
- **任务链位置**: `Task 1` 和 `Task 2` 的 schema / pass owner 设计文档
- **定位**: 不替代总体设计；只回答
  `Normalized Tile TIR -> SpatialPlan companion -> TTProgram companion`
  这条主链里，哪些 truth 留在 TIR，哪些只是 analysis facts，
  哪些必须持久化进 companion
- **适用范围**: spatial/dataflow 类后端；当前以 Blackhole 为第一落地点
- **非目标**:
  - 不把本文档变成第二份总体设计文档
  - 不独立于 TIR 发明一套完整的新语义表示
  - 不把 GPU/CPU 等所有 target 的 leaf lowering 统一进同一套表面 API

## 1. 设计原则

这轮设计最重要的约束只有一条：

> **TIR 已经完整表达的东西，不允许再在 Spatial/TT 层重复编码。**

因此主链现在应当理解为：

```text
Normalized Tile TIR
  -> SpatialPlan companion
  -> TTProgram companion
  -> ExecutableSpec / BlackholeModule
```

这里：

- `Normalized Tile TIR`
  是唯一语义本体
- `SpatialPlan companion`
  不是第二份 semantic IR；
  它只是从 TIR 中提炼出的最小 planning projection
- `TTProgram companion`
  不是第二份 compute IR；
  它只是 target block/resource/ABI/execution 的 companion

也就是说，后续不该再以“构造一张图来重新表示 TIR”作为目标，
而应以“只把 TIR 没有对象化、但 planning 必须跨 pass 共享的事实提出来”
作为目标。

## 2. 三类信息

### 2.1 留在 `Normalized Tile TIR` 的

这些东西本来就在 TIR 里完整表达，不能重复抬成 companion schema：

- loop/domain 结构
- index expr
- predicate / mask
- indirect address / page-table 风格访问
- tile-op 参数
- region/subscript 形态
- `actual_rows`、`group_idx_for_bx`、`block_table[...]`
  这类 sample-local expr

一句话说：
**TIR 继续拥有程序“怎么访问、怎么算”的全部细节。**

### 2.2 只存在于 analysis pass 内部的

这些是从 TIR 上临时推出来的事实，默认不持久化：

- def-use 追踪细节
- exact region overlap 证明
- predicate simplification 结果
- indirect access dependence 证明
- loop-carried recurrence 检测过程
- closure 候选枚举过程
- footprint 估算中间结果

这些都是算法材料，不是长期 schema。

### 2.3 必须进入 companion 的

只有一类东西需要进入 companion：

- 后续 planning 仍然要消费
- 但在进一步 lowering 后就无法再稳定恢复
- 并且它不是某一个 expr，而是跨 pass 共享的 planning truth

这类 truth 才应该进入 `SpatialPlan companion`
或 `TTProgram companion`。

## 3. `SpatialPlan companion`

### 3.1 角色

`SpatialPlan companion` 只负责回答 4 个问题：

1. 哪些 anchored sub-TIR 天然构成局部执行闭包
2. 闭包之间有哪些稳定 boundary
3. 哪些 frontier 允许 cut，哪些不允许
4. 用户 hints 在空间/dataflow 语义下经过验证后，变成了哪些 planner input

它**不**负责：

- 重复表达访问模式
- 重复表达 tile-op 语义
- 直接决定最终 block 大小
- 直接决定 CB/semaphore/kernel/runtime args

### 3.2 最小持久对象

`SpatialPlan companion` 只保留 3 类持久对象。

#### `ExecutionClosure`

表示一个局部执行闭包。

它必须回答：

- 哪些 `TIRAnchor` 属于这个闭包
- 哪些 frontier 允许 cut
- 这个闭包的 symbolic footprint summary
- 这个闭包是否带 locality / carry / aggregation obligation

它不需要重复存：

- index expr
- predicate
- access formula
- tile-op 参数

这些都在 anchored sub-TIR 里。

#### `ClosureBoundary`

表示两个闭包之间的稳定 planning boundary。

它只保留极少数正交关系：

- `source_closure`
- `target_closure`
- `subject`
  例如某个 logical state / logical buffer identity
- `boundary_kind`
  只允许最小集合：
  - `flow`
  - `carry`
  - `join`

这里故意不把 workload-specific 访问模式做成新 kind。
`topk`、`fusedmoe`、`paged decode`、`retention`
的访问细节继续留在 TIR expr 里；
boundary 只表达 planning 需要的关系。

#### `ValidatedHintSet`

表示从 DSL literal / imported hint / target-specific hint
收正后的 planner 输入。

它只负责记录：

- 哪些 imported hints 被接纳
- 哪些 portable hints 被 canonicalize
- 哪些 target-specific hints 被 validate 成功
- 哪些 hint 被 shrink / reject，以及诊断理由

它不替代 TIR，也不替代 target plan。

### 3.3 `Task / Channel` 的新地位

`Task / Channel` 仍然可以存在，但只作为 derived view：

- `Task`
  是 `ExecutionClosure` 的粗粒度展示/调试名词
- `Channel`
  是 `ClosureBoundary` 在 transport/materialization 视角下的展示/调试名词

它们不再是 primary truth owner。

### 3.4 为什么这已经够用

对我们已经核过的 workload：

- [example_topk.py](/root/dev/vibe_dsl/tilelang_repo/examples/topk/example_topk.py)
  需要的是 closure、flow boundary 和 join/carry 判断；
  predicate/index 细节本来就在 TIR 里
- [example_fusedmoe_tilelang.py](/root/dev/vibe_dsl/tilelang_repo/examples/fusedmoe/example_fusedmoe_tilelang.py)
  需要的是 irregular work closure 与 boundary；
  `actual_rows` 和 grouped offsets 继续留在 expr 里
- [example_mla_decode_paged.py](/root/dev/vibe_dsl/tilelang_repo/examples/deepseek_mla/example_mla_decode_paged.py)
  需要的是 split/join closure 边界；
  page-table 寻址继续留在 expr 里
- [example_retention_fwd.py](/root/dev/vibe_dsl/tilelang_repo/examples/linear_attention/example_retention_fwd.py)
  需要的是 carry boundary；
  `h` 的具体访问与更新公式继续留在 TIR 里

所以 `SpatialPlan companion` 真正需要补出来的不是“访问模式 schema”，
而是 **closure/boundary/cut/hint** 这几个 planning truth。

## 4. `TTProgram companion`

### 4.1 角色

`TTProgram companion` 只负责承接 target owner：

- block sizing / slicing
- kernel grouping
- transport realization
- synchronization realization
- ABI/runtime args
- execution order / wave scheduling

它不再承担：

- semantic recovery
- task graph recovery
- 从 lowered loop matcher 恢复 compute/transport intent

### 4.2 最小持久对象

`TTProgram companion` 长期只保留 6 类 primary owner object：

- `TTBlockPlan`
- `TTKernelPlan`
- `TTTransportPlan`
- `TTSyncPlan`
- `TTABIPlan`
- `TTExecutionPlan`

其中：

- `TTBlockPlan`
  是当前缺失的核心 owner，
  负责 slice、容量、buffering、work packet 形成
- `TTTransportPlan`
  负责把 boundary realization 成 local handoff、CB、multicast、
  cross-core exchange 等 target primitive
- `TTSyncPlan`
  负责 barrier / semaphore / completion relation

像 `CB`、`Semaphore`、`CoreGroup`
都更接近这些 plan 的 realization detail，
不应再和 target truth owner 混为一谈。

`TTProgram` 本身则是这 6 类 owner object 的稳定聚合结果。

### 4.3 与 TT-Metal 的对应关系

这和 TT-Metal 当前真实编程模型是对齐的：

- [host_api.hpp](/root/dev/vibe_dsl/tt_metal_repo/tt_metal/api/tt-metalium/host_api.hpp)
  表明 host 侧长期稳定原语就是
  `Program / Kernel / CircularBuffer / Semaphore / RuntimeArgs`
- [program_descriptors.hpp](/root/dev/vibe_dsl/tt_metal_repo/tt_metal/api/tt-metalium/program_descriptors.hpp)
  表明最终 program descriptor 也是
  `kernels / cbs / semaphores`
- [lab_multicast.cpp](/root/dev/vibe_dsl/tt_metal_repo/ttnn/examples/lab_multicast/lab_multicast.cpp)
  和
  [sort_program_factory.cpp](/root/dev/vibe_dsl/tt_metal_repo/ttnn/cpp/ttnn/operations/data_movement/sort/device/sort_program_factory.cpp)
  说明 multicast、cross-core exchange 也不是新语义层，
  而是 kernel/CB/semaphore/runtime args 的组合

所以 `TTProgram companion` 不需要发明更大的 ontology，
只需要把这些 target owner 收正。

## 5. Pass 设计

这套模型下，pass 不再围绕“构造另一份 IR”组织，
而是围绕“analysis -> companion -> target plan”组织。

### 5.1 `NormalizeTileFrontend`

保留现有真正属于 frontend 规范化的 pass：

- `BindTarget`
- `AddWrapperForSingleBufStore`
- `LegalizeNegativeIndex`
- `VerifyParallelLoop`
- `InjectAssumes`
- `Simplify`

这一步结束后：

- TIR 仍然保留 tile-op、loop、expr、predicate、region truth
- 还没有进入 GPU realization

### 5.2 `AnalyzeSpatialStructureFacts`

纯 analysis pass。

职责：

- 从 normalized TIR 中识别 closure candidates
- 识别 carry / join / flow boundary candidates
- 推导 admissible cut frontiers
- 估算 symbolic footprint
- 读取并验证 imported / portable / target-specific hints

它不写长期协议 attrs，
只产出 analysis facts。

当前已落地边界：

- 直接从 `Simplify` 后 normalized TIR 的 top-level executable statements
  提取 closure candidates
- 当前冻结的最小 summary 包括：
  `stmt_indices / read_buffers / write_buffers / execution_role / cut_frontiers`
- `ValidatedHintSet` 当前先以空集合落地；
  hint canonicalization / reject diagnostics 留给后续 widening
- `flow / carry / join` boundary 当前按 buffer identity def-use 关系建立，
  不在 companion 中重复编码 expr 细节

### 5.3 `BuildSpatialPlanCompanion`

把上一步 analysis facts 压缩成最小持久 companion：

- `ExecutionClosure`
- `ClosureBoundary`
- `ValidatedHintSet`

它的职责不是“重写 TIR”，
而是把 planning 所需、又不能留给后面再恢复的事实冻结下来。

当前已落地边界：

- 把 `SpatialStructureFacts`
  压成 `tl.spatial_plan`
- 不改写 TIR body
- 当前只建立 primary truth，
  不在这里再派生新的 `Task / Channel` owner

### 5.4 `PlanTTBlocks`

输入：

- normalized TIR
- spatial companion
- `TTHardwareModel`

职责：

- 选择 closure 的合法 slice
- 做 SRAM/L1/CB/buffering feasibility
- 形成 `TTBlockPlan`

这里才决定：

- 真正的 task/block 大小
- 是否沿某些 frontier 再切
- 每个 slice 的 symbolic footprint 如何落成 target 尺寸

### 5.5 `PlanTTTransport`

职责：

- 根据 `ClosureBoundary + TTBlockPlan`
  选择 local handoff / CB / cross-core exchange / multicast 等 realization
- 形成 `TTTransportPlan`

它不再从 lowered TIR 的 builtin 排列顺序恢复 transport intent。

### 5.6 `PlanTTSync`

职责：

- 根据 closure/boundary 和 transport choice
  形成 barrier / semaphore / completion relation
- 产出 `TTSyncPlan`

### 5.7 `PlanTTABI`

职责：

- 从 anchored TIR + `TTBlockPlan / TTTransportPlan / TTSyncPlan`
  推导 runtime args、accessors、compile-time args
- 形成 `TTABIPlan`

这里不允许再通过 `work_linear_id` 或 arg kind 去猜语义。

### 5.8 `PlanTTExecution`

职责：

- 形成 kernel grouping
- 形成 launch / wave / order
- 形成 `TTKernelPlan` 与 `TTExecutionPlan`

### 5.9 `BuildTTProgram`

职责：

- 聚合 `TTBlockPlan / TTKernelPlan / TTTransportPlan /
  TTSyncPlan / TTABIPlan / TTExecutionPlan`
- 构造稳定 `TTProgram companion`
- 不重新恢复 spatial semantics

### 5.10 `ValidateTTProgram`

职责：

- 检查 `TTProgram companion` 的 completeness / consistency
- 明确拒绝 legacy attrs、seed attrs、payload bag
  重新回升为 target truth

### 5.11 `MaterializeBlackholeExecutable`

职责：

- 只消费经过验证的 `TTProgram companion`
- 物化 leaf device kernels、Program/Kernel/CB/Semaphore descriptors、
  host/device packaging 与 `ExecutableSpec`
- 成为唯一稳态 writer

这里才允许使用 target leaf lowering。

## 6. 现有 pass 的长期归位

### 6.1 退出 active owner 链的

下面这些 pass 在长期主链里都不该再承担 owner-building 职责：

- `LayoutReducer`
- `LayoutInference`
- `CollectSemanticManifestSeeds`
- `LowerTileOp`
- `ProjectSemanticSeeds`
- `ProjectSemanticManifest`
- `AugmentSemanticManifest`
- `AnalyzeBlackholeWorkDecomposition`
- `AnalyzeBlackholeFragmentRegions`
- `AnalyzeBlackholePipelineStages`
- `AnalyzeSemanticStructure`
- `LiftStatefulSemanticIR`
- `AnalyzeSpatialDomainPlan`
- `AnalyzeSpatialExecutionPlan`
- `MaterializeSpatialProgram`
- `LowerBlackholeOps`
- `PlanBlackholeCB`
- `AssignBlackholeCores`
- `LowerSpatialProgramToTTTarget`
- `MaterializeTTExecutableSpec`

这些 pass 的问题域要么：

- 被新的 analysis/companion/TT planning 吸收

要么：

- 退成 leaf materialization 工具

### 6.2 被新主链吸收的问题域

- `PlanAndUpdateBufferAllocationLocation`
  的 lifetime / placement 问题域
  -> `PlanTTBlocks`
- `PipelinePlanning`
  的 overlap / buffering 问题域
  -> `AnalyzeSpatialStructureFacts + PlanTTBlocks`
- `SplitBlackholeKernel`
  的 kernel grouping 问题域
  -> `PlanTTExecution`
- `PlanBlackholeCB`
  的 CB 资源问题域
  -> `PlanTTTransport`
- `AssignBlackholeCores`
  的 mapping 问题域
  -> `PlanTTBlocks + PlanTTExecution`

### 6.3 后移到最终物化阶段的

- `AnnotateDeviceRegions`
- `CollectDevicePrograms`
- `SplitHostDevice`
- `AnnotateReadOnlyParams`
- `MakePackedAPI`
- `LowerDeviceKernelLaunch`

这些仍然有工程价值，
但只应该消费冻结后的 TT truth。

## 7. Sufficiency 结论

基于当前工作负载和 TT-Metal 能力面，
我现在的结论是：

1. 不需要再新增第三层 stable IR
2. 不需要独立于 TIR 再造一份 graph 语义体
3. 需要的只是：
   - minimal `SpatialPlan companion`
   - minimal `TTProgram companion`
   - analysis-first 的 pass 主链

换句话说：

- 当前设计的风险不在“层数不够”
- 而在“如果 companion 开始重复 TIR，就会重新膨胀”

所以后续所有 schema 和 pass 设计都必须继续 obey：

> **TIR 是 semantic body；companions 只保存 planning 必需但 TIR 没有对象化的事实。**

## 8. 验证与推进

推进顺序固定为：

1. 先按本文件收正 total design 和 supporting docs
2. 再实现：
   - `AnalyzeSpatialStructureFacts`
   - `BuildSpatialPlanCompanion`
   - 当前状态：已完成
3. 再实现：
   - `PlanTTBlocks`
   - `PlanTTTransport`
   - `PlanTTSync`
   - `PlanTTABI`
   - `PlanTTExecution`
4. 最后切掉旧 recovery 主链

验证口径至少覆盖：

- `flash-attn`
- `topk`
- `fusedmoe`
- `paged decode`
- `chunk recurrence`

判断标准不是“是不是能重建更多 payload”，
而是：

- 是否不再重复 TIR
- 是否不再依赖 late recovery
- 是否能支撑 TT target planning 与最终 runtime materialization

## 9. 与现有文档的分工

- `final_blackhole_backend_redesign.md`
  - 唯一总体设计
  - 只保留长期分层、主链、真源规则与 cutover 结论
- `task0_ir_layering_root_cause.md`
  - 根因诊断与方向收敛
- 本文档
  - 只负责 companion 边界、analysis/pass owner 与 schema 最小化原则
- `task2_ttprogram_companion_cutover.md`
  - 记录 `Task 2` 的 target owner cutover、materialization 边界与完成判定
- `task3_runtime_gate_and_workload_cutover.md`
  - 记录 `Task 3` 的 runtime gate、support surface 与 workload cutover
- `tasks/progress.md`
  - 继续记录当前切换状态与下一步
