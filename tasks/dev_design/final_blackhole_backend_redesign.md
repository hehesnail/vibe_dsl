# TileLang Blackhole 后端重设计

## 基本信息

- **文档 ID**: `final_blackhole_backend_redesign`
- **日期**: `2026-04-13`
- **状态**: 当前唯一权威总体设计文档
- **定位**: 轻量总纲；只保留长期架构、层间边界、真源规则、cutover 判断与当前阶段结论
- **当前设计入口**:
  - `tasks/dev_design/ir_layering_root_cause_and_direction.md`
  - `tasks/dev_design/spatial_dataflow_program_model.md`
  - `tasks/dev_design/stage4_phase_c_tt_target_ir.md`
- **历史阶段参考**:
  - `tasks/dev_design/stage4_phase_b_spatial_ir.md`
  - `tasks/dev_design/stage4_phase_a_semantic_ir.md`
  - `tasks/dev_design/stage4_stage0_guardrails.md`

## 1. 问题定义

Blackhole 当前的核心问题不是“还差一个 kernel emitter”，而是主链在错误层次上
承载 truth：

- `destroy-then-recover`
  - tile 级计算先被 `LowerTileOp` 打散成 loop/buffer/intrin 形态
  - `Fragment/layout` truth 先在 `OptimizeForTarget / SplitHostDevice` 中丢失
  - 后段再靠 `semantic_manifest`、`fragment_layout_seeds`、`LowerBlackholeOps::Match*`
    恢复语义
- `enum-of-forms`
  - `FragmentMaterializationInfo`、`companion_base.h`、`SpatialProgram.payload`
    这类 stringly-typed schema 让每扩一种 family 就多一轮 kind 字符串、
    reader 特判和 matcher
- `GPU realization leakage`
  - `fragment/shared/layout_map/blackhole.acc/blackhole.cb`
    这类对象本质上是 TileLang 现有 GPU/realization 层概念
  - 它们不应该成为 spatial/dataflow 主链的长期 ontology

这已经被多类 family 共同暴露出来：

- `flash-attn / online softmax / attention_sink`
- `topk / selection`
- `fusedmoe / grouped dispatch`
- `paged decode / sparse decode`
- `chunk recurrence / scan`

因此当前总体结论只有一条：

```text
Normalized Tile TIR
  -> SpatialPlan companion
  -> TTProgram companion
  -> ExecutableSpec / BlackholeModule
```

长期稳定主链不是“再造两层独立 semantic IR”，而是：

- `Normalized Tile TIR` 作为唯一语义本体
- `SpatialPlan companion` 作为第一层 planning companion
- `TTProgram companion` 作为第二层 target companion

`ExecutableSpec / runtime` 只做最终物化，不再承担语义恢复。

## 2. 目标与硬约束

### 2.1 目标

1. 保持 TileLang Python DSL 主体写法基本稳定
2. 结束 late target-specific semantic guessing
3. 明确 `TIR body / analysis facts / persistent companions`
   的边界，避免重复编码已有 TIR 信息
4. 让 `Task / Channel` 继续存在，但退回 coarse execution view，
   不再是 owner
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

## 3. 权威架构

### 3.1 `TIR + companion` 主链

- `Normalized Tile TIR`
  - 唯一语义本体
  - 保留 loop、expr、predicate、indirection、tile-op 参数、
    region/access 细节
- `SpatialPlan companion`
  - 只回答：哪些 anchored sub-TIR 构成局部执行闭包，
    闭包之间有哪些稳定 boundary，哪些 frontier 允许 cut，
    以及 hints 被验证后变成了哪些 planner input
- `TTProgram companion`
  - 只回答：这些 closure/boundary 在当前 target 上如何变成
    block/resource/transport/sync/ABI/execution truth
- `ExecutableSpec / BlackholeModule`
  - 只回答：冻结后的 TT contract 如何被物化并执行
  - 真源：materialized target schema

### 3.2 `SpatialPlan companion` 的 primary truth

`SpatialPlan companion` 不复制 TIR 语义，只保留最小 planning truth：

- `ExecutionClosure`
- `ClosureBoundary`
- `ValidatedHintSet`

设计纪律：

- `fragment/shared/layout_map/blackhole.acc/blackhole.cb`
  只能作为 frontend hint 或 target realization，不能作为 ontology
- index expr、predicate、indirection、tile-op 参数、访问公式
  继续留在 TIR 里，不在 companion 里重复编码
- `Task / Channel`
  不是 primary truth，而是 derived view

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

`TTProgram companion` 的长期 object set 以 target owner 为准：

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

## 4. 长期真源规则

1. 语义 body 只存在于 `Normalized Tile TIR`
2. `SpatialPlan companion` 只保存 TIR 未对象化、但 planning 必须持久化的 truth
3. `Task / Channel` 只作为 derived execution view 存在
4. TT 资源与 ABI 只存在于 `TTProgram companion`
5. `ExecutableSpec` 只由 `TTProgram companion` 物化，不是第二真源

## 5. 交接纪律

### 5.1 `Normalized Tile TIR -> SpatialPlan companion`

- 允许：
  - 从 structured tile TIR 直接抽取 closure / boundary / validated hints
  - 建 `ExecutionClosure / ClosureBoundary / ValidatedHintSet`
- 禁止：
  - 先经 `LayoutInference / LowerTileOp / semantic_manifest`
    再恢复 planning truth
  - 让 GPU storage/layout noun 成为 ontology
  - 把 expr/predicate/indirection 再复制一份到 companion

### 5.2 `SpatialPlan companion -> TTProgram companion`

- 允许：
  - task slicing
  - SRAM/L1 feasibility
  - block/core mapping
  - CB/transport/semaphore planning
  - ABI/runtime arg generation
- 禁止：
  - 发明新的算法依赖
  - 在 target 层重新创造 closure/boundary truth
  - 从 lowered TIR matcher 恢复 producer-consumer intent

### 5.3 `TTProgram companion -> ExecutableSpec / runtime`

- 允许：
  - host/device packaging
  - launch emission
  - final executable materialization
- 禁止：
  - semantic recovery
  - protocol patching
  - target fallback guessing

## 6. 介入层次与 pass 重写

### 6.1 介入层次

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
- 在 `CollectSemanticManifestSeeds / LowerTileOp` 之前

原因：

- 到 `Simplify` 为止，tile 级计算、structured loop、region truth 仍然活着
- `LayoutReducer / LayoutInference`
  已经开始往 GPU realization 收
- `LowerTileOp`
  是第一个真正把 tile graph 打散的断点

### 6.2 新主链

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
  -> MaterializeBlackholeExecutable
```

### 6.3 当前 TileLang pass 的长期归位

保留在 planning companion 之前的 frontend 规范化 pass：

- `LetInline`
- `AddWrapperForSingleBufStore`
- `LegalizeNegativeIndex`
- `VerifyParallelLoop`
- `InjectAssumes`
- `Simplify`

问题域保留、但实现吸收到新 IR pass 的：

- `IfStmtBinding`
- `PlanAndUpdateBufferAllocationLocation`
- `PipelinePlanning`
- `InjectSoftwarePipeline`
- `MergeIfStmt`

应整体退出 spatial/dataflow 主链的 GPU/SIMT realization pass：

- `LayoutReducer`
- `LayoutInference`
- `LowerTileOp`
- `LowerL2Persistent`
- `LegalizeVectorizedLoop`
- `LegalizeSafeMemoryAccess`
- `LowerAccessPtr`
- `LowerSharedBarrier`
- `LowerSharedTmem`
- `FlattenBuffer`
- `StorageRewrite`
- `LowerThreadAllreduce`

应整体退出 active owner 链的 recovery/companion pass：

- `CollectSemanticManifestSeeds`
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

仅作为最终 packaging/materialization 保留并后移的：

- `AnnotateDeviceRegions`
- `CollectDevicePrograms`
- `SplitHostDevice`
- `AnnotateReadOnlyParams`
- `MakePackedAPI`
- `LowerDeviceKernelLaunch`

## 7. Cutover 设计

### 7.1 Cutover 原则

- 开发顺序可以分层推进
- 但合入形态必须一次性切到 `TIR + two companions` 主链
- 不保留长期双轨
- 新主链建成后，旧的三层 recovery 路线整体退场

### 7.2 Cutover 顺序

1. 建立 `ExecutionClosure / ClosureBoundary / ValidatedHintSet / TTBlockPlan` schema
2. 实现 `AnalyzeSpatialStructureFacts`
3. 实现 `BuildSpatialPlanCompanion`
4. 实现 `PlanTTBlocks / PlanTTTransport / PlanTTSync / PlanTTABI / PlanTTExecution`
5. 实现 `MaterializeBlackholeExecutable`
6. 在同一轮 cutover 中切掉旧链

### 7.3 Cutover invariants

- Blackhole active compile path 中不再出现 `tl.semantic_*` 主协议
- 不再写 `tl.fragment_layout_seeds`
- 不再依赖 `LowerTileOp` 后 matcher/recovery 恢复 task/transport/materialization
- `TTProgram companion` 必须拥有 `TTBlockPlan`
- 若 `AnalyzeSpatialStructureFacts` 不能安全总结某段结构，必须 explicit unsupported，
  不能回退到旧 recovery 路线

## 8. 当前阶段判断

### 8.1 当前代码基线

当前已交付的执行基线仍然是旧主链：

```text
Stateful Semantic IR
  -> Spatial Program IR
  -> TT Target IR
```

这条线当前仍承担：

- `ExecutableSpec -> rt_mod_blackhole -> BlackholeModule`
  的正式 direct host path
- copy / GEMM / export 当前支持面
- `flash-attn` 等已打通 regression 基线

### 8.2 当前架构结论

旧三层链不再代表长期总设计。

从本文件起：

- 三层链只代表“当前代码基线/历史实现状态”
- `Normalized Tile TIR + SpatialPlan companion + TTProgram companion`
  才是后续所有架构工作的唯一总设计

### 8.3 当前主 blocker

当前总体 blocker 已经从“某个后段 runtime protocol 缺失”
转成“需要把 active owner 链从 recovery 主链切到两层主链”。

具体说：

- 当前问题不再适合继续按
  `Fragment layout seed -> FragmentMaterializationInfo typed uplift -> LowerTileOp 时机修补`
  这种局部切片推进
- 后续工作应该直接围绕：
  - `AnalyzeSpatialStructureFacts`
  - `BuildSpatialPlanCompanion`
  - `TTBlockPlan`
  - `MaterializeBlackholeExecutable`
  这条新主链展开

## 9. 验证口径

新的两层主链不能只用 copy / GEMM 证明自己。
compile-path 与后续 runtime/correctness 设计必须至少覆盖：

- `flash-attn`
- `topk`
- `fusedmoe`
- `paged decode`
- `chunk recurrence`

它们共同覆盖：

- explicit tile op
- region compute
- selection/index state
- routed/grouped work
- split partial merge
- recurrence/carry

## 10. 当前文档分工

- `final_blackhole_backend_redesign.md`
  - 唯一总体设计
  - 只保留长期架构、层间边界、真源规则、cutover 判断与阶段结论
- `ir_layering_root_cause_and_direction.md`
  - 两层 IR 设计的根因诊断、事实依据与方向收敛文档
- `spatial_dataflow_program_model.md`
  - 继续承担 spatial/dataflow program model 细化设计，
    且已按
    `TIR body / SpatialPlan companion / TTProgram companion`
    边界重写
- `stage4_phase_c_tt_target_ir.md`
  - 作为当前已落地 `TTProgram` 基线、支持面与 gate 的实现参考保留；
    不再承担总体 layering 权威
- `stage4_phase_b_spatial_ir.md`
  - 已完成的旧 `SpatialProgram` 边界文档；作为历史实现边界保留
- `stage4_phase_a_semantic_ir.md`
  - 已完成的旧 `SemanticProgram` 边界文档；作为历史实现边界保留
- `tasks/progress.md`
  - 当前代码基线、设计切换状态与下一步
- `archive/`
  - 历史审计快照、旧实施计划与已退场叙述；不再作为当前任务入口
