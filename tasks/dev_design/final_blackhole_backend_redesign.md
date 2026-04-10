# TileLang Blackhole 后端重设计

## 基本信息

- **文档 ID**: `final_blackhole_backend_redesign`
- **日期**: `2026-04-10`
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
  -> SpatialGraph
  -> TTProgram
  -> ExecutableSpec / BlackholeModule
```

长期 stable IR 只有两层：

- `SpatialGraph`
- `TTProgram`

`ExecutableSpec / runtime` 只做最终物化，不再承担语义恢复。

## 2. 目标与硬约束

### 2.1 目标

1. 保持 TileLang Python DSL 主体写法基本稳定
2. 结束 late target-specific semantic guessing
3. 让 spatial/dataflow 主链围绕“计算、依赖、state、task、映射”建模，
   而不是围绕 GPU storage/layout noun 建模
4. 让 `Task / Channel` 继续存在，但退回 coarse execution grouping，
   不再是 primary truth owner
5. 让 `TTProgram` 只承接 target block/resource/ABI/execution truth
6. 让 codegen/runtime 回到 materialization 与 execution，
   不再承担 graph 恢复

### 2.2 当前硬约束

- `BlackholeModule` 进程内 direct host path 仍是唯一正式执行路径
- `ExecutableSpec` 仍是 runtime 消费的最终物化产物
- copy / GEMM / export 当前支持面不能回退
- 不引入第二条正式执行路径
- 不允许名字匹配、位置猜测、单 case matcher 进入长期协议
- 当前重设计必须建立在现有 Blackhole 主链上完成，不是 greenfield compiler

## 3. 权威架构

### 3.1 两层稳定 IR

- `SpatialGraph`
  - 只回答：tile 级计算本身是什么，哪些逻辑对象在流动，
    哪些 dependence / recurrence / partial merge 存在，
    以及哪些局部计算闭包应形成 virtual task
  - 真源：计算图、依赖图、state/carry/merge truth、
    task/channel 的抽象组织 truth
- `TTProgram`
  - 只回答：这个 spatial/dataflow 程序如何变成合法 TT contract
  - 真源：block/resource/CB/semaphore/core/ABI/execution truth
- `ExecutableSpec / BlackholeModule`
  - 只回答：冻结后的 TT contract 如何被物化并执行
  - 真源：materialized target schema

### 3.2 `SpatialGraph` 的 primary truth

`SpatialGraph` 的最小长期对象集合是：

- `WorkDomain`
- `LogicalObject`
- `ObjectVersion`
- `ComputeNode`
- `Access`
- `DependenceEdge`

设计纪律：

- `fragment/shared/layout_map/blackhole.acc/blackhole.cb`
  只能作为 frontend hint 或 target realization，不能作为第一层 IR 的 ontology
- `Task / Channel`
  不是 primary truth，而是 graph 上的 derived view

### 3.3 `Task / Channel` 的长期边界

- `Task`
  - 继续存在
  - 含义变成 graph 上的 coarse execution grouping
  - 在 `SpatialGraph` 层是 `VirtualTask`
  - 在 `TTProgram` 层才物化成具体 block/task instance
- `Channel`
  - 继续存在
  - 含义变成 graph boundary edge 的 transport/materialization grouping
  - 不再承载 producer-consumer 语义真源

### 3.4 `TTProgram` 的 primary truth

`TTProgram` 的长期 object set 以 target owner 为准：

- `TTBlockPlan`
- `TTKernel`
- `TTCoreGroup`
- `TTCBPlan`
- `TTTransportPlan`
- `TTSemaphorePlan`
- `TTABIPlan`
- `TTExecutionPlan`

其中 `TTBlockPlan` 是必须补齐的新 owner：

- `work_packets`
  不能直接挂在 `TTCoreGroup` 上兼职 decomposition owner
- `CB / semaphore / accessor / runtime args`
  不能再从 lowered TIR matcher 恢复
- `TTProgram`
  必须直接从 `SpatialGraph` 的 task/boundary truth 派生

## 4. 长期真源规则

1. 计算、依赖、state/carry/merge truth 只存在于 `SpatialGraph`
2. `Task / Channel` 只作为 graph 的 derived execution view 存在
3. TT 资源与 ABI 只存在于 `TTProgram`
4. `ExecutableSpec` 只由 `TTProgram` 物化，不是第二真源

## 5. 交接纪律

### 5.1 `Normalized Tile TIR -> SpatialGraph`

- 允许：
  - 从 structured tile TIR 直接抽取 graph
  - 建 `WorkDomain / LogicalObject / ObjectVersion / ComputeNode / Access / DependenceEdge`
  - 形成 `VirtualTask`
- 禁止：
  - 先经 `LayoutInference / LowerTileOp / semantic_manifest`
    再恢复 graph
  - 让 GPU storage/layout noun 成为 graph ontology

### 5.2 `SpatialGraph -> TTProgram`

- 允许：
  - task slicing
  - SRAM/L1 feasibility
  - block/core mapping
  - CB/transport/semaphore planning
  - ABI/runtime arg generation
- 禁止：
  - 发明新的算法依赖
  - 在 target 层重新切 task graph
  - 从 lowered TIR matcher 恢复 producer-consumer intent

### 5.3 `TTProgram -> ExecutableSpec / runtime`

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

新的 `SpatialGraph` 主链介入点固定在：

```text
BindTarget
  -> AddWrapperForSingleBufStore
  -> LegalizeNegativeIndex
  -> VerifyParallelLoop
  -> InjectAssumes
  -> Simplify
  -> ExtractSpatialGraph
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
  -> ExtractSpatialGraph
  -> NormalizeSpatialGraph
  -> PlanSpatialTasks
  -> PlanTTBlocks
  -> PlanTTResources
  -> PlanTTABI
  -> PlanTTExecution
  -> MaterializeBlackholeExecutable
```

### 6.3 当前 TileLang pass 的长期归位

保留在 `SpatialGraph` 之前的 frontend 规范化 pass：

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
- 但合入形态必须一次性切到两层主链
- 不保留长期双轨
- 新主链建成后，旧的三层 recovery 路线整体退场

### 7.2 Cutover 顺序

1. 建立 `SpatialGraph` / `VirtualTask` / `TTBlockPlan` schema
2. 实现 `ExtractSpatialGraph`
3. 实现 `NormalizeSpatialGraph + PlanSpatialTasks`
4. 实现 `PlanTTBlocks / PlanTTResources / PlanTTABI / PlanTTExecution`
5. 实现 `MaterializeBlackholeExecutable`
6. 在同一轮 cutover 中切掉旧链

### 7.3 Cutover invariants

- Blackhole active compile path 中不再出现 `tl.semantic_*` 主协议
- 不再写 `tl.fragment_layout_seeds`
- 不再依赖 `LowerTileOp` 后 matcher/recovery 恢复 task/transport/materialization
- `TTProgram` 必须拥有 `TTBlockPlan`
- 若 `ExtractSpatialGraph` 不能安全总结某段结构，必须 explicit unsupported，
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
- 两层链才是后续所有架构工作的唯一总设计

### 8.3 当前主 blocker

当前总体 blocker 已经从“某个后段 runtime protocol 缺失”
转成“需要把 active owner 链从 recovery 主链切到两层主链”。

具体说：

- 当前问题不再适合继续按
  `Fragment layout seed -> FragmentMaterializationInfo typed uplift -> LowerTileOp 时机修补`
  这种局部切片推进
- 后续工作应该直接围绕：
  - `ExtractSpatialGraph`
  - `PlanSpatialTasks`
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
    但后续需要按 `SpatialGraph / VirtualTask / TTBlockPlan`
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
