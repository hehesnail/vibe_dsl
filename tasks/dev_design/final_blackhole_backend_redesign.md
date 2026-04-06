# TileLang Blackhole 后端重设计

## 基本信息

- **文档 ID**: `final_blackhole_backend_redesign`
- **日期**: 2026-03-19（创建），2026-04-03（重写并收敛为当前版本），2026-04-06（在 `Phase A` 完成后按当前分阶段结构再次精简，并完成一轮状态对齐审计）
- **状态**: 当前唯一权威总体设计文档
- **定位**: 常青总纲；只保留长期架构、层间边界、单一真源与 cutover 原则
- **阶段文档**:
  - `tasks/dev_design/stage4_stage0_guardrails.md`
  - `tasks/dev_design/stage4_phase_a_semantic_ir.md`
  - `tasks/dev_design/stage4_phase_a_formalization_note.md`
  - `tasks/dev_design/stage4_phase_b_spatial_ir.md`
  - `tasks/dev_design/stage4_phase_c_tt_target_ir.md`

## 1. 问题定义

Blackhole 当前面对的核心问题，不是“再打印一个 TT-Metal kernel 字符串”，而是：

- 复杂前端计算语义
- spatial/dataflow 程序结构
- TT target 资源与 ABI

这三类 truth 长期混在一个层里。

这个问题已经被多类 workload family 共同暴露出来：

- `flash-attn / online softmax / attention_sink`
  - `carry / normalized recurrence / reduction-update`
- `topk / selection`
  - `selection / index generation / selected subset`
- `fusedmoe / grouped dispatch`
  - `routed / grouped / ragged dispatch`
- `paged decode / sparse decode`
  - `paged / indexed / sparse access`
- `chunk recurrence / scan`
  - `cross-step carry / ordered recurrence / chunk state`

因此当前总论点非常简单：

> 编译器不能继续在一个层里同时做 semantic recovery、spatial organization 和 TT target planning。

必须改成三层 compiler-internal IR：

```text
Stateful Semantic IR
  -> Spatial Program IR
  -> TT Target IR
```

每层只承接自己的语义真相，下层只能消费上层冻结后的事实，不能反向猜测上层。

## 2. 设计目标与非目标

### 2.1 目标

1. 保持 TileLang Python DSL 主体写法基本稳定。
2. 结束 late target-specific semantic guessing。
3. 让 domain、state、update、task、layout、sync、TT resource、ABI 各自在正确层里成为一等对象。
4. 让 codegen/runtime 回到 materialization 与 execution，而不是继续承担语义重建。
5. 让这套设计不仅服务 `flash-attn`，也能覆盖 selection、routing、paged decode、chunk recurrence 等 workload family。

### 2.2 可证明性边界

本设计不承诺下面这个强命题：

- 固定不变的 `Stateful Semantic IR` vocabulary 足以覆盖任意未来 workload family

本设计只追求更弱、也更正确的命题：

- 对一个有限的 semantic core，如果某类 workload 的算法语义可以归约到这套 core，
  则 `Stateful Semantic IR` 可以作为该 workload family 的有界抽象域

因此这里的“通用性”指的是：

1. 长期 vocabulary 保持小闭集
2. 新增 workload 时先证明是否可归约到现有 core
3. 只有跨 family 复用、且无法归约到现有 core 的新语义轴，才允许扩 semantic core
4. task/layout/sync/placement/transport/ABI 这类信息必须进入 `Spatial Program IR` 或 `TT Target IR`

### 2.3 非目标

1. 不设计 TT-Metal 专用用户 DSL。
2. 不把 `task / channel / CB / semaphore / runtime_args` 暴露成 Python 前端一等概念。
3. 不把全部复杂度重新塞回一个 super IR。
4. 不引入第二条正式执行路径；当前正式路径仍是 direct host path。
5. 不为了单个 consumer 固化协议或 matcher。

## 3. 当前硬约束

这次重设计不是 greenfield compiler，而是在现有 Blackhole 主链上重构边界。

1. `BlackholeModule` 进程内 direct host path 仍是唯一正式执行路径。
2. `ExecutableSpec` 仍是 runtime 消费的最终物化产物。
3. copy / GEMM / export 当前支持面必须保持不回退。
4. 现有 recovery-oriented analysis pass 与 manifest capture 路径仍然是 semantic recovery 的起点：
   - `AnalyzeBlackholeWorkDecomposition`
   - `AnalyzeBlackholeFragmentRegions`（当前已退化为 compatibility fallback / residual reduction evidence）
   - `AnalyzeBlackholePipelineStages`
   - `CollectSemanticManifestSeeds -> ProjectSemanticManifest -> AugmentSemanticManifest`
5. `PlanBlackholeCB`、`AssignBlackholeCores`、`rt_mod_blackhole` 在迁移期间仍保留，但它们的长期职责必须收回到 target/runtime 边界。

## 4. 权威架构

### 4.1 总流程

```text
TileLang DSL / Python
  -> PrimFunc / TIR
  -> Semantic Recovery
  -> Stateful Semantic IR
  -> Spatialization
  -> Spatial Program IR
  -> Hardware-Aware Mapping
  -> TT Target IR
  -> MaterializeTTExecutableSpec
  -> Codegen / rt_mod_blackhole / BlackholeModule
```

### 4.2 各层回答的问题

| 层 | 它回答的问题 | 真源 | 稳态产物 |
|----|--------------|------|----------|
| `PrimFunc / TIR` | 用户和通用 lowering 表达了什么计算结构？ | 通用 TileLang / TVM IR | 规范化 TIR |
| `Stateful Semantic IR` | 程序在逻辑域上如何更新算法状态？ | 算法语义 | `SemanticProgram` |
| `Spatial Program IR` | 这个算法如何组织成 spatial/dataflow 程序？ | task/channel/layout/sync/work truth | `SpatialProgram` |
| `TT Target IR` | 这个 spatial program 如何变成合法 TT contract？ | TT 资源与 ABI 合约 | `TTProgram` |
| `ExecutableSpec / runtime` | 冻结后的 TT contract 如何被物化并执行？ | 目标物化 schema | `ExecutableSpec` / host objects |

### 4.3 工作负载覆盖边界

当前总体设计面向的 family 是：

| family | Semantic 层必须表达 | Spatial 层必须表达 | TT 层必须冻结 |
|--------|----------------------|--------------------|---------------|
| Dense tiled compute | tile domain、tensor state、map/reduce update | load/compute/store task、layout、partition | reader/compute/writer、CB、ABI、placement |
| Selection / indexing | selection update、index-valued state | select task、index channel、selected-subset partition | index scratch、selector runtime ABI |
| Routed / grouped / ragged dispatch | remapped domain、segmented/indirect access、expert/index state | route/compute/combine task、grouped layout、expert partition | routed buffer、dispatch ABI、core-group mapping |
| Paged / indexed sparse decode | paged access、carry state、merge update | page-stream task、paged layout、split/merge boundary | page/index descriptors、transport、execution plan |
| Stateful reduction-update | carry state、normalized recurrence、predicate-bound domain | update task、carry channel、ordered completion sync | dst layout、persistent carry、kernel contract |
| Chunked recurrence / scan | cross-step state、chunk domain、ordered recurrence | chunk task graph、chunk partition、state carry | persistent carry、dst/CB realization、runtime chunk descriptors |

这张表是总设计的覆盖声明；具体 gate 和分阶段实现以阶段文档为准。

## 5. 三层 IR 的核心合同

### 5.1 `Stateful Semantic IR`

这一层只回答：

- 程序在逻辑域上如何更新算法状态

长期 core 只保留：

- `SemanticProgram`
- `Domain`
- `State`
- `Update`
- `AccessMap`
- `UpdateLaw`
- `SemanticSupplement`

关键纪律：

1. semantic core 必须保持小闭集
2. analysis evidence 不是 vocabulary
3. evidence 必须可归约到 core
4. 不允许名字匹配恢复语义
5. 这一层不承接 task/layout/sync/placement/transport/ABI

`Phase A` 当前已经完成；实现边界和当前状态见：

- `tasks/dev_design/stage4_phase_a_semantic_ir.md`
- `tasks/dev_design/stage4_phase_a_formalization_note.md`

### 5.2 `Spatial Program IR`

这一层只回答：

- 这个算法应该如何组织成 spatial/dataflow 程序

长期 core 只保留：

- `SpatialProgram`
- `ProgramPhase`
- `Task`
- `Channel`
- `Layout`
- `WorkPartition`
- `Placement`
- `SyncEdge`
- `ResourceIntent`

关键纪律：

1. 只消费冻结后的 semantic truth
2. `ProgramPhase` 的 cross-function 真相固定挂在 `tl.device_programs`
3. analysis 决定 legality，policy 只在合法空间内选择
4. 不允许回头发明 semantic truth
5. 不允许让 `Task:TTKernel = 1:1` 退化成隐式默认
6. 这层必须是 execution-bearing contract，不允许退化成结构化 summary
7. `Task / Channel / Layout / WorkPartition / ProgramPhase / SyncEdge` 必须冻结
   task formation、state/data flow、domain remap/partition、phase boundary 与 ordering
   这些执行相关但非 TT-specific 的 truth
8. 如果 `Phase C` 需要某个 non-TT-specific truth 才能合法 mapping，
   那个 truth 必须先进入 `Spatial Program IR`，不能在 target translator 里临时恢复

`Phase B` 的详细对象边界和实施计划见：

- `tasks/dev_design/stage4_phase_b_spatial_ir.md`

### 5.3 `TT Target IR`

这一层只回答：

- 这个 spatial program 如何变成合法且稳定的 TT contract

长期 core 只保留：

- `TTProgram`
- `TTKernel`
- `TTCoreGroup`
- `TTCBPlan`
- `TTTransportPlan`
- `TTSemaphorePlan`
- `TTComputeSyncPlan`
- `TTDstLayoutPlan`
- `TTABIPlan`
- `TTExecutionPlan`
- `TTHardwareModel`

关键纪律：

1. `TTProgram` 是 target contract 真源
2. common-runtime ABI 必须是一等对象
3. hardware model 必须是 typed object，而不是散落常量
4. `MaterializeTTExecutableSpec` 是唯一稳态 writer
5. runtime/codegen 不得继续补 target contract

`Phase C` 的详细对象边界、cutover 与 deletion gates 见：

- `tasks/dev_design/stage4_phase_c_tt_target_ir.md`

## 6. 层间不变量

### 6.1 真源规则

1. 算法语义只存在于 `Stateful Semantic IR`
2. 空间组织只存在于 `Spatial Program IR`
3. TT 资源与 ABI 只存在于 `TT Target IR`
4. `ExecutableSpec` 只由 `TT Target IR` 物化，不是第二真源

### 6.2 交接契约

| From | To | 必须交付的契约 | 允许做的决策 | 明确禁止 |
|------|----|----------------|--------------|----------|
| `Semantic Recovery` | `Stateful Semantic IR` | `Domain / State / Update / AccessMap / UpdateLaw` 事实 | 对象化与冻结 | 泄漏 TT resource 事实 |
| `Stateful Semantic IR` | `Spatial Program IR` | 冻结后的算法语义 | 构造 task/channel/layout/sync/work | 改变语义含义 |
| `Spatial Program IR` | `TT Target IR` | 冻结后的空间结构 | TT mapping、resource planning、ABI 定义 | 发明新的 task graph 或 semantic update/access law |
| `TT Target IR` | `ExecutableSpec / runtime` | 冻结后的 TT contract | API materialization 与 launch emission | semantic recovery 或 protocol patching |

### 6.3 Companion 生命周期

1. semantic lift 之后，companion IR 默认进入 hard-freeze 管理
2. post-lift pass 只能显式属于：
   - `preserve`
   - `typed_rebind`
   - `invalidate`
3. unsafe mutation 后必须整体删除并重建：
   - `tl.semantic_structure`
   - `tl.semantic_witnesses`
   - `tl.semantic_program`
   - `tl.spatial_program`
   - `tl.tt_program`
4. materialized `blackhole.*` attrs 不是上游 IR 的真源；它们只能整体重建，不能被下游 patch 成第二真源

### 6.4 禁止反向推断

下面这些行为明确禁止：

1. 用 `CB / dst layout / runtime args` 反推 state semantics
2. 用 TT kernel 名字反推 task graph
3. 让 runtime 补丢失的 sync 或 carry strategy
4. 因为 backend 需要 `task / channel / semaphore`，就把它们直接暴露成 Python DSL 表面概念

## 7. 当前执行状态

### 7.1 阶段状态

- **Stage 0**: 已完成
  - `tl.device_programs`
  - `tl.semantic_seeds`
  - hard-freeze / invalidation 护栏
- **Phase A**: 已完成
  - `SemanticProgram`
  - witness algebra
  - refinement validator
  - internal state/effect graph
  - lifecycle contract
  - `stage4_semantic_manifest` `Phase 1-2`
- **Phase B**: 当前主实施阶段
- **Phase C**: 已定义；待 `Phase B` 后推进

### 7.2 当前主 blocker

当前 blocker 已经不是 `Phase A` 语义恢复本身，而是：

- `Spatial Program IR -> TT Target IR` 的单一真源切换还未完成

这也是当前 `blackhole.acc` correctness payoff 仍未完全兑现的根因。

### 7.3 当前主设备链事实

当前 Blackhole 设备侧 pass 主线中已经稳定接入 `Phase A`：

```text
LowerDeviceStorageAccessInfo
  -> AugmentSemanticManifest
  -> LowerIntrin
  -> Simplify
  -> HoistBroadcastValues
  -> SplitBlackholeKernel
  -> AnalyzeBlackholeWorkDecomposition
  -> AnalyzeBlackholeFragmentRegions
  -> AnalyzeBlackholePipelineStages
  -> AnalyzeSemanticStructure
  -> LiftStatefulSemanticIR
  -> ValidateStatefulSemanticIR
  -> ValidateSemanticRefinement
  -> LowerBlackholeOps
  -> PlanBlackholeCB
  -> AssignBlackholeCores
```

其中当前 `Phase A` 相关的稳定事实已经是：

- `AnalyzeSemanticStructure` 对 manifest structural evidence 采用 manifest-first 消费
- `blackhole.fragment_regions` 只剩 residual reduction evidence 与 compatibility fallback
- `Phase B / C` 不能再把 `fragment_regions` 当 semantic truth source

`Phase B / C` 相关 pass 以阶段文档推进，不在总设计中重复实施 checklist。

## 8. 文档分工

当前文档分工固定为：

- `final_blackhole_backend_redesign.md`
  - 唯一总体设计
  - 只保留长期架构、层间边界、真源规则、cutover 原则
- `stage4_stage0_guardrails.md`
  - Stage 0 护栏与前置 contract
- `stage4_phase_a_semantic_ir.md`
  - `Phase A` 工程边界与已完成状态
- `stage4_phase_a_formalization_note.md`
  - `Phase A` 理论化 / 证明化并行文档
- `stage4_phase_b_spatial_ir.md`
  - `Phase B` 核心设计边界与实施计划
- `stage4_phase_c_tt_target_ir.md`
  - `Phase C` 核心设计边界、cutover 与实施计划

## 9. 历史文档

下面这些文档只作为历史记录或实现历史参考，不再作为当前实现依据：

- `tasks/dev_design/archive/legacy_blackhole_runtime_architecture.md`
- `tasks/dev_design/archive/2026-04-02-stateful-tiled-ir-phase1-implementation-plan.md`

## 10. 参考论文

下面这些论文影响了本文档中的分层、validation 与 target-mapping 方向。它们是设计输入，不是协议真源。

- `Dato: A Task-Based Programming Model for Dataflow Accelerators` (2025)
  - https://arxiv.org/abs/2509.06794
- `TL: Automatic End-to-End Compiler of Tile-Based Languages for Spatial Dataflow Architectures` (2025)
  - https://arxiv.org/abs/2512.22168
- `SPADA: A Spatial Dataflow Architecture Programming Language` (2025)
  - https://arxiv.org/abs/2511.09447
- `Revet: A Language and Compiler for Dataflow Threads` (2023/2024)
  - https://arxiv.org/abs/2302.06124
- `Programmatic Control of a Compiler for Generating High-performance Spatial Hardware` (`T2S`, 2017)
  - https://arxiv.org/abs/1711.07606
