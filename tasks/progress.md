# TileLang Blackhole Backend Progress

> 当前唯一总体设计文档: `tasks/dev_design/final_blackhole_backend_redesign.md`
> `tasks/dev_design/archive/` 下的文档全部是历史记录，不再作为当前任务安排入口。

## 当前阶段

- **日期**: `2026-04-05`
- **阶段**: Stage 4 — 复杂 workload family 的 layered IR architecture transition
- **当前主线**:
  - 以新的分层架构推进后续实现：
    `Stateful Semantic IR -> Spatial Program IR -> TT Target IR`
  - 在保持 copy / GEMM / current direct-path 稳定的前提下，解决复杂前端计算在 Blackhole 上的统一承接问题
  - 权威总设计已改为中文主叙述，并明确覆盖：
    - selection / indexing
    - routed / grouped / ragged dispatch
    - paged / indexed sparse access
    - stateful reduction-update
    - chunked recurrence / scan
  - 权威总设计已补齐实现层 contract：
    - `TIR + companion IR` 混合承载模型
    - `PrimFunc.attrs + IRModule.global_infos + materialized blackhole.* attrs` 的职责分层
    - `TIRAnchor / TIRValueBinding`
    - `AtomicEffect -> Update -> SemanticRegion` 的通用恢复 / 导出边界
    - `Domain / State / Update` semantic core
    - `AccessMap / UpdateLaw` 作为 `Update` 的 typed 组成部分
    - `TIRAnchor / TIRValueBinding` 已开始向 semantic-core-aligned 的 typed bridge 收口
    - `SemanticProgram -> SpatialProgram` 的空间化构造规则与 policy 边界
    - `SpatialCandidate / SpatialPolicy / SpatialCostModel` 的 planning contract
    - `SpatialProgram + hardware model -> TTProgram` 的 target mapping 规则与 materialization 边界
    - `TTHardwareModel` 的 typed schema（含 capability-oriented `TTComputeModel` 子模型）
    - `TTProgram -> TT-lowered PrimFunc + ExecutableSpec` 的唯一物化路径
  - 权威总设计已完成系统性 review 并补齐以下 gap（2026-04-02）：
    - Semantic core 收敛：从较重的 `Domain/State/Relation/Phase/SemanticRegion + descriptor family` 收敛为 `Domain / State / Update`
    - `AccessMap / UpdateLaw`：吸收 paged/routed/selection/recurrence 语义，不再平行维护 `*Spec` 家族
    - `StateSSA` 方向：长期 public schema 保持小，内部保留 `StateVersion / StateJoin` 式分析图
    - Recovery Boundary：统一语义系统下的自动恢复 / 最小显式语义补充边界 + workload validation matrix
    - `ProgramPhase`：Spatial Program IR 新增多 kernel 组合的全局阶段边界（fusedmoe 双 T.Kernel、flash_decoding split+combine）
    - `TTComputeModel`：TTHardwareModel 新增 capability-oriented compute 子模型（execution units、dst hazard、pack/unpack、accumulator/data-format rules）
    - `TTKernel`：收回到小闭 family + trait axes，不再用 `role_set + role_flags` 继续堆 target noun
    - `TTCBPlan / TTDstLayoutPlan / TTComputeSyncPlan`：区分 program-level transport/storage plan 与 compute-kernel internal sync / dst residency plan
    - Companion IR invalidation：post-semantic-lift 默认 `unsafe`；只有 audited `identity-preserving / rebind-aware` pass 才能声明 `safe`
    - `SplitBlackholeKernel` 过渡边界：pre-semantic 仅作为 canonicalization / temporary signal producer，compatibility `blackhole.segment_plan` 不是真源
    - Materialized attr ownership：`blackhole.segment_plan/runtime_args/common_runtime_args/accessors/cb_configs/semaphore_plan/core_plan` 的稳态唯一 writer 固定为 `MaterializeTTExecutableSpec`
    - Phase A 细分为 A1（最小 multi-family recovery）和 A2（泛化 + 更宽 `AccessMap / UpdateLaw` traits），A1 gate 强制包含一个 non-attention semantic skeleton case
    - Section 7 workload 示例已切换到 `Domain / State / Update` 叙述，去掉 workload-specific semantic enum
  - 2026-04-03 进一步完成活动文档收口：
    - 总设计已明确 `SemanticProgram(Domain / State / Update)` 是第一层真源
    - `AccessMap / UpdateLaw` 明确固定为 `Update` 的组成部分，而不是平行 schema
    - `TIRAnchor / AtomicEffect / SemanticRegion` 已重新收口为 recovery / binding / projection helper
    - 已补充 semantic 层在 freeze 之后的主要作用：validation truth、spatialization input、invalidation cut、workload normalization
    - `stage4_flash_attention_forward_subset.md` 已改成从属 consumer 视角，不再暗含自己的 semantic schema
  - 2026-04-05 进一步收口 semantic helper 设计：
    - `TIRValueBinding` 明确为 typed field-binding index，而不是泛化 value bag
    - `TIRAnchor` 只保留结构锚点职责，不再重复承担字段级 binding
    - `SemanticRegion` 明确改成从 `Update` 图导出的非真源视图，只用于 debug / diagnosis / spatial clustering
    - “DSL 补语义” 已拆成独立的“显式语义补充边界”部分，不再和 semantic core 正文混写
    - Phase A 当前不再提前承诺公开 `T.annotate_semantic()`；第一版先定义 compiler-internal `SemanticSupplement` / `tl.semantic_supplement`
    - supplement 只允许裁决少数 IR 无法唯一决定的语义事实，不能覆盖结构恢复，也不能表达 spatial/target 细节
  - 2026-04-05 继续收口 Spatial IR：
    - `ProgramPhase` 明确为强边界；跨 phase 通信只能走 materialized `shared_buffers` + global sync，`Channel` 不允许跨 phase
    - `Task / Channel / Layout / WorkPartition / Placement / SyncEdge / ResourceIntent` 全部改成“小闭枚举 + trait set”，避免 workload-specific kind 无限制膨胀
    - `Task / Channel / Layout / WorkPartition` 的第一版 base family 已进一步收窄，新增 kind 现在要求真的改变一级 legality/candidate/target 分派
    - `traits` 已进一步收成固定轴系统，不再允许自由字符串；每类 spatial object 只允许来自少数预定义 trait axes
    - core schema 已补齐显式 bindings：`payload_states`、`domain_bindings`、`update_or_state_bindings`、`attachment_ref`
    - spatial planning contract 新增 `SpatialLegalityFacts`，并进一步收成 `Cut/Flow/Phase/Layout/Partition/Sync` 这组 typed legality entries，再生成 `SpatialCandidate`
  - 2026-04-05 继续收口 TT Target IR：
    - `TTSemaphorePlan` 收窄为 program-level semaphore / barrier / multicast contract，compute-kernel internal sync 单独拆成 `TTComputeSyncPlan`
    - `TTKernel / TTCoreGroup / TTCBPlan / TTSemaphorePlan / TTComputeSyncPlan / TTDstLayoutPlan / TTABIPlan / TTExecutionPlan` 已补齐与 spatial truth 的显式 bindings
    - `TTKernel` 改成小闭 `kind + traits`，不再依赖自由组合的 `role_set + role_flags`
    - `TTCBPlan.resource_class` 收成小闭 family，细粒度差异走 traits
    - `TTComputeModel` 从操作名清单收回到 capability classes + legality rules
    - TT materialization outputs 已与当前 supporting docs 对齐，`blackhole.semaphore_plan` 明确回到 `MaterializeTTExecutableSpec` 的稳定产物集合
    - `ExecutableSpec` 已进一步收成 program container + `KernelSpec[]` 的物化边界，顶层 aggregate ABI view 明确降为 compatibility-only
    - 当前仍保留的 runtime/codegen compatibility shim 已明确点名：segment 聚合顶层 `runtime_args/common_runtime_args`、顶层 `accessors` 回写、legacy attr device-kernel 检测、buffer-role positional fallback；Phase C cutover 后必须删除
  - `flash-attn` 仍是第一批 consumer，但不再作为总架构边界；`topk / fusedmoe / paged decode / chunk recurrence` 同样属于当前设计覆盖面

## 当前稳定基线

- `ExecutableSpec -> rt_mod_blackhole -> BlackholeModule` direct host path 已稳定
- `tilelang.compile(..., execution_backend="tvm_ffi")` 的 Blackhole wrapper/export path 已恢复
- copy / GEMM current support surface 已打通，multi-core formal direct host path 已完成
- TT-Metal contract formalization 当前已完成到：
  - P0：compute contract 主链
  - P3：current copy / GEMM formal runtime surface
  - P4：最小 interleaved stick/page copy 主路径
  - P5：program-local worker semaphore + remote-core descriptor formalization
- flash-attn forward subset 当前已完成：
  - `AnalyzeBlackholeWorkDecomposition`
  - `AnalyzeBlackholeFragmentRegions`
  - `AnalyzeBlackholePipelineStages`
  - 最小 fragment/dataflow builtin lowering
  - 当前支持的 MHA / GQA forward compile-path

## 当前主 blocker

- flash-attn runtime 已不再 hang，但 correctness 仍未通过。
- 根因已经收敛为 `blackhole.acc` 混合语义：
  - 一部分 lowering 仍把它当 TT compute-side tile scratch / matmul destination
  - 另一部分 helper 仍把它当线性 fragment scratch 数组
- 这正是当前架构切换到 layered IR 的直接动机：
  - `Stateful Semantic IR` 冻结算法 `domain / state / update` 真语义
  - `Spatial Program IR` 单独表达 task / channel / layout / sync / work partition
  - `TT Target IR` 统一承接 CB / semaphore / dst layout / kernel role / ABI
- 但这套分层设计的目标不再是“只修 flash-attn”：
  - 它还需要承接 `topk`、`fusedmoe`、`paged decode`、`mamba chunk state` 等复杂前端计算 family

## 下一步

1. 基于 `tasks/dev_design/final_blackhole_backend_redesign.md` 重写新的 layered-IR implementation plan。
2. 执行 **Phase A: Stateful Semantic IR**：
   - `AnalyzeSemanticStructure`
   - `LiftToStatefulSemanticIR`
   - `ValidateStatefulSemanticIR`
   - 冻结 `domain / state / update`
   - 明确 `AccessMap / UpdateLaw` 的最小 schema
   - 明确 `PrimFunc.attrs["tl.semantic_program"]`、typed `TIRAnchor / TIRValueBinding` 与 rebind contract
   - 明确独立的显式语义补充边界，只先定义 compiler-internal `SemanticSupplement` / `tl.semantic_supplement`
   - 明确 `AtomicEffect / SemanticRegion` 只是 recovery helper，不是第一层 core schema
   - 明确 `AtomicEffect -> Update` 恢复与 `Update -> SemanticRegion` 导出之间的边界
   - 收紧 post-semantic-lift invalidation：默认 `unsafe`，仅允许 audited `safe` pass
   - 明确 pre-semantic compatibility attrs 不参与 semantic/spatial truth 判定
   - 保证 copy / GEMM compile-path zero-regression + 至少一个 non-attention semantic skeleton case
3. 执行 **Phase B: Spatial Program IR**：
   - 把 selection/indexing、routed/grouped dispatch、paged decode、stateful update、chunk recurrence 的 `task / channel / layout / sync / work partition` 一等化
   - 明确 `PrimFunc.attrs["tl.spatial_program"]` 承载 contract
   - 明确 `ProgramPhase` 的强边界与 phase-boundary materialization 规则
   - 先落 `SpatialLegalityFacts`，再落 `SpatialCandidate / SpatialPolicy / SpatialCostModel`
   - 保持 `kind` 小闭枚举，细粒度 workload 差异通过 trait/binding 表达，不再靠不断增补 kind
   - 明确 spatial analysis 与 `SpatialCandidate / SpatialPolicy / SpatialCostModel` 的边界
   - 拆掉 `LowerBlackholeOps` 当前同时承担语义理解和 target lowering 的边界
4. 执行 **Phase C: TT Target IR**：
   - 统一 `CB / semaphore / dst layout / kernel role / ABI / execution plan`
   - 明确 `PrimFunc.attrs["tl.tt_program"]` 与 `IRModule.global_infos` 中 `TTHardwareModel` 的承载 contract
   - 明确 target legality analysis 与 TT policy 的边界
   - 把 `ExecutableSpec` 与 TT-lowered `PrimFunc` 一并改成从 `TT Target IR` 唯一物化
   - 明确 `ExecutableSpec` program-shared metadata 与 `KernelSpec` per-kernel ABI 的 ownership
   - 清除 `rt_mod_blackhole / BlackholeModule / lower.py` 中仍保留的 compatibility aggregation / fallback / legacy attr detection
5. 在新分层下继续扩更宽 flash-attn forward 支持面，以及 P4 / P5 更宽执行面。

## 当前代码事实

- 当前 Blackhole 设备侧 pass 主线：
  `LowerDeviceStorageAccessInfo`
  -> `LowerIntrin`
  -> `Simplify`
  -> `HoistBroadcastValues`
  -> `SplitBlackholeKernel`
  -> `AnalyzeBlackholeWorkDecomposition`
  -> `AnalyzeBlackholeFragmentRegions`
  -> `AnalyzeBlackholePipelineStages`
  -> `LowerBlackholeOps`
  -> `PlanBlackholeCB`
  -> `AssignBlackholeCores`
- `SplitBlackholeKernel` 已接入管线：
  - 纯 copy 走 `fused_dataflow` 单 kernel
  - GEMM 走 3-kernel（reader / compute / writer）
- direct runtime 当前正式支持面：
  - copy：equal source/dest range，且 stride = 1
  - GEMM：A/B-separated reader range + writer output range
  - accessor：仅 interleaved + DRAM + `common_runtime_arg_count = 0`
- TT-Sim 当前正式入口是顶层 `scripts/setup_tt_sim.sh`，并且必须和后续测试命令在同一个 shell 中执行

## 最新验证

### 当前环境

| 测试 | 结果 |
|------|------|
| `test_blackhole_flash_attention_pipeline.py` | 当前通过；已覆盖 `acc_s_cast` 发布与 `blackhole.acc` GEMM 输出不重复 reserve 回归 |
| `test_blackhole_flash_attention_runtime.py` | runtime 已不再 hang；`-k mha` 当前执行完成但 correctness 未通过 |
| `test_blackhole_copy_pipeline.py` | `40 passed, 10 skipped, 1 xfailed` |
| `test_blackhole_copy_runtime.py` | `2 passed, 9 skipped` |
| `test_blackhole_gemm.py` | `24 passed, 11 skipped` |
| `test_blackhole_tvm_ffi_export.py` | `1 passed` |

### 已验证 full-env 基线

| 测试 | 结果 |
|------|------|
| `test_blackhole_copy_pipeline.py` | `30 passed, 1 xfailed` |
| `test_blackhole_copy_runtime.py` | `11 passed` |
| `test_blackhole_gemm.py` | `31 passed` |

## 当前活动文档

- `tasks/dev_design/final_blackhole_backend_redesign.md`
- `tasks/dev_design/stage4_flash_attention_forward_subset.md`
- `tasks/dev_design/stage2d_ttmetal_contract_audit.md`
- `tasks/dev_design/stage4_semaphore_schema.md`
- `tasks/dev_design/README.md`

## 说明

- 旧的 runtime 架构说明、旧单层 implementation plan、以及过去阶段的详细执行记录都已移入 `tasks/dev_design/archive/`。
- 如果需要查看历史决策或旧阶段实现背景，去 archive 或 git history；不要再把历史文档当当前任务安排入口。
