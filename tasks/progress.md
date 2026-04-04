# TileLang Blackhole Backend Progress

> 当前唯一总体设计文档: `tasks/dev_design/final_blackhole_backend_redesign.md`
> `tasks/dev_design/archive/` 下的文档全部是历史记录，不再作为当前任务安排入口。

## 当前阶段

- **日期**: `2026-04-03`
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
    - `AtomicEffect -> SemanticRegion` 的通用恢复规则
    - `Domain / State / Update` semantic core
    - `AccessMap / UpdateLaw` 作为 `Update` 的 typed 组成部分
    - `TIRAnchor / TIRValueBinding` 已开始向 semantic-core-aligned 的 typed bridge 收口
    - `SemanticProgram -> SpatialProgram` 的空间化构造规则与 policy 边界
    - `SpatialCandidate / SpatialPolicy / SpatialCostModel` 的 planning contract
    - `SpatialProgram + hardware model -> TTProgram` 的 target mapping 规则与 materialization 边界
    - `TTHardwareModel` 的 typed schema（含 `TTComputeModel` 子模型）
    - `TTProgram -> TT-lowered PrimFunc + ExecutableSpec` 的唯一物化路径
  - 权威总设计已完成系统性 review 并补齐以下 gap（2026-04-02）：
    - Semantic core 收敛：从较重的 `Domain/State/Relation/Phase/SemanticRegion + descriptor family` 收敛为 `Domain / State / Update`
    - `AccessMap / UpdateLaw`：吸收 paged/routed/selection/recurrence 语义，不再平行维护 `*Spec` 家族
    - `StateSSA` 方向：长期 public schema 保持小，内部保留 `StateVersion / StateJoin` 式分析图
    - Recovery Boundary：统一语义系统下的自动恢复 / 最小 DSL 补语义边界 + workload validation matrix + `T.annotate_semantic()` protocol
    - `ProgramPhase`：Spatial Program IR 新增多 kernel 组合的全局阶段边界（fusedmoe 双 T.Kernel、flash_decoding split+combine）
    - `TTComputeModel`：TTHardwareModel 新增 compute 子模型（FPU/SFPU 独立性、dst 争用、pack/unpack mode）
    - `TTKernel.role` 改为 composable `role_set` + `role_flags`（承接 MoE unified kernel pattern）
    - `TTCBPlan / TTDstLayoutPlan` 新增 `data_format`、`pack_mode`、`unpack_mode`、`l1_acc_mode`、`fpu_sfpu_ordering` 等字段
    - Companion IR invalidation：post-semantic-lift 默认 `unsafe`；只有 audited `identity-preserving / rebind-aware` pass 才能声明 `safe`
    - `SplitBlackholeKernel` 过渡边界：pre-semantic 仅作为 canonicalization / temporary signal producer，compatibility `blackhole.segment_plan` 不是真源
    - Materialized attr ownership：`blackhole.segment_plan/runtime_args/cb_configs/core_plan` 的稳态唯一 writer 固定为 `MaterializeTTExecutableSpec`
    - Phase A 细分为 A1（最小 multi-family recovery）和 A2（泛化 + 更宽 `AccessMap / UpdateLaw` traits），A1 gate 强制包含一个 non-attention semantic skeleton case
    - Section 7 workload 示例已切换到 `Domain / State / Update` 叙述，去掉 workload-specific semantic enum
  - 2026-04-03 进一步完成活动文档收口：
    - 总设计已明确 `SemanticProgram(Domain / State / Update)` 是第一层真源
    - `AccessMap / UpdateLaw` 明确固定为 `Update` 的组成部分，而不是平行 schema
    - `TIRAnchor / AtomicEffect / SemanticRegion` 已重新收口为 recovery / binding / projection helper
    - 已补充 semantic 层在 freeze 之后的主要作用：validation truth、spatialization input、invalidation cut、workload normalization
    - `stage4_flash_attention_forward_subset.md` 已改成从属 consumer 视角，不再暗含自己的 semantic schema
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
   - 明确 `AtomicEffect / SemanticRegion` 只是 recovery helper，不是第一层 core schema
   - 收紧 post-semantic-lift invalidation：默认 `unsafe`，仅允许 audited `safe` pass
   - 明确 pre-semantic compatibility attrs 不参与 semantic/spatial truth 判定
   - 保证 copy / GEMM compile-path zero-regression + 至少一个 non-attention semantic skeleton case
3. 执行 **Phase B: Spatial Program IR**：
   - 把 selection/indexing、routed/grouped dispatch、paged decode、stateful update、chunk recurrence 的 `task / channel / layout / sync / work partition` 一等化
   - 明确 `PrimFunc.attrs["tl.spatial_program"]` 承载 contract
   - 明确 spatial analysis 与 `SpatialCandidate / SpatialPolicy / SpatialCostModel` 的边界
   - 拆掉 `LowerBlackholeOps` 当前同时承担语义理解和 target lowering 的边界
4. 执行 **Phase C: TT Target IR**：
   - 统一 `CB / semaphore / dst layout / kernel role / ABI / execution plan`
   - 明确 `PrimFunc.attrs["tl.tt_program"]` 与 `IRModule.global_infos` 中 `TTHardwareModel` 的承载 contract
   - 明确 target legality analysis 与 TT policy 的边界
   - 把 `ExecutableSpec` 与 TT-lowered `PrimFunc` 一并改成从 `TT Target IR` 唯一物化
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
