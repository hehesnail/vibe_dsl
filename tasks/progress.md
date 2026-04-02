# TileLang Blackhole Backend Progress

> 当前唯一总体设计文档: `tasks/dev_design/final_blackhole_backend_redesign.md`
> `tasks/dev_design/archive/` 下的文档全部是历史记录，不再作为当前任务安排入口。

## 当前阶段

- **日期**: `2026-04-02`
- **阶段**: Stage 4 — flash-attn forward subset / TT-Metal contract cleanup / layered IR architecture transition
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
    - `AtomicEffect -> semantic region` 的通用恢复规则
    - `SemanticProgram -> SpatialProgram` 的空间化构造规则与 policy 边界
    - `SpatialProgram + hardware model -> TTProgram` 的 target mapping 规则与 materialization 边界
  - `flash-attn` 仍是第一批 consumer，但不再作为总架构边界

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
  - `Stateful Semantic IR` 冻结算法 state / relation / phase 真语义
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
   - 冻结 `domain / state / relation / phase`
   - 明确 `PrimFunc.attrs["tl.semantic_program"]` 与 `TIRAnchor` 承载 contract
   - 保证 copy / GEMM compile-path zero-regression
3. 执行 **Phase B: Spatial Program IR**：
   - 把 selection/indexing、routed/grouped dispatch、paged decode、stateful update、chunk recurrence 的 `task / channel / layout / sync / work partition` 一等化
   - 明确 `PrimFunc.attrs["tl.spatial_program"]` 承载 contract
   - 明确 spatial analysis 与 spatial policy 的边界
   - 拆掉 `LowerBlackholeOps` 当前同时承担语义理解和 target lowering 的边界
4. 执行 **Phase C: TT Target IR**：
   - 统一 `CB / semaphore / dst layout / kernel role / ABI / execution plan`
   - 明确 `PrimFunc.attrs["tl.tt_program"]` 与 `IRModule.global_infos` 中 hardware model/topology 的承载 contract
   - 明确 target legality analysis 与 TT policy 的边界
   - 把 `ExecutableSpec` 改成从 `TT Target IR` 物化
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
