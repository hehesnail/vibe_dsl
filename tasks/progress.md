# TileLang Blackhole Backend Progress

> 当前唯一总体设计文档: `tasks/dev_design/final_blackhole_backend_redesign.md`
> `tasks/dev_design/archive/` 下的文档全部视为历史记录，不再作为当前任务安排入口。
> `Phase C` 的详细完成判定、runtime gate 与支持面边界统一维护在
> `tasks/dev_design/stage4_phase_c_tt_target_ir.md`；这里只保留当前状态摘要。

## 当前阶段

- **日期**: `2026-04-13`
- **总阶段**: Stage 4
- **目标主线**: `Normalized Tile TIR -> SpatialPlan companion -> TTProgram companion -> ExecutableSpec`
- **当前代码基线**: 仍为旧 `Stateful Semantic IR -> Spatial Program IR -> TT Target IR`
- **阶段状态**:
  - 当前执行基线仍是旧 `Phase C` 主链
  - 两层 companion 总设计与 supporting docs 已收正；下一阶段进入 architecture cutover
  - `Stage 0 / Phase A / Phase B` 文档保留为历史实现边界，不再代表长期总设计

## 当前状态摘要

- `ExecutableSpec -> rt_mod_blackhole -> BlackholeModule` direct host path 仍是唯一正式执行路径
- `tasks/dev_design/final_blackhole_backend_redesign.md`
  已改写为新的唯一总设计：
  `Normalized Tile TIR -> SpatialPlan companion -> TTProgram companion -> ExecutableSpec`
- 从当前设计起：
  - `SemanticProgram` 与旧 `SpatialProgram` 只代表当前代码基线/历史实现边界
  - 后续所有架构推进以
    `TIR body / SpatialPlan companion / TTProgram companion`
    为 owner 链
  - `Task / Channel` 继续存在，但只作为
    `ExecutionClosure / ClosureBoundary` 的 derived view
- `tilelang.compile(..., execution_backend="tvm_ffi")` 的 Blackhole wrapper/export path 已恢复
- `TTProgram` translator / validator / materializer 已进入正式主链；
  runtime/codegen 已切到 `TTProgram` direct reader
- `per_work_arg_specs` 已收成 kernel-local `TTKernel / ExecutableSpec` truth；
  synthetic segment codegen、direct runtime arg materialization
  与 direct runtime launch 统一消费同一套 `value_kind` contract，
  不再按 arg kind 出现情况或 `work_linear_id -> blockIdx` 猜语义
- spatial/dataflow program model 的 cross-layer feature 设计
  已独立收口到 `tasks/dev_design/spatial_dataflow_program_model.md`，
  其中已明确：
  - `SpatialPlan companion`
    只持久化 `ExecutionClosure / ClosureBoundary / ValidatedHintSet`
  - `TTProgram companion`
    只持久化
    `TTBlockPlan / TTKernelPlan / TTTransportPlan / TTSyncPlan /
    TTABIPlan / TTExecutionPlan`
  - 新 pass owner 链为
    `AnalyzeSpatialStructureFacts -> BuildSpatialPlanCompanion ->
    PlanTTBlocks -> PlanTTTransport -> PlanTTSync -> PlanTTABI ->
    PlanTTExecution -> MaterializeBlackholeExecutable`
- 活动设计入口、完成阶段边界文档与 archive 历史记录
  已重新对齐；根目录只保留当前入口与完成阶段边界，
  审计快照与过时评审文档已回收到 `tasks/dev_design/archive/`
- TT-Sim fatal simulator taxonomy / hard-gate 扫描
  已独立记录到 `memory/tt_simulator_constraints.md`，
  后续 triage 先按 simulator capability boundary 与 target contract 回归分流
- `flash-attn` compile-path / metadata / pipeline 主链保持可用；
  direct runtime 当前会对缺失显式 per-work contract
  或 typed fragment materialization/merge protocol 尚未执行化的 kernel 显式 fail-fast；
  在当前 regression 子集上，显式 per-work contract 已 materialize 完整，
  剩余 skip/gate 主要来自 typed fragment materialization contract
- fragment materialization contract 的 owner-side 识别
  已从 `AnalyzeSemanticStructure` 里的 `gemm_py` family matcher
  前移到 tile-op typed metadata；
  analysis pass 现在只把上游暴露的事实归约成 generic contract，
  不再自己按 family 名字判断协议
- 当前这条线的第一 blocker
  已从“补一个 runtime protocol”
  收敛为“把 active owner 链从 recovery 主链切到两层主链”：
  旧 `SpatialProgram` 对跨-op intermediate edge 的
  `dataflow contract`、per-buffer `work/access contract`
  和 fragment live-form 问题，只是这条根因在当前代码基线上的暴露形式；
  长期解法不再是继续给旧链补 `*_kind` 或 matcher，
  而是直接落
  `SpatialPlan companion -> TTProgram companion` cutover
- 对这组 blocker 的根因判定已独立沉淀到
  `tasks/dev_design/ir_layering_root_cause_and_direction.md`：
  `destroy-then-recover`（`Fragment` 在
  `OptimizeForTarget / SplitHostDevice` 被丢弃后靠 `fragment_layout_seeds`
  反补；tile 算子在 `LowerTileOp` 被拆成 scalar loop 后靠
  `LowerBlackholeOps` 的 17 个 `Match*` 反推）和
  `enum-of-forms`（`FragmentMaterializationInfo` 6 个 `ffi::String`
  字段、`companion_base.h` 6 个 stringly-typed 命名空间、
  `SpatialProgram` 节点 `Map<String, Any> payload`）是同一条根因在不同层的
  投影；后续任何“再加一个 `*_kind` 字符串”的修补都会继续喂这条根因
- 新的总设计已经把这组根因收敛为两条明确决策：
  - 第一层 stable companion 必须前移到 `Simplify` 后、
    `LayoutReducer / LayoutInference / LowerTileOp` 前，
    从 normalized TIR 直接分析并冻结
    `ExecutionClosure / ClosureBoundary / ValidatedHintSet`
  - 第二层 stable companion 必须补齐 `TTBlockPlan`，
    让 block/resource/transport/sync/ABI/execution
    直接从 closure/boundary truth 派生，不再通过 lowered TIR matcher 恢复
- owner-side `fragment_buffer_flow_contracts`
  已开始在 `AnalyzeSemanticStructure -> SpatialProgram`
  显式 materialize，并由 `LowerBlackholeOps` 直接消费；
  这一步已经覆盖 fragment/local intermediate buffer
  与 compute kernel 内同一 producer-consumer 协议下的 CB-backed input buffer，
  不再允许 lower 侧本地扫 `SeqStmt` 恢复
  `write / compute_consume / transport_consume / republish` 语义
- real `lower()` 主链里原先会在
  `OptimizeForTarget -> SplitHostDevice` 之后丢掉
  `layout_map / tl.Fragment` truth；
  当前已把这部分 truth 投影成 device-side
  `tl.fragment_layout_seeds`，
  并由 `AnalyzeBlackholeFragmentRegions`
  重新 materialize 成 typed `fragment_layout_contracts`
- 但这次回溯也证明当前 blocker 不只是 post-merge case：
  `blackhole.acc` fragment 的真实 live form 仍未被上游 contract 说清。
  当前 device-side physical buffer 只保留 per-lane local extent
  （典型 case：逻辑 `32x32` fragment，physical local extent 只有 `8`），
  但 `LowerBlackholeOps / codegen` 仍会把它当成
  已 materialized 的线性 logical fragment 使用。
  结果是：
  `fragment_fill -> cast -> publish` direct runtime 输出全零，
  `clear_accum=false` merge 后继续 cast consumer 时只覆盖一小条 slice，
  说明问题是通用的 thread-distributed fragment live-form /
  execution-lane contract 缺口，而不是单个 post-merge 特例
- 人为移除 `compute_epilogue_ops` gate 后，
  small `bf16` MHA direct runtime 仍会明显错算
  （当前采样：`max diff=1.2265625`, `mean diff=0.2021484375`），
  说明 fragment materialization/merge 问题不是 gate 过严，而是执行语义本身还没闭环
- Blackhole runtime / direct-runtime 回归基线已统一到 `bf16` 输入；
  `fp16` 不再作为当前 TT-Sim 上的正式 runtime 测试基线
- 无显式 `semaphore / remote-core` synchronization contract 的
  oversubscribed `work_packets` executable 已可按 packet truth 做 host-side
  wave scheduling
- 当前 admitted support surface 仍然偏窄：
  - copy：equal source/dest range 且 stride = 1
  - GEMM：A/B-separated reader range + writer output range
  - accessor：仅 interleaved + DRAM + `common_runtime_arg_count = 0`

## 当前未完成项

- 按新总设计切出并实现两层主链：
  - `AnalyzeSpatialStructureFacts`
  - `BuildSpatialPlanCompanion`
  - `PlanTTBlocks / PlanTTTransport / PlanTTSync / PlanTTABI / PlanTTExecution`
  - `MaterializeBlackholeExecutable`
- 在同一轮 cutover 中退场旧 recovery 主链：
  `semantic_manifest / SemanticProgram / 旧 SpatialProgram /
  LowerBlackholeOps matcher 路线`
- 用新主链重新承接
  `flash-attn / topk / fusedmoe / paged decode / chunk recurrence`
- 在 `TTProgram` 层补齐 `TTBlockPlan`，
  收正 SRAM/CB/semaphore/runtime arg/work packet owner
- 在 cutover 完成前，继续保持 copy / GEMM / export 当前正式支持面不回退

## 当前边界

- oversubscribed direct runtime 目前不是通用同步执行模型；
  一旦 executable 带显式 `TTSemaphorePlan`、`semaphore_bindings`
  或 `remote_core_descriptors`，仍应 fail-fast
- `flash-attn` direct runtime 当前也不是已恢复支持面；
  对缺失显式 per-work contract 或 typed fragment materialization/merge protocol
  的 kernel，应继续通过 `direct_runtime_unsupported_reasons` skip / fail-fast，
  而不是回退到后段语义恢复
- TT-Sim `fp16` 路径当前仍可能命中 simulator fatal taxonomy；
  该路径不属于当前 Blackhole runtime 的正式 correctness baseline，
  统一按 simulator capability boundary 处理
- `TT_METAL_WATCHER=10` 调试 multicore direct path 时，
  watcher 线程仍可能自己抛错或卡在 dump；
  正式 baseline 应在标准 watcher-off 环境下判断

## 下一步

1. 在 `Simplify` 后实现 `AnalyzeSpatialStructureFacts`，
   直接替掉 `LayoutInference -> semantic_manifest -> SemanticProgram`
   这条 recovery 入口
2. 实现 `BuildSpatialPlanCompanion`，
   把 `Task / Channel` 退回 derived execution grouping
3. 实现带 `TTBlockPlan` 的 `TTProgram companion` cutover，
   吸收 `SplitBlackholeKernel / PlanBlackholeCB / AssignBlackholeCores`
   的 owner 职责，并拆成
   `PlanTTBlocks / PlanTTTransport / PlanTTSync / PlanTTABI / PlanTTExecution`

## 最新验证摘要

- `tilelang_repo/build` fresh rebuild 通过
- 所有 runtime 检查均在标准 TT-Sim 环境入口下完成
- 本轮 `flash-attn` regression 通过：
  `test_blackhole_flash_attention_pipeline.py` -> `62 passed`
  `test_blackhole_flash_attention_runtime.py` -> `9 passed, 5 skipped`
- 本轮 copy regression 通过：
  `test_blackhole_copy_pipeline.py` -> `43 passed, 10 skipped, 1 xfailed`
  `test_blackhole_copy_runtime.py` -> `12 passed`
- 本轮 copy baseline 也通过：
  `test_blackhole_copy_pipeline.py` -> `52 passed, 1 skipped, 1 xfailed`
  `test_blackhole_copy_runtime.py` -> `12 passed`
- 当前 `flash-attn` runtime skip 均来自显式 `direct_runtime_unsupported_reasons`，
  当前 regression 子集里不再出现缺失 explicit per-work descriptor 的 skip；
  剩余 skip 来自 typed fragment materialization contract 已存在、
  但 direct runtime 尚未执行 fragment materialization/merge protocol 的边界
