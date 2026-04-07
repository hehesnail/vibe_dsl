# TileLang Blackhole Backend Progress

> 当前唯一总体设计文档: `tasks/dev_design/final_blackhole_backend_redesign.md`
> `tasks/dev_design/archive/` 下的文档全部视为历史记录，不再作为当前任务安排入口。

## 当前阶段

- **日期**: `2026-04-07`
- **总阶段**: Stage 4
- **当前主线**: layered IR architecture transition
  - `Stateful Semantic IR -> Spatial Program IR -> TT Target IR`

## 当前稳定基线

- `ExecutableSpec -> rt_mod_blackhole -> BlackholeModule` direct host path 仍是唯一正式执行路径
- `tilelang.compile(..., execution_backend="tvm_ffi")` 的 Blackhole wrapper/export path 已恢复
- copy / GEMM 当前支持面必须保持不回退
- `SplitBlackholeKernel` 已接入主线：
  - 纯 copy 走 `fused_dataflow` 单 kernel
  - GEMM 走 `reader / compute / writer` 3-kernel
- direct runtime 当前正式支持面：
  - copy：equal source/dest range 且 stride = 1
  - GEMM：A/B-separated reader range + writer output range
  - accessor：仅 interleaved + DRAM + `common_runtime_arg_count = 0`

## 阶段状态

- **Stage 0**: 已完成
  - `tl.device_programs`
  - `tl.semantic_seeds`
  - hard-freeze / invalidation 护栏
- **Phase A**: 已完成
  - `Stateful Semantic IR` 已落地并完成 Phase B 前 hardening
  - 工程边界文档：`tasks/dev_design/stage4_phase_a_semantic_ir.md`
  - 理论并行文档：`tasks/dev_design/stage4_phase_a_formalization_note.md`
  - `stage4_semantic_manifest` `Phase 1-2` 已落地：
    `CollectSemanticManifestSeeds -> ProjectSemanticManifest -> AugmentSemanticManifest`
  - `AnalyzeSemanticStructure` 已改成对 manifest structural evidence 的 manifest-first 消费，
    `fragment_regions` 退化为 compatibility fallback
- **Phase B**: 仍在进行中
  - 已完成的 contract-hardening 子阶段：
    - `Spatial*` object/vocab/shared key 已从 semantic infra 中拆出：
      - `companion_base.{h,cc}`
      - `spatial_vocab.{h,cc}`
      - `spatial_program.{h,cc}`
    - `LowerToSpatialProgram -> ValidateSpatialProgram` 已接入 Blackhole 主线，
      `LowerBlackholeOps` 已硬要求 `tl.spatial_program`
    - `IRModule.global_infos` 现已正式发布：
      - `tl.tt_hardware_model`
      - `tl.spatial_capability_model`
    - `LowerToSpatialProgram` 已消费来自 SoC descriptor 的最小 capability snapshot：
      - layout / partition / flow family 选择经 capability model 收口
      - placement 不再泄漏 `brisc / trisc / ncrisc`，改用 neutral `affinity_kind`
      - `Channel.kind` 已收正为 flow family；`payload_kind / delivery_kind`
        成为显式 contract
    - `ValidateSpatialProgram` 当前口径已收正为：
      structure/coherence/completeness gate，而不是 capability legality pass
    - read-only translator demand probe 已落地：
      `LowerSpatialProgramToTTTargetProbe`
  - 仍未完成的核心职责：
    - `Task` formation / abstract execution role
    - `Flow shaping`
    - `Domain realization`
    - `Phase / ordering synthesis`
    - `ValidateSpatialProgram` 对这些 stronger contract 的 fail-fast 校验
- **Phase C**: 已定义；当前只有准备轨落地
  - `TT Target IR` 已定义
  - read-only translator demand probe 与 hardware intake 已落地
  - 正式 `TTProgram / MaterializeTTExecutableSpec` cutover
    仍受剩余 `Phase B` 工作阻塞

## 当前主 blocker

当前 blocker 先落在剩余 `Phase B`，然后才是正式 `Phase C` cutover：

- `Phase B` 的 contract-hardening 子阶段虽然已经完成，但正文定义的
  spatial synthesis algorithm 与 stronger execution-bearing contract 仍未收实
- `SpatialProgram` 因而还不能视为最终形态的 virtual spatial program
- `LowerSpatialProgramToTTTargetProbe` 已能消费
  `SpatialProgram + TTHardwareModel + SpatialCapabilityModel`，
  但这条路径还只是 read-only intake 验真，不负责物化 target object
- `TTProgram / MaterializeTTExecutableSpec` 仍不存在
- 当前 target/runtime contract 仍主要停留在
  `LowerBlackholeOps -> PlanBlackholeCB -> AssignBlackholeCores -> rt_mod_blackhole`
  的旧主链上

这也是当前 `blackhole.acc` correctness payoff 仍未完全兑现的根因：主 blocker
依然包含 `Phase B` 的 spatial synthesis 本体，而不只是 target materialization。

## 下一步

1. 继续完成 `Phase B` 正文定义的 spatial synthesis algorithm
   - `Task` formation
   - `Flow shaping`
   - `Domain realization`
   - `Phase / ordering synthesis`
   - capability-informed legality / policy
2. 按 `stage4_phase_b_spatial_ir.md` 第 6.5 / 6.6 节继续补强 stronger contract 与 validator
   - 收实 formation / ordering / visibility / materialization basis
   - 让 grouped / paged / routed / chunked 不再被拍平成最小启发式 contract
3. 只有在 `Phase B` 整体完成后，再启动正式 `TTProgram / MaterializeTTExecutableSpec`
   cutover
4. 在 `TTProgram` 主链建立后，再推进 `Phase C2`
   - `flash-attn` `blackhole.acc` correctness payoff
   - 更宽 workload family expansion

## 当前代码事实

- Blackhole 设备侧 pass 主线当前为：
  `LowerDeviceStorageAccessInfo`
  -> `AugmentSemanticManifest`
  -> `LowerIntrin`
  -> `Simplify`
  -> `HoistBroadcastValues`
  -> `SplitBlackholeKernel`
  -> `AnalyzeBlackholeWorkDecomposition`
  -> `AnalyzeBlackholeFragmentRegions`
  -> `AnalyzeBlackholePipelineStages`
  -> `AnalyzeSemanticStructure`
  -> `LiftStatefulSemanticIR`
  -> `ValidateStatefulSemanticIR`
  -> `ValidateSemanticRefinement`
  -> `LowerToSpatialProgram`
  -> `ValidateSpatialProgram`
  -> `LowerBlackholeOps`
  -> `PlanBlackholeCB`
  -> `AssignBlackholeCores`
- `Phase A` 当前稳定事实：
  - `SemanticProgram` / witness algebra / refinement validator 已收口
  - semantic manifest 已进入正式主线，`AnalyzeSemanticStructure` 采用 manifest-first 消费
  - semantic-owned structural evidence 与 direct-runtime binding 已切到 handle-first，
    不再依赖名字匹配恢复语义
  - `fragment_regions` 只剩 residual reduction evidence 与 lowering compatibility
- `Phase B` 当前稳定事实：
  - `SpatialProgram / ProgramPhase / Task / Channel / Layout / WorkPartition /
    Placement / SyncEdge / ResourceIntent` 已落地
  - `LowerToSpatialProgram -> ValidateSpatialProgram` 已接入主线，
    `LowerBlackholeOps` 已硬要求 `tl.spatial_program`
  - `Spatial*` object/vocab/shared key 已从 semantic infra 拆出，
    `TIRAnchor` 与 shared key 由 `companion_base` 承载
  - module-scope hardware/capability snapshot 已落地：
    `tl.tt_hardware_model` / `tl.spatial_capability_model`
  - spatial linkage 已收成 execution-bearing 最小 contract：
    `domain_index`、`target_kind / target_index`、`phase/task/channel linkage payload`、
    `placement.affinity_kind`、以及 channel `payload_kind / delivery_kind`
  - `ValidateSpatialProgram` 现已把
    `Task / Channel / Layout / WorkPartition / Placement / SyncEdge / ResourceIntent.kind`
    收成 closed vocabulary fail-fast，并校验 channel / placement completeness；
    但 validator 仍只做结构/一致性/semantic alignment gate，不做 capability legality
  - `Channel.kind` 当前已是 flow family：
    `point_to_point / broadcast / gather / scatter / reduce_merge / carry`
  - `LowerSpatialProgramToTTTargetProbe` 已落地，明确承担 read-only translator
    intake 诊断，不恢复 non-TT-specific spatial semantics
  - representative family gate 当前覆盖
    `copy / GEMM / flash-attn / topk / chunk_o / fusedmoe_routed / mla_decode_paged`
- 当前仍未完成的事实：
  - `Phase B` 正文定义的 spatial synthesis algorithm 仍未整体收实
  - `Task / Layout / WorkPartition / ProgramPhase / SyncEdge` 的 stronger execution-bearing
    contract 仍未完成
  - `SpatialProgram` 当前只达到了 read-only probe intake 的最小上游 contract，
    还不能视为最终形态的 virtual spatial program
  - `TTProgram / MaterializeTTExecutableSpec` 仍不存在
  - `Phase C` 还没有启动正式 cutover 实现，也还没有证明
    `SpatialProgram -> TTProgram` 的单一真源切换
  - `flash-attn` 的 `blackhole.acc` correctness payoff 仍归属 `Phase C2`

## 最新验证摘要

- `pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -q`
  - `40 passed`
- `pytest tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py -q`
  - `7 passed`
- `pytest tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py -q`
  - `44 passed`
- `pytest tilelang_repo/testing/python/transform/test_blackhole_tt_target_probe.py -q`
  - `6 passed`
- `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py -q -k test_blackhole_copy_pass_attrs`
  - `1 passed`
- `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -q -k test_blackhole_gemm_pipeline_attaches_spatial_program`
  - `1 passed`
- `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q -k test_flash_attention_forward_pipeline_attaches_multi_phase_spatial_program`
  - `1 passed`
- `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py -q`
  - `41 passed, 10 skipped, 1 xfailed`
- `source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh && export TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo && pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py -q`
  - `12 passed`
- `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -q`
  - `26 passed, 11 skipped`
- `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py -q`
  - `1 passed`
- `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q`
  - `27 passed`

说明：

- `Phase A` compile-path 和 semantic gate 当前稳定
- `Phase B` contract hardening / probe / hardware intake 当前稳定
- `Phase B` 整体仍未完成；当前主工作仍是 spatial synthesis 本体与 stronger contract
- `flash-attn` correctness 仍不应被写成已完成稳定基线

## 当前文档入口

- `tasks/dev_design/final_blackhole_backend_redesign.md`
- `tasks/dev_design/stage4_phase_b_spatial_ir.md`
- `tasks/dev_design/stage4_phase_c_tt_target_ir.md`
- `tasks/dev_design/stage4_semantic_manifest.md`
- `tasks/dev_design/stage4_phase_a_semantic_ir.md`
- `tasks/dev_design/stage4_phase_a_formalization_note.md`
- `tasks/dev_design/stage4_stage0_guardrails.md`
- `tasks/dev_design/README.md`

## 说明

- `progress.md` 现在只保留相对稳定的阶段状态、主 blocker、当前下一步和验证摘要。
- 详细逐步实现记录、历史 checklist 和理论证明草稿分别留在阶段文档、git history 与 formalization note 中。
- `2026-04-07` 已再次完成一轮活动文档状态审计：
  `README.md`、`AGENTS.md`、`CLAUDE.md`、active stage docs 已统一到
  `Phase A` 完成、`Phase B` contract-hardening 子阶段已收口但整体仍未完成、
  当前主实施重点回到 `Phase B` 正文定义的 spatial synthesis algorithm、
  以及 manifest-first、正式 pass 链与 `fragment_regions` 残余职责的当前口径。
  同一轮也已把 `Stage 0 / Phase A / Phase C` 文档里的旧 checklist、
  一次性验证数字和 task/step 计划体压回长期有效的边界描述。
- `stage4_semantic_manifest.md` 当前作为 `Phase A` 信息源重构文档保留：
  - 它的定位是把 `Phase A` 需要的 explicit-op evidence 从 late matcher 前移到真实销毁边界：
    - `LowerTileOp` 边界的 early capture
    - `SplitHostDevice` 之后的显式 projection
    - device-side pre-`LowerIntrin` 的 residual augment
  - semantic recovery 主体仍然只在 `AnalyzeSemanticStructure -> LiftStatefulSemanticIR -> Validate*`
  - 当前 `Phase 1-2` 已实施；`selection / arg-reduce / recurrence` structural evidence
    已前移到 manifest 并改成 manifest-first 消费
  - `fragment_regions` 当前仍保留 `row_reductions` 与 residual compatibility 职责
  - `structural_regions` / `fragment_regions` 当前的 state/edge entry 已扩成
    `name + typed Buffer handle` 双通道 schema；后续 consumer 新逻辑默认应吃 handle-first
  - `fragment_regions` 的剩余存在理由已经收敛：
    - semantic 侧：`reduce_*` update 的 residual evidence
    - lowering 侧：`LowerBlackholeOps` 当前的 fragment subset / legality summary
  - 后续删除顺序应是：
    先迁走 `row_reductions` 的 semantic-owned 部分，再给 lowering 建独立 summary，
    最后删除 `fragment_regions`
  - 它不是新的跨层真源，也不改变 `Phase B / C` 只消费冻结后 companion IR 的边界
- `final_blackhole_backend_redesign.md` 已在 `2026-04-06` 按当前执行状态刷新：
  - 已收成轻量总体设计文档
  - `Phase A/B/C` 细节默认下沉到对应 stage 文档
