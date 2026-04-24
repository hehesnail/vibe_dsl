# TileLang Blackhole Backend Progress

> 当前唯一总体设计文档:
> `tasks/dev_design/final_blackhole_backend_redesign.md`

> 已完成 cleanup 的边界索引:
> `tasks/dev_design/2026-04-16-blackhole-final-legacy-protocol-cleanup.md`

> 本文件只记录 repo HEAD 当前状态与下一步。

## 1. 当前总体状态

- **日期**: `2026-04-24`
- **当前总体 blocker**:
  cleanup blocker
  已解除；
  当前活动 lane
  是
  `support surface / workload payoff`
  deferred lane
  重新打开。
  `Legacy Protocol Deletion / cleanup task5 convergence`
  已在 repo HEAD 完成；
  当前后续工作
  回到
  `SpatialPlan`
  logical live-value /
  materialization-boundary
  first-class 表示的消费与细化、
  leaf contract-family typed 化、
  runtime-support-surface
  扩展
- **当前推进原则**:
  layered IR 主链
  `Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec`
  已完成收口；
  这里的 completed
  只表示 broad legacy protocol /
  public wrapper /
  cross-pass bag
  已退出 active chain，
  不表示所有 leaf compatibility
  payload /
  runtime fallback
  已经删除；
  后续只在
  explicit representation admission /
  runtime gate /
  workload payoff
  上推进，
  不重新引入
  legacy attr /
  bag /
  wrapper

## 2. 当前主线任务状态

### 2.1 主线任务看板

- `Task 1: SpatialPlan Representation Cutover`
  - 状态：`completed`
  - 当前结论：
    `SpatialPlan`
    已成为
    target-independent
    virtual spatial/dataflow
    owner truth；
    `BuildSpatialPlan`
    已成为
    唯一 public direct builder，
    `AnalyzeSpatialStructureFacts` /
    `BuildSpatialPlanCompanion` /
    `tl.spatial_structure_facts`
    已退出
    active chain
    和 public surface；
    producer-side logical bridge handoff
    已从
    `blackhole.compute_regions`
    切到 direct capture
- `Task 2: TTProgram Representation Cutover`
  - 状态：`completed`
  - 当前结论：
    `TTProgram`
    已成为
    target planning
    的唯一 TT-specific realization owner truth；
    `BuildTTProgram`
    已退成 staged slice aggregation /
    finalize /
    cleanup 点，
    `ValidateTTProgram`
    已成为
    exact TT-Metal legality
    的正式 hard gate
- `Task 3: ExecutableSpec / Leaf Reader Cutover`
  - 状态：`completed boundary cutover / leaf contract-family deletion remains open debt`
  - 当前结论：
    `MaterializeBlackholeExecutable`
    已成为
    唯一 executable writer；
    build / codegen / runtime /
    `BlackholeModule`
    已只站在
    `tl.blackhole_executable`
    / `ExecutableSpec`
    projection 边界上；
    `blackhole.copy_semantics`
    与 cross-pass
    `blackhole.segment_kind`
    已退出 active path，
    leaf body slicing
    已改成
    executable-kind +
    lowered builtin
    direct structural extraction；
    但
    `compute_contract` /
    `gemm_contract` /
    `multi_*_contracts`
    仍作为 leaf compatibility
    metadata /
    runtime fallback
    存活，
    不属于 task3
    终态 leaf schema
- `Legacy Protocol Deletion`
  - 状态：`completed broad-protocol convergence / narrow leaf debt remains`
  - 说明：
    cleanup `task0-task5`
    是 overlap residue workstream，
    不是另一条主路线
  - 当前 convergence 备注：
    `BuildSpatialPlan`
    已不再依赖
    `tl.tileop.gemm_py`
    / `arg[2]`
    这类 op-specific 恢复；
    statement 级 read/write
    现在直接从
    `tl.region` access mask
    和 tileop typed
    `compute_consume`
    contract 恢复。
    `BlackholeDeviceResourceCanonicalization`
    也已删掉
    GEMM-only accumulator fallback，
    debug waypoint emission
    已不再按
    `scores_max / acc_o / ...`
    这类 workload buffer 名分支
    `blackhole.resource_plan`
    /
    `tl.internal_tt_*`
    已从
    `tilelang_repo/src`
    和
    `tilelang_repo/tilelang`
    的定义面退出；
    `blackhole.lowering_requirements`
    broad bag
    与
    `AnalyzeBlackholeLoweringSupportFacts`
    旧命名
    也已退出 active chain，
    只剩
    pass-local
    lowering collector /
    strip-before-leaf
    mechanics

### 2.2 cleanup overlap 看板

- `Cleanup Task 1`
  - 状态：`completed`
  - 已完成：
    `SpatialPlan`
    已收成
    `BuildSpatialPlan`
    direct builder；
    public wrapper /
    facts object /
    facts attr
    已删除；
    direct logical bridge capture
    已落到
    `CaptureBlackholeLogicalBridgeSpecs`
    +
    `tl.blackhole_logical_buffer_tile_bridge_specs`
- `Cleanup Task 2`
  - 状态：`completed`
  - 已完成：
    public/internal
    `AnalyzeBlackhole*`
    analysis surface 删除，
    `tl.blackhole_lowering_requirements_seed`
    已退出 active chain，
    `blackhole_lowering_requirements`
    已收成
    current TIR + `SpatialPlan`
    direct、pass-local analysis facts
- `Cleanup Task 0`
  - 状态：`completed`
  - 已完成：
    `blackhole.cb_requirements`
    已退出 active chain，
    staged CB contract
    已改由
    `TTProgram.cb_plans`
    / typed `TTCBPlan`
    承载，
    unresolved
    `unsupported_compute_ops`
    已由
    `ValidateTTProgram`
    fail-close 拒绝
- `Cleanup Task 3`
  - 状态：`completed`
  - 已完成：
    `AnnotateBlackholeCopySemantics`
    public wrapper /
    active prepass /
    实现文件
    已删除，
    compiler-side consumer
    已回到
    current TIR /
    direct structural recovery
- `Cleanup Task 4`
  - 状态：`completed`
  - 已完成：
    cross-pass
    `blackhole.segment_kind`
    已退出 active chain，
    `SplitBlackholeKernel`
    不再发 marker，
    planner 侧 marker
    只保留 pass-local mechanics
    并在最终 body strip，
    `SegmentBodyExtractor`
    已改为
    基于 lowered builtin
    的 structural slicer
- `Cleanup Task 5`
  - 状态：`completed overlap / convergence gate 已收口`

### 2.3 Deferred lane

- support surface /
  workload payoff 扩展
  已解除冻结；
  `SpatialPlan`
  logical live-value /
  materialization-boundary
  一等表示已完成第一轮落地，
  当前 dataflow edge
  会生成
  `LiveValue` /
  `LiveValueEdge` /
  `MaterializationBoundary`
  并由
  `ValidateSpatialPlan`
  fail-close
  检查；
  live-form /
  materialization
  的 TT physical /
  leaf owner truth
  已完成第一轮收口；
  `SpatialPlan`
  logical live-value /
  materialization-boundary
  后续仍需继续被
  `PlanTT*`
  消费并细化到更宽 live-in /
  recurrence /
  workload payoff
  支持面；
  `fragment_fill -> cast -> publish`
  的 constant fill
  `thread_distributed + cb_republish`
  已通过
  `pack_thread_direct_store`
  晋级为 admitted
  bf16 direct runtime；
  GEMM post-merge
  direct cast consumer
  的 zero-preclear
  full-tile shape
  已通过
  `pack_tile`
  晋级为 admitted
  bf16 direct runtime；
  当前剩余工作
  按顺序收束为：
  先让
  `PlanTT*`
  消费并细化
  `SpatialPlan`
  live-value /
  materialization-boundary
  schema，
  再收敛 leaf contract-family
  typed schema，
  然后扩大
  materialization admission
  支持面与 workload payoff；
  当前任务级设计
  已固定为
  `tasks/dev_design/2026-04-23-blackhole-live-form-materialization-admission.md`

## 3. repo HEAD 当前代码现状

- 当前长期主链固定为：

```text
Normalized Tile TIR
  -> SpatialPlan
  -> TTProgram
  -> ExecutableSpec
```

- repo HEAD 当前 active pass/phase 实现链是：

```text
Normalized Tile TIR
  -> BuildSpatialPlan
  -> ValidateSpatialPlan
  -> SplitBlackholeKernel
  -> CaptureBlackholeLogicalBridgeSpecs
  -> PlanTTBlocks
  -> SelectBlackholeTTMetalBuiltins
  -> PlanTTCompute / PlanTTTransport / PlanTTSync / PlanTTABI / PlanTTExecution
  -> BuildTTProgram
  -> ValidateTTProgram
  -> MaterializeBlackholeExecutable
```

- `AnalyzeBlackholeWorkDecomposition`
  / `AnalyzeBlackholeComputeRegions`
  / `AnalyzeBlackholePipelineStages`
  public wrapper、
  C++ pass registration、
  对应 legacy test
  已从 repo HEAD 删除
- `BuildSpatialPlan`
  当前 statement access
  恢复逻辑
  已改成：
  `tl.region` access mask
  负责通用 read/write edge，
  tileop typed
  `GetDataflowAccessInfo()`
  只负责
  `compute_consume`
  contract；
  不再在 generic pass
  里按
  `tl.tileop.gemm_py`
  做 closure role /
  write-edge 特判
- `BlackholeDeviceResourceCanonicalization`
  当前只接受
  fragment/layout 结构证据
  来把 local buffer
  提升到
  `blackhole.acc`；
  `gemm_py`
  fallback
  已退出
- `codegen_blackhole`
  的 debug waypoint
  当前只保留 generic op-kind tags；
  workload-private
  buffer 名
  已不再进入
  debug/source contract
- `SpatialPlan`
  现已拥有 typed
  `live_values` /
  `live_value_edges` /
  `materialization_boundaries`
  slices；
  `BuildSpatialPlan`
  会从当前 TIR
  buffer metadata
  和 dataflow edges
  构造 logical live-value
  及跨 phase /
  same phase
  materialization boundary，
  `ValidateSpatialPlan`
  会检查引用完整性、
  subject /
  producer /
  consumer
  与 dataflow edge
  一致性、
  shape / dtype
  以及 phase boundary
  visibility
- `AnalyzeSpatialStructureFacts`
  / `BuildSpatialPlanCompanion`
  public wrapper、
  C++ pass registration、
  `tl.spatial_structure_facts`
  facts attr / object
  已从 repo HEAD 删除
- `lower.py`
  / target test helper
  已切到
  `LowerToBlackholePhaseB`
  direct capture，
  不再通过
  `AnalyzeBlackholeComputeRegions`
  / `blackhole.compute_regions`
  handoff
- `blackhole.compute_regions`
  / `blackhole.work_decomposition`
  / `blackhole.pipeline_stages`
  已退出 active chain
- `SplitBlackholeKernel`
  已接入主链：
  pure copy 走 `fused_dataflow` 单 kernel，
  GEMM 走 3-kernel（reader / compute / writer），
  且不再发出
  cross-pass
  `blackhole.segment_kind`
  marker
- exact TT-Metal builtin selector
  已接入 active chain
- `AnnotateBlackholeCopySemantics`
  Python wrapper /
  FFI registration /
  实现文件
  已从 repo HEAD 删除
- `SegmentBodyExtractor`
  已切到
  segment kind +
  lowered builtin
  direct structural slicing，
  不再读
  source-level marker
- Blackhole 正式执行路径只剩
  `BlackholeModule`
  进程内 direct host path
- `TTProgram`
  现已拥有 typed
  `live_form_plans` /
  `materialization_plans` /
  `consumer_binding_plans`
  slices；
  `ValidateTTProgram`
  会检查 live-form 引用、
  materialization protocol、
  consumer binding
  与 typed CB 资源的一致性
- `MaterializeBlackholeExecutable`
  现会把上述 typed slices
  投影到
  `ExecutableSpec`
  /
  runtime metadata；
  leaf reader
  不再需要从 TIR body /
  builtin 序列 /
  buffer 名
  恢复 fragment/cast/publish
  live form
- `fragment_fill -> cast -> publish`
  与
  GEMM post-merge cast consumer
  现在都能在 projection 层暴露
  `thread_distributed_slice -> cb_materialized_tile`
  的
  `cb_republish`
  materialization owner truth
- `TTMaterializationPlan`
  /
  `ExecutableSpec.materialization_plans`
  /
  `buffer_materializations`
  现显式携带
  `publication_protocol`；
  `materialization_protocol=cb_republish`
  表示 consumer-visible
  CB live form，
  `publication_protocol`
  表示 producer-side
  device publication 方式
- `fragment_fill -> cast -> publish`
  的 constant fill case
  现由当前 IR 中的
  `tl.blackhole.fill_fragment`
  事实推出
  `publication_protocol=pack_thread_direct_store`，
  codegen 使用 PACK thread
  直接写 reserved CB page，
  不再使用 mailbox write-pointer transfer；
  该 case 已进入
  bf16 TT-Sim
  direct runtime correctness gate
- GEMM post-merge cast consumer
  的 zero-preclear
  full-tile shape
  现由当前 IR
  zero fill fact
  和 typed materialization contract
  共同 admitted：
  merge 侧不再走 accumulator reload
  mailbox helper，
  而是从 partials CB
  copy 到 DST register
  后用
  `pack_tile`
  发布
  `D_local`
  materialized CB；
  planner 仍会在 matmul /
  merge /
  add /
  reduction /
  scalar update /
  cast 等 producer
  写目标时失效旧 constant-fill fact，
  防止 preclear fill
  被误当成后续 cast source truth
  或穿过非零 live-in merge
  错误 admitted

## 4. 当前显式 Debt / 非 Blocker

- `buffer_tile_bridge_specs`
  /
  `unsupported_compute_ops`
  /
  `compute_contract`
  /
  `gemm_contract`
  /
  `multi_*_contracts`
  仍作为
  executable projection
  / leaf compatibility
  metadata
  存在；
  它们已经不是
  `TTProgram`
  owner truth，
  也不再作为
  task5 blocker，
  后续只允许按
  explicit leaf debt
  继续收敛
- `tl.blackhole_logical_buffer_tile_bridge_specs`
  仍是当前唯一窄 bridge attr；
  它不是
  `SpatialPlan`
  /
  `TTProgram`
  /
  `ExecutableSpec`
  长期边界。
  `SpatialPlan`
  logical live-value /
  materialization-boundary
  第一轮 schema
  已存在，
  但该窄 bridge
  仍要等
  layout /
  materialization boundary
  和 typed executable
  materialization schema
  全面承接后删除
- `compute_contract` /
  `gemm_contract` /
  `multi_*_contracts`
  仍存在
  `TTProgram.payload -> tl.blackhole_executable -> ExecutableSpec -> runtime`
  compatibility 链；
  `ValidateTTProgram`
  当前对它们做 shape check
  只表示 debt containment，
  不是 owner truth。
  required end-state
  是 typed
  `TTProgram`
  compute /
  ABI /
  materialization
  slices
  和 typed
  `ExecutableSpec`
  records
  承接这些事实，
  删除 runtime
  `compute_contract <- gemm_contract`
  fallback
- `fragment_fill -> cast -> publish`
  /
  direct cast consumer
  的
  live-form /
  materialization
  owner truth
  已进入
  `TTProgram`
  和
  `ExecutableSpec`；
  constant fill
  `fragment_fill -> cast -> publish`
  已用
  `pack_thread_direct_store`
  admitted；
  zero-preclear
  GEMM post-merge
  direct cast consumer
  已用
  `pack_tile`
  admitted；
  非零 live-in /
  更宽 fragment materialization
  与后续 workload payoff
  仍需继续按显式 IR
  扩支持面
- `flash-attn` direct runtime
  仍不在当前
  correctness gate；
  它属于
  materialization admission
  之后的
  workload payoff
  lane

## 5. 当前稳定基线

- `Legacy Protocol Deletion / cleanup task5`
  convergence baseline
  当前已通过：
  - `cd tilelang_repo && cmake --build build -j32`
  - `cd tilelang_repo && pytest -q testing/python/transform/test_blackhole_spatial_ir.py testing/python/target/blackhole/test_blackhole_copy_pipeline.py testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py`
  - `source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh && export TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo && cd /root/dev/vibe_dsl/tilelang_repo && pytest -q testing/python/target/blackhole/test_blackhole_copy_runtime.py`
  - `source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh && export TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo && cd /root/dev/vibe_dsl/tilelang_repo && pytest -q testing/python/target/blackhole/test_blackhole_gemm.py`
- `live-form / materialization owner truth`
  baseline
  当前已通过：
  - `cd tilelang_repo && cmake --build build -j32`
  - `PYTHONPATH=/root/dev/vibe_dsl/tilelang_repo pytest -q testing/python/target/blackhole/test_blackhole_gemm.py`
  - `PYTHONPATH=/root/dev/vibe_dsl/tilelang_repo pytest -q testing/python/target/blackhole/test_blackhole_copy_pipeline.py testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py`
  - `source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh && export TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo && cd /root/dev/vibe_dsl/tilelang_repo && PYTHONPATH=/root/dev/vibe_dsl/tilelang_repo pytest -q testing/python/target/blackhole/test_blackhole_copy_runtime.py`
- `thread_distributed + cb_republish`
  non-mailbox admission
  baseline
  当前已通过：
  - `cd tilelang_repo && cmake --build build -j32`
  - `PYTHONPATH=/root/dev/vibe_dsl/tilelang_repo TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_fragment_fill_cast_publish_exposes_typed_live_form_owner_truth tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_fragment_fill_cast_publish_projects_leaf_materialization_plans tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_fragment_fill_cast_publish_admits_non_mailbox_cb_republish tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_gemm_post_merge_cast_consumer_uses_pack_tile_materialization tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_gemm_post_merge_cast_consumer_without_zero_preclear_keeps_materialization_gate`
  - `source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh && export TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo && cd /root/dev/vibe_dsl/tilelang_repo && PYTHONPATH=/root/dev/vibe_dsl/tilelang_repo pytest -q testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_fragment_fill_cast_publish_runtime`
  - `source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh && export TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo && cd /root/dev/vibe_dsl/tilelang_repo && PYTHONPATH=/root/dev/vibe_dsl/tilelang_repo pytest -q testing/python/target/blackhole/test_blackhole_gemm.py`
- direct runtime 当前 admitted 支持面：
  - copy：equal source/dest range，且 stride = 1
  - GEMM：A/B-separated reader range + writer output range；
    fresh fragment / preclear zero-init
    统一走 `clear_accum=true` direct path；
    zero-preclear
    GEMM post-merge cast consumer
    支持
    `publication_protocol=pack_tile`
    的 bf16 materialized output
  - accessor：仅 interleaved + DRAM +
    `common_runtime_arg_count = 0`
  - communication：non-oversubscribed explicit semaphore /
    remote-endpoint subset
- `flash-attn`
  compile-path / source/spec baseline 已稳定，
  但 direct runtime correctness
  还不是 admitted support surface
- direct cast consumer
  和
  `fragment_fill -> cast -> publish`
  当前拥有 typed
  live-form /
  materialization projection
  与 explicit
  publication protocol；
  constant fill
  `fragment_fill -> cast -> publish`
  已进入
  TT-Sim hard-execute
  correctness gate，
  zero-preclear
  direct cast consumer
  也已进入
  TT-Sim hard-execute
  correctness gate；
  无 zero-preclear /
  非零 live-in
  仍保持 explicit
  unsupported gate
- `SpatialPlan logical live-value / materialization-boundary`
  baseline
  当前已通过：
  - `cd tilelang_repo && cmake --build build -j32`
  - `PYTHONPATH=/root/dev/vibe_dsl/tilelang_repo TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo pytest -q tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py`
  - `PYTHONPATH=/root/dev/vibe_dsl/tilelang_repo TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py tilelang_repo/testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py`
  - `source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh && export TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo && PYTHONPATH=/root/dev/vibe_dsl/tilelang_repo pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_fragment_fill_cast_publish_exposes_typed_live_form_owner_truth tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_fragment_fill_cast_publish_projects_leaf_materialization_plans tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_fragment_fill_cast_publish_admits_non_mailbox_cb_republish tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_gemm_post_merge_cast_consumer_uses_pack_tile_materialization tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_gemm_post_merge_cast_consumer_without_zero_preclear_keeps_materialization_gate`

## 6. 当前下一步

当前下一步固定为：

1. 先让
   `PlanTT*`
   消费并细化
   已落地的
   `SpatialPlan`
   logical live-value /
   materialization-boundary
   表示；
   不能让
   `PlanTT*`
   或 leaf reader
   继续用 body-order matcher
   承担跨阶段 owner truth
2. 将
   `compute_contract` /
   `gemm_contract` /
   `multi_*_contracts`
   从 payload/fallback
   收敛到 typed
   `TTProgram`
   和
   `ExecutableSpec`
   schema，
   然后删除 runtime fallback
3. 在上述 typed owner truth
   基础上扩大
   materialization admission
   支持面：
   非零 live-in merge、
   更宽 fragment/cast producer
   和后续 workload payoff
   都必须继续通过
   explicit representation
   boundary
   表达，
   不回退到 leaf matcher
4. 保持
   compile / projection /
   admitted runtime
   gate
   继续只站在
   explicit representation
   boundary 上
5. 等 layout /
   materialization boundary
   和 typed executable
   materialization schema
   全面承接后，
   删除窄 bridge attr
   `tl.blackhole_logical_buffer_tile_bridge_specs`
6. flash-attn direct runtime
   只作为上述 admission
   完成后的 integration payoff，
   不作为当前设计驱动
