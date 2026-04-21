# TileLang Blackhole Backend Progress

> 当前唯一总体设计文档:
> `tasks/dev_design/final_blackhole_backend_redesign.md`

> 当前 cleanup 执行总览:
> `tasks/dev_design/2026-04-16-blackhole-final-legacy-protocol-cleanup.md`

> 本文件只记录 repo HEAD 当前状态与下一步。

## 1. 当前总体状态

- **日期**: `2026-04-22`
- **当前总体 blocker**:
  `Task 2: TTProgram Representation Cutover`
  还没收口；
  当前最直接的 repo HEAD blocker
  是和它重叠的
  cleanup task2/task0 residue
- **当前推进原则**:
  主线固定按
  `Task 2 -> Task 3 -> Legacy Protocol Deletion`
  推进；
  `Task 1`
  已在 repo HEAD 收口

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
    producer-side logical bridge handoff
    已从
    `blackhole.compute_regions`
    切到 direct capture
- `Task 2: TTProgram Representation Cutover`
  - 状态：`in progress / 当前主线`
  - 当前直接 blocker：
    legacy planning bag /
    exact builtin legality residue /
    `TTProgram.payload`
    compatibility residue
  - 当前目标：
    让 `TTProgram`
    真正成为唯一
    TT-specific realization
- `Task 3: ExecutableSpec / Leaf Reader Cutover`
  - 状态：`pending / 受 Task 2 阻塞`
  - 当前目标：
    让 build / codegen / runtime /
    `BlackholeModule`
    只站在
    `ExecutableSpec`
    projection 边界上
- `Legacy Protocol Deletion`
  - 状态：`pending`
  - 说明：
    cleanup `task0-task5`
    是 overlap residue workstream，
    不是另一条主路线

### 2.2 cleanup overlap 看板

- `Cleanup Task 1`
  - 状态：`completed`
  - 已完成：
    direct logical bridge capture
    已落到
    `CaptureBlackholeLogicalBridgeSpecs`
    +
    `tl.blackhole_logical_buffer_tile_bridge_specs`
- `Cleanup Task 2`
  - 状态：`active overlap / 当前卡在 Task 2`
  - 已完成：
    public/internal
    `AnalyzeBlackhole*`
    analysis surface 删除
  - 当前剩余：
    `blackhole_lowering_requirements`
    /
    `tl.blackhole_lowering_requirements_seed`
    /
    active planner bag residue 删除
- `Cleanup Task 0`
  - 状态：`active overlap / 当前与 Task 2 并行`
  - 当前剩余：
    exact builtin legality /
    `blackhole.cb_requirements`
    residue 删除
- `Cleanup Task 3`
  - 状态：`pending overlap / 对应 Task 2-Task 3`
  - 目标：删除
    `blackhole.copy_semantics`
- `Cleanup Task 4`
  - 状态：`pending overlap / 对应 Task 3`
  - 目标：删除
    `blackhole.segment_kind`
    并收掉 leaf-local body slicing residue
- `Cleanup Task 5`
  - 状态：`pending overlap / 最终 convergence gate`

### 2.3 Deferred lane

- support surface /
  workload payoff 扩展
  当前冻结；
  只有 Task 2 / Task 3
  主链收口后才恢复

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
  -> AnalyzeSpatialStructureFacts
  -> BuildSpatialPlanCompanion
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
  GEMM 走 3-kernel（reader / compute / writer）
- exact TT-Metal builtin selector
  已接入 active chain
- Blackhole 正式执行路径只剩
  `BlackholeModule`
  进程内 direct host path

## 4. 当前未收口项

- `Task 2 / TTProgram`
  - `blackhole_lowering_requirements.cc`
    虽然已经从
    `AnalyzeBlackhole*Evidence(...)`
    切到
    `SpatialPlan + current TIR`
    direct analysis，
    但 broad bag
    仍然存在
  - `tl.blackhole_lowering_requirements_seed`
    仍在 active chain
  - `blackhole.cb_requirements`
    仍在 active chain
  - selector / validator
    还没共用一份
    exact builtin legality contract
  - `buffer_tile_bridge_specs`
    /
    `unsupported_compute_ops`
    /
    `compute_contract`
    /
    `gemm_contract`
    等 payload residue
    仍停在
    `TTProgram.payload`
    和 leaf compatibility path
- `Task 3 / ExecutableSpec / leaf reader`
  - `blackhole.copy_semantics`
    仍在 compiler-side
    active chain
  - `blackhole.segment_kind`
    仍在 planner / leaf
    两侧存活
  - `SegmentBodyExtractor`
    仍按 marker
    切 raw body
  - leaf reader / runtime / codegen
    还没只站在
    `ExecutableSpec`
    projection 边界上

## 5. 当前稳定基线

- `Task 1`
  相关结构层 / target helper / flash-attn target pipeline
  当前基线已通过：
  - `cmake --build tilelang_repo/build -j32`
  - `pytest -q tilelang_repo/testing/python/transform -k blackhole`
  - `pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py`
- direct runtime 当前 admitted 支持面：
  - copy：equal source/dest range，且 stride = 1
  - GEMM：A/B-separated reader range + writer output range；
    fresh fragment / preclear zero-init
    统一走 `clear_accum=true` direct path
  - accessor：仅 interleaved + DRAM +
    `common_runtime_arg_count = 0`
  - communication：non-oversubscribed explicit semaphore /
    remote-endpoint subset
- `flash-attn`
  compile-path / source/spec baseline 已稳定，
  但 direct runtime correctness
  还不是 admitted support surface

## 6. 当前下一步

当前下一步固定为：

1. 推进 `Task 2: TTProgram Representation Cutover`
   - 删除
     `blackhole_lowering_requirements`
     /
     `tl.blackhole_lowering_requirements_seed`
     这组 active planning bag
   - 收掉
     `blackhole.cb_requirements`
     和 exact builtin legality residue
   - 把仍然需要跨边界保留的 leaf/build gate data
     统一收回
     `TTProgram`
2. 然后推进
   `Task 3: ExecutableSpec / Leaf Reader Cutover`
   - 删除
     `blackhole.copy_semantics`
   - 删除
     `blackhole.segment_kind`
   - 收紧 leaf reader /
     runtime /
     codegen 边界
3. 最后做
   `Legacy Protocol Deletion`
   的 convergence /
   scans /
   verification /
   delivery 收尾
