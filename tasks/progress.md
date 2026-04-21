# TileLang Blackhole Backend Progress

> 当前唯一总体设计文档:
> `tasks/dev_design/final_blackhole_backend_redesign.md`

> 当前 cleanup 执行总览:
> `tasks/dev_design/2026-04-16-blackhole-final-legacy-protocol-cleanup.md`

> 本文件只记录 repo HEAD 当前状态与下一步。
> task 文档里的合同、旧文档里的 landed/completed 表述、以及 repo HEAD 上已有代码，
> 都不能直接当成当前设计口径下的完成声明。

## 1. 当前总体状态

- **日期**: `2026-04-22`
- **当前总体 blocker**:
  cleanup 主线 `Cleanup Task 0-5`
  还没收口；
  repo HEAD 当前活跃 blocker
  队列从 `Cleanup Task 1`
  开始，
  但 `Cleanup Task 0`
  仍有必须回收的
  full-contract debt
- **当前推进原则**:
  把长期合同层
  和当前 cleanup 执行切片
  分开看；
  先完成 cleanup 主线，
  再恢复 support surface / workload payoff 扩展

## 2. 当前路线状态

### 2.1 长期合同 lane

- `SpatialPlan` 合同
  - 未收口
  - 主要被 `Cleanup Task 1 / 2` 阻塞
- `TTProgram` 合同
  - 未收口
  - 主要被 `Cleanup Task 0 / 2 / 3 / 4` 阻塞
- `ExecutableSpec / leaf reader` 合同
  - 未收口
  - 主要被 `Cleanup Task 3 / 4 / 5` 阻塞
- `Legacy Protocol Deletion`
  - 未收口
  - 需要 `Cleanup Task 0-5` 全部完成

### 2.2 cleanup 当前任务看板

- `Cleanup Task 1`
  - 状态：`当前第一 blocker / 应立即推进`
  - 目标：把 logical bridge handoff
    从 `blackhole.compute_regions`
    切到 direct capture
- `Cleanup Task 2`
  - 状态：`当前第二 blocker / 待 Task 1 后推进`
  - 目标：删除 public/internal
    legacy analysis carrier
  - 说明：它完成后，
    `Task 0`
    剩余 full-contract work
    必须回到关键路径
- `Cleanup Task 0`
  - 状态：`partial landed / 未完成 / 待 Task 2 后立即回收`
  - 当前只完成：
    `selector-forwarding`
    局部前移
  - 仍未完成：
    exact builtin selection /
    legality /
    `blackhole.cb_requirements`
    删除
- `Cleanup Task 3`
  - 状态：`pending / 待 Task 1 + Task 2 + Task 0-remain 后推进`
  - 目标：删除
    `blackhole.copy_semantics`
- `Cleanup Task 4`
  - 状态：`pending / 待 Task 3 后推进`
  - 目标：删除
    `blackhole.segment_kind`
    并收掉 leaf-local body slicing residue
- `Cleanup Task 5`
  - 状态：`pending / 最终 convergence gate`
  - 目标：统一文档 /
    residue scans /
    verification /
    delivery
  - 说明：不是新的协议 owner

### 2.3 Deferred lane

- support surface /
  workload payoff 扩展
  当前冻结；
  只有 cleanup 路线收口后
  才恢复

说明：

- `task1_spatial_plan_companion.md`
- `task2_ttprogram_companion_cutover.md`
- `task3_runtime_gate_and_workload_cutover.md`

这组文件
只定义长期 end-state /
完成判据，
不再直接充当
repo HEAD 的执行路线图。

当前真正的执行切片
以 cleanup `task0-task5`
为准。

## 3. repo HEAD 当前代码现状

- 当前实际 active chain
  仍是 cleanup 文档和 `AGENTS.md`
  里记录的那条 pass/phase 实现链；
  这描述的是当前实现顺序，
  不是新的长期 IR 层
- 当前长期主链仍然只有：

```text
Normalized Tile TIR
  -> SpatialPlan
  -> TTProgram
  -> ExecutableSpec
```

- exact TT-Metal builtin selector
  已接入 active chain
- helper/composite builtin residue
  与 `compute_epilogue_ops`
  已退出 active compute 主链
- `SplitBlackholeKernel`
  已接入主链：
  pure copy 走 `fused_dataflow` 单 kernel，
  GEMM 走 3-kernel（reader / compute / writer）
- Blackhole 正式执行路径只剩
  `BlackholeModule`
  进程内 direct host path
- `tilelang.compile(..., execution_backend="tvm_ffi")`
  的 Blackhole wrapper/export path 已恢复

上面这些都只是当前 repo HEAD 的现状，
不表示顶层任务已经收口。

## 4. 当前未收口项（按 cleanup task 归类）

- `Cleanup Task 0`
  - `SelectBlackholeTTMetalBuiltins`
    仍是
    `PlanTTKernelABI::SelectComputeBuiltins()`
    的薄前门
  - `blackhole.cb_requirements`
    仍在 active chain
  - selector / validator
    还没共用一份
    exact builtin legality contract
- `Cleanup Task 1`
  - `blackhole.compute_regions`
    仍是
    logical bridge handoff
    的 producer-side owner truth
  - `lower.py`
    / target test helper
    仍从 broad bag
    抽
    `buffer_tile_bridge_specs`
- `Cleanup Task 2`
  - public
    `AnalyzeBlackhole*`
    wrappers
    仍暴露
  - internal
    `*Evidence(...)`
    helpers
    仍被 consumer 直接读取
  - `blackhole.lowering_requirements`
    仍由
    `blackhole_lowering_requirements.cc`
    聚合成 broad bag
- `Cleanup Task 3`
  - `AnnotateBlackholeCopySemantics`
    /
    `blackhole.copy_semantics`
    仍在 active chain
  - `blackhole.resource_plan`
    仍由
    `BlackholeDeviceResourceCanonicalization`
    发出
- `Cleanup Task 4`
  - `blackhole.segment_kind`
    仍在 planner / leaf
    两侧存活
  - `SegmentBodyExtractor`
    仍按 marker
    切 raw body
- cross-task leaf compatibility debt
  - `buffer_tile_bridge_specs`
    仍**错误地**同时停在
    `TTProgram.payload`
    /
    executable projection
    /
    `codegen_blackhole.cc`
    reader 上
  - `PlanTTSync / PlanTTABI / PlanTTExecution`
    虽已落地为显式 planner passes，
    但 leaf reader / cleanup
    还没同时收口

## 5. 当前稳定基线

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
- direct cast consumer
  当前只保留 build/source contract gate，
  不作为 TT-Sim direct-runtime correctness gate
- TT-Sim `fp16`
  仍按 simulator capability boundary 处理，
  不作为当前 correctness gate

## 6. 当前 blocker 顺序

cleanup 的架构依赖顺序
仍按 cleanup 总览里的
`Task 0 -> Task 1 -> Task 2 -> Task 3 -> Task 4 -> Task 5`
理解。

本节只记录
repo HEAD 当前还未收口的 blocker 顺序，
因此固定为：

1. `Cleanup Task 1`
2. `Cleanup Task 2`
3. `Cleanup Task 0` 剩余 full-contract 回收
4. `Cleanup Task 3`
5. `Cleanup Task 4`
6. `Cleanup Task 5`
7. cleanup 完成后，再恢复 support surface / workload payoff 扩展

补充：

- `Cleanup Task 0`
  没有从路线里消失；
  只是 repo HEAD
  当前更先被
  `Task 1 / 2`
  卡住
- 一旦 `Task 2`
  把 analysis /
  lowering bag
  依赖切掉，
  就必须马上回收
  `Task 0`
  剩余 contract，
  不能继续把它悬空
- `Cleanup Task 5`
  不是实现 owner，
  只做最终 convergence gate

## 7. 当前下一步

当前下一步固定为：

1. 完成 `Cleanup Task 1`
   - 用 direct logical bridge capture
     取代 compute-region bag
2. 然后完成 `Cleanup Task 2`
   - 删除 public / internal legacy analysis bag
   - 同时把 `SpatialPlan`
     收成单一 direct builder implementation
3. 紧接着回收 `Cleanup Task 0` 剩余 full-contract work
   - 让 selector
     真正变成独立 rewrite
   - 删除
     `blackhole.cb_requirements`
     作为 builtin-selection owner truth
   - 让 selector /
     validator
     共用 exact builtin legality contract
4. 再推进 `Cleanup Task 3 / 4`
   - 删除 `blackhole.copy_semantics`
   - 删除 `blackhole.segment_kind`
   - `Cleanup Task 4`
     先切 `TTKernel / executable segment_plan`
     的 kind owner truth，
     再单独处理 leaf-local body slicing residue
5. 最后做 `Cleanup Task 5`
   - 统一文档 /
     residue scans /
     verification /
     delivery 收尾
