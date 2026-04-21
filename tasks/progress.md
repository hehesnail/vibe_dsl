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
  `Task 1: SpatialPlan Representation Cutover`
  还没收口；
  当前最直接的 repo HEAD
  blocker
  是和它重叠的
  cleanup task1/task2 residue
- **当前推进原则**:
  主线固定按
  `Task 1 -> Task 2 -> Task 3 -> Legacy Protocol Deletion`
  推进；
  cleanup 只记录
  和主线重叠的 residue workstream。
  先收口主设计任务，
  再恢复 support surface / workload payoff 扩展

## 2. 当前主线任务状态

### 2.1 主线任务看板

- `Task 1: SpatialPlan Representation Cutover`
  - 状态：`in progress / 当前主线`
  - 当前直接 blocker：
    cleanup task1/task2
    这组 overlap residue
  - 当前目标：
    让 `SpatialPlan`
    真正成为
    target-independent
    virtual spatial/dataflow
    owner truth
- `Task 2: TTProgram Representation Cutover`
  - 状态：`pending / 受 Task 1 阻塞`
  - overlap cleanup：
    task2/task3，
    以及仍未删完的
    exact builtin /
    legality residue
  - 当前目标：
    让 `TTProgram`
    真正成为唯一
    TT-specific realization
- `Task 3: ExecutableSpec / Leaf Reader Cutover`
  - 状态：`pending / 受 Task 2 阻塞`
  - overlap cleanup：
    task3/task4/task5
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
    是这条线上的
    overlap residue workstream，
    不是另一条主路线

### 2.2 cleanup overlap 看板

- `Cleanup Task 1`
  - 状态：`active overlap / 当前卡在 Task 1`
  - 目标：把 logical bridge handoff
    从 `blackhole.compute_regions`
    切到 direct capture
- `Cleanup Task 2`
  - 状态：`active overlap / 当前卡在 Task 1-Task 2 交界`
  - 目标：删除 public/internal
    legacy analysis carrier
  - 说明：
    它是 `Task 1`
    收口的组成部分，
    同时也给 `Task 2`
    清理 planning 输入边界
- `Cleanup Task 0`
  - 状态：`open residue / 不是主线 owner doc`
  - 当前只完成：
    `selector-forwarding`
    局部前移
  - 仍未完成：
    exact builtin selection /
    legality /
    `blackhole.cb_requirements`
    删除
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

- `task1/task2/task3`
  这组文档
  定义主线任务和完成判据
- cleanup `task0-task5`
  只定义 overlap residue
  的删除范围和 convergence gate

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

## 4. 当前未收口项（按主线任务归类）

- `Task 1 / SpatialPlan`
  - `blackhole.compute_regions`
    仍是 bridge handoff
    的 producer-side owner truth
  - `lower.py`
    / target test helper
    仍通过
    `AnalyzeBlackholeComputeRegions`
    走 legacy hand-off
  - public
    `AnalyzeBlackhole*`
    wrappers
    仍暴露
  - internal
    `*Evidence(...)`
    helper
    和
    `blackhole.lowering_requirements`
    broad bag
    仍在 active chain
- `Task 2 / TTProgram`
  - `SelectBlackholeTTMetalBuiltins`
    仍是
    `PlanTTKernelABI::SelectComputeBuiltins()`
    的薄前门
  - `blackhole.cb_requirements`
    仍在 active chain
  - selector / validator
    还没共用一份
    exact builtin legality contract
  - `buffer_tile_bridge_specs`
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

## 6. 当前下一步

当前下一步固定为：

1. 继续收口 `Task 1: SpatialPlan Representation Cutover`
   - 先推进 cleanup task1
     对应的 direct bridge capture
   - 再推进 cleanup task2
     对应的
     public/internal legacy analysis bag
     删除
2. `Task 1`
   收口后，
   转入 `Task 2: TTProgram Representation Cutover`
   - 同时处理与它重叠的
     cleanup task0
     exact builtin / legality residue
   - 以及 cleanup task3
     对应的 payload /
     compatibility residue
3. 然后推进
   `Task 3: ExecutableSpec / Leaf Reader Cutover`
   - 删除
     `blackhole.copy_semantics`
   - 删除
     `blackhole.segment_kind`
   - 收紧 leaf reader /
     runtime /
     codegen 边界
4. 最后做
   `Legacy Protocol Deletion`
   的 convergence /
   scans /
   verification /
   delivery 收尾
