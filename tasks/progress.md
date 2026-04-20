# TileLang Blackhole Backend Progress

> 当前唯一总体设计文档:
> `tasks/dev_design/final_blackhole_backend_redesign.md`

> 当前 cleanup 执行总览:
> `tasks/dev_design/2026-04-16-blackhole-final-legacy-protocol-cleanup.md`

> 本文件只记录 repo HEAD 当前状态与下一步。
> task 文档里的合同、旧文档里的 landed/completed 表述、以及 repo HEAD 上已有代码，
> 都不能直接当成当前设计口径下的完成声明。

## 1. 当前总体状态

- **日期**: `2026-04-17`
- **当前总体 blocker**:
  cleanup 主线 `Cleanup Task 1-5`
  还没做完；
  当前问题不是单一 cutover 点，
  也不是 support surface 不够
- **当前推进原则**:
  先完成 cleanup 主线，
  再恢复 support surface / workload payoff 扩展

## 2. 顶层任务状态

- `Task 1: SpatialPlan Representation Cutover`
  - 未完成
- `Task 2: TTProgram Representation Cutover`
  - 未完成
- `Task 3: ExecutableSpec / Leaf Reader Cutover`
  - 未完成
- `Legacy Protocol Deletion`
  - 未完成

说明：

- `task0/1/2/3`
  当前定义的是目标合同和完成判据，
  不是 repo HEAD 的完成状态
- repo HEAD 上已有的局部实现、
  旧文档中的 landed/completed 描述、
  以及某些已接入主链的 pass，
  都只算当前代码现状，
  不能直接折算成顶层任务完成
- `Cleanup Task 0`
  也按同样口径处理：
  当前只有 selector-forwarding 局部结果，
  不按完成计

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

## 4. 当前未收口项

- `AnalyzeSpatialStructureFacts`
  仍在 active chain，
  且仍通过 public wrapper 暴露
- `SpatialPlan -> TTProgram`
  之间仍有 legacy transition attrs /
  narrow bridge seeds / helper residue
- `blackhole.copy_semantics`
- `blackhole.segment_kind`
- `blackhole.lowering_requirements`
- `blackhole.resource_plan`
- `PlanTTSync / PlanTTABI / PlanTTExecution`
  虽已落地为显式 planner passes，
  但 leaf reader / cleanup
  还没同时收口
- `buffer_tile_bridge_specs`
  仍同时承担
  planning residue
  和 leaf compatibility payload；
  当前固定分工是：
  - `Cleanup Task 1`
    先切 owner truth
    到 direct capture / narrow attr
  - `Cleanup Task 3`
    再删 payload / projection / codegen reader

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

## 6. 当前执行顺序

当前实际执行顺序固定为：

1. `Cleanup Task 1`
2. `Cleanup Task 2`
3. `Cleanup Task 3`
4. `Cleanup Task 4`
5. `Cleanup Task 5`
6. cleanup 完成后，再恢复 support surface / workload payoff 扩展

## 7. 当前下一步

当前下一步固定为：

1. 完成 `Cleanup Task 1`
   - 用 direct logical bridge capture
     取代 compute-region bag
2. 然后完成 `Cleanup Task 2`
   - 删除 public / internal legacy analysis bag
   - 同时把 `SpatialPlan`
     收成单一 direct builder implementation
3. 再推进 `Cleanup Task 3 / 4`
   - 删除 `blackhole.copy_semantics`
   - 删除 `blackhole.segment_kind`
   - `Cleanup Task 4`
     先切 `TTKernel / executable segment_plan`
     的 kind owner truth，
     再单独处理 leaf-local body slicing residue
