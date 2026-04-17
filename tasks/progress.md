# TileLang Blackhole Backend Progress

> 当前唯一总体设计文档:
> `tasks/dev_design/final_blackhole_backend_redesign.md`

> 当前 cleanup 执行总览:
> `tasks/dev_design/2026-04-16-blackhole-final-legacy-protocol-cleanup.md`

> 本文件只记录 repo HEAD 的当前事实、当前主线和下一步。
> 长期边界看设计文档；具体实现细节看代码和对应 task 文档。

## 1. 当前状态

- **日期**: `2026-04-17`
- **总体状态**: cleanup 主线 `Task 1-5` 仍未完成
- **当前 blocker**:
  不是单一 cutover 点，也不是 support surface 不够；
  而是 cleanup 主线还没同时收口
- **当前下一步**:
  从 `Cleanup Task 1`
  继续推进

## 2. 当前事实

- Blackhole 正式执行路径只剩
  `BlackholeModule`
  进程内 direct host path
- `tilelang.compile(..., execution_backend="tvm_ffi")`
  的 Blackhole wrapper/export path 已恢复
- 当前长期主链仍然只有：

```text
Normalized Tile TIR
  -> SpatialPlan
  -> TTProgram
  -> ExecutableSpec
```

- 当前 repo HEAD 的实际实现顺序，
  仍是 cleanup 文档描述的那条 active chain；
  它描述的是当前 pass/phase 实现，
  不是新的长期 IR 层
- `Cleanup Task 0` 已完成：
  exact TT-Metal builtin surface 已锁定，
  dedicated selector 已接入 active chain，
  helper/composite builtin residue
  与 `compute_epilogue_ops`
  已退出 active compute 主链

## 3. 当前未收口项

当前已知仍未收口的点包括：

- `AnalyzeSpatialStructureFacts`
  仍在 active chain，且仍通过 public wrapper 暴露
- `SpatialPlan -> TTProgram`
  之间仍有 legacy transition attrs /
  narrow bridge seeds / helper residue
- `blackhole.copy_semantics`
- `blackhole.segment_kind`
- `blackhole.lowering_requirements`
- `blackhole.resource_plan`
- `PlanTTSync / PlanTTABI / PlanTTExecution`
  虽已落地为显式 planner passes，
  但 leaf reader / cleanup 还没同时收口

## 4. 当前稳定基线

- `SplitBlackholeKernel`
  已接入主链：
  pure copy 走 `fused_dataflow` 单 kernel，
  GEMM 走 3-kernel（reader / compute / writer）
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

## 5. 当前执行顺序

当前实际执行顺序固定为：

1. `Cleanup Task 1`
2. `Cleanup Task 2`
3. `Cleanup Task 3`
4. `Cleanup Task 4`
5. `Cleanup Task 5`
6. cleanup 完成后，再恢复 support surface / workload payoff 扩展

## 6. 当前下一步

当前下一步固定为：

1. 完成 `Cleanup Task 1`
   - 用 direct logical bridge capture
     取代 compute-region bag
2. 然后完成 `Cleanup Task 2`
   - 删除 public / internal legacy analysis bags
   - 把 `SpatialPlan`
     收成单一 direct builder implementation
3. 再推进 `Cleanup Task 3 / 4`
   - 删除 `blackhole.copy_semantics`
   - 删除 `blackhole.segment_kind`

## 7. 文档职责

- `final_blackhole_backend_redesign.md`
  - 总设计和长期边界
- `task0/1/2/3`
  - 各层合同与完成判据
- `progress.md`
  - repo HEAD 的当前事实、当前主线和下一步
