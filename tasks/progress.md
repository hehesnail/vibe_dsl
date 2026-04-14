# TileLang Blackhole Backend Progress

> 当前唯一总体设计文档: `tasks/dev_design/final_blackhole_backend_redesign.md`

## 当前阶段

- **日期**: `2026-04-14`
- **总阶段**: Stage 4
- **目标主线**:
  `Normalized Tile TIR -> SpatialPlan companion -> TTProgram companion -> ExecutableSpec`

命名约定：

- `R0 / R1 / ...` 表示当前 roadmap 总优先级
- `Tn.x` 表示 task / batch 内部顺序
  （例如 `T2.0`、`T3B.0`）

## 当前代码现实

当前代码已经完成 `Task 3B cleanup`
（`T3B.0-T3B.4`）
这批旧 side-contract 清理，
但当前 roadmap `R0`
还没有完全站到最终 planner 主链上。

当前实际链路是：

```text
Normalized Tile TIR
  -> SpatialPlan companion
  -> residual Blackhole analysis facts / lowering requirements helper
  -> BuildTTProgram helper bridge
  -> TTProgram companion
  -> ExecutableSpec
```

这里的两类残留：

- `blackhole.work_decomposition / blackhole.compute_regions / blackhole.pipeline_stages`
- `BuildTTProgram` 内部的
  `PlanTTKernelABI / PlanTTCBAlloc / PlanTTCoreGroups`

都只视为**迁移残留**，不属于长期架构。

已经退出 active path 的旧对象：

- `SpatialProgram` 作为 execution-bearing IR
- `buffer_distribution_contract`
- runtime/codegen 侧把 top-level stale per-work payload
  当成 multi-segment ABI 真源的 fallback

## 当前目标

要收敛到的真实链路是：

```text
Normalized Tile TIR
  -> SpatialPlan companion
  -> PlanTTBlocks
  -> PlanTTTransport
  -> PlanTTCompute
  -> PlanTTSync
  -> PlanTTABI
  -> PlanTTExecution
  -> TTProgram companion
  -> ExecutableSpec
```

当前第一缺口不是某个单独 workload case，
而是 **target builtin mapping 还没有完全前移到
anchored sub-TIR 仍保留 tile-op / layout / load-store truth 的边界**。

## 已完成

- `Task 1` 已完成：
  `AnalyzeSpatialStructureFacts -> BuildSpatialPlanCompanion`
  已进入 active path
- `Task 2` 的外层 owner cutover 已完成：
  canonical bundle 已固定为
  `BuildTTProgram -> ValidateTTProgram -> MaterializeBlackholeExecutable`
- 独立 semantic companion / semantic witness / `tl.semantic_*`
  主协议已从 active path 删除
- projection bridge、fragment-layout side-channel、
  中间 seed attr 的公开 surface 已删除
- runtime / codegen 当前只读 `TTProgram / ExecutableSpec`
  的正式 reader 路线，不再保留公开 legacy entry
- `Task 3B cleanup`
  （`T3B.0-T3B.4`）
  旧 side-contract 清理批次已完成：
  - `SpatialProgram` pass / companion / 相关测试入口
    已从 active path 删除
  - `buffer_distribution_contract`
    已从 lowering / codegen / regression surface 删除，
    统一收敛到 `buffer_tile_bridge_specs`
  - multi-segment `TTProgram` 的
    segment-local `per_work_arg_specs`
    已成为 reader/writer ABI 真源，
    `flash-attn` 不再回退到 top-level stale copy descriptor
  - `flash-attn` compile-path 回归基线
    已在新 ABI truth 下重新稳定

## 当前未完成

1. 完成当前 roadmap `R0`：
   把 target builtin mapping 真正前移到
   anchored sub-TIR 仍保留
   `tile-op / layout / load-store` truth 的边界，
   由真实 `PlanTTTransport + PlanTTCompute`
   取代 `BuildTTProgram` helper bridge
2. 在当前 `TTProgram / ExecutableSpec` 真源下完成
   `flash-attn` admitted subset payoff / correctness 收口
3. 在新 route 上承接
   `topk / fusedmoe / paged decode / chunk recurrence`
4. 扩更宽的 copy / data movement / sync 支持面

## 当前优先级

1. **R0: 真实 `PlanTTTransport + PlanTTCompute` cut-in**
   - target builtin mapping 还没有完全前移到
     anchored sub-TIR 边界
   - active path 仍残留
     `blackhole.*` analysis facts 和
     `BuildTTProgram` helper bridge
2. **R1: runtime gate 收口**
   - 继续把当前新主链上的 gate 收到
     明确的 admitted / unsupported 边界
3. **R2: `flash-attn` payoff**
   - 在当前新 route 上兑现 multi-phase transport / reduction / broadcast
   - 继续把 compile-path 稳定性兑现成 correctness/admitted subset
4. **R3: wider family cutover**
   - `topk -> fusedmoe -> paged decode -> chunk recurrence`
5. **R4: wider support surface**
   - copy / dataflow / sync

最近完成的局部批次：

- `Task 3B cleanup`
  （`T3B.0-T3B.4`）已完成；
  对应的是旧链清理，不等于当前 roadmap `R0` 完成

## 当前稳定基线

- `ExecutableSpec -> rt_mod_blackhole -> BlackholeModule`
  direct host path 仍是唯一正式执行路径
- `tilelang.compile(..., execution_backend="tvm_ffi")`
  的 Blackhole wrapper/export path 已恢复
- canonical engine bundle:
  `LowerToBlackholePhaseB -> LowerToBlackholeTTProgram -> LowerToBlackholeExecutable`
- copy / GEMM / export 当前支持面不能回退
- Blackhole runtime / direct-runtime 正式 correctness baseline
  统一使用 `bf16`

## 当前 admitted 支持面

- copy：
  equal source/dest range，stride = 1
- GEMM：
  A/B-separated reader range + writer output range
- accessor：
  interleaved + DRAM + `common_runtime_arg_count = 0`

## 当前边界

- oversubscribed direct runtime
  还不是通用同步执行模型；
  带显式 `TTSemaphorePlan` / remote descriptors 的 executable
  仍应 fail-fast
- `flash-attn` direct runtime
  compile-path / source/spec baseline 已稳定，
  但 runtime correctness 还不是 admitted support surface
- TT-Sim `fp16`
  仍按 simulator capability boundary 处理，
  不作为当前 correctness gate

## 最新验证摘要

- `tilelang` 构建通过
- `test_blackhole_copy_pipeline.py`
  `41 passed, 10 skipped, 1 xfailed`
- `test_blackhole_gemm.py`
  `27 passed, 15 skipped`
- `test_blackhole_spatial_ir.py`
  `5 passed`
- `test_blackhole_flash_attention_analysis.py`
  `7 passed`
- `test_blackhole_flash_attention_pipeline.py`
  `62 passed`

## 下一步

1. 先完成 `R0`
2. 再推进 `R1/R2`
3. 然后进入 `R3/R4`
