# TileLang Blackhole Backend Progress

> 当前唯一总体设计文档: `tasks/dev_design/final_blackhole_backend_redesign.md`

## 当前阶段

- **日期**: `2026-04-14`
- **总阶段**: Stage 4
- **目标主线**:
  `Normalized Tile TIR -> SpatialPlan companion -> TTProgram companion -> ExecutableSpec`

## 当前代码现实

当前代码还没有完全站到最终主链上。

当前实际链路是：

```text
Normalized Tile TIR
  -> SpatialPlan companion
  -> residual Blackhole analysis facts
  -> BuildTTProgram helper bridge
  -> TTProgram companion
  -> ExecutableSpec
```

这里的两类残留：

- `blackhole.work_decomposition / blackhole.compute_regions / blackhole.pipeline_stages`
- `BuildTTProgram` 内部的
  `PlanTTKernelABI / PlanTTCBAlloc / PlanTTCoreGroups`

都只视为**迁移残留**，不属于长期架构。

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

## 当前未完成

1. 用真实的
   `PlanTTTransport + PlanTTCompute`
   替掉 `BuildTTProgram` 内部 helper bridge
2. 删除剩余 late matcher / side contract
3. 在新 route 上完成 `flash-attn` admitted subset payoff
4. 在新 route 上承接
   `topk / fusedmoe / paged decode / chunk recurrence`
5. 扩更宽的 copy / data movement / sync 支持面

## 当前优先级

1. **P0: 真实 `PlanTTTransport + PlanTTCompute` cut-in**
   - 用 anchored sub-TIR 上仍保留的
     tile-op / layout / `BufferLoad / BufferStore`
     完成 target builtin mapping
   - 删除 `row_* / broadcast_sources / index map / access pattern /
     buffer_distribution_contract`
     这类旧 side contract
2. **P1: `flash-attn` payoff**
   - 在新 route 上兑现 multi-phase transport / reduction / broadcast
3. **P2: wider family cutover**
   - `topk -> fusedmoe -> paged decode -> chunk recurrence`
4. **P3: wider support surface**
   - copy / dataflow / sync

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
  当前还不是 admitted support surface；
  缺显式 per-work contract 或 transport/compute protocol truth
  的 kernel 继续 unsupported
- TT-Sim `fp16`
  仍按 simulator capability boundary 处理，
  不作为当前 correctness gate

## 最新验证摘要

- `tilelang` 构建通过
- `test_blackhole_spatial_ir.py`
  当前基线通过
- copy pipeline 定向回归当前基线通过
- selected `flash-attn` pipeline regression
  当前文档与 active path 对齐通过

## 下一步

1. 落地 `PlanTTTransport + PlanTTCompute`
2. 删除剩余 helper bridge / old side contract
3. 推进 `flash-attn` payoff
4. 再进 wider family / support surface
