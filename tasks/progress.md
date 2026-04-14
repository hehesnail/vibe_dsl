# TileLang Blackhole Backend Progress

> 当前唯一总体设计文档: `tasks/dev_design/final_blackhole_backend_redesign.md`

## 当前阶段

- **日期**: `2026-04-15`
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
并且当前 roadmap `R0`
已经把 owner cut-in 站回 active path。

当前实际链路是：

```text
Normalized Tile TIR
  -> SpatialPlan companion
  -> residual Blackhole analysis facts / lowering requirements helper
  -> PlanTTBlocks
  -> PlanTTCompute
  -> PlanTTTransport
  -> BuildTTProgram
  -> TTProgram companion
  -> ExecutableSpec
```

这里当前剩下的两类残留：

- `blackhole.work_decomposition / blackhole.compute_regions / blackhole.pipeline_stages`
  这批过渡分析 facts
- `PlanTTSync / PlanTTABI / PlanTTExecution`
  还没有独立站成和 `R0` 同等级的显式 owner pass

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
  -> PlanTTCompute
  -> PlanTTTransport
  -> PlanTTSync
  -> PlanTTABI
  -> PlanTTExecution
  -> TTProgram companion
  -> ExecutableSpec
```

当前第一缺口不再是 target builtin mapping 边界，
而是 **communication owner/runtime semantics**
还没有像 `R0 / R1` 一样显式站成主链。

按总设计的第一性原理，
当前目标不是单一 workload 绿测，
而是下面 4 条同时成立：

1. mapping 边界正确：
   target builtin mapping
   回到 anchored sub-TIR
2. TT-Metal 三类语义面都有 owner：
   compute / memory-access / communication
3. 真源位置收紧：
   `Normalized Tile TIR / SpatialPlan / TTProgram / ExecutableSpec`
   各守边界
4. 后段不再补语义：
   runtime / codegen
   不再做 late recovery

当前 roadmap 与这 4 条的对应关系是：

- `R0`：mapping 边界 + compute/memory-access owner cut-in
- `R1`：`TTProgram / ExecutableSpec`
  reader-gate / host-truth 收口（已完成）
- `R2`：communication owner/runtime semantics 收口
- `R3-R5`：payoff / wider family / support surface

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
- `R1` 已完成：
  - runtime/codegen 现在只接受
    kernel-local `per_work_arg_specs`
    作为 per-work/access truth；
    不再从 top-level `TTProgram.payload`
    或 `ABI payload`
    回填
  - build/codegen 已显式禁止
    从 `work_linear_id`
    恢复 block/tile 语义；
    多 work kernel 缺显式 per-work binding
    直接 fail-fast
  - `PlanTTKernelABI`
    写 top-level `per_work_arg_specs`
    的旧 payload bag 已删除；
    相关过期 regression
    已改成 kernel-local mutation

## 当前未完成

1. 完成 `R2`：
   在 admitted scope 内把
   communication semantics
   的 `routing / multicast / semaphore / completion`
   收口到 owner/runtime semantics，
   不再把第三类语义压缩成 sync-only
2. 在第一性原理目标完成之后，
   再在当前 `TTProgram / ExecutableSpec` 真源下完成
   `flash-attn` admitted subset payoff / correctness 收口
3. 在新 route 上承接
   `topk / fusedmoe / paged decode / chunk recurrence`
4. 扩更宽的 copy / data movement / wider communication 支持面

## 当前优先级

1. **R2: admitted-scope communication semantics 收口**
   - 第一性原理目标里的
     communication owner/runtime semantics
     不能留到更宽 support surface 再处理
2. **R3: `flash-attn` payoff**
   - 在当前新 route 上兑现 multi-phase transport / reduction / broadcast
   - 继续把 compile-path 稳定性兑现成 correctness/admitted subset
3. **R4: wider family cutover**
   - `topk -> fusedmoe -> paged decode -> chunk recurrence`
4. **R5: wider support surface**
   - copy / dataflow / wider communication

最近完成的局部批次：

- `Task 3B cleanup`
  （`T3B.0-T3B.4`）已完成；
  对应的是旧链清理，
  不等于第一性原理目标完成

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
  还不是通用 communication 执行模型；
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
- `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q`
  `131 passed, 25 skipped, 1 xfailed`
- `source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh && export TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo && cd /root/dev/vibe_dsl/tilelang_repo && pytest testing/python/target/blackhole/test_blackhole_copy_runtime.py -k 'grid_indexed_copy_multicore_launch or accepts_oversubscribed_multi_core_launch or richer_copy_schema_with_explicit_per_work_spec' -q`
  `3 passed`

## 下一步

1. 先完成 `R2`
2. 再推进 `R3`
3. 然后进入 `R4/R5`
