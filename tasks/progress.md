# TileLang Blackhole Backend Progress

> 当前唯一总体设计文档: `tasks/dev_design/final_blackhole_backend_redesign.md`

## 当前阶段

- **日期**: `2026-04-15`
- **总阶段**: Stage 4
- **目标主线**:
  `Normalized Tile TIR -> SpatialPlan companion -> TTProgram companion -> ExecutableSpec`

命名约定：

- **当前活动任务顺序**统一写成 `Rn.m`
  - `Rn`
    表示 roadmap 阶段
  - `.m`
    表示该阶段内的当前执行顺序
  - 例如：
    `R0.1 -> R0.2 -> R0.3 -> R1.1`
- `Tn.x`
  只保留给历史 batch / 已完成清理项 /
  代码 grep 兼容，
  不再作为当前主阅读顺序

## 当前代码现实

当前代码已经完成 `Task 3B cleanup`
（`T3B.0-T3B.4`）
这批旧 side-contract 清理，
并且当前 roadmap `R0`
已经把 owner cut-in
站回 active path，
但 `R0-R2`
都还没有到“可按总设计宣告完成”的程度。

当前实际链路是：

```text
Normalized Tile TIR
  -> SpatialPlan companion
  -> SplitBlackholeKernel
  -> blackhole.work_decomposition / compute_regions / pipeline_stages
  -> PlanTTBlocks
  -> PlanTTCompute(PlanTTKernelABI wrapper)
  -> PlanTTTransport(PlanTTCBAlloc wrapper)
  -> BuildTTProgram
  -> TTProgram companion
  -> ValidateTTProgram
  -> MaterializeBlackholeExecutable(writer attr: tl.blackhole_executable)
  -> executable extraction / codegen / BlackholeModule
```

这里当前剩下的是几类迁移残留：

- `blackhole.work_decomposition / blackhole.compute_regions / blackhole.pipeline_stages`
  这批过渡分析 facts
- `blackhole.lowering_requirements`
  仍被 build/codegen/executable extraction
  当作 active gate 输入
- `PlanTTKernelABI / PlanTTCBAlloc / PlanTTCoreGroups`
  仍在 `PlanTTCompute / PlanTTTransport / PlanTTBlocks`
  名字下面承担 owner residue
- `PlanTTSync / PlanTTABI / PlanTTExecution`
  还没有独立站成显式 owner pass；
  `BuildTTProgram`
  仍直接合成 sync / execution 结果
- `MaterializeBlackholeExecutable`
  已成为显式 writer boundary：
  负责把已验证的 `TTProgram`
  物化成
  `tl.blackhole_executable`
  companion attr，
  build 侧显式要求该 attr

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

当前第一缺口不是单一 workload payoff，
而是 `R0-R2`
本身还没有同时收口。

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
  reader-gate / host-truth 收口
- `R2`：communication owner/runtime semantics 收口
- `R3-R5`：payoff / wider family / support surface

当前审计后的状态口径：

- `R0`：**active-path cut-in 已完成，owner closure 未完成**
- `R1`：**per-work/access gate 大体完成，reader/writer 边界未完成**
- `R2`：**communication gate / admitted subset 大体完成，owner-pass 化未完成**

## 已完成

- `Task 1` 已完成：
  `AnalyzeSpatialStructureFacts -> BuildSpatialPlanCompanion`
  已进入 active path
- `Task 2` 的外层 owner cutover 已进入 active path：
  canonical bundle 已固定为
  `BuildTTProgram -> ValidateTTProgram -> MaterializeBlackholeExecutable`
- 独立 semantic companion / semantic witness / `tl.semantic_*`
  主协议已从 active path 删除
- projection bridge、fragment-layout side-channel、
  中间 seed attr 的公开 surface 已删除
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
- `R0` 已完成的部分：
  - active path 已显式经过
    `PlanTTBlocks -> PlanTTCompute -> PlanTTTransport -> BuildTTProgram`
  - `SpatialProgram`
    已退出 active compile/runtime path
  - `buffer_distribution_contract`
    已退出 active lowering/codegen/regression surface
- `R1` 已完成的部分：
  - `T2.4`
    已完成：
    `MaterializeBlackholeExecutable`
    不再是 no-op shell；
    现在显式写出
    `tl.blackhole_executable`
    writer attr，
    `rt_mod_blackhole` /
    executable extraction
    直接读取该 writer projection，
    build 缺失该 attr
    直接 fail-fast
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
- `R2` 已完成的部分：
  - runtime/build 现在把
    `get_semaphore` / remote semaphore builtin
    收回 explicit communication schema，
    不再接受“先写 builtin，
    再让后段补协议”
  - `get_semaphore(<id>)`
    必须引用已计划的 `TTSemaphorePlan`
    或显式绑定的 `semaphore_id_u32`
    runtime arg；
    缺 planned semaphore truth
    直接 fail-fast
  - remote semaphore routing
    必须来自显式
    `logical_core_noc_x / logical_core_noc_y`
    runtime arg 与
    `remote_core_descriptors`；
    不再接受 literal/body-recovered NOC 坐标
  - non-oversubscribed direct runtime
    已承认显式 semaphore / remote-endpoint
    communication subset；
    oversubscribed 显式 communication contract
    继续 fail-fast
  - 旧的 source-only semaphore regression
    已改成显式携带 owner truth；
    新增 communication gate / oversubscribed boundary regression

## 当前未完成

1. **`R0-close`**
   - 让 active owner planning
     不再依赖
     `blackhole.work_decomposition /
      blackhole.compute_regions /
      blackhole.pipeline_stages`
     作为正式协议输入
   - 先把当前混在
     `blackhole.lowering_requirements`
     里的
     effect/use-role、
     liveness、
     materialization decision
     拆开：
     - 独立的 buffer effect / use-role analysis
     - 独立的 buffer liveness analysis
     - 独立消费前两者 facts 的 contract/planner 决策
   - 收紧 `PlanTTKernelABI / PlanTTCBAlloc / PlanTTCoreGroups`
     这组 helper residue，
     避免继续被文档误写成“已完成 owner pass”
2. **`R1-close`**
   - 让 build/codegen/executable extraction
     停止消费
     `blackhole.lowering_requirements`
     这类过渡 attr，
     把剩余 gate 收回
     `TTProgram / ExecutableSpec`
     的 typed truth
3. **`R2-close`**
   - 把 `PlanTTSync / PlanTTABI / PlanTTExecution`
     站成显式 owner pass，
     或至少把当前 helper owner contract
     明确冻结下来，
     不再继续藏在 `BuildTTProgram`
     的临时 synthesis 里
   - 让 communication owner
     不只停在 consumer-side gate，
     还要在 planner/builder 侧有独立边界
4. 在 `R0-R2` 真正收口之后，
   再在当前 `TTProgram / ExecutableSpec` 真源下完成
   `flash-attn` admitted subset payoff / correctness 收口
5. 在新 route 上承接
   `topk / fusedmoe / paged decode / chunk recurrence`
6. 扩更宽的 copy / data movement / wider communication 支持面

## 当前优先级

1. **`R0.1`: 拆出独立的 buffer effect / use-role analysis**
   - 只从 anchored sub-TIR
     产出
     `defs / uses / write-effect / use-role / recurrence edge`
     facts
   - 不直接写 merge/live-form contract
   - 旧文档别名：
     `T3C.0a`
2. **`R0.2`: 拆出独立的 buffer liveness analysis**
   - 只消费
     `defs / uses + recurrence edge`
   - 用标准 backward dataflow
     计算 `live_in / live_out`
   - 旧文档别名：
     `T3C.0b`
3. **`R0.3`: 把 materialization / source-live-form decision 移到独立 planner 阶段**
   - 由 planner
     消费
     `effect/use-role + liveness`
     facts
   - 退役
     `blackhole.lowering_requirements`
     里当前混合式判定
   - 旧文档别名：
     `T3C.0c`
4. **`R1.1`: 去掉 build/codegen 对 `blackhole.lowering_requirements` 的依赖**
   - unsupported-compute / bridge-spec /
     materialization gate
     回收到 `TTProgram / ExecutableSpec`
   - 旧文档别名：
     `T2.5`
5. **`R2.1`: 显式化 sync / ABI / execution owner**
   - `PlanTTSync -> PlanTTABI -> PlanTTExecution`
     要么落地成 pass，
     要么在文档和 gate 里明确仍属过渡实现
   - 旧文档别名：
     `T3C.1`
6. **`R3.1`: `flash-attn` payoff**
7. **`R4.1`: wider family cutover**
8. **`R5.1`: wider support surface**

最近完成的局部批次：

- `T2.4`
  已完成：
  executable writer boundary
  收口到
  `MaterializeBlackholeExecutable -> tl.blackhole_executable`
- `Task 3B cleanup`
  （`T3B.0-T3B.4`）已完成；
  对应的是旧链清理，
  不等于第一性原理目标完成
- `2026-04-15` 的 `T2.4` follow-up
  已收口到主链语义：
  - GEMM reader runtime-arg 绑定
    不再让
    `a_tile_stride / b_tile_stride`
    覆盖
    `A_addr / B_addr`
  - fresh fragment / preclear zero-init
    不再因为
    `clear_accum=False`
    自动落到
    `intermediate_accumulator_merge`
    旧链；
    当前统一按
    `TIR execution order + recurrence/live consumer`
    判定是否真的需要 merge
  - preclear-only GEMM
    已 canonicalize 到
    `clear_accum=true`
    direct path
  - direct cast consumer
    的 build/source contract
    仍保留，
    但其 runtime 执行
    还不纳入当前 TT-Sim correctness gate
  - 上面这批
    `TIR execution order + recurrence/live consumer`
    判定
    当前仍属于过渡实现；
    下一批
    `R0.1-R0.3`
    会把它拆成
    独立的 effect/use-role analysis、
    独立 liveness pass
    和独立 planner decision，
    不把这种混合逻辑固化成长期主链

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
  A/B-separated reader range + writer output range；
  fresh fragment / preclear zero-init
  已统一走
  `clear_accum=true`
  direct path
- accessor：
  interleaved + DRAM + `common_runtime_arg_count = 0`
- communication：
  non-oversubscribed direct runtime
  已承认显式
  `TTSemaphorePlan` /
  `semaphore_id_u32` binding /
  `logical_core_noc_x/y + remote_core_descriptors`
  的 local/remote semaphore subset

## 当前边界

- oversubscribed direct runtime
  还不是通用 communication 执行模型；
  带显式 `TTSemaphorePlan` / remote descriptors 的 executable
  仍应 fail-fast
- communication builtin
  不能单独充当协议真源；
  缺 planned semaphore truth
  或缺显式 remote endpoint schema
  一律 build-time fail-fast
- remote / collective / fabric communication
  目前只承认 explicit semaphore + remote endpoint subset；
  multicast / 更宽 topology / collective
  仍未 admitted
- `flash-attn` direct runtime
  compile-path / source/spec baseline 已稳定，
  但 runtime correctness 还不是 admitted support surface
- direct cast consumer
  仍依赖旧的 merge/live-form bridge；
  当前只保留 build/source contract gate，
  不作为 TT-Sim direct-runtime correctness gate
- TT-Sim `fp16`
  仍按 simulator capability boundary 处理，
  不作为当前 correctness gate

## 最新验证摘要

- `tilelang` 构建通过
- `pytest /root/dev/vibe_dsl/tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_compute_contract_attr_is_materialized /root/dev/vibe_dsl/tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_fresh_fragment_gemm_does_not_materialize_accumulator_merge_contract /root/dev/vibe_dsl/tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_precleared_fragment_gemm_does_not_materialize_accumulator_merge_contract /root/dev/vibe_dsl/tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_gemm_compile_time_abi_is_materialized /root/dev/vibe_dsl/tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_gemm_reader_binds_tensor_accessor_to_buffer_addrs /root/dev/vibe_dsl/tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_gemm_post_merge_cast_consumer_exposes_republish_contract -q`
  `6 passed`
- `bash -lc 'source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh && export TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo && cd /root/dev/vibe_dsl/tilelang_repo && pytest -q testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_gemm_basic testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_gemm_direct_runtime_materializes_compile_time_abi_schema testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_precleared_fragment_gemm_canonicalizes_to_clear_accum_true testing/python/target/blackhole/test_blackhole_gemm.py::test_blackhole_gemm_direct_runtime_preserves_clear_accum_false_fragment_for_cast_consumer testing/python/target/blackhole/test_blackhole_copy_pipeline.py::test_blackhole_copy_direct_runtime_materializes_compile_time_abi_schema -q'`
  `4 passed, 1 skipped`
- `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q`
  `133 passed, 25 skipped, 1 xfailed`
- `bash -lc 'source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh && export TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo && cd /root/dev/vibe_dsl/tilelang_repo && pytest testing/python/target/blackhole/test_blackhole_copy_runtime.py::test_blackhole_module_direct_call_grid_indexed_copy_worker_semaphore_handshake testing/python/target/blackhole/test_blackhole_copy_runtime.py::test_blackhole_module_direct_call_accepts_oversubscribed_multi_core_launch testing/python/target/blackhole/test_blackhole_copy_runtime.py::test_blackhole_module_direct_call_rejects_oversubscribed_communication_contract testing/python/target/blackhole/test_blackhole_copy_pipeline.py::test_blackhole_copy_direct_runtime_accepts_semaphore_id_runtime_arg -q'`
  `4 passed`

## 下一步

1. 先推进 `R3`
2. 再进入 `R4`
3. 然后继续放宽 `R5`
