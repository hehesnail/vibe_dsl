# TileLang Blackhole Backend Progress

> 当前唯一总体设计文档: `tasks/dev_design/final_blackhole_backend_redesign.md`
> `tasks/dev_design/archive/` 下的文档全部视为历史记录，不再作为当前任务安排入口。
> `Phase C` 的详细完成判定、runtime gate 与支持面边界统一维护在
> `tasks/dev_design/stage4_phase_c_tt_target_ir.md`；这里只保留当前状态摘要。

## 当前阶段

- **日期**: `2026-04-09`
- **总阶段**: Stage 4
- **当前主线**: `Stateful Semantic IR -> Spatial Program IR -> TT Target IR`
- **阶段状态**:
  - `Stage 0 / Phase A / Phase B` 已完成
  - `Phase C` 进行中；`TTProgram` cutover 已完成，剩余工作集中在 `Phase C2`
    与更宽支持面兑现

## 当前状态摘要

- `ExecutableSpec -> rt_mod_blackhole -> BlackholeModule` direct host path 仍是唯一正式执行路径
- `tilelang.compile(..., execution_backend="tvm_ffi")` 的 Blackhole wrapper/export path 已恢复
- `TTProgram` translator / validator / materializer 已进入正式主链；
  runtime/codegen 已切到 `TTProgram` direct reader
- spatial/dataflow program model 的 cross-layer feature 设计
  已独立收口到 `tasks/dev_design/spatial_dataflow_program_model.md`，
  后续 literal semantics、planner owner 链与 expert hint API
  统一以该文档为设计入口
- TT-Sim fatal simulator taxonomy / hard-gate 扫描
  已独立记录到 `memory/tt_simulator_constraints.md`，
  后续 triage 先按 simulator capability boundary 与 target contract 回归分流
- 当前支持的 `flash-attn` forward 子集已经拿到 direct runtime correctness milestone
- Blackhole runtime / direct-runtime 回归基线已统一到 `bf16` 输入；
  `fp16` 不再作为当前 TT-Sim 上的正式 runtime 测试基线
- 无显式 `semaphore / remote-core` synchronization contract 的
  oversubscribed `work_packets` executable 已可按 packet truth 做 host-side
  wave scheduling
- 当前 admitted support surface 仍然偏窄：
  - copy：equal source/dest range 且 stride = 1
  - GEMM：A/B-separated reader range + writer output range
  - accessor：仅 interleaved + DRAM + `common_runtime_arg_count = 0`

## 当前未完成项

- 完成 `flash-attn` `Phase C2`：
  把当前 correctness milestone 扩成更宽 `MHA / GQA` runtime / correctness 支持面，
  并继续把剩余 multi-GEMM compute contract 收成 typed target truth
- 在当前主链上继续承接
  `topk / fusedmoe / paged decode / chunk recurrence` 等 family
- 继续扩大 copy/dataflow 与 synchronization 支持面
- 继续完成 `Placement / SpatialCapabilityModel / payload-backed node schema`
  的 typed uplift 与真实 consumer 验证

## 当前边界

- oversubscribed direct runtime 目前不是通用同步执行模型；
  一旦 executable 带显式 `TTSemaphorePlan`、`semaphore_bindings`
  或 `remote_core_descriptors`，仍应 fail-fast
- TT-Sim `fp16` 路径当前仍可能命中 simulator fatal taxonomy；
  该路径不属于当前 Blackhole runtime 的正式 correctness baseline，
  统一按 simulator capability boundary 处理
- `TT_METAL_WATCHER=10` 调试 multicore direct path 时，
  watcher 线程仍可能自己抛错或卡在 dump；
  正式 baseline 应在标准 watcher-off 环境下判断

## 下一步

1. 扩大 `flash-attn` `Phase C2` 支持面，并在当前 `bf16` runtime baseline 上继续兑现更宽 correctness
2. 在当前 layered mainline 上继续承接
   `topk / fusedmoe / paged decode / chunk recurrence`
3. 继续扩大 copy/dataflow 与 synchronization 支持面
4. 继续完成剩余 object-boundary typed uplift

## 最新验证摘要

- `tilelang_repo/build` fresh rebuild 通过
- 所有 runtime 检查均在标准 TT-Sim 环境入口下完成
- Blackhole copy/runtime + flash-attn runtime `bf16` baseline regressions 通过：
  `71 passed, 1 skipped, 1 xfailed`
- `flash-attn` pipeline regressions 通过：`2 passed, 46 deselected`
- GEMM direct runtime regressions 通过：`2 passed, 38 deselected`
- `flash-attn` 当前支持 runtime regression 通过：`1 passed, 6 deselected`
- 手工 `512x512x512` `bf16` pure GEMM direct runtime 已数值对齐
