# TileLang Blackhole 后端开发进度

> 当前唯一总体设计文档: `tasks/dev_design/final_blackhole_backend_redesign.md`

## 当前阶段

- **阶段**: Stage 4 — TT-Metal contract formalization / flash-attn forward subset
- **状态**: Stage 3 formal direct host path 已完成；当前主线已正式转到 TT-Metal contract formalization、backend cleanup 和 flash-attn forward subset
- **当前稳定状态**:
  - `ExecutableSpec -> rt_mod_blackhole -> BlackholeModule` 主链已稳定
  - P0 已完成；P3 在 current copy/GEMM formal surface 上已完成收口
  - P4 已完成 interleaved stick/page copy 主路径；P5 已完成 program-local worker semaphore 与 remote-core descriptor formalization
  - backend cleanup A1/A2/A3、B1/B2/B3、C1/C2 已完成当前计划内收敛
- **当前活动主线**:
  - flash-attn forward subset：analysis、fragment lowering、dataflow bridging 与 codegen 已接通当前支持面
  - 当前不再有 compile-path 主 blocker；当前剩余工作转为 runtime/更宽支持面验证
- **日期**: 2026-03-31
- **相关设计**:
  - `tasks/dev_design/final_blackhole_backend_redesign.md`
  - `tasks/dev_design/stage4_flash_attention_forward_subset.md`
  - `tasks/dev_design/2026-03-31-flash-attention-forward-subset-implementation-plan.md`

### Flash-Attention 当前推进

- **analysis / legality**:
  - `AnalyzeBlackholeWorkDecomposition`
  - `AnalyzeBlackholeFragmentRegions`
  - `AnalyzeBlackholePipelineStages`
  已全部接入 `SplitBlackholeKernel` 之后的主链
- **当前已完成的 fragment lowering 子集**:
  - `tl.blackhole.reduce_row`
  - `tl.blackhole.mul_row_bcast`
  - `tl.blackhole.mul_grouped_row_bcast`
  - `tl.blackhole.div_row_bcast`
  - `tl.blackhole.div_grouped_row_bcast`
  - `tl.blackhole.scalar_fma`
  - `tl.blackhole.exp2_row_bcast_affine`
  - `tl.blackhole.exp2_grouped_row_bcast_affine`
  - `tl.blackhole.scalar_exp2_affine`
  - `tl.blackhole.fill_fragment`
  - `tl.blackhole.scalar_max`
  - `tl.blackhole.cast_fragment_slice`
  - `tl.blackhole.write_local_slice_to_cb`
- **当前 codegen 状态**:
  - 上述 fragment builtin 已接入 `codegen_blackhole`
  - `blackhole.acc` 局部符号映射与 `tl.infinity` 相关的 device-only codegen 噪声已收正
  - full `lower()` 已不再卡在旧的 fragment-subset gate、`Find undefined Variable acc_o`、`tl.infinity` unresolved call、或 `local/accumulator -> shared(CB)` staged copy 残留
- **当前状态收口**:
  - `CopyDirection::kLocalToCB` 已接入 `LowerBlackholeOps`
  - `local/accumulator -> shared(CB)` staged copy 已 lower 成 `tl.blackhole.write_local_slice_to_cb`
  - 当前支持的 MHA/GQA forward compile-path 已打通
  - copy 正式主链已去掉 `input0/output0` 默认 runtime-arg fallback；缺 schema 现在在 `rt_mod_blackhole` build-time 显式失败
- **下一步**:
  - 在当前环境继续补 runtime / direct-path 验证
  - 继续扩更宽 flash-attn forward 支持面与 P4/P5 主项

### 最新回归结果（当前环境）

| 测试 | 结果 |
|------|------|
| `test_blackhole_flash_attention_analysis.py` | 7 passed |
| `test_blackhole_flash_attention_pipeline.py` | 16 passed |
| `test_blackhole_copy_pipeline.py` | 30 passed, 6 skipped, 1 xfailed |
| `test_blackhole_copy_runtime.py` | 2 passed, 9 skipped |
| `test_blackhole_gemm.py` | 21 passed, 10 skipped |
| `test_blackhole_tvm_ffi_export.py` | 1 passed |

### 已验证 full-env 结果

| 测试 | 结果 |
|------|------|
| `test_blackhole_copy_pipeline.py` | 30 passed, 1 xfailed |
| `test_blackhole_copy_runtime.py` | 11 passed |
| `test_blackhole_gemm.py` | 31 passed |

---

## 分阶段总览

| 阶段 | 目标 | 状态 |
|------|------|------|
| Stage 0 | 协议与执行载体（ExecutableSpec, BlackholeModule） | ✅ |
| Stage 1 | single-core copy bring-up | ✅ |
| Stage 2A | pass 主链接入 | ✅ |
| Stage 2B | single-core copy 正式主链 | ✅ |
| Stage 2C | split-before 语义规划（AnnotateBlackholeCopySemantics） | ✅ |
| Stage 2D | single-core GEMM + true E2E | ✅ |
| Stage 2E | 设备资源 IR 语义扩展（StorageRank + Canonicalization） | ✅ |
| **Stage 3** | **multi-core runtime 调度** | **✅ direct host path 完成** |

---

## Stage 3 实施计划

关键调研结论：
- `blockIdx.*` 不被 `ZeroThreadAndLoopVars` 零化 → tile index 自动含 per-core offset
- `BindThreadIndex` 已把 `blockIdx.x/y` → `work_id % grid_x` / `work_id / grid_x`
- **copy 和 GEMM 多核都不需要改 lowering/codegen**，只需 host 侧分发 + DSL kernel 用 `bx/by` 索引

| Step | 内容 | 改动范围 | 依赖 | 状态 |
|------|------|---------|------|------|
| 1 | `AssignBlackholeCores` 解除 `cores_needed=1` | `assign_blackhole_cores.cc` | 无 | ✅ |
| 2 | `BlackholeModule` 单 Program 多核 launch | `blackhole_module.cc/h` | Step 1 | ✅ |
| 3 | Copy 多核 E2E 验证（TT-Sim） | 测试 | Step 1+2 | ✅ |
| 4 | GEMM 多核 E2E 验证（TT-Sim） | 测试（新 DSL kernel 用 `bx/by`） | Step 1+2 | ✅ |
| 5 | 文档同步与提交 | progress/design/memory | Step 3+4 | ✅ |

不在 Stage 3 范围：K 维度切分、核间数据流、semaphore/multicast

注：
- Stage 3 本体以及后续关联文档已同步到当前状态
- `git commit` / `git push` 已完成；当前阶段剩余工作属于 Stage 3 之后的 TT-Metal contract formalization

### Stage 3 结果

- copy multi-core direct host path 已完成并通过 TT-Sim：`test_blackhole_copy_runtime.py` `6 passed`
- GEMM multi-core direct host path 已完成并通过 TT-Sim：`test_blackhole_gemm.py` `7 passed`
- multicore GEMM 真正 blocker 是 direct path contract mismatch，不是 `core_plan`：
  - host runtime 之前把 `num_k_tiles` 误从整张输入 buffer 大小推导，single-core 碰巧正确、multi-core 失真
  - writer 之前按整张 output tensor 形状消费 output CB，导致 `cb_wait_front` 多消费而挂死
  - `transpose_B=True` 时，reader 之前仍按未转置的 tile 线性序读 B，导致 multi-core 数值错误
- 独立 wrapper/export blocker 已解决：
  - 根因不是 direct path contract，而是 host C codegen 对 `tvm_call_packed_lowered` 只支持语句形态，不支持 `LetStmt` 中的结果表达式
  - 现已修复为显式从 `TVMFFIAny result` 取回返回值，Blackhole `tvm_ffi` export 最小 case 通过

---

## 已完成阶段的关键记录

### Stage 2D（GEMM E2E）

- GEMM 根因：`transpose_B` 丢失 + host row-major upload 无 tilize/untilize
- 已补：`blackhole.gemm_contract`、host-side transpose/tilize/untilize
- CB identity 唯一协议收正：`LowerBlackholeOps` → `requirement_index`，`PlanBlackholeCB` → IR 回写
- 额外收正：`scratch_l1` 全链路移除、copy codegen 统一、`GetRuntimeArgVarForBuffer` preferred_kind 重构

### Stage 2G（Richer Runtime Work Schema）

- copy runtime ABI 已从 `current_work_linear_id` / `tile_count` 收正为 `work_linear_id + a_tile_* + output_tile_*`
- GEMM segment runtime ABI 已收正为 reader 的 `work_linear_id + a_tile_* + b_tile_* + k_tile_*`、compute 的 `k_tile_*`、writer 的 `work_linear_id + output_tile_*`
- `rt_mod_blackhole` / `ExecutableSpec` / `KernelSpec` 已统一消费 richer work descriptor kinds
- `BindThreadIndex` 不再从 copy range 字段静默猜 work id；缺失 `work_linear_id` 时直接 fail-fast
- compile-time ABI schema 已进入 segment plan / `KernelSpec` / direct runtime：
  - accessor CTA 不再只是匿名 compile-time args 位置约定
  - GEMM `Mt/Kt/Nt`、`transpose_A/B` 已作为显式 `compile_time_arg_specs` 进入主协议
  - `launch_spec` 已成为 `CreateKernel` host materialization 的正式输入
- direct runtime 当前正式支持面：
  - copy: equal source/dest range，且 stride = 1
  - GEMM: A/B-separated reader range + writer output range
  - accessor: 仅 interleaved + DRAM + `common_runtime_arg_count = 0`
  - accessor fail-fast 已补齐到统一 schema 校验层：即使 kernel 走 `compile_time_arg_specs` 主路径，只要 accessor 声明 `common_runtime_arg_count > 0` 也会被 direct runtime 明确拒绝
  - copy/GEMM 已新增 direct runtime reject 回归，覆盖 accessor-level `common_runtime_arg_count > 0`

### Stage 2J（Compute Contract Formalization）

- `blackhole.compute_contract` 已进入 `LowerBlackholeOps -> ExecutableSpec -> BlackholeModule` 主链
- GEMM direct runtime 的 shape 校验、transpose/tilize、output untilize、`num_k_tiles` / logical N tiles 推导已优先消费 `compute_contract`
- `blackhole.gemm_contract` 仍保留为兼容字段，但新增 compute 语义不再继续堆在旧字段上
- `compute_contract` 已继续 formalize：
  - block/subblock ABI：`block_m_tiles/block_n_tiles/block_k_tiles`、`subblock_m_tiles/subblock_n_tiles`
  - compute precision ABI：`math_fidelity/fp32_dest_acc_en/dst_full_sync_en/math_approx_mode/unpack_to_dest_mode/bfp8_pack_precise`
  - compute kernel config extras：`defines/named_compile_args`
- `tl.gemm_py` 现有 compute ABI 参数 `clear_accum/k_pack/wg_wait` 已进入 `compute_contract` 和 compute-side `compile_time_arg_specs`
- `tl.gemm_py` 现有 warp-level compute ABI 参数 `policy` 已进入 `compute_contract.policy_type/policy_name` 和 compute-side `compile_time_arg_specs`
- `tl.gemm_py` 可选 `mbar` 绑定已进入 `compute_contract.has_mbarrier/mbarrier_buffer/mbarrier_scope/mbarrier_index_exprs`
- compute segment 已显式产出 `compute_config`、`gemm_block_shape`、`gemm_subblock_shape`
- compute segment 已显式产出 `gemm_clear_accum`、`gemm_k_pack`、`gemm_wg_wait`
- compute segment 已显式产出 `gemm_policy`
- compute segment / `KernelSpec.compute_config` 已改为从 `compute_contract` 派生，不再各自维护独立默认值
- `KernelSpec.compute_config` / direct runtime `CreateKernel(ComputeConfig)` 已收正到更完整 TT-Metal 口径：
  - `dst_full_sync_en`
  - `bfp8_pack_precise`
  - `defines`
  - `named_compile_args`
- `T.gemm` 已补上 producer 输入面：
  - `dst_full_sync_en`
  - `bfp8_pack_precise`
  - `defines`
  - `named_compile_args`
- richer compute-config extras 不再依赖测试注入或 attrs 变异；DSL -> `LowerBlackholeOps` -> `ExecutableSpec` ->
  `KernelSpec.compute_config` -> `CreateKernel(ComputeConfig)` 主链已闭环
- GEMM direct runtime correctness 已补到 richer compute-config case：
  - 新增 non-default `dst_full_sync_en/bfp8_pack_precise/defines/named_compile_args` 的 direct runtime 数值对比
  - TT-Sim 下单测通过，整份 `test_blackhole_gemm.py` 已回到 `30 passed`
- `mbar` 当前按 barrier binding formalize 到 `compute_contract`，未被错误编码成新的 compile-time literal ABI；direct runtime 对 `has_mbarrier=True` 明确 fail-fast
- `BlackholeModule` 已改为按 `KernelSpec.compute_config` materialize TT-Metal `ComputeConfig`，不再把 `math_fidelity/fp32_dest_acc_en/math_approx_mode` 写死
- P0 已完成；当前 remaining gap 已转移到 P3/P4/P5 和更宽 dtype / execution surface，而不是 compute contract producer 链路
- multicore GEMM direct runtime 额外收正：
  - `compute_contract.N/Nt` 在 multicore 语义上是每个 work/core 的 local output shape
  - runtime 侧全局 logical N tile 数需按 `Nt * logical_grid_x` 推导，不能直接把 per-core `Nt` 拿去和 grid 宽度比较
- 新增 `transpose_A=True, transpose_B=True` 的更宽 GEMM compute case 测试；当前环境 direct runtime 用例因执行前置条件不足而跳过，但 schema/spec 主链已验证

### Stage 2E（设备资源 IR）

- `StorageRank::kBlackholeCB`、`StorageRank::kBlackholeAccumulator` 已引入
- `BlackholeDeviceResourceCanonicalization` pass 已接入管线
- generic pass（FlattenBuffer/VectorizeLoop/MergeSharedMemory）不再误解 Blackhole 资源

---

## 未完成的 TT-Metal contract 收正

来源：`stage2d_ttmetal_contract_audit.md`（审计已完成，收正部分落地）

| 优先级 | 内容 | 状态 | 备注 |
|--------|------|------|------|
| P0 | GEMM compute contract / compute config / producer ABI 正式化 | ✅ | `compute_contract` 已成为 compute 真源，`compute_config` 为 materialization 视图；`dst_full_sync_en/bfp8_pack_precise/defines/named_compile_args` 与 `clear_accum/k_pack/wg_wait/policy` 已走通 DSL -> attrs/spec -> runtime 主链 |
| P1 | CB transport schema | ✅ | 已统一到 codegen CB transport，无 scratch |
| P2 | host tilize/untilize | ✅ | transpose_B + tilize/untilize 已补齐 |
| P3 | accessor / runtime work schema | ✅ | current copy/GEMM formal surface 上的 richer work descriptor、accessor/common-runtime schema、compile-time ABI/launch schema、kernel-level shared `common_runtime_args` host materialization、以及 accessor `args_config_bits` 真源关系都已正式化；更宽 accessor/CRTA/non-tile execution surface 已转移到 P4/P5 或后续专项 |
| P4 | copy/dataflow 泛化（non-tile/stick/sharded） | 部分完成 | interleaved stick/page copy 已扩到 `M x W`（`M` 为 32 的倍数）并支持静态 offset subrange，formal direct-path boundary 为 `transport_page_size` 必须 64B 对齐、transport offset 必须 page-aligned、global width 必须能整除 shared width；更宽 non-tile/sharded 仍未做 |
| P5 | multi-core synchronization 预埋（semaphore/multicast） | 部分完成 | program-local `semaphore_plan` schema、kernel-level `semaphore_bindings`、`semaphore_id_u32` runtime materialization、最小 device-side dataflow semaphore builtin、以及 worker producer/consumer direct-runtime TT-Sim E2E 已接入；当前仍只支持 worker semaphore，multicast / global semaphore / compute-kernel semaphore primitive / pass-level producer 仍未做 |

---

## 已知结构问题

| 问题 | 优先级 | 备注 |
|------|--------|------|
| `PlanBlackholeCB` 是 MVP allocator | 低 | 当前足够 |
| `StorageRewrite` 不兼容 Blackhole CB | — | 永久排除 |
| copy/GEMM segment 模型不统一（fused_dataflow vs 3-kernel） | 中 | 架构债，Stage 3 后再做 |
| `BlackholeModule` host materialization 过重 | 中 | cleanup roadmap 已建档 |

---

## 设计文档索引

### 活动文档

| 文档 | 用途 | 状态 |
|------|------|------|
| `final_blackhole_backend_redesign.md` | 唯一总设计 | 常青 |
| `stage3_multicore_design.md` | 多核设计 | ✅ 已实施（formal direct host path） |
| `stage2g_unified_work_schema.md` | richer runtime work schema 设计 | ✅ 已实施（copy/GEMM 主路径） |
| `stage2h_accessor_schema.md` | accessor/common-runtime schema 设计 | ✅ 已实施（schema/spec） |
| `stage2i_compile_time_abi_schema.md` | compile-time ABI schema 设计 | ✅ 已实施（schema/spec/direct runtime） |
| `stage2j_compute_contract_schema.md` | compute contract 正式化设计 | ✅ 已实施（schema/spec/runtime 主链） |
| `stage2k_common_runtime_materialization.md` | kernel-level shared common runtime materialization | ✅ 已实施（shared buffer/semaphore common args） |
| `stage2l_accessor_args_config_bits.md` | accessor args_config_bits 协议收正 | ✅ 已实施（与 TT-Metal ArgConfig 对齐） |
| `stage2d_ttmetal_contract_audit.md` | TT-Metal contract 缺口审计 | 收正进行中（P0/P1/P2/P3 ✅，P4 已完成最小 interleaved stick/page path，P5 已推进到 worker semaphore producer/consumer E2E） |
| `stage4_semaphore_schema.md` | P5 semaphore schema 预埋 | 已实现（program-local worker semaphore + kernel binding + 最小 dataflow semaphore builtin + worker producer/consumer E2E） |
| `stage4_copy_stick_generalization.md` | P4 stick/page copy 泛化 | ✅ 已实施（interleaved + DRAM + `M x W`, `M % 32 == 0`，支持静态 offset subrange；未对齐 64B transport page、未 page-align 的 offset、以及 non-divisible global width fail-fast） |
| `stage4_backend_cleanup_roadmap.md` | backend cleanup 收敛路线图 | 活动中（分档：短期必须收 / P4 前应收 / P5 前应收） |

### 已完成（仍有参考价值）

| 文档 | 用途 |
|------|------|
| `stage2d_gemm_direct_cb_io.md` | GEMM contract 修复（transpose_B + tilize/untilize） |
| `stage2d_cb_identity_protocol.md` | CB identity 唯一协议 |
| `stage2e_blackhole_device_resource_semantics.md` | 设备资源 IR 语义扩展 |
