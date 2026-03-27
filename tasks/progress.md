# TileLang Blackhole 后端开发进度

> 当前唯一总体设计文档: `tasks/dev_design/final_blackhole_backend_redesign.md`

## 当前阶段

- **阶段**: Stage 3 — multi-core runtime 调度
- **状态**: ✅ formal direct host path 已完成；`tvm_ffi` wrapper/export blocker 已修复；TT-Metal contract formalization 已继续推进到 P0 compute contract 正式化（统一 `compute_contract` 承载 GEMM shape/flags/dtype 分层）、P3 richer runtime work schema + accessor/common-runtime schema，以及 compile-time ABI schema/launch schema 的主路径正式化；direct runtime 对未支持 execution 面显式 fail-fast
- **日期**: 2026-03-27
- **设计文档**: `tasks/dev_design/stage3_multicore_design.md`

### 最新回归结果（当前环境）

| 测试 | 结果 |
|------|------|
| `test_blackhole_copy_pipeline.py` | 20 passed, 1 skipped, 1 xfailed |
| `test_blackhole_copy_runtime.py` | 2 passed, 5 skipped |
| `test_blackhole_gemm.py` | 6 passed, 3 skipped |
| `test_blackhole_tvm_ffi_export.py` | 1 passed |

### 已验证 full-env 结果

| 测试 | 结果 |
|------|------|
| `test_blackhole_copy_runtime.py` | 6 passed |
| `test_blackhole_gemm.py` | 7 passed |

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
| 5 | 文档同步与提交 | progress/design/memory | Step 3+4 | ⏳ |

不在 Stage 3 范围：K 维度切分、核间数据流、semaphore/multicast

注：
- 本轮文档已同步到当前状态
- 若以 `AGENTS.md` 的“任务完成”口径结项，仍需在当前批次结束时完成 `git commit` / `git push`

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

### Stage 2J（Compute Contract Formalization）

- `blackhole.compute_contract` 已进入 `LowerBlackholeOps -> ExecutableSpec -> BlackholeModule` 主链
- GEMM direct runtime 的 shape 校验、transpose/tilize、output untilize、`num_k_tiles` / logical N tiles 推导已优先消费 `compute_contract`
- `blackhole.gemm_contract` 仍保留为兼容字段，但新增 compute 语义不再继续堆在旧字段上
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
| P0 | GEMM compile-time ABI 正式化（dtype 分层进 attrs） | 部分完成 | `gemm_contract` 已补 tensor/CB/accumulator dtype 分层，`Mt/Kt/Nt/transpose_A/B` 已进入 `compile_time_arg_specs`；更丰富 compute ABI 仍未做 |
| P1 | CB transport schema | ✅ | 已统一到 codegen CB transport，无 scratch |
| P2 | host tilize/untilize | ✅ | transpose_B + tilize/untilize 已补齐 |
| P3 | accessor / runtime work schema | 部分完成 | richer work descriptor + accessor/common-runtime schema 已进入 segment plan / KernelSpec，compile-time ABI schema/launch schema 也已收正到主路径；current direct runtime 仅正式支持 interleaved 且对 richer execution 面 fail-fast |
| P4 | copy/dataflow 泛化（non-tile/stick/sharded） | ❌ | 不阻塞 Stage 3 |
| P5 | multi-core synchronization 预埋（semaphore/multicast） | ❌ | Stage 3 不涉及核间同步 |

---

## 已知结构问题

| 问题 | 优先级 | 备注 |
|------|--------|------|
| `PlanBlackholeCB` 是 MVP allocator | 低 | 当前足够 |
| `StorageRewrite` 不兼容 Blackhole CB | — | 永久排除 |
| copy/GEMM segment 模型不统一（fused_dataflow vs 3-kernel） | 中 | 架构债，Stage 3 后再做 |

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
| `stage2d_ttmetal_contract_audit.md` | TT-Metal contract 缺口审计 | 收正进行中（P1/P2 ✅，P0 部分，P3 部分完成，P4-P5 未做） |

### 已完成（仍有参考价值）

| 文档 | 用途 |
|------|------|
| `stage2d_gemm_direct_cb_io.md` | GEMM contract 修复（transpose_B + tilize/untilize） |
| `stage2d_cb_identity_protocol.md` | CB identity 唯一协议 |
| `stage2e_blackhole_device_resource_semantics.md` | 设备资源 IR 语义扩展 |
