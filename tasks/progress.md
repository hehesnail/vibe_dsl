# TileLang Blackhole 后端开发进度

> 当前唯一总体设计文档: `tasks/dev_design/final_blackhole_backend_redesign.md`

## 当前阶段

- **阶段**: Stage 3 — multi-core runtime 调度
- **状态**: 设计完成，待实施
- **日期**: 2026-03-26
- **设计文档**: `tasks/dev_design/stage3_multicore_design.md`

### 最新测试结果

| 测试 | 结果 |
|------|------|
| `test_blackhole_copy_pipeline.py` | 16 passed, 1 xfailed |
| `test_blackhole_gemm.py` | 4 passed, 1 skipped |

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
| **Stage 3** | **multi-core runtime 调度** | **⏳ 待实施** |

---

## Stage 3 实施计划

关键调研结论：
- `blockIdx.*` 不被 `ZeroThreadAndLoopVars` 零化 → tile index 自动含 per-core offset
- `BindThreadIndex` 已把 `blockIdx.x/y` → `work_id % grid_x` / `work_id / grid_x`
- **copy 和 GEMM 多核都不需要改 lowering/codegen**，只需 host 侧分发 + DSL kernel 用 `bx/by` 索引

| Step | 内容 | 改动范围 | 依赖 | 状态 |
|------|------|---------|------|------|
| 1 | `AssignBlackholeCores` 解除 `cores_needed=1` | `assign_blackhole_cores.cc` ~5 行 | 无 | ⏳ |
| 2 | `BlackholeModule` 单 Program 多核 launch | `blackhole_module.cc/h` ~40 行 | Step 1 | ⏳ |
| 3 | Copy 多核 E2E 验证（TT-Sim） | 测试 | Step 1+2 | ⏳ |
| 4 | GEMM 多核 E2E 验证（TT-Sim） | 测试（新 DSL kernel 用 `bx/by`） | Step 1+2 | ⏳ |
| 5 | 文档同步与提交 | progress/design/memory | Step 3+4 | ⏳ |

不在 Stage 3 范围：K 维度切分、核间数据流、semaphore/multicast

---

## 已完成阶段的关键记录

### Stage 2D（GEMM E2E）

- GEMM 根因：`transpose_B` 丢失 + host row-major upload 无 tilize/untilize
- 已补：`blackhole.gemm_contract`、host-side transpose/tilize/untilize
- CB identity 唯一协议收正：`LowerBlackholeOps` → `requirement_index`，`PlanBlackholeCB` → IR 回写
- 额外收正：`scratch_l1` 全链路移除、copy codegen 统一、`GetRuntimeArgVarForBuffer` preferred_kind 重构

### Stage 2E（设备资源 IR）

- `StorageRank::kBlackholeCB`、`StorageRank::kBlackholeAccumulator` 已引入
- `BlackholeDeviceResourceCanonicalization` pass 已接入管线
- generic pass（FlattenBuffer/VectorizeLoop/MergeSharedMemory）不再误解 Blackhole 资源

---

## 已知结构问题

| 问题 | 优先级 | 备注 |
|------|--------|------|
| `PlanBlackholeCB` 是 MVP allocator | 低 | 当前足够 |
| `StorageRewrite` 不兼容 Blackhole CB | — | 永久排除 |
| copy/GEMM segment 模型不统一（fused_dataflow vs 3-kernel） | 中 | 架构债，Stage 3 后再做 |
| TT-Metal contract P3-P5 缺层 | 低 | 见 `stage2d_ttmetal_contract_audit.md` |

---

## 设计文档索引

### 活动文档

| 文档 | 用途 |
|------|------|
| `final_blackhole_backend_redesign.md` | 唯一总设计 |
| `stage3_multicore_design.md` | 多核设计（当前活动） |

### 已完成（仍有参考价值）

| 文档 | 用途 |
|------|------|
| `stage2d_ttmetal_contract_audit.md` | TT-Metal contract 缺口审计（P3-P5 结论仍有效） |
| `stage2d_gemm_direct_cb_io.md` | GEMM contract 修复（transpose_B + tilize/untilize） |
| `stage2d_cb_identity_protocol.md` | CB identity 唯一协议 |
| `stage2e_blackhole_device_resource_semantics.md` | 设备资源 IR 语义扩展 |
