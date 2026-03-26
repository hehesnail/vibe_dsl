# TileLang Blackhole 后端开发进度

> 当前唯一总体设计文档: `tasks/dev_design/final_blackhole_backend_redesign.md`

## 当前阶段

- **阶段**: Stage 3 — multi-core runtime 调度（⏳ 设计完成，待实施）
- **日期**: 2026-03-26
- **当前测试结果**：
  - `test_blackhole_copy_pipeline.py`: `16 passed, 1 xfailed`
  - `test_blackhole_gemm.py`: `4 passed, 1 skipped`
  - `test_blackhole_gemm_basic`：TT-Sim direct path 数值通过

---

## 分阶段任务

| 阶段 | 目标 | 状态 |
|------|------|------|
| Stage 0 | 协议与执行载体 | ✅ |
| Stage 1 | single-core copy bring-up | ✅ |
| Stage 2A | pass 主链接入 | ✅ |
| Stage 2B | single-core copy 正式主链 | ✅ |
| Stage 2C | split-before 语义规划 | ✅ |
| Stage 2D | single-core true E2E | ✅ copy + GEMM |
| Stage 2E | 设备资源 IR 语义扩展 | ✅ StorageRank + Canonicalization |
| Stage 3 | multi-core runtime 调度 | ⏳ 设计完成，待实施 |

---

## Stage 3 实施计划

设计文档：`tasks/dev_design/stage3_multicore_design.md`

关键调研结论：
- `blockIdx.*` 不被 `ZeroThreadAndLoopVars` 零化 → tile index 自动含 per-core offset
- `BindThreadIndex` 已把 `blockIdx.x/y` → `work_id % grid_x` / `work_id / grid_x`
- **copy 和 GEMM 多核都不需要改 lowering/codegen**，只需 host 侧分发 + DSL kernel 用 `bx/by` 索引

### 任务分解

| Step | 内容 | 改动范围 | 依赖 | 状态 |
|------|------|---------|------|------|
| 1 | `AssignBlackholeCores` 解除 `cores_needed=1` | `assign_blackhole_cores.cc` ~5 行 | 无 | ⏳ |
| 2 | `BlackholeModule` 单 Program 多核 launch | `blackhole_module.cc/h` ~40 行 | Step 1 | ⏳ |
| 3 | Copy 多核 E2E 验证（TT-Sim） | 测试 | Step 1+2 | ⏳ |
| 4 | GEMM 多核 E2E 验证（TT-Sim） | 测试（新 DSL kernel） | Step 1+2 | ⏳ |
| 5 | 文档同步与提交 | progress/design/memory | Step 3+4 | ⏳ |

不在 Stage 3 范围：K 维度切分、核间数据流、semaphore/multicast

---

## Stage 2D 完成记录

- Steps 1-6 全部完成
- GEMM 根因：`transpose_B` 丢失 + host row-major upload 无 tilize/untilize
- 已补：`blackhole.gemm_contract`、host-side transpose/tilize/untilize
- 额外收正：`scratch_l1` 全链路移除、copy codegen 统一、`GetRuntimeArgVarForBuffer` preferred_kind 重构

---

## 已知结构问题

- `PlanBlackholeCB` 仍是 MVP allocator，非正式 memory planner
- `StorageRewrite` 与 Blackhole CB 模型不兼容（永久排除）
- copy 用 fused_dataflow 单 kernel，GEMM 用 3-kernel（后续统一）
- TT-Metal contract 缺层审计见 `stage2d_ttmetal_contract_audit.md`（P3-P5 仍有欠账）

---

## 当前活动设计文档

- `final_blackhole_backend_redesign.md` — 唯一总设计
- `stage3_multicore_design.md` — 多核设计（当前活动）
- `stage2d_ttmetal_contract_audit.md` — TT-Metal contract 审计（已完成，结论仍有效）
- `stage2d_gemm_direct_cb_io.md` — GEMM contract 修复（已完成）
