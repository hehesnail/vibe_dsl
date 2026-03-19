# TileLang Blackhole 后端开发进度

> 当前唯一总体设计文档: `tasks/dev_design/final_blackhole_backend_redesign.md`
> 说明: 本文档只维护阶段状态、任务拆分和当前下一步。

## 当前阶段

- **阶段**: Stage 2A/2B 主链收正与 copy 正式主链推进
- **日期**: 2026-03-19
- **当前目标**: 先按 TileLang 正式主链收正 Blackhole 的 split 前语义规划 / split 后正式 plan 提取 / host-side materialization 三层，再在这条主链上完成 copy 与后续 GEMM 的语义集成

## 当前状态判断

- Stage 0 的协议与执行载体已经落地：
  - `ExecutableSpec`
  - `rt_mod_blackhole`
  - `BlackholeModule`
- Blackhole 已重新接回正式 host/device 主链：
  - `AnnotateDeviceRegions`
  - `SplitHostDevice`
  - `MakePackedAPI`
  - `LowerDeviceKernelLaunch`
- copy 已开始从 runtime 特化迁回 pass / builtin / codegen 主链：
  - `blackhole.runtime_args`
  - `blackhole.segment_plan`
  - `blackhole.cb_requirements`
  - `blackhole.cb_configs`
  - `tl.blackhole.read_tile_to_cb / write_tile_from_cb`
- staged copy 的最小 direct execution 已覆盖：
  - `32x32`
  - `32x64`
  - `64x32`

当前仍然存在的主要结构问题：

- split 前语义规划仍不够强，copy/gemm 语义仍偏依赖 split 后 matcher 恢复
- `blackhole.core_plan` 仍偏摘要，尚未成为正式 execution plan
- `PlanBlackholeCB` 仍偏 MVP allocator，尚未成为正式 memory planner
- `BlackholeModule` 还没有彻底成为唯一正式 host-side execution path
- `FlattenBuffer` / `VectorizeLoop` / `StorageRewrite` 等通用中后段 pass 仍未安全接回

当前新增设计收束：

- 已明确 Blackhole 应采用三层模型：
  - split 前语义规划
  - split 后正式 plan 提取
  - host-side materialization
- 已明确 external runner 只是 bring-up/debug 工具：
  - 不是正式执行路径
  - 不是阶段完成标准
  - 后续可删除
- 已明确 copy 的正式验收必须补齐：
  - `grid > 1`
  - `bx/by` 参与索引
  - large-shape copy（总数据量 > `1.5MB`）
  - per-core memory plan oversubscription 负例

## 分阶段任务

| 阶段 | 目标 | 状态 | 当前重点 |
|------|------|------|----------|
| Stage 0 | 协议与执行载体 | ✅ 已基本完成 | `ExecutableSpec`、`rt_mod_blackhole`、`BlackholeModule` 已落地 |
| Stage 1 | single-core copy bring-up | ✅ 已完成 | 最小 copy 路径已 bring-up，但不再扩大为正式主线 |
| Stage 2A | pass 主链接入收正 | 🔄 进行中 | 固定 split 前语义规划 / split 后正式 plan 提取 / host-side materialization 三层 |
| Stage 2B | single-core copy 正式主链 | 🔄 进行中 | 让 copy 通过正式 direct host path 完成 E2E |
| Stage 2C | single-core GEMM 语义集成 | ⏳ 未开始 | 用与 copy 相同的结构接入 GEMM |
| Stage 2D | single-core true E2E | ⏳ 未完成 | copy + GEMM 都通过正式 host-device 主路径执行 |
| Stage 3 | multi-core runtime 调度 | ⏳ 未开始 | `CorePlan`、per-core runtime args、多核执行 |

## Stage 2 当前任务拆分

### 任务 1: 固定三层边界

- split 前语义规划
- split 后正式 plan 提取
- `BlackholeModule` direct host-side materialization

### 任务 2: 收正 split 前语义规划

- 在 `LowerTileOp` 保留 copy/gemm 的 Blackhole-preserving 语义
- 不再把 split 后 matcher 作为唯一语义来源

### 任务 3: 收正 split 后 requirement extraction

- `LowerBlackholeOps` 正式提取：
  - `blackhole.segment_plan`
  - `blackhole.runtime_args`
  - `blackhole.cb_requirements`

### 任务 4: 收正 memory planner

- `PlanBlackholeCB` 生成正式 `blackhole.cb_configs`
- `cb_id` deterministic allocation
- `role + lifetime` 复用
- `1572864` bytes hard check

### 任务 5: 收正 execution planner

- `AssignBlackholeCores` 生成正式 `blackhole.core_plan`
- 补足：
  - `logical_grid_x/y`
  - `linearization`
  - `physical_cores`
  - `work_packets`

### 任务 6: 收正 `BlackholeModule` direct path

- 不再依赖 external runner 作为正式执行路径
- 在模块内直接 materialize TT-Metal host objects

### 任务 7: 用 copy 完成正式 E2E

- staged copy
- `grid > 1`
- `bx/by` 参与索引
- large-shape copy
- oversubscription 负例

## 当前下一步

1. 先把 split 前 / split 后 / host-side 三层边界在代码中收正到一致。
2. 收正 `LowerTileOp` 的 Blackhole-aware branch，先把 split 前 copy 语义规划固定下来。
3. 再收正 `LowerBlackholeOps -> PlanBlackholeCB -> AssignBlackholeCores`，让 split 后 attrs 真正变成正式 `runtime_args / cb_configs / core_plan`。
4. 把 `BlackholeModule` direct host path 收正成唯一正式执行路径，不再以 external runner 为主路径。
5. 用 staged copy 补齐：
   - `grid > 1`
   - `bx/by` 索引
   - large-shape copy
   - oversubscription 负例
6. 在不破坏 copy 正式 E2E 的前提下，分批接回 `FlattenBuffer` / `VectorizeLoop` / `StorageRewrite`。
7. 最后用同一结构推进 GEMM。

## 当前活动设计文档

- `tasks/dev_design/final_blackhole_backend_redesign.md`
- `tasks/dev_design/stage2_pass_reuse_matrix.md`
- `tasks/dev_design/stage2_single_core_pass_integration.md`
- `tasks/dev_design/stage2_blackhole_logical_block_launch_plan.md`
- `tasks/dev_design/stage2_concrete_dev_task_plan.md`
