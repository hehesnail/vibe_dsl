# 任务开发设计文档

> 当前唯一总体设计文档: `final_blackhole_backend_redesign.md`

本目录中的文档分三类：

## 1. 当前活动文档

- `final_blackhole_backend_redesign.md`
  - 当前唯一总体设计
- `stage2d_gemm_integration.md`
  - 当前主任务设计：Stage 2D Step 6 GEMM direct-path E2E 验收
- `stage2e_blackhole_device_resource_semantics.md`
  - 刚完成的 Stage 2E 设计与实现依据

## 2. 当前仍有效的支撑设计

- `stage2_pass_reuse_matrix.md`
  - Stage 2 pass 接入边界与复用矩阵
- `stage2_single_core_pass_integration.md`
  - Stage 2 分层目标与阶段拆分的支撑设计
- `stage2_blackhole_logical_block_launch_plan.md`
  - block/core/memory/launch plan 的支撑设计

## 3. 已完成阶段的历史设计

- `stage0_executable_spec_attr_alignment.md`
  - Stage 0 协议与 `ExecutableSpec` 落地历史记录
- `stage1_single_core_copy_closure.md`
  - Stage 1 single-core copy 执行闭环历史记录
- `stage2_concrete_dev_task_plan.md`
  - 早期 Stage 2 任务拆解，已被 `tasks/progress.md` 与后续 Stage 2D/2E 设计取代

这些文档保留是为了回溯阶段决策，不再承载最新总体架构。

## 4. 历史环境准备记录

- `phase0_tilelang_setup.md`
- `phase0_tt_metal_build.md`
- `phase0_tt_sim_build.md`
- `phase0_tilelang_blackhole_config.md`

这些文档只保留环境与早期尝试的历史信息，不再作为当前 Blackhole 架构或阶段拆分的依据。
