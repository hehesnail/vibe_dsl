# Stage 4 Stage 0: P0 Guardrails And Cutover Gates

## 基本信息

- **文档角色**: `Stage 0` 已落地边界文档
- **当前状态**: 已完成；作为后续 `Phase A / B / C` 的前置假设保留
- **唯一总体设计**: `tasks/dev_design/final_blackhole_backend_redesign.md`

## 1. 目标

`Stage 0` 不直接解决任何特定 workload 的 correctness。
它只负责建立 layered IR 迁移护栏，防止后续阶段在新增 companion IR 的同时，
继续让 legacy attrs 长成第二真源。

## 2. 已落地的稳定边界

`Stage 0` 已经固定了下面这些长期约束：

- `IRModule.global_infos["tl.device_programs"]` 是唯一 module-scope device-program registry
- `PrimFunc.attrs["tl.semantic_seeds"]` 是 pre-lift typed input 通道
- post-lift hard-freeze 与 unsafe mutation 失效规则已经建立
- `tl.semantic_program / tl.spatial_program / tl.tt_program` 的 invalidation
  contract 已定义
- compatibility deletion gates 已成为正式迁移纪律，而不是临时口头约定

## 3. 代码落点

`Stage 0` 当前对应的稳定实现包括：

- [collect_device_programs.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/collect_device_programs.cc)
- [project_semantic_seeds.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/project_semantic_seeds.cc)
- [semantic_program.h](/root/dev/vibe_dsl/tilelang_repo/src/transform/common/semantic_program.h)
- [phase.py](/root/dev/vibe_dsl/tilelang_repo/tilelang/engine/phase.py)
- [__init__.py](/root/dev/vibe_dsl/tilelang_repo/tilelang/transform/__init__.py)
- [test_blackhole_semantic_ir.py](/root/dev/vibe_dsl/tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py)

## 4. 后续阶段可以直接假定的事实

`Phase A / B / C` 现在可以直接依赖：

- `SplitHostDevice` 之前已经存在稳定的 device-program registry
- semantic lift 之前已经存在 typed seed 通道
- unsafe TIR mutation 不会让 companion IR 进入 silent stale 状态
- deletion gates 已经要求新主链稳定后再删除 compatibility path

## 5. 不属于 Stage 0 的事情

这份文档不再承担：

- 逐步实施 checklist
- 一次性验证数字
- workload-specific semantic recovery 说明

这些内容现在分别归属：

- 当前验证快照: [progress.md](/root/dev/vibe_dsl/tasks/progress.md)
- semantic recovery 边界: [stage4_phase_a_semantic_ir.md](/root/dev/vibe_dsl/tasks/dev_design/stage4_phase_a_semantic_ir.md)
- spatial / target cutover: [stage4_phase_b_spatial_ir.md](/root/dev/vibe_dsl/tasks/dev_design/stage4_phase_b_spatial_ir.md) 和 [stage4_phase_c_tt_target_ir.md](/root/dev/vibe_dsl/tasks/dev_design/stage4_phase_c_tt_target_ir.md)
