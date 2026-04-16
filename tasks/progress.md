# TileLang Blackhole Backend Progress

> 当前唯一总体设计文档:
> `tasks/dev_design/final_blackhole_backend_redesign.md`

> 当前协议面 disposition table:
> `tasks/dev_design/blackhole_first_principles_protocol_audit.md`

> 当前 cleanup 执行总览:
> `tasks/dev_design/2026-04-16-blackhole-final-legacy-protocol-cleanup.md`

## 1. 当前判断

- **日期**: `2026-04-17`
- 这份 `progress.md` 已按当前 redesign 重新定级，不再继承之前那套
  `Task 1 / Task 2 / Task 3 / Legacy Protocol Deletion 已完成`
  的结论。
- `task1_spatial_plan_companion.md`、
  `task2_ttprogram_companion_cutover.md`、
  `task3_runtime_gate_and_workload_cutover.md`
  现在负责定义 **owner boundary / completion contract**；
  当前 repo HEAD 的实际收口顺序，
  由
  `2026-04-16-blackhole-final-legacy-protocol-cleanup*.md`
  这组 cleanup 文档驱动。
- 当前长期主链仍然只有：

```text
Normalized Tile TIR
  -> SpatialPlan
  -> TTProgram
  -> ExecutableSpec
```

- 但 repo HEAD 仍保留大量旧实现残留：
  helper/composite builtins、
  public analysis wrapper、
  attr bag / evidence bag、
  leaf-time semantic recovery、
  `blackhole.copy_semantics`、
  `blackhole.segment_kind`、
  `blackhole.lowering_requirements`、
  `compute_epilogue_ops`。
- 因此，
  旧的“前面几个任务已经做完，只剩 support surface 扩展”
  这一判断作废；
  当前必须先把 cleanup 主线做完，
  再谈 runtime admission / workload payoff 扩展。

## 2. 以当前设计重判 repo HEAD 状态

| 项目 | 新状态 | 说明 |
|------|--------|------|
| `Task 0: Root Cause and Rewrite Direction` | 已完成 | 作为根因诊断与 pass 纪律基线已经完成；它不再单独形成实现路线 |
| `Task 1: SpatialPlan Owner Cutover` | 重新打开 / 部分完成 | `AnalyzeSpatialStructureFacts` 仍在 active chain 且仍有 public wrapper；`BuildSpatialPlanCompanion` 还没有成为唯一 canonical structural builder |
| `Task 2: TTProgram Owner Cutover` | 重新打开 / 部分完成 | exact TT-Metal builtin basis 还没锁定；`lowering_requirements`、helper builtins、`compute_epilogue_ops`、planner residue 仍在主链 |
| `Task 3: ExecutableSpec / Leaf Reader Cutover` | 重新打开 / 部分完成 | runtime / projection / executable writer 仍消费 `blackhole.segment_kind`、`compute_epilogue_ops` 等旧叶子协议 |
| `Legacy Protocol Deletion` | 未完成 / 当前主任务 | `blackhole.copy_semantics`、`blackhole.segment_kind`、`AnalyzeBlackhole*`、`blackhole.lowering_requirements`、`blackhole.resource_plan`、helper/composite builtins 都还在 repo HEAD |

补充重分类：

- `buffer effect / use-role`
- `liveness`
- `materialization / source-live-form`

这些仍然只算
`Task 1: SpatialPlan Owner Cutover`
里的 preparatory substeps，
不能再拿来充当顶层阶段完成判据。

## 3. 当前代码现实（以 repo HEAD 为准）

当前 active chain 应按下面的现实理解，而不是按旧文档里的“已收口主链”理解：

```text
Normalized Tile TIR
  -> AnalyzeSpatialStructureFacts
  -> BuildSpatialPlanCompanion
  -> SplitBlackholeKernel
  -> legacy attrs / helper builtins / lowering_requirements residue
  -> PlanTT* / BuildTTProgram
  -> MaterializeBlackholeExecutable
  -> runtime / codegen leaf readers
```

当前确认仍在 repo HEAD 上的旧面包括：

- `tilelang_repo/tilelang/engine/phase.py`
  仍把
  `AnalyzeSpatialStructureFacts`
  放在 active chain 上
- `tilelang_repo/tilelang/transform/__init__.py`
  仍对外暴露
  `AnnotateBlackholeCopySemantics`、
  `AnnotateBlackholeSegmentKind`、
  `AnalyzeSpatialStructureFacts`
- `tilelang_repo/src/tir/builtin_blackhole.h`
  和
  `tilelang_repo/src/transform/lower_blackhole_ops.cc`
  仍保留并使用
  `blackhole_copy_tile_from_cb`、
  `blackhole_reduce_row`、
  `blackhole_mul_row_bcast`、
  `blackhole_exp2_row_bcast_affine`、
  `blackhole_scalar_max`
- `lower_blackhole_ops.cc`、
  `blackhole_device_resource_canonicalization.cc`、
  `rt_mod_blackhole.cc`、
  `tt_program_projection.h`
  仍依赖
  `blackhole.copy_semantics`、
  `blackhole.segment_kind`、
  `blackhole.lowering_requirements`、
  `compute_epilogue_ops`
- `analyze_blackhole_compute_regions.cc`
  与
  `common/blackhole_lowering_requirements.cc`
  仍构成旧 analysis / evidence path
- 测试仍显式校验这些旧 surface，
  所以必须伴随 cleanup 一起重写

结论固定为：

- repo HEAD 已经有一部分 owner object / validator / direct runtime 骨架，
  但这不能再被表述为
  `Task 1 / Task 2 / Task 3 / Legacy Protocol Deletion`
  已完成
- 当前真正的阻塞点，
  是 cleanup 主线还没做完，
  不是 support surface 还没扩够

## 4. 当前 canonical 执行顺序

从现在开始，
`progress` 里的执行顺序固定改成
**cleanup task 驱动，
owner cutover 文档给 completion contract**：

1. **`Cleanup Task 0`**
   - 文档：
     `tasks/dev_design/2026-04-16-blackhole-final-legacy-protocol-cleanup-task0.md`
   - 目标：
     锁定 exact TT-Metal builtin surface，
     引入 dedicated builtin-selection pass，
     删除不与 TT-Metal API 一一对应的 helper/composite builtin 方向
   - 这是第一优先级，
     因为后面的
     `compute_epilogue_ops`、
     helper builtin、
     fake compute legality
     都依赖它先收口

2. **`Cleanup Task 1`**
   - 文档：
     `tasks/dev_design/2026-04-16-blackhole-final-legacy-protocol-cleanup-task1.md`
   - 目标：
     用 direct logical bridge capture
     取代 compute-region bag，
     只允许保留窄的
     `tl.blackhole_logical_buffer_tile_bridge_specs`

3. **`Cleanup Task 2`**
   - 文档：
     `tasks/dev_design/2026-04-16-blackhole-final-legacy-protocol-cleanup-task2.md`
   - 目标：
     删除 public / internal legacy analysis bags
   - 这一步同时重开并完成
     `Task 1: SpatialPlan Owner Cutover`
     剩余的 pass-discipline 收口：
     `BuildSpatialPlanCompanion`
     必须成为 canonical visitor / matcher / builder；
     `AnalyzeSpatialStructureFacts`
     若还需要，只能退回同一 `.cc` 里的 pass-local helper，
     不能再作为 public wrapper / active-chain owner pass 存在

4. **`Cleanup Task 3`**
   - 文档：
     `tasks/dev_design/2026-04-16-blackhole-final-legacy-protocol-cleanup-task3.md`
   - 目标：
     删除
     `blackhole.copy_semantics`，
     让
     `SplitBlackholeKernel` /
     resource canonicalization /
     lowering
     直接从当前 IR + SpatialPlan 恢复 copy/dataflow meaning

5. **`Cleanup Task 4`**
   - 文档：
     `tasks/dev_design/2026-04-16-blackhole-final-legacy-protocol-cleanup-task4.md`
   - 目标：
     删除
     `blackhole.segment_kind`，
     让 planner / projection / runtime
     直接构造并消费
     kernel kind / segment plan
   - 这一步同时承担
     `Task 2: TTProgram Owner Cutover`
     和
     `Task 3: ExecutableSpec / Leaf Reader Cutover`
     剩余 owner residue 的关键收口

6. **`Cleanup Task 5`**
   - 文档：
     `tasks/dev_design/2026-04-16-blackhole-final-legacy-protocol-cleanup-task5.md`
   - 目标：
     最终 cleanup scan、
     文档同步、
     memory 沉淀、
     测试与验证基线收口

7. **最后才恢复 support surface / workload payoff 扩展**
   - 只有在
     `Cleanup Task 0-5`
     做完后，
     才重新推进
     runtime admission widening、
     direct-runtime admitted surface 扩展、
     更大 workload 的 correctness / payoff 恢复
   - 这些不再允许反向塑形
     `SpatialPlan / TTProgram / ExecutableSpec`
     的 owner 边界

## 5. 各文档在当前阶段里的职责

- `final_blackhole_backend_redesign.md`
  - 唯一总体设计；
    定义长期层边界与 pass 纪律
- `task0_ir_layering_root_cause.md`
  - 根因与第一性原理约束；
    已完成，作为设计基线保留
- `task1_spatial_plan_companion.md`
  - 定义
    `SpatialPlan`
    的 owner object、builder 纪律、validator 与完成判据
- `task2_ttprogram_companion_cutover.md`
  - 定义
    `TTProgram`
    的 owner object 与 planner / builder 完成判据
- `task3_runtime_gate_and_workload_cutover.md`
  - 定义
    `ExecutableSpec / leaf reader`
    的 reader 纪律与 execution gate 边界
- `2026-04-16-blackhole-final-legacy-protocol-cleanup.md`
  - 当前 repo HEAD 的实际实现顺序总览
- `2026-04-16-blackhole-final-legacy-protocol-cleanup-task0.md`
  到
  `task5.md`
  - 当前真正的逐步执行列表

## 6. 当前下一步

当前下一步固定为：

1. 从
   `Cleanup Task 0`
   开始，
   锁定 exact TT-Metal builtin surface
2. 改写 builtin selection，
   删除 helper/composite builtin 路线
3. 同步测试与 `progress / README / audit`
   的相关表述

在 `Cleanup Task 0` 完成前，
不再把下面这些说法当成合法阶段结论：

- “`TTProgram` 主链已经站稳”
- “只差 support surface 扩展”
- “`Legacy Protocol Deletion` 只剩零碎收尾”

## 7. 当前阶段的完成判据

当前这轮重排完成，
至少要同时满足下面几点：

1. `progress.md`
   不再复述旧阶段完成结论，
   并且执行顺序与 cleanup 文档一致
2. `README` /
   总设计里的任务入口不再和 `progress` 打架
3. 后续实现推进统一按
   `Cleanup Task 0 -> 5`
   顺序收口
4. 完整 cleanup 完成前，
   不再把 support surface 扩展当主线
