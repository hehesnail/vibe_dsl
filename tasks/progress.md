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
- `Cleanup Task 0`
  已在 repo HEAD 落地：
  exact TT-Metal builtin surface 已锁定，
  `SelectBlackholeTTMetalBuiltins`
  已插入 active chain，
  helper/composite builtin residue
  与
  local pseudo builtin residue、
  `compute_epilogue_ops`
  已从 active compute surface 移除；
  selector 创建的 exact temporary CB requirement
  当前暂经
  `blackhole.cb_requirements`
  这个窄 seed
  传到
  `PlanTTCompute / PlanTTCBAlloc`；
  这是 pre-planner exact selection 的过渡残留，
  不是新的 planning 语义。
- 当前硬约束固定为：
  跨 pass / 跨阶段语义
  只允许存在于
  `Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec`
  这几层显式表示里；
  analysis bag / payload / helper wrapper / narrowing seed
  只要不能在 analysis 失效后由当前层重新推出，
  就必须继续收回对应 IR，
  不能扩成新的协议面。
- `task1_spatial_plan_companion.md`、
  `task2_ttprogram_companion_cutover.md`、
  `task3_runtime_gate_and_workload_cutover.md`
  现在负责定义 **显式表示层边界 / completion contract**；
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
  public analysis wrapper、
  attr bag / evidence bag、
  leaf-time semantic recovery、
  `blackhole.copy_semantics`、
  `blackhole.segment_kind`、
  `blackhole.lowering_requirements`
  及其窄化过渡 seed、
  `blackhole.resource_plan`
  等。
- 因此，
  旧的“前面几个任务已经做完，只剩 support surface 扩展”
  这一判断作废；
  当前必须从
  `Cleanup Task 1`
  继续把 cleanup 主线做完，
  再谈 runtime admission / workload payoff 扩展。

## 2. 以当前设计重判 repo HEAD 状态

| 项目 | 新状态 | 说明 |
|------|--------|------|
| `Task 0: Root Cause and Rewrite Direction` | 已完成 | 作为根因诊断与 pass 纪律基线已经完成；它不再单独形成实现路线 |
| `Task 1: SpatialPlan Owner Cutover` | 重新打开 / 部分完成 | `SpatialPlan` 仍未在 active chain 中以单一 direct builder 直接从当前 TIR 构造；`AnalyzeSpatialStructureFacts` 仍以 public wrapper 形态存在。当前 `BuildSpatialPlanCompanion` 只是历史实现名，不应被当成架构边界 |
| `Task 2: TTProgram Owner Cutover` | 重新打开 / 部分完成 | exact TT-Metal builtin basis 已锁定并前移到 dedicated selector；但 `blackhole.lowering_requirements` 窄 seed、planner residue 仍在主链 |
| `Task 3: ExecutableSpec / Leaf Reader Cutover` | 重新打开 / 部分完成 | runtime / projection / executable writer 已不再消费 `compute_epilogue_ops`；但 `blackhole.segment_kind` 等旧叶子协议仍未删净 |
| `Legacy Protocol Deletion` | 未完成 / 当前主任务 | helper/composite builtin active surface 与 `compute_epilogue_ops` 已清掉；`blackhole.copy_semantics`、`blackhole.segment_kind`、`AnalyzeBlackhole*`、`blackhole.lowering_requirements`、`blackhole.resource_plan` 仍在 repo HEAD |

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
  -> legacy attrs / helper wrapper residue
  -> PlanTTBlocks
  -> SelectBlackholeTTMetalBuiltins
  -> PlanTTCompute / PlanTTTransport / PlanTTSync / PlanTTABI / PlanTTExecution
  -> BuildTTProgram
  -> ValidateTTProgram
  -> MaterializeBlackholeExecutable
  -> runtime / codegen leaf readers
```

这里列的是当前代码里的 pass/phase，
不是额外的长期 IR 层；
长期主链仍然只有
`Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec`。

当前确认仍在 repo HEAD 上的旧面包括：

- `tilelang_repo/tilelang/engine/phase.py`
  仍把
  `AnalyzeSpatialStructureFacts`
  放在 active chain 上，
  但现在已经在
  `PlanTTBlocks`
  后、
  `PlanTTCompute`
  前
  插入
  `SelectBlackholeTTMetalBuiltins`
- `tilelang_repo/tilelang/transform/__init__.py`
  仍对外暴露
  `AnnotateBlackholeCopySemantics`、
  `AnnotateBlackholeSegmentKind`、
  `AnalyzeSpatialStructureFacts`
- `tilelang_repo/src/tir/builtin_blackhole.h`
  和
  `tilelang_repo/src/target/codegen_blackhole.cc`
  的 helper/composite builtin declarations / codegen entries
  与 local pseudo builtin surface
  已删掉；
  当前只剩 selector denylist
  和少量内部 matcher / counter 名字
  作为 cleanup 守卫，
  不再属于 active IR protocol
- `lower_blackhole_ops.cc`、
  `blackhole_device_resource_canonicalization.cc`、
  `rt_mod_blackhole.cc`
  仍依赖
  `blackhole.copy_semantics`、
  `blackhole.segment_kind`、
  `blackhole.lowering_requirements`
  或其过渡窄 seed；
  其中
  `compute_epilogue_ops`
  已不再进入
  `tt_program_projection.h / rt_mod_blackhole.cc / codegen_blackhole.cc`
- `analyze_blackhole_compute_regions.cc`
  与
  `common/blackhole_lowering_requirements.cc`
  仍构成旧 analysis / evidence path
- 测试已改成显式校验
  helper/composite builtin residue
  与
  `compute_epilogue_ops`
  不再出现；
  后续 cleanup 不能回退这些 gate

结论固定为：

- repo HEAD 已经有一部分显式表示层 / validator / direct runtime 骨架，
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
cutover 文档给 completion contract**：

1. **`Cleanup Task 0`**（已完成，`2026-04-17`）
   - 文档：
     `tasks/dev_design/2026-04-16-blackhole-final-legacy-protocol-cleanup-task0.md`
   - 目标：
     锁定 exact TT-Metal builtin surface，
     引入 dedicated builtin-selection pass，
     删除不与 TT-Metal API 一一对应的 helper/composite builtin 方向
   - 当前 repo HEAD 结果：
     `SelectBlackholeTTMetalBuiltins`
     已前移到
     `PlanTTCompute`
     之前，
     helper/composite builtin residue
     与
     `compute_epilogue_ops`
     已 fail-closed

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
     `SpatialPlan`
     必须收成单一 direct builder implementation；
     `AnalyzeSpatialStructureFacts`
     若还需要，只能退回同一 `.cc` 里的 pass-local helper，
     不能再作为 public wrapper / active-chain 语义 pass 存在；
     当前
     `BuildSpatialPlanCompanion`
     这个名字如果继续保留，
     也只表示实现入口，
     不表示新的表示层

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
     剩余表示层 residue 的关键收口

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
    的显式表示边界

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
    的显式表示内容、builder 纪律、validator 与完成判据；
    文件名中的 `companion`
    只是历史索引
- `task2_ttprogram_companion_cutover.md`
  - 定义
    `TTProgram`
    的显式表示内容与 planner / builder 完成判据；
    文件名中的 `companion`
    同样只作历史索引
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
   `Cleanup Task 1`
   开始，
   用 direct logical bridge capture
   取代 compute-region / helper-side bridge 恢复
2. 保持
   `Cleanup Task 0`
   已建立的 gate：
   `PlanTTCompute`
   必须要求
   `tl.blackhole_tt_metal_builtin_selection`，
   helper/composite builtin residue
   与
   `compute_epilogue_ops`
   不得回流
3. 把当前仅为 selector-forwarding 保留的窄过渡 seed
   继续限制在
   `buffer_materialization_contracts /
    buffer_tile_bridge_specs`
   这两个稳定 surface，
   并在对应表示层收口时删掉；
   不把它扩成新的 planning 语义

当前已确认的实现约束补充为：

- `Cleanup Task 0`
  已证明 exact builtin selector
  必须前移到 pre-planner anchored TIR，
  不能等
  `PlanTTKernelABI`
  已经塌缩出 helper/composite builtin
  后再靠 rename / patch 修复
- 当前仍保留的一点表示层 residue 是：
  CB / materialization-sensitive bridge publication
  还没有完全离开
  `PlanTTKernelABI`
  因为 CB requirement index
  仍在那一层规划；
  这正是
  `Cleanup Task 1 / 2`
  接下来要收掉的边界
- 对应的 TT-Metal exact 参考实现
  （例如
  `tt_metal_repo/tests/tt_metal/tt_metal/test_kernels/compute/softmax.cpp`
  与
  `tt_metal_repo/tests/tt_metal/tt_metal/test_kernels/misc/sdpa/reduce_block_max_row/compute.cpp`）
  依赖的是
  `CB + DST` residency、
  exact init/uninit pairing、
  tile-level reduce/broadcast sequence；
  这些现在必须在 selector 之前的 anchored IR 上被看见

在 `Cleanup Task 1-5` 完成前，
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
