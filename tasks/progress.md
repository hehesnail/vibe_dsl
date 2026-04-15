# TileLang Blackhole Backend Progress

> 当前唯一总体设计文档:
> `tasks/dev_design/final_blackhole_backend_redesign.md`

> 当前协议面 disposition table:
> `tasks/dev_design/blackhole_first_principles_protocol_audit.md`

## 1. 当前目标主线

- **日期**: `2026-04-16`
- **总阶段**: Stage 4
- **长期目标链路**:
  `Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec`
- **任务顺序**:
  只按活动设计文档里的 owner cutover 顺序推进：
  1. `task1_spatial_plan_companion.md`
  2. `task2_ttprogram_companion_cutover.md`
  3. `task3_runtime_gate_and_workload_cutover.md`
  4. `blackhole_first_principles_protocol_audit.md`
     中列出的 legacy protocol deletion

`buffer effect / use-role`、`liveness`、
`materialization / source-live-form`
这些都只是
`SpatialPlan owner cutover`
里的子问题，
不再单独充当顶层路线。

## 2. 当前任务状态（以 repo HEAD 为准）

- `Task 0`
  - 已完成它作为根因诊断与设计约束入口的职责
  - 不再单独形成实现路线
- `Task 1: SpatialPlan owner cutover`
  - **状态**: 进行中
  - **当前主任务**
    - 把 `SpatialPlan`
      从
      `ExecutionClosure / ClosureBoundary`
      过渡 companion
      收成
      `ExecutionUnit / DataflowEdge / LayoutSpec / PhasePlan / ValidatedHintSet`
    - 新增 `ValidateSpatialPlan`
    - 吸收
      `work_decomposition / compute_regions / pipeline_stages`
      这组过渡 surface
- `Task 2: TTProgram owner cutover`
  - **状态**: 未开始
  - **前置条件**
    - `Task 1`
      必须先把 validated `SpatialPlan`
      站稳
- `Task 3: ExecutableSpec / leaf reader cutover`
  - **状态**: 未开始
  - **前置条件**
    - `Task 2`
      必须先把 `TTProgram`
      收成唯一 physical realization truth
- `legacy protocol deletion`
  - **状态**: 未开始
  - **前置条件**
    - 前三项 owner cutover
      都已经站稳

## 3. 当前代码现实（以 repo HEAD 为准）

当前 repo HEAD 还没有站在长期目标链路上。

当前 active chain 是：

```text
Normalized Tile TIR
  -> AnalyzeSpatialStructureFacts
  -> BuildSpatialPlanCompanion
  -> SplitBlackholeKernel
  -> AnalyzeBlackholeWorkDecomposition
  -> AnalyzeBlackholeComputeRegions
  -> AnalyzeBlackholePipelineStages
  -> PlanTTBlocks
  -> PlanTTCompute   (PlanTTKernelABI wrapper)
  -> PlanTTTransport (PlanTTCBAlloc wrapper)
  -> BuildTTProgram
  -> ValidateTTProgram
  -> MaterializeBlackholeExecutable
  -> executable extraction / codegen / BlackholeModule
```

对这条链的当前判断固定为：

- `SpatialPlan`
  已接入主链，
  但代码里的实际 object 仍然是
  `ExecutionClosure / ClosureBoundary / ValidatedHintSet`
  这套过渡 companion，
  还不是设计文档要求的
  `ExecutionUnit / DataflowEdge / LayoutSpec / PhasePlan / ValidatedHintSet`
- `ValidateSpatialPlan`
  还没有落地，
  所以上游 virtual spatial/dataflow truth
  还没有自己的正式 fail-closed gate
- `PlanTTCompute`
  和 `PlanTTTransport`
  虽然已经换成 canonical pass 名，
  但 owner planning 仍然分别包在
  `PlanTTKernelABI`
  和 `PlanTTCBAlloc`
  这两套旧实现里
- `BuildTTProgram`
  已经成为 target 聚合入口，
  但聚合输入仍然带着 helper residue /
  internal attrs /
  payload bag 的历史包袱
- `blackhole.work_decomposition /
  blackhole.compute_regions /
  blackhole.pipeline_stages`
  仍然是 active path 的过渡 surface，
  还没有被新的 owner object set 吸收完

## 4. 当前锁定基线

下面这些基线不回退：

- `ExecutableSpec -> rt_mod_blackhole -> BlackholeModule`
  direct host path
- copy / GEMM 当前 admitted support surface
- `bf16`
  作为正式 runtime correctness baseline
- 缺 truth 时 explicit unsupported / fail-fast

## 5. 当前安排的下一批任务

下一批任务固定只做 `Task 1`，
不提前切到 `Task 2/3`：

1. 重写 `SpatialPlan` code schema
   - `ExecutionClosure / ClosureBoundary`
     -> `ExecutionUnit / DataflowEdge`
   - 补 `LayoutSpec / PhasePlan`
     到正式 owner object set
2. 新增 `ValidateSpatialPlan`
   - 让 TT target planning
     只接受 validated `SpatialPlan`
3. 回收过渡 surface
   - 把
     `work_decomposition / compute_regions / pipeline_stages`
     拆回
     `SpatialPlan` owner object /
     owner-side fact family
4. 完成 `Task 1`
   之后，
   再开始安排
   `Task 2: TTProgram owner cutover`
