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
- **当前阶段命名约定**:
  - `Task 0: Root Cause and Rewrite Direction`
  - `Task 1: SpatialPlan Owner Cutover`
  - `Task 2: TTProgram Owner Cutover`
  - `Task 3: ExecutableSpec / Leaf Reader Cutover`
  - `Legacy Protocol Deletion`
- **任务顺序**:
  只按活动设计文档里的 canonical stage label 顺序推进：
  1. `Task 1: SpatialPlan Owner Cutover`
     (`task1_spatial_plan_companion.md`)
  2. `Task 2: TTProgram Owner Cutover`
     (`task2_ttprogram_companion_cutover.md`)
  3. `Task 3: ExecutableSpec / Leaf Reader Cutover`
     (`task3_runtime_gate_and_workload_cutover.md`)
  4. `Legacy Protocol Deletion`
     (`blackhole_first_principles_protocol_audit.md` disposition table)

`buffer effect / use-role`、`liveness`、
`materialization / source-live-form`
这些都只是
`Task 1: SpatialPlan Owner Cutover`
里的子问题，
不再单独充当顶层路线。

## 2. 当前任务状态（以 repo HEAD 为准）

- `Task 0: Root Cause and Rewrite Direction`
  - 已完成它作为根因诊断与设计约束入口的职责
  - 不再单独形成实现路线
- `Task 1: SpatialPlan Owner Cutover`
  - **状态**: 已完成
  - **repo HEAD 已收口**
    - `SpatialPlan`
      已收成
      `ExecutionUnit / DataflowEdge / LayoutSpec / PhasePlan / ValidatedHintSet`
      这组 primary owner object
    - `ExecutionClosure / ClosureBoundary`
      只保留为 compatibility projection
    - `ValidateSpatialPlan`
      已落地，
      并通过
      `tl.spatial_plan_validated`
      成为 `PlanTT*`
      和 lowering requirement builder
      的正式前置 gate
    - `PlanTTTransport / BuildTTProgram / lowering requirement`
      已改为消费
      `DataflowEdge / PhasePlan`
      这组主 truth，
      不再从 legacy spatial projection
      恢复 virtual spatial/dataflow 关系
- `Task 2: TTProgram Owner Cutover`
  - **状态**: 已完成
  - **repo HEAD 已收口**
    - `PlanTTSync / PlanTTABI / PlanTTExecution`
      已显式落地，
      并已接入
      Python wrapper
      与
      `LowerToBlackholeTTProgram`
      canonical bundle
    - `TTProgram`
      已显式携带
      `block_plans / kernel_plans / transport_plans /
       sync_plans / abi_plans / execution_plans`
      这组 owner slice；
      `TTKernel / TTCoreGroup / TTCBPlan /
       TTSemaphorePlan / TTComputeSyncPlan / TTDstLayoutPlan`
      只保留 compatibility / realization detail
    - `BuildTTProgram`
      已退成纯聚合器，
      只读取显式 owner attrs，
      不再内联生成
      sync / dst-layout / execution / hardware payload
    - `PlanTTKernelABI / PlanTTCBAlloc`
      仍可作为
      `PlanTTCompute / PlanTTTransport`
      的实现细节存在，
      但不再以 public owner pass 身份对外暴露
- `Task 3: ExecutableSpec / Leaf Reader Cutover`
  - **状态**: 已完成
  - **repo HEAD 已收口**
    - build / codegen / runtime / `BlackholeModule`
      已只消费
      `tl.blackhole_executable`
      与其内部
      `ExecutableSpec`
      投影，
      不再读取
      `tl.tt_program`
      或
      `blackhole.lowering_requirements`
    - `MaterializeBlackholeExecutable`
      现在承接
      `TTProgram.payload`
      里的 leaf-only build contracts，
      包括
      `buffer_tile_bridge_specs`
      和
      `unsupported_compute_ops`
    - synthetic segment materialization
      已改成只重建
      segment-local
      `tl.blackhole_executable`
      视图，
      不再在内部 leaf func
      重新挂最小 `TTProgram`
- `Legacy Protocol Deletion`
  - **状态**: 未开始
  - **前置条件**
    - `Task 1 / Task 2 / Task 3`
      都已经站稳

## 3. 当前代码现实（以 repo HEAD 为准）

当前 active chain 是：

```text
Normalized Tile TIR
  -> AnalyzeSpatialStructureFacts
  -> BuildSpatialPlanCompanion
  -> ValidateSpatialPlan
  -> SplitBlackholeKernel
  -> AnalyzeBlackholeWorkDecomposition
  -> AnalyzeBlackholeComputeRegions
  -> AnalyzeBlackholePipelineStages
  -> PlanTTBlocks
  -> PlanTTCompute
  -> PlanTTTransport
  -> PlanTTSync
  -> PlanTTABI
  -> PlanTTExecution
  -> BuildTTProgram
  -> ValidateTTProgram
  -> MaterializeBlackholeExecutable
  -> executable extraction / codegen / BlackholeModule
```

对这条链的当前判断固定为：

- `SpatialPlan`
  已经收成
  `ExecutionUnit / DataflowEdge / LayoutSpec / PhasePlan / ValidatedHintSet`
  这组 primary owner object；
  `ExecutionClosure / ClosureBoundary`
  只保留 compatibility projection
- `ValidateSpatialPlan`
  已落地，
  并通过
  `tl.spatial_plan_validated`
  成为后续 target planning / lowering requirement
  的正式 fail-closed gate
- `PlanTTTransport / BuildTTProgram / lowering requirement`
  已切到从
  `DataflowEdge / PhasePlan`
  读取 virtual spatial/dataflow truth，
  不再从 legacy spatial projection
  恢复 phase / channel 关系
- `BuildTTProgram`
  已退成纯聚合器，
  只消费
  `TTBlockPlan / TTKernelPlan / TTTransportPlan /
   TTSyncPlan / TTABIPlan / TTExecutionPlan`
  这组显式 owner slice
- `TTProgram.payload`
  现在只承接
  executable writer
  所需的 leaf projection contract；
  真正的 leaf reader
  已切到
  `tl.blackhole_executable`
- `blackhole.work_decomposition /
  blackhole.compute_regions /
  blackhole.pipeline_stages`
  仍然是 active path 的迁移 residue，
  但当前只承接
  TT owner 细化 / lowering support facts；
  不再承担 `SpatialPlan`
  的 primary owner 身份

## 4. 当前临时验证面

下面这些只是重写期间的临时验证面，
不是架构真源，
也不能否决第一性原理 redesign：

- `ExecutableSpec -> rt_mod_blackhole -> BlackholeModule`
  direct host path
- copy / GEMM 当前 admitted support surface
- `bf16`
  作为当前 runtime correctness gate
- 缺 truth 时 explicit unsupported / fail-fast

如果新主链建立后，
这些验证面和当前 Owner Cutover 阶段冲突，
应该改的是验证面，
不是回头把旧实现抬成长期协议。

## 5. 当前安排的下一批任务

下一批任务固定切到
`Legacy Protocol Deletion`：

1. 显式拉出
   已失效的
   legacy transition attrs /
   helper bridge residue
2. 继续清理
   `TTProgram`
   过渡期 compatibility residue
3. 在 leaf reader cutover 站稳后，
   再恢复更宽
   workload payoff /
   support surface
