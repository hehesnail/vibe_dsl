# TileLang Blackhole Backend Progress

> 当前唯一总体设计文档:
> `tasks/dev_design/final_blackhole_backend_redesign.md`

> 当前协议面 disposition table:
> `tasks/dev_design/blackhole_first_principles_protocol_audit.md`

## 当前阶段

- **日期**: `2026-04-16`
- **总阶段**: Stage 4
- **目标主线**:
  `Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec`

## 当前代码现实

当前代码还没有完全站在新主链上。

当前实际链路仍然是：

```text
Normalized Tile TIR
  -> SpatialPlan companion
  -> SplitBlackholeKernel
  -> legacy transition attrs / helper wrappers
  -> PlanTTBlocks
  -> PlanTTCompute / PlanTTTransport with legacy helper residue
  -> BuildTTProgram
  -> TTProgram companion
  -> ValidateTTProgram
  -> MaterializeBlackholeExecutable
  -> executable extraction / codegen / BlackholeModule
```

当前根因口径已经更新为：

- `SpatialPlan`
  过薄，不是真正的 virtual spatial/dataflow layer
- `TTProgram`
  仍混有 bridge / helper residue
- build/codegen/runtime
  仍在读 fake protocol

当前任务重排口径：

- 不再按旧 `R0 / R1 / R2`
  编号阅读当前 roadmap
- 当前 roadmap 只按层 owner cutover 排序：
  `SpatialPlan -> TTProgram -> ExecutableSpec/leaf -> legacy deletion`
- 旧 `buffer effect / liveness / materialization`
  这些工作现在只作为
  `SpatialPlan owner cutover`
  里的子问题存在，
  不是顶层 roadmap

## 当前文档状态

当前文档已切换到新的 layered IR 口径：

- `SpatialPlan`
  = virtual spatial/dataflow program
- `TTProgram`
  = 唯一 physical realization truth
- `ExecutableSpec`
  = 只做 leaf projection

但代码还没有完全跟上这套边界。

## 当前未完成

1. **`SpatialPlan owner cutover`**
   - 引入
     `ExecutionUnit / DataflowEdge / LayoutSpec / PhasePlan / ValidatedHintSet`
   - 新增 `ValidateSpatialPlan`
   - 让当前 analysis / planner 子步骤
     显式服务这组 owner object
2. **`TTProgram owner cutover`**
   - 显式收正
     `TTBlockPlan / TTKernelPlan / TTTransportPlan /
      TTSyncPlan / TTABIPlan / TTExecutionPlan`
   - 让 `BuildTTProgram`
     退成纯聚合器
3. **`ExecutableSpec / leaf reader cutover`**
   - 去掉 build/codegen/runtime
     对 legacy gate attrs
     和其它 fake protocol 的依赖
   - 保证 `ExecutableSpec`
     只做 `TTProgram` 投影
4. **`legacy protocol deletion`**
   - 审计表列出的 fake/legacy protocol
     全部退出长期协议面

## 当前执行优先级

当前主路线固定为：

1. **先完成 `SpatialPlan owner cutover`**
   - 先把 object model 和 validator 立起来
   - 再把当前零散 analysis/planner
     收到这层边界里
2. **再完成 `TTProgram owner cutover`**
   - 把 target planning owner
     从 helper residue 里拉出来
3. **再完成 `ExecutableSpec / leaf reader cutover`**
   - 把 leaf gate 收回
     `TTProgram / ExecutableSpec`
4. **最后做 `legacy protocol deletion`**
   - 只在新 owner truth
     已经稳定后删旧面

当前只完成了一个 preparatory substep：

- `buffer effect / use-role analysis`
  已接入 active path：
  `LowerToBlackholePhaseB`
  现在显式产出
  `blackhole.buffer_effect_use_role_facts`
- `PlanTTCompute`
  现在对这份 analysis attr
  fail-closed；
  缺失时不再默默回退到后段猜语义

这不等于
`SpatialPlan owner cutover`
已经完成；
它只是其中一个前置子步骤。

## 当前正式基线

下面这些基线不回退：

- `ExecutableSpec -> rt_mod_blackhole -> BlackholeModule`
  direct host path
- copy / GEMM 当前 admitted support surface
- `bf16`
  作为正式 runtime correctness baseline
- 缺 truth 时 explicit unsupported / fail-fast

## 下一步

1. 先把 `SpatialPlan`
   的 owner object
   (`ExecutionUnit / DataflowEdge / LayoutSpec / PhasePlan`)
   立起来
2. 再把 `liveness`
   和 `materialization / source-live-form`
   提成这层的独立 analysis / planner
3. 再让 `PlanTT*`
   和 `BuildTTProgram`
   开始读这组新 truth
4. 然后才进入
   `TTProgram owner cutover`
   与 leaf reader cutover
