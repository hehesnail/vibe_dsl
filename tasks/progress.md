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

当前 closure 进度补充：

- `R0.1`
  已接入 active path：
  `LowerToBlackholePhaseB`
  现在显式产出
  `blackhole.buffer_effect_use_role_facts`
- `PlanTTCompute`
  现在对这份 analysis attr
  fail-closed；
  缺失时不再默默回退到后段猜语义
- `R0.2 / R0.3`
  仍未开始 code cutover；
  `liveness` 和
  `materialization / source-live-form`
  还没有独立站成 pass

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

1. **`SpatialPlan` 重建**
   - 引入
     `ExecutionUnit / DataflowEdge / LayoutSpec / PhasePlan`
   - 新增 `ValidateSpatialPlan`
2. **`TTProgram` owner 收口**
   - 显式化
     `PlanTTSync / PlanTTABI / PlanTTExecution`
   - 让 `BuildTTProgram`
     退成纯聚合器
3. **leaf reader 收口**
   - 去掉 build/codegen/runtime
     对 legacy gate attrs
     和其它 fake protocol 的依赖
4. **legacy protocol 退场**
   - 审计表列出的 fake/legacy protocol
     全部退出长期协议面

## 当前执行优先级

代码 cutover 顺序仍固定为：

1. `R0.2`
   - buffer liveness analysis
2. `R0.3`
   - materialization / source-live-form planner decision
3. `R1.1`
   - 去掉 build/codegen/executable extraction
     对 legacy gate attrs 的依赖
4. `R2.1`
   - 显式化 `PlanTTSync / PlanTTABI / PlanTTExecution`

已完成的当前 closure 项：

- `R0.1`
  - `buffer effect / use-role analysis`
    已落地并接入
    `LowerToBlackholePhaseB`

文档重写不是新的 roadmap 项；
它是这组代码任务的前置对齐。

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
   的 object model 和 validator 立起来
2. 再让 `PlanTT*`
   和 `BuildTTProgram`
   读新 truth
3. 再切 build/codegen/runtime readers
4. 最后删 fake protocol 和 helper residue
