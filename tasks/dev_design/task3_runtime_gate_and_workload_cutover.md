# Task 3: Runtime Gate 与 Workload Cutover

## 基本信息

- **文档角色**: `Task 3` 的 runtime/build/codegen gate 与 workload cutover 文档
- **当前状态**: `2026-04-16` 活动设计文档
- **任务链位置**: `Task 1/2` owner cutover 之后的 leaf 收口与 support surface 承接
- **唯一总体设计**: `tasks/dev_design/final_blackhole_backend_redesign.md`

## 1. 目标

`Task 3` 只负责两件事：

1. 让 build / codegen / runtime
   严格消费 typed owner truth
2. 在新主链上重新承接 workload payoff

`Task 3`
不重新定义 `SpatialPlan` 或 `TTProgram`。

## 2. Reader 纪律

长期 reader 纪律固定为：

- build / codegen / runtime / `BlackholeModule`
  只允许读 `ExecutableSpec`
- `ExecutableSpec`
  只允许投影自 `TTProgram`

明确禁止：

- 读审计表列出的 legacy gate attrs / transition attrs
- 从 `work_linear_id`、arg kind、payload bag、
  builtin 序列恢复 planning truth

## 3. 当前 closure set

当前代码 closure set 仍固定为：

1. `R0.1`
   - buffer effect / use-role analysis
2. `R0.2`
   - buffer liveness analysis
3. `R0.3`
   - materialization / source-live-form planner decision
4. `R1.1`
   - 去掉 build/codegen/executable extraction
     对 legacy gate attrs 的依赖
5. `R2.1`
   - 显式化 `PlanTTSync / PlanTTABI / PlanTTExecution`

其中：

- `R0`
  收口 compute / memory-access owner
- `R1`
  收口 leaf reader / writer 边界
- `R2`
  收口 communication owner / runtime semantics

## 4. Support surface 纪律

support surface 扩张只能经由：

- `SpatialPlan`
  的显式 virtual truth
- `TTProgram`
  的显式 physical truth
- `ExecutableSpec`
  的显式 leaf projection

缺 truth 时只能：

- 补 analysis
- 补 schema
- 补更早层 planner
- 显式 unsupported

不允许：

- 新增 workaround attr
- 新增 runtime fallback
- 新增 late matcher

## 5. 旧链退场顺序

退场顺序固定为：

1. 先让上游 owner truth 站稳
2. 再切 leaf readers
3. 再删 fake protocol
4. 最后删 helper bridge residue

当前明确要删的读取面：

- 审计表列出的 legacy transition attrs / internal bridge payload

## 6. 当前执行重点

`Task 3`
当前不以扩 workload family 为主，
而是先把 leaf gate 收口。

当前优先级：

1. build/codegen/runtime
   停止读取 legacy gate attrs
2. `PlanTTSync / PlanTTABI / PlanTTExecution`
   独立站成 owner pass
3. 在此基础上再恢复
   `flash-attn` payoff 与 wider family
