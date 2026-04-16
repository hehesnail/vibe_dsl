# Task 3: ExecutableSpec / Leaf Reader Cutover

## 基本信息

- **文档角色**: `Task 3: ExecutableSpec / Leaf Reader Cutover` 文档
- **当前状态**: `2026-04-16` 已完成（repo HEAD）
- **任务链位置**:
  `Task 1: SpatialPlan Owner Cutover`
  和
  `Task 2: TTProgram Owner Cutover`
  之后的 leaf 收口与 support surface 承接
- **唯一总体设计**: `tasks/dev_design/final_blackhole_backend_redesign.md`

## 1. 目标

`Task 3` 只负责两件事：

1. 让 build / codegen / runtime
   严格消费 canonical owner truth
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

## 3. 前置条件

`Task 3`
只依赖前两层 Owner Cutover，
不单独维护额外 roadmap 编号。

它的前置条件固定为：

1. `Task 1: SpatialPlan Owner Cutover`
   已经把 virtual truth
   对象化并可验证
2. `Task 2: TTProgram Owner Cutover`
   已经把 target owner
   显式收回
3. leaf readers
   可以直接读
   `TTProgram / ExecutableSpec`

因此：

- `buffer effect / use-role`
- `liveness`
- `materialization / source-live-form`

这些工作属于前两层 Owner Cutover，
不是 `Task 3`
自己的顶层路线

## 4. Support surface 纪律

support surface 扩张只能经由：

- `SpatialPlan`
  的显式 virtual truth
- `TTProgram`
  的显式 physical truth
- `ExecutableSpec`
  的显式 leaf projection

缺 truth 时只能：

- 补更早层 IR / owner object
- 补更早层 planner 的构造/改写逻辑
- 补 validator
- 显式 unsupported

不允许：

- 新增 workaround attr
- 新增 runtime fallback
- 新增 late matcher
- 新增 leaf-time visitor/matcher 去恢复 planning truth

## 5. 旧链退场顺序

退场顺序固定为：

1. 先让上游 owner truth 站稳
2. 再切 leaf readers
3. 再删 fake protocol
4. 最后删 helper bridge residue

当前明确要删的读取面：

- 审计表列出的 legacy transition attrs / internal bridge payload

## 6. 当前执行重点

repo HEAD 当前已经完成的 leaf cutover：

1. build / codegen / runtime / `BlackholeModule`
   只读
   `tl.blackhole_executable`
   与其内部
   `ExecutableSpec`
   投影
2. leaf-only build contracts
   已从
   `blackhole.lowering_requirements`
   上提到
   `TTProgram.payload`
   并经
   `MaterializeBlackholeExecutable`
   投影到
   `tl.blackhole_executable`
3. synthetic segment materialization
   只重建
   segment-local executable
   视图，
   不再给内部 leaf func
   回挂最小 `TTProgram`

## 7. 完成判定

`Task 3`
只有在下面这些条件同时满足后才算完成：

1. build / codegen / runtime / `BlackholeModule`
   只读 `ExecutableSpec`
2. `ExecutableSpec`
   只投影自 `TTProgram`
3. leaf readers
   不再读取审计表列出的
   legacy gate attrs / transition attrs / internal payload
4. support surface 扩张
   已经回到
   `SpatialPlan -> TTProgram -> ExecutableSpec`
   这条主链上进行

repo HEAD 对上述四项完成判定都已满足。

## 8. Workload 承接顺序

workload payoff
只能在
`Task 3: ExecutableSpec / Leaf Reader Cutover`
之后按下面顺序恢复：

1. `flash-attn`
   compile/runtime payoff
2. wider family cutover
3. wider support surface

在这之前，
不允许把 workload payoff
重新写成 Owner Cutover 的 blocker。
