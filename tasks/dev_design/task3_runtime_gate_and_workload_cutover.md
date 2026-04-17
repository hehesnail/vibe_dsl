# Task 3: ExecutableSpec / Leaf Reader Cutover

## 基本信息

- **文档角色**: `ExecutableSpec / leaf reader` 合同文档
- **任务链位置**:
  `SpatialPlan -> TTProgram -> ExecutableSpec`
  之后的 leaf 收口与 workload 承接
- **唯一总体设计**: `tasks/dev_design/final_blackhole_backend_redesign.md`

说明：

- 本文档只定义 leaf 边界和 workload 承接合同
- 它不单独声明 repo HEAD 是否已经完成
- 当前实现状态统一只看 `tasks/progress.md`

## 1. 目标

`Task 3` 只负责两件事：

1. 让 build / codegen / runtime
   严格消费 `ExecutableSpec`
2. 在这条主链上重新承接 workload payoff

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
  builtin 序列恢复 planning 语义
- 在 leaf-time 再做中间层或 target-side 语义恢复

## 3. 前置条件

`Task 3`
只依赖前两层表示合同，
不单独维护额外 roadmap 编号。

它的前置条件固定为：

1. `Task 1`
   已经把 `SpatialPlan`
   的 virtual spatial/dataflow 语义
   显式化并可验证
2. `Task 2`
   已经把 `TTProgram`
   的 target realization 表示
   显式化并可验证
3. leaf writer
   可以直接从
   `TTProgram`
   投影到
   `ExecutableSpec`

因此：

- `buffer effect / use-role`
- `liveness`
- `materialization / source-live-form`

这些工作都属于更早层的表示与构造问题，
不是 `Task 3`
自己的顶层路线。

## 4. Support Surface 纪律

support surface 扩张只能经由：

- `SpatialPlan`
  的显式 virtual spatial/dataflow 表示
- `TTProgram`
  的显式 target realization 表示
- `ExecutableSpec`
  的显式 leaf projection

缺语义时只能：

- 补更早层 IR / 显式对象
- 补更早层 planner 的构造/改写逻辑
- 补 validator
- 显式 unsupported

不允许：

- 新增 workaround attr
- 新增 runtime fallback
- 新增 late matcher
- 新增 leaf-time visitor/matcher 去恢复 planning 语义

## 5. 旧链退场顺序

退场顺序固定为：

1. 先让上游显式表示站稳
2. 再切 leaf readers
3. 再删 fake protocol
4. 最后删 helper bridge residue

当前明确要删的读取面：

- 审计表列出的 legacy transition attrs / internal bridge payload

## 6. Leaf Writer / Reader 边界

### `MaterializeBlackholeExecutable`

是唯一 leaf writer。

它负责：

- 从 `TTProgram`
  投影出
  `ExecutableSpec`
- 冻结 leaf-only build/runtime/codegen 所需记录

### build / codegen / runtime / `BlackholeModule`

只允许消费：

- `tl.blackhole_executable`
- 或其内部 `ExecutableSpec` 记录

它们不允许再回头读取：

- `TTProgram`
  内部 planning residue
- 任意 legacy attr / helper payload
- 任意 leaf-time 推导出的伪 planning 语义

## 7. Completion Contract

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

## 8. Workload 承接顺序

workload payoff
只能在 leaf reader 边界收紧后，
按下面顺序恢复：

1. `flash-attn`
   compile/runtime payoff
2. wider family cutover
3. wider support surface

在这之前，
不允许把 workload payoff
重新写成更早层表示 cutover 的 blocker。
