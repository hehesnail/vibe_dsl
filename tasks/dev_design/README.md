# 任务开发设计文档

> 当前唯一总体设计文档: `final_blackhole_backend_redesign.md`

> 当前协议面删除/迁移表:
> `blackhole_first_principles_protocol_audit.md`

## 1. 当前入口

当前入口顺序固定为：

1. `final_blackhole_backend_redesign.md`
2. `tasks/progress.md`
3. `2026-04-16-blackhole-final-legacy-protocol-cleanup.md`
4. `task0_ir_layering_root_cause.md`
5. `task1_spatial_plan_companion.md`
6. `task2_ttprogram_companion_cutover.md`
7. `task3_runtime_gate_and_workload_cutover.md`

额外参考：

- `layered_ir_references.md`
  - 研究输入和方法论参考，不是当前活动设计入口
- `blackhole_first_principles_protocol_audit.md`
  - 历史 surface 的删除/迁移落点表
- `2026-04-16-blackhole-final-legacy-protocol-cleanup.md`
  - cleanup 执行总览；配套 task0-task5 分文件一起阅读
  - 这些 cleanup task 文档默认采用 current IR/current object 上的
    visitor / matcher / mutator / builder 设计，
    不再授权新增 bag / attr / wrapper 式跨阶段语义层

## 2. 当前长期设计骨架

当前唯一长期主链是：

```text
Normalized Tile TIR
  -> SpatialPlan
  -> TTProgram
  -> ExecutableSpec
```

其中：

- `Normalized Tile TIR`
  - 承载算法与访存语义
- `SpatialPlan`
  - 承载 target-independent 的 virtual spatial/dataflow 表示
- `TTProgram`
  - 承载 TT-specific physical realization 表示
- `ExecutableSpec`
  - 只做 leaf projection 和执行物化

当前活动协议只以上面四层显式表示为准。
pass 名字、helper、bag、payload、bridge attr
都不是长期协议边界。

## 3. 当前活动文档

| 文档 | 角色 |
|------|------|
| `final_blackhole_backend_redesign.md` | 唯一总体设计；定义长期层边界、validator 纪律、rewrite 方向 |
| `task0_ir_layering_root_cause.md` | 根因诊断与 IR-first 纪律基线；解释为什么必须立起中间显式表示层 |
| `task1_spatial_plan_companion.md` | `SpatialPlan` 合同文档；定义这一层的显式对象、validator、construction/lowering 边界 |
| `task2_ttprogram_companion_cutover.md` | `TTProgram` 合同文档；定义 target realization 的显式 slice、reader/writer 边界与完成判据 |
| `task3_runtime_gate_and_workload_cutover.md` | `ExecutableSpec / leaf reader` 合同文档；定义 leaf reader 纪律、workload 恢复顺序与完成判据 |
| `blackhole_first_principles_protocol_audit.md` | 删除/迁移表；列出现存 fake/legacy protocol 的表示层落点、validator 和 disposition |
| `2026-04-16-blackhole-final-legacy-protocol-cleanup.md` | cleanup 执行总览；把 legacy protocol cleanup 拆成总览 + task0-task5 分文件 |

补充说明：

- `task1_spatial_plan_companion.md`
- `task2_ttprogram_companion_cutover.md`

这两个文件名里的 `companion`
只是历史文件名，
不是新的 IR 层命名。

## 4. 当前执行优先级

当前执行顺序不再直接等同于
`Task 1 / Task 2 / Task 3 / Legacy Protocol Deletion`
这组历史任务标签。

cleanup 的**架构依赖顺序**
固定按 cleanup 文档理解：

1. `2026-04-16-blackhole-final-legacy-protocol-cleanup-task0.md`
2. `2026-04-16-blackhole-final-legacy-protocol-cleanup-task1.md`
3. `2026-04-16-blackhole-final-legacy-protocol-cleanup-task2.md`
4. `2026-04-16-blackhole-final-legacy-protocol-cleanup-task3.md`
5. `2026-04-16-blackhole-final-legacy-protocol-cleanup-task4.md`
6. `2026-04-16-blackhole-final-legacy-protocol-cleanup-task5.md`
7. cleanup 完成后，再恢复 support surface / workload payoff 扩展

这里必须和
`tasks/progress.md`
分开理解：

- 这里维护的是 cleanup 的依赖顺序
- `progress.md`
  维护的是 repo HEAD 当前 blocker
  和下一步
- `task0`
  出现在依赖顺序里，
  不等于 repo HEAD
  已按完成口径收口；
  当前只有 selector-forwarding
  局部结果，
  不能折算成完成声明

其中：

- `task1_spatial_plan_companion.md`
- `task2_ttprogram_companion_cutover.md`
- `task3_runtime_gate_and_workload_cutover.md`

负责定义表示层边界和完成判据，
不单独充当“当前代码已经完成到哪一步”的状态来源。

当前 repo HEAD 的总体状态 /
当前 blocker /
当前下一步，
统一只看 `tasks/progress.md`。

其中：

- `buffer effect / use-role`
- `liveness`
- `materialization / source-live-form`

都只是前置分析/局部构造子问题，
不单独充当顶层路线。

## 5. 文档维护规则

- 不再把历史层名词、legacy transition attrs
  或 bridge attr 写成长期协议
- 不新增第二份总体设计文档
- `progress.md`
  只维护 repo HEAD 的总体状态 /
  当前 blocker /
  当前下一步
- `task1/task2/task3`
  这组表示层合同文档
  只定义目标合同和完成判据，
  不维护 repo HEAD 的阶段性状态快照
- cleanup `task0-task5`
  分文件
  可以记录各自 residue 的
  per-task current-state evidence /
  required end-state /
  verification contract，
  但不替代 `progress.md`
  作为总体状态来源
- `README`
  只做入口索引，不重复维护详细 backlog
- `archive/`
  下全部文档只作历史参考

## 6. Archive

查看 `archive/README.md`。
