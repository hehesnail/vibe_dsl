# 任务开发设计文档

> 当前唯一总体设计文档: `final_blackhole_backend_redesign.md`

> 当前协议面审计与 disposition table:
> `blackhole_first_principles_protocol_audit.md`

## 1. 当前入口

当前入口顺序固定为：

1. `final_blackhole_backend_redesign.md`
2. `tasks/progress.md`
3. `task0_ir_layering_root_cause.md`
4. `task1_spatial_plan_companion.md`
5. `task2_ttprogram_companion_cutover.md`
6. `task3_runtime_gate_and_workload_cutover.md`

额外参考：

- `layered_ir_references.md`
  - 研究输入，不是协议真源
- `blackhole_first_principles_protocol_audit.md`
  - 现存 surface 的 owner / non-owner / validator / cutover disposition

## 2. 当前设计骨架

当前唯一长期主链是：

```text
Normalized Tile TIR
  -> SpatialPlan
  -> TTProgram
  -> ExecutableSpec
```

其中：

- `Normalized Tile TIR`
  - 持有算法与访存真相
- `SpatialPlan`
  - 持有 target-independent 的 virtual spatial/dataflow truth
- `TTProgram`
  - 持有 TT-specific physical realization truth
- `ExecutableSpec`
  - 只做 leaf projection 和执行物化

## 3. 当前活动文档

| 文档 | 角色 |
|------|------|
| `final_blackhole_backend_redesign.md` | 唯一总体设计；定义长期层边界、validator 纪律、rewrite 方向 |
| `task0_ir_layering_root_cause.md` | 根因诊断与研究结论；解释为什么必须把中间 virtual layer 立起来 |
| `task1_spatial_plan_companion.md` | `SpatialPlan` 设计；定义 virtual spatial/dataflow program 的 object set 与 validator |
| `task2_ttprogram_companion_cutover.md` | `TTProgram` 设计；定义 physical realization owner objects、planner pass 与 reader/writer 边界 |
| `task3_runtime_gate_and_workload_cutover.md` | runtime/codegen/build gate、support surface 和 legacy attr 退场顺序 |
| `blackhole_first_principles_protocol_audit.md` | disposition table；列出现存 fake/legacy protocol 的去留与迁移落点 |

## 4. 当前执行优先级

不再按旧 `R0 / R1 / R2`
编号阅读当前 roadmap。

当前只按层 owner cutover 排序：

1. `SpatialPlan owner cutover`
2. `TTProgram owner cutover`
3. `ExecutableSpec / leaf reader cutover`
4. `legacy protocol deletion`

其中：

- `buffer effect / use-role`
- `liveness`
- `materialization / source-live-form`

都只是
`SpatialPlan owner cutover`
里的子问题，
不再单独充当顶层路线。

## 5. 清理规则

- 不再把历史层名词、legacy transition attrs
  或 bridge attr 写成长期协议
- 不新增第二份总体设计文档
- `progress.md`
  只维护当前真实代码状态与下一步
- `README`
  只做入口索引，不重复维护详细 backlog
- `archive/`
  下全部文档只作历史参考

## 6. Archive

查看 `archive/README.md`。
