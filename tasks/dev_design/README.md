# 任务开发设计文档

> 本目录只维护设计入口和任务级合同。
> 当前 repo HEAD 状态、blocker、下一步只看
> `../progress.md`。

## Entry Order

1. `final_blackhole_backend_redesign.md`
2. `task0_ir_layering_root_cause.md`
3. `task1_spatial_plan_companion.md`
4. `task2_ttprogram_companion_cutover.md`
5. `task3_runtime_gate_and_workload_cutover.md`
6. `../progress.md`
7. Current activity / support-lane docs listed below

## Core Contracts

| Document | Role |
| --- | --- |
| `final_blackhole_backend_redesign.md` | 唯一总体设计；定义长期 IR 主链、层边界、validator 纪律、fake protocol 删除规则、hardware-codegen usefulness gate。 |
| `task0_ir_layering_root_cause.md` | 根因诊断；解释为什么必须以显式 IR 主链替代 late matcher / bag / payload。 |
| `task1_spatial_plan_companion.md` | `SpatialPlan` 表示层合同；定义 target-independent virtual spatial/dataflow program。 |
| `task2_ttprogram_companion_cutover.md` | `TTProgram` 表示层合同；定义 TT-specific target realization。 |
| `task3_runtime_gate_and_workload_cutover.md` | `ExecutableSpec` / leaf reader 合同；定义 leaf projection、backend admission、runtime-module build 边界。 |

## Current Lane Docs

| Document | Role |
| --- | --- |
| `2026-04-23-blackhole-live-form-materialization-admission.md` | Live-form / materialization support surface；direct cast、`fragment_fill -> cast -> publish`、flash-attn runtime admission 合同。 |
| `2026-04-27-blackhole-tile-compute-preservation.md` | Tile compute preservation；要求 TT-Metal API 粒度 compute semantics 在 `Normalized Tile TIR` 中保留或规范化。 |
| `2026-04-27-blackhole-post-preservation-pass-shrink.md` | Post-preservation pass shrink；约束 `lower_blackhole_ops.cc` 拆分、helper 复用和 heavy-pass cleanup。 |
| `2026-04-28-blackhole-lower-tile-op-normalizer-dedup.md` | `lower_tile_op.cc` normalizer 边界；定义 explicit leaf tile-compute normalization。 |
| `2026-04-28-blackhole-algorithmic-generalization.md` | `AccessRegion` / dependence graph / `LiveValueSSA` / TT live-form solver 合同。 |
| `2026-04-28-blackhole-tile-compute-legalizer-dag-covering.md` | `TileComputeDAG` / legalizer / leaf pattern covering 合同。 |
| `2026-04-29-blackhole-resource-planning-roadmap.md` | Resource planning roadmap；定义 CB/L1 admission、core placement、buffer distribution、later NoC work 的依赖关系。 |
| `blackhole_first_principles_protocol_audit.md` | Historical fake/legacy protocol 删除/迁移表。 |

## Stable Architecture Skeleton

长期主链只有：

```text
Normalized Tile TIR
  -> SpatialPlan
  -> TTProgram
  -> ExecutableSpec
```

解释：

- `Normalized Tile TIR`
  承载算法、访存、tile compute leaf semantics。
- `SpatialPlan`
  承载 target-independent virtual spatial/dataflow program。
- `TTProgram`
  承载 TT-specific physical realization。
- `ExecutableSpec`
  承载 leaf projection 和 backend admission。

Pass 名字、helper、payload、bag、bridge attr
都不是长期协议边界。

## Current Execution Priority

当前执行顺序不在 README 重复维护。
唯一看板是
`tasks/progress.md`。

截至当前状态，
下一条主线是：

```text
Hardware-model-backed core and buffer placement
  -> wider runtime admission
```

## Maintenance Rules

- 总体设计只写长期架构和不可违反的合同。
- 任务级设计只写该任务的 goal / non-goal / representation contract /
  validation contract / completion criteria。
- `progress.md`
  只写当前 HEAD 状态、blocker、下一步、最近验证。
- 经验沉淀写入
  `memory/`；
  不要倒灌回核心设计文档。
- 历史流水、阶段日志、已完成 patch notes
  不写入核心入口；
  如果确实需要保留，放入
  `archive/`
  或对应历史文档。
- 不新增第二份总体设计。
- 不把当前 implementation residue
  写成 design legitimacy。

## Archive

`archive/`
只作历史参考。
归档文档不能作为当前 active design entry，
也不能作为保留旧 wrapper /
facts /
bag /
payload /
matcher
兼容面的理由。
