# 任务开发设计文档

> 本目录只维护设计入口和任务级合同。
> 当前执行状态、blocker、下一步只看
> `../progress.md`。

## Entry Order

1. `final_blackhole_backend_redesign.md`
2. `task0_ir_layering_root_cause.md`
3. `task1_spatial_plan_companion.md`
4. `task2_ttprogram_companion_cutover.md`
5. `task3_runtime_gate_and_workload_cutover.md`
6. `../progress.md`
7. Current lane docs listed below

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
| `2026-04-23-blackhole-live-form-materialization-admission.md` | Live-form / materialization support surface。 |
| `2026-04-27-blackhole-tile-compute-preservation.md` | Tile compute preservation and explicit leaf normalization boundary。 |
| `2026-04-27-blackhole-post-preservation-pass-shrink.md` | Post-preservation implementation responsibility split rules。 |
| `2026-04-28-blackhole-lower-tile-op-normalizer-dedup.md` | `LowerTileOp` / Blackhole normalizer boundary。 |
| `2026-04-28-blackhole-algorithmic-generalization.md` | `AccessRegion` / dependence graph / `LiveValueSSA` / TT live-form solver contract。 |
| `2026-04-28-blackhole-tile-compute-legalizer-dag-covering.md` | `TileComputeDAG` legalizer / covering contract。 |
| `2026-04-29-blackhole-resource-planning-roadmap.md` | Resource planning roadmap；CB/L1 admission、core placement、buffer distribution、later NoC work。 |
| `2026-05-02-blackhole-tensor-sharding-and-reshard.md` | Tensor/value sharding and explicit reshard design；separates TTNN-style `MemoryConfig` intent from low-level `TTBufferDistributionPlan` address ABI。 |
| `blackhole_first_principles_protocol_audit.md` | Historical fake/legacy protocol audit table。 |

## Stable Architecture Skeleton

```text
Normalized Tile TIR
  -> SpatialPlan
  -> TTProgram
  -> ExecutableSpec
```

Pass 名字、helper、payload、bag、bridge attr
都不是长期协议边界。

## Current Execution Priority

唯一看板是 `tasks/progress.md`。

当前主线：

```text
T1 Buffer address ABI execution integration (complete)
  -> T2 Leaf compute / GEMM baseline (active)
  -> T3 tensor/value sharding and explicit reshard
  -> T4 external accessor / runtime ABI expansion
  -> T5 sharded GEMM / layout variants
  -> T6 topk
  -> T7 exact-CB / materialization primitives
  -> T8 grouped / ragged work packets
  -> T9 workload first paths
  -> T10 production distributed variants
```

Admission levels
compile / source-spec / direct runtime / TT-Sim correctness / typed reject
are per-task acceptance gates, not a separate cleanup lane.

## Maintenance Rules

- 总体设计只写长期架构和不可违反的合同。
- 任务级设计只写 goal / non-goal / representation contract /
  validation contract / completion criteria。
- `progress.md`
  只写当前执行状态、blocker、下一步、最近验证摘要；
  不写按 HEAD 滚动的实现库存或历史流水。
- 经验沉淀写入 `memory/`；
  不要倒灌回核心设计文档。
- 历史流水、阶段日志、patch notes、完整命令矩阵
  不写入 active design docs。
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
