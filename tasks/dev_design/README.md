# 任务开发设计文档

> 当前唯一总体设计文档: `final_blackhole_backend_redesign.md`

> 当前协议面删除/迁移表:
> `blackhole_first_principles_protocol_audit.md`

## 1. 当前入口

当前入口顺序固定为：

1. `final_blackhole_backend_redesign.md`
2. `task0_ir_layering_root_cause.md`
3. `task1_spatial_plan_companion.md`
4. `task2_ttprogram_companion_cutover.md`
5. `task3_runtime_gate_and_workload_cutover.md`
6. `tasks/progress.md`
7. `2026-04-23-blackhole-live-form-materialization-admission.md`
8. `2026-04-27-blackhole-tile-compute-preservation.md`
9. `2026-04-27-blackhole-post-preservation-pass-shrink.md`
10. `2026-04-28-blackhole-lower-tile-op-normalizer-dedup.md`
11. `2026-04-28-blackhole-algorithmic-generalization.md`
12. `2026-04-28-blackhole-tile-compute-legalizer-dag-covering.md`

当前 support surface / workload payoff lane
的任务级设计固定为：

- `2026-04-23-blackhole-live-form-materialization-admission.md`
  - 只定义 direct cast /
    `fragment_fill -> cast -> publish`
    /
    flash-attn direct runtime admission
    的 live-form / materialization
    representation contract
  - 不替代总体设计，
    不引入新的长期 IR 层

当前 Blackhole tile compute preservation lane
的任务级设计记录为：

- `2026-04-27-blackhole-tile-compute-preservation.md`
  - 定义 TT-Metal API 粒度 tile compute semantics
    在 `Normalized Tile TIR`
    中被保留 / 规范化的合同
  - 覆盖 matmul / reduce / unary / binary /
    broadcast / copy / pack /
    tilize / untilize
    等通用 tile compute API 粒度，
    不是 reduce 或 flash-attn 专项设计
  - 明确 P2.2/P2.3
    late scalar-loop idiom recovery
    已作为本 lane 清理目标删除，
    后续不能重新作为 guard /
    wrapper /
    compatibility shell 引入

当前 post-preservation pass shrink lane
的任务级设计记录为：

- `2026-04-27-blackhole-post-preservation-pass-shrink.md`
  - 定义 tile-compute preservation 完成后
    `lower_blackhole_ops.cc`
    的责任缩小边界
  - 只允许 implementation split /
    pass-local helper reuse，
    不引入新的长期 IR 层或 side-channel
  - 记录其他 heavy mixed-responsibility pass
    的后续候选，
    尤其是 `lower_tile_op.cc`
    中重复的 Blackhole tile compute normalization

当前 `lower_tile_op.cc` cleanup
的任务级设计记录为：

- `2026-04-28-blackhole-lower-tile-op-normalizer-dedup.md`
  - 定义 `LowerTileOpPass`
    和 `BlackholeTileComputeNormalizer`
    共享同一个 Blackhole tile compute normalizer
    implementation surface
  - 继续产出显式
    `tl.tileop.blackhole_compute`
    和 TT-Metal leaf API 粒度 operation name
  - 不引入新的 IR 层、
    downstream matcher
    或跨阶段 side-channel

当前 Blackhole algorithmic generalization
的任务级设计记录为：

- `2026-04-28-blackhole-algorithmic-generalization.md`
  - 定义
    `AccessRegion` /
    graph-backed `SpatialPlan` dependence model /
    `LiveValueSSA` /
    TT live-form solver /
    decision-use cutover
    的重构设计
  - 借鉴 LLVM Dependence Graph /
    MemorySSA /
    VPlan /
    LoopAccessAnalysis
    和 MLIR affine/dataflow/Linalg
    的算法骨架，
    但不引入新的长期 IR 层
  - Phase A-D
    foundation
    和 Phase E decision-use cutover
    已完成；
    当前活动 lane 是
    `Tile compute legalizer / DAG covering Phase C-D`
  - 强制执行
    anti-overdesign pay-rent rule
    和 problem-family generality rule：
    typed evidence 必须改变 legality /
    query /
    typed plan /
    unsupported diagnostic，
    当前 workload case
    只能作为 witness，
    不能成为协议定义

当前 Blackhole tile compute legalizer /
DAG covering
的任务级设计记录为：

- `2026-04-28-blackhole-tile-compute-legalizer-dag-covering.md`
  - 定义
    `TileComputeDAG`
    /
    legalization
    /
    target leaf pattern covering
    的 selection 架构
  - 对齐 LLVM SelectionDAG /
    TableGen pattern selection
    的思想，
    但第一版只用 C++ typed schema
  - 目标是让新增 TT-Metal leaf compute op
    变成 pattern + legality + tests，
    而不是新增手写 per-op branch family
  - Algorithmic Generalization
    Phase E 的 decision-use gate
    已对 admitted compute surface 生效；
    Phase A-B foundation
    已完成；
    Phase C-D production migration
    已完成：
    covering selection gates
    `TTComputeOpPlan`
    recording、
    explicit tile-compute source dispatch
    和
    `ValidateTTProgram`；
    typed
    `TileComputeDAG`
    builder
    connects producer-use edges by IR object identity；
    DAG covering reports selected patterns、
    cost、
    fanout decisions、
    materialization policy
    和 stale-fallback rejection；
    remaining emitter branch deletion
    是后续 Phase E cleanup
  - read-only DAG dump /
    pattern table /
    generic covering class
    只算 foundation work；
    只有影响 typed plans /
    unsupported diagnostics
    或删除旧 per-op branch
    才算 production completion

当前执行顺序不在 README 中重复维护；
唯一状态看板是 `tasks/progress.md`。
截至当前 repo HEAD，
已完成的 foundation 包括
`AccessRegion`、
graph-backed `SpatialPlan` dependence、
`LiveValueSSA`、
第一版 TT live-form solver，
以及 Phase E decision-use cutover。
当前已完成的 production migration 包括
`Tile compute legalizer / DAG covering Phase C-D`。
后续顺序是
tile compute covering Phase E emitter cleanup、
multi-block flash-attn direct-runtime admission、
multi-page exact-CB event admission、
mesh/distributed runtime admission
和 wider flash-attn workload-scale admission。

额外参考：

- `archive/layered_ir_references.md`
  - 研究输入和方法论参考，不是当前活动设计入口
- `blackhole_first_principles_protocol_audit.md`
  - 历史 surface 的删除/迁移落点表
- `archive/2026-04-16-blackhole-final-legacy-protocol-cleanup.md`
  - 已完成 cleanup 的历史边界索引；
    配套 task0-task5 分文件也在
    `archive/`
  - 只作历史参考，
    不是当前活动设计入口

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
| `task2_ttprogram_companion_cutover.md` | `TTProgram` 合同文档；定义 target realization 的显式 slice、mesh/buffer distribution、reader/writer 边界与完成判据 |
| `task3_runtime_gate_and_workload_cutover.md` | `ExecutableSpec / leaf reader` 合同文档；定义 leaf reader 纪律、direct runtime 与 codegen/export backend 分离、workload 恢复顺序与完成判据 |
| `2026-04-23-blackhole-live-form-materialization-admission.md` | cleanup 之后 support surface lane 的任务级设计；定义 live-form / materialization owner truth、admission 顺序和禁止的 runtime-only patch |
| `2026-04-27-blackhole-tile-compute-preservation.md` | Blackhole tile compute preservation lane 的完成记录；定义 TT-Metal API 粒度 compute semantics 在 `Normalized Tile TIR` 的保留 / 规范化边界和 late matcher 删除边界 |
| `2026-04-27-blackhole-post-preservation-pass-shrink.md` | tile-compute preservation 之后的实现瘦身 lane；定义 `lower_blackhole_ops.cc` 拆分边界、helper 复用纪律和后续 heavy pass audit 候选 |
| `2026-04-28-blackhole-lower-tile-op-normalizer-dedup.md` | `lower_tile_op.cc` cleanup 任务设计；定义 Blackhole tile compute normalization 的单一实现面和验证边界 |
| `2026-04-28-blackhole-algorithmic-generalization.md` | Blackhole passes 算法化重构设计；定义 AccessRegion、SpatialPlan dependence graph、LiveValueSSA、TT live-form solver、Phase E decision-use cutover，以及 anti-overdesign / problem-family guardrails |
| `2026-04-28-blackhole-tile-compute-legalizer-dag-covering.md` | Blackhole tile compute selection 算法化设计；定义 TileComputeDAG、legalizer、leaf pattern covering、cost model 和迁移边界；production covering 受 Phase E decision-use gate 约束 |
| `blackhole_first_principles_protocol_audit.md` | 删除/迁移表；列出 historical fake/legacy protocol 的表示层落点、validator 和 disposition |

### Runtime / mesh / distributed 调研索引

runtime 重构和 TT-Metal mesh/distributed API
调研结论不单独另立总体设计文档，
统一落在下面几处：

- `final_blackhole_backend_redesign.md`
  - 固定 direct runtime
    只是 `ExecutableSpec`
    的 leaf execution backend；
    当前 unit-mesh /
    replicated-buffer /
    admitted subset
    不能作为 codegen /
    export /
    `TTProgram`
    的能力上限
  - 定义 `TTMeshPlan` /
    `TTBufferDistributionPlan`
    作为 `TTProgram`
    的显式表示对象
- `task0_ir_layering_root_cause.md`
  - 记录根因：
    multi-device /
    distributed /
    mesh /
    fabric
    语义不能藏在 runtime fallback
    或 host-side recovery 里
- `task1_spatial_plan_companion.md`
  - 记录 logical mesh axis、
    distributed-slice consumer
    和 live value 边界
- `task2_ttprogram_companion_cutover.md`
  - 记录 TT-Metal
    `MeshDevice` /
    `MeshWorkload` /
    `MeshBuffer`
    对应的 physical mesh、
    buffer distribution、
    device-coordinate
    schema
- `task3_runtime_gate_and_workload_cutover.md`
  - 记录 direct runtime
    与 codegen/export
    的边界：
    `BlackholeModule`
    当前可以用 TT-Metal distributed API
    执行 unit mesh /
    replicated `MeshBuffer`
    admitted subset，
    但 distributed /
    mesh /
    fabric
    未 admission 时只能是该 backend
    fail-closed unsupported
- 历史调研和开发记录只作辅助参考：
  `blackhole_first_principles_protocol_audit.md`、
  `archive/layered_ir_references.md`、
  `archive/stage3_multicore_design.md`、
  `archive/stage2_concrete_dev_task_plan.md`、
  `archive/task2_task3_tt_target_cutover.md`、
  `memory/general_dev.md`

当前结论：

- Blackhole runtime 主路径已经收敛到
  `BlackholeModule`
  进程内 direct host path；
  legacy external runner
  不再是支持路径
- 当前 direct runtime
  只 admission
  unit mesh /
  replicated `MeshBuffer`
  /
  已验证 workload subset；
  这不等于完整 multi-device /
  fabric collective /
  distributed runtime
  已支持
- 完整 mesh/distributed 能力必须继续通过
  `TTProgram -> ExecutableSpec`
  的 typed schema
  和 leaf admission
  扩展，
  不能回到 runtime-only patch
  或隐式 payload 通道

补充说明：

- `task1_spatial_plan_companion.md`
- `task2_ttprogram_companion_cutover.md`

这两个文件名里的 `companion`
只是历史文件名，
不是新的 IR 层命名。

## 4. 当前执行优先级

当前 repo HEAD 的总体状态 /
当前 blocker /
当前下一步，
仍统一只看 `tasks/progress.md`。

历史 cleanup 主线按下面这条理解；
这不是当前 active backlog：

`Task 1 -> Task 2 -> Task 3 -> Legacy Protocol Deletion`

这里不再重复维护当前 repo HEAD
的阶段队列。

- `tasks/progress.md`
  - 唯一当前执行顺序 / 状态看板
- `task1_spatial_plan_companion.md`
- `task2_ttprogram_companion_cutover.md`
- `task3_runtime_gate_and_workload_cutover.md`
  - 主设计路线和 completion contract
- cleanup `task0-task5`
  已归档到
  `archive/2026-04-16-blackhole-final-legacy-protocol-cleanup*.md`；
  它们只作完成期历史记录，
  不再参与当前执行优先级

其中：

- `buffer effect / use-role`
- `liveness`
- `materialization / source-live-form`

曾经只是前置分析/局部构造子问题；
cleanup 收口后，
`materialization / source-live-form`
已重新收束为
`2026-04-23-blackhole-live-form-materialization-admission.md`
里的 support surface admission
任务级设计，
仍不单独充当新的顶层路线。
当前执行顺序只在
`tasks/progress.md`
维护；
README 不再重复维护 backlog。

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
  定义主设计路线 /
  目标合同 /
  完成判据，
  不维护 repo HEAD 的阶段性状态快照
- cleanup `task0-task5`
  分文件已归档；
  archive 里的完成期状态 /
  grep 合同 /
  residue 表述
  不替代 `progress.md`
  作为当前状态来源
- `README`
  只做入口索引，不重复维护详细 backlog
- `archive/`
  下全部文档只作历史参考
- `tasks/dev_design/`
  根目录只保留当前入口文档、
  当前活动 / 已完成但仍约束实现的 lane 设计文档
  和 protocol audit；
  方法论 / 研究输入 /
  已完成阶段边界 /
  旧阶段设计全部放在
  `archive/`

## 6. Archive

查看 `archive/README.md`。
