# 任务开发设计文档

> 当前唯一总体设计文档: `final_blackhole_backend_redesign.md`

> 入口规则: 根目录只保留当前需要持续引用的文档；`archive/` 下的内容一律视为历史记录，不再作为当前任务安排入口。

## 1. 当前入口

按仓库当前工作规则，默认按下面顺序进入：

1. `final_blackhole_backend_redesign.md`
2. `tasks/progress.md`
3. `ir_layering_root_cause_and_direction.md`
4. `spatial_dataflow_program_model.md`
5. `stage4_phase_c_tt_target_ir.md`
6. `stage4_phase_b_spatial_ir.md`

## 2. 当前活动文档

| 文档 | 角色 | 当前状态 |
|------|------|----------|
| `final_blackhole_backend_redesign.md` | 唯一总体设计 | 已切到两层主链：`Normalized Tile TIR -> SpatialGraph -> TTProgram -> ExecutableSpec` |
| `ir_layering_root_cause_and_direction.md` | IR layering 根因诊断与整改方向 | 已形成当前架构收敛依据；保留为两层 IR redesign 的问题诊断入口 |
| `spatial_dataflow_program_model.md` | spatial/dataflow program model 细化设计 | 仍为活动文档，但后续需按 `SpatialGraph / VirtualTask / TTBlockPlan` 边界重写 |
| `stage4_phase_c_tt_target_ir.md` | 当前已落地 `TTProgram` 基线与支持面文档 | 仍为活动文档，但角色降为当前代码基线/支持面参考，不再承担总体 layering 权威 |

## 3. 已完成但仍保留的边界文档

| 文档 | 角色 | 当前状态 |
|------|------|----------|
| `stage4_stage0_guardrails.md` | `Stage 0` 护栏与 cutover 前提 | 已完成；作为 layered IR 迁移前置假设保留 |
| `stage4_phase_a_semantic_ir.md` | 旧 `SemanticProgram` 边界文档 | 已完成；现作为历史实现边界保留 |
| `stage4_semantic_manifest.md` | `Phase A` 信息源边界 | 已完成；作为 evidence ownership 参考保留 |
| `stage4_phase_b_spatial_ir.md` | 旧 `SpatialProgram` 边界文档 | 已完成；现作为历史实现边界保留，并记录旧主链审计结论 |
| `stage4_phase_a_formalization_note.md` | `Phase A` 理论化说明 | research reference；不承担当前工程入口 |

## 4. 参考文档

按需要再读：

- `layered_ir_references.md`
  - 分层 IR 设计的研究参考与跨论文启发；仅作设计输入，不承担协议真源职责

## 5. 清理规则

- 总纲只保留长期架构、层间边界、真源规则与阶段判断
- 当前代码基线、支持面和 gate 继续维护在对应阶段文档
- 新两层主链的长期 owner 以总纲为准；旧 `SemanticProgram / SpatialProgram` 阶段文档不再回升为总体权威
- 进度、验证摘要与下一步统一放在 `tasks/progress.md`
- `README` 只做索引和分工说明，不重复维护阶段 backlog
- 已完成审计快照、历史实现计划和 legacy 架构说明全部放入 `archive/`

## 6. Archive

查看 `archive/README.md`。
