# 任务开发设计文档

> 当前唯一总体设计文档: `final_blackhole_backend_redesign.md`

> 入口规则: 根目录只保留当前需要持续引用的文档；`archive/` 下的文档一律视为历史记录、完成记录，或已被后续设计取代。

## 1. 当前实施入口

当前工程推进默认从下面三份文档进入：

| 文档 | 角色 | 状态 |
|------|------|------|
| `final_blackhole_backend_redesign.md` | 唯一总体设计 | 常青 |
| `stage4_phase_b_spatial_ir.md` | 当前主实施文档；承接 `Spatial Program IR` 详细设计与实施 | 当前进行中 |
| `stage4_phase_c_tt_target_ir.md` | `Phase B` 之后的下一阶段实施文档；承接 `TT Target IR` 详细设计、cutover 与 deletion gates | 已定义，待推进 |

## 2. 当前阶段文档

| 文档 | 角色 | 状态 |
|------|------|------|
| `stage4_stage0_guardrails.md` | Stage 0 护栏与 cutover 前提 | 已落地；作为边界参考保留 |
| `stage4_phase_a_semantic_ir.md` | `Phase A` 工程边界、已落地对象与 `Phase B` 输入约束 | 已完成；作为实现参考保留 |
| `stage4_phase_a_formalization_note.md` | `Phase A` 理论化 / 证明化并行文档 | 并行 research track；非工程 blocker |
| `stage4_phase_b_spatial_ir.md` | `Spatial Program IR` 实施文档 | 当前进行中 |
| `stage4_phase_c_tt_target_ir.md` | `TT Target IR`、cutover、family expansion | 下一阶段 |

## 3. 支撑与审计文档

| 文档 | 角色 | 状态 |
|------|------|------|
| `review_final_blackhole_backend_redesign.md` | 总设计的系统性 review 与代码交叉审计记录 | 已完成；作为审计参考保留 |

## 4. 当前任务安排

当前默认执行顺序：

1. `final_blackhole_backend_redesign.md`
2. `stage4_phase_b_spatial_ir.md`
3. `stage4_phase_c_tt_target_ir.md`

在需要回看前置边界时，再引用：

- `stage4_stage0_guardrails.md`
- `stage4_phase_a_semantic_ir.md`

只有在做理论化 / refinement validator 研究项时，才默认进入：

- `stage4_phase_a_formalization_note.md`

## 5. 清理规则

- 根目录不再保留额外的总 implementation plan。
- `final_blackhole_backend_redesign.md` 现在保持为轻量总纲；阶段细节默认下沉到对应 stage 文档。
- 已完成专项设计、历史 implementation plan、早期阶段记录、legacy 架构说明，全部移动到 `archive/`。
- `archive/README.md` 说明了归档区的使用规则。

## 6. Archive

查看 `archive/README.md`。
