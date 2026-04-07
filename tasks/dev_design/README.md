# 任务开发设计文档

> 当前唯一总体设计文档: `final_blackhole_backend_redesign.md`

> 入口规则: 根目录只保留当前需要持续引用的文档；`archive/` 下的内容一律视为历史记录，不再作为当前任务安排入口。

## 1. 当前入口

默认按下面顺序进入：

1. `final_blackhole_backend_redesign.md`
2. `stage4_phase_c_tt_target_ir.md`
3. `stage4_phase_b_spatial_ir.md`
4. `tasks/progress.md`

## 2. 当前活动文档

| 文档 | 角色 | 当前状态 |
|------|------|----------|
| `final_blackhole_backend_redesign.md` | 唯一总体设计 | 常青总纲 |
| `stage4_phase_b_spatial_ir.md` | `Spatial Program IR` 已完成阶段文档 | 已完成；作为 `Phase C` 输入边界保留 |
| `stage4_phase_c_tt_target_ir.md` | `TT Target IR` 当前设计与 cutover 文档 | 准备轨已完成；当前主实施入口 |
| `stage4_phase_a_semantic_ir.md` | `Phase A` 已落地语义边界 | 已完成；作为 `Phase B` 输入参考保留 |
| `stage4_semantic_manifest.md` | `Phase A` 信息源边界 | 已完成；作为 ownership 参考保留 |

## 3. 参考文档

按需要再读：

- `layered_ir_references.md`
  - 分层 IR 设计的研究参考与跨论文启发；仅作设计输入，不承担协议真源职责
- `stage4_stage0_guardrails.md`
  - Stage 0 护栏与 cutover 前提
- `stage4_phase_a_formalization_note.md`
  - `Phase A` 并行理论化 / 证明化说明
- `review_final_blackhole_backend_redesign.md`
  - 总设计审计快照，不承担当前状态跟踪

## 4. 清理规则

- 总纲只保留长期架构、层间边界、真源规则与阶段判断
- 阶段细节、完成条件和基线命令下沉到对应阶段文档
- 进度、验证摘要与下一步统一放在 `tasks/progress.md`
- 已完成专项设计、历史实现计划和 legacy 架构说明全部放入 `archive/`

## 5. Archive

查看 `archive/README.md`。
