# 任务开发设计文档

> 当前唯一总体设计文档: `final_blackhole_backend_redesign.md`

> 入口规则: 根目录只保留当前活动文档；`archive/` 下的文档一律视为历史记录、完成记录，或已被后续设计取代。

## 1. 活动文档

| 文档 | 用途 | 状态 |
|------|------|------|
| `final_blackhole_backend_redesign.md` | 唯一总体设计 | 常青 |
| `review_final_blackhole_backend_redesign.md` | 总设计的系统性 review（含代码交叉审计） | 已完成；设计已基于 review 修订 |
| `stage4_flash_attention_forward_subset.md` | `flash-attn` 这一类 consumer 的支持设计 | 活动中（consumer-specific，不再定义总体架构方向） |
| `stage2d_ttmetal_contract_audit.md` | TT-Metal contract 与 execution surface 缺口审计 | 支持中（TT Target IR 的 supporting audit） |
| `stage4_semaphore_schema.md` | TT Target IR 的 semaphore/sync 支持设计 | 支持中 |

## 2. 当前清理结果

- 根目录只保留当前活动文档。
- 已完成专项设计、历史 implementation plan、早期阶段记录、legacy 架构说明，全部移动到 `archive/`。
- `archive/README.md` 说明了归档区的使用规则。

## 3. 当前任务安排

当前没有活动中的旧 `stateful_tiled_ir` 实施计划。下一份实施计划应直接基于：

- `final_blackhole_backend_redesign.md`
- 其中 Phase A 的第一层 semantic core：`Domain / State / Update`，以及 `AccessMap / UpdateLaw`
- 其 Phase A / B / C 迁移顺序
- 其 workload family 覆盖矩阵
- 当前仍活动的 supporting docs

已归档、不可再作为当前任务安排入口：

- `archive/2026-04-02-stateful-tiled-ir-phase1-implementation-plan.md`
- 其他所有历史 implementation plan

## 4. Archive

查看 `archive/README.md`。
