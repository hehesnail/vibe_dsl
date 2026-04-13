# 任务开发设计文档

> 当前唯一总体设计文档: `final_blackhole_backend_redesign.md`

> 入口规则: 根目录只保留当前需要持续引用的文档；`archive/` 下的内容一律视为历史记录，不再作为当前任务安排入口。

## 1. 当前入口

按仓库当前工作规则，默认按下面顺序进入：

1. `final_blackhole_backend_redesign.md`
2. `tasks/progress.md`
3. `task0_ir_layering_root_cause.md`
4. `task1_spatial_plan_companion.md`
5. `task2_ttprogram_companion_cutover.md`
6. `task3_runtime_gate_and_workload_cutover.md`

额外参考：

- `layered_ir_references.md`
  - 只在需要研究背景或方法论参照时再读；不参与当前状态判断

## 2. 当前活动文档

| 任务链位置 | 文档 | 角色 |
|------|------|----------|
| 总纲 / `Task 0` | `final_blackhole_backend_redesign.md` | 唯一总体设计；只保留长期主链、真源规则、任务链与 cutover invariant |
| `Task 0` | `task0_ir_layering_root_cause.md` | layered IR 根因诊断与整改方向；解释为什么必须切到 companion 主链 |
| `Task 1` | `task1_spatial_plan_companion.md` | `SpatialPlan companion` 的 schema / pass owner 文档；为后续 cutover 提供 owner 边界 |
| `Task 2` | `task2_ttprogram_companion_cutover.md` | `TTProgram companion` cutover、target owner 边界、materialization 规则与完成判定文档 |
| `Task 3` | `task3_runtime_gate_and_workload_cutover.md` | runtime gate、support surface 与 workload re-enable 文档 |

## 2.1 当前执行优先级

当前按下面顺序推进：

1. `Task 2A`
   - `Blackhole` active compile path 的 internal owner cutover
2. `Task 2B`
   - `MaterializeBlackholeExecutable` writer cutover
   - 旧 `Phase C` owner 退位
3. `Task 2C`
   - phase bundle / test helper 固化
4. `Task 3A`
   - runtime gate 与 `flash-attn` payoff
5. `Task 3B`
   - wider family / support surface

当前明确不作为优先项：

- non-Blackhole backend 统一收口
- repo-wide frontend 统一
- public Python `transform` API 改名

## 3. 已归档文档

| 文档 | 角色 | 当前状态 |
|------|------|----------|
| `archive/stage4_stage0_guardrails.md` | `Stage 0` 护栏与 cutover 前提 | 已完成；作为 layered IR 迁移前置假设保留 |
| `archive/stage4_phase_a_semantic_ir.md` | 旧 `SemanticProgram` 边界文档 | 已完成；仅保留历史实现边界参考 |
| `archive/stage4_semantic_manifest.md` | `Phase A` 信息源边界 | 已完成；仅保留 evidence ownership 历史参考 |
| `archive/stage4_phase_b_spatial_ir.md` | 旧 `SpatialProgram` 边界文档 | 已完成；仅保留旧主链审计结论 |
| `archive/stage4_phase_a_formalization_note.md` | `Phase A` 理论化说明 | research reference；不承担当前工程入口 |
| `archive/task2_task3_tt_target_cutover.md` | 旧 `Task 2 / Task 3` 合并过渡稿 | 已归档；当前 active 设计已拆分为独立的 `Task 2` / `Task 3` 文档 |

## 4. 参考文档

按需要再读：

- `layered_ir_references.md`
  - 分层 IR 设计的研究参考与跨论文启发；仅作设计输入，不承担协议真源职责

## 5. 清理规则

- 总纲只保留长期架构、层间边界、真源规则、任务链与 cutover invariant
- `progress.md` 只保留当前代码基线、当前 blocker、任务链状态、边界与最新验证
- 根因解释统一放在 `task0_ir_layering_root_cause.md`
- companion schema、pass owner 与旧 pass 归位统一放在 `task1_spatial_plan_companion.md`
- `Task 2` 的 target owner cutover 统一维护在
  `task2_ttprogram_companion_cutover.md`
- `Task 3` 的支持面和 runtime gate 统一维护在
  `task3_runtime_gate_and_workload_cutover.md`
- 新两层 companion 主链的长期 owner 以总纲为准；旧 `SemanticProgram / SpatialProgram` 阶段文档不再回升为总体权威
- 不在 `final` 和 `progress` 里重复维护根因长文、pass 细化列表或 phase baseline 细节
- `README` 只做索引和分工说明，不重复维护阶段 backlog
- 已完成的 `Stage 0 / Phase A / Phase B` 文档全部放入 `archive/`
- 根目录文件名按当前任务链命名；旧 `stage4_phase_*` 名称不再作为活动入口出现

## 6. Archive

查看 `archive/README.md`。
