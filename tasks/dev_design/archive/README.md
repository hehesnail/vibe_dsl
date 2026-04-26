# Archive

`tasks/dev_design/archive/` contains documents that are no longer current entry points.

Rules:

1. Files under `archive/` are historical references only.
2. They may still contain useful implementation history, debugging context, or completed design details.
3. They must not be treated as the current architecture or current task plan.
4. Internal links, task names, and file paths inside archived documents may intentionally remain historical rather than fully rewritten.
4.1 Archived docs may still use historical `P0 / P1 / ...` labels.
    These are legacy milestone names, not the current active numbering.
    Current active numbering is defined only by `tasks/progress.md`;
    do not infer active priority from archived labels.
5. Current work should start from:
   - `tasks/dev_design/final_blackhole_backend_redesign.md`
   - `tasks/progress.md`
   - `tasks/dev_design/README.md`
   - `tasks/dev_design/task0_ir_layering_root_cause.md`
   - `tasks/dev_design/task1_spatial_plan_companion.md`
   - `tasks/dev_design/task2_ttprogram_companion_cutover.md`
   - `tasks/dev_design/task3_runtime_gate_and_workload_cutover.md`

Notable archived files:

- `legacy_blackhole_runtime_architecture.md`: old mixed runtime/compiler architecture narrative
- `2026-04-02-stateful-tiled-ir-phase1-implementation-plan.md`: superseded pre-layered implementation-plan draft
- `phase_b_code_audit.md`: `Phase B` 代码审计快照，结论已吸收进完成阶段文档
- `review_final_blackhole_backend_redesign.md`: 总设计审计快照，当前只保留历史参考价值
- `layered_ir_references.md`: 分层 IR 研究输入和方法论参考，
  当前只作为历史/方法论资料，不是活动设计入口
- `2026-04-16-blackhole-final-legacy-protocol-cleanup.md`
  and `2026-04-16-blackhole-final-legacy-protocol-cleanup-task0.md`
  through `task5.md`: completed legacy protocol cleanup boundary records.
  Cleanup is not the current active lane; use `tasks/progress.md`
  for current status and task order.
- `task2_task3_tt_target_cutover.md`: 已归档的旧 `Task 2 / Task 3`
  合并过渡稿；当前 active 设计已拆分为独立的 `Task 2` / `Task 3` 文档
- `stage4_stage0_guardrails.md`: completed Stage 0 guardrails doc
- `stage4_phase_a_semantic_ir.md` / `stage4_semantic_manifest.md`: completed Phase A boundary docs
- `stage4_phase_b_spatial_ir.md`: completed old SpatialProgram boundary doc
- `stage4_phase_a_formalization_note.md`: archived theory note for the old Phase A narrative
- Stage 0-3 and completed Stage 4 design/implementation-plan history
