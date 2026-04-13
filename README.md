# TileLang Blackhole Backend Workspace

这是 TileLang Tenstorrent Blackhole 后端的主开发工作区。

## 当前唯一主入口

- 总设计: `tasks/dev_design/final_blackhole_backend_redesign.md`
- 当前进度: `tasks/progress.md`
- 活动设计索引: `tasks/dev_design/README.md`
- 仓库工作规范: `AGENTS.md`、`CLAUDE.md`
- 稳定经验与问题记录: `memory/general_dev.md`、`memory/bugs.md`

## 当前方向

- 当前权威架构已经切换到分层 IR 主线:
  - `Stateful Semantic IR`
  - `Spatial Program IR`
  - `TT Target IR`
- `Phase A` 与 `stage4_semantic_manifest` `Phase 1-2` 已完成：
  - `AnalyzeSemanticStructure` 已是 manifest-first
  - `blackhole.compute_regions` 只剩 residual reduction evidence 与 lowering compatibility 职责
- `Phase B / Spatial Program IR` 的 compile-path cutover 已收口，当前作为
  `Phase C` 的活跃输入边界保留
- 当前主实施阶段是 `Phase B / Spatial Program IR` 的 contract hardening
- 当前总体 blocker 是 `Spatial Program IR -> TT Target IR` 的单一真源切换尚未完成
- `Spatial Program IR` 当前定位已经收紧为
  target-informed but non-target-materialized virtual spatial/dataflow program；
  它必须承载 task/flow/layout/partition/order 这些执行相关但非 TT-specific 的 truth
- 这套分层不是为单个 consumer 设计，而是用于统一承接复杂前端计算 family：
  - selection / indexing
  - routed / grouped / ragged dispatch
  - paged / indexed sparse access
  - stateful reduction-update
  - chunked recurrence / scan
- 旧的单层方案、历史 runtime 架构说明、旧 implementation plan 都已移入 `tasks/dev_design/archive/`，不再作为当前实现入口。
- 当前稳定执行基线仍是 `ExecutableSpec -> rt_mod_blackhole -> BlackholeModule` direct host path：
  - copy / GEMM direct path 已稳定
  - `flash-attn` 只是第一批 consumer，当前 compile-path 已打通
  - `fusedmoe`、`topk`、`paged decode`、`mamba chunk state` 已进入总设计覆盖面
  - `flash-attn` 的 `blackhole.acc` correctness payoff 属于 `Phase C2`，不再是当前总体架构 blocker

## 推荐阅读顺序

1. `tasks/dev_design/final_blackhole_backend_redesign.md`
2. `tasks/progress.md`
3. `tasks/dev_design/README.md`
4. `AGENTS.md` 或 `CLAUDE.md`
5. `memory/general_dev.md`
6. `memory/bugs.md`
7. 相关源码与测试

## 仓库结构

- `tilelang_repo/`：TileLang 开发仓库，Blackhole 后端实现主要在这里
- `tt_metal_repo/`：TT-Metal 开发仓库，API、示例和运行时参考主要在这里
- `tasks/`：设计文档、进度和任务安排
- `memory/`：稳定工程经验和问题记录
- `scripts/`：环境准备和辅助脚本

## 当前约束

- 不再新增第二份总体设计文档。
- 当前实现和后续计划都以 `tasks/dev_design/final_blackhole_backend_redesign.md` 为准。
- `tasks/dev_design/` 根目录只保留活动文档；`archive/` 下全部视为历史记录。
- 如果文档与代码发生冲突，先同步设计，再动实现。
