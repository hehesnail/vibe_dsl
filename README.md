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
  - `Normalized Tile TIR`
  - `SpatialPlan`
  - `TTProgram`
  - `ExecutableSpec`
- semantic manifest 路径已完成：
  - `AnalyzeSemanticStructure` 已是 manifest-first
  - 当前 compile path 已不再把 semantic mirror 当长期 owner 层
- tile-compute preservation
  和 post-preservation pass shrink
  已完成：
  downstream scalar-loop matcher /
  composite generate family
  不再是 active compute truth
- 当前主实施阶段是
  `Algorithmic generalization Phase E: Decision-Use Cutover`
- 当前重点是让
  `AccessRegion`、
  `DependenceComponent`、
  `LiveValueSSA`
  和 TT live-form solver
  真正驱动 legality /
  query /
  typed plan /
  unsupported diagnostic，
  然后再启动 production
  `TileComputeDAG` /
  legalizer /
  covering
- `SpatialPlan` 当前定位已经收紧为
  virtual spatial/dataflow owner layer；
  它必须承载 task/flow/layout/partition/order 这些执行相关但非 TT-specific 的 truth
- 算法化重构受两条 guardrail 约束：
  - anti-overdesign pay-rent rule：
    新结构必须改变 active-chain 决策或删除旧 side channel
  - problem-family generality rule：
    当前 workload case 只能作为 witness，
    不能成为协议定义
- 这套分层不是为单个 consumer 设计，而是用于统一承接复杂前端计算 family：
  - selection / indexing
  - routed / grouped / ragged dispatch
  - paged / indexed sparse access
  - stateful reduction-update
  - chunked recurrence / scan
- 旧的单层方案、历史 runtime 架构说明、旧 implementation plan 都已移入 `tasks/dev_design/archive/`，不再作为当前实现入口。
- 当前验证面落在 `ExecutableSpec -> rt_mod_blackhole -> BlackholeModule` direct host path：
  - copy / GEMM direct path 已稳定
  - `flash-attn` compile/source/spec baseline 已稳定
  - small / 32x32 bf16 flash-attn
    direct runtime subset 已 admission
  - seq64 / multi-K-step
    compile/source/spec lowering 已稳定，
    direct-runtime correctness
    仍通过 typed unsupported-reason metadata gate
    fail closed
  - full mesh/distributed runtime
    仍是后续 admission lane；
    schema 已在 `TTProgram`
    层表达，
    runtime 当前只 admission unit mesh /
    replicated `MeshBuffer`
    subset

## 推荐阅读顺序

1. `tasks/dev_design/final_blackhole_backend_redesign.md`
2. `tasks/progress.md`
3. `tasks/dev_design/README.md`
4. `tasks/dev_design/2026-04-28-blackhole-algorithmic-generalization.md`
5. `tasks/dev_design/2026-04-28-blackhole-tile-compute-legalizer-dag-covering.md`
6. `AGENTS.md` 或 `CLAUDE.md`
7. `memory/general_dev.md`
8. `memory/bugs.md`
9. 相关源码与测试

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
