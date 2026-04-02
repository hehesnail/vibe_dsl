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
- 旧的单层方案、历史 runtime 架构说明、旧 implementation plan 都已移入 `tasks/dev_design/archive/`，不再作为当前实现入口。
- 当前稳定执行基线仍是 `ExecutableSpec -> rt_mod_blackhole -> BlackholeModule` direct host path：
  - copy / GEMM direct path 已稳定
  - flash-attn forward subset 的 analysis、fragment/dataflow lowering 和 codegen 已接通当前支持面
  - 当前主 blocker 已收敛为 `blackhole.acc` 混合语义导致的 compute correctness 问题

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
