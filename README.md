# TileLang Blackhole 后端扩展项目

这是一个为 TileLang 增加 Tenstorrent Blackhole 后端的开发工作区。

## 当前入口

当前仓库只保留一份总体设计：

- `tasks/dev_design/final_blackhole_backend_redesign.md`

当前工作规范：

- `AGENTS.md`

当前状态追踪：

- `tasks/progress.md`

## 仓库组成

- `tilelang_repo/`：TileLang 开发仓库，Blackhole 后端代码主要改这里
- `tt_metal_repo/`：TT-Metal 开发仓库，示例、API 参考主要看这里
- 顶层仓库：任务文档、经验记录、测试、脚本和总控

## 常用目录

- `tilelang_repo/src/target/`
- `tilelang_repo/src/transform/`
- `tilelang_repo/tilelang/engine/`
- `tilelang_repo/tools/blackhole_runner/`
- `tt_metal_repo/tt_metal/api/tt-metalium/`
- `tests/target/`
- `tests/transform/`
- `memory/`
- `docs/`
- `tasks/`

## 当前工程约束

- Blackhole 后端总体设计只维护一份，不再保留平行架构文档
- 当前主路径不是“单个 kernel 字符串”，而是朝 `ExecutableSpec -> runner` 收敛
- 先统一协议，再补 copy/gemm，再做 multi-core
- 先设计后编码，设计要落到仓库文档里
- 完成任务后要同步更新进度、经验、问题记录，并提交推送

## 推荐阅读顺序

1. `tasks/dev_design/final_blackhole_backend_redesign.md`
2. `AGENTS.md`
3. `tasks/progress.md`
4. `memory/general_dev.md`
5. `memory/bugs.md`
6. 相关源码与测试

## 说明

旧的总体设计文档和过时的阶段设计文档已经移除，以减少信息干扰。
