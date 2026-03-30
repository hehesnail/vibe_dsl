# TileLang Blackhole 后端扩展项目

这是一个为 TileLang 增加 Tenstorrent Blackhole 后端的开发工作区。

## 当前入口

当前仓库只保留一份总体设计：

- `tasks/dev_design/final_blackhole_backend_redesign.md`

当前工作规范：

- `AGENTS.md`
- `CLAUDE.md`

当前状态追踪：

- `tasks/progress.md`

## 当前状态

- 当前日期：`2026-03-30`
- 当前阶段：Stage 3 multi-core runtime 调度
- 当前状态：
  - formal direct host path 已完成
  - copy / GEMM multi-core direct path 已通过
  - `tilelang.compile(..., execution_backend="tvm_ffi")` 的 Blackhole wrapper/export path 已恢复
  - TT-Metal contract formalization 正在继续推进，当前主线在 P0/P3 收正，以及 P5 的 program-local semaphore schema / kernel binding / 最小 device-side dataflow semaphore builtin 预埋
- 通用 pass 当前结论：
  - `FlattenBuffer` / `VectorizeLoop` 已验证可接回
  - `StorageRewrite` 当前确认不兼容 Blackhole CB 模型
- Blackhole 正式执行路径已收敛到 `ExecutableSpec -> BlackholeModule` 进程内 direct host path

## 仓库组成

- `tilelang_repo/`：TileLang 开发仓库，Blackhole 后端代码主要改这里
- `tt_metal_repo/`：TT-Metal 开发仓库，示例、API 参考主要看这里
- 顶层仓库：任务文档、经验记录、测试、脚本和总控

## 常用目录

- `tilelang_repo/src/target/`
- `tilelang_repo/src/transform/`
- `tilelang_repo/tilelang/engine/`
- `tilelang_repo/build/`
- `tilelang_repo/testing/python/target/blackhole/`
- `tt_metal_repo/tt_metal/api/tt-metalium/`
- `memory/`
- `tasks/`

## 当前工程约束

- Blackhole 后端总体设计只维护一份，不再保留平行架构文档
- 当前主路径不是“单个 kernel 字符串”，也不是 external runner，而是 `ExecutableSpec -> BlackholeModule` direct host path
- 当前 pass 主线：`AnnotateBlackholeCopySemantics` → `BlackholeDeviceResourceCanonicalization` → `SplitHostDevice` → `SplitBlackholeKernel` → `LowerBlackholeOps` → `PlanBlackholeCB`
- 先统一协议与执行路径，再补 copy / GEMM，再做 multi-core
- 先设计后编码，设计要落到仓库文档里
- 完成任务后要同步更新进度、经验、问题记录，并提交推送

## 推荐阅读顺序

1. `tasks/dev_design/final_blackhole_backend_redesign.md`
2. `tasks/progress.md`
3. `AGENTS.md` 或 `CLAUDE.md`
4. `memory/general_dev.md`
5. `memory/bugs.md`
6. 相关源码与测试

## 说明

旧的总体设计文档、`build_blackhole/` 和 legacy runner 已移除，以减少信息干扰。
