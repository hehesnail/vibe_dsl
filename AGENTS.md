# AGENTS.md

## 作用

本文件用于告诉 Codex 在这个仓库里应该如何工作。

它不是总体架构设计文档。涉及 Blackhole 后端架构、主路径、设计取舍时，只看这一份：

- `tasks/dev_design/final_blackhole_backend_redesign.md`

如果历史做法、旧笔记、旧认知与它冲突，以这份总设计为准。

## 仓库结构

这个仓库主要由三部分组成：

- `tilelang_repo/`：TileLang 开发仓库，Blackhole 后端代码主要改这里
- `tt_metal_repo/`：TT-Metal 开发仓库，TT-Metal API、示例和运行时参考主要看这里
- 顶层仓库：任务文档、经验记录、测试、脚本和总控

常用目录：

- `tilelang_repo/src/target/`
- `tilelang_repo/src/transform/`
- `tilelang_repo/tilelang/engine/`
- `tilelang_repo/build/`
- `tilelang_repo/testing/python/target/blackhole/`
- `tt_metal_repo/tt_metal/api/tt-metalium/`
- `tasks/`
- `memory/`

## 开始任务前

每次开始工作，按这个顺序：

1. 先读 `tasks/dev_design/final_blackhole_backend_redesign.md`
2. 再看 `tasks/progress.md`
3. 如果任务涉及构建、调试、历史问题，再看：
   - `memory/general_dev.md`
   - `memory/bugs.md`
4. 然后读代码，不要只看文档

## 先设计，后写代码

这是本仓库的强约束。

如果任务不是非常小的局部修复，在写代码前必须先形成设计，并保留可回溯的设计记录。

规则：

- 总体架构设计只保留一份：`tasks/dev_design/final_blackhole_backend_redesign.md`
- 如果是某个具体实现任务，需要先在 `tasks/dev_design/` 下新增或更新对应设计文档
- 设计文档应至少说明：
  - 目标
  - 影响范围
  - 协议/接口变化
  - 验证方式
- 设计与实现冲突时，先更新设计，再继续写代码

不要跳过设计直接堆代码，也不要把设计只留在对话里。

## 开发中的工作方式

- 优先做小而完整的改动，不做大而散的猜测式重构
- 先统一协议，再补功能
- 先让实现闭环，再谈优化
- 读实际代码、测试、示例，再下判断
- 关键设计取舍必须与 `final_blackhole_backend_redesign.md` 一致

当前 Blackhole 后端默认推进顺序（2026-03-24 更新）：

1. ~~attrs / 协议~~ ✅
2. ~~`ExecutableSpec`~~ ✅
3. ~~`rt_mod_blackhole`~~ ✅
4. ~~`BlackholeModule` direct path 补全~~ ✅
5. ~~Copy E2E 验收（direct path）~~ ✅
6. ~~split-before 语义规划（`AnnotateBlackholeCopySemantics` + `SplitBlackholeKernel`）~~ ✅
7. 通用 pass 回收（`FlattenBuffer` / `VectorizeLoop` 已验证；`StorageRewrite` 当前确认不兼容 Blackhole CB 模型）
8. GEMM 接入（Steps 1-3 ✅，Steps 4-6 **当前首要**）
9. multi-core

## 经验与问题记录

### 什么时候更新 `memory/general_dev.md`

当你在本次任务中发现了以后还会反复用到的稳定经验，就更新：

- 通用开发模式
- 构建/调试技巧
- 代码组织经验
- 后端开发中的稳定方法论

不要把一次性的 workaround 或已经淘汰的旧方案写成“最佳实践”。

### 什么时候更新 `memory/bugs.md`

当你遇到了真实问题，并且问题本身或解决过程以后可能复用时，就更新：

- 现象
- 根本原因
- 解决方式
- 仍然存在的限制

## 完成任务后必须做的事

完成一个任务后，必须主动检查是否需要更新这些文件：

- `tasks/progress.md`
- `memory/general_dev.md`
- `memory/bugs.md`
- 受影响的设计文档
- 受影响的测试或脚本说明

收尾规则：

1. **更新进度**
   - 如果阶段状态、任务状态、下一步重点变化了，更新 `tasks/progress.md`

2. **同步设计**
   - 如果实现改变了原计划，更新对应设计文档
   - 不要让设计文档长期落后于代码

3. **沉淀经验与问题**
   - 有稳定经验，更新 `memory/general_dev.md`
   - 有可复用问题记录，更新 `memory/bugs.md`

4. **明确未完成项**
   - 没做完就写没做完
   - 没验证就写没验证
   - 有限制就写限制，不要默认后续的人会自己猜到

5. **提交并推送**
   - 任务完成后，记得 `git commit` 和 `git push`
   - 提交信息要能反映本次改动的主题
   - 文档改动和代码改动可以分开提交，避免混乱

## 不要做的事

- 不要再新增第二份总体设计文档
- 不要把单个 kernel 源码字符串当成后端主产物
- 不要把 `SplitBlackholeKernel` 当成当前前置条件
- 不要把多核调度主要放在 codegen 层
- 不要重新引入或扩展 legacy external runner 路径
- 不要把 codegen-only 或 reference-only 测试称为 true E2E
- 不要让文档和代码长期处于协议错位状态

## 当前事实约束

- Blackhole 正式执行路径只允许 `BlackholeModule` 进程内 direct host path
- 默认开发构建目录固定为 `tilelang_repo/build/`
- `build_blackhole/` 与 legacy runner 已删除；如果文档或旧记录提到它们，按历史语境理解，不要恢复
- 当前 pass 主线按 `AnnotateBlackholeCopySemantics` → `SplitBlackholeKernel` → `LowerBlackholeOps` → `PlanBlackholeCB` 推进
- `SplitBlackholeKernel` 已接入管线：纯 copy 维持 `fused_dataflow` 单 kernel，GEMM 当前走 reader / compute / writer 三段 schema

## 什么算完成

一个任务完成，至少要满足：

- 实现符合 `tasks/dev_design/final_blackhole_backend_redesign.md`
- 相关层的协议一致
- 做了与任务匹配的验证
- 相关状态文档仍然真实
- 如果本次工作产生了稳定经验或可复用问题，已经同步到 `memory/` 中
- 已完成 `git commit` 和 `git push`
- 明确写出未完成项和限制，而不是假装它们不存在

## 一句话原则

在这个仓库里，Codex 的工作方式应该是：

- **按唯一总设计推进**
- **先设计后编码**
- **做完后把进度、经验、问题记录和 git 提交补齐**
