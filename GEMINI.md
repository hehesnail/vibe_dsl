# GEMINI.md

## 作用

本文件用于指示 **Gemini CLI** 在这个仓库里应该如何工作。作为基础强制指令（Contextual Precedence），本文件中的规则具有最高优先级。

它不是总体架构设计文档。涉及 Blackhole 后端架构、主路径、设计取舍时，只看这一份：

- `tasks/dev_design/final_blackhole_backend_redesign.md`

如果历史做法、旧笔记、旧认知与它冲突，以这份总设计为准。

## 仓库结构

这个仓库主要由三部分组成：

- `tilelang_repo/`：TileLang 开发仓库，Blackhole 后端代码主要改这里
- `tt_metal_repo/`：TT-Metal 开发仓库，runner、示例、API 参考主要看这里
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

这是本仓库的强约束。对于复杂改动，**建议使用 `enter_plan_mode` 工具**进行设计。

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

当前 Blackhole 后端默认推进顺序：

1. ~~attrs / 协议~~ ✅
2. ~~`ExecutableSpec`~~ ✅
3. ~~`rt_mod_blackhole`~~ ✅
4. ~~`BlackholeModule` direct path 补全~~ ✅
5. ~~Copy E2E 验收（direct path）~~ ✅
6. ~~split-before 语义规划~~ ✅
7. ~~通用 pass 回收~~ ✅（FlattenBuffer/VectorizeLoop 已验证；StorageRewrite 永久排除）
8. ~~GEMM 接入 Steps 1-5~~ ✅（CB identity 唯一协议已收正）
9. ~~GEMM E2E 验收~~ ✅（transpose_B + host tilize/untilize 已补齐）
10. ~~multi-core~~ ✅（formal direct host path 已完成）
11. TT-Metal contract formalization 收尾：
    - P0：更丰富 compute ABI / dtype 分层继续收正
    - P3：更宽 accessor / runtime work execution surface
    - P4：copy/dataflow 泛化（non-tile/stick/sharded）
    - P5：multi-core synchronization 预埋

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

5. **版本控制**
   - 除非用户明确要求提交，否则 **不要擅自 staging (git add) 或提交 (git commit)**。这是 Gemini CLI 的重要安全准则。
   - 当用户要求提交时，记得先用 `git status`, `git diff HEAD` 检查，并提供 draft 提交信息给用户确认。

## 不要做的事

- 不要再新增第二份总体设计文档
- 不要把单个 kernel 源码字符串当成后端主产物
- 不要重新引入或扩展 legacy external runner 路径
- 不要把多核调度主要放在 codegen 层
- 不要把 codegen-only 或 reference-only 测试称为 true E2E
- 不要让文档和代码长期处于协议错位状态

## 当前事实约束

- Blackhole 正式执行路径只剩 `BlackholeModule` 进程内 direct host path
- `tilelang.compile(..., execution_backend="tvm_ffi")` 的 Blackhole wrapper/export path 已恢复
- 默认开发构建目录固定为 `tilelang_repo/build/`
- `build_blackhole/` 和 legacy runner 都已删除
- Pass 管线顺序：`AnnotateBlackholeCopySemantics` → `BlackholeDeviceResourceCanonicalization` → `SplitHostDevice` → `SplitBlackholeKernel` → `LowerBlackholeOps` → `PlanBlackholeCB`
- `SplitBlackholeKernel` 已实现并已接入管线；纯 copy 走 `fused_dataflow` 单 kernel，GEMM 走 3-kernel（reader/compute/writer）
- direct runtime 当前正式支持面：
  - copy：equal source/dest range，且 stride = 1
  - GEMM：A/B-separated reader range + writer output range
  - accessor：仅 interleaved + DRAM + `common_runtime_arg_count = 0`

## 什么算完成

一个任务完成，至少要满足：

- 实现符合 `tasks/dev_design/final_blackhole_backend_redesign.md`
- 相关层的协议一致
- 做了与任务匹配的验证
- 相关状态文档仍然真实
- 如果本次工作产生了稳定经验或可复用问题，已经同步到 `memory/` 中
- 明确写出未完成项和限制，而不是假装它们不存在

## 一句话原则

在这个仓库里，Gemini CLI 的工作方式应该是：

- **按唯一总设计推进**
- **先设计后编码**
- **做完后把进度、经验和问题记录补齐，并听从指令决定是否提交**
