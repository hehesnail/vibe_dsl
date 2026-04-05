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
   - `tasks/dev_design/README.md`
4. 然后读代码，不要只看文档

## TT-Sim 环境入口

凡是要做 Blackhole direct runtime / TT-Sim 真执行验证，不要每次重新搜索环境变量或直接说“没环境”；先走仓库里已经固定的 TT-Sim 环境入口：

```bash
source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh
export TILELANG_HOME=<当前 checkout 或 worktree>/tilelang_repo
cd <当前 checkout 或 worktree>/tilelang_repo
```

注意：

- `setup_tt_sim.sh` 必须和后续测试命令在**同一个 shell**里执行
- 如果当前在 git worktree 中工作，**不要 source worktree 里的 `scripts/setup_tt_sim.sh` 副本**；要 source 顶层 checkout 的 `/root/dev/vibe_dsl/scripts/setup_tt_sim.sh`
- source 完 TT-Sim 脚本后，再把 `TILELANG_HOME` 显式指回当前 checkout/worktree 的 `tilelang_repo`
- 具体跑哪个 `pytest` case/selector，按当前 task / design / progress 文档决定，不要在这里硬编码成单个 case
- 一旦进入 TT-Sim / runtime debug，先看 `memory/general_dev.md` 和 `memory/bugs.md`，优先复用已记录的环境、watcher、runtime 边界经验，不要从头试错

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

1. 保持当前稳定基线不回退：
   - `ExecutableSpec -> rt_mod_blackhole -> BlackholeModule` direct host path
   - copy / GEMM current support surface
   - 第一批复杂 consumer 已打通的 compile-path 子集（当前以 `flash-attn` 为主）
2. 文档、任务安排和实现边界统一以 `tasks/dev_design/final_blackhole_backend_redesign.md` 为准。
3. 当前 Stage 4 直接按分阶段文档执行，不再保留单一总 implementation plan 入口：
   - `tasks/dev_design/stage4_stage0_guardrails.md`
   - `tasks/dev_design/stage4_phase_a_semantic_ir.md`
   - `tasks/dev_design/stage4_phase_b_spatial_ir.md`
   - `tasks/dev_design/stage4_phase_c_tt_target_ir.md`
4. 按新总设计执行：
   - Phase A1：`Stateful Semantic IR`（最小 `Domain/State/Update` + `MapLaw/ReduceLaw` full payload + early semantic seed + post-lift hard freeze）
   - Phase A2：`Stateful Semantic IR`（泛化 recovery + wider `AccessMap/UpdateLaw` traits + `SemanticSupplement` + rebind-aware contract）
   - Phase B：`Spatial Program IR`（`ProgramPhase` module-scope 宿主 + simple-workload fast-path + non-trivial multi-phase gate）
   - Phase C：`TT Target IR`（`TTHardwareModel` stub 先行 + `TTTransportPlan` + common-runtime ABI + `MaterializeTTExecutableSpec` 唯一物化）
5. 在新分层下继续推进：
   - `flash-attn` `blackhole.acc` 语义收正
   - `topk / fusedmoe / paged decode / chunk recurrence` 等 family 的统一承接
   - 更宽 copy/dataflow 支持面（P4）
   - 更宽 synchronization 支持面（P5）

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
- `flash-attn` 当前是第一批 consumer，不是总体架构边界；总体设计同时面向 selection/indexing、routed/grouped dispatch、paged decode、chunk recurrence 等 workload family
- TT-Sim 当前正式环境入口是顶层 `scripts/setup_tt_sim.sh`
- `setup_tt_sim.sh` 与后续测试必须在同一个 shell 中执行
- 如果在 worktree 中运行测试，source 后必须把 `TILELANG_HOME` 指回当前 checkout/worktree

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
