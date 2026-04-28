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
2. 再读任务级设计合同：
   - `tasks/dev_design/task0_ir_layering_root_cause.md`
   - `tasks/dev_design/task1_spatial_plan_companion.md`
   - `tasks/dev_design/task2_ttprogram_companion_cutover.md`
   - `tasks/dev_design/task3_runtime_gate_and_workload_cutover.md`
3. 再看 `tasks/progress.md`
4. 再看 `tasks/dev_design/README.md`
5. 再看当前 algorithmic lane 设计：
   - `tasks/dev_design/2026-04-28-blackhole-algorithmic-generalization.md`
   - `tasks/dev_design/2026-04-28-blackhole-tile-compute-legalizer-dag-covering.md`
6. 如果任务涉及构建、调试、历史问题，再看：
   - `memory/general_dev.md`
   - `memory/bugs.md`
7. 然后读代码，不要只看文档

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

1. 文档、任务安排和实现边界统一以当前入口为准：
   - `tasks/dev_design/final_blackhole_backend_redesign.md`
   - `tasks/progress.md`
   - `tasks/dev_design/README.md`
   - `tasks/dev_design/task0_ir_layering_root_cause.md`
   - `tasks/dev_design/task1_spatial_plan_companion.md`
   - `tasks/dev_design/task2_ttprogram_companion_cutover.md`
   - `tasks/dev_design/task3_runtime_gate_and_workload_cutover.md`
   - `tasks/dev_design/2026-04-28-blackhole-algorithmic-generalization.md`
   - `tasks/dev_design/2026-04-28-blackhole-tile-compute-legalizer-dag-covering.md`
2. cleanup `task0-task5`
   已完成并归档到
   `tasks/dev_design/archive/`；
   它们只作完成期历史记录，
   不再作为当前活动路线图
3. 当前 active lane
   统一看 `tasks/progress.md`；
   当前是
   `Algorithmic generalization Phase E: Decision-Use Cutover`
4. `materialization / source-live-form`
   已重新收束为
   `tasks/dev_design/2026-04-23-blackhole-live-form-materialization-admission.md`
   下的 support-surface admission lane
5. Algorithmic generalization
   Phase A-D
   已作为 foundation 完成；
   当前 Phase E
   必须让
   `AccessRegion`、
   `DependenceComponent`、
   `LiveValueSSA`
   和 TT live-form solver
   成为 active legality /
   query /
   action inputs
6. `TileComputeDAG` /
   legalizer /
   covering
   必须等 Phase E decision-use gate
   对 admitted compute surface 生效后再进入 production migration
7. 后续更宽 workload /
   mesh /
   distributed runtime
   支持面必须通过
   `SpatialPlan -> TTProgram -> ExecutableSpec`
   typed schema
   和 leaf admission
   扩展，
   不能回到 runtime-only patch
   或隐式 payload 通道

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
- 当前实际 active chain 是：
  `Normalized Tile TIR -> BuildSpatialPlan -> ValidateSpatialPlan -> SplitBlackholeKernel -> PlanTTBlocks -> SelectBlackholeTTMetalBuiltins -> PlanTTCompute / PlanTTTransport / PlanTTSync / PlanTTABI / PlanTTExecution -> BuildTTProgram -> ValidateTTProgram -> MaterializeBlackholeExecutable -> runtime / codegen leaf readers`
- 上面这串名字描述的是当前 pass/phase 实现，不是新的长期 IR 层；
  长期主链仍然只有
  `Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec`
- `tl.blackhole_logical_buffer_tile_bridge_specs`
  已从 active code path 删除，
  不能再作为 bridge exception
  或长期边界重新引入
- `compute_contract` / `gemm_contract` /
  `multi_*_contracts`
  已退出 active chain；
  compute truth 只能经 typed
  `TTComputeOpPlan`
  / `KernelSpec.compute_ops`
  流动
- `SplitBlackholeKernel` 已实现并已接入管线；纯 copy 走 `fused_dataflow` 单 kernel，GEMM 走 3-kernel（reader/compute/writer）
- direct runtime 当前 admitted 支持面：
  - copy：equal source/dest range，且 stride = 1
  - GEMM：A/B-separated reader range + writer output range；fresh fragment / preclear zero-init 统一走 `clear_accum=true` direct path
  - accessor：仅 interleaved + DRAM + `common_runtime_arg_count = 0`
  - communication：non-oversubscribed explicit semaphore / remote-endpoint subset
  - live-form / materialization：
    当前 admitted bf16 subset 包括
    `fragment_fill -> cast -> publish`
    的 `pack_thread_direct_store`
    path，
    以及 GEMM post-merge
    direct cast consumer
    zero-preclear full-tile
    `pack_tile`
    path
- `flash-attn` compile-path / source/spec baseline 已稳定；
  当前 admitted bf16 direct runtime subset
  覆盖 small single-work-item
  和 32x32 MHA / GQA；
  seq64 / multi-K-step
  只完成 compile/source/spec
  exact-CB republish admission，
  direct-runtime correctness
  仍通过
  `multi-block exact CB-republish flash-attention direct runtime correctness`
  typed unsupported reason gate 住
- 当前 blocker 统一以 `tasks/progress.md` 为准；
  当前下一步是完成
  `Algorithmic generalization Phase E`
  剩余 decision-use cutover：
  wider subject-map deletion、
  graph-wide worklist/lattice solver、
  TTProgram / ExecutableSpec
  projection admission audit
- Algorithmic generalization
  必须遵守 anti-overdesign pay-rent rule
  和 problem-family generality rule；
  当前 workload case
  只能作为 active-chain witness，
  不能成为协议定义
- full mesh/distributed runtime
  仍是后续 admission lane；
  schema 已在 `TTProgram`
  层表达，
  runtime 当前只 admission unit mesh /
  replicated `MeshBuffer`
  subset
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
