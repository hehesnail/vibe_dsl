# CLAUDE.md

本文件是 Claude Code 在这个仓库里的工作规范。

架构设计只看一份：`tasks/dev_design/final_blackhole_backend_redesign.md`

## 工作原则

- 结果优先于礼貌式询问；任务的一部分就直接做
- 只在真正存在歧义、继续会产出相反结果时停下来问
- 先服从完成标准，再服从项目既有风格，最后服从明确无歧义的用户指令
- 汇报只说对工程判断有用的事实，不做过程噪音

## 仓库入口

- `tilelang_repo/`
  - TileLang 开发仓库；Blackhole 后端主改这里
- `tt_metal_repo/`
  - TT-Metal API、示例、运行时参考
- `tasks/`
  - 设计文档与进度
- `memory/`
  - 稳定经验与问题记录

常用目录：

- `tilelang_repo/src/transform/`
- `tilelang_repo/src/target/`
- `tilelang_repo/tilelang/engine/`
- `tilelang_repo/build/`
- `tt_metal_repo/tt_metal/api/tt-metalium/`

## 开工顺序

每次开始前按顺序读：

1. `tasks/dev_design/final_blackhole_backend_redesign.md`
2. `tasks/progress.md`
3. `tasks/dev_design/README.md`
4. 需要构建/调试/历史背景时再读 `memory/general_dev.md` 和 `memory/bugs.md`
5. 然后读代码，不要只看文档

## 工作区与环境

- 默认不要用 `git worktree`
- 直接在当前 checkout 上工作；只有用户明确要求时才新建 worktree
- 如果当前就在 worktree 中，继续遵守已有 worktree 约束

TT-Sim / direct runtime 验证统一从这里进：

```bash
source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh
export TILELANG_HOME=<当前 checkout 或 worktree>/tilelang_repo
cd <当前 checkout 或 worktree>/tilelang_repo
```

注意：

- `setup_tt_sim.sh` 和后续测试命令必须在同一个 shell
- 如果在 worktree 中工作，仍然 source 顶层 checkout 的脚本
- source 之后，把 `TILELANG_HOME` 显式指回当前 checkout / worktree
- 进入 TT-Sim / runtime debug 前，先看 `memory/general_dev.md` 和 `memory/bugs.md`

## 工程纪律

### 设计与实现

- 非小修复先更新或创建 `tasks/dev_design/` 下的设计文档
- 设计文档至少说明：目标、影响范围、协议变化、验证方式
- 设计与实现冲突时，先改设计，再改代码
- 做小而完整的改动：先收正协议，再补功能，先闭环，再优化

### 代码结构

- analysis 只产出 facts，transformation 只消费 facts 并改写 IR
- 多阶段流动的中间状态必须有显式类型，不允许靠散落局部变量隐式传递
- 跨模块共享的 attr key / schema field / enum value 只定义一处
- 一个函数 / pass / 文件只解决一个问题

### 设计约束

- 先定位根因，不优先做 workaround
- 不做单 case 特判；优先统一 schema、统一 IR 语义、统一 pass 边界
- 需要的信息优先从 IR 分析；缺失就扩 IR / attrs / schema / DSL
- IR 分析必须基于结构与类型，不允许名字匹配恢复语义

### 对 Blackhole 的具体要求

- `runtime_args`、`buffer`、`cb`、`segment` 等绑定必须由 IR / schema 明确表达或稳定推导
- 不要为了绕开卡点新增并行执行路径或额外 emitter；优先修主路径
- 优先收正协议和主链，不要堆 workaround

### 不要做的事

- 不要新增第二份总体设计文档
- 不要把单个 kernel 字符串当成后端主产物
- 不要重新引入或扩展 legacy external runner 路径
- 不要让文档和代码长期协议错位
- 不要把名字匹配带进分析协议判断

## 当前推进顺序

1. 保持当前稳定基线不回退：
   - `ExecutableSpec -> rt_mod_blackhole -> BlackholeModule`
   - copy / GEMM 当前支持面
   - 已打通的复杂 consumer compile-path 子集
2. 文档、任务安排和实现边界统一以
   `tasks/dev_design/final_blackhole_backend_redesign.md` 为准
3. Stage 4 按阶段文档执行：
   - `tasks/dev_design/stage4_stage0_guardrails.md`
   - `tasks/dev_design/stage4_phase_a_semantic_ir.md`
   - `tasks/dev_design/stage4_phase_b_spatial_ir.md`
   - `tasks/dev_design/stage4_phase_c_tt_target_ir.md`
4. 当前重点：
   - `Phase A` 已完成
   - `Phase B` 的 contract-hardening 子阶段已完成，但整体未完成；
     当前主任务是补齐正文定义的 spatial synthesis algorithm
     与 stronger execution-bearing contract / validator
   - `Phase C` 准备轨已落地，但正式 `TTProgram / MaterializeTTExecutableSpec`
     cutover 仍以前置的 `Phase B` 整体完成为前提
5. 后续承接：
   - `flash-attn` `blackhole.acc` correctness payoff（`Phase C2`）
   - `topk / fusedmoe / paged decode / chunk recurrence`
   - 更宽 copy/dataflow 与 synchronization 支持面

## 收尾要求

完成任务后必须：

1. 更新 `tasks/progress.md`
2. 同步受影响的设计文档
3. 稳定经验写入 `memory/general_dev.md`，可复用问题写入 `memory/bugs.md`
4. 明确未完成项和限制
5. 清理残留后台测试 / 构建 / 长命令
6. `git commit` 并 `git push`

## 什么算完成

- 实现符合 `tasks/dev_design/final_blackhole_backend_redesign.md`
- 相关层协议一致
- 做了与任务匹配的验证
- 没有残留后台长命令仍在运行
- 进度文档仍然真实
- 稳定经验和问题已同步
- `git commit` / `git push` 已完成
- 未完成项和限制已写清楚

## 当前事实约束

- Blackhole 正式执行路径只剩 `BlackholeModule` 进程内 direct host path
- `tilelang.compile(..., execution_backend="tvm_ffi")` 的 Blackhole wrapper/export path 已恢复
- 默认开发构建目录固定为 `tilelang_repo/build/`
- 默认并行编译线程数按 `-j32`
- `build_blackhole/` 与 legacy runner 已删除
- `tasks/dev_design/` 根目录只保留活动文档；`archive/` 全视为历史记录
- `Phase A` 与 semantic manifest 已完成；
  `AnalyzeSemanticStructure` 已全面 manifest-first，
  `blackhole.fragment_regions` 当前只剩 lowering-facing 兼容职责
- 当前 Blackhole 设备侧主链：
  `LowerDeviceStorageAccessInfo -> AugmentSemanticManifest -> LowerIntrin -> Simplify -> HoistBroadcastValues -> SplitBlackholeKernel -> AnalyzeBlackholeWorkDecomposition -> AnalyzeBlackholeFragmentRegions -> AnalyzeBlackholePipelineStages -> AnalyzeSemanticStructure -> LiftStatefulSemanticIR -> ValidateStatefulSemanticIR -> ValidateSemanticRefinement -> LowerToSpatialProgram -> ValidateSpatialProgram -> LowerBlackholeOps -> PlanBlackholeCB -> AssignBlackholeCores`
- `SplitBlackholeKernel` 已接入；纯 copy 走单 kernel，GEMM 走 reader / compute / writer 三 kernel
- direct runtime 当前正式支持面：
  - copy：equal source/dest range 且 stride = 1
  - GEMM：A/B-separated reader range + writer output range
  - accessor：仅 interleaved + DRAM + `common_runtime_arg_count = 0`
- `flash-attn` forward subset 已打通当前 compile-path；
  `blackhole.acc` correctness payoff 归属 `Phase C2`
- 当前总体 blocker 先是 `Phase B` 正文定义的 spatial synthesis 未完成，
  然后才是 `Spatial Program IR -> TT Target IR` 的单一真源切换
- `SpatialProgram` 已是当前唯一 spatial 主链，但 stronger contract 仍未完成
- 后续所有架构推进都以
  `Stateful Semantic IR -> Spatial Program IR -> TT Target IR`
  为准
