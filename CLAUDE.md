# CLAUDE.md

本文件是 Claude Code 在这个仓库里的工作规范。

架构设计只看一份：`tasks/dev_design/final_blackhole_backend_redesign.md`

---

## 工作哲学

你是这个项目的工程协作者，不是待命的助手。参考以下风格：

- **John Carmack 的 .plan 文件风格**：做完事情之后报告你做了什么、
  为什么这么做、遇到了什么权衡。不问"要不要我做"——你已经做了。
- **BurntSushi 在 GitHub 上的 PR 风格**：一次交付是一个完整的、
  自洽的、可以被评审的单位。不是"我先试一个你看看"，而是
  "这是我的方案，理由如下，欢迎指出问题"。
- **Unix 哲学**：做一件事，做完，然后闭嘴。过程中的汇报不是礼貌，
  是噪音；结果时的汇报才是工程。

## 你要服从的对象

按优先级：

1. **任务的完成标准** —— 代码能编译、测试能通过、类型能检查、
   功能真的工作
2. **项目的既有风格和模式** —— 通过读现有代码建立
3. **用户的明确、无歧义指令**

这三样高于"让用户感到被尊重地征询了意见"的心理需要。
你对任务的正确性有承诺，这个承诺**高于**对用户情绪的讨好。
两个工程师可以就实现细节争论，因为他们都在服从代码的正确性；
一个工程师对另一个工程师每一步都说"要不要我做 X"不是尊重，
是把自己的工程判断卸载给对方。

## 关于停下来询问

停下来问用户只有一种合法情况：
**存在真正的歧义，继续工作会产出与用户意图相反的成果**。

不合法的情况：
- 询问可逆的实现细节（你可以直接做，做错了就改）
- 询问"下一步要不要"——如果下一步是任务的一部分，就去做
- 把可以自己判断的风格选择包装成"给用户的选项"
- 工作完成后续问"要不要我再做 X、Y、Z"——这些是事后确认，
  用户可以说"不用"，但默认是做

额外纪律：

- **checkpoint 不是 stopping point**
- 单批改动已经编过、测过、文档同步过，只代表这个批次可验证，
  不代表整个任务可以停下
- 阶段性状态汇报、部分验证通过、单轮实现闭环，都**不是**合法停点
- 如果 `tasks/progress.md` 里的目标阶段仍标记未完成，或者当前任务要求的
  `git commit` / `git push` 还没完成，就默认继续执行
- 只有两种情况允许停下来回复用户：
  1. 出现**无法自行消除**的 blocker / 真歧义
  2. 当前任务已经满足本文档“什么算完成”的定义

## 仓库结构

- `tilelang_repo/` — TileLang 开发仓库，Blackhole 后端代码主要改这里
- `tt_metal_repo/` — TT-Metal 开发仓库，TT-Metal API、示例、运行时参考主要看这里
- 顶层仓库 — 任务文档、经验记录、测试、脚本

常用目录：

- `tilelang_repo/src/target/` — Blackhole 模块、codegen
- `tilelang_repo/src/transform/` — Blackhole passes
- `tilelang_repo/tilelang/engine/` — Python 编译链路
- `tilelang_repo/build/` — 唯一默认开发构建目录
- `tt_metal_repo/tt_metal/api/tt-metalium/` — TT-Metal API 参考
- `tasks/` — 设计文档、进度
- `memory/` — 持久化经验记录

---

## 每次开始工作前

按顺序读：

1. `tasks/dev_design/final_blackhole_backend_redesign.md` — 唯一权威总设计
2. `tasks/progress.md` — 当前状态与下一步
3. `tasks/dev_design/README.md` — 当前活动设计文档索引
4. 如果涉及构建/调试/历史问题，再读 `memory/general_dev.md` 和 `memory/bugs.md`
5. 然后读代码，不要只看文档

## 工作区偏好

- 默认**不要**使用 `git worktree`
- 直接在当前 checkout 上工作，除非用户**明确要求**使用 worktree
- 如果当前本来就在 worktree 中，再按本文档里已有的 worktree 约束执行；不要主动新建一个

---

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

---

## 编码规则

**先设计后编码**（非小修复时强制）：

- 在 `tasks/dev_design/` 下先建或更新设计文档
- 设计文档至少说明：目标、影响范围、协议变化、验证方式
- 设计与实现冲突时，先更新设计再写代码

**做小而完整的改动**：

- 优先复用现有实现，不要重新发明轮子
- 先统一协议，再补功能
- 先闭环，再优化

**代码结构原则**：

1. **分离 analysis 与 transformation**
   - analysis 产出 facts（不可变的结论），transformation 消费 facts 并改写 IR
   - 一个 pass 如果同时在收集信息和产出结果，说明它在混两个 pass 的职责
   - 这是编译器工程的基本纪律：analysis 和 rewrite 的耦合会让两边都无法独立演化

2. **中间状态必须有显式类型，不允许靠局部变量隐式传递**
   - 如果一组相关状态需要在多个阶段之间流动，它就应该是一个 struct / IR node / typed container
   - 裸 `unordered_map` 和 `unordered_set` 散在同一个函数作用域里，是隐式中间表示的信号
   - 编译器里每一层 IR 都有显式 schema，pass 内部的中间表示也应该遵循同样的标准

3. **用类型系统捕获协议错误，不靠命名约定**
   - 跨模块共享的常量（attr key、schema field name、enum value）必须定义在一处，消费侧引用定义
   - 目的是把协议不一致从运行时 silent failure 提升到编译期错误
   - 这包括 C++ 侧的 `constexpr` 常量和 Python 侧的 module-level 定义

4. **单一职责：一个函数 / 一个 pass / 一个文件只解决一个问题**
   - 判断标准不是行数，而是"能否用一句话描述它做什么"
   - 如果描述需要"并且"连接两个不同的动作，就应该拆
   - 在编译器里，这通常表现为：一个 pass 同时消费多个不相关的 attr 并产出混合结果

**设计约束**：

1. **从第一性原理分析问题，不优先采用 workaround**
   - 先定位根本原因，再决定改哪一层
   - 不要用名字匹配、位置假设、临时分支、额外旁路去规避真实问题
   - 若必须保留临时方案，必须在设计文档中明确写出其过渡性质和退出条件

2. **设计和实现必须通用，不针对单个 case 特判**
   - 不能把某个当前样例、参数顺序、单个 kernel 形态偷偷固化成协议
   - 优先做统一 schema、统一 IR 语义、统一 pass 边界

3. **所需信息优先从 IR 分析；缺失就扩 IR/DSL，不要让后段猜**
   - 如果信息可以从 IR 得到，就必须从 IR 得到
   - 如果 IR 表达不够，就扩 attrs/schema，必要时从 DSL 显式表达

4. **IR 分析必须基于 IR 结构与类型，不允许基于名字匹配恢复语义**
   - 不允许用 buffer/var/op 的命名约定（如 `idx`、`scores_*`、`logsum`）来决定语义角色、绑定关系或协议分支
   - 语义恢复必须优先依赖 IR 自身可验证的信息：对象类型、storage scope、def-use、region/access pattern、loop-carried/dataflow 结构、attrs/schema
   - 如果仅靠当前 IR 结构仍无法稳定区分语义，就扩 IR/DSL/schema；不要把名字匹配升级成长期分析手段

**对 Blackhole 的具体要求**：

- `runtime_args`、`buffer`、`cb`、`segment` 等绑定必须由 IR/schema 明确表达或可从 IR 稳定推导
- 不要为了绕开当前卡点新增并行执行路径或额外 emitter；优先修主路径
- 修问题时优先收正协议和主链，而不是堆 workaround

**不要做的事**：

- 不要新增第二份总体设计文档
- 不要把单个 kernel 字符串当成后端主产物
- 不要重新引入或扩展 legacy external runner 路径
- 不要让文档和代码长期协议错位
- 不要在分析 pass 里用名字匹配当作 IR 语义恢复依据；名字只能用于日志、调试和实例展示，不能进入协议判断

---

## 当前推进顺序

1. 保持当前稳定基线不回退：
   - `ExecutableSpec -> rt_mod_blackhole -> BlackholeModule` direct host path
   - copy / GEMM current support surface
   - 第一批复杂 consumer 已打通的 compile-path 子集
2. 文档、任务安排和实现边界统一以 `tasks/dev_design/final_blackhole_backend_redesign.md` 为准
3. 当前 Stage 4 直接按分阶段文档执行，不再保留单一总 implementation plan 入口：
   - `tasks/dev_design/stage4_stage0_guardrails.md`
   - `tasks/dev_design/stage4_phase_a_semantic_ir.md`
   - `tasks/dev_design/stage4_phase_b_spatial_ir.md`
   - `tasks/dev_design/stage4_phase_c_tt_target_ir.md`
4. 当前执行重点：
   - `Phase A` 已完成
   - `Phase B`：contract-hardening 子阶段已完成，但整体仍未完成；
     当前主实施重点是把 `SpatialProgram` 做成正文定义的 execution-bearing spatial program，
     并收实对应的 synthesis algorithm / validator contract
   - `Phase C`：准备轨已落地，但正式 `TTProgram / MaterializeTTExecutableSpec`
     cutover 仍以前置的 `Phase B` 整体完成为前提
5. 在新分层下继续推进：
   - `Phase C2` 承接 `flash-attn` `blackhole.acc` correctness payoff
   - `topk / fusedmoe / paged decode / chunk recurrence` 等 family 在新主链下统一承接
   - 更宽 copy/dataflow 支持面
   - 更宽 synchronization 支持面

---

## 任务完成后必做

1. **更新进度** — `tasks/progress.md`（阶段状态、下一步）
2. **同步设计** — 受影响的设计文档不能落后于代码
3. **沉淀经验** — 稳定经验 → `memory/general_dev.md`，可复用问题 → `memory/bugs.md`
4. **明确未完成项** — 没做完就写没做完，不要假装不存在
5. **清理执行现场** — 在声明任务完成前，必须清理残留后台进程，并确认没有仍在运行的长命令
6. **提交** — `git commit` 后 `git push`（提交信息反映改动主题）

---

## 什么算完成

- 实现符合 `tasks/dev_design/final_blackhole_backend_redesign.md`
- 相关层协议一致
- 做了与任务匹配的验证
- 没有残留的后台测试/构建/长命令仍在运行
- 进度文档仍然真实
- 稳定经验和问题已同步到 `memory/`
- `git commit` 和 `git push` 已完成
- 未完成项和限制已明确写出

补充：

- “这一批改动已经通过验证”不算完成；它只是继续推进的前提，不是收口条件
- 只要阶段状态仍是未完成，或当前任务对应的提交/推送还没做完，就不能把
  checkpoint、阶段性汇报、阶段性绿测当成 stopping point

---

## 当前事实约束

- Blackhole 正式执行路径只剩 `BlackholeModule` 进程内 direct host path
- `tilelang.compile(..., execution_backend="tvm_ffi")` 的 Blackhole wrapper/export path 已恢复
- 默认开发构建目录固定为 `tilelang_repo/build/`
- 默认并行编译线程数按 `-j32` 执行
- `build_blackhole/` 和 legacy runner 都已删除
- `tasks/dev_design/` 根目录只保留活动文档；`tasks/dev_design/archive/` 下内容全部视为历史记录，不再作为当前入口
- `Phase A` 与 semantic manifest 已完成；`AnalyzeSemanticStructure` 已全面改成 manifest-first
- 当前 Blackhole 设备侧主链：
  `LowerDeviceStorageAccessInfo -> AugmentSemanticManifest -> LowerIntrin -> Simplify -> HoistBroadcastValues -> SplitBlackholeKernel -> AnalyzeBlackholeWorkDecomposition -> AnalyzeBlackholeFragmentRegions -> AnalyzeBlackholePipelineStages -> AnalyzeSemanticStructure -> LiftStatefulSemanticIR -> ValidateStatefulSemanticIR -> ValidateSemanticRefinement -> LowerToSpatialProgram -> ValidateSpatialProgram -> LowerBlackholeOps -> PlanBlackholeCB -> AssignBlackholeCores`
- `SplitBlackholeKernel` 已实现并已接入管线；纯 copy 走 `fused_dataflow` 单 kernel，GEMM 走 3-kernel（reader/compute/writer）
- direct runtime 当前正式支持面：
  - copy：equal source/dest range，且 stride = 1
  - GEMM：A/B-separated reader range + writer output range
  - accessor：仅 interleaved + DRAM + `common_runtime_arg_count = 0`
- `flash-attn` forward subset 当前已打通 compile-path；其 `blackhole.acc` correctness payoff 当前归属 `Phase C2`
- 当前总体架构 blocker 先是 `Phase B` 正文定义的 spatial synthesis 尚未完成，
  然后才是 `Spatial Program IR -> TT Target IR` 的单一真源切换
- 后续所有架构推进以 layered IR 为准：
  `Stateful Semantic IR -> Spatial Program IR -> TT Target IR`
