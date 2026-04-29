# AGENTS.md

本文件用于告诉 Codex 在这个仓库里应该如何工作。

总体设计只看一份：`tasks/dev_design/final_blackhole_backend_redesign.md`

任务级设计合同看：

- `tasks/dev_design/task0_ir_layering_root_cause.md`
- `tasks/dev_design/task1_spatial_plan_companion.md`
- `tasks/dev_design/task2_ttprogram_companion_cutover.md`
- `tasks/dev_design/task3_runtime_gate_and_workload_cutover.md`

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
2. `tasks/dev_design/task0_ir_layering_root_cause.md` — 根因与 rewrite 方向
3. `tasks/dev_design/task1_spatial_plan_companion.md` — `SpatialPlan` 主设计合同
4. `tasks/dev_design/task2_ttprogram_companion_cutover.md` — `TTProgram` 主设计合同
5. `tasks/dev_design/task3_runtime_gate_and_workload_cutover.md` — `ExecutableSpec / leaf reader` 主设计合同
6. `tasks/progress.md` — 当前 repo HEAD 状态与下一步
7. `tasks/dev_design/README.md` — 当前活动设计文档索引
8. `tasks/dev_design/2026-04-28-blackhole-algorithmic-generalization.md` — 当前 algorithmic decision-use cutover 设计合同
9. `tasks/dev_design/2026-04-28-blackhole-tile-compute-legalizer-dag-covering.md` — 后续 tile compute legalizer / covering gate 设计合同
10. 如果涉及构建/调试/历史问题，再读 `memory/general_dev.md` 和 `memory/bugs.md`
11. 然后读代码，不要只看文档

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
- Blackhole runtime / direct-runtime 的正式测试基线统一使用 `bf16` 输入；
  不要再把 `fp16` 当成当前 TT-Sim 上的 correctness gate
- TT-Sim `fp16` 路径命中的 `UntestedFunctionality` / simulator fatal taxonomy
  属于 simulator capability boundary；
  记录到 `memory/tt_simulator_constraints.md` / `memory/bugs.md`，不要把它表述成当前主任务 blocker

---

## 编码规则

**先设计后编码**（非小修复时强制）：

- 在 `tasks/dev_design/` 下先建或更新设计文档
- 设计文档至少说明：目标、影响范围、协议变化、验证方式
- 设计与实现冲突时，先更新设计再写代码

**代码结构原则**：

1. **分离 analysis 与 transformation**
   - analysis 只能从当前 IR 派生局部、可失效、可重算的结果；如果某个结果需要跨阶段保留，它就不能停留在 analysis 里，而必须进入 IR
   - transformation 只能读取当前 IR 和当前 pass 内的局部 analysis，并直接改写当前 IR 或显式构造下一层 IR
   - 一个 pass 如果一边旁路传协议、一边等后段补语义，说明它已经混乱了职责边界

2. **跨阶段中间状态必须进入显式 IR，不允许靠局部变量或 bag 隐式传递**
   - 任何需要跨阶段保留、被下游依赖、且不能在 analysis 失效后由当前 IR 重新推出的内容，都必须进入 IR 自身
   - 裸 `unordered_map` / `unordered_set` / `Map<String, Any>` 在多个 pass 之间流动，是隐藏语义通道的信号
   - pass 内的局部 mechanics 可以临时存在，但不能长成新的公共协议面

3. **用类型和显式 IR 字段捕获协议错误，不靠命名约定**
   - 跨模块共享的常量（attr key、field name、enum value）必须定义在一处，消费侧引用定义
   - 目的是把协议不一致从运行时 silent failure 提升到编译期 / validator 错误
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
   - 优先做统一 IR 语义、统一 attrs/类型表达、统一 pass 边界

3. **所需信息优先从 IR 分析；缺失就扩 IR/DSL，不要让后段猜**
   - 如果信息可以从 IR 得到，就必须从 IR 得到
   - 如果 IR 表达不够，就扩 IR / attrs / 类型，必要时从 DSL 显式表达

4. **IR 分析必须基于 IR 结构与类型，不允许基于名字匹配恢复语义**
   - 不允许用 buffer/var/op 的命名约定（如 `idx`、`scores_*`、`logsum`）来决定语义角色、绑定关系或协议分支
   - 语义恢复必须优先依赖 IR 自身可验证的信息：对象类型、storage scope、def-use、region/access pattern、loop-carried/dataflow 结构、显式 attrs / types
   - 如果仅靠当前 IR 结构仍无法稳定区分语义，就扩 IR / DSL / attrs / types；不要把名字匹配升级成长期分析手段

5. **整个编译器必须以显式 IR 为中心，不允许在 IR 之外再造语义通道**
   - 程序语义只能存在于当前 IR 层本身；analysis 只能是从当前 IR 派生出的、可失效可重算的临时结果
   - pass 只能在当前 IR 上做规范化、重写，或向下一层 IR 做显式 lowering；不允许靠 bag、payload、helper、wrapper、命名约定或其他旁路机制承载跨阶段语义
   - 任何需要跨阶段保留、被下游依赖、且不能在 analysis 失效后由当前 IR 重新推出的内容，都必须进入 IR 本身；如果当前 IR 表达不了，就先扩 IR，再继续实现

6. **设计边界必须写成 IR 层和显式表示对象，不能写成 pass 名字**
   - `SpatialPlan`、`TTProgram`、`ExecutableSpec` 这类表示层才是长期边界；`BuildSpatialPlanCompanion`、`PlanTTCompute`、`PlanTTTransport` 之类只是在当前代码里构造或改写这些表示的实现手段
   - pass 可以重命名、合并、拆分、内联；只要 IR 契约不变，架构就不应被描述成“某个 pass 拥有什么语义”
   - 任务文档里沿用的历史文件名只作为索引，不得被理解成新的 IR 名词或长期协议对象

7. **主线 task 默认按终态实现，不接受过渡式收口**
   - 除非对应 task 文档明确把某个 residue 写成允许保留的 narrow exception，否则不要保留旧 `wrapper / facts / bag / payload / public surface` 当“兼容壳”
   - 不允许把当前 repo HEAD / 当前 active chain / 当前残留实现当成继续保留旧面的授权；这些描述默认只是问题现状，不是目标形态
   - 不要把一个 task 拆成“这一轮先迁 owner truth，下一轮再删旧壳”的 transition-minded implementation；如果旧 pass / 旧 attr / 旧 object / 旧 Python wrapper 仍在当前 task 边界里活着，就默认这个 task 还没按原则完成

8. **硬件代码生成有效性准则**
   - 每个新设计对象、算法结构、pass、typed field 或 validator 都必须回答：它是否让 DSL 写出来的 kernel 更可靠或更高效地 lower 到真实 TT-Metal 硬件代码
   - 只构造对象、dump、shape-only check、metadata projection、测试覆盖或 paper-like algorithm name，都不算主线完成；它必须改变 leaf normalization、legality、typed plan、resource plan、admission diagnostic，或删除旧 matcher / payload / fallback / side channel
   - 如果当前只能说明“未来可能有用”，只能记录为 future candidate，不能进入 active-chain completion 或被当作当前任务完成依据

**对 Blackhole 的具体要求**：

- `runtime_args`、`buffer`、`cb`、`segment` 等绑定必须由 IR / attrs / typed fields 明确表达或可从 IR 稳定推导
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

1. 文档、任务安排和实现边界统一以当前入口文档为准：
   - `tasks/dev_design/final_blackhole_backend_redesign.md`
   - `tasks/progress.md`
   - `tasks/dev_design/README.md`
   - `tasks/dev_design/task0_ir_layering_root_cause.md`
   - `tasks/dev_design/task1_spatial_plan_companion.md`
   - `tasks/dev_design/task2_ttprogram_companion_cutover.md`
   - `tasks/dev_design/task3_runtime_gate_and_workload_cutover.md`
   - `tasks/dev_design/2026-04-28-blackhole-algorithmic-generalization.md`
   - `tasks/dev_design/2026-04-28-blackhole-tile-compute-legalizer-dag-covering.md`
2. 历史 cleanup 主线
   `Task 1 -> Task 2 -> Task 3 -> Legacy Protocol Deletion`
   已完成；
   它不是当前 active backlog
3. 当前 active lane
   统一看 `tasks/progress.md`；
   当前是
   `Algorithmic generalization Phase E: Decision-Use Cutover`
4. `tasks/progress.md`
   只维护 repo HEAD 当前状态 /
   blocker /
   下一步
5. cleanup `task0-task5`
   已完成并归档到
   `tasks/dev_design/archive/`
   - 它们只作完成期历史记录
   - 不再作为当前活动设计入口
   - 当前任务顺序只看
     `tasks/progress.md`
6. Algorithmic generalization
   Phase A-D
   已作为 foundation 完成：
   - `AccessRegion`
   - graph-backed `SpatialPlan` dependence
   - `LiveValueSSA`
   - 第一版 TT live-form solver
   当前 Phase E 必须让这些结构成为 active
   legality / query / action inputs，
   不能只停留在 typed object / dump / validator coverage
7. `materialization / source-live-form`
   已重新收束为
   `tasks/dev_design/2026-04-23-blackhole-live-form-materialization-admission.md`
   下的 support-surface admission lane，
   不再作为单独顶层路线
8. `TileComputeDAG` /
   legalizer /
   covering
   只有在 Phase E decision-use gate
   对 admitted compute surface 生效后才能进入 production migration

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
- 主线 task 默认按终态实现收口；除 task 文档明确允许的 narrow exception 外，
  不能把“owner truth 已迁走，但旧 wrapper / facts / bag / payload / public surface 还在”
  当作完成
- 对当前 task 边界，active chain、public API、测试断言应在同一轮一起切到终态；
  旧 pass / 旧 attr / 旧 object / 旧 Python wrapper 任一还活着，就不能报该 task 完成

---

## 当前事实约束

- Blackhole 正式执行路径只剩 `BlackholeModule` 进程内 direct host path
- `tilelang.compile(..., execution_backend="tvm_ffi")` 的 Blackhole wrapper/export path 已恢复
- 默认开发构建目录固定为 `tilelang_repo/build/`
- 默认并行编译线程数按 `-j32` 执行
- `build_blackhole/` 和 legacy runner 都已删除
- `tasks/dev_design/` 根目录只保留当前入口文档、
  当前活动 / 已完成但仍约束实现的 lane 设计文档
  和 protocol audit；
  `tasks/dev_design/archive/` 下内容全部视为历史记录，不再作为当前入口
- semantic manifest 路径已完成；`AnalyzeSemanticStructure` 已全面改成 manifest-first
- 当前实际 active chain 是：
  `Normalized Tile TIR -> BuildSpatialPlan -> ValidateSpatialPlan -> SplitBlackholeKernel -> PlanTTBlocks -> SelectBlackholeTTMetalBuiltins -> PlanTTCompute / PlanTTTransport / PlanTTSync / PlanTTABI / PlanTTExecution -> BuildTTProgram -> ValidateTTProgram -> MaterializeBlackholeExecutable -> runtime / codegen leaf readers`
- 上面这串名字描述的是当前 pass/phase 实现，不是新的长期 IR 层；
  长期主链仍然只有 `Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec`
- 当前协议边界事实：
  - `tl.blackhole_logical_buffer_tile_bridge_specs`
    已从 active code path 删除；
    不能再作为新的 bridge exception
    或长期边界重新引入
  - `compute_contract` / `gemm_contract` /
    `multi_*_contracts`
    已退出
    `TTProgram -> ExecutableSpec -> runtime`
    active chain；
    compute truth 只能经
    typed `TTComputeOpPlan`
    / `KernelSpec.compute_ops`
    流动
  - `blackhole.segment_kind`
    只允许作为
    `lower_blackhole_ops.cc`
    内部 pass-local mechanics，
    final IR / leaf reader 前必须剥离
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
- TT-Sim `fp16` 仍按 simulator capability boundary 处理，不作为当前 correctness gate
- 当前总体 blocker
  统一以 `tasks/progress.md`
  里的主线任务状态为准；
  当前首先要修复
  tile-compute source lowering
  中的 composite pseudo-leaf residue：
  `exp2_tile(mode, lhs, rhs, scale, ...)`
  和
  `mul_tiles_bcast_cols("div", ...)`
  不能继续作为 leaf-looking payload
  或 source hook expansion
  存在；
  相关 TIR 计算必须在
  `Normalized Tile TIR`
  中显式分解成 TT-Metal leaf op sequence，
  再交给
  `TileComputeDAG`
  做 leaf covering
- Algorithmic generalization
  必须同时遵守：
  - anti-overdesign pay-rent rule：
    新结构必须改变 legality /
    query /
    typed plan /
    unsupported diagnostic
    或删除旧 matcher/helper/payload/fallback
    但不能为了 pay rent
    把 analysis / DAG
    变成 composite lowering owner
  - problem-family generality rule：
    当前 workload case
    只能作为 active-chain witness，
    不能成为协议定义
- 后续所有架构推进以当前 layered IR 为准：
  `Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec`
