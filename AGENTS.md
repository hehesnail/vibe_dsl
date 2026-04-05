# AGENTS.md

本文件用于告诉 Codex 在这个仓库里应该如何工作。

架构设计只看一份：`tasks/dev_design/final_blackhole_backend_redesign.md`

---

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

---

## 任务完成后必做

1. **更新进度** — `tasks/progress.md`（阶段状态、下一步）
2. **同步设计** — 受影响的设计文档不能落后于代码
3. **沉淀经验** — 稳定经验 → `memory/general_dev.md`，可复用问题 → `memory/bugs.md`
4. **明确未完成项** — 没做完就写没做完，不要假装不存在
5. **清理执行现场** — 可以并行跑多个测试/构建，但在声明任务完成前，必须清理残留后台进程，并确认没有仍在运行的长命令；否则不能把任务表述为“已完成”
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

---

## 当前事实约束

- Blackhole 正式执行路径只剩 `BlackholeModule` 进程内 direct host path
- `tilelang.compile(..., execution_backend="tvm_ffi")` 的 Blackhole wrapper/export path 已恢复
- 默认开发构建目录固定为 `tilelang_repo/build/`
- 默认并行编译线程数按 `-j32` 执行
- `build_blackhole/` 和 legacy runner 都已删除
- `tasks/dev_design/` 根目录只保留活动文档；`tasks/dev_design/archive/` 下内容全部视为历史记录，不再作为当前入口
- 当前 Blackhole 设备侧 pass 主线：
  `LowerDeviceStorageAccessInfo` → `LowerIntrin` → `Simplify` → `HoistBroadcastValues` → `SplitBlackholeKernel` → `AnalyzeBlackholeWorkDecomposition` → `AnalyzeBlackholeFragmentRegions` → `AnalyzeBlackholePipelineStages` → `LowerBlackholeOps` → `PlanBlackholeCB` → `AssignBlackholeCores`
- `SplitBlackholeKernel` 已实现并已接入管线；纯 copy 走 `fused_dataflow` 单 kernel，GEMM 走 3-kernel（reader/compute/writer）
- direct runtime 当前正式支持面：
  - copy：equal source/dest range，且 stride = 1
  - GEMM：A/B-separated reader range + writer output range
  - accessor：仅 interleaved + DRAM + `common_runtime_arg_count = 0`
- `flash-attn` forward subset 当前已完成 analysis、最小 fragment/dataflow builtin/codegen 接入，并打通当前支持的 MHA/GQA forward compile-path；runtime hang 已解，当前主 blocker 是 `blackhole.acc` 混合语义导致的 compute correctness 问题
- 总设计的目标不再局限于 `flash-attn`：后续实现需要同时面向 selection/indexing、routed/grouped dispatch、paged decode、chunk recurrence 等 workload family
- 后续所有架构推进以 layered IR 为准：
  `Stateful Semantic IR -> Spatial Program IR -> TT Target IR`
- TT-Sim 当前正式环境入口是顶层 `scripts/setup_tt_sim.sh`
- `setup_tt_sim.sh` 与后续测试必须在同一个 shell 中执行
- 如果在 worktree 中运行测试，source 后必须把 `TILELANG_HOME` 指回当前 checkout/worktree
