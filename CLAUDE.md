# CLAUDE.md

本文件是 Claude Code 在这个仓库里的工作规范。

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
3. 如果涉及构建/调试/历史问题，再读 `memory/general_dev.md` 和 `memory/bugs.md`
4. 然后读代码，不要只看文档

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

**对 Blackhole 的具体要求**：

- `runtime_args`、`buffer`、`cb`、`segment` 等绑定必须由 IR/schema 明确表达或可从 IR 稳定推导
- 不要为了绕开当前卡点新增并行执行路径或额外 emitter；优先修主路径
- 修问题时优先收正协议和主链，而不是堆 workaround

**不要做的事**：

- 不要新增第二份总体设计文档
- 不要把单个 kernel 字符串当成后端主产物
- 不要重新引入或扩展 legacy external runner 路径
- 不要让文档和代码长期协议错位

---

## 当前推进顺序

1. ~~attrs / 协议~~ ✅
2. ~~`ExecutableSpec`~~ ✅
3. ~~`rt_mod_blackhole`~~ ✅
4. ~~`BlackholeModule` direct path 补全~~ ✅
5. ~~Copy E2E 验收（direct path）~~ ✅
6. ~~split-before 语义规划~~ ✅
7. ~~通用 pass 回收~~ ✅（FlattenBuffer/VectorizeLoop 已验证；StorageRewrite 永久排除）
8. ~~GEMM 接入 Steps 1-5~~ ✅（CB identity 唯一协议已收正）
9. **GEMM E2E 验收** — CB 同步修复 → 数值验证 → schema 收正（设计见 `tasks/dev_design/2026-03-26-stage2d-gemm-contract-implementation-plan.md`）
10. multi-core

---

## 任务完成后必做

1. **更新进度** — `tasks/progress.md`（阶段状态、下一步）
2. **同步设计** — 受影响的设计文档不能落后于代码
3. **沉淀经验** — 稳定经验 → `memory/general_dev.md`，可复用问题 → `memory/bugs.md`
4. **明确未完成项** — 没做完就写没做完，不要假装不存在
5. **提交** — `git commit` 后 `git push`（提交信息反映改动主题）

---

## 什么算完成

- 实现符合 `tasks/dev_design/final_blackhole_backend_redesign.md`
- 相关层协议一致
- 做了与任务匹配的验证
- 进度文档仍然真实
- 稳定经验和问题已同步到 `memory/`
- `git commit` 和 `git push` 已完成
- 未完成项和限制已明确写出

---

## 当前事实约束

- Blackhole 正式执行路径只剩 `BlackholeModule` 进程内 direct host path
- 默认开发构建目录固定为 `tilelang_repo/build/`
- 默认并行编译线程数按 `-j32` 执行
- `build_blackhole/` 和 legacy runner 都已删除
- Pass 管线顺序：`AnnotateBlackholeCopySemantics` → `BlackholeDeviceResourceCanonicalization` → `SplitHostDevice` → `SplitBlackholeKernel` → `LowerBlackholeOps` → `PlanBlackholeCB`
- `SplitBlackholeKernel` 已实现并已接入管线；纯 copy 走 `fused_dataflow` 单 kernel，GEMM 走 3-kernel（reader/compute/writer）
