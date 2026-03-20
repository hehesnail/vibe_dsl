# CLAUDE.md

本文件是 Claude Code 在这个仓库里的工作规范。

架构设计只看一份：`tasks/dev_design/final_blackhole_backend_redesign.md`

---

## 仓库结构

- `tilelang_repo/` — TileLang 开发仓库，Blackhole 后端代码主要改这里
- `tt_metal_repo/` — TT-Metal 开发仓库，runner、示例、API 参考主要看这里
- 顶层仓库 — 任务文档、经验记录、测试、脚本

常用目录：

- `tilelang_repo/src/target/` — Blackhole 模块、codegen
- `tilelang_repo/src/transform/` — Blackhole passes
- `tilelang_repo/tilelang/engine/` — Python 编译链路
- `tilelang_repo/tools/blackhole_runner/` — External runner（参考蓝本，不改）
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

**不要做的事**：

- 不要新增第二份总体设计文档
- 不要把单个 kernel 字符串当成后端主产物
- 不要把 `SplitBlackholeKernel` 当成当前前置条件
- 不要把 external runner 当成正式执行路径
- 不要让文档和代码长期协议错位

---

## 当前推进顺序

1. ~~attrs / 协议~~ ✅
2. ~~`ExecutableSpec`~~ ✅
3. ~~`rt_mod_blackhole`~~ ✅
4. ~~`BlackholeModule` direct path 补全~~ ✅ Phase 1 代码完成
5. Copy E2E 验收（direct path）— **当前首要**
6. split-before 语义规划（方案 A: `AnnotateBlackholeCopySemantics` pass）
7. 通用 pass 回收（FlattenBuffer / VectorizeLoop / StorageRewrite）
8. GEMM 接入
9. multi-core

关键参考：`runner.cpp` 是 direct path 的完整参考蓝本。

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

## memory 系统

持久记忆存放在 `/root/.claude/projects/-root-dev-vibe-dsl/memory/`，包含：

- 用户偏好与协作风格
- 已验证的开发模式
- 项目上下文（当前阶段、关键决策）
- 外部资源指针

每次工作中发现新的稳定经验时，主动更新 memory。
