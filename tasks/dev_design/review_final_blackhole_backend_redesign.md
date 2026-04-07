# Review: TileLang Blackhole 后端重设计总文档

## 基本信息

- **Review 日期**: `2026-04-05`
- **被 Review 文档**: `tasks/dev_design/final_blackhole_backend_redesign.md`
- **角色**: 设计审计快照，不承担当前阶段状态跟踪
- **当前状态入口**: `tasks/progress.md` 与各阶段文档

## 1. 核心结论

这次审计的核心判断没有变：

- 三层 IR 分离方向是正确的
- 现有代码事实能够支撑“semantic recovery / spatial organization / target planning
  不能继续混在一层里”这个总论点
- 设计的主要风险不在方向，而在工程落地难度与对象复杂度

## 2. 这份审计确认过的优点

- 总设计没有退化成 case-by-case matcher registry
- `Domain / State / Update` 与小闭集 trait 轴的方向是对的
- 设计与当前 Blackhole pass 事实基本对齐
- `TTProgram / ABI / CB / transport / sync` 的 target object 化方向与 TT-Metal 模型一致
- compatibility deletion gate 的思路是对的，避免“以后再删”的长期悬挂状态

## 3. 审计时确认过的关键代码事实

- `LowerBlackholeOps` 同时承担语义恢复与 target lowering，属于混层实现
- `ExecutableSpec` 当前仍被多方共同写入，不是单一真源
- `blackhole.acc` 的混义问题是结构性问题，不是局部 emitter bug
- Blackhole 路径在 `OptimizeForTarget` 里已经明显早分叉，不是“通用管线最后几步 codegen”
- TT-Metal 侧确实存在 compile-time / common-runtime / per-work 三层 ABI、
  CB、dst、semaphore、NoC transport 等稳定 target contract

## 4. 仍有价值的风险提示

### 4.1 P0: semantic recovery 难度不能低估

这是当时最大的落地风险，今天看结论仍成立：

- 后段 IR 已经被明显压碎
- 对 copy / GEMM 之外的 family，post-lowering semantic recovery 难度显著更高
- 因此“尽量前移 evidence capture，而不是纯靠 late matcher”仍然是正确方向

### 4.2 P1: 对象数量与实现复杂度要强控

当时的判断仍然有效：

- companion object 数量很容易失控
- 只有真正进入长期协议的对象才值得 object 化
- internal helper 和阶段性 analysis state 不应无条件升级成长期 typed schema

### 4.3 P1: rebind contract 要务实

这条风险也仍然成立：

- semantic lift 之后，companion truth 的生命周期必须被严格约束
- `typed_rebind` 只能在审计过的安全边界内使用
- 不能默认“所有后续 TIR 变换都能无损保活 companion program”

### 4.4 P2: 示例不能变 hidden spec

文档里的 workload 例子只能是说明，不是 matcher 输入。
这条纪律今天仍然重要。

## 5. 后续响应情况

审计时提出的 `R1-R11` 建议已经被后续文档和实现逐步吸收。
今天最值得保留的，不是当时每一条建议的展开，而是下面这些方向性结论：

- 能前移保存的语义证据就前移，不要全压给 late recovery
- 总设计与阶段文档要分工，不能让总纲背 backlog
- companion lifecycle 必须显式化
- `ProgramPhase` 等跨函数 truth 需要 module-scope 宿主
- `TTHardwareModel` 应收拢散落常量

## 6. 现在如何使用这份文档

这份审计文档现在只适合用来回答三类问题：

1. 为什么总设计要分三层
2. 当时审计确认过哪些代码事实
3. 哪些工程风险值得持续记住

它不再承担：

- 当前阶段状态汇报
- 当前 blocker 判断
- 当前完成条件定义
- 当前验证快照
