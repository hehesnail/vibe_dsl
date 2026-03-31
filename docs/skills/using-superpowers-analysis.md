# `using-superpowers` Skill 源码式拆解

## 目的

这份文档把 `using-superpowers` 当成“流程代码”来读，不把它当普通说明文。

目标不是复述它说了什么，而是提炼：

- 每个片段在解决什么问题
- 为什么要这样写
- 写新 skill 时可以直接复用什么手法

相关来源：

- `/root/.codex/superpowers/skills/using-superpowers/SKILL.md`
- `/root/.codex/superpowers/skills/writing-skills/SKILL.md`
- `/root/.codex/skills/.system/skill-creator/SKILL.md`

---

## 先给结论

`using-superpowers` 不是知识型 skill，而是一个元调度 skill。

它的职责只有一件事：

- 在任何任务开始前，强制先做 skill 选择与启用

它优秀的地方不在“信息多”，而在：

- 触发条件清楚
- 适用边界清楚
- 顺序约束清楚
- 反模式清楚
- 优先级清楚

从 skill 写作角度看，它最大的价值是展示了如何把“原则”写成“不可误解的执行约束”。

---

## 片段 1：Frontmatter

```yaml
---
name: using-superpowers
description: Use when starting any conversation - establishes how to find and use skills, requiring Skill tool invocation before ANY response including clarifying questions
---
```

### 这一段在做什么

它定义了 skill 的检索入口。

其中真正关键的是 `description`，因为未来 agent 通常先看到的是 metadata，而不是正文。

### 为什么这样写

这段写法有 3 个优点：

1. 先写触发时机：`starting any conversation`
2. 再写关键限制：`before ANY response`
3. 不把完整流程塞进 description

### 你写 skill 时该学什么

- `description` 只写“什么时候该用”
- 不写“这个 skill 很有帮助”
- 不写完整流程摘要

### 对照

差的写法：

```yaml
description: Helps use skills better
```

好的写法：

```yaml
description: Use when debugging flaky tests, intermittent CI failures, or timing-dependent behavior
```

---

## 片段 2：`<SUBAGENT-STOP>`

```md
<SUBAGENT-STOP>
If you were dispatched as a subagent to execute a specific task, skip this skill.
</SUBAGENT-STOP>
```

### 这一段在做什么

它在定义“不适用边界”。

### 为什么这样写

如果没有这段，skill 很可能在子 agent 上反复触发，导致：

- 无限套娃
- 总调度和执行 worker 混在一起
- 子 agent 失去任务聚焦

### 你写 skill 时该学什么

一个成熟 skill 不只要写：

- 什么时候用

还要写：

- 什么时候不要用
- 哪些上下文下必须跳过

如果你的 skill 只写适用条件、不写排除条件，后续很容易误触发。

---

## 片段 3：`<EXTREMELY-IMPORTANT>`

```md
If you think there is even a 1% chance a skill might apply ...
```

### 这一段在做什么

它在降低模型自由度，防止“先做了再说”。

### 为什么这样写

模型天然倾向于：

- 觉得问题太简单
- 先看点文件
- 先问点问题
- 先做一步小动作

这段文字本质上是在改默认策略：

- 不是“除非必要才用 skill”
- 而是“只要可能相关，就先检查和启用 skill”

### 你写 skill 时该学什么

如果你的 skill 是纪律型、流程型、错误代价高，就不要只写温和建议。

要敢于写成：

- `You MUST ...`
- `Do not ... until ...`
- `This is not optional`

这类文字不是为了“语气强”，而是为了阻止 agent 合理化偷懒。

---

## 片段 4：`Instruction Priority`

```md
1. User's explicit instructions
2. Superpowers skills
3. Default system prompt
```

### 这一段在做什么

它在处理规则冲突。

### 为什么这样写

skill 往往会引入新的强流程。如果不声明优先级，模型可能：

- 把 skill 看得比用户指令还高
- 把 skill 当成绝对法则
- 在冲突时做出错误取舍

### 你写 skill 时该学什么

只要你的 skill 可能和下面任一项冲突，就应该显式写优先级：

- 用户要求
- 仓库规则
- 系统工作流
- 其他 skill

更好的写法是像这里一样，不只给顺序，还给冲突例子。

---

## 片段 5：`How to Access Skills` / `Platform Adaptation`

### 这一段在做什么

它在做运行环境适配。

### 为什么这样写

同一个 skill 可能运行在不同 agent 环境里：

- Claude Code
- Codex
- Gemini CLI

如果正文直接绑死某个平台，skill 会很快退化成“只能在作者那台机器上工作”的私货。

### 你写 skill 时该学什么

如果 skill 依赖工具入口、平台命令或特定目录结构，可以写最小必要的适配层，但要克制：

- 给导航，不给冗长手册
- 指向参考，而不是把参考全贴进正文

这符合 `skill-creator` 里的 “progressive disclosure”。

---

## 片段 6：总规则 + Flowchart

```md
Invoke relevant or requested skills BEFORE any response or action.
```

然后是 flowchart。

### 这一段在做什么

它把 skill 定义成一个状态机。

### 为什么这样写

流程型 skill 最怕散文叙述，因为散文容易：

- 漏步骤
- 打乱顺序
- 把条件分支写模糊

而 flowchart 天然表达：

- 入口
- 判断点
- 分支
- 终点

### 你写 skill 时该学什么

如果 skill 本质是“按顺序执行的判断流程”，优先考虑：

- 流程图
- checklist
- numbered state transitions

不要只写段落。

### 实际上这个图表达了什么状态机

`using-superpowers` 的内在状态机是：

1. 收到请求
2. 检查 skill 是否可能适用
3. 若适用，先启用
4. announce 当前使用的 skill
5. 若 skill 有 checklist，则生成任务列表
6. 严格按 skill 执行
7. 最后才响应或行动

这也是你设计流程型 skill 时该先在脑子里画出来的东西。

---

## 片段 7：`Red Flags`

### 这一段在做什么

它在枚举模型最常见的绕路方式。

例如：

- “This is just a simple question”
- “Let me explore first”
- “I already know this skill”

### 为什么这样写

一个 skill 失败，往往不是因为 agent 看不懂，而是因为 agent 会自我合理化：

- 觉得这次例外
- 觉得先做一步没关系
- 觉得自己已经记得 skill 内容

这段就是提前把这些借口列出来，形成防火墙。

### 你写 skill 时该学什么

如果你要写可执行 skill，几乎都建议加一节：

- `Red Flags`
- `Common Rationalizations`
- `Failure Modes`

这会比继续扩正向说明更有效。

### 一个实用问题

写 skill 时可以直接问自己：

- 模型最可能在哪一步偷懒？
- 哪 3 个借口最常见？
- 哪些“看起来很合理”的想法其实是在绕规则？

把答案写出来，就是这一节。

---

## 片段 8：`Skill Priority` / `Skill Types`

### 这一段在做什么

它在解决 skill 系统内部的调度问题。

### 为什么这样写

现实里经常多个 skill 同时命中，例如：

- 一个是流程 skill
- 一个是具体实现 skill

如果不写优先级，agent 很可能直接冲进实现，跳过前置流程。

这里定义的是：

- 先 process skill
- 再 implementation skill

并且额外区分：

- rigid skill
- flexible skill

### 你写 skill 时该学什么

不要让所有规则强度一样。

最好显式区分：

- 哪些规则不能变通
- 哪些规则可以按上下文调整
- 哪些只是推荐

这能让 agent 在复杂场景里更稳定。

---

## 从这份 skill 提炼出来的写作原则

### 1. 先写触发条件，再写正文

skill 能不能被正确用到，往往取决于 frontmatter，不取决于正文文采。

### 2. 一个 skill 最好只接管一个关键决策点

`using-superpowers` 只接管一件事：

- “先判断和启用 skill”

这让它特别稳定。

### 3. 原则必须转成顺序约束

不要只写：

- “最好先……”

更有效的是：

- `Before X, do Y`
- `Do not Z until ...`

### 4. 必须主动写反模式

好 skill 不只是教程，也是漏洞修补器。

### 5. 用状态机表达流程

流程 skill 用 flowchart，通常比 500 字说明更靠谱。

### 6. 强弱规则分层

写清楚：

- MUST
- SHOULD
- MAY

或者：

- rigid
- flexible

---

## 你以后写 skill 时最值得直接复用的骨架

```md
---
name: your-skill
description: Use when [具体触发条件]
---

# Skill Name

## Overview
一句话说明它接管什么决策点。
一句话说明核心原则。

## Hard Rule
列出 1-3 条不能绕过的规则。

## Instruction Priority
如果可能冲突，写清优先级。

## Workflow
写状态流、checklist 或 flowchart。

## Red Flags
列出最常见的偷懒方式和合理化借口。

## Common Mistakes
列典型误用。

## Output Contract
如果有固定产物，写清楚必须产出什么。
```

---

## 最后一句

如果把普通 skill 看成“说明文”，你会写出很多正确但不稳定的东西。

如果把 skill 看成“给 agent 的流程代码”，你就会自然去写：

- 触发器
- 边界
- 状态机
- 守卫条件
- 反模式

`using-superpowers` 值得学的，正是这一点。
