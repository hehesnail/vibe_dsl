# Stage 4: Semantic Manifest

## 基本信息

- **文档角色**: `Phase A` 信息源重构初设
- **当前状态**: 设计中
- **唯一总体设计**: `tasks/dev_design/final_blackhole_backend_redesign.md`
- **直接前置**: `tasks/dev_design/stage4_phase_a_semantic_ir.md`

## 1. 定位

这份文档只讨论一件事：

- 怎样重构 `Phase A` 的 **pre-lift evidence input**

它**不**讨论：

- 重定义 semantic layer
- 把 semantic recovery 挪出 `Phase A`
- 给 `Phase B / C` 新增 shortcut 输入
- 给 `LowerBlackholeOps` 设计新 lowering contract

总设计和 `Phase A` 文档里的前提保持不变：

1. 算法语义真源仍然只属于 `Stateful Semantic IR`
2. `SemanticProgram` 仍然是唯一冻结后的语义产物
3. `AnalyzeSemanticStructure -> LiftStatefulSemanticIR -> Validate*`
   - 仍然是 semantic recovery 的主体
4. manifest 如果存在，只能是 `Phase A` 上游的 evidence carrier

这里最重要的是区分两条边界：

| 边界 | 含义 | 本文是否改变 |
|---|---|---|
| ownership boundary | 谁拥有算法语义真相 | 不改变，仍然只属于 `Stateful Semantic IR` |
| capture boundary | 某类 evidence 需要在哪个 lower 边界前先保住 | 这是本文要解决的事 |

## 2. 结论先行

最后结论可以先压成这张表：

| 问题 | 结论 |
|---|---|
| semantic recovery 属于哪一层 | 仍然只属于 `Phase A / Stateful Semantic IR` |
| manifest 是什么 | `Phase A` 的 pre-lift evidence input |
| manifest 能不能替代 `AnalyzeSemanticStructure` | 不能 |
| manifest 能不能直接生成 `SemanticProgram` | 不能 |
| manifest 第一阶段要解决什么 | 把 explicit-op evidence 从 late matcher 路径中切出来，稳定输入给 `Phase A` |
| manifest 第一阶段不解决什么 | `selection_pairs / arg_reduce_targets / recurrence_edges` 这类 residual structural evidence 仍不应被强行塞进 manifest |
| 为什么要看完整 lower 链 | 因为 evidence 会在不同边界被销毁，capture 点不能凭感觉定 |

## 3. 问题定义

当前真正的问题不是：

- `AnalyzeBlackholeFragmentRegions` 写得还不够多

而是：

- `Phase A` 需要的一部分 evidence 在 compiler IR 中本来是显式的
- 这些 evidence 会在 lower 链的不同边界被销毁
- 当前却主要依赖 `fragment_regions` 在 lowered 结构上回收这些事实

于是形成了现在这条路：

```text
concrete IR structure
  -> generic fragment/pattern analysis
  -> fragment_regions
  -> AnalyzeSemanticStructure
  -> witnesses
  -> SemanticProgram
```

这条路的问题不在于“恢复”这个动作本身，而在于：

1. 有些 evidence 根本不该等到 lowered 之后再猜
2. `fragment_regions` 同时背 semantic evidence 和 target-facing lowering requirement，职责已经混在一起
3. 每扩一个新 family，都容易继续往 `fragment_regions` 里堆 matcher

`stage4_phase_a_semantic_ir.md` 已经把 `Phase A` 的信任边界写得很清楚：

- `Phase A` 只保证：
  - 给定上游 evidence，witness/core/validator 正确
- `Phase A` 不保证：
  - 上游 evidence 缺失时自己发明出正确语义

因此这份文档要解决的不是 “Phase A 要不要恢复语义”，而是：

> `Phase A` 应该消费什么 evidence，以及这些 evidence 应该在哪些边界前被稳定保留下来。

## 4. 与总设计和 Phase A 的关系

### 4.1 总设计约束

根据 `final_blackhole_backend_redesign.md`：

- `Semantic Recovery -> Stateful Semantic IR`
  - 交付的是 `Domain / State / Update / AccessMap / UpdateLaw` 事实
- `Stateful Semantic IR`
  - 不能泄漏 task/layout/sync/placement/ABI

所以 manifest 如果存在，必须满足：

1. 它只服务 `Semantic Recovery -> Stateful Semantic IR`
2. 它本身不是 semantic core
3. 它不能成长为第二套长期 schema

### 4.2 Phase A 约束

根据 `stage4_phase_a_semantic_ir.md`：

- `Phase A` 的长期对象只有：
  - `SemanticProgram`
  - `Domain`
  - `State`
  - `Update`
  - `UpdateLaw`
  - `AccessMap`
  - `SemanticSupplement`
- `selection_targets / selection_pairs / arg_reduce_targets / update_sources / recurrence_edges`
  - 只能作为 pre-lift evidence，不能成长为第二套 schema

因此 manifest 也必须遵守同一纪律：

- 它可以承接 evidence
- 但不能直接越级变成 semantic core

## 5. 当前代码事实

这里只列和 evidence capture 直接相关的事实，不展开无关实现细节。

### 5.1 capture boundary 不是一个点

相关 evidence 在当前主链上会在不同位置消失：

| 边界 | 当前代码事实 | 对 manifest 的含义 |
|---|---|---|
| `LowerTileOp` | `copy / fill / reduce / cumsum` 这类 tileop 会在这里被 lowered 掉 | 这部分 evidence 不能等到后面再收 |
| `SplitHostDevice` | 会创建新的 device `PrimFunc`，不会自动保留任意 pre-split attr | 早采集之后必须有显式 projection |
| device-side pre-`LowerIntrin` | 仍能看到少量 residual explicit op，例如 `gemm_py` | 仍需要一个晚边界 augment |

这就是为什么 collector 不可能只是一个单点 pass。

### 5.2 当前 Phase A 真实消费什么

当前 `AnalyzeSemanticStructure` 直接或间接消费：

| 输入 | 当前职责 |
|---|---|
| `blackhole.work_decomposition` | domain skeleton |
| `blackhole.pipeline_stages` | pipeline trait |
| `blackhole.fragment_regions` | fragment evidence 与 residual structural evidence |
| `tl.semantic_seeds` | pre-lift typed seed |

这里最大的问题不是 “它消费太多输入”，而是：

- 本该更早保住的 explicit-op evidence
- 现在也被迫走 `fragment_regions` 那条恢复路径

### 5.3 当前不该动的东西

下面这些前提在 manifest 文档里不能被写歪：

| 项 | 当前结论 |
|---|---|
| semantic recovery 主体 | 仍是 `AnalyzeSemanticStructure -> Lift -> Validate` |
| `LowerBlackholeOps` | 当前仍可继续读现有 target-facing attrs |
| `Phase B / C` | 仍只消费冻结后的 companion IR，不直接读 manifest |

## 6. 设计目标

这份初设只追求下面四个目标：

1. 把 explicit-op evidence 从 `fragment_regions` 的 case-by-case 路径中切出来
2. 把 capture 点放在 evidence 真正被销毁之前
3. 让这些 evidence 仍然统一进入 `AnalyzeSemanticStructure`
4. 不改变 `Phase A` 的 ownership boundary

对应地，这份设计**不**追求：

1. 第一步就删除 `AnalyzeBlackholeFragmentRegions`
2. 第一步就让 `LowerBlackholeOps` 改成消费 `SemanticProgram`
3. 第一步就把所有 residual structural evidence 全部 manifest 化
4. 第一步就改写 `Phase B / C` 接口

## 7. 方案总述

### 7.1 正确的总体说法

正确说法不是：

- “把 semantic recovery 提前到 `LowerTileOp` / split / device pipeline`”

正确说法是：

- `Phase A` 的 semantic recovery 主体保持不变
- 只是把 `Phase A` 上游的一部分 evidence source 改成更稳定的 capture + projection + augment

### 7.2 manifest 的角色

manifest 的唯一角色是：

> 在 semantic recovery 发生之前，把 explicit-op evidence 从更稳定的 capture 边界带到 `AnalyzeSemanticStructure`。

因此 manifest 的职责表如下：

| 项 | manifest 是否负责 |
|---|---|
| 保留 explicit-op evidence | 负责 |
| 统一挂到 device `PrimFunc` | 负责 |
| 生成 witness | 不负责 |
| 归约成 `State / Update / UpdateLaw` | 不负责 |
| 冻结成 `SemanticProgram` | 不负责 |

## 8. truth ownership 表

这一版文档最核心的表是这张：

| 通道 | 它承接什么 | 它不承接什么 | 主要消费者 |
|---|---|---|---|
| `tl.semantic_seeds` | pre-lift typed seed，device program / pipeline skeleton | explicit-op payload | `AnalyzeSemanticStructure` |
| `tl.semantic_manifest_seeds` | 会在 early lowering 被销毁的 explicit-op evidence | semantic core | `ProjectSemanticManifest` |
| `tl.semantic_manifest` | 投影并补全后的 explicit-op evidence | witness / `SemanticProgram` | `AnalyzeSemanticStructure` |
| `blackhole.fragment_regions` | residual structural evidence + 当前兼容职责 | 长期 semantic truth owner | `AnalyzeSemanticStructure`, `LowerBlackholeOps` |
| `tl.semantic_witnesses` | 统一 witness axis | raw lowering payload | `LiftStatefulSemanticIR`, validator |
| `tl.semantic_program` | 冻结后的算法语义 | raw analysis attr | `Phase B` |

这张表的关键结论是：

- manifest 只是在 `fragment_regions` 和 `semantic_witnesses` 之间补进一个更稳定的 evidence carrier
- 它不是 `semantic_program` 的并行真源

## 9. evidence 分类

manifest 第一阶段到底该管什么，可以直接按 evidence 类型来分：

| evidence 类型 | 例子 | 第一阶段是否进 manifest | 原因 |
|---|---|---|---|
| explicit-op payload | `copy / fill / reduce / cumsum / gemm_py` | 是 | 这类事实在某些 lower 边界前是显式的，不该靠晚期 matcher 反推 |
| domain skeleton | work axes / derived indices | 否 | 已有 `work_decomposition`，本来也不属于 manifest |
| pipeline skeleton | stage trait | 否 | 已有 `pipeline_stages` / `semantic_seeds` |
| residual structural evidence | `selection_pairs / arg_reduce_targets / recurrence_edges` | 第一阶段否 | 当前仍需要结构恢复，不能假装已经显式化 |
| target-facing lowering requirement | CB / segment / runtime lowering 需求 | 否 | 不属于 `Phase A` evidence carrier |

这也是为什么这份设计不能写成 “把所有 semantic facts 都并入 manifest”。

## 10. capture 方案

### 10.1 不是单点 collector

完整可行方案是三段式：

| 阶段 | 作用 | 产物 |
|---|---|---|
| early capture | 在 `LowerTileOp` 前保住会被 early lowering 吃掉的 explicit-op payload | `tl.semantic_manifest_seeds` |
| projection | 在 `SplitHostDevice` 后把 early seeds 投影到对应 device `PrimFunc` | `tl.semantic_manifest` |
| late augment | 在 device-side pre-`LowerIntrin` 补上仍显式存在的 residual op payload | 增补后的 `tl.semantic_manifest` |

### 10.2 early capture

第一阶段 early capture 只针对：

- `copy`
- `fill`
- `reduce`
- `cumsum`
- 被 early lowering 吃掉的 `gemm`

它只做：

1. 记录 op payload
2. 记录必要 buffer metadata
3. 记录最小 ordered-loop / region anchor

它不做：

1. companion 推断
2. recurrence 推断
3. witness 生成
4. semantic core 归约

### 10.3 projection

projection 必须显式设计，不能默认 attrs 会跟着走。

最低要求是：

| 要求 | 原因 |
|---|---|
| early seeds 带稳定 region anchor | 否则 split 后无法投影 |
| projection 基于 split 稳定顺序 | 不能靠名字匹配 |
| 每个 device `PrimFunc` 只拿到属于自己的 evidence slice | 避免 manifest 漂浮成 module-wide 杂项 |

### 10.4 late augment

late augment 只做一件事：

- 把在 device-side pre-`LowerIntrin` 仍显式存在的 explicit-op payload 补进同一个 manifest

第一阶段典型对象是：

- `gemm_py`

这里仍然只是 evidence augment，不是 semantic recovery。

## 11. manifest schema

manifest schema 应保持最小，只表达 explicit-op evidence 本身。

### 11.1 建议结构

| 字段 | 含义 |
|---|---|
| `buffers` | manifest 内引用的 buffer descriptor |
| `operations` | explicit-op payload 列表 |
| `ordered_regions` | 最小 ordered/serial region anchor |
| `anchors` | split / region 级稳定锚点 |

### 11.2 明确不要塞进去的东西

manifest 第一阶段不应直接塞：

- `selection_pairs`
- `arg_reduce_targets`
- `recurrence_edges`
- `State.role`
- `UpdateLaw.kind`
- `paired_value_state`

因为这些都已经不再是 raw evidence，而是在往 witness/core 滑。

## 12. `AnalyzeSemanticStructure` 怎么接 manifest

`AnalyzeSemanticStructure` 的职责不变，只是输入面收正。

### 12.1 第一阶段输入

第一阶段它统一读取：

| 输入 | 用途 |
|---|---|
| `tl.semantic_manifest` | explicit-op evidence |
| `blackhole.fragment_regions` | residual structural evidence |
| `blackhole.work_decomposition` | domain skeleton |
| `blackhole.pipeline_stages` | pipeline trait |
| `tl.semantic_seeds` | pre-lift typed seed |

### 12.2 第一阶段输出

输出仍然是：

- `tl.semantic_structure`
- `tl.semantic_witnesses`
- `tl.semantic_program`

也就是说，第一阶段真正变化的是：

| 项 | 变化 |
|---|---|
| explicit-op evidence 来源 | 从主要依赖 `fragment_regions` 改为优先读 manifest |
| witness 生成位置 | 不变，仍在 `AnalyzeSemanticStructure` |
| semantic core lift 位置 | 不变，仍在 `LiftStatefulSemanticIR` |

## 13. 分阶段迁移

### 13.1 Phase 1

目标：

- 先把 explicit-op evidence 从 late matcher 里切出来

完成条件：

1. 新增 early capture
2. 新增 projection
3. 新增 late augment
4. `AnalyzeSemanticStructure` 可统一读取 manifest
5. explicit-op witness 优先来自 manifest
6. `fragment_regions` 继续承接 residual structural evidence

### 13.2 Phase 2

目标：

- 继续把 semantic evidence 从 `fragment_regions` 迁走

重点对象：

- `selection_targets`
- `selection_pairs`
- `arg_reduce_targets`
- `recurrence_edges`

完成后的预期：

- `fragment_regions` 对 `AnalyzeSemanticStructure` 退化成 compatibility fallback

### 13.3 Phase 3

目标：

- 如果 DSL / IR 继续显式化 selection / arg-reduce / recurrence
- 就把 residual structural evidence 继续前移

长期方向是：

- semantic evidence 尽量前移
- semantic truth 仍只在 `Phase A` 冻结

## 14. 不做的事

1. 不改 DSL 用户接口
2. 不改 `SemanticProgram` core
3. 不新增第二套 semantic schema
4. 不让 manifest 成为 `Phase B / C` 输入
5. 不让 manifest 直接承担 `LowerBlackholeOps` lowering contract
6. 不用函数名、buffer 名、helper 名做语义恢复或 projection

## 15. 验证

第一阶段验证应覆盖三类东西：

| 验证项 | 目标 |
|---|---|
| manifest capture 测试 | explicit-op payload 确实在正确边界被保住 |
| `AnalyzeSemanticStructure` 集成测试 | explicit-op witness 确实优先来自 manifest |
| 回归测试 | `fragment_regions` 现有 residual 语义与 `LowerBlackholeOps` 当前行为不回退 |

建议最小测试集合：

1. early capture：
   - `reduce`
   - `copy`
   - `fill`
2. projection：
   - split 后 evidence slice 正确落到对应 device `PrimFunc`
3. late augment：
   - `gemm_py` 能补入 manifest
4. semantic integration：
   - `AnalyzeSemanticStructure` 能从 manifest 产出对应 witness

## 16. 最终判断

这份初设如果要成立，必须坚持下面这张判断表：

| 判断 | 正确说法 |
|---|---|
| manifest 是什么 | `Phase A` evidence-input refactor |
| manifest 解决什么 | explicit-op evidence 不该继续全靠 `fragment_regions` 反推 |
| manifest 不解决什么 | semantic truth ownership 不变；residual structural evidence 第一阶段也不强塞进去 |
| 文档应该站在哪个视角写 | `Phase A` 视角，不是 codegen / lowering 视角 |
| lower 链代码事实为什么还要看 | 只为了确定 evidence capture boundary，不是为了改写 semantic layer 边界 |
