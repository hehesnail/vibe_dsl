# Stage 4: Semantic Manifest

## 基本信息

- **文档角色**: `Phase A` 信息源重构初设
- **当前状态**: 设计中
- **唯一总体设计**: `tasks/dev_design/final_blackhole_backend_redesign.md`
- **直接前置**: `tasks/dev_design/stage4_phase_a_semantic_ir.md`

## 1. 结论

| 项 | 结论 |
|---|---|
| semantic recovery 属于哪一层 | 仍然只属于 `Phase A / Stateful Semantic IR` |
| manifest 是什么 | `Phase A` 的 pre-lift evidence input |
| manifest 能不能替代 `AnalyzeSemanticStructure` | 不能 |
| manifest 能不能直接生成 `SemanticProgram` | 不能 |
| 第一阶段要做什么 | 把 explicit-op evidence 从 `fragment_regions` 的 late matcher 路径中切出来 |
| 第一阶段不做什么 | 不强行把 `selection_pairs / arg_reduce_targets / recurrence_edges` 并进 manifest |

一句话版本：

> manifest 不是新语义层，只是把 `Phase A` 需要的一部分 evidence 从更稳定的 capture 边界带到 `AnalyzeSemanticStructure`。

## 2. 问题

当前问题不是 `Phase A` 要不要做 semantic recovery，而是 `Phase A` 上游 evidence source 不稳。

现状有两个问题：

1. 一部分 evidence 在 lower 链里本来是显式的，但会在不同边界被销毁。
2. 当前主要还是靠 `blackhole.fragment_regions` 从 lowered 结构里回收这些事实，容易继续 case-by-case。

相关代码事实只需要记三条：

| 边界 | 当前事实 | 含义 |
|---|---|---|
| `LowerTileOp` | `copy / fill / reduce / cumsum` 会在这里被 lowered 掉 | 这部分 evidence 不能等到后面再收 |
| `SplitHostDevice` | 会创建新的 device `PrimFunc`，不会自动保留任意 pre-split attr | 早采集之后必须显式 projection |
| device-side pre-`LowerIntrin` | 仍能看到少量 residual explicit op，例如 `gemm_py` | 仍需要 late augment |

所以问题不是 “把 semantic recovery 挪走”，而是：

- `Phase A` 消费什么 evidence
- 这些 evidence 应该在哪些边界前先保住

## 3. 边界

### 3.1 ownership boundary

这条不变：

- 算法语义真源只属于 `Stateful Semantic IR`
- 冻结产物仍然只有 `SemanticProgram`

### 3.2 capture boundary

这才是本文要改的：

- 某些 evidence 必须在被 `LowerTileOp` / split / `LowerIntrin` 扰动前先保住

### 3.3 通道分工

| 通道 | 承接什么 | 不承接什么 | 主要消费者 |
|---|---|---|---|
| `tl.semantic_seeds` | device program / pipeline skeleton | explicit-op payload | `AnalyzeSemanticStructure` |
| `tl.semantic_manifest_seeds` | early lowering 会销毁的 explicit-op evidence | semantic core | `ProjectSemanticManifest` |
| `tl.semantic_manifest` | 投影并补全后的 explicit-op evidence | witness / `SemanticProgram` | `AnalyzeSemanticStructure` |
| `blackhole.fragment_regions` | residual structural evidence + 当前兼容职责 | 长期 semantic truth owner | `AnalyzeSemanticStructure`, `LowerBlackholeOps` |
| `tl.semantic_witnesses` | 统一 witness axis | raw lowering payload | lift / validator |
| `tl.semantic_program` | 冻结后的算法语义 | raw analysis attr | `Phase B` |

这张表的关键点只有两个：

1. manifest 只是一层 evidence carrier。
2. semantic truth 仍然只在 `AnalyzeSemanticStructure -> LiftStatefulSemanticIR -> Validate*` 里冻结。

## 4. 方案

### 4.1 总体形态

不是单点 collector，而是三段式：

| 阶段 | 作用 | 产物 |
|---|---|---|
| early capture | 在 `LowerTileOp` 前保住会被 early lowering 吃掉的 explicit-op payload | `tl.semantic_manifest_seeds` |
| projection | 在 `SplitHostDevice` 后把 early seeds 投影到对应 device `PrimFunc` | `tl.semantic_manifest` |
| late augment | 在 device-side pre-`LowerIntrin` 补上仍显式存在的 residual op payload | 增补后的 `tl.semantic_manifest` |

对应流程：

```text
LowerAndLegalize
  -> CollectSemanticManifestSeeds
  -> LowerTileOp
OptimizeForTarget
  -> SplitHostDevice
  -> ProjectSemanticManifest
blackhole_codegen(device_mod)
  -> AugmentSemanticManifest
  -> LowerIntrin
  -> AnalyzeSemanticStructure
  -> LiftStatefulSemanticIR
```

### 4.2 每段做什么

| pass | 做什么 | 不做什么 |
|---|---|---|
| `CollectSemanticManifestSeeds` | 记录 explicit-op payload、必要 buffer metadata、最小 region anchor | 不做 witness，不做 semantic core 归约 |
| `ProjectSemanticManifest` | 按稳定 split anchor 把 evidence slice 挂到对应 device `PrimFunc` | 不做新的语义推断 |
| `AugmentSemanticManifest` | 补入 device-side 仍显式存在的 residual payload，例如 `gemm_py` | 不直接生成 `SemanticProgram` |

### 4.3 manifest schema

第一阶段保持最小：

| 字段 | 含义 |
|---|---|
| `buffers` | manifest 内引用的 buffer descriptor |
| `operations` | explicit-op payload 列表 |
| `ordered_regions` | 最小 ordered / serial region anchor |
| `anchors` | split / region 级稳定锚点 |

第一阶段明确不放：

- `selection_pairs`
- `arg_reduce_targets`
- `recurrence_edges`
- `State.role`
- `UpdateLaw.kind`
- `paired_value_state`

原因很简单：这些已经不是 raw evidence，而是在往 witness / semantic core 滑。

## 5. 第一阶段范围

按 evidence 类型看：

| evidence 类型 | 例子 | 第一阶段是否进 manifest |
|---|---|---|
| explicit-op payload | `copy / fill / reduce / cumsum / gemm_py` | 是 |
| domain skeleton | work axes / derived indices | 否 |
| pipeline skeleton | pipeline trait | 否 |
| residual structural evidence | `selection_pairs / arg_reduce_targets / recurrence_edges` | 否 |
| target-facing lowering requirement | CB / segment / runtime lowering 需求 | 否 |

也就是说，第一阶段的准确表述是：

- 把 explicit-op evidence 从 late matcher 路径切出来
- residual structural evidence 先不硬塞进 manifest

## 6. `AnalyzeSemanticStructure` 怎么接

`AnalyzeSemanticStructure` 的职责不变，只是输入面收正。

第一阶段输入：

| 输入 | 用途 |
|---|---|
| `tl.semantic_manifest` | explicit-op evidence |
| `blackhole.fragment_regions` | residual structural evidence |
| `blackhole.work_decomposition` | domain skeleton |
| `blackhole.pipeline_stages` | pipeline trait |
| `tl.semantic_seeds` | pre-lift typed seed |

第一阶段输出不变：

- `tl.semantic_structure`
- `tl.semantic_witnesses`
- `tl.semantic_program`

所以真正变化的是：

| 项 | 变化 |
|---|---|
| explicit-op evidence 来源 | 从主要依赖 `fragment_regions` 改为优先读 manifest |
| witness 生成位置 | 不变，仍在 `AnalyzeSemanticStructure` |
| semantic core lift 位置 | 不变，仍在 `LiftStatefulSemanticIR` |

## 7. 迁移

### 7.1 Phase 1

目标：

- 先把 explicit-op evidence 从 late matcher 里切出来

完成条件：

1. 新增 early capture
2. 新增 projection
3. 新增 late augment
4. `AnalyzeSemanticStructure` 可统一读取 manifest
5. explicit-op witness 优先来自 manifest
6. `fragment_regions` 继续承接 residual structural evidence

### 7.2 Phase 2

目标：

- 继续把 semantic evidence 从 `fragment_regions` 迁走

重点对象：

- `selection_targets`
- `selection_pairs`
- `arg_reduce_targets`
- `recurrence_edges`

预期结果：

- `fragment_regions` 对 `AnalyzeSemanticStructure` 退化成 compatibility fallback

## 8. 不做的事

1. 不改 DSL 用户接口
2. 不改 `SemanticProgram` core
3. 不新增第二套 semantic schema
4. 不让 manifest 成为 `Phase B / C` 输入
5. 不让 manifest 直接承担 `LowerBlackholeOps` lowering contract
6. 不用函数名、buffer 名、helper 名做语义恢复或 projection

## 9. 验证

第一阶段验证只看三类东西：

| 验证项 | 目标 |
|---|---|
| manifest capture 测试 | explicit-op payload 确实在正确边界被保住 |
| `AnalyzeSemanticStructure` 集成测试 | explicit-op witness 确实优先来自 manifest |
| 回归测试 | `fragment_regions` 现有 residual 语义与 `LowerBlackholeOps` 当前行为不回退 |

建议最小测试集合：

1. early capture：`reduce / copy / fill`
2. projection：split 后 evidence slice 正确落到对应 device `PrimFunc`
3. late augment：`gemm_py` 能补入 manifest
4. semantic integration：`AnalyzeSemanticStructure` 能从 manifest 产出对应 witness
