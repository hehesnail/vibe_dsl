# Stage 4: Semantic Manifest

## 基本信息

- **文档角色**: `Phase A` 信息源重构设计与收口文档
- **当前状态**: `Phase 1-2` 已实施（`2026-04-06`），当前作为 `Phase A` 信息源边界文档保留
- **唯一总体设计**: `tasks/dev_design/final_blackhole_backend_redesign.md`
- **直接前置**: `tasks/dev_design/stage4_phase_a_semantic_ir.md`

## 1. 结论

| 项 | 结论 |
|---|---|
| semantic recovery 属于哪一层 | 仍然只属于 `Phase A / Stateful Semantic IR` |
| manifest 是什么 | `Phase A` 的 pre-lift evidence input |
| manifest 能不能替代 `AnalyzeSemanticStructure` | 不能 |
| manifest 能不能直接生成 `SemanticProgram` | 不能 |
| 当前已完成什么 | explicit-op evidence 与 selection / arg-reduce / recurrence structural evidence 已前移到 manifest |
| 当前剩余债务 | `row_reductions` 仍留在 `fragment_regions`，因为 semantic/lowering mixed ownership 尚未拆开 |

一句话版本：

> manifest 不是新语义层，只是把 `Phase A` 需要的 explicit-op 和 structural evidence 从更稳定的 capture 边界带到 `AnalyzeSemanticStructure`。

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
| `tl.semantic_manifest` | 投影并补全后的 explicit-op + structural evidence | witness / `SemanticProgram` | `AnalyzeSemanticStructure` |
| `blackhole.fragment_regions` | residual reduction evidence + lowering compatibility summary | semantic truth owner | `AnalyzeSemanticStructure` fallback, `LowerBlackholeOps` |
| `tl.semantic_witnesses` | 统一 witness axis | raw lowering payload | lift / validator |
| `tl.semantic_program` | 冻结后的算法语义 | raw analysis attr | `Phase B` |

这张表的关键点只有两个：

1. manifest 只是一层 evidence carrier。
2. semantic truth 仍然只在 `AnalyzeSemanticStructure -> LiftStatefulSemanticIR -> Validate*` 里冻结。

### 3.4 `fragment_regions` 当前真实角色

在 `Phase 2` 之后，`blackhole.fragment_regions` 的角色已经收缩成两部分：

- `AnalyzeSemanticStructure` 的 compatibility fallback
- `LowerBlackholeOps` 当前仍在消费的 fragment/lowering summary

它已经**不是** semantic truth owner，也不该继续被当成 `Phase A` 的主 evidence source。

当前仍保留它，不是因为 semantic manifest 不够，而是因为 `row_reductions` 这类事实现在仍是
**mixed ownership**：

- semantic 侧需要它来恢复 `reduce_*` update
- lowering 侧也需要它来做 fragment subset summary / legality / fail-fast

因此当前不能把整块 `row_reductions` 粗暴平移到 manifest。那会重新制造两份混合真源：

- semantic 侧一份
- lowering 侧一份

这正是本文要避免的。

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

当前已落地 schema：

| 字段 | 含义 |
|---|---|
| `buffers` | manifest 内引用的 buffer descriptor |
| `operations` | explicit-op payload 列表 |
| `ordered_regions` | 最小 ordered / serial region anchor |
| `anchors` | split / region 级稳定锚点 |
| `structural_regions` | selection / arg-reduce / recurrence 相关 structural evidence |

仍然明确不放：

- `State.role`
- `UpdateLaw.kind`
- `paired_value_state`
- lowering-facing fragment subset / legality summary

原因很简单：这些已经不是 raw evidence，而是在往 witness / semantic core 滑。

`row_reductions` 已在 `2026-04-06` 迁入 manifest structural evidence（`manifest_key::kRowReductions`），
`AnalyzeSemanticStructure` 对 reduction evidence 改成 manifest-first 消费。
`LowerBlackholeOps` 仍通过 `blackhole.fragment_regions` 消费 `row_reduction_targets`（lowering-facing，不归 manifest 管）。

### 4.4 当前实现落点（`2026-04-06`）

- `LowerAndLegalize` 已在 `LayoutInference` 之后、`LowerTileOp` 之前插入
  `CollectSemanticManifestSeeds`
- `OptimizeForTarget` 已在 `SplitHostDevice` 之后插入 `ProjectSemanticManifest`
- Blackhole device-side pass 主线已在 `LowerDeviceStorageAccessInfo` 之后、`LowerIntrin` 之前插入
  `AugmentSemanticManifest`
- `CollectSemanticManifestSeeds` 当前只抓会被 `LowerTileOp` 吃掉的 explicit-op：
  `copy / fill / reduce / cumsum`
- `AugmentSemanticManifest` 当前同时承担：
  - residual explicit-op augment：`gemm_py`
  - structural evidence 前移：
    `fragment_buffers / selection_targets / selection_pairs / arg_reduce_targets /
    update_sources / loop_carried_state / recurrence_edges`
- manifest schema 当前已经以 typed attr 形式落地：
  - `buffers` 保留 `buffer / name / scope / dtype / shape`
  - `operations` 保留 op kind、capture stage、ordered-region anchor 与 typed payload
  - `ordered_regions` 保留最小 lexical ordered region
  - `anchors` 保留 operation / ordered-region anchor
  - `structural_regions` 保留：
    - `fragment_buffers`
    - `selection_targets`
    - `selection_pairs`
    - `arg_reduce_targets`
    - `update_sources`
    - `loop_carried_state`
    - `recurrence_edges`
- `AnalyzeSemanticStructure` 当前对 manifest 的消费边界也已经落地：
  - 追加 `explicit_op_manifest` seed
  - 发出 `boundary / ordered_region` witness，`evidence_source = semantic_manifest`
  - 增加 `semantic_boundary` supplement
  - `selection / arg-reduce / recurrence` witness 现在 manifest-first
  - manifest-only 输入已经足够支撑 `select / recurrence` witness 与 semantic lift
  - 不改变 `SemanticProgram` core 的 `Domain / State / Update` 词汇边界

## 5. 当前 evidence 覆盖面

按当前实现状态看：

| evidence 类型 | 例子 | 当前是否进 manifest |
|---|---|---|
| explicit-op payload | `copy / fill / reduce / cumsum / gemm_py` | 是 |
| structural evidence | `selection_targets / selection_pairs / arg_reduce_targets / recurrence_edges / update_sources / loop_carried_state` | 是 |
| domain skeleton | work axes / derived indices | 否，仍由 `blackhole.work_decomposition` 承接 |
| pipeline skeleton | pipeline trait | 否，仍由 `blackhole.pipeline_stages` 承接 |
| reduction evidence | `row_reductions` | 是（`2026-04-06` 已迁入 manifest structural evidence） |
| target-facing lowering requirement | CB / segment / runtime lowering 需求 | 否 |

也就是说，当前 manifest 的准确边界是：

- 把会被 destructive lowering 吃掉的 explicit-op evidence 前移保存
- 把 selection / arg-reduce / recurrence 相关 structural evidence 前移为 typed carrier
- 不把 domain / pipeline skeleton、lowering summary 混进 manifest
- `row_reductions` 已迁入 manifest；`LowerBlackholeOps` 仍独立从 `fragment_regions` 读 lowering-facing `row_reduction_targets`

## 6. `AnalyzeSemanticStructure` 当前怎么接

`AnalyzeSemanticStructure` 的职责不变，只是输入面收正。

当前输入：

| 输入 | 用途 |
|---|---|
| `tl.semantic_manifest` | explicit-op evidence + manifest-backed structural evidence（含 `row_reductions`） |
| `blackhole.fragment_regions` | lowering compatibility fallback（semantic 侧仅在 manifest 缺失时回退） |
| `blackhole.work_decomposition` | domain skeleton |
| `blackhole.pipeline_stages` | pipeline trait |
| `tl.semantic_seeds` | pre-lift typed seed |

当前输出不变：

- `tl.semantic_structure`
- `tl.semantic_witnesses`
- `tl.semantic_program`

所以当前真正变化的是：

| 项 | 变化 |
|---|---|
| explicit-op evidence 来源 | 从主要依赖 `fragment_regions` 改为优先读 manifest |
| selection / recurrence structural evidence 来源 | 从主要依赖 `fragment_regions` 改为 manifest-first |
| witness 生成位置 | 不变，仍在 `AnalyzeSemanticStructure` |
| semantic core lift 位置 | 不变，仍在 `LiftStatefulSemanticIR` |

## 7. 迁移

### 7.1 Phase 1（已实施，`2026-04-06`）

目标：

- 先把 explicit-op evidence 从 late matcher 里切出来

完成条件：

1. 新增 early capture
2. 新增 projection
3. 新增 late augment
4. `AnalyzeSemanticStructure` 可统一读取 manifest
5. explicit-op witness 优先来自 manifest
6. `fragment_regions` 继续承接 residual structural evidence

当前状态：

- 1-6 已完成

### 7.2 Phase 2（已实施，`2026-04-06`）

目标：

- 继续把 semantic evidence 从 `fragment_regions` 迁走

重点对象：

- `selection_targets`
- `selection_pairs`
- `arg_reduce_targets`
- `recurrence_edges`

完成条件：

1. `AugmentSemanticManifest` 能前移 `selection_targets / selection_pairs / arg_reduce_targets /
   recurrence_edges`
2. 为了让 manifest-only consumer 成立，同时带上 `fragment_buffers / update_sources /
   loop_carried_state`
3. `AnalyzeSemanticStructure` 对上述结构 evidence 改成 manifest-first
4. `fragment_regions` 对 `AnalyzeSemanticStructure` 退化成 compatibility fallback

当前状态：

- 1-4 已完成
- `row_reductions` 已在 `2026-04-06` 完成 manifest 迁入：
  - `EncodeStructuralManifestRegion` 现在 copy `manifest_key::kRowReductions`
  - `AnalyzeSemanticStructure` 对 reduction evidence 改成 manifest-first 消费
    （通过 `EvidenceAccumulator::IngestStructuralRegion` 统一处理 manifest 和 fragment_regions 来源）
  - `LowerBlackholeOps` 仍独立从 `blackhole.fragment_regions` 读 `row_reduction_targets`（lowering-facing，不归 manifest 管）
  - semantic 侧已完成所有 evidence 的 manifest-first 切换
- manifest schema 所有 key 已集中到 `manifest_key::` 命名空间（`semantic_program.h`）

### 7.3 `fragment_regions` 的退出条件

`blackhole.fragment_regions` 的最终命运仍然是退场，但顺序必须明确：

1. ~~先把 semantic 侧剩余需要的 `row_reductions` 相关 evidence 拆成 manifest-first / semantic-owned 输入~~ **已完成（`2026-04-06`）**
2. 再把 lowering 侧需要的 reduction / fragment subset summary 迁到独立 lowering-facing summary
3. 等 `LowerBlackholeOps` 不再直接读 `fragment_regions`
4. 最后让 `AnalyzeBlackholeFragmentRegions` 降成内部 helper 或调试路径，删除该 companion attr

也就是说：

- semantic 侧不再依赖 `fragment_regions` 作为 primary evidence source（manifest-first 已全面生效）
- `fragment_regions` 当前唯一剩余消费者是 `LowerBlackholeOps`（lowering-facing）
- 后面要删，但要先给 lowering 建独立 summary，再踢掉 attr 本身

## 8. 不做的事

1. 不改 DSL 用户接口
2. 不改 `SemanticProgram` core
3. 不新增第二套 semantic schema
4. 不让 manifest 成为 `Phase B / C` 输入
5. 不让 manifest 直接承担 `LowerBlackholeOps` lowering contract
6. 不用函数名、buffer 名、helper 名做语义恢复或 projection

## 9. 验证

当前验证主要看三类东西：

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

当前最新稳定验证快照（`2026-04-06`，与 `tasks/progress.md` 对齐）：

- `pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -q`
  - `38 passed`
- `source scripts/setup_tt_sim.sh && pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py -q`
  - `12 passed`
- `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py -q`
  - `40 passed, 10 skipped, 1 xfailed`
- `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -q`
  - `24 passed, 11 skipped`
- `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py -q`
  - `1 passed`
- `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q`
  - `26 passed`
