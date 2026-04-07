# Stage 4: Semantic Manifest

## 基本信息

- **文档角色**: `Phase A` 信息源边界文档
- **当前状态**: `2026-04-07` `Phase 1-2` 已落地；当前作为 evidence ownership 参考保留
- **唯一总体设计**: `tasks/dev_design/final_blackhole_backend_redesign.md`
- **直接前置**: `tasks/dev_design/stage4_phase_a_semantic_ir.md`

## 1. 一句话结论

semantic manifest 不是新语义层，也不是 `SemanticProgram` 的替代品。
它只是把 `Phase A` 需要的 explicit-op 与 structural evidence，
从更稳定的 capture 边界带到 `AnalyzeSemanticStructure`。

## 2. 作用域

semantic manifest 只负责一件事：

- 承接 pre-lift evidence

它必须回答：

- 哪些 evidence 需要在 `LowerTileOp` / split / `LowerIntrin` 之前保住
- 这些 evidence 如何在 host/device split 后稳定投影到对应 device `PrimFunc`
- `AnalyzeSemanticStructure` 当前应该优先从哪里读取这些 evidence

它不负责：

- 直接生成 `SemanticProgram`
- 取代 `AnalyzeSemanticStructure`
- 成为 `Phase B / C` 的输入
- 承担 lowering-facing target summary

## 3. ownership boundary

这里的边界固定不变：

- 算法语义真源只属于 `Stateful Semantic IR`
- manifest 只是一层 evidence carrier
- semantic truth 仍然只在
  `AnalyzeSemanticStructure -> LiftStatefulSemanticIR -> Validate*`
  里冻结

## 4. capture boundary

manifest 的存在原因只有一个：

- 有些 evidence 如果等到更后面再收，就会被 destructive lowering 吃掉

当前 capture 路径固定为三段：

1. early capture
   - `CollectSemanticManifestSeeds`
   - 在 `LowerTileOp` 之前保住会被 early lowering 吃掉的 explicit-op payload
2. projection
   - `ProjectSemanticManifest`
   - 在 `SplitHostDevice` 之后把 evidence slice 投影到对应 device `PrimFunc`
3. late augment
   - `AugmentSemanticManifest`
   - 在 device-side pre-`LowerIntrin` 补上仍显式存在的 residual payload

对应主链：

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
```

## 5. 当前 evidence 通道分工

- `tl.semantic_seeds`
  - device program / pipeline skeleton 的 typed seed
- `tl.semantic_manifest_seeds`
  - early lowering 会销毁的 explicit-op evidence
- `tl.semantic_manifest`
  - 投影并补全后的 explicit-op + structural evidence
- `blackhole.fragment_regions`
  - lowering compatibility fallback 与 lowering-facing fragment summary
- `tl.semantic_witnesses`
  - witness layer
- `tl.semantic_program`
  - 冻结后的算法语义

关键结论只有两条：

1. manifest 只是一层 evidence carrier
2. `blackhole.fragment_regions` 已经不是 semantic truth owner

## 6. 当前 schema 边界

### 6.1 当前已落地字段

- `buffers`
- `operations`
- `ordered_regions`
- `anchors`
- `structural_regions`

### 6.2 当前已覆盖的 evidence

- explicit-op payload：
  - `copy / fill / reduce / cumsum / gemm_py`
- structural evidence：
  - `fragment_buffers`
  - `selection_targets`
  - `selection_pairs`
  - `arg_reduce_targets`
  - `update_sources`
  - `loop_carried_state`
  - `recurrence_edges`
  - `row_reductions`

### 6.3 明确不进入 manifest 的东西

- `State.role`
- `UpdateLaw.kind`
- `paired_value_state`
- domain skeleton
- pipeline skeleton
- lowering-facing fragment subset / legality summary
- TT / runtime / CB / segment / ABI 需求

这些对象要么属于 semantic core，要么属于别的 companion channel，
都不应混进 manifest。

## 7. 当前实现边界

### 7.1 manifest-first 已完成的部分

- explicit-op witness 优先来自 manifest
- `selection / arg-reduce / recurrence` 相关 structural evidence 已切到 manifest-first
- `row_reductions` 已迁入 manifest structural evidence
- `AnalyzeSemanticStructure` 当前统一优先读 typed handle，再回退 legacy 字符串字段

### 7.2 `fragment_regions` 当前真实角色

`blackhole.fragment_regions` 现在只剩两类职责：

- `AnalyzeSemanticStructure` 的 compatibility fallback
- `LowerBlackholeOps` 当前仍在消费的 lowering-facing fragment summary

更具体地说：

- semantic 侧已经不是 primary source
- lowering 侧仍会读取 `row_reduction_targets` 等 lowering-facing summary
- 因此当前还不能直接删掉这个 attr

## 8. `AnalyzeSemanticStructure` 当前输入边界

`AnalyzeSemanticStructure` 当前输入面固定为：

- `tl.semantic_manifest`
  - explicit-op evidence + manifest-backed structural evidence
- `blackhole.fragment_regions`
  - lowering compatibility fallback
- `blackhole.work_decomposition`
  - domain skeleton
- `blackhole.pipeline_stages`
  - pipeline trait
- `tl.semantic_seeds`
  - pre-lift typed seed

当前真正变化的是：

- explicit-op evidence 来源：manifest-first
- selection / recurrence structural evidence 来源：manifest-first
- witness 生成位置：不变，仍在 `AnalyzeSemanticStructure`
- semantic core lift 位置：不变，仍在 `LiftStatefulSemanticIR`

## 9. `fragment_regions` 的退出条件

`blackhole.fragment_regions` 的最终命运仍然是退场，但顺序必须固定：

1. semantic 侧继续保持 manifest-first，不再回升为 primary source
2. 给 `LowerBlackholeOps` 建立独立的 lowering-facing summary
3. 让 `LowerBlackholeOps` 不再直接读取 `fragment_regions`
4. 最后再把 `AnalyzeBlackholeFragmentRegions` 降成内部 helper 或调试路径

当前结论：

- semantic 侧的 manifest-first 切换已完成
- `fragment_regions` 当前唯一剩余硬依赖在 lowering 侧

## 10. 不做的事

- 不改 DSL 用户接口
- 不改 `SemanticProgram` core
- 不新增第二套 semantic schema
- 不让 manifest 变成 `Phase B / C` 输入
- 不让 manifest 承担 `LowerBlackholeOps` lowering contract
- 不用函数名、buffer 名或 helper 名做语义恢复或 projection

## 11. 使用方式

这份文档现在只承担三件事：

1. 说明 manifest 的 ownership boundary
2. 说明当前 evidence capture / projection / augment 路径
3. 说明 `fragment_regions` 为什么还在、以及什么时候能删

验证快照与测试数字统一以下沉到：

- `tasks/progress.md`
