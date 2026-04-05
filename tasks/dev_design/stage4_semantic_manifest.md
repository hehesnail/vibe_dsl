# Stage 4: Semantic Manifest — 正向语义传递

## 基本信息

- **文档角色**: `Phase A` 信息源重构初设
- **当前状态**: 设计中；已按完整 lower 链代码事实重写
- **唯一总体设计**: `tasks/dev_design/final_blackhole_backend_redesign.md`
- **前置**: `tasks/dev_design/stage4_phase_a_semantic_ir.md`

## 0. 先摆正前提

这份文档讨论的不是“把 semantic recovery 挪出 semantic layer”，而是：

- 在 `Semantic Recovery -> Stateful Semantic IR` 这条既定交接里
- 怎么把 **Phase A 需要的 pre-lift evidence** 从更稳定的边界输入给
  `AnalyzeSemanticStructure -> LiftStatefulSemanticIR -> Validate*`

也就是说：

- 算法语义真源仍然只存在于 `Stateful Semantic IR`
- `SemanticProgram` 仍然是唯一冻结产物
- manifest 如果存在，也只能是 `Phase A` 的 evidence carrier
- 它不能长成新的语义层、也不能绕开 `Phase A` 的 witness/core/validator

所以这里必须严格区分两件事：

1. **ownership boundary**
   - 语义 truth 只属于 `Stateful Semantic IR`
2. **capture boundary**
   - 某些 evidence 必须在被 `LowerTileOp` / `LowerIntrin` / split 扰动前先保住

我前一版写错的地方，就在于把这两条边界混成了一条。

## 1. 问题定义

当前问题不是 “`AnalyzeBlackholeFragmentRegions` 写得还不够聪明”，而是：

- `Phase A` 需要的一部分 algorithmic facts，本来在 compiler IR 里是显式的
- 这些 facts 并不是都在同一个边界被销毁
- 但当前主链仍把它们统一交给 split-after / simplify-after 的结构恢复

真正的问题链路是：

```text
TileLang DSL / Python
  -> frontend helper / macro expansion
  -> tl.tileop.* calls
  -> LowerTileOp                         <- copy / fill / reduce / cumsum 在这里就被吃掉
  -> OptimizeForTarget / SplitHostDevice
  -> blackhole_codegen
       -> LowerIntrin                    <- gemm_py 这类残余显式 op 在这里继续被吃掉
       -> AnalyzeBlackholeFragmentRegions
       -> AnalyzeSemanticStructure
       -> LiftStatefulSemanticIR
```

所以根因不是单一 pass 写坏了，而是：

1. compiler 先在多个不同边界销毁显式语义
2. 后段再试图从已经 lowered 的结构里统一反推
3. 每扩一个 workload family，就容易继续往 `fragment_regions` 里堆 case-by-case matcher

这也是为什么别人指出：

> `AnalyzeBlackholeFragmentRegions` 这条路线本身不可扩，后面会一直 case-by-case。

这个判断是对的。要解决它，不能只在 `blackhole_codegen` 里加一个“更聪明的恢复器”，而是要把：

- `Phase A` 到底消费什么 evidence
- 这些 evidence 在整条 lower 链上会在哪些边界被销毁

这两件事重新摆正。

但这里要再次强调：

> 这仍然是在修 `Phase A` 的 evidence input，不是在重定义 semantic layer。

## 2. 当前代码事实

这份初设必须建立在完整链路上，而不是只看 `Analyze*` 几个 pass。

### 2.1 frontend DSL 到 compiler IR

当前前端路径里，相关事实是：

- `@T.prim_func` / `@tilelang.jit`
  - 先通过 parser / eager builder 生成 `PrimFunc`
- DSL surface 上的 `T.copy / T.fill / T.reduce / T.cumsum / T.gemm`
  - 会在 Python 侧构造成 `tl.tileop.*` call，或者 macro 展开后再落成这些 call
- `T.reduce` / `T.cumsum` 这类 helper 并不是 “一个用户调用 = 一个最终 op”
  - 它们可能先插 staging copy / alloc / writeback
  - 然后才把核心 effect 落成 `tl.tileop.reduce` / `tl.tileop.cumsum`

因此 manifest 要承接的是：

- compiler IR 里真实存在的 op/effect evidence
- 不是 Python helper 的表面名字
- 更不是 workload noun

### 2.2 `LowerAndLegalize`：第一个真正的语义断点是 `LowerTileOp`

`tilelang/engine/phase.py` 里，`LowerAndLegalize` 会很早执行：

```text
BindTarget
  -> Simplify
  -> LayoutReducer / LayoutInference
  -> LowerTileOp
  -> LowerL2Persistent
  -> ...
```

这里最关键的代码事实是：

- `LowerTileOp` 会解析 `Evaluate(Call tl.tileop.*)` 并直接调用各个 `TileOperator::Lower(...)`
- 对 Blackhole，只有 `GemmPyNode` 被特意保留：
  - `if (TargetIsBlackhole(target_) && tile_op->IsInstance<GemmPyNode>()) return op;`
- 也就是说：
  - `copy / fill / reduce / cumsum`
    - 在 `LowerTileOp` 阶段就被 lowered 成普通 TIR 结构
  - `gemm_py`
    - 在 Blackhole 路径上会继续保留到后面的 device pipeline

因此：

> 如果目标是接住“intrinsic-backed algorithmic truth”，就不能把 collector 统一放到
> `blackhole_codegen` / pre-device-`LowerIntrin`，因为那时 `copy / fill / reduce / cumsum`
> 已经不在了。

### 2.3 `OptimizeForTarget`：已有 pre-split capture 先例

在 `OptimizeForTarget(..., target="blackhole")` 里，当前已经有两类 “先保事实，再降形状” 的做法：

1. `AnnotateBlackholeCopySemantics`
   - 在 `SplitHostDevice` 前给 copy loop 打 `blackhole.copy_semantics`
   - 它已经承认 copy 语义如果等后段再猜会不稳
2. `ProjectSemanticSeeds` / `CollectDevicePrograms`
   - 在 split 前记录 `tl.semantic_seeds` 和 `tl.device_programs`
   - 它已经承认某些 pre-lift truth 必须在 split 前固定

也就是说，代码库本身已经接受下面这个原则：

- truth 的 capture 点必须跟着“它会在哪被销毁”来定
- 不同 truth 可以走不同通道
- 不能指望一个晚期 attr 把所有 upstream 明确信息都兜回来

### 2.4 `SplitHostDevice`：第二个关键边界是 projection，不是分析

如果把 manifest 前移到 `LowerTileOp` 边界，还会遇到第二个真实工程问题：

- `SplitHostDevice` 会创建新的 device `PrimFunc`
- 它不会自动继承任意 pre-split `PrimFunc` attr
- 当前实现里，新 device `PrimFunc` 只显式写入：
  - `target`
  - `tir.noalias`
  - `tir.is_global_func`
  - `tl.non_restrict_params`
  - `cluster_dims`（如果有）

因此：

> 早采集不是唯一问题；早采集之后怎么跨 `SplitHostDevice` 投影到 device `PrimFunc`，
> 也是必须显式设计的合同。

这件事不能靠：

- 侥幸认为 attrs 会自动保留
- 函数名字匹配
- 事后再扫描 host root 去猜哪个 kernel 对应哪个 seed

projection 必须基于稳定结构：

- device-region 顺序
- `tl.device_programs` 里已经规划好的 member kernel 顺序
- 或显式 region ordinal

### 2.5 `blackhole_codegen`：late boundary 仍然存在，但只承接 residual explicit ops

当前 Blackhole 设备侧主线是：

```text
LowerDeviceStorageAccessInfo
  -> LowerIntrin
  -> Simplify
  -> HoistBroadcastValues
  -> SplitBlackholeKernel
  -> AnalyzeBlackholeWorkDecomposition
  -> AnalyzeBlackholeFragmentRegions
  -> AnalyzeBlackholePipelineStages
  -> AnalyzeSemanticStructure
  -> LiftStatefulSemanticIR
  -> ValidateStatefulSemanticIR
  -> ValidateSemanticRefinement
  -> LowerBlackholeOps
```

这条链路里仍然有两个重要事实：

1. `SplitBlackholeKernel`
   - 当前不会制造新的 `PrimFunc` 身份
   - 它只是补 segment annotation / plan
2. `blackhole_codegen` 在 `LowerIntrin` 前仍能看到少量显式 op
   - 当前主要是 `tl.tileop.gemm_py`

所以 late capture 仍然有意义，但它只能承接：

- `LowerTileOp` 之后仍然活着的那部分显式 evidence

而不能假装自己是全量入口。

### 2.6 当前 `AnalyzeSemanticStructure` 的真实依赖

当前 `AnalyzeSemanticStructure` 仍主要依赖：

- `blackhole.fragment_regions`
  - `row_reductions`
  - `selection_targets`
  - `selection_pairs`
  - `update_sources`
  - `arg_reduce_targets`
  - `recurrence_edges`
  - `loop_carried_state`
- `blackhole.work_decomposition`
- `blackhole.pipeline_stages`
- `tl.semantic_seeds`（如果 device `PrimFunc` 上真有）

`LowerBlackholeOps` 也仍直接消费：

- `blackhole.fragment_regions`
- `blackhole.work_decomposition`
- `blackhole.pipeline_stages`
- `blackhole.copy_semantics`

因此这份 manifest 初设要同时满足两点：

1. 长期目标不收窄
   - `fragment_regions` 不应继续做 semantic truth owner
2. 迁移现实不能假装消失
   - `LowerBlackholeOps` 眼下还直接吃这些 attrs

## 3. 设计目标

这份初设追求下面五个目标：

1. 按真实销毁边界前移 capture，而不是把所有 truth 都硬塞到 pre-`LowerIntrin`
2. 把 `copy / fill / reduce / cumsum` 这类在 `LowerTileOp` 就显式存在的事实，从 late matcher 中切出来
3. 把 `gemm_py` 这类在 Blackhole device path 里继续显式存在的事实，也纳入同一 semantic evidence 通道
4. 明确 `SplitHostDevice` 之后的 projection 合同，避免 pre-split attr 漂浮
5. 在迁移期内不同时搅乱 `Phase A` core、`Phase B/C` 边界和 `LowerBlackholeOps` 当前行为

对应地，这份设计**不**追求：

1. 第一步就删除 `AnalyzeBlackholeFragmentRegions`
2. 第一步就让 `LowerBlackholeOps` 只吃 `SemanticProgram`
3. 第一步就把 `selection / companion / arg-reduce / carried-from` 全部前移完成
4. 在 manifest 里引入第二套 semantic core vocabulary

## 4. 方案总述

### 4.1 结论：manifest 是 `Phase A` 的 evidence-input 重构，不是新层

这里的正确说法不是：

- semantic recovery 要搬到 `LowerTileOp` / `SplitHostDevice` / `blackhole_codegen`

而是：

- `Phase A` 作为 semantic recovery 的主体不变
- 只是它要消费的 canonical evidence，不能再单押在当前 `fragment_regions`
- 因此需要在 evidence 会被销毁的边界提前保留，再统一喂给 `AnalyzeSemanticStructure`

也就是说，下面这些 pass 如果存在，它们都只是：

- `Phase A` 上游的 evidence collector / projector / normalizer

而不是：

- 新的语义真源
- 绕过 `AnalyzeSemanticStructure` 的快捷路径
- 脱离 `Phase A` 的单独 lowering 主线

### 4.2 collector 不是一个点，而是两段 capture + 一次 projection

完整 lower 链上的可行方案不是：

- “在 `blackhole_codegen` / pre-device-`LowerIntrin` 新增一个 `CollectSemanticManifest`”

而是：

1. **Early capture**
   - 在 `LowerTileOp` 边界抓住会被它直接销毁的 tileop payload
   - 当前重点是：
     - `copy`
     - `fill`
     - `reduce`
     - `cumsum`
     - 被 `LowerTileOp` 吃掉的 `gemm`
2. **Projection**
   - 让 early capture 的结果跨 `SplitHostDevice` 落到每个 device `PrimFunc`
3. **Late augment**
   - 在 `blackhole_codegen` / pre-device-`LowerIntrin` 把还显式存在的 residual op 补进来
   - 当前重点是：
     - `gemm_py`

最终 `AnalyzeSemanticStructure` 看到的仍然是一个统一的：

- `PrimFunc.attrs["tl.semantic_manifest"]`

但它的来源不是单点 pass，而是：

```text
pre-LowerTileOp capture
  -> split-aware projection
  -> pre-device-LowerIntrin residual augment
  -> tl.semantic_manifest
```

### 4.2 它在整条链路里的真实位置

更准确的总流程应写成：

```text
LowerAndLegalize
  -> CollectSemanticManifestSeeds        <- 新增：抓 LowerTileOp 会销毁的 tileop payload
  -> LowerTileOp
  -> ...
OptimizeForTarget
  -> AnnotateBlackholeCopySemantics
  -> ProjectSemanticSeeds
  -> CollectDevicePrograms
  -> SplitHostDevice
  -> ProjectSemanticManifest            <- 新增：把 early seeds 投影到 device PrimFunc
Filter(host/device)
blackhole_codegen(device_mod)
  -> LowerDeviceStorageAccessInfo
  -> AugmentSemanticManifest            <- 新增：补进 gemm_py 这类 residual explicit ops
  -> LowerIntrin
  -> Simplify
  -> HoistBroadcastValues
  -> SplitBlackholeKernel
  -> Analyze*
  -> Lift / Validate / LowerBlackholeOps
```

这里要明确：

- `CollectSemanticManifestSeeds`
  - 必须发生在 `LowerTileOp` 之前，或者作为 `LowerTileOp` 的 sidecar
  - 它只是在 `Phase A` 之前保住 evidence，不是提前冻结 semantic truth
- `ProjectSemanticManifest`
  - 必须在 `SplitHostDevice` 之后显式做
  - 不能假设 attrs 自动继承
- `AugmentSemanticManifest`
  - 只承接晚边界还存在的 residual explicit ops
  - 它补的是 evidence，不是直接产出 `SemanticProgram`

### 4.3 为什么不是只放在 `LowerTileOp` 前

只放在 `LowerTileOp` 前也不够，因为：

- Blackhole 还特意保留了 `gemm_py`
- 后段 device path 上还能看到这类显式 payload
- 如果不接这部分，manifest 仍然不是完整的 explicit-op evidence 通道

### 4.4 为什么不是只放在 `blackhole_codegen`

只放在 `blackhole_codegen` 也不行，因为：

- `copy / fill / reduce / cumsum` 在那之前已经被 `LowerTileOp` 吃掉
- 到 device-side pre-`LowerIntrin` 时，已经没有 tileop payload 可抓
- 只能又退回 loop/pattern recovery

## 5. Manifest 不是什么

- 不是新的 compiler IR 层
- 不是新的长期 schema
- 不是 `SemanticProgram` 的替代品
- 不是 `LowerBlackholeOps` 的 lowering contract
- 不是 `Phase B` 的 shortcut 输入
- 不是 `Phase C` 的 ABI 输入
- 不是用户可见 DSL 接口

它只是：

> 一个围绕显式 algorithmic evidence 设计的 `Phase A` 输入通道。

如果后面发现 `Phase B` 或 `Phase C` 想直接读取 manifest，结论不应该是“manifest 很有用所以继续复用”，而应该是：

- 要么该 truth 本来属于 `Phase A`，那就把它收进 witness / `SemanticProgram`
- 要么该 truth 根本不属于 `Phase A`，那就直接进入 `Spatial Program IR` 或 `TT Target IR`

## 6. 生命周期与投影合同

### 6.1 生命周期

Manifest 的完整生命周期是：

1. 在 `LowerTileOp` 边界之前抓早期 tileop payload
2. 以 split-aware seed 形式保存在 pre-split 载体上
3. 在 `SplitHostDevice` 之后显式投影到 device `PrimFunc`
4. 在 device-side `LowerIntrin` 前补充 residual explicit ops
5. 作为 pre-lift analysis input 继续向后传递
6. 不进入 post-lift hard-freeze / typed_rebind 合同

也就是说：

- `tl.semantic_manifest` 不是 post-lift companion IR
- 它和 `tl.semantic_seeds` 一样属于 pre-lift 输入
- 它的唯一职责是给 `AnalyzeSemanticStructure` 提供更稳定的 evidence
- 但它比 `tl.semantic_seeds` 多一个必要动作：
  - **split-aware projection**

### 6.2 投影必须显式化

因为 `SplitHostDevice` 不会自动保留任意 attrs，所以必须引入显式投影规则。

这条合同至少要满足：

1. early seeds 必须带 region ordinal 或等价稳定锚点
2. projection 必须依据：
   - split 的 device-region 顺序
   - 或 `tl.device_programs` 的 planned member 顺序
3. 不允许靠：
   - kernel 名字字符串猜测
   - buffer 名字匹配
   - host root 再扫描回填

### 6.3 `SplitBlackholeKernel` 不是这里的 blocker

当前 `SplitBlackholeKernel` 不会新建 `PrimFunc`，所以：

- manifest 在投影到 device `PrimFunc` 之后
- 不需要再处理一次 cross-function split contract

如果将来它真的变成 multi-`PrimFunc` split，再补新的 projection 规则。

## 7. Truth Ownership

当前相关通道的职责必须分开：

| 通道 | 阶段 | 它承接的 truth | 主要消费者 |
|---|---|---|---|
| `blackhole.copy_semantics` | pre-split | staged copy / CB 方向 | `LowerBlackholeOps` |
| `tl.semantic_seeds` | pre-split | device program membership / pipeline skeleton | `AnalyzeSemanticStructure` / future Phase B |
| `tl.semantic_manifest_seeds` | pre-`LowerTileOp`, pre-split | 会在 `LowerTileOp` 被销毁的 explicit op payload | `ProjectSemanticManifest` |
| `tl.semantic_manifest` | post-`SplitHostDevice`, pre-lift | 统一后的 explicit algorithmic evidence | `AnalyzeSemanticStructure` |
| `blackhole.work_decomposition` | split-after | launch/work axes、derived indices | `AnalyzeSemanticStructure`, `LowerBlackholeOps` |
| `blackhole.pipeline_stages` | split-after | stage-local / pipelined loop shape | `AnalyzeSemanticStructure`, `LowerBlackholeOps` |
| `blackhole.fragment_regions` | split-after | 当前 residual structural evidence + current lowering requirements | `AnalyzeSemanticStructure`, `LowerBlackholeOps` |

这张表里的关键纪律是：

- manifest 只拿走本来就不该靠 lowered-TIR matcher 才能拿到的显式事实
- manifest 进入 `Phase A` 之后仍然必须先过 witness normalization，再能进入 semantic core
- `fragment_regions` 目前对 `LowerBlackholeOps` 仍然是正式输入，但这是过渡约束，不是最终 truth ownership 结论

## 8. 通用性约束

manifest 是否成立，不看它能不能先救一个 case，而看它是否满足下面这些通用性条件：

1. 它记录的是 op/effect-level evidence，不是 workload noun
   - 可以有 `reduce / copy / fill / gemm / cumsum`
   - 不能有 `topk / flash_attn / fusedmoe / paged_decode` 这类 family-specific tag
2. 它的关系建立必须来自 IR 对象和结构
   - tileop payload
   - buffer identity
   - ordered loop region
   - storage scope / dtype / explicit binding
   - device-region ordinal
   - 不允许名字匹配补语义
3. 它只承接会在 early / late lowering 边界被销毁、但 `Phase A` 又必须稳定使用的那类 truth
4. 新 workload family 如果还需要补 truth，优先顺序必须是：
   - 先问该 truth 是否本质上就是某个 explicit op/effect 的 payload
   - 若是，则扩 manifest collector
   - 若不是，则扩 dedicated semantic collector，或直接扩 DSL/IR/schema

## 9. Manifest 结构

最终 `tl.semantic_manifest` 仍挂在 device `PrimFunc` 上，类型仍然是 `Map<String, Any>`。

它的 schema 必须围绕 “统一后的 explicit evidence” 设计，而不是复制 `SemanticProgram`。

### 9.1 buffers

记录 manifest 内部引用的 buffer 元信息：

```python
"buffers": [
  {"buffer_id": "b0", "scope": "local.fragment", "dtype": "float32", "shape": [64, 32]},
  {"buffer_id": "b1", "scope": "local.fragment", "dtype": "float32", "shape": [64]},
  {"buffer_id": "b2", "scope": "blackhole.acc", "dtype": "float32", "shape": [64, 64]}
]
```

说明：

- `buffer_id` 是内部关系锚点
- `name` 可以保留作调试字段，但不能进入语义判断

### 9.2 operations

每个 explicit op/effect 生成一个 entry：

```python
"operations": [
  {
    "op_id": "op0",
    "kind": "copy",
    "capture_stage": "pre_lower_tile_op",
    "src_buffer": "b_src",
    "dst_buffer": "b0"
  },
  {
    "op_id": "op1",
    "kind": "reduce",
    "capture_stage": "pre_lower_tile_op",
    "src_buffer": "b0",
    "dst_buffer": "b1",
    "reduce_type": "max",
    "dim": 1,
    "clear": True
  },
  {
    "op_id": "op2",
    "kind": "gemm",
    "capture_stage": "pre_device_lower_intrin",
    "a_buffer": "bA",
    "b_buffer": "bB",
    "c_buffer": "b2",
    "payload_kind": "gemm_py"
  }
]
```

这里：

- `kind` 是 evidence kind，不是 `UpdateLaw.kind`
- `capture_stage` 只是审计字段，不是语义分支条件
- `AnalyzeSemanticStructure` 仍必须把这些 evidence 归约到已有 semantic core：
  - `map`
  - `reduce`
  - `select`
  - `recurrence`

### 9.3 serial_regions

Manifest 只记录最小 ordered-loop skeleton，以及 op 与 region 的隶属关系：

```python
"serial_regions": [
  {
    "region_id": "r0",
    "region_ordinal": 0,
    "loop_var_repr": "k",
    "extent_repr": "4",
    "op_ids": ["op0", "op1"]
  }
]
```

这里故意只做最小记录，不在 manifest 内部直接做：

- companion 推断
- recurrence 推断
- arg-reduce 推断

### 9.4 split anchors

为了跨 `SplitHostDevice` 投影，early seeds 至少要有：

```python
"device_region_ordinal": 0
```

或等价稳定锚点。

这部分不是语义本体，但没有它，早采集就落不到 device `PrimFunc`。

## 10. 它能覆盖什么，不能覆盖什么

### 10.1 可以直接覆盖的 explicit-op facts

| 事实 | 显式存在的边界 | 由谁 capture |
|---|---|---|
| `copy(src, dst)` | pre-`LowerTileOp` | early manifest seeds |
| `fill(dst, value)` | pre-`LowerTileOp` | early manifest seeds |
| `reduce(src, dst, kind, dim, clear)` | pre-`LowerTileOp` | early manifest seeds |
| `cumsum(src, dst, dim, reverse)` | pre-`LowerTileOp` | early manifest seeds |
| `gemm_py(A, B, C, ...)` | pre-device-`LowerIntrin` | late manifest augment |

### 10.2 第一阶段还不能直接覆盖的事实

| 事实 | 当前现状 | 后续前移方向 |
|---|---|---|
| `selection_targets` | 仍来自 `if_then_else` 结构恢复 | dedicated residual semantic collector，长期进入显式 `select` 表达 |
| `selection_pairs / companion` | 仍来自 shared-source overlap recovery | 与 `select` 同步前移，长期不再靠 pair overlap matcher |
| `arg_reduce_targets` | 当前不能只靠 manifest 完整给出 | 基于显式 `select/index` 关系重建 |
| `recurrence_edges / carried_from` | 当前不能只靠 manifest.loop skeleton 保证正确 | 从 ordered update + typed state/effect recovery 前移 |
| `LowerBlackholeOps` lowering requirements | 仍来自现有 split-after attrs | 迁到 dedicated lowering attr，后续再并入 Spatial / TT Target IR |

因此这份初设的准确表述是：

- **第一阶段**：先把 explicit-op slice 从 late recovery 里切出来
- **后续阶段**：继续把 selection / companion / recurrence 需要的 truth 前移到更稳定的显式表达

## 11. `AnalyzeSemanticStructure` 的变化

`AnalyzeSemanticStructure` 需要改成“统一消费 manifest + residual attrs”，但只能在职责上收窄，不允许顺手把 Phase A core 搅大。

这里最关键的约束是：

- manifest 不是 `AnalyzeSemanticStructure` 的替代品
- manifest 只是把 upstream evidence 先 canonicalize / preserve 住
- 真正把异构 evidence 收成 witness axis、再 lift 成 semantic core 的地方，仍然只能是
  `AnalyzeSemanticStructure -> LiftStatefulSemanticIR`

### 11.1 保持现状的输入

- domain axes 继续来自 `blackhole.work_decomposition`
- pipeline trait 继续来自 `blackhole.pipeline_stages`
- seeds 继续来自 `tl.semantic_seeds`

manifest 不接管这些 truth。

### 11.2 states

当前 `LocalBufferCollector` 已经能从实际 alloc buffer 收集大部分 state 候选。

因此第一阶段不要求 “state 一律 manifest-first”。

更稳的做法是：

1. 继续以真实 alloc buffer / fragment buffer 为主
2. manifest 只在需要补 explicit op anchor / scope / dtype / pre-destruction identity 时作为辅助

### 11.3 updates / witnesses

Manifest 直接替掉的，是当前对 explicit-op 的 pattern recovery：

1. `reduce`
   - 直接从 manifest 生成 `reduce_*` update entry
   - 直接生成对应 `law_family=reduce` / `source_set` witness
2. `copy / fill / gemm / cumsum`
   - 不引入新的 `UpdateLaw.kind`
   - 只作为 `map`-family 的 evidence / anchor / supplement 输入
   - 如果当前 Phase A 还不需要把它们落成独立 update，就不要硬加

### 11.4 residual recovery

下面这些在第一阶段仍保留在 `fragment_regions` 路径里：

- `selection_targets`
- `selection_pairs`
- `arg_reduce_targets`
- `recurrence_edges`
- 当前 `LowerBlackholeOps` 所需的 lowering requirements

也就是说，`AnalyzeBlackholeFragmentRegions` 在第一阶段不会被删。

但这不能被写成长期架构终点。正确表述应该是：

- 第一阶段：保留 residual recovery
- 后续阶段：继续把 semantic facts 从 `fragment_regions` 移走
- 直到它只剩兼容职责，或者直接被删除

## 12. Pass 设计

### 12.1 `CollectSemanticManifestSeeds`

- 阶段：
  - `LowerAndLegalize`
  - 位置：不晚于 `LowerTileOp`
- 输入：
  - pre-split `PrimFunc`
  - 只扫描 device region
- 输出：
  - `tl.semantic_manifest_seeds`

收集规则：

1. 识别 `tl.tileop.copy / fill / reduce / cumsum / gemm`
2. 记录 op payload、buffer 元信息、最小 ordered-loop skeleton
3. 记录 device-region ordinal
4. 不在这里做 companion / recurrence / arg-reduce 推断

说明：

- 实现上可以是独立 pass，也可以和 `LowerTileOp` 共用 parser
- 但架构约束不变：
  - capture 必须发生在 `LowerTileOp` 销毁这些 payload 之前
  - 它只负责 evidence capture，不负责 semantic recovery 本体

### 12.2 `ProjectSemanticManifest`

- 阶段：
  - `OptimizeForTarget`
  - 位置：`SplitHostDevice` 之后
- 输入：
  - `tl.semantic_manifest_seeds`
  - `tl.device_programs` / split order
- 输出：
  - 每个 device `PrimFunc` 上的 `tl.semantic_manifest`

职责：

1. 按稳定 split 锚点选择属于该 device kernel 的 manifest slice
2. 把 early-captured evidence 附着到新 device `PrimFunc`
3. 不引入新的语义推断

### 12.3 `AugmentSemanticManifest`

- 阶段：
  - `blackhole_codegen`
  - 位置：`LowerDeviceStorageAccessInfo` 之后、device-side `LowerIntrin` 之前
- 输入：
  - 已投影到 device `PrimFunc` 的 `tl.semantic_manifest`
  - 还活着的 explicit residual op
- 输出：
  - 增补后的 `tl.semantic_manifest`

收集规则：

1. 扫描 `tl.tileop.gemm_py`
2. 把 payload 规范化成 manifest op entry
3. 只补增，不重做 early-captured slice
4. 不直接生成 witness / semantic core

## 13. 迁移路径

### 13.1 Phase 1：先把 explicit-op slice 建成统一输入

1. 新增 `CollectSemanticManifestSeeds`
2. 新增 `ProjectSemanticManifest`
3. 新增 `AugmentSemanticManifest`
4. `AnalyzeSemanticStructure` 同时读取：
   - `tl.semantic_manifest`
   - `blackhole.fragment_regions`
   - `blackhole.work_decomposition`
   - `blackhole.pipeline_stages`
   - `tl.semantic_seeds`
5. manifest 只替换 explicit-op evidence 输入
6. explicit-op 的 witness/core 仍由 `AnalyzeSemanticStructure` / `LiftStatefulSemanticIR` 统一产出
7. `fragment_regions` 继续承接 residual recovery 与 `LowerBlackholeOps` 当前输入

### 13.2 Phase 2：把 semantic facts 持续从 `fragment_regions` 切走

- 对 manifest 已覆盖的 `reduce / copy / fill / cumsum / gemm`
  - `AnalyzeSemanticStructure` 不再读 `fragment_regions` 的对应 slice
- 对 `selection / companion / arg-reduce / carried-from`
  - 继续引入更稳定的前移表达
- 到这一阶段结束时，`fragment_regions` 对 `AnalyzeSemanticStructure` 应退化成 compatibility fallback，而不是主信息源

建议把这一步再拆成两个子任务：

1. `Phase 2A`
   - 新增 residual semantic collector
   - 输出不再叫 `fragment_regions`
2. `Phase 2B`
   - `AnalyzeSemanticStructure` 改成只吃：
     - `tl.semantic_manifest`
     - residual semantic collector
     - `tl.semantic_seeds`
     - `work_decomposition / pipeline_stages`

### 13.3 Phase 3（长期）

继续完成两件事：

1. semantic 侧
   - 把 `selection / companion / arg-reduce / carried-from` 从 `fragment_regions`
     拆到 dedicated collector
   - 如果 DSL/IR 显式引入 `T.select` / `T.argreduce` / recurrence annotations，
     就继续把 residual facts 前移到显式表示
2. target 侧
   - 把 `LowerBlackholeOps` 当前依赖的 lowering requirements 从 `fragment_regions`
     迁到 dedicated lowering attr / Spatial IR / TT Target IR 物化链

## 14. 不做的事

1. 不改 DSL 用户接口
2. 不改 `SemanticProgram` core 结构
3. 不新增 `UpdateLaw.kind`
4. 不把 manifest 变成第二套 semantic schema
5. 不把 `LowerBlackholeOps` 的 target-facing lowering contract 偷偷混进 manifest 第一阶段职责
6. 不声称仅靠 loop skeleton 就能稳定恢复 `carried_from`
7. 不依赖函数名、buffer 名或 helper 名做语义恢复或 split projection

## 15. 验证方式

1. 现有 `test_blackhole_semantic_ir.py` 继续通过
2. 新增 early-capture 测试
   - `test_manifest_seeds_capture_reduce_before_lower_tile_op`
   - `test_manifest_seeds_capture_copy_before_lower_tile_op`
   - `test_manifest_seeds_capture_fill_before_lower_tile_op`
3. 新增 projection 测试
   - `test_manifest_projection_attaches_region_slice_to_device_kernel`
4. 新增 late-augment 测试
   - `test_manifest_augment_captures_gemm_py_before_lower_intrin`
5. 新增 `AnalyzeSemanticStructure` 集成测试
   - explicit-op `reduce` witness 来自 manifest，而不是 `fragment_regions`
   - manifest 缺失时仍能 fallback 到旧路径
6. 新增回归测试确保 `LowerBlackholeOps` 当前 lowering requirements 不回退

## 16. 与已有设计的关系

- `tl.semantic_seeds`
  - 仍是 pre-split typed input
  - 负责 device-program / pipeline skeleton
  - 不负责 explicit op payload
- `blackhole.copy_semantics`
  - 仍是 pre-split target-facing metadata
  - 不进入 `Phase A` semantic core
- `tl.semantic_manifest_seeds`
  - 新增的 pre-`LowerTileOp` seed 通道
  - 负责捕获会在 `LowerTileOp` 被销毁的 explicit evidence
- `tl.semantic_manifest`
  - 是投影并补全后的 device-side explicit evidence 视图
  - 只服务 `Phase A` 的 evidence normalization
- `blackhole.fragment_regions`
  - 第一阶段仍保留两个职责：
    - residual semantic recovery
    - `LowerBlackholeOps` lowering requirements
  - 但这两项都不应被写成长期架构终点
- `tl.semantic_structure / tl.semantic_witnesses / tl.semantic_program`
  - 仍由 `AnalyzeSemanticStructure` / `LiftStatefulSemanticIR` 产出
  - manifest 只是它们的上游 evidence source 之一
