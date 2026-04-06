# Stage 4 Phase A: Stateful Semantic IR

## 基本信息

- **文档角色**: `Phase A` 工程实现与边界文档
- **当前状态**: `2026-04-06` 按设计边界已完成；当前作为实现参考与 `Phase B` 输入边界保留
- **唯一总体设计**: `tasks/dev_design/final_blackhole_backend_redesign.md`
- **并行理论文档**: `tasks/dev_design/stage4_phase_a_formalization_note.md`

## 1. Phase A 的职责

`Phase A` 只负责一件事：

- 冻结 workload 的 **algorithmic truth**

它不负责：

- `ProgramPhase / Task / Channel / Layout / WorkPartition`
- `TT resource / transport / ABI`
- 为后段保留“再猜一次上层语义”的空间

因此，`Phase A` 的长期语义边界是：

- `SemanticProgram`
- `Domain`
- `State`
- `Update`
- `UpdateLaw`
- `AccessMap`
- `SemanticSupplement`

以及少量固定语义轴：

- state role
- update-law kind
- access trait

`Phase A` 的设计纪律仍然是：

1. 不允许名字匹配恢复语义
2. 不允许 workload-specific noun 进入长期协议
3. analysis evidence 必须能归约到 semantic core
4. 不能归约的 truth 必须二选一：
   - 若它是跨 family 复用的基础语义轴，则扩 core
   - 否则进入 `Phase B` 或 `Phase C`

像下面这些对象，只能作为 pre-lift evidence，不能成长为第二套 schema：

- `selection_targets`
- `selection_pairs`
- `arg_reduce_targets`
- `update_sources`
- `recurrence_edges`

## 2. 当前已落地的退出状态

### 2.1 A1 最小语义层

`Phase A1` 已完成并稳定接入主链：

- 最小对象集：
  - `SemanticProgram`
  - `Domain`
  - `State`
  - `Update`
  - `UpdateLaw`
  - `AccessMap`
  - `TIRAnchor`
  - `TIRValueBinding`
- 主链：
  - `AnalyzeSemanticStructure`
  - `LiftStatefulSemanticIR`
  - `ValidateStatefulSemanticIR`
- 当前主设备链中，semantic lift 已位于 `LowerBlackholeOps` 之前

### 2.2 A2 语义扩面

`Phase A2` 已完成，当前语义层已明确承接：

- workload-agnostic state roles：
  - `carry`
  - `reduction_accumulator`
  - `selection_state`
  - `index_state`
  - `transient`
- wider `UpdateLaw.kind`：
  - `map`
  - `reduce`
  - `select`
  - `recurrence`
- wider `AccessMap.traits`
- typed `SemanticSupplement`

当前 `flash-attn / topk / chunk recurrence` 只作为 validation family：

- 不进入 schema 命名空间
- 不以 workload noun 参与协议分派

### 2.3 进入 Phase B 前的收口加固

`Phase A` 为避免继续膨胀，已补齐以下机制：

- generic witness algebra：
  - `tl.semantic_witnesses`
- typed vocabulary / decoder / rule table：
  - `semantic_vocab`
  - `semantic_witness_decoder`
  - `semantic_refinement_rules`
- typed payload family：
  - `semantic_witness_payloads`
- stronger refinement validation：
  - `ValidateSemanticRefinement`
- internal state/effect normalization：
  - `StateVersion`
  - `StateDef`
  - `StateUse`
  - `StateJoin`
- companion lifecycle contract：
  - `preserve`
  - `typed_rebind`
  - `invalidate`
- audited-safe rebind pass：
  - `TypedRebindBlackholeCompanionPrograms`

## 2.4 信任边界与已知假设

`Phase A` 的 witness quality 取决于上游 evidence source 的质量。当前已经分成两类：

- `AnalyzeBlackholeWorkDecomposition` 决定 domain skeleton
- `CollectSemanticManifestSeeds -> ProjectSemanticManifest -> AugmentSemanticManifest`
  决定 explicit-op evidence 与 manifest-backed structural evidence
  （当前包括 `fragment_buffers / selection_targets / selection_pairs / arg_reduce_targets /
  update_sources / loop_carried_state / recurrence_edges`）
- `AnalyzeBlackholeFragmentRegions` 当前退化为 compatibility fallback / residual reduction evidence
- `AnalyzeBlackholePipelineStages` 决定 pipeline trait

这里要特别区分两件事：

- `FragmentRegionAnalyzer` 这套**分析逻辑**本身仍然有用
- `blackhole.fragment_regions` 这个 **attr 形态** 已不再应该承担 semantic truth ownership

当前之所以还保留 `blackhole.fragment_regions`，是因为 `row_reductions` 仍是 mixed ownership：

- semantic 侧还会用它恢复 `reduce_*` update
- lowering 侧的 `LowerBlackholeOps` 也还会直接消费它做 lowering-facing summary

因此，当前不能把 `row_reductions` 简单理解成“还没迁完的 semantic truth”，它同时也是
未拆分完成的 lowering compatibility contract。

`Phase A` 内部的 `AnalyzeSemanticStructure -> LiftStatefulSemanticIR -> ValidateSemanticRefinement` 只能保证：
给定上游 evidence，lift 和 validation 是正确的。但如果上游 evidence 本身遗漏或错误，`Phase A` 不会发明出正确语义。

因此，扩展新 workload family（如 fusedmoe、paged decode）时，**首先要确认上游 manifest / fragment evidence source 能正确收集该 family 的 evidence**，然后才是 `Phase A` 的 witness/core/validator 是否覆盖。

另外，`CanonicalBufferName`（`analyze_semantic_structure.cc`）假设 lowering 只在 buffer 名末尾追加 `_<digits>` suffix。如果未来 lowering pass 改变命名规则，这里需要同步更新。

`TypedRebindBlackholeCompanionPrograms` 在 rebind 时会重建 state/effect graph（调用 `BuildStateEffectGraph`），不会沿用旧 graph。`body_hash` 校验确保 body 不会在 rebind 后被静默修改。

## 3. 当前代码落点

核心代码面现在集中在：

- `tilelang_repo/src/transform/analyze_semantic_structure.cc`
- `tilelang_repo/src/transform/lift_stateful_semantic_ir.cc`
- `tilelang_repo/src/transform/validate_stateful_semantic_ir.cc`
- `tilelang_repo/src/transform/validate_semantic_refinement.cc`
- `tilelang_repo/src/transform/typed_rebind_blackhole_companion_programs.cc`
- `tilelang_repo/src/transform/common/semantic_program.h`
- `tilelang_repo/src/transform/common/semantic_vocab.h`
- `tilelang_repo/src/transform/common/semantic_witness_decoder.h`
- `tilelang_repo/src/transform/common/semantic_witness_payloads.h`
- `tilelang_repo/src/transform/common/semantic_refinement_rules.h`
- `tilelang_repo/src/transform/common/semantic_state_effect_graph.h`

当前 Blackhole 设备侧 pass 主线中与 `Phase A` 直接相关的部分是：

```text
AugmentSemanticManifest
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

`Phase A` 当前使用并维护的 companion attrs 包括：

- `tl.semantic_seeds`
- `tl.semantic_manifest_seeds`
- `tl.semantic_manifest`
- `tl.semantic_structure`
- `tl.semantic_witnesses`
- `tl.semantic_program`
- `tl.semantic_hard_freeze`
- `tl.companion_invalidation_reason`

## 4. 一个端到端例子：`topk / selection`

下面这个例子直接对应真实 `example_topk.tl_topk.jit_impl.get_tir(M=64, N=32, topk=4, blk_m=64, threads=128)` 的
输出，而不是抽象化的伪代码。之所以选它，是因为这一个 kernel 里同时存在：

- value reduction：`max_val`
- index companion：`expand_max_idx`
- final arg-reduce index：`max_idx`
- 迭代式 mask / carry 行为：`logits_frag`

也就是说，它能把 `Phase A` 里最关键的三类 truth 一次看清：

- `reduce`
- `select`
- `recurrence`

当前这条链路也已经被真实测试覆盖：

- `test_topk_semantic_program_lifts_select_updates_and_selection_roles`
- `test_selection_pairing_is_recovered_from_compute_pattern`
- `test_topk_fragment_analysis_recovers_arg_reduce_targets`
- `test_topk_semantic_witnesses_expose_generic_fact_axes`

另外，文档最后会专门说明：为了证明这套恢复不是靠名字匹配，仓库里还额外有两个 synthetic test
去掉名字提示，只保留 IR 结构。

### 4.1 这个例子本身在算什么

先说业务位置：这个 `topk` 在 LLM 里最典型的用途是 **MoE routing**，不是生成阶段常说的
`top-k sampling`。

在这个例子里：

- `logits`
  - shape 是 `[M, N]`
  - 表示 `M` 个 token 对 `N` 个 candidate expert 的 router 分数
- `topk_gates`
  - shape 是 `[M, topk]`
  - 表示每个 token 最终选中的前 `k` 个 expert 分数
- `topk_indices`
  - shape 是 `[M, topk]`
  - 表示每个 token 最终选中的前 `k` 个 expert 编号

把它放回完整的 MoE LLM 流程里，可以理解成：

1. router / gate 网络先算出每个 token 对每个 expert 的分数，也就是这里的 `logits`
2. 这个 `topk` kernel 为每个 token 选出前 `k` 个 expert
3. 后面的 routed dispatch / grouped dispatch 会按 `topk_indices` 把 token 发到对应 expert
4. combine / weighted merge 会继续消费 `topk_gates`

所以这个例子的业务含义不是“把一堆数排个序”这么简单，而是：

- 决定 token 去哪些 expert
- 决定每个 expert 的路由权重

这也是为什么它特别适合拿来解释 `Phase A`。从语义角度看，它天然包含：

- selection / indexing family
  - 从一整行 candidate 里选出少数目标 expert
- companion index flow
  - 选中的 value 和选中的 expert id 必须成对流动
- carried state
  - 选中过的位置会被 mask 掉，再进入下一轮 `k+1`

也就是说，这个例子不是随便挑的一个小 kernel，而是 LLM 里很典型的
`selection -> routing` 前置步骤，正好能把 `Phase A` 想冻结的 algorithmic truth 讲清楚。

在进入 IR 之前，先看这个例子在 DSL 里到底写了什么。`tilelang_repo/examples/topk/example_topk.py`
里的核心 kernel 是：

```python
with T.Kernel(T.ceildiv(M, blk_m), threads=threads) as bx:
    logits_frag = T.alloc_fragment([blk_m, N], dtype=dtype)
    max_val = T.alloc_fragment([blk_m], dtype=dtype)
    expand_max_idx = T.alloc_fragment([blk_m, N], T.int32)
    max_idx = T.alloc_fragment([blk_m], T.int32)

    T.copy(logits[bx * blk_m, 0], logits_frag)

    for k in T.serial(topk):
        T.fill(expand_max_idx, -1)
        T.reduce_max(logits_frag, max_val, dim=1, clear=True)

        for i, j in T.Parallel(blk_m, N):
            expand_max_idx[i, j] = T.if_then_else(
                max_val[i] == logits_frag[i, j], j, expand_max_idx[i, j]
            )

        T.reduce_max(expand_max_idx, max_idx, dim=1, clear=True)

        for i, j in T.Parallel(blk_m, N):
            logits_frag[i, j] = T.if_then_else(
                max_val[i] == logits_frag[i, j], -10000.0, logits_frag[i, j]
            )

        for i in T.Parallel(blk_m):
            topk_gates[bx * blk_m + i, k] = max_val[i]
            topk_indices[bx * blk_m + i, k] = max_idx[i]
```

先把这段代码里的记号讲清楚：

- `M`
  - 输入矩阵 `logits` 的总行数，也就是总 token / row 数。
- `N`
  - 每一行的列数，也就是每个 token 要在多少个 candidate 上做 `topk`。
- `topk`
  - 每一行要选几轮最大值；这里的外层 `for k in T.serial(topk)` 就是在做第 `k` 轮选择。
- `blk_m`
  - 一个 kernel block 一次处理多少行。
- `bx`
  - `T.Kernel(... ) as bx` 里的 block 索引。第 `bx` 个 block 负责输入里的
    `bx * blk_m : (bx + 1) * blk_m` 这段行范围。
- `i`
  - block 内的行下标，也就是当前 block 负责的第几行。
- `j`
  - 当前行内的列下标，也就是 candidate / expert / column 位置。
- `tx`
  - lowering 到 Blackhole TIR 之后出现的 `threadIdx.x` 线程索引，用来把 fragment 内的并行访问展开到具体线程。
- `logits_frag`
  - 当前 block 的行片段，存在 `blackhole.acc` 里，是后面多轮 `topk` 都会反复更新的工作集。
- `max_val`
  - 当前第 `k` 轮、每一行选出来的最大值。
- `expand_max_idx`
  - 把“哪一列命中了当前最大值”扩展到整行上的中间 companion/index buffer。
- `max_idx`
  - 当前第 `k` 轮最终归并出来的列号。
- `topk_gates`
  - 输出 value，shape 是 `[M, topk]`。
- `topk_indices`
  - 输出 index，shape 是 `[M, topk]`。

这个算法做的事情很直接：对每一行 `logits`，连续做 `topk` 轮“找最大值并把它拿走”。

可以按下面这个顺序理解：

1. `T.copy(..., logits_frag)`
   - 先把当前 block 的一批行搬到 fragment-local `logits_frag` 里。
2. `T.reduce_max(logits_frag, max_val, dim=1, clear=True)`
   - 对每一行先算出当前最大值 `max_val[i]`。
3. `expand_max_idx[i, j] = if_then_else(max_val[i] == logits_frag[i, j], j, -1)`
   - 在每个位置上判断“这一列是不是当前最大值”，命中就写列号 `j`，没命中保持 `-1`。
4. `T.reduce_max(expand_max_idx, max_idx, dim=1, clear=True)`
   - 再把这一整行 candidate index 归并成最终列号 `max_idx[i]`。
5. `logits_frag[i, j] = if_then_else(max_val[i] == logits_frag[i, j], -10000.0, logits_frag[i, j])`
   - 把这轮选中的位置 mask 掉，这样下一轮 `k+1` 不会再选到它。
6. `topk_gates[..., k] = max_val[i]` / `topk_indices[..., k] = max_idx[i]`
   - 把这一轮选到的 value 和 index 写到第 `k` 个输出槽位。

所以这个例子同时包含三类语义：

- `max_val`
  - 每一轮的 value reduction
- `expand_max_idx -> max_idx`
  - 从 value condition 派生出来的 companion/index flow
- `logits_frag`
  - 被重复更新、mask、再参与下一轮 reduction 的 carried state

这正是它适合做 `Phase A` walkthrough 的原因：算法本身已经天然同时包含
`reduce + select + recurrence` 三种语义。

### 4.2 初始输入：真实 TIR 里先出现了什么

在 `SplitBlackholeKernel` 之后，这个 kernel 的关键 body 片段已经长成下面这样：

```python
bx = T.launch_thread("blockIdx.x", 1)
logits_frag = T.decl_buffer((16,), scope="blackhole.acc")
max_val = T.decl_buffer((4,), scope="blackhole.acc")
expand_max_idx = T.decl_buffer((16,), "int32", scope="blackhole.acc")
max_idx = T.decl_buffer((4,), "int32", scope="blackhole.acc")
tx = T.launch_thread("threadIdx.x", 128)

for i in T.unroll(4):
    for vec in T.vectorized(4):
        logits_frag_1[i * 4 + vec] = logits[i * 16 + T.shift_right(tx, 3), T.bitwise_and(tx, 7) * 4 + vec]

max_val_1[i_2] = T.max(max_val_1[i_2], logits_frag_1[i_2 * 4 + rv])
max_val_1[i_2] = T.call_extern("float32", "tl::AllReduce<tl::MaxOp, 8, 1, 0>::run", max_val_1[i_2])

expand_max_idx_1[i_3] = T.if_then_else(
    max_val_1[T.shift_right(i_3, 2)] == logits_frag_1[i_3],
    T.bitwise_and(tx, 7) * 4 + T.bitwise_and(i_3, 3),
    expand_max_idx_1[i_3],
)

max_idx_1[i_4] = T.max(max_idx_1[i_4], expand_max_idx_1[i_4 * 4 + rv])
max_idx_1[i_4] = T.call_extern("int32", "tl::AllReduce<tl::MaxOp, 8, 1, 0>::run", max_idx_1[i_4])

logits_frag_1[i_5] = T.if_then_else(
    max_val_1[T.shift_right(i_5, 2)] == logits_frag_1[i_5],
    T.float32(-10000.0),
    logits_frag_1[i_5],
)

topk_gates[i_6 * 16 + T.shift_right(tx, 3), k] = max_val_1[i_6]
topk_indices[i_6 * 16 + T.shift_right(tx, 3), k] = max_idx_1[i_6]
```

这里还要补一个读 TIR 的小提示：

- `bx` 和 DSL 里的 `bx` 含义一样，仍然是 block 索引。
- `tx` 是 lowering 后显式出现的 `threadIdx.x`。
- `i_2 / i_3 / i_4 / i_5 / i_6 / rv / vec` 这些带后缀的名字只是 lowering 之后拆出来的 loop var /
  reduction var；它们本身不带额外算法语义，不要把注意力放在这些编号上。
- 真正重要的是：
  - 这些循环体在读写哪个 fragment buffer
  - 它们之间的 def-use / compare / reduce / overwrite 关系是什么

也就是说，读这段 TIR 时，重点不是“`i_4` 是谁”，而是：

- `max_val_1` 是怎么从 `logits_frag_1` 里归约出来的
- `expand_max_idx_1` 是怎么依赖 `max_val_1` 和 `logits_frag_1` 的
- `max_idx_1` 是怎么再从 `expand_max_idx_1` 里归约出来的
- `logits_frag_1` 是怎么在每轮之后被 mask 掉的

这段真实 IR 已经包含了后面要恢复语义所需的结构事实：

- `max_val` 对 `logits_frag` 做 row reduction
- `expand_max_idx` 通过 `if_then_else` 和 `max_val` / `logits_frag` 关联起来
- `max_idx` 对 `expand_max_idx` 再做一次 index-style reduction
- `logits_frag` 在每轮 `k` 之后被重新写回，形成 recurrence-like carry

可以按下面这个顺序直接读这段 IR：

1. `logits_frag = decl_buffer((16,), scope="blackhole.acc")`
   - 说明这里有一个 fragment-local 浮点 state，后面很多 update 都围绕它展开。
2. `expand_max_idx = decl_buffer((16,), "int32", scope="blackhole.acc")`
   - 说明这里还有一个 fragment-local 整型 state，它不是最终输出，但会先承接 companion/index 信息。
3. `logits_frag_1[...] = logits[...]`
   - 先把当前 tile 的 logits 搬进 fragment buffer，后面所有 reduction/select 都发生在 fragment-local state 上。
4. `max_val_1[i_2] = T.max(... logits_frag_1 ...)`
   - 这是第一层 value reduction；它只看 value，不直接生成 index。
5. `expand_max_idx_1[i_3] = T.if_then_else(max_val == logits_frag, lane_id, old_value)`
   - 这一步才把 “哪个位置命中了当前 max” 编进 companion/index state。
   - 它同时依赖 `max_val` 和 `logits_frag`，所以后面 `update_sources` 会把这两个都记下来。
6. `max_idx_1[i_4] = T.max(... expand_max_idx_1 ...)`
   - 把 fragment 内部的 candidate index 再归并成最终 index。
7. `logits_frag_1[i_5] = T.if_then_else(..., -10000.0, logits_frag_1[i_5])`
   - 把当前已选中的位置 mask 掉，为下一轮 `k` 做准备。
   - 这就是为什么 `logits_frag` 后面不仅参与 `select`，还会参与 `recurrence`。
8. `topk_gates[...] = max_val_1[...]` / `topk_indices[...] = max_idx_1[...]`
   - 这里才是最终 materialization；也就是说，前面这些 `blackhole.acc` buffer 才是 `Phase A` 真正关心的算法 state。

但在这一步，语义仍然只是结构事实：

- 还没有 `State.role`
- 还没有 `UpdateLaw.kind`
- 也还没有 `paired_value_state`

### 4.3 `AnalyzeBlackholeWorkDecomposition`：先固定 domain skeleton

这个例子的真实输出是：

```python
{"axes": ["bx"], "derived_index_exprs": [], "work_dependent_loop_bounds": []}
```

这一步只告诉 `Phase A`：

- 当前 semantic domain 至少锚在 `bx` 这个 work axis 上

它刻意不做的事是：

- 不去判断 `max_idx` 是不是 index state
- 不去判断 `expand_max_idx` 是不是 selection companion

原因很直接：domain truth 和 state/update truth 必须分开。`SemanticProgram.Domain` 需要真实 work
decomposition，但 work analysis 本身不应该偷着长成 semantic recovery。

### 4.4 `AnalyzeBlackholeFragmentRegions`：把结构事实收成可消费 evidence

这个例子最关键的一步，是把 fragment-local 结构收成下面这份真实 attr：

```python
{
  "fragment_buffers": [
    {"name": "logits_frag", "scope": "blackhole.acc", "is_integer": 0},
    {"name": "expand_max_idx", "scope": "blackhole.acc", "is_integer": 1},
    {"name": "max_val", "scope": "blackhole.acc", "is_integer": 0},
    {"name": "max_idx", "scope": "blackhole.acc", "is_integer": 1},
  ],
  "row_reductions": [
    {"target": "max_val", "kind": "max"},
    {"target": "max_idx", "kind": "max"},
  ],
  "arg_reduce_targets": ["max_val", "max_idx"],
  "selection_targets": ["expand_max_idx", "logits_frag"],
  "selection_pairs": [
    {
      "value_target": "max_val",
      "companion_target": "expand_max_idx",
      "source_states": ["logits_frag"],
    }
  ],
  "update_sources": [
    {"target": "max_val", "sources": ["logits_frag"]},
    {"target": "expand_max_idx", "sources": ["logits_frag", "max_val"]},
    {"target": "max_idx", "sources": ["expand_max_idx"]},
    {"target": "logits_frag", "sources": ["max_val"]},
  ],
  "loop_carried_state": [
    {"name": "logits_frag"},
    {"name": "expand_max_idx"},
    {"name": "max_val"},
    {"name": "max_idx"},
  ],
  "recurrence_edges": [
    {"target": "logits_frag", "source_states": ["max_val"]},
    {"target": "expand_max_idx", "source_states": ["logits_frag", "max_val"]},
    {"target": "max_val", "source_states": ["logits_frag"]},
    {"target": "max_idx", "source_states": ["expand_max_idx"]},
  ],
}
```

这一层已经很强，但仍然是 evidence，不是最终 semantic truth。这里的设计点正好能从这份 attr 看出来：

- `selection_pairs` 只是 “companion relation evidence”
- `arg_reduce_targets` 只是 “arg-reduce evidence”
- `update_sources` 只是 “source-flow evidence”

它们都还不是长期 schema。后面如果直接让 `Phase B/C` 继续消费这堆 attr，就会重新变成一套 ad hoc
协议；所以它们必须先被规范化再 lift。

### 4.5 `AnalyzeBlackholePipelineStages`：这个例子里它可以为空

这个例子还有个很适合写清楚的点：

- `AnalyzeBlackholePipelineStages` 跑了
- 但 `blackhole.pipeline_stages` 在这个具体 kernel 上并不存在

也就是说，这一步不是“每个例子都一定会给 `Phase A` 新增一份 pipeline truth”，而是：

- `Phase A` 必须工作在已经过 canonical pipeline analysis 的宿主上
- 如果某个 kernel 没有可记录的 pipeline stage，就明确为空，而不是下游默认猜一个

这也解释了为什么 `AnalyzeSemanticStructure` 只是在 attr 存在时给 domain 补 `pipeline` trait。

### 4.6 `AnalyzeSemanticStructure`：把异构 evidence 变成统一 witness

到这一步，`Phase A` 才开始把上面的 evidence 收成统一的 `tl.semantic_witnesses`。这个例子里的真实
witness 片段长这样：

```python
{"subject_kind": "state", "subject_anchor_id": "logits_frag",
 "fact_axis": "role", "fact_value": {"role": "selection_state"}}
{"subject_kind": "state", "subject_anchor_id": "expand_max_idx",
 "fact_axis": "role", "fact_value": {"role": "index_state"}}
{"subject_kind": "update", "subject_anchor_id": "select_expand_max_idx",
 "fact_axis": "law_family", "fact_value": {"kind": "select"}}
{"subject_kind": "update", "subject_anchor_id": "select_expand_max_idx",
 "fact_axis": "source_set", "fact_value": {"sources": ["logits_frag", "max_val"]}}
{"subject_kind": "relation", "subject_anchor_id": "select_expand_max_idx",
 "fact_axis": "companion", "related_anchor_ids": ["max_val"],
 "fact_value": {"binding_kind": "paired_value_state"}}
{"subject_kind": "relation", "subject_anchor_id": "max_idx",
 "fact_axis": "derives_index_from", "fact_value": {}}
{"subject_kind": "update", "subject_anchor_id": "recur_logits_frag",
 "fact_axis": "ordering", "fact_value": {"ordering": "ordered"}}
```

这一步最重要的变化不是“更接近最终结果”，而是“协议被收正了”：

- fragment analysis 产出的异构 attr，被压成统一 witness axes
- witness 已经不再暴露 `topk` 这种 workload noun
- downstream 不需要理解 `selection_pairs` 和 `arg_reduce_targets` 的局部格式，只需要理解通用 axis：
  - `state.role`
  - `update.law_family`
  - `update.source_set`
  - `relation.companion`
  - `relation.derives_index_from`
  - `update.ordering`

这就是为什么 `AnalyzeSemanticStructure` 必须单独存在。没有这层，`LiftStatefulSemanticIR` 就会直接绑死在
具体 analysis attrs 上，慢慢重新变回 case-by-case matcher。

### 4.7 `LiftStatefulSemanticIR`：把 witness 冻结成小闭集 semantic core

在这个例子里，lift 之后的真实 `SemanticProgram` 关键信息可以压成下面这些条目：

```python
states = [
  {"name": "logits_frag", "role": "selection_state"},
  {"name": "expand_max_idx", "role": "index_state"},
  {"name": "max_val", "role": "reduction_accumulator"},
  {"name": "max_idx", "role": "index_state"},
]

updates = [
  {"name": "reduce_max_val", "law_kind": "reduce", "source_states": ["logits_frag"]},
  {"name": "reduce_max_idx", "law_kind": "reduce", "source_states": ["expand_max_idx"]},
  {"name": "select_logits_frag", "law_kind": "select", "source_states": ["max_val"]},
  {
    "name": "select_expand_max_idx",
    "law_kind": "select",
    "source_states": ["logits_frag", "max_val"],
    "bindings": [{"kind": "paired_value_state", "value_repr": "max_val"}],
  },
  {"name": "recur_logits_frag", "law_kind": "recurrence", "source_states": ["max_val"]},
]
```

当前实现里，`max_val` 被 lift 成 `reduction_accumulator`，`max_idx` 被 lift 成 `index_state`。
虽然上游 fragment analysis 的 `arg_reduce_targets` 同时包含 `max_val` 和 `max_idx`，但 `Phase A`
只对 **integer** arg-reduce target 赋予 `index_state` 角色。`max_val` 是 float 值，不携带 index
信息，因此保持 `reduction_accumulator`。

这里能直接看出 `Phase A` 的核心设计：

- 它冻结的是 workload-agnostic truth
- 它不保留 `selection_pairs / arg_reduce_targets / update_sources` 这些局部 analysis attrs
- 它只保留 `State / Update / UpdateLaw / bindings / normalized graph`

对这个 `topk` 例子来说，最重要的冻结结果是：

- `expand_max_idx` 已经不再只是 `selection_targets` 里的一个字符串，而是一个 `index_state`
- `max_val -> expand_max_idx` 的 companion 关系，已经不再是局部 pair，而是 `select_expand_max_idx`
  update 上的 `paired_value_state`
- `logits_frag` 的每轮重写，也不再只是 body 里的 `if_then_else`，而是 `recurrence` update

### 4.8 internal graph：为什么 `StateVersion / Def / Use / Join` 不是装饰

如果只看 `State` 和 `Update`，这个例子其实还不够稳，因为很多依赖关系还会是隐式的。真实 lift 之后，
这个 kernel 还会长出下面这些 graph object：

```python
state_uses += [
  {"consumer_update": "select_expand_max_idx", "state_name": "max_val", "kind": "companion_state"},
  {"consumer_update": "recur_logits_frag", "state_name": "max_val", "kind": "carried_state"},
]

state_joins += [
  {
    "name": "join_recur_logits_frag",
    "state_name": "logits_frag",
    "kind": "loop_carried",
    "input_versions": ["update_select_logits_frag_out", "update_recur_logits_frag_out"],
    "output_version": "update_recur_logits_frag_out",
  }
]
```

这层 graph 的作用在这个例子里很直观：

- `paired_value_state` 不再只是 binding 注释，还会落成 companion `StateUse`
- recurrence 不再只是 `law_kind = recurrence`，还要配套 `carried_state` use 和 `loop_carried`
  join

所以 `StateVersion / Def / Use / Join` 不是实现细节，而是 semantic truth 被真正规范化的地方。

### 4.9 `ValidateStatefulSemanticIR`：先保证它是自洽的语言对象

对这个例子，这一步不是重新推理语义，而是保证 `SemanticProgram` 至少结构闭合：

- 每个 `Update.state_name` 都能在 states 里找到
- 每个 `StateVersion / StateDef / StateUse / StateJoin` 的引用都一致
- `State.role` / `UpdateLaw.kind` / graph kind 都属于受支持闭集

也就是说，这一层保证的是：

- semantic core 至少是一个合法对象

而不是：

- semantic core 一定已经被证据证明

### 4.10 `ValidateSemanticRefinement`：再证明这不是 lift 自己编出来的

这一层才真正把 witness 和 semantic core 对起来。对这个例子，它会逐条检查：

- `state.role(logits_frag)` witness 是否真的对应 `selection_state`
- `update.law_family(select_expand_max_idx)` witness 是否真的对应 `select`
- `update.source_set(select_expand_max_idx)` 是否真的落成 `["logits_frag", "max_val"]`
- `relation.companion(select_expand_max_idx -> max_val)` 是否真的落成 binding + companion `StateUse`
- `relation.derives_index_from(max_idx)` 是否真的只指向 `index_state`
- `update.ordering(recur_*)` 是否真的有 `loop_carried` join

这一步的价值是：

- `SemanticProgram` 不是 `AnalyzeSemanticStructure` 说了算
- 它必须能被 witness 反向核对
- 如果后面 body 变了，contract 要么 `typed_rebind`，要么 `invalidate`

### 4.11 这个真实例子最后说明了什么

把上面这一整条真实链路连起来，`Phase A` 的核心思想其实很简单：

1. 先从真实 TIR 收集可验证结构证据
2. 再把异构 evidence 归一化成统一 witness axes
3. 然后把 witness 冻结成小闭集 semantic core
4. 最后用 validator 保证 semantic truth 不是 lift pass 自己发明的

所以 `Phase A` 不是在“识别 topk”，而是在做：

> concrete IR structure -> canonical evidence -> generic witness -> frozen semantic truth

这也是为什么同一套 `Phase A` 协议可以同时承接：

- `topk / selection`
- `flash-attn`
- `chunk recurrence`

而协议里又不需要引入这些 workload noun。

### 4.12 为什么这仍然不是名字匹配

上面这个真实 `topk` 例子里用到了 `logits_frag / max_val / max_idx` 这些具体名字，容易让人误会恢复逻辑是靠名字。
仓库里专门有两个 synthetic test 在反证这一点：

- `test_topk_semantic_program_recovers_index_state_from_integer_ir_not_names`
- `test_selection_pairing_recovers_index_role_without_integer_hints`

这两个测试会去掉 `best_slot / max_idx` 之类的名字提示，只保留：

- integer / non-integer fragment 类型
- `if_then_else` companion 结构
- `selection_pairs`
- `update_sources`

然后验证最后仍然能恢复出：

- `index_state`
- `select`
- `paired_value_state`

也就是说，这里在文档里展示具体名字，只是为了让 pass-by-pass walkthrough 可读；真实协议仍然是基于
IR 结构、类型、def-use 和 canonical evidence，而不是变量命名。

## 5. 当前验证摘要

截至 `2026-04-06`，当前稳定验证快照是：

- `pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -q`
  - `38 passed`
- `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py -q`
  - `40 passed, 10 skipped, 1 xfailed`
- `source scripts/setup_tt_sim.sh && pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py -q`
  - `12 passed`
- `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -q`
  - `24 passed, 11 skipped`
- `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py -q`
  - `1 passed`
- `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q`
  - `26 passed`

这些验证说明：

- `Phase A` 当前 compile-path 与 semantic gate 是稳定的
- `copy / GEMM / current flash-attn compile-path` 没有因 semantic layer 回退

但这不等价于 `flash-attn` correctness 已完整闭环并进入稳定基线。

`blackhole.acc` 的最终 correctness payoff 已经是 `Phase B / C` 单一真源切换问题，不再是
`Phase A` 语义恢复本身的 blocker。

## 6. 现在如何使用这份文档

这份文档现在只承担三件事：

1. 说明 `Phase A` 的工程边界
2. 说明当前代码里已经落地了哪些 semantic objects / contracts / validators
3. 为 `Phase B` 提供明确输入约束

它不再承担：

- 逐步实施 checklist
- 逐次测试流水账
- formal proof 草稿

这些内容已经分别分流到：

- git history
- `tasks/progress.md`
- `tasks/dev_design/stage4_phase_a_formalization_note.md`

## 7. 给 Phase B 的交接约束

`Phase B` 应只消费冻结后的 semantic truth：

- `SemanticProgram`
- internal state/effect graph
- lifecycle contract

`Phase B` 不应再直接依赖：

- raw fragment attrs
- ad hoc relation attrs
- 任何名字匹配或 late semantic guessing

如果 `Phase B` 发现缺失的 truth 仍必须回到 semantic recovery 层补充，处理原则仍然是：

1. 先判断该 truth 是否真属于 `Phase A`
2. 若属于 `Phase A`，优先补 witness/core/validator，不补 matcher
3. 若不属于 `Phase A`，直接进入 `Spatial Program IR` 或 `TT Target IR`

## 8. Parallel Theory Track

并行的理论化 / 证明化内容已单独放到：

- [stage4_phase_a_formalization_note.md](/root/dev/vibe_dsl/tasks/dev_design/stage4_phase_a_formalization_note.md)

那里专门承接：

- canonical evidence / abstract domain 的 formalization
- `E / A / alpha / R`
- theorem / obligation checklist
- `Phase A -> Phase B` refinement interface

这条线是并行 research track，不阻塞 `Phase B` 工程实现。
