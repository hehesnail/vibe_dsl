# Review: TileLang Blackhole 后端重设计总文档

- **Review 日期**: 2026-04-05
- **被 Review 文档**: `tasks/dev_design/final_blackhole_backend_redesign.md`
- **Review 范围**: 设计合理性、可落地性、通用性、与代码/TT-Metal 实际对齐度
- **交叉审计材料**:
  - TileLang 编译管线源码（`tilelang/engine/lower.py`, `tilelang/engine/phase.py`）
  - 当前 Blackhole pass 实现（`src/transform/` 下 `split_blackhole_kernel`、`analyze_blackhole_*`、`lower_blackhole_ops`、`plan_blackhole_cb`、`assign_blackhole_cores`）
  - TT-Metal API（`tt_metal/api/tt-metalium/`、SDPA program factory、compute kernel API）
  - TileLang DSL 表面与前端 workload 示例

> 注：这份文档是 `2026-04-05` 的设计审计快照，用来解释总设计为什么这样收敛。
> `R1-R11` 已全部响应；当前工程状态、阶段主线与 blocker 以
> `tasks/progress.md` 和各阶段文档为准，而不是以本文中的未来式建议段落为准。

---

## 一、总体评价

**设计方向正确，是当前架构的合理演进。**

三层 IR 分离的核心论点——"编译器不能在一个层里同时解决 semantic recovery、spatial organization 和 TT target planning"——被代码事实充分支撑：

- `LowerBlackholeOps`（176KB C++）确实同时在做语义恢复（从 buffer scope 推断 copy 方向、从 op args 提取 GEMM 维度）和 target lowering（发射 CB reserve/push/pop、NOC 命令序列）。这正是设计中说的"三类责任混在一起"。
- `blackhole.acc` 混合语义问题确实是结构性的：`LowerBlackholeOps` 无法区分"算法 carry state"和"target scratch"，因为在它执行时，这些信息已经在通用 lowering 中被压碎了。
- DSL 层确实有信息丢失：`T.Pipelined(N, num_stages=K)` 变成 buffer rotation 后，原始 stage 结构需要 `AnalyzeBlackholePipelineStages` 重新恢复；`T.GemmWarpPolicy.FullRow` 被 lower 后，policy 语义需要从分布式代码反推。

三层分层与 TT-Metal 实际编程模型高度吻合：

- **Semantic 层**对应"程序在算什么"——TT-Metal 自己的 SDPA program factory 也需要知道 attention 的 carry/reduce/normalize 语义才能正确配置 dst 和 CB。
- **Spatial 层**对应"task/channel/layout/sync"——TT-Metal 的 `CreateKernel` + `CreateCircularBuffer` + `SetRuntimeArgs` 模型本质上就是一个 spatial program 的物化。
- **TT Target 层**对应"TT-Metal contract"——CB ID 分配、semaphore 协议、dst layout、ABI 组织都是纯 target 决策。

---

## 二、设计优点

### 2.1 通用性设计没有陷入 case-by-case

1. **`Domain / State / Update` 三元组**是 workload-neutral 抽象。flash-attn 的 carry/reduce、topk 的 selection、MoE 的 routed dispatch、paged decode 的 indirect access、chunk recurrence 的 ordered update，都能用这三个对象 + `AccessMap` + `UpdateLaw` 的组合表达。设计没有为任何 workload 引入特殊 enum 或 spec。

2. **小闭枚举 + fixed trait axes** 是防止膨胀的关键约束。`Task.kind` 只有 4 个（transfer/compute/collective/control），`Channel` 靠 `payload_kind × transport_semantics` 组合，`Layout.kind` 只有 3 个。这比 `role_set + Map<String, Bool>` 的旧模式严格得多。每个 trait 都属于预定义的固定轴，同一 object 在同一 axis 上至多取一个值。

3. **workload validation matrix**（Section 4.4）把 6 类 workload family 作为设计边界声明而非单一 consumer，Section 7 用具体 `Domain/State/Update` 实例展开。Phase A1 的验收 gate 强制要求至少一个非 attention 的 semantic skeleton case。

4. **`SemanticSupplement` 的限制机制**设计得很克制：只允许 4 类裁决事实（state identity、access trait、update law trait、semantic boundary），不允许表达 workload family 名字或 spatial/target 结构。

5. **不再维护 `*Spec` 家族。** `CombineSpec / SelectionSpec / SegmentSpec / PageSpec / RecurrenceSpec` 被统一收进 `AccessMap.traits` + `UpdateLaw` 变体。这是从 noun bag 到 typed orthogonal schema 的关键转变。

### 2.2 与现有代码的衔接清晰

- 设计明确了 semantic lift 的 canonicalization 点：在 `AnnotateBlackholeCopySemantics` / `BlackholeDeviceResourceCanonicalization` / `LowerOpaqueBlock` / `SplitHostDevice` 之后，但在 `SplitBlackholeKernel` 的 TT-specific planning 之前。这与当前 `OptimizeForTarget` → `blackhole_codegen` 的实际分界对齐。
- 现有 analysis pass（`AnalyzeBlackholeWorkDecomposition` / `FragmentRegions` / `PipelineStages`）被保留为 semantic recovery 的输入生产者，而不是推倒重写。
- compatibility shim 有显式的 deletion gate（5 条条件），不是模糊的"以后再删"。

### 2.3 TT-Metal 对齐度高

| 设计对象 | TT-Metal 实际对应 | 对齐? |
|----------|------------------|-------|
| `TTKernel.kind`（data_movement/compute/collective/control）| `DataMovementConfig` / `ComputeConfig` | ✅ |
| `TTABIPlan` 三层 ABI（compile-time / common-runtime / per-work）| `compile_args` / `SetCommonRuntimeArgs` / `SetRuntimeArgs` | ✅ |
| `TTComputeSyncPlan`（FPU/SFPU 协调）| `t6_semaphore_init/wait_on_zero/get` | ✅ |
| `TTCBPlan`（data_format/pack_mode/unpack_mode/l1_acc_mode）| `CircularBufferConfig` + `pack_reconfig_l1_acc` | ✅ |
| `TTDstLayoutPlan`（dst residency/accumulator_mode）| `fp32_dest_acc_en` + `tile_regs_acquire/release` | ✅ |
| `TTTransportPlan`（unicast/multicast/ring/line/fabric）| TT-Metal NoC API + Remote CB | ✅ |

---

## 三、设计 vs 代码事实对齐检查

| 设计声明 | 代码事实 | 对齐? |
|----------|---------|-------|
| `LowerBlackholeOps` 混合 semantic recovery + target lowering | 确认：176KB C++ 文件，同时做 copy/GEMM 语义提取和 CB/NOC builtin emission | ✅ |
| `SplitBlackholeKernel` 产出 `blackhole.segment_plan` | 确认：写入 3-kernel segment schema（reader/compute/writer）| ✅ |
| `PlanBlackholeCB` 分配 CB ID 0-15/16-31/32-63 | 确认：input/output/intermediate 三区 CB 分配 | ✅ |
| TT-Metal 有 compile-time / common-runtime / per-work 三层 ABI | 确认：`compile_args` / `SetCommonRuntimeArgs` / `SetRuntimeArgs`（host_api.h、dataflow_api.h）| ✅ |
| TT-Metal compute kernel 内 FPU/SFPU 通过 `t6_semaphore` 协调 | 确认：`t6_semaphore_init/wait_on_zero/get`，FPU(MATH) 和 SFPU(PACK) 共享 dst | ✅ |
| DSL 语义在 lowering 后被压碎需要恢复 | 部分确认：copy/GEMM pattern 固定恢复较易；复杂 workload 恢复难度显著更高 | ⚠️ |
| `ExecutableSpec` 当前被多方同时写入 | 确认：`LowerBlackholeOps`、`codegen_blackhole`、`rt_mod_blackhole` 都参与 | ✅ |
| Blackhole pipeline 在 `OptimizeForTarget` 中早分叉 | 确认：`phase.py` line 236 分叉，跳过所有 CUDA-only passes | ✅ |
| Semantic lift 点应在 generic canonicalization 之后、TT-specific lowering 之前 | 确认：对应 `SplitHostDevice` 之后、`SplitBlackholeKernel` 之前 | ✅ |

---

## 四、风险与问题

### 4.1 P0：Semantic Recovery 的工程难度可能被低估

**这是最大的落地风险。**

设计要求从"被 scalarize 的 `BufferLoad/BufferStore`"中自动恢复 `Domain / State / Update / AccessMap / UpdateLaw`。但当前 TileLang 的 lowering 链在到达 Blackhole codegen 入口时，语义已经被显著压碎：

- `LowerTileOp` 把 `T.gemm`/`T.copy`/`T.reduce_*` 展开成具体的 buffer 操作
- `InjectSoftwarePipeline` 引入 buffer rotation 和 double buffering
- `LowerOpaqueBlock` 消除 opaque block 边界
- `SplitHostDevice` 改变 buffer/var identity

对于 **copy 和 GEMM**，recovery 相对简单（pattern 固定，当前 `LowerBlackholeOps` 已经在做）。但对于 flash-attn 的 normalized recurrence、topk 的 iterative selection、MoE 的 ragged dispatch，要从被压碎的 TIR 中稳定恢复出 `RecurrenceLaw(normalized, stable_order_required)` 或 `SelectLaw(topk, selector_axis=k)`，工程难度远高于 copy/GEMM pattern matching。

**重要背景**：当前 TileLang 编译管线中，Blackhole 路径在 `OptimizeForTarget` 中**已经走完全不同的路径**（`phase.py:236`），跳过了 warp specialization、TMA lowering、vectorization 等全部 CUDA-only pass。`AnnotateBlackholeCopySemantics` 和 `BlackholeDeviceResourceCanonicalization` 已经是 Phase 2 中的 Blackhole-specific 早期 pass。Blackhole 路径早已不是"通用管线 + 最后几步 target codegen"了。

**建议**：

1. 认真评估是否可以在 `LowerTileOp` 期间或之前就捕获关键语义信息，而不是完全依赖 post-lowering recovery。DSL 层的 `T.gemm` / `T.reduce_max` / `T.Pipelined` 本身携带了丰富语义，在它们被 lower 成 buffer 操作之前提取会容易得多。
2. 建议不要把 recovery 和 preservation 对立起来。第一版可以混用：能在 lowering 过程中廉价保留的信息先作为 lightweight annotation 附着到 TIR（例如 `tl.op_semantic` / `tl.reduction_kind`），recovery 只负责重建结构关系和补齐缺失的高层语义。
3. 这些 annotation 不是 `SemanticProgram`（真源仍然只有 `SemanticProgram`），也不是 `SemanticSupplement`（那是用户侧裁决通道），而是编译器自己在早期 lowering 阶段的内部信号，供后续 `AnalyzeSemanticStructure` 消费。

### 4.2 P1：对象数量与实现复杂度

统计设计中需要新增的 typed companion object：

| 层 | 新增 typed object | 数量 |
|----|------------------|------|
| Semantic core | `SemanticProgram`, `Domain`, `State`, `Update`, `AccessMap`, `UpdateLaw`(4 variants) | ~8 |
| Semantic helper | `TIRAnchor`, `TIRValueBinding`, `AtomicEffect`, `SemanticRegion`, `SemanticSupplement` | ~5 |
| Semantic internal | `StateVersion`, `StateDef`, `StateUse`, `StateJoin` | ~4 |
| Spatial core | `SpatialProgram`, `ProgramPhase`, `Task`, `Channel`, `Layout`, `WorkPartition`, `Placement`, `SyncEdge`, `ResourceIntent` | ~9 |
| Spatial planning | `SpatialLegalityFacts`(6 entry types), `SpatialCandidate`, `SpatialPolicy`, `SpatialCostModel` | ~10 |
| TT Target core | `TTProgram`, `TTKernel`, `TTCoreGroup`, `TTCBPlan`, `TTTransportPlan`, `TTSemaphorePlan`, `TTComputeSyncPlan`, `TTDstLayoutPlan`, `TTABIPlan`, `TTExecutionPlan` | ~10 |
| TT Hardware model | `TTTopologyModel`, `TTMemoryModel`, `TTNoCModel`, `TTSyncModel`, `TTDstModel`, `TTABILimitModel`, `TTComputeModel` | ~7 |

**总计约 50+ 个新 typed object**，每个需要 C++ `ObjectNode/ObjectRef` 定义、reflection、printer、structural equal/hash、visitor/mutator、Python FFI。

**建议**：

1. 实现时必须严格分阶段，Phase A 只实现 Semantic 层 object，不要提前铺 Spatial/TT object 的骨架。
2. 部分 internal analysis object（`StateVersion` / `StateDef` / `StateUse` / `StateJoin`、`AtomicEffect`、`SemanticRegion`）可以先用轻量级 C++ struct 而非完整的 TVM Object，降低 boilerplate。
3. `SpatialCostModel` / `SpatialCandidate` 在 Phase B 初期可以先不实现（设计已留口子："初期走 deterministic heuristic 路线"）。
4. Phase A1 的最小必要对象集建议为：`SemanticProgram`, `Domain`, `State`, `Update`, `AccessMap`, `UpdateLaw`（仅 MapLaw + ReduceLaw）, `TIRAnchor`, `TIRValueBinding`。其余 defer。

### 4.3 P1：`TIRValueBinding` / rebind contract 的实操性

设计要求 audited-safe pass 产出完整的 typed rebind 结果（Section 5.6.3.2），禁止用 buffer 名字或位置序号匹配。这在原则上正确，但在实践中：

- TVM 的很多标准 pass（`Simplify`、`ConstantFolding`、`DCE`）都会改变 `PrimExpr` identity
- 当前 `BlackholeDeviceResourceCanonicalization` **已经需要 name-based fallback** 才能维持逻辑绑定（设计 Section 3.1 P0.2 自己承认了这一点）
- semantic lift 之后到 `LowerBlackholeOps` 之间几乎没有 pass 能是 safe 的

这实际上等于说：companion IR 的有效生命周期是"从 lift 到 materialization 之间的一段不做 TIR 变换的区间"。

**建议**：

1. 简化第一版 rebind contract。直接采用"lift 之后不允许 TIR mutation，直到 materialization 消费完 companion IR"的硬规则。
2. `TIRValueBinding` 的 rebind 机制 defer 到真正需要跨 TIR pass 维护 companion IR 时再实现。
3. 第一版 binding 基于 `Buffer` object identity（Buffer 在 TVM 中是 ref-counted object，identity 比 PrimExpr 稳定），对 `PrimExpr` 级 binding 允许 structural equality match 作为 fallback 但记录 warning。

### 4.4 P2：Spatial 层对当前基线可能过重

当前稳定基线只有 copy（单 kernel）和 GEMM（三 kernel reader/compute/writer）。对这些 workload，Spatial IR 的 9 个 core object + 10 个 planning object 几乎全部退化为 trivial case。

**建议**：

1. 在 `LowerToSpatialProgram` 中实现 fast-path：当 `SemanticProgram` 只包含单个 `MapLaw` 或单个 `ReduceLaw` 且无 carry/indirect/routed access 时，直接构造 canonical `SpatialProgram` 而不走 candidate search。这不是 case-specific hack——它是对"退化到简单 workload 时不引入不必要复杂度"的合法优化。
2. Phase B 的验收 gate 必须包含至少一个 non-trivial multi-task workload（flash-attn 或 MoE）的 compile-to-spatial 验证。

### 4.5 P2：`ProgramPhase` 的宿主规则可能过于复杂

设计定义了两种 `ProgramPhase` 宿主模式（单 `PrimFunc` 挂 attrs vs 多 device-function 提升到 `IRModule.global_infos`）。但当前编译流程中，`SplitHostDevice` 之后每个 device function 都是独立 `PrimFunc`。多 `T.Kernel` 程序（fusedmoe、split-decode）在到达 Blackhole codegen 入口时，每个 kernel block 已经是独立函数。Case 2（module-scope device program）是这些 workload 的常态，不是边缘情况。

**建议**：统一用 `IRModule.global_infos["tl.device_programs"]` 承载所有 program 的 phase truth，单 `PrimFunc` 程序只是 `member_funcs` 长度为 1 的退化情况。避免运行时判断"当前是 case 1 还是 case 2"的额外复杂度。

### 4.6 P2：`ProgramPhase` 与 `SplitHostDevice` 的时序问题

如果 semantic lift 必须在 `SplitHostDevice` 之后执行（设计 Section 5.6.3.1 要求如此），而 `SplitHostDevice` 已经把多个 `T.Kernel` block 拆成了独立 device function，那么 semantic recovery 如何重建跨 function 的 `ProgramPhase` 关系？

**建议**：要么在 `SplitHostDevice` 之前就建立 module-scope 的 multi-kernel program registry（作为 Phase 2 早期 Blackhole-specific pass），要么让 semantic lift 在 `SplitHostDevice` 之前执行。后者可能更干净，但需要重新评估 canonicalization 点要求。

### 4.7 观察：`UpdateLaw` 的 4 种 variant 可能不够

当前 `UpdateLaw` 定义了 `MapLaw / ReduceLaw / SelectLaw / RecurrenceLaw`。但 MoE 的 `combine_output_update`（weighted sum of expert outputs with routing weights）不完全是 reduce 也不完全是 map——它是一个 weighted accumulate with routing predicate。

**建议**：

- 要么把 `ReduceLaw` 的语义扩展为可以带 `weight_expr` 和 `routing_predicate`
- 要么确认 MoE combine 可以被分解为 `MapLaw(weighted_scale) + ReduceLaw(accumulate)` 的组合
- 在 Phase A2 实际做 MoE recovery 时验证这一点

### 4.8 观察：`TTHardwareModel` 的数据来源

7 个子模型需要来自 TT-Metal 真实硬件规格的数据。当前代码中没有 typed hardware model——硬件约束散落在各个 pass 的 hardcoded constant 中（例如 `PlanBlackholeCB` 中的 L1 1.5MB 限制、64 CB 上限）。

**建议**：

- Phase C 之前，先建立一个 minimal `TTHardwareModel` stub，把当前散落的 hardcoded constant 收集到统一 config object 中。这可以作为 Phase A/B 的基础设施准备。
- Phase C 第一轮优先实现 `TTMemoryModel`（CB/L1 约束）、`TTDstModel`（dst capacity/accumulator）、`TTComputeModel`（FPU/SFPU capability，直接解决 `blackhole.acc` 问题）。
- `TTTopologyModel` / `TTNoCModel` / `TTSyncModel` 推到多核/multicast 支持扩展时再实现。

### 4.9 观察：Section 7 示例可能变成 hidden spec

Section 7 为 attention 列出了 `qk_scores_update`、`softmax_normalize_update`、`online_attention_update` 等具体 Update 名字。如果实现者把这些名字硬编码成 pattern matcher，就又回到了 case-by-case。

**建议**：Section 7 的示例应该明确标注"这些对象名字只是叙述用例，不是 schema 约束"。recovery 不应该通过匹配 `online_attention_update` 这个名字来识别 `RecurrenceLaw`。

---

## 五、具体改进建议汇总

| 编号 | 优先级 | 建议 |
|------|--------|------|
| R1 | P0 | 补充"早期语义捕获"策略：允许在 `LowerTileOp` 期间或之前把关键语义事实作为 lightweight annotation 附着到 TIR，供后续 `AnalyzeSemanticStructure` 消费 |
| R2 | P0 | 明确 Phase A1 的最小 typed object 集（~8 个），其余 defer |
| R3 | P0 | 简化第一版 rebind contract：semantic lift 之后不允许 TIR mutation 直到 materialization，避免 Phase A1 就要求 safe/unsafe pass 声明 |
| R4 | P1 | Phase A1 的验收 gate 明确粒度：non-attention case 只要求 `Domain + State + UpdateLaw.kind` 正确恢复，完整 trait 验证推到 A2 |
| R5 | P1 | `ProgramPhase` 统一用 `IRModule.global_infos["tl.device_programs"]` 承载，单 PrimFunc 是退化情况 |
| R6 | P1 | 解决 `SplitHostDevice` 与 multi-kernel program 的时序问题：建议在 SplitHostDevice 之前建立 module-scope program registry |
| R7 | P1 | Spatial 层实现 fast-path：simple workload 直接构造 canonical SpatialProgram |
| R8 | P2 | Phase B 验收 gate 包含至少一个 Task:TTKernel 不是 1:1 的 test case |
| R9 | P2 | Phase C 之前先建立 minimal `TTHardwareModel` stub，收集当前 hardcoded constant |
| R10 | P2 | 在 Phase A2 做 MoE recovery 时验证 `UpdateLaw` 4 种 variant 是否够用 |
| R11 | P2 | Section 7 示例明确标注"名字只是叙述用例，不是 schema 约束" |

> **后记**（2026-04-05）：设计文档已基于 R1-R10 完成修订。R11 已在后续补丁中落入 Section 7 开头。全部 11 条建议均已响应。

---

## 六、评分总结

| 维度 | 评分 | 说明 |
|------|------|------|
| 问题诊断 | ★★★★★ | 准确识别了 monolithic lowering 的结构性问题，`blackhole.acc` 分析深入 |
| 架构方向 | ★★★★★ | 三层分离与 TT-Metal 编程模型高度吻合 |
| 通用性 | ★★★★☆ | `Domain/State/Update` + 小闭枚举 + trait axes 设计通用；`UpdateLaw` 可能需要扩展 |
| 与现有代码衔接 | ★★★★☆ | canonicalization 点、pass 保留策略、compatibility shim 都清晰；rebind contract 的实操性需简化 |
| 可落地性 | ★★★☆☆ | 50+ 个新 typed object 工程量大；semantic recovery 难度可能被低估；建议更积极利用 DSL 层早期语义信息 |
| 文档完整性 | ★★★★★ | 覆盖 workload matrix、validation matrix、migration plan、deletion gates、所有层 object schema 和 contract |
| 防止 case-by-case | ★★★★★ | 多处明确禁止 workload-specific enum/matcher/noun bag；trait axes 固定；Phase A1 gate 强制 non-attention case |

---

## 七、核心结论

设计方向正确、通用性设计扎实、TT-Metal 对齐度高。主要风险在 semantic recovery 的工程难度和 companion object 的实现量。

**核心建议**：在 Phase A1 实现时，优先探索"DSL 层早期语义捕获 + post-lowering recovery"的混合策略，而不是完全依赖后段恢复。这不违反设计原则——`SemanticProgram` 仍然是唯一真源，只是它的构造输入多了一个编译器内部的 annotation 通道。
