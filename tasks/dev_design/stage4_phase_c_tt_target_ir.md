# Stage 4 Phase C: TT Target IR And New-Mainline Cutover

## 基本信息

- **文档角色**: `Phase C` 当前设计边界与 cutover 文档
- **当前状态**: `2026-04-08` `Phase C` 进行中；
  `TTProgram` cutover 主链已完成，runtime/codegen 已切到 `TTProgram` direct reader，
  shared generic fallback 已删除，synthetic segment 也已切到最小 `TTProgram`；
  regression 主断言面与 producer-side translator 输入也已切到 typed companion truth，
  但 `Phase C2` runtime payoff 与 wider support surface 仍未完成
- **已完成子阶段**: read-only translator demand probe、`TTHardwareModel` intake、
  `TTProgram` core object set、`LowerSpatialProgramToTTTarget`、
  `ValidateTTTargetProgram`、`MaterializeTTExecutableSpec`
- **仍未完成**:
  - `flash-attn` `Phase C2` runtime / correctness payoff
  - `topk / fusedmoe / paged decode / chunk recurrence` family
  - 更宽 copy/dataflow 支持面
  - 更宽 synchronization 支持面
  - quantitative capability field consumption 与 payload-backed node schema 继续上提
- **上游输入**: 冻结后的 `SpatialProgram`
- **下游输出**: 冻结后的 `TTProgram` 与 `ExecutableSpec` 物化结果
- **唯一总体设计**: `tasks/dev_design/final_blackhole_backend_redesign.md`

## 1. 作用域

`Phase C` 只负责一件事：

- 把冻结后的 virtual spatial program 物化成稳定的 TT target contract

它必须回答：

- 哪些 `Task / Channel / Layout / WorkPartition / SyncEdge / ResourceIntent`
  会落成哪些 TT object
- 哪些 kernel、CB、transport、semaphore、dst layout、ABI、execution plan
  必须显式存在
- 哪些 materialized `blackhole.*` attrs 还能临时保留为 projection，哪些必须删除

它不负责：

- 重新恢复 semantic truth
- 重新发明 task graph、phase truth、domain transform 或 flow semantics
- 让 runtime/codegen 再补 target contract

如果 `Phase C` 发现仅靠当前 `SpatialProgram` 还无法合法决定 target mapping，
结论只能是 `Phase B` contract 还不够。
`Phase C` 不允许自己再补一层 non-TT-specific 语义恢复。

## 2. 必须交付的 TT Contract

### 2.1 Core Objects

`Phase C` 的长期 core object set 只保留：

- `TTProgram`
- `TTKernel`
- `TTCoreGroup`
- `TTCBPlan`
- `TTTransportPlan`
- `TTSemaphorePlan`
- `TTComputeSyncPlan`
- `TTDstLayoutPlan`
- `TTABIPlan`
- `TTExecutionPlan`
- `TTHardwareModel`

### 2.2 TTProgram 必须显式承载的 truth

- `TTKernel`: kernel role、core-group membership、kernel-local binding
- `TTCBPlan`: CB purpose、buffer class、producer-consumer binding
- `TTTransportPlan`: transport family、route / fanout / merge requirement
- `TTSemaphorePlan / TTComputeSyncPlan`: dependency、completion、barrier、
  multicast arrival 等 target sync contract
- `TTDstLayoutPlan`: dst/register legality、carry / reduction / output realization
- `TTABIPlan`: compile-time / common-runtime / per-work 三层 ABI
- `TTExecutionPlan`: launch order、phase cut、host/runtime materialization requirement

### 2.3 基本纪律

- `TTProgram` 是唯一 target truth；`ExecutableSpec` 不是第二真源
- `Phase C` 只消费冻结后的 spatial truth，不消费 display 字段当主链接
- `TTHardwareModel` 必须是 typed object，不允许散落常量继续主导 legality
- `MaterializeTTExecutableSpec` 必须成为唯一稳态 writer
- runtime/codegen 只能读 target truth，不得再补 CB / semaphore / ABI / route contract
- materialized `blackhole.*` attrs 只能是 compatibility projection，不能回升为主协议

### 2.4 小闭集 target family

`Phase C` 继续遵守 small-closed family：

- `TTKernel.kind`: `data_movement / compute / collective / control`
- `TTCBPlan.resource_class`: `transport / scratch / carry / output`
- `TTTransportPlan.kind`: `unicast / multicast / tree / ring / line / fabric_mux`
- `TTSemaphorePlan.kind`: `local / remote / multicast / barrier`

更细 specialization 通过 typed traits、bindings 和 ABI schema 表达，
不靠 target noun 爆炸。

## 3. `Phase C` 的前置输入约束

正式 translator 只能把下面这些东西当成主输入：

- `Task / Channel / Layout / WorkPartition / Placement / SyncEdge / ProgramPhase`
  的 typed contract
- `*_index / *_indices / state_index / domain_index` 这类正式 linkage
- `SpatialCapabilityModel` 已经裁过合法空间后的结果
- `TTHardwareModel` 提供的 concrete target capability

并且要带着 `Phase B` 完成后设计审计的风险清单进入 cutover：

- `Placement` 只有在能被稳定消费成 target mapping constraint 时，
  才证明其长期 object boundary 站得住
- `SpatialCapabilityModel` 的 quantitative hardware fields
  只有在 planning / mapping 真正消费后，才证明其长期存在价值
- `ResourceIntent` 必须继续保持 small-closed kind discipline；
  `Phase C` 不应把更多杂项支持信息重新塞回这一层
- `Phase B` 里仍保留为 typed field + payload 并存的 schema，
  需要在 `Phase C` translator 消费稳定后继续把长期 truth 从 payload 上提，
  避免 node 继续停留在半 stringly-typed 状态
- 各类 spatial node 的 object boundary 仍需通过 `TTProgram` translator 的真实消费
  继续验证；如果某些 truth 只有 payload key 区分而没有 node-level schema 差异，
  应在 `Phase C` 中继续分化 typed schema，而不是让 target 侧 reader 长期手动解包

正式 translator 不能把下面这些字段当主语义：

- `task_name`
- `phase_name`
- `source_task`
- `target_task`
- `channel_names`

这些字段只保留 display / debug / identity 职责。

## 4. Cutover 规则

### 4.1 真源切换顺序

1. `TTProgram / TTABIPlan / TTExecutionPlan` 先成为稳定真源
2. `MaterializeTTExecutableSpec` 成为唯一 writer
3. runtime/codegen 切到只读消费
4. 只有在上面三步稳定后，compatibility writer / reader / fallback 才允许删除

### 4.2 待切对象

下面这些当前还是旧主链 materialization 的字段，后续只能降为 projection，
最终按 deletion gate 删除：

- `blackhole.segment_plan`
- `blackhole.runtime_args`
- `blackhole.common_runtime_args`
- `blackhole.accessors`
- `blackhole.cb_configs`
- `blackhole.semaphore_plan`
- `blackhole.core_plan`

### 4.3 不能做的事

- 让 `LowerBlackholeOps / PlanBlackholeCB / AssignBlackholeCores`
  继续当 target contract writer
- 用 TT kernel 名字或 runtime object 反推 spatial semantics
- 因为当前样例方便，就把 `TTProgram` 退化成大号 attr bag
- 把 `ExecutableSpec` 当作与 `TTProgram` 并列的双真源

## 5. 当前已完成的部分

- `TTHardwareModel` 已作为 module-scope global info 进入主线
- `LowerSpatialProgramToTTTargetProbe` 已落地
- probe 已能消费
  `SpatialProgram + TTHardwareModel + SpatialCapabilityModel`
- probe 不写 `TTProgram`
- probe 不写 `ExecutableSpec`
- probe 不恢复 non-TT-specific spatial semantics
- probe 已把当前最小 demand 面收成显式诊断：
  - `Task.kind + placement.affinity_kind`
  - `Channel.kind + payload_kind + delivery_kind`
  - `Channel` linkage contract：`source_task_index / target_task_index / state_index`
  - `Layout / WorkPartition.domain_index`
  - `ProgramPhase / SyncEdge` 的基本 ordering closure

当前结论：

- `Phase C` 准备轨已完成
- `Phase B` 前置输入已满足
- 正式 `TTProgram` cutover 已完成
- 但 `Phase C` 总体仍未完成；后续支持面扩展仍算本阶段工作

## 6. 当前已完成的部分

- shared generic reader fallback 已删除：
  `tt_program_projection` 现在已经收成
  `TTProgram`-only reader，
  原始 device func 与 synthetic segment 都不再通过 shared getter 回退到
  `blackhole.segment_plan / runtime_args / common_runtime_args / cb_configs /
  semaphore_plan / core_plan / gemm_contract / compute_contract /
  direct_runtime_unsupported_reasons`
- `TTProgram` companion object 现在已经发布 Python/FFI constructor；
  中心 regression 可以直接重建
  `TTProgram / TTKernel / TTCoreGroup / TTABIPlan / TTSemaphorePlan`
  做 typed mutation，不必再通过“改 bridge attrs -> 重跑 translator”间接施压
- 原始 Blackhole device build 输入现在已经硬要求 `tl.tt_program`；
  缺失时会直接 fail-fast，不再靠残留 legacy attrs 继续 build
- transform / target regression 的主断言面已切到
  `TTProgram` / `ExecutableSpec`；
  bridge-stage helper 若需在 `TTProgram` materialize 之前取 target truth，
  现在也优先读取
  `tl.tt_kernel_seeds / tl.tt_abi_plans / tl.tt_program_payload`
- `LowerSpatialProgramToTTTarget` 当前通过 typed upstream truth 构造 `TTProgram`：
  `LowerBlackholeOps` 先发布
  `tl.tt_kernel_seeds / tl.tt_abi_plans / tl.tt_program_payload`，
  `PlanBlackholeCB / AssignBlackholeCores`
  分别发布 `tl.tt_cb_plans / tl.tt_core_groups`，
  final translator 不再依赖 legacy bridge attrs
- `LowerBlackholeOps` 在 typed seed materialization 后会剥离
  `blackhole.segment_plan / runtime_args / common_runtime_args /
  gemm_contract / compute_contract / direct_runtime_unsupported_reasons`
  等 legacy projection attrs；
  final output 的 `LowerSpatialProgramToTTTarget` 也会继续做 residual strip
- `flash-attn` 的 `Phase C2` runtime gate 已从 hang 收敛成显式 unsupported：
  multi-GEMM compute kernel 会带
  `direct_runtime_unsupported_reasons = ["multi_gemm_compute_kernel"]`
  并在 direct runtime test 中 skip；
  但这只算 gate 收敛，不算 `Phase C2` payoff 完成
- `SpatialCapabilityModel` 的 quantitative hardware fields
  还没有进入正式 planning / mapping 主链
- `Placement / ResourceIntent / Layout / WorkPartition` 的长期 object boundary
  还没有经过 `TTProgram` materialization 的最终验证
- 部分 spatial node 仍保留 payload-backed truth；
  更彻底的 typed schema 分化仍属于 `Phase C` 演进内容
- `ValidateTTTargetProgram` 已补进 kernel payload、execution plan、
  ABI / contract payload 的结构校验，
  但仍未把 runtime/codegen 真正消费的 ABI / accessor / launch / core-group
  全量细节全部收进最终 validator gate

因此当前 `Phase C` 的 `TTProgram` cutover 本体已完成，
但阶段总目标仍未完成；剩余工作继续留在 `Phase C` 范围内推进。

## 6.2 当前 deletion gate 的真实拓扑

当前稳定拓扑是：

```text
SpatialProgram
  -> SplitBlackholeKernel / internal segment planning attrs
  -> LowerBlackholeOps
  -> tl.tt_kernel_seeds / tl.tt_abi_plans / tl.tt_program_payload
  -> PlanBlackholeCB
  -> tl.tt_cb_plans
  -> AssignBlackholeCores
  -> tl.tt_core_groups
  -> LowerSpatialProgramToTTTarget
  -> TTProgram
  -> tt_program_projection TTProgram-only readers
  -> rt_mod_blackhole / codegen_blackhole
  -> synthetic segment TTProgram materialization
  -> internal codegen / ExecutableSpec assembly
```

当前实现的含义是：

- `rt_mod_blackhole` / `codegen_blackhole`
  已经通过共享 decoder 读取 `TTProgram` truth，
  shared decoder 不再保留 generic legacy fallback
- `TTProgram` 已有可承载这些信息的 typed object：
  `TTABIPlan / TTCBPlan / TTCoreGroup / TTSemaphorePlan / TTExecutionPlan`
  以及 program payload 承载的
  `gemm_contract / compute_contract / direct_runtime_unsupported_reasons`
- regression 主断言面与 mutation ingress
  现在已覆盖 `TTProgram` / `ExecutableSpec` 与 bridge-stage typed seed attrs
- producer-side translator 输入也已不再依赖 legacy bridge attrs；
  legacy projection 只剩 `SplitBlackholeKernel -> LowerBlackholeOps`
  之间的内部 planning 过渡职责

因此当前 deletion gate 已经收口，但这不等于 `Phase C` 整体收口。

## 6.3 Reader-Side Deletion Gate 的执行顺序

当前收口顺序固定为：

1. 先在 `rt_mod_blackhole` / `codegen_blackhole`
   引入共享的 `TTProgram` direct reader / decoder，
   让 `ExecutableSpec` 组装与 kernel codegen 优先消费 typed target truth
   这一步对当前 reader surface 已完成，包括
   `segment/runtime/common-runtime/cb/semaphore/core`
   与 function-level `gemm_contract / compute_contract /
   direct_runtime_unsupported_reasons`
2. 把 shared generic fallback 拆成 `TTProgram`-only reader，
   并让 synthetic segment 也挂最小 `TTProgram`
   这一步对当前 reader / codegen surface 已完成
3. 把 transform / target regression 的主断言面从
   `blackhole.*` attrs 迁到 `tl.tt_program` / `ExecutableSpec`
4. 在 copy / GEMM / `flash-attn` baseline 稳定后，
   删除 `LowerBlackholeOps` bridge-stage 输出中的剩余 projection attrs
5. 把 `BuildCBPlans / BuildCoreGroups / BuildSemaphorePlans / BuildABIPlans`
   全部切到 typed upstream truth

## 6.1 本次正式 cutover 的实现策略

本次 `Phase C` 落地按下面的 bridge discipline 执行：

1. 先把 `TTProgram` core object set 落成 companion truth，并注册成
   `tl.tt_program`
2. `LowerSpatialProgramToTTTarget` 负责生成 `TTProgram`
3. `ValidateTTTargetProgram` 负责做 target-truth 完整性与 object-boundary 校验
4. `rt_mod_blackhole` / `codegen_blackhole` 通过共享 direct reader 直接消费
   `TTProgram`；`MaterializeTTExecutableSpec` 仅保留为 pipeline continuity 用的
   no-op bridge pass

桥接期纪律：

- 正式 runtime/codegen 的稳态真源只能是 `TTProgram`
- `LowerBlackholeOps / PlanBlackholeCB / AssignBlackholeCores` 在 bridge 期仍可保留为
  target planning helper，但它们写出的 legacy attrs 只能当 translator 输入，
  不能继续当最终协议真源
- synthetic segment 若需要内部重建 target-truth，
  必须直接挂最小 `TTProgram`，不能再回退到局部
  `blackhole.*` projection attrs
- 任何 function-level contract（如
  `gemm_contract / compute_contract / direct_runtime_unsupported_reasons`）
  一旦进入 reader-side正式消费面，就必须提升进 `TTProgram.payload`

当前实现结果：

- `TTProgram` 已成为 target truth owner
- `rt_mod_blackhole` / `codegen_blackhole`
  已通过共享 decoder 直接消费 `TTProgram`
- `rt_mod_blackhole` 的原始 entry 提取已不再对 legacy attrs 做兼容回退；
  `codegen_blackhole` 对原始 device func 与 synthetic segment
  也都不再经由 shared getter 回退到 legacy attrs
- `MaterializeTTExecutableSpec` 已退化成 no-op compatibility pass
- `LowerBlackholeOps / PlanBlackholeCB / AssignBlackholeCores`
  已降为 bridge planning helper
- copy / GEMM / `flash-attn`
  的核心 target-truth regression 已切到 `TTProgram` / `ExecutableSpec`
- `LowerBlackholeOps` bridge-stage 输出现在以
  `tl.tt_kernel_seeds / tl.tt_abi_plans / tl.tt_program_payload`
  为主真相，并在 seed 发布后剥离 legacy projection attrs
- `PlanBlackholeCB / AssignBlackholeCores`
  现在分别只发布 `tl.tt_cb_plans / tl.tt_core_groups`
- `LowerSpatialProgramToTTTarget` 现在以 typed upstream truth materialize `TTProgram`

当前仍未完成并且属于 `Phase C` 的工作包括：

- `flash-attn` multi-GEMM direct runtime 的真正 enablement 与 correctness payoff
- `topk / fusedmoe / paged decode / chunk recurrence` 等 family 在新主链上的统一承接
- 更宽 copy/dataflow 支持面
- 更宽 synchronization 支持面
- quantitative capability field 与 object-boundary 的继续验证
- 部分 payload-backed node schema 的继续 typed uplift

## 7. 完成判定

只有在下面这些条件全部成立后，`Phase C` 才能算完成：

1. `TTProgram` core object set 已进入稳定主链
2. `TTProgram` 已显式承载第 2.2 节列出的 target truth
3. `MaterializeTTExecutableSpec` 已成为唯一稳态 writer
4. copy / GEMM / `flash-attn` 的 target truth 都从 `TTProgram` 物化，
   不再依赖旧 planning 主链补洞
5. runtime/codegen 与核心回归的主断言面已切到
   `TTProgram` / `ExecutableSpec`，不再以 `blackhole.*` projection 为主真相
6. compatibility writer / reader / fallback 已按 cutover 规则删除
7. 新 family 进入主链时走
   `Stateful Semantic IR -> Spatial Program IR -> TT Target IR`
   的统一路径，而不是新增 case-by-case matcher
8. `flash-attn` 的 `Phase C2` runtime / correctness payoff 已完成，
   不再停留在 explicit unsupported gate
9. `topk / fusedmoe / paged decode / chunk recurrence`
   等 family 已按新主链统一承接
10. 更宽 copy/dataflow 支持面已按新 target-truth 主链落地
11. 更宽 synchronization 支持面已按新 target-truth 主链落地
12. shared zero-regression baseline 与 `Phase C2` 的 runtime gate 持续通过

当前结论：

- `TTProgram` cutover 子线已完成
- `Phase C` 总体仍未完成
- 当前 blocker 已转到 `Phase C2` runtime/correctness payoff 与 wider support surface

## 8. Shared Zero-Regression Baseline

```bash
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q
```

## 9. `Phase C2` Runtime Gate

```bash
source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh
export TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo
cd /root/dev/vibe_dsl/tilelang_repo
pytest testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py -k 'mha or gqa' -q
```

`Phase C2` 当前已达到的 gate 是：

- copy direct runtime 正常通过
- `flash-attn` multi-GEMM compute kernel 显式暴露
  `direct_runtime_unsupported_reasons = ["multi_gemm_compute_kernel"]`
  并在 runtime test 中 skip

这只代表“不会再 hang，unsupported 行为可诊断”。
真正的 multi-GEMM direct runtime enablement 与 correctness payoff
仍然阻塞 `Phase C` 完成。
