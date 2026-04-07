# Stage 4 Phase C: TT Target IR And New-Mainline Cutover

## 基本信息

- **文档角色**: `Phase C` 当前设计边界与 cutover 文档
- **当前状态**: `2026-04-07` 只有准备轨已落地；正式 cutover 尚未开始
- **已完成子阶段**: read-only translator demand probe、`TTHardwareModel` intake
- **仍未完成**: `TTProgram`、`MaterializeTTExecutableSpec`、旧主链删除
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
- 这不等于正式 `TTProgram` cutover 已开始或已完成

## 6. 当前仍未完成的部分

- `TTProgram` 还不存在
- 正式 `LowerSpatialProgramToTTTarget` translator 还不存在
- `ValidateTTTargetProgram` 还不存在
- `MaterializeTTExecutableSpec` 还不存在
- copy / GEMM 还没有从 TT truth 重新物化成 `ExecutableSpec`
- runtime/codegen 还没有切到只读消费 `TTProgram`
- compatibility writer / fallback reader 还没有按 deletion gate 删除
- `flash-attn` 的 `blackhole.acc` correctness payoff 仍归属 `Phase C2`

因此当前 `Phase C` 还不能判定为开始正式 cutover。

## 7. 完成判定

只有在下面这些条件全部成立后，`Phase C` 才能算完成：

1. `TTProgram` core object set 已进入稳定主链
2. `TTProgram` 已显式承载第 2.2 节列出的 target truth
3. `MaterializeTTExecutableSpec` 已成为唯一稳态 writer
4. copy / GEMM / `flash-attn` 的 target truth 都从 `TTProgram` 物化，
   不再依赖旧 planning 主链补洞
5. compatibility writer / reader / fallback 已按 cutover 规则删除
6. 新 family 进入主链时走
   `Stateful Semantic IR -> Spatial Program IR -> TT Target IR`
   的统一路径，而不是新增 case-by-case matcher
7. shared zero-regression baseline 与 `Phase C2` 的 runtime gate 持续通过

当前结论：

- `Phase C0` 已完成
- `Phase C1 / C2 / C3` 未完成
- 当前 blocker 仍先落在剩余 `Phase B`

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
