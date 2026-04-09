# Stage 4 Phase C: TT Target IR And New-Mainline Cutover

## 基本信息

- **文档角色**: `Phase C` 当前设计边界、剩余项与完成判定文档
- **当前状态**: `2026-04-09` `Phase C` 进行中；
  `TTProgram` cutover 主链已完成，runtime/codegen 已切到 `TTProgram` direct reader，
  shared generic fallback 已删除，synthetic segment 也已切到最小 `TTProgram`；
  regression 主断言面与 producer-side translator 输入也已切到 typed companion truth，
  `flash-attn` small bf16 MHA direct runtime correctness 已兑现，
  GEMM oversubscribed `work_packets` host scheduling 已兑现，
  但 `Phase C2` wider runtime payoff 与 wider support surface 仍未完成
- **已完成子阶段**: read-only translator demand probe、`TTHardwareModel` intake、
  当前已落地的 `TTProgram` core object set、`LowerSpatialProgramToTTTarget`、
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
- **相关 cross-layer feature 文档**: `tasks/dev_design/spatial_dataflow_program_model.md`

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
- `TTBlockPlan`
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
- `TTBlockPlan`: target-side block sizing、reduction slicing、subblock、
  buffering 与 block-to-core decomposition truth
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

## 5. `TTProgram` Cutover 已完成的部分

`Phase C` 当前已经完成的是 target-truth cutover 本体，而不是整个阶段。
这部分的完成边界如下：

- `TTHardwareModel` 已作为 module-scope global info 进入主线
- `LowerSpatialProgramToTTTargetProbe` 已落地，并且只消费
  `SpatialProgram + TTHardwareModel + SpatialCapabilityModel`
- 除本轮新增的 `TTBlockPlan` 扩展外，`TTProgram` core object set 已落地，
  `LowerSpatialProgramToTTTarget / ValidateTTTargetProgram /
  MaterializeTTExecutableSpec` 已进入正式主链
- runtime/codegen 已切到 `TTProgram` direct reader；
  shared generic fallback 已删除
- synthetic segment 已改成挂最小单-kernel `TTProgram`
- 原始 Blackhole device build 输入已硬要求 `tl.tt_program`
- transform / target regression 的主断言面已切到
  `TTProgram` / `ExecutableSpec`
- producer-side translator 输入也已切到 typed companion truth：
  `LowerBlackholeOps` 发布
  `tl.tt_kernel_seeds / tl.tt_abi_plans / tl.tt_program_payload`，
  `PlanBlackholeCB / AssignBlackholeCores`
  发布 `tl.tt_cb_plans / tl.tt_core_groups`
- `TTProgram` companion object 已有 Python/FFI constructor；
  regression 可直接重建
  `TTProgram / TTKernel / TTCoreGroup / TTABIPlan / TTSemaphorePlan`

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

这意味着：

- reader-side deletion gate 已收口
- final translator 已不再依赖 legacy bridge attrs
- legacy projection 只剩
  `SplitBlackholeKernel -> LowerBlackholeOps`
  之间的内部 planning 过渡职责
- 后续 `TTBlockPlan`、literal taxonomy、validated hint intake、
  SRAM/L1/CB-aware planning 的跨层 owner 链统一以上面的
  cross-layer feature 文档为准；本阶段文档只保留 `Phase C` 的
  target-side 落地、完成判定与 runtime gate

## 6. 当前仍属于 `Phase C` 的工作

下面这些不是“后续可选优化”，而是当前 `Phase C` 仍然必须完成的工作。

### 6.1 `Phase C2`: `flash-attn` runtime / correctness payoff

当前状态：

- `flash-attn` forward subset 已打通 compile-path
- `flash-attn` multi-GEMM compute kernel 不再 hang
- supported small bf16 MHA 子集现在会真实执行 direct runtime，
  并在 TT-Sim 上和 reference 数值对齐
- `K` staged-copy reader 的 transpose truth 已从
  `TTProgram -> ExecutableSpec -> accessor/materialization schema`
  显式下沉为 `transpose_2d`，host tilize / readback 会按该 truth 做 2D transpose
- 但更宽 `MHA / GQA` 子集、较大 shape 与 TT-Sim `float16` 路径仍未完成；
  当前大 shape `float16` case 在 simulator 上仍会命中
  `UntestedFunctionality: tensix_execute_unpacr: fp16`

这还不算完成。
`Phase C2` 完成必须同时满足：

1. 支持的 `MHA / GQA` subset 不再依赖 skip；
   runtime case 必须真实执行而不是走 unsupported gate
2. multi-GEMM compute contract 必须成为显式 target truth，
   进入 `TTProgram` / `ExecutableSpec`；
   不能靠 legacy attrs、名字匹配或 codegen heuristics 侧推
3. TT-Sim runtime 结果必须和 reference 数值对齐，
   而不是只证明“不挂死”
4. 对仍未支持的 shape / variant，必须继续 fail-fast 并给出明确诊断，
   不能退回 silent fallback 或 late runtime failure

### 6.2 Wider Family 承接

`Phase C` 不是只为 copy / GEMM / 当前 `flash-attn` subset 服务。
以下 family 仍在本阶段范围内：

- `topk`
- `fusedmoe`
- `paged decode`
- `chunk recurrence`

对每个 family，完成的最低标准不是“能 lower 一点点”，而是：

1. 走完整主链：
   `Stateful Semantic IR -> Spatial Program IR -> TT Target IR`
2. target truth 从 `TTProgram` / `ExecutableSpec` 物化，
   不靠 family-specific fallback 或 runtime/codegen 补洞
3. 至少有一组明确支持的 subset，
   带有对应的 transform / target regression
4. 对未支持部分有显式 unsupported / fail-fast 边界，
   而不是晚到 codegen/runtime 才炸

### 6.3 Target-Side Block Planning And Partition Intent

当前 GEMM / multi-GEMM 主链暴露出的问题，不是“还差一个更大的 test tile”，
而是三层 truth 仍然混在一起：

- case / lowering 里的 staging shape
- target-side block sizing
- `TTCoreGroup.work_packets` 调度结果

长期设计必须把三者拆开。

#### `SpatialProgram` 负责什么

`SpatialProgram.work_partitions` 只负责抽象切分空间，而不负责 exact target size。
它的长期职责是表达：

- 哪些轴属于 parallel partition space
- 哪些轴属于 reduction / serial partition space
- tail 是否允许、如何处理
- overlap / halo 是否允许
- producer-consumer reuse / closure 约束

换句话说，前面各层“写什么”只应该通过这些 abstract partition intent
影响后端；它们不应该直接决定 Blackhole 上每块具体有多大。

#### `TTProgram` 负责什么

具体 block 大小必须在 `Phase C` 里由 target planner 决定，
因为这一步显式依赖 `TTHardwareModel`：

- tile atom / legal divisibility
- L1 / CB 容量
- buffering policy
- subblock legality
- family-specific compute / transport constraints

因此 `TTProgram` 需要新增 `TTBlockPlan`，作为 target-side concrete sizing truth。
`TTBlockPlan` 的最小职责是承接：

- 绑定哪个 `work_partition`
- 绑定哪个 `kernel / core_group`
- `parallel_block_shape`
- `reduction_slice_shape`
- `subblock_shape`
- `buffering_policy`
- `tail_policy`
- 必要的 target-local traits / payload

这里的 `reduction_slice_shape` 是 general 数据切分概念；
GEMM 里的 `K` 只是其中一个实例，不应把长期抽象命名绑死在 GEMM noun 上。

#### Planner 与下游对象的职责

`TTBlockPlan` 一旦存在，后续对象的 owner 关系必须改成：

```text
WorkPartition intent
  + TT local compute / transport truth
  + TTHardwareModel
  -> TTBlockPlan
  -> TTCBPlan
  -> TTCoreGroup.work_packets
```

其中：

- `TTCBPlan` 只承接 block plan 的资源后果，不再从 case-local shared shape
  直接抄大小
- `TTCoreGroup.work_packets` 只表示最终调度，
  不再承担 decomposition truth owner 职责
- `AssignBlackholeCores` 只负责把已决定好的 block work
  分配到 physical cores；它不是 first-principles block planner

#### 对前面各层的影响边界

这套设计不会要求 `Stateful Semantic IR` 或 `SpatialProgram`
提前知道硬件上的“合适大小”。
前面各层只会在下面两类信息缺失时影响我们：

1. 没有把 partition legality / reuse / tail truth 表达清楚，
   导致 target planner 被迫重新猜测“哪些切分是合法的”
2. 没有把 op-local compute / transport truth 表达清楚，
   导致 target planner 无法正确估算 target-local legality和资源成本

反过来，前面各层写下的 case-local staging shape、
单个 sample 的 shared-buffer 维度、或 ad-hoc loop carving，
都不应该继续作为长期 target block size owner。

这意味着：

- migration 期间它们可以暂时作为 lowering hint 存在
- 但 `Phase C` 结束前，typed `TTBlockPlan` 必须成为唯一稳定真源

完成标准：

1. 支持的 GEMM / multi-GEMM family 已显式物化 `TTBlockPlan`
2. `PlanBlackholeCB` 与 core assignment 只消费 `TTBlockPlan`，
   不再从 case-local shape 或松散 payload 猜 sizing
3. validator、runtime、codegen 对 block legality 的判断共享同一套 truth
4. 未支持 family / shape 继续 fail-fast，而不是退回 ad-hoc fallback

### 6.4 Wider Copy / Dataflow 支持面

当前稳定支持面仍然很窄：

- copy：equal source/dest range，且 stride = 1
- GEMM：A/B-separated reader range + writer output range；
  当 `core_plan.work_packets` oversubscribe physical cores 时，
  direct runtime 现在会按 packet truth 分 wave 发射，
  但仅限没有显式 `semaphore / remote-core` synchronization contract 的 executable
- accessor：仅 interleaved + DRAM + `common_runtime_arg_count = 0`

因此“更宽 copy/dataflow 支持面”必须明确理解为：

1. 新 admitted shape 要通过 typed ABI / accessor / transport contract 表达，
   不允许再靠 ad-hoc attrs 或 runtime 猜测
2. `ValidateTTTargetProgram`、runtime 和 codegen
   必须共享同一套 legality gate
3. 每个新增支持形态都要有
   pipeline/spec regression，必要时再加 direct runtime regression
4. 不能把“unsupported 但可诊断”误算成支持面完成

补充边界：

- current direct runtime 还没有把 `work_packets.work_count`
  下沉成 device-side per-core serial loop contract；
  对 oversubscribed executable 的当前正式实现是 host-side wave scheduling
- 一旦 executable 带显式 `TTSemaphorePlan`、`semaphore_bindings`
  或 `remote_core_descriptors`，oversubscribed launch 仍应 fail-fast，
  不能假设 repeated launch 和单次并发 launch 语义等价

### 6.5 Wider Synchronization 支持面

当前 `TTSemaphorePlan / TTComputeSyncPlan` 对象已经在位，
但“对象存在”不等于同步支持面已经完成。

本工作流的完成标准是：

1. 需要承接的 barrier / multicast arrival / phase handoff /
   compute-sync 模式，必须成为显式 target object，
   而不是散在 runtime/codegen 的特殊分支
2. host-side materialization 必须只从
   `TTProgram` / `ExecutableSpec` 读取同步 truth
3. validator、runtime 和 codegen 对同步协议的合法性判断必须一致
4. 新同步模式进入支持面时，必须有对应 regression；
   不能靠“当前 case 没挂”代替协议闭环

### 6.6 Object-Boundary / Typed Uplift 后续

下面这些还不能长期停留在“payload 里先放着”的状态：

- `Placement`
- `SpatialCapabilityModel` 的 quantitative fields
- 部分 payload-backed spatial / target node truth

这个工作流的完成标准是：

1. 真正被 planning / mapping / legality consumer 使用，
   并对 target decision 有可见影响
2. 或者被明确收窄、移出当前 active contract，
   不再伪装成“将来可能有用”的长期字段

换句话说，`Phase C` 结束时，这些字段不能继续只是“挂着但没人消费”的 metadata。

## 7. 完成判定

只有下面这些条件全部成立，`Phase C` 才能算完成：

1. `TTProgram` cutover 主链已稳定：
   `TTProgram` 是唯一 target truth，
   reader-side deletion gate 已收口
2. `flash-attn` 的 `Phase C2` runtime / correctness payoff 已完成，
   不再停留在 explicit unsupported gate
3. `topk / fusedmoe / paged decode / chunk recurrence`
   等 family 已按新主链统一承接，并具备明确支持 subset 与 regression
4. copy/dataflow 支持面已超出当前窄 baseline，
   并通过 typed contract + validator + regression 落地
5. synchronization 支持面已通过
   `TTSemaphorePlan / TTComputeSyncPlan` 等显式对象稳定承接
6. `Placement / SpatialCapabilityModel / payload-backed truth`
   不再只是悬空 metadata，而是被真实消费或被明确收窄
7. shared zero-regression baseline 与 `Phase C2` runtime gate 持续通过

当前状态是：

- 条件 1 已满足
- 条件 2 已取得 small bf16 correctness milestone，但整体仍未满足
- 条件 3-7 仍未满足

## 8. Shared Zero-Regression Baseline

```bash
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q
pytest tilelang_repo/testing/python/transform/test_blackhole_tt_target_probe.py -q
```

## 9. `Phase C2` Runtime Gate

```bash
source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh
export TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo
export PYTHONPATH=/root/dev/vibe_dsl/tilelang_repo:/root/dev/vibe_dsl/tilelang_repo/3rdparty/tvm/python
export TVM_LIBRARY_PATH=/root/dev/vibe_dsl/tilelang_repo/build/lib
export LD_LIBRARY_PATH=/root/dev/vibe_dsl/tilelang_repo/build/lib:$LD_LIBRARY_PATH
cd /root/dev/vibe_dsl/tilelang_repo
pytest -q testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py \
       -k 'small_bf16_compute_source_keeps_acc_s_cast_cb_pages_consistent or \
           small_bf16_metadata_marks_k_materialization_as_transposed'
pytest -q testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py \
       -k 'small_bf16_forward_direct_runtime'
```

当前 gate 证明三件事：

- `flash-attn` small bf16 lowering 仍保持
  grouped reduce/cast page contract 与 `K.transpose_2d` metadata
- supported small bf16 MHA direct runtime 会真实执行，
  而不是依赖 unsupported skip
- small bf16 TT-Sim runtime 输出和 reference 数值对齐

当前 Blackhole runtime/direct-runtime regression baseline 统一使用 `bf16` 输入；
`fp16` 不再作为当前 TT-Sim 上的正式 runtime gate。

它不证明：

- 更宽 `MHA / GQA` runtime correctness 已经完成
- 大 shape `float16` TT-Sim runtime 已经可作为正式 correctness gate
- `Phase C2` 已完成
