# Stage 4 Phase B: Spatial Program IR

## 基本信息

- **文档角色**: `Phase B` 当前主实施文档
- **当前状态**: `2026-04-07` 仍在进行中
- **已完成子阶段**: boundary cleanup、capability intake、probe、最小 contract hardening
- **仍未完成**: spatial synthesis algorithm 本体与 stronger execution-bearing contract
- **上游输入**: 冻结后的 `SemanticProgram`
- **下游输出**: 冻结后的 `SpatialProgram`
- **唯一总体设计**: `tasks/dev_design/final_blackhole_backend_redesign.md`

## 1. 作用域

`Phase B` 只负责一件事：

- 把冻结后的 algorithmic truth 组织成稳定的 spatial/dataflow program

它必须回答：

- 哪些 `Update` 组成哪些 `Task`
- 哪些 `State` 之间形成哪些 `Channel`
- 哪些 `Layout / WorkPartition / ProgramPhase / SyncEdge / Placement / ResourceIntent`
  必须显式存在
- 多 member device program 的 phase truth 如何固定

它不负责：

- 再做 semantic recovery
- 发明 TT resource / CB / semaphore / ABI
- 从 raw TIR、名字或 late builtin 恢复 non-TT-specific spatial semantics

## 2. 必须交付的 Spatial Contract

### 2.1 Core Objects

`Phase B` 的长期 core object set 只保留：

- `SpatialProgram`
- `ProgramPhase`
- `Task`
- `Channel`
- `Layout`
- `WorkPartition`
- `Placement`
- `SyncEdge`
- `ResourceIntent`

### 2.2 基本纪律

- 只消费冻结后的 semantic truth
- cross-function `ProgramPhase` 真相固定挂在 `tl.device_programs`
- analysis 决定 legality，policy 只在合法空间内选择
- 不允许名字匹配、raw fragment attrs、或 `Task:TTKernel = 1:1` 默认心智模型
- 不允许把 TT noun 写进 `Task / Channel / Layout / WorkPartition`
- 如果 `Phase C` 需要某个 non-TT-specific truth 才能合法 mapping，
  那个 truth 必须先进入 `SpatialProgram`

### 2.3 SpatialProgram 必须显式承载的 truth

- `Task`: formation basis、execution role、phase membership、update membership
- `Channel`: source/target、flow kind、payload kind、delivery kind、state/version contract
- `Layout / WorkPartition`: semantic domain basis、domain transform、partition family、
  work-dependent bounds
- `ProgramPhase / SyncEdge`: partial order、closure basis、ordering / visibility /
  materialization requirement
- `Placement / ResourceIntent`: execution / communication / phase-boundary obligation、
  state residency、boundary materialization、pipeline / fragment support

### 2.4 `SpatialCapabilityModel`

`Phase B` 读取抽象 `SpatialCapabilityModel`，不读取 TT resource noun。

最小必需能力面：

- topology / neighborhood
- placement domain
- supported flow kinds
- supported ordering / synchronization kinds
- supported residency / persistence kinds
- supported layout / partition kinds

职责边界：

- builder / probe 读取 capability
- validator 只做 coherence / completeness gate，不做 capability legality solver

## 3. `Phase B` 必须完成的算法职责

### 3.1 Task Formation

- 从 `update-state` 图出发，而不是从 workload 名字或 kernel 形态出发
- mandatory cut 至少由以下事实驱动：
  - materialized state/version boundary
  - incompatible law class
  - incompatible access/layout/partition requirement
  - incompatible placement / communication domain
  - ordered update / carry / reduction completion requirement
- 只在 phase、layout/partition、flow locality、capability 都兼容时允许 fuse
- `Task.kind` 由 execution signature 决定，不靠命名习惯

### 3.2 Flow Shaping

- 从 `state_defs / state_uses / state_joins` 派生 producer-consumer relation
- 显式区分 `point_to_point / broadcast / gather / scatter / reduce_merge / carry`
- 显式写出 `payload_kind / delivery_kind / state_index / version contract`
- capability 不支持的 flow family 必须 fail-fast，不能偷偷降级成 generic flow

### 3.3 Domain Realization

- 从 `Domain + AccessMap + UpdateLaw` 决定 `Layout / WorkPartition`
- 必须识别 derived / filtered / grouped / routed / paged / chunked 这些 transform
- 不允许只按轴数或 `derived_indices` 决定 `regular / indexed / blocked / replicated`

### 3.4 Phase And Ordering Synthesis

- 从 task graph 构造 must-happen-before relation
- 明确 carry、reduction completion、selection/index handoff、phase-boundary materialization
  这些 ordering-critical edge
- 决定哪些边可同 phase 局部闭包，哪些必须跨 phase materialize
- 生成 `ProgramPhase / SyncEdge / phase_boundary_materialization`
- 不允许靠固定 `phase0_compute / phase1_stateful` 模板凑 phase

### 3.5 Capability-Informed Legality And Policy

- 先裁掉 capability 不支持的 candidate
- 再在合法空间内做 split/fuse、layout/partition、communication shaping 选择
- 不允许 policy 先选，再让 validator 兜底

## 4. 当前已完成的部分

- `Spatial*` object/vocab/shared key 已从 semantic infra 拆出
- `LowerToSpatialProgram -> ValidateSpatialProgram` 已接入主线，
  `LowerBlackholeOps` 已硬要求 `tl.spatial_program`
- `tl.tt_hardware_model` / `tl.spatial_capability_model` 已作为 module-scope global info 落地
- `LowerToSpatialProgram` 已消费来自 SoC descriptor 的最小 capability snapshot
- `Channel.kind + payload_kind + delivery_kind` 与 `placement.affinity_kind`
  已收成当前 probe intake 所需的最小 contract
- `LowerSpatialProgramToTTTargetProbe` 已落地，并且不会恢复 non-TT-specific spatial semantics
- `ValidateSpatialProgram` 已收正为 coherence / completeness gate
- representative compile-path family gate 已覆盖：
  `copy / GEMM / flash-attn / topk / chunk_o / fusedmoe_routed / mla_decode_paged`

## 5. 当前仍未完成的部分

- generic builder 还没有把第 3 节定义的五类算法完整收进主链
- `Task` 仍缺 formation basis 与 abstract execution role
- `Layout / WorkPartition` 还不能把 grouped / paged / routed / chunked
  一等表达成稳定 contract
- `ProgramPhase / SyncEdge` 还缺 closure / ordering / visibility /
  materialization basis
- `ValidateSpatialProgram` 还没有对这些 stronger contract 做完整 fail-fast 校验
- 因此当前 `SpatialProgram` 只达到 read-only probe intake 的最小上游 contract，
  还不是最终形态的 virtual spatial program

## 6. 完成判定

只有在下面这些条件全部成立后，`Phase B` 才能算完成：

1. 第 3 节定义的五类算法职责已进入稳定 builder 主链
2. 第 2.3 节列出的 execution-bearing truth 已显式进入 `SpatialProgram`
3. `Phase C` 不再需要 probe 驱动的非 TT 语义补洞
4. `ValidateSpatialProgram` 已能对 stronger contract 做 fail-fast 校验
5. 实现不再依赖本文明确禁止的 shortcut：
   - workload-name tasking
   - generic flow collapse
   - axis-count-only layout / partition
   - 固定 phase 模板
   - policy 先行再让 validator 兜底
6. shared zero-regression baseline 持续通过

当前结论：

- contract-hardening 子阶段已完成
- `Phase B` 整体未完成

## 7. Shared Zero-Regression Baseline

```bash
pytest tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q
```
