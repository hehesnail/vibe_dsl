# Stage 4 Phase B: Spatial Program IR

## 基本信息

- **文档角色**: `Phase B` 已完成阶段文档
- **当前状态**: `2026-04-08` 已完成并完成审计收口
- **已完成子阶段**: boundary cleanup、capability intake、probe、最小 contract hardening、execution-bearing 收尾
- **仍未完成**: 无；后续工作已转入 `Phase C`
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
- `AnalyzeSpatialDomainPlan -> AnalyzeSpatialExecutionPlan -> MaterializeSpatialProgram
  -> ValidateSpatialProgram` 已接入主线，
  `LowerToSpatialProgram` 退化为兼容 wrapper，
  `LowerBlackholeOps` 已硬要求 `tl.spatial_program`
- `tl.tt_hardware_model` / `tl.spatial_capability_model` 已作为 module-scope global info 落地
- `AnalyzeSpatialDomainPlan` / `AnalyzeSpatialExecutionPlan`
  已消费来自 SoC descriptor 的最小 capability snapshot
- `Channel.kind + payload_kind + delivery_kind` 与 `placement.affinity_kind`
  已收成当前 probe intake 所需的最小 contract
- `LowerSpatialProgramToTTTargetProbe` 已落地，并且不会恢复 non-TT-specific spatial semantics
- `ValidateSpatialProgram` 已收正为 coherence / completeness gate
- `SpatialDomainPlan` / `SpatialExecutionPlan` 已落成 `Phase B` 内部 typed 中间契约，
  用于分离 domain/layout synthesis 与 task/channel/phase synthesis
- representative compile-path family gate 已覆盖：
  `copy / GEMM / flash-attn / topk / chunk_o / fusedmoe_routed / mla_decode_paged`

## 5. 本轮收尾结果

- generic builder 已把第 3 节定义的五类算法收进稳定主链
- `Task / Channel / Layout / WorkPartition / ProgramPhase / SyncEdge /
  Placement / ResourceIntent` 的 execution-bearing truth 已冻结成 typed spatial payload
- `AnalyzeSpatialDomainPlan -> AnalyzeSpatialExecutionPlan -> MaterializeSpatialProgram`
  已拆成显式 pass 链，`LowerToSpatialProgram` 仅保留兼容 wrapper；
  `spatial_program_builder.cc` 成为 shared synthesis owner
- `spatial_analysis.h/cc` 已成为 Phase B 的共享 helper / contract 真源；
  builder、validator、probe、domain plan 不再各自维护同逻辑拷贝
- `Task / Channel / Placement / SyncEdge / ProgramPhase / Layout /
  WorkPartition / ResourceIntent` 的高频链接字段已上提为 typed field，
  downstream consumer 主链接改走 typed/index contract
- name 字段继续保留，但只承担 display / debug / identity 角色；
  validator 的 referential integrity 已以 index linkage 为 canonical
- `phase_boundary_materialization` 已收窄为真实跨 phase state handoff，
  不再把“任何后续 phase 仍会读取的 state”过度扩成边界物化集合
- `ValidateSpatialProgram` 已对 stronger contract 做 coherence / completeness /
  semantic alignment / ordering legality fail-fast 校验
- `LowerSpatialProgramToTTTargetProbe` 与 `LowerBlackholeOps`
  已只消费冻结后的 typed spatial truth，不再回补 generic spatial 语义洞
- copy / GEMM fast path 仍保留，但它们只作为 execution-bearing
  `SpatialProgram` contract 的退化特例
- copy / GEMM fast path 与 generic path 现已共享 task / channel / placement /
  phase / sync contract construction helper，不再并行手写整套 schema

## 5.1 审计收口结论

对 `tasks/dev_design/phase_b_code_audit.md` 中列出的问题，`Phase C` 之前需要收口的部分已完成：

- 大规模重复 helper 已集中到 `spatial_analysis.h/cc`
- `lower_to_spatial_program.cc` monolith 已拆分
- capability model 已在 `AnalyzeSpatialDomainPlan` 发布到 module global info，
  后续 pass 直接复用
- validator 已改为共享分析 helper + structural / referential integrity gate
- dual linkage 已改成 index canonical、name display-only
- 高频 payload truth 已提升成 typed field，减少 stringly-typed 消费
- fast path 已共享 contract construction helper，避免 generic/fast path 漂移

仍保留为 `Phase C` 继续演进的项：

- 更深层的 IR node schema 分化
- quantitative capability field 被 planning / mapping 正式消费

## 5.2 当前收尾策略

`Phase B` 的剩余工作按单一路径收尾：

- 继续保留 copy / GEMM fast path，但它们只允许作为
  execution-bearing `SpatialProgram` contract 的退化特例
- generic builder 必须成为 `SemanticProgram -> SpatialProgram`
  的唯一 synthesis owner
- `Phase C` probe / downstream lowering 只允许消费冻结后的 typed spatial truth，
  不再补 non-TT-specific 语义洞

本轮不做的事：

- 不启动正式 `TTProgram / MaterializeTTExecutableSpec` cutover
- 不把 family-specific matcher 扩成新的主路径
- 不新增并行执行路径或额外 emitter 绕开当前 blocker

## 5.3 本轮必须收实的协议

为了让 `SpatialProgram` 成为 execution-bearing spatial program，
本轮必须把下面这些 truth 显式收成 typed payload / linkage contract：

- `Task`
  - formation basis
  - execution role
  - phase membership
  - update membership
- `Channel`
  - source / target task linkage
  - flow kind
  - payload kind
  - delivery kind
  - state / version linkage
- `Layout / WorkPartition`
  - semantic domain basis
  - domain transform kind
  - partition family
  - work-dependent bounds / partition evidence
- `ProgramPhase / SyncEdge`
  - partial-order basis
  - phase closure basis
  - ordering / visibility / materialization requirement
- `Placement / ResourceIntent`
  - execution / communication / phase-boundary obligation
  - state residency target
  - fragment / pipeline / lowering support contract

约束：

- display 字段继续保留，但只能承担 debug / identity /日志职责
- downstream consumer 的主链接一律走 typed payload / index linkage
- 若当前 schema 仍不足以稳定表达这些 truth，就先扩 schema，再写 builder / validator

## 5.4 收尾算法

### 5.3.1 Task Formation

- 以 `state_defs / state_uses / state_joins` 构成的 update-state 图为主输入
- mandatory cut 至少由以下事实驱动：
  - materialized state/version boundary
  - incompatible law class
  - incompatible domain transform / partition requirement
  - incompatible placement / communication domain
  - reduction completion / selection-index handoff / recurrence carry 的 ordering boundary
- 只有在 phase、flow locality、layout/partition 与 capability 都兼容时才允许 fuse
- `Task.kind` 与 execution role 必须来自 update signature / boundary requirement，
  不允许从 workload 名字、kernel 名字或固定 family 模板恢复

### 5.3.2 Flow Shaping

- 从 `state_defs / state_uses / state_joins` 派生 producer-consumer relation
- 显式区分：
  - `point_to_point`
  - `broadcast`
  - `gather`
  - `scatter`
  - `reduce_merge`
  - `carry`
- `selection_state / index_state / reduction_accumulator / carry`
  的 version handoff 必须显式进入 `Channel` contract
- capability 不支持的 flow family 直接 fail-fast，不做 generic collapse

### 5.3.3 Domain Realization

- 以 `Domain + UpdateLaw.access_maps + state role + supplement payload`
  为 domain realization 主输入
- 必须把下列 transform family 稳定表达进 `Layout / WorkPartition` contract：
  - derived
  - filtered
  - grouped
  - routed
  - paged
  - chunked
- 不允许继续靠“轴数 + `derived_indices` trait”二分出
  `regular/indexed` 与 `replicated/blocked`

### 5.3.4 Phase / Ordering Synthesis

- 先构造 task graph，再从 ordering-critical edge 合成 phase closure
- 需要进入 ordering synthesis 的 edge 至少包括：
  - carry
  - reduction completion
  - selection/index handoff
  - phase-boundary materialization
- `ProgramPhase / SyncEdge / phase_boundary_materialization`
  必须由 graph synthesis 冻结，不允许靠固定 `phase0_compute / phase1_stateful`
  模板命名来凑 phase

### 5.3.5 Validator Hardening

- `ValidateSpatialProgram` 必须把 stronger contract 当主真源
- validator 负责：
  - coherence
  - completeness
  - semantic alignment
  - ordering / phase closure legality
- validator 不负责 capability legality solver；
  capability 选择仍由 builder 在合法空间内完成

## 5.4 实施顺序

本轮实现顺序固定为：

1. 先补 failing tests，锁住 stronger contract 与 synthesis behavior
2. 再补 builder / schema，使 generic path 成为主 synthesis owner
3. 再补 `ValidateSpatialProgram` fail-fast 校验
4. 再验证 `LowerSpatialProgramToTTTargetProbe` 与 `LowerBlackholeOps`
   不再依赖 non-TT-specific spatial 补洞
5. 最后跑 shared zero-regression baseline

这一步之后，`Phase C` 的前置输入边界应收敛为：

```text
SemanticProgram
  -> AnalyzeSpatialDomainPlan
  -> AnalyzeSpatialExecutionPlan
  -> MaterializeSpatialProgram
  -> ValidateSpatialProgram
  -> LowerSpatialProgramToTTTargetProbe
```

probe 只能继续暴露 TT-specific demand，不能再回收 generic spatial truth。

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

- contract-hardening 与 execution-bearing 收尾已完成
- `Phase B` 已完成
- `Phase C` 可以正式把 `SpatialProgram` 当成单一上游真源，
  启动 `TTProgram / MaterializeTTExecutableSpec` cutover

## 6.1 完成后设计审计结论

`Phase B` 判定为已完成，不等于当前 object boundary 已经没有继续收敛空间。

本轮设计审计的结论是：

- 三层主分层本身成立：`Semantic -> Spatial -> TT Target`
  仍然是被现有 family 与旧主链 truth 混层问题共同逼出来的，不属于 cargo-cult 分层
- `ProgramPhase / SyncEdge / phase_boundary_materialization`
  当前仍有明确工程必要性；`Phase C` 应消费它们冻结后的 truth，
  不应重新发明 phase / ordering 语义
- `Placement` 是当前 `Phase B` object set 里最值得继续警惕的过度设计风险点；
  其长期保留价值取决于 `Phase C` 是否会把它消费成真实 target mapping constraint，
  而不只是停留在 builder / validator / probe contract
- `SpatialCapabilityModel` 的 categorical legality 面当前是成立的；
  其中 quantitative hardware fields 的长期价值，取决于
  `Phase C` 是否会把它们真正消费进 planning / mapping
- `ResourceIntent` 当前有真实 consumer，
  但后续必须继续保持 small-closed kind discipline；
  如果持续吸收“支持性信息”，它会重新退化成 attr bag

因此，`Phase B` 当前的正确处理方式不是重新打开返工，
而是带着这份风险清单进入 `Phase C`，
并在 target cutover 中继续验证这些边界是否真的必要、是否被下游稳定消费。

## 7. Shared Zero-Regression Baseline

```bash
pytest tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py -q
pytest tilelang_repo/testing/python/transform/test_blackhole_tt_target_probe.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q
```
