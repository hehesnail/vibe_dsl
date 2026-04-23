# Blackhole First-Principles Protocol Audit

> 本文档不是新的总体设计。
>
> 它只做一件事：
> **基于第一性原理，对现存 historical surface 做表示层落点、validator 和删除/切换 disposition。**
>
> 下表里出现的现存 surface 名，
> 全部按当前仓库里的历史字面名列出，
> 目的只有一个：做删除与切换清单。
> 它们不是当前允许继续扩展的协议名。

## 1. 判定标准

对 spatial/dataflow target，
长期语义只能稳定存在于下面四层显式表示之一：

1. `Normalized Tile TIR`
2. `SpatialPlan`
3. `TTProgram`
4. `ExecutableSpec`

如果某个 historical surface
不能稳定归到这四层之一，
或者它本身不是 verifier 可见的显式表示，
它就不能继续作为长期协议。

## 2. Deletion / Migration Table

| 现存 surface | 长期表示层落点 | 显式对象 / 语义来源 | 为什么当前 surface 必须退场 | validator / gate | 去留 |
|---|---|---|---|---|---|
| `blackhole.copy_semantics` | `Normalized Tile TIR -> SpatialPlan -> TTProgram` | `BufferLoad / BufferStore`、`DataflowEdge`、`TTTransportPlan` | 不能继续充当 copy 方向/角色的长期表示 | `ValidateSpatialPlan` 检查 edge completeness；`ValidateTTProgram` 检查 transport realization | 删除 |
| `blackhole.segment_kind` | `TTProgram -> ExecutableSpec` | `TTKernelPlan.kind`、投影后的 executable kernel/segment 记录 | 不应再写回 TIR attr | `ValidateTTProgram` 检查 kernel/ABI/transport 闭合；leaf readers 只读投影记录 | 删除 |
| `blackhole.work_decomposition` | `TTProgram` | `TTBlockPlan`、`TTExecutionPlan` | 不属于 `SpatialPlan` 公开表示 | `ValidateTTProgram` 检查 work placement / wave legality | 删除 |
| `blackhole.compute_regions` | `Normalized Tile TIR -> SpatialPlan` | anchored sub-TIR、`ExecutionUnit` | 不能继续作为 compute 语义 bag | `ValidateSpatialPlan` 检查 execution-unit coverage | 删除 |
| `blackhole.pipeline_stages` | `SpatialPlan -> TTProgram` | `PhasePlan`、`TTSyncPlan` | 不能继续作为跨层 bag | `ValidateSpatialPlan` 检查 phase/order；`ValidateTTProgram` 检查 completion/materialization | 删除 |
| `blackhole.lowering_requirements` | 拆回当前表示层直接读取 + `TTProgram / ExecutableSpec` | 当前 IR / 当前显式对象上的直接读取、必要的 leaf projection 记录 | 不是长期公共协议，也不应继续保留 internal builder bag | 各层 validator 分别检查 completeness；leaf 禁止反向依赖 planning bag | 删除 |
| `blackhole.resource_plan` | `TTProgram` | `TTTransportPlan`、`TTSyncPlan`、`TTExecutionPlan` | 是 canonicalization 时代的影子产物 | `ValidateTTProgram` 直接验证 canonical target representation | 删除 |
| `tl.internal_tt_*` | `TTProgram` | `TTProgram` 显式 slices | 只能短期 bridge，不是正式协议 | `ValidateTTProgram` 只接受显式 slices，不接受 internal attr bag | 删除 |
| `TTProgram.payload` 大袋子 | `TTProgram -> ExecutableSpec` | `TTProgram` 显式 slices + leaf projection payload | 只能保留 leaf 投影级 payload，不能反向充当 planning source | `ValidateTTProgram` 禁止 payload 反客为主；`ValidateExecutableSpecProjection` 约束 leaf-only projection | 已收紧 |
| `ExecutableSpec` 中的 raw payload | `ExecutableSpec` | leaf projection | 只能是投影结果，不能反向变成 planning source | `ValidateExecutableSpecProjection` | 已收紧 |

## 3. 当前 cleanup 解释

`Legacy Protocol Deletion`
在 repo HEAD 的目标含义固定为：

- canonical `LowerToBlackholePhaseB`
  不再发布
  `blackhole.work_decomposition /
   blackhole.compute_regions /
   blackhole.pipeline_stages`
- `PlanTTBlocks -> PlanTTExecution`
  直接增量写 staged
  `tl.tt_program`；
  `BuildTTProgram`
  不再接受
  `tl.internal_tt_*`
  这种过渡语义载体
- leaf / resource path
  不再发布
  `blackhole.lowering_requirements`
  与
  `blackhole.resource_plan`
- 为了让 optimized/helper 入口
  还能把 pre-opt logical tile bridge spec
  对齐回 optimized device func，
  只允许一个窄 internal bridge attr：
  `tl.blackhole_logical_buffer_tile_bridge_specs`
  它只承接
  `buffer_tile_bridge_specs`
  这一个 leaf-local bridge surface，
  不重新引入整袋
  `blackhole.compute_regions`
- `AnalyzeBlackholeWorkDecomposition /
   AnalyzeBlackholeComputeRegions /
   AnalyzeBlackholePipelineStages`
  这些 public wrapper
  与对应的 internal evidence helper
  都应从 active chain 删除，
  不能继续以 debug / regression helper 名义常驻

### 3.1 `2026-04-17` Task 0 落地补充

- `SelectBlackholeTTMetalBuiltins`
  现在已经位于
  `PlanTTBlocks`
  与
  `PlanTTCompute`
  之间，
  compute-side exact builtin 选择前移到 planner helper 路线之前；
  但 repo HEAD 里
  它仍只是 front-door wrapper，
  真正的 primitive idiom
  match + rewrite owner
  仍在 `PlanTTKernelABI`
- task0 的 owner truth
  仍然只能写成
  当前 `Normalized Tile TIR`
  的 selected exact-builtin
  postcondition；
  `CB / runtime arg / semaphore / launch`
  这类 TT-Metal
  program-construction 事实
  仍属于
  `TTProgram -> ExecutableSpec`
  边界，
  不能因为 repo HEAD
  还保留 seed / payload residue
  就倒灌回 task0
- `compute_epilogue_ops`
  不再属于 repo HEAD 的 active protocol：
  顶层 key
  已从
  `TTProgram.payload`、
  executable projection
  和 payload/spec 测试基线移除；
  但 nested
  `compute_contract.epilogue_ops`
  仍在 runtime compatibility metadata
  中存活；
  direct runtime
  也仍保留
  `compute_contract -> multi_compute_contracts -> gemm_contract`
  fallback，
  这些都只属于 task3 负责删除的
  leaf compatibility debt
- 旧 helper/composite builtin 名称
  不再是 active IR surface；
  selector / validator
  会按 exact op 名 fail-closed 拒绝 residue
- rowwise flash-attn 相关的 local pseudo builtin surface
  已从 active lowered IR 退场；
  但 `builtin_blackhole.{h,cc}`
  和 `codegen_blackhole.cc`
  里仍有 helper-named alias accessor /
  alias dispatch residue，
  所以 builtin/codegen surface
  还没完全收口
- selector 创建的 exact temporary CB requirement
  必须经由
  `blackhole.cb_requirements`
  seed 到
  `PlanTTCompute / PlanTTCBAlloc`；
  否则下游只会看见 dangling `requirement_index`；
  这代表的是 forced implementation debt，
  不是合法中间层边界
- `blackhole.cb_requirements`
  仍是
  `PlanTTCBAlloc`
  的 live planner input，
  不是已经只剩文档清理的死字段；
  required end-state
  是 selection 不再依赖它，
  downstream planner
  只能从当前表示做局部 derived analysis
  或从显式 `TTProgram`
  slice 读取结果
- `tl.blackhole_lowering_requirements_seed`：
  它只承接
  `buffer_materialization_contracts`
  与
  `buffer_tile_bridge_specs`
  这两个稳定 seed，
  供 selector-forwarding 跨 rewrite 保持桥接事实，
  并在最终
  `TTProgram`
  物化前剥离；
  它不是新的 planning 语义，
  也不是可被文档合法化的
  中期协议层

### 3.2 `2026-04-21` Task 1 边界收紧

- task1 当前真正的 wrong-now boundary
  不是 leaf reader
  本体，
  而是
  `blackhole.compute_regions`
  /
  `AnalyzeBlackholeComputeRegionEvidence`
  仍然充当
  logical bridge handoff
  的 producer-side owner truth
- repo HEAD 当前
  `tl.blackhole_logical_buffer_tile_bridge_specs`
  还不是 direct capture owner；
  它只是
  `lower.py`
  /
  测试 helper
  从
  `blackhole.compute_regions`
  复制出来的窄 handoff。
  task1 的 required end-state
  是把这段 handoff
  切回 current-stage
  direct capture，
  不是继续保留
  broad bag copy
- `tl.blackhole_logical_buffer_tile_bridge_specs`
  只允许被写成
  cleanup exception：
  它是 leaf-local
  temporary carrier，
  不是新的
  `SpatialPlan`
  /
  `TTProgram`
  /
  `ExecutableSpec`
  语义层，
  也不是 TT-Metal
  target/runtime contract
- `buffer_tile_bridge_specs`
  在 repo HEAD
  里同时活在两边：
  planner 侧
  `blackhole_lowering_requirements`
  /
  `lower_blackhole_ops`
  仍把它当 active residue；
  leaf 侧
  `TTProgram.payload`
  /
  executable projection
  /
  `codegen_blackhole.cc`
  仍把它当 compatibility surface。
  task1 的 required wording
  不是给这条路径
  找新合法身份，
  而是把它锁死成
  后续必须删除的 debt
- `rt_mod_blackhole.cc`
  当前不是
  `buffer_tile_bridge_specs`
  的直接 reader；
  直接 leaf consumer
  是 `codegen_blackhole.cc`。
  因此 task1
  不能把
  “runtime/codegen debt”
  写成模糊大口袋
- 这条 broad bag
  -> Python/helper reread
  -> narrow attr
  的路径，
  在 repo 内也没有成熟 backend 先例。
  当前成熟模式
  是 pass-local collector
  或 narrow temporary attr
  直接服务同层 rewrite /
  projection，
  不是把 public structured bag
  当长期跨阶段协议面

### 3.3 `2026-04-20` Task 2 边界收紧

- task2 当前真正的 wrong-now 边界
  不在
  `BuildTTProgram`
  / runtime / codegen reader
  本体，
  而在 planner 侧仍然复用：
  public
  `AnalyzeBlackhole*`
  wrapper、
  internal
  `*Evidence(...)`
  helper、
  以及
  `blackhole_lowering_requirements`
  这类 broad
  `Map<String, Any>`
  semantic bag
- `BuildTTProgram`
  在 repo HEAD
  里已经是 staged
  `TTProgram`
  的聚合 / 清理点；
  它会剥离
  `tl.internal_tt_*`
  与
  `tl.blackhole_lowering_requirements_seed`
  之类中间 attr。
  task2 的 required end-state
  不是“再给 `BuildTTProgram`
  发明一层 replacement staging bag”，
  而是让
  `PlanTTCompute`
  之前的 active consumer
  直接读取当前 IR /
  当前 `SpatialPlan`
  或 pass-local helper
- repo 内成熟 backend
  的共同模式
  也是：
  local collector /
  pass-local state
  直接服务当前 rewrite，
  然后冻结成显式 attr /
  typed object /
  leaf projection。
  没有成熟先例
  会把 public wrapper
  + internal evidence helper
  + broad
  `Map<String, Any>`
  总包
  合法化成长期 planner 边界
- `BuildBlackholeLoweringRequirements`
  当前还会产出一批
  repo 内找不到 active reader
  的 bag-only residue，
  例如
  `work_axes`、
  `derived_index_expr_count`、
  `work_dependent_loop_bound_count`、
  `spatial_phase_count`、
  `spatial_channel_count`、
  `spatial_phase_boundary_buffers`
  和
  `pipeline_loop_vars`。
  这些字段不能迁到 replacement bag；
  required end-state
  是直接删除，
  或把仍然需要的事实
  收回当前 IR /
  当前 `SpatialPlan` /
  pass-local helper
- runtime / codegen / build
  reader
  现在已经主要站在
  `TTProgram -> tl.blackhole_executable`
  projection
  边界上，
  不再直接消费
  public / internal legacy analysis bag。
  这不等于 leaf residue
  已经收口：
  `buffer_tile_bridge_specs`
  /
  contract payload
  这类 payload/projection debt
  仍属于 task3，
  `blackhole.segment_kind`
  这类 leaf-local slicing residue
  仍属于 task4
- 因此 task2 文档必须同时写清楚两件事：
  1. public/internal legacy analysis bag
     是 architecturally wrong，
     必须删除
  2. task3 / task4
     尚未删除的 leaf payload /
     projection /
     slicing residue
     不能反过来把这些 bag
     合法化成“仍然需要的中间层”
- TT-Metal
  的稳定 target truth
  也只有
  program / kernel /
  circular-buffer /
  semaphore /
  runtime-arg
  这类显式对象与 API。
  它没有提供任何
  target-model 理由，
  允许
  `work_decomposition`
  /
  `compute_regions`
  /
  `pipeline_stages`
  /
  `lowering_requirements`
  这种 planner-side broad bag
  继续存活，
  也没有理由把
  `BuildTTProgram`
  重新写成 semantic owner

### 3.4 `2026-04-21` Task 3 边界收紧

- task3 当前真正的 wrong-now boundary
  不是 target / codegen / runtime
  reader，
  而是 compiler-side
  active prepass
  `AnnotateBlackholeCopySemantics`
  以及它的三个
  implementation consumer：
  `BlackholeDeviceResourceCanonicalization`、
  `SplitBlackholeKernel`、
  `PlanTTKernelABI`
- `blackhole.copy_semantics`
  在 repo HEAD
  里实际写在
  `For.annotations`
  上，
  不是新的
  `AttrStmt`
  wrapper；
  因此 task3 的 deletion target
  是 loop annotation carrier
  本身，
  不是把旧 schema
  平移到另一个 wrapper / helper
- 这三个 consumer
  也不共享同一个合法可见表示层：
  `BlackholeDeviceResourceCanonicalization`
  运行早于
  `SpatialPlan`
  构造，
  只能从当前 TIR
  做 direct recovery；
  `SplitBlackholeKernel`
  运行时 validated
  `SpatialPlan`
  已存在，
  但 repo HEAD
  里它自己已经有部分 direct fallback，
  所以 task3 的要求是
  完成 owner-truth cutover，
  不是把它误写成
  “从零开始统一改成
   `SpatialPlan.DataflowEdge`”；
  `PlanTTKernelABI`
  已经有大块 direct copy lowering logic，
  真正还靠 annotation
  的是 copy buffer/shape/runtime-arg
  bookkeeping
- repo 内成熟模式
  也不是
  “先发一个 shared copy contract，
   再让多个 pass 去读”。
  更接近的先例是
  `lower_ptx_async_copy`
  这种 local structural recovery：
  从当前
  `BufferLoad / BufferStore`
  subtree
  直接匹配，
  然后在本地完成 rewrite。
  task3 的 required end-state
  应按这个模式写，
  不是再造 exported copy matcher /
  helper bag
- `SelectBlackholeTTMetalBuiltins`
  不是 task3 当前的主 blocker：
  compute builtin selection
  走
  `select_compute_builtins_only_`
  路径，
  不应再被文档描述成
  copy annotation
  的主要 consumer
- target / codegen / build / runtime
  现在已经主要站在
  `TTProgram -> tl.blackhole_executable -> ExecutableSpec`
  projection
  边界上，
  不直接读取
  `blackhole.copy_semantics`；
  现存
  `buffer_tile_bridge_specs`
  /
  `compute_contract`
  /
  `multi_compute_contracts`
  /
  `gemm_contract`
  fallback
  /
  `blackhole.segment_kind`
  这类 leaf residue
  不能反过来把
  copy annotation
  合法化成“还需要保留的中间层”
- 其中
  `rt_mod_blackhole.cc`
  与
  `blackhole_module.cc`
  当前还保留
  `compute_contract <- gemm_contract`
  的 leaf/runtime compatibility recovery。
  这属于 task3 的
  wrong-now, resolve-later
  debt：
  required end-state
  是 leaf/runtime
  直接消费显式
  `compute_contract`
  /
  `multi_compute_contracts`
  或其它明确 projection，
  不是继续在 executable side
  从旧 contract family
  推断 compute truth
- TT-Metal
  的稳定 target truth
  也只有
  kernel / circular-buffer /
  semaphore / runtime-arg /
  launch
  这类显式对象与 API；
  copy/data movement
  是 kernel 行为，
  不是 target-side semantic tag。
  因此没有任何 target-model 理由
  要求保留
  compiler-side
  `blackhole.copy_semantics`

### 3.5 `2026-04-21` Task 4 边界收紧

- task4 当前真正的 wrong-now boundary
  不是
  `BuildTTProgram`
  /
  `MaterializeBlackholeExecutable`
  /
  `codegen_blackhole.cc`
  这些 downstream reader，
  而是两处还在直接依赖
  `blackhole.segment_kind`
  的地方：
  planner 侧
  `PlanTTKernelABI`
  仍把 body attr
  当成
  `segment_plan_`
  truth 来源，
  并继续通过
  `AnalyzeCBDepthEffect`
  /
  `current_segment_kind_`
  驱动
  segment-sensitive
  CB depth /
  accessor bookkeeping；
  leaf 侧
  `rt_mod_blackhole.cc`
  仍用
  `SegmentBodyExtractor`
  按 attr
  切 raw body
- `TTKernel.kind`
  /
  `TTKernelPlan.kind`
  /
  projected executable
  `segment_plan.kind`
  在 repo HEAD
  里已经是显式字段，
  并且 downstream projection /
  build /
  codegen /
  runtime metadata
  reader
  已主要站在这条显式链上。
  因此 task4 的 required end-state
  不是“去下游 reader
  重新找一个新的 kind 来源”，
  而是让 planner-side
  owner truth
  直接构造到这些显式对象里
- `BuildTTProgram`
  只是 staged
  `TTProgram`
  的聚合 / 校验点，
  不是 kernel-kind classifier；
  `MaterializeBlackholeExecutable`
  只是 projection writer；
  `codegen_blackhole.cc`
  只是 projection reader。
  这些点不能再被文档误写成
  task4 的主 attr consumer
- repo 内成熟模式
  也不支持
  `blackhole.segment_kind`
  继续当中期边界：
  `SplitHostDevice`
  /
  `KernelInfo`
  /
  `runtime::FunctionInfo`
  /
  `kernel_metadata`
  这类 repo-local
  先例，
  都是把 per-kernel truth
  冻结到显式对象 /
  显式 projection，
  而不是保留
  cross-pass marker attr
- `SplitBlackholeKernel`
  当前仍会发出
  `blackhole.segment_kind`
  marker，
  但它不是长期 kind truth；
  只要
  `rt_mod_blackhole.cc`
  仍按 raw TIR body
  做 segment slicing，
  这个 marker
  就只能被记成
  architecturally wrong
  的 leaf-local residue，
  不能再回升成 planning /
  projection /
  runtime-schema
  的合法中间层
- `rt_mod_blackhole.cc`
  当前其余 runtime metadata
  reader
  已经主要站在
  projected executable
  `segment_plan`
  和显式
  `kernel.kind`
  /
  `kernel.core_type`
  上；
  真正还碰
  `blackhole.segment_kind`
  的地方
  只剩
  `SegmentBodyExtractor`
- TT-Metal
  target model
  也不要求
  source-level
  kernel-kind marker。
  它要求的是
  `Program`
  里的显式 kernel objects、
  `CreateKernel`
  时选定的
  kernel class/config、
  以及最终 source /
  runtime args。
  所以 task4
  里的 body slicing
  只能被写成
  compiler-local
  wrong-now residue，
  不能被描述成
  target/runtime
  必需协议
- task4 还必须显式保留
  task3 前置依赖：
  `SplitBlackholeKernel`
  当前 reader/writer
  classification
  仍受
  `blackhole.copy_semantics`
  影响。
  task4 可以先切
  `segment_kind`
  owner truth，
  但不能把
  `copy_semantics`
  依赖
  合法化成 splitter
  的终态设计
- task4 当前测试主面
  已经主要落在
  `TTProgram.kernels[*].kind`
  /
  executable
  `segment_plan[*].kind`。
  Python 树里
  没有一批
  必须保留的
  raw
  `blackhole.segment_kind`
  schema tests；
  required end-state
  是继续让测试
  站在显式 kind truth
  上，
  而不是为了“兼容旧测试”
  给 marker 续命

### 3.6 `2026-04-21` Task 5 收敛合同收紧

- task5 不是新的协议删除 task，
  也不是
  “前面 residue
   还没删干净时，
   最后再跑一轮 grep /
   build /
   push”
  的扫尾脚本；
  它只负责证明
  task0-task4
  已定义的 forbidden residue
  真的退出 active chain，
  并把最终状态
  同步到
  `progress` /
  cleanup docs /
  protocol audit /
  memory /
  verification matrix
- task5 不能本地豁免
  task0-task4
  尚未删除的 wrong-now carrier。
  当前 repo HEAD
  里仍然 live 的
  residue
  包括：
  public
  `AnalyzeBlackhole*`、
  `blackhole.lowering_requirements`、
  `blackhole.resource_plan`、
  `blackhole.copy_semantics`、
  `blackhole.segment_kind`、
  `blackhole.work_decomposition`、
  `blackhole.compute_regions`、
  `blackhole.pipeline_stages`、
  `blackhole.cb_requirements`
  和
  `tl.internal_tt_*`
  等 internal companion attrs。
  task5 的职责
  不是替这些 residue
  重新找合法理由，
  而是要求前序 task
  先完成删除，
  然后再做最终收敛验证
- task5 的 grep / scan
  合同必须直接继承
  cleanup overview、
  task0-task4 completion contract
  和当前 progress
  里真实还在跟踪的 residue；
  不能继续使用
  `ComputeLoweringFacts`、
  `MatchTTMetalComputeLoweringWindows`、
  `TryLowerRowwiseFlashAttnRegion`
  这类已经脱离 active cleanup 边界的历史名字
- repo-local
  成熟后端先例
  已经把最终 completion truth
  收到
  `tl.tt_program`
  /
  `tl.blackhole_executable`
  /
  `ExecutableSpec`
  /
  `artifact.rt_mod`
  这条显式 artifact /
  module 链上：
  `tvm_ffi`
  路径要求
  `artifact.rt_mod`
  才能执行 / 导出，
  而
  `test_blackhole_copy_build_reads_executable_without_legacy_projection_attrs`
  /
  `test_blackhole_gemm_spec_survives_without_legacy_contract_attrs`
  也已经直接证明
  legacy projection/contract attrs
  不是最终交付 truth
- task5 的验证矩阵
  必须按当前 admitted support surface
  写：
  compile /
  source /
  projection /
  codegen /
  `tvm_ffi` export
  是 hard gate；
  copy / GEMM
  admitted direct-runtime
  是 hard gate；
  `flash-attn`
  direct-runtime correctness
  仍属于后续 workload payoff，
  不能被误写成
  cleanup 完成条件；
  TT-Sim `fp16`
  仍属于 simulator capability boundary，
  不是 cleanup correctness gate
- TT-Metal
  target model
  也只要求
  `Program`
  /
  `MeshWorkload`
  /
  kernel /
  circular-buffer /
  semaphore /
  runtime-arg /
  launch
  这些显式对象与 API；
  它不要求
  更宽的 workload payoff，
  也不要求
  simulator capability breadth。
  因此 task5
  只能验证
  当前 admitted surface
  与显式 materialization boundary，
  不能把
  非 admitted runtime
  或 TT-Sim `fp16`
  重新升级成
  cleanup completion truth
- `memory/general_dev.md`
  / `memory/bugs.md`
  在 task5
  里只负责记录
  长期可复用经验和 bug taxonomy，
  不负责承载阶段状态；
  `git commit` / `git push`
  也只是 repo workflow
  的交付动作，
  不能被写成
  协议完成定义本身

### 3.7 `2026-04-23` Task 5 convergence note

- `tilelang_repo/src`
  /
  `tilelang_repo/tilelang`
  已不再保留
  `blackhole.resource_plan`
  或
  `tl.internal_tt_*`
  定义面；
  task5 source-scan
  已把这两类 surface
  压成零命中
- lowering support
  只剩
  pass-local
  `CollectBlackholeLoweringSupportFacts`
  helper；
  broad
  `blackhole.lowering_requirements`
  / public
  `AnalyzeBlackhole*`
  surface
  不再是 active protocol
- `blackhole.segment_kind`
  仍可作为
  `lower_blackhole_ops.cc`
  内部的
  segment slicing mechanics
  暂存，
  但它必须在 leaf reader /
  runtime 边界前剥离；
  不能再回升成
  cross-pass contract
- admitted runtime gate
  当前只覆盖
  copy / GEMM；
  direct cast consumer
  与
  `fragment_fill -> cast -> publish`
  继续留在
  build/source contract gate，
  不反写
  layered IR
  completion contract
- preclear zero-init GEMM
  一旦 canonicalize 到
  `clear_accum=true`，
  lowering
  也必须同步删除
  紧邻 full-overwrite matmul
  的 selected zero-fill builtin；
  否则等于在
  `ExecutableSpec`
  已收正之后
  又把 runtime
  拖回旧 live-form path

## 4. 长期保留的表示与 transform 纪律

补充约束：

- 只允许显式表示层定义长期协议面
- IR 不是 read-only 观察对象；
  pass 的职责是把当前 stage 的 IR/object
  改写到下一个更具体 stage
- helper 只允许作为同一 `.cc`
  内的局部 visitor / matcher / mutator mechanics
- 如果一个 pass 能从当前 IR 或当前显式对象直接恢复所需信息，
  就必须直接恢复，
  不能先发明 bag 再读 bag
- helper 必须复用已有显式 enum / handle / object identity
- 不能在 helper 里重新发明一套
  `kind / direction / role`
  字符串或平行 enum
- 不能用 `Map<String, Any>`
  充当跨 pass 语义协议

### `Normalized Tile TIR`

长期保留：

- tile op
- `BufferLoad / BufferStore`
- address expr
- region / predicate / loop/domain
- loop-carried / dataflow structure

### `SpatialPlan`

长期保留：

- `ExecutionUnit`
- `DataflowEdge`
- `LayoutSpec`
- `PhasePlan`
- `ValidatedHintSet`

### `TTProgram`

长期保留：

- `TTBlockPlan`
- `TTKernelPlan`
- `TTTransportPlan`
- `TTSyncPlan`
- `TTABIPlan`
- `TTExecutionPlan`

### `ExecutableSpec`

长期保留：

- leaf projection
- runtime/build/codegen 消费视图

## 5. 当前诊断

当前代码的真实问题不是“旧协议太散”，
而是：

- 中间 spatial/dataflow 表示太薄
- 下游被迫补出 fake protocol
- leaf readers 仍在消费 fake protocol

因此 disposition 的执行顺序固定为：

1. 先把 `SpatialPlan`
   重写成 virtual spatial/dataflow representation
2. 再把 `TTProgram`
   收回到 target realization 边界
3. 再切 leaf readers
4. 最后删 fake protocol
