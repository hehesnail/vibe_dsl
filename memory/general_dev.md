# 通用开发模式与当前有效经验

> 当前 Blackhole 后端唯一设计依据: `tasks/dev_design/final_blackhole_backend_redesign.md`
> 本文档只保留长期可复用的工作模式；不承担阶段状态、bug 目录或验证快照职责。

## 1. 文档使用模式

- 当前活动入口固定为：
  - `tasks/dev_design/final_blackhole_backend_redesign.md`
  - `tasks/dev_design/README.md`
  - `tasks/progress.md`
- `tasks/dev_design/archive/` 只当历史记录，不再作为当前任务入口
- 设计边界写在设计文档，阶段状态写在 `progress.md`，`memory/` 只记长期经验
- 活跃设计文档只按
  `Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec`
  这四层显式表示描述边界；
  不把 pass 名、helper、bag、payload、bridge attr
  写成长期协议
- task 文档定义目标合同和完成判据；
  `progress.md` 只记录当前执行看板、active task、blocker、下一步；
  `tasks/dev_design/archive/layered_ir_references.md`
  只做研究输入，不充当活动设计入口

## 2. 构建模式

- C++ 改动后，先确认 `libtilelang.so` 已重编，再跑 pytest
- 当前 `tilelang_repo/CMakeLists.txt` 的 Blackhole 源列表不是“目录自动全量收集”；
  split 新 `.cc` 文件后，必须把它显式接进 `TILE_LANG_BLACKHOLE_SRCS`，
  然后重新执行一次 CMake 并重链 `libtilelang.so`
- 当前顶层 `file(GLOB ...)` 删除/改名 `.cc` 文件后也需要重新执行一次 CMake；
  否则现有 `build.make` 仍会尝试编译已删除源文件
- 同一个 `tilelang_repo/build/` 不要并行跑 `cmake --build` 和 pytest
- `pip install -e .` 不是默认开发路径；更稳的是用 `.pth` 指向本地构建产物
- `3rdparty/` 和 `build/` 不进主仓库

## 3. 验证模式

测试层级固定分三层：

- 结构层：
  - lowered TIR、显式表示 attrs / projection records
- planner 层：
  - `ExecutableSpec`、`KernelSpec`、bindings、CB / core / semaphore 规划结果
- runtime 层：
  - direct path 真执行

经验上：

- 只做 codegen/reference compare 不算 true E2E
- 当前支持面和 fail-fast 边界都应该在更早层被看见，不要全部压到 runtime

## 4. Layered IR / explicit representation 模式

- module-scope truth 放 `IRModule.global_infos`
  或当前层显式对象，
  不要回退到单个 `PrimFunc.attrs`
- unsafe TIR mutation
  必须整体 invalidate
  当前层派生 analysis /
  plan /
  projection truth；
  不要只删单个旧 attr
- cross-pass schema 一律 handle-first；
  字符串只保留
  display / debug / compatibility 角色
- 当前长期边界只按
  `Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec`
  描述；
  pass 名、helper 名、历史文件名
  只能当代码索引
- `Normalized Tile TIR`
  是唯一 semantic body；
  `SpatialPlan`
  负责 target-independent
  execution-unit / dataflow / layout / phase truth；
  `TTProgram`
  负责 target realization；
  `ExecutableSpec`
  只负责 leaf projection /
  runtime-module build contract
- 对 Blackhole 这类 tile-based compute target，
  TT-Metal API 粒度的 tile compute semantics
  必须在
  `Normalized Tile TIR`
  中保留或规范化。
  这包括 matmul / reduce / unary /
  binary / broadcast / copy / pack /
  tilize / untilize
  等通用 leaf API 粒度，
  不是 reduce 或 flash-attn 专项例外。
  不要先 scalar-expand
  再在后段按 workload idiom
  恢复 compute DAG。
- Blackhole preserve 某个 `TileOperator`
  时，这个 op 自身必须实现
  `GetDataflowAccessInfo()`；
  `SpatialPlan` 不应该再依赖
  scalar-expanded `BufferStore` /
  `BufferLoad`
  作为 producer / consumer truth。
  reduce 这类 op 需要显式记录
  source consume、destination produce，
  `clear=false` 时还要记录
  destination consume。
- `TTComputeOpPlan.operation_name`
  和 executable compute ops
  应保持 TT-Metal leaf API 粒度。
  `softmax` /
  `exp2_affine` /
  `row_broadcast_exp2_affine`
  这类 composite/helper 名
  只能是历史债务或调试描述，
  不能成为生产协议。
- `TileComputeDAG`
  这类 selection foundation
  只能作为 pass-local diagnostic /
  covering model。
  Phase A 读图可以通过 FFI diagnostic
  给测试看，
  但不能写回
  `PrimFunc.attrs`
  或
  `TTProgram`
  payload
  变成新的跨 pass truth。
  生产代码也不要在
  `PlanTTKernelABI`
  这类 lowering object
  中持久化 DAG covering cache /
  active decision guard；
  source emission 应只消费当前 selected leaf pattern
  和 typed plan/diagnostic，
  DAG covering object 只能留在 diagnostic /
  test-visible API。
  但 diagnostic-only DAG
  不能算 production payoff；
  如果要保留在 active backlog，
  下一步必须把 fanout /
  materialization decision
  投影成
  `ResourceDemand`
  /
  `ResourcePressureReport`
  或 typed unsupported reason，
  并让 validator /
  admission 实际消费。
- TT resource planning 不属于
  `TileComputeDAG`
  的职责。
  `TileComputeDAG`
  只负责 pass-local tile compute
  legalizer / covering；
  CB ID allocation、
  L1/SRAM pressure、
  core placement、
  buffer distribution、
  NoC / multicast scheduling
  都应进入
  `TTProgram`
  /
  `ExecutableSpec`
  的 typed resource planning /
  admission surface。
- CB allocation 更像寄存器分配：
  第一版优先用 live-interval / linear-scan
  模型和 arch-aware CB limits，
  不要先上全局 graph coloring /
  ILP。
  L1/SRAM 先做 pressure admission
  和 typed diagnostics，
  不替代 TT-Metal allocator
  的 physical address assignment。
- CB page 需求和 L1 buffer placement 要分开：
  `page_size`
  /
  `num_pages`
  由 op lowering、
  data movement、
  exact-CB publication、
  materialization lifetime
  这类 protocol 决定，
  进入
  `CBRequirement`
  /
  `TTCBPlan`；
  L1 admission 只消费 typed CB plan
  汇总 CB-backed L1 bytes
  和 allocator-managed L1 bytes。
  buffer placement /
  address ABI
  不应从 buffer 名、
  source hook
  或 runtime reader fallback
  重新推 CB 深度。
- TileLang/GPU 风格的
  `T.Kernel(grid_x, grid_y)`
  是 logical work item 域，
  Blackhole
  `TTCoreGroup.physical_cores`
  是实际常驻 worker，
  `work_packets`
  表达 logical work id
  到 physical worker
  的 temporal 映射。
  当 logical block 数超过 physical core 数时，
  每个 worker 在自己的 packet 上循环执行，
  per-worker L1 /
  CB scratch
  随 temporal work item
  复用；
  不能按 logical block 数复制 resident L1 /
  CB allocation。
- GPU-style
  `alloc_shared((tile_m, tile_n))`
  在 Blackhole 后端中应先解释为
  per-worker、
  per-work-item
  的 L1 /
  CB-backed scratch shape。
  这个 shape 可能远小于 Blackhole worker L1，
  但这不能被包装成完整 tensor sharding。
  TT-Metal / TTNN 的真实 sharding owner 是 caller /
  model /
  op contract 提供的
  `MemoryConfig + ShardSpec / NdShardSpec`；
  program factory 和 runtime 消费这些配置。
  TileLang 当前的
  `TTBufferDistributionPlan`
  只是 low-level buffer placement /
  address ABI。
  完整 sharding 必须另外表达 user / DSL intent、
  op-level placement contract、
  producer/consumer placement conflict、
  以及 explicit reshard /
  layout conversion plan。
  当前设计名分别是
  `TensorPlacementIntent`、
  `TTTensorMemoryConfigPlan`、
  `TTOpShardingContract`、
  `TTPlacementResolutionPlan`
  和
  `TTReshardPlan`。
  Frontend 上不要把
  `T.annotate_layout`、
  `scope`
  或
  `T.Kernel`
  解释成 sharding API：
  `annotate_layout`
  只表达 buffer index layout /
  local transform，
  `scope`
  只表达 storage class，
  `T.Kernel`
  只表达 logical work-item grid。
  真实用户入口应是 TTNN 对齐的
  `T.MemoryConfig`
  /
  `T.ShardSpec`
  /
  `T.NDShardSpec`
  和
  `T.annotate_memory_config`
  这类显式 memory-config intent，
  再 lowering 到
  `SpatialPlan.TensorPlacementIntent`。
  但 baseline correctness
  必须尊重前端形状并做 capacity gate。
  想利用更多 L1
  要显式引入 TT-specific retile /
  work-coarsening plan，
  同时更新 logical work mapping
  和 source-region /
  core-local address mapping；
  buffer placement
  不能静默改大 shared tile shape。
- 当前 sharded tensor 讨论里的关键区分：
  DRAM/global tensor
  是完整 logical source，
  L1 sharded tensor
  是绑定到 core group
  的 working view /
  materialization。
  当前实现里的
  `TTBufferDistributionPlan`
  已拆开
  `shard_grid_shape`、
  `sharding_strategy`、
  per-core data
  `shard_shape`、
  source buffer /
  source region binding
  和 logical-index
  到 core-local address mapping。
  不要把 core-grid shape
  写进
  `shard_shape`
  后再让 leaf reader
  猜它到底是 grid
  还是 data shard。
  TT-Metal `ShardSpec`
  的核心拆分是
  `grid`
  /
  per-core `shape`
  /
  `orientation`；
  `orientation`
  只能是
  `row_major`
  /
  `col_major`。
  `block`
  /
  `height`
  /
  `width`
  属于 sharded memory-layout strategy，
  不能写进 `shard_orientation`。
  source-region binding 是 all-or-none：
  只有确实从 DRAM/global buffer
  materialize 的 L1 view
  才设置
  `source_buffer` /
  `source_region_kind` /
  `source_region_shape`；
  纯 worker-local sharded scratch
  不应伪造 source binding。
- Blackhole core placement
  不能长期硬编码 worker grid。
  规划应从
  `TTHardwareModel`
  和 TT-Metal / UMD
  的 logical / translated / NoC
  coordinate 事实出发；
  logical coordinates
  是 harvesting-safe baseline，
  NoC-proximity / multicast optimization
  属于后续 scoring / scheduling lane。
- tile compute legalizer
  要同时卡住 producer 和 validator：
  当前选择器记录
  `TTComputeOpPlan`
  前先跑 legality，
  `ValidateTTProgram`
  也要重跑 legality，
  这样 synthetic /
  corrupted
  `operation_name`
  不会穿过 projection。
- 任何需要跨 pass 保留、
  且不能从当前层重新推出的事实，
  都应进入当前层显式对象；
  不要继续停在 bag / payload / seed /
  helper wrapper
- `BuildTTProgram`
  只应做 staged aggregation /
  completeness check /
  cleanup；
  `MaterializeBlackholeExecutable`
  是唯一 canonical writer
- build / codegen / runtime /
  `BlackholeModule`
  只应读取
  `tl.blackhole_executable`
  或解析后的
  `ExecutableSpec`；
  不要回读
  `TTProgram`
  或 legacy lowering attrs
- leaf-side host/codegen buffer binding
  必须以
  `ExecutableSpec`
  formal buffer identity
  和 runtime/common runtime arg
  explicit `buffer`
  role schema
  exact match
  为准。
  不要用
  `PackedArgs`
  顺序、
  `_handle`
  suffix、
  runtime arg kind
  或
  compile-time ABI
  `spec.name`
  fallback
  恢复 buffer 身份；
  缺失或不一致应在 codegen /
  direct-runtime schema
  处 fail-close
- per-work runtime args
  需要用 typed descriptor
  表达：
  `arg_identity`
  绑定具体 runtime arg，
  `descriptor_kind`
  描述 tile/start/count/stride
  这类 per-work 角色，
  `value_source`
  描述来自 work-linear-id /
  logical block /
  compute-op dims /
  constant
  的值来源。
  codegen/runtime
  不应按
  `a_tile_start_id`
  /
  `b_tile_start_id`
  /
  `output_tile_start_id`
  的名字优先级恢复 block axis
- `tl.blackhole_logical_buffer_tile_bridge_specs`
  与
  `buffer_tile_bridge_specs`
  已退出 active chain；
  logical tile layout
  只从
  `SpatialPlan.LayoutSpec`
  进入
  `TTBufferDistributionPlan`
  和 executable
  `buffer_distribution_plans`
- `TTProgram.payload`
  不能再承载 bridge /
  contract-family
  compatibility fallback。
  保留 payload 的 plan
  只能写局部 realization /
  admission metadata；
  一旦字段成为 leaf reader truth，
  就要提升为 typed slice /
  typed executable schema
- 扩 `TTProgram` /
  `TTCBPlan`
  这类显式对象时，
  要同步更新：
  - object schema / reflection
  - producer pass
  - executable projection
  - validator
  - runtime metadata / host metadata copy
  - Python 侧 typed rebuild helper
  - Python 侧回归断言；
  否则很容易出现
  C++ 已经 typed 化，
  但 leaf reader /
  测试
  还在读旧 payload
- 给 `TTProgram`
  增加跨 leaf 的 owner truth
  时，
  不能只写 typed slice；
  同一轮必须把
  schema / constructor callsites /
  planner producer /
  validator /
  `tt_program_projection` /
  `ExecutableSpec` parser /
  `BlackholeModule` host copy /
  Python test helper
  一起对齐。
  缺其中任何一层，
  都会把新 truth
  变成“C++ pass 看得见、leaf/runtime 看不见”的半切换
- `direct_runtime_unsupported_reasons`
  只能由
  `ExecutableSpec`
  leaf metadata
  推出，
  不能读回
  `TTProgram.payload`
  或 TIR body
  做 semantic recovery。
  gate 也要按当前 admitted support surface
  精确收窄；
  例如 single-contract
  `thread_distributed + cb_republish`
  可以 gate，
  但不应污染
  flash-attn
  multi-contract
  compile/source baseline
- direct runtime
  是 leaf execution backend
  的 admitted subset，
  不是 Blackhole
  codegen/export
  的能力边界；
  runtime 工作要拆成
  TT-Metal
  `Program / MeshWorkload / MeshBuffer`
  emission capability、
  direct-runtime backend admission、
  硬件/TT-Sim
  验证 lane
  三件事。
  当前 direct runtime
  若只支持
  unit mesh /
  replicated buffer /
  interleaved DRAM，
  只能在 backend admission
  中 fail-close，
  不能反向裁掉
  `TTProgram`
  的 mesh /
  sharded buffer /
  fabric
  表示。
- exact TT-Metal builtin 选择
  必须发生在 anchored sub-TIR
  仍保留 tile-op / layout / load-store /
  address expr 的边界；
  不要等 compute 语义塌缩后
  再靠 late matcher / bridge attr
  恢复
- materialization admission
  只能从当前 IR
  和 typed
  `TTMaterializationPlan`
  /
  `ExecutableSpec`
  推出；
  例如
  `SelectBlackholeTTMetalBuiltins`
  已把 fragment fill
  规范化成
  `tl.blackhole.fill_fragment`
  后，
  `PlanTTCompute`
  必须读取这个当前 IR builtin，
  不能依赖上一 pass
  的原始 loop matcher
  局部状态
- `SpatialPlan`
  的 logical live-value
  应从当前 TIR
  buffer metadata
  和 dataflow edge
  派生，
  并作为
  `LiveValue` /
  `LiveValueEdge` /
  `MaterializationBoundary`
  进入 IR；
  跨 phase /
  execution-unit
  的 producer-consumer truth
  不应留给
  `PlanTT*`
  或 leaf reader
  从 body order /
  buffer 名
  恢复
- live-form / materialization planning
  要从
  `SpatialPlan`
  的 indexed
  `LiveValueEdge`
  /
  `MaterializationBoundary`
  记录查询 owner truth。
  pass-local fact 可以携带这些 index
  作为当前 IR 派生证据，
  但 subject 名只能用于核对当前 TIR buffer
  是否匹配 indexed source/target，
  不能重新成为 latest-live-value map。
- live-form graph solver
  里 self carry boundary
  是 lifetime / recurrence evidence；
  不应当成物理 form transfer
  覆盖 source live value。
  物理转移只发生在 source/target live value index
  不同的 materialization boundary 上。
- source/codegen regression
  要区分 logical element count
  和 thread-distributed
  physical local extent；
  例如 flash-attn
  的 `acc_s`
  logical 形状可以是 full matrix，
  但 device local array
  只保存 producer thread lane slice，
  full logical materialization
  应在 CB publication /
  materialization boundary
  上断言
- constant-fill
  这类局部 analysis fact
  只能在当前 IR
  def/write 边界内使用；
  matmul / merge / add /
  reduction / scalar update /
  cast 等后续 producer
  写同一 buffer
  时必须失效，
  否则 preclear fill
  会被误当成后续 cast source truth
- post-merge GEMM
  direct cast consumer
  可以通过
  `TTMaterializationPlan(publication_protocol=pack_tile)`
  admitted，
  但前提是当前 IR
  还能证明 accumulator live-in
  是 zero-preclear；
  merge sequence
  必须整体非 mailbox：
  partials CB
  copy 到 DST register
  后再
  `pack_tile`
  发布 target CB，
  不能只把最终 cast publication
  改成 `pack_tile`
  而保留 accumulator reload
  mailbox helper
- materialized output
  的 host copy
  要优先消费
  `BufferMaterializationSpec.live_form_kind`
  /
  materialization spec；
  不要把外部 output
  机械按 GEMM accumulator
  dtype 或 compute contract
  校验
- 对 builtin-surface / residue 回归，
  测试应收集实际 TIR `Call`
  的 op 名；
  不要用字符串子串匹配
- selector 在 rewrite 前后若必须携带 bridge /
  materialization 事实，
  只能 seed 稳定字段；
  带顺序索引的结构
  必须从当前 IR 重新构建
- 对 tileop statement
  的通用 read/write
  恢复，
  先读
  `tl.region`
  的
  `access_mask`
  （`r/w/rw`）；
  不要在 generic pass
  里按
  `gemm_py`
  / 参数位次
  恢复写边。
  另外，
  visitor
  遇到
  `tl.region`
  时
  不应继续递归到
  内部 `BufferLoad`，
  否则 write-only region
  会被误记成 read
- tileop typed contract
  和 region contract
  要分工：
  `tl.region access_mask`
  负责通用
  read/write edge，
  `TileOperator::GetDataflowAccessInfo()`
  只负责额外的 typed
  compute-consume /
  planner contract；
  不要把两者重新混成
  op-name matcher
- 当前 IR 上重建 logical shape registry 时，
  同 data identity 的 flattened alias
  可能会把原始高维 logical shape
  覆盖成 1-D static shape；
  registry 必须保留
  更高优先级 /
  更高维度的 logical shape，
  不能让 flatten alias
  反向污染 logical truth
- staged copy 在
  `FlattenBuffer / VectorizeLoop`
  之后如果 shared staging buffer
  已经变成 1-D，
  不要回退到新的 copy annotation；
  正确做法是绑定
  transport var
  的静态 extent，
  直接从当前 global access
  的 row/col coverage
  推出 shared matrix shape
- 兼容 attr 的删除顺序固定为：
  先移走 semantic consumer，
  再移走 lowering consumer，
  最后删 attr 本身
- 旧 pass/link 清理要三层一起删：
  Python wrapper、
  FFI global registration、
  测试 helper / fallback；
  否则旧入口会继续漂回 active path
- 在这个仓库的主线 task 上，
  不要套用
  “先迁 owner truth，
   后删兼容壳”
  的通用重构习惯；
  除 task 文档明确写出的
  narrow exception 外，
  旧 `wrapper / facts / bag / payload / public surface`
  还活着，
  就应视为该 task
  仍未按原则收口
- 把 analysis + builder
  两段式 cutover 成
  direct builder 时，
  要四件事一起做：
  - 主链改成单入口 builder
  - 删除 facts attr / facts object
  - 删除 Python / FFI 旧入口
  - 把测试改成显式断言旧入口不存在；
  只改 pass 顺序、
  不删 facts object /
  facts attr
  仍然是在保留旧协议面
- 如果 selector 在 pre-planner rewrite 里创建了 exact temporary CB，
  就必须同时把这些 temporary requirement
  通过
  `blackhole.cb_requirements`
  持久化，
  并在
  `PlanTTCompute`
  入口重新装回 requirement table。
  否则
  `PlanTTCBAlloc`
  只能看到 IR 里残留的 `requirement_index`
  却没有对应 mapping，
  最终会报
  `Missing final cb_id for requirement_index=*`
- fragment-cast publish 的强制回写只能对
  `blackhole.acc`
  且确实带 materialization contract 的目标生效；
  像
  `O_shared_local_cast`
  这类短生命周期 local temp
  仍要保持 local，
  等单独的 tilize / pack 步骤处理
- `compute_epilogue_ops`
  不再是正确的调试或验证入口；
  debug / source contract
  若要插 waypoint
  或做 source 断言，
  应该绑定
  generic op-kind /
  phase-kind /
  structural marker，
  不要绑定
  `scores_max / acc_o / logsum`
  这类 workload-private
  buffer 名
  现在应直接看 builtin 调用序列
  与
  `buffer_tile_bridge_specs`
- `PlanTTTransport` 负责 `TensorAccessor / CB / NoC / semaphore / multicast`
  这组 data movement protocol，
  `PlanTTCompute` 负责 TT-Metal compute family；
  不要再引入 `row_* / broadcast_sources / index map / access pattern`
  这类 side contract 当长期 owner truth
- seed / manifest / witness / program 分层存放
- 显式 plan / projection truth 一旦扩层级，
  unsafe TIR mutation 侧也要同步 strip
  新 analysis facts / plan attr；
  只清旧层会留下 stale truth
- intermediate typed plan
  只要进入 pass 链，
  就要和最终显式表示 truth
  一起纳入 invalidation；
  不能只删一层旧字段
- workload noun 不进入长期 semantic schema
- evidence carrier 不是 truth owner
- 兼容 attr 的删除顺序固定为：
  先移走 semantic consumer，再移走 lowering consumer，最后删 attr 本身
- 当新的 typed truth 需要 bridge 回 legacy projection 时，必须显式区分
  “节点原生拥有该字段” 和 “只是借用 top-level fallback”；
  否则 materializer 会把 fallback 意外下沉成 per-node 真相，
  破坏旧测试和 reader contract
- 旧 pass/link 清理要三层一起删：
  Python wrapper、FFI global registration、测试 helper/fallback；
  只删其中一层会留下可达旧入口，active path 很快又会漂回去。
  做“入口已经不存在”的回归时，优先显式断言查询抛错，
  不要把“允许缺失”本身再写成一层兼容语义
- function-level target contract
  一旦进入 runtime/codegen 正式消费面，就应提升进 typed
  `TTProgram` slice
  和 typed
  `ExecutableSpec`
  schema；
  `gemm_contract / compute_contract`
  family
  已删除，
  不要再以 compatibility fallback
  形式恢复
- leaf-only build/codegen gate data（如
  `unsupported_compute_ops`）
  也一样：
  需要先进入 typed
  `TTProgram`
  object，
  再由
  `MaterializeBlackholeExecutable`
  投影到 typed
  `tl.blackhole_executable`
  leaf schema；
  leaf reader 不应再直接摸
  `blackhole.lowering_requirements`
- 这类 function-level contract 若先在 device `ExecutableSpec` 上补充，
  host entry metadata 也必须同步拷回；否则 Python/runtime gate 仍会看见过时视图，
  以为 kernel 没有 unsupported reason
- 一旦原始 device func 已切到 typed target truth，
  shared projection helper 就不能再同时承担
  “`TTProgram` reader” 和 “legacy attr fallback” 两种职责；
  必须拆成 `TTProgram`-only reader 与本地 materialization helper，
  否则会把单一真源再次偷偷变成双真源
- synthetic segment / internal kernel emission 也应遵守同一规则：
  在 `ExecutableSpec` leaf cutover 之后，
  内部 leaf func
  只应重建最小单-segment `tl.blackhole_executable` 视图；
  不要再回挂 `TTProgram`
  或重新降回局部 `blackhole.*` attrs
- per-work/access truth 一旦 formalize 成 `per_work_arg_specs`，
  就要先 canonicalize 成 kernel-local `TTKernel / ExecutableSpec` contract；
  codegen/runtime 只能解释 typed `descriptor_kind` /
  `value_source`，
  不能再按 arg kind 名字推语义，
  也不能保留旧 `value_kind`
  作为兼容真源
- `per_work_arg_specs`
  一旦完成 kernel-local canonicalization，
  就不要再保留 top-level `TTProgram.payload`
  版本给 reader 当 fallback；
  否则 single-kernel/multi-kernel 两条 host path
  会重新出现“segment truth 缺了但 top-level bag 还能兜住”的双真源
- leaf reader / codegen
  一旦切到 typed executable projection，
  不要再保留默认恢复逻辑：
  缺 `cb_configs`
  时不要生成 `default_cb`，
  缺 `host_buffer`
  时不要用 device buffer 代替，
  accessor 不要读旧 `slot`，
  GEMM 不要从 `M/N/K`
  反推 tile/block/subblock，
  segment 不要默认
  `fused_dataflow/brisc`。
  这些都应在 projection/schema
  缺失处 fail-close。
- 只做 device `global_symbol` 对齐时，必须保留优化后的 device `PrimFunc`
  和对应 `global_infos`；
  不能把 source func 重新 `with_attr("global_symbol", ...)` 后塞回去。
  否则会把 Blackhole lowering 后的真实 device body回退成旧 body，
  重新暴露 free loop var（例如 `tile_row`）这类已经在优化版里消失的问题
- Python 侧若需要做 typed plan /
  explicit representation mutation regression，
  优先通过 `tl.TT*` constructor 直接重建
  `TTProgram / TTKernel / TTCoreGroup / TTABIPlan / TTSemaphorePlan`
  并重新跑 `ValidateTTProgram`；
  不要先改 bridge attrs 再依赖 translator 刷新 typed truth
- `BuildTTProgram` 不应再经由
  `tl.tt_kernel_seeds / tl.tt_abi_plans / tl.tt_cb_plans /
  tl.tt_core_groups / tl.tt_program_payload`
  这组 bridge attrs 传递 target truth；
  planner object 一旦可直接聚合成 `TTProgram`，
  中间 attr 就应该停产，helper/test 也不能再把它们当作回退面
- 一旦
  `PlanTTBlocks / PlanTTCompute / PlanTTTransport /
   PlanTTSync / PlanTTABI / PlanTTExecution`
  都开始提前发布 target truth，
  过渡面应直接增量写 staged
  `tl.tt_program`，
  再由
  `BuildTTProgram`
  只做 completeness validation + final aggregation；
  不要再引入
  `tl.internal_tt_*`
  这种第二层 bridge bag
- canonical `Phase B`
  不再负责发布
  `blackhole.work_decomposition /
   blackhole.compute_regions /
   blackhole.pipeline_stages`
  这组三件 analysis attr。
  如果后续 pass 仍需这些 facts，
  要么本地重跑 helper analysis，
  要么只桥接真正缺的最小 schema；
  不能把“analysis 结果方便拿”
  再写回成新的 pass-to-pass 协议
- 如果 optimized/helper path
  在 destructive pass 之后
  仍需要 logical tile layout truth，
  不要跳过
  `BuildSpatialPlan`，
  也不要恢复 bridge attr。
  当前有效做法是：
  在 pre-opt 阶段保留 typed
  `SpatialPlan.LayoutSpec`
  作为 merge source，
  然后重建 optimized
  `SpatialPlan`
  并只把缺失的 typed layout fields
  按 subject
  合并回来；
  这样不会丢优化后 execution units /
  dataflow truth
- active Blackhole path 不再保留
  独立 semantic mirror bridge；
  凡是能从 `Normalized Tile TIR + SpatialPlan +
  当前过渡残留`
  稳定得到的信息，就必须直接从这些当前 owner truth / transition residue
  读取，并持续把这批残留往真正的 `PlanTT*` owner 上迁移。
  若仍然不够，优先扩 TIR/schema，不要再造一层 semantic mirror
- TT fast path / short path 的判定必须看
  `SpatialPlan` 自身的执行单元 / 数据流边 / phase 形状和
  当前 target plan 能否直接承接，
  不能只看零散 segment kind 或历史 semantic 角色；
  否则 `flash-attn` 这类多闭包程序会被错误压成简单 GEMM / compute fast path
- `BlackholeDeviceResourceCanonicalization`
  不能只回写 TIR body。
  一旦它把 `local.fragment / local` canonicalize 成 `blackhole.acc`
  或把 shared canonicalize 成 `blackhole.cb.*`，
  就必须同步改写
  过渡 attrs 与 projection records
  里对应 contract 的 `scope`；
  否则 planning / codegen 会命中
  “IR 已经切到新资源类，projection/typed record 还停在旧 scope” 的双真源裂缝
- transport / layout / logical distribution 如果已经要进入正式 target truth，
  直接进入 `TTTransportPlan / TTABIPlan`；
  不要再额外造 `buffer_distribution_contract`
  一类中间 side contract 承接这类信息

## 5. Schema / ABI 模式

- compile-time、common-runtime、per-work 三层 ABI 必须严格分开
- runtime 参数布局必须显式、可验证；不要依赖默认顺序、默认名字或 host 猜位置
- `common_runtime_args` 只放共享 metadata；per-work / per-core 值单独下发
- work descriptor 要用角色化字段，不用单值默认去反推整套语义
- 64-bit 地址需要明确拆分 / 重组规则

对象分组与派生规则：

- arg identity 必须由 lowering / split 正式产出
- dedup key 一律用 `identity + ":" + kind`
- 多个字段共同表达一个对象时，应尽快上提成 schema object
- schema-only 路径一旦成立，派生物也必须能从 schema 单独重建
- 未正式支持的 ABI / accessor / transport 组合，要 build-time fail-fast
- 不保留默认 ABI、默认 core、默认 packet 这类补洞
- multi-GEMM / staged-copy reader 的 transpose truth
  不能只留在 compute contract；如果 host materialization 也要配合，
  就必须显式进 accessor/materialization schema
  （例如 `transpose_2d`），并由 host tilize / readback 真正执行
- materialization host binding
  必须由
  `TTMaterializationPlan.host_buffer`
  和 accessor /
  compile-time ABI
  中的
  `transport_page_size` /
  `host_axis_order` /
  `transpose_2d`
  显式表达；
  不要从
  `_local`
  suffix、
  single-output、
  CB role、
  first-CB、
  固定 page size
  或 tensor shape/work split
  恢复这些语义
- 如果 planner 已正式产出 `core_plan.work_packets` 且允许
  `work_count > 1`，direct runtime 不能再把 packet 扁平成
  “单波次 one-work-per-core” 假设；对还没把 `work_count`
  下沉成 device-side loop contract 的 executable，至少要按 packet truth
  做 repeated launch / wave scheduling，避免同一 core 的 runtime args 被后写覆盖
- 一旦 `ExecutableSpec / Leaf Reader Cutover` 成立，
  原始 device build/codegen/runtime 输入就应硬要求
  `tl.blackhole_executable`；
  `tl.tt_program`
  和
  `blackhole.lowering_requirements`
  只能停留在上游 owner / planner / writer 边界，
  不应再进入 leaf reader

## 6. analysis / lowering / planner / codegen 模式

- analysis 产出事实
- lowering 消费事实并改写 IR
- planner 只消费显式 requirement schema
- codegen 只打印已确定 contract
- 像
  `clear_accum=false`
  这种 op-level flag
  不能直接当最终 lowering contract；
  是否真的需要
  accumulator merge /
  live-form bridge
  要由
  `TIR execution order + recurrence/live consumer facts`
  共同决定。
  fresh fragment /
  preclear zero-init
  这类 case
  应优先 canonicalize 到主链，
  不要默认续到旧 merge path

稳定做法：

- 如果 pass 只把规划结果写到 attrs 而不回写 IR body，就会制造两套真源；
  优先让 pass 同时完成 IR 回写
- 如果当前 layered 主链里一个 lowering 同时做 domain synthesis、task formation、
  ordering 和 final materialization，应优先拆成
  `Analyze... -> Analyze... -> Materialize...` 的 pass 链，让 analysis facts
  先以 typed plan 落地，再由 materialize pass 组装最终显式表示
- 当 canonical pass 命名切换完成后，
  旧 TT target probe / validator / executable materializer
  这类命名应直接删除；
  不要再保留 compatibility shell、probe 或测试入口
- unsupported subset gate 应在所有后端出口共享
- gate 应按具体 contract / op family 报错，不要长期用黑盒总括词
- 需要的信息优先从 typed IR / schema 拿；拿不到就扩 IR / schema，
  不要把猜测沉淀成长期协议
- 如果某条 runtime/codegen 路径持续需要 gate
  才能避免错跑，先回头判断它是否其实在暴露
  上游 `TT transport / compute / ABI` owner truth 的缺口；
  不要把“后段还没实现协议执行”误判成第一 blocker

## 7. Resource / storage 模式

- Blackhole `shared` 的主映射是 CB / L1 资源，不是普通 C 数组
- `local` 是中间语义桶；进入后端后要继续分流成：
  - 真正的小标量临时
  - fragment / accumulator
  - 显式 dataflow transfer
- 一旦 residual `local` 已经明显表示“结果写回 CB 的桥接语义”，
  就应尽快 lower 成正式 builtin / direction
- planner 的正式输入必须是上游显式 schema；
  不要一边吃 schema，一边从 IR 形态做 fallback inference
- target 资源模型与 generic backend 不一致时，优先扩 IR / schema，
  不要给后段 pass 打豁免

## 8. TT-Sim / TT-Metal 模式

统一入口：

```bash
source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh
export TILELANG_HOME=<当前 checkout 或 worktree>/tilelang_repo
cd <当前 checkout 或 worktree>/tilelang_repo
```

稳定经验：

- `setup_tt_sim.sh` 和测试命令必须在同一个 shell
- `TT_METAL_SLOW_DISPATCH_MODE=1` 对 TT-Sim 很关键
- direct path kernel 临时目录必须每次执行唯一化
- 优先消费 TT-Metal local install tree，不要把 `.cpmcache` 整片塞进 include path
- Blackhole runtime/direct-runtime regression 默认统一用 `bf16` 输入；
  不要再把 `fp16` 当成 TT-Sim 上的正式 runtime baseline。
- Admitted runtime support boundary 是动态状态；
  只看 `tasks/progress.md`，
  不在 `memory/` 里维护第二份支持矩阵。
- 扩更宽 direct cast /
  live-in materialization /
  multi-page event
  时，必须先扩 explicit live-form /
  materialization protocol，
  不要写 runtime-only patch。
- 对 `flash-attn` / multi-op compute kernel，
  不要再靠后段从 `SeqStmt`、builtin 序列或 buffer 形态
  恢复 producer-consumer / republish / reduce / broadcast 语义；
  这些都必须在
  `PlanTTTransport + PlanTTCompute`
  能直接消费的 owner truth 上落地
- `TTProgram` 的 physical live-form /
  materialization /
  consumer-binding plan
  如果承接 `SpatialPlan`
  logical relation，
  必须用 typed refs
  指回
  `LiveValue` /
  `MaterializationBoundary` /
  `LiveValueEdge`
  并在 `ValidateTTProgram`
  对照 `SpatialPlan` fail-close；
  不能只把这些名字塞进 payload
  或让 leaf reader 再从 body order 恢复
- Data movement 与 compute 的边界要保持明确：
  `TensorAccessor / CB / NoC / semaphore`
  属于 transport protocol；
  `matmul / eltwise / reduce / sfpu / pack`
  属于 compute family；
  不要再发明介于两者之间的 side contract
- TT-Sim `float16` 路径是否可用要和 target contract 问题分开判断；
  如果 small bf16 correctness 已过、但大 shape `float16` 命中
  `UntestedFunctionality: tensix_execute_unpacr: fp16`，
  优先视为 simulator 能力边界，而不是先回退刚验证过的 target contract 修复
- active 设计/进度文档里的状态职责要锁紧：
  总体状态 / blocker / 下一步
  统一只写在 `tasks/progress.md`；
  task / cleanup 分文件
  只写各自 residue 的 owner boundary /
  required end-state /
  verification contract，
  不要把局部切片完成
  写成总体路线完成
- 对当前 Blackhole rewrite，
  “第一性原理目标”不能被缩写成
  单个 workload payoff
  或单个 owner pass cut-in；
  要同时检查：
  - mapping 边界是否放回 anchored sub-TIR
  - compute semantics
    是否由 `PlanTTCompute`
    收口 builtin family / operand-result binding /
    tile-reg protocol
  - memory-access semantics
    是否由 `PlanTTTransport + PlanTTABI`
    收口 buffer / accessor / CB / runtime-arg truth
  - communication semantics
    是否由 `PlanTTTransport + PlanTTSync + PlanTTExecution`
    收口 routing / multicast / completion truth
  - truth 是否回到
    `TIR / SpatialPlan / TTProgram / ExecutableSpec`
    各自边界
  - runtime / codegen
    是否已经退回 reader/gate，
    不再补 planning truth
- 清理旧 target 链时要从外往里收：
  先删 projection / side-channel，
  再删最终 `TTProgram / ExecutableSpec` 输出上的 seed bridge attr，
  再删 active path 上的 `blackhole.*` compatibility attr synthesis，
  最后再把 canonical bundle 上的显式 legacy pass 链内收到单一入口；
  canonical `LowerToBlackholeTTProgram` 产物应只保留 `tl.tt_program`，
  不应再把 `tl.tt_kernel_seeds / tl.tt_abi_plans / tl.tt_cb_plans /
  tl.tt_core_groups / tl.tt_program_payload` 当作稳定输出面
- 与之对应的 regression 也要同步收口：
  probe/test 只能验证当前 admitted 的 canonical 输出，
  不要把中间 bridge attrs 或未 admitted 的 `flash-attn` 前提
  固化成长期绿测
- multi-segment `TTProgram`
  的 per-work ABI truth 必须留在 segment 本地并能经
  `tt_program_projection::EncodeSegmentPlan`
  完整 round-trip；
  top-level aggregate/per-work payload
  不能再给 reader/writer multi-kernel path 充当 fallback 真源
- `flash-attn` 这类 optimized path 如果会在后续 pass 里
  canonicalize 资源或折叠 compute region，
  需要显式保留一份 pre-canonical logical compute-region truth
  （例如一份独立 typed explicit fact），
  让 transition requirement reader 还能恢复
  row-state / grouped-row / fragment logical shape；
  否则后段会被迫重新猜 shape 或回退到旧 contract

稳定 host-side 抽象：

- `Program`
- `CreateCircularBuffer`
- `CreateKernel` / `CreateKernelFromString`
- `SetRuntimeArgs`

稳定同步 / 资源模式：

- CB 既需要地址共享，也需要同步原语
- semaphore 在 host/runtime 侧的正式对象是 id，不是地址
- remote worker core 的 logical -> NOC 坐标转换由 host materialize
- communication builtin 只能消费显式 schema truth：
  `get_semaphore` 必须命中 planned semaphore id /
  bound `semaphore_id_u32`，
  remote semaphore route 必须命中
  `logical_core_noc_x/y + remote_core_descriptors`；
  不要再让后段从 literal 坐标、
  裸地址或 builtin 序列补协议
- leaf compute schema 要按 generic op 数组扩展：
  `KernelSpec.compute_ops`
  是长期方向，
  entry 用 `kind` 分派 family-specific fields；
  GEMM 可以有
  `M/N/K`、operand buffer、transpose、dtype、work-decomposition
  字段，
  但它只是 `kind=gemm` entry，
  不能把 `gemm_compute`
  升成所有 compute kernel 的 public field。
  后续 eltwise / reduction / SFPU / pack-materialization
  都应在同一数组下新增 typed entry，
  runtime 只消费 admitted `kind`
  并对未 admitted subset 走 explicit unsupported gate
- public protocol audit
  不能只盯某个 compute 指令名：
  凡是被 projection /
  `ExecutableSpec` parser /
  runtime /
  codegen /
  wrapper
  读取的字段，
  都已经是 public schema。
  payload seed、
  runtime arg order、
  PackedArgs position、
  handle suffix、
  `_local`
  suffix、
  single-output fallback、
  arg-kind priority、
  shape-based axis heuristic
  都不能承担 owner truth；
  正确收口是 typed
  `TTProgram`
  /
  `ExecutableSpec`
  fields
  加 validator
  fail-close，
  不是把新 workload
  继续挂到旧 fallback
  上
- 删除 public
  contract-family
  fields
  时要同时检查 admission gate
  是否被旧字段遮住：
  P0.1
  删除
  `multi_compute_contracts`
  runtime metadata
  后，
  flash-attn
  暴露出真实 typed
  materialization
  unsupported reason
  `thread-distributed cb_republish materialization`。
  测试应断言新的 typed
  gate
  和 public field
  缺席，
  不应为了维持旧
  "no unsupported reasons"
  绿测而重新保留 compatibility
  metadata
- GEMM
  compute operand binding
  不能从 reader/writer
  runtime arg
  顺序恢复。
  正确 schema
  是在
  `KernelSpec.compute_ops`
  entry
  内写 typed
  `operand_bindings`：
  `role`
  表达 op-local
  operand 角色，
  `buffer`
  表达 compute-side
  operand buffer，
  `host_buffer`
  表达 runtime tensor
  binding。
  对 staged copy
  场景，
  host mapping
  应从结构化 copy
  关系记录，
  不能通过
  `input_buffer_addr32`
  或
  `output_buffer_addr32`
  的出现顺序猜
- executable projection writer
  不能以 child typed-node
  的
  `payload`
  为 seed
  再覆盖字段；
  应 fresh-map 构造 leaf record，
  对尚未上提成 typed field
  但必须 leaf-visible 的字段
  使用显式 allowlist。
  回归测试要直接检查
  `tl.blackhole_executable`
  attr，
  因为最终 runtime metadata parser
  可能已经丢掉未知字段，
  掩盖 projection writer
  的 payload 泄漏
- SpatialPlan materialization boundary
  需要同时携带 source / target
  logical live value typed refs。
  TTProgram planner
  消费 materialization 时应按
  source -> target
  boundary identity
  取 ref，
  不能只按 source subject
  first-match，
  否则同一 buffer subject
  在不同 producer/consumer relation
  下会选错 live value。
- TT-Metal mesh /
  buffer distribution
  要作为 `TTProgram`
  owner truth：
  `TTMeshPlan`
  表达 physical mesh /
  device range，
  `TTBufferDistributionPlan`
  表达 per-buffer
  replicated/sharded distribution
  和 DRAM/L1 memory space。
  direct runtime 当前只支持
  unit mesh /
  replicated subset
  只能留在 leaf admission，
  不能反向限制 codegen /
  executable projection
  的 schema。
- Compute kind 不能再用
  `gemm_contract`
  / `compute_contract`
  这类 family payload
  当 owner truth。
  `TTComputeOpPlan`
  是 `TTProgram`
  层的 typed compute slice；
  executable
  `KernelSpec.compute_ops`
  应优先由它投影。
  GEMM 只是
  `kind=gemm`
  的 entry，
  operand role /
  host buffer /
  compute-side buffer
  必须来自 typed
  `operand_bindings`，
  `TTKernel.payload["compute_ops"]`
  不应存在；
  projection 只能由
  `TTComputeOpPlan`
  生成 leaf
  `KernelSpec.compute_ops`。
- top-level
  `TTProgram.payload`
  删除后，
  admission /
  diagnostic
  也不能再挂回
  `TTProgram`
  公共 bag；
  未支持能力应在构造
  `TTProgram`
  前 fail-close，
  或进入明确 typed leaf
  admission 字段。
- plan-local
  `TT*Plan.payload`
  也不能作为局部兼容壳保留。
  `TTProgram`
  内部 plan object 需要跨 builder /
  validator /
  projection 保留的信息，
  应直接建 typed field；
  例如 CB requirement index/name、
  dst layout page size、
  materialization host/boundary、
  consumer binding target 等，
  都应由 typed plan field
  进入 executable projection。
- lowering-support analysis
  不能用
  `Array<Any>` /
  `Map<String, Any>`
  contract-map
  在 helper 和 lowering pass
  之间传协议。
  `BlackholeLoweringSupportFacts`
  这类只在当前 pass 内消费的结果
  应保持 typed C++ struct；
  一旦事实需要跨阶段保留，
  就应升级到
  `SpatialPlan` /
  `TTProgram`
  typed IR 字段，
  不能重新变成 public bag。
- compute-op planning
  不要保留 pass-local
  `Map<String, Any>`
  seed
  再转
  `TTComputeOpPlan`。
  这类 map seed
  会把旧
  contract-family
  reader 形式留在 pass 内；
  直接用 typed fact /
  typed plan constructor
  表达 GEMM 或后续 compute kind。
- `TTComputeOpPlan`
  的 `kind`
  只表达 compute family
  （如 `gemm` /
  `binary` /
  `unary` /
  `reduce`）；
  具体 exact TT-Metal builtin
  应进 typed
  `operation_name`。
  非 GEMM 内部 operand
  不要为了复用 GEMM leaf schema
  伪造 `host_buffer`；
  只有 GEMM direct-runtime
  admission 需要显式 host runtime buffer。
- `TTKernel`
  / `TTABIPlan`
  这类 public `TTProgram`
  object
  不应把 leaf launch /
  compute config /
  per-work descriptor /
  runtime arg /
  compile-time arg /
  accessor /
  semaphore binding
  挂成
  `Map<String, Any>` /
  `Array<Any>`；
  用 typed object
  表达 owner truth，
  只在
  `ExecutableSpec`
  projection boundary
  编码成 leaf segment map。

## 9. 调试模式

- 先判断问题落在哪一层：
  - 结构层
  - planner/spec 层
  - runtime 执行层
- 手工 pass 链和 full `lower()` 不一致时，先比对 optimized path 的 IR 形态，
  再检查入口 `PrimFunc` 是否被提前过滤
- compile / launch 已经通过时，优先查 CB 生命周期、同步协议和 runtime arg materialization
- copy 跑通只证明字节路径可用，不证明 matmul / tile layout contract 正确
- execution hang 优先配合 Watcher 看状态组合，而不是先堆日志
- flash-attn direct-runtime
  gate-bypass probe
  如果看到
  `tilelang_get_cb_write_ptr_bytes`
  /
  `CircularBuffer::get_tile_address`
  仍在 compute source，
  说明 local-fragment scratch staging
  还在 mailbox CB pointer path；
  TT-Sim bf16 correctness
  不能 admission。
  probe 后必须撤回 gate 放宽并重编
  `tilelang_repo/build/`
  里的 `libtilelang.so`。
- MATH-side direct access to
  `get_local_cb_interface(cb).fifo_wr_ptr`
  is not a valid replacement for mailbox CB pointer exchange.
  A gate-open flash-attn probe linked only after removing call sites;
  when the helper itself was called from compute/MATH source, TT kernel link failed with
  undefined `cb_interface`.
  Publication from compute must be expressed through TT compute APIs
  (`tile_regs_*`, `copy_tile`, `pack_tile`, typed live-form CBs),
  not raw CB write-pointer arithmetic in MATH code.
- For flash-attn exact op admission,
  generated source being non-mailbox is necessary but not sufficient.
  The old P2.1 failure mode was stale synthetic fill being published as the
  row-reduction input when source live-form truth was incomplete.  The P2.1
  source fix seeds selected source-live only for explicit single 32x32 matmul
  outputs and lets the row-reduction input borrow that streamed CB-live value;
  P2.2 later admitted the small/32x32 bf16 direct-runtime subset under
  TT-Sim.  Seq64 / multi-K-step remains a separate multi-block correctness
  admission gate, not the same old P2.1 blocker.
- For P2.2 flash-attn exact CB admission,
  row scalar broadcast uses TT-Metal `bcast_cols`, not `bcast_rows`.
  The semantic shape is a per-row scalar / column-vector broadcast even though
  the logical transform reads like a row reduction follow-up.
- Keep logical float32 softmax exact values physically stored in BF16 tiled
  CB pages for the admitted Blackhole direct-runtime subset. TT-Metal SDPA
  reference kernels keep these intermediates as `Float16_b`; using Float32 CB
  storage in this lane can produce simulator overflow/format failures rather
  than a useful correctness signal.
- Standalone accumulating row reductions such as
  `scores_sum += row_reduce(...)` must lower as typed exact reduce plus
  add/max CB ops. Do not fall back to fragment add helpers or raw local
  fragment staging.
- Helper/composite Blackhole builtin names such as
  `exp2_row_bcast_affine` and `scalar_exp2_affine` must not exist in
  production TIR builtin names, `TTComputeOpPlan.operation_name`,
  `ExecutableSpec.compute_ops`, or source/codegen protocol. Exact compute
  truth should be recorded at TT-Metal API granularity such as `mul_tiles`,
  `add_tiles`, `*_bcast_cols`, `exp2_tile`, and `pack_tile`.
  Do not stop at deleting the old names:
  leaf-looking payloads such as
  `exp2_tile(mode, lhs, rhs, scale, ...)`
  or
  `mul_tiles_bcast_cols("div", ...)`
  are the same violation if they encode composite semantics.
  The fix is explicit leaf TIR sequence normalization before DAG covering.
- For exact CB live-form republish, distinguish total CB capacity from
  per-event lifetime and from runtime correctness admission. Seq64 flash-attn
  uses multi-page CBs and has compile/source/spec support for one-page
  publish/consume events, but multi-block direct-runtime correctness remains
  behind the typed unsupported-reason metadata gate. Stage2/block64 shapes
  that require one event to publish or consume multiple pages still need a
  wider support contract and should fail closed.
- When a borrowed live source CB is copied/repacked into a new CB, pop the
  borrowed source as soon as there is no future read before the next write.
  Events at the next write boundary are a redefinition, not a consumer of the
  old live page. Do not keep an early deferred `cb_reserve_back` for buffers
  whose next producer is already a typed materialization/live-form writer.
- When recording or clearing an ordinary tiled CB live-form alias, also clear
  stale exact-output live-form aliases for the same logical buffer. Otherwise a
  later exact compute can wait on an old CB page that has already been popped
  after a typed materialization/republish event.
- Broadcast-cols leaf RHS values should be reader-produced as a full-tile CB
  page selected through the direct-copy source chain. Treat this as a
  producer-live CB distinct from a borrowed-live CB: compute-side publish is
  skipped because the reader already pushed the page, but the compute consumer
  still owns the pop when the value is not borrowed from another compute
  consumer.
- Multi-block flash-attn direct runtime must stay gated until the online-softmax
  source-live-form and event lifetime contract is admitted. Compile/source/spec
  stability is independent from direct-runtime correctness; do not reopen the
  runtime path by bypassing the unsupported-reason gate.
- 文档收口时，
  `tasks/progress.md`
  是唯一当前状态 /
  next-task source；
  `AGENTS.md`
  只能记录 repo-wide working facts，
  不能滞留已经删除的 bridge attr /
  contract-family fallback。
  `tasks/dev_design/`
  根目录只保留入口 /
  support lane /
  audit；
  completed boundary docs
  也放入
  `tasks/dev_design/archive/`；
  方法论和历史研究输入放
  `tasks/dev_design/archive/`。
- 如果开 `TT_METAL_WATCHER` 后症状从 hang 变成 `SIGABRT` 或只在 dump 期间卡住，
  先抓 native backtrace；问题可能在 `WatcherServer` 线程，而不是 direct runtime 主链
- 需要保留 watcher 现场但避免立即 abort 时，可临时开 `TT_METAL_WATCHER_TEST_MODE=1`
- Blackhole tile-compute preservation 之后，
  `lower_blackhole_ops.cc`
  的瘦身应按 implementation responsibility
  拆文件 / helper，
  不新增 IR 层或 side-channel。
  explicit preserved tile compute lowering
  已拆到
  `lower_blackhole_tile_compute.cc`；
  exact tiled-CB live-form /
  local materialization helpers
  已拆到
  `lower_blackhole_exact_cb.cc`。
  该条是 2026-04-27 的中间状态；
  2026-04-28 已继续拆完 fragment/local materialization、
  ABI/accessor、
  state/live-form、
  staged transport
  和 matmul 责任文件。
  `lower_tile_op.cc`
  还有重复的 Blackhole tile compute normalization，
  后续应集中成单一实现面，
  继续产出显式
  `tl.tileop.blackhole_compute`。
- 2026-04-28 的 post-preservation shrink 已把
  `lower_blackhole_ops.cc`
  收缩到 pass driver /
  CB requirement /
  logical shape /
  validator /
  visitor orchestration surface。
  责任文件为：
  `lower_blackhole_tile_compute.cc`
  （explicit tile compute）、
  `lower_blackhole_exact_cb.cc`
  （exact tiled-CB helper）、
  `lower_blackhole_materialization.cc`
  （fragment/local materialization）、
  `lower_blackhole_abi.cc`
  （segment/accessor/TTKernel/TTABI planning）、
  `lower_blackhole_state.cc`
  （live-form/materialization/buffer-flow state）、
  `lower_blackhole_transport.cc`
  （staged copy/transport emission）、
  `lower_blackhole_matmul.cc`
  （GEMM/matmul/merge lowering）。
  后续第一个 cleanup task
  应转向 `lower_tile_op.cc`
  的 Blackhole normalizer 去重。
- `lower_tile_op.cc` 的 Blackhole tile compute normalization
  已收成单一匿名命名空间 helper，
  由 `LowerTileOpPass`
  和 `BlackholeTileComputeNormalizer`
  共享调用。
  后续增加新的 admitted tile compute normalization
  时，应扩这个共享 helper，
  继续产出显式
  `tl.tileop.blackhole_compute`
  和 TT-Metal leaf API 粒度 operation name；
  不要在两个 pass class 里重新复制 matcher /
  generator surface。
- 2026-04-28 后续 Blackhole refactor
  已优先走算法化基础：
  affine-lite `AccessRegion`、
  graph-backed `SpatialPlan` dependence construction、
  `LiveValueSSA` / BufferSSA-style version and event model。
  这些仍然落在现有
  `Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec`
  主链内，
  不是新 IR 层。
  tile compute selection 的下一层设计是
  `TileComputeDAG` + legalizer + TT-Metal leaf pattern covering；
  新增 compute 支持应扩 pattern / legality / cost / tests，
  不要新增 workload-specific matcher branch。
- 2026-04-28 对上述算法化设计做架构复核后，
  固定两条实现纪律：
  任何 `AccessRegion` /
  dependence /
  live-version /
  event-lifetime
  结果一旦被后续 phase 消费，
  必须进入 typed
  `SpatialPlan` /
  `TTProgram`
  字段，
  不能留作 parallel map /
  helper side channel；
  `TileComputeDAG`
  只能是 pass-local selection model，
  covering 输出必须是 typed plans
  和 leaf API 粒度 `operation_name`，
  不能变成新的 matcher payload
  或 source-string protocol。
- 2026-04-28 早先记录的 Blackhole 任务顺序
  后来又被 2026-04-29 hardware-codegen usefulness gate 修正；
  该记录只说明当时路线为何调整，
  不代表当前 active lane：
  `AccessRegion` /
  graph-backed `SpatialPlan` dependence /
  `LiveValueSSA` /
  第一版 TT live-form solver
  已是 foundation-complete，
  并在 admitted live-form /
  materialization surface
  上有 selected decision-use；
  当前 active lane 统一只看 `tasks/progress.md`。
  `TileComputeDAG` /
  legalizer /
  covering
  只能在 explicit leaf graph
  上证明 fanout /
  materialization /
  physical-form /
  resource-demand /
  typed reject 决策价值；
  不能靠 composite pseudo-leaf
  或 source hook expansion
  证明 production 价值。
- 2026-04-28 Algorithmic generalization Phase A
  已添加 typed `AccessRegion` 到 `SpatialPlan`。
  builder 从 execution-unit `read_buffers` /
  `write_buffers`
  和 layout / buffer metadata
  生成 access regions；
  validator 要求 unit/subject/kind/coverage
  与 execution unit read/write 对齐。
  Phase A 不改变 TTProgram 或 source generation 行为。
- 2026-04-28 Algorithmic generalization Phase B
  已添加 shared `spatial_dependence_graph` helper。
  `BuildSpatialPlan` 现在从 typed `AccessRegion`
  事件顺序生成 flow / carry / join closure boundaries，
  从 local value-flow evidence 生成 same-unit materialize
  `DataflowEdge`，
  并把 Tarjan SCC / self-cycle 诊断保存为 typed
  `SpatialPlan.dependence_components`。
  `build_spatial_plan.cc` 不再本地维护 read/write boundary
  matcher 或 materialize edge constructor。
- 2026-04-28 Algorithmic generalization Phase C
  已把 `SpatialPlan` live-value surface 改成 versioned SSA 形态：
  `LiveValue` 带 version / definition / defining access event，
  `LiveValueEdge` 带 use kind / consumer access / source-target
  version，
  `MaterializationBoundary` 带 source-target access /
  event lifetime / publish-consume page bounds。
  local materialize edge 的 source 现在解析为 reaching live version；
  不再把 consumer unit 上的 source buffer 合成一个新的 stale
  definition。
- 2026-04-28 Algorithmic generalization Phase D
  已添加 `tt_live_form_solver.{h,cc}`。
  fragment/cast live-form transfer 的 physical form、
  topology、
  ownership、
  materialization/publication protocol
  和 consumer slice/full-tile requirement
  现在由 solver request/result 决定，
  `PlanTTKernelABI` 只把 solver 结果投影成 typed
  `TTLiveFormPlan` /
  `TTMaterializationPlan` /
  `TTConsumerBindingPlan`。
  seq64 / multi-K-step flash-attn direct runtime 仍通过 typed
  unsupported-reason metadata fail closed；
  这轮没有做 runtime-only source-live-form patch。
- 2026-04-28 对 Algorithmic generalization 做第二轮 prior-art
  对齐后，修正完成标准：
  LLVM GlobalISel / MLIR conversion / MemorySSA /
  dependence graph / Cranelift ISLE 的共同点是算法结构必须成为
  query / legality / action 的入口。
  因此 `AccessRegion`,
  `DependenceComponent`,
  `LiveValueSSA`,
  和 live-form solver
  只要还只是构造、dump、validator coverage，
  就只能算 foundation-complete，
  不能算 usage-complete。
  后续必须先做
  `Algorithmic generalization Phase E: Decision-Use Cutover`：
  让这些 typed objects 改变 legality、solver output、
  typed TT plans 或 unsupported diagnostic，
  再进入 `TileComputeDAG` / legalizer / covering。
- 2026-04-28 用户指出 Algorithmic generalization 不能过快收窄成
  单个实现切片。补充调研 LLVM VPlan / LoopAccessAnalysis /
  MLIR Affine / Linalg 后，Phase E 被重新定义为宽口径
  decision-use cutover：
  `SpatialPlan` legality、
  `LiveValueSSA` query、
  `TTProgram` action、
  `ExecutableSpec` admission
  和后续 `TileComputeDAG` covering
  必须一起按 typed evidence dependency 排序。
  fragment/cast solver 接入只能算 E1/E3 的局部验证切片，
  不能代表 Phase E 完成。
- 2026-04-28 Algorithmic generalization Phase E 第一条
  decision-use 切片已接入：
  `BuildSpatialPlan`
  用 `AccessRegion`
  compatibility 决定 distributed-slice live edge，
  用 recurrence `DependenceComponent`
  决定 loop-carried materialization lifetime；
  `ValidateSpatialPlan`
  对缺失 access-region evidence
  和缺失 recurrence evidence fail closed；
  `PlanTTKernelABI`
  的 fragment/cast materialization boundary lookup
  从 source/target subject pair 改成 source/target live-value index pair；
  `tt_live_form_solver`
  消费 boundary logical coverage
  来决定 consumer full-tile vs distributed-slice requirement。
  后续 Phase E 已删除 subject 到 latest live value 的 wider owner truth；
  该记录保留的是 cutover 前的 migration checkpoint，
  不再代表当前 active 状态。
- 2026-04-28 记录 Algorithmic generalization anti-overdesign
  pay-rent rule：
  新的算法对象 / 字段 / solver 状态 / DAG pattern
  只有在改变 legality、
  回答 query 并删除 side channel、
  选择 typed TT plans、
  提前产生 typed unsupported diagnostic、
  或删除旧 matcher/helper/payload/fallback 时，
  才算真正实现进展。
  仅构造对象、dump、shape-only validator、
  constant-output solver、
  或不影响生产计划的 pattern/DAG scaffolding
  只能算 foundation work，
  不能报告为 Algorithmic generalization 完成。
- 2026-04-28 记录与 pay-rent rule 配套的
  problem-family generality rule：
  当前 case 只是证明抽象已经接入 active chain 的 witness，
  不能成为抽象规格本身。
  每个新算法结构必须同时说明：
  它服务的可复用问题族是什么，
  以及哪个最小当前 witness
  证明它改变了 legality / query / typed plan /
  unsupported diagnostic。
  只能说明问题族的是架子货风险；
  只能说明当前 witness 的是 workload overfit 风险。
- 2026-04-29 记录硬件代码生成有效性准则：
  新设计对象、算法结构、pass、typed field 或 validator
  必须能说明它如何让 DSL 写出来的 kernel
  更可靠或更高效地 lower 到真实 TT-Metal 硬件代码。
  只构造对象、dump、shape-only check、metadata projection、
  测试覆盖或 paper-like algorithm name
  都不能算主线完成；
  它必须改变 leaf normalization、legality、typed plan、
  resource plan、admission diagnostic，
  或删除旧 matcher / payload / fallback / side channel。
- 2026-04-28 文档维护纪律：
  入口文档必须按“当前事实 vs 历史记录”逐篇审计；
  不能只扫几个已知旧短语。
  `README.md`,
  agent docs,
  `tasks/dev_design/README.md`,
  task contracts,
  support lane,
  protocol audit,
  and `progress.md`
  must keep a single status owner:
  dynamic lane / blocker / support boundary
  live in `tasks/progress.md`.
  Archive docs are historical only and cannot authorize legacy residue.
- 2026-04-28 状态文件边界：
  `tasks/progress.md`
  is a status board, not a changelog.
  Keep only current execution status,
  blocker,
  support boundary,
  open debt,
  next task order,
  and latest verification summary.
  Put detailed rationale in design docs,
  reusable lessons in `memory/`,
  and per-change narrative in commits.
- 2026-04-28 Tile compute covering productionization:
  the first useful migration step is to route both durable
  `TTComputeOpPlan`
  recording and source dispatch through the same covering selection,
  while keeping source emission output unchanged.
  This proves the selector is active on the main chain without turning the
  pass-local covering object into a new cross-stage payload.
  后续 boundary review 发现：
  selected-pattern ownership of source-plan hook
  只能算 mechanics；
  如果 hook 仍然展开 composite pseudo-leaf，
  不能报告为 production completion。
- 2026-04-28 Tile compute covering source dispatch:
  once covering is on the production path,
  do not keep a second operation-name selector below it.
  Put the source-emitter hook in the pattern metadata,
  carry that hook in the covering decision,
  and make low-level emitters receive already-selected TT-Metal builtins
  instead of rediscovering add-vs-mul or similar choices from
  `operation_name`.
- 2026-04-28 Tile compute DAG covering:
  if production covering needs the DAG,
  expose a typed pass-local C++ DAG builder and let diagnostics encode that
  structure.
  Do not make C++ production code consume an FFI diagnostic map as a semantic
  source.
  Producer-use edges should be connected by IR object identity for buffer
  values;
  textual value strings / buffer names are only diagnostic output.
  `ValidateTTProgram`
  should check selected covering patterns before falling through to older
  legality helpers, otherwise the old helper remains the real production gate.
- 2026-04-28 Tile compute covering Phase E cleanup:
  an inline source-emitter lambda table below covering is still a second
  selector, even if it is keyed by selected metadata.
  Keep pattern
  `source_emitter`
  values covered by a single hook registry,
  route generic explicit ops such as
  `tl.tileop.reduce`
  through the same covering dispatch before emission,
  and make source emitters that are not admitted as standalone explicit calls
  fail closed before emission instead of falling back to branch-only
  lowering.
  The hook registry is only a leaf projection table;
  it must not become the place where one source node expands into multiple
  semantic compute plans.
- 2026-04-28 Tile compute covering implementation cleanup:
  after the selector is active,
  the next useful code cleanup is not an OOP strategy hierarchy;
  it is shrinking string drift.
  Use typed enums for operation / result kind / operand role / value form /
  side-effect / source-emitter kind,
  make source-emitter metadata optional for patterns without standalone
  explicit source emission,
  and keep blackhole/generic call operand layouts in the pattern table so DAG
  construction and source argument parsing do not each maintain their own
  operation-name branches.
- 2026-04-28 Tile compute pattern schema boilerplate:
  typed enums should not imply one handwritten
  `ToString`
  switch per enum family.
  Keep enum/string conversion table-driven,
  use a generic vector-to-name helper for schema encoding,
  and prefer direct aggregate initialization for pattern call-operand vectors
  over wrapper helpers that only restate
  `std::vector`
  construction.
- 2026-04-29 DAG-backed resource pressure:
  if resource planning needs `TileComputeDAG`
  fanout,
  capture it before
  `SelectBlackholeTTMetalBuiltins`.
  Builtin selection deliberately reduces the visible compute graph to selected
  leaf ops and can erase useful multi-consumer evidence.
  Store the durable result as typed
  `TTResourceDemand`
  /
  `TTResourcePressureReport`,
  then let later TTProgram phases refresh kernel,
  core,
  CB,
  semaphore,
  transport,
  and distribution counters.
  Do not rebuild production resource demand from the reduced post-selection
  DAG.
- 2026-04-29 CB / L1 resource admission:
  do not let `TTResourcePressureReport`
  become another diagnostic dump.
  Hardware facts that affect admission belong in `TTHardwareModel`
  and must be copied into the typed report so `ValidateTTProgram`
  can fail closed before codegen.
  `PlanTTCBAlloc`
  and the report validator should read the same CB-count / worker-L1 /
  L1-alignment facts; stale local constants are only defaults when no target
  model exists.
- 2026-04-29 TileComputeDAG lower-plan payoff:
  a DAG analysis only pays rent when its selected decisions change the
  production lowering path or typed plans.
  For Blackhole tile compute,
  source lowering may consume a pass-local DAG lower plan only as a selected
  leaf projection.
  Exact compute plans should carry the source DAG node /
  materialization /
  fanout decision into typed plans when those fields drive validators,
  resource admission,
  or old branch deletion.
  A leaf source hook is projection metadata,
  not runtime-facing semantic truth.
  Boundary correction:
  that source hook must be one-to-one with the semantic leaf op selected for
  the DAG node.
  It is invalid for one source DAG decision such as a leaf-looking
  `exp2_tile`
  call to expand into several semantic leaf compute ops.
  Composite TIR expressions such as
  `exp2(lhs * s0 - rhs * s1)`
  or division by a scalar-load operand must first be decomposed into explicit
  `Normalized Tile TIR`
  leaf statements.
  If a computation is not admitted,
  classify it as
  `lowering_missing`,
  `backend_op_missing`,
  `admission_blocked`,
  or true semantic unsupported only after TT-Metal primitive coverage audit.
  Also do not assume `CollectExecutionOrderedStmts`
  order is identical to `StmtMutator`
  visitation in select-only phases;
  if stricter binding is needed,
  add a real statement/source identity to the DAG instead of using a linear
  cursor as owner truth.
- 2026-04-29 explicit tile-compute leaf normalization repair:
  composite tile compute must be normalized before DAG construction.
  The active Blackhole boundary now treats
  `exp2(lhs * s0 - rhs * s1)`
  as a sequence of explicit
  `copy_tile` /
  `fill_tile` /
  `mul_tiles` /
  `add_tiles` or
  `add_tiles_bcast_cols` /
  `exp2_tile`
  leaf statements,
  and division by a scalar-load operand as
  `recip_tile`
  plus
  `mul_tiles_bcast_cols`.
  Do not reintroduce string-mode source payloads such as
  `exp2_tile("binary", ...)`,
  `exp2_tile("bcast_cols", ...)`,
  or
  `mul_tiles_bcast_cols("div", ...)`.
  Source hooks in
  `PlanTTKernelABI`
  are one-leaf projections for selected typed plans;
  they are not a fallback place to expand composite expressions.
- 2026-04-29 core-doc maintenance:
  keep core docs role-specific.
  `final_blackhole_backend_redesign.md`
  is for durable architecture contracts only;
  `tasks/progress.md`
  is for current HEAD status, blocker, next task, and verification;
  `tasks/dev_design/README.md`
  is an index and maintenance policy;
  task-level design docs should contain goal, non-goal, representation
  contract, validation, and completion criteria.
  Do not dump implementation patch notes, phase logs, repeated verification,
  or historical justification into these entry docs.
- 2026-04-29 Blackhole lowering boundary cleanup:
  target-specific scalar-loop normalization should not live inside generic
  `lower_tile_op.cc`.
  Keep the generic pass to a narrow normalizer call and put Blackhole
  leaf-call builders, temp construction, and composite decomposition in a
  dedicated normalizer module.
  Also keep explicit tile-compute source projection out of
  `PlanTTKernelABI`'s header surface; a small projection helper can friend the
  ABI class and dispatch selected leaf emitters without exposing a long method
  family as protocol.
  When adding new common `.cc` files, rerun `cmake -S . -B build` before
  rebuilding because the current CMake source glob is resolved at configure
  time.
- 2026-04-30 Blackhole duplicate-lowering cleanup:
  after splitting a large lowering file by responsibility, immediately check
  whether same-shaped logic was merely moved into per-op methods.
  Binary, broadcast-cols, and unary tile compute leaves should share category
  emitters when only builtin call names differ.
  Row-reduction and other exact tile compute sequences should reuse the common
  CB / tile-register / pack emitter instead of open-coding a parallel source
  sequence.
  Avoid pure `Try* -> Normalize*` forwarding wrappers; keep `Try*` only for
  real may-fail matchers.
- 2026-04-30 Blackhole normalizer cleanup correction:
  target leaf normalization should stay a bounded local normalizer, not an
  open-ended rule registry / benefit table / workload-pattern catalog.
  A shared builder is still useful for rendering repeated unary / binary leaf
  call shapes, but source-expression helpers must not create named composite
  semantic families.
  Local normalization should produce an ordered explicit leaf-call plan, and
  the builder should be the only place that renders
  `tl.tileop.blackhole_compute` statements.
  Source projection metadata for same-shaped leaves belongs in the leaf
  pattern schema; do not maintain a parallel hook table that repeats the same
  operation-to-emitter mapping.
- 2026-04-30 Blackhole modern C++ audit baseline:
  host/runtime and generated scalar bitcasts must use memcpy-style bit casts,
  never aliasing `reinterpret_cast` or union punning.
  Raw DLTensor direct-runtime copies are only valid for compact row-major
  layouts; non-compact strides must fail before memcpy.
  Blackhole hardware facts should have one owner truth in `TTHardwareModel`
  and target attrs (`max_cb_count == 64` for current Blackhole), with CB/core
  validators consuming the same facts before source/runtime emission.
  Runtime leaf readers should require typed `ExecutableSpec` fields instead of
  defaulting to empty maps or `(0, 0)` cores.
  TVM export of imported Blackhole modules needs real non-empty
  `SaveToBytes` plus `ffi.Module.load_from_bytes.blackhole`; returning empty
  bytes is worse than failing because it produces a load-time trap detached
  from the original module.
- 2026-04-30 Blackhole buffer placement boundary:
  do not classify every L1 layout as sharded.
  Shared-visible or CB-backed L1 buffers should carry attached-core sharded
  placement; ordinary per-worker local L1 state can remain device-local
  replicated until the address ABI grows per-work indexed descriptors.
  CB-backed buffers should take page size from `TTCBPlan` and must not be
  double-counted as allocator-managed L1 in resource pressure.
  Validator L1 checks should compare aligned size against worker L1 budget,
  not require every logical local byte size to already be alignment-sized.
  Copy/transport-only kernels still need resource-only `TTResourceDemand`
  records even when the tile-compute DAG is empty.
- 2026-05-01 Blackhole sharded placement residue:
  current sharded placement marks the L1-side working view/materialization,
  not the DRAM/global tensor itself.
  `TTBufferDistributionPlan.shard_shape` is a coarse core-grid marker,
  not a real per-core data shard shape.
  Before sharded runtime address emission, split the representation into
  shard grid shape, sharding strategy, per-core shard data shape,
  source buffer / source region binding, explicit logical-index to core-local
  address mapping, and DRAM-source region to L1-shard copy/address mapping.
  Do not let source or runtime readers infer sharded addressing from buffer
  names, layout strings, or the old `shard_shape` field.
- 2026-04-30 active-doc cleanup rule:
  when a design doc starts carrying implementation closeout notes, command
  matrices, or historical explanations needed only once, rewrite it back to
  role / goal / non-goal / contract / validation.
  Keep exact verification commands in final reports, commits, or short
  progress summaries; do not let active design entry files become
  chronological notebooks.
- 2026-05-01 Blackhole executable address contract:
  `TTBufferDistributionPlan` must be projected into `ExecutableSpec` before
  direct-runtime admission.
  Runtime-bound external buffers still need explicit accessor /
  compile-time `transport_page_size`; buffer distribution may validate
  consistency, but must not become a fallback that hides missing accessor ABI.
  For CB-backed L1 resident views, source binding should come from the
  explicit `read_*_to_cb(source_var, ..., cb_id, ...)` builtin argument and
  the typed CB requirement names.  Do not depend only on `buffer_map`, because
  after lowering / packing the source Var may remain explicit in the builtin
  body while `buffer_map` is no longer a complete identity table.
- 2026-05-02 Blackhole tensor placement / reshard projection:
  tensor placement intent should enter `SpatialPlan.TensorPlacementIntent`,
  resolve to `TTTensorMemoryConfigPlan` / `TTOpShardingContract` /
  `TTPlacementResolutionPlan` / `TTReshardPlan`, and only then project to
  `ExecutableSpec`.
  Direct runtime may admit or reject reshard kinds from projected
  `TTReshardPlan` records, but must not infer conversions from
  `TTBufferDistributionPlan.source_buffer`, source text, names, or accessor
  strings.
  `TTOpShardingContract` coverage is for operands with a matching
  `TTTensorMemoryConfigPlan.subject`; scalar or constant exact-CB operands
  such as a reduce scaler do not own tensor placement and should not be
  forced into a memory-config contract.
  `tile_compute_dag_node_id` is local evidence for the DAG/lowering plan that
  produced it; seeded and final compute plans can share numeric node ids.
  Tests and validators should not treat that id as a cross-stage global
  identity unless a separate DAG identity/version is represented explicitly.
