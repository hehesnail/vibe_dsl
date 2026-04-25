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
  `progress.md` 只记录 repo HEAD 状态与下一步；
  `layered_ir_references.md`
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
  codegen/runtime 只能解释 `value_kind`，不能再按 arg kind 名字推语义
- `per_work_arg_specs`
  一旦完成 kernel-local canonicalization，
  就不要再保留 top-level `TTProgram.payload`
  版本给 reader 当 fallback；
  否则 single-kernel/multi-kernel 两条 host path
  会重新出现“segment truth 缺了但 top-level bag 还能兜住”的双真源
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
  repo HEAD 的做法是：
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
- 现阶段 Blackhole runtime/direct-runtime regression 默认统一用 `bf16` 输入；
  不要再把 `fp16` 当成 TT-Sim 上的正式 runtime baseline
- 当前 cleanup correctness gate
  的 admitted runtime
  只包括
  copy / GEMM，
  以及 live-form /
  materialization admission
  后的当前 supported shapes：
  constant
  和
  `fragment_fill -> cast -> publish`
  的
  `pack_thread_direct_store`
  path，
  以及 zero-preclear
  GEMM post-merge
  direct cast consumer
  的
  `pack_tile`
  path；
  更宽 direct cast /
  live-in materialization
  仍不能混进
  TT-Sim hard gate，
  需要先按
  `tasks/dev_design/2026-04-23-blackhole-live-form-materialization-admission.md`
  扩 explicit live-form /
  materialization protocol，
  不要写 runtime-only patch
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
- 如果开 `TT_METAL_WATCHER` 后症状从 hang 变成 `SIGABRT` 或只在 dump 期间卡住，
  先抓 native backtrace；问题可能在 `WatcherServer` 线程，而不是 direct runtime 主链
- 需要保留 watcher 现场但避免立即 abort 时，可临时开 `TT_METAL_WATCHER_TEST_MODE=1`
