# 问题与 Bug 记录

> 本文档只保留仍有复用价值的问题模式。
> 阶段状态、总体 blocker 与完成判定以设计文档和 `tasks/progress.md` 为准。

## 1. 当前未解决

### TT-Sim 的 fatal taxonomy 需要先按 simulator 约束判断，不要直接误判成 target contract 回归

- **现象**:
  - direct runtime / TT-Sim 运行时可能直接报
    `UntestedFunctionality`、`UnimplementedFunctionality`
    或 `UnsupportedFunctionality`
- **根因**:
  - 当前 `libttsim_bh.so` 二进制里有公共 fatal helper，
    这三类 taxonomy 都会直接打印并 `_Exit(1)`
  - 也就是说，这些不是普通 warning，而是 simulator-side hard gate
- **当前结论**:
  - 首次命中这三类错误时，先查
    `memory/tt_simulator_constraints.md`
  - 先把问题分类成 simulator capability boundary
    还是 TileLang target contract 回归，再继续分析
  - 当前已确认 `fp16` unpack 只是其中一个显式 gate，不是唯一约束面

### TT-Sim 上的较大 `float16` flash-attn runtime 属于 simulator fp16 boundary

- **现象**:
  - `flash-attn` small bf16 MHA direct runtime 已能真实执行并和 reference 对齐
  - 但较大 `float16` MHA case 在当前 TT-Sim 上仍会命中
    `UntestedFunctionality: tensix_execute_unpacr: fp16`
- **根因**:
  - 失败点来自 simulator 自身对该 `fp16` 执行路径的能力边界，
    不是 `direct_runtime_unsupported_reasons`
- **当前结论**:
  - 现阶段应把 small bf16 runtime case 当作 correctness gate
  - 不要把 TT-Sim `float16` 能力边界直接误判成 TileLang target contract 回归
  - 更宽 `MHA / GQA` / 大 shape runtime payoff
    当前不属于第一性原理收口集；
    归到后续 support-surface / workload payoff backlog
  - 该问题的 simulator-side 旁证和更宽 fatal taxonomy 扫描，
    统一见 `memory/tt_simulator_constraints.md`

### exact CB republish 不能靠 raw compute-side CB interface 晋级 direct runtime admission

- **现象**:
  - 将 flash-attn
    `thread_distributed + cb_republish`
    的 publication protocol
    标成
    旧名 `cast_fragment_slice_to_tiled_cb`
    并从 direct runtime gate 中放行后，
    TT-Sim bf16 runtime 会在 TT-Metal JIT 阶段失败：
    `trisc2`
    编译报
    `get_operand_id was not declared`，
    `trisc1`
    链接报
    `undefined reference to cb_interface`
- **根因**:
  - compute kernel 中直接读
    `get_local_cb_interface`
    或手写维护 CB read/write pointer，
    不等价于一个 TT compute-side 可链接的 publication protocol
  - 该路径会重新落回 memory 中已知的
    compute-side CB interface / mailbox boundary，
    不能作为 admitted runtime support surface
- **当前结论**:
  - active protocol 名已收为
    `tilize_cast_fragment_slice`；
    旧名
    `cast_fragment_slice_to_tiled_cb`
    只应作为历史 bug / forbidden regression label
  - typed
    `TTMaterializationPlan.publication_protocol`
    和 executable metadata
    可以表达 exact CB republish，
    但 source 实现必须走
    `copy_tile` /
    `pack_tile`
    或等价 TT compute-linkable API
  - direct runtime admission
    接受已证明的
    `pack_thread_direct_store`
    /
    `pack_tile`
    /
    per-event one-page exact CB republish
    subset；
    stage2/block64
    multi-page publish/consume event
    仍需后续 typed contract

### Blackhole tile compute 不能先 scalar expand 再靠 late matcher 恢复

- **现象**:
  - P2.2/P2.3 为了 admit flash-attn，
    在 `lower_blackhole_ops.cc`
    中从 post-`LowerTileOp`
    scalar loop / local expression
    恢复 row reduction、
    broadcast、
    exp2 affine、
    scalar max/fma/copy/fill/cast
    等 TT-Metal compute sequence
- **根因**:
  - Blackhole 是 tile-based compute target；
    TT-Metal 已经以 `matmul_tiles`、
    `reduce_tile`、
    `add_tiles`、
    `mul_tiles`、
    `*_bcast_rows/cols`、
    `exp2_tile`、
    `copy_tile`、
    `pack_tile`、
    `tilize_block`、
    `untilize_block`
    等 leaf API 表达 compute semantics
  - generic scalar lowering
    在 exact builtin selection 前破坏这些语义，
    后段 matcher 被迫重新从 scalar idiom
    猜回 tile compute intent
- **当前结论**:
  - 这是通用架构债务，
    不是 reduce-only
    或 flash-attn-only 问题
  - 后续实现必须按
    `tasks/dev_design/2026-04-27-blackhole-tile-compute-preservation.md`
    把 TT-Metal API 粒度 tile compute semantics
    上移到 `Normalized Tile TIR`
    preservation / normalization
  - `softmax` /
    `exp2_affine` /
    `row_broadcast_exp2_affine`
    等 composite helper
    不能进入生产 compute op 协议

## 2. 已解决但值得记住的模式

### sharded L1 source-region ABI 必须 all-or-none，不能给纯 local scratch 伪造 source

- **症状**:
  - 将 `TTBufferDistributionPlan.source_region_shape`
    对所有 sharded L1 plan 都填上后，
    flash-attn、GEMM 和 fragment/local buffer 用例在
    `ValidateTTProgram`
    处失败：
    validator 看到 source-region 字段存在，
    但没有对应 `source_buffer`。
- **根因**:
  - L1 sharded plan 有两类对象：
    从 DRAM/global buffer materialize 出来的 resident L1 view，
    以及纯 worker-local scratch / fragment / intermediate。
    前者需要 source buffer / source region binding；
    后者没有全局 source，
    不能为了让 shape 字段完整而伪造 source binding。
- **修法**:
  - `BuildTTProgram`
    只在能从当前 IR / CB plan 稳定证明 materialized source
    时设置
    `source_buffer` /
    `source_region_kind` /
    `source_region_shape`。
  - `ValidateTTProgram`
    对 source-region group 做 all-or-none 校验；
    sharded L1 placement 仍必须有
    `shard_grid_shape`、
    `sharding_strategy`、
    `shard_shape`
    和 address mapping。
- **教训**:
  - source-region ABI 和 resident placement ABI 是两个对象。
    validator 要 fail-close 不完整 source binding，
    但不能把“没有 source”的 pure local scratch
    误判成缺协议。

### live-form solver 不能把 self carry boundary 当成 physical transfer

- **症状**:
  - Phase E 把 materialization planning 切到 graph/worklist solver 后，
    `fragment_fill -> cast -> publish`
    的 planner 测试会在 live-form solver 内部拒绝 selected boundary
  - 调试 dump 显示同一个图里既有
    `C_local -> C_local`
    /
    `D_local -> D_local`
    的 loop-carried self boundary，
    又有
    `C_local -> D_local`
    的 materialize boundary
- **根因**:
  - self carry boundary 是 recurrence / lifetime evidence，
    表示同一个 logical live value 跨事件保持可见；
    初版 solver 把它当成 physical transfer edge，
    导致 source live value 的
    `Fragment`
    状态和 self boundary 推出的
    `ExactCB(multi_event)`
    状态 join 成 conflict
- **修法**:
  - worklist solver 仍加载 self carry boundary
    作为 validated graph evidence，
    但 transfer 阶段跳过
    `source_live_value_index == target_live_value_index`
    的 boundary
  - selected materialization boundary
    仍按 indexed source/target live value
    做 physical live-form transfer
- **教训**:
  - `MaterializationBoundary`
    不是每条都代表物理 publication；
    carry/self edge
    和 materialize edge
    在 graph 上都重要，
    但 transfer function 必须按 live value identity
    区分 lifetime evidence 和 physical form movement

### preserved tile op 缺少 dataflow access 会让 SpatialPlan 漏 producer truth

- **症状**:
  - Blackhole 上保留 `tl.tileop.reduce`
    后，`SpatialPlan`
    的 compute unit 能看到 source read，
    但漏掉 reduce destination write；
    flash-attn 里表现为
    `scores_sum`
    carry/dataflow truth 消失
- **根因**:
  - `ReduceOpNode`
    之前没有实现
    `GetDataflowAccessInfo()`；
    旧路径靠 scalar-expanded
    `BufferStore`
    偶然提供 producer truth，
    preserve tile op 后这个旁路不再存在
- **修法**:
  - 在 tile op 类型自身记录
    `src` compute consume、
    `dst` compute produce；
    对 `clear=false`
    reduce 还要记录
    `dst` compute consume
- **教训**:
  - 将 TT-Metal API 粒度语义前移到
    `Normalized Tile TIR`
    时，operator-level dataflow contract
    必须和 preservation 同轮补齐；
    不能继续依赖 lower 后的 scalar IR
    帮 `SpatialPlan` 恢复读写关系

### reduce explicit lowering 不能提前清掉 accumulator live/fill truth

- **症状**:
  - preserved `tl.tileop.reduce`
    接入 selector 后，
    TT-Sim flash-attn runtime source
    重新出现
    `tilelang_cb_write_ptr_bytes_direct`
    /
    `get_local_cb_interface`，
    并在 TRISC link 阶段报
    `undefined reference to cb_interface`
- **根因**:
  - explicit reduce lowering 在调用
    `GenerateRowReductionSequence()`
    前提前 invalidated destination
    fill/live facts。
    对 `clear=false`
    row max accumulator，
    generator 因此无法复用
    `-inf` fill 或已有 exact live CB，
    退回到 raw fragment-to-CB tilize bridge
- **修法**:
  - 让 row-reduction generator
    在消费 accumulator truth 后
    自己通过
    `RecordExactOutputLiveForm()`
    更新/失效输出；
    不在 match 分支提前清理
    destination live/fill facts
- **教训**:
  - 对 read-write compute op，
    “写 destination”
    的失效点必须晚于
    “读旧 destination”
    的 materialization 决策；
    否则会把 typed live-form path
    降级回 forbidden direct CB interface

### pre-opt `SpatialPlan` 只能作为 typed layout merge source，不能整份替换 optimized plan

- **症状**:
  - 删除 bridge attr 后，为了保留 logical tile layout，
    如果直接跳过 optimized path 上的
    `BuildSpatialPlan`，
    后续会丢优化后的 execution units /
    ingress-egress units /
    dataflow truth
- **根因**:
  - pre-opt plan 的 layout truth 有价值，
    但它的 execution/dataflow truth
    不是 optimized body 的 owner truth
- **修法**:
  - pre-opt 阶段只保留 typed
    `SpatialPlan.LayoutSpec`
    作为 merge source
  - optimized body 仍重建
    `SpatialPlan`
  - 按 subject
    只合并当前 optimized plan 缺失的
    logical/local/thread/replicate/inverse-index
    typed layout fields
- **教训**:
  - 删除 bridge attr 时不能用“保留旧 plan”
    替代重建当前层 IR；
    analysis-derived truth
    必须回到当前 IR 层的 typed object

### fragment-cast materialization 的 logical size 不能用 local slice size 代替

- **症状**:
  - `fragment_fill -> cast -> publish`
    的 leaf materialization plan
    在 bridge attr 删除后仍能生成，
    但 `logical_element_count`
    可能只剩单个 slice 的 8，
    而不是完整 logical tile 的 1024
  - 即使 metadata 已经是 1024，
    生成的
    `pack_fill_fragment_to_tiled_cb`
    调用仍可能保留
    `num_elements=8 / row_width=8`，
    导致 direct runtime
    只写出 tile 的局部片段
- **根因**:
  - materialization planner
    只看了当前 contract/slice extent，
    没有从 typed layout truth
    恢复完整 logical shape
  - source emission
    也不能只按
    Buffer object identity
    查 layout；
    fragment-view buffer
    需要同时用 materialization contract
    的 source/target subject
    去查 typed layout
- **修法**:
  - materialization logical size
    取 contract extent
    和
    `SpatialPlan.LayoutSpec.logical_shape`
    product
    的保守上界
  - pack-thread direct-store
    source call
    的
    `num_elements`
    和
    `row_width`
    同样按 typed layout shape
    覆盖 local slice contract
- **教训**:
  - live-form/materialization
    的 logical quantity
    应来自 typed layout object，
    不能退回到局部执行 slice

### post-merge `pack_tile` admission 不能只修最后一次 materialization

- **症状**:
  - `gemm + post-merge cast consumer`
    已经能在
    `TTProgram`
    /
    `ExecutableSpec`
    暴露 typed
    live-form /
    materialization
    owner truth，
    但最初只把
    `D_local`
    的 publication 改成
    `pack_tile`
    后，
    TT-Sim 仍在 accumulator reload
    helper 上命中 mailbox-style
    CB write-pointer path
  - host 侧随后又会把
    materialized bf16 output
    误按 GEMM accumulator
    `float32`
    dtype 校验
- **根因**:
  - direct runtime admission
    需要整个 device sequence
    都避开 mailbox helper；
    只修最终 cast publication
    不够
  - zero-preclear GEMM
    的 merge live-in
    可以由当前 IR
    `tl.blackhole.fill_fragment`
    zero fact
    证明为零，
    因此不需要把旧 accumulator
    先写入 reload CB
  - output host copy
    不能只看 GEMM compute contract；
    materialized output
    必须优先按
    `BufferMaterializationSpec.live_form_kind`
    读取
- **修法**:
  - post-merge cast consumer
    只在当前 IR
    仍有 zero-preclear fact
    且 target materialization contract
    完整时 admitted
  - merge 侧直接等待 partials CB，
    copy 到 DST register，
    再用
    `pack_tile`
    发布
    `D_local`
    materialized CB
  - `TTMaterializationPlan`
    记录
    `publication_protocol=pack_tile`，
    无 zero-preclear /
    非零 live-in
    保留 explicit unsupported gate
  - host output copy
    通过
    `BufferMaterializationSpec.live_form_kind`
    识别 materialized bf16 output，
    不再强套 accumulator dtype
- **教训**:
  - `pack_tile`
    admission 是 typed materialization protocol，
    不是 leaf source string patch
  - 当前 IR
    zero fact
    是局部 analysis，
    只能在 mutation 前使用；
    非零 live-in merge
    需要新的显式协议，
    不能被这个 admitted shape
    顺带放行

### 2.0 constant fill cb_republish admission 必须从当前 IR 的 fill builtin 推出，并在后续写入时失效

- **症状**:
  - `fragment_fill -> cast -> publish`
    增加
    `publication_protocol`
    后，
    初始实现仍把
    materialization
    判成
    `mailbox_write_ptr`
  - 修到读取 fill fact 后，
    `gemm + post-merge cast consumer`
    又被错误 admitted，
    因为 preclear fill
    的事实穿过了后续 matmul /
    merge 写入
- **根因**:
  - `fill`
    在
    `SelectBlackholeTTMetalBuiltins`
    阶段已经规范化为
    `tl.blackhole.fill_fragment`；
    到
    `PlanTTCompute`
    时不能再依赖上一 pass
    对原始 `For`
    的局部 matcher 状态
  - constant-fill fact
    只是当前 IR
    可重算的局部 analysis；
    一旦同一 buffer
    被 matmul / merge / add /
    reduction / scalar update /
    cast 等 producer 写入，
    必须立即失效
- **修法**:
  - `PlanTTCompute`
    从当前 IR 的
    `tl.blackhole.fill_fragment`
    builtin 记录 constant fill fact
  - 后续 producer 写目标时清掉该 buffer
    的 fill fact
  - 只有最后一个有效 producer
    仍是 constant full-tile fill
    的
    `cb_republish`
    才能选择
    `publication_protocol=pack_thread_direct_store`
- **教训**:
  - admission logic
    不能读上一阶段的 pass-local state；
    必须从当前 IR /
    typed materialization contract
    推出
  - 任何局部 analysis fact
    一旦跨过 mutation
    就是 stale fact；
    要么进入显式 IR，
    要么严格按当前 IR
    def/write
    失效

### 2.1 compute residual gate 不能把 row-state scalar / 1D carry buffer 当成 tile residue

- **症状**:
  - `PlanTTCompute`
    在 flash-attn / GQA
    会因为
    `scores_max` /
    `scores_max_prev`
    这类 row-state local store
    直接报
    `residual local store remains`
- **根因**:
  - residual gate
    把所有
    `local / blackhole.acc`
    store
    都当成必须 lower 掉的 tile residue
  - 但 `shape.size()==1`
    的 row-state carry buffer
    属于合法 leaf-local bookkeeping，
    不应和 tile/vector residue 混为一谈
- **修法**:
  - residual gate
    只拦截真正的 tile-like local residue；
    1D row-state carry store
    允许保留
- **教训**:
  - compute subset validator
    要按表示对象区分
    “tile fragment residue”
    和
    “row-state bookkeeping”
  - 不能只按 storage scope
    粗暴 fail-fast

### 2.2 grouped row / row-state distribution contract 不能让 generic layout 覆盖专用语义

- **症状**:
  - `flash-attn` / GQA 的 grouped `reduce_row` 会报
    `grouped_rows distribution contract` 缺失
  - 过渡 projection contract 的 `scope` / `shape`
    可能仍停在 generic `thread_distributed` /
    完整二维 tile 形状
- **根因**:
  - layout-derived generic distribution contract
    比 row reduction / row broadcast 的结构化证据更早落表，
    后面的专用语义没有覆盖前面的 generic truth
  - 资源 canonicalization 只改了 TIR body，
    没同步改 projection contract 的 `scope`
- **修法**:
  - `AnalyzeBlackholeComputeRegions`
    允许 row-reduction / row-broadcast evidence
    覆盖 generic `thread_distributed`
    为 `grouped_rows / row_state`
  - `buffer_distribution_contract.shape`
    只保留 logical distribution shape：
    `grouped_rows -> [row_width]`，
    `row_state -> [1]`
  - `BlackholeDeviceResourceCanonicalization`
    同步回写
    过渡 attrs 与 projection records
    的 `scope`
- **教训**:
  - 专用结构化证据必须能覆盖 generic layout truth，
    否则后段会重新掉回 matcher / fallback 思维
  - 过渡 projection attrs 只要保留旧 scope，
    就等于还在系统里保留一条旧链

### 2.3 ABI / schema

#### generic statement-access recovery 不能把 `tl.region` 里的 `BufferLoad` 当成真实 read，也不能退回 op-name 特判

- **症状**:
  - `BuildSpatialPlan`
    为了给 closure / dataflow
    恢复 read/write set，
    一边把
    `tl.region(..., access_mask="w")`
    的内部
    `BufferLoad`
    误记成 read，
    一边又用
    `tl.tileop.gemm_py`
    /
    `arg[2]`
    人工补写边
- **根因**:
  - `tl.region`
    是 transport bridge；
    真正的读写语义
    在
    `access_mask`
    上，
    不在它内部那层
    `BufferLoad`
  - 直接递归 visitor
    会把 write-only region
    误分类成 read，
    进一步诱导出
    `gemm` 专用修补
- **修法**:
  - statement access
    恢复改成：
    遇到
    `tl.region`
    直接按
    `access_mask`
    记 read/write，
    不再递归到内部
    `BufferLoad`
  - compute role /
    locality trait
    改成消费
    tileop typed
    `GetDataflowAccessInfo()`
    的
    `compute_consume`
    contract，
    不再按
    `tl.tileop.gemm_py`
    做 generic pass
    特判
- **教训**:
  - bridge op
    自己就是语义 carrier 时，
    consumer
    要读 bridge contract，
    不要把桥里面的实现细节
    当成 owner truth
  - “先让 visitor 递归跑一遍，
    再给特殊 op 打补丁”
    在 generic analysis
    里几乎一定会长成
    case-coupled residue

#### generic debug/source contract 不能按 workload-private buffer 名分支

- **症状**:
  - `codegen_blackhole`
    的 debug waypoint
    直接按
    `scores_max` /
    `acc_o` /
    `acc_s_cast` /
    `O_shared`
    等 buffer 名
    发不同 tag
- **根因**:
  - 调试 contract
    被绑定到了
    当前 flash-attn
    workload 的实例名，
    不是稳定的 op /
    phase /
    structural 边界
- **修法**:
  - 删除 workload-name
    分支，
    waypoint
    只保留 generic op-kind
    tag
    （例如
    `FILL` /
    `AFCB` /
    `CAST`）
- **教训**:
  - debug/source
    也属于 contract surface；
    一旦测试开始断言它，
    workload-private 名字
    就会反向固化成协议
  - 想保留可复用的 debug gate，
    就只能绑稳定结构，
    不能绑当前 kernel
    里那几个变量名

#### pipeline legality 不能只盯 `num_stages` 注解；annotation 消失后要从 stage-local buffer 反推

- **症状**:
  - 删除
    `pipeline_stage_counts`
    legacy bag
    后，
    `num_stages=4`
    的 GQA
    不再在 legality gate 处 fail-fast，
    反而晚到 residual validation 才炸
- **根因**:
  - body-side legality check
    只看 loop annotation 上的
    `num_stages`
  - 某些优化后形态里，
    stage count
    只能从 stage-local shared / CB buffer
    的 leading dimension 反推出
- **修法**:
  - legality check
    先读
    `num_stages / tl_pipelined_num_stages`
  - 读不到时，
    再从 stage-local buffer shape[0]
    直接推断
- **教训**:
  - 删除 legacy pipeline bag 时，
    fail-fast 语义必须同步回收到当前 TIR
  - 不能把“bag 删了”
    误写成
    “legality 不再需要”

#### GEMM reader 的 buffer 绑定不能让 stride runtime arg 覆盖 buffer address

- **症状**:
  - `test_blackhole_gemm_basic`
    和
    `test_blackhole_gemm_direct_runtime_materializes_compile_time_abi_schema`
    在 TT-Sim 上直接报
    `UndefinedBehavior: noc_cmd_ctrl ... src_addr=0x1`
- **根因**:
  - codegen 侧
    `buffer_runtime_arg_map`
    之前按
    `bound_buffer_name`
    盲收所有 runtime arg，
    后写入的
    `a_tile_stride / b_tile_stride`
    覆盖了真正的
    `A_addr / B_addr`
- **修法**:
  - 只让
    `input_buffer_addr{,32} / output_buffer_addr{,32}`
    这类 buffer-address runtime arg
    进入
    `buffer_runtime_arg_map`
- **教训**:
  - buffer identity 到 runtime arg 的绑定
    必须由 typed arg kind 决定，
    不能把同一 buffer 上的 stride / shape / address
    混成同一槽位

#### schema-only ABI 一旦成立，派生物也必须能从 schema 重建

- **症状**: strip 掉 legacy `accessors` 后，runtime 先报缺失 `buffer_materialization`
- **根因**: 物化信息仍只从 legacy accessor 路径推导
- **修法**: 从 `compile_time_arg_specs` 的 `buffer/layout/memory_space` 元数据恢复 materialization
- **教训**: schema 既然宣称自己是主路径，派生物也必须能从它单独重建

#### runtime / common-runtime arg 去重必须用 `identity:kind`

- **症状**: 同一 remote core 的 `logical_core_noc_x/y` 丢半边
- **根因**: 只按 `identity` 去重，把“同组对象的不同分量”合并掉了
- **修法**: dedup key 统一改成 `identity + ":" + kind`
- **教训**: `identity` 是分组标识，不是唯一字段

#### remote core 这种“多字段表达一个对象”的东西，应尽快上提成 schema object

- **症状**: runtime 侧长期从若干 runtime arg 手工重建 remote core
- **根因**: descriptor 没进 `KernelSpec`
- **修法**: 提升为 `KernelSpec.remote_core_descriptors`
- **教训**: 一旦多个字段共同表达一个对象，就别长期只留在 arg 列表里

#### synchronization schema 应在 spec / module build 边界校验，而不是留到执行期

- **症状**: `semaphore_binding` 缺失或 remote core x/y 不成对，只在 direct execution 时炸
- **根因**: semaphore 与 remote-core 解析散在多处 kind-switch，缺统一校验
- **修法**: 在 `ExecutableSpec` / `BlackholeModuleNode` 构造期统一校验
- **教训**: 只要已经进入正式 schema，对象合法性就应尽早 fail-fast

#### copy/dataflow 主路径不能退回默认 ABI

- **症状**: schema 缺失时仍然继续 build，到后段才报 buffer binding 缺失
- **根因**: 保留了 `input0/output0` 这类默认 runtime-arg fallback
- **修法**: 删除默认 fallback；schema 缺失 build-time 直接失败
- **教训**: 正式 ABI 不应该靠默认名字兜底

#### typed target truth reader 不能和 legacy projection fallback 共用同一套 getter

- **症状**: 原始 device build / codegen 看似已切到 `tl.tt_program`，
  但仍能因为 shared getter 的 fallback 静默吃到 `blackhole.*` attrs
- **根因**: `tt_program_projection` 同时承担了
  `TTProgram` direct reader 和 legacy attr fallback 两种职责
- **修法**: 拆成 `TTProgram`-only reader 与 synthetic/local attr helper，
  并让原始 device build 输入硬要求 `tl.tt_program`
- **教训**: 一旦 typed target truth 建立，generic projection helper
  不能再偷偷 multiplex 两套真源

#### leaf reader 所需 gate data 不能继续挂在 `blackhole.lowering_requirements`

- **症状**:
  - strip 掉 device func 上的
    `blackhole.lowering_requirements`
    之后，
    build/codegen 会丢
    `buffer_tile_bridge_specs`
    或静默放过
    unsupported compute subset
- **根因**:
  - leaf-only contract
    没有先进入
    typed `TTProgram`
    object /
    leaf schema
    和临时
    `tl.blackhole_executable`
  - build/codegen/runtime
    仍然直接消费
    `blackhole.lowering_requirements`
- **修法**:
  - 在 `PlanTTCompute`
    把
    `buffer_tile_bridge_specs /
     unsupported_compute_ops`
    先上提进
    typed `TTProgram`
    object；
    当前仍经
    `TTProgram.payload`
    暂存的字段
    只能按 leaf compatibility debt
    处理
  - `MaterializeBlackholeExecutable`
    再把它们投影到
    typed
    `tl.blackhole_executable`
    leaf schema
  - leaf reader
    统一改读 executable projection
- **教训**:
  - 只要字段需要越过
    `BuildTTProgram`
    继续活到 build/codegen/runtime，
    它就已经是 leaf contract，
    必须变成
    `TTProgram / ExecutableSpec`
    的显式 truth，
    不能继续寄生在 lowering attr 上

#### host/device symbol 对齐不能把优化后的 device body 回退成 source body

- **症状**: copy pipeline 在 codegen/build 阶段突然报
  `Find undefined Variable tile_row`
- **根因**:
  - Python 侧 symbol-align helper
    为了把 optimized device func 的 `global_symbol`
    对齐回 source 名字，
    直接返回了
    `source_func.with_attr("global_symbol", target_symbol)`
  - 结果把已经过 Blackhole lowering 的真实 device body
    换回了较早阶段的 source body
- **修法**:
  - 继续使用 optimized device func，
    只在它身上改 `global_symbol`
  - `global_infos` 也同步保留 optimized device module 的版本

#### optimized helper 若在 `OptimizeForTarget` 之后才补 logical bridge analysis，会丢 row-reduction bridge spec

- **症状**:
  - flash-attn / gqa
    通过 test helper
    走
    `OptimizeForTarget -> LowerToBlackholeTTProgram`
    时，
    `PlanTTKernelABI`
    在 grouped row reduction
    报
    `missing buffer_tile_bridge_spec for acc_s`
- **根因**:
  - helper 在 destructive optimize 之后
    才重新跑
    `AnalyzeBlackholeComputeRegions`
  - 这时局部 logical tile shape
    已经被 lower 成更晚的表示，
    无法再完整恢复
    `buffer_tile_bridge_specs`
- **修法**:
  - 像正式 `lower()`
    一样，
    在 `OptimizeForTarget`
    之前先跑
    `AnalyzeBlackholeComputeRegions(LowerToBlackholePhaseB(...))`
  - 只把最小
    `buffer_tile_bridge_specs`
    对齐回 optimized device func，
    再进入
    `LowerToBlackholeTTProgram`
- **教训**:
  - 任何 helper / test bundle
    只要绕开 canonical engine helper，
    就必须共享同一 pre-opt analysis capture point；
    否则 optimized path
    会先坏在 helper 漂移上
- **教训**:
  - symbol/name 对齐只能改 symbol；
    不能顺手把 owner object 一起换回旧版本，
    否则等于重新引入一条隐式旧链

#### staged-copy 的 transpose truth 不能只留在 GEMM contract；host materialization 也必须显式消费

- **症状**: `flash-attn` direct runtime 能执行但数值明显不对，
  实际结果更接近 `softmax(Q @ K) @ V` 而不是 `softmax(Q @ K^T) @ V`
- **根因**:
  - `multi_gemm_contracts` 已经知道 reader 侧 `transpose_B=1`
  - 但 host staged-copy / tilize materialization 只看 `host_axis_order`，
    没有显式的 tile 内 2D transpose truth
  - 对单 tile `K` 来说，只改 tile 索引顺序不会做 tile 内转置，
    最终仍会按未转置的内容喂给 compute
- **修法**:
  - 在 accessor/materialization schema 增加 typed `transpose_2d`
  - lowering 在 staged-copy reader 注册该 truth
  - runtime host tilize / readback 按 `transpose_2d` 做 2D transpose
- **教训**:
  - compute contract 里的 transpose 若还影响 host 传输/布局，
    就必须成为 accessor/materialization 的显式 schema 字段；
    不能指望 host 从 GEMM contract 侧推

#### bridge-stage target truth 不应再落成过渡 attrs

- **症状**: 想删除 TT kernel ABI planner 输出上的
  `blackhole.segment_plan / runtime_args / gemm_contract`，
  却被中间 bridge attr 或测试 fallback 卡住
- **根因**: target truth 先被落成
  `tl.tt_kernel_seeds / tl.tt_abi_plans / tl.tt_program_payload`
  这类过渡 attrs，后续清理就会被 attr 兼容面反向绑定
- **修法**: `BuildTTProgram` 直接聚合 planner result，
  不再物化 `tl.tt_*` seed attrs；随后继续把
  `blackhole.segment_plan / runtime_args / gemm_contract`
  这组 compatibility attr synthesis 一并删掉，
  helper/test 也只验证 `tl.tt_program`
- **教训**: producer-side 清理的真正前提不是“删代码”，而是
  bridge-stage 的 typed truth 已经能脱离 attrs 被独立消费

### 2.2 planner / runtime contract

#### `clear_accum=false` 不能直接等价成“必须走 merge path”

- **症状**:
  - fresh fragment 和 preclear zero-init GEMM
    明明没有真实 live-in accumulator state，
    却仍被物化成
    `intermediate_accumulator_merge`
    并在 TT-Sim 上打到旧 merge/live-form 桥
- **根因**:
  - lowering 之前直接从
    `gemm_py(clear_accum=False)`
    落 merge contract，
    没有结合
    `TIR execution order`
    去区分
    “真实 live-in state”
    和
    “只是 zero-init / fresh fragment”
- **修法**:
  - `buffer_materialization_contract`
    改为基于
    `TIR execution order + recurrence facts + future cast-consumer relation`
    生成
  - 只有
    recurrence/live consumer
    或真正的 prior live-in state
    才保留 merge contract；
    fresh / preclear-only
    统一 canonicalize 到
    `clear_accum=true`
  - 一旦 canonicalize 到
    `clear_accum=true`，
    还要继续删除
    紧邻 full-overwrite matmul
    的 selected
    `tl.blackhole.fill_fragment`
    zero-fill；
    只改 contract
    不改已选 builtin body，
    runtime 仍会掉回
    旧 live-form /
    `t_tile_mmio_wr32`
    边界
- **教训**:
  - accumulator merge
    是 producer/consumer 关系问题，
    应从 IR 事实推导；
    不能把 op flag
    直接当最终 lowering contract

#### partial-write output 必须先把 host 初值同步到 device

- **症状**: 单测单跑看似正确，整套顺序执行时 output 未覆盖区域读回脏数据
- **根因**: runtime 只初始化 input，不初始化 output device buffer
- **修法**: 执行前统一同步所有 host tensor 当前内容
- **教训**: 只要 schema 允许 partial write，output 初值就是 contract 的一部分

#### stick/page transport 需要显式 64B 对齐边界

- **症状**: TT-Metal NOC 报地址对齐错误
- **根因**: `transport_page_size`、offset 或全局宽度没有满足底层 page / alignment 约束
- **修法**: 把 `transport_page_size` 显式写进 schema，并在 lowering 阶段 fail-fast
- **教训**: transport 合法性要前移到 schema / lowering，不要留给 runtime

#### planner 缺 work plan 时，runtime 不能自动补默认 core / packet

- **症状**: planner/runtime contract break 被伪装成“还能跑”
- **根因**: spec 提取层和 runtime 都在补默认 work packet / fallback core
- **修法**: 删掉默认值；空 `work_packets` 直接 fail-fast
- **教训**: host/runtime 计划缺失时必须显式报错，不能补“最小可运行默认值”

#### `work_packets` 一旦允许 `work_count > 1`，direct runtime 不能再假设单波次 one-work-per-core

- **症状**: `512x512x512` pure GEMM 在 direct runtime launch 前就报
  `oversubscribed direct launch is not supported`
- **根因**:
  - planner 已经合法产出 `work_offset/work_count`
  - 但 runtime 先把 packet 扁平成多个 logical work item，
    再强制 `launch_cores.size() == work_items.size()`
  - 如果只是去掉这个检查，同一 core 的多次 `SetRuntimeArgs(...)`
    也只会留下最后一份参数，仍然不对
- **修法**:
  - direct runtime 改为按 `work_packets` 建 `launch wave`
  - 对无显式 `semaphore / remote-core` synchronization contract 的 executable，
    以 repeated launch 串行执行各 wave
  - 对带显式同步 truth 的 oversubscribed executable，继续 fail-fast
- **教训**:
  - `work_packets` 是正式调度 truth，不是展示用 metadata
  - 若 device kernel 还没有 per-core serial packet loop contract，
    runtime 至少要尊重 packet truth 做 wave scheduling，
    不能回退成“默认每 core 只跑一个 logical work item”的隐式假设

#### logical core 坐标和 physical / NOC 坐标不能混用

- **症状**: core lookup 失败、range 越界、launch/core 映射错位
- **根因**: planner 产出旧 physical-style 坐标，runtime 消费 logical worker grid
- **修法**: planner/runtime 统一到 logical worker grid；logical -> NOC 由 host materialize
- **教训**: core descriptor 必须明确语义，不能让两端各自猜

#### 缺失 typed access / accumulation contract 时，flash-attn direct runtime 必须 gate，而不是继续猜

- **症状**:
  - multi-work `flash-attn` case 会非法从错误 tile/page 地址读数据
  - single-work small `bf16` case 可能直接跑出全零或明显错误结果
- **根因**:
  - reader / writer 虽然已有 `a_tile_start_id / b_tile_start_id / output_tile_start_id`
    这类 ABI 描述符，但后段仍在按 `work_linear_id -> blockIdx`
    或“arg kind 恰好出现了”的局部规则重建访问语义
  - compute epilogue 仍含 `add_fragment_from_cb_front` scratch accumulation，
    且若没有对应的 typed fragment materialization truth，
    runtime 就会被迫在 lower 后的 builtin 序列上猜
- **修法**:
  - 在 `TTProgram -> ExecutableSpec` materialization 阶段
    把 `per_work_arg_specs` canonicalize 成 kernel-local truth，
    并让 codegen/runtime 都按 `value_kind` 消费
  - 在当前主链的
    `buffer effect/use-role analysis -> buffer liveness analysis ->
    planner decision -> compute_epilogue_ops`
    这条链上，
    显式 materialize generic `buffer_materialization_contract`
    （`intermediate_accumulator_merge / intermediate_buffer /
    accumulator_delta / accumulator_add`），
    不再把 `matmul` 这类 family 名字编码进 contract
  - 在 `ExecutableSpec` build 阶段追加
    `direct_runtime_unsupported_reasons`
  - 对缺失 kernel-local explicit per-work spec 的组合，
    报缺失 explicit per-work access descriptor
  - 对已 materialize fragment materialization contract、
    但 runtime 还未实现对应 materialize-then-merge protocol 的 kernel，
    显式报 unsupported；不要静默错跑
  - 同时把这些 unsupported reason 从 device spec 透传回 host metadata，
    让 Python/runtime gate 真正看得到
- **教训**:
  - 一旦 typed IR / ABI 已经暴露出 access 或 materialization/merge contract 的缺口，
    codegen/runtime 的正确动作就是 fail-fast 并把需求前移到上层 IR，
    不能继续执行会错跑的 heuristic path
  - 当前这条 gate 也不是“过度保守”：
    人为清空 `compute_epilogue_ops` 后，small `bf16` MHA 仍然错算
    （`max diff=1.2265625`, `mean diff=0.2021484375`），
    说明 fragment materialization/merge 的执行语义本身
    还没和真实 device protocol 对齐

#### thread-distributed fragment 的 layout truth 不等于 live-form truth

- **症状**:
  - real `lower()` 主链里把 fragment layout truth 投影回 device side 之后，
    `fragment_fill -> cast -> publish` direct runtime 仍输出全零
  - `clear_accum=false` merge 后继续给 cast consumer 的 case
    只会覆盖一小条 slice，当前采样
    `max diff=37.25`, `mean diff=8.8125`
- **根因**:
  - `OptimizeForTarget -> SplitHostDevice` 之后，
    `layout_map / tl.Fragment` 原始 truth 会消失；
    当前虽然已用 `tl.fragment_layout_seeds`
    把 layout truth 投影回 device side，
    但这只能说明 logical layout
  - 对 thread-distributed fragment，
    device-side `blackhole.acc` buffer 仍只是 per-lane physical slice，
    不是已经 materialized 的 full logical fragment。
    典型 case：逻辑 `32x32`，physical local extent 只有 `8`
  - 如果上游 contract 没有显式给出
    `live_form / execution_lane / physical_local_extent`，
    lower/codegen 就会继续犯两类错误：
    1. 按 logical extent 误用 per-lane physical buffer
       （例如 `fill_fragment` 把 `1024` 当作 `blackhole.acc[8]` 的 fill extent）
    2. 在 republish/cast bridge 里默认 lane-0，
       最终只 materialize 出单 lane slice
- **修法**:
  - 把 `buffer_distribution_contract` 扩成 owner-side live-form contract，
    至少显式带出
    `live_form_kind / execution_topology_kind / physical_local_extent`
  - 这层 truth 的 owner 应该是
    `Normalized Tile TIR + 更早层 semantic/spatial analysis`
  - `TTProgram / TT kernel ABI planner / codegen`
    只消费这份 typed truth 做 target materialization；
    `CB` overlap / reserve / push / pop 之类物理资源分析仍留在 target 侧
  - 当前 `SpatialPlan`
    侧已经补上
    `LiveValue` /
    `LiveValueEdge` /
    `MaterializationBoundary`
    骨架；
    regression 里应同时检查 logical shape
    和 physical local extent，
    不要再用 full logical matrix
    反推 device local array
    大小
- **教训**:
  - `layout truth restored` 不代表
    `fragment materialization protocol closed`
  - 只要 device-side live form 还是 per-lane distributed，
    就不能把 `blackhole.acc` 指针直接当成 full logical fragment 去线性读写

### 2.3 CB / synchronization / compute lifecycle

#### GEMM output / writer bridge CB 去重不能只看 `Buffer` 对象或 `buffer->data`

- **症状**: single-core GEMM direct runtime 里 compute 发布到一个 CB，writer 却在另一个 CB 上 `cb_wait_front`，最终稳定挂死
- **根因**: `C_local` 在 GEMM extract 路径和 writer / decl-buffer 路径上出现成多个逻辑等价但对象身份不同的 `Buffer`；若 requirement 去重只看 `Buffer` 或 `buffer->data`，同一逻辑资源会被拆成两个 CB requirement
- **修法**: `AllocateRequirementIndex` 去重要覆盖稳定的 logical buffer identity，并在较晚看到更强 `input/output` 角色时把已建 requirement 从 `intermediate` 升级成正确角色
- **教训**: planner / lowering 的 dedupe key 不能只依赖对象身份；只要 logical resource 能跨 pass / canonicalization 漂移，就必须保留稳定 identity

#### 新 builtin 只要带 cb_id，就必须注册回写位置

- **症状**: compute kernel 写错 CB，consumer 永远等不到数据
- **根因**: CB allocator 的 cb_id 回写位置注册表漏注册参数位置
- **修法**: 补注册，并加 post-condition guard
- **教训**: “新增 builtin -> 必须声明 cb_id 回写位置” 是正式协议，不是习惯

#### `blackhole.acc` 结果若会再喂 matmul，producer 侧发布页数必须按未来 consumer 算

- **症状**: 第二次 matmul 前挂在 `cb_wait_front` / `mm_init`
- **根因**: producer 只按当前 pointwise/cast 写入页数发布，没有按未来 matmul 需求 push_back
- **修法**: 预扫描 future matmul consumer，按其页数需求发布
- **教训**: scratch CB 的 producer 不只要“写进去”，还要按 future consumer 的协议正式发布

#### `blackhole.acc` GEMM 输出不能机械套 transport-CB reserve 模板

- **症状**: scratch CB 生命周期被破坏，compute hang 或错乱
- **根因**: matmul output path 无条件沿用 transport/output CB 的 reserve/push 模板
- **修法**: `blackhole.acc` 输出不再重复 reserve；按 scratch 生命周期处理
- **教训**: transport CB 和 scratch CB 不是同一类资源

#### 跨核 semaphore 握手必须下发真实 remote NOC 坐标

- **症状**: TT-Sim 在 enqueue 后挂死
- **根因**: device kernel 直接把 logical core 坐标塞给 `get_noc_addr`
- **修法**: host 用 `worker_core_from_logical_core(...)` 求真实 NOC 坐标后下发
- **教训**: remote route 信息必须 host-materialized，不能让 device 代码猜

#### communication builtin 不能单独携带 semaphore / routing 协议

- **症状**:
  - `get_semaphore(0)` 在没有 `TTSemaphorePlan` 时仍能 build/source
  - remote semaphore builtin 可以直接吃 literal NOC 坐标
- **根因**:
  - runtime/codegen 只看到了 builtin 序列，
    但没有把 communication protocol
    收回 explicit owner truth
- **修法**:
  - `get_semaphore`
    必须命中 planned semaphore id
    或显式绑定的 `semaphore_id_u32`
  - remote semaphore route
    必须命中
    `logical_core_noc_x/y + remote_core_descriptors`
  - oversubscribed direct runtime
    若带显式 communication contract，
    继续 fail-fast
- **教训**:
  communication builtin 只是执行表达，
  不是协议真源；
  不能让 literal 坐标、裸地址或 source-only builtin
  绕过 owner/runtime schema

### 2.4 analysis / lowering / gate

#### semantic-owned truth 缺失时，要回补更早层 semantic analysis，不要让 spatial/target 层借旧 attrs 自救

- **症状**: `row_reduction.kind` 缺失后，早层 reduce update truth 丢失，
  后续 spatial closure 会退化成单 phase
- **根因**: formal device 主链缺 semantic-owned fact
- **修法**: 在 manifest / fragment analysis / semantic lift 把 truth 补齐
- **教训**: 缺的是 semantic truth，就回更早层 semantic analysis 收；
  不要让 spatial / target 层临时绕回 raw attrs

#### `local/accumulator -> shared(CB)` bridge 应尽快变成正式 copy direction

- **症状**: compile-path 晚到 codegen 才报 residual shared store / undefined variable
- **根因**: fragment/local 结果写回 CB 的桥接语义仍以普通 `BufferStore` 漏到后段
- **修法**: 新增正式 copy direction / builtin，codegen 只消费 builtin
- **教训**: 对 Blackhole，`local` 只是中间态，不应长期作为最终资源语义

#### unsupported-op gate 不能只挂在一条出口

- **症状**: 一条路径按预期 fail-fast，另一条路径晚到 codegen 才炸 `undefined variable`
- **根因**: device-only codegen 绕过了 `ExecutableSpec` 路径上的 gate
- **修法**: spec 提取层和 codegen 入口共享同一套 gate
- **教训**: 只要仓库里有多条后端出口，shared lowering boundary 就要双边同时守住

#### kernel-local `per_work_arg_specs` 一旦漏掉，runtime/codegen 会重新吃 top-level stale descriptor 或 `work_linear_id` 反推语义

- **症状**:
  - `flash-attn` reader/writer 的
    `a_tile_start_id / b_tile_start_id / output_tile_start_id`
    重新掉回 `current_work_linear_id`
  - segment source 里的 block index 又开始从线性 work id 反推
  - grid-indexed copy
    即使删掉 kernel-local `per_work_arg_specs`
    也还能构建通过
  - `flash-attn` pipeline 多条 regression 一起变红
- **根因**:
  - `tt_program_projection::EncodeSegmentPlan`
    没有把 segment-local `per_work_arg_specs`
    round-trip 给 runtime/codegen reader
  - runtime/codegen 仍接受
    top-level `TTProgram.payload`
    或 `work_linear_id`
    作为兜底语义来源
- **修法**:
  - 只保留 kernel-local
    `per_work_arg_specs`
    reader 路线
  - multi-work kernel
    缺显式 per-work binding
    直接在 build/codegen fail-fast
- **教训**:
  - multi-kernel 和 single-kernel
    都要守同一条 host-truth 纪律；
    top-level aggregate/payload
    最多做摘要，不能再当 fallback 真源

#### fragment analysis 必须按结构 / 数据流识别，不能靠全局 op 扫描或名字匹配

- **症状**: copy/GEMM 被误伤成 `pointwise_chain`，或 MHA/GQA 的 row reduction / row broadcast 被漏掉
- **根因**:
  - 全局扫描 `tir.add/mul/div/max/...` 会把普通索引算术也算进去
  - 只识别 `CallNode`、只认 `floor_div`、或只认 split-after 某一种 IR 包装形态，都会漏真实 optimized path
- **修法**:
  - 只在 fragment/local region 自身的数据流里识别 pointwise
  - 同时识别 `AddNode/MaxNode/MulNode/DivNode` 等原生节点
  - 先剥掉无语义包装，再匹配 reduction / broadcast 形态
- **教训**: 对复杂 TIR，先看真实 IR 结构，再决定 matcher；不要把源码层直觉当 IR 协议

#### gate 应该按具体未支持子集收窄，而不是长期挡整类 blocker

- **症状**: `row_broadcast` / `pointwise_chain` 这种总括词掩盖哪些子集已可 lower
- **根因**: blocker 设计得太黑盒
- **修法**: 先吃掉稳定子集，再让 gate 随真实 lowering 一步步收窄
- **教训**: 细粒度 unsupported 集合比黑盒大类更有工程价值

### 2.5 低层基础设施

#### pass 拆分后，新 `.cc` 若没接进 `TILE_LANG_BLACKHOLE_SRCS`，会在 Python 导入时炸成共享库未定义符号

- **症状**: C++ 编译似乎通过，但 Python/pytest 一加载 `libtilelang.so` 就报
  `symbol lookup error: undefined symbol: BuildSpatialExecutionPlanForFunc(...)`
- **根因**: 新 split 出来的 translation unit 没被编进 `tilelang` 共享库，
  旧对象里只留下未解析引用
- **修法**: 把新文件显式加入 `tilelang_repo/CMakeLists.txt` 的
  `TILE_LANG_BLACKHOLE_SRCS`，重新 `cmake` + `cmake --build`
- **教训**: “文件已存在”不等于“目标已链接”；对 split pass，先用
  `nm -D libtilelang.so | c++filt` 确认符号真的进库

#### `TT_METAL_WATCHER` 改变症状时，先区分 direct runtime 回归还是 watcher 线程自己炸了

- **症状**: multicore GEMM direct call 在 `TT_METAL_WATCHER=10` 下于 `Dump #2` 前后 `SIGABRT`，或开 `TT_METAL_WATCHER_TEST_MODE=1` 后卡在同一 dump；但关闭 watcher 后 direct runtime baseline 仍能通过
- **根因**: native backtrace 落在 `tt::tt_metal::WatcherServer::Impl::poll_watcher_data()`，不是 `BlackholeModule` 主执行线程
- **修法**: 用 gdb / native bt 先确认 abort 源头；把 watcher-side failure 与 direct runtime regression 分开判断，正式 baseline 在 `TT_METAL_WATCHER` unset 的环境下跑
- **教训**: watcher 是调试器，不是真源。只要 watcher 改变了现象，先证明是 workload 坏了还是 watcher 自己坏了

#### 共享 protocol struct 必须只有一个定义

- **症状**: 改字段后随机崩溃、排序或字符串拷贝崩
- **根因**: 同 namespace 出现两份对象定义，布局漂移导致 ODR / ABI 错位
- **修法**: 共享协议 struct 集中到单一定义
- **教训**: 协议对象分叉定义迟早会炸成随机崩溃

#### `RemapBufferData` 之后，同源 Buffer 需要缓存，不能让 identity 漂掉

- **症状**: canonicalization 后下游去重或 `buffer_to_cb_` 查找失效
- **根因**: 对同一原始 buffer 多次 remap 产生多个不同对象
- **修法**: 在 remap helper 内缓存结果
- **教训**: 只要下游逻辑依赖 buffer identity，就必须保证 remap 后 identity 稳定

#### 不要对临时 `ObjectRef` 调 `CopyOnWrite()`

- **症状**: dangling pointer、随机崩溃
- **根因**: 临时 `ObjectRef` 析构后 COW 指针悬空
- **修法**: 不对临时对象做 COW；改为直接构造返回值
- **教训**: TVM object 生命周期问题会伪装成完全无关的崩溃

#### kernel 临时目录必须每次执行唯一

- **症状**: 同一 pytest 进程内 direct-call case 顺序相关、复用旧编译结果
- **根因**: TT-Metal JIT 复用固定临时路径
- **修法**: kernel 临时目录每次执行唯一化
- **教训**: JIT 缓存串扰首先要怀疑路径复用，而不是数值逻辑本身

#### flash-attn gate bypass 不能当作 direct-runtime admission

- **症状**:
  - 临时把
    `cast_fragment_slice_to_tiled_cb`
    放进 admitted publication protocol 后，
    executable projection 先在内部
    `acc_s_cast`
    materialization 上触发
    `host_buffer`
    为空的 assert
  - 继续临时绕过该 assert 后，
    small bf16 MHA
    能创建 reader /
    compute /
    writer kernels，
    但 TT-Sim 立刻报
    `UnimplementedFunctionality: t_tile_mmio_wr32`
- **根因**:
  - `acc_s -> acc_s_cast`
    只是第一个 typed gate；
    compute source 里仍有多处
    `tilelang_get_cb_write_ptr_bytes`
    /
    `CircularBuffer::get_tile_address`
    做 local-fragment <-> CB
    scratch staging
  - 这些 helper 依赖 mailbox /
    CB address exchange；
    TT-Sim hard execution
    不支持这条 MMIO path
  - 内部 live-form republish
    不能伪装成 host-buffer
    materialization 塞进 leaf
    `BufferMaterializationSpec`
- **修法**:
  - 保留 explicit unsupported gate
  - 后续 admission 必须先把内部 scratch
    local-fragment staging
    表达成 typed live-form /
    materialization /
    consumer-binding plan
  - publication 实现必须走非 mailbox、
    TT compute-linkable 的
    PACK /
    DST
    路径
- **教训**:
  - direct-runtime admission
    不能通过放宽 gate 验出来；
    gate-bypass probe
    只用于定位真实下游 failure，
    probe 后必须撤回并重编
  - 如果 generated compute source
    仍出现 mailbox-backed
    CB pointer helper，
    当前 TT-Sim bf16 correctness
    不能 admission

#### non-mailbox publication 后曾卡在 source live-form truth

- **症状**:
  - small bf16 flash-attn
    targeted compute source
    已不再调用
    `tilelang_get_cb_write_ptr_bytes`
    /
    `CircularBuffer::get_tile_address`
    /
    mailbox helper
  - 临时打开
    `cast_fragment_slice_to_tiled_cb`
    direct-runtime gate 后，
    TT-Sim 执行失败：
    `UnsupportedFunctionality: tensix_execute_gmpool: src_b_val=0x0 must be 1.0f`
  - 源码检查显示第一处 exact row-reduction
    的 source CB
    仍由 synthetic zero fill 发布，
    没有消费前面 matmul 产生的 CB-live value
- **根因**:
  - publication helper 已经不是主要 blocker；
    剩余问题是 source live-form /
    physical alias truth
    没有完整覆盖 exact row-reduction input
  - gate 放开会把 stale fill fallback
    伪装成 admitted runtime source，
    导致 simulator 在 reduce/gmpool 上首先报错
- **修法**:
  - direct runtime gate 保持 fail-closed
  - row-reduction input 必须从显式 live-form state
    绑定到 upstream matmul CB-live value
  - `2026-04-26` P2.1 收口：
    selected source-live producer 只由显式
    `M == 32 && N == 32`
    的 single full-tile matmul output
    种下；
    exact row-reduction source
    优先借用该 streamed CB-live value，
    并在 matmul 覆写时失效旧 fragment-fill fact。
    大 shape /
    thread-distributed 临时 tile
    不进入这个 admitted lane。
  - 不要把
    `cast_fragment_slice_to_tiled_cb`
    加入 admitted set
    作为 correctness shortcut
- **教训**:
  - “generated source 无 mailbox”
    只是 admission 的必要条件；
    source live-form truth
    和 stale fill invalidation
    也必须被验证
  - 当前 small / 32x32 bf16
    flash-attn direct-runtime subset
    已完成 admission；
    seq64 / multi-K-step
    是独立 multi-block correctness gate，
    不要把它重新描述成旧 P2.1
    live-form blocker。

#### flash-attn row scalar broadcast 方向不能按名字直觉选 `bcast_rows`

- **症状**:
  - exact softmax path 已经进入 tiled CB ops，
    但 TT-Sim bf16 结果明显偏离 reference
  - generated source 使用
    `mul_bcast_rows` /
    `add_bcast_rows`
    处理 row-reduction 后的 scalar
- **根因**:
  - TT-Metal 的 `BroadcastType::COL`
    才对应当前 flash-attn
    per-row scalar / column-vector
    broadcast 需求；
    按名字直觉使用 `bcast_rows`
    会把缩放维度搞反
- **修法**:
  - exact row-broadcast 和 exp2 row-broadcast affine
    改用
    `mul_bcast_cols_init_short` /
    `mul_tiles_bcast<BroadcastType::COL>` /
    `add_bcast_cols_init_short` /
    `add_tiles_bcast_cols`
  - TTProgram `operation_name`
    也同步写成
    `*_bcast_cols`
- **教训**:
  - broadcast 方向必须由 tile API
    的实际 operand semantics
    验证，
    不能只靠高层 buffer 名称或“row/col”直觉

#### flash-attn exact softmax 中间 CB 不能用 Float32 物理页作为 admitted BF16 lane

- **症状**:
  - P2.2 gate 打开后，
    small bf16 flash-attn
    可以跑到更深处，
    但输出出现 huge / inf 类错误
    或 simulator format failure
  - 参考 TT-Metal SDPA 路径的 softmax
    intermediate 使用 BF16 CB
- **根因**:
  - logical float32 exact value
    不等于当前 Blackhole direct-runtime
    admitted physical storage dtype；
    softmax exact tiled-CB lane
    用 Float32 page/data_format
    会偏离 TT-Metal admitted path
- **修法**:
  - 为 exact tiled-CB
    增加 physical storage dtype 选择：
    logical float32 softmax intermediate
    在 admitted direct path
    使用 `Float16_b` page/data format
  - GEMM ordinary output
    仍保持自身 dtype；
    只有 live-form exact CB
    走 BF16 storage
- **教训**:
  - direct-runtime admission
    的 dtype truth
    必须分清 logical value dtype
    和 physical CB storage dtype

#### standalone accumulating row-reduction 不能残留 fragment add fallback

- **症状**:
  - seq64 flash-attn pipeline source
    仍出现 unsupported `add`
    或 raw fragment add helper
  - `scores_sum += row_reduce(...)`
    这类 update 没有被 exact tiled CB pipeline
    完整接住
- **根因**:
  - matcher 只覆盖了直接 row-reduction，
    没覆盖 accumulator already-live
    的 standalone update 形态
- **修法**:
  - 为 row-reduction match
    增加 `accumulate_existing`
    语义
  - lowering 先 produce reduced CB，
    再用 typed exact
    `add_tiles` /
    `binary_max_tile`
    与 existing accumulator 合成
  - 已知 zero-fill accumulator
    可直接 canonicalize，
    避免多余 CB 占用
- **教训**:
  - recurrence/update 形态要进入 typed exact op，
    不能让 fragment helper 成为 fallback

#### exact CB republish 要区分总页数和单次 publish/consume 页数

- **症状**:
  - seq64 / multi-K-step flash-attn
    需要 multi-page CB capacity，
    但单次 publish/consume
    仍是 one page
  - stage2/block64 flash-attn
    会出现真正的 multi-page
    publish/consume event，
    仍应 fail-closed
- **根因**:
  - `num_pages > 1`
    只是 CB capacity；
    direct runtime admission
    的关键是
    `publish_pages_per_event`
    /
    `consume_pages_per_event`
  - one-page event 可以用已有
    wait / copy / pack / pop / push
    lifetime 证明；
    multi-page event
    需要更宽 live-form ownership、
    page lifetime
    和 consumer binding
    语义
- **修法**:
  - P2.3 compile/source/spec admission
    放行 seq64 /
    multi-K-step
    per-event one-page
    exact CB republish；
    direct-runtime correctness
    仍由
    `multi-block exact CB-republish flash-attention direct runtime correctness`
    typed unsupported reason
    gate 住
  - stage2/block64
    仍用
    `multi-page exact CB-republish live-form`
    queryable unsupported reason
    gate 住
- **教训**:
  - 不要用 CB 总页数判断 admission；
    要看每次 producer/consumer event
    的 page-count contract

#### borrowed exact CB live source 必须在下一次重写前消费并 pop

- **症状**:
  - seq64 flash-attn 第一轮
    `acc_s -> acc_s_cast`
    republish 后，
    第二个 K step
    可能重新写 `acc_s`
  - 如果旧 `acc_s` live source
    没有在重写前 `cb_pop_front`，
    后续 row-reduction /
    republish 会读到 stale page
  - 另一个相邻症状是
    `acc_s_cast`
    被 matmul 消费后，
    old deferred reacquire
    先 `cb_reserve_back`，
    后续 typed materialization writer
    又再次 reserve，
    造成 reserve/push 不配对
- **根因**:
  - future-use classification
    把下一次 write boundary
    附近的事件当成旧 live page
    consumer；
    实际上 write boundary
    是 redefinition
  - old reacquire mechanics
    仍假设未来 producer
    不会自己 reserve，
    但 typed materialization /
    live-form writer
    已经拥有
    `cb_reserve_back` /
    `cb_push_back`
    lifetime
- **修法**:
  - future live-CB read classifier
    只统计下一次 write 之前的 reads；
    write boundary 及之后不算旧 page consumer
  - borrowed source copy/repack 完成后，
    若下一次 write 前没有 read，
    立即 `cb_pop_front`
    并清掉 live-form alias
  - 对已有 typed materialization /
    tiled-CB live-form owner 的 buffer，
    禁用旧 deferred reacquire；
    让实际 producer writer
    自己 reserve/push
- **教训**:
  - exact CB live-form lifetime
    要按 producer/consumer event
    证明；
    不要让旧 early-reserve mechanics
    和 typed materialization writer
    同时拥有同一个 page lifetime

#### row-scalar division 不要走 scalar-only reciprocal SFPU macro 路径

- **症状**:
  - flash-attn row division
    若直接对 per-row scalar CB
    调 reciprocal，
    TT-Sim 可能命中
    `recip_tile<false>(VectorMode::C)`
    相关 SFPU macro / simulator boundary，
    或出现 scalar lane 为 0 的数值异常
- **根因**:
  - 当前 admitted path
    需要完整 tile 形态的 denominator
    才能稳定接入
    TT-Metal `recip_tile`
    /
    `mul_tiles`
    组合；
    scalar-only VectorMode
    不是这条 direct-runtime
    correctness gate
- **修法**:
  - 在
    `Normalized Tile TIR`
    中显式生成 leaf sequence：
    ones tile
    经 `mul_tiles_bcast_cols`
    构造 full-tile denominator，
    full tile
    执行 `recip_tile`，
    再用 `mul_tiles`
    完成 division
  - 不允许把这个 sequence
    隐藏在
    `mul_tiles_bcast_cols("div", ...)`
    或其他 leaf-looking composite payload
    后面
- **教训**:
  - 即便高层语义是 row scalar，
    admitted TT-Metal API 粒度仍应落在
    已验证的 tile op 序列上；
    不要为了追求“更小”粒度
    走 simulator 未覆盖的 scalar SFPU path
  - admission diagnostic
    不能把 normalizer /
    builtin coverage
    缺口直接说成 semantic unsupported；
    必须先区分
    `lowering_missing`、
    `backend_op_missing`
    和
    `admission_blocked`

#### exact-output live-form alias 必须随 tiled live-form 更新失效

- **症状**:
  - flash-attn seq64 / multi-K-step
    在 `acc_o` merge 之后可能在 TT-Sim
    卡住
  - 生成 source 中后续 compute
    会对旧 exact-output CB
    `cb_wait_front`，
    但该 CB page
    已在前一个 materialization /
    republish event 后被消费并 pop
- **根因**:
  - ordinary tiled live-form alias
    更新 / 清除时，
    没有同步清除同一 logical buffer
    的 exact-output live-form alias
  - 后续 exact compute
    优先复用了 stale exact-output source identity，
    把已经失效的 CB page
    当成当前 live producer
- **修法**:
  - `RecordTiledCBLiveFormAliases`
    和 `ClearTiledCBLiveFormIdentity`
    同步失效 exact-output live-form aliases
  - exact source selection
    只在当前 live-form identity
    仍有效时复用 exact-output CB
- **教训**:
  - exact-output alias
    是从当前 live-form 派生出的临时 truth，
    不是独立 owner truth；
    一旦 tiled live-form owner 改写或清除，
    exact-output alias 必须一起失效

#### Blackhole runtime module 不能用空 bytes 冒充 binary serialization

- **症状**:
  - `tilelang.compile(..., execution_backend="tvm_ffi")`
    的 Blackhole export path 需要 imported runtime module
    通过 TVM import-tree packing
  - 如果 `BlackholeModule` 声明 `kBinarySerializable`
    但 `SaveToBytes` 返回空 bytes，
    export 可能生成看似有效的 host shim，
    但 load/import 阶段没有可恢复的
    `ExecutableSpec`
- **根因**:
  - TVM `export_library`
    对非 DSO imported modules 会实际调用
    `SaveToBytes`
    并依赖
    `ffi.Module.load_from_bytes.<kind>`
    恢复 import tree
  - Blackhole 不能只靠 property mask
    通过 traversal；
    serialization bytes 和 loader 必须同属一个真实 contract
- **修法**:
  - `BlackholeModule::SaveToBytes`
    写出 versioned module payload、
    kernel dir
    和 typed `ExecutableSpec` map
  - 注册
    `ffi.Module.load_from_bytes.blackhole`
    读回同一 payload，
    并复用 `BlackholeModuleNode`
    构造校验
  - 文件级 `WriteToFile`
    在没有真实 file format 前继续 fail closed
- **教训**:
  - 对 TVM runtime modules，
    `kBinarySerializable`
    是 loadable import-tree contract，
    不是“允许 export 通过”的标签
  - 空 bytes / warning-return
    会把错误推迟到更远的 load/runtime 边界，
    应改成真实序列化或明确 fail closed

#### Blackhole direct runtime raw memcpy 必须先验证 DLTensor compact layout

- **症状**:
  - 非 compact stride 的输入或输出 tensor
    可能被 direct runtime 当成连续 buffer
    原样 memcpy，
    造成 silent wrong copy 或覆盖错误区域
- **根因**:
  - `DLTensor`
    的 shape 和 dtype 只能给出元素总量，
    不能证明 host memory layout compact
  - direct runtime 的当前 transfer path
    没有 stride-aware pack/unpack 实现
- **修法**:
  - host input transfer 和 output copy-back
    在 raw memcpy 前统一要求
    compact row-major layout
  - 非 compact tensor 先 fail closed；
    以后若要支持 stride，
    必须实现显式 stride-aware staging
- **教训**:
  - direct runtime 的 admitted subset
    要把 host tensor layout 写进边界条件；
    不要让 DLPack 的可表达 stride
    被低层 memcpy silently ignored

#### TTComputeOpPlan helper 漂移会伪装成 target runtime 回归

- **症状**:
  - target tests 在重建 mutated `TTComputeOpPlan`
    时失败：
    `Expected 21 but got 16 arguments`
- **根因**:
  - 生产侧 `TTComputeOpPlan`
    schema 增加了 tile-compute DAG /
    materialization /
    fanout fields
  - `testing/python/target/blackhole/common.py`
    的 rebuild helper
    仍按旧 16 参数构造对象
- **修法**:
  - 测试 helper 必须完整透传当前 typed plan fields，
    和 transform 测试中的 rebuild helper 保持一致
- **教训**:
  - 看到 FFI constructor arity mismatch
    先查测试 helper / schema drift，
    不要误判成 codegen 或 direct runtime 行为失败

#### Blackhole 32B bf16 page transport 会命中 TT-Sim NOC 对齐 fatal

- **症状**:
  - 将 staged stick copy 的 page transport 放宽到
    32B bf16 stick page 后，
    TT-Sim direct runtime 在执行 `noc_async_read`
    时 fatal：
    `noc_cmd_ctrl: read: alignment of src_addr=0x40 and dst_addr=... does not match`
- **根因**:
  - 当前 page transport 的 single NOC read/write
    需要 source / destination alignment 兼容。
    32B bf16 sub-tile stick page 会让 DRAM source 和 CB L1 destination
    alignment 不匹配。
- **修法**:
  - 保留 64B-aligned page transport admission。
  - bf16 sub-tile page transport 不作为当前 direct-runtime admitted path；
    要支持它必须重新设计 source/destination packing 或 NOC transfer
    granularity，而不是简单放宽 validator。
- **教训**:
  - page-indexed ABI 的 typed metadata 通过不等于硬件 transfer 合法。
    新 page size 必须跑 TT-Sim correctness；
    simulator fatal 不能被记录成普通 unsupported reason 后继续执行。

## 3. 环境问题速查

| 问题 | 解决 |
|------|------|
| `pip install -e .` 失败 | 用 `.pth` 指向本地构建产物 |
| Python 加载旧库 | 统一使用 `tilelang_repo/build/` 单一构建目录，并确认已重编 |
| TT-Sim 报 `Root Directory is not set` | 设置 `TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME` |
| TT-Sim 报 `No chips detected` | 设置 `TT_METAL_SIMULATOR` 与 `TT_METAL_MOCK_CLUSTER_DESC_PATH` |
