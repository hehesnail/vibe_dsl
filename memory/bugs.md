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
    归到后续 `R3.1+` payoff / support-surface backlog
  - 该问题的 simulator-side 旁证和更宽 fatal taxonomy 扫描，
    统一见 `memory/tt_simulator_constraints.md`

### direct cast consumer 的 direct-runtime 仍属于旧 merge/live-form bridge 边界

- **现象**:
  - `gemm + post-merge cast consumer`
    的 build/source contract
    已能稳定物化出
    `republished_logical_tile`
    consumer contract
  - 但 direct runtime / TT-Sim
    真执行仍会命中
    `t_tile_mmio_wr32`
    这类旧 merge/live-form bridge
    的 simulator/runtime 边界
- **当前结论**:
  - 当前 admitted 主链只把
    fresh fragment /
    preclear zero-init GEMM
    收到 `clear_accum=true`
    direct path
  - direct cast consumer
    仍保留 build/source contract gate，
    但不进入当前 TT-Sim correctness gate
  - 后续若要继续收这条 runtime，
    应补
    source live-form bridge
    的主链 owner，
    而不是继续在尾部堆 merge fallback

## 2. 已解决但值得记住的模式

### 2.0 grouped row / row-state distribution contract 不能让 generic layout 覆盖专用语义

- **症状**:
  - `flash-attn` / GQA 的 grouped `reduce_row` 会报
    `grouped_rows distribution contract` 缺失
  - companion contract 的 `scope` / `shape`
    可能仍停在 generic `thread_distributed` /
    完整二维 tile 形状
- **根因**:
  - layout-derived generic distribution contract
    比 row reduction / row broadcast 的结构化证据更早落表，
    后面的专用语义没有覆盖前面的 generic truth
  - 资源 canonicalization 只改了 TIR body，
    没同步改 companion contract 的 `scope`
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
    过渡 attrs 与 companion projection
    的 `scope`
- **教训**:
  - 专用结构化证据必须能覆盖 generic layout truth，
    否则后段会重新掉回 matcher / fallback 思维
  - companion attr 只要保留旧 scope，
    就等于还在系统里保留一条旧链

### 2.1 ABI / schema

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
    `TTProgram.payload`
    和
    `tl.blackhole_executable`
  - build/codegen/runtime
    仍然直接消费
    `blackhole.lowering_requirements`
- **修法**:
  - 在 `PlanTTCompute`
    把
    `buffer_tile_bridge_specs /
     unsupported_compute_ops`
    上提进
    `TTProgram.payload`
  - `MaterializeBlackholeExecutable`
    再把它们投影到
    `tl.blackhole_executable`
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

## 3. 环境问题速查

| 问题 | 解决 |
|------|------|
| `pip install -e .` 失败 | 用 `.pth` 指向本地构建产物 |
| Python 加载旧库 | 统一使用 `tilelang_repo/build/` 单一构建目录，并确认已重编 |
| TT-Sim 报 `Root Directory is not set` | 设置 `TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME` |
| TT-Sim 报 `No chips detected` | 设置 `TT_METAL_SIMULATOR` 与 `TT_METAL_MOCK_CLUSTER_DESC_PATH` |
