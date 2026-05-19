# 问题与 Bug 记录

> 本文档只保留仍有复用价值的问题模式。
> 阶段状态、总体 blocker 与完成判定以设计文档和 `tasks/progress.md` 为准。

## 1. 当前未解决

### 当前 TT-Sim fabric path 无法完成 multi-device CCL runtime correctness

- **现象**:
  - 按仓库固定入口 `scripts/setup_tt_sim.sh` 配置后，TTNN Python 默认只能
    发现 `1` 个设备。
  - 额外设置
    `TT_METAL_MOCK_CLUSTER_DESC_PATH=tt_metal_repo/tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_P300_both_mmio.yaml`
    后，TT-Sim 可以打开 `1x2` Blackhole mesh，`ttnn.get_num_devices()`
    返回 `2`。
  - 但最小 TTNN bf16 CCL probe 设置 `FABRIC_1D` 后，Fabric 在两个
    simulated devices 上初始化成功，随后在第一个 all-gather step 命中
    simulator fatal：
    `UnimplementedFunctionality: eth_txq_regs_wr32: eth_txq_cmd=0x2`。
  - 不设置 fabric config 时，同一 all-gather smoke 会失败于
    `Trying to get un-initialized fabric context`，不能作为 correctness
    fallback。
  - `TTSIM_SEMIHOSTING=1` 不改变该 fatal。
  - 临时测试上游 TT-Sim `v1.6.1` 的 `libttsim_bh.so` 仍命中同一
    `eth_txq_cmd=0x2` fatal。
  - 上游 `tenstorrent/ttsim` public tag 当前只发布二进制和文档；没有可在本地
    patch/build 的 simulator source。公开 issue 搜索也未发现 `eth_txq` / CCL
    / fabric workaround。
- **复现 / probe**:
  - `scripts/probe_tt_sim_ccl.sh`
  - 当前预期输出包含 `mesh_without_fabric_ok=true`、
    `mesh_num_devices=2`、
    `probe_status=fabric_ccl_unsupported` 和
    `unsupported_reason=eth_txq_cmd=0x2`。
- **当前结论**:
  - 原始 multi-device all-gather / reduce-scatter / all-to-all 的
    `BlackholeModule + TT-Sim bf16 + host reference` 正向 correctness
    在当前本地 TT-Sim 环境不可完成。
  - 2026-05-18 用户已把当前 T10.1/T10.2/T10.3 完成口径改为
    single-card multi-tile value semantics；当前 scoped gate 使用
    `scripts/probe_single_card_multitile_ccl_semantics.py` 和
    `tilelang_repo/testing/python/target/blackhole/test_blackhole_t10_single_card_multitile_ccl_runtime.py`，
    不是 fabric probe。
  - 后续若重新收口 multi-device fabric correctness，仍不能把 typed CCL
    contract、source/spec projection、fail-closed reject 当作完成；需要当前
    TT-Sim 支持 fabric `eth_txq_cmd=0x2`，或使用真实多设备 Blackhole 目标。

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

### `tilize_cast_fragment_slice` CB republish 在当前 TT-Sim 上会 PACR fatal

- **现象**:
  - bf16 flash-attn / paged decode 等路径能投出 typed
    `TTMaterializationPlan` / `ExecutableSpec.materialization_plans`：
    `materialization_protocol=cb_republish`，
    `publication_protocol=tilize_cast_fragment_slice`。
  - 若 direct runtime 放行，当前 TT-Sim 会在执行时进程级 fatal：
    `UnsupportedFunctionality: tensix_execute_pacr: intermediate_format=0 late_from_format=5`。
- **当前结论**:
  - 这不是精度容差问题；不能让测试跑到 simulator fatal，也不能把该路径算作
    runtime correctness 正例。
  - admission gate 应基于 typed materialization records，而不是 workload 名字。
  - 当前 direct runtime 通过
    `ExecutableSpec.direct_runtime_unsupported_reasons`
    fail closed：
    `tilize_cast_fragment_slice CB-republish direct runtime is gated: TT-Sim reports tensix_execute_pacr: intermediate_format=0 late_from_format=5 for the current fragment publication path`。
  - 后续若 TT-Sim / TT-Metal API 支持该 PACR 形态，删除 gate 前必须先恢复
    small MHA / GQA / paged decode runtime correctness 正向数值比较。

### T9.6 split-block flash decode 当前停在 TTProgram materialization validator

- **现象**:
  - `test_blackhole_t9_split_block_flash_decode_bf16_direct_runtime`
    当前 lower 阶段触发：
    `TTMaterializationPlan source_live_form must refer to boundary source_live_value`
    (`live_carry_acc_s_6` vs `live_carry_acc_s_12`)。
- **当前结论**:
  - 该 case 还没有进入 direct runtime / TT-Sim 数值比较；不能把它纳入
    runtime correctness admitted set。
  - 这是 TTProgram materialization-boundary owner-truth 问题，不应通过
    Python skip 或 runtime arg 猜测兜底。

### 单个 monolithic `T.gemm` 的超大 K 仍需要自动 temporal K lowering

- **现象**:
  - 单 tile `T.gemm(32xK @ 32xK)` 在随机 bf16 输入下，`K=128/256/512`
    与 torch reference 基本对齐；但 `K=1024/2048` 的 monolithic lowering
    会产生明显错误值。
  - all-ones 输入仍可对齐，说明这不是简单 tile index 或全局 reducer 的
    错误。
- **当前判断**:
  - 该路径一次 DST acquire/commit 中发出超过当前已验证窗口的
    `matmul_tiles` 序列。单纯把 input CB window 分段并不能修复，问题更像
    缺少 generated partial-C + reload continuation，或者需要在更早层把大
    K 自动拆成合法 temporal chunks。
- **当前边界**:
  - T10 large-MNK 已验证路径显式使用 `k_tile=256` 的 core-internal K
    chunks；这覆盖当前 admitted partial-K reducer correctness。
  - 后续要 claiming 任意 monolithic large-K GEMM shape-general correctness
    时，必须新增自动 temporal K lowering / partial reload guard，不能把
    当前 explicit `k_tile` 路径当成该能力已经完成。

### loop-carried input exact-CB backedge 在 TT-Sim 上需要 typed `pacr count=1` gate

- **现象**:
  - bf16 flash-attn seq128/256/512 source/spec 已能投出 loop-carried
    exact-CB virtual value、interval、allocation、release event，但 direct
    runtime 在当前 TT-Sim 上会进程级 fatal：
    `UnimplementedFunctionality: tensix_execute_pacr: count=1`
  - seq64 accumulator-only loop-carried exact-CB state 仍能 direct runtime
    正确执行，不应被同一个 gate 误伤
- **当前结论**:
  - admission gate 应看 typed ExecutableSpec：只有
    `loop_backedge_transfer` release 对应 input-role physical CB 时才加
    simulator boundary reason
  - 不要用 workload 名字、buffer 名字或 Python test skip 来兜底；先证明
    exact-CB lifecycle/source/spec admission，再由 runtime metadata 暴露 typed
    simulator reason
  - 如果后续 TT-Sim 支持该 PACR 形态，删除这个 gate 时要保留 seq64 正例和
    seq128/256/512 source/spec exact-CB metadata 断言

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

### GEMM direct-runtime broad suite can still hit TT-Sim/JIT execution boundaries

- **现象**:
  - `timeout 240s pytest -q testing/python/target/blackhole/test_blackhole_gemm.py --tb=short`
    reached the direct-runtime section and timed out without pytest summary
    after the non-runtime GEMM schema/source tests had already passed.
  - Earlier runs that forced accumulator reload paths could also surface
    TT-Metal JIT `undefined reference to cb_interface` failures.
- **根因 / 当前判断**:
  - The `cb_interface` form is the same compute-side CB interface boundary
    recorded above.
  - A separate lowering bug was fixed by not treating generic future writer
    transport or same-subject SpatialPlan self-edges as proof that a GEMM
    accumulator reload is required.
- **当前结论**:
  - For accessor/runtime ABI work, use copy/page/sharded direct-runtime bf16
    tests as the T4 gate.
  - For adjacent GEMM regression during T4/T5 setup, run the non-direct
    schema/source selection:
    `pytest -q testing/python/target/blackhole/test_blackhole_gemm.py -k 'not direct_runtime and not direct_call and not gemm_basic and not multicore' --tb=short`.
  - Do not convert the whole broad GEMM direct-runtime timeout into a T4
    external accessor blocker without first isolating a typed executable
    contract regression.

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

### exact-CB direct input copies 不能退回未初始化 local fragment republish

- **现象**:
  - Standalone leaf compute 的 binary / broadcast / unary 正例在宽
    `rtol=2e-2` 下没有暴露，但改成 absolute-only 后输出等价于丢掉 lhs：
    `binary_add` max diff `2.953125`，`binary_mul` max diff `6.71875`。
  - row-reduction 更明显，旧路径从未初始化 local fragment 发布输入，输出全 0，
    max diff `12.625`。
- **根因**:
  - tile-compute copy / materialization 在 `T.copy(A_local -> C_local)` 这类
    direct input alias 上没有继承 reader 已经发布的 typed input CB，而是从
    `ResolvePhysicalComputeBuffer(...)` 后的 local fragment 重新 tilize。
  - standalone row-reduction 的 input creation 也没有复用 transport-backed
    direct input CB。
- **修复**:
  - copy lowering 对 tile-aligned、transport-backed direct input source 使用
    `PrepareExactTiledCBRequirement(..., kInput)` 建立 live exact-CB alias。
  - fragment materialization 对 identity copy publication 追踪 direct-copy
    source 的 live/input CB，而不是从目标 local fragment republish。
  - row-reduction input creation 同样复用 transport-backed direct input CB。
  - `test_blackhole_leaf_compute_runtime.py` 改为 absolute-only gates：
    binary/unary/broadcast `atol=2e-2,rtol=0.0`；bf16 row-reduction 因 TT
    reduce 累加顺序与 torch row sum 有最多 `0.0625` abs diff，使用
    `atol=8e-2,rtol=0.0`。

### Copy runtime exact gate 暴露出的 stick / reshard / remote-core ABI 问题

- **现象**:
  - 将 `test_blackhole_copy_runtime.py` 中 copy 结果对比从
    `1e-3` / `1e-5` 收紧为 `atol=0,rtol=0` 后，
    `tall_stick_copy` 输出不是微小误差，而是把 `64x16` stick 写成了
    错误区域，max diff `4.570222854614258`。
  - 同一轮 full-suite 还暴露出两个非数值 contract gap：多 resident
    projected reshard 丢失 `reshard_plans`，手动 worker semaphore test 追加
    `logical_core_noc_x/y` runtime args 但没有 matching remote-core
    descriptor。
- **根因**:
  - writer-side full-tile normalization 只比较元素总数，
    `64*16 == 32*32` 被错当成 full tile。
  - CB allocator 复用不同 requirement name 的 physical CB，只保留第一个
    logical target name，导致 `resident_a` / `resident_c` 的 source binding
    和 reshard projection 被吞掉。
  - 测试 helper 构造半截 ABI，validator 正确拒绝缺失 descriptor 的
    logical-core NOC runtime args。
- **修复**:
  - full-tile normalization 现在要求实际 `shared_rows == 32` 且
    `shared_cols == 32`。
  - CB allocator 不再跨不同 requirement name 复用 CB，保留 typed target
    identity 和多个 projected reshard records。
  - copy runtime helper 从 `logical_core_noc_x/y` runtime args 派生
    `remote_core_descriptors`，保持手动重建的 ABI 完整。
  - `test_blackhole_copy_runtime.py` 全文件通过 exact comparison：
    `49 passed`。

### 多 tile per-work tile-compute local fragment 已改为 admitted runtime 正例

- **旧现象**:
  - T3 elementwise `block_rect_128x512` 使用 `tile_m=32,tile_n=64`，
    每个 per-work fragment 覆盖两个 `32x32` tiles。
  - 旧 direct runtime admission reasons 为空但 TT-Sim 输出整块为 0；
    max abs diff `0.94140625`、mean abs diff 约 `0.356`。
- **根因**:
  - exact input CB live path 只接受单个 full tile，导致 tile-aligned 多页输入
    落回 local fragment republish，而对应 local fragment 并没有按多 tile
    形态被正确建立。
- **修复**:
  - exact input CB creation 现在接受 tile-aligned multi-tile logical matrix，
    并直接消费 transport-backed input CB pages。
  - `test_blackhole_t3_compute_runtime.py` 不再 skip 多 tile case；
    `block_rect_128x512` 和 `block_large_1024x1024` 都作为 bf16
    `BlackholeModule + TT-Sim` 正例运行。

### Existing-TIR TopK repeated row-reduction page order 必须和 writer 消费顺序一致

- **旧现象**:
  - Existing-TIR TopK direct runtime 会重复 partial maxima，fp32 / bf16
    values 和 indices 都错；`M=64,N=128,k=6` 的 max value diff 为
    `0.0234375`，indices 不匹配。
- **根因**:
  - compute-side repeated row-reduction 发布输出页时按 `group -> repeat`
    顺序写 CB，但 TopK writer 按 `repeat -> group` 消费 value/index pages。
  - 这不是 bf16 误差问题；page order 错会直接重复旧 partial maxima。
- **修复**:
  - `EmitTypedReductionRegionIfSupported` 的输出 publication 顺序改为
    repeat-major。
  - TopK runtime 测试恢复为 admitted positive path；value 和 index 均 exact
    compare，unsupported reasons 必须为空。

### partial-K 大 shape 的 logical work grid 不能直接当 L1 shard grid

- **症状**:
  - `M=640,N=640,K=1024,k_shards=4` 对应 logical work grid
    `20x20x4`。如果把 C 的 sharded L1 grid 也设成 `20x20` 且
    `shard_shape=(32,32)`，TT-Metal buffer 创建阶段会要求 `400`
    shards，超过当前单卡 compute-core L1 bank capacity `130`。
  - 这个错误不能解释为 reducer 数值边界；它说明 resident L1 grid 给错
    了。
- **根因**:
  - Logical work grid 表示需要计算的 output tile 数；resident sharded L1
    grid 表示物理常驻 shard / bank 布局。两者不能默认相等。
  - 大 shape 应该让 resident grid 满足 core/bank 限制，并通过
    `shard_shape` 让每个 resident shard 覆盖多个 logical tiles；temporal
    work packets / launch waves 再覆盖完整 logical grid。
- **修法 / 验证**:
  - `20x20x4` 使用 C resident grid `10x10`、`shard_shape=(64,64)`，
    每个 resident shard 覆盖 `2x2` output tiles。
  - `test_blackhole_t10_partial_k_reducer_supports_large_temporal_output_grid_bf16`
    验证 logical grid 仍是 `20x20x4`、physical launch cores 是 `110`、
    C distribution 是 `10x10`/`64x64`，并通过 TT-Sim direct runtime
    与 torch bf16 reference 对比。
  - `13x10x4` 仍覆盖 temporal overflow wave `110..129` 的 reducer
    correctness；不要把这两个问题混成一个 allocator blocker。

### partial-K 更大 MNK 不能靠继续放大 logical output grid 或 full-output L1 shard

- **症状**:
  - 把“大 case”理解成继续放大 logical output grid，例如
    `40x40x8`，会绕开真正问题：这仍然是一 work item 负责一个 C tile。
    用户要求的是 MNK 本身变大，core grid 仍满足硬件/core 数限制，
    单个 core 内部再 temporal 覆盖多个 M/N output tiles 和多个 K chunks。
  - 如果把 full C 继续放进 sharded L1，通过增大 resident shard 来覆盖
    更大 full-output tensor，容易撞上 L1/CB 工作区重叠或超过单卡 L1 bank
    mapping 能力。这是 placement/resource 问题，不是 reducer 数值问题。
- **根因**:
  - Large MNK 的正确表示是 bounded logical/core grid + core-internal
    tiling。logical grid 表示并行 work item 数，不能直接替代每个 work
    item 内部需要覆盖的 output-tile set。
  - 当 full C 不能作为 sharded-L1 resident tensor 放下时，full-output
    owner 应该是 interleaved DRAM；L1 只承载 kernel staging/CB 工作集。
- **修法 / 验证**:
  - 大 MNK 的 full-output owner 走 interleaved DRAM output/scratch
    reducer：target distribution 是 interleaved DRAM，scratch layout /
    memory space 跟 target 一致，route kind 是
    `local_same_device_interleaved_tile`。
  - runtime 对 interleaved DRAM partial-K reducer 在每个非零 producer
    shard 完成后，把完整 scratch C buffer 以 float32 加到 final C，
    而不是假设一个 work item 只写一个 output tile。
  - `test_blackhole_t10_partial_k_reducer_supports_core_tiled_large_mnk_bf16`
    验证 `M=N=512,K=2048,k_shards=4`，logical/core grid 是 `4x4x4`，
    每个 core 写 `4x4` output tiles，每个 K shard 由两个 `k_tile=256`
    chunks 组成，并通过 TT-Sim direct runtime 与 torch bf16 reference 对比。
  - `test_blackhole_t10_partial_k_reducer_supports_full_core_core_tiled_large_mnk_bf16`
    验证 `M=640,N=704,K=2048,k_shards=4`，logical/core grid 是
    `11x10x4`，使用 `110` 个 unique Blackhole physical cores 和 `110` 个
    work packets 覆盖 `440` 个 logical producer work items，每个 core 写
    `2x2` output tiles，并通过 TT-Sim direct runtime 与 torch bf16
    reference 对比。这个 guard 不应继续使用宽 `rtol=0.2`；当前改为纯
    absolute gate `atol=0.1,rtol=0.0`，最近一次 full-core run 的
    max/mean/p99/p999 abs diff 是
    `0.083786` / `0.010080` / `0.037431` / `0.051130`。

### `clear_accum=false` 的 core-internal K chunk 不能把 Float32 accumulator live form 降成 bf16

- **症状**:
  - `M=N=64,K=512` 被拆成两个 `k_tile=256` GEMM chunk 时，单个
    `T.gemm(K=512)` 路径 max/mean abs diff 只有约
    `0.000023` / `0.000003`，但 `clear_accum=false` chunked 路径一度达到
    max/mean/p99 `0.132195` / `0.019104` / `0.104960`。
  - executable metadata 里第一个 live-form partial C CB 被投成
    `Float16_b`，而 scratch partial 是 `Float32`。
- **根因**:
  - compute-only tiled-CB output dtype 选择把
    `ExactTiledCBStorageDType(Float32)` 当成 live-form storage dtype，导致
    Float32 accumulator partial 在 `clear_accum=false` continuation 之前
    被写成 bf16。
  - 后续 continuation 还走“当前 partial + previous partial”的 merge path，
    而不是把 previous partial reload 进 DST 后继续 `matmul_tiles`。
- **修法 / 验证**:
  - Float32 GEMM accumulator live-form CB 保持 Float32 storage。
  - final transport continuation 在无需保留 local state、无需 cast、无需复用
    loop-carried live-form CB 的 common path 上，使用 partial reload
    continuation，避免把 previous partial 当普通 bf16 tile 合并。
  - `test_blackhole_gemm_clear_accum_false_preserves_float32_accumulator_bf16`
    断言两个 GEMM op 的 `c_tensor_dtype` 和 `c_cb_dtype` 都是 `Float32`，
    并通过 TT-Sim direct runtime；修复后该 repro max/mean abs diff 是
    `0.031204` / `0.004421`。
  - 同一修复把 full-core `M=640,N=704,K=2048,k_shards=4`
    partial-K reducer guard 改善到 max/mean/p99 abs diff
    `0.083786` / `0.010080` / `0.037431`，并把 gate 收紧到
    `atol=0.1,rtol=0.0`。

### core-internal tiled GEMM 不能跨 serial loop 盲目 retain input CB pages

- **症状**:
  - core-tiled large-MNK partial-K case 初始可以执行完，但输出和 torch
    reference 明显不一致。
  - compute segment 里 reader 为每个 `local_x/local_y` output tile 推入新的
    A/B tile pages；compute 却把 `cb_pop_front(0/1, 8)` 延迟到外层
    `local_y` 之后，导致后续 `local_x` 重复消费旧 B tiles。
- **根因**:
  - repeated serial loop 不是 input CB page loop-invariant 的证明。
    旧 retained-input 逻辑只看到 compute body 里有 serial loop，就把输入
    pages 留到 loop suffix 再 pop；但 core-internal tiled GEMM 的 reader
    events 已经表达了每个 local output tile 都有新的 A/B window。
- **修法 / 验证**:
  - 取消该隐式 serial-loop input retention，回到显式 reader/compute
    event 驱动的 per-consume pop/reacquire 协议。
  - 同一个 core-tiled large-MNK runtime case 从数值 mismatch 变为通过；
    T10 partial-K/CCL focused selector 同时报告 `9 passed`。

### Remote core endpoints cannot be recovered from logical_core_noc ABI pairs

- **症状**:
  - `KernelSpec.remote_core_descriptors`
    曾经在 `TTProgram -> ExecutableSpec` projection 时从
    `logical_core_noc_x/y` runtime args 配对生成。
  - leaf reader 虽然会要求 descriptor 存在，但 endpoint owner truth
    实际仍藏在 ABI arg pair 里。
- **根因**:
  - runtime arg 是 ABI value binding，不是 synchronization endpoint
    record。把 x/y arg pair 当 descriptor 来源会让 projection 层恢复
    TT sync 语义，违反 P0 target execution contract。
- **修法**:
  - 新增显式 `TTRemoteCoreDescriptorSpec`。
  - `ValidateTTProgram` 要求所有 `logical_core_noc_*` runtime args
    引用 matching descriptor，并校验 core 坐标一致和 x/y 成对。
  - `TTProgram -> ExecutableSpec` 只从 descriptor records 投影
    `KernelSpec.remote_core_descriptors`；缺 descriptor 时 fail closed。
- **验证**:
  - remote descriptor recovery 负例、descriptor materialization 正例、
    logical_core_noc unpaired / missing descriptor 负例，以及 P0 source
    guard 均通过。

### T7/T9 online-softmax runtime failure was backend live-form/codegen, not TT-Sim `t_tile_mmio_wr32`

- **症状**:
  - T7 seq64 flash-attn exact-CB partial combine initially timed out or
    mismatched against the host reference, and the same full online-softmax
    path affected T9.2 paged GQA and T9.3 paged MLA.
  - An intermediate diagnosis classified these as a typed TT-Sim
    `t_tile_mmio_wr32` simulator gate; that classification was wrong for the
    current runtime correctness boundary.
- **根因**:
  - Source codegen treated an empty `threadIdx.x` guard as a live use and
    serialized the whole tile-compute body under a 128-iteration loop.
  - Accumulating GEMM live reload preferred a stale buffer live-form alias over
    the newer `ExactOutputLiveForm`; after `acc_o *= scores_scale`, final merge
    read the unscaled old accumulator CB instead of the scaled exact-CB output.
- **修法**:
  - Codegen now ignores no-op branches when deciding whether a thread var is
    live and drops no-op emission pieces, so tile compute is not wrapped by an
    empty thread loop.
  - Accumulating GEMM reload now checks `TryCreateExactOutputLiveTiledCBValue`
    before older buffer-live aliases, so the latest producer-owned exact output
    is consumed by the merge.
- **验证**:
  - `test_blackhole_t7_seq64_mha_bf16_exact_cb_partial_combine_direct_runtime`
    passed through `BlackholeModule` TT-Sim correctness.
  - T9 page-addressed QK/AV page1, T9.2 full paged GQA, T9.3 dual-score MLA,
    T9.3 full paged MLA, and T9.1 grouped GEMM direct-runtime selectors passed.

### Physical CB queue replay must model cross-kernel producers

- **症状**:
  - A first executable source gate that replayed compute-kernel CB events from
    an empty queue rejected valid T3 sharded elementwise/reduce runtime cases:
    `physical CB queue wait_front exceeds visible pages in main_kernel_compute`.
- **根因**:
  - Some compute-visible CBs are not `role=input` but are still produced by a
    non-compute kernel before the compute kernel runs.  T3 reader materializes
    resident/intermediate CBs and the compute kernel legitimately starts with
    `wait_front` on them.  Replaying only the compute kernel without importing
    projected reader `cb_push_back` evidence treats those pages as missing.
- **修法**:
  - Derive externally produced CB IDs from projected non-compute
    `cb_push_back` events and allow compute `wait_front` / `pop_front` on
    those CBs while still checking the requested page count against physical
    CB capacity.
  - Keep exact-CB producer/release/storage checks in `ValidateTTProgram`; the
    executable queue gate now consumes structured `KernelSpec.queue_events`
    instead of parsing generated source text.
- **验证**:
  - Focused typed verifier tests reported `9 passed`.
  - T3 sharded elementwise/reduce mix plus T7/T9 direct-runtime selectors
    reported `16 passed`.

### T3 staged-copy reshard hardening exposed multi-record ABI and executable validation gaps

- **症状**:
  - 单个 copy 的 `interleaved_to_sharded` runtime case 能过，但同一个
    fused dataflow segment 内出现两个 independent copy/reshard record 时，
    codegen 只给最后一组输入/输出绑定 runtime buffer args。
  - 修改 `ExecutableSpec` 中的 tensor memory config / reshard records 后，
    build/runtime 没有全部 fail closed。
  - flash-attn 邻近回归暴露出 executable reader 过度要求所有
    buffer distribution 都有 positive page size，以及 `fill_tile` 被记录
    为 unary input/output op 的类型错误。
- **根因**:
  - fused dataflow ABI 仍按 single-copy surface 保存
    `copy_input_buffer_name_` / `copy_output_buffer_name_`，accessor slot
    也硬编码到第一组 read/write slot。
  - executable placement validation 只检查局部字段存在，没有把
    `TTTensorMemoryConfigPlan`、`TTReshardPlan` 和 indexed
    `TTBufferDistributionPlan` 交叉校验。
  - replicated local L1 intermediates 不是 runtime-visible page address
    ABI；它们的 storage/page ownership 可以来自 CB/materialization plan，
    不能被当成 interleaved/sharded runtime distribution 一样强制
    positive page size。
  - `fill_tile` 只有 output operand，不是 unary input/output op；
    copy-only live-CB republish/pack path 也不应该插入
    `unary_op_init_common`。
- **修法**:
  - fused dataflow 记录所有 input/output buffer identities，并为同一
    segment 内的 accessor 按 buffer identity 分配稳定 slots。
  - 多 target CB materialization 按 CB requirement order 绑定 source，
    避免把合法的多 resident target 全部标成 ambiguous。
  - `BlackholeModule` / executable reader 交叉验证 tensor memory config、
    reshard record 和 buffer distribution 的 subject、index、layout、
    buffer type、source binding 与 source region。
  - executable page-size 检查只强制 interleaved / sharded
    runtime-visible address distributions；replicated local L1 记录允许无
    page-size。
  - `fill_tile` 使用独立 `fill` compute kind；copy-only republish/pack
    去掉无语义的 unary init。
- **教训**:
  - T3 runtime gate 必须覆盖多 record、serialization、corrupted executable
    records 和邻近 workload regression。只测一个小 copy case 会漏掉
    single-record 假设和 leaf-reader 过度校验。

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
  - page-addressed ABI 的 typed metadata 通过不等于硬件 transfer 合法。
    新 page size 必须跑 TT-Sim correctness；
    simulator fatal 不能被记录成普通 unsupported reason 后继续执行。

#### broadcast-cols rank-1 RHS 不能用 scalar NOC 读散写 tile 位置

- **症状**:
  - standalone `add_tiles_bcast_cols` / `mul_tiles_bcast_cols`
    direct runtime 若按每个 bf16 标量直接 NOC 读到 tile 第一列位置，
    TT-Sim 会报 NOC address alignment mismatch。
- **根因**:
  - 当前 NOC transfer path 需要源地址和 L1 目标地址的对齐关系稳定；
    rank-1 bf16 scalar/short page 直接散写到 tile layout 的 first-column
    element address 不是 admitted transfer granularity。
- **修法**:
  - reader 先把 rank-1 RHS 作为一个对齐 page 读到目标 CB tile 尾部
    scratch 区，再在 BRISC 本地清零 tile 并 scatter 到 first-column
    nfaces 位置，最后清掉 scratch 区。
- **教训**:
  - broadcast 语义可以是列向量，但 runtime transport 仍必须选择硬件
    transfer 合法的 page 粒度；不要把 scalar shape 直接等同于 NOC
    transaction shape。

#### broadcast-cols reader source copy 不能在 vector/thread loop 内重复 publish

- **症状**:
  - broadcast-cols standalone leaf direct runtime 在 reader 阶段挂住；
    source copy 位于 `tx < 32` 这类 vector/thread loop 中时，
    reader 会对同一个 RHS CB page reserve/push 多次，而 compute 只消费
    一次。
  - T9.5 后续相邻验证又暴露了同类问题的完整 tile 版本：
    `A_local` full-tile source publication 落在 `tx` loop 内，reader 重复
    push A CB，compute 仍只消费一次，enqueue 卡住。
- **根因**:
  - rank-1 RHS materialization 是一次 per-work tile source event，不是每个
    vector lane / thread lane 的独立 transport event。
  - full-tile DRAM source publication 也是同一个 per-work CB event；thread
    lane loop 是执行组织，不是额外 transport event 粒度。
- **修法**:
  - 对 broadcast-cols source copy 使用当前 loop/thread guard，只在所有
    active lane var 为 0 时执行 reader-side materialization。
  - 对 thread-lane 内生成的 full-tile DRAM-to-device publication 同样加
    active-thread-zero guard，但只在该 source publication feeds tile
    compute 时这么做；纯 copy 的 reader/writer 成对事件必须保持相同执行
    粒度，否则 writer 会在后续 thread lane 等一个没有发布的新 CB page。
- **教训**:
  - 把 scalar/vector source loop lower 成 CB publication 时，必须重新确认
    publication event 的粒度；CB event 粒度错了会表现为 runtime hang，
    不是数值错误。

#### compute-only terminal publish 和 Int32 row-max reduce 的 TT-Sim / LLK 边界要 typed gate

- **症状**:
  - standalone `reduce_tile` bf16 direct runtime 曾命中
    `UnimplementedFunctionality: tensix_execute_pacr: count=1`；后续确认这是
    Blackhole lowering 初始化 PACK/UNPACK 格式状态不足导致的 emitted-sequence
    bug，不是 bf16 row reduce 整体不支持。
  - compute-only terminal publish 也会命中同类 TT-Sim pack/publish
    capability boundary 或输出不可靠；`fill_tile` / `typecast_tile` 是当前
    已知 witness，不是 gate owner。
  - T6 existing-TIR value/index selection 继续执行到 index 侧时，
    `T.reduce_max(expand_max_idx, max_idx, dim=1)` 会以
    `Int32 reduce_tile<MAX, REDUCE_ROW>` 形式进入 TT-Sim，并命中
    `tensix_execute_pacr` 的 format 组合 unsupported。
- **根因**:
  - bf16 standalone row-reduce 的根因是 compute-side scaler fill/pack 前未先
    做 `binary_op_init_common`，PACK/UNPACK data-format state 没被设置；修正后
    leaf `reduction_sum` direct runtime 能通过。
  - `Int32` index-side row max 的根因不同：TT-Metal 公开 reduce / topk /
    argmax 形态没有把 row-wise value/index selection 表达成
    `Int32 reduce_tile<MAX, REDUCE_ROW>`。相关支持面是
    `max_reduce_with_indices`、argmax reader/dataflow scan，或
    `topk_local_sort` / `topk_merge` / `topk_rebuild` 这类 topk-family compute
    primitive。
  - 这不是 `topk` 作为算子不支持。现有 T6 输入是普通 Tile TIR
    selection 结构，当前卡住的是 index 侧 row max 被 emit 成了错误硬件形态。
  - Flash attention 里的 row reduce 是已 admitted 的不同形态：
    GEMM-produced stream CB 进入 `reduce_tile`，再被 softmax / broadcast /
    matmul 链继续消费。small bf16 flash-attention direct runtime 能通过，
    所以不能把问题归纳成 Blackhole reduce 一概不支持。
  - 2026-05-02 修掉了先前混在同一个 gate 下的结构问题：
    reduce exact live form 绑定到了临时 `local.fragment` 输出，writer 因而
    等另一个 rank-1 logical buffer CB；writer 还会按每个 active lane
    wait/pop，并且 segment extractor 会丢掉只含 barrier/pop 的 writer
    内层 sync if。
  - 2026-05-02 的对照实验显示，把 `reduce_uninit` 放在 `pack_tile`
    前后两种序列都会在 TT-Sim direct runtime 命中
    `tensix_execute_pacr: count=1`；这个 fatal 不是单次调序引入的。
- **修法**:
  - reduce live form destination 从当前 IR 的 direct-copy source、logical
    shape、dtype 和 DRAM writer use 唯一证明，绑定到 writer 实际读取的
    logical local buffer；不靠 `C_local` / `C_local_1` 命名。
  - rank-1 exact tiled CB writer 对 full-tile reduce output 只 wait/pop
    一次，按 logical row 生成 scalar page id，并使用 tiled row L1 byte
    offset 写回 host rank-1 输出。
  - codegen thread emission 用完整 Stmt/Expr visitor 判定 thread var
    使用，segment extraction 在父节点已判定 segment 时继承上下文，保留
    writer final barrier/pop。
  - 保留 typed leaf records，并在 `ExecutableSpec` direct-runtime reasons
    中对 compute-only terminal publish 和 standalone `Int32` row max
    `reduce_tile` fail closed。不把 bf16 standalone reduce 或 GEMM/flash-attn
    内已经 admitted 的 compute chain 全局 gate 掉。
  - T6 正向修复从现有 TIR 的 value/index dataflow lowering 到 backend
    typed value/index scan；不能通过新增 frontend `topk` op、
    `selection_plans` 或 source-name recovery 绕过。
- **教训**:
  - simulator capability boundary 应写成 queryable typed unsupported reason；
    不要把它伪装成 semantic unsupported，也不要为绕过它新增旧 matcher /
    runtime guessing path。
  - row reduce 诊断必须区分 dtype、reduce kind、消费链和硬件 primitive；
    flash attention 的 float/bf16 row reduce 可运行，不能推导出
    `Int32 reduce_tile<MAX, REDUCE_ROW>` 也合法。

#### T3 sharded compute runtime exposed exact-CB event/role and grouped-broadcast gaps

- **症状**:
  - 多核 sharded resident-L1 elementwise chain 读到每个 core 都从 tile 0
    开始的数据，runtime correctness 失败。
  - full-tensor division 先被误判成 broadcast division，后续 reciprocal
    临时 exact-CB 只覆盖一个物理 fragment tile，32x64 等 case 输出不全。
  - mixed `elementwise + reduce + elementwise` 中 writer 可能消费 reduce
    中间值所在的同一个硬件 CB，而不是最终输出。
  - 修复 reduce materialization 后，flash-attn 暴露 `acc_o`
    accumulator merge 把 `tiled_cb_republish` fact 按
    `dst_cb_binary_pack` 路径断言，以及 `acc_o[i] / logsum[i >> 2]`
    这类一维分组 broadcast 未归一化导致残留 cast。
- **根因**:
  - 非 GEMM staged-copy reader ABI 仍把 per-work tile start 当常量 0，
    没有绑定到 `work_linear_id`。
  - division matcher 没区分 full-tensor identity RHS 和 grouped
    broadcast RHS；unary temp shape 又从当前 physical fragment 推导，
    没继承 input live-CB 的 logical coverage。
  - exact-CB `num_pages` 被误当成 event size / logical shape，导致
    compute wait 多于 reader push，或把 rank-1 logical output 膨胀成
    full tile logical shape。
  - accumulator merge 对 materialization fact 的校验没有按 reload 来源
    分支；有 live tiled-CB reload 时，`tiled_cb_republish` 是合法 live
    form，不是 local-fragment binary-pack reload。
- **修法**:
  - 非 GEMM reader start-id ABI 改为从 `work_linear_id` 取值。
  - `sub_tiles` 纳入 typed leaf pattern / builtin / codegen；full division
    降成 `recip_tile + mul_tiles`，broadcast division 只接受结构化
    broadcast load。
  - exact-CB live value refinement 只用 publish/consume event 和 logical
    live value tile count，不用 capacity `num_pages` 扩 logical shape；
    writer 消费 live exact-CB 时把该 CB 角色提升为 output。
  - grouped one-dimensional broadcast 识别 `i / group` 和
    `i >> log2(group)` 这类结构，并从 output/RHS 元素数推出 group 宽度。
  - accumulator merge 只在需要 local-fragment reload 时要求
    `dst_cb_binary_pack`；zero reload 或 live tiled-CB reload 不套用该
    断言。
- **教训**:
  - runtime correctness 要覆盖连续 compute 链和 reduce 后继续消费的链；
    单 leaf 或 projection-only tests 很容易漏掉 CB reuse、writer role、
    event-size 和 per-core tile-start 问题。
  - exact-CB 的 capacity、event size、logical value shape 是三件事；
    混用会表现成 hang、错 CB、或 rank-1 writer 结构回归。

#### Output CB front-retention rewrites broke multi-page tile writers

- **症状**:
  - T3 `block_rect_128x512` bf16 elementwise chain produced a stable mismatch:
    the writer emitted two `cb_wait_front(output_cb, 1)` events for the
    32x64 output but only one `cb_pop_front(output_cb, 1)`.
  - Generated writer source wrote both output tiles from `get_read_ptr(cb)`,
    so the second output tile reread the first FIFO page.
- **根因**:
  - `RetainLocalCBFrontForFutureWaits` delayed a pop when it saw a future
    wait before the next producer.  That is unsafe for writer-visible output
    CBs because `write_tile_from_cb` has no page-offset argument; retaining the
    FIFO front changes which page the later write observes.
- **修法**:
  - Do not apply generic front-retention pop rewriting to `role=output` CBs.
    Structured `KernelSpec.queue_events` now include a writer regression that
    requires output wait/pop page balance for the admitted tile writer surface.
- **验证**:
  - `cmake --build build -j32` passed.
  - `test_blackhole_typed_tile_cb_queue_verifier.py` reported `9 passed`.
  - Focused TT-Sim runtime selectors reported `16 passed`, including the T3
    32x64 elementwise chain.

#### flash-attn seq64 exposed stale exact-CB fronts across k-block iterations

- **症状**:
  - small flash-attn runtime 仍然通过，但 seq64 MHA/GQA direct runtime
    数值大幅偏离 reference。
  - source 级 CB event replay 进一步暴露两类队列问题：`acc_s` 的 QK
    score CB 在下一次 QK publish 前没有释放旧 front page；`acc_o`
    缩放后 PV merge 又对已经释放的旧 live-form CB 做第二次 pop。
- **根因**:
  - 多 block flash-attn 让同一个 logical state 在多个 exact-CB live
    forms 间流动。旧实现把部分释放职责推迟到 accumulator merge 的
    late discard 路径，等同于在后端重新猜当前 CB queue state。
  - select/source projection 也曾禁止 borrowed selected-source live value
    在消费点释放，导致 QK score 的 stale front page 留在 CB 里，下一轮
    reduce 读到旧分数。
  - 单页 exact-CB 输入还曾用动态 tile index 读取，`copy_tile(cb, tile)`
    会在 single-page CB 上读越当前 event 的逻辑页。
- **修法**:
  - exact-CB consumer 在消费点基于 live identity、future uses 和下一次
    write 判定是否释放 borrowed live page；single-page selected-source
    live values 允许在 source projection 阶段释放，多页 selected-source
    值仍保守保留。
  - 删除 accumulator merge 的 late `discard_live_form_before_publish`
    兜底；重新发布到 live-form CB 前必须由真正的消费者负责释放旧页。
  - single-page exact-CB 输入的 leaf copy/binary/broadcast tile index 固定
    为 0；多页输入才按 output tile 做取模。
  - tests 增加 seq64 CB event replay、QK score release、PV merge consumed
    scaled `acc_o` live form，以及 single-page CB tile-index 检查。
- **教训**:
  - flash-attn 小 shape 不能覆盖 multi-block exact-CB state lifetime；
    seq64 是必要 runtime gate。
  - CB queue lifetime 是源/IR 可验证的协议，不应靠后段 merge 或 codegen
    根据当前 source 形状补救。
  - 修 runtime correctness 时，必须同时跑 source queue replay 和 direct
    runtime；source queue 绿不等价于数值正确，数值绿也不能证明没有
    latent CB overflow/underflow。

#### Larger flash-attn exposes missing exact-CB liveness/resource allocation

- **症状**:
  - seq64 bf16 flash-attn exact-CB path can pass, but larger seqlen coverage
    such as 128 / 256 / 512 exposes wrong exact-CB state lifetime.
  - Generated source can pop a loop-carried state CB at loop exit even though
    the value is consumed later, or can satisfy a full-tile after-loop
    consumer from a partial local fragment such as local `acc_o`.
- **根因**:
  - exact-CB resident tiles are still handled as emitter-local live-form maps
    instead of virtual values with def-use, live intervals, physical CB
    allocation, and release events.
  - loop-carried values such as flash-attn `acc_o` need initial / body /
    backedge / loop-exit semantics.  They cannot be recovered reliably from
    buffer identities or completed-state maps.
- **当前结论**:
  - The fix belongs to the explicit chain:
    `SpatialPlan` carry/live-value evidence ->
    `TTProgram` exact-CB liveness/allocation ->
    `ExecutableSpec` projected lifecycle/release records.
  - Covered old paths must be deleted during cutover: map-based exact-CB
    owner truth, completed loop-carried state recovery, and local
    `ReleaseExactInputAfterUse` release decisions must not remain as fallback
    behavior.
  - The task-level design is
    `tasks/dev_design/2026-05-05-blackhole-exact-cb-liveness-allocation.md`.

#### Overbroad sharded distribution work-coverage validation rejects device-local materialization

- **症状**:
  - Adding a T5 check that every sharded `TTBufferDistributionPlan`
    `shard_grid_shape` be covered by `TTCoreGroup.physical_cores`, then by
    `work_packets`, rejected existing flash-attn TTProgram tests.
  - The failing flash-attn plans had device-local materialization
    distributions with `shard_grid_shape = [3, 11]` but only 32 work packets.
- **根因**:
  - The validation was attached to the distribution layer, which also carries
    internal materialization records.  T5 needed a runtime-visible external
    accessor contract, not a rule for every device-local sharded buffer.
  - `host_visibility` was not a reliable external-buffer discriminator for
    static L1 tensors; the actual external ABI evidence is
    `TTABIPlan.accessors`.
- **修法**:
  - Move the work-coverage guard to `ValidateShardedAccessorWorkMapping` and
    invoke it only for `TTABIPlan` sharded L1 accessors.
  - Compare external accessor `shard_grid_shape` against total
    `TTCoreGroup.work_packets.work_count`, not raw physical core count.
- **教训**:
  - Validate runtime-visible ABI contracts at the ABI boundary.  A low-level
    distribution field can be shared by external buffers and internal
    materialization mechanics with different legality rules.

#### T5 multi-core sharded GEMM exposed TensorAccessor rank and bf16 publish gaps

- **症状**:
  - 2x2 multi-core external sharded-L1 GEMM 的 fp32-output direct runtime
    先在 C accessor 上报
    `expected 10, materialized 8`；把 rank/shard_shape 计数硬减后又在
    A/B 上报 `expected 6, materialized 7`。
  - all-external-bf16 variant 如果直接把 fp32 accumulator copy 到 bf16
    output，会在 `PlanTTCompute` 因残留 `cast` 被拒；如果显式
    `fp32 fragment -> bf16 fragment -> C`，但不走 post-merge pack publish，
    direct runtime 会报
    `thread-distributed cb_republish materialization is not admitted`。
- **根因**:
  - TileLang 侧用 shard-grid 的非 1 维度数估算
    `TensorAccessorArgs` rank；TT-Metal 实际在
    `BufferDistributionSpec::from_shard_spec` 中先转 tile pages，再调用
    `squeeze_shape_ranks`。因此 2x2 block-sharded 输出在 accessor ABI 中
    可以是 rank 1。
  - all-bf16 输出不是 writer 可隐式兜底的 cast。要被当前 direct runtime
    接收，post-GEMM bf16 cast 必须被 GEMM lowering 识别为 post-merge
    publish，并用 `pack_tile` 发布。
- **修法**:
  - `BuildTTProgram` 的 sharded accessor count 改为按 tile-page shape
    squeeze 后的 rank 计算：static args 为
    `args_config + aligned_page_size + rank + num_banks + tensor_shape +
    shard_shape + packed_bank_coords`。
  - T5 测试增加 64x64x128、2x2 work grid 的 external sharded-L1 GEMM：
    A/B height-sharded，C block-sharded，并断言 A/B/C accessor counts 为
    7/7/8。
  - all-external-bf16 case 使用 `T.clear(C_local)` +
    `T.gemm(..., clear_accum=False)` + post-merge cast 到 bf16 fragment，
    并断言 materialization `publication_protocol = pack_tile`。
- **教训**:
  - sharded accessor ABI 的 rank 是 TT-Metal page-space rank，不是硬件
    core-grid rank。测试必须覆盖 multi-core block sharding，否则 1x1 /
    1D shape 会把这个问题藏住。
  - "全 bf16" runtime correctness 要检查外部 ABI dtype 和 materialization
    protocol；只看输出 dtype 或 projection metadata 容易漏掉 direct runtime
    还未接收的 cast/publication path。

#### T5 110-core sharded GEMM exposed square-shape axis masking

- **症状**:
  - 2x2 external sharded-L1 GEMM 通过后，把 shape 扩到
    `M=320, N=352, K=256`、logical grid `11x10`、110 个 worker core 的
    all-bf16 direct runtime，TT-Metal 在写入 B 的 sharded L1 MeshBuffer 时
    报 `No L1 bank exists for core (x=0,y=10)`。
- **根因**:
  - B tensor 的形状是 `(N, K)`，height shard 对应输出列方向 `bx`。
    测试 helper 之前把 B 的 sharding grid 写成 `CoreGrid(x=1, y=grid_x)`。
    在 2x2 下 `x/y` 交换不可见；在 Blackhole 逻辑 worker grid
    `11x10` 下，`grid_x=11` 被错误放到 y 轴，越过合法 y 范围。
- **修法**:
  - B 的 external sharded-L1 memory config 改为
    `CoreGrid(x=grid_x, y=1)`。
  - 增加 110-core all-external-bf16 direct runtime gate，断言 logical grid
    `11x10`、110 个 physical cores、110 个 one-work packets，以及 A/B/C
    sharded accessor counts `11/12/61`。
- **教训**:
  - 多 core sharding 测试不能只用方形小 grid；至少要覆盖硬件 grid 的
    非方形边界，否则 operand-to-axis 错误会被对称 shape 掩盖。

#### T5 K-sharded GEMM exposed missing logical-z and writer xy mapping

- **症状**:
  - K 维 sharding case 最初只能作为 reject。改成 runtime correctness 后，
    projection 先缺 `logical_grid_z`，随后 A/B width-sharded accessor
    compile-time ABI 报 `expected 7, materialized 9`。
  - ABI count 修正后 direct runtime 能执行完两个 K shard，但输出只在一个
    tile 附近正确，整体 max diff 很大。
- **根因**:
  - core assignment / `TTCoreGroup` / `ExecutableSpec` 原先只保存
    `grid_x/grid_y`，第三个 `T.Kernel` 维度没有进入 leaf/runtime
    contract。
  - sharded accessor count 推导只看 `TTBufferDistributionPlan.logical_shape`；
    对显式 external placement intent，真实 tensor logical shape 在
    `TTTensorMemoryConfigPlan` 中，K-width sharding 的 TT-Metal
    TensorAccessorArgs 因此被低估。
  - writer 的 per-work arg 已经被设成 `logical_block_xy_linear`，但 codegen
    没把这个 value source 还原成 `blockIdx.x/y`，导致 partial-K writer
    退回 `0 /* core_x/y */` 并把所有 partial 写到 tile 0。
- **修法**:
  - `logical_grid_z` 进入 core assignment、`TTCoreGroup`、projection、
    serialization、codegen 和 direct-runtime work context。
  - sharded accessor count 从 `TTTensorMemoryConfigPlan.logical_shape` 和
    `TTBufferDistributionPlan.shard_shape/shard_grid_shape` 一起推导。
  - direct runtime 对 partial-K GEMM 按 K shard 构造 z-wave，`bk=0` 写
    final `C`，后续 shard 写 partial-C scratch，然后用 runtime-issued
    TT-Metal tile-add reduction program 在设备端把 partial `C` 合进 final
    `C`。
  - codegen 支持 `logical_block_xy_linear` 的 x/y 反解，writer 使用正确
    output tile。
- **教训**:
  - K sharding 的测试必须真的让 A/B 在 K 维 width-sharded，并验证
    `logical_grid_z > 1` 和 final correctness。多 core M/N sharding 不等价
    于 cross-core partial sum。
  - direct-runtime device-side partial reduction 目前依赖 blocking z-wave /
    reduction wave barriers；不要把它表述成 production single-launch
    semaphore/atomic reduce 已经完成。

#### Generic tiled-CB fragment bridge lost `threadIdx.x`

- **症状**:
  - T6 existing-TIR topk lowering moved DRAM input materialization through the
    typed reader CB path, but generated TRISC source still emitted generic
    tiled-CB fragment bridge code with `const uint32_t thread_idx_x = 0`
    inside a real `for (tx = 0; tx < 128; ++tx)` loop.
  - The same compute source previously also fell back to raw packed-argument
    handle loads when the compute segment had no explicit runtime args.
- **根因**:
  - `CodeGenBlackhole::VisitStmt_(AttrStmtNode)` has a custom
    `thread_extent` emission path that bypasses `BindThreadIndex`; it updated
    `var_idmap_` for the loop variable but not the codegen-side
    `thread_idx_x_expr_` consumed by generic logical tile-layout bridges.
  - Empty compute-segment `runtime_args` made codegen choose the legacy raw
    function-parameter fallback instead of typed executable runtime args.
- **修法**:
  - Preserve and restore `thread_idx_x_expr_` while emitting both one-shot
    thread bindings and emitted thread loops.
  - Give compute segments an explicit `work_linear_id` runtime arg so TRISC
    kernels use the typed runtime-arg path; unexpected global-buffer use should
    fail as a missing binding instead of being silently reconstructed.
- **教训**:
  - Codegen-local bridge state is still part of the typed lowering contract.
    When a builtin consumes a thread binding implicitly, the custom thread
    emitter must keep that binding live; do not repair it by adding a
    workload- or algorithm-specific semantic object.

#### TRISC CB write pointer and live-form invalidation hazards

- **症状**:
  - After routing existing-TIR T6 materialization through typed CB bridges,
    standalone leaf compute initially hit TRISC link failures for direct
    `get_local_cb_interface(cb_id)` use from ordinary compute code.
  - Moving the read through a PACK-to-MATH/UNPACK mailbox fixed the link, but
    omitting the historical `<< 4` byte-address conversion left compute with
    word-address pointers and produced `binary_add` mismatches.
  - A separate attempted fix that cleared all local/`blackhole.acc`
    live-form aliases on ordinary local stores broke the leaf A/B live-CB
    path: reader CB pages were no longer recognized as the compute operands,
    so source emitted a bogus local republish path.
- **根因**:
  - `get_local_cb_interface(...).fifo_wr_ptr` is in 16-byte words; codegen
    consumers need byte pointers.
  - Live-form invalidation must be tied to the actual written buffer/event.
    A broad dtype/scope rule treats local mechanics as a new semantic
    category and can delete valid aliases needed by later TT leaf ops.
- **修法**:
  - In `tilelang_cb_write_ptr_bytes_direct`, read `fifo_wr_ptr << 4` on the
    PACK thread and deliver that byte address through the mailbox.
  - Do not add a blanket local-store alias invalidation path.  For T6 stale
    fill publication, invalidate the retained fill fact at the `T.fill`
    lowering point when a future scalar write precedes the next compute
    consume.
- **验证**:
  - `testing/python/target/blackhole/test_blackhole_leaf_compute_runtime.py`
    reports `16 passed, 2 skipped` under the repository TT-Sim setup.

#### T6 topk was a backend value/index lowering bug, not a frontend-op gap

- **症状**:
  - Existing Tile TIR topk-style value/index selection was initially described
    as if it needed a new frontend topk op or selection plan.
  - The direct runtime path either fell into the standalone
    `Int32 reduce_tile<MAX, REDUCE_ROW>` index boundary or timed out in the
    bf16 case.
  - An intermediate fp32 scalar-output writer fix produced exact value matches
    but corrupted several index rows with float value bits.
- **根因**:
  - The existing TIR already expresses the computation: value reduce, index
    mask/update, index reduce, value mask, and explicit value/index stores.
    The missing piece was an admitted backend value/index selection lowering,
    not frontend syntax.
  - TTProgram/codegen represents bf16 accumulator dtype as `Float16_b`; a
    narrower dtype check let the bf16 case fall back to the old reduce path.
  - The scalar writer's event schedule is dtype-dependent.  bf16 value output
    waits on 16-row groups, so publishing only the fp32-style 4 events per rank
    left the writer blocked on the fifth event.
  - Reusing source-CB tail scratch for small scalar page writes overlapped live
    payload and allowed later value bytes to be written as index output.
- **修法**:
  - Keep the frontend as ordinary TIR and emit one backend typed scan for the
    existing value/index selection records.  The scan reads from the typed
    reader CB, computes values and indices together, and publishes value/index
    output CB pages for the normal writer.
  - Accept `Float16_b` for the bf16 value side and publish bf16 values as
    bfloat16 bits.
  - Match the writer event grouping: fp32 uses 32-row grouping, bf16 uses
    16-row grouping.
  - Stage small output pages through TT-Metal inline L1 scratch rather than
    direct unaligned writes, stack scratch, or source-CB tail scratch.
- **验证**:
  - `testing/python/target/blackhole/test_blackhole_topk_runtime.py` reports
    `4 passed` under TT-Sim, covering structure, fp32 single-work, fp32
    multi-work, and bf16 values with exact `int32` indices.

#### CB-backed local allocations over-reserved shared state CBs

- **症状**:
  - Flash attention seq64 bf16 source queue validation reported
    `cb28 reserve would exceed capacity 4`, then after the first correction
    `cb19 reserve would exceed capacity 1`, and then `cb17 push has only 0
    reserved pages`.
- **根因**:
  - Codegen treated CB capacity `num_pages` as the initial reserve size for
    each CB-backed `blackhole.acc` allocation.  Small fragments such as
    `logsum`, `scores_max`, and `scores_sum` each needed one page but each
    reserved the full four-page CB capacity.
  - Some producer builtins reuse an allocation-owned writable window.  Skipping
    every later reserve while a CB-backed handle is active was too broad:
    once a previous push consumes the allocation reserve, later producers for
    the same CB need a fresh reserve.
- **修法**:
  - Derive CB-backed allocation initial reserve pages from allocation byte
    size and CB page size, capped by CB capacity unless an explicit event
    contract overrides it.
  - Track allocation-reserve credit per active CB-backed handle.  Skip a later
    same-CB reserve only while unconsumed allocation reserve credit exists, and
    consume that credit on `cb_push_back`.
- **验证**:
  - `test_flash_attention_seq64_bf16_compute_source_keeps_cb_events_queue_consistent`
    passes, and the focused flash attention structural selector set reports
    `3 passed`.

#### Row-reduce exact-CB results must not be eagerly untilized before tiled consumers

- **症状**:
  - seq64 bf16 flash-attn partial-combine source checks failed with
    `get_tile_address(0)` in the compute source after T6 row-reduce work.
- **根因**:
  - `GenerateRowReductionSequence` materialized every row-reduce exact-CB
    result back into local fragment state to support standalone value/index
    selection.  In flash-attn, the next real consumer of `scores_sum` is a
    tiled exact-CB combine, so the local materialization was a stale
    pre-rewrite reference path.
- **修法**:
  - Only materialize a row-reduce exact-CB result to local state when the first
    future use before the next write is a true reference.  If the first future
    use is compute or transport consume, keep the value as the typed exact-CB
    live form.
- **教训**:
  - T6 standalone row-reduce/select needs local materialization for scalar
    reference consumers, but T7 flash reduce/combine must preserve exact-CB
    live-form ownership.  Do not use unconditional post-reduce untilize as a
    cross-workload bridge.

#### Exact-CB virtual intervals inherited allocator lifetime and hid CB interference

- **症状**:
  - After adding a validator gate for overlapping exact-CB intervals sharing a
    physical CB, seq64/128/256/512 flash-attn lowering failed before runtime.
    The first seq64 failure reported:
    `exact_cb_interval_acc_s_39_0 [0, 53]` and
    `exact_cb_interval_acc_s_39_5 [0, 538]` both using the same physical CB.
- **根因**:
  - `TTExactCBLiveInterval` construction used the merged CB requirement
    lifetime as the virtual value begin point.  A CB requirement is an
    allocator slot and may cover multiple exact-CB versions; it is not the
    virtual value's semantic lifetime.
  - `PlanTTCBAlloc` assigned physical CB IDs from IR builtin use intervals
    without incorporating typed exact-CB interval bounds, so TTProgram
    lifecycle records and resource allocation could disagree.
- **修法**:
  - Build exact-CB virtual interval begin/end from producer/use program-point
    evidence.
  - Feed typed exact-CB interval bounds into `PlanTTCBAlloc` before assigning
    physical CB IDs.
  - Keep the validator gate: overlapping exact-CB virtual intervals may not
    share one physical CB.
- **验证**:
  - Final T7.5 selector reported `10 passed, 3 skipped`: the skips remain the
    typed TT-Sim `tensix_execute_pacr: count=1` boundary for seq128/256/512,
    while seq64 direct runtime passed.

### exact-CB materialization-pop fallback hid missing logical release identity

- **症状**:
  - Removing the local materialization fallback
    `blackhole_cb_pop_front(cb_value.cb_id, cb_value.num_tiles)` exposed
    failures in flash-attn loop-carried materialization: source lowering could
    not find a typed release event and hit
    `Exact-CB materialization pop requires a typed release event`.
- **根因**:
  - The materialized loop-carried exact-CB output reached
    `MaterializeExactTiledCBToLocalBuffer` without a stable logical
    `live_identity`, so release lookup used an ephemeral local buffer identity
    rather than the TTProgram exact-CB virtual value / allocation record.
- **修法**:
  - `MaterializeLoopCarriedExactOutput` now binds the exact-CB value to the
    destination buffer identity before materialization.
  - `MaterializeExactTiledCBToLocalBuffer(..., pop_front=true)` requires a
    typed `TTExactCBReleaseEvent` and no longer emits a local fallback pop.
  - `ValidateTTProgram` rejects full-logical-tile consumers bound to
    `thread_distributed_slice` live forms so after-loop full-tile consumers
    cannot silently consume partial slice coverage.
- **验证**:
  - T7.5 selector reported `10 passed, 3 skipped`: the new structure gates
    cover materialization-pop fallback deletion and full-tile/slice rejection;
    the skips remain the typed TT-Sim `tensix_execute_pacr: count=1` boundary
    for seq128/256/512 after source/spec admission.

### Larger flash shapes exposed false indexed evidence and local accumulator reload gaps

- **症状**:
  - A larger GQA flash pipeline first failed validation with
    `indexed access requires loop_vars evidence` for constant full-tile reads
    such as `index_exprs ['0', '0']`.
  - After tightening that evidence, clear-accum=false PV merge failed with
    `PlanTTKernelABI requires buffer materialization fact or exact live-CB state`
    for loop-carried `acc_o`.
  - A follow-up source regression showed `acc_s` and `acc_o` rendered from the
    same physical CB write pointer because metadata-only CB configs were used
    to CB-back local `blackhole.acc` allocations.
- **根因**:
  - `BuildSpatialPlan` treated rank-aligned indices as indexed evidence even
    when the TIR expression had no index variable.
  - Loop-carried collection did not model exact-CB leaf read/write effects
    (`tilize_*_fragment_slice`, `untilize_cb_front_tile_fragment`) and earlier
    matmul logic treated loop presence as accumulator liveness.
  - Source codegen interpreted any CB config for a `blackhole.acc` variable as
    permission to allocate it from `tilelang_cb_write_ptr_bytes_direct`, even
    when TTProgram had not projected `initial_reserve_pages`.
- **修法**:
  - Require a real index variable before projecting `AccessRegion.index_exprs`
    as indexed evidence.
  - Derive clear-accum=false loop-carried liveness from TIR read-before-write,
    matmul `clear_accum`, and exact-CB leaf accesses; admit local accumulator
    reload only when typed loop-carried evidence and full static local shape
    agree.
  - Require explicit `initial_reserve_pages` before CB-backing
    `blackhole.acc` source allocations.
- **验证**:
  - Original larger GQA pipeline regression passed.
  - `seq_len=128,256,512` flash metadata gate passed; direct runtime still
    skips at the typed TT-Sim `tensix_execute_pacr: count=1` boundary.
  - Flash source/lifecycle selector reported `8 passed`, T8 grid-indexed
    selector reported `9 passed`, and seq64 bf16 MHA flash direct runtime
    passed.

### T8 table-indexed copy leaked index-table BufferLoad into source/runtime ABI

- **症状**:
  - Minimal `BlockIndices[bx]` staged copy failed source codegen with
    `Find undefined Variable BlockIndices`.
  - After lowering the source-side load to a runtime arg, direct runtime first
    rejected the formal `BlockIndices` tensor because it had no explicit
    input/output role binding.
- **根因**:
  - The predicated copy value was represented as a `tir.if_then_else` call, not
    a `SelectNode`, so guarded copies were not recognized as copy loads.
  - `AccessRegion` recording did not substitute active `LetStmt` bindings,
    losing the relation between `tile_id` and `BlockIndices[bx]`.
  - The executable role/materialization gates only considered address-bearing
    runtime args; an index table used solely to compute per-work args was not
    registered as a named input buffer.
- **修法**:
  - Recognize guarded zero-fill copy loads for both `SelectNode` and
    `tir.if_then_else`.
  - Substitute `LetStmt` bindings into `AccessRegion.index_exprs` and derive a
    table-backed tile-start binding from that evidence.
  - Project `index_buffer` / `index_value_scale` through TTProgram,
    executable metadata, serialization, and Python helpers.
  - Register index tables as explicit page-addressed interleaved DRAM input
    materializations and
    include them in direct-runtime buffer role checks.
  - Validate table-derived tile starts against the target buffer's typed page
    count so invalid table entries fail closed.
- **验证**:
  - The table-indexed pipeline/runtime selectors reported `3 passed`, and the
    broader T8 copy selector reported `5 passed`.

### Two-dimensional index tables were flattened by work-linear order

- **症状**:
  - A `BlockIndices[bx, by]` staged copy carried only `index_buffer` /
    `index_value_scale` in the A tile-start binding.
  - Direct runtime read the index table at `work_linear_id`, so a non-symmetric
    table shaped `(grid_x, grid_y)` produced the sequence
    `[table[0,0], table[0,1], ...]` instead of the launch-axis order
    `[table[0,0], table[1,0], ...]`.
- **根因**:
  - The binding expressed which table to read but not how the TIR table load
    was addressed.  `work_linear_id` was a hidden addressing assumption that
    happened to match the first one-dimensional case.
- **修法**:
  - Derive table shape from the index-table buffer and table index sources from
    the TIR `BufferLoad` indices plus `blockIdx.x/y/z` launch-axis tags.
  - Project `index_table_shape` and `index_table_index_sources` through
    `TTPerWorkArgSpec`, `ExecutableSpec`, runtime metadata, Python rebuild
    helpers, and `BlackholeModule` binary serialization.
  - 2026-05-05 update: this shape/source evaluator was later replaced by
    generic serialized TIR `value_expr`; the fields remain projection residue
    and diagnostics, with no `work_linear_id` fallback.
- **验证**:
  - The non-symmetric `2x3` direct-runtime case passed through
    `BlackholeModule`.
  - The serialized module preserved the table-addressing fields and passed the
    same correctness check.

### Ragged bf16 row copy zero-filled TT face pages instead of logical rows

- **症状**:
  - `RowCounts[bx]` ragged staged copy initially compiled and launched but
    `RowCounts=17` copied 18 rows, small counts copied in pairs, and counts
    around 8-16 saturated at 16 rows.
  - Adding explicit CB zero pages fixed stale output for fully invalid blocks
    but did not fix the rounded row counts.
- **根因**:
  - The direct-runtime host transfer helper treated any bf16 64-byte
    page-addressed materialization as an nfaces tiled plan because 64 bytes is
    32 bf16 elements.  The source reader/writer page IDs were logical row
    pages, but host data was laid out as TT face pages.
  - Skipping invalid writes was also semantically incomplete for
    `if_then_else(load, 0)` because output buffers are not a correctness
    contract for zero-fill.
- **修法**:
  - Restrict host-side nfaces tilization to complete 32x32 tile pages.
    Sub-tile page-addressed row/stick pages remain raw row-major pages.
  - For ragged row predicates, reader publishes one page per logical row:
    valid rows read DRAM, invalid rows zero the reserved CB page, and writer
    writes all pages.
- **验证**:
  - Ragged row runtime selector passed for `RowCounts=[32,17,0]`.
  - T8 indexed/ragged aggregate selectors reported `7 passed`.
  - Existing 64-byte page-addressed stick direct-runtime selectors remained
    green.

### Segment row offsets were lowered as tile-start bindings

- **症状**:
  - A non-uniform segmented row copy using `SegmentOffsets[bx]` and
    `SegmentCounts[bx]` compiled to `tile_index = a_segment_row_start / 32`
    and tried to materialize A with 2048-byte tile pages.
  - Direct runtime failed before correctness with a TT-Metal buffer
    `size % page_size == 0` fatal for a source tensor whose row count was not
    a multiple of a 32x32 tile page.
  - The executable also carried a stale A-side `a_tile_start_id` binding
    pointing at `SegmentOffsets`, even though the source address semantics
    were row offsets.
- **根因**:
  - The first table-indexed rewrite classified any int32 table load used in a
    copy index as `tile_start`.  It did not distinguish coefficient-32 tile-id
    usage (`tile_id * 32 + row`) from coefficient-1 row-start usage
    (`segment_start + row`).
  - Fused dataflow ABI synthesis then added default A tile-start/count/stride
    runtime args even after the segmented input address had a stronger
    TIR-derived value binding.
  - The row predicate matcher attempted to re-match the original predicate
    against a flattened shared index instead of consuming the projected
    bound-value binding.
- **修法**:
  - Compute the table variable coefficient in copy index expressions by
    substituting table var `0` and `1`; coefficient 1 lowers to
    `a_segment_row_start`, while the existing coefficient-32 case remains the
    tile-start path.
  - Lower the predicate table load to a generic bound-value binding when a
    segment base-value binding is active.
  - Use the evaluated base value plus `page_row` for reader page IDs and
    compare `page_row` against the evaluated bound value, with zero CB pages for
    invalid rows.
  - Suppress default A tile-start/count/stride bindings for the segmented
    input path.
- **验证**:
  - Segmented structure and direct-runtime selectors passed.
  - T8 indexed/ragged/segmented aggregate selectors reported `9 passed`.

### Multiple segmented row ranges exposed row-page/tile-page confusion

- **症状**:
  - A two-range segmented row copy projected independent
    multiple base/bound value bindings, but
    direct runtime read row 0 when `SegmentOffsets[0,0]` was 3.
  - The same WIP also regressed the single-range segmented row runtime case.
- **根因**:
  - The row-page source path reused `base_tile_index` from full-tile
    linearization.  For `segment_start + row`, that value is
    `floor(segment_start / 32)`, which loses the intra-tile row offset before
    the 64-byte row-page reader renders page IDs.
- **修法**:
  - Derive row-page source IDs from the current zeroed TIR row expression and
    add the unrolled local row, so segmented row pages use
    the evaluated base value plus `page_row`.
  - Keep one per-work arg identity per independent table load and reuse the
    TIR predicate rewrite for both single- and multi-range row counts.
- **验证**:
  - Focused single/two-range segmented structure and direct-runtime selectors
    reported `4 passed`.
  - T8 indexed/ragged/segmented aggregate selectors including grid-indexed,
    block-indexed, sparse, ragged, segmented, and paged cases reported
    `24 passed`.

### Paged cache-length copy lost its ragged predicate through fused tile transport

- **症状**:
  - A copy-shaped paged surface using `PageTable[bx, by]` and
    `CacheSeqLens[bx]` projected page and predicate-bound values, but
    generated source still emitted a full-tile
    `read_tile_to_cb`.
  - Direct runtime copied rows beyond the cache length because invalid rows
    were never zero-filled.
- **根因**:
  - The row-bound admission check only recognized predicates whose top-level
    expression was directly `row < bound`.
  - The paged TIR predicate is a conjunction:
    `logical_page * page_rows + local_row < cache_len` plus page-id bounds.
    The A source address is `page_id * page_rows + local_row`, so the local
    row expression must be derived before matching the predicate.
  - The staged copy reader/writer pair was also eligible for the fused
    full-tile shortcut, which discards row predicate evidence.
- **修法**:
  - Flatten conjunctions and select the comparison conjunct that contains the
    TIR-derived local row expression.
  - Rewrite `logical_block_y` in that comparison to a generic per-work value
    binding, while `CacheSeqLens[bx]` remains a table-backed generic
    predicate-bound binding.
  - Disable fused full-tile transport when predicate-bound bindings are active,
    so the reader/writer use row-page transport with explicit invalid-row
    zero-fill.
- **验证**:
  - Focused paged structure/runtime selectors reported `2 passed`.
  - T8 indexed/ragged/segmented aggregate selectors including the paged case
    reported `14 passed`.

### Scaled indexed block copies double-applied the table scale in source

- **症状**:
  - A two-tile block-indexed copy projected A `tile_start` with
    `index_value_scale=2`, but generated source read
    `a_tile_start_id * 2` and `a_tile_start_id * 2 + 1`.
  - Direct runtime already scaled the table value before passing
    `a_tile_start_id`, so source and runtime disagreed on whether the arg was
    a block id or a tile id.
- **根因**:
  - The table coefficient was correctly derived from `AccessRegion` evidence
    and stored on `TTPerWorkArgSpec`, but source tile-index inference still
    evaluated the original TIR block-id expression under the Let-bound runtime
    arg variable.
  - That left the binding scale as owner truth for runtime while source
    kept an implicit copy of the same scale in the address expression.
- **修法**:
  - Record the TIR-derived tile scale as pass-local mechanics on the
    runtime-arg Let variable.
  - Normalize base tile-index expressions such as `block_id * 2` back to
    `block_id` when `block_id` is bound to a scaled tile-start runtime arg.
- **验证**:
  - Focused scaled-block structure/runtime selectors reported `2 passed`.
  - T8 indexed/ragged/segmented aggregate selectors including scaled-block and
    paged cases reported `16 passed`.

### T9 paged GQA decode reused stale `acc_s` front pages across KV pages

- **症状**:
  - The two-page paged GQA decode compiled and launched but the full output
    disagreed with the host flash-attention reference across batches and
    heads.
  - Generated source for `acc_s` pushed and waited on the first page, then
    reserved/pushed the second page without popping the first front page; the
    tail cleanup tried to `cb_pop_front(..., 2)`.
- **根因**:
  - Producer-side front-pop management only covered state-like CB
    requirements.  Local stream intermediates such as `acc_s` can also be
    produced, consumed, and then produced again on the same physical CB.
  - The static pop planner counted total published pages instead of pages
    actually visible on the local intermediate front, so adding the missing
    pre-producer pop initially exposed a tail over-pop.
- **修法**:
  - Allow local intermediate stream requirements to auto-pop stale front pages
    before a later producer reserve/push.
  - Use event pages as the local-intermediate producer capacity and clamp
    generated local-intermediate pops to currently available front pages.
  - Keep this clamp scoped to locally produced intermediate CBs so reader-fed
    input lifetime remains owned by the normal input release contract.
- **验证**:
  - Full T9.2 paged GQA bf16 direct runtime reported `1 passed`.
  - The focused T9/T7 selector set covering page-table projection, QK/AV
    page 0 and page 1, seq64 flash, exact-CB partial combine, and full T9.2
    reported `9 passed`.

### Larger flash shapes missed CB metadata in no-runtime-arg compute segments

- **症状**:
  - `seq_len=128/256/512` flash metadata/runtime coverage failed source
    generation with `Missing CB data_format for cb_id=19` when rendering
    `untilize_cb_front_tile_fragment`.
  - The executable `cb_configs` did contain the physical CB entries, including
    the reduce-output CB, so allocation/projection was not the failing layer.
- **根因**:
  - Codegen loaded CB config metadata as part of runtime-arg binding.
    Compute segments with no runtime args skipped that path but still needed
    CB metadata to render typed CB operations.
- **修法**:
  - Load CB config metadata from the `PrimFunc` independently of runtime-arg
    binding before generic kernel body generation.
  - Keep runtime-arg binding responsible only for argument values, not for the
    existence of CB schema metadata.
- **验证**:
  - Extended flash `seq_len=128,256,512` metadata/runtime selector reported
    `3 passed, 3 skipped`; the skips are the existing typed TT-Sim
    `tensix_execute_pacr: count=1` capability boundary.

### Sparse indexed copy bound both tile-start bindings to the first A read region

- **症状**:
  - The sparse two-entry indexed copy emitted two runtime args,
    `a_tile_start_id` and `a_tile_start_id_1`, but both
    `TTPerWorkArgSpec` records referenced the same SpatialPlan access region
    `access_closure_0_read_A_0`.
  - Runtime correctness could still pass because legacy `index_table_*`
    projection fields evaluated different table columns, hiding the IR binding
    bug.
- **根因**:
  - `BuildSpatialPlan` collapsed distinct same-subject read patterns into one
    `AccessRegion`.
  - The TT per-work binding lookup used `subject|access_kind` first-match, and
    the pass-local subject indices were compared before substituting the
    Let-bound table load back to the original TIR expression.
- **修法**:
  - Preserve distinct same-subject access patterns in `SpatialPlan` by
    structural `index_exprs`.
  - Store pass-local subject `index_exprs` on indexed per-work runtime args
    only for matching, substitute active Let table loads, and select the
    matching `AccessRegion` by structural equality.
- **验证**:
  - `test_t8_spatial_plan_preserves_distinct_same_subject_indexed_access_regions`
    passed.
  - `test_blackhole_sparse_2tile_copy_uses_two_value_expr_tile_start_bindings`
    now asserts distinct access-region bindings and passed.
  - T8 irregular aggregate selector reported `14 passed, 66 deselected`.

### Guarded AccessRegion recorded only a kind, not the predicate expression

- **症状**:
  - Ragged and segmented copies could mark an `AccessRegion` as `guarded`
    while the region itself did not preserve the boolean TIR predicate that
    guarded the read.
  - Downstream code could still pass by relying on per-work subrole names
    such as `valid_rows`, which is exactly the schema-shaped semantic recovery
    the IR-first design forbids.
- **根因**:
  - `BuildSpatialPlan` tracked only predicate depth, not predicate
    expressions, so `predicate_kind=guarded` was a label without owner-truth
    evidence.
  - `ValidateSpatialPlan` did not reject guarded regions with empty predicate
    evidence.
- **修法**:
  - `AccessRegion` now carries generic `predicate_exprs` for guarded regions.
  - The access-pattern collector records predicates through statement
    `IfThenElse`, expression `Select`, and `tir.if_then_else` calls while
    preserving Let-substituted TIR expressions.
  - `ValidateSpatialPlan` rejects guarded regions without boolean
    `predicate_exprs` and rejects predicate expressions on non-guarded
    regions.
- **验证**:
  - The positive SpatialPlan test checks that the ragged A read records
    `T.shift_right(tx, 2) < RowCounts[bx]`.
  - The negative validator test fails closed when that guarded region is
    rebuilt with empty `predicate_exprs`.

### Index table addressing had a buffer-wide fallback cache

- **症状**:
  - TT lowering kept a pass-local `index_buffer -> addressing` cache and ABI
    lowering could fill missing binding table shape/source fields by
    looking up only the index-buffer name.
  - That fallback was redundant for current explicit per-work bindings and
    unsafe for sparse forms where one table is addressed through multiple
    independent constants or launch-axis expressions.
- **根因**:
  - Early table-backed bindings were brought up one buffer at a time, so
    table addressing was cached by buffer identity before same-subject /
    same-table multi-entry cases existed.
- **修法**:
  - Delete `index_table_addressing_by_buffer_` and
    `RecordIndexTableAddressing`.
  - Require table addressing to be carried by the concrete per-work binding
    produced from the matching TIR table load / `AccessRegion`.
  - Add a source-level regression test that rejects reintroducing the
    buffer-wide side cache.
- **验证**:
  - The new no-side-cache regression test passed after deletion.
  - Focused 1D/2D/scaled/sparse binding tests reported `4 passed`.
  - The T8 copy-pipeline selector covering per-work, sparse, ragged, paged,
    and segmented cases reported `14 passed, 66 deselected`.

### Index-table bindings without addressing fell back to work-linear order

- **症状**:
  - Direct runtime evaluated a table-backed per-work binding at
    `work_linear_id` when `index_table_shape` and
    `index_table_index_sources` were absent.
  - After removing the buffer-wide addressing cache, the segmented row-start
    path exposed another old ABI branch that synthesized
    `a_segment_row_start` from only `segment_row_start_index_buffer_name_`,
    overwriting the concrete binding that carried the table shape/source
    evidence.
- **根因**:
  - Early one-dimensional table bindings treated launch linearization as a
    harmless default.  Once table loads can be `[bx, by]`, `[bx, k]`, or
    constants, launch order is not semantic owner truth.
- **修法**:
  - The intermediate fix made `value_source=index_table` require explicit
    table shape and one index source per dimension during executable
    extraction and direct-runtime admission; this was later replaced by
    generic serialized `value_expr`.
  - Delete the `work_linear_id` fallback in direct runtime.
  - Delete old ABI synthesis branches for `valid_rows`,
    `segment_row_start`, and `segment_row_count` bindings that only knew
    the index-buffer name.
- **验证**:
  - A source-level regression test rejects the direct-runtime work-linear
    fallback.
  - The T8 selector covering per-work, sparse, ragged, paged, and segmented
    cases reported `14 passed, 68 deselected` after deletion.

### Index-table shape/source metadata started acting like a second value evaluator

- **症状**:
  - After moving table addressing onto per-work bindings, direct runtime
    still computed table-backed per-work values from
    `index_table_shape/index_table_index_sources`.
  - That made the legacy projection fields look like owner truth and invited
    more case-shaped schema additions, even though the original TIR already
    contains the exact value expression.
- **根因**:
  - The binding carried enough metadata for the first table cases, so the
    runtime grew an index-table-specific evaluator instead of consuming a
    generic expression lowered through the IR chain.
- **修法**:
  - Add generic `value_expr` to `TTPerWorkArgSpec` and project it through
    `ExecutableSpec`, runtime extraction, `BlackholeModule` metadata, and
    binary serialization.
  - Direct runtime now evaluates the serialized TIR expression, including
    integer `BufferLoad` from materialized host-side table data, and rejects
    table-backed bindings without `value_expr`.
  - Delete the direct-runtime `EvaluateIndexTable*` value evaluator and guard
    against new selection/topk/index-table-constant schema fields.
- **验证**:
  - The value-expression projection test passed.
  - The transform architecture selector reported `11 passed, 108 deselected`.
  - The TT-Sim direct-runtime selector covering indexed, ragged, segmented,
    paged, sparse, and serialized indexed copies reported
    `10 passed, 37 deselected`.

### Per-work descriptor_kind / row-page identities became a second schema

- **症状**:
  - Public per-work schema had grown workload-shaped subroles such as
    `descriptor_kind`, row/page identities, and implementation names like
    `page_value_arg_name`.
  - Even after `value_expr` became the real owner truth, docs/tests still
    described row-start, row-count, and page-index as public binding kinds.
- **根因**:
  - The cleanup stopped at moving table evaluation into `value_expr`, but left
    a second semantic vocabulary beside IR evidence.  That made it too easy
    for later passes to infer meaning from subrole names instead of
    `AccessRegion`, `predicate_exprs`, and the serialized TIR expression.
- **修法**:
  - Delete public `descriptor_kind` and row/page identities from
    `TTPerWorkArgSpec`, `ExecutableSpec`, runtime metadata, and module
    serialization.
  - Use generic `per_work_value`, `per_work_value_1`, ... identities for
    dynamic base/bound/launch-axis values.  Keep `arg_kind` only as the leaf
    ABI consumption point and keep owner truth in `value_source`,
    `value_expr`, and `AccessRegion` evidence.
  - Rename implementation-only `page_value_arg_name` style variables to
    generic dynamic value names, and add a public-schema guard against
    reintroducing `descriptor_kind`, row/page identities, selection/topk, or
    index-table fields.
  - Update projection/architecture tests so they reject all `kDescriptor*`
    constants instead of preserving an expected row/page descriptor whitelist.
    Tests must validate `AccessRegion` evidence through generic per-work
    fields (`arg_kind`, `value_source`, `access_region`) rather than
    `descriptor_kind`.
  - Audit shared companion/schema headers as well as Blackhole leaf records.
    Unused `selection_targets` / `selection_pairs` manifest keys and stale
    `buffer_*_contracts` schema constants in `companion_base.h` were deleted
    because unused header constants can still normalize a fake protocol in
    future patches.
- **验证**:
  - Focused structural/projection selectors covering indexed, ragged,
    segmented, paged, grouped-GEMM, and paged GQA/MLA binding projection
    passed.
  - Follow-up schema guard selector covering no index-table side cache,
    no work-linear-id fallback, no descriptor constants, generic
    AccessRegion evidence, public schema field forbids, and no selection
    plan projection reported `6 passed`.
  - Follow-up guard now also scans `companion_base.h`; focused selectors
    covering stale contract/selection keys, generic value-expr schema, and
    public schema field forbids reported `3 passed`.
  - TT-Sim direct-runtime copy selectors and T9.3 paged MLA selectors passed.
    At that checkpoint T9.2 paged GQA still stopped at a typed PACR simulator
    boundary; later 2026-05-06 work reclassified the full online-softmax
    failure as backend live-form/codegen bugs and made the admitted T7/T9 full
    runtime paths pass.

### T9.2 paged GQA previously reached TT-Sim PACR capability boundary

- **症状**:
  - An earlier paged GQA decode direct-runtime selector compiled, created
    reader / compute / writer kernels, and launched the multi-core workload,
    then TT-Sim aborted with
    `UnsupportedFunctionality: tensix_execute_pacr: intermediate_format=0 late_from_format=5`.
- **根因**:
  - The failure is below the per-work binding/schema layer: source/spec
    projection has already reached TT-Metal execution, and the thrown reason
    is TT-Sim PACR format capability.
- **修法**:
  - Track this as historical T9.2 runtime/simulator-boundary work.  Do not work around it
    by reintroducing row/page descriptors, page-value side variables,
    source-name recovery, or a paged-GQA-specific execution path.
  - Do not treat this old PACR message as the current active full flash decode
    gate without rerunning the typed source/spec path; the admitted current
    T9.2 full paged GQA path now passes bf16 direct-runtime correctness.
- **验证**:
  - The focused T9.2 selector reproduced the PACR boundary after the IR-first
    per-work schema cleanup at that checkpoint.

### GEMM direct-runtime selectors failed before value-expression execution on missing buffer roles

- **症状**:
  - A fresh run of
    `test_blackhole_t9_grouped_gemm_direct_runtime_bf16` failed before device
    execution with
    `missing explicit buffer role schema; direct runtime requires named input/output buffer bindings and must not recover output positionally`.
- **根因**:
  - The explicit buffer-role gate treated every `value_expr` as if it had to
    reference a `BufferLoad`.  After GEMM K-tile count, N-tile stride, and
    logical-z K offset moved to generic `value_expr`, those pure
    work/compute-context expressions had no referenced buffer and were
    misclassified as missing buffer-role schema.
- **修法**:
  - Let `value_expr` `BufferLoad`s contribute explicit input-buffer evidence,
    but treat `value_expr`s with no buffer loads as neutral for buffer-role
    binding.  They still evaluate through the work/typed-compute context.
    Do not work around this by reintroducing compute-shaped `value_source`
    enums.
- **验证**:
  - `test_blackhole_t9_grouped_gemm_projects_segmented_a_bindings` asserts the
    missing-buffer-role reason stays absent.
  - `test_blackhole_t9_grouped_gemm_direct_runtime_bf16` passed under TT-Sim
    after the gate fix.

### Direct-runtime value_expr fallback masked missing IR normalization

- **症状**:
  - Removing `Var.name_hint` fallback from `blackhole_module.cc` first exposed
    `Blackhole direct runtime value_expr requires work-dependent values to be
    normalized into explicit runtime_arg_u32 calls` on grouped GEMM per-work
    table expressions.
- **根因**:
  - `PlanTTKernelABI` recorded table-derived `value_expr`s while inside
    `blockIdx.*` scopes, but erased `block_index_source_by_var_` when leaving
    the scope.  The later ABI projection step therefore could not rewrite the
    stored block-axis `Var` inside `BufferLoad` indices.
- **修法**:
  - Treat `block_index_source_by_var_` as pass-local analysis that survives
    until per-work specs are projected.  Normalize work-axis `Var`s into
    explicit `tl.blackhole.runtime_arg_u32(...)` calls, and make direct
    runtime fail closed on any remaining naked `Var`.
- **验证**:
  - Focused projection tests assert dynamic grouped-GEMM value expressions
    contain `runtime_arg_u32` and no non-handle `tir.Var` nodes.
  - A minimal grouped-GEMM TT-Sim direct-runtime probe then reached enqueue but
    timed out after `180s`; treat that as remaining runtime/simulator work,
    not as permission to restore name recovery.

### Remote core descriptors were still leaf-recovered from logical_core_noc args

- **症状**:
  - An executable segment with `logical_core_noc_x/y` runtime args but no
    `remote_core_descriptors` field still built successfully.
  - The guard intended to forbid this did not fire because
    `test_blackhole_copy_pipeline.py` had duplicate dict keys for
    `rt_mod_blackhole.cc`; the later key masked the forbidden snippets.
- **根因**:
  - `rt_mod_blackhole.cc` rebuilt `KernelSpec.remote_core_descriptors` from
    the runtime arg pair during executable extraction and again during kernel
    materialization fallback.  That made the descriptor object a leaf-time
    reconstruction instead of explicit `ExecutableSpec` truth.
- **修法**:
  - Project `remote_core_descriptors` into executable segment records from
    typed ABI runtime args, with pair/identity/coordinate validation.
  - Make `rt_mod_blackhole.cc` parse only the explicit
    `remote_core_descriptors` field and let `BlackholeModule` validation fail
    when logical-core NOC args lack that descriptor.
- **验证**:
  - Source guard plus missing-descriptor, unpaired-arg, and descriptor
    materialization selectors report `4 passed`.
  - Worker semaphore producer/consumer direct-runtime selector reports
    `1 passed`.

### Leaf-time segment recovery duplicated or lost segment body statements

- **症状**:
  - `rt_mod_blackhole.cc` reconstructed per-segment bodies by reading
    `blackhole.segment_kind` from final TIR and inferring ambiguous CB ops
    from neighboring builtins.
  - When segment body selection moved earlier, the first generic extractor
    kept unmarked `Evaluate` leaves outside the requested marker, which copied
    retained input `cb_pop_front` statements into reader / compute / writer
    bodies.  After dropping unmarked leaves, compute then lacked the retained
    input pops because their generation point had never marked them as
    compute.
- **根因**:
  - Segment membership was split between pass-local markers, leaf-time
    recovery, and unmarked CB side effects.  The leaf reader could silently
    repair or corrupt ownership because the segment body was not explicit in
    `TTProgram` / `ExecutableSpec`.
- **修法**:
  - Add a generic `TTKernel.body` field and project it into executable segment
    records before stripping `blackhole.segment_kind`.
  - Delete the leaf-time `SegmentBodyExtractor`, neighbor inference, and final
    marker read from `rt_mod_blackhole.cc`.
  - Make `SegmentBodyFromMarkers` drop unmarked executable leaves outside the
    requested segment and wrap retained serial-loop input pops as compute
    segment statements where they are produced.
- **验证**:
  - The new GEMM segment-body guard first failed because reader carried
    `cb_pop_front(0/1, 4)`, then failed because compute lacked those pops,
    and passed after both root fixes.
  - `cmake --build build -j32` passed.
  - Focused guard/projection/runtime-schema selectors reported `8 passed`.
  - Baseline `test_blackhole_gemm_basic` timed out after `300s` in the
    current TT-Sim run, so direct GEMM correctness was not used as completion
    evidence for this checkpoint.

### Marker-free segment recording dropped `DeclBuffer`-wrapped producers

- **症状**:
  - After removing `blackhole.segment_kind` from active lowering, the seq64
    flash-attention exact-CB path failed the physical queue gate before
    runtime execution:
    `physical CB queue wait_front exceeds visible pages in main_kernel_compute`.
  - The lowered function body still contained the first row-reduction producer
    sequence, but the staged compute `TTKernel` seed started at later state
    fills and then consumed the reduction CB without that producer.
- **根因**:
  - Marker-free recording tracked concrete `Evaluate` / `BufferStore` leaves,
    but treated `DeclBuffer` as an opaque statement.  Row-reduction producers
    were wrapped in `Allocate(DeclBuffer(...))`, so the recorder skipped the
    nested leaves and the extracted segment body lost the CB producer.
- **修法**:
  - Treat `DeclBuffer` as a transparent wrapper during segment leaf recording.
  - Teach the segment-body extractor to reconstruct `DeclBuffer` only when the
    selected child leaves still use the declared buffer data var.
  - Keep the fix structural: no state-CB pruning, no queue-gate relaxation,
    and no marker attr reintroduction.
- **验证**:
  - `cmake --build build -j32` passed.
  - The segment-kind source guard passed.
  - Focused seq64 T7 direct-runtime selector passed after previously failing
    at queue admission.
  - Current P0/typed guards, T8 copy runtime selectors, and T7/T9 workload
    runtime selectors passed.

### T6 value/index direct runtime hung after enqueue from unresolved CB identities and BRISC lane overpublish

- **症状**:
  - T6 value/index selection source projection built, but TT-Sim direct
    runtime for fp32 and bf16 values stayed after
    `enqueue multi-core workload`.
  - Watcher changed the symptom into a BRISC NOC error on the first DRAM read,
    while a simple bf16 copy with the same DRAM base address still passed.
- **根因**:
  - Generated source mixed pre-allocation CB requirement indices with physical
    CB ids: reader wrote CB 7, compute waited on CB 21, compute published
    16/17, and writer waited on 1/4.  The executable record already carried
    the correct `requirement_indices -> cb_id` mapping, but codegen did not
    consume it for constant CB operands.
  - After CB identity was fixed, BRISC reader source still serialized the same
    loop-invariant input publish under the `threadIdx.x` lane loop, producing
    128 copies of an 8-page input event for a compute kernel that consumes one
    event.
- **修法**:
  - Load `cb_configs.requirement_indices` in `CodeGenBlackhole` and resolve
    all CB operation operands to the physical `cb_id` during source emission.
  - Make thread-lane use analysis follow the current core's emitted body so a
    loop-invariant CB publish is emitted once even when skipped compute-local
    stores mention `threadIdx.x`, and guard source projection against both
    unresolved requirement indices and CB reserve/publish under a thread loop.
- **验证**:
  - `cmake --build build -j32` passed.
  - Focused projection/source/schema selectors reported `4 passed`.
  - TT-Sim T6 direct runtime passed for bf16 values with int32 indices, fp32
    single-work, and fp32 multi-work.

### T6 reduce compute waited on an internal CB instead of the reader-published boundary CB

- **症状**:
  - After adding explicit compute operand CB links, the T6 fp32 single-work
    direct-runtime selector still hung after enqueue.
  - Introspection showed the primary reduce input binding resolved to the
    internal `logits_frag_reduce_src_0` requirement on physical CB 18, while
    the BRISC reader published the boundary `logits_frag` requirement on
    physical CB 21.
- **根因**:
  - The first implementation treated the reduction source staging CB as the
    compute operand boundary.  That made compute wait on a CB that no
    producer published.  The internal staging CB is real lifecycle state, but
    it is not the operand edge between reader and compute.
- **修法**:
  - Carry generic `TTComputeOperandBindingPlan.cb_requirement_indices` through
    TTProgram / ExecutableSpec / BlackholeModule.
  - During final TTProgram transport attachment, replace exact-CB non-output
    operand bindings with the allocated boundary exact-CB requirement indices
    for the same logical value.  Leave internal CBs represented by exact-CB
    lifecycle/allocation records.
  - Make codegen resolve compute operand CBs only through those requirement
    indices, not requirement names, output data formats, or generated suffixes.
- **验证**:
  - Structural T6 coverage asserts all reduce operand bindings carry valid
    requirement indices and that the primary reduce input CB id is among the
    reader-published CB ids.
  - TT-Sim T6 direct runtime passed for fp32 single-work, fp32 multi-work, and
    bf16 values with exact int32 indices after the fix.

### T6 scalarized compute-region body overproduced CB events after deleting the old emitter

- **症状**:
  - After deleting the old whole-kernel repeated-reduction emitter and letting
    the normal compute body source-render directly, T6 structural source tests
    passed but direct runtime hung during enqueue.
  - The emitted TRISC source serialized the `threadIdx.x` region as a C loop
    and published output CB pages inside that loop.
- **根因**:
  - The authored Tile TIR describes an axis-1 reduction region with reductions
    and scalar update loops, but leaf source emission cannot treat that
    GPU-style thread loop as an ordinary sequential loop with CB publication
    side effects.  It multiplies producer events beyond the writer's typed
    consume protocol and eventually stalls on CB/event availability.
  - The same experiment also exposed an old no-runtime-arg fallback in
    `GenerateGenericKernelMain` that loaded formal host-buffer pointers into
    executable compute kernels even when the body did not use them.
- **修法**:
  - Keep the T6 reduction-region source path as a typed executable
    compute-region lowering over compute records, logical tile layout, buffer
    distribution, and physical CB IDs, rather than raw scalar-body emission.
  - Delete the raw formal-argument fallback for executable kernels with no
    runtime args; executable source should consume projected runtime args and
    CB/accessor records, not original host pointer params.
- **验证**:
  - `cmake --build build -j32` passed.
  - Focused T6 source/schema selectors reported `4 passed`.
  - TT-Sim direct runtime passed for fp32 single-work, fp32 multi-work, and
    bf16 values with exact int32 indices.

### T6 reduction-region refactor added duplicate coordinate history and hit TT-Sim tile MMIO

- **症状**:
  - After renaming the T6 source path to a generic reduction region and adding
    channel-vector lowering, fp32 single-work direct runtime aborted with
    `UnimplementedFunctionality: t_tile_mmio_wr32`.
  - The previous row-shaped emitter had passed the same runtime selector.
- **根因**:
  - The first generic refactor added a separate coordinate-history array while
    the admitted T6 region already had an `Int32` coordinate output channel.
    That extra TRISC local state changed the generated kernel resource shape
    enough to hit a simulator tile-MMIO boundary.
  - Semantically the history is not a new channel: it is the same coordinate
    projection needed by repeated max suppression.
- **修法**:
  - Keep the region abstraction generic, but reuse the first `Int32`
    coordinate projection channel as repeat-history storage.
  - Only allocate a separate internal history array when a repeated reduction
    region has no coordinate projection channel.
- **验证**:
  - `cmake --build build -j32` passed.
  - Focused T6 structure/source selectors reported `4 passed`.
  - TT-Sim direct runtime passed for fp32 single-work, fp32 multi-work, and
    bf16 values with exact int32 indices.

### Output-CB retained-front pre-drain caused a 2-tile fused-dataflow hang

- **症状**:
  - The block-indexed 2-tile copy direct-runtime selector hung in TT-Sim.
  - Generated fused-dataflow source pushed the first output-CB page, then
    waited/popped it before reading the second page, and later tried to write
    two output tiles from only one remaining FIFO page.
- **根因**:
  - `RetainLocalCBFrontForFutureWaits` had been disabled for `role=output`
    pop adjustment, but its reserve-side drain insertion still treated the
    output CB as retainable state.
  - The pass also skipped front-page accounting for disabled output pops, so
    subsequent reserve handling could observe stale front depth.  Output CB
    pressure should be handled by the capacity-aware reserve pass, not by the
    generic retained-front rewrite.
- **修法**:
  - When retention is disabled for a physical CB, still account explicit pops
    in the local front-depth model.
  - Skip retained-front pre-reserve drain insertion for output CBs.  This
    preserves FIFO order for writer-visible pages while leaving real
    over-capacity cases to `InsertPhysicalPopsBeforeBlockingReserve`.
- **验证**:
  - The structured `KernelSpec.queue_events` regression for the 2-tile copy
    now asserts `reserve,push,reserve,push,wait,pop,wait,pop`.
  - The previously hanging TT-Sim selector
    `test_blackhole_module_direct_call_block_indexed_2tile_copy_uses_scaled_table`
    passed.
  - The broader T8 direct-runtime selector covering indexed, sparse, ragged,
    segmented, and paged copies reported `11 passed`.

### T8 indexed AccessRegion fallback masked missing owner evidence

- **症状**:
  - After deleting the remaining ABI-side `AccessRegion` reattachment helpers,
    T8 indexed/ragged/segmented/paged projection tests first failed with missing
    access-region diagnostics.
  - A three-range segmented copy then reused one generic per-work value arg for
    both `SegmentOffsets[bx, k]` and `SegmentCounts[bx, k]`, and the paged
    cache-length guard tried to bind an access region for raw `page_id` indices
    instead of the original `PageTable[bx, by]` expression.
- **根因**:
  - `FindSpatialAccessRegionRef(subject, kind, index_exprs)` still had a
    first-match fallback for same-buffer/same-kind regions when an indexed exact
    match failed.
  - ABI/transport subject-index collection did not consistently apply active
    Let substitutions, so it compared a later local variable shape against the
    original SpatialPlan expression.
  - Generic `value_expr` dedup compared expression shape but did not include the
    referenced table-buffer identity or `value_usage`.
- **修法**:
  - Make indexed access-region lookup fail closed when explicit structural
    `index_exprs` are present and no exact match exists.
  - Track active Let bindings through Blackhole lowering and apply them when
    deriving subject indices for per-work runtime args, including transport
    guard predicate values.
  - Validate every buffer-bound per-work spec in `ValidateTTProgram` against
    explicit access-region evidence, and include buffer-load identity plus
    `value_usage` in generic value-expression dedup.
- **验证**:
  - Focused T8 projection/source selector covering deleted recovery helpers,
    fail-closed indexed lookup, explicit page size owner truth, indexed, sparse,
    ragged, paged, segmented, three-range segmented, and per-work
    access-region negative coverage reported `19 passed, 66 deselected`.
  - TT-Sim direct-runtime selector covering indexed, sparse, ragged, paged,
    one/two/three-range segmented copies reported `13 passed, 35 deselected`.

### T9.5 loop-carried state CB aliasing hung TT-Sim

- **症状**:
  - The first chunk-scan direct-runtime shape hung when generated compute used
    the same physical CB for the loop-carried state input and output, or when
    the writer consumed the state CB directly for per-chunk output.
- **根因**:
  - The loop-carried state live-in/backedge CB and writer-visible publication
    stream have different ownership.  Sharing the physical CB lets the writer
    steal the state page or makes compute publish a self-backedge that the
    simulator/runtime cannot progress.
- **修法**:
  - Keep the state lifecycle in typed exact-CB records, but render the first
    admitted three-chunk recurrence with alternate state CBs and a distinct
    writer publication CB.  The `X` input stream is retained as a three-page
    loop window and popped at the final chunk.
- **验证**:
  - `test_blackhole_t9_chunk_scan_bf16_direct_runtime` passed under the
    repository TT-Sim bf16 setup.
  - The full T9.5 chunk-scan file reported `2 passed`.

### CB queue events drifted back into runtime body recovery

- **症状**:
  - The protocol docs said `KernelSpec.queue_events` were structured
    executable facts, but `rt_mod_blackhole.cc` still rebuilt them by scanning
    segment-body TIR during runtime-module construction; later, projection
    still parsed `TTKernel.body`, so body-only mutations could change the
    executable queue trace.
- **根因**:
  - The TTProgram projection carried kernel bodies before it had a typed
    `TTKernel.queue_events` contract.  That left runtime, and then projection
    itself, as secondary owners of queue semantics.
- **修法**:
  - Record `cb_reserve_back`, `cb_push_back`, `cb_wait_front`, and
    `cb_pop_front` into `TTKernel.queue_events`; refresh that typed field after
    allocation-time kernel-body rewrites; project executable segment
    `queue_events` only from `TTKernel.queue_events` plus
    `TTCBPlan.requirement_indices`.
  - Make runtime parse only the projected event array and delete the
    runtime/projection body-scanner paths.
- **验证**:
  - Added a source guard that fails if runtime body-recovery helpers return.
  - Added a projection source guard that fails if queue events are projected by
    parsing `TTKernel.body`.
  - Added a behavior regression where appending a body-only CB queue call to a
    compute `TTKernel.body` does not change projected `KernelSpec.queue_events`.
  - The typed tile-CB verifier suite passed with structured event projection.

### Codegen recovered runtime buffer bindings from final TIR bodies

- **症状**:
  - `CodeGenBlackhole::EmitRuntimeArgLoads` scanned the final TIR body for
    `BufferLoad` / `BufferStore` and `tl.blackhole.read_*_to_cb` /
    `write_*_from_cb` calls to repopulate runtime-backed buffer handle
    mappings after packed-entrypoint lowering.
- **根因**:
  - The executable runtime arg schema already carried the exact bound buffer,
    but codegen still treated the final leaf body as a second owner for buffer
    ABI semantics when formal params or `buffer_map` were unavailable.
- **修法**:
  - Make projected `ExecutableSpec.runtime_args[].buffer` the primary binding
    source.  Codegen records the explicit buffer-name mapping unconditionally
    and only adds pointer-keyed bindings when the current function signature or
    `buffer_map` directly exposes the same handle.
- **验证**:
  - Added a source guard for the deleted body-recovery strings.
  - `cmake --build build -j32` passed.
  - TT-Sim direct-runtime selectors for page-addressed copy and sharded T3
    bf16 execution passed with the repository setup.

### Runtime recovered host launch association from packed host bodies

- **症状**:
  - `rt_mod_blackhole.cc` mapped host entries to device executable specs by
    scanning packed host TIR for `tvm_call_packed` string callees that matched
    Blackhole device kernel symbols.
- **根因**:
  - `LowerDeviceKernelLaunch` knew the cross-target launch target when it
    rewrote the call, but it only encoded that target as a call argument in the
    rewritten host body.  Runtime then became a second semantic owner by
    reading the lowered body shape.
- **修法**:
  - Record launched kernel symbols as the explicit `tl.launched_kernel_symbols`
    IR attr on the host PrimFunc during `LowerDeviceKernelLaunch`.
  - Make Blackhole runtime consume the attr and fail closed if one host entry
    names multiple Blackhole device kernels under the current single-association
    module contract.
- **验证**:
  - Added a structural test that the host PrimFunc carries
    `tl.launched_kernel_symbols == ["main_kernel"]` for a staged copy kernel.
  - Added a source guard that `FindLaunchedKernelSymbol` stays deleted.
  - `cmake --build build -j32` passed.
  - TT-Sim direct-runtime selectors for page-addressed copy and sharded T3
    bf16 execution, plus the tvm_ffi export host-shim selector, passed.

### Runtime recovered materialization shape facts from device bodies

- **症状**:
  - Buffer materialization and multidimensional per-work descriptor admission
    in `rt_mod_blackhole.cc` used `CollectStaticBufferInfo` to scan device
    `PrimFunc` `buffer_map` and final bodies for buffer load/store shapes.
- **根因**:
  - The executable already carried tensor memory-config logical shapes, but
    runtime still treated the device body as a second source of materialization
    rank/dtype truth for host-axis-order and descriptor gates.
- **修法**:
  - Delete the body/buffer-map scanner and derive a local
    `buffer -> logical_shape` map from
    `ExecutableSpec.tensor_memory_config_plans`.
  - Make conflicting logical shapes for one executable subject fail closed.
- **验证**:
  - Added a source guard that `CollectStaticBufferInfo` stays deleted.
  - `cmake --build build -j32` passed.
  - TT-Sim selectors for host shim export, page-addressed copy, and sharded T3
    bf16 direct runtime passed.

### Codegen recovered reduction-region semantics from final bodies

- **症状**:
  - `CodeGenBlackhole::EmitTypedReductionRegionIfSupported` consumed typed
    compute operands and CB bindings, but recovered reduction kind, reduction
    dimension, and loop repeat extent by scanning the final TIR body.
- **根因**:
  - Reduce execution facts were not owned by `TTComputeOpPlan`, so codegen
    treated builtin neighborhoods and serial loop structure as a second owner
    of target execution semantics.
- **修法**:
  - Add typed `reduction_kind`, `reduction_dim`, and `repeat_extent` fields to
    reduce `TTComputeOpPlan` records, validate them in `ValidateTTProgram`,
    project them into `ExecutableSpec.compute_ops`, and make Blackhole codegen
    consume only those executable records.
- **验证**:
  - Added a source guard that `InferReductionSignature` and
    `InferReductionRepeatExtent` stay deleted.
  - Added projection assertions for topk reduce ops.
  - `cmake --build build -j32` passed.
  - TT-Sim topk runtime file passed with `7 passed`.
  - TT-Sim host-shim, page-addressed copy, and sharded T3 bf16 selectors
    passed.

### Guarded row-page and guard-mask CB events used mismatched thread ownership

- **症状**:
  - Paged GQA decode hung at `EnqueueMeshWorkload` because guard-mask CBs had
    one physical page but every `tx` lane reserved and pushed that page.
  - T8 ragged row and sparse/ragged row-page copies hung when the reader was
    guarded to `tx == 0` but the writer loop still waited and popped from every
    `tx` lane.
- **根因**:
  - The leaf source represented one per-work FIFO event, but source generation
    did not apply a consistent active-thread owner predicate to both producer
    and consumer sides.  This was an execution-event ownership bug, not a GQA
    or sparse workload special case.
- **修法**:
  - Add a shared `PlanTTKernelABI::WrapActiveThreadSinglePublication` helper and
    apply it to guard-mask source publication, guarded row-page reader
    publication, and guarded row-page writer consumption.
  - Keep the fix structural: derive the predicate from active TIR thread vars
    already tracked by lowering, not from buffer names or workload labels.
- **验证**:
  - Added source guards for guard-mask CB publication and guarded row-page CB
    producer/consumer ownership.
  - `cmake --build build -j32` passed.
  - TT-Sim direct-runtime selectors for paged GQA, T8 ragged/sparse row-page
    copies, full T8 indexed/ragged/paged/segmented copy gates, and the active
    T7/T9 workload group passed.

### Terminal flash decode publication repacked from a stale local fragment

- **症状**:
  - Paged and split-block flash decode direct runtime produced an all-zero
    final output even though the intermediate exact/live CB value held the
    expected partial-combine result.
- **根因**:
  - Final local-to-CB publication treated a slice-shaped local fragment as the
    value owner and repacked from that local storage.  On this path the local
    fragment was only a witness for the terminal value; the actual complete
    tile was available through the exact/live CB lifecycle.
- **修法**:
  - Allow terminal local-to-CB slice lowering to republish from a matching
    full-tile exact/live CB when the logical matrix shape and element count
    match the destination publication.
- **验证**:
  - `cmake --build build -j32` passed.
  - T7 paged decode and T9.6 split-block flash decode bf16 direct-runtime
    selectors passed under the repository TT-Sim setup.

### Retained stream input rewrite offset the first grouped GEMM wait

- **症状**:
  - Grouped GEMM direct runtime produced wrong rows after compute source
    generation rewrote `matmul_tiles` to read logical tile indices `3..6`
    after the first `cb_wait_front(cb, 4)`.
- **根因**:
  - The retained stream-input rewrite interpreted the initial multi-page
    logical wait depth as retained front history.  For grouped GEMM the four
    pages are the first logical GEMM input event, so tile reads must start at
    zero.
- **修法**:
  - Only advance the active event base when a wait observes previously retained
    front pages.  Initial absolute-depth waits with no retained base keep tile
    reads at logical index zero.
- **验证**:
  - `cmake --build build -j32` passed.
  - The grouped GEMM bf16 direct-runtime selector passed under the repository
    TT-Sim setup.

### Exact-CB local state was not admitted after CB-to-local untilize

- **症状**:
  - `test_tile_compute_dag_feeds_typed_resource_pressure_report` failed in
    `PlanTTCompute` for `acc_o` because a clear-accum=false GEMM could not
    prove a legal loop-carried reload.
  - The source guards for live-form solver literals and subject live-value maps
    also failed while the state layer still owned physical-form decisions.
- **根因**:
  - Exact-output live-CB evidence was stored as a current map, so planning could
    not distinguish a valid prior producer from a future marker.
  - The CB-to-local `blackhole_untilize_cb_front_tile_fragment` event was not
    recorded as typed local exact-CB state, so the following GEMM only accepted
    zero-fill local state.
  - Exact-CB virtual live-form creation collapsed a materialization boundary's
    source and target decisions into the target CB-materialized form.
- **修法**:
  - Keep exact-output live-CB history by lowering order and only use the latest
    record visible at the current program point.
  - Record CB-to-local untilize as local exact-CB live state and allow
    clear-accum=false GEMM reload when loop-carried evidence and shape checks
    match.
  - Resolve exact-CB virtual live forms through the indexed
    `SpatialPlan` materialization boundary and select the TT live-form solver's
    source or target decision for that boundary.
- **验证**:
  - `cmake --build build -j32` passed.
  - Full `test_blackhole_spatial_ir.py` passed.
  - Small bf16 flash-attn and seq64 MHA exact-CB partial-combine direct-runtime
    selectors passed under the repository TT-Sim setup.

## 3. 环境问题速查

| 问题 | 解决 |
|------|------|
| `pip install -e .` 失败 | 用 `.pth` 指向本地构建产物 |
| Python 加载旧库 | 统一使用 `tilelang_repo/build/` 单一构建目录，并确认已重编 |
| TT-Sim 报 `Root Directory is not set` | 设置 `TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME` |
| TT-Sim 报 `No chips detected` | 设置 `TT_METAL_SIMULATOR` 与 `TT_METAL_MOCK_CLUSTER_DESC_PATH` |
