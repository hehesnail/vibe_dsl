# 问题与 Bug 记录

> 本文档只保留仍然有复用价值的问题记录。已解决且无复用价值的条目已归档删除。

## 未解决

### flash-attention `mha` direct runtime 在 enqueue 后仍 hang，当前死锁签名稳定收敛到 execution-time CB/synchronization 协议

- **时间**: 2026-04-01
- **问题**: 在修掉 `write_local_slice_to_cb` CB ID 回写、runtime arg dedup、`fused_dataflow` buffer args 过滤、`cast_fragment_slice` 漏发页数、以及 `blackhole.acc` GEMM 输出重复 reserve 后，`test_blackhole_flash_attention_runtime.py -k mha_forward_direct_runtime` 仍会在 workload enqueue 后 hang
- **当前现象**:
  - direct runtime 已能 build、launch，并进入 workload execution
  - TT-Metal Watcher 当前稳定复现同一组状态码：reader `CRBW`、writer `CWFW`、compute `MWDD`
  - 说明当前剩余 blocker 已经不是 compile/codegen/build-time 层，而是 execution-time 的同步/CB/dataflow 协议
- **当前推断**:
  - `blackhole.acc` 当前仍混合承担 fragment scratch 与 CB queue 两类语义
  - 虽然已修掉两个局部协议错误，但完整 reserve/publish/wait/consume 时序可能仍与 TT-Metal 对 compute-local scratch/CB 生命周期的期望不一致
- **下一步方向**:
  - 继续沿 `acc_s / acc_s_cast / acc_o` 等 scratch CB 做完整生产-消费时序核对
  - 结合 Watcher / waypoints，把 hang 从“某个 kernel 没返回”缩到“compute 的具体阶段没推进”

### Blackhole direct path 缺少 TT-Metal 正式 contract 分层

- **时间**: 2026-03-26
- **问题**: 当前 Blackhole schema 仍未完全覆盖 TT-Metal 正式 contract，剩余缺口主要在更宽的 execution surface：host logical tensor layout 泛化、更丰富 dtype/compute ABI、以及 sharded/non-tile/common-runtime accessor 执行面
- **影响**: copy 在最简单 tile/interleaved case 上可通过，但更复杂场景无正式 schema 承载
- **解决方向**: 按 `stage2d_ttmetal_contract_audit.md` 的 P0-P5 分层推进；当前 P0 已完成到统一 `compute_contract`，P1、P2、P3 主路径 formalization 已落地，后续继续做 P4-P5 和更宽的 P3 execution surface
- **当前状态**: 部分解决。P0 已完成：dtype 分层、compute config、以及 `dst_full_sync_en/bfp8_pack_precise/defines/named_compile_args` 已走通 DSL producer -> `compute_contract` -> `ExecutableSpec/KernelSpec` -> direct runtime 主链。P3 richer runtime work schema 已落到 `work_linear_id` + role-explicit `a/b/output/k` descriptors；accessor schema、`common_runtime_args`、`compile_time_arg_specs`、`launch_spec` 已进入 segment/kernel schema 并被 direct runtime 消费。P4 已完成最小 interleaved stick/page copy 主路径：`transport_page_size` 进入 accessor schema，`32x16` row-major/stick copy 已通过 TT-Sim。当前 remaining gap 是更广泛的 range/stride/batch、sharded accessor，以及更宽 execution surface。

### direct runtime 若不先把 output tensor 初值同步到 device，partial-write copy 会读回脏数据

- **时间**: 2026-03-30
- **问题**: 最小 stick/page copy 只覆盖 output tensor 的一部分页面；单独运行时看起来可能正确，但在整份 `test_blackhole_copy_runtime.py` 顺序执行时，未覆盖区域会读回脏数据，导致 `test_blackhole_module_direct_call_stick_copy` 随机失败
- **根本原因**:
  - `BlackholeModule::ExecuteDirect` 之前只把 input buffer 写到 device，output buffer 只分配不初始化
  - tile-aligned full overwrite case 不会暴露这个问题，因为 kernel 会覆盖整个 output
  - stick/non-tile partial-write case 会保留未写区域，因此 device buffer 的历史内容会泄漏回 host
- **解决**:
  - direct runtime 统一在执行前把所有 host tensor 当前内容同步到对应 device buffer
  - 不再假设 output buffer 一定会被 kernel 全覆盖
- **教训**:
  - host direct runtime 不能把“output”当成“无需初始化”的同义词；只要 schema 允许 partial write，output tensor 初值就是 contract 的一部分
  - 顺序相关失败优先怀疑“未初始化设备内存 + 部分写入”，不要只盯 JIT 缓存或 simulator 污染

### stick/page copy 若 transport page 未对齐 64B，会在 TT-Metal NOC 层触发地址对齐错误

- **时间**: 2026-03-30
- **问题**: interleaved stick/page copy 在 schema、lowering、codegen 都看起来正确时，TT-Sim 仍可能在 runtime 报 `alignment of src_addr ... and dst_addr ... does not match`
- **根本原因**:
  - 当前 direct path 的 stick transport 走 `TensorAccessor(...).get_noc_addr(page_id)` + `noc_async_read/write`
  - 当 `transport_page_size` 不是 64B 对齐时，例如 `tile_n=24, dtype=float32 -> page_bytes=96`，page-based NOC 传输会落到 TT-Metal 的地址对齐约束
  - 这不是 pipeline/schema 丢字段，而是当前正式执行面尚未覆盖更一般的 unaligned stick width
- **解决**:
  - 把 `transport_page_size` 明确保留在 accessor schema
  - `LowerBlackholeOps` 对 stick/page transport 新增统一 fail-fast：`page_bytes % 64 == 0`
  - 进一步补齐 direct-path 边界：source / destination offset 必须 page-aligned，global width 必须能整除 shared width
  - 现阶段正式支持 `64B` 对齐且 page-aligned 的 interleaved row-major stick transport；未对齐或 non-divisible case 在 lowering 阶段直接拒绝
- **教训**:
  - 对 TT-Metal stick/page transport，pipeline 看起来“语义对了”不代表 runtime 一定可执行；还要检查底层 NOC/page 对齐约束
  - 对 row-major stick transport，不能只检查单次 `page_bytes`；source/destination 是否落在整页边界、全局 width 是否仍满足整页推进，也属于同一个 transport contract
  - 当 direct path 还没覆盖更宽执行面时，应把边界收成 schema/lowering fail-fast，而不是把用户带到 runtime 才撞底层地址错误

## 已解决（仍有复用价值）

### `cast_fragment_slice` 若把 `blackhole.acc` 结果写到后续 matmul 输入 CB，却不按未来 matmul 需求 `cb_push_back`，compute 会在第二次 matmul 前挂死

- **时间**: 2026-04-01
- **问题**: flash-attention compute source 里，`acc_s_cast` 这类由 `cast_fragment_slice` 产出的 `blackhole.acc` scratch 结果会被后续第二次 matmul 继续消费；旧 lowering 只做了写入，没有按 future matmul 所需页数正式发布到 CB 生命周期
- **根本原因**:
  - `cast_fragment_slice` 之前只被看成局部 pointwise/cast 结果，没有把“后续仍会被 matmul 作为 CB 输入消费”这层生命周期一起编码
  - 结果 compute source 会在第二次 `mm_init` 前缺少对应的 `cb_push_back`
  - consumer 侧等待不到完整页数，最终表现成 execution hang
- **解决**:
  - `LowerBlackholeOps` 预扫描 future matmul consumer，记录 `blackhole.acc` data name 到所需页数
  - 对会被 future matmul 消费的 fragment cast 结果，`GenerateFragmentCastSequence` 按该页数正式 `cb_push_back`
  - 新增 compute-source 回归，要求 `acc_s_cast` 在第二次 matmul 前发布 4 页
- **教训**:
  - scratch CB 的生产者不只要“写进去”，还要按未来消费者的协议“正式发布出去”
  - 当 pointwise/fragment 结果继续进入 matmul 时，生命周期应该按 matmul input CB 看，而不是按 pointwise op 自己的局部语义看

### `GenerateMatmulSequence` 若对 `blackhole.acc` GEMM 输出沿用 transport-CB 模板重复 reserve，会破坏 scratch CB 生命周期

- **时间**: 2026-04-01
- **问题**: flash-attention compute source 里，`acc_s` / `acc_o` 这类 `blackhole.acc` GEMM 输出 CB 在 `pack_tile` 前又被 `cb_reserve_back` 一次，和已经持有的 scratch CB 生命周期冲突
- **根本原因**:
  - 旧 `GenerateMatmulSequence` 无条件沿用普通 transport/output CB 模式
  - 但 `blackhole.acc` 输出在 flash-attn 里本质上是 compute-local scratch，和 reader->compute->writer 的 FIFO output CB 不是同一类资源
  - 在这种 scope 上重复 reserve，会把 scratch storage 和 queue 生命周期混成两套互相冲突的协议
- **解决**:
  - matmul lowering 在识别 GEMM 输出 scope 为 `blackhole.acc` 时，不再在 `pack_tile` 前重复 `cb_reserve_back`
  - 新增 compute-source 回归，要求 `blackhole.acc` GEMM 输出前不再出现重复 reserve
- **教训**:
  - 不能把 transport CB 的模板机械套到 scratch CB 上
  - 同一个 CB id 若同时承担 storage 与 queue 语义，必须先明确哪一套生命周期才是真协议，再决定是否 reserve/push

### Blackhole TRISC compute source 若漏接最终 `exp2f` 形态，会把 libc/newlib math 依赖重新带回 device link

- **时间**: 2026-04-01
- **问题**: flash-attention forward 的 compile/codegen 主链已经把 `exp2` 相关 fragment builtin 接上后，compute kernel 仍可能在最终 source 中残留 `exp2f(...)`，随后在 TT-Metal TRISC link 阶段触发额外 libc/newlib math 依赖，表现成 RW segment 膨胀或链接失败
- **根本原因**:
  - `tir.exp2` 在前段不一定以同一种形态走到最终 codegen
  - 即使 IR 级别已经是 builtin 或 affine helper，最终仍可能在 `codegen_blackhole` 末端回退成 `T.call_pure_extern("exp2f", ...)` / `T.call_extern("exp2f", ...)`
  - 如果 backend 只拦截早期 `tir.exp2` 形态，而漏掉最终 `exp2f` 形态，TRISC source 就会重新显式引用 libc math
- **解决**:
  - `codegen_blackhole` 统一拦截 `tir.exp2`、`call_pure_extern("exp2f", ...)`、`call_extern("exp2f", ...)`
  - backend 内部统一走自带 fast-math helper，而不是继续把 `exp2f` 发给 device toolchain
  - 新增 compute-source 回归，要求 flash-attn compute source 不得再出现直接 `exp2f` / `std::exp2`
- **教训**:
  - 对 device math builtin，codegen 不能只看“理想 IR 形态”；还要覆盖最终可能落到打印器的外部调用形态
  - 一旦 TRISC link 开始报 RW segment / libc 相关问题，优先检查最终 emitted source 是否重新漏出了标准库 math 调用

### `AssignBlackholeCores` 若继续产出旧的 physical-style 坐标，会和 direct runtime 的 logical worker grid contract 冲突

- **时间**: 2026-04-01
- **问题**: flash-attention runtime 在 compile/codegen 主 blocker 去掉后，direct runtime 先后暴露 `No core coordinate found at location: (14, 2, TENSIX, LOGICAL)`、core range 越界和 launch/core 映射错位
- **根本原因**:
  - `AssignBlackholeCores` 仍沿用了旧的 `14 x 10` physical-style 坐标直觉
  - 但当前 direct runtime / TT-Metal worker launch contract 消费的是连续 logical worker grid，而不是物理/NOC 风格坐标
  - 当 planner 和 runtime 不在同一套 core descriptor 语义上时，问题会先表现成 lookup 失败，进一步又演变成更脏的 launch/runtime 噪声
- **解决**:
  - 把 planner/runtime 统一收正到 logical worker grid contract
  - 当前环境固定为 `11 x 10 = 110` worker cores，`core_idx -> (x, y)` 按连续逻辑坐标线性化
  - `rt_mod_blackhole` 的 target metadata 也同步改成 `num_cores = 110`
- **教训**:
  - `core_plan` 里的 core descriptor 必须明确是“logical worker coords”还是“physical/NOC coords”，不能让 planner/runtime 各自猜
  - 多核 direct runtime 一旦出现 core lookup / core range 异常，优先检查是不是坐标语义混用了 logical 与 physical 两套系统

### flash-attention runtime 若已越过 compile/launch blocker 仍在 enqueue 后挂住，剩余问题通常已收敛到执行期同步或 CB 协议

- **时间**: 2026-04-01
- **问题**: 在修掉 flash-attn 的 CB pointer codegen、`exp2f` libc 依赖、以及 core-plan 坐标协议后，`test_blackhole_flash_attention_runtime.py -k mha` 已能完成 build 和 workload enqueue，但执行阶段会长时间卡住，精度对比无法完成
- **根本原因**:
  - **已定位**：`PlanBlackholeCB` 的 `GetCBArgPositions` 漏掉了 `tl.blackhole.write_local_slice_to_cb`
  - 该 builtin 的 cb_id 参数（position 1）没有被回写成最终分配的 cb_id，停留在 requirement_index 值
  - 表现：compute kernel 里 `cb_reserve_back(24, 16)` / `cb_push_back(24, 16)` 操作 CB 24（正确），但 `write_local_slice_to_cb` 通过 `get_local_cb_interface(11)` 写 CB 11（错误的 requirement_index）
  - writer kernel `cb_wait_front(24, ...)` 等待 CB 24 的数据永远等不到 → hang
  - 更广泛的根因是 CB ID 回写协议依赖手动枚举，新增 builtin 容易遗漏
- **解决**:
  - 在 `GetCBArgPositions` 中注册 `write_local_slice_to_cb` 的 cb_id position
  - 在 `PlanBlackholeCB` 回写后新增 post-condition 校验，防止未来再遗漏
- **状态**: **已修复** (2026-04-01)
- **教训**:
  - 当 runtime 已经能进入 enqueue/execution，却只表现为 hang，应该把排查焦点转到执行期同步与 CB 协议，而不是继续在 compile-path 上堆 workaround
  - CB ID 回写是 “add new builtin → must register” 的协议约束，单靠手动枚举不够安全；需要 post-condition guard
  - 这类进展要明确记录成”问题层级前移”，否则容易误判成仍停留在旧 blocker

### `ExtractRuntimeArgs` identity dedup 过于激进，会丢弃同 identity 不同 kind 的 runtime arg

- **时间**: 2026-04-01
- **问题**: `test_blackhole_copy_remote_core_descriptor_is_materialized` 失败，`BlackholeModuleNode` 构造时报 “remote_consumer_core must define both logical_core_noc_x and logical_core_noc_y”
- **根本原因**:
  - `ExtractRuntimeArgs` 在聚合 segment runtime args 时用 `arg.identity` 作为 dedup key
  - `logical_core_noc_x` 和 `logical_core_noc_y` 共享 identity `”remote_consumer_core”` 但 kind 不同
  - dedup 导致第二个 arg 被跳过，只保留了 `_noc_x`，`_noc_y` 丢失
  - 此 bug 在 commit `3fb1f37` 引入：把 dedup key 从 `arg.kind + “:” + arg.name` 改成了 `arg.identity`
- **解决**: 将 dedup key 改为 `arg.identity + “:” + arg.kind`，同时修 `ExtractCommonRuntimeArgs`
- **状态**: **已修复** (2026-04-01)
- **教训**: identity 是分组标识（同一个 remote core 的 x/y），不是唯一标识。dedup key 必须包含 kind 才能区分同组内的不同角色

### `MakeSegmentPrimFunc` fallback_arg_allowed 对 fused_dataflow 过滤掉所有 buffer args

- **时间**: 2026-04-01
- **问题**: copy pipeline 回归失败，报 “Missing runtime arg binding for buffer var: A”
- **根本原因**:
  - `fallback_arg_allowed` 在 `fused_dataflow` segment kind 下之前的默认返回值为 `false`
  - 这导致所有 input/output buffer args 都被过滤掉，fused_dataflow kernel 拿不到任何 buffer 绑定
- **解决**: 将默认返回值改为 `true`，fused_dataflow 和其他 segment kind 同时需要 input 和 output buffer args
- **状态**: **已修复** (2026-04-01)

### flash-attention forward 的 `local/accumulator -> shared(CB)` staged copy 若不进入正式 copy direction，会把真实 blocker晚报成 residual shared store

- **时间**: 2026-03-31
- **问题**: 在 fragment analysis、row reduction / row broadcast / fill / scalar max / cast slice lowering，以及对应 builtin codegen 都接上后，flash-attn forward 的 full `lower()` 仍然卡在 residual 二维 `BufferStore`
- **根本原因**:
  - 优化后的 device IR 里残留类似 `O_shared_1[tx, i * 8 + vec] = O_shared_local_cast[vec]` 的 staged writeback
  - 这不是普通 shared store，而是 fragment/local 结果写回 CB staging 的语义
  - 如果 `LowerBlackholeOps` 不把它收成正式 direction/builtin，这类写回就会晚漏到 codegen，表现成 shared 非扁平 store 或其他噪声错误
- **解决**:
  - 新增 `CopyDirection::kLocalToCB`
  - 新增 `tl.blackhole.write_local_slice_to_cb`
  - `LowerBlackholeOps` 识别并 lower `local/accumulator -> shared(CB)` staged copy
  - `codegen_blackhole` 发射对应的 CB write primitive，不再让 residual 二维 `BufferStore` 漏到后段
- **结果**:
  - 当前支持的 MHA/GQA flash-attn forward compile-path 已打通
- **教训**:
  - 对 Blackhole/TT-Metal，`local` 只是中间态，不应该作为最终资源语义长期留在后段
  - 一旦某类 `local` 明显处在 fragment 结果写回 CB 的桥接位置，就应尽快收成正式 dataflow primitive，而不是靠 codegen 去兜二维 store

### copy 若缺失显式 runtime arg schema，不应退回 `input0/output0` 默认 ABI

- **时间**: 2026-04-01
- **问题**: `rt_mod_blackhole` 里保留的 `MakeDefaultCopyRuntimeArgs()` 会在 copy/dataflow kernel 缺失 `blackhole.runtime_args` / `segment_plan[*].runtime_args` 时，退回到 `input0/output0` 这类默认 ABI
- **根本原因**:
  - 这是早期 bring-up 残留的 fallback
  - 它让正式主链在 schema 缺失时仍试图继续 build，问题会晚到 codegen 报 `Missing runtime arg binding for buffer var: A`
- **解决**:
  - 删除默认 copy runtime-arg fallback
  - `ExtractRuntimeArgs()` 对含 copy builtins 的 kernel 改为 build-time 显式失败：必须有 IR/segment 提出来的 runtime arg schema
- **教训**:
  - 正式主链里，不要保留“默认参数数量/顺序/名字”的 ABI 兜底
  - 如果 runtime ABI 本来应该由 IR/schema 提供，那 schema 缺失时就该尽早 fail-fast，而不是退回通用占位名字

### device-only codegen 若绕过 `ExecutableSpec` gate，会把 fragment 子集缺失晚报成 `Find undefined Variable acc_o`

- **时间**: 2026-03-31
- **问题**: 在 flash-attn forward 的 `row_broadcast` lowering 进一步收窄后，full `lower()` 不再先撞 `rt_mod_blackhole` 的 build-time gate，而是晚到 device-only codegen 报 `Find undefined Variable acc_o`
- **根本原因**:
  - `rt_mod_blackhole` 的 fragment-subset gate 只覆盖 `ExecutableSpec` 提取路径
  - `BuildTileLangBlackholeWithoutHost` 生成 device code 时会直接走 `codegen_blackhole`
  - 一旦该路径绕过 spec 提取层，尚未 lower 的 fragment local loop 就会在 codegen 里以未绑定局部变量形式爆炸，错误层级明显滞后
- **解决**:
  - 保留 `rt_mod_blackhole` 的 unsupported-op gate
  - 在 `codegen_blackhole` 入口补同一套基于 `blackhole.lowering_requirements` / `pointwise_op_kinds` 的 gate
  - 让 device-only codegen 与 `ExecutableSpec` 路径在 `fill / max / add / cast` 等尚未 lower 的 pointwise 子集上共享同样的 fail-fast 口径
- **教训**:
  - 对同一 lowering 边界，如果仓库里存在多条后端出口，就不能只在其中一条出口做语义 gate
  - 一旦错误重新晚到 codegen 内部报“undefined variable”或类似噪声，优先检查是不是另一个出口绕过了你以为已经建立的 schema/spec 边界

### flash-attention 的 `row_broadcast` 若一直作为整类 blocker 保留，会掩盖哪些广播 loop 已经具备稳定 TIR lowering 形态

- **时间**: 2026-03-31
- **问题**: 在 `row_reduction` 已 lower 后，flash-attn forward 的 full `lower()` 仍统一报 `row_broadcast` 未支持，看起来像所有广播路径都同样缺失
- **根本原因**:
  - `row_broadcast` 实际包含多种不同复杂度的 fragment loop
  - 其中最简单的一类已经有稳定 optimized-path TIR 形态：`dst[i] = dst[i] * scalar[0]` / `dst[i] = dst[i] / scalar[0]`
  - 如果继续把整类 `row_broadcast` 一起 gate 掉，就会掩盖哪些 loop 已经可以在 `LowerBlackholeOps` 里直接匹配和 lower，哪些仍需要更深的 compute role / canonicalization 支撑
- **解决**:
  - 新增 `tl.blackhole.mul_row_bcast` / `tl.blackhole.div_row_bcast` builtin
  - `LowerBlackholeOps` 新增对最小 row-broadcast 子集的 matcher，先吃掉 `acc_o *= scores_scale[0]` 和 `acc_o /= logsum[0]`
  - 再把 `logsum = logsum * scores_scale + scores_sum` 这类 scalar fragment 融合更新单独 lower 成 `tl.blackhole.scalar_fma`，不要继续把它和 vector broadcast 混成同一个 blocker
  - 新增 optimized-path pipeline 回归，要求这两条 loop 已被 lower 成 builtin，而不是继续残留为普通 fragment store
- **教训**:
  - 对复杂 fragment compute blocker，应该优先把“可稳定 lower 的最小子集”从整类 gate 中拆出来，而不是一直把整个大类当黑盒
  - 对 flash-attn 这类 kernel，progress 最稳的方式是让 gate 随真实 lowering 一步步收窄：先 simple broadcast，再 scalar update，最后再做 fused broadcast；不要企图一步把所有 `row_broadcast` 全开

### flash-attention 的 row-reduction lowering 若只匹配 split-after 形态，会在 full `lower()` 的 optimized path 上残留旧 blocker

- **时间**: 2026-03-31
- **问题**: 手动串 `LowerAndLegalize -> SplitBlackholeKernel -> Analyze* -> LowerBlackholeOps` 时，flash-attn forward 的 `row_reduction` 已经能 lower；但真实 full `lower()` 路径仍会在 `rt_mod_blackhole` 报 `row_reduction, row_broadcast` 未支持
- **根本原因**:
  - 真正差异不在 `rt_mod_blackhole`，而在 `OptimizeForTarget` 之后的 device IR 形态
  - split-after 形态里的 direct sum/max reduction 常带 `for extent=1` 包装，旧 matcher 正好能命中
  - optimized path 里，这些 reduction 会被改写成同级 `SeqStmt`，同时经常带 `AttrStmt(pragma_unroll_explicit)` 包裹；max 临时归约也不再一定额外包一层 `for extent=1`
  - 结果是同一逻辑 reduction 在手动路径能 lower，在 full `lower()` 上却被漏掉，最后又被 build-time gate 当成旧 blocker 报出
- **解决**:
  - `LowerBlackholeOps` 新增对同级 `SeqStmt(init_store, reduce_loop)` 的 direct row-reduction 匹配
  - `MatchAllocatedRowReduction` 改为同时接受直接 `SeqStmt` 和单层 `for extent=1` 包装
  - reduction matcher 统一先剥掉无语义的 `AttrStmt` 包装，再匹配 `ForNode` / `BufferStoreNode`
  - 新增 optimized-path 回归，要求 post-`OptimizeForTarget` 的 `LowerBlackholeOps` 也必须产出 `tl.blackhole.reduce_row`
- **教训**:
  - 对 TIR lowering，先确认真实 full pipeline 的 IR 形态，再设计 matcher；不要只对着手工简化 pass 链做模式匹配
  - 如果“手动 pass 链绿、full `lower()` 仍红”，优先怀疑 prepasses 改写后的 IR 结构，而不是先改 runtime gate

### split-after flash-attention fragment analysis 若只识别 `CallNode`，会漏掉真实 TIR 里的 row reduction / row broadcast

- **时间**: 2026-03-31
- **问题**: `AnalyzeBlackholeFragmentRegions` 在 GQA 上能识别出 `row_reduction` / `row_broadcast`，但 MHA split-after IR 仍只产出 `gemm + pointwise_chain`
- **根本原因**:
  - split-after MHA 的关键表达式不是都编码成 `CallNode`
  - `scores_sum[0] + acc_s[rv]`、`T.max(scores_max[0], scores_max_clear[0])`、`acc_o[i] * scores_scale[0]` 这类关系在 TVM TIR 里分别是 `AddNode` / `MaxNode` / `MulNode` / `DivNode`
  - 旧分析器只在 `CallNode` 上找 `tir.add/max/div/...`，并且把广播主要建模成索引 rank 差，结果 direct sum/max reduction 和 scalar fragment -> vector fragment broadcast 在 MHA IR 上整片漏掉
- **解决**:
  - `AnalyzeBlackholeFragmentRegions` 改为显式识别 `AddNode` / `MaxNode` / `MulNode` / `DivNode`
  - row reduction 增加对 scalar fragment target 的 direct self-reduction 模式识别
  - row broadcast 增加对 scalar fragment source -> vector fragment target 的显式识别，不再只依赖索引 rank 差或 floor-div 形态
  - 新增 MHA fragment-region 回归，要求与 GQA 一样暴露 `gemm + row_reduction + row_broadcast + pointwise_chain`
- **教训**:
  - 对 split-after TIR 做结构分析时，先确认真实 IR 节点种类，再决定匹配逻辑；不要把 Python/T.script 表面的 `+/*//max` 直觉等同于 `CallNode`
  - 广播/归约语义优先按 buffer role 和 shape 关系识别，比按索引字符串或 rank 差猜更稳

### flash-attention forward 若在 `LowerBlackholeOps` 里直接按 raw analysis attrs 硬炸，会让 analysis 消费与 build-time gate 混在一起

- **时间**: 2026-03-31
- **问题**: 当 `AnalyzeBlackholeFragmentRegions` 已经能稳定识别 MHA/GQA 的 `row_reduction/row_broadcast/pointwise_chain` 后，`LowerBlackholeOps` 之前会直接对 raw `blackhole.fragment_regions` 做 `ICHECK(false)`，导致主链还没把 analysis 真正接住，就在 transform 层终止
- **根本原因**:
  - analysis attrs 和 build-time legality gate 混在了同一层
  - 这样既看不出 `LowerBlackholeOps` 是否已经开始消费 analysis，也迫使 compile boundary 卡在 transform pass，而不是更合适的 runtime-module/spec 抽取边界
- **解决**:
  - `LowerBlackholeOps` 先把 `work_decomposition` / `fragment_regions` / `pipeline_stages` 归一化成一层很薄的 `blackhole.lowering_requirements` IR attrs
  - 当前 fragment subset 尚未真正 lower 的显式 gate 挪到 `rt_mod_blackhole` 的 `ExtractExecutableSpecFromDeviceFunc`
  - 保持 `ExecutableSpec` 不直接承载 raw analysis attrs，也不新增 `flash_attention_plan` / `attention_work_contract`
- **教训**:
  - analysis 被 lowering 消费，和“不支持的 subset 在哪一层 fail-fast”是两件事，不能混在一个 `ICHECK` 里
  - 当某类 analysis 还处在“已识别、未执行”的阶段，优先让 transform 层产出最小归一化 summary，再把最终显式 gate 放到更靠近 spec/runtime 的边界

### full `lower()` 路径下的 GQA staged/shared GEMM view 若要求 `load->indices.size() == ndim`，会把更前面的 generic stage legality 噪声化

- **时间**: 2026-03-31
- **问题**: 给 GQA flash-attention 前向喂 `num_stages=4` 时，理论上当前应先被 `fragment pipeline legality` 拒绝；但沿着 full `lower()` 目标链走时，实际会先在 `LowerBlackholeOps::ExtractGemmInfo` 的 `RegionOp` 规范化里报 `load->indices.size() != ndim`
- **根本原因**:
  - 优化后的 device IR 会把 staged shared GEMM operand 表达成类似 `T.region(K_shared_1[stage, 0, 0], 1, 64, 128)` 的视图
  - 旧的 `RegionOp` bridge 假设 “provided extents 个数 == load indices 个数”，因此无法表达“leading stage index + trailing tile extents”的 staged/shared region
  - 结果 full target path 会先撞内部 canonicalization 错误，把本来应该更前置、更通用的 `num_stages` legality 噪声化
- **解决**:
  - `RegionOp` 现在允许 `provided_ndim <= load_ndim`
  - 未匹配的前导 load indices 会被收成 singleton axes，提供的 extents 则用于重建 trailing region axes
  - `AnalyzeBlackholePipelineStages` 也同步扩到 optimized path：兼容 `tl_pipelined_num_stages`、`blackhole.cb.*` scope 和 suffixed buffer views
  - 修复后，full `lower()` 的 GQA `num_stages=4` 已会先稳定命中 `Blackhole fragment pipeline legality: unsupported stage count 4`
- **教训**:
  - 对 staged/shared tensor view，region bridge 不应把“索引数等于 extents 数”当作普遍真理；这只适合最简单的平面视图
  - 复杂 kernel 的 legality 需要尽量绑在更早、更语义化的 IR 信号上；否则深层 canonicalization 的窄假设会把真正的边界淹掉

### GQA 更宽 pipeline 形态的 row-broadcast 若只认 `floor_div`，会在 analysis 层少报 `scores_max/scores_scale/logsum` 广播关系

- **时间**: 2026-03-31
- **问题**: GQA `num_stages=4` 的 split-after analysis 已经能暴露 `pipeline_stages` 和 `row_reduction`，但 `fragment_regions` 里缺少 `row_broadcast`
- **根本原因**:
  - 这条形态的广播索引不是 `floor_div(i, k)`，而是 `T.shift_right(i, k)`
  - 旧的 `ExprUsesFloorDivLikeIndex` 只认 `FloorDiv/FloorMod`，把右移这种同类的“coarsened row index”漏掉了
- **解决**:
  - fragment analysis 现在把 `Call(tir.shift_right, ...)` 也视作 row-broadcast 的索引归并信号
  - 新增 `test_gqa_forward_wider_pipeline_still_exposes_row_broadcast_roles`
- **教训**:
  - 对 split-after TIR 的索引分析，不要把“语义上等价的除法/位移归并”拆成两套完全不同的规则
  - GQA 这类更宽 pipeline 形态很容易把 analysis 写法上的窄假设暴露出来，应该用它来收正通用规则，而不是给 GQA 再加特判

### Blackhole `lower()` 若在 `SplitBlackholeKernel` 前按旧 device attrs 过滤，会把真实入口 `PrimFunc` 静默排除出 Blackhole pass 主链

- **时间**: 2026-03-31
- **问题**: 直接用 `lower(..., target="blackhole")` 编译 flash-attention forward 时，明明 `AnalyzeBlackhole*` 和 `LowerBlackholeOps` 手工串起来已经能看到预期边界，但真实 target-level 编译仍然晚到 codegen 才报 `Find undefined Variable acc_o`
- **根本原因**:
  - `tilelang.engine.lower.lower()` 先用 `get_device_call()` 过滤 `device_mod`
  - Blackhole entry `PrimFunc` 在 `SplitBlackholeKernel` / `Analyze*` / `LowerBlackholeOps` 之前只有 `target=blackhole` 和 entry attrs，还没有 `blackhole.segment_plan` / `blackhole.runtime_args` 这类 device attrs
  - 结果 `device_mod` 被错误过滤成空模块，Blackhole 专属 pass 实际没跑在真实入口 `main` 上
- **解决**:
  - `is_device_call()` 对 `target=blackhole` 且 `tir.is_entry_func` 的 `PrimFunc` 直接返回 `True`
  - 保持 `lower()` 的 host/device 过滤位置不变，但确保 Blackhole entry kernel 能进入 `blackhole_codegen()` 的主链
- **教训**:
  - 对依赖 target-specific split/lowering pass 才形成 device attrs 的后端，不能用“已经形成的 device attrs”去决定它是否进入那条 pass 链
  - 如果手工 pass 链和真实 `lower()` 路径的行为不一致，优先检查入口模块是否在 pass 之前就被过滤掉了

### fragment region analysis 若把全局 `tir.add/mul/div/max/...` 都记成 `pointwise_chain`，会误伤普通 copy/GEMM kernel

- **时间**: 2026-03-31
- **问题**: 在收正 Blackhole entry `PrimFunc` 过滤后，普通 copy/gemm 的 `lower()` 也开始被 flash-attention 用的 fragment-subset fail-fast 拦下
- **根本原因**:
  - `AnalyzeBlackholeFragmentRegions` 早期在 `VisitExpr_(CallNode)` 里全局扫描 `tir.exp2/max/multiply/add/div/if_then_else`
  - 普通 kernel 的索引算术、边界表达式、甚至非 fragment 区域里的 call 都会被误记成 `pointwise_chain`
  - 后续若按 `pointwise_chain` 直接做 fail-fast，就会把不属于 fragment compute 的 kernel 一起拦下
- **解决**:
  - 保留对 `gemm` 的全局识别
  - `pointwise_chain` 的识别收回到 fragment/local region 自身的 `BufferStore` 数据流分析中，只在真实 fragment compute 关系里标记
- **教训**:
  - analysis pass 一旦要被后续 lowering / legality 真正消费，就不能继续靠“全局扫到某类 op 名”这种宽泛近似
  - 对 fragment region 这类结构分析，判定边界必须和 region 自身的数据流绑定，而不是和整个函数的普通算术绑定

### fragment fill 的线性化 matcher 如果用 `same_as` 比较 extent，会漏掉优化后 IR 里的合法 contiguous fill

- **时间**: 2026-03-31
- **问题**: flash-attn forward 的 optimized device IR 里，`acc_o[i * 4 + vec] = 0` 这种明显 contiguous 的二维线性化 fill 没有被 `LowerBlackholeOps` 匹配到，导致 `fill` 一直残留在 `blackhole.lowering_requirements` 和 full `lower()` 的 unsupported 集合里
- **根本原因**:
  - 旧 matcher 试图按子项拆解 `i * inner_extent + inner_var`
  - 其中 `mul->b.same_as(inner_loop->extent)` 依赖对象同一性，而优化后 IR 里的 `IntImm(4)` 即使语义相等，也不保证和 loop extent 复用同一个节点
  - 同时如果先递归改写内层 loop，再让外层 matcher 去看，就会丢掉原始二维 fill 结构
- **解决**:
  - direct fragment fill 的匹配前移到递归前，避免内层先被改写
  - 不再对子项做对象级比较，改为用 `Analyzer::Simplify(expr - (outer * inner_extent + inner))` 判零，直接匹配整个 affine 关系
  - 新增 `tl.blackhole.fill_fragment` builtin，并覆盖 scalar fill 与线性化二维 fill
- **教训**:
  - 对优化后 TIR 的 pattern matching，优先比较“整个关系是否代数等价”，不要比较中间常量子节点是不是同一个对象

### fragment pointwise residual 如果在整棵表达式树里扫 `AddNode`，会把 cast/load 的索引算术误判成未 lower 的 `add`

- **时间**: 2026-03-31
- **问题**: flash-attn forward 在 `fill` 已 lower 后，full `lower()` 仍把 `add` 留在 unsupported 集合里，看起来像还有真正的 fragment `add` 没 lower
- **根本原因**:
  - pointwise residual 剪枝最初通过遍历 residual local store 的整棵 value 表达式树来找 `AddNode`
  - 但像 `T.Cast("float16", acc_s[i * 4 + vec])`、`T.Cast("float16", acc_o[i * 8 + vec * 4 + vec_1])` 这类合法 residual `cast`，其 load 索引表达式天然就带 `AddNode`
  - 结果 helper 会把索引算术错当成真正未 lower 的 pointwise `add`
- **解决**:
  - pointwise residual 剪枝改为看 residual store 的**根表达式类型**
  - 只有当 `store->value` 根节点本身仍是 `AddNode` 时，才继续把 `add` 保留在 unsupported 集合里
  - `add` 因此从 flash-attn 当前的 explicit blocker 中移除
- **教训**:
  - 对复杂 TIR 的 residual scan，必须区分“索引算术”和“值语义”；扫整棵表达式树通常会把这两者混在一起

### `ExtractCorePlan` / direct runtime 若为空 work plan 自动补默认 packet/core，会把 planner/runtime contract break 伪装成正常执行

- **时间**: 2026-03-30
- **问题**: Blackhole direct path 早期在两层都保留了 fallback：
  - `rt_mod_blackhole::ExtractCorePlan` 若 `work_packets` 为空，会自动补一个默认 `WorkPacket`
  - `BlackholeModule::ExecuteDirect` 若 `work_items` 为空，会再补一个 fallback core
- **影响**:
  - planner 没有正确产出 work plan 时，build/runtime 仍会继续往下走
  - 结果不是立即暴露 contract break，而是变成“看起来还能执行”的假象，后续只会以数值错误、core 映射错位或更脏的 runtime 问题形式出现
- **解决**:
  - `ExtractCorePlan` 不再为空 `work_packets` 注入默认 packet
  - `ExtractExecutableSpecFromDeviceFunc` 新增 core-plan 校验：空 `work_packets` 或零 `work_count` 直接 fail-fast
  - `ExecuteDirect` 删除 fallback core，要求 `work_items` 必须从 `core_plan.work_packets` 正式导出
- **教训**:
  - 对 host/runtime 执行计划，planner 产物缺失时应直接报 schema/spec 错误，不能靠 runtime “补一个最小可运行默认值”
  - 如果 direct runtime 里还存在默认 core / 默认 work packet 之类的补洞，通常说明真正的 contract 边界还没收正

### worker semaphore 跨核握手如果直接把 logical core 坐标塞进 `get_noc_addr`，TT-Sim 会挂死

- **时间**: 2026-03-30
- **问题**: 在最小 multi-core copy producer/consumer 验证里，consumer 先 `semaphore_wait`，producer 再 remote signal。最初实现直接把 `core_plan` 里的 worker 坐标当成 `get_noc_addr(x, y, ...)` 的目标坐标，TT-Sim 会在 `EnqueueMeshWorkload` 后卡死
- **根本原因**:
  - `core_plan` 当前携带的是 direct runtime 使用的 logical worker core descriptor
  - TT-Metal kernel 内 `get_noc_addr(...)` 需要的是设备映射后的 worker NOC 坐标
  - remote core descriptor 必须由 host runtime 结合设备映射显式 materialize，不能让 device code 从 logical core 坐标里猜
- **解决**:
  - runtime arg schema 新增 `logical_core_noc_x` / `logical_core_noc_y`
  - `BlackholeModule::BuildRuntimeArgsFromSpec` 用 `device.worker_core_from_logical_core(...)` 把 logical core descriptor 转成真正的 NOC 坐标
  - dataflow TIR 新增 `tl.blackhole.runtime_arg_u32(name)`，让 kernel 能显式读取按名字下发的 runtime arg
  - 最小 worker producer/consumer E2E 改为 `consumer: semaphore_wait` / `producer: noc_semaphore_inc(remote_sem, 1)`，TT-Sim 闭环通过
- **教训**:
  - 对跨核同步，remote core descriptor 必须作为正式 schema 进入 host materialization；不要让 device code 从测试常量、logical core 坐标或本地 semaphore 地址去猜远端路由信息

### Blackhole builtin 缺少 `TCallEffectKind` 会在控制流改写后被 TVM 当成非法纯表达式

- **时间**: 2026-03-30
- **问题**: 在 semaphore E2E 测试里，把现有 copy builtins 放进 `IfThenElse` / `LetStmt` 并重建 `PrimFunc` 后，TVM 会报 `Attribute TCallEffectKind has not been registered` 或 purity/struct-info 相关错误
- **根本原因**:
  - 多个 Blackhole builtin 之前只注册了 op 名和参数，没有把副作用属性注册到 TVM
  - 一旦这些 op 进入更严格的控制流/函数重写路径，TVM 会要求它们显式声明 `kOpaque` 或 `kPure`
- **解决**:
  - 为 CB、NOC、tile transport、matmul、tile-reg、semaphore 等 Blackhole builtin 补齐 `TCallEffectKind`
  - 其中 `get_semaphore` / `runtime_arg_u32` 标为 `kPure`，wait/set/read/write 等副作用操作标为 `kOpaque`
- **教训**:
  - 对 target-specific builtin，`TCallEffectKind` 不是可选装饰；只要它们会出现在 TIR 控制流和函数重写里，就必须一开始注册完整

### accessor-level `common_runtime_arg_count` 在 compile-time ABI 主路径下曾绕过 direct runtime fail-fast

- **时间**: 2026-03-30
- **问题**: P3 已经 formalize 了 `accessors` 和 `common_runtime_args`，也约定 direct runtime 只支持 `layout=interleaved` 且 `common_runtime_arg_count=0`。但 `BlackholeModule` 之前只在旧 accessor materialization 路径里检查 `accessor.common_runtime_arg_count`；当 kernel 走 `compile_time_arg_specs` 主路径时，只检查了 `kernel.common_runtime_args.empty()`，没有同步拒绝 accessor 自己声明的 `common_runtime_arg_count > 0`
- **影响**: richer accessor schema 可以在 schema/spec 层被正确提取，但 direct runtime 的 reject 边界不完整；未支持的 accessor common-runtime 组合可能绕过统一 schema 校验
- **解决**:
  - 把 `accessor.layout / accessor.memory_space / accessor.common_runtime_arg_count` 的 direct-runtime 约束统一收进 `ValidateKernelDirectRuntimeSchema`
  - 新增 copy / GEMM direct runtime reject 测试，专门覆盖 accessor-level `common_runtime_arg_count > 0`
  - 修正 copy semaphore runtime-arg 测试 helper：当测试通过 `segment.runtime_args` 注入额外 runtime arg 时，必须在原有顶层 `blackhole.runtime_args` 基础上追加；否则 segment 非空会遮蔽顶层 buffer runtime ABI
- **教训**:
  - `KernelSpec.accessors` 不是 compile-time ABI schema 的冗余影子；只要 direct runtime 仍消费 accessor descriptor，所有 materialization 路径都必须共享同一条 schema 校验
  - `segment.runtime_args` 在 `KernelSpec` 上是 override，不是 merge；测试或变异如果只塞新增字段而不带回原始 buffer args，会制造伪 bug

### `tilelang.compile(..., execution_backend="tvm_ffi")` 的 Blackhole wrapper/export path 会生成非法 host shim

- **时间**: 2026-03-26
- **问题**: 走 `tilelang.compile(..., target="blackhole", execution_backend="tvm_ffi")` 时，生成的 host shim `lib0.c` 会出现 `int32_t kernel_error_code = ;` 这样的非法代码，导致编译失败
- **影响**: 这是 Blackhole 的通用 wrapper/export blocker；即使最小 single-core copy probe 也能复现，不是 multicore GEMM 特有问题
- **根本原因**:
  - host TIR 在 `SplitHostDevice` / `LowerDeviceKernelLaunch` 主链下会形成 `kernel_error_code = T.call_packed("main_kernel", ...)`
  - 这本身是合法的：host 侧需要消费 packed call 的 `int32` 返回值
  - 真正的断点在 host C codegen：`CodeGenCHost` 对 `tvm_call_packed_lowered` 只实现了“发出调用语句”，没有在表达式上下文里把 `TVMFFIAny result` 再打印成可用返回值
  - 因此 `LetStmt` 最终被打印成 `int32_t kernel_error_code = ;`
- **解决**:
  - 在 TileLang 自己的 `src/target/codegen_c_host.cc` 中，为 `tvm_call_packed_lowered` / `tvm_call_cpacked_lowered` 补齐表达式返回值打印
  - packed call 发出后，若 `op->dtype` 非 `void`，显式从 `TVMFFIAny result` 取回 `.v_int64` / `.v_float64` / `.v_ptr` 并按目标 dtype cast
  - 新增 Blackhole 最小 `tvm_ffi` export 测试，验证 export 成功且 `lib0.c` 不再包含坏的 `kernel_error_code = ;`
- **教训**:
  - `call_packed` 既可能作为语句使用，也可能作为表达式使用；host codegen 不能只覆盖“调用成功/失败”这一半
  - 当 host TIR 已经能准确表示行为时，优先修 codegen 对 IR 语义的承载能力，而不是回头压扁 IR 语义去迁就打印器
  - 对这类 compile/export blocker，最短闭环是“固定最小复现 + 保留中间 `lib0.c` + 对照 host TIR 和最终 C 文本”

### multicore GEMM direct path 会先挂死、再暴露 `transpose_B` 数值错误

- **时间**: 2026-03-26
- **问题**: `test_blackhole_gemm_multicore_direct_call` 在 formal `BlackholeModule` direct host path 下最初会挂在 `EnqueueMeshWorkload`，修掉挂死后又继续出现明显数值错误
- **根本原因**:
  - host runtime 之前把 GEMM `num_k_tiles` 从整张输入 buffer 字节数反推，single-core 碰巧等于 `K/32`，multi-core 下会放大成错误的 K-tile 次数
  - segmented GEMM writer 之前按整张 output tensor 形状消费 output CB，而 compute 每个 core 只 `pack/push` 自己的一个 output tile，导致 writer 第二次 `cb_wait_front` 卡死
  - `transpose_B=True` 时，reader 仍按未转置的 tile 线性序读取 B；single-core 因 `N_tiles=1` 未暴露，multi-core 才显式读错 tile
- **解决**:
  - `BlackholeModule` 的 GEMM `num_k_tiles` runtime arg 改为直接按 `spec.gemm_contract.K / 32` 下发
  - `LowerBlackholeOps` 的 segmented GEMM writer 改为按 per-core `gemm_m_ x gemm_n_` output tile 生成 `write_tile_from_cb`
  - `LowerBlackholeOps` 在 `transpose_B=True` 的 GEMM B-reader 路径上，按 host-transposed tiled layout 生成 tile index
- **教训**:
  - multi-core bring-up 不能只看 `core_plan` 和 launch；host transfer contract、reader tile index 和 writer tile consumption 只要有一层还保留 single-core 偶然成立的假设，就会在 multi-core 下立刻暴露
  - 对 `transpose_B` 这类 contract，single-core `N_tiles=1` 很容易把错误掩盖掉；多核/多列 tile case 必须专门验证
  - `compute_contract.N/Nt` 在 segmented multicore GEMM 里默认表达的是每个 work/core 的 local output shape，不是全局 output 宽度；runtime 如果需要全局 logical N tile 数，必须再结合 `core_plan.logical_grid_x` 推导，不能直接把 per-core `Nt` 当全局值使用

### `fused_dataflow` 单段 runtime_args / KernelSpec 错位会让 direct runtime 静默读错或拿不到参数

- **时间**: 2026-03-26
- **问题**: copy `fused_dataflow` 从 scratch fallback 切回 codegen 主路径后，编译侧最初报 `Missing runtime arg binding for buffer var: A`；修掉后 direct runtime 又出现 `kernel[0] count=0`，结果全零
- **根本原因**:
  - `blackhole.segment_plan` 的单段 `fused_dataflow` 没有自己携带 `runtime_args`
  - `MakeSegmentPrimFunc` / `PopulateKernelSpecsForDeviceFunc` 之前直接使用空的 `segment.runtime_args`
  - 结果是：
    - codegen 侧丢失原函数上的 `blackhole.runtime_args`
    - runtime 侧 `KernelSpec.runtime_args` 也变成空数组
- **解决**:
  - 单段 `fused_dataflow` segment 必须继承原函数上的 `blackhole.runtime_args`
  - `KernelSpec.runtime_args` 也必须回退到 `ExecutableSpec.runtime_args`
  - codegen 对 copy builtins 的 buffer 绑定在按名字恢复失败时，按 `input_buffer_addr*` / `output_buffer_addr*` 角色回退
- **教训**:
  - segment source 和 runtime launch schema 必须同时继承，不然会出现“源码有 arg load / launch 没下发”或反过来的错位
  - 对单段 `fused_dataflow`，不能默认 `segment.runtime_args` 一定存在
  - `scratch` fallback 一旦删除，就会立刻暴露 schema 继承问题，这正说明 fallback 之前在掩盖主路径错位

### work schema 以单值隐式默认驱动时，copy/GEMM 会在 split/runtime 间静默错位

- **时间**: 2026-03-27
- **问题**: Blackhole copy 和 GEMM 早期都曾依赖 `current_work_linear_id` / `tile_count` 这种单值默认来表达整套 work 范围，导致 split 后的 reader / compute / writer 以及 direct runtime 在同一语义上并没有对齐
- **影响**: schema 看起来“有值”，但不同层对 tile start、tile count 和 output range 的理解不一致，容易出现静默读错、参数绑定错位，或者只在 multi-core / segmented case 才暴露的问题
- **解决**: 把 runtime work schema 改成显式角色化字段，copy 使用 `work_linear_id + a_tile_* + output_tile_*`，GEMM reader / compute / writer 分别携带自己的 `work_linear_id/a_tile_*/b_tile_*/output_tile_*/k_tile_*` 语义，并对缺失 `work_linear_id` 或超出当前支持面的 richer 组合做 fail-fast
- **教训**:
  - work 描述不是“一个线性 id + 一个 tile 数”就能覆盖的抽象，split 之后必须按角色把 range 明确写进 schema
  - `work_linear_id` 和 per-buffer range 不是一回事；前者是逻辑工作身份，后者是 reader/writer 真正消费的范围，不能互相偷代
  - 只要 runtime 还在从单值默认推导整套工作范围，或 codegen 还在从 range 字段反推 work id，就很容易把协议错位伪装成普通的 launch bug

### GEMM direct-path 数值错误由 `transpose_B` 丢失和 host row-major upload 引起

- **时间**: 2026-03-26
- **问题**: `test_blackhole_gemm_basic` direct execution 能完成，但结果明显错误（最初观察到 `max_diff=37.24`，复现时可达 `59.53`）
- **错误假设（已排除）**:
  - 不是 `PrintReadTileToCB` / `PrintWriteTileFromCB` 丢了 CB 同步原语
  - 实际检查 lowered TIR 可见 reader/writer 周围已经有：
    - `cb_reserve_back/cb_push_back`
    - `cb_wait_front/cb_pop_front`
- **根本原因**:
  - `LowerBlackholeOps::ExtractGemmInfo` 之前没有把 `transpose_B` 语义正式带到 runtime/spec
  - `BlackholeModule` 直接把 host row-major tensor 原样 memcpy 到 DRAM buffer
  - 但 TT-Metal matmul reader/compute path 期待的是：
    - B 已按 `transpose_B` 语义变成 `K x N`
    - A/B 已做 `tilize_nfaces`
    - C readback 后做 `untilize_nfaces`
- **解决**:
  - 新增 `blackhole.gemm_contract`
  - `rt_mod_blackhole` 将该 contract 进入 `ExecutableSpec`
  - `BlackholeModule` 在 direct path 下：
    - A: row-major → tilize
    - B: row-major `N x K` → transpose → tilize
    - C: tiled output → untilize → row-major tensor
- **教训**:
  - copy 路径通过只说明“字节搬运”没问题，不能证明 GEMM contract 正确
  - 对 TT-Metal matmul，host layout conversion 和 transpose 语义是 correctness contract，不是后续优化
  - 当 generated kernel source 看起来“差不多”时，仍必须继续追到 host upload / readback 层，不能过早停在 codegen 直觉上

### CB identity 唯一真源问题

- **时间**: 2026-03-25
- **问题**: reader/compute/writer 三段没有在同一个 CB identity 上同步
- **根本原因**: `LowerBlackholeOps` 同时产出局部 CB id 和 placeholder id；`PlanBlackholeCB` 允许重名 requirement；codegen 按名字恢复 binding 时取到不同 CB
- **解决**: `LowerBlackholeOps` 统一写 `requirement_index` → `PlanBlackholeCB` 回写 IR → codegen 直接读最终 `cb_id`
- **教训**: planner 的 identity（requirement_index）和 lifetime（lifetime_begin/end）必须分开建模

### ODR/ABI 错位导致随机崩溃

- **时间**: 2026-03-19
- **问题**: 给 `CBRequirement` 新增字段后，`PlanBlackholeCB` 随机以字符串拷贝/vector 排序崩溃
- **根本原因**: 两个头文件在同一 namespace 重复定义 `CBRequirement`，只更新一份导致对象布局不一致
- **教训**: 共享 protocol struct 必须集中到单一定义

### TVM `RemapBufferData` 破坏下游去重

- **时间**: 2026-03-25
- **问题**: canonicalization 后同一 buffer 经两次 `GetNewBuffer` 变成两个不同对象，`buffer_to_cb_` 查不到已分配 id
- **解决**: 在 `GetNewBuffer` 内缓存结果，相同原始 BufferNode 返回同一 Buffer 对象

### TVM `CopyOnWrite()` 对临时 ObjectRef 产生 dangling pointer

- **时间**: 2026-03-25
- **问题**: 对 `Downcast<BufferLoad>(base).CopyOnWrite()` 中的临时 ObjectRef 调用 COW，析构后指针悬空
- **解决**: 不对临时 ObjectRef 调用 CopyOnWrite，改为直接构造返回值

### JIT 缓存串扰

- **时间**: 2026-03-23
- **问题**: 同一 pytest 进程内多个 direct-call case 复用固定 kernel 临时路径，TT-Metal JIT 复用旧编译结果
- **解决**: kernel 临时目录改成每次执行唯一

### runtime/common-runtime arg heuristic 去重

- **时间**: 2026-03-30
- **问题**: `rt_mod_blackhole` 之前按 `kind + name/buffer` 聚合 `runtime_args` / `common_runtime_args`，相同 kind 但语义不同的跨 segment 参数会被错误合并，而且 metadata 不显式暴露稳定 identity
- **根本原因**: arg identity 没有从 lowering/split 端正式产出，spec 提取层只能做 host-side heuristic dedupe
- **解决**:
  - `KernelArgSpec` 新增显式 `identity`
  - `LowerBlackholeOps` / `SplitBlackholeKernel` 产出 `runtime_args` 时显式写入 `identity`
  - `rt_mod_blackhole` 聚合仅按 `identity` 去重
  - 缺失 `identity` 的 runtime/common-runtime arg schema build-time 直接拒绝
  - `ExecutableSpec` 顶层现已显式暴露 `common_runtime_args`
- **教训**:
- schema identity 必须由 IR/lowering 真源提供，不能留给 host-side 提取层猜
- 如果一个字段决定 cross-kernel 聚合行为，就必须进入 metadata/spec 真链路，而不是只存在于隐式 dedupe 规则里

### synchronization schema 直到 direct execution 才暴露 malformed runtime args

- **时间**: 2026-03-31
- **问题**: `semaphore_id_u32` 缺失 `semaphore_binding`，以及 `logical_core_noc_x/y` 只给单边或坐标不一致时，runtime module build 仍然成功，问题只会在 direct execution 时由分散的 kind-switch 临时撞出来
- **根本原因**:
  - synchronization schema 没有在 `ExecutableSpec` / `KernelSpec` 边界统一校验
  - semaphore 解析和 remote-core NOC 解析分别散在 shared/per-work runtime arg materializer 里
- **解决**:
  - 在 `BlackholeModuleNode` 构造期新增 synchronization schema 校验
  - `semaphore_id_u32` 现要求有唯一匹配 `semaphore_binding`，且 binding 必须引用 `ExecutableSpec.semaphores` 中已规划 semaphore
  - `logical_core_noc_x/y` 现要求带显式 `identity`、成对出现，并共享同一 logical core 坐标
  - runtime materialization 已统一到同步 helper/context，而不是继续由两套 kind-switch 分头解释
- **教训**:
  - 只要某类 runtime arg 已经形成正式对象层（这里是 semaphore binding 和 remote-core descriptor），就该在 spec 边界集中校验，不能把 malformed schema 留给执行期偶然触发
  - helper 拆分如果不顺手把协议边界收紧，通常只是在挪代码，不是在减债

### remote core schema 只留在 runtime_args 里，没进入 KernelSpec 真链路

- **时间**: 2026-03-31
- **问题**: `logical_core_noc_x/y` 虽然已有 identity 和成对校验，但最初仍只是两条 runtime arg；`BlackholeModule` 运行时还得从每条 arg 上拿 `core_x/core_y`，`KernelSpec` metadata 看不到正式 remote-core 对象
- **根本原因**:
  - remote worker descriptor 没有从 runtime arg 层上提成独立 schema
  - segment-local `runtime_args` 和顶层 `blackhole.runtime_args` 两条入口都可能携带 remote-core 信息，如果不统一提取，很容易一条链路有 descriptor、一条链路没有
- **解决**:
  - 新增 `KernelSpec.remote_core_descriptors`
  - `rt_mod_blackhole` 从 kernel 实际消费的 `runtime_args` 统一提取 descriptor，并兼容 segment-local 与顶层 runtime-arg 两条入口
  - `BlackholeModule` 解析 `logical_core_noc_x/y` 时改为优先消费 descriptor，而不是继续把 arg 上的 `core_x/core_y` 当真源
- **教训**:
  - 一旦 runtime arg 里的多个字段共同表达“一个对象”，就应该尽早上提成 spec/schema 对象；否则 host/runtime 层会长期被迫重复做 grouping 和一致性检查

### 环境问题速查

| 问题 | 解决 |
|------|------|
| `pip install -e .` 失败 | 用 `.pth` 文件指向本地构建产物 |
| `inspect.getsourcelines()` 在内联 `python -c` 中失败 | 写入 `.py` 文件再执行 |
| TT-Metal 示例在 TT-Sim 下报 `Root Directory is not set` | 设置 `TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME` |
| Python 加载旧 `build/` 库 | 统一使用 `tilelang_repo/build/` 单一构建目录 |
| TT-Sim 初始化报 `No chips detected` | 设置 `TT_METAL_SIMULATOR` 和 `TT_METAL_MOCK_CLUSTER_DESC_PATH` |
