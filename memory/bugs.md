# 问题与 Bug 记录

> 本文档只保留仍然有复用价值的问题记录。已解决且无复用价值的条目已归档删除。

## 未解决

### flash-attention forward 当前会在 `LowerBlackholeOps` staged-copy path 上过早撞到 tile-aligned copy legality

- **时间**: 2026-03-31
- **问题**: 直接编译 `examples/flash_attention/example_mha_fwd_bshd.py` 到 Blackhole 时，当前最先报错的是 `Blackhole staged copy currently expects global width aligned to 32`
- **影响**: 这说明 flash-attention forward 还没进入后续的 fragment/pipeline lowering 消费阶段，主链首先被 staged-copy legality 卡住；如果不先定位是哪类 copy 被送进了当前 tile-aligned boundary，后面继续扩 `LowerBlackholeOps` 只会在错误层面上叠逻辑
- **当前证据**:
  - split-after 通用 analysis 已完成：`AnalyzeBlackholeWorkDecomposition`、`AnalyzeBlackholeFragmentRegions`、`AnalyzeBlackholePipelineStages`
  - `test_blackhole_flash_attention_analysis.py` 已是 `3 passed`
  - target 级直接编译 `example_mha_fwd_bshd` 仍在 `LowerBlackholeOps` 报 `global width aligned to 32`
- **解决方向**:
  - 先定位具体是哪类 copy / 哪个 buffer shape 被送进当前 staged-copy path
  - 再决定是应走更通用的 legality fail-fast、还是要把这类非当前 tile/stick copy 从旧 copy lowering 边界中分流出来
  - 不要跳过这个 root cause，直接往 `ExecutableSpec` 或 runtime schema 里加 attention-specific 字段

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
