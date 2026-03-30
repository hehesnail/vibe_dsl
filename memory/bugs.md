# 问题与 Bug 记录

> 本文档只保留仍然有复用价值的问题记录。已解决且无复用价值的条目已归档删除。

## 未解决

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

## 已解决（仍有复用价值）

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

### 环境问题速查

| 问题 | 解决 |
|------|------|
| `pip install -e .` 失败 | 用 `.pth` 文件指向本地构建产物 |
| `inspect.getsourcelines()` 在内联 `python -c` 中失败 | 写入 `.py` 文件再执行 |
| TT-Metal 示例在 TT-Sim 下报 `Root Directory is not set` | 设置 `TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME` |
| Python 加载旧 `build/` 库 | 统一使用 `tilelang_repo/build/` 单一构建目录 |
| TT-Sim 初始化报 `No chips detected` | 设置 `TT_METAL_SIMULATOR` 和 `TT_METAL_MOCK_CLUSTER_DESC_PATH` |
