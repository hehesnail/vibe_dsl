# Stage 2H 设计：Accessor Schema 与 Common Runtime Args 正式化

## 基本信息

- **文档ID**: `stage2h_accessor_schema`
- **日期**: 2026-03-27
- **状态**: ✅ 已实现（schema/spec 正式化；direct runtime 对 richer accessor execution 面 fail-fast）
- **对应任务**: TT-Metal contract formalization 的 P3 子项
- **关联文档**:
  - `tasks/dev_design/final_blackhole_backend_redesign.md`
  - `tasks/dev_design/stage2d_ttmetal_contract_audit.md`
  - `tasks/dev_design/stage2g_unified_work_schema.md`

---

## 1. 目标

把当前 Blackhole dataflow kernel 中“只留下一个 `accessor_slot` 占位，但 spec/runtime 并不知道它代表什么”的状态，收正成显式的 accessor schema，并贯通：

`LowerBlackholeOps / segment_plan -> rt_mod_blackhole -> ExecutableSpec/KernelSpec -> BlackholeModule::CreateKernel`

本轮目标分两层：

1. **interleaved accessor** 继续作为当前正式执行面，并保留 host/runtime materialization
2. **sharded accessor + accessor-derived common runtime args** 已正式进入 schema/spec，但 direct runtime 继续 fail-fast

---

## 2. 当前问题

当前主链虽然已经在 builtin 里保留了：

- `read_tile_to_cb(buffer, tile_index, cb_id, tile_bytes, accessor_slot)`
- `write_tile_from_cb(cb_id, buffer, tile_index, tile_bytes, accessor_slot)`

但协议仍然缺层：

1. `accessor_slot` 目前只够表达最窄的 interleaved compile-time offset
2. `segment_plan / ExecutableSpec / KernelSpec` 还没有 accessor-derived common runtime args
3. host/runtime 仍只理解“compile-time args 里追加两项 interleaved accessor args”
4. sharded accessor 所需的 compile-time/common-runtime schema 还没有正式对象层

这意味着 accessor 现在只是“语法上留了一个参数”，不是正式协议对象。

---

## 3. 设计原则

1. **先收协议，再扩执行面**
   - 本轮把 interleaved/sharded accessor schema 与 common runtime args 都收正进主协议
   - 但只保留 interleaved 为正式执行面

2. **accessor slot 必须是协议真源，不是 codegen 自己猜**
   - slot 由 lowering 明确写入 builtin
   - kernel compile-time args / common runtime args 都按 schema materialize

3. **host/runtime 不再靠 buffer 角色隐式猜 accessor 布局**
   - segment/kernel 要显式列出 accessor descriptors
   - `CreateKernel` 直接消费 descriptor + runtime buffer binding

4. **不把 P3 和 P4 混做**
   - 这轮只 formalize accessor contract
   - 不承诺新增 sharded/non-tile 执行能力

---

## 4. 协议变化

### 4.1 segment-level accessor descriptor

在 `blackhole.segment_plan` 的每个 dataflow segment 下新增：

- `accessors`
- `common_runtime_args`

每个 accessor descriptor 至少包含：

- `buffer`
- `layout`
- `memory_space`
- `compile_time_arg_offset`
- `compile_time_arg_count`
- `common_runtime_arg_offset`
- `common_runtime_arg_count`
- `args_config_bits`

当前约定：

- `layout` 当前至少允许：`interleaved` / `sharded`
- `memory_space` 当前至少允许：`dram` / `l1`
- compile-time / common-runtime 两段 offset-count 都是显式协议字段

### 4.2 当前 slot 约定

由于当前 interleaved accessor 的 `TensorAccessorArgs` 固定占 2 个 compile-time args，本轮 interleaved case 的 CTA offset 仍按 2 项一组推进：

- reader 第一个输入 accessor: `compile_time_arg_offset = 0`
- reader 第二个输入 accessor: `compile_time_arg_offset = 2`
- writer 第一个输出 accessor: `compile_time_arg_offset = 0`
- fused copy:
  - input accessor: `compile_time_arg_offset = 0`
  - output accessor: `compile_time_arg_offset = 2`

这里的 `2` 来自 TT-Metal `TensorAccessorArgs` 对 interleaved accessor 的固定 compile-time arg 宽度：

- `args_config`
- `aligned_page_size`

对应地，当前 interleaved case：

- `compile_time_arg_count = 2`
- `common_runtime_arg_offset = 0`
- `common_runtime_arg_count = 0`

### 4.3 common runtime args schema

`common_runtime_args` 是 kernel/segment 级别的第二条 ABI 通道，专门承载 accessor 自身派生出的 runtime metadata。

它与当前 `runtime_args` 的分工必须分开：

- `runtime_args`: work/business ABI，例如 `work_linear_id`、`a_tile_start_id`
- `common_runtime_args`: accessor ABI，例如 rank、tensor shape、bank coords、shard shape 等

本轮结果：

- interleaved case 显式携带空数组
- sharded case 可以进入 schema/spec，但当前 direct runtime 不负责 materialize 执行，而是显式 fail-fast

### 4.3 `KernelSpec`

`KernelSpec` 新增 accessor descriptors 和 `common_runtime_args` descriptors，作为 host-side kernel materialization 的正式输入。

`ExecutableSpec` 继续通过 `kernels[*]` 承载这层协议，不再让 `CreateKernel` 从 builtin 或 buffer 角色隐式猜 accessor。

---

## 5. 实现方案

### 5.1 `LowerBlackholeOps`

- 在生成 `read_tile_to_cb/write_tile_from_cb` 时继续写入当前执行面所需的 compile-time accessor offset
- 同时按 segment kind 记录正式 `AccessorSpec`
- interleaved case 明确写出 `compile_time_arg_* / common_runtime_arg_* / args_config_bits`
- sharded/accessor-runtime richer case 预留 schema 编码入口；缺信息时显式拒绝，不猜

### 5.2 `rt_mod_blackhole`

- 从 `blackhole.segment_plan[*].accessors` 提取正式 accessor descriptors
- 从 `blackhole.segment_plan[*].common_runtime_args` 提取 accessor-derived CRTA descriptors
- 写入 `KernelSpec.accessors` / `KernelSpec.common_runtime_args`

### 5.3 `BlackholeModule`

- `CreateKernel` 前，根据 `KernelSpec.accessors` 和 runtime `MeshBuffer` 绑定 materialize compile-time accessor args
- 当前只正式接受：
  - `layout = interleaved`
  - `common_runtime_arg_count = 0`
- 如果看到 `layout != interleaved` 或 `common_runtime_arg_count > 0`，direct runtime 直接 fail-fast

这保证 host/runtime 已经开始消费正式协议，但不会假装已经支持 sharded 执行。

### 5.4 `codegen_blackhole`

- `read_tile_to_cb/write_tile_from_cb` 不再直接打印 `InterleavedAddrGen`
- 改为：
  - `constexpr auto accessor_args = TensorAccessorArgs<CTA>()`
  - `const auto accessor = TensorAccessor(accessor_args, addr, tile_bytes)`
  - `noc_async_read_tile / noc_async_write_tile`
- 当前继续只生成 interleaved 可执行 kernel；但 schema 已经能表达未来 `TensorAccessorArgs<CTA, CRTA>` 所需信息

---

## 6. 不在本轮范围

- sharded 执行面
- row-major / stick / non-tile accessor schema
- semaphore / multicast / remote accessor
- compute kernel compile-time ABI 泛化

---

## 7. 验证方式

1. 结构测试
   - copy / GEMM 的 `accessors` 从旧 `slot` 语义升级为完整 `AccessorSpec`
   - interleaved case 的 `common_runtime_args` 显式存在且为空
   - 可以构造 sharded/common-runtime schema 进入 spec 提取层

2. 提取测试
   - `rt_mod_blackhole` 能把 `accessors` / `common_runtime_args` 提取到 `KernelSpec`

3. codegen 测试
   - kernel source 使用 `TensorAccessorArgs<slot>()` / `TensorAccessor(...)`
   - 不再打印手写 `InterleavedAddrGen`

4. fail-fast 测试
   - 对 `layout = sharded` 或 `common_runtime_arg_count > 0` 的 kernel，direct runtime 明确拒绝
   - reject 覆盖必须同时命中两条 materialization 路径：
     - `KernelSpec.accessors`
     - `compile_time_arg_specs` 主路径下仍保留的 accessor descriptors

5. 回归
   - copy pipeline 测试
   - GEMM lowering / contract 结构测试
   - copy / GEMM direct runtime 针对 accessor-level `common_runtime_arg_count > 0` 的拒绝测试

---

## 8. 完成标准

- accessor descriptor 与 common runtime args descriptor 都成为 split 后正式 schema 的一部分
- `KernelSpec` / `ExecutableSpec` 能显式表达 interleaved + sharded accessor contract
- `BlackholeModule` 开始消费正式 accessor schema，并对未支持执行面 fail-fast
- 当前 copy + GEMM dataflow kernel 的 interleaved accessor CTA offset 与 host compile-time args 对齐
- 文档状态与实现一致
