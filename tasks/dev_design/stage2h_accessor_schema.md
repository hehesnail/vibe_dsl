# Stage 2H 设计：Interleaved Accessor Schema 正式化

## 基本信息

- **文档ID**: `stage2h_accessor_schema`
- **日期**: 2026-03-27
- **状态**: 实施中
- **对应任务**: TT-Metal contract formalization 的 P3 子项
- **关联文档**:
  - `tasks/dev_design/final_blackhole_backend_redesign.md`
  - `tasks/dev_design/stage2d_ttmetal_contract_audit.md`
  - `tasks/dev_design/stage2g_unified_work_schema.md`

---

## 1. 目标

把当前 Blackhole dataflow kernel 中“只留下一个 `accessor_slot` 占位，但 spec/runtime 并不知道它代表什么”的状态，收正成显式的 interleaved accessor schema，并贯通：

`LowerBlackholeOps / segment_plan -> rt_mod_blackhole -> ExecutableSpec/KernelSpec -> BlackholeModule::CreateKernel`

本轮只做 **interleaved DRAM accessor**，不做 sharded / runtime common accessor args。

---

## 2. 当前问题

当前主链虽然已经在 builtin 里保留了：

- `read_tile_to_cb(buffer, tile_index, cb_id, tile_bytes, accessor_slot)`
- `write_tile_from_cb(cb_id, buffer, tile_index, tile_bytes, accessor_slot)`

但协议仍然缺层：

1. `accessor_slot` 目前在 lowering 里基本恒等于 `0`
2. `segment_plan / ExecutableSpec / KernelSpec` 没有 accessor descriptors
3. host `CreateKernel` 没有根据 buffer/accessor contract 生成 `TensorAccessorArgs(...)`
4. codegen 仍直接打印 `InterleavedAddrGen`，而不是消费显式 accessor slot

这意味着 accessor 现在只是“语法上留了一个参数”，不是正式协议对象。

---

## 3. 设计原则

1. **先把 interleaved accessor 做实，再做 sharded/general accessor**
   - 本轮只覆盖当前已经验证通过的 copy + GEMM 主路径
   - 不把 sharded/runtime common args 一起卷进来

2. **accessor slot 必须是协议真源，不是 codegen 自己猜**
   - slot 由 lowering 明确写入 builtin
   - kernel compile-time args 也按同一顺序 materialize

3. **host/runtime 不再靠 buffer 角色隐式猜 accessor 布局**
   - segment/kernel 要显式列出 accessor descriptors
   - `CreateKernel` 直接消费 descriptor + runtime buffer binding

---

## 4. 协议变化

### 4.1 segment-level accessor descriptor

在 `blackhole.segment_plan` 的每个 dataflow segment 下新增：

- `accessors`

每个 accessor descriptor 至少包含：

- `buffer`
- `slot`
- `layout`
- `memory_space`

当前约定：

- `layout = "interleaved"`
- `memory_space = "dram"`
- `slot` 是该 kernel 的 compile-time accessor CTA offset

### 4.2 当前 slot 约定

由于本轮 dataflow kernel 不再携带其它 compile-time args，interleaved accessor 的 CTA offset 固定按 2 个 compile-time args 一组推进：

- reader 第一个输入 accessor: `slot = 0`
- reader 第二个输入 accessor: `slot = 2`
- writer 第一个输出 accessor: `slot = 0`
- fused copy:
  - input accessor: `slot = 0`
  - output accessor: `slot = 2`

这里的 `2` 来自 TT-Metal `TensorAccessorArgs` 对 interleaved accessor 的固定 compile-time arg 宽度：

- `args_config`
- `aligned_page_size`

### 4.3 `KernelSpec`

`KernelSpec` 新增 accessor descriptors，作为 host-side kernel materialization 的正式输入。

`ExecutableSpec` 继续通过 `kernels[*]` 承载这层协议，不再让 `CreateKernel` 从 builtin 或 buffer 角色隐式猜 accessor。

---

## 5. 实现方案

### 5.1 `LowerBlackholeOps`

- 在生成 `read_tile_to_cb/write_tile_from_cb` 时写入真实 accessor slot
- 同时按 segment kind 记录 accessor descriptors：
  - copy `fused_dataflow`: input/output
  - GEMM `reader`: A/B
  - GEMM `writer`: C

### 5.2 `rt_mod_blackhole`

- 从 `blackhole.segment_plan[*].accessors` 提取 accessor descriptors
- 写入 `KernelSpec.accessors`

### 5.3 `BlackholeModule`

- `CreateKernel` 前，根据 `KernelSpec.accessors` 和 runtime `MeshBuffer` 绑定追加 `TensorAccessorArgs`
- 当前只接受：
  - `layout = interleaved`
  - `memory_space = dram`

其余组合直接 fail-fast。

### 5.4 `codegen_blackhole`

- `read_tile_to_cb/write_tile_from_cb` 不再直接打印 `InterleavedAddrGen`
- 改为：
  - `constexpr auto accessor_args = TensorAccessorArgs<slot>()`
  - `const auto accessor = TensorAccessor(accessor_args, addr, tile_bytes)`
  - `noc_async_read_tile / noc_async_write_tile`

---

## 6. 不在本轮范围

- sharded accessor descriptors
- accessor common runtime args
- row-major / stick / non-tile accessor schema
- semaphore / multicast / remote accessor
- compute kernel compile-time ABI 泛化

---

## 7. 验证方式

1. 结构测试
   - copy `segment_plan[0]["accessors"]` 存在 input/output descriptor
   - GEMM reader/writer segment 有正确 accessor descriptors

2. codegen 测试
   - kernel source 使用 `TensorAccessorArgs<slot>()` / `TensorAccessor(...)`
   - 不再打印手写 `InterleavedAddrGen`

3. 回归
   - copy pipeline 测试
   - GEMM lowering / contract 结构测试

---

## 8. 完成标准

- accessor descriptor 成为 split 后正式 schema 的一部分
- `KernelSpec` / `BlackholeModule` 显式消费 accessor schema
- 当前 copy + GEMM dataflow kernel 的 accessor slot 与 host compile-time args 对齐
- 文档状态与实现一致
