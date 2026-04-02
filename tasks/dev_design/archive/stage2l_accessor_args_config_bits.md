# Stage 2L 设计：Accessor `args_config_bits` 协议收正

## 基本信息

- **文档ID**: `stage2l_accessor_args_config_bits`
- **日期**: 2026-03-30
- **状态**: ✅ 已实现（args_config_bits 与 TT-Metal ArgConfig 对齐）
- **对应任务**: TT-Metal contract formalization 的 P3 子项
- **关联文档**:
  - `tasks/dev_design/final_blackhole_backend_redesign.md`
  - `tasks/dev_design/stage2h_accessor_schema.md`
  - `tasks/dev_design/stage2i_compile_time_abi_schema.md`

---

## 1. 目标

把当前 accessor schema 里的 `args_config_bits`，从“名字已经对齐 TT-Metal，但编码值与 TT-Metal `tensor_accessor::ArgConfig` 不一致”的状态，收正成与 TT-Metal 同一套 bit contract。

本轮覆盖：

- `LowerBlackholeOps` 产出 accessor / compile-time ABI 时的 `args_config_bits`
- `rt_mod_blackhole` / `ExecutableSpec` / `KernelSpec` 提取层
- `BlackholeModule` host-side accessor compile-time arg materialization

---

## 2. 当前问题

TT-Metal `tensor_accessor::ArgConfig` 当前位定义是：

- `Sharded = 1 << 0`
- `IsDram = 1 << 1`

但 Blackhole 当前主路径把 interleaved accessor 默认编码成：

- `args_config_bits = 1`

这会把“interleaved DRAM accessor”错误编码成 “sharded”。之所以现在没炸，是因为 direct runtime 还没有正式按 schema 里的 `args_config_bits` 去 materialize accessor compile-time args，而是继续靠 `TensorAccessorArgs(mesh_buffer)` 自行推导。

也就是说，现在 schema 和 runtime 只是“碰巧跑通”，不是协议一致。

---

## 3. 设计原则

1. **协议字段必须和下游真实语义一致**
   - 既然字段名已经是 `args_config_bits`，就必须等价于 TT-Metal `ArgConfig.raw()`
   - 不能再保留一套 Blackhole 私有编码

2. **compile-time ABI 真源必须消费这份字段**
   - accessor schema 不只是结构测试字段
   - host materialization 必须按 `args_config_bits` 生成最终 compile-time args

3. **当前支持面仍然收窄**
   - direct runtime 继续只支持 interleaved + DRAM + no accessor CRTA
   - 本轮不顺带开放 sharded 或 runtime accessor args

---

## 4. 协议变化

### 4.1 `AccessorSpec.args_config_bits`

改为严格等价于 TT-Metal `tensor_accessor::ArgConfig.raw()`。

当前最小映射：

- interleaved + dram -> `IsDram` -> `2`
- interleaved + l1 -> `0`
- sharded + dram -> `Sharded | IsDram` -> `3`
- sharded + l1 -> `Sharded` -> `1`

### 4.2 `CompileTimeArgSpec`

`interleaved_accessor_cta` 也要显式携带 `args_config_bits`，避免 compile-time ABI 真源丢掉这份信息。

---

## 5. 实现方案

### 5.1 `LowerBlackholeOps`

- 不再用 `layout == "interleaved" ? 1 : 0` 这种占位编码
- 改为按 `layout` + `memory_space` 计算 TT-Metal 原生 bitmask

### 5.2 `rt_mod_blackhole`

- `CompileTimeArgSpec` 提取层新增 `args_config_bits`
- `AccessorSpec` / `CompileTimeArgSpec` 都保留同一份编码

### 5.3 `BlackholeModule`

- host materialization accessor CTA 时，按 schema 的 `args_config_bits` 构造 `TensorAccessorArgs`
- 如果 `args_config_bits` 请求了 runtime accessor bits，而 schema 仍没有 accessor CRTA 执行面，则 direct runtime 明确 fail-fast

---

## 6. 验证方式

1. 结构测试
   - copy / GEMM accessor schema 的 interleaved DRAM `args_config_bits == 2`
   - `compile_time_arg_specs` 中对应 accessor entry 也携带相同值

2. host/runtime 验证
   - direct runtime 继续通过当前 interleaved DRAM case
   - 如果把 accessor `args_config_bits` 改成需要 runtime CRTA 的值，direct runtime 明确拒绝

3. 回归
   - copy pipeline / GEMM schema 结构测试
   - 增量构建 `libtilelang.so`

---

## 7. 完成标准

- `args_config_bits` 与 TT-Metal `ArgConfig` 位定义一致
- `AccessorSpec` 和 `CompileTimeArgSpec` 不再丢失这份语义
- host accessor CTA materialization 消费正式字段，而不是继续隐式推导
- direct runtime 对未支持的 accessor runtime bits 明确 fail-fast
