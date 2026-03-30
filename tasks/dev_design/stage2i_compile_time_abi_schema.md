# Stage 2I 设计：Compile-Time ABI Schema 正式化

## 基本信息

- **文档ID**: `stage2i_compile_time_abi_schema`
- **日期**: 2026-03-27
- **状态**: ✅ 已实现（schema/spec/direct runtime）
- **对应任务**: TT-Metal contract formalization 的 P3 延续项
- **关联文档**:
  - `tasks/dev_design/final_blackhole_backend_redesign.md`
  - `tasks/dev_design/stage2d_ttmetal_contract_audit.md`
  - `tasks/dev_design/stage2g_unified_work_schema.md`
  - `tasks/dev_design/stage2h_accessor_schema.md`

---

## 1. 目标

把当前 Blackhole kernel ABI 中“匿名 `compile_time_args` 数组 + 少量分散 launch 字段”的状态，收正成显式 schema，并贯通：

`LowerBlackholeOps / split-after attrs -> rt_mod_blackhole -> ExecutableSpec/KernelSpec -> BlackholeModule::CreateKernel`

本轮目标是把当前已经真实存在、且已经被 direct runtime 消费的 compile-time ABI 正式对象化，而不是继续依赖位置约定或 host/codegen 的隐式理解。

本轮覆盖：

1. interleaved accessor compile-time ABI
2. GEMM 当前稳定存在的 compile-time ABI：`Mt/Kt/Nt`、`transpose_A/B`
3. kernel launch metadata 中已直接影响 TT-Metal host `CreateKernel` 的项：`core_type`、`processor`、`noc`

## 1.1 实施结果（2026-03-27）

- `LowerBlackholeOps` 已为 copy/GEMM 主路径产出 `compile_time_arg_specs` 与 `launch_spec`
- `rt_mod_blackhole` 已提取并写入 `KernelSpec` / `ExecutableSpec`
- `BlackholeModule` 已以 `compile_time_arg_specs + launch_spec` 为真源 materialize `CreateKernel`
- direct runtime 对未知 `compile_time_arg_spec.kind`、不支持的 accessor CTA、以及 `launch_spec.core_type` 不一致都已显式 fail-fast
- 当前正式支持的 compile-time ABI kinds：
  - `literal_u32`
  - `interleaved_accessor_cta`
  - `gemm_shape`
  - `gemm_transpose_flags`

---

## 2. 实施前问题

实施前，`KernelSpec` 仍保留：

- `compile_time_args: vector<uint32_t>`
- `core_type: string`

这只能表达“有一组 compile args 会被传进 CreateKernel”，但不能表达：

1. 这些 compile args 各段分别代表什么
2. 哪些段来自 lowering 时就已知的常量
3. 哪些段来自 host runtime buffer/accessor materialization
4. launch 侧真正决定 kernel 注册方式的 metadata 是什么

这会导致：

- accessor compile-time ABI 虽然已经 formalize 了 offset/count，但 kernel 其余 compile-time ABI 仍是匿名数组
- GEMM `Mt/Kt/Nt/transpose_*` 虽已进入 contract attrs，却还没进入统一 kernel compile-time ABI schema
- `CreateKernel` 仍需要同时读 `compile_time_args`、`core_type`、并在 host 侧额外猜 `processor/noc`

因此在本设计启动前，compile-time ABI 还不是正式协议对象。

---

## 3. 设计原则

1. **先收 schema，再扩功能**
   - 本轮只 formalize 当前已在用的 compile-time ABI
   - 不借此扩新的执行面

2. **compile-time ABI 与 runtime ABI 严格分层**
   - `runtime_args` / `common_runtime_args` 继续表达运行时下发信息
   - `compile_time_arg_specs` 只表达 kernel 注册时固定的 ABI 段

3. **launch metadata 也是 kernel ABI 的一部分**
   - `core_type`、`processor`、`noc` 不再散落在 host 逻辑里
   - 应进入 kernel-level launch schema

4. **允许最小过渡兼容，但不保留双真源**
   - `compile_time_args` 可暂时保留作为兼容载体
   - 正式真源切换到 `compile_time_arg_specs`

5. **不把 P3 和更远的 ABI 泛化混做**
   - 本轮不做 `defines`、semaphore、multicast、sharded accessor materialization
   - 不把 compute kernel 全量 ABI 一次性抽象完

---

## 4. 协议变化

### 4.1 `CompileTimeArgSpec`

在 `KernelSpec` 下新增：

- `compile_time_arg_specs`

每个 descriptor 至少包含：

- `name`
- `kind`
- `dtype`
- `offset`
- `count`
- `buffer`（仅 accessor 类需要）
- `segment_role`（可选）
- `args_config_bits`（仅 accessor 类需要）

按 kind 补充：

- `values`（对 lowering 时已知的常量段）
- `layout` / `memory_space`（对 accessor 类）

### 4.2 当前最小 kind 集合

本轮先只引入以下 kinds：

- `literal_u32`
- `interleaved_accessor_cta`
- `gemm_shape`
- `gemm_transpose_flags`

语义：

- `literal_u32`
  - 兜底表达当前已稳定但尚未专门命名的一段常量 compile args
- `interleaved_accessor_cta`
  - 表示由 host runtime `TensorAccessorArgs(mesh_buffer)` materialize 的 compile-time ABI 段
  - 当前固定 `count = 2`
  - `args_config_bits` 严格等价于 TT-Metal `tensor_accessor::ArgConfig.raw()`
- `gemm_shape`
  - 承载 `Mt/Kt/Nt`
- `gemm_transpose_flags`
  - 承载 `transpose_A/B`

### 4.3 `KernelLaunchSpec`

在 `KernelSpec` 下新增：

- `launch_spec`

字段至少包含：

- `core_type`
- `processor`
- `noc`

作用：

- `core_type` 保留 kernel 所属处理器域（如 `brisc` / `ncrisc` / `trisc`）
- `processor` / `noc` 显式化 data-movement kernel 的 host `CreateKernel` 选择

### 4.4 过渡兼容

本轮允许 `KernelSpec.compile_time_args` 暂时保留，但角色降为：

- 旧字段兼容载体
- schema 缺失时的过渡 fallback

正式真源改为：

- `compile_time_arg_specs`
- `launch_spec`

---

## 5. 实现方案

### 5.1 `LowerBlackholeOps`

- 在 split-after attrs 中新增 compile-time ABI schema 编码入口
- 当前先对已稳定存在的 compile-time ABI 发出显式 descriptors：
  - interleaved accessor CTA
  - GEMM `Mt/Kt/Nt`
  - GEMM `transpose_A/B`
- 仍允许用 `literal_u32` 包住当前尚未拆细的常量 compile args

### 5.2 `rt_mod_blackhole`

- 新增 `CompileTimeArgSpec` / `KernelLaunchSpec`
- 从 attrs 提取：
  - `compile_time_arg_specs`
  - `launch_spec`
- 写入 `KernelSpec`
- `compile_time_args` 仍可被填充，但不再是协议真源

### 5.3 `BlackholeModule`

- `CreateKernel` 前优先按 `compile_time_arg_specs` materialize 最终 compile args
- `interleaved_accessor_cta` 通过 runtime `MeshBuffer` 转 `TensorAccessorArgs`
- 当前 direct runtime 对 `interleaved_accessor_cta` 还要求 `layout=interleaved`、`memory_space=dram`、`args_config_bits=2`
- `gemm_shape` / `gemm_transpose_flags` 直接展开成 compile args
- 遇到未知 `kind` 明确 fail-fast
- `launch_spec` 成为决定 `ComputeConfig` / `DataMovementConfig` 的正式输入

### 5.4 `codegen_blackhole`

- 本轮尽量少动
- 继续只消费 builtin / lowering 已确定的 compile-time offset 与 kernel attrs
- 不强行让 codegen 直接依赖完整 `compile_time_arg_specs`

---

## 6. 不在本轮范围

- sharded accessor materialization
- `defines`
- semaphore / multicast
- non-tile / stick / sharded copy
- compute kernel compile-time ABI 的全量泛化

---

## 7. 验证方式

1. 结构测试
   - copy / GEMM 的 `KernelSpec` 能提取 `compile_time_arg_specs`
   - interleaved accessor CTA 有显式 `offset/count`
   - GEMM kernel 能显式携带 `gemm_shape` 与 `gemm_transpose_flags`

2. host materialization 测试
   - `BlackholeModule` 能按 `compile_time_arg_specs` 生成最终 compile args

3. fail-fast 测试
   - 遇到未知 `compile_time_arg_spec.kind` 时 direct runtime 明确拒绝

4. 回归
   - `test_blackhole_copy_pipeline.py`
   - `test_blackhole_gemm.py`

---

## 8. 完成标准

- `KernelSpec` / `ExecutableSpec` 不再只靠匿名 `compile_time_args` 表达 kernel compile-time ABI
- accessor CTA、GEMM `Mt/Kt/Nt`、`transpose_A/B` 至少进入正式 compile-time ABI schema
- `BlackholeModule` 开始以 `compile_time_arg_specs + launch_spec` 为真源 materialize `CreateKernel`
- direct runtime 对未知/未支持 compile-time ABI kind 明确 fail-fast
- 文档状态与代码一致
