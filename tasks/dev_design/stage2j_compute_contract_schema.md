# Stage 2J 设计：Compute Contract 正式化

## 基本信息

- **文档ID**: `stage2j_compute_contract_schema`
- **日期**: 2026-03-27
- **状态**: ✅ 已实现；2026-03-30 继续收尾 `compute_contract -> compute_config` 真源关系
- **对应任务**: TT-Metal contract formalization 的 P0 收尾
- **关联文档**:
  - `tasks/dev_design/final_blackhole_backend_redesign.md`
  - `tasks/dev_design/stage2d_ttmetal_contract_audit.md`
  - `tasks/dev_design/stage2i_compile_time_abi_schema.md`

---

## 1. 目标

把当前 GEMM compute 相关协议从“`blackhole.gemm_contract` + `compile_time_arg_specs` 分散承载”的状态，收正为统一的 `compute_contract`。

这轮目标不是只加一个新 attrs 名字，而是让以下三层围绕同一份协议对齐：

1. split-after attrs / segment plan
2. `ExecutableSpec` / `KernelSpec`
3. `BlackholeModule` direct runtime materialization

同时，这轮要用至少一个比当前最小 GEMM case 更宽的 compute case 做 E2E 闭环，证明新协议不是“只存在于 spec 里”。

---

## 2. 当前问题

当前 GEMM 主路径存在两个根本问题：

1. **compute 语义没有正式协议本体**
   - `M/N/K`、`transpose_A/B`、buffer 绑定、dtype 分层放在 `blackhole.gemm_contract`
   - `Mt/Kt/Nt` 与 transpose compile-time materialization 放在 `compile_time_arg_specs`
   - direct runtime 做 host tensor shape 校验、transpose/tilize、output readback 时直接读 `gemm_contract`

2. **`compile_time_arg_specs` 还是 materialization view，不是协议本体**
   - 它能表达最终下发到 `CreateKernel` 的 compile-time args
   - 但不能单独承担 host logical tensor contract、dtype 分层、buffer role、以及后续更宽 compute ABI 的正式来源

结果是：compute ABI 仍然是“分散协议 + runtime 拼装理解”，还没有收成统一 contract。

---

## 3. 设计原则

1. **先立正式 compute contract，再由它派生 materialization**
   - `compile_time_arg_specs` 继续保留
   - 但它降级为由 `compute_contract` 派生的 materialization view

2. **dtype / shape / flags 同层 formalize**
   - 不只做 `Mt/Kt/Nt`
   - 也不只做 tensor dtype
   - 统一由 `compute_contract` 承载 compute kernel 真正依赖的形状、转置与 dtype 分层

3. **runtime 不再直接依赖分散历史字段猜 GEMM 语义**
   - `BlackholeModule` 优先消费 `compute_contract`
   - 旧 `gemm_contract` 仅作为过渡兼容输入，不再继续扩语义

4. **不把 P0 和 P3/P5 混做**
   - 本轮不扩 sharded accessor、common runtime args、semaphore、multicast
   - 只收正 GEMM compute contract

---

## 4. 协议方案

### 4.1 新增 `blackhole.compute_contract`

在 device PrimFunc attrs 上新增：

- `blackhole.compute_contract`

当前 v1 已支持：

- `kind = "gemm"`

字段至少包含：

- `enabled`
- `kind`
- `a_buffer`
- `b_buffer`
- `c_buffer`
- `M`
- `N`
- `K`
- `Mt`
- `Nt`
- `Kt`
- `transpose_A`
- `transpose_B`
- `a_tensor_dtype`
- `b_tensor_dtype`
- `c_tensor_dtype`
- `a_cb_dtype`
- `b_cb_dtype`
- `c_cb_dtype`
- `accumulator_dtype`

语义约束：

- `M/N/K` 是 logical problem shape
- `Mt/Nt/Kt` 是 compute kernel compile-time tile counts
- tensor dtype、CB transport dtype、accumulator dtype 明确分层

### 4.2 `compute_contract` v2 扩展

在 v1 基础上，继续新增两组正式字段。

#### A. `compute_shape`

- `tile_shape`
  - `Mt`
  - `Nt`
  - `Kt`
- `block_shape`
  - `m_tiles`
  - `n_tiles`
  - `k_tiles`
- `subblock_shape`
  - `m_tiles`
  - `n_tiles`

命名原则：

- 不直接绑定 TT-Metal 某一版 API 的局部命名
- 也不把 TileLang lowering 内部细节暴露成正式协议
- 但必须能无损映射到 TT-Metal compute kernel compile-time ABI

#### B. `compute_precision`

- `math_fidelity`
- `fp32_dest_acc_en`
- `math_approx_mode`
- `unpack_to_dest_mode`

这组字段作为 compute kernel host `ComputeConfig` 与 compile-time ABI 的正式来源。

### 4.2 `blackhole.gemm_contract` 的地位

本轮不立即删除 `blackhole.gemm_contract`，原因是：

- 避免一次改动过大导致回归难定位
- 允许 host/runtime 在过渡期兼容旧 spec

但它的地位改为：

- 旧字段兼容层
- 不再继续承载新增 compute 语义

新增/扩展的正式字段一律进 `compute_contract`。

### 4.3 `ExecutableSpec`

在 `ExecutableSpec` 中新增：

- `compute_contract`

当前先支持 GEMM。

direct runtime 对 GEMM 的以下行为改为优先读 `compute_contract`：

- host input/output tensor shape 校验
- input transpose / tilize 路径
- output untilize / readback 校验
- `num_k_tiles` / logical N tiles 推导

v2 继续扩展：

- compute kernel `ComputeConfig`
- compute-side precision / block-subblock compile-time materialization

---

## 5. 实现方案

### 5.1 `LowerBlackholeOps`

- 在现有 GEMM 信息提取完成后，同时编码：
  - `blackhole.gemm_contract`（兼容）
  - `blackhole.compute_contract`（正式）
- `Mt/Nt/Kt` 直接从 `M/N/K` 与 tile 常量推导并写入 `compute_contract`
- compute segment 的 `compute_config` 不再自带一套独立默认值；它必须由 `compute_contract`
  中的 compute precision / compute ABI 字段投影得到

### 5.2 `rt_mod_blackhole`

- 新增 `ComputeContractSpec`
- 提取 `blackhole.compute_contract`
- 写入 `ExecutableSpec`
- 若 `compute_contract` 缺失但 `gemm_contract` 存在，则允许兼容回填最小 GEMM compute contract
- `KernelSpec.compute_config` 继续保留为 per-kernel materialization view，但它只能消费正式 schema，
  不能再和 `compute_contract` 各自维护不同默认值

### 5.3 `BlackholeModule`

- 新增统一的 `GetComputeContract(spec)` 访问入口
- GEMM direct runtime 的校验与 host-side materialization 统一优先读 `compute_contract`
- 保留旧 `gemm_contract` fallback，但只用于兼容，不再作为新增逻辑真源
- 对 compute kernel `CreateKernel(ComputeConfig)`，优先消费 `KernelSpec.compute_config`；若缺失则从
  `ExecutableSpec.compute_contract` 派生，不允许默默回退到硬编码默认值

### 5.4 测试

先写失败测试，再补实现。

覆盖面：

1. attrs/schema 测试
   - `blackhole.compute_contract` 被 materialize
   - `Mt/Nt/Kt`、transpose、dtype 分层正确

2. spec 测试
   - `ExecutableSpec` 带有 `compute_contract`
   - `compile_time_arg_specs.gemm_shape` 与 `compute_contract.Mt/Kt/Nt` 对齐

3. E2E 测试
   - 新增至少一个更宽的 GEMM compute case
   - 当前优先选择 `transpose_A=True` 的 direct runtime case

4. v2 ABI 测试
   - `compute_contract.compute_precision` 进入 attrs/spec
   - `compute_contract.block_shape/subblock_shape` 进入 attrs/spec

### 5.5 2026-03-30 收尾约束

本轮 P0 收尾只解决一个问题：把 GEMM compute ABI 的正式真源收敛到 `compute_contract`。

明确约束：

- `compute_contract` 是 compute 语义真源
- `compute_config` 是 compute kernel host materialization 视图
- `compute_config` 可以保留在 segment/kernel schema 中，但其字段值必须从 `compute_contract` 派生
- 不允许 `LowerBlackholeOps`、`rt_mod_blackhole`、`BlackholeModule` 各自再维护一套
  `HiFi4/true/false/...` 的局部默认值

本轮不做：

- 更宽 accessor execution surface
- sharded / non-interleaved runtime 支持
- multicast / global semaphore
   - compute kernel `compile_time_arg_specs` 与 v2 contract 对齐
   - `BlackholeModule` 不再把 `ComputeConfig.math_fidelity/fp32_dest_acc_en/math_approx_mode` 写死

## 8. 实施结果（2026-03-27）

- `compute_contract` 已继续扩展到：
  - `block_m_tiles/block_n_tiles/block_k_tiles`
  - `subblock_m_tiles/subblock_n_tiles`
  - `math_fidelity/fp32_dest_acc_en/math_approx_mode/unpack_to_dest_mode`
- `LowerBlackholeOps` 已为 compute segment 正式产出：
  - `compute_config`
  - `gemm_block_shape`
  - `gemm_subblock_shape`
- `rt_mod_blackhole` 已提取并写入 `KernelSpec.compute_config`
- `BlackholeModule` 已按 `KernelSpec.compute_config` materialize `ComputeConfig`
  - `math_fidelity`
  - `fp32_dest_acc_en`
  - `math_approx_mode`
  - `unpack_to_dest_mode`
- 对未知 `math_fidelity` / `unpack_to_dest_mode` 的 direct runtime fail-fast 已补齐
- `tl.gemm_py` 现有 IR 参数 `clear_accum/k_pack/wg_wait` 已继续 formalize 到：
  - `compute_contract.clear_accum/k_pack/wg_wait`
  - compute-side `compile_time_arg_specs`:
    - `gemm_clear_accum`
    - `gemm_k_pack`
    - `gemm_wg_wait`
- 这三项当前已完成“正式 ABI 不再丢失”的闭环；是否被 Blackhole compute kernel 语义消费仍以后续 codegen 演进为准
- `tl.gemm_py` 的 `policy` 也已 formalize 到：
  - `compute_contract.policy_type/policy_name`
  - compute-side `compile_time_arg_specs.gemm_policy`
- 当前 `policy` 已完成“正式 ABI 不再丢失”的闭环；是否进一步驱动 Blackhole compute codegen 仍可后续继续收正
- `tl.gemm_py` 的可选 `mbar` 绑定也已 formalize 到：
  - `compute_contract.has_mbarrier`
  - `compute_contract.mbarrier_buffer/mbarrier_scope/mbarrier_index_exprs`
- `mbar` 当前按第一性原理建模为 barrier binding，而不是 compile-time literal，因此本轮不新增 `gemm_mbarrier` compile-time ABI kind
- Blackhole direct runtime 当前会对 `has_mbarrier=True` 的 GEMM compute contract 显式 fail-fast；barrier 资源的正式执行面仍属于后续 synchronization / execution-surface 工作
- 残留说明：和 TT-Metal 正式 `ComputeConfig` 相比，`dst_full_sync_en`、`bfp8_pack_precise`、以及 `named_compile_args/defines` 仍未进入 Blackhole 主链 formalization；当前状态是“主要字段与主机物化路径已对齐”，不是 `ComputeConfig` 的全量镜像

---

## 6. 验证方式

1. `test_blackhole_gemm.py`
   - 新增 `compute_contract` attrs/spec 断言
   - 新增更宽 GEMM compute case 的 direct runtime E2E

2. 回归
   - 现有 copy/GEMM compile-time ABI 测试不回退

---

## 7. 完成标准

- GEMM compute 相关 shape/flags/dtype 分层有统一 `compute_contract`
- `ExecutableSpec` 与 direct runtime 都能消费这层协议
- `compile_time_arg_specs` 与 `compute_contract` 一致，不再是唯一真源
- 至少一个更宽的 GEMM compute case 通过 direct runtime E2E
