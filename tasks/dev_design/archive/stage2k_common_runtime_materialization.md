# Stage 2K 设计：Kernel-Level Common Runtime Args Materialization

## 基本信息

- **文档ID**: `stage2k_common_runtime_materialization`
- **日期**: 2026-03-30
- **状态**: ✅ 已实现（kernel-level shared common runtime channel）
- **对应任务**: TT-Metal contract formalization 的 P3 子项
- **关联文档**:
  - `tasks/dev_design/final_blackhole_backend_redesign.md`
  - `tasks/dev_design/stage2d_ttmetal_contract_audit.md`
  - `tasks/dev_design/stage2h_accessor_schema.md`
  - `tasks/dev_design/stage2i_compile_time_abi_schema.md`

---

## 1. 目标

把当前已经进入 `KernelSpec` / `ExecutableSpec` 的 `common_runtime_args`，从“host 侧提取但 direct runtime 一律拒绝”的状态，推进到 **kernel-level shared common runtime channel 正式 materialize**：

`LowerBlackholeOps / segment_plan -> rt_mod_blackhole -> ExecutableSpec/KernelSpec -> BlackholeModule::SetCommonRuntimeArgs`

本轮目标只覆盖 **对所有 core/work item 共享、且不依赖 accessor CRTA 推导** 的 common runtime kinds。

---

## 2. 当前问题

当前主链里：

- `rt_mod_blackhole` 已能把 `segment.common_runtime_args` 提取进 `KernelSpec.common_runtime_args`
- `BlackholeModule` 仍要求 `kernel.common_runtime_args.empty()`
- direct runtime 只会调用 `SetRuntimeArgs`，不会调用 `SetCommonRuntimeArgs`

结果是：

1. 协议已经有 common runtime channel，但执行面还是“零支持”
2. 未来需要 shared runtime metadata 的 kernel 仍只能继续走 fail-fast
3. accessor/common-runtime schema 与 TT-Metal host API (`SetCommonRuntimeArgs`) 之间仍差最后一跳

---

## 3. 设计原则

1. **先开 shared common-runtime 主通道，不混入 richer accessor execution**
   - 本轮支持 kernel-level common runtime args materialization
   - accessor-derived CRTA、sharded accessor 继续 reject

2. **只支持 truly-common 的 arg kinds**
   - 只接受与 work id / per-core 逻辑无关的 shared metadata
   - `work_linear_id`、tile range、logical core coord 等仍不能进入 common channel

3. **不新增第二套 runtime arg 解释器**
   - common runtime arg 的 kind 继续复用现有 `KernelArgSpec.kind`
   - 由 `BlackholeModule` 在 host 侧按 “shared” 语义 materialize

4. **支持面必须显式、失败必须早**
   - 能 materialize 的 kinds 明确列出
   - 不支持的 common kind 直接 fail-fast，不隐式降级为 unique runtime arg

---

## 4. 本轮支持边界

本轮 direct runtime 仅支持以下 `kernel.common_runtime_args[*].kind`：

- `input_buffer_addr`
- `input_buffer_addr32`
- `output_buffer_addr`
- `output_buffer_addr32`
- `semaphore_id_u32`

明确不支持进入 common channel 的 kinds：

- `work_linear_id`
- `current_work_linear_id`
- `a_tile_*`
- `b_tile_*`
- `output_tile_*`
- `k_tile_start_id`
- `num_k_tiles`
- `logical_core_noc_x`
- `logical_core_noc_y`
- accessor-derived richer kinds（例如 `accessor_common_u32`）

解释：

- 上述支持 kinds 都能在 host 侧用一次性 shared 值 materialize
- accessor-derived richer kinds 仍依赖 accessor ABI/CRTA/codegen 一起收正，不在本轮做

---

## 5. 实现方案

### 5.1 `BlackholeModule`

- 新增 common-runtime schema 校验：
  - 不再要求 `kernel.common_runtime_args.empty()`
  - 改为校验其中每个 kind 是否属于当前 shared 支持集
- 新增 common-runtime materialization：
  - 根据 `KernelSpec.common_runtime_args` 生成 shared `std::vector<uint32_t>`
  - 在 `CreateKernelFromSpec` 之后、`SetRuntimeArgs` 之前调用 `SetCommonRuntimeArgs`

### 5.2 运行时语义

- `input/output_buffer_addr(32)`：由 runtime buffer binding materialize
- `semaphore_id_u32`：沿用现有 semaphore binding 解析逻辑

### 5.3 保持 reject 的部分

- accessor `common_runtime_arg_count > 0`
- `layout != interleaved`
- `memory_space != dram`
- 任意需要 per-work/per-core 动态值的 common-runtime kind
- 任意 accessor-derived common-runtime kind
- `scalar_u32`（当前 host ABI 还没有为 common-vs-unique scalar 建立独立索引语义）

---

## 6. 验证方式

1. 运行时正向测试
   - copy direct runtime 在注入 shared `scalar_u32` common runtime arg 后仍能执行成功
   - copy direct runtime 在注入 shared `input_buffer_addr32` common runtime arg 后仍能执行成功

2. fail-fast 测试
   - `work_linear_id` 放进 `common_runtime_args` 时 direct runtime 明确拒绝
   - accessor `common_runtime_arg_count > 0` 仍保持 reject

3. 回归
   - `test_blackhole_copy_pipeline.py`
   - 必要时补一条 GEMM 结构/运行时回归

---

## 7. 完成标准

- `KernelSpec.common_runtime_args` 不再被 direct runtime 全量拒绝
- `BlackholeModule` 会按正式 schema 调用 `SetCommonRuntimeArgs`
- shared common-runtime 支持边界清晰，且对 unsupported kinds 明确 fail-fast
- accessor CRTA / sharded execution 面仍保持显式 reject
- 文档、进度、代码状态一致
