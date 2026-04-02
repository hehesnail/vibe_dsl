# Stage 2G 设计：Richer Runtime Work Schema（copy + GEMM）

## 基本信息

- **文档ID**: `stage2g_unified_work_schema`
- **日期**: 2026-03-27
- **状态**: ✅ 已实现（copy equal-range + GEMM richer work descriptor）
- **对应任务**: TT-Metal contract formalization 的 P3 子项
- **关联文档**:
  - `tasks/dev_design/final_blackhole_backend_redesign.md`
  - `tasks/dev_design/stage2d_ttmetal_contract_audit.md`
  - `tasks/dev_design/stage2f_gemm_dtype_layering.md`

---

## 1. 目标

把当前 Blackhole 的 runtime work ABI 从“过薄且依赖隐式推导”的形态，收正成 **能真实表达当前 copy/GEMM work range** 的 richer work descriptor schema，并且在本轮同时覆盖：

1. copy `fused_dataflow`
2. GEMM `reader / compute / writer`

本轮目标不是一次性把 accessor schema 一起补齐，而是先让 `LowerBlackholeOps / SplitBlackholeKernel -> rt_mod_blackhole -> BlackholeModule` 主路径拥有足够表达力，使 work descriptor 本身就能承载当前已经验证通过的 multicore copy/GEMM 语义，而不是继续依赖：

- `current_work_linear_id`
- `tile_count`
- 单组 `start/count`
- host runtime 对 `num_k_tiles`、output tile consumption 的额外隐式推导

---

## 2. 当前问题

`stage2d_ttmetal_contract_audit.md` 已指出 runtime/work schema 过薄。上一轮收正虽然把 schema 从单值默认推进到了显式字段，但最终 review 进一步确认：**只有 `start_id + num_tiles` 仍不足以表达当前 multicore GEMM 的真实 reader work**。

根因是：

- GEMM A-reader 的 tile start 由 `by` 驱动
- GEMM B-reader 的 tile start 由 `bx` 驱动
- `transpose_B=True` 时，B-reader 还需要 `kt * N_tiles` 这类 stride 语义

因此如果 schema 只有一组 `input_tile_start_id/input_num_tiles`，它并不能成为 reader 的正式真源；代码仍然只能依赖保留在 TIR 里的 `bx/by` 隐式兜底。这不满足 formalization 的要求。

copy 也有同类问题：

- schema 若声称 input/output range 可独立表达，但 codegen/runtime 仍默认两者相等
- 那么协议就会“名义上更通用，实际上静默误解释”

---

## 3. 设计原则

1. **先把 work descriptor 真正做实，再做 accessor descriptor**
   - 本轮先收正“做哪段工作”
   - 不把“如何访问 buffer”一起卷进来

2. **schema 必须能独立承载当前已验证行为**
   - 不能只给单值 ABI 起一个更显式的名字
   - 必须能在不偷读 TIR 隐式语义的前提下表达当前 multicore copy/GEMM 的 work 分布

3. **per-buffer range 必须按角色拆开**
   - copy source/output 分开
   - GEMM reader A/B 分开
   - writer output 单独表达

4. **当前不支持的 richer 组合必须 fail-fast**
   - 如果 schema 表达力大于当前 runtime/codegen 支持面
   - 必须显式拒绝，而不是静默退化成“按 work_linear_id 猜”

---

## 4. Richer Work Descriptor Schema

### 4.1 统一 kind

本轮统一引入以下 runtime arg kinds：

- `work_linear_id`
- `a_tile_start_id`
- `a_tile_num_tiles`
- `a_tile_stride`
- `b_tile_start_id`
- `b_tile_num_tiles`
- `b_tile_stride`
- `output_tile_start_id`
- `output_tile_num_tiles`
- `output_tile_stride`
- `k_tile_start_id`
- `num_k_tiles`

当前约定：

- 全部为 `uint32`
- 仍通过现有 `KernelArgSpec` 数组表达
- 本轮不新增更大的 `WorkDescriptorSpec` struct，但语义上已经视为 richer work descriptor，而不是单值 runtime arg 的简单替换

### 4.2 copy 的语义

copy `fused_dataflow` 使用：

- `input_buffer_addr32`
- `output_buffer_addr32`
- `work_linear_id`
- `a_tile_start_id`
- `a_tile_num_tiles`
- `a_tile_stride`
- `output_tile_start_id`
- `output_tile_num_tiles`
- `output_tile_stride`

当前正式支持范围下，copy 额外约束为：

- `a_tile_start_id == output_tile_start_id`
- `a_tile_num_tiles == output_tile_num_tiles`
- `a_tile_stride == output_tile_stride == 1`

也就是说，本轮不是让 copy 立即支持独立 input/output range，而是：

- schema 先能表达 richer range
- 当前 runtime/codegen 明确只接受“source/dest range 相等”的正式支持面
- 一旦出现不相等，必须 fail-fast，而不是静默误解释

### 4.3 GEMM 的语义

#### reader

reader 使用：

- A/B buffer addr
- `work_linear_id`
- `a_tile_start_id`
- `a_tile_num_tiles`
- `a_tile_stride`
- `b_tile_start_id`
- `b_tile_num_tiles`
- `b_tile_stride`
- `k_tile_start_id`
- `num_k_tiles`

其中：

- `a_*` 描述 A-reader 的 tile range
- `b_*` 描述 B-reader 的 tile range
- `*_stride` 允许表达 `start + i * stride` 型访问
- 对 `transpose_B=True` 的 multicore GEMM，B-reader 当前最关键的正式表达是：
  - `b_tile_start_id = bx`
  - `b_tile_num_tiles = num_k_tiles`
  - `b_tile_stride = N_tiles`
- `k_tile_start_id` 当前先固定为 0，为后续 K 维切分预埋

#### compute

compute 使用：

- `k_tile_start_id`
- `num_k_tiles`

compute 不直接关心 A/B/output tile range，因为本轮不改 compute 的 CB 消费语义。

#### writer

writer 使用：

- `work_linear_id`
- `output_tile_start_id`
- `output_tile_num_tiles`
- `output_tile_stride`

writer 不再通过 output tensor 整体形状或 `current_work_linear_id` 间接推导 output tile range。

---

## 5. 产出层级

### 5.1 `LowerBlackholeOps`

职责变化：

- copy path 不再只产出 `current_work_linear_id / tile_count`
- 统一产出 richer work descriptor kinds
- copy 先只产出当前正式支持的等范围 schema

### 5.2 `SplitBlackholeKernel`

职责变化：

- reader / compute / writer segment 的 `runtime_args` 改为 richer work descriptor
- GEMM reader 显式区分 A/B 两路输入的 work range

### 5.3 `rt_mod_blackhole`

职责不变，但要正确提取 richer runtime arg schema 进入：

- `ExecutableSpec.runtime_args`
- `KernelSpec.runtime_args`

### 5.4 `BlackholeModule`

职责变化：

- 统一按 `KernelArgSpec.kind` 下发 richer work descriptor
- 对当前支持面之外的 range/stride 组合直接报错
- 不再保留“从单值 work id 自动猜整套 range”的兜底

---

## 6. 兼容与迁移

本轮是 **schema 收正**，不是长期双栈兼容：

- copy 和 GEMM 的测试、codegen、runtime 一起切到 richer schema
- `current_work_linear_id` 和 `tile_count` 不再作为正式 ABI 的真源；`work_linear_id` 仅保留为逻辑 work identity 字段

原因：

- 单值 ABI 本来就是 bring-up 阶段最小协议，不值得继续保留双真源
- 如果继续让 runtime 从单值默认推导整套 range，就会把协议错位伪装成 launch bug

---

## 7. 风险与边界

### 7.1 本轮显式不做

- accessor descriptor 本体
- copy non-tile / stick / sharded 泛化
- K 维切分真正落地
- semaphore / multicast
- GEMM compute tile index 生成逻辑改写

### 7.2 风险控制

本轮只改：

- runtime arg schema
- host 下发逻辑
- 与 schema 直接耦合的 codegen arg 读取
- 对不支持的 range/stride 组合做 fail-fast 验证

不改：

- CB planner
- physical core 分配
- host tilize/untilize
- GEMM `transpose_B` 已收正逻辑

---

## 8. 验证方式

1. **copy 结构测试**
   - `blackhole.runtime_args` 改为 richer range schema
   - codegen/runtime 对 input/output range 不一致或 stride ≠ 1 的 case 明确 fail-fast

2. **GEMM 结构测试**
   - `blackhole.segment_plan` 中 reader / compute / writer 各自携带正确 richer work descriptor
   - reader 显式区分 A/B range
   - `transpose_B=True` 时，B-reader stride 能表达 `kt * N_tiles`

3. **runtime 回归**
   - `test_blackhole_copy_runtime.py`
   - `test_blackhole_gemm.py`

4. **非目标回归**
   - `test_blackhole_tvm_ffi_export.py`
   - `test_blackhole_copy_pipeline.py`

当前环境回归结果（2026-03-27）：

- `test_blackhole_copy_pipeline.py`: 19 passed, 1 xfailed
- `test_blackhole_copy_runtime.py`: 2 passed, 5 skipped
- `test_blackhole_gemm.py`: 5 passed, 2 skipped
- `test_blackhole_tvm_ffi_export.py`: 1 passed

---

## 9. 完成标准

- copy 和 GEMM runtime ABI 都切到 richer work descriptor schema
- `ExecutableSpec` / `KernelSpec` 能完整表达当前 copy/GEMM 的真实 work range
- `BlackholeModule` 不再依赖 `current_work_linear_id` / `tile_count` 或单组 `start/count` 猜 work 语义
- 对当前不支持的 range/stride 组合有明确 fail-fast 覆盖
- 现有 copy / GEMM direct-path 回归继续通过
- 文档与代码同步
