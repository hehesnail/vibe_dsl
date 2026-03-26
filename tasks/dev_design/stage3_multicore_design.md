# Stage 3: Multi-Core Runtime 调度设计

## 基本信息

- **文档ID**: `stage3_multicore_design`
- **日期**: 2026-03-26
- **状态**: formal direct host path 已实施完成
- **前置**: Stage 2D 已完成（copy + GEMM single-core E2E）
- **关联文档**:
  - `final_blackhole_backend_redesign.md` — 唯一总设计
  - `stage2d_ttmetal_contract_audit.md` — TT-Metal contract 缺口审计

---

## 1. 目标

让 copy 和 GEMM 在 Blackhole 14x10 Tensix 核心上做真正的 multi-core 并行执行。

当前状态：`AssignBlackholeCores` 已存在，能分析 logical grid，但 **hard-codes `cores_needed = 1`**，所有 work items 串行跑在同一个物理核心上。`BlackholeModule` 已有 per-work-item 循环，每个 work item 创建独立 `Program` 并串行 enqueue。

目标状态：多个 work items 分发到多个物理核心，在同一个 `Program` 内并行执行。

本阶段只解决 **host/runtime 调度**：

- `AssignBlackholeCores` 不再把所有 work item 压到单核
- `BlackholeModule` 不再按 work item 创建多个 `Program`
- copy 保持现有 lowering/codegen
- GEMM 通过 DSL kernel 中的 `bx/by` 索引获得多核 tile offset

本阶段不引入：

- K 维度切分
- 核间同步 / semaphore / multicast
- GEMM lowering/codegen 协议改写
- accessor / dtype layering 正式化

## 1.1 实施结果（2026-03-26）

- `AssignBlackholeCores` 已按 logical grid 分发多核 `work_packets`
- `BlackholeModule` 已切到单 `Program` + `CoreRangeSet` 多核 launch
- copy multi-core direct host path 已通过 TT-Sim
- GEMM multi-core direct host path 已通过 TT-Sim（`test_blackhole_gemm.py`: `7 passed`）

本轮真正补到的 GEMM multicore direct-path contract 缺口有 3 个：

1. host runtime 的 `num_k_tiles` 不能再从整张输入 buffer 大小反推  
   multi-core 下必须按 GEMM contract 的 `K / 32` 下发；single-core 之前只是偶然算对
2. segmented GEMM writer 不能按整张 output tensor 形状消费 output CB  
   每个 core 只应消费自己 `gemm_m_ x gemm_n_` 的一个 output tile
3. `transpose_B=True` 时，reader 必须按 host-transposed tiled-B layout 计算 tile index  
   multi-core 下 B tile 序列应是 `bx + kt * N_tiles`，不是未转置布局的连续 4 个 tile

另有一项明确不在本阶段内、且仍未解决的独立问题：

- `tilelang.compile(..., execution_backend="tvm_ffi")` 的 Blackhole wrapper/export path 仍会生成非法 host shim（`kernel_error_code = ;`）
- 这不是 formal `BlackholeModule` direct host path 的 blocker，因此没有与 Stage 3 direct-path 修复混做

---

## 2. 第一性原理调研结论

### 2.1 TT-Metal 多核 API（已验证）

TT-Metal 的 `CreateCircularBuffer`、`CreateKernel`、`CreateKernelFromString` 均接受 `std::variant<CoreCoord, CoreRange, CoreRangeSet>`。

`SetRuntimeArgs` 有三种重载：
1. `SetRuntimeArgs(program, kernel, CoreCoord, args)` — per-core 设置（当前已用）
2. `SetRuntimeArgs(program, kernel, vector<CoreCoord>, vector<vector<uint32_t>>)` — 批量 per-core
3. `SetRuntimeArgs(program, kernel, CoreRangeSet, args)` — 所有核心相同 args

**关键约束**：同一 `CoreRangeSet` 内的所有核心 CB config 必须相同。per-core 差异只能通过 runtime args 实现。这与我们的模型完全匹配。

参考 `tt_metal_repo/tt_metal/programming_examples/vecadd_multi_core/vecadd_multi_core.cpp`：
- kernel 创建用 `CoreRange`
- `SetRuntimeArgs` 对每个核心单独调用，传不同的 `start_tile_id`
- 一个 Program，一次 `EnqueueMeshWorkload`

### 2.2 Copy 多核已经自动工作（已验证）

**关键发现**：`codegen_blackhole.cc:BindThreadIndex` 已经把 `blockIdx.x/y` 映射成 `current_work_linear_id % grid_x` / `current_work_linear_id / grid_x`。

**传导链路**（以 `grid_indexed_staged_copy_kernel(grid_x=2, grid_y=3)` 为例）：

```
DSL: T.copy(A[by * tile_m, bx * tile_n], A_shared)
  → TIR: buffer index 包含 blockIdx.y * tile_m, blockIdx.x * tile_n
  → LowerBlackholeOps::InferStagedCopyBaseTileIndex:
      ZeroThreadAndLoopVars 只零化 threadIdx.*, 不零化 blockIdx.*
      → base_tile_index 保留 blockIdx.y * tile_m_tiles + blockIdx.x 表达式
  → read_tile_to_cb(buffer, tile_index_expr, cb_id, ...)
  → codegen: BindThreadIndex 把 blockIdx.x → (work_id % 2), blockIdx.y → (work_id / 2)
  → 最终: tile_index = (work_id / 2) * tiles_per_row + (work_id % 2)
```

这意味着 **copy 在 DSL kernel 定义 `T.Kernel(grid_x, grid_y)` 的前提下，只要 host 给不同核心不同 `current_work_linear_id`，tile index 就自动正确**。不需要改任何 pass 或 codegen。

### 2.3 GEMM 多核需要 DSL 级别变化（已验证）

**当前状态**：

```python
# 当前 GEMM 测试 kernel — T.Kernel(1, 1)
with T.Kernel(1, 1) as (bx, by):
    T.copy(A[0:block_M, 0:block_K], A_shared)      # 硬编码 offset
    T.copy(B[0:block_N, 0:block_K], B_shared)      # 硬编码 offset
    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
    T.copy(C_local, C[0:block_M, 0:block_N])       # 硬编码 offset
```

- `GenerateMatmulSequence` 只生成 K 维度循环，**没有 M/N 维度循环**
- `matmul_tiles(in0, in1, 0, 0, 0)` 的 tile index 全是常量 0
- `current_work_linear_id` 在 GEMM segment plan 中声明了但 **codegen 中未被使用**

**关键判断**：GEMM 多核的正确做法不是修改 `GenerateMatmulSequence`，而是让 DSL kernel 用 `bx/by` 来索引 A/B/C：

```python
# 多核 GEMM kernel — T.Kernel(Nt, Mt)
with T.Kernel(Nt, Mt) as (bx, by):
    T.copy(A[by*block_M : (by+1)*block_M, 0:K], A_shared)
    T.copy(B[bx*block_N : (bx+1)*block_N, 0:K], B_shared)
    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
    T.copy(C_local, C[by*block_M : (by+1)*block_M, bx*block_N : (bx+1)*block_N])
```

这样 `bx/by` 就进入了 reader/writer 的 buffer index 表达式，经过 `InferStagedCopyBaseTileIndex`（不零化 blockIdx）和 `BindThreadIndex`（映射到 work_id），**tile index 自动包含 per-core offset**。compute 部分（`matmul_tiles(0,0,0)`）仍然正确——每个核心的 CB 里只有自己需要的 tile。

**这意味着**：
- `GenerateMatmulSequence` **不需要改**
- `LowerBlackholeOps` **不需要改**
- `codegen_blackhole.cc` **不需要改**
- 只需要定义多核 GEMM 的 DSL kernel 并扩展测试

### 2.4 `BlackholeModule` 改动范围确认（已验证）

当前 `ExecuteDirect` 在 per-work-item 循环内创建：
1. `Program` — 需移到循环外
2. `CreateCircularBuffer` — 需移到循环外，参数改为 `CoreRangeSet`
3. `CreateKernel` — 需移到循环外，参数改为 `CoreRangeSet`
4. `SetRuntimeArgs` — 保持在循环内，per-core per-work_id

GEMM tilize/untilize 在循环外执行（per-buffer，不是 per-work-item），不受影响。

`mesh_device->shape()` 返回完整设备范围（14x10），不假设单核。

---

## 3. 设计方案

### 3.1 核心原则

1. **host 侧做分发，device kernel 不变**：`BindThreadIndex` 已经把 `blockIdx` 映射到 `current_work_linear_id`，多核只需给不同核心不同的 `work_id`
2. **同一 Program、多个 CoreRange**：TT-Metal 的 `Program` 天然支持多核
3. **改动集中在 `AssignBlackholeCores` + `BlackholeModule`**：pass 管线和 codegen 不需要改
4. **GEMM 多核通过 DSL kernel 的 `bx/by` 索引自然获得**，不需要改 lowering/codegen

### 3.2 执行模型变化

**当前（Stage 2D）**：
```
for work_id in [0..total_work):
    program = CreateProgram()
    CreateCB(program, single_core, ...)
    CreateKernel(program, single_core, ...)
    SetRuntimeArgs(program, kernel, single_core, args_with_work_id)
    EnqueueMeshWorkload(program, blocking=true)  // 串行
```

**目标（Stage 3）**：
```
program = CreateProgram()
core_range = CoreRangeSet(all_assigned_cores)
CreateCB(program, core_range, ...)         // CB 对所有核心创建
kernels = CreateKernel(program, core_range, ...)  // kernel 注册到多核
for each (core, work_id) in work_items:
    SetRuntimeArgs(program, kernel, core, args_with_work_id)  // per-core args
EnqueueMeshWorkload(program, blocking=true)  // 一次 launch
```

---

## 4. 分步实施

### Step 1: `AssignBlackholeCores` 解除单核限制

**改动范围**：`assign_blackhole_cores.cc`

**设计**：

```cpp
void AssignBlackholeCores::CalculateWorkDistribution(CoreAssignment& assignment) {
    const int total_work = std::max(1, assignment.grid_x * assignment.grid_y);
    // Stage 3: distribute work across multiple physical cores
    assignment.cores_needed = std::min(total_work, kBlackholeGridX * kBlackholeGridY);
    assignment.work_per_core = 1;  // 初版每核一个 work item
}
```

`StoreAssignment` 已有多 `work_packets` 生成的循环（line 202），只是当前 `cores_needed=1` 限制了只生成一个。

**约束**：
- 初版保持 `work_per_core = 1`（每核一个 work item），避免引入 `work_per_core > 1` 的复杂性
- `cores_needed = min(total_work, available_cores)`，不再 hard-code 为 1
- 如果 `total_work > available_cores`，允许多个 work item 复用同一物理核心，但这不是本阶段主验证路径
- 不改变 `core_plan` schema

**验证**：
- `test_blackhole_core_plan_preserves_logical_block_launch` — 验证 `grid_x=2, grid_y=3` 生成 6 个 work_packets 和 6 个 physical_cores
- `test_blackhole_copy_pass_attrs` — 验证 single-grid case（grid=1x1）不受影响

### Step 2: `BlackholeModule::ExecuteDirect` 多核 Program

**改动范围**：`blackhole_module.cc`、`blackhole_module.h`

**设计**：

```cpp
// Step 1: 收集所有 unique 物理核心
std::set<CoreCoord> all_cores;
for (const auto& item : work_items) all_cores.insert(item.core);

// Step 2: 构建 CoreRangeSet
// 每个 CoreCoord 包装为 CoreRange{core, core}
std::set<CoreRange> ranges;
for (const auto& core : all_cores) {
    ranges.insert(CoreRange{core, core});
}
CoreRangeSet core_range_set(ranges);

// Step 3: 单个 Program
Program program = CreateProgram();
CreateCircularBuffersFromSpec(program, core_range_set, spec);

// Step 4: Kernel 注册到多核
std::vector<KernelHandle> kernels;
for (size_t ki = 0; ki < spec.kernels.size(); ++ki) {
    kernels.push_back(CreateKernelFromSpec(
        program, core_range_set, spec.kernels[ki], kernel_paths[ki]));
}

// Step 5: Per-core runtime args（保持在循环内）
for (const auto& item : work_items) {
    for (size_t ki = 0; ki < spec.kernels.size(); ++ki) {
        auto args = BuildRuntimeArgsFromSpec(
            spec.kernels[ki], spec, item.work_id, ...);
        SetRuntimeArgs(program, kernels[ki], item.core, args);
    }
}

// Step 6: 一次 launch
distributed::MeshWorkload workload;
distributed::MeshCoordinateRange device_range(mesh_device->shape());
workload.add_program(device_range, std::move(program));
distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);
```

**函数签名变化**：

```cpp
// 当前
static void CreateCircularBuffersFromSpec(
    Program& program, const CoreCoord& core, const ExecutableSpec& spec);
static KernelHandle CreateKernelFromSpec(
    Program& program, const CoreCoord& core,
    const KernelSpec& kernel, const std::string& kernel_path);

// 改为
static void CreateCircularBuffersFromSpec(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const ExecutableSpec& spec);
static KernelHandle CreateKernelFromSpec(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const KernelSpec& kernel, const std::string& kernel_path);
```

TT-Metal API 端不需要任何改动，`CreateCircularBuffer` 和 `CreateKernel` 已经接受
`std::variant<CoreCoord, CoreRange, CoreRangeSet>`。

**实现约束**：
- 所有 segment kernel 注册到同一个 `CoreRangeSet`
- `SetRuntimeArgs` 继续按 `CoreCoord` 单独下发，保持每核 `current_work_linear_id` 独立
- 单核 case 必须仍然走同一条主路径，不能新增单核专用 fallback

**验证**：
- 单核 case（grid=1x1）所有现有 copy/GEMM 测试不回退
- 新增 multi-core copy runtime 测试

### Step 3: Copy 多核 E2E

**改动范围**：测试

Copy 已有 `grid_indexed_staged_copy_kernel(grid_x=2, grid_y=3)` 的 pipeline 测试。需要新增 runtime 测试（TT-Sim 真执行）。

**验证内容**：
- `grid_x=2, grid_y=3` 的 96x64 float16 copy，6 个 work items 分发到 6 个物理核心
- 结果与 PyTorch 参考一致
- 单核 copy runtime 测试不回退

### Step 4: GEMM 多核

**改动范围**：测试（DSL kernel 定义）

**关键设计决策**：不改 lowering/codegen，只定义使用 `bx/by` 索引的 GEMM DSL kernel。

```python
def grid_gemm_kernel(M=64, N=64, K=128, block_M=32, block_N=32):
    Mt = M // block_M   # 2
    Nt = N // block_N   # 2

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "bfloat16"),
        B: T.Tensor((N, K), "bfloat16"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(Nt, Mt) as (bx, by):
            A_shared = T.alloc_shared((block_M, K), "bfloat16")
            B_shared = T.alloc_shared((block_N, K), "bfloat16")
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            T.copy(A[by * block_M : (by + 1) * block_M, 0:K], A_shared)
            T.copy(B[bx * block_N : (bx + 1) * block_N, 0:K], B_shared)
            T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C[by * block_M : (by + 1) * block_M,
                              bx * block_N : (bx + 1) * block_N])

    return main
```

**传导链路**（以 `Mt=2, Nt=2, work_id=3` 即 `bx=1, by=1` 为例）：

```
Reader:
  T.copy(A[1*32:2*32, 0:128], A_shared)
  → buffer index: row = by*32 = blockIdx.y * 32
  → InferStagedCopyBaseTileIndex: base = (blockIdx.y * 32 / 32) * (128/32) = blockIdx.y * 4
  → codegen: blockIdx.y = work_id / 2 = 3 / 2 = 1
  → final: base_tile_index = 1 * 4 = 4  ✓（读 A 的第 2 行 tile）

  T.copy(B[1*32:2*32, 0:128], B_shared)
  → buffer index: row = bx*32 = blockIdx.x * 32
  → codegen: blockIdx.x = work_id % 2 = 3 % 2 = 1
  → final: base_tile_index = 1 * 4 = 4  ✓（读 B 的第 2 行 tile）

Compute:
  matmul_tiles(in0, in1, 0, 0, 0)  — 不变，CB 内只有本核的 tile

Writer:
  T.copy(C_local, C[1*32:2*32, 1*32:2*32])
  → buffer index: row = by*32, col = bx*32
  → final: base_tile_index = by * (N/32) + bx = 1 * 2 + 1 = 3  ✓
```

**GEMM tilize/untilize 影响**：
- `BlackholeModule` 的 tilize/untilize 是对整个 host tensor 做的（全量），不是 per-core
- 多核 GEMM 每个核心读取 A/B 的不同 tile 范围，但 DRAM buffer 包含完整 tilized 数据
- 每个核心写入 C 的不同 tile 范围，最终 untilize 对整个 C tensor 做

**验证**：
- `grid_gemm_kernel(M=64, N=64, K=128, block_M=32, block_N=32)` → 4 核并行
- 结果与 PyTorch `torch.matmul` 参考一致

### Step 5: 文档同步

- `progress.md` — Stage 3 状态推进
- `final_blackhole_backend_redesign.md` — Stage 3 状态更新
- `memory/general_dev.md` — 多核经验沉淀

---

## 4.1 最终实施选择

经复核，Stage 3 采用以下正式路径：

1. **只改 host/runtime 分发**
   - `AssignBlackholeCores` 放开多核分配
   - `BlackholeModule` 改为单 `Program` 多核 launch
2. **copy 不改 lowering/codegen**
   - 依赖现有 `BindThreadIndex` 与 `blockIdx` 传导链路
3. **GEMM 不改 lowering/codegen**
   - 通过 DSL kernel 中的 `bx/by` 索引让 reader/writer 自然获得 per-core tile offset

不采用的方案：

- 在 `GenerateMatmulSequence` 或 `codegen_blackhole.cc` 中新增 GEMM per-core offset 特判
- 保留多 `Program` 串行 launch 作为“伪多核”过渡方案

原因：

- 当前第一性原理调研已经证明 `blockIdx -> current_work_linear_id` 的映射链路成立
- Stage 3 的目标是验证 host/runtime multi-core materialization，而不是再改一轮 lowering/codegen 协议
- 继续往 codegen/lowering 补 per-core 特判会违反当前“优先从 IR 获取信息，不让 runtime/codegen 猜”的仓库约束

---

## 5. 不在 Stage 3 范围内的

| 项目 | 原因 | 什么时候做 |
|------|------|-----------|
| K 维度切分 | 需跨核累加 + semaphore | Stage 4 |
| 核间数据流 / multicast | 需 semaphore + noc_semaphore_inc/wait | Stage 4 |
| work_per_core > 1 的不均匀分发 | 初版每核 1 work item 足够 | 后续优化 |
| accessor schema 正式化 | 协议质量，不影响多核正确性 | P3 |
| dtype 分层正式化 | 协议质量 | 后续 |
| copy 统一进 reader+writer 2-kernel 模型 | 架构债，不影响多核 | 后续 |
| dynamic CB | 当前 static CB 足够 | 后续 |

---

## 6. 影响范围

### 6.1 必须改的文件

| 文件 | 改动内容 | 改动量 |
|------|---------|--------|
| `assign_blackhole_cores.cc` | `CalculateWorkDistribution` 解除 `cores_needed=1` | ~5 行 |
| `blackhole_module.cc` | `ExecuteDirect` 改为单 Program 多核；`CreateCircularBuffersFromSpec`/`CreateKernelFromSpec` 签名变为 `CoreRangeSet` | ~40 行 |
| `blackhole_module.h` | 函数声明 | ~2 行 |
| 测试文件 | 新增多核 copy/GEMM 测试 kernel 和 runtime 测试 | 新增 |

### 6.2 不需要改的文件

| 文件 | 原因 |
|------|------|
| `lower_blackhole_ops.cc` | `blockIdx.*` 不被零化，tile index 自然包含 per-core offset |
| `codegen_blackhole.cc` | `BindThreadIndex` 已正确映射 `blockIdx` → `work_id` |
| `split_blackhole_kernel.cc` | segment 结构不变 |
| `rt_mod_blackhole.cc` | `CorePlan` 提取逻辑已通用 |
| `phase.py` / `lower.py` | pass 管线不变 |

---

## 7. 协议变化

### 7.1 `CorePlan` — 不变

当前 schema 已经能表达多核。不需要新增字段。

### 7.2 `ExecutableSpec` — 不变

CB configs、kernel specs、runtime args schema 都不需要因多核而改变。不同核心用相同的 CB config 和 kernel source，只有 runtime args 不同（`current_work_linear_id`）。

### 7.3 TT-Metal API 使用变化

| API | 当前 | 目标 |
|-----|------|------|
| `CreateCircularBuffer` | `CoreCoord` | `CoreRangeSet` |
| `CreateKernel` / `CreateKernelFromString` | `CoreCoord` | `CoreRangeSet` |
| `SetRuntimeArgs` | `CoreCoord` per work_item | `CoreCoord` per work_item（不变）|
| `EnqueueMeshWorkload` | N 次（per work_item） | 1 次 |
| `Program` | N 个 | 1 个 |

---

## 8. 风险与降级

| 风险 | 影响 | 降级策略 |
|------|------|---------|
| TT-Sim 多核行为与真实硬件不一致 | 测试通过但真机 fail | 先在 TT-Sim 验证正确性 |
| CoreRangeSet 构造不正确（非连续核心） | TT-Metal 报错 | 每个 CoreCoord 独立包装为 `CoreRange{core, core}`，避免假定连续 |
| GEMM 多核 tilize/untilize 不匹配 | 数值错误 | tilize/untilize 是全量的，与多核无关 |
| 某个核心的 work_id 对应的 tile 超出 buffer 范围 | crash | `AssignBlackholeCores` 保证 `cores_needed ≤ total_work` |
| copy 和 GEMM 的 segment plan 不同（单 kernel vs 3 kernel），CoreRangeSet 注册需要对齐 | 注册错误 | 每个 kernel spec 都注册到相同的 CoreRangeSet |

---

## 9. 完成标准

- Copy 在 `grid > 1` 时真正分发到多个物理核心，且 TT-Sim E2E 结果正确
- GEMM 在 M/N 维度切分后多核执行（DSL kernel 用 `bx/by` 索引），且 TT-Sim E2E 结果正确
- 单核 case（grid=1x1）不回退
- 所有现有测试不回退
- `BlackholeModule` 从 N 个 Program 串行 enqueue 变为 1 个 Program 多核 launch
- `progress.md` 与 `final_blackhole_backend_redesign.md` 同步更新
