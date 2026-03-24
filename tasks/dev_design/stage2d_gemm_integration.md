# Stage 2D 设计：GEMM 接入

## 基本信息

- **文档ID**: `stage2d_gemm_integration`
- **日期**: 2026-03-24（2026-03-24 修订：架构重设计，引入 SplitBlackholeKernels pass）
- **状态**: 执行中
- **对应阶段**: Stage 2D — single-core GEMM 语义集成 + true E2E

---

## 1. TT-Metal GEMM 编程模型（已验证）

### 1.1 三 kernel 模型

```
Reader  (RISCV_0/BRISC)    — noc_async_read DRAM→CB0(A), DRAM→CB1(B)
Compute (TRISC)             — mm_init(0,1,16) → k_loop(wait, matmul_tiles, pop) → pack_tile
Writer  (RISCV_1/NCRISC)   — cb_wait_front(16) → noc_async_write CB16→DRAM
```

Host 侧 `CreateKernel` 配置：
- Reader：`DataMovementConfig{RISCV_0, NOC_0_default}`
- Compute：`ComputeConfig{HiFi4, fp32_dest_acc_en=true}`
- Writer：`DataMovementConfig{RISCV_1, NOC_1_default}`

### 1.2 CB ID 是全局共享的（关键约束）

Host 创建 CB 一次，3 个 kernel 引用**同一批物理 L1 地址**：

```
Host:     CreateCircularBuffer(CB0, size=2*TILE_A)   ← A input, double-buffered
          CreateCircularBuffer(CB1, size=2*TILE_B)   ← B input, double-buffered
          CreateCircularBuffer(CB16, size=2*TILE_C)  ← C output

Reader:   cb_reserve_back(CB0); cb_push_back(CB0)   ← 写 CB0/CB1
          cb_reserve_back(CB1); cb_push_back(CB1)

Compute:  cb_wait_front(CB0); matmul_tiles(CB0,CB1,0,0,0); cb_pop_front(CB0)
          cb_reserve_back(CB16); pack_tile(0,CB16); cb_push_back(CB16)

Writer:   cb_wait_front(CB16); noc_async_write; cb_pop_front(CB16) ← 读 CB16
```

CB ID **不按 kernel 隔离**，reader 写入的 CB0/CB1 就是 compute 读的 CB0/CB1。

### 1.3 用户必须显式写出完整数据流

GEMM kernel 必须包含全部三段：

```python
T.copy(A, A_shared)                    # DRAM→SRAM → reader kernel
T.copy(B, B_shared)                    # DRAM→SRAM → reader kernel
T.gemm(A_shared, B_shared, C_local)    # compute   → compute kernel
T.copy(C_local, C)                     # SRAM→DRAM → writer kernel
```

纯 copy（无 compute op）= reader + writer，可用单 kernel（fused_dataflow）。

### 1.4 Runtime args（各 kernel 独立）

- Reader：`{a_dram_addr, b_dram_addr, num_k_tiles, core_id}`
- Compute：`{num_k_tiles}`
- Writer：`{c_dram_addr, core_id}`

---

## 2. 架构设计（2026-03-24 修订版）

### 2.1 核心原则

1. **`SplitBlackholeKernels` 是通用 pass**，不写死 GEMM 逻辑，按数据流方向（dram_to_cb / compute / cb_to_dram）通用分类
2. **CB ID 全局统一**，`PlanBlackholeCB` 跑在整个 device func 上，3 个 kernel 共享同一份 `blackhole.cb_configs`
3. **`LowerBlackholeOps` 只负责 lower**，不负责 kernel split；segment plan 由 `SplitBlackholeKernels` 写入

### 2.2 Pass 管线顺序

```
AnnotateBlackholeCopySemantics      [Stage 2C，已完成]
  ↓ ForNode.annotations["blackhole.copy_semantics"] = {direction, ...}

LowerTileOp (Blackhole GEMM skip)   [Step 1，已完成]
  ↓ T.gemm_py(...) 保留为 Evaluate(Call("tl.tileop.gemm_py", ...))
  ↓ T.copy → ForNode(BufferStore)

SplitBlackholeKernels               [Step 2，本次新增]
  ↓ 为每个顶层 stmt 加 AttrStmt("blackhole.segment_kind", "reader"/"compute"/"writer")
  ↓ 写入 blackhole.segment_plan（3-kernel 或 1-kernel）

LowerBlackholeOps                   [Step 3，本次修正]
  ↓ lower ForNode/EvaluateNode 到 builtin 调用
  ↓ 收集 cb_requirements（A/B/C 三个 CB）
  ↓ 不再写 segment_plan（已由 SplitBlackholeKernels 写入）

PlanBlackholeCB                     [已完成]
  ↓ 全局分配 CB0/CB1/CB16

rt_mod_blackhole.cc                 [Step 4，扩展]
  ↓ 读 segment_plan → 生成 3 个 KernelSpec + kernel source

BlackholeModule::ExecuteDirect()    [Step 5，扩展]
  ↓ 按 kind 注册 reader(RISCV_0)/compute(TRISC)/writer(RISCV_1)
```

### 2.3 AttrStmt 注解格式

```
AttrStmt(
  node = StringImm("blackhole.segment_kind"),
  attr_key = "blackhole.segment_kind",
  value = StringImm("reader" | "compute" | "writer"),
  body = original_stmt
)
```

### 2.4 `blackhole.segment_plan` schema（3-kernel）

```python
blackhole.segment_plan = [
  {"name": "reader",  "kind": "reader",  "core_type": "brisc",
   "runtime_args": [
     {"name": "a_addr",   "kind": "input_buffer_addr32",    "dtype": "uint32"},
     {"name": "b_addr",   "kind": "input_buffer_addr32",    "dtype": "uint32"},
     {"name": "num_k_tiles", "kind": "num_k_tiles",         "dtype": "uint32"},
     {"name": "core_id",  "kind": "current_work_linear_id", "dtype": "uint32"},
   ]},
  {"name": "compute", "kind": "compute", "core_type": "trisc",
   "runtime_args": [
     {"name": "num_k_tiles", "kind": "num_k_tiles", "dtype": "uint32"},
   ]},
  {"name": "writer",  "kind": "writer",  "core_type": "ncrisc",
   "runtime_args": [
     {"name": "c_addr",  "kind": "output_buffer_addr32",    "dtype": "uint32"},
     {"name": "core_id", "kind": "current_work_linear_id",  "dtype": "uint32"},
   ]},
]
```

runtime_args 里的 buffer name 来自全局 DRAM 参数（`A`, `B`, `C`），而不是 shared buffer 名称。

---

## 3. 各步骤说明

### Step 1：LowerTileOp Blackhole GEMM skip ✅ 已完成

- `src/transform/lower_tile_op.cc`：`GemmPyNode` 对 Blackhole target 跳过展开
- `tilelang/tileop/gemm/__init__.py`：`_select_gemm_instruction` Blackhole 返回 `GemmInst.Scalar`
- `src/target/utils.h/cc`：新增 `TargetIsBlackhole()`
- 验证：`T.gemm_py(...)` 保留在 TIR，copy 测试 `15 passed, 5 skipped, 1 xfailed`

---

### Step 2：`SplitBlackholeKernels` pass（新增）

**文件**：`src/transform/split_blackhole_kernels.h/cc`

#### 分类规则（通用，不写死 GEMM）

扫描 func body 内顶层 stmt 列表，按状态机分类：

| 条件 | 分类 |
|------|------|
| ForNode 带 `blackhole.copy_semantics.direction == "dram_to_cb"` | reader |
| ForNode 带 `blackhole.copy_semantics.kind == "fused_staged_copy"` | reader（先读后写，暂按 reader 处理）|
| EvaluateNode 的 op 是 compute op（`tl.tileop.gemm_py` 等） | compute |
| ForNode 带 `blackhole.copy_semantics.direction == "cb_to_dram"` | writer |
| Allocate / 其他 | 透传，不加注解 |

#### 触发条件

**只有当 body 中存在 compute op 时才做 3-kernel 分割**。纯 copy（无 compute op）不触发此 pass，走原有 fused_dataflow 单 kernel 路径。

#### 产出

1. 每个需要分类的 stmt 包裹在 `AttrStmt("blackhole.segment_kind", kind, stmt)` 中
2. 写入 `blackhole.segment_plan`（3-kernel schema，runtime_args 基于 DRAM 参数名）

---

### Step 3：`LowerBlackholeOps` 修正（去掉 segment plan 职责，补 planner-driven CB 绑定）

**文件**：`src/transform/lower_blackhole_ops.cc/h`

#### 当前已完成

- `IsMatmulCall` 修正（`"tl.tileop.gemm_py"`）
- `ExtractGemmInfo` — 从 call args 提取 A/B/C buffer + M/N/K，向 `cb_requirements_` 注册 CB0/CB1/CB16
- `GenerateMatmulSequence` 已生成正式 compute builtin 序列（`mm_init` / `matmul_tiles` / `pack_tile`）
- `DataTypeToDataFormat` helper

#### 删除

- `StoreGemmSegmentPlan()` — 移出，此职责归 `SplitBlackholeKernels`
- `StoreSegmentPlan()` 的 GEMM 路由 (`saw_matmul_op_` → `StoreGemmSegmentPlan`)

#### 当前仍需补齐

- `GenerateMatmulSequence` 不能继续写死 `CB0/CB1/CB16`
- compute builtin 的 CB ID 必须从 planner 产出的正式 binding/schema 读取，而不是依赖 allocator 当前顺序
- 目标是让 GEMM compute 与 copy 一样，消费 split 后显式 plan，而不是把 `PlanBlackholeCB` 的当前分配顺序当成隐式协议

#### 本次实现收束

- `LowerBlackholeOps` 中的 GEMM builtin 先写入**显式占位符 CB ID**，不再把 `0/1/16` 当最终硬件 ID
- 同时写出 `blackhole.gemm_cb_placeholders`：
  - `a.requirement_name`
  - `b.requirement_name`
  - `c.requirement_name`
  - 对应 placeholder id
- `CodeGenBlackhole` 在发射 compute/dataflow builtin 时，读取：
  - `blackhole.gemm_cb_placeholders`
  - `blackhole.cb_bindings`
  并将 placeholder 解析成 planner 最终分配的 `cb_id`
- 最终效果：
  - CB 分配仍只由 `PlanBlackholeCB` 决定
  - codegen / runtime 只消费 planner 结果
  - 不再把当前 allocator 顺序偷渡成协议

#### 修正

`StoreSegmentPlan()` 改为：若 `blackhole.segment_plan` 已由 `SplitBlackholeKernels` 写入，则跳过（不覆盖）：

```cpp
void LowerBlackholeOps::StoreSegmentPlan(PrimFunc& func) {
  // If already set by SplitBlackholeKernels, do not overwrite
  if (func->GetAttr<ffi::Array<ffi::Any>>("blackhole.segment_plan")) return;
  if (!needs_copy_runtime_args_) return;
  // ... existing fused_dataflow single-kernel path for pure copy ...
}
```

---

### Step 4：`rt_mod_blackhole.cc` 多 segment 提取

**文件**：`src/target/rt_mod_blackhole.cc`

- `ExtractSegmentPlan` 改为遍历所有 segment（不只取 `[0]`），为每个 segment 提取：
  - `name`
  - `kind`
  - `core_type`
  - `runtime_args`
- `KernelArgSpec` 扩展 `buffer` 字段，segment-level `runtime_args` 显式绑定 DRAM 参数名（`A` / `B` / `C`），避免 direct path 再按位置猜第二个 input 属于哪个 reader arg
- `ExecutableSpec` 额外保留 `tvm_arg_names`，供 runtime 把 packed-func 传入的 tensor 绑定回原始 buffer 名
- source generation 不再新写 3 套模板 emitter；改为：
  - 从原始 device `PrimFunc` 生成 segment-specific `PrimFunc`
  - body 只保留对应 `blackhole.segment_kind` 的 stmt，非本 segment 的 annotated stmt 直接删除
  - `blackhole.core_type` / `blackhole.runtime_args` 改写成当前 segment 的 attrs
  - 继续复用 `CodeGenBlackhole`
- 这样 reader / compute / writer 仍共享：
  - `blackhole.cb_configs`
  - `blackhole.cb_bindings`
  - `blackhole.gemm_cb_placeholders`
  - `blackhole.core_plan`
  从而保持 planner → codegen 的现有协议不变

---

### Step 5：`BlackholeModule::ExecuteDirect()` 3-kernel 支持

**文件**：`src/target/blackhole_module.cc`

按 `KernelSpec::kind` 路由：
- `"reader"`  → `DataMovementConfig{RISCV_0, NOC_0_default}`
- `"compute"` → `ComputeConfig{HiFi4, fp32_dest_acc_en=false}`（先与当前 copy/compute 配置保持一致，后续再按真实 GEMM 需要收正）
- `"writer"`  → `DataMovementConfig{RISCV_1, NOC_1_default}`
- `"fused_dataflow"` → 现有 BRISC 路径（兼容 copy）

runtime arg materialization 同步收正：

- 不再把所有 input/output tensor 拼成一个大 DRAM buffer
- direct path 改为为每个 TVM buffer 参数分别创建 `MeshBuffer`
- `BuildRuntimeArgsFromSpec()` 按 `KernelArgSpec.buffer` 查找对应 buffer address：
  - reader 的两个 `input_buffer_addr32` 分别绑定 `A` / `B`
  - writer 的 `output_buffer_addr32` 绑定 `C`
- 对仍未显式带 `buffer` 的旧 copy schema 保留位置回退，避免打断 fused_dataflow 路径

---

### Step 6：E2E 测试

**文件**：`tilelang_repo/testing/python/target/blackhole/test_blackhole_e2e.py`

```python
def test_blackhole_gemm_basic():
    # 32x32, K=128 (4 K-tiles), bfloat16, single core
    # 用户写: T.copy(A,A_s) + T.copy(B,B_s) + T.gemm(A_s,B_s,C_l) + T.copy(C_l,C)
    # 期望: CB0/CB1/CB16 正确分配，3 kernel 正确注册，结果与 numpy.matmul 一致
```

---

## 4. 不做的事（Stage 2D 范围外）

- 不做 multi-core GEMM（单核）
- 不做 transA/transB（non-transposed only）
- 不支持 fused_staged_copy（dram→sram→dram）作为 reader+writer 融合（暂作 reader 处理）
- 不做 dynamic shape GEMM（M/N/K 必须是编译期常量 IntImm）
- 不单独引入三套模板 emitter；仍优先复用 TIR → `CodeGenBlackhole`
- **不统一 copy 的 segment 模型**（见下方架构说明）

---

## 5. 架构说明：copy 与 GEMM 的 segment 模型不对称性

### 5.1 当前状态

| 场景 | segment 模型 | kernel 数量 | 执行核 |
|------|-------------|-------------|--------|
| 纯 copy | `fused_dataflow` | 1（BRISC） | BRISC 顺序完成 read + write |
| GEMM | `reader + compute + writer` | 3 | BRISC read、TRISC compute、NCRISC write |

`SplitBlackholeKernel` 对无 compute op 的函数是 strict no-op，copy 的 segment plan 由 `LowerBlackholeOps::StoreSegmentPlan` 写入（`fused_dataflow`）。

### 5.2 为何当前不统一

纯 copy 不包含 TRISC compute，BRISC 自身可以顺序完成 read→CB→write，不强制要求 reader+writer 拆分。单 kernel 实现更简单，现有 `LowerBlackholeOps` 已稳定产出该路径，无必要在 GEMM 接入期间同步修改。

### 5.3 后续统一方向（架构债）

将 copy 也统一进 reader+writer 2-kernel 模型是合理的后续任务，收益包括：

- `SplitBlackholeKernel` 统一覆盖所有情形（copy 和 GEMM）
- `rt_mod_blackhole` / `BlackholeModule` 只维护一套多 segment 路径，不需要同时处理 `fused_dataflow` 单 kernel 和 3-kernel 两种 schema
- BRISC（read）+ NCRISC（write）并行执行，大数据量时有潜在吞吐提升

**触发条件**：GEMM E2E（Step 4/5/6）稳定通过，且 `rt_mod_blackhole` / `BlackholeModule` 已有成熟的多 segment 分发逻辑后，再做统一重构。

---

## 6. 已完成状态

| 步骤 | 状态 |
|------|------|
| Step 1：LowerTileOp skip | ✅ 完成 |
| Step 2：SplitBlackholeKernels | ✅ 完成 |
| Step 3：LowerBlackholeOps 修正 | ✅ 完成（planner-driven CB placeholders + codegen resolve 已落地） |
| Step 4：rt_mod_blackhole 多 segment | ✅ 完成（segment-specific PrimFunc + KernelSpec 生成） |
| Step 5：BlackholeModule 3-kernel | ✅ 完成（多 buffer direct path + kind-aware kernel registration） |
| Step 6：E2E 测试 | 🔄 进行中（当前被 `MergeSharedMemoryAllocations expects flat memory buffers` 阻塞） |
