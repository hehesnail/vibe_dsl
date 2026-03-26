# Stage 2E Blackhole Device Resource Semantics

- 文档ID: `stage2e_blackhole_device_resource_semantics`
- 状态: 已完成
- 日期: 2026-03-25（设计定稿）
- 相关总设计: `tasks/dev_design/final_blackhole_backend_redesign.md`

## 1. 背景

当前 Blackhole 后端已经把 copy 主路径收回到正式 pipeline，并为 GEMM 接入了：

- `LowerTileOp` 对 Blackhole 保留 `tl.gemm_py`
- `SplitBlackholeKernel`
- `LowerBlackholeOps`
- `PlanBlackholeCB`
- `ExecutableSpec -> rt_mod_blackhole -> BlackholeModule`

但 GEMM 在 `lower()` 主线中仍会卡在 generic host/device pass 链上。当前观察到的报错包括：

- `MergeSharedMemoryAllocations expects flat memory buffers`
- `In PrimFunc main variables [C_local] are used, but are not passed in as API arguments`
- `Only one dynamic shared memory allocation is allowed`

这些问题不是 GEMM 单点逻辑错误，而是 Blackhole 编程模型与 generic TIR host/device pipeline 的资源语义边界不一致。

根据 `tt_metal_repo` 中的 programming examples、matmul lab 和 advanced topics，TT-Metal/Blackhole 的核心对象分为三层：

1. host-visible objects
   - DRAM buffer
   - scalar/runtime args
   - kernel handle / core range / kernel kind
2. transport resources
   - Circular Buffer (CB)
   - L1 上的 FIFO/page queue
   - producer/consumer 协议：`cb_reserve_back / cb_wait_front / cb_push_back / cb_pop_front`
3. compute-private resources
   - Dst/tile registers
   - unpacker/packer/FPU 配置
   - compute kernel 内部 accumulator / fragment state

而当前 TIR 中：

- `shared/shared.dyn` 仍被当作 generic shared memory buffer
- `local.fragment` 仍被当作 generic local buffer/handle

这导致 generic pass 被迫按错误模型解释 Blackhole 设备内部资源。

## 2. 问题定义

### 2.1 当前错位

当前 pipeline 在 Blackhole 上混淆了两类边界：

1. host/device ABI 边界
2. device 内部资源边界

generic pass 默认认为：

- device region 中的 undefined var 应升成 device func 参数
- `shared.dyn` 是 launch-time / software-managed dynamic shared memory
- compute kernel 的 local object 若跨语句使用，仍可按普通 local var/buffer 处理

但 Blackhole 实际上要求：

- `local.fragment` 不应越过 host/device ABI 边界
- `shared/shared.dyn` 不应等价为 generic dynamic shared allocation
- CB / tile registers / unpack-pack-FPU 协议都属于 device-private resource model

### 2.2 为什么 copy 已部分缓解，而 GEMM 没有

copy 已通过 `AnnotateBlackholeCopySemantics` 在 destructive pass 之前把数据流语义显式化，因此 `LowerBlackholeOps` 不再完全依赖晚期 loop/buffer 形态恢复 copy 语义。

GEMM 当前只有：

- `tl.gemm_py` call 被保留
- `SplitBlackholeKernel` 能识别其 segment kind

但“device-private resource”和“host-visible ABI”的边界仍未显式表达，因此：

- `SplitHostDevice` 会把 `local.fragment` 对应对象视为 undefined var 并升成参数
- `MergeSharedMemoryAllocations` 会把 `shared.dyn` 当作 generic shared buffer 去合并
- `LowerDeviceKernelLaunch` 会把多个 `shared.dyn` 当作多个 dynamic shared alloc

### 2.3 这不是 GEMM 特例

后续若接入：

- element-wise compute
- reduction
- softmax
- 其他需要 compute-private state 的 tile op

只要继续把 Blackhole 设备内部资源伪装成普通 TIR buffer/scope，generic pass 仍会重复发生同类误判。

因此问题的本质是：

**Blackhole 缺少一层统一的 device resource semantics/canonicalization。**

## 3. 目标

本设计的目标不是直接修一个 GEMM case，而是建立可扩展的 Blackhole 设备资源语义边界。

具体目标：

1. 在进入 generic host/device ABI pass 前，显式区分：
   - host-visible tensors/scalars
   - device-private CB resources
   - device-private compute-private resources
2. 不再让 `local.fragment` / `shared.dyn` 以 generic buffer/alloc 形态泄漏到 ABI 层
3. 让后续 GEMM、eltwise、reduction 等 compute op 可以共用同一套资源语义，而不是按算子逐个补特判
4. 保持与 `ExecutableSpec / segment_plan / runtime_args / cb_configs` 现有主协议兼容

## 4. 非目标

本设计不在本阶段直接完成：

- 新 tile op 的完整 lower
- multi-core 调度策略重写
- `LowerBlackholeOps` 的全量算子重构
- 立即替换所有现有 TIR 节点为全新 IR dialect

本设计优先解决的是“语义承载层”和“pass 边界”。

## 5. 核心判断

### 5.1 哪些资源是 host-visible

以下对象应保留在 host/device ABI 边界上：

- DRAM tensor 参数
- scalar 参数
- runtime args 中需要 host materialization 的值
- core/segment/kernel kind 等 host-side launch metadata

这些对象可以继续走 generic `SplitHostDevice` / `MakePackedAPI` 风格的 ABI 主线。

### 5.2 哪些资源是 device-private

以下对象不应跨 ABI 边界：

- CB identity / role / page queue
- L1 transport slots
- fragment / accumulator / dst-register state
- unpacker / packer / FPU 配置依赖

这些对象属于 device-private resource model，应在进入 ABI pass 之前就完成 canonicalization。

### 5.3 `shared.dyn` 和 `local.fragment` 当前信息不足

`scope` 只表达“像什么”，不表达“怎么被用”。

例如：

- `shared.dyn` 不能表达 CB role / page size / producer-consumer 关系
- `local.fragment` 不能表达其是 Dst register-backed accumulator，还是普通 local scratch

因此仅靠现有 storage scope，不足以支撑 Blackhole 后续可扩展 compute lowering。

结论：

**需要扩展 IR 语义，但应优先扩“结构化资源语义”，而不是立刻发明一整套全新节点。**

## 6. 第一性原理分析

### 6.1 根本冲突

TIR 的 `StorageScope` 系统继承自 GPU 内存模型，把两个正交概念混为一谈：

- **存储层级**（数据在哪里：寄存器、L1、L2、DRAM）
- **资源语义**（数据如何管理：裸内存、FIFO 队列、寄存器文件）

GPU 两者是 1:1 映射：`shared` = L1 且是一块连续内存，`local` = 寄存器且可自由寻址。所以混为一谈没问题。

**Blackhole 打破了这个假设：**

| 资源 | 存储位置 | 资源语义 | 当前 TIR scope | 语义匹配？ |
|------|---------|---------|---------------|-----------|
| CB (Circular Buffer) | L1 SRAM | 独立 FIFO 队列，有 reserve/push/wait/pop 协议 | `shared.dyn` | ❌ GPU 共享内存是一块连续区域 |
| Dst 累加器 | Tile 寄存器 | 寄存器文件，有 acquire/commit/wait/release 语义 | `local.fragment` → `local` | ❌ GPU local 是可寻址内存 |
| DRAM Tensor | DRAM | 可寻址内存，host 分配 device 按地址访问 | `global` | ✅ 语义匹配 |

这不是命名问题，是 **IR 的类型系统无法表达 Blackhole 硬件资源的本质**：

- CB 不是”一块共享内存”——它是多个独立的 FIFO 队列，每个有 index、page_size、role、producer-consumer 关系
- Dst 累加器不是”一片本地内存”——它是寄存器文件，没有地址，不可跨 ABI 传递，有 acquire/release 生命周期

### 6.2 已有先例

`StorageRank` 里已经有类似的硬件资源类型扩展：

- `kWMMAMatrixA/B` (NVIDIA WMMA 矩阵碎片)
- `kMMAMatrixA/B/C` (NVIDIA MMA 矩阵碎片)
- `kMetalSimdGroup` (Apple Metal SIMD group 内存)
- `kAMXTMM` (Intel AMX tile matrix)

这些都是 **特定硬件资源的正式 IR 表达**，不是 workaround。Blackhole 的 CB 和 Dst 累加器同样值得拥有正式的 IR 资源类型。

### 6.3 为什么 copy 已部分缓解而 GEMM 没有

copy 有 **一个** `shared.dyn` buffer（`MergeSharedMemoryAllocations` 对单个 buffer 是 no-op）且 **没有** `local.fragment` buffer。

GEMM 有 **两个** `shared.dyn`（触发 merge crash：2D buffer 不满足 flat-buffer 前提）和 **一个** `local.fragment`（被 `PlanAndUpdateBufferAllocationLocation` 提升到设备区域外 → `SplitHostDevice` 的 `VarUseDefAnalyzer` 视其为 undefined var → 提升为设备函数参数）。

## 7. 设计方案：扩展 StorageRank + 规范化 Pass

### 7.1 扩展 `StorageRank` 枚举

**文件**：`tilelang_repo/3rdparty/tvm/src/runtime/thread_storage_scope.h`

```cpp
enum class StorageRank {
  // ... 现有 0-12 ...
  kMetalSimdGroup = 12,
  // --- Blackhole 硬件资源类型 ---
  /*! \brief Blackhole Circular Buffer — L1 FIFO queue resource */
  kBlackholeCB = 13,
  /*! \brief Blackhole Dst register-backed tile accumulator */
  kBlackholeAccumulator = 14,
};
```

对应 scope 字符串：

- `”blackhole.cb”` → `rank=kBlackholeCB, tag=””`
- `”blackhole.cb.input”` → `rank=kBlackholeCB, tag=”.input”`
- `”blackhole.cb.output”` → `rank=kBlackholeCB, tag=”.output”`
- `”blackhole.cb.intermed”` → `rank=kBlackholeCB, tag=”.intermed”`
- `”blackhole.acc”` → `rank=kBlackholeAccumulator, tag=””`

### 7.2 更新 `StorageScope::Create()` 和 `to_string()`

同一文件，在 `Create()` 的 else-if 链中新增：

```cpp
} else if (s.compare(0, 12, “blackhole.cb”) == 0) {
  r.rank = StorageRank::kBlackholeCB;
  r.tag = s.substr(12, std::string::npos);
} else if (s.compare(0, 13, “blackhole.acc”) == 0) {
  r.rank = StorageRank::kBlackholeAccumulator;
  r.tag = s.substr(13, std::string::npos);
}
```

`to_string()` 的 switch 中新增：

```cpp
case StorageRank::kBlackholeCB:
  return “blackhole.cb” + tag;
case StorageRank::kBlackholeAccumulator:
  return “blackhole.acc” + tag;
```

### 7.3 新 Pass：`BlackholeDeviceResourceCanonicalization`

**新建文件**：`tilelang_repo/src/transform/blackhole_device_resource_canonicalization.cc`

#### 7.3.1 Pass 职责

不是”隐藏”资源，而是 **把错误的 IR 类型替换为正确的 IR 类型**：

| 前端 scope | 硬件实际 | 规范化后 scope | 原因 |
|-----------|---------|--------------|------|
| `shared.dyn` | CB input (reader→compute) | `blackhole.cb.input` | CB 是 FIFO 队列，不是共享内存 |
| `shared.dyn` | CB output (compute→writer) | `blackhole.cb.output` | 同上 |
| `shared` | CB (静态) | `blackhole.cb` | 同上 |
| `local.fragment` | Dst 寄存器累加器 | `blackhole.acc` | Dst 是寄存器，不是可寻址内存 |
| `local` (经 LowerTileOp) | Dst 寄存器累加器 | `blackhole.acc` | 同上（LowerTileOp 可能已转 scope） |
| `global` / `””` | DRAM tensor | 不变 | 语义本身正确 |

#### 7.3.2 CB role 判定逻辑

根据数据流分析判断 CB 的 role：

- 被 DRAM→CB copy loop 写入的 CB → `blackhole.cb.input`（reader 生产，compute 消费）
- 被 CB→DRAM copy loop 读取的 CB → `blackhole.cb.output`（compute 生产，writer 消费）
- 既不直接连 DRAM 的 CB → `blackhole.cb.intermed`

可复用 `AnnotateBlackholeCopySemantics` 已产出的 `blackhole.copy_semantics` annotation 来辅助判定。

#### 7.3.3 累加器判定逻辑

- scope 为 `local.fragment` 的 buffer → `blackhole.acc`
- scope 为 `local` 且被 `tl.tileop.gemm_py` 的第 3 个参数引用（C buffer）→ `blackhole.acc`
- 其他 `local` buffer → 保持 `local`（普通设备 scratch）

#### 7.3.4 结构化资源声明（annotation）

在每个重写后的 `AllocateNode` 上附加 `blackhole.resource_decl` annotation：

**CB 资源声明**：
```json
{
  “resource_class”: “cb”,
  “role”: “input”,
  “original_scope”: “shared.dyn”,
  “data_format”: “bfloat16”,
  “tile_shape”: [32, 32],
  “producer_kernel”: “reader”,
  “consumer_kernel”: “compute”
}
```

**Accumulator 资源声明**：
```json
{
  “resource_class”: “accumulator”,
  “role”: “accumulator”,
  “original_scope”: “local.fragment”,
  “data_format”: “float32”,
  “tile_shape”: [32, 32],
  “compute_op”: “matmul”,
  “dst_slots”: 1
}
```

`compute_op` 字段为未来操作语义扩展预留——codegen 据此选择 FPU 还是 SFPU 路径、生成正确的 init/unpack/compute/pack 序列。

#### 7.3.5 Allocation 重定位

将被 `PlanAndUpdateBufferAllocationLocation` 提升到设备区域外的 device-private Allocate 移回 `thread_extent` AttrStmt 内部。

原因：`SplitHostDevice` 通过 `VarUseDefAnalyzer` 判断 undefined var。如果 Allocate 在设备区域外，其 buffer_var 在设备体内就是 undefined → 被提升为设备函数参数。

移入后：`SplitHostDevice` 在设备体内通过 `AllocateNode` 看到定义 → 不提升。

#### 7.3.6 `blackhole.resource_plan`

在 PrimFunc attrs 上记录完整资源分类，供下游 pass 消费：

```json
[
  {“name”: “A_shared”, “class”: “cb”, “role”: “input”, “scope”: “blackhole.cb.input”, “host_visible”: false},
  {“name”: “B_shared”, “class”: “cb”, “role”: “input”, “scope”: “blackhole.cb.input”, “host_visible”: false},
  {“name”: “C_local”,  “class”: “accumulator”, “role”: “accumulator”, “scope”: “blackhole.acc”, “host_visible”: false},
  {“name”: “A”,        “class”: “dram_tensor”, “role”: “input”, “scope”: “global”, “host_visible”: true},
  {“name”: “B”,        “class”: “dram_tensor”, “role”: “input”, “scope”: “global”, “host_visible”: true},
  {“name”: “C”,        “class”: “dram_tensor”, “role”: “output”, “scope”: “global”, “host_visible”: true}
]
```

### 7.4 为什么 generic pass 自然正确

规范化后的 IR 用 `kBlackholeCB` / `kBlackholeAccumulator` rank。generic pass 的匹配逻辑：

| Generic Pass | 匹配条件 | 是否命中新 rank | 结果 |
|-------------|---------|--------------|------|
| `MergeSharedMemoryAllocations` | `rank == kShared && tag == “.dyn”` | ❌ rank 不匹配 | 跳过（正确） |
| `LowerDeviceKernelLaunch` | `rank == kShared && tag == “.dyn”` | ❌ rank 不匹配 | 不计入 dyn_shmem（正确） |
| `ThreadSync(“shared.dyn”)` | 精确匹配 scope 字符串 | ❌ 字符串不同 | 跳过（正确） |
| `SplitHostDevice` | `VarUseDefAnalyzer` | Allocate 已在设备区域内 → var 已定义 | 不提升为参数（正确） |
| `MakePackedAPI` | 检查 host 函数 free vars | CB/acc 不在 host 侧 | 不报错（正确） |
| `StorageRewrite` | `scope.rank >= kWarp` | 会命中，但已永久排除于 Blackhole pipeline | 无影响 |
| `VerifyMemory` | 只检查 GPU device type | Blackhole 不是 GPU | 跳过 |

**这不是”隐藏”——generic pass 本来就不该处理 Blackhole 的 CB 和寄存器资源，因为它们语义上就不是 `shared` 或 `local`。**

## 8. 管线位置

```text
AnnotateBlackholeCopySemantics          ← 用原始 scope 判断 copy 方向
BlackholeDeviceResourceCanonicalization ← 【新增】scope 替换为正确类型 + allocation 重定位
AnnotateDeviceRegions                   ← 在正确的 IR 上标记设备区域
SplitHostDevice                         ← 只看到 host-visible 参数
AnnotateReadOnlyParams
MergeSharedMemoryAllocations            ← 无 kShared/.dyn → no-op
MakePackedAPI                           ← 只有 DRAM tensor 参数
LowerDeviceKernelLaunch                 ← 无 dyn_shmem
```

然后在 `blackhole_codegen()`（lower.py:209-235）中：

```text
LowerDeviceStorageAccessInfo            ← 需更新以识别新 scope
...
SplitBlackholeKernel                    ← 不改（基于 annotation 和 CallNode 名）
LowerBlackholeOps                       ← 改为用 kBlackholeCB rank 匹配
PlanBlackholeCB                         ← 不改（消费 attrs，不检查 scope）
AssignBlackholeCores                    ← 不改（消费 attrs）
```

## 9. Compute Pipeline 硬件分析

### 9.1 Tensix Core 内部计算流水线

```text
CB (input) ──→ [Unpacker/TRISC0] ──→ SRCA/SRCB ──→ [FPU or SFPU/TRISC1] ──→ DST Register ──→ [Packer/TRISC2] ──→ CB (output)
```

| 硬件单元 | 处理器 | 职责 | 输入来源 | 输出去向 |
|---------|--------|------|---------|---------|
| **Unpacker** | TRISC0 | 从 CB 读 tile → SRCA/SRCB 寄存器 | CB (L1 FIFO) | SRCA/SRCB 寄存器 |
| **FPU** (Matrix Engine) | TRISC1 | 矩阵运算 (matmul)，累加到 DST | SRCA/SRCB | DST Register |
| **SFPU** (Special FPU) | TRISC1 | 逐元素运算 (relu, exp, sqrt)，原地操作 DST | DST | DST (in-place) |
| **Packer** | TRISC2 | 从 DST 读 tile → 写入 output CB | DST Register | CB (L1 FIFO) |

### 9.2 DST 寄存器的关键特性

DST 是 **单一共享物理资源**，包含 16 个 tile slot（32×32 each）。有严格的锁同步协议：

```text
MATH (TRISC1):  tile_regs_acquire()  →  matmul/sfpu 操作  →  tile_regs_commit()
                     ↓ 独占锁                                    ↓ 释放独占
PACK (TRISC2):                           tile_regs_wait()  →  pack_tile()  →  tile_regs_release()
                                              ↑ 等待 MATH commit              ↑ 释放给下一轮 MATH
```

### 9.3 建模判断

| 层次 | 示例 | 是否需要 IR 建模 | 理由 |
|------|------|-----------------|------|
| **存储资源** | CB、DST 寄存器 | ✅ **需要 StorageRank** | 有独立身份和生命周期的硬件资源容器 |
| **操作语义** | matmul、relu、exp、reduce | ✅ **需要操作级 annotation/intrinsic** | 决定走 FPU 还是 SFPU，影响 Unpacker 配置 |
| **流水线协调** | acquire/commit/wait/release | ❌ **codegen 处理** | 固定时序协议，不需要 IR 决策 |
| **硬件配置** | math_fidelity、pack/unpack format | ❌ **codegen 推导** | 可从 dtype + operation 类型推导 |
| **SRCA/SRCB 源寄存器** | Unpacker 填充的临时寄存器 | ❌ **codegen 处理** | 完全临时，不跨任何边界 |

**结论**：`kBlackholeCB` + `kBlackholeAccumulator` 已覆盖所有需要 IR 存储级建模的硬件资源。FPU/SFPU/Packer/Unpacker 是操作属性，不是存储资源，通过操作级 intrinsic（当前 `gemm_py`，未来 `eltwise`/`reduction`）和 `blackhole.resource_decl` 的 `compute_op` 字段处理。

## 10. Pass / Codegen / Runtime 兼容性审查

### 10.1 不需要改的 Pass

| Pass | 原因 |
|------|------|
| `AnnotateBlackholeCopySemantics` | 在规范化 **之前** 运行，仍看到原始 `shared.dyn` scope |
| `SplitBlackholeKernel` | 基于 `blackhole.copy_semantics` annotation + `gemm_py` CallNode 名，不直接检查 buffer scope |
| `PlanBlackholeCB` | 消费 `blackhole.cb_requirements` attr（结构化数据），不直接检查 buffer scope |
| `AssignBlackholeCores` | 消费 attrs，不检查 scope |

### 10.2 需要更新的 Pass

| Pass | 当前检查 | 问题 | 修改方式 |
|------|---------|------|---------|
| `LowerBlackholeOps` | `scope.find(“shared”) == 0`（line 499） | `blackhole.cb.input` 不以 `shared` 开头 | 改为 `StorageScope::Create(scope).rank == kBlackholeCB` |
| `CodeGenBlackhole` (Allocate skip) | `scope == “shared” \|\| scope == “shared.dyn” \|\| scope == “shared.barrier”`（line 492） | `blackhole.cb.*` 不匹配 | 加入 `kBlackholeCB` rank 检查 |
| `CodeGenBlackhole` (PrintStorageScope) | `scope == “shared” \|\| scope == “shared.dyn”`（line 583） | 同上 | 加入 `kBlackholeCB` case |
| `CodeGenBlackhole` (acc handling) | `scope == “local”` → 生成 C 局部变量 | `blackhole.acc` 不匹配；Dst 是硬件管理的 | 新增 `kBlackholeAccumulator` → 跳过 Allocate |
| `LowerDeviceStorageAccessInfo` | tag 白名单（line 47） | `.input`/`.output` 等 tag 不在白名单 | 对 `kBlackholeCB`/`kBlackholeAccumulator` rank 直接跳过 |

### 10.3 硬件约束执行情况

| 硬件约束 | 当前执行位置 | 新 IR 后是否仍有效 |
|---------|------------|------------------|
| CB 总数 ≤ 64 | `PlanBlackholeCB` | ✅ 消费 attrs，不依赖 scope |
| L1 总量 ≤ 1,572,864 bytes | `PlanBlackholeCB` | ✅ 同上 |
| Input CB ID 0-15，Output CB ID 16-31 | `PlanBlackholeCB` + `LowerBlackholeOps` | ✅ 分配逻辑在 attrs 层 |
| CB page_size = single_tile_size | `LowerBlackholeOps` 提取 | ⚠️ 需更新 scope 检测（Step 4） |
| DST 寄存器 16 slot | codegen（tile index ≤ 15） | ✅ codegen 逻辑不变 |
| DST acquire/release 协议 | codegen 生成固定序列 | ✅ codegen 逻辑不变 |

### 10.4 Runtime 审查

Runtime 由两个组件组成：

- **`rt_mod_blackhole.cc`** — 从编译后的 PrimFunc 提取 `ExecutableSpec`
- **`blackhole_module.cc`** — 将 spec 实体化为 TT-Metal host objects 并执行

两者 **完全不检查 buffer scope 字符串**，工作在结构化 attrs 层（`blackhole.cb_configs`、`blackhole.runtime_args` 等）。

```text
Compile-time pass chain: IR scope → 结构化 attrs (blackhole.*)
Runtime:                 结构化 attrs → TT-Metal host objects → 执行
```

**Scope 变更完全封闭在编译时 pass 链内部，runtime 零改动。**

## 11. 分步实施计划

### Step 1: 扩展 StorageRank / StorageScope（IR 层）

**文件**：`3rdparty/tvm/src/runtime/thread_storage_scope.h`

1. `StorageRank` 枚举新增 `kBlackholeCB = 13`、`kBlackholeAccumulator = 14`
2. `StorageScope::Create()` 新增 `”blackhole.cb”` 和 `”blackhole.acc”` 前缀解析
3. `StorageScope::to_string()` 新增对应 case

### Step 2: 新 pass 实现

**新建文件**：`src/transform/blackhole_device_resource_canonicalization.cc`（~350 行）

核心结构：

```cpp
namespace tvm { namespace tl {

// Phase 1：资源分类
class BlackholeResourceClassifier : public StmtExprVisitor {
  // 遍历 AllocateNode + CallNode
  // 按 scope + 使用模式分类为 cb/accumulator/abi/local_scratch
  // 利用 blackhole.copy_semantics annotation 判断 CB role
  // 利用 gemm_py 调用的 arg 判断 accumulator
};

// Phase 2：Scope 重写 + Allocation 重定位
class BlackholeResourceCanonicalizer : public StmtExprMutator {
  // 替换 buffer_var 的 PointerType scope
  // 识别 thread_extent AttrStmt 边界
  // 收集其上方的 device-private Allocate
  // 剥离并移入 thread_extent 内部
  // 附加 blackhole.resource_decl annotation
};

// Phase 3：写 blackhole.resource_plan
// 在 PrimFunc attrs 上记录资源分类

Pass BlackholeDeviceResourceCanonicalization();
TVM_FFI_REGISTER_GLOBAL(“tl.transform.BlackholeDeviceResourceCanonicalization”);
}}
```

### Step 3: Python 注册 + 管线接入

**修改**：`tilelang/transform/__init__.py` — 新增 FFI 绑定

**修改**：`tilelang/engine/phase.py`（line 252-253 之间）

```python
mod = tilelang.transform.AnnotateBlackholeCopySemantics()(mod)
mod = tilelang.transform.BlackholeDeviceResourceCanonicalization()(mod)  # 新增
mod = tilelang.transform.AnnotateDeviceRegions()(mod)
```

### Step 4: 下游 pass 更新

- `codegen_blackhole.cc`：Allocate 跳过 + `PrintStorageScope` 用 rank 替代字符串匹配
- `lower_device_storage_access_info.cc`：新 rank 的 scope 白名单
- `lower_blackhole_ops.cc`：CB 识别改为 `rank == kBlackholeCB`
- `annotate_blackhole_copy_semantics.cc`：**不改**（在规范化之前运行）

### Step 5: 构建系统

`src/transform/CMakeLists.txt` 中添加新 `.cc` 文件。

## 12. 验证方式

### 12.1 回归测试

```bash
pytest testing/python/target/blackhole/test_blackhole_copy_pipeline.py -v
# 期望：14 passed, 1 skipped, 1 xfailed（与当前一致）
```

### 12.2 GEMM 解锁

```bash
pytest testing/python/target/blackhole/test_blackhole_gemm.py -v
# 期望：test_gemm_lower_basic 通过（三个 generic pass 错误全部消除）
```

### 12.3 结构验证（新增 test case）

1. 规范化后 IR 中无 `shared.dyn` / `local.fragment` scope buffer
2. `blackhole.resource_plan` attr 存在且分类正确
3. CB buffer scope 为 `blackhole.cb.*`，accumulator scope 为 `blackhole.acc`
4. 设备函数参数只包含 DRAM tensor（global scope），不包含 CB 或累加器

## 13. 风险与取舍

### 13.1 风险

- 如果新增 schema 只服务 GEMM，仍会退化成特判设计
- 如果语义扩展过重，短期接入成本会过高

### 13.2 取舍

本设计选择扩展 IR 类型系统（`StorageRank`），而不是：

- 在 generic pass 上打 Blackhole 特判补丁
- 用 scope 改名隐藏资源
- 直接大规模发明新 IR node

这与 WMMA/MMA/Metal/AMX 在 TVM 中的既有扩展实践完全同构，是最正交的解法。

## 14. 结论

Blackhole 当前的真正冲突不是 GEMM 本身，而是：

**TIR 的 StorageScope 类型系统把存储层级与资源语义混为一谈，而 Blackhole 的 CB（L1 FIFO 队列）和 Dst 累加器（寄存器文件）打破了 GPU 时代的 1:1 假设。**

正确解法是扩展 IR 类型系统，为 Blackhole 硬件资源引入正式的 `StorageRank`，然后通过规范化 pass 在 generic host/device 边界 pass 之前完成类型替换。

这使得：

1. generic pass 自然正确（rank 不匹配 → 跳过）
2. 下游 Blackhole pass 用类型安全的 rank 匹配（而非字符串前缀）
3. 后续新算子（eltwise、reduction、softmax）复用同一套 `kBlackholeCB` / `kBlackholeAccumulator` 类型，无需逐算子特判
