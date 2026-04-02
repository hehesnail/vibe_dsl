# Blackhole 架构深度 Review 与落地行动计划

**日期**: 2026-03-20
**主题**: 针对 TileLang Blackhole 后端的架构诊断与具操作性的改造路径

---

## 1. 核心诊断：从 SIMT 到空间数据流的错位

经过对 `tilelang_repo` 与 `tt_metal_repo` 的源码比对，当前架构试图将面向 GPU 设计的 SIMT（单指令多线程）抽象，生硬地套用到 Tenstorrent (Blackhole) 的 Spatial Dataflow（空间数据流）架构上，这导致了严重的逻辑断层：

1.  **物理核心模型的硬冲突（单体 TIR vs 异构多处理器）**
    *   **现状**：代码将 `noc_async_read` (NOC 数据搬运) 和 `matmul_tiles` (TRISC 计算) 生成到了**同一个**顺序执行的 TIR 函数中。
    *   **问题**：Tensix 核心由 Reader (RISCV_0)、Writer (RISCV_1) 和 Compute (TRISC) 组成，必须独立编译和并发运行。生成单体 C++ 函数不仅无法通过 `ttc` 编译，更扼杀了双缓冲（Double Buffering）的通信掩盖能力。
2.  **内存寻址范式的错位（平坦内存 vs 环形队列）**
    *   **现状**：前端使用 `T.alloc_shared` 并依赖后端的 `StorageRewrite` 等通行 Pass。
    *   **问题**：Blackhole 依赖 Circular Buffer (CB)，不支持通过指针偏移 `[i, k]` 进行随机访问。通用的平坦内存优化会直接摧毁 CB 依赖的独立 ID 和 FIFO 机制。
3.  **同步机制的死锁隐患**
    *   **现状**：强行将 `noc_async_read_barrier()` 直接紧贴着 `noc_async_read` 放在最内层循环里。
    *   **问题**：NOC 搬运变成了纯同步操作，彻底阻塞了 Reader 核心。真正的反压应该交给 CB 的 `cb_reserve_back`，或者通过批量 Barrier 释放异步发射能力。
4.  **ABI 开销对弱处理器的降维打击**
    *   **现状**：保留了 TVM 庞大的 `PackedAPI` 机制，在设备端解析。
    *   **问题**：RISC-V 的 IRAM 极小，这不仅浪费宝贵的 L1 空间，更严重挤占了内核主循环的指令槽。

---

## 2. “外科手术式”改造路径 (Actionable Implementation Path)

为了打破上述困境，我们需要在代码库中实施以下 5 个明确的、可操作的改造步骤：

### 第一把刀：实现 `SplitBlackholeKernel` (AST 强制分离)
**目标文件**: `tilelang_repo/src/transform/split_blackhole_kernel.cc`
*   **具体动作**:
    1.  实现三个基于 `StmtExprMutator` 的类：`ReaderExtractor`、`ComputeExtractor` 和 `WriterExtractor`。
    2.  利用 AST 匹配将单体 TIR 强行裂变：
        *   **ReaderExtractor**: 拦截 `CallNode`，仅保留 `read_tile_to_cb`, `noc_async_read_barrier`, `cb_reserve_back`, `cb_push_back`。其余计算指令替换为 `SeqStmt({})`。化简去除空循环。
        *   **WriterExtractor**: 仅保留 `write_tile_from_cb`, `noc_async_write_barrier`, `cb_wait_front`, `cb_pop_front`。
        *   **ComputeExtractor**: 仅保留 `mm_init`, `matmul_tiles`, `tile_regs_*`, `pack_tile` 以及计算专用 CB 操作。
    3.  复制原始 `PrimFunc` 3份，分别打上 `blackhole.kernel_kind = "reader" / "compute" / "writer"` 的 Attribute。
    4.  修改 Pass 返回值为包含这 3 个函数的 `IRModule`。

### 第二把刀：改造 TVM 编译管道 (Python 胶水层)
**目标文件**: `tilelang_repo/tilelang/engine/lower.py`
*   **具体动作**:
    在 `blackhole_codegen` 函数中，在执行 `AssignBlackholeCores` 后，正式调用刚刚激活的 `tilelang.transform.SplitBlackholeKernel()`。确保下游的 C++ Build 函数接收到的是已被切分好的 `device_mod`。

### 第三把刀：分化代码生成 (`CodeGenBlackhole`)
**目标文件**: `tilelang_repo/src/target/rt_mod_blackhole.cc` & `codegen_blackhole.cc`
*   **具体动作**:
    1.  废除当前将所有 `PrimFunc` 塞进一个 `CodeGenBlackhole` 实例的做法。
    2.  在 `BuildTileLangBlackhole` 循环遍历 `mod->functions`：检查 `blackhole.kernel_kind`，为 Reader、Compute、Writer **各自实例化**一个 `tl::CodeGenBlackhole cg`。
    3.  **头文件隔离**：在 `codegen_blackhole.cc` 终结生成前，根据 kind 强制插入头文件（Reader/Writer 包含 `dataflow_api.h`，Compute 包含 `compute_kernel_api.h`）。

### 第四把刀：Direct Host Path 转正与零开销 ABI
**目标文件**: `tilelang_repo/CMakeLists.txt` & `tilelang_repo/src/target/blackhole_module_direct.cc`
*   **具体动作**:
    1.  在 CMake 中用 `blackhole_module_direct.cc` 替换掉依赖外部子进程的 `blackhole_module.cc`。
    2.  重写 `Execute`：废除旧的 `DLTensor` 解析。遍历 `inputs` 和 `outputs`，直接抓取张量所在的设备物理地址 `data_address_on_device`，并将其与 `scalar_args` 拼接为一维的 `std::vector<uint32_t> rt_args`。
    3.  直接调用 TT-Metal 的 `SetRuntimeArgs` 将 `rt_args` 平铺推入内核，由内核直接通过 `get_arg_val<uint32_t>(i)` 消费。

### 第五把刀：NOC Barrier 与地址生成器外提 (AST 重写)
**目标文件**: `tilelang_repo/src/transform/lower_blackhole_ops.cc` & `codegen_blackhole.cc`
*   **具体动作**:
    1.  **Barrier 提权**：在 `GenerateStagedCopyLoopSequence` 中，将 `noc_async_read_barrier` 的注入移到多 Tile 循环体的外部（或每 N 次循环发一次），而不是紧贴单次 Read。
    2.  **`InterleavedAddrGen` 外提**：修改 `CodeGenBlackhole` 的 AST 遍历逻辑，识别涉及 `DRAM <-> CB` 的最外层 `ForNode`，在循环进入前的 Prologue 区域统一打印出 `InterleavedAddrGen<true> src_gen = {...};` 的初始化代码，只在循环内保留 `get_noc_addr(tile_index, src_gen)`。

---

## 3. 基于源码审查的逐刀评估（2026-03-20 更新）

对照实际源码审查后的结论：

### 第一刀（SplitBlackholeKernel）：正确但时机不对

**结论**：Copy 阶段不需要，GEMM 阶段才需要。

- TT-Metal 的 `fused_dataflow` kernel 类型允许在 BRISC 上同时做 NOC 读写
- Copy 操作本质是 DRAM→L1→DRAM 的数据搬运，不涉及 TRISC 计算
- 历史 runner 实现已证明多 kernel（遍历 `spec.kernels`）的 host-side materialization 方式可行
- 推迟到 GEMM 阶段

### 第二刀（编译 Pipeline 改造）：已基本完成

- `lower.py` 和 `phase.py` 已有 Blackhole-specific 路径
- Pass 链 `LowerBlackholeOps → PlanBlackholeCB → AssignBlackholeCores` 已建立

### 第三刀（差异化代码生成）：Copy 已够用

- `codegen_blackhole.cc` 已区分 BRISC/NCRISC/TRISC 的 header 包含
- 当前只生成单个 kernel 文件，GEMM 需要生成 3 个独立 .cpp

### 第四刀（Direct Host Path）：最严重阻塞 → 已修正

原始 `blackhole_module_direct.cc` 存在三个基本功能缺失：
1. **完全没有 CreateCircularBuffer 调用** — 已修正
2. **Runtime args 构造不匹配 schema** — 已修正，改为按 `KernelArgSpec.kind` 逐项构造
3. **单核硬编码，无 work-packet 迭代** — 已修正，改为遍历 `work_packets`

修正方案：以历史 runner 实现为参考蓝本，将其逻辑合并到 `blackhole_module.cc` 的 `ExecuteDirect()` 方法中。`blackhole_module_direct.cc` 已合并后删除。

### 第五刀（NOC Barrier 优化）：正确但非阻塞

功能上正确，性能上有优化空间。可推迟到性能优化阶段。

---

## 4. 当前状态（2026-03-20）

- 第一刀：推迟到 GEMM 阶段
- 第二刀：已完成
- 第三刀：Copy 已够用，GEMM 时扩展
- 第四刀：**已修正** — direct path 已补全 CB 创建、runtime args、work-packet 迭代
- 第五刀：推迟到性能优化阶段

## 原始结论

通过上述外科手术式的改造，我们将使 TileLang 的编译链路从根本上契合 TT-Metal 的硬件编程范式。这不仅仅是”修复 Bug”，而是补齐了将 SIMT IR 转换为空间数据流架构最关键的拼图。
