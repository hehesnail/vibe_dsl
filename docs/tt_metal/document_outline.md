# TT Metalium 完整参考文档大纲

> **文档版本**: 1.0
> **最后更新**: 2026-03-12
> **适用范围**: TT-Metalium v0.55+

---

## 文档结构总览

本文档库采用分层递进结构，从入门概念到高级优化，帮助开发者系统掌握 TT Metalium 编程。

```
TT Metalium 文档体系
│
├── 第一部分: 基础入门
│   ├── 第1章: 架构概述
│   ├── 第2章: 核心概念
│   └── 第3章: 编程示例
│
├── 第二部分: API 参考
│   ├── 第4章: Host API 参考
│   ├── 第5章: Circular Buffer API
│   ├── 第6章: Data Movement API
│   └── 第7章: Compute Kernel API
│
├── 第三部分: 进阶主题
│   ├── 第8章: 性能优化指南
│   └── 第9章: 调试工具
│
└── 附录
    ├── API 快速索引
    ├── 常见任务速查表
    └── 术语表
```

---

## 第一部分: 基础入门

### 第1章: 架构概述 (`section_architecture.md`)

**章节定位**: 入门必读，建立整体认知

**主要内容**:
- 1.1 什么是 TT Metalium - 软件栈定位与职责
- 1.2 支持的硬件代际 - Grayskull/Wormhole/Blackhole 对比
- 1.3 核心设计原则 - 裸机编程、三 Kernel 流水线、显式数据移动、Tile 计算
- 1.4 多芯片架构 - Mesh 拓扑与 TT-Fabric
- 1.5 子设备架构 - Sub-Device 概念与资源隔离
- 1.6 内存层次结构 - Host DRAM/Device DRAM/L1 SRAM
- 1.7 Network-on-Chip (NoC) - 寻址与通信机制

**前置知识**: 无
**阅读时长**: 30-45 分钟
**交叉引用**: 第2章(核心概念)

---

### 第2章: 核心概念 (`section_core_concepts.md`)

**章节定位**: 理论基础，深入理解硬件抽象

**主要内容**:
- 2.1 Tensix 核心架构 - 5 个 RISC-V Baby Core 详解
- 2.2 Host-Device 内存模型 - 地址空间隔离与传输机制
- 2.3 NoC 通信机制 - 物理/逻辑坐标、双通道设计、路由
- 2.4 Circular Buffer - CB 状态机与生产者-消费者模式
- 2.5 Tile 格式 - 32×32 数据块、Face/Channel 概念
- 2.6 Kernel 类型对比 - Data Movement/Compute/Ethernet Kernel
- 2.7 关键规则总结 - 内存访问、CB 使用、NoC 规则

**前置知识**: 第1章
**阅读时长**: 45-60 分钟
**交叉引用**: 第3章(编程示例)、第5-7章(API参考)

---

### 第3章: 编程示例 (`section_programming_examples.md`)

**章节定位**: 实践入门，通过代码学习编程模式

**主要内容**:
- 3.1 Hello World Kernel - 最基本的内核程序
- 3.2 信号量同步示例 - 多核协作同步机制
- 3.3 双缓冲模式示例 - 计算与传输重叠
- 3.4 分片张量操作示例 - Sharded Tensor 处理
- 3.5 矩阵块操作示例 - Matmul 块级实现
- 3.6 归约操作示例 - Reduce 操作实现
- 3.7 SFPU 完整操作示例 - 特殊函数单元使用
- 3.8 子设备管理示例 - Sub-Device 创建与管理
- 3.9 Metal Trace 完整示例 - 追踪与重放
- 3.10 多队列并行示例 - 并行命令队列

**前置知识**: 第1-2章
**阅读时长**: 60-90 分钟
**交叉引用**: 第4-7章(API参考)、第8章(性能优化)

---

## 第二部分: API 参考

### 第4章: Host API 参考 (`section_host_api.md`)

**章节定位**: Host 端编程手册，C++ API 完整参考

**主要内容**:
- 4.1 设备管理 API - CreateDevice/CloseDevice/GetNumAvailableDevices
- 4.2 Buffer 管理 API - CreateBuffer/DeallocateBuffer (Interleaved/Sharded)
- 4.3 Kernel 创建 API - CreateKernel/CreateKernelFromString
- 4.4 程序执行 API - EnqueueProgram/Finish/LaunchProgram/CompileProgram
- 4.5 运行时参数 API - SetRuntimeArgs/SetCommonRuntimeArgs/GetRuntimeArgs
- 4.6 子设备管理 API - create_sub_device_manager/load_sub_device_manager
- 4.7 Event/Semaphore API - CreateSemaphore/CreateGlobalSemaphore
- 4.8 直接内存访问 API - WriteToDeviceL1/ReadFromDeviceL1/WriteToDeviceDRAMChannel
- 4.9 配置结构体详解 - DataMovementConfig/ComputeConfig/EthernetConfig/BufferConfig
- 4.10 完整示例 - 矩阵乘法程序

**前置知识**: 第1-3章
**阅读时长**: 参考手册，按需查阅
**交叉引用**: 第5-7章(Device端API)

---

### 第5章: Circular Buffer API (`section_circular_buffer_api.md`)

**章节定位**: CB 机制完整参考

**主要内容**:
- 5.1 概述 - CB 核心概念与特性
- 5.2 Host 端 CB 创建配置 - CreateCircularBuffer/CircularBufferConfig
- 5.3 Host 端 CB 管理 - CB 生命周期与配置
- 5.4 Device 端 CB 操作 - cb_reserve_back/cb_push_back/cb_wait_front/cb_pop_front
- 5.5 CB 查询函数 - get_write_ptr/get_read_ptr/cb_get_num_tiles
- 5.6 CB 高级用法 - 双缓冲/多生产者-消费者
- 5.7 常见问题排查 - CB 配置错误与调试

**前置知识**: 第2章(核心概念)
**阅读时长**: 参考手册，按需查阅
**交叉引用**: 第4章(Host API)、第6章(Data Movement API)

---

### 第6章: Data Movement API (`section_data_movement_api.md`)

**章节定位**: 数据移动内核编程手册

**主要内容**:
- 6.1 核心 NOC 读写操作 - noc_async_read/noc_async_write
- 6.2 单包操作 - noc_async_read_one_packet
- 6.3 状态管理函数 - noc_async_read_barrier/noc_async_write_barrier
- 6.4 多播操作 - noc_async_write_multicast
- 6.5 页面操作 - noc_async_read_page/noc_async_write_page
- 6.6 分片操作 - Shard 数据移动
- 6.7 信号量操作 - noc_semaphore_wait/noc_semaphore_set/noc_semaphore_inc
- 6.8 屏障函数 - 各种 barrier 函数
- 6.9 地址函数 - get_noc_addr/get_dram_noc_addr
- 6.10 核心坐标与参数访问 - get_arg_val/get_common_arg_val
- 6.11 Circular Buffer 操作 - CB 相关函数
- 6.12 Tile 信息查询 - get_tile_size/get_tile_stride

**前置知识**: 第2章(核心概念)、第5章(CB API)
**阅读时长**: 参考手册，按需查阅
**交叉引用**: 第4章(Host API)、第7章(Compute API)

---

### 第7章: Compute Kernel API (`section_compute_api.md`)

**章节定位**: 计算内核编程手册

**主要内容**:
- 7.1 概述 - TRISC 核心与计算流水线
- 7.2 Tile 寄存器管理 - tile_regs_acquire/tile_regs_commit/tile_regs_wait/tile_regs_release
- 7.3 矩阵运算 API - matmul_tiles/mm_init
- 7.4 逐元素二元操作 - add_tiles/sub_tiles/mul_tiles
- 7.5 逐元素一元操作 (SFPU) - sigmoid/gelu/relu/exp/log/sqrt
- 7.6 归约操作 - reduce_tile/reduce_init
- 7.7 数据格式转换 - tilize/untilize
- 7.8 打包操作 - pack_tile/pack_init
- 7.9 SFPI 条件执行 - SFPU 条件指令
- 7.10 使用示例 - 完整计算内核示例

**前置知识**: 第2章(核心概念)、第5-6章
**阅读时长**: 参考手册，按需查阅
**交叉引用**: 第3章(编程示例)、第8章(性能优化)

---

## 第三部分: 进阶主题

### 第8章: 性能优化指南 (`section_performance_optimization.md`)

**章节定位**: 高级优化技术，性能调优手册

**主要内容**:
- 8.1 内存优化技巧 - SRAM/DRAM 层次优化、L1 布局策略、Bank 冲突避免
- 8.2 计算优化建议 - Math Fidelity 选择、SFPU 优化、向量化
- 8.3 数据传输优化 - 批量传输、双缓冲、DMA 优化
- 8.4 并行化策略 - 多核并行、流水线优化
- 8.5 Metal Trace 深度优化 - 追踪与重放优化
- 8.6 多芯片优化 - CCL 优化、Fabric 路由
- 8.7 性能调试方法论 - 瓶颈分析流程
- 8.8 优化检查清单 - 系统化优化步骤

**前置知识**: 第1-7章
**阅读时长**: 60-90 分钟
**交叉引用**: 第9章(调试工具)

---

### 第9章: 调试工具 (`section_debugging_tools.md`)

**章节定位**: 调试与性能分析工具使用指南

**主要内容**:
- 9.1 工具概览 - Tracy/Device Profiler/Watcher/DPRINT 对比
- 9.2 Tracy Profiler 详细配置 - 构建/C++/Python/行级分析/GUI
- 9.3 Device Profiler 使用指南 - Kernel 级计时分析
- 9.4 Watcher 调试技巧 - 运行时监控与死锁检测
- 9.5 DPRINT 调试方法 - Kernel printf 调试
- 9.6 系统性调试方法 - 调试流程与策略
- 9.7 高级调试技巧 - 复杂问题排查
- 9.8 性能瓶颈分析 - 性能问题定位
- 9.9 工具互斥性说明 - 工具使用限制

**前置知识**: 第1-8章
**阅读时长**: 45-60 分钟
**交叉引用**: 第8章(性能优化)

---

## 附录

### 附录 A: API 快速索引

#### A.1 Host API 快速索引

| API 类别 | 关键函数 | 所在章节 |
|---------|---------|---------|
| 设备管理 | CreateDevice, CloseDevice, GetNumAvailableDevices | 4.1 |
| Buffer 管理 | CreateBuffer, DeallocateBuffer | 4.2 |
| Kernel 创建 | CreateKernel, CreateKernelFromString | 4.3 |
| 程序执行 | EnqueueProgram, Finish, LaunchProgram | 4.4 |
| 运行时参数 | SetRuntimeArgs, SetCommonRuntimeArgs | 4.5 |
| 子设备管理 | create_sub_device_manager, load_sub_device_manager | 4.6 |
| 信号量 | CreateSemaphore, CreateGlobalSemaphore | 4.7 |
| 直接内存访问 | WriteToDeviceL1, ReadFromDeviceL1 | 4.8 |

#### A.2 Device API 快速索引

| API 类别 | 关键函数 | 所在章节 |
|---------|---------|---------|
| CB 操作 | cb_reserve_back, cb_push_back, cb_wait_front, cb_pop_front | 5.4 |
| NOC 读取 | noc_async_read, noc_async_read_barrier | 6.1 |
| NOC 写入 | noc_async_write, noc_async_write_barrier | 6.1 |
| 地址获取 | get_noc_addr, get_dram_noc_addr | 6.9 |
| 信号量 | noc_semaphore_wait, noc_semaphore_set | 6.7 |
| Tile 寄存器 | tile_regs_acquire, tile_regs_commit | 7.2 |
| 矩阵运算 | matmul_tiles, mm_init | 7.3 |
| 逐元素操作 | add_tiles, mul_tiles, sub_tiles | 7.4 |
| SFPU 操作 | sigmoid_tile, gelu_tile, relu_tile | 7.5 |
| 打包 | pack_tile, pack_init | 7.8 |

---

### 附录 B: 常见任务速查表

#### B.1 设备初始化速查

```cpp
// 基本设备创建
IDevice* device = CreateDevice(0);

// 多命令队列设备
IDevice* device = CreateDevice(0, 2);

// 设备关闭
CloseDevice(device);
```

#### B.2 Buffer 创建速查

```cpp
// DRAM Buffer
InterleavedBufferConfig dram_config{
    .device = device,
    .size = 1024 * 1024,
    .page_size = 2048,
    .buffer_type = BufferType::DRAM
};
auto dram_buffer = CreateBuffer(dram_config);

// L1 Buffer
InterleavedBufferConfig l1_config{
    .device = device,
    .size = 32768,
    .page_size = 2048,
    .buffer_type = BufferType::L1
};
auto l1_buffer = CreateBuffer(l1_config);
```

#### B.3 Kernel 创建速查

```cpp
// Reader Kernel (BRISC)
auto reader = CreateKernel(
    program,
    "reader.cpp",
    core,
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default
    }
);

// Writer Kernel (NCRISC)
auto writer = CreateKernel(
    program,
    "writer.cpp",
    core,
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::RISCV_1_default
    }
);

// Compute Kernel (TRISC)
auto compute = CreateKernel(
    program,
    "compute.cpp",
    core,
    ComputeConfig{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = false,
        .math_approx_mode = false
    }
);
```

#### B.4 CB 配置速查

```cpp
// 基本 CB 配置
CircularBufferConfig cb_config(
    num_pages * page_size,
    {{cb_index, tt::DataFormat::Float16_b}}
).set_page_size(cb_index, page_size);

auto cb = CreateCircularBuffer(program, core, cb_config);
```

#### B.5 程序执行速查

```cpp
// 非阻塞执行
EnqueueProgram(cq, program, false);
Finish(cq);  // 等待完成

// 阻塞执行
EnqueueProgram(cq, program, true);
```

#### B.6 调试环境变量速查

| 工具 | 环境变量 | 用途 |
|-----|---------|------|
| Tracy Profiler | `TT_METAL_TRACY=1` | 性能分析 |
| Device Profiler | `TT_METAL_DEVICE_PROFILER=1` | Kernel 计时 |
| Watcher | `TT_METAL_WATCHER=120` | 运行时监控 |
| DPRINT | `TT_METAL_DPRINT_CORES=all` | Kernel 打印 |

---

### 附录 C: 术语表

#### C.1 硬件术语

| 术语 | 英文 | 解释 |
|-----|------|------|
| Tensix 核心 | Tensix Core | Tenstorrent AI 加速器的计算单元，包含 5 个 RISC-V 核心 |
| BRISC | BRISC | RISC-V 0，负责数据读取 (Reader) |
| NCRISC | NCRISC | RISC-V 1，负责数据写入 (Writer) |
| TRISC | TRISC | RISC-V 2-4，负责计算 (Unpack/Math/Pack) |
| ERISC | ERISC | 以太网核心，负责芯片间通信 |
| NoC | Network-on-Chip | 片上网络，用于核心间通信 |
| L1 SRAM | L1 SRAM | 每个 Tensix 核心上的高速内存 (~1.5MB) |
| DRAM | DRAM | 设备全局内存 (GDDR6，12-32GB) |
| Tile | Tile | 32×32 元素的数据块，基本计算单位 |
| Face | Face | Tile 的子单元，通常为 16×16 元素 |

#### C.2 软件术语

| 术语 | 英文 | 解释 |
|-----|------|------|
| CB | Circular Buffer | 循环缓冲区，L1 SRAM 中的环形队列 |
| Kernel | Kernel | 在 Tensix 核心上运行的程序 |
| Program | Program | 包含多个 Kernel 的程序容器 |
| Host | Host | 运行主程序的 CPU 端 |
| Device | Device | Tenstorrent 硬件设备 |
| FPU | Float Point Unit | 浮点运算单元 |
| SFPU | Special Function Unit | 特殊函数单元，执行激活函数等 |
| CCL | Collective Communication Library | 集合通信库 |
| Mesh | Mesh | 多芯片网格拓扑 |
| Sub-Device | Sub-Device | 子设备，物理设备的逻辑分区 |

#### C.3 编程术语

| 术语 | 英文 | 解释 |
|-----|------|------|
| Data Movement Kernel | 数据移动内核 | 负责数据读取/写入的 Kernel |
| Compute Kernel | 计算内核 | 负责数学运算的 Kernel |
| Ethernet Kernel | 以太网内核 | 负责芯片间通信的 Kernel |
| Runtime Args | 运行时参数 | 从 Host 传递给 Kernel 的参数 |
| Compile Args | 编译时参数 | Kernel 编译时的常量参数 |
| Sharded | 分片 | 数据分布在多个核心上 |
| Interleaved | 交织 | 数据在 DRAM Bank 间交错存储 |
| Math Fidelity | 数学精度 | 计算精度设置 (LoFi/HiFi2/HiFi3/HiFi4) |
| SFPI | SFPU Instruction | SFPU 指令集 |

---

## 文档使用指南

### 学习路径建议

#### 路径一: 快速上手 (2-3 小时)
1. 阅读第1章(架构概述) - 了解整体架构
2. 阅读第2章(核心概念) - 理解基本概念
3. 完成第3章的 Hello World 示例
4. 参考第4章创建自己的第一个程序

#### 路径二: 系统学习 (1-2 周)
1. 完整阅读第1-3章
2. 完成第3章所有高优先级示例
3. 按需查阅第4-7章 API 参考
4. 阅读第8-9章了解优化和调试

#### 路径三: 深度开发 (持续)
1. 完整阅读所有章节
2. 深入研究第4-7章 API 细节
3. 实践第8章优化技术
4. 熟练使用第9章调试工具

### 文档交叉引用约定

- `-> 第X章`: 前置依赖，建议先阅读
- `<- 第X章`: 后续扩展，可深入阅读
- `参见 X.Y`: 引用具体章节
- `例如: ...`: 代码示例

### 版本兼容性说明

| 文档版本 | TT-Metalium 版本 | 状态 |
|---------|-----------------|------|
| 1.0 | v0.55+ | 活跃维护 |

---

## 文档维护信息

- **主文档**: TT_Metalium_Complete_Reference.md (待创建)
- **章节文档**: section_*.md (9个章节)
- **生成日期**: 2026-03-12
- **维护者**: Claude Code Assistant

---

*本文档大纲基于 TT Metalium 官方文档和源代码结构编写*
