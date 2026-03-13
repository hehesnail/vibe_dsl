# 1. 架构概述

## 1.1 什么是 TT Metalium

TT Metalium 是 Tenstorrent 开源的底层软件开发套件（SDK），提供对 Tenstorrent Tensix 硬件的直接访问能力。它是整个软件栈的基础层，位于高级框架和硬件之间，为开发者提供裸机级别的硬件控制能力。

**软件栈层次结构**:

```
┌─────────────────────────────────────────────────────────────┐
│  TT-Forge / TT-MLIR (编译器层)                               │
│  - 高层图优化和编译                                          │
│  - 自动算子融合和调度                                        │
├─────────────────────────────────────────────────────────────┤
│  TT-NN (高级 ML 操作库)                                      │
│  - PyTorch-like API                                         │
│  - 自动微分和训练支持                                        │
├─────────────────────────────────────────────────────────────┤
│  TT-Metalium ⬅ (本层)                                       │
│  - 底层 Kernel 编程                                          │
│  - 显式内存管理                                              │
│  - 硬件抽象层                                                │
├─────────────────────────────────────────────────────────────┤
│  TT-LLK (底层 Kernel 库)                                     │
│  - 芯片特定的微内核                                          │
│  - 计算和数据移动原语                                        │
├─────────────────────────────────────────────────────────────┤
│  Tenstorrent 硬件 (Tensix 核心)                              │
│  - Grayskull / Wormhole / Blackhole                         │
└─────────────────────────────────────────────────────────────┘
```

### 各层职责说明

| 层级 | 主要职责 | 目标用户 |
|------|---------|---------|
| **TT-Forge/TT-MLIR** | 图级优化、自动并行化、算子融合 | 框架开发者 |
| **TT-NN** | 高级 ML 操作、自动微分、模型构建 | ML 工程师 |
| **TT-Metalium** | Kernel 开发、显式内存管理、性能优化 | 系统程序员 |
| **TT-LLK** | 底层硬件原语、微架构优化 | 底层开发者 |
| **Hardware** | Tensix 核心执行、NoC 路由、SRAM/DRAM | - |

## 1.2 支持的硬件代际

| 代际 | 制程 | Tensix 核心数 | 每核心 SRAM | 外部内存 | 状态 |
|------|------|--------------|------------|---------|------|
| **Grayskull** (第1代) | 12nm | 96-120 | 1 MB | 8 GB LPDDR4 | v0.55 后停止支持 |
| **Wormhole** (第2代) | 12nm | 72-128 | 1.5 MB | 12-24 GB GDDR6 | 活跃支持 |
| **Blackhole** (第3代) | 6nm | ~140 | 1.5 MB | 32 GB GDDR6 | 活跃支持 |

### Grayskull (第1代)

- **制程**: 12nm
- **Tensix 核心**: 96-120 个
- **每核心 SRAM**: 1 MB
- **外部内存**: 8 GB LPDDR4
- **状态**: v0.55 后停止支持
- **定位**: 初代 AI 加速器，用于推理工作负载

### Wormhole (第2代)

- **制程**: 12nm
- **Tensix 核心**: 72-128 个（根据 SKU 不同）
- **每核心 SRAM**: 1.5 MB
- **外部内存**: 12-24 GB GDDR6
- **状态**: 活跃支持
- **互联**: 16× 100Gbps 以太网
- **定位**: 训练和推理均衡，支持多芯片扩展

### Blackhole (第3代)

- **制程**: 6nm
- **Tensix 核心**: ~140 个
- **每核心 SRAM**: 1.5 MB
- **外部内存**: 32 GB GDDR6
- **状态**: 活跃支持

**Blackhole 创新特性**:

1. **集成 RISC-V CPU 子系统**:
   - 24 个 RISC-V CPU 核心 (SiFive X280)
   - 可直接在芯片上运行 Linux 操作系统
   - 支持主机卸载和嵌入式部署

2. **大规模计算核心**:
   - 752 个小型 RISC-V 核心用于计算/数据移动
   - 增强的并行处理能力

3. **高速互联**:
   - 12× 400Gbps 以太网 (对比 Wormhole 的 16× 100Gbps)
   - 总计 4.8 Tbps 横向扩展带宽
   - 支持构建大规模集群 (Galaxy 系统)

4. **先进内存子系统**:
   - 更大容量的 GDDR6 (32 GB)
   - 更高内存带宽

## 1.3 核心设计原则

### 1.3.1 裸机编程模型

TT Metalium 采用"编程核心，而非线程"的哲学：

- **显式控制**: 无隐藏抽象，开发者直接控制每个核心的行为
- **无操作系统**: Kernel 直接在硬件上运行，无调度开销
- **确定性执行**: 精确控制执行时序，适合实时应用

```cpp
// 示例: 显式控制数据移动
void kernel_main() {
    // 直接操作硬件地址
    uint64_t noc_addr = get_noc_addr(x, y, addr);
    noc_async_read(noc_addr, l1_buffer, size);
    noc_async_read_barrier();  // 显式同步
}
```

### 1.3.2 三 Kernel 流水线

TT Metalium 使用三种协作 Kernel 实现高效流水线：

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Reader    │────▶│   Compute   │────▶│   Writer    │
│   Kernel    │ CB0 │   Kernel    │ CB1 │   Kernel    │
│  (BRISC)    │     │ (TRISC0-2)  │     │  (NCRISC)   │
└─────────────┘     └─────────────┘     └─────────────┘
      │                                          │
      └────────────── DRAM ↔ L1 ─────────────────┘
```

| Kernel 类型 | 处理器 | 职责 | 关键操作 |
|------------|--------|------|---------|
| **Reader** | BRISC (RISC-V 0) | 从 DRAM 读取数据到 L1 SRAM | `noc_async_read()` |
| **Compute** | TRISC0-2 (RISC-V 2-4) | 执行计算操作 | `matmul_tiles()`, `add_tiles()` |
| **Writer** | NCRISC (RISC-V 1) | 从 L1 SRAM 写回 DRAM | `noc_async_write()` |

**流水线优势**:
- 数据移动和计算重叠
- 最大化硬件利用率
- 显式控制避免缓存未命中

### 1.3.3 显式数据移动

TT Metalium 采用**零硬件缓存**设计：

- **无透明缓存**: 每个字节移动都由程序员显式控制
- **显式 DMA**: 通过 NoC 进行异步数据传输
- **程序员控制**: 决定何时、何地、如何移动数据

```cpp
// 显式数据移动示例
void kernel_main() {
    // 1. 预留 CB 空间
    cb_reserve_back(cb_id, num_tiles);

    // 2. 获取写入地址
    uint32_t l1_addr = get_write_ptr(cb_id);

    // 3. 发起异步读取
    noc_async_read(dram_addr, l1_addr, size);

    // 4. 显式等待完成
    noc_async_read_barrier();

    // 5. 提交数据到 CB
    cb_push_back(cb_id, num_tiles);
}
```

**设计理由**:
1. **可预测性**: 无缓存未命中的性能抖动
2. **效率**: 数据只移动一次，无冗余拷贝
3. **控制**: 程序员可优化特定工作负载的数据流

### 1.3.4 基于 Tile 的计算

TT Metalium 原生支持 32×32 Tile 操作：

**Tile 格式**:
- 默认大小: 32×32 元素
- 数据格式: Float16_b, Bfloat16, Float32, Int32 等
- 内存布局: 行优先，交织存储

**Tile 操作优势**:
- **向量化**: 一次操作 1024 个元素
- **数据局部性**: 适合 L1 SRAM 容量
- **硬件优化**: FPU/SFPU 针对 Tile 操作优化

```cpp
// Tile 操作示例
void MAIN {
    mm_init(cb_in0, cb_in1, cb_out);

    // 执行 32×32 矩阵乘法
    matmul_tiles(cb_in0, cb_in1, 0, 0, 0);

    // 逐元素操作
    add_tiles(cb_a, cb_b, 0, 0, 0);

    // 激活函数
    relu_tile_init();
    relu_tile(0, 0);
}
```

## 1.4 多芯片架构概述

### 1.4.1 Mesh 拓扑

TT Metalium 支持多芯片 Mesh 拓扑配置：

```
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│ Chip 0,0│──│ Chip 0,1│──│ Chip 0,2│──│ Chip 0,3│
└────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘
     │            │            │            │
┌────┴────┐  ┌────┴────┐  ┌────┴────┐  ┌────┴────┐
│ Chip 1,0│──│ Chip 1,1│──│ Chip 1,2│──│ Chip 1,3│
└─────────┘  └─────────┘  └─────────┘  └─────────┘
```

**支持的拓扑**:
- **2D 网格**: 行和列连接
- **3D 环形**: 支持 Galaxy 系统 (32+ 芯片)
- **自定义拓扑**: 通过 Mesh Graph Descriptor 定义

**MeshDevice API**:
```cpp
// 创建 2×4 芯片网格
auto mesh_device = distributed::MeshDevice::create(
    {2, 4},  // 行, 列
    {0, 1, 2, 3, 4, 5, 6, 7}  // 物理设备 ID
);

// 在网格上执行程序
EnqueueProgram(mesh_device->get_command_queue(), program, false);
```

### 1.4.2 TT-Fabric

TT-Fabric 是 Tenstorrent 的高速互联结构，提供芯片间通信能力：

**架构组件**:

```
┌─────────────────────────────────────────────────────────┐
│                     TT-Fabric 架构                       │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │   Chip A    │◄──►│   Chip B    │◄──►│   Chip C    │ │
│  │  (erisc)    │    │  (erisc)    │    │  (erisc)    │ │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘ │
│         │                  │                  │        │
│         └──────────────────┼──────────────────┘        │
│                            │                          │
│                    ┌───────┴───────┐                  │
│                    │  tt_fabric    │                  │
│                    │  (路由层)      │                  │
│                    └───────────────┘                  │
└─────────────────────────────────────────────────────────┘
```

**关键特性**:

1. **以太网 Kernel (erisc)**:
   - 专用 RISC-V 核心处理芯片间通信
   - 支持发送和接收模式
   - 透明多芯片执行

2. **集合通信库 (CCL)**:
   ```cpp
   // AllGather: 收集所有芯片的数据
   all_gather(input_tensor, output_tensor, dim, device_mesh);

   // ReduceScatter: 归约并分散数据
   reduce_scatter(input_tensor, output_tensor, dim, reduce_op, device_mesh);

   // AllReduce: 全归约操作
   all_reduce(input_tensor, math_op, num_links, memory_config);
   ```

3. **Fabric 路由**:
   - 自动路由选择
   - 支持多跳通信
   - 拓扑感知优化

**配置示例**:
```cpp
// 配置以太网 Kernel
EthernetConfig eth_config;
eth_config.eth_mode = Eth::SENDER;
eth_config.dataflow = EthDataflow::DISABLED;

auto eth_kernel = CreateKernel(
    program,
    "eth_kernel.cpp",
    eth_core,
    eth_config
);
```

## 1.5 子设备架构

TT Metalium 支持子设备 (Sub-Device) 概念，允许将单个物理设备划分为多个逻辑执行域：

### 1.5.1 子设备概念

```
┌─────────────────────────────────────────┐
│            物理设备 (Chip)               │
│  ┌─────────────────────────────────┐    │
│  │         子设备 0                 │    │
│  │  ┌─────┐ ┌─────┐ ┌─────┐       │    │
│  │  │Core0│ │Core1│ │Core2│ ...   │    │
│  │  └──┬──┘ └──┬──┘ └──┬──┘       │    │
│  │     └───────┴───────┘          │    │
│  │         专用 NoC 区域            │    │
│  └─────────────────────────────────┘    │
│  ┌─────────────────────────────────┐    │
│  │         子设备 1                 │    │
│  │  ┌─────┐ ┌─────┐ ┌─────┐       │    │
│  │  │Core8│ │Core9│ │Core10│ ...  │    │
│  │  └──┬──┘ └──┬──┘ └──┬──┘       │    │
│  │     └───────┴───────┘          │    │
│  │         专用 NoC 区域            │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

### 1.5.2 子设备用途

1. **多租户隔离**:
   - 不同工作负载运行在不同子设备
   - 资源隔离避免干扰

2. **细粒度调度**:
   - 独立命令队列
   - 并行执行不同程序

3. **资源管理**:
   - 核心分区
   - NoC 带宽分配
   - 内存隔离

### 1.5.3 子设备 API

```cpp
// 创建子设备配置
SubDeviceConfig sub_device_config{
    .core_ranges = {CoreRange({0, 0}, {3, 3})},  // 4×4 核心网格
    .noc_config = NocConfig::DEFAULT
};

// 创建子设备
SubDeviceId sub_device_id = device->create_sub_device(sub_device_config);

// 在子设备上创建 Buffer
auto buffer = CreateBuffer(config, sub_device_id);

// 同步特定子设备
Synchronize(device, cq_id, {sub_device_id});
```

### 1.5.4 子设备与 Buffer 管理

子设备影响 Buffer 的创建和访问：

```cpp
// 在特定子设备上分配 Buffer
InterleavedBufferConfig config{
    .device = device,
    .size = buffer_size,
    .page_size = page_size,
    .buffer_type = BufferType::DRAM
};

// 版本 3: 指定子设备 ID
std::shared_ptr<Buffer> buffer = CreateBuffer(config, sub_device_id);
```

**注意事项**:
- 子设备间的 Buffer 访问需要显式同步
- 每个子设备有自己的命令队列上下文
- 子设备配置影响 NoC 路由选择

## 1.6 内存层次结构

TT Metalium 的内存层次结构是显式管理的：

```
┌─────────────────────────────────────────┐
│           HOST DRAM                     │
│    (主机主内存，通过 PCIe 访问)           │
└─────────────────┬───────────────────────┘
                  │ PCIe 传输 (16-32 GB/s)
                  ▼
┌─────────────────────────────────────────┐
│         DEVICE DRAM (GDDR6)             │
│    (12-32 GB, 512 GB/s+ 带宽)            │
│    - 全局可访问存储                      │
│    - 通过 NoC 分块访问                   │
└─────────────────┬───────────────────────┘
                  │ NoC DMA (片上网络)
                  ▼
┌─────────────────────────────────────────┐
│      L1 SRAM (片上, 每核心)              │
│    (~1.5 MB 每 Tensix 核心)              │
│    - Circular Buffer (CB)               │
│    - 显式管理，不是缓存                  │
│    - 核心本地访问延迟最低                │
└─────────────────────────────────────────┘
```

**关键设计原则**:
- **零硬件缓存**: 无透明缓存机制
- **显式管理**: 程序员控制所有数据移动
- **分层访问**: 不同内存类型有不同访问方式和延迟

## 1.7 Network-on-Chip (NoC)

NoC 是 Tensix 核心间的通信网络：

### 1.7.1 NoC 寻址

使用**物理坐标**进行通信：

```cpp
// 地址格式: (x, y, local_addr) 元组
uint64_t noc_addr = get_noc_addr(x, y, addr_on_target_tile);

// 异步读写
noc_async_read(noc_addr, ptr_l1_buffer, size);
noc_async_write(ptr_l1_buffer, noc_addr, size);
```

### 1.7.2 NoC 规则

- RISC-V 核心只能直接访问私有内存和本地 L1 SRAM
- 访问 DRAM 或其他核心的 SRAM 需要 NoC DMA
- 栈变量不能作为 DMA 源/目标
- 支持多播操作高效广播数据

```cpp
// 多播示例
uint64_t multicast_addr = get_noc_multicast_addr(
    x_start, y_start, x_end, y_end, local_addr
);
noc_async_write_multicast(
    src_addr, multicast_addr, size, num_dests, linked
);
```

---

## 1.8 【技术报告补充】矩阵引擎架构详解

> **来源**: [Tech Report - Matrix Engine](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/matrix_engine/matrix_engine.md)

### 1.8.1 矩阵引擎操作类型

Wormhole 矩阵引擎支持以下操作：矩阵乘法、归约 (Reduce)、逐元素加法/减法/乘法、以及转置。

#### 矩阵乘法性能

Wormhole 矩阵引擎在**单周期**内可执行：
- **8×16 × 16×16 = 8×16** 矩阵乘法
- 相当于 **4096 次乘加操作 (MACs)** 每周期
- 在 1GHz 频率下达到 **4 TFLOPS** 每矩阵引擎

**输入尺寸要求**:
- 输入 A (in0): 最小 8×16
- 输入 B (in1): 最小 16×16
- 若输入小于最小尺寸，有效吞吐按比例下降

#### 归约操作性能

Wormhole 矩阵引擎在**单周期**内可执行：
- **16×16 归约** (Max/Average/Sum)
- 相当于 **512 次乘加操作** 每周期
- 在 1GHz 频率下达到 **0.512 TFLOPS** 每矩阵引擎

### 1.8.2 Math Fidelity 配置

Wormhole 使用 **5b × 7b 乘法器**，Math Fidelity 控制运算精度：

| Fidelity 级别 | SrcA 精度 | SrcB 精度 | 相对性能 | TFLOPS |
|--------------|-----------|-----------|----------|--------|
| **LoFi** | 1隐位 + 4 MSB | 1隐位 + 6 MSB | 1.0× | 4.0 |
| **HiFi2** | 1隐位 + 4 LSB | 1隐位 + 6 MSB | 0.5× | 2.0 |
| **HiFi3** | 1隐位 + 4 MSB | 1隐位 + 6 LSB | 0.33× | 1.33 |
| **HiFi4** | 1隐位 + 4 LSB | 1隐位 + 6 LSB | 0.25× | 1.0 |

```cpp
// Wormhole 计算核配置
struct WormholeComputeKernelConfig {
    MathFidelity math_fidelity = MathFidelity::LoFi;  // 默认 LoFi
    bool math_approx_mode = true;    // 近似模式 (SFPU 操作)
    bool fp32_dest_acc_en = false;   // FP32 累加
    bool packer_l1_acc = false;      // L1 累加模式
};
```

### 1.8.3 关键配置参数

**Math Approx Mode**:
- 某些 SFPU 操作支持近似模式
- 在性能和精度之间权衡
- 影响的操作包括：指数、GELU、平方根等

**FP32 Dest Acc (DST_ACCUM_MODE)**:
- Wormhole FPU 支持 Float16/Float16_b 或 Float32 累加
- 启用 `fp32_dest_acc_en` 时，目的寄存器容量减半
- Float16_b: 8 tiles; Float32: 4 tiles (使用 DstSync::Half)

**Packer L1 Accumulation**:
- 在 L1 内存中执行累加
- Packer 读取输入地址，与 Dest 值累加后写回
- 适用于高精度累加后转低精度输出

---

## 1.9 【技术报告补充】DRAM 带宽饱和策略

> **来源**: [Tech Report - Saturating DRAM Bandwidth](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Saturating_DRAM_bandwidth/Saturating_DRAM_bandwidth.md)

### 1.9.1 Wormhole DRAM 架构

Wormhole 架构特点：
- **12 个 DRAM bank** (Grayskull 有 8 个)
- 理论带宽：**288 GB/s** (@12Gbps) / **336 GB/s** (@14Gbps)
- 实际达成：**>92%** 带宽利用率

### 1.9.2 单 Bank 带宽饱和策略

**问题**: 屏障(barrier)导致带宽损失
- 等待当前块读取完成后才发送后续请求
- DRAM 在请求间隙处于空闲状态

**解决方案**: 事务 ID (tag) 流水线
```cpp
// 为每个数据块分配事务 ID
// Block 0 -> tag 1
// Block 1 -> tag 2
// ...
// 在第 N 迭代等待 tag N-1，同时发送 tag N 请求
// 确保始终有一个读取请求在飞行中
```

### 1.9.3 全 DRAM 带宽饱和策略

**NoC 拥塞避免**:
1. **每 bank 一个 reader**: 避免多个 reader 访问同一 bank
2. **就近放置 reader**: 将 reader 放置在距离 DRAM bank 最近的位置
3. **使用不同虚拟通道 (VC)**: 同一行的 reader 使用不同 NoC VC

**Wormhole harvested row 处理**:
- N150: 1 行 harvested
- N300: 2 行 harvested
- Reader 需要移位到可用坐标，避免路由重叠

### 1.9.4 Shard 张量在 DRAM 中的应用

**适用场景**:
- MatMul 中 in0 小高度大宽度 (如 32×1024)
- in1 大宽度 (如 1024×8192)
- DRAM bound 的计算

**优化策略**:
1. in0 按宽度 shard 到顶部行
2. in1 按宽度 shard 到 12 个 DRAM bank
3. 使用多播(multicast)将 in0 shard 广播到所有 worker core
4. 输入张量双缓冲，重叠数据移动与计算

**性能数据**:

| Test | DRAM BW @12GBps | DRAM BW @14GBps |
|------|-----------------|-----------------|
| DRAM spec speed | 288 GB/s | 336 GB/s |
| DRAM u-benchmark | 267 GB/s | 310 GB/s |
| Llama3-70 decode | 239-260 GB/s | 247-294 GB/s |
| Mixtral8x7b decode | 243-261 GB/s | 267-300 GB/s |

---

## 1.10 【技术报告补充】数据格式详解

> **来源**: [Tech Report - Data Formats](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/data_formats/data_formats.md)

### 1.10.1 Block Float 格式

Tenstorrent 支持 Block Float 数据格式以平衡精度和内存效率：

**BFloat16 → Block Float 转换**:
- 16 个数据共享一个指数
- 每个数据有 7 位尾数 (mantissa)
- BFP8 数据大小约为 BF16 的一半

**支持的 Block Float 格式**:
- **BFP8**: 16 数据共享指数，7 位尾数
- **BFP4**: 16 数据共享指数，3 位尾数
- **BFP2**: 16 数据共享指数，1 位尾数

### 1.10.2 尾数舍入规则

从高精度转换到低精度时：
1. 尾数舍入到最近值
2. 遇到平局 (tie) 时，舍入到最近的偶数值

**舍入过程**:
- Float32 到 BFP8: 23 位尾数舍入到 7 位
- 保留 6 位 + 1 个隐位 (hidden bit)
- 使用 guard bit 判断舍入方向

---

## 1.11 【技术报告补充】张量布局与内存

> **来源**: [Tech Report - Tensor Layouts](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/tensor_layouts/tensor_layouts.md)

### 1.11.1 Tensor Layout 类型

**Row-Major Layout**:
- 每行对应 buffer 中的一个 page
- 64×64 张量 = 64 个 pages

**Tiled Layout**:
- Pages 表示为 2D tiles (默认 32×32)
- 64×64 张量 = 4 个 tiles (每个 32×32)

### 1.11.2 Tile 内部结构

**Face (面)**:
- 每个 tile 分成 4 个 face (16×16)
- 矩阵引擎原生支持 16×16 乘法
- Tile 乘法分解为多个 face 乘法

**Face 内存布局**:
```
Tile (32×32):
┌──────────────┬──────────────┐
│   Face 0     │   Face 1     │  (rows 0-15)
│  (16×16)     │  (16×16)     │
├──────────────┼──────────────┤
│   Face 2     │   Face 3     │  (rows 16-31)
│  (16×16)     │  (16×16)     │
└──────────────┴──────────────┘

内存顺序: face0 -> face1 -> face2 -> face3 (行优先)
```

### 1.11.3 Memory Layout 类型

**Interleaved (交织)**:
- Pages 在多个 bank 间轮询分配
- 新张量总是从 bank 0 开始分配
- 可能导致张量间的内存碎片

**Sharded (分片)**:
- 张量被分成 shards 分布在指定 cores 上
- 提高数据局部性，减少通信开销
- 支持 Height/Width/Block sharding

---

*本文档基于 TT Metalium 官方文档和源代码结构编写*
*技术报告补充内容来源: tenstorrent/tt-metal/tech_reports/*
*参考: /root/dev/vibe_dsl/TT_Metal_Documentation_Summary.md*
*      /root/dev/vibe_dsl/docs/tt_metal/api_reference_scraped.md*
*      /root/dev/vibe_dsl/docs/tt_metal/github_repo_structure.md*
