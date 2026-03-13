# 2. 核心概念

## 2.1 Tensix 核心架构

### 2.1.1 5 个 RISC-V "Baby Core" 详解

每个 Tensix 核心包含 **5 个 RISC-V "Baby Core"**，这些核心共享同一个 L1 SRAM 和计算单元，但各自负责不同的任务：

| 核心 | 名称 | RISC-V ID | 用途 | 指令集特性 |
|------|------|-----------|------|-----------|
| **BRISC** | RISC-V 0 | RISCV_0 | 数据移动 (Reader) | 标准 RISC-V + NoC DMA 指令 |
| **NCRISC** | RISC-V 1 | RISCV_1 | 数据移动 (Writer) | 标准 RISC-V + NoC DMA 指令 |
| **TRISC0** | RISC-V 2 | RISCV_2 | 计算 - Unpack 解包 | 专用数据解包指令 |
| **TRISC1** | RISC-V 3 | RISCV_3 | 计算 - Math 数学 | FPU/SFPU 数学运算指令 |
| **TRISC2** | RISC-V 4 | RISCV_4 | 计算 - Pack 打包 | 专用数据打包指令 |

### 2.1.2 BRISC/NCRISC 职责分工

**BRISC (RISC-V 0) - Reader 核心：**
- 负责从 Device DRAM 或其他 Tensix 核心读取数据
- 执行 `noc_async_read()` 操作
- 通过 `cb_push_back()` 将数据推入 Circular Buffer
- 配置为 `DataMovementProcessor::RISCV_0`
- 使用 `NOC::RISCV_0_default` 作为默认 NoC 通道

**NCRISC (RISC-V 1) - Writer 核心：**
- 负责将数据写入 Device DRAM 或其他 Tensix 核心
- 执行 `noc_async_write()` 操作
- 通过 `cb_wait_front()` 等待输出 CB 就绪
- 配置为 `DataMovementProcessor::RISCV_1`
- 使用 `NOC::RISCV_1_default` 作为默认 NoC 通道

### 2.1.3 TRISC 计算流水线

**TRISC0 (Unpack)：**
- 从 Circular Buffer 读取原始数据
- 解包数据到计算引擎的输入寄存器
- 处理数据格式转换（如从 Float16 到内部格式）

**TRISC1 (Math)：**
- 执行实际的数学运算（矩阵乘法、逐元素操作等）
- 使用 FPU (Float Point Unit) 和 SFPU (Special Function Unit)
- 支持累加操作（如矩阵乘法的累加）

**TRISC2 (Pack)：**
- 将计算结果从输出寄存器打包
- 写入到输出 Circular Buffer
- 处理数据格式转换回存储格式

### 2.1.4 指令集特性

**数据移动核心 (BRISC/NCRISC)：**
- 标准 32 位 RISC-V 指令集（RV32I 基础指令集）
- 扩展 NoC DMA 指令：
  - `noc_async_read()` - 发起异步读取
  - `noc_async_write()` - 发起异步写入
  - `noc_async_read_barrier()` - 等待读取完成
  - `noc_async_write_barrier()` - 等待写入完成

**计算核心 (TRISC0-2)：**
- 专用指令集针对张量运算优化
- 支持 Tile 级别的并行操作
- 包含矩阵乘法、卷积、激活函数等专用指令

---

## 2.2 Host-Device 内存模型

### 2.2.1 内存层次结构详细说明

TT Metalium 采用显式内存管理模型，没有硬件缓存，所有数据移动都由程序员控制：

```
┌─────────────────────────────────────────────────────────────┐
│                    HOST DRAM                                │
│  • 系统主内存 (DDR4/DDR5)                                    │
│  • 容量：数十 GB                                             │
│  • Host 端代码运行于此                                       │
└───────────────────────┬─────────────────────────────────────┘
                        │ PCIe 传输 (16-32 GB/s)
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                  DEVICE DRAM (GDDR6)                        │
│  • Wormhole: 12-24 GB                                       │
│  • Blackhole: 32 GB                                         │
│  • 带宽：~512 GB/s                                          │
│  • 持久存储，核心间共享                                      │
└───────────────────────┬─────────────────────────────────────┘
                        │ NoC DMA (片上网络)
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              L1 SRAM (每 Tensix 核心)                       │
│  • Wormhole: ~1.5 MB 每核心                                 │
│  • Blackhole: ~1.5 MB 每核心                                │
│  • 带宽：~10+ TB/s (片上)                                   │
│  • 存储：Circular Buffer、临时数据                          │
│  • 注意：不是缓存 - 显式管理                                 │
└─────────────────────────────────────────────────────────────┘
```

### 2.2.2 数据传输路径

**Host DRAM → Device DRAM：**
```cpp
// Host 端代码
EnqueueWriteMeshBuffer(cq, device_buffer, host_data, blocking);
```
- 通过 PCIe 接口传输
- 由 Host 驱动程序管理
- 异步或同步操作

**Device DRAM → L1 SRAM：**
```cpp
// Device 端 Reader Kernel
noc_async_read_tile(tile_id, dram_accessor, l1_buffer_addr);
noc_async_read_barrier();
```
- 通过 NoC (Network on Chip) 传输
- BRISC 核心发起 DMA 操作
- 非阻塞异步操作，需要 barrier 同步

**L1 SRAM → Device DRAM：**
```cpp
// Device 端 Writer Kernel
noc_async_write_tile(tile_id, dram_accessor, l1_buffer_addr);
noc_async_write_barrier();
```
- 通过 NoC 传输
- NCRISC 核心发起 DMA 操作
- 非阻塞异步操作

### 2.2.3 地址空间隔离

**关键原则：Host 和 Device 有独立的地址空间**

```
┌─────────────────────────────────────────────────────────────┐
│  Host 地址空间                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Host DRAM 地址 (64-bit)                             │   │
│  │  • 应用程序虚拟地址                                  │   │
│  │  • 对 Device 不可见                                  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ PCIe 传输 (数据拷贝)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Device 地址空间                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Device DRAM 地址 (32-bit)                           │   │
│  │  • 通过 Buffer 对象管理                              │   │
│  │  • buffer->address() 获取                            │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  L1 SRAM 地址 (32-bit)                               │   │
│  │  • 每核心独立地址空间                                │   │
│  │  • 通过 get_write_ptr()/get_read_ptr() 获取          │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**重要约束：**
- Host 指针 Device 不可见，反之亦然
- 必须通过显式 API 进行数据传输
- Device 端只能访问本地 L1 SRAM 和通过 NoC 访问的 DRAM/其他 L1

### 2.2.4 PCIe 和 NoC 传输机制

**PCIe 传输：**
- 用于 Host 和 Device 之间的数据传输
- 由 `EnqueueWriteMeshBuffer()` 和 `EnqueueReadMeshBuffer()` 触发
- 需要大页内存 (Huge Pages) 支持以获得最佳性能
- 带宽：PCIe Gen4 x16 约 32 GB/s

**NoC 传输：**
- 用于 Device 内部的数据传输
- 二维网格拓扑结构
- 每个 Tensix 核心可以通过 NoC 访问：
  - 本地和其他核心的 L1 SRAM
  - Device DRAM
- 双通道设计 (NOC_0 和 NOC_1) 提高带宽

---

## 2.3 NoC 通信机制深化

### 2.3.1 物理坐标 vs 逻辑坐标

**物理坐标 (Physical Coordinates)：**
- 芯片上实际的硬件位置
- 格式：`(x, y)`，其中 x 和 y 是整数
- 用于 NoC 地址计算
- 示例：`CoreCoord{0, 0}` 表示左上角第一个核心

**逻辑坐标 (Logical Coordinates)：**
- 编程时使用的相对位置
- 可以映射到不同的物理位置
- 用于工作负载分配

```cpp
// 物理坐标示例
CoreCoord core(2, 3);  // 第 3 列，第 4 行的核心

// 核心网格范围
CoreRange cores({0, 0}, {7, 7});  // 8x8 核心网格
```

### 2.3.2 NOC_0 和 NOC_1 双通道设计

Tenstorrent 芯片采用双 NoC 通道设计以提高带宽和减少拥塞：

```
┌─────────────────────────────────────────────────────────────┐
│                    Tensix 核心网格                           │
│                                                             │
│  ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐               │
│  │ 0,0 │─────│ 1,0 │─────│ 2,0 │─────│ 3,0 │               │
│  │     │ NOC │     │ NOC │     │ NOC │     │               │
│  └─────┘  0  └─────┘  1  └─────┘  0  └─────┘               │
│     │           │           │           │                   │
│   NOC         NOC         NOC         NOC                   │
│     1           0           1           0                   │
│     │           │           │           │                   │
│  ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐               │
│  │ 0,1 │─────│ 1,1 │─────│ 2,1 │─────│ 3,1 │               │
│  │     │ NOC │     │ NOC │     │ NOC │     │               │
│  └─────┘  1  └─────┘  0  └─────┘  1  └─────┘               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**通道分配：**
- `NOC::RISCV_0_default` - BRISC 默认使用 NOC_0
- `NOC::RISCV_1_default` - NCRISC 默认使用 NOC_1
- 双通道可以同时传输，提高总带宽

### 2.3.3 地址编码方式 (get_noc_addr)

NoC 地址是一个 64 位值，编码了目标位置和本地地址：

```cpp
// 获取其他核心的 L1 地址
uint64_t noc_addr = get_noc_addr(x, y, local_addr);

// 获取 DRAM 地址
uint64_t dram_addr = get_dram_noc_addr(tile_id, dram_buffer, bank_base_address);

// 使用 NoC 地址进行传输
noc_async_read(noc_addr, l1_buffer, size);
noc_async_write(l1_buffer, noc_addr, size);
```

**地址格式：**
- 高 32 位：目标坐标和元数据
- 低 32 位：目标地址空间内的偏移

### 2.3.4 路由和拥塞避免

**NoC 路由特性：**
- 确定性路由算法
- XY 路由：先沿 X 方向，再沿 Y 方向
- 避免死锁的设计

**拥塞避免策略：**
1. **双通道分离**：读操作和写操作使用不同通道
2. **批量传输**：合并小传输为大块传输
3. **空间分布**：将工作负载分散到不同区域

```cpp
// 批量读取以减少 NoC 开销
const uint32_t batch_size = 8;
for (uint32_t b = 0; b < num_tiles; b += batch_size) {
    cb_reserve_back(cb_id, batch_size);
    uint32_t l1_addr = get_write_ptr(cb_id);

    // 批量读取
    for (uint32_t i = 0; i < batch_size; i++) {
        noc_async_read_tile(b + i, dram_buffer, l1_addr + i * tile_size);
    }
    noc_async_read_barrier();
    cb_push_back(cb_id, batch_size);
}
```

---

## 2.4 Circular Buffer (循环缓冲区)

### 2.4.1 CB 概念详解

Circular Buffer (CB) 是 Tensix 核心上 L1 SRAM 中的环形缓冲区，用于：
- Reader Kernel 和 Compute Kernel 之间的数据传递
- Compute Kernel 和 Writer Kernel 之间的数据传递
- 不同计算阶段之间的数据缓冲

**CB 特性：**
- 固定大小，创建时分配
- 循环使用，自动回绕
- 支持多生产者-多消费者模式
- 通过索引访问（CBIndex::c_0 到 CBIndex::c_31）

### 2.4.2 CB 状态机

```
┌─────────────────────────────────────────────────────────────┐
│                    CB 状态转换                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   EMPTY ──cb_reserve_back()──> RESERVED ──写入数据──>       │
│                                    │                        │
│                                    cb_push_back()           │
│                                    ▼                        │
│   POPPED <──cb_pop_front()── FILLED <──等待消费者──         │
│      │                                                      │
│      └──────────────────────────────────────────────┐       │
│                                                     │       │
│   状态说明：                                        │       │
│   • EMPTY: 缓冲区为空，可写入                       │       │
│   • RESERVED: 生产者预留了空间                      │       │
│   • FILLED: 数据已写入，等待消费                    │       │
│   • WAITED: 消费者正在读取                          │       │
│   • POPPED: 消费者释放空间                          │       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.4.3 CB API 详解

**Host 端创建 CB：**
```cpp
CircularBufferConfig cb_config(
    num_pages * page_size,                    // 总大小
    {{cb_index, data_format}}                 // 数据格式
).set_page_size(cb_index, page_size);         // 设置页大小

auto cb = CreateCircularBuffer(program, core, cb_config);
```

**Device 端生产者操作：**
```cpp
// 1. 预留写入空间
cb_reserve_back(cb_id, num_tiles);

// 2. 获取写入地址
uint32_t l1_addr = get_write_ptr(cb_id);

// 3. 写入数据（通过 NoC 或计算）
noc_async_read_tile(tile_id, accessor, l1_addr);
noc_async_read_barrier();

// 4. 标记数据就绪
cb_push_back(cb_id, num_tiles);
```

**Device 端消费者操作：**
```cpp
// 1. 等待数据可用
cb_wait_front(cb_id, num_tiles);

// 2. 获取读取地址
uint32_t l1_addr = get_read_ptr(cb_id);

// 3. 读取/处理数据
// ... 计算操作 ...

// 4. 释放空间
cb_pop_front(cb_id, num_tiles);
```

### 2.4.4 CB 配置最佳实践

**双缓冲配置：**
```cpp
// 双缓冲：允许 Reader 读取下一个 tile 的同时 Compute 处理当前 tile
uint32_t num_input_tiles = 2;  // 双缓冲

CircularBufferConfig cb_config(
    num_input_tiles * single_tile_size,
    {{cb_index, cb_data_format}}
).set_page_size(cb_index, single_tile_size);
```

**CB 大小计算：**
```cpp
// 计算 CB 大小
// tile_size = TILE_HEIGHT * TILE_WIDTH * element_size
// 对于 Float16 (2字节): 32 * 32 * 2 = 2048 bytes

uint32_t tile_size = 32 * 32 * 2;  // 2048 bytes
uint32_t cb_size = num_tiles * tile_size;
```

---

## 2.5 Tile 格式详细说明

### 2.5.1 Tile 基本概念

**Tile** 是 TT Metalium 的基本数据单位：
- 默认大小：**32×32 元素**
- 总元素数：1024 个
- 内存布局：针对张量运算优化的特殊格式

```
┌─────────────────────────────────────────────┐
│              Tile (32×32)                   │
│  ┌────┬────┬────┬────┬────┬────┬────┐      │
│  │ 0  │ 1  │ 2  │ 3  │ .. │    │ 31 │      │
│  ├────┼────┼────┼────┼────┼────┼────┤      │
│  │ 32 │ 33 │ 34 │ 35 │ .. │    │ 63 │      │
│  ├────┼────┼────┼────┼────┼────┼────┤      │
│  │ .. │    │    │    │    │    │    │      │
│  ├────┼────┼────┼────┼────┼────┼────┤      │
│  │    │    │    │    │    │    │1023│      │
│  └────┴────┴────┴────┴────┴────┴────┘      │
│                                             │
│  内存布局：行优先，但内部有交织优化          │
└─────────────────────────────────────────────┘
```

### 2.5.2 面 (Face) 概念

每个 Tile 可以划分为多个 **Face**（面）：
- 一个 Face 通常是 16×16 元素
- 一个 32×32 Tile 包含 4 个 Face
- Face 是计算引擎处理的基本单元

```
┌─────────────────────────────────────────────┐
│              Tile (32×32)                   │
│  ┌──────────────┬──────────────┐            │
│  │   Face 0     │   Face 1     │            │
│  │  (16×16)     │  (16×16)     │            │
│  │  rows 0-15   │  rows 0-15   │            │
│  │  cols 0-15   │  cols 16-31  │            │
│  ├──────────────┼──────────────┤            │
│  │   Face 2     │   Face 3     │            │
│  │  (16×16)     │  (16×16)     │            │
│  │  rows 16-31  │  rows 16-31  │            │
│  │  cols 0-15   │  cols 16-31  │            │
│  └──────────────┴──────────────┘            │
└─────────────────────────────────────────────┘
```

### 2.5.3 通道 (Channel) 概念

对于多通道数据（如 RGB 图像），Tile 可以包含多个通道：

```
┌─────────────────────────────────────────────┐
│         Multi-Channel Tile                  │
│                                             │
│  Channel 0 (R)    Channel 1 (G)   Channel 2 (B)│
│  ┌───────────┐   ┌───────────┐   ┌───────────┐ │
│  │  32×32    │   │  32×32    │   │  32×32    │ │
│  │  Red      │   │  Green    │   │  Blue     │ │
│  └───────────┘   └───────────┘   └───────────┘ │
│                                             │
│  总大小 = 3 × 32 × 32 × element_size        │
└─────────────────────────────────────────────┘
```

### 2.5.4 数据格式支持

| 数据格式 | 描述 | 元素大小 | 用途 |
|----------|------|----------|------|
| `Float32` | 单精度浮点 | 4 bytes | 高精度计算 |
| `Float16_b` / `Bfloat16` | 脑浮点 | 2 bytes | 深度学习（推荐）|
| `Float16` | 半精度浮点 | 2 bytes | 通用计算 |
| `Int32` | 32位整数 | 4 bytes | 索引、计数 |
| `Int16` | 16位整数 | 2 bytes | 量化计算 |
| `UInt32` | 无符号32位整数 | 4 bytes | 索引、掩码 |
| `UInt16` | 无符号16位整数 | 2 bytes | 量化计算 |
| `UInt8` | 8位无符号整数 | 1 byte | 低精度量化 |

### 2.5.5 Tile 格式转换

**线性数据 → Tile 格式 (Tilize)：**
```cpp
#include "compute_kernel_api/tilize.h"

tilize_init_short(cb_in);
tile_regs_acquire();
tilize_block(cb_in, num_tiles, cb_out);
tile_regs_commit();
```

**Tile 格式 → 线性数据 (Untilize)：**
```cpp
#include "compute_kernel_api/untilize.h"

untilize_init_short(cb_in);
untilize_block(cb_in, num_tiles, cb_out);
```

---

## 2.6 Kernel 类型对比表

### 2.6.1 三种 Kernel 类型概览

| 特性 | Data Movement Kernel | Compute Kernel | Ethernet Kernel |
|------|---------------------|----------------|-----------------|
| **运行核心** | BRISC / NCRISC | TRISC0-2 | ERISC |
| **处理器 ID** | RISCV_0 / RISCV_1 | FPU/SFPU | ERISC |
| **主要任务** | 数据读取/写入 | 数学计算 | 芯片间通信 |
| **NoC 使用** | NOC_0 / NOC_1 | 不直接使用 | 以太网链路 |
| **CB 操作** | 是（生产者/消费者） | 是（消费者/生产者） | 是 |
| **配置结构** | `DataMovementConfig` | `ComputeConfig` | `EthernetConfig` |

### 2.6.2 Data Movement Kernel 详细对比

| 特性 | Reader Kernel | Writer Kernel |
|------|---------------|---------------|
| **核心** | BRISC (RISCV_0) | NCRISC (RISCV_1) |
| **方向** | DRAM → L1 | L1 → DRAM |
| **NoC 通道** | NOC_0_default | NOC_1_default |
| **主要 API** | `noc_async_read()` | `noc_async_write()` |
| **CB 操作** | `cb_reserve_back()`, `cb_push_back()` | `cb_wait_front()`, `cb_pop_front()` |
| **角色** | 生产者 | 消费者 |

### 2.6.3 Compute Kernel 详细对比

| 特性 | Unpack (TRISC0) | Math (TRISC1) | Pack (TRISC2) |
|------|-----------------|---------------|---------------|
| **阶段** | 第 1 阶段 | 第 2 阶段 | 第 3 阶段 |
| **输入** | CB (原始数据) | 解包后的数据 | 计算结果 |
| **输出** | 解包后的数据 | 计算结果 | CB (打包数据) |
| **主要 API** | 解包指令 | `matmul_tiles()`, `add_tiles()` | `pack_tile()` |
| **初始化** | `unpack_init()` | `mm_init()` | `pack_init()` |

### 2.6.4 Kernel 协作模式

```
┌─────────────────────────────────────────────────────────────┐
│                  三 Kernel 协作流水线                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   DRAM          L1 SRAM              L1 SRAM         DRAM   │
│    │               │                    │              │    │
│    │  noc_async_   │   ┌──────────┐     │   noc_async_ │    │
│    │  read_tile()  └──>│   CB 0   │─────┘   write_tile()   │
│    │               │   │ (输入 A) │     │              │    │
│    │               │   └──────────┘     │              │    │
│    │               │         │          │              │    │
│    │    Reader     │    Unpack         │              │    │
│    │   (BRISC)     │   (TRISC0)        │              │    │
│    │               │         │          │              │    │
│    │               │   ┌──────────┐     │              │    │
│    │               └──>│   CB 1   │─────┘              │    │
│    │  noc_async_       │ (输入 B) │     │              │    │
│    │  read_tile()      └──────────┘     │              │    │
│    │               │         │          │              │    │
│    │               │       Math        │              │    │
│    │               │      (TRISC1)     │              │    │
│    │               │         │          │              │    │
│    │               │   ┌──────────┐     │              │    │
│    │               │   │  CB 16   │<────┘              │    │
│    │               │   │ (输出 C) │     │              │    │
│    │               │   └──────────┘     │              │    │
│    │               │         │          │              │    │
│    │               │       Pack        │              │    │
│    │               │      (TRISC2)     │              │    │
│    │               │         │          │              │    │
│    │               │   ┌──────────┐     │              │    │
│    │               └──>│  CB 16   │─────┐              │    │
│    │               │   │ (输出 C) │     │              │    │
│    │               │   └──────────┘     │              │    │
│    │               │                    │   noc_async_ │    │
│    │               │                    │   write_tile()    │
│    │               │                    │              │    │
│    │               │                    │    Writer    │    │
│    │               │                    │   (NCRISC)   │    │
│    │               │                    │              │    │
│                                                             │
│  数据流: DRAM → CB → Unpack → Math → Pack → CB → DRAM      │
│  并行性: Reader 读取下一个 tile 时，Compute 处理当前 tile   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.6.5 Kernel 配置代码示例

**Data Movement Kernel 配置：**
```cpp
// Reader Kernel (BRISC)
auto reader = CreateKernel(
    program,
    "reader.cpp",
    core,
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .compile_args = compile_time_args
    }
);

// Writer Kernel (NCRISC)
auto writer = CreateKernel(
    program,
    "writer.cpp",
    core,
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::RISCV_1_default,
        .compile_args = compile_time_args
    }
);
```

**Compute Kernel 配置：**
```cpp
auto compute = CreateKernel(
    program,
    "compute.cpp",
    core,
    ComputeConfig{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = false,
        .math_approx_mode = false,
        .compile_args = {Mt, Kt, Nt}
    }
);
```

**Ethernet Kernel 配置：**
```cpp
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

---

## 2.7 关键规则总结

### 2.7.1 内存访问规则

1. **RISC-V 核心只能直接访问：**
   - 私有内存（栈、寄存器）
   - 本地 L1 SRAM

2. **访问 DRAM 或其他核心 SRAM 需要：**
   - 使用 NoC DMA 操作
   - 通过 `noc_async_read()` / `noc_async_write()`

3. **栈变量限制：**
   - 栈变量不能作为 DMA 源/目标
   - 必须使用 L1 SRAM 中的缓冲区

### 2.7.2 CB 使用规则

1. **生产者流程：**
   - `cb_reserve_back()` → 写入数据 → `cb_push_back()`

2. **消费者流程：**
   - `cb_wait_front()` → 读取数据 → `cb_pop_front()`

3. **CB 索引范围：**
   - 有效索引：0 - 31
   - 常用：CB 0-15 用于输入，CB 16-31 用于输出

### 2.7.3 NoC 使用规则

1. **地址获取：**
   - 使用 `get_noc_addr()` 获取其他核心地址
   - 使用 `get_dram_noc_addr()` 获取 DRAM 地址

2. **同步要求：**
   - 异步操作后必须调用 barrier
   - `noc_async_read_barrier()` 等待读取完成
   - `noc_async_write_barrier()` 等待写入完成

3. **批量优化：**
   - 批量传输比多次小传输更高效
   - 使用双缓冲重叠计算和通信

---

## 2.8 【技术报告补充】张量分片 (Tensor Sharding) 详解

> **来源**: [Tech Report - Tensor Sharding](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/tensor_sharding/tensor_sharding.md)

### 2.8.1 分片基本概念

张量分片是将大张量分割成小的非重叠片段（shards），分布在多个内存 bank 上的技术。这是实现 Tenstorrent 设备高性能的关键。

**分片 vs 交织 (Interleaved)**:

| 特性 | Interleaved | Sharded |
|------|-------------|---------|
| 数据分布 | 轮询分布在所有 bank | 集中在指定 cores |
| 数据局部性 | 低 | 高 |
| 通信开销 | 高 | 低 |
| 适用场景 | 通用计算 | 大规模并行计算 |

### 2.8.2 2D 分片策略

#### Height Sharding (高度分片)

沿高度维度（除最后一维外的所有维度）分割张量：

```python
import ttnn
import torch

# 定义 core ranges (8 cores 在 2x4 网格)
core_ranges = ttnn.CoreRangeSet({
    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 3))
})

# 创建 height sharded tensor spec
tensor_spec = ttnn.TensorSpec(
    shape=(2, 128, 256),  # Batch=2, Height=128, Width=256
    dtype=ttnn.float32,
    layout=ttnn.TILE_LAYOUT,
    buffer_type=ttnn.BufferType.L1
).height_sharded(core_ranges)

# 每个 core 获得: 32 rows × 256 columns
# 总高度 128 * 2 / 8 cores = 32 rows per core
```

**适用场景**:
- 行级操作（如 LayerNorm 的行方向）
- 数据在行方向有大量并行性

#### Width Sharding (宽度分片)

沿宽度维度（最后一维）分割张量：

```python
# 定义 core ranges (4 cores 在 1x4 网格)
core_ranges = ttnn.CoreRangeSet({
    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))
})

# 创建 width sharded tensor spec
tensor_spec = ttnn.TensorSpec(
    shape=(1, 64, 512),  # Batch=1, Height=64, Width=512
    dtype=ttnn.float32,
    layout=ttnn.TILE_LAYOUT,
    buffer_type=ttnn.BufferType.L1
).width_sharded(core_ranges)

# 每个 core 获得: 64 rows × 128 columns
# 总宽度 512 / 4 cores = 128 columns per core
```

**适用场景**:
- 列级操作（如全连接层的权重矩阵）
- 数据在列方向有大量并行性

#### Block Sharding (块分片)

同时在高度和宽度维度分割，形成 2D 网格：

```python
# 定义 core ranges (16 cores 在 4x4 网格)
core_ranges = ttnn.CoreRangeSet({
    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))
})

# 创建 block sharded tensor spec
tensor_spec = ttnn.TensorSpec(
    shape=(1, 256, 256),  # Batch=1, Height=256, Width=256
    dtype=ttnn.float32,
    layout=ttnn.TILE_LAYOUT,
    buffer_type=ttnn.BufferType.L1
).block_sharded(core_ranges)

# 每个 core 获得: 64 rows × 64 columns
# 256/4 = 64 per dimension in 4x4 core grid
```

**内存映射**:
- core (0,0): 左上角块
- core (1,0): 右上角块
- core (0,1): 左下角块
- core (1,1): 右下角块

**适用场景**:
- 2D 卷积
- 矩阵乘法中的大矩阵分块
- 空间局部性要求高的操作

### 2.8.3 高级分片配置

使用 `MemoryConfig` 进行精细控制：

```python
import ttnn

# Height sharding 高级配置
memory_config = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.BufferType.L1,
    ttnn.ShardSpec(
        grid=ttnn.num_cores_to_corerangeset(
            target_num_cores=8,
            grid_size=[8, 7],
            row_wise=True,
        ),
        shard_shape=[64, 512],  # 每个 shard: 64 rows x 512 columns
        shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
    ),
)

# Block sharding 高级配置
memory_config = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ttnn.BufferType.L1,
    ttnn.ShardSpec(
        grid=ttnn.CoreRangeSet({
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))
        }),
        shard_shape=[64, 64],  # 每个 block: 64x64
        shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
    ),
)
```

### 2.8.4 分片最佳实践

1. **选择合适的分片策略**:
   - 行操作多 → Height Sharding
   - 列操作多 → Width Sharding
   - 2D 操作 → Block Sharding

2. **平衡计算负载**:
   - 确保每个 core 获得大致相等的工作量
   - 考虑 tile 对齐要求 (32 的倍数)

3. **内存对齐**:
   - Shard shape 应该与 tensor 对应维度匹配
   - Width sharding: H dimension of shard = H dimension of tensor
   - Height sharding: W dimension of shard = W dimension of tensor

---

## 2.9 【技术报告补充】Programming Examples 关键模式

> **来源**: [Tech Report - Programming Examples](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/prog_examples/prog_examples.md)
>
> 以及 `tt_metal/programming_examples/` 目录下的示例代码

### 2.9.1 Hello World Compute Kernel 模式

最基本的 compute kernel 执行模式：

```cpp
// Host 端代码
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>

int main() {
    // 1. 创建 MeshDevice (1x1)
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = mesh_device->mesh_command_queue();

    // 2. 创建 Program
    Program program = CreateProgram();
    CoreCoord core = {0, 0};

    // 3. 创建 Compute Kernel
    // TRISC0 (Unpack), TRISC1 (Math), TRISC2 (Pack) 协作执行
    KernelHandle compute_kernel = CreateKernel(
        program,
        "compute/void_compute_kernel.cpp",
        core,
        ComputeConfig{}
    );

    // 4. 提交执行
    distributed::MeshWorkload workload;
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    mesh_device->close();
    return 0;
}
```

**输出解读**:
```
0:(x=0,y=0):TR0: Hello, I am the UNPACK core running the compute kernel
0:(x=0,y=0):TR1: Hello, I am the MATH core running the compute kernel
0:(x=0,y=0):TR2: Hello, I am the PACK core running the compute kernel
```
- `0`: Device ID
- `(x=0,y=0)`: Tensix core 坐标
- `TR0/TR1/TR2`: 三个计算核心 (UNPACK/MATH/PACK)

### 2.9.2 Hello World Data Movement Kernel 模式

Data Movement Kernel 的基本模式：

```cpp
// Host 端代码
Program program = CreateProgram();

// 创建 Reader Kernel (BRISC - RISC-V 0)
KernelHandle data_movement_kernel_0 = CreateKernel(
    program,
    "dataflow/void_dataflow_kernel.cpp",
    core,
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default
    }
);

// 创建 Writer Kernel (NCRISC - RISC-V 1)
KernelHandle data_movement_kernel_1 = CreateKernel(
    program,
    "dataflow/void_dataflow_kernel.cpp",
    core,
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::RISCV_1_default
    }
);
```

**输出解读**:
```
0:(x=0,y=0):NC: My logical coordinates are 0,0
0:(x=0,y=0):NC: Hello, host, I am running on Data Movement core 1.
0:(x=0,y=0):BR: My logical coordinates are 0,0
0:(x=0,y=0):BR: Hello, host, I am running on Data Movement core 0.
```
- `NC`: Data Movement core 1 (NCRISC)
- `BR`: Data Movement core 0 (BRISC)

### 2.9.3 三 Kernel 协作完整示例

典型的 Reader → Compute → Writer 流水线：

```cpp
// Host 端配置
Program program = CreateProgram();

// 1. 创建输入/输出 Circular Buffers
CircularBufferConfig cb_in_config(
    num_tiles * tile_size,
    {{cb_in0, data_format}}
).set_page_size(cb_in0, tile_size);
auto cb_in = CreateCircularBuffer(program, core, cb_in_config);

CircularBufferConfig cb_out_config(
    num_tiles * tile_size,
    {{cb_out0, data_format}}
).set_page_size(cb_out0, tile_size);
auto cb_out = CreateCircularBuffer(program, core, cb_out_config);

// 2. 创建 Reader Kernel
auto reader = CreateKernel(
    program, "reader.cpp", core,
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default
    }
);

// 3. 创建 Writer Kernel
auto writer = CreateKernel(
    program, "writer.cpp", core,
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::RISCV_1_default
    }
);

// 4. 创建 Compute Kernel
auto compute = CreateKernel(
    program, "compute.cpp", core,
    ComputeConfig{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = false
    }
);

// 5. 设置 Runtime Args
SetRuntimeArgs(program, reader, core, {dram_addr, num_tiles});
SetRuntimeArgs(program, writer, core, {dram_addr, num_tiles});
SetRuntimeArgs(program, compute, core, {num_tiles});
```

### 2.9.4 常用编程模式

**模式1: 双缓冲 (Double Buffering)**
```cpp
// Reader 在 Compute 处理当前 tile 时读取下一个 tile
uint32_t num_input_tiles = 2;  // 双缓冲
CircularBufferConfig cb_config(
    num_input_tiles * single_tile_size,
    {{cb_index, data_format}}
).set_page_size(cb_index, single_tile_size);
```

**模式2: 多 Tile 批量处理**
```cpp
// 一次处理多个 tile 减少 overhead
const uint32_t batch_size = 8;
cb_reserve_back(cb_id, batch_size);
// ... 批量读取/处理 ...
cb_push_back(cb_id, batch_size);
```

**模式3: 分片并行**
```cpp
// 在多个 cores 上并行处理不同数据分片
CoreRange cores({0, 0}, {7, 7});  // 8x8 grid
for (int y = 0; y < 8; y++) {
    for (int x = 0; x < 8; x++) {
        CoreCoord core(x, y);
        SetRuntimeArgs(program, kernel, core,
            {shard_addr, shard_size, x, y});
    }
}
```
