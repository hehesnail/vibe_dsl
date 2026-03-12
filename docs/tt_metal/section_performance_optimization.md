# TT-Metalium 性能优化深度指南

> **文档版本**: 1.0
> **最后更新**: 2026-03-12
> **适用架构**: Wormhole / Blackhole

---

## 目录

1. [内存优化技巧](#1-内存优化技巧)
2. [计算优化建议](#2-计算优化建议)
3. [数据传输优化](#3-数据传输优化)
4. [并行化策略](#4-并行化策略)
5. [Metal Trace 深度优化](#5-metal-trace-深度优化)
6. [多芯片优化](#6-多芯片优化)
7. [性能调试方法论](#7-性能调试方法论)
8. [优化检查清单](#8-优化检查清单)

---

## 1. 内存优化技巧

### 1.1 SRAM/DRAM 层次结构优化

TT-Metalium 的内存层次结构具有显著的性能差异，理解并优化内存访问模式是性能调优的基础。

#### 内存层次结构性能对比

| 内存类型 | 带宽 | 延迟 | 容量 | 最佳用途 |
|----------|------|------|------|----------|
| **L1 SRAM (本地)** | 94 TB/s | ~1-2 周期 | ~1.5 MB/核心 | 活跃计算数据、CB |
| **L1 SRAM (邻居)** | 47 TB/s | ~5-10 周期 | - | Halo 数据交换 |
| **L1 SRAM (多播)** | 24 TB/s | - | - | 广播数据分发 |
| **DRAM (GDDR6)** | 512 GB/s | ~100-200 周期 | 12-32 GB | 大容量存储 |
| **以太网 (芯片间)** | 1 TB/s | ~1-5 µs | - | 多芯片扩展 |

#### 优化策略

**1. 数据局部性最大化**

```cpp
// 不良实践：频繁访问 DRAM
for (uint32_t i = 0; i < num_tiles; i++) {
    noc_async_read_tile(i, dram_buffer, l1_addr);  // 每次循环访问 DRAM
    noc_async_read_barrier();
    process_tile(l1_addr);
}

// 最佳实践：批量读取到 L1
const uint32_t batch_size = 8;
for (uint32_t b = 0; b < num_tiles; b += batch_size) {
    cb_reserve_back(cb_id, batch_size);
    uint32_t l1_addr = get_write_ptr(cb_id);

    // 批量读取
    for (uint32_t i = 0; i < batch_size && (b + i) < num_tiles; i++) {
        noc_async_read_tile(b + i, dram_buffer, l1_addr + i * tile_size);
    }
    noc_async_read_barrier();
    cb_push_back(cb_id, batch_size);
}
```

**2. 双缓冲策略**

双缓冲允许计算和数据传输重叠，隐藏内存延迟：

```cpp
// 双缓冲配置
constexpr uint32_t num_cb_tiles = 8;  // 4 tiles ping + 4 tiles pong
CircularBufferConfig cb_config(
    num_cb_tiles * tile_size,
    {{cb_index, data_format}}
).set_page_size(cb_index, tile_size);

// Kernel 中的双缓冲模式
void kernel_main() {
    uint32_t cb_ping = CBIndex::c_0;
    uint32_t cb_pong = CBIndex::c_1;

    // 预填充 ping buffer
    cb_reserve_back(cb_ping, 4);
    // ... 读取数据到 ping ...
    cb_push_back(cb_ping, 4);

    for (uint32_t i = 0; i < num_iterations; i++) {
        // 在当前 buffer 计算的同时，准备下一个 buffer
        cb_wait_front(cb_ping, 4);

        // 启动下一个数据的读取 (pong)
        cb_reserve_back(cb_pong, 4);
        // ... 异步读取到 pong ...
        cb_push_back(cb_pong, 4);

        // 处理 ping buffer (与上面的读取并行)
        process_tiles(cb_ping);
        cb_pop_front(cb_ping, 4);

        // 交换 ping/pong
        std::swap(cb_ping, cb_pong);
    }
}
```

### 1.2 L1 内存布局策略

#### L1 内存映射结构

每个 Tensix 核心的 L1 内存布局如下：

```
┌─────────────────────────────────────┐ 0x0
│           保留区域 (16KB)            │
├─────────────────────────────────────┤
│         程序代码/常量                 │
├─────────────────────────────────────┤
│         Circular Buffer 0            │
├─────────────────────────────────────┤
│         Circular Buffer 1            │
├─────────────────────────────────────┤
│              ...                     │
├─────────────────────────────────────┤
│         Circular Buffer N            │
├─────────────────────────────────────┤
│         栈空间 (默认 8KB)             │
├─────────────────────────────────────┤
│         调试/分析保留区               │
└─────────────────────────────────────┘ 0x18000 (1.5MB)
```

#### 布局优化原则

**1. CB 大小对齐**

```cpp
// 确保 CB 大小是 32 字节对齐的
uint32_t tile_size = 32 * 32 * element_size;
uint32_t aligned_tile_size = (tile_size + 31) & ~31;  // 32 字节对齐

// CB 总大小计算
uint32_t cb_size = num_tiles * aligned_tile_size;
```

**2. 减少 CB 数量**

每个 CB 都有元数据开销，合并可以合并的 CB：

```cpp
// 不良实践：为每个中间结果创建单独的 CB
CircularBufferConfig cb_temp1(...);
CircularBufferConfig cb_temp2(...);
CircularBufferConfig cb_temp3(...);

// 最佳实践：重用 CB
// 如果 temp1、temp2、temp3 的生命周期不重叠，可以共用同一个 CB
CircularBufferConfig cb_reusable(total_size, ...);
```

**3. 栈大小优化**

```cpp
// 在 kernel 编译参数中调整栈大小
// 如果 kernel 不需要大量栈空间，可以减少以腾出 L1 空间
constexpr uint32_t STACK_SIZE = 4096;  // 4KB 替代默认 8KB
```

### 1.3 DRAM Bank 冲突避免

#### DRAM Bank 组织

Wormhole/Blackhole DRAM 分为多个 bank，同时访问不同 bank 可以并行化：

| 架构 | DRAM Bank 数量 | Bank 大小 |
|------|---------------|-----------|
| Wormhole | 8 | ~1.5-3 GB/bank |
| Blackhole | 8 | ~4 GB/bank |

#### Bank 冲突避免策略

**1. 交错访问模式**

```cpp
// 不良实践：顺序访问同一 bank
for (uint32_t i = 0; i < num_tiles; i++) {
    // 如果 tile 0,1,2,3 都在同一 bank，会导致串行访问
    noc_async_read_tile(i, dram_buffer, l1_addr);
}

// 最佳实践：bank 间交错访问
constexpr uint32_t num_banks = 8;
for (uint32_t bank = 0; bank < num_banks; bank++) {
    for (uint32_t i = bank; i < num_tiles; i += num_banks) {
        // 访问不同 bank，可以并行
        noc_async_read_tile(i, dram_buffer, l1_addr);
    }
}
noc_async_read_barrier();
```

**2. 数据放置策略**

```cpp
// 使用交织缓冲区配置实现自动 bank 分布
InterleavedBufferConfig config{
    .device = device,
    .size = total_size,
    .page_size = tile_size,
    .buffer_type = BufferType::DRAM
};
// 交织布局会自动将页面分布到不同 bank
```

### 1.4 CB 大小计算最佳实践详解

#### CB 大小计算公式

```
CB 大小 = num_pages × page_size

其中:
- page_size = tile_size = 32 × 32 × element_size_bytes
- element_size_bytes:
  - Float32: 4 bytes
  - Float16/Bfloat16: 2 bytes
  - Bfloat8_b: 1 byte (带指数共享)
  - Bfloat4_b: 0.5 byte (带指数共享)
```

#### 不同数据格式的 CB 大小参考

| 数据格式 | 每 Tile 大小 | 4-Tile CB | 8-Tile CB | 16-Tile CB |
|----------|-------------|-----------|-----------|------------|
| Float32 | 4,096 bytes | 16 KB | 32 KB | 64 KB |
| Bfloat16 | 2,048 bytes | 8 KB | 16 KB | 32 KB |
| Bfloat8_b | 1,080 bytes* | ~4.2 KB | ~8.4 KB | ~16.9 KB |
| Bfloat4_b | 592 bytes* | ~2.3 KB | ~4.6 KB | ~9.3 KB |

*包含指数共享开销

#### CB 大小选择决策树

```
工作负载类型?
├── 计算密集型 (Matmul/Conv)
│   ├── 输入激活 CB: 2-4 tiles (双缓冲)
│   ├── 权重 CB: 1-2 tiles
│   └── 输出 CB: 2-4 tiles
├── 内存密集型 (Element-wise)
│   ├── 输入 CB: 4-8 tiles
│   └── 输出 CB: 4-8 tiles
└── 混合类型
    └── 平衡配置: 4 tiles 输入 / 2 tiles 权重 / 4 tiles 输出
```

#### 实际配置示例

```cpp
// Matmul 优化的 CB 配置
void configure_matmul_cbs(Program& program, CoreCoord core) {
    // 输入 A: 双缓冲 2 tiles
    uint32_t cb_a_size = 2 * 2048;  // 2 tiles × 2KB (bfloat16)
    CircularBufferConfig cb_a_config(
        cb_a_size,
        {{CBIndex::c_0, DataFormat::Float16_b}}
    ).set_page_size(CBIndex::c_0, 2048);
    CreateCircularBuffer(program, core, cb_a_config);

    // 输入 B: 单缓冲 1 tile (权重复用)
    uint32_t cb_b_size = 1 * 2048;
    CircularBufferConfig cb_b_config(
        cb_b_size,
        {{CBIndex::c_1, DataFormat::Float16_b}}
    ).set_page_size(CBIndex::c_1, 2048);
    CreateCircularBuffer(program, core, cb_b_config);

    // 输出: 双缓冲 2 tiles
    uint32_t cb_out_size = 2 * 2048;
    CircularBufferConfig cb_out_config(
        cb_out_size,
        {{CBIndex::c_16, DataFormat::Float16_b}}
    ).set_page_size(CBIndex::c_16, 2048);
    CreateCircularBuffer(program, core, cb_out_config);
}
```

### 1.5 内存池管理

#### 内存分配器工作原理

TT-Metalium 使用基于区域的内存分配器：

```
DRAM 内存池:
┌─────────────────────────────────────────┐
│  已分配区域 1 (Buffer A)                 │
├─────────────────────────────────────────┤
│  空闲区域                                │
├─────────────────────────────────────────┤
│  已分配区域 2 (Buffer B)                 │
├─────────────────────────────────────────┤
│  已分配区域 3 (Buffer C)                 │
├─────────────────────────────────────────┤
│  空闲区域                                │
└─────────────────────────────────────────┘
```

#### 内存碎片最小化

**1. 预分配策略**

```cpp
// 在程序开始时预分配所有需要的缓冲区
class MemoryPool {
public:
    void allocate_all_buffers(Device* device) {
        // 按大小降序分配，减少碎片
        input_buffer = CreateBuffer({device, largest_size, ...});
        weight_buffer = CreateBuffer({device, medium_size, ...});
        output_buffer = CreateBuffer({device, medium_size, ...});
        temp_buffer = CreateBuffer({device, small_size, ...});
    }

    void deallocate_all() {
        // 按相反顺序释放
        DeallocateBuffer(temp_buffer);
        DeallocateBuffer(output_buffer);
        DeallocateBuffer(weight_buffer);
        DeallocateBuffer(input_buffer);
    }
};
```

**2. 缓冲区重用**

```cpp
// 使用缓冲区视图而非重新分配
Buffer* activation_buffer = CreateBuffer({device, max_size, ...});

// 在不同层之间重用
for (auto& layer : model_layers) {
    // 重用同一个缓冲区，只更新元数据
    layer.set_input_buffer(activation_buffer);
    layer.execute();
}
```

---

## 2. 计算优化建议

### 2.1 Math Fidelity 选择指南

Math Fidelity 控制计算精度与性能的权衡，理解各等级的差异对优化至关重要。

#### Math Fidelity 等级详解

| 等级 | 描述 | 累加器精度 | 乘法精度 | 典型性能 | 适用场景 |
|------|------|-----------|---------|---------|----------|
| **LoFi** | 最低精度 | 16-bit | 8-bit | 100% (基准) | 推理、探索性训练 |
| **HiFi2** | 低精度 | 32-bit | 8-bit | ~85% | 对精度敏感的推理 |
| **HiFi3** | 中精度 | 32-bit | 16-bit | ~70% | 训练前向传播 |
| **HiFi4** | 最高精度 | 32-bit | 32-bit | ~50% | 训练反向传播、调试 |

#### 详细精度对比

```cpp
// Math Fidelity 配置示例
ComputeConfig{
    .math_fidelity = MathFidelity::HiFi2,  // 推荐起点
    .fp32_dest_acc_en = false,
    .math_approx_mode = true,
    .compile_args = {Mt, Kt, Nt}
};
```

**精度与性能权衡表：**

| 操作类型 | LoFi | HiFi2 | HiFi3 | HiFi4 |
|----------|------|-------|-------|-------|
| Matmul (BF16) | 1.0x | 0.85x | 0.70x | 0.50x |
| Matmul (FP32) | N/A | N/A | N/A | 1.0x |
| Element-wise | 1.0x | 0.95x | 0.90x | 0.80x |
| Activation | 1.0x | 0.90x | 0.85x | 0.75x |

#### 选择决策矩阵

```
应用场景?
├── 生产推理 (已验证模型)
│   └── 推荐: LoFi 或 HiFi2
│       └── 验证 PCC >= 0.99
├── 模型开发/调试
│   └── 推荐: HiFi4
│       └── 确保数值正确性优先
├── 训练前向传播
│   └── 推荐: HiFi2 或 HiFi3
│       └── 平衡精度与速度
└── 训练反向传播
    └── 推荐: HiFi3 或 HiFi4
        └── 梯度计算需要更高精度
```

#### 实际配置建议

```cpp
// 推理优化配置
ComputeConfig inference_config{
    .math_fidelity = MathFidelity::LoFi,
    .fp32_dest_acc_en = false,
    .math_approx_mode = true  // 使用近似数学函数
};

// 训练前向传播配置
ComputeConfig training_fwd_config{
    .math_fidelity = MathFidelity::HiFi2,
    .fp32_dest_acc_en = false,
    .math_approx_mode = true
};

// 训练反向传播配置
ComputeConfig training_bwd_config{
    .math_fidelity = MathFidelity::HiFi3,
    .fp32_dest_acc_en = true,   // FP32 累加
    .math_approx_mode = false   // 精确数学函数
};
```

### 2.2 FP32 累加模式使用场景

FP32 累加模式 (`fp32_dest_acc_en`) 在 dest 寄存器中使用 32 位浮点累加，提高数值稳定性。

#### 使用场景

**1. 大批量训练**

```cpp
// 大批量 Matmul 容易累积数值误差
// 使用 FP32 累加提高稳定性
ComputeConfig{
    .math_fidelity = MathFidelity::HiFi2,
    .fp32_dest_acc_en = true,  // 启用 FP32 累加
    .compile_args = {batch_size, seq_len, hidden_dim}
};
```

**2. 深度网络层**

```cpp
// 深层 Transformer 的注意力计算
// 多层 softmax 和 matmul 累积误差
void configure_deep_attention() {
    ComputeConfig config{
        .math_fidelity = MathFidelity::HiFi3,
        .fp32_dest_acc_en = true  // 深层网络需要更高精度累加
    };
}
```

**3. 混合精度训练**

```cpp
// 混合精度训练中的 master weights 更新
// 即使激活使用 BF16，权重更新使用 FP32 累加
ComputeConfig master_weight_config{
    .math_fidelity = MathFidelity::HiFi4,
    .fp32_dest_acc_en = true
};
```

#### 性能影响

| 配置 | 相对性能 | 内存开销 |
|------|---------|---------|
| BF16 累加 | 100% | 基准 |
| FP32 累加 | 70-80% | +50% dest 寄存器 |

### 2.3 fast_and_approx 参数影响

`math_approx_mode` (fast_and_approx) 启用近似数学函数，牺牲精度换取速度。

#### 近似模式影响的操作

| 操作 | 精确模式 | 近似模式 | 速度提升 |
|------|---------|---------|---------|
| `exp_tile` | 泰勒展开 | 查找表 | ~2x |
| `log_tile` | 迭代算法 | 查找表 | ~2x |
| `sqrt_tile` | 牛顿迭代 | 快速逆平方根 | ~1.5x |
| `sin/cos` | CORDIC | 查找表 | ~3x |
| `gelu_tile` | 精确公式 | 近似公式 | ~1.3x |
| `erf_tile` | 数值积分 | 多项式近似 | ~2x |

#### 精度验证方法

```cpp
// 在 kernel 中验证近似精度
void verify_approx_precision() {
    // 使用 DPRINT 输出中间结果进行验证
    DPRINT << "Exact exp(1.0): " << exp_tile_exact(1.0) << ENDL();
    DPRINT << "Approx exp(1.0): " << exp_tile_approx(1.0) << ENDL();

    // 计算相对误差
    // 相对误差 < 0.1% 通常可接受
}
```

#### 推荐配置

```cpp
// 推理 - 使用近似
ComputeConfig inference_fast{
    .math_fidelity = MathFidelity::LoFi,
    .math_approx_mode = true  // 快速近似
};

// 训练 - 避免近似
ComputeConfig training_accurate{
    .math_fidelity = MathFidelity::HiFi3,
    .math_approx_mode = false  // 精确计算
};

// 验证后切换
// 先用精确模式验证 PCC >= 0.999
// 再切换到近似模式验证 PCC >= 0.99
```

### 2.4 FPU vs SFPU 选择

TT-Metalium 有两个计算引擎：FPU (矩阵引擎) 和 SFPU (向量引擎)。

#### 引擎特性对比

| 特性 | FPU (矩阵引擎) | SFPU (向量引擎) |
|------|---------------|----------------|
| **最佳操作** | Matmul, Conv | Element-wise, Activation |
| **数据格式** | Bfloat16, Block FP | Float32, Bfloat16, Int32 |
| **峰值性能** | 373 TLOPs (BF16) | 94 TLOPs (BF16) |
| **精度控制** | 通过 Math Fidelity | 通过 SFPI 指令 |
| **编程模型** | 高级 API | SFPI 向量指令 |

#### 选择指南

**使用 FPU 的场景：**

```cpp
// 矩阵乘法 - 必须使用 FPU
#include "compute_kernel_api/matmul.h"

void MAIN {
    mm_init(cb_in0, cb_in1, cb_out);
    matmul_tiles(cb_in0, cb_in1, 0, 0, 0, false);
}
```

**使用 SFPU 的场景：**

```cpp
// 自定义激活函数 - 使用 SFPU
#include "compute_kernel_api/eltwise_unary/sfpu.h"

void MAIN {
    // 使用 SFPI 编写自定义向量操作
    vFloat val = dst_reg[0];
    vFloat result = sfpu_exp(val);  // SFPU 指数
    dst_reg[0] = result;
}
```

**混合使用：**

```cpp
// Matmul + Activation 融合
void MAIN {
    // FPU 执行 Matmul
    mm_init(cb_in0, cb_in1, cb_out);
    matmul_tiles(cb_in0, cb_in1, 0, 0, 0, false);

    // SFPU 执行 GELU 激活
    gelu_tile_init();
    gelu_tile(cb_out, cb_out, 0, 0);
}
```

### 2.5 数据格式优化

#### 数据格式性能对比

| 格式 | 每元素位数 | 内存带宽 | 计算性能 | 精度 | 推荐场景 |
|------|-----------|---------|---------|------|----------|
| **Float32** | 32 | 1x | 0.5x | 最高 | 训练、调试 |
| **Bfloat16** | 16 | 2x | 1x | 高 | 通用推理/训练 |
| **Bfloat8_b** | 8* | 4x | 1x | 中 | 权重、激活 |
| **Bfloat4_b** | 4* | 8x | 1x | 低 | 量化推理 |
| **Int32** | 32 | 1x | 0.8x | 精确整数 | 索引、计数 |
| **Int8** | 8 | 4x | 1x | 低 | 量化模型 |

*包含指数共享块

#### 格式选择策略

```cpp
// 混合精度配置示例
void configure_mixed_precision() {
    // 激活: Bfloat16 (精度敏感)
    DataFormat activation_format = DataFormat::Float16_b;

    // 权重: Bfloat8_b (内存节省)
    DataFormat weight_format = DataFormat::Bfp8_b;

    // 输出: Bfloat16
    DataFormat output_format = DataFormat::Float16_b;

    // 在 ComputeConfig 中配置
    ComputeConfig{
        .math_fidelity = MathFidelity::HiFi2,
        // 输入格式通过 CB 配置
    };
}
```

#### PCC 验证流程

```python
import torch
import ttnn

# 验证数据格式转换的精度
def verify_format_conversion():
    # PyTorch 参考
    torch_input = torch.randn(1, 32, 32, 32)
    torch_result = torch_model(torch_input)

    # TT-Metalium 实现
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16)
    tt_result = tt_model(tt_input)

    # 计算 PCC
    pcc = calculate_pcc(torch_result, ttnn.to_torch(tt_result))

    assert pcc >= 0.99, f"PCC too low: {pcc}"
    print(f"PCC: {pcc} - Format conversion acceptable")
```

---

## 3. 数据传输优化

### 3.1 NOC 路由优化 (NOC_0 vs NOC_1)

#### NOC 架构

TT-Metalium 有两个独立的 NoC (Network-on-Chip)：

```
┌─────────────────────────────────────┐
│           Tensix 核心网格            │
│  ┌───┬───┬───┬───┬───┬───┬───┬───┐  │
│  │   ║   ║   ║   ║   ║   ║   ║   │  │  NOC_0 (垂直)
│  ├───╫───╫───╫───╫───╫───╫───╫───┤  │  ║ = 垂直通道
│  │   ║   ║   ║   ║   ║   ║   ║   │  │
│  ├───╫───╫───╫───╫───╫───╫───╫───┤  │
│  │   ║   ║   ║   ║   ║   ║   ║   │  │  NOC_1 (水平)
│  └───┴───┴───┴───┴───┴───┴───┴───┘  │  ═ = 水平通道
│    ═══╦═══╦═══╦═══╦═══╦═══╦═══      │
└─────────────────────────────────────┘
```

#### 路由选择策略

| 场景 | 推荐 NOC | 原因 |
|------|---------|------|
| Reader Kernel (DRAM→L1) | NOC_0 | 默认读取路径 |
| Writer Kernel (L1→DRAM) | NOC_1 | 默认写入路径 |
| 核心间通信 (水平) | NOC_1 | 水平路由优化 |
| 核心间通信 (垂直) | NOC_0 | 垂直路由优化 |
| 高带宽需求 | 两者都用 | 负载均衡 |

#### 配置示例

```cpp
// Reader Kernel - 使用 NOC_0
auto reader = CreateKernel(
    program,
    "reader.cpp",
    core,
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,  // BRISC
        .noc = NOC::RISCV_0_default,  // NOC_0
        .compile_args = reader_args
    }
);

// Writer Kernel - 使用 NOC_1
auto writer = CreateKernel(
    program,
    "writer.cpp",
    core,
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_1,  // NCRISC
        .noc = NOC::RISCV_1_default,  // NOC_1
        .compile_args = writer_args
    }
);
```

### 3.2 多播策略和最佳实践

#### 多播机制

多播允许单个写操作将数据分发到多个目标核心：

```cpp
// 多播到矩形区域的所有核心
noc_async_write_multicast(
    src_addr,                    // 源 L1 地址
    dst_noc_addr_multicast,      // 目标多播地址
    size,                        // 传输大小
    num_dests,                   // 目标核心数
    multicast_flags              // 多播标志
);
```

#### 多播模式

| 模式 | 描述 | 使用场景 |
|------|------|----------|
| **矩形多播** | 多播到核心矩形区域 | 权重广播、参数分发 |
| **线性多播** | 多播到一行或一列 | Halo 交换、行/列广播 |
| **环形多播** | 多播到环形拓扑 | AllGather、ReduceScatter |

#### 最佳实践

```cpp
// 权重广播示例 - 多播到所有计算核心
void broadcast_weights_to_grid(
    uint32_t weight_addr,
    CoreRange compute_grid,
    uint32_t weight_size
) {
    // 计算多播目标地址
    uint64_t multicast_addr = get_noc_multicast_addr(
        compute_grid.start.x,
        compute_grid.start.y,
        compute_grid.end.x,
        compute_grid.end.y,
        weight_addr
    );

    // 计算目标核心数
    uint32_t num_cores = compute_grid.grid_size();

    // 执行多播
    noc_async_write_multicast(
        weight_addr,
        multicast_addr,
        weight_size,
        num_cores,
        0  // 标志
    );
    noc_async_write_barrier();
}
```

#### 多播性能对比

| 方法 | 传输次数 | 相对性能 |
|------|---------|---------|
| 单独写入 | N 次 | 1x (基准) |
| 多播 | 1 次 | Nx |
| 多播 + 链接列表 | 1 次 | Nx + 开销 |

### 3.3 批量传输优化

#### 批量读取策略

```cpp
// 优化前：逐个 tile 读取
for (uint32_t i = 0; i < num_tiles; i++) {
    noc_async_read_tile(i, dram_buffer, l1_addr + i * tile_size);
    noc_async_read_barrier();  // 每次等待
}

// 优化后：批量读取，单次等待
constexpr uint32_t BATCH_SIZE = 8;
for (uint32_t b = 0; b < num_tiles; b += BATCH_SIZE) {
    uint32_t batch_count = std::min(BATCH_SIZE, num_tiles - b);

    // 启动批量读取
    for (uint32_t i = 0; i < batch_count; i++) {
        noc_async_read_tile(b + i, dram_buffer, l1_addr + i * tile_size);
    }

    // 等待整个批次完成
    noc_async_read_barrier();
}
```

#### 状态保持优化

```cpp
// 使用状态保持 API 减少设置开销
noc_async_read_set_state(dram_base_addr);

for (uint32_t i = 0; i < num_tiles; i++) {
    // 复用之前设置的状态
    noc_async_read_with_state(l1_addr + i * tile_size, tile_size);
}

noc_async_read_barrier();
```

### 3.4 异步流水线深度优化

#### 流水线深度选择

```
流水线深度 vs 内存占用:

深度 1 (最小重叠):
[读取 1] [计算 1] [写入 1] [读取 2] [计算 2] [写入 2]

深度 2 (双缓冲):
[读取 1] [读取 2]
         [计算 1] [计算 2]
                  [写入 1] [写入 2]

深度 4 (最大重叠):
[读取 1] [读取 2] [读取 3] [读取 4]
         [计算 1] [计算 2] [计算 3] [计算 4]
                  [写入 1] [写入 2] [写入 3] [写入 4]
```

#### 实现示例

```cpp
// 深度流水线实现
constexpr uint32_t PIPELINE_DEPTH = 4;

void kernel_main() {
    uint32_t cb_id = CBIndex::c_0;
    uint32_t tiles_per_iter = 1;

    // 预填充流水线
    for (uint32_t i = 0; i < PIPELINE_DEPTH && i < num_tiles; i++) {
        cb_reserve_back(cb_id, tiles_per_iter);
        uint32_t l1_addr = get_write_ptr(cb_id);
        noc_async_read_tile(i, dram_buffer, l1_addr);
        noc_async_read_barrier();
        cb_push_back(cb_id, tiles_per_iter);
    }

    // 主循环 - 处理与预取重叠
    for (uint32_t i = PIPELINE_DEPTH; i < num_tiles + PIPELINE_DEPTH; i++) {
        // 等待当前数据可用
        cb_wait_front(cb_id, tiles_per_iter);

        // 启动下一个读取 (如果还有数据)
        if (i < num_tiles) {
            cb_reserve_back(cb_id, tiles_per_iter);
            uint32_t next_l1_addr = get_write_ptr(cb_id);
            noc_async_read_tile(i, dram_buffer, next_l1_addr);
        }

        // 处理当前数据 (与上面的读取并行)
        process_tiles(cb_id);
        cb_pop_front(cb_id, tiles_per_iter);

        // 等待读取完成 (如果启动了)
        if (i < num_tiles) {
            noc_async_read_barrier();
            cb_push_back(cb_id, tiles_per_iter);
        }
    }
}
```

### 3.5 NoC 拥塞避免

#### 拥塞检测

```cpp
// 使用 Watcher 监控 NoC 状态
export TT_METAL_WATCHER=1
export TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=0  // 启用 NOC 检查
```

#### 拥塞避免策略

**1. 分散访问模式**

```cpp
// 不良实践：所有核心同时访问同一 DRAM bank
// 会导致热点和拥塞

// 最佳实践：交错访问不同 bank
uint32_t bank_id = core_id % NUM_DRAM_BANKS;
uint32_t tile_offset = bank_id * tiles_per_bank;
```

**2. 时间分散**

```cpp
// 添加小延迟分散访问
uint32_t delay = core_id * 10;  // 每个核心不同延迟
for (uint32_t i = 0; i < delay; i++) {
    asm volatile("nop");
}
noc_async_read(...);
```

**3. 空间分散**

```cpp
// 使用核心坐标决定访问模式
CoreCoord core = get_core_coord();
bool use_noc_0 = (core.x + core.y) % 2 == 0;

if (use_noc_0) {
    noc_async_read(...);  // NOC_0
} else {
    // 通过不同路径
    noc_async_read(...);  // NOC_1
}
```

---

## 4. 并行化策略

### 4.1 核心网格划分最佳实践

#### 划分策略

| 工作负载类型 | 推荐划分 | 说明 |
|-------------|---------|------|
| **Matmul (M×K × K×N)** | 2D 网格 (M/网格行, N/网格列) | 输出并行 |
| **Conv2d** | 输出通道 + 空间维度 | 滤波器并行 |
| **Element-wise** | 数据并行 | 均匀分割 |
| **Reduce** | 归约维度 | 树形归约 |

#### Matmul 划分示例

```cpp
// M×N 输出矩阵划分到 core_grid 核心
void partition_matmul(
    uint32_t M, uint32_t N,
    CoreRange core_grid,
    std::vector<CoreCoord>& core_assignments
) {
    uint32_t num_cores_x = core_grid.end.x - core_grid.start.x + 1;
    uint32_t num_cores_y = core_grid.end.y - core_grid.start.y + 1;

    uint32_t per_core_M = M / num_cores_y;
    uint32_t per_core_N = N / num_cores_x;

    for (uint32_t y = 0; y < num_cores_y; y++) {
        for (uint32_t x = 0; x < num_cores_x; x++) {
            CoreCoord core(core_grid.start.x + x, core_grid.start.y + y);

            // 计算该核心的工作负载
            uint32_t m_start = y * per_core_M;
            uint32_t n_start = x * per_core_N;

            SetRuntimeArgs(program, compute_kernel, core, {
                m_start, n_start, per_core_M, per_core_N, K
            });
        }
    }
}
```

#### 负载均衡考虑

```cpp
// 处理不均匀划分
void balanced_partition(
    uint32_t total_work,
    uint32_t num_cores,
    std::vector<uint32_t>& work_per_core
) {
    work_per_core.resize(num_cores);

    uint32_t base_work = total_work / num_cores;
    uint32_t remainder = total_work % num_cores;

    for (uint32_t i = 0; i < num_cores; i++) {
        // 前 remainder 个核心多分配一个单位
        work_per_core[i] = base_work + (i < remainder ? 1 : 0);
    }
}
```

### 4.2 负载均衡技术

#### 动态负载均衡

```cpp
// 工作窃取模式
void work_stealing_kernel() {
    uint32_t my_work = get_initial_work();
    uint32_t work_done = 0;

    while (work_done < my_work) {
        // 执行工作
        process_tile();
        work_done++;

        // 检查邻居是否需要帮助
        if (work_done % CHECK_INTERVAL == 0) {
            check_neighbor_load();
        }
    }

    // 尝试从繁忙核心窃取工作
    while (has_available_work()) {
        steal_work_from_busy_core();
    }
}
```

#### 静态负载均衡

```cpp
// 基于数据大小的预分配
void size_based_partition(
    const std::vector<uint32_t>& tile_sizes,
    uint32_t num_cores,
    std::vector<std::vector<uint32_t>>& core_assignments
) {
    // 按大小降序排序
    std::vector<std::pair<uint32_t, uint32_t>> sized_tiles;
    for (uint32_t i = 0; i < tile_sizes.size(); i++) {
        sized_tiles.push_back({tile_sizes[i], i});
    }
    std::sort(sized_tiles.rbegin(), sized_tiles.rend());

    // 使用最小堆分配
    std::priority_queue<std::pair<uint32_t, uint32_t>,
                        std::vector<std::pair<uint32_t, uint32_t>>,
                        std::greater<>> core_loads;

    for (uint32_t i = 0; i < num_cores; i++) {
        core_loads.push({0, i});
    }

    for (auto& [size, tile_id] : sized_tiles) {
        auto [load, core_id] = core_loads.top();
        core_loads.pop();

        core_assignments[core_id].push_back(tile_id);
        core_loads.push({load + size, core_id});
    }
}
```

### 4.3 同步开销最小化

#### 减少全局同步

```cpp
// 不良实践：每个 tile 后全局同步
for (uint32_t i = 0; i < num_tiles; i++) {
    process_tile(i);
    noc_semaphore_set(global_sem, 1);  // 全局同步
    noc_semaphore_wait(global_sem, num_cores);
}

// 最佳实践：批量后同步
for (uint32_t b = 0; b < num_tiles; b += BATCH_SIZE) {
    for (uint32_t i = 0; i < BATCH_SIZE; i++) {
        process_tile(b + i);
    }
    // 批量后同步
    noc_semaphore_set(global_sem, 1);
    noc_semaphore_wait(global_sem, num_cores);
}
```

#### 层次化同步

```cpp
// 使用层次化同步减少开销
void hierarchical_sync() {
    // 第一层：行内同步
    noc_semaphore_set(row_sem[row_id], 1);
    noc_semaphore_wait(row_sem[row_id], cores_per_row);

    // 第二层：全局同步 (仅行首核心参与)
    if (is_row_head) {
        noc_semaphore_set(global_sem, 1);
        noc_semaphore_wait(global_sem, num_rows);
        // 通知行内其他核心
        noc_semaphore_set(row_done_sem[row_id], 1);
    }

    // 等待行同步完成
    noc_semaphore_wait(row_done_sem[row_id], 1);
}
```

### 4.4 子设备并行执行模式

#### 子设备概念

子设备允许将单个物理设备划分为多个逻辑设备，实现更细粒度的并行控制：

```cpp
// 创建子设备管理器
auto sub_device_manager = device->create_sub_device_manager(
    {CoreRange({0, 0}, {3, 3})},  // 子设备 0: 4x4 核心
    0  // 子设备 ID
);

// 加载子设备管理器
device->load_sub_device_manager(sub_device_manager);

// 获取子设备 ID
auto sub_device_ids = device->get_sub_device_ids();
```

#### 并行执行模式

```cpp
// 模式 1: 流水线并行
void pipeline_parallel_example() {
    // 子设备 0: 嵌入层
    // 子设备 1: Transformer 层
    // 子设备 2: 输出层

    auto embedding_cq = device->command_queue(0, sub_device_ids[0]);
    auto transformer_cq = device->command_queue(0, sub_device_ids[1]);
    auto output_cq = device->command_queue(0, sub_device_ids[2]);

    // 流水线执行
    for (uint32_t i = 0; i < batch_size; i++) {
        EnqueueProgram(embedding_cq, embedding_program, false);
        EnqueueProgram(transformer_cq, transformer_program, false);
        EnqueueProgram(output_cq, output_program, false);
    }
}

// 模式 2: 张量并行
void tensor_parallel_example() {
    // 将大 Matmul 划分到多个子设备

    auto left_half_cq = device->command_queue(0, sub_device_ids[0]);
    auto right_half_cq = device->command_queue(0, sub_device_ids[1]);

    // 并行执行
    EnqueueProgram(left_half_cq, matmul_left_program, false);
    EnqueueProgram(right_half_cq, matmul_right_program, false);

    // 同步
    Finish(left_half_cq);
    Finish(right_half_cq);
}
```

---

## 5. Metal Trace 深度优化

### 5.1 Trace 捕获最佳实践

#### Trace 工作流程

```python
import ttnn

# 1. 准备阶段 - 分配输入/输出张量
input_tensor = ttnn.allocate_tensor_on_device(input_shape, device)
output_tensor = ttnn.allocate_tensor_on_device(output_shape, device)

# 2. 预热运行 (可选但推荐)
for _ in range(3):
    output_tensor = model(input_tensor)

# 3. 开始 Trace 捕获
ttnn.begin_trace_capture(device, cq_id=0)

# 4. 执行模型 (将被记录)
output_tensor = model(input_tensor)

# 5. 结束 Trace 捕获
ttnn.end_trace_capture(device, trace_id, cq_id=0)

# 6. 重放 Trace (多次)
for _ in range(num_iterations):
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)

# 7. 释放 Trace
ttnn.release_trace(device, trace_id)
```

#### 捕获优化

```python
# 最佳实践 1: 固定形状
# Trace 要求所有张量形状固定
fixed_shape = [1, 32, 32, 64]  # 批大小、通道等必须固定

# 最佳实践 2: 预分配所有缓冲区
class TracedModel:
    def __init__(self, device):
        self.device = device
        # 预分配所有中间缓冲区
        self.intermediate_bufs = {
            'layer1_out': ttnn.allocate_tensor_on_device(...),
            'layer2_out': ttnn.allocate_tensor_on_device(...),
        }

    def forward(self, input_tensor):
        # 使用预分配缓冲区，无动态分配
        x = layer1(input_tensor, output_tensor=self.intermediate_bufs['layer1_out'])
        x = layer2(x, output_tensor=self.intermediate_bufs['layer2_out'])
        return x
```

### 5.2 动态形状处理

#### 形状桶策略

```python
# 为不同输入形状创建多个 Trace
class MultiTraceModel:
    def __init__(self, device):
        self.device = device
        self.traces = {}

        # 为常见形状预编译 Trace
        for seq_len in [128, 256, 512, 1024]:
            trace_id = self.capture_trace(seq_len)
            self.traces[seq_len] = trace_id

    def capture_trace(self, seq_len):
        input_shape = [1, seq_len, 768]
        input_tensor = ttnn.allocate_tensor_on_device(input_shape, self.device)

        ttnn.begin_trace_capture(self.device, cq_id=0)
        output = self.model(input_tensor)
        trace_id = ttnn.end_trace_capture(self.device, cq_id=0)

        return trace_id

    def forward(self, input_tensor):
        seq_len = input_tensor.shape[1]

        # 找到最接近的预编译 Trace
        closest_seq_len = self.find_closest(seq_len)
        trace_id = self.traces[closest_seq_len]

        # 如果形状不完全匹配，可能需要填充
        if seq_len != closest_seq_len:
            input_tensor = self.pad_to_length(input_tensor, closest_seq_len)

        ttnn.execute_trace(self.device, trace_id, cq_id=0)
        return self.get_output()
```

### 5.3 Trace 缓冲区管理

#### 缓冲区大小计算

```python
def calculate_trace_buffer_size(model, input_shape):
    """估算 Trace 缓冲区需求"""

    # 基础开销
    base_overhead = 1024 * 1024  # 1MB 基础

    # 每个操作的命令开销
    num_ops = count_operations(model)
    per_op_overhead = 4 * 1024  # 4KB 每操作

    # 内核二进制大小
    kernel_size = estimate_kernel_binary_size(model)

    # 运行时参数
    runtime_args_size = num_ops * 256  # 256 bytes 每操作

    total_size = base_overhead + (num_ops * per_op_overhead) + \
                 kernel_size + runtime_args_size

    # 添加 20% 余量
    return int(total_size * 1.2)

# 配置 Trace 缓冲区
ttnn.set_trace_buffer_size(device, calculate_trace_buffer_size(model, input_shape))
```

### 5.4 多 Trace 切换

```python
# 管理多个 Trace
class TraceManager:
    def __init__(self, device):
        self.device = device
        self.traces = {}
        self.active_trace = None

    def register_trace(self, name, trace_id):
        self.traces[name] = trace_id

    def switch_trace(self, name):
        if self.active_trace:
            # 可选：暂停当前 trace
            pass

        self.active_trace = self.traces[name]
        return self.active_trace

    def execute(self, name):
        trace_id = self.switch_trace(name)
        ttnn.execute_trace(self.device, trace_id, cq_id=0)

    def cleanup(self):
        for name, trace_id in self.traces.items():
            ttnn.release_trace(self.device, trace_id)
        self.traces.clear()

# 使用示例
trace_mgr = TraceManager(device)

# 捕获不同阶段的 trace
ttnn.begin_trace_capture(device, cq_id=0)
embedding_output = embedding_layer(input_ids)
trace_mgr.register_trace('embedding', ttnn.end_trace_capture(device, cq_id=0))

ttnn.begin_trace_capture(device, cq_id=0)
for layer in transformer_layers:
    hidden_states = layer(hidden_states)
trace_mgr.register_trace('transformer', ttnn.end_trace_capture(device, cq_id=0))

# 执行时切换
trace_mgr.execute('embedding')
trace_mgr.execute('transformer')
```

---

## 6. 多芯片优化

### 6.1 芯片间拓扑感知

#### 拓扑结构

```
Galaxy 4×8 Mesh 拓扑:

    N (北)
    ↑
W ← → E (西/东)
    ↓
    S (南)
    +
    Z (垂直)

Chip (0,0) ←──→ Chip (1,0) ←──→ Chip (2,0) ←──→ Chip (3,0)
    ↑               ↑               ↑               ↑
    ↓               ↓               ↓               ↓
Chip (0,1) ←──→ Chip (1,1) ←──→ Chip (2,1) ←──→ Chip (3,1)
    ↑               ↑               ↑               ↑
    .               .               .               .
    .               .               .               .
    ↑               ↑               ↑               ↑
Chip (0,7) ←──→ Chip (1,7) ←──→ Chip (2,7) ←──→ Chip (3,7)
```

#### 拓扑感知数据放置

```cpp
// 根据拓扑放置数据以最小化跳数
void topology_aware_placement(
    DeviceMesh& mesh,
    Tensor& tensor,
    ShardSpec& shard_spec
) {
    auto grid = mesh.get_grid();

    // 分析通信模式
    if (is_all_reduce_pattern(tensor)) {
        // 环形放置减少最大跳数
        place_in_ring_pattern(mesh, tensor);
    } else if (is_pipeline_pattern(tensor)) {
        // 流水线放置沿最短路径
        place_in_pipeline_pattern(mesh, tensor);
    } else {
        // 默认：块放置
        place_in_block_pattern(mesh, tensor);
    }
}
```

### 6.2 CCL 算法选择

#### 集体通信操作

| 操作 | 算法 | 适用场景 | 复杂度 |
|------|------|----------|--------|
| **AllGather** | 环形 | 大数据量 | O(N) |
| **AllGather** | 树形 | 小数据量 | O(log N) |
| **ReduceScatter** | 环形 | 规约操作 | O(N) |
| **AllReduce** | Ring-Reduce | 通用 | O(N) |
| **AllReduce** | Recursive Halving | 小数据量 | O(log N) |
| **Broadcast** | 树形 | 单到多 | O(log N) |

#### 算法选择指南

```python
def select_ccl_algorithm(operation, data_size, num_chips):
    """
    选择最优 CCL 算法

    Args:
        operation: 'all_gather', 'reduce_scatter', 'all_reduce'
        data_size: 数据大小 (bytes)
        num_chips: 芯片数量
    """

    if operation == 'all_reduce':
        if data_size < 1 * 1024 * 1024:  # < 1MB
            return 'recursive_halving'  # 低延迟
        else:
            return 'ring_reduce'  # 高带宽

    elif operation == 'all_gather':
        if data_size > 10 * 1024 * 1024:  # > 10MB
            return 'ring'  # 带宽最优
        else:
            return 'tree'  # 延迟最优

    elif operation == 'reduce_scatter':
        return 'ring'  # 通常环形最优

    return 'default'

# 使用示例
algorithm = select_ccl_algorithm('all_reduce', tensor_size, 8)
ttnn.all_reduce(tensor, algorithm=algorithm)
```

### 6.3 以太网带宽优化

#### 带宽特性

| 架构 | 以太网端口 | 单端口带宽 | 总带宽 |
|------|-----------|-----------|--------|
| Wormhole | 16× 100G | 12.5 GB/s | 200 GB/s |
| Blackhole | 12× 400G | 50 GB/s | 600 GB/s |

#### 优化策略

**1. 链路聚合**

```cpp
// 使用多个以太网链路并行传输
void multi_link_transfer(
    Tensor& src_tensor,
    Device* src_device,
    Device* dst_device
) {
    uint32_t num_links = get_num_eth_links(src_device, dst_device);
    uint32_t chunk_size = src_tensor.size() / num_links;

    // 分割数据到多个链路
    for (uint32_t i = 0; i < num_links; i++) {
        uint32_t offset = i * chunk_size;
        eth_send_chunk(src_device, dst_device, i, offset, chunk_size);
    }

    // 等待所有链路完成
    for (uint32_t i = 0; i < num_links; i++) {
        eth_wait_for_completion(src_device, i);
    }
}
```

**2. 通信与计算重叠**

```cpp
// 重叠芯片间通信与计算
void overlapping_communication_compute() {
    // 阶段 1: 计算当前块
    compute_current_block();

    // 阶段 2: 异步启动下一块的传输
    eth_send_next_block_async();

    // 阶段 3: 继续当前计算 (与传输重叠)
    continue_compute();

    // 阶段 4: 等待传输完成
    eth_wait_for_completion();
}
```

**3. 拓扑感知路由**

```python
# 最小化跨芯片跳数
def topology_aware_all_gather(tensor, mesh):
    """
    根据芯片拓扑执行 AllGather
    优先使用最短路径
    """
    grid = mesh.get_grid()

    # 确定最佳收集顺序
    if grid.width >= grid.height:
        # 水平优先
        for row in range(grid.height):
            # 行内收集
            gather_along_row(mesh, row)
        for col in range(grid.width):
            # 列广播
            broadcast_along_column(mesh, col)
    else:
        # 垂直优先
        for col in range(grid.width):
            gather_along_column(mesh, col)
        for row in range(grid.height):
            broadcast_along_row(mesh, row)

    return gathered_tensor
```

---

## 7. 性能调试方法论

### 7.1 性能分析流程

```
性能优化迭代流程:

1. 建立基线
   └── 测量当前性能 (吞吐量、延迟)

2. 识别瓶颈
   ├── 使用 Device Profiler 分析 kernel 时间
   ├── 使用 Tracy 分析 host 开销
   └── 确定主要瓶颈 (计算/内存/通信)

3. 针对性优化
   ├── 计算瓶颈 → 优化 Math Fidelity、并行度
   ├── 内存瓶颈 → 优化 CB 大小、数据布局
   └── 通信瓶颈 → 优化 NoC 路由、多播

4. 验证优化
   ├── 检查 PCC >= 0.99
   ├── 测量性能提升
   └── 确保没有回归

5. 迭代
   └── 返回步骤 2 直到满足目标
```

### 7.2 瓶颈识别

#### 计算瓶颈特征

```cpp
// 迹象 1: TRISC 核心长时间忙碌
// Device Profiler 显示 TRISC-KERNEL 区域占主导

// 迹象 2: BRISC/NCRISC 等待
// 数据移动核心在等待计算完成

// 解决方案: 增加并行度或减少计算量
void optimize_compute_bound() {
    // 1. 使用更多核心
    CoreRange grid({0, 0}, {7, 7});  // 8x8 网格

    // 2. 降低 Math Fidelity
    ComputeConfig{
        .math_fidelity = MathFidelity::LoFi  // 从 HiFi4 降低
    };

    // 3. 使用近似函数
    .math_approx_mode = true;
}
```

#### 内存瓶颈特征

```cpp
// 迹象 1: noc_async_read/write_barrier 等待时间长
// 迹象 2: cb_reserve_back/cb_wait_front 等待时间长

// 解决方案: 优化内存访问模式
void optimize_memory_bound() {
    // 1. 增加 CB 大小
    CircularBufferConfig cb_config(
        16 * tile_size,  // 增加缓冲区
        {{cb_index, data_format}}
    );

    // 2. 使用双缓冲
    // 重叠计算和数据传输

    // 3. 批量传输
    const uint32_t batch_size = 8;

    // 4. 使用多播减少重复传输
    noc_async_write_multicast(...);
}
```

#### 通信瓶颈特征

```cpp
// 迹象 1: 多芯片场景下以太网传输时间长
// 迹象 2: NoC 拥塞导致的延迟

// 解决方案: 优化通信模式
void optimize_communication_bound() {
    // 1. 重叠通信与计算
    // 使用异步操作

    // 2. 拓扑感知放置
    place_data_topology_aware();

    // 3. 减少同步点
    // 使用层次化同步替代全局同步
}
```

### 7.3 性能指标

#### 关键指标

| 指标 | 计算方法 | 目标值 |
|------|---------|--------|
| **吞吐量** | tiles/second 或 ops/second | 接近理论峰值 |
| **内存带宽利用率** | 实际带宽 / 峰值带宽 | > 80% |
| **计算利用率** | 实际 TFLOPs / 峰值 TFLOPs | > 70% |
| **PCIe 带宽** | 传输速率 / 理论峰值 | > 10 GB/s |
| **PCC** | 皮尔逊相关系数 | >= 0.99 |

#### 测量代码

```cpp
// 设备端计时
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    DeviceZoneScopedN("KernelTotal");

    {
        DeviceZoneScopedN("ReadPhase");
        // 读取代码
    }

    {
        DeviceZoneScopedN("ComputePhase");
        // 计算代码
    }

    {
        DeviceZoneScopedN("WritePhase");
        // 写入代码
    }
}
```

```python
# Host 端计时
import time

# 预热
for _ in range(3):
    run_model()

# 测量
start = time.perf_counter()
for _ in range(num_iterations):
    run_model()
    ttnn.synchronize_device(device)
end = time.perf_counter()

throughput = (batch_size * num_iterations) / (end - start)
print(f"Throughput: {throughput} samples/sec")
```

---

## 8. 优化检查清单

### 8.1 内存优化检查清单

- [ ] CB 大小是否针对 L1 容量优化？
- [ ] 是否使用了双缓冲重叠计算和通信？
- [ ] DRAM 访问是否批量执行？
- [ ] 是否最小化了 CB 的 push/pop 操作？
- [ ] 是否使用了多播减少冗余传输？
- [ ] 内存布局是否避免了 bank 冲突？
- [ ] 是否预分配了所有需要的缓冲区？
- [ ] 栈大小是否适当（不过大）？

### 8.2 计算优化检查清单

- [ ] Math Fidelity 是否针对精度要求优化？
- [ ] 是否使用了近似模式（如精度允许）？
- [ ] FP32 累加是否仅在需要时启用？
- [ ] 数据格式是否最优（BF16 vs FP32）？
- [ ] 是否融合了多个操作减少内存往返？
- [ ] 是否使用了正确的计算引擎（FPU vs SFPU）？
- [ ] PCC 是否 >= 0.99？

### 8.3 通信优化检查清单

- [ ] NoC 路由是否最优（NOC_0 vs NOC_1）？
- [ ] 是否使用了多播进行广播？
- [ ] 异步操作是否正确使用？
- [ ] 是否避免了 NoC 拥塞？
- [ ] 多芯片通信是否与计算重叠？
- [ ] 是否使用了拓扑感知的 CCL 算法？

### 8.4 并行化检查清单

- [ ] 核心网格划分是否平衡？
- [ ] 工作负载是否均匀分布？
- [ ] 同步点是否最小化？
- [ ] 是否使用了层次化同步？
- [ ] Metal Trace 是否启用（如适用）？
- [ ] 多 CQ 是否用于 I/O 重叠？

---

## 参考资源

### 官方文档

- [TT-Metalium 性能优化指南](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/performance.html)
- [Metal Trace 文档](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/tracy_profiler.html)
- [Device Profiler 文档](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/device_program_profiler.html)

### 技术报告

- `tech_reports/AdvancedPerformanceOptimizationsForModels/` - Metal Trace & Multi-CQ 深度指南
- `tech_reports/data_formats/data_formats.md` - 数据格式详细说明
- `tech_reports/tensor_sharding/tensor_sharding.md` - 张量分片策略
- `tech_reports/memory/allocator.md` - 内存分配器内部机制

### 示例代码

- `tt_metal/programming_examples/matmul_multi_core/` - 多核 Matmul 优化
- `tt_metal/programming_examples/matmul_multichip/` - 多芯片 Matmul
- `tests/ttnn/unit_tests/test_trace.py` - Metal Trace 示例

---

*文档生成时间: 2026-03-12*
*基于 Tenstorrent TT-Metalium 最新文档整理*
