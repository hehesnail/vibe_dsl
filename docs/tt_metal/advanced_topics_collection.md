# TT-Metalium 高级主题与优化指南

本文档收集了 Tenstorrent TT-Metalium 的进阶内容，包括多芯片编程、性能优化最佳实践以及调试与剖析工具的使用方法。

---

## 目录

1. [多芯片编程指南](#1-多芯片编程指南)
2. [性能优化最佳实践](#2-性能优化最佳实践)
3. [调试与剖析工具](#3-调试与剖析工具)

---

## 1. 多芯片编程指南

### 1.1 多芯片架构概述

TT-Metalium 支持 Tenstorrent 处理器的多芯片扩展，采用 **Mesh 拓扑结构** 实现横向扩展：

| 配置 | 芯片数量 | 拓扑结构 |
|------|----------|----------|
| N300 | 2 芯片 | 双芯片卡 |
| QuietBox | 8 Wormhole / 4 Blackhole | Mesh |
| Galaxy | 32 芯片 | 4×8 Mesh |
| 多主机 | 跨主机连接 | 扩展 Mesh |

**TT-Fabric** 是 Tenstorrent 的原生多设备网络层，基于以太网实现：

| 特性 | 规格 |
|------|------|
| Blackhole 以太网带宽 | 1 TB/s |
| Galaxy I/O (32 芯片) | 11.2 TB/s |
| 单链路带宽 | 200 GB/s (N/S/W/E/Z 方向) |
| 支持的拓扑 | 2D/3D Torus、任意拓扑 |

### 1.2 芯片间通信机制

#### 核心能力

- **原生多设备内核**：支持跨芯片融合和重叠操作的内核
- **直接核心间通信**：任何核心可以直接读写/同步任何其他核心或芯片
- **融合计算与通信**：在内核中融合和重叠计算与芯片间通信

#### 以太网 Mesh 拓扑

```
Galaxy 配置 (4×8 Mesh):
+----+----+----+----+----+----+----+----+
| C0 | C1 | C2 | C3 | C4 | C5 | C6 | C7 |
+----+----+----+----+----+----+----+----+
| C8 | C9 |... |    |    |    |    |C15 |
+----+----+----+----+----+----+----+----+
|... |    |    |    |    |    |    |... |
+----+----+----+----+----+----+----+----+
|C24 |C25 |C26 |C27 |C28 |C29 |C30 |C31 |
+----+----+----+----+----+----+----+----+
```

每个芯片可以通过以太网链路与其北、南、西、东、Z 方向的邻居通信。

### 1.3 多芯片编程模型

#### 编程哲学（自底向上）

1. **单 Tensix 核心内核**：首先在单个核心上开发内核
2. **多核心调度**：将内核调度到多个 Tensix 核心（带同步）
3. **多设备扩展**：扩展到多个设备（大型模型部署的关键）

#### 集体通信库 (CCL)

TT-Metalium/TTNN 提供内置的集体操作：

| 操作 | 描述 |
|------|------|
| `ttnn.all_reduce` | 跨所有处理器归约张量 |
| `ttnn.broadcast` | 从一个处理器广播到所有其他处理器 |
| `ttnn.all_gather` | 从所有处理器收集 |
| `ttnn.reduce_scatter` | 跨处理器归约并分散 |

#### 编程示例

```cpp
// 创建设备
IDevice* device = CreateDevice(device_id);
CommandQueue& cq = device->command_queue();

// 内核可以通过以太网定位远程芯片
// 数据移动内核处理芯片间传输
auto reader = CreateKernel(program, "reader_kernel.cpp", core, config);
auto compute = CreateKernel(program, "compute_kernel.cpp", core, compute_config);
auto writer = CreateKernel(program, "writer_kernel.cpp", core, config);
```

#### 与 GPU 编程模型对比

| 特性 | TT-Metalium | 传统 GPU |
|------|-------------|----------|
| 内核语言 | 纯 C++ | CUDA/HIP/SYCL |
| 数据移动 | 显式、可编程 | 通常隐式 |
| 多设备 | 内核语言原生支持 | 通常主机协调 |
| 通信 | 与计算融合 | 独立内核/API |
| 拓扑 | 任意 Mesh/Torus | 通常固定 (NVLink 等) |

---

## 2. 性能优化最佳实践

### 2.1 内存优化技巧

#### 内存层次结构

| 内存类型 | 带宽 | 访问模式 |
|----------|------|----------|
| SRAM (本地/分片) | 94 TB/s | 最快，计算首选 |
| SRAM (邻居/Halo) | 47 TB/s | 空间数据重用 |
| SRAM (多播) | 24 TB/s | 高效广播 |
| DRAM | 512 GB/s | 大容量存储，使用 NoC 访问 |
| 以太网 | 1 TB/s | 扩展通信 |

#### 内存优化策略

| 技术 | 描述 | 收益 |
|------|------|------|
| **L1 vs DRAM** | 将频繁访问的数据放入快速 L1 内存 | 降低延迟 |
| **分片 (Sharding)** | 跨核心分布张量 | 并行访问，更高带宽 |
| **双缓冲** | 重叠数据传输与计算 | 隐藏延迟 |
| **内存分析** | 使用 `ttnn.device.dump_device_memory_state()` | 识别瓶颈 |

#### 内存受限工作负载解决方案

- 卷积减少 `act_block_h_override`
- 将跳跃连接存储在 DRAM 而非 L1
- 基于张量访问模式优化分片策略

### 2.2 计算优化建议

#### 数据格式优化

- **使用低精度格式**：选择 `bfloat16` 或 `bfloat8_b` 作为权重，减少内存带宽并提高计算吞吐量
- **bfloat8_b** 对神经网络权重特别有效，在最小精度损失的情况下提供显著的内存节省
- 更改数据格式后始终验证 **PCC (Pearson 相关系数) ≥ 0.99** 以确保模型正确性

#### 向量引擎能力

| 数据格式 | 累加器 | 用例 |
|----------|--------|------|
| FP32 | FP32 | 高精度 |
| INT16 | INT32 | 量化操作 |
| INT32 | INT32 | 整数运算 |

#### 矩阵引擎（高性能）

| 格式 | TLOPs |
|------|-------|
| Block FP2/FP4/FP8 | 745 |
| BFLOAT16 | 373 |
| TF32 | 186 |

#### 计算优化技术

**CNN/视觉模型：**
- 卷积特定优化：分片策略、块调优 (`act_block_h` 调优)
- 批归一化融合
- 优化 `act_block_h_override` 参数

**矩阵乘法：**
- **多核心分布**：将工作负载分配到多个 Tensix 核心
- **数据重用**：从 DRAM 读取一次，通过 NoC 广播到其他核心
- **避免 NoC 拥塞**：仔细放置读取器/计算核心

### 2.3 并行化策略

#### Metal Trace（捕获与重放）

```python
# 启用 Metal Trace 的模式
allocate_tensor → begin_trace_capture → model() → end_trace_capture
```

**收益：**
- 消除主机调度开销
- **LLM 推理**优化的关键
- 需要固定计算图（无动态形状）

**最适合：** 主机调度开销降低、Transformer/基于注意力的模型

#### 多 CQ（多命令队列）

- **使用多个命令队列重叠 I/O 与计算**
- 将独立操作提交到不同队列以并行执行
- 减少空闲时间并提高流水线利用率

#### 关键优化策略

1. **重叠计算与数据移动** - 计算运行时使��异步 NoC 操作
2. **基于瓦片的处理** - 将数据组织成瓦片以有效利用矩阵引擎
3. **分片数据布局** - 保持数据本地以最小化 NoC 流量
4. **多播广播** - 使用硬件多播而非单独写入
5. **最小化 DRAM 访问** - 通过仔细的数据放置优先使用 SRAM

### 2.4 优化工作流程（7 步骤）

| 步骤 | 重点领域 | 关键操作 |
|------|----------|----------|
| 1 | 数据格式 | 选择 `bfloat16`/`bfloat8_b` |
| 2 | 张量布局 | 配置瓦片布局 |
| 3 | 内存与分片 | L1/DRAM 优化、双缓冲 |
| 4 | Metal Trace | 捕获/重放操作 |
| 5 | 多 CQ | 重叠 I/O 与计算 |
| 6 | Conv2d 优化 | 分片、块调优 (CNN) |
| 7 | 多设备 | 跨多芯片扩展 |

### 2.5 配置文件驱动的优化反馈循环

```
1. 分析当前性能（基线）
2. 一次应用一个优化
3. 验证 PCC ≥ 0.99
4. 如果 PCC 下降 → 回退并尝试不同方法
5. 测量性能改进
6. 重复直到达到目标
```

### 2.6 架构特定提示

| 模型类型 | 优先优化 |
|----------|----------|
| **LLM/Transformer** | 数据格式 → Metal Trace → 多 CQ |
| **CNN/视觉** | 分片 → Conv2d 调优 → `act_block_h` |
| **内存受限** | 内存与分片 → 双缓冲 |
| **主机受限** | Metal Trace → 多 CQ |

---

## 3. 调试与剖析工具

### 3.1 Tracy Profiler

#### 概述

TT-Metalium 使用 **Tracy 的分支**（一种开源 C++ 性能分析工具），适配 Tenstorrent 的 Tensix 处理器。Tracy 为主机端代码提供**采样和代码插桩**分析功能，包括 **Python 和 C++** 代码。

#### 启用 Tracy

Tracy 分析支持在构建 Metalium 时**默认启用**：

```bash
# 通过构建脚本
./build_metal.sh

# 或通过 CMake 标志
cmake . -DENABLE_TRACY=ON
ninja
ninja install
```

旧版本使用：
```bash
build_metal.sh --enable-profiler
```

#### C++ 主机端分析

在 C++ 代码中使用 Tracy 宏标记区域：

```cpp
#include <tracy/Tracy.hpp>

Device::Device(chip_id_t device_id, const std::vector<uint32_t>& l1_bank_remap) : id_(device_id)
{
    ZoneScoped;  // 自动分析此作用域
    this->initialize(l1_bank_remap);
}
```

常用宏：
- `ZoneScoped;` - 分析当前作用域
- `ZoneScopedN("name");` - 命名区域分析

#### Python 主机端分析

TT Metal 使用 Python 标准的 **`sys.setprofile`** 和 **`sys.settrace`** 函数集成 Python 分析。

**方法 1：命令行分析（整个脚本）**

```bash
python -m tracy {test_script}.py
```

**方法 2：Pytest 集成**

```bash
python -m tracy -m pytest models/experimental/bert_large_performant/unit_tests/test_bert_large_split_and_transform_qkv_heads.py::test_split_query_key_value_and_split_heads_with_program_cache
```

**方法 3：手动插桩**

```python
from tracy import Profiler

def function_under_test():
    child_function_1()
    child_function_2()

profiler = Profiler()
profiler.enable()
function_under_test()
profiler.disable()
```

**方法 4：标记事件的 Signpost**

```python
from tracy import signpost

signpost(header="Run number 5", message="This is the run after 5 warmup runs")
run_inference()

signpost(header="Run result post proc")
post_proc()
```

**行级分析**

```bash
python -m tracy -p -l -m pytest {test_path}
```

添加 `-l` 标志启用行级分析。

#### GUI 设置

Tracy 需要桌面 GUI 应用程序查看分析结果：

**macOS 安装**
```bash
brew uninstall tracy  # 移除旧版本
wget -P ~/ --no-check-certificate --no-cache --no-cookies https://raw.githubusercontent.com/tenstorrent-metal/tracy/master/tracy.rb
brew install ~/tracy.rb
rm ~/tracy.rb
```

**连接**
1. 启动 Tracy GUI
2. 将客户端地址设置为远程机器 IP 和端口 8086（例如 `172.27.28.132:8086`）
3. 点击连接 - 分析的应用程序启动后数据将实时流式传输

### 3.2 Device Profiler

#### 概述

**Device Program Profiler** 用于 **Tenstorrent 硬件上的内核性能分析**（包括计算和数据移动内核的所有 RISC-V 核心）。

#### 启用分析器

```bash
# 设置环境变量
export TT_METAL_DEVICE_PROFILER=1

# 运行应用程序
./your_tt_metal_application

# 或用于 TT-NN 操作
pytest tests/path/to/test.py  # 启用分析器
```

#### 使用 DeviceZoneScopedN 宏

```cpp
#include "debug/zone.h"

void kernel_main() {
    DeviceZoneScopedN("CUSTOM-ZONE");
    // 内核代码...
}
```

#### CSV 输出结构

分析器生成名为 **`profile_log_device.csv`** 的文件，位于：
```
${TT_METAL_HOME}/generated/profiler/.logs/
```

**CSV 头部格式：**
```
ARCH: grayskull, CHIP_FREQ[MHz]: 1202
PCIe slot, core_x, core_y, RISC processor type, timer_id, time[cycles since reset], stat value, Run ID, zone name, zone phase, source line, source file
```

**Blackhole 版本：**
```
ARCH: blackhole, CHIP_FREQ[MHz]: 1350, Max Compute Cores: 120
PCIe slot, core_x, core_y, RISC processor type, timer_id, time[cycles since reset], data, run host ID, trace id, trace id counter, zone name, type, source line, source file, meta data
```

#### 列描述

| 列 | 描述 |
|----|------|
| **PCIe slot** | 设备 PCIe 槽位标识符 |
| **core_x, core_y** | 芯片上的核心坐标 |
| **RISC processor type** | 处理器类型：`BRISC`、`NCRISC` 或 `TRISC` |
| **timer_id** | 定时器事件的唯一标识符 |
| **time[cycles since reset]** | 设备重置后的周期计数 |
| **zone name** | 分析区域标识符 |
| **zone phase** | `begin` 或 `end` 标记 |

#### 默认区域

| 区域 | 描述 |
|------|------|
| `BRISC-FW` / `NCRISC-FW` / `TRISC-FW` | 固件循环的单次迭代持续时间 |
| `BRISC-KERNEL` / `NCRISC-KERNEL` / `TRISC-KERNEL` | 内核 `main()` 函数执行持续时间 |

#### 性能分析方法

**计算固件执行时间：**
```python
# 找到最小和最大 time[cycles since reset] 值
elapsed_cycles = max_time - min_time
# 使用头部中的芯片频率转换为秒
elapsed_time = elapsed_cycles / (CHIP_FREQ_MHz * 1e6)
```

**区域持续时间分析：**
```python
import pandas as pd

# 读取 CSV（跳过 ARCH 信息的头部行）
df = pd.read_csv('profile_log_device.csv', skiprows=1)

# 过滤特定区域
zone_data = df[df['zone name'] == 'YOUR-ZONE']

# 计算 begin/end 对的持续时间
begins = zone_data[zone_data['zone phase'] == 'begin']['time[cycles since reset]']
ends = zone_data[zone_data['zone phase'] == 'end']['time[cycles since reset]']
durations = ends.values - begins.values

# 转换为微秒（示例：1202 MHz 芯片）
durations_us = durations / 1202
print(f"Mean duration: {durations_us.mean():.2f} µs")
```

#### 重要考虑事项

1. **构建要求**：使用 `--build-type Release` 进行准确的性能测量
2. **DPRINT 冲突**：分析时禁用 `TT_METAL_DPRINT_CORES` - 它们共享有限的片上 SRAM
3. **RISC-V 处理器**：
   - `BRISC` / `NCRISC`：控制路由器（RISC-V 0 和 4）
   - `TRISC`：Tensix 计算处理器（RISC-V 1-3）

### 3.3 Watcher 工具

#### 概述

**Watcher** 是 TT-Metalium 的调试工具，包含两个组件：
1. **插桩组件** - 插桩固件和内核以捕获常见编程错误
2. **主机监控线程** - 定期监控 TT 设备状态

#### 关键特性

- **记录路点 (Waypoints)** - 跟踪每个 RISC-V 核心最后执行的代码片段
- **清理的 NOC 事务** - 防止无效坐标/地址提交到硬件
- **内存损坏检测** - 检测 L1 内存地址 0 处的损坏
- **内核跟踪** - 显示当前执行内核的内核路径和名称
- **RISC-V 执行标志** - 指示当前内核中哪些 RISC-V 正在执行

#### 启用 Watcher

设置这些环境变量：

```bash
# 必需：设置轮询间隔（秒）（越长 = 侵入性越小）
export TT_METAL_WATCHER=120

# 可选：追加到现有日志文件而非创建新文件
export TT_METAL_WATCHER_APPEND=1

# 可选：转储所有状态包括不安全状态（谨慎使用）
export TT_METAL_WATCHER_DUMP_ALL=1
```

#### 禁用特定特性

```bash
export TT_METAL_WATCHER_DISABLE_ASSERT=1
export TT_METAL_WATCHER_DISABLE_PAUSE=1
export TT_METAL_WATCHER_DISABLE_RING_BUFFER=1
export TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1
export TT_METAL_WATCHER_DISABLE_WAYPOINT=1
export TT_METAL_WATCHER_DISABLE_STACK_USAGE=1
export TT_METAL_WATCHER_DISABLE_ETH_LINK_STATUS=1
```

#### 路点（代码跟踪）

在内核代码中添加路点以跟踪执行：

```cpp
#include "debug/waypoint.h"

void noc_semaphore_wait(volatile tt_l1_ptr uint32_t* sem_addr, uint32_t val) {
    WAYPOINT("NSW");  // 等待中
    while ((*sem_addr) != val)
        ;
    WAYPOINT("NSD");  // 完成
}
```

路点命名约定：
- `W` = 在循环顶部等待
- `D` = 循环结束后完成等待

#### 断言 (Asserts)

```cpp
#include "debug/assert.h"
#include "debug/waypoint.h"

void kernel_main() {
    uint32_t a = get_arg_val<uint32_t>(0);
    uint32_t b = get_arg_val<uint32_t>(1);

    WAYPOINT("AST1");
    ASSERT(a != b);  // 条件失败时触发错误
}
```

#### 暂停执行

```cpp
#include "debug/pause.h"

void kernel_main() {
    // 其他代码...
    PAUSE();  // 暂停直到用户按 ENTER
    // 内核其余部分...
}
```

#### 环形缓冲区 (Ring Buffer)

```cpp
#include "debug/ring_buffer.h"

void kernel_main() {
    for (uint32_t idx = 0; idx < 40; idx++) {
        WATCHER_RING_BUFFER_PUSH(idx+1);  // 推送值到调试缓冲区
    }
}
```

#### 堆栈使用测量

每次内核运行后自动测量堆栈使用。报告溢出错误并显示每个 RISC-V 核心的使用情况。

#### 调试延迟

为调试插入 NOC 事务延迟：

```bash
TT_METAL_WATCHER=1 \
TT_METAL_WATCHER_DEBUG_DELAY=10 \
TT_METAL_READ_DEBUG_DELAY_CORES=0,0 \
TT_METAL_WRITE_DEBUG_DELAY_CORES=0,0 \
TT_METAL_READ_DEBUG_DELAY_RISCVS=BR \
TT_METAL_WRITE_DEBUG_DELAY_RISCVS=BR \
./build/test/tt_metal/test_eltwise_binary
```

#### GDB 集成

无论是否启用 Watcher，都可以从 GDB 转储 Watcher 状态：

```gdb
thread 1           # 确保主线程存在
up                 # 导航到 "tt" 命名空间
call tt::watcher::dump(stderr, true)  # true = 包含硬件寄存器
```

### 3.4 DPRINT 调试

#### 概述

**DPRINT (Kernel Debug Print)** 用于从**设备到主机**打印瓦片、标量和字符串。

#### 头文件

```cpp
#include "api/debug/dprint.h"
```

#### 特性

- **直接打印**：字符串、字符、uint32_t、浮点数、BF16
- **格式化**：`SETPRECISION`、`FIXED`、`DEFAULTFLOAT`、`SETW`、`HEX`/`DEC`/`OCT`
- **瓦片打印**：`TSLICE()` 用于打印带切片范围的瓦片数据
- **核心特定打印**：`DPRINT_MATH`、`DPRINT_PACK`、`DPRINT_UNPACK`、`DPRINT_DATA0`、`DPRINT_DATA1`

#### 基本用法

```cpp
#include "api/debug/dprint.h"

void kernel_main() {
    DPRINT << "Hello from kernel!" << ENDL();
    DPRINT << "Value: " << 42 << ENDL();
}
```

#### 核心特定打印宏

| 宏 | 描述 |
|----|------|
| `DPRINT_MATH({ ... })` | 仅在 MATH RISC-V 上执行 |
| `DPRINT_PACK({ ... })` | 仅在 PACK RISC-V 上执行 |
| `DPRINT_UNPACK({ ... })` | 仅在 UNPACK RISC-V 上执行 |
| `DPRINT_DATA0({ ... })` | 仅在 DATA0 (BRISC) RISC-V 上执行 |
| `DPRINT_DATA1({ ... })` | 仅在 DATA1 (NCRISC) RISC-V 上执行 |

#### TileSlice (TSLICE) 瓦片打印

**TileSlice 构造函数参数：**

| 参数 | 类型 | 描述 |
|------|------|------|
| `cb_id` | `uint8_t` | 打印的循环缓冲区 ID |
| `tile_idx` | `int` | CB 内的瓦片索引 |
| `slice_range` | `SliceRange` | H/W 维度的起始/结束索引和步幅 |
| `cb_type` | `dprint_tslice_cb_t` | `TSLICE_INPUT_CB` 或 `TSLICE_OUTPUT_CB`（仅数据移动） |
| `ptr_type` | `dprint_tslice_ptr_t` | `TSLICE_RD_PTR` 或 `TSLICE_WR_PTR`（仅数据移动） |
| `endl_rows` | `bool` | 行间添加换行符（默认：`true`） |
| `print_untilized` | `bool` | 打印时反瓦片化数据（默认：`true`） |

**SliceRange 结构体字段：**

```cpp
SliceRange {
    .h0 = start_row,      // 起始行索引
    .h1 = end_row,        // 结束行索引（不包含）
    .hs = row_stride,     // 行步幅
    .w0 = start_col,      // 起始列索引
    .w1 = end_col,        // 结束列索引（不包含）
    .ws = col_stride      // 列步幅
};
```

**使用示例：**

```cpp
#include "api/debug/dprint.h"

// 简单切片使用 TSLICE 宏
DPRINT << TSLICE(CBIndex::c_25, 0, SliceRange::hw0_32_16()) << ENDL();

// 逐行打印完整瓦片
for (int32_t r = 0; r < 32; ++r) {
    SliceRange sr = SliceRange{.h0 = r, .h1 = r+1, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};

    // 数据移动 RISC - 指定 CB 类型和指针类型
    DPRINT_DATA0({ DPRINT << (uint)r << " --READ--cin1-- "
                   << TileSlice(0, 0, sr, TSLICE_INPUT_CB, TSLICE_RD_PTR, true, false) << ENDL(); });

    // Unpacker RISC - 仅从前面读取，仅输入 CB
    DPRINT_UNPACK({ DPRINT << (uint)r << " --READ--cin1-- "
                    << TileSlice(0, 0, sr, true, false) << ENDL(); });

    // Packer RISC - 仅写入后面
    DPRINT_PACK({ DPRINT << (uint)r << " --READ--cin1-- "
                  << TileSlice(0, 0, sr, true, false) << ENDL(); });
}
```

#### 重要约束

1. **时间**：打印必须发生在 CB API 调用之间：
   - **从 CB 读取**：`cb_wait_front()` 和 `cb_pop_front()` 之间
   - **写入 CB**：`cb_reserve_back()` 和 `cb_push_back()` 之间

2. **RISC 特定限制**：
   - **MATH 核心**：无法访问 CB（TSLICE 无效）
   - **UNPACK RISC**：只有 `rd_ptr`，只有输入 CB
   - **PACK RISC**：只有 `wr_ptr`

3. **支持的数据格式**：`Float32`、`Float16_b`、`Bfp8_b`、`Bfp4_b`、`Int8`、`UInt8`、`UInt16`、`Int32`、`UInt32`

4. **TileSlice 容量有限** - 完整瓦片一次打印一行

#### 缓冲区刷新

仅在以下情况刷新缓冲区：
- 调用 `ENDL()`
- 读取到 `\n`
- 设备关闭

### 3.5 工具互斥性

> **重要约束：互斥性**
>
> **分析、内核调试打印 (DPRINT) 和 Watcher 不能同时使用！**
>
> 这三个特性都使用大量 SRAM 进行数据存储，会相互冲突。确保 `TT_METAL_DPRINT_CORES`、`TT_METAL_WATCHER` 和 `TT_METAL_DEVICE_PROFILER` **不同时设置**。

---

## 参考资源

### 官方文档链接

- [TT-Metalium 文档](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/index.html)
- [Tracy Profiler 文档](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/tracy_profiler.html)
- [Device Program Profiler 文档](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/device_program_profiler.html)
- [Watcher 文档](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/watcher.html)
- [Kernel Debug Print 文档](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/kernel_print.html)
- [编程模型文档](https://docs.tenstorrent.com/tt-metal/v0.57.0/tt-metalium/tt_metal/programming_model/index.html)
- [编程示例](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/index.html)

### 技术报告

- `AdvancedPerformanceOptimizationsForModels/` (Metal Trace & Multi-CQ)
- `tech_reports/data_formats/data_formats.md`
- `tech_reports/tensor_layouts/tensor_layouts.md`
- `tech_reports/tensor_sharding/tensor_sharding.md`
- `tech_reports/memory/allocator.md`
- `tech_reports/matrix_engine/matrix_engine.md`

### 其他资源

- [HotChips 2024 演示](https://hc2024.hotchips.org/assets/program/conference/day1/88_HC2024.Tenstorrent.Jasmina.Davor.v7.pdf) - Blackhole & TT-Metalium 架构详解
- [Tracy 官方文档](https://github.com/wolfpld/tracy/releases/latest/download/tracy.pdf)
- [METALIUM_GUIDE.md](https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md)

---

*文档生成时间：2026-03-12*
*基于 Tenstorrent TT-Metalium 最新文档整理*
