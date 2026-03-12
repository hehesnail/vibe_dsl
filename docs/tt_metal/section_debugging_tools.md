# 调试与剖析工具详解

本文档详细介绍 TT-Metalium 框架的调试和性能分析工具，包括 Tracy Profiler、Device Profiler、Watcher 和 DPRINT 等工具的配置、使用方法以及高级调试技巧。

---

## 目录

1. [工具概览](#1-工具概览)
2. [Tracy Profiler 详细配置](#2-tracy-profiler-详细配置)
3. [Device Profiler 使用指南](#3-device-profiler-使用指南)
4. [Watcher 调试技巧](#4-watcher-调试技巧)
5. [DPRINT 调试方法](#5-dprint-调试方法)
6. [系统性调试方法](#6-系统性调试方法)
7. [高级调试技巧](#7-高级调试技巧)
8. [性能瓶颈分析](#8-性能瓶颈分析)
9. [工具互斥性说明](#9-工具互斥性说明)

---

## 1. 工具概览

TT-Metalium 提供四种主要调试和剖析工具，每种工具适用于不同的调试场景：

| 工具 | 用途 | 适用场景 | 环境变量 |
|------|------|----------|----------|
| **Tracy Profiler** | Host + Device 性能分析 | 性能瓶颈定位、调用链分析 | `TT_METAL_TRACY=1` |
| **Device Profiler** | Kernel 级计时分析 | 内核执行时间测量 | `TT_METAL_DEVICE_PROFILER=1` |
| **Watcher** | 运行时调试监控 | 死锁检测、状态监控 | `TT_METAL_WATCHER=120` |
| **DPRINT** | Kernel printf 调试 | 变量值检查、数据流验证 | `TT_METAL_DPRINT_CORES=all` |

**重要警告**: 这些工具互斥，不能同时运行。它们共享有限的片上 SRAM 资源，同时启用会导致冲突。

---

## 2. Tracy Profiler 详细配置

Tracy 是一个开源 C++ 性能分析工具，TT-Metalium 使用其分支版本适配 Tenstorrent 硬件。

### 2.1 构建配置

Tracy 支持在构建 Metalium 时默认启用：

```bash
# 通过构建脚本（默认启用）
./build_metal.sh

# 或通过 CMake 标志显式启用
cmake . -DENABLE_TRACY=ON
ninja
ninja install
```

旧版本使用：
```bash
./build_metal.sh --enable-profiler
```

### 2.2 C++ 主机端分析

在 C++ 代码中使用 Tracy 宏标记分析区域：

```cpp
#include <tracy/Tracy.hpp>

// 构造函数分析示例
Device::Device(chip_id_t device_id, const std::vector<uint32_t>& l1_bank_remap) : id_(device_id)
{
    ZoneScoped;  // 自动分析此作用域
    this->initialize(l1_bank_remap);
}

// 命名区域分析
void performComputation() {
    ZoneScopedN("MatrixMultiplication");
    // 计算代码...
}

// 嵌套区域分析
void complexOperation() {
    ZoneScopedN("ComplexOperation");

    {
        ZoneScopedN("DataLoad");
        // 数据加载代码...
    }

    {
        ZoneScopedN("Computation");
        // 计算代码...
    }
}
```

**常用 C++ 宏**：

| 宏 | 描述 |
|----|------|
| `ZoneScoped;` | 分析当前作用域 |
| `ZoneScopedN("name");` | 命名区域分析 |
| `ZoneScopedC(color);` | 带颜色的区域分析 |
| `FrameMark;` | 标记帧边界 |
| `TracyPlot("name", value);` | 绘制数值图表 |

### 2.3 Python 主机端分析

TT Metal 使用 Python 标准的 `sys.setprofile` 和 `sys.settrace` 函数集成 Python 分析。

**方法 1：命令行分析（整个脚本）**

```bash
python -m tracy script.py
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

### 2.4 行级分析

启用行级分析以获取更细粒度的性能数据：

```bash
python -m tracy -p -l -m pytest {test_path}
```

添加 `-l` 标志启用行级分析，可以精确到代码行的性能热点识别。

### 2.5 GUI 设置和操作

Tracy 需要桌面 GUI 应用程序查看分析结果。

**macOS 安装**：
```bash
brew uninstall tracy  # 移除旧版本
wget -P ~/ --no-check-certificate --no-cache --no-cookies https://raw.githubusercontent.com/tenstorrent-metal/tracy/master/tracy.rb
brew install ~/tracy.rb
rm ~/tracy.rb
```

**Linux 安装**：
```bash
# 从源码编译
git clone https://github.com/wolfpld/tracy.git
cd tracy
mkdir build && cd build
cmake ..
make -j
sudo make install
```

**连接和查看**：
1. 启动 Tracy GUI: `tracy-profiler`
2. 将客户端地址设置为远程机器 IP 和端口 8086（例如 `172.27.28.132:8086`）
3. 点击连接 - 分析的应用程序启动后数据将实时流式传输

**GUI 操作技巧**：
- 使用鼠标滚轮缩放时间轴
- 点击区域查看详细信息
- 使用搜索功能定位特定函数
- 查看统计面板了解热点函数

### 2.6 设备端 Kernel 标记

在 Kernel 代码中使用 DeviceZoneScopedN 进行设备端分析：

```cpp
#include <tools/profiler/kernel_profiler.hpp>

void kernel_main() {
    DeviceZoneScopedN("DataMovementKernel");

    DeviceZoneScopedN("ReadPhase");
    // 读取代码

    DeviceZoneScopedN("WritePhase");
    // 写入代码
}
```

---

## 3. Device Profiler 使用指南

Device Program Profiler 用于 Tenstorrent 硬件上的内核性能分析，包括计算和数据移动内核的所有 RISC-V 核心。

### 3.1 启用分析器

```bash
# 设置环境变量
export TT_METAL_DEVICE_PROFILER=1

# 运行应用程序
./your_tt_metal_application

# 或用于 TT-NN 操作
pytest tests/path/to/test.py
```

### 3.2 ZoneScoped 宏使用

在 Kernel 代码中使用 `DeviceZoneScopedN` 宏标记分析区域：

```cpp
#include "debug/zone.h"

void kernel_main() {
    DeviceZoneScopedN("CUSTOM-ZONE");

    // 记录特定区域
    {
        DeviceZoneScopedN("MatmulLoop");
        for (int i = 0; i < iterations; i++) {
            // matmul 代码
        }
    }

    {
        DeviceZoneScopedN("DataTransfer");
        // 数据传输代码
    }
}
```

**宏使用最佳实践**：
- 在关键循环外部使用，避免过多开销
- 使用描述性命名，便于识别
- 嵌套区域不要超过 3-4 层

### 3.3 CSV 输出结构解析

分析器生成名为 `profile_log_device.csv` 的文件，位于：
```
${TT_METAL_HOME}/generated/profiler/.logs/
```

**CSV 头部格式（Wormhole/Grayskull）**：
```
ARCH: grayskull, CHIP_FREQ[MHz]: 1202
PCIe slot, core_x, core_y, RISC processor type, timer_id, time[cycles since reset], stat value, Run ID, zone name, zone phase, source line, source file
```

**CSV 头部格式（Blackhole）**：
```
ARCH: blackhole, CHIP_FREQ[MHz]: 1350, Max Compute Cores: 120
PCIe slot, core_x, core_y, RISC processor type, timer_id, time[cycles since reset], data, run host ID, trace id, trace id counter, zone name, type, source line, source file, meta data
```

**列描述**：

| 列 | 描述 |
|----|------|
| **PCIe slot** | 设备 PCIe 槽位标识符 |
| **core_x, core_y** | 芯片上的核心坐标 |
| **RISC processor type** | 处理器类型：`BRISC`、`NCRISC` 或 `TRISC` |
| **timer_id** | 定时器事件的唯一标识符 |
| **time[cycles since reset]** | 设备重置后的周期计数 |
| **zone name** | 分析区域标识符 |
| **zone phase** | `begin` 或 `end` 标记 |
| **source line** | 源代码行号 |
| **source file** | 源文件名 |

### 3.4 默认分析区域

| 区域 | 描述 |
|------|------|
| `BRISC-FW` / `NCRISC-FW` / `TRISC-FW` | 固件循环的单次迭代持续时间 |
| `BRISC-KERNEL` / `NCRISC-KERNEL` / `TRISC-KERNEL` | 内核 `main()` 函数执行持续时间 |

### 3.5 性能指标分析方法

**计算固件执行时间**：
```python
# 找到最小和最大 time[cycles since reset] 值
elapsed_cycles = max_time - min_time
# 使用头部中的芯片频率转换为秒
elapsed_time = elapsed_cycles / (CHIP_FREQ_MHz * 1e6)
```

**区域持续时间分析（Python 示例）**：
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
print(f"Min duration: {durations_us.min():.2f} µs")
print(f"Max duration: {durations_us.max():.2f} µs")
```

**多核心性能对比**：
```python
# 按核心分组分析
for (x, y), group in df.groupby(['core_x', 'core_y']):
    kernel_time = group[group['zone name'] == 'BRISC-KERNEL']
    if not kernel_time.empty:
        duration = kernel_time['time[cycles since reset]'].diff().dropna()
        print(f"Core ({x}, {y}): {duration.mean() / 1202:.2f} µs")
```

### 3.6 重要考虑事项

1. **构建要求**：使用 `--build-type Release` 进行准确的性能测量
2. **DPRINT 冲突**：分析时禁用 `TT_METAL_DPRINT_CORES` - 它们共享有限的片上 SRAM
3. **RISC-V 处理器**：
   - `BRISC` / `NCRISC`：控制路由器（RISC-V 0 和 4）
   - `TRISC`：Tensix 计算处理器（RISC-V 1-3）

---

## 4. Watcher 调试技巧

Watcher 是 TT-Metalium 的调试工具，包含插桩组件和主机监控线程，用于捕获常见编程错误。

### 4.1 启用 Watcher

设置这些环境变量：

```bash
# 必需：设置轮询间隔（秒）（越长 = 侵入性越小）
export TT_METAL_WATCHER=120

# 可选：追加到现有日志文件而非创建新文件
export TT_METAL_WATCHER_APPEND=1

# 可选：转储所有状态包括不安全状态（谨慎使用）
export TT_METAL_WATCHER_DUMP_ALL=1
```

### 4.2 禁用特定特性

```bash
export TT_METAL_WATCHER_DISABLE_ASSERT=1        # 禁用断言检查
export TT_METAL_WATCHER_DISABLE_PAUSE=1         # 禁用暂停功能
export TT_METAL_WATCHER_DISABLE_RING_BUFFER=1   # 禁用环形缓冲区
export TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1  # 禁用 NOC 清理
export TT_METAL_WATCHER_DISABLE_WAYPOINT=1      # 禁用路点跟踪
export TT_METAL_WATCHER_DISABLE_STACK_USAGE=1   # 禁用堆栈使用测量
export TT_METAL_WATCHER_DISABLE_ETH_LINK_STATUS=1  # 禁用以太网链路状态
```

### 4.3 路点跟踪

在内核代码中添加路点以跟踪执行：

```cpp
#include "debug/waypoint.h"

void noc_semaphore_wait(volatile tt_l1_ptr uint32_t* sem_addr, uint32_t val) {
    WAYPOINT("NSW");  // NOC Semaphore Wait - 等待中
    while ((*sem_addr) != val)
        ;
    WAYPOINT("NSD");  // NOC Semaphore Done - 完成
}
```

**路点命名约定**：
- `W` = 在循环顶部等待 (Waiting)
- `D` = 循环结束后完成 (Done)
- `A` = 断言检查点 (Assert)
- `P` = 暂停点 (Pause)

**常用路点模式**：
```cpp
void kernel_main() {
    WAYPOINT("INI");  // 初始化开始

    // 初始化代码...

    WAYPOINT("RDY");  // 准备就绪

    for (uint32_t i = 0; i < num_tiles; i++) {
        WAYPOINT("CBW");  // CB 等待
        cb_wait_front(cb_id, 1);
        WAYPOINT("CBD");  // CB 完成等待

        // 处理代码...

        WAYPOINT("CBP");  // CB 推送
        cb_push_back(cb_id, 1);
    }

    WAYPOINT("END");  // 内核结束
}
```

### 4.4 断言调试

```cpp
#include "debug/assert.h"
#include "debug/waypoint.h"

void kernel_main() {
    uint32_t a = get_arg_val<uint32_t>(0);
    uint32_t b = get_arg_val<uint32_t>(1);

    WAYPOINT("AST1");
    ASSERT(a != b);  // 条件失败时触发错误

    WAYPOINT("AST2");
    ASSERT(num_tiles > 0 && num_tiles <= MAX_TILES);
}
```

**断言最佳实践**：
- 在关键检查点前设置路点
- 使用描述性路点名称
- 避免在频繁执行的循环中使用断言

### 4.5 暂停执行

```cpp
#include "debug/pause.h"

void kernel_main() {
    // 其他代码...
    PAUSE();  // 暂停直到用户按 ENTER
    // 内核其余部分...
}
```

暂停功能用于交互式调试，可以在特定点检查系统状态。

### 4.6 环形缓冲区监控

```cpp
#include "debug/ring_buffer.h"

void kernel_main() {
    for (uint32_t idx = 0; idx < 40; idx++) {
        WATCHER_RING_BUFFER_PUSH(idx+1);  // 推送值到调试缓冲区
    }
}
```

环形缓冲区用于记录运行时变量值，可以在 Watcher 输出中查看历史值。

### 4.7 堆栈使用测量

每次内核运行后自动测量堆栈使用。报告溢出错误并显示每个 RISC-V 核心的使用情况。

**查看堆栈使用**：
- 在 Watcher 输出中查找 `STACK` 相关行
- 检查是否有 `OVERFLOW` 警告
- 优化大数组分配以避免溢出

### 4.8 调试延迟

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

### 4.9 GDB 集成

无论是否启用 Watcher，都可以从 GDB 转储 Watcher 状态：

```gdb
thread 1           # 确保主线程存在
up                 # 导航到 "tt" 命名空间
call tt::watcher::dump(stderr, true)  # true = 包含硬件寄存器
```

**GDB 调试脚本示例**：
```gdb
# 保存为 debug.gdb
set pagination off
break kernel_main
run

# 当命中断点时，转储 Watcher 状态
up
call tt::watcher::dump(stderr, true)

# 继续执行
continue
```

---

## 5. DPRINT 调试方法

DPRINT (Kernel Debug Print) 用于从设备到主机打印瓦片、标量和字符串。

### 5.1 头文件和基本用法

```cpp
#include "api/debug/dprint.h"

void kernel_main() {
    DPRINT << "Hello from kernel!" << ENDL();
    DPRINT << "Value: " << 42 << ENDL();
    DPRINT << "Float: " << 3.14159f << ENDL();
}
```

**启用 DPRINT**：
```bash
# 打印所有核心
export TT_METAL_DPRINT_CORES=all

# 或指定特定核心
export TT_METAL_DPRINT_CORES=1,1

# 多个核心
export TT_METAL_DPRINT_CORES="0,0;1,1;2,2"
```

### 5.2 核心特定打印宏

| 宏 | 描述 | 适用 Kernel 类型 |
|----|------|-----------------|
| `DPRINT_MATH({ ... })` | 仅在 MATH RISC-V 上执行 | Compute Kernel |
| `DPRINT_PACK({ ... })` | 仅在 PACK RISC-V 上执行 | Compute Kernel |
| `DPRINT_UNPACK({ ... })` | 仅在 UNPACK RISC-V 上执行 | Compute Kernel |
| `DPRINT_DATA0({ ... })` | 仅在 DATA0 (BRISC) RISC-V 上执行 | Data Movement Kernel |
| `DPRINT_DATA1({ ... })` | 仅在 DATA1 (NCRISC) RISC-V 上执行 | Data Movement Kernel |

**使用示例**：
```cpp
void kernel_main() {
    // 所有核心都会执行
    DPRINT << "Common message" << ENDL();

    // 仅特定核心执行
    DPRINT_MATH({
        DPRINT << "Math core executing" << ENDL();
    });

    DPRINT_UNPACK({
        DPRINT << "Unpack core: tile ready" << ENDL();
    });

    DPRINT_DATA0({
        DPRINT << "BRISC: Reading from DRAM" << ENDL();
    });
}
```

### 5.3 TileSlice 瓦片打印

TileSlice 用于打印 CB 中的瓦片数据。

**TileSlice 构造函数参数**：

| 参数 | 类型 | 描述 |
|------|------|------|
| `cb_id` | `uint8_t` | 打印的循环缓冲区 ID |
| `tile_idx` | `int` | CB 内的瓦片索引 |
| `slice_range` | `SliceRange` | H/W 维度的起始/结束索引和步幅 |
| `cb_type` | `dprint_tslice_cb_t` | `TSLICE_INPUT_CB` 或 `TSLICE_OUTPUT_CB`（仅数据移动） |
| `ptr_type` | `dprint_tslice_ptr_t` | `TSLICE_RD_PTR` 或 `TSLICE_WR_PTR`（仅数据移动） |
| `endl_rows` | `bool` | 行间添加换行符（默认：`true`） |
| `print_untilized` | `bool` | 打印时反瓦片化数据（默认：`true`） |

**SliceRange 结构体**：
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

**预定义 SliceRange**：
- `SliceRange::hw0_32_16()` - 完整 32x32 瓦片，步幅 16
- `SliceRange::hw0_32_1()` - 完整 32x32 瓦片，步幅 1
- `SliceRange::hw0_16_1()` - 16x16 区域

**使用示例**：
```cpp
#include "api/debug/dprint.h"

void kernel_main() {
    // 简单切片使用 TSLICE 宏
    DPRINT << TSLICE(CBIndex::c_25, 0, SliceRange::hw0_32_16()) << ENDL();

    // 逐行打印完整瓦片
    for (int32_t r = 0; r < 32; ++r) {
        SliceRange sr = SliceRange{.h0 = r, .h1 = r+1, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};

        // 数据移动 RISC - 指定 CB 类型和指针类型
        DPRINT_DATA0({
            DPRINT << (uint)r << " --READ--cin1-- "
                   << TileSlice(0, 0, sr, TSLICE_INPUT_CB, TSLICE_RD_PTR, true, false)
                   << ENDL();
        });

        // Unpacker RISC - 仅从前面读取，仅输入 CB
        DPRINT_UNPACK({
            DPRINT << (uint)r << " --READ--cin1-- "
                   << TileSlice(0, 0, sr, true, false)
                   << ENDL();
        });

        // Packer RISC - 仅写入后面
        DPRINT_PACK({
            DPRINT << (uint)r << " --READ--cin1-- "
                   << TileSlice(0, 0, sr, true, false)
                   << ENDL();
        });
    }
}
```

### 5.4 格式化选项

```cpp
DPRINT << SETPRECISION(4) << 3.14159265f << ENDL();  // 设置精度
DPRINT << FIXED << 3.14159265f << ENDL();            // 固定小数位
DPRINT << DEFAULTFLOAT << 3.14159265f << ENDL();     // 默认格式
DPRINT << SETW(32) << "Formatted" << ENDL();         // 设置宽度
DPRINT << HEX << 255 << ENDL();                      // 十六进制
DPRINT << DEC << 255 << ENDL();                      // 十进制
DPRINT << OCT << 255 << ENDL();                      // 八进制
```

### 5.5 条件打印技巧

```cpp
void kernel_main() {
    uint32_t tile_id = get_arg_val<uint32_t>(0);

    // 条件打印 - 仅打印特定 tile
    if (tile_id == 0 || tile_id == num_tiles - 1) {
        DPRINT << "Tile " << tile_id << " of " << num_tiles << ENDL();
    }

    // 周期性打印 - 每 N 个 tile 打印一次
    if (tile_id % 100 == 0) {
        DPRINT << "Progress: " << tile_id << "/" << num_tiles << ENDL();
    }

    // 错误条件打印
    float value = get_value();
    if (value < 0.0f || value > 1.0f) {
        DPRINT << "ERROR: Invalid value " << value << " at tile " << tile_id << ENDL();
    }
}
```

### 5.6 重要约束

1. **时间**：打印必须发生在 CB API 调用之间：
   - **从 CB 读取**：`cb_wait_front()` 和 `cb_pop_front()` 之间
   - **写入 CB**：`cb_reserve_back()` 和 `cb_push_back()` 之间

2. **RISC 特定限制**：
   - **MATH 核心**：无法访问 CB（TSLICE 无效）
   - **UNPACK RISC**：只有 `rd_ptr`，只有输入 CB
   - **PACK RISC**：只有 `wr_ptr`

3. **支持的数据格式**：`Float32`、`Float16_b`、`Bfp8_b`、`Bfp4_b`、`Int8`、`UInt8`、`UInt16`、`Int32`、`UInt32`

4. **TileSlice 容量有限** - 完整瓦片一次打印一行

### 5.7 缓冲区刷新

仅在以下情况刷新缓冲区：
- 调用 `ENDL()`
- 读取到 `\n`
- 设备关闭

**确保输出可见**：
```cpp
DPRINT << "Important message" << ENDL();  // 使用 ENDL() 刷新
```

---

## 6. 系统性调试方法

### 6.1 调试方法论概述

系统性调试遵循以下流程：

```
1. 问题识别
   ↓
2. 信息收集（日志、状态、复现步骤）
   ↓
3. 假设形成（可能的原因）
   ↓
4. 验证假设（使用调试工具）
   ↓
5. 修复验证
   ↓
6. 回归测试
```

### 6.2 常见错误代码手册

#### 主机端错误

| 错误代码/消息 | 可能原因 | 解决方案 |
|--------------|----------|----------|
| `TT_ASSERT @ ... failed` | 断言失败 | 检查输入参数和状态 |
| `Device not found` | 设备未连接或驱动问题 | 检查硬件连接和驱动 |
| `Out of memory` | 内存分配失败 | 减少缓冲区大小或分批处理 |
| `Timeout waiting for kernel` | 内核死锁或无限循环 | 使用 Watcher 检查路点 |
| `NOC address out of range` | 无效 NOC 地址 | 验证坐标和地址计算 |

#### 设备端错误

| 错误代码/消息 | 可能原因 | 解决方案 |
|--------------|----------|----------|
| `CB overflow` | Circular Buffer 溢出 | 增加 CB 大小或减少推送 |
| `CB underflow` | Circular Buffer 下溢 | 检查 pop/push 平衡 |
| `Stack overflow` | 堆栈溢出 | 减少局部变量或大数组 |
| `NOC transaction error` | NOC 传输错误 | 检查地址对齐和大小 |
| `Invalid kernel args` | 内核参数错误 | 验证运行时参数 |

#### 同步错误

| 错误代码/消息 | 可能原因 | 解决方案 |
|--------------|----------|----------|
| `Semaphore timeout` | 信号量等待超时 | 检查信号量初始值和递增 |
| `Barrier mismatch` | 屏障同步失败 | 确保所有核心到达屏障 |
| `Deadlock detected` | 死锁 | 使用 Watcher 检查路点循环 |

### 6.3 问题排查流程图

```
程序崩溃/异常
    │
    ├── 主机端崩溃？
    │       ├── 是 → 检查 TT_ASSERT 消息
    │       │           ├── 参数错误 → 验证输入
    │       │           ├── 空指针 → 检查初始化
    │       │           └── 其他 → 查看堆栈跟踪
    │       └── 否 → 继续
    │
    ├── 设备端挂起？
    │       ├── 是 → 启用 Watcher
    │       │           ├── 查看路点状态
    │       │           ├── 检查 CB 状态
    │       │           └── 验证 NOC 事务
    │       └── 否 → 继续
    │
    ├── 结果错误？
    │       ├── 是 → 启用 DPRINT
    │       │           ├── 打印中间值
    │       │           ├── 检查 Tile 数据
    │       │           └── 验证计算步骤
    │       └── 否 → 继续
    │
    └── 性能问题？
            ├── 是 → 使用 Profilers
            │           ├── Tracy 分析主机端
            │           ├── Device Profiler 分析内核
            │           └── 识别热点和瓶颈
            └── 否 → 查看日志和文档
```

### 6.4 调试检查清单

**编译前检查**：
- [ ] 代码通过编译器警告检查
- [ ] 所有内核参数正确传递
- [ ] CB 大小和索引正确配置
- [ ] NOC 坐标计算正确

**运行时检查**：
- [ ] 设备初始化成功
- [ ] 缓冲区分配成功
- [ ] 内核编译无错误
- [ ] 运行时参数在有效范围

**调试时检查**：
- [ ] 使用正确的调试工具
- [ ] 环境变量设置正确
- [ ] 输出日志可访问
- [ ] 核心选择正确

---

## 7. 高级调试技巧

### 7.1 GDB 调试设备内核

**启动 GDB**：
```bash
gdb ./your_application
```

**常用 GDB 命令**：
```gdb
# 设置断点
break kernel_main
break reader.cpp:42

# 运行程序
run

# 查看堆栈
backtrace

# 查看变量
print variable_name
print/x variable_name  # 十六进制

# 单步执行
step
next

# 继续执行
continue

# 查看 Watcher 状态
up
call tt::watcher::dump(stderr, true)
```

**调试内核挂起**：
```gdb
# 当程序挂起时，按 Ctrl+C
call tt::watcher::dump(stderr, true)

# 查看特定核心状态
print device->cores_[{x,y}]->risc_v_[0]->waypoint_
```

### 7.2 死锁检测和解决

**常见死锁模式**：

1. **CB 死锁**：Reader 等待 CB 空间，Writer 等待 CB 数据
```cpp
// 错误示例 - 死锁
void reader() {
    cb_reserve_back(cb_out, 1);  // 等待输出空间
    // 但 writer 还没有 pop，因为它在等待输入
}

void writer() {
    cb_wait_front(cb_in, 1);  // 等待输入数据
    // 但 reader 还没有 push，因为它在等待输出空间
}
```

**检测方法**：
```bash
# 启用 Watcher 查看路点
export TT_METAL_WATCHER=10

# 查看路点输出
# 如果看到 "CBW" 但没有 "CBD"，可能是 CB 死锁
```

**解决方案**：
- 确保 CB 大小足够
- 检查 push/pop 顺序
- 使用双缓冲模式

2. **信号量死锁**：信号量等待永远不会满足
```cpp
// 检测信号量死锁
WAYPOINT("SW");  // Semaphore Wait
uint32_t timeout = 1000000;
while (*sem_addr != expected_val && timeout-- > 0);
if (timeout == 0) {
    DPRINT << "Semaphore timeout!" << ENDL();
    ASSERT(false);
}
WAYPOINT("SD");  // Semaphore Done
```

### 7.3 内存泄漏检测

**主机端内存泄漏检测**：
```bash
# 使用 Valgrind
valgrind --leak-check=full --show-leak-kinds=all ./your_application

# 使用 AddressSanitizer
./build_metal.sh --enable-asan
```

**设备端内存检查**：
```cpp
// 在程序结束时检查缓冲区状态
void cleanup() {
    // 确保所有缓冲区已释放
    DeallocateBuffer(buffer);

    // 检查设备内存状态
    DPRINT << "L1 used: " << get_l1_used() << ENDL();
}
```

### 7.4 内存越界检测

**使用 Watcher NOC 清理**：
```bash
# 启用 NOC 地址检查
export TT_METAL_WATCHER=1
# 确保 TT_METAL_WATCHER_DISABLE_NOC_SANITIZE 未设置
```

**手动边界检查**：
```cpp
void kernel_main() {
    uint32_t l1_addr = get_write_ptr(cb_id);
    uint32_t l1_size = get_cb_size(cb_id);

    // 检查写入是否在边界内
    ASSERT(l1_addr >= L1_BASE_ADDR);
    ASSERT(l1_addr + write_size <= L1_BASE_ADDR + l1_size);

    // 执行写入
    noc_async_write(src, l1_addr, write_size);
}
```

**GDB 检查内存**：
```gdb
# 查看 L1 内存
x/32x 0x10000000  # 查看 L1 起始地址的 32 个字

# 查看特定变量地址
print &variable
x/16x &variable
```

### 7.5 CB 溢出处理

**检测 CB 溢出**：
```cpp
#include "debug/assert.h"

void kernel_main() {
    // 检查 CB 可用空间
    uint32_t available = cb_pages_reservable_at_back(cb_id);
    ASSERT(available >= num_pages_needed);

    cb_reserve_back(cb_id, num_pages_needed);
    // ... 写入数据 ...
    cb_push_back(cb_id, num_pages_needed);
}
```

**CB 大小计算**：
```cpp
// 计算所需 CB 大小
uint32_t tile_size = 32 * 32 * element_size;  // 2048 bytes for Float16
uint32_t num_tiles = 4;  // 双缓冲
uint32_t cb_size = num_tiles * tile_size;  // 8192 bytes

// 确保不超过 L1 限制
ASSERT(cb_size <= MAX_L1_SIZE - RESERVED_SPACE);
```

**调试 CB 状态**：
```cpp
void debug_cb_state(uint32_t cb_id) {
    DPRINT << "CB " << cb_id << " state:" << ENDL();
    DPRINT << "  Pages available: " << cb_pages_available_at_front(cb_id) << ENDL();
    DPRINT << "  Pages reservable: " << cb_pages_reservable_at_back(cb_id) << ENDL();
    DPRINT << "  Read ptr: " << get_read_ptr(cb_id) << ENDL();
    DPRINT << "  Write ptr: " << get_write_ptr(cb_id) << ENDL();
}
```

---

## 8. 性能瓶颈分析

### 8.1 系统性性能分析方法

**分析流程**：

1. **建立基线**
   ```bash
   # 使用 Release 构建
   ./build_metal.sh --build-type Release

   # 运行基准测试并记录时间
   time ./benchmark
   ```

2. **识别瓶颈类型**
   - 主机端瓶颈：使用 Tracy Profiler
   - 设备端瓶颈：使用 Device Profiler
   - 内存瓶颈：检查 DRAM 带宽利用率
   - 计算瓶颈：检查 FPU 利用率

3. **收集详细数据**
   ```bash
   # 主机端分析
   python -m tracy benchmark.py

   # 设备端分析
   export TT_METAL_DEVICE_PROFILER=1
   ./benchmark
   ```

4. **分析热点**
   - 查看 Tracy 的统计面板
   - 分析 Device Profiler 的 CSV 输出
   - 识别耗时最长的区域

### 8.2 热点识别

**使用 Tracy 识别热点**：
```
1. 运行 Tracy 分析
2. 查看 "Statistics" 面板
3. 按 "Total time" 排序
4. 识别耗时最长的函数
```

**使用 Device Profiler 识别热点**：
```python
import pandas as pd

df = pd.read_csv('profile_log_device.csv', skiprows=1)

# 按区域名称分组计算总时间
zone_times = df.groupby('zone name').apply(
    lambda x: x[x['zone phase'] == 'end']['time[cycles since reset]'].values -
              x[x['zone phase'] == 'begin']['time[cycles since reset]'].values
)

# 找出最耗时的区域
for zone, times in zone_times.items():
    total_cycles = sum(times)
    print(f"{zone}: {total_cycles / 1202:.2f} µs")
```

**常见热点模式**：

| 热点类型 | 特征 | 优化方向 |
|----------|------|----------|
| DRAM 带宽受限 | 大量 noc_async_read/write | 使用 L1 缓存、双缓冲 |
| 计算受限 | 长时间 matmul 操作 | 优化 Math Fidelity、并行化 |
| 同步受限 | 长时间 barrier 等待 | 减少同步点、异步流水线 |
| 主机开销 | 主机端函数耗时 | 使用 Metal Trace |

### 8.3 优化效果验证

**验证步骤**：

1. **性能对比**
   ```bash
   # 优化前
   time ./benchmark > baseline.txt

   # 优化后
   time ./benchmark > optimized.txt

   # 对比
   diff baseline.txt optimized.txt
   ```

2. **精度验证**
   ```python
   # 计算 PCC (Pearson Correlation Coefficient)
   from scipy.stats import pearsonr

   pcc, _ = pearsonr(baseline_output.flatten(), optimized_output.flatten())
   assert pcc >= 0.99, f"PCC too low: {pcc}"
   ```

3. **回归测试**
   ```bash
   # 运行完整测试套件
   pytest tests/ -xvs
   ```

**性能报告模板**：
```
优化项目: [名称]
优化前性能: [X] µs
优化后性能: [Y] µs
提升比例: [(X-Y)/X * 100]%
PCC: [值]
验证状态: [通过/失败]
```

---

## 9. 工具互斥性说明

### 9.1 互斥原因

**Profiling/DPRINT/Watcher 不能同时使用**，原因如下：

1. **共享 SRAM 资源**：这三个特性都使用大量片上 SRAM 进行数据存储
2. **硬件资源冲突**：它们竞争相同的硬件调试资源
3. **性能干扰**：同时启用会导致严重的性能干扰

### 9.2 工具选择指南

根据调试目标选择合适的工具：

| 调试目标 | 推荐工具 | 替代方案 |
|----------|----------|----------|
| 主机端性能瓶颈 | Tracy Profiler | 手动计时 |
| 内核执行时间 | Device Profiler | Watcher 路点 |
| 内核死锁/挂起 | Watcher | DPRINT |
| 变量值检查 | DPRINT | Watcher 环形缓冲区 |
| 数据流验证 | DPRINT | 断言检查 |
| 内存越界检测 | Watcher | 手动边界检查 |
| 堆栈溢出检测 | Watcher | 减少栈使用 |

### 9.3 工具切换流程

```bash
# 1. 清除所有调试环境变量
unset TT_METAL_DEVICE_PROFILER
unset TT_METAL_WATCHER
unset TT_METAL_DPRINT_CORES
unset TT_METAL_TRACY

# 2. 设置需要的工具
export TT_METAL_DEVICE_PROFILER=1

# 3. 重新构建（如果需要）
./build_metal.sh

# 4. 运行测试
./your_application
```

### 9.4 组合使用策略

虽然不能同时启用，但可以分阶段使用：

**阶段 1：功能验证**
```bash
export TT_METAL_DPRINT_CORES=all
# 验证正确性
```

**阶段 2：死锁检测**
```bash
unset TT_METAL_DPRINT_CORES
export TT_METAL_WATCHER=120
# 检查死锁
```

**阶段 3：性能分析**
```bash
unset TT_METAL_WATCHER
export TT_METAL_DEVICE_PROFILER=1
# 收集性能数据
```

**阶段 4：主机端优化**
```bash
unset TT_METAL_DEVICE_PROFILER
# 使用 Tracy
python -m tracy script.py
```

---

## 参考资源

### 官方文档链接

- [TT-Metalium 文档](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/index.html)
- [Tracy Profiler 文档](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/tracy_profiler.html)
- [Device Program Profiler 文档](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/device_program_profiler.html)
- [Watcher 文档](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/watcher.html)
- [Kernel Debug Print 文档](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/kernel_print.html)

### 技术报告

- `tech_reports/profiling/` - 性能分析技术报告
- `tech_reports/debugging/` - 调试技术报告

### 其他资源

- [Tracy 官方文档](https://github.com/wolfpld/tracy/releases/latest/download/tracy.pdf)
- [METALIUM_GUIDE.md](https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md)

---

*文档生成时间：2026-03-12*
*基于 Tenstorrent TT-Metalium 最新文档整理*
