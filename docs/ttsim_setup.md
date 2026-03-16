# TT-Sim 仿真器配置指南

## 概述

TT-Sim 是 Tenstorrent 的硬件仿真器，可以在没有实际硬件的情况下测试 TT-Metal 程序。

## 配置步骤

### 1. 下载 TT-Sim 库

从 GitHub Release 下载对应架构的库文件：

```bash
mkdir -p /work/ttsim
cd /work/ttsim

# 下载 Blackhole 版本
curl -L -o libttsim_bh.so "https://github.com/tenstorrent/ttsim/releases/download/v1.4.3/libttsim_bh.so"

# 创建符号链接
ln -s libttsim_bh.so libttsim.so
```

### 2. 复制 SoC 描述文件

```bash
cp $TT_METAL_HOME/tt_metal/soc_descriptors/blackhole_140_arch.yaml \
   /work/ttsim/soc_descriptor.yaml
```

**注意**: 使用原始配置即可，不需要修改 eth cores。官方编程示例可以正常工作。

### 3. 设置环境变量

```bash
export TT_METAL_SIMULATOR_HOME=/work/ttsim
export TT_METAL_SIMULATOR=/work/ttsim/libttsim.so
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_DISABLE_SFPLOADMACRO=1
```

## 环境变量说明

| 变量 | 说明 |
|------|------|
| `TT_METAL_SIMULATOR_HOME` | TT-Sim 主目录，包含 libttsim.so 和 soc_descriptor.yaml |
| `TT_METAL_SIMULATOR` | libttsim.so 的完整路径 |
| `TT_METAL_SLOW_DISPATCH_MODE` | 禁用 Fast Dispatch，仿真器只支持 Slow Dispatch |
| `TT_METAL_DISABLE_SFPLOADMACRO` | 禁用 SFPU 加载宏，避免仿真器不支持的功能 |

## 测试 TT-Sim

### 官方编程示例

编译并运行 TT-Metal 官方示例：

```bash
# 编译示例（启用 BUILD_PROGRAMMING_EXAMPLES）
cmake -B build -DBUILD_PROGRAMMING_EXAMPLES=ON -DENABLE_DISTRIBUTED=OFF ...
cmake --build build --target metal_example_add_2_integers_in_riscv

# 设置环境变量
export TT_METAL_HOME=/path/to/tt_metal
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME
export TT_METAL_SIMULATOR=$TT_METAL_HOME/sim/libttsim.so
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_DISABLE_SFPLOADMACRO=1

# 设置库路径（关键步骤）
export LD_LIBRARY_PATH=$TT_METAL_HOME/build_Release/tt_metal:\
$TT_METAL_HOME/build_Release/tt_stl:\
$TT_METAL_HOME/build_Release/ttnn:\
$TT_METAL_HOME/build_Release/lib:\
$TT_METAL_HOME/build_Release/tt_metal/third_party/umd/device:\
$TT_METAL_HOME/build_Release/_deps/fmt-build:\
$TT_METAL_HOME/build_Release/_deps/benchmark-build/src:\
$TT_METAL_HOME/sim:$LD_LIBRARY_PATH

# 运行示例
./build/programming_examples/metal_example_add_2_integers_in_riscv
# 预期输出: Success: Result is 21
```

### 运行多个官方示例测试

使用提供的测试脚本运行多个官方示例：

```bash
cd /root/dev/vibe_dsl/tests/tt_sim
./run_official_examples.sh
```

该脚本会自动运行以下测试：
- `metal_example_add_2_integers_in_riscv` - 基础 RISC-V 计算
- `metal_example_hello_world_datamovement_kernel` - 数据移动
- `metal_example_loopback` - 环回测试
- `metal_example_eltwise_binary` - 二元运算

## TT-Sim API 函数

TT-Sim 提供以下 C 接口函数：

```c
// 初始化和关闭
void libttsim_init(void);
void libttsim_exit(void);

// PCI 配置空间读取
uint32_t libttsim_pci_config_rd32(uint32_t bus_device_function, uint32_t offset);

// PCI 内存读写
void libttsim_pci_mem_rd_bytes(uint64_t paddr, void* p, uint32_t size);
void libttsim_pci_mem_wr_bytes(uint64_t paddr, const void* p, uint32_t size);

// Tile (核心) 内存读写
void libttsim_tile_rd_bytes(uint32_t x, uint32_t y, uint64_t addr, void* p, uint32_t size);
void libttsim_tile_wr_bytes(uint32_t x, uint32_t y, uint64_t addr, const void* p, uint32_t size);

// 时钟推进
void libttsim_clock(uint32_t n_clocks);
```

## Blackhole 架构信息

### Worker 核心布局

- **总数**: 140 个 Tensix 核心
- **排列**: 10 行 x 14 列 (实际分布在 17x12 网格中)
- **每行**: 14 个核心
- **行数**: 10 行 (y=2 到 y=11)
- **L1 大小**: 1.5 MB (1572864 bytes)

### 核心坐标映射

Worker 核心分布在以下坐标（translated 坐标系）：

```
行 2-11: x ∈ [1-7] ∪ [10-16] (跳过 x=8,9)
```

注意：x=8,9 是 DRAM/ARC/Eth 区域，不是 Tensix 核心。

## 与 TT-Metal 集成

TT-Metal 通过 UMD (Unified Metal Driver) 的 `TTSimChip` 和 `TTSimTTDevice` 类与 TT-Sim 集成：

```cpp
// 创建仿真设备
auto device = tt::umd::TTSimChip::create("/work/ttsim");

// 启动设备
device->start_device();

// 读写 L1 内存
device->write_to_device(core, src, l1_addr, size);
device->read_from_device(core, dest, l1_addr, size);

// 关闭设备
device->close_device();
```

## 已知限制

1. **只支持 Slow Dispatch**: Fast Dispatch 在仿真器中不可用
2. **SFPU 限制**: 某些 SFPU 指令可能不被支持
3. **DMA 操作**: 仿真器不支持 DMA 操作
4. **性能**: 仿真速度比实际硬件慢很多

## 版本信息

- TT-Sim 版本: v1.4.3
- 支持的架构: Wormhole B0, Blackhole
- GitHub: https://github.com/tenstorrent/ttsim
