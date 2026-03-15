# Phase 0: TT-Sim 仿真器配置

## 任务目标

配置 TT-Sim 硬件仿真器环境，用于在无硬件条件下验证 TileLang Blackhole 后端功能。

## 背景知识

TT-Sim 是 Tenstorrent 提供的硬件仿真器，可以模拟 Blackhole/Wormhole 芯片行为。它是一个独立的共享库（`libttsim.so`），通过 UMD (User Mode Driver) 接口与 TT-Metal 集成。

## 技术方案

### 方案对比

| 方案 | 说明 | 选择 |
|------|------|------|
| 方案1 | 从源码编译 ttsim | ❌ 不开源，无法编译 |
| 方案2 | 从 GitHub Releases 下载预编译库 | ✅ 官方推荐方式 |

### 环境变量配置

```bash
# 必需环境变量
export TT_METAL_SIMULATOR_HOME=/path/to/sim
export TT_METAL_SIMULATOR=$TT_METAL_SIMULATOR_HOME/libttsim.so
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_DISABLE_SFPLOADMACRO=1

# UMD 测试需要（如果使用 UMD 单元测试）
export TT_UMD_SIMULATOR=$TT_METAL_SIMULATOR

# 库路径（运行时）
export LD_LIBRARY_PATH=$TT_METAL_HOME/build_Release/tt_metal:\
$TT_METAL_HOME/build_Release/tt_stl:\
$TT_METAL_HOME/build_Release/ttnn:\
$TT_METAL_HOME/build_Release/lib:\
$TT_METAL_HOME/build_Release/tt_metal/third_party/umd/device:\
$TT_METAL_HOME/sim:$LD_LIBRARY_PATH
```

### 文件结构

```
$TT_METAL_SIMULATOR_HOME/
├── libttsim.so              # 仿真器库（下载）
└── soc_descriptor.yaml      # Blackhole 架构描述（复制）
```

## 实施步骤

### 1. 下载 libttsim_bh.so

从 tenstorrent/ttsim 仓库 releases 下载：
- 版本: v1.4.3 (最新版本)
- 文件: libttsim_bh.so (Blackhole 版本)
- URL: https://github.com/tenstorrent/ttsim/releases

```bash
mkdir -p /work/ttsim
cd /work/ttsim
curl -L -o libttsim_bh.so \
  "https://github.com/tenstorrent/ttsim/releases/download/v1.4.3/libttsim_bh.so"
ln -sf libttsim_bh.so libttsim.so
```

### 2. 复制 soc 描述文件

```bash
# 使用完整的 soc descriptor（不是 sim/ 目录下的简化版本）
cp /path/to/tt_metal/tt_metal/soc_descriptors/blackhole_140_arch.yaml \
   /work/ttsim/soc_descriptor.yaml
```

**重要**: 必须使用 `soc_descriptors/blackhole_140_arch.yaml` 完整版本，因为：
- sim/soc_descriptor.yaml 是简化版，缺少 `dram_view_size` 和 `dram_views` 字段
- Metal 的 `metal_SocDescriptor` 需要这些字段来初始化 DRAM 元数据
- 完整版本包含所有必需的字段和正确的 eth cores 配置

### 3. 设置环境变量

创建 source 脚本：`scripts/setup_tt_sim.sh`

### 4. 验证测试

运行 UMD simulation 测试验证环境。

## 验证方法与测试结果

### 1. 基础功能测试

编写最小化测试程序直接调用 TT-Sim API：

```bash
cd /root/dev/vibe_dsl/tests/tt_sim
./bin/test_ttsim_v3
```

**测试结果**:
```
=== TT-Sim 最小化测试 v3 ===
✓ 成功加载 libttsim.so
✓ 成功获取函数指针
✓ TT-Sim 初始化成功
✓ Tenstorrent 设备识别成功 (Vendor ID: 0x1e52)
✓ Blackhole 设备识别成功 (Device ID: 0xb140)
✓ 数据验证成功 (Tile 读写)
```

### 2. 完整 Worker 核心测试

测试所有 140 个 Blackhole worker 核心：

```bash
./bin/test_ttsim_workers
```

**测试结果**:
```
=== TT-Sim Worker 核心测试 (Blackhole) ===
Worker 核心数量: 140
✓ 成功加载 libttsim.so
✓ TT-Sim 初始化成功
测试所有 Worker 核心的 L1 内存读写...

测试结果:
  通过: 140/140
  失败: 0/140
  耗时: 0 ms
  平均: 0 µs/核心
```

**结论**: 所有 140 个 Tensix 核心的 L1 内存读写功能验证通过。

### 3. TT-Metal 官方示例测试

运行 TT-Metal 官方 `add_2_integers_in_riscv` 示例验证完整功能链：

```bash
export TT_METAL_HOME=/root/dev/vibe_dsl/tt_metal_repo
export TT_METAL_SIMULATOR=$TT_METAL_HOME/sim/libttsim_bh.so
export TT_UMD_SIMULATOR=$TT_METAL_HOME/sim/libttsim_bh.so

# 设置库路径
export LD_LIBRARY_PATH=$TT_METAL_HOME/build_Release/tt_metal:\
$TT_METAL_HOME/build_Release/tt_stl:\
$TT_METAL_HOME/build_Release/ttnn:\
$TT_METAL_HOME/build_Release/lib:\
$TT_METAL_HOME/build_Release/tt_metal/third_party/umd/device:\
$TT_METAL_HOME/build_Release/_deps/fmt-build:\
$TT_METAL_HOME/build_Release/_deps/benchmark-build/src:\
$TT_METAL_HOME/sim:$LD_LIBRARY_PATH

cd $TT_METAL_HOME
TT_METAL_SLOW_DISPATCH_MODE=1 \
  ./build/programming_examples/metal_example_add_2_integers_in_riscv
```

**测试结果**:
```
Success: Result is 21
[10970] 0.3 seconds (36.0 KHz)
```

**关键步骤**:
1. ✅ 仿真器库加载成功
2. ✅ PCI 设备识别 (vendor_id=0x1e52, device_id=0xb140)
3. ✅ SocDescriptor 解析成功（需使用完整的 soc_descriptor.yaml）
4. ✅ Fabric 控制平面初始化
5. ✅ RISC-V 内核执行加法运算 (10 + 11 = 21)
6. ✅ 结果验证通过

**重要发现**:
- sim/soc_descriptor.yaml（简化版）缺少 `dram_view_size` 和 `dram_views` 字段
- 必须使用 tt_metal/soc_descriptors/blackhole_140_arch.yaml（完整版）

### 关键成功指标

| 指标 | 期望值 | 实际值 | 状态 |
|------|--------|--------|------|
| Vendor ID | 0x1e52 | 0x1e52 | ✅ |
| Device ID | 0xb140 | 0xb140 | ✅ |
| Worker 核心数 | 140 | 140 | ✅ |
| L1 读写测试 | 全部通过 | 140/140 | ✅ |

### 核心坐标映射

有效的 Tensix worker 核心坐标（translated）：
- x ∈ [1-7, 10-16] (14 列)
- y ∈ [2-11] (10 行)
- 总计: 14 × 10 = 140 核心

注意: x=8,9 是 DRAM/ARC/Eth 区域，不是 Tensix 核心。

### 已知限制

1. **只支持 Slow Dispatch**: TT-Sim 不支持 Fast Dispatch 模式
2. **SFPU 限制**: 某些 SFPU load macro 不被支持（需设置 `TT_METAL_DISABLE_SFPLOADMACRO=1`）
3. **DMA 操作**: 仿真器不支持 DMA 操作（会抛出 runtime_error）
4. **性能**: 仿真速度比实际硬件慢很多

## 预期产出

1. ✅ `libttsim_bh.so` 库文件已下载 (182KB)
2. ✅ `libttsim.so` 符号链接已创建
3. ✅ `soc_descriptor.yaml` 已配置（使用完整版本）
4. ✅ 环境变量已确定
5. ✅ 基础功能验证通过（PCI 识别、Tile 读写）
6. ✅ 完整 Worker 核心测试通过（140/140）
7. ✅ 测试程序 `tests/tt_sim/bin/test_ttsim_v3` 和 `tests/tt_sim/bin/test_ttsim_workers`
8. ✅ TT-Metal 官方示例测试通过 (`metal_example_add_2_integers_in_riscv`)
9. ✅ 配置文档 `docs/ttsim_setup.md`

## 参考文档

- `.github/workflows/ttsim.yaml` - CI 配置参考
- `tt_metal/third_party/umd/README.md` - UMD 文档
- `tt_metal/third_party/umd/tests/simulation/test_simulation_device.cpp` - 测试示例
