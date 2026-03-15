# TT-Sim 测试程序

本目录包含用于验证 TT-Sim 仿真器功能的测试程序。

## 目录结构

```
tests/tt_sim/
├── bin/           # 编译后的可执行文件
│   ├── test_ttsim_v3        # 基础功能测试（推荐）
│   ├── test_ttsim_workers   # 140 核心 L1 内存测试
│   ├── test_ttsim_minimal   # 最小化 API 测试
│   ├── test_ttsim_full      # 完整功能测试
│   └── test_ttsim_v2        # v2 版本测试
├── src/           # 源代码
│   ├── test_ttsim_v3.cpp
│   ├── test_ttsim_workers.cpp
│   ├── test_ttsim_minimal.cpp
│   ├── test_ttsim_full.cpp
│   ├── test_ttsim_minimal_v2.cpp
│   └── test_ttsim_simple.cpp
└── README.md      # 本文件
```

## 快速开始

### 1. 环境准备

确保已设置 TT-Sim 环境变量：

```bash
source scripts/setup_tt_sim.sh
```

### 2. 运行测试

```bash
cd tests/tt_sim

# 基础功能测试（验证 PCI 识别和 Tile 读写）
./bin/test_ttsim_v3

# 140 核心 L1 内存测试（验证所有 Worker 核心）
./bin/test_ttsim_workers
```

## 测试说明

### test_ttsim_v3

**功能**: 验证 TT-Sim 基础功能

**测试内容**:
- 加载 libttsim.so
- 初始化 TT-Sim 仿真器
- PCI 设备识别 (Vendor ID: 0x1e52, Device ID: 0xb140)
- Tile L1 内存读写

**期望输出**:
```
=== TT-Sim 最小化测试 v3 ===
✓ 成功加载 libttsim.so
✓ 成功获取函数指针
✓ TT-Sim 初始化成功
✓ Tenstorrent 设备识别成功 (Vendor ID: 0x1e52)
✓ Blackhole 设备识别成功 (Device ID: 0xb140)
✓ 数据验证成功 (Tile 读写)
```

### test_ttsim_workers

**功能**: 测试所有 140 个 Blackhole Worker 核心

**测试内容**:
- 枚举所有 Tensix Worker 核心
- 对每个核心进行 L1 内存读写测试
- 验证数据完整性

**期望输出**:
```
=== TT-Sim Worker 核心测试 (Blackhole) ===
Worker 核心数量: 140
✓ 成功加载 libttsim.so
✓ TT-Sim 初始化成功
测试所有 Worker 核心的 L1 内存读写...

测试结果:
  通过: 140/140
  失败: 0/140
```

## 重新编译

如需重新编译测试程序：

```bash
cd tests/tt_sim/src

# 编译 test_ttsim_v3
g++ -o ../bin/test_ttsim_v3 test_ttsim_v3.cpp -ldl -std=c++17

# 编译 test_ttsim_workers
g++ -o ../bin/test_ttsim_workers test_ttsim_workers.cpp -ldl -std=c++17
```

## 注意事项

1. **运行目录**: 测试程序需要在 `tests/tt_sim` 目录下运行，以便正确找到 `libttsim.so`

2. **环境变量**: 确保 `TT_METAL_SIMULATOR` 指向正确的库文件

3. **LD_LIBRARY_PATH**: 某些测试可能需要设置库路径：
   ```bash
   export LD_LIBRARY_PATH=$TT_METAL_HOME/sim:$LD_LIBRARY_PATH
   ```

## 参考

- [TT-Sim 配置文档](../../docs/ttsim_setup.md)
- [Phase 0 设计文档](../../tasks/dev_design/phase0_tt_sim_build.md)
