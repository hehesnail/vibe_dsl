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
data
export TT_METAL_SIMULATOR=$TT_METAL_SIMULATOR_HOME/libttsim.so
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_DISABLE_SFPLOADMACRO=1
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
- 版本: v1.4.2
- 文件: libttsim_bh.so (Blackhole 版本)
- URL: https://github.com/tenstorrent/ttsim/releases

```bash
mkdir -p $TT_METAL_HOME/sim
curl -L -o $TT_METAL_HOME/sim/libttsim_bh.so \
  "https://github.com/tenstorrent/ttsim/releases/download/v1.4.2/libttsim_bh.so"
ln -sf $TT_METAL_HOME/sim/libttsim_bh.so $TT_METAL_HOME/sim/libttsim.so
```

### 2. 复制 soc 描述文件

```bash
cp $TT_METAL_HOME/tt_metal/third_party/umd/tests/soc_descs/blackhole_140_arch.yaml \
   $TT_METAL_SIMULATOR_HOME/soc_descriptor.yaml
```

**注意**: Blackhole 架构需要调整 eth cores 配置。原配置有 14 个 eth cores，但 UMD 测试默认的 harvesting_masks 为 0，导致检查失败。修改 soc 描述文件中的 eth 列表，移除前两个 cores（保留 12 个），可绕过此检查。

### 3. 设置环境变量

创建 source 脚本：`scripts/setup_tt_sim.sh`

### 4. 验证测试

运行 UMD simulation 测试验证环境。

## 验证方法

```bash
# 运行 UMD simulation 测试
cd tt_metal_repo/tt_metal/third_party/umd
cmake -B build -G Ninja -DTT_UMD_BUILD_ALL=ON
cmake --build build
./build/tests/simulation/test_simulation_device
```

## 预期产出

1. `libttsim_bh.so` 库文件已下载
2. `soc_descriptor.yaml` 已配置
3. `scripts/setup_tt_sim.sh` 环境脚本
4. 验证测试通过

## 参考文档

- `.github/workflows/ttsim.yaml` - CI 配置参考
- `tt_metal/third_party/umd/README.md` - UMD 文档
- `tt_metal/third_party/umd/tests/simulation/test_simulation_device.cpp` - 测试示例
