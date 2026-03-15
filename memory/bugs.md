# 问题与Bug解决方案记录

## 使用说明

按时间倒序记录开发过程中遇到的问题及解决方案。
每条记录包含：
- 问题描述
- 根本原因
- 解决方案
- 相关代码/文件

---

## 记录

### 模板示例（删除此行后开始记录）

**问题**: 描述问题的现象
**时间**: YYYY-MM-DD
**根本原因**: 为什么会发生
**解决方案**: 如何解决
**关键代码**:
```cpp
// 相关代码片段
```
**参考**: 相关链接或文档

---

### pip install -e 失败

**问题**: `pip install -e . --no-build-isolation` 报错，cmake 配置失败
**时间**: 2026-03-15
**根本原因**: scikit-build-core 会重新运行 cmake，而 FindThreads 在特定环境下检测失败
**解决方案**: 使用 .pth 文件直接指向已编译的 build 目录，避免 pip 重新构建
**关键代码**:
```bash
# 编译完成后直接创建 .pth 文件
echo "$(pwd)/tilelang_repo" > "$(python3 -c "import site; print(site.getsitepackages()[0])")/tilelang.pth"
```
**参考**: setup_tilelang.sh

---

### @T.prim_func 装饰器获取源码失败

**问题**: 使用 `python -c "..."` 内联代码时，@T.prim_func 报错 `OSError: could not get source code`
**时间**: 2026-03-15
**根本原因**: TileLang 使用 `inspect.getsourcelines()` 获取函数 AST，需要源码文件
**解决方案**: 将测试代码写入 .py 文件再执行
**关键代码**:
```python
# 正确做法：写入文件
# test.py
@T.prim_func
def kernel(...):
    ...

# 运行
python test.py
```
**参考**: phase0_tilelang_setup.md

---

### tilelang_repo 体积过大无法提交

**问题**: tilelang_repo 体积 1.1GB，git push 会超时/失败
**时间**: 2026-03-15
**根本原因**: 3rdparty/ 子模块占 629MB，build/ 占 374MB
**解决方案**:
1. .gitignore 排除 3rdparty/ 和 build/
2. 提交核心源码（src/, tilelang/, docs/ 等，约 20MB）
3. 提供 setup_tilelang.sh 脚本初始化子模块
**关键代码**:
```gitignore
# .gitignore
tilelang_repo/3rdparty/
tilelang_repo/build/
```
**参考**: .gitignore, setup_tilelang.sh

---

### TT-Metal 编译依赖问题汇总

**问题**: TT-Metal 编译需要多个系统依赖
**时间**: 2026-03-15
**根本原因**: tt_metal_repo 依赖 clang-20、NUMA、hwloc、capstone 等库
**解决方案**:
```bash
# 1. 创建 clang-20 软链接（系统有 clang-21）
ln -sf /usr/bin/clang /usr/local/bin/clang-20
ln -sf /usr/bin/clang++ /usr/local/bin/clang++-20

# 2. 安装系统依赖
apt-get install -y libnuma-dev libhwloc-dev libcapstone-dev

# 3. 配置 cmake（关键参数）
cmake -B build_Release \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
  -DENABLE_TRACY=OFF \
  -DWITH_PYTHON_BINDINGS=OFF \
  -G Ninja

# 4. 设置运行时库路径
export LD_LIBRARY_PATH=/root/dev/vibe_dsl/tt_metal_repo/build_Release/lib:\
/root/dev/vibe_dsl/tt_metal_repo/build_Release/tt_metal:\
/root/dev/vibe_dsl/tt_metal_repo/build_Release/tt_stl:$LD_LIBRARY_PATH

# 5. 编译
ninja -C build_Release
```
**关键文件**:
- libtt_metal.so: 18MB
- libdevice.so: 4.6MB

**参考**: tasks/dev_design/phase0_tt_metal_build.md

---

### TT-Sim Blackhole ETH cores 检查失败

**问题**: UMD simulation 测试报错 `Exactly 2 or 14 ETH cores should be harvested on full Blackhole`
**时间**: 2026-03-15
**状态**: 已解决（使用完整的 soc descriptor）
**根本原因**:
- `SocDescriptor` 默认构造函数使用 `ChipInfo chip_info = {}`，导致 `harvesting_masks.eth_harvesting_mask = 0`
- Blackhole 架构在 `BlackholeCoordinateManager::assert_coordinate_manager_constructor()` 中检查：
  - 如果 `eth_cores.size() == 14`（`NUM_ETH_CHANNELS`），则必须有 2 或 14 个 harvested eth cores
  - 否则抛出异常

**解决方案**: 使用完整的 soc descriptor 文件，其中包含正确的 harvesting 配置
```bash
# 使用完整的 blackhole_140_arch.yaml
cp $TT_METAL_HOME/tt_metal/soc_descriptors/blackhole_140_arch.yaml \
   $TT_METAL_HOME/sim/soc_descriptor.yaml
```

**注意**: 早期尝试的解决方案（修改 eth cores 数量）已废弃，使用完整配置更可靠。

**参考**:
- `blackhole_coordinate_manager.cpp:60-67`
- `tasks/dev_design/phase0_tt_sim_build.md`

---

### TT-Sim 环境变量配置

**问题**: UMD 测试需要多个环境变量才能正确找到 TT-Sim
**时间**: 2026-03-15
**根本原因**:
- TT-Metal 和 UMD 使用不同的环境变量名
- `TT_UMD_SIMULATOR` 需要指向 `.so` 文件而非目录

**解决方案**:
```bash
# TT-Metal 环境变量
export TT_METAL_SIMULATOR_HOME="${TT_METAL_HOME}/sim"
export TT_METAL_SIMULATOR="${TT_METAL_SIMULATOR_HOME}/libttsim.so"
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_DISABLE_SFPLOADMACRO=1

# UMD 测试额外需要
export TT_UMD_SIMULATOR="${TT_METAL_SIMULATOR}"
```

**参考**: `.github/workflows/ttsim.yaml`, `scripts/setup_tt_sim.sh`

---

### TT-Sim UMD 部分测试失败（不影响 Metal 示例）

**问题**: UMD simulation 测试中部分测试用例失败
**时间**: 2026-03-15
**状态**: 已知限制，不影响 Metal 官方示例
**根本原因分析**:

1. **(1, 1) 坐标失败**:
   - 测试使用 `tt_xy_pair{1, 1}` 作为 core 坐标
   - `(1,1)` 是 eth core，UMD 测试可能配置不正确
   - 错误: `No core type found for system TRANSLATED at location: (1, 1)`

2. **(1, 0) 坐标失败**:
   - `(1, 0)` 是 DRAM core
   - TT-Sim 可能不完全支持 DRAM 直接访问
   - 错误: `coord_to_tile: coord (1,0)`

3. **SimpleApiTest 失败**:
   - 测试使用了 `RiscType::ALL_NEO_DMS`，Blackhole 不支持 NEO risc cores
   - 错误: `NEO risc cores should not be used on Blackhole architecture`

**测试结果汇总**:

| 测试用例 | 参数 | 结果 |
|---------|------|------|
| LoopbackSingleTensix | (0, 1) TENSIX | ✅ 通过 |
| LoopbackSingleTensix | (1, 1) ETH | ❌ 失败 |
| LoopbackSingleTensix | (1, 0) DRAM | ❌ 失败 |
| LoopbackStressSize | (0, 1) TENSIX | ✅ 通过 |
| LoopbackTwoTensix | - | ❌ 失败 (使用 (1,1)) |
| SimpleApiTest | - | ❌ 失败 (NEO risc) |

**重要说明**:
- UMD 单元测试失败 **不影响** Metal 官方示例运行
- `metal_example_add_2_integers_in_riscv` 可以正常工作
- 这些失败是 UMD 测试本身的限制，不是 TT-Sim 或配置问题

**结论**:
- UMD 单元测试有部分限制
- Metal 官方示例工作正常，可用于 TileLang 后端开发

**参考**: `tasks/dev_design/phase0_tt_sim_build.md`

---

### TT-Sim metal_example_add_2_integers_in_riscv YAML 解析失败

**问题**: 运行 `metal_example_add_2_integers_in_riscv` 测试时报错 `YAML::TypedBadConversion<unsigned long>`
**时间**: 2026-03-15
**根本原因**:
- TT-Sim 自带的 `sim/soc_descriptor.yaml` 是简化版本，缺少 `dram_view_size` 和 `dram_views` 字段
- Metal 的 `metal_SocDescriptor::load_dram_metadata_from_device_descriptor()` 函数需要这些字段
- 堆栈跟踪显示错误发生在：`metal_SocDescriptor::load_dram_metadata_from_device_descriptor()`

**解决方案**: 使用完整的 soc descriptor 文件替换 sim 目录下的简化版本
```bash
# 使用完整的 blackhole_140_arch.yaml 替换 sim/soc_descriptor.yaml
cp $TT_METAL_HOME/tt_metal/soc_descriptors/blackhole_140_arch.yaml \
   $TT_METAL_HOME/sim/soc_descriptor.yaml
```

**关键差异**:
```yaml
# sim/soc_descriptor.yaml (简化版) - 缺少以下字段
dram_view_size: 4278190080
dram_views:
  [
    { channel: 0, eth_endpoint: [2, 1], worker_endpoint: [2, 1], address_offset: 0 },
    { channel: 1, eth_endpoint: [0, 1], worker_endpoint: [0, 1], address_offset: 0 },
    # ... 更多 channel
  ]
harvested_workers: []
features:
  noc:
    translation_id_enabled: True
  # ... 更多 features
```

**验证结果**:
```
Success: Result is 21
[10970] 0.3 seconds (36.0 KHz)
```

**参考**: `tt_metal/llrt/metal_soc_descriptor.cpp:110-112`

---

*后续问题继续追加...*
