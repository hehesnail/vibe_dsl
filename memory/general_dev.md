# 通用开发模式与最佳实践

## 编译器后端开发模式

### 1. 代码生成器设计

**模式**: 继承 `CodeGenC` 基类，重写关键方法

```cpp
class CodeGenTileLangXXX final : public CodeGenC {
public:
  void PrintFuncPrefix(std::ostream &os) final;
  void PrintType(DataType t, std::ostream &os) final;
  void VisitStmt_(const ForNode *op) final;
  // ... 其他重写方法
};
```

**关键方法**:
- `PrintFuncPrefix`: 函数前缀（如 `__global__`）
- `PrintType`: 数据类型映射
- `VisitStmt_`: 语句处理
- `VisitExpr_`: 表达式处理

---

### 2. 类型系统映射

**常见陷阱**: 特殊浮点类型（FP8/FP6/FP4）的向量类型处理

**经验**:
- 先支持标量类型，再扩展向量类型
- 使用 `type.lanes()` 获取向量宽度
- 注意对齐要求（通常 16 字节对齐）

---

### 3. 内存 Scope 映射

**模式**: 通过 `PrintStorageScope` 方法映射内存层级

```cpp
void PrintStorageScope(const std::string &scope, std::ostream &os) {
  if (scope == "shared") {
    os << "__shared__";
  } else if (scope == "local") {
    // 寄存器变量，无需修饰
  }
}
```

---

### 4. 头文件管理

**策略**: 按需包含，使用标志位控制

```cpp
private:
  bool enable_fp16_ = false;
  bool need_math_constants_h_ = false;

public:
std::string Finish() {
  if (enable_fp16_) {
    decl_stream << "#include <cuda_fp16.h>\n";
  }
  return CodeGenC::Finish();
}
```

---

### 5. 调试技巧

**生成代码检查**:
- 使用 `BuildTileLangXXXWithoutCompile` 仅生成代码不编译
- 检查生成的代码是否符合目标平台语法

**常见问题**:
- 关键字冲突：使用 `ReserveKeywordsAsUnique_` 保留关键字
- 类型不匹配：检查 `PrintType` 实现
- 语法错误：对比参考生成的 CUDA/HIP 代码

---

### 6. 测试策略

**层级测试**:
1. 单元测试：代码生成器各方法独立测试
2. 集成测试：完整编译流程测试
3. 端到端测试：实际执行验证结果正确性

**快速迭代**:
- 先实现 WithoutCompile 版本，验证代码生成
- 再实现完整编译流程

---

## 项目特定模式

### TileLang 环境配置模式

#### 1. 大仓库管理策略

**问题**: tilelang_repo 体积 1.1GB（3rdparty 占 629MB），不适合完整提交

**解决方案**:
```gitignore
# .gitignore - 排除大目录
tilelang_repo/3rdparty/
tilelang_repo/build/
```

**初始化流程**:
```bash
# 首次克隆后初始化子模块
cd tilelang_repo
git submodule update --init --recursive  # 获取 3rdparty
```

#### 2. Python 包安装模式

**问题**: `pip install -e .` 会重新运行 cmake，可能因环境变量问题失败

**解决方案** - 使用 .pth 文件:
```bash
# 编译完成后，指向 build 目录
echo "$(pwd)/tilelang_repo" > "$(python3 -c "import site; print(site.getsitepackages()[0])")/tilelang.pth"
```

#### 3. 开发工作流

```
vibe_dsl/                    # 主仓库（提交文档+配置）
├── tilelang_repo/           # 代码仓库（提交源码修改）
│   ├── src/target/          # ← Blackhole CodeGen 开发位置
│   └── 3rdparty/            # 子模块（不提交）
└── tasks/                   # 任务文档
```

### TT-Metal 编译模式

#### 1. 依赖安装

```bash
# 创建 clang-20 软链接（如果系统只有 clang-21）
ln -sf /usr/bin/clang /usr/local/bin/clang-20
ln -sf /usr/bin/clang++ /usr/local/bin/clang++-20

# 安装系统依赖
apt-get install -y libnuma-dev libhwloc-dev libcapstone-dev
```

#### 2. CMake 配置参数

```bash
cmake -B build_Release \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \  # 解决 RPATH 问题
  -DENABLE_TRACY=OFF \                    # 禁用 profiler 加速编译
  -DWITH_PYTHON_BINDINGS=OFF \            # 不需要 Python 绑定
  -DENABLE_DISTRIBUTED=OFF \              # 不需要分布式
  -DCMAKE_TOOLCHAIN_FILE=cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake \
  -G Ninja
```

#### 3. 运行时库路径

```bash
export LD_LIBRARY_PATH=$TT_METAL_HOME/build_Release/lib:\
$TT_METAL_HOME/build_Release/tt_metal:\
$TT_METAL_HOME/build_Release/tt_stl:$LD_LIBRARY_PATH
```

### TT-Sim 仿真器配置模式

#### 1. 文件下载与安装

TT-Sim 是预编译库，直接从 GitHub Releases 下载：

```bash
mkdir -p /work/ttsim
cd /work/ttsim

# 下载 Blackhole 版本 (v1.4.3 - 最新版本)
curl -L -o libttsim_bh.so \
  "https://github.com/tenstorrent/ttsim/releases/download/v1.4.3/libttsim_bh.so"

# 创建符号链接（TT-Metal 期望 libttsim.so 名称）
ln -sf libttsim_bh.so libttsim.so
```

#### 2. 环境变量配置

```bash
export TT_METAL_SIMULATOR_HOME="/work/ttsim"
export TT_METAL_SIMULATOR="/work/ttsim/libttsim.so"
export TT_METAL_SLOW_DISPATCH_MODE=1      # 禁用快速 dispatch
export TT_METAL_DISABLE_SFPLOADMACRO=1    # 禁用 SFPU load macro
```

**注意**: `TT_METAL_SLOW_DISPATCH_MODE` 是必需的，TT-Sim 不支持 Fast Dispatch。

#### 3. soc 描述文件配置

```bash
# 复制 soc 描述文件（Blackhole 140 cores）
cp /path/to/tt_metal/repo/tt_metal/soc_descriptors/blackhole_140_arch.yaml \
   $TT_METAL_SIMULATOR_HOME/soc_descriptor.yaml
```

**文件结构要求**:
```
$TT_METAL_SIMULATOR_HOME/
├── libttsim.so          # TT-Sim 库（符号链接到 libttsim_bh.so）
├── libttsim_bh.so       # 实际的 Blackhole 仿真库
└── soc_descriptor.yaml  # SoC 描述文件
```

#### 4. 核心架构信息

**Blackhole 140 核心布局**:
- 网格: 17x12 (包含非 Tensix 区域)
- Worker cores: 140 个 (10 行 x 14 列)
- 坐标范围: x ∈ [1-7, 10-16], y ∈ [2-11]
- x=8,9 是 DRAM/ARC/Eth 区域

**PCI 设备 ID**:
- Vendor ID: 0x1e52 (Tenstorrent)
- Device ID: 0xb140 (Blackhole)
- Device ID: 0x401e (Wormhole)

#### 5. 验证测试

**基础测试** (直接调用 TT-Sim API):
```cpp
// 加载库
void* handle = dlopen("/work/ttsim/libttsim.so", RTLD_LAZY);

// 获取函数指针
auto libttsim_init = (void(*)())dlsym(handle, "libttsim_init");
auto libttsim_tile_wr = (void(*)(uint32_t, uint32_t, uint64_t, const void*, uint32_t))dlsym(handle, "libttsim_tile_wr_bytes");
auto libttsim_tile_rd = (void(*)(uint32_t, uint32_t, uint64_t, void*, uint32_t))dlsym(handle, "libttsim_tile_rd_bytes");

// 初始化
libttsim_init();

// 读写核心内存
uint32_t data = 0xDEADBEEF;
libttsim_tile_wr(0, 0, 0x10000, &data, sizeof(data));
libttsim_tile_rd(0, 0, 0x10000, &data, sizeof(data));
```

**完整 Worker 核心测试**:
```bash
# 测试所有 140 个核心
cd /root/dev/vibe_dsl/tests/tt_sim
./bin/test_ttsim_workers

# 期望输出:
# Worker 核心数量: 140
# 通过: 140/140
```

#### 6. TT-Sim API 函数

**核心函数** (C 接口):
```c
void libttsim_init(void);
void libttsim_exit(void);
uint32_t libttsim_pci_config_rd32(uint32_t bus_dev_fn, uint32_t offset);
void libttsim_tile_rd_bytes(uint32_t x, uint32_t y, uint64_t addr, void* data, uint32_t size);
void libttsim_tile_wr_bytes(uint32_t x, uint32_t y, uint64_t addr, const void* data, uint32_t size);
void libttsim_clock(uint32_t n_clocks);
```

**注意**: 这些函数使用全局状态，不需要实例指针。

### TT-Metal 后端开发注意事项

#### 待补充

随着开发深入，记录 TT-Metal 特有的：
- Kernel 签名规范
- 内存模型细节
- 编译和加载流程
- 调试技巧

---

## 代码审查清单

提交代码前检查：
- [ ] 遵循 TVM 命名规范
- [ ] 关键逻辑有注释
- [ ] 已测试代码生成功能
- [ ] 无调试用的 print/printf 残留
- [ ] 错误处理完善
