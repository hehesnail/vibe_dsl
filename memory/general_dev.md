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

**运行 TT-Metal 官方示例**:
```bash
# 运行官方编程示例验证 TT-Sim
cd /root/dev/vibe_dsl/tests/tt_sim
./run_official_examples.sh

# 或使用脚本设置环境后手动运行
source scripts/setup_tt_sim.sh
cd $TT_METAL_HOME
./build/programming_examples/metal_example_add_2_integers_in_riscv
# 期望输出: Success: Result is 21
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

**参考详细问题记录**: `memory/bugs.md` - "TileLang Blackhole 后端编译错误汇总"

#### 1. CodeGen 类继承关键要点

```cpp
// 不能覆盖的 final 方法
- PrintFuncPrefix, PrintType
- VisitStmt_(const AttrStmtNode*), VisitStmt_(const ForNode*)

// 解决方案：在 AddFunction 中预处理，或改用 IR Pass
```

#### 2. TVM FFI 类型速查

| 旧方式 | 新方式 |
|--------|--------|
| `tvm::String` | `tvm::ffi::String` |
| `tvm::attr::kKernel` | `tvm::attr::kGlobalSymbol` |
| `TVMContext` | `Device` (即 `DLDevice`) |
| `opt.defined()` | `if (opt)` |

#### 3. 头文件包含

```cpp
#include <tvm/runtime/device_api.h>
#include <tvm/ffi/reflection/registry.h>
```

#### 4. Runtime 模块开发模式

**Build 函数标准结构**:

```cpp
ffi::Module BuildTileLangXXX(IRModule mod, Target target) {
  // 1. 初始化 CodeGen
  CodeGenXXX cg;
  cg.Init(output_ssa, emit_asserts, emit_fwd_func_decl, target->str(), devices);

  // 2. 处理所有 PrimFunc
  for (auto kv : mod->functions) {
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<PrimFunc>(kv.second);
    cg.AddFunction(gvar, f);
  }

  // 3. 生成代码
  std::string code = cg.Finish();

  // 4. 提取函数名
  ffi::Array<ffi::String> func_names;
  // ... populate func_names

  // 5. 创建模块
  return CSourceModuleCreate(code, "cc", func_names, {});
}
```

**注册机制**:

```cpp
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("target.build.tilelang_xxx", BuildTileLangXXX)
      .def("target.build.tilelang_xxx_without_host", BuildTileLangXXXWithoutHost)
      .def("device_api.xxx", []() -> void* {
        return static_cast<void*>(XXXDeviceAPI::Global());
      });
}
```

**Device API 实现要点**:

```cpp
class XXXDeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(Device dev) final;
  void GetAttr(Device dev, DeviceAttrKind kind, ffi::Any* rv) final;
  void* AllocDataSpace(Device dev, size_t nbytes, size_t alignment,
                       DLDataType type_hint) final;
  void FreeDataSpace(Device dev, void* ptr) final;
  void StreamSync(Device dev, TVMStreamHandle stream) final;

  static XXXDeviceAPI* Global();
};
```

**关键注意点**:
- `CSourceModuleCreate` 需要 `ffi::Array<ffi::String>` 类型，不是 `std::vector<std::string>`
- 需要包含 `<tvm/target/codegen.h>` 获取 `CSourceModuleCreate`
- Device API 使用单例模式，`Global()` 方法返回静态实例

---

## TT-Sim 内核开发与测试模式

### 1. TT-Sim 兼容内核编写要点

**必须使用 InterleavedAddrGen**:
```cpp
// TT-Sim 要求使用 InterleavedAddrGen 进行地址转换
// 直接使用物理地址会导致 UnimplementedFunctionality 错误

InterleavedAddrGen<true> src_gen = {
    .bank_base_address = src_dram_addr,
    .page_size = TILE_SIZE
};

// 使用 get_noc_addr 获取 NOC 地址
uint64_t src_noc_addr = get_noc_addr(tile_idx, src_gen);
noc_async_read(src_noc_addr, l1_buffer_addr, TILE_SIZE);
```

**单核限制**:
- TT-Sim 目前只支持 BRISC (RISCV_0) 的完整 NOC 操作
- NCRISC (RISCV_1) 的 NOC write 在 TT-Sim 上会报错
- 多核场景需要将 Reader 和 Writer 合并到单个 BRISC kernel

### 2. TT-Sim JIT 环境配置

**符号链接清单** (build_Release 目录):
```bash
# 1. SFPI 编译器
ln -sf $TT_METAL_HOME/runtime/sfpi $TT_METAL_HOME/build_Release/runtime/sfpi

# 2. Hardware 定义
ln -sf $TT_METAL_HOME/tt_metal/hw $TT_METAL_HOME/build_Release/tt_metal/hw

# 3. TT-LLK 库
ln -sf $TT_METAL_HOME/tt_metal/third_party/tt_llk \
       $TT_METAL_HOME/build_Release/tt_metal/third_party/tt_llk

# 4. Host-Device 通用接口
ln -sf $TT_METAL_HOME/tt_metal/hostdevcommon \
       $TT_METAL_HOME/build_Release/tt_metal/hostdevcommon

# 5. API 头文件
ln -sf $TT_METAL_HOME/tt_metal/api/tt-metalium \
       $TT_METAL_HOME/build_Release/tt_metal/api/tt-metalium

# 6. 工具链
ln -sf $TT_METAL_HOME/runtime/hw $TT_METAL_HOME/build_Release/runtime/hw

# 7. Profiler 工具
ln -sf $TT_METAL_HOME/tt_metal/tools/profiler \
       $TT_METAL_HOME/build_Release/tt_metal/tools/profiler
```

### 3. TT-Sim 测试流程

**完整测试步骤**:
```bash
# 1. 设置环境
source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh

# 2. 从 TT_METAL_HOME 运行测试
cd $TT_METAL_HOME
./build_Release/programming_examples/tilelang_copy_test

# 3. 验证输出
# 期望: "✓ SUCCESS: Copy kernel test passed!"
```

**重要注意事项**:
- 必须从 `$TT_METAL_HOME` 目录运行测试
- Kernel 文件不要使用 `#include "dataflow_api.h"` (TT-Metal 会自动包含)
- 使用 `OVERRIDE_KERNEL_PREFIX` 宏指定 kernel 路径

### 4. CodeGen 更新模式

**生成 TT-Sim 兼容代码**:
```cpp
std::string CodeGenBlackhole::GenerateSimpleCopyKernel(...) {
  std::ostringstream os;

  // 生成 InterleavedAddrGen 风格的代码
  os << "InterleavedAddrGen<true> src_gen = {\n";
  os << "    .bank_base_address = src_dram_addr,\n";
  os << "    .page_size = TILE_SIZE\n";
  os << "};\n";

  os << "uint64_t src_noc_addr = get_noc_addr(i, src_gen);\n";
  os << "noc_async_read(src_noc_addr, l1_buffer_addr, TILE_SIZE);\n";

  return os.str();
}
```

---

## TIR Transform Pass 开发模式

### 1. Pass 基本结构

**模式**: 继承 `StmtExprMutator`，实现 `Transform` 方法

```cpp
// 头文件: my_pass.h
class MyPass : public tvm::tir::StmtExprMutator {
 public:
  // Main entry point
  tvm::tir::PrimFunc Transform(const tvm::tir::PrimFunc& func);

  // 辅助方法
  ResultType AnalyzeSomething(const PrimFunc& func);
  void ProcessData(DataType& data);

 private:
  // 内部状态
  InternalState state_;
};

// 创建 Pass 的工厂函数
tvm::tir::transform::Pass MyPassPass();
```

**实现文件**:
```cpp
// my_pass.cc
PrimFunc MyPass::Transform(const PrimFunc& func) {
  // 1. 分析阶段
  state_ = AnalyzeSomething(func);

  // 2. 处理阶段
  ProcessData(state_);

  // 3. 创建可变的函数副本
  PrimFunc new_func = func;

  // 4. 存储结果到函数属性
  StoreResult(new_func, state_);

  return new_func;
}

// Pass 注册
class MyPassPassNode : public transform::PassNode {
 public:
  IRModule operator()(IRModule mod, const transform::PassContext& pass_ctx) const final {
    for (const auto& [gvar, func] : mod->functions) {
      if (auto* prim_func = func.as<PrimFuncNode>()) {
        PrimFunc updated_func = MyPass().Transform(GetRef<PrimFunc>(prim_func));
        mod.CopyOnWrite()->Add(gvar, updated_func);
      }
    }
    return mod;
  }

  TVM_OBJECT_ENABLE(MyPassPassNode, transform::PassNode);
};

tvm::tir::transform::Pass MyPassPass() {
  return tvm::make_object<MyPassPassNode>();
}

TVM_REGISTER_GLOBAL("tl.transform.MyPass")
    .set_body_typed(MyPassPass);
```

### 2. StmtExprVisitor 收集信息

**模式**: 使用 Visitor 模式遍历 IR 收集信息

```cpp
class MyCollector : public StmtExprVisitor {
 public:
  std::vector<Buffer> collected_buffers;

 private:
  void VisitStmt_(const BufferLoadNode* op) final {
    collected_buffers.push_back(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AllocateNode* op) final {
    // 检查存储 scope
    auto storage_scope = op->buffer_var->type_annotation.as<PointerTypeNode>();
    if (storage_scope && storage_scope->storage_scope == "shared") {
      // 处理 shared memory
    }
    StmtExprVisitor::VisitStmt_(op);
  }
};

// 使用
MyCollector collector;
collector(func->body);
```

### 3. 函数属性存储

**模式**: 使用 Map<String, ObjectRef> 存储 Pass 结果

```cpp
void StoreResult(PrimFunc& func, const Result& result) {
  Map<String, ObjectRef> attrs = func->attrs;

  // 存储简单值
  Map<String, Integer> result_map;
  result_map.Set("field1", Integer(result.field1));
  result_map.Set("field2", Integer(result.field2));

  attrs.Set("tl_pass_result", result_map);
  func.CopyOnWrite()->attrs = attrs;
}
```

### 4. 单元测试模式

**模式**: 使用 gtest 框架，mock TVM 依赖

```cpp
// 1. Mock 核心数据结构
struct MockConfig {
  int field1;
  int field2;
};

// 2. Mock Pass 类（简化版，不依赖 TVM）
class MockMyPass {
 public:
  MockConfig Transform(int input) {
    MockConfig config;
    config.field1 = input * 2;
    config.field2 = input + 10;
    return config;
  }
};

// 3. 测试 Fixture
class MyPassTest : public ::testing::Test {
 protected:
  MockMyPass pass;
  void SetUp() override {}
  void TearDown() override {}
};

// 4. 测试用例
TEST_F(MyPassTest, BasicTest) {
  auto result = pass.Transform(5);
  EXPECT_EQ(result.field1, 10);
  EXPECT_EQ(result.field2, 15);
}

// 5. 边界测试
TEST_F(MyPassTest, EdgeCases) {
  // 测试边界值
  auto result = pass.Transform(0);
  EXPECT_EQ(result.field1, 0);
}
```

### 5. CMake 集成

**模式**: 使用 GLOB 自动收集源文件

```cmake
# CMakeLists.txt 中使用 GLOB 自动包含新的 .cc 文件
file(GLOB TILE_LANG_SRCS
  src/transform/*.cc    # ← 新增文件自动被包含
)
```

**新增文件清单**:
```
tilelang_repo/src/transform/
├── assign_blackhole_cores.h    # Core 分配 Pass 头文件
├── assign_blackhole_cores.cc   # Core 分配 Pass 实现
├── plan_blackhole_cb.h         # CB 分配 Pass 头文件
├── plan_blackhole_cb.cc        # CB 分配 Pass 实现
├── split_blackhole_kernel.h    # Kernel 拆分 Pass 头文件
└── split_blackhole_kernel.cc   # Kernel 拆分 Pass 实现
```

### 6. Blackhole 特有约束处理

**Core 坐标映射**:
```cpp
// 逻辑索引 (0-139) 映射到物理坐标 (避开 x=8,9)
CoreCoord GetCoreCoord(int core_idx) {
  int x_in_grid = core_idx % 14;  // 0-13
  int y_in_grid = core_idx / 14;  // 0-9

  // 物理 x: 1-7, 10-16 (跳过 x=8,9)
  int physical_x = (x_in_grid < 7) ? x_in_grid + 1 : x_in_grid + 3;
  int physical_y = y_in_grid + 2;   // 2-11

  return CoreCoord{physical_x, physical_y};
}
```

**CB 约束检查**:
```cpp
bool ValidateCBAllocation(const std::vector<CBConfig>& configs) {
  // 检查数量
  if (configs.size() > 64) return false;

  // 检查总大小
  int total_size = 0;
  for (const auto& cfg : configs) {
    total_size += cfg.total_size;
  }
  if (total_size > 1572864) return false;  // 1.5MB

  return true;
}
```

---

## CodeGen Visitor 模式实现

### 1. TT-Metal Builtin 识别与分发

**模式**: 重写 `VisitExpr_` 识别特定 Op，分发到专用处理函数

```cpp
void CodeGenBlackhole::VisitExpr_(const tvm::tir::CallNode *op,
                                  std::ostream &os) {
  // Try to handle TT-Metal builtin calls
  if (HandleBlackholeBuiltin(op, os)) {
    return;
  }
  // Fall back to parent class for other calls
  CodeGenCHost::VisitExpr_(op, os);
}

bool CodeGenBlackhole::HandleBlackholeBuiltin(const tvm::tir::CallNode *op,
                                               std::ostream &os) {
  if (!op->op->IsInstance<OpNode>()) return false;

  Op call_op = Downcast<Op>(op->op);
  std::string op_name = call_op->name;

  // Check for TT-Metal builtin prefix
  const std::string prefix = "tl.blackhole.";
  if (op_name.find(prefix) != 0) return false;

  std::string builtin_name = op_name.substr(prefix.length());

  // Handle each builtin type
  if (builtin_name == "matmul_tiles") {
    PrintMatmulTiles(op, os);
    return true;
  } else if (builtin_name == "cb_wait_front") {
    PrintCBWaitFront(op, os);
    return true;
  }
  // ... more builtins

  return false;
}
```

**关键要点**:
- 使用前缀匹配识别自定义 builtin (`tl.blackhole.xxx`)
- 返回 `bool` 表示是否已处理，便于 fallback
- 分离解析逻辑和代码生成逻辑

### 2. TT-Metal 算子代码生成

**模式**: 每个 builtin 对应一个 Print 函数

```cpp
void CodeGenBlackhole::PrintMatmulTiles(const tvm::tir::CallNode *op,
                                        std::ostream &os) {
  need_compute_api_h_ = true;  // 标记需要包含头文件
  os << "matmul_tiles(";
  PrintExpr(op->args[0], os);  // in0_cb_id
  os << ", ";
  PrintExpr(op->args[1], os);  // in1_cb_id
  os << ", ";
  PrintExpr(op->args[2], os);  // in0_tile_index
  os << ", ";
  PrintExpr(op->args[3], os);  // in1_tile_index
  os << ", ";
  PrintExpr(op->args[4], os);  // dst_tile_index
  os << ")";
}
```

**头文件管理**:
```cpp
class CodeGenBlackhole : public CodeGenCHost {
 private:
  bool need_compute_api_h_ = false;   // compute_kernel_api.h
  bool need_dataflow_api_h_ = false;  // dataflow_api.h

 public:
  std::string Finish() {
    // 按需添加头文件
    if (need_compute_api_h_) {
      decl_stream << "#include \"compute_kernel_api.h\"\n";
    }
    return CodeGenCHost::Finish();
  }
};
```

### 3. 端到端测试模式

**Python E2E 测试**: DSL -> TIR -> 参考验证

```python
def test_blackhole_gemm_e2e():
    # 1. 定义 TileLang kernel
    @T.prim_func
    def matmul_kernel(...):
        with T.Kernel(...) as (bx, by):
            # ... 分配共享内存
            T.clear(C_local)
            for k in T.Pipelined(K_tiles):
                T.copy(A[...], A_shared)
                T.copy(B[...], B_shared)
                T.gemm(A_shared, B_shared, C_local)

    # 2. Lower 到 TIR
    target = tvm.target.Target("cuda")
    with target:
        artifact = tilelang.lower(matmul_kernel)

    # 3. 生成参考实现
    A = torch.randn(M, K, dtype=torch.float16)
    B = torch.randn(K, N, dtype=torch.float16)
    C_ref = torch.matmul(A, B)

    # 4. 保存供 TT-Sim 验证
    np.save("/tmp/blackhole_gemm_A.npy", A.cpu().numpy())
    np.save("/tmp/blackhole_gemm_B.npy", B.cpu().numpy())
    np.save("/tmp/blackhole_gemm_C_ref.npy", C_ref.cpu().numpy())
```

**C++ 单元测试**: 独立测试 CodeGen 逻辑

```cpp
// 不依赖 TVM 的完整编译，仅测试代码生成逻辑
TEST(CodeGenBlackholeGEMM, BasicMatmulTiles) {
  // 构建 mock TIR
  auto func = CreateGemmComputeFunc();

  // 生成代码
  CodeGenBlackhole cg;
  cg.AddFunction(gvar, func);
  std::string code = cg.Finish();

  // 验证生成的代码包含期望的模式
  EXPECT_NE(code.find("matmul_tiles("), std::string::npos);
  EXPECT_NE(code.find("tile_regs_acquire()"), std::string::npos);
  // ...
}
```

---

## 代码审查清单

提交代码前检查：
- [ ] 遵循 TVM 命名规范
- [ ] 关键逻辑有注释
- [ ] 已测试代码生成功能
- [ ] 无调试用的 print/printf 残留
- [ ] 错误处理完善
- [ ] TT-Sim 兼容性验证（如适用）
- [ ] Visitor 模式正确处理所有分支（有 fallback）
- [ ] 头文件按需包含（使用标志位控制）
