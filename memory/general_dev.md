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

---

## Blackhole Runtime Module 实现经验

### 1. TVM Module 架构理解

**核心洞察**: TileLang/TVM 的 Runtime 采用 Host-Device 分离架构

```
TIR PrimFunc (Device)
    ↓ CodeGen
TT-Metal Kernel C++
    ↓ JIT Compile
RISC-V ELF
    ↓ Runtime Module
Python callable (PackedFunc)
```

**关键类**:
- `ffi::ModuleObj`: TVM Module 基类
- `ffi::Module`: Module 智能指针包装
- `BlackholeWrappedFunc`: 实际执行器，实现 `operator()(PackedArgs, Any*)`

### 2. BlackholeModule 设计模式

**模式**: 延迟初始化 + 缓存

```cpp
class BlackholeModuleNode : public ffi::ModuleObj {
  // 延迟初始化（首次调用时）
  void EnsureDeviceInitialized() {
    if (!device_initialized_) {
      mesh_device_ = MeshDevice::create_unit_mesh(0);
    }
  }

  // Program 缓存（避免重复 JIT 编译）
  CompiledProgram& GetOrCompileProgram(const std::string& func_name) {
    if (cache_.count(func_name)) return cache_[func_name];
    // 创建 Program、配置 CB、编译 Kernels
    return cache_[func_name] = CompileProgram(func_name);
  }
};
```

### 3. 参数传递映射

**TVM Packed Args → TT-Metal Runtime Args**:

| TVM Arg | TT-Metal | 处理 |
|---------|----------|------|
| `DLTensor*` (buffer) | `MeshBuffer` | 创建 buffer，写入数据，传递 address |
| `uint32_t` (scalar) | `uint32_t` | 直接传递 |

### 4. 多 Kernel 执行顺序

**TT-Metal 约束**: RISC-V 内核按提交顺序执行

```cpp
// Reader
EnqueueMeshWorkload(cq, reader_workload, blocking=true);
// Compute
EnqueueMeshWorkload(cq, compute_workload, blocking=true);
// Writer
EnqueueMeshWorkload(cq, writer_workload, blocking=false);
Finish(cq);  // 全局同步
```

### 5. 关键注意事项

1. **文件系统**: 使用 `std::filesystem` (C++17)，需 GCC 8+
2. **Kernel 文件**: TT-Metal JIT 编译需要文件路径，需保存到临时目录
3. **Buffer 生命周期**: MeshBuffer 需保持存活直到 kernel 执行完成
4. **错误处理**: TT-Metal 使用异常，建议用 try-catch 包裹

---

## CodeGenBlackhole 重写经验（2026-03-16 更新）

### 1. 从继承到重写的转变

**原方案（失败）**:
```cpp
class CodeGenBlackhole : public CodeGenCHost {
  // 继承导致生成标准C函数格式
};
```

**新方案（成功）**:
```cpp
class CodeGenBlackhole : public CodeGenCHost {
 public:
  void AddFunction(const GlobalVar& gvar, const PrimFunc& f) override {
    // 完全重写，生成 kernel_main 格式
    GenerateKernelMain(gvar, f);
  }

  void GenerateKernelMain(const GlobalVar& gvar, const PrimFunc& f) {
    stream << "void kernel_main() {\n";
    // 生成 get_arg_val 参数加载
    // 生成函数体
    stream << "}\n";
  }
};
```

### 2. 关键实现要点

**参数加载生成**:
```cpp
// Buffer 参数（64-bit 地址）
stream << "  uint32_t A_lo = get_arg_val<uint32_t>(0);\n";
stream << "  uint32_t A_hi = get_arg_val<uint32_t>(1);\n";
stream << "  uint64_t A_addr = ((uint64_t)A_hi << 32) | A_lo;\n";
stream << "  half* A = (half*)(uintptr_t)A_addr;\n";

// 标量参数（直接加载）
stream << "  uint32_t scalar = get_arg_val<uint32_t>(2);\n";
```

---

## Blackhole Pipeline 返工经验（2026-03-17 更新）

### 1. 问题：设计与实现脱节

**症状**：
- Transform Pass 在文档中标记为"完成"，实际都是 Stub
- lower.py 未调用任何 Blackhole Pass
- CodeGen 存在硬编码的 Copy kernel 路径

**根本原因**：
- "文档先行"陷阱：先写设计文档 → 写 Stub → 写 Mock 测试 → 标记完成
- 缺乏端到端验证

**解决方案**：
- 按照 [design_review.md](../tasks/design_review.md) 重新设计 Pipeline
- Pass 顺序调整为：`LowerOps → PlanCB → AssignCores`（去掉 Split）
- 所有 Pass 真正走通 IR → IR 转换

### 2. Pattern: Pass 结果通过 IR Attrs 传递

**反模式**（避免）：
```cpp
// 不要这样：Pass 之间通过全局状态或 C++ 对象传递
class PassA {
  std::vector<Config> configs_;  // 私有状态，其他 Pass 看不到
};
```

**正确模式**：
```cpp
// Pass A: 将结果写入 func attrs
void PassA::Transform(PrimFunc& func) {
  Map<String, ObjectRef> attrs = func->attrs;
  attrs.Set("blackhole.cb_requirements", cb_reqs);
  func.CopyOnWrite()->attrs = DictAttrs(attrs);
}

// Pass B: 从 attrs 读取
void PassB::Transform(PrimFunc& func) {
  auto cb_reqs = func->GetAttr<Array<ObjectRef>>("blackhole.cb_requirements");
  // ...
}
```

### 3. Pattern: 删除硬编码路径

**反模式**（避免）：
```cpp
// 不要这样：启发式检测 + 硬编码生成
bool DetectSimpleCopyKernel(PrimFunc f) {
  return f->params.size() == 2;  // 太脆弱！
}

void GenerateCopyKernelMain(...) {
  stream << "cb_reserve_back(0, 1);\n";  // 硬编码 CB 0
  stream << "noc_async_read(...);\n";   // 硬编码逻辑
}
```

**正确模式**：
```cpp
// 统一走 IR Visitor，所有信息来自 IR
void GenerateGenericKernelMain(PrimFunc f) {
  // 参数加载从 f->params 生成
  // 函数体通过 VisitStmt(f->body) 生成
  // Builtin 调用通过 HandleBlackholeBuiltin 分发
}
```

### 4. Pattern: 用 Op 比较代替字符串匹配

**反模式**（避免）：
```cpp
// 不要这样：字符串匹配脆弱，容易误判
bool IsMatmulCall(const CallNode* op) {
  std::string name = Downcast<Op>(op->op)->name;
  return name.find("gemm") != std::string::npos ||  // 会误判！
         name.find("matmul") != std::string::npos;
}
```

**正确模式**：
```cpp
// 直接 Op 对象比较
bool IsMatmulCall(const CallNode* op) {
  static const Op& tl_matmul = Op::Get("tl.matmul");
  static const Op& tl_gemm = Op::Get("tl.gemm");
  Op call_op = Downcast<Op>(op->op);
  return call_op.same_as(tl_matmul) || call_op.same_as(tl_gemm);
}
```

### 5. Pattern: attrs 合并而非覆盖

**反模式**（避免）：
```cpp
// 不要这样：会丢失其他 Pass 写入的 attrs
void StoreResult(PrimFunc& func, Result r) {
  Map<String, ObjectRef> new_attrs;
  new_attrs.Set("my_result", r);  // 只有 my_result
  func.CopyOnWrite()->attrs = DictAttrs(new_attrs);  // 覆盖所有！
}
```

**正确模式**：
```cpp
// 先读取现有 attrs，合并后再写回
void StoreResult(PrimFunc& func, Result r) {
  Map<String, ObjectRef> attrs;
  if (func->attrs.defined()) {
    attrs = func->attrs->dict;  // 保留现有
  }
  attrs.Set("my_result", r);     // 添加新的
  func.CopyOnWrite()->attrs = DictAttrs(attrs);
}
```

### 6. Pattern: 实例变量代替 static 变量

**反模式**（避免）：
```cpp
// 不要这样：static 变量在进程生命周期内不重置
void AddFunction(...) {
  static bool headers_emitted = false;
  if (!headers_emitted) {
    EmitHeaders();
    headers_emitted = true;  // 下次调用不会重置！
  }
}
```

**正确模式**：
```cpp
// 使用实例变量
class CodeGenBlackhole {
 private:
  bool headers_emitted_{false};  // 每个实例独立
};

void CodeGenBlackhole::Init(...) {
  headers_emitted_ = false;  // 重置状态
}
```

**头文件管理**:
```cpp
// 根据 core 类型选择头文件
switch (core_type_) {
  case CoreType::kBRISC:
  case CoreType::kNCRISC:
    decl_stream << "#include \"dataflow_api.h\"\n";
    break;
  case CoreType::kTRISC:
    decl_stream << "#include \"compute_kernel_api.h\"\n";
    decl_stream << "#include \"dataflow_api.h\"\n";
    break;
}
```

### 3. 测试验证

**生成代码验证**:
```bash
# E2E 测试
python tests/target/test_blackhole_e2e.py
# 输出: ✓ Build successful! Generated code length: 1743 chars

# GEMM 测试
python tests/target/test_blackhole_gemm_true_e2e.py
# 输出: ✓ ALL TESTS PASSED
```

### 4. 经验教训

1. **不要假设继承能解决问题**: TT-Metal kernel 格式与普通C差异太大，必须重写核心方法
2. **分离声明和实现**: 头文件在 `decl_stream`，实现在 `stream`
3. **静态标志防止重复**: 使用 `static bool headers_emitted` 避免重复包含头文件
4. **测试驱动开发**: 先写测试验证格式，再实现代码生成

---

## 端到端验证经验（2026-03-16）

### 分层验证策略

**三层验证架构**:

```
┌─────────────────────────────────────────────────────────────────┐
│ 第一层: CodeGen 正确性                                          │
│ - 验证生成的代码是否符合 TT-Metal 格式                          │
│ - 检查函数入口、参数获取、头文件包含                            │
│ - 工具: `test_blackhole_e2e.py`                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 第二层: 参考数据生成（PyTorch）                                 │
│ - 生成参考输入/输出数据                                          │
│ - 使用 PyTorch/NumPy 计算参考结果                                │
│ - 保存参考数据供后续对比使用                                     │
│ - ⚠️ 注意：当前未实际执行 kernel，仅生成参考                     │
│ - 工具: `test_blackhole_true_e2e.py`（命名有误，实际非E2E）     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 第三层: 真正的端到端测试（TT-Sim/Hardware）                     │
│ - Runtime 执行 kernel 并返回结果到 Python                        │
│ - 编译 kernel 到 RISC-V ELF                                      │
│ - 在 TT-Sim 或实际硬件上执行                                     │
│ - Python 端对比执行结果与参考结果（np.allclose）                 │
│ - 工具: 需要完整的 Runtime 实现（BlackholeModule.Execute）       │
└─────────────────────────────────────────────────────────────────┘
```

**重要区分**:
- **分层验证**（当前）：DSL → CodeGen + 参考生成（前两层）
- **真正端到端**（待实现）：DSL → CodeGen → Execute → Compare

**优点**:
- 前两层可以在无硬件环境下完成验证
- 问题分层定位：CodeGen 问题 vs 算法问题 vs 执行问题
- PyTorch 作为金标准，验证结果可信

### TT-Metal 库集成策略对比

| 方案 | 优点 | 缺点 | 选择 |
|------|------|------|------|
| A: 直接链接 TT-Metal | 性能最好，控制最细 | 依赖链复杂，编译困难 | ❌ 放弃 |
| B: 外部进程调用 | 实现简单，隔离性好 | 进程间通信开销 | ✅ 选择 |
| C: Python ctypes | 灵活性高 | 需要维护 C 接口 | ⏳ 备选 |

**选择方案 B 的原因**:
- TT-Metal 依赖 fmt, nlohmann_json, tt_stl, umd 等多个库
- CMake 配置复杂，头文件路径难以完全正确
- 外部进程调用可以保持 TileLang 编译简单
- 与 Phase 1 成功的测试模式一致

### 关键测试用例设计

**最小可行测试集**（⚠️ 当前仅实现分层验证，真正端到端待完成）：

1. **Copy Kernel**: 验证基本数据传输
   - 输入: FP16 向量
   - 输出: 相同向量
   - 当前: 仅生成代码 + 参考数据（未执行kernel）
   - 目标: Runtime 执行后逐元素相等

2. **Element-wise Add**: 验证简单计算
   - 输入: 两个 FP16 向量
   - 输出: 逐元素相加结果
   - 当前: 仅生成代码 + 参考数据（未执行kernel）
   - 目标: Runtime 执行后与 PyTorch 参考对比

3. **GEMM**: 验证复杂计算
   - 输入: 两个 FP16 矩阵
   - 输出: FP32 累加结果
   - 当前: lower pass 失败
   - 目标: Runtime 执行后与 PyTorch 参考对比

**测试文件位置**:
```
tests/target/
├── test_blackhole_e2e.py          # 第一层: CodeGen 正确性
├── test_blackhole_true_e2e.py     # ⚠️ 非真正E2E，实际仅生成参考数据
└── test_blackhole_gemm_e2e.py     # GEMM 专用测试
```

**真正端到端测试待实现**:
- DSL → TIR → CodeGen → Runtime Execute → Python Compare
- 需要: BlackholeModule.Execute() + 结果回传 + np.allclose 对比

### 环境配置要点

**编译环境**:
```bash
# TileLang 编译
export TT_METAL_HOME=/path/to/tt_metal_repo
cmake -B tilelang_repo/build -S tilelang_repo \
  -DUSE_BLACKHOLE=ON \
  -DCMAKE_BUILD_TYPE=Release
make -C tilelang_repo/build -j32
```

**运行时环境**:
```bash
# Python 路径
export PYTHONPATH=/path/to/tilelang_repo:$PYTHONPATH

# 库路径（仅 Runtime 需要）
export LD_LIBRARY_PATH=$TT_METAL_HOME/build/lib:$LD_LIBRARY_PATH
```

**TT-Sim 环境**:
```bash
export TT_METAL_SIMULATOR_HOME="/work/ttsim"
export TT_METAL_SIMULATOR="/work/ttsim/libttsim.so"
export TT_METAL_SLOW_DISPATCH_MODE=1
```

---

---

## 外部进程执行模式经验（2026-03-16）

### 背景

直接链接 TT-Metal 库到 TileLang 遇到复杂依赖链问题：
- TT-Metal 依赖 fmt, nlohmann_json, tt_stl, umd 等多个库
- CMake 配置复杂，头文件路径难以完全正确
- 编译失败率高，调试困难

### 解决方案：外部进程执行模式

**架构设计**:

```
TileLang Python
    ↓
BlackholeModule (C++)
    ↓ fork/exec
 tilelang_blackhole_runner (独立可执行文件)
    ↓
TT-Metal Runtime (libtt_metal.so)
    ↓
TT-Sim / Hardware
```

**通信协议**:

1. **Kernel Code**: 通过文件传递 (`.cpp`)
2. **Input Data**: 通过二进制文件传递 (`.bin`)
3. **Output Data**: 通过二进制文件传递 (`.bin`)
4. **Command Line**: 参数传递 (`argv[]`)

**优点**:
- ✅ 避免复杂的依赖链问题
- ✅ 独立的 runner 可单独测试和调试
- ✅ 保持 TileLang 编译简单
- ✅ 与 Phase 1 成功的测试模式一致

### 实现要点

**BlackholeModule 实现**:

```cpp
class BlackholeModuleNode : public ffi::ModuleObj {
 public:
  void ExecuteExternal(const std::string& func_name,
                       const std::vector<DLTensor*>& inputs,
                       const std::vector<uint32_t>& scalar_args,
                       const std::vector<DLTensor*>& outputs) {
    // 1. 保存 kernel 代码到文件
    std::string kernel_path = kernel_dir_ + "/" + func_name + "_kernel.cpp";
    SaveKernelCode(kernel_path, info.kernel_code);

    // 2. 准备输入数据文件
    std::string input_path = tmp_dir + "/input.bin";
    WriteInputData(input_path, inputs);

    // 3. fork/exec 外部 runner
    pid_t pid = fork();
    if (pid == 0) {
      // Child process
      execl(runner_path_.c_str(), runner_path_.c_str(),
            kernel_path.c_str(), input_path.c_str(), output_path.c_str(),
            input_size_str.c_str(), output_size_str.c_str(),
            nullptr);
      _exit(1);
    }

    // 4. 等待执行完成
    waitpid(pid, &status, 0);

    // 5. 读取输出数据
    ReadOutputData(output_path, outputs);
  }
};
```

**Runner 实现** (tilelang_blackhole_runner):

```cpp
int main(int argc, char* argv[]) {
  // 解析参数
  std::string kernel_path = argv[1];
  std::string input_path = argv[2];
  std::string output_path = argv[3];
  size_t input_size = std::stoul(argv[4]);
  size_t output_size = std::stoul(argv[5]);

  // 初始化 TT-Metal
  auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);

  // 创建 Program 和 CB
  Program program = CreateProgram();
  CircularBufferConfig cb_config = ...;
  CreateCircularBuffer(program, core, cb_config);

  // 加载并执行 kernel
  KernelHandle kernel = CreateKernel(program, kernel_path, core, config);

  // 执行并读取结果
  EnqueueMeshWorkload(cq, workload, false);
  EnqueueReadMeshBuffer(cq, output_data, output_buffer, true);

  // 写入输出文件
  write_file(output_path, output_data);

  return 0;
}
```

**标量参数提取** (TVM FFI):

```cpp
uint32_t ExtractScalar(const ffi::AnyView& arg, DLDataType dtype) {
  if (dtype.code == kDLInt) {
    auto opt = arg.try_cast<int64_t>();
    if (opt.has_value()) return static_cast<uint32_t>(opt.value());
  }
  if (dtype.code == kDLFloat) {
    auto opt = arg.try_cast<double>();
    if (opt.has_value()) {
      float f = static_cast<float>(opt.value());
      return *reinterpret_cast<uint32_t*>(&f);
    }
  }
  // ...
}
```

### 关键注意点

1. **TVM Target Context**: `lower()` 需要在 target context 中执行
   ```cpp
   with target:
       artifact = lower(kernel, target=target)
   ```

2. **文件路径**: 使用临时目录，避免权限问题
   ```cpp
   std::string tmp_dir = "/tmp/tilelang_blackhole_" + std::to_string(getpid());
   ```

3. **数据格式**: 使用二进制文件直接传递原始字节
   ```cpp
   input_file.write(static_cast<char*>(tensor->data), size);
   ```

4. **错误处理**: Runner 返回非零退出码表示失败
   ```cpp
   if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
     LOG(FATAL) << "External runner failed";
   }
   ```

### 测试验证

**单元测试**:
```bash
# 运行 Blackhole E2E 测试
export TT_METAL_HOME=/path/to/tt_metal_repo
pytest testing/python/target/blackhole/test_blackhole_e2e.py -v

# 预期输出:
# test_blackhole_codegen_only PASSED
# test_blackhole_true_e2e SKIPPED (需要 TT_METAL_SIMULATOR=1)
# test_blackhole_kernel_compilation SKIPPED (需要 CodeGen 完善)
```

**手动测试 Runner**:
```bash
# 直接运行 runner
./tilelang_blackhole_runner kernel.cpp input.bin output.bin 2048 2048
```

### 状态更新

- ✅ BlackholeModule 外部进程执行实现完成
- ✅ tilelang_blackhole_runner 编译成功 (705KB)
- ✅ Python E2E 测试框架 (testing/python/target/blackhole/)
- ⏳ 完整 TT-Sim 验证（需 CodeGen 完善后）

*2026-03-16: BlackholeModule 外部进程执行模式实现完成*
*2026-03-16: CodeGenBlackhole 重写完成，生成 kernel_main 格式*
