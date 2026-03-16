# Phase 1: Runtime 框架 - 运行时模块与 Build 集成

## 任务目标

实现 Blackhole 后端的 Runtime 框架，包括：
1. `BuildTileLangBlackhole` 函数 - 连接 TVM build 系统
2. `BlackholeDeviceAPI` - 设备 API 实现
3. 与 CodeGen 的集成 - 生成 TT-Metal 代码
4. 测试框架 - 验证 Runtime 功能

## 背景

TileLang 的编译流程最后一步是通过 `target.build.xxx` 将 TIR 转换为可执行代码。对于 Blackhole 后端，需要：

1. **Build 函数**: 将 IRModule 传递给 CodeGen，生成 TT-Metal C++ 代码
2. **Runtime Module**: 包装生成的代码，支持后续 JIT 编译
3. **Device API**: 提供设备查询、内存分配等功能

## 技术方案

### Build 函数实现

参考 CUDA 后端的 `BuildTileLangCUDA`，实现 `BuildTileLangBlackhole`：

```cpp
ffi::Module BuildTileLangBlackhole(IRModule mod, Target target) {
  // 1. 初始化 CodeGen
  CodeGenBlackhole cg;
  cg.Init(output_ssa, emit_asserts, emit_fwd_func_decl, target->str(), devices);

  // 2. 遍历并处理所有 PrimFunc
  for (auto kv : mod->functions) {
    cg.AddFunction(gvar, f);
  }

  // 3. 生成代码
  std::string code = cg.Finish();

  // 4. 创建 Runtime Module
  return CSourceModuleCreate(code, "cc", func_names, {});
}
```

### Device API 实现

`BlackholeDeviceAPI` 继承自 TVM 的 `DeviceAPI`，实现以下功能：

| 方法 | 功能 | Blackhole 实现 |
|------|------|----------------|
| `SetDevice` | 选择设备 | 单设备，no-op |
| `GetAttr` | 查询设备属性 | 返回 140 cores, 1.5MB L1 等 |
| `AllocDataSpace` | 分配内存 | 暂时使用 posix_memalign |
| `FreeDataSpace` | 释放内存 | 暂时使用 free |
| `StreamSync` | 流同步 | 同步执行，no-op |

### 注册机制

通过 `TVM_FFI_STATIC_INIT_BLOCK` 注册：

```cpp
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("target.build.tilelang_blackhole", BuildTileLangBlackhole)
      .def("target.build.tilelang_blackhole_without_host", BuildTileLangBlackholeWithoutHost)
      .def("device_api.blackhole", []() -> void* {
        return static_cast<void*>(BlackholeDeviceAPI::Global());
      });
}
```

## 实施步骤

### 1. 更新 rt_mod_blackhole.cc

添加 Build 函数和注册：

```cpp
// Build function for Blackhole target
ffi::Module BuildTileLangBlackhole(IRModule mod, Target target) {
  LOG(INFO) << "BuildTileLangBlackhole: Generating TT-Metal code...";

  // Initialize CodeGen
  tl::CodeGenBlackhole cg;
  cg.Init(output_ssa, emit_asserts, emit_fwd_func_decl, target->str(), devices);

  // Process all functions
  for (auto kv : mod->functions) {
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<tir::PrimFunc>(kv.second);
    cg.AddFunction(gvar, f);
  }

  // Generate code
  std::string code = cg.Finish();

  // Create module
  ffi::Array<ffi::String> func_names;
  // ... populate func_names
  return CSourceModuleCreate(code, "cc", func_names, {});
}
```

### 2. 完善 Device API

实现设备属性查询：

```cpp
void GetAttr(Device dev, DeviceAttrKind kind, ffi::Any* rv) final {
  switch (kind) {
    case kMaxSharedMemoryPerBlock:
      rv->operator=(1572864);  // 1.5 MB L1 per core
      break;
    case kMultiProcessorCount:
      rv->operator=(140);  // 140 Tensix cores
      break;
    // ... other attributes
  }
}
```

### 3. 创建测试

**Runtime 测试** (`tests/target/test_runtime_blackhole.cc`):

- Device API 单例测试
- 设备属性验证
- Build 函数签名检查
- 生成代码结构验证

### 4. 编译验证

```bash
# Compile test
g++ -std=c++17 tests/target/test_runtime_blackhole.cc \
    -o tests/target/test_runtime_blackhole

# Run test
./tests/target/test_runtime_blackhole
```

## 文件变更

### 新增/修改文件

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `tilelang_repo/src/target/rt_mod_blackhole.cc` | 修改 | 添加 Build 函数和注册 |
| `tests/target/test_runtime_blackhole.cc` | 新增 | Runtime 测试 |

### 关键代码片段

**Build 函数** (rt_mod_blackhole.cc:170-260):
```cpp
ffi::Module BuildTileLangBlackhole(IRModule mod, Target target) {
  // 初始化 CodeGen
  bool output_ssa = false;
  bool emit_asserts = false;
  bool emit_fwd_func_decl = true;
  std::unordered_set<std::string> devices;
  devices.insert("blackhole");

  tl::CodeGenBlackhole cg;
  cg.Init(output_ssa, emit_asserts, emit_fwd_func_decl, target->str(), devices);

  // 处理所有函数
  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<tir::PrimFuncNode>())
        << "CodeGenBlackhole: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<tir::PrimFunc>(kv.second);
    cg.AddFunction(gvar, f);
  }

  std::string code = cg.Finish();

  // 提取函数信息
  ffi::Array<ffi::String> func_names;
  auto func_info = ExtractFuncInfo(mod);
  for (const auto& kv : func_info) {
    func_names.push_back(kv.first);
  }

  return CSourceModuleCreate(code, "cc", func_names, {});
}
```

**注册** (rt_mod_blackhole.cc:305-310):
```cpp
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("target.build.tilelang_blackhole", BuildTileLangBlackhole)
      .def("target.build.tilelang_blackhole_without_host", BuildTileLangBlackholeWithoutHost)
      .def("device_api.blackhole", []() -> void* {
        return static_cast<void*>(BlackholeDeviceAPI::Global());
      });
}
```

## 验证方法

### 1. 单元测试

```bash
# 编译测试
g++ -std=c++17 tests/target/test_runtime_blackhole.cc \
    -o tests/target/test_runtime_blackhole

# 运行测试
./tests/target/test_runtime_blackhole
```

**预期输出**:
```
======================================
Blackhole Runtime Module Test Suite
======================================

=== Test: Device API Singleton ===
[PASS] Device API Singleton

=== Test: Device Attributes ===
Device attributes verified:
  - L1 per core: 1572864 bytes (1.5 MB)
  - Tensix cores: 140
[PASS] Device Attributes

...

Test Summary: 6 passed, 0 failed
```

### 2. 编译验证

验证 `rt_mod_blackhole.cc` 可以编译：

```bash
g++ -std=c++17 -c src/target/rt_mod_blackhole.cc \
  -I. -I./src \
  -I./3rdparty/tvm/include \
  -I./3rdparty/tvm/src \
  -I./3rdparty/tvm/3rdparty/tvm-ffi/include \
  -I./3rdparty/tvm/3rdparty/tvm-ffi/3rdparty/dlpack/include \
  -I./3rdparty/tvm/3rdparty/dmlc-core/include \
  -o /tmp/rt_mod_blackhole.o
```

### 3. 集成测试（后续阶段）

```python
# Python 集成测试
import tilelang as tl

# 编译到 blackhole target
mod = tl.build(func, target="blackhole")

# 验证返回的模块包含 TT-Metal 代码
code = mod.get_source()
assert "dataflow_api.h" in code
assert "cb_reserve_back" in code
```

## 实际产出

1. ✅ **更新的 `rt_mod_blackhole.cc`** - Build 函数和 Device API 实现
2. ✅ **测试文件 `test_runtime_blackhole.cc`** - Runtime 测试
3. ✅ **编译通过的验证** - 无编译错误
4. ✅ **TT-Sim 手动 Kernel 测试** - 运行成功（⚠️ 非 TileLang DSL 生成的代码）

## TT-Sim 手动 Kernel 验证

**状态**: ✅ 2026-03-16 完成（⚠️ 手动编写，非 TileLang DSL 生成）

- 测试程序: `phase1_tilelang_ttsim`
- 测试内核: 手动编写的 `phase1_copy_kernel.cpp`
- 验证数据: 4096 个 FP16 元素全部正确复制
- 详细报告: [PHASE1_TTSIM_TEST_REPORT](../../tests/target/PHASE1_TTSIM_TEST_REPORT.md)

**关键意义**: 证明 TT-Sim 环境配置正确，可作为 TileLang DSL 生成代码的执行目标

**JIT 环境修复**:
- 创建了 7 个符号链接解决编译路径问题
- 详细记录: `memory/bugs.md` - "TT-Sim JIT 编译环境缺失文件"

**真正的端到端测试待实现**:
- TileLang DSL → TIR → CodeGen → Runtime Execute → Python Compare
- 当前状态: CodeGen 可生成代码，但 Runtime 未接入 TT-Metal 执行

## 后续工作

### Phase 2 准备

1. **多核拆分**: SplitBlackholeKernel Pass
2. **CB 分配**: PlanBlackholeCB Pass
3. **核分配**: AssignBlackholeCores Pass
4. **TT-Sim 多核验证**

### 可能的改进

1. 实现真正的 TT-Metal 内存分配
2. 添加更多设备属性查询
3. 支持多设备配置
4. 添加性能 profiling 接口

## 经验总结

### 关键决策及原因

**1. 延迟初始化 (Lazy Initialization)**
- **决策**: Device API 使用单例模式，首次调用时初始化
- **原因**: 避免静态初始化顺序问题，TT-Metal 环境变量需要在 main 之后设置
- **实现**: `BlackholeDeviceAPI::Global()` 返回静态局部变量

**2. 暂时使用 posix_memalign 替代 TT-Metal 内存分配**
- **决策**: Phase 1 使用标准库内存分配
- **原因**: 先打通端到端流程，真实的 DRAM 分配在后续阶段实现
- **计划**: Phase 3/4 集成真正的 `tt::tt_metal::Buffer` 分配

**3. Build 函数与 CodeGen 分离**
- **决策**: `BuildTileLangBlackhole` 负责流程控制，CodeGen 负责代码生成
- **原因**: 单一职责，便于测试和替换 CodeGen 实现
- **收益**: 可以独立测试 CodeGen 不依赖 TVM Runtime

**4. 使用 `CSourceModuleCreate` 而非自定义 Module**
- **决策**: 复用 TVM 现有的 CSourceModule 机制
- **原因**: 简化实现，自动获得代码缓存、编译管理等功能
- **注意**: 参数类型必须是 `ffi::Array<ffi::String>`

### 可复用的模式

**1. Build 函数标准结构**
```cpp
ffi::Module BuildTileLangXXX(IRModule mod, Target target) {
  // 1. 初始化 CodeGen
  XXXCodeGen cg;
  cg.Init(output_ssa, emit_asserts, emit_fwd_func_decl, target->str(), devices);

  // 2. 处理所有 PrimFunc
  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<tir::PrimFuncNode>())
        << "Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<tir::PrimFunc>(kv.second);
    cg.AddFunction(gvar, f);
  }

  // 3. 生成代码
  std::string code = cg.Finish();

  // 4. 提取函数名
  ffi::Array<ffi::String> func_names;
  auto func_info = ExtractFuncInfo(mod);
  for (const auto& kv : func_info) {
    func_names.push_back(kv.first);
  }

  // 5. 创建模块
  return CSourceModuleCreate(code, "cc", func_names, {});
}
```

**2. Device API 单例模式**
```cpp
class BlackholeDeviceAPI final : public DeviceAPI {
 public:
  void* AllocDataSpace(Device dev, size_t nbytes, size_t alignment,
                       DLDataType type_hint) final {
    void* ptr;
    if (posix_memalign(&ptr, alignment, nbytes) != 0) {
      throw std::bad_alloc();
    }
    return ptr;
  }

  void FreeDataSpace(Device dev, void* ptr) final {
    free(ptr);
  }

  static BlackholeDeviceAPI* Global() {
    static BlackholeDeviceAPI* inst = new BlackholeDeviceAPI();
    return inst;
  }
};
```

**3. TVM FFI 注册模式**
```cpp
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("target.build.tilelang_blackhole", BuildTileLangBlackhole)
      .def("target.build.tilelang_blackhole_without_host",
           BuildTileLangBlackholeWithoutHost)
      .def("device_api.blackhole", []() -> void* {
        return static_cast<void*>(BlackholeDeviceAPI::Global());
      });
}
```

**4. 设备属性映射模式**
```cpp
void GetAttr(Device dev, DeviceAttrKind kind, ffi::Any* rv) final {
  switch (kind) {
    case kMaxSharedMemoryPerBlock:
      // Blackhole L1: 1.5 MB per core
      rv->operator=(1572864);
      break;
    case kMultiProcessorCount:
      // Blackhole Tensix cores: 140
      rv->operator=(140);
      break;
    case kMaxThreadsPerBlock:
      // 单个 core 的计算单元数
      rv->operator=(1024);
      break;
    // ... 其他属性
    default:
      *rv = ffi::Any(nullptr);  // 不支持的属性返回 null
  }
}
```

### 踩过的坑

**1. `CSourceModuleCreate` 参数类型不匹配**
- **问题**: 使用 `std::vector<std::string>` 传递函数名列表
- **错误**: `invalid initialization of reference of type 'const tvm::ffi::Array<tvm::ffi::String>&'`
- **解决**: 使用 `ffi::Array<ffi::String>` 类型
```cpp
// 错误
std::vector<std::string> func_names;
return CSourceModuleCreate(code, "cc", func_names, {});

// 正确
ffi::Array<ffi::String> func_names;
for (...) { func_names.push_back(name); }
return CSourceModuleCreate(code, "cc", func_names, {});
```

**2. 缺少必要的头文件**
- **问题**: 编译时找不到 `CSourceModuleCreate` 声明
- **错误**: `'CSourceModuleCreate' was not declared in this scope`
- **解决**: 需要包含 `<tvm/target/codegen.h>`

**3. Device API 方法未实现**
- **问题**: 继承 `DeviceAPI` 但未实现所有纯虚方法
- **错误**: `abstract class cannot be instantiated`
- **解决**: 实现所有必需的虚方法，即使是 no-op

**4. TT-Sim 运行时路径问题**
- **问题**: TT-Metal JIT 编译时找不到 firmware 源文件
- **错误**: `fatal error: brisc.cc: No such file or directory`
- **根本原因**: build_Release 目录缺少源码符号链接
- **解决**: 创建 7 个符号链接（详见 `memory/bugs.md`）
```bash
# 关键链接
ln -sf $TT_METAL_HOME/tt_metal/hw $TT_METAL_HOME/build_Release/tt_metal/hw
ln -sf $TT_METAL_HOME/runtime/sfpi $TT_METAL_HOME/build_Release/runtime/sfpi
ln -sf $TT_METAL_HOME/tt_metal/third_party/tt_llk \
       $TT_METAL_HOME/build_Release/tt_metal/third_party/tt_llk
# ... 共 7 个
```

**5. 环境变量配置顺序**
- **问题**: 程序运行时找不到 TT-Sim 库
- **错误**: `cannot open shared object file: libttsim.so`
- **解决**: 必须在程序启动前设置 `LD_LIBRARY_PATH`
```bash
export LD_LIBRARY_PATH=$TT_METAL_HOME/build_Release/lib:$LD_LIBRARY_PATH
```

### 验证检查清单

Build 函数实现后验证：
- [ ] 编译通过，无警告
- [ ] `target.build.tilelang_blackhole` 可正常调用
- [ ] 生成的代码可通过 `mod.get_source()` 获取
- [ ] Device API 单例正常工作
- [ ] 设备属性返回正确值
- [ ] TT-Sim 端到端测试通过

### 与 CUDA 后端的差异

| 方面 | CUDA | Blackhole |
|------|------|-----------|
| Build 输出 | PTX/CUBIN | C++ source (TT-Metal) |
| 运行时编译 | Driver API | TT-Metal JIT |
| 内存分配 | `cudaMalloc` | `posix_memalign` (临时) |
| Kernel 启动 | `cuLaunchKernel` | `EnqueueProgram` |
| 设备属性 | CUDA 设备查询 | 硬编码 Blackhole 规格 |

### 后续改进方向

1. **真实 DRAM 分配**: 使用 `tt::tt_metal::Buffer` 替代 `posix_memalign`
2. **Command Queue 管理**: 实现真正的异步执行
3. **Profiling 接口**: 集成 TT-Metal profiler
4. **多设备支持**: 支持多芯片 Blackhole 配置

## 参考

- `rt_mod_cuda.cc` - CUDA 后端实现参考
- `rt_mod_cpp.cc` - CPP 后端实现参考
- TVM DeviceAPI 文档
- TT-Metal 设备管理 API
