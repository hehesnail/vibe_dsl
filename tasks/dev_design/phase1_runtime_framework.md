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

## 预期产出

1. **更新的 `rt_mod_blackhole.cc`** - Build 函数和 Device API 实现
2. **测试文件 `test_runtime_blackhole.cc`** - Runtime 测试
3. **编译通过的验证** - 无编译错误

## 后续工作

### Phase 2 准备

1. **JIT 编译集成**: 添加 TT-Metal 编译器调用
2. **Kernel 执行**: 实现设备端代码执行
3. **TT-Sim 验证**: 在仿真器上运行生成的代码

### 可能的改进

1. 实现真正的 TT-Metal 内存分配
2. 添加更多设备属性查询
3. 支持多设备配置
4. 添加性能 profiling 接口

## 参考

- `rt_mod_cuda.cc` - CUDA 后端实现参考
- `rt_mod_cpp.cc` - CPP 后端实现参考
- TVM DeviceAPI 文档
- TT-Metal 设备管理 API
