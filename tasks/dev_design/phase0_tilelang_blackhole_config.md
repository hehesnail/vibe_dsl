# Phase 0: TileLang + Blackhole 配置

## 任务目标

配置 TileLang 以支持 Blackhole (TT-Metal) 后端，使 TileLang DSL 可以编译生成 TT-Metal 代码。

## 背景知识

TileLang 是一个基于 TVM 的深度学习编译器，用于生成高性能 GPU kernel。它目前支持 CUDA、ROCm、Metal 后端。

我们需要为 TileLang 添加 Blackhole 后端支持，架构如下：

```
TileLang DSL (Python)
       ↓ LowerAndLegalize
TIR (TVM IR)
       ↓ Blackhole Passes
├─ AssignBlackholeCores (14x10 grid)
├─ PlanBlackholeCB (64 CBs, 1.5MB L1)
└─ SplitBlackholeKernel (R/C/W 拆分)
       ↓ CodeGen
TT-Metal C++ (BRISC/TRISC/NCRISC)
       ↓ JIT Build (libtt_metal.so)
RISC-V ELF
       ↓ Runtime
Blackhole Hardware (140 cores)
```

## 技术方案

### 现有后端分析

TileLang 现有后端结构：

| 文件 | 说明 |
|------|------|
| `codegen_cuda.cc/h` | CUDA 代码生成 |
| `codegen_hip.cc/h` | HIP/ROCm 代码生成 |
| `codegen_c_host.cc/h` | C host 代码生成 |
| `rt_mod_cuda.cc` | CUDA 运行时模块 |
| `rt_mod_hip.cc` | HIP 运行时模块 |

### Blackhole 后端设计

新增文件：

| 文件 | 说明 |
|------|------|
| `codegen_blackhole.h/cc` | Blackhole 代码生成 |
| `rt_mod_blackhole.cc` | Blackhole 运行时模块 |

### CMake 配置修改

在 `CMakeLists.txt` 中添加：

```cmake
set(TILELANG_BACKENDS CUDA ROCM METAL BLACKHOLE)

set(TILELANG_BACKEND_DOC_BLACKHOLE "Enable Blackhole backend (requires TT-Metal)")
option(USE_BLACKHOLE "Enable Blackhole backend" OFF)
```

## 实施步骤

### 1. 创建 CodeGen 框架文件

创建 `src/target/codegen_blackhole.h`（实际实现）：

```cpp
#ifndef TL_TARGET_SOURCE_CODEGEN_BLACKHOLE_H_
#define TL_TARGET_SOURCE_CODEGEN_BLACKHOLE_H_

#include <string>
#include <unordered_set>

#include "codegen_c_host.h"

namespace tvm {
namespace tl {

class CodeGenBlackhole : public CodeGenCHost {
 public:
  CodeGenBlackhole();

  // Note: Parent class Init is not virtual, so we just shadow it
  void Init(bool output_ssa, bool emit_asserts, bool emit_fwd_func_decl,
            std::string target_str,
            const std::unordered_set<std::string> &devices);

  void AddFunction(const tvm::GlobalVar &gvar,
                   const tvm::tir::PrimFunc &f) override;

  // Note: Visitor methods are marked 'final' in parent, cannot override
  // Blackhole-specific IR handling via preprocessing passes

  // Blackhole core type enumeration
  enum class CoreType {
    kBRISC,   // Broadcast RISC - control core
    kTRISC,   // Tensix RISC - compute core
    kNCRISC,  // NOC RISC - data movement core
    kUnknown
  };

  void SetCoreType(CoreType core_type) { core_type_ = core_type; }
  CoreType GetCoreType() const { return core_type_; }

 protected:
  void PrintKernelAttributes();
  void PrintCBDeclare(const std::string &name, tvm::DataType dtype,
                      int num_pages, int page_size);
  void PrintCBWaitFront(const std::string &name, int num_tiles);
  void PrintCBPopFront(const std::string &name, int num_tiles);
  void PrintCBReserveBack(const std::string &name, int num_tiles);
  void PrintCBPushBack(const std::string &name, int num_tiles);
  void PrintNOCRead(const std::string &src_addr, const std::string &dst_addr, int size);
  void PrintNOCWrite(const std::string &src_addr, const std::string &dst_addr, int size);
  void PrintNOCWait();
  void PrintSemInit(int sem_id, int value);
  void PrintSemWait(int sem_id, int value);
  void PrintSemPost(int sem_id);

 private:
  CoreType core_type_{CoreType::kUnknown};
  bool need_tt_metal_h_{false};
  bool need_dataflow_api_h_{false};
  bool need_compute_api_h_{false};
  bool emit_kernel_wrapper_{true};
  std::unordered_set<std::string> declared_cbs_;
  static constexpr int kL1Alignment = 16;
};

} // namespace tl
} // namespace tvm

#endif // TL_TARGET_SOURCE_CODEGEN_BLACKHOLE_H_
```

### 2. 创建 Runtime 模块

创建 `src/target/rt_mod_blackhole.cc`（简化版 DeviceAPI）：

```cpp
#include <tvm/runtime/device_api.h>
#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace runtime {

class BlackholeDeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(Device dev) final;
  void GetAttr(Device dev, DeviceAttrKind kind, ffi::Any* rv) final;
  void* AllocDataSpace(Device dev, size_t nbytes, size_t alignment,
                       DLDataType type_hint) final;
  void FreeDataSpace(Device dev, void* ptr) final;
  void* AllocWorkspace(Device dev, size_t size, DLDataType type_hint) final;
  void FreeWorkspace(Device dev, void* ptr) final;
  void StreamSync(Device dev, TVMStreamHandle stream) final;
  TVMStreamHandle CreateStream(Device dev) final;
  void FreeStream(Device dev, TVMStreamHandle stream) final;
  static BlackholeDeviceAPI* Global();

 private:
  static constexpr size_t kL1Alignment = 16;
};

// Registration via TVM_FFI_STATIC_INIT_BLOCK
TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::ffi::reflection::GlobalDef().def("device_api.blackhole", []() -> void* {
    return static_cast<void*>(BlackholeDeviceAPI::Global());
  });
}

}  // namespace runtime
}  // namespace tvm
```

### 3. 修改 CMakeLists.txt

在 `CMakeLists.txt` 中添加 Blackhole 支持：

```cmake
set(TILELANG_BACKENDS CUDA ROCM METAL BLACKHOLE)

# ... existing backend options ...
set(TILELANG_BACKEND_DOC_BLACKHOLE "Enable Blackhole (TT-Metal) backend")

# ... inside the backend selection logic ...
elseif(USE_BLACKHOLE)
  message(STATUS "Enable Blackhole (TT-Metal) backend")
  find_library(TT_METAL_LIB tt_metal PATHS ${TT_METAL_HOME}/build_Release/tt_metal)
  if(NOT TT_METAL_LIB)
    message(FATAL_ERROR "TT-Metal library not found. Set TT_METAL_HOME")
  endif()
  include_directories(${TT_METAL_HOME}/tt_metal ${TT_METAL_HOME}/tt_metal/third_party/umd)
  file(GLOB TILE_LANG_BLACKHOLE_SRCS
    src/target/codegen_blackhole.cc
    src/target/rt_mod_blackhole.cc
  )
  list(APPEND TILE_LANG_SRCS ${TILE_LANG_BLACKHOLE_SRCS})
```

### 4. 注册 Target

在 TVM 中注册 Blackhole target：

```python
# tilelang/python/target.py 或 TVM 的 target 注册
@tvm.target.register_target("blackhole")
def target_blackhole(target):
    return Target("blackhole -keys=blackhole -mtriple=riscv64-unknown-elf")
```

## 验证方法

### 1. 编译测试

```bash
# CMake 配置
cmake -B build_blackhole -DUSE_BLACKHOLE=ON -DUSE_CUDA=OFF

# 编译
cd build_blackhole && make -j4
```

**验证点**：
- `libtilelang.so` 生成成功
- 包含 Blackhole 符号：`nm -D lib/libtilelang.so | grep -i blackhole`

### 2. 功能测试（待实现）

当前阶段仅完成框架编译，功能测试需要：
- [ ] CodeGen 生成 TT-Metal 代码测试
- [ ] DeviceAPI 注册测试
- [ ] 简单的 kernel 编译测试（无需实际运行）

**示例测试代码**（待实现）：
```python
# test_blackhole_codegen.py
import tilelang as tl
import tvm

# 测试 target 注册
target = tvm.target.Target("blackhole")
print(f"Target: {target}")

# 测试简单的函数生成（无需完整编译）
@T.prim_func
def simple_add(A: T.Buffer((16,), "float32"),
               B: T.Buffer((16,), "float32"),
               C: T.Buffer((16,), "float32")):
    for i in T.serial(16):
        C[i] = A[i] + B[i]

# TODO: 测试代码生成功能
```

## 当前限制

- 仅完成 CodeGen 框架和 DeviceAPI 基础实现
- 未实现实际的 TT-Metal 代码生成逻辑
- 未实现 kernel 加载和执行功能
- 未连接 TT-Metal 库

## 预期产出

1. ✅ `src/target/codegen_blackhole.h/cc` - Blackhole 代码生成框架
2. ✅ `src/target/rt_mod_blackhole.cc` - Blackhole 运行时模块
3. ✅ `CMakeLists.txt` - 添加 USE_BLACKHOLE 支持
4. ✅ `tilelang.target("blackhole")` - Python target 注册

## 状态

- [x] CMake 配置修改
- [x] CodeGenBlackhole 框架实现
- [x] 编译测试 (✅ 2026-03-15 完成)
- [x] 问题修复 (✅ 2026-03-15 完成)

**验证结果**: `USE_BLACKHOLE=ON` 编译成功，`libtilelang.so` (21MB) 包含 Blackhole 符号

## 遇到的问题与解决

1. **问题**: `Init` 方法标记为 `override` 但父类方法不是 `virtual`
   - **解决**: 移除 `override` 关键字，改为普通方法重载

2. **问题**: `PrintFuncPrefix`, `PrintType`, `VisitStmt_` 等方法在父类中标记为 `final`
   - **解决**: 移除这些方法的 override，改用其他机制实现 Blackhole 特定功能

3. **问题**: `TVMContext` 类型不存在
   - **解决**: 使用 `Device` (即 `DLDevice`) 替代

4. **问题**: `tvm::String` 命名空间错误
   - **解决**: 使用 `tvm::ffi::String`

5. **问题**: `defined()` 方法在 `ffi::Optional` 上不存在
   - **解决**: 直接使用 `if (optional_value)` 语法

## 参考文档

- TileLang CUDA/HIP 后端实现 (`src/target/codegen_cuda.cc`, `codegen_hip.cc`)
- TVM Target 注册机制
- TT-Metal 官方示例
