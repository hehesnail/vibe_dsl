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

创建 `src/target/codegen_blackhole.h`：

```cpp
#ifndef TILELANG_CODEGEN_BLACKHOLE_H_
#define TILELANG_CODEGEN_BLACKHOLE_H_

#include <tvm/target/codegen.h>
#include "codegen_c_host.h"

namespace tilelang {

class CodeGenBlackhole : public CodeGenCHost {
 public:
  CodeGenBlackhole();
  virtual ~CodeGenBlackhole();

  // Override CodeGenCHost methods for Blackhole-specific codegen
  void Init(bool output_ssa, bool emit_asserts);
  void AddFunction(const PrimFunc& f);

  // Blackhole-specific code generation
  void VisitStmt_(const AttrStmtNode* op) override;
  void VisitExpr_(const CallNode* op) override;

 private:
  // TT-Metal specific state
  bool is_brisc_;  // BRISC core
  bool is_trisc_;  // TRISC core
  bool is_ncrisc_; // NCRISC core
};

}  // namespace tilelang

#endif  // TILELANG_CODEGEN_BLACKHOLE_H_
```

创建 `src/target/codegen_blackhole.cc`：

```cpp
#include "codegen_blackhole.h"

#include <tvm/ir/transform.h>
#include <tvm/tir/transform.h>

namespace tilelang {

CodeGenBlackhole::CodeGenBlackhole() : is_brisc_(false), is_trisc_(false), is_ncrisc_(false) {}

CodeGenBlackhole::~CodeGenBlackhole() {}

void CodeGenBlackhole::Init(bool output_ssa, bool emit_asserts) {
  CodeGenCHost::Init(output_ssa, emit_asserts);
}

void CodeGenBlackhole::AddFunction(const PrimFunc& f) {
  // TODO: Implement Blackhole-specific function generation
  CodeGenCHost::AddFunction(f);
}

void CodeGenBlackhole::VisitStmt_(const AttrStmtNode* op) {
  // TODO: Handle Blackhole-specific attributes
  CodeGenCHost::VisitStmt_(op);
}

void CodeGenBlackhole::VisitExpr_(const CallNode* op) {
  // TODO: Handle Blackhole-specific intrinsics
  CodeGenCHost::VisitExpr_(op);
}

}  // namespace tilelang
```

### 2. 创建 Runtime 模块

创建 `src/target/rt_mod_blackhole.cc`：

```cpp
#include <tvm/runtime/registry.h>
#include <tvm/runtime/c_runtime_api.h>

namespace tvm {
namespace runtime {

// Blackhole runtime module
// TODO: Implement TT-Metal device API integration

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

1. **CMake 配置**：`cmake -DUSE_BLACKHOLE=ON -DTT_METAL_HOME=/path/to/tt_metal ..`
2. **编译检查**：`ninja` 能成功编译
3. **Python 导入**：`import tilelang` 正常
4. **目标注册**：`tl.target("blackhole")` 可用

## 预期产出

1. ✅ `src/target/codegen_blackhole.h/cc` - Blackhole 代码生成框架
2. ✅ `src/target/rt_mod_blackhole.cc` - Blackhole 运行时模块
3. ✅ `CMakeLists.txt` - 添加 USE_BLACKHOLE 支持
4. ✅ `tilelang.target("blackhole")` - Python target 注册

## 参考文档

- TileLang CUDA/HIP 后端实现 (`src/target/codegen_cuda.cc`, `codegen_hip.cc`)
- TVM Target 注册机制
- TT-Metal 官方示例
