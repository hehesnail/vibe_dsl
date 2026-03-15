# Phase 1: CodeGen 框架 - 单核 Copy

## 任务目标

实现 CodeGenBlackhole 的具体代码生成逻辑，支持单核 Copy 算子，生成可在 TT-Sim 上运行的 TT-Metal C++ kernel。

## 背景

TileLang 的 Copy 算子在 Blackhole 后端需要拆分为：
- **Reader Kernel** (BRISC): 从 DRAM 读取数据到 Circular Buffer
- **Writer Kernel** (NCRISC): 从 Circular Buffer 写入数据到 DRAM

单核 Copy 是最简单的场景：一个 Tensix Core 执行 Reader + Writer。

## 技术方案

### TT-Metal Copy Kernel 结构

```cpp
// Reader Kernel (BRISC)
#include "dataflow_api.h"

void kernel_main() {
    // 运行时参数
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);

    // CB 配置
    constexpr uint32_t cb_id = 0;
    constexpr uint32_t tile_size = 32 * 32 * 2;  // FP16 tile

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_reserve_back(cb_id, 1);
        uint32_t write_ptr = get_write_ptr(cb_id);

        // NoC 异步读取
        noc_async_read(src_addr + i * tile_size, write_ptr, tile_size);
        noc_async_read_barrier();

        cb_push_back(cb_id, 1);
    }
}

// Writer Kernel (NCRISC)
#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_id = 0;
    constexpr uint32_t tile_size = 32 * 32 * 2;

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_id, 1);
        uint32_t read_ptr = get_read_ptr(cb_id);

        noc_async_write(read_ptr, dst_addr + i * tile_size, tile_size);
        noc_async_write_barrier();

        cb_pop_front(cb_id, 1);
    }
}
```

### CodeGen 策略

1. **Reader CodeGen**: 生成 dataflow_api.h 风格的 kernel
   - 处理 `cb_reserve_back`, `noc_async_read`, `cb_push_back`

2. **Writer CodeGen**: 生成 dataflow_api.h 风格的 kernel
   - 处理 `cb_wait_front`, `noc_async_write`, `cb_pop_front`

3. **单核执行**: Reader 和 Writer 在同一个 Core 上顺序执行（通过 CB 同步）

## 实施步骤

### 1. 更新 CodeGenBlackhole 类

修改 `src/target/codegen_blackhole.h/cc`：

```cpp
// 新增方法
class CodeGenBlackhole : public CodeGenCHost {
 public:
  // 生成 Reader Kernel
  std::string GenerateReaderKernel(const tvm::tir::PrimFunc& func);

  // 生成 Writer Kernel
  std::string GenerateWriterKernel(const tvm::tir::PrimFunc& func);

  // 生成完整的 Copy kernel（Reader + Writer）
  std::string GenerateCopyKernel(const tvm::tir::PrimFunc& func);

 private:
  // TT-Metal 特定代码生成
  void PrintCBReserveBack(const std::string& cb_name, int num_tiles);
  void PrintCBPushBack(const std::string& cb_name, int num_tiles);
  void PrintCBWaitFront(const std::string& cb_name, int num_tiles);
  void PrintCBPopFront(const std::string& cb_name, int num_tiles);
  void PrintNOCRead(const std::string& src, const std::string& dst, int size);
  void PrintNOCWrite(const std::string& src, const std::string& dst, int size);
};
```

### 2. 实现 Copy CodeGen

**Reader 生成逻辑**：
```cpp
std::string CodeGenBlackhole::GenerateReaderKernel(const PrimFunc& func) {
  std::ostringstream os;

  // 头文件
  os << "#include \"dataflow_api.h\"\n\n";

  // kernel_main 函数
  os << "void kernel_main() {\n";

  // 获取参数
  os << "  uint32_t src_addr = get_arg_val<uint32_t>(0);\n";
  os << "  uint32_t num_tiles = get_arg_val<uint32_t>(1);\n";
  os << "  constexpr uint32_t cb_id = 0;\n";
  os << "  constexpr uint32_t tile_size = 32 * 32 * 2;\n\n";

  // 循环
  os << "  for (uint32_t i = 0; i < num_tiles; i++) {\n";
  os << "    cb_reserve_back(cb_id, 1);\n";
  os << "    uint32_t write_ptr = get_write_ptr(cb_id);\n";
  os << "    noc_async_read(src_addr + i * tile_size, write_ptr, tile_size);\n";
  os << "    noc_async_read_barrier();\n";
  os << "    cb_push_back(cb_id, 1);\n";
  os << "  }\n";

  os << "}\n";
  return os.str();
}
```

### 3. 注册 Build 函数

在 `rt_mod_blackhole.cc` 中注册：

```cpp
// Build 函数
Module BuildTileLangBlackhole(IRModule mod, Target target) {
  // 遍历 mod 中的函数
  // 使用 CodeGenBlackhole 生成代码
  // 返回 Module
}

// 注册
TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::ffi::reflection::GlobalDef().def(
      "target.build.tilelang_blackhole", BuildTileLangBlackhole);
}
```

### 4. 创建测试

**测试文件**: `tests/target/test_codegen_blackhole.cc`

```cpp
TEST(CodeGenBlackhole, SimpleCopy) {
  // 创建一个简单的 Copy PrimFunc
  // 调用 CodeGenBlackhole::GenerateCopyKernel
  // 验证生成的代码包含 dataflow_api.h
  // 验证包含 cb_reserve_back, noc_async_read 等
}
```

**Python 集成测试**: `tests/blackhole/test_blackhole_copy.py`

```python
def test_simple_copy():
    @T.prim_func
    def copy_kernel(A: T.Buffer((32, 32), "float16"),
                    B: T.Buffer((32, 32), "float16")):
        T.copy(A, B)

    # 编译到 blackhole target
    # 生成代码应该在 TT-Sim 上可运行
```

## 验证方法

### 1. 单元测试

```bash
# 编译测试
./build/tests/test_codegen_blackhole

# 验证生成的代码语法正确
# 验证包含必要的 TT-Metal API 调用
```

### 2. 代码生成检查

手动检查生成的代码：
```cpp
// 应该生成类似这样的代码
#include "dataflow_api.h"

void kernel_main() {
    // 参数解析
    // CB 操作
    // NoC 操作
}
```

### 3. TT-Sim 验证（可选，Phase 1.5 完整测试）

- 生成完整可执行程序
- 在 TT-Sim 上运行
- 验证数据正确拷贝

## 预期产出

1. 更新的 `codegen_blackhole.h/cc` - 支持 Copy CodeGen
2. 更新的 `rt_mod_blackhole.cc` - 注册 Build 函数
3. `tests/target/test_codegen_blackhole.cc` - 单元测试
4. `tests/blackhole/test_blackhole_copy.py` - Python 集成测试框架

## 参考

- TT-Metal dataflow_api.h 文档
- `tt_metal/programming_examples/` 中的 copy 示例
- TileLang CUDA CodeGen 实现模式
