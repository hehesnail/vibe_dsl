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

### 3. TT-Sim 验证（已完成 ✅）

**状态**: 2026-03-16 完成

- ✅ 生成完整可执行程序
- ✅ 在 TT-Sim 上运行
- ✅ 验证数据正确拷贝 (4096 个 FP16 元素)

**详细报告**: [PHASE1_TTSIM_TEST_REPORT](../../tests/target/PHASE1_TTSIM_TEST_REPORT.md)

**关键实现**:
```cpp
// TT-Sim 兼容代码生成
void PrintNOCRead(const std::string& src_addr, const std::string& dst_addr, int size) {
  // 生成 InterleavedAddrGen 风格的代码
  stream << "InterleavedAddrGen<true> src_gen = {";
  stream << ".bank_base_address = " << src_addr << ", .page_size = " << size << "};";
  stream << "uint64_t src_noc_addr = get_noc_addr(i, src_gen);";
  stream << "noc_async_read(src_noc_addr, " << dst_addr << ", " << size << ");";
}
```

## 实际产出

1. ✅ 更新的 `codegen_blackhole.h/cc` - 支持 Copy CodeGen
2. ✅ 更新的 `rt_mod_blackhole.cc` - 注册 Build 函数
3. ✅ `tests/target/test_codegen_blackhole.cc` - 单元测试
4. ✅ `tests/target/phase1_ttsim_test.cpp` - TT-Sim 测试
5. ✅ `tests/target/PHASE1_TTSIM_TEST_REPORT.md` - 详细报告

## 经验总结

### 关键决策及原因

**1. 继承 CodeGenCHost 而非 CodeGenC**
- **决策**: 选择 `CodeGenCHost` 作为基类而非 `CodeGenC`
- **原因**: Blackhole 生成的是 Host 端可编译的 C++ 代码（通过 TT-Metal JIT 编译），不是直接生成设备汇编
- **教训**: 最初尝试覆盖 `PrintFuncPrefix` 等 `final` 方法导致编译错误，后改为在 `AddFunction` 中预处理

**2. TT-Sim 兼容代码优先**
- **决策**: 使用 `InterleavedAddrGen` + `get_noc_addr()` 而非直接物理地址
- **原因**: TT-Sim 不支持直接使用物理地址的 NOC 操作
- **收益**: 同一套代码可在仿真器和真实硬件上运行

**3. 单核 Copy 作为 MVP**
- **决策**: Phase 1 只实现单核 Copy，不处理多核并行
- **原因**: 降低复杂度，先打通端到端流程
- **验证**: 4096 个 FP16 元素全部正确复制，证明 CodeGen 逻辑正确

### 可复用的模式

**1. TVM FFI 类型使用模式**
```cpp
// 旧 TVM API vs 新 TVM FFI API
// String 类型
std::string str;                    // C++ 标准类型
ffi::String ffi_str = str;          // TVM FFI 类型

// Optional 类型判断
ffi::Optional<ffi::String> opt;
if (opt) { /* 有值 */ }             // 正确
if (opt.defined()) { /* 错误 */ }   // 旧 API 已废弃

// Array 类型
ffi::Array<ffi::String> func_names; // 用于 CSourceModuleCreate
```

**2. CodeGen 类设计模式**
```cpp
class CodeGenBlackhole : public CodeGenCHost {
 public:
  void Init(bool output_ssa, bool emit_asserts,
            bool emit_fwd_func_decl,
            const std::string& target_str,
            const std::unordered_set<std::string>& devices);

  void AddFunction(GlobalVar gvar, PrimFunc func);
  std::string Finish();

 private:
  std::ostringstream decl_stream_;  // 头文件部分
  std::ostringstream stream_;       // 代码主体
};
```

**3. TT-Metal Kernel 生成模板**
```cpp
// 标准 Reader Kernel 结构
void GenerateReaderTemplate(std::ostringstream& os) {
  os << "#include \"dataflow_api.h\"\n\n";
  os << "void kernel_main() {\n";
  os << "  uint32_t src_addr = get_arg_val<uint32_t>(0);\n";
  os << "  uint32_t num_tiles = get_arg_val<uint32_t>(1);\n";
  os << "  // CB/循环逻辑...\n";
  os << "}\n";
}
```

### 踩过的坑

**1. `override` 关键字误用**
- **问题**: 标记 `Init` 方法为 `override`，但父类方法非 `virtual`
- **错误**: `'void tvm::tl::CodeGenBlackhole::Init(...)' marked 'override', but does not override`
- **解决**: 移除 `override` 关键字

**2. 尝试覆盖 `final` 方法**
- **问题**: 尝试覆盖 `PrintFuncPrefix`, `VisitStmt_` 等标记为 `final` 的方法
- **错误**: `virtual function 'PrintFuncPrefix' overriding final function`
- **解决**: 不在 CodeGen 层覆盖，改用 IR Pass 预处理

**3. `CSourceModuleCreate` 参数类型**
- **问题**: 使用 `std::vector<std::string>` 传递函数名
- **错误**: `invalid initialization of reference of type 'const tvm::ffi::Array<tvm::ffi::String>&'`
- **解决**: 使用 `ffi::Array<ffi::String>` 类型

**4. TT-Sim 地址访问方式**
- **问题**: 直接使用 `src_addr + i * tile_size` 作为 NOC 地址
- **错误**: `UnimplementedFunctionality: noc_cmd_ctrl: write: src_addr=0x180000`
- **解决**: 使用 `InterleavedAddrGen` + `get_noc_addr()` 进行地址转换

**5. JIT 编译环境缺失**
- **问题**: TT-Sim 运行时 JIT 编译失败，缺少 firmware 源文件
- **错误**: `fatal error: brisc.cc: No such file or directory`
- **解决**: 创建 7 个符号链接将源码目录链接到 build 目录（详见 `memory/bugs.md`）

### 性能数据

| 指标 | 数值 | 说明 |
|------|------|------|
| 复制数据量 | 4096 FP16 元素 | 8KB 数据 |
| 验证结果 | 100% 正确 | 所有元素值匹配 |
| 首次端到端 | 2026-03-16 | Phase 1 里程碑 |

### 后续改进方向

1. **支持多核并行**: Phase 2 已实现（AssignBlackholeCores Pass）
2. **支持 Compute Kernel**: Phase 3 将实现 GEMM
3. **优化 CB 分配**: Phase 2 已实现（PlanBlackholeCB Pass）
4. **性能调优**: 双缓冲、流水线并行（Phase 4）

## 参考

- TT-Metal dataflow_api.h 文档
- `tt_metal/programming_examples/` 中的 copy 示例
- TileLang CUDA CodeGen 实现模式
- `memory/bugs.md` - 详细问题记录
