# Blackhole Backend Implementation Status

## 已完成的工作

### 1. BlackholeModule 核心类 ✅

**文件**: `tilelang_repo/src/target/blackhole_module.h`, `blackhole_module.cc`

实现内容:
- `BlackholeModuleNode`: 继承 `ffi::ModuleObj`，实现 TVM 模块接口
- `BlackholeWrappedFunc`: 执行封装器，处理 TVM packed args -> TT-Metal 执行
- 延迟初始化 MeshDevice
- Kernel/Program 缓存机制
- Buffer 管理和数据传输 (DLTensor -> MeshBuffer)
- Reader -> Compute -> Writer 串行执行流程

### 2. Build 函数更新 ✅

**文件**: `tilelang_repo/src/target/rt_mod_blackhole.cc`

实现内容:
- `ExtractBlackholeFuncInfo`: 从 TIR 提取函数信息（参数类型、CB配置等）
- `ExtractCBConfig`: 提取 CB 配置
- `BuildTileLangBlackhole`: 使用 BlackholeModuleCreate 创建模块
- 注册 `target.build.tilelang_blackhole`

### 3. CMakeLists.txt 更新 ✅

**文件**: `tilelang_repo/CMakeLists.txt`

更新内容:
- 添加 `src/target/blackhole_module.cc` 到 TILE_LANG_BLACKHOLE_SRCS

## 关键实现细节

### 参数传递流程

```
Python 调用
    ↓
TVM Packed Args (DLTensor* for buffers, uint32_t for scalars)
    ↓
BlackholeWrappedFunc::operator()
    ├── 解包参数
    ├── 创建 MeshBuffer (输入 buffer 写入数据)
    ├── 设置 RuntimeArgs (buffer addresses)
    ├── 执行 Reader kernel (if exists)
    ├── 执行 Compute kernel (if exists)
    ├── 执行 Writer kernel
    ├── 同步 (Finish)
    └── 读取输出数据到 DLTensor
    ↓
返回结果
```

### CB 配置

CB 配置从 TIR PrimFunc 的 attrs 中读取:
- Key: `tl.blackhole_cb_config`
- Value: Map<cb_id, {num_pages, page_size, data_format}>

### Kernel 拆分支持

支持三种模式:
1. 单一 kernel (has_writer=true only)
2. Reader-Writer 模式 (has_reader, has_writer)
3. Reader-Compute-Writer 模式 (has_reader, has_compute, has_writer)

通过 `tl.blackhole_kernel_split` attr 配置。

## 待完成的工作

### 1. 代码编译验证 ⏳

需要重新编译 TileLang 验证代码是否正确:
```bash
cd tilelang_repo
mkdir -p build && cd build
cmake .. -DUSE_BLACKHOLE=ON
make -j
```

### 2. 可能的编译错误修复

潜在问题:
- TT-Metal 头文件路径需要正确配置 (`TT_METAL_HOME`)
- C++17 filesystem 支持 (GCC 版本要求)
- TT-Metal 库链接

### 3. CodeGen 调整

当前的 `codegen_blackhole.cc` 生成的是标准 C 代码格式，需要调整为生成 TT-Metal kernel 格式:

当前生成 (错误):
```cpp
void kernel(half* A, half* B) { ... }
```

需要生成 (正确):
```cpp
void kernel_main() {
    uint32_t A_addr = get_arg_val<uint32_t>(0);
    uint32_t B_addr = get_arg_val<uint32_t>(1);
    // ... kernel logic
}
```

### 4. 端到端测试

创建测试脚本 `tests/target/test_blackhole_e2e.py` 已准备好，待编译通过后运行。

## 下一步行动建议

1. **配置编译环境**:
   ```bash
   export TT_METAL_HOME=/root/dev/vibe_dsl/tt_metal_repo
   export LD_LIBRARY_PATH=$TT_METAL_HOME/build_Release/lib:$LD_LIBRARY_PATH
   ```

2. **重新编译 TileLang**:
   ```bash
   cd tilelang_repo/build
   cmake .. -DUSE_BLACKHOLE=ON -DCMAKE_BUILD_TYPE=Release
   make -j4 2>&1 | tee build.log
   ```

3. **修复编译错误**（如果出现）:
   - 检查头文件路径
   - 检查 TT-Metal 库链接
   - 修复 C++ 语法错误

4. **运行测试**:
   ```bash
   python tests/target/test_blackhole_e2e.py
   ```

## 关键设计决策确认

| 决策项 | 选择 | 说明 |
|--------|------|------|
| Inplace 操作 | 不支持 | Phase 1 先不考虑 |
| 执行模式 | 单核 (0,0) | 单核版本先跑通 |
| Kernel 缓存 | Module 级别 | 避免重复 JIT 编译 |
| 多 Kernel | 串行执行 | R->C->W 顺序执行 |
| Device 初始化 | 延迟初始化 | 首次调用时初始化 |
| Buffer 管理 | 每次调用创建 | 输入输出都创建 MeshBuffer |

## 参考实现

- CUDA Module: `tilelang_repo/3rdparty/tvm/src/runtime/cuda/cuda_module.cc`
- Phase 1 TT-Sim Test: `tt_metal_repo/tt_metal/programming_examples/tilelang_copy_test/`
