# Phase 1 TT-Sim 测试报告

## 测试概述

**测试目标**: 验证 TileLang CodeGen 生成的 Blackhole 后端代码可以在 TT-Sim 上正确运行。

**测试时间**: 2026-03-16

**测试状态**: ✅ **全部通过**

---

## 测试内容

### 1. CodeGen 代码生成测试

**测试文件**: `tests/target/phase1_ttsim_test.cpp`

**验证项目**:
| 检查项 | 状态 | 说明 |
|--------|------|------|
| dataflow_api.h header | ✅ | 内核头文件引用 |
| kernel_main 函数 | ✅ | 内核入口函数 |
| InterleavedAddrGen | ✅ | TT-Sim 兼容的地址生成 |
| get_noc_addr | ✅ | NOC 地址转换 |
| noc_async_read | ✅ | NOC 异步读取 |
| noc_async_write | ✅ | NOC 异步写入 |
| noc_async_read_barrier | ✅ | 读取同步 |
| noc_async_write_barrier | ✅ | 写入同步 |

**生成代码位置**: `tests/target/phase1_generated_kernel.cpp`

---

### 2. TT-Sim 端到端测试

**测试文件**: `tt_metal/programming_examples/tilelang_copy_test/phase1_ttsim_host.cpp`

**Kernel 文件**: `tt_metal/programming_examples/tilelang_copy_test/kernels/phase1_copy_kernel.cpp`

**测试配置**:
- Tile 大小: 32x32 FP16 (2048 bytes)
- Tile 数量: 4
- 总数据量: 8192 bytes (8 KB)
- 数据模式: 0, 1, 2, ..., 4095 (FP16)

**运行命令**:
```bash
cd $TT_METAL_HOME
source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh
./build_Release/programming_examples/phase1_tilelang_ttsim
```

---

## 测试结果

```
=== Phase 1: TileLang CodeGen Blackhole TT-Sim Test ===
Testing CodeGen-generated kernel on TT-Sim
Tile size: 2048 bytes (32x32 FP16)
Num tiles: 4
Total data: 8192 bytes (8 KB)
...
Executing CodeGen-generated copy kernel...
Kernel execution complete
Verifying results...

✓ SUCCESS: Phase 1 TT-Sim Test Passed!
  CodeGen-generated kernel executed successfully on TT-Sim
  All 4096 elements copied correctly
```

**性能数据**:
- 仿真时间: 0.3 秒
- 仿真频率: 40.0 KHz

---

## 关键实现细节

### 1. TT-Sim 兼容的 Kernel 代码

```cpp
void kernel_main() {
    // Runtime arguments
    uint32_t src_dram_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_dram_addr = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
    uint32_t l1_buffer_addr = get_arg_val<uint32_t>(3);

    constexpr uint32_t TILE_SIZE = 2048;

    // TT-Sim 要求使用 InterleavedAddrGen 进行地址转换
    InterleavedAddrGen<true> src_gen = {
        .bank_base_address = src_dram_addr,
        .page_size = TILE_SIZE
    };
    InterleavedAddrGen<true> dst_gen = {
        .bank_base_address = dst_dram_addr,
        .page_size = TILE_SIZE
    };

    // Process each tile
    for (uint32_t i = 0; i < num_tiles; i++) {
        // Read tile from DRAM to L1
        uint64_t src_noc_addr = get_noc_addr(i, src_gen);
        noc_async_read(src_noc_addr, l1_buffer_addr, TILE_SIZE);
        noc_async_read_barrier();

        // Write tile from L1 to DRAM
        uint64_t dst_noc_addr = get_noc_addr(i, dst_gen);
        noc_async_write(l1_buffer_addr, dst_noc_addr, TILE_SIZE);
        noc_async_write_barrier();
    }
}
```

### 2. CodeGen 更新

`codegen_blackhole.cc` 中的 `GenerateSimpleCopyKernel` 方法已更新为生成 TT-Sim 兼容代码：
- 使用 `InterleavedAddrGen<true>` 替代直接地址访问
- 使用 `get_noc_addr()` 进行地址转换
- 单 BRISC kernel 执行（TT-Sim 不支持 NCRISC NOC write）

---

## 文件清单

| 文件 | 说明 |
|------|------|
| `tests/target/phase1_ttsim_test.cpp` | CodeGen 测试程序 |
| `tests/target/phase1_generated_kernel.cpp` | 生成的内核代码 |
| `tt_metal/programming_examples/tilelang_copy_test/kernels/phase1_copy_kernel.cpp` | TT-Sim 内核 |
| `tt_metal/programming_examples/tilelang_copy_test/phase1_ttsim_host.cpp` | Host 测试代码 |
| `tt_metal/programming_examples/tilelang_copy_test/CMakeLists.txt` | 构建配置 |

---

## 结论

Phase 1 目标已达成：

1. ✅ **CodeGen 框架**: 可以生成 TT-Metal 风格的内核代码
2. ✅ **TT-Sim 兼容性**: 生成的代码可以在 TT-Sim 上正确运行
3. ✅ **端到端验证**: 完整的 Host + Kernel 测试通过

### 下一步工作

1. **Phase 2**: 实现多核拆分 (SplitBlackholeKernel)
2. **Phase 2**: CB 分配 (PlanBlackholeCB)
3. **Phase 2**: 140 核分配 (AssignBlackholeCores)
4. **Phase 3**: GEMM 支持

---

## 附录: 快速测试命令

```bash
# 1. 编译并运行 CodeGen 测试
g++ -std=c++17 tests/target/phase1_ttsim_test.cpp -o tests/target/phase1_ttsim_test
./tests/target/phase1_ttsim_test

# 2. 编译 TT-Sim 测试
cd tt_metal_repo/build_Release
cmake --build . --target phase1_tilelang_ttsim -j4

# 3. 运行 TT-Sim 测试
cd $TT_METAL_HOME
source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh
./build_Release/programming_examples/phase1_tilelang_ttsim
```
