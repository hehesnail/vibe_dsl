// 完整的 TT-Sim 测试程序
// 测试 Blackhole 14x10 网格的所有核心

#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <chrono>

// TT-Sim 函数类型定义
typedef void (*libttsim_init_t)(void);
typedef void (*libttsim_exit_t)(void);
typedef void (*libttsim_clock_t)(uint32_t);
typedef uint32_t (*libttsim_pci_config_rd32_t)(uint32_t, uint32_t);
typedef void (*libttsim_pci_mem_rd_bytes_t)(uint64_t, void*, uint32_t);
typedef void (*libttsim_pci_mem_wr_bytes_t)(uint64_t, const void*, uint32_t);
typedef void (*libttsim_tile_rd_bytes_t)(uint32_t, uint32_t, uint64_t, void*, uint32_t);
typedef void (*libttsim_tile_wr_bytes_t)(uint32_t, uint32_t, uint64_t, const void*, uint32_t);

int main() {
    std::cout << "=== TT-Sim 完整功能测试 (Blackhole 14x10) ===" << std::endl;

    // 加载 TT-Sim 库
    const char* sim_path = "/work/ttsim/libttsim.so";
    void* handle = dlopen(sim_path, RTLD_LAZY);
    if (!handle) {
        std::cerr << "无法加载 libttsim.so: " << dlerror() << std::endl;
        return 1;
    }
    std::cout << "✓ 成功加载 libttsim.so" << std::endl;

    // 获取函数指针
    auto libttsim_init = (libttsim_init_t)dlsym(handle, "libttsim_init");
    auto libttsim_exit = (libttsim_exit_t)dlsym(handle, "libttsim_exit");
    auto libttsim_tile_rd_bytes = (libttsim_tile_rd_bytes_t)dlsym(handle, "libttsim_tile_rd_bytes");
    auto libttsim_tile_wr_bytes = (libttsim_tile_wr_bytes_t)dlsym(handle, "libttsim_tile_wr_bytes");

    if (!libttsim_init || !libttsim_exit || !libttsim_tile_rd_bytes || !libttsim_tile_wr_bytes) {
        std::cerr << "无法获取函数指针: " << dlerror() << std::endl;
        dlclose(handle);
        return 1;
    }

    // 初始化 TT-Sim
    libttsim_init();
    std::cout << "✓ TT-Sim 初始化成功\n" << std::endl;

    // Blackhole 核心网格: 14x10
    const int GRID_X = 14;
    const int GRID_Y = 10;
    const uint32_t TEST_ADDR = 0x10000;

    std::cout << "测试 " << GRID_X << "x" << GRID_Y << " 核心网格的 L1 内存读写..." << std::endl;

    int passed = 0;
    int failed = 0;
    auto start = std::chrono::high_resolution_clock::now();

    for (int y = 0; y < GRID_Y; y++) {
        for (int x = 0; x < GRID_X; x++) {
            // 生成测试数据 (基于坐标)
            uint32_t test_data = (x << 16) | (y << 8) | 0xAB;
            uint32_t read_data = 0;

            // 写入
            libttsim_tile_wr_bytes(x, y, TEST_ADDR, &test_data, sizeof(test_data));

            // 读取
            libttsim_tile_rd_bytes(x, y, TEST_ADDR, &read_data, sizeof(read_data));

            // 验证
            if (read_data == test_data) {
                passed++;
            } else {
                failed++;
                std::cerr << "✗ Core (" << x << "," << y << ") 验证失败: "
                          << "期望 0x" << std::hex << test_data
                          << ", 实际 0x" << read_data << std::dec << std::endl;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "\n测试结果:" << std::endl;
    std::cout << "  通过: " << passed << "/" << (GRID_X * GRID_Y) << std::endl;
    std::cout << "  失败: " << failed << "/" << (GRID_X * GRID_Y) << std::endl;
    std::cout << "  耗时: " << duration.count() << " ms" << std::endl;

    // 清理
    libttsim_exit();
    dlclose(handle);

    std::cout << "\n=== TT-Sim 完整测试完成 ===" << std::endl;
    return failed > 0 ? 1 : 0;
}
