// TT-Sim Worker 核心测试程序
// 测试 Blackhole 的有效 worker 核心

#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <chrono>

// TT-Sim 函数类型定义
typedef void (*libttsim_init_t)(void);
typedef void (*libttsim_exit_t)(void);
typedef void (*libttsim_tile_rd_bytes_t)(uint32_t, uint32_t, uint64_t, void*, uint32_t);
typedef void (*libttsim_tile_wr_bytes_t)(uint32_t, uint32_t, uint64_t, const void*, uint32_t);

// Blackhole 有效 worker 核心列表 (从 soc_descriptor.yaml 提取)
// 格式: x-y
const std::vector<std::pair<int, int>> WORKER_CORES = {
    // Row 2
    {1,2}, {2,2}, {3,2}, {4,2}, {5,2}, {6,2}, {7,2}, {10,2}, {11,2}, {12,2}, {13,2}, {14,2}, {15,2}, {16,2},
    // Row 3
    {1,3}, {2,3}, {3,3}, {4,3}, {5,3}, {6,3}, {7,3}, {10,3}, {11,3}, {12,3}, {13,3}, {14,3}, {15,3}, {16,3},
    // Row 4
    {1,4}, {2,4}, {3,4}, {4,4}, {5,4}, {6,4}, {7,4}, {10,4}, {11,4}, {12,4}, {13,4}, {14,4}, {15,4}, {16,4},
    // Row 5
    {1,5}, {2,5}, {3,5}, {4,5}, {5,5}, {6,5}, {7,5}, {10,5}, {11,5}, {12,5}, {13,5}, {14,5}, {15,5}, {16,5},
    // Row 6
    {1,6}, {2,6}, {3,6}, {4,6}, {5,6}, {6,6}, {7,6}, {10,6}, {11,6}, {12,6}, {13,6}, {14,6}, {15,6}, {16,6},
    // Row 7
    {1,7}, {2,7}, {3,7}, {4,7}, {5,7}, {6,7}, {7,7}, {10,7}, {11,7}, {12,7}, {13,7}, {14,7}, {15,7}, {16,7},
    // Row 8
    {1,8}, {2,8}, {3,8}, {4,8}, {5,8}, {6,8}, {7,8}, {10,8}, {11,8}, {12,8}, {13,8}, {14,8}, {15,8}, {16,8},
    // Row 9
    {1,9}, {2,9}, {3,9}, {4,9}, {5,9}, {6,9}, {7,9}, {10,9}, {11,9}, {12,9}, {13,9}, {14,9}, {15,9}, {16,9},
    // Row 10
    {1,10}, {2,10}, {3,10}, {4,10}, {5,10}, {6,10}, {7,10}, {10,10}, {11,10}, {12,10}, {13,10}, {14,10}, {15,10}, {16,10},
    // Row 11
    {1,11}, {2,11}, {3,11}, {4,11}, {5,11}, {6,11}, {7,11}, {10,11}, {11,11}, {12,11}, {13,11}, {14,11}, {15,11}, {16,11},
};

int main() {
    std::cout << "=== TT-Sim Worker 核心测试 (Blackhole) ===" << std::endl;
    std::cout << "Worker 核心数量: " << WORKER_CORES.size() << std::endl;

    // 加载 TT-Sim 库
    const char* sim_path = "/work/ttsim/libttsim.so";
    void* handle = dlopen(sim_path, RTLD_LAZY);
    if (!handle) {
        std::cerr << "无法加载 libttsim.so: " << dlerror() << std::endl;
        return 1;
    }
    std::cout << "✓ 成功加载 libttsim.so\n" << std::endl;

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

    const uint32_t TEST_ADDR = 0x10000;
    const uint32_t PATTERN = 0xA5A5A5A5;

    std::cout << "测试所有 Worker 核心的 L1 内存读写..." << std::endl;

    int passed = 0;
    int failed = 0;
    auto start = std::chrono::high_resolution_clock::now();

    for (const auto& core : WORKER_CORES) {
        int x = core.first;
        int y = core.second;

        uint32_t test_data = PATTERN ^ (x << 8) ^ (y << 16);
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
            std::cerr << "✗ Core (" << x << "," << y << ") 验证失败" << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "\n测试结果:" << std::endl;
    std::cout << "  通过: " << passed << "/" << WORKER_CORES.size() << std::endl;
    std::cout << "  失败: " << failed << "/" << WORKER_CORES.size() << std::endl;
    std::cout << "  耗时: " << duration.count() << " ms" << std::endl;
    std::cout << "  平均: " << (duration.count() * 1000.0 / WORKER_CORES.size()) << " µs/核心" << std::endl;

    // 清理
    libttsim_exit();
    dlclose(handle);

    std::cout << "\n=== TT-Sim Worker 核心测试完成 ===" << std::endl;
    return failed > 0 ? 1 : 0;
}
