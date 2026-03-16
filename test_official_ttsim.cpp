// 官方 TT-Sim 使用方式测试
// 基于 TT-Metal 编程示例模式

#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>

// 模拟官方示例的最小化测试
// 直接调用 TT-Sim 底层 API

#include <dlfcn.h>

// TT-Sim 函数类型
typedef void (*libttsim_init_t)(void);
typedef void (*libttsim_exit_t)(void);
typedef void (*libttsim_tile_rd_bytes_t)(uint32_t, uint32_t, uint64_t, void*, uint32_t);
typedef void (*libttsim_tile_wr_bytes_t)(uint32_t, uint32_t, uint64_t, const void*, uint32_t);

int main() {
    std::cout << "=== 官方 TT-Sim 配置测试 ===" << std::endl;

    // 官方配置方式:
    // export TT_METAL_SIMULATOR=~/sim/libttsim_bh.so
    // cp $TT_METAL_HOME/tt_metal/soc_descriptors/blackhole_140_arch.yaml ~/sim/soc_descriptor.yaml

    const char* sim_path = getenv("TT_METAL_SIMULATOR");
    if (!sim_path) {
        sim_path = "/work/ttsim/libttsim.so";
        std::cout << "使用默认路径: " << sim_path << std::endl;
    } else {
        std::cout << "TT_METAL_SIMULATOR=" << sim_path << std::endl;
    }

    // 加载 TT-Sim
    void* handle = dlopen(sim_path, RTLD_LAZY);
    if (!handle) {
        std::cerr << "无法加载: " << dlerror() << std::endl;
        return 1;
    }
    std::cout << "✓ 加载成功" << std::endl;

    // 获取函数
    auto init = (libttsim_init_t)dlsym(handle, "libttsim_init");
    auto exit = (libttsim_exit_t)dlsym(handle, "libttsim_exit");
    auto tile_wr = (libttsim_tile_wr_bytes_t)dlsym(handle, "libttsim_tile_wr_bytes");
    auto tile_rd = (libttsim_tile_rd_bytes_t)dlsym(handle, "libttsim_tile_rd_bytes");

    if (!init || !exit || !tile_wr || !tile_rd) {
        std::cerr << "函数查找失败" << std::endl;
        return 1;
    }

    // 初始化
    init();
    std::cout << "✓ TT-Sim 初始化成功" << std::endl;

    // 测试: 在 (0,1) 核心读写（官方测试使用的坐标）
    // 注意: 经过 eth core 修改后，(0,1) 不是有效的 TENSIX core
    // 需要使用有效的 worker core 坐标

    // Blackhole 有效 worker core: x∈[1-7,10-16], y∈[2-11]
    // 使用 (1,2) 作为测试坐标
    int core_x = 1, core_y = 2;
    uint32_t addr = 0x100;
    std::vector<uint32_t> wdata = {1, 2, 3, 4, 5};
    std::vector<uint32_t> rdata(wdata.size(), 0);

    std::cout << "\n测试核心 (" << core_x << "," << core_y << ") L1 内存读写..." << std::endl;

    tile_wr(core_x, core_y, addr, wdata.data(), wdata.size() * sizeof(uint32_t));
    tile_rd(core_x, core_y, addr, rdata.data(), rdata.size() * sizeof(uint32_t));

    if (wdata == rdata) {
        std::cout << "✓ 数据验证成功" << std::endl;
    } else {
        std::cerr << "✗ 数据验证失败" << std::endl;
        exit();
        dlclose(handle);
        return 1;
    }

    // 关闭
    exit();
    dlclose(handle);

    std::cout << "\n=== 官方 TT-Sim 配置测试通过 ===" << std::endl;
    return 0;
}
