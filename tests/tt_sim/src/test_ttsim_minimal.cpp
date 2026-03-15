// 最小化的 TT-Sim 测试程序
// 直接测试 TT-Sim 仿真器的核心功能

#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>

// TT-Sim 函数类型定义
typedef void* (*libttsim_init_t)(void);
typedef void (*libttsim_exit_t)(void*);
typedef void (*libttsim_clock_t)(void*, uint64_t);
typedef uint32_t (*libttsim_pci_config_rd32_t)(void*, int, int);
typedef void (*libttsim_pci_config_wr32_t)(void*, int, int, uint32_t);
typedef void (*libttsim_pci_mem_rd_bytes_t)(void*, uint64_t, void*, uint32_t);
typedef void (*libttsim_pci_mem_wr_bytes_t)(void*, uint64_t, const void*, uint32_t);

int main() {
    std::cout << "=== TT-Sim 最小化测试 ===" << std::endl;

    // 加载 TT-Sim 库
    void* handle = dlopen("/root/dev/vibe_dsl/tt_metal_repo/sim/libttsim.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "无法加载 libttsim.so: " << dlerror() << std::endl;
        return 1;
    }
    std::cout << "✓ 成功加载 libttsim.so" << std::endl;

    // 获取函数指针
    auto libttsim_init = (libttsim_init_t)dlsym(handle, "libttsim_init");
    auto libttsim_exit = (libttsim_exit_t)dlsym(handle, "libttsim_exit");
    auto libttsim_clock = (libttsim_clock_t)dlsym(handle, "libttsim_clock");
    auto libttsim_pci_config_rd32 = (libttsim_pci_config_rd32_t)dlsym(handle, "libttsim_pci_config_rd32");
    auto libttsim_pci_mem_rd_bytes = (libttsim_pci_mem_rd_bytes_t)dlsym(handle, "libttsim_pci_mem_rd_bytes");
    auto libttsim_pci_mem_wr_bytes = (libttsim_pci_mem_wr_bytes_t)dlsym(handle, "libttsim_pci_mem_wr_bytes");

    if (!libttsim_init || !libttsim_exit || !libttsim_pci_config_rd32) {
        std::cerr << "无法获取函数指针: " << dlerror() << std::endl;
        dlclose(handle);
        return 1;
    }
    std::cout << "✓ 成功获取函数指针" << std::endl;

    // 初始化 TT-Sim
    std::cout << "\n初始化 TT-Sim..." << std::endl;
    void* sim = libttsim_init();
    if (!sim) {
        std::cerr << "TT-Sim 初始化失败" << std::endl;
        dlclose(handle);
        return 1;
    }
    std::cout << "✓ TT-Sim 初始化成功" << std::endl;

    // 读取 PCI 配置
    std::cout << "\n读取 PCI 配置..." << std::endl;
    uint32_t pci_id = libttsim_pci_config_rd32(sim, 0, 0);
    uint16_t vendor_id = pci_id & 0xFFFF;
    uint16_t device_id = pci_id >> 16;
    std::cout << "  Vendor ID: 0x" << std::hex << vendor_id << std::dec << std::endl;
    std::cout << "  Device ID: 0x" << std::hex << device_id << std::dec << std::endl;

    if (vendor_id == 0x1e52) {
        std::cout << "✓ Tenstorrent 设备识别成功" << std::endl;
    } else {
        std::cout << "✗ 未知的 vendor ID" << std::endl;
    }

    if (device_id == 0xb140) {
        std::cout << "✓ Blackhole 设备识别成功" << std::endl;
    } else if (device_id == 0x401e) {
        std::cout << "✓ Wormhole 设备识别成功" << std::endl;
    }

    // 清理
    std::cout << "\n关闭 TT-Sim..." << std::endl;
    libttsim_exit(sim);
    dlclose(handle);

    std::cout << "\n=== TT-Sim 最小化测试完成 ===" << std::endl;
    return 0;
}
