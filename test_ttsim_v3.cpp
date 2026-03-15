// 最小化的 TT-Sim 测试程序 v3
// 直接测试 TT-Sim 仿真器的核心功能 - 使用正确的函数签名

#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>

// TT-Sim 函数类型定义 - 基于 tt_sim_communicator.hpp
// 注意：这些函数不使用 sim 实例指针，它们是全局的
typedef void (*libttsim_init_t)(void);
typedef void (*libttsim_exit_t)(void);
typedef void (*libttsim_clock_t)(uint32_t);
typedef uint32_t (*libttsim_pci_config_rd32_t)(uint32_t, uint32_t);
typedef void (*libttsim_pci_mem_rd_bytes_t)(uint64_t, void*, uint32_t);
typedef void (*libttsim_pci_mem_wr_bytes_t)(uint64_t, const void*, uint32_t);
typedef void (*libttsim_tile_rd_bytes_t)(uint32_t, uint32_t, uint64_t, void*, uint32_t);
typedef void (*libttsim_tile_wr_bytes_t)(uint32_t, uint32_t, uint64_t, const void*, uint32_t);

int main() {
    std::cout << "=== TT-Sim 最小化测试 v3 ===" << std::endl;

    // 加载 TT-Sim 库
    const char* sim_path = "/work/ttsim/libttsim.so";
    void* handle = dlopen(sim_path, RTLD_LAZY);
    if (!handle) {
        std::cerr << "无法加载 libttsim.so: " << dlerror() << std::endl;
        return 1;
    }
    std::cout << "✓ 成功加载 libttsim.so from " << sim_path << std::endl;

    // 获取函数指针
    auto libttsim_init = (libttsim_init_t)dlsym(handle, "libttsim_init");
    auto libttsim_exit = (libttsim_exit_t)dlsym(handle, "libttsim_exit");
    auto libttsim_clock = (libttsim_clock_t)dlsym(handle, "libttsim_clock");
    auto libttsim_pci_config_rd32 = (libttsim_pci_config_rd32_t)dlsym(handle, "libttsim_pci_config_rd32");
    auto libttsim_pci_mem_rd_bytes = (libttsim_pci_mem_rd_bytes_t)dlsym(handle, "libttsim_pci_mem_rd_bytes");
    auto libttsim_pci_mem_wr_bytes = (libttsim_pci_mem_wr_bytes_t)dlsym(handle, "libttsim_pci_mem_wr_bytes");
    auto libttsim_tile_rd_bytes = (libttsim_tile_rd_bytes_t)dlsym(handle, "libttsim_tile_rd_bytes");
    auto libttsim_tile_wr_bytes = (libttsim_tile_wr_bytes_t)dlsym(handle, "libttsim_tile_wr_bytes");

    if (!libttsim_init || !libttsim_exit || !libttsim_pci_config_rd32) {
        std::cerr << "无法获取函数指针: " << dlerror() << std::endl;
        dlclose(handle);
        return 1;
    }
    std::cout << "✓ 成功获取函数指针" << std::endl;

    // 初始化 TT-Sim
    std::cout << "\n初始化 TT-Sim..." << std::endl;
    libttsim_init();
    std::cout << "✓ TT-Sim 初始化成功" << std::endl;

    // 读取 PCI 配置
    std::cout << "\n读取 PCI 配置..." << std::endl;
    uint32_t pci_id = libttsim_pci_config_rd32(0, 0);
    uint16_t vendor_id = pci_id & 0xFFFF;
    uint16_t device_id = pci_id >> 16;
    std::cout << "  PCI ID: 0x" << std::hex << pci_id << std::dec << std::endl;
    std::cout << "  Vendor ID: 0x" << std::hex << vendor_id << std::dec << std::endl;
    std::cout << "  Device ID: 0x" << std::hex << device_id << std::dec << std::endl;

    if (vendor_id == 0x1e52) {
        std::cout << "✓ Tenstorrent 设备识别成功" << std::endl;
    } else {
        std::cout << "✗ 未知的 vendor ID: 0x" << std::hex << vendor_id << std::dec << std::endl;
    }

    if (device_id == 0xb140) {
        std::cout << "✓ Blackhole 设备识别成功" << std::endl;
    } else if (device_id == 0x401e) {
        std::cout << "✓ Wormhole 设备识别成功" << std::endl;
    } else {
        std::cout << "✗ 未知的 device ID: 0x" << std::hex << device_id << std::dec << std::endl;
    }

    // 测试 tile 读写（core 0,0）
    std::cout << "\n测试 Tile 读写..." << std::endl;
    uint32_t test_addr = 0x10000;  // L1 地址
    uint32_t write_data = 0xDEADBEEF;
    uint32_t read_data = 0;

    libttsim_tile_wr_bytes(0, 0, test_addr, &write_data, sizeof(write_data));
    std::cout << "✓ 写入数据 0x" << std::hex << write_data << std::dec << " 到 core(0,0) L1[0x" << std::hex << test_addr << std::dec << "]" << std::endl;

    libttsim_tile_rd_bytes(0, 0, test_addr, &read_data, sizeof(read_data));
    std::cout << "✓ 读取数据 0x" << std::hex << read_data << std::dec << " 从 core(0,0) L1[0x" << std::hex << test_addr << std::dec << "]" << std::endl;

    if (read_data == write_data) {
        std::cout << "✓ 数据验证成功" << std::endl;
    } else {
        std::cout << "✗ 数据验证失败: 期望 0x" << std::hex << write_data << ", 实际 0x" << read_data << std::dec << std::endl;
    }

    // 清理
    std::cout << "\n关闭 TT-Sim..." << std::endl;
    libttsim_exit();
    dlclose(handle);

    std::cout << "\n=== TT-Sim 测试完成 ===" << std::endl;
    return 0;
}
