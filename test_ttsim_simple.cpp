// 简单的 TT-Sim 测试程序
// 验证 TT-Sim 仿真器基本功能

#include <iostream>
#include <vector>
#include <cstdint>

// UMD 头文件
#include "umd/device/simulation/simulation_chip.hpp"
#include "umd/device/soc_descriptor.hpp"
#include "umd/device/types/core_coordinates.hpp"

using namespace tt::umd;

int main() {
    std::cout << "=== TT-Sim 简单测试 ===" << std::endl;

    // 检查环境变量
    const char* sim_path = getenv("TT_UMD_SIMULATOR");
    if (!sim_path) {
        std::cerr << "错误: TT_UMD_SIMULATOR 环境变量未设置" << std::endl;
        return 1;
    }
    std::cout << "TT_UMD_SIMULATOR: " << sim_path << std::endl;

    try {
        // 加载 soc 描述文件
        std::string soc_desc_path = SimulationChip::get_soc_descriptor_path_from_simulator_path(sim_path);
        std::cout << "SoC 描述文件: " << soc_desc_path << std::endl;

        SocDescriptor soc_descriptor(soc_desc_path);
        std::cout << "架构: " << arch_to_str(soc_descriptor.arch) << std::endl;
        std::cout << "网格大小: " << soc_descriptor.grid_size.x << "x" << soc_descriptor.grid_size.y << std::endl;

        // 创建仿真设备
        std::cout << "\n创建仿真设备..." << std::endl;
        auto device = SimulationChip::create(sim_path, soc_descriptor, 0, 1);
        std::cout << "仿真设备创建成功" << std::endl;

        // 启动设备
        std::cout << "启动设备..." << std::endl;
        device->start_device();
        std::cout << "设备启动成功" << std::endl;

        // 获取第一个 TENSIX core
        auto tensix_cores = soc_descriptor.get_cores(CoreType::TENSIX);
        if (tensix_cores.empty()) {
            std::cerr << "错误: 没有找到 TENSIX cores" << std::endl;
            return 1;
        }

        CoreCoord core = tensix_cores[0];
        std::cout << "\n使用 TENSIX core: (" << core.x << ", " << core.y << ")" << std::endl;

        // 简单的读写测试
        std::vector<uint32_t> write_data = {0x12345678, 0xABCDEF01, 0x55555555, 0xAAAAAAAA};
        std::vector<uint32_t> read_data(write_data.size(), 0);
        uint64_t addr = 0x100;

        std::cout << "\n写入数据到 L1..." << std::endl;
        device->write_to_device(core, write_data.data(), addr, write_data.size() * sizeof(uint32_t));
        std::cout << "写入成功" << std::endl;

        std::cout << "从 L1 读取数据..." << std::endl;
        device->read_from_device(core, read_data.data(), addr, read_data.size() * sizeof(uint32_t));
        std::cout << "读取成功" << std::endl;

        // 验证数据
        std::cout << "\n验证数据..." << std::endl;
        bool success = true;
        for (size_t i = 0; i < write_data.size(); i++) {
            if (write_data[i] != read_data[i]) {
                std::cout << "  不匹配 [" << i << "]: 期望 0x" << std::hex << write_data[i]
                          << ", 实际 0x" << read_data[i] << std::dec << std::endl;
                success = false;
            }
        }

        if (success) {
            std::cout << "\n✓ 所有数据验证通过！" << std::endl;
        } else {
            std::cout << "\n✗ 数据验证失败" << std::endl;
        }

        // 关闭设备
        std::cout << "\n关闭设备..." << std::endl;
        device->close_device();
        std::cout << "设备关闭成功" << std::endl;

        std::cout << "\n=== TT-Sim 测试完成 ===" << std::endl;
        return success ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "\n错误: " << e.what() << std::endl;
        return 1;
    }
}
