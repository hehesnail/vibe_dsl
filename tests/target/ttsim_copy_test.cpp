// SPDX-FileCopyrightText: © 2025 TileLang Blackhole Test
//
// SPDX-License-Identifier: Apache-2.0

// TT-Sim Copy Kernel Test
// Based on TT-Metal add_2_integers_in_riscv example

#include <cstdint>
#include <memory>
#include <vector>
#include <iostream>
#include <cstring>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>

using namespace tt;
using namespace tt::tt_metal;

// Tile configuration
constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t TILE_COLS = 32;
constexpr uint32_t ELEMENT_SIZE = 2;  // FP16
constexpr uint32_t TILE_SIZE = TILE_ROWS * TILE_COLS * ELEMENT_SIZE;  // 2048 bytes
constexpr uint32_t NUM_TILES = 4;
constexpr uint32_t TOTAL_SIZE = TILE_SIZE * NUM_TILES;  // 8192 bytes

int main() {
    std::cout << "=== TileLang Blackhole Copy Kernel TT-Sim Test ===" << std::endl;
    std::cout << "Tile size: " << TILE_SIZE << " bytes (" << TILE_ROWS << "x" << TILE_COLS << " FP16)" << std::endl;
    std::cout << "Num tiles: " << NUM_TILES << std::endl;
    std::cout << "Total data: " << TOTAL_SIZE << " bytes (" << TOTAL_SIZE/1024 << " KB)" << std::endl;

    // Create device
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

    // Create program
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};  // Single core

    // Buffer configurations
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = TILE_SIZE,
        .buffer_type = BufferType::DRAM};
    distributed::DeviceLocalBufferConfig l1_config{
        .page_size = TILE_SIZE,
        .buffer_type = BufferType::L1};
    distributed::ReplicatedBufferConfig src_buffer_config{
        .size = TOTAL_SIZE,
    };
    distributed::ReplicatedBufferConfig dst_buffer_config{
        .size = TOTAL_SIZE,
    };

    // Create DRAM buffers
    auto src_dram_buffer = distributed::MeshBuffer::create(src_buffer_config, dram_config, mesh_device.get());
    auto dst_dram_buffer = distributed::MeshBuffer::create(dst_buffer_config, dram_config, mesh_device.get());

    // Create L1 buffers (for CB simulation)
    auto l1_buffer_config = distributed::ReplicatedBufferConfig{.size = TILE_SIZE * 2};  // Double buffering
    auto cb_buffer = distributed::MeshBuffer::create(l1_buffer_config, l1_config, mesh_device.get());

    // Create source data (FP16 pattern)
    std::vector<uint16_t> src_data(TOTAL_SIZE / sizeof(uint16_t));
    for (size_t i = 0; i < src_data.size(); i++) {
        src_data[i] = static_cast<uint16_t>(i % 65536);  // Pattern: 0, 1, 2, 3, ...
    }

    std::cout << "Source data pattern: 0, 1, 2, ... , " << src_data.back() << std::endl;

    // Write source data to DRAM
    std::cout << "Writing source data to DRAM..." << std::endl;
    EnqueueWriteMeshBuffer(cq, src_dram_buffer, src_data, /*blocking=*/true);

    // Create reader kernel
    KernelHandle reader_kernel = CreateKernel(
        program,
        "tests/target/kernels/copy_reader_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                          .noc = NOC::RISCV_0_default});

    // Create writer kernel
    KernelHandle writer_kernel = CreateKernel(
        program,
        "tests/target/kernels/copy_writer_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                          .noc = NOC::RISCV_1_default});

    // Set runtime args for reader
    // Args: src_dram_addr_lo, src_dram_addr_hi, cb_addr_lo, cb_addr_hi, num_tiles
    SetRuntimeArgs(
        program,
        reader_kernel,
        core,
        {static_cast<uint32_t>(src_dram_buffer->device_address()),
         static_cast<uint32_t>(src_dram_buffer->device_address() >> 32),
         static_cast<uint32_t>(cb_buffer->device_address()),
         static_cast<uint32_t>(cb_buffer->device_address() >> 32),
         NUM_TILES});

    // Set runtime args for writer
    // Args: dst_dram_addr_lo, dst_dram_addr_hi, cb_addr_lo, cb_addr_hi, num_tiles
    SetRuntimeArgs(
        program,
        writer_kernel,
        core,
        {static_cast<uint32_t>(dst_dram_buffer->device_address()),
         static_cast<uint32_t>(dst_dram_buffer->device_address() >> 32),
         static_cast<uint32_t>(cb_buffer->device_address()),
         static_cast<uint32_t>(cb_buffer->device_address() >> 32),
         NUM_TILES});

    // Execute program
    std::cout << "Executing copy kernel..." << std::endl;
    EnqueueProgram(cq, program, /*blocking=*/false);
    Finish(cq);
    std::cout << "Kernel execution complete" << std::endl;

    // Read back results
    std::vector<uint16_t> dst_data(TOTAL_SIZE / sizeof(uint16_t));
    EnqueueReadMeshBuffer(cq, dst_data, dst_dram_buffer, /*blocking=*/true);

    // Verify results
    std::cout << "Verifying results..." << std::endl;
    bool success = true;
    int mismatch_count = 0;
    for (size_t i = 0; i < src_data.size() && mismatch_count < 10; i++) {
        if (src_data[i] != dst_data[i]) {
            std::cout << "Mismatch at index " << i << ": expected " << src_data[i]
                      << ", got " << dst_data[i] << std::endl;
            success = false;
            mismatch_count++;
        }
    }

    if (success) {
        std::cout << "\n✓ SUCCESS: Copy kernel test passed!" << std::endl;
        std::cout << "  All " << src_data.size() << " elements copied correctly" << std::endl;
    } else {
        std::cout << "\n✗ FAILED: Copy kernel test failed!" << std::endl;
    }

    return success ? 0 : 1;
}
