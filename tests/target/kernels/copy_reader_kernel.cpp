// SPDX-FileCopyrightText: © 2025 TileLang Blackhole Test
//
// SPDX-License-Identifier: Apache-2.0

// Copy Reader Kernel - BRISC
// Reads data from DRAM to L1 (CB)

#include <cstdint>

void kernel_main() {
    // Runtime arguments
    uint32_t src_dram_lo = get_arg_val<uint32_t>(0);
    uint32_t src_dram_hi = get_arg_val<uint32_t>(1);
    uint32_t cb_addr_lo = get_arg_val<uint32_t>(2);
    uint32_t cb_addr_hi = get_arg_val<uint32_t>(3);
    uint32_t num_tiles = get_arg_val<uint32_t>(4);

    // Construct 64-bit addresses
    uint64_t src_dram_addr = (static_cast<uint64_t>(src_dram_hi) << 32) | src_dram_lo;
    uint64_t cb_addr = (static_cast<uint64_t>(cb_addr_hi) << 32) | cb_addr_lo;

    // Tile configuration
    constexpr uint32_t TILE_SIZE = 2048;  // 32x32 FP16

    // Create address generator for DRAM
    InterleavedAddrGen<true> src_dram = {
        .bank_base_address = src_dram_addr,
        .page_size = TILE_SIZE
    };

    // Read each tile from DRAM to L1
    for (uint32_t i = 0; i < num_tiles; i++) {
        // Calculate CB address for this tile (double buffering)
        uint32_t cb_tile_offset = (i % 2) * TILE_SIZE;
        uint32_t l1_write_addr = cb_addr + cb_tile_offset;

        // Read tile from DRAM
        uint64_t dram_noc_addr = get_noc_addr(i, src_dram);
        noc_async_read(dram_noc_addr, l1_write_addr, TILE_SIZE);
        noc_async_read_barrier();

        // Signal tile ready (in real implementation, would use CB push)
    }
}
