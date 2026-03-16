// SPDX-FileCopyrightText: © 2025 TileLang Blackhole
// SPDX-License-Identifier: Apache-2.0

// Phase 3: GEMM Compute Kernel (TRISC)
// Operation: C = A @ B

#include "compute_kernel_api.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tile_move_copy.h"

void kernel_main() {
    // Compile-time tile dimensions
    constexpr uint32_t Mt = 1;
    constexpr uint32_t Kt = 4;
    constexpr uint32_t Nt = 1;

    // CB configuration
    constexpr uint32_t cb_in0 = 0;  // A matrix
    constexpr uint32_t cb_in1 = 1;  // B matrix
    constexpr uint32_t cb_out = 16; // C matrix

    // Initialize matrix engine
    mm_init(cb_in0, cb_in1, cb_out);

    // Outer product loop over tiles
    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            // Acquire DST registers
            tile_regs_acquire();

            // Accumulate over K dimension
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                // Wait for input tiles
                cb_wait_front(cb_in0, 1);
                cb_wait_front(cb_in1, 1);

                // Perform matmul: C += A * B
                matmul_tiles(cb_in0, cb_in1, 0, 0, 0);

                // Release input tiles
                cb_pop_front(cb_in0, 1);
                cb_pop_front(cb_in1, 1);
            }

            // Commit compute results
            tile_regs_commit();
            tile_regs_wait();

            // Reserve output space and pack result
            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);

            // Release DST registers
            tile_regs_release();
        }
    }
}
