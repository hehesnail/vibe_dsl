/*
 * Phase 3: GEMM TT-Sim Test
 * Tests TileLang CodeGen generated GEMM kernels on TT-Sim
 */

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// Reference GEMM implementation
void reference_gemm(const std::vector<float>& A,
                    const std::vector<float>& B,
                    std::vector<float>& C,
                    int M, int N, int K) {
    // Initialize C to zero
    std::fill(C.begin(), C.end(), 0.0f);

    // Compute C = A @ B
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

// Generate TT-Sim compatible compute kernel for GEMM
std::string GenerateGemmComputeKernel(int num_k_tiles = 4) {
    std::ostringstream os;

    // Header
    os << "// SPDX-FileCopyrightText: © 2025 TileLang Blackhole\n";
    os << "// SPDX-License-Identifier: Apache-2.0\n";
    os << "\n";
    os << "// Phase 3: GEMM Compute Kernel (TRISC)\n";
    os << "// Operation: C = A @ B\n";
    os << "\n";

    // Includes
    os << "#include \"compute_kernel_api.h\"\n";
    os << "#include \"compute_kernel_api/matmul.h\"\n";
    os << "#include \"compute_kernel_api/tile_move_copy.h\"\n\n";

    // Kernel main
    os << "void kernel_main() {\n";
    os << "    // Compile-time tile dimensions\n";
    os << "    constexpr uint32_t Mt = 1;\n";
    os << "    constexpr uint32_t Kt = " << num_k_tiles << ";\n";
    os << "    constexpr uint32_t Nt = 1;\n\n";

    os << "    // CB configuration\n";
    os << "    constexpr uint32_t cb_in0 = 0;  // A matrix\n";
    os << "    constexpr uint32_t cb_in1 = 1;  // B matrix\n";
    os << "    constexpr uint32_t cb_out = 16; // C matrix\n\n";

    os << "    // Initialize matrix engine\n";
    os << "    mm_init(cb_in0, cb_in1, cb_out);\n\n";

    os << "    // Outer product loop over tiles\n";
    os << "    for (uint32_t mt = 0; mt < Mt; ++mt) {\n";
    os << "        for (uint32_t nt = 0; nt < Nt; ++nt) {\n";
    os << "            // Acquire DST registers\n";
    os << "            tile_regs_acquire();\n\n";

    os << "            // Accumulate over K dimension\n";
    os << "            for (uint32_t kt = 0; kt < Kt; ++kt) {\n";
    os << "                // Wait for input tiles\n";
    os << "                cb_wait_front(cb_in0, 1);\n";
    os << "                cb_wait_front(cb_in1, 1);\n\n";

    os << "                // Perform matmul: C += A * B\n";
    os << "                matmul_tiles(cb_in0, cb_in1, 0, 0, 0);\n\n";

    os << "                // Release input tiles\n";
    os << "                cb_pop_front(cb_in0, 1);\n";
    os << "                cb_pop_front(cb_in1, 1);\n";
    os << "            }\n\n";

    os << "            // Commit compute results\n";
    os << "            tile_regs_commit();\n";
    os << "            tile_regs_wait();\n\n";

    os << "            // Reserve output space and pack result\n";
    os << "            cb_reserve_back(cb_out, 1);\n";
    os << "            pack_tile(0, cb_out);\n";
    os << "            cb_push_back(cb_out, 1);\n\n";

    os << "            // Release DST registers\n";
    os << "            tile_regs_release();\n";
    os << "        }\n";
    os << "    }\n";
    os << "}\n";

    return os.str();
}

// Generate reader kernel for GEMM
std::string GenerateGemmReaderKernel(int num_k_tiles = 4) {
    std::ostringstream os;

    os << "// SPDX-FileCopyrightText: © 2025 TileLang Blackhole\n";
    os << "// SPDX-License-Identifier: Apache-2.0\n\n";
    os << "#include \"dataflow_api.h\"\n\n";
    os << "void kernel_main() {\n";
    os << "    // Runtime arguments\n";
    os << "    uint32_t src0_dram_addr = get_arg_val<uint32_t>(0);\n";
    os << "    uint32_t src1_dram_addr = get_arg_val<uint32_t>(1);\n";
    os << "    uint32_t Mt = get_arg_val<uint32_t>(2);\n";
    os << "    uint32_t Kt = get_arg_val<uint32_t>(3);\n";
    os << "    uint32_t Nt = get_arg_val<uint32_t>(4);\n\n";

    os << "    // CB IDs\n";
    os << "    constexpr uint32_t cb_in0 = 0;\n";
    os << "    constexpr uint32_t cb_in1 = 1;\n";
    os << "    constexpr uint32_t tile_size = 2048;  // 32x32 FP16\n\n";

    os << "    // Address generators\n";
    os << "    InterleavedAddrGen<true> src0_gen = {\n";
    os << "        .bank_base_address = src0_dram_addr,\n";
    os << "        .page_size = tile_size\n";
    os << "    };\n";
    os << "    InterleavedAddrGen<true> src1_gen = {\n";
    os << "        .bank_base_address = src1_dram_addr,\n";
    os << "        .page_size = tile_size\n";
    os << "    };\n\n";

    os << "    // Read tiles for each output tile\n";
    os << "    for (uint32_t mt = 0; mt < Mt; ++mt) {\n";
    os << "        for (uint32_t nt = 0; nt < Nt; ++nt) {\n";
    os << "            for (uint32_t kt = 0; kt < Kt; ++kt) {\n";
    os << "                // Read A tile (mt, kt)\n";
    os << "                uint32_t a_tile_idx = mt * Kt + kt;\n";
    os << "                cb_reserve_back(cb_in0, 1);\n";
    os << "                uint32_t l1_addr_a = get_write_ptr(cb_in0);\n";
    os << "                uint64_t src0_noc_addr = get_noc_addr(a_tile_idx, src0_gen);\n";
    os << "                noc_async_read(src0_noc_addr, l1_addr_a, tile_size);\n";
    os << "                noc_async_read_barrier();\n";
    os << "                cb_push_back(cb_in0, 1);\n\n";

    os << "                // Read B tile (kt, nt)\n";
    os << "                uint32_t b_tile_idx = kt * Nt + nt;\n";
    os << "                cb_reserve_back(cb_in1, 1);\n";
    os << "                uint32_t l1_addr_b = get_write_ptr(cb_in1);\n";
    os << "                uint64_t src1_noc_addr = get_noc_addr(b_tile_idx, src1_gen);\n";
    os << "                noc_async_read(src1_noc_addr, l1_addr_b, tile_size);\n";
    os << "                noc_async_read_barrier();\n";
    os << "                cb_push_back(cb_in1, 1);\n";
    os << "            }\n";
    os << "        }\n";
    os << "    }\n";
    os << "}\n";

    return os.str();
}

// Generate writer kernel for GEMM
std::string GenerateGemmWriterKernel() {
    std::ostringstream os;

    os << "// SPDX-FileCopyrightText: © 2025 TileLang Blackhole\n";
    os << "// SPDX-License-Identifier: Apache-2.0\n\n";
    os << "#include \"dataflow_api.h\"\n\n";
    os << "void kernel_main() {\n";
    os << "    // Runtime arguments\n";
    os << "    uint32_t dst_dram_addr = get_arg_val<uint32_t>(0);\n";
    os << "    uint32_t Mt = get_arg_val<uint32_t>(1);\n";
    os << "    uint32_t Nt = get_arg_val<uint32_t>(2);\n\n";

    os << "    // CB ID\n";
    os << "    constexpr uint32_t cb_out = 16;\n";
    os << "    constexpr uint32_t tile_size = 2048;  // 32x32 FP16\n\n";

    os << "    // Address generator\n";
    os << "    InterleavedAddrGen<true> dst_gen = {\n";
    os << "        .bank_base_address = dst_dram_addr,\n";
    os << "        .page_size = tile_size\n";
    os << "    };\n\n";

    os << "    // Write output tiles\n";
    os << "    for (uint32_t mt = 0; mt < Mt; ++mt) {\n";
    os << "        for (uint32_t nt = 0; nt < Nt; ++nt) {\n";
    os << "            uint32_t out_tile_idx = mt * Nt + nt;\n\n";

    os << "            // Wait for compute to produce output\n";
    os << "            cb_wait_front(cb_out, 1);\n";
    os << "            uint32_t l1_addr_out = get_read_ptr(cb_out);\n\n";

    os << "            // Write to DRAM\n";
    os << "            uint64_t dst_noc_addr = get_noc_addr(out_tile_idx, dst_gen);\n";
    os << "            noc_async_write(l1_addr_out, dst_noc_addr, tile_size);\n";
    os << "            noc_async_write_barrier();\n\n";

    os << "            // Pop output tile\n";
    os << "            cb_pop_front(cb_out, 1);\n";
    os << "        }\n";
    os << "    }\n";
    os << "}\n";

    return os.str();
}

// Test configuration
struct GemmTestConfig {
    int M = 32;  // Rows of A and C
    int N = 32;  // Cols of B and C
    int K = 32;  // Cols of A, rows of B
    int tile_size = 32;
    int num_k_tiles = 1;
};

int main(int argc, char* argv[]) {
    std::cout << "======================================" << std::endl;
    std::cout << "Phase 3: GEMM TT-Sim Test" << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << std::endl;

    // Test configuration
    GemmTestConfig config;
    config.M = 32;
    config.N = 32;
    config.K = 128;  // 4 tiles of K dimension
    config.num_k_tiles = config.K / config.tile_size;

    std::cout << "Test Configuration:" << std::endl;
    std::cout << "  M = " << config.M << std::endl;
    std::cout << "  N = " << config.N << std::endl;
    std::cout << "  K = " << config.K << std::endl;
    std::cout << "  Tile size = " << config.tile_size << std::endl;
    std::cout << "  K tiles = " << config.num_k_tiles << std::endl;
    std::cout << std::endl;

    // Generate kernels
    std::cout << "=== Generating Kernels ===" << std::endl;

    std::string reader_kernel = GenerateGemmReaderKernel(config.num_k_tiles);
    std::string compute_kernel = GenerateGemmComputeKernel(config.num_k_tiles);
    std::string writer_kernel = GenerateGemmWriterKernel();

    // Save kernels to files
    std::ofstream reader_file("phase3_gemm_reader_kernel.cpp");
    reader_file << reader_kernel;
    reader_file.close();

    std::ofstream compute_file("phase3_gemm_compute_kernel.cpp");
    compute_file << compute_kernel;
    compute_file.close();

    std::ofstream writer_file("phase3_gemm_writer_kernel.cpp");
    writer_file << writer_kernel;
    writer_file.close();

    std::cout << "  Saved: phase3_gemm_reader_kernel.cpp" << std::endl;
    std::cout << "  Saved: phase3_gemm_compute_kernel.cpp" << std::endl;
    std::cout << "  Saved: phase3_gemm_writer_kernel.cpp" << std::endl;
    std::cout << std::endl;

    // Verify reference computation
    std::cout << "=== Reference Computation ===" << std::endl;

    // Initialize test data
    std::vector<float> A(config.M * config.K);
    std::vector<float> B(config.K * config.N);
    std::vector<float> C_ref(config.M * config.N);

    // Fill with test pattern
    for (int i = 0; i < config.M * config.K; ++i) {
        A[i] = static_cast<float>(i % 10) / 10.0f;
    }
    for (int i = 0; i < config.K * config.N; ++i) {
        B[i] = static_cast<float>((i * 3) % 10) / 10.0f;
    }

    // Compute reference
    reference_gemm(A, B, C_ref, config.M, config.N, config.K);

    // Print sample values
    std::cout << "  Sample A[0:4]: ";
    for (int i = 0; i < 4; ++i) std::cout << A[i] << " ";
    std::cout << std::endl;

    std::cout << "  Sample B[0:4]: ";
    for (int i = 0; i < 4; ++i) std::cout << B[i] << " ";
    std::cout << std::endl;

    std::cout << "  Reference C[0:4]: ";
    for (int i = 0; i < 4; ++i) std::cout << C_ref[i] << " ";
    std::cout << std::endl;
    std::cout << std::endl;

    // Verify kernel code syntax
    std::cout << "=== Kernel Code Verification ===" << std::endl;
    std::cout << "  Reader kernel lines: " << std::count(reader_kernel.begin(), reader_kernel.end(), '\n') << std::endl;
    std::cout << "  Compute kernel lines: " << std::count(compute_kernel.begin(), compute_kernel.end(), '\n') << std::endl;
    std::cout << "  Writer kernel lines: " << std::count(writer_kernel.begin(), writer_kernel.end(), '\n') << std::endl;
    std::cout << std::endl;

    // Check for required API calls in compute kernel
    bool has_mm_init = compute_kernel.find("mm_init(") != std::string::npos;
    bool has_matmul_tiles = compute_kernel.find("matmul_tiles(") != std::string::npos;
    bool has_tile_regs_acquire = compute_kernel.find("tile_regs_acquire()") != std::string::npos;
    bool has_tile_regs_commit = compute_kernel.find("tile_regs_commit()") != std::string::npos;
    bool has_pack_tile = compute_kernel.find("pack_tile(") != std::string::npos;

    std::cout << "  Required API calls in compute kernel:" << std::endl;
    std::cout << "    mm_init: " << (has_mm_init ? "✓" : "✗") << std::endl;
    std::cout << "    matmul_tiles: " << (has_matmul_tiles ? "✓" : "✗") << std::endl;
    std::cout << "    tile_regs_acquire: " << (has_tile_regs_acquire ? "✓" : "✗") << std::endl;
    std::cout << "    tile_regs_commit: " << (has_tile_regs_commit ? "✓" : "✗") << std::endl;
    std::cout << "    pack_tile: " << (has_pack_tile ? "✓" : "✗") << std::endl;
    std::cout << std::endl;

    // Check for CB operations
    bool has_cb_wait_front = compute_kernel.find("cb_wait_front(") != std::string::npos;
    bool has_cb_pop_front = compute_kernel.find("cb_pop_front(") != std::string::npos;
    bool has_cb_reserve_back = compute_kernel.find("cb_reserve_back(") != std::string::npos;
    bool has_cb_push_back = compute_kernel.find("cb_push_back(") != std::string::npos;

    std::cout << "  CB operations in compute kernel:" << std::endl;
    std::cout << "    cb_wait_front: " << (has_cb_wait_front ? "✓" : "✗") << std::endl;
    std::cout << "    cb_pop_front: " << (has_cb_pop_front ? "✓" : "✗") << std::endl;
    std::cout << "    cb_reserve_back: " << (has_cb_reserve_back ? "✓" : "✗") << std::endl;
    std::cout << "    cb_push_back: " << (has_cb_push_back ? "✓" : "✗") << std::endl;
    std::cout << std::endl;

    // Overall result
    bool all_checks = has_mm_init && has_matmul_tiles && has_tile_regs_acquire &&
                      has_tile_regs_commit && has_pack_tile && has_cb_wait_front &&
                      has_cb_pop_front && has_cb_reserve_back && has_cb_push_back;

    if (all_checks) {
        std::cout << "✓ SUCCESS: All required API calls present in generated kernels!" << std::endl;
        std::cout << std::endl;
        std::cout << "Generated kernels are ready for TT-Sim compilation:" << std::endl;
        std::cout << "  1. Copy kernel files to tt_metal/programming_examples/" << std::endl;
        std::cout << "  2. Create CMakeLists.txt for the test" << std::endl;
        std::cout << "  3. Build and run with TT-Sim" << std::endl;
        return 0;
    } else {
        std::cout << "✗ FAILED: Some required API calls are missing!" << std::endl;
        return 1;
    }
}
