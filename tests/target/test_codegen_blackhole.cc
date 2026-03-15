/*
 * Simple test for Blackhole CodeGen
 * Phase 1: Verify code generation produces correct TT-Metal style code
 */

#include <iostream>
#include <fstream>
#include <cstring>
#include <cassert>

// Include the header
#include "tilelang_repo/src/target/codegen_blackhole.h"

using namespace tvm::tl;

// Simple test helper
#define CHECK_CONTAINS(code, substr) \
  do { \
    if ((code).find(substr) == std::string::npos) { \
      std::cerr << "FAIL: Generated code missing '" << substr << "'" << std::endl; \
      return false; \
    } \
  } while(0)

bool test_simple_copy_kernel() {
  std::cout << "\n=== Test: Simple Copy Kernel ===" << std::endl;

  CodeGenBlackhole codegen;

  // Generate a simple copy kernel
  std::string code = codegen.GenerateSimpleCopyKernel(
      "simple_copy",    // func_name
      "src_dram",       // src_buf
      "dst_dram",       // dst_buf
      4,                // num_tiles
      2048              // tile_size_bytes (32x32x2 for FP16)
  );

  // Verify the generated code contains expected elements
  CHECK_CONTAINS(code, "#include \"dataflow_api.h\"");
  CHECK_CONTAINS(code, "void simple_copy(");
  CHECK_CONTAINS(code, "cb_reserve_back");
  CHECK_CONTAINS(code, "cb_push_back");
  CHECK_CONTAINS(code, "cb_wait_front");
  CHECK_CONTAINS(code, "cb_pop_front");
  CHECK_CONTAINS(code, "noc_async_read");
  CHECK_CONTAINS(code, "noc_async_write");
  CHECK_CONTAINS(code, "noc_async_read_barrier");
  CHECK_CONTAINS(code, "noc_async_write_barrier");
  CHECK_CONTAINS(code, "get_write_ptr");
  CHECK_CONTAINS(code, "get_read_ptr");

  // Save generated code to file
  std::ofstream out("generated_copy_kernel.cc");
  out << code;
  out.close();

  std::cout << "Generated code saved to: generated_copy_kernel.cc" << std::endl;
  std::cout << "Code size: " << code.size() << " bytes" << std::endl;

  return true;
}

bool test_reader_kernel() {
  std::cout << "\n=== Test: Reader Kernel ===" << std::endl;

  CodeGenBlackhole codegen;

  std::string code = codegen.GenerateReaderKernel(
      "reader_kernel",   // func_name
      "src_dram",        // src_buf
      0,                 // cb_id
      8,                 // num_tiles
      2048               // tile_size_bytes
  );

  CHECK_CONTAINS(code, "void reader_kernel()");
  CHECK_CONTAINS(code, "get_arg_val");
  CHECK_CONTAINS(code, "cb_reserve_back");
  CHECK_CONTAINS(code, "noc_async_read");

  // Save to file
  std::ofstream out("generated_reader_kernel.cc");
  out << code;
  out.close();

  std::cout << "Generated code saved to: generated_reader_kernel.cc" << std::endl;

  return true;
}

bool test_writer_kernel() {
  std::cout << "\n=== Test: Writer Kernel ===" << std::endl;

  CodeGenBlackhole codegen;

  std::string code = codegen.GenerateWriterKernel(
      "writer_kernel",   // func_name
      "dst_dram",        // dst_buf
      0,                 // cb_id
      8,                 // num_tiles
      2048               // tile_size_bytes
  );

  CHECK_CONTAINS(code, "void writer_kernel()");
  CHECK_CONTAINS(code, "cb_wait_front");
  CHECK_CONTAINS(code, "noc_async_write");
  CHECK_CONTAINS(code, "cb_pop_front");

  // Save to file
  std::ofstream out("generated_writer_kernel.cc");
  out << code;
  out.close();

  std::cout << "Generated code saved to: generated_writer_kernel.cc" << std::endl;

  return true;
}

void print_generated_code(const std::string& filename) {
  std::ifstream file(filename);
  if (file.is_open()) {
    std::cout << "\n--- Content of " << filename << " ---" << std::endl;
    std::string line;
    while (std::getline(file, line)) {
      std::cout << line << std::endl;
    }
    std::cout << "--- End of " << filename << " ---" << std::endl;
    file.close();
  }
}

int main() {
  std::cout << "======================================" << std::endl;
  std::cout << "CodeGen Blackhole Test Suite" << std::endl;
  std::cout << "======================================" << std::endl;

  int passed = 0;
  int failed = 0;

  // Run tests
  if (test_simple_copy_kernel()) {
    std::cout << "[PASS] Simple Copy Kernel" << std::endl;
    passed++;
  } else {
    std::cout << "[FAIL] Simple Copy Kernel" << std::endl;
    failed++;
  }

  if (test_reader_kernel()) {
    std::cout << "[PASS] Reader Kernel" << std::endl;
    passed++;
  } else {
    std::cout << "[FAIL] Reader Kernel" << std::endl;
    failed++;
  }

  if (test_writer_kernel()) {
    std::cout << "[PASS] Writer Kernel" << std::endl;
    passed++;
  } else {
    std::cout << "[FAIL] Writer Kernel" << std::endl;
    failed++;
  }

  // Print generated code for inspection
  print_generated_code("generated_copy_kernel.cc");

  // Summary
  std::cout << "\n======================================" << std::endl;
  std::cout << "Test Summary: " << passed << " passed, " << failed << " failed" << std::endl;
  std::cout << "======================================" << std::endl;

  return failed > 0 ? 1 : 0;
}
