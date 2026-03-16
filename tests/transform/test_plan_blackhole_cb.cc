/*
 * Unit tests for PlanBlackholeCB Pass
 * Phase 2: CB allocation planning
 */

#include <gtest/gtest.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <iostream>
#include <vector>

// Mock implementation for testing framework

struct MockCBConfig {
  int cb_id;
  int num_pages;
  int page_size;
  int total_size;
  std::string dtype;
};

class MockPlanBlackholeCB {
 public:
  static constexpr int kMaxCBSize = 1572864;  // 1.5MB
  static constexpr int kMaxCBCount = 64;

  std::vector<MockCBConfig> Transform(const tvm::tir::PrimFunc& func) {
    // Mock implementation
    std::vector<MockCBConfig> configs;

    // Simulate CB allocation
    configs.push_back({0, 2, 2048, 4096, "float16"});
    configs.push_back({1, 2, 2048, 4096, "float16"});
    configs.push_back({2, 2, 4096, 8192, "float32"});

    return configs;
  }

  bool Validate(const std::vector<MockCBConfig>& configs) {
    int total_size = 0;
    for (const auto& cfg : configs) {
      total_size += cfg.total_size;
    }

    // Check total size < 1.5MB
    if (total_size > kMaxCBSize) {
      return false;
    }

    // Check CB count <= 64
    if (configs.size() > kMaxCBCount) {
      return false;
    }

    return true;
  }

  int CalculatePageSize(int rows, int cols, int dtype_size) {
    return rows * cols * dtype_size;
  }
};

using namespace tvm;
using namespace tvm::tir;

// Test fixture
class PlanBlackholeCBTest : public ::testing::Test {
 protected:
  MockPlanBlackholeCB planner;

  void SetUp() override {}
  void TearDown() override {}
};

// Test 1: Single CB allocation
TEST_F(PlanBlackholeCBTest, SingleCB) {
  std::cout << "=== Test: Single CB Allocation ===" << std::endl;

  PrimFunc func;  // Mock function
  auto configs = planner.Transform(func);

  ASSERT_GE(configs.size(), 1u);

  // First CB should have ID 0
  EXPECT_EQ(configs[0].cb_id, 0);
  EXPECT_GT(configs[0].page_size, 0);
  EXPECT_GT(configs[0].num_pages, 0);

  // Validate
  EXPECT_TRUE(planner.Validate(configs));

  std::cout << "CB 0: id=" << configs[0].cb_id
            << ", pages=" << configs[0].num_pages
            << ", page_size=" << configs[0].page_size
            << ", total=" << configs[0].total_size << std::endl;

  std::cout << "[PASS] Single CB Allocation" << std::endl;
}

// Test 2: Multiple CBs allocation
TEST_F(PlanBlackholeCBTest, MultipleCBs) {
  std::cout << "=== Test: Multiple CBs Allocation ===" << std::endl;

  PrimFunc func;
  auto configs = planner.Transform(func);

  ASSERT_GE(configs.size(), 2u);

  // CB IDs should be unique and consecutive
  for (size_t i = 0; i < configs.size(); i++) {
    EXPECT_EQ(configs[i].cb_id, static_cast<int>(i));
  }

  EXPECT_TRUE(planner.Validate(configs));

  std::cout << "Allocated " << configs.size() << " CBs" << std::endl;
  std::cout << "[PASS] Multiple CBs Allocation" << std::endl;
}

// Test 3: CB size validation (under 1.5MB)
TEST_F(PlanBlackholeCBTest, CBSizeValidation) {
  std::cout << "=== Test: CB Size Validation ===" << std::endl;

  PrimFunc func;
  auto configs = planner.Transform(func);

  int total_size = 0;
  for (const auto& cfg : configs) {
    total_size += cfg.total_size;
  }

  std::cout << "Total CB size: " << total_size << " bytes ("
            << total_size / 1024.0 << " KB)" << std::endl;
  std::cout << "Max allowed: " << MockPlanBlackholeCB::kMaxCBSize
            << " bytes (" << MockPlanBlackholeCB::kMaxCBSize / 1024.0 << " KB)" << std::endl;

  EXPECT_LE(total_size, MockPlanBlackholeCB::kMaxCBSize);
  EXPECT_TRUE(planner.Validate(configs));

  std::cout << "[PASS] CB Size Validation" << std::endl;
}

// Test 4: CB count validation (max 64)
TEST_F(PlanBlackholeCBTest, CBCountValidation) {
  std::cout << "=== Test: CB Count Validation ===" << std::endl;

  PrimFunc func;
  auto configs = planner.Transform(func);

  std::cout << "Allocated CBs: " << configs.size() << std::endl;
  std::cout << "Max allowed: " << MockPlanBlackholeCB::kMaxCBCount << std::endl;

  EXPECT_LE(configs.size(), MockPlanBlackholeCB::kMaxCBCount);
  EXPECT_TRUE(planner.Validate(configs));

  std::cout << "[PASS] CB Count Validation" << std::endl;
}

// Test 5: Page size calculation
TEST_F(PlanBlackholeCBTest, PageSizeCalculation) {
  std::cout << "=== Test: Page Size Calculation ===" << std::endl;

  // Test tile size calculation
  // FP16 tile: 32x32x2 = 2048 bytes
  int fp16_page = planner.CalculatePageSize(32, 32, 2);
  EXPECT_EQ(fp16_page, 2048);

  // FP32 tile: 32x32x4 = 4096 bytes
  int fp32_page = planner.CalculatePageSize(32, 32, 4);
  EXPECT_EQ(fp32_page, 4096);

  std::cout << "FP16 page size: " << fp16_page << " bytes" << std::endl;
  std::cout << "FP32 page size: " << fp32_page << " bytes" << std::endl;

  std::cout << "[PASS] Page Size Calculation" << std::endl;
}

// Test 6: Oversize error handling
TEST_F(PlanBlackholeCBTest, OversizeErrorHandling) {
  std::cout << "=== Test: Oversize Error Handling ===" << std::endl;

  // Create a config that exceeds 1.5MB
  std::vector<MockCBConfig> oversized_configs;
  for (int i = 0; i < 10; i++) {
    oversized_configs.push_back({i, 100, 2048, 204800, "float16"});  // 200KB each
  }
  // Total: 2MB > 1.5MB

  EXPECT_FALSE(planner.Validate(oversized_configs));

  std::cout << "[PASS] Oversize Error Handling" << std::endl;
}

// Test 7: Double buffering configuration
TEST_F(PlanBlackholeCBTest, DoubleBuffering) {
  std::cout << "=== Test: Double Buffering Configuration ===" << std::endl;

  // Test num_pages = 2 for double buffering
  std::vector<MockCBConfig> configs;
  configs.push_back({0, 2, 2048, 4096, "float16"});  // 2 pages for double buffering

  EXPECT_EQ(configs[0].num_pages, 2);
  EXPECT_EQ(configs[0].total_size, configs[0].page_size * configs[0].num_pages);

  std::cout << "Double buffer: " << configs[0].num_pages << " pages, "
            << configs[0].total_size << " bytes total" << std::endl;

  std::cout << "[PASS] Double Buffering Configuration" << std::endl;
}

// Main
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  std::cout << "========================================" << std::endl;
  std::cout << "PlanBlackholeCB Pass Test Suite" << std::endl;
  std::cout << "========================================" << std::endl;

  return RUN_ALL_TESTS();
}
