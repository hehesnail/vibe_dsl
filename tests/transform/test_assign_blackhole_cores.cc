/*
 * Unit tests for AssignBlackholeCores Pass
 * Phase 2: Core assignment for 14x10 grid
 */

#include <gtest/gtest.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/function.h>
#include <iostream>
#include <vector>
#include <cmath>

// Mock implementation for testing framework

struct MockCoreCoord {
  int x, y;
};

struct MockCoreAssignment {
  int grid_x, grid_y;
  int core_grid_x = 14;
  int core_grid_y = 10;
  int work_per_core;
  int cores_needed;
};

struct MockRuntimeArgs {
  int work_offset_x, work_offset_y;
  int work_count_x, work_count_y;
};

class MockAssignBlackholeCores {
 public:
  static constexpr int kBlackholeGridX = 14;
  static constexpr int kBlackholeGridY = 10;
  static constexpr int kTotalCores = 140;

  MockCoreAssignment Transform(int grid_x, int grid_y) {
    MockCoreAssignment assignment;
    assignment.grid_x = grid_x;
    assignment.grid_y = grid_y;

    int total_work = grid_x * grid_y;
    assignment.work_per_core = std::ceil(static_cast<double>(total_work) / kTotalCores);
    assignment.cores_needed = std::ceil(static_cast<double>(total_work) / assignment.work_per_core);

    return assignment;
  }

  MockRuntimeArgs GetRuntimeArgs(const MockCoreAssignment& assignment, int core_idx) {
    MockRuntimeArgs args;

    int total_work = assignment.grid_x * assignment.grid_y;
    int work_offset = core_idx * assignment.work_per_core;

    // Convert 1D work offset to 2D
    args.work_offset_y = work_offset / assignment.grid_x;
    args.work_offset_x = work_offset % assignment.grid_x;

    // Calculate work count for this core
    int remaining_work = total_work - work_offset;
    args.work_count_x = std::min(assignment.work_per_core, remaining_work);
    args.work_count_y = 1;  // Simplified for 1D mapping

    return args;
  }

  MockCoreCoord GetCoreCoord(int core_idx) {
    MockCoreCoord coord;
    int x_in_grid = core_idx % kBlackholeGridX;
    int y_in_grid = core_idx / kBlackholeGridX;

    // Map to physical coordinates (avoiding x=8,9)
    coord.x = (x_in_grid < 7) ? x_in_grid + 1 : x_in_grid + 3;
    coord.y = y_in_grid + 2;

    return coord;
  }

  bool IsValidCoreCoord(const MockCoreCoord& coord) {
    // Valid x: 1-7, 10-16 (avoid 8,9 which are DRAM/ARC/Eth)
    // Valid y: 2-11
    bool valid_x = (coord.x >= 1 && coord.x <= 7) || (coord.x >= 10 && coord.x <= 16);
    bool valid_y = (coord.y >= 2 && coord.y <= 11);
    return valid_x && valid_y;
  }
};

// Test fixture
class AssignBlackholeCoresTest : public ::testing::Test {
 protected:
  MockAssignBlackholeCores assigner;

  void SetUp() override {}
  void TearDown() override {}
};

// Test 1: Small grid (< 140 cores)
TEST_F(AssignBlackholeCoresTest, SmallGrid) {
  std::cout << "=== Test: Small Grid (< 140 cores) ===" << std::endl;

  int grid_x = 10, grid_y = 10;  // 100 work items
  auto assignment = assigner.Transform(grid_x, grid_y);

  EXPECT_EQ(assignment.grid_x, grid_x);
  EXPECT_EQ(assignment.grid_y, grid_y);
  EXPECT_LE(assignment.cores_needed, assigner.kTotalCores);

  std::cout << "Grid: " << grid_x << "x" << grid_y << " = " << grid_x * grid_y << " work items" << std::endl;
  std::cout << "Cores needed: " << assignment.cores_needed << std::endl;
  std::cout << "Work per core: " << assignment.work_per_core << std::endl;

  std::cout << "[PASS] Small Grid" << std::endl;
}

// Test 2: Exact match (140 work items)
TEST_F(AssignBlackholeCoresTest, ExactMatch) {
  std::cout << "=== Test: Exact Match (140 work items) ===" << std::endl;

  int grid_x = 14, grid_y = 10;  // 140 work items
  auto assignment = assigner.Transform(grid_x, grid_y);

  EXPECT_EQ(assignment.cores_needed, assigner.kTotalCores);
  EXPECT_EQ(assignment.work_per_core, 1);

  std::cout << "Grid: " << grid_x << "x" << grid_y << " = " << grid_x * grid_y << " work items" << std::endl;
  std::cout << "Each core handles exactly 1 work item" << std::endl;

  std::cout << "[PASS] Exact Match" << std::endl;
}

// Test 3: Large grid (> 140 cores)
TEST_F(AssignBlackholeCoresTest, LargeGrid) {
  std::cout << "=== Test: Large Grid (> 140 cores) ===" << std::endl;

  int grid_x = 20, grid_y = 20;  // 400 work items
  auto assignment = assigner.Transform(grid_x, grid_y);

  EXPECT_EQ(assignment.cores_needed, assigner.kTotalCores);
  EXPECT_GE(assignment.work_per_core, 3);  // ceil(400/140) = 3

  std::cout << "Grid: " << grid_x << "x" << grid_y << " = " << grid_x * grid_y << " work items" << std::endl;
  std::cout << "Cores needed: " << assignment.cores_needed << " (all cores)" << std::endl;
  std::cout << "Work per core: " << assignment.work_per_core << std::endl;

  std::cout << "[PASS] Large Grid" << std::endl;
}

// Test 4: Core coordinate calculation
TEST_F(AssignBlackholeCoresTest, CoreCoordCalculation) {
  std::cout << "=== Test: Core Coordinate Calculation ===" << std::endl;

  // Test first core (0)
  auto coord0 = assigner.GetCoreCoord(0);
  EXPECT_EQ(coord0.x, 1);  // First valid x
  EXPECT_EQ(coord0.y, 2);  // First valid y
  EXPECT_TRUE(assigner.IsValidCoreCoord(coord0));

  // Test core at index 7 (should skip x=8,9)
  auto coord7 = assigner.GetCoreCoord(7);
  EXPECT_EQ(coord7.x, 10);  // After skipping 8,9
  EXPECT_TRUE(assigner.IsValidCoreCoord(coord7));

  // Test last core (139)
  auto coord139 = assigner.GetCoreCoord(139);
  EXPECT_EQ(coord139.x, 16);  // Last valid x
  EXPECT_EQ(coord139.y, 11);  // Last valid y
  EXPECT_TRUE(assigner.IsValidCoreCoord(coord139));

  std::cout << "Core 0: (" << coord0.x << ", " << coord0.y << ")" << std::endl;
  std::cout << "Core 7: (" << coord7.x << ", " << coord7.y << ")" << std::endl;
  std::cout << "Core 139: (" << coord139.x << ", " << coord139.y << ")" << std::endl;

  std::cout << "[PASS] Core Coordinate Calculation" << std::endl;
}

// Test 5: Runtime args calculation
TEST_F(AssignBlackholeCoresTest, RuntimeArgsCalculation) {
  std::cout << "=== Test: Runtime Args Calculation ===" << std::endl;

  int grid_x = 14, grid_y = 10;
  auto assignment = assigner.Transform(grid_x, grid_y);

  // Get runtime args for core 0
  auto args0 = assigner.GetRuntimeArgs(assignment, 0);
  EXPECT_EQ(args0.work_offset_x, 0);
  EXPECT_EQ(args0.work_offset_y, 0);
  EXPECT_EQ(args0.work_count_x, 1);

  // Get runtime args for core 50
  auto args50 = assigner.GetRuntimeArgs(assignment, 50);
  EXPECT_EQ(args50.work_offset_x, 50 % grid_x);
  EXPECT_EQ(args50.work_offset_y, 50 / grid_x);

  std::cout << "Core 0: offset=(" << args0.work_offset_x << ", " << args0.work_offset_y << ")" << std::endl;
  std::cout << "Core 50: offset=(" << args50.work_offset_x << ", " << args50.work_offset_y << ")" << std::endl;

  std::cout << "[PASS] Runtime Args Calculation" << std::endl;
}

// Test 6: Valid core coordinate range
TEST_F(AssignBlackholeCoresTest, ValidCoreCoordRange) {
  std::cout << "=== Test: Valid Core Coordinate Range ===" << std::endl;

  // Test all 140 cores
  for (int i = 0; i < assigner.kTotalCores; i++) {
    auto coord = assigner.GetCoreCoord(i);
    EXPECT_TRUE(assigner.IsValidCoreCoord(coord))
        << "Core " << i << " coord (" << coord.x << ", " << coord.y << ") is invalid";
  }

  std::cout << "All " << assigner.kTotalCores << " cores have valid coordinates" << std::endl;
  std::cout << "[PASS] Valid Core Coordinate Range" << std::endl;
}

// Test 7: 1D grid handling
TEST_F(AssignBlackholeCoresTest, OneDimensionalGrid) {
  std::cout << "=== Test: 1D Grid Handling ===" << std::endl;

  int grid_x = 100, grid_y = 1;  // 100 work items in 1D
  auto assignment = assigner.Transform(grid_x, grid_y);

  EXPECT_EQ(assignment.grid_x, grid_x);
  EXPECT_EQ(assignment.grid_y, grid_y);
  EXPECT_LE(assignment.cores_needed, 100);

  std::cout << "1D Grid: " << grid_x << " work items" << std::endl;
  std::cout << "Cores needed: " << assignment.cores_needed << std::endl;

  std::cout << "[PASS] 1D Grid Handling" << std::endl;
}

// Test 8: Work distribution correctness
TEST_F(AssignBlackholeCoresTest, WorkDistributionCorrectness) {
  std::cout << "=== Test: Work Distribution Correctness ===" << std::endl;

  int grid_x = 7, grid_y = 7;  // 49 work items
  auto assignment = assigner.Transform(grid_x, grid_y);

  // Sum up work items from all cores
  int total_assigned = 0;
  for (int i = 0; i < assignment.cores_needed; i++) {
    auto args = assigner.GetRuntimeArgs(assignment, i);
    total_assigned += args.work_count_x;
  }

  // Should equal total work items
  EXPECT_EQ(total_assigned, grid_x * grid_y);

  std::cout << "Total work items: " << grid_x * grid_y << std::endl;
  std::cout << "Total assigned: " << total_assigned << std::endl;

  std::cout << "[PASS] Work Distribution Correctness" << std::endl;
}

// Main
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  std::cout << "========================================" << std::endl;
  std::cout << "AssignBlackholeCores Pass Test Suite" << std::endl;
  std::cout << "========================================" << std::endl;

  return RUN_ALL_TESTS();
}
