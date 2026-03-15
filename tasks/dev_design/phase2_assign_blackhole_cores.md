# Phase 2: AssignBlackholeCores Pass

## 任务目标

将 T.Kernel 的 grid 工作负载分配到 Blackhole 的 14x10 Tensix core grid 上。

## 背景

Blackhole 架构特点：
- **Tensix Grid**: 14 x 10 = 140 cores
- **Physical Grid**: 17 x 12 (包含以太网/DMA cores)
- **工作方式**: 软件显式切分工作负载到每个 core

与 GPU 不同，Blackhole 没有硬件 GigaThread Engine，需要软件显式分配。

## 技术方案

### 输入输出

**输入**:
```python
@T.prim_func
def kernel(A: T.Buffer, B: T.Buffer):
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M)) as (bx, by):
        # bx in [0, grid_x), by in [0, grid_y)
        T.copy(A[by*block_M], B)
```

**输出**:
```cpp
// Core 分配配置 (存储在 function attributes)
CoreAssignment {
    grid_dims: {grid_x, grid_y},        // T.Kernel grid 维度
    core_grid: {14, 10},                // Blackhole core grid
    work_per_core: {work_x, work_y},    // 每个 core 处理多少 work items
    total_cores_needed: N               // 需要的 core 数量
}

// 每个 core 运行时接收的参数
RuntimeArgs {
    work_offset_x,  // 此 core 负责的 work item 起始 x
    work_offset_y,  // 此 core 负责的 work item 起始 y
    work_count_x,   // 此 core 处理的 work items x 数量
    work_count_y    // 此 core 处理的 work items y 数量
}
```

### 分配策略

**1D Grid** (grid_x only):
```cpp
// grid_x work items 分配到 140 cores
work_per_core = ceil(grid_x / 140)
cores_needed = ceil(grid_x / work_per_core)

// Core (x, y) 处理 work items:
// work_offset = (x + y*14) * work_per_core
// work_count = min(work_per_core, grid_x - work_offset)
```

**2D Grid** (grid_x, grid_y):
```cpp
// 2D grid 映射到 1D core list，再映射回 2D
// Core index: idx = x + y * 14 (0 <= idx < 140)
// Work item index: widx = bx + by * grid_x

work_per_core = ceil(grid_x * grid_y / 140)
```

### Core 坐标计算

```cpp
// Blackhole Tensix core 有效坐标
// x: 1-7, 10-16 (避开 x=8,9 的 DRAM/ARC/Eth 区域)
// y: 2-11

CoreCoord GetCoreCoord(int core_idx) {
    int x_in_grid = core_idx % 14;
    int y_in_grid = core_idx / 14;

    // 映射到实际物理坐标
    int physical_x = (x_in_grid < 7) ? x_in_grid + 1 : x_in_grid + 3;
    int physical_y = y_in_grid + 2;

    return CoreCoord{physical_x, physical_y};
}
```

## 实施步骤

### 1. 实现 AssignBlackholeCores Pass

**文件**: `src/transform/assign_blackhole_cores.h/cc`

```cpp
struct CoreAssignment {
    int grid_x, grid_y;           // T.Kernel grid dimensions
    int core_grid_x, core_grid_y; // Blackhole core grid (14, 10)
    int work_per_core;            // Work items per core
    int cores_needed;             // Total cores needed
};

struct RuntimeArgs {
    int work_offset_x, work_offset_y;
    int work_count_x, work_count_y;
};

class AssignBlackholeCores : public tvm::tir::StmtExprMutator {
 public:
  // Main entry point
  tvm::tir::PrimFunc Transform(const tvm::tir::PrimFunc& func);

  // Get core assignment
  CoreAssignment GetCoreAssignment() const { return assignment_; }

  // Calculate runtime args for a specific core
  RuntimeArgs GetRuntimeArgs(int core_idx) const;

  // Get physical core coordinate
  CoreCoord GetCoreCoord(int core_idx) const;

 private:
  // Analyze T.Kernel grid dimensions
  CoreAssignment AnalyzeGrid(const tvm::tir::PrimFunc& func);

  // Calculate work distribution
  void CalculateWorkDistribution(CoreAssignment& assignment);

  // Store assignment in function attributes
  void StoreAssignment(tvm::tir::PrimFunc& func, const CoreAssignment& assignment);

  CoreAssignment assignment_;
  static constexpr int kBlackholeGridX = 14;
  static constexpr int kBlackholeGridY = 10;
  static constexpr int kTotalCores = 140;
};
```

### 2. 创建单元测试

**文件**: `tests/transform/test_assign_blackhole_cores.cc`

测试用例：
1. **SmallGrid**: grid < 140，使用部分 cores
2. **ExactMatch**: grid = 140，每个 core 一个 work item
3. **LargeGrid**: grid > 140，每个 core 多个 work items
4. **2DGrid**: 2D grid 映射到 cores
5. **CoreCoord**: 验证物理坐标计算正确

## 验证方法

### 单元测试检查点

1. **Work Distribution**:
   - 所有 work items 都被分配
   - 没有重复分配
   - work_per_core 计算正确

2. **Core Coord**:
   - 物理坐标在有效范围内 (x: 1-7,10-16, y: 2-11)
   - 不分配到 DRAM/Eth cores

3. **Runtime Args**:
   - work_offset + work_count 不越界
   - 每个 core 的参数正确计算

4. **边界情况**:
   - grid = 1 (最小情况)
   - grid = 140 (恰好匹配)
   - grid = 1000 (超大 grid)

## 开发记录

### 2026-03-16

- **完成**: Pass 框架实现
- **实现**: AssignBlackholeCores 类，支持 Transform/GetRuntimeArgs/GetCoreCoord
- **实现**: GridAnalyzer Visitor 分析 T.Kernel grid 维度
- **单元测试**: 8 个测试用例 (SmallGrid, ExactMatch, LargeGrid, CoreCoordCalculation, RuntimeArgsCalculation, ValidCoreCoordRange, OneDimensionalGrid, WorkDistributionCorrectness)

## 经验总结

- **设计决策**: 使用逻辑 core index (0-139) 作为中间层，再映射到物理坐标，简化计算
- **实现技巧**: Core 坐标映射公式 `physical_x = (x < 7) ? x + 1 : x + 3` 巧妙避开 x=8,9
- **测试策略**: 边界情况测试很重要，特别是 grid=140 恰好匹配的情况
- **踩坑记录**: 物理坐标 y 的范围是 2-11，不是 0-9，需要 +2 偏移

## 预期产出

1. `src/transform/assign_blackhole_cores.h`
2. `src/transform/assign_blackhole_cores.cc`
3. `tests/transform/test_assign_blackhole_cores.cc`

## 参考

- `tasks/arch_design.md`: 任务切分映射设计
- Blackhole core 布局: 14x10 Tensix cores
