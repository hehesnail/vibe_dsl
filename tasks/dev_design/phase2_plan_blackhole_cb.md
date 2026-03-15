# Phase 2: PlanBlackholeCB Pass

## 任务目标

将 `T.alloc_shared` 映射到 Blackhole 的 Circular Buffer (CB)，规划 CB 的分配策略。

## 背景

Blackhole 架构特点：
- **CB 数量**: 64 (CB 0-63)
- **L1 大小**: 1.5MB (1,572,864 bytes) 每 core
- **CB 用途**: Reader/Compute/Writer 之间传递数据

TileLang 的 `T.alloc_shared` 对应 Blackhole 的 CB。

## 技术方案

### 输入输出

**输入**:
```python
@T.prim_func
def kernel():
    A_shared = T.alloc_shared((block_M, block_K), "float16")
    B_shared = T.alloc_shared((block_K, block_N), "float16")
    C_local = T.alloc_fragment((block_M, block_N), "float32")
```

**输出**:
```cpp
// CB 配置 (存储在 function attributes)
CBConfig {
    {cb_id: 0, num_pages: 2, page_size: 2048, total_size: 4096},  // A_shared
    {cb_id: 1, num_pages: 2, page_size: 2048, total_size: 4096},  // B_shared
    {cb_id: 2, num_pages: 2, page_size: 4096, total_size: 8192},  // C_local
}

// 验证: total_size_sum = 16384 < 1.5MB, num_cbs = 3 <= 64
```

### 分配策略

1. **顺序分配**: CB ID 从 0 开始递增分配
2. **Page 大小**: tile_size (32x32xsizeof(dtype))
3. **Num Pages**: 由 double_buffer / num_stages 决定

### 计算公式

```cpp
// 计算 CB 大小
page_size = tile_rows * tile_cols * sizeof(dtype)
num_pages = num_stages * 2  // double buffering

// 验证
total_cb_size = sum(cb_size for all CBs)
assert total_cb_size <= 1572864  // 1.5MB
assert num_cbs <= 64
```

## 实施步骤

### 1. 实现 PlanBlackholeCB Pass

**文件**: `src/transform/plan_blackhole_cb.h/cc`

```cpp
struct CBConfig {
    int cb_id;
    int num_pages;
    int page_size;
    int total_size;
    DataType dtype;
};

class PlanBlackholeCB : public tvm::tir::StmtExprMutator {
 public:
  // Main entry point
  tvm::tir::PrimFunc Transform(const tvm::tir::PrimFunc& func);

  // Get CB configuration
  std::vector<CBConfig> GetCBConfigs() const { return cb_configs_; }

  // Validate CB allocation
  bool Validate() const;

 private:
  // Analyze alloc_shared statements
  std::vector<CBConfig> AnalyzeAllocShared(const tvm::tir::PrimFunc& func);

  // Assign CB IDs
  void AssignCBIds(std::vector<CBConfig>& configs);

  // Store CB config in function attributes
  void StoreCBConfig(tvm::tir::PrimFunc& func, const std::vector<CBConfig>& configs);

  std::vector<CBConfig> cb_configs_;
  static constexpr int kMaxCBSize = 1572864;  // 1.5MB
  static constexpr int kMaxCBCount = 64;
};
```

### 2. 创建单元测试

**文件**: `tests/transform/test_plan_blackhole_cb.cc`

测试用例：
1. **SingleCB**: 单个 CB 分配
2. **MultipleCBs**: 多个 CB 分配
3. **DoubleBuffer**: 双缓冲配置
4. **OversizeError**: 超出 1.5MB 限制的错误处理
5. **TooManyCBs**: 超出 64 个 CB 的错误处理

## 验证方法

### 单元测试检查点

1. **CB ID 分配**:
   - CB ID 从 0 开始连续分配
   - 不重复，不越界

2. **大小计算**:
   - page_size = tile_rows * tile_cols * dtype_size
   - num_pages 符合预期

3. **验证**:
   - total_size <= 1.5MB
   - num_cbs <= 64

4. **属性存储**:
   - CB 配置正确存储在 function attributes

## 开发记录

### 2026-03-16

- **完成**: Pass 框架实现
- **实现**: PlanBlackholeCB 类，支持 Transform/Validate/CalculatePageSize
- **实现**: AllocSharedCollector Visitor 收集 shared buffer 分配
- **单元测试**: 7 个测试用例 (SingleCB, MultipleCBs, CBSizeValidation, CBCountValidation, PageSizeCalculation, OversizeErrorHandling, DoubleBuffering)

## 经验总结

- **设计决策**: CB ID 顺序分配，从 0 开始递增，简单且可预测
- **实现技巧**: 使用 static constexpr 定义约束常量 (kMaxCBSize, kMaxCBCount)，便于维护和测试
- **测试策略**: 边界测试很重要，特别是 1.5MB 边界和 64 CB 边界
- **踩坑记录**: 注意 page_size 计算时要考虑 dtype.bytes()，不同数据类型大小不同

## 预期产出

1. `src/transform/plan_blackhole_cb.h`
2. `src/transform/plan_blackhole_cb.cc`
3. `tests/transform/test_plan_blackhole_cb.cc`

## 参考

- `tasks/arch_design.md`: 内存映射设计
- Blackhole 规格: 1.5MB L1, 64 CBs
