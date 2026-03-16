# Phase 2: SplitBlackholeKernel Pass

> ⚠️ **2026-03-17 设计审查更新**：本任务已降级为 Phase 4 可选优化项。
> - **实际状态**：Stub（`Transform()` 返回原函数）
> - **降级原因**：TT-Sim 不支持 NCRISC NOC write，拆分后 Writer kernel 无法测试；合并 R/C/W 到 BRISC 单核已验证可行
> - **详见**：[design_review.md](../design_review.md) 问题8、第4.6节

## 任务目标

将统一的 PrimFunc 拆分为 Reader、Compute、Writer 三个独立的 kernel，并在其间插入 CB 同步操作。

## 背景

Blackhole 架构有不同类型的核心：
- **BRISC**: 控制核心，负责从 DRAM 读取数据到 CB
- **TRISC**: 计算核心，负责在 CB 上执行计算
- **NCRISC**: 数据移动核心，负责将 CB 数据写回 DRAM

TileLang DSL 的 kernel 是统一编写的，需要通过此 Pass 拆分为三个独立的 kernel。

## 技术方案

### 输入输出

**输入**:
```cpp
// Unified PrimFunc (before split)
void kernel_main() {
    // DRAM -> CB (Reader)
    cb_reserve_back(cb_id, 1);
    noc_async_read(src_addr, cb_addr, size);
    noc_async_read_barrier();
    cb_push_back(cb_id, 1);

    // Compute on CB (Compute)
    cb_wait_front(cb_id, 1);
    // compute...
    cb_pop_front(cb_id, 1);

    // CB -> DRAM (Writer)
    cb_reserve_back(cb_id, 1);
    noc_async_write(cb_addr, dst_addr, size);
    noc_async_write_barrier();
    cb_push_back(cb_id, 1);
}
```

**输出**:
```cpp
// Reader Kernel (BRISC)
void reader_kernel() {
    cb_reserve_back(cb_id, 1);
    noc_async_read(src_addr, cb_addr, size);
    noc_async_read_barrier();
    cb_push_back(cb_id, 1);
}

// Compute Kernel (TRISC)
void compute_kernel() {
    cb_wait_front(cb_id, 1);
    // compute...
    cb_pop_front(cb_id, 1);
}

// Writer Kernel (NCRISC)
void writer_kernel() {
    cb_wait_front(cb_id, 1);  // Wait for compute to produce data
    noc_async_write(cb_addr, dst_addr, size);
    noc_async_write_barrier();
    cb_pop_front(cb_id, 1);
}
```

### 拆分策略

1. **分析数据流**: 识别 `T.copy(DRAM→CB)`, `T.compute`, `T.copy(CB→DRAM)` 模式
2. **插入同步**: 在 kernel 之间插入 CB 同步原语
3. **生成独立函数**: 创建三个独立的 PrimFunc

### CB 同步原语

```cpp
// Reader 到 Compute 同步
Reader:  cb_reserve_back(cb_id, num_tiles) → noc_async_read → cb_push_back(cb_id, num_tiles)
Compute: cb_wait_front(cb_id, num_tiles) → compute → cb_pop_front(cb_id, num_tiles)

// Compute 到 Writer 同步
Compute: cb_reserve_back(cb_id, num_tiles) → store result → cb_push_back(cb_id, num_tiles)
Writer:  cb_wait_front(cb_id, num_tiles) → noc_async_write → cb_pop_front(cb_id, num_tiles)
```

## 实施步骤

### 1. 实现 SplitBlackholeKernel Pass

**文件**: `src/transform/split_blackhole_kernel.h/cc`

```cpp
class SplitBlackholeKernel : public tvm::tir::StmtExprMutator {
 public:
  // Main entry point
  tvm::tir::PrimFunc Transform(const tvm::tir::PrimFunc& func);

  // Generate reader kernel
  tvm::tir::PrimFunc GenerateReaderKernel(const tvm::tir::PrimFunc& func);

  // Generate compute kernel
  tvm::tir::PrimFunc GenerateComputeKernel(const tvm::tir::PrimFunc& func);

  // Generate writer kernel
  tvm::tir::PrimFunc GenerateWriterKernel(const tvm::tir::PrimFunc& func);

 private:
  // Analyze dataflow pattern
  struct DataFlowPattern {
    std::vector<Stmt> reader_stmts;
    std::vector<Stmt> compute_stmts;
    std::vector<Stmt> writer_stmts;
  };
  DataFlowPattern AnalyzeDataFlow(const tvm::tir::PrimFunc& func);

  // Insert CB synchronization
  Stmt InsertCBSync(const Stmt& stmt, int cb_id, const std::string& sync_type);
};
```

### 2. 创建单元测试

**文件**: `tests/transform/test_split_blackhole_kernel.cc`

测试用例：
1. **SimpleCopy**: 简单 Copy 算子拆分
2. **GEMM**: 矩阵乘法拆分 (Reader + Compute + Writer)
3. **MultiStage**: 多阶段流水线拆分

## 验证方法

### 单元测试检查点

1. **拆分正确性**:
   - Reader kernel 包含 DRAM→CB 操作
   - Compute kernel 包含计算操作
   - Writer kernel 包含 CB→DRAM 操作

2. **同步正确性**:
   - Reader 有 `cb_push_back`
   - Compute 有 `cb_wait_front` 和 `cb_pop_front`
   - Writer 有 `cb_wait_front`

3. **独立性**:
   - 三个 kernel 之间没有共享变量
   - 通过 CB 同步解耦

## 预期产出

1. `src/transform/split_blackhole_kernel.h`
2. `src/transform/split_blackhole_kernel.cc`
3. `tests/transform/test_split_blackhole_kernel.cc`

## 开发记录

### 2026-03-16

- **完成**: Pass 框架实现
- **实现**: SplitBlackholeKernel 类，支持 Transform/GenerateReaderKernel/GenerateComputeKernel/GenerateWriterKernel
- **实现**: DataFlowAnalysis 分析数据流，识别 read/compute/write 操作
- **单元测试**: 5 个测试用例 (SimpleCopySplit, GEMMSplit, CBSyncInsertion, KernelIndependence, MultiStagePipeline)

## 经验总结

- **设计决策**: 使用 StmtExprVisitor 分析数据流，而不是基于 pattern matching，更灵活且易于扩展
- **实现技巧**: Pass 返回 SplitResult 结构体，包含三个独立的 PrimFunc，方便后续处理
- **测试策略**: Mock 实现不依赖 TVM，专注于算法逻辑验证
- **踩坑记录**: SplitResult 中的 PrimFunc 需要正确处理 defined() 检查，避免空指针访问

## 参考

- `codegen_blackhole.h`: CodeGen 生成的 kernel 结构
- `tasks/arch_design.md`: 架构设计中的 lowering pipeline
