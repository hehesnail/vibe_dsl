# TileLang C++ 核心：归约与原子操作模块

本文档详细分析 TileLang 项目中归约操作(Reduction)和原子操作(Atomic Operations)的 C++ 核心实现。

## 1. 模块概述

归约与原子操作模块是 TileLang 中用于高性能 GPU 计算的核心组件，主要包括：

- **归约操作 (`reduce.cc/h`)**: 实现张量归约运算（sum、max、min 等）
- **原子加法 (`atomic_add.cc`)**: 实现原子加法操作，支持 TMA 和普通模式
- **原子归约 (`atomic_reduce.cc`)**: 实现原子最大值/最小值操作
- **并行处理 (`parallel.cc`)**: 提供并行循环和布局推断支持

这些模块协同工作，为 TileLang 提供了高效的 GPU 归约和原子操作能力。

## 2. 归约操作实现详解

### 2.1 归约类型定义 (`src/op/reduce.h:18-79`)

```cpp
enum class ReduceTypeEnum : uint8_t {
  kSum,    // 求和归约
  kAbsSum, // 绝对值求和
  kMax,    // 最大值
  kMin,    // 最小值
  kAbsMax, // 最大绝对值
  kBitAnd, // 按位与
  kBitOr,  // 按位或
  kBitXor, // 按位异或
};
```

归约类型通过字符串构造：
- `"sum"` -> `kSum`
- `"abssum"` -> `kAbsSum`
- `"max"` -> `kMax`
- `"min"` -> `kMin`
- `"absmax"` -> `kAbsMax`
- `"bitand"` -> `kBitAnd`
- `"bitor"` -> `kBitOr`
- `"bitxor"` -> `kBitXor`

### 2.2 ReduceOpNode 核心结构 (`src/op/reduce.h:82-121`)

```cpp
class ReduceOpNode : public TileOperatorNode {
public:
  tir::Buffer src, dst;        // 源和目标缓冲区
  BufferRegion srcRegion_, dstRegion_;  // 缓冲区区域
  int dim;                     // 归约维度
  ReduceType type;             // 归约类型
  bool clear;                  // 是否清空目标

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T, InferLevel level) const override;

private:
  PrimExpr MakeInitValue() const;        // 生成初始值
  PrimExpr MakeReduce(const PrimExpr &acc, const PrimExpr &b) const;  // 生成归约表达式
  std::string MakeCodegenReducer() const; // 生成代码生成器名称
};
```

### 2.3 初始值生成 (`src/op/reduce.cc:57-102`)

每种归约类型有对应的初始值：

| 归约类型 | 整数初始值 | 浮点数初始值 |
|---------|-----------|-------------|
| Sum/AbsSum | 0 | 0 |
| Max | 最小整数 | -INFINITY |
| Min | 最大整数 | +INFINITY |
| AbsMax | 0 | 0 |
| BitAnd | -1 (全1) | - |
| BitOr/BitXor | 0 | - |

代码实现：`src/op/reduce.cc:57-102`

```cpp
PrimExpr ReduceOpNode::MakeInitValue() const {
  if (type->isSum()) {
    return make_zero(dst->dtype);
  } else if (type->isMax()) {
    if (is_int) {
      return make_const(dst->dtype, -(1 << (bits - 1)));
    } else {
      return make_const(dst->dtype, -INFINITY);
    }
  }
  // ... 其他类型
}
```

### 2.4 归约表达式生成 (`src/op/reduce.cc:104-129`)

```cpp
PrimExpr ReduceOpNode::MakeReduce(const PrimExpr &acc, const PrimExpr &b) const {
  if (type->isSum()) {
    return acc + rhs;
  } else if (type->isAbsSum()) {
    return acc + Max(rhs, -rhs);
  } else if (type->isMax()) {
    return Max(acc, rhs);
  } else if (type->isMin()) {
    return Min(acc, rhs);
  } else if (type->isAbsMax()) {
    return Max(acc, tvm::abs(rhs));
  } else if (type->isBitAnd()) {
    return acc & rhs;
  }
  // ... 其他类型
}
```

### 2.5 核心 Lower 实现 (`src/op/reduce.cc:224-494`)

归约操作的 Lower 过程包含以下步骤：

1. **布局计算** (`src/op/reduce.cc:240`): 使用 `ComputeReducerLayout` 计算归约后的布局
2. **初始化** (`src/op/reduce.cc:326-331`): 如果需要，写入初始值
3. **线程本地归约** (`src/op/reduce.cc:333-355`): 展开内层循环进行本地归约
4. **线程间归约** (`src/op/reduce.cc:357-410`): 使用 `AllReduce` 进行跨线程归约
5. **结果写回** (`src/op/reduce.cc:438-467`): 将结果写回目标缓冲区

#### AllReduce 调用生成 (`src/op/reduce.cc:381-394`)

```cpp
if (TargetHasSMVersionGE(T.target, 90)) {
  // Hopper 架构使用 NamedBarrier
  ss << "tl::AllReduce<" << this->MakeCodegenReducer() << ", "
     << reducing_threads << ", " << (*scale) << ", " << thread_offset
     << ", tl::NamedBarrier<" << all_threads << ">>::run";
} else {
  // 其他架构使用 SyncThreadsBarrier
  ss << "tl::AllReduce<" << this->MakeCodegenReducer() << ", "
     << reducing_threads << ", " << (*scale) << ", " << thread_offset
     << ">>::run";
}
```

### 2.6 CumSumOp 累积和操作 (`src/op/reduce.cc:550-618`)

累积和操作支持 1D 和 2D 缓冲区：

```cpp
CumSumOp::CumSumOp(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  // args[0]: src buffer
  // args[1]: dst buffer
  // args[2]: dim (归约维度)
  // args[3]: reverse (是否反向累积)
}
```

Lower 实现调用 `tl::CumSum1D` 或 `tl::CumSum2D` 运行时函数。

## 3. 原子操作实现详解

### 3.1 原子加法 AtomicAdd (`src/op/atomic_add.cc`)

#### 构造函数 (`src/op/atomic_add.cc:43-64`)

```cpp
AtomicAdd::AtomicAdd(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  // args[0]: 源（Buffer 或标量值）
  // args[1]: 目标 Buffer

  if (IsBufferLikeExpr(args[0])) {
    auto region = NormalizeToBufferRegion(args[0]);
    node->src = region->buffer;
    node->src_range = region->region;
  } else {
    node->src_value = args[0];  // 标量值
  }

  auto region = NormalizeToBufferRegion(args[1]);
  node->dst = region->buffer;
  node->dst_range = region->region;
}
```

#### 向量化长度 (`src/op/atomic_add.cc:96-106`)

```cpp
int AtomicAddNode::GetVectorizeLength(Target target) const {
  DataType dtype = dst->dtype;
  if (dtype.is_float16() || dtype.is_bfloat16()) {
    return 2;  // FP16/BF16 使用 2 元素向量化
  }
  if (dtype.is_float() && dtype.bits() == 32 &&
      TargetHasSMVersionGE(target, 90)) {
    return 4;  // SM90+ FP32 使用 4 元素向量化
  }
  return 1;
}
```

#### SIMT 循环生成 (`src/op/atomic_add.cc:149-221`)

```cpp
For AtomicAddNode::MakeSIMTLoop(arith::Analyzer *analyzer) const {
  // 1. 创建迭代变量
  Array<IterVar> loop_vars = MakeIterVars();

  // 2. 计算索引
  Array<PrimExpr> dst_indices = MakeIndices(loop_vars, 1);

  // 3. 加载源值
  PrimExpr src_value_arg = BufferLoad(src, src_indices);

  // 4. 类型转换
  if (src_value_arg->dtype != dst->dtype)
    src_value_arg = Cast(dst->dtype, src_value_arg);

  // 5. 构建访问指针
  PrimExpr dst_ptr = Call(DataType::Handle(), tl::access_ptr(),
                         {BufferLoad(dst, dst_indices), ...});

  // 6. 生成原子调用
  Call atomicadd_call = tvm::tir::Call(dst->dtype, atomic_add_elem_op(),
                                       new_args, annotations);

  // 7. 包装并行循环
  body = For(loop_vars[i]->var, 0, loop_vars[i]->dom->extent,
             ForKind::kParallel, body, ...);
}
```

#### TMA 原子加法 Lower (`src/op/atomic_add.cc:361-580`)

TMA (Tensor Memory Accelerator) 原子加法使用 `cp.reduce.async.bulk.tensor` 指令：

```cpp
Stmt AtomicAddNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  if (GetUseTMA()) {
    // 1. 构建 TMA 描述符
    TMADesc desc;
    desc.rank = global_tensor->shape.size();
    desc.data_type = to_CUtensorMapDataType(global_tensor->dtype);
    desc.global_addr = global_tensor->data;
    desc.global_shape = ReverseArray(global_tensor->shape);

    // 2. 检测 Swizzle 布局
    if (StructuralEqual()(shared_layout, makeQuarterBankSwizzleLayout(...))) {
      desc.swizzle = CU_TENSOR_MAP_SWIZZLE_32B;
    } else if (...) {
      desc.swizzle = CU_TENSOR_MAP_SWIZZLE_64B;
    }

    // 3. 生成 TMA reduce 调用
    Call create_descriptor = Call(DataType::Handle(), create_tma_descriptor(),
                                  desc.EncodeCallArgs());

    // 4. 生成存储操作
    tma_reduce = Evaluate(Call(DataType::Handle(), tma_store(), args, ...));

    // 5. 添加同步
    seq.push_back(Evaluate(Call(DataType::Handle(), tma_store_arrive(), {})));
    seq.push_back(Evaluate(Call(DataType::Handle(), tma_store_wait(), {})));
  }
}
```

### 3.2 原子归约 AtomicMax/AtomicMin (`src/op/atomic_reduce.cc`)

#### AtomicMax 实现 (`src/op/atomic_reduce.cc:29-59`)

```cpp
AtomicMax::AtomicMax(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  ICHECK(args.size() >= 2)
      << "AtomicMax expects at least 2 arguments (src, dst), got "
      << args.size();

  if (IsBufferLikeExpr(args[0])) {
    auto region = NormalizeToBufferRegion(args[0]);
    node->src = region->buffer;
    node->src_range = region->region;
  } else {
    node->src_value = args[0];  // 标量值
  }

  auto region = NormalizeToBufferRegion(args[1]);
  node->dst = region->buffer;
  node->dst_range = region->region;
}
```

#### 基类 AtomicOpBaseNode (`src/op/atomic_reduce.cc:101-286`)

提供原子操作的通用功能：

**迭代变量生成** (`src/op/atomic_reduce.cc:101-122`):
```cpp
Array<IterVar> AtomicOpBaseNode::MakeIterVars() const {
  // 根据目标范围生成迭代变量
  for (size_t i = 0; i < dst_range.size(); i++) {
    if (is_one(dst_range[i]->extent))
      continue;
    Var var = Var(std::string{char('i' + idx)}, dst_range[i]->extent->dtype);
    loop_vars.push_back({Range(0, dst_range[i]->extent), var, IterVarType::kDataPar});
  }
  // 标量情况创建虚拟循环变量
  if (loop_vars.empty()) {
    Var var = Var("i");
    loop_vars.push_back({Range(0, 1), var, IterVarType::kDataPar});
  }
}
```

**SIMT 循环生成** (`src/op/atomic_reduce.cc:176-242`):
```cpp
For AtomicOpBaseNode::MakeSIMTLoop(arith::Analyzer *analyzer) const {
  // 1. 生成迭代变量
  Array<IterVar> loop_vars = MakeIterVars();

  // 2. 计算索引
  Array<PrimExpr> dst_indices = MakeIndices(loop_vars, 1);

  // 3. 加载/使用源值
  if (!src_value.defined()) {
    Array<PrimExpr> src_indices = MakeIndices(loop_vars, 0);
    src_value_arg = BufferLoad(src, src_indices);
  } else {
    src_value_arg = src_value;
  }

  // 4. 构建访问指针
  PrimExpr dst_ptr = Call(DataType::Handle(), tl::access_ptr(),
                         {BufferLoad(dst, dst_indices), ...});

  // 5. 生成原子调用
  Call atomic_call = tvm::tir::Call(dst->dtype, GetElemOp(), new_args, annotations);

  // 6. 包装并行循环
  body = For(loop_vars[i]->var, 0, loop_vars[i]->dom->extent,
             ForKind::kParallel, body, ...);
}
```

## 4. 并行处理详解 (`src/op/parallel.cc`)

### 4.1 ParallelOpNode 概述

`ParallelOpNode` 是 TileLang 中处理并行循环的核心类，负责：
- 布局推断 (Layout Inference)
- 循环分区 (Loop Partitioning)
- 向量化 (Vectorization)

### 4.2 布局推断等级 (`src/op/parallel.cc:236-253`)

```cpp
/*! \brief Infer the layout for parallel operations based on different inference
 * levels
 *
 * - kStrict (2): 最保守级别，只允许显式定义的布局
 * - kCommon (1): 中间级别，允许常见布局模式
 * - kFree (0): 最自由级别，允许最大优化自由度
 */
```

### 4.3 InferLayout 核心逻辑 (`src/op/parallel.cc:254-494`)

```cpp
LayoutMap ParallelOpNode::InferLayout(const LayoutInferArgs &T,
                                      InferLevel level) const {
  // 1. 严格模式：推断完全复制的缓冲区
  if (level == InferLevel::kStrict) {
    // 处理 fragment[0] 访问模式
  }

  // 2. 确定是否允许布局传播
  bool allow_layout_propgate = ...;

  // 3. 从源缓冲区推断循环布局
  if (!loop_layout_.defined() && source_buffer.defined() &&
      allow_layout_propgate) {
    loop_layout_ = ComputeLoopLayoutFromBuffer(source_buffer, T);
  }

  // 4. 自由推断模式
  if (!loop_layout_.defined() && level == InferLevel::kFree) {
    // 尝试两种机制并选择最佳
    candidate_from_buffer = ComputeLoopLayoutFromBuffer(read_source_buffer, T);
    candidate_from_plan = ComputePlanCandidate(T);
    loop_layout_ = ChooseBestCandidate(candidate_from_buffer,
                                       candidate_from_plan, T);
  }

  // 5. 验证候选布局
  ValidateCandidateAgainstFragments(loop_layout_, T, ...);

  // 6. 构建复制保护
  BuildReplicationGuardsIfNeeded(...);

  // 7. 收集缓冲区片段
  for (const auto &[buffer, access] : indice_map_) {
    auto dst_layout = CompleteBufferFragment(buffer);
    results.Set(buffer, dst_layout);
  }
}
```

### 4.4 缓冲区片段完成 (`src/op/parallel.cc:504-556`)

```cpp
Fragment ParallelOpNode::CompleteBufferFragment(const Buffer &buffer) const {
  if (IsCommonAccessIndice(buffer)) {
    return loop_layout_;  // 直接使用循环布局
  }

  // 尝试双射映射检测
  auto res2d = arith::DetectIterMap(...);
  if (res2d->errors.empty()) {
    // 使用逆布局
    Layout ind_inv2d = Layout(loop_vars_, GetAccessInfo(buffer).indices)->Inverse();
    return Fragment(buffer->shape, {}, thd_b2, dest_buffer_rep_extent, ...);
  }

  // 否则使用展平表达式
  PrimExpr rep_b = MakeFlattenedExpression(...);
  // ...
}
```

### 4.5 候选选择策略 (`src/op/parallel.cc:756-817`)

```cpp
Fragment ParallelOpNode::ChooseBestCandidate(
    const Fragment &candidate_from_buffer,
    const Fragment &candidate_from_plan,
    const LayoutInferArgs &T) const {
  // 1. 验证每个候选
  bool buf_ok = ValidateCandidateAgainstFragments(candidate_from_buffer, T);
  bool plan_ok = ValidateCandidateAgainstFragments(candidate_from_plan, T);

  // 2. 只有一个有效时选择它
  if (buf_ok && !plan_ok) return candidate_from_buffer;
  if (plan_ok && !buf_ok) return candidate_from_plan;

  // 3. 都有效时比较包含关系
  bool buf_contains_plan = contains(candidate_from_buffer, candidate_from_plan);
  bool plan_contains_buf = contains(candidate_from_plan, candidate_from_buffer);

  if (buf_contains_plan && !plan_contains_buf) return candidate_from_plan;
  if (plan_contains_buf && !buf_contains_plan) return candidate_from_buffer;

  // 4. 比较复制范围
  if (analyzer_.CanProve(rep_plan <= rep_buf)) return candidate_from_plan;

  // 5. 默认回退
  return candidate_from_buffer;
}
```

## 5. 关键设计模式

### 5.1 操作符注册模式

所有操作符使用 `TIR_REGISTER_TL_TILE_OP` 宏注册：

```cpp
// src/op/reduce.cc:535-538
TIR_REGISTER_TL_TILE_OP(ReduceOp, reduce)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// src/op/atomic_add.cc:605-608
TIR_REGISTER_TL_TILE_OP(AtomicAdd, atomicadd)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// src/op/atomic_reduce.cc:292-300
TIR_REGISTER_TL_TILE_OP(AtomicMax, atomicmax)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
```

### 5.2 Lower 方法模式

所有操作符实现统一的 Lower 接口：

```cpp
Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
```

`LowerArgs` 包含：
- `target`: 目标架构
- `thread_bounds`: 线程边界
- `layout_map`: 布局映射
- `thread_var`: 线程变量
- `buffer_remap`: 缓冲区重映射

### 5.3 布局推断模式

```cpp
LayoutMap InferLayout(const LayoutInferArgs &T, InferLevel level) const override;
```

推断等级：
- `kStrict`: 仅验证，不生成新布局
- `kCommon`: 允许常见布局模式
- `kFree`: 允许自由优化

## 6. 文件引用汇总

| 文件 | 行数 | 主要功能 |
|-----|------|---------|
| `src/op/reduce.h` | 178 | 归约操作类型定义和类声明 |
| `src/op/reduce.cc` | 668 | 归约操作实现（ReduceOp, CumSumOp） |
| `src/op/atomic_add.h` | - | 原子加法头文件 |
| `src/op/atomic_add.cc` | 614 | 原子加法实现 |
| `src/op/atomic_reduce.h` | - | 原子归约头文件 |
| `src/op/atomic_reduce.cc` | 309 | 原子最大/最小实现 |
| `src/op/parallel.h` | - | 并行操作头文件 |
| `src/op/parallel.cc` | 821 | 并行循环和布局推断 |
| `src/op/builtin.h` | 946 | 内置函数声明 |
| `src/op/builtin.cc` | 546 | 内置函数注册 |

## 7. 核心类继承关系

```
TileOperatorNode (基类)
    ├── ReduceOpNode
    ├── CumSumOpNode
    ├── AtomicAddNode
    ├── AtomicMaxNode
    ├── AtomicMinNode
    └── ParallelOpNode
```

每个节点类都实现：
- `Lower()`: 降级为 TIR 语句
- `InferLayout()`: 布局推断
- `Clone()`: 深拷贝
