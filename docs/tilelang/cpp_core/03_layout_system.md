# TileLang 布局系统 (Layout System)

## 1. 概述

TileLang 的布局系统是其核心组件之一，负责定义和管理张量数据在内存中的分布方式，以及线程与数据之间的映射关系。布局系统主要包含两个核心抽象：`Layout` 和 `Fragment`。

**核心文件位置：**
- `/root/dev/vibe_dsl/tilelang/src/layout/layout.h` - 布局系统头文件，定义核心类
- `/root/dev/vibe_dsl/tilelang/src/layout/layout.cc` - 布局系统实现
- `/root/dev/vibe_dsl/tilelang/src/layout/gemm_layouts.cc` - GEMM 专用布局实现
- `/root/dev/vibe_dsl/tilelang/src/layout/utils.h` 和 `utils.cc` - 布局工具函数
- `/root/dev/vibe_dsl/tilelang/tilelang/layout/` - Python 布局接口

## 2. 核心概念

### 2.1 Layout 类

`Layout` 是 TileLang 中最基础的布局抽象，定义了从逻辑索引到物理索引的映射关系。

```cpp
// src/layout/layout.h:45-113
class LayoutNode : public Object {
public:
  LayoutNode(Array<PrimExpr> input_size, Array<PrimExpr> forward_index);

  size_t InputDim() const { return input_size_.size(); }
  size_t OutputDim() const { return forward_index_.size(); }

  Array<PrimExpr> InputShape() const { return input_size_; }
  Array<PrimExpr> OutputShape() const;
  Array<PrimExpr> GetForwardIndex() const { return forward_index_; }

  // 核心操作
  virtual Array<PrimExpr> Forward(const Array<PrimExpr> &vars) const;
  virtual Layout Repeat(int dim, int factor) const;
  virtual Layout Expand(const Array<PrimExpr> &leading_shape) const;
  virtual Layout Inverse() const;
  virtual Layout Reshape(const Array<PrimExpr> &shape,
                         arith::Analyzer *analyzer = nullptr,
                         const PrimExpr rescale_num = Integer(1),
                         const PrimExpr rescale_den = Integer(1)) const;

protected:
  Array<PrimExpr> forward_index_;  // 前向索引映射表达式
  Array<PrimExpr> input_size_;     // 输入形状
};
```

**关键方法说明：**

- **`Forward`** (`src/layout/layout.cc:133-161`): 将输入变量通过前向索引表达式计算输出索引
- **`Repeat`** (`src/layout/layout.cc:163-201`): 沿单个输入维度重复布局，用于构建更大的布局
- **`Expand`** (`src/layout/layout.cc:203-243`): 通过前置新的输入维度来扩展布局
- **`Inverse`** (`src/layout/layout.cc:562-565`): 计算布局的逆变换，用于反向索引映射
- **`Reshape`** (`src/layout/layout.cc:386-470`): 重塑布局的输入形状，支持元素大小变化

### 2.2 Fragment 类

`Fragment` 继承自 `Layout`，专门用于描述线程级数据分布，是 GPU 张量核心编程的核心抽象。

```cpp
// src/layout/layout.h:126-184
class FragmentNode : public LayoutNode {
public:
  FragmentNode(Array<PrimExpr> input_size, Array<PrimExpr> forward_index,
               PrimExpr forward_thread, PrimExpr replicate_size);

  PrimExpr GetForwardThread() const { return forward_thread_; }
  PrimExpr ThreadExtent() const;
  PrimExpr ReplicateExtent() const { return replicate_size_; }

  // Fragment 特有操作
  PrimExpr ForwardThread(const Array<PrimExpr> &vars,
                         const Optional<PrimExpr> &rep_var) const;
  Fragment Repeat(const Array<PrimExpr> &repeats, bool repeat_on_thread,
                  bool lower_dim_first = true) const;
  Fragment Replicate(int repeats) const;
  Fragment DeReplicate() const;
  Fragment CondenseReplicateVar() const;
  Fragment BindThreadRange(Range thread_range) const;

  // 检查是否为完全复制布局
  bool IsCompletedReplicated() const;

protected:
  Range thread_range_;
  PrimExpr forward_thread_;    // 线程映射表达式
  PrimExpr replicate_size_;    // 复制因子
};
```

**Fragment 的核心特性：**

1. **线程映射** (`forward_thread_`): 定义逻辑索引如何映射到线程 ID
2. **复制机制** (`replicate_size_`): 支持数据在多个线程间的复制
3. **完全复制布局** (`IsCompletedReplicated`): 所有线程持有相同数据的完整副本

**创建完全复制布局：**
```cpp
// src/layout/layout.cc:638-643
Fragment Fragment::FullyReplicated(Array<PrimExpr> shape,
                                   PrimExpr thread_extent) {
  return Fragment(shape, {}, ReplicationPlaceholder(), thread_extent, std::nullopt)
      ->BindThreadRange(Range(0, thread_extent));
}
```

## 3. Swizzle 操作详解

Swizzle 是 TileLang 中用于优化共享内存访问模式的关键技术，通过重新排列数据布局来避免银行冲突。

### 3.1 Swizzle 模式

```cpp
// src/layout/layout.h:277-284
enum class SwizzleMode {
  kNone = 0,    // 非 swizzle 布局（线性或填充）
  kQuarter = 1, // 32B swizzle (CU_TENSOR_MAP_SWIZZLE_32B)
  kHalf = 2,    // 64B swizzle (CU_TENSOR_MAP_SWIZZLE_64B)
  kFull = 3     // 128B swizzle (CU_TENSOR_MAP_SWIZZLE_128B)
};
```

### 3.2 XOR Swizzle 实现

TileLang 使用 XOR 操作实现高效的 swizzle 模式：

```cpp
// src/layout/gemm_layouts.cc:362-380
PrimExpr xor2x2(const PrimExpr &i, const PrimExpr &j) {
  return FloorMod(i + j, 2);
}

PrimExpr xor4x4(const PrimExpr &i, const PrimExpr &j) {
  PrimExpr i0 = FloorMod(i, 2);
  PrimExpr j0 = FloorMod(j, 2);
  PrimExpr i1 = FloorDiv(i, 2);
  PrimExpr j1 = FloorDiv(j, 2);
  return 2 * xor2x2(i1, j1) + xor2x2(i0, j0);
}

PrimExpr xor8x8(const PrimExpr &i, const PrimExpr j) {
  PrimExpr i0 = FloorMod(i, 2);
  PrimExpr j0 = FloorMod(j, 2);
  PrimExpr i1 = FloorDiv(i, 2);
  PrimExpr j1 = FloorDiv(j, 2);
  return 2 * xor4x4(i1, j1) + xor2x2(i0, j0);
}
```

### 3.3 三级 Swizzle 布局实现

#### Quarter Bank Swizzle (32B)
```cpp
// src/layout/gemm_layouts.cc:418-436
static Layout MakeQuarterBankSwizzleLayout2D(int stride, int continuous,
                                             int element_size) {
  // Swizzle 1 bit
  Var i = InputPlaceholder(0);
  Var j = InputPlaceholder(1);
  int vector_size = 128 / element_size;

  PrimExpr ts = FloorDiv(i, 8);
  PrimExpr s = FloorMod(i, 8);
  PrimExpr tc = FloorDiv(FloorDiv(j, vector_size), 2);
  PrimExpr c = FloorMod(FloorDiv(j, vector_size), 2);
  PrimExpr vec = FloorMod(j, vector_size);

  // XOR swizzle: c_swizzle = c ^ (s / 4)
  PrimExpr c_swizzle = xor2x2(c, FloorDiv(s, 4));
  PrimExpr index = vec + (c_swizzle + s * 2) * vector_size;

  return Layout(Array<PrimExpr>{stride, continuous}, {tc, ts, index});
}
```

#### Half Bank Swizzle (64B)
```cpp
// src/layout/gemm_layouts.cc:451-469
static Layout MakeHalfBankSwizzleLayout2D(int stride, int continuous,
                                          int element_size) {
  // Swizzle 2 bit
  Var i = InputPlaceholder(0);
  Var j = InputPlaceholder(1);
  int vector_size = 128 / element_size;

  PrimExpr ts = FloorDiv(i, 8);
  PrimExpr s = FloorMod(i, 8);
  PrimExpr tc = FloorDiv(FloorDiv(j, vector_size), 4);
  PrimExpr c = FloorMod(FloorDiv(j, vector_size), 4);
  PrimExpr vec = FloorMod(j, vector_size);

  // XOR swizzle: c_swizzle = c ^ (s / 2)
  PrimExpr c_swizzle = xor4x4(c, FloorDiv(s, 2));
  PrimExpr index = vec + (c_swizzle + s * 4) * vector_size;

  return Layout(Array<PrimExpr>{stride, continuous}, {tc, ts, index});
}
```

#### Full Bank Swizzle (128B)
```cpp
// src/layout/gemm_layouts.cc:484-502
static Layout MakeFullBankSwizzleLayout2D(int stride, int continuous,
                                          int element_size) {
  // Swizzle 3 bit
  Var i = InputPlaceholder(0);
  Var j = InputPlaceholder(1);
  int vector_size = 128 / element_size;

  PrimExpr ts = FloorDiv(i, 8);
  PrimExpr s = FloorMod(i, 8);
  PrimExpr tc = FloorDiv(FloorDiv(j, vector_size), 8);
  PrimExpr c = FloorMod(FloorDiv(j, vector_size), 8);
  PrimExpr vec = FloorMod(j, vector_size);

  // XOR swizzle: c_swizzle = c ^ s
  PrimExpr c_swizzle = xor8x8(c, s);
  PrimExpr index = vec + (c_swizzle + s * 8) * vector_size;

  return Layout(Array<PrimExpr>{stride, continuous}, {tc, ts, index});
}
```

### 3.4 Swizzle 布局检测与合并

```cpp
// src/layout/gemm_layouts.cc:905-933
SwizzleMode DetectSwizzleMode(const Layout &layout, const Buffer &buffer) {
  SwizzleShapeInfo info;
  if (!TryGetSwizzleShapeInfo(buffer, &info)) {
    return SwizzleMode::kNone;
  }
  int vector_size = 128 / info.element_size;

  // 从小到大粒度检查
  if (info.stride % 8 == 0 &&
      info.continuous % (static_cast<int64_t>(vector_size) * 2) == 0) {
    if (StructuralEqual()(layout, makeQuarterBankSwizzleLayout(buffer))) {
      return SwizzleMode::kQuarter;
    }
  }
  // ... 类似检查 Half 和 Full
  return SwizzleMode::kNone;
}

// 合并两个 swizzle 布局，取较小粒度
Optional<Layout> MergeSwizzleLayouts(const Layout &layout1,
                                     const Layout &layout2,
                                     const Buffer &buffer) {
  SwizzleMode mode1 = DetectSwizzleMode(layout1, buffer);
  SwizzleMode mode2 = DetectSwizzleMode(layout2, buffer);

  if (mode1 == SwizzleMode::kNone || mode2 == SwizzleMode::kNone) {
    return std::nullopt;
  }

  // 取较小的 swizzle 粒度
  SwizzleMode min_mode = std::min(mode1, mode2);

  switch (min_mode) {
  case SwizzleMode::kQuarter:
    return makeQuarterBankSwizzleLayout(buffer);
  case SwizzleMode::kHalf:
    return makeHalfBankSwizzleLayout(buffer);
  case SwizzleMode::kFull:
    return makeFullBankSwizzleLayout(buffer);
  default:
    return std::nullopt;
  }
}
```

## 4. Fragment 布局详解

### 4.1 GEMM Fragment 基础布局

#### 8x8 Fragment (用于 ldmatrix/stmatrix)
```cpp
// src/layout/gemm_layouts.cc:31-38
Fragment makeGemmFragment8x8() {
  IterVar i = make_itervar("i", 8);
  IterVar j = make_itervar("j", 8);
  IterVar rep = make_itervar("rep", 1);

  // 线程映射: thread = (j / 2) + 4 * i
  PrimExpr forward_thread = FloorDiv(j->var, 2) + 4 * i;
  // 值索引: index = j % 2
  PrimExpr index = FloorMod(j->var, 2);

  return Fragment({i, j}, {index}, forward_thread, rep);
}
```

#### 转置 8x8 Fragment
```cpp
// src/layout/gemm_layouts.cc:49-56
Fragment makeGemmFragment8x8Transposed() {
  IterVar i = make_itervar("i", 8);
  IterVar j = make_itervar("j", 8);
  IterVar rep = make_itervar("rep", 1);

  // 转置后的线程映射
  PrimExpr forward_thread = FloorDiv(i->var, 2) + 4 * j;
  PrimExpr index = FloorMod(i->var, 2);

  return Fragment({i, j}, {index}, forward_thread, rep);
}
```

### 4.2 矩阵 C 的 Fragment 布局

```cpp
// src/layout/gemm_layouts.cc:122-137
Fragment makeGemmFragmentC(const int block_m, const int block_n,
                           const int warp_m, const int warp_n,
                           const int element_size) {
  if (element_size == 64)
    return makeGemmFragmentC_F64(block_m, block_n, warp_m, warp_n);

  ICHECK(block_m % warp_m == 0);
  ICHECK(block_n % warp_n == 0);
  ICHECK(warp_m % 16 == 0);
  ICHECK(warp_n % 8 == 0);

  // 基础 8x8 布局，先在 M 维度重复 2 次
  auto base_layout = makeGemmFragment8x8()->Repeat({2, 1}, false);
  // Warp 级布局，在 thread 维度重复
  auto warp_layout =
      base_layout->Repeat({block_m / warp_m, block_n / warp_n}, true, false);
  // Block 级布局
  auto block_layout =
      warp_layout->Repeat({warp_m / 16, warp_n / 8}, false, false);
  return block_layout;
}
```

### 4.3 AMD CDNA 架构 Fragment

```cpp
// src/layout/gemm_layouts.cc:99-106
Fragment makeGemmFragmentC16x16CDNA() {
  IterVar i = make_itervar("i", 16);
  IterVar j = make_itervar("j", 16);
  IterVar rep = make_itervar("rep", 1);

  // CDNA 架构使用不同的线程映射
  PrimExpr forward_thread = 16 * FloorDiv(j->var, 4) + i;
  PrimExpr index = FloorMod(j->var, 4);

  return Fragment({i, j}, {index}, forward_thread, rep);
}
```

### 4.4 Fragment 的 Repeat 操作

```cpp
// src/layout/layout.cc:245-289
Fragment FragmentNode::Repeat(const Array<PrimExpr> &repeats,
                              bool repeat_on_thread,
                              bool lower_dim_first) const {
  // 计算新的输入大小
  Array<PrimExpr> new_input_size;
  Map<Var, PrimExpr> vmap;
  for (size_t i = 0; i < InputDim(); i++) {
    new_input_size.push_back(input_size_[i] * repeats[i]);
    vmap.Set(InputPlaceholder(i),
             FloorMod(InputPlaceholder(i), InputShape()[i]));
  }

  // 计算重复索引
  PrimExpr repeats_index = 0, repeat_stride = 1;
  if (lower_dim_first) {
    for (int i = InputDim() - 1; i >= 0; i--) {
      repeats_index +=
          repeat_stride * FloorDiv(InputPlaceholder(i), InputShape()[i]);
      repeat_stride *= repeats[i];
    }
  } else {
    for (size_t i = 0; i < InputDim(); i++) {
      repeats_index +=
          repeat_stride * FloorDiv(InputPlaceholder(i), InputShape()[i]);
      repeat_stride *= repeats[i];
    }
  }

  if (repeat_on_thread) {
    // 在线程维度上重复
    PrimExpr thread_size = ThreadExtent();
    auto new_forward_index = forward_index_.Map(
        [&](const PrimExpr &e) { return Substitute(e, vmap); });
    auto new_forward_thread =
        Substitute(forward_thread_, vmap) + thread_size * repeats_index;
    return Fragment(new_input_size, new_forward_index, new_forward_thread,
                    replicate_size_, std::nullopt);
  } else {
    // 在值索引维度上重复
    ICHECK(OutputDim() == 1);
    PrimExpr frag_len = OutputShape()[0];
    Array<PrimExpr> new_forward_index = {Substitute(forward_index_[0], vmap) +
                                         frag_len * repeats_index};
    PrimExpr new_forward_thread = Substitute(forward_thread_, vmap);
    return Fragment(new_input_size, new_forward_index, new_forward_thread,
                    replicate_size_, std::nullopt);
  }
}
```

## 5. Python 布局接口

TileLang 提供了 Python 层的布局接口，便于用户定义自定义布局。

### 5.1 Layout 类

```python
# tilelang/layout/layout.py:11-255
@tvm_ffi.register_object("tl.Layout")
class Layout(Node):
    def __init__(self, shape, forward_fn):
        """
        初始化 Layout 对象

        Parameters
        ----------
        shape : list of int
            布局的形状，定义每个维度的元素数量
        forward_fn : function
            将索引变量映射到计算的前向索引的函数
        """
        forward_vars = []
        for idx, size in enumerate(shape):
            iv = IterVar(Range(0, size), Var(f"i{idx}", "int32"), 0)
            forward_vars.append(iv)

        vars = [iv.var for iv in forward_vars]
        forward_index = forward_fn(*vars)

        if isinstance(forward_index, PrimExpr):
            forward_index = [forward_index]

        self.__init_handle_by_constructor__(_ffi_api.Layout, forward_vars, forward_index)
```

### 5.2 Fragment 类

```python
# tilelang/layout/fragment.py:12-206
@tvm_ffi.register_object("tl.Fragment")
class Fragment(Layout):
    def __init__(self, shape, forward_fn=None, forward_thread_fn=None,
                 replicate=1, forward_index_fn=None):
        """
        初始化 Fragment 对象

        Parameters
        ----------
        shape : list[int]
            Fragment 的每个维度大小
        forward_fn : callable, optional
            计算 (forward_thread, forward_index) 的函数
        forward_thread_fn : callable, optional
            计算线程映射的函数
        replicate : int, optional
            线程复制因子，默认为 1
        forward_index_fn : callable, optional
            计算索引映射的函数
        """
        forward_vars = []
        for idx, size in enumerate(shape):
            iv = IterVar(Range(0, size), Var(f"i{idx}", "int32"), 0)
            forward_vars.append(iv)

        vars = [iv.var for iv in forward_vars]

        if forward_fn is not None:
            if replicate > 1:
                thread_replicate = IterVar(Range(0, replicate), Var("rep", "int32"), 0)
                forward_thread, forward_index = forward_fn(*vars, thread_replicate)
            else:
                forward_thread, forward_index = forward_fn(*vars)
        # ... 其他初始化逻辑
```

### 5.3 Swizzle 布局辅助函数

```python
# tilelang/layout/swizzle.py:66-78
def make_swizzled_layout(buffer: BufferLikeType, k_major: bool = True, allow_pad: bool = True):
    """
    创建 swizzle 布局以优化内存访问模式

    Args:
        buffer: 输入 buffer
        k_major: K 维度是否为主要维度
        allow_pad: 是否允许填充
    """
    _, shape, _ = _get_buffer_info(buffer)
    stride, continuous = _get_stride_continuous(buffer)
    element_size = _get_element_size(buffer)
    base = _ffi_api.make_swizzled_layout(
        stride, continuous, element_size, k_major, allow_pad
    )
    return base.reshape(shape)

# 不同架构的 swizzle 布局
def make_volta_swizzled_layout(buffer, is_a=True, k_inner=True):
    """Volta 架构的 swizzle 布局"""

def make_wgmma_swizzled_layout(buffer, continuity=None, k_major=True):
    """Hopper 架构 WGMMA 的 swizzle 布局"""

def make_tcgen05mma_swizzled_layout(buffer, continuity=None, k_major=True):
    """Blackwell 架构 TCGEN05 MMA 的 swizzle 布局"""
```

## 6. 布局推断系统

TileLang 使用布局推断系统自动推导张量的最优布局。

```cpp
// src/transform/layout_inference.cc:61-65
struct LayoutInferenceResult {
  Map<Buffer, Layout> layout_map;      // Buffer 到布局的映射
  Map<For, Fragment> for_map;          // For 循环到 Fragment 的映射
  Map<For, PrimExpr> predicate_map;    // 谓词映射
};
```

布局推断的核心流程：

1. **收集 Buffer 使用定义** (`BufferUseDefCollector`)
2. **执行推断步骤** (`RunInferStep`): 使用 BFS 遍历操作符图
3. **处理布局冲突**: 尝试合并 swizzle 布局或报错
4. **传播布局到别名 Buffer**: 确保数据一致性

```cpp
// src/transform/layout_inference.cc:172-230
// 布局冲突处理逻辑
if (layout_map.count(buffer)) {
  // 检查新布局是否包含旧布局（Fragment 情况）
  if (IsFragmentBuffer(buffer) && level != InferLevel::kStrict) {
    if (ProveFragmentContains(src_layout, dst_layout, indices, indices,
                              inner_analyzer)) {
      layout_map.Set(buffer, layout);
      continue;
    }
  }

  // 尝试合并 swizzle 布局
  if (!layout.as<Fragment>() && !existing.as<Fragment>()) {
    if (auto merged = MergeSwizzleLayouts(existing, layout, buffer)) {
      LOG(WARNING) << "Swizzle layout conflict for buffer " << buffer
                   << ", merging to smaller granularity";
      layout_map.Set(buffer, merged.value());
      continue;
    }
  }

  // 布局冲突报错
  LOG(FATAL) << "Get different layout for " << buffer;
}
```

## 7. 工具函数

### 7.1 迭代器工具

```cpp
// src/layout/utils.h:35-38
Array<arith::IterSplitExpr>
DivideUnusedIterators(const Array<PrimExpr> &exprs,
                      const Array<IterVar> input_iters,
                      arith::Analyzer *analyzer);

// src/layout/utils.h:46-49
std::pair<PrimExpr, IterVar> CompressIterator(const PrimExpr &expr,
                                              const Array<IterVar> input_iters,
                                              const Var &var,
                                              arith::Analyzer *analyzer);
```

### 7.2 Fragment 包含性证明

```cpp
// src/layout/utils.h:86-90
bool ProveFragmentContains(Fragment small_frag, Fragment large_frag,
                           Array<PrimExpr> small_frag_indices,
                           Array<PrimExpr> large_frag_indices,
                           arith::Analyzer &analyzer,
                           bool check_forward_index = false);
```

该函数验证小 fragment 的线程是否是大 fragment 线程的子集，用于确保数据访问的有效性。

## 8. 属性标记

TileLang 使用以下属性标记来传递布局信息：

```cpp
// src/layout/layout.h:295-304
namespace attr {
// Block 属性，包含所有 buffer 的布局映射
constexpr const char *kLayoutMap = "layout_map";
// For 属性，包含并行循环的布局
constexpr const char *kParallelLoopLayout = "parallel_loop_layout";
// For 属性，包含并行循环的谓词
constexpr const char *kParallelLoopPredicate = "parallel_loop_predicate";
// For 属性，合并内存访问的宽度
constexpr const char *kCoalescedWidth = "coalesced_width";
}
```

## 9. 总结

TileLang 的布局系统是一个强大而灵活的框架，具有以下特点：

1. **双层次抽象**: `Layout` 用于通用内存布局，`Fragment` 用于线程级数据分布
2. **高效的 Swizzle**: 通过 XOR 操作实现无银行冲突的共享内存访问
3. **自动布局推断**: 基于数据流分析自动推导最优布局
4. **跨架构支持**: 支持 NVIDIA (CUDA) 和 AMD (HIP) 的不同张量核心布局
5. **Python/C++ 统一接口**: 提供一致的编程接口
