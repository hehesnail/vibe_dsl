# TileLang C++ 核心 - 拷贝与内存操作

## 模块概述

拷贝操作模块 (`src/op/copy.cc` 和 `src/op/copy.h`) 实现了 TileLang 中各种内存传输策略，包括普通拷贝、批量/TMA 拷贝、LDSM/STSM 矩阵拷贝以及 Tensor Memory 操作。该模块是 TileLang 与底层 GPU 硬件交互的核心。

---

## 拷贝指令类型

**文件**: `/root/dev/vibe_dsl/tilelang/src/op/copy.h:17-30`

```cpp
enum class CopyInst : uint8_t {
  kNormal = 0,     // 普通缓冲区拷贝
  kLDSM = 1,       // ldmatrix 内存拷贝 (Shared -> Fragment)
  kSTSM = 2,       // stmatrix 内存拷贝 (Fragment -> Shared)
  kBulkLoad = 3,   // TMA 批量加载 (Global -> Shared)
  kBulkStore = 4,  // TMA 批量存储 (Shared -> Global)
  kCPAsync = 5,    // cp.async 异步拷贝 (Global -> Shared)
  kBulkLoad1D = 6,  // TMA 1D 批量加载
  kBulkStore1D = 7, // TMA 1D 批量存储
  kTMemLoad = 8,   // tcgen05.ld (Tensor Memory -> Register)
  kTMemStore = 9,  // tcgen05.st (Register -> Tensor Memory)
};
```

---

## TMA 描述符

### TMADesc - TMA 拷贝描述符

**文件**: `/root/dev/vibe_dsl/tilelang/src/op/copy.h:61-76`

```cpp
struct TMADesc {
  size_t rank;                    // 张量维度数
  int data_type;                  // 数据类型标识符
  Array<PrimExpr> global_shape;   // 全局内存中的形状
  Array<PrimExpr> global_stride;  // 全局内存中的步长
  Array<PrimExpr> smem_box;       // 共享内存块形状
  Array<PrimExpr> smem_stride;    // 共享内存步长
  PrimExpr global_addr;           // 全局内存基地址
  int swizzle;                    // 内存布局 swizzle 参数
  int interleave;                 // 内存交错参数
  int oob_fill;                   // 越界填充策略
  int l2_promotion;               // L2 缓存提升标志

  Array<PrimExpr> EncodeCallArgs() const;
};
```

### TMAIm2ColDesc - Im2Col TMA 描述符

**文件**: `/root/dev/vibe_dsl/tilelang/src/op/copy.h:84-106`

用于 Conv2D 的 im2col 变换，支持从输入图像中提取 patches。

```cpp
struct TMAIm2ColDesc {
  size_t rank;
  int data_type;
  Array<PrimExpr> global_shape;
  Array<PrimExpr> global_stride;
  Array<PrimExpr> elem_stride;    // 元素级步长
  Array<PrimExpr> lower_corner;   // 提取窗口的下界偏移
  Array<PrimExpr> upper_corner;   // 提取窗口的上界偏移
  PrimExpr global_addr;
  int smem_box_pixel;
  int smem_box_channel;
  int swizzle;
  int interleave;
  int oob_fill;
  int l2_promotion;
};
```

---

## Copy 操作节点

### CopyNode

**文件**: `/root/dev/vibe_dsl/tilelang/src/op/copy.h:117-368`

```cpp
class CopyNode : public TileOperatorNode {
public:
  Buffer src, dst;                    // 源和目标缓冲区
  Array<Range> src_range, dst_range;  // 源和目标范围
  Map<String, ObjectRef> annotations; // 注解
  mutable ParallelOp par_op_;         // 并行操作

  // 核心方法
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T, InferLevel level) const override;

  // 检查方法
  bool CheckBulkLoad(Target target, arith::Analyzer *analyzer, bool check_last_dim = true) const;
  bool CheckBulkStore(Target target, arith::Analyzer *analyzer, bool check_last_dim = true) const;
  bool CheckLDSMCopy(Target target) const;
  bool CheckSTSMCopy(Target target) const;
  bool CheckTMemLoad(Target target) const;
  bool CheckTMemStore(Target target) const;
  bool CheckCPAsyncCopy(Target target, const LayoutMap &layout_map, arith::Analyzer *analyzer) const;

  // 获取拷贝指令类型
  CopyInst GetCopyInst(Target target, bool disable_tma_lower,
                       const LayoutMap &layout_map, arith::Analyzer *analyzer,
                       bool buffer_oob = false, bool in_pipeline = false) const;
};
```

### 支持的注解键

**文件**: `/root/dev/vibe_dsl/tilelang/src/op/copy.h:122-129`

| 注解键 | 类型 | 说明 |
|--------|------|------|
| `coalesced_width` | IntImm | 合并内存访问宽度 |
| `disable_tma` | Bool | 是否禁用 TMA 加速 |
| `eviction_policy` | IntImm | 缓存驱逐策略 (0=正常, 1=优先, 2=最后) |
| `parallel_loop_layout` | Fragment | 并行循环布局提示 |
| `is_async_copy` | Bool | 是否为异步拷贝 |

---

## 拷贝操作实现详解

### 1. 构造函数

**文件**: `/root/dev/vibe_dsl/tilelang/src/op/copy.cc:169-183`

```cpp
Copy::Copy(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  ObjectPtr<CopyNode> node = tvm::ffi::make_object<CopyNode>();
  Array<Range> rgs[2];
  Buffer bf[2];
  for (int i = 0; i < 2; i++) {
    auto region = NormalizeToBufferRegion(args[i]);
    rgs[i] = region->region;
    bf[i] = region->buffer;
  }
  std::tie(node->src, node->dst) = std::tie(bf[0], bf[1]);
  std::tie(node->src_range, node->dst_range) = std::tie(rgs[0], rgs[1]);
  node->annotations = annotations;
  data_ = std::move(node);
}
```

### 2. 拷贝指令选择

**文件**: `/root/dev/vibe_dsl/tilelang/src/op/copy.cc:827-886`

```cpp
CopyInst CopyNode::GetCopyInst(Target target, bool disable_tma_lower,
                               const LayoutMap &layout_map,
                               arith::Analyzer *analyzer, bool buffer_oob,
                               bool in_pipeline) const {
  // 优先级顺序：
  // 1. BulkLoad1D / BulkStore1D (1D TMA)
  // 2. BulkLoad / BulkStore (多维 TMA)
  // 3. LDSM / STSM (矩阵加载/存储)
  // 4. TMemLoad / TMemStore (Tensor Memory)
  // 5. CPAsync (异步拷贝)
  // 6. Normal (普通拷贝)
}
```

### 3. Lower 主入口

**文件**: `/root/dev/vibe_dsl/tilelang/src/op/copy.cc:890-927`

```cpp
Stmt CopyNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  Target target = T.target;
  // 获取 PassContext 配置
  bool disable_tma_lower = pass_ctx->GetConfig<Bool>(kDisableTMALower, Bool(false)).value();

  auto copy_inst = GetCopyInst(target, disable_tma_lower || GetDisableTMA(),
                               T.layout_map, analyzer, /*buffer_oob=*/false,
                               /*in_pipeline=*/T.in_pipeline);

  // 根据指令类型分发到对应的 Lower 方法
  switch (copy_inst) {
    case CopyInst::kTMemLoad:
    case CopyInst::kTMemStore:
      return LowerTmemCopy(T, analyzer);
    case CopyInst::kBulkLoad1D:
    case CopyInst::kBulkStore1D:
      return LowerBulkCopy1D(T, analyzer, copy_inst);
    case CopyInst::kBulkLoad:
    case CopyInst::kBulkStore:
      return LowerBulkCopy(T, analyzer, copy_inst);
    case CopyInst::kLDSM:
    case CopyInst::kSTSM:
      return LowerLDSMCopy(T, analyzer, copy_inst);
    case CopyInst::kCPAsync:
      return LowerCPAsyncCopy(T, analyzer);
    case CopyInst::kNormal:
      return LowerNormalCopy(T, analyzer);
  }
}
```

---

## 各种 Lower 实现

### 1. 普通拷贝 (LowerNormalCopy)

**文件**: `/root/dev/vibe_dsl/tilelang/src/op/copy.cc:986-1023`

实现标准的 SIMT 循环拷贝，支持向量化优化。

```cpp
Stmt CopyNode::LowerNormalCopy(const LowerArgs &T, arith::Analyzer *analyzer) const {
  bool is_cpu_target = T.target->GetTargetDeviceType() == kDLCPU;
  auto simt_loop = MakeSIMTLoop(analyzer);
  auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));

  if (is_cpu_target || IsLocalBuffer(src) || IsLocalBuffer(dst)) {
    // CPU 目标或本地缓冲区：直接向量化
    return VectorizeLoop(fused_loop, T.layout_map);
  } else {
    // GPU 目标：使用并行操作进行布局推断和循环优化
    auto par_op = ParallelOp(fused_loop);
    // ... 布局推断和循环 lowering
    return LowerParallelLoop(par_op->GetRoot(), loop_layout, T.thread_var,
                             analyzer, T.layout_map, par_op->GetPredicate(T.thread_var));
  }
}
```

### 2. TMA 批量拷贝 (LowerBulkCopy)

**文件**: `/root/dev/vibe_dsl/tilelang/src/op/copy.cc:1427-1726`

实现 Hopper GPU 的 Tensor Memory Accelerator (TMA) 批量拷贝。

**关键步骤**:
1. 验证全局布局非 swizzled
2. 构建 TMA 描述符
3. 检测共享内存 swizzle 布局
4. 生成 TMA 拷贝指令

```cpp
Stmt CopyNode::LowerBulkCopy(const LowerArgs &T, arith::Analyzer *analyzer,
                             CopyInst copy_inst) const {
  TMADesc desc;
  desc.rank = global_tensor->shape.size();
  desc.data_type = to_CUtensorMapDataType(global_tensor->dtype);
  desc.global_addr = global_tensor->data;
  desc.global_shape = ReverseArray(global_tensor->shape);
  // ... 填充其他字段

  // 检测 swizzle 布局
  if (StructuralEqual()(shared_layout, makeQuarterBankSwizzleLayout(...))) {
    desc.swizzle = CU_TENSOR_MAP_SWIZZLE_32B;
  } else if (...) {
    desc.swizzle = CU_TENSOR_MAP_SWIZZLE_64B;
  } else if (...) {
    desc.swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
  }

  // 生成 TMA 拷贝
  Call create_descriptor = Call(DataType::Handle(), create_tma_descriptor(), desc.EncodeCallArgs());
  // ...
}
```

### 3. LDSM/STSM 矩阵拷贝 (LowerLDSMCopy)

**文件**: `/root/dev/vibe_dsl/tilelang/src/op/copy.cc:1027-1220`

实现 warp 级 8x8 矩阵加载/存储指令。

**约束条件**:
- 需要 2D 循环
- 无边界谓词
- 全范围访问
- 8x8 矩阵布局
- 16 位数据类型 (half)

```cpp
Stmt CopyNode::LowerLDSMCopy(const LowerArgs &T, arith::Analyzer *analyzer,
                             CopyInst copy_inst) const {
  // 检查约束
  if (loop_vars.size() < 2) return LowerNormalCopy(T, analyzer);
  if (src_predicate.defined() || dst_predicate.defined())
    return LowerNormalCopy(T, analyzer);

  // 验证 8x8 布局
  PrimExpr matrix_8x8_thread_map = makeGemmFragment8x8()->ForwardThread(...);
  if (!analyzer->CanProveEqual(matrix_8x8_thread_map, local_layout_thread_map))
    return LowerNormalCopy(T, analyzer);

  // 生成 ldmatrix/stmatrix 指令
  const Op &op = is_ldmatrix ? tl::ptx_ldmatrix() : tl::ptx_stmatrix();
  // ...
}
```

### 4. Tensor Memory 拷贝 (LowerTmemCopy)

**文件**: `/root/dev/vibe_dsl/tilelang/src/op/copy.cc:1224-1423`

实现 Blackwell (SM100) 的 Tensor Memory 操作 (tcgen05.ld/st)。

**支持的操作**:
- `tcgen05.ld`: Tensor Memory -> Register
- `tcgen05.st`: Register -> Tensor Memory
- `tcgen05.cp`: Shared Memory -> Tensor Memory (TODO)

```cpp
Stmt CopyNode::LowerTmemCopy(const LowerArgs &T, arith::Analyzer *analyzer) const {
  // 确定拷贝类型
  bool is_ld = (src.scope() == "shared.tmem" && IsFragmentBuffer(dst));
  bool is_st = (IsFragmentBuffer(src) && dst.scope() == "shared.tmem");

  // 检查布局兼容性
  auto [is_success, target_frag, num_chunks_each_wg] = expandTcgen05Layout(
      meta, tmem_phy_col_extent, num_useful_threads, row_dom, col_dom);

  // 生成 tcgen05 指令
  args.push_back(StringImm(meta.intrinsics_name + "<" + ... + ">"));
  args.push_back(BufferLoad(tmem_buf, ...));
  args.push_back(col_offset);
  args.push_back(reg_buf.access_ptr(...));
}
```

### 5. CPAsync 异步拷贝 (LowerCPAsyncCopy)

**文件**: `/root/dev/vibe_dsl/tilelang/src/op/copy.cc:933-983`

实现 Ampere/Hopper 的异步拷贝指令。

```cpp
Stmt CopyNode::LowerCPAsyncCopy(const LowerArgs &T, arith::Analyzer *analyzer) const {
  auto simt_loop = MakeSIMTLoop(analyzer);
  auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));
  auto par_op = ParallelOp(fused_loop);

  // 推断布局
  par_op->InferLayout(...);
  auto loop_layout = par_op->GetLoopLayout();

  // 重写为 cp.async
  CPAsyncStoreRewriter cp_async_rewriter;
  Stmt cp_async_loop = cp_async_rewriter.Rewrite(lowered_loop);

  // 添加 commit 和 wait
  Stmt commit_group = Evaluate(Call(DataType::Handle(), builtin::ptx_commit_group(), {}));
  Stmt wait_group = Evaluate(Call(DataType::Handle(), builtin::ptx_wait_group(), {IntImm(DataType::Int(32), 0)}));
  return SeqStmt({cp_async_loop, commit_group, wait_group});
}
```

---

## 辅助方法

### MakeSIMTLoop - 创建 SIMT 循环

**文件**: `/root/dev/vibe_dsl/tilelang/src/op/copy.cc:344-400`

```cpp
For CopyNode::MakeSIMTLoop(arith::Analyzer *analyzer) const {
  Array<IterVar> loop_vars = MakeIterVars();
  // 绑定变量到分析器
  for (const auto &iv : loop_vars)
    analyzer->Bind(iv->var, iv->dom);

  // 生成索引和谓词
  Array<PrimExpr> src_indices = MakeIndices(loop_vars, 0);
  Array<PrimExpr> dst_indices = MakeIndices(loop_vars, 1);
  PrimExpr src_predicate = MakePredicate(analyzer, loop_vars, src->shape, 0);
  PrimExpr dst_predicate = MakePredicate(analyzer, loop_vars, dst->shape, 1);

  // 构建循环体
  PrimExpr value = BufferLoad(src, src_indices);
  if (src->dtype != dst->dtype)
    value = Cast(dst->dtype, value);
  if (src_predicate.defined())
    value = if_then_else(src_predicate, value, make_zero(dst->dtype));

  Stmt body = BufferStore(dst, value, dst_indices);
  if (dst_predicate.defined())
    body = IfThenElse(dst_predicate, body);
  // ... 构建嵌套循环
}
```

### MakeIterVars - 创建迭代变量

**文件**: `/root/dev/vibe_dsl/tilelang/src/op/copy.cc:195-287`

根据源和目标缓冲区的内存作用域级别选择基础范围。

```cpp
Array<IterVar> CopyNode::MakeIterVars() const {
  // 作用域级别：global < shared < local.fragment
  auto scope_level = [](const Buffer &b) -> int {
    String s = b.scope();
    if (s == "local.fragment" || s == "local") return 2;
    if (s == "shared" || s == "shared.dyn" || s == "shared.tmem") return 1;
    return 0;
  };

  // 选择更高级别的作用域作为基础范围
  bool base_is_src = (scope_level(src) >= scope_level(dst));
  const Array<Range> &base_ranges = base_is_src ? src_range : dst_range;
  // ...
}
```

---

## Conv2D Im2Col 操作

### Conv2DIm2ColOpNode

**文件**: `/root/dev/vibe_dsl/tilelang/src/op/copy.h:394-450`

专门用于 Conv2D 的 im2col 变换操作。

```cpp
class Conv2DIm2ColOpNode : public TileOperatorNode {
public:
  BufferRegion srcRegion_, dstRegion_;
  Buffer src_, dst_;
  int stride_;
  int padding_;
  int dilation_;
  int kernel_;
  int eviction_policy_;
  PrimExpr nhw_step_;
  PrimExpr c_step_;

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T, InferLevel level) const override;
};
```

### Lower 实现

**文件**: `/root/dev/vibe_dsl/tilelang/src/op/copy.cc:1866-1980`

```cpp
Stmt Conv2DIm2ColOpNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  ICHECK(TargetIsHopper(T.target));
  // 构建 TMAIm2ColDesc
  TMAIm2ColDesc desc;
  desc.rank = src_->shape.size();
  desc.data_type = to_CUtensorMapDataType(src_->dtype);
  // ... 填充描述符

  // 生成 im2col TMA 加载
  Call create_desc = Call(DataType::Handle(), create_tma_im2col_descriptor(), desc.EncodeCallArgs());
  // ...
}
```

---

## Python 绑定

**文件**: `/root/dev/vibe_dsl/tilelang/src/op/copy.cc:2039-2077`

```cpp
TIR_REGISTER_TL_TILE_OP(Copy, copy)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.tileop.async_copy")
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "async_copy")
    .set_attr<OpBuilderFunc>("TLOpBuilder", ...);

TIR_REGISTER_TL_TILE_OP(Conv2DIm2ColOp, c2d_im2col)
    .set_num_inputs(9)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() {
  CopyNode::RegisterReflection();
  Conv2DIm2ColOpNode::RegisterReflection();
}
```

---

## 代码引用汇总

| 组件 | 文件路径 | 行号范围 |
|------|----------|----------|
| CopyInst 枚举 | `/root/dev/vibe_dsl/tilelang/src/op/copy.h` | 17-30 |
| TMADesc | `/root/dev/vibe_dsl/tilelang/src/op/copy.h` | 61-76 |
| TMAIm2ColDesc | `/root/dev/vibe_dsl/tilelang/src/op/copy.h` | 84-106 |
| CopyNode | `/root/dev/vibe_dsl/tilelang/src/op/copy.h` | 117-368 |
| Copy 构造函数 | `/root/dev/vibe_dsl/tilelang/src/op/copy.cc` | 169-183 |
| GetCopyInst | `/root/dev/vibe_dsl/tilelang/src/op/copy.cc` | 827-886 |
| Lower | `/root/dev/vibe_dsl/tilelang/src/op/copy.cc` | 890-927 |
| LowerNormalCopy | `/root/dev/vibe_dsl/tilelang/src/op/copy.cc` | 986-1023 |
| LowerLDSMCopy | `/root/dev/vibe_dsl/tilelang/src/op/copy.cc` | 1027-1220 |
| LowerTmemCopy | `/root/dev/vibe_dsl/tilelang/src/op/copy.cc` | 1224-1423 |
| LowerBulkCopy | `/root/dev/vibe_dsl/tilelang/src/op/copy.cc` | 1427-1726 |
| LowerBulkCopy1D | `/root/dev/vibe_dsl/tilelang/src/op/copy.cc` | 1728-1812 |
| LowerCPAsyncCopy | `/root/dev/vibe_dsl/tilelang/src/op/copy.cc` | 933-983 |
| Conv2DIm2ColOpNode | `/root/dev/vibe_dsl/tilelang/src/op/copy.h` | 394-450 |
| Conv2DIm2Col Lower | `/root/dev/vibe_dsl/tilelang/src/op/copy.cc` | 1866-1980 |
