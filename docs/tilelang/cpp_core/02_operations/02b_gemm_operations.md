# TileLang C++ 核心 - GEMM 操作

## 模块概述

GEMM (General Matrix Multiplication) 操作模块 (`src/op/gemm.cc` 和 `src/op/gemm.h`) 实现了 TileLang 中矩阵乘法的核心功能。该模块支持多种 GPU 架构（Volta、Turing、Ampere、Hopper、Blackwell/SM100）和多种指令类型（MMA、WGMMA、TCGEN5MMA、MFMA）。

---

## GEMM 指令类型

**文件**: `/root/dev/vibe_dsl/tilelang/src/op/gemm.h:42-60`

```cpp
enum class GemmInst : uint8_t {
  kMMA,        // Tensor Core MMA (Volta/Turing/Ampere)
  kWGMMA,      // Warp Group MMA (Hopper)
  kTCGEN5MMA,  // Tensor Core GEN5 MMA (Blackwell/SM100)
  kMFMA,       // Matrix FMA (AMD CDNA)
  kScalar      // 标量实现
};
```

### 指令类型选择逻辑

**文件**: `/root/dev/vibe_dsl/tilelang/src/op/gemm.cc:125-138`

```cpp
GemmInst GemmNode::getGemmInst(int block_size, Target target) const {
  if (allowTcgen5Mma(target)) {
    return GemmInst::kTCGEN5MMA;
  } else if (allowWgmma(block_size, target)) {
    return GemmInst::kWGMMA;
  } else if (TargetIsCDNA(target)) {
    return GemmInst::kMFMA;
  } else if (TargetIsCuda(target)) {
    return GemmInst::kMMA;
  } else {
    ICHECK(0) << "Unsupported target for gemm: " << target;
    return GemmInst::kMMA;
  }
}
```

---

## Warp 分区策略

### GemmWarpPolicyType

**文件**: `/root/dev/vibe_dsl/tilelang/src/op/gemm.h:18-23`

```cpp
enum class GemmWarpPolicyType : uint8_t {
  kSquare = 0,   // 方形分区：尽量平衡 M/N 维度
  kFullRow = 1,  // 全行分区：优先填充 M 维度
  kFullCol = 2,  // 全列分区：优先填充 N 维度
  kFree = 3      // 自由分区：用户指定 m_warp/n_warp
};
```

### GemmWarpPolicyNode

**文件**: `/root/dev/vibe_dsl/tilelang/src/op/gemm.h:62-118`

```cpp
class GemmWarpPolicyNode : public Object {
public:
  mutable int m_warp{0};  // M 维度 warp 数
  mutable int n_warp{0};  // N 维度 warp 数
  int policy_type;        // 策略类型

  // 计算 warp 分区
  std::pair<int, int> computeWarpPartition(int M, int N, int block_size,
                                           Target target, GemmInst gemm_inst) const;

  bool isSquare() const { return policy_type == int(GemmWarpPolicyType::kSquare); }
  bool isFullRow() const { return policy_type == int(GemmWarpPolicyType::kFullRow); }
  bool isFullCol() const { return policy_type == int(GemmWarpPolicyType::kFullCol); }
  bool isFree() const { return policy_type == int(GemmWarpPolicyType::kFree); }
};
```

### Warp 分区计算

**文件**: `/root/dev/vibe_dsl/tilelang/src/op/gemm.cc:140-324`

```cpp
std::pair<int, int> GemmWarpPolicyNode::computeWarpPartition(
    int M, int N, int block_size, Target target, GemmInst gemm_inst) const {
  int num_warps = block_size / TargetGetWarpSize(target);

  if (gemm_inst == GemmInst::kTCGEN5MMA) {
    // TCGEN5MMA 不关心 warp 分区
    this->m_warp = 1;
    this->n_warp = num_warps;
    return {1, num_warps};
  }

  constexpr int kMPerWarp = 16;  // 每个 warp 处理的行数
  int kNPerWarp = 8;             // 每个 warp 处理的列数

  if (gemm_inst == GemmInst::kWGMMA) {
    // WGMMA 需要 warp-group (4 warps) 对齐
    ICHECK(num_warps % 4 == 0) << "Warp-Group MMA requires 128xk threads.";
    constexpr int kGroup = 4;

    if (this->isFullRow()) {
      // 尽量在 M 维度放置更多 warp-groups
      for (int cand = num_warps; cand >= kGroup; cand -= kGroup) {
        if (M % (cand * kMPerWarp) == 0) {
          m_warp = cand;
          n_warp = num_warps / m_warp;
          break;
        }
      }
    } else if (this->isFullCol()) {
      // 尽量在 N 维度放置 warps
      // ...
    } else if (this->isSquare()) {
      // 穷举搜索最佳平衡分区
      float ideal = N > 0 ? static_cast<float>(M) / N : 1.f;
      float best_score = std::numeric_limits<float>::max();
      for (int m = kGroup; m <= num_warps && m <= max_m; m += kGroup) {
        // ... 计算最佳分区
      }
    }
  }
  // ...
}
```

---

## GEMM 操作节点

### GemmNode

**文件**: `/root/dev/vibe_dsl/tilelang/src/op/gemm.h:120-178`

```cpp
class GemmNode : public TileOperatorNode {
public:
  // 缓冲区
  tir::Buffer a_, b_, c_;
  BufferRegion aRegion_, bRegion_, cRegion_;

  // 矩阵属性
  bool transA_, transB_;  // 转置标志
  int m_, n_, k_;         // 矩阵维度
  int strideA_, strideB_; // 步长
  int offsetA_, offsetB_; // 偏移

  // 控制标志
  PrimExpr clearAccum_ = const_false();  // 是否清零累加器
  int kPack_ = 1;                        // k-pack (CDNA MFMA)
  int wgWait_ = 0;                       // warp-group 等待值
  tir::BufferLoad mbar_;                 // mbarrier (TCGEN5MMA)
  Array<PrimExpr> cCoords_;              // C 缓冲区坐标
  mutable GemmWarpPolicy policy_;        // Warp 分区策略

  // 核心方法
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T, InferLevel level) const override;
  TileOperator Clone() const;

private:
  GemmInst getGemmInst(int block_size, Target target) const;
  bool allowTcgen5Mma(Target target) const;
  bool allowWgmma(int block_size, Target target) const;
  bool checkWgmma() const;
};
```

### 构造函数

**文件**: `/root/dev/vibe_dsl/tilelang/src/op/gemm.cc:53-91`

```cpp
Gemm::Gemm(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  ObjectPtr<GemmNode> node = tvm::ffi::make_object<GemmNode>();

  node->aRegion_ = NormalizeToBufferRegion(args[0]);
  node->bRegion_ = NormalizeToBufferRegion(args[1]);
  node->cRegion_ = NormalizeToBufferRegion(args[2]);

  node->a_ = node->aRegion_->buffer;
  node->b_ = node->bRegion_->buffer;
  node->c_ = node->cRegion_->buffer;
  node->transA_ = args[3].as<Bool>().value();
  node->transB_ = args[4].as<Bool>().value();
  node->m_ = args[5].as<IntImm>().value()->value;
  node->n_ = args[6].as<IntImm>().value()->value;
  node->k_ = args[7].as<IntImm>().value()->value;
  node->policy_ = GemmWarpPolicy(args[8].as<IntImm>().value()->value);
  node->clearAccum_ = args[9].as<PrimExpr>().value();
  node->strideA_ = args[10].as<IntImm>().value()->value;
  node->strideB_ = args[11].as<IntImm>().value()->value;
  node->offsetA_ = args[12].as<IntImm>().value()->value;
  node->offsetB_ = args[13].as<IntImm>().value()->value;
  if (args.size() > 14) {
    node->kPack_ = args[14].as<IntImm>().value()->value;
    ICHECK(node->kPack_ == 1 || node->kPack_ == 2) << "kPack must be 1 or 2";
  }
  if (args.size() > 15) {
    node->wgWait_ = args[15].as<IntImm>().value()->value;
  }
  if (args.size() > 16) {
    if (const auto *load = args[16].as<BufferLoadNode>()) {
      node->mbar_ = Downcast<BufferLoad>(args[16]);
    }
  }
  node->cCoords_ = Array<PrimExpr>(
      {args[17].as<PrimExpr>().value(), args[18].as<PrimExpr>().value()});
  data_ = std::move(node);
}
```

---

## WGMMA 支持检查

**文件**: `/root/dev/vibe_dsl/tilelang/src/op/gemm.cc:356-395`

```cpp
bool GemmNode::checkWgmma() const {
  // B 必须在共享内存
  if (b_.scope() != "shared.dyn" && b_.scope() != "shared") {
    return false;
  }

  // 检查数据类型组合和 K 对齐
  if (c_->dtype == DataType::Float(16)) {
    if (a_->dtype == DataType::Float(16) && b_->dtype == DataType::Float(16))
      return k_ % 16 == 0;
    else if (a_->dtype.is_float8() && b_->dtype.is_float8())
      return (!transA_) && transB_ && k_ % 32 == 0;
  } else if (c_->dtype == DataType::Float(32)) {
    if (a_->dtype == DataType::Float(16) && b_->dtype == DataType::Float(16))
      return k_ % 16 == 0;
    else if (a_->dtype == DataType::BFloat(16) && b_->dtype == DataType::BFloat(16))
      return k_ % 16 == 0;
    else if (a_->dtype == DataType::Float(32) && b_->dtype == DataType::Float(32))
      return (!transA_) && transB_ && k_ % 8 == 0;
    else if (a_->dtype.is_float8() && b_->dtype.is_float8())
      return (!transA_) && transB_ && k_ % 32 == 0;
  } else if (c_->dtype == DataType::Int(32)) {
    // 8-bit 整数组合
    if (a_->dtype == DataType::Int(8) && b_->dtype == DataType::Int(8))
      return (!transA_) && transB_ && k_ % 32 == 0;
    // ... 其他整数类型组合
  }
  return false;
}
```

---

## Lower 实现

### Lower 主函数

**文件**: `/root/dev/vibe_dsl/tilelang/src/op/gemm.cc:437-572`

```cpp
Stmt GemmNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  auto block_size = *as_const_int(T.thread_bounds->extent);
  GemmInst gemm_inst = getGemmInst(block_size, T.target);
  auto [warp_m, warp_n] = policy_->computeWarpPartition(m_, n_, block_size, T.target, gemm_inst);

  // 构建访问指针
  PrimExpr Aptr = MakeAccessPtrFromRegion(aRegion_, /*r*/ 1, /*require_2d*/ true);
  PrimExpr Bptr = MakeAccessPtrFromRegion(bRegion_, /*r*/ 1, /*require_2d*/ true);
  PrimExpr Cptr = MakeAccessPtrFromRegion(cRegion_, /*rw*/ 3, /*require_2d*/ true);

  std::stringstream ss;
  std::string op_name;

  if (gemm_inst == GemmInst::kTCGEN5MMA) {
    // TCGEN5MMA 特殊处理
    auto [can_use_tcgen5mma, meta] = GetTCGEN5MMAMeta(m_, n_, k_, a_->dtype, c_->dtype);
    // ...
    ss << op_name << "<" << m_ << ", " << n_ << ", " << k_ << ", ";
    ss << meta.atom_m << ", " << meta.atom_n << ", " << meta.atom_k << ", ";
    ss << transA_ << ", " << transB_ << ", " << accum_dtype << ">";
    // ...
  }

  // 选择 gemm 变体
  if (IsFragmentBuffer(a_)) {
    op_name = "tl::gemm_rs";  // Register-Shared
  } else if (IsFragmentBuffer(b_)) {
    op_name = "tl::gemm_sr";  // Shared-Register
  } else {
    op_name = "tl::gemm_ss";  // Shared-Shared
  }

  // 构建模板参数
  ss << op_name << "<" << m_ << ", " << n_ << ", " << k_ << ", ";
  ss << warp_m << ", " << warp_n << ", ";
  ss << transA_ << ", " << transB_;
  ss << ", " << bool(clear_accum_bool.value());

  // 架构特定参数
  if (TargetIsCuda(T.target) && (GetArchInt(T.target) >= 75)) {
    ss << ", " << strideA_ << ", " << strideB_;
    ss << ", " << offsetA_ << ", " << offsetB_;
  }
  if (TargetIsCDNA(T.target)) {
    ss << ", " << kPack_;
  } else if (TargetIsHopper(T.target)) {
    ss << ", " << (gemm_inst == GemmInst::kWGMMA ? "true" : "false");
  }

  // wg_wait 参数
  if (TargetIsHopper(T.target) && wgWait_ != 0) {
    ss << ", " << wgWait_;
  }
  ss << ">";

  auto new_call = Call(DataType::Handle(), tl::tl_gemm(),
                       Array<PrimExpr>{StringImm(ss.str()), Aptr, Bptr, Cptr});
  return Evaluate(new_call);
}
```

---

## 布局推断

### InferLayout 主函数

**文件**: `/root/dev/vibe_dsl/tilelang/src/op/gemm.cc:593-818`

```cpp
LayoutMap GemmNode::InferLayout(const LayoutInferArgs &T, InferLevel level) const {
  if (completed_) return {};
  LayoutMap results;

  auto block_size = *as_const_int(T.thread_bounds->extent);
  GemmInst gemm_inst = getGemmInst(block_size, T.target);
  auto [warp_m, warp_n] = policy_->computeWarpPartition(m_, n_, block_size, T.target, gemm_inst);

  if (TargetIsVolta(T.target)) {
    // Volta 布局
    auto fragment = makeGemmVoltaFragmentC(m_, n_, m_ / warp_m, n_ / warp_n, c_->dtype.bits());
    results.Set(c_, fragment->BindThreadRange(thread_range));
    // ... A 和 B 布局
  } else if (TargetIsAmpere(T.target) || TargetIsTuring(T.target) || TargetIsSM120(T.target)) {
    // Ampere/Turing/SM120 MMA 布局
    auto fragment = makeGemmFragmentC(m_, n_, m_ / warp_m, n_ / warp_n, c_->dtype.bits());
    results.Set(c_, fragment->BindThreadRange(thread_range));
    // ...
  } else if (TargetIsHopper(T.target)) {
    // Hopper WGMMA/MMA 布局
    auto fragment = gemm_inst == GemmInst::kWGMMA
                        ? makeGemmFragmentCHopper(m_, n_, m_ / warp_m, n_ / warp_n, c_->dtype.bits())
                        : makeGemmFragmentC(m_, n_, m_ / warp_m, n_ / warp_n, c_->dtype.bits());
    // ...
  } else if (gemm_inst == GemmInst::kTCGEN5MMA) {
    // Blackwell TCGEN5MMA 布局
    ICHECK(c_.scope() == "shared.tmem");
    // ... 特殊 TMEM 布局
  } else if (TargetIsCDNA(T.target)) {
    // AMD CDNA MFMA 布局
    auto fragment = makeGemmFragmentCCDNA(m_, n_, m_ / warp_m, n_ / warp_n, c_->dtype.bits());
    // ...
  }
  completed_ = true;
  return results;
}
```

### TCGEN5MMA 布局推断

**文件**: `/root/dev/vibe_dsl/tilelang/src/op/gemm.cc:716-776`

```cpp
} else if (gemm_inst == GemmInst::kTCGEN5MMA) {
  ICHECK(c_.scope() == "shared.tmem")
      << "TCGEN5MMA only supports C in shared.tmem scope, got " << c_.scope();

  auto [can_use_tcgen5mma, meta] = GetTCGEN5MMAMeta(m_, n_, k_, a_->dtype, c_->dtype);
  ICHECK(can_use_tcgen5mma);

  // A 和 B 的共享内存布局
  {
    auto layout = makeGemmABLayoutSm100(mat_stride, mat_continuous, mat_continuous,
                                        a_->dtype.bits(), transA_ ? 1 : 2);
    results.Set(a_, ExpandLayoutToMatchBuffer(layout, a_));
  }
  {
    auto layout = makeGemmABLayoutSm100(mat_stride, mat_continuous, continuity,
                                        b_->dtype.bits(), transB_ ? 2 : 1);
    results.Set(b_, ExpandLayoutToMatchBuffer(layout, b_));
  }

  // C 的 TMEM 布局
  {
    Layout res;
    IterVar i = make_itervar("i", m_);
    IterVar j = make_itervar("j", n_);
    PrimExpr atom_idx = FloorDiv(i, meta.atom_m) + FloorDiv(j, meta.atom_n) * (m_ / meta.atom_m);
    PrimExpr ai = FloorMod(i, meta.atom_m);
    PrimExpr aj = FloorMod(j, meta.atom_n);

    if (meta.atom_m == 128) {
      // Layout D
      res = Layout(Array{i, j}, {ai, aj + atom_idx * meta.atom_n});
    } else if (meta.atom_m == 64) {
      // Layout E (.ws variant)
      res = Layout(Array{i, j}, {FloorDiv(ai, 32) * 32 + FloorMod(ai, 32) +
                                     FloorDiv(aj, meta.atom_n / 2) * 64,
                                 FloorMod(aj, meta.atom_n / 2) +
                                     atom_idx * (meta.atom_n / 2)});
    } else if (meta.atom_m == 128) {
      // Layout G
      res = Layout(Array{i, j},
                   {FloorMod(ai, 32) + FloorDiv(aj, meta.atom_n / 4) * 32,
                    FloorMod(aj, meta.atom_n / 4) + atom_idx * (meta.atom_n / 4)});
    }
    results.Set(c_, res);
  }
}
```

---

## 目标架构检查

### TCGEN5MMA 支持检查

**文件**: `/root/dev/vibe_dsl/tilelang/src/op/gemm.cc:106-113`

```cpp
bool GemmNode::allowTcgen5Mma(Target target) const {
  return TargetIsSm100(target) &&
         ((a_.scope() == "shared.dyn" || a_.scope() == "shared" ||
           a_.scope() == "shared.tmem") &&
          (b_.scope() == "shared.dyn" || b_.scope() == "shared") &&
          c_.scope() == "shared.tmem") &&
         GetTCGEN5MMAMeta(m_, n_, k_, a_->dtype, c_->dtype).first;
}
```

### WGMMA 支持检查

**文件**: `/root/dev/vibe_dsl/tilelang/src/op/gemm.cc:115-123`

```cpp
bool GemmNode::allowWgmma(int block_size, Target target) const {
  tvm::transform::PassContext ctxt = tvm::transform::PassContext::Current();
  int warp_size = TargetGetWarpSize(target);
  int num_warps = block_size / warp_size;
  return !ctxt->GetConfig(kDisableWGMMA, Optional<Bool>()).value_or(false) &&
         TargetIsHopper(target) && (this->m_ >= 64) && (num_warps % 4 == 0) &&
         checkWgmma();
}
```

---

## Python 绑定

**文件**: `/root/dev/vibe_dsl/tilelang/src/op/gemm.cc:820-838`

```cpp
TIR_REGISTER_TL_TILE_OP(Gemm, gemm)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.GemmWarpPolicy")
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "GemmWarpPolicy");

TVM_FFI_STATIC_INIT_BLOCK() {
  GemmNode::RegisterReflection();
  GemmWarpPolicyNode::RegisterReflection();
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.GemmWarpPolicyComputeWarpPartition",
                        [](GemmWarpPolicy policy, int M, int N, int block_size,
                           Target target, GemmInst gemm_inst) {
                          policy->computeWarpPartition(M, N, block_size, target,
                                                       gemm_inst);
                        });
}
```

---

## 代码引用汇总

| 组件 | 文件路径 | 行号范围 |
|------|----------|----------|
| GemmInst 枚举 | `/root/dev/vibe_dsl/tilelang/src/op/gemm.h` | 42-60 |
| GemmWarpPolicyType | `/root/dev/vibe_dsl/tilelang/src/op/gemm.h` | 18-23 |
| GemmWarpPolicyNode | `/root/dev/vibe_dsl/tilelang/src/op/gemm.h` | 62-118 |
| GemmNode | `/root/dev/vibe_dsl/tilelang/src/op/gemm.h` | 120-178 |
| Gemm 构造函数 | `/root/dev/vibe_dsl/tilelang/src/op/gemm.cc` | 53-91 |
| GemmNode::Clone | `/root/dev/vibe_dsl/tilelang/src/op/gemm.cc` | 101-104 |
| allowTcgen5Mma | `/root/dev/vibe_dsl/tilelang/src/op/gemm.cc` | 106-113 |
| allowWgmma | `/root/dev/vibe_dsl/tilelang/src/op/gemm.cc` | 115-123 |
| getGemmInst | `/root/dev/vibe_dsl/tilelang/src/op/gemm.cc` | 125-138 |
| computeWarpPartition | `/root/dev/vibe_dsl/tilelang/src/op/gemm.cc` | 140-324 |
| checkWgmma | `/root/dev/vibe_dsl/tilelang/src/op/gemm.cc` | 356-395 |
| GetArchInt | `/root/dev/vibe_dsl/tilelang/src/op/gemm.cc` | 411-422 |
| Lower | `/root/dev/vibe_dsl/tilelang/src/op/gemm.cc` | 437-572 |
| InferLayout | `/root/dev/vibe_dsl/tilelang/src/op/gemm.cc` | 593-818 |
| Python 绑定 | `/root/dev/vibe_dsl/tilelang/src/op/gemm.cc` | 820-838 |
