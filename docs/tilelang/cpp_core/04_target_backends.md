# TileLang 目标后端实现 (Target Backends)

## 1. 概述

TileLang 支持多种目标后端，主要包括 NVIDIA CUDA 和 AMD HIP。每个后端都包含代码生成器 (CodeGen) 和运行时模块 (Runtime Module) 两部分。

**核心文件位置：**
- `/root/dev/vibe_dsl/tilelang/src/target/codegen_cuda.h` 和 `.cc` - CUDA 代码生成
- `/root/dev/vibe_dsl/tilelang/src/target/codegen_hip.h` 和 `.cc` - HIP 代码生成
- `/root/dev/vibe_dsl/tilelang/src/target/rt_mod_cuda.cc` - CUDA 运行时模块
- `/root/dev/vibe_dsl/tilelang/src/target/rt_mod_hip.cc` - HIP 运行时模块
- `/root/dev/vibe_dsl/tilelang/src/target/intrin_rule_cuda.cc` - CUDA 内建规则
- `/root/dev/vibe_dsl/tilelang/src/target/intrin_rule_hip.cc` - HIP 内建规则

## 2. CUDA 后端

### 2.1 CUDA 代码生成器架构

```cpp
// src/target/codegen_cuda.h:21-174
class CodeGenTileLangCUDA final : public CodeGenC {
public:
  CodeGenTileLangCUDA();
  std::string Finish();

  // 重写基类行为
  void PrintFuncPrefix(std::ostream &os) final;
  void PrintExtraAttrs(const PrimFunc &f);
  void VisitStmt_(const ForNode *op) final;
  void PrintStorageSync(const CallNode *op) final;
  void PrintStorageScope(const std::string &scope, std::ostream &os) final;
  void PrintVecBinaryOp(const std::string &op, DataType t, PrimExpr lhs,
                        PrimExpr rhs, std::ostream &os) final;
  void PrintType(DataType t, std::ostream &os) final;
  void PrintVecElemLoad(const std::string &vec, DataType t, int i,
                        std::ostream &os) final;
  void PrintVecElemStore(const std::string &vec, DataType t, int i,
                         const std::string &value) final;
  void BindThreadIndex(const IterVar &iv) final;
  std::string CastFromTo(std::string value, DataType from, DataType target) final;

  // 表达式访问器重载
  void VisitExpr_(const RampNode *op, std::ostream &os) final;
  void VisitExpr_(const BroadcastNode *op, std::ostream &os) final;
  void VisitExpr_(const FloatImmNode *op, std::ostream &os) final;
  void VisitExpr_(const CallNode *op, std::ostream &os) final;
  void VisitExpr_(const CastNode *op, std::ostream &os) final;
  void VisitExpr_(const MinNode *op, std::ostream &os) final;
  void VisitExpr_(const MaxNode *op, std::ostream &os) final;
  void VisitStmt_(const EvaluateNode *op) final;
  void VisitStmt_(const AllocateNode *op) final;
  void VisitStmt_(const AttrStmtNode *op) final;
  void VisitExpr_(const BufferLoadNode *op, std::ostream &os) final;
  void VisitStmt_(const BufferStoreNode *op) final;

  void AddFunction(const GlobalVar &gvar, const PrimFunc &f);
  void PrintFunctionSignature(const ffi::String &function_name,
                              const PrimFunc &func, std::ostream &os);

protected:
  void ReserveKeywordsAsUnique_();
  std::string GetBufferRef(DataType t, const BufferNode *buffer,
                           PrimExpr index) final;
  void PrintCallExtern(Type ret_type, ffi::String global_symbol,
                       const ffi::Array<PrimExpr> &args, bool skip_first_arg,
                       std::ostream &os) final;

private:
  bool need_global_barrier_{false};
  std::string vid_global_barrier_state_;
  std::string vid_global_barrier_expect_;

  // 数据类型支持标志
  bool enable_fp16_{false};
  bool enable_bf16_{false};
  bool enable_fp8_{false};
  bool enable_fp6_{false};
  bool enable_fp4_{false};
  bool enable_int8_{false};
  bool enable_sparse_gemm_{false};
  bool enable_warp_shuffle_{false};

  // 头文件依赖标志
  bool need_math_constants_h_{false};
  bool need_mma_h_{false};
  bool need_mma_instruction_h_{false};
  bool need_wgmma_instruction_h_{false};
  bool need_tcgen05mma_instruction_h_{false};
  bool need_mma_sm70_instruction_h_{false};
  bool need_tcgen05_common_h_{false};
  bool need_cast_smem_ptr_to_int_{false};
  bool need_cooperative_groups_{false};
  bool need_curand_kernel_h_{false};
  bool need_cluster_h_{false};

  // Barrier 配置
  const std::string barrier_name_ = "barrier";
  int barrier_count_ = -1;
  const std::string mbarrier_name_ = "mbarrier";
  const std::string mbarrier_dtype_ = "Barrier";
  const int barrier_alignment_bytes_ = 16;

  // Fragment 和 FP4 相关映射
  std::unordered_map<const VarNode *, std::string> fragment_shapes;
  std::unordered_map<const VarNode *, std::string> fragment_layouts;
  std::unordered_map<const VarNode *, IntImm> unroll_factor;
  std::unordered_map<const VarNode *, std::string> fp4_packed_buffers_;
};
```

### 2.2 函数前缀与启动配置

```cpp
// src/target/codegen_cuda.cc:401-446
void CodeGenTileLangCUDA::PrintFuncPrefix(std::ostream &os) {
  os << "extern \"C\" __global__ ";
}

class LaunchConfigExtractor : public tir::StmtVisitor {
private:
  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->var->name_hint == "threadIdx.x" ||
          iv->thread_tag == "threadIdx.x") {
        threadIdx_x_ext = op->value;
      } else if (iv->var->name_hint == "threadIdx.y" ||
                 iv->thread_tag == "threadIdx.y") {
        threadIdx_y_ext = op->value;
      } else if (iv->var->name_hint == "threadIdx.z" ||
                 iv->thread_tag == "threadIdx.z") {
        threadIdx_z_ext = op->value;
      }
    }
    StmtVisitor::VisitStmt_(op);
  }

public:
  PrimExpr threadIdx_x_ext = Integer(1);
  PrimExpr threadIdx_y_ext = Integer(1);
  PrimExpr threadIdx_z_ext = Integer(1);
};

void CodeGenTileLangCUDA::PrintExtraAttrs(const PrimFunc &f) {
  LaunchConfigExtractor extractor;
  extractor(f->body);
  arith::Analyzer analyzer;
  PrimExpr threadIdx_ext =
      analyzer.Simplify(extractor.threadIdx_x_ext * extractor.threadIdx_y_ext *
                        extractor.threadIdx_z_ext);
  if (const IntImmNode *const threadIdx_ext_int =
          threadIdx_ext.as<IntImmNode>()) {
    if (threadIdx_ext_int->value == 1) {
      return;
    }
    stream << " __launch_bounds__(" << threadIdx_ext_int->value << ", 1)";
  }
}
```

### 2.3 类型打印系统

CUDA 代码生成器支持多种数据类型，包括浮点、整数、向量类型和特殊类型：

```cpp
// src/target/codegen_cuda.cc:544-800 (节选)
void CodeGenTileLangCUDA::PrintType(DataType t, std::ostream &os) {
  int lanes = t.lanes();
  if (t.is_handle()) {
    ICHECK(t.is_scalar()) << "do not yet support vector types";
    os << "void*";
    return;
  }

  if (t.is_void()) {
    os << "void";
    return;
  }

  if (t == tl::cuTensorMapType()) {
    os << "CUtensorMap";
    return;
  }

  // 浮点类型处理
  if (t.is_float()) {
    switch (t.bits()) {
    case 16:
      enable_fp16_ = true;
      if (t.is_scalar()) {
        os << "half_t";
      } else if (lanes <= 8) {
        // half4 存储为 uint2
        ICHECK_EQ(lanes % 2, 0) << "only support even lane for half type";
        os << "uint" << lanes / 2;
      } else if (lanes <= 16) {
        ICHECK_EQ(lanes % 4, 0);
        os << "ulonglong" << lanes / 4;
      }
      break;
    case 32:
      if (lanes <= 4) {
        os << "float";
      } else if (lanes <= 8) {
        ICHECK_EQ(lanes % 2, 0);
        os << "ulonglong" << lanes / 2;
      }
      break;
    case 64:
      os << "double";
      break;
    }
  }
  // ... BF16, FP8, FP6, FP4, 整数类型等处理
}
```

### 2.4 FP8/FP6/FP4 特殊类型支持

```cpp
// src/target/codegen_cuda.cc:112-208
std::string GetTileLangFP8Type(DataType type) {
  std::stringstream stream;
  int32_t lanes = type.lanes();
  std::string vec;
  if (type.is_scalar()) {
    vec = "";
  } else if (lanes == 2) {
    vec = "_2";
  } else if (lanes == 4) {
    vec = "_4";
  } else if (lanes == 8) {
    vec = "_8";
  } else if (lanes == 16) {
    vec = "_16";
  } else if (lanes == 32) {
    vec = "_32";
  } else {
    LOG(FATAL) << "Only support scalar and vector types of width (2, 4, 8, 16, 32) for FP8";
  }
  if (type.is_float8_e4m3() || type.is_float8_e4m3fn()) {
    stream << "fp8_e4" << vec << "_t";
  } else if (type.is_float8_e5m2()) {
    stream << "fp8_e5" << vec << "_t";
  } else if (type.is_float8_e8m0fnu()) {
    stream << "fp8_e8" << vec << "_t";
  } else {
    LOG(FATAL) << "Unsupported FP8 type in CUDA codegen but got " << type;
  }
  return stream.str();
}

std::string GetTileLangFP6Type(DataType type) {
  std::stringstream stream;
  int32_t lanes = type.lanes();
  std::string vec;
  if (type.is_scalar()) {
    vec = "";
  } else if (lanes == 2) {
    vec = "x2";
  } else if (lanes == 4) {
    vec = "x4";
  } else if (lanes == 8) {
    vec = "x8";
  } else if (lanes == 16) {
    vec = "x16";
  } else {
    LOG(FATAL) << "Only support scalar and vector types of width (2, 4) for FP6";
  }
  stream << "__nv_fp6";
  std::string suffix;
  if (type.code() == DataType::kFloat6_e2m3fn) {
    suffix = "_e2m3";
  } else if (type.code() == DataType::kFloat6_e3m2fn) {
    suffix = "_e3m2";
  }
  stream << vec << suffix;
  return stream.str();
}

std::string GetTileLangFP4Type(DataType type) {
  std::stringstream stream;
  int32_t lanes = type.lanes();
  std::string vec;
  if (type.is_scalar()) {
    vec = "";
  } else if (lanes == 2) {
    vec = "_2";
  } else if (lanes == 4) {
    vec = "_4";
  } else if (lanes == 8) {
    vec = "_8";
  } else if (lanes == 16) {
    vec = "_16";
  } else if (lanes == 32) {
    vec = "_32";
  } else if (lanes == 64) {
    vec = "_64";
  } else {
    LOG(FATAL) << "Only support scalar and vector types of width (2, 4, 8, 16, 32, 64) for FP4";
  }

  std::string suffix;
  if (type.code() == DataType::kFloat4_e2m1fn) {
    suffix = "_e2";
  }

  stream << "fp4" << suffix << vec << "_t";
  return stream.str();
}
```

### 2.5 头文件生成

```cpp
// src/target/codegen_cuda.cc:448-510
std::string CodeGenTileLangCUDA::Finish() {
  if (need_mma_h_) {
    decl_stream << "#include <mma.h>\n";
  }
  if (need_mma_instruction_h_) {
    decl_stream << "#include <tl_templates/cuda/instruction/mma.h>\n";
  }
  if (need_wgmma_instruction_h_) {
    decl_stream << "#include <tl_templates/cuda/instruction/wgmma.h>\n";
  }
  if (need_tcgen05mma_instruction_h_) {
    decl_stream << "#include <tl_templates/cuda/instruction/tcgen05mma.h>\n";
  }
  if (need_mma_sm70_instruction_h_) {
    decl_stream << "#include <tl_templates/cuda/instruction/mma_sm70.h>\n";
  }
  if (need_tcgen05_common_h_) {
    decl_stream << "#include <tl_templates/cuda/tcgen_05.h>\n";
  }
  if (enable_fp8_) {
    decl_stream << "#include <tl_templates/cuda/cuda_fp8.h>\n";
  }
  if (enable_fp4_) {
    decl_stream << "#include <tl_templates/cuda/cuda_fp4.h>\n";
  }
  if (need_math_constants_h_) {
    decl_stream << "#include <math_constants.h>\n";
  }
  if (need_cooperative_groups_) {
    decl_stream << "#include <cooperative_groups.h>\n";
  }
  if (need_cluster_h_) {
    decl_stream << "#include <tl_templates/cuda/cluster.h>\n";
  }
  if (need_curand_kernel_h_) {
    decl_stream << "#include <curand_kernel.h>\n";
  }

  decl_stream << "#include <tl_templates/cuda/gemm.h>\n";
  if (enable_sparse_gemm_) {
    decl_stream << "#include <tl_templates/cuda/gemm_sp.h>\n";
  }
  decl_stream << "#include <tl_templates/cuda/copy.h>\n";
  decl_stream << "#include <tl_templates/cuda/reduce.h>\n";
  decl_stream << "#include <tl_templates/cuda/ldsm.h>\n";
  decl_stream << "#include <tl_templates/cuda/threadblock_swizzle.h>\n";
  decl_stream << "#include <tl_templates/cuda/debug.h>\n";
  decl_stream << "#ifdef ENABLE_BF16\n";
  decl_stream << "#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>\n";
  decl_stream << "#endif\n";

  if (need_global_barrier_) {
    decl_stream << "__device__ unsigned " << vid_global_barrier_state_
                << " = 0;\n";
  }
  decl_stream << "\n";

  return CodeGenC::Finish();
}
```

## 3. HIP 后端

### 3.1 HIP 代码生成器架构

```cpp
// src/target/codegen_hip.h:20-96
class CodeGenTileLangHIP final : public CodeGenC {
public:
  CodeGenTileLangHIP();
  std::string Finish();

  void PrintFuncPrefix(std::ostream &os) final;
  void PrintExtraAttrs(const PrimFunc &f, std::ostream &os) final;
  void VisitStmt_(const ForNode *op) final;
  void PrintStorageSync(const CallNode *op) final;
  void PrintStorageScope(const std::string &scope, std::ostream &os) final;
  void PrintVecBinaryOp(const std::string &op, DataType t, PrimExpr lhs,
                        PrimExpr rhs, std::ostream &os) final;
  void PrintType(DataType t, std::ostream &os) final;
  void PrintVecElemLoad(const std::string &vec, DataType t, int i,
                        std::ostream &os) final;
  void PrintVecElemStore(const std::string &vec, DataType t, int i,
                         const std::string &value) final;
  void BindThreadIndex(const IterVar &iv) final;
  void PrintVecElemLoadExpr(DataType t, int i, const std::string &value,
                            std::ostream &os) final;
  std::string CastFromTo(std::string value, DataType from, DataType target) final;

  void VisitExpr_(const RampNode *op, std::ostream &os) final;
  void VisitExpr_(const BroadcastNode *op, std::ostream &os) final;
  void VisitExpr_(const FloatImmNode *op, std::ostream &os) final;
  void VisitExpr_(const CallNode *op, std::ostream &os) final;
  void VisitExpr_(const CastNode *op, std::ostream &os) final;
  void VisitStmt_(const AllocateNode *op) final;
  void VisitStmt_(const AttrStmtNode *op) final;

  void AddFunction(const PrimFunc &f);

protected:
  std::string GetBufferRef(DataType t, const BufferNode *buffer,
                           PrimExpr index) final;
  void PrintCallExtern(Type ret_type, ffi::String global_symbol,
                       const ffi::Array<PrimExpr> &args, bool skip_first_arg,
                       std::ostream &os) final;

private:
  void HandleVolatileLoads(const std::string &value, const BufferLoadNode *op,
                           std::ostream &os) final;
  bool IsScopePartOfType() const final { return false; }

  bool need_math_constants_h_{false};
  bool need_wmma_h_{false};
  bool enable_fp8_{false};
  int barrier_count_ = -1;
  bool need_mma_h_{false};
  bool need_cast_smem_ptr_to_int_{false};
  const std::string barrier_name_ = "barrier";
  const int barrier_alignment_bytes_ = 16;
};
```

### 3.2 HIP 类型系统

HIP 代码生成器处理 FP8 类型的方式：

```cpp
// src/target/codegen_hip.cc:22-52
static std::string GetFP8Type(DataType type) {
  std::stringstream stream;
  int32_t lanes = type.lanes();
  std::string vec;
  if (type.is_scalar()) {
    vec = "";
  } else if (lanes == 2) {
    vec = "_2";
  } else if (lanes == 4) {
    vec = "_4";
  } else if (lanes == 8) {
    vec = "_8";
  } else if (lanes == 16) {
    vec = "_16";
  } else {
    LOG(FATAL) << "Only support scalar and vector types of width (2, 4, 8, 16) for FP8";
  }
  if (type.is_float8_e4m3fn() || type.is_float8_e4m3fnuz() ||
      type.is_float8_e4m3() || type.code() == DataType::kFloat8_e4m3b11fnuz) {
    stream << "fp8_e4" << vec << "_t";
  } else if (type.is_float8_e5m2() || type.is_float8_e5m2fnuz() ||
             type.code() == DataType::kFloat8_e5m2) {
    stream << "fp8_e5" << vec << "_t";
  } else if (type.code() == DataType::kFloat8_e8m0fnu) {
    stream << "fp8_e8" << vec << "_t";
  } else {
    LOG(FATAL) << "Unsupported FP8 type in HIP codegen: " << type;
  }
  return stream.str();
}
```

## 4. 运行时模块

### 4.1 CUDA 运行时模块

```cpp
// src/target/rt_mod_cuda.cc:1-134
#include "codegen_cuda.h"
#include "runtime/cuda/cuda_module.h"
#include "runtime/meta_data.h"
#include "runtime/pack_args.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>

namespace tvm {
namespace codegen {

// 提取函数信息
static std::unordered_map<std::string, runtime::FunctionInfo>
ExtractFuncInfo(const IRModule &mod) {
  std::unordered_map<std::string, runtime::FunctionInfo> fmap;

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<tir::PrimFuncNode>())
        << "Can only lower IR Module with PrimFuncs";
    auto f = Downcast<tir::PrimFunc>(kv.second);

    runtime::FunctionInfo info;
    for (size_t i = 0; i < f->params.size(); ++i) {
      if (f->params[i]->dtype.is_handle()) {
        auto ptr = f->params[i]->type_annotation.as<PointerTypeNode>();
        if (ptr && ptr->storage_scope == "grid_constant") {
          info.arg_types.push_back(DataType(runtime::kDLGridConstant, 64, 1));
          continue;
        }
      }
      DataType dtype = f->params[i].dtype();
      if (dtype.is_bool())
        dtype = DataType::Int(32);
      info.arg_types.push_back(dtype);
    }
    // 处理启动参数标签
    if (f->HasNonzeroAttr(tl::attr::kHasGridSync)) {
      info.launch_param_tags.push_back(
          runtime::launch_param::kUseProgramaticDependentLaunch);
    }
    if (f->HasNonzeroAttr("use_cooperative_groups")) {
      info.launch_param_tags.push_back(
          runtime::launch_param::kUseCooperativeLaunch);
    }
    if (f->GetAttr<ffi::Array<Integer>>("cluster_dims").defined()) {
      info.launch_param_tags.push_back(runtime::launch_param::kClusterDimX);
      info.launch_param_tags.push_back(runtime::launch_param::kClusterDimY);
      info.launch_param_tags.push_back(runtime::launch_param::kClusterDimZ);
    }
    if (auto opt = f->GetAttr<ffi::Array<ffi::String>>(
            tir::attr::kKernelLaunchParams)) {
      for (const auto &tag : opt.value()) {
        if (tag != runtime::launch_param::kClusterDimX &&
            tag != runtime::launch_param::kClusterDimY &&
            tag != runtime::launch_param::kClusterDimZ) {
          info.launch_param_tags.push_back(tag);
        }
      }
    }
    auto global_symbol = f->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
    fmap[static_cast<std::string>(global_symbol.value())] = info;
  }
  return fmap;
}

// 构建 CUDA 模块
ffi::Module BuildTileLangCUDA(IRModule mod, Target target) {
  bool output_ssa = false;
  CodeGenTileLangCUDA cg;
  cg.Init(output_ssa);

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangCUDA: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch);
    cg.AddFunction(gvar, f);
  }

  std::string code = cg.Finish();
  // 后处理回调
  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_cuda_postproc")) {
    code = (*f)(code, target).cast<std::string>();
  }
  // 编译为 PTX 或 CUBIN
  std::string fmt = "ptx";
  std::string ptx;
  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_cuda_compile")) {
    tvm::transform::PassContext pass_ctx =
        tvm::transform::PassContext::Current();
    ptx = (*f)(code, target, pass_ctx->config).cast<std::string>();
    if (ptx[0] != '/')
      fmt = "cubin";
  } else {
    ICHECK(0);
  }
  return runtime::CUDAModuleCreate(ptx, fmt, ExtractFuncInfo(mod), code);
}

// 不编译的构建（仅生成代码）
ffi::Module BuildTileLangCUDAWithoutCompile(IRModule mod, Target target) {
  bool output_ssa = false;
  CodeGenTileLangCUDA cg;
  cg.Init(output_ssa);

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangCUDA: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch);
    cg.AddFunction(gvar, f);
  }

  std::string code = cg.Finish();
  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_cuda_postproc")) {
    code = (*f)(code, target).cast<std::string>();
  }
  return runtime::CUDAModuleCreate("ptx", "ptx", ExtractFuncInfo(mod), code);
}

// 注册构建函数
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("target.build.tilelang_cuda", BuildTileLangCUDA)
      .def("target.build.tilelang_cuda_without_compile",
           BuildTileLangCUDAWithoutCompile);
}

} // namespace codegen
} // namespace tvm
```

### 4.2 HIP 运行时模块

```cpp
// src/target/rt_mod_hip.cc:1-128
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include "codegen_hip.h"
#include "runtime/rocm/rocm_module.h"
#include <tvm/ffi/function.h>

namespace tvm {
namespace codegen {

static std::unordered_map<std::string, runtime::FunctionInfo>
ExtractFuncInfo(const IRModule &mod) {
  std::unordered_map<std::string, runtime::FunctionInfo> fmap;

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<tir::PrimFuncNode>())
        << "Can only lower IR Module with PrimFuncs";
    auto f = Downcast<tir::PrimFunc>(kv.second);

    runtime::FunctionInfo info;
    for (size_t i = 0; i < f->params.size(); ++i) {
      if (f->params[i]->dtype.is_handle()) {
        auto ptr = f->params[i]->type_annotation.as<PointerTypeNode>();
        if (ptr && ptr->storage_scope == "grid_constant") {
          info.arg_types.push_back(DataType(kTVMGridConstant, 64, 1));
          continue;
        }
      }
      DataType dtype = f->params[i].dtype();
      if (dtype.is_bool())
        dtype = DataType::Int(32);
      info.arg_types.push_back(dtype);
    }
    if (auto opt = f->GetAttr<ffi::Array<ffi::String>>(
            tir::attr::kKernelLaunchParams)) {
      for (const auto &tag : opt.value()) {
        info.launch_param_tags.push_back(tag);
      }
    }
    auto global_symbol = f->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
    fmap[static_cast<std::string>(global_symbol.value())] = info;
  }
  return fmap;
}

ffi::Module BuildTileLangHIP(IRModule mod, Target target) {
  bool output_ssa = false;
  CodeGenTileLangHIP cg;
  cg.Init(output_ssa);

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangHIP: Can only take PrimFunc";
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch);
    cg.AddFunction(f);
  }

  std::string code = cg.Finish();

  // 使用新的 FFI API 获取注册函数
  using ffi::Function;
  if (auto f = Function::GetGlobal("tilelang_callback_hip_postproc")) {
    code = (*f)(code, target).cast<std::string>();
  }

  std::string fmt = "ptx";
  std::string ptx;

  if (auto f = Function::GetGlobal("tilelang_callback_hip_compile")) {
    ptx = (*f)(code, target).cast<std::string>();
    if (ptx[0] != '/')
      fmt = "hsaco";
  } else {
    ICHECK(false) << "tilelang_callback_hip_compile is not set";
  }

  return ROCMModuleCreate(ptx, fmt, ExtractFuncInfo(mod), code, std::string());
}

ffi::Module BuildTileLangHIPWithoutCompile(IRModule mod, Target target) {
  bool output_ssa = false;
  CodeGenTileLangHIP cg;
  cg.Init(output_ssa);

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangHIP: Can only take PrimFunc";
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch);
    cg.AddFunction(f);
  }

  std::string code = cg.Finish();

  using ffi::Function;
  if (auto f = Function::GetGlobal("tilelang_callback_hip_postproc")) {
    code = (*f)(code, target).cast<std::string>();
  }

  return ROCMModuleCreate("ptx", "fmt", ExtractFuncInfo(mod), code,
                          std::string());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("target.build.tilelang_hip", BuildTileLangHIP)
      .def("target.build.tilelang_hip_without_compile",
           BuildTileLangHIPWithoutCompile);
}

} // namespace codegen
} // namespace tvm
```

## 5. 内建规则 (Intrinsic Rules)

### 5.1 CUDA 内建规则

```cpp
// src/target/intrin_rule_cuda.cc:1-161
namespace tvm {
namespace codegen {
namespace intrin {

struct CUDAMath {
  std::string operator()(DataType t, std::string name) const {
    if (t.is_float()) {
      switch (t.bits()) {
      case 64:
        return name;
      case 32:
        return name + 'f';
      case 16: {
        if (name == "fabs") {
          return "__habs";
        } else if (name == "round") {
          return "hrint";
        } else {
          return "h" + name;
        }
      }
      default:
        return "";
      }
    } else if (t.is_bfloat16()) {
      if (name == "fabs") {
        return "__habs";
      } else if (name == "round") {
        return "hrint";
      } else {
        return "h" + name;
      }
    } else if (t.is_int() || t.is_uint()) {
      switch (t.bits()) {
      case 32:
        return "__" + name;
      case 64:
        return "__" + name + "ll";
      default:
        return "";
      }
    }
    return "";
  }
};

struct CUDAFastMath : public CUDAMath {
  std::string operator()(DataType t, std::string name) const {
    if (t.is_float() && t.bits() == 32) {
      return "__" + name + 'f';
    } else {
      return CUDAMath::operator()(t, name);
    }
    return "";
  }
};

struct CUDAWarpIntrinsic {
  const Op operator()(DataType t, const Op &orig_op) const {
    if (orig_op.same_as(builtin::tvm_warp_shuffle())) {
      return Op::Get("tir.cuda.__shfl_sync");
    } else if (orig_op.same_as(builtin::tvm_warp_shuffle_up())) {
      return Op::Get("tir.cuda.__shfl_up_sync");
    } else {
      ICHECK(orig_op.same_as(builtin::tvm_warp_shuffle_down()));
      return Op::Get("tir.cuda.__shfl_down_sync");
    }
  }
};

static PrimExpr DispatchCUDAWarpActiveMask(const PrimExpr &e) {
  const CallNode *call = e.as<CallNode>();
  return Call(call->dtype, Op::Get("tir.cuda.__activemask"), call->args,
              call->annotations);
}

static PrimExpr DispatchCUDAIsFinite(const PrimExpr &e) {
  const CallNode *call = e.as<CallNode>();
  ICHECK(call != nullptr);
  ICHECK_EQ(call->args.size(), 1U);

  DataType arg_dtype = call->args[0].dtype();
  if (arg_dtype.is_float() &&
      (arg_dtype.bits() == 32 || arg_dtype.bits() == 64)) {
    Array<PrimExpr> new_args = {StringImm("isfinite"), call->args[0]};
    return Call(call->dtype, builtin::call_pure_extern(), new_args,
                call->annotations);
  }

  return e;
}

TVM_REGISTER_OP("tir.rsqrt")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic",
                               DispatchPureExtern<CUDAMath>);

TVM_REGISTER_OP("tir.isfinite")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchCUDAIsFinite);

} // namespace intrin
} // namespace codegen
} // namespace tvm
```

### 5.2 HIP 内建规则

```cpp
// src/target/intrin_rule_hip.cc:1-291
struct HIPMath {
  std::string operator()(DataType t, std::string name) const {
    if (t.is_float()) {
      switch (t.bits()) {
      case 64:
        return name;
      case 32:
        return name + 'f';
      case 16: {
        if (name == "fabs") {
          return "__habs";
        } else if (name == "round") {
          return "hrint";
        } else {
          return "h" + name;
        }
      }
      default:
        return "";
      }
    } else if (t.is_bfloat16()) {
      // BF16 处理与 CUDA 相同
    } else if (t.is_int() || t.is_uint()) {
      switch (t.bits()) {
      case 32:
        return "__" + name;
      case 64:
        return "__" + name + "ll";
      default:
        return "";
      }
    }
    return "";
  }
};

struct HIPWarpIntrinsic {
  const Op operator()(DataType t, const Op &orig_op) const {
    if (orig_op.same_as(builtin::tvm_warp_shuffle())) {
      return Op::Get("tir.hip.__shfl_sync");
    } else if (orig_op.same_as(builtin::tvm_warp_shuffle_up())) {
      return Op::Get("tir.hip.__shfl_up_sync");
    } else {
      ICHECK(orig_op.same_as(builtin::tvm_warp_shuffle_down()));
      return Op::Get("tir.hip.__shfl_down_sync");
    }
  }
};

// 注册 HIP 内建操作
TVM_REGISTER_OP("tir.floor")
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic",
                               DispatchPureExtern<HIPMath>);

TVM_REGISTER_OP("tir.exp")
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic",
                               DispatchPureExtern<HIPFastMath>);

TVM_REGISTER_OP("tir.tvm_warp_shuffle")
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic",
                               DispatchHIPShuffle<HIPWarpIntrinsic>);

// 注册低级别内置操作
TVM_REGISTER_OP("tir.hip.__shfl_sync")
    .set_num_inputs(4)
    .add_argument("mask", "Expr", "The thread mask.")
    .add_argument("var", "Expr", "The variable to sync.")
    .add_argument("lane", "Expr", "The source thread id.")
    .add_argument("width", "Expr", "The warp thread width, must be a power of 2.")
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "__shfl_sync")
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque))
    .set_attr<bool>("hip.need_warp_shuffle", true);
```

## 6. 模板系统

TileLang 使用模板系统来生成高效的 CUDA/HIP 代码。代码生成器会自动包含必要的模板头文件：

### 6.1 CUDA 模板头文件

| 头文件 | 用途 |
|--------|------|
| `<tl_templates/cuda/instruction/mma.h>` | MMA (Tensor Core) 指令 |
| `<tl_templates/cuda/instruction/wgmma.h>` | WGMMA (Hopper) 指令 |
| `<tl_templates/cuda/instruction/tcgen05mma.h>` | TCGEN05 MMA (Blackwell) 指令 |
| `<tl_templates/cuda/instruction/mma_sm70.h>` | Volta MMA 指令 |
| `<tl_templates/cuda/gemm.h>` | GEMM 模板 |
| `<tl_templates/cuda/gemm_sp.h>` | 稀疏 GEMM 模板 |
| `<tl_templates/cuda/copy.h>` | 内存拷贝模板 |
| `<tl_templates/cuda/reduce.h>` | 归约操作模板 |
| `<tl_templates/cuda/ldsm.h>` | 共享内存加载模板 |
| `<tl_templates/cuda/threadblock_swizzle.h>` | Threadblock Swizzle |
| `<tl_templates/cuda/debug.h>` | 调试工具 |

### 6.2 特殊数据类型头文件

```cpp
// FP8 支持
#include <tl_templates/cuda/cuda_fp8.h>

// FP4 支持
#include <tl_templates/cuda/cuda_fp4.h>

// BF16 回退支持
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif
```

## 7. 关键字保留系统

代码生成器保留 CUDA/HIP 关键字以避免命名冲突：

```cpp
// src/target/codegen_cuda.cc:219-399
void CodeGenTileLangCUDA::ReserveKeywordsAsUnique_() {
  CodeGenC::ReserveKeywordsAsUnique();
  name_supply_->ReserveName("max");
  name_supply_->ReserveName("min");
  name_supply_->ReserveName("isfinite");
  name_supply_->ReserveName("isinf");
  name_supply_->ReserveName("isnan");

  // 单精度数学函数
  name_supply_->ReserveName("acosf");
  name_supply_->ReserveName("acoshf");
  name_supply_->ReserveName("asinf");
  // ... 更多函数

  // 双精度数学函数
  name_supply_->ReserveName("acos");
  name_supply_->ReserveName("acosh");
  name_supply_->ReserveName("asin");
  // ... 更多函数
}
```

## 8. 线程索引绑定

```cpp
// src/target/codegen_cuda.cc:538-542
void CodeGenTileLangCUDA::BindThreadIndex(const IterVar &iv) {
  ICHECK(!var_idmap_.count(iv->var.get()));
  var_idmap_[iv->var.get()] =
      CastFromTo(iv->thread_tag, DataType::UInt(32), iv->var.dtype());
}
```

## 9. 存储同步

代码生成器处理 GPU 存储同步操作：

```cpp
void CodeGenTileLangCUDA::PrintStorageSync(const CallNode *op) {
  // 处理 __syncthreads, __threadfence 等同步操作
}
```

## 10. 总结

TileLang 的目标后端实现具有以下特点：

1. **统一架构**: CUDA 和 HIP 后端共享相似的代码结构和接口
2. **丰富的数据类型支持**: 支持 FP16, BF16, FP8, FP6, FP4, INT8 等多种数据类型
3. **模板化代码生成**: 使用模板系统生成高效的 GPU 代码
4. **灵活的编译流程**: 支持完整编译和仅代码生成两种模式
5. **内建函数映射**: 自动将 TVM 内建函数映射到目标平台的原生函数
6. **关键字保护**: 保留目标平台的关键字避免命名冲突
7. **启动配置提取**: 自动提取线程块配置生成 `__launch_bounds__` 属性
