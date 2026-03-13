# TileLang Transform System

## Overview

TileLang's transform system is a comprehensive compilation pipeline that progressively lowers high-level TileLang IR into target-specific executable code. The system consists of Python-level pass wrappers and C++-level transform implementations.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Python Layer                             │
│  tilelang/transform/__init__.py                             │
│  - Pass wrappers and factory functions                      │
│  - Phase orchestration (phase.py)                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    C++ Transform Layer                      │
│  src/transform/*.cc                                         │
│  - IR mutation and analysis passes                          │
│  - Lowering and optimization transforms                     │
└─────────────────────────────────────────────────────────────┘
```

## Python Transform Interface

Located at `/root/dev/vibe_dsl/tilelang/tilelang/transform/__init__.py`.

### Pass Factory Functions

Each transform is exposed through a Python factory function:

```python
def LayoutInference():
    """infer the fragment/shared memory layout

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LayoutInference()  # type: ignore
```

### Complete Pass Registry

| Pass Name | Description | Location |
|-----------|-------------|----------|
| `LayoutInference` | Infer memory layouts for fragments and shared memory | `__init__.py:41` |
| `LowerTileOp` | Lower high-level tile operations | `__init__.py:52` |
| `InjectSoftwarePipeline` | Inject software pipelining | `__init__.py:63` |
| `WarpSpecialized` | Apply warp specialization for Hopper | `__init__.py:239` |
| `InjectTmaBarrier` | Inject TMA barriers | `__init__.py:265` |
| `InjectFenceProxy` | Inject proxy fences | `__init__.py:276` |
| `SplitHostDevice` | Split host and device code | `__init__.py:342` |
| `LowerIntrin` | Lower intrinsics | `__init__.py:527` |
| `ThreadSync` | Insert thread synchronization | `__init__.py:163` |
| `VectorizeLoop` | Vectorize loops | `__init__.py:368` |
| `MergeIfStmt` | Merge if statements | `__init__.py:206` |
| `LoopUnswitching` | Hoist loop-invariant ifs | `__init__.py:217` |
| `StorageRewrite` | Rewrite storage allocation | `__init__.py:506` |
| `FlattenBuffer` | Flatten buffer dimensions | `__init__.py:430` |
| `ConfigIndexBitwidth` | Configure index bitwidth | `__init__.py:418` |
| `LegalizeNegativeIndex` | Legalize negative indices | `__init__.py:96` |
| `LegalizeSafeMemoryAccess` | Add memory safety checks | `__init__.py:298` |
| `LegalizeVectorizedLoop` | Legalize vectorized loops | `__init__.py:287` |
| `LowerAccessPtr` | Lower access pointer ops | `__init__.py:309` |
| `LowerSharedBarrier` | Lower shared barriers | `__init__.py:486` |
| `LowerSharedTmem` | Lower TMEM operations | `__init__.py:545` |
| `LowerThreadAllreduce` | Lower thread allreduce | `__init__.py:522` |
| `LowerLDGSTG` | Lower ldg/stg intrinsics | `__init__.py:580` |
| `LowerHopperIntrin` | Lower Hopper intrinsics | `__init__.py:130` |
| `LowerPTXAsyncCopy` | Lower PTX async copy | `__init__.py:379` |
| `LowerDeviceKernelLaunch` | Lower device kernel launch | `__init__.py:532` |
| `LowerDeviceStorageAccessInfo` | Lower storage access info | `__init__.py:403` |
| `MergeSharedMemoryAllocations` | Merge shared memory | `__init__.py:446` |
| `MakePackedAPI` | Create packed API | `__init__.py:320` |
| `AnnotateDeviceRegions` | Annotate device regions | `__init__.py:331` |
| `AnnotateReadOnlyParams` | Annotate read-only params | `__init__.py:353` |
| `AnnotateWarpGroupRegAlloc` | Annotate register allocation | `__init__.py:250` |
| `InjectAssumes` | Inject assumptions | `__init__.py:107` |
| `VerifyParallelLoop` | Verify parallel loops | `__init__.py:119` |
| `PipelinePlanning` | Plan software pipelines | `__init__.py:30` |
| `ClusterPlanning` | Plan cluster configuration | `__init__.py:19` |
| `MultiVersionBuffer` | Multi-version buffers | `__init__.py:228` |
| `IfStmtBinding` | Bind if statements | `__init__.py:195` |
| `PlanAndUpdateBufferAllocationLocation` | Plan buffer allocation | `__init__.py:491` |
| `OptimizeCPAsyncSync` | Optimize cp.async sync | `__init__.py:74` |
| `PersistThreadblock` | Persist threadblocks | `__init__.py:467` |
| `RewriteWgmmaSync` | Rewrite WGMMA sync | `__init__.py:152` |
| `UnrollLoop` | Unroll loops | `__init__.py:562` |
| `LayoutReducer` | Layout reduction | `__init__.py:550` |
| `EliminateStorageSyncForMBarrier` | Eliminate storage sync | `__init__.py:441` |
| `AlignDynamicSharedMemoryAllocations` | Align dynamic shared mem | `__init__.py:472` |
| `HoistNonRestrictParams` | Hoist non-restrict params | `__init__.py:502` |
| `FrontendLegalize` | Frontend legalization | `__init__.py:85` |

## Phase-Based Lowering

Located at `/root/dev/vibe_dsl/tilelang/tilelang/engine/phase.py`.

### Pre-Lower Semantic Check (lines 136-150)

```python
def PreLowerSemanticCheck(mod: IRModule) -> None:
    """
    Check whether the module is valid before lowering.
    Validation-only pipeline that does not modify the module.
    """
    # Print AST for debugging
    if should_enable_ast_print():
        tilelang.analysis.ASTPrinter()(mod)
    # Check invalid nested loops
    tilelang.analysis.NestedLoopChecker()(mod)
    # Check invalid symbolic T.Parallel + fragment access
    tilelang.analysis.FragmentLoopChecker()(mod)
```

### LowerAndLegalize Phase (lines 152-216)

The first major lowering phase:

```python
def LowerAndLegalize(mod: IRModule, target: Target) -> IRModule:
    mod = tir.transform.BindTarget(target)(mod)

    if should_force_let_inline():
        mod = tilelang.transform.LetInline()(mod)

    # Add wrapper for single buf store
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)

    # Normalize negative indices
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)

    # Verify parallel loop correctness
    if should_enable_race_check():
        mod = tilelang.transform.VerifyParallelLoop()(mod)

    # Inject assumptions for prover
    mod = tilelang.transform.InjectAssumes()(mod)

    # Simplify IR expressions
    mod = tilelang.transform.Simplify()(mod)

    # Set layouts for reducers
    mod = tilelang.transform.LayoutReducer()(mod)

    # Infer memory layouts
    mod = tilelang.transform.LayoutInference()(mod)

    # Visualize layouts if enabled
    LayoutVisual(mod)

    # Lower high-level tile operations
    mod = tilelang.transform.LowerTileOp()(mod)

    # Lower L2 persistent maps
    mod = tilelang.transform.LowerL2Persistent()(mod)

    # Decouple type cast vectorization
    mod = tilelang.transform.DecoupleTypeCast()(mod)

    # Legalize vectorized loops
    mod = tilelang.transform.LegalizeVectorizedLoop()(mod)

    # Add memory safety checks
    mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)

    # Lower access pointer operations
    mod = tilelang.transform.LowerAccessPtr()(mod)

    # Simplify again
    mod = tilelang.transform.Simplify()(mod)

    # Hoist non-restrict params
    mod = tilelang.transform.HoistNonRestrictParams()(mod)

    return mod
```

### OptimizeForTarget Phase (lines 219-316)

The second phase applies target-specific optimizations:

```python
def OptimizeForTarget(mod: IRModule, target: Target) -> IRModule:
    pass_ctx = tilelang.transform.get_pass_context()

    # Lower shared barriers and TMEM
    mod = tilelang.transform.LowerSharedBarrier()(mod)
    mod = tilelang.transform.LowerSharedTmem()(mod)

    # TMA and warp specialization path
    if allow_tma_lower(pass_ctx=pass_ctx, target=target):
        mod = tilelang.transform.IfStmtBinding()(mod)
        mod = tilelang.transform.MultiVersionBuffer()(mod)
        mod = tilelang.transform.WarpSpecialized()(mod)
        mod = tilelang.transform.InjectTmaBarrier()(mod)
        mod = tilelang.transform.PipelinePlanning()(mod)
        mod = tilelang.transform.InjectSoftwarePipeline()(mod)
        mod = tilelang.transform.LowerOpaqueBlock()(mod)
        if is_hopper(target):
            mod = tilelang.transform.RewriteWgmmaSync()(mod)
    else:
        # Non-TMA path
        mod = tilelang.transform.IfStmtBinding()(mod)
        mod = tilelang.transform.PlanAndUpdateBufferAllocationLocation()(mod)
        mod = tilelang.transform.PipelinePlanning()(mod)
        mod = tilelang.transform.InjectSoftwarePipeline()(mod)

    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.OptimizeCPAsyncSync()(mod)
    mod = tilelang.transform.Simplify()(mod)

    # Data type narrowing and buffer flattening
    mod = tir.transform.NarrowDataType(32)(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)

    # Vectorization and storage rewrite
    mod = tir.transform.Simplify()(mod)
    mod = tilelang.transform.VectorizeLoop(enable_vectorize=allow_vectorize(pass_ctx=pass_ctx))(mod)
    mod = tilelang.transform.StorageRewrite()(mod)

    # Loop optimizations
    mod = tilelang.transform.LoopUnswitching()(mod)
    mod = tilelang.transform.UnrollLoop()(mod)
    mod = tir.transform.RenormalizeSplitPattern()(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tir.transform.RemoveNoOp()(mod)
    mod = tir.transform.HoistIfThenElse()(mod)

    # Memory verification and annotation
    mod = tir.transform.VerifyMemory()(mod)
    mod = tir.transform.AnnotateEntryFunc()(mod)

    # Thread-level allreduce
    mod = tir.transform.InferFragment()(mod)
    mod = tilelang.transform.LowerThreadAllreduce()(mod)

    # Lower ldg/stg and Hopper intrinsics
    mod = tilelang.transform.LowerLDGSTG()(mod)
    mod = tilelang.transform.LowerHopperIntrin()(mod)

    # Global barrier synchronization (optional)
    if allow_global_thread_synchronization():
        mod = tilelang.transform.ThreadSync("global")(mod)

    # Host-device split
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)

    # Mark CUDA sync calls
    mod = tilelang.transform.MarkCudaSyncCalls(have_pdl(target))(mod)

    # Annotate read-only params
    mod = tilelang.transform.AnnotateReadOnlyParams()(mod)

    # Merge shared memory allocations
    enable_aggressive_merge = should_enable_aggressive_merge(pass_ctx=pass_ctx, target=target)
    mod = tilelang.transform.MergeSharedMemoryAllocations(enable_aggressive_merge=enable_aggressive_merge)(mod)

    # Inject fence proxy
    if allow_tma_and_warp_specialized(pass_ctx=pass_ctx, target=target):
        mod = tilelang.transform.InjectFenceProxy()(mod)
    else:
        if allow_fence_proxy(target=target):
            mod = tilelang.transform.InjectFenceProxy()(mod)

    # Thread synchronization
    mod = tilelang.transform.ThreadSync("shared")(mod)
    mod = tilelang.transform.ThreadSync("shared.dyn")(mod)

    # Merge if statements
    mod = tilelang.transform.MergeIfStmt()(mod)

    # Register allocation annotation
    if allow_tma_and_warp_specialized(pass_ctx=pass_ctx, target=target):
        mod = tilelang.transform.AnnotateWarpGroupRegAlloc()(mod)

    # Create packed API and finalize
    mod = tilelang.transform.MakePackedAPI()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.LowerDeviceKernelLaunch()(mod)

    # Persist threadblock
    mod = tilelang.transform.PersistThreadblock()(mod)

    return mod
```

## C++ Transform Implementations

### Layout Inference

Located at `/root/dev/vibe_dsl/tilelang/src/transform/layout_inference.cc`.

The layout inference pass determines optimal memory layouts for fragments and shared memory buffers.

```cpp
// ThreadBindingCollector collects thread extent information
class ThreadBindingCollector : public StmtExprVisitor {
  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      thread_binding_[iv->var.get()] = iv;
    }
    StmtExprVisitor::VisitStmt_(op);
  }
  std::unordered_map<const VarNode *, IterVar> thread_binding_;
};

// Layout inference result structure
struct LayoutInferenceResult {
  Map<Buffer, Layout> layout_map;
  Map<Buffer, IterVar> buffer_remap;
  Map<Var, Buffer> alloc_buffer_remap;
};
```

### Intrinsic Lowering

Located at `/root/dev/vibe_dsl/tilelang/src/transform/lower_intrin.cc`.

```cpp
class IntrinInjecter : public tvm::arith::IRMutatorWithAnalyzer {
  using FLowerGeneral = ffi::TypedFunction<PrimExpr(PrimExpr)>;

  // Constructor sets up target-specific lowering functions
  IntrinInjecter(arith::Analyzer *analyzer, std::string target,
                 std::string mtriple = "")
      : IRMutatorWithAnalyzer(analyzer) {
    // Register target-specific patterns
    patterns.push_back(target + ".FLowerIntrinsic");
    patterns.push_back(target + ".FLegalize");
    // ...
  }

  // FMA injection
  PrimExpr VisitExpr_(const AddNode *op) final {
    if (const MulNode *mb = op->b.as<MulNode>()) {
      return MakeFMA(mb->a, mb->b, op->a, op);
    }
    return IRMutatorWithAnalyzer::VisitExpr_(op);
  }

  // FloorDiv lowering
  PrimExpr VisitExpr_(const FloorDivNode *op) final {
    // Convert to right shift for power-of-2
    if (support_bitwise_op_ && is_const_power_of_two_integer(op->b, &shift)) {
      return op->a >> make_const(dtype, shift);
    }
    // Convert to truncdiv with correction
    return truncdiv(op->a, op->b) + correction;
  }
};
```

### Host-Device Splitting

Located at `/root/dev/vibe_dsl/tilelang/src/transform/split_host_device.cc`.

```cpp
class HostDeviceSplitter : public tir::StmtMutator {
  // Traverses AST and splits into host/device functions
  tir::Stmt VisitStmt_(const tir::AttrStmtNode *op) final {
    if (op->attr_key == tvm::attr::kTarget) {
      found_device_region_ = true;
      auto device_target = op->node.as<tvm::Target>().value().WithoutHost();
      return SplitDeviceFunc(op->body, device_target);
    }
    // Collect assume statements from host side
    else if (op->attr_key == tir::attr::tilelang_assume) {
      host_assumes_.push_back(op);
    }
    return tir::StmtMutator::VisitStmt_(op);
  }

  // Creates device function with proper parameter mapping
  tir::Stmt SplitDeviceFunc(tir::Stmt body, tvm::Target device_target) {
    // Analyze undefined variables
    tir::VarUseDefAnalyzer use_def;
    use_def(body);

    // Create new parameters for device function
    Array<tir::Var> params;
    Map<tir::Var, PrimExpr> var_remap;
    for (const auto &old_var : old_params) {
      tir::Var new_var(old_var->name_hint, old_var->type_annotation);
      params.push_back(new_var);
      var_remap.Set(old_var, new_var);
    }

    // Substitute variables in body
    body = tir::Substitute(body, var_remap);

    // Create device PrimFunc
    tir::PrimFunc device_func(params, body, kernel_ret_type);
    // Add attributes
    device_func = WithAttrs(std::move(device_func), device_attrs);

    // Add to device module
    (*device_mod_)->Add(kernel_symbol_global, device_func);

    // Return host-side call
    return tir::Evaluate(tir::Call(DataType::Void(), kernel_symbol_global, args));
  }
};
```

### Software Pipeline Injection

Located at `/root/dev/vibe_dsl/tilelang/src/transform/inject_pipeline.cc`.

```cpp
// Collects buffers used in pipeline loops
class BufferUsageCollector : public StmtExprVisitor {
  Array<Buffer> Collect(const Stmt &stmt) {
    this->VisitStmt(stmt);
    Array<Buffer> result;
    for (const auto &buffer : used_buffers_) {
      result.push_back(buffer);
    }
    return result;
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    AddBuffer(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    AddBuffer(op->buffer);
    StmtExprVisitor::VisitExpr_(op);
  }
};

// Main pipeline injection pass
Stmt InjectSoftwarePipeline(Stmt stmt, const Map<IterVar, Range> &dom_map,
                           int max_stage) {
  // Find annotated pipeline loops
  // Create prologue, steady-state, and epilogue
  // Insert pipeline stages with proper synchronization
}
```

## Pass Configuration

Located at `/root/dev/vibe_dsl/tilelang/tilelang/transform/pass_config.py`.

### Configuration Keys

```python
class PassConfigKey:
    # Warp specialization control
    TL_DISABLE_WARP_SPECIALIZED = "tl.disable_warp_specialized"
    TL_DISABLE_TMA_LOWER = "tl.disable_tma_lower"

    # Compilation options
    TL_ENABLE_FAST_MATH = "tl.enable_fast_math"
    TL_PTXAS_REGISTER_USAGE_LEVEL = "tl.ptxas_register_usage_level"
    TL_ENABLE_PTXAS_VERBOSE_OUTPUT = "tl.enable_ptxas_verbose_output"
    TL_DEVICE_COMPILE_FLAGS = "tl.device_compile_flags"

    # Memory optimization
    TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE = "tl.enable_aggressive_shared_memory_merge"

    # Debugging and visualization
    TL_AST_PRINT_ENABLE = "tl.ast_print_enable"
    TL_LAYOUT_VISUALIZATION_ENABLE = "tl.layout_visualization_enable"
    TL_LAYOUT_VISUALIZATION_FORMATS = "tl.layout_visualization_formats"
    TL_DISABLE_DATA_RACE_CHECK = "tl.disable_data_race_check"

    # Control flow
    TL_FORCE_LET_INLINE = "tl.force_let_inline"
    TL_ENABLE_AGGRESSIVE_LOOP_UNSWITCHING = "tl.enable_aggressive_loop_unswitching"
```

### Feature Detection Functions

```python
def allow_warp_specialized(pass_ctx: PassContext | None = None, target: Target | None = None) -> bool:
    """Check if warp specialization is enabled for the target."""
    if not is_cuda_target(target) or not have_tma(target):
        return False
    disable_warp_specialized = pass_ctx.config.get("tl.disable_warp_specialized", False)
    return not disable_warp_specialized

def allow_tma_lower(pass_ctx: PassContext | None = None, target: Target | None = None) -> bool:
    """Check if TMA lowering is enabled."""
    if not have_tma(target):
        return False
    disable_tma_lower = pass_ctx.config.get("tl.disable_tma_lower", False)
    return not disable_tma_lower

def allow_vectorize(pass_ctx: PassContext | None = None) -> bool:
    """Check if vectorization is enabled."""
    disable_vectorize = pass_ctx.config.get("tir.disable_vectorize", False)
    return not disable_vectorize
```

## Additional Python Transforms

### Simplification

Located at `/root/dev/vibe_dsl/tilelang/tilelang/transform/simplify.py`:

```python
def Simplify():
    """Simplify the IR using arithmetic analysis."""
    return _ffi_api.Simplify()

def simplify_prim_func(func: tir.PrimFunc) -> tir.PrimFunc:
    """Simplify a PrimFunc."""
    return _ffi_api.SimplifyPrimFunc(func)

def LetInline():
    """Inline Let expressions."""
    return _ffi_api.LetInline()
```

### Type Cast Decoupling

Located at `/root/dev/vibe_dsl/tilelang/tilelang/transform/decouple_type_cast.py`:

```python
def DecoupleTypeCast():
    """
    Decouple type cast vectorization constraints before vectorization.
    Separates type conversion from vectorized operations for better codegen.
    """
    return _ffi_api.DecoupleTypeCast()
```

### Broadcast Hoisting

Located at `/root/dev/vibe_dsl/tilelang/tilelang/transform/hoist_broadcast_values.py`:

```python
def HoistBroadcastValues():
    """
    Hoist broadcast values outside of loops when possible.
    Reduces redundant computations in generated code.
    """
    return _ffi_api.HoistBroadcastValues()
```

## Summary

The TileLang transform system provides:

1. **Python API**: Clean factory functions for creating transform passes
2. **Phase-Based Lowering**: Two-phase approach (LowerAndLegalize, OptimizeForTarget)
3. **C++ Implementations**: High-performance IR transformations
4. **Configurable Pipeline**: Pass context configuration for fine-tuning
5. **Target Specialization**: Different paths for CUDA, HIP, Metal backends
6. **Advanced Optimizations**: Warp specialization, software pipelining, TMA support

The transform system is the core compilation engine that enables TileLang to generate efficient GPU kernels from high-level Python descriptions.
