# TileLang Runtime System

## Overview

TileLang's runtime system is responsible for compiling, lowering, and executing kernel programs on various hardware targets. This document describes the core runtime components including the lowering pipeline, code generation, and execution model.

## Lowering Pipeline Architecture

The lowering pipeline transforms high-level TileLang IR into target-specific executable code through multiple phases.

### Main Lowering Entry Point

The primary lowering function is defined in `/root/dev/vibe_dsl/tilelang/tilelang/engine/lower.py:224-277`:

```python
def lower(
    func_or_mod: tir.PrimFunc | tvm.IRModule,
    target: str | Target = "auto",
    target_host: str | Target | None = None,
    runtime_only=False,
    enable_host_codegen=False,
    enable_device_compile=False,
) -> CompiledArtifact:
```

This function orchestrates the entire compilation process:
1. Extracts kernel parameters from the function
2. Determines target device (CUDA, HIP, Metal, etc.)
3. Applies semantic checks
4. Runs lowering and legalization phases
5. Applies target-specific optimizations
6. Generates host and device code

### Two-Phase Lowering

The lowering process is divided into two main phases defined in `/root/dev/vibe_dsl/tilelang/tilelang/engine/phase.py`:

#### Phase 1: LowerAndLegalize (lines 152-216)

```python
def LowerAndLegalize(mod: IRModule, target: Target) -> IRModule:
```

This phase:
- Binds target information to the module (`tir.transform.BindTarget`)
- Normalizes negative indices (`LegalizeNegativeIndex`)
- Verifies parallel loop correctness (`VerifyParallelLoop`)
- Injects assumptions for the prover (`InjectAssumes`)
- Simplifies IR expressions (`Simplify`)
- Sets layouts for reducers (`LayoutReducer`)
- Infers memory layouts (`LayoutInference`)
- Lowers high-level tile operations (`LowerTileOp`)
- Lowers L2 persistent maps (`LowerL2Persistent`)
- Legalizes vectorized loops (`LegalizeVectorizedLoop`)
- Adds memory access safety checks (`LegalizeSafeMemoryAccess`)
- Lowers access pointer operations (`LowerAccessPtr`)

#### Phase 2: OptimizeForTarget (lines 219-316)

```python
def OptimizeForTarget(mod: IRModule, target: Target) -> IRModule:
```

This phase applies target-specific optimizations:
- Lowers shared barriers and TMEM (`LowerSharedBarrier`, `LowerSharedTmem`)
- Applies warp specialization for Hopper GPUs
- Plans and injects software pipelining
- Vectorizes loops
- Rewrites storage
- Splits host and device code
- Merges shared memory allocations

## Host-Device Splitting

A critical transformation is the host-device split implemented in `/root/dev/vibe_dsl/tilelang/src/transform/split_host_device.cc`.

### HostDeviceSplitter Class (lines 55-279)

The splitter traverses the AST to separate host and device code:

```cpp
class HostDeviceSplitter : public tir::StmtMutator {
  // Collects assume statements from host side
  Array<const tir::AttrStmtNode *> host_assumes_;

  // Splits device function into separate module
  tir::Stmt SplitDeviceFunc(tir::Stmt body, tvm::Target device_target);
};
```

Key operations:
1. Collects assume statements from host-side code
2. Identifies device regions via `tvm::attr::kTarget`
3. Creates new device functions with proper parameter mapping
4. Wraps device calls with error checking for CPU targets

### SplitHostDevice Pass (lines 319-356)

```cpp
tvm::transform::Pass SplitHostDevice() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext ctx) {
    // Creates separate device module
    IRModule device_mod = IRModule(Map<GlobalVar, BaseFunc>({}));

    // Processes each function in the module
    for (const auto &[gvar, base_func] : mod->functions) {
      func = ::tvm::tl::SplitHostDevice(std::move(func), &device_mod, var_supply);
    }

    // Updates module and converts to SSA form
    mod->Update(updates);
    mod->Update(device_mod);
    return tir::transform::ConvertSSA()(mod);
  };
}
```

## Code Generation

### Device Code Generation

Device code generation in `/root/dev/vibe_dsl/tilelang/tilelang/engine/lower.py:180-196`:

```python
def device_codegen(device_mod: tvm.IRModule, target: Target) -> tvm.IRModule:
    device_mod = tilelang.transform.LowerDeviceStorageAccessInfo()(device_mod)
    device_mod = tilelang.transform.LowerIntrin()(device_mod)
    device_mod = tir.transform.Simplify()(device_mod)
    device_mod = tilelang.transform.HoistBroadcastValues()(device_mod)

    if target.kind.name == "cuda":
        global_func = "target.build.tilelang_" + ("cutedsl" if "cutedsl" in target.keys else "cuda")
        device_mod = tvm.ffi.get_global_func(global_func)(device_mod, target)
    elif target.kind.name == "hip":
        device_mod = tvm.ffi.get_global_func("target.build.tilelang_hip")(device_mod, target)
    elif target.kind.name == "metal":
        device_mod = tvm.ffi.get_global_func("target.build.metal")(device_mod, target)
```

### Host Code Generation

Host code generation in `/root/dev/vibe_dsl/tilelang/tilelang/engine/lower.py:147-177`:

```python
def host_codegen(host_mod: tvm.IRModule, target_host: Target, target: Target | None = None) -> tvm.IRModule:
    host_mod = tir.transform.BindTarget(target_host)(host_mod)
    host_mod = tir.transform.FP8StorageLegalize()(host_mod)
    host_mod = tir.transform.BF16StorageLegalize()(host_mod)
    host_mod = tir.transform.LowerTVMBuiltin()(host_mod)
    host_mod = tir.transform.LowerCustomDatatypes()(host_mod)
    host_mod = tilelang.transform.LowerIntrin()(host_mod)
    host_mod = tilelang.transform.LowerDeviceStorageAccessInfo()(host_mod)
    host_mod = tir.transform.CombineContextCall()(host_mod)
    if target is not None and target.kind.name == "metal":
        host_mod = MarkHostMetalContext()(host_mod)
```

## CUDA Compilation Callback

The CUDA compilation callback in `/root/dev/vibe_dsl/tilelang/tilelang/engine/lower.py:58-111`:

```python
@tvm_ffi.register_global_func("tilelang_callback_cuda_compile", override=True)
def tilelang_callback_cuda_compile(code, target, pass_config=None):
    target_arch = nvcc.get_target_arch(nvcc.get_target_compute_version(target))
    arch = [f"-arch=sm_{target_arch}"]
    compile_format = "cubin"

    # Configuration options
    cfg = pass_config or {}
    enable_fast_math = bool(cfg.get(PassConfigKey.TL_ENABLE_FAST_MATH, False))
    ptxas_usage_level = cfg.get(PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL, None)
    verbose_ptxas_output = bool(cfg.get(PassConfigKey.TL_ENABLE_PTXAS_VERBOSE_OUTPUT, False))

    options = [
        "-std=c++17",
        "-I" + TILELANG_TEMPLATE_PATH,
        "-I" + CUTLASS_INCLUDE_DIR,
    ]

    # Compile to PTX/CUBIN
    ptx = nvcc.compile_cuda(code, compile_format, arch, options=options, verbose=verbose)
    return ptx
```

## Intrinsic Lowering

The intrinsic lowering pass in `/root/dev/vibe_dsl/tilelang/src/transform/lower_intrin.cc` converts high-level operations to device-specific intrinsics.

### IntrinInjecter Class (lines 42-407)

```cpp
class IntrinInjecter : public tvm::arith::IRMutatorWithAnalyzer {
  using FLowerGeneral = ffi::TypedFunction<PrimExpr(PrimExpr)>;

  // Injects FMA (fused multiply-add) instructions
  PrimExpr VisitExpr_(const AddNode *op) final {
    if (const MulNode *mb = op->b.as<MulNode>()) {
      return MakeFMA(mb->a, mb->b, op->a, op);
    }
    // ...
  }

  // Lowers floor division to native truncdiv
  PrimExpr VisitExpr_(const FloorDivNode *op) final {
    // Converts to right shift for power-of-2 divisors
    if (support_bitwise_op_ && is_const_power_of_two_integer(op->b, &shift)) {
      return op->a >> make_const(dtype, shift);
    }
    // ...
  }

  // Lowers floor modulo to native truncmod
  PrimExpr VisitExpr_(const FloorModNode *op) final {
    // Converts to masking for power-of-2 divisors
    if (support_bitwise_op_ && is_const_power_of_two_integer(op->b, &shift)) {
      int64_t mask = (static_cast<int64_t>(1) << static_cast<int64_t>(shift)) - 1;
      return op->a & make_const(dtype, mask);
    }
    // ...
  }
};
```

## Pass Configuration

Pass configuration keys are defined in `/root/dev/vibe_dsl/tilelang/tilelang/transform/pass_config.py`:

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

    # Debugging
    TL_AST_PRINT_ENABLE = "tl.ast_print_enable"
    TL_LAYOUT_VISUALIZATION_ENABLE = "tl.layout_visualization_enable"
    TL_DISABLE_DATA_RACE_CHECK = "tl.disable_data_race_check"
```

## Thread Synchronization

Thread synchronization passes ensure correct memory ordering:

```cpp
// From split_host_device.cc and other transforms
mod = tilelang.transform.ThreadSync("shared")(mod)      // Shared memory sync
mod = tilelang.transform.ThreadSync("shared.dyn")(mod)  // Dynamic shared memory sync
mod = tilelang.transform.ThreadSync("global")(mod)      // Global memory sync (optional)
```

## Software Pipelining

The software pipelining transform in `/root/dev/vibe_dsl/tilelang/src/transform/inject_pipeline.cc` automatically parallelizes producer-consumer loops to hide memory latency.

Key components:
- `BufferUsageCollector`: Identifies buffers used in pipeline loops
- `PipelineRewriter`: Transforms loops into pipelined form
- Support for multi-version buffers to enable concurrent read/write

## Warp Specialization

For NVIDIA Hopper (SM90+) GPUs, TileLang supports warp specialization:

```python
# From phase.py
def allow_warp_specialized(pass_ctx: PassContext | None = None, target: Target | None = None) -> bool:
    if not is_cuda_target(target) or not have_tma(target):
        return False
    disable_warp_specialized = pass_ctx.config.get("tl.disable_warp_specialized", False)
    return not disable_warp_specialized
```

When enabled:
1. `WarpSpecialized()` pass partitions warps into producer/consumer groups
2. `InjectTmaBarrier()` inserts TMA (Tensor Memory Accelerator) barriers
3. `InjectFenceProxy()` adds proxy fences for async operations
4. `AnnotateWarpGroupRegAlloc()` manages register allocation per warp group

## Summary

The TileLang runtime system provides a comprehensive compilation pipeline:

1. **Frontend Lowering**: Converts TileLang-specific constructs to standard TIR
2. **Layout Inference**: Determines optimal memory layouts for fragments and shared memory
3. **Optimization**: Applies target-specific optimizations including pipelining and warp specialization
4. **Host-Device Split**: Separates host and device code for heterogeneous execution
5. **Code Generation**: Produces executable code for the target backend
6. **Compilation**: Uses NVCC/Clang to compile to binary (CUBIN/HSACO)

This architecture enables TileLang to generate highly optimized kernels for various GPU architectures while maintaining a high-level Python frontend.
