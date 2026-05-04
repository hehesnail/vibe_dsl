# TileLang Source Analysis

This directory contains background analysis notes for the TileLang codebase.
It is a reference index, not the source of truth for active Blackhole backend
work.

For active Blackhole design and status, read:

- `../../tasks/dev_design/final_blackhole_backend_redesign.md`
- `../../tasks/dev_design/README.md`
- `../../tasks/progress.md`

## Index

### Python Core

- [Package entry and initialization](./python_core/01_package_init.md)
- [Kernel and frame](./python_core/03_language_core/03a_kernel_and_frame.md)
- [Builtin operations](./python_core/03_language_core/03b_builtin_ops.md)
- [Memory operations](./python_core/03_language_core/03c_memory_ops.md)
- [Compute operations](./python_core/03_language_core/03d_compute_ops.md)
- [Control flow](./python_core/03_language_core/03e_control_flow.md)
- [Data types and proxy](./python_core/03_language_core/03f_data_types_and_proxy.md)
- [AST and parser](./python_core/03_language_core/03g_ast_and_parser.md)
- [Transform system](./python_core/04_transform.md)

### JIT

- [JIT core](./jit/01_jit_core.md)
- [TVM FFI adapter](./jit/02_tvm_ffi_adapter.md)
- [CuTeDSL adapter](./jit/03_cutedsl_adapter.md)
- [NVRTC adapter](./jit/04_nvrtc_adapter.md)
- [Cython adapter](./jit/05_cython_adapter.md)

### C++ Core

- [IR definition](./cpp_core/01_ir_definition.md)
- [Copy and memory operations](./cpp_core/02_operations/02a_copy_and_memory.md)
- [GEMM operations](./cpp_core/02_operations/02b_gemm_operations.md)
- [Reduction and atomic operations](./cpp_core/02_operations/02c_reduction_and_atomic.md)
- [Builtins and math](./cpp_core/02_operations/02d_builtin_and_math.md)
- [Layout system](./cpp_core/03_layout_system.md)
- [Target backends](./cpp_core/04_target_backends.md)
- [Runtime](./cpp_core/05_runtime.md)
- [Templates](./cpp_core/06_templates.md)

### Advanced Topics

- [Auto differentiation](./advanced/01_autodd.md)
- [Autotuner](./advanced/02_autotuner.md)
- [Profiler](./advanced/03_profiler.md)
- [Quantization](./advanced/04_quantization.md)
- [Analysis tools](./advanced/05_analysis.md)
- [Carver](./advanced/06_carver.md)

### Examples And Tests

- [Basic examples](./examples/01_basic_examples.md)
- [FlashAttention](./examples/02_flash_attention.md)
- [Linear attention](./examples/03_linear_attention.md)
- [DeepSeek MLA](./examples/04_deepseek_mla.md)
- [Dequantize GEMM](./examples/05_dequantize_gemm.md)
- [Other examples](./examples/06_other_examples.md)
- [Testing](./testing.md)
- [Benchmarks](./benchmark.md)
- [Analysis completion note](./ANALYSIS_COMPLETE.md)

## Maintenance

- Do not maintain current task state, file counts, or backend-support status
  here.
- Update the active Blackhole task design when these notes affect backend
  contracts.
