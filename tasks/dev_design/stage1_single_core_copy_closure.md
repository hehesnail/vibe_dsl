# Stage 1 Single-Core Copy 闭环设计

## 目标

- 打通 Blackhole 后端的最小 single-core copy 真执行路径。
- 让 `target_mode` 和最小 `kernels[].runtime_args` 不再完全由 `rt_mod_blackhole` 猜测。
- 让现有 `spec.json -> runner` 协议至少能稳定执行一个 TT-Sim 兼容的 copy kernel。

## 影响范围

- `tilelang_repo/src/transform/lower_blackhole_ops.cc`
- `tilelang_repo/src/target/rt_mod_blackhole.cc`
- `tilelang_repo/tools/blackhole_runner/runner.cpp`
- `tilelang_repo/testing/python/target/blackhole/test_blackhole_e2e.py`
- `tasks/progress.md`
- `memory/general_dev.md`

## 初始问题

- 现有 simple copy 没有被稳定 lower 成 TT-Metal 可执行语义，`LowerBlackholeOps` 会在 `global -> global` copy 上回退。
- `CodeGenBlackhole` 对这类 copy 最终仍会落到二维 `BufferStore`，当前直接在 codegen 阶段失败。
- `rt_mod_blackhole` 目前使用 `target_mode == single_core_copy` 的保守默认值来猜 `runtime_args`，这不足以支撑真实执行。
- runner 现在只会为输入/输出 DRAM buffer 组 runtime args，还不会按 schema 额外准备 L1 scratch buffer。

## 方案

### 1. single-core copy 先走最小专用 kernel path

对于当前 Stage 1，只要求跑通 TT-Sim 兼容的最小 copy kernel，不要求先把通用 copy lowering 彻底做好。

因此采取分层收敛：

- `LowerBlackholeOps` 负责识别“纯 copy PrimFunc”的最小模式，并明确写出：
  - `blackhole.target_mode = "single_core_copy"`
- `rt_mod_blackhole` 在 `single_core_copy` 模式下，生成一个稳定的 TT-Sim 兼容 copy kernel 源码，而不是依赖当前 generic codegen 处理二维 store。

这仍然符合总设计，因为：

- 后端主产物仍然是 `ExecutableSpec`
- runtime 仍然完全走 spec-driven runner
- 这里只是为 Stage 1 补一个最小可执行 kernel emitter，不回退到旧 CLI 协议

### 2. 显式扩充 single-core copy 的 runtime args schema

single-core copy kernel 运行时最少需要：

- `input_buffer_addr32`
- `output_buffer_addr32`
- `tile_count`
- `scratch_l1_buffer_addr32`

runner 需要按 schema 识别 `scratch_l1_buffer_addr32`，并在 host 侧创建一块最小 L1 buffer，把其地址作为 runtime arg 传给 kernel。

### 3. E2E 测试切到 spec-driven 路径

Python E2E 测试不再直接按旧 CLI 传：

- `kernel.cpp`
- `input.bin`
- `output.bin`
- `sizes`

而是应通过 TileLang 编译产物实际调用 `BlackholeModule`，或者至少按新的 `spec.json + input.bin + output.bin` 协议驱动 runner。

当前优先目标是把测试切到新协议，不再保留旧 runner CLI 假设。

## 协议变化

### 新增/明确的 runtime arg kind

- `input_buffer_addr32`
- `output_buffer_addr32`
- `tile_count`
- `scratch_l1_buffer_addr32`

### `target_mode`

- single-core copy 不再只是 `AssignBlackholeCores` 的默认占位值
- 当检测到纯 copy PrimFunc 时，由 lowering 明确写出该模式

## 验证方式

- 编译验证：
  - `make -C tilelang_repo/build -j16`
  - `./scripts/build_blackhole_runner.sh`
- 协议验证：
  - `spec.json` 中出现 single-core copy 对应 runtime args schema
- 执行验证：
  - TT-Sim 下跑通 TileLang single-core copy
- 文档验证：
  - 更新 `tasks/progress.md`
  - 记录稳定经验与真实问题

## 当前边界

- 已完成：
  - `LowerBlackholeOps` 为纯 copy PrimFunc 写出 `blackhole.target_mode = "single_core_copy"`
  - `rt_mod_blackhole` 在 `single_core_copy` 模式下生成最小 copy kernel，并输出 32-bit copy runtime arg schema
  - runner 支持 `scratch_l1_buffer_addr32`
  - Python E2E 已切到 `spec.json + input.bin + output.bin` 新协议
  - TT-Sim 下已跑通 `32x32 float16` single-core copy，输出与 reference 一致
- 当前未完成：
  - `BlackholeModule` 从 Python 直接调 packed func 时仍会在 `ExecuteExternal` 路径崩溃
  - 通用 copy lowering、multi-kernel segmentation、multi-core runtime args 仍未覆盖
  - GEMM 仍留到后续阶段推进
