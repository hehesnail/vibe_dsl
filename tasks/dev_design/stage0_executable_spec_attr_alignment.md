# Stage 0 ExecutableSpec 与 Attr 对齐设计

## 目标

- 为 Blackhole 后端引入可落地的 `ExecutableSpec` 数据结构骨架。
- 将 `rt_mod_blackhole` 从旧的 `tl.blackhole_*` attr 读取逻辑切换到 `blackhole.*`。
- 让 `AssignBlackholeCores` 直接产出 `blackhole.core_plan` 与 `blackhole.target_mode`，收敛 Stage 0 协议。

## 影响范围

- `tilelang_repo/src/target/blackhole_module.h`
- `tilelang_repo/src/target/rt_mod_blackhole.cc`
- `tilelang_repo/src/transform/assign_blackhole_cores.cc`
- `tilelang_repo/tools/blackhole_runner/`
- `tasks/progress.md`
- `memory/general_dev.md`

## 协议与接口变化

### 1. Runtime 主数据结构

新增以下结构，作为 `BlackholeModule` 的主协议骨架：

- `KernelArgSpec`
- `KernelSpec`
- `CorePlan`
- `ExecutableSpec`

当前阶段先保证字段可承载 Stage 0 所需信息：

- entry name
- target mode
- CB configs
- core plan
- kernels
- TVM 调用侧参数类型与 buffer/scalar 标记
- 外部 runner 所需的最小 JSON 序列化字段

### 2. Attr 读取协议

`rt_mod_blackhole` 统一改为读取：

- `blackhole.cb_configs`
- `blackhole.core_plan`
- `blackhole.target_mode`

不再把以下旧 attr 作为主路径：

- `tl.blackhole_cb_config`
- `tl.blackhole_kernel_split`

### 3. 外部 runner 协议

`BlackholeModule` 改为写出：

- `spec.json`
- `input.bin`
- `output.bin`
- `*.cpp` kernel 源文件

当前 `spec.json` 最小字段：

- `entry_name`
- `target_mode`
- `input_size_bytes`
- `output_size_bytes`
- `scalar_args`
- `cb_configs`
- `core_plan`
- `kernels[].kernel_path`
- `kernels[].compile_time_args`
- `kernels[].runtime_args`

runner 源码与构建入口归属：

- runner 放在 `tilelang_repo/tools/blackhole_runner/`
- runner 由 `tilelang_repo/tools/blackhole_runner/CMakeLists.txt` 独立构建
- runner 二进制默认产物位置收敛到 `tilelang_repo/build-blackhole-runner/tilelang_blackhole_runner`
- 顶层总控入口为 `scripts/build_blackhole_stack.sh`，显式串起 TileLang 构建、TT-Metal bootstrap、TT-Sim smoke test 和 runner 构建
- `scripts/build_blackhole_runner.sh` 会先 bootstrap `TT_METAL_HOME/build_Release`，再构建 runner
- bootstrap 后会通过 TT-Sim 运行 `metal_example_add_2_integers_in_riscv` 作为 smoke test
- runner 构建只消费 `TT_METAL_HOME/build_Release` 的头文件和库，不再要求修改 `tt_metal_repo` 源码
- 不再以 `tt_metal_repo/tt_metal/programming_examples/tilelang_blackhole_runner/` 作为主维护位置

### 4. Core assignment 输出协议

`AssignBlackholeCores` 除保留兼容性标量 attr 外，新增：

- `blackhole.core_plan`
- `blackhole.target_mode`

其中：

- 默认 `target_mode = "single_core_copy"`，仅作为 Stage 0/MVP 协议占位
- `core_plan` 写出 `grid_x / grid_y / cores_needed / work_per_core / core_grid_x / core_grid_y`

## 验证方式

- 编译级验证：确认 `rt_mod_blackhole.cc` 与 `blackhole_module.h` 类型一致。
- 代码级验证：检查 pass 输出 attr 名与 runtime extractor 读取名称一致。
- runner 级验证：确认 runner 能编译并消费 `spec.json`。
- 环境级验证：确认 `scripts/setup_tt_sim.sh` 能设好 `TT_METAL_RUNTIME_ROOT` 并在 TT-Sim 上跑通 `metal_example_add_2_integers_in_riscv`。
- 文档级验证：更新 `tasks/progress.md`，使阶段状态与当前实现一致。

## 当前限制

- `ExecutableSpec` 虽已接到 runner JSON 协议，但当前仍只可靠覆盖单核、单输入、单输出的最小路径。
- `kernels[].runtime_args` 目前由 `rt_mod_blackhole` 按 `target_mode` 提供保守默认 schema，尚未由 lowering 精确生成。
- `target_mode` 当前采用保守默认值，后续需由 lowering/segment 规划精化。
