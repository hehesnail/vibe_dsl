# Stage 0 ExecutableSpec 与 Attr 对齐设计

## 目标

- 为 Blackhole 后端引入可落地的 `ExecutableSpec` 数据结构骨架。
- 将 `rt_mod_blackhole` 从旧的 `tl.blackhole_*` attr 读取逻辑切换到 `blackhole.*`。
- 让 `AssignBlackholeCores` 直接产出 `blackhole.core_plan` 与 `blackhole.target_mode`，收敛 Stage 0 协议。

## 影响范围

- `tilelang_repo/src/target/blackhole_module.h`
- `tilelang_repo/src/target/rt_mod_blackhole.cc`
- `tilelang_repo/src/transform/assign_blackhole_cores.cc`
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

### 2. Attr 读取协议

`rt_mod_blackhole` 统一改为读取：

- `blackhole.cb_configs`
- `blackhole.core_plan`
- `blackhole.target_mode`

不再把以下旧 attr 作为主路径：

- `tl.blackhole_cb_config`
- `tl.blackhole_kernel_split`

### 3. Core assignment 输出协议

`AssignBlackholeCores` 除保留兼容性标量 attr 外，新增：

- `blackhole.core_plan`
- `blackhole.target_mode`

其中：

- 默认 `target_mode = "single_core_copy"`，仅作为 Stage 0/MVP 协议占位
- `core_plan` 写出 `grid_x / grid_y / cores_needed / work_per_core / core_grid_x / core_grid_y`

## 验证方式

- 编译级验证：确认 `rt_mod_blackhole.cc` 与 `blackhole_module.h` 类型一致。
- 代码级验证：检查 pass 输出 attr 名与 runtime extractor 读取名称一致。
- 文档级验证：更新 `tasks/progress.md`，使阶段状态与当前实现一致。

## 当前限制

- 本次不重写 runner 的文件协议，runner 仍是旧 CLI 路径。
- `ExecutableSpec` 暂时仍只承载单段 kernel 源码，不实现多 kernel/per-core runtime args。
- `target_mode` 当前采用保守默认值，后续需由 lowering/segment 规划精化。
