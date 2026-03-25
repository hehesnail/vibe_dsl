# Stage 2 Blackhole Logical Block / Launch Plan 设计

## 基本定位

- **状态**: Stage 2 支撑设计（仍有效，当前主任务不再以本文件为唯一入口）
- **前置总体设计**: `final_blackhole_backend_redesign.md`
- **相关阶段设计**:
  - `stage2_pass_reuse_matrix.md`
  - `stage2_single_core_pass_integration.md`

本文件只处理一个问题：

- **如何把 TileLang 现有的逻辑 block 语义，落实成 Blackhole 的 execution plan、memory plan 与 host launch materialization**

## 背景与结论

### 1. `T.Kernel` 已经定义了逻辑 block/grid 语义

TileLang 前端里，`T.Kernel(grid_x, grid_y, ..., threads=...)` 会直接生成：

- `blockIdx.x/y/z`
- `threadIdx.x/y/z`

这说明 DSL 程序中“每个 block 处理哪块数据”的语义，本来就在 TIR 中，而不是 host/runtime 侧再猜。

### 2. CUDA 路径验证了正确分层

CUDA/Hopper 路径说明：

- split 前更偏：
  - execution / memory / pipeline 语义规划
- split 后更偏：
  - host/device ABI、kernel launch 边界、最终 materialization-friendly attrs

因此 Blackhole 也应遵循同样分层：

- **split 前** 保留并规划逻辑 block / memory/dataflow 语义
- **split 后** 提取正式 `core_plan` / `cb_configs` / `runtime_args`
- **host side** 只 materialize 这些计划

### 3. TT-Metal host path 是正式落地方

TT-Metal 的 host-side 主抽象明确是：

- `Program`
- `CreateCircularBuffer`
- `CreateKernel`
- `SetRuntimeArgs`
- `ConfigureDeviceWithProgram`
- `LaunchProgram`

因此 Blackhole 的正式 execution / memory / launch plan 必须最终服务于 `BlackholeModule` direct host path，而不是 external runner。

## 分层设计

### A. Logical block semantics

这层来自 DSL/TIR，本轮不重新发明协议。

包含：

- `blockIdx.x/y/z`
- `threadIdx.x/y/z`
- block 对输入/输出 tile 的索引关系
- block 级 pipeline / dataflow 结构

这层语义必须继续保留到 split 前 Blackhole planning 可见的阶段，不能在 codegen 前被提前折叠掉。

### B. Blackhole execution planning

这是 split 后正式 plan 的一部分。

它需要表达：

- logical grid shape
- logical block linearization
- `current_work_linear_id`
- logical block -> physical core / work packet 映射

当前 single-core 最小正确模型：

- `physical_core_count = 1`
- `logical_block_count = grid_x * grid_y`
- 一个 physical core 顺序处理多个 logical blocks
- 当前 logical block 由 work packet / runtime arg 提供，而不是由 codegen 常量化

### C. Blackhole memory planning

这是 split 后正式 plan 的另一部分。

它需要表达：

- kernel-level memory objects
- segment 与 CB 的使用关系
- `page_size_bytes`
- `num_pages`
- role / data format
- 生命周期与复用
- 单核 per-core L1 峰值

当前 worker L1 约束：

- `1572864` bytes

### D. Host/runtime materialization

这层只负责：

- `Program`
- `CreateCircularBuffer`
- `CreateKernel`
- `SetRuntimeArgs`
- `ConfigureDeviceWithProgram`
- `LaunchProgram`
- readback

它不重新解释 DSL/TIR 语义，只 materialize 已规划好的 plan。

对当前 single-core 最小模型，host/runtime 还应显式承担：

- 顺序消费 `work_packets`
- 对每个 logical block 传入对应的 `current_work_linear_id`

也就是说，single-core 的“多个 logical blocks”不能只停留在 metadata 上；host materialization 必须真的把它们逐个执行完。

## 单核最小正确模型

当前阶段只要求 single-core，但 single-core 的正确模型不应等于“把 `blockIdx=0` 写死”。

正确的最小模型应为：

- `physical_core_count = 1`
- `logical_grid_x / logical_grid_y` 保留
- `current_work_linear_id = blockIdx.y * logical_grid_x + blockIdx.x`
- 一个 physical core 顺序处理 `[0, logical_block_count)` 中的一段或全部 work

### 单核 work packet 最小语义

单核最小 execution plan 至少需要表达：

- `logical_grid_x`
- `logical_grid_y`
- `linearization = row_major`
- `physical_cores = [{core_x, core_y}]`
- `work_packets = [{core_x, core_y, work_offset, work_count}]`

## 对现有 pass 的职责调整

### `LowerBlackholeOps`

要求：

- 不再默认 `blockIdx.x/y` 最终会在 codegen 中被抹成常量
- runtime arg / kernel ABI 设计要允许传入 `current_work_linear_id`
- staged copy 的 tile-index 恢复不能在 split 后 matcher 里无条件把 `blockIdx.x/y` 归零；
  局部 element loop 可以清零，但逻辑 block 坐标必须保留到最终 tile-index 公式
- `segment_plan` 与 logical work / memory object 使用关系对齐

输出：

- `blackhole.segment_plan`
- `blackhole.runtime_args`
- `blackhole.cb_requirements`

### `PlanBlackholeCB`

要求：

- memory plan 必须与 execution plan 结合
- 输出应是正式 memory objects，而不是仅做 `cb_id` allocator
- 进行：
  - deterministic `cb_id` 分配
  - role + lifetime 复用
  - `1572864` bytes hard check

输出：

- `blackhole.cb_configs`

### `AssignBlackholeCores`

要求：

- 从逻辑 grid 中生成可执行的 work distribution
- 单核下显式生成：
  - logical grid shape
  - row-major linearization
  - per-core work packets
  - physical core coords

输出：

- `blackhole.core_plan`

### `SplitHostDevice` / `LowerDeviceKernelLaunch`

本轮仍不要求它们承担 block planning。

它们继续负责：

- host/device ABI 边界
- launch materialization 边界

Blackhole 后续基于这层正式 ABI，把 split 后 device kernel 的 plan 接到 `BlackholeModule` host materialization。

## 协议与接口变化

### 1. `blackhole.core_plan`

后续必须至少表达：

- `logical_grid_x`
- `logical_grid_y`
- `linearization`
- `physical_cores`
- `work_packets`

不再接受只包含：

- `grid_x`
- `grid_y`
- `cores_needed`
- `work_per_core`

### 2. `blackhole.cb_configs`

后续必须至少表达：

- `cb_id`
- `role`
- `page_size_bytes`
- `num_pages`
- `data_format`
- `total_size_bytes`

并隐含或显式反映：

- 生命周期与复用结果
- per-core memory plan 合法性

### 3. `ExecutableSpec`

保持现有协议名，但补足：

- `CorePlan` 的 work packet 语义
- `CBConfig` 的正式 memory materialization 语义
- `KernelSpec` 的 kernel-level ABI / processor role / runtime args

## 实现顺序

### 第一步：split 前保住逻辑 block 与 memory/dataflow 语义

- 不再让这些语义只靠 split 后 matcher 恢复

### 第二步：split 后形成正式 `runtime_args / cb_configs / core_plan`

- `current_work_linear_id`
- per-core memory plan
- work packets

### 第三步：让 `BlackholeModule` 直接消费这些计划

- 不再以 runner 为主路径

### 第四步：用 direct host path 验证 logical block / memory plan

- `grid > 1` case
- large-shape copy case
- memory-plan 超预算负例

## 验证方式

### 结构验证

- `T.Kernel(2, 3)` 一类非 `1x1` grid 在 split 后 device kernel 中仍可恢复 logical grid 信息
- `AssignBlackholeCores` 输出包含可检查的 logical grid / work packet 信息
- `CodeGenBlackhole` 不再把 `blockIdx` 绑定为常量 `0`
- `PlanBlackholeCB` 输出正式 `cb_configs`

### 执行验证

正式执行验证只覆盖：

- TileLang 暴露的 host callable
- `BlackholeModule` direct host path

copy 必须覆盖：

- `grid > 1` 且 `bx/by` 参与索引的 case
- 总数据量大于 `1.5MB` 的 large-shape copy

large-shape copy 的目标是验证：

- planner 没有把整 tensor 错误放进 per-core L1
- kernel 通过 tile/page/work packet 多轮执行完成搬运
- host-device 交互在多轮搬运下仍正确

另补一个负例：

- 构造 per-core memory plan 超出 `1572864` bytes 的 case
- 编译期直接失败
