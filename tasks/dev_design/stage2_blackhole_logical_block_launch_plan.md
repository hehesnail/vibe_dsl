# Stage 2 Blackhole Logical Block / Launch Plan 设计

## 基本定位

- **状态**: 当前活动阶段设计
- **前置总体设计**: `final_blackhole_backend_redesign.md`
- **相关阶段设计**:
  - `stage2_pass_reuse_matrix.md`
  - `stage2_single_core_pass_integration.md`

本文件只处理一个问题：

- **如何把 TileLang 现有的逻辑 block 语义，落实为 Blackhole 的 execution / memory / launch plan**

它不替代总体设计，也不替代 Stage 2 总体拆分。

## 背景与结论

### 1. `T.Kernel` 已经定义了逻辑 block/grid 语义

TileLang 前端里，`T.Kernel(grid_x, grid_y, ..., threads=...)` 会直接生成：

- `blockIdx.x/y/z`
- `threadIdx.x/y/z`

这说明 DSL 程序中“每个 block 处理哪块数据”的语义，本来就属于 TIR，而不是 host/runtime 侧再猜的。

### 2. CUDA 路径验证了正确分层

TileLang 的 CUDA/Hopper 路径里：

- `PlanAndUpdateBufferAllocationLocation`
- `PipelinePlanning`
- `InjectSoftwarePipeline`
- `WarpSpecialized`

这些决定 memory hierarchy、pipeline、producer/consumer 编排的 pass，都发生在 `SplitHostDevice` / `LowerDeviceKernelLaunch` 之前或其周边的 target-aware planning 阶段。

相反：

- `SplitHostDevice`
- `LowerDeviceKernelLaunch`

更偏 host/device ABI 与 launch materialization。

因此 Blackhole 的正确方向也应一致：

- **先在 TIR/pass 里形成 execution plan**
- **再在 host/device 和 runtime 层 materialize**

### 3. TT-Metal host main 体现的是“落地形式”，不是“规划归属”

TT-Metal 示例中的 host main 会显式创建：

- CB
- kernel
- runtime args
- semaphore
- program

这说明 Blackhole 最终一定需要 host-side materialization。

但对 TileLang 编译器来说，真正的规划归属仍应尽量前移到 TIR/pass：

- memory hierarchy
- reader / compute / writer 编排
- compile-time / runtime arg schema
- logical block -> physical core 映射

host 侧应主要负责把这些计划翻译成 TT-Metal host calls。

## 当前问题

当前 Blackhole 已经开始读取逻辑 block 语义，但还没有把它作为正式执行模型落实。

### 已经存在的部分

- `AssignBlackholeCores` 会从 `thread_extent` 中提取 `grid_x / grid_y`
- `blackhole.core_plan` 已开始记录 grid/work 信息

### 当前仍然错误或过渡的部分

1. `CodeGenBlackhole` 直接把 `blockIdx.x/y/z` 绑定为常量 `0`
2. runner 仍固定只在单核 `{0, 0}` 上执行
3. `LowerBlackholeOps` 当前 runtime arg schema 仍是 copy-first 模板
4. `BlackholeModule` / runner 仍残留 ABI 猜测逻辑

这意味着当前 single-core 路径虽然能跑，但它表达的是：

- `grid == 1x1` 的 bring-up 路径

而不是：

- `logical grid` 被正确保留
- `single-core` 只是 `physical_core_count == 1`

## 目标

本阶段需要把 Blackhole 的执行模型收正为：

1. 逻辑 block/grid 语义保留在 TIR 中
2. TIR / target-aware passes 产出 execution / memory / launch plan
3. host/runtime 只 materialize 该 plan
4. 当前先只要求 single-core，但接口和协议按 multi-core 可扩展方式设计

## 分层设计

### A. Logical block semantics

这层来自 DSL/TIR，本轮不重新发明协议。

包含：

- `blockIdx.x/y/z`
- `threadIdx.x/y/z`
- block 对输入/输出 tile 的索引关系
- block 级 pipeline / dataflow 结构

这层语义必须继续保留到 Blackhole target-aware planning 可见的阶段，不能在 codegen 前被提前折叠掉。

### B. Blackhole execution planning

这是当前最缺的中间层。

它需要从 TIR 中产出：

- logical grid shape
- kernel segments
- kernel-level ABI
- memory objects / CB plan
- logical block -> physical core/work packet 映射

这层 plan 的职责是把：

- “程序想怎么按 block 切”

转换成：

- “Blackhole/TT-Metal 上怎么按 core 和 runtime packet 执行”

### C. Host/runtime materialization

这层只负责：

- `Program`
- CB / memory objects
- kernel creation
- runtime arg binding
- enqueue / readback

它不再负责重新解释 DSL/TIR 语义。

## 单核最小正确模型

当前阶段只要求 single-core，但 single-core 的正确模型不应等于“把 `blockIdx=0` 写死”。

正确的最小模型应为：

- `physical_core_count = 1`
- `logical_block_count = grid_x * grid_y`
- 一个 physical core 顺序处理一个或多个 logical blocks
- 当前 logical block 由 work packet 提供，而不是由 codegen 常量化

### 单核 work packet 最小语义

单核最小 execution plan 至少需要表达：

- `logical_grid_x`
- `logical_grid_y`
- `work_offset_linear` 或 `work_offset_{x,y}`
- `work_count`

对于二维 grid，可以先采用 row-major 线性化：

- `linear_id = blockIdx.y * grid_x + blockIdx.x`

single-core 下：

- core 0 处理 `[0, logical_block_count)` 这段 work

后续 multi-core 只是在此基础上把 work packet 分发给多个 core。

## 对现有 pass 的职责调整

### `LowerBlackholeOps`

新增要求：

- 不再默认 `blockIdx.x/y` 最终会在 codegen 中被抹成常量
- runtime arg / kernel ABI 设计要允许传入 work packet 或当前 logical block 信息
- segment plan 不再只描述“单个 fused copy kernel”，而要允许与 logical work 分发配合

当前仍允许：

- copy-first 的 builtin lowering

当前不再允许长期保留：

- 默认 runtime args 只有 `input/output/tile_count/scratch`
- 默认一个 kernel 就能覆盖所有 single-core block 语义而无需 work packet

### `PlanBlackholeCB`

新增要求：

- memory plan 必须与 execution plan 结合
- 至少考虑：
  - segment 对 CB 的使用关系
  - page size / num_pages
  - scratch / intermediate 需求
  - 单核下的 L1 峰值约束

当前阶段仍可只覆盖单核，但输出应是明确 memory objects，而不是仅做 CB id allocator。

### `AssignBlackholeCores`

新增要求：

- 从逻辑 grid 中生成可执行的 work distribution
- 单核下也要显式生成 core/work mapping
- 不能只写 `cores_needed / work_per_core` 这种摘要信息

至少应补出的信息：

- logical grid shape
- logical block linearization policy
- per-core work range 或 work packet
- physical core coords

### `SplitHostDevice` / `LowerDeviceKernelLaunch`

本轮不要求它们承担 block planning。

它们继续主要负责：

- host/device ABI 边界
- launch materialization

但 Blackhole 后续需要基于 execution plan，把 host launch 与 TT-Metal materialization 接起来。

## 协议与接口变化

### 1. `blackhole.core_plan`

当前 `blackhole.core_plan` 只包含：

- `grid_x`
- `grid_y`
- `cores_needed`
- `work_per_core`
- `core_grid_x`
- `core_grid_y`

这还不足以支撑正确执行。

后续应扩充为至少能表达：

- logical grid shape
- logical block linearization policy
- per-core work packets
- physical core coordinates

### 2. `ExecutableSpec`

不新增第二套平行协议名，但需要让现有结构足够表达：

- memory objects
- kernel ABI
- launch packets

建议方向：

- `CorePlan` 补足 work packet 语义
- `KernelSpec` 补足 kernel-level ABI / processor role
- `CBConfig` 继续承接 memory materialization

### 3. runner / module 边界

runner 与 `BlackholeModule` 必须收缩为 plan consumer。

不再允许长期保留的行为：

- 猜最后一个 buffer 是 output
- 固定单核 `{0, 0}`
- 用 copy 专用固定 runtime arg 模板代表一般 DSL ABI

## 实现顺序

### 第一步：先收正设计和协议边界

- 明确 logical block semantics 不丢
- 明确 single-core work packet 的最小语义
- 明确 `core_plan` / runner / codegen 的职责边界

### 第二步：停止过早常量化 `blockIdx`

- `CodeGenBlackhole` 不再长期把 `blockIdx.x/y/z` 直接写成 `0`
- 改为从 execution plan 的当前 work item 取值

### 第三步：把单核执行改成“单核串行处理 logical blocks”

- runner 从 `core_plan` 取单核 work packet
- 单核执行仍可只用一个 core
- 但 core 处理的是 plan 指定的 logical block 范围

### 第四步：收正 runtime arg / memory plan

- `LowerBlackholeOps` / `PlanBlackholeCB` 输出与 work packet 对齐的 ABI / memory objects
- runner 只消费这些结构

## 验证方式

### 结构验证

- `T.Kernel(2, 3)` 之类非 `1x1` grid 在 Blackhole device IR 中仍可恢复 logical grid 信息
- `AssignBlackholeCores` 输出包含可检查的 logical grid / work packet 信息
- `CodeGenBlackhole` 不再把 `blockIdx` 绑定为常量 `0`

### 协议验证

- `ExecutableSpec` / `spec.json` 中可见单核 work packet 与 physical core mapping
- runner 不再依赖位置猜测 input/output

### 执行验证

- 先覆盖 single-core、grid > 1 的 copy 样例
- 再扩到更多 tile shape
- 后续复用同一 execution plan 结构推进 GEMM
