# Stage 2E Blackhole Device Resource Semantics

- 文档ID: `stage2e_blackhole_device_resource_semantics`
- 状态: draft
- 日期: 2026-03-25
- 相关总设计: `tasks/dev_design/final_blackhole_backend_redesign.md`

## 1. 背景

当前 Blackhole 后端已经把 copy 主路径收回到正式 pipeline，并为 GEMM 接入了：

- `LowerTileOp` 对 Blackhole 保留 `tl.gemm_py`
- `SplitBlackholeKernel`
- `LowerBlackholeOps`
- `PlanBlackholeCB`
- `ExecutableSpec -> rt_mod_blackhole -> BlackholeModule`

但 GEMM 在 `lower()` 主线中仍会卡在 generic host/device pass 链上。当前观察到的报错包括：

- `MergeSharedMemoryAllocations expects flat memory buffers`
- `In PrimFunc main variables [C_local] are used, but are not passed in as API arguments`
- `Only one dynamic shared memory allocation is allowed`

这些问题不是 GEMM 单点逻辑错误，而是 Blackhole 编程模型与 generic TIR host/device pipeline 的资源语义边界不一致。

根据 `tt_metal_repo` 中的 programming examples、matmul lab 和 advanced topics，TT-Metal/Blackhole 的核心对象分为三层：

1. host-visible objects
   - DRAM buffer
   - scalar/runtime args
   - kernel handle / core range / kernel kind
2. transport resources
   - Circular Buffer (CB)
   - L1 上的 FIFO/page queue
   - producer/consumer 协议：`cb_reserve_back / cb_wait_front / cb_push_back / cb_pop_front`
3. compute-private resources
   - Dst/tile registers
   - unpacker/packer/FPU 配置
   - compute kernel 内部 accumulator / fragment state

而当前 TIR 中：

- `shared/shared.dyn` 仍被当作 generic shared memory buffer
- `local.fragment` 仍被当作 generic local buffer/handle

这导致 generic pass 被迫按错误模型解释 Blackhole 设备内部资源。

## 2. 问题定义

### 2.1 当前错位

当前 pipeline 在 Blackhole 上混淆了两类边界：

1. host/device ABI 边界
2. device 内部资源边界

generic pass 默认认为：

- device region 中的 undefined var 应升成 device func 参数
- `shared.dyn` 是 launch-time / software-managed dynamic shared memory
- compute kernel 的 local object 若跨语句使用，仍可按普通 local var/buffer 处理

但 Blackhole 实际上要求：

- `local.fragment` 不应越过 host/device ABI 边界
- `shared/shared.dyn` 不应等价为 generic dynamic shared allocation
- CB / tile registers / unpack-pack-FPU 协议都属于 device-private resource model

### 2.2 为什么 copy 已部分缓解，而 GEMM 没有

copy 已通过 `AnnotateBlackholeCopySemantics` 在 destructive pass 之前把数据流语义显式化，因此 `LowerBlackholeOps` 不再完全依赖晚期 loop/buffer 形态恢复 copy 语义。

GEMM 当前只有：

- `tl.gemm_py` call 被保留
- `SplitBlackholeKernel` 能识别其 segment kind

但“device-private resource”和“host-visible ABI”的边界仍未显式表达，因此：

- `SplitHostDevice` 会把 `local.fragment` 对应对象视为 undefined var 并升成参数
- `MergeSharedMemoryAllocations` 会把 `shared.dyn` 当作 generic shared buffer 去合并
- `LowerDeviceKernelLaunch` 会把多个 `shared.dyn` 当作多个 dynamic shared alloc

### 2.3 这不是 GEMM 特例

后续若接入：

- element-wise compute
- reduction
- softmax
- 其他需要 compute-private state 的 tile op

只要继续把 Blackhole 设备内部资源伪装成普通 TIR buffer/scope，generic pass 仍会重复发生同类误判。

因此问题的本质是：

**Blackhole 缺少一层统一的 device resource semantics/canonicalization。**

## 3. 目标

本设计的目标不是直接修一个 GEMM case，而是建立可扩展的 Blackhole 设备资源语义边界。

具体目标：

1. 在进入 generic host/device ABI pass 前，显式区分：
   - host-visible tensors/scalars
   - device-private CB resources
   - device-private compute-private resources
2. 不再让 `local.fragment` / `shared.dyn` 以 generic buffer/alloc 形态泄漏到 ABI 层
3. 让后续 GEMM、eltwise、reduction 等 compute op 可以共用同一套资源语义，而不是按算子逐个补特判
4. 保持与 `ExecutableSpec / segment_plan / runtime_args / cb_configs` 现有主协议兼容

## 4. 非目标

本设计不在本阶段直接完成：

- 新 tile op 的完整 lower
- multi-core 调度策略重写
- `LowerBlackholeOps` 的全量算子重构
- 立即替换所有现有 TIR 节点为全新 IR dialect

本设计优先解决的是“语义承载层”和“pass 边界”。

## 5. 核心判断

### 5.1 哪些资源是 host-visible

以下对象应保留在 host/device ABI 边界上：

- DRAM tensor 参数
- scalar 参数
- runtime args 中需要 host materialization 的值
- core/segment/kernel kind 等 host-side launch metadata

这些对象可以继续走 generic `SplitHostDevice` / `MakePackedAPI` 风格的 ABI 主线。

### 5.2 哪些资源是 device-private

以下对象不应跨 ABI 边界：

- CB identity / role / page queue
- L1 transport slots
- fragment / accumulator / dst-register state
- unpacker / packer / FPU 配置依赖

这些对象属于 device-private resource model，应在进入 ABI pass 之前就完成 canonicalization。

### 5.3 `shared.dyn` 和 `local.fragment` 当前信息不足

`scope` 只表达“像什么”，不表达“怎么被用”。

例如：

- `shared.dyn` 不能表达 CB role / page size / producer-consumer 关系
- `local.fragment` 不能表达其是 Dst register-backed accumulator，还是普通 local scratch

因此仅靠现有 storage scope，不足以支撑 Blackhole 后续可扩展 compute lowering。

结论：

**需要扩展 IR 语义，但应优先扩“结构化资源语义”，而不是立刻发明一整套全新节点。**

## 6. 设计方案

### 6.1 新增统一层：Blackhole Device Resource Canonicalization

在 Blackhole pipeline 中新增一层 split-before canonicalization，位于：

```text
LowerTileOp / split-before annotations
  -> BlackholeDeviceResourceCanonicalization
  -> generic host/device boundary passes
  -> SplitBlackholeKernel / LowerBlackholeOps / PlanBlackholeCB / ...
```

这层 pass 的职责是：

1. 识别 device-private resources
2. 将其从 generic buffer/var 视角提升为结构化 Blackhole 资源语义
3. 保证 generic ABI pass 只看到 host-visible 参数

### 6.2 资源分类模型

引入统一分类：

#### A. ABI resources

- `dram_tensor`
- `scalar`
- `host_runtime_arg`

特点：

- 可跨 host/device 边界
- 可进入 packed API / direct runtime materialization

#### B. Transport resources

- `cb_input`
- `cb_output`
- `cb_intermediate`
- `l1_scratch`

特点：

- core-local
- 不进入 host ABI
- 由 pass/planner/codegen/runtime 共同消费

#### C. Compute-private resources

- `fragment_accumulator`
- `fragment_operand`
- `dst_register_tile`
- `compute_private_scratch`

特点：

- 只存在于 compute kernel 语义内部
- 不进入 host ABI
- 应能表达 acquire/commit/wait/release 或等价 ownership 关系

### 6.3 IR 语义扩展方式

优先采用“结构化 attrs/annotations + 显式 resource schema”的方式，而不是直接引入大量新节点。

建议新增/扩展以下语义承载：

1. `blackhole.resource_plan`
   - 记录当前 PrimFunc 中的 device-private resources
   - 每项至少带：
     - `name`
     - `kind`
     - `scope`
     - `data_format`
     - `tile_shape`
     - `page_size_bytes`
     - `lifetime_begin/end`
     - `host_visible: bool`

2. `blackhole.compute_regions`
   - 标记哪些 stmt/op 属于 compute-private region
   - 用于后续把 compute-private 资源约束在 device 内部

3. `blackhole.op_semantics`
   - 对 copy 之外的 compute op，逐步引入结构化语义
   - 不只靠 `scope` 和 `CallNode` 名字恢复

### 6.4 与现有 schema 的关系

新增语义层不是替代现有 `blackhole.*` attrs，而是补前置边界。

现有协议继续保留：

- `blackhole.segment_plan`
- `blackhole.runtime_args`
- `blackhole.cb_requirements`
- `blackhole.cb_configs`
- `blackhole.core_plan`

新增语义层的作用是让这些已有协议能从更稳定的资源语义生成，而不是从晚期普通 TIR 猜。

### 6.5 对 `SplitHostDevice` 的约束

本设计不建议修改 generic `SplitHostDevice` 逻辑去“识别 Blackhole fragment 并跳过”。

更稳的约束是：

- 在进入 `SplitHostDevice` 前，Blackhole PrimFunc 必须已经满足：
  - host-visible 参数已明确
  - device-private resources 不再表现为 ABI free vars

也就是说，正确修法应是：

- **收正输入给 `SplitHostDevice` 的 IR**
- 而不是修改 `SplitHostDevice` 去做 Blackhole-specific workaround

### 6.6 对 `shared.dyn` 的处理原则

`shared.dyn` 在 Blackhole 上不应再被视为 generic dynamic shared memory 的充分表达。

处理原则：

1. `shared/shared.dyn` 只可作为前端过渡语法
2. 在 canonicalization 后应被收正为：
   - `cb_*`
   - `l1_scratch`
   - 或其他明确 transport resource
3. 进入 generic shared-memory passes 前，不应再只靠二维/一维 buffer 形态承载其全部语义

### 6.7 对 `local.fragment` 的处理原则

`local.fragment` 在 Blackhole 上不应继续被视为普通 buffer scope。

处理原则：

1. `local.fragment` 只可作为前端/中间过渡标记
2. canonicalization 后应显式区分：
   - fragment accumulator
   - fragment operand
   - compute-private scratch
3. 它们不应变成 device func ABI 参数

## 7. 对后续算子扩展的意义

该设计一旦成立，后续新算子就不需要再重复回答“这个 local/shared 对象能不能过 SplitHostDevice”。

统一处理逻辑应变为：

1. 新算子先声明自己的 Blackhole op semantics
2. resource canonicalization 将其映射到：
   - host-visible inputs/outputs
   - transport resources
   - compute-private resources
3. planner/codegen/runtime 消费这些结构化资源语义

因此：

- GEMM 只是第一批受益者
- element-wise、reduction、softmax 等都应复用这套边界

## 8. Pass 影响范围

### 8.1 直接影响

- `tilelang_repo/tilelang/engine/phase.py`
- `src/transform/split_host_device.cc`
- `src/transform/lower_blackhole_ops.cc`
- `src/transform/split_blackhole_kernel.cc`
- `src/transform/annotate_blackhole_copy_semantics.cc`
- 未来新增的 `src/transform/blackhole_device_resource_canonicalization.cc`

### 8.2 间接影响

- `src/target/codegen_blackhole.cc`
- `src/target/rt_mod_blackhole.cc`
- `src/target/blackhole_module.cc`
- `tasks/dev_design/stage2d_gemm_integration.md`
- `tasks/progress.md`
- `memory/general_dev.md`
- `memory/bugs.md`

## 9. 实施建议

建议分三步做，而不是一次性大改：

### Step A: 设计与观测层

- 新增 `BlackholeDeviceResourceCanonicalization` 设计文档与调试测试
- 先把 resource classification 和 dump/inspection 跑通
- 不立即改 runtime/codegen 协议

### Step B: ABI 边界收正

- 让 canonicalization 后的 IR 不再把 device-private resources 暴露给 `SplitHostDevice`
- 解除 `C_local` / multiple `shared.dyn` / flat-buffer 假设冲突

### Step C: 语义消费层收敛

- 让 `LowerBlackholeOps / SplitBlackholeKernel / PlanBlackholeCB` 改为优先消费 canonicalized resource semantics
- 逐步减少对晚期普通 TIR matcher 的依赖

## 10. 验证方式

### 10.1 结构验证

新增 pipeline-level 测试，验证 canonicalization 后：

- host-visible 参数集合正确
- device-private resources 不再出现在 ABI free vars 中
- `blackhole.resource_plan` 结构完整

### 10.2 负例验证

验证以下旧错误不再出现：

- `MergeSharedMemoryAllocations expects flat memory buffers`
- `variables [C_local] are used, but are not passed in as API arguments`
- `Only one dynamic shared memory allocation is allowed`

### 10.3 回归验证

至少覆盖：

- copy pipeline tests
- GEMM lower/build tests
- 后续新增一个最小 compute-private 非 GEMM case

## 11. 风险与取舍

### 11.1 风险

- 如果新增 schema 只服务 GEMM，仍会退化成特判设计
- 如果语义扩展过重，短期接入成本会过高
- 如果仍然让 generic buffer/scope 承载关键资源语义，后续问题会反复出现

### 11.2 取舍

本设计选择：

- 不在 generic pass 上打 Blackhole 特判补丁
- 不直接大规模发明新 IR node
- 先引入一层结构化 resource semantics

这是在通用性和实现成本之间更稳的折中。

## 12. 结论

Blackhole 当前的真正冲突不是 GEMM 本身，而是：

**TT-Metal 编程模型中的 CB / Dst-register / compute-private resource，没有在 IR 中以足够明确的结构化语义承载。**

因此后续正式方向应是：

1. 在 `SplitHostDevice` 前引入 Blackhole device resource canonicalization
2. 显式区分 host-visible 与 device-private resources
3. 用统一资源语义支撑 GEMM 以及未来非 GEMM compute op

这应作为 Stage 2D 之后进入更通用 compute 接入前的前置设计工作。
