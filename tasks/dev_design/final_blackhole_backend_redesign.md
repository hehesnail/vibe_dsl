# TileLang Blackhole 后端最终重构设计

## 基本信息

- **文档ID**: `final_blackhole_backend_redesign`
- **日期**: 2026-03-18
- **适用范围**: `tilelang_repo` Blackhole 后端与 TileLang 管理的 runner、`tt_metal_repo` 依赖与 API 参考、相关测试与设计文档
- **状态**: 当前唯一权威设计

## 1. 文档定位

本文档替代以下文档中已经过时或与源码现状冲突的总体设计结论：

- `tasks/design_review.md`
- `tasks/arch_design.md`
- `tasks/progress.md` 中关于总体架构的描述

旧文档仍保留历史记录价值，但后续实现、评审、拆任务、验收，统一以本文档为准。

## 2. 最终结论

当前工程的主要问题不是“少几个 pass”，而是后端主抽象放错了。

当前实现默认的核心假设是：

- 后端产物主要是一段 TT-Metal kernel C++ 字符串
- runtime/runner 再根据这段代码和若干固定参数去执行

这个假设与 TT-Metal 的稳定编程模型冲突。TT-Metal 的真实主抽象是：

- `Program`
- `CircularBuffer`
- `Kernel`
- `DataMovementConfig / ComputeConfig`
- compile-time args
- runtime args
- multi-core scheduling

因此，Blackhole 后端的正式目标不应是“把 TIR 直接打印成单个 kernel 文件”，而应是：

- **从 TileLang TIR 提取 TT-Metal `ExecutableSpec`**
- **由 spec-driven runtime/runner 去构建并执行 TT-Metal Program**

## 3. 当前实现的状态诊断

### 3.1 已经存在且应保留的部分

- `lower.py` 已经接入：
  - `LowerBlackholeOps`
  - `PlanBlackholeCB`
  - `AssignBlackholeCores`
- `LowerBlackholeOps` 对 matmul builtin lowering 已有基础框架
- `PlanBlackholeCB` 已有基本的 CB 规划和约束检查
- `AssignBlackholeCores` 已有 grid 分析和 core 规划基础
- `CodeGenBlackhole` 已有 builtin visitor 框架
- 外部 runner 路径已经打通了“runtime module -> 外部可执行程序”的工程边界

### 3.2 已确认错误或不再采用的设计

- 以“单个 kernel 字符串”为后端主产物
- 继续把 `SplitBlackholeKernel` 放在关键路径
- 以裸 `noc_async_read/write(src_addr, dst_addr, size)` 作为主要中间抽象
- 让 codegen 负责多核的物理 core 映射
- 继续扩展当前固定命令行 runner 协议
- 将 `blackhole_module_direct.cc` 作为主路径

### 3.3 已确认的断层

- `PlanBlackholeCB` 写的是 `blackhole.cb_configs`
- `rt_mod_blackhole.cc` 仍在读旧 attr 名
- 现有 runtime 和 pass 产物没有真正接通
- `LowerBlackholeOps` 的 copy/dataflow lowering 仍不足以恢复 TT-Metal 真正需要的 tile access 语义
- `CodeGenBlackhole` 仍默认一个 PrimFunc 对应一个 `kernel_main()`
- runner 只支持固定输入，不支持多 kernel、多 CB、compile-time args、per-kernel runtime args schema 和 per-core runtime args
- Blackhole 目前在 `OptimizeForTarget` 中过早退出，绕过了大量 TileLang/TVM 通用 TIR 优化与 host/device 约束 pass
- Blackhole 当前跳过 `SplitHostDevice` / `MakePackedAPI` / `LowerDeviceKernelLaunch`，导致 PrimFunc 参数、host/device 边界和 runtime 参数语义长期由 `rt_mod_blackhole` / `BlackholeModule` 间接补洞

## 4. 正式架构

### 4.1 总体结构

```text
TileLang DSL / TIR
  -> 现有 TileLang legalize / optimize
  -> LowerBlackholeOps
  -> PlanBlackholeCB
  -> AssignBlackholeCores
  -> BuildTileLangBlackholeWithoutHost
      -> Extract ExecutableSpec
      -> Emit kernel source(s)
      -> Build BlackholeModule(spec)
  -> BlackholeModule
      -> serialize spec + tensor/scalar args
      -> invoke runner
  -> runner
      -> CreateProgram
      -> CreateCircularBuffer(s)
      -> CreateKernelFromString / CreateKernel
      -> SetRuntimeArgs(per-kernel, per-core)
      -> Enqueue workload
```

### 4.2 设计原则

1. 以后端执行模型约束 lowering，不以后端打印器约束执行模型。
2. 先建立稳定协议，再扩算子。
3. compile-time args 与 runtime args 必须严格分层。
4. multi-core 主要由 host/runtime 实现，不由 codegen 主导。
5. `SplitBlackholeKernel` 不进入 MVP 主路径。
6. Blackhole 以后端差异最小化为目标，优先复用 TileLang/TVM 现有 PrimFunc/TIR pass 主链。
7. host/device 划分、Packed API 参数语义和通用 TIR 优化默认沿用 TileLang/TVM 现有约束，不再长期由 Blackhole 自定义模型旁路。

### 4.3 Pass 主链接入原则

Blackhole 不应再被视为“在 `LowerAndLegalize` 之后直接自建 device pipeline”的特例目标，而应尽量回到：

```text
DSL / PrimFunc
  -> TileLang/TVM 通用 legalize / normalize / optimize
  -> LowerTileOp(Blackhole-aware)
  -> AnnotateDeviceRegions / SplitHostDevice / MakePackedAPI / LowerDeviceKernelLaunch
  -> Blackhole-specific device passes
  -> BuildTileLangBlackholeWithoutHost
```

其中：

- `LowerTileOp` 是 Blackhole 最关键的 target-aware 接入点
- `LowerBlackholeOps` 不再主要承担晚期 loop 语义恢复
- `rt_mod_blackhole` 与 `BlackholeModule` 不再长期承担 PrimFunc 参数语义与 host/device 边界定义

## 5. 核心数据结构

建议在 `src/target/blackhole_module.h` 一带定义并替换当前以 `BlackholeFunctionInfo` 为中心的弱结构。

```cpp
struct CBConfig {
  uint32_t cb_id;
  std::string name;
  std::string role;              // input/output/intermediate
  uint32_t page_size_bytes;
  uint32_t num_pages;
  std::string data_format;       // Float16_b, Float32, ...
};

struct CorePlan {
  uint32_t grid_x;
  uint32_t grid_y;
  uint32_t cores_needed;
  uint32_t work_per_core;
  uint32_t core_grid_x;
  uint32_t core_grid_y;
};

struct KernelArgSpec {
  std::string name;
  std::string kind;              // buffer_addr, scalar_u32, start_tile_id, work_per_core
  std::string dtype;
};

struct KernelSpec {
  std::string name;
  std::string kind;              // reader / compute / writer / fused_dataflow
  std::string core_type;         // brisc / ncrisc / trisc
  std::string source_code;
  std::vector<uint32_t> compile_time_args;
  std::vector<KernelArgSpec> runtime_args;
};

struct ExecutableSpec {
  std::string entry_name;
  std::string target_mode;       // single_core_copy / single_core_gemm / multi_core_gemm
  std::vector<CBConfig> cb_configs;
  CorePlan core_plan;
  std::vector<KernelSpec> kernels;
};
```

## 6. Attr Schema

只保留以下命名：

- `blackhole.cb_requirements`
- `blackhole.cb_configs`
- `blackhole.core_plan`
- `blackhole.segment_plan`
- `blackhole.target_mode`

明确移除旧协议：

- `tl.blackhole_cb_config`
- `tl.blackhole_kernel_split`

## 7. 各模块最终职责

### 7.1 `LowerBlackholeOps`

职责：

- 将 TileLang 高层语义降为 TT-Metal 可还原的段内语义
- 写入 `cb_requirements`
- 写入 segment 所需的中间信息

前置约束：

- 长期输入应为经过 `LowerTileOp(Blackhole-aware)` 后的 PrimFunc
- 不应继续依赖“`LowerTileOp` 完全展开后，再从普通 loop/load/store 里恢复绝大多数 tile 语义”作为主路径

保留 compute builtin：

- `cb_reserve_back`
- `cb_push_back`
- `cb_wait_front`
- `cb_pop_front`
- `mm_init`
- `matmul_tiles`
- `tile_regs_*`
- `pack_tile`

新增中层 dataflow builtin：

- `blackhole.read_tile_to_cb(buffer, tile_index, cb_id, tile_bytes, accessor_slot)`
- `blackhole.write_tile_from_cb(cb_id, buffer, tile_index, tile_bytes, accessor_slot)`

### 7.2 `PlanBlackholeCB`

职责：

- 从 `blackhole.cb_requirements` 规划 CB
- 输出 runtime-ready `blackhole.cb_configs`
- 验证 Blackhole 约束

建议 CB 分区：

- input: `0..15`
- output: `16..31`
- intermediate: `32..63`

### 7.3 `AssignBlackholeCores`

职责：

- 只产生 host scheduling plan
- 不负责 device code 中 thread index 的最终物理化

输出：

- `grid_x`
- `grid_y`
- `cores_needed`
- `work_per_core`
- physical grid size

### 7.4 `SplitBlackholeKernel`

结论：

- 退出主路径
- 文件可保留
- 不作为当前阶段前置条件

### 7.5 `CodeGenBlackhole`

拆为两层：

- `BlackholeSpecBuilder`
  - 从 `PrimFunc + attrs` 构造 `ExecutableSpec`
- `CodeGenBlackhole`
  - 为单个 `KernelSpec` 生成源代码

约束：

- 不再假设一个 PrimFunc 只生成一个 `kernel_main`
- 不再在 codegen 里把 `blockIdx` 固化为常量

### 7.6 `rt_mod_blackhole`

职责：

- 从 IRModule/PrimFunc 提取 `ExecutableSpec`
- 构造 `BlackholeModule(spec)`

约束：

- 只消费 pass 产出的 device-side 语义、attrs 和 schema
- 不再长期把“没有 `calling_conv` 的 PrimFunc”视为正式 device kernel 模型
- 不再定义 PrimFunc 参数类别、host/device 边界和 runtime 参数意义

### 7.7 `BlackholeModule`

职责：

- TVM runtime adapter
- 序列化 `ExecutableSpec + tensors + scalars`
- 作为 host-side 执行载体调用 runner

约束：

- 不再长期承担 Packed API / host-device 边界语义
- 输入输出 buffer、scalar、dynamic shape 参数如何映射到 runtime arg，应由 PrimFunc + pass schema 显式决定
- 调用外部 runner

runner 输入协议：

- `spec.json`
- `input.bin`
- `output.bin`

### 7.8 `blackhole_module_direct.cc`

结论：

- 退出主线
- 作为实验代码保留即可
- 不继续扩展

### 7.9 runner

职责：

- 读取 `spec.json`
- 创建设备与 `Program`
- 创建所有 CB
- 创建所有 kernels
- 设置 compile-time args 与 runtime args
- 执行并回读

优先建议：

- 使用 `CreateKernelFromString`

## 8. 算子策略

### 8.1 Copy

MVP 目标：

- single-core
- `reader + writer` 两段为正式模式
- 可临时兼容 `fused_dataflow`

### 8.2 GEMM

MVP 目标：

- single-core
- `reader + compute + writer`
- compute 对齐 TT-Metal `mm_init / matmul_tiles / pack_tile`

### 8.3 Multi-core

实现位置：

- runtime/runner

实现方式：

- 基于 `CorePlan`
- materialize `start_tile_id`
- materialize `work_per_core`
- per-core `SetRuntimeArgs`

## 9. 测试矩阵

### 9.1 Pass tests

- 真 `PrimFunc`
- 真 attrs/assert
- 不再使用 mock-only pass 测试作为主验证

### 9.2 Spec tests

- 输入 lowered PrimFunc
- 断言 `ExecutableSpec` 结构正确

### 9.3 Codegen tests

- reader/writer/compute 各自源代码结构
- compile-time args / runtime args schema

### 9.4 True E2E

- `copy single_core`
- `gemm single_core`
- `copy multi_core`
- `gemm multi_core`

注意：只做 codegen 或只做 reference compare 的脚本，不再称为 true E2E。

## 10. 分阶段实施顺序

### Stage 0: 协议重构

- 引入 `ExecutableSpec`
- 统一 attrs
- 重构 `rt_mod_blackhole`
- 重构 `BlackholeFunctionInfo`

### Stage 1: copy 单核闭环

- 重做 copy lowering
- spec-driven runner
- 跑通 single-core copy

### Stage 2: single-core pass integration

- 先把 copy/gemm 的关键执行语义迁回 pass/schema
- `kernels[]` / `runtime_args` / CB 依赖开始由 pass 产物主导
- 先迁 copy，再迁 gemm
- 最后完成 single-core copy/gemm 真执行闭环

### Stage 3: multi-core runtime 调度

- runtime 消费 `CorePlan`
- per-core runtime args

### Stage 4: 优化项

- 评估是否需要 `SplitBlackholeKernel`
- 评估 direct mode
- 复杂算子扩展

## 11. 文件级改造建议

优先改造：

- `tilelang_repo/src/target/blackhole_module.h`
- `tilelang_repo/src/target/blackhole_module.cc`
- `tilelang_repo/src/target/rt_mod_blackhole.cc`
- `tilelang_repo/src/target/codegen_blackhole.cc`
- `tilelang_repo/src/transform/lower_blackhole_ops.cc`
- `tilelang_repo/src/transform/plan_blackhole_cb.cc`
- `tilelang_repo/src/transform/assign_blackhole_cores.cc`
- `tilelang_repo/tools/blackhole_runner/runner.cpp`

降级处理：

- `tilelang_repo/src/target/blackhole_module_direct.cc`
- `tilelang_repo/src/transform/split_blackhole_kernel.cc`

## 12. 需要显式废弃的旧结论

以下说法不再成立：

- “Blackhole 主产物是一个 kernel 字符串”
- “SplitBlackholeKernel 是当前阶段核心前置”
- “codegen 负责 blockIdx 到物理 core 的主要映射”
- “当前 Python/E2E 测试已经证明完整执行闭环”
- “外部 runner 现有命令行协议足够扩展”

## 13. 验收标准

一个阶段是否真正完成，以以下标准判断：

- Stage 0: pass attrs 与 runtime 协议完全一致
- Stage 1: single-core copy 真执行并结果正确
- Stage 2: single-core copy/gemm 真执行并结果正确，且 copy/gemm 关键执行语义主要由 pass 产物而非 runtime 特判提供
- Stage 3: multi-core copy/gemm 真执行并结果正确

任何只验证 codegen、只验证字符串包含关系、只验证参考算法的脚本，都不能单独作为阶段完成依据。
