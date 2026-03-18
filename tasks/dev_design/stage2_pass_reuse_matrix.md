# Stage 2 Pass 复用矩阵设计

## 目标

- 明确 Blackhole 后端应如何复用 TileLang / TVM 现有 PrimFunc/TIR pass 主链。
- 避免继续沿用当前 `OptimizeForTarget` 中对 Blackhole 的 early return 结构，导致大段通用 pass、host/device 分离和参数约束被旁路。
- 为后续 copy 首条完整编译链实现提供一份可执行的 pass 接入矩阵，而不是继续在 `rt_mod_blackhole` / `BlackholeModule` 里补语义。

## 核心结论

### 1. Blackhole 应以整个 PrimFunc/TIR 主链为基础，而不是自建平行流水线

后端目标不是“只处理几个 tile 算子”，而是：

- 复用整个 PrimFunc/TIR 的通用合法化、规范化和 host/device 处理
- 仅在少量 target-aware 边界插入 Blackhole 分支

当前正确主线应收敛为：

```text
DSL
  -> PrimFunc/TIR
  -> 通用 legalize / normalize / optimize passes
  -> LowerTileOp(Blackhole-aware)
  -> 通用 host/device / packed API passes
  -> Blackhole-specific device passes
  -> ExecutableSpec / CodeGenBlackhole / runner
```

### 2. `LowerTileOp` 是 Blackhole 最关键的 target-aware 接入点

`LowerTileOp` 当前已经在整个 PrimFunc 上工作，并在这里展开：

- `tl.copy`
- `tl.gemm`

因此 Blackhole 不应继续主要依赖 `LowerTileOp` 之后的晚期 loop pattern 恢复，而应：

- 保留整个 TIR 函数体
- 在 `LowerTileOp` 中为 Blackhole 增加分支
- 让 tile 相关语义 lower 成“合法 TIR 中仍可识别的 Blackhole-preserving 形式”

### 3. host/device 与参数语义应尽量对齐 TileLang / TVM 主模型

Blackhole 长期不应继续跳过：

- `AnnotateDeviceRegions`
- `SplitHostDevice`
- `MakePackedAPI`
- `LowerDeviceKernelLaunch`

当前 `rt_mod_blackhole` 把“没有 `calling_conv` 的 PrimFunc”直接当 device kernel 收进去，只能视为过渡逻辑，不能再作为正式模型。

## Pass 复用矩阵

### A. 前端合法化 / 语义归一化

| pass | 当前 Blackhole | 目标状态 | 说明 |
|------|----------------|----------|------|
| `BindTarget` | 已复用 | `reuse` | 通用 target 绑定，应保持一致 |
| `AddWrapperForSingleBufStore` | 已复用 | `reuse` | 前端形状归一化，不应由 Blackhole 特化 |
| `LegalizeNegativeIndex` | 已复用 | `reuse` | 通用索引合法化 |
| `VerifyParallelLoop` | 已复用 | `reuse` | 通用语义检查 |
| `InjectAssumes` | 已复用 | `reuse` | 为后续参数/边界检查保留 assume 信息 |
| `Simplify` | 已复用 | `reuse` | 通用规范化 |
| `LowerAccessPtr` | 已复用 | `reuse` | 前端 pointer metadata 到标准 TIR |
| `HoistNonRestrictParams` | 已复用 | `reuse` | 供 host/device 与 codegen 使用 |

### B. TileLang 高层 tile 语义层

| pass | 当前 Blackhole | 目标状态 | 说明 |
|------|----------------|----------|------|
| `LayoutReducer` | 已复用 | `reuse` | 布局推导前置，应保留 |
| `LayoutInference` | 已复用 | `reuse` | 为 tile/dataflow 语义提供布局基础 |
| `LowerTileOp` | 已复用，但无 Blackhole-aware 分支 | `reuse_with_blackhole_branch` | Blackhole 最关键接入点；不能再完全等晚期恢复 |
| `LowerL2Persistent` | 已复用 | `reuse` | 与当前 Blackhole 主问题正交，可保留 |

### C. 通用 TIR 规范化 / 优化

| pass | 当前 Blackhole | 目标状态 | 说明 |
|------|----------------|----------|------|
| `DecoupleTypeCast` | 已复用 | `reuse` | 通用表达式合法化 |
| `LegalizeVectorizedLoop` | 已复用 | `reuse` | 通用 loop 合法化 |
| `LegalizeSafeMemoryAccess` | 已复用 | `reuse` | 边界安全检查 |
| `LowerOpaqueBlock` | 目前 device-only 辅助复用 | `reuse` | Blackhole 也应进入主线 |
| `NarrowDataType` | 当前被 early return 绕过 | `reuse` | 通用 dtype 规范化 |
| `FlattenBuffer` | 当前被 early return 绕过 | `reuse` | 便于后续参数/寻址和 codegen |
| `ConfigIndexBitwidth` | 当前被 early return 绕过 | `reuse` | 统一 index 计算宽度 |
| `VectorizeLoop` | 当前被 early return 绕过 | `reuse` | 是否启用可 target-gate，但 pass 本身应纳入主线 |
| `StorageRewrite` | 当前被 early return 绕过 | `reuse` | 通用存储规划，不应长期缺席 |
| `LoopUnswitching` | 当前被 early return 绕过 | `reuse` | 通用 loop 优化 |
| `UnrollLoop` | 当前被 early return 绕过 | `reuse` | 通用 loop 展开 |
| `RenormalizeSplitPattern` | 当前被 early return 绕过 | `reuse` | 规范化循环分裂模式 |
| `RemoveNoOp` | 当前被 early return 绕过 | `reuse` | 通用清理 |
| `HoistIfThenElse` | 当前被 early return 绕过 | `reuse` | 通用控制流规范化 |
| `VerifyMemory` | 当前被 early return 绕过 | `reuse` | 通用内存验证 |
| `AnnotateEntryFunc` | 当前被 early return 绕过 | `reuse` | entry 语义统一 |

### D. host/device 与调用约束

| pass | 当前 Blackhole | 目标状态 | 说明 |
|------|----------------|----------|------|
| `AnnotateDeviceRegions` | 当前被 early return 绕过 | `reuse` | host/device 边界识别必须恢复 |
| `SplitHostDevice` | 当前跳过 | `reuse` | Blackhole 不应长期自定义 kernel model |
| `AnnotateReadOnlyParams` | 当前被 early return 绕过 | `reuse` | 参数约束与 codegen/runner 协议一致性需要 |
| `MakePackedAPI` | 当前跳过 | `reuse` | host 侧 PackedFunc 参数解析不应由 `BlackholeModule` 重新定义 |
| `LowerDeviceKernelLaunch` | 当前跳过 | `reuse_with_blackhole_branch` | 应恢复 `calling_conv` 语义，必要时只在 launch lowering 上加分支 |

### E. 硬件专属调度 / 搬运 / 同步

| pass | 当前 Blackhole | 目标状态 | 说明 |
|------|----------------|----------|------|
| `IfStmtBinding` | 当前被 early return 绕过 | `reuse_with_blackhole_branch` | 通用绑定思路可保留，具体后续是否必须按 Blackhole 数据流调整待定 |
| `PlanAndUpdateBufferAllocationLocation` | 当前被 early return 绕过 | `reuse_with_blackhole_branch` | 其职责与 Blackhole L1/CB 资源规划相关，优先复用思路 |
| `PipelinePlanning` | 当前被 early return 绕过 | `reuse_with_blackhole_branch` | 这是未来 copy/gemm dataflow 的重要承载点 |
| `InjectSoftwarePipeline` | 当前被 early return 绕过 | `reuse_with_blackhole_branch` | 是否直接复用待验证，但职责应保留在 pass 层 |
| `MergeSharedMemoryAllocations` | 当前被 early return 绕过 | `reuse_with_blackhole_branch` | 与 Blackhole shared/CB/L1 资源有映射关系 |
| `ThreadSync("global"/"shared"/"shared.dyn")` | 当前被 early return 绕过 | `reuse_with_blackhole_branch` | 同步职责必须在 pass 层，不能留给 runtime 猜 |
| `PersistThreadblock` | 当前被 early return 绕过 | `do_not_reuse_impl` | 目前更偏 GPU block 模型，先不直接复用实现 |
| `WarpSpecialized` | 当前被 early return 绕过 | `do_not_reuse_impl` | CUDA warp 专属 |
| `InjectTmaBarrier` | 当前被 early return 绕过 | `do_not_reuse_impl` | Hopper/TMA 专属 |
| `MultiVersionBuffer` | 当前被 early return 绕过 | `do_not_reuse_impl` | 与当前 Blackhole MVP 无直接对应 |
| `RewriteWgmmaSync` | 当前被 early return 绕过 | `do_not_reuse_impl` | WGMMA 专属 |
| `LowerThreadAllreduce` | 当前被 early return 绕过 | `do_not_reuse_impl` | 线程归约模型暂不对齐 |
| `LowerLDGSTG` | 当前被 early return 绕过 | `do_not_reuse_impl` | Nvidia 专属 load/store 降法 |
| `LowerHopperIntrin` | 当前被 early return 绕过 | `do_not_reuse_impl` | Hopper 专属 |
| `InjectFenceProxy` | 当前被 early return 绕过 | `do_not_reuse_impl` | async proxy 专属 |
| `AnnotateWarpGroupRegAlloc` | 当前被 early return 绕过 | `do_not_reuse_impl` | warp-group register 语义不适用 |
| `MarkCudaSyncCalls` | 当前被 early return 绕过 | `do_not_reuse_impl` | CUDA PDL 专属 |

## Blackhole 专属 pass 的最终边界

### `LowerBlackholeOps`

输入应为：

- 已经过 `LowerTileOp(Blackhole-aware)` 的 PrimFunc

职责应收缩为：

- 消费 Blackhole-preserving TIR
- 产出 `blackhole.*` attrs
- 提取 segment 计划
- 提取 runtime arg schema

不再承担：

- 从晚期普通 loop 中恢复大部分 copy/gemm 语义
- 猜 PrimFunc 参数结构
- 用 runtime 特判兜底 device kernel 结构

### `PlanBlackholeCB`

继续保留，职责不变：

- 从 `blackhole.cb_requirements` / segment 信息收敛到 runtime-ready `blackhole.cb_configs`

### `AssignBlackholeCores`

继续保留，职责不变：

- 只生成 host/runtime 消费的 core scheduling plan

## 基于矩阵的改造顺序

### 第一步：恢复 pass 主链

- 去掉 Blackhole 在 `OptimizeForTarget` 的长期 early return 设计
- 优先恢复：
  - 通用 TIR 规范化 / 优化
  - host/device 与参数约束相关 pass

### 第二步：收正 host/device 与参数边界

- 让 Blackhole 主路径恢复：
  - `AnnotateDeviceRegions`
  - `SplitHostDevice`
  - `MakePackedAPI`
  - `LowerDeviceKernelLaunch` 或其 Blackhole 分支
- `rt_mod_blackhole` 不再长期依赖“缺少 `calling_conv` 也当 device kernel”

### 第三步：将差异集中到 `LowerTileOp` 和 Blackhole device passes

- 在 `LowerTileOp` 中保留 copy/gemm 的 Blackhole-preserving 语义
- `LowerBlackholeOps` 消费这些语义
- `rt_mod_blackhole` / `BlackholeModule` 只消费 pass 结果

### 第四步：以 copy 为首条完整链路验证

copy 的完成标准不再只是“能跑”，而是：

- 经过通用 TIR pass 主链
- 经过 host/device 与 Packed API 主链
- 经过 Blackhole-specific device passes
- 最终 spec 和 runtime 参数绑定不再依赖 runtime/module 猜测

## 验证方式

### 文档一致性

以下文档结论必须保持一致：

- `final_blackhole_backend_redesign.md`
- `stage2_single_core_pass_integration.md`
- `progress.md`

### Pass 级验证

- 逐项确认矩阵中的 pass 当前是否被 Blackhole 复用
- 对 `reuse_with_blackhole_branch` 的 pass，明确后续需要修改的接入点
- 对 `do_not_reuse_impl` 的 pass，明确 Blackhole 对应职责位置

### 实现前置检查

在开始下一轮代码改造前，必须先满足：

- 不再把 Blackhole 视为“独立于 TIR 主链的自定义 kernel model”
- 不再让 `BlackholeModule` / `rt_mod_blackhole` 定义 PrimFunc 参数和 host/device 语义

