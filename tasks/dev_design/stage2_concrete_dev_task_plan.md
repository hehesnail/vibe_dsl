# Stage 2 当前具体开发任务拆解

## 基本定位

- **状态**: 当前活动任务规划
- **前置总体设计**: `final_blackhole_backend_redesign.md`
- **前置阶段设计**:
  - `stage2_pass_reuse_matrix.md`
  - `stage2_single_core_pass_integration.md`
  - `stage2_blackhole_logical_block_launch_plan.md`

本文件只回答一个问题：

- **基于当前最新设计收束，接下来按什么顺序推进 Blackhole 后端开发任务**

## 当前判断

结合当前主线与最新设计结论，Blackhole 现在已经具备：

- `ExecutableSpec`
- split 后 host/device 主链
- `LowerBlackholeOps -> PlanBlackholeCB -> AssignBlackholeCores` 的雏形
- staged copy 的最小 direct execution 闭环

但要进入正式主线，还存在四个硬阻塞：

1. split 前语义规划仍不够强，copy/gemm 语义仍偏依赖 split 后 matcher 恢复
2. `blackhole.core_plan` 仍是摘要信息，不是正式 execution plan
3. `blackhole.cb_configs` 仍偏 MVP allocator，不是正式 memory plan
4. `BlackholeModule` 还没有成为唯一正式 host-side execution path

因此当前阶段的正确推进顺序是：

1. 先固定 split 前 / split 后 / host-side 三层边界
2. 再收正 split 后正式 plan
3. 再把 `BlackholeModule` 收成正式 direct host path
4. 再用 copy 验证 large-shape / grid>1 / memory-plan correctness
5. 最后再接回更多通用 pass 与 GEMM

## 任务总原则

### 1. 先分层，再补 pass

不再从旧 Blackhole pass 的问题倒推设计。先固定：

- split 前语义规划
- split 后正式 plan 提取
- host-side materialization

### 2. `BlackholeModule` direct path 是唯一正式执行路径

runner 只保留为调试工具，不再进入主任务依赖链。

### 3. copy 继续作为首个正式验收对象

copy 不只验证最小 case，还必须验证：

- `grid > 1`
- `bx/by` 参与索引
- large-shape copy
- memory-plan 超预算负例

## 当前任务包

### 任务包 A: 固定三层边界与正式主链

#### 目标

把现有实现和文档统一到这三层：

- split 前语义规划
- split 后正式 plan 提取
- `BlackholeModule` direct host materialization

#### 影响范围

- `tasks/dev_design/*`
- `tilelang_repo/tilelang/engine/phase.py`
- `tilelang_repo/tilelang/engine/lower.py`
- `tilelang_repo/src/transform/*blackhole*`
- `tilelang_repo/src/target/*blackhole*`

#### 完成标准

- 不再把 external runner 视为主路径
- 不再把 split 后 matcher 视为唯一语义来源
- 不再混淆“语义规划”和“正式 plan 提取”

### 任务包 B: split 前语义规划收正

#### 目标

让 copy/gemm 的 Blackhole 关键语义在 split 前就保留下来，而不是只靠 split 后恢复。

#### 主要接入点

- `LowerTileOp` 的 Blackhole-aware branch

#### 核心产物

- `Blackhole-preserving TIR`

#### 完成标准

- copy/gemm/shared/dataflow/block 语义在 split 前仍可识别
- 不在这一步直接生成最终 `runtime_args / cb_configs / core_plan`

### 任务包 C: split 后 requirement extraction 收正

#### 目标

从 split 后 device kernel 稳定提取正式运行需求。

#### 主要接入点

- `LowerBlackholeOps`

#### 核心产物

- `blackhole.segment_plan`
- `blackhole.runtime_args`
- `blackhole.cb_requirements`

#### 完成标准

- runtime arg schema 支持 `current_work_linear_id`
- copy/gemm requirement 不再主要来自 runtime/module 猜测

### 任务包 D: memory planner 收正

#### 目标

把 `PlanBlackholeCB` 升级成正式 memory planner。

#### 主要接入点

- `PlanBlackholeCB`

#### 核心产物

- `blackhole.cb_configs`

#### 必做能力

- deterministic `cb_id` 分配
- `role + lifetime` 复用
- page size / num pages / data format / total size
- `1572864` bytes worker L1 hard check

#### 当前实现收正切口

- 先把 `cb_configs` 收成更正式的 memory object，而不是只保留最小 allocator 结果
- 当前这一轮优先补：
  - preserve extractor order as the current deterministic requirement order source
  - explicit `total_size_bytes`
  - explicit lifetime span（先至少覆盖 begin/end）
- 当前已落地的 planner 收正切口：
  - 先只做保守 reuse，不做全量 allocator
  - 仅在 requirement 满足这几个条件时允许复用同一个 memory object：
    - role 相同
    - `page_size/num_pages/data_format` 相同
    - lifetime 不重叠
  - `cb_configs` 继续表达“实际要 materialize 的 memory object”，而不是每条 requirement 一份
  - `cb_configs.requirement_names` 记录被合并的 requirement 名集合
  - `blackhole.cb_bindings` 显式记录：
    - `requirement_index`
    - `requirement_name`
    - `cb_id`
    - `cb_config_index`
- 下一轮 planner 收正切口：
  - 扩展兼容性判断，不再只覆盖“同型 requirement”
  - 继续扩 binding protocol，让下游不必只靠 `memory_object_name`/`cb_id` 二选一
- 在不打断现有 copy true E2E 的前提下，为后续真正的 lifetime/reuse planner 留协议位

#### 完成标准

- 正向 large-shape copy 合法执行
- 反向 oversubscription 编译失败

### 任务包 E: execution planner 收正

#### 目标

把 `AssignBlackholeCores` 升级成正式 execution planner。

#### 主要接入点

- `AssignBlackholeCores`

#### 核心产物

- `blackhole.core_plan`

#### 必做能力

- `logical_grid_x/y`
- `linearization = row_major`
- `physical_cores`
- `work_packets`

#### 完成标准

- `blockIdx` 不再被 codegen 常量化
- `grid > 1` 的 logical block 语义能进入正式 plan

### 任务包 F: `BlackholeModule` direct path 收正

#### 目标

把 `BlackholeModule` 收正成唯一正式 host-side execution path。

#### 主要接入点

- `rt_mod_blackhole`
- `BlackholeModule`

#### 核心产物

- `ExecutableSpec`
- direct host materialization / launch / readback

#### 必做能力

- 直接 materialize：
  - `Program`
  - `CreateCircularBuffer`
  - `CreateKernel`
  - `SetRuntimeArgs`
  - `ConfigureDeviceWithProgram`
  - `LaunchProgram`
- TileLang 正式 host callable 可直接执行

#### 完成标准

- 不再把 `spec.json -> runner` 作为正式执行路径
- 不再要求手动按 `"main"` 调内部符号

### 任务包 G: copy 正式 E2E

#### 目标

用 copy 完成首条 Blackhole 正式 compiler/runtime 主链。

#### 必测 case

- `32x32`
- `32x64`
- `64x32`
- 至少一个 `grid > 1` 且 `bx/by` 参与索引的 staged copy
- 至少一个总数据量大于 `1.5MB` 的 large-shape copy
- 至少一个 per-core memory plan 超预算的负例

#### 完成标准

- 通过 TileLang host callable + `BlackholeModule` direct path 执行
- 与 PyTorch 参考一致

### 任务包 H: 分批接回中后段通用 pass

#### 目标

在不破坏 copy 正式 E2E 的前提下，把通用主链继续收正。

#### 建议顺序

1. `FlattenBuffer`
2. `VectorizeLoop`
3. `StorageRewrite`

#### 完成标准

- split 前语义规划不被破坏
- split 后 attrs / plan 仍稳定
- copy 正式 E2E 不回退

### 任务包 I: GEMM 接入

#### 目标

让 GEMM 只复用 copy 已建立的正式主链。

#### 前置条件

- `BlackholeModule` direct path 已稳定
- `cb_configs` 已成为正式 memory plan
- `core_plan` 已成为正式 execution plan
- copy 的 large-shape / grid>1 / oversubscription 验证已通过

#### 完成标准

- GEMM 不新增 runtime-only 或 runner-only 路径
- 最小 GEMM direct E2E 通过

## 推荐执行顺序

当前建议按下面顺序推进：

1. 任务包 A: 固定三层边界
2. 任务包 B: split 前语义规划
3. 任务包 C: split 后 requirement extraction
4. 任务包 D: memory planner
5. 任务包 E: execution planner
6. 任务包 F: `BlackholeModule` direct path
7. 任务包 G: copy 正式 E2E
8. 任务包 H: 分批接回中后段通用 pass
9. 任务包 I: GEMM 接入

## 近期交付建议

### 交付 1: 正式 plan 骨架

- `runtime_args`
- `cb_configs`
- `core_plan`
- `BlackholeModule` direct path 骨架

### 交付 2: grid>1 + large-shape copy

- 逻辑 block 保真
- large-shape copy
- memory-plan correctness

### 交付 3: 负例与通用 pass 回收

- oversubscription compile-time error
- 分批接回中后段通用 pass

### 交付 4: GEMM bring-up

- 只复用已稳定的正式主链

## 验收口径

后续每个实现任务都必须明确：

- 属于哪个任务包
- 它是 split 前语义规划、split 后正式 plan 提取，还是 host-side materialization
- 影响哪些正式产物：
  - `Blackhole-preserving TIR`
  - `blackhole.runtime_args`
  - `blackhole.cb_configs`
  - `blackhole.core_plan`
  - `ExecutableSpec`
- 做了哪些 direct host path 结构验证与执行验证

如果某项实现改变了本文件的执行顺序或边界假设，应先更新本文件，再继续编码。
