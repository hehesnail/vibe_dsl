# Stage 2 当前具体开发任务拆解

## 基本定位

- **状态**: 当前活动任务规划
- **前置总体设计**: `final_blackhole_backend_redesign.md`
- **前置阶段设计**:
  - `stage2_pass_reuse_matrix.md`
  - `stage2_single_core_pass_integration.md`
  - `stage2_blackhole_logical_block_launch_plan.md`

本文件只回答一个问题：

- **基于当前最新实现状态，接下来按什么顺序推进 Blackhole 后端开发任务**

它不替代总体设计，也不替代阶段设计；它只把“下一批要做的事”压缩成可执行任务包。

## 当前判断

结合 `progress.md`、当前源码入口和已有 true E2E 状态，Blackhole 现在已经具备：

- `host entry -> device kernel -> ExecutableSpec -> runner` 的基本闭环
- staged single-core copy 在 `32x32`、`32x64`、`64x32` 上的最小真执行闭环
- `LowerBlackholeOps -> PlanBlackholeCB -> AssignBlackholeCores` 这条 Blackhole device pass 子链

但当前实现仍有三个硬阻塞，决定了后续任务不能并行乱推：

1. `blackhole.core_plan` 仍是摘要信息，不是可执行 work distribution plan
2. `CodeGenBlackhole` 仍把 `blockIdx.x/y/z` 常量化成 `0`
3. `LowerBlackholeOps` 仍依赖一条受控 staged-copy 形态，尚未吃下更稳定的 split 后规范化 kernel 结构

因此当前阶段的正确推进顺序不是直接上 GEMM，而是：

1. 先收正 single-core execution plan
2. 再把 copy lowering 接到更稳定的 device-kernel 形态
3. 再逐步接回剩余通用 pass
4. 最后再让 GEMM 复用这条路径

## 任务总原则

### 1. 先收协议，再改执行

凡是会影响 `rt_mod_blackhole`、`BlackholeModule`、runner、`CodeGenBlackhole` 协作边界的事项，必须先收口协议字段，再做实现。

### 2. single-core 也按 multi-core 可扩展模型设计

当前可以只实现单核，但单核语义必须表示为：

- logical grid 保留
- 一个 physical core 顺序处理多个 logical blocks
- host/runtime 显式 materialize work packet

而不是：

- device code 写死 `blockIdx=0`
- runner 写死只跑一次 `{0, 0}`

### 3. copy 继续作为主验收对象

GEMM 只能复用 copy 已经跑通的协议和 pass 结构，不再允许先为 GEMM 扩展 runtime 特化。

## 当前任务包

### 任务包 A: `core_plan` 收正为可执行单核 work plan

#### 目标

把当前摘要型 `blackhole.core_plan` 收正为 host/runtime 真正可执行的 single-core logical work distribution plan。

#### 影响范围

- `tilelang_repo/src/transform/assign_blackhole_cores.cc`
- `tilelang_repo/src/target/rt_mod_blackhole.cc`
- `tilelang_repo/src/target/blackhole_module.h`
- `tilelang_repo/src/target/blackhole_module.cc`
- `tilelang_repo/tools/blackhole_runner/runner.cpp`

#### 当前问题

当前 `AssignBlackholeCores` 只写入：

- `grid_x`
- `grid_y`
- `cores_needed`
- `work_per_core`
- `core_grid_x`
- `core_grid_y`

这只能描述“理论上需要多少核”，不能描述：

- logical block 如何线性化
- 单核当前负责哪段 logical work
- runner 是否应重复设 runtime args / 多次 enqueue / 还是由单个 kernel 内部消费 work packet

#### 协议变化

`blackhole.core_plan` / `ExecutableSpec::core_plan` 至少应扩充到能表达：

- `logical_grid_x`
- `logical_grid_y`
- `linearization`
- `physical_core_count`
- `physical_cores`
- `work_packets`
  - 每个 packet 至少包含：
    - `core_x`
    - `core_y`
    - `work_offset`
    - `work_count`

当前阶段允许只落单个 packet，但字段形态不能再是摘要值。

#### 验证方式

- pass 级检查 `blackhole.core_plan` attr 结构
- `ExecutableSpec` / `spec.json` roundtrip 检查
- runner 能正确解析新 schema
- single-core copy 的 spec-driven 执行继续通过

### 任务包 B: 去掉 `blockIdx -> 0` 常量化

#### 目标

让 device code 不再通过 codegen 常量折叠抹掉 logical block 语义。

#### 影响范围

- `tilelang_repo/src/target/codegen_blackhole.cc`
- 可能联动 `LowerBlackholeOps` 生成的 runtime arg / kernel ABI
- 可能联动 runner 对 work packet 的 materialization

#### 当前问题

`CodeGenBlackhole::BindThreadIndex()` 当前直接把：

- `blockIdx.x -> 0`
- `blockIdx.y -> 0`
- `blockIdx.z -> 0`

这会让后续所有基于 `T.Kernel(grid_x, grid_y)` 的程序失真，即使 `AssignBlackholeCores` 已经分析出了 grid。

#### 实现要求

本任务不要求一步到位支持 multi-core，但至少要做到：

- device code 可见“当前 logical block”
- 该信息来自 execution plan / runtime arg / work packet
- 不再由 codegen 常量硬编码

推荐最小落地方式：

- 先引入 single-core `current_work_linear_id` 或 `(current_block_x, current_block_y)` 运行时值
- runner 对单核 work packet 进行 materialize
- codegen 将 `blockIdx` 绑定到这些运行时值

#### 验证方式

- 结构测试：生成源码中不再出现 `0 /* core_x/core_y */`
- IR / codegen 测试：`blockIdx` 参与 tile index 计算时仍保留动态来源
- 执行测试：至少新增一个依赖 logical block 的 single-core copy case

### 任务包 C: 把 single-core 执行改成“单核串行处理 logical blocks”

#### 目标

把当前“只在 `{0,0}` 上跑一次”的 bring-up 路径，收正成正式 single-core execution model。

#### 影响范围

- `tilelang_repo/tools/blackhole_runner/runner.cpp`
- `tilelang_repo/src/target/blackhole_module.cc`
- `tilelang_repo/src/target/rt_mod_blackhole.cc`
- 与任务包 A/B 联动

#### 当前问题

runner 当前固定：

- `constexpr CoreCoord core = {0, 0};`
- 每个 kernel 只设一次 runtime args
- 没有按 logical block/work packet 驱动执行

这只能证明 kernel 能被送进 TT-Metal，不能证明 single-core block semantics 已经成立。

#### 实现要求

最小正式模型：

- physical core 仍可固定为 `{0, 0}`
- 但该 core 需按 `work_packets` 串行处理 `[0, logical_block_count)` 的 logical blocks
- runtime args / launch ABI 需要能区分当前处理的 logical block

#### 验证方式

- spec-driven 执行能覆盖 `grid_x * grid_y > 1` 的 case
- direct-call 执行能覆盖同一 case
- 输出与 PyTorch 参考一致

### 任务包 D: 扩展 `LowerBlackholeOps`，直接消费更稳定的 split 后 device kernel 形态

#### 目标

把 copy lowering 从“当前受控 staged-copy 形态可识别”推进到“更一般、经过更多规范化后的 split device kernel 仍可识别”。

#### 影响范围

- `tilelang_repo/src/transform/lower_blackhole_ops.cc`
- `tilelang_repo/src/transform/lower_blackhole_ops.h`
- `tests/transform/`
- `tests/target/`

#### 当前问题

当前 `LowerBlackholeOps` 虽然已经补到支持矩形 staged copy，但 matcher 仍明显偏向：

- staged loop 形态
- copy-first runtime arg 模板
- 少量固定结构推导

一旦更早接回 `FlattenBuffer` / `VectorizeLoop` / `StorageRewrite`，当前识别大概率又会失效。

#### 实现要求

本任务要先把 matcher 的关注点从“局部 BufferStore 形态”继续前移到“整段 copy/dataflow 结构”，至少做到：

- 优先按 split 后 device kernel 的函数体级 copy 片段识别
- staged copy 的 tile index / tile bytes 继续来自实际 DSL tile shape
- 不为更晚的 runtime fallback 扩接口

#### 验证方式

- pass 级 attrs 检查：
  - `blackhole.runtime_args`
  - `blackhole.segment_plan`
  - `blackhole.cb_requirements`
- lowered TIR 检查：
  - `tl.blackhole.read_tile_to_cb`
  - `tl.blackhole.write_tile_from_cb`
- 新增多种 DSL tile shape / loop shape 回归

### 任务包 E: 分批接回 `FlattenBuffer` / `VectorizeLoop` / `StorageRewrite`

#### 目标

把当前 Blackhole 的“受控 pass 子链”继续收正到更完整的通用 TIR 主链。

#### 影响范围

- `tilelang_repo/tilelang/engine/lower.py`
- 与 Blackhole target build/lower 相关的 pass 组装位置
- `LowerBlackholeOps` 的结构兼容性测试

#### 当前问题

当前问题已经不是“这些 pass 要不要复用”，而是：

- 它们一接回就会打断 copy 识别

因此这项工作必须在任务包 D 之后做，而不能反过来。

#### 实现要求

建议按以下顺序逐个接回并验证，而不是一次性恢复：

1. `FlattenBuffer`
2. `VectorizeLoop`
3. `StorageRewrite`

每接回一个 pass，都要先验证：

- copy attrs 是否还在
- copy builtin 是否还在
- spec/codegen/runner 真执行是否还在

#### 验证方式

- 单独 pass 级结构回归
- `spec.json -> runner`
- `artifact.codegen_mod["main"](...)`

### 任务包 F: 启动 GEMM 接入，但只复用既有主链

#### 目标

在 copy 路径稳定后，按同一结构接入 GEMM。

#### 影响范围

- `LowerTileOp` 的 Blackhole-aware GEMM lowering
- `LowerBlackholeOps` 的 GEMM schema 提取
- `PlanBlackholeCB` 的 reader/compute/writer 资源规划
- `CodeGenBlackhole` 的 GEMM builtin / kernel emission

#### 前置条件

只有以下条件同时满足，才应正式进入 GEMM：

- `core_plan` 已收正为 work-plan 结构
- `blockIdx` 不再常量化
- single-core logical-block copy 已闭环
- 至少一部分通用中后段 pass 已重新接回

#### 不允许的做法

- 不为 GEMM 再扩一套 runtime 专用协议
- 不在 runner 里直接拼 GEMM 语义
- 不绕开 copy 已建立的 `ExecutableSpec -> runner` 主路径

#### 验证方式

- 先做 pass/schema 级验证
- 再做最小 GEMM true E2E

## 推荐执行顺序

当前建议按下面顺序推进：

1. 任务包 A: `core_plan` 协议收正
2. 任务包 B: `blockIdx` 运行时绑定
3. 任务包 C: runner 单核串行 logical work
4. 任务包 D: `LowerBlackholeOps` 更稳定形态识别
5. 任务包 E: 分批接回剩余通用 pass
6. 任务包 F: GEMM 接入

原因很直接：

- A/B/C 决定 single-core execution model 是否真实成立
- D/E 决定 copy 是否真正回到 compiler 主链
- F 才能在更稳定的底座上展开

## 近期交付建议

为了保持每次改动小而完整，建议把最近三轮实现切成如下交付：

### 交付 1: single-core work-plan MVP

- 收正 `blackhole.core_plan`
- runner 解析/消费 `work_packets`
- `blockIdx` 不再常量化
- 新增一个 `grid>1` 的 single-core copy 结构测试

### 交付 2: logical-block true execution

- runner 支持单核串行 logical blocks
- `spec.json -> runner` 跑通 `grid>1` copy
- direct-call 路径跑通同一 case

### 交付 3: pass 主链继续收正

- 扩展 `LowerBlackholeOps`
- 分批接回 `FlattenBuffer` / `VectorizeLoop` / `StorageRewrite`
- 保持 copy true E2E 不回退

### 交付 4: GEMM bring-up

- 只复用前述主链
- 从最小 single-core GEMM case 开始

## 验收口径

本轮任务规划完成后，后续每个实现任务都应明确标注：

- 属于哪个任务包
- 影响哪些协议字段 / 文件
- 做了哪些结构验证
- 做了哪些真执行验证
- 还有哪些未完成项和限制

如果某项改动需要改变本文件的执行顺序或前置关系，应先更新本文件，再继续实现。
