# TileLang Blackhole 后端最终重构设计

## 基本信息

- **文档ID**: `final_blackhole_backend_redesign`
- **日期**: 2026-03-19（创建），2026-03-30（最近更新）
- **状态**: 当前唯一权威总体设计
- **适用范围**: `tilelang_repo` Blackhole 后端、host/device 主链、运行时执行路径、相关阶段设计

## 1. 当前目标

Blackhole 后端当前的正式目标收敛为三点：

1. **后端主产物是 `ExecutableSpec`，不是单个 kernel 字符串**
2. **实现路径必须复用 TileLang / TVM 的正式 PrimFunc/TIR 与 host/device 主链**
3. **正式执行路径必须是 `BlackholeModule` 进程内 direct host path，而不是 external runner**

因此 Blackhole 后端不再以“写出一个能跑的 TT-Metal kernel 字符串”或“`spec.json -> runner` 能跑”为完成标准，而是以：

- DSL / PrimFunc / TIR 主链保持成立
- tile/dataflow/shared/block 语义在正确层级被保留并转换
- split 后 device kernel 产出正式 execution plan / memory plan / kernel ABI
- `ExecutableSpec -> BlackholeModule direct host materialization -> TT-Metal launch`

作为正式目标。

## 2. 当前状态（2026-03-30）

### 已完成

- `ExecutableSpec`、`rt_mod_blackhole`、`BlackholeModule` direct host path 已落地
- Blackhole 已接回 `AnnotateDeviceRegions / SplitHostDevice / MakePackedAPI / LowerDeviceKernelLaunch`
- split 前语义规划（`AnnotateBlackholeCopySemantics`）、split 后 plan 提取（`LowerBlackholeOps` → `PlanBlackholeCB` → `AssignBlackholeCores`）、host-side materialization（`BlackholeModule`）三层已分清
- CB identity 唯一协议已收正：`LowerBlackholeOps` 统一产出 `requirement_index`，`PlanBlackholeCB` 回写最终 `cb_id`，codegen 直读
- `BlackholeDeviceResourceCanonicalization` 已引入 `StorageRank::kBlackholeCB` / `kBlackholeAccumulator`，generic pass 不再误解 Blackhole 资源
- Copy single-core E2E 通过（16 passed, 1 xfailed）
- GEMM single-core E2E 通过（4 passed, 1 skipped）：`transpose_B` + host tilize/untilize 已补齐
- Copy multi-core E2E 通过（18 passed, 1 xfailed / runtime 6 passed）
- GEMM multi-core formal direct host path E2E 通过（7 passed）：`num_k_tiles`、writer output-tile consumption、`transpose_B` tiled-B reader index 已收正
- `scratch_l1_buffer_addr32` 全链路移除
- legacy external runner 已删除

### 已知结构限制

- `PlanBlackholeCB` 仍是 MVP allocator，非正式 memory planner
- `StorageRewrite` 与 Blackhole CB 模型不兼容（永久排除）
- copy 用 `fused_dataflow` 单 kernel，GEMM 用 3-kernel（后续统一为 reader+writer 模型是架构债）
- TT-Metal contract 收正未完成项：P0（compute ABI / dtype 分层）已完成到统一 `compute_contract`，并已打通 DSL producer -> attrs/spec -> runtime 主链；P3（unified runtime work schema + accessor/common-runtime schema + compile-time ABI schema）在 current copy/GEMM formal surface 上已完成收口：kernel-level shared `common_runtime_args` 已打通到 `SetCommonRuntimeArgs` host materialization，accessor `args_config_bits` 已与 TT-Metal `ArgConfig.raw()` 对齐并进入 compile-time ABI / host materialization 真链路；更宽的 accessor-derived CRTA / non-tile execution surface 已转移到 P4 或后续专项；P4（copy 泛化）已完成 interleaved DRAM stick/page 主路径，支持 `M x W`（`M % 32 == 0`）与静态 offset subrange，当前 formal direct-path boundary 为 `transport_page_size` 需 64B 对齐；P5（synchronization）已完成 program-local semaphore schema、kernel binding、最小 dataflow semaphore builtin、worker producer/consumer direct-runtime E2E，以及 `logical_core_noc_x/y -> KernelSpec.remote_core_descriptors` 最小 remote-core descriptor formalization，但 multicast / global semaphore / pass-level producer 仍未做，见 `stage2d_ttmetal_contract_audit.md` 和 `stage4_semaphore_schema.md`

### 当前活动

- **TT-Metal contract formalization**
- Stage 3 formal direct host path 已完成，设计文档：`stage3_multicore_design.md`
- `tilelang.compile(..., execution_backend="tvm_ffi")` 的 Blackhole wrapper/export path 已恢复；host C codegen 已支持 packed call 结果表达式
- P5 当前已从“零语义层”推进到：program-local semaphore plan、kernel-level semaphore binding、最小 device-side dataflow semaphore builtin、以及 worker producer/consumer direct-runtime E2E
- backend cleanup review 与重文件边界拆分草案已建档：`stage4_backend_cleanup_roadmap.md`；cleanup A1/A3 已完成，A2 已落首轮 schema-driven buffer materialization 骨架，B1 已完成四轮 `BlackholeModule` helper 边界拆分，B2 已完成两轮 staged-copy boundary/geometry/index 收敛，B3 已收紧为只消费 explicit `blackhole.cb_requirements`，C1 已收正 compile-time-only accessor codegen 边界，C2 已完成首轮 synchronization host/runtime boundary 收紧

## 3. 正式架构

### 3.1 总体结构

```text
TileLang DSL
  -> PrimFunc / TIR
  -> LowerAndLegalize
  -> split 前 Blackhole 语义规划
      -> LowerTileOp(Blackhole-aware)
      -> 通用 planning / normalize 子集
  -> AnnotateDeviceRegions
  -> SplitHostDevice
  -> AnnotateReadOnlyParams
  -> MakePackedAPI
  -> LowerDeviceKernelLaunch
  -> split 后 Blackhole 正式 plan 提取
      -> LowerBlackholeOps
      -> PlanBlackholeCB
      -> AssignBlackholeCores
  -> rt_mod_blackhole
      -> Extract ExecutableSpec
      -> Emit kernel source(s)
      -> Build BlackholeModule(spec)
  -> BlackholeModule
      -> Program / CreateCircularBuffer / CreateKernel
      -> SetRuntimeArgs / ConfigureDeviceWithProgram / LaunchProgram
      -> readback
```

### 3.2 设计原则

1. **以后端执行模型约束 lowering，不以后端打印器约束执行模型**
2. **优先复用现有 PrimFunc/TIR 与 host/device 主链，只在少量 target-aware 边界定制**
3. **host/device 划分与 Packed API 参数语义沿用 TileLang / TVM 正式模型**
4. **split 前做语义规划，split 后做正式 plan 提取，host side 做 materialization**
5. **compile-time attrs/schema 与 runtime args 必须严格分层**
6. **逻辑 block/grid 语义应保留在 TIR/pass 中，不在 runtime 侧重建**
7. **multi-core 主要由 host/runtime materialize，不由 codegen 主导**
8. **`SplitBlackholeKernel`、`blackhole_module_direct.cc`、external runner 都不再是主路径设计前提**

### 3.3 三层分工

#### A. split 前语义规划

职责：

- 保留 copy/gemm/tile/shared/pipeline/block 语义
- 稳定成 Blackhole 后续仍可识别的 TIR

产物：

- `Blackhole-preserving TIR`

不在这一层做的事：

- 不生成最终 `runtime_args`
- 不分配最终 `cb_id`
- 不生成最终 `core_plan`
- 不定义 host-side launch/materialization 细节

#### B. split 后正式 plan 提取

职责：

- 面向 split 后 device kernel
- 提取 runtime ABI、memory plan、execution plan

产物：

- `blackhole.segment_plan`
- `blackhole.runtime_args`
- `blackhole.cb_requirements`
- `blackhole.cb_configs`
- `blackhole.core_plan`

#### C. host-side materialization

职责：

- 直接 materialize TT-Metal host objects
- 按正式 plan 完成 launch/readback

产物：

- 真正执行结果

## 4. 模块边界

### 4.1 `LowerTileOp`

这是 Blackhole 最关键的 split 前 target-aware 接入点。

长期要求：

- 继续在整个 PrimFunc 上工作
- 保留普通控制流、标量计算、索引与边界逻辑
- 对 `tl.copy` / `tl.gemm` 等 tile 语义增加 Blackhole-aware 分支
- 不再把这些语义完全压碎后再让 Blackhole 从晚期 loop 恢复

职责：

- 只负责 split 前语义规划
- 输出 Blackhole-preserving TIR

### 4.2 `LowerBlackholeOps`

职责：

- 消费 split 后 device kernel
- 提取 segment / runtime arg schema / CB requirements
- 产出 `blackhole.segment_plan`、`blackhole.runtime_args`、`blackhole.cb_requirements`
- IR body 中所有 blackhole builtin 的 CB 参数统一使用 `requirement_index`（cb_requirements 数组下标）

不再承担：

- 分配最终 cb_id（由 PlanBlackholeCB 统一分配）
- 区分 copy / GEMM 使用不同的 CB identity 体系
- 产出 GEMM placeholder（-1/-2/-3）或 `blackhole.gemm_cb_placeholders`
- 从晚期普通 loop 中恢复绝大多数 copy/gemm 语义
- 猜 PrimFunc 参数结构
- 用 runtime 特判兜底 device kernel 结构

### 4.3 `PlanBlackholeCB`

职责：

- 从 `blackhole.cb_requirements` 收敛到正式 `blackhole.cb_configs`
- 形成 per-device-kernel memory plan
- **回写 IR body**：把所有 blackhole builtin 中的 `requirement_index` 替换成分配后的最终 `cb_id`

至少覆盖：

- `cb_id`
- role
- page size / num pages / format
- 生命周期与复用
- per-core L1 峰值约束
- requirement_index → cb_id 映射（`cb_bindings` 以 `requirement_index` 为主键）

正式边界：

- planner 的正式输入是 explicit `blackhole.cb_requirements`
- 不再默认从 `alloc_shared` / shared allocation 形态推断 planner 输入
- 如果缺失 `blackhole.cb_requirements`，应显式失败，而不是让 planner 自己猜一套 CB requirement

### 4.4 `AssignBlackholeCores`

职责：

- 保留并消费逻辑 grid/block 语义
- 生成 host/runtime 消费的 execution plan

至少覆盖：

- `logical_grid_x/y`
- linearization policy
- `physical_cores`
- `work_packets`

### 4.5 `rt_mod_blackhole`

职责：

- 从 split 后 device kernel 及其 `blackhole.*` attrs 提取 `ExecutableSpec`
- 构造 `BlackholeModule(spec)`

约束：

- 只消费 pass 产出的 device-side 语义和 schema
- 不再长期把“没有 `calling_conv` 的 PrimFunc”视为正式 device kernel 模型
- 不再定义 PrimFunc 参数类别、host/device 边界或 runtime 参数意义

### 4.6 `BlackholeModule`

职责：

- 作为 TVM runtime adapter / host-side 执行载体
- 在进程内 direct materialize TT-Metal host objects
- 完成 launch / readback

约束：

- 不再通过 external runner 作为正式主路径
- 不再长期承担 Packed API 或 host/device 语义定义
- 输入输出 buffer、scalar、dynamic shape 如何映射到 runtime args，必须由 PrimFunc + pass schema 决定
- 不再按位置规则猜 ABI

### 4.7 external runner

状态：

- 已删除，仅保留历史语境中的设计参考价值

约束：

- 不是正式执行路径
- 不是阶段完成标准
- 不再重新引入

## 5. 核心协议

### 5.1 `ExecutableSpec`

当前主协议结构保持：

- `CBConfig`
- `CorePlan`
- `KernelArgSpec`
- `KernelSpec`
- `ExecutableSpec`

后续工作重点不是继续改协议名字，而是让这些字段的来源真正回到 split 后 device-side attrs/schema。

允许扩展的方向：

- `CorePlan` 补足 logical grid / work packet 语义
- `KernelSpec` 补足 kernel-level ABI / launch information
- `CBConfig` 补足 memory hierarchy materialization 所需字段

不允许扩展的方向：

- 为 runner 新增脱离 TIR 的平行 ABI
- 在 module/runtime 里定义第二套 host/device 语义

### 5.2 Attr Schema

当前主线只保留：

- `blackhole.cb_requirements`
- `blackhole.cb_configs`
- `blackhole.cb_bindings`（主键为 `requirement_index`，`requirement_name` 仅辅助调试）
- `blackhole.core_plan`
- `blackhole.segment_plan`
- `blackhole.runtime_args`

已淘汰：

- `blackhole.gemm_cb_placeholders`（由 CB identity 唯一协议收正取代，见 `stage2d_cb_identity_protocol.md`）

旧协议不再扩展。

## 6. 算子策略

### 6.1 Copy

copy 的正式目标是：

- 以 TileLang 原始 `T.copy` 语义为验收对象
- 在 Blackhole 上收敛成 `global -> shared/CB -> global`
- 经历 split 前语义规划、host/device 主链、split 后正式 plan 提取、`BlackholeModule` direct host path 后执行

正式验证至少覆盖：

- `32x32`
- `32x64`
- `64x32`
- 至少一个 `grid > 1` 且 `bx/by` 参与索引的 case
- 至少一个总数据量大于 `1.5MB` 的 large-shape copy case

**当前 segment 模型**：纯 copy 使用 `fused_dataflow` 单 kernel（BRISC 顺序完成 read+write）。`SplitBlackholeKernel` 对无 compute op 的函数是 strict no-op。

**架构债**：技术上 copy 可以统一进 reader+writer 2-kernel 模型（BRISC read + NCRISC write 并行），使 `SplitBlackholeKernel` 统一覆盖所有情形，并消除 `rt_mod_blackhole` / `BlackholeModule` 对 `fused_dataflow` 和多 kernel 两种 schema 的双重处理。触发条件：GEMM E2E 稳定后再做，不在 Stage 2D 内。

### 6.2 GEMM

GEMM 的后续集成必须复用与 copy 相同的结构：

- 不再接受 runtime 侧大块 GEMM 特化
- reader / compute / writer 语义必须主要来自 split 前语义规划和 split 后正式 plan 提取

**当前 segment 模型**：GEMM 使用 3-kernel（reader BRISC + compute TRISC + writer NCRISC），由 `SplitBlackholeKernel` pass 产出 `blackhole.segment_plan`。

### 6.3 Multi-core

multi-core 的主要实现位置保持不变：

- host/runtime
- 基于 `CorePlan`
- materialize per-core runtime args / work packets

## 7. 分阶段路线

### Stage 0: 协议与执行载体

目标：

- attrs 收口到 `blackhole.*`
- 引入 `ExecutableSpec`
- 改造 `rt_mod_blackhole`
- 引入 `BlackholeModule`

状态：

- **已基本完成**

### Stage 1: single-core copy bring-up

目标：

- 证明最小 copy 路径能执行

状态：

- **已完成**

说明：

- 该阶段的 runner/scratch/固定 schema 只视为 bring-up 过渡路径，不再扩大为正式主线

### Stage 2A: pass 主链接入收正

目标：

- 复用 TileLang / TVM 正式主链
- 建立 split 前语义规划 / split 后 plan 提取 / host-side materialization 三层

### Stage 2B: single-core copy 正式主链

目标：

- copy 由 pass 主导并走 `BlackholeModule` direct host path 完成正式 E2E

状态：

- **已完成**（18 passed, 1 skipped；含 grid>1 / large-shape / oversubscription 负例）

### Stage 2C: split-before 语义规划

> 注：本阶段在实施过程中从原计划的"GEMM 语义集成"调整为先落地 split-before 语义规划边界，GEMM 接入顺延至 2C 完成后进行。

目标：

- 新增 `AnnotateBlackholeCopySemantics` pass
- 明确 split 前语义规划 / split 后 matcher / codegen 三者的职责边界
- copy 语义不再主要靠 split 后 matcher 从晚期 loop 恢复
- 为 GEMM 接入准备最小 semantic schema

状态：

- **已完成**（15 passed, 5 skipped, 1 xfailed）
- `AnnotateBlackholeCopySemantics` 已落地；`FlattenBuffer` / `VectorizeLoop` 已专项验证
- `StorageRewrite` 确认不兼容 Blackhole CB 模型，永久排除；Phase 4 需先加 shared-scope 豁免

### Stage 2D: single-core GEMM 语义集成 + true E2E

目标：

- copy 与 GEMM 都由正式主链执行并完成 direct host-device E2E

状态：

- **已完成** ✅
- Steps 1-6 全部完成
- CB identity 唯一协议已收正（设计文档 `stage2d_cb_identity_protocol.md`）
- GEMM 根因已修复：`transpose_B` 丢失 + host row-major upload 无 tilize/untilize
- 额外收正：`scratch_l1` 全链路移除、copy codegen 统一、`GetRuntimeArgVarForBuffer` preferred_kind 重构
- 测试：copy 16 passed / 1 xfailed，GEMM 4 passed / 1 skipped

架构债（不在 2D 内）：copy 统一进 reader+writer 2-kernel 模型，消除 fused_dataflow / 多 kernel 双重 schema。

### Stage 2E: Blackhole 设备资源 IR 语义扩展

目标：

- 扩展 TIR `StorageRank` 类型系统，为 Blackhole CB 和 Dst 累加器引入正式 IR 资源类型
- 新增 `BlackholeDeviceResourceCanonicalization` pass，在 `SplitHostDevice` 前完成 scope 替换和 allocation 重定位
- 解除 GEMM lowering 的三个 generic pass 阻塞

设计文档：`tasks/dev_design/stage2e_blackhole_device_resource_semantics.md`

核心设计：

- 新增 `StorageRank::kBlackholeCB = 13`、`StorageRank::kBlackholeAccumulator = 14`
- `shared.dyn` → `blackhole.cb.input` / `blackhole.cb.output`（CB 是 FIFO 队列，不是共享内存）
- `local.fragment` → `blackhole.acc`（Dst 是寄存器文件，不是可寻址内存）
- generic pass 自然正确（rank 不匹配 → 跳过），无需特判
- 与 WMMA/MMA/Metal/AMX 在 TVM 中的既有 StorageRank 扩展完全同构

状态：

- **已完成**（StorageRank 扩展、canonicalization pass 与 GEMM lower 解锁均已落地）

### Stage 3: multi-core runtime 调度

目标：

- 让 copy 和 GEMM 在多个 Tensix 核心上真正并行执行
- `AssignBlackholeCores` 解除 `cores_needed=1` 限制
- `BlackholeModule` 从 N 个 Program 串行 enqueue 改为 1 个 Program 多核 launch

状态：

- **已完成** ✅
- 设计文档：`stage3_multicore_design.md`
- 关键结论：copy/GEMM 多核不需要改 lowering/codegen，只需 host 侧分发 + DSL kernel 用 `bx/by` 索引
- 已落实结果：
  - `AssignBlackholeCores` 已解除 `cores_needed=1`
  - `BlackholeModule` 已切到单 `Program` 多核 launch
  - copy / GEMM multi-core direct host path 已通过
  - Stage 3 后独立修复的 `tvm_ffi` wrapper/export blocker 不影响本阶段主结论

## 8. 架构可扩展性评估

### 8.1 当前架构的可扩展性边界

当前三层模型的核心路径是"先压碎（LowerTileOp 标量化）再恢复（LowerBlackholeOps pattern match）"。这个模式的可扩展性随算子复杂度急剧下降：

| 算子类型 | Pattern Match 可行性 | 多核模型 | 当前架构是否覆盖 |
|---------|---------------------|---------|----------------|
| Copy | 极简单 (`dst[i,j]=src[i,j]`) | 数据并行 | 已覆盖 |
| GEMM | 可识别但脆弱 | M/N 维度切分 | 部分覆盖 |
| Element-wise | 中等 | 数据并行 | **未覆盖** |
| Reduction | 复杂 | 需跨 tile 累加 | **未覆盖** |
| Softmax | 很复杂 | 需两趟扫描 | **未覆盖** |
| FlashAttention | **基本不可能从标量 TIR 恢复** | 需核间数据流 | **未覆盖** |

### 8.2 根本性限制

1. **语义恢复不可扩展**：每增加一种 op 需要新的 pattern matcher，到 FlashAttention 级别的融合算子时 pattern match 完全失效
2. **其他 IR ops 未处理**：element-wise、reduction、transpose、typecast 等在 Blackhole 上需要不同的 TT-Metal API，当前设计完全没有涉及
3. **多核模型有上限**：当前 `work_packets` 能表达数据并行，但无法表达核间数据流

### 8.3 中期架构演进方向：Operation-Level Lowering

与其"先压碎再恢复"，更可扩展的方案是**保留 tile-level 操作语义直到最后一刻**：

```text
TileLang DSL → PrimFunc/TIR（保留 T.copy/T.gemm/T.reduce/T.elementwise）
  → [Blackhole target 时跳过 LowerTileOp 的标量降级]
  → MapBlackholeOps：直接从 tile-level op 映射到 TT-Metal 操作序列
  → 同时确定 CB requirements / kernel split / runtime args
  → codegen 只做格式化
```

### 8.4 对当前规划的影响

- **Stage 0-2E 已完成**：direct path + copy/GEMM single-core E2E 是无论哪种架构都需要的基础设施
- **Stage 3（多核）不受影响**：多核只是 host 侧分发，不涉及语义恢复问题
- **后续算子扩展**：如果要支持 element-wise/reduction/softmax 等，需要考虑 operation-level lowering 路线

## 9. 关键源码审查结论

### 9.1 历史 runner 实现曾是 direct path 的参考蓝本

已删除的历史 runner 实现曾经提供过一套完整、正确的 TT-Metal 执行参考，核心包括：
- `create_circular_buffers()` — 按 spec 创建所有 CB
- `build_runtime_args()` — 按 `KernelArgSpec.kind` 逐项构造 runtime args
- work-packet 迭代 — 遍历 `work_packets` 为每个 work unit 执行独立 program

Direct path 的实现本质上就是把这套 host-side materialization 逻辑收进 `BlackholeModule` 的 `ExecuteDirect()` 方法中。当前仓库已不再保留独立 runner 代码。

### 9.2 CB 创建是 host-side 必做项

`CreateCircularBuffer` 是 TT-Metal 编程的**基本必需步骤**，不是可选优化。没有它 kernel 里的 `cb_reserve_back(cb_id, ...)` 会失败。

### 9.3 SplitBlackholeKernel 推迟到 GEMM 阶段

Copy 操作本质是 DRAM→L1→DRAM 的数据搬运，不涉及 TRISC 计算，单 kernel 在 BRISC 上（fused_dataflow）完全正确。GEMM 才需要拆分为 Reader/Compute/Writer 三个独立 kernel。

### 9.4 split-before 语义规划方案

推荐方案 A：在 `LowerTileOp` 之后、`FlattenBuffer` 之前新增 `AnnotateBlackholeCopySemantics` pass，识别 copy pattern 并添加 annotation。不修改 `LowerTileOp` 的核心降级逻辑。

## 10. 正式验收标准

正式阶段完成标准只看：

- TileLang 正式编译产物对外暴露的 host callable
- 通过 `BlackholeModule` 进程内 direct host path 执行
- 由模块内部完成 TT-Metal host materialization / launch / readback
- 与 PyTorch 参考结果一致

以下都不再是正式阶段完成标准：

- `spec.json -> runner`
- external runner 单独执行通过
- 手动按 `"main"` 名称去调用内部符号
