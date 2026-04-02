# TileLang Blackhole 后端最终重构设计

## 基本信息

- **文档ID**: `final_blackhole_backend_redesign`
- **日期**: 2026-03-19（创建），2026-04-02（最近更新）
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

## 2. 当前状态（2026-04-02）

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
- P5 当前已从“零语义层”推进到：program-local semaphore plan、kernel-level semaphore binding、最小 device-side dataflow semaphore builtin、worker producer/consumer direct-runtime E2E，以及 `logical_core_noc_x/y -> KernelSpec.remote_core_descriptors` 最小 remote-core descriptor formalization
- backend cleanup review 与重文件边界拆分草案已建档：`stage4_backend_cleanup_roadmap.md`；cleanup A1/A3 已完成，A2 已落首轮 schema-driven buffer materialization 骨架，B1 已完成四轮 `BlackholeModule` helper 边界拆分，B2 已完成两轮 staged-copy boundary/geometry/index 收敛，B3 已收紧为只消费 explicit `blackhole.cb_requirements`，C1 已收正 compile-time-only accessor codegen 边界，C2 已完成首轮 synchronization host/runtime boundary 收紧
- 前向 Flash-Attention 设计已建档：`stage4_flash_attention_forward_subset.md`
  - 方向已收敛为：以 `mha_fwd_bshd` / `gqa_fwd_bshd` 为牵引，补齐通用 work decomposition / fragment compute region / pipelined staging 三类分析与 lowering 能力，并坚持 “IR 优先、spec 最小化”
  - 当前 `AnalyzeBlackholeWorkDecomposition` / `AnalyzeBlackholeFragmentRegions` / `AnalyzeBlackholePipelineStages` 已接入 Blackhole 主链，`lower()` 的 Blackhole device 入口也已收正为 entry `PrimFunc` 先进入 `SplitBlackholeKernel` 后的 analysis/lowering 管线
  - `reduce_row / row_broadcast / fill / scalar_max / cast_fragment_slice / local_to_cb staging` 等最小 fragment/dataflow 子集已进入真实 lowering 与 codegen 主链，但这批线性 fragment helper 现已被明确标记为过渡实现，不再作为长期协议前提
  - `exp2` 路径大部分已收正为 backend 自有 fast-math helper；但仍有部分 `exp2_row_bcast_affine` 形态未命中 matcher，会回退成原始 `exp2f` call，需继续收敛
  - `AssignBlackholeCores` / direct runtime 的 core-plan 协议已收正到连续 logical worker grid；当前环境按 `11 x 10 = 110` worker cores formalize
  - 当前支持的 MHA/GQA forward compile-path 已打通；runtime 也已越过旧的 fragment-subset fail-fast、`local/accumulator -> shared(CB)` 残留、TRISC math link、core lookup 和 execution hang blocker
  - `cast_fragment_slice -> blackhole.acc` 这类会被后续 matmul 继续消费的 scratch 结果，必须按未来 matmul 所需页数正式 `cb_push_back`；`GenerateMatmulSequence` 也不能再对同一 `blackhole.acc` 输出 CB 重复 `cb_reserve_back`；这些修正已经解掉之前的 deadlock
  - 当前 flash-attn forward 剩余主 blocker 已从 execution hang 收敛为 **compute 语义设计问题**：`blackhole.acc` 不能再同时承担“线性 fragment scratch”和“TT-Metal tile/CB scratch”双重语义
  - 当前正式方向已收敛为 **TT-Metal-first**：
    - `blackhole.acc` 后续只表示 compute-side tile scratch
    - flash-attn compute 主链以 `CB / tile / dst-reg` 流为正式协议
    - 线性 `fragment` helper 仅保留为过渡层，不再作为 `blackhole.acc` 的长期语义
  - 上游 TIR 对接也需同步收正：Blackhole 后段不再从线性 `BufferLoad/BufferStore` 形态猜 tile 语义；凡是后端无法稳定恢复的 tile contract，必须通过 analysis attrs 或显式 builtin 交付
  - 这一轮也明确暴露出更上层的 compiler 问题：现有 TileLang/TIR 对 `stateful / routed / phased / segmented tiled program` 的表达能力不足；下一阶段正式方向已重写为 **多层 compiler-internal IR**：`Stateful Semantic IR -> Spatial Program IR -> TT Target IR`

### TT-Sim 固定验证入口

对当前仓库，Blackhole direct runtime / TT-Sim 的固定验证入口应视为一条明确环境入口，而不是运行时再临时搜索环境变量或直接说“没环境”：

```bash
source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh
export TILELANG_HOME=<当前 checkout 或 worktree>/tilelang_repo
cd <当前 checkout 或 worktree>/tilelang_repo
```

约束：

- `setup_tt_sim.sh` 必须和 `pytest` 在同一个 shell 中执行
- 当前若在 git worktree 中工作，应 source 顶层 checkout 的 `/root/dev/vibe_dsl/scripts/setup_tt_sim.sh`，不要 source worktree 里的同名副本；否则 `TT_METAL_HOME` 会错误指向 worktree 下的 `tt_metal_repo`
- source TT-Sim 脚本后，再把 `TILELANG_HOME` 显式指回当前 checkout/worktree 的 `tilelang_repo`
- 具体跑哪个 runtime `pytest` case/selector，应按当前 task / design / progress 文档决定，不要把某个单独 case 固化成环境入口的一部分
- 进入 TT-Sim / runtime debug 前，应先看 `memory/general_dev.md` 和 `memory/bugs.md` 里的已有经验，尤其是环境、watcher、runtime contract 与已知失败模式；不要每次从头试错

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

说明：

- 上述结构描述的是当前已经落地的正式主链
- 下一阶段不会推翻这条主链，而是在 `PrimFunc / TIR` 与 target-specific plan/lowering 之间插入多层 compiler-internal IR（`Stateful Semantic IR -> Spatial Program IR -> TT Target IR`），用来结束晚期 target-specific 语义猜测

### 3.2 下一阶段多层编译结构：`Stateful Semantic IR -> Spatial Program IR -> TT Target IR`

下一阶段正式方向不是继续把复杂度压进单层 `Stateful Tiled IR`，而是在现有 TileLang/TIR 与 TT-Metal target lowering 之间建立一套多层 compiler-internal IR，把算法语义、空间程序结构、目标硬件 contract 明确拆开：

```text
TileLang DSL / Python examples
  -> PrimFunc / TIR（保留用户写法、通用 loop/tileop/buffer 结构）
  -> Semantic Recovery
  -> Stateful Semantic IR
  -> Semantic Validation
  -> Spatialization
  -> Spatial Program IR
  -> Spatial Validation
  -> Hardware-Aware Mapping
  -> TT Target IR
  -> Target Validation
  -> Codegen / Runtime Materialization
```

这一分层的设计输入来自四类外部经验，但不会被其中任一工作完整绑定：

- `T2S`：算法语义和空间映射应明确分层
- `Dato`：`task / channel / layout` 与 `virtual -> physical mapping` 应是一等概念
- `TL`：hardware representation 和 mapping 是独立编译问题，不是 codegen 细节
- `SPADA`：routing / synchronization correctness 需要独立 validation，而不是 runtime hang 后再倒查

各层职责固定如下：

1. **PrimFunc / TIR 层**
   - 保留用户程序的自然写法
   - 承接 `loop / alloc / copy / gemm / reduce / pipeline` 等通用结构
   - 不直接暴露 TT-Metal 的 `CB / semaphore / runtime args / kernel role`

2. **Stateful Semantic IR 层**
   - 统一表达 `stateful / routed / phased / segmented` 算法语义
   - 显式承接 carry/update、domain constraints、phase live-in/live-out、selection/scatter/combine 语义
   - 到这一层为止，算法语义恢复必须结束
   - 这一层回答的是“程序在算什么”，不回答“怎么在 TT 上跑”

3. **Spatial Program IR 层**
   - 表达空间程序结构：`task / channel / layout / work partition / virtual placement / sync edge / resource intent`
   - 把算法语义转换成 dataflow/spatial machine 上的逻辑程序
   - 这一层回答的是“这些语义如何组织成空间任务和通信图”
   - 这一层仍不出现 TT-specific `CB / semaphore_id / dst offset / runtime arg position`

4. **TT Target IR 层**
   - 在 TT-Metal hardware model 约束下，把 spatial program 映射成正式 target contract
   - 这一层才允许出现 `kernel role / CB / semaphore / compile-time args / runtime args / accessor binding`
   - 这一层必须包含 **dst register layout planning** 和 **carry strategy 选择**（register-resident carry vs CB-round-trip carry）
   - 这一层回答的是“在 TT-Metal 上具体如何合法、稳定、可执行地落成 host/runtime contract”

5. **Codegen / Runtime Materialization 层**
   - 只负责格式化、装配和执行
   - 不再承担结构推理和语义恢复

关键分层决策表如下：

| 关注点 | 应落在哪一层 | 不应落在哪一层 |
|--------|--------------|----------------|
| `carry / combine / phase / routed / segmented / predicate / index_remapping` | `Stateful Semantic IR` | `TT Target IR` / runtime |
| `task graph / channel / layout / work partition / virtual placement / sync edge` | `Spatial Program IR` | `Stateful Semantic IR` / codegen |
| `reader / compute / writer`、`CB`、`semaphore`、`dst layout`、`runtime args`、`core placement` | `TT Target IR` | `Stateful Semantic IR` / `Spatial Program IR` |
| `CreateCircularBuffer / CreateKernel / SetRuntimeArgs` 等 API materialization | `Codegen / Runtime Materialization` | 任一 compiler-internal IR |

等价地说：

- 不允许用 `CB / dst layout / runtime args` 反推 algorithm semantics
- 不允许把 `task / channel / semaphore` 暴露成 Python DSL 的正式一等概念
- 不允许让 codegen/runtime 再次承担结构恢复

### 3.3 设计原则

1. **以后端执行模型约束 lowering，不以后端打印器约束执行模型**
2. **优先复用现有 PrimFunc/TIR 与 host/device 主链，只在少量 target-aware 边界定制**
3. **host/device 划分与 Packed API 参数语义沿用 TileLang / TVM 正式模型**
4. **split 前做语义规划，split 后做正式 plan 提取，host side 做 materialization**
5. **compile-time attrs/schema 与 runtime args 必须严格分层**
6. **逻辑 block/grid 语义应保留在 TIR/pass 中，不在 runtime 侧重建**
7. **multi-core 主要由 host/runtime materialize，不由 codegen 主导**
8. **`SplitBlackholeKernel`、`blackhole_module_direct.cc`、external runner 都不再是主路径设计前提**
9. **如果信息不能从现有 TIR 稳定恢复，就必须在 compiler-internal IR 中一等化；只有在语义仍不唯一时，才允许 Python 侧追加极小 annotation**
10. **用户侧 annotation 只能表达算法语义，不能泄露 TT-Metal/Blackhole 的硬件编程模型**

### 3.4 当前主线路径的三层分工

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

**CB ID 回写协议约束**：

- 任何新增的 blackhole builtin，若包含 cb_id 参数，**必须**在 `GetCBArgPositions` 中注册该参数的 position
- `PlanBlackholeCB` 在回写完成后应做 post-condition 校验：遍历所有 blackhole builtin 的 IntImm 参数，确认不存在残留的 requirement_index 值落在合法 cb_id 范围之外
- 教训：`write_local_slice_to_cb` 因漏注册导致 cb_id 未被回写，runtime 读写了错误的 CB，引发 hang

**CB 分配策略**：

- input CB: 0-7（`kInputCBStart` ~ `kInputCBEnd`）
- compute/output/intermediate CB: 16-31（`kOutputCBStart` ~ `kOutputCBEnd`），kOutput 与 kIntermediate 共享此池
- spill: 32+（仅当 16-31 耗尽时）
- TRISC compute kernel 只能访问 0-31 范围的 CB；fragment scratch CB 必须落在此范围内
- flash-attn forward 当前使用 12 CB，16-slot compute 池仍有余量，但后续更复杂 kernel 需关注池压力

**CB-as-compute-scratch 模式**：

- compute-side scratch CB 分为两类：
  - transport CB：reader push → compute pop → compute push → writer pop 的 FIFO 协议
  - compute scratch CB：TRISC compute 长期持有的 tile scratch / scalar scratch
- `blackhole.acc` 后续只允许表示 compute scratch 中的 **tile scratch**
- scalar state（如 flash-attn 的 `scores_max` / `logsum` / `scores_sum`）不应再和 tile scratch 复用同一长期语义
- 当 `PlanBlackholeCB` 升级到真正的 memory planner 时，必须区分 transport CB、tile scratch CB、scalar scratch 三类资源的 lifetime 和 reuse 策略

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

### 5.3 下一阶段 compiler-internal IR 体系：`Stateful Semantic IR / Spatial Program IR / TT Target IR`

下一阶段不再把 compiler-internal 抽象收敛成单层 `Stateful Tiled IR`，而是正式采用三层 IR：

1. **`Stateful Semantic IR`**
   - 只表达算法语义真相，不表达空间程序结构和目标硬件协议
   - 核心对象：
     - `Domain`
     - `State`
     - `Relation`
     - `Phase`
     - `SemanticRegion`
   - 其中：
     - `Domain` 表达 `dense / segmented / routed / paged` 与 `bound_expr / predicate / index_remapping`
     - `State` 表达 `matrix_state / vector_state / scalar_state / index_state` 与 `ephemeral / carry / cross_phase`
     - `Relation` 表达 `reduced_from / applies_to / indexes / scatters_to / carried_across`
     - `Phase` 只表达 **algorithm phase**
     - `SemanticRegion` 表达一段单一语义责任的计算区域（如 `matmul / reduce / normalize / scatter / recurrence`）
   - 硬约束：
     - 每个可变对象都必须有明确 `state kind`
     - 每个 carry/update/merge 都必须显式，不允许继续靠普通 store/load 隐含语义
     - 每个 gather/scatter/routed access 都必须显式说明 index 来源或 combine 规则
     - 任何 TT-Metal 特有对象（`CB / semaphore / kernel role / runtime arg`）都不进入这一层

2. **`Spatial Program IR`**
   - 只表达空间程序结构，不表达 TT-Metal 具体资源与 ABI
   - 核心对象：
     - `Task`
     - `Channel`
     - `Layout`
     - `WorkPartition`
     - `Placement`
     - `SyncEdge`
     - `ResourceIntent`
   - 这一层回答的是：
     - semantic regions 如何组织成逻辑 tasks
     - task 间有哪些 channel / sync / work partition / virtual placement 约束
     - 哪些 state 需要 transport、scratch、persistent carry、reduction carrier
   - 硬约束：
     - 允许 `virtual placement`
     - 不允许 `CBIndex / semaphore_id / dst offset / runtime arg position`
     - 不允许 host/runtime API 名字进入这一层

3. **`TT Target IR`**
   - 只表达 TT-Metal/Blackhole 可执行 contract
   - 核心对象：
     - `TTKernel`
     - `TTCoreGroup`
     - `TTCBPlan`
     - `TTSemaphorePlan`
     - `TTDstLayoutPlan`
     - `TTABIPlan`
     - `TTExecutionPlan`
   - 这一层才允许出现：
     - `reader / compute / writer`
     - `CB / semaphore / remote-core descriptor`
     - `compile-time args / runtime args / accessor bindings`
     - `dst register layout`
   - `ExecutableSpec` 是从这一层**物化**出来的，而不是再由散落 attrs 拼装

三层之间的 source-of-truth 规则固定如下：

- 算法语义的 source-of-truth 在 `Stateful Semantic IR`
- 空间组织的 source-of-truth 在 `Spatial Program IR`
- TT 资源与 ABI 的 source-of-truth 在 `TT Target IR`
- 下层不得回推上层；validation 必须分层做，而不是只在最终 codegen 前做一次

这三层不是为了“把一件事拆成三份文档”，而是因为它们回答的是三类本质不同的问题：

| 层级 | 它回答的问题 | 为什么不能放到别层解决 | 这一层冻结的真相 |
|------|--------------|------------------------|------------------|
| `Stateful Semantic IR` | 程序到底在算什么，哪些值在 carry，哪些 domain/phase/relation 是算法定义的一部分 | 如果语义继续留在 TIR matcher、target lowering 或 runtime 才恢复，就会重复出现 `blackhole.acc` 这类混合语义 | 算法语义真相 |
| `Spatial Program IR` | 这些语义如何组织成 task/channel/layout/sync/work partition | 如果直接从 semantic 层跳到 TT 资源层，`task/channel/layout/sync` 会被压成 target 细节，最终又回到单层黑洞 pass | 空间程序真相 |
| `TT Target IR` | 在 TT-Metal 上具体要用哪些 kernel role、CB、semaphore、dst layout、ABI、placement 才能合法执行 | 如果把这些内容提前到 semantic/spatial 层，会把 TT 编程模型污染到本应通用的内部抽象；如果放到 codegen/runtime 才决定，又会重新引入晚期猜协议 | TT 可执行 contract 真相 |

#### 5.3.1 `Stateful Semantic IR`

**设计目标**

- 结束当前 “先压碎到 TIR，再在 `LowerBlackholeOps`/runtime 附近猜语义” 的模式
- 把 carry/update/combine/domain constraint/phase ordering 变成 compiler 内部的正式真源
- 让 flash-attn、online-softmax、Welford、routed/paged/select 这类 workload 可以共享一套算法语义对象，而不是继续堆 workload-specific matcher

**为什么必须单独存在**

- `State / Relation / Phase / Domain` 是算法定义的一部分，不是执行策略
- 如果这一层不存在，下游只能从 `BufferLoad/BufferStore`、fragment helper 名字、或 target builtins 反推语义
- 这正是当前 `blackhole.acc` 混合了 “线性 helper 语义” 与 “TT tile scratch 语义” 的根因

**输入**

- `PrimFunc / TIR`
- `AnalyzeBlackholeWorkDecomposition`
- `AnalyzeBlackholeFragmentRegions`
- `AnalyzeBlackholePipelineStages`
- 必要时极小 Python annotation（仅表达 `carry / routing / selection / combine` 语义）

**输出**

- `SemanticProgram`
  - `Domain`
  - `State`
  - `Relation`
  - `Phase`
  - `SemanticRegion`

**这一层必须冻结的事实**

- 哪些对象是 `matrix/vector/scalar/index state`
- 哪些 state 是 `ephemeral / carry / cross_phase`
- 哪些 domain 带 `bound_expr / predicate / index_remapping`
- 哪些 `combine_function / reduced_from / carried_across` 关系存在
- 哪些 `algorithm phase` 与 `live_in/live_out` 是正式语义

**这一层明确不做的事**

- 不拆 `reader / compute / writer`
- 不建 `task/channel`
- 不选 `register-resident carry` 还是 `CB-round-trip carry`
- 不分配 `CB / semaphore / dst offset / core`
- 不定义 compile-time/runtime ABI

**这一层的验证职责**

- `state kind`、`lifetime`、`shape` 一致性
- `carried_across` 与 `live_in/live_out` 一致性
- `bound_expr / predicate / index_remapping` 完整性
- `combine_function` 存在性与绑定对象完整性
- 禁止同一对象同时承担算法 state 与 target scratch 双重语义

**交给下一层的 contract**

`Spatialization` 只能消费已经冻结的算法事实，不允许再：

- 通过 buffer 名、builtin 名、kernel 名猜 state kind
- 通过 target resource 反推 phase/relation
- 补写缺失的 combine/routing 语义

#### 5.3.2 `Spatial Program IR`

**设计目标**

- 把算法语义翻译成“可在 spatial/dataflow machine 上组织”的逻辑程序
- 显式表达 `task / channel / layout / work partition / sync / placement`
- 让 mapping、routing、synchronization 成为正式编译问题，而不是 TT codegen 附带行为

**为什么必须单独存在**

- `task/channel/layout/sync` 不是算法语义本身，但也远高于 TT 具体资源
- 如果跳过这一层，semantic 层会被迫长出 execution topology；或者 target 层会变成 “一边理解算法、一边做 TT 资源规划” 的黑洞
- `Dato/TL/SPADA` 给出的可借鉴点，本质上都落在这层，而不是 semantic 层

**输入**

- `SemanticProgram`
- target-neutral spatialization policy
  - task fusion/splitting policy
  - work partition policy
  - layout selection policy
  - sync construction policy

**输出**

- `SpatialProgram`
  - `Task`
  - `Channel`
  - `Layout`
  - `WorkPartition`
  - `Placement`
  - `SyncEdge`
  - `ResourceIntent`

**这一层必须冻结的事实**

- 哪些 `SemanticRegion` 被组织成哪些 logical task
- task 间需要哪些 channel，payload/ordering/transport 语义是什么
- 数据按什么 `layout / shard / group / route / page` 组织
- 工作如何 partition 到 task 实例
- 哪些同步关系是正式需要的
- 哪些资源需求属于 `transport / scratch / persistent carry / reduction carrier / output`

**这一层明确不做的事**

- 不引入 `CBIndex / semaphore_id / dst offset`
- 不决定 `CreateCircularBuffer` / `SetRuntimeArgs`
- 不绑定 TT physical cores
- 不把 logical task 直接等同于最终某个 TT kernel source 文件

**这一层的验证职责**

- task/channel 图是否闭合
- producer-consumer / barrier / multicast-ready 同步是否足够
- work partition 与 layout 是否一致
- carried state 是否在正确 task 边界上传递
- virtual placement 约束是否自洽
- 是否存在明显 race/deadlock/routing inconsistency 风险

**交给下一层的 contract**

`Hardware-Aware Mapping` 只能在既有 `Task / Channel / Layout / Sync / ResourceIntent` 基础上落 TT 资源，不允许再：

- 反向改变算法 phase/relation
- 省略 channel/sync 后靠 TT runtime 隐式补协议
- 把 work partition 重新发明成另一套隐藏 host 逻辑

#### 5.3.3 `TT Target IR`

**设计目标**

- 把 TT-Metal 原生编程模型吸收成正式 compiler target contract
- 让 `CB / semaphore / dst layout / kernel role / ABI / execution plan` 成为一等对象
- 让 codegen/runtime 只做 materialization，不再承担编程模型设计

**为什么必须单独存在**

- TT-Metal 的核心复杂度不只是 emit builtin，而是显式程序结构
- `reader / compute / writer`、`CreateCircularBuffer`、`SetRuntimeArgs`、per-core launch schema 都是 program contract
- 如果没有这一层，`LowerBlackholeOps`、`PlanBlackholeCB`、`AssignBlackholeCores`、`rt_mod_blackhole` 会继续散落地共同决定协议

**输入**

- `SpatialProgram`
- TT hardware model
  - topology
  - memory hierarchy
  - NoC / multicast / semaphore capabilities
  - dst/register capacity
  - core kinds and placement constraints

**输出**

- `TTProgram`
  - `TTKernel`
  - `TTCoreGroup`
  - `TTCBPlan`
  - `TTSemaphorePlan`
  - `TTDstLayoutPlan`
  - `TTABIPlan`
  - `TTExecutionPlan`

**这一层必须冻结的事实**

- 哪些 logical task 对应哪些 `reader / compute / writer / relay / reduction` 角色
- virtual placement 如何映射到 TT physical core groups
- 哪些 `ResourceIntent` 落成哪些 CB class / capacity / producer-consumer binding
- 同步关系如何落成 semaphore / multicast / barrier protocol
- 哪些 state 驻留在 dst，offset 与 tile span 如何规划
- compile-time/runtime/accessor/launch ABI 如何稳定编码

**这一层明确不做的事**

- 不再修改 semantic state/relation/phase
- 不再修改 spatial task/channel/layout/sync 结构
- 不把 TT API 调用细节混成语义对象

**这一层的验证职责**

- L1 / CB / dst 容量约束
- semaphore / multicast / routing 合法性
- core placement 合法性
- compile-time/runtime ABI 完整性
- `ExecutableSpec` 所需信息是否齐全且无二义

**交给下一层的 contract**

`Codegen / Runtime Materialization` 只能消费冻结后的 `TTProgram`，不允许再：

- 因为 codegen 方便而改 kernel role / CB binding / runtime arg schema
- 通过 host 侧猜测 remote core、segment layout、dst 角色
- 重新决定 carry strategy 或 semaphore protocol

#### 5.3.4 层间对接协议

三层 IR 的 handoff 必须满足下面的接口约束：

| From | To | 必须提供的输入 contract | 这一跳允许做的决定 | 这一跳禁止做的事 |
|------|----|-------------------------|--------------------|------------------|
| `Semantic Recovery` | `Stateful Semantic IR` | 已恢复的 domain/state/relation/phase 事实 | 语义对象化、真源冻结 | 偷渡 TT 资源信息 |
| `Stateful Semantic IR` | `Spatial Program IR` | 冻结后的算法语义 | task/channel/layout/sync/work partition 构造 | 反向修改语义定义 |
| `Spatial Program IR` | `TT Target IR` | 冻结后的空间程序结构 | TT mapping、resource planning、ABI 定义 | 反向发明新 task graph 或补算法 combine |
| `TT Target IR` | `ExecutableSpec / runtime` | 冻结后的 target contract | API materialization、kernel/codegen emission | 回推语义、隐式补协议 |

从实现角度看，这对应新的 pass 边界：

- `LiftToStatefulSemanticIR`
- `ValidateStatefulSemanticIR`
- `LowerToSpatialProgram`
- `ValidateSpatialProgram`
- `LowerSpatialProgramToTTTarget`
- `ValidateTTTargetProgram`
- `MaterializeTTExecutableSpec`

#### 5.3.5 以 flash-attn forward 为例的端到端对接

为了避免三层 IR 停留在抽象口号，下面用当前最关键的 flash-attn forward 子集说明每层到底接什么、产什么。

**A. 输入到 `Semantic Recovery` 的事实**

当前从 TIR 与 `AnalyzeBlackhole*` 至少可以恢复出以下事实：

- 存在以 `Q x K^T` 为核心的 tiled matmul
- 存在按 row 的 `max / sum / rescale / accumulate` 递推
- `acc_o`、`scores_max`、`scores_sum/logsum` 是 loop-carried state
- causal/GQA 变体会给 domain 增加 `bound_expr` 或 `index_remapping`
- block-sparse/paged/routed 变体会给 domain 增加 `predicate` 或 `index` 来源

**B. `Stateful Semantic IR` 应该长成什么**

在 semantic 层，flash-attn 不是 “reader/compute/writer 三段 kernel”，而是一个带 recurrence 的 attention 算法：

- `Domain`
  - `q_tile_domain`
  - `kv_chunk_domain`
  - 可选 `causal_bound_expr`
  - 可选 `gqa_index_remapping`
- `State`
  - `acc_o : matrix_state(carry)`
  - `scores_max : vector_state(carry)`
  - `scores_sum : vector_state(carry)`
  - `logsum : scalar_state(cross_phase)` 或作为 epilogue 派生结果
- `Relation`
  - `scores_max reduced_from qk_scores`
  - `scores_sum reduced_from exp_scores`
  - `acc_o carried_across kv_chunk_phase`
  - `scores_max carried_across kv_chunk_phase`
  - `scores_sum carried_across kv_chunk_phase`
- `Phase`
  - `kv_chunk_phase`
  - `epilogue_phase`
- `SemanticRegion`
  - `qk_matmul`
  - `row_max_reduce`
  - `row_sum_reduce`
  - `rescale_update`
  - `ov_matmul`
  - `epilogue_writeback`

这一层的关键是：它已经完整回答了“attention 算法在算什么”，但还没有回答“这些区域如何拆成 task、怎么同步、怎么落到 TT kernel role”。

**C. `Spatial Program IR` 应该长成什么**

在 spatial 层，flash-attn 变成逻辑任务图，而不是 TT 资源图：

- `Task`
  - `load_q_task`
  - `stream_k_task`
  - `stream_v_task`
  - `attention_step_task`
  - `store_out_task`
- `Channel`
  - `q_tiles`
  - `k_tiles`
  - `v_tiles`
  - `carry_state`
  - `out_tiles`
- `Layout`
  - `q_row_layout`
  - `kv_chunk_layout`
  - 可选 `grouped_layout` / `paged_layout`
- `WorkPartition`
  - `row_partition`
  - causal 时可变成 `causal_pair_partition`
- `SyncEdge`
  - `load_q_task -> attention_step_task`
  - `stream_k_task -> attention_step_task`
  - `stream_v_task -> attention_step_task`
  - `attention_step_task -> store_out_task`
- `ResourceIntent`
  - `transport_buffer(q/k/v)`
  - `persistent_carry(acc_o/max/sum)`
  - `output`

这一层的关键是：已经正式定义了 task graph、channel graph、work partition、sync requirements；但仍然没有决定 CB 编号、dst offset、semaphore id、runtime arg slot。

**D. `TT Target IR` 应该长成什么**

在 TT target 层，flash-attn 才正式落成 TT-Metal 可执行 contract：

- `TTKernel`
  - `reader_qkv`
  - `compute_attention`
  - `writer_out`
- `TTCoreGroup`
  - `reader_cores`
  - `compute_cores`
  - `writer_cores`
- `TTCBPlan`
  - transport CB：`q/k/v`
  - temporary CB：`qk_result / exp_diff`
  - state CB：需要时承接 `CB-round-trip carry`
  - output CB
- `TTDstLayoutPlan`
  - `acc_o` offset
  - `scores_max` offset
  - `scores_sum` offset
  - `qk_temp` offset
- `TTSemaphorePlan`
  - reader -> compute ready
  - compute -> writer completion
  - 多核变体下的 multicast / barrier protocol
- `TTABIPlan`
  - compile-time tile shape / transpose / pack config
  - runtime args：tile count、logical row id、kv chunk bounds、buffer base address
- `TTExecutionPlan`
  - virtual row partition 到物理 core group 的映射
  - per-core work packet

这一层的关键是：到这里为止，host/runtime 已经不需要再懂 attention recurrence，也不应该再猜 carry state 该放 dst 还是 CB。

**E. 这条例子体现的分层纪律**

- semantic 层如果没有冻结 `acc_o / scores_max / scores_sum` 的 carry 语义，下游不允许靠 `dst` 或 CB 名字补回来
- spatial 层如果没有冻结 `task/channel/sync/work partition`，target 层不允许靠 TT kernel 名字现场发明 task graph
- target 层如果没有冻结 `CB / semaphore / dst layout / ABI`，runtime 不允许靠 host 侧 heuristics 补协议

#### 5.3.6 新架构与当前 pass/模块的映射关系

当前代码不会被一次性推倒；需要把已有 pass 和模块逐步归位到新架构。映射关系如下：

| 当前 pass / 模块 | 在新架构中的正式归属 | 长期命运 | 备注 |
|------------------|----------------------|----------|------|
| `AnalyzeBlackholeWorkDecomposition` | `Semantic Recovery` 输入生产者 | 保留并泛化 | 主要产 `Domain / bound_expr / index_remapping` 相关事实 |
| `AnalyzeBlackholeFragmentRegions` | `Semantic Recovery` 输入生产者 | 保留并泛化 | 主要产 `State / Relation / SemanticRegion` 相关事实 |
| `AnalyzeBlackholePipelineStages` | `Semantic Recovery` 输入生产者 | 保留并收紧语义 | 主要帮助恢复 `algorithm phase` 与 pipeline 残余信息 |
| `LowerBlackholeOps` | 当前混合层；未来应拆到 `LowerToSpatialProgram` + `LowerSpatialProgramToTTTarget` | 最终被拆薄或消失 | 现在同时承担语义理解、TT lowering、ABI/schema 提取，正是要被消解的黑洞 |
| `PlanBlackholeCB` | `TT Target IR` planner 子模块 | 保留但降位 | 未来只做 `ResourceIntent -> TTCBPlan`，不再凭 `cb_requirements` 决定世界 |
| `AssignBlackholeCores` | `TT Target IR` planner 子模块 | 保留但改职责 | 未来只做 `virtual placement -> TTCoreGroup / TTExecutionPlan` |
| `rt_mod_blackhole` | `Codegen / Runtime Materialization` | 保留并收边界 | 未来只消费冻结后的 `TTProgram/ExecutableSpec`，不再提取或补协议 |
| `ExecutableSpec` | `TT Target IR` 的物化结果 | 保留 | 不再是散落 attrs 的二次拼装结果 |

对应到新的 pass 链，目标顺序应为：

```text
SplitBlackholeKernel
  -> AnalyzeBlackholeWorkDecomposition
  -> AnalyzeBlackholeFragmentRegions
  -> AnalyzeBlackholePipelineStages
  -> LiftToStatefulSemanticIR
  -> ValidateStatefulSemanticIR
  -> LowerToSpatialProgram
  -> ValidateSpatialProgram
  -> LowerSpatialProgramToTTTarget
  -> ValidateTTTargetProgram
  -> MaterializeTTExecutableSpec
  -> rt_mod_blackhole
```

**过渡策略也必须明确**：

1. **Phase A**
   - 插入 `LiftToStatefulSemanticIR / ValidateStatefulSemanticIR`
   - 保留 `LowerBlackholeOps -> PlanBlackholeCB -> AssignBlackholeCores -> rt_mod_blackhole`
   - 但要求 `LowerBlackholeOps` 优先消费 semantic 真源，而不是继续从普通 loop 猜主语义

2. **Phase B**
   - 新增 `LowerToSpatialProgram / ValidateSpatialProgram`
   - 开始把 `task/channel/layout/sync/work partition` 从 `LowerBlackholeOps` 中抽出
   - `LowerBlackholeOps` 在这一阶段缩成 TT-target helper，而不再是总 lowering 黑洞

3. **Phase C**
   - 用 `LowerSpatialProgramToTTTarget / ValidateTTTargetProgram / MaterializeTTExecutableSpec` 取代当前散装 target 协议生成
   - `PlanBlackholeCB` 与 `AssignBlackholeCores` 降为 target planner 子模块
   - `rt_mod_blackhole` 只做 materialization，不再提取 schema 或兜底解释

### 5.4 现有 DSL/TIR 与多层 IR 的对接边界

默认路径是：

`现有 TileLang DSL -> PrimFunc/TIR -> Semantic Recovery -> Stateful Semantic IR -> Spatialization -> Spatial Program IR -> Hardware-Aware Mapping -> TT Target IR`

也就是说，下一阶段默认仍以**兼容现有 Python DSL 写法**为第一优先级；但兼容不是无条件的，边界如下：

| 语义类别 | 自动 lift 为主 | 需要极小 annotation 消歧 |
|---------|----------------|-------------------------|
| dense tile compute + 常规 reduce/map | 是 | 否 |
| `tile -> row/vector reduce -> apply-back-to-tile` 稳定模式 | 是 | 否 |
| 单层或双层 loop-carried state | 大多可自动恢复 | 少数 state kind 不唯一时需要 |
| data-dependent loop bound（causal attention） | 大多可从 IR 恢复 bound_expr | 当 bound 依赖多个 kernel 参数时可能需要 |
| grouped index remapping（GQA `by // groups`） | 可自动恢复 | 否（index 算术可识别） |
| block-level sparsity predicate（block sparse attn） | 不能完全自动 | 需要标记 `predicate tensor` |
| 显式 pipeline/barrier/ws 带出的 phase 边界 | 是 | 否 |
| routed/paged access 且 index/page-table tensor 已显式进入 kernel | 大多可自动恢复 | 当 index 角色不唯一时需要 |
| MoE routing / segmented domain / scatter_reduce | 不能完全依赖自动恢复 | 需要 |
| topk / argmax / selection state machine | 不能完全依赖自动恢复 | 需要 |
| combine 规则（多变量耦合如 Welford / online-softmax rescale） | 不能稳定猜测 | 需要标记 combine function |

annotation policy 明确如下：

- 只接受 **annotation/helper 风格**
- annotation 只表达算法语义，例如：
  - 这是 `carry state`
  - 这是 `routing index`
  - 这是 `selection state`
  - 这是 `scatter_reduce(sum/max)` 的 combine 规则
- annotation **不得**暴露硬件语义，例如：
  - `CB`
  - `semaphore`
  - `reader/compute/writer kernel`
  - `runtime_args`

这条边界的目标是：

- 内部 IR 可以大幅增强，但不把复杂度转嫁给用户
- 不让用户学习 TT-Metal 的编程模型
- 不把 `task/channel/semaphore` 变成用户侧 DSL 的一等概念

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

### 8.3 中期架构演进方向：多层 compiler-internal IR

与其继续在 target-specific 后段做“先压碎再恢复”，下一阶段正式方向是引入 **多层 compiler-internal IR**，把算法理解、空间程序结构和目标硬件映射硬切开。

统一性要求如下：

| workload | Semantic IR 的核心组合 | Spatial Program IR 的核心组合 | TT Target IR 的核心组合 |
|---------|-------------------------|-------------------------------|-------------------------|
| FlashAttention / online softmax | `matrix_state(carry) + vector_state(carry) + scalar_state(carry) + reduced_from + carried_across` | `compute/load/store task + channel + sync edge + row/pair work partition` | `reader/compute/writer + CB plan + dst layout + runtime args` |
| Causal Flash-Attn | 同上 + `bound_expr = f(bx)` | `balanced pair partition` 或 `causal row partition` | `per-core runtime args + carry strategy + semaphore plan` |
| GQA | 同上 + `group index remapping` | `grouped layout + grouped work partition` | `core-group placement + accessor/runtime binding` |
| Block Sparse Attention | 同上 + `predicate(block_mask)` | `predicated channel / sparse partition` | `transport CB + sync protocol + ABI` |
| Welford LayerNorm / RMSNorm | `vector_state(carry) × 2-3 + coupled combine function` | `reduction task + persistent carrier` | `dst scratch / CB scratch / reduction ABI` |
| Linear Attention / Mamba | `matrix_state/vector_state(carry) + recurrence` | `phase-ordered tasks + carried channel` | `carry strategy + placement + execution plan` |
| Split-K GEMM | `matrix_state(carry) + cross-core combine function` | `split-k partition + combine channel` | `multicast / reduction semaphore / CB plan` |
| MoE / routed grouped GEMM | `routed + segmented + index_state + scatters_to` | `routed task graph + segmented layout + sync edge` | `core-group placement + remote-core descriptors + runtime schema` |
| topk / argmax / selection | `index_state(carry) + reduced_from + compact/select rewrite` | `selection/rewrite task + channel` | `target reduction ABI + scratch plan` |
| paged decode / sparse MLA | `paged + routed + index_state` | `paged layout + routing channel` | `page-table ABI + placement + transport plan` |

也就是说：

- attention 不应要求专用 `flash_attention` IR
- MoE 不应要求专用 `moe` IR
- topk 不应要求专用 `topk` IR
- paged decode 不应要求专用 `paged_attention` IR

真正的一致性来自：

- 语义层的 `Domain / State / Relation / Phase / SemanticRegion`
- 空间程序层的 `Task / Channel / Layout / WorkPartition / Placement / SyncEdge / ResourceIntent`
- 目标层的 `TTKernel / TTCBPlan / TTSemaphorePlan / TTDstLayoutPlan / TTABIPlan / TTExecutionPlan`

### 8.4 对 Python DSL 兼容性的正式要求

下一阶段设计明确采用：

- **内部 IR 大幅增强**
- **Python DSL 只允许极小变化**
- **任何新增 Python 侧能力都必须保持算法语义视角，不泄露硬件编程模型**

正式边界：

1. 现有复杂 examples 应尽量保持主体写法不变
2. 真正需要补充的 Python 侧信息，以 annotation/helper 方式出现，而不是要求用户改写 kernel 主体结构
3. annotation 的心智模型应保持在“carry / routing / selection / combine 规则”层，而不是“task / channel / CB / semaphore / runtime args”层

### 8.5 对当前规划的影响与 rollout

- **Stage 0-3 已完成的基础设施不推翻**：
  - `ExecutableSpec`
  - `BlackholeModule` direct host path
  - copy / GEMM / multi-core 主链
- **当前 flash-attn compile/runtime bring-up 仍有价值**：
  - 它已经把问题收敛到真实语义边界，而不是把问题藏在 build/codegen 噪声里
- **当前 `2026-04-02-stateful-tiled-ir-phase1-implementation-plan.md` 的定位需要降位**：
  - 它不再代表下一阶段总体架构
  - 它只保留为新总设计下 **Phase A（Semantic IR）** 的历史草案和子计划参考
- **下一阶段实施顺序**：
  - **Phase A: Stateful Semantic IR**
    1. 新增 `Domain / State / Relation / Phase / SemanticRegion`
    2. 落 `LiftToStatefulSemanticIR` 与 `ValidateStatefulSemanticIR`
    3. 验收标准：semantic lift 能跑、semantic validate 能拦、现有 GEMM/copy compile-path 不回归
  - **Phase B: Spatial Program IR**
    1. 新增 `Task / Channel / Layout / WorkPartition / Placement / SyncEdge / ResourceIntent`
    2. 把 flash-attn / online-softmax / GEMM 的 task/channel/layout/sync 一等化
    3. 验收标准：不再由 `LowerBlackholeOps` 直接同时承担语义理解和 TT lowering
  - **Phase C: TT Target IR**
    1. 把 `CB / semaphore / dst layout / kernel role / ABI / execution plan` 统一到 TT target contract
    2. `ExecutableSpec` 改为从 `TT Target IR` 物化
    3. 验收标准：flash-attn correctness 通过；runtime/codegen 不再回推上层语义
- **验证分层**：
  - semantic 层：IR snapshot / semantic validation
  - spatial 层：task/channel/layout/sync 结构验证
  - target 层：CB / dst layout / semaphore / runtime-arg / accessor binding 结构验证
  - runtime 层：TT-Sim correctness / 对拍

### 8.6 Recovery / Spatialization 的时机与长期迁移路径

当前 `AnalyzeBlackhole*` 仍位于 Blackhole pipeline 的**后段**——`SplitBlackholeKernel` 之后：

```text
SplitBlackholeKernel
  -> AnalyzeBlackholeWorkDecomposition
  -> AnalyzeBlackholeFragmentRegions
  -> AnalyzeBlackholePipelineStages
  -> Semantic Recovery / Lift
```

在到达这个位置之前，以下关键信息已丢失：

| 信息 | 丢失位置 | 到达 Recovery 时还在？ |
|------|---------|-------------------------|
| `T.Pipelined` 的 stage/order/group | `inject_pipeline` | 否 |
| `T.gemm` 的 M/N/K/transpose | `LowerTileOp`（Blackhole: 保留为 `gemm_py`） | 是 |
| `T.copy` 的 tile 语义 | `LowerTileOp` | 否——已变成标量 loop |
| `T.reduce` 的 reduction type/dim | `LowerTileOp` | 否——已变成标量 loop |
| `T.alloc_fragment` 的 scope | `LowerTileOp`（fragment → local） | 否 |
| fragment 上的 element-wise ops | `LowerTileOp` | 否——已变成标量 BufferStore/Load |
| loop-carried state（如 `scores_max *= scale`） | 从未丢失，但变成普通标量赋值 | 部分——需要从标量 pattern match |

阶段性路径如下：

- **路径 A（当前现实）**：接受 Recovery 在后段，依赖 `AnalyzeBlackhole*` 从压碎后的 TIR 恢复算法语义；`LiftToStatefulSemanticIR` 只做格式化与真源冻结。
  - 优点：不改现有 lowering pipeline 前半段
  - 缺点：Analyze pass 的 matcher 泛化会成为瓶颈

- **路径 B（长期目标）**：把 Semantic Recovery 前移到 `inject_pipeline` 之后、`LowerTileOp` 之前；把 Spatialization 放在 semantic lift 之后、target mapping 之前。
  - 优点：semantic 层不再从标量 TIR 猜语义
  - 缺点：需要重构 pipeline 前半段，且更早介入 split 前结构

**迁移条件**：当 Phase B 开始覆盖 Welford / routed / paged / selection 等更宽语义时，如果 `AnalyzeBlackhole*` matcher 数量快速膨胀，应启动 Recovery 前移。

### 8.7 TT-Metal SDPA Ground Truth 与 `TT Target IR` 设计约束

基于 TT-Metal 原生 SDPA 实现的审查结论，TT target lowering 必须处理以下设计约束。当前 ground truth 以这三个具体 reference 文件为准：

- `tt_metal_repo/tt-train/sources/ttml/metal/ops/sdpa_fw/device/kernels/compute/sdpa_fw_compute_kernel.cpp`
- `tt_metal_repo/models/demos/deepseek_v3_b1/kernel_includes/tt_metal/include/compute_kernel_api/sdpa.h`
- `tt_metal_repo/tt-train/sources/ttml/metal/ops/sdpa_fw/device/sdpa_fw_program_factory.cpp`

#### 8.7.1 TT 程序模型本质上是 program-level structure

TT-Metal 原生程序天然由 `reader / compute / writer` 多 kernel、host-side `CreateCircularBuffer / CreateKernel / SetRuntimeArgs` materialization、以及 per-core runtime args 组成。这意味着：

- `reader / compute / writer` 应进入 `TT Target IR`
- `CreateCircularBuffer / CreateKernel / ComputeConfig` 是 materialization，不应进入 semantic/spatial 层
- per-core runtime args / core-group placement 是 program-level contract，不是 codegen 小细节

#### 8.7.2 dst 寄存器是跨 K-loop 长期持有的

TT-Metal SDPA 的 `tile_regs_acquire/release` 包裹**整个** chunk loop，不是每次迭代：

```text
tile_regs_acquire()              // 整个 chunk loop 之前
for chunk in 0..num_chunks:
    compute_sdpa_chunk(...)      // 内部不 acquire/release
tile_regs_commit/wait/release()  // 整个 chunk loop 之后
```

这意味着 `acc_o / scores_max / logsum` 全部 live 在 dst 寄存器里原地更新，不做 CB round-trip。

#### 8.7.3 dst 寄存器布局是编译时静态规划的

TT-Metal SDPA 的 dst 空间静态分区：

```text
[0, mm2_dst_offset + num_tiles_v * packed_tile_size)  = output accumulator
[max_dst_offset, max_dst_offset + 2)                  = running max (col 0)
[sum_dst_offset, ...]                                 = running sum (col 32+)
[corr_exp_dst_offset, ...]                            = correction factor
[mm1_dst_offset, ...]                                 = QK temporary
```

因此 `TTDstLayoutPlan` 必须是 `TT Target IR` 的一等对象，而不是隐藏在 compute codegen helper 里。

#### 8.7.4 CB / semaphore 是目标资源，不是普通 buffer lowering 副产品

SDPA 使用多类 CB：

- **Transport CB**（输入 Q/K/V/mask）
- **Ping-pong state CB**（prev/cur max、sum、mm_out）
- **Temporary CB**（qk_result、exp_diff）
- **Output CB**

同时，TT-Metal 文档和多核 matmul/multicast 示例明确表明：

- `cb_reserve_back / cb_push_back / cb_wait_front / cb_pop_front` 反映的是显式 channel protocol
- `noc_async_write_multicast / noc_semaphore_set_multicast` 与 semaphore handshake 反映的是显式 sync/routing protocol

因此：

- `TTCBPlan` 和 `TTSemaphorePlan` 必须是 `TT Target IR` 的一等对象
- `PlanBlackholeCB` 未来应降为 target planner 子模块，而不是“凭 cb_requirements 决定世界”的独立主层

#### 8.7.5 Carry strategy 是 target lowering 决策

同一个 `carry` state，在不同实现里可以有不同 target strategy：

- `register-resident carry`
- `CB-round-trip carry`

这一选择不属于 semantic 层，必须在 `TT Target IR` 的 target mapping 阶段依据 `dst` 容量、CB 压力、task/channel 结构做出。

#### 8.7.6 MATH/PACK/UNPACK 三线程并行只进入 target 层

TT-Metal 的 fused matmul 中，某些 elementwise/SFPU 工作会在 PACK thread 上执行。这是 Tensix 核心的硬件事实，应作为 `TT Target IR` lowering 与 codegen 的调度约束，而不应污染 semantic/spatial 层对象。

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
