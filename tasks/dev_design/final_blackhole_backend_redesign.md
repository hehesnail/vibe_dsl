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
  - 这一轮也明确暴露出更上层的 compiler 问题：现有 TileLang/TIR 对 `stateful / routed / phased / segmented tiled program` 的表达能力不足；下一阶段正式方向是在保持 Python DSL 主体写法稳定的前提下，引入 compiler-internal `Stateful Tiled IR`

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
- 下一阶段不会推翻这条主链，而是在 `PrimFunc / TIR` 与 target-specific plan/lowering 之间插入一层新的 compiler-internal semantic IR，用来结束晚期 target-specific 语义猜测

### 3.2 下一阶段四层编译结构：`Stateful Tiled IR`

下一阶段正式方向不是继续在 Blackhole 后段堆 matcher，而是在现有 TileLang/TIR 与 target program lowering 之间引入一层新的内部语义 IR：

```text
TileLang DSL / Python examples
  -> PrimFunc / TIR（保留用户写法、通用 loop/tileop/buffer 结构）
  -> Semantic Recovery / Lift
  -> Stateful Tiled IR（算法语义层）
  -> Target Program Lowering（TT-Metal / CUDA / 其他后端）
  -> Codegen / Runtime Materialization
```

四层职责固定如下：

1. **PrimFunc / TIR 层**
   - 保留用户程序的自然写法
   - 承接 `loop / alloc / copy / gemm / reduce / pipeline` 等通用结构
   - 不直接暴露 TT-Metal 的 `CB / semaphore / runtime args / kernel role`

2. **Stateful Tiled IR 层**
   - 统一表达 `stateful / routed / phased / segmented tiled program`
   - 显式承接 carry/update、tile-state 与 row-state 绑定、routed/paged domain、phase live-in/live-out
   - 到这一层为止，算法语义恢复必须结束
   - 注意：当前 Lift 位置在 `SplitBlackholeKernel` 之后，此时 `T.Pipelined` 的 stage 结构和 `T.copy/T.reduce` 的 tile 语义已在 `inject_pipeline` / `LowerTileOp` 阶段丢失，Lift 仍需从压碎后的 TIR + analysis attrs 做 pattern match（见 §8.6 Lift 时机与长期迁移路径）

3. **Target Program Lowering 层**
   - 把已明确的算法语义映射成目标硬件协议
   - 对 TT-Metal/Blackhole 来说，这一层才允许出现 `kernel role / CB / semaphore / compile-time args / runtime args / accessor binding`
   - 这一层必须包含 **dst register layout planning**（静态分配 tile scratch / stats scratch / matmul accumulator 在 dst 寄存器空间中的 offset）和 **carry strategy 选择**（register-resident carry vs CB-round-trip carry），这两者是生成正确 TT-Metal compute kernel 的前提（见 §8.7）

4. **Codegen / Runtime Materialization 层**
   - 只负责格式化、装配和执行
   - 不再承担结构推理和语义恢复

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

### 5.3 下一阶段 compiler-internal 语义协议：`Stateful Tiled IR`

`Stateful Tiled IR` 是下一阶段 compiler 内部使用的统一语义层，不直接暴露给 Python 用户，也不直接长成 TT-Metal 的硬件协议。当前 Phase 1 的核心对象是 `Domain / State / Relation / Phase` 四类；`Op` 只作为 Phase 2 的设计预留，不在 Phase 1 固化：

1. **`Domain`**
   - 表达计算迭代域的形态：`dense`、`segmented`、`routed`、`paged`
   - 回答”这是规则 tile 域，还是按 expert/page/index 重映射后的域”
   - Domain 必须支持 **constraints / predicates**，不能只靠 kind enum：
     - `data-dependent bound`：causal attention 的 loop range 依赖 `bx`（如 `T.min(ceildiv(seq_kv, block_N), ceildiv((bx+1)*block_M + past_len, block_N))`）
     - `masked / predicated`：block sparse attention 的 `block_mask[k] != 0` 筛选
     - `grouped`：GQA 的 `by // groups` index remapping
   - 这三种形态出现在最常用的 attention variants 中，不是 exotic case

2. **`State`**
   - 所有可变对象不再统称普通 buffer，而是显式区分 kind：
     - `matrix_state`（如 `acc_o`，二维 tile 形态）
     - `vector_state`（如 `scores_max`，一维 row 形态）
     - `scalar_state`（如 `logsum`）
     - `index_state`（如 route idx / page idx / selection idx）
   - 命名使用中性的 `matrix / vector / scalar`，不使用 `tile_state`——“tile” 只在 target lowering 层出现，这样同一份 Stateful Tiled IR 对 CUDA/AMX 等后端仍然成立
   - selection state machine 在 Phase 1 中先编码为 `index_state + Relation / Phase pattern`，不再单独引入新的 state kind
   - 每个 state 需带生命周期：`ephemeral`、`carry`、`cross_phase`
   - carry state 在硬件上有两种实现策略：`register-resident carry`（dst 寄存器长期持有，不 pack/unpack）和 `CB-round-trip carry`（每次迭代 pack/unpack）。**这是 target lowering 的决策，不是 Stateful Tiled IR 层的**——IR 层只标 `carry`，target lowering 根据 dst 寄存器压力选择策略

3. **`Relation`**
   - 表达 state / tensor / domain 之间的绑定关系，例如：
     - `reduced_from`
     - `applies_to`
     - `indexes`
     - `scatters_to`
     - `carried_across`
   - combine 规则**不使用固定 enum**（`sum/max/min/overwrite`），而是引用 TIR body 中的 reduction function 或保持为 function reference。理由：Welford combine 是多变量耦合的 `(mean, M2, count) = welford_combine(old, new)`，online softmax 的 rescale+accumulate 也不是简单的 sum/max，固定 enum 无法覆盖
   - `carried_across` 和 Phase 的 `live-in/live-out` 可以互推——只维护一个作为 source-of-truth，另一个作为 derived；Validate pass 必须检查一致性

4. **`Phase`**
   - 显式表达阶段边界、live-in/live-out state、以及阶段间 carry
   - 不直接编码 `reader/compute/writer`，而是先表达算法阶段关系
   - **必须区分两种 Phase**：
     - `algorithm phase`：compute 内的逻辑步骤。flash-attn 的整个 K-loop 在 TT-Metal 上是**单个 algorithm phase**——所有 carried state 在同一个 `tile_regs_acquire/release` block 内原地更新
     - `pipeline phase`：reader/writer data prefetch 流水线。`T.Pipelined(num_stages=2)` 的 stage 是 data prefetch pipeline，和 compute 内的 algorithm phase 正交
   - Phase 1 主要关注 algorithm phase；pipeline phase 的 stage structure 在 `inject_pipeline` 后已丢失，后续 Lift 时机前移后再覆盖

5. **`Op`**（Phase 2 预留，Phase 1 不固化）
   - 设计意图：表达算法级原语（`tile_compute / state_reduce / state_map / tile_apply_state / state_update / gather / compact / scatter_reduce`），而不是硬件指令
   - 当前不把它做成 Phase 1 的一等对象。理由：
     - TT-Metal SDPA 的 dest-reuse matmul + rescale 组合（`custom_mm_reuse_dest_srcb_block`）无法用固定 Op kind 表达
     - Op kind 一旦固化，每增加一种新 workload 可能都要加新 Op，这和”不为 workload 定制专属 IR”的原则矛盾
   - Phase 1 用 `Phase` 内的原始 TIR statement + State/Relation annotation 来表达语义
   - Phase 2 覆盖 MoE/topk/scan 等更多 pattern 后，再回收统一 Op 分类

硬约束：

- 每个可变对象都必须有明确 `state kind`
- 每个 carry/update/merge 都必须显式，不允许继续靠普通 store/load 隐含语义
- 每个 gather/scatter/routed access 都必须显式说明 index 来源或 combine 规则
- 任何 TT-Metal 特有对象（`CB / semaphore / kernel role / runtime arg`）都不进入这一层

### 5.4 现有 DSL/TIR 与 `Stateful Tiled IR` 的对接边界

默认路径是：

`现有 TileLang DSL -> PrimFunc/TIR -> Semantic Recovery / Lift -> Stateful Tiled IR`

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

这条边界的目标是：**内部 IR 可以大幅增强，但不把复杂度转嫁给用户，不让用户学习 TT-Metal 的编程模型。**

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

### 8.3 中期架构演进方向：`Stateful Tiled IR`

与其继续在 target-specific 后段做"先压碎再恢复"，下一阶段正式方向是引入 compiler-internal `Stateful Tiled IR`，把算法理解与硬件映射硬切开。

统一性要求如下：

| workload | Domain | State / Relation 的核心组合 |
|---------|--------|----------------------------------|
| FlashAttention / online softmax | `dense`（部分变体叠 `paged`） | `matrix_state(carry) + vector_state(carry) + scalar_state(carry) + reduced_from + carried_across` |
| Causal Flash-Attn | `dense + data-dependent bound` | 同上 + `bound_expr = f(bx)` |
| GQA | `dense + grouped` | 同上 + `group index remapping` |
| Block Sparse Attention | `dense + predicated` | 同上 + `sparsity predicate on block_mask` |
| Welford LayerNorm / RMSNorm | `dense` | `vector_state(carry) × 2-3 (mean/M2/count) + coupled combine function` |
| Linear Attention / Mamba | `dense` | `matrix_state(carry) + direct accumulate (h += K^T V)` 或 `recurrence (A*h + B*x)` |
| Split-K GEMM | `dense + partitioned-K` | `matrix_state(carry) + cross-core combine function` |
| MoE / routed grouped GEMM | `routed + segmented` | `index_state + indexes + scatters_to + segmented domain boundary` |
| topk / argmax / selection | 多数为 `dense` | `index_state(carry) + reduced_from + carried_across + compact/select rewrite` |
| paged decode / sparse MLA | `paged`，经常叠 `routed` | `index_state + indexes + carried_across` |
| scan / Mamba recurrence | 多数为 `dense` | `matrix_state/vector_state(carry) + carried_across + phase-ordered recurrence` |

也就是说：

- attention 不应要求专用 `flash_attention` IR
- MoE 不应要求专用 `moe` IR
- topk 不应要求专用 `topk` IR
- paged decode 不应要求专用 `paged_attention` IR

真正的一致性来自统一的 `Domain / State / Relation / Op / Phase` 模型，而不是 workload-specific builtin 集合。

### 8.4 对 Python DSL 兼容性的正式要求

下一阶段设计明确采用：

- **内部 IR 大幅增强**
- **Python DSL 只允许极小变化**
- **任何新增 Python 侧能力都必须保持算法语义视角，不泄露硬件编程模型**

正式边界：

1. 现有复杂 examples 应尽量保持主体写法不变
2. 真正需要补充的 Python 侧信息，以 annotation/helper 方式出现，而不是要求用户改写 kernel 主体结构
3. annotation 的心智模型应保持在“carry / routing / selection / combine 规则”层，而不是“CB / semaphore / kernel split / runtime args”层

### 8.5 对当前规划的影响与 rollout

- **Stage 0-3 已完成的基础设施不推翻**：
  - `ExecutableSpec`
  - `BlackholeModule` direct host path
  - copy / GEMM / multi-core 主链
- **当前 flash-attn compile/runtime bring-up 仍有价值**：
  - 它已经把问题收敛到真实语义边界，而不是把问题藏在 build/codegen 噪声里
- **下一阶段实施顺序（Phase 1a / 1b 拆分）**：
  - **Phase 1a**（纯 IR 增量，零回归风险）：
    1. 引入 `Domain / State / Relation / Phase` 核心对象（Op kind 延后到 Phase 2）
    2. 引入 `LiftToStatefulTiledIR` 与 `ValidateStatefulTiledIR`
    3. 验收标准：lift 能跑、validate 能拦、现有 GEMM/copy compile-path 测试不回归；TT-Sim runtime 总回归留到 Phase 1b
  - **Phase 1b**（语义迁移，target lowering 重构）：
    1. 实现 `BlackholeStatefulProgramLowerer`，包含 dst register layout planning
    2. 把 flash-attn compute 从混合 `blackhole.acc` 语义迁到 Stateful Tiled IR 消费路径
    3. 验收标准：flash-attn runtime correctness 通过
  - **Phase 2**（更宽覆盖面）：
    1. 回收统一 Op kind 分类
    2. 把现有 `AnalyzeBlackhole*` 中承担”算法理解”的部分前移并泛化
    3. 按语义类别扩到 Welford norm / linear-attn / topk / routed / paged / scan
- **验证分层**：
  - lift 层：IR snapshot / structural tests
  - target-program 层：CB / dst register layout / runtime-arg / accessor binding 结构验证
  - runtime 层：TT-Sim correctness / 对拍

### 8.6 Lift 时机与长期迁移路径

当前 Lift 位置在 Blackhole pipeline 的**后段**——`SplitBlackholeKernel` 之后：

```text
SplitBlackholeKernel → Analyze* → LiftToStatefulTiledIR → Validate → LowerBlackholeOps
```

在到达这个位置之前，以下关键信息已丢失：

| 信息 | 丢失位置 | 到达 Lift 时还在？ |
|------|---------|-------------------|
| `T.Pipelined` 的 stage/order/group | `inject_pipeline` | 否 |
| `T.gemm` 的 M/N/K/transpose | `LowerTileOp`（Blackhole: 保留为 `gemm_py`） | 是 |
| `T.copy` 的 tile 语义 | `LowerTileOp` | 否——已变成标量 loop |
| `T.reduce` 的 reduction type/dim | `LowerTileOp` | 否——已变成标量 loop |
| `T.alloc_fragment` 的 scope | `LowerTileOp`（fragment → local） | 否 |
| fragment 上的 element-wise ops | `LowerTileOp` | 否——已变成标量 BufferStore/Load |
| loop-carried state（如 `scores_max *= scale`） | 从未丢失，但变成普通标量赋值 | 部分——需要从标量 pattern match |

**两种路径**：

- **路径 A（Phase 1 选择）**：接受 Lift 在后段，依赖 Analyze pass 从压碎后的 TIR 恢复语义。本质是”把 pattern match 搬到 Analyze pass 里，Lift 只做格式转换”。
  - 优点：不改现有 lowering pipeline 前半段
  - 缺点：Analyze pass 的泛化是核心瓶颈，每种新 pattern 要写新 matcher

- **路径 B（长期目标）**：把 Lift 前移到 `inject_pipeline` 之后、`LowerTileOp` 之前。在这个位置 `T.gemm/T.copy/T.reduce/T.Pipelined` 的结构信息全部还在，Lift 做结构化提取而非 pattern match。
  - 优点：Lift 不再从标量 TIR 猜语义
  - 缺点：需要改 pipeline 前半段，且此时还没有 host/device split

**迁移条件**：当 Phase 1b 完成、且 Phase 2 开始覆盖 Welford/linear-attn 等新 pattern 时，Analyze pass 的 matcher 膨胀会成为工程瓶颈。此时应启动 Lift 时机前移。

### 8.7 TT-Metal SDPA Ground Truth 与 Target Lowering 设计约束

基于 TT-Metal 原生 SDPA 实现的审查结论，target lowering 必须处理以下设计约束。Phase 1b 的 lowering ground truth 以这三个具体 reference 文件为准：

- `tt_metal_repo/tt-train/sources/ttml/metal/ops/sdpa_fw/device/kernels/compute/sdpa_fw_compute_kernel.cpp`
- `tt_metal_repo/models/demos/deepseek_v3_b1/kernel_includes/tt_metal/include/compute_kernel_api/sdpa.h`
- `tt_metal_repo/tt-train/sources/ttml/metal/ops/sdpa_fw/device/sdpa_fw_program_factory.cpp`

#### 8.7.1 dst 寄存器是跨 K-loop 长期持有的

TT-Metal SDPA 的 `tile_regs_acquire/release` 包裹**整个** chunk loop，不是每次迭代：
```text
tile_regs_acquire()              // 整个 chunk loop 之前
for chunk in 0..num_chunks:
    compute_sdpa_chunk(...)      // 内部不 acquire/release
tile_regs_commit/wait/release()  // 整个 chunk loop 之后
```

这意味着 `acc_o / scores_max / logsum` 全部 live 在 dst 寄存器里原地更新，不做 CB round-trip。

#### 8.7.2 dst 寄存器布局是编译时静态规划的

TT-Metal SDPA 的 dst 空间静态分区：
```text
[0, mm2_dst_offset + num_tiles_v * packed_tile_size)  = output accumulator
[max_dst_offset, max_dst_offset + 2)                  = running max (col 0)
[sum_dst_offset, ...]                                 = running sum (col 32+)
[corr_exp_dst_offset, ...]                            = correction factor
[mm1_dst_offset, ...]                                 = QK temporary
```

**`BlackholeStatefulProgramLowerer` 必须包含 `PlanDstRegisterLayout` 步骤**，根据 State 的 shape/kind 做静态 offset 分配。这和 `PlanBlackholeCB` 是平行的另一个 resource planner。

#### 8.7.3 CB 分类需要从 State kind 推导

SDPA 使用 16 个 CB，按功能分为：
- **Transport CB**（输入 Q/K/V/mask）：reader push → compute pop，标准 FIFO
- **Ping-pong state CB**（prev/cur max、sum、mm_out）：跨迭代 state 的 host-visible 持久化
- **Temporary CB**（qk_result、exp_diff）：compute 内单步用完即弃
- **Output CB**：最终结果写出

`PlanBlackholeCB` 的 resource class 应该从 `StatefulTiledProgram` 的 State kind + lifetime 推导，而不是从 CB requirement 的 role 字符串推导。

#### 8.7.4 Carry strategy 是 target lowering 决策

同一个 `carry` state，在不同变体中可能选择不同策略：
- **Chunk-based SDPA**：register-resident carry（dst 长期持有，性能最优但 dst 空间有限）
- **Row-based SDPA**：CB-round-trip carry（每次迭代 pack/unpack，dst 压力更小但带宽开销大）

Target lowering 必须根据所有 carry state 的总 dst 空间需求，决定哪些走 register-resident、哪些退化到 CB-round-trip。

#### 8.7.5 MATH/PACK 线程并行

TT-Metal 的 fused matmul 里 Sigmoid/SiLU 在 PACK thread 上执行（和 MATH thread 并行）。这是 Tensix 核心的 MATH/PACK/UNPACK 三线程并行特性。Stateful Tiled IR 不需要为此建模（纯硬件细节），但 `BlackholeStatefulProgramLowerer` 必须有能力做”这个 elementwise op 可以调度到 PACK thread”的决策。

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
