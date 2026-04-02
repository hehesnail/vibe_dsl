# Blackhole 前向 Flash-Attention 语义子集设计

## 基本信息

- **文档ID**: `stage4_flash_attention_forward_subset`
- **日期**: 2026-03-31（创建），2026-04-02（最近更新）
- **状态**: 活动中（作为 `flash-attn` consumer 支持设计；analysis 与当前支持面的 compile-path 已打通；execution hang 已解；剩余问题已收敛为 layered IR 迁移前的 compute 语义正确性问题）
- **范围**: `tilelang_repo/examples/flash_attention/example_mha_fwd_bshd.py` 与 `example_gqa_fwd_bshd.py` 的前向完整语义；不包含 backward、varlen、wgmma

> 角色说明：本文档现在只作为 **Flash-Attention 这一类 consumer 的支持设计**。总体架构、IR 分层、实现顺序和对象 schema 一律看 `final_blackhole_backend_redesign.md`。本文不再提供总体方向，也不再作为 implementation plan 入口。旧的 `stateful_tiled_ir` 实施计划已归档到 `archive/2026-04-02-stateful-tiled-ir-phase1-implementation-plan.md`。

## 1. 目标

当前 Blackhole 后端正式主链已能跑通 copy 与 GEMM，但仍无法承接真实复杂前向 kernel。

本设计的目标不是新增一个 `flash_attention` 专用旁路，也不是把当前样例结构硬编码成协议，而是：

1. 以 `mha_fwd_bshd` / `gqa_fwd_bshd` 为牵引，补齐 Blackhole 对复杂 tiled kernel 的通用分析与 lowering 能力
2. 不修改 example 本身，支持其当前已经写出的前向完整语义：
   - `causal / non-causal`
   - `groups`
   - `num_stages`
   - 不同 `block_M / block_N / threads` 组合
3. 同时遵守 TileLang DSL 设计意图与 TT-Metal / Blackhole 硬件执行现实
4. 优先把语义保留在 IR 与 analysis pass 中；只把 host/runtime/codegen 无法再从 IR 恢复、且必须冻结的最终结论下沉到最小化 `ExecutableSpec`

### 当前实施位置

- `AnalyzeBlackholeWorkDecomposition` / `AnalyzeBlackholeFragmentRegions` / `AnalyzeBlackholePipelineStages` 已落地
- `LowerBlackholeOps` 已开始把 fragment 子集 lower 成 Blackhole builtin
- `codegen_blackhole` 已接上当前最小 fragment/dataflow builtin 子集
- `local/accumulator -> shared(CB)` staged copy 已经 lower 成正式 builtin
- 当前支持的 MHA/GQA forward compile-path 已打通；剩余工作已从 execution hang 转为 compute 语义重构与更宽支持面
- 当前又收掉了两类执行期协议问题：
  - `cast_fragment_slice` 写入的 `blackhole.acc` scratch CB 若会被后续 matmul 读取，必须按未来 matmul 所需页数 `cb_push_back`
  - `blackhole.acc` 作为 GEMM 输出时，compute 侧不能在 `pack_tile` 前对同一输出 CB 重复 `cb_reserve_back`
- direct runtime `mha` 已不再 hang，当前失败形态已收敛为 execution 完成后的 correctness mismatch
- 当前已确认的根因方向是：`blackhole.acc` 不能继续同时承担“线性 fragment scratch”和“TT-Metal tile/CB scratch”双重语义

## 2. 非目标

- 不支持 backward
- 不支持 `varlen`
- 不支持 `wgmma`
- 不引入 `flash_attention_plan`、`attention_work_contract` 之类 attention 专属协议
- 不新增第二条执行路径；正式执行路径仍是 `ExecutableSpec -> BlackholeModule`

## 3. 现状与缺口

`example_mha_fwd_bshd.py` / `example_gqa_fwd_bshd.py` 实际依赖的是一组复合语义，而不是单个新 op：

- 3D `T.Kernel(... ) as (bx, by, bz)`
- work-dependent 派生索引与 loop bound：
  - `by // groups`
  - causal 时 `loop_range` 依赖 `bx`
- `T.Pipelined(loop_range, num_stages=...)`
- fragment state：
  - `acc_s`
  - `acc_s_cast`
  - `acc_o`
  - `scores_max`
  - `scores_max_prev`
  - `scores_scale`
  - `scores_sum`
  - `logsum`
- fragment 上的复合计算：
  - `fill`
  - `reduce_max(dim=1, clear=False)`
  - `reduce_sum(dim=1)`
  - `max`
  - `exp2`
  - `mul/add/div`
  - row-wise broadcast
  - `if_then_else` 掩码初始化
- 两次 GEMM 与中间 online softmax update 的组合

当前 Blackhole 主链缺的不是一个 “attention op”，而是三类更通用的能力：

1. `work decomposition analysis`
2. `fragment compute region analysis + lowering`
3. `pipelined stage analysis`

但截至 2026-04-01，除了“缺能力”之外，还新增了一个更根本的结论：

- 当前 flash-attn 的 correctness blocker 不再是单纯 execution protocol bug
- 当前更深层的问题是 **compute 语义模型和 TT-Metal 主路径不一致**
- 具体表现为：
  - 上游 TIR / analysis 仍然允许把 `acc_s` / `acc_o` 这类状态理解成线性 fragment
  - `GenerateMatmulSequence` 却在 compute 里生成 `mm_init -> matmul_tiles -> pack_tile -> cb_push_back` 的 tile/CB 流
  - 后续 pointwise / reduction / cast helper 又把同一份 `blackhole.acc` scratch 当线性 `float*` / `half*` 连续数组解释
- 这套混合语义在死锁问题收口之后，已经暴露为稳定的 correctness mismatch，而不是偶发 runtime 行为

## 4. 设计原则

### 4.1 IR 优先，schema 最小化

- 能从 IR 分析得到的信息，不下沉到 host/runtime schema
- 只有 runtime/codegen 不在 IR 现场、且必须消费的最终冻结结论，才进入 `ExecutableSpec`
- 不把 analysis result 默认设计成新的 `*DescriptorSpec`

### 4.2 通用 pass 优先，attention 只是第一批 consumer

- 不为 flash attention 设计专属协议
- 把本次需要的能力尽量上提为新的通用 analysis / canonicalization / legality pass
- flash attention forward 只是第一批需要这些 pass 足够强的复杂 kernel

### 4.3 TileLang 语义与 TT-Metal 硬件共同约束

- 不能只从 TileLang 语义出发，假装硬件是通用 GPU
- 也不能只从 TT-Metal 约束出发，把 DSL 语义压成不可复用的特判
- 正确目标是定义面向 Blackhole 的 TileLang 复杂 tiled-kernel 映射，而不是 attention 专属后门

### 4.4 TT-Metal-first compute 语义

- flash-attn compute 正式方向改为 **TT-Metal-first**
- `blackhole.acc` 不再作为“线性 fragment scratch”的长期抽象
- `blackhole.acc` 后续只表示 compute-side **tile scratch**
- `CB / tile / dst-reg` 流是 flash-attn compute 的正式主路径
- 现有线性 fragment helper 仅保留为过渡实现，不再作为长期协议前提

## 5. 分层设计

### 5.1 split 前：保留复杂 kernel 关键语义

split 前必须保留，而不是压碎后再让后段恢复的内容：

- work-dependent index / loop bound
- fragment producer / consumer 关系
- row-wise reduction 与 row-broadcast 结构
- `T.Pipelined` 的 stage 结构
- loop-carried fragment state

split 前不做的事：

- 不直接生成最终 runtime args
- 不直接生成 TT-Metal host objects
- 不按 flash attention 模板识别整段算法

### 5.2 split 后：消费通用分析结果，生成 Blackhole 冻结结论

split 后的目标不是重新理解 attention，而是消费通用 analysis pass 的结果，生成：

- Blackhole 侧 kernel role 划分
- CB / shared / semaphore / launch / runtime ABI 需求
- 当前 Blackhole fragment compute subset 能否承接该 region 的 legality 结论
- `ExecutableSpec` 所需的最小冻结事实

### 5.3 host/runtime：不再理解算法，只 materialize 计划

`BlackholeModule` 只应负责：

- materialize kernel / buffer / CB / semaphore / runtime args
- launch program
- readback

`BlackholeModule` 不应负责：

- 重建 online softmax 语义
- 根据几个 arg 名字推断 groups / causal / reduction
- 在 host 侧重新分析 fragment 计算结构

## 6. 新增通用 pass 轮廓

### 6.1 `AnalyzeBlackholeWorkDecomposition`

职责：

- 提取 launch axes
- 识别 derived index expr
- 识别 work-dependent loop bound
- 形成统一 work decomposition 视图

输出形式：

- 优先作为 split 后 IR attrs / canonical form
- 不直接进入 `ExecutableSpec`

### 6.2 `AnalyzeBlackholeFragmentRegions`

职责：

- 识别 fragment buffer 的 region 与角色
- 识别 row reduction / row broadcast / pointwise chain
- 识别 loop-carried fragment state
- 形成稳定 fragment compute region

输出形式：

- 优先作为 split 后 IR attrs / canonical form
- 不直接进入 `ExecutableSpec`

### 6.3 `AnalyzeBlackholePipelineStages`

职责：

- 识别 `T.Pipelined` 对应的 stage 结构
- 提取 stage-local shared buffers
- 提取 loop-carried state
- 形成 stage legality 输入

输出形式：

- 优先作为 split 后 IR attrs / canonical form
- 不直接进入 `ExecutableSpec`

## 7. `LowerBlackholeOps` 的新边界

`LowerBlackholeOps` 应变成：

- 通用 analysis 的 consumer
- Blackhole-specific legality / schema extractor

它负责：

- 消费 work decomposition / fragment region / pipeline stage analysis
- 产出 Blackhole 真正需要冻结的 segment / runtime/common-runtime / compile-time ABI / CB requirements / compute-side lowering 结论

它不负责：

- 重新从晚期 loop 结构恢复 fragment semantics
- 重新解释 `T.Pipelined`
- 做 flash attention 模板匹配

## 8. 第一版 Blackhole fragment compute subset

为承接这两个前向例子，第一版需要正式支持的 fragment compute 子集为：

- `fill(fragment, scalar)`
- fragment/shared 相关 `copy`
- `reduce_max(dim=1, clear=False)`
- `reduce_sum(dim=1)`
- `max`
- `exp2`
- row-broadcast `mul/add/div`
- fragment pointwise `mul/add/div`
- mask init 中的 `if_then_else`
- 与现有 `gemm` 的串接

这不是 attention subset，而是：

**row-wise reduction + broadcast + pointwise + gemm 的 fragment compute subset**

## 9. TT-Metal / 硬件约束如何进入设计

第一版必须明确接受以下事实：

- 不是所有 `block_M/block_N/threads/num_stages` 都合法
- `num_stages` 必须经过 shared/L1/CB 资源 gate
- 不追求第一版单 kernel 吞完全部前向 attention
- 允许在一条正式主链内通过多 kernel role / 多 segment 完成
- 不引入 GPU / WGMMA 心智

### 9.1 当前 runtime 收敛结论

- compile/codegen 主 blocker 已进一步减少，execution hang 已解
- `blackhole.acc` 当前同时承担线性 fragment scratch 与 TT-Metal tile/CB scratch 两类语义；虽然已经修掉了“漏发页数”和“重复 reserve”两类问题，但 hang 解掉后暴露出这套混合语义本身会产生 correctness mismatch
- 当前最有价值的调试手段仍是 TT-Metal Watcher；本仓库下的 watcher 输出当前默认落在 `generated/watcher/watcher.log`
- 当前最该优先收正的不是更多 execution workaround，而是 `acc_s / acc_s_cast / acc_o` 这一组 compute state 的 **tile-scratch-only 语义**
- TT-Metal matmul / SDPA 参考更接近纯 `CB / tile / dst-reg` 流，而不是当前这种 `blackhole.acc` “直接指针 scratch + CB queue”双重语义

这意味着 legality 需要成为正式设计的一部分，而不是失败时再由后段临时拒绝。

## 10. 新的 compute 语义收敛方向

### 10.1 `blackhole.acc` 的新定义

- `blackhole.acc` 后续只表示 compute-side **tile scratch**
- 它不再表示线性 `float*` / `half*` fragment 数组
- 与 `matmul_tiles / pack_tile / copy_tile / dst-reg` 相关的 compute 主链都必须围绕这个定义收正

### 10.2 scalar state 单独建模

- `scores_max / scores_max_prev / scores_scale / scores_sum / logsum` 这类状态不应继续和 tile scratch 共享同一抽象
- 后续应在 analysis / lowering / planner 上把 scalar scratch 和 tile scratch 显式区分

### 10.3 上游 TIR 对接约束

- Blackhole 后段不再从线性 `BufferLoad/BufferStore` 长相里猜 tile 语义
- 凡是后端无法稳定恢复的 tile contract，必须通过：
  - split 前 preserved TIR 结构
  - split 后 analysis attrs
  - 或显式 Blackhole builtin
  稳定交付
- `tl.tileop.gemm_py` 的全局 `M/N/K` 不能再被直接等价成 compute local tile materialization；flash-attn 这类 kernel 需要更明确的 local tile contract

## 11. 现有 schema 的收缩建议

### 11.1 明确保留在 `ExecutableSpec` 的

- `cb_configs`
- `core_plan`
- `semaphores`
- `buffer_materializations`
- `KernelSpec.launch_spec`
- `KernelSpec.compute_config`
- `KernelSpec.runtime_args`
- `KernelSpec.common_runtime_args`
- kernel role / source / processor / noc

这些都属于 host/runtime 必须知道的冻结事实。

### 11.2 优先保留在 IR / split 后 attrs 的

- work-dependent index / loop-bound 结构
- fragment compute 结构
- pipeline stage 结构
- 可由 split 后 IR 再分析得到的 region / dependency 信息

### 11.3 需要警惕继续膨胀的

- `blackhole.segment_plan`：应偏向 segment 边界与角色，不应继续增长为半个 `ExecutableSpec`
- `AccessorSpec` / `compile_time_arg_specs`：只承载 ABI / materialization 必需部分，避免重复表达同一事实
- `grid_x/grid_y/work_per_core/...` 这类摘要 attrs：若 `core_plan` 已是真源，应尽量降级为调试信息或短生命周期 attrs
- 后续新增 `*DescriptorSpec`：只有 host/runtime 真要消费、且 IR 不在场时才允许新增

## 12. 推荐实施顺序

1. split 前保留并规整 work decomposition / fragment region / pipelined stage 语义
2. 实现 `AnalyzeBlackholeWorkDecomposition`
3. 实现 `AnalyzeBlackholeFragmentRegions`
4. 实现 `AnalyzeBlackholePipelineStages`
5. 扩 `LowerBlackholeOps`，消费 analysis 结果并生成最小冻结结论
6. 在 codegen / runtime 侧只消费冻结结论，不新增算法解释逻辑
7. 以 `mha_fwd_bshd` / `gqa_fwd_bshd` 前向为目标做 correctness / legality 验证

## 13. 当前剩余主 blocker

`local/accumulator -> shared(CB)` staged copy 已 lower 成 `tl.blackhole.write_local_slice_to_cb`，compile-path 不再卡在 residual shared store。

当前剩余点集中在 **compute correctness / 语义模型**：

1. **`blackhole.acc` 混合语义**：`acc_s / acc_s_cast / acc_o` 仍在部分路径里同时被解释成“线性 fragment scratch”和“TT-Metal tile scratch”，这已经从旧的 hang 演变成稳定 correctness mismatch
2. **stats-state 尚未成为正式一等语义**：`scores_max / scores_scale / scores_sum / logsum` 仍缺少和 tile scratch 分离后的正式 state model，导致 row-reduce / row-broadcast / exp-affine 还不能完全收进同一主路径
3. **late matcher 仍在承担过多语义恢复责任**：只要后段还在从普通 `BufferLoad/BufferStore/For` 长相猜 tile contract，flash-attn 这类复杂 kernel 的 correctness 风险就不会真正收口

因此当前下一步已经不是继续围绕 execution-time hang 缩小，而是把这批语义前移到新的分层 compiler-internal IR：先冻结到 `Stateful Semantic IR`，再逐步下沉到 `Spatial Program IR` 与 `TT Target IR`。

## 13.1 已发现的设计约束

### CB-as-fragment-scratch

flash-attn compute kernel 中 fragment state（`acc_o`、`scores_max`、`logsum` 等）通过 CB interface 访问 L1，而非 FIFO push/pop 协议。这是 TRISC 访问 L1 的正确方式，但与 transport CB 的 FIFO 语义不同。详见 `final_blackhole_backend_redesign.md` 中 PlanBlackholeCB 段落。

### kIntermediate CB 池变更

fragment scratch CB 必须在 16-31 范围内（TRISC 可访问），因此 `kIntermediate` 现与 `kOutput` 共享 16-31 池。flash-attn 当前使用 12 CB。

## 14. 验证方式

设计完成后的实现验证应至少覆盖：

1. IR / pass 层：
- 新 pass 能稳定分析 `mha_fwd_bshd` / `gqa_fwd_bshd`
- split 前后关键语义未丢失

2. lowering / spec 层：
- `LowerBlackholeOps` 不依赖 attention pattern hack
- `ExecutableSpec` 没有引入不必要的新 attention-specific schema

3. runtime / codegen 层：
- 对支持面内配置可 materialize
- 对不支持面显式 fail-fast，错误边界清晰

4. example 层：
- `example_mha_fwd_bshd.py`
- `example_gqa_fwd_bshd.py`

## 15. 结论

本设计不建议沿着“新增更多 descriptor”继续扩，而建议：

- 优先把复杂 kernel 语义保留在 IR
- 通过新的通用 analysis pass 进行形式化
- 只把 host/runtime/codegen 无法再从 IR 恢复、且必须冻结的结论下沉到最小化 `ExecutableSpec`

一句话概括：

**以 flash-attn forward 为牵引，补齐 Blackhole 对复杂 tiled kernel 的通用 work decomposition、fragment compute region、pipelined staging 三类分析与 lowering 能力；IR 优先，spec 最小化。**
