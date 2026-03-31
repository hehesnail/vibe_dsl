# Blackhole 前向 Flash-Attention 语义子集设计

## 基本信息

- **文档ID**: `stage4_flash_attention_forward_subset`
- **日期**: 2026-03-31
- **状态**: 活动中（analysis 已落地，fragment/dataflow lowering 正在推进）
- **范围**: `tilelang_repo/examples/flash_attention/example_mha_fwd_bshd.py` 与 `example_gqa_fwd_bshd.py` 的前向完整语义；不包含 backward、varlen、wgmma

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
- `codegen_blackhole` 已接上当前最小 fragment builtin 子集
- 当前真实 blocker 已收敛为 `local/accumulator -> shared(CB)` staged copy 还没有正式 lowering

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

这意味着 legality 需要成为正式设计的一部分，而不是失败时再由后段临时拒绝。

## 10. 现有 schema 的收缩建议

### 10.1 明确保留在 `ExecutableSpec` 的

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

### 10.2 优先保留在 IR / split 后 attrs 的

- work-dependent index / loop-bound 结构
- fragment compute 结构
- pipeline stage 结构
- 可由 split 后 IR 再分析得到的 region / dependency 信息

### 10.3 需要警惕继续膨胀的

- `blackhole.segment_plan`：应偏向 segment 边界与角色，不应继续增长为半个 `ExecutableSpec`
- `AccessorSpec` / `compile_time_arg_specs`：只承载 ABI / materialization 必需部分，避免重复表达同一事实
- `grid_x/grid_y/work_per_core/...` 这类摘要 attrs：若 `core_plan` 已是真源，应尽量降级为调试信息或短生命周期 attrs
- 后续新增 `*DescriptorSpec`：只有 host/runtime 真要消费、且 IR 不在场时才允许新增

## 11. 推荐实施顺序

1. split 前保留并规整 work decomposition / fragment region / pipelined stage 语义
2. 实现 `AnalyzeBlackholeWorkDecomposition`
3. 实现 `AnalyzeBlackholeFragmentRegions`
4. 实现 `AnalyzeBlackholePipelineStages`
5. 扩 `LowerBlackholeOps`，消费 analysis 结果并生成最小冻结结论
6. 在 codegen / runtime 侧只消费冻结结论，不新增算法解释逻辑
7. 以 `mha_fwd_bshd` / `gqa_fwd_bshd` 前向为目标做 correctness / legality 验证

## 12. 当前剩余主 blocker

当前不是 analysis、schema、或 pointwise fragment 子集本身在卡住主线。  
当前最真实的剩余点是：

- 优化后 device IR 里仍残留 `O_shared_1[tx, ...] = O_shared_local_cast[...]` 这类二维 `BufferStore`
- 其语义本质是 `local/accumulator -> shared(CB)` staged copy
- 这条方向还没有被 `LowerBlackholeOps` 正式 lower 成 Blackhole dataflow primitive
- 因此 full path 目前仍会晚到 shared 非扁平 store 失败

## 12. 验证方式

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

## 13. 结论

本设计不建议沿着“新增更多 descriptor”继续扩，而建议：

- 优先把复杂 kernel 语义保留在 IR
- 通过新的通用 analysis pass 进行形式化
- 只把 host/runtime/codegen 无法再从 IR 恢复、且必须冻结的结论下沉到最小化 `ExecutableSpec`

一句话概括：

**以 flash-attn forward 为牵引，补齐 Blackhole 对复杂 tiled kernel 的通用 work decomposition、fragment compute region、pipelined staging 三类分析与 lowering 能力；IR 优先，spec 最小化。**
