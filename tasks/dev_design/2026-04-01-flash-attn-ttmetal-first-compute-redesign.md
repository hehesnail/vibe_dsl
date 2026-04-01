# Blackhole Flash-Attn TT-Metal-First Compute 语义重构设计

## 基本信息

- **文档ID**: `2026-04-01-flash-attn-ttmetal-first-compute-redesign`
- **日期**: 2026-04-01
- **状态**: 提议中
- **适用范围**: `tilelang_repo` Blackhole 后端的 split 后 compute 语义、TIR 对接界面、planner / codegen / runtime materialization 边界
- **关联文档**:
  - `tasks/dev_design/final_blackhole_backend_redesign.md`
  - `tasks/dev_design/stage4_flash_attention_forward_subset.md`
  - `tasks/dev_design/stage2j_compute_contract_schema.md`
  - `tasks/dev_design/stage2e_blackhole_device_resource_semantics.md`

## 1. 背景与问题定义

Stage 4 当前 compile-path 已基本打通，flash-attn forward 的执行期 hang 也已解掉；但随后暴露出更根本的问题：

1. `blackhole.acc` 当前同时承担两种不兼容语义：
   - TT-Metal compute 主路径里的 **tile scratch / matmul destination**
   - 线性 fragment helper 眼中的 **`float*` / `half*` scratch 数组**
2. `LowerBlackholeOps` 当前仍会从普通 `BufferLoad/BufferStore/For` 长相里恢复 compute 语义，导致：
   - 多 tile matmul output 被错误塌缩成单次 `pack_tile / cb_push_back`
   - `scores_max / scores_sum / logsum / exp(max diff)` 这类 statistics state 与 tile scratch 混到同一资源语义
3. `codegen_blackhole` 仍保留对 `blackhole.acc` 的线性寻址主路径，说明后段依然在“猜” compute state，而不是消费已经明确的 IR 协议

这不是某一个 runtime 协议 bug，而是 **compute 状态模型和 TT-Metal 正式编程模型不一致**。

TT-Metal 的正式模型是：

- transport 数据通过 CB 按 `reserve / wait / push / pop` 流动
- matmul 通过 `mm_init -> tile_regs_acquire -> matmul_tiles -> tile_regs_commit / wait -> pack_tile -> cb_push_back` 工作
- SDPA / flash-attn 的中间态按 `qk_im / out_im / max / sum / exp_diff / out` 等不同资源角色拆开，而不是混成一个线性 fragment 数组

因此 Stage 4 下一步不应该继续给现有线性 helper 打补丁，而应该正式收正：

- compute state 类型系统
- TIR 与 Blackhole 后段的对接协议
- split 后 compute lowering 边界
- planner / codegen / runtime 只消费冻结结果的职责边界

## 2. 目标

本设计的目标是：

1. 把 Blackhole compute 主链收正为 **TT-Metal-first** 的 `CB / tile / stats / dst-reg` 语义
2. 明确 TIR 对接界面，不再允许后段从线性 `BufferLoad/BufferStore` 形态猜 tile contract
3. 让新的 compute 主链直接覆盖当前已 bring-up 的 GEMM 与 flash-attn case，而不是保留 legacy path
4. 只对 `ExecutableSpec` 和 host/runtime 增加最小必要冻结事实，不引入 attention 专属 schema

## 3. 非目标

本设计不做以下事情：

- 不新增第二条执行路径
- 不新增 `flash_attention_spec`、`sdpa_spec` 之类 attention 专属协议
- 不把 TT-Metal 某个现有 SDPA kernel 的具体实现细节直接固化成 TileLang 后端协议
- 不在 host/runtime 侧恢复 attention 算法结构
- 不把 P4/P5 的更宽 copy/synchronization 与本设计绑成同一专项

## 4. 设计原则

### 4.1 TT-Metal-first，但不 attention-special-case

- Blackhole compute 语义必须贴近 TT-Metal 正式模型
- 但协议边界必须保持通用，不能把某个 flash-attn 样例结构硬编码成后端协议

### 4.2 IR 真源优先

- 能在 TIR / split 后 IR 中表达的事实，不能下沉到 host/runtime 猜
- 后段不再从普通 loop/store 形态恢复 tile 语义
- 缺失的信息必须通过新的 storage scope、attrs、canonical compute region 或 builtin 显式表达

### 4.3 不保留 legacy compute path

- 旧的线性 fragment compute 语义不再作为正式路径保留
- 之前已经 bring-up 的 case 必须迁移到新主链上
- 如果新主链无法覆盖某个 case，应当补 IR 语义和 lowering，不应重新保留旧路径

### 4.4 split 后分层清晰

- analysis 负责识别 compute state 和 compute region
- canonicalization 负责把可稳定恢复的 TIR 规整成正式 lowering 输入
- lowering 只消费显式 region / state，不再从晚期 loop 猜语义
- planner / codegen / runtime 只消费冻结后的事实

## 5. 新的 compute state 模型

### 5.1 state kind 分层

split 后 compute-local state 正式分成四类：

1. `transport_cb`
- 表示 producer/consumer 间或 phase 间的 FIFO 传输资源
- 正式语义是 `reserve / wait / push / pop`

2. `tile_scratch`
- 表示 compute-side tile scratch
- 对应 matmul destination、tile pointwise、tile cast / tile pack / tile copy 等 tile 语义
- 在 storage scope 上对应 `blackhole.acc`

3. `stats_state`
- 表示 row-reduced / broadcast-driven statistics state
- 覆盖 `scores_max`、`scores_max_prev`、`scores_scale`、`scores_sum`、`logsum`、`exp(max diff)` 这类状态
- 在 storage scope 上引入新的 `blackhole.stats`

4. `plain_local`
- 只保留给真正的小临时量
- 不允许再承担长期 tile scratch 或 statistics state 语义

### 5.2 `blackhole.acc` 的新定义

- `blackhole.acc` 后续只表示 `tile_scratch`
- 它不再表示线性 `float*` / `half*` fragment 数组
- 所有与 `mm_init / matmul_tiles / tile_regs_* / pack_tile` 相关的 compute 主链，都必须围绕这个定义收正

### 5.3 `blackhole.stats` 的新定义

- `blackhole.stats` 表示 statistics state，而不是 transport FIFO
- 它可以被 row reduction、row broadcast apply、exp-affine、stats combine 等 compute region 读写
- 它不能作为 `matmul_tiles` 的 output，也不能被当作 transport CB 走 `wait_front / pop_front`

### 5.4 legality 约束

- `blackhole.acc` 不能再出现在 generic 线性 `BufferLoad/BufferStore` compute 主路径中
- `blackhole.stats` 不能被 `matmul_tiles` 直接读写
- `transport_cb` 不能承担 core-local 长期 scratch 生命周期
- 任何跨 phase 的 `local -> shared(CB)` / `stats -> transport` 行为，都必须在 lowering 中显式变成正式 builtin

## 6. TIR 对接界面

### 6.1 正式目标

Blackhole 后段后续不再从“普通 TIR 长相”猜 tile contract。split 前 / split 后 TIR 必须稳定交付以下事实：

1. 每个 compute-local buffer 的 state kind
2. 每个 compute region 的正式语义类别
3. matmul 的 local tile contract
4. phase 间哪些状态需要 publish / consume，哪些只是 core-local 持有

### 6.2 storage scope 语义增强

- 引入新的 storage scope：`blackhole.stats`
- `blackhole.acc` 与 `blackhole.stats` 作为 compute state kind 的一等语义，而不是 codegen 侧 comment
- `BlackholeDeviceResourceCanonicalization` 需要同步认识 `blackhole.stats` 对应的新资源 rank

### 6.3 compute region contract

split 后 IR 必须能把以下 region 稳定交给后段：

1. `matmul output tile region`
2. `row-reduce stats region`
3. `stats update / stats combine region`
4. `stats apply-to-tile region`
5. `tile cast / tile publish region`

对这些 region，后段不再接受“只是一个普通 `For + BufferStore`”的隐式接口。

允许的正式交付方式只有三种：

1. 通过新的 canonical compute region
2. 通过 split 后 analysis attrs
3. 通过显式 Blackhole builtin

### 6.4 local tile contract

matmul 的 compile-time contract 必须表达 **local output tile shape**，而不是继续把全局 `M/N/K` 默认等价成单个输出 tile。

至少需要稳定表达：

- local `Mt`
- local `Nt`
- local `Kt`
- matmul output 对应多少个 output tiles

这个 contract 可以继续进入 compile-time ABI / kernel compile args，但真源必须来自 split 后 compute region，而不是由 `GenerateMatmulSequence()` 从全局 shape 猜。

## 7. 新的 pass 分层

### 7.1 `AnalyzeBlackholeComputeState`

新增 pass，职责：

- 识别 split 后 compute-local buffer 的 state kind
- 输出 state kind 相关 attrs / canonical metadata
- 为后续 legality / canonicalization 提供真源

它不负责：

- 直接 lower 成 builtin
- 生成 host/runtime schema

### 7.2 `CanonicalizeBlackholeComputeRegions`

新增 pass，职责：

- 把当前还能稳定识别的线性 TIR 结构规整成正式 compute region
- 去掉“对普通 loop 长相敏感”的脆弱 matcher 前提
- 为 `LowerBlackholeComputeOps` 生成稳定输入

它不负责：

- 直接生成 TT-Metal host objects
- 直接做 planner 分配

### 7.3 `LowerBlackholeTransportOps`

从当前 `LowerBlackholeOps` 中拆出 transport lowering，职责只保留：

- DRAM -> CB
- CB -> DRAM
- CB -> CB
- staged publish / consume
- accessor / runtime arg / common runtime arg 对应的 transport schema

它不再承担 compute region 语义判断。

### 7.4 `LowerBlackholeComputeOps`

新增 compute lowering pass，职责：

- 消费 `AnalyzeBlackholeComputeState` 与 `CanonicalizeBlackholeComputeRegions` 的结果
- 生成 tile / stats / transport-aware 的正式 builtin 序列
- 产出 compute 侧 CB requirements、compile-time ABI 和 legality 结论

它不再承担：

- 从晚期 loop 恢复 fragment semantics
- 重新解释 `T.Pipelined`
- 根据 buffer 名字猜 `acc_s / acc_o / scores_*` 角色

### 7.5 管线位置

推荐管线顺序调整为：

`AnnotateBlackholeCopySemantics`
-> `BlackholeDeviceResourceCanonicalization`
-> `SplitHostDevice`
-> `SplitBlackholeKernel`
-> `AnalyzeBlackholeWorkDecomposition`
-> `AnalyzeBlackholeFragmentRegions`
-> `AnalyzeBlackholePipelineStages`
-> `AnalyzeBlackholeComputeState`
-> `CanonicalizeBlackholeComputeRegions`
-> `LowerBlackholeTransportOps`
-> `LowerBlackholeComputeOps`
-> `PlanBlackholeCB`
-> `AssignBlackholeCores`

## 8. builtin 与 lowering 协议

### 8.1 transport builtin

沿用并整理现有 transport builtin 体系，用于：

- tile/page read/write
- `cb_reserve_back`
- `cb_wait_front`
- `cb_push_back`
- `cb_pop_front`
- staged publish / consume

### 8.2 tile-compute builtin

tile-compute builtin 族必须覆盖：

- matmul init / execute / commit / wait / release
- multi-tile output pack / publish
- tile cast
- tile copy
- tile-wise pointwise

关键要求：

- 多 tile GEMM output 必须按 local `Mt/Nt` output tile 数发射，不允许再固定单次 `pack_tile / cb_push_back`

### 8.3 stats builtin

stats builtin 族必须覆盖：

- row reduce max / sum
- exp-affine / scale-rescale
- stats combine / update
- stats apply-to-tile

这些 builtin 的输入输出都必须显式区分 `blackhole.stats` 和 `blackhole.acc`，不能再共用“线性 fragment helper ABI”。

## 9. planner 设计

### 9.1 `PlanBlackholeCB` 升级为 class-aware planner

不新增第二个 planner，仍使用 `PlanBlackholeCB`，但 requirement 与 config 需要带上 `resource_class`：

- `transport`
- `tile_scratch`
- `stats_scratch`

### 9.2 planner 职责

`PlanBlackholeCB` 后续需要：

1. 按 class 分配不同生命周期策略
2. 回写新的 tile/stats builtin 参数
3. 做 class-aware post-condition 校验

### 9.3 post-condition 约束

planner 至少要检查：

- `matmul` output 不能绑定到 `stats_scratch`
- `stats reduce/update` 结果不能绑定到 `transport`
- `transport` CB 必须只出现在 FIFO builtin 中
- 所有携带 CB 参数的新 builtin 都必须注册回写位置

## 10. `ExecutableSpec` 与 host/runtime 边界

### 10.1 `ExecutableSpec` 最小新增

本设计不新增 attention 专属 schema，只做最小扩展：

- `cb_configs[*].resource_class`
- 必要的 compile-time tile/stats specialization ABI

不应新增：

- `flash_attention_spec`
- `stats_descriptor_spec`
- 会在 spec 中重复表达 compute region 结构的字段

### 10.2 `rt_mod_blackhole`

`rt_mod_blackhole` 只负责：

- 提取 compile-time ABI
- 提取 runtime args / common runtime args / accessor schema
- 提取 class-aware `cb_configs`
- 构造 `ExecutableSpec`

它不再负责：

- 从 attrs 或 buffer 名字恢复算法结构
- 二次推断 compute state kind

### 10.3 `codegen_blackhole`

`codegen_blackhole` 只消费：

- transport builtin
- tile-compute builtin
- stats builtin
- planner 回写后的 class-aware CB binding

它不再允许：

- 为 `blackhole.acc` / `blackhole.stats` 生成线性 `reinterpret_cast<float*>` / `half*` 主路径
- 从 `AllocateNode` / `BufferStore` 长相推断 tile scratch 或 stats state

### 10.4 `BlackholeModule`

`BlackholeModule` 继续只做 materialization：

- `CreateCircularBuffer`
- `CreateKernel`
- `SetRuntimeArgs`
- `SetCommonRuntimeArgs`
- launch / readback

host/runtime 不理解 attention 算法，不重建 online softmax 结构。

## 11. 迁移与兼容策略

### 11.1 总策略

- 不保留旧的线性 fragment compute path
- 新主链必须覆盖当前已 bring-up 的 copy / GEMM / flash-attn case
- 如果旧 case 依赖错误语义，应通过新的 state typing 与 canonicalization 吸收，而不是保留 legacy 路径

### 11.2 旧 case 收正映射

1. 纯 copy case
- 继续走 transport 主链
- 不受 compute 语义重构影响

2. 纯 GEMM case
- 直接切到新的 tile-compute 主链
- `blackhole.acc` 只保留 tile scratch 语义
- local `Mt/Nt/Kt` 必须显式进入 contract

3. flash-attn case
- `acc_s / acc_o` 收正为 `tile_scratch`
- `scores_max / scores_max_prev / scores_scale / scores_sum / logsum / exp(max diff)` 收正为 `stats_state`
- 原先的线性 pointwise / reduction / cast loop 通过 canonical region + new lowering 吸收

### 11.3 实施顺序

推荐实施顺序：

1. `blackhole.stats` + state typing
2. canonical compute regions
3. new compute lowering + class-aware planner
4. strict codegen/runtime boundary
5. 清理旧 linear fragment compute path

## 12. 影响范围

预计影响：

- `tilelang_repo/src/transform/`
  - 新增 compute-state / canonicalization pass
  - 拆分并重构 `LowerBlackholeOps`
  - 升级 `PlanBlackholeCB`
- `tilelang_repo/src/target/`
  - `rt_mod_blackhole`
  - `codegen_blackhole`
  - `blackhole_module`
- `tilelang_repo/src/runtime/` 与 builtin 注册
  - 新 storage scope / builtin / effect-kind 注册
- `tilelang_repo/testing/python/target/blackhole/`
  - GEMM / flash-attn pipeline/runtime 回归
- `tilelang_repo/testing/python/transform/`
  - state typing / canonicalization / legality 回归

## 13. 验证方式

### 13.1 IR / pass 层

- 新增回归验证 split 后 IR 已显式区分：
  - `transport_cb`
  - `tile_scratch`
  - `stats_state`
  - `plain_local`
- 新增回归验证 `CanonicalizeBlackholeComputeRegions` 后，flash-attn compute 主链不再残留裸线性 fragment loop
- 新增 legality 回归验证 state kind 混用会在 lowering 前显式失败

### 13.2 lowering / codegen 层

- GEMM 回归验证多 tile output 按 local `Mt/Nt` 正确发射 `pack_tile / cb_push_back`
- flash-attn pipeline 回归验证：
  - `blackhole.acc` / `blackhole.stats` 不再以线性数组主路径访问
  - `scores_* / logsum` 不再以 `blackhole.acc` requirement 出现
  - `acc_s / acc_o` 不再被塌成单 tile output publish

### 13.3 spec / runtime 层

- `ExecutableSpec` 回归验证只新增最小 class-aware 字段
- direct runtime 在有 TT-Metal 环境时重新验证：
  - copy
  - GEMM
  - flash-attn MHA/GQA
- 目标不是只证明“不 hang”，而是证明 correctness 回到正式主链

## 14. 决策总结

本设计的最终结论是：

1. Blackhole compute 主链正式采用 TT-Metal-first 的 `transport / tile_scratch / stats_state` 分层模型
2. `blackhole.acc` 后续只表示 tile scratch
3. 新增 `blackhole.stats`，把 statistics state 从 `blackhole.acc` 中彻底剥离
4. TIR 必须显式交付 compute state kind 与 compute region contract；后段不再从线性 TIR 长相猜 tile 语义
5. split 后 pass 负责 state typing、canonicalization 和 lowering；spec / runtime 只消费冻结结果
6. 不保留 legacy compute path；当前 bring-up case 全部迁移到这套新主链
