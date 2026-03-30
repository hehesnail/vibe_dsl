# Stage 2D 设计补充：TT-Metal contract 缺口审计

## 基本信息

- **文档ID**: `stage2d_ttmetal_contract_audit`
- **日期**: 2026-03-30
- **状态**: 审计已完成；收正部分落地（P0 已继续 formalize 到统一 `compute_contract`，P1/P2 ✅，P3 已对 copy + GEMM 主路径 formalize，P4 未做，P5 已完成 semaphore schema/kernel binding/最小 device builtin 预埋）
- **对应阶段**: Stage 2D Step 6
- **关联文档**:
  - `tasks/dev_design/final_blackhole_backend_redesign.md`
  - `tasks/dev_design/stage2d_gemm_direct_cb_io.md`
  - `tasks/dev_design/stage2d_cb_identity_protocol.md`

---

## 1. 目标

把当前 Blackhole direct path 与 TT-Metal 正式编程模型之间仍缺失的 contract 层一次性盘清，避免后续继续把“设计缺层”误诊成单个 codegen/runtime bug。

这份文档不替代总体设计；它只补充 Stage 2D 当前已经确认的协议缺口、影响范围和后续收正顺序。

---

## 2. 当前判断

当前 GEMM direct path 暴露出来的问题，不是单一 `matmul` 样例 bug，而是 Blackhole 后端还没有把 TT-Metal 至少以下五层 contract 正式拆开：

1. host logical tensor contract
2. device buffer/accessor contract
3. CB transport contract
4. compute kernel contract
5. work distribution / synchronization contract

copy 当前之所以能过，主要是因为测试只覆盖了最窄的 tile-aligned / interleaved / replicated case；这不能证明主线协议已经完整。

---

## 3. 审计结论

### 3.1 host logical tensor contract 仍缺失

当前 Blackhole direct path 默认假设：

- host `DLTensor` 的逻辑布局
- device DRAM buffer 的物理布局
- kernel reader/writer 访问布局

三者相同。

这与 TT-Metal 正式模型不一致。官方 `matmul_single_core` 明确要求：

- host 输入先 `tilize`
- device 侧按 tiled layout 计算
- host 输出再 `untilize`

因此当前至少缺少：

- logical tensor layout schema
- host/device layout conversion responsibility
- output packed layout -> final tensor layout 的回写协议

### 3.2 Tensor logical dtype / CB packed dtype / accumulator dtype 被混成一层

当前 Blackhole 主线默认把：

- Tensor dtype
- CB data format
- writer 输出格式

视为同一层。

但 TT-Metal matmul 正式模型里至少分成：

- compute accumulator/result dtype
- CB packed transport dtype
- host-visible tensor dtype

当前 `LowerBlackholeOps` 直接按 TileLang `C` tensor dtype 规划 output CB，说明这层分离尚未建立。

### 3.3 accessor schema 曾经没有进入 ExecutableSpec，现已对主路径 formalize

TT-Metal 的 dataflow reader/writer 主路径广泛依赖：

- `TensorAccessorArgs(...)`
- `TensorAccessor(...)`
- compile-time accessor offsets
- common runtime args

当前主路径已经补上：

- `accessors`
- `common_runtime_args`
- `compile_time_arg_offset/count`
- `common_runtime_arg_offset/count`
- `layout / memory_space / args_config_bits`

并已从 `segment_plan` 进入 `ExecutableSpec / KernelSpec`，由 host `CreateKernel` 正式 materialize 当前 interleaved accessor case。

当前剩余缺口是：

- 更丰富的 accessor execution surface 仍未开放
- `layout != interleaved` 或 `common_runtime_arg_count > 0` 仍是 direct runtime fail-fast 面
- sharded / non-tile / richer CRTA 仍未进入正式执行面

### 3.4 runtime work description ABI 太薄

Blackhole runtime/work schema 在 bring-up 阶段最初基本只覆盖：

- `current_work_linear_id`
- `tile_count`
- 简单 input/output buffer 地址

当前状态已部分收正：

- copy 已切到显式 `input/output_tile_start_id` + `input/output_num_tiles`
- GEMM segment 已切到显式 `input/output/k_tile_start_id` + 对应 tile count
- 但 accessor-derived common runtime args、stride/range/batch 等 richer work descriptors 仍未正式化

而 TT-Metal 官方 dataflow 程序普遍需要更正式的 work ABI，例如：

- `start_tile_id`
- `num_tiles`
- `output_tile_start_id`
- `num_output_tiles`
- block stride / next-block stride
- start dimension offsets
- barrier batch size

当前 `CorePlan + runtime_args` 还不足以表达这些工作描述。

### 3.5 compile-time ABI 曾经没有正式化，现已对主路径部分 formalize

TT-Metal kernel contract 不只是一组匿名 compile-time ints。

实际还包括：

- matmul `Mt/Kt/Nt`
- transpose 位
- reader/writer accessor offsets
- named compile args
- optional `defines`
- data-movement processor / NOC 选择
- semaphore ids

当前主路径已经补上：

- `compile_time_arg_specs`
- `launch_spec`
- accessor CTA compile-time ABI
- GEMM `Mt/Kt/Nt`
- GEMM `transpose_A/B`

并已由 `BlackholeModule` 作为 `CreateKernel` 的正式输入消费。

当前仍未 formalize 的部分包括：

- 更宽的 compute kernel compile-time ABI
- `defines`
- semaphore / multicast / sharded accessor materialization

### 3.6 CB model 还是 allocator 级别，不是 transport-object model

当前 `CBConfig` 只表达：

- `cb_id`
- `role`
- `num_pages`
- `page_size`
- `data_format`

但 TT-Metal `CircularBufferConfig` 实际可表达的更多：

- 同一个 config 挂多个 buffer index
- per-index data format
- per-index page size
- local/remote buffer index
- globally allocated backing address
- tile dims
- dynamic CB

因此当前 `PlanBlackholeCB` 仍更接近“最小 L1 allocator”，不是正式的 TT-Metal CB transport schema。

### 3.7 copy path 不是完整，只是 case 窄

当前 copy direct-path 通过的，是最简单的 tile dataflow case。

TT-Metal 例子里还广泛存在：

- row-major / stick reader-writer
- padding / alignment copy
- sharded copy
- untilize / tilize data movement
- local L1 manual copy / CB-backed scratch manipulation

而我们当前 builtin 只有：

- `read_tile_to_cb`
- `write_tile_from_cb`

这不足以表达 TT-Metal 的更大 dataflow 面。

### 3.8 synchronization / core-to-core transport 没有语义层

TT-Metal 还存在：

- `CreateSemaphore`
- `get_semaphore`
- `noc_semaphore_inc/wait`
- multicast
- remote core addressing
- core-to-core L1 forwarding

当前状态已从“完全没有这些对象”推进到：

- `ExecutableSpec.semaphores`
- `KernelSpec.semaphore_bindings`
- runtime arg kind `semaphore_id_u32`
- 最小 device-side dataflow semaphore builtin：
  - `get_semaphore`
  - `semaphore_wait`
  - `semaphore_set`

但 multicast、global semaphore、remote core coordinate、以及 producer/consumer E2E 执行面仍未建立，因此未来 multi-core 或 ring/mcast pipeline 仍不能只靠补 codegen 自动长出来。

---

## 4. 对当前 Blackhole struct/schema 的直接含义

### 4.1 `ExecutableSpec`

当前仍缺：

- host logical layout
- device physical layout
- accessor descriptors
- transport dtype vs tensor dtype 分层
- richer semaphore / multicast descriptors（program-local worker semaphore 已补）
- richer work descriptors

### 4.2 `CBConfig`

当前仍缺：

- multi-index config
- local/remote index
- globally allocated backing address
- tile dims
- dynamic CB metadata

### 4.3 `KernelArgSpec`

当前缺：

- accessor-derived common runtime args
- tile/stick/block range descriptors
- remote-core / multicast arguments（program-local semaphore id runtime arg 已补）
- output transport vs final writeback distinction

### 4.4 builtin 层

当前仍缺：

- transpose-aware GEMM builtin contract
- untilize/tilize contract
- non-tile/stick dataflow builtin
- multicast/core-to-core builtin（最小 semaphore builtin 已补）
- local CB-backed L1 manipulation contract

---

## 5. 当前 blocker 的收正

### 5.1 精确描述

> **2026-03-26 实施修正**：本轮先后排除了两个错误根因：
> 1. reader/writer 没对接真实 CB backing store
> 2. reader/writer 丢了 CB 同步原语
>
> 实际落实后确认：
> - `read_tile_to_cb/write_tile_from_cb` 已走真实 CB backing store
> - lowered TIR 中 reader/writer 周围也已有正确的 reserve/push/wait/pop
>
> 当前 `test_blackhole_gemm_basic` 错结果的真实根因是：
> - `transpose_B` 语义没有进入 runtime/spec
> - `BlackholeModule` 直接上传 row-major host tensor，没有按 TT-Metal matmul contract 做 tilize/untilize

> **2026-03-26 审查修正**：原描述为”reader/writer 还没对接真实 CB backing store”。
> 实际查看 `codegen_blackhole.cc:789-821`，`PrintReadTileToCB` 已使用 `get_write_ptr(cb_id)`
> （真实 CB 地址），`PrintWriteTileFromCB` 已使用 `get_read_ptr(cb_id)`。
>
> 真正落地后的 blocker 是 **host layout / transpose contract 缺失**。

### 5.2 最短 E2E 正确性路径 vs 完整 contract 收正

本审计文档罗列的 7 个 contract 缺口是”最终要做什么”。但在”当前该做什么”层面必须区分：

| 问题 | 对当前 GEMM basic 结果的影响 | 何时做 |
|------|---------------------------|--------|
| `transpose_B` 处理 | **直接影响当前 GEMM basic** | 立刻修 |
| host tilize/untilize | **直接影响当前 GEMM basic** | 立刻修 |
| CB 同步原语缺失 | 当前 lowered TIR 中已成立，不是最终根因 | 已排除 |
| output dtype 分层 | 如果 accumulator 和 output 同 dtype，当前不影响 | GEMM basic 通过后再补 |
| accessor schema 正式化 | **零影响**（InterleavedAddrGen 对 interleaved 功能等价） | P3 / Stage 3 |
| work description ABI | **零影响** | 后续 |
| semaphore/multicast | **零影响** | Stage 3 |

**原则**：先让 host/runtime/layout contract 与 current GEMM case 一致，再按 P0-P5 做正式 schema 收正。

### 5.3 背后的完整缺层（不变）

GEMM direct path 当前最靠前的执行断点已经确认是 host/runtime layout contract 缺失，但其背后确实暴露了更大的 TT-Metal contract 缺层：
  - host layout
  - accessor schema
  - transpose/compute ABI
  - packed dtype 分层
  - richer work/runtime schema

这些缺层不影响当前 GEMM basic 的数值正确性，但会影响后续更复杂 case 的可扩展性。

---

## 6. 建议的收正顺序

### 6.1 第一优先级：让当前 Stage 2D E2E 有正确的最小正式 contract

先修当前确定会导致 GEMM basic 数值错误的最短路径：

1. `transpose_B` 正式进入 contract
2. host tilize/untilize 进入 direct path
3. 立即恢复 `test_blackhole_gemm_basic` true E2E 验证
4. 若仍有残余误差，再按 output dtype / tile index / runtime ABI 逐项排查

### 6.2 第二优先级：把 copy/runtime 从 case-specific 收成通用 schema

包括：

- `scratch_l1_buffer_addr32` 死代码确认与移除
- tile range/work range ABI
- buffer/accessor descriptors
- role-level page size 退回 per-buffer/per-accessor schema

### 6.3 第三优先级：为后续 multi-core 留正式对象层

包括：

- semaphore descriptors
- remote/core-to-core transport descriptors
- richer CB object model

---

## 7. Stage 2D 任务拆分与优先级

### P0: GEMM compute 语义补齐

目标：

- 让当前 single-core GEMM direct-path 至少具备正确的 compute contract

需要补的语义：

1. `transpose_B` 正式进入 builtin / lowering / codegen
2. GEMM output 的 accumulator dtype / packed CB dtype / final tensor dtype 分层
3. compute kernel compile-time ABI 不再只是一组匿名 ints，至少显式承载 `Mt/Kt/Nt` 与 transpose

为什么优先：

- 当前 `test_blackhole_gemm_basic` 的数值错误里，这层是最直接影响结果正确性的语义
- 即使 CB data path 修正，如果 transpose 和 output format contract 仍缺失，结果依然不可信

TT-Metal 参考 case：

- `tt_metal_repo/tt_metal/hw/inc/api/compute/matmul.h`
- `tt_metal_repo/tt_metal/programming_examples/matmul/matmul_single_core/matmul_single_core.cpp`
- `tt_metal_repo/tt_metal/programming_examples/tilelang_gemm_test/ttsim_gemm_host.cpp`

### P1: CB transport / tile dataflow 语义补齐

目标：

- 让 reader / compute / writer 三段通过真实 CB backing store 共享 tile 数据面

需要补的语义：

1. `read_tile_to_cb/write_tile_from_cb` 改成真实 CB transport
2. `scratch_l1_buffer_addr32` 从主协议降级
3. output transport 与最终 writeback 的责任边界明确

为什么排第二：

- 这是当前 direct-path 最显性的执行断点
- 但它依赖 P0 的 output format contract，否则 transport 写对了，数据解释仍可能错

TT-Metal 参考 case：

- `tt_metal_repo/tt_metal/programming_examples/matmul/matmul_single_core/kernels/dataflow/reader_single_core_mm.cpp`
- `tt_metal_repo/tt_metal/programming_examples/matmul/matmul_single_core/kernels/dataflow/writer_single_core_mm.cpp`
- `tt_metal_repo/tt_metal/programming_examples/loopback/loopback.cpp`
- `tt_metal_repo/tt_metal/programming_examples/vecadd_multi_core/vecadd_multi_core.cpp`

### P2: host layout / tilize-untilize 语义补齐

目标：

- 把 host visible tensor layout 与 device physical tiled layout 正式分层

需要补的语义：

1. host tensor logical layout schema
2. tilize / untilize responsibility 进入 direct path
3. packed output -> final tensor writeback contract

为什么排第三：

- 这层决定 direct path 是否真的能覆盖 TT-Metal 正式 matmul 路径
- 目前 GEMM 之所以还能跑到“有结果”，是因为 host/device layout 错层还没有在最早阶段直接炸掉，不代表语义正确

TT-Metal 参考 case：

- `tt_metal_repo/tt_metal/programming_examples/matmul/matmul_single_core/matmul_single_core.cpp`
- `tt_metal_repo/docs/source/tt-metalium/tt_metal/examples/matmul_single_core.rst`
- `tt_metal_repo/ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/untilize_with_unpadding_device_operation.hpp`
- `tt_metal_repo/ttnn/cpp/ttnn/operations/sliding_window/halo/device/untilize_with_halo_program_factory.cpp`

### P3: accessor / runtime work schema 补齐

目标：

- 把 `TensorAccessorArgs` 和 richer work packet 正式放进 `ExecutableSpec`

需要补的语义：

1. accessor descriptors 进入 spec
2. per-buffer/per-accessor page size 和 layout 元信息进入 spec
3. runtime work description 从 `tile_count` 升级为 `start_id/range/stride/barrier batch`

为什么排第四：

- 这是让 copy/gemm 不再依赖 codegen/runtime 猜测的关键层
- 它同时是后续 row-major/stick/sharded 路径的必要前置

当前进展：

- copy 和 GEMM 的 runtime work schema 已经从 `current_work_linear_id` / `tile_count` 迁移到显式 work descriptor 形态
- copy 侧已经收正为 `work_linear_id` + `a_tile_*` + `output_tile_*`，其中 direct runtime/codegen 当前只正式支持 equal-range + stride=1
- GEMM segment 侧已经收正为 reader 的 `work_linear_id + a_tile_* + b_tile_* + k_tile_*`、compute 的 `k_tile_*`、writer 的 `work_linear_id + output_tile_*`
- codegen/runtime 对缺失 `work_linear_id` 或超出当前支持面的 richer 组合已改为 fail-fast，而不是再从单值默认静默猜测
- 当前又进一步补上了 accessor descriptors：copy fused_dataflow 与 GEMM reader/writer 已把 `buffer + compile_time_arg_offset/count + common_runtime_arg_offset/count + args_config_bits + layout + memory_space` 进入 segment/kernel schema，并由 host `CreateKernel` 正式 materialize当前 interleaved case 的 `TensorAccessorArgs`
- accessor-derived `common_runtime_args` 也已进入 segment/kernel schema；当前 direct runtime 已开始正式 materialize kernel-level shared `common_runtime_args`（buffer-address / semaphore kinds），但对 `layout != interleaved`、`common_runtime_arg_count > 0`、以及 accessor-derived CRTA 执行面仍显式 fail-fast
- 以上主路径 formalization 现已收口完成：current copy/GEMM formal surface 上，work descriptor、accessor/common-runtime schema、compile-time ABI、shared common-runtime host materialization、以及 accessor `args_config_bits` 真源关系都已正式进入主协议
- 剩余项已重新归类：
  - per-accessor page-size/layout 泛化 -> P4
  - accessor CRTA / sharded execution surface -> P4 或后续专项
  - 更宽的 range/stride/batch work execution surface -> 后续按 producer 需求单列，不再挂在本轮 P3 名下

TT-Metal 参考 case：

- `tt_metal_repo/tt_metal/api/tt-metalium/tensor_accessor_args.hpp`
- `tt_metal_repo/tt_metal/programming_examples/pad_multi_core/pad_multi_core.cpp`
- `tt_metal_repo/tt_metal/programming_examples/shard_data_rm/kernels/reader_sharded_rm.cpp`
- `tt_metal_repo/tt_metal/programming_examples/matmul/matmul_multi_core/kernels/dataflow/reader_mm_output_tiles_partitioned.cpp`
- `tt_metal_repo/tt_metal/programming_examples/matmul/matmul_multi_core/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

### P4: copy/dataflow 泛化语义补齐

目标：

- 让 Stage 2D 之后的 copy 不再只覆盖 tile/interleaved 窄 case

需要补的语义：

1. non-tile / stick / row-major dataflow builtin 或等价 schema
2. padding / alignment / local L1 manual copy contract
3. sharded/distributed buffer contract

为什么排第五：

- 这层不是当前 GEMM true E2E 的最早 blocker
- 但如果不补，copy 仍然只是“样例级通过”，不是协议闭环

TT-Metal 参考 case：

- `tt_metal_repo/tt_metal/programming_examples/pad_multi_core/pad_multi_core.cpp`
- `tt_metal_repo/tt_metal/programming_examples/pad_multi_core/kernels/pad_reader_dims_rm_interleaved.cpp`
- `tt_metal_repo/tt_metal/programming_examples/shard_data_rm/kernels/reader_sharded_rm.cpp`
- `tt_metal_repo/tt_metal/programming_examples/distributed/3_distributed_eltwise_add/distributed_eltwise_add.cpp`

### P5: multi-core synchronization / multicast 语义预埋

目标：

- 为 Stage 2D 之后的 multi-core 留正式对象层，而不是到时再补旁路
- 先立住 semaphore 的 host/runtime/device 最小主链，再继续扩 multicast 和更完整同步执行面

当前已补的语义：

1. program-local `semaphore_plan` / semaphore descriptors
2. kernel-level `semaphore_bindings`
3. runtime arg materialization：`semaphore_id_u32`
4. 最小 device-side dataflow semaphore builtin：`get_semaphore` / `semaphore_wait` / `semaphore_set`

仍需补的语义：

1. remote core coordinate / multicast descriptors
2. global semaphore object model
3. 更完整的 semaphore opcode 家族与 producer/consumer E2E
4. core-to-core transport / multicast execution plan

为什么排最后：

- 当前 single-core GEMM 不直接依赖这层
- 但 TT-Metal 多核主路径明确依赖它，后续不能再把它拖给 codegen/runtime 猜

TT-Metal 参考 case：

- `tt_metal_repo/tt_metal/programming_examples/contributed/multicast/multicast.cpp`
- `tt_metal_repo/tt_metal/programming_examples/contributed/multicast/kernels/dataflow/inbound_kernel.cpp`
- `tt_metal_repo/docs/source/tt-metalium/tt_metal/labs/matmul/lab3/lab3.rst`
- `tt_metal_repo/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp`

---

## 8. 执行顺序（实际落地记录）

> **2026-03-26 实际结论**：CB 同步原语不是根因（lowered TIR 中已有）。
> 真实根因是 `transpose_B` 丢失 + host row-major upload 无 tilize/untilize。

已完成：

1. ✅ 复核 generated kernel source 与 lowered TIR — 确认 CB 同步已成立
2. ✅ 定位真实根因 — `transpose_B` + host tilize/untilize
3. ✅ 新增 `blackhole.gemm_contract`（Mt/Kt/Nt/transpose_B）
4. ✅ `BlackholeModule` host-side transpose/tilize/untilize
5. ✅ `scratch_l1_buffer_addr32` 全链路移除
6. ✅ `tilelang_gemm_test` 删除
7. ✅ Copy codegen 统一回 `EmitKernelSourceForPrimFunc`（不再有 scratch fallback source）
8. ✅ `GetRuntimeArgVarForBuffer` preferred_kind 重构

未完成（协议质量，不阻塞 Stage 3）：

- P0: GEMM compile-time ABI 正式化（dtype 分层 + 更丰富 compute ABI）— `blackhole.gemm_contract` 已携带核心字段，`Mt/Kt/Nt/transpose_A/B` 已进入 `compile_time_arg_specs`，更丰富 ABI 可后续推进
- P1: CB transport schema — 已统一到 codegen CB transport，无 scratch
- P3: accessor / runtime work schema（✅ current formal surface 已完成；更宽 accessor/CRTA/non-tile execution surface 已转移到 P4 或后续专项）
- P4: copy/dataflow 泛化
- P5: multi-core synchronization 预埋 → semaphore schema / kernel binding / 最小 device builtin 已补；更完整 multicast / synchronization execution surface 仍待后续推进，见 `stage4_semaphore_schema.md`

---

## 9. 验证方式

这份审计文档本身不以“pytest 全过”为验收标准；它的验收是：

- 当前缺层已经有文档化收束，不再停留在口头判断
- `tasks/progress.md` 的 blocker 描述与本结论一致
- 后续实现按这里列出的 contract 层推进，而不是继续点修 workaround

真正的功能验收仍以：

- `test_blackhole_copy_runtime.py`
- `test_blackhole_gemm.py`
- 之后的更复杂 layout/accessor case

为准。
