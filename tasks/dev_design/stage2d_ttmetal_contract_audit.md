# Stage 2D 设计补充：TT-Metal contract 缺口审计

## 基本信息

- **文档ID**: `stage2d_ttmetal_contract_audit`
- **日期**: 2026-03-26
- **状态**: 进行中
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

### 3.3 accessor schema 没有进入 ExecutableSpec

TT-Metal 的 dataflow reader/writer 主路径广泛依赖：

- `TensorAccessorArgs(...)`
- `TensorAccessor(...)`
- compile-time accessor offsets
- common runtime args

当前 Blackhole 的 `ExecutableSpec / KernelSpec / KernelArgSpec` 只表达了：

- compile-time `uint32_t` 数组
- runtime arg kind
- buffer 名字

它并没有把 accessor schema 作为一等对象建模。

这意味着 codegen/runtime 仍在“猜 buffer 是如何被访问的”，而不是消费显式 contract。

### 3.4 runtime work description ABI 太薄

当前 Blackhole runtime/work schema 基本只覆盖：

- `current_work_linear_id`
- `tile_count`
- 简单 input/output buffer 地址

而 TT-Metal 官方 dataflow 程序普遍需要更正式的 work ABI，例如：

- `start_tile_id`
- `num_tiles`
- `output_tile_start_id`
- `num_output_tiles`
- block stride / next-block stride
- start dimension offsets
- barrier batch size

当前 `CorePlan + runtime_args` 还不足以表达这些工作描述。

### 3.5 compile-time ABI 也没有正式化

TT-Metal kernel contract 不只是一组匿名 compile-time ints。

实际还包括：

- matmul `Mt/Kt/Nt`
- transpose 位
- reader/writer accessor offsets
- named compile args
- optional `defines`
- data-movement processor / NOC 选择
- semaphore ids

当前 Blackhole 虽然有 `compile_time_args` 和 `core_type`，但还远不到“正式 ABI schema”的程度。

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

当前 Blackhole IR/schema 完全没有这些对象，因此未来 multi-core 或 ring/mcast pipeline 不可能只靠补 codegen 自动长出来。

---

## 4. 对当前 Blackhole struct/schema 的直接含义

### 4.1 `ExecutableSpec`

当前缺：

- host logical layout
- device physical layout
- accessor descriptors
- transport dtype vs tensor dtype 分层
- semaphore descriptors
- richer work descriptors

### 4.2 `CBConfig`

当前缺：

- multi-index config
- local/remote index
- globally allocated backing address
- tile dims
- dynamic CB metadata

### 4.3 `KernelArgSpec`

当前缺：

- accessor-derived common runtime args
- tile/stick/block range descriptors
- semaphore / remote-core arguments
- output transport vs final writeback distinction

### 4.4 builtin 层

当前缺：

- transpose-aware GEMM builtin contract
- untilize/tilize contract
- non-tile/stick dataflow builtin
- semaphore/core-to-core builtin
- local CB-backed L1 manipulation contract

---

## 5. 当前 blocker 的收正

因此，Stage 2D 当前 blocker 不应再表述成单点问题：

- 不是单纯 “`read_tile_to_cb/write_tile_from_cb` 还没对接真实 CB backing store”

更准确的描述应是：

- GEMM direct path 首先卡在真实 CB data path
- 但其背后暴露的是更大的 TT-Metal contract 缺层：
  - host layout
  - accessor schema
  - transpose/compute ABI
  - packed dtype 分层
  - richer work/runtime schema

也就是说，真实 CB IO 只是当前最靠前的显性断点，不是唯一缺口。

---

## 6. 建议的收正顺序

### 6.1 第一优先级：让当前 Stage 2D E2E 有正确的最小正式 contract

先补最小但正式的一组 schema：

1. GEMM transpose contract
2. output packed dtype vs final tensor dtype 分层
3. accessor schema 进入 `ExecutableSpec`
4. `read_tile_to_cb/write_tile_from_cb` 改成真实 CB data path
5. host tilize/untilize responsibility 明确

### 6.2 第二优先级：把 copy/runtime 从 case-specific 收成通用 schema

包括：

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

需要补的语义：

1. semaphore descriptors
2. remote core coordinate / multicast descriptors
3. core-to-core transport builtin 或等价 schema

为什么排最后：

- 当前 single-core GEMM 不直接依赖这层
- 但 TT-Metal 多核主路径明确依赖它，后续不能再把它拖给 codegen/runtime 猜

TT-Metal 参考 case：

- `tt_metal_repo/tt_metal/programming_examples/contributed/multicast/multicast.cpp`
- `tt_metal_repo/tt_metal/programming_examples/contributed/multicast/kernels/dataflow/inbound_kernel.cpp`
- `tt_metal_repo/docs/source/tt-metalium/tt_metal/labs/matmul/lab3/lab3.rst`
- `tt_metal_repo/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp`

---

## 8. 推荐执行顺序

Stage 2D 当前最稳的推进顺序应改成：

1. P0: GEMM compute 语义
2. P1: CB transport 语义
3. P2: host layout / tilize-untilize
4. P3: accessor / runtime work schema
5. 以 `test_blackhole_gemm_basic` 恢复 true E2E
6. P4: copy/dataflow 泛化
7. P5: multi-core synchronization 预埋

这个顺序的原则是：

- 先补当前 GEMM 结果正确性必需的最小 contract
- 再补会影响更多 case 的通用 schema
- 最后为 multi-core 预埋正式对象层

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
