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

## 7. 验证方式

这份审计文档本身不以“pytest 全过”为验收标准；它的验收是：

- 当前缺层已经有文档化收束，不再停留在口头判断
- `tasks/progress.md` 的 blocker 描述与本结论一致
- 后续实现按这里列出的 contract 层推进，而不是继续点修 workaround

真正的功能验收仍以：

- `test_blackhole_copy_runtime.py`
- `test_blackhole_gemm.py`
- 之后的更复杂 layout/accessor case

为准。
