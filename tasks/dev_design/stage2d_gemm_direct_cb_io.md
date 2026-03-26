# Stage 2D 设计补充：GEMM direct-path 的最小正式 contract（P0 + P1）

## 基本信息

- **文档ID**: `stage2d_gemm_direct_cb_io`
- **日期**: 2026-03-26
- **状态**: 进行中
- **对应阶段**: Stage 2D Step 6 true E2E 验收
- **前置上下文**:
  - `stage2d_cb_identity_protocol` 已实施完成
  - GEMM direct path 已从“CB identity 错位 / enqueue deadlock”推进到“真执行完成但结果错误”

---

## 1. 目标

把当前 Stage 2D Step 6 的首要任务收敛为一个统一实现任务：

1. `P0`: GEMM compute 语义补齐
2. `P1`: CB transport / tile dataflow 语义补齐

注意：

- 本文档覆盖 `P0 + P1` 的最小正式 contract
- 更完整的 Stage 2D 拆分、优先级和 TT-Metal 参考 case 见 `tasks/dev_design/stage2d_ttmetal_contract_audit.md`

完成标准：

- `test_blackhole_gemm_basic` 在 TT-Sim direct path 下通过
- 结果与 PyTorch 参考在当前容差内一致
- `test_blackhole_copy_runtime.py` 不再依赖 `scratch_l1_buffer_addr32` 作为主协议字段
- 删除 `tt_metal_repo/tt_metal/programming_examples/tilelang_gemm_test/`，避免 bring-up 样例继续充当错误参考实现

---

## 2. 当前现象

当前 direct path 已经能够：

- 完成 `LowerBlackholeOps -> PlanBlackholeCB -> CodeGenBlackhole -> BlackholeModule`
- 正常创建 3 个 segment kernel
- `EnqueueMeshWorkload(blocking=true)` 返回

但 `test_blackhole_gemm_basic` 结果明显错误：

- `max diff=37.24`
- `mean diff=8.91`

---

## 3. 根因分析

### 3.1 当前 copy-style builtin 仍在用 scratch L1 伪装“CB data”

`CodeGenBlackhole` 当前对：

- `tl.blackhole.read_tile_to_cb`
- `tl.blackhole.write_tile_from_cb`

生成的是：

- `cb_reserve_back/cb_push_back` 或 `cb_wait_front/cb_pop_front`
- 再配合 `scratch_l1_addr + cb_head/tail * page_size`
- 用 `noc_async_read/noc_async_write` 直接搬到 scratch buffer

这对 pure copy 路径还能工作，因为：

- reader 和 writer 都只看这块 scratch L1
- CB 只承担逻辑同步角色

但对 GEMM 不成立，因为：

- compute kernel 的 `mm_init/cb_wait_front/matmul_tiles/pack_tile`
- 消费的是 **真实 CB backing store**
- 不是 `scratch_l1_addr` 那块临时地址

结果是：

- reader 往 scratch L1 写
- compute 从真实 CB 读
- writer 又从 scratch L1 读

三段虽然在 `cb_id` 上同步了，但**并没有在同一块真实 tile 数据存储上协作**。

### 3.2 当前 direct path 的 scratch 方案不是 GEMM 可复用协议

这不是某个 tile index 或 runtime arg 小 bug，而是数据面模型不对：

- copy path 当前是“假的 CB data path”
- GEMM 需要“真的 CB data path”

因此不能继续在 `scratch_l1_addr`、tile index、等待顺序上打补丁。

### 3.3 当前 compute contract 仍然偏弱，不足以支撑真实 transport

对照 TT-Metal 官方 `matmul_single_core` 主路径，当前 Blackhole 仍缺至少以下正式字段：

- GEMM compile-time ABI：
  - `Mt`
  - `Kt`
  - `Nt`
  - `transpose_B`
- output dtype 分层：
  - accumulator dtype
  - CB transport packed dtype
  - final tensor dtype
- 最小 accessor contract：
  - reader/writer 访问 DRAM tile 的 compile-time descriptor
  - 不再默认 “buffer 地址 + page_size + interleaved” 就等于正式访问协议

如果这些字段不先显式进入 split 后 schema，那么即使 reader/writer 改成真实 CB backing store，结果仍可能因为：

- B 的 tile 解释方式
- output pack/writeback dtype
- DRAM tile accessor 解释方式

继续错误。

### 3.4 `tilelang_gemm_test` 是 bring-up 样例，不应继续充当主线参考

对照：

- `tt_metal_repo/tt_metal/programming_examples/matmul/matmul_single_core/*`
- `tt_metal_repo/tt_metal/api/tt-metalium/tensor_accessor_args.hpp`

可以确认当前仓库内的：

- `tt_metal_repo/tt_metal/programming_examples/tilelang_gemm_test/`

只体现了早期 bring-up 级最小通路：

- 手工 `InterleavedAddrGen`
- 无 `TensorAccessorArgs`
- 无 host `tilize/untilize`
- runtime/compile-time ABI 过薄

因此它不应继续保留在仓库中充当“当前 TileLang GEMM 正式参考”，否则会持续把：

- 过薄 ABI
- 过渡 scratch/dataflow 思路
- host layout 缺层

重新带回当前设计讨论与实现判断。

---

## 4. 影响范围

- `tilelang_repo/src/transform/lower_blackhole_ops.cc`
- `tilelang_repo/src/transform/split_blackhole_kernel.cc`
- `tilelang_repo/src/target/rt_mod_blackhole.cc`
- `tilelang_repo/src/target/codegen_blackhole.cc`
- `tilelang_repo/src/target/blackhole_module.cc`
- `tilelang_repo/src/target/blackhole_module.h`
- `tasks/progress.md`
- `memory/general_dev.md`
- `memory/bugs.md`
- 删除：
  - `tt_metal_repo/tt_metal/programming_examples/tilelang_gemm_test/`
- 测试：
  - `testing/python/target/blackhole/test_blackhole_copy_pipeline.py`
  - `testing/python/target/blackhole/test_blackhole_copy_runtime.py`
  - `testing/python/target/blackhole/test_blackhole_gemm.py`

---

## 5. 协议/接口变化

### 5.1 GEMM compute contract 要先正式化

在 split 后 device-side attrs / `ExecutableSpec` 中，至少新增并稳定以下信息：

- GEMM compile-time ABI：
  - `Mt`
  - `Kt`
  - `Nt`
  - `transpose_B`
- output dtype 分层：
  - `accumulator_dtype`
  - `transport_dtype`
  - `final_tensor_dtype`
- 最小 accessor descriptor：
  - reader 的输入 A/B tile accessor compile-time descriptor
  - writer 的输出 C tile accessor compile-time descriptor

约束：

- 这些字段必须来自 split 后 device kernel attrs/schema
- 不允许继续由 `CodeGenBlackhole` / `BlackholeModule` 通过参数位置或默认 page size 规则猜
- host logical layout 与 device tiled layout 的责任边界必须在 spec 中明确记录；本轮不要求一次完成完整通用 layout 系统，但不能继续把两者视为天然相同

### 5.2 `read_tile_to_cb` / `write_tile_from_cb` 的 codegen 语义要收正

reader/writer 不再以 `scratch_l1_addr` 为主数据面，而应改成：

- reader:
  - `cb_reserve_back(cb, n)`
  - `get_write_ptr(cb)`
  - `noc_async_read_tile(...)` 或等价 tile reader API
  - `cb_push_back(cb, n)`
- writer:
  - `cb_wait_front(cb, n)`
  - `get_read_ptr(cb)`
  - `noc_async_write_tile(...)` 或等价 tile writer API
  - `cb_pop_front(cb, n)`

### 5.3 `scratch_l1_buffer_addr32` 应退出主协议

一旦读写真正走 CB backing store：

- copy/GEMM 主路径的 runtime arg schema 不应再保留 `scratch_l1_buffer_addr32`
- `BlackholeModule` 不再为主路径分配、下发或解释这类 scratch 参数
- 测试也应同步改成验证“主 schema 中不存在 scratch runtime arg”

### 5.4 `tilelang_gemm_test` 删除，不做兼容迁移

该目录是历史 bring-up 产物，不再是当前主线协议的一部分。

本轮约束：

- 直接删除 `tt_metal_repo/tt_metal/programming_examples/tilelang_gemm_test/`
- 不再让任何文档、测试或实现引用它作为“当前 GEMM 参考 case”
- 后续统一以：
  - `tt_metal_repo/tt_metal/programming_examples/matmul/matmul_single_core/*`
  - `tt_metal_repo/tt_metal/programming_examples/matmul/matmul_common/*`
  - `tt_metal_repo/tt_metal/api/tt-metalium/tensor_accessor_args.hpp`
  作为 TT-Metal 正式参考

---

## 6. 实施顺序

1. 删除 `tilelang_gemm_test`，避免实现过程中继续误对照过渡样例
2. 在 `LowerBlackholeOps` / `SplitBlackholeKernel` / `rt_mod_blackhole` 中补齐 GEMM 最小正式 contract：
   - `Mt/Kt/Nt`
   - `transpose_B`
   - output dtype 分层
   - 最小 accessor descriptor
3. 对齐 TT-Metal 官方 reader/writer 示例，明确 Blackhole 下 tile read/write 的正式 API 组合
4. 收正 `CodeGenBlackhole::PrintReadTileToCB` / `PrintWriteTileFromCB`
5. 收缩 `scratch_l1_buffer_addr32`：从 runtime schema 和 `BlackholeModule` 主路径中移除
6. 跑 copy direct runtime 回归
7. 跑 GEMM direct-path true E2E 回归

对应 TT-Metal 参考 case：

- `tt_metal_repo/tt_metal/programming_examples/matmul/matmul_single_core/kernels/dataflow/reader_single_core_mm.cpp`
- `tt_metal_repo/tt_metal/programming_examples/matmul/matmul_single_core/kernels/dataflow/writer_single_core_mm.cpp`
- `tt_metal_repo/tt_metal/programming_examples/matmul/matmul_single_core/matmul_single_core.cpp`
- `tt_metal_repo/tt_metal/api/tt-metalium/tensor_accessor_args.hpp`
- `tt_metal_repo/tt_metal/programming_examples/loopback/loopback.cpp`
- `tt_metal_repo/tt_metal/programming_examples/vecadd_multi_core/vecadd_multi_core.cpp`

---

## 7. 验证方式

- `pytest -q testing/python/target/blackhole/test_blackhole_copy_pipeline.py`
- `pytest -q testing/python/target/blackhole/test_blackhole_copy_runtime.py`
- `pytest -q testing/python/target/blackhole/test_blackhole_gemm.py`
- 重点验：
  - split 后 attrs / `ExecutableSpec` 中已带 GEMM compile-time ABI、dtype 分层和最小 accessor descriptor
  - copy/GEMM 主 schema 中不再出现 `scratch_l1_buffer_addr32`
  - GEMM 不再挂在 enqueue
  - GEMM 结果与参考一致
  - copy direct runtime 不回退
