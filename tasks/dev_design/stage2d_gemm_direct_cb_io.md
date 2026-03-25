# Stage 2D 设计补充：GEMM direct-path 的真实 CB dataflow IO

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

让 GEMM 的 reader / compute / writer 三段在 direct path 下通过 **真实 TT-Metal CB backing store** 完成 tile 输入输出，而不是通过 scratch L1 临时地址模拟。

注意：

- 本文档只覆盖 Stage 2D 中的 `P1: CB transport / tile dataflow` 子任务
- 更完整的 Stage 2D 拆分、优先级和 TT-Metal 参考 case 见 `tasks/dev_design/stage2d_ttmetal_contract_audit.md`

完成标准：

- `test_blackhole_gemm_basic` 在 TT-Sim direct path 下通过
- 结果与 PyTorch 参考在当前容差内一致

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

---

## 4. 影响范围

- `tilelang_repo/src/target/codegen_blackhole.cc`
- `tilelang_repo/src/target/blackhole_module.cc`
- 可能涉及 `rt_mod_blackhole.cc` 的 runtime arg schema 收缩
- 测试：
  - `testing/python/target/blackhole/test_blackhole_copy_runtime.py`
  - `testing/python/target/blackhole/test_blackhole_gemm.py`

---

## 5. 协议/接口变化

### 5.1 `read_tile_to_cb` / `write_tile_from_cb` 的 codegen 语义要收正

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

### 5.2 `scratch_l1_buffer_addr32` 应降级

一旦读写真正走 CB backing store：

- copy/GEMM 主路径不应再依赖 `scratch_l1_buffer_addr32`
- 它若保留，也只能是过渡字段，不应再承担正式 tile data path

---

## 6. 实施顺序

1. 先以 `stage2d_ttmetal_contract_audit.md` 中的 `P0` 结果收正 GEMM compute contract，避免 transport 先落地后仍被 transpose/output format 语义拖回
2. 对齐 TT-Metal 官方 reader/writer 示例，明确 Blackhole 下 tile read/write 的正式 API 组合
3. 收正 `CodeGenBlackhole::PrintReadTileToCB` / `PrintWriteTileFromCB`
4. 收缩 `scratch_l1_buffer_addr32` 在 runtime schema 里的职责
5. 跑 `test_blackhole_copy_runtime.py`
6. 再跑 `test_blackhole_gemm_basic`

对应 TT-Metal 参考 case：

- `tt_metal_repo/tt_metal/programming_examples/matmul/matmul_single_core/kernels/dataflow/reader_single_core_mm.cpp`
- `tt_metal_repo/tt_metal/programming_examples/matmul/matmul_single_core/kernels/dataflow/writer_single_core_mm.cpp`
- `tt_metal_repo/tt_metal/programming_examples/loopback/loopback.cpp`
- `tt_metal_repo/tt_metal/programming_examples/vecadd_multi_core/vecadd_multi_core.cpp`

---

## 7. 验证方式

- `pytest -q testing/python/target/blackhole/test_blackhole_copy_runtime.py`
- `pytest -q testing/python/target/blackhole/test_blackhole_gemm.py`
- 重点验：
  - GEMM 不再挂在 enqueue
  - GEMM 结果与参考一致
  - copy direct runtime 不回退
