# Stage 4 设计：最小 Stick Copy 泛化

## 基本信息

- **文档ID**: `stage4_copy_stick_generalization`
- **日期**: 2026-03-30
- **状态**: ✅ 已实施（最小 interleaved stick/page path）
- **对应任务**: P4 copy/dataflow 泛化
- **关联文档**:
  - `tasks/dev_design/final_blackhole_backend_redesign.md`
  - `tasks/dev_design/stage2d_ttmetal_contract_audit.md`
  - `tasks/progress.md`

---

## 1. 目标

把 Blackhole copy 正式支持面从“tile-aligned interleaved tile copy”扩一小步到：

- interleaved DRAM accessor
- single-kernel `fused_dataflow`
- row-major / stick-style contiguous page transport
- 当前正式支持目标：`M x W`，其中 `M` 是 32 的倍数、`W` 不要求是 32 的倍数

本轮目标不是做完整 non-tile/sharded 泛化，而是先建立一个正式 page/stick transport 主路径，让 `32x16` 这类最小 non-tile copy 能进入 direct runtime。

---

## 2. 边界

本轮只做：

- copy，不碰 GEMM
- interleaved accessor，不碰 sharded
- DRAM <-> CB page transport
- static shape
- `shared_rows % 32 == 0`

本轮不做：

- sharded / banked accessor
- common runtime args
- arbitrary row counts或更宽 batch/range 语义
- tilize/untilize 或 layout conversion

---

## 3. 协议方案

在现有 tile transport builtin 之外，新增最小 page/stick transport builtin：

- `tl.blackhole.read_page_to_cb(buffer, page_id, cb_id, page_bytes, accessor_slot, cb_offset_bytes)`
- `tl.blackhole.write_page_from_cb(cb_id, buffer, page_id, page_bytes, accessor_slot, cb_offset_bytes)`

语义：

- page 不是 tile
- page 大小由 `page_bytes` 显式给出
- page 的寻址仍走现有 interleaved `TensorAccessorArgs + TensorAccessor`
- page 粒度的 DRAM 地址通过 `TensorAccessor(...).get_noc_addr(page_id)` 求出
- 读写仍然经过当前 CB / runtime / accessor 主链，不引入第二条执行路径

这意味着：

- `CBRequirement.page_size` 不再默认等于 tile bytes
- 对于外部 interleaved accessor，`transport_page_size` 必须显式进入 accessor schema，并驱动 runtime buffer page-size materialization
- stick transport 的单 stick 字节数通过 accessor `transport_page_size` 表达；当前最小实现的共享 CB 仍保持 `2048 x 1` 的单页布局，stick 通过 `cb_offset_bytes` 落到同一 shared page 内不同偏移

---

## 4. lowering 方案

当前 `LowerBlackholeOps` 在 staged copy 路径里硬编码：

- shared height 必须是 32 的倍数
- shared width 必须是 32 的倍数
- global width 必须是 32 的倍数
- base index 用 tile id 推导

本轮调整为双路径：

1. tile path
   - 继续保持当前 `read_tile_to_cb/write_tile_from_cb`
   - 条件：shared width 仍是 32 的倍数

2. stick path
   - 条件：`shared_rows % 32 == 0` 且 shared width 不是 32 的倍数
   - 以“每行一个 page/stick”的方式 materialize copy
   - `page_bytes = shared_cols * dtype.bytes()`
   - `pages_per_row = global_cols / shared_cols`
   - `base_page_id = row * pages_per_row + col / shared_cols`
   - 逐行发出 `read_page_to_cb/write_page_from_cb`
   - `cb_offset_bytes = row * page_bytes`，把 `shared_rows` 个 stick 顺序堆进一个 shared page

stick path 仍要求：

- global width 能被 shared width 整除
- source/destination slice 没有额外 stride 变化
- accessor 继续是 interleaved + DRAM

---

## 5. codegen / runtime 方案

codegen 为新 builtin 打印 TT-Metal 正式 dataflow API：

- `uint64_t noc_addr = TensorAccessor(...).get_noc_addr(page_id)`
- `noc_async_read(noc_addr, l1_addr, page_bytes)`
- `noc_async_write(l1_addr, noc_addr, page_bytes)`

runtime 侧不新增第二条执行路径：

- 继续使用现有 `ExecutableSpec -> KernelSpec -> BlackholeModule`
- accessor 仍走现有 `TensorAccessorArgs` compile-time ABI
- direct runtime 用 accessor `transport_page_size` 创建外部 DRAM buffer，而不是继续回退到 tile-sized `page_size`
- 对 partial-write output buffer，host 侧必须先把当前 tensor 内容同步到 device，再执行 kernel；否则 stick/non-tile copy 未覆盖区域会读回脏数据
- runtime args 仍沿用 `work_linear_id + a_tile_* + output_tile_*` 这组最小 copy work schema

---

## 6. 验证

新增最小 `32x16` 和 `64x16` stick copy case：

- pipeline/spec 测试验证 lowering 不再因为 width 非 32 对齐而失败
- direct runtime 测试验证数值正确
- TT-Sim 下验证同一个 case 真执行通过

回归范围：

- `pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py`
- `pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py`
- `source scripts/setup_tt_sim.sh && pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py`
- `pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`

---

## 7. 完成标准

- `M x W`（`M` 为 32 的倍数）interleaved stick copy 能进入 lowering/spec/runtime 主链
- 不引入新的执行后门或 legacy emitter
- 现有 tile copy / GEMM 回归保持通过
- 文档、进度、经验同步
