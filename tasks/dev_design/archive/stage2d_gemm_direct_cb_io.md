# Stage 2D 设计补充：GEMM direct-path 的最小正式 contract（P0 + P1）

## 基本信息

- **文档ID**: `stage2d_gemm_direct_cb_io`
- **日期**: 2026-03-26
- **状态**: 已完成
- **对应阶段**: Stage 2D Step 6 true E2E 验收
- **前置上下文**:
  - `stage2d_cb_identity_protocol` 已实施完成
  - GEMM direct path 已从“CB identity 错位 / enqueue deadlock”推进到“真执行完成但结果错误”

---

## 1. 目标

> **2026-03-26 审查修正**：原目标把 schema 正式化和 CB 同步修复混为同一优先级。
> 后续实现复核确认：CB 同步并不是最终根因。lowered TIR 中 reader/writer 周围已经有
> `cb_reserve_back/cb_push_back/cb_wait_front/cb_pop_front`。当前 direct-path GEMM 错结果的真实根因是：
> `transpose_B` 语义未进入 runtime/spec，且 `BlackholeModule` 直接上传 row-major tensor，
> 没有按 TT-Metal matmul contract 做 host-side tilize / untilize。

把当前 Stage 2D Step 6 拆分为两个阶段：

**阶段 A（正确性）**：修复 CB 同步协议，让 GEMM 数值跑对
- `PrintReadTileToCB` 加 `cb_reserve_back/cb_push_back`
- `PrintWriteTileFromCB` 加 `cb_wait_front/cb_pop_front`
- 验证后排查残余数值问题（transpose、dtype、tile index）

**阶段 B（协议质量）**：schema 收正和清理
- GEMM compile-time ABI 正式化（Mt/Kt/Nt/transpose_B/dtype 分层）
- `scratch_l1_buffer_addr32` 退出主协议
- 删除 `tilelang_gemm_test`

完成标准：

- `test_blackhole_gemm_basic` 在 TT-Sim direct path 下通过
- 结果与 PyTorch 参考在当前容差内一致
- `test_blackhole_copy_runtime.py` 不回退
- 更完整的 Stage 2D 拆分、优先级和 TT-Metal 参考 case 见 `tasks/dev_design/stage2d_ttmetal_contract_audit.md`

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

### 3.1 reader/writer 已使用真实 CB backing store 地址，CB 同步不是最终根因

> **2026-03-26 复核修正**：原分析先后误判为：
> 1. codegen 用 scratch L1 伪装 CB data
> 2. codegen 丢了 CB 同步原语
>
> 实际查看 lowered TIR 与 generated source 后确认：
> - `PrintReadTileToCB` / `PrintWriteTileFromCB` 已使用真实 CB backing store 地址
> - reader/writer 周围也已有 `cb_reserve_back/cb_push_back/cb_wait_front/cb_pop_front`
>
> 因此 CB 同步问题不是最终根因。
>
> 真实根因转而落在 host/layout contract。

实际查看
> `codegen_blackhole.cc:789-821`，`PrintReadTileToCB` 已使用 `get_write_ptr(cb_id)` 获取
> 真实 CB backing store 地址，`PrintWriteTileFromCB` 已使用 `get_read_ptr(cb_id)`。

### 3.2 当前真实根因：`transpose_B` 和 host tilize / untilize contract 缺失

对照 TT-Metal 官方 `matmul_single_core`：

- host 输入必须先 `tilize_nfaces`
- B 若以 `N x K` 形态提供且 `transpose_B=True`，则 host/runtime 必须先转成 `K x N`
- host 输出必须在 readback 后 `untilize_nfaces`

当前 Blackhole direct path 在修复前的真实行为是：

- `transpose_B` 没有正式进入 runtime/spec
- `BlackholeModule` 直接把 host row-major tensor 原样 memcpy 到 DRAM buffer
- output 也原样 memcpy 回 host tensor

因此，即使 reader/compute/writer 的 CB data path 与同步都成立，compute 看到的 A/B tile 语义和
host 侧期望的矩阵语义仍然不一致。

### 3.3 `scratch_l1_buffer` 是 copy 路径的历史遗留，对 GEMM 可能是死代码

`blackhole_module.cc` 中仍有 `scratch_l1_buffer_addr32` 的分配和传递逻辑（lines 199-201, 435-456）。
但由于 codegen 已经在用 `get_write_ptr/get_read_ptr`（即 CB 真实地址），scratch buffer 在 GEMM 3-kernel
路径中可能根本没被使用。

需要确认：
- copy 路径是否仍然依赖 scratch（如果 codegen 对 copy 也走 `get_write_ptr`，则 scratch 是全局死代码）
- 删除 scratch 前需先跑 copy runtime tests 回归确认

### 3.4 compute contract 偏弱，但需区分”影响当前正确性”和”后续协议质量”

对照 TT-Metal 官方 `matmul_single_core` 主路径，当前 Blackhole 仍缺以下正式字段：

- GEMM compile-time ABI：`Mt`、`Kt`、`Nt`、`transpose_B`
- output dtype 分层：accumulator dtype、CB transport packed dtype、final tensor dtype
- 最小 accessor contract：reader/writer 访问 DRAM tile 的 compile-time descriptor

但需要区分两个层次：

**直接影响当前 GEMM basic 数值正确性的**：
- CB 同步原语缺失 — **确定影响**，必须先修
- transpose_B — 取决于测试是否需要 transpose，需验证
- output dtype 分层 — 如果 accumulator 和 output 是同一 dtype，当前可能不影响

**属于协议质量提升但不直接影响当前正确性的**：
- accessor schema 正式化 — 当前 `InterleavedAddrGen` 对 interleaved 场景功能等价
- Mt/Kt/Nt 进入 compile-time ABI — 当前已通过 runtime args 传递，只是没正式化

不应在修复同步 bug 之前做大量 schema 工程。

### 3.5 `tilelang_gemm_test` 是 bring-up 样例，不应继续充当主线参考

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

### 5.1 host transpose / tilize / untilize contract 修复（当前已落地）

本轮已补的最小正确性 contract：

- `LowerBlackholeOps` 产出 `blackhole.gemm_contract`
- `rt_mod_blackhole` 将该 contract 进入 `ExecutableSpec`
- `BlackholeModule` 在 direct path 下：
  - A: row-major → tilize
  - B: row-major `N x K` + `transpose_B=True` → transpose → tilize
  - C: tiled output → untilize → row-major tensor

这一步是当前 `test_blackhole_gemm_basic` 数值通过的直接原因。

### 5.2 数值验证后按需补齐的 compute contract

在 CB 同步修复后，如果数值仍有残余错误，按以下顺序逐项排查：

1. `transpose_B` — 确认测试是否需要、codegen 是否已处理
2. output dtype 分层 — 确认 accumulator 与 output CB 是否 dtype 一致
3. tile 索引公式 — 确认 A/B/C 的 linear index 是否与参考一致

### 5.3 `scratch_l1_buffer_addr32` 已退出主协议

本轮实际落地：

- copy `fused_dataflow` 单 kernel 不再走 `rt_mod_blackhole` 的 scratch fallback source
- copy 与 GEMM 都统一回到 codegen 生成的 CB transport
- copy/GEMM 主路径的 runtime arg schema 已移除 `scratch_l1_buffer_addr32`
- `BlackholeModule` 不再为主路径分配、下发或解释 scratch L1 buffer

额外修正：

- `fused_dataflow` 单 segment 必须继承原函数的 `blackhole.runtime_args`
- `KernelSpec` 也必须继承该 runtime arg schema，不能只让 segment source 继承
- codegen 对 `read_tile_to_cb` / `write_tile_from_cb` 的 buffer 绑定，优先按 `buffer` 名，恢复失败时按 `input_buffer_addr*` / `output_buffer_addr*` 角色回退

### 5.4 GEMM schema 正式化（协议质量，排在正确性之后）

在数值正确之后，作为协议质量提升：

- `Mt/Kt/Nt/transpose_B` 进入 split 后 device kernel attrs
- output dtype 分层（`accumulator_dtype` / `transport_dtype` / `final_tensor_dtype`）
- host logical layout 与 device tiled layout 的责任边界在 spec 中记录

约束：不在修正确性 bug 的同一步做这些 schema 工程

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

## 6. 实施顺序（2026-03-26 审查修正）

> **原则**：先让 GEMM 数值跑对（最小改动验证），再做协议正式化（schema 质量提升）。
> 不要用 schema 工程来解决同步 bug。

1. **导出当前 GEMM 3-kernel 生成代码**，与 TT-Metal `matmul_single_core` 参考逐行对比，确认问题不在 CB 同步
2. **补 `blackhole.gemm_contract`**，让 `transpose_B` 与 GEMM 维度进入 runtime/spec
3. **补 host-side transpose / tilize / untilize**
4. **移除 `scratch_l1` 主路径依赖**（已完成）
5. **删除 `tilelang_gemm_test`**（低风险清理，已完成）
6. **Schema 收正**（Mt/Kt/Nt/transpose_B/dtype 分层）— 作为协议质量改进
7. **Accessor schema** — 留给 P3/Stage 3

对应 TT-Metal 参考 case：

- `tt_metal_repo/tt_metal/programming_examples/matmul/matmul_single_core/kernels/dataflow/reader_single_core_mm.cpp`
- `tt_metal_repo/tt_metal/programming_examples/matmul/matmul_single_core/kernels/dataflow/writer_single_core_mm.cpp`
- `tt_metal_repo/tt_metal/programming_examples/matmul/matmul_single_core/kernels/compute/mm.cpp`
- `tt_metal_repo/tt_metal/programming_examples/matmul/matmul_single_core/matmul_single_core.cpp`

---

## 7. 验证方式

- `pytest -q testing/python/target/blackhole/test_blackhole_copy_pipeline.py`
- `pytest -q testing/python/target/blackhole/test_blackhole_copy_runtime.py`
- `pytest -q testing/python/target/blackhole/test_blackhole_gemm.py`

### 7.1 阶段 A（正确性）验证

- `test_blackhole_gemm_basic` 不再挂在 enqueue
- `test_blackhole_gemm_basic` 数值与参考一致
- `test_blackhole_copy_runtime.py` 不回退
- `blackhole.gemm_contract` 已进入 split 后 attrs / `ExecutableSpec`
- direct path 已对 GEMM 输入做 host-side tilize / transpose，对输出做 untilize

### 7.2 阶段 B（协议质量）验证

- copy/GEMM 主 schema 中不再出现 `scratch_l1_buffer_addr32`
- `tilelang_gemm_test` 已删除，且无主线引用残留
- split 后 attrs / `ExecutableSpec` 中已带：
  - GEMM compile-time ABI
  - dtype 分层
  - host/device layout 责任边界记录
