# Stage 3 设计补充：GEMM Multi-Core Direct Path Hang

## 基本信息

- **文档ID**: `stage3_gemm_multicore_direct_hang`
- **日期**: 2026-03-26
- **状态**: 设计完成，待实施
- **对应阶段**: Stage 3 multi-core runtime 调度
- **关联文档**:
  - `tasks/dev_design/final_blackhole_backend_redesign.md`
  - `tasks/dev_design/stage3_multicore_design.md`
  - `tasks/dev_design/stage2d_ttmetal_contract_audit.md`

---

## 1. 目标

只修正式 `BlackholeModule` direct host path 上的 multi-core GEMM hang，不把问题扩散到更外层的
`tilelang.compile(..., execution_backend="tvm_ffi")` export/runtime wrapper。

本轮明确目标：

- `lower() + artifact.codegen_mod["main"]` 的 multi-core GEMM 不再 hang
- 当前最小复现（`M=64, N=64, K=128`, `grid_x=2, grid_y=2`）数值通过
- single-core GEMM 和 multi-core copy 不回退

本轮明确不做：

- 不修 `tilelang.compile/tvm_ffi` 的 Blackhole host shim/export bug
- 不做 GEMM ABI / dtype layering formalization
- 不做 oversubscribed multi-core GEMM 支持
- 不新增 runtime fallback 或单独旁路

---

## 2. 当前现象

当前 multi-core GEMM 有两个彼此独立的问题：

1. **更外层 wrapper 问题**
   - `tilelang.compile(..., target="blackhole", execution_backend="tvm_ffi")` 的 runtime/export 路径会生成非法 host shim：
   - `int32_t kernel_error_code = ;`
   - 这一点不是 multi-core GEMM 特有，最小 single-core copy 也能复现。

2. **正式 direct path 问题**
   - 在 `lower() + artifact.codegen_mod["main"]` 路径下，multi-core GEMM 最初先卡在 host tilize 输入尺寸仍按 single-core `gemm_contract` 校验。
   - 把输入/输出 tilize-untilize 改成按 runtime tensor 实际形状后，测试不再在 host shape check 处失败，而是进入设备执行阶段后 hang。

因此当前 direct path 的真实 blocker 已经从：

- host shape 假设错误

推进到：

- multi-core GEMM 在 device execution 阶段存在新的协议/同步问题

---

## 3. 根因判断边界

### 3.1 已排除的方向

以下方向本轮不应继续混入根因判断：

- `tilelang.compile/tvm_ffi` 包装层
- `SaveToBytes` / `WriteToFile` 缺失
- `Executable.export_library()` 导出的 host shim 非法 C

理由：

- 最小 single-core copy 也会触发同样的 wrapper/export 错误
- 这是 Blackhole runtime wrapper 的通用 blocker，不是 Stage 3 multi-core GEMM hang 的特异原因
- 仓库当前正式执行路径仍然是 `BlackholeModule` direct host path

### 3.2 当前只看 direct host path

本轮根因分析只允许沿这条主路径展开：

```text
TileLang DSL GEMM kernel
  -> lower(..., target="blackhole")
  -> artifact.device_mod / artifact.codegen_mod
  -> artifact.codegen_mod["main"]
  -> BlackholeModule::ExecuteDirect()
  -> TT-Metal multi-core Program launch
```

如果问题不能在这条链路上解释，不属于本轮。

---

## 4. 分层排查顺序

### 4.1 Host contract 层

先确认 host 侧输入输出处理已不再带 single-core 假设：

- A/B tilize 必须按 runtime tensor 实际 `rows/cols` 处理整张 tensor
- `transpose_B=True` 必须对整张 B tensor 生效
- C readback / untilize 也必须按 runtime output tensor 实际形状处理

这一层的目标不是“最终 schema 更漂亮”，而是保证 multi-core reader/writer 看到的是完整 tiled tensor，
而不是只为 single-core block 准备的半张/四分之一张 tensor。

### 4.2 Launch / runtime args 层

确认 multi-core GEMM 的 launch 语义和 copy 一致：

- `core_plan.logical_grid_x/y`
- `physical_cores`
- `work_packets`
- per-core `current_work_linear_id`
- reader / compute / writer 三段 kernel registration

重点检查：

- GEMM 3-segment launch 是否共享同一组 launch cores
- per-core runtime args 是否确实与 `work_packets` 一致
- 是否有某个 segment 仍然隐含单核 `work_id=0`

### 4.3 Device execution 层

最后再查 device-side 行为：

- reader 是否按 `bx/by` 派生的 tile index 读正确 tile
- compute 是否仍能在每核本地 CB 上独立完成 `matmul_tiles`
- writer 是否写回每核对应的输出 tile
- CB wait/push/pop/reserve 是否在 multi-core GEMM 下仍满足 producer/consumer 顺序

这里不预设根因一定是 CB，同样也不预设根因一定是 compute tile index。必须以 dump 出来的 lowered TIR、
generated source 和 direct path 执行迹象为准。

---

## 5. 最小复现与验收

### 5.1 最小复现

最小复现固定为：

- `M=64`
- `N=64`
- `K=128`
- `block_M=32`
- `block_N=32`
- `grid_x=2`
- `grid_y=2`
- `transpose_B=True`

DSL kernel 形式固定为：

```python
with T.Kernel(2, 2) as (bx, by):
    T.copy(A[by * 32:(by + 1) * 32, 0:128], A_shared)
    T.copy(B[bx * 32:(bx + 1) * 32, 0:128], B_shared)
    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
    T.copy(C_local, C[by * 32:(by + 1) * 32, bx * 32:(bx + 1) * 32])
```

### 5.2 验收标准

本轮完成标准：

- `test_blackhole_gemm_multicore_direct_call` 不再 hang
- 数值在当前容差内通过
- `test_blackhole_gemm_basic` 不回退
- `test_blackhole_copy_pipeline.py` 不回退
- `test_blackhole_copy_runtime.py` 不回退

### 5.3 非验收项

以下不属于本轮验收：

- `tilelang.compile/tvm_ffi` Blackhole wrapper 是否可用
- oversubscribed GEMM multi-core 是否支持
- richer accessor/schema 是否已经 formalize

---

## 6. 对文档与 bug 记录的要求

如果本轮实施中继续确认：

- `tilelang.compile/tvm_ffi` Blackhole runtime/export bug 是独立 blocker

则必须把它写入：

- `tasks/progress.md`
- `memory/bugs.md`

但该 blocker 只能作为“独立未完成项”记录，不能覆盖 Stage 3 direct host path 的真实进展判断。
