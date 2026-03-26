# TileLang Blackhole 后端开发进度

> 当前唯一总体设计文档: `tasks/dev_design/final_blackhole_backend_redesign.md`

## 当前阶段

- **阶段**: Stage 2D Step 6 — GEMM direct-path E2E 验收（✅ 已完成）
- **日期**: 2026-03-26
- **当前测试结果**：
  - `test_blackhole_copy_pipeline.py`: `16 passed, 1 xfailed`
  - `test_blackhole_copy_runtime.py`（TT-Sim）: `5 passed`
  - `test_blackhole_gemm.py`: `3 passed, 1 skipped`
  - `test_blackhole_gemm_basic`：TT-Sim direct path 数值通过
- **当前结论**：
  - 本轮真实根因不是“CB 同步原语缺失”
  - `LowerBlackholeOps` 生成的 reader/compute/writer TIR 已带正确的 `cb_reserve_back/cb_push_back/cb_wait_front/cb_pop_front`
  - 当前 direct-path GEMM 错结果的真实根因是：
    - `transpose_B` 语义在 Blackhole path 上被丢弃
    - `BlackholeModule` 直接上传 row-major host tensor，没有按 TT-Metal matmul contract 做 host-side tilize / untilize
  - 本轮已补：
    - `blackhole.gemm_contract`
    - `rt_mod_blackhole` → `ExecutableSpec` GEMM contract 传递
    - `BlackholeModule` 对 GEMM 输入做 host-side transpose/tilize，对输出做 untilize
- **下一步**：
  - `scratch_l1_buffer_addr32` 死代码确认与移除
  - GEMM compile-time ABI / dtype 分层正式化
  - richer accessor / layout schema 进入 `ExecutableSpec`

---

## 分阶段任务

| 阶段 | 目标 | 状态 |
|------|------|------|
| Stage 0 | 协议与执行载体 | ✅ ExecutableSpec, rt_mod_blackhole, BlackholeModule |
| Stage 1 | single-core copy bring-up | ✅ |
| Stage 2A | pass 主链接入 | ✅ AnnotateDeviceRegions → SplitHostDevice → MakePackedAPI → LowerDeviceKernelLaunch |
| Stage 2B | single-core copy 正式主链 | ✅ direct path copy E2E on TT-Sim |
| Stage 2C | split-before 语义规划 | ✅ AnnotateBlackholeCopySemantics + FlattenBuffer/VectorizeLoop; StorageRewrite 永久排除 |
| Stage 2D | single-core true E2E | ✅ Steps 1-6 完成 |
| Stage 2E | 设备资源 IR 语义扩展 | ✅ StorageRank + BlackholeDeviceResourceCanonicalization |
| Stage 3 | multi-core runtime 调度 | ⏳ 未开始 |

---

## Stage 2D 当前状态

### 已完成

- Step 1: `LowerTileOp` Blackhole GEMM skip ✅
- Step 2: `SplitBlackholeKernel` pass（3-kernel reader/compute/writer）✅
- Step 3: `LowerBlackholeOps` GEMM lower + planner-driven CB binding ✅
- Step 4: `rt_mod_blackhole` 多 segment extractor ✅
- Step 5: `BlackholeModule` 3-kernel 注册 ✅
- Step 6 前置: CB identity 唯一协议收正 ✅
  - `LowerBlackholeOps` 统一用 `requirement_index`
  - `PlanBlackholeCB` 回写 IR body，替换为最终 `cb_id`
  - 删除 placeholder/alias 修补逻辑

### Step 6: GEMM E2E 验收（已完成）

当前 blocker 和修复计划详见：
- `tasks/dev_design/stage2d_gemm_direct_cb_io.md` — 根因分析
- `tasks/dev_design/stage2d_ttmetal_contract_audit.md` — TT-Metal contract 缺口审计
- `tasks/dev_design/2026-03-26-stage2d-gemm-contract-implementation-plan.md` — 实施计划

本轮实际完成：
1. 复核 generated kernel source 与 lowered TIR，确认 CB 同步并未丢失
2. 定位 `transpose_B` 与 host row-major upload / no-untilize 才是 direct-path 数值错误根因
3. 新增 `blackhole.gemm_contract`，把 GEMM 维度与 transpose 语义从 lowering 传到 runtime
4. `BlackholeModule` 按 contract 对 GEMM 输入做 host-side transpose/tilize，对输出做 untilize
5. TT-Sim 验证：
   - `test_blackhole_gemm_basic` 通过
   - `test_blackhole_copy_runtime.py` 不回退
6. 删除 `tt_metal_repo/tt_metal/programming_examples/tilelang_gemm_test/`

---

## Copy 验收状态（已完成）

- 32x32, 32x64, 64x32 staged copy ✅
- grid > 1 (grid_x=2, grid_y=3, 96x64 float16) ✅
- large-shape (800x1024 float16, 1.6MB) ✅
- oversubscription 负例 (1024x1024, PlanBlackholeCB 编译期失败) ✅
- 测试: 16 passed, 1 xfailed

---

## 已知结构问题

- `PlanBlackholeCB` 仍是 MVP allocator，非正式 memory planner
- `StorageRewrite` 与 Blackhole CB 模型不兼容（永久排除）
- copy 用 fused_dataflow 单 kernel，GEMM 用 3-kernel，导致 rt_mod/BlackholeModule 双路径维护（后续统一）
- TT-Metal contract 缺层审计见 `stage2d_ttmetal_contract_audit.md`（本轮已补最小 host layout/transpose contract，P0/P1/P2/P3 仍有正式化欠账）

---

## 当前活动设计文档

- `final_blackhole_backend_redesign.md` — 唯一总设计
- `stage2d_gemm_direct_cb_io.md` — GEMM CB transport 修复
- `stage2d_ttmetal_contract_audit.md` — TT-Metal contract 审计
- `2026-03-26-stage2d-gemm-contract-implementation-plan.md` — 实施计划
- `stage2d_cb_identity_protocol.md` — CB identity 协议（已完成）
- `stage2d_gemm_integration.md` — GEMM 接入设计
- `stage2e_blackhole_device_resource_semantics.md` — 设备资源语义（已完成）
