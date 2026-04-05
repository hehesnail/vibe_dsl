# Stage 4 Phase C: TT Target IR And New-Mainline Cutover

## 基本信息

- **文档角色**: `Phase C` 实施与设计边界文档
- **当前状态**: 已定义；待 `Phase B` 后推进
- **上游输入**: 冻结后的 `SpatialProgram`
- **下游输出**: `TTProgram` 与 `MaterializeTTExecutableSpec` 物化结果
- **唯一总体设计**: `tasks/dev_design/final_blackhole_backend_redesign.md`

## 1. Phase C 的职责

`Phase C` 只负责一件事：

- 把冻结后的 spatial structure 映射成合法且稳定的 TT target contract

它回答的问题是：

- `Task / Channel / Layout / SyncEdge / ResourceIntent` 如何落成 TT program object
- 哪些 kernel、CB、transport、semaphore、dst layout、ABI 和 execution plan 必须显式存在
- 哪些 legacy `blackhole.*` attrs 还能保留为 compatibility projection，哪些必须删除

它不负责：

- 重新理解 semantic truth
- 重新发明 task graph 或 `ProgramPhase`
- 让 runtime/codegen 继续补 target contract

## 2. Core Design Boundary

### 2.1 核心对象

`Phase C` 的长期 core object set 是：

- `TTProgram`
- `TTKernel`
- `TTCoreGroup`
- `TTCBPlan`
- `TTTransportPlan`
- `TTSemaphorePlan`
- `TTComputeSyncPlan`
- `TTDstLayoutPlan`
- `TTABIPlan`
- `TTExecutionPlan`
- `TTHardwareModel`

### 2.2 小闭集 target family

TT 层仍然遵守 small-closed family 设计：

- `TTKernel.kind`
  - `data_movement`
  - `compute`
  - `collective`
  - `control`
- `TTCBPlan.resource_class`
  - `transport`
  - `scratch`
  - `carry`
  - `output`
- `TTSemaphorePlan.kind`
  - `local`
  - `remote`
  - `multicast`
  - `barrier`
- `TTTransportPlan.kind`
  - `unicast`
  - `multicast`
  - `tree`
  - `ring`
  - `line`
  - `fabric_mux`

更细 specialization 通过 typed traits、bindings 和 ABI schema 表达，不通过 target noun 爆炸。

### 2.3 Common-Runtime ABI 是一等对象

`Phase C` 必须把 ABI 固定成三层：

- `compile-time`
- `common-runtime`
- `per-work runtime`

因此：

- `TTKernel` 必须显式拥有 common-runtime bindings
- `TTABIPlan` 必须显式拥有 `common_runtime_arg_specs`
- `MaterializeTTExecutableSpec` 必须把这层稳定物化到 per-kernel schema

顶层 `blackhole.common_runtime_args` 只允许作为 compatibility aggregate view 过渡存在。

### 2.4 Hardware Model 不是附属品

`TTProgram` 的合法性不能建立在裸常量和 ad-hoc 判断上。
`Phase C` 至少需要一个 typed `TTHardwareModel`，覆盖：

- topology
- memory / CB / L1 bounds
- semaphore / sync capabilities
- dst / register legality
- ABI limits
- compute-unit hazard rules

### 2.5 Materialization 是唯一 writer

`MaterializeTTExecutableSpec` 是唯一允许把 `TTProgram` 物化成：

- `ExecutableSpec`
- `KernelSpec`
- materialized `blackhole.*` attrs
- target-lowered executable `PrimFunc`

因此：

- `LowerBlackholeOps`
- `PlanBlackholeCB`
- `AssignBlackholeCores`
- `codegen_blackhole`
- `rt_mod_blackhole`

都不能继续作为 target contract 的第二真源 writer。

### 2.6 不能回退的约束

`Phase C` 不允许：

- 重新发明 semantic/spatial truth
- 继续把 `ExecutableSpec` 当第二真源
- 让 runtime 通过 fallback 补齐缺失的 CB / semaphore / ABI / route contract
- 因为当前样例方便，就把 TT 层重新写成 monolithic attr bag

## 3. Cutover And Deletion Gates

`Phase C` 的核心不是“再加一层对象”，而是完成真源切换。

切换规则是：

1. 上游 `TTProgram` / `TTABIPlan` / `TTExecutionPlan` 先成为稳态真源
2. `MaterializeTTExecutableSpec` 成为唯一 writer
3. 只有当稳态字段齐备、验证器已能 fail-fast、runtime/codegen 已切到只读消费后，
   对应 compatibility writer / reader / fallback 才允许删除

典型待切对象包括：

- `blackhole.segment_plan`
- `blackhole.runtime_args`
- `blackhole.common_runtime_args`
- `blackhole.accessors`
- `blackhole.cb_configs`
- `blackhole.semaphore_plan`
- `blackhole.core_plan`

## 4. 当前实施重点

当前 `Phase C` 的实施重点是：

1. 建立最小 TT target object 集
2. 让 copy / GEMM 先完成 target materialization cutover
3. 删除 compatibility 路径
4. 兑现 `flash-attn` correctness payoff
5. 在新主链下做 family expansion

## 5. Shared Zero-Regression Baseline

```bash
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q
```

## 6. TT-Sim Runtime Gate

```bash
source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh
export TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo
cd /root/dev/vibe_dsl/tilelang_repo
pytest testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py -k 'mha or gqa' -q
```

## 7. Task 4: Stage 4 - Phase C1 TT Target IR And Materialization Cutover

**Files:**
- Create: `tilelang_repo/src/transform/common/tt_target_program.h`
- Create: `tilelang_repo/src/transform/common/tt_target_program.cc`
- Create: `tilelang_repo/src/transform/lower_spatial_program_to_tt_target.cc`
- Create: `tilelang_repo/src/transform/validate_tt_target_program.cc`
- Create: `tilelang_repo/src/transform/materialize_tt_executable_spec.cc`
- Create: `tilelang_repo/testing/python/transform/test_blackhole_tt_target_ir.py`
- Modify: `tilelang_repo/src/transform/plan_blackhole_cb.cc`
- Modify: `tilelang_repo/src/transform/assign_blackhole_cores.cc`
- Modify: `tilelang_repo/src/target/rt_mod_blackhole.cc`
- Modify: `tilelang_repo/src/target/blackhole_module.cc`
- Modify: `tilelang_repo/src/target/codegen_blackhole.cc`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`

- [ ] **Step 1: Introduce minimal TT target object set**

Required objects:

- `TTProgram`
- `TTKernel`
- `TTCBPlan`
- `TTTransportPlan`
- `TTSemaphorePlan`
- `TTComputeSyncPlan`
- `TTDstLayoutPlan`
- `TTABIPlan`
- `TTExecutionPlan`
- `TTHardwareModelStub`

- [ ] **Step 2: Move copy / GEMM materialization to `MaterializeTTExecutableSpec`**

Run:

```bash
pytest tilelang_repo/testing/python/transform/test_blackhole_tt_target_ir.py -k 'copy or gemm or materialize' -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -q
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py -q
```

Expected:

- `ExecutableSpec` / `KernelSpec` 只从 TT target truth 物化
- `blackhole.runtime_args` / `common_runtime_args` / `cb_configs` / `core_plan` 只作为 compatibility projection 存在
- copy / GEMM runtime 不回退

- [ ] **Step 3: Re-run shared zero-regression baseline**

Run the shared zero-regression baseline above.

- [ ] **Step 4: Stage 4 exit gate**

Only proceed when:

- `TTABIPlan` 已拥有 compile-time / common-runtime / per-work 三层 ABI
- `TTTransportPlan` / `TTHardwareModelStub` 已接入合法性检查
- copy / GEMM spec/runtime 已经由 `MaterializeTTExecutableSpec` 接管

## 8. Task 5: Stage 5 - Phase C2 Compatibility Deletion And Flash-Attn Correctness

**Files:**
- Modify: `tilelang_repo/src/transform/lower_blackhole_ops.cc`
- Modify: `tilelang_repo/src/transform/plan_blackhole_cb.cc`
- Modify: `tilelang_repo/src/transform/assign_blackhole_cores.cc`
- Modify: `tilelang_repo/src/target/rt_mod_blackhole.cc`
- Modify: `tilelang_repo/src/target/blackhole_module.cc`
- Modify: `tilelang_repo/src/target/codegen_blackhole.cc`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py`

- [ ] **Step 1: Delete compatibility writers and fallback readers by gate**

Delete only after corresponding TT stable fields exist:

- `blackhole.segment_plan`
- `blackhole.runtime_args`
- `blackhole.common_runtime_args`
- `blackhole.accessors`
- `blackhole.cb_configs`
- `blackhole.semaphore_plan`
- `blackhole.core_plan`

- [ ] **Step 2: Re-run flash-attn pipeline gate on the new mainline**

Run:

```bash
pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -q
```

Expected:

- `flash-attn` compile-path 不再依赖 late target-specific semantic guessing
- `blackhole.acc` 不再同时承载 algorithm state 和 TT scratch 两类真语义

- [ ] **Step 3: Run direct runtime correctness gate in TT-Sim**

Run:

```bash
source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh
export TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo
cd /root/dev/vibe_dsl/tilelang_repo
pytest testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py -k 'mha or gqa' -q
```

Expected:

- `mha` direct runtime correctness pass
- `gqa` direct runtime correctness pass
- 不再出现当前 `blackhole.acc` 混合语义导致的 mismatch

- [ ] **Step 4: Re-run shared zero-regression baseline**

Run the shared zero-regression baseline above.

- [ ] **Step 5: Stage 5 exit gate**

Only proceed when:

- `flash-attn` MHA/GQA direct runtime correctness 通过
- 基线 copy / GEMM / export / pipeline 全绿
- compatibility writer / reader / fallback 已按 deletion gates 收掉

## 9. Task 6: Stage 6 - Family Expansion Under The New Mainline

**Files:**
- Modify: `tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py`
- Modify: `tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py`
- Modify: `tilelang_repo/testing/python/transform/test_blackhole_tt_target_ir.py`
- Create or modify consumer-specific tests alongside selected workloads
- Modify: `tasks/progress.md`

- [ ] **Step 1: Add the first non-attention family to the new path**

Recommended order:

1. `topk`
2. paged decode
3. `fusedmoe`
4. chunk recurrence

- [ ] **Step 2: Add compile gates before runtime gates**

Run:

```bash
pytest tilelang_repo/testing/python/transform/test_blackhole_semantic_ir.py -q
pytest tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py -q
pytest tilelang_repo/testing/python/transform/test_blackhole_tt_target_ir.py -q
```

Expected:

- 每个 family 先在 semantic / spatial / TT 层有稳定结构 gate
- 只有进入正式 direct runtime 支持面后，才加 TT-Sim runtime gate

- [ ] **Step 3: Re-run shared zero-regression baseline**

Run the shared zero-regression baseline above.

- [ ] **Step 4: Stage 6 exit gate**

Only proceed when:

- 新增 family 不引入 case-by-case matcher
- 每个 family 都先通过 compile gate，再决定是否进入 runtime gate
