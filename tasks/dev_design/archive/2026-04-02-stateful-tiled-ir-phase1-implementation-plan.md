# Stateful Tiled IR Phase 1 Implementation Plan

> **Status (2026-04-02, superseded as overall architecture):** 这份文档不再代表下一阶段总体架构。当前唯一权威总体设计已改为多层 compiler-internal IR：`Stateful Semantic IR -> Spatial Program IR -> TT Target IR`，见 `tasks/dev_design/final_blackhole_backend_redesign.md`。旧的 runtime/混合架构说明已归档到 `tasks/dev_design/archive/legacy_blackhole_runtime_architecture.md`。本文件仅保留为新总设计下 **Phase A（Semantic IR）** 的历史草案与迁移参考，后续如继续执行应先按新总设计重写为对应子计划。除顶部状态说明外，正文中的旧 `Stateful Tiled IR` / `LiftToStatefulTiledIR` / `BlackholeStatefulProgramLowerer` 命名暂未整体回收，阅读时应按“已 supersede 的旧 Phase A 草案”理解。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在尽量不改用户 Python DSL 主体写法的前提下，先完成新总设计中 **Phase A: Stateful Semantic IR** 的早期实现草案；本文件原始目标中的单层 `Stateful Tiled IR` 现在应理解为“多层 IR 体系中的语义层草案”，而不再是最终中间层。

**Architecture:** 本文件描述的内容只覆盖新总设计中的语义层草案：先在 compiler 内部新增 `Domain / State / Relation / Phase` 四类核心对象（Op kind 延后到后续阶段）与语义层 `Lift / Validate` pass；`Spatial Program IR` 和 `TT Target IR` 不在本文件内定义。

**Tech Stack:** TileLang Python DSL, TVM PrimFunc/TIR, TVM ObjectRef/ObjectNode, TileLang transform passes, Blackhole lowering/codegen/runtime, pytest, CMake, TT-Sim

---

## Scope Note

这份计划覆盖 **Phase 1**，拆分为 **Phase 1a（纯 IR 增量）** 和 **Phase 1b（语义迁移 + target lowering）**：

**Phase 1a（零回归风险）：**

- 引入 compiler-internal `Stateful Tiled IR` 核心对象（Domain / State / Relation / Phase）
- 引入 `LiftToStatefulTiledIR` / `ValidateStatefulTiledIR` pass
- 打通 dense + carry/state-coupled 形态的 lift 与 validation
- 验收标准：lift 能跑、validate 能拦、现有 GEMM/copy compile-path 测试不回归；TT-Sim runtime 总回归留到 Phase 1b 统一执行

**Phase 1b（语义迁移，单独调试）：**

- 实现 `BlackholeStatefulProgramLowerer`，包含 dst register layout planning
- 把 flash-attn compute 从混合 `blackhole.acc` 语义迁到 Stateful Tiled IR 消费路径
- 验收标准：flash-attn runtime correctness 通过

**Phase 1 不固化 Op kind enum**——Phase 1 用 Phase 内的 TIR statement + State/Relation annotation 表达语义；Op 分类在 Phase 2 覆盖更多 workload 后再统一定义。

**不在本计划内完整落地**：

- Op kind 统一分类（Phase 2）
- MoE routed/segmented lifting 的完整 bring-up
- topk / argmax selection state machine 的完整 lowering
- paged decode / sparse MLA 的完整 page-table lowering
- 面向用户的公开 annotation API 扩展
- Lift 时机前移到 `inject_pipeline` 之后（见总设计 §8.6，Phase 2+ 条件成熟后迁移）

这些内容在 Phase 1 完成后另起后续实施计划。

## Design Review Conclusions (2026-04-02)

以下关键设计决策来自对 TileLang examples 全景、TT-Metal 原生 SDPA/LayerNorm/MoE 实现、以及 DSL lowering 信息丢失链的系统性审查：

### D1. Domain 需要 constraint/predicate 能力

TileLang examples 里的 causal attention（data-dependent loop bound）、GQA（grouped index remapping）、block sparse attention（block_mask predicate）都不是 exotic case——它们是核心 use case。Domain 不能只靠 kind enum，至少需要 `bound_expr` 和 `predicate` 字段。

### D2. State kind 使用中性命名

使用 `matrix_state / vector_state / scalar_state`，不使用 `tile_state`。”tile” 只在 target lowering 层出现，确保 IR 对 CUDA/AMX 等后端仍成立。

### D3. Op kind Phase 1 不固化

TT-Metal SDPA 的 `custom_mm_reuse_dest_srcb_block`（dest-reuse matmul + rescale 融合）无法用固定 Op kind 表达。Phase 1 只用 Phase + State/Relation 表达语义，Phase 2 再回收 Op 分类。

### D4. combine 规则不用固定 enum

Welford combine 是多变量耦合的 `(mean, M2, count) = welford_combine(old, new)`，online softmax 的 rescale+accumulate 也不是简单 sum/max。combine rule 用 function reference 或保留 TIR body。

### D5. Phase 区分 algorithm phase 和 pipeline phase

TT-Metal SDPA 的整个 K-loop 是**单个 algorithm phase**（一个 `tile_regs_acquire/release` block）。`T.Pipelined` 的 stage 是 data prefetch pipeline，和 compute 内的 algorithm phase 正交。

### D6. Target lowering 必须包含 dst register layout planning

TT-Metal SDPA 的 dst 空间是编译时静态分区的（output accumulator / running max / running sum / correction factor / QK temp 各有固定 offset）。`BlackholeStatefulProgramLowerer` 必须有 `PlanDstRegisterLayout` 步骤。

### D7. Carry strategy 是 target lowering 决策

同一个 `carry` state 可以走 register-resident carry（dst 长期持有）或 CB-round-trip carry（每次 pack/unpack）。IR 层只标 `carry`，target lowering 根据总 dst 空间需求决定。

### D8. TT-Metal SDPA ground truth 作为 lowering 对标物

Phase 1b 在实现 `BlackholeStatefulProgramLowerer` 前，必须先把 TT-Metal 原生 SDPA compute kernel 的目标代码作为 lowering 的 ground truth 写进设计。参考文件：
- `tt_metal_repo/tt-train/sources/ttml/metal/ops/sdpa_fw/device/kernels/compute/sdpa_fw_compute_kernel.cpp`
- `tt_metal_repo/models/demos/deepseek_v3_b1/kernel_includes/tt_metal/include/compute_kernel_api/sdpa.h`
- `tt_metal_repo/tt-train/sources/ttml/metal/ops/sdpa_fw/device/sdpa_fw_program_factory.cpp`

## File Map

- Create: `tilelang_repo/src/transform/common/stateful_tiled_ir.h`
  - 定义 `StatefulTiledProgram / Domain / State / Relation / Phase` ObjectNode/ObjectRef（`ops` 字段只预留，不在 Phase 1 固化 `Op` ObjectNode）
- Create: `tilelang_repo/src/transform/common/stateful_tiled_ir.cc`
  - 注册对象、`ReprPrinter`、最小 debug dump / attr encode 辅助
- Create: `tilelang_repo/src/transform/lift_stateful_tiled_ir.cc`
  - 从现有 TIR + analysis attrs 生成 `tl.stateful_tiled_program`
- Create: `tilelang_repo/src/transform/validate_stateful_tiled_ir.cc`
  - 检查 state kind、carry、combine rule、phase live-in/live-out 的一致性
- Create: `tilelang_repo/testing/python/transform/test_stateful_tiled_ir.py`
  - 通用 lift / validate / structural snapshot 回归
- Create: `tilelang_repo/src/transform/lower_blackhole_stateful_tiled_program.h`
  - Blackhole target-specific lowering helper 接口
- Create: `tilelang_repo/src/transform/lower_blackhole_stateful_tiled_program.cc`
  - 把 `StatefulTiledProgram` 映射成 TT-Metal-first compute builtins / CB requirements / state classes
- Modify: `tilelang_repo/CMakeLists.txt`
  - 把新 transform/common/blackhole lowering 源文件接入 `USE_BLACKHOLE`
- Modify: `tilelang_repo/tilelang/transform/__init__.py`
  - 暴露 `LiftToStatefulTiledIR` / `ValidateStatefulTiledIR`
- Modify: `tilelang_repo/tilelang/engine/lower.py`
  - 在 Blackhole 主链插入 `LiftToStatefulTiledIR` 和 `ValidateStatefulTiledIR`
- Modify: `tilelang_repo/src/transform/analyze_blackhole_work_decomposition.cc`
  - 输出 lift 需要的稳定 work/domain 信息
- Modify: `tilelang_repo/src/transform/analyze_blackhole_fragment_regions.cc`
  - 输出 lift 需要的 tile-state / row-state / region role 信息
- Modify: `tilelang_repo/src/transform/analyze_blackhole_pipeline_stages.cc`
  - 输出 phase / live-in / live-out / carry 边界信息
- Modify: `tilelang_repo/src/transform/lower_blackhole_ops.h`
  - 缩减为 transport / CB requirement / host-visible schema 提取入口，并委托 stateful compute lowering
- Modify: `tilelang_repo/src/transform/lower_blackhole_ops.cc`
  - 接入 `StatefulTiledProgram` 消费路径，删除对 flash-attn compute 语义的晚期 loop 猜测
- Modify: `tilelang_repo/src/transform/plan_blackhole_cb.cc`
  - 区分 transport CB、tile scratch CB、stats/scalar scratch 资源类
- Modify: `tilelang_repo/src/tir/builtin_blackhole.h`
  - 为 tile/stats-state lowering 增加或收正 target builtins 声明
- Modify: `tilelang_repo/src/tir/builtin_blackhole.cc`
  - 注册 tile/stats-state builtins
- Modify: `tilelang_repo/src/target/codegen_blackhole.cc`
  - 发射 tile/stats-state builtins 对应的 TT-Metal-first compute 协议
- Modify: `tilelang_repo/src/target/rt_mod_blackhole.cc`
  - 只提取 target-program lowering 冻结出来的 schema，不再直接理解 fragment helper 语义
- Modify: `tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py`
  - 增加 `StatefulTiledIR` lift 入口覆盖
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py`
  - 增加 `StatefulTiledIR` 消费、tile/stats-state class、legacy fragment helper 移除回归
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py`
  - 用 TT-Sim 验证 flash-attn runtime correctness 不再依赖混合 `blackhole.acc` 语义
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`
  - 验证 GEMM 仍走新路径且 multi-tile output / CB classes 正常
- Modify: `tasks/dev_design/final_blackhole_backend_redesign.md`
  - 实施后同步实际边界
- Modify: `tasks/progress.md`
  - 同步阶段状态和下一步
- Modify: `memory/general_dev.md`
  - 记录 `StatefulTiledIR` / TT-Sim / plan-to-runtime 的稳定经验
- Modify: `memory/bugs.md`
  - 记录 `blackhole.acc` 混合语义问题及新 IR 收敛方式

## Reading Guide

- 先读：`tasks/dev_design/final_blackhole_backend_redesign.md`
- 再读：`tasks/progress.md`
- 代码主入口：
  - `tilelang_repo/tilelang/engine/lower.py`
  - `tilelang_repo/src/transform/lower_blackhole_ops.cc`
  - `tilelang_repo/src/target/codegen_blackhole.cc`
  - `tilelang_repo/src/transform/analyze_blackhole_*.cc`
- 验证入口：
  - `tilelang_repo/testing/python/transform/test_stateful_tiled_ir.py`
  - `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py`
  - `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py`
  - `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`

### Task 1: Introduce `Stateful Tiled IR` Core Objects And Pass Entrypoints [Phase 1a]

> **Phase 1a scope**: 这个 Task 属于纯 IR 增量，不改 lowering/codegen/runtime，不影响现有 GEMM/copy 路径。

**Files:**
- Create: `tilelang_repo/src/transform/common/stateful_tiled_ir.h`
- Create: `tilelang_repo/src/transform/common/stateful_tiled_ir.cc`
- Create: `tilelang_repo/src/transform/lift_stateful_tiled_ir.cc`
- Create: `tilelang_repo/src/transform/validate_stateful_tiled_ir.cc`
- Modify: `tilelang_repo/CMakeLists.txt`
- Modify: `tilelang_repo/tilelang/transform/__init__.py`
- Test: `tilelang_repo/testing/python/transform/test_stateful_tiled_ir.py`

- [ ] **Step 1: Write the failing transform tests**

```python
def test_lift_stateful_tiled_ir_emits_program_attr():
    mod = _split_and_analyze(_make_flash_attn_program())
    mod = tilelang.transform.LiftToStatefulTiledIR()(mod)
    body = str(mod["main"])
    assert "tl.stateful_tiled_program" in body


def test_validate_stateful_tiled_ir_accepts_minimal_dense_program():
    mod = _split_and_analyze(_make_online_softmax_program())
    mod = tilelang.transform.LiftToStatefulTiledIR()(mod)
    mod = tilelang.transform.ValidateStatefulTiledIR()(mod)
    body = str(mod["main"])
    assert "tl.stateful_tiled_program" in body
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest -q tilelang_repo/testing/python/transform/test_stateful_tiled_ir.py -k "program_attr or minimal_dense_program"
```

Expected:

```text
FAIL ... AttributeError: module 'tilelang.transform' has no attribute 'LiftToStatefulTiledIR'
```

- [ ] **Step 3: Add core ObjectNode definitions and empty pass stubs**

注意：Phase 1 只引入 `Domain / State / Relation / Phase` 四类对象。`Op` kind 延后到 Phase 2。`StatefulTiledProgramNode` 预留 `ops` 字段但 Phase 1 不填充。

```cpp
class StatefulTiledDomainNode : public Object {
 public:
  String kind;                        // "dense", "segmented", "routed", "paged"
  Optional<PrimExpr> bound_expr;      // data-dependent bound (e.g., causal loop range)
  Optional<PrimExpr> predicate;       // dynamic predicate (e.g., block_mask[k] != 0)
  Optional<Map<String, PrimExpr>> index_remapping;  // grouped index (e.g., by // groups)
  TVM_DECLARE_FINAL_OBJECT_INFO(StatefulTiledDomainNode, Object);
};

class StatefulTiledStateNode : public Object {
 public:
  String name;
  String kind;      // "matrix_state", "vector_state", "scalar_state", "index_state"
  String lifetime;   // "ephemeral", "carry", "cross_phase"
  Optional<Buffer> backing_buffer;
  TVM_DECLARE_FINAL_OBJECT_INFO(StatefulTiledStateNode, Object);
};

class StatefulTiledRelationNode : public Object {
 public:
  String type;       // "reduced_from", "applies_to", "indexes", "scatters_to", "carried_across"
  String source;
  String target;
  // combine rule: NOT an enum. References TIR body or named function.
  // Welford combine, online-softmax rescale+accumulate, simple sum/max
  // are all valid — expressiveness comes from the reference, not a fixed set.
  Optional<String> combine_function;
  TVM_DECLARE_FINAL_OBJECT_INFO(StatefulTiledRelationNode, Object);
};

class StatefulTiledPhaseNode : public Object {
 public:
  String name;
  String phase_kind;  // "algorithm" vs "pipeline" — see design review D5
  Array<String> live_in;
  Array<String> live_out;
  TVM_DECLARE_FINAL_OBJECT_INFO(StatefulTiledPhaseNode, Object);
};

class StatefulTiledProgramNode : public Object {
 public:
  Array<ObjectRef> domains;
  Array<ObjectRef> states;
  Array<ObjectRef> relations;
  Array<ObjectRef> ops;       // Phase 1: empty; Phase 2: populated after Op kind unification
  Array<ObjectRef> phases;
  TVM_DECLARE_FINAL_OBJECT_INFO(StatefulTiledProgramNode, Object);
};

tir::transform::Pass LiftToStatefulTiledIRPass() {
  auto fpass = [](tir::PrimFunc func, IRModule m, tir::transform::PassContext ctx) {
    ICHECK(false) << "LiftToStatefulTiledIRPass not implemented";
    return func;
  };
  return tir::transform::CreatePrimFuncPass(
      fpass, 0, "tl.transform.LiftToStatefulTiledIR", {});
}
```

- [ ] **Step 4: Expose the pass entrypoints to Python and build**

```python
def LiftToStatefulTiledIR():
    return tvm.ffi.get_global_func("tl.transform.LiftToStatefulTiledIR")()


def ValidateStatefulTiledIR():
    return tvm.ffi.get_global_func("tl.transform.ValidateStatefulTiledIR")()
```

```cmake
file(GLOB TILE_LANG_BLACKHOLE_SRCS
  ...
  src/transform/lift_stateful_tiled_ir.cc
  src/transform/validate_stateful_tiled_ir.cc
  src/transform/common/stateful_tiled_ir.cc
)
```

- [ ] **Step 5: Re-run the targeted test to verify the new failure point is the pass stub**

Run:

```bash
pytest -q tilelang_repo/testing/python/transform/test_stateful_tiled_ir.py -k "program_attr or minimal_dense_program"
```

Expected:

```text
FAIL ... LiftToStatefulTiledIRPass not implemented
```

- [ ] **Step 6: Commit**

```bash
git add tilelang_repo/src/transform/common/stateful_tiled_ir.h \
        tilelang_repo/src/transform/common/stateful_tiled_ir.cc \
        tilelang_repo/src/transform/lift_stateful_tiled_ir.cc \
        tilelang_repo/src/transform/validate_stateful_tiled_ir.cc \
        tilelang_repo/CMakeLists.txt \
        tilelang_repo/tilelang/transform/__init__.py \
        tilelang_repo/testing/python/transform/test_stateful_tiled_ir.py
git commit -m "transform: add stateful tiled ir skeleton"
```

### Task 2: Lift Dense / Carry Semantics Into `Stateful Tiled IR` [Phase 1a]

> **Phase 1a scope**: 纯 lift 实现，不改 lowering/codegen。

**Files:**
- Modify: `tilelang_repo/src/transform/lift_stateful_tiled_ir.cc`
- Modify: `tilelang_repo/src/transform/analyze_blackhole_work_decomposition.cc`
- Modify: `tilelang_repo/src/transform/analyze_blackhole_fragment_regions.cc`
- Modify: `tilelang_repo/src/transform/analyze_blackhole_pipeline_stages.cc`
- Modify: `tilelang_repo/testing/python/transform/test_stateful_tiled_ir.py`
- Modify: `tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py`

- [ ] **Step 1: Add failing structural expectations for flash-attn, online-softmax, and constrained domains**

Phase 1a 不再断言旧的 `tile_state/state_reduce/tile_apply_state/state_update` marker；这里只验证 `Domain / State / Relation / Phase` 四类核心对象及其字段。

```python
def test_flash_attention_lift_emits_matrix_and_vector_carry_state():
    mod = _lift_stateful_tiled_ir(_make_flash_attn_program())
    body = str(mod["main"])
    assert "matrix_state" in body
    assert "vector_state" in body
    assert "carried_across" in body


def test_online_softmax_lift_emits_reduce_relation_and_algorithm_phase():
    mod = _lift_stateful_tiled_ir(_make_online_softmax_program())
    body = str(mod["main"])
    assert "reduced_from" in body
    assert "carried_across" in body
    assert "algorithm" in body


def test_causal_flash_attention_lift_emits_bound_expr():
    mod = _lift_stateful_tiled_ir(_make_causal_flash_attn_program())
    body = str(mod["main"])
    assert "bound_expr" in body


def test_gqa_flash_attention_lift_emits_index_remapping():
    mod = _lift_stateful_tiled_ir(_make_gqa_flash_attn_program())
    body = str(mod["main"])
    assert "index_remapping" in body


def test_predicated_domain_lift_emits_predicate():
    mod = _lift_stateful_tiled_ir(_make_predicated_domain_program())
    body = str(mod["main"])
    assert "predicate" in body
```

- [ ] **Step 2: Run the transform tests to verify they fail**

Run:

```bash
pytest -q tilelang_repo/testing/python/transform/test_stateful_tiled_ir.py -k "matrix_and_vector_carry_state or reduce_relation_and_algorithm_phase or bound_expr or index_remapping or predicate"
```

Expected:

```text
FAIL ... expected matrix_state/reduced_from/bound_expr/index_remapping/predicate markers not found
```

- [ ] **Step 3: Implement lift from existing analysis attrs instead of re-parsing late loop shape**

注意：State kind 使用中性命名（`matrix_state / vector_state / scalar_state`），不使用 `tile_state`。Domain 包含 `bound_expr` 以支持 causal attention 的 data-dependent bound。Phase 1 不填充 `ops` 字段。

```cpp
StatefulTiledProgram program;
// Domain 可以包含 data-dependent bound / predicate / index remapping
program->domains.push_back(MakeDenseDomain(work_attr, bound_expr));

for (const auto& region : fragment_regions) {
  if (region_role == "matmul_output_tile") {
    // "matrix_state" not "tile_state" — "tile" is target-lowering vocabulary
    program->states.push_back(MakeMatrixState(region_name, "carry"));
  } else if (region_role == "stats_reduce_row") {
    program->states.push_back(MakeVectorState(region_name, "carry"));
    // combine rule references TIR body, not fixed enum
    program->relations.push_back(MakeReducedFrom(region_name, source_tile, "row"));
  } else if (region_role == "stats_scalar") {
    program->states.push_back(MakeScalarState(region_name, "carry"));
  }
}

for (const auto& stage : pipeline_stages) {
  program->phases.push_back(MakePhase(stage_name, "algorithm", live_in, live_out));
}

// carried_across is derived from Phase live-in/live-out — validate consistency
for (const auto& state : program->states) {
  if (state->lifetime == "carry") {
    program->relations.push_back(MakeCarriedAcross(state->name, domain_name));
  }
}

attrs.Set("tl.stateful_tiled_program", program);
func.CopyOnWrite()->attrs = DictAttrs(attrs);
```

- [ ] **Step 4: Tighten the analysis passes so lift consumes stable inputs**

```cpp
attrs.Set("blackhole.fragment_regions", EncodeFragmentRegionsWithStateRoles(...));
attrs.Set("blackhole.pipeline_stages", EncodePipelineStagesWithLiveSets(...));
attrs.Set("blackhole.work_decomposition", EncodeWorkDomains(...));
```

- [ ] **Step 5: Run the focused transform coverage**

Run:

```bash
pytest -q tilelang_repo/testing/python/transform/test_stateful_tiled_ir.py
pytest -q tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py -k "stateful or fragment or causal or gqa"
```

Expected:

```text
... passed
```

- [ ] **Step 6: Commit**

```bash
git add tilelang_repo/src/transform/lift_stateful_tiled_ir.cc \
        tilelang_repo/src/transform/analyze_blackhole_work_decomposition.cc \
        tilelang_repo/src/transform/analyze_blackhole_fragment_regions.cc \
        tilelang_repo/src/transform/analyze_blackhole_pipeline_stages.cc \
        tilelang_repo/testing/python/transform/test_stateful_tiled_ir.py \
        tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py
git commit -m "transform: lift dense stateful tiled semantics"
```

### Task 3: Validate Ambiguous Semantics Early And Insert The New Passes Into The Pipeline [Phase 1a]

> **Phase 1a scope**: Validation rules + pipeline wiring. 新 pass 插入 pipeline 但不改 `LowerBlackholeOps` 的消费逻辑——Phase 1a 的 `LowerBlackholeOps` 只 passthrough `tl.stateful_tiled_program` attr，不消费它。

**Files:**
- Modify: `tilelang_repo/src/transform/validate_stateful_tiled_ir.cc`
- Modify: `tilelang_repo/tilelang/engine/lower.py`
- Modify: `tilelang_repo/testing/python/transform/test_stateful_tiled_ir.py`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py`

- [ ] **Step 1: Add failing validation tests for mixed state semantics**

```python
def test_validate_rejects_state_without_kind():
    mod = _make_invalid_stateful_tiled_ir_missing_kind()
    with pytest.raises(tvm.TVMError, match="state kind"):
        tilelang.transform.ValidateStatefulTiledIR()(mod)


def test_blackhole_pipeline_rejects_mixed_acc_semantics_before_codegen():
    mod = _lower_flash_attn_to_stateful_tiled_ir_with_legacy_acc_mix()
    with pytest.raises(tvm.TVMError, match="tile scratch|vector_state|blackhole.acc"):
        tilelang.transform.ValidateStatefulTiledIR()(mod)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest -q tilelang_repo/testing/python/transform/test_stateful_tiled_ir.py -k "missing_kind"
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -k "mixed_acc_semantics"
```

Expected:

```text
FAIL ... did not raise TVMError
```

- [ ] **Step 3: Implement validation rules and wire passes into the Blackhole lowering pipeline**

```cpp
// State kind validation
if (state->kind.empty()) {
  LOG(FATAL) << "StatefulTiledIR validation failed: missing state kind for " << state->name;
}
// Valid state kinds: matrix_state, vector_state, scalar_state, index_state
if (state->kind != "matrix_state" && state->kind != "vector_state" &&
    state->kind != "scalar_state" && state->kind != "index_state") {
  LOG(FATAL) << "StatefulTiledIR validation failed: unknown state kind " << state->kind;
}

// Scatter relation requires combine function reference
if (relation->type == "scatters_to" && !relation->combine_function.defined()) {
  LOG(FATAL) << "StatefulTiledIR validation failed: scatter relation missing combine function";
}

// carried_across / phase live-in-out consistency check
for (const auto& rel : program->relations) {
  if (rel->type == "carried_across") {
    bool found_in_live_out = false;
    for (const auto& phase : program->phases) {
      if (Contains(phase->live_out, rel->source)) found_in_live_out = true;
    }
    ICHECK(found_in_live_out) << "carried_across state " << rel->source
                              << " not found in any phase live_out";
  }
}

// Phase kind validation
for (const auto& phase : program->phases) {
  if (phase->phase_kind != "algorithm" && phase->phase_kind != "pipeline") {
    LOG(FATAL) << "StatefulTiledIR validation failed: unknown phase kind " << phase->phase_kind;
  }
}

// Reject mixed blackhole.acc semantics
if (UsesBlackholeAccAsBothMatrixAndVector(program)) {
  LOG(FATAL) << "StatefulTiledIR validation failed: blackhole.acc mixed matrix/vector semantics";
}
```

```python
# Pipeline insertion: Lift + Validate before LowerBlackholeOps
# Phase 1a: LowerBlackholeOps passes through tl.stateful_tiled_program without consuming it
# Phase 1b: LowerBlackholeOps delegates compute lowering to BlackholeStatefulProgramLowerer
device_mod = tilelang.transform.AnalyzeBlackholePipelineStages()(device_mod)
device_mod = tilelang.transform.LiftToStatefulTiledIR()(device_mod)
device_mod = tilelang.transform.ValidateStatefulTiledIR()(device_mod)
device_mod = tilelang.transform.LowerBlackholeOps()(device_mod)
```

- [ ] **Step 4: Run the focused pass-pipeline regression**

Run:

```bash
pytest -q tilelang_repo/testing/python/transform/test_stateful_tiled_ir.py
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -k "mixed_acc_semantics or stateful"
```

Expected:

```text
... passed/skipped
```

- [ ] **Step 5: Commit**

```bash
git add tilelang_repo/src/transform/validate_stateful_tiled_ir.cc \
        tilelang_repo/tilelang/engine/lower.py \
        tilelang_repo/testing/python/transform/test_stateful_tiled_ir.py \
        tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py
git commit -m "transform: validate stateful tiled ir before blackhole lowering"
```

### Task 4: Migrate Blackhole Compute Lowering To Consume `Stateful Tiled IR` [Phase 1b]

> **Phase 1b scope**: 这个 Task 是真正的语义迁移——改 lowering/codegen/runtime。必须在 Phase 1a（Tasks 1-3）完成并验证后才开始。

> **前置要求**：在实现 `BlackholeStatefulProgramLowerer` 前，先把 TT-Metal 原生 SDPA compute kernel 的目标代码作为 lowering 的 ground truth 对标物（见 Design Review D8）。参考 `tt_metal_repo/tt-train/sources/ttml/metal/ops/sdpa_fw/device/kernels/compute/sdpa_fw_compute_kernel.cpp` 和 `tt_metal_repo/models/demos/deepseek_v3_b1/kernel_includes/tt_metal/include/compute_kernel_api/sdpa.h`。

**Files:**
- Create: `tilelang_repo/src/transform/lower_blackhole_stateful_tiled_program.h`
- Create: `tilelang_repo/src/transform/lower_blackhole_stateful_tiled_program.cc`
- Modify: `tilelang_repo/src/transform/lower_blackhole_ops.h`
- Modify: `tilelang_repo/src/transform/lower_blackhole_ops.cc`
- Modify: `tilelang_repo/src/transform/plan_blackhole_cb.cc`
- Modify: `tilelang_repo/src/tir/builtin_blackhole.h`
- Modify: `tilelang_repo/src/tir/builtin_blackhole.cc`
- Modify: `tilelang_repo/src/target/codegen_blackhole.cc`
- Modify: `tilelang_repo/src/target/rt_mod_blackhole.cc`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py`
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`

- [ ] **Step 1: Add failing pipeline tests that prove the new path is required**

```python
def test_flash_attention_pipeline_emits_stateful_tile_stats_builtins():
    mod = _lower_flash_attn_pipeline()
    source = str(mod["main"])
    assert "tl.blackhole.stats_reduce_row" in source
    assert "tl.blackhole.tile_apply_stats" in source


def test_gemm_pipeline_marks_output_as_tile_scratch_not_fragment_array():
    mod = _lower_gemm_pipeline()
    source = str(mod["main"])
    assert "tile_scratch" in source
    assert "fragment_linear_helper" not in source
```

- [ ] **Step 2: Run tests to verify they fail on the legacy path**

Run:

```bash
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -k "stateful_tile_stats_builtins"
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -k "tile_scratch_not_fragment_array"
```

Expected:

```text
FAIL ... tl.blackhole.stats_reduce_row / tile_scratch markers not found
```

- [ ] **Step 3: Implement a dedicated stateful-program lowering helper with dst register planning**

```cpp
class BlackholeStatefulProgramLowerer {
 public:
  explicit BlackholeStatefulProgramLowerer(const StatefulTiledProgram& program);

  // Core lowering: consume StatefulTiledProgram and emit TT-Metal-first builtins
  Stmt LowerComputeRegion(const Stmt& stmt);

  // Resource planning outputs
  Array<Any> EncodeCBRequirements() const;
  Array<Any> EncodeStateClasses() const;

  // dst register layout: static allocation of state → dst offset
  // References TT-Metal SDPA ground truth layout:
  //   mm2_dst_offset = 0 (output accumulator)
  //   max_dst_offset = mm2_dst_offset + packed_tile_size * num_tiles_v
  //   sum_dst_offset = max_dst_offset + 2
  //   corr_exp_dst_offset = max_dst_offset + packed_tile_size
  //   mm1_dst_offset = corr_exp_dst_offset + packed_tile_size
  Map<String, Integer> PlanDstRegisterLayout() const;

  // Carry strategy: decide register-resident vs CB-round-trip per state
  // Based on total dst space pressure from all carry states
  Map<String, String> DecideCarryStrategy() const;
};

Stmt LowerBlackholeOps::VisitStmt_(const ForNode* op) {
  if (stateful_program_.defined()) {
    if (auto lowered = stateful_program_lowerer_.TryLower(op)) {
      return lowered.value();
    }
  }
  return StmtExprMutator::VisitStmt_(op);
}
```

- [ ] **Step 4: Add tile/stats-state builtins and planner resource classes**

```cpp
TVM_REGISTER_OP("tl.blackhole.stats_reduce_row");
TVM_REGISTER_OP("tl.blackhole.tile_apply_stats");
TVM_REGISTER_OP("tl.blackhole.stats_update");
```

CB resource class 从 State kind + lifetime 推导，不从 role 字符串推导：
```cpp
// Derive CB resource class from StatefulTiledProgram State kind
for (const auto& state : program->states) {
  if (state->kind == "matrix_state" && state->lifetime == "carry") {
    // Carry strategy decides: register-resident → no CB needed;
    // CB-round-trip → ping-pong CB pair (like SDPA's cb_prev_mm_out/cb_cur_mm_out)
    auto strategy = carry_strategies[state->name];
    if (strategy == "cb_round_trip") {
      config.resource_class = "state_ping_pong";
    }
    // register-resident carry: only needs output CB at final step
  } else if (state->kind == "vector_state" && state->lifetime == "carry") {
    config.resource_class = "stats_scratch";
  } else if (state->lifetime == "ephemeral") {
    config.resource_class = "tile_scratch";
  } else {
    config.resource_class = "transport";
  }
}
```

- [ ] **Step 5: Update codegen/runtime extraction to consume only the frozen target-program contract**

```cpp
if (call->op.same_as(builtin::tl_blackhole_stats_reduce_row())) {
  EmitStatsReduceRow(call);
  return;
}
```

```cpp
ICHECK(attrs.GetAttr<runtime::Array<runtime::String>>("blackhole.state_classes").defined())
    << "Blackhole lowering now requires frozen state class schema";
```

- [ ] **Step 6: Run compile-path regressions**

Run:

```bash
cmake --build /root/dev/vibe_dsl/tilelang_repo/build -j32
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -k "tile_scratch_not_fragment_array or multi_tile_output"
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -k "stateful_tile_stats_builtins or gqa"
```

Expected:

```text
BUILD SUCCESS
... passed
```

- [ ] **Step 7: Commit**

```bash
git add tilelang_repo/src/transform/lower_blackhole_stateful_tiled_program.h \
        tilelang_repo/src/transform/lower_blackhole_stateful_tiled_program.cc \
        tilelang_repo/src/transform/lower_blackhole_ops.h \
        tilelang_repo/src/transform/lower_blackhole_ops.cc \
        tilelang_repo/src/transform/plan_blackhole_cb.cc \
        tilelang_repo/src/tir/builtin_blackhole.h \
        tilelang_repo/src/tir/builtin_blackhole.cc \
        tilelang_repo/src/target/codegen_blackhole.cc \
        tilelang_repo/src/target/rt_mod_blackhole.cc \
        tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py \
        tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py
git commit -m "blackhole: lower stateful tiled ir through compute path"
```

### Task 5: End-To-End Verification, TT-Sim Runtime, And Documentation Sync [Phase 1b]

**Files:**
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py`
- Modify: `tasks/dev_design/final_blackhole_backend_redesign.md`
- Modify: `tasks/progress.md`
- Modify: `memory/general_dev.md`
- Modify: `memory/bugs.md`

- [ ] **Step 1: Add runtime tests that lock the new contract in place**

```python
def test_flash_attention_runtime_uses_stateful_tiled_ir_contract():
    mod = compile_flash_attn_for_blackhole(...)
    assert "tl.stateful_tiled_program" in str(mod.ir_mod["main"])
    out = mod(...)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
```

- [ ] **Step 2: Run the runtime test once to verify the pre-fix failure mode**

Run in one shell:

```bash
source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh
export TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo
cd /root/dev/vibe_dsl/tilelang_repo
pytest -q -rs testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py -k "mha or gqa"
```

Expected:

```text
FAIL ... numerical mismatch or missing stateful_tiled_program contract
```

- [ ] **Step 3: Run the full regression after implementation**

Run in one shell:

```bash
source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh
export TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo
cd /root/dev/vibe_dsl/tilelang_repo
pytest -q testing/python/transform/test_stateful_tiled_ir.py
pytest -q testing/python/transform/test_blackhole_flash_attention_analysis.py
pytest -q testing/python/target/blackhole/test_blackhole_gemm.py
pytest -q testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py
pytest -q -rs testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py
```

Expected:

```text
all selected tests passed
```

- [ ] **Step 4: Sync docs and memory with the implemented architecture**

```markdown
- `final_blackhole_backend_redesign.md`: 把 `Stateful Tiled IR` 从“下一阶段方向”更新为“已落地基础设施 + 当前覆盖面”
- `progress.md`: 当前 blocker / 下一步改成 routed/selection/paged 语义扩展
- `memory/general_dev.md`: 记录 TT-Sim 运行方式、`Lift -> Validate -> Lower` 调试顺序
- `memory/bugs.md`: 记录 `blackhole.acc` 混合语义问题如何被 `Stateful Tiled IR` 消灭
```

- [ ] **Step 5: Commit**

```bash
git add tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_runtime.py \
        tasks/dev_design/final_blackhole_backend_redesign.md \
        tasks/progress.md \
        memory/general_dev.md \
        memory/bugs.md
git commit -m "docs: close out stateful tiled ir phase 1"
```

## Self-Review Checklist

- Spec coverage:
  - `Stateful Tiled IR` 核心对象 → Task 1
  - 现有 DSL/TIR 自动 lift 边界 → Task 2
  - 验证与 fail-fast → Task 3
  - Blackhole target-program lowering → Task 4
  - runtime / docs / memory 收口 → Task 5
- Placeholder scan:
  - 无 `TODO/TBD/implement later`
  - 每个任务都有明确文件、代码片段、命令和 commit
- Type consistency:
  - 统一使用 `StatefulTiledProgram`
  - pass 名统一为 `LiftToStatefulTiledIR` / `ValidateStatefulTiledIR`
  - runtime attr 统一写作 `tl.stateful_tiled_program`

## Follow-On Plans After Phase 1

Phase 1 完成后，再单独起计划处理：

1. routed / segmented / MoE lifting and lowering
2. selection / topk state-machine lifting and lowering
3. paged / page-table / sparse MLA lifting and lowering
4. 面向用户的最小 annotation/helper API
