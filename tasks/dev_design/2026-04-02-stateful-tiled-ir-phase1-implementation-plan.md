# Stateful Tiled IR Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在尽量不改用户 Python DSL 主体写法的前提下，引入 compiler-internal `Stateful Tiled IR`，并让当前 Blackhole 的 GEMM / online-softmax / flash-attn compute 主链正式迁到这层语义之上。

**Architecture:** 先在 compiler 内部新增 `Domain / State / Relation / Op / Phase` 五类对象与 `LiftToStatefulTiledIR / ValidateStatefulTiledIR` 两个通用 pass；再把现有 Blackhole analysis pass 的“算法理解”部分前移到这层 IR；最后让 Blackhole target-specific lowering 只消费这层已冻结的语义并映射到 TT-Metal-first `CB / tile / dst-reg / stats-state` 协议。

**Tech Stack:** TileLang Python DSL, TVM PrimFunc/TIR, TVM ObjectRef/ObjectNode, TileLang transform passes, Blackhole lowering/codegen/runtime, pytest, CMake, TT-Sim

---

## Scope Note

这份计划只覆盖 **Phase 1**：

- 引入 compiler-internal `Stateful Tiled IR`
- 打通 dense + carry/state-coupled 形态
- 迁移当前 Blackhole GEMM / online-softmax / flash-attn 主链
- 为后续 routed / segmented / selection / paged 语义留接口

**不在本计划内完整落地**：

- MoE routed/segmented lifting 的完整 bring-up
- topk / argmax selection state machine 的完整 lowering
- paged decode / sparse MLA 的完整 page-table lowering
- 面向用户的公开 annotation API 扩展

这些内容在 Phase 1 完成后另起后续实施计划。

## File Map

- Create: `tilelang_repo/src/transform/common/stateful_tiled_ir.h`
  - 定义 `StatefulTiledProgram / Domain / State / Relation / Op / Phase` ObjectNode/ObjectRef
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

### Task 1: Introduce `Stateful Tiled IR` Core Objects And Pass Entrypoints

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

```cpp
class StatefulTiledStateNode : public Object {
 public:
  String name;
  String kind;
  String lifetime;
  Optional<Buffer> backing_buffer;
  TVM_DECLARE_FINAL_OBJECT_INFO(StatefulTiledStateNode, Object);
};

class StatefulTiledProgramNode : public Object {
 public:
  Array<ObjectRef> domains;
  Array<ObjectRef> states;
  Array<ObjectRef> relations;
  Array<ObjectRef> ops;
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

### Task 2: Lift Dense / Carry Semantics Into `Stateful Tiled IR`

**Files:**
- Modify: `tilelang_repo/src/transform/lift_stateful_tiled_ir.cc`
- Modify: `tilelang_repo/src/transform/analyze_blackhole_work_decomposition.cc`
- Modify: `tilelang_repo/src/transform/analyze_blackhole_fragment_regions.cc`
- Modify: `tilelang_repo/src/transform/analyze_blackhole_pipeline_stages.cc`
- Modify: `tilelang_repo/testing/python/transform/test_stateful_tiled_ir.py`
- Modify: `tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py`

- [ ] **Step 1: Add failing structural expectations for flash-attn and online-softmax**

```python
def test_flash_attention_lift_distinguishes_tile_and_vector_state():
    mod = _lift_stateful_tiled_ir(_make_flash_attn_program())
    body = str(mod["main"])
    assert "tile_state" in body
    assert "vector_state" in body
    assert "carried_across" in body


def test_online_softmax_lift_emits_reduce_apply_update_chain():
    mod = _lift_stateful_tiled_ir(_make_online_softmax_program())
    body = str(mod["main"])
    assert "state_reduce" in body
    assert "tile_apply_state" in body
    assert "state_update" in body
```

- [ ] **Step 2: Run the transform tests to verify they fail**

Run:

```bash
pytest -q tilelang_repo/testing/python/transform/test_stateful_tiled_ir.py -k "tile_and_vector_state or reduce_apply_update_chain"
```

Expected:

```text
FAIL ... expected tile_state/vector_state/state_update markers not found
```

- [ ] **Step 3: Implement lift from existing analysis attrs instead of re-parsing late loop shape**

```cpp
StatefulTiledProgram program;
program->domains.push_back(MakeDenseDomain(work_attr));

for (const auto& region : fragment_regions) {
  if (region_role == "matmul_output_tile") {
    program->states.push_back(MakeTileState(region_name, "carry"));
  } else if (region_role == "stats_reduce_row") {
    program->states.push_back(MakeVectorState(region_name, "carry"));
    program->relations.push_back(MakeReducedFrom(region_name, source_tile, "row"));
  }
}

for (const auto& stage : pipeline_stages) {
  program->phases.push_back(MakePhase(stage_name, live_in, live_out));
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
pytest -q tilelang_repo/testing/python/transform/test_blackhole_flash_attention_analysis.py -k "stateful or fragment"
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

### Task 3: Validate Ambiguous Semantics Early And Insert The New Passes Into The Pipeline

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
if (state->kind.empty()) {
  LOG(FATAL) << "StatefulTiledIR validation failed: missing state kind for " << state->name;
}
if (relation->type == "scatters_to" && relation->combine.empty()) {
  LOG(FATAL) << "StatefulTiledIR validation failed: scatter relation missing combine rule";
}
if (UsesBlackholeAccAsBothTileAndVector(program)) {
  LOG(FATAL) << "StatefulTiledIR validation failed: blackhole.acc mixed tile/vector semantics";
}
```

```python
device_mod = tilelang.transform.AnalyzeBlackholePipelineStages()(device_mod)
device_mod = tilelang.transform.LiftToStatefulTiledIR()(device_mod)
device_mod = tilelang.transform.ValidateStatefulTiledIR()(device_mod)
device_mod = tilelang.transform.LowerBlackholeOps()(device_mod)
```

- [ ] **Step 4: Run the focused pass-pipeline regression**

Run:

```bash
pytest -q tilelang_repo/testing/python/transform/test_stateful_tiled_ir.py
pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py -k "mixed_acc_semantics or stateful"
```

Expected:

```text
... passed
```

- [ ] **Step 5: Commit**

```bash
git add tilelang_repo/src/transform/validate_stateful_tiled_ir.cc \
        tilelang_repo/tilelang/engine/lower.py \
        tilelang_repo/testing/python/transform/test_stateful_tiled_ir.py \
        tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py
git commit -m "transform: validate stateful tiled ir before blackhole lowering"
```

### Task 4: Migrate Blackhole Compute Lowering To Consume `Stateful Tiled IR`

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

- [ ] **Step 3: Implement a dedicated stateful-program lowering helper and shrink `LowerBlackholeOps`**

```cpp
class BlackholeStatefulProgramLowerer {
 public:
  explicit BlackholeStatefulProgramLowerer(const StatefulTiledProgram& program);
  Stmt LowerComputeRegion(const Stmt& stmt);
  Array<Any> EncodeCBRequirements() const;
  Array<Any> EncodeStateClasses() const;
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

```cpp
if (requirement.role == "tile_scratch") {
  config.resource_class = "tile_scratch";
} else if (requirement.role == "stats_scratch") {
  config.resource_class = "stats_scratch";
} else {
  config.resource_class = "transport";
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

### Task 5: End-To-End Verification, TT-Sim Runtime, And Documentation Sync

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
