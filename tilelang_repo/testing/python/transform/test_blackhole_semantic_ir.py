import sys
from pathlib import Path

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.engine.phase import LowerAndLegalize, OptimizeForTarget
from tvm.target import Target

THIS_DIR = Path(__file__).resolve().parent
BLACKHOLE_TARGET_TEST_DIR = THIS_DIR.parent / "target" / "blackhole"
TOPK_EXAMPLE_DIR = THIS_DIR.parents[2] / "examples" / "topk"
GDN_EXAMPLE_DIR = THIS_DIR.parents[2] / "examples" / "gdn"
if str(BLACKHOLE_TARGET_TEST_DIR) not in sys.path:
    sys.path.append(str(BLACKHOLE_TARGET_TEST_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))
if str(TOPK_EXAMPLE_DIR) not in sys.path:
    sys.path.append(str(TOPK_EXAMPLE_DIR))
if str(GDN_EXAMPLE_DIR) not in sys.path:
    sys.path.append(str(GDN_EXAMPLE_DIR))

from common import gemm_kernel, staged_copy_kernel
import example_topk
import example_chunk_delta_h as chunk_delta_h_example
from test_blackhole_flash_attention_analysis import (
    _lower_flash_attention_example,
    gqa_example,
    mha_example,
)


@T.prim_func
def _stage0_seed_kernel(
    a: T.Buffer((16,), "float32"),
    b: T.Buffer((16,), "float32"),
):
    with T.Kernel(1, threads=32):
        for i in T.serial(16):
            b[i] = a[i]


def _prepare_blackhole_stage0_module():
    mod = tvm.IRModule({"main": _stage0_seed_kernel.with_attr("global_symbol", "main")})
    mod = tvm.tir.transform.BindTarget(Target("blackhole"))(mod)
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    return mod


def _prepare_blackhole_phase_a_module(prim_func):
    mod = tvm.IRModule({"main": prim_func.with_attr("global_symbol", "main")})
    target = Target("blackhole")
    with target:
        if not (mod["main"].attrs and mod["main"].attrs.get("target") is not None):
            mod = LowerAndLegalize(mod, target)
        mod = OptimizeForTarget(mod, target)
    mod = tilelang.transform.LowerDeviceStorageAccessInfo()(mod)
    mod = tilelang.transform.LowerIntrin()(mod)
    mod = tvm.tir.transform.Simplify()(mod)
    mod = tilelang.transform.HoistBroadcastValues()(mod)
    mod = tilelang.transform.SplitBlackholeKernel()(mod)
    mod = tilelang.transform.AnalyzeBlackholeWorkDecomposition()(mod)
    mod = tilelang.transform.AnalyzeBlackholeFragmentRegions()(mod)
    mod = tilelang.transform.AnalyzeBlackholePipelineStages()(mod)
    mod = tilelang.transform.AnalyzeSemanticStructure()(mod)
    mod = tilelang.transform.LiftStatefulSemanticIR()(mod)
    mod = tilelang.transform.ValidateStatefulSemanticIR()(mod)
    return mod


def _lift_semantic_program_from_existing_structure(prim_func):
    mod = tvm.IRModule({"main": prim_func.with_attr("global_symbol", "main")})
    mod = tilelang.transform.AnalyzeSemanticStructure()(mod)
    mod = tilelang.transform.LiftStatefulSemanticIR()(mod)
    mod = tilelang.transform.ValidateStatefulSemanticIR()(mod)
    return mod


def test_device_program_registry_is_collected_before_split_host_device():
    mod = _prepare_blackhole_stage0_module()
    mod = tilelang.transform.CollectDevicePrograms()(mod)

    registry = mod.global_infos["tl.device_programs"]
    assert len(registry) == 1
    assert registry[0].root_symbol == "main"
    assert list(registry[0].member_funcs) == ["main_kernel"]


def test_semantic_seeds_are_projected_before_semantic_lift():
    mod = _prepare_blackhole_stage0_module()
    mod = tilelang.transform.ProjectSemanticSeeds()(mod)

    seeds = mod["main"].attrs["tl.semantic_seeds"]
    assert list(seeds["device_kernel_regions"]) == ["main_kernel"]
    assert list(seeds["capture_kinds"]) == ["device_program_membership"]
    freeze = mod["main"].attrs["tl.semantic_hard_freeze"]
    assert str(freeze["unsafe_mutation_policy"]) == "invalidate_companion_programs"


def test_hard_freeze_invalidates_companion_programs_after_unsafe_mutation():
    mod = _prepare_blackhole_stage0_module()
    mod = tilelang.transform.ProjectSemanticSeeds()(mod)
    main = mod["main"].with_attr("tl.semantic_program", {"frozen": True})
    main = main.with_attr("tl.spatial_program", {"frozen": True})
    main = main.with_attr("tl.tt_program", {"frozen": True})
    mod.update_func(mod.get_global_var("main"), main)

    mod = tilelang.transform.InvalidateBlackholeCompanionPrograms("unit_test_unsafe_mutation")(mod)

    attrs = mod["main"].attrs
    assert "tl.semantic_program" not in attrs
    assert "tl.spatial_program" not in attrs
    assert "tl.tt_program" not in attrs
    assert str(attrs["tl.companion_invalidation_reason"]) == "unit_test_unsafe_mutation"


def test_copy_semantic_program_lifts_minimal_domain_and_map_update():
    mod = _prepare_blackhole_phase_a_module(staged_copy_kernel(tile_rows=1, tile_cols=1))

    program = mod["main"].attrs["tl.semantic_program"]
    assert len(program.domains) == 1
    assert str(program.domains[0].name) == "device_program"
    assert len(program.updates) >= 1
    assert {str(update.law.kind) for update in program.updates} == {"map"}


def test_gemm_semantic_program_lifts_fragment_state_and_map_update():
    mod = _prepare_blackhole_phase_a_module(gemm_kernel())

    program = mod["main"].attrs["tl.semantic_program"]
    state_names = {str(state.name) for state in program.states}
    assert "C_local" in state_names
    assert any(str(state.role) == "transient" for state in program.states)
    assert "map" in {str(update.law.kind) for update in program.updates}


def test_flash_attention_semantic_program_lifts_carry_state_and_reduce_updates():
    mod = _prepare_blackhole_phase_a_module(
        _lower_flash_attention_example(
            mha_example,
            1,
            32,
            256,
            128,
            False,
            block_M=128,
            block_N=128,
            num_stages=1,
            threads=128,
        )
    )

    program = mod["main"].attrs["tl.semantic_program"]
    state_names = {str(state.name) for state in program.states}
    assert {"scores_max", "logsum", "acc_o"}.issubset(state_names)
    assert "reduce" in {str(update.law.kind) for update in program.updates}
    assert "carry" in {str(state.role) for state in program.states}


def test_flash_attention_gqa_semantic_program_lifts_fragment_state_subset():
    mod = _prepare_blackhole_phase_a_module(
        _lower_flash_attention_example(
            gqa_example,
            1,
            16,
            1024,
            128,
            False,
            groups=16,
            block_M=64,
            block_N=64,
            num_stages=2,
            threads=128,
        )
    )

    program = mod["main"].attrs["tl.semantic_program"]
    state_names = {str(state.name) for state in program.states}
    assert {"acc_s", "acc_s_cast", "scores_sum"}.issubset(state_names)
    assert len(program.domains) == 1


def test_topk_semantic_program_lifts_select_updates_and_selection_roles():
    mod = _prepare_blackhole_phase_a_module(
        example_topk.tl_topk.jit_impl.get_tir(M=64, N=32, topk=4, blk_m=64, threads=128)
    )

    program = mod["main"].attrs["tl.semantic_program"]
    law_kinds = {str(update.law.kind) for update in program.updates}
    state_roles = {str(state.role) for state in program.states}

    assert "select" in law_kinds
    assert "selection_state" in state_roles
    assert "index_state" in state_roles


@T.prim_func
def _synthetic_selection_without_name_hints(
    logits: T.Buffer((16,), "float32"),
    out_values: T.Buffer((4,), "float32"),
    out_slots: T.Buffer((4,), "int32"),
):
    T.func_attr(
        {
            "global_symbol": "main",
            "target": tvm.target.Target("blackhole"),
            "blackhole.fragment_regions": [
                {
                    "fragment_buffers": [
                        {"name": "score_fragment", "scope": "blackhole.acc", "is_integer": 0},
                        {"name": "carry_slots", "scope": "blackhole.acc", "is_integer": 1},
                        {"name": "best_value", "scope": "blackhole.acc", "is_integer": 0},
                        {"name": "best_slot", "scope": "blackhole.acc", "is_integer": 1},
                    ],
                    "ops": ["pointwise_chain", "row_reduction", "row_broadcast"],
                    "pointwise_ops": ["if_then_else", "max"],
                    "row_reductions": [
                        {"target": "best_value", "kind": "max"},
                        {"target": "best_slot", "kind": "max"},
                    ],
                    "row_broadcasts": [{"source": "score_fragment"}, {"source": "best_value"}],
                    "selection_targets": ["best_slot"],
                    "update_sources": [
                        {"target": "best_value", "sources": ["score_fragment"]},
                        {"target": "best_slot", "sources": ["score_fragment", "carry_slots"]},
                    ],
                    "loop_carried_state": [
                        {"name": "score_fragment"},
                        {"name": "carry_slots"},
                        {"name": "best_value"},
                        {"name": "best_slot"},
                    ],
                }
            ],
        }
    )
    with T.Kernel(1, threads=32):
        score_fragment = T.decl_buffer((16,), "float32", scope="blackhole.acc")
        carry_slots = T.decl_buffer((16,), "int32", scope="blackhole.acc")
        best_value = T.decl_buffer((4,), "float32", scope="blackhole.acc")
        best_slot = T.decl_buffer((4,), "int32", scope="blackhole.acc")
        tx = T.launch_thread("threadIdx.x", 32)

        if tx < 16:
            score_fragment[tx] = logits[tx]
            carry_slots[tx] = tx
        if tx < 4:
            best_value[tx] = T.max(score_fragment[tx * 4], score_fragment[tx * 4 + 1])
            best_slot[tx] = T.if_then_else(
                score_fragment[tx * 4] >= score_fragment[tx * 4 + 1],
                carry_slots[tx * 4],
                carry_slots[tx * 4 + 1],
            )
            out_values[tx] = best_value[tx]
            out_slots[tx] = best_slot[tx]


def test_topk_semantic_program_recovers_index_state_from_integer_ir_not_names():
    mod = _lift_semantic_program_from_existing_structure(_synthetic_selection_without_name_hints)

    program = mod["main"].attrs["tl.semantic_program"]
    state_roles_by_name = {str(state.name): str(state.role) for state in program.states}
    select_updates = {
        str(update.name): [str(state) for state in update.law.source_states]
        for update in program.updates
        if str(update.law.kind) == "select"
    }

    assert state_roles_by_name["best_slot"] == "index_state"
    assert state_roles_by_name["best_value"] == "reduction_accumulator"
    assert "carry_slots" in select_updates["select_best_slot"]


def test_selection_pairing_is_recovered_from_compute_pattern():
    mod = _prepare_blackhole_phase_a_module(
        example_topk.tl_topk.jit_impl.get_tir(M=64, N=32, topk=4, blk_m=64, threads=128)
    )

    program = mod["main"].attrs["tl.semantic_program"]
    selection_pairs = {
        str(update.name): {str(binding.kind): str(binding.value_repr) for binding in update.bindings}
        for update in program.updates
        if str(update.law.kind) == "select"
    }

    assert any("paired_value_state" in bindings for bindings in selection_pairs.values())


def test_chunk_recurrence_semantic_program_lifts_recurrence_updates():
    mod = _prepare_blackhole_phase_a_module(
        chunk_delta_h_example.tilelang_chunk_gated_delta_rule_fwd_h.jit_impl.get_tir(
            B=1,
            S=64,
            H=4,
            DK=32,
            DV=32,
            input_dtype=T.float16,
            output_dtype=T.float16,
            accum_dtype=T.float32,
            gate_dtype=T.float16,
            state_dtype=T.float32,
            chunk_size=32,
            use_g=True,
            use_initial_state=True,
            store_final_state=True,
            save_new_value=True,
            block_DK=32,
            block_DV=32,
            threads=128,
            num_stages=1,
        )
    )

    program = mod["main"].attrs["tl.semantic_program"]
    law_kinds = {str(update.law.kind) for update in program.updates}
    state_roles = {str(state.role) for state in program.states}
    recurrence_sources = [
        (str(update.state_name), [str(state) for state in update.law.source_states])
        for update in program.updates
        if str(update.law.kind) == "recurrence"
    ]

    assert "recurrence" in law_kinds
    assert "carry" in state_roles
    assert any(
        any(source != target for source in sources) for target, sources in recurrence_sources
    )


def test_flash_attention_semantic_program_separates_algorithmic_state_from_transient_scratch():
    mod = _prepare_blackhole_phase_a_module(
        _lower_flash_attention_example(
            mha_example,
            1,
            32,
            256,
            128,
            False,
            block_M=128,
            block_N=128,
            num_stages=1,
            threads=128,
        )
    )

    program = mod["main"].attrs["tl.semantic_program"]
    state_roles = {str(state.role) for state in program.states}

    assert "carry" in state_roles
    assert "reduction_accumulator" in state_roles
    assert "transient" in state_roles
