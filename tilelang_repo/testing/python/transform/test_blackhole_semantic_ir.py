import sys
from pathlib import Path

import pytest
import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.engine.phase import LowerAndLegalize, OptimizeForTarget
from tvm import tir
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
    mod = tilelang.transform.ValidateSemanticRefinement()(mod)
    return mod


def _prepare_blackhole_semantic_witness_module(prim_func):
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
    return mod


def _prepare_blackhole_phase_a_refined_module(prim_func):
    return _prepare_blackhole_phase_a_module(prim_func)


def _prepare_blackhole_fragment_analysis_module(prim_func):
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
    return mod


def _lift_semantic_program_from_existing_structure(prim_func):
    mod = tvm.IRModule({"main": prim_func.with_attr("global_symbol", "main")})
    mod = tilelang.transform.AnalyzeSemanticStructure()(mod)
    mod = tilelang.transform.LiftStatefulSemanticIR()(mod)
    mod = tilelang.transform.ValidateStatefulSemanticIR()(mod)
    mod = tilelang.transform.ValidateSemanticRefinement()(mod)
    return mod


def _append_noop_to_main_body(mod):
    func = mod["main"]
    body = func.body
    if isinstance(body, tir.SeqStmt):
        new_body = tir.SeqStmt(list(body.seq) + [tir.Evaluate(0)])
    else:
        new_body = tir.SeqStmt([body, tir.Evaluate(0)])
    mod.update_func(mod.get_global_var("main"), func.with_body(new_body))
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
    freeze = attrs["tl.semantic_hard_freeze"]
    assert str(freeze["contract_mode"]) == "invalidate"
    assert str(freeze["state"]) == "invalidated"


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


@T.prim_func
def _synthetic_selection_pair_without_integer_hints(
    logits: T.Buffer((16,), "float32"),
    out_values: T.Buffer((4,), "float32"),
    out_slots: T.Buffer((4,), "float32"),
):
    T.func_attr(
        {
            "global_symbol": "main",
            "target": tvm.target.Target("blackhole"),
            "blackhole.fragment_regions": [
                {
                    "fragment_buffers": [
                        {"name": "score_fragment", "scope": "blackhole.acc", "is_integer": 0},
                        {"name": "carry_slots", "scope": "blackhole.acc", "is_integer": 0},
                        {"name": "best_value", "scope": "blackhole.acc", "is_integer": 0},
                        {"name": "best_slot", "scope": "blackhole.acc", "is_integer": 0},
                    ],
                    "ops": ["pointwise_chain", "row_reduction", "row_broadcast"],
                    "pointwise_ops": ["if_then_else", "max"],
                    "row_reductions": [
                        {"target": "best_value", "kind": "max"},
                    ],
                    "row_broadcasts": [{"source": "score_fragment"}, {"source": "best_value"}],
                    "selection_targets": ["best_slot"],
                    "selection_pairs": [
                        {
                            "value_target": "best_value",
                            "companion_target": "best_slot",
                            "source_states": ["score_fragment"],
                        }
                    ],
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
        carry_slots = T.decl_buffer((16,), "float32", scope="blackhole.acc")
        best_value = T.decl_buffer((4,), "float32", scope="blackhole.acc")
        best_slot = T.decl_buffer((4,), "float32", scope="blackhole.acc")
        tx = T.launch_thread("threadIdx.x", 32)

        if tx < 16:
            score_fragment[tx] = logits[tx]
            carry_slots[tx] = T.Cast("float32", tx)
        if tx < 4:
            best_value[tx] = T.max(score_fragment[tx * 4], score_fragment[tx * 4 + 1])
            best_slot[tx] = T.if_then_else(
                score_fragment[tx * 4] >= score_fragment[tx * 4 + 1],
                carry_slots[tx * 4],
                carry_slots[tx * 4 + 1],
            )
            out_values[tx] = best_value[tx]
            out_slots[tx] = best_slot[tx]


def test_selection_pairing_recovers_index_role_without_integer_hints():
    mod = _lift_semantic_program_from_existing_structure(
        _synthetic_selection_pair_without_integer_hints
    )

    program = mod["main"].attrs["tl.semantic_program"]
    state_roles_by_name = {str(state.name): str(state.role) for state in program.states}

    assert state_roles_by_name["best_slot"] == "index_state"


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


def test_topk_fragment_analysis_recovers_arg_reduce_targets():
    mod = _prepare_blackhole_fragment_analysis_module(
        example_topk.tl_topk.jit_impl.get_tir(M=64, N=32, topk=4, blk_m=64, threads=128)
    )

    fragment_regions = mod["main"].attrs["blackhole.fragment_regions"]
    arg_reduce_targets = {
        str(target)
        for region in fragment_regions
        for target in region["arg_reduce_targets"]
    }

    assert "max_idx" in arg_reduce_targets


def test_topk_semantic_witnesses_expose_generic_fact_axes():
    mod = _prepare_blackhole_semantic_witness_module(
        example_topk.tl_topk.jit_impl.get_tir(M=64, N=32, topk=4, blk_m=64, threads=128)
    )

    witnesses = mod["main"].attrs["tl.semantic_witnesses"]
    fact_axes = {(str(w.subject_kind), str(w.fact_axis)) for w in witnesses}

    assert ("state", "role") in fact_axes
    assert ("update", "law_family") in fact_axes
    assert ("update", "source_set") in fact_axes
    assert ("relation", "companion") in fact_axes
    assert ("relation", "derives_index_from") in fact_axes


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


def test_chunk_recurrence_edges_are_recovered_from_compute_pattern():
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
    recurrence_bindings = [
        {str(binding.kind): str(binding.value_repr) for binding in update.bindings}
        for update in program.updates
        if str(update.law.kind) == "recurrence"
    ]

    assert any("recurrence_source_state" in bindings for bindings in recurrence_bindings)


def test_chunk_recurrence_semantic_witnesses_capture_generic_carried_facts():
    mod = _prepare_blackhole_semantic_witness_module(
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

    witnesses = mod["main"].attrs["tl.semantic_witnesses"]
    fact_axes = {(str(w.subject_kind), str(w.fact_axis)) for w in witnesses}

    assert ("update", "ordering") in fact_axes
    assert ("relation", "carried_from") in fact_axes


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


def test_refinement_validator_rejects_orphan_relation_witness():
    mod = _prepare_blackhole_phase_a_module(staged_copy_kernel(tile_rows=1, tile_cols=1))

    make_witness = tvm.get_global_func("tl.SemanticWitness")
    orphan = make_witness(
        "relation",
        "orphan_anchor",
        "companion",
        {"kind": "companion"},
        ["missing_anchor"],
        ["unit_test"],
        "post_analyze",
    )
    main = mod["main"].with_attr("tl.semantic_witnesses", [orphan])
    mod.update_func(mod.get_global_var("main"), main)

    with pytest.raises(tvm.TVMError):
        tilelang.transform.ValidateSemanticRefinement()(mod)


def test_refinement_validator_rejects_body_mutation_without_invalidation():
    mod = _prepare_blackhole_phase_a_refined_module(staged_copy_kernel(tile_rows=1, tile_cols=1))
    mod = _append_noop_to_main_body(mod)

    with pytest.raises(tvm.TVMError):
        tilelang.transform.ValidateSemanticRefinement()(mod)


def test_invalidation_contract_clears_semantic_companions_after_unsafe_mutation():
    mod = _prepare_blackhole_phase_a_refined_module(staged_copy_kernel(tile_rows=1, tile_cols=1))
    mod = _append_noop_to_main_body(mod)
    mod = tilelang.transform.InvalidateBlackholeCompanionPrograms("unit_test_body_mutation")(mod)

    attrs = mod["main"].attrs
    assert "tl.semantic_structure" not in attrs
    assert "tl.semantic_witnesses" not in attrs
    assert "tl.semantic_program" not in attrs
    assert "tl.spatial_program" not in attrs
    assert "tl.tt_program" not in attrs
    assert str(attrs["tl.companion_invalidation_reason"]) == "unit_test_body_mutation"
    freeze = attrs["tl.semantic_hard_freeze"]
    assert str(freeze["contract_mode"]) == "invalidate"
    assert str(freeze["state"]) == "invalidated"

    tilelang.transform.ValidateSemanticRefinement()(mod)


def test_semantic_vocab_normalizes_known_closed_vocabulary():
    normalize_binding_kind = tvm.get_global_func("tl.SemanticVocabNormalizeBindingKind")
    normalize_contract_mode = tvm.get_global_func("tl.SemanticVocabNormalizeContractMode")
    normalize_subject_kind = tvm.get_global_func("tl.SemanticVocabNormalizeWitnessSubjectKind")

    assert str(normalize_binding_kind("paired_value_state")) == "paired_value_state"
    assert str(normalize_contract_mode("preserve")) == "preserve"
    assert str(normalize_subject_kind("relation")) == "relation"


def test_semantic_vocab_rejects_unknown_closed_vocabulary_symbol():
    normalize_binding_kind = tvm.get_global_func("tl.SemanticVocabNormalizeBindingKind")

    with pytest.raises(tvm.TVMError):
        normalize_binding_kind("definitely_not_a_binding_kind")


def test_semantic_payload_normalizes_known_typed_payload_families():
    normalize_state_role = tvm.get_global_func("tl.SemanticPayloadNormalizeStateRole")
    normalize_law_family = tvm.get_global_func("tl.SemanticPayloadNormalizeUpdateLawFamily")
    normalize_source_set = tvm.get_global_func("tl.SemanticPayloadNormalizeSourceSet")
    normalize_relation_binding = tvm.get_global_func("tl.SemanticPayloadNormalizeRelationBinding")

    role_payload = normalize_state_role({"role": "selection_state"})
    law_payload = normalize_law_family({"kind": "select"})
    source_payload = normalize_source_set({"sources": ["scores", "carry"]})
    binding_payload = normalize_relation_binding({"binding_kind": "paired_value_state"})

    assert str(role_payload["role"]) == "selection_state"
    assert str(law_payload["kind"]) == "select"
    assert list(source_payload["sources"]) == ["scores", "carry"]
    assert str(binding_payload["binding_kind"]) == "paired_value_state"


def test_semantic_payload_rejects_malformed_payload_shape():
    normalize_source_set = tvm.get_global_func("tl.SemanticPayloadNormalizeSourceSet")
    normalize_relation_binding = tvm.get_global_func("tl.SemanticPayloadNormalizeRelationBinding")

    with pytest.raises(tvm.TVMError):
        normalize_source_set({"sources": "not_an_array"})

    with pytest.raises(tvm.TVMError):
        normalize_relation_binding({"binding_kind": "not_a_binding_kind"})


def test_semantic_program_exposes_state_effect_graph():
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
    assert len(program.state_versions) > 0
    assert len(program.state_defs) > 0
    assert len(program.state_uses) > 0
    assert len(program.state_joins) > 0
    assert any(str(join.kind) == "loop_carried" for join in program.state_joins)


def test_refinement_validator_rejects_missing_loop_carried_join_for_carry_state():
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
    rebuilt_program = tvm.get_global_func("tl.SemanticProgram")(
        list(program.domains),
        list(program.states),
        list(program.updates),
        list(program.supplements),
        list(program.seeds),
        list(program.anchors),
        list(program.state_versions),
        list(program.state_defs),
        list(program.state_uses),
        [],
    )
    main = mod["main"].with_attr("tl.semantic_program", rebuilt_program)
    mod.update_func(mod.get_global_var("main"), main)

    with pytest.raises(tvm.TVMError):
        tilelang.transform.ValidateSemanticRefinement()(mod)


def test_typed_rebind_contract_allows_safe_body_refresh():
    mod = _prepare_blackhole_phase_a_refined_module(staged_copy_kernel(tile_rows=1, tile_cols=1))
    mod = _append_noop_to_main_body(mod)
    mod = tilelang.transform.TypedRebindBlackholeCompanionPrograms(
        {
            "reason": "unit_test_safe_rebind",
            "rebind_scope": "body_hash_refresh",
        }
    )(mod)

    attrs = mod["main"].attrs
    freeze = attrs["tl.semantic_hard_freeze"]
    assert str(freeze["contract_mode"]) == "typed_rebind"
    assert int(freeze["rebind_epoch"]) >= 1
    assert str(freeze["rebind_scope"]) == "body_hash_refresh"

    tilelang.transform.ValidateSemanticRefinement()(mod)


def test_typed_rebind_requires_trace_metadata():
    mod = _prepare_blackhole_phase_a_refined_module(staged_copy_kernel(tile_rows=1, tile_cols=1))
    mod = _append_noop_to_main_body(mod)
    main = mod["main"].with_attr(
        "tl.semantic_hard_freeze",
        {
            "state": "lifted_a1",
            "body_hash": str(tvm.ir.structural_hash(mod["main"].body)),
            "contract_mode": "typed_rebind",
            "rebind_epoch": 1,
        },
    )
    mod.update_func(mod.get_global_var("main"), main)

    with pytest.raises(tvm.TVMError):
        tilelang.transform.ValidateSemanticRefinement()(mod)
