import sys
from pathlib import Path

import pytest

import tilelang
from tilelang.engine.lower import lower
from tilelang.engine.phase import LowerAndLegalize
from tilelang.engine.phase import OptimizeForTarget
from tilelang import tvm
from tvm.target import Target

from .common import check_blackhole_codegen_requirements
from .test_blackhole_copy_pipeline import (
    _extract_blackhole_executable_spec,
    _require_blackhole_kernel,
)


EXAMPLE_DIR = Path(__file__).resolve().parents[4] / "examples" / "flash_attention"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.append(str(EXAMPLE_DIR))

import example_mha_fwd_bshd as mha_example
import example_gqa_fwd_bshd as gqa_example


def _lower_flash_attention_through_blackhole_ops(*, is_causal=False):
    target = Target("blackhole")
    mod = tvm.IRModule(
        {
            "main": mha_example.flashattn.jit_impl.get_tir(
                1,
                32,
                256,
                128,
                is_causal,
                block_M=128,
                block_N=128,
                num_stages=1,
                threads=128,
            )
        }
    )
    with target:
        mod = LowerAndLegalize(mod, target)
    mod = tilelang.transform.SplitBlackholeKernel()(mod)
    mod = tilelang.transform.AnalyzeBlackholeWorkDecomposition()(mod)
    mod = tilelang.transform.AnalyzeBlackholeFragmentRegions()(mod)
    mod = tilelang.transform.AnalyzeBlackholePipelineStages()(mod)
    mod = tilelang.transform.LowerBlackholeOps()(mod)
    return mod["main"]


def _run_flash_attention_lower_blackhole_ops(example_module, *args, **kwargs):
    target = Target("blackhole")
    mod = tvm.IRModule({"main": example_module.flashattn.jit_impl.get_tir(*args, **kwargs)})
    with target:
        mod = LowerAndLegalize(mod, target)
    mod = tilelang.transform.SplitBlackholeKernel()(mod)
    mod = tilelang.transform.AnalyzeBlackholeWorkDecomposition()(mod)
    mod = tilelang.transform.AnalyzeBlackholeFragmentRegions()(mod)
    mod = tilelang.transform.AnalyzeBlackholePipelineStages()(mod)
    return tilelang.transform.LowerBlackholeOps()(mod)


def _run_flash_attention_lower_blackhole_ops_after_optimize(example_module, *args, **kwargs):
    target = Target("blackhole")
    mod = tvm.IRModule({"main": example_module.flashattn.jit_impl.get_tir(*args, **kwargs)})
    with target:
        mod = LowerAndLegalize(mod, target)
        mod = OptimizeForTarget(mod, target)
    mod = tilelang.transform.SplitBlackholeKernel()(mod)
    mod = tilelang.transform.AnalyzeBlackholeWorkDecomposition()(mod)
    mod = tilelang.transform.AnalyzeBlackholeFragmentRegions()(mod)
    mod = tilelang.transform.AnalyzeBlackholePipelineStages()(mod)
    return tilelang.transform.LowerBlackholeOps()(mod)


def test_flash_attention_forward_lower_blackhole_ops_emits_generic_lowering_requirements():
    lowered = _lower_flash_attention_through_blackhole_ops()

    attrs = lowered.attrs
    assert "flash_attention_plan" not in attrs
    assert "attention_work_contract" not in attrs

    lowering_requirements = attrs["blackhole.lowering_requirements"]
    assert list(lowering_requirements["work_axes"]) == ["bx", "by", "bz"]
    assert {
        "gemm",
        "pointwise_chain",
    }.issubset(set(lowering_requirements["fragment_op_kinds"]))
    assert "row_broadcast" not in set(lowering_requirements["fragment_op_kinds"])
    assert "row_broadcast_sources" not in lowering_requirements
    assert {"exp2", "mul", "div"}.issubset(
        set(lowering_requirements["pointwise_op_kinds"])
    )
    assert "fill" not in set(lowering_requirements["pointwise_op_kinds"])
    assert "add" not in set(lowering_requirements["pointwise_op_kinds"])
    assert "max" not in set(lowering_requirements["pointwise_op_kinds"])
    assert "cast" not in set(lowering_requirements["pointwise_op_kinds"])


def test_flash_attention_forward_optimized_path_lowers_scores_max_updates():
    lowered = _run_flash_attention_lower_blackhole_ops_after_optimize(
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
    )["main"]
    script = lowered.script()
    lowering_requirements = lowered.attrs["blackhole.lowering_requirements"]

    assert "tl.blackhole.scalar_max" in script
    assert "max" not in set(lowering_requirements["pointwise_op_kinds"])


def test_flash_attention_forward_lower_blackhole_ops_lowers_row_reductions_to_builtins():
    lowered = _lower_flash_attention_through_blackhole_ops()
    script = lowered.script()

    assert "tl.blackhole.reduce_row" in script
    assert "\"max\"" in script
    assert "\"sum\"" in script


def test_flash_attention_forward_optimized_path_lowers_row_reductions_to_builtins():
    lowered = _run_flash_attention_lower_blackhole_ops_after_optimize(
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
    )["main"]
    script = lowered.script()
    lowering_requirements = lowered.attrs["blackhole.lowering_requirements"]

    assert "tl.blackhole.reduce_row" in script
    assert "row_reduction" not in set(lowering_requirements["fragment_op_kinds"])


def test_flash_attention_forward_optimized_path_lowers_acc_o_row_broadcast_updates():
    lowered = _run_flash_attention_lower_blackhole_ops_after_optimize(
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
    )["main"]
    script = lowered.script()

    assert "tl.blackhole.mul_row_bcast" in script
    assert "tl.blackhole.div_row_bcast" in script


def test_flash_attention_forward_optimized_path_lowers_logsum_scalar_fma():
    lowered = _run_flash_attention_lower_blackhole_ops_after_optimize(
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
    )["main"]
    script = lowered.script()

    assert "tl.blackhole.scalar_fma" in script


def test_flash_attention_forward_optimized_path_lowers_scores_exp2_affine_updates():
    lowered = _run_flash_attention_lower_blackhole_ops_after_optimize(
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
    )["main"]
    script = lowered.script()
    lowering_requirements = lowered.attrs["blackhole.lowering_requirements"]

    assert "tl.blackhole.exp2_row_bcast_affine" in script
    assert "tl.blackhole.scalar_exp2_affine" in script
    assert "row_broadcast" not in set(lowering_requirements["fragment_op_kinds"])


def test_flash_attention_forward_optimized_path_lowers_fragment_fills():
    lowered = _run_flash_attention_lower_blackhole_ops_after_optimize(
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
    )["main"]
    script = lowered.script()
    lowering_requirements = lowered.attrs["blackhole.lowering_requirements"]

    assert "tl.blackhole.fill_fragment" in script
    assert "fill" not in set(lowering_requirements["pointwise_op_kinds"])


def test_flash_attention_forward_optimized_path_lowers_fragment_casts():
    lowered = _run_flash_attention_lower_blackhole_ops_after_optimize(
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
    )["main"]
    script = lowered.script()
    lowering_requirements = lowered.attrs["blackhole.lowering_requirements"]

    assert "tl.blackhole.cast_fragment_slice" in script
    assert "cast" not in set(lowering_requirements["pointwise_op_kinds"])


def test_flash_attention_forward_optimized_path_lowers_local_to_cb_staging():
    lowered = _run_flash_attention_lower_blackhole_ops_after_optimize(
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
    )["main"]
    script = lowered.script()

    assert "tl.blackhole.write_local_slice_to_cb" in script
    assert "O_shared_1[tx" not in script


def test_flash_attention_forward_runtime_shape_lowers_local_to_cb_without_thread_offset():
    lowered = _run_flash_attention_lower_blackhole_ops_after_optimize(
        mha_example,
        1,
        32,
        128,
        128,
        False,
        block_M=128,
        block_N=128,
        num_stages=1,
        threads=128,
    )["main"]
    script = lowered.script()

    assert "tl.blackhole.write_local_slice_to_cb" in script
    assert "tx *" not in script


def test_flash_attention_gqa_optimized_path_lowers_grouped_row_broadcasts():
    lowered = _run_flash_attention_lower_blackhole_ops_after_optimize(
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
    )["main"]
    script = lowered.script()
    lowering_requirements = lowered.attrs["blackhole.lowering_requirements"]

    assert "tl.blackhole.mul_grouped_row_bcast" in script
    assert "tl.blackhole.div_grouped_row_bcast" in script
    assert "tl.blackhole.exp2_grouped_row_bcast_affine" in script
    assert "row_broadcast" not in set(lowering_requirements["fragment_op_kinds"])


def test_flash_attention_gqa_reader_runtime_args_cover_all_accessor_buffers():
    lowered = _run_flash_attention_lower_blackhole_ops_after_optimize(
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
    )["main"]

    reader_segments = [seg for seg in lowered.attrs["blackhole.segment_plan"] if seg["kind"] == "reader"]
    assert len(reader_segments) == 1
    reader = reader_segments[0]

    accessor_buffers = [acc["buffer"] for acc in reader["accessors"]]
    runtime_arg_buffers = [
        arg["buffer"]
        for arg in reader["runtime_args"]
        if arg["kind"] == "input_buffer_addr32"
    ]

    assert len(accessor_buffers) == 3
    assert runtime_arg_buffers == accessor_buffers


def test_flash_attention_gqa_top_level_runtime_args_aggregate_segment_buffers():
    lowered = _run_flash_attention_lower_blackhole_ops_after_optimize(
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
    )["main"]

    assert "blackhole.runtime_args" in lowered.attrs

    top_level_runtime_arg_buffers = [
        arg["buffer"]
        for arg in lowered.attrs["blackhole.runtime_args"]
        if arg["kind"] == "input_buffer_addr32"
    ]
    reader_segments = [seg for seg in lowered.attrs["blackhole.segment_plan"] if seg["kind"] == "reader"]
    assert len(reader_segments) == 1
    reader_runtime_arg_buffers = [
        arg["buffer"]
        for arg in reader_segments[0]["runtime_args"]
        if arg["kind"] == "input_buffer_addr32"
    ]

    assert len(reader_runtime_arg_buffers) == 3
    assert top_level_runtime_arg_buffers[: len(reader_runtime_arg_buffers)] == reader_runtime_arg_buffers


def test_flash_attention_forward_lowers_mha_pipeline_end_to_end():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            mha_example.flashattn.jit_impl.get_tir(
                1,
                32,
                256,
                128,
                False,
                block_M=128,
                block_N=128,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )
    assert artifact is not None


def test_flash_attention_forward_pipeline_lifts_semantic_roles_without_workload_specific_schema():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            mha_example.flashattn.jit_impl.get_tir(
                1,
                32,
                256,
                128,
                False,
                block_M=128,
                block_N=128,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    program = artifact.device_mod["main_kernel"].attrs["tl.semantic_program"]
    state_roles = {str(state.role) for state in program.states}
    law_kinds = {str(update.law.kind) for update in program.updates}

    assert {"carry", "reduction_accumulator", "transient"}.issubset(state_roles)
    assert "recurrence" in law_kinds
    assert "selection_state" in state_roles


def test_flash_attention_forward_lowers_gqa_pipeline_for_supported_stage_count():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            gqa_example.flashattn.jit_impl.get_tir(
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
            ),
            target=target,
        )
    assert artifact is not None


def test_flash_attention_forward_compute_cb_ids_stay_in_compute_window():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            mha_example.flashattn.jit_impl.get_tir(
                1,
                32,
                128,
                128,
                False,
                block_M=128,
                block_N=128,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    spec = _extract_blackhole_executable_spec(artifact)
    compute_kernel = _require_blackhole_kernel(spec["kernels"], kind="compute", core_type="trisc")
    compute_source = str(compute_kernel["source_code"])

    compute_cb_ids = {
        int(cb["cb_id"])
        for cb in spec["cb_configs"]
        if int(cb["cb_id"]) >= 16 and f"({int(cb['cb_id'])}" in compute_source
    }

    assert compute_cb_ids
    assert compute_cb_ids == {cb_id for cb_id in compute_cb_ids if 16 <= cb_id <= 31}


def test_flash_attention_mha_reader_accessors_have_distinct_compile_time_slots():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            mha_example.flashattn.jit_impl.get_tir(
                1,
                32,
                128,
                128,
                False,
                block_M=128,
                block_N=128,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    spec = _extract_blackhole_executable_spec(artifact)
    reader_kernels = [kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "reader"]
    assert len(reader_kernels) == 1
    reader = reader_kernels[0]

    accessors = list(reader["accessors"])
    runtime_arg_buffers = [
        str(arg["buffer"])
        for arg in reader["runtime_args"]
        if str(arg["kind"]) == "input_buffer_addr32"
    ]
    assert len(runtime_arg_buffers) == 3

    input_accessors = [
        accessor for accessor in accessors if str(accessor["buffer"]) in runtime_arg_buffers
    ]
    assert len(input_accessors) == 3

    offsets = sorted(int(accessor["compile_time_arg_offset"]) for accessor in input_accessors)
    counts = {int(accessor["compile_time_arg_count"]) for accessor in input_accessors}
    assert counts == {2}
    assert offsets == [0, 2, 4]


def test_flash_attention_gqa_executable_spec_materializes_all_reader_inputs():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            gqa_example.flashattn.jit_impl.get_tir(
                1,
                16,
                128,
                128,
                False,
                groups=16,
                block_M=64,
                block_N=64,
                num_stages=2,
                threads=128,
            ),
            target=target,
        )

    spec = _extract_blackhole_executable_spec(artifact)
    reader_kernels = [kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "reader"]
    assert len(reader_kernels) == 1
    reader = reader_kernels[0]

    runtime_arg_buffers = {
        str(arg["buffer"])
        for arg in reader["runtime_args"]
        if str(arg["kind"]) == "input_buffer_addr32"
    }
    accessor_buffers = {str(accessor["buffer"]) for accessor in reader["accessors"]}
    materialized_buffers = {str(entry["buffer"]) for entry in spec["buffer_materializations"]}

    assert len(runtime_arg_buffers) == 3
    assert accessor_buffers == runtime_arg_buffers
    assert runtime_arg_buffers.issubset(materialized_buffers)


def test_flash_attention_segment_kernels_keep_buffer_runtime_args_role_local():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            mha_example.flashattn.jit_impl.get_tir(
                1,
                32,
                128,
                128,
                False,
                block_M=128,
                block_N=128,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    spec = _extract_blackhole_executable_spec(artifact)
    reader = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "reader")
    compute = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "compute")
    writer = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "writer")

    reader_buffer_kinds = {str(arg["kind"]) for arg in reader["runtime_args"] if "buffer" in arg}
    compute_buffer_kinds = {str(arg["kind"]) for arg in compute["runtime_args"] if "buffer" in arg}
    writer_buffer_kinds = {str(arg["kind"]) for arg in writer["runtime_args"] if "buffer" in arg}

    assert reader_buffer_kinds == {"input_buffer_addr32"}
    assert compute_buffer_kinds == set()
    assert writer_buffer_kinds == {"output_buffer_addr32"}


def test_flash_attention_segment_kernels_do_not_leak_compute_resources_into_writer():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            mha_example.flashattn.jit_impl.get_tir(
                1,
                32,
                128,
                128,
                False,
                block_M=128,
                block_N=128,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    spec = _extract_blackhole_executable_spec(artifact)
    compute = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "compute")
    writer = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "writer")

    compute_source = str(compute["source_code"])
    writer_source = str(writer["source_code"])

    # Fragment state must appear in compute kernel (as CB-backed pointers or array refs)
    assert "acc_o" in compute_source
    assert "scores_max" in compute_source
    assert "acc_s_cast" in compute_source

    # Fragment state must NOT leak into writer kernel
    assert "acc_o" not in writer_source
    assert "scores_max" not in writer_source
    assert "acc_s_cast" not in writer_source
    assert "/* blackhole managed resource */ half" not in writer_source


def test_flash_attention_compute_source_does_not_materialize_fragment_arrays():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            mha_example.flashattn.jit_impl.get_tir(
                1,
                32,
                128,
                128,
                False,
                block_M=128,
                block_N=128,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    spec = _extract_blackhole_executable_spec(artifact)
    compute = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "compute")
    compute_source = str(compute["source_code"])

    assert "/* blackhole managed resource */ float acc_o[" not in compute_source
    assert "/* blackhole managed resource */ float acc_s[" not in compute_source
    assert "/* blackhole managed resource */ float scores_max[" not in compute_source
    assert "/* blackhole managed resource */ float logsum[" not in compute_source
    assert "/* blackhole managed resource */ float scores_scale[" not in compute_source
    assert "/* blackhole managed resource */ float scores_sum[" not in compute_source
    assert "/* blackhole managed resource */ half acc_s_cast[" not in compute_source
    assert "float* acc_o = reinterpret_cast<float*>(get_local_cb_interface(" not in compute_source
    assert "float* acc_s = reinterpret_cast<float*>(get_local_cb_interface(" not in compute_source
    assert "float* scores_max = reinterpret_cast<float*>(get_local_cb_interface(" not in compute_source
    assert "float* logsum = reinterpret_cast<float*>(get_local_cb_interface(" not in compute_source
    assert "float* scores_scale = reinterpret_cast<float*>(get_local_cb_interface(" not in compute_source
    assert "float* scores_sum = reinterpret_cast<float*>(get_local_cb_interface(" not in compute_source
    assert "half* acc_s_cast = reinterpret_cast<half*>(get_local_cb_interface(" not in compute_source
    assert " = exp2f(" not in compute_source
    assert "std::exp2" not in compute_source


def test_flash_attention_compute_source_publishes_acc_s_cast_cb_before_second_matmul():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            mha_example.flashattn.jit_impl.get_tir(
                1,
                32,
                128,
                128,
                False,
                block_M=128,
                block_N=128,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    spec = _extract_blackhole_executable_spec(artifact)
    compute = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "compute")
    compute_source = str(compute["source_code"])

    cast_pos = compute_source.find("tilelang_cast_fragment_slice(dst, src, dst_offset, src_offset, num_elements);")
    publish_pos = compute_source.find("cb_push_back(23, 4);")
    second_mm_pos = compute_source.find("mm_init(23, 2, 17);")

    assert cast_pos != -1
    assert publish_pos != -1
    assert second_mm_pos != -1
    assert cast_pos < publish_pos < second_mm_pos


def test_flash_attention_compute_source_does_not_rereserve_blackhole_acc_gemm_outputs():
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    target = Target("blackhole")
    with target:
        artifact = lower(
            mha_example.flashattn.jit_impl.get_tir(
                1,
                32,
                128,
                128,
                False,
                block_M=128,
                block_N=128,
                num_stages=1,
                threads=128,
            ),
            target=target,
        )

    spec = _extract_blackhole_executable_spec(artifact)
    compute = next(kernel for kernel in spec["kernels"] if str(kernel["kind"]) == "compute")
    compute_source = str(compute["source_code"])

    assert "tile_regs_wait();\ncb_reserve_back(16, 1);\npack_tile(0, 16);" not in compute_source
    assert "tile_regs_wait();\ncb_reserve_back(17, 1);\npack_tile(0, 17);" not in compute_source


def test_flash_attention_forward_rejects_unsupported_pipeline_stage_count():
    with pytest.raises(
        tvm.TVMError,
        match="Blackhole fragment pipeline legality: unsupported stage count 4",
    ):
        target = Target("blackhole")
        with target:
            lower(
                gqa_example.flashattn.jit_impl.get_tir(
                    1,
                    16,
                    1024,
                    128,
                    False,
                    groups=16,
                    block_M=64,
                    block_N=64,
                    num_stages=4,
                    threads=128,
                ),
                target=target,
            )
