from pathlib import Path

from tilelang.engine.lower import lower
from tvm.target import Target

from .common import staged_copy_kernel


def _repo_src_path(*parts):
    return Path(__file__).resolve().parents[4].joinpath("src", *parts)


def _function_source(source, start_marker, end_marker):
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    return source[start:end]


def test_codegen_does_not_recover_runtime_buffer_bindings_from_body():
    source = _repo_src_path("target", "codegen_blackhole.cc").read_text(encoding="utf-8")
    emit_runtime_arg_loads = _function_source(
        source,
        "void CodeGenBlackhole::EmitRuntimeArgLoads",
        "std::string CodeGenBlackhole::GetRuntimeArgVarForBuffer",
    )

    assert "tir::PostOrderVisit(f->body" not in emit_runtime_arg_loads
    assert "Recover exact runtime-backed buffer vars from the" not in emit_runtime_arg_loads
    assert "op_name == \"tl.blackhole.read_tile_to_cb\"" not in emit_runtime_arg_loads
    assert "buffer_vars_by_name[store->buffer->name]" not in emit_runtime_arg_loads


def test_host_launch_kernel_association_is_explicit_ir_attr():
    target = Target("blackhole")
    with target:
        artifact = lower(staged_copy_kernel(tile_rows=1, tile_cols=1), target=target)

    host_func = artifact.host_mod["main"]
    launched = host_func.attrs.get("tl.launched_kernel_symbols")

    assert [str(symbol) for symbol in launched] == ["main_kernel"]


def test_runtime_does_not_recover_host_launch_from_body_scan():
    source = _repo_src_path("target", "rt_mod_blackhole.cc").read_text(encoding="utf-8")

    assert "FindLaunchedKernelSymbol" not in source


def test_runtime_does_not_recover_static_buffer_info_from_device_body():
    source = _repo_src_path("target", "rt_mod_blackhole.cc").read_text(encoding="utf-8")

    assert "CollectStaticBufferInfo" not in source


def test_codegen_does_not_recover_reduction_region_from_final_body():
    source = _repo_src_path("target", "codegen_blackhole.cc").read_text(encoding="utf-8")

    assert "InferReductionSignature" not in source
    assert "InferReductionRepeatExtent" not in source


def test_projection_does_not_recover_remote_descriptors_from_runtime_args():
    source = _repo_src_path("target", "tt_program_projection.h").read_text(encoding="utf-8")

    assert "EncodeRemoteCoreDescriptorsFromRuntimeArgs" not in source


def test_segment_kind_marker_is_not_active_lowering_protocol():
    forbidden = "blackhole" + ".segment_kind"
    active_sources = sorted(_repo_src_path("transform").glob("lower_blackhole*"))

    offenders = [
        str(path.relative_to(_repo_src_path()))
        for path in active_sources
        if forbidden in path.read_text(encoding="utf-8")
    ]

    assert offenders == []
