from pathlib import Path


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
