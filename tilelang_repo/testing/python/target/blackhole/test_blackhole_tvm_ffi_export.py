from pathlib import Path

import pytest

import tilelang
import tvm

from .common import check_blackhole_codegen_requirements, staged_copy_kernel


def test_blackhole_tvm_ffi_export_generates_valid_host_shim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    can_run, msg = check_blackhole_codegen_requirements()
    if not can_run:
        pytest.skip(f"Blackhole requirements not met: {msg}")

    debug_dir = tmp_path / "export_debug"
    monkeypatch.setenv("TILELANG_DEBUG_EXPORT_DIR", str(debug_dir))
    monkeypatch.setenv("TILELANG_DISABLE_CACHE", "1")

    kernel = tilelang.compile(
        staged_copy_kernel(tile_rows=1, tile_cols=1),
        target="blackhole",
        execution_backend="tvm_ffi",
    )
    output = tmp_path / "blackhole_export.so"
    kernel.export_library(str(output))

    lib0_c = debug_dir / "lib0.c"
    assert output.exists()
    assert tvm.runtime.load_module(str(output)) is not None
    assert lib0_c.exists()
    assert "kernel_error_code = ;" not in lib0_c.read_text()
