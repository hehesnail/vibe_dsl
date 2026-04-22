import os

import pytest
import torch

import tilelang
from tilelang import language as T
from tilelang.engine.lower import _align_blackhole_device_symbol
from tilelang.engine.phase import LowerToBlackholePhaseB


def find_loop_annotation(stmt, attr_key):
    """Find the first For loop carrying the given annotation key."""
    if isinstance(stmt, tilelang.tvm.tir.For) and stmt.annotations.get(attr_key) is not None:
        return stmt.annotations[attr_key]
    if isinstance(stmt, tilelang.tvm.tir.BlockRealize):
        return find_loop_annotation(stmt.block.body, attr_key)
    if isinstance(stmt, tilelang.tvm.tir.Block):
        return find_loop_annotation(stmt.body, attr_key)
    if isinstance(stmt, tilelang.tvm.tir.SeqStmt):
        for child in stmt.seq:
            found = find_loop_annotation(child, attr_key)
            if found is not None:
                return found
        return None
    if isinstance(stmt, tilelang.tvm.tir.IfThenElse):
        found = find_loop_annotation(stmt.then_case, attr_key)
        if found is not None:
            return found
        if stmt.else_case is not None:
            return find_loop_annotation(stmt.else_case, attr_key)
        return None
    if hasattr(stmt, "body"):
        return find_loop_annotation(stmt.body, attr_key)
    return None


def contains_attr_stmt_key(stmt, attr_key):
    """Return whether a TIR stmt tree still carries the given AttrStmt key."""
    found = False

    def visit(node):
        nonlocal found
        if found:
            return
        if isinstance(node, tilelang.tvm.tir.AttrStmt) and str(node.attr_key) == attr_key:
            found = True

    tilelang.tvm.tir.stmt_functor.post_order_visit(stmt, visit)
    return found


def check_blackhole_codegen_requirements():
    """Check if Blackhole compilation requirements are met."""
    tilelang_home = os.environ.get("TILELANG_HOME")
    if not tilelang_home:
        return False, "TILELANG_HOME not set"
    return True, "OK"


def get_loaded_tilelang_cmake_cache():
    lib_path = getattr(tilelang, "_LIB_PATH", None)
    if not lib_path:
        return None
    build_dir = os.path.dirname(os.path.dirname(lib_path))
    return os.path.join(build_dir, "CMakeCache.txt")


def direct_build_enabled():
    cache_candidates = []
    loaded_cache = get_loaded_tilelang_cmake_cache()
    if loaded_cache:
        cache_candidates.append(loaded_cache)

    tilelang_home = os.environ.get("TILELANG_HOME")
    if tilelang_home:
        cache_candidates.append(os.path.join(tilelang_home, "build", "CMakeCache.txt"))

    for cache_path in cache_candidates:
        if not os.path.exists(cache_path):
            continue
        with open(cache_path, "r", encoding="utf-8") as f:
            cache_text = f.read()
            if "USE_BLACKHOLE_DIRECT" in cache_text and "=ON" in cache_text:
                return True
    return False


def check_blackhole_direct_execution_requirements():
    """Check if BlackholeModule direct execution requirements are met."""
    can_codegen, msg = check_blackhole_codegen_requirements()
    if not can_codegen:
        return False, msg

    tt_metal_runtime_root = os.environ.get("TT_METAL_RUNTIME_ROOT")
    if not tt_metal_runtime_root:
        return False, "TT_METAL_RUNTIME_ROOT not set"
    if not os.path.isdir(os.path.join(tt_metal_runtime_root, "tt_metal")):
        return False, f"TT_METAL_RUNTIME_ROOT does not contain tt_metal/: {tt_metal_runtime_root}"

    if not direct_build_enabled():
        return False, "TileLang build does not appear to have USE_BLACKHOLE_DIRECT=ON"

    return True, "OK"


def prepare_blackhole_phase_b_module(mod):
    """Run the Blackhole Phase B mainline up to SpatialPlan plus lowering facts."""
    return tilelang.engine.phase.LowerToBlackholePhaseB(mod)


def lower_blackhole_to_tt_target(mod):
    """Lower a Blackhole module through the validated TTProgram target contract."""
    source_mod = LowerToBlackholePhaseB(mod)
    mod = _align_blackhole_device_symbol(source_mod, mod)
    return tilelang.engine.phase.LowerToBlackholeTTProgram(mod)


def require_tt_program(func):
    """Return the materialized TTProgram attached to a PrimFunc."""
    if not (func.attrs and "tl.tt_program" in func.attrs):
        pytest.fail("Expected PrimFunc to carry tl.tt_program")
    return func.attrs["tl.tt_program"]

def extract_tt_program_segments(func):
    """Extract segment-like kernel/ABI records from TTProgram for regression tests."""
    tt_program = require_tt_program(func)
    abi_plans = list(tt_program.abi_plans)
    segments = []
    for kernel in tt_program.kernels:
        payload = dict(kernel.payload)
        payload.setdefault("name", str(kernel.name))
        payload.setdefault("kind", str(kernel.kind))
        payload.setdefault("core_type", str(kernel.core_type))
        abi_plan_index = int(kernel.abi_plan_index)
        if 0 <= abi_plan_index < len(abi_plans):
            abi = abi_plans[abi_plan_index]
            payload.setdefault("runtime_args", list(abi.runtime_args))
            payload.setdefault("common_runtime_args", list(abi.common_runtime_args))
            payload.setdefault("compile_time_arg_specs", list(abi.compile_time_arg_specs))
            payload.setdefault("accessors", list(abi.accessors))
            payload.setdefault("semaphore_bindings", list(abi.semaphore_bindings))
        segments.append(payload)
    return segments


def extract_blackhole_segment_plan(func):
    """Return segment-like target truth from TTProgram."""
    return extract_tt_program_segments(func)


def extract_blackhole_runtime_args(func, *, kind=None, core_type=None, common=False):
    """Return TTProgram ABI runtime args."""
    tt_program = require_tt_program(func)
    if kind is not None or core_type is not None:
        kernel = require_tt_kernel(tt_program, kind=kind, core_type=core_type)
        abi = tt_abi_for_kernel(tt_program, kernel)
        return list(abi.common_runtime_args if common else abi.runtime_args)

    aggregated = []
    seen = set()
    for abi in tt_program.abi_plans:
        args = abi.common_runtime_args if common else abi.runtime_args
        for arg in args:
            identity = str(arg["identity"]) if "identity" in arg else None
            key = identity or (str(arg["name"]) if "name" in arg else repr(dict(arg)))
            if key in seen:
                continue
            seen.add(key)
            aggregated.append(arg)
    return aggregated


def extract_blackhole_core_plan(func):
    """Return TTProgram core-group truth."""
    tt_program = require_tt_program(func)
    if not tt_program.core_groups:
        pytest.fail("Expected TTProgram to carry a TTCoreGroup")
    core_group = tt_program.core_groups[0]
    core_plan = dict(core_group.payload)
    core_plan.setdefault("logical_grid_x", int(core_group.logical_grid_x))
    core_plan.setdefault("logical_grid_y", int(core_group.logical_grid_y))
    core_plan.setdefault("linearization", str(core_group.linearization))
    core_plan.setdefault("physical_cores", list(core_group.physical_cores))
    core_plan.setdefault("work_packets", list(core_group.work_packets))
    return core_plan


def extract_blackhole_work_per_core(func):
    """Return the max work-count assigned to any core."""
    core_plan = extract_blackhole_core_plan(func)
    return max((int(packet["work_count"]) for packet in core_plan["work_packets"]), default=0)


def extract_blackhole_cb_configs(func):
    """Return TTProgram CB-plan payloads."""
    cb_plans = list(require_tt_program(func).cb_plans)
    configs = []
    for cb_plan in cb_plans:
        config = dict(cb_plan.payload)
        config.setdefault("cb_id", int(cb_plan.cb_id))
        config.setdefault("name", str(cb_plan.name))
        config.setdefault("role", str(cb_plan.resource_class))
        config.setdefault("num_pages", int(cb_plan.num_pages))
        config.setdefault("page_size", int(cb_plan.page_size_bytes))
        config.setdefault(
            "total_size_bytes", int(cb_plan.num_pages) * int(cb_plan.page_size_bytes)
        )
        config.setdefault("data_format", str(cb_plan.data_format))
        configs.append(config)
    return configs


def extract_blackhole_total_l1_bytes(func):
    """Return total L1 bytes consumed by CB plans."""
    return sum(int(config["total_size_bytes"]) for config in extract_blackhole_cb_configs(func))


def extract_tt_program_payload_map(func):
    """Return TTProgram payload as a Python dict for regression tests."""
    return dict(require_tt_program(func).payload)


def extract_blackhole_compute_contract(func):
    """Return compute contract from TTProgram payload."""
    payload = extract_tt_program_payload_map(func)
    if "compute_contract" not in payload:
        pytest.fail("Expected TTProgram payload to carry compute_contract")
    return dict(payload["compute_contract"])


def extract_blackhole_gemm_contract(func):
    """Return GEMM contract from TTProgram payload."""
    payload = extract_tt_program_payload_map(func)
    if "gemm_contract" not in payload:
        pytest.fail("Expected TTProgram payload to carry gemm_contract")
    return dict(payload["gemm_contract"])


def rebuild_tt_kernel(kernel, *, name=None, kind=None, core_type=None, abi_plan_index=None, payload=None):
    """Rebuild a TTKernel with optional field overrides."""
    make_tt_kernel = tilelang.tvm.get_global_func("tl.TTKernel")
    return make_tt_kernel(
        str(kernel.name) if name is None else name,
        str(kernel.kind) if kind is None else kind,
        str(kernel.core_type) if core_type is None else core_type,
        int(kernel.abi_plan_index) if abi_plan_index is None else abi_plan_index,
        dict(kernel.payload) if payload is None else payload,
    )


def rebuild_tt_core_group(
    core_group,
    *,
    name=None,
    logical_grid_x=None,
    logical_grid_y=None,
    linearization=None,
    physical_cores=None,
    work_packets=None,
    payload=None,
):
    """Rebuild a TTCoreGroup with optional field overrides."""
    make_tt_core_group = tilelang.tvm.get_global_func("tl.TTCoreGroup")
    return make_tt_core_group(
        str(core_group.name) if name is None else name,
        int(core_group.logical_grid_x) if logical_grid_x is None else logical_grid_x,
        int(core_group.logical_grid_y) if logical_grid_y is None else logical_grid_y,
        str(core_group.linearization) if linearization is None else linearization,
        list(core_group.physical_cores) if physical_cores is None else physical_cores,
        list(core_group.work_packets) if work_packets is None else work_packets,
        dict(core_group.payload) if payload is None else payload,
    )


def rebuild_tt_kernel_plan(
    kernel_plan,
    *,
    name=None,
    kind=None,
    core_type=None,
    block_plan_index=None,
    abi_plan_index=None,
    payload=None,
):
    """Rebuild a TTKernelPlan with optional field overrides."""
    make_tt_kernel_plan = tilelang.tvm.get_global_func("tl.TTKernelPlan")
    return make_tt_kernel_plan(
        str(kernel_plan.name) if name is None else name,
        str(kernel_plan.kind) if kind is None else kind,
        str(kernel_plan.core_type) if core_type is None else core_type,
        int(kernel_plan.block_plan_index) if block_plan_index is None else block_plan_index,
        int(kernel_plan.abi_plan_index) if abi_plan_index is None else abi_plan_index,
        dict(kernel_plan.payload) if payload is None else payload,
    )


def rebuild_tt_semaphore_plan(
    semaphore_plan,
    *,
    name=None,
    kind=None,
    semaphore_id=None,
    initial_value=None,
    core_type=None,
    source_task_index=None,
    target_task_index=None,
    core_ranges=None,
    payload=None,
):
    """Rebuild a TTSemaphorePlan with optional field overrides."""
    make_tt_semaphore_plan = tilelang.tvm.get_global_func("tl.TTSemaphorePlan")
    return make_tt_semaphore_plan(
        str(semaphore_plan.name) if name is None else name,
        str(semaphore_plan.kind) if kind is None else kind,
        int(semaphore_plan.semaphore_id) if semaphore_id is None else semaphore_id,
        int(semaphore_plan.initial_value) if initial_value is None else initial_value,
        str(semaphore_plan.core_type) if core_type is None else core_type,
        int(semaphore_plan.source_task_index) if source_task_index is None else source_task_index,
        int(semaphore_plan.target_task_index) if target_task_index is None else target_task_index,
        list(semaphore_plan.core_ranges) if core_ranges is None else core_ranges,
        dict(semaphore_plan.payload) if payload is None else payload,
    )


def rebuild_tt_abi_plan(
    abi_plan,
    *,
    name=None,
    kernel_name=None,
    runtime_args=None,
    common_runtime_args=None,
    compile_time_arg_specs=None,
    accessors=None,
    semaphore_bindings=None,
    payload=None,
):
    """Rebuild a TTABIPlan with optional field overrides."""
    make_tt_abi_plan = tilelang.tvm.get_global_func("tl.TTABIPlan")
    return make_tt_abi_plan(
        str(abi_plan.name) if name is None else name,
        str(abi_plan.kernel_name) if kernel_name is None else kernel_name,
        list(abi_plan.runtime_args) if runtime_args is None else runtime_args,
        list(abi_plan.common_runtime_args)
        if common_runtime_args is None
        else common_runtime_args,
        list(abi_plan.compile_time_arg_specs)
        if compile_time_arg_specs is None
        else compile_time_arg_specs,
        list(abi_plan.accessors) if accessors is None else accessors,
        list(abi_plan.semaphore_bindings) if semaphore_bindings is None else semaphore_bindings,
        dict(abi_plan.payload) if payload is None else payload,
    )


def rebuild_tt_program(
    program,
    *,
    entry_name=None,
    member_func=None,
    block_plans=None,
    kernel_plans=None,
    sync_plans=None,
    kernels=None,
    core_groups=None,
    cb_plans=None,
    transport_plans=None,
    semaphore_plans=None,
    compute_sync_plans=None,
    dst_layout_plans=None,
    abi_plans=None,
    execution_plans=None,
    payload=None,
):
    """Rebuild a TTProgram with optional field overrides."""
    make_tt_program = tilelang.tvm.get_global_func("tl.TTProgram")
    if kernels is not None and kernel_plans is None and len(program.kernel_plans) == len(kernels):
        kernel_plans = [
            rebuild_tt_kernel_plan(
                program.kernel_plans[index],
                name=str(kernel.name),
                kind=str(kernel.kind),
                core_type=str(kernel.core_type),
                abi_plan_index=int(kernel.abi_plan_index),
            )
            for index, kernel in enumerate(kernels)
        ]
    return make_tt_program(
        str(program.entry_name) if entry_name is None else entry_name,
        str(program.member_func) if member_func is None else member_func,
        list(program.block_plans) if block_plans is None else block_plans,
        list(program.kernel_plans) if kernel_plans is None else kernel_plans,
        list(program.transport_plans) if transport_plans is None else transport_plans,
        list(program.sync_plans) if sync_plans is None else sync_plans,
        list(program.abi_plans) if abi_plans is None else abi_plans,
        list(program.execution_plans) if execution_plans is None else execution_plans,
        list(program.kernels) if kernels is None else kernels,
        list(program.core_groups) if core_groups is None else core_groups,
        list(program.cb_plans) if cb_plans is None else cb_plans,
        list(program.semaphore_plans) if semaphore_plans is None else semaphore_plans,
        list(program.compute_sync_plans) if compute_sync_plans is None else compute_sync_plans,
        list(program.dst_layout_plans) if dst_layout_plans is None else dst_layout_plans,
        dict(program.payload) if payload is None else payload,
    )


def require_tt_kernel(tt_program, *, kind, core_type=None, name=None):
    """Require a unique TTProgram kernel matching the requested target role."""
    matches = []
    for kernel in tt_program.kernels:
        if str(kernel.kind) != kind:
            continue
        if core_type is not None and str(kernel.core_type) != core_type:
            continue
        if name is not None and str(kernel.name) != name:
            continue
        matches.append(kernel)

    if not matches:
        available = [
            f"{str(kernel.name)}:{str(kernel.kind)}:{str(kernel.core_type)}"
            for kernel in tt_program.kernels
        ]
        pytest.fail(
            f"Missing TTProgram kernel kind={kind!r} core_type={core_type!r} name={name!r}; "
            f"available kernels: {available}"
        )
    if len(matches) > 1:
        matched = [f"{str(kernel.name)}:{str(kernel.kind)}:{str(kernel.core_type)}" for kernel in matches]
        pytest.fail(
            f"Ambiguous TTProgram kernel kind={kind!r} core_type={core_type!r} name={name!r}; "
            f"matched kernels: {matched}"
        )
    return matches[0]


def tt_abi_for_kernel(tt_program, kernel):
    """Return the ABI plan referenced by a TTProgram kernel."""
    abi_plan_index = int(kernel.abi_plan_index)
    assert abi_plan_index >= 0
    return tt_program.abi_plans[abi_plan_index]


def staged_copy_kernel(
    tile_rows: int,
    tile_cols: int = 1,
    tile_m: int = 32,
    tile_n: int = 32,
    dtype: str = "bfloat16",
):
    """Define an explicit TileLang T.copy(global->shared->global) kernel."""
    m = tile_rows * tile_m
    n = tile_cols * tile_n

    if dtype == "bfloat16":
        @T.prim_func
        def main(
            A: T.Tensor((m, n), "bfloat16"),
            B: T.Tensor((m, n), "bfloat16"),
        ):
            with T.Kernel(1, 1) as (bx, by):
                A_shared = T.alloc_shared((tile_m, tile_n), "bfloat16")
                for tile_idx in T.serial(tile_rows * tile_cols):
                    tile_row = tile_idx // tile_cols
                    tile_col = tile_idx % tile_cols
                    T.copy(A[tile_row * tile_m, tile_col * tile_n], A_shared)
                    T.copy(A_shared, B[tile_row * tile_m, tile_col * tile_n])

        return main

    if dtype == "float16":
        @T.prim_func
        def main(
            A: T.Tensor((m, n), "float16"),
            B: T.Tensor((m, n), "float16"),
        ):
            with T.Kernel(1, 1) as (bx, by):
                A_shared = T.alloc_shared((tile_m, tile_n), "float16")
                for tile_idx in T.serial(tile_rows * tile_cols):
                    tile_row = tile_idx // tile_cols
                    tile_col = tile_idx % tile_cols
                    T.copy(A[tile_row * tile_m, tile_col * tile_n], A_shared)
                    T.copy(A_shared, B[tile_row * tile_m, tile_col * tile_n])

        return main

    raise ValueError(f"Unsupported staged_copy_kernel dtype: {dtype}")


def grid_indexed_staged_copy_kernel(
    grid_x: int,
    grid_y: int,
    tile_m: int = 32,
    tile_n: int = 32,
    dtype: str = "bfloat16",
):
    """Define a copy kernel whose indices depend on bx/by logical block coordinates."""
    m = grid_y * tile_m
    n = grid_x * tile_n

    if dtype == "bfloat16":
        @T.prim_func
        def main(
            A: T.Tensor((m, n), "bfloat16"),
            B: T.Tensor((m, n), "bfloat16"),
        ):
            with T.Kernel(grid_x, grid_y) as (bx, by):
                A_shared = T.alloc_shared((tile_m, tile_n), "bfloat16")
                T.copy(A[by * tile_m, bx * tile_n], A_shared)
                T.copy(A_shared, B[by * tile_m, bx * tile_n])

        return main

    if dtype == "float16":
        @T.prim_func
        def main(
            A: T.Tensor((m, n), "float16"),
            B: T.Tensor((m, n), "float16"),
        ):
            with T.Kernel(grid_x, grid_y) as (bx, by):
                A_shared = T.alloc_shared((tile_m, tile_n), "float16")
                T.copy(A[by * tile_m, bx * tile_n], A_shared)
                T.copy(A_shared, B[by * tile_m, bx * tile_n])

        return main

    raise ValueError(f"Unsupported grid_indexed_staged_copy_kernel dtype: {dtype}")


def staged_stick_copy_kernel(
    tile_m: int = 32,
    tile_n: int = 16,
    global_n: int = 32,
    dtype: str = "float32",
    src_col: int = 0,
    dst_col: int = 0,
):
    """Define a minimal interleaved stick-style copy with non-32-aligned width."""
    assert tile_m % 32 == 0
    assert 0 <= src_col <= global_n - tile_n
    assert 0 <= dst_col <= global_n - tile_n

    @T.prim_func
    def main(
        A: T.Tensor((tile_m, global_n), dtype),
        B: T.Tensor((tile_m, global_n), dtype),
    ):
        with T.Kernel(1, 1) as (bx, by):
            A_shared = T.alloc_shared((tile_m, tile_n), dtype)
            T.copy(A[0:tile_m, src_col : src_col + tile_n], A_shared)
            T.copy(A_shared, B[0:tile_m, dst_col : dst_col + tile_n])

    return main


def gemm_kernel(M: int = 32, N: int = 32, K: int = 128):
    """GEMM kernel with explicit data-movement: T.copy A/B in, gemm, T.copy C out."""
    block_M, block_N, block_K = M, N, K

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "bfloat16"),
        B: T.Tensor((N, K), "bfloat16"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1, 1) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "bfloat16")
            B_shared = T.alloc_shared((block_N, block_K), "bfloat16")
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            T.copy(A[0:block_M, 0:block_K], A_shared)
            T.copy(B[0:block_N, 0:block_K], B_shared)
            T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C[0:block_M, 0:block_N])

    return main


def gemm_kernel_with_transpose_flags(
    M: int = 32,
    N: int = 32,
    K: int = 128,
    transpose_A: bool = False,
    transpose_B: bool = True,
):
    """GEMM kernel with configurable transpose flags."""

    a_shape = (K, M) if transpose_A else (M, K)
    b_shape = (N, K) if transpose_B else (K, N)
    @T.prim_func
    def main(
        A: T.Tensor(a_shape, "bfloat16"),
        B: T.Tensor(b_shape, "bfloat16"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1, 1) as (bx, by):
            A_shared = T.alloc_shared(a_shape, "bfloat16")
            B_shared = T.alloc_shared(b_shape, "bfloat16")
            C_local = T.alloc_fragment((M, N), "float32")
            T.copy(A[0 : a_shape[0], 0 : a_shape[1]], A_shared)
            T.copy(B[0 : b_shape[0], 0 : b_shape[1]], B_shared)
            T.gemm(
                A_shared,
                B_shared,
                C_local,
                transpose_A=transpose_A,
                transpose_B=transpose_B,
            )
            T.copy(C_local, C[0:M, 0:N])

    return main


def gemm_kernel_with_compute_abi(
    M: int = 32,
    N: int = 32,
    K: int = 128,
    *,
    clear_accum: bool = True,
    k_pack: int = 2,
    wg_wait: int = 3,
    preclear_output_fragment: bool = False,
):
    """GEMM kernel with non-default compute ABI knobs."""

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "bfloat16"),
        B: T.Tensor((N, K), "bfloat16"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1, 1) as (bx, by):
            A_shared = T.alloc_shared((M, K), "bfloat16")
            B_shared = T.alloc_shared((N, K), "bfloat16")
            C_local = T.alloc_fragment((M, N), "float32")
            T.copy(A[0:M, 0:K], A_shared)
            T.copy(B[0:N, 0:K], B_shared)
            if preclear_output_fragment:
                T.clear(C_local)
            T.gemm(
                A_shared,
                B_shared,
                C_local,
                transpose_B=True,
                clear_accum=clear_accum,
                k_pack=k_pack,
                wg_wait=wg_wait,
            )
            T.copy(C_local, C[0:M, 0:N])

    return main


def gemm_kernel_with_post_merge_cast_consumer(
    M: int = 32,
    N: int = 32,
    K: int = 128,
    *,
    clear_accum: bool = False,
    k_pack: int = 1,
    wg_wait: int = 0,
    preclear_output_fragment: bool = True,
):
    """GEMM kernel whose accumulated fragment stays live for a later fragment-cast consumer."""

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "bfloat16"),
        B: T.Tensor((N, K), "bfloat16"),
        D: T.Tensor((M, N), "bfloat16"),
    ):
        with T.Kernel(1, 1) as (bx, by):
            A_shared = T.alloc_shared((M, K), "bfloat16")
            B_shared = T.alloc_shared((N, K), "bfloat16")
            C_local = T.alloc_fragment((M, N), "float32")
            D_local = T.alloc_fragment((M, N), "bfloat16")
            T.copy(A[0:M, 0:K], A_shared)
            T.copy(B[0:N, 0:K], B_shared)
            if preclear_output_fragment:
                T.clear(C_local)
            T.gemm(
                A_shared,
                B_shared,
                C_local,
                transpose_B=True,
                clear_accum=clear_accum,
                k_pack=k_pack,
                wg_wait=wg_wait,
            )
            for i, j in T.Parallel(M, N):
                D_local[i, j] = T.cast(C_local[i, j], "bfloat16")
            T.copy(D_local, D[0:M, 0:N])

    return main


def fragment_fill_cast_publish_kernel(
    M: int = 32,
    N: int = 32,
    *,
    fill_value: float = 3.5,
):
    """Kernel that isolates local-fragment -> tiled-CB cast publication."""

    @T.prim_func
    def main(D: T.Tensor((M, N), "bfloat16")):
        with T.Kernel(1, 1) as (bx, by):
            C_local = T.alloc_fragment((M, N), "float32")
            D_local = T.alloc_fragment((M, N), "bfloat16")
            for i, j in T.Parallel(M, N):
                C_local[i, j] = T.float32(fill_value)
            for i, j in T.Parallel(M, N):
                D_local[i, j] = T.cast(C_local[i, j], "bfloat16")
            T.copy(D_local, D[0:M, 0:N])

    return main


def gemm_kernel_with_policy(
    M: int = 32,
    N: int = 32,
    K: int = 128,
    *,
    policy=T.GemmWarpPolicy.FullRow,
):
    """GEMM kernel with non-default warp policy."""

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "bfloat16"),
        B: T.Tensor((N, K), "bfloat16"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1, 1) as (bx, by):
            A_shared = T.alloc_shared((M, K), "bfloat16")
            B_shared = T.alloc_shared((N, K), "bfloat16")
            C_local = T.alloc_fragment((M, N), "float32")
            T.copy(A[0:M, 0:K], A_shared)
            T.copy(B[0:N, 0:K], B_shared)
            T.gemm(A_shared, B_shared, C_local, transpose_B=True, policy=policy)
            T.copy(C_local, C[0:M, 0:N])

    return main


def gemm_kernel_with_mbar(
    M: int = 32,
    N: int = 32,
    K: int = 128,
):
    """GEMM kernel with an explicit barrier binding passed as mbar."""

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "bfloat16"),
        B: T.Tensor((N, K), "bfloat16"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1, 1) as (bx, by):
            A_shared = T.alloc_shared((M, K), "bfloat16")
            B_shared = T.alloc_shared((N, K), "bfloat16")
            C_local = T.alloc_fragment((M, N), "float32")
            mbar = T.alloc_barrier(128)
            T.copy(A[0:M, 0:K], A_shared)
            T.copy(B[0:N, 0:K], B_shared)
            T.gemm(A_shared, B_shared, C_local, transpose_B=True, mbar=mbar)
            T.copy(C_local, C[0:M, 0:N])

    return main


def gemm_kernel_with_compute_config_extras(
    M: int = 32,
    N: int = 32,
    K: int = 128,
):
    """GEMM kernel with richer TT-Metal compute config extras on the DSL surface."""

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "bfloat16"),
        B: T.Tensor((N, K), "bfloat16"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1, 1) as (bx, by):
            A_shared = T.alloc_shared((M, K), "bfloat16")
            B_shared = T.alloc_shared((N, K), "bfloat16")
            C_local = T.alloc_fragment((M, N), "float32")
            T.copy(A[0:M, 0:K], A_shared)
            T.copy(B[0:N, 0:K], B_shared)
            T.gemm(
                A_shared,
                B_shared,
                C_local,
                transpose_B=True,
                dst_full_sync_en=True,
                bfp8_pack_precise=True,
                defines={
                    "BLACKHOLE_TEST_DEFINE": "1",
                    "BLACKHOLE_ACC_MODE": "fp32",
                },
                named_compile_args={
                    "c_0": 0,
                    "c_1": 1,
                    "c_16": 16,
                },
            )
            T.copy(C_local, C[0:M, 0:N])

    return main


def assert_tensors_close_or_dump(actual, expected, atol, rtol, failure_message):
    if torch.allclose(actual, expected, atol=atol, rtol=rtol):
        return
    diff = (actual - expected).abs()
    pytest.fail(
        f"{failure_message}; max diff={diff.max().item()}, mean diff={diff.mean().item()}"
    )
