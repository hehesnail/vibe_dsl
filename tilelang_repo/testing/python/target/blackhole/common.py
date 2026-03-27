import os

import pytest
import torch

import tilelang
from tilelang import language as T


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


def staged_copy_kernel(tile_rows: int, tile_cols: int = 1, tile_m: int = 32, tile_n: int = 32):
    """Define an explicit TileLang T.copy(global->shared->global) kernel."""
    m = tile_rows * tile_m
    n = tile_cols * tile_n

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


def grid_indexed_staged_copy_kernel(grid_x: int, grid_y: int, tile_m: int = 32, tile_n: int = 32):
    """Define a copy kernel whose indices depend on bx/by logical block coordinates."""
    m = grid_y * tile_m
    n = grid_x * tile_n

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


def make_blackhole_cb_requirements_mod(cb_requirements):
    """Build a split/lowered Blackhole module with an explicit CB requirement list."""
    kernel = staged_copy_kernel(tile_rows=1, tile_cols=1)
    mod = tilelang.tvm.IRModule({"main": kernel})
    target = tilelang.tvm.target.Target("blackhole")
    with target:
        mod = tilelang.engine.phase.LowerAndLegalize(mod, target)
    mod = tilelang.transform.LowerBlackholeOps()(mod)
    func = mod["main"].with_attr("blackhole.cb_requirements", cb_requirements)
    return tilelang.tvm.IRModule({"main": func})


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


def assert_tensors_close_or_dump(actual, expected, atol, rtol, failure_message):
    if torch.allclose(actual, expected, atol=atol, rtol=rtol):
        return
    diff = (actual - expected).abs()
    pytest.fail(
        f"{failure_message}; max diff={diff.max().item()}, mean diff={diff.mean().item()}"
    )
