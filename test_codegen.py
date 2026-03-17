"""Test code generation for Blackhole target."""
import sys
# Ensure we use the correct tilelang
if '/root/dev/vibe_dsl/tilelang_repo' in sys.path:
    sys.path.remove('/root/dev/vibe_dsl/tilelang_repo')
sys.path.insert(0, '/root/dev/vibe_dsl/tilelang_repo')
import tilelang
from tilelang import language as T
from tilelang.engine.lower import lower
from tvm.target import Target

# Simple copy kernel
M, N = 32, 32
tile_m, tile_n = 32, 32

@T.prim_func
def main(A: T.Tensor((M, N), 'float16'), B: T.Tensor((M, N), 'float16')):
    with T.Kernel(T.ceildiv(N, tile_n), T.ceildiv(M, tile_m)) as (bx, by):
        for i, j in T.Parallel(tile_m, tile_n):
            y = by * tile_m + i
            x = bx * tile_n + j
            if y < M and x < N:
                B[y, x] = A[y, x]

# Lower to Blackhole target
target = Target('blackhole')
with target:
    artifact = lower(main, target=target)

# Check the generated source
if hasattr(artifact, 'kernel_source'):
    print('=== Generated Kernel Source ===')
    print(artifact.kernel_source[:3000])
    print('...')
    print(f"\nTotal length: {len(artifact.kernel_source)} chars")

    # Check that it has the correct include
    if 'api/dataflow/dataflow_api.h' in artifact.kernel_source:
        print("\n✓ Correct include path found: api/dataflow/dataflow_api.h")
    elif 'dataflow_api.h' in artifact.kernel_source:
        print("\n✗ Wrong include path found (missing api/dataflow/ prefix)")
    else:
        print("\n✗ No dataflow_api.h include found")

    # Check that TVM headers are NOT present
    if 'tvm/runtime/base.h' in artifact.kernel_source:
        print("✗ TVM header found (should not be in kernel code)")
    else:
        print("✓ No TVM headers found (correct)")
else:
    print('No kernel_source attribute')
