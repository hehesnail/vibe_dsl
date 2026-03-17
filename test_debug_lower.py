"""
Debug LowerBlackholeOps recursion
"""
import sys
sys.path.insert(0, '/root/dev/vibe_dsl/tilelang_repo')

import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tvm.target import Target

# Simple kernel
@T.prim_func
def simple_kernel(A: T.Buffer((32,), 'float16'), B: T.Buffer((32,), 'float16')):
    for i in range(32):
        B[i] = A[i]

print("Step 1: Lower with cuda target to get base IR...")
target_cuda = Target('cuda')
try:
    with target_cuda:
        from tilelang.engine.lower import lower
        artifact = lower(simple_kernel, target=target_cuda)
    print("CUDA lower succeeded")
    if hasattr(artifact, 'device_mod'):
        print("\nCUDA device_mod TIR:")
        print(artifact.device_mod)
except Exception as e:
    print(f"CUDA lower failed: {e}")

print("\n" + "="*60)
print("Step 2: Call LowerBlackholeOps directly on the IR...")

try:
    # Get the PrimFunc from the module
    mod = artifact.device_mod
    for gvar, func in mod.functions.items():
        print(f"Function: {gvar}")
        print(f"Function type: {type(func)}")

        if hasattr(func, 'body'):
            print(f"Body type: {type(func.body)}")

            # Count the depth of the IR tree
            def count_depth(stmt, depth=0, max_depth=[0], types_seen=set()):
                if depth > max_depth[0]:
                    max_depth[0] = depth
                types_seen.add(type(stmt).__name__)

                # Visit children based on node type
                if hasattr(stmt, 'body'):
                    count_depth(stmt.body, depth+1, max_depth, types_seen)
                if hasattr(stmt, 'then_case'):
                    count_depth(stmt.then_case, depth+1, max_depth, types_seen)
                if hasattr(stmt, 'else_case') and stmt.else_case:
                    count_depth(stmt.else_case, depth+1, max_depth, types_seen)
                if hasattr(stmt, 'seq'):
                    for s in stmt.seq:
                        count_depth(s, depth+1, max_depth, types_seen)

                return max_depth[0], types_seen

            max_d, types = count_depth(func.body)
            print(f"Max IR depth: {max_d}")
            print(f"Node types: {types}")

        # Try calling LowerBlackholeOps
        print("\nCalling LowerBlackholeOps...")
        lower_ops = tvm.ffi.get_global_func("tl.transform.LowerBlackholeOps")
        result = lower_ops(func)
        print(f"LowerBlackholeOps result: {result}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
