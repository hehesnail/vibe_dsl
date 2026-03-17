"""
Debug LowerBlackholeOps recursion - direct approach
"""
import sys
sys.path.insert(0, '/root/dev/vibe_dsl/tilelang_repo')

import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tvm.target import Target

# Simple kernel that works
@T.prim_func
def simple_kernel(A: T.Buffer((32,), 'float16'), B: T.Buffer((32,), 'float16')):
    for i in range(32):
        B[i] = A[i]

print("Step 1: Create Blackhole target...")
target = Target('blackhole')
print(f"Target: {target}")

print("\nStep 2: Lower without device compile to get IR...")
from tilelang.engine.lower import lower

# Use enable_device_compile=False to get IR without full compilation
try:
    with target:
        artifact = lower(simple_kernel, target=target, enable_device_compile=False)
    print(f"Lower succeeded!")
    print(f"Artifact type: {type(artifact)}")

    if hasattr(artifact, 'device_mod'):
        mod = artifact.device_mod
        print(f"\nDevice mod:")
        print(mod)

        print(f"\nNumber of functions: {len(mod.functions)}")

        for gvar, func in mod.functions.items():
            print(f"\nFunction: {gvar}")
            print(f"Function type: {type(func)}")

            if hasattr(func, 'body'):
                print(f"Body type: {type(func.body)}")

                # Count the depth of the IR tree
                def count_depth(stmt, depth=0, max_depth=[0], types_seen=set(), max_per_type={}):
                    stmt_type = type(stmt).__name__
                    if depth > max_depth[0]:
                        max_depth[0] = depth
                    types_seen.add(stmt_type)

                    if stmt_type not in max_per_type:
                        max_per_type[stmt_type] = depth
                    else:
                        max_per_type[stmt_type] = max(max_per_type[stmt_type], depth)

                    # Visit children based on node type
                    if hasattr(stmt, 'body') and stmt.body is not None:
                        count_depth(stmt.body, depth+1, max_depth, types_seen, max_per_type)
                    if hasattr(stmt, 'then_case') and stmt.then_case is not None:
                        count_depth(stmt.then_case, depth+1, max_depth, types_seen, max_per_type)
                    if hasattr(stmt, 'else_case') and stmt.else_case is not None:
                        count_depth(stmt.else_case, depth+1, max_depth, types_seen, max_per_type)
                    if hasattr(stmt, 'seq') and stmt.seq:
                        for s in stmt.seq:
                            count_depth(s, depth+1, max_depth, types_seen, max_per_type)

                    return max_depth[0], types_seen, max_per_type

                max_d, types, per_type = count_depth(func.body)
                print(f"Max IR depth: {max_d}")
                print(f"Node types: {types}")
                print(f"Max depth per type: {per_type}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
