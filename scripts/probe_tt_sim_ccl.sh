#!/bin/bash
# Probe the current TT-Sim Blackhole CCL runtime boundary.
#
# Exit 0 means the minimal 1x2 bf16 all-gather, reduce-scatter, and
# all-to-all probes produced numerically correct output.  A nonzero exit means
# T10.1a/T10.1d are still blocked or the probe could not reach the runtime path.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/setup_tt_sim.sh" >/dev/null

if [[ -n "${TT_SIM_LIB_OVERRIDE:-}" ]]; then
    export TT_METAL_SIMULATOR="$TT_SIM_LIB_OVERRIDE"
    export TT_UMD_SIMULATOR="$TT_SIM_LIB_OVERRIDE"
    export TT_METAL_SIMULATOR_HOME="$(cd "$(dirname "$TT_SIM_LIB_OVERRIDE")" && pwd)"
fi

export TILELANG_HOME="${TILELANG_HOME:-$REPO_ROOT/tilelang_repo}"
export TT_METAL_HOME="${TT_METAL_HOME:-$REPO_ROOT/tt_metal_repo}"
export TT_METAL_BUILD_DIR="${TT_METAL_BUILD_DIR:-$TT_METAL_HOME/build_Release}"
export TT_METAL_MOCK_CLUSTER_DESC_PATH="${TT_METAL_MOCK_CLUSTER_DESC_PATH:-$TT_METAL_HOME/tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_P300_both_mmio.yaml}"
export PYTHONPATH="$TT_METAL_HOME/ttnn:$TT_METAL_BUILD_DIR/ttnn:$TT_METAL_BUILD_DIR:${PYTHONPATH:-}"

PROBE_TIMEOUT_SECONDS="${PROBE_TIMEOUT_SECONDS:-180}"
MESH_LOG_FILE="$(mktemp -t tt-sim-ccl-mesh-probe.XXXXXX.log)"
CCL_LOG_FILE="$(mktemp -t tt-sim-ccl-runtime-probe.XXXXXX.log)"
trap 'rm -f "$MESH_LOG_FILE" "$CCL_LOG_FILE"' EXIT

set +e
timeout "$PROBE_TIMEOUT_SECONDS" python - <<'PY' >"$MESH_LOG_FILE" 2>&1
import os
import sys

import ttnn


def emit(key, value):
    print(f"{key}={value}", flush=True)


emit("probe_phase", "mesh_without_fabric")
emit("tt_metal_simulator", os.environ.get("TT_METAL_SIMULATOR", ""))
emit("tt_metal_mock_cluster_desc_path", os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", ""))

num_devices = ttnn.get_num_devices()
emit("ttnn_num_devices", num_devices)

mesh = None
try:
    emit("mesh_request", "1x2")
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 2))
    emit("mesh_without_fabric_ok", "true")
    emit("mesh_num_devices", mesh.get_num_devices())
    sys.exit(0)
except Exception as exc:
    emit("mesh_without_fabric_ok", "false")
    emit("exception_type", type(exc).__name__)
    emit("exception", repr(exc))
    sys.exit(10)
finally:
    if mesh is not None:
        ttnn.close_mesh_device(mesh)
PY
mesh_status=$?
set -e

cat "$MESH_LOG_FILE"

if [[ "$mesh_status" -ne 0 ]] || ! grep -q '^mesh_without_fabric_ok=true$' "$MESH_LOG_FILE"; then
    echo "probe_status=mesh_unavailable"
    echo "child_status=$mesh_status"
    exit "$mesh_status"
fi

set +e
timeout "$PROBE_TIMEOUT_SECONDS" python - <<'PY' >"$CCL_LOG_FILE" 2>&1
import os
import sys

import torch
import ttnn


def emit(key, value):
    print(f"{key}={value}", flush=True)


emit("probe_phase", "fabric_ccl")
emit("required_collectives", "all_gather,reduce_scatter,all_to_all")
emit("tt_metal_simulator", os.environ.get("TT_METAL_SIMULATOR", ""))
emit("tt_metal_mock_cluster_desc_path", os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", ""))

num_devices = ttnn.get_num_devices()
emit("ttnn_num_devices", num_devices)

ttnn.set_fabric_config(
    ttnn.FabricConfig.FABRIC_1D,
    ttnn.FabricReliabilityMode.STRICT_INIT,
    None,
    ttnn.FabricTensixConfig.DISABLED,
    ttnn.FabricUDMMode.DISABLED,
    ttnn.FabricManagerMode.DEFAULT,
)

mesh = None
sub_device_stall_group = None
try:
    emit("mesh_request", "1x2")
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 2))
    emit("mesh_ok", "true")
    emit("mesh_num_devices", mesh.get_num_devices())

    participant_count = mesh.get_num_devices()

    torch.manual_seed(0)
    torch_input = torch.rand([1, 1, 32, 32 * participant_count], dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
        device=mesh,
        dtype=ttnn.bfloat16,
    )

    emit("running_collective", "all_gather")
    tt_output = ttnn.all_gather(tt_input, dim=3, topology=ttnn.Topology.Linear)
    ttnn.synchronize_device(mesh)

    torch_output = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
    )
    torch_reference = torch_input.repeat([participant_count, 1, 1, 1])
    max_abs_diff = (torch_output.to(torch.float32) - torch_reference.to(torch.float32)).abs().max().item()
    all_gather_ok = torch.equal(torch_output, torch_reference)
    emit("collective_ok", f"all_gather:{str(all_gather_ok).lower()}")
    emit("max_abs_diff_all_gather", max_abs_diff)
    if not all_gather_ok:
        sys.exit(11)

    torch_input = torch.rand([1, 1, 32, 32 * participant_count], dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        device=mesh,
        dtype=ttnn.bfloat16,
    )

    emit("running_collective", "reduce_scatter")
    tt_output = ttnn.reduce_scatter(tt_input, dim=3, topology=ttnn.Topology.Linear)
    ttnn.synchronize_device(mesh)

    torch_output = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=3),
    )
    torch_reference = (torch_input.to(torch.float32) * participant_count).to(torch.bfloat16)
    max_abs_diff = (torch_output.to(torch.float32) - torch_reference.to(torch.float32)).abs().max().item()
    reduce_scatter_ok = torch.equal(torch_output, torch_reference)
    emit("collective_ok", f"reduce_scatter:{str(reduce_scatter_ok).lower()}")
    emit("max_abs_diff_reduce_scatter", max_abs_diff)
    if not reduce_scatter_ok:
        sys.exit(12)

    compute_grid_size = mesh.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1),
            )
        }
    )
    worker_sub_device_id = ttnn.SubDeviceId(0)
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    sub_device_stall_group = [worker_sub_device_id]
    sub_device_manager = mesh.create_sub_device_manager([worker_sub_device], 0)
    mesh.load_sub_device_manager(sub_device_manager)
    mesh.set_sub_device_stall_group(sub_device_stall_group)

    logical_shape = [1, 1, 64, 64]
    in_dim = 2
    out_dim = 3
    torch_input = torch.rand(logical_shape, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.create_mesh_mapper(
            mesh,
            ttnn.MeshMapperConfig(
                [ttnn.PlacementReplicate(), ttnn.PlacementShard(in_dim)],
                ttnn.MeshShape(1, participant_count),
            ),
        ),
        device=mesh,
        dtype=ttnn.bfloat16,
    )
    output_shape = list(logical_shape)
    output_shape[out_dim] //= participant_count
    persistent_intermediate_buffer = ttnn.from_torch(
        torch.zeros(output_shape, dtype=torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        device=mesh,
        dtype=ttnn.bfloat16,
    )
    persistent_output_buffer = ttnn.from_torch(
        torch.zeros(output_shape, dtype=torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        device=mesh,
        dtype=ttnn.bfloat16,
    )
    semaphore = ttnn.create_global_semaphore(mesh, ccl_sub_device_crs, 0)

    emit("running_collective", "all_to_all")
    tt_output = ttnn.experimental.all_to_all_async(
        tt_input,
        persistent_intermediate_buffer=persistent_intermediate_buffer,
        persistent_output_buffer=persistent_output_buffer,
        in_dim=in_dim,
        out_dim=out_dim,
        multi_device_global_semaphore=semaphore,
        num_links=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Ring,
        subdevice_id=worker_sub_device_id,
    )
    ttnn.synchronize_device(mesh, sub_device_ids=sub_device_stall_group)

    all_to_all_ok = True
    max_abs_diff = 0.0
    torch_reference_chunks = torch.chunk(torch_input, participant_count, out_dim)
    for index, device_tensor in enumerate(ttnn.get_device_tensors(tt_output)):
        torch_output = device_tensor.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        torch_reference = torch_reference_chunks[index]
        max_abs_diff = max(
            max_abs_diff,
            (torch_output.to(torch.float32) - torch_reference.to(torch.float32)).abs().max().item(),
        )
        all_to_all_ok = all_to_all_ok and torch.equal(torch_output, torch_reference)
    emit("collective_ok", f"all_to_all:{str(all_to_all_ok).lower()}")
    emit("max_abs_diff_all_to_all", max_abs_diff)
    if not all_to_all_ok:
        sys.exit(13)

    emit("fabric_ccl_ok", "true")
    sys.exit(0)
except Exception as exc:
    emit("fabric_ccl_ok", "false")
    emit("exception_type", type(exc).__name__)
    emit("exception", repr(exc))
    sys.exit(10)
finally:
    if mesh is not None:
        if sub_device_stall_group is not None:
            mesh.reset_sub_device_stall_group()
            mesh.clear_loaded_sub_device_manager()
        ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
PY
child_status=$?
set -e

cat "$CCL_LOG_FILE"

if [[ "$child_status" -eq 0 ]] && grep -q '^fabric_ccl_ok=true$' "$CCL_LOG_FILE"; then
    echo "probe_status=ok"
    exit 0
fi

if grep -q 'eth_txq_cmd=0x2' "$CCL_LOG_FILE"; then
    echo "probe_status=fabric_ccl_unsupported"
    echo "unsupported_reason=eth_txq_cmd=0x2"
    exit 20
fi

if grep -q 'Trying to get un-initialized fabric context' "$CCL_LOG_FILE"; then
    echo "probe_status=fabric_context_missing"
    echo "unsupported_reason=uninitialized_fabric_context"
    exit 21
fi

if [[ "$child_status" -eq 124 ]]; then
    echo "probe_status=timeout"
    echo "timeout_seconds=$PROBE_TIMEOUT_SECONDS"
    exit 22
fi

echo "probe_status=failed"
echo "child_status=$child_status"
exit "$child_status"
