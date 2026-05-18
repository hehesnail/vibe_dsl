#!/bin/bash
# Probe the current TT-Sim Blackhole CCL runtime boundary.
#
# Exit 0 means the minimal 1x2 bf16 all-gather produced numerically correct
# output.  A nonzero exit means T10.1a is still blocked or the probe could not
# reach the runtime path.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/setup_tt_sim.sh" >/dev/null

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


emit("probe_phase", "fabric_all_gather")
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
try:
    emit("mesh_request", "1x2")
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 2))
    emit("mesh_ok", "true")
    emit("mesh_num_devices", mesh.get_num_devices())

    torch.manual_seed(0)
    participant_count = mesh.get_num_devices()
    torch_input = torch.rand([1, 1, 32, 32 * participant_count], dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
        device=mesh,
        dtype=ttnn.bfloat16,
    )

    emit("running_all_gather", "true")
    tt_output = ttnn.all_gather(tt_input, dim=3, topology=ttnn.Topology.Linear)
    ttnn.synchronize_device(mesh)

    torch_output = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
    )
    torch_reference = torch_input.repeat([participant_count, 1, 1, 1])
    max_abs_diff = (torch_output.to(torch.float32) - torch_reference.to(torch.float32)).abs().max().item()
    emit("fabric_ccl_ok", str(torch.equal(torch_output, torch_reference)).lower())
    emit("max_abs_diff", max_abs_diff)
    sys.exit(0 if torch.equal(torch_output, torch_reference) else 11)
except Exception as exc:
    emit("fabric_ccl_ok", "false")
    emit("exception_type", type(exc).__name__)
    emit("exception", repr(exc))
    sys.exit(10)
finally:
    if mesh is not None:
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
