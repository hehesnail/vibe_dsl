#!/bin/bash
# Run the TT-Metal add_2_integers_in_riscv programming example under TT-Sim

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TT_METAL_HOME="${TT_METAL_HOME:-$REPO_ROOT/tt_metal_repo}"
TT_METAL_BUILD_DIR="${TT_METAL_BUILD_DIR:-$TT_METAL_HOME/build_Release}"
ADD_TEST_BIN="${TT_METAL_BUILD_DIR}/programming_examples/metal_example_add_2_integers_in_riscv"

if [[ ! -x "$ADD_TEST_BIN" ]]; then
    echo "Error: add_2_integers_in_riscv test not found at $ADD_TEST_BIN"
    echo "Build TT-Metal first so the programming example is available."
    exit 1
fi

source "$SCRIPT_DIR/setup_tt_sim.sh"

echo "Running TT-Sim smoke test: $ADD_TEST_BIN"
"$ADD_TEST_BIN"

