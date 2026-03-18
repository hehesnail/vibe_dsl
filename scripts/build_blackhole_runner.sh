#!/bin/bash
# Bootstrap TT-Metal, verify it under TT-Sim, then build the TileLang Blackhole runner

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TILELANG_HOME="${TILELANG_HOME:-$REPO_ROOT/tilelang_repo}"
TT_METAL_HOME="${TT_METAL_HOME:-/root/dev/vibe_dsl/tt_metal_repo}"
TT_METAL_BUILD_DIR="${TT_METAL_BUILD_DIR:-$TT_METAL_HOME/build_Release}"
BUILD_THREADS="${BUILD_THREADS:-$(nproc)}"
RUNNER_DIR="$TILELANG_HOME/tools/blackhole_runner"
BUILD_DIR="${TILELANG_BLACKHOLE_RUNNER_BUILD_DIR:-$TILELANG_HOME/build-blackhole-runner}"
TT_METAL_EXAMPLE_TARGET="${TT_METAL_EXAMPLE_TARGET:-metal_example_add_2_integers_in_riscv}"
SKIP_TT_SIM_SMOKE_TEST="${SKIP_TT_SIM_SMOKE_TEST:-0}"

if [ -z "$TT_METAL_HOME" ]; then
    echo "Error: TT_METAL_HOME environment variable not set"
    echo "Please set TT_METAL_HOME to your TT-Metal installation directory"
    exit 1
fi

TT_METAL_TOOLCHAIN_ARGS=()
if [ -n "${TT_METAL_TOOLCHAIN_FILE:-}" ]; then
    TT_METAL_TOOLCHAIN_ARGS+=("-DCMAKE_TOOLCHAIN_FILE=$TT_METAL_TOOLCHAIN_FILE")
elif command -v clang++-20 >/dev/null 2>&1 && [ -f "$TT_METAL_HOME/cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake" ]; then
    TT_METAL_TOOLCHAIN_ARGS+=("-DCMAKE_TOOLCHAIN_FILE=$TT_METAL_HOME/cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake")
elif command -v g++-12 >/dev/null 2>&1 && [ -f "$TT_METAL_HOME/cmake/x86_64-linux-gcc-12-toolchain.cmake" ]; then
    TT_METAL_TOOLCHAIN_ARGS+=("-DCMAKE_TOOLCHAIN_FILE=$TT_METAL_HOME/cmake/x86_64-linux-gcc-12-toolchain.cmake")
fi

echo "Bootstrapping TT-Metal..."
echo "TT_METAL_HOME: $TT_METAL_HOME"
echo "TT_METAL_BUILD_DIR: $TT_METAL_BUILD_DIR"
echo "TT-Metal example target: $TT_METAL_EXAMPLE_TARGET"

echo "Configuring TT-Metal..."
cmake -S "$TT_METAL_HOME" -B "$TT_METAL_BUILD_DIR" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_PROGRAMMING_EXAMPLES=ON \
      -DTT_METAL_BUILD_TESTS=OFF \
      -DTTNN_BUILD_TESTS=OFF \
      -DTT_UMD_BUILD_TESTS=OFF \
      -DBUILD_TT_TRAIN=OFF \
      "${TT_METAL_TOOLCHAIN_ARGS[@]}"

echo "Building TT-Metal example..."
cmake --build "$TT_METAL_BUILD_DIR" --target "$TT_METAL_EXAMPLE_TARGET" -j"$BUILD_THREADS"

if [ "$SKIP_TT_SIM_SMOKE_TEST" != "1" ]; then
    echo "Running TT-Sim smoke test..."
    TT_METAL_HOME="$TT_METAL_HOME" TT_METAL_BUILD_DIR="$TT_METAL_BUILD_DIR" \
        "$SCRIPT_DIR/run_tt_sim_add_test.sh"
fi

echo "Building TileLang Blackhole Runner..."
echo "TILELANG_HOME: $TILELANG_HOME"
echo "Runner directory: $RUNNER_DIR"
echo "Runner build directory: $BUILD_DIR"

echo "Configuring..."
cmake -S "$RUNNER_DIR" -B "$BUILD_DIR" \
      -DCMAKE_BUILD_TYPE=Release \
      -DTT_METAL_HOME="$TT_METAL_HOME" \
      -DTT_METAL_BUILD_DIR="$TT_METAL_BUILD_DIR"

echo "Building..."
cmake --build "$BUILD_DIR" --target tilelang_blackhole_runner -j"$BUILD_THREADS"

echo ""
echo "Build complete!"
echo "Runner executable: $BUILD_DIR/tilelang_blackhole_runner"
echo ""
echo "To use the runner, either:"
echo "1. Add $BUILD_DIR to your PATH, or"
echo "2. Set TILELANG_BLACKHOLE_RUNNER environment variable:"
echo "   export TILELANG_BLACKHOLE_RUNNER=$BUILD_DIR/tilelang_blackhole_runner"
