#!/bin/bash
# Build the TileLang Blackhole external runner

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TILELANG_HOME="${TILELANG_HOME:-$REPO_ROOT/tilelang_repo}"
TT_METAL_HOME="${TT_METAL_HOME:-/root/dev/vibe_dsl/tt_metal_repo}"

if [ -z "$TT_METAL_HOME" ]; then
    echo "Error: TT_METAL_HOME environment variable not set"
    echo "Please set TT_METAL_HOME to your TT-Metal installation directory"
    exit 1
fi

RUNNER_DIR="$TILELANG_HOME/tools/blackhole_runner"
BUILD_DIR="${TILELANG_BLACKHOLE_RUNNER_BUILD_DIR:-$TILELANG_HOME/build-blackhole-runner}"

echo "Building TileLang Blackhole Runner..."
echo "TILELANG_HOME: $TILELANG_HOME"
echo "TT_METAL_HOME: $TT_METAL_HOME"
echo "Runner directory: $RUNNER_DIR"

# Build
echo "Configuring..."
cmake -S "$RUNNER_DIR" -B "$BUILD_DIR" \
      -DCMAKE_BUILD_TYPE=Release \
      -DTT_METAL_HOME="$TT_METAL_HOME" \
      -DTT_METAL_BUILD_DIR="$TT_METAL_HOME/build_Release"

echo "Building..."
cmake --build "$BUILD_DIR" --target tilelang_blackhole_runner -j$(nproc)

echo ""
echo "Build complete!"
echo "Runner executable: $BUILD_DIR/tilelang_blackhole_runner"
echo ""
echo "To use the runner, either:"
echo "1. Add $BUILD_DIR to your PATH, or"
echo "2. Set TILELANG_BLACKHOLE_RUNNER environment variable:"
echo "   export TILELANG_BLACKHOLE_RUNNER=$BUILD_DIR/tilelang_blackhole_runner"
