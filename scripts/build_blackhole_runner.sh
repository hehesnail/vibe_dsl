#!/bin/bash
# Build the TileLang Blackhole external runner

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_HOME="${TT_METAL_HOME:-/root/dev/vibe_dsl/tt_metal_repo}"

if [ -z "$TT_METAL_HOME" ]; then
    echo "Error: TT_METAL_HOME environment variable not set"
    echo "Please set TT_METAL_HOME to your TT-Metal installation directory"
    exit 1
fi

RUNNER_DIR="$TT_METAL_HOME/tt_metal/programming_examples/tilelang_blackhole_runner"
BUILD_DIR="$RUNNER_DIR/build"

echo "Building TileLang Blackhole Runner..."
echo "TT_METAL_HOME: $TT_METAL_HOME"
echo "Runner directory: $RUNNER_DIR"

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure
echo "Configuring..."
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_STANDARD=17 \
      ..

# Build
echo "Building..."
make -j$(nproc)

echo ""
echo "Build complete!"
echo "Runner executable: $BUILD_DIR/tilelang_blackhole_runner"
echo ""
echo "To use the runner, either:"
echo "1. Add $BUILD_DIR to your PATH, or"
echo "2. Set TILELANG_BLACKHOLE_RUNNER environment variable:"
echo "   export TILELANG_BLACKHOLE_RUNNER=$BUILD_DIR/tilelang_blackhole_runner"
