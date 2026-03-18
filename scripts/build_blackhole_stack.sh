#!/bin/bash
# Build the TileLang Blackhole stack: TileLang itself, then TT-Metal + TT-Sim smoke test + runner

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TILELANG_HOME="${TILELANG_HOME:-$REPO_ROOT/tilelang_repo}"
TILELANG_BUILD_DIR="${TILELANG_BUILD_DIR:-$TILELANG_HOME/build}"
BUILD_THREADS="${BUILD_THREADS:-$(nproc)}"

if [[ ! -d "$TILELANG_BUILD_DIR" || ! -f "$TILELANG_BUILD_DIR/Makefile" ]]; then
    echo "Error: TileLang build directory not ready: $TILELANG_BUILD_DIR"
    echo "Expected an existing CMake build tree with a Makefile."
    echo "Configure TileLang first, or set TILELANG_BUILD_DIR to a valid build directory."
    exit 1
fi

echo "Building TileLang..."
echo "TILELANG_HOME: $TILELANG_HOME"
echo "TILELANG_BUILD_DIR: $TILELANG_BUILD_DIR"
make -C "$TILELANG_BUILD_DIR" -j"$BUILD_THREADS"

echo ""
echo "Building TT-Metal + TT-Sim smoke test + TileLang runner..."
"$SCRIPT_DIR/build_blackhole_runner.sh"

echo ""
echo "Blackhole stack build complete."

