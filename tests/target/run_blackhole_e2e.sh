#!/bin/bash
# Launcher script for Blackhole E2E test
# Sets up correct Python paths to use Blackhole-enabled TileLang

export PYTHONPATH=/root/dev/vibe_dsl/tilelang_repo/python:/root/dev/vibe_dsl/tilelang_repo/build/lib
export TVM_LIBRARY_PATH=/root/dev/vibe_dsl/tilelang_repo/build/lib

cd /root/dev/vibe_dsl
python3 tests/target/test_blackhole_gemm_true_e2e.py "$@"
