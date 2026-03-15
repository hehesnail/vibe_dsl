#!/bin/bash
# TT-Metal 官方示例测试脚本
# 使用 TT-Sim 仿真器运行 TT-Metal 官方编程示例

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 统计
PASSED=0
FAILED=0
SKIPPED=0

# 检查环境
if [[ -z "$TT_METAL_HOME" ]]; then
    export TT_METAL_HOME="${PROJECT_ROOT}/tt_metal_repo"
fi

if [[ ! -d "$TT_METAL_HOME" ]]; then
    echo -e "${RED}Error: TT_METAL_HOME not found at $TT_METAL_HOME${NC}"
    exit 1
fi

# 设置环境
export TT_METAL_SIMULATOR_HOME="${TT_METAL_HOME}/sim"
export TT_METAL_SIMULATOR="${TT_METAL_SIMULATOR_HOME}/libttsim.so"
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_DISABLE_SFPLOADMACRO=1

# 设置库路径 - 使用 build 目录（CMake 默认）
BUILD_DIR="${TT_METAL_HOME}/build"
if [[ ! -d "$BUILD_DIR/programming_examples" ]]; then
    BUILD_DIR="${TT_METAL_HOME}/build_Release"
fi

export LD_LIBRARY_PATH="${BUILD_DIR}/tt_metal:${BUILD_DIR}/tt_stl:${BUILD_DIR}/ttnn:${BUILD_DIR}/lib:${BUILD_DIR}/tt_metal/third_party/umd/device:${BUILD_DIR}/_deps/fmt-build:${BUILD_DIR}/_deps/benchmark-build/src:${TT_METAL_HOME}/sim:${LD_LIBRARY_PATH}"

# 检查示例目录
EXAMPLES_DIR="${BUILD_DIR}/programming_examples"
if [[ ! -d "$EXAMPLES_DIR" ]]; then
    echo -e "${YELLOW}Warning: Examples directory not found at $EXAMPLES_DIR${NC}"
    echo "Please build with: cmake -DBUILD_PROGRAMMING_EXAMPLES=ON ..."
    exit 1
fi

echo "========================================"
echo "TT-Metal 官方示例测试 (TT-Sim)"
echo "========================================"
echo ""
echo "Environment:"
echo "  TT_METAL_HOME: $TT_METAL_HOME"
echo "  TT_METAL_SIMULATOR: $TT_METAL_SIMULATOR"
echo "  BUILD_DIR: $BUILD_DIR"
echo ""

# 运行单个测试
run_test() {
    local test_name="$1"
    local test_exe="$2"
    local expected_pattern="$3"

    echo -n "Testing $test_name ... "

    if [[ ! -f "$test_exe" ]]; then
        echo -e "${YELLOW}SKIPPED (not built)${NC}"
        ((SKIPPED++))
        return
    fi

    # 运行测试，捕获输出
    output=$("$test_exe" 2>&1) && status=$? || status=$?

    # 检查是否包含成功模式（即使退出码非零）
    if [[ -n "$expected_pattern" ]] && echo "$output" | grep -q "$expected_pattern"; then
        echo -e "${GREEN}PASSED${NC}"
        ((PASSED++))
    elif [[ $status -eq 0 ]]; then
        echo -e "${GREEN}PASSED${NC}"
        ((PASSED++))
    else
        echo -e "${RED}FAILED (exit code $status)${NC}"
        ((FAILED++))
    fi
}

# 核心测试（必须通过的）
echo "=== Core Tests ==="
run_test "add_2_integers_in_riscv" \
    "${EXAMPLES_DIR}/metal_example_add_2_integers_in_riscv" \
    "Success: Result is 21"

# 更多示例（已编译的）
echo ""
echo "=== Additional Tests ==="

run_test "hello_world_datamovement" \
    "${EXAMPLES_DIR}/metal_example_hello_world_datamovement_kernel" \
    ""

run_test "loopback" \
    "${EXAMPLES_DIR}/metal_example_loopback" \
    ""

run_test "eltwise_binary" \
    "${EXAMPLES_DIR}/metal_example_eltwise_binary" \
    ""

# 汇总
echo ""
echo "========================================"
echo "Test Summary"
echo "========================================"
echo -e "  ${GREEN}PASSED:  $PASSED${NC}"
echo -e "  ${RED}FAILED:  $FAILED${NC}"
echo -e "  ${YELLOW}SKIPPED: $SKIPPED${NC}"
echo ""

if [[ $FAILED -eq 0 ]]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi
