#!/bin/bash
# TT-Sim 环境设置脚本
# 用于配置 Blackhole 硬件仿真器环境

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# TT-Metal 根目录
export TT_METAL_HOME="${PROJECT_ROOT}/tt_metal_repo"

# TT-Sim 配置
export TT_METAL_SIMULATOR_HOME="${TT_METAL_HOME}/sim"
export TT_METAL_SIMULATOR="${TT_METAL_SIMULATOR_HOME}/libttsim.so"
export TT_UMD_SIMULATOR="${TT_METAL_SIMULATOR}"  # UMD 测试使用
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_DISABLE_SFPLOADMACRO=1

# 库路径设置（运行 TT-Metal 程序必需）
BUILD_DIR="${TT_METAL_HOME}/build_Release"
export LD_LIBRARY_PATH="${BUILD_DIR}/tt_metal:\
${BUILD_DIR}/tt_stl:\
${BUILD_DIR}/ttnn:\
${BUILD_DIR}/lib:\
${BUILD_DIR}/tt_metal/third_party/umd/device:\
${BUILD_DIR}/_deps/fmt-build:\
${BUILD_DIR}/_deps/benchmark-build/src:\
${TT_METAL_HOME}/sim:\
${LD_LIBRARY_PATH}"

# 验证文件存在
if [[ ! -f "$TT_METAL_SIMULATOR" ]]; then
    echo "Error: TT-Sim library not found at $TT_METAL_SIMULATOR"
    echo "Please run download script first"
    return 1
fi

if [[ ! -f "$TT_METAL_SIMULATOR_HOME/soc_descriptor.yaml" ]]; then
    echo "Error: SoC descriptor not found at $TT_METAL_SIMULATOR_HOME/soc_descriptor.yaml"
    return 1
fi

echo "TT-Sim environment configured:"
echo "  TT_METAL_SIMULATOR_HOME=$TT_METAL_SIMULATOR_HOME"
echo "  TT_METAL_SIMULATOR=$TT_METAL_SIMULATOR"
echo "  TT_METAL_SLOW_DISPATCH_MODE=$TT_METAL_SLOW_DISPATCH_MODE"
echo "  TT_METAL_DISABLE_SFPLOADMACRO=$TT_METAL_DISABLE_SFPLOADMACRO"
echo ""
echo "To run the example:"
echo "  cd \$TT_METAL_HOME"
echo "  ./build/programming_examples/metal_example_add_2_integers_in_riscv"
