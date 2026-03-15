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
