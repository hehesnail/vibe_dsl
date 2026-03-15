#!/bin/bash
# TT-Metal 环境设置脚本
# 用于初始化 tt_metal_repo 和编译

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_DIR="${SCRIPT_DIR}/tt_metal_repo"
BUILD_DIR="${TT_METAL_DIR}/build_Release"

echo "=== TT-Metal 环境设置 ==="

# 1. Clone TT-Metal（如果不存在）
if [ ! -d "${TT_METAL_DIR}" ]; then
    echo "[1/6] 克隆 TT-Metal 仓库..."
    git clone --recursive https://github.com/tenstorrent/tt-metal.git "${TT_METAL_DIR}"
else
    echo "[1/6] TT-Metal 已存在，跳过克隆"
    echo "      路径: ${TT_METAL_DIR}"
fi

cd "${TT_METAL_DIR}"

# 2. 检查/创建 clang-20 软链接
echo "[2/6] 检查 clang-20..."
if ! command -v clang-20 &> /dev/null; then
    if command -v clang &> /dev/null; then
        echo "      创建 clang-20 软链接..."
        sudo ln -sf "$(which clang)" /usr/local/bin/clang-20
        sudo ln -sf "$(which clang++)" /usr/local/bin/clang++-20
    else
        echo "      错误: 未找到 clang"
        exit 1
    fi
else
    echo "      clang-20 已存在"
fi

# 3. 安装系统依赖
echo "[3/6] 检查系统依赖..."
MISSING_PKGS=""
for pkg in libnuma-dev libhwloc-dev libcapstone-dev; do
    if ! dpkg -l | grep -q "^ii  $pkg "; then
        MISSING_PKGS="$MISSING_PKGS $pkg"
    fi
done

if [ -n "$MISSING_PKGS" ]; then
    echo "      安装缺失依赖:$MISSING_PKGS"
    sudo apt-get update -qq
    sudo apt-get install -y -qq $MISSING_PKGS
else
    echo "      所有依赖已安装"
fi

# 4. 配置 CMake
echo "[4/6] 配置 CMake..."
if [ ! -d "${BUILD_DIR}" ]; then
    cmake -B build_Release \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
        -DENABLE_TRACY=OFF \
        -DWITH_PYTHON_BINDINGS=OFF \
        -DENABLE_DISTRIBUTED=OFF \
        -DCMAKE_TOOLCHAIN_FILE=cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake \
        -G Ninja
else
    echo "      构建目录已存在，跳过配置"
fi

# 5. 编译
echo "[5/6] 编译 TT-Metal..."
export LD_LIBRARY_PATH=${BUILD_DIR}/lib:${BUILD_DIR}/tt_metal:${BUILD_DIR}/tt_stl:$LD_LIBRARY_PATH

ninja -C build_Release -j$(nproc)

# 6. 验证
echo "[6/6] 验证安装..."
if [ -f "${BUILD_DIR}/tt_metal/libtt_metal.so" ]; then
    echo "      ✓ libtt_metal.so 生成成功"
    ls -lh ${BUILD_DIR}/tt_metal/libtt_metal.so
else
    echo "      ✗ libtt_metal.so 未找到"
    exit 1
fi

if [ -f "${BUILD_DIR}/tt_metal/third_party/umd/device/libdevice.so" ]; then
    echo "      ✓ libdevice.so 生成成功"
    ls -lh ${BUILD_DIR}/tt_metal/third_party/umd/device/libdevice.so
else
    echo "      ✗ libdevice.so 未找到"
    exit 1
fi

echo ""
echo "=== 设置完成 ==="
echo "TT-Metal 路径: ${TT_METAL_DIR}"
echo "构建目录: ${BUILD_DIR}"
echo ""
echo "环境变量设置:"
echo "  export TT_METAL_HOME=${TT_METAL_DIR}"
echo "  export LD_LIBRARY_PATH=${BUILD_DIR}/lib:${BUILD_DIR}/tt_metal:${BUILD_DIR}/tt_stl:\$LD_LIBRARY_PATH"
echo ""
echo "后续开发:"
echo "1. 在 tilelang_repo 中添加 TT-Metal 运行时支持"
echo "2. 链接 libtt_metal.so 开发 Blackhole 后端"
