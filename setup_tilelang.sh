#!/bin/bash
# TileLang 环境设置脚本
# 用于初始化 tilelang_repo 和编译

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TILELANG_DIR="${SCRIPT_DIR}/tilelang_repo"

echo "=== TileLang 环境设置 ==="

# 1. Clone TileLang（如果不存在）
if [ ! -d "${TILELANG_DIR}" ]; then
    echo "[1/4] 克隆 TileLang 仓库..."
    git clone --recursive https://github.com/tile-ai/tilelang.git "${TILELANG_DIR}"
else
    echo "[1/4] TileLang 已存在，跳过克隆"
fi

cd "${TILELANG_DIR}"

# 2. 添加上游远程（方便后续同步）
if ! git remote | grep -q upstream; then
    echo "[2/4] 配置上游远程..."
    git remote add upstream https://github.com/tile-ai/tilelang.git
fi

# 3. 编译 TileLang
echo "[3/4] 编译 TileLang（CUDA 启用）..."
mkdir -p build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="80" \
    -DTILELANG_ENABLE_CUDA=ON

make -j$(nproc)

cd "${TILELANG_DIR}"

# 4. 安装 Python 包
echo "[4/4] 安装 Python 包..."
pip uninstall tilelang -y 2>/dev/null || true

echo "${TILELANG_DIR}" > "$(python3 -c "import site; print(site.getsitepackages()[0])")/tilelang.pth"

# 5. 验证
echo ""
echo "=== 验证安装 ==="
python3 -c "import tilelang; print(f'TileLang version: {tilelang.__version__}')"

echo ""
echo "=== 设置完成 ==="
echo "TileLang 路径: ${TILELANG_DIR}"
echo ""
echo "后续开发:"
echo "1. 在 ${TILELANG_DIR}/src/target/ 添加 Blackhole 后端代码"
echo "2. 提交到 fork 的仓库: git push origin <branch>"
echo "3. 在 vibe_dsl 中提交相关文档和配置"
