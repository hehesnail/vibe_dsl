# 问题与 Bug 记录

> 说明: 本文档只保留仍然有复用价值的问题与环境坑，不再记录已经被新总体设计淘汰的方案细节。

## 仍然有效的问题记录

### `pip install -e .` 失败

- **时间**: 2026-03-15
- **问题**: `pip install -e . --no-build-isolation` 触发重新配置/构建，可能因 CMake 环境问题失败。
- **根本原因**: scikit-build-core 会重新运行 cmake，特定环境下线程或工具链检查可能失败。
- **解决方案**: 使用 `.pth` 文件指向本地源码/构建产物，避免无关重构建。

### `@T.prim_func` 在内联 `python -c` 中获取源码失败

- **时间**: 2026-03-15
- **问题**: `inspect.getsourcelines()` 无法从内联命令中恢复源码。
- **解决方案**: 将测试代码写入 `.py` 文件再执行。

### tilelang_repo 体积过大无法直接提交

- **时间**: 2026-03-15
- **问题**: `3rdparty/` 和 `build/` 导致仓库过大。
- **解决方案**: 排除大目录，只提交核心源码和文档。

### TT-Metal 构建依赖较多

- **时间**: 2026-03-15
- **问题**: TT-Metal 构建依赖 clang、NUMA、hwloc、capstone 等系统组件。
- **解决方案**: 明确工具链、依赖与运行时库路径。

### TT-Sim 需要完整 soc descriptor 和正确环境变量

- **时间**: 2026-03-15
- **问题**: soc descriptor 不完整或环境变量不对时，TT-Sim/UMD/Metal 示例会出现各种初始化失败。
- **解决方案**:
  - 使用完整的 Blackhole soc descriptor
  - 正确设置 simulator 路径
  - 启用 slow dispatch 模式

## 与当前设计直接相关的记录

### GEMM 寻址与 tile access 语义不一致

- **时间**: 2026-03-16
- **问题**: 早期 GEMM TT-Sim 测试结果与参考结果不匹配。
- **根本原因**: 手工 `InterleavedAddrGen` 方案与 TT-Metal 主流 `TensorAccessorArgs`/tile accessor 语义不一致，导致 tile 索引和寻址模型偏离官方用法。
- **当前结论**: 这不是单纯“调一个地址公式”的小问题，而是后端中间抽象需要向 tile-access 语义收敛。
- **后续处理原则**: Blackhole copy/gemm dataflow 路径应优先对齐 `TensorAccessorArgs` 风格，而不是继续扩展裸地址模式。
