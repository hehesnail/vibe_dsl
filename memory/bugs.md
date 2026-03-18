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

### TT-Metal 示例在 TT-Sim 下缺少 runtime root

- **时间**: 2026-03-18
- **问题**: `metal_example_add_2_integers_in_riscv` 在 TT-Sim 下启动时报 `Root Directory is not set`。
- **根本原因**: 运行环境只设置了 simulator 和 `LD_LIBRARY_PATH`，但没有设置 `TT_METAL_RUNTIME_ROOT`，导致 runtime 无法定位 `tt_metal/` 根目录。
- **解决方案**:
  - 在 TT-Sim 环境脚本中显式导出 `TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME`
  - 或者从 `tt_metal_repo` 根目录启动示例
- **当前限制**: 这类环境问题不会在“仅编译通过”阶段暴露，必须通过一次真实 TT-Sim 执行才能发现。

## 与当前设计直接相关的记录

### GEMM 寻址与 tile access 语义不一致

- **时间**: 2026-03-16
- **问题**: 早期 GEMM TT-Sim 测试结果与参考结果不匹配。
- **根本原因**: 手工 `InterleavedAddrGen` 方案与 TT-Metal 主流 `TensorAccessorArgs`/tile accessor 语义不一致，导致 tile 索引和寻址模型偏离官方用法。
- **当前结论**: 这不是单纯“调一个地址公式”的小问题，而是后端中间抽象需要向 tile-access 语义收敛。
- **后续处理原则**: Blackhole copy/gemm dataflow 路径应优先对齐 `TensorAccessorArgs` 风格，而不是继续扩展裸地址模式。

### `BlackholeModule` 从 Python 直接调用时在 `ExecuteExternal` 路径崩溃

- **时间**: 2026-03-18
- **问题**: 通过 Python 直接调用 `artifact.codegen_mod["main"](...)` 时，执行会在 `BlackholeModuleNode::ExecuteExternal` 路径触发 segfault。
- **现象**:
  - `spec-driven` runner 手工驱动可以在 TT-Sim 上成功执行 single-core copy
  - 但从 Python 直接调 packed func 时，会在进入 `ExecuteExternal` 后崩溃，尚未稳定打印出完整 runner 调用日志
- **当前判断**: 问题更可能在 `BlackholeModule` 的 packed-arg / `DLTensor*` 调用面，而不是 copy kernel、runner 协议或 TT-Sim 环境本身。
- **根本原因**: `PackFuncVoidAddr` 对 handle 参数传入的 `void_args[i]` 是 `raw_args[i].v_ptr` 槽位地址，不是最终的 `DLTensor*`；BlackholeModule 直接把它当成 `DLTensor*` 解读，导致后续取 shape/data 时踩坏内存。
- **解决方案**:
  - 优先用 `ffi::AnyView::try_cast<DLTensor*>()` 解码 tensor 参数
  - 仅把 `void_args` 作为保守回退，并在使用时先解引用成真正的 `DLTensor*`
  - 补 direct-call 测试覆盖 `artifact.codegen_mod["main"](...)`
- **当前状态**: 已解决。`spec.json -> runner` 和 Python direct-call 两条 single-core copy 路径都已在 TT-Sim 上通过。
