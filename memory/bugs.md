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

### pure copy 仍可能被错误地 lower 成“循环内重复 tile0 copy”

- **时间**: 2026-03-18
- **问题**: 当前 Blackhole copy 虽然已经回到 `LowerBlackholeOps -> CodeGenBlackhole` 主链，但仍可能在逐标量 `BufferStore` 改写时，为循环体中的每个元素访问都发射一组 tile-level builtin。
- **现象**:
  - lowered TIR 中可以看到 `for i/j` 循环体内重复出现：
    - `tl.blackhole.read_tile_to_cb(..., tile_index=0, ...)`
    - `tl.blackhole.write_tile_from_cb(..., tile_index=0, ...)`
  - 生成的 TT-Metal kernel 里也会在循环中重复访问同一个 tile
  - 对 `32x32` one-tile case，这类错误可能被“结果看起来仍正确”掩盖
- **根本原因**: `LowerBlackholeOps` 当前还是在 `VisitStmt_(BufferStoreNode)` 粒度上识别并替换 copy，而不是先从整段 TIR/循环体中恢复出真正的 tile/dataflow copy 语义。
- **解决方向**:
  - 不再把“逐标量 store -> tile builtin”当成可接受终态
  - 应先做函数体级别的 copy 语义识别，再发射整 tile / dataflow 级 builtin
  - codegen 应消费这种整段 copy lowering 结果，而不是依赖 one-tile 特例
- **当前状态**: 未解决。已拆掉 codegen 对固定 runtime arg 槽位的假设，但 copy 的整函数体语义分析仍需继续推进。

### `CodeGenBlackhole` 当前仍把 `blockIdx.x/y` 绑定成常量 0

- **时间**: 2026-03-18
- **问题**: 当前 `CodeGenBlackhole::BindThreadIndex()` 会把 `blockIdx.x` / `blockIdx.y` 直接映射成常量 `0 /* core_x/core_y */`。
- **现象**:
  - 对 `with T.Kernel(grid_x, grid_y)` 风格的 multi-tile single-core copy，如果 tile 语义依赖 `bx/by`，生成代码里这些 tile 坐标会被常量化
  - 这会掩盖或破坏多 tile copy 的真实 tile index 语义
- **当前处理原则**:
  - Stage 2A copy 测试先使用单 kernel + 显式 serial tile loop 的 `T.copy(global -> shared -> global)` 样例
  - 让 tile loop 以普通 TIR `for` 变量的形式保留下来，先验证 `LowerTileOp 后 staged loop -> Blackhole builtin -> codegen` 这条主链
- **后续处理方向**:
  - 不应长期依赖 `blockIdx` 常量化
  - 后续需要把单核 tile 遍历语义和 host/runtime scheduling 的边界重新收口，避免 device code 在 tile index 上失真
- **当前状态**: 未解决。当前 staged copy MVP 通过显式 serial tile loop 绕开该限制。

### Blackhole 当前旁路了大量 TileLang/TVM 主线 pass

- **时间**: 2026-03-19
- **问题**: Blackhole 当前在 `OptimizeForTarget` 中过早 early return，导致很多本应复用的通用 TIR、host/device 与 Packed API pass 没有进入主路径。
- **现象**:
  - `AnnotateDeviceRegions` / `SplitHostDevice` / `MakePackedAPI` / `LowerDeviceKernelLaunch` 没有在 Blackhole 主线上运行
  - `rt_mod_blackhole` 会把缺少 `calling_conv` 的 PrimFunc 也当成 device kernel 处理
  - `BlackholeModule` / `rt_mod_blackhole` 间接承担了部分 PrimFunc 参数、host/device 边界和 runtime 参数语义
- **根本原因**: 早期为了快速打通 `ExecutableSpec -> runner` 执行路径，Blackhole 采用了自定义 kernel model，并在 `OptimizeForTarget` 中对 Blackhole 提前返回，绕开了 TileLang/TVM 主线中后段 pass。
- **解决方向**:
  - 先做 pass 复用矩阵
  - 优先恢复通用 TIR 规范化和 host/device / Packed API pass 的主线复用
  - 将 Blackhole 差异收缩到 `LowerTileOp` 附近和少量 device-specific pass
  - 停止长期依赖“无 `calling_conv` 也可当 device kernel”的路径
- **当前状态**: 部分已解决。`AnnotateDeviceRegions` / `SplitHostDevice` / `MakePackedAPI` / `LowerDeviceKernelLaunch` 已恢复到 Blackhole `lower()` 主路径，但中后段通用规范化 pass 仍未全部接回。

### split 后 device kernel 进入中后段规范化后会打断当前 copy 识别

- **时间**: 2026-03-19
- **问题**: Blackhole 在恢复 `SplitHostDevice` / `MakePackedAPI` / `LowerDeviceKernelLaunch` 主线后，如果继续把 `FlattenBuffer` / `VectorizeLoop` / `StorageRewrite` 等 pass 提前放到 `LowerBlackholeOps` 之前，当前 copy staged-lowering 会失效。
- **现象**:
  - split 后的 device kernel 不再保留 `LowerBlackholeOps` 当前依赖的 staged copy loop 形态
  - `blackhole.runtime_args` / `blackhole.segment_plan` / copy builtin 不再稳定产出
  - codegen 会退回普通指针式 load/store 路径
- **根本原因**: `LowerBlackholeOps` 当前仍主要依赖 split 后 device kernel 中较高层的 staged copy 结构；而 `FlattenBuffer` / `VectorizeLoop` / `StorageRewrite` 会把它重塑成当前 matcher 还无法识别的形态。
- **解决方案**:
  - 当前先恢复 host/device 主线，但暂不把上述会破坏 copy 结构的 pass 提前到 `LowerBlackholeOps` 之前
  - 后续需要扩展 `LowerBlackholeOps`，让它能消费 split 后、进一步规范化后的 device kernel 形态
- **当前限制**: 这意味着 Stage 2A 目前恢复的是一条“受控 pass 子链”，而不是完整的中后段通用 pass 集。

### split 后 device kernel 的 runtime arg buffer 绑定不能只依赖 `buffer_map`

- **时间**: 2026-03-19
- **问题**: 在 split 后的 Blackhole device kernel 上，`CodeGenBlackhole` 按 `blackhole.runtime_args` 生成 builtin 代码时，可能报 `Missing runtime arg binding for buffer var: A`。
- **根本原因**: split 后 device kernel 往往直接以 handle 参数 `A` / `B` 为 builtin buffer 绑定，而 `buffer_map` 为空；如果 codegen 只从 `buffer_map` 建 `buffer -> runtime arg` 映射，就无法把 `blackhole.runtime_args` 中的 `"buffer": "A"` 绑定回对应参数。
- **解决方案**:
  - `CodeGenBlackhole::EmitRuntimeArgLoads()` 需要同时从两处建映射：
    - `PrimFunc::buffer_map`
    - handle 参数名
- **当前状态**: 已解决。

### Blackhole copy kernel codegen 不能为 `shared.dyn` 打印伪 C 数组

- **时间**: 2026-03-19
- **问题**: 在 TT-Sim true E2E copy 中，runner 已经成功把 kernel 送入 TT-Metal JIT，但内核编译会因为生成了 `/* CB */ half A_shared[1024];` 这类声明而失败。
- **现象**:
  - TT-Metal JIT 报 `error: 'half' was not declared in this scope`
  - 出错位置对应 Blackhole copy kernel 中的 `shared.dyn` allocation
- **根本原因**: Blackhole 的 `shared/shared.dyn` 语义已经被收口成 CB/L1 资源，但 `CodeGenC::VisitStmt_(AllocateNode)` 仍按普通 C backend 的方式为它打印本地数组声明。
- **解决方案**:
  - 在 `CodeGenBlackhole::VisitStmt_(AllocateNode)` 中识别 `shared/shared.dyn/shared.barrier`
  - 只保留类型/作用域注册，跳过 C 数组声明
  - 继续让 runtime/builtin 路径负责真实 CB/L1 资源使用
- **当前状态**: 已解决。TT-Sim 下 copy 的 `spec.json -> runner` 和 direct-call 两条 true E2E 路径都已通过。
