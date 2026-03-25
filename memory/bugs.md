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

### direct 构建已完成但 Python 仍加载旧 `build/` 库

- **时间**: 2026-03-23
- **问题**: `build_blackhole` 已经启用 `USE_BLACKHOLE_DIRECT=ON` 并构建完成，但 `pytest`/Python direct-call 仍然落回 `ExecuteExternal`。
- **根本原因**: TileLang 开发态默认只从仓库根下的 `build/lib` 加载 `libtilelang.so`；如果同时存在默认 `build/` 和专用 `build_blackhole/`，Python 进程会静默加载旧库，导致测试前置检查与真实执行库不一致。
- **解决方案**:
  - direct 测试优先检查当前进程实际加载的 `libtilelang.so` 所属构建目录的 `CMakeCache.txt`
- **补充修正**:
  - 仓库后续约定统一以 `tilelang_repo/build/` 作为默认开发构建目录
  - `TILELANG_DEV_LIB_ROOT` 只保留给临时切换到其他构建目录的场景
  - 历史过渡目录 `build_blackhole/` 已删除
- **当前状态**: 已解决。默认加载固定为单一 `build/` 目录；这类问题的长期规避方式是不要保留第二套常驻开发构建目录。

## 与当前设计直接相关的记录

### copy codegen 的 runtime-arg buffer 绑定不能依赖“新路径 + 指针 identity”组合

- **时间**: 2026-03-24
- **问题**: 在 Stage 2D 为 GEMM 引入 multi-segment source generation 后，原本稳定的 copy `lower()` / codegen 回归测试重新失败，报：
  - `Missing runtime arg binding for buffer var: A`
- **根本原因**:
  - 为了支持 GEMM 的 reader / compute / writer，`rt_mod_blackhole` 临时把更多函数送入 segment-specific codegen 路径
  - 但 copy builtin 在 codegen 阶段看到的是 `A.data` / `B.data` 这类 buffer expr，不保证继续沿用先前 `VarNode*` identity
  - 如果 `CodeGenBlackhole` 只依赖某个新分支里的对象 identity，或者把 copy 也卷入新的 source-generation 路径，就会把本来稳定的 single-kernel copy 路径带回归
- **解决方式**:
  - 保持 pure copy 继续走原有 single-kernel build/codegen 主路径
  - 只让 multi-segment GEMM 走 segment-specific codegen
  - `CodeGenBlackhole` 的 buffer 绑定逻辑必须从 IR/schema 恢复，而不是依赖“新增分支里的对象 identity 恰好一致”
- **当前状态**: 已解决。Blackhole 测试目录当前结果为 `16 passed, 7 skipped, 1 xfailed`；GEMM Step 6 的 blocker 仍是 `MergeSharedMemoryAllocations` 的 flat-buffer 前置条件。

### GEMM `lower()` 当前会卡在 `MergeSharedMemoryAllocations` 的 flat-buffer 前置条件

- **时间**: 2026-03-24
- **问题**: Stage 2D 在补完 `rt_mod_blackhole` 多 segment extractor 和 `BlackholeModule` 3-kernel direct path 后，`test_blackhole_gemm_basic` 仍无法走完整 `lower()`，当前会在主线 pass 中报：
  - `MergeSharedMemoryAllocations expects flat memory buffers`
- **根本原因**: Blackhole 当前对 GEMM 采用 `LowerTileOp` skip，shared alloc 仍保持二维 tile buffer；而 `lower()` 后段的 `MergeSharedMemoryAllocations` 仍假设自己运行在 `FlattenBuffer` 之后，只接受一维扁平 shared buffer。
- **解决方向**:
  - 要么在 Blackhole GEMM 进入该 pass 之前先补 `FlattenBuffer`/等价扁平化前置条件
  - 要么给 `MergeSharedMemoryAllocations` 增加 Blackhole/非扁平 shared buffer 豁免
  - 在此问题解决前，Step 4/5 可以做编译级和 segment-plan 级验证，但 Step 6 true E2E 仍会被前置 pass 挡住
- **当前状态**: 未解决。由 Stage 2E StorageRank 扩展方案统一解决（见下条）。

### Blackhole device-private resource 当前会被 generic host/device pass 误解释

- **时间**: 2026-03-25
- **问题**: GEMM lowering 沿 `AnnotateDeviceRegions -> SplitHostDevice -> MakePackedAPI -> LowerDeviceKernelLaunch` 暴露三个错误：
  - `MergeSharedMemoryAllocations expects flat memory buffers`
  - `In PrimFunc main variables [C_local] are used, but are not passed in as API arguments`
  - `Only one dynamic shared memory allocation is allowed`
- **根本原因**: TIR `StorageScope` 把存储层级与资源语义混为一谈。GPU 两者 1:1 映射，但 Blackhole 的 CB（L1 FIFO 队列）和 Dst 累加器（寄存器文件）打破了这个假设。`shared.dyn` / `local.fragment` 让 generic pass 按 GPU 模型误解释 Blackhole 资源。
- **解决方案（Stage 2E）**:
  - 扩展 `StorageRank` 枚举：`kBlackholeCB = 13`、`kBlackholeAccumulator = 14`
  - 新增 `BlackholeDeviceResourceCanonicalization` pass 在 `SplitHostDevice` 前完成 scope 替换 + allocation 重定位
  - generic pass 自然正确（rank 不匹配 → 跳过），与 WMMA/MMA/Metal/AMX 扩展同构
  - 设计文档：`tasks/dev_design/stage2e_blackhole_device_resource_semantics.md`
- **当前状态**: 设计已定稿，实现进行中。

### Stage 2C copy semantics 不能继续用 `AttrStmt` 承载结构化 schema

- **时间**: 2026-03-23
- **问题**: Stage 2C 最初设计把 `blackhole.copy_semantics` 规划成 `AttrStmt(attr_key, value=<Map<String, ObjectRef>>)`；实现时发现这条路在当前 TIR API 下不成立。
- **根本原因**: `AttrStmt.value` 语义仍是 `PrimExpr`，不适合直接承载 `Map<String, Any>` 这类结构化对象；真正适合放结构化 pass 元数据的位置是 `ForNode::annotations` / `BlockNode::annotations`。
- **解决方案**:
  - 改为把 `blackhole.copy_semantics` 写到 `ForNode::annotations["blackhole.copy_semantics"]`
  - schema 用 `Map<String, Any>`，显式带 `src_shape/dst_shape/mid_shape`
  - `LowerBlackholeOps` 优先消费 loop annotation，再回退旧 matcher
- **当前状态**: 已解决。`FlattenBuffer + VectorizeLoop` 后的 Stage 2C 专项测试已通过；`StorageRewrite` 仍待后续验证。

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
- **当前状态**: 部分已解决。`CodeGenBlackhole` 现在已支持在存在 `current_work_linear_id + logical_grid_x` 时按 row-major 绑定 `blockIdx.x/y`；但 grid-indexed staged copy 仍可能在更早的 copy 语义恢复阶段把 `bx/by` 抹平，因此这还不是完整修复。

### grid-indexed staged copy 仍可能在 copy tile-index 恢复前丢失 `bx/by`

- **时间**: 2026-03-19
- **问题**: 即使 `AssignBlackholeCores` 已产出正式 `core_plan`，copy runtime arg schema 也已补 `current_work_linear_id`，某些 `grid > 1` staged copy 生成的最终 device code 仍会退化成固定 `tile_index = 0`。
- **现象**:
  - split 后 device kernel 已带有：
    - `blackhole.core_plan.logical_grid_x/y`
    - `blackhole.core_plan.work_packets`
    - `blackhole.runtime_args.current_work_linear_id`
  - 但 grid-indexed copy 的最终源码里仍可能只看到固定 tile index，而没有 `bx/by` 推导表达式
- **根本原因**: `LowerBlackholeOps` 当前在 staged copy tile-index 推断时，会先对 thread/loop vars 做零化归一；这会在 codegen 绑定 `blockIdx` 之前，就把 `bx/by` 参与的 tile-index 语义提前抹掉。
- **解决方向**:
  - 不要在 copy tile-index 恢复前无条件把 `blockIdx` 相关变量归零
  - 让 split 前 planning 或 split 后 copy matcher 至少保留 `bx/by -> tile_index` 的映射
  - 再由 codegen/runtime ABI 消费 `current_work_linear_id`
- **当前状态**: 已解决。`LowerBlackholeOps` 现在只在 tile-index 提取时清零 `threadIdx.*`，不再清零 `blockIdx.*`；grid-indexed staged copy 已能在 lowered TIR 和生成源码中保留 `bx/by -> tile_index` 公式。当前剩余问题转为 host/runtime 真执行验证是否完整覆盖全部 logical work items。

### single-core `grid > 1` 执行不能只把 `work_packets` 停留在 metadata

- **时间**: 2026-03-19
- **问题**: 即使 `AssignBlackholeCores` 已产出 `work_packets`，如果 runner / host materialization 仍只执行一次 program，并固定传入一个 `current_work_linear_id`，那 `grid > 1` copy 仍只会跑第一个 logical block。
- **根本原因**: 执行计划和 kernel ABI 已经形成，但 host/runtime 没有把 `work_packets -> current_work_linear_id` 真的 materialize 成多次 single-core logical work 执行。
- **解决方案**:
  - runner 侧顺序展开 `work_packets`
  - 对每个 logical work item 传入对应的 `current_work_linear_id`
  - 逐次 launch program，直到当前 single-core work packet 消费完
- **当前状态**: 已解决。runner 现在会按 `work_packets/current_work_linear_id` 顺序执行 single-core logical work items，`grid_x=2, grid_y=3, 96x64 float16` 的 grid-indexed staged copy 已在 TT-Sim direct-call 上通过，结果与 PyTorch 参考一致。

### oversubscription 不应延后到 runtime 才暴露

- **时间**: 2026-03-19
- **问题**: `PlanBlackholeCB` 之前即使检测到总 L1 占用超过 `1572864` bytes，也只记日志并继续生成 attrs，导致 oversubscription 会滑到更晚的 codegen/runtime 阶段才暴露。
- **根本原因**: planner 的 `Validate()` 结果没有被当成编译期约束，而是被当成“可恢复 warning”。
- **解决方案**:
  - 把 `PlanBlackholeCB` 的 L1/CB 约束收成真正的编译期失败
  - 用 oversize shared-tile copy 补负例，例如 `1024x1024` shared tile
- **当前状态**: 已解决。`PlanBlackholeCB` 现在会在 oversubscription 时直接让编译失败；对应负例已补到 Python 测试。

### large-shape copy 需要验证“总数据量大”但“per-core L1 合法”

- **时间**: 2026-03-19
- **问题**: 如果只验证最小 tile copy 或只验证 oversubscription 负例，很难确认 planner 没把“总 tensor bytes > 1.5MB”误当成“per-core L1 超限”。
- **解决方案**:
  - 增加一个总数据量大于 `1.5MB`、但 shared tile 仍是 `32x32` 的 staged copy 正例
  - 同时检查：
    - `blackhole.total_l1_bytes` 仍为 `4096`
    - TT-Sim direct-call 执行结果与 PyTorch 一致
- **当前状态**: 已解决。`800x1024 float16` staged copy 已在 TT-Sim direct-call 上通过，输入总字节数 `1638400`，结果与 PyTorch 参考一致。

### duplicated `CBRequirement` schema 会触发跨 pass 的 ODR/ABI 错位

- **时间**: 2026-03-19
- **问题**: 给 `PlanBlackholeCB` 的 `CBRequirement` 新增 `lifetime_begin/end` 后，`PlanBlackholeCB` 在运行时会随机以字符串拷贝、vector 排序/赋值崩溃。
- **现象**:
  - 崩点看起来在 `PlanBlackholeCB::AssignCBIds()` 或排序阶段
  - backtrace 会落到 `std::string` copy / `memcpy`
  - Python 侧表现为 segfault 或 abort，而不是干净的 ICHECK 失败
- **根本原因**: `lower_blackhole_ops.h` 和 `plan_blackhole_cb.h` 在同一 namespace 下重复定义了 `CBRequirement`；当只更新其中一份 schema 时，会触发 ODR/ABI 错位，导致 extractor 写出的对象布局与 planner 读取布局不一致。
- **解决方案**:
  - 至少保证两处重复定义完全同步
  - 更稳的长期方向是把共享 protocol struct 收敛成单一定义
  - 本轮同时让 `LowerBlackholeOps` 显式写出 `lifetime_begin/end` attrs，避免 planner 侧再隐式猜测
- **当前状态**: 已进一步解决。`CBType/CBRequirement` 已收敛到共享头文件，不再依赖两份重复定义保持人工同步；当前剩余维护点主要是后续是否还要把更多 planner protocol type 一并集中。

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

### `blackhole.target_mode` 会把过渡期 copy 标签误当成正式协议

- **时间**: 2026-03-19
- **问题**: `blackhole.target_mode = "single_core_copy"` 同时出现在 pass attrs、`ExecutableSpec`、`spec.json` 和 runtime fallback 里，容易让人误以为 Blackhole 仍靠模式字符串驱动执行。
- **现象**:
  - `LowerBlackholeOps` / `AssignBlackholeCores` 会产出或补默认 `target_mode`
  - `rt_mod_blackhole` 会把它当作 runtime arg 默认值和旧 device-kernel 识别依据的一部分
  - true E2E 已经主要靠 `runtime_args` / `segment_plan` / copy builtin 跑通，但协议表面仍残留 `single_core_copy`
- **根本原因**: Stage 1 为了快速打通 copy 闭环，引入了最小专用 emitter；后续虽然主语义已经迁回 pass/schema，但 `target_mode` 这个过渡字段没有及时从主协议移除。
- **解决方案**:
  - 从 `LowerBlackholeOps` / `AssignBlackholeCores` 中移除 `blackhole.target_mode`
  - 从 `ExecutableSpec`、`spec.json` 和 runner 中移除 `target_mode`
  - 让 copy fallback 改为按 device-side copy builtin / runtime schema 判断
  - `IsBlackholeDeviceKernel()` 不再把 `target_mode` 当识别依据
- **当前状态**: 已解决。copy true E2E 继续通过；`tilelang.engine.lower.is_device_call()` 也已去掉对 `blackhole.target_mode` 的兼容分支，Blackhole device kernel 识别现在只看正式 `calling_conv` 或正式 `blackhole.*` plan attrs。

### rectangular staged copy 会因为 scratch/L1 覆盖而执行错误

- **时间**: 2026-03-19
- **问题**: `32x64` / `64x32` 这类 staged copy 虽然已经能在 lowered TIR 中展开成多个硬件 `32x32` subtile，但 direct-call 真执行最初仍会输出错误结果。
- **现象**:
  - `LowerBlackholeOps` 已能生成正确的 `tile_index` 序列，例如 `tile_row * 2` / `tile_row * 2 + 1`
  - 但 TT-Sim direct-call 的输出与 PyTorch 参考不一致，`max_diff` 明显非零
- **根本原因**:
  - codegen 的 `read_tile_to_cb` / `write_tile_from_cb` 实际都只使用一块 scratch L1 基地址
  - 当 TIR 里出现 `read-read-write-write` 这类多 subtile 序列时，后一次 read 会覆盖前一次 scratch 内容
  - runner 也只按单 page size 分配 scratch L1，无法承载最小 CB FIFO 语义
- **解决方案**:
  - `LowerBlackholeOps` 把 rectangular staged copy 的 `tile_index` 和 page size 收正到硬件 `32x32` subtile 级
  - `CodeGenBlackhole` 按 `cb_configs` 为每个 `cb_id` 维护最小 head/tail page queue，并在 scratch L1 上按 page offset 读写
  - runner 分配 scratch L1 时按 `cb.num_pages * cb.page_size_bytes` 预留足够空间，而不是只分配单 page
- **当前状态**: 已解决。`32x64 float16` staged copy 的 Python direct-call 已在 TT-Sim 下通过，结果与 PyTorch 参考一致。

### TVM `RemapBufferData` 每次创建新 BufferNode，破坏下游 unordered_map 去重

- **时间**: 2026-03-25
- **问题**: `BlackholeDeviceResourceCanonicalization` 把 `shared.dyn` 改成 `blackhole.cb.input`。两个 For 循环（`dram_to_cb` / `cb_to_dram`）共享同一个 `A_shared` buffer。经过 canonicalization 后，`LowerBlackholeOps::buffer_to_cb_` 里 `AllocateCBId` 对第一个 loop 分配 id=32，但第二个 loop 再次分配 id=33，导致 codegen 找不到 id=33 的 page size（`Missing CB page size for cb_id=33`）。
- **根本原因**: `GetNewBuffer` 内部调用 `RemapBufferData(buf, new_data)`，每次调用都创建一个新的 `BufferNode`（新的指针）。两个 loop 里同一个原始 `A_shared` buffer 经过两次 `GetNewBuffer` 变成了两个不同的对象。`buffer_to_cb_` 以 `Buffer`（即 `ObjectRef`，pointer equality）为 key，所以第二个 Buffer 对象查不到第一个 loop 已经分配的 id。
- **解决方案**: 在 `GetNewBuffer` 内缓存结果：`std::unordered_map<const BufferNode*, Buffer> buf_remap_`；每次创建后存入，再次遇到同一原始 `BufferNode` 时直接返回已缓存的 Buffer 对象。

### TVM `CopyOnWrite()` 对临时 ObjectRef 产生 dangling pointer

- **时间**: 2026-03-25
- **问题**: 在 `VisitExpr_(BufferLoadNode*)` / `VisitStmt_(BufferStoreNode*)` 中使用 `auto node = Downcast<BufferLoad>(base).CopyOnWrite()` → 堆腐败（`corrupted double-linked list`），在 Python 进程退出时崩溃。
- **根本原因**: `Downcast<BufferLoad>(base)` 创建了一个临时 ObjectRef（右值），在语句结束的分号处 ref count 降为 0，析构器释放底层对象；`CopyOnWrite()` 返回的裸指针 `node` 随即悬空。所有后续 `node->xxx` 访问都是 UB（heap corruption）。
- **解决方案**: 不要对临时 ObjectRef 调用 `CopyOnWrite()`。改为直接构造返回值：
  ```cpp
  const auto* bl = base.as<BufferLoadNode>();
  return BufferLoad(new_buf, bl->indices);
  ```
  对 `BufferStore` 同理。

### direct path 复用固定 kernel 临时路径会触发 JIT 缓存串扰

- **时间**: 2026-03-23
- **问题**: `large-shape` / `rectangular` direct-call 用例单独运行能过，但在同一个 pytest 进程里和其他 direct-call case 连续运行时会输出错误结果；同条件下强制切回 runner 又全部通过。
- **根本原因**: `BlackholeModule::ExecuteDirect()` 会把 kernel 源码写到只按 `pid` 固定的临时目录和固定文件名里。TT-Metal JIT 对该路径产生了可观察的编译结果复用，导致后续 case 可能错误复用前一个 case 的已编译 kernel。
- **解决方案**:
  - direct path 的 kernel 临时目录改成“每次执行唯一”
  - 保持 runner / direct 都以新的源码路径触发独立 JIT 编译
- **当前状态**: 已解决。修复后 `test_blackhole_e2e.py` 已在 TT-Sim 下整体通过，结果为 `18 passed, 1 skipped`。
