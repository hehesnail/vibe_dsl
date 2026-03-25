# 通用开发模式与当前有效经验

> 当前 Blackhole 后端唯一设计依据: `tasks/dev_design/final_blackhole_backend_redesign.md`
> 说明: 本文档只保留稳定、可复用的工程经验，不再承载任何替代性的 Blackhole 总体设计。

## 一般编译器后端开发经验

### 1. 代码生成器开发模式

稳定经验：

- 代码生成器通常继承 `CodeGenC` 或同类基类，按需重写 `VisitStmt_` / `VisitExpr_` / `PrintType` / `PrintStorageScope`。
- 先让生成代码语法正确，再谈 target-specific 优化。
- 在复杂 target 上，不要让 codegen 同时承担“协议推断”和“字符串打印”两类职责。

当前适用原则：

- Blackhole codegen 应只负责把已经明确的 segment/spec 打印成源码。
- 不要再把 codegen 当成 runtime 规划器或多核调度器。
- **codegen 不应做 "register allocation 回写" 级别的工作**——如果 IR 中的标识符需要被替换成最终值，这应该是前面某个 pass 的职责，不是 codegen 的。
- 如果一个 pass 只把规划结果写到 attrs 而不回写 IR body，那后续 pass/codegen 必须从 attrs 恢复状态，这会形成"IR 和 attrs 两套真源"的维护负担。优先让产出规划结果的 pass 同时完成 IR 回写。

### 2. 类型系统与参数处理

稳定经验：

- 特殊数据类型和向量类型应分阶段支持，先做标量，再做向量。
- runtime 参数布局必须显式、可验证，不能依赖隐式猜测。
- 64-bit 地址传递时，需要明确拆分与重组规则。

当前适用原则：

- compile-time args 和 runtime args 必须严格区分。
- Blackhole 侧不要继续让 runtime 从 kernel 源码反推参数语义。

### 3. 存储与内存 scope

稳定经验：

- 不同后端的 `shared/local/global` 映射只是表象，关键是目标平台的真实资源模型。
- 如果目标平台的内存资源由 runtime 管理，codegen 不应伪造本地数组声明来冒充真实资源。

当前适用原则：

- Blackhole `shared` 的主映射应被视为 `CircularBuffer`/L1 资源规划问题，而不是简单的 C 层数组声明问题。

### 4. 头文件和生成代码调试

稳定经验：

- 头文件按需包含，不要依赖静态全局状态决定是否输出。
- 优先提供“不编译、只生成代码”的路径用于检查输出。
- 生成代码调试时，先验证：
  - include 路径
  - 类型映射
  - builtin 调用是否完整
  - 参数顺序是否一致

### 5. 测试策略

稳定经验：

- 单元测试：验证局部逻辑
- 集成测试：验证 pass/codegen/runtime 串接
- 端到端测试：必须包含真实执行与结果验证

当前适用原则：

- 只做 codegen 或只做 reference compare 的脚本，不应再称为 true E2E。

## TileLang 工程经验

### 1. 大仓库管理

仍然有效：

- `tilelang_repo/3rdparty/` 和 `tilelang_repo/build/` 不应进入主仓库提交。
- 需要通过子模块初始化和本地构建恢复完整开发环境。

### 2. Python 包开发模式

仍然有效：

- `pip install -e .` 在复杂 CMake 工程上可能重新触发构建并失败。
- 使用 `.pth` 指向本地源码/构建产物是一种实用的本地开发手段。

### 3. TileLang 运行时边界

当前应明确：

- `CompiledArtifact` 可以承载 host/device/runtime 相关对象。
- Blackhole 后端当前的关键问题不是 DSL 或普通 lowering，而是 target runtime 协议没有收敛。

## TT-Metal / TT-Sim 环境经验

### 1. TT-Metal 构建

以下经验仍然有效：

- 系统依赖、clang 版本、RPATH、`LD_LIBRARY_PATH` 都会直接影响编译和运行。
- `ENABLE_TRACY=OFF`、关闭不需要的 bindings 能减少构建复杂度。
- 如果 TileLang 侧要直接 include TT-Metal 新版 host API（而不是只链接旧 runner），构建约束不能只补库路径；至少还要对齐：
  - `C++20`
  - `TT_METAL_HOME` repo root
  - `tt_stl`
  - `hostdevcommon/api`
  - `umd/device/api`
  - `.cpmcache` 下的第三方 include
  否则常见现象是“先缺头文件，补完头文件后又在 `std::span` / `requires` / `<=>` 上报错”。
- 如果 TT-Metal 没有给出可直接消费的完整 install tree，不要在 TileLang 里一次性把整片 `.cpmcache` 加进 include path。更稳的过渡方式是：
  - 能走 package config 的包（如 `fmt` / `nlohmann_json` / `spdlog`）优先走 package config
  - 只对剩余缺少稳定入口的头（如 `tt-logger` / `enchantum` / `umd_asio`）做定向发现
  这样至少能把“依赖哪个包”固定下来，而不是把缓存目录本身当接口。
- 如果本地已经有 `TT_METAL_HOME/build_Release`，更稳的做法是先执行一次本地 staging install，再让 TileLang direct 模式优先消费 install tree：
  - 例如：`cmake --install $TT_METAL_HOME/build_Release --prefix $TT_METAL_HOME/build_Release/stage`
  - 然后让 CMake 走 `find_package(tt-metalium CONFIG REQUIRED)`
  这样能显著减少对 build-tree 导出细节和 `.cpmcache` 目录布局的依赖。
- TileLang 开发态默认应固定使用 `<repo>/build/lib/libtilelang.so` 作为唯一标准产物。direct-path 验证前应优先核对“当前进程实际加载的库”对应的 `CMakeCache.txt`，不要只扫描磁盘上的候选构建目录；否则很容易出现“看的是这份库，跑的是另一份库”的假阳性。
- 更干净的长期做法不是在多套开发构建目录之间发明自动优先级，而是直接统一成单一开发构建目录：
  - 以 `tilelang_repo/build/` 作为唯一默认构建目录
  - 临时试验目录只在明确需要时通过 `TILELANG_DEV_LIB_ROOT` 显式切换，并在试验结束后清理掉
  这样 Python、pytest、CMake cache 和调试日志都只对应一套产物，最不容易出现“看的是这份库，跑的是另一份库”。

### 2. TT-Sim 配置

以下经验仍然有效：

- `TT_METAL_SLOW_DISPATCH_MODE=1` 对 TT-Sim 很关键。
- simulator 库和完整 soc descriptor 的路径必须明确。
- `scripts/setup_tt_sim.sh` 必须在**实际执行测试/脚本的同一个 shell**里 `source`；只在先前某个终端里配过一次环境，不等于当前进程里的 `pytest` / `python` 也会自动继承。
  - 对当前仓库，稳定做法是：
    - 在命令前缀里显式写 `source ../scripts/setup_tt_sim.sh && ...`（从 `tilelang_repo/` 下执行）
    - 或 `source scripts/setup_tt_sim.sh && ...`（从顶层仓库执行）
  - 重点检查变量至少包括：
    - `TT_METAL_RUNTIME_ROOT`
    - `TT_METAL_SIMULATOR`
    - `TT_METAL_SLOW_DISPATCH_MODE`
    - `LD_LIBRARY_PATH`
- 如果既没有 `TT_METAL_SIMULATOR`，也没有 `TT_METAL_MOCK_CLUSTER_DESC_PATH`，runtime 会按真机模式探测设备；在没有可见芯片的环境里，这会直接在 Metal 初始化阶段报 `No chips detected in the cluster`。
- UMD 测试不完全等价于 TT-Metal 编程示例可运行性。
- 如果 direct path 在运行时把不同 kernel case 的源码反复写到同一个临时路径，TT-Metal JIT 可能按路径复用已编译结果，导致“单测单跑通过、同进程组合跑错结果”。临时 kernel 目录/文件名应该按每次执行唯一化，而不是只按 `pid` 固定。

### 3. TT-Metal 核心接口使用经验

当前有效经验：

- TT-Metal 的稳定 host-side 抽象是：
  - `Program`
  - `CreateCircularBuffer`
  - `CreateKernel` / `CreateKernelFromString`
  - `SetRuntimeArgs`
- compile-time args 与 runtime args 是一等概念，不是实现细节。
- multi-core 调度主要是 host/runtime 责任。
- Blackhole 测试应直接围绕进程内 direct host path 组织：
  - direct-call 用例只检查 direct path 编译/运行条件
  - compile-time / codegen-only 用例单独覆盖非执行层语义
  不要再让独立 runner 二进制成为主测试入口或前置条件。

## Blackhole 后端当前有效开发原则

以下内容是当前阶段应严格遵守的经验总结：

1. 不再把“单个 kernel 字符串”当成后端主产物。
2. 不再把 `SplitBlackholeKernel` 当成当前关键路径。
3. 不再让 codegen 主导多核物理映射。
4. 不再继续扩展旧 runner 的固定命令行协议。
5. 新功能优先落到 `ExecutableSpec -> BlackholeModule::ExecuteDirect()` 这条主路径上。
6. 任何局部设计都必须服从 `final_blackhole_backend_redesign.md`。

### 文档与状态治理经验

当前新增的稳定经验：

- 当阶段状态发生切换时，不要只改一处“总览”文档；至少要同步检查：
  - `tasks/progress.md` 的页首阶段状态
  - 分阶段任务表
  - “当前下一步/当前活动设计文档”区块
  - 对应阶段设计文档的状态字段
  否则最容易出现“页首已完成、表格仍进行中”的假状态。
- 对这类长期演进的后端仓库，设计文档应显式分成三类并在索引里写清：
  - 当前活动文档
  - 仍有效的支撑设计
  - 历史设计
  如果不做这层分类，后续人很容易把旧阶段计划误读成当前执行入口。
- 当历史设计文档仍需要保留回溯价值时，更稳的做法不是重写正文，而是在文档开头加统一头注，明确：
  - 这是历史文档还是支撑设计
  - 当前状态看哪里
  - 当前总体设计看哪里
  这样既能减少误导，又不会破坏历史决策上下文。

### Stage 2E：IR 语义扩展经验

当前新增的稳定经验：

- 当某个 target 的硬件资源模型与 generic backend 假设根本不一致时，优先扩 IR 类型系统/资源语义，而不是继续给后段 pass 打豁免或特判。
  - 本轮对 Blackhole，更稳的解法是扩 `StorageRank`
  - 再通过 `BlackholeDeviceResourceCanonicalization` 在 generic host/device pass 之前完成 scope canonicalization
  - 让 `SplitHostDevice` / `MakePackedAPI` / `LowerDeviceKernelLaunch` 自然跳过不该碰的 device-private resource
- 判断“该不该扩 IR”时，一个实用信号是：同一根因在多个 generic pass 上反复以不同报错出现。
  - 例如同一轮 GEMM 问题同时表现为：
    - `MergeSharedMemoryAllocations`
    - device func 参数提升错误
    - dynamic shared allocation 约束错误
  这通常说明问题不在单个 pass，而在更上层的资源语义承载不足。
- 对这种“IR 语义扩展”型改动，验证不能只看最终功能是否恢复；还应单独验证：
  - 新类型系统已进入正式管线
  - canonicalization 后 IR 的 scope/alloc 形态符合预期
  - 原先那组 generic pass 断点不再出现
  这样才能确认修的是语义边界，而不是偶然绕开了某个报错。

### Stage 2：主链接入与分层收正方法论

当前新增的稳定经验：

- Stage 2A 的首要目标不是“先让某个算子跑起来”，而是先把 target 接回 TileLang / TVM 的正式 host-device 主链：
  - `AnnotateDeviceRegions`
  - `SplitHostDevice`
  - `MakePackedAPI`
  - `LowerDeviceKernelLaunch`
  如果这条链还没接回，就不要继续扩大 runtime-only / unsplit-only 路径；否则后面每补一个功能，都会再背一份旁路债。
- Stage 2B/2C 最重要的方法论是三层分工固定：
  - split 前：保语义
  - split 后：提正式 plan
  - host side：只做 materialization
  一旦这三层边界不清，`LowerTileOp`、`LowerBlackholeOps`、`rt_mod_blackhole`、`BlackholeModule` 就会互相抢职责，最后又退回 runtime 猜协议。
- 对 split 前 / split 后的职责划分，一个实用检查是看“某个字段应该在哪层第一次变成显式协议”：
  - tile/dataflow/block 语义应该在 split 前保住
  - `segment_plan` / `runtime_args` / `cb_requirements` / `cb_configs` / `core_plan` 应该在 split 后显式产出
  - 真正的 TT-Metal object 创建应只在 host-side materialization 层发生
- Stage 2C 的通用模式是：如果某段语义后面会经过 `FlattenBuffer` / `VectorizeLoop` / 类似 destructive pass，就要在前面先挂结构化 annotation，不要等 pass 之后再从 loop/buffer 形态猜回来。
  - 对 Blackhole copy，这意味着在 `LowerTileOp` 之后、destructive pass 之前显式写 `blackhole.copy_semantics`
  - annotation 需要带足够的 shape / direction / buffer identity，不能只留最小名字信息
- Stage 2C/Phase 4 的 pass 回收不要“一次全开”；更稳的做法是逐个 pass 接回、逐个验语义保真。
  - 先验证 `FlattenBuffer`
  - 再验证 `VectorizeLoop`
  - 对 `StorageRewrite` 这类与 target 资源模型直接冲突的 pass，要允许明确记录为“不兼容”，而不是为了“全复用”硬接
- Stage 2D 的增量接入经验是：新 schema 先隔离到新场景，不要把已稳定路径一起卷入。
  - 纯 copy 保持原稳定 schema
  - GEMM 单独走 multi-segment schema
  - 等新路径稳定后再考虑统一模型
  这样能避免“为了接新功能，把旧功能一起回退”。
- Stage 2D 做 runtime / direct path 集成时，先收协议，再做 materialization。
  - 先让 planner/codegen/runtime 对 `buffer`、segment-level runtime args、CB binding 等 schema 对齐
  - 再让 `BlackholeModule` 去按 schema 创建 DRAM buffer、CB、kernel、runtime args
  否则 runtime 很容易再次退回位置规则和命名规则猜 ABI。
- 对 planner 型协议，`identity` 和 `lifetime` 必须分成两个独立维度建模。
  - `requirement_index` 负责“这是哪一个 requirement instance”
  - `lifetime_begin/end` 负责“它何时可复用”
  - 不能把两者偷合成同一个字段，否则一旦进入 lifetime-aware reuse，planner 就会把“不同 identity 但重叠存活”的资源错误合并。
- Stage 2D 的测试要分层，不要只盯着最终 direct-call。
  - 结构层：看 lowered TIR / attrs / builtin 是否已收正
  - planner 层：看 `cb_configs` / `core_plan` / bindings 是否正确
  - runtime 层：再看 direct path 真执行
  这样才能区分“语义没产出”和“环境没配好”。
- Stage 2B/2D 对 execution plan 的一个稳定经验是：single-core `grid > 1` 的最小正确模型，不是把 `blockIdx=0` 写死，而是保留 logical grid 语义，再由 host/runtime 用 `work_packets + current_work_linear_id` 顺序 materialize。
  这样可以先把 logical block 语义和 host-side execution plan 闭环，再把真正 multi-core 留到后续阶段。
- Stage 2 的总体推进顺序，已经形成一个可复用模式：
  - 先收主链
  - 再固定 split 前/后/host 三层边界
  - 再用最小可验收对象把协议闭环
  - 再逐步接回 destructive/generic pass
  - 最后处理更高阶的资源语义或 multi-segment 扩展
  如果顺序反过来，通常会陷入“局部能跑、整体协议越来越乱”的状态。

### Stage 0 协议落地经验

当前新增的稳定经验：

- 在从旧 `BlackholeFunctionInfo` 迁移到 `ExecutableSpec` 时，先保留 TVM 调用侧最小元信息（参数类型、buffer/scalar 标记），再逐步把 host materialization 逻辑迁到 `BlackholeModule`，能避免一次性打断 module 调用链。
- attr 统一不能只改 pass 或只改 runtime；至少要成对同步：
  - `PlanBlackholeCB` / `AssignBlackholeCores` 产出 `blackhole.*`
  - `rt_mod_blackhole` 读取同一套 `blackhole.*`
- `blackhole.core_plan` 这种结构化 attr 比散落的 `grid_x/grid_y/...` 标量 attr 更适合后续 spec extractor 和 direct host path 消费。
- 旧 runner 协议已删除；当前应由 `BlackholeModule` 直接从 `ExecutableSpec` 驱动 CB 创建、runtime args materialization 和 launch。
- 对这类“源码在 TileLang、依赖在 TT-Metal”的工具，优先采用：
  - 源码放在 `tilelang_repo/tools/...`
  - 由 TileLang 自己的 CMake/脚本产出二进制
  - 只链接 `TT_METAL_HOME/build_Release` 里的头文件和库
  - 不要求修改 `tt_metal_repo` 源码
- 如果需要从源码自举 `tt_metal` 再编 TileLang 工具，优先把流程收敛成：
  - 先由 TileLang 顶层脚本 configure/build `TT_METAL_HOME/build_Release`
  - 再跑一个 TT-Sim smoke test 验证编译产物可执行
  - 最后再编 TileLang 自己的 runner
  这样能尽早把“编得过”和“跑得起来”分层验证掉。
- 如果还需要同时重编 `tilelang_repo/build`，更稳的做法是再包一层显式总控脚本，而不是把 runner/TT-Metal 链路塞进 TileLang 默认 `all` 目标；这样普通 TileLang 编译不会被 Blackhole 外部依赖拖慢。
- 如果 standalone CMake 需要复刻 TT-Metal 的关键编译约束，至少要显式继承：
  - `cxx_std_20`
  - runner 所需的 compile definitions
  - `BUILD_RPATH/INSTALL_RPATH`
  否则很容易出现“能找到头文件但编译标准或运行时依赖不一致”的假通过。
- `.cpmcache` 下的目录名带哈希，但真正不稳定的点不是“有哈希”本身，而是把某个具体哈希写死在源码里；更稳的做法是脚本先 bootstrap TT-Metal，再由 CMake 按包名前缀动态解析对应目录。

### Stage 1 single-core copy 闭环经验

当前新增的稳定经验：

- 在 `global -> global` simple copy 还没有完全 lower 成通用 TT-Metal dataflow 语义时，可以先让 `rt_mod_blackhole` 保留一个最小专用 copy emitter；但它应只按 device-side builtin / runtime schema 回退，不能再把模式字符串当正式协议。
- 对最小 TT-Sim copy 路径，runner 侧至少要支持这组 runtime arg kind：
  - `input_buffer_addr32`
  - `output_buffer_addr32`
  - `tile_count`
  - `scratch_l1_buffer_addr32`
- `scratch_l1_buffer_addr32` 不能只在 spec 里声明；runner 必须按 schema 显式分配一块 L1 mesh buffer，否则 copy kernel 只是“协议看起来完整”，实际无法执行。
- Python 侧验证 Blackhole true E2E 时，即使 `BlackholeModule` 顶层调用面还没收口，也应至少切到 `spec.json + input.bin + output.bin` 新协议；不要继续保留旧 runner CLI 的测试假设。
- 对最小 single-core copy，`32x32 float16` 恰好是一 tile（2048 bytes），适合作为 TT-Sim 下的最小真执行 case；这类 case 能先验证协议和 runner，再把 module 调用面问题单独隔离出来。
- 当 `PackFuncVoidAddr` 包装 `kDLOpaqueHandle` 参数时，`void_args[i]` 指向的是 `raw_args[i].v_ptr` 这个槽位地址，而不是最终的 `DLTensor*`；对 Blackhole 这类外部 runner 路径，更稳的做法是优先通过 `ffi::AnyView::try_cast<DLTensor*>()` 取 tensor，只把 `void_args` 当保守回退。
- 对 Stage 2 这类“pass integration”工作，测试不应只看 `lower(...).codegen_mod` 是否能执行；更稳的做法是再加一层直接检查 pass 后 IR attrs，例如：
  - `blackhole.cb_configs`
  - `blackhole.runtime_args`
  - `blackhole.segment_plan`
  这样能直接防止语义又滑回 runtime 猜测路径。
- 对 Blackhole 这类正在收 protocol schema 的 backend，如果同一个 protocol struct 在多个 pass 头文件里重复定义，并且处于同一 namespace，那么每次 schema 扩展都必须同步更新所有副本，或者尽快集中到单一定义；否则很容易出现 ODR/ABI 错位，最后在完全不相关的位置以字符串拷贝或 vector 操作崩溃。
- 一旦确认某组 pass 之间共享同一份 protocol schema，不要长期依赖“复制一份完全相同的 struct 定义”来维持一致；更稳的做法是尽快抽出共享头文件，让 extractor/planner 直接 include 同一份定义。
- 当 `memory plan` 已经形成结构化 attrs 时，像 `total_size_bytes`、`lifetime_begin/end` 这类派生字段应由 planner 显式写出，而不是让 codegen/runtime 再从 `page_size * num_pages` 或 requirement 顺序二次反推；这样更容易把 protocol 和 planner 职责固定下来。
- 对 `PlanBlackholeCB` 这类 split 后 planner，如果要验证 lifetime/reuse，不必强行等真实 lowering 自然产出复杂 requirement；更稳的做法是：
  - 先复用一条真实 lowered Blackhole PrimFunc 作为 body 载体
  - 再在测试里显式覆写 `blackhole.cb_requirements`
  - 直接断言 `blackhole.cb_configs/total_l1_bytes/num_cbs`
  这样能把 planner 行为和前面 extractor 的偶然形态分层验证。
- 对这类本地 C++/Python 混合工程，如果刚改了 pass 并重链 `libtilelang.so`，不要把构建和 pytest 完全并行跑后直接相信第一次结果；更稳的做法是至少做一次串行 `build && pytest`，避免测试先加载旧 `.so` 产物形成假阴性。
- 如果目标是把路径重新接回 `TIR -> pass -> codegen` 主链，不要只检查 attrs；还要检查 lowered TIR body 里是否真的出现了目标 builtin call。本阶段对 copy 至少要看到：
  - `tl.blackhole.read_tile_to_cb`
  - `tl.blackhole.write_tile_from_cb`
- 当准备把某个 Blackhole 路径从 runtime emitter 切回 codegen 主路径时，可以先采用“codegen 优先、runtime emitter 回退”的切换方式：
  - 先让 `CodeGenBlackhole` 能消费 builtin
  - 再让 `rt_mod_blackhole` 优先使用 codegen 产物
- 对 TIR 里的 split-before 结构化元数据，如果 value 需要承载 `Map/Array/String` 等混合对象，不要继续套用 `AttrStmt.value`；更稳的做法是直接挂到 `ForNode::annotations` / `BlockNode::annotations` 这类 `Map<String, Any>` 容器上。`AttrStmt.value` 仍是 `PrimExpr` 语义，适合标量/表达式型 attr，不适合 Stage 2C 这种结构化 copy schema。
- 如果后续还要让 Blackhole copy 语义跨 `FlattenBuffer` / `VectorizeLoop` 保持可恢复，annotation 里不能只存 buffer 名和方向；还要显式带：
  - `src_shape`
  - `dst_shape`
  - `mid_shape`
  否则线性化后一旦 global/shared buffer 被压成一维，`LowerBlackholeOps` 就无法仅靠访问表达式稳定恢复 tile 宽高。
- `VectorizeLoop` 后常见的 copy 索引形态会从标量下标变成 `Ramp(base, stride, lanes)`；如果后端后续只关心 tile 基址，恢复逻辑应先把这类向量索引标量化到 `base`，再继续做 tile-index 计算，不要直接把整个 `Ramp` 当成“不支持的新模式”。
  - 最后只在 codegen 为空或失败时回退到手写 emitter
  这样更容易分阶段验证，不会一次性打断现有真执行链路。
- 对 Blackhole copy 这类“看起来最简单”的路径，也不能只盯着 `32x32 one-tile` 是否能跑通；这类 case 容易掩盖“循环体里重复发射同一个 `tile_index=0` copy”这类结构错误。更稳的做法是同时检查：
  - lowered TIR 是否仍在逐标量 store 上发射 tile builtin
  - 生成源码是否仍在循环里反复访问同一 tile
  - codegen 是否仍偷吃固定 runtime arg 槽位命名
- 对 staged copy，如果 DSL tile shape 本身会跨多个硬件 `32x32` tiles，不能简单把整个 shared tile 当成一个 runtime page。更稳的做法是：
  - `LowerBlackholeOps` 先把 DSL tile 形态展开成多个硬件 subtile 的 `tile_index`
  - `cb_configs.page_size` 对齐硬件 tile 大小
  - codegen/runtime 再按 `cb_id` 的 page queue 在 scratch/L1 上模拟最小 FIFO
  否则很容易出现“结构上看见了多个 subtile，执行时却因为 scratch 覆盖而结果错误”。
- 如果 Blackhole device code 依赖 `blackhole.runtime_args`，codegen 应直接消费 pass 产出的 schema 和 buffer 绑定，而不是固定假设某个 target mode 对应固定参数位。参数槽位顺序、名字、buffer 对应关系都应来自 IR attrs / schema，而不是写死在 builtin printer 里。
- 对 Blackhole 这类还在清理过渡协议的 backend，像 `target_mode` 这种“看起来像分类、实际上又可能被拿来驱动 fallback”的字段要尽早从主协议里移除；否则 pass/schema 已经收口，runtime 仍可能沿着旧标签继续分叉。
- 对 Blackhole copy，要先确认 TileLang pipeline 的真实断点：`LowerTileOp()` 先于 `LowerBlackholeOps()` 执行，所以 target pass 通常已经看不到原始 `tl.copy` 节点。更稳的做法是：
  - 先把 copy 主验收对象定义成 TileLang 原始 `T.copy(global -> shared -> global)` 语义
  - 再让 `LowerBlackholeOps` 去识别 `LowerTileOp` 之后留下来的 staged copy loop
  - 而不是继续围绕 `global -> global` 标量赋值循环做 target-specific 猜测
- 对这种 post-`LowerTileOp` staged copy lowering，更稳的回归测试是：
  - 直接在 DSL 样例里写显式 `T.copy(global -> shared)` 和 `T.copy(shared -> global)`
  - 然后检查 pass 后是否只剩一组 tile/dataflow builtin，而不是 vectorized 元素循环里重复发射 builtin
- 当一个 target 正在从“自定义 unsplit kernel model”切回 `SplitHostDevice` / `MakePackedAPI` / `LowerDeviceKernelLaunch` 主线时，不要一次性把所有中后段规范化 pass 都塞回去。更稳的做法是：
  - 先恢复 host/device 主链和 entry/kernel 语义
  - 再检查 target-specific device pass 还能否识别 split 后的 kernel 形态
  - 最后再逐步把 `FlattenBuffer` / `VectorizeLoop` / `StorageRewrite` 这类会重塑 copy/dataflow 形态的 pass 接回去
- 如果 runtime module 既要复用 split 后的 host entry 名，又要直接承载外部执行入口，spec 提取不要沿用 Packed API PrimFunc 的低层参数签名。更稳的做法是：
  - 由 host entry 解析出它 launch 的 device kernel
  - 继续用 device kernel 的用户参数签名构造 runtime module entry
  - 只把 host Packed API PrimFunc 当作 host/device 语义来源，而不是最终外部调用签名来源
- 对 Blackhole 这类把 `shared/shared.dyn` 映射成 CB/L1 资源的 backend，device codegen 不应再为这些 allocation 打印 C 数组声明；更稳的做法是：
  - 让 pass/runtime 负责 CB/L1 资源分配
  - codegen 只保留 builtin 调用和 runtime arg 绑定
  否则很容易在 TT-Metal JIT 编译阶段引入无意义、甚至不可编译的伪局部数组声明。
- 真执行测试需要把编译链和环境问题分层处理。对 Blackhole，`TT_METAL_RUNTIME_ROOT` 缺失时 runner 和 direct-call 都会在 Metal 初始化前失败；这类情况应该在测试前置检查里显式 skip，而不是记成 codegen/pass 回归。
- 当一个新 target 已经接入 TileLang/TVM 的 PrimFunc/TIR 主链时，优先问题不应是“补多少自定义 pass”，而应先核对它是否旁路了现有主线中的关键 pass。对 Blackhole，当前最关键的结构检查项是：
  - 是否过早在 target-specific optimize 阶段 early return
- Blackhole 这类 C++ backend 改动做完后，Python 结构测试如果仍然看到旧 attrs / 旧 runtime arg schema，先不要急着改逻辑；更稳的做法是先增量重编 `tilelang_repo/build`，确认 `libtilelang.so` 和 Python wrapper 已更新，再重跑 pytest。否则很容易把“加载了旧扩展”误判成 pass/codegen 回归。
- 对 Blackhole staged copy 的 tile-index 恢复，局部 element/worker 级 `threadIdx.*` 通常应该在 matcher 里清零，但 `blockIdx.*` 不能一起清零；否则 split 后虽然已经有 `core_plan/current_work_linear_id`，copy builtin 仍会退化回固定 `tile_index=0`。
- 对当前 single-core `grid > 1` bring-up，如果 execution plan 采用 `work_packets + current_work_linear_id`，更稳的做法是让 runner/host 逐个 logical work item 顺序 launch，而不是试图在一次 launch 里模拟完整 grid；这能先把 execution plan 和 tile-index 语义闭环，再把真正的多核 materialization 留到后续阶段。
- 对 Blackhole copy 的 large-shape 验收，真正需要区分的是“总张量数据量”与“per-core L1/CB 占用”。像 `800x1024 float16` 这类总数据量已经超过 `1.5MB` 的 case，只要 shared tile 仍是单个 `32x32` 硬件 tile，`PlanBlackholeCB` 的合法 `total_l1_bytes` 仍应保持在 `4096` bytes，而不是被总 tensor bytes 误伤。
- 当一个 target 的真实硬件编程模型天然区分：
  - host-visible object
  - transport resource（例如 CB/L1 FIFO）
  - compute-private resource（例如 Dst/tile registers / fragment）
  更稳的做法是尽早在 IR/pass 层把这三类资源分开，而不是继续让它们共用 generic `Buffer/Allocate/scope` 形态。对 Blackhole，这意味着：
  - DRAM tensor / scalar 可以继续走 host/device ABI 主线
  - `shared/shared.dyn` 不应再被默认视为 generic shared-memory buffer；它更接近 CB/L1 transport resource
  - `local.fragment` 不应再被默认视为普通 local buffer；它更接近 compute-private Dst/fragment resource
  - 如果这层区分不先收正，generic `SplitHostDevice / MakePackedAPI / LowerDeviceKernelLaunch` 只会按错误模型解释它们
  - 是否仍复用了通用 TIR 规范化 pass
  - 是否仍复用了 `AnnotateDeviceRegions` / `SplitHostDevice` / `MakePackedAPI` / `LowerDeviceKernelLaunch`
  - runtime/module 是否还在间接定义 PrimFunc 参数和 host/device 语义
- 对这类 target backend，更稳的开发顺序通常是：
  - 先做一份 pass 复用矩阵
  - 再把差异压缩到少量 target-aware 接入点
  - 最后再在这些接入点上推进 copy/gemm 等算子实现
  否则很容易在 runtime/spec/module 里积累“补洞式语义”。

## 建议的开发顺序

当前推荐顺序：

1. 统一 attrs
2. 引入 `ExecutableSpec`
3. 重构 `rt_mod_blackhole`
4. 重构 `BlackholeModule`
5. 重写 runner 协议
6. 跑通 single-core copy
7. 跑通 single-core gemm
8. 最后再做 multi-core
