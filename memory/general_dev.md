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

### 2. TT-Sim 配置

以下经验仍然有效：

- `TT_METAL_SLOW_DISPATCH_MODE=1` 对 TT-Sim 很关键。
- simulator 库和完整 soc descriptor 的路径必须明确。
- UMD 测试不完全等价于 TT-Metal 编程示例可运行性。

### 3. TT-Metal 核心接口使用经验

当前有效经验：

- TT-Metal 的稳定 host-side 抽象是：
  - `Program`
  - `CreateCircularBuffer`
  - `CreateKernel` / `CreateKernelFromString`
  - `SetRuntimeArgs`
- compile-time args 与 runtime args 是一等概念，不是实现细节。
- multi-core 调度主要是 host/runtime 责任。

## Blackhole 后端当前有效开发原则

以下内容是当前阶段应严格遵守的经验总结：

1. 不再把“单个 kernel 字符串”当成后端主产物。
2. 不再把 `SplitBlackholeKernel` 当成当前关键路径。
3. 不再让 codegen 主导多核物理映射。
4. 不再继续扩展旧 runner 的固定命令行协议。
5. 新功能优先落到 `ExecutableSpec -> runner` 这条主路径上。
6. 任何局部设计都必须服从 `final_blackhole_backend_redesign.md`。

### Stage 0 协议落地经验

当前新增的稳定经验：

- 在从旧 `BlackholeFunctionInfo` 迁移到 `ExecutableSpec` 时，先保留 TVM 调用侧最小元信息（参数类型、buffer/scalar 标记），再逐步把 runner 协议迁过去，能避免一次性打断 module 调用链。
- attr 统一不能只改 pass 或只改 runtime；至少要成对同步：
  - `PlanBlackholeCB` / `AssignBlackholeCores` 产出 `blackhole.*`
  - `rt_mod_blackhole` 读取同一套 `blackhole.*`
- `blackhole.core_plan` 这种结构化 attr 比散落的 `grid_x/grid_y/...` 标量 attr 更适合后续 spec extractor 和 runner 直接消费。
- 在切 runner 协议时，优先让 `BlackholeModule` 落 `spec.json + input.bin + output.bin + kernel.cpp`，再让 runner 从 spec 驱动创建 CB / kernel / runtime args；不要继续扩展固定位置命令行参数。
- runner 虽然依赖 `TT::Metalium`，但源码和默认构建入口都应留在 `tilelang_repo`，避免把 TileLang 自己定义的协议实现长期挂在 `tt_metal_repo` 的 programming examples 下面。
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

- 在 `global -> global` simple copy 还没有完全 lower 成通用 TT-Metal dataflow 语义时，可以先让 `rt_mod_blackhole` 对 `single_core_copy` 走一个最小专用 kernel emitter；只要主产物仍是 `ExecutableSpec`，这不会破坏总体设计。
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
  - `blackhole.target_mode`
  - `blackhole.cb_configs`
  - `blackhole.runtime_args`
  这样能直接防止语义又滑回 runtime 猜测路径。

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
