# TileLang Blackhole 后端开发进度

> 当前唯一总体设计文档: `tasks/dev_design/final_blackhole_backend_redesign.md`
> 说明: 本文档只维护阶段状态、任务拆分和当前下一步。

## 当前阶段

- **阶段**: Stage 2D Step 6 — GEMM direct-path TT-Metal contract 收正
- **日期**: 2026-03-25
- **已完成目标**: Stage 2E 已完成；CB identity 唯一协议已按设计收正并通过 compile/lower 回归；GEMM direct path 已从 enqueue deadlock 推进到真实执行完成
- **当前测试结果**：
  - `test_blackhole_copy_pipeline.py`: `16 passed, 1 xfailed`
  - `test_blackhole_copy_runtime.py`（TT-Sim 环境）: grid-indexed direct call 已重新通过
  - `test_blackhole_gemm.py`: 结构层 `3 passed, 1 skipped`
  - `test_blackhole_gemm_basic`：已进入 direct execution 并返回结果，但数值错误
- **当前 blocker**：GEMM direct path 的显性断点仍是 `CodeGenBlackhole` 的 `read_tile_to_cb/write_tile_from_cb` 没有走真实 TT-Metal CB backing store；但本轮对照 TT-Metal 后已确认，这背后是更大的 contract 缺层：host logical layout、TensorAccessor schema、packed dtype vs tensor dtype、transpose/compute ABI、runtime work description 都还没有正式进入 `ExecutableSpec`
- **修正设计文档**：`tasks/dev_design/stage2d_cb_identity_protocol.md`
- **活动设计文档**：`tasks/dev_design/stage2d_gemm_direct_cb_io.md`
- **补充审计文档**：`tasks/dev_design/stage2d_ttmetal_contract_audit.md`
- **下一步**: 先按 `stage2d_ttmetal_contract_audit.md` 把 Stage 2D 当前缺失的最小正式 contract 收束成 schema，再继续实现真实 CB dataflow IO、transpose、layout/accessor 相关修正，之后恢复 `test_blackhole_gemm_basic` true E2E 验收

## 当前状态判断

- Stage 0 的协议与执行载体已经落地：
  - `ExecutableSpec`
  - `rt_mod_blackhole`
  - `BlackholeModule`
- Blackhole 已重新接回正式 host/device 主链：
  - `AnnotateDeviceRegions`
  - `SplitHostDevice`
  - `MakePackedAPI`
  - `LowerDeviceKernelLaunch`
- 基于本轮 TT-Metal / TT-Metalium 文档、示例和 pass 链复盘，当前已确认：
  - Blackhole/TT-Metal 编程模型里的核心对象至少分三层：
    - host-visible：DRAM tensor / scalar / runtime args / kernel handle
    - transport resource：CB / L1 FIFO / producer-consumer 协议
    - compute-private：Dst/tile registers / fragment / accumulator / unpack-pack-FPU 配置
  - 当前 `shared/shared.dyn` 与 `local.fragment` 仍以 generic TIR buffer/scope 形态进入 host/device 主线，语义层级不足
  - 这会让 generic pass 误把 device-private resource 当成：
    - ABI 参数
    - generic dynamic shared memory
    - launch-time shared-memory slab
  - 因此当前 GEMM blocker 本质上不是 “`gemm_py` 特例”，而是 **Blackhole device resource semantics 尚未在 IR 中显式承载**
- copy 已开始从 runtime 特化迁回 pass / builtin / codegen 主链：
  - `blackhole.runtime_args`
  - `blackhole.segment_plan`
  - `blackhole.cb_requirements`
  - `blackhole.cb_configs`
  - `current_work_linear_id`
  - `tl.blackhole.read_tile_to_cb / write_tile_from_cb`
- `AssignBlackholeCores` 已开始产出正式 execution-plan 元数据：
  - `logical_grid_x/y`
  - `linearization = row_major`
  - `physical_cores`
  - `work_packets`
- `LowerBlackholeOps` 已开始为 grid-indexed staged copy 保留 `bx/by -> tile_index` 公式
- `BlackholeModule::ExecuteDirect()` 已开始按 `work_packets/current_work_linear_id` 顺序执行 single-core logical work items
- staged copy 的最小 direct execution 已覆盖：
  - `32x32`
  - `32x64`
  - `64x32`
- Stage 2B copy E2E 验收已完成：
  - 当前测试入口已拆分为：
    - `testing/python/target/blackhole/test_blackhole_copy_pipeline.py`
    - `testing/python/target/blackhole/test_blackhole_copy_runtime.py`
  - 本地当前结果：
    - `test_blackhole_copy_pipeline.py`: `14 passed, 1 skipped, 1 xfailed`
    - `test_blackhole_copy_runtime.py`: `1 passed, 4 skipped`
  - 当前 `copy_runtime` 的 skip 原因是环境未配置 `TT_METAL_RUNTIME_ROOT`，不是新的功能回退
- `grid > 1` 且 `bx/by` 参与索引的 direct-call staged copy 已在 TT-Sim 上通过：
  - `grid_x=2`
  - `grid_y=3`
  - `96x64 float16`
- large-shape staged copy（总数据量 > `1.5MB`）已在 TT-Sim 上通过：
  - `800x1024 float16`
  - `1638400` bytes
- per-core memory plan oversubscription 负例已补齐：
  - `1024x1024` shared tile copy
  - `PlanBlackholeCB` 编译期直接失败
- `LowerBlackholeOps -> PlanBlackholeCB` 的 memory-plan schema 已进一步收正：
  - `blackhole.cb_requirements` 显式带 `lifetime_begin/end`
  - `blackhole.cb_configs` 显式带 `role`
  - `blackhole.cb_configs` 显式带 `total_size_bytes`
  - `blackhole.cb_configs` 显式带 `lifetime_begin/end`
- `PlanBlackholeCB` 已开始做保守的 lifetime-aware reuse：
  - 同 role
  - 同 `page_size/num_pages/data_format`
  - lifetime 不重叠
  - `cb_configs` 只保留真正要 materialize 的 memory object
  - `cb_configs.requirement_names` 记录被合并的 requirement 名集合
- `PlanBlackholeCB` 已补 requirement-to-memory-object 显式绑定协议：
  - `blackhole.cb_bindings.requirement_index`
  - `blackhole.cb_bindings.requirement_name`
  - `blackhole.cb_bindings.cb_id`
  - `blackhole.cb_bindings.cb_config_index`
- 当前新增结论（2026-03-25）：
  - Stage 2E 与当前 GEMM direct-path blocker 不冲突
  - Stage 2E 解决的是 device resource 语义承载（`blackhole.acc` / `blackhole.cb.*`）
  - 当前 direct-path blocker 是更靠后的 CB identity 唯一性问题：
    - `LowerBlackholeOps` 仍同时产出局部 CB id（如 `0/1/16`）和 GEMM placeholder CB id（`-1/-2/-3`）
    - `PlanBlackholeCB` 允许同名 `requirement_name` 出现多份不同 binding，`requirement_name` 不再是唯一键
    - `CodeGenBlackhole` / segment source generation 继续按名字恢复 binding 时，会让 reader/compute/writer 取到不同的 CB 身份
  - 当前这一步的真实问题不是 TT-Metal `cb_reserve_back` API 或枚举类型，而是后端内部没有收敛”谁是 CB identity 的唯一真源”
  - **问题是通用架构缺陷，不是 GEMM 特有**：任何 multi-segment kernel 模式（未来的 fused pipeline、multi-stage copy 等）都会碰到
  - **设计分析结论**：pass 设计本身没问题，问题在实现偏离了设计。设计文档已规定 LowerBlackholeOps “不分配最终 cb_id”，但 copy 路径直接写入了实际 id；PlanBlackholeCB 只写 attrs 不回写 IR body
  - **修正方案**：统一用 requirement_index → PlanBlackholeCB 回写 IR → codegen 直接读最终 cb_id。详见 `tasks/dev_design/stage2d_cb_identity_protocol.md`
  - **当前实现状态（2026-03-25 晚）**：
    - `LowerBlackholeOps` 已统一把 copy/GEMM 的 CB 参数写成 `requirement_index`
    - `PlanBlackholeCB` 已回写 IR body，把 `requirement_index` 统一替换成最终 `cb_id`
    - `CodeGenBlackhole` / `SplitBlackholeKernel` 已删除 `gemm_cb_placeholders` 和 placeholder/alias 修补逻辑
    - GEMM requirement 的 `lifetime` 已与 `requirement_index` 解耦，避免 planner 误把 `A/B` 输入 CB 复用到同一个 FIFO
    - 本地验证：
      - `pytest -q testing/python/target/blackhole/test_blackhole_copy_pipeline.py` → `15 passed, 1 xfailed`
      - `pytest -q testing/python/target/blackhole/test_blackhole_gemm.py` → `3 passed, 1 skipped`
- planner protocol struct 已进一步收敛：
  - `CBType/CBRequirement` 已集中到共享头文件
  - 不再依赖 `lower_blackhole_ops.h` / `plan_blackhole_cb.h` 的重复定义保持人工同步
- Blackhole host/device 设备函数识别已不再依赖过渡标签：
  - `tilelang.engine.lower.is_device_call()` 不再把 `blackhole.target_mode` 视为 device-kernel 判据
  - 只有正式 `calling_conv` 或正式 `blackhole.*` plan attrs 才参与 Blackhole device function 识别

基于源码审查的新进展（2026-03-20）：

- `blackhole_module.cc` 已完成 direct path 补全（Phase 1 代码实现）：
  - `ExecuteDirect()` 方法：直接调用 TT-Metal API
  - `CreateCircularBuffersFromSpec()`：按 spec 创建所有 CB
  - `BuildRuntimeArgsFromSpec()`：按 `KernelArgSpec.kind` 逐项构造
  - work-packet 迭代：遍历 `work_packets` 为每个 work unit 创建独立 Program
  - role-aware `ChoosePageSize()` 用于 DRAM buffer 创建
- `blackhole_module_direct.cc` 已合并到 `blackhole_module.cc` 后删除
- CMakeLists.txt 新增 `USE_BLACKHOLE_DIRECT` 编译选项
- 架构审查文档已同步更新（五刀方案逐项评估）

基于本轮推进的新进展（2026-03-23）：

- Blackhole 测试已拆分为按关注点分层的入口，不再依赖 legacy runner 二进制：
  - `test_blackhole_copy_pipeline.py`
  - `test_blackhole_copy_runtime.py`
  - `test_blackhole_gemm.py`
- direct 模式的 CMake 接入已补第一轮构建对齐：
  - 加入 TT-Metal repo root / `tt_stl` / `hostdevcommon` / `umd` 相关 include 路径
  - direct 模式编译标准提升到 C++20 以匹配 TT-Metal 头文件要求
- direct 依赖发现已从”整片 `.cpmcache` include sweep”收缩到：
  - `fmt` / `nlohmann_json` / `spdlog` 走 `build_Release/_deps/*-build` 的 package config
  - `tt-logger` / `enchantum` / `umd_asio` 只按必需头文件做定向发现
- direct 模式已开始优先消费 TT-Metal local install tree：
  - `TT_METAL_BUILD_DIR=/root/dev/vibe_dsl/tt_metal_repo/build_Release`
  - `TT_METAL_INSTALL_DIR=/root/dev/vibe_dsl/tt_metal_repo/build_Release/stage`
  - `find_package(tt-metalium CONFIG REQUIRED)` 成功
- `blackhole_module.cc` 已可用 direct 模式单文件编译通过
- `tilelang_repo/build/` 的完整 `tilelang` 目标已在 direct 配置下全量构建通过（`cmake --build ... --target tilelang -j32`）
- direct-call 测试现已确认会真正进入 `BlackholeModule::ExecuteDirect()`
- 当前收尾约定已改回单一开发构建目录：
  - 以后统一以 `tilelang_repo/build/` 为准
  - 旧 `build_blackhole/` 过渡目录已删除，避免继续误用
  - 如需临时指向其他构建目录，仍使用 `TILELANG_DEV_LIB_ROOT`
- `direct` 类测试的前置检查已改成优先核对”当前进程实际加载的 `libtilelang.so` 对应的 CMakeCache 是否启用 `USE_BLACKHOLE_DIRECT=ON`”
- 当前 shell 的 TT-Sim 环境已通过 `scripts/setup_tt_sim.sh` 恢复：
  - 官方 `metal_example_add_2_integers_in_riscv` smoke test 已在本机再次跑通
  - direct path 已在 TT-Sim 上通过 `32x32` / `32x64` / `64x32` / `grid>1` / `large-shape`
- `BlackholeModule::ExecuteDirect()` 已补唯一 kernel 临时目录：
  - 避免同一 pytest 进程内多个 direct-call case 复用同一路径触发 TT-Metal JIT 缓存串扰
  - 修复”单测单跑通过、组合跑 large-shape / rectangular 错结果”的问题
- `blackhole_module.cc` 已修正若干 Stage 2B 遗留 bug（2026-03-23）：
  - `ExecuteDirect()` 核坐标：从硬编码 `{0,0}` 改为读 `work_packet.core_x/core_y`；`{0,0}` 在真实硬件上不是合法 Tensix core
  - `BlackholeWrappedFunc::operator()` input/output 分类：改为按 `runtime_args` 的 kind（`input_buffer_addr32` / `output_buffer_addr32`）顺序判定，不再依赖”最后一个 buffer = output”位置启发式
  - 删除死代码：`EnsureDeviceInitialized()`、`GetOrCompileProgram()`、`CompiledProgram` struct、`mesh_device_`/`mesh_command_queue_`/`device_initialized_`/`program_cache_`（direct path 每次调用自建局部 `MeshDevice`，这套成员从未被触达）
  - `MakeUniqueTempDir()` 用于 direct path 内部的唯一 kernel 临时目录，消除同进程内多次调用路径冲突
  - 修复后 copy/runtime 相关测试仍保持通过；当前测试结果以拆分后的 `copy_pipeline` / `copy_runtime` 入口为准
- Stage 2C 本轮推进已补第一轮实现与专项验证（2026-03-23）：
  - `AnnotateBlackholeCopySemantics` 已从旧的 `AttrStmt/string` 方案收正为 `ForNode::annotations["blackhole.copy_semantics"]`
  - copy 语义 schema 已改为结构化 `Map<String, Any>`，不再依赖冒号拼接字符串协议
  - schema 当前显式带：
    - `kind`
    - `direction`
    - `src_buffer` / `dst_buffer` / `mid_buffer`
    - `src_scope` / `dst_scope`
    - `dtype`
    - `src_shape` / `dst_shape` / `mid_shape`
  - `LowerBlackholeOps` 已开始优先消费上述 loop annotation，再回退到旧 matcher
  - `LowerBlackholeOps` 已补 shape-aware tile-index 恢复：
    - 可从 `FlattenBuffer` 后的线性化 global/shared shape 恢复 tile 语义
    - 可从 `VectorizeLoop` 后的 `Ramp(base, 1, lanes)` 索引恢复标量 tile 基址
  - 已新增 Stage 2C 专项测试：
    - split-before annotation schema 产出检查
    - `AnnotateBlackholeCopySemantics -> FlattenBuffer -> VectorizeLoop -> LowerBlackholeOps` 稳定性检查
  - 在统一后的 `build/` 上，当前对应验证入口为：
    - `testing/python/target/blackhole/test_blackhole_copy_pipeline.py`
    - 结果：`14 passed, 1 skipped, 1 xfailed`

基于本轮状态核查的新同步（2026-03-24）：

- 当前 Blackhole 测试文件布局为：
  - `testing/python/target/blackhole/test_blackhole_copy_pipeline.py`
  - `testing/python/target/blackhole/test_blackhole_copy_runtime.py`
  - `testing/python/target/blackhole/test_blackhole_gemm.py`
- 当前本地核查结果：
  - `test_blackhole_copy_pipeline.py`: `14 passed, 1 skipped, 1 xfailed`
  - `test_blackhole_copy_runtime.py`: `1 passed, 4 skipped`
  - `test_blackhole_gemm.py`: `1 passed, 2 skipped`
- 当前 GEMM 的两个 skip 分别对应：
  - direct runtime 环境未配置 `TT_METAL_RUNTIME_ROOT`
  - lowering 仍卡在 `MergeSharedMemoryAllocations expects flat memory buffers`

当前仍然存在的主要结构问题：

- `PlanBlackholeCB` 仍偏 MVP allocator，尚未成为正式 memory planner
- `StorageRewrite` 确认不兼容 Blackhole CB 模型（VectorTypeAccessChecker 不识别 DeclBuffer），永久排除于 Blackhole pipeline；Phase 4 若需引入必须先加 shared-scope 豁免
- GEMM 仍未接入正式 direct host path
- **架构债（已记录）**：copy 当前用 `fused_dataflow` 单 kernel（BRISC 顺序 read+write），GEMM 用 3-kernel（reader+compute+writer）。两种 schema 并存导致 `rt_mod_blackhole` / `BlackholeModule` 需要维护双重路径。后续应将 copy 也统一进 reader+writer 2-kernel 模型，消除不对称。触发条件：GEMM E2E 稳定后。

基于 Stage 2C 完成的新结论（2026-03-24）：

- `AnnotateBlackholeCopySemantics` 已完全落地（ForNode annotation schema）
- `FlattenBuffer` + `VectorizeLoop` 已在 Stage 2C 范围内专项验证通过
- `StorageRewrite` 不兼容性已通过 `xfail` 测试记录并文档化
- Stage 2C 验收：`15 passed, 5 skipped, 1 xfailed`（全部符合预期）

当前新增设计收束：

- 已明确 Blackhole 应采用三层模型：
  - split 前语义规划
  - split 后正式 plan 提取
  - host-side materialization
- 已明确 external runner 只是 bring-up/debug 工具：
  - 不是正式执行路径
  - 不是阶段完成标准
  - 相关代码与测试入口已删除
- 已明确 copy 的正式验收必须补齐：
  - `grid > 1`
  - `bx/by` 参与索引
  - large-shape copy（总数据量 > `1.5MB`）
  - per-core memory plan oversubscription 负例

## 分阶段任务

| 阶段 | 目标 | 状态 | 当前重点 |
|------|------|------|----------|
| Stage 0 | 协议与执行载体 | ✅ 已基本完成 | `ExecutableSpec`、`rt_mod_blackhole`、`BlackholeModule` 已落地 |
| Stage 1 | single-core copy bring-up | ✅ 已完成 | 最小 copy 路径已 bring-up，但不再扩大为正式主线 |
| Stage 2A | pass 主链接入收正 | 🔄 进行中 | 固定 split 前语义规划 / split 后正式 plan 提取 / host-side materialization 三层 |
| Stage 2B | single-core copy 正式主链 | ✅ 已完成 | direct path copy E2E 已在 TT-Sim 上验收通过 |
| Stage 2C | split-before 语义规划 | ✅ 已完成 | annotation + FlattenBuffer/VectorizeLoop 验证通过；StorageRewrite 确认不兼容 Blackhole CB 模型，永久排除 |
| Stage 2D | single-core true E2E | 🔄 进行中（Step 1-5 已完成；Step 6 当前为 direct-path E2E 验收） | copy + GEMM 都通过正式 host-device 主路径执行 |
| Stage 2E | Blackhole 设备资源 IR 语义扩展 | ✅ 已完成 | StorageRank 扩展 + 规范化 pass 已落地，GEMM generic pass 阻塞已解除 |
| Stage 3 | multi-core runtime 调度 | ⏳ 未开始 | `CorePlan` 已补 formal schema，后续补 per-core runtime args 与多核执行 |

## Stage 2 当前任务拆分

### 任务 1: 固定三层边界

- split 前语义规划
- split 后正式 plan 提取
- `BlackholeModule` direct host-side materialization

### 任务 2: 收正 split 前语义规划

- 在 `LowerTileOp` 保留 copy/gemm 的 Blackhole-preserving 语义
- 不再把 split 后 matcher 作为唯一语义来源

### 任务 3: 收正 split 后 requirement extraction

- `LowerBlackholeOps` 正式提取：
  - `blackhole.segment_plan`
  - `blackhole.runtime_args`
  - `blackhole.cb_requirements`

### 任务 4: 收正 memory planner

- `PlanBlackholeCB` 生成正式 `blackhole.cb_configs`
- `cb_id` deterministic allocation
- `role + lifetime` 复用
- `1572864` bytes hard check

### 任务 5: 收正 execution planner

- `AssignBlackholeCores` 生成正式 `blackhole.core_plan`
- 补足：
  - `logical_grid_x/y`
  - `linearization`
  - `physical_cores`
  - `work_packets`

### 任务 6: 收正 `BlackholeModule` direct path ✅

- 不再依赖 external runner 作为正式执行路径
- 在模块内直接 materialize TT-Metal host objects
- 核坐标从 work_packet 读取，input/output 按 runtime_args kind 区分

### 任务 7: 用 copy 完成正式 E2E

- staged copy
- `grid > 1`
- `bx/by` 参与索引
- large-shape copy
- oversubscription 负例

基于 Stage 2D Step 1-2 完成的新进展（2026-03-24）：

- `LowerTileOp` 已对 Blackhole target 跳过 `GemmPyNode` 展开（Step 1，已完成）
- `SplitBlackholeKernel` pass 已实现并接入管线（Step 2，已完成）：
  - 扫描 func body 中含 `gemm_py` 调用的 `SeqStmt`，按状态机分类每个 stmt
  - `dram_to_cb` ForNode → reader，`gemm_py` EvaluateNode → compute，DRAM 输出 ForNode → writer
  - 对 `past_compute` + `dst_scope == "global"` 的 copy 正确识别为 writer（处理 `local.fragment → global` 情形）
  - 写入 `blackhole.segment_plan`（3-kernel schema，reader/compute/writer）
  - 纯 copy 函数不触发（无 compute op 时直接返回）
- `LowerBlackholeOps` 已移除 `StoreGemmSegmentPlan` 分叉逻辑：
  - `StoreSegmentPlan` 现在先检查 `blackhole.segment_plan` 是否已由 `SplitBlackholeKernel` 写入
  - 若已写入则跳过，保留 3-kernel schema；否则走 fused_dataflow 单 kernel 路径
- Stage 2D Step 3 已补第一轮 planner-driven CB binding 收口（2026-03-24）：
  - `LowerBlackholeOps` 的 GEMM compute builtin 不再把 `0/1/16` 当最终硬件 `cb_id`
  - 当前改为写入显式 placeholder CB IDs，并同步写出 `blackhole.gemm_cb_placeholders`
  - `CodeGenBlackhole` 已开始读取：
    - `blackhole.gemm_cb_placeholders`
    - `blackhole.cb_bindings`
    并在 codegen 阶段把 placeholder 解析成 planner 最终分配的 `cb_id`
  - 这样 GEMM compute 已不再把 allocator 当前顺序偷渡成协议
  - 聚焦验证：
    - `cmake --build tilelang_repo/build --target tilelang -j8`
    - `pytest -k 'gemm_cb_placeholders_resolve_via_planner or split_kernel_gemm_segment_plan or copy_pass_attrs'`
    - 结果：`2 passed, 1 skipped`
- Stage 2D Step 4/5 已补第一轮实现（2026-03-24）：
  - `rt_mod_blackhole` 不再只消费 `segment_plan[0]`
  - `KernelArgSpec` 已扩展 `buffer` 字段，segment-level runtime args 可显式绑定 `A/B/C`
  - `ExecutableSpec` 已额外保留 `tvm_arg_names`
  - `rt_mod_blackhole` 现已为 reader / compute / writer 生成 segment-specific `PrimFunc`
    - body 只保留当前 `blackhole.segment_kind`
    - `blackhole.core_type` / `blackhole.runtime_args` 按当前 segment 改写
    - 继续复用 `CodeGenBlackhole`
  - `BlackholeModule::ExecuteDirect()` 已改为按 TVM buffer 参数分别创建 DRAM `MeshBuffer`
  - `BuildRuntimeArgsFromSpec()` 已可按 `KernelArgSpec.buffer` 绑定多输入/输出 buffer 地址
  - `CreateKernelFromSpec()` 已按 kind/core_type 路由 reader(BRISC) / compute(TRISC) / writer(NCRISC)
- 历史 blocker（Stage 2D Step 6 前置，2026-03-24，已由 Stage 2E 解决）：
  - GEMM 走完整 `lower()` 时，当前会在 `MergeSharedMemoryAllocations` 失败：
    - `MergeSharedMemoryAllocations expects flat memory buffers`
  - 进一步沿 pass 链定位后，当前已确认这只是最早暴露的一个 generic pass 断点；同一根因还会表现为：
    - `In PrimFunc main variables [C_local] are used, but are not passed in as API arguments`
    - `Only one dynamic shared memory allocation is allowed`
  - 根因已从“GEMM shared buffer 没 flatten”收敛为更一般的问题：
    - `shared/shared.dyn` 在 Blackhole 上本质更接近 CB/L1 transport resource，不等价于 generic shared buffer
    - `local.fragment` 在 Blackhole 上本质更接近 compute-private Dst/fragment resource，不应越过 host/device ABI 边界
    - 当前这些资源仍以普通 TIR buffer/var/scope 进入 `SplitHostDevice / MakePackedAPI / LowerDeviceKernelLaunch`
  - 这属于当时 Step 6 之前的结构性前置问题，不是 `rt_mod_blackhole` / `BlackholeModule` 的多 segment 实现本身
- 本轮设计收敛（2026-03-25）：
  - 已新增通用设计文档：
    - `tasks/dev_design/stage2e_blackhole_device_resource_semantics.md`
  - 新设计方向不是在 `MakePackedAPI` / `LowerDeviceKernelLaunch` 上给 Blackhole 开特判，而是：
    - 在 `SplitHostDevice` 之前新增 `BlackholeDeviceResourceCanonicalization`
    - 显式区分：
      - host-visible resources
      - transport resources
      - compute-private resources
    - 让后续 GEMM / 非 GEMM compute op 共用统一语义层，而不是继续按算子补 workaround
- Stage 2E 已完成（2026-03-25）：
  - `StorageRank` 已扩展：
    - `kBlackholeCB`
    - `kBlackholeAccumulator`
  - `BlackholeDeviceResourceCanonicalization` 已接入正式管线
  - GEMM `lower()` 已通过，先前三个 generic pass 阻塞已解除：
    - `MergeSharedMemoryAllocations expects flat memory buffers`
    - `variables [C_local] are used, but are not passed in as API arguments`
    - `Only one dynamic shared memory allocation is allowed`
  - 当前 Stage 2D Step 6 的剩余工作已从“打通 lower()”切换为“验证 direct path 实际执行”
- 本轮回归修正（2026-03-24）：
  - Blackhole 测试文件已按关注点拆分：
    - `common.py`
    - `test_blackhole_copy_pipeline.py`
    - `test_blackhole_copy_runtime.py`
    - `test_blackhole_gemm.py`
  - `rt_mod_blackhole` 当前已收正为：
    - 纯 copy 继续沿用原有 single-kernel build/codegen 主路径
    - 只有 multi-segment GEMM 才走 segment-specific codegen
  - `CodeGenBlackhole` 的 copy runtime-arg buffer 绑定回归已修复：
    - 不再因为 `A.data` / `B.data` 这类 builtin buffer expr 进入 codegen 时丢失绑定
    - copy codegen / host-device split / rectangular-shape / logical-block 相关回归已恢复通过
- 当前目录级验收：`tilelang_repo/testing/python/target/blackhole/` 结果为 `16 passed, 7 skipped, 1 xfailed`

当前 Stage 2D 剩余步骤：

- Step 3: `LowerBlackholeOps` GEMM compute 的 planner-driven CB binding 已接入 codegen；剩余工作转为确认 multi-segment source generation 时继续沿用这套协议
- Step 4: `rt_mod_blackhole` 多 segment extractor 已落地；剩余工作转为随 Step 6 一起验证实际 lower/build 链路
- Step 5: `BlackholeModule` 3-kernel 注册已落地；剩余工作转为 TT-Sim / lower 级联验证
- Step 6: E2E 测试 `test_blackhole_gemm_basic`，`lower()` 前置阻塞已解除；当前剩余工作是 direct runtime 环境就绪后的执行验收

## 当前下一步

### 当前剩余事项优先级（2026-03-24 更新）

1. ~~`BlackholeModule` direct host path~~ → **已完成**
2. ~~Copy E2E 验收~~ → **已完成**（18 passed, 1 skipped，含 grid>1 / large-shape / oversubscription 负例）
3. ~~`ExecuteDirect` 核坐标 / input-output 分类 / 死代码~~ → **已修正**
4. ~~split 前语义规划（Stage 2C）~~ → **已完成**（`AnnotateBlackholeCopySemantics` + FlattenBuffer/VectorizeLoop 验证 + StorageRewrite 不兼容性确认）
5. GEMM 接入（Stage 2D / Phase 5）— **进行中（Step 1-5 已落地；Step 6 blocker 已定位，前置修正设计已完成）**
   - ~~Step 1: `LowerTileOp` Blackhole GEMM skip~~ → **已完成**
   - ~~Step 2: `SplitBlackholeKernel` pass~~ → **已完成**
   - ~~Step 3: `LowerBlackholeOps` GEMM lower / planner-driven CB binding~~ → **已完成（placeholder 方案将被 CB identity 唯一协议取代）**
   - ~~Step 4: `rt_mod_blackhole` 多 segment extractor~~ → **已完成**
   - ~~Step 5: `BlackholeModule` 3-kernel 注册~~ → **已完成**
   - Step 6 前置修正: CB identity 唯一协议收正 → **设计已完成，待实施**（`stage2d_cb_identity_protocol.md`）
   - Step 6: E2E 测试 → **待前置修正完成后恢复**
6. 分批接回中后段通用 pass（Phase 4）— 可并行或后置
7. 通用设备资源语义收正（Stage 2E）— **已完成**

### 当前具体下一步

执行 Stage 2D Step 6 前置修正：CB Identity 唯一协议收正。相关设计文档：

- `tasks/dev_design/stage2d_cb_identity_protocol.md`（本次修正的详细设计）
- `tasks/dev_design/stage2d_gemm_integration.md`
- `tasks/dev_design/stage2e_blackhole_device_resource_semantics.md`

修正完成后恢复 Stage 2D Step 6 GEMM direct-path E2E 验收。

## Stage 2E 任务拆分

### Step 1: 扩展 StorageRank / StorageScope（IR 层）✅

**文件**：`3rdparty/tvm/src/runtime/thread_storage_scope.h`

- 新增 `kBlackholeCB = 13`、`kBlackholeAccumulator = 14`
- 更新 `StorageScope::Create()` 解析 `"blackhole.cb"` / `"blackhole.acc"` 前缀
- 更新 `StorageScope::to_string()` 新增对应 case
- 验证：构建通过

### Step 2: 新 pass `BlackholeDeviceResourceCanonicalization` 实现 ✅

**新建**：`src/transform/blackhole_device_resource_canonicalization.cc`

- Phase 1：资源分类（BlackholeResourceClassifier）
  - 按 scope + 使用模式分类：cb / accumulator / abi / local_scratch
  - 利用 `blackhole.copy_semantics` annotation 判断 CB role
  - 利用 `gemm_py` 调用参数判断 accumulator
- Phase 2：Scope 重写 + Allocation 重定位（BlackholeResourceCanonicalizer）
  - `shared.dyn` → `blackhole.cb.input` / `blackhole.cb.output`
  - `local.fragment` / `local`(gemm C) → `blackhole.acc`
  - device-private Allocate 移回 `thread_extent` AttrStmt 内部
  - 附加 `blackhole.resource_decl` annotation
- Phase 3：写 `blackhole.resource_plan` attr
- 验证：构建通过 + 单元测试

### Step 3: Python 注册 + 管线接入 ✅

- `tilelang/transform/__init__.py` 新增 FFI 绑定
- `tilelang/engine/phase.py` 在 `AnnotateBlackholeCopySemantics` 之后、`AnnotateDeviceRegions` 之前接入
- `src/transform/CMakeLists.txt` 添加新源文件
- 验证：copy pipeline 回归通过

### Step 4: 下游 pass 更新 ✅

- `codegen_blackhole.cc`：Allocate 跳过 + PrintStorageScope 用 rank 替代字符串
- `lower_device_storage_access_info.cc`：新 rank 白名单
- `lower_blackhole_ops.cc`：CB 识别改为 `rank == kBlackholeCB`
- 验证：copy pipeline 回归 + GEMM lower 通过

### Step 5: GEMM 解锁验证 ✅

- `test_blackhole_gemm.py::test_gemm_lower_basic` 通过
- 三个 generic pass 错误不再出现：
  - `MergeSharedMemoryAllocations expects flat memory buffers`
  - `variables [C_local] are used, but are not passed in as API arguments`
  - `Only one dynamic shared memory allocation is allowed`

### Stage 2E 收尾结论

- Stage 2E 已完成，其职责边界停留在：
  - 扩展 IR 资源类型系统
  - 在 generic host/device pass 前完成 device-private resource canonicalization
  - 解除 GEMM `lower()` 的结构性阻塞
- 它不再是 Stage 2D Step 6 的当前 blocker
- 当前剩余风险转为 runtime 环境与 direct-path 执行验收，而不是 IR/pass 语义缺失
- 新增结构验证 test case：
  - 规范化后 IR 无 `shared.dyn` / `local.fragment`
  - `blackhole.resource_plan` 存在且分类正确
  - 设备函数参数只含 DRAM tensor

## 当前活动设计文档

- `tasks/dev_design/final_blackhole_backend_redesign.md`
- `tasks/dev_design/stage2d_gemm_integration.md`
- `tasks/dev_design/stage2d_cb_identity_protocol.md`
- `tasks/dev_design/stage2e_blackhole_device_resource_semantics.md`
- `tasks/dev_design/stage2_pass_reuse_matrix.md`
- `tasks/dev_design/stage2_single_core_pass_integration.md`
- `tasks/dev_design/stage2_blackhole_logical_block_launch_plan.md`
