# TileLang Blackhole 后端开发进度

> 当前唯一总体设计文档: `tasks/dev_design/final_blackhole_backend_redesign.md`
> 说明: 本文档只维护阶段状态、任务拆分和当前下一步。

## 当前阶段

- **阶段**: Stage 2C split-before 语义规划 ✅ 已完成 → 下一步：通用 pass 回收（Phase 4）+ GEMM 接入
- **日期**: 2026-03-24
- **当前目标**: Stage 2C 已完成（annotation + FlattenBuffer/VectorizeLoop 验证 + StorageRewrite 不兼容性确认）；下一步进入 GEMM 接入（Stage 2D）或 通用 pass 回收（Phase 4）

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
  - `test_blackhole_e2e.py` 在 TT-Sim 环境下结果为 `18 passed, 1 skipped`
  - direct path / compile-time 负例已在同一套环境下共同验证
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

- `test_blackhole_e2e.py` 已收敛到 direct-call / compile-time 两类验证，不再依赖 legacy runner 二进制
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
- `test_blackhole_e2e.py` 的 direct 前置检查已改成优先核对”当前进程实际加载的 `libtilelang.so` 对应的 CMakeCache 是否启用 `USE_BLACKHOLE_DIRECT=ON`”
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
  - 修复后全套 E2E 验收保持 18 passed, 1 skipped
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
  - 在统一后的 `build/` 上串行验证 `testing/python/target/blackhole/test_blackhole_e2e.py` 结果为 `15 passed, 5 skipped`

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
| Stage 2D | single-core true E2E | ⏳ 未完成 | copy + GEMM 都通过正式 host-device 主路径执行 |
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
- 当前新增 blocker（Stage 2D Step 6 前置，2026-03-24）：
  - GEMM 走完整 `lower()` 时，当前会在 `MergeSharedMemoryAllocations` 失败：
    - `MergeSharedMemoryAllocations expects flat memory buffers`
  - 说明当前 `LowerTileOp` Blackhole GEMM skip 后，shared alloc 仍保持二维形态；而主线后段某些 pass 仍假设 `FlattenBuffer` 之后的一维 shared buffer
  - 这属于 Step 6 之前的新前置问题，不是 `rt_mod_blackhole` / `BlackholeModule` 的多 segment 实现本身
- 验收：`testing/python/target/blackhole/test_blackhole_e2e.py` 结果为 `16 passed, 5 skipped, 1 xfailed`

当前 Stage 2D 剩余步骤：

- Step 3: `LowerBlackholeOps` GEMM compute 的 planner-driven CB binding 已接入 codegen；剩余工作转为确认 multi-segment source generation 时继续沿用这套协议
- Step 4: `rt_mod_blackhole` 多 segment extractor 已落地；剩余工作转为随 Step 6 一起验证实际 lower/build 链路
- Step 5: `BlackholeModule` 3-kernel 注册已落地；剩余工作转为 TT-Sim / lower 级联验证
- Step 6: E2E 测试 `test_blackhole_gemm_basic`，当前被 `MergeSharedMemoryAllocations` shared-buffer flatten 前置条件阻塞

## 当前下一步

### 当前剩余事项优先级（2026-03-24 更新）

1. ~~`BlackholeModule` direct host path~~ → **已完成**
2. ~~Copy E2E 验收~~ → **已完成**（18 passed, 1 skipped，含 grid>1 / large-shape / oversubscription 负例）
3. ~~`ExecuteDirect` 核坐标 / input-output 分类 / 死代码~~ → **已修正**
4. ~~split 前语义规划（Stage 2C）~~ → **已完成**（`AnnotateBlackholeCopySemantics` + FlattenBuffer/VectorizeLoop 验证 + StorageRewrite 不兼容性确认）
5. GEMM 接入（Stage 2D / Phase 5）— **进行中（Step 4/5 已落地，Step 6 当前首要）**
   - ~~Step 1: `LowerTileOp` Blackhole GEMM skip~~ → **已完成**
   - ~~Step 2: `SplitBlackholeKernel` pass~~ → **已完成**
   - ~~Step 3: `LowerBlackholeOps` GEMM lower / planner-driven CB binding~~ → **已完成**
   - ~~Step 4: `rt_mod_blackhole` 多 segment extractor~~ → **已完成**
   - ~~Step 5: `BlackholeModule` 3-kernel 注册~~ → **已完成**
   - Step 6: E2E 测试 → **进行中**（当前被 `MergeSharedMemoryAllocations` / shared flatten 前置条件阻塞）
6. 分批接回中后段通用 pass（Phase 4）— 可并行或后置

### 当前具体下一步

1. 处理 GEMM `lower()` 当前的 shared-buffer flatten blocker（`MergeSharedMemoryAllocations expects flat memory buffers`）
2. 在 blocker 解除后重跑 `test_blackhole_gemm_basic` direct path（Step 6）
3. 回填 Step 4/5 的实际 TT-Sim 验收结果

## 当前活动设计文档

- `tasks/dev_design/final_blackhole_backend_redesign.md`
- `tasks/dev_design/stage2_pass_reuse_matrix.md`
- `tasks/dev_design/stage2_single_core_pass_integration.md`
- `tasks/dev_design/stage2_blackhole_logical_block_launch_plan.md`
- `tasks/dev_design/stage2_concrete_dev_task_plan.md`
