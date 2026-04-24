# Task 5: Final Convergence, Documentation Sync, And Verification

## 1. 任务目标

task5 不是新的协议删除任务，
也不是
“前四个 task 还没收干净时，
最后补一轮 grep / build / push
就算完成”的扫尾脚本。

它只负责一件事：

> 在 task0-task4
> 都已经把各自的 wrong-now carrier
> 收回到
> `Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec`
> 这条显式链之后，
> 把最终状态同步到文档 / 审计 / 记忆 / 验证矩阵，
> 并证明 active chain
> 里没有遗漏的 forbidden residue。

因此 task5 的正确 end-state
不是“代码刚好能跑”或“已经 push”；
而是：

- 前面 cleanup task
  定义的删除目标
  已经真的退出 active chain
- `tasks/progress.md`、
  `tasks/dev_design/README.md`、
  cleanup docs、
  protocol audit
  的口径一致
- `memory/`
  只记录稳定经验和可复用 bug taxonomy，
  不承担阶段状态快照职责
- 验证矩阵只把
  当前 admitted support surface
  当成 hard gate，
  不把未来 workload payoff
  误写成 cleanup 完成条件

## 2. 范围

### 2.1 允许做的事

- 同步
  `tasks/progress.md`
  / `tasks/dev_design/README.md`
  / cleanup task docs
  / `blackhole_first_principles_protocol_audit.md`
  到真实 end-state
- 把本轮 cleanup
  里稳定下来的架构经验、
  runtime gate 经验、
  simulator 边界
  记录进 `memory/general_dev.md`
  / `memory/bugs.md`
- 按前面 task 的 completion contract
  运行源码 residue scans
- 运行 full build /
  compile-path /
  projection-path /
  admitted direct-runtime
  verification
- 确认没有残留后台命令、
  悬挂测试、
  长时间构建进程
- 在以上都完成后，
  再做 repo workflow
  要求的 `git status` /
  `git commit` /
  `git push`

### 2.2 不允许做的事

- 用 task5
  本地豁免
  task0-task4
  还没删除的 residue
- 把 task5
  写成新的协议 owner，
  重新定义 task0-task4
  的边界
- 把阶段状态、
  临时 checkpoint、
  “当前还没做完什么”
  写进 `memory/`
- 把
  `flash-attn`
  direct-runtime correctness
  写成 cleanup hard gate
- 把 TT-Sim `fp16`
  写成当前 correctness gate
- 把
  `git commit` / `git push`
  当成架构完成的定义

## 3. 完成更新 (`2026-04-23`)

当前 task5
已经完成
convergence /
delivery gate。

本轮 repo HEAD
完成证据：

- `rg -n "blackhole\\.resource_plan|tl\\.internal_tt_" tilelang_repo/src tilelang_repo/tilelang`
  已为零命中；
  `blackhole.resource_plan`
  与
  `tl.internal_tt_*`
  定义面
  已退出
  active source tree
- `AnalyzeBlackholeLoweringSupportFacts`
  旧命名
  已退出；
  lowering support
  只剩
  pass-local
  `CollectBlackholeLoweringSupportFacts`
  helper，
  不再作为
  public analysis /
  protocol surface
- `blackhole.segment_kind`
  只剩
  `lower_blackhole_ops.cc`
  内部的
  pass-local mechanics /
  final strip，
  不再被
  leaf reader /
  runtime
  当作 cross-pass protocol
- admitted runtime gate
  已明确只保留
  copy / GEMM
  和后续已 admission 的
  live-form /
  materialization
  bf16 subset；
  direct cast consumer
  和
  `fragment_fill -> cast -> publish`
  在 cleanup 收口时
  不是 hard gate，
  但已经由
  `2026-04-23-blackhole-live-form-materialization-admission.md`
  按 typed projection /
  explicit materialization
  晋级为 admitted
  bf16 direct-runtime subset
- preclear zero-init GEMM
  canonicalize 到
  `clear_accum=true`
  时，
  lowering
  现在会同步删除
  已选中的
  redundant
  `tl.blackhole.fill_fragment`
  zero-fill，
  不再继续落到
  旧 merge/live-form
  路径
- 本轮完成验证：
  - `cd tilelang_repo && cmake --build build -j32`
  - `cd tilelang_repo && pytest -q testing/python/transform/test_blackhole_spatial_ir.py testing/python/target/blackhole/test_blackhole_copy_pipeline.py testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py`
  - `source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh && export TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo && cd /root/dev/vibe_dsl/tilelang_repo && pytest -q testing/python/target/blackhole/test_blackhole_copy_runtime.py`
  - `source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh && export TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo && cd /root/dev/vibe_dsl/tilelang_repo && pytest -q testing/python/target/blackhole/test_blackhole_gemm.py`

本轮 task5
最终确认的职责边界是：

1. 哪些 residue
   必须由前面 task
   先删除
2. task5 自己
   只负责怎样证明
   删除已经完成
3. 哪些验证是 cleanup hard gate，
   哪些仍然只是后续 workload payoff

## 4. 当前代码现实

### 4.1 task5 是最终收敛 gate，不是新的协议删除 task

cleanup 总文档
已经把 task0-task4
分别绑定到各自 residue：

- task0：
  `blackhole.cb_requirements`
- task1：
  `blackhole.compute_regions`
  与 bridge hand-off
- task2：
  public/internal legacy analysis
  与
  `blackhole.lowering_requirements`
- task3：
  `blackhole.copy_semantics`
- task4：
  `blackhole.segment_kind`

因此 task5 的职责
不是再发明一轮
“新的最终协议”，
而是验证：

- 前述删除目标
  已经退出 active chain
- 文档和测试
  已经切到同一口径
- runtime / codegen / build
  没有借 cleanup 机会
  重新长出新的 side channel

### 4.2 当前 task5 文本里的 grep 合同已经过时

现稿里仍把

- `ComputeLoweringFacts`
- `MatchTTMetalComputeLoweringWindows`
- `TryLowerRowwiseFlashAttnRegion`

写成 forbidden residue。

但这些名字在当前 active tree
已经不是 repo HEAD
正在清理的协议面，
而真正仍然 live 的 residue
反而包括：

- `AnalyzeBlackhole*`
- `blackhole.copy_semantics`
- `blackhole.segment_kind`
- `blackhole.lowering_requirements`
- `blackhole.resource_plan`
- `blackhole.work_decomposition`
- `blackhole.compute_regions`
- `blackhole.pipeline_stages`
- `blackhole.cb_requirements`
- `tl.internal_tt_*`

所以 task5 的 scan contract
不能继续用过时 grep 名单；
必须直接继承
cleanup overview、
protocol audit、
task0-task4 completion contract
里实际还在跟踪的 residue 集合。

### 4.3 `progress` / task docs / `memory` 各自职责不同

这几个文件不能再混写：

- `tasks/progress.md`
  负责写
  repo HEAD 总体状态、
  当前 blocker、
  当前下一步
- cleanup task docs
  负责写
  各 residue 的 owner boundary、
  required end-state、
  verification contract，
  以及必要时的
  per-task current-state evidence
- `blackhole_first_principles_protocol_audit.md`
  负责把前述边界
  收成统一的架构审计口径
- `memory/general_dev.md`
  / `memory/bugs.md`
  只记录长期可复用经验、
  bug taxonomy、
  environment/runtime 事实，
  不是阶段状态快照

因此 task5 的文档同步
不是“所有文件都更新一点”；
而是让每份文档
回到自己该做的事。

### 4.4 runtime / codegen / build 的 gate 边界已经是 projection contract

repo HEAD 上，
build / codegen / runtime
当前主要站在：

```text
TTProgram
  -> tl.blackhole_executable
  -> ExecutableSpec
```

这条 projection 边界上。

`MaterializeBlackholeExecutable`
会把
`segment_plan`、
`cb_configs`、
`core_plan`、
`semaphore_plan`、
`buffer_tile_bridge_specs`、
`direct_runtime_unsupported_reasons`
等 payload
投影到 executable attr，
而 target/runtime/codegen
主要消费的也是这层显式记录，
不是回头去读
`blackhole.copy_semantics`
或
`blackhole.segment_kind`
这种 legacy attr。

repo-local
成熟后端先例
也已经站在这条
显式 artifact /
module 边界上：

- `tilelang.jit.kernel.JITKernel`
  在 `tvm_ffi`
  路径上
  明确要求
  `artifact.rt_mod`
  存在，
  否则既不能执行
  也不能
  `export_library`
- `test_blackhole_copy_build_reads_executable_without_legacy_projection_attrs`
  已经直接证明：
  删除 legacy projection attrs 后，
  显式 executable
  仍足以驱动 build
- `test_blackhole_gemm_spec_survives_without_legacy_contract_attrs`
  已经直接证明：
  删除 legacy contract attrs 后，
  显式 executable/spec
  仍是充分 truth

因此 task5 的最终验证
必须保护的是：

- projection contract
  仍然成立
- `tl.tt_program`
  /
  `tl.blackhole_executable`
  /
  `ExecutableSpec`
  /
  `artifact.rt_mod`
  这条显式链
  仍然是唯一
  completion truth
- `BlackholeModule`
  / codegen /
  export 路径
  没有重新变成
  legacy attr reader

而不是把下游 reader
误写成前面 residue
还存在的理由。

### 4.5 runtime correctness gate 必须按 admitted support surface 写

当前 `tasks/progress.md`
已经明确：

- copy direct runtime
  只 gate
  equal-range /
  stride=1
  admitted shape
- GEMM direct runtime
  只 gate
  当前 admitted
  reader/compute/writer contract
  与
  `clear_accum=true`
  主路径
- accessor
  只 gate
  interleaved + DRAM +
  `common_runtime_arg_count = 0`
- communication
  只 gate
  non-oversubscribed explicit semaphore /
  remote-endpoint subset
- `flash-attn`
  compile-path / source/spec baseline
  已稳定，
  但 direct-runtime correctness
  还不是 admitted support surface
- direct cast consumer
  在 cleanup 收口时
  只保留 build/source contract gate；
  之后已按
  live-form /
  materialization
  admission 设计
  晋级当前 supported shape
- `fragment_fill -> cast -> publish`
  在 cleanup 收口时
  也只保留
  build/source contract gate；
  之后已按
  explicit materialization
  admission 设计
  晋级当前 supported shape
- TT-Sim `fp16`
  属于 simulator capability boundary，
  不是当前 correctness gate

因此 task5 的验证矩阵
必须显式区分：

1. compile / projection / codegen hard gate
2. admitted copy/GEMM direct-runtime hard gate
3. 非 admitted workload
   或 simulator capability boundary

其中
`test_blackhole_flash_attention_runtime.py`
  现在本身
  就把
  `direct_runtime_unsupported_reasons`
  当成 queryable gate：
  若 metadata
  声明当前 kernel
  不在 admitted surface，
  测试会显式 `skip`，
  而不是把它升级成
  cleanup failure。

不能把第 3 类
再误升格成 cleanup 完成条件。

### 4.6 `git commit` / `git push` 是交付动作，不是协议完成定义

repo 工作流要求
最终完成时
必须有
`git commit` / `git push`。

但 task5 文档
不应把这一步
写成架构完成判据，
否则就会把：

- residue 是否真的删干净
- projection contract 是否稳定
- 验证是否真的覆盖 admitted surface

这些问题，
偷换成
“有没有推上去”。

正确写法应该是：

- **架构完成**
  先由 residue /
  文档 /
  验证矩阵
  定义
- **repo 交付**
  再在其后完成

## 5. 修正后的任务内容

### 5.1 task5 只负责收敛，不负责重新定义前面 task

task5 必须显式继承
task0-task4
已经写定的 residue contracts。

它不能：

- 本地豁免
  `blackhole.lowering_requirements`
  或 `blackhole.resource_plan`
- 把
  `buffer_tile_bridge_specs`
  / `direct_runtime_unsupported_reasons`
  这类显式 payload
  反向解释成
  还需要保留 legacy attr
- 把当前仍未删除的 residue
  偷换成
  “task5 统一兜底处理”

### 5.2 文档同步必须围绕显式 IR 链，而不是 pass 名字

最终同步后的文档口径
必须统一到：

```text
Normalized Tile TIR
  -> SpatialPlan
  -> TTProgram
  -> ExecutableSpec
```

只允许把 pass/file 名字
作为代码索引，
不允许再把：

- `AnalyzeBlackholeComputeRegions`
- `BuildTTProgram`
- `MaterializeBlackholeExecutable`

写成长期协议层名。

### 5.3 `memory/` 只记录稳定经验，不记录阶段快照

task5 可以更新
`memory/general_dev.md`
和 `memory/bugs.md`，
但只限于：

- pass boundary /
  explicit-IR discipline
  这类长期经验
- runtime gate /
  simulator fatal taxonomy /
  environment 约束
  这类长期可复用事实
- “先替换 reader，
   再删 emitter”
  这类稳定 cleanup 经验

不允许写入：

- “当前 task3 还没做完”
- “今天剩余 blocker 是什么”
- “下一个 checkpoint 是什么”

这些都应留在
`tasks/progress.md`
或对应 task doc。

### 5.4 residue scans 必须对准当前真实删除目标

task5 应运行的源码扫描
必须覆盖当前真实 residue，
而不是历史名字。

最低限度要覆盖：

```bash
rg -n "AnalyzeBlackhole|blackhole\\.copy_semantics|blackhole\\.segment_kind|blackhole\\.lowering_requirements|blackhole\\.resource_plan|blackhole\\.work_decomposition|blackhole\\.compute_regions|blackhole\\.pipeline_stages|blackhole\\.cb_requirements|tl\\.internal_tt_" tilelang_repo/src tilelang_repo/tilelang tilelang_repo/testing/python/transform tilelang_repo/testing/python/target/blackhole
rg -n "blackhole_(reduce_row|mul_row_bcast|mul_grouped_row_bcast|div_row_bcast|div_grouped_row_bcast|exp2_row_bcast_affine|exp2_grouped_row_bcast_affine|scalar_max|scalar_exp2_affine|copy_tile_from_cb)" tilelang_repo/src tilelang_repo/testing/python/target/blackhole
```

预期不是
“绝对零命中”，
而是：

- active chain
  中不再有 live consumer /
  live emitter
- 旧 analysis tests
  也不能继续
  直接断言
  public
  `AnalyzeBlackhole*`
  与
  legacy attrs
- 剩余命中
  只能来自：
  删除断言测试、
  历史记录、
  或明确标记为 archive 的文档

### 5.5 验证矩阵必须拆成 compile gate 和 admitted runtime gate

task5 的 required verification
至少应包含：

```bash
cmake --build build -j32
pytest -q testing/python/transform/test_blackhole_spatial_ir.py
pytest -q testing/python/target/blackhole/test_blackhole_copy_pipeline.py
pytest -q testing/python/target/blackhole/test_blackhole_copy_runtime.py
pytest -q testing/python/target/blackhole/test_blackhole_gemm.py
pytest -q testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py
pytest -q testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py
```

这里的 gate 语义必须写清楚：

- `test_blackhole_spatial_ir.py`
  /
  `test_blackhole_copy_pipeline.py`
  /
  `test_blackhole_flash_attention_pipeline.py`
  /
  `test_blackhole_tvm_ffi_export.py`
  共同覆盖
  compile / source /
  projection / codegen /
  export baseline
- `test_blackhole_copy_runtime.py`
  和
  `test_blackhole_gemm.py`
  覆盖当前 admitted
  copy / GEMM direct-runtime surface
- `test_blackhole_flash_attention_runtime.py`
  如果保留在回归面，
  只用于证明
  unsupported kernel
  会通过
  `direct_runtime_unsupported_reasons`
  自描述并自我降级；
  它不是当前 cleanup hard gate
- TT-Sim `fp16`
  失败仍按 simulator capability boundary
  处理，
  不能把它重写成 cleanup blocker

### 5.6 交付收尾必须放在架构完成之后

当且仅当：

- residue scans
  已经符合预期
- required verification
  已经通过
- 文档 / audit / memory
  已经同步
- 没有残留后台命令

之后，
再执行：

- `git status -sb`
- `git add`
- `git commit`
- `git push`

这一步是 repo workflow 收尾，
不是 task5
用来定义协议完成的核心内容。

## 6. 执行切片

1. 同步
   `tasks/progress.md`
   / `tasks/dev_design/README.md`
   / cleanup docs /
   protocol audit
2. 只在出现稳定经验或稳定 bug taxonomy
   时更新 `memory/`
3. 运行真实 residue scans，
   逐项核对 active-chain hits
4. 运行 build /
   compile-path /
   admitted runtime
   verification
5. 确认没有残留后台进程
6. 最后做
   `git commit` / `git push`

## 7. 相关文件

- [2026-04-16-blackhole-final-legacy-protocol-cleanup.md](/root/dev/vibe_dsl/tasks/dev_design/2026-04-16-blackhole-final-legacy-protocol-cleanup.md)
- [2026-04-16-blackhole-final-legacy-protocol-cleanup-task0.md](/root/dev/vibe_dsl/tasks/dev_design/2026-04-16-blackhole-final-legacy-protocol-cleanup-task0.md)
- [2026-04-16-blackhole-final-legacy-protocol-cleanup-task1.md](/root/dev/vibe_dsl/tasks/dev_design/2026-04-16-blackhole-final-legacy-protocol-cleanup-task1.md)
- [2026-04-16-blackhole-final-legacy-protocol-cleanup-task2.md](/root/dev/vibe_dsl/tasks/dev_design/2026-04-16-blackhole-final-legacy-protocol-cleanup-task2.md)
- [2026-04-16-blackhole-final-legacy-protocol-cleanup-task3.md](/root/dev/vibe_dsl/tasks/dev_design/2026-04-16-blackhole-final-legacy-protocol-cleanup-task3.md)
- [2026-04-16-blackhole-final-legacy-protocol-cleanup-task4.md](/root/dev/vibe_dsl/tasks/dev_design/2026-04-16-blackhole-final-legacy-protocol-cleanup-task4.md)
- [blackhole_first_principles_protocol_audit.md](/root/dev/vibe_dsl/tasks/dev_design/blackhole_first_principles_protocol_audit.md)
- [progress.md](/root/dev/vibe_dsl/tasks/progress.md)
- [README.md](/root/dev/vibe_dsl/tasks/dev_design/README.md)
- [lower.py](/root/dev/vibe_dsl/tilelang_repo/tilelang/engine/lower.py)
- [__init__.py](/root/dev/vibe_dsl/tilelang_repo/tilelang/transform/__init__.py)
- [blackhole_lowering_requirements.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/common/blackhole_lowering_requirements.cc)
- [blackhole_device_resource_canonicalization.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/blackhole_device_resource_canonicalization.cc)
- [lower_blackhole_ops.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.cc)
- [split_blackhole_kernel.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/split_blackhole_kernel.cc)
- [rt_mod_blackhole.cc](/root/dev/vibe_dsl/tilelang_repo/src/target/rt_mod_blackhole.cc)
- [test_blackhole_copy_runtime.py](/root/dev/vibe_dsl/tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py)
- [test_blackhole_gemm.py](/root/dev/vibe_dsl/tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py)
- [test_blackhole_flash_attention_pipeline.py](/root/dev/vibe_dsl/tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py)
- [test_blackhole_tvm_ffi_export.py](/root/dev/vibe_dsl/tilelang_repo/testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py)

## 8. 验证要求

- residue scans
  必须覆盖
  task0-task4
  当前真实删除目标，
  不能继续使用过时 grep 名单
- helper / composite builtin
  residue
  必须继续保持为零
- full build
  必须通过
- compile-path /
  projection-path /
  `tvm_ffi` export
  baseline
  必须通过
- admitted copy / GEMM
  和当前已 admission 的
  live-form /
  materialization
  bf16 subset
  direct-runtime baseline
  必须通过
- `flash-attn`
  direct-runtime correctness
  不是 cleanup hard gate
- TT-Sim `fp16`
  不是 cleanup correctness gate
- 声明完成前
  必须确认没有残留后台构建或测试命令

## 9. 完成判据

### 9.1 架构完成

只有当以下条件同时满足，
task5 才算在架构上完成：

- task0-task4
  定义的 forbidden residue
  已经退出 active chain
- 文档口径已经统一到
  `Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec`
- repo-local
  成熟交付边界
  仍然是
  `tl.tt_program`
  /
  `tl.blackhole_executable`
  /
  `ExecutableSpec`
  /
  `artifact.rt_mod`
  这条显式链，
  不再依赖
  legacy attr
- runtime / codegen / build
  仍然只站在显式 projection contract 上
- `memory/`
  只记录长期经验，
  不承担阶段状态职责
- required verification
  已经通过
- 没有残留后台命令

### 9.2 交付完成

在满足上面的架构完成之后，
repo workflow
还要求：

- `git status`
  结果合理
- 完成
  `git commit`
- 完成
  `git push`

但这是交付收尾，
不是 task5
用来重新定义协议完成的核心合同。
