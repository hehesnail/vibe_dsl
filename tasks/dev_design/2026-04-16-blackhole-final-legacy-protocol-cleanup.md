# Blackhole Final Legacy Protocol Cleanup Boundary

> 已完成 cleanup 的边界索引；默认和 task0-task5 分文件一起阅读。

> 本文只定义 cleanup 的 task ownership、forced debt、唯一 exception、
> 依赖顺序和 final convergence gate。
> 它不是第二份总体设计文档，
> 也不单独声明 repo HEAD 的总体 blocker / 当前下一步；
> 这些总体状态统一只看 `tasks/progress.md`。
> task0-task5 分文件如果保留各自 residue 的
> per-task current-state evidence，
> 也不能替代 `progress.md`
> 作为总体状态来源。
> cleanup 只是和
> `Task 1 / Task 2 / Task 3 / Legacy Protocol Deletion`
> 重叠的 residue workstream，
> 不是主设计路线图。

## 1. 文档定位

这条 cleanup 主线
只做一件事：

> 把当前 Blackhole active chain 上
> 仍然存活的 legacy protocol /
> analysis bag /
> marker attr /
> helper residue
> 收回到显式表示层，
> 不再用新的 side channel
> 替代旧 side channel。

长期边界始终只有：

```text
Normalized Tile TIR
  -> SpatialPlan
  -> TTProgram
  -> ExecutableSpec
```

这意味着：

- task0-task5
  只是 cleanup 执行切片，
  不是新的 IR 层
- pass / file / helper 名字
  只能作为代码索引，
  不是长期协议边界
- 当前 cleanup 完成时，
  编译链的最终交付 truth
  必须继续收在
  `tl.tt_program`
  /
  `tl.blackhole_executable`
  /
  `ExecutableSpec`
  /
  `artifact.rt_mod`
  这条显式链上

## 2. Compiler Design Baseline

cleanup 必须服从
当前唯一总体设计文档
`final_blackhole_backend_redesign.md`
里的 IR-first 纪律。

固定基线如下：

1. 当前 IR / current object
   必须被直接 rewrite 成同层或下一层显式表示；
   不能在多个阶段之间反复靠 bag / payload /
   helper wrapper 传真语义。
2. 只有显式表示层
   能承载跨阶段语义：
   - `Normalized Tile TIR`
   - `SpatialPlan`
   - `TTProgram`
   - `ExecutableSpec`
3. analysis 只能是
   derived / temporary /
   invalidatable /
   recomputable mechanics，
   不能升级成协议面。
4. pass-local matcher / visitor / helper
   只允许留在同一实现文件内部，
   直接支撑当前 pass 的 rewrite /
   builder；
   它们不能长成新的 shared layer。
5. 不允许新增：
   - 新 attr bag
   - 新 public analysis wrapper
   - 新 helper bridge layer
   - 新的 stringly-typed
     `kind` / `role` / `direction`
     协议词表
   - workload-specific lowering path
6. runtime / materialization truth
   也必须是显式对象：
   - TileLang 侧
     站在
     `TTProgram -> ExecutableSpec -> artifact.rt_mod`
   - TT-Metal 侧
     站在显式
     `Program` /
     `MeshWorkload` /
     kernel /
     circular-buffer /
     semaphore /
     runtime-arg /
     launch API
   它们都不是 helper attr
   或 compiler marker

repo-local
成熟 GPU passes
已经在用同样纪律：

- [lower_hopper_intrin.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_hopper_intrin.cc)
- [lower_ldg_stg.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_ldg_stg.cc)
- [wgmma_sync_rewriter.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/wgmma_sync_rewriter.cc)
- [annotate_warp_group_reg_alloc.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/annotate_warp_group_reg_alloc.cc)

Blackhole cleanup
也必须按同样模式收口：
current IR /
current object /
direct rewrite /
explicit builder，
而不是 helper-side semantic recovery。

## 3. Task Ownership Split

### 3.1 Task 0

task0 的 owner truth
是当前 `Normalized Tile TIR`
上的 exact TT-Metal builtin selection
和 exact legality，
不是新的 helper / payload layer。

它固定负责：

- 把 compute-side exact builtin selection
  收成当前 IR 的 checked postcondition
- 删除 helper / composite /
  pseudo builtin surface
- 删除
  `blackhole.cb_requirements`
  作为 selection owner truth

它固定**不**负责：

- 把 `CB / runtime arg / semaphore / launch`
  倒灌成 selector contract
- 把 `compute_epilogue_ops`
  或 leaf/runtime compatibility residue
  合法化成 task0 边界

### 3.2 Task 1

task1 只切
bridge handoff 的来源和 owner truth。

它固定负责：

- 把
  `blackhole.compute_regions`
  broad bag
  从 producer-side owner truth
  上切掉
- 在 current stage
  直接 capture
  logical buffer/tile handoff

它固定**不**负责：

- 当场删完全部 downstream bridge consumer
- 把 leaf/codegen compatibility path
  重新写成合法中间层

### 3.3 Task 2

task2 真正要删的是
“analysis 作为跨阶段语义 carrier”
这件事。

它固定负责：

- 删除 public
  `AnalyzeBlackhole*`
  wrappers
- 删除 internal evidence helper /
  broad lowering bag
  作为 active consumer truth
- 不允许把
  `blackhole_lowering_requirements.cc`
  留成新的内部总包

它固定**不**负责：

- 回退
  `BuildTTProgram`
  已经接近正确的 staged aggregator 角色
- 把 runtime / codegen /
  build reader
  重新拉回 legacy analysis bag

### 3.4 Task 3

task3 的 required end-state
是
`blackhole.copy_semantics`
owner-truth cutover，
不是复制旧 annotation schema。

它固定负责：

- 让 compiler-side consumers
  各自在所在阶段
  直接从 current IR / dataflow
  恢复 copy 含义
- 删除
  `blackhole.copy_semantics`
  作为 shared semantic carrier
- 维持 target / codegen / build / runtime
  继续站在
  `TTProgram -> ExecutableSpec`
  projection boundary 上

它固定**不**负责：

- 新造一个 exported copy contract /
  helper bag
- 把
  `compute_contract`
  /
  `multi_compute_contracts`
  /
  `gemm_contract`
  fallback
  这类 leaf/runtime compatibility debt
  合法化成 copy owner truth

### 3.5 Task 4

task4 的 required end-state
是
`blackhole.segment_kind`
owner-truth cutover，
不是继续维持 shared marker。

唯一合法的跨阶段 kind truth
只有：

- `TTKernel.kind`
- `TTKernelPlan.kind`
- projected executable
  `segment_plan[*].kind`

task4 固定负责：

- 先把 planner-side
  `segment_plan_`
  改成 direct construction
- 再单独删除
  leaf-local body slicing residue

它固定**不**负责：

- 把
  `BuildTTProgram`
  /
  projection /
  codegen
  重新拉回 attr boundary
- 把
  `SegmentBodyExtractor`
  这类 leaf-local residue
  合法化成 target/runtime contract

### 3.6 Task 5

task5 不是新的删除 owner。

它只负责：

- 同步
  `progress` /
  cleanup docs /
  protocol audit /
  `memory/`
  的最终口径
- 按 task0-task4
  的 completion contract
  跑 residue scans
- 执行最终 verification matrix
- 在架构完成之后
  再执行 repo workflow
  的 `git commit` / `git push`

它固定**不**负责：

- 本地豁免前序 task
  仍未删除的 residue
- 把 unsupported workload /
  simulator capability boundary
  误写成 cleanup hard gate
- 把 `git commit` / `git push`
  写成协议完成定义

## 4. 唯一 Narrow Cleanup Exception

`tl.blackhole_logical_buffer_tile_bridge_specs`
是当前 cleanup
唯一允许存在的窄 temporary handoff。

它的正确口径固定为：

- 当前允许的窄 exception
- leaf-local handoff，
  不是 planning representation
- 不是
  `SpatialPlan`
  / `TTProgram`
  / `ExecutableSpec`
  的长期表示
- 不是 TT-Metal program /
  runtime contract
- 不是新的 medium-term bridge layer

这条 exception
当前存在的唯一理由，
是 optimized/helper entry
仍需要一段 leaf-local handoff。

对应 forced debt
也必须写死：

- task1 只切 owner truth
  到 direct capture
- downstream
  `buffer_tile_bridge_specs`
  payload / projection / codegen path
  仍只允许被写成
  wrong-now leaf compatibility debt
- 这条 debt
  由 task3
  继续删除；
  它不能被 overview
  合法化成新层

## 5. Ordering Note

repo HEAD 当前 cleanup 顺序和状态
统一只看 `tasks/progress.md`。

本文不再重复维护
当前任务队列；
这里只保留当前完成边界：

- cleanup `task0-task5`
  的 broad protocol convergence
  已完成
- broad legacy analysis /
  wrapper /
  bag /
  cross-pass attr
  不再是当前 active chain
  的 owner truth
- `tl.blackhole_logical_buffer_tile_bridge_specs`
  仍是唯一窄 bridge exception，
  只能按 forced leaf debt
  继续收敛
- `compute_contract`
  /
  `gemm_contract`
  /
  `multi_*_contracts`
  仍在
  `TTProgram.payload -> ExecutableSpec -> runtime`
  fallback 链里，
  只属于 task3 /
  leaf contract-family debt
- 后续顺序固定回到
  `tasks/progress.md`
  中的 support-surface lane：
  先补
  `SpatialPlan`
  logical live-value /
  materialization-boundary，
  再 typed 化 leaf contract-family
  并删除 runtime fallback

## 6. Primary File Surfaces

下面这些文件名
只作为当前代码索引，
不作为架构边界。

- task0：
  [builtin_blackhole.h](/root/dev/vibe_dsl/tilelang_repo/src/tir/builtin_blackhole.h)、
  [builtin_blackhole.cc](/root/dev/vibe_dsl/tilelang_repo/src/tir/builtin_blackhole.cc)、
  [select_blackhole_tt_metal_builtins.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/select_blackhole_tt_metal_builtins.cc)、
  [validate_tt_program.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/validate_tt_program.cc)
- task1/task2：
  [lower.py](/root/dev/vibe_dsl/tilelang_repo/tilelang/engine/lower.py)、
  [phase.py](/root/dev/vibe_dsl/tilelang_repo/tilelang/engine/phase.py)、
  [__init__.py](/root/dev/vibe_dsl/tilelang_repo/tilelang/transform/__init__.py)、
  [blackhole_lowering_requirements.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/common/blackhole_lowering_requirements.cc)、
  [lower_blackhole_ops.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.cc)
- task3/task4：
  [split_blackhole_kernel.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/split_blackhole_kernel.cc)、
  [blackhole_device_resource_canonicalization.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/blackhole_device_resource_canonicalization.cc)、
  [materialize_blackhole_executable.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/materialize_blackhole_executable.cc)、
  [tt_program_projection.h](/root/dev/vibe_dsl/tilelang_repo/src/target/tt_program_projection.h)、
  [codegen_blackhole.cc](/root/dev/vibe_dsl/tilelang_repo/src/target/codegen_blackhole.cc)、
  [rt_mod_blackhole.cc](/root/dev/vibe_dsl/tilelang_repo/src/target/rt_mod_blackhole.cc)
- task5：
  [test_blackhole_spatial_ir.py](/root/dev/vibe_dsl/tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py)、
  [test_blackhole_copy_pipeline.py](/root/dev/vibe_dsl/tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_pipeline.py)、
  [test_blackhole_copy_runtime.py](/root/dev/vibe_dsl/tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py)、
  [test_blackhole_gemm.py](/root/dev/vibe_dsl/tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py)、
  [test_blackhole_flash_attention_pipeline.py](/root/dev/vibe_dsl/tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py)、
  [test_blackhole_tvm_ffi_export.py](/root/dev/vibe_dsl/tilelang_repo/testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py)、
  [tasks/progress.md](/root/dev/vibe_dsl/tasks/progress.md)、
  [blackhole_first_principles_protocol_audit.md](/root/dev/vibe_dsl/tasks/dev_design/blackhole_first_principles_protocol_audit.md)

## 7. Split Docs

- [Task 0: Lock exact TT-Metal builtin surface and dedicated builtin selection](./2026-04-16-blackhole-final-legacy-protocol-cleanup-task0.md)
- [Task 1: Replace compute-region owner truth with direct logical bridge capture](./2026-04-16-blackhole-final-legacy-protocol-cleanup-task1.md)
- [Task 2: Remove public and internal legacy analysis carriers](./2026-04-16-blackhole-final-legacy-protocol-cleanup-task2.md)
- [Task 3: Replace `blackhole.copy_semantics` with direct IR/dataflow recovery](./2026-04-16-blackhole-final-legacy-protocol-cleanup-task3.md)
- [Task 4: Replace `blackhole.segment_kind` with explicit kernel-kind truth](./2026-04-16-blackhole-final-legacy-protocol-cleanup-task4.md)
- [Task 5: Final convergence, documentation sync, and verification](./2026-04-16-blackhole-final-legacy-protocol-cleanup-task5.md)

## 8. Final Convergence Gate

cleanup 的最终收敛
必须按 task5 口径判断，
不能再沿用旧的平面 residue 列表。

### 8.1 Residue Scan Contract

源码扫描
必须对准当前真实 live residue set，
而不是过时 helper 名字。

当前固定 scan 面
至少包括：

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

预期不是
“绝对零命中”，
而是：

- active chain
  中没有 live consumer /
  live emitter
- 旧 analysis tests
  不再直接断言
  public
  `AnalyzeBlackhole*`
  与 legacy attrs
- 剩余命中
  只能来自：
  删除断言测试、
  历史记录、
  或 archive 文档

### 8.2 Verification Matrix

最终 hard gate
必须拆成三层：

1. **compile / projection / codegen / export hard gate**
   - full build
   - source / compile / projection baseline
   - `tvm_ffi` export baseline
2. **admitted runtime hard gate**
   - admitted copy direct-runtime
   - admitted GEMM direct-runtime
3. **non-gates**
   - unsupported workload
   - `flash-attn` runtime self-demotion
   - TT-Sim `fp16` simulator boundary

因此 overview
固定承认下面这些事实：

- `test_blackhole_spatial_ir.py`
  /
  `test_blackhole_copy_pipeline.py`
  /
  `test_blackhole_flash_attention_pipeline.py`
  /
  `test_blackhole_tvm_ffi_export.py`
  共同覆盖
  compile / projection /
  codegen / export baseline
- `test_blackhole_copy_runtime.py`
  和
  `test_blackhole_gemm.py`
  覆盖当前 admitted
  copy / GEMM runtime gate
- `test_blackhole_flash_attention_runtime.py`
  不是 cleanup hard gate；
  它只用于证明
  unsupported kernel
  会通过
  `direct_runtime_unsupported_reasons`
  自描述并自我降级
- TT-Sim `fp16`
  不是 cleanup correctness gate

### 8.3 Architecture Completion Vs Delivery Completion

只有当以下条件同时满足，
cleanup 才算在架构上完成：

- task0-task4
  定义的 forbidden residue
  已退出 active chain
- overview / task docs / audit /
  `progress` / `memory`
  已同步到同一口径
- runtime / codegen / build
  继续只站在显式
  projection / artifact truth
  上
- 没有残留后台构建 /
  测试 /
  长命令
- required verification
  已通过

在这之后，
repo workflow
才继续要求：

- `git status`
- `git add`
- `git commit`
- `git push`

这一步只是交付收尾，
不是 cleanup
用来定义协议完成的核心合同。
