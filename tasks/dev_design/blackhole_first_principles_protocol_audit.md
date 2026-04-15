# Blackhole First-Principles Protocol Audit

> 本文档不是新的总体设计。
>
> 它只做一件事：
> **严格从 `final_blackhole_backend_redesign.md` 第 2 节“第一性原理”出发，
> 对当前代码里的协议面做真语义 / 需收紧载体 / fake 过渡协议分类。**
>
> 若本文与总纲冲突，以
> `final_blackhole_backend_redesign.md`
> 第 2 节为准。

## 1. 唯一判定标准

对 spatial target，一个算子最终只会落成三类事实：

1. **访存**
   - 从哪里读
   - 搬到哪里
   - 怎么写回
   - 是否跨 core / multicast / gather / remote write
2. **计算**
   - 在 tile 上执行什么 compute builtin
3. **通信**
   - 哪些 core 之间交换数据
   - 走哪条 NoC / multicast / remote path
   - 谁等谁、何时可见
   - 哪些 barrier / semaphore / global semaphore / completion relation 生效
   - 哪些 topology / packet / launch-order truth 需要冻结

因此，当前代码里的任何
`attr / schema / payload / helper object / pass output`
都只能按下面这条规则判定：

- **如果它不能直接回答访存、计算、通信中的至少一类问题，
  它就不是长期 owner truth。**

这条规则比
“它有没有帮助某个 pass 工作”
“它是不是结构化 map”
“它是不是已经有测试”
都更高。

## 2. 三类分类

### 2.1 真语义载体：应该长期保留

这类东西直接承载三类真语义本身，不是辅助恢复标签。

#### A. `Normalized Tile TIR`

文件和边界：

- `Normalized Tile TIR`
  自身持有的
  `BufferLoad / BufferStore`
- address expr
- region/subscript
- tile-op 参数

原因：

- 这些对象直接回答
  “从哪里读 / 写到哪里 / tile 上算什么”
- 这是语义 body，
  不是 side contract

结论：

- 只要这些信息还能在 TIR 中稳定表达，
  就不允许复制到 companion 或 attr bag

#### B. `tl.blackhole.*` target builtins

文件：

- `tilelang_repo/src/tir/builtin_blackhole.h`
- `tilelang_repo/src/tir/builtin_blackhole.cc`

属于真语义的原因：

- `cb_*`、`read_*`、`write_*`
  直接是访存 primitive
- `mm_init`、`matmul_tiles`、`pack_tile`、
  `fill_fragment`、`reduce_row`、
  `scalar_*`
  直接是计算 primitive
- `noc_async_*`、`get_semaphore`、
  `semaphore_*`
  直接是通信 primitive

结论：

- builtin family 本身是 target 语义面，
  不是 fake 协议
- 后续收口应该围绕
  “谁负责把 TIR 映射到这些 builtins”
  展开，
  而不是再造新的中间标签层

#### C. `TTProgram companion` 中真正承载 target realization 的 typed object

文件：

- `tilelang_repo/src/transform/common/tt_target_program.h`
- `tilelang_repo/src/transform/common/tt_target_program.cc`

当前直接属于真语义 owner 的对象：

- `TTKernel`
- `TTTransportPlan`
- `TTSemaphorePlan`
- `TTABIPlan`
- `TTExecutionPlan`
- `TTProgram`

原因：

- 它们回答的正是：
  - kernel realization
  - transport / routing / delivery
  - semaphore / completion / ordering
  - accessor / runtime-arg / ABI
  - execution / launch order

注意：

- 文档总纲中的理想 owner 名是
  `TTKernelPlan / TTSyncPlan / TTABIPlan / TTExecutionPlan`
- 当前代码里的具体类名和理想名并不完全一致，
  但只要对象边界仍然在回答三类真语义，
  它们就属于应该保留和继续收口的主链对象

#### D. `ExecutableSpec` / executable projection 作为 leaf materialization

文件：

- `tilelang_repo/src/target/tt_program_projection.h`
- `tilelang_repo/src/transform/materialize_blackhole_executable.cc`

原因：

- 它负责冻结已经存在的 target truth
- 它本身不是第二真源，
  但它承载的是 leaf materialization，
  不是 fake 恢复协议

结论：

- 可以存在
- 但只能投影和冻结真语义，
  不能反过来变成 planning source

### 2.2 语义域是真，但当前表达必须继续收紧

这类东西回答的问题属于三类真语义，
但当前表达方式仍然太 bag / 太 raw / 太桥接。

#### A. `TTProgram.payload` / executable projection raw payload

文件：

- `tilelang_repo/src/target/tt_program_projection.h`
- `tilelang_repo/src/transform/validate_tt_program.cc`
- `tilelang_repo/src/target/blackhole_module.h`
- `tilelang_repo/src/target/rt_mod_blackhole.cc`

典型字段：

- `segment_plan`
- `cb_configs`
- `core_plan`
- `semaphore_plan`
- `gemm_contract`
- `compute_contract`
- `multi_gemm_contracts`
- `multi_compute_contracts`
- `compute_epilogue_ops`
- `per_work_arg_specs`

为什么说它们不是 fake：

- 这些字段描述的仍然是
  transport / accessor / compute / communication /
  execution / ABI
  这三类真语义域

为什么说它们必须继续收紧：

- 当前很多地方还是
  `Map<String, Any>` +
  string key +
  leaf consumer 自己解码
- 这意味着 owner truth 还没有完全压回 typed object

结论：

- 这类 surface 的正确方向是
  **typed owner object -> leaf projection**
- 不是继续把 raw payload 扩成更大的公共 schema bag

#### B. `blackhole_runtime_arg_schema`

文件：

- `tilelang_repo/src/transform/common/blackhole_runtime_arg_schema.h`

为什么不算 fake：

- runtime arg / per-work arg
  是访存与 execution 边界上的真语义

为什么仍需收紧：

- 现在很多 identity / kind / value_kind
  还是 string-coded schema

结论：

- 可以保留
- 但它应该服务于
  `TTABIPlan`
  的稳定 owner，
  不是给别的 fake attr 当通用补洞协议

#### C. `blackhole_module` / runtime side 的 leaf decode contract

文件：

- `tilelang_repo/src/target/blackhole_module.h`
- `tilelang_repo/src/target/blackhole_module.cc`

为什么属于本类：

- runtime 最终确实需要拿到
  `cb_configs / core_plan / semaphore_plan /
   compute_contract / runtime args`
- 这些东西本身是 target realization truth

为什么仍需收紧：

- runtime decode 目前仍然绑定了过多 raw key、
  legacy compatibility field、
  以及部分从过渡 attr 兜底恢复的路径

结论：

- 它应该只消费
  `TTProgram / ExecutableSpec`
  的 leaf projection
- 不应该继续接受 fake 协议上浮

### 2.3 fake 过渡协议：必须视为待清算对象

这类东西也许“有帮助”，
也许“暂时让某条链跑起来”，
但它们不直接回答三类真语义。

所以它们不配成为长期 owner truth。

#### A. 标签式 / 恢复式 attr

1. `blackhole.copy_semantics`
   - 文件：
     `tilelang_repo/src/transform/annotate_blackhole_copy_semantics.cc`
   - 问题：
     它不是访存 truth，
     而是给后段一个恢复 copy 意图的标签
   - 判定：
     fake

2. `blackhole.segment_kind`
   - 文件：
     `tilelang_repo/src/transform/split_blackhole_kernel.cc`
   - 问题：
     `reader / compute / writer`
     只是阶段标签，
     不是 transport / compute / communication truth
   - 判定：
     fake

3. `blackhole.resource_plan`
   - 文件：
     `tilelang_repo/src/transform/blackhole_device_resource_canonicalization.cc`
   - 问题：
     它是把 scope/class/role
     事后再编码成 attr，
     本质是 side contract
   - 判定：
     fake

4. `blackhole.cb_requirements`
   - 文件：
     `tilelang_repo/src/transform/lower_blackhole_ops.cc`
     `tilelang_repo/src/transform/plan_blackhole_cb.cc`
   - 问题：
     它不是 TT transport owner truth，
     而是旧 helper 链之间传 requirement bag
   - 判定：
     fake

#### B. 分析期的过渡事实袋

1. `blackhole.work_decomposition`
   - 文件：
     `tilelang_repo/src/transform/analyze_blackhole_work_decomposition.cc`
   - 内容：
     `axes / derived_index_exprs / work_dependent_loop_bounds`
   - 问题：
     它不是 execution owner，
     只是对 launch/work 结构的辅助描述
   - 判定：
     fake

2. `blackhole.compute_regions`
   - 文件：
     `tilelang_repo/src/transform/analyze_blackhole_compute_regions.cc`
   - 内容：
     `ops / pointwise_ops / reductions / broadcasts /
      selection_targets / selection_pairs / update_sources /
      loop_carried_state / recurrence_edges / buffer_tile_bridge_specs`
   - 问题：
     这是一整袋恢复式 side contract，
     不是 compute / transport / communication owner
   - 判定：
     fake

3. `blackhole.pipeline_stages`
   - 文件：
     `tilelang_repo/src/transform/analyze_blackhole_pipeline_stages.cc`
   - 内容：
     `loop_var / num_stages / stage_local_buffers`
   - 问题：
     这是 helper analysis 结果，
     不是 execution owner truth
   - 判定：
     fake

4. `blackhole.buffer_effect_use_role_facts`
   - 文件：
     `tilelang_repo/src/transform/common/blackhole_buffer_effect_use_role_analysis.h`
     `tilelang_repo/src/transform/common/blackhole_buffer_effect_use_role_analysis.cc`
   - 内容：
     `defs / uses / use_role`
   - 问题：
     这类 facts 最多只能是 analysis 内部结果；
     一旦上升成公开 PrimFunc attr 协议，
     就又成了新的 side contract
   - 判定：
     fake

5. `defs / uses / use_role / recurrence_edges`
   - 问题：
     不论 schema 多“干净”，
     只要它们以公开 attr bag 方式存在，
     就不是三类真语义 owner
   - 判定：
     fake

#### C. 混合大杂烩 attr

1. `blackhole.lowering_requirements`
   - 文件：
     `tilelang_repo/src/transform/common/blackhole_lowering_requirements.cc`
     `tilelang_repo/src/transform/lower_blackhole_ops.cc`
   - 内容来源：
     work decomposition、
     phase facts、
     pipeline stage facts、
     compute support、
     materialization contract、
     flow contract
   - 问题：
     它把本来应该分属不同 owner /
     不同层级 /
     不同语义面的东西全揉在一起
   - 判定：
     **当前 fake surface 中最核心的黑洞**

2. `companion_base.h` 巨型 registry
   - 文件：
     `tilelang_repo/src/transform/common/companion_base.h`
   - 问题：
     大量
     `manifest_key / schema_key / buffer_* / selection_* / recurrence_*`
     并不直接对应三类真语义 owner
   - 结论：
     `tl.spatial_plan / tl.tt_program / tl.blackhole_executable`
     这类顶层 attr 名可以保留；
     但巨型 side-contract registry
     本身按第一性原理属于 fake 扩张面

3. `tl.internal_tt_*`
   - 问题：
     这类 attr 是中间桥接物，
     不是最终 target owner truth
   - 判定：
     fake

#### D. 名字表 / 位置表 / helper patch surface

这类东西的问题不是“有没有用”，
而是它们不该承担主协议职责。

1. `PlanTTCBAlloc::GetCBArgPositions`
   - 文件：
     `tilelang_repo/src/transform/plan_blackhole_cb.cc`
   - 问题：
     builtin-name 到 cb arg slot 的大表，
     是 late helper patch，
     不是 target owner truth

2. `PlanTTCoreGroups`
   - 文件：
     `tilelang_repo/src/transform/assign_blackhole_cores.h`
     `tilelang_repo/src/transform/assign_blackhole_cores.cc`
   - 问题：
     `11x10` 常量、
     `blockIdx.x/by` 名字恢复、
     default grid
     都是 heuristic helper
   - 判定：
     不应上升为稳定 execution owner 协议

3. `lower_blackhole_ops.cc` 中 legacy helper attr
   - 例子：
     `tl_cb_in0 / tl_cb_in1 / tl_cb_out / tl_k_tiles`
   - 判定：
     fake legacy residue

## 3. 按文件面的归类

### 3.1 主链真语义面

- `src/tir/builtin_blackhole.*`
- `src/transform/common/tt_target_program.*`
- `src/transform/common/blackhole_runtime_arg_schema.h`
- `src/target/tt_program_projection.h`
- `src/target/blackhole_module.*`

### 3.2 可以存在，但只能继续收紧的载体

- `src/target/tt_program_projection.h`
- `src/transform/validate_tt_program.cc`
- `src/target/blackhole_module.*`
- `src/target/rt_mod_blackhole.cc`
  中直接 decode
  `TTProgram / ExecutableSpec`
  leaf projection 的部分

### 3.3 fake 过渡协议生产者

- `src/transform/annotate_blackhole_copy_semantics.cc`
- `src/transform/split_blackhole_kernel.cc`
- `src/transform/analyze_blackhole_work_decomposition.cc`
- `src/transform/analyze_blackhole_compute_regions.cc`
- `src/transform/analyze_blackhole_pipeline_stages.cc`
- `src/transform/common/blackhole_buffer_effect_use_role_analysis.*`
- `src/transform/common/blackhole_lowering_requirements.cc`
- `src/transform/blackhole_device_resource_canonicalization.cc`
- `src/transform/lower_blackhole_ops.cc`
- `src/transform/plan_blackhole_cb.cc`
- `src/transform/assign_blackhole_cores.*`
- `src/transform/common/companion_base.h`

### 3.4 fake 过渡协议消费者

- `src/target/codegen_blackhole.cc`
- `src/target/rt_mod_blackhole.cc`
- `tilelang/engine/phase.py`
- `tilelang/transform/__init__.py`
- `testing/python/target/blackhole/*`
  中锁定
  `blackhole.lowering_requirements /
   blackhole.compute_regions /
   blackhole.resource_plan /
   blackhole.buffer_effect_use_role_facts`
  的测试

## 4. 直接后果

这份分类一旦成立，后续实现就必须服从下面三条纪律：

1. **不能再新增 fake attr bag**
   - 不能再用
     `补一个 blackhole.* attr`
     来接住缺失 truth

2. **analysis facts 不能再公开协议化**
   - `defs / uses / use_role / recurrence_edges`
     即使作为 analysis 内部结果存在，
     也不能再上升成
     runtime / build / codegen
     稳定输入

3. **后段必须退出 semantic recovery**
   - runtime / codegen / executable extraction
     不能继续从
     `segment_kind / lowering_requirements /
      compute_regions / resource_plan`
     这类 fake surface 恢复语义

## 5. 对当前 R0-R2 的约束

从本审计出发，
`R0.1-R0.3`
不允许走下面这条路：

- 先发明新的
  `buffer_effect_*` attr bag
- 再让
  `lowering_requirements`
  或 build/codegen/runtime
  读取它

因为这只是用新的 fake 协议替换旧的 fake 协议。

`R0.1-R0.3`
唯一允许的方向是：

1. analysis 内部产出 typed facts
2. planner 消费这些 facts
3. 最终只把结果收回
   `TTProgram companion`
   和它的 leaf projection

也就是说，
**R0-R2 的目标不是把 fake facts 写得更干净，
而是让 fake facts 失去公开协议地位。**
