# Task 0: Lock Exact TT-Metal Builtin Surface And Add Dedicated Builtin Selection

## 1. 任务目标

这个 task 只做一件事：
把 compute-side builtin selection 收成一条正常的 IR rewrite 主链。

完成后应满足三条硬约束：

1. `builtin_blackhole.*` 只保留和 TT-Metal API 一一对应的 exact builtin；
2. `SelectBlackholeTTMetalBuiltins` 自己完成 primitive idiom 的 match + rewrite；
3. exact builtin legality 由 selector 和 validator 共享，而不是散落在旧 planner residue、黑名单和旁路字段里。

## 2. 范围

这个 task 允许做的事：

- 收正 exact builtin surface
- 把 compute idiom 到 exact builtin sequence 的改写职责收进 dedicated selector pass
- 把 selector / validator 之间需要共享的 legality 收成一处
- 删除 task0 范围内还在 active chain 上的 helper/composite builtin residue

这个 task 不允许做的事：

- 再引入新的语义层
- 把 helper/composite builtin 换个名字继续保留
- 继续用 broad lowering bag 或 pass-to-pass seed 托着 builtin 选择走
- 用 workload-specific matcher 充当 builtin selection 主路径

额外边界固定为：

- task0 的主输出仍是当前 `Normalized Tile TIR`
  上的 selected exact-builtin compute IR，
  不是新的 companion / payload / helper layer
- `TTProgram`
  只能消费这层已经合法化的 exact builtin selection 结果，
  不能反过来替这层补 legality

## 3. 当前状态 (`2026-04-17`)

当前 **不算完成**。

repo HEAD 里已经落地的局部结果只有：

- `SelectBlackholeTTMetalBuiltins` 已经接入 active chain；
- helper/composite builtin residue 在 active chain 上已 fail-closed 拒绝；
- covered executable projection 主链里，
  顶层 `compute_epilogue_ops` key
  基本不再作为 owner truth 产出；
- selector 产物会打上 `tl.blackhole_tt_metal_builtin_selection`。

但 task0 要求的主合同还没有成立：

- `SelectBlackholeTTMetalBuiltins` 现在仍只是对
  `PlanTTKernelABI::SelectComputeBuiltins()` 的薄包装；
- 真正的 compute builtin sequence synthesis
  仍在 `lower_blackhole_ops.cc / PlanTTKernelABI`；
- selector 仍依赖
  `BuildLoweringRequirementsFromAnalysis()`、
  `blackhole.cb_requirements`、
  `tl.blackhole_lowering_requirements_seed`；
- `ValidateTTProgram` 还没有和 selector 共用一份 exact builtin legality contract；
- `compute_epilogue_ops` 仍有 compatibility residue 留在 runtime / projection 侧代码里。

所以当前只能写成：

> selector-forwarding slice 已落地

不能写成：

> `Task 0` 已完成

## 4. 当前代码现实

### 4.1 selector pass 还不是独立 rewrite

当前 [select_blackhole_tt_metal_builtins.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/select_blackhole_tt_metal_builtins.cc#L23)
做的事情本质上只有：

- 维护 helper/composite builtin 黑名单
- 调 `PlanTTKernelABI::SelectComputeBuiltins()`
- 检查结果里没有 helper/composite residue
- 打选择完成标记

这只是把旧 planner 的一段逻辑前移到链上，
还不是一个真正独立的 selector pass。

### 4.2 真正的选择逻辑仍在旧 planner

当前仍由 [lower_blackhole_ops.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.cc#L1918)
里的 `PlanTTKernelABI`
负责大部分 compute idiom 选择和 sequence emission。

直接还能看到的逻辑包括：

- `MatchDirectRowReduction`
- `MatchGroupedRowReduction`
- `GenerateRowReductionSequence`
- `GenerateRowBroadcastSequence`
- `GenerateExp2RowBroadcastAffineSequence`
- `GenerateScalarMaxSequence`
- `PublishLocalBufferToExactTiledCB`

这说明“谁决定 exact sequence 长什么样”
这个职责还没有从旧 planner 里迁出去。

### 4.3 selector 仍然吃旧 residue

当前 [PlanTTKernelABI::SelectComputeBuiltins](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.cc#L1918)
还会：

- 调 `BuildLoweringRequirementsFromAnalysis()`
- 回写 `blackhole.cb_requirements`
- 把 bridge/materialization 相关数据重新塞进
  `tl.blackhole_lowering_requirements_seed`

这和 task0 的目标冲突。
task0 的 builtin selection 应该直接面向当前 IR 和显式表示层，
不该继续靠旧 bag / seed / forwarding residue 跨 pass 传递语义。

### 4.4 validator 现在还只是 fail-closed

当前 [validate_tt_program.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/validate_tt_program.cc#L342)
只会：

- 拒绝 helper/composite builtin residue
- 检查 `TTProgram` 结构

它现在仍是 `TTProgram` 结构 / payload validator，
不是 selected exact-builtin TIR legality 的 owner truth。

## 5. 基于审计结果修正后的任务内容

### 5.1 收正 task0 的 owner truth

task0 完成后，
owner truth 固定是：

- 当前 `Normalized Tile TIR`
  上的 selected exact-builtin compute IR

不是：

- `TTProgram`
- payload side channel
- helper/composite builtin surface

这层完成后必须成立的合同是：

- active compute path 上只允许 exact TT-Metal builtin
- exact builtin sequence legality
  在当前 IR 层已经成立并可验证
- 下游 consumer 只能消费这层已经合法化的结果，
  不能在 `TTProgram` / runtime / codegen
  再补一次“真正是什么意思”

### 5.2 收正 exact builtin surface

task0 完成时，
`builtin_blackhole.{h,cc}`
里只能保留 exact TT-Metal builtin。

下面这类东西不能再属于 active builtin surface：

- `tl.blackhole.reduce_row`
- `tl.blackhole.mul_row_bcast`
- `tl.blackhole.mul_grouped_row_bcast`
- `tl.blackhole.div_row_bcast`
- `tl.blackhole.div_grouped_row_bcast`
- `tl.blackhole.exp2_row_bcast_affine`
- `tl.blackhole.exp2_grouped_row_bcast_affine`
- `tl.blackhole.scalar_max`
- `tl.blackhole.scalar_exp2_affine`
- `tl.blackhole.copy_tile_from_cb`
- 各种 `_local` pseudo builtin

如果某个语义需要多步 TT-Metal 调用，
那就由 selector 直接改写成 exact sequence，
而不是把组合动作封成 fake builtin。

### 5.3 收正 builtin 分类 owner truth

exact / helper / alias builtin 的分类
必须收成单一协议定义。

这份分类定义负责：

- selector reject
- builtin surface audit
- codegen acceptance cleanup
- runtime acceptance cleanup
- negative tests

不允许继续维持：

- selector 一份黑名单
- builtin registry 一份暴露面
- codegen / runtime
  再各自写一份字符串判断

### 5.4 selector 自己完成 match + rewrite

`SelectBlackholeTTMetalBuiltins`
必须自己完成：

- 遍历当前 IR
- 识别 primitive idiom
- 直接发出 exact builtin sequence

不能继续维持现在这种形态：

- pass 只是 wrapper
- 真正 rewrite 逻辑藏在 `PlanTTKernelABI`

### 5.5 切掉 selection 对旧 forwarding residue 的依赖

task0 改完之后，
selector 不应再把下列东西作为 builtin selection 的长期依赖：

- `BuildLoweringRequirementsFromAnalysis()`
- `blackhole.cb_requirements`
- `tl.blackhole_lowering_requirements_seed`

如果 selector 真缺某类长期存在的区分，
那说明对应信息应该落在显式表示层，
而不是继续靠旧 residue 流动。

### 5.6 收正 exact legality owner

exact builtin legality
属于当前 selected exact-builtin TIR 合同，
不属于 `TTProgram` validator。

这份 legality 至少覆盖：

- init / uninit pairing
- DST ownership / tile-reg lifecycle
- operand residency
- broadcast / reduce signature legality
- data-format / reconfig 前置条件
- CB protocol ordering

task0 需要建立的是：

- selector 产物的 postcondition
- 在第一个 downstream consumer
  读取 selector 产物之前，
  这份 legality 已经 fail-closed 成立

`ValidateTTProgram`
只能复验那些已经显式进入 `TTProgram`
的结果；
它不能替代当前 IR 层 legality 的建立。

### 5.7 shared legality 的共享方式

selector 和 validator
必须共享同一份 exact legality 定义，
但它们消费的对象不同：

- selected exact-builtin TIR legality
  由当前 IR 层建立并验证
- `TTProgram` validator
  只消费已经显式进入 `TTProgram`
  的 realization 结果

重点不是做一个很重的“框架”，
而是不能再维持：

- selector 一套隐式规则
- `PlanTTKernelABI`
  一套历史规则
- `TTProgram` validator
  只靠黑名单拒绝 residue

### 5.8 明确 `compute_epilogue_ops` 的正确口径

task0 文档里必须把两个层面拆开写：

- covered executable projection 主链里，
  顶层 `compute_epilogue_ops` key
  已基本不再作为 owner truth 产出
- 但 runtime compatibility metadata
  里的 nested `compute_contract.epilogue_ops`
  仍属于未删除 residue

所以 task0 只能写：

- 顶层 projection key 已基本退出 covered 主链

不能写：

- 这个协议面已经从整个仓库彻底删除

### 5.9 收正 `blackhole.cb_requirements` 的删除目标

`blackhole.cb_requirements`
不应继续作为跨阶段 owner truth。

task0 收口后的默认目标是：

- selector 不再把它当成 builtin selection 依赖
- transport / CB planning
  如仍需要相同信息，
  默认从当前表示做局部 derived analysis 重算
- 真正需要跨阶段保留的结果，
  只允许进入显式表示层对象，
  例如最终 `TTProgram` 里的 `cb_plans`

如果后续发现其中某一部分
无法从当前表示稳定重算，
结论应当是补显式表示，
不是继续保留整个 `blackhole.cb_requirements`
作为 bag 式协议面。

## 6. 执行切片

1. 把 primitive idiom 的 match / rewrite
   从 `PlanTTKernelABI`
   迁到 `select_blackhole_tt_metal_builtins.cc`
2. 把 selector 产物的 legality
   收成当前 selected exact-builtin TIR 的显式合同，
   并在第一个 downstream consumer 之前 fail-closed 验证
3. 把 exact / helper / alias builtin 分类
   收成单一协议定义，
   同步收正 selector / builtin surface / codegen / runtime cleanup
4. 把 selector 对旧 lowering bag / seed / CB requirement forwarding 的依赖切掉；
   `blackhole.cb_requirements`
   不再作为 selection owner truth，
   后续 transport / CB planning 默认改成局部 derived analysis
5. 删除 task0 范围内仍在 active chain 上的 helper/composite/local pseudo builtin residue
6. 把 `compute_epilogue_ops`
   的 runtime / projection residue
   明确记录成 compatibility cleanup，
   不再误写成“已经删完”
7. 重新校正文档和测试，
   保证“局部切入”不再被写成“任务完成”

## 7. 相关文件

- `tilelang_repo/src/transform/select_blackhole_tt_metal_builtins.cc`
- `tilelang_repo/src/transform/lower_blackhole_ops.h`
- `tilelang_repo/src/transform/lower_blackhole_ops.cc`
- `tilelang_repo/src/transform/validate_tt_program.cc`
- `tilelang_repo/src/transform/build_tt_program.cc`
- `tilelang_repo/src/tir/builtin_blackhole.h`
- `tilelang_repo/src/tir/builtin_blackhole.cc`
- `tilelang_repo/src/target/tt_program_projection.h`
- `tilelang_repo/src/target/blackhole_module.h`
- `tilelang_repo/src/target/rt_mod_blackhole.cc`
- `tilelang_repo/testing/python/transform/test_blackhole_spatial_ir.py`
- `tilelang_repo/testing/python/target/blackhole/test_blackhole_flash_attention_pipeline.py`
- `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`

## 8. 验证要求

task0 的验证要证明“主链已经换了”，
不是只证明某些旧字符串没有再出现。

至少要覆盖：

1. selector IR 测试：
   primitive idiom 被直接改写成 exact builtin sequence
2. negative test：
   helper/composite/local pseudo builtin 不再允许出现在 active IR
3. selected TIR legality test：
   非法 exact builtin sequence
   会在第一个 downstream consumer 之前被 fail-closed 拒绝
4. payload/spec test：
   顶层 `compute_epilogue_ops` key
   不再出现在 covered executable projection 主链产物里，
   同时 runtime compatibility residue 仍按未删除记录
5. chain test：
   `PlanTTCompute`
   依赖的是已经合法化的 selector 产物本身，
   不是旧 planner 内部旁路

## 9. 完成判据

只有下面这些同时成立，
task0 才算完成：

- exact builtin surface 已收干净
- exact / helper / alias builtin 分类
  已收成单一协议定义
- selector 已成为独立 rewrite pass
- selected exact-builtin TIR legality
  已在第一个 downstream consumer 之前显式成立
- selection 不再依赖旧 lowering residue
- `blackhole.cb_requirements`
  不再作为 selection owner truth 跨阶段流动
- task0 文档已把
  `compute_epilogue_ops`
  的顶层 projection 收缩
  与 runtime compatibility residue
  区分清楚
- task0 文档和代码口径一致，
  不再把局部前移写成任务完成
