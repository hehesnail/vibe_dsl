# Task 0: Lock Exact TT-Metal Builtin Surface And Add Dedicated Builtin Selection

## 1. 任务目标

这个 task 只做一件事：
把 compute-side builtin selection 收成一条正常的 IR rewrite 主链。

完成后应满足三条硬约束：

1. active lowered IR 只保留和 TT-Metal API 一一对应的 exact builtin，
   `builtin_blackhole.{h,cc}` 里的 helper-named alias accessor
   不能再继续定义 public builtin surface；
2. `SelectBlackholeTTMetalBuiltins` 自己完成 primitive idiom 的 match + rewrite；
3. exact builtin legality 的规则定义 / 常量 / predicate
   由 selector 和下游 validator 共享，
   并且当前 IR legality
   必须在进入 `TTProgram`
   之前独立成立，
   而不是散落在旧 planner residue、
   黑名单和旁路字段里。

## 2. 范围

这个 task 允许做的事：

- 收正 exact builtin surface
- 把 compute idiom 到 exact builtin sequence 的改写职责收进 dedicated selector pass
- 把 selector / validator 之间需要共享的 legality
  收成单一 rule definition / constants / predicates
- 删除 task0 范围内还在 active chain 上的 helper/composite builtin residue

这个 task 不允许做的事：

- 再引入新的语义层
- 把 helper/composite builtin 换个名字继续保留
- 继续用 broad lowering bag 或 pass-to-pass seed 托着 builtin 选择走
- 用 workload-specific matcher 充当 builtin selection 主路径
- 把 `ValidateTTProgram`
  包装成 selected exact-builtin TIR
  legality owner
- 把 codegen / runtime
  的 compatibility reader
  写成 builtin legality owner

额外边界固定为：

- task0 的主输出仍是当前 `Normalized Tile TIR`
  上的 selected exact-builtin compute IR，
  不是新的 companion / payload / helper layer
- 这里的 selected exact-builtin compute IR
  只是当前 `Normalized Tile TIR`
  的一个 checked postcondition /
  admitted subset，
  不是新的中间表示层
- `TTProgram`
  只能消费这层已经合法化的 exact builtin selection 结果，
  不能反过来替这层补 legality
- `CB / runtime arg / semaphore / launch`
  这类 TT-Metal
  program-construction 事实
  仍属于 `TTProgram`
  及其后续 leaf projection /
  runtime binding 边界，
  不能因为 repo HEAD
  里 selector 还吃 seed
  就倒灌回 task0 owner truth

## 3. 当前状态 (`2026-04-20`)

当前 **不算完成**。

repo HEAD 里已经落地的局部结果只有：

- `SelectBlackholeTTMetalBuiltins`
  已经作为 front-door selector wrapper
  接入 active chain；
- helper/composite builtin residue
  在 selected-TIR active chain 上
  已 fail-closed 拒绝；
- covered executable projection 主链里，
  顶层 `compute_epilogue_ops` key
  已不再由当前 executable projection writer
  作为 owner truth 产出；
- selector 产物会打上 `tl.blackhole_tt_metal_builtin_selection`。
- `PlanTTCompute`
  也已经要求这个选择完成标记
  才进入 downstream planning；
  但这只说明 consumer ordering
  已经前移，
  不等于 selected exact-builtin TIR
  legality gate
  已经独立建成。

但 task0 要求的主合同还没有成立：

- `SelectBlackholeTTMetalBuiltins` 现在仍只是对
  `PlanTTKernelABI::SelectComputeBuiltins()` 的薄包装；
- 真正的 compute builtin sequence synthesis
  仍在 `lower_blackhole_ops.cc / PlanTTKernelABI`；
- selector 仍依赖
  `BuildLoweringRequirementsFromAnalysis()`、
  `blackhole.cb_requirements`、
  `tl.blackhole_lowering_requirements_seed`；
- `PlanTTCBAlloc`
  仍显式要求
  `blackhole.cb_requirements`，
  所以下游 planner contract
  还没有从这个 carrier 上切走；
- `ValidateTTProgram` 还没有和 selector 共用一份 exact builtin legality contract；
- 顶层 `compute_epilogue_ops` key
  虽已退出当前 writer 的 covered owner path
  `TTProgram.payload / executable projection`
  主链，
  但 nested `compute_contract.epilogue_ops`
  仍在 runtime compatibility metadata 里存活；
- direct runtime
  仍会在缺失显式 `compute_contract`
  时回退到 legacy `gemm_contract`；
  这属于 task3 负责删除的
  leaf compatibility debt，
  不是 task0 的合法边界；
- `builtin_blackhole.{h,cc}`
  和 `codegen_blackhole.cc`
  里仍有 helper-named alias accessor /
  alias dispatch residue。

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

它现在仍是
`TTProgram` 结构 / payload validator，
不是 selected exact-builtin TIR legality
的 owner truth，
也还没有和 selector
共享一份显式 exact legality rule definition。

当前链上实际还是：

- selector 输出
- `PlanTTCompute` 直接读取这份结果
- 更后面才到 `ValidateTTProgram`

所以 repo HEAD 里
还不存在一个
“在第一个 downstream consumer 之前
建立 selected exact-builtin TIR legality”
的独立 gate。

### 4.5 exact surface 仍有 public alias surface / dual-dispatch residue

当前 active lowered IR
已经基本切到 exact op spellings，
helper/composite 名称
不再被当作 active IR owner truth。

但 builtin / leaf 侧的 public surface
还没有完全收干净。

repo HEAD 里还能直接看到：

- `builtin_blackhole.{h,cc}`
  仍通过 helper-named alias accessor
  暴露
  `write_local_slice_to_cb /
   write_local_fragment_tile_to_cb /
   write_local_fragment_slice_to_tiled_cb /
   cast_fragment_slice_to_tiled_cb /
   read_cb_front_tile_to_local /
   read_cb_front_tile_to_local_fragment`
- `codegen_blackhole.cc`
  仍接受
  helper-named spellings
  和 exact op name
  并行的 dual dispatch 入口

所以当前只能写成：

- active IR surface
  已基本切到 exact op spellings

不能写成：

- builtin registry / leaf codegen
  已经只剩单一 public exact surface

### 4.6 `blackhole.cb_requirements` 仍是 live planner contract

当前不只是 selector
会写回 `blackhole.cb_requirements`；
下游 `PlanTTCBAlloc`
也还把它当成 formal input。

这意味着 repo HEAD 里：

- `blackhole.cb_requirements`
  还不是“已经退场、只差清理文档”的残留
- 它仍是 active chain
  上真实会被读取的 planner carrier

因此 task0 文档必须把它写成：

- 当前 live contract
  但属于 **wrong-now bag boundary**，
  不属于终态 owner truth

不能把它写成：

- 已经只剩死字段或 leaf-local metadata

## 5. 基于审计结果修正后的任务内容

### 5.1 repo HEAD 当前必须按 debt 处理的边界

task0 后续正文里的 owner / contract / residue
统一按下面口径理解：

- `SelectBlackholeTTMetalBuiltins`
  当前只是 front-door selector wrapper；
  真正的 primitive idiom
  match + rewrite owner
  仍在 `PlanTTKernelABI`
- `blackhole.cb_requirements`
  /
  `tl.blackhole_lowering_requirements_seed`
  当前仍是 live contract，
  但属于 forced implementation debt，
  不是合法 bag boundary
- helper-named alias accessor
  /
  builtin registry
  /
  codegen dual dispatch
  当前都只能按待删 compatibility surface
  描述，
  不能写成合法 public protocol
- `ValidateTTProgram`
  当前仍只是 downstream structural / payload validator，
  不是 selected exact-builtin TIR
  legality owner
- nested `compute_contract.epilogue_ops`
  与
  `ExecutableSpec::compute_epilogue_ops`
  当前仍是 task3 负责删除的
  leaf compatibility debt
- `blackhole_module.cc / rt_mod_blackhole.cc`
  当前仍保留
  `compute_contract -> multi_compute_contracts -> gemm_contract`
  的 direct-runtime compatibility fallback；
  这同样属于 task3 负责删除的
  leaf/runtime debt，
  不是 task0 的 owner truth

这些边界在 repo HEAD 里
即使还被实现依赖，
也不能被文档写成
“暂时合理的中间层”。

### 5.2 收正 task0 的 owner truth

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
- `CB / runtime arg / semaphore / launch`
  这些 TT-Metal
  program-construction 事实
  仍由 `TTProgram`
  的显式 slice
  负责承载，
  不是 task0
  倒推 builtin selection owner truth

### 5.3 收正 exact builtin surface

task0 完成时，
必须同时满足两件事：

1. active lowered IR
   只剩 exact TT-Metal builtin op name
2. `builtin_blackhole.{h,cc}`
   里遗留的 helper-named alias accessor
   要么删除，
   要么降成 pass-local / compatibility-local helper，
   不能继续定义 public builtin surface

这里的“exact op name”
指的是 active IR / public registry / leaf codegen
共同承认的单一 op spelling，
不是“IR 用 exact spelling，
但 registry / codegen
继续平行接受 helper alias”
这种双轨表述。

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

对应地，
leaf codegen
也不能继续长期接受：

- helper-named alias spellings
- “exact op name / helper op name 二选一”
  这种兼容式 dispatch

### 5.4 收正 builtin 分类 owner truth

exact / helper / alias builtin 的分类
必须收成单一协议定义。

这份分类定义负责：

- selector reject
- builtin surface audit
- codegen acceptance cleanup
- negative tests

不允许继续维持：

- selector 一份黑名单
- builtin registry 一份暴露面
- codegen
  再单独写一份平行字符串判断
- runtime / test
  再把 helper alias surface
  写成另外一套 owner truth 词表

### 5.5 selector 自己完成 match + rewrite

`SelectBlackholeTTMetalBuiltins`
在终态必须自己完成：

- 遍历当前 IR
- 识别 primitive idiom
- 直接发出 exact builtin sequence

repo HEAD 当前仍处于：

- pass 只是 wrapper
- 真正 rewrite 逻辑藏在 `PlanTTKernelABI`

所以 task0 文档必须把这件事写成：

- 当前仍未完成的 implementation debt

不能把 selector wrapper
已经接入主链
直接写成
rewrite owner 已完成切换。

### 5.6 切掉 selection 对旧 forwarding residue 的依赖

task0 改完之后，
selector 不应再把下列东西作为 builtin selection 的长期依赖：

- `BuildLoweringRequirementsFromAnalysis()`
- `blackhole.cb_requirements`
- `tl.blackhole_lowering_requirements_seed`

这里要特别写清楚：

- repo HEAD 当前
  `tl.blackhole_lowering_requirements_seed`
  即使只剩
  `buffer_materialization_contracts`
  /
  `buffer_tile_bridge_specs`
  这类窄 seed，
  也不等于 task0
  可以把它合法化成
  current-IR builtin selection
  的长期 owner truth
- 如果 selector
  真还需要其中某类长期事实，
  结论仍然是：
  要么把事实补成显式表示，
  要么把它留成后续 task
  负责的窄 bridge /
  leaf-local contract，
  不是继续维持
  broad bag / seed

如果 selector 真缺某类长期存在的区分，
那说明对应信息应该落在显式表示层，
而不是继续靠旧 residue 流动。

### 5.7 收正 exact legality owner

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

repo HEAD 当前真正已经成立的，
只有：

- selection stamp
- helper/composite residue reject
- coarse compute-pipeline gate
  （例如 `num_stages` /
   `pipeline_stage_counts`
   这类前置限制）

这还不等于
selected exact-builtin TIR
自己的完整 legality owner
已经立起来。

`ValidateTTProgram`
只能复验那些已经显式进入 `TTProgram`
的结果；
它当前仍是 downstream structural / payload validator，
不能替代当前 IR 层 legality 的建立。

### 5.8 shared legality 的共享方式

这里描述的是 required end-state，
不是 repo HEAD 当前已成立事实。

selector 和 validator
必须共享同一份 exact legality
规则定义，
但它们消费的对象不同：

- selected exact-builtin TIR legality
  由当前 IR 层建立并验证
- `TTProgram` validator
  只消费已经显式进入 `TTProgram`
  的 realization 结果

这里“共享”的含义固定是：

- 共享同一组
  enum / constexpr / predicate /
  legality rule definitions
- 保证 selector postcondition
  和 downstream validator
  对同一类 exact builtin
  使用同一套显式规则
- 共享范围
  只限于 exact primitive
  本身的 legality；
  不把 `CB` 配置、
  runtime-arg shape、
  semaphore binding、
  core launch
  这类 program-construction /
  ABI 事实
  混进 task0 legality owner

这里“共享”的含义明确**不是**：

- 共享 bag / seed / payload
- 共享 analysis object / evidence object
- 共享需要跨阶段流动的 helper wrapper
- 再造一个独立于当前 IR /
  `TTProgram`
  之外的“legality framework object”

重点不是做一个很重的“框架”，
而是不能再维持：

- selector 一套隐式规则
- `PlanTTKernelABI`
  一套历史规则
- `TTProgram` validator
  只靠黑名单拒绝 residue

如果某条 legality
不能同时以
当前 IR 上的 selector postcondition
和 `TTProgram` 上的显式结果
来表达，
结论应当是补显式表示或补局部 derived check，
不是新增共享协议对象。

### 5.9 明确 `compute_epilogue_ops` 的正确口径

task0 文档里必须把两个层面拆开写：

- covered executable projection 主链里，
  顶层 `compute_epilogue_ops` key
  已经不再作为 owner truth 产出
- `TTProgram -> executable projection`
  的主 writer
  不再发布这个顶层 key
- 但 runtime compatibility metadata
  里的 nested `compute_contract.epilogue_ops`
  仍属于未删除 residue
- `ExecutableSpec`
  的 in-memory struct /
  leaf runtime facade
  里也仍保留
  `compute_epilogue_ops`
  compatibility field
- runtime reader
  仍会从 nested
  `epilogue_ops`
  恢复 compatibility interpretation

所以 task0 只能写：

- 顶层 projection key
  已退出 covered owner path
- 当前 executable projection writer
  不再发布这个顶层 key
- nested runtime compatibility metadata
  与 in-memory compatibility field
  仍未删除
- 直接 runtime
  仍保留
  `compute_contract -> multi_compute_contracts -> gemm_contract`
  fallback，
  这只是 compatibility reader，
  不是 owner truth

不能写：

- runtime / metadata 侧
  已经完全不再接受 epilogue residue
- direct runtime
  已经只读单一显式 `compute_contract`

### 5.10 收正 `blackhole.cb_requirements` 的删除目标

`blackhole.cb_requirements`
不应继续作为跨阶段 owner truth。

但 repo HEAD 当前必须明确写清楚：

- `PlanTTCBAlloc`
  仍显式依赖它，
  所以它现在仍是 live planner contract
- 这代表的是 forced implementation debt，
  不是合法中间层边界
- “selection 不再依赖它”
  和
  “整个 active chain 已不再依赖它”
  不是一回事

task0 收口后的默认目标是：

- selector 不再把它当成 builtin selection 依赖
- transport / CB planning
  如仍需要相同信息，
  默认从当前表示做局部 derived analysis 重算
- 真正需要跨阶段保留的结果，
  只允许进入显式表示层对象，
  例如最终 `TTProgram` 里的 `cb_plans`
- 同样，
  runtime-arg / accessor /
  semaphore / launch
  这类跨阶段事实
  如确实需要保留，
  也必须进入
  `TTProgram` /
  `ExecutableSpec`
  的显式字段，
  不能继续借
  selection residue
  流动

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
5. 删除 task0 范围内仍在 active chain 上的 helper/composite/local pseudo builtin residue，
   并同时收掉 builtin registry / leaf codegen
   对 helper-named alias 的兼容入口
6. 把 `compute_epilogue_ops`
   的顶层 projection 清理
   与 nested runtime compatibility residue
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
   不再出现在
   `TTProgram.payload / executable projection`
   主链产物里，
   同时 nested `compute_contract.epilogue_ops`
   的 runtime compatibility residue
   仍按未删除记录；
   direct runtime / tests
   即使还保留兼容读取，
   也不能被当成 owner truth
5. chain test：
   `PlanTTCompute`
   依赖的是已经合法化的 selector 产物本身，
   不是旧 planner 内部旁路
6. surface audit test：
   builtin registry / leaf codegen
   不再把 helper-named alias
   当成并行 public surface

## 9. 完成判据

只有下面这些同时成立，
task0 才算完成：

- exact builtin surface 已收干净
- active IR / builtin registry / leaf codegen
  不再同时维持 helper-named alias surface
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
