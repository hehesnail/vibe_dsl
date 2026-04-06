# 通用开发模式与当前有效经验

> 当前 Blackhole 后端唯一设计依据: `tasks/dev_design/final_blackhole_backend_redesign.md`
> 本文档只保留稳定、可复用的工程经验。
> 文中若出现 `flash-attn` / `GQA` / `MoE` 等例子，只作为具体经验来源，不代表总体架构边界。

## 当前文档入口

- 当前活动文档只看 `tasks/dev_design/` 根目录和 `tasks/progress.md`。
- `tasks/dev_design/archive/` 下的内容全部是历史记录，不再作为当前任务安排入口。
- 当前 Stage 4 直接按四份分阶段文档执行，不再额外维护单一总 implementation plan。

## 编译器后端开发

### codegen 职责

- codegen 只负责把已明确的 segment/spec 打印成源码，不要让它同时承担协议推断
- 如果 pass 只把规划结果写到 attrs 而不回写 IR body，后续 pass/codegen 必须从 attrs 恢复状态，形成"两套真源"维护负担。优先让 pass 同时完成 IR 回写
- 当 host TIR 中的 packed call 以表达式形式出现（如 `LetStmt(x, call_packed(...), ...)`），host codegen 必须同时支持：
  - 发出 packed runtime 调用语句
  - 把 `result` 容器重新打印成可用表达式
  否则最终 C 文本会退化成 `x = ;` 这类空表达式错误

### 类型与参数

- compile-time args 和 runtime args 必须严格区分
- compile-time ABI 一旦开始 formalize，就要把“匿名 `vector<uint32_t>` + host 侧猜位置”的约定收正成显式 schema（例如 `compile_time_arg_specs` / `launch_spec`）；兼容字段可以暂留，但不能继续当真源
- runtime 参数布局必须显式、可验证，不能依赖隐式猜测
- 当 TileOp 已有稳定主参数 ABI、但需要继续 formalize 更宽 schema 时，优先把扩展 payload 显式追加到 IR 末尾或扩独立 schema，不要让后段从分散 attrs 或隐式默认值猜 richer contract
- 统一 work descriptor 时，优先把 `start_id` / `num_tiles` 这类角色化字段写进 schema；不要继续让 host/runtime 从 `current_work_linear_id`、`tile_count` 之类的单值默认推导整套工作范围
- 对需要 `blockIdx` 重建的 Blackhole kernel，把 `work_linear_id` 作为独立字段保留；不要让 codegen 从 `a_tile_start_id` / `output_tile_start_id` 之类的 range 字段反推逻辑 work identity
- 64-bit 地址需明确拆分与重组规则

### 存储 scope

- Blackhole `shared` 的主映射是 CB/L1 资源，不是简单的 C 数组声明
- 对 TT-Metal/Blackhole，`local` 更适合被看作中间语义桶，而不是最终资源语义。进入后端后应继续分流成：
  - 真正的小标量临时
  - fragment / accumulator 对象
  - `local/accumulator -> shared(CB)` 这类显式 dataflow transfer
  如果 residual `local` 明显处在 fragment 结果写回 CB 的桥接位置，就不要继续让它以普通 `BufferStore` 漏到 codegen；应尽快收成正式 copy direction / builtin
- 对这种 `local/accumulator -> shared(CB)` 桥接语义，不要在 codegen 层兜二维 shared store。更稳的主链是：在 `LowerBlackholeOps` 明确成正式 `CopyDirection`，lower 成单独 builtin，再让 `codegen_blackhole` 只消费该 builtin。这样 compile-path blocker 会停在真正的 dataflow 边界，而不是晚到 codegen 的 residual store 噪声
- 如果 split 后 device kernel 会再按 segment 生成新的 `PrimFunc` 给 codegen/runtime 消费，就不要假设 reader/runtime ABI 只存在于 `segment_plan` 里。更稳的做法是把 segment-level `runtime_args` / `common_runtime_args` 聚合回 segment function 顶层 attrs，并让 `KernelSpec` 从 segment function 自己提取 ABI；否则 codegen/runtime 很容易只看到 compute 子集，最后报 `Missing runtime arg binding for buffer var: K` 这类假象
- 对正式主链，不要保留 `input0/output0` 这类默认 runtime-arg fallback。copy/dataflow kernel 的 ABI 必须来自 IR/segment schema；一旦 `blackhole.runtime_args` 或 `segment_plan[*].runtime_args` 缺失，就该在 `rt_mod_blackhole` build-time 直接 fail-fast，而不是继续 build 到 codegen 才报 buffer binding 缺失
- 当 target 硬件资源模型与 generic backend 不一致时，优先扩 IR 类型系统（如 StorageRank），不要给后段 pass 打豁免
- 判断"该不该扩 IR"的信号：同一根因在多个 generic pass 上以不同报错出现

### 测试策略

- 只做 codegen/reference compare 的脚本不算 true E2E
- 测试分层：结构层（lowered TIR/attrs）→ planner 层（cb_configs/bindings）→ runtime 层（direct path 真执行）

## TileLang 工程

- `3rdparty/` 和 `build/` 不应进入主仓库提交
- `pip install -e .` 可能重新触发构建并失败，用 `.pth` 指向本地构建产物
- C++ 改动后 pytest 前先确认 `libtilelang.so` 已重编，避免加载旧库假阴性
- 当前 `tilelang_repo/CMakeLists.txt` 通过 `file(GLOB ...)` 收集源码；新增 `.cc` 文件后只跑 `make` 不够，必须先在 `tilelang_repo/build/` 里重新执行一次 `cmake ..`，否则新文件不会进 build graph，Python 侧会表现成“wrapper 已加但 global func 仍然找不到”
- 不要对同一个 `tilelang_repo/build/` 并行跑 `cmake --build` 和 pytest。共享构建目录在链接进行中时，测试可能加载到旧/半更新的 `libtilelang.so`，制造假阴性或顺序相关噪声
- 对新的 split-after analysis pass，优先把结果写成结构化 IR attrs（`Array<Map<...>>`、`PrimExpr` 等），不要先字符串化再让后续测试/consumer 反解析。测试也应直接断言 attr 结构和 `PrimExpr` 语义，而不是只查字符串片段
- 对 layered IR 迁移的 Stage 0 护栏，不要继续把 program registry 或 pre-lift semantic 输入挂在单个 `PrimFunc.attrs` 上。更稳的主链是：
  - module-scope registry 进 `IRModule.global_infos["tl.device_programs"]`
  - pre-lift typed 输入进 `PrimFunc.attrs["tl.semantic_seeds"]`
  - unsafe TIR mutation 统一通过 companion invalidation contract 使 `tl.semantic_program / tl.spatial_program / tl.tt_program` 整体失效
- 对 `tl.device_programs` 的 pre-`SplitHostDevice` 退化场景，也不要假设 module 里一定已经存在
  规划出来的 `*_kernel` member `PrimFunc`。当当前 module 仍只保留 root `PrimFunc` 时，
  `Phase B` 的 registry 聚合和 validator 需要允许 `root_symbol` fallback；
  否则 companion IR 已经生成，module-scope phase truth 仍会被空 registry 误判失败
- 对 Phase A semantic schema，state role / update law / supplement 都必须保持 workload-agnostic。`flash-attn`、`topk`、chunk recurrence 这类 family 只用来验证抽象角色能否恢复，不要把 workload noun 直接升格成 schema
- 对 Phase A typed witness / refinement contract，也不要直接做 workload-shaped witness class。更稳的主链是：
  - `AnalyzeSemanticStructure` 先把开放 analysis attrs 投影成通用 `tl.semantic_witnesses`
  - `LiftStatefulSemanticIR` 再从 witness 投影到 `SemanticProgram`
  - `ValidateSemanticRefinement` 负责核对 witness 与 semantic core 一致性
  - 一旦 unsafe mutation 发生，`InvalidateBlackholeCompanionPrograms` 必须整体清掉
    `tl.semantic_structure / tl.semantic_witnesses / tl.semantic_program / tl.spatial_program / tl.tt_program`，
    而不是只删 `tl.semantic_program`
- 当 witness/core 合同已经稳定后，不要继续把 pass 内部逻辑写成散落的字符串比较。更稳的形态是：
  - `semantic_vocab` 统一维护 closed enum vocabulary
  - `semantic_witness_decoder` 统一做 raw witness -> typed view 的 payload 解析
  - `semantic_refinement_rules` 统一维护 relation/binding/contract legality
  - string 只留在 FFI reflection、attr 存储和报错打印边界
- 进一步收紧 witness payload 时，也不要让 `AnalyzeSemanticStructure`、lift、validator
  各自手拼/手拆 `Map<String, Any>`。更稳的形态是：
  - `semantic_witness_payloads` 统一维护 canonical payload family
  - analysis 通过 payload builder 发射 witness
  - lift / refinement validator 只消费 typed payload decoder
  - 对不需要额外值的 axis，允许 empty payload；不要保留冗余 `"kind": ...` 协议
- 类似地，不要在 `AnalyzeSemanticStructure` 末端用“本 region 里出现过 `if_then_else` / `gemm`”这种全局命中来直接判 `selection_state` / `recurrence`。更稳的做法是把局部计算关系先提升成 typed attr，例如 `selection_targets`、loop-carried update facts，再由 semantic lift 消费
- 对 selection/indexing family，仅仅恢复出 `selection_state` 和 `index_state` 还不够。如果后续语义需要知道“哪个 value state 和哪个 companion/index state 属于同一次 selection”，就应把这层 pairing 作为上游 typed analysis attr 显式导出，例如 `blackhole.fragment_regions[*].selection_pairs = {value_target, companion_target, source_states}`，再由 semantic lift 写进对应 `select` update 的 typed binding（如 `paired_value_state`）
- 对 selection/indexing family 的 arg-reduction target，也不要再靠 integer hint 去判 `index_state`。更稳的做法是让 fragment analysis 显式导出 `blackhole.fragment_regions[*].arg_reduce_targets`，把“这个 reduction target 属于 selection companion/value flow”作为 typed relation 固化下来，再由 semantic lift 恢复角色
- 对 chunk recurrence / carry family，也不要满足于“有 `carry` role 和 `UpdateLaw.kind == recurrence` 就算完成”。如果后续语义需要知道 carried update 的具体 edge，应把它作为上游 typed attr 显式导出，例如 `blackhole.fragment_regions[*].recurrence_edges = {target, source_states}`，再由 semantic lift 写进对应 `recurrence` update 的 typed binding（如 `recurrence_source_state`）
- 对 Phase A semantic recovery，不要让 `LiftStatefulSemanticIR` 长期把 `UpdateLaw.source_states` 默认回填成 `[target_state]`。如果 `select / reduce / recurrence` 的 source-state 关系对后续语义判断有意义，就应先在上游 analysis attrs 中显式导出，例如 `blackhole.fragment_regions[*].update_sources = {target, sources}`，再由 lift 原样消费
- 对会在 lowering 边界被销毁的 Phase A explicit-op evidence，也不要继续等到
  `AnalyzeBlackholeFragmentRegions` 之后再从 lowered loop 里倒推。更稳的主链是：
  `CollectSemanticManifestSeeds -> ProjectSemanticManifest -> AugmentSemanticManifest`。
  其中 early capture 只抓会被 `LowerTileOp` 吃掉的 op（当前 `copy / fill / reduce / cumsum`），
  late augment 只补 device-side residual explicit op（当前 `gemm_py`）；`AnalyzeSemanticStructure`
  只把 manifest 当 evidence / supplement 使用，不把它升级成新的 semantic truth source
- 对 semantic manifest 的 structural evidence 迁移，也不要只搬
  `selection_targets / selection_pairs / arg_reduce_targets / recurrence_edges` 四个名字本身。
  如果想让 `AnalyzeSemanticStructure` 真正做到 manifest-only / manifest-first，
  还必须把这些关系引用到的 `fragment_buffers / update_sources / loop_carried_state`
  一起带上；否则 witness/lift 仍会因为缺失局部 state descriptor 或 source-set 而退回
  `fragment_regions`
- 当 structural evidence 或 cross-pass annotation 里既要保留可读名字、又要保留真实绑定对象时，
  不要再把 display name 当协议主键。更稳的 schema 是：
  - attr entry 同时携带 `name + typed Buffer handle`
  - consumer 一律 handle-first，按 `Buffer.data` identity 恢复 state / segment / accessor / copy 绑定
  - string 只留作日志、调试、兼容回退
- 对 `blackhole.fragment_regions` 的后续收尾，也不要把“删 attr”当成第一步。更稳的顺序是：
  先踢掉 semantic consumer，再踢掉 lowering consumer，最后再删 attr 本身。当前
  `row_reductions` 之所以还留在 `fragment_regions`，不是因为 semantic manifest 不够，而是因为
  这块事实仍然 mixed ownership：semantic recovery 和 `LowerBlackholeOps` 都在消费
- 对 `row_reductions` 这类同时牵涉 semantic truth 和 lowering summary 的事实，不要偷懒让
  `fragment_regions` 继续从 lowered TIR 字符串（例如 `AllReduce<...>` callee）回推关键语义。
  更稳的拆法是：
  - semantic 需要的 `reduce_kind` 由 explicit-op / manifest payload 直接提供
  - `fragment_regions` 只保留 lowering 真需要的 target / handle summary
  - 如果 fragment matcher 还要识别 reduction target，本体依据也应是 `AddNode/MaxNode` 等
    IR 结构，而不是模板名字
- 如果正式 device `main_kernel` 路径缺的正好是 `row_reduction.kind` 这类 semantic-owned 事实，
  修法也必须继续往 `Phase A` 收：
  - 让 fragment analysis / manifest structural evidence 把 `kind` 带齐
  - 让 `AnalyzeSemanticStructure -> LiftStatefulSemanticIR` 恢复统一 semantic truth
  - 不要为了补某个 multi-phase spatial gate，临时让 `Phase B` 直接回读
    `blackhole.fragment_regions`
- 对 `Phase B` 的 `Layout / WorkPartition` scaffolding，也不要继续把
  `blackhole.work_decomposition` 当 primary truth source。更稳的主链是：
  - `AnalyzeSemanticStructure -> LiftStatefulSemanticIR` 先把 domain `axes / traits`
    收进 `SemanticProgram`
  - `LowerToSpatialProgram` 优先从 `SemanticProgram.domain` 构造 spatial scaffolding
  - `blackhole.work_decomposition` 只保留过渡 fallback，直到 `Phase B`/`C` consumer
    全部脱钩
- 对 `Phase B` validator，也不要只校验 object graph “有没有串起来”。
  更稳的最小 legality gate 是：
  - `SpatialLayout / WorkPartition.axes` 必须和 `SemanticProgram.domain.axes` 对齐
  - `layout.kind == indexed` 必须和 semantic domain 的 `derived_indices` trait 对齐
  这样 spatial builder 一旦回退到错误真源，会在 `ValidateSpatialProgram` 立刻 fail-fast
- 对 `Phase B` validator 的下一层 hardening，也不要让 multi-phase contract 只停留在
  “有 phase_boundary_materialization 就算通过”。更稳的 legality contract 是：
  - downstream phase 必须显式引用自己的 channel contract
  - phase 引用的 channel，其 `target_task` 必须属于该 phase
  - `tl.device_programs` 不能只校验 phase 数量；要至少核对
    `ProgramPhase(name/task_names/channel_names)` signature
- 对 `LowerBlackholeOps` 的 lowering requirements，`work_axes / derived_index_expr_count`
  应优先从 `tl.spatial_program` 恢复；`blackhole.work_decomposition` 只保留 compatibility
  fallback。这样 consumer 才不会在 `Phase B` 已 object 化后继续回头吃旧 attr
- 对 `Phase B` 的 GEMM spatial builder，也不要继续把 `blackhole.segment_plan`
  当成 `reader / compute / writer` task graph 的真源。更稳的主链是：
  - `SplitBlackholeKernel` 在 IR body 上留下 `blackhole.segment_kind`
  - `LowerToSpatialProgram` 直接从这些 segment annotation 恢复 task/placement 顺序
  - `segment_plan` 只保留给后续 lowering/runtime compatibility 使用，不再参与
    `SpatialProgram` 构造
- 对 `LowerBlackholeOps` 的 fragment lowering requirements，也不要让
  `blackhole.fragment_regions` 成为唯一输入。更稳的过渡边界是：
  - `fragment_regions` 存在时，把它当 lowering-facing compatibility summary 消费
  - `fragment_regions` 缺失时，从 `SemanticProgram + residual body scan`
    恢复最小 `fragment_op_kinds / pointwise_op_kinds / row_reduction_targets`
  - 这样 `fragment_regions` 可以继续缩到 compatibility path，而不会反向卡死
    `Phase B` consumer cutover
- 对 Blackhole `lower()` 主链，不能在 `SplitBlackholeKernel` / `Analyze*` / `LowerBlackholeOps` 之前就用旧的 device attrs 过滤掉入口 `PrimFunc`。Blackhole entry kernel 在这条链之前通常还没有 `blackhole.*` attrs，因此 `is_device_call()` 必须把 entry `PrimFunc` 视为 device 输入，否则专属 pass 实际上跑在空 `device_mod` 上
- fragment region analysis 里的 `pointwise_chain` 不能通过全局扫描所有 `tir.add/mul/div/max/...` 来判定；那样会把普通索引算术也误记成 fragment compute。更稳的做法是只在 fragment/local region 自身的 store / dataflow 关系里识别 pointwise
- 对 split-after TIR 的 fragment analysis，不要只盯 `CallNode`。像 `scores_sum[0] + acc_s[rv]`、`T.max(scores_max[0], tmp[0])` 这类模式在 TVM IR 里常常是 `AddNode` / `MaxNode` / `MulNode` / `DivNode` 等原生表达式节点；如果只扫 `CallNode`，row reduction 和 scalar-to-vector broadcast 会在真实 MHA/GQA IR 上整片漏掉
- 当 fragment analysis 需要为后续 legality/lowering 提供更细粒度输入时，不要只保留一个粗粒度 `pointwise_chain` 标签。先在 analysis attrs 里把 `fill / exp2 / cast / max / add / mul / div / if_then_else` 这类点算子枚举出来，再让后续 lowering 决定哪些是当前子集、哪些要继续 fail-fast；否则 build-time gate 永远只能对一个黑盒 `pointwise_chain` 报错
- 当 build-time gate 还没真正支持 fragment 子集执行时，也不要继续按 `pointwise_chain` 这种总括词报错。更稳的做法是：`LowerBlackholeOps` 先汇总细粒度 `pointwise_op_kinds`，`rt_mod_blackhole` 再按具体 unsupported 集合（例如 `row_reduction / row_broadcast / fill / mul / ...`）显式 fail-fast。这样后续实现 fragment lowering 时，可以逐项消掉 blocker，而不是每次都改一整块黑盒 gate
- 如果某类 fragment 计算暂时还不能执行，但它本身又不是当前最核心的 blocker，不要让它继续挤占 build-time gate。gate 应优先收窄到真正阻塞当前 consumer 的最小 op family，否则你很难看清下一步该先补哪类 lowering
- 如果同一 backend 既有 `ExecutableSpec` 路径，也有 device-only codegen 路径，fragment-subset fail-fast 不能只放在 `rt_mod_blackhole` 这类 spec 提取层。否则一旦某条路径绕过 spec，错误就会晚到 codegen 内部爆成 `Find undefined Variable ...` 之类的噪声。对这类 shared lowering boundary，要让 codegen 入口和 spec 提取层共享同一套 unsupported-op gate
- 复杂 consumer 的 analysis 测试要按 IR 所在阶段写预期，不要强行要求 split-after IR 和 optimized device IR 暴露完全相同的点算子集合。像 `if_then_else`、`exp2`、predicate init、selection/update 这类语义，在 prepasses 前后本来就可能以不同节点或已折叠形态出现；测试应锁住该层 IR 仍然可见的结构信号，而不是复用源码层的完整 op 清单
- 当 analysis 结果还不能直接 lower 成可执行 kernel 时，不要把 raw analysis attrs 直接塞进 `ExecutableSpec`。更稳的过渡做法是：先在 `LowerBlackholeOps` 里把它们归一化成一层很薄的 IR attrs summary（例如 `blackhole.lowering_requirements`），再由更后面的 build/runtime gate 消费并 fail-fast。这样主链能先完成“analysis 被 lowering 真正接住”，又不会把半成品 descriptor 永久冻结进 runtime schema
- 对 pipelined fragment subset 的早期 legality，优先直接读 loop annotation（例如 `ForNode.annotations["num_stages"]`），不要假设更后面的 region canonicalization 一定先成功。这样即使更宽 GQA/MHA 形态还会在别的 lowering 细节上炸，主链也能先把“这组 stage 配置本来就不支持”显式拦下来
- row-broadcast 的索引归并信号不要只认 `floor_div/floor_mod`。在 split-after TIR 里，同一类“coarsened row index”也可能被写成 `tir.shift_right(i, k)`；如果 analysis 只认除法不认右移，GQA 这类更宽 fragment pipeline 很容易少报 `row_broadcast`
- 对优化后的 device IR 做 analysis 时，不要再靠 `_1/_2/...` 这类 view 后缀归一化 buffer 名来恢复同一逻辑对象。更稳的做法是：
  - 能带 typed `Buffer` handle 的 schema/attr 就直接带 handle
  - 不能直接带 handle 的 matcher 至少按 backing `Buffer.data` / `Var` identity 合并
  - 如果现有 IR 还表达不出这层稳定 identity，就扩 schema，而不是发明新的命名约定
- `T.Pipelined` 经过 device prepasses 后，stage 注解不一定还叫 `num_stages`；Blackhole analysis 至少要同时兼容 `num_stages` 和 `tl_pipelined_num_stages`，否则 optimized path 会比 split-after path 少一层 pipeline 语义
- `tl.region` 不要假设 `BufferLoad` 索引数必须与 extents 个数完全相等。对 staged/shared view，常见形态是“leading stage index + trailing tile extents”；更稳的 bridge 是把未匹配的前导索引收成 singleton axes，再用提供的 extents 重建尾部 region
- 对 fragment/reduction lowering，不要假设 split-after 和 optimized device IR 会保留完全相同的包裹结构。`OptimizeForTarget` 之后，`for extent=1` 常会被抹平成同级 `SeqStmt`，而 `pragma_unroll_explicit` 之类则会额外包成 `AttrStmt`；matcher 应先剥掉这类无语义包装，再匹配真正的 reduction 形态，否则手动 pass 链能 lower、full `lower()` 反而会漏掉同一逻辑
- 对复杂 fragment compute，不要把整类 `row_broadcast` / `select-update` / `fused pointwise` 一起当成一个 blocker 或一次性全开。更稳的推进方式是先在 `LowerBlackholeOps` 里吃掉最小、形态稳定、且和目标后端现有 compute primitive 对得上的子集，再把剩余融合路径继续单独收敛。这样 gate 会随着真实 lowering 一步步收窄，而不是永远把整类语义黑盒化
- 对 scalar fragment 链，也应优先抽成通用 scalar primitive，而不是继续把它混在更大的 vector-broadcast / fused-update blocker 里。这样剩余 blocker 会更准确地聚焦到真正还缺的复合路径
- 做 TIR matcher 时，不要用 `same_as` 去判断两个“语义上相等的 extent/stride 常量”是否相等。优化前后 IR 很容易生成不同节点实例的 `IntImm(4)` / `IntImm(32)`，这会让像 `i * 4 + vec` 这种明显线性化的模式白白 miss。更稳的写法是直接对整个 affine 关系做 `Analyzer::Simplify` 后判零，例如比较 `expr - (outer * inner_extent + inner)` 是否可化简为 0
- 对 fragment pointwise 的 residual 剪枝，不要在整棵表达式树里无差别找 `AddNode` / `MulNode`。像 `Cast(acc_s[i * 4 + vec])` 这种合法 residual `cast`，它的索引表达式天然会带 `AddNode`；如果 helper 扫全树，就会把索引算术误判成尚未 lower 的 pointwise `add`。更稳的口径是先看 residual store 的**根表达式类型**，只在根值本身仍是 `Add/Max/Cast/...` 时才把对应 op 继续保留为 blocker
- 对 plain-local fragment state 和临时 reduction scratch 的区分，也不要再靠 `_clear`、`tmp` 之类命名暗示。更稳的依据是：
  - `layout_map` / storage-role 是否把该 buffer 声明成 persistent fragment state
  - store / reduce / finalize 的 def-use 结构是否说明它只是短生命周期 scratch
  - 一旦结构上已经能证明是 temp scratch，就应尽早从 fragment-state 集合里剔除，避免后续 semantic / lowering 把它当真状态
- 改 C++/链接 `libtilelang.so` 之后，不要再把 `cmake --build` 和 `pytest` 并行跑。pytest 很容易在新 `.so` 链接完成前启动，看到旧实现，进而把问题误判成“修复没生效”。这类验证应该顺序执行：先 build，确认链接完成，再跑 Python 测试
- 对 `SemanticProgram` 的 internal state/effect graph，不要默认“每个 update 都必须产出 version/def”。像
  copy pipeline 这类 target-less `map` update，本身没有 semantic state，应当留在 semantic core 的
  update 集合里，但不要强行进入 state/effect graph；否则会制造 orphan version，并把 validator
  假设抬得比 schema 还强
- `loop_carried` graph fact 也不要只靠 `carry role` 或显式 `carried_from` relation 来生成。对
  `UpdateLaw.kind == recurrence` 的 ordered update，同样要产出 `StateJoin(loop_carried)`；否则
  synthetic topk / selection recurrence gate 会在 refinement 中被误判为缺失 carried effect

## TT-Metal / TT-Sim 环境

### 构建

- 系统依赖、clang 版本、RPATH、LD_LIBRARY_PATH 都影响编译和运行
- TileLang direct 模式需对齐 C++20、TT_METAL_HOME、tt_stl、hostdevcommon、umd
- 优先消费 TT-Metal local install tree（`find_package(tt-metalium CONFIG REQUIRED)`），不要把 `.cpmcache` 整片加进 include path

### TT-Sim

- `TT_METAL_SLOW_DISPATCH_MODE=1` 对 TT-Sim 很关键
- `scripts/setup_tt_sim.sh` 必须在执行测试的同一 shell 里 source
- 如果当前在 git worktree 中工作，不要 source worktree 里的 `scripts/setup_tt_sim.sh` 副本；应 source 顶层 checkout 的 `/root/dev/vibe_dsl/scripts/setup_tt_sim.sh`，再把 `TILELANG_HOME` 指回当前 worktree 的 `tilelang_repo`
- 关键变量：`TT_METAL_RUNTIME_ROOT`、`TT_METAL_SIMULATOR`、`TT_METAL_SLOW_DISPATCH_MODE`、`LD_LIBRARY_PATH`
- direct path kernel 临时目录必须每次执行唯一化，避免 JIT 缓存串扰

### TT-Metal API

- 稳定 host-side 抽象：Program、CreateCircularBuffer、CreateKernel/CreateKernelFromString、SetRuntimeArgs
- CB 数据路径需要地址共享（get_write_ptr/get_read_ptr）**和** 同步原语（cb_reserve_back/cb_push_back/cb_wait_front/cb_pop_front）两者缺一不可
- 对 tile/dataflow target 至少分层：host tensor layout、device buffer/accessor、CB transport、compute ABI、work distribution
- 当 direct runtime 目前只支持某个 dtype 组合时，也要先把 tensor dtype / CB transport dtype / accumulator dtype 明确写进 schema，再由 runtime 显式校验“不支持的组合”；不要继续把 bring-up 时的硬编码假设当协议
- 对 TT-Metal GEMM，host layout conversion 是 correctness contract，不是后续优化：
  - host 侧必须明确谁负责 `transpose_B`
  - 上传前必须把 row-major tensor 转成设备期望的 tiled layout
  - 读回后必须做 untilize
  - copy E2E 通过不能证明 matmul contract 正确，因为 copy 只验证字节保持，不验证 tile 语义
- richer schema 先于更大支持面：如果 schema 已经能表达更多 range/stride 组合，但 direct runtime/codegen 还没正式支持，必须 `ICHECK` fail-fast，不能静默退回旧默认
- 当前 formal direct-path boundary 如果会在多个 lowering 阶段点被重复检查，就要把校验 helper 和错误口径统一起来；否则后续扩支持面时很容易出现“同一限制，多种报错文案，多处散落特判”
- 当 staged copy 同时存在 tile path 和 page path 时，不要让每个 lowering 分支各自重新推 shared/global shape、subtile 维度和 transport 字节数；应先把这些量收成单一 geometry helper（例如 `ResolveStaticShape2DFromBufferOrMetadata` + `StagedCopyTransportGeometry`），再让具体分支只消费 geometry。这样 boundary 校验、index 计算和 transport 选择才能保持同一真源
- staged copy 的 index 提取也不要独立维护第二套几何假设。`InferCopyTileIndex` / `InferStagedCopyBaseTileIndex` 这类“只负责线性化 transport index”的逻辑，应直接复用 shared/global shape helper 和统一的 transport index linearizer；否则 flattened-index path、transpose path、page-vs-tile path 很快又会各自漂出一套边界
- 当 direct runtime 同时保留 schema path 和兼容 legacy path 时，不要让两条路径各自维护一份 accessor 校验或 compile-time arg materialization。应先把 accessor direct-runtime 约束和 append 逻辑收成共享 helper，再让两条路径都调用这份 helper；否则 schema 收正后，legacy path 很容易悄悄漂回旧假设
- 对 Blackhole direct runtime，shared runtime metadata（buffer address / semaphore id）如果同时出现在 `common_runtime_args` 和 `runtime_args` 两条通道里，不要复制两份 kind-switch。应先抽出共享 materializer，再让 common/per-work 构造器复用；这样后面收窄或扩展 shared arg 支持面时，约束不会在两处漂移
- 对 Blackhole direct runtime，per-work 派生值（如 `work_linear_id`、`bx/by`、`num_k_tiles`、`logical_n_tiles`）也不要散在 runtime arg 构造循环里现算现判。先收成 `work context`，再让 per-work materializer 消费它；这样 GEMM reader / writer / copy 的共用语义才能稳定下来，后面扩更多 kind 时也不会继续把主函数撑胖
- 对 `PlanBlackholeCB` 这类 planner pass，正式输入必须是上游显式产出的 schema（这里是 `blackhole.cb_requirements`），不要默认从 `alloc_shared` 这类 IR 形态做推断补洞。planner 一旦既吃正式 schema 又吃 fallback inference，很快就会把“猜出来的行为”沉淀成隐式协议
- 对 codegen 层的 accessor ABI，也要把“当前只是 compile-time-only”写成显式边界，而不是模糊约定。像 `TensorAccessorArgs<CTA>()` 这种路径，如果 slot 不是 compile-time 常量，就应该在 codegen 阶段直接 fail-fast；不要让更宽 accessor / CRTA execution 面以“先打印出来再说”的方式悄悄混入当前主链
- 对 non-tile/stick copy，外部 DRAM buffer 的真实 `page_size` 不是 CB `page_size` 的别名；需要把单次 transport 的 `page_bytes` 明确收进 accessor schema（如 `transport_page_size`），再由 direct runtime 用这份 schema 创建 TT-Metal buffer/accessor
- 当 host runtime 当前只支持一种 buffer materialization（例如 replicated DRAM）时，也不要把它硬编码成执行流里的隐式默认值；应先把每个 runtime buffer 的 materialization descriptor 显式收进 `ExecutableSpec`，再让 runtime 按 descriptor 校验并 materialize
- 当 direct runtime 开始正式支持 compile-time ABI schema-only 路径时，`buffer_materializations` 不能只从 legacy `accessors` 推导；也必须能从 `compile_time_arg_specs` 上带的 `buffer/layout/memory_space` 元数据恢复。否则一旦测试或新 pass strip 掉 `accessors`，runtime 会在真正的 ABI 校验之前先报 “Missing buffer materialization spec”
- 当多个 kernel/segment 共享 runtime arg 或 common runtime arg 时，不要在 spec 提取层继续靠 `kind + name/buffer` 推断“是不是同一个参数”；应由 lowering/split 直接产出稳定 `identity`，`rt_mod_blackhole` 只按 `identity` 聚合，缺失 identity 直接 build-time 拒绝
- runtime/common-runtime arg 的 dedup key 只要进入“同一对象多个分量”的 schema，就必须统一成 `identity + ":" + kind`，而不能在 codegen、segment fallback、spec 提取里各写一套更弱的 dedup。remote core 的 `logical_core_noc_x/y` 就是稳定案例：`identity` 只表示“同一 remote core”，不是唯一 arg key
- TT-Metal 的 `SetCommonRuntimeArgs` 是 kernel-level shared channel，只适合对所有 core/work item 相同的 metadata（如 buffer address、semaphore id）；`work_linear_id`、tile range、logical core coord 这类 per-work/per-core 值不能塞进 common channel
- accessor schema 里凡是叫 `args_config_bits` 的字段，都必须等价于 TT-Metal `tensor_accessor::ArgConfig.raw()`；不要自造“interleaved=1”这类本地编码。当前最小稳定映射是：interleaved+dram=`2`，sharded+dram=`3`，sharded+l1=`1`，interleaved+l1=`0`
- TT-Metal program-local semaphore 当前正式 host API 是 `CreateSemaphore(program, core_ranges, initial_value)`；如果上层 schema 还保留 `core_type`，应把它当校验字段，不要为了“对齐字段”继续依赖 deprecated 的 `CreateSemaphore(..., core_type)`
- 对 TT-Metal program-local semaphore，host/runtime 正式下发的是 semaphore id；device dataflow kernel 再显式 `get_semaphore(id)` 取本地 L1 地址后做 `noc_semaphore_wait/set`。不要把 semaphore 地址或 barrier 绑定错误建模成 compile-time ABI
- 需要跨 worker core 访问 semaphore 时，不要让 kernel 直接猜 remote NOC 坐标，也不要把 logical core 坐标直接塞给 `get_noc_addr(...)`。应把“logical core -> NOC 坐标”收成正式 runtime arg materialization，由 host 用 `worker_core_from_logical_core(...)` 求值后下发
- 对 synchronization runtime schema，`semaphore_id_u32` 和 `logical_core_noc_x/y` 都不该等到 direct execution 时才由 kind-switch 临时发现问题。更稳的边界是：在 `ExecutableSpec` / `KernelSpec` 构造期就校验 semaphore binding 是否唯一且引用已规划 semaphore，同时要求 remote core descriptor 以共享 `identity` 的 `x/y` 成对出现并指向同一 logical core
- 当 shared/common-runtime 路径和 per-work 路径都要消费 synchronization metadata 时，不要一边在 shared arg materializer 里解析 semaphore，一边在 per-work materializer 里单独解析 remote core。应先抽统一 synchronization helper/context，再让两条路径复用；否则 P5 一扩 multicast/global semaphore，很快又会长回两套分叉逻辑
- 如果某类 runtime arg 已经稳定表达成“同一个对象的多个分量”，不要长期只把它们留在 runtime arg 列表里。像 remote worker core 这种 `logical_core_noc_x/y` 成对字段，应该尽快上提成 `KernelSpec` 里的显式 descriptor，再让 runtime arg 只引用 descriptor identity；这样 host/runtime materialization 才不会继续把分量字段本身当真源
- 当测试或 pass 会把 Blackhole builtins 放进 `IfThenElse` / `LetStmt` / 新 `PrimFunc` 里时，这些 builtin 必须注册正确的 `TCallEffectKind`；否则 TVM 会在 purity / struct-info 校验阶段直接拒绝，表现成控制流重写失败
- 任何新增的 blackhole builtin，如果携带 cb_id 参数，**必须**在 `PlanBlackholeCB::GetCBArgPositions` 中注册该参数的 position；否则 `PlanBlackholeCB` 的 IR 回写会跳过该 builtin，导致 cb_id 停留在 requirement_index 值。实际案例：`write_local_slice_to_cb` 漏注册 → compute kernel 写错误 CB → writer 永远等不到数据 → runtime hang
- 对 segment body extraction，当 compute segment 用 `retain_unmarked_stmts_=true` 保留所有未标注语句时，这依赖于 reader/writer 操作都已被 `blackhole.segment_kind` annotation 包裹。如果有未标注的 copy/dataflow 语句，它们会错误地泄漏进 compute kernel
- `ExtractRuntimeArgs` / `ExtractCommonRuntimeArgs` 在聚合 segment runtime args 时做 dedup，dedup key 必须同时包含 `identity` 和 `kind`（即 `identity:kind`）。`identity` 是分组标识（如同一 remote core 的 x/y 分量），不是唯一标识。只用 `identity` 做 dedup 会丢弃同组内 kind 不同的 arg。实际案例：`logical_core_noc_x/y` 共享 identity `remote_consumer_core`，只用 identity dedup 导致 `_noc_y` 被跳过
- 对 TT-Metal execution hang，最先用的不是盲目加日志，而是 Watcher。当前仓库环境下，开启 `TT_METAL_WATCHER=2` 后的 watcher 输出默认写到工作目录下的 `generated/watcher/watcher.log`；复现同一 hang 时，稳定不变的 BRISC/NCRISC/TRISC 状态码组合很适合判断“修掉的是局部协议 bug，还是已经推动了死锁边界”
- 对 `blackhole.acc` 这类 scratch CB，如果结果会被后续 matmul 当输入消费，producer 侧发布页数必须按未来 consumer 的 tile/page 需求来算，而不是按当前 pointwise/cast 自己写了几次就发几页；否则 compute 会在下一次 `mm_init` / `cb_wait_front` 上静默挂死
- 当 `blackhole.acc` 既承载 scratch storage 又承载 CB 生命周期时，matmul output path 不能再无条件沿用 transport-CB 的 `cb_reserve_back -> pack_tile -> cb_push_back` 模板；是否允许 reserve/push 必须由该输出 scope 的生命周期模型决定
- 对 `Phase B` 的 pipeline-stage truth migration，不能把 `blackhole.pipeline_stages` 直接从 `LowerBlackholeOps` 里删掉。更稳的 cutover 顺序是：
  `AnalyzeSemanticStructure` 先把 stage truth 收成 `SemanticSupplement(kind=pipeline_structure)`，
  `LowerToSpatialProgram` 再把它投影成
  `ResourceIntent(kind=synchronization_support, traits+=pipeline_contract, payload=...)`，
  然后 `LowerBlackholeOps` 改成 spatial-program-first 读取，legacy attr 和 body annotation 最后才降成 fallback
- 对 `Phase B` 的 work-decomposition truth migration，也不要让 `LowerBlackholeOps` 继续直接绑定
  `blackhole.work_decomposition.work_dependent_loop_bounds`。更稳的顺序是：
  `AnalyzeSemanticStructure` 先把这份信息收成
  `SemanticSupplement(kind=work_decomposition_structure)`，
  `LowerToSpatialProgram` 再把它投影成
  `WorkPartition.payload.work_dependent_loop_bounds`，
  `ValidateSpatialProgram` 对 `work_dependent_bounds` domain 强制要求 payload，
  最后 `LowerBlackholeOps` 再改成 spatial-program-first 恢复
  `work_dependent_loop_bound_count`
- 对 `Phase B` 的 fragment truth migration，也不要让 `LowerBlackholeOps` 直接把
  `blackhole.fragment_regions` 当 primary input。更稳的顺序是：
  `AnalyzeSemanticStructure` 先把 lowering-facing fragment summary 收成
  `SemanticSupplement(kind=fragment_lowering_structure)`，
  `LowerToSpatialProgram` 再把它投影成
  `ResourceIntent(kind=lowering_support, traits+=fragment_contract, payload=...)`，
  `ValidateSpatialProgram` 对 fragment program 强制要求 contract，
  最后 `LowerBlackholeOps` 再改成 spatial-program-first 恢复
  `fragment_op_kinds / row_reduction_targets / row_broadcast_sources /
  pointwise_op_kinds / fragment_loop_carried_state`
- 一旦 `LowerBlackholeOps` 的 lowering-requirements 已经能完全从
  `tl.spatial_program` 恢复，就不要继续保留“没 `SpatialProgram` 也能凑合跑”的 legacy
  fallback。更稳的 cutover 是直接让 `LowerBlackholeOps` 硬要求
  `tl.spatial_program`，然后把 target/transform tests 一起切回真实主线；否则测试会长期
  伪装成“后端还支持 legacy-only 输入”，把 `Phase B` 的单一真源边界重新污染掉
- 对 `Phase B` generic spatial builder，也不要用 `root_map` 这类 update name 去判
  “这个 update 该不该 materialize 成 task”。更稳的主链是 generic path 直接按
  semantic `Update` object 建 task；名字只能留给 IR object identity、调试和打印，
  不承担协议分支职责
- 对 `Phase B` stronger-contract schema，也不要只做“名字还在，但大家约定优先不用”。
  更稳的 cutover 是先把跨层 linkage 收成显式 contract，再让 validator / consumer
  改成优先吃这层 contract。当前稳定模式是：
  - `SpatialLayout.payload.domain_index` /
    `WorkPartition.payload.domain_index`
    绑定 `SemanticProgram.domains[*]`
  - `ResourceIntent.payload.target_kind + target_index`
    绑定 semantic-state target
  - `LowerBlackholeOps` 这类 consumer 再按 `target_index` 恢复对象，
    而不是回头按 `target_name` 字符串重新查表
- 对 `Task / Channel / Placement / SyncEdge / ProgramPhase` 这类
  `SpatialProgram` 结构对象，也不要继续把 `phase_name / task_name / source_task /
  target_task / channel_names` 当隐式 linkage 协议。更稳的 cutover 是：
  - `Task.payload.phase_index`
  - `Channel.payload.source_task_index / target_task_index / state_index`
  - `Placement.payload.task_index`
  - `SyncEdge.payload.source_task_index / target_task_index`
  - `ProgramPhase.payload.phase_index / task_indices / channel_indices`
  先变成 mandatory contract，
  然后 `ValidateSpatialProgram` 再改成 contract-first 校验；
  display-name 字段只保留 identity / debug 角色
- 对 Blackhole 的 device resource canonicalization，不要把
  `blackhole.resource_plan` 当唯一真源。`grouped / routed / paged` 这类 family 的
  block-local shared alloc_buffer 很可能还没先出现在 plan 里，但 IR storage scope
  已经足够表达它应该变成 `blackhole.cb` 还是 `blackhole.acc`。更稳的做法是只在
  Blackhole-only canonicalizer 里补 IR-structural fallback
  （`shared* -> blackhole.cb`、`local.fragment -> blackhole.acc`），
  而不是去改跨平台公用的 `MergeSharedMemoryAllocations`

## Blackhole 后端开发原则

1. 不把单个 kernel 字符串当成后端主产物
2. 新功能落到 `ExecutableSpec → BlackholeModule::ExecuteDirect()` 主路径
3. 架构分层要和当前总设计一致：`Stateful Semantic IR → Spatial Program IR → TT Target IR → host/runtime materialization`
4. planner 的 identity 和 lifetime 必须分成两个独立维度（requirement_index ≠ lifetime）
5. 新 schema 先隔离到新场景，不卷入已稳定路径；稳定后再统一
6. 先修 bug（最小改动验证），再做 schema 工程（协议质量提升）

## 文档治理

- 阶段状态切换时，至少同步检查：progress.md 页首、任务表、活动设计文档列表
- 顶层 `README.md`、`AGENTS.md`、`CLAUDE.md` 只能描述当前主线，不要继续保留被 supersede 的阶段入口或旧计划路径
- 做活动文档审计时，不要只改 `tasks/progress.md`。更稳的做法是把 `README.md`、`AGENTS.md`、`CLAUDE.md`、
  `final_blackhole_backend_redesign.md` 和当前 active stage docs 一起对齐；否则 pass 主线、当前 blocker、
  `fragment_regions` 角色这类事实很快会在不同入口里漂移
- 排障文档一旦问题已解决，要在文档头部把状态改成“已实施/历史记录”，并把最终根因补回去；不要让“初始假设”继续伪装成当前结论
- 设计文档分三类：当前活动、仍有效支撑、历史记录
- 历史文档统一移到 `tasks/dev_design/archive/`；根目录只保留当前活动文档
- 历史文档不重写正文，在开头加头注标明历史性质
