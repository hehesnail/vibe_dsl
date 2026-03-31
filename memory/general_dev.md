# 通用开发模式与当前有效经验

> 当前 Blackhole 后端唯一设计依据: `tasks/dev_design/final_blackhole_backend_redesign.md`
> 本文档只保留稳定、可复用的工程经验。

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
- 当 target 硬件资源模型与 generic backend 不一致时，优先扩 IR 类型系统（如 StorageRank），不要给后段 pass 打豁免
- 判断"该不该扩 IR"的信号：同一根因在多个 generic pass 上以不同报错出现

### 测试策略

- 只做 codegen/reference compare 的脚本不算 true E2E
- 测试分层：结构层（lowered TIR/attrs）→ planner 层（cb_configs/bindings）→ runtime 层（direct path 真执行）

## TileLang 工程

- `3rdparty/` 和 `build/` 不应进入主仓库提交
- `pip install -e .` 可能重新触发构建并失败，用 `.pth` 指向本地构建产物
- C++ 改动后 pytest 前先确认 `libtilelang.so` 已重编，避免加载旧库假阴性
- 不要对同一个 `tilelang_repo/build/` 并行跑 `cmake --build` 和 pytest。共享构建目录在链接进行中时，测试可能加载到旧/半更新的 `libtilelang.so`，制造假阴性或顺序相关噪声
- 对新的 split-after analysis pass，优先把结果写成结构化 IR attrs（`Array<Map<...>>`、`PrimExpr` 等），不要先字符串化再让后续测试/consumer 反解析。测试也应直接断言 attr 结构和 `PrimExpr` 语义，而不是只查字符串片段
- 对 Blackhole `lower()` 主链，不能在 `SplitBlackholeKernel` / `Analyze*` / `LowerBlackholeOps` 之前就用旧的 device attrs 过滤掉入口 `PrimFunc`。Blackhole entry kernel 在这条链之前通常还没有 `blackhole.*` attrs，因此 `is_device_call()` 必须把 entry `PrimFunc` 视为 device 输入，否则专属 pass 实际上跑在空 `device_mod` 上
- fragment region analysis 里的 `pointwise_chain` 不能通过全局扫描所有 `tir.add/mul/div/max/...` 来判定；那样会把普通索引算术也误记成 fragment compute。更稳的做法是只在 fragment/local region 自身的 store / dataflow 关系里识别 pointwise
- 对 split-after TIR 的 fragment analysis，不要只盯 `CallNode`。像 `scores_sum[0] + acc_s[rv]`、`T.max(scores_max[0], tmp[0])` 这类模式在 TVM IR 里常常是 `AddNode` / `MaxNode` / `MulNode` / `DivNode` 等原生表达式节点；如果只扫 `CallNode`，row reduction 和 scalar-to-vector broadcast 会在真实 MHA/GQA IR 上整片漏掉
- 当 fragment analysis 需要为后续 legality/lowering 提供更细粒度输入时，不要只保留一个粗粒度 `pointwise_chain` 标签。先在 analysis attrs 里把 `fill / exp2 / cast / max / add / mul / div / if_then_else` 这类点算子枚举出来，再让后续 lowering 决定哪些是当前子集、哪些要继续 fail-fast；否则 build-time gate 永远只能对一个黑盒 `pointwise_chain` 报错
- 当 build-time gate 还没真正支持 fragment 子集执行时，也不要继续按 `pointwise_chain` 这种总括词报错。更稳的做法是：`LowerBlackholeOps` 先汇总细粒度 `pointwise_op_kinds`，`rt_mod_blackhole` 再按具体 unsupported 集合（例如 `row_reduction / row_broadcast / fill / mul / ...`）显式 fail-fast。这样后续实现 fragment lowering 时，可以逐项消掉 blocker，而不是每次都改一整块黑盒 gate
- flash-attention analysis 的测试要按 IR 所在阶段写预期，不要强行要求 split-after IR 和 optimized device IR 暴露完全相同的点算子集合。像 `if_then_else` mask init、`exp2` 这类语义，在 causal 与 non-causal 之间、以及 prepasses 之前和之后，本来就可能以不同节点或已折叠形态出现；测试应锁住该层 IR 仍然可见的结构信号，而不是复用源码层的完整 op 清单
- 当 analysis 结果还不能直接 lower 成可执行 kernel 时，不要把 raw analysis attrs 直接塞进 `ExecutableSpec`。更稳的过渡做法是：先在 `LowerBlackholeOps` 里把它们归一化成一层很薄的 IR attrs summary（例如 `blackhole.lowering_requirements`），再由更后面的 build/runtime gate 消费并 fail-fast。这样主链能先完成“analysis 被 lowering 真正接住”，又不会把半成品 descriptor 永久冻结进 runtime schema
- 对 pipelined fragment subset 的早期 legality，优先直接读 loop annotation（例如 `ForNode.annotations["num_stages"]`），不要假设更后面的 region canonicalization 一定先成功。这样即使更宽 GQA/MHA 形态还会在别的 lowering 细节上炸，主链也能先把“这组 stage 配置本来就不支持”显式拦下来
- row-broadcast 的索引归并信号不要只认 `floor_div/floor_mod`。在 split-after TIR 里，同一类“coarsened row index”也可能被写成 `tir.shift_right(i, k)`；如果 analysis 只认除法不认右移，GQA 这类更宽 fragment pipeline 很容易少报 `row_broadcast`
- 对优化后的 device IR 做 analysis 时，buffer view 名经常会带 `_1/_2/...` 这类阶段化后缀；analysis 优先按 canonical buffer name 工作，再把具体 view 当作观察入口。否则同一个逻辑 fragment/shared buffer 会在 pass 里被看成多个对象
- `T.Pipelined` 经过 device prepasses 后，stage 注解不一定还叫 `num_stages`；Blackhole analysis 至少要同时兼容 `num_stages` 和 `tl_pipelined_num_stages`，否则 optimized path 会比 split-after path 少一层 pipeline 语义
- `tl.region` 不要假设 `BufferLoad` 索引数必须与 extents 个数完全相等。对 staged/shared view，常见形态是“leading stage index + trailing tile extents”；更稳的 bridge 是把未匹配的前导索引收成 singleton axes，再用提供的 extents 重建尾部 region

## TT-Metal / TT-Sim 环境

### 构建

- 系统依赖、clang 版本、RPATH、LD_LIBRARY_PATH 都影响编译和运行
- TileLang direct 模式需对齐 C++20、TT_METAL_HOME、tt_stl、hostdevcommon、umd
- 优先消费 TT-Metal local install tree（`find_package(tt-metalium CONFIG REQUIRED)`），不要把 `.cpmcache` 整片加进 include path

### TT-Sim

- `TT_METAL_SLOW_DISPATCH_MODE=1` 对 TT-Sim 很关键
- `scripts/setup_tt_sim.sh` 必须在执行测试的同一 shell 里 source
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
- 当多个 kernel/segment 共享 runtime arg 或 common runtime arg 时，不要在 spec 提取层继续靠 `kind + name/buffer` 推断“是不是同一个参数”；应由 lowering/split 直接产出稳定 `identity`，`rt_mod_blackhole` 只按 `identity` 聚合，缺失 identity 直接 build-time 拒绝
- TT-Metal 的 `SetCommonRuntimeArgs` 是 kernel-level shared channel，只适合对所有 core/work item 相同的 metadata（如 buffer address、semaphore id）；`work_linear_id`、tile range、logical core coord 这类 per-work/per-core 值不能塞进 common channel
- accessor schema 里凡是叫 `args_config_bits` 的字段，都必须等价于 TT-Metal `tensor_accessor::ArgConfig.raw()`；不要自造“interleaved=1”这类本地编码。当前最小稳定映射是：interleaved+dram=`2`，sharded+dram=`3`，sharded+l1=`1`，interleaved+l1=`0`
- TT-Metal program-local semaphore 当前正式 host API 是 `CreateSemaphore(program, core_ranges, initial_value)`；如果上层 schema 还保留 `core_type`，应把它当校验字段，不要为了“对齐字段”继续依赖 deprecated 的 `CreateSemaphore(..., core_type)`
- 对 TT-Metal program-local semaphore，host/runtime 正式下发的是 semaphore id；device dataflow kernel 再显式 `get_semaphore(id)` 取本地 L1 地址后做 `noc_semaphore_wait/set`。不要把 semaphore 地址或 barrier 绑定错误建模成 compile-time ABI
- 需要跨 worker core 访问 semaphore 时，不要让 kernel 直接猜 remote NOC 坐标，也不要把 logical core 坐标直接塞给 `get_noc_addr(...)`。应把“logical core -> NOC 坐标”收成正式 runtime arg materialization，由 host 用 `worker_core_from_logical_core(...)` 求值后下发
- 对 synchronization runtime schema，`semaphore_id_u32` 和 `logical_core_noc_x/y` 都不该等到 direct execution 时才由 kind-switch 临时发现问题。更稳的边界是：在 `ExecutableSpec` / `KernelSpec` 构造期就校验 semaphore binding 是否唯一且引用已规划 semaphore，同时要求 remote core descriptor 以共享 `identity` 的 `x/y` 成对出现并指向同一 logical core
- 当 shared/common-runtime 路径和 per-work 路径都要消费 synchronization metadata 时，不要一边在 shared arg materializer 里解析 semaphore，一边在 per-work materializer 里单独解析 remote core。应先抽统一 synchronization helper/context，再让两条路径复用；否则 P5 一扩 multicast/global semaphore，很快又会长回两套分叉逻辑
- 如果某类 runtime arg 已经稳定表达成“同一个对象的多个分量”，不要长期只把它们留在 runtime arg 列表里。像 remote worker core 这种 `logical_core_noc_x/y` 成对字段，应该尽快上提成 `KernelSpec` 里的显式 descriptor，再让 runtime arg 只引用 descriptor identity；这样 host/runtime materialization 才不会继续把分量字段本身当真源
- 当测试或 pass 会把 Blackhole builtins 放进 `IfThenElse` / `LetStmt` / 新 `PrimFunc` 里时，这些 builtin 必须注册正确的 `TCallEffectKind`；否则 TVM 会在 purity / struct-info 校验阶段直接拒绝，表现成控制流重写失败

## Blackhole 后端开发原则

1. 不把单个 kernel 字符串当成后端主产物
2. 新功能落到 `ExecutableSpec → BlackholeModule::ExecuteDirect()` 主路径
3. 三层分工：split 前保语义 → split 后提正式 plan → host side 只做 materialization
4. planner 的 identity 和 lifetime 必须分成两个独立维度（requirement_index ≠ lifetime）
5. 新 schema 先隔离到新场景，不卷入已稳定路径；稳定后再统一
6. 先修 bug（最小改动验证），再做 schema 工程（协议质量提升）

## 文档治理

- 阶段状态切换时，至少同步检查：progress.md 页首、任务表、活动设计文档列表
- 排障文档一旦问题已解决，要在文档头部把状态改成“已实施/历史记录”，并把最终根因补回去；不要让“初始假设”继续伪装成当前结论
- 设计文档分三类：当前活动、仍有效支撑、历史记录
- 历史文档不重写正文，在开头加头注标明历史性质
