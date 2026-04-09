# 通用开发模式与当前有效经验

> 当前 Blackhole 后端唯一设计依据: `tasks/dev_design/final_blackhole_backend_redesign.md`
> 本文档只保留长期可复用的工作模式；不承担阶段状态、bug 目录或验证快照职责。

## 1. 文档使用模式

- 当前活动入口固定为：
  - `tasks/dev_design/final_blackhole_backend_redesign.md`
  - `tasks/dev_design/README.md`
  - `tasks/progress.md`
- `tasks/dev_design/archive/` 只当历史记录，不再作为当前任务入口
- 设计边界写在设计文档，阶段状态写在 `progress.md`，`memory/` 只记长期经验

## 2. 构建模式

- C++ 改动后，先确认 `libtilelang.so` 已重编，再跑 pytest
- 当前 `tilelang_repo/CMakeLists.txt` 的 Blackhole 源列表不是“目录自动全量收集”；
  split 新 `.cc` 文件后，必须把它显式接进 `TILE_LANG_BLACKHOLE_SRCS`，
  然后重新执行一次 CMake 并重链 `libtilelang.so`
- 同一个 `tilelang_repo/build/` 不要并行跑 `cmake --build` 和 pytest
- `pip install -e .` 不是默认开发路径；更稳的是用 `.pth` 指向本地构建产物
- `3rdparty/` 和 `build/` 不进主仓库

## 3. 验证模式

测试层级固定分三层：

- 结构层：
  - lowered TIR、attrs、companion IR
- planner 层：
  - `ExecutableSpec`、`KernelSpec`、bindings、CB / core / semaphore 规划结果
- runtime 层：
  - direct path 真执行

经验上：

- 只做 codegen/reference compare 不算 true E2E
- 当前支持面和 fail-fast 边界都应该在更早层被看见，不要全部压到 runtime

## 4. Layered IR / companion 模式

- module-scope truth 放 `IRModule.global_infos`，不要回退到单个 `PrimFunc.attrs`
- unsafe TIR mutation 必须整体 invalidate companion truth，不要只删单个 attr
- cross-pass schema 一律 handle-first；字符串只保留 display / debug / compatibility 角色
- semantic truth 属于 `Phase A`，spatial truth 属于 `Phase B`，TT target truth 属于 `Phase C`
- 下游 consumer 优先读最近的 typed IR，不要回头把 legacy attrs 当 primary truth
- 如果 analysis 需要某类 operator-specific 事实，
  让 operator/semantic owner 暴露 typed metadata；
  不要把 `Analyze...` pass 写成 op family 名字匹配器

当前稳定 companion 习惯：

- seed / manifest / witness / program 分层存放
- intermediate typed plan 只要进入 pass 链，就要和最终 companion truth 一起纳入
  invalidation；只删 `tl.spatial_program` 而保留 `tl.spatial_domain_plan` /
  `tl.spatial_execution_plan` 会制造 stale plan
- workload noun 不进入长期 semantic schema
- evidence carrier 不是 truth owner
- 兼容 attr 的删除顺序固定为：
  先移走 semantic consumer，再移走 lowering consumer，最后删 attr 本身
- 当新的 typed truth 需要 bridge 回 legacy projection 时，必须显式区分
  “节点原生拥有该字段” 和 “只是借用 top-level fallback”；
  否则 materializer 会把 fallback 意外下沉成 per-node 真相，
  破坏旧测试和 reader contract
- function-level target contract（如
  `gemm_contract / compute_contract / direct_runtime_unsupported_reasons`）
  一旦进入 runtime/codegen 正式消费面，就应提升进 `TTProgram.payload`；
  bridge attr 只能留作 compatibility fallback
- 这类 function-level contract 若先在 device `ExecutableSpec` 上补充，
  host entry metadata 也必须同步拷回；否则 Python/runtime gate 仍会看见过时视图，
  以为 kernel 没有 unsupported reason
- 一旦原始 device func 已切到 typed target truth，
  shared projection helper 就不能再同时承担
  “`TTProgram` reader” 和 “legacy attr fallback” 两种职责；
  必须拆成 `TTProgram`-only reader 与本地 materialization helper，
  否则会把单一真源再次偷偷变成双真源
- synthetic segment / internal kernel emission 也应遵守同一规则：
  如果内部还需要重建 target-truth，就直接挂最小单-kernel `TTProgram`，
  不要再把 `segment/runtime/cb/core` 重新降回局部 `blackhole.*` attrs
- per-work/access truth 一旦 formalize 成 `per_work_arg_specs`，
  就要先 canonicalize 成 kernel-local `TTKernel / ExecutableSpec` contract；
  codegen/runtime 只能解释 `value_kind`，不能再按 arg kind 名字推语义
- Python 侧若需要做 companion IR mutation regression，
  优先通过 `tl.TT*` constructor 直接重建
  `TTProgram / TTKernel / TTCoreGroup / TTABIPlan / TTSemaphorePlan`
  并重新跑 `ValidateTTTargetProgram`；
  不要先改 bridge attrs 再依赖 translator 刷新 typed truth
- bridge-stage 若还没 materialize `tl.tt_program`，
  regression/helper 也应优先读取
  `tl.tt_kernel_seeds / tl.tt_abi_plans / tl.tt_program_payload`；
  一旦这些 typed seeds 已发布，`LowerBlackholeOps` 输出就应立即剥离
  `blackhole.segment_plan / runtime_args / gemm_contract` 等 projection attrs，
  不要让 producer-side cleanup 被测试层 fallback 反向卡住

## 5. Schema / ABI 模式

- compile-time、common-runtime、per-work 三层 ABI 必须严格分开
- runtime 参数布局必须显式、可验证；不要依赖默认顺序、默认名字或 host 猜位置
- `common_runtime_args` 只放共享 metadata；per-work / per-core 值单独下发
- work descriptor 要用角色化字段，不用单值默认去反推整套语义
- 64-bit 地址需要明确拆分 / 重组规则

对象分组与派生规则：

- arg identity 必须由 lowering / split 正式产出
- dedup key 一律用 `identity + ":" + kind`
- 多个字段共同表达一个对象时，应尽快上提成 schema object
- schema-only 路径一旦成立，派生物也必须能从 schema 单独重建
- 未正式支持的 ABI / accessor / transport 组合，要 build-time fail-fast
- 不保留默认 ABI、默认 core、默认 packet 这类补洞
- multi-GEMM / staged-copy reader 的 transpose truth
  不能只留在 compute contract；如果 host materialization 也要配合，
  就必须显式进 accessor/materialization schema
  （例如 `transpose_2d`），并由 host tilize / readback 真正执行
- 如果 planner 已正式产出 `core_plan.work_packets` 且允许
  `work_count > 1`，direct runtime 不能再把 packet 扁平成
  “单波次 one-work-per-core” 假设；对还没把 `work_count`
  下沉成 device-side loop contract 的 executable，至少要按 packet truth
  做 repeated launch / wave scheduling，避免同一 core 的 runtime args 被后写覆盖
- 一旦 reader-side cutover 成立，原始 device build 输入就应硬要求
  `tl.tt_program`；不要让 build 在缺失 TT truth 时再悄悄回退到 legacy attrs

## 6. analysis / lowering / planner / codegen 模式

- analysis 产出事实
- lowering 消费事实并改写 IR
- planner 只消费显式 requirement schema
- codegen 只打印已确定 contract

稳定做法：

- 如果 pass 只把规划结果写到 attrs 而不回写 IR body，就会制造两套真源；
  优先让 pass 同时完成 IR 回写
- `Phase B` 内部若出现一个 lowering 同时做 domain synthesis、task formation、
  phase ordering 和 final materialization，应优先拆成
  `Analyze... -> Analyze... -> Materialize...` 的 pass 链，让 analysis facts
  先以 typed plan 落地，再由 materialize pass 组装最终 companion IR
- unsupported subset gate 应在所有后端出口共享
- gate 应按具体 contract / op family 报错，不要长期用黑盒总括词
- 需要的信息优先从 typed IR / schema 拿；拿不到就扩 IR / schema，
  不要把猜测沉淀成长期协议
- 如果某条 runtime/codegen 路径持续需要 gate
  才能避免错跑，先回头判断它是否其实在暴露
  上游 `SpatialProgram` 的 `dataflow/work-access contract` 缺口；
  不要把“后段还没实现协议执行”误判成第一 blocker

## 7. Resource / storage 模式

- Blackhole `shared` 的主映射是 CB / L1 资源，不是普通 C 数组
- `local` 是中间语义桶；进入后端后要继续分流成：
  - 真正的小标量临时
  - fragment / accumulator
  - 显式 dataflow transfer
- 一旦 residual `local` 已经明显表示“结果写回 CB 的桥接语义”，
  就应尽快 lower 成正式 builtin / direction
- planner 的正式输入必须是上游显式 schema；
  不要一边吃 schema，一边从 IR 形态做 fallback inference
- target 资源模型与 generic backend 不一致时，优先扩 IR / schema，
  不要给后段 pass 打豁免

## 8. TT-Sim / TT-Metal 模式

统一入口：

```bash
source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh
export TILELANG_HOME=<当前 checkout 或 worktree>/tilelang_repo
cd <当前 checkout 或 worktree>/tilelang_repo
```

稳定经验：

- `setup_tt_sim.sh` 和测试命令必须在同一个 shell
- `TT_METAL_SLOW_DISPATCH_MODE=1` 对 TT-Sim 很关键
- direct path kernel 临时目录必须每次执行唯一化
- 优先消费 TT-Metal local install tree，不要把 `.cpmcache` 整片塞进 include path
- 现阶段 Blackhole runtime/direct-runtime regression 默认统一用 `bf16` 输入；
  不要再把 `fp16` 当成 TT-Sim 上的正式 runtime baseline
- 对 `flash-attn` / multi-op compute kernel 的
  producer-consumer / republish 判断，
  owner 应该是上游 `fragment_lowering_structure -> SpatialProgram`
  的 typed `fragment_buffer_flow_contracts`；
  `LowerBlackholeOps` 不应再本地扫 `SeqStmt`
  恢复 `write / compute_consume / transport_consume` 语义
- 这组 flow contract 不能只覆盖 fragment/local intermediate buffer；
  还要覆盖 compute kernel 内参与同一 producer-consumer 协议的 CB-backed input buffer，
  否则 retain/pop 策略会在 lower 侧退回成“缺 contract 默认即时 pop”
- TT-Sim `float16` 路径是否可用要和 target contract 问题分开判断；
  如果 small bf16 correctness 已过、但大 shape `float16` 命中
  `UntestedFunctionality: tensix_execute_unpacr: fp16`，
  优先视为 simulator 能力边界，而不是先回退刚验证过的 target contract 修复

稳定 host-side 抽象：

- `Program`
- `CreateCircularBuffer`
- `CreateKernel` / `CreateKernelFromString`
- `SetRuntimeArgs`

稳定同步 / 资源模式：

- CB 既需要地址共享，也需要同步原语
- semaphore 在 host/runtime 侧的正式对象是 id，不是地址
- remote worker core 的 logical -> NOC 坐标转换由 host materialize

## 9. 调试模式

- 先判断问题落在哪一层：
  - 结构层
  - planner/spec 层
  - runtime 执行层
- 手工 pass 链和 full `lower()` 不一致时，先比对 optimized path 的 IR 形态，
  再检查入口 `PrimFunc` 是否被提前过滤
- compile / launch 已经通过时，优先查 CB 生命周期、同步协议和 runtime arg materialization
- copy 跑通只证明字节路径可用，不证明 matmul / tile layout contract 正确
- execution hang 优先配合 Watcher 看状态组合，而不是先堆日志
- 如果开 `TT_METAL_WATCHER` 后症状从 hang 变成 `SIGABRT` 或只在 dump 期间卡住，
  先抓 native backtrace；问题可能在 `WatcherServer` 线程，而不是 direct runtime 主链
- 需要保留 watcher 现场但避免立即 abort 时，可临时开 `TT_METAL_WATCHER_TEST_MODE=1`
