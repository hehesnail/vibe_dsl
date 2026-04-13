# Task 3: Runtime Gate 与 Workload Cutover

## 基本信息

- **文档角色**: `Task 3` 的 runtime gate、support surface
  与 workload re-enable 设计文档
- **当前状态**: `2026-04-13` 活动设计文档；`Task 2` 已完成，
  当前工作进入 `Task 3`
- **任务链位置**:
  `Task 2` owner cutover 完成之后，
  负责 runtime/correctness payoff 与 wider family 承接
- **非目标**:
  - 不重新定义 `SpatialPlan` / `TTProgram` 的 owner 边界
  - 不在 runtime/codegen 里补 planning contract
  - 不恢复旧 recovery 主链
  - 不把 non-Blackhole backend 一并收口到同一套 runtime / artifact contract
  - 不把 public Python `transform` API 改名当作本任务目标
- **唯一总体设计**: `tasks/dev_design/final_blackhole_backend_redesign.md`
- **上游设计输入**:
  - `tasks/dev_design/task1_spatial_plan_companion.md`
  - `tasks/dev_design/task2_ttprogram_companion_cutover.md`

## 1. 作用域

`Task 3` 只负责两件事：

- 在新主链上兑现 runtime / correctness payoff
- 在新主链上重新承接 workload family 与更宽 support surface

这里的“新主链”固定指：

```text
Normalized Tile TIR
  -> SpatialPlan companion
  -> TTProgram companion
  -> ExecutableSpec
  -> runtime / codegen / BlackholeModule
```

`Task 3` 不重新讨论 owner 链本身。
如果 runtime 或某个 workload 仍然缺 truth，
结论只能回到：

- `Task 2` owner cutover 还没完成
- 上游 `TIR/DSL` 或 analysis 还缺 truth
- 当前 variant 必须显式 `unsupported`

补充边界：

- `Task 3` 默认只讨论 `Blackhole` active runtime / codegen path
- non-Blackhole backend 的统一化不进入当前 backlog

## 2. Shared Zero-Regression Baseline

无论 `Task 3` 扩多少支持面，下面这些基线都不能回退：

- `ExecutableSpec -> rt_mod_blackhole -> BlackholeModule`
  direct host path 仍是唯一正式执行路径
- `tilelang.compile(..., execution_backend="tvm_ffi")`
  的 Blackhole wrapper/export path 不回退
- copy / GEMM / export 当前正式支持面不回退
- Blackhole runtime / direct-runtime 正式 correctness baseline
  统一使用 `bf16`
- 对未支持功能继续 explicit unsupported / fail-fast，
  不回退到 silent fallback 或 late runtime failure

## 3. Runtime Gate

runtime / codegen 的消费纪律固定为：

- 只读 `ExecutableSpec`
- 不读 legacy attrs
- 不补 target planning
- 不从 `work_linear_id`、arg kind、builtin 序列、
  payload bag 恢复语义

当前及后续 gate 一律按同一原则收口：

1. **per-work access truth 缺失**
   - 如果 executable 没有显式 per-work access descriptor，
     runtime 不允许再从 `work_linear_id` 重建 tile access
2. **fragment / intermediate execution protocol 缺失**
   - 如果 executable 还缺显式 materialization / merge /
     republish / transport truth，
     runtime 不允许再从 builtin 序列猜语义
3. **target sync truth 缺失**
   - 一旦 executable 带显式同步对象或绑定，
     但 runtime 还没有对应执行语义，
     必须 fail-fast

这条 gate 的目的不是“先挡住错误 case”，
而是强迫所有新支持面都经由 owner-side typed truth 落地。

## 4. `flash-attn` 与 runtime payoff

`flash-attn` 在 `Task 3` 里的角色是：

- 作为第一批验证新主链是否真的能承接
  multi-op / multi-work / intermediate dataflow 的 workload

对这条线，完成标准不是“compile-path 通了”，而是：

1. 支持的 `MHA / GQA` subset
   在 admitted support surface 内真实执行
2. runtime 结果与 reference 数值对齐，
   不只是“不挂死”
3. 对仍未支持的 shape / variant
   给出显式 unsupported / fail-fast
4. 不再依赖旧 recovery 主链、
   legacy attrs、名字匹配或 codegen heuristics

## 5. Wider Family Cutover

`Task 3` 必须在新主链下重新承接下面这些 family：

- `topk`
- `fusedmoe`
- `paged decode`
- `chunk recurrence`

对每个 family，最低完成标准固定为：

1. 走完整新主链：
   `Normalized Tile TIR -> SpatialPlan companion ->
   TTProgram companion -> ExecutableSpec`
2. target truth 从 `TTProgram / ExecutableSpec` 物化，
   不是 runtime/codegen 侧补洞
3. 至少有一组明确支持的 subset，
   带有对应的 transform / target regression
4. 对未支持部分有显式 unsupported / fail-fast 边界

## 5.1 当前执行优先级

`Task 3` 的推进顺序固定为：

1. **P0: runtime gate 收口**
   - 先确保新主链上的 copy / GEMM / export
     仍维持当前 zero-regression baseline
2. **P1: `flash-attn` payoff**
   - 先拿它验证 multi-op / multi-work /
     intermediate dataflow 的 admitted subset
3. **P2: wider family cutover**
   - `topk -> fusedmoe -> paged decode -> chunk recurrence`
4. **P3: wider copy / dataflow**
   - 在 admitted subset 内逐步放宽 range / stride / staged-dataflow
5. **P4: wider sync**
   - 只在 `TTSyncPlan + executable binding + runtime semantics`
     三者都稳定后进入 admitted surface

## 6. Wider Copy / Dataflow / Sync 支持面

支持面扩张也必须服从同一条 owner 纪律。

### 6.1 Copy / Dataflow

后续可以扩张：

- wider copy range / stride / layout subset
- wider staged-copy / multi-stage dataflow subset
- 跨 op intermediate dataflow 的更多 admitted subset

但扩张前提只能是：

- 上游 `TIR/DSL` 已经显式表达所需 truth
- analysis 已能稳定读出
- `TTProgram / ExecutableSpec`
  已把对应 target truth 冻结下来

### 6.2 Synchronization

后续可以扩张：

- 显式 semaphore / compute sync / barrier admitted subset
- 更多 local / remote / multicast synchronization case

但所有同步支持都必须满足：

- `TTSyncPlan` 已成为稳定 target truth
- executable 已能稳定承载对应 binding / ordering
- runtime 对该 sync family 有明确执行语义

在这之前，一律 fail-fast。

## 7. 完成判定

`Task 3` 完成必须同时满足：

1. `flash-attn` admitted subset
   在新主链上真实通过 runtime/correctness gate
2. `topk / fusedmoe / paged decode / chunk recurrence`
   至少各有一组支持 subset
3. wider copy / dataflow / sync 支持面
   通过 `TTProgram / ExecutableSpec` 扩张，
   不靠 fallback
4. 对未支持部分仍保持显式 unsupported / fail-fast
5. 旧 recovery 主链不再作为任何 family 的兜底路线
