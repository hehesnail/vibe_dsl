# TileLang Blackhole 后端重设计

## 基本信息

- **文档 ID**: `final_blackhole_backend_redesign`
- **日期**: `2026-04-07`
- **状态**: 当前唯一权威总体设计文档
- **定位**: 轻量总纲；只保留长期架构、层间边界、真源规则与当前阶段判断
- **活动阶段文档**:
  - `tasks/dev_design/stage4_stage0_guardrails.md`
  - `tasks/dev_design/stage4_phase_a_semantic_ir.md`
  - `tasks/dev_design/stage4_phase_b_spatial_ir.md`
  - `tasks/dev_design/stage4_phase_c_tt_target_ir.md`

## 1. 问题定义

Blackhole 当前的核心问题不是“还差一个 kernel emitter”，而是三类 truth
长期混在一层里：

- 算法语义
- spatial/dataflow 程序结构
- TT target 资源与 ABI

这已经被多类 family 共同暴露出来：

- `flash-attn / online softmax / attention_sink`
- `topk / selection`
- `fusedmoe / grouped dispatch`
- `paged decode / sparse decode`
- `chunk recurrence / scan`

因此当前总体结论只有一条：

```text
Stateful Semantic IR
  -> Spatial Program IR
  -> TT Target IR
```

每层只承接自己的语义真相。
下层只能消费上层冻结后的事实，不能反向猜测上层。

## 2. 目标与硬约束

### 2.1 目标

1. 保持 TileLang Python DSL 主体写法基本稳定
2. 结束 late target-specific semantic guessing
3. 让 domain、state、update、task、layout、sync、TT resource、ABI
   各自在正确层里成为一等对象
4. 让 codegen/runtime 回到 materialization 与 execution，
   不再承担语义恢复
5. 让主链同时面向 attention、selection、routing、paged decode、
   chunk recurrence 等 family

### 2.2 当前硬约束

- `BlackholeModule` 进程内 direct host path 仍是唯一正式执行路径
- `ExecutableSpec` 仍是 runtime 消费的最终物化产物
- copy / GEMM / export 当前支持面不能回退
- 不引入第二条正式执行路径
- 不允许名字匹配、位置猜测、单 case matcher 进入长期协议
- 当前重设计必须建立在现有 Blackhole 主链上完成，不是 greenfield compiler

## 3. 权威架构

### 3.1 三层分工

- `Stateful Semantic IR`
  - 只回答：程序在逻辑域上如何更新算法状态
  - 真源：算法语义
  - 稳态产物：`SemanticProgram`
- `Spatial Program IR`
  - 只回答：这个算法如何组织成 virtual spatial/dataflow program
  - 真源：task/channel/layout/sync/work truth
    与 abstract hardware capability constraints
  - 稳态产物：`SpatialProgram`
- `TT Target IR`
  - 只回答：这个 spatial program 如何变成合法 TT contract
  - 真源：TT resource 与 ABI contract
  - 稳态产物：`TTProgram`
- `ExecutableSpec / runtime`
  - 只回答：冻结后的 TT contract 如何被物化并执行
  - 真源：materialized target schema
  - 稳态产物：`ExecutableSpec` 与 host-side objects

### 3.2 长期真源规则

1. 算法语义只存在于 `Stateful Semantic IR`
2. 空间组织只存在于 `Spatial Program IR`
3. TT 资源与 ABI 只存在于 `TT Target IR`
4. `ExecutableSpec` 只由 `TT Target IR` 物化，不是第二真源

### 3.3 交接纪律

- `Stateful Semantic IR -> Spatial Program IR`
  - 允许：task/channel/layout/sync/work synthesis
  - 禁止：改变语义含义、泄漏 TT noun
- `Spatial Program IR -> TT Target IR`
  - 允许：target mapping、resource planning、ABI 定义
  - 禁止：发明新的 task graph、phase truth、update/access law
- `TT Target IR -> ExecutableSpec / runtime`
  - 允许：materialization 与 launch emission
  - 禁止：semantic recovery、protocol patching、target fallback guessing

### 3.4 明确禁止

- 用 `CB / dst layout / runtime args` 反推 algorithm state semantics
- 用 TT kernel 名字反推 task graph
- 让 runtime/codegen 补缺失的 sync、carry、route 或 ABI contract
- 把 materialized `blackhole.*` attrs 当成与 typed IR 并列的第二真源
- 因为 backend 需要某个对象，就把它直接暴露成 Python DSL 表面概念

## 4. 各层当前完成度

### 4.1 `Phase A`

- 已完成
- `AnalyzeSemanticStructure` 已采用 manifest-first 消费
- `fragment_regions` 只剩 residual reduction evidence 与 lowering compatibility

### 4.2 `Phase B`

- 当前仍在进行中
- 已完成子阶段：
  - boundary cleanup
  - capability intake
  - read-only translator demand probe 对接所需的最小 contract hardening
- 仍未完成：
  - task formation
  - flow shaping
  - domain realization
  - phase / ordering synthesis
  - stronger execution-bearing contract 与 validator

结论：

- `SpatialProgram` 已成为当前唯一 spatial 主链
- 但 `Phase B` 整体仍未完成

### 4.3 `Phase C`

- 已定义；当前只有准备轨已落地
- 已完成：
  - `TTHardwareModel` intake
  - `LowerSpatialProgramToTTTargetProbe`
- 未开始：
  - 正式 `TTProgram / MaterializeTTExecutableSpec` cutover
  - 旧 planning 主链删除

结论：

- `Phase C` 当前不能写成“已开始正式 cutover”
- 当前主 blocker 仍先落在剩余 `Phase B`

## 5. 当前主 blocker

当前总体 blocker 只有两件事：

1. `SpatialProgram` 还只达到 probe intake 的最小上游 contract，
   还不是完整的 execution-bearing virtual spatial program
2. `TTProgram / MaterializeTTExecutableSpec` 还不存在，
   target/runtime 仍主要停留在旧 planning 主链

这也是当前 `blackhole.acc` correctness payoff 还没有完全兑现的根因。

## 6. 当前稳定基线

- `ExecutableSpec -> rt_mod_blackhole -> BlackholeModule`
  仍是唯一正式 direct host path
- copy / GEMM 当前支持面保持不回退
- `tilelang.compile(..., execution_backend="tvm_ffi")`
  的 Blackhole wrapper/export path 已恢复
- `flash-attn` forward subset 已打通当前支持的 compile-path，
  但其 `blackhole.acc` correctness payoff 归属 `Phase C2`

## 7. 当前文档分工

- `final_blackhole_backend_redesign.md`
  - 唯一总体设计
  - 只保留长期架构、层间边界、真源规则、阶段判断
- `stage4_phase_b_spatial_ir.md`
  - `Spatial Program IR` 的当前主实施文档
- `stage4_phase_c_tt_target_ir.md`
  - `TT Target IR` 的当前设计与 cutover 文档
- `tasks/progress.md`
  - 当前执行状态、验证摘要、下一步

阶段细节、完成条件和基线命令默认下沉到对应阶段文档，
总纲不再重复维护 backlog 级别的文件清单或子阶段脚本。
