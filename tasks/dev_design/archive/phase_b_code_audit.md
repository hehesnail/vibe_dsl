# Phase B Spatial IR Code Audit

- **日期**: 2026-04-08
- **范围**: Stage 4 Phase B 全部实现代码
- **目的**: 从第一性原理审计设计合理性、代码质量、模块化程度
- **结论**: 架构设计成立，实现质量有结构性工程债务

## 审计涉及的文件

### 数据结构
- `src/transform/common/spatial_program.h/cc` — 9 个 IR 类型 + 2 个中间计划 + capability model
- `src/transform/common/spatial_vocab.h/cc` — closed enum vocabulary
- `src/transform/common/spatial_analysis.h/cc` — 共享分析 helper

### Pass 实现
- `src/transform/analyze_spatial_domain_plan.cc` — AnalyzeSpatialDomainPlan
- `src/transform/lower_to_spatial_program.cc` — AnalyzeSpatialExecutionPlan + MaterializeSpatialProgram + LowerToSpatialProgram
- `src/transform/validate_spatial_program.cc` — ValidateSpatialProgram

---

## 1. 大规模代码重复（最严重）

以下函数 / 结构体在 `lower_to_spatial_program.cc`、`validate_spatial_program.cc`、
`spatial_analysis.cc`、`analyze_spatial_domain_plan.cc` 之间存在 2~4 份几乎完全相同的拷贝：

| 重复函数 / 结构体 | 出现位置数 |
|---|---|
| `ProducerVersionEdge` struct | 3 |
| `DomainRealizationContract` / `ExpectedDomainContract` | 3（同逻辑不同名）|
| `BuildStateRoleByName` | 3 |
| `BuildDomainAxisNameSet` | 3 |
| `AccessMapTouchesDomain` | 3 |
| `UpdateTouchesDomain` | 3 |
| `DomainHasStateRole` | 3 |
| `DomainHasAccessTrait` | 3 |
| `HasSupplementPayload` | 3 |
| `DeriveDomainRealizationContract` | 3 |
| `BuildUniqueUpdateResultVersionByUpdate` | 3 |
| `ResolveProducerEdgesForVersion` | 3 |
| `BuildVersionProducerEdges` | 3 |
| `DeriveOrderingKindForChannel` | 3 |
| `DeriveMaterializationKindForChannel` | 3 |
| `ToStringArray` / `MakeTraits` / `HasTrait` | 3 |
| `GetPayloadIndex` / `GetPayloadString` / `GetPayloadIndices` | 3 |
| `SameStringArray` / `SameIntegerAnyArray` | 2 |
| `ContainsKind` / `RequireCapabilitySupport` | 2 |
| `SelectLayoutKind` / `SelectPartitionKind` | 2 |
| `MakeAnchors` / `GetMemberFuncName` | 2 |
| `BuildDomainPayload` / `BuildWorkPartitionPayload` | 2 |

`spatial_analysis.h/cc` 已经把这些函数抽出来了，但 `lower_to_spatial_program.cc`
和 `validate_spatial_program.cc` 并没有使用它——各自在匿名命名空间里又写了一份。

**风险**：如果将来修改 `DeriveDomainRealizationContract` 的逻辑，
需要同时改 3 个文件，少改一处就是 silent divergence。

**修复方向**：所有消费侧统一 `#include "spatial_analysis.h"` 并删除本地拷贝。

---

## 2. 名义上 typed，实际上 stringly-typed

设计文档反复强调 "typed payload / linkage contract"、"用类型系统捕获协议错误"。
但实际 IR 定义（`spatial_program.h`）里：

```cpp
class TaskNode : public Object {
  ffi::String name;
  ffi::String kind;                        // 存的是 enum 的字符串形式
  ffi::String phase_name;                  // 字符串交叉引用
  ffi::Array<ffi::String> traits;
  ffi::Map<ffi::String, ffi::Any> payload; // 无 schema 的 any-map
  ffi::Array<TIRAnchor> anchors;
};
```

- `kind` 是字符串。`spatial_vocab.h` 定义了 enum，但 IR node 不直接使用，
  所有类型检查推迟到运行时 `Parse*Kind()` + `ICHECK`。
- `payload` 是 `Map<String, Any>`——这是一个 attr bag。关键信息如
  `phase_index`、`execution_role`、`formation_basis`、`source_task_index`、
  `target_task_index`、`affinity_kind`、`obligation_kind` 全部塞在这里面，
  既没有编译期类型保证，也没有 schema 定义。
- 结果是 validator 和 downstream consumer 大量
  `GetPayloadString(task->payload, schema_key::kExecutionRole)` + `ICHECK` 模式——
  这是在运行时手动做 schema validation，不是类型系统在捕获协议错误。

**修复方向**：分批把高频消费的 payload field（`phase_index`、`source_task_index`、
`target_task_index`、`execution_role`、`formation_basis`）提升为 IR node 的正式 typed field。
不需要一次做完，可以随 Phase C 消费需求逐步演进。

---

## 3. 双重引用系统

每个 Channel 同时有 `source_task`（string name）和 `payload["source_task_index"]`（int）。
每个 Task 同时有 `phase_name`（string）和 `payload["phase_index"]`（int）。
Placement 同时有 `task_name`（string）和 `payload["task_index"]`（int）。

每个交叉引用存了两份——一份字符串名，一份数值索引——
validator 必须检查两者是否一致。这不是 redundancy for safety，
是 schema 没有选择一种 canonical 引用方式的表现。

**修复方向**：选择 index 为 canonical reference。
name 保留为 display / debug 字段，不再承担 linkage 职责。
validator 只检查 index 的 referential integrity。

---

## 4. Validator 重新推导 builder 的逻辑

`validate_spatial_program.cc` 有一个 `DeriveExpectedDomainContract`（validator 自己的版本），
与 `DeriveDomainRealizationContract`（builder 的版本）做的是同一件事。
validator 还重新跑了 `BuildVersionProducerEdges`、`BuildUniqueUpdateResultVersionByUpdate` 等。

这让 validator 变成了 "用同一个算法重新跑一遍，然后检查结果是否相同"。
如果两份代码有 bug，可能犯同样的错误；如果有 divergence，
validator 成了检测代码不一致的工具，而不是检测 IR 语义正确性的工具。

**修复方向**：
- 让 validator 复用 `spatial_analysis.h/cc` 里的共享函数，
  消除 `ExpectedDomainContract` 这个重复类型
- Validator 的核心职责收窄为 structural / referential integrity 检查

---

## 5. `lower_to_spatial_program.cc` 是 2285 行 monolith

该文件包含 3 个 pass（`AnalyzeSpatialExecutionPlan`、`MaterializeSpatialProgram`、
`LowerToSpatialProgram`）加上完整的 task formation、flow shaping、phase synthesis、
fast path builder、domain realization，以及大量重复的 helper 函数。

**修复方向**：
- 把 `AnalyzeSpatialExecutionPlan` 提取为 `analyze_spatial_execution_plan.cc`
- 把 `MaterializeSpatialProgram` 提取为 `materialize_spatial_program.cc`
- `LowerToSpatialProgram` 保留为 thin wrapper
- CMakeLists.txt 里新文件已被 `src/transform/*.cc` glob 覆盖，无需额外注册

---

## 6. Fast path 不是 generic path 的退化

设计文档说 "copy / GEMM fast path 只作为 execution-bearing SpatialProgram contract 的退化特例"。
但代码中 `BuildCopyFastPath`（~60 行）和 `BuildGemmFastPath`（~140 行）
是完全独立的手写构造逻辑，和 `BuildGenericSpatialProgram`（~270 行）没有任何代码共享。

- Generic path 的 task formation 用 `BuildTaskRecords` + `UpdateGraph` + fusion
- Fast path 直接手写 `Task(String("copy"), ...)`

如果 generic path 的 contract 演进（新增 mandatory payload field），
fast path 可能忘记跟上。

**修复方向**：让 generic builder 在 simple copy / simple GEMM 场景下
自然产出期望的 SpatialProgram，消除并行构造路径。
或至少提取共享的 contract-construction 函数让两条路径都走。

---

## 7. 所有 IR node 结构雷同

`Task`、`Channel`、`SpatialLayout`、`WorkPartition`、`Placement`、
`SyncEdge`、`ResourceIntent` 这 7 个类的字段结构几乎一模一样：

```
name: String
kind: String
<1-2 个 domain-specific string field>
traits: Array<String>
payload: Map<String, Any>
anchors: Array<TIRAnchor>
```

它们之间的区别完全靠 payload 里的 key 来表达。这说明 schema 设计还没有真正分化——
这些不是 7 个不同的类型，而是 1 个 generic spatial object 的 7 种使用方式。

**修复方向**：这是长期演进项。当 Phase C 开始真正消费这些 object 时，
把 domain-specific truth 从 payload 提升到 node 的 typed field，
让各类型之间的差异体现在编译期而不是运行时。

---

## 8. Capability model 重复构造

`AnalyzeSpatialDomainPlan`、`AnalyzeSpatialExecutionPlan`、`MaterializeSpatialProgram`
每个 pass 都独立地从 target 构造 `TTHardwareModel` 和 `SpatialCapabilityModel`。
`MaterializeSpatialProgram` 在函数间缓存了，但 pass 间没有共享。
module-level global info 在 `MaterializeSpatialProgram` 之后才写入，前两个 pass 用不到。

**修复方向**：让 `AnalyzeSpatialDomainPlan` 在构造后
把 capability model 写入 module global info，后续 pass 直接读取。

---

## 值得肯定的地方

1. **Vocabulary 设计**：`spatial_vocab.h` 的 closed enum set 是正确的方向，
   完整覆盖了 task kind、channel kind、layout kind 等。

2. **Task formation 算法**：update-state 图 -> topo sort -> mandatory boundary cut -> fusion
   的逻辑链清晰，`BuildTaskRecords` / `CanFuseTaskFormationCandidates` /
   `HasMandatoryTaskBoundary` 的设计思路正确。

3. **Phase synthesis**：从 ordering-critical channel 构造 phase rank、
   再通过 topo sort 产生 phase order 的算法合理。

4. **验证覆盖面**：validator 检查了大量 referential integrity 约束，
   fail-fast 做得彻底。

5. **测试覆盖**：69 个 spatial IR 测试 + 7 个 family 的 compile-path gate 覆盖面好。

---

## 建议的改进优先级

| 优先级 | 改进项 | 紧迫度 | 风险 |
|---|---|---|---|
| 1 | 消除代码重复（复用 spatial_analysis.h/cc） | 高 | 低 |
| 2 | 拆分 lower_to_spatial_program.cc | 中 | 低 |
| 3 | 统一引用方式（index canonical, name display-only） | 中 | 中 |
| 4 | Capability model 构造缓存 | 低 | 低 |
| 5 | 关键 payload 字段升级为 typed field | 低 | 中（scope 大）|
| 6 | Fast path 退化为 generic path 特例 | 低 | 中 |
| 7 | IR node 结构分化 | 低 | 大（长期演进）|
