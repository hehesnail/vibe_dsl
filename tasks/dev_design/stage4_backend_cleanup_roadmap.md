# Blackhole 后端清理与收敛路线图

## 基本信息

- **文档ID**: `stage4_backend_cleanup_roadmap`
- **日期**: 2026-03-30
- **状态**: 活动文档
- **目的**: 把当前 review 暴露出的设计问题、实现问题、hack 边界和重文件风险，整理成可执行的收敛顺序

## 1. 目标

当前 Blackhole 后端的主链已经成立：

- `LowerBlackholeOps -> PlanBlackholeCB -> AssignBlackholeCores -> rt_mod_blackhole -> BlackholeModule`
- current copy/GEMM formal surface 上的 P3 已完成

当前工作的重点不再是“补一条能跑通的路径”，而是：

1. 收掉会污染后续 P4/P5 的 hack 边界
2. 把 schema/spec/runtime 的职责边界再收紧
3. 给后续 copy/dataflow 泛化与 synchronization 深化留出真正可扩的实现骨架

## 2. 范围与非目标

### 范围

- host runtime / planner / lowering / spec extraction 四层的设计与实现问题
- 当前明显的 case-shaped 假设
- 当前最重、最脆、最容易继续变脏的文件
- 按优先级排序的收敛任务

### 非目标

- 不是重新设计总体架构；总体架构继续以 `final_blackhole_backend_redesign.md` 为准
- 不是立即实现 P4/P5 全部能力
- 不是为了“代码看起来更优雅”而大拆已经稳定工作的主链

## 3. 系统性问题清单

### 3.1 边界仍有隐式补洞

当前最典型的问题是 runtime 在 planner/schema 缺口出现时仍会做隐式补洞，而不是在边界处显式失败。

代表例子：

- [blackhole_module.cc](/root/dev/vibe_dsl/tilelang_repo/src/target/blackhole_module.cc) 在 `work_items` 为空时仍会发明 fallback core

风险：

- 掩盖 planner/runtime contract break
- 把真正的协议错误延后到执行期甚至数值错误期才暴露
- 后续 P4/P5 扩展时会继续诱发“host runtime 帮前后端兜底”的坏模式

### 3.2 schema 已 formalize，但 host materialization 仍然过重

`ExecutableSpec` / `KernelSpec` / runtime/common-runtime/accessor schema 已经明显收正，但 host runtime 仍承担大量协议解释。

代表例子：

- [blackhole_module.cc](/root/dev/vibe_dsl/tilelang_repo/src/target/blackhole_module.cc) 里同时承担：
  - buffer materialization
  - runtime args/common runtime args materialization
  - accessor compile-time arg materialization
  - work/core launch 落地

风险：

- 每扩 execution surface 都要继续改 `blackhole_module.cc`
- host runtime 会逐渐演化成“大型协议解释器”
- schema 与 materialization 的边界继续被磨平

### 3.3 部分 identity 仍然靠 heuristic，而不是 schema 真源

当前最明显的是 runtime/common-runtime args 的 identity 仍不够强。

代表例子：

- [rt_mod_blackhole.cc](/root/dev/vibe_dsl/tilelang_repo/src/target/rt_mod_blackhole.cc) 仍按 `kind + name/buffer_name` 做聚合/去重

风险：

- richer runtime work / accessor surface 扩大后，语义不同的参数可能被误合并
- `rt_mod_blackhole` 容易从 spec 提取层滑向“语义修正层”

### 3.4 lowering 里仍混有当前 case 的实现策略

`LowerBlackholeOps` 既承担协议提取，也承担部分当前 copy/GEMM 形态的具体策略派生。

代表例子：

- [lower_blackhole_ops.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.cc) 的 interleaved stick/page 路径仍带有 rank、静态宽度、整除和对齐等强约束

风险：

- 通用协议逻辑和当前阶段策略逻辑继续耦合
- 后续 P4 扩 execution surface 时更难拆边界

### 3.5 planner 仍偏 MVP/heuristic

`PlanBlackholeCB` 现在足够支撑主链，但它还不是一个真正的正式 memory planner。

代表例子：

- [plan_blackhole_cb.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/plan_blackhole_cb.cc) 仍存在明显的 heuristic/inference 风格

风险：

- 当前“够用”的行为继续沉淀成协议事实
- P4/P5 往前推进时，planner 可能成为新的补洞中心

### 3.6 底层 codegen 还没为更宽 surface 预备好

当前对 richer accessor execution surface 的处理，本质上还是“显式不支持”，不是“已准备好只待打开”。

代表例子：

- [codegen_blackhole.cc](/root/dev/vibe_dsl/tilelang_repo/src/target/codegen_blackhole.cc) 仍是 `TensorAccessorArgs<CTA>()` compile-time-only 发射

风险：

- 如果上层 scope 管控松动，很容易误判底层 readiness

## 4. 当前最重、最危险的文件

### 4.1 第一优先级风险文件

- [blackhole_module.cc](/root/dev/vibe_dsl/tilelang_repo/src/target/blackhole_module.cc)

问题：

- 文件职责过多
- 承担了大量 host-side 协议解释
- 已出现 fallback / 固定 DRAM replicated buffer / 集中式 arg materialization 等硬编码假设

结论：

- 这是后续最需要控制复杂度的文件
- 后面如果继续扩 execution surface，不拆 helper 边界几乎一定继续变胖

### 4.2 第二优先级风险文件

- [lower_blackhole_ops.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.cc)

问题：

- 协议提取和当前策略派生混在一起
- 当前 copy stick/page 边界虽已文档化，但实现形态仍容易被误当成“通用 lowering”

结论：

- 需要逐步把“协议真源”和“当前执行策略”拆清

### 4.3 第三优先级风险文件

- [rt_mod_blackhole.cc](/root/dev/vibe_dsl/tilelang_repo/src/target/rt_mod_blackhole.cc)

问题：

- 当前还算可控，但已出现基于字段组合的聚合/去重 heuristics

结论：

- 需要把它稳住在“schema/spec 映射层”
- 不应继续承接隐式语义修正

### 4.4 结构性风险文件

- [plan_blackhole_cb.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/plan_blackhole_cb.cc)
- [codegen_blackhole.cc](/root/dev/vibe_dsl/tilelang_repo/src/target/codegen_blackhole.cc)

结论：

- 前者风险在于 planner 职责已重要，但实现仍偏 MVP
- 后者风险在于 richer accessor/runtime surface 的底层 readiness 不足

## 5. 优先级分档

### A. 短期必须收

这些任务不要求先等 P4/P5 完成；相反，它们应该优先于继续大幅扩 surface。

#### A1. 去掉 invented fallback core

目标：

- planner/runtime contract 缺失时显式失败

主要文件：

- [blackhole_module.cc](/root/dev/vibe_dsl/tilelang_repo/src/target/blackhole_module.cc)
- [assign_blackhole_cores.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/assign_blackhole_cores.cc)
- 对应测试：[test_blackhole_copy_runtime.py](/root/dev/vibe_dsl/tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py)、[test_blackhole_gemm.py](/root/dev/vibe_dsl/tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py)

要求：

- `work_items` 为空时不再发明默认 core
- 错误信息要明确指向 planner/runtime contract
- 新增负向回归，验证 empty work plan 明确失败

#### A2. 把 buffer materialization 从固定 replicated DRAM 抽成 schema-driven 骨架

目标：

- 不再把“当前是 replicated DRAM”硬编码成 runtime 默认事实

主要文件：

- [blackhole_module.cc](/root/dev/vibe_dsl/tilelang_repo/src/target/blackhole_module.cc)
- [blackhole_module.h](/root/dev/vibe_dsl/tilelang_repo/src/target/blackhole_module.h)
- [rt_mod_blackhole.cc](/root/dev/vibe_dsl/tilelang_repo/src/target/rt_mod_blackhole.cc)
- 相关设计：[stage2d_ttmetal_contract_audit.md](/root/dev/vibe_dsl/tasks/dev_design/stage2d_ttmetal_contract_audit.md)

要求：

- 明确区分“当前 formal direct-path 仅 materialize replicated DRAM”与“schema 可以表达的 buffer shape”
- host runtime 先引入 buffer materialization helper / descriptor 边界
- 当前不必一次性支持 sharded/non-DRAM，但要去掉把当前实现当协议默认值的写法

#### A3. 为 runtime/common-runtime arg 引入稳定 identity

目标：

- 去掉 `kind + name/buffer_name` 主导的 heuristic 去重

主要文件：

- [rt_mod_blackhole.cc](/root/dev/vibe_dsl/tilelang_repo/src/target/rt_mod_blackhole.cc)
- [blackhole_module.cc](/root/dev/vibe_dsl/tilelang_repo/src/target/blackhole_module.cc)
- [lower_blackhole_ops.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.cc)

要求：

- 由 schema/spec 真源提供稳定 identity
- runtime/common-runtime arg 聚合只消费 identity，不再自作主张推断等价关系
- 补结构测试覆盖 identity 稳定性

### B. P4 前应收

这些任务不一定立刻阻塞当前 formal surface，但如果不处理，P4 很容易继续靠补丁前进。

#### B1. 拆 `BlackholeModule` 的 host materialization helper 边界

目标：

- 让主执行流只负责 orchestration，不再直接承载全部 materialization 细节

主要文件：

- [blackhole_module.cc](/root/dev/vibe_dsl/tilelang_repo/src/target/blackhole_module.cc)
- [blackhole_module.h](/root/dev/vibe_dsl/tilelang_repo/src/target/blackhole_module.h)

建议拆分方向：

- buffer materialization
- runtime/common-runtime arg materialization
- accessor compile-time arg materialization
- launch/work item application

要求：

- 先拆 helper，不强行大规模重构执行流程
- 行为保持与当前 formal surface 一致

#### B2. 清理 `LowerBlackholeOps` 中协议提取与当前策略派生的边界

目标：

- 降低 copy/dataflow 泛化时继续积累 case-specific lowering 的风险

主要文件：

- [lower_blackhole_ops.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.cc)
- [lower_blackhole_ops.h](/root/dev/vibe_dsl/tilelang_repo/src/transform/lower_blackhole_ops.h)
- 相关设计：[stage4_copy_stick_generalization.md](/root/dev/vibe_dsl/tasks/dev_design/stage4_copy_stick_generalization.md)

要求：

- 识别哪些字段属于正式 schema 真源
- 识别哪些约束只是当前 stick/page direct-path boundary
- 对 boundary 做统一 fail-fast，不要在 lowering 深处散落特判

#### B3. 明确 `PlanBlackholeCB` 的后续定位

目标：

- 决定它是继续演化成正式 planner，还是长期保持 MVP allocator + 更窄职责

主要文件：

- [plan_blackhole_cb.cc](/root/dev/vibe_dsl/tilelang_repo/src/transform/plan_blackhole_cb.cc)
- [stage2d_ttmetal_contract_audit.md](/root/dev/vibe_dsl/tasks/dev_design/stage2d_ttmetal_contract_audit.md)
- [final_blackhole_backend_redesign.md](/root/dev/vibe_dsl/tasks/dev_design/final_blackhole_backend_redesign.md)

要求：

- 明确 planner 的输入、输出、允许推断的范围
- 不要让 heuristic 长期伪装成正式协议

### C. P5 前应收

这些任务和 synchronization 深化关系更大，可以排在 P4 的主扩展之后，但不应无限拖延。

#### C1. 明确 richer accessor/codegen readiness 边界

目标：

- 把当前“compile-time-only accessor codegen”写成正式限制，而不是模糊状态

主要文件：

- [codegen_blackhole.cc](/root/dev/vibe_dsl/tilelang_repo/src/target/codegen_blackhole.cc)
- [stage2h_accessor_schema.md](/root/dev/vibe_dsl/tasks/dev_design/stage2h_accessor_schema.md)
- [stage2i_compile_time_abi_schema.md](/root/dev/vibe_dsl/tasks/dev_design/stage2i_compile_time_abi_schema.md)

要求：

- 明确 codegen 目前支持的 accessor ABI
- richer accessor/CRTA 路径要么进入正式设计，要么保持 fail-fast

#### C2. 把 synchronization 扩展前的 host/runtime 边界收紧

目标：

- 避免 semaphore/global sync 继续堆在 `BlackholeModule` 里做 ad-hoc materialization

主要文件：

- [blackhole_module.cc](/root/dev/vibe_dsl/tilelang_repo/src/target/blackhole_module.cc)
- [stage4_semaphore_schema.md](/root/dev/vibe_dsl/tasks/dev_design/stage4_semaphore_schema.md)

要求：

- semaphore/binding/work application 的 host materialization 保持 schema-driven
- 为 multicast / global semaphore / pass-level producer 预留更清晰的接点

## 6. 建议执行顺序

建议按下面顺序推进，而不是并行发散：

1. `A1` 去掉 fallback core
2. `A2` 建立 buffer materialization 边界
3. `A3` 收正 runtime/common-runtime arg identity
4. `B1` 拆 `BlackholeModule` helper
5. `B2` 清理 `LowerBlackholeOps` 边界
6. `B3` 明确 `PlanBlackholeCB` 定位
7. 进入 P4 主体扩展
8. `C1/C2` 与 P5 深化协同推进

原因：

- `A1-A3` 是当前最像“协议还在被 host runtime 补洞”的问题
- `B1-B3` 是继续做 P4 前最应该收的结构债
- `C1-C2` 更贴近 synchronization 深化，不必抢在前面，但必须有计划

## 7. 验证口径

本路线图对应的各项任务，在实施时至少应满足以下验证要求：

1. 结构验证
   - `ExecutableSpec` / attrs / runtime materialization 的 schema 对齐测试
   - 对 unsupported surface 的显式 fail-fast 回归

2. 行为验证
   - copy runtime / copy pipeline / GEMM 的最小回归
   - 针对新增边界的负向测试

3. 构建验证
   - `cmake --build tilelang_repo/build -j32`

4. 文档验证
   - `final_blackhole_backend_redesign.md`
   - `tasks/progress.md`
   - 对应专项设计文档

## 8. 当前结论

当前 Blackhole 后端最大的问题不是“主链没成型”，而是：

- 主链已经成型
- formal surface 已比之前清楚很多
- 但若干关键边界仍在靠 hack、heuristic 和当前 case 假设支撑

因此接下来的正确方向不是重新发明一条路径，而是沿当前主链继续收正：

1. 去掉 runtime 的隐式补洞
2. 让 schema/spec 真正主导 host materialization
3. 控制重文件继续膨胀
4. 为 P4/P5 留出能扩的骨架，而不是继续堆 workaround
