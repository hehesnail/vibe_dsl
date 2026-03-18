# Stage 2 Single-Core GEMM Pass Integration 设计

## 目标

- 将 Stage 2 的重点从“再做一个可跑的专用 gemm emitter”调整为“把 single-core gemm 所需语义前移到 pass 产物中”。
- 在保留 `ExecutableSpec -> BlackholeModule -> runner` 主路径不变的前提下，让 gemm 的 kernel schema、CB 规划依赖和 runtime args 主要由 lowering / segment / attr 产物驱动。
- 避免把 Stage 1 copy 的 runtime 专用补洞模式复制到 gemm，造成“执行链路验证通过，但编译 pass 仍未真正打通”的假完成。

## 影响范围

- `tilelang_repo/src/transform/lower_blackhole_ops.cc`
- `tilelang_repo/src/transform/plan_blackhole_cb.cc`
- `tilelang_repo/src/transform/assign_blackhole_cores.cc`
- `tilelang_repo/src/target/rt_mod_blackhole.cc`
- `tilelang_repo/src/target/codegen_blackhole.cc`
- `tilelang_repo/src/target/blackhole_module.h`
- `tilelang_repo/tools/blackhole_runner/runner.cpp`
- `tilelang_repo/testing/python/target/blackhole/test_blackhole_e2e.py`
- `tasks/progress.md`
- `tasks/dev_design/final_blackhole_backend_redesign.md`

## 背景与问题

Stage 1 single-core copy 已完成，但它有明确边界：

- copy 走的是受控的最小专用 kernel emitter
- 该路径证明了 `ExecutableSpec -> runner` 主执行模型可行
- 但它并不等价于“通用 lowering / pass 语义已经完全打通”

如果 Stage 2 gemm 继续沿用同样思路，在 `rt_mod_blackhole` 侧按 `target_mode` 做大量语义补完，那么最终只能证明：

- 可以手工或半手工生成一个能跑的 gemm spec
- runner / module 能执行这个 spec

但不能证明：

- `LowerBlackholeOps`
- `PlanBlackholeCB`
- `AssignBlackholeCores`
- `BlackholeSpecBuilder`

已经形成稳定、可扩展、可复用的编译器路径。

因此 Stage 2 必须显式加入“pass integration”目标，而不是只看 gemm 能否执行。

## 核心结论

### 1. Stage 1 copy 特化允许保留，但不允许复制为 Stage 2 gemm 主策略

允许保留：

- `single_core_copy` 的最小 runtime emitter

不允许作为 Stage 2 主策略继续扩展：

- 在 `rt_mod_blackhole` 中基于 `target_mode == single_core_gemm` 推导完整 gemm kernel 语义
- 让 runtime 继续猜 `kernels[].runtime_args`
- 让 gemm 的 reader / compute / writer 拆分长期停留在 runtime 专用逻辑中

### 2. Stage 2 要先做 pass/schema 收口，再做 gemm true E2E

Stage 2 应拆成两个子目标：

- `Stage 2A`: pass/schema 收口
- `Stage 2B`: single-core gemm true E2E

其中 `2B` 的完成必须建立在 `2A` 之上，不能跳过。

## Stage 2A：pass/schema 收口

### 目标

- 让 gemm 所需执行语义从 pass 产物中显式可见，而不是让 runtime 猜。
- 让 `ExecutableSpec` 的关键字段逐步成为 pass 结果的直接映射。

### 要求

#### `LowerBlackholeOps`

至少要负责写出：

- gemm 对应的 `blackhole.target_mode`
- dataflow 级 tile-access builtin
- segment 所需的中间信息
- runtime arg schema 所需的高层槽位信息

这里的重点不是“把所有细节都固化在 lowering”，而是让 gemm 的语义来源发生前移。

#### `PlanBlackholeCB`

至少要负责把 gemm 需要的 CB 角色和约束显式化：

- A/B 输入 CB
- accumulation / intermediate CB
- output CB

Stage 2 不要求一次性做到最终最优规划，但要求 CB 配置不再主要由 runtime 按惯例猜。

#### `AssignBlackholeCores`

在 single-core gemm 阶段可以继续保持单核 host scheduling plan，但要保证：

- `core_plan` 是明确、稳定、可被 runner 直接消费的
- gemm 阶段不再额外把 device-side 物理映射塞回 codegen

#### `BlackholeSpecBuilder` / `rt_mod_blackhole`

职责应调整为：

- 消费 pass attrs / segment 信息
- 组装 `ExecutableSpec`
- 只补充最小、无语义歧义的默认值

不应继续承担：

- 通过 `target_mode` 反推完整 kernel 结构
- 猜完整 runtime arg schema
- 用 runtime 特判替代 pass 产物缺失

## Stage 2B：single-core gemm true E2E

### 完成标准

single-core gemm 的 true E2E 只有在以下条件同时满足时才算完成：

- gemm 的 `kernels[]` 结构主要由 pass / segment 产物驱动
- gemm 的 `runtime_args` schema 主要由 pass 产物驱动
- runner 只消费 schema，不承担 gemm 语义补完
- 在 TT-Sim 或真实设备上执行并与 reference 对齐

### 不算完成的情况

以下情况不能宣称 Stage 2 完成：

- gemm 能跑，但 reader / compute / writer 仍由 runtime 大量特判拼出
- gemm 能跑，但 `runtime_args` 仍主要由 `rt_mod_blackhole` 猜
- gemm 只验证 codegen 输出，不验证真实执行
- gemm 执行通过，但 pass 产物与最终 spec 长期脱节

## 协议变化

### `target_mode`

Stage 2 对 `target_mode` 的使用约束：

- 保留其作为阶段性模式选择字段
- 但它只应用于选择有限、明确的执行模式
- 不应用作“用一个字符串兜底整套 gemm 语义”

### `kernels[].runtime_args`

Stage 2 的要求是：

- copy 路径的最小 runtime arg schema 可以保留
- gemm 路径的 runtime arg schema 必须开始从 pass 产物显式生成
- runtime 只能消费 schema，不应长期拥有 kernel 语义定义权

## 验证方式

### 设计验证

- `final_blackhole_backend_redesign.md`
- `progress.md`
- Stage 2 设计文档

三者对 Stage 2 的目标、边界、验收标准描述一致。

### 协议验证

- pass 产出的 attrs / segment 信息能支撑 `ExecutableSpec` 构造
- `ExecutableSpec` 中的 gemm kernels / CB / runtime args 不再主要来自 runtime 猜测

### 执行验证

- 保留 single-core copy 测试，防止回归
- 新增或改造 single-core gemm true E2E 测试
- direct-call 与 `spec.json -> runner` 两条主路径都应继续保持一致

## 当前边界

当前允许保留的过渡项：

- `single_core_copy` 最小专用 emitter

当前不允许再新增的过渡项：

- 与 copy 对应的 `single_core_gemm` 大块 runtime 语义特化
- 继续扩大 `rt_mod_blackhole` 对 kernel 语义的反推职责

## 结论

Stage 2 的本质不应定义为“把 gemm 跑起来”，而应定义为：

- 在 single-core gemm 上开始把执行语义从 runtime 迁回 pass
- 让 `ExecutableSpec` 真正成为 pass 产物的执行载体
- 在此基础上完成 gemm true E2E
