# Stage 2 Single-Core Pass Integration 设计

## 目标

- 将 Stage 2 的重点从“分别做可跑的 copy/gemm 专用 emitter”调整为“把 single-core copy 与 gemm 所需语义统一前移到 pass 产物中”。
- 在保留 `ExecutableSpec -> BlackholeModule -> runner` 主路径不变的前提下，让 copy/gemm 的 kernel schema、CB 规划依赖和 runtime args 主要由 lowering / segment / attr 产物驱动。
- 避免把 Stage 1 copy 的 runtime 专用补洞模式复制到 gemm，同时也避免让 copy 长期停留在 runtime 特化旁路上，造成“执行链路验证通过，但编译 pass 仍未真正打通”的假完成。

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

如果 Stage 2 继续让 copy 停留在 runtime 特化路径、同时对 gemm 也沿用类似思路，在 `rt_mod_blackhole` 侧按 `target_mode` 做大量语义补完，那么最终只能证明：

- 可以手工或半手工生成一个能跑的 copy/gemm spec
- runner / module 能执行这个 spec

但不能证明：

- `LowerBlackholeOps`
- `PlanBlackholeCB`
- `AssignBlackholeCores`
- `BlackholeSpecBuilder`

已经形成稳定、可扩展、可复用的编译器路径。

因此 Stage 2 必须显式加入“single-core pass integration”目标，而不是只看某个算子能否执行。

## 核心结论

### 1. Stage 1 copy 特化允许作为过渡起点保留，但 Stage 2 必须开始把 copy 与 gemm 一起迁回 pass

允许保留：

- `single_core_copy` 的最小 runtime emitter

但 Stage 2 不允许：

- 让 copy 长期停留在 runtime 专用 emitter，完全不回迁
- 在 `rt_mod_blackhole` 中基于 `target_mode == single_core_gemm` 推导完整 gemm kernel 语义
- 让 runtime 继续猜 `kernels[].runtime_args`
- 让 copy/gemm 的 kernel 拆分长期停留在 runtime 专用逻辑中

### 2. Stage 2 要先做 copy pass integration，再做 gemm pass integration，最后统一做 true E2E

Stage 2 应拆成三个子目标：

- `Stage 2A`: single-core copy pass integration
- `Stage 2B`: single-core gemm pass integration
- `Stage 2C`: single-core copy + gemm true E2E

其中后一步都必须建立在前一步之上，不能跳过。

## Stage 2A：single-core copy pass integration

### 目标

- 让 copy 所需执行语义从 pass 产物中显式可见，而不是让 runtime 猜。
- 让 `ExecutableSpec` 的关键字段逐步成为 pass 结果的直接映射。

### 要求

#### `LowerBlackholeOps`

至少要负责写出：

- copy 对应的 `blackhole.target_mode`
- copy 的 dataflow 级 tile-access builtin
- copy segment 所需的中间信息
- copy runtime arg schema 所需的高层槽位信息
- 为后续 gemm 复用同一套 schema 迁移路径

当前阶段对 copy 的直接要求进一步明确为：

- `T.copy` 不能只留下 attrs/schema，然后由 `rt_mod_blackhole` 手写 kernel 主体
- pure copy 必须开始被 lower 成真实的 Blackhole 中层 builtin call
- 即使执行路径仍暂时保留 runtime emitter 回退，copy 语义本身也必须先在 TIR body 中存在

建议中层 builtin 形态：

- `blackhole.read_tile_to_cb(buffer, tile_index, cb_id, tile_bytes, accessor_slot)`
- `blackhole.write_tile_from_cb(cb_id, buffer, tile_index, tile_bytes, accessor_slot)`

这样 Stage 2A 的第一完成标志不是“copy 还能跑”，而是：

- copy 已重新接回 `TIR AST -> pass visitor -> builtin-based TIR` 主链

#### `PlanBlackholeCB`

至少要负责把 copy 需要的 CB 角色和约束显式化：

- input CB
- output CB
- 如需保留 scratch / intermediate，也应以 pass 产物显式表达

要求是 copy 的 CB 配置不再主要由 runtime 按惯例猜。

#### `BlackholeSpecBuilder` / `rt_mod_blackhole`

职责应调整为：

- 消费 copy 的 pass attrs / segment 信息
- 组装 copy 的 `ExecutableSpec`
- 只补充最小、无语义歧义的默认值

不应继续承担：

- 长期保有 copy 完整 kernel 语义定义权
- 猜完整 copy runtime arg schema
- 用 runtime 特判替代 copy pass 产物缺失

## Stage 2B：single-core gemm pass integration

### 目标

- 让 gemm 所需执行语义从 pass 产物中显式可见，而不是让 runtime 猜。
- 在 copy 已开始回迁 pass 的前提下，用同一套 schema/segment 机制承接 gemm。

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

## Stage 2C：single-core copy + gemm true E2E

### 完成标准

Stage 2 的 true E2E 只有在以下条件同时满足时才算完成：

- copy 的 `kernels[]` 结构主要由 pass / segment 产物驱动
- copy 的 `runtime_args` schema 主要由 pass 产物驱动
- gemm 的 `kernels[]` 结构主要由 pass / segment 产物驱动
- gemm 的 `runtime_args` schema 主要由 pass 产物驱动
- runner 只消费 schema，不承担 gemm 语义补完
- copy 与 gemm 都在 TT-Sim 或真实设备上执行并与 reference 对齐

### 不算完成的情况

以下情况不能宣称 Stage 2 完成：

- copy 能跑，但仍主要依赖 runtime 专用 emitter
- gemm 能跑，但 reader / compute / writer 仍由 runtime 大量特判拼出
- gemm 能跑，但 `runtime_args` 仍主要由 `rt_mod_blackhole` 猜
- copy/gemm 只验证 codegen 输出，不验证真实执行
- gemm 执行通过，但 pass 产物与最终 spec 长期脱节

## 协议变化

### `target_mode`

Stage 2 对 `target_mode` 的使用约束：

- 保留其作为阶段性模式选择字段
- 但它只应用于选择有限、明确的执行模式
- 不应用作“用一个字符串兜底整套 copy/gemm 语义”

### `kernels[].runtime_args`

Stage 2 的要求是：

- copy 路径的最小 runtime arg schema 只能作为过渡起点，必须开始从 pass 产物显式生成
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
- `ExecutableSpec` 中的 copy/gemm kernels / CB / runtime args 不再主要来自 runtime 猜测

### 执行验证

- 保留并改造 single-core copy 测试，验证其开始走 pass-driven schema
- 新增或改造 single-core gemm true E2E 测试
- direct-call 与 `spec.json -> runner` 两条主路径都应继续保持一致

## 当前边界

当前允许保留的过渡项：

- `single_core_copy` 最小专用 emitter 作为短期回退路径

当前不允许再新增的过渡项：

- 将 copy 的 runtime 特化继续扩大为长期主路径
- 与 copy 对应的 `single_core_gemm` 大块 runtime 语义特化
- 继续扩大 `rt_mod_blackhole` 对 kernel 语义的反推职责

## 当前进展

- 已完成的 Stage 2A 落地点：
  - `LowerBlackholeOps` 对 pure `global -> global` copy 不再只依赖 runtime 专用猜测
  - pure copy 已能显式写出：
    - `blackhole.target_mode = "single_core_copy"`
    - `blackhole.runtime_args`
    - `blackhole.segment_plan`
    - input/output `blackhole.cb_requirements`
  - pure copy 已开始被 lower 成真实的 Blackhole 中层 builtin call：
    - `tl.blackhole.read_tile_to_cb`
    - `tl.blackhole.write_tile_from_cb`
  - `PlanBlackholeCB` 已能把这些 requirements 落成 input/output `blackhole.cb_configs`
  - `rt_mod_blackhole` 已优先读取 pass 产出的：
    - `blackhole.runtime_args`
    - `blackhole.segment_plan`
- 当前仍未完成：
  - `CodeGenBlackhole` 已开始让上述 copy builtin 成为当前 single-core copy 主执行来源
  - `rt_mod_blackhole` 仍保留最小专用 emitter 作为回退
  - copy 的更真实 tile/dataflow 语义还没有完全从 pass 直达 kernel emission
  - gemm 仍未开始接入同一套 pass-driven schema

## 结论

Stage 2 的本质不应定义为“把 gemm 跑起来”，而应定义为：

- 在 single-core copy 与 gemm 上开始把执行语义从 runtime 迁回 pass
- 让 `ExecutableSpec` 真正成为 pass 产物的执行载体
- 在此基础上完成 copy + gemm 的 single-core true E2E
