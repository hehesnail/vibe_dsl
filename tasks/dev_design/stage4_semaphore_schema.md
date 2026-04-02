# Stage 4 设计：Semaphore Schema 预埋

## 基本信息

- **文档ID**: `stage4_semaphore_schema`
- **日期**: 2026-03-27
- **状态**: 已实现（program-local semaphore schema + kernel binding schema + 最小 dataflow semaphore builtin + worker producer/consumer E2E + remote-core descriptor formalization）
- **对应任务**: P5 multi-core synchronization 预埋
- **关联文档**:
  - `tasks/dev_design/final_blackhole_backend_redesign.md`
  - `tasks/dev_design/stage2d_ttmetal_contract_audit.md`
  - `tasks/dev_design/archive/stage3_multicore_design.md`

> 角色说明：本文档现在只作为 **TT Target IR 中 semaphore 子问题** 的支持设计。总体架构与当前阶段方向以 `final_blackhole_backend_redesign.md` 为准。

---

## 1. 目标

为 Blackhole 主链预埋 TT-Metal program-local semaphore 的正式对象层。

本轮只解决：

1. device attrs 上的正式 `blackhole.semaphore_plan`
2. `ExecutableSpec` / `BlackholeModule` 对 semaphore plan 的提取与 host materialization
3. direct runtime 对 malformed / unsupported semaphore schema 的 fail-fast
4. device-side dataflow kernel 对 program-local semaphore 的最小 builtin 入口
5. worker semaphore 的最小 producer/consumer 执行闭环

当前文档覆盖到第四轮扩展，但仍不解决：

- multicast
- global semaphore
- 从现有 `mbar` / `shared.barrier` 自动推断 TT-Metal semaphore
- pass 自动生成的更宽 producer/consumer pipeline

---

## 2. 设计边界

### 2.1 为什么不把 `mbar` 直接映射成 semaphore

`mbar` 当前在 TileLang IR 里表达的是 barrier binding / wait-arrive 语义，来源于 CUDA / tcgen05 一侧的 barrier 模型。

TT-Metal host API 的 semaphore 是另一类正式对象：

- `CreateSemaphore(program, core_ranges, initial_value)`
- `CreateGlobalSemaphore(device, core_ranges, initial_value, buffer_type)`

二者不是同一层对象，当前没有足够 IR 信息可以无损映射。因此本轮明确不做：

- `mbar -> semaphore` 猜测映射
- `shared.barrier -> semaphore` 猜测映射

如果后续确实需要从 DSL/IR 表达 TT-Metal semaphore，就应扩正式 IR/schema，而不是让 runtime 猜。

### 2.2 过渡策略

本轮的 `blackhole.semaphore_plan` 是正式 schema，但 producer 暂时不接自动 pass。

当前状态：

- consumer 已存在：`rt_mod_blackhole` / `BlackholeModule`
- producer 暂缺：后续由真正的 synchronization pass 产出

退出条件：

- 有 dedicated pass 或 DSL/IR 扩展稳定产出 `blackhole.semaphore_plan`
- direct runtime / codegen 不再需要靠测试注入 attrs 验证

---

## 3. 协议方案

在 device PrimFunc attrs 上新增：

- `blackhole.semaphore_plan`

每个 entry 表达一个 program-local semaphore descriptor，最小字段：

- `id`
- `initial_value`
- `core_type`
- `core_ranges`

其中：

- `id` 对齐 TT-Metal `SemaphoreDescriptor.id`
- `initial_value` 对齐 `CreateSemaphore(..., initial_value, ...)`
- `core_type` 作为 Blackhole schema 校验字段保留；当前 direct runtime 仅正式支持 `worker`，未知值或非 `worker` 值 fail-fast
- `core_ranges` 表达该 semaphore 覆盖的 logical core set

第一轮未引入：

- semaphore address 预分配
- global semaphore buffer type
- remote core / multicast descriptors

第二轮继续补：

- per-kernel `semaphore_bindings`
- runtime arg kind `semaphore_id_u32`

也就是把协议从“program 里有哪些 semaphore”推进到“哪个 kernel 需要哪个 semaphore id，以及 host/runtime 如何把它作为正式 runtime arg 下发给 kernel”。

---

## 4. 实现方案

### 4.1 `ExecutableSpec`

新增：

- `SemaphoreSpec`
- `ExecutableSpec.semaphores`

### 4.2 `rt_mod_blackhole`

- 提取 `blackhole.semaphore_plan`
- 写入 `ExecutableSpec.semaphores`

### 4.3 `BlackholeModule`

- 在 direct runtime 创建 `Program` 后、创建 kernels 前，按 `ExecutableSpec.semaphores` 调用 `CreateSemaphore(...)`
- 当前 direct runtime 只 materialize `worker` semaphore；不试图通过 deprecated TT-Metal API 强行创建其他 core type
- 当前仅要求 host 对象正确 materialize，不要求 kernel 已消费 semaphore id

### 4.4 测试

先写失败测试，再补实现。

覆盖面：

1. spec 测试
   - `blackhole.semaphore_plan` 能进入 `ExecutableSpec.semaphores`
2. direct runtime schema 测试
   - malformed `core_type` / 缺字段会 fail-fast
3. host materialization 测试
   - 当前以 schema/spec + runtime validation 为主

### 4.5 第二轮扩展：kernel-level semaphore binding

新增：

- `SemaphoreBindingSpec`
- `KernelSpec.semaphore_bindings`

每个 binding 最小字段：

- `name`
- `semaphore_id`
- `arg_kind`

第二轮正式支持：

- `arg_kind = "semaphore_id_u32"`

语义：

- program 级 `ExecutableSpec.semaphores` 负责创建 TT-Metal semaphore objects
- kernel 级 `KernelSpec.semaphore_bindings` 负责声明 kernel 需要哪个 planned semaphore
- direct runtime materialize semaphore 后，建立 `planned semaphore id -> created TT-Metal semaphore id` 映射
- `BuildRuntimeArgsFromSpec` 遇到 runtime arg kind `semaphore_id_u32` 时，按 kernel binding 找到对应 semaphore id 并写入最终 runtime args

仍不解决：

- device-side wait/post builtin
- kernel source 对 semaphore 的真实消费语义
- `mbar -> semaphore` 自动绑定

### 4.6 第三轮扩展：device-side semaphore builtin（最小 dataflow 入口）

新增 Blackhole builtin：

- `tl.blackhole.get_semaphore(semaphore_id)`
- `tl.blackhole.semaphore_wait(semaphore_addr, value)`
- `tl.blackhole.semaphore_set(semaphore_addr, value)`

映射原则：

- runtime 继续只下发 `semaphore_id_u32`
- device kernel 显式先调用 `get_semaphore(id)` 拿本地 L1 semaphore 地址
- dataflow codegen 直接打印 TT-Metal `dataflow_api.h` 里的正式原语：
  - `get_semaphore(...)`
  - `noc_semaphore_wait(...)`
  - `noc_semaphore_set(...)`

本轮刻意不做：

- multicast semaphore primitives
- compute kernel 侧 semaphore primitive
- 更宽的 producer/consumer 拓扑（multicast/global）

### 4.7 第四轮扩展：最小 worker producer/consumer E2E

在不引入新 pass 或新执行路径的前提下，用现有 multi-core `fused_dataflow` copy kernel 做最小跨核握手验证。

最终收敛方案不是让 kernel 猜 remote core 坐标，而是把 remote worker descriptor 正式写进 runtime schema：

- runtime arg kind `logical_core_noc_x`
- runtime arg kind `logical_core_noc_y`
- 由 host runtime 用 `device.worker_core_from_logical_core(...)` materialize 成真正的 NOC 坐标
- device TIR 通过 `tl.blackhole.runtime_arg_u32(name)` 读取这些显式 runtime args

producer/consumer 语义采用最小 remote signal：

- `bx == 0` 的 worker 作为 producer
  - 执行原有 tile copy
  - 对 consumer core 上的同一个 program-local semaphore 执行 `noc_semaphore_inc(..., 1)`
- `bx == 1` 的 worker 作为 consumer
  - 先 `semaphore_wait(..., 1)`
  - 再执行自己的 tile copy

约束：

- 继续只使用 `worker` semaphore
- 继续只在 dataflow kernel 上验证
- 不要求从 pass 自动产出 semaphore producer；测试可以通过 body/attrs/runtime-arg 变异构造最小闭环

补充边界（2026-03-31 cleanup）：

- `semaphore_id_u32` 不是“只要 runtime arg 名字对了就能下发”，而是必须在 `KernelSpec.semaphore_bindings` 里有唯一匹配 binding，且 binding 指向 `ExecutableSpec.semaphores` 中已规划的 semaphore
- `logical_core_noc_x/y` 不是两条彼此独立的 runtime arg，而是同一个 remote-core descriptor 的两个分量：
  - 必须共享显式 `identity`
  - 必须成对出现
  - 必须共享同一 logical core 坐标
- 这些约束应在 `ExecutableSpec` / `KernelSpec` 边界完成校验，而不是等到 direct execution 时再由 runtime kind-switch 临时发现

下一步 formalization（已落实到最小主链）：

- 现有 `logical_core_noc_x/y` 已不再只是两条散装 runtime arg
- `KernelSpec` 现在显式携带 `remote_core_descriptors`
- 每个 descriptor 最小表达：
  - `identity`
  - logical `core_x/core_y`
- `logical_core_noc_x/y` runtime arg 继续保留给 device 侧按名字读取，但 host/runtime materialization 现已优先消费 descriptor，而不是继续把每条 arg 上的 `core_x/core_y` 当真源

这个闭环的目的不是扩 execution surface，而是证明：

- host `CreateSemaphore(...)` 物化出来的对象能被跨核 device builtin 真实消费
- direct runtime 的 multi-core launch、semaphore plan、runtime-materialized remote NOC coords、device builtin 四者在同一程序里已经能协同工作
- P5 已从“schema + binding + builtin 预埋”推进到“最小真实握手 E2E”

---

## 5. 验证标准

- `ExecutableSpec` 正式携带 semaphore descriptors
- `BlackholeModule` 能消费 semaphore plan，而不是忽略它
- 未支持的 semaphore schema 会在 direct runtime 早失败
- `KernelSpec` 能正式携带 semaphore binding，并让 direct runtime 把 semaphore id materialize 成 runtime arg
- dataflow codegen 能把最小 semaphore builtin 打印成 TT-Metal 正式 device API，并能从 runtime schema 读取显式 remote NOC 坐标
- TT-Sim 下存在一个真实的 multi-core producer/consumer direct-runtime 用例，验证 `semaphore_wait` / `noc_semaphore_inc` 可以跨 worker core 闭环执行
- 文档与 `tasks/progress.md` 同步
