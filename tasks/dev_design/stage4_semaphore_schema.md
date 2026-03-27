# Stage 4 设计：Semaphore Schema 预埋

## 基本信息

- **文档ID**: `stage4_semaphore_schema`
- **日期**: 2026-03-27
- **状态**: 已实现（program-local semaphore schema + kernel binding schema）
- **对应任务**: P5 multi-core synchronization 预埋
- **关联文档**:
  - `tasks/dev_design/final_blackhole_backend_redesign.md`
  - `tasks/dev_design/stage2d_ttmetal_contract_audit.md`
  - `tasks/dev_design/stage3_multicore_design.md`

---

## 1. 目标

为 Blackhole 主链预埋 TT-Metal program-local semaphore 的正式对象层。

本轮只解决：

1. device attrs 上的正式 `blackhole.semaphore_plan`
2. `ExecutableSpec` / `BlackholeModule` 对 semaphore plan 的提取与 host materialization
3. direct runtime 对 malformed / unsupported semaphore schema 的 fail-fast

本轮不解决：

- multicast
- global semaphore
- remote core coordinate / fabric routing
- 从现有 `mbar` / `shared.barrier` 自动推断 TT-Metal semaphore
- 真正的跨核 producer/consumer pipeline 执行闭环

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

---

## 5. 验证标准

- `ExecutableSpec` 正式携带 semaphore descriptors
- `BlackholeModule` 能消费 semaphore plan，而不是忽略它
- 未支持的 semaphore schema 会在 direct runtime 早失败
- `KernelSpec` 能正式携带 semaphore binding，并让 direct runtime 把 semaphore id materialize 成 runtime arg
- 文档与 `tasks/progress.md` 同步
