# 问题与 Bug 记录

> 本文档只保留仍然有复用价值的问题记录。已解决且无复用价值的条目已归档删除。

## 未解决

### Blackhole direct path 缺少 TT-Metal 正式 contract 分层

- **时间**: 2026-03-26
- **问题**: 当前 Blackhole schema 停留在最小 bring-up 级别，缺少：host logical tensor layout 分层、tensor dtype / CB packed dtype / accumulator dtype 分层、accessor schema、rich work description
- **影响**: copy 在最简单 tile/interleaved case 上可通过，但更复杂场景无正式 schema 承载
- **解决方向**: 按 `stage2d_ttmetal_contract_audit.md` 的 P0-P5 分层推进，当前做 P0+P1
- **当前状态**: 未解决。属于协议质量问题，不直接影响当前 GEMM basic 数值正确性。

## 已解决（仍有复用价值）

### `fused_dataflow` 单段 runtime_args / KernelSpec 错位会让 direct runtime 静默读错或拿不到参数

- **时间**: 2026-03-26
- **问题**: copy `fused_dataflow` 从 scratch fallback 切回 codegen 主路径后，编译侧最初报 `Missing runtime arg binding for buffer var: A`；修掉后 direct runtime 又出现 `kernel[0] count=0`，结果全零
- **根本原因**:
  - `blackhole.segment_plan` 的单段 `fused_dataflow` 没有自己携带 `runtime_args`
  - `MakeSegmentPrimFunc` / `PopulateKernelSpecsForDeviceFunc` 之前直接使用空的 `segment.runtime_args`
  - 结果是：
    - codegen 侧丢失原函数上的 `blackhole.runtime_args`
    - runtime 侧 `KernelSpec.runtime_args` 也变成空数组
- **解决**:
  - 单段 `fused_dataflow` segment 必须继承原函数上的 `blackhole.runtime_args`
  - `KernelSpec.runtime_args` 也必须回退到 `ExecutableSpec.runtime_args`
  - codegen 对 copy builtins 的 buffer 绑定在按名字恢复失败时，按 `input_buffer_addr*` / `output_buffer_addr*` 角色回退
- **教训**:
  - segment source 和 runtime launch schema 必须同时继承，不然会出现“源码有 arg load / launch 没下发”或反过来的错位
  - 对单段 `fused_dataflow`，不能默认 `segment.runtime_args` 一定存在
  - `scratch` fallback 一旦删除，就会立刻暴露 schema 继承问题，这正说明 fallback 之前在掩盖主路径错位

### GEMM direct-path 数值错误由 `transpose_B` 丢失和 host row-major upload 引起

- **时间**: 2026-03-26
- **问题**: `test_blackhole_gemm_basic` direct execution 能完成，但结果明显错误（最初观察到 `max_diff=37.24`，复现时可达 `59.53`）
- **错误假设（已排除）**:
  - 不是 `PrintReadTileToCB` / `PrintWriteTileFromCB` 丢了 CB 同步原语
  - 实际检查 lowered TIR 可见 reader/writer 周围已经有：
    - `cb_reserve_back/cb_push_back`
    - `cb_wait_front/cb_pop_front`
- **根本原因**:
  - `LowerBlackholeOps::ExtractGemmInfo` 之前没有把 `transpose_B` 语义正式带到 runtime/spec
  - `BlackholeModule` 直接把 host row-major tensor 原样 memcpy 到 DRAM buffer
  - 但 TT-Metal matmul reader/compute path 期待的是：
    - B 已按 `transpose_B` 语义变成 `K x N`
    - A/B 已做 `tilize_nfaces`
    - C readback 后做 `untilize_nfaces`
- **解决**:
  - 新增 `blackhole.gemm_contract`
  - `rt_mod_blackhole` 将该 contract 进入 `ExecutableSpec`
  - `BlackholeModule` 在 direct path 下：
    - A: row-major → tilize
    - B: row-major `N x K` → transpose → tilize
    - C: tiled output → untilize → row-major tensor
- **教训**:
  - copy 路径通过只说明“字节搬运”没问题，不能证明 GEMM contract 正确
  - 对 TT-Metal matmul，host layout conversion 和 transpose 语义是 correctness contract，不是后续优化
  - 当 generated kernel source 看起来“差不多”时，仍必须继续追到 host upload / readback 层，不能过早停在 codegen 直觉上

### CB identity 唯一真源问题

- **时间**: 2026-03-25
- **问题**: reader/compute/writer 三段没有在同一个 CB identity 上同步
- **根本原因**: `LowerBlackholeOps` 同时产出局部 CB id 和 placeholder id；`PlanBlackholeCB` 允许重名 requirement；codegen 按名字恢复 binding 时取到不同 CB
- **解决**: `LowerBlackholeOps` 统一写 `requirement_index` → `PlanBlackholeCB` 回写 IR → codegen 直接读最终 `cb_id`
- **教训**: planner 的 identity（requirement_index）和 lifetime（lifetime_begin/end）必须分开建模

### ODR/ABI 错位导致随机崩溃

- **时间**: 2026-03-19
- **问题**: 给 `CBRequirement` 新增字段后，`PlanBlackholeCB` 随机以字符串拷贝/vector 排序崩溃
- **根本原因**: 两个头文件在同一 namespace 重复定义 `CBRequirement`，只更新一份导致对象布局不一致
- **教训**: 共享 protocol struct 必须集中到单一定义

### TVM `RemapBufferData` 破坏下游去重

- **时间**: 2026-03-25
- **问题**: canonicalization 后同一 buffer 经两次 `GetNewBuffer` 变成两个不同对象，`buffer_to_cb_` 查不到已分配 id
- **解决**: 在 `GetNewBuffer` 内缓存结果，相同原始 BufferNode 返回同一 Buffer 对象

### TVM `CopyOnWrite()` 对临时 ObjectRef 产生 dangling pointer

- **时间**: 2026-03-25
- **问题**: 对 `Downcast<BufferLoad>(base).CopyOnWrite()` 中的临时 ObjectRef 调用 COW，析构后指针悬空
- **解决**: 不对临时 ObjectRef 调用 CopyOnWrite，改为直接构造返回值

### JIT 缓存串扰

- **时间**: 2026-03-23
- **问题**: 同一 pytest 进程内多个 direct-call case 复用固定 kernel 临时路径，TT-Metal JIT 复用旧编译结果
- **解决**: kernel 临时目录改成每次执行唯一

### 环境问题速查

| 问题 | 解决 |
|------|------|
| `pip install -e .` 失败 | 用 `.pth` 文件指向本地构建产物 |
| `inspect.getsourcelines()` 在内联 `python -c` 中失败 | 写入 `.py` 文件再执行 |
| TT-Metal 示例在 TT-Sim 下报 `Root Directory is not set` | 设置 `TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME` |
| Python 加载旧 `build/` 库 | 统一使用 `tilelang_repo/build/` 单一构建目录 |
| TT-Sim 初始化报 `No chips detected` | 设置 `TT_METAL_SIMULATOR` 和 `TT_METAL_MOCK_CLUSTER_DESC_PATH` |
