# Stage 2 Single-Core Pass Integration 设计

## 基本定位

- **状态**: 当前活动阶段设计
- **前置总体设计**: `final_blackhole_backend_redesign.md`
- **前置接入矩阵**: `stage2_pass_reuse_matrix.md`

本文件只描述 Stage 2 的目标、阶段拆分和验收标准，不再重复承载总体架构结论。

## 阶段目标

Stage 2 的正式目标已经收紧为：

- **先把 Blackhole 重新接回 TileLang / TVM 的 PrimFunc/TIR pass 主链**
- **再在这条主链上完成 single-core copy 与 GEMM 的语义集成**
- **最后完成由 pass 主导的 single-core true E2E**

因此 Stage 2 不再等价于“把 gemm 跑起来”，也不再接受：

- copy 长期停留在 runtime 专用 emitter
- gemm 继续靠 runtime 侧特化拼语义
- 绕过 `SplitHostDevice` / `MakePackedAPI` / `LowerDeviceKernelLaunch` 继续推进

## 阶段拆分

### Stage 2A: pass 主链接入收正

目标：

- 恢复 Blackhole 对通用 TIR / host-device / Packed API pass 的复用
- 结束当前 `OptimizeForTarget` early return 的长期结构

任务：

- 按 `stage2_pass_reuse_matrix.md` 逐项收正 pass 接入
- 恢复：
  - `AnnotateDeviceRegions`
  - `SplitHostDevice`
  - `MakePackedAPI`
  - `LowerDeviceKernelLaunch` 或其 Blackhole 分支
- 收正 `rt_mod_blackhole` / `BlackholeModule` 的边界

完成标准：

- Blackhole 主路径重新回到 TIR / host-device / Packed API 主链
- 不再长期依赖“无 `calling_conv` 也可当 device kernel”的路径

### Stage 2B: single-core copy 语义集成

目标：

- 在已收正的 pass 主链上完成 copy 的 Blackhole-aware lowering

任务：

- 在 `LowerTileOp` 中保留 copy 的 Blackhole-preserving 语义
- `LowerBlackholeOps` 从该语义提取：
  - `blackhole.runtime_args`
  - `blackhole.segment_plan`
  - `blackhole.cb_requirements`
- `PlanBlackholeCB` 生成 runtime-ready `blackhole.cb_configs`
- copy 的 spec/codegen 主要由 pass 产物驱动

完成标准：

- copy 的 runtime args / CB / segment / kernel 结构主要来自 pass
- runtime emitter 只允许保留短期回退，不再是主路径

### Stage 2C: single-core GEMM 语义集成

目标：

- 用与 copy 相同的结构接入 GEMM

任务：

- 在 `LowerTileOp` 中保留 GEMM 的 Blackhole-preserving 语义
- `LowerBlackholeOps` 提取 GEMM 的 reader / compute / writer 所需 schema
- 停止扩展 runtime 侧 gemm 特化路径

完成标准：

- GEMM 的关键执行语义主要来自 pass，而不是 runtime/module 特判

### Stage 2D: single-core true E2E

目标：

- copy 与 GEMM 都在 TT-Sim 或真实设备上完成 true E2E

完成标准：

- `spec.json -> runner` 路径通过
- `artifact.codegen_mod["main"](...)` 路径通过
- copy / GEMM 的关键执行语义主要来自 pass 产物

## 当前边界

当前允许保留的过渡项：

- copy 最小专用 emitter 作为短期回退，但它只能按 device-side builtin/schema 回退，不能再靠 `target_mode` 这类模式标签驱动

当前不允许继续扩大的过渡项：

- copy runtime 特化继续扩大为正式主路径
- GEMM 继续复制 copy 阶段的 runtime 特化做法
- `rt_mod_blackhole` 继续承担 kernel 语义恢复、PrimFunc 参数分类和 host/device 语义定义

当前实现备注：

- Blackhole 现在已经恢复 `AnnotateDeviceRegions -> SplitHostDevice -> MakePackedAPI -> LowerDeviceKernelLaunch` 主链。
- `lower()` 产物重新分成 host `main` 和 device `main_kernel`，Blackhole build 入口也已改为消费两者的组合模块。
- 但为了不打断当前 copy 的 staged-copy lowering，`FlattenBuffer` / `VectorizeLoop` / `StorageRewrite` 这类会破坏现有 copy 识别形态的 pass 还未提前恢复到 `LowerBlackholeOps` 之前。
- 因此当前 Stage 2A 的实现状态应理解为：
  - **host/device 主线已恢复**
  - **通用中后段规范化 pass 只恢复到一条受控子集**

## 当前进展

- copy 已开始产出：
  - `blackhole.runtime_args`
  - `blackhole.segment_plan`
  - `blackhole.cb_requirements`
  - `tl.blackhole.read_tile_to_cb / write_tile_from_cb`
- `PlanBlackholeCB` 已能将 copy requirements 落成 `blackhole.cb_configs`
- `CodeGenBlackhole` 已开始消费 copy builtin
- `BlackholeModule` 对外 entry 的参数签名已重新对齐 split 后 device kernel，不再错误沿用 Packed API 的底层 4 参数签名
- copy true E2E 已通过：
  - `spec.json -> runner`
  - `artifact.codegen_mod["main"](...)`
  - TT-Sim 下 `32x32 float16` staged copy 与 PyTorch 参考一致
- 但 Stage 2A 仍未完成，因此当前 copy 语义集成仍只是中间态，不应被视为正式 compiler path

## 验证方式

### 结构验证

- Blackhole target 恢复 pass 主链验证：
  - `AnnotateDeviceRegions`
  - `SplitHostDevice`
  - `MakePackedAPI`
  - `LowerDeviceKernelLaunch`

### copy / GEMM 语义验证

- pass 产出的 attrs / builtin / segment 能支撑 `ExecutableSpec`
- `ExecutableSpec` 中的 kernels / runtime args / CB 不再主要来自 runtime 猜测

### 执行验证

- copy 与 GEMM 都需要覆盖：
  - `spec.json -> runner`
  - `artifact.codegen_mod["main"](...)`
- 环境不满足时应显式 skip，而不是混成编译链失败
