# Blackhole `tvm_ffi` Wrapper/Export Blocker 设计

## 基本信息

- **文档ID**: `2026-03-26-blackhole-tvm-ffi-wrapper-export-blocker`
- **日期**: 2026-03-26
- **状态**: 已实施
- **关联文档**:
  - `tasks/dev_design/final_blackhole_backend_redesign.md`
  - `tasks/progress.md`
  - `memory/bugs.md`

---

## 1. 目标

修正 `tilelang.compile(..., target="blackhole", execution_backend="tvm_ffi")` 的 Blackhole wrapper/export 主链，使其生成合法的 host shim 并完成后续编译，不再出现 `lib0.c` 中 `int32_t kernel_error_code = ;` 这类非法代码。

本设计只处理 **wrapper/export blocker**，不把 TT-Metal contract formalization（P0/P3/P4/P5）并入本轮。

---

## 2. 问题描述

当前 Blackhole formal direct host path 已完成，且不依赖 `tvm_ffi` wrapper/export 主链。

但独立地，走：

```python
tilelang.compile(..., target="blackhole", execution_backend="tvm_ffi")
```

时，host shim 导出路径会生成非法 C 代码。当前已知现象是：

- 生成的 `lib0.c` 中存在 `int32_t kernel_error_code = ;`
- 因为 host shim 本身不合法，后续编译失败
- 最小 single-core copy probe 也可复现，说明这不是 multicore/GEMM 专属问题

---

## 3. 根因判断（已验证）

最终验证结果不是 `SplitHostDevice` 的 return contract 错位，而是 host C codegen 对 packed call 结果表达式支持不完整：

- host TIR 中存在合法的 `LetStmt(x, call_packed(...), ...)`
- `CodeGenCHost` 对 `tvm_call_packed_lowered` / `tvm_call_cpacked_lowered` 只发出了调用语句
- 但没有把 `TVMFFIAny result` 重新打印成表达式值
- 因此最终 `lib0.c` 会退化成 `int32_t kernel_error_code = ;`

因此本轮第一性原理结论是：

> 问题不在 `BlackholeModule` direct path，也不在 Blackhole 专用 wrapper 旁路；真正断点在 host C codegen 没有完整承载 host TIR 中“packed call 作为表达式”的语义。

---

## 4. 设计原则

1. **修主链，不做 Blackhole 特判式 workaround**
   - 不在 export/codegen 末端补字符串
   - 不新增 Blackhole 专用 wrapper/export 旁路

2. **从 host/device ABI 一致性修正**
   - 只有当 split 后 device func 的返回约定真正成立时，host shim 才能生成 error-propagation 检查
   - 若该返回约定不成立，就应统一走合法的 void-call host shim

3. **不影响 formal direct host path**
   - 本轮修复只针对 `tvm_ffi` wrapper/export blocker
   - 不改 `ExecutableSpec -> BlackholeModule` direct execution 协议

---

## 5. 影响范围

### 主要改动点

- `tilelang_repo/src/target/codegen_c_host.cc`

### 可能联动检查点

- `tilelang_repo/src/transform/make_packed_api.cc`
- Blackhole target 在 split 后进入 host shim/export 的调用链
- `tvm_ffi` host wrapper 生成与导出路径

### 明确不在本轮范围

- `tilelang_repo/src/target/blackhole_module.cc`
- `ExecutableSpec` schema 扩展
- accessor/work schema/P0 dtype formalization
- 新增 legacy runner 或第二套执行路径

---

## 6. 实施方案

### Step 1: 建立最小复现

用最小 Blackhole compile/export case 复现当前 `lib0.c` 非法 host shim，确认：

- 复现不依赖 multicore
- 复现不依赖 GEMM
- 失败点确实在 host shim 编译阶段，而不是更早的 lowering/codegen

### Step 2: 检查 host TIR 与最终 host C 文本的一致性

围绕 host packed call 表达式确认：

- host TIR 中 `call_packed` 是否被合法地用于 `LetStmt`
- host C codegen 是否既发出调用语句，也重新打印结果表达式
- 是否存在“调用被发出，但表达式值为空”的情况

### Step 3: 在 host C codegen 主链补齐结果表达式打印

修正原则：

- 不新增 Blackhole 特判
- packed call 作为语句和作为表达式两种形态都要正确支持
- 非 `void` 返回值必须从 `TVMFFIAny result` 显式取回并按目标 dtype cast

### Step 4: 验证 compile/export 主链

至少验证：

- 最小 Blackhole `tvm_ffi` compile/export case 成功生成并编译 host shim
- 不再出现 `kernel_error_code = ;`
- formal direct host path 现有测试不回归

---

## 7. 验证方式

### 必做验证

1. 最小 `tilelang.compile(..., target="blackhole", execution_backend="tvm_ffi")` 复现脚本
2. 检查导出的 `lib0.c`，确认 host shim 合法
3. 跑与本改动直接相关的编译链验证，确认 wrapper/export 通过

### 回归验证

- `tilelang_repo/testing/python/target/blackhole/` 中 direct path 关键测试至少抽样回归：
  - `test_blackhole_copy_runtime.py`
  - `test_blackhole_gemm.py`

---

## 8. 风险与边界

### 风险

- `SplitHostDevice` 是通用主链 pass，修得过宽可能影响其他 target 的 host/device return contract
- 如果问题不只在 split pass，而是 split 之后某个 wrapper/export 层继续错误假设“kernel call 一定有返回值”，则需要把定位继续追到 host wrapper 导出链

### 边界控制

- 先用最小 case 锁定 Blackhole 触发条件
- 修复时坚持“按 contract 生成功能”，不做 target-name 特判
- 若必须扩展判断条件，应以 IR/return type/target capability 为依据，而不是按 kernel 名或 Blackhole 字符串兜底

---

## 9. 完成标准

满足以下条件才算该 blocker 关闭：

1. `tilelang.compile(..., target="blackhole", execution_backend="tvm_ffi")` 的最小复现 case 不再生成非法 host shim
2. `lib0.c` 中不再出现 `kernel_error_code = ;` 或等价非法 host call 代码
3. 修改符合 `SplitHostDevice` 主链语义，不引入 Blackhole 专用旁路
4. direct host path 不回归
