# Stage 2F 设计：GEMM dtype 分层正式化

## 基本信息

- **文档ID**: `stage2f_gemm_dtype_layering`
- **日期**: 2026-03-27
- **状态**: 实施中
- **对应任务**: TT-Metal contract formalization 的 P0 子项
- **关联文档**:
  - `tasks/dev_design/final_blackhole_backend_redesign.md`
  - `tasks/dev_design/stage2d_ttmetal_contract_audit.md`
  - `tasks/dev_design/stage3_multicore_design.md`

---

## 1. 目标

把当前 GEMM direct path 中混在一起的 dtype 语义拆成显式 schema，至少分清三层：

1. host-visible tensor dtype
2. CB transport dtype
3. compute accumulator dtype

本轮目标不是一次性做完整 accessor/work schema，而是先把 P0 中最直接影响 ABI 清晰度的 dtype 分层补齐，并让 `ExecutableSpec -> BlackholeModule` 主路径消费显式 contract，而不是继续靠“当前测试只用了 bf16/fp32”这种隐式假设运行。

---

## 2. 当前问题

当前 `blackhole.gemm_contract` 只包含：

- `ab_dtype`
- `c_dtype`

这会把至少三种不同层级的语义压成两项：

- host A/B/C tensor 的逻辑 dtype
- A/B/C 经 host layout conversion 后进入 CB 的 packed transport dtype
- compute accumulator / output tile dtype

这与 `stage2d_ttmetal_contract_audit.md` 中对 P0 的判断一致：dtype contract 还没有正式拆层。

---

## 3. 设计范围

本轮只覆盖 GEMM direct path：

- `LowerBlackholeOps` 产出更完整的 `blackhole.gemm_contract`
- `rt_mod_blackhole` 提取并写入 `ExecutableSpec`
- `BlackholeModule` 从显式 dtype contract 做输入/输出校验与 host transfer 选择
- 测试覆盖 contract materialization 和 spec/runtime 消费

本轮不做：

- accessor schema（P3）
- rich runtime work schema（P3）
- copy/dataflow 泛化（P4）
- semaphore / multicast（P5）
- 让 codegen 从新 dtype contract 派生更多 kernel compile args

---

## 4. 协议变化

### 4.1 `blackhole.gemm_contract`

由当前：

- `ab_dtype`
- `c_dtype`

扩展为：

- `a_tensor_dtype`
- `b_tensor_dtype`
- `c_tensor_dtype`
- `a_cb_dtype`
- `b_cb_dtype`
- `c_cb_dtype`
- `accumulator_dtype`

当前默认映射保持最小闭环：

- A/B host tensor dtype = A/B CB transport dtype = 原始输入 buffer dtype
- C host tensor dtype = accumulator dtype = 原始输出 buffer dtype
- C CB transport dtype 暂时与 accumulator dtype 相同

这次收正的重点是“显式分层”，不是引入新的 dtype 组合。

### 4.2 `ExecutableSpec`

`GemmContractSpec` 同步扩展到上述字段，替代旧的：

- `ab_dtype`
- `c_dtype`

### 4.3 runtime 消费

`BlackholeModule` 不再把：

- GEMM 输入必须是“任意 16-bit”
- GEMM 输出必须是“固定 float32”

当成唯一真源，而是：

- 先按 `gemm_contract` 校验 host tensor dtype
- 再按 `gemm_contract` 决定 host transfer 是否允许、如何解释输入输出

本轮只保留当前正式支持组合：

- A/B tensor dtype = `Float16_b`
- C tensor dtype / accumulator dtype / C transport dtype = `Float32`

若 schema 与 runtime 当前支持矩阵不一致，直接报错，不再默默依赖硬编码假设。

---

## 5. 验证方式

1. 结构测试：
   - `test_blackhole_gemm_contract_attr_is_materialized`
   - 断言新 dtype 字段存在且值正确
2. spec/runtime 测试：
   - 新增对 `ExecutableSpec.gemm_contract` 提取结果的断言
   - direct runtime 继续通过现有 GEMM single-core / multi-core E2E
3. 回归测试：
   - `test_blackhole_tvm_ffi_export.py`
   - copy pipeline/runtime 测试至少抽样回归，确认本轮未误伤非 GEMM 路径

---

## 6. 影响范围

- `tilelang_repo/src/transform/lower_blackhole_ops.cc`
- `tilelang_repo/src/target/rt_mod_blackhole.cc`
- `tilelang_repo/src/target/blackhole_module.h`
- `tilelang_repo/src/target/blackhole_module.cc`
- `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`

---

## 7. 完成标准

- `blackhole.gemm_contract` 的 dtype 分层成为显式 schema
- `ExecutableSpec` 与 direct runtime 消费新 schema
- 当前 GEMM direct path 测试继续通过
- 相关设计/进度/经验文档与代码一致
