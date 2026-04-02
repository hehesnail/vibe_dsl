# Stage 2D 补丁设计：CB Identity 唯一协议收正

## 基本信息

- **文档ID**: `stage2d_cb_identity_protocol`
- **日期**: 2026-03-25
- **状态**: 已实施
- **对应阶段**: Stage 2D Step 6 的前置修正
- **前置上下文**: Stage 2D Step 1-5 已完成，Step 6 E2E 验收被 CB identity 协议错位阻塞

---

## 1. 问题定义

### 1.1 现象

`test_blackhole_gemm_basic` 的 direct-path E2E 验收中，reader/compute/writer 三段没有在同一个 CB identity 上同步。具体表现：

- reader 的 `cb_reserve_back/cb_push_back` 可能使用 cb_id=0/1（局部实际 id）
- compute 的 `mm_init/cb_wait_front/matmul_tiles` 使用 planner 解析后的 cb_id=3/4/17
- writer 的 `cb_wait_front/cb_pop_front` 使用另一组 id

producer 和 consumer 不在同一个 FIFO 上同步，导致执行结果错误。

### 1.2 根因分析

问题不在 TT-Metal API 误用，也不是 GEMM 特有的。根因是 **CB identity 在管线中没有唯一的写入点和消费点**，三个 pass 各自用了不同的方式处理 CB id：

#### `LowerBlackholeOps` 同时产出两套互不兼容的 CB identity

- **Copy 路径**：`AllocateCBId()` 立即分配实际 id（0/1/16/32），`IntImm32(cb_id)` 作为字面常量写入 IR body
- **GEMM 路径**：使用 placeholder（`-1/-2/-3`），作为 `IntImm32` 写入 IR body

设计文档明确说 LowerBlackholeOps "不分配最终 cb_id"，但 copy 路径的实现违反了这一原则。

#### `PlanBlackholeCB` 不回写 IR body

PlanBlackholeCB 读 `blackhole.cb_requirements` → 分配 `cb_id` → 写到 `blackhole.cb_configs` 和 `blackhole.cb_bindings` attrs。但它**完全不修改 IR body**。

IR body 中的 cb_id 仍然是 LowerBlackholeOps 写入的原始值。这导致 IR 和 attrs 中存在两套独立的 cb_id：
- IR body 中：来自 LowerBlackholeOps 的实际 id 或 placeholder
- Function attrs 中：来自 PlanBlackholeCB 的正式分配 id

对 copy，两套恰好一致（分配逻辑相同），属于**巧合而非协议保证**。

#### `CodeGenBlackhole` 承担了不该做的 placeholder 替换

`LoadGemmCBPlaceholders` 在 codegen 阶段用 `requirement_name` 做 key 查 `cb_bindings`，遇到 `-1/-2/-3` 替换成实际 id。

- `unordered_map<string, int>` 按 name 查——同名覆盖，last-one-wins
- placeholder 替换本质是"register allocation 回写"，应该是 IR pass 的职责，不该推到 codegen

#### `cb_bindings` 的 key 不唯一

`PlanBlackholeCB` 的 lifetime reuse 允许多个 requirement 合并到同一 CB，`cb_bindings` 中可以出现多条同 `requirement_name` 的 entry。`requirement_name` 不是唯一键。

### 1.3 通用性

任何满足以下条件的场景都会触发此问题：

| 场景 | 触发条件 |
|------|----------|
| Multi-segment kernel | 有 `SplitBlackholeKernel` 拆分 |
| CB lifetime reuse | `PlanBlackholeCB` 合并同 role+format 的 requirement |
| 跨 segment 共享 CB | producer/consumer 必须同步 |

这意味着除当前 GEMM 外，未来的 fused pipeline、multi-stage copy、element-wise + reduction 融合等都会碰到。

---

## 2. 设计原则

1. **CB id 在管线中只有一个写入点**：`PlanBlackholeCB` 是唯一分配最终 cb_id 的地方
2. **`LowerBlackholeOps` 不分配最终 cb_id**：对齐设计文档已有规定
3. **IR body 是 cb_id 的唯一真源**：codegen 直接读 IR 中的 cb_id，不做任何查找/替换
4. **binding 用整数索引做唯一键**：`requirement_index`（cb_requirements 数组下标），不用 `requirement_name`
5. **不为 GEMM 做特殊路径**：copy 和 GEMM 使用完全相同的 CB identity 协议

---

## 3. 目标数据流

```
LowerBlackholeOps
  ├── IR body 中所有 blackhole builtin 的 cb_id 参数统一写 requirement_index（0, 1, 2, ...）
  ├── 产出 blackhole.cb_requirements 数组（不含 cb_id，只含 name/type/page_size/num_pages/format/lifetime）
  └── 不区分 copy/GEMM，都是同一套 requirement_index

PlanBlackholeCB
  ├── 读 blackhole.cb_requirements
  ├── 分配 cb_id（考虑 role 范围、lifetime reuse、L1 约束）
  ├── 产出 blackhole.cb_configs + blackhole.cb_bindings（key 为 requirement_index）
  └── **回写 IR body**：遍历所有 blackhole builtin 调用，把 requirement_index 替换成最终 cb_id

CodeGenBlackhole
  └── 直接读 IR 中的 cb_id，打印。不做任何 placeholder 查找/替换/alias 映射。
```

---

## 4. 具体改动

### 4.1 `LowerBlackholeOps`（`src/transform/lower_blackhole_ops.cc`）

**目标**：所有 CB 引用统一用 `requirement_index`。

改动点：

1. **删除 `kGemmInputAPlaceholderCB` / `kGemmInputBPlaceholderCB` / `kGemmOutputCPlaceholderCB` 常量**
   - 不再需要 `-1/-2/-3` placeholder 体系

2. **`AllocateCBId()` 改名为 `AllocateRequirementIndex()`**
   - 返回值语义从"分配实际 cb_id"变为"在 cb_requirements_ 数组中注册并返回下标"
   - 分配逻辑不再区分 input/output/intermediate 的 id 范围
   - 只是递增的 requirement_index（0, 1, 2, ...）

3. **`GenerateMatmulSequence()` 使用 requirement_index**
   - `mm_init(req_idx_a, req_idx_b, req_idx_c)` 而非 `mm_init(-1, -2, -3)`
   - 所有 `cb_wait_front/cb_pop_front/cb_reserve_back/pack_tile/cb_push_back` 同理

4. **Copy 路径统一**
   - `GenerateCopySequence()` / `GenerateStagedCopyLoopSequence()` / `GenerateFusedStagedCopySequence()` 中的 `IntImm32(cb_id)` 改为 `IntImm32(requirement_index)`
   - 不再有 `if (segmented_gemm)` 分支

5. **删除 `StoreGemmCBPlaceholders()`**
   - 不再需要 `blackhole.gemm_cb_placeholders` attr
   - 删除 `ResolveGemmPlaceholderCBId()` / `ResolveGemmPlaceholderCBIdFromAttrs()`

6. **`StoreCBRequirements()` 补充 `requirement_index` 字段**
   - 每个 requirement 显式写出自己的 index，便于后续 pass 和 codegen 验证

### 4.2 `PlanBlackholeCB`（`src/transform/plan_blackhole_cb.cc`）

**目标**：分配 cb_id 后回写 IR body。

改动点：

1. **新增 IR body 回写步骤**
   - 在 `StoreCBConfig()` 之后，新增 `RewriteCBIdsInIR()` 方法
   - 遍历 IR body，找到所有 blackhole builtin 调用（`cb_reserve_back/cb_push_back/cb_wait_front/cb_pop_front/read_tile_to_cb/write_tile_from_cb/mm_init/matmul_tiles/pack_tile`）
   - 这些 builtin 中携带 cb_id 的参数位置是固定的（第 0 个参数或第 2 个参数，取决于 builtin）
   - 用 `requirement_index → cb_id` 映射表做替换

2. **`cb_bindings` 改用 `requirement_index` 做主键**
   - 当前 binding 已含 `requirement_index` 字段，只需要把消费侧也改为按 index 查找
   - `requirement_name` 保留为辅助信息（调试/日志），但不再作为查找键

3. **建立 `requirement_index → cb_id` 映射表**
   - 在 `AssignCBIds()` 返回后构建
   - 对于被 lifetime reuse 合并的 requirement：多个 requirement_index 映射到同一个 cb_id（这正是 reuse 的语义）

### 4.3 `SplitBlackholeKernel`（`src/transform/split_blackhole_kernel.cc`）

**目标**：删除 placeholder 写入。

改动点：

1. **删除 `StoreGemmCBPlaceholders()`**
   - `SplitBlackholeKernel` 不再写 `blackhole.gemm_cb_placeholders`
   - 该 attr 在统一协议后不再需要

2. **其他逻辑不变**
   - segment kind 标注、segment_plan 写入保持不变

### 4.4 `CodeGenBlackhole`（`src/target/codegen_blackhole.cc`）

**目标**：删除所有 placeholder 替换逻辑。

改动点：

1. **删除 `LoadGemmCBPlaceholders()`**
   - 删除 `placeholder_cb_id_map_`
   - 删除 `cb_id_by_requirement_name` 查找逻辑

2. **删除 `VisitExpr_(CallNode)` 中的 placeholder 替换**
   - codegen 遇到 `cb_reserve_back(cb_id, ...)` 时，`cb_id` 已经是最终值，直接打印

3. **删除 `resolved_cb_by_runtime_arg` alias 映射**
   - 不再需要从 placeholder → buffer name → runtime arg name 的间接映射链

4. **`cb_page_size_by_id_` / `cb_num_pages_by_id_` 的填充保持不变**
   - 仍从 `blackhole.cb_configs` 读取，因为 cb_configs 中的 cb_id 是 PlanBlackholeCB 分配的最终值

### 4.5 `rt_mod_blackhole`（`src/target/rt_mod_blackhole.cc`）

**目标**：segment 提取不再需要特殊处理 CB binding。

改动点：

1. **`MakeSegmentPrimFunc()` 保持不变**
   - 全局 attrs 原样复制给每个 segment PrimFunc 是正确的（三个 segment 共享同一组 CB）
   - IR body 中的 cb_id 已经是最终值，segment 提取后仍然正确

2. **删除对 `blackhole.gemm_cb_placeholders` 的任何引用**（如果有的话）

---

## 5. Attr Schema 变更

### 删除

- `blackhole.gemm_cb_placeholders`：不再需要

### 不变

- `blackhole.cb_requirements`：保持现有格式（requirement_index 已隐含在数组下标中，可选显式写出）
- `blackhole.cb_configs`：保持现有格式
- `blackhole.segment_plan`：保持现有格式
- `blackhole.runtime_args`：保持现有格式
- `blackhole.core_plan`：保持现有格式

### 变更

- `blackhole.cb_bindings`：`requirement_index` 成为主键（消费侧改为按 index 查找），`requirement_name` 降级为辅助信息

---

## 6. IR body 中 CB id 参数位置（builtin → 参数映射）

PlanBlackholeCB 回写时需要知道每个 builtin 的 cb_id 在第几个参数：

| Builtin | cb_id 参数位置 | 说明 |
|---------|---------------|------|
| `blackhole_cb_reserve_back(cb, n)` | arg[0] | |
| `blackhole_cb_push_back(cb, n)` | arg[0] | |
| `blackhole_cb_wait_front(cb, n)` | arg[0] | |
| `blackhole_cb_pop_front(cb, n)` | arg[0] | |
| `blackhole_read_tile_to_cb(buf, tile_idx, cb, page_size, offset)` | arg[2] | |
| `blackhole_write_tile_from_cb(cb, buf, tile_idx, page_size, offset)` | arg[0] | |
| `blackhole_mm_init(cb_a, cb_b, cb_out)` | arg[0], arg[1], arg[2] | 三个参数都是 cb_id |
| `blackhole_matmul_tiles(cb_a, cb_b, dst, r, c)` | arg[0], arg[1] | |
| `blackhole_pack_tile(dst_idx, cb)` | arg[1] | |

---

## 7. 验证计划

### 7.1 Copy 回归

所有现有 copy 测试必须继续通过：
- `test_blackhole_copy_pipeline.py`：14 passed, 1 skipped, 1 xfailed

验证要点：copy 路径虽然内部改为 requirement_index → PlanBlackholeCB 回写 → 最终 cb_id，但最终生成的 kernel source 应与当前一致（因为分配逻辑不变）。

### 7.2 GEMM lower

- `test_blackhole_gemm.py::test_gemm_lower_basic`：lowered IR 中 builtin 的 cb_id 应为 PlanBlackholeCB 分配的最终值
- `test_blackhole_gemm.py::test_blackhole_gemm_cb_placeholders_resolve_via_planner`：此测试需要适配——不再检查 placeholder 替换，改为检查 IR 中 cb_id 的一致性

### 7.3 GEMM E2E

- `test_blackhole_gemm.py::test_blackhole_gemm_basic`：reader/compute/writer 三段使用完全相同的 cb_id

### 7.4 结构验证

新增或修改测试用例：
- 验证 lowered IR 中无 placeholder（-1/-2/-3）
- 验证 lowered IR 中无 requirement_index（PlanBlackholeCB 回写后全部替换为最终 cb_id）
- 验证 `blackhole.gemm_cb_placeholders` attr 不再出现

---

## 8. 实施结果（2026-03-25）

- `LowerBlackholeOps` 已删除 placeholder/局部实际 `cb_id` 双轨写入：
  - copy/GEMM 统一写 `requirement_index`
  - `StoreGemmCBPlaceholders()` 已删除
- `PlanBlackholeCB` 已实现 `RewriteCBIdsInIR()`：
  - planner 分配后直接把 IR body 中的 `requirement_index` 回写成最终 `cb_id`
  - codegen 不再承担任何“后补替换”职责
- `CodeGenBlackhole` 已删除 placeholder/alias 解析路径：
  - 不再读取 `blackhole.gemm_cb_placeholders`
  - `ResolveCBId()` 只接受最终非负 `cb_id`
- `SplitBlackholeKernel` 已停止写 `blackhole.gemm_cb_placeholders`
- 实施过程中新增确认：
  - `requirement_index` 与 `lifetime_begin/end` 不能混用成同一语义
  - 对 GEMM，`A/B` 输入 requirement 必须显式设置重叠 lifetime，避免 planner 误复用为同一个 input CB

本地验证结果：

- `pytest -q testing/python/target/blackhole/test_blackhole_copy_pipeline.py` → `15 passed, 1 xfailed`
- `pytest -q testing/python/target/blackhole/test_blackhole_gemm.py` → `3 passed, 1 skipped`

当前未完成项：

- `test_blackhole_gemm_basic` 仍未在本轮完成 true E2E 验收；skip 原因是 direct runtime 环境未配置，不是新的代码级回退

---

## 8. 影响范围评估

| 文件 | 改动类型 | 风险 |
|------|----------|------|
| `lower_blackhole_ops.cc` | 统一 CB identity 产出 | 中（copy 行为变化需要回归验证） |
| `plan_blackhole_cb.cc` | 新增 IR 回写 | 中（新增 StmtMutator 逻辑） |
| `split_blackhole_kernel.cc` | 删除 placeholder 写入 | 低 |
| `codegen_blackhole.cc` | 删除 placeholder 替换 | 低（纯删除） |
| `rt_mod_blackhole.cc` | 可能删除 placeholder 引用 | 低 |
| `codegen_blackhole.h` | 删除 placeholder 相关成员 | 低 |
| `lower_blackhole_ops.h` | rename + 删除 placeholder 相关 | 低 |
| `test_blackhole_gemm.py` | 适配测试用例 | 低 |

---

## 9. 不做的事

- 不改 `CBRequirement` / `CBConfig` struct 布局（只变语义，不变结构）
- 不改 `blackhole.cb_requirements` 的 attr schema 格式
- 不改 `PlanBlackholeCB` 的 lifetime reuse 算法
- 不改 `SplitBlackholeKernel` 的 segment 分类逻辑
- 不改 `BlackholeModule::ExecuteDirect()` 的执行逻辑
- 不在本次统一 copy 的 segment 模型（fused_dataflow 与多 kernel 的统一是独立架构债）
