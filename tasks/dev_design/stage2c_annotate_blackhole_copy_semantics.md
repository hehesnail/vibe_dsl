# Stage 2C 设计：AnnotateBlackholeCopySemantics

## 基本信息

- **文档ID**: `stage2c_annotate_blackhole_copy_semantics`
- **日期**: 2026-03-23
- **状态**: 设计中
- **对应阶段**: Stage 2C — split-before 语义规划

---

## 1. 问题描述

### 当前链路

```
LowerAndLegalize
  -> LowerTileOp          ← 将 tl.copy 压碎为 BufferStore(BufferLoad) 循环
  -> ...
OptimizeForTarget (Blackhole 分支)
  -> IfStmtBinding / PlanAndUpdateBufferAllocationLocation / PipelinePlanning
  -> LowerOpaqueBlock → Simplify
  -> AnnotateDeviceRegions → SplitHostDevice
  -> ...
blackhole_codegen
  -> LowerBlackholeOps    ← 靠模式匹配从 loop body 中恢复 copy 语义
  -> PlanBlackholeCB
  -> AssignBlackholeCores
```

### 核心脆弱点

`LowerBlackholeOps::VisitStmt_(ForNode)` 在 split 后靠
`CollectNestedCopyStores` / `FindNestedCopyStore` 对 For loop body 做
`BufferStore(BufferLoad)` 模式匹配，来恢复：

- copy 方向（`kDramToCB` / `kCBToDram` / `kDramToDram`）
- 源/目标 buffer 名称和 scope
- 数据类型、shape 信息

**一旦以下任一 pass 在 split 前执行过**，pattern 就可能失效：

- `FlattenBuffer`：将多维 BufferStore 展平为线性访问
- `VectorizeLoop`：将 loop 体矢量化，循环结构改变
- `StorageRewrite`：可能合并或重排 buffer 访问

这是当前 `FlattenBuffer` / `VectorizeLoop` / `StorageRewrite` 在 Blackhole
分支中还被绕过的根本原因（`stage2_pass_reuse_matrix.md` 中的遗留问题）。

---

## 2. 目标

在 split 前、LowerOpaqueBlock+Simplify 之后，显式地为每个 copy loop
注入结构化元数据 `blackhole.copy_semantics`，使：

1. `LowerBlackholeOps` 可以直接消费 annotation，不再依赖 ForNode 体模式匹配
2. 中后段通用 pass（FlattenBuffer 等）后 copy 语义仍可完整恢复
3. split 前语义规划与 split 后 plan 提取之间建立稳定的显式协议

---

## 3. 新 Pass：`AnnotateBlackholeCopySemantics`

### 3.1 插入位置

`phase.py` Blackhole 分支，`Simplify` 之后、`AnnotateDeviceRegions` 之前：

```python
# Blackhole 分支 OptimizeForTarget
mod = tilelang.transform.LowerOpaqueBlock()(mod)
mod = tilelang.transform.Simplify()(mod)
mod = tir.transform.VerifyMemory()(mod)
mod = tir.transform.AnnotateEntryFunc()(mod)
# ... (ThreadSync, etc.)
mod = tilelang.transform.AnnotateBlackholeCopySemantics()(mod)   # ← 新增
mod = tilelang.transform.AnnotateDeviceRegions()(mod)
mod = tilelang.transform.SplitHostDevice()(mod)
```

### 3.2 输入

split 前的 PrimFunc，`LowerTileOp` 已将 `tl.copy` 变成：

```
for i in range(N):
  B[i, j] = A[i, j]           # BufferStore(BufferLoad)
```

其中 A 的 scope = `global`，B 的 scope = `shared`（或反之）。

### 3.3 输出

将 copy 所在的**最外层** For loop 包裹为：

```
attr [{"blackhole.copy_semantics": <map>}]:
  for i in range(N):
    B[i, j] = A[i, j]
```

Annotation 挂在 `AttrStmt` 的 `attr_key = "blackhole.copy_semantics"` 上，
value 为一个 `Map<String, ObjectRef>`（见 3.4）。

Fused copy（同一 loop 内含 dram→cb 和 cb→dram 两个 copy）：

```
attr [{"blackhole.copy_semantics": <map with kind="fused_staged_copy">}]:
  for i in range(N):
    B[i, j] = A[i, j]     # dram→cb
    C[k, l] = B[i, j]     # cb→dram
```

### 3.4 Annotation Schema

```
blackhole.copy_semantics → Map<String, ObjectRef>
  "kind"        : String
      "staged_copy"                 一个方向
      "fused_staged_copy"           dram→cb + cb→dram，公用同一 shared buffer
  "direction"   : String
      "dram_to_cb" | "cb_to_dram" | "dram_to_dram"
      (fused_staged_copy 时设为 "dram_to_cb_to_dram")
  "src_buffer"  : String    源 buffer 名称
  "dst_buffer"  : String    目标 buffer 名称
  "mid_buffer"  : String    仅 fused 时：shared 中间 buffer 名称
  "src_scope"   : String    源 buffer storage scope（"global" / "shared"）
  "dst_scope"   : String    目标 buffer storage scope
  "dtype"       : String    "float16" / "float32" / "int32" 等
  "src_shape"   : Array<Integer>    源 buffer 静态 shape（尽力提取）
  "dst_shape"   : Array<Integer>    目标 buffer 静态 shape（尽力提取）
```

### 3.5 实现要点

```cpp
class BlackholeCopyAnnotator : public StmtMutator {
public:
  Stmt VisitStmt_(const ForNode* op) final;
private:
  bool HasCopyStore(const Stmt&) const;
  void CollectCopyStores(const Stmt&, std::vector<const BufferStoreNode*>*) const;
  bool IsCopyOp(const BufferStoreNode*) const;
  std::string GetScope(const Buffer&) const;
  std::string DirectionStr(const BufferStoreNode*) const;
  bool IsDramScope(const std::string&) const;
  bool IsCBScope(const std::string&) const;
  Map<String, ObjectRef> BuildAnnotation(...) const;
};
```

`VisitStmt_(ForNode)`:
1. 调用 `CollectCopyStores` 遍历 loop body，收集所有 `BufferStore(BufferLoad)` copy
2. 如果找到 dram→cb + cb→dram（fused 情形），构建 `kind="fused_staged_copy"` annotation
3. 如果找到单方向 copy（dram→cb 或 cb→dram），构建 `kind="staged_copy"` annotation
4. 用 `AttrStmt(op->node = NullValue, attr_key = "blackhole.copy_semantics", value = annotation, body = GetRef<For>(op))` 包裹
5. 没有 copy 时，走默认 `StmtMutator::VisitStmt_(op)` 继续递归

**不修改** LowerTileOp 的任何逻辑。

---

## 4. LowerBlackholeOps 修改

### 4.1 新增 AttrStmt 处理

在现有 `VisitStmt_(AttrStmtNode)` 里增加对 `"blackhole.copy_semantics"` 的处理：

```cpp
Stmt LowerBlackholeOps::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == "blackhole.copy_semantics") {
    return ConsumeCopySemantics(op);
  }
  // ... 现有逻辑
}

Stmt LowerBlackholeOps::ConsumeCopySemantics(const AttrStmtNode* op) {
  auto sem = Downcast<Map<String, ObjectRef>>(op->value);
  // 提取 direction、buffer 名称，设置 copy_input_buffer_name_ 等
  // 标记 needs_copy_runtime_args_ = true, saw_copy_op_ = true
  // 继续 VisitStmt(op->body) 以生成实际 builtin 调用序列
  return VisitStmt(op->body);
}
```

### 4.2 保持现有模式匹配作为 fallback

`VisitStmt_(ForNode)` 现有逻辑**不删除**，作为 fallback：
- 如果 For loop 外已有 `blackhole.copy_semantics` annotation，会先走 `VisitStmt_(AttrStmtNode)` 进入 `ConsumeCopySemantics`，body 继续分派到 `VisitStmt_(ForNode)`（此时 `needs_copy_runtime_args_` 已设置，逻辑不变）
- 如果没有 annotation（老路径 / 其他非 Blackhole path），仍走现有模式匹配

---

## 5. Python 侧变更

### 5.1 `tilelang/transform/__init__.py`

```python
def AnnotateBlackholeCopySemantics():
    return _ffi_api.AnnotateBlackholeCopySemantics()
```

### 5.2 `tilelang/engine/phase.py`

```python
# Blackhole 分支，Simplify 之后 AnnotateDeviceRegions 之前
mod = tilelang.transform.AnnotateBlackholeCopySemantics()(mod)
mod = tilelang.transform.AnnotateDeviceRegions()(mod)
```

---

## 6. 文件变更范围

| 文件 | 类型 | 说明 |
|------|------|------|
| `src/transform/annotate_blackhole_copy_semantics.cc` | 新增 | 主 pass 实现 |
| `src/transform/lower_blackhole_ops.cc` | 修改 | 增加 ConsumeCopySemantics |
| `tilelang/engine/phase.py` | 修改 | 插入新 pass |
| `tilelang/transform/__init__.py` | 修改 | 新增 Python binding |
| `CMakeLists.txt`（transform 目录）| 修改 | 新增 .cc 文件 |

---

## 7. 验证方式

1. `test_blackhole_e2e.py` 全套用例保持 `18 passed, 1 skipped`（无回归）
2. 在 TIR dump 中能看到 `attr blackhole.copy_semantics` 包裹在 copy For loop 外层
3. 人工插入 FlattenBuffer（仅测试）后，copy 语义仍能被 `LowerBlackholeOps` 正确识别
4. `blackhole.runtime_args` / `blackhole.cb_requirements` 内容与当前路径一致

---

## 8. 不做的事

- 不修改 `LowerTileOp` 核心降级逻辑
- 不在这一层生成 `runtime_args` / `cb_id` / `core_plan`（这些仍在 split 后做）
- 不处理 GEMM 语义（留给后续 stage）
- 不删除 `LowerBlackholeOps` 中现有的 ForNode 模式匹配 fallback

---

## 9. 遗留问题与限制

- 当前只处理静态 shape copy（动态 shape 的 tile index 计算留后续）
- `AnnotateBlackholeCopySemantics` 只处理 `kDramToCB` / `kCBToDram` / `kDramToDram`；GEMM-related copy 语义（kCBToCB）另行处理
- 通用 pass（FlattenBuffer 等）接回 Blackhole 主线是 Phase 4 的事，本 pass 只是铺路
