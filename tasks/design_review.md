# TileLang Blackhole 后端设计审查与改进方案（最终版）

> 审查日期: 2026-03-17
> 审查范围: Phase 0-3 所有设计文档 + 全部源码实现
> 版本: v2 (修订版)

---

## 一、执行摘要

当前 Blackhole 后端完成度约 **40-50%**，存在若干根本性架构问题：

1. **设计与实现严重脱节**：Phase 2 的三个 Transform Pass 在文档中标记为"已完成"，但实际代码均为 stub
2. **CodeGen 与 TIR Pipeline 未正确对接**：CodeGen 使用启发式方式而不是通过 IR 属性驱动
3. **多 Kernel 协同问题被忽略**：Reader/Compute/Writer 三核协同是 Blackhole 编程模型的核心，但整个设计对此处理极为粗糙
4. **缺乏完整的 Lowering Pipeline**：CUDA 有 30+ 个 pass 的完整 pipeline，Blackhole 的 lowering 几乎空白
5. **CodeGen 中存在大量硬编码**：Copy kernel 整个是字符串拼接，没有走 IR visitor 路径
6. **arch_design.md 中的 Pipeline 设计只停留在文档**：`lower.py` 中 Blackhole 分支没有调用任何 Blackhole Pass

---

## 二、问题清单（按严重程度排序）

### 【P0 - 阻塞性问题】

#### 问题1：Transform Pass 全部是 Stub

**证据**：
- `split_blackhole_kernel.cc:42-49` — `Transform()` 返回原函数，注释写 "For now"
- `plan_blackhole_cb.cc:41-44` — `Transform()` 返回原函数
- `plan_blackhole_cb.cc:48-50` — `Validate()` 永远返回 true

**影响**：整个 Phase 2 的 IR 变换层完全缺失，CodeGen 拿到的是未经处理的原始 TIR。

#### 问题2：lower.py 未接入任何 Blackhole Pass

**证据**（`lower.py:180-194`）：
```python
def device_codegen(device_mod, target):
    # 这四行 pass 对所有 target 都执行
    device_mod = tilelang.transform.LowerDeviceStorageAccessInfo()(device_mod)
    device_mod = tilelang.transform.LowerIntrin()(device_mod)
    device_mod = tir.transform.Simplify()(device_mod)
    device_mod = tilelang.transform.HoistBroadcastValues()(device_mod)
    # Blackhole 分支直接跳到 CodeGen，没有任何 Blackhole Pass！
    elif target.kind.name == "blackhole":
        device_mod = tvm.ffi.get_global_func("target.build.tilelang_blackhole")(...)
```

**对比 arch_design.md 中的设计**：
```python
# arch_design.md 写的是这样，但实际未实现：
#   device_mod = AssignBlackholeCores()(device_mod)
#   device_mod = PlanBlackholeCB()(device_mod)
#   device_mod = SplitBlackholeKernel()(device_mod)
```

#### 问题3：CodeGen 用启发式而非 IR 属性驱动

**证据**（`codegen_blackhole.cc:126-134`）：
```cpp
bool DetectSimpleCopyKernel(const PrimFunc &f) {
  int buffer_params = 0;
  for (const auto &param : f->params) {
    if (param->dtype.is_handle()) buffer_params++;
  }
  return buffer_params == 2;  // 2个buffer参数就认为是copy！
}
```

任何接收2个buffer参数的函数都会被误判为copy kernel，包括 element-wise add、transpose 等。

#### 问题4：Copy kernel 是硬编码字符串拼接

**证据**（`codegen_blackhole.cc:137-220`）：`GenerateCopyKernelMain()` 完全不走 IR visitor 路径，而是直接用 `stream << "cb_reserve_back(cb_id, 1);\n"` 这样的字符串拼接。

**问题**：
- 与 generic kernel path 完全割裂
- tile_size 硬编码为 2048
- CB id 硬编码为 0
- Reader 和 Writer 合并在同一个 kernel 中（违反三核模型）

### 【P1 - 重大问题】

#### 问题5：LowerBlackholeOps 的 Copy/Clear 未实现

**证据**（`lower_blackhole_ops.cc:224-242`）：
```cpp
Stmt GenerateCopySequence(const BufferStoreNode* op) {
  // ... 注释说 "Placeholder implementation"
  return VisitStmt_(op);  // 直接 fallback，什么都没做
}
```

Matmul 序列实现了（`GenerateMatmulSequence` 148-221行），但 Copy 和 Clear 没有。这意味着即使 Pipeline 接上了，copy 操作也无法被 lower。

#### 问题6：LowerBlackholeOps 用字符串匹配识别算子

**证据**（`lower_blackhole_ops.cc:110-123`）：
```cpp
bool IsMatmulCall(const CallNode* op) const {
  Op call_op = Downcast<Op>(op->op);
  std::string op_name = call_op->name;
  return (op_name.find("gemm") != std::string::npos ||
          op_name.find("matmul") != std::string::npos ||
          op_name == "tl.matmul");
}
```

这种字符串 find 方式非常脆弱，应该用 Op 对象比较。

#### 问题7：CodeGen 中 `static bool headers_emitted` 是全局状态

**证据**（`codegen_blackhole.cc:61`）：
```cpp
static bool headers_emitted = false;
```

这意味着如果编译多个 kernel，第二个之后的 kernel 不会有头文件。而且进程生命周期内这个状态不会重置。

#### 问题8：三核模型与 TT-Sim 限制的矛盾

`memory/general_dev.md` 记录 TT-Sim 只支持 BRISC 的完整 NOC 操作，NCRISC 的 NOC write 会报错。但设计中 Writer kernel 运行在 NCRISC 上。

当前的 copy kernel 将 reader 和 writer 合并到一个 kernel 中（运行在 BRISC），这实际上是对 TT-Sim 限制的 workaround，但文档中没有明确说明这是临时方案。

### 【P2 - 中等问题】

#### 问题9：两套 Module 实现并存

- `blackhole_module.cc` — 外部进程模式（当前激活）
- `blackhole_module_direct.cc` — 直接 TT-Metal 链接（完整但未使用）

外部进程模式的根本问题：每次调用 fork/exec，数据通过临时文件传输，无法处理大 tensor。

#### 问题10：AssignBlackholeCores 实际上比文档说的做了更多也更少

**实际实现**（`assign_blackhole_cores.cc`）：
- ✅ `AnalyzeGrid()` 解析 thread_extent — 有真实实现
- ✅ `GetCoreCoord()` 逻辑→物理映射 — 正确
- ❌ 但 Pass 注册函数（`fpass`）只是调用 `Transform()` 后返回原函数，没有把结果存到 attrs

#### 问题11：测试是 Mock 测试，不测试真实 TIR

所有 Phase 2 的 gtest 都是在测试 Mock 对象（`MockSplitBlackholeKernel` 等），不操作任何 TVM/TIR 数据结构。这些测试通过并不代表 Pass 能处理真实的 TIR。

#### 问题12：core_type 检测基于函数名子串

**证据**（`codegen_blackhole.cc:72-79`）：
```cpp
if (symbol.find("_brisc") != std::string::npos) {
  core_type_ = CoreType::kBRISC;
}
```
应该用 IR attribute `"blackhole.core_type"` 驱动，而不是函数名。

---

## 三、根本原因分析

### 3.1 两条路径的割裂

当前代码存在两条完全独立的路径：

**路径 A（IR 驱动，正确但未完成）**：
```
TIR → LowerBlackholeOps → builtin calls → CodeGen.VisitExpr_ → HandleBlackholeBuiltin → 打印
```
这条路径设计正确，matmul 部分已实现，但 copy/clear 未实现。

**路径 B（硬编码，能工作但不可扩展）**：
```
PrimFunc → DetectSimpleCopyKernel() → GenerateCopyKernelMain() → 字符串拼接
```
这条路径能生成可用的 copy kernel，但完全绕过了 IR。

**问题**：两条路径互不相交。路径 B 是 Phase 1 的产物，路径 A 是 Phase 3 的方向，但切换还没发生。

### 3.2 对 Blackhole 编程模型的理解进展

从代码演进看，开发者对 TT-Metal 的理解在不断加深：
- Phase 1：合并 R/W 到单核（TT-Sim workaround）
- Phase 3：开始实现三核分离的 IR 表示

但这个理解进展没有回溯更新到 Phase 1/2 的实现中。

### 3.3 "文档先行"的陷阱

Phase 2 的三个 Pass 走了"写文档 → 写 stub → 写 mock 测试 → 标记完成"的路径。这在项目早期用于探索方向是合理的，但不应标记为"完成"。

---

## 四、改进后的完整设计

### 4.1 设计原则（修订）

1. **单一路径**：消除路径 B（硬编码），所有 kernel 生成走路径 A（IR 驱动）
2. **IR 属性驱动**：所有 Pass 通过 TIR attribute 传递信息，CodeGen 只读 attribute
3. **TT-Sim 兼容优先**：考虑到 TT-Sim 的 NCRISC 限制，短期允许合并 R/W kernel
4. **渐进实现**：先 Copy → 再 GEMM → 再复杂算子

### 4.2 整体架构（修订版）

```
TileLang DSL
    |
    v
PreLowerSemanticCheck()         [已有，复用]
    |
    v
LowerAndLegalize()              [已有，复用]
    |
    v
OptimizeForTarget()             [已有，部分复用]
    |
    v
[lower.py Blackhole 分支]       [★ 关键修改点]
    |
    +-- LowerBlackholeOps()     [Pass A: 高层 op -> TT-Metal builtin]
    +-- PlanBlackholeCB()       [Pass B: 规划 CB 布局]
    +-- AssignBlackholeCores()  [Pass C: 分配 Tensix 核]
    |
    v
CodeGenBlackhole                [纯 IR->字符串翻译]
    |-- 读 attrs 确定 kernel 类型和配置
    |-- Visitor 遍历 body，翻译 builtin calls
    |-- 生成 kernel_main() C++ 文件
    |
    v
BlackholeModule
    |-- 短期: 外部进程模式（TT-Sim 验证）
    |-- 长期: C 接口桥接层 + dlopen
    |
    v
TT-Sim / Hardware 执行
```

**Pass 顺序变更说明**：

原设计：`AssignCores → PlanCB → Split → LowerOps`
修订为：`LowerOps → PlanCB → AssignCores`（去掉 Split）

**理由**：
1. **LowerOps 应最先执行**：因为它将 `T.copy/T.gemm/T.clear` 转为具体的 CB/NOC builtin 调用。后续 Pass 需要看到这些 builtin 才能分析 CB 使用情况。
2. **去掉 SplitBlackholeKernel 的理由**：
   - TT-Sim 不支持 NCRISC NOC write，三核拆分后无法测试
   - TT-Metal 的 `CreateKernel` API 天然支持同一 core 上注册多个 kernel
   - 短期策略：合并 R/C/W 到单个 BRISC kernel（TT-Sim 兼容）
   - 长期策略：在 Runtime 层面按 RISC 核分发，不需要在 IR 层面拆分
3. **PlanCB 在 LowerOps 之后**：因为 LowerOps 生成的 `cb_reserve_back(cb_id, ...)` 等 builtin 调用包含了具体的 CB 使用信息，PlanCB 可以据此分配 CB ID。


### 4.3 Pass A：LowerBlackholeOps（重新设计）

**目标**：将 TileLang 高层操作转换为 TT-Metal builtin 序列

**位置**：`src/transform/lower_blackhole_ops.cc`（已有框架，需补全）

**当前状态**：Matmul ✅ | Copy ❌ | Clear ❌

**Copy 操作 lower 逻辑**：

需要在 `VisitStmt_` 中识别 copy 操作并展开为 CB/NOC 序列。
关键判断方式不应是字符串匹配，而是检查 buffer scope：

```cpp
// 判断 copy 方向的正确方式：检查 buffer 的 storage scope
bool IsDRAMToShared(const BufferStoreNode* op) {
  // 目标是 shared scope
  auto dst_scope = GetStorageScope(op->buffer);
  if (dst_scope != "shared") return false;
  // 源是 global scope (DRAM)
  if (auto* load = op->value.as<BufferLoadNode>()) {
    auto src_scope = GetStorageScope(load->buffer);
    return src_scope == "" || src_scope == "global";
  }
  return false;
}
```

**展开为**：
```
DRAM->CB (Reader):
  cb_reserve_back(cb_id, 1)
  noc_async_read(get_noc_addr(tile_idx, addr_gen), get_write_ptr(cb_id), tile_size)
  noc_async_read_barrier()
  cb_push_back(cb_id, 1)

CB->DRAM (Writer):
  cb_wait_front(cb_id, 1)
  noc_async_write(get_read_ptr(cb_id), get_noc_addr(tile_idx, addr_gen), tile_size)
  noc_async_write_barrier()
  cb_pop_front(cb_id, 1)
```

**Clear 操作**：
```
T.clear(buf) -> tile_regs_acquire()  // 清零 DST 寄存器
```

**Matmul 操作**（已实现，无需修改）：
```
T.gemm(A, B, C) -> mm_init + tile_regs_acquire + matmul_tiles loop + pack_tile
```

### 4.4 Pass B：PlanBlackholeCB（重新设计）

**目标**：分析 LowerBlackholeOps 输出中的 CB 使用，分配 CB ID

**输入**：经过 LowerBlackholeOps 处理后的 PrimFunc，包含 `cb_reserve_back(cb_id, ...)` 等调用

**当前状态**：完全 Stub ❌

**实现策略**：两阶段

**阶段1（短期 MVP）**：从函数 attributes 中读取 CB 配置
- LowerBlackholeOps 在生成 builtin 时，将 CB 需求写入 func attrs
- PlanBlackholeCB 读取 attrs，验证约束，分配 CB ID
- 约束验证：`sum(page_size * num_pages) <= 1,572,864 bytes` 且 `num_cbs <= 64`

**阶段2（长期）**：通过 IR 分析自动推断
- 遍历 IR 中所有 `cb_reserve_back` / `cb_wait_front` 等调用
- 收集所有引用的 CB ID
- 分析各 CB 的 data format 和 size 需求
- 自动分配和优化 CB layout

**CB 分配约定**（与 TT-Metal 官方示例一致）：
```
CB 0-15   = 输入 buffer（Reader -> Compute）
CB 16-31  = 输出 buffer（Compute -> Writer）
CB 32-63  = 中间 buffer / 保留
```

**输出**：在 PrimFunc attrs 中写入：
```python
{
  "blackhole.cb_configs": [
    {"cb_id": 0, "page_size": 2048, "num_pages": 2, "data_format": "Float16"},
    {"cb_id": 1, "page_size": 2048, "num_pages": 2, "data_format": "Float16"},
    {"cb_id": 16, "page_size": 4096, "num_pages": 1, "data_format": "Float32"},
  ],
  "blackhole.total_l1_bytes": 12288,
}
```

### 4.5 Pass C：AssignBlackholeCores（修正现有实现）

**当前状态**：~60%，核心逻辑正确但 Pass 未正确存储结果

**需要修正**：
1. `Transform()` 的结果需要写入 func attrs（当前只是计算了但没存）
2. Pass 注册函数需要返回修改后的函数（当前返回原函数）

**坐标映射**（已验证正确，保持不变）：
```cpp
px = (lx < 7) ? lx + 1 : lx + 3;  // 跳过 x=8,9
py = ly + 2;                        // 偏移 +2
```

**输出 attrs**：
```python
{
  "blackhole.grid_shape": [grid_x, grid_y],
  "blackhole.cores_needed": min(grid_x * grid_y, 140),
  "blackhole.work_per_core": ceil(total_work / cores_needed),
}
```

### 4.6 SplitBlackholeKernel — 降级为可选

**决策**：短期不实现，长期可选

**理由**：
1. TT-Sim 限制：NCRISC NOC write 不工作，拆分后无法测试 Writer kernel
2. 合并 R/C/W 到 BRISC 单核可以工作（Phase 1 已验证）
3. TT-Metal API 允许同一 core 上注册多个 kernel，Runtime 层面可以处理分发
4. 真正需要三核并行时（性能优化），可以在 Runtime 层面实现，不需要 IR 层面拆分

**保留 stub**：不删除文件，但在文档中标记为"Phase 4 - 优化"。

### 4.7 CodeGen（重构方向）

**核心变更**：删除所有硬编码路径，统一走 IR Visitor

**删除**：
- `DetectSimpleCopyKernel()` — 启发式检测
- `GenerateCopyKernelMain()` — 字符串拼接
- `GenerateSimpleCopyKernel()` / `GenerateReaderKernel()` / `GenerateWriterKernel()` — 硬编码模板
- `static bool headers_emitted` — 全局状态

**保留并强化**：
- `GenerateGenericKernelMain()` — 作为唯一入口
- `HandleBlackholeBuiltin()` — 处理所有 builtin 的分发逻辑
- 所有 `Print*()` 方法 — builtin 到 C++ 的翻译

**修改 AddFunction**：
```cpp
void CodeGenBlackhole::AddFunction(const GlobalVar& gvar, const PrimFunc& f) {
  // 1. 读 attrs 确定 core_type（不再从函数名猜测）
  auto core_type_str = f->GetAttr<String>("blackhole.core_type");
  CoreType core_type = core_type_str ? ParseCoreType(core_type_str.value())
                                     : CoreType::kBRISC;  // 默认 BRISC

  // 2. 生成头文件（每个函数独立，不用 static 变量）
  EmitHeaders(core_type);

  // 3. 统一走 GenericKernelMain
  GenerateGenericKernelMain(f, func_name);
}
```

**头文件生成改为实例级别**：
```cpp
void EmitHeaders(CoreType core_type) {
  if (headers_emitted_) return;  // 实例变量，不是 static
  headers_emitted_ = true;

  decl_stream << "#include <cstdint>\n";
  switch (core_type) {
    case CoreType::kBRISC:
    case CoreType::kNCRISC:
      decl_stream << "#include \"dataflow_api.h\"\n";
      break;
    case CoreType::kTRISC:
      decl_stream << "#include \"compute_kernel_api.h\"\n";
      decl_stream << "#include \"compute_kernel_api/matmul.h\"\n";
      break;
  }
}
```

### 4.8 Runtime Module（分阶段设计）

**阶段1（当前，保持不变）**：外部进程模式
- 继续用 `blackhole_module.cc` + `tilelang_blackhole_runner`
- 足够用于 TT-Sim 验证
- 但需要修改：传递 CB 配置信息（当前缺失）

**阶段2（中期）**：C 接口桥接层

在 tt_metal_repo 中创建 C 接口：
```c
// tt_metal_c_api.h
extern "C" {
  void* ttml_create_device(int device_id);
  void  ttml_destroy_device(void* device);
  void* ttml_create_program();
  void  ttml_create_cb(void* program, int core_x, int core_y,
                       int cb_id, int num_pages, int page_size, int data_format);
  void* ttml_create_kernel(void* program, const char* kernel_path,
                           int core_x, int core_y, int processor_type);
  void  ttml_set_runtime_args(void* program, void* kernel,
                              int core_x, int core_y,
                              const uint32_t* args, int num_args);
  void  ttml_enqueue(void* device, void* program, bool blocking);
  void* ttml_create_dram_buffer(void* device, size_t size);
  void  ttml_write_buffer(void* device, void* buffer, const void* data, size_t size);
  void  ttml_read_buffer(void* device, void* buffer, void* data, size_t size);
  void  ttml_free_buffer(void* device, void* buffer);
}
```

TileLang 通过 dlopen 加载 `libtt_metal_capi.so`，完全避免编译时依赖。

**阶段3（长期）**：直接集成
- 当依赖管理问题解决后，重新激活 `blackhole_module_direct.cc`

### 4.9 完整数据流示例（GEMM E2E）

```
输入 DSL:
  @T.prim_func
  def matmul(A: T.Buffer([M,K], "float16"),
             B: T.Buffer([K,N], "float16"),
             C: T.Buffer([M,N], "float32")):
      with T.Kernel(M//32, N//32, threads=128) as (bx, by):
          A_s = T.alloc_shared([32, 32], "float16")
          B_s = T.alloc_shared([32, 32], "float16")
          C_l = T.alloc_fragment([32, 32], "float32")
          T.clear(C_l)
          for k in range(K//32):
              T.copy(A[by*32:, k*32:], A_s)
              T.copy(B[k*32:, bx*32:], B_s)
              T.gemm(A_s, B_s, C_l)
          T.copy(C_l, C[by*32:, bx*32:])

After LowerAndLegalize + OptimizeForTarget:
  -> TIR 包含 buffer_store/buffer_load 和高层 op 调用

After LowerBlackholeOps:
  -> T.copy(A, A_s) 变为: cb_reserve_back(0,1) + noc_async_read(...) + cb_push_back(0,1)
  -> T.copy(B, B_s) 变为: cb_reserve_back(1,1) + noc_async_read(...) + cb_push_back(1,1)
  -> T.clear(C_l)   变为: tile_regs_acquire()
  -> T.gemm(...)     变为: mm_init + matmul_tiles loop + pack_tile + cb_push_back(16,1)
  -> T.copy(C_l, C)  变为: cb_wait_front(16,1) + noc_async_write(...) + cb_pop_front(16,1)
  -> func attrs 包含 CB 需求: {in0:cb0, in1:cb1, out:cb16}

After PlanBlackholeCB:
  -> 验证 total L1 < 1.5MB
  -> attrs["blackhole.cb_configs"] = [
       {cb_id:0, page_size:2048, num_pages:2, format:"Float16"},
       {cb_id:1, page_size:2048, num_pages:2, format:"Float16"},
       {cb_id:16, page_size:4096, num_pages:1, format:"Float32"},
     ]

After AssignBlackholeCores:
  -> attrs["blackhole.grid_shape"] = [M//32, N//32]
  -> attrs["blackhole.work_per_core"] = ceil((M//32)*(N//32) / 140)

CodeGenBlackhole:
  -> 读取 attrs
  -> 生成 kernel_main() 包含所有 builtin calls
  -> 输出单个 .cpp 文件（合并 R/C/W，运行在 BRISC 上）

BlackholeModule.Execute():
  -> 从 attrs 创建 CB 配置
  -> 保存 kernel .cpp 到临时目录
  -> 调用 tilelang_blackhole_runner 执行
  -> 读取结果，对比 PyTorch 参考
```

---

## 五、实施路线图（修订版）

### 优先级 P0：让 Pipeline 真正连通

| # | 任务 | 文件 | 预期工作量 |
|---|------|------|-----------|
| 1 | 修改 lower.py 接入 Blackhole Pass | `tilelang/engine/lower.py` | 小 |
| 2 | 实现 LowerBlackholeOps copy 序列 | `src/transform/lower_blackhole_ops.cc` | 中 |
| 3 | 实现 PlanBlackholeCB (MVP 版) | `src/transform/plan_blackhole_cb.cc` | 中 |
| 4 | 修复 AssignBlackholeCores 结果存储 | `src/transform/assign_blackhole_cores.cc` | 小 |
| 5 | CodeGen 删除硬编码路径 | `src/target/codegen_blackhole.cc` | 中 |
| 6 | Copy kernel E2E 验证 | 测试文件 | 中 |

**目标**：`T.copy(A, B)` 能从 DSL 到 TT-Sim 执行通过

### 优先级 P1：GEMM E2E

| # | 任务 | 文件 | 预期工作量 |
|---|------|------|-----------|
| 7 | LowerBlackholeOps matmul 与 copy 集成 | `lower_blackhole_ops.cc` | 中 |
| 8 | Runtime 传递 CB 配置到 runner | `blackhole_module.cc` | 中 |
| 9 | GEMM kernel E2E 验证 | 测试文件 | 大 |

### 优先级 P2：长期优化

| # | 任务 | 说明 |
|---|------|------|
| 10 | C 接口桥接层 | 替代外部进程模式 |
| 11 | 三核拆分（可选） | 性能优化时再考虑 |
| 12 | 多核并行 | 从单核扩展到 140 核 |

---

## 六、关键决策记录

| 决策 | 选择 | 理由 |
|------|------|------|
| Pass 顺序 | LowerOps → PlanCB → AssignCores | LowerOps 生成 CB 信息供 PlanCB 使用 |
| 是否拆分三核 | 短期不拆，长期可选 | TT-Sim NCRISC 限制，合并方案已验证可行 |
| Runtime 方案 | 短期外部进程，中期 C 桥接，长期直连 | 渐进式降低风险 |
| CodeGen 入口 | 统一 GenericKernelMain | 消除硬编码路径，可维护性 |
| CB ID 分配 | 0-15 输入，16-31 输出 | 与 TT-Metal 官方示例一致 |
| 算子识别方式 | Op 对象比较代替字符串 find | 健壮性 |
| Core type 检测 | IR attrs 代替函数名子串 | 正确性 |

---

## 七、文档和进度修正

### 进度修正

| 任务 | 当前标记 | 应修正为 | 说明 |
|------|---------|---------|------|
| SplitBlackholeKernel | ✅ 完成 | ⏸️ 搁置（P2 优化项） | Stub，短期不需要 |
| PlanBlackholeCB | ✅ 完成 | ❌ 未实现 | Stub，P0 需补全 |
| AssignBlackholeCores | ✅ 完成 | 🔄 60%（逻辑正确，结果未存储） | P0 需修复 |
| LowerBlackholeOps | ✅ 完成 | 🔄 40%（matmul✅ copy❌ clear❌） | P0 需补全 |
| CodeGen | 🔄 进行中 | 🔄 50%（builtin visitor✅ 统一入口❌） | P0 需重构 |
| BlackholeModule | 🔄 进行中 | 🔄 30%（外部进程基本可用） | P1 需完善 |

### 架构文档修正

`arch_design.md` 需要更新：
1. Pass 顺序调整为 `LowerOps → PlanCB → AssignCores`
2. 移除 SplitBlackholeKernel 作为必需 Pass
3. 增加"短期合并 R/C/W 到 BRISC"的决策说明
4. Runtime 方案从"直接链接"改为"分阶段：外部进程 → C 桥接 → 直连"

---

*本文档基于对 Phase 0-3 所有设计文档和实际代码实现（逐行审阅关键文件）的全面审查*
*2026-03-17 v2*
