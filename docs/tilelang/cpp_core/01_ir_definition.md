# TileLang C++ 核心 - IR 定义与配置系统

## 模块概述

TileLang 的 IR 定义模块位于 `src/ir.cc`，它扩展了 TVM 的脚本前端（TVM Script Frontend），提供了 TileLang 特有的 IR 节点和框架。该模块定义了内核启动、并行循环、流水线循环、持久化循环以及 Warp 特化等核心概念。

配置系统位于 `src/config.h`，提供了编译时配置选项的访问接口。

---

## IR 节点详解

### 1. KernelLaunchFrame - 内核启动框架

**文件**: `/root/dev/vibe_dsl/tilelang/src/ir.cc:217-259`

`KernelLaunchFrame` 表示一个内核启动的框架，管理多个 TIR 框架的嵌套作用域。

```cpp
class KernelLaunchFrameNode : public TIRFrameNode {
public:
  Array<TIRFrame> frames;

  void EnterWithScope() final;
  void ExitWithScope() final;
};
```

**功能**:
- 管理 GPU/CPU 内核启动时的线程层次结构
- 对于 GPU：创建 blockIdx.x/y/z 和 threadIdx.x/y/z 环境线程
- 对于 CPU：创建网格循环变量

**内核启动逻辑** (`/root/dev/vibe_dsl/tilelang/src/ir.cc:261-327`):

```cpp
KernelLaunchFrame KernelLaunch(const Array<PrimExpr> &grid_size,
                               const Optional<Array<PrimExpr>> &block_size_opt,
                               const Map<String, ffi::Any> &attrs) {
  // GPU 内核：创建 blockIdx 和 threadIdx 线程
  // CPU 内核：创建网格循环变量
}
```

### 2. 并行循环 (ParallelFor)

**文件**: `/root/dev/vibe_dsl/tilelang/src/ir.cc:62-102`

`ParallelFor` 创建并行执行的循环结构，支持多维扩展。

```cpp
ForFrame ParallelFor(const Array<PrimExpr> &extents,
                     const Map<String, tvm::ffi::Any> &annotations);
```

**关键特性**:
- 支持多维循环并行化
- 注释仅附加到最外层循环（设计决策：外层循环可以管理和转换整个嵌套区域）
- 使用 `ForKind::kParallel` 标记并行循环

### 3. 流水线循环 (PipelinedFor)

**文件**: `/root/dev/vibe_dsl/tilelang/src/ir.cc:104-137`

`PipelinedFor` 实现软件流水线优化，用于隐藏内存访问延迟。

```cpp
ForFrame PipelinedFor(PrimExpr start, const PrimExpr &stop, int num_stages,
                      const Array<PrimExpr> &order,
                      const Array<PrimExpr> &stages,
                      const Array<Array<PrimExpr>> &sync,
                      const Array<Array<PrimExpr>> &groups);
```

**支持的注解**:
- `num_stages`: 流水线阶段数
- `tl_pipeline_order`: 流水线顺序
- `tl_pipeline_stage`: 流水线阶段分配
- `tl_pipeline_group`: 流水线分组

### 4. 持久化循环 (PersistentFor)

**文件**: `/root/dev/vibe_dsl/tilelang/src/ir.cc:139-210`

`PersistentFor` 实现持久化内核模式，允许内核在多个工作项之间保持状态。

```cpp
ForFrame PersistentFor(const Array<PrimExpr> &domain, const PrimExpr &wave_size,
                       const PrimExpr &index, PrimExpr group_size);
```

**实现细节**:
- 计算波次数量：`waves = ceildiv(domain_size, wave_size)`
- 支持分组处理
- 生成条件退出逻辑（`loop_break` 调用）

### 5. Warp 特化框架 (WarpSpecialize)

**文件**: `/root/dev/vibe_dsl/tilelang/src/ir.cc:338-417`

`WarpSpecialize` 允许针对特定 Warp 组进行代码特化。

```cpp
WarpSpecializeFrame WarpSpecialize(const Array<IntImm> &warp_group_ids,
                                   const PrimExpr &thread_idx,
                                   int warp_group_size = 128);
```

**功能**:
- 合并连续的 Warp 组
- 生成条件判断逻辑
- 设置 `warp_specialize` 属性

---

## 辅助函数

### 环境线程创建

**文件**: `/root/dev/vibe_dsl/tilelang/src/ir.cc:25-38`

```cpp
static Var CreateEnvThread(String name, String thread_tag, DataType dtype);
```

创建环境线程变量（如 `blockIdx.x`, `threadIdx.x`）。

### 迭代变量框架

**文件**: `/root/dev/vibe_dsl/tilelang/src/ir.cc:40-60`

```cpp
static ForFrame MakeIterVarFrame(const std::string &name, const PrimExpr &dom);
```

创建单维度的迭代变量框架。

---

## 配置系统

**文件**: `/root/dev/vibe_dsl/tilelang/src/config.h`

配置系统提供了对 TVM PassContext 中 TileLang 特定配置的访问。

### 可用配置

| 配置键 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `tl.enable_vectorize_planner_verbose` | Bool | false | 启用向量化规划器的详细输出 |
| `tl.disable_vectorize_256` | Bool | false | 禁用 256 位向量化 |

### 配置访问接口

```cpp
namespace tl_config {
  // 检查是否启用了向量化规划器详细输出
  inline bool VectorizePlannerVerboseEnabled();

  // 检查是否禁用了 256 位向量化
  inline bool Vectorize256Disabled();
}
```

**实现机制**:
```cpp
inline bool VectorizePlannerVerboseEnabled() {
  auto ctxt = transform::PassContext::Current();
  return ctxt
      ->GetConfig("tl.enable_vectorize_planner_verbose", Optional<Bool>())
      .value_or(Bool(false));
}
```

---

## Python 绑定

**文件**: `/root/dev/vibe_dsl/tilelang/src/ir.cc:329-336, 419-426`

使用 TVM FFI 反射系统注册 Python 可调用的函数：

```cpp
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.Parallel", ParallelFor)
      .def("tl.Pipelined", PipelinedFor)
      .def("tl.Persistent", PersistentFor)
      .def("tl.KernelLaunch", KernelLaunch)
      .def("tl.WarpSpecialize", WarpSpecialize)
      .def("tl.SideEffect", SideEffect);
}
```

---

## 反射注册

**文件**: `/root/dev/vibe_dsl/tilelang/src/ir.cc:221-225, 342-346, 419-426`

各个框架节点注册反射信息以支持序列化和 Python 访问：

```cpp
static void RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<KernelLaunchFrameNode>()
      .def_ro("frames", &KernelLaunchFrameNode::frames);
}
```

---

## 代码引用汇总

| 组件 | 文件路径 | 行号范围 |
|------|----------|----------|
| KernelLaunchFrameNode | `/root/dev/vibe_dsl/tilelang/src/ir.cc` | 217-259 |
| KernelLaunch | `/root/dev/vibe_dsl/tilelang/src/ir.cc` | 261-327 |
| ParallelFor | `/root/dev/vibe_dsl/tilelang/src/ir.cc` | 62-102 |
| PipelinedFor | `/root/dev/vibe_dsl/tilelang/src/ir.cc` | 104-137 |
| PersistentFor | `/root/dev/vibe_dsl/tilelang/src/ir.cc` | 139-210 |
| WarpSpecializeFrameNode | `/root/dev/vibe_dsl/tilelang/src/ir.cc` | 338-375 |
| WarpSpecialize | `/root/dev/vibe_dsl/tilelang/src/ir.cc` | 377-417 |
| 配置系统 | `/root/dev/vibe_dsl/tilelang/src/config.h` | 1-39 |
| FFI 绑定 | `/root/dev/vibe_dsl/tilelang/src/ir.cc` | 329-336, 419-426 |
