# 通用开发模式与最佳实践

## 编译器后端开发模式

### 1. 代码生成器设计

**模式**: 继承 `CodeGenC` 基类，重写关键方法

```cpp
class CodeGenTileLangXXX final : public CodeGenC {
public:
  void PrintFuncPrefix(std::ostream &os) final;
  void PrintType(DataType t, std::ostream &os) final;
  void VisitStmt_(const ForNode *op) final;
  // ... 其他重写方法
};
```

**关键方法**:
- `PrintFuncPrefix`: 函数前缀（如 `__global__`）
- `PrintType`: 数据类型映射
- `VisitStmt_`: 语句处理
- `VisitExpr_`: 表达式处理

---

### 2. 类型系统映射

**常见陷阱**: 特殊浮点类型（FP8/FP6/FP4）的向量类型处理

**经验**:
- 先支持标量类型，再扩展向量类型
- 使用 `type.lanes()` 获取向量宽度
- 注意对齐要求（通常 16 字节对齐）

---

### 3. 内存 Scope 映射

**模式**: 通过 `PrintStorageScope` 方法映射内存层级

```cpp
void PrintStorageScope(const std::string &scope, std::ostream &os) {
  if (scope == "shared") {
    os << "__shared__";
  } else if (scope == "local") {
    // 寄存器变量，无需修饰
  }
}
```

---

### 4. 头文件管理

**策略**: 按需包含，使用标志位控制

```cpp
private:
  bool enable_fp16_ = false;
  bool need_math_constants_h_ = false;

public:
std::string Finish() {
  if (enable_fp16_) {
    decl_stream << "#include <cuda_fp16.h>\n";
  }
  return CodeGenC::Finish();
}
```

---

### 5. 调试技巧

**生成代码检查**:
- 使用 `BuildTileLangXXXWithoutCompile` 仅生成代码不编译
- 检查生成的代码是否符合目标平台语法

**常见问题**:
- 关键字冲突：使用 `ReserveKeywordsAsUnique_` 保留关键字
- 类型不匹配：检查 `PrintType` 实现
- 语法错误：对比参考生成的 CUDA/HIP 代码

---

### 6. 测试策略

**层级测试**:
1. 单元测试：代码生成器各方法独立测试
2. 集成测试：完整编译流程测试
3. 端到端测试：实际执行验证结果正确性

**快速迭代**:
- 先实现 WithoutCompile 版本，验证代码生成
- 再实现完整编译流程

---

## 项目特定模式

### TT-Metal 后端开发注意事项

#### 待补充

随着开发深入，记录 TT-Metal 特有的：
- Kernel 签名规范
- 内存模型细节
- 编译和加载流程
- 调试技巧

---

## 代码审查清单

提交代码前检查：
- [ ] 遵循 TVM 命名规范
- [ ] 关键逻辑有注释
- [ ] 已测试代码生成功能
- [ ] 无调试用的 print/printf 残留
- [ ] 错误处理完善
