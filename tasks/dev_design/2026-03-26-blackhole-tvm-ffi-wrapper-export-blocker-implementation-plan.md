# Blackhole `tvm_ffi` Wrapper/Export Blocker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 Blackhole `tilelang.compile(..., execution_backend="tvm_ffi")` wrapper/export 主链，避免生成非法 host shim，并保证 direct host path 不回归。

**Architecture:** 先用最小 Blackhole case 锁定 `tvm_ffi -> export_library -> host shim` 的失败点，再在 `SplitHostDevice` 主链修正 host/device return contract 与 host shim error-propagation 的一致性。修复只落在正式 split/export 主链，不新增 Blackhole 专用旁路，不改 direct path runtime 协议。

**Tech Stack:** TileLang Python JIT/export path, TVM TIR passes, C++ `SplitHostDevice`, pytest, TT-Metal Blackhole target tests

---

### Task 1: 建立最小复现并固化成测试

**Files:**
- Create: `tilelang_repo/testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py`
- Modify: `tilelang_repo/tilelang/jit/kernel.py`
- Test: `tilelang_repo/testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py`

- [ ] **Step 1: 写最小失败测试**

```python
import pathlib

import pytest
import tilelang
from tilelang import language as T


def _make_minimal_copy():
    @T.prim_func
    def main(
        A: T.Buffer((32, 32), "float16"),
        B: T.Buffer((32, 32), "float16"),
    ):
        with T.Kernel(1, 1, threads=1):
            for i, j in T.grid(32, 32):
                B[i, j] = A[i, j]

    return main


def test_blackhole_tvm_ffi_export_generates_valid_host_shim(tmp_path: pathlib.Path):
    kernel = tilelang.compile(
        _make_minimal_copy(),
        target="blackhole",
        execution_backend="tvm_ffi",
    )
    out = tmp_path / "blackhole_export.so"
    kernel.export_library(str(out))
    assert out.exists()
```

- [ ] **Step 2: 跑测试确认当前失败**

Run: `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py -k valid_host_shim -v`

Expected: FAIL，失败栈落在 `export_library` 或其内部编译阶段；如果有中间产物，`lib0.c` 可见非法 `kernel_error_code = ;`

- [ ] **Step 3: 为调试暴露 host shim 临时目录**

```python
def export_library(self, kernel_file: str) -> None:
    if self.artifact.rt_mod is None:
        raise ValueError(
            'Runtime module is not available. Please compile the kernel with `execution_backend="tvm_ffi"` before exporting.'
        )
    debug_dir = os.environ.get("TILELANG_DEBUG_EXPORT_DIR")
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        self.artifact.rt_mod.export_library(kernel_file, workspace_dir=debug_dir)
        return
    self.artifact.rt_mod.export_library(kernel_file)
```

- [ ] **Step 4: 重跑测试并保存失败产物**

Run: `TILELANG_DEBUG_EXPORT_DIR=/tmp/tl_bh_export_debug pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py -k valid_host_shim -v`

Expected: 仍然 FAIL，但 `/tmp/tl_bh_export_debug` 下可检查 `lib0.c`

- [ ] **Step 5: 提交**

```bash
git add tilelang_repo/testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py tilelang_repo/tilelang/jit/kernel.py
git commit -m "test: add blackhole tvm_ffi export reproducer"
```

### Task 2: 锁定 split 主链上的 return-contract 错位

**Files:**
- Modify: `tilelang_repo/src/transform/split_host_device.cc`
- Test: `tilelang_repo/testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py`

- [ ] **Step 1: 先写结构性断言，证明 split 后 contract 与 host shim 生成条件不一致**

```python
def test_blackhole_tvm_ffi_export_no_invalid_kernel_error_code(tmp_path: pathlib.Path, monkeypatch):
    debug_dir = tmp_path / "export_debug"
    monkeypatch.setenv("TILELANG_DEBUG_EXPORT_DIR", str(debug_dir))
    kernel = tilelang.compile(
        _make_minimal_copy(),
        target="blackhole",
        execution_backend="tvm_ffi",
    )
    with pytest.raises(Exception):
        kernel.export_library(str(tmp_path / "broken.so"))
    host_c = (debug_dir / "lib0.c").read_text()
    assert "kernel_error_code = ;" not in host_c
```

- [ ] **Step 2: 跑测试，确认当前确实命中坏 host shim**

Run: `TILELANG_DEBUG_EXPORT_DIR=/tmp/tl_bh_export_debug pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py -k invalid_kernel_error_code -v`

Expected: FAIL，断言显示 `lib0.c` 中存在 `kernel_error_code = ;`

- [ ] **Step 3: 在 `SplitHostDevice` 中把 host shim 生成条件绑定到合法 return contract**

```cpp
bool can_propagate_errors = [&]() {
  auto kind = device_target->GetTargetDeviceType();
  return kind == kDLCPU || kind == kDLExtDev || kind == kDLHexagon;
}();

bool emit_error_check = can_propagate_errors && kernel_ret_type.as<PrimTypeNode>();

if (emit_error_check) {
  tir::Var kernel_error_code("kernel_error_code", DataType::Int(32));
  tir::Call kernel_call(DataType::Int(32), kernel_symbol_global, args);
  tir::AssertStmt assert_success(
      kernel_error_code == IntImm(DataType::Int(32), 0),
      tir::StringImm("Error executing compute kernel"), tir::Evaluate(0));
  return tir::LetStmt(kernel_error_code, kernel_call, assert_success);
}

return tir::Evaluate(tir::Call(DataType::Void(), kernel_symbol_global, args));
```

- [ ] **Step 4: 如 Step 3 不能单独闭环，继续把判断条件改成“device func 的实际返回类型”，不是 target name**

```cpp
const bool returns_status_code =
    kernel_ret_type.as<PrimTypeNode>() &&
    kernel_ret_type == PrimType(DataType::Int(32));

if (returns_status_code) {
  ...
} else {
  ...
}
```

- [ ] **Step 5: 重跑复现测试**

Run: `TILELANG_DEBUG_EXPORT_DIR=/tmp/tl_bh_export_debug pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py -k "valid_host_shim or invalid_kernel_error_code" -v`

Expected: PASS；`lib0.c` 不再包含 `kernel_error_code = ;`

- [ ] **Step 6: 提交**

```bash
git add tilelang_repo/src/transform/split_host_device.cc tilelang_repo/testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py
git commit -m "fix: repair blackhole tvm_ffi host shim generation"
```

### Task 3: 验证 export 主链确实闭环

**Files:**
- Modify: `tilelang_repo/testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py`
- Test: `tilelang_repo/testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py`

- [ ] **Step 1: 把“坏代码不存在”升级为“export 真成功”的稳定断言**

```python
def test_blackhole_tvm_ffi_export_generates_valid_host_shim(tmp_path: pathlib.Path, monkeypatch):
    debug_dir = tmp_path / "export_debug"
    monkeypatch.setenv("TILELANG_DEBUG_EXPORT_DIR", str(debug_dir))
    kernel = tilelang.compile(
        _make_minimal_copy(),
        target="blackhole",
        execution_backend="tvm_ffi",
    )
    out = tmp_path / "blackhole_export.so"
    kernel.export_library(str(out))
    assert out.exists()
    host_c = (debug_dir / "lib0.c").read_text()
    assert "kernel_error_code = ;" not in host_c
```

- [ ] **Step 2: 跑完整新测试文件**

Run: `TILELANG_DEBUG_EXPORT_DIR=/tmp/tl_bh_export_debug pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py -v`

Expected: PASS

- [ ] **Step 3: 提交**

```bash
git add tilelang_repo/testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py
git commit -m "test: verify blackhole tvm_ffi export path"
```

### Task 4: 回归 direct host path

**Files:**
- Test: `tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py`
- Test: `tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py`

- [ ] **Step 1: 跑 copy direct runtime 回归**

Run: `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_copy_runtime.py -v`

Expected: PASS 或在环境不满足时明确 SKIP，不出现新的 compile/export regression

- [ ] **Step 2: 跑 GEMM direct runtime 回归**

Run: `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_gemm.py -v`

Expected: PASS 或在环境不满足时明确 SKIP，不出现新的 compile/export regression

- [ ] **Step 3: 若任一回归失败，先定位是否由 `SplitHostDevice` 通用改动引入**

```text
检查点：
1. split 后 device func return type 是否变化
2. host shim 是否错误进入 int32-return 分支
3. Blackhole direct path 是否仍走 `ExecutableSpec -> BlackholeModule`
```

- [ ] **Step 4: 提交**

```bash
git add -A
git commit -m "test: regress blackhole direct path after tvm_ffi fix"
```

### Task 5: 同步文档与记录

**Files:**
- Modify: `tasks/progress.md`
- Modify: `memory/bugs.md`
- Modify: `tasks/dev_design/2026-03-26-blackhole-tvm-ffi-wrapper-export-blocker.md`

- [ ] **Step 1: 更新 progress**

```markdown
- `tilelang.compile(..., execution_backend="tvm_ffi")` 的 Blackhole wrapper/export blocker 已修复
- 最小 Blackhole export case 已通过
```

- [ ] **Step 2: 更新 bug 记录**

```markdown
### `tilelang.compile(..., execution_backend="tvm_ffi")` 的 Blackhole wrapper/export path 会生成非法 host shim

- **当前状态**: 已解决
- **根本原因**: host/device split 生成的 error-propagation 与实际 device return contract 不一致
- **解决**: 仅在合法 status-return contract 成立时生成 `kernel_error_code` 检查，否则生成合法 void host shim
```

- [ ] **Step 3: 回写设计文档状态**

```markdown
- **状态**: 已实施
```

- [ ] **Step 4: 提交**

```bash
git add tasks/progress.md memory/bugs.md tasks/dev_design/2026-03-26-blackhole-tvm-ffi-wrapper-export-blocker.md
git commit -m "docs: record blackhole tvm_ffi export fix"
```

### Task 6: 最终验证与交付

**Files:**
- Modify: `tasks/progress.md`

- [ ] **Step 1: 运行本轮最小必要验证集合**

Run: `pytest tilelang_repo/testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py -v`

Expected: PASS

- [ ] **Step 2: 确认无残留长命令或后台进程**

Run: `ps -ef | rg "pytest|ctest|ninja|cmake" || true`

Expected: 没有本轮残留后台任务，或只剩与本任务无关的用户进程

- [ ] **Step 3: 检查工作树**

Run: `git status --short`

Expected: 只有本轮预期改动；无意外生成物

- [ ] **Step 4: 最终提交**

```bash
git add -A
git commit -m "fix: restore blackhole tvm_ffi wrapper export path"
```

