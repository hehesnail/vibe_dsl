# TileLang Blackhole 后端开发进度

> 当前唯一设计文档: `tasks/dev_design/final_blackhole_backend_redesign.md`
> 说明: 本文档只维护阶段状态和任务流转，不再承载任何总体架构设计。

## 当前阶段

- **阶段**: Stage 2 single-core pass integration
- **日期**: 2026-03-18
- **目标**: 在已完成的 single-core copy 执行闭环之上，先把 copy/gemm 统一迁回 pass/schema，再推进 single-core copy + gemm true E2E

## 当前判断

- pass 已接入 lowering pipeline。
- `spec.json -> runner` 的最小 single-core copy 已在 TT-Sim 上真实执行通过。
- `BlackholeModule` 从 Python 直接调用 packed func 已在 TT-Sim 上跑通 single-core copy，Stage 1 主调用面已收口。
- Stage 2 不能让 copy 长期停留在 runtime 专用旁路，也不能继续为 gemm 复制类似特化，否则只能证明执行链路通，不能证明编译 pass 已打通。
- 当前不再把“能生成 kernel 字符串”视为阶段完成。

## 任务状态总览

| 阶段 | 任务 | 状态 | 备注 |
|------|------|------|------|
| Stage 0 | 统一 attrs 到 `blackhole.*` | 🔄 进行中 | `rt_mod_blackhole` 已切到 `blackhole.cb_configs/core_plan/target_mode` 主路径 |
| Stage 0 | 引入 `ExecutableSpec` | 🔄 进行中 | 已落头文件、extractor 和最小 runner JSON 对接 |
| Stage 0 | 重构 `rt_mod_blackhole` | 🔄 进行中 | 已抽取 Stage 0 spec，并生成最小 runtime arg schema |
| Stage 0 | 重构 `BlackholeModule` | 🔄 进行中 | 已开始写 `spec.json + input.bin + output.bin + kernel.cpp` |
| Stage 0 | 重写 runner 协议 | 🔄 进行中 | 已切到 `spec.json + input.bin + output.bin`，runner 构建入口已收回 `tilelang_repo/tools/blackhole_runner/` |
| Stage 1 | single-core copy 闭环 | ✅ 已完成 | spec-driven runner 与 `BlackholeModule` direct call 都已在 TT-Sim 上通过 |
| Stage 2 | single-core pass integration | 🔄 进行中 | 先迁 copy，再迁 gemm，最后做统一 true E2E |
| Stage 3 | multi-core runtime 调度 | ⏳ 未开始 | per-core args |

## 当前下一步

1. 新增并落实 Stage 2 设计：copy/gemm 关键执行语义都必须开始从 pass/segment 产物生成，而不是由 `rt_mod_blackhole` 猜。
2. 先让 copy 从 runtime 专用 emitter 迁向 pass-driven schema，再用同一套机制承接 gemm。
3. 在 copy/gemm 的 pass/schema 收口后，再推进 single-core copy + gemm true E2E。
4. 在 single-core pass integration 路径稳定后，再推进 multi-core runtime 调度。

## 最近更新

- 2026-03-18:
  - 新增 `tasks/dev_design/stage0_executable_spec_attr_alignment.md`
  - `blackhole_module.h` 已引入 `ExecutableSpec / KernelSpec / CorePlan` 骨架
  - `AssignBlackholeCores` 已输出 `blackhole.core_plan` 和默认 `blackhole.target_mode`
  - `rt_mod_blackhole` 已读取新 attr schema 并抽取 Stage 0 spec
  - `BlackholeModule` 已改为写 `spec.json + input.bin + output.bin + kernel.cpp`
  - runner 已改为读取 `spec.json` 并按 spec 建 CB / kernel / runtime args
 - runner 源码与构建入口已收敛到 `tilelang_repo/tools/blackhole_runner/`
  - 已新增顶层总控脚本 `scripts/build_blackhole_stack.sh`
  - `scripts/build_blackhole_runner.sh` 现会先 bootstrap `TT_METAL_HOME/build_Release`，再构建 runner
  - runner 构建仍由 TileLang 侧 standalone CMake 管理，不再要求修改 `tt_metal_repo` 源码
  - `scripts/setup_tt_sim.sh` 已补 `TT_METAL_RUNTIME_ROOT`
  - 已通过 `./scripts/build_blackhole_runner.sh` 完成：
    - `metal_example_add_2_integers_in_riscv` 编译
    - TT-Sim smoke test
    - `tilelang_blackhole_runner` 编译
  - 新增 `tasks/dev_design/stage1_single_core_copy_closure.md`
  - `LowerBlackholeOps` 已为纯 copy PrimFunc 写出 `blackhole.target_mode = "single_core_copy"`
  - `rt_mod_blackhole` 已为 `single_core_copy` 生成最小 TT-Sim 兼容 copy kernel，并切到 `input/output_buffer_addr32 + tile_count + scratch_l1_buffer_addr32` schema
  - runner 已支持 `scratch_l1_buffer_addr32` 并按 schema 自动分配 L1 scratch buffer
  - `testing/python/target/blackhole/test_blackhole_e2e.py::test_blackhole_true_e2e` 已切到 `spec.json + input.bin + output.bin` 新协议
  - 已通过 `pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_e2e.py -k true_e2e -s` 在 TT-Sim 上跑通 single-core copy，结果与 PyTorch reference 一致
  - `BlackholeModule` 已修正 packed buffer 参数提取，不再把 `void_args` 误解为 `DLTensor*`
  - 已新增 `test_blackhole_module_direct_call`
  - 已通过 `pytest -q tilelang_repo/testing/python/target/blackhole/test_blackhole_e2e.py -k 'true_e2e or module_direct_call' -s` 验证：
    - `spec.json -> runner` single-core copy 继续通过
    - Python 侧直接调用 `artifact.codegen_mod["main"](...)` 已在 TT-Sim 上通过
  - 新增 `tasks/dev_design/stage2_single_core_pass_integration.md`
  - 已明确 Stage 2 不再接受“runtime 特化 copy/gemm 后跑通”作为阶段完成标准
