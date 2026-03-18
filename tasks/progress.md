# TileLang Blackhole 后端开发进度

> 当前唯一设计文档: `tasks/dev_design/final_blackhole_backend_redesign.md`
> 说明: 本文档只维护阶段状态和任务流转，不再承载任何总体架构设计。

## 当前阶段

- **阶段**: Stage 0 协议重构
- **日期**: 2026-03-18
- **目标**: 建立 `ExecutableSpec`、统一 attrs、替换旧 runner 协议

## 当前判断

- pass 已接入 lowering pipeline。
- runtime、codegen、runner 仍未形成真实执行闭环。
- 当前不再把“能生成 kernel 字符串”视为阶段完成。

## 任务状态总览

| 阶段 | 任务 | 状态 | 备注 |
|------|------|------|------|
| Stage 0 | 统一 attrs 到 `blackhole.*` | 🔄 进行中 | `rt_mod_blackhole` 已切到 `blackhole.cb_configs/core_plan/target_mode` 主路径 |
| Stage 0 | 引入 `ExecutableSpec` | 🔄 进行中 | 已落头文件、extractor 和最小 runner JSON 对接 |
| Stage 0 | 重构 `rt_mod_blackhole` | 🔄 进行中 | 已抽取 Stage 0 spec，并生成最小 runtime arg schema |
| Stage 0 | 重构 `BlackholeModule` | 🔄 进行中 | 已开始写 `spec.json + input.bin + output.bin + kernel.cpp` |
| Stage 0 | 重写 runner 协议 | 🔄 进行中 | 已切到 `spec.json + input.bin + output.bin`，runner 构建入口已收回 `tilelang_repo/tools/blackhole_runner/` |
| Stage 1 | single-core copy 闭环 | ⏳ 未开始 | true E2E |
| Stage 2 | single-core gemm 闭环 | ⏳ 未开始 | true E2E |
| Stage 3 | multi-core runtime 调度 | ⏳ 未开始 | per-core args |

## 当前下一步

1. 让 lowering/segment 生成真实的 `kernels[].runtime_args` 与 `target_mode`，去掉当前保守默认值。
2. 扩展 runner 和 `BlackholeModule` 到多 kernel、multi-core、per-core runtime args。
3. 基于新协议做 single-core copy 的真实 E2E 验证。
4. 之后再做 gemm 的真实执行闭环。

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
