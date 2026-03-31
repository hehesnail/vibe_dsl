# 任务开发设计文档

> 当前唯一总体设计文档: `final_blackhole_backend_redesign.md`

## 1. 活动文档

| 文档 | 用途 | 状态 |
|------|------|------|
| `final_blackhole_backend_redesign.md` | 唯一总体设计 | 常青 |
| `stage2d_ttmetal_contract_audit.md` | TT-Metal contract 缺口审计 | 收正进行中（P0/P1/P2/P3 ✅，P4 已完成最小 interleaved stick/page path，P5 已推进到 worker semaphore producer/consumer E2E + remote-core descriptor formalization） |
| `stage4_backend_cleanup_roadmap.md` | 当前主链问题分档与收敛顺序 | 活动中 |
| `stage4_flash_attention_forward_subset.md` | 前向 Flash-Attention 语义子集设计 | 活动中（analysis 已接入，fragment/codegen 正在推进） |
| `2026-03-31-flash-attention-forward-subset-implementation-plan.md` | 前向 Flash-Attention 实施计划 | 活动中 |

## 2. 已完成（仍有参考价值）

| 文档 | 用途 | 完成日期 |
|------|------|---------|
| `stage3_multicore_design.md` | Stage 3 多核设计 | 2026-03-26 |
| `stage2e_blackhole_device_resource_semantics.md` | 设备资源 IR 语义扩展 | 2026-03-25 |
| `stage2d_cb_identity_protocol.md` | CB identity 唯一协议 | 2026-03-25 |
| `stage2d_gemm_direct_cb_io.md` | GEMM contract 修复 | 2026-03-26 |
| `stage2j_compute_contract_schema.md` | compute contract 正式化 | 2026-03-30 |
| `stage2g_unified_work_schema.md` | richer runtime work schema 正式化 | 2026-03-27 |
| `stage2h_accessor_schema.md` | accessor/common-runtime schema 正式化 | 2026-03-27 |
| `stage2i_compile_time_abi_schema.md` | compile-time ABI schema 正式化 | 2026-03-27 |
| `stage4_semaphore_schema.md` | P5 semaphore schema 预埋 | 2026-03-27 |
| `stage4_copy_stick_generalization.md` | P4 stick/page copy 泛化 | 2026-03-30 |
| `2026-03-26-stage2d-gemm-contract-implementation-plan.md` | GEMM contract 实施计划 | 2026-03-26 |
| `stage2d_gemm_integration.md` | GEMM 接入设计（Steps 1-5） | 2026-03-25 |
| `stage2c_annotate_blackhole_copy_semantics.md` | copy 语义 annotation | 2026-03-24 |
| `stage2_pass_reuse_matrix.md` | pass 复用矩阵 | 2026-03-23 |
| `stage2_single_core_pass_integration.md` | 分层目标与阶段拆分 | 2026-03-22 |
| `stage2_blackhole_logical_block_launch_plan.md` | block/core/memory plan | 2026-03-22 |

## 3. 历史记录

| 文档 | 用途 |
|------|------|
| `blackhole_architecture_review_and_action_plan.md` | 早期架构审查 |
| `2026-03-26-stage3-gemm-multicore-direct-hang.md` | Stage 3 GEMM 多核排障记录 |
| `2026-03-26-blackhole-tvm-ffi-wrapper-export-blocker.md` | `tvm_ffi` wrapper/export blocker 排障记录 |
| `stage0_executable_spec_attr_alignment.md` | Stage 0 协议落地 |
| `stage1_single_core_copy_closure.md` | Stage 1 copy 闭环 |
| `stage2_concrete_dev_task_plan.md` | 早期 Stage 2 任务拆解（已被后续设计取代） |
| `phase0_*.md` | 环境准备历史记录 |
