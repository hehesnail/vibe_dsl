# TileLang Blackhole Backend Progress

> 当前 HEAD 看板只放状态、blocker、下一步和最近验证。
> 设计合同不要写在这里；设计依据看
> `tasks/dev_design/final_blackhole_backend_redesign.md`
> 和对应任务级设计文档。

## Status

- Date: `2026-04-30`
- Active lane: `Hardware-model-backed core and buffer placement`
- Main chain:
  `Normalized Tile TIR -> SpatialPlan -> TTProgram -> ExecutableSpec`

## Current HEAD

- Legacy external runner / `build_blackhole/` 已删除。
- Blackhole 正式执行路径是进程内 `BlackholeModule` direct host path；
  `tilelang.compile(..., execution_backend="tvm_ffi")`
  Blackhole wrapper/export path 可用。
- Broad legacy protocols 已退出 active chain：
  `compute_contract`,
  `gemm_contract`,
  `multi_*_contracts`,
  top-level `TTProgram.payload`,
  bridge attrs,
  lowering facts contract maps,
  compute-op seed maps,
  leaf name/default fallbacks。
- Tile compute truth 保持 TT-Metal leaf API 粒度。
  已删除 late scalar-loop matcher / generate family。
  Blackhole scalar-loop normalization 已从通用
  `lower_tile_op.cc`
  抽到独立 common normalizer；
  source projection 已从
  `PlanTTKernelABI`
  大接口面收窄到
  `BlackholeTileComputeSourceProjection`。
  当前实现继续收缩重复 lowering mechanics：
  normalizer 用 unary / binary leaf builders 生成同形 calls；
  source projection 用 binary / broadcast-cols / unary category emitters；
  row-reduction source emission 复用
  `ExactTileComputeEmitter`
  的 CB / tile-register / pack sequence。
- Algorithmic generalization foundation 已存在并在 admitted live-form /
  materialization 决策中使用：
  `AccessRegion`,
  graph-backed `SpatialPlan` dependence,
  `LiveValueSSA`,
  first TT live-form solver。
  这些不是 compute expression lowering 或全局 resource allocator。
- `TileComputeDAG`
  只允许作为 pass-local explicit-leaf graph legalization /
  covering model。
  它不能做 composite lowering、
  resource allocation、
  core placement、
  NoC scheduling
  或跨阶段 payload。
- Known composite pseudo-leaf source payload 已清理：
  `exp2_tile(mode, lhs, rhs, scale, ...)`
  改为 normalized explicit leaf sequence；
  `mul_tiles_bcast_cols("div", ...)`
  改为
  `recip_tile + mul_tiles_bcast_cols`。
  Source hooks 现在只能投影一个 selected semantic TT-Metal leaf op。
  Blackhole tile-compute normalizer 保留共享 leaf-call builder，
  但不再保留 rule registry /
  benefit table
  或 workload-pattern catalog；
  store normalization 只做 bounded root dispatch，
  并立即产出 explicit TT-Metal leaf statements。
- First typed resource-pressure surface 已存在：
  `TTResourceDemand`
  /
  `TTResourcePressureReport`
  进入 `TTProgram`，
  被 `ValidateTTProgram` 消费，
  并投影到 `ExecutableSpec`。
- CB / L1 admission 已接入第一版 hardware facts：
  `TTHardwareModel.max_cb_count`,
  worker L1 budget,
  L1 alignment,
  aligned CB bytes,
  allocator-managed L1 pressure。
  `ValidateTTProgram`
  会在 source / runtime emission 前 fail closed。
- Core group planning 已开始消费
  `TTHardwareModel`
  的 logical worker grid / functional worker count。
  `PlanTTCoreGroups`
  现在只 materialize hardware-grid 内的 physical cores，
  用 deterministic work packets 覆盖更大的 logical grid；
  `ValidateTTProgram`
  校验 core 坐标、重复 core、work packet membership
  和 harvested worker count。
- Blackhole runtime / codegen C++ audit 第一批已收口：
  host/generated scalar bitcast 改为 memcpy-style bit cast；
  direct runtime DLTensor raw transfer 要求 compact row-major layout；
  runtime leaf readers 对缺失 typed fields fail closed；
  device resource canonicalizer 删除跨对象 Var name fallback；
  Blackhole imported module 提供真实 non-empty bytes serialization
  和 `ffi.Module.load_from_bytes.blackhole` loader，
  同时保留 file export 的 fail-closed 行为。

## Current Blocker

Buffer distribution 仍然过粗：

- `TTBufferDistributionPlan`
  仍主要是
  `unit_mesh`
  /
  `replicated`。
- Wider runtime admission
  还需要 buffer placement
  继续消费
  `TTHardwareModel`
  facts，
  并在 source / runtime emission 前给出 typed reject。

## Next Task Order

1. Upgrade buffer placement:
   expand `TTBufferDistributionPlan`
   beyond `unit_mesh` / `replicated`,
   use `TTHardwareModel`
   for L1 / DRAM facts,
   and produce source/runtime-visible typed rejects before emission.
2. Resume wider runtime admission:
   multi-block flash-attn direct runtime,
   wider exact-CB events,
   mesh / distributed runtime,
   later NoC / multicast / scheduling optimization.

## Support Boundary

- Direct runtime admitted subset:
  copy equal source/dest range with stride 1;
  GEMM A/B-separated reader range plus writer output range;
  interleaved DRAM accessor with `common_runtime_arg_count = 0`;
  non-oversubscribed explicit semaphore / remote-endpoint subset;
  admitted bf16 live-form paths.
- Flash-attn admitted direct-runtime subset:
  small single-work-item and 32x32 MHA / GQA bf16.
- Flash-attn compile/source/spec stable but runtime-gated subset:
  seq64 / multi-K-step MHA and GQA.
- Not admitted:
  multi-block flash-attn direct-runtime correctness,
  larger multi-page exact-CB publish/consume events,
  full multi-device / sharded / fabric collective runtime.

## Latest Verification

Latest code implementation batch:
current HEAD after
modern C++ audit fixes for Blackhole runtime/codegen/typed plan readers
and hardware-model-backed core-group repair.

Verification for this batch:

- `cmake --build build -j32`
- `pytest -q testing/python/transform/test_blackhole_spatial_ir.py -k 'modern_cpp_audit'`
  (`4 passed, 82 deselected`)
- `pytest -q testing/python/transform/test_blackhole_spatial_ir.py`
  (`86 passed`)
- `source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh && export TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo && cd /root/dev/vibe_dsl/tilelang_repo && pytest -q testing/python/target/blackhole/test_blackhole_tvm_ffi_export.py testing/python/target/blackhole/test_blackhole_copy_runtime.py -k 'tvm_ffi_export or noncompact or empty_work_packets or large_shape_copy_keeps_per_core_l1_small'`
  (`5 passed, 11 deselected`)
- `source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh && export TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo && cd /root/dev/vibe_dsl/tilelang_repo && pytest -q testing/python/target/blackhole/test_blackhole_copy_pipeline.py -k 'leaf_readers_do_not_keep_legacy_defaults_or_slot_fallbacks or direct_runtime_materializes_compile_time_abi_schema or rejects_accessor_common_runtime_arg_count'`
  (`3 passed, 60 deselected`)
- `source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh && export TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo && cd /root/dev/vibe_dsl/tilelang_repo && pytest -q testing/python/target/blackhole/test_blackhole_copy_runtime.py`
  (`15 passed`)
- `source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh && export TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo && cd /root/dev/vibe_dsl/tilelang_repo && pytest -q testing/python/target/blackhole/test_blackhole_gemm.py`
  (`59 passed`)
- `git diff --check`
  (`passed`)
- Source scan found no active
  empty map leaf-reader fallback,
  default physical core fallback,
  empty `SaveToBytes`,
  resource Var name fallback,
  old host scalar aliasing bitcast,
  or stale `kBlackholeMaxCBs`.

Latest doc cleanup verification:

- Core active docs were reduced back to role-specific contracts:
  `final_blackhole_backend_redesign.md`,
  `README.md`,
  `progress.md`,
  `2026-04-28-blackhole-algorithmic-generalization.md`,
  `2026-04-28-blackhole-tile-compute-legalizer-dag-covering.md`,
  `2026-04-29-blackhole-resource-planning-roadmap.md`,
  and
  `2026-04-28-blackhole-lower-tile-op-normalizer-dedup.md`.
- `rg`
  stale-log scan over those docs found no lingering
  phase-log /
  stale production-completion wording.
- `git diff --check`
  passed.
- Docs-only cleanup;
  no build or pytest was required for this change.
