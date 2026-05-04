# TT-Metal Source Analysis

This directory contains background notes for Tenstorrent TT-Metal /
TT-Metalium.  Use it as a reference while working on Blackhole backend
contracts, not as the active task board.

For active Blackhole design and status, read:

- `../../tasks/dev_design/final_blackhole_backend_redesign.md`
- `../../tasks/dev_design/README.md`
- `../../tasks/progress.md`

## Contents

| Directory | Role |
| --- | --- |
| `guide/` | TT-Metalium usage and API guide. |
| `source_analysis/` | Source-analysis notes for TT-Metal implementation modules. |

## Source-Analysis Index

- `api/`: host API interfaces and implementation.
- `impl/`: allocator, dispatch, program, and other core implementation pieces.
- `llrt/`: low-level runtime, HAL, firmware, and cluster management.
- `hw/`: hardware abstraction, firmware, compute kernels, and registers.
- `distributed/`: distributed execution and mesh workload references.
- `fabric/`: inter-chip communication and collectives.
- `jit_build/`: JIT build system and kernel cache.
- `tools/`: profiling and debug tooling.

## Maintenance

- Do not maintain current Blackhole status, blockers, or verification logs
  here.
- Do not treat these notes as protocol owner truth when an active
  `tasks/dev_design/` contract exists.
- If a TT-Metal finding changes the Blackhole architecture contract, update
  the relevant design document.
