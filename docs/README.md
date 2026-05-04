# Source Analysis Documents

This directory contains background source-analysis notes for TileLang,
TT-Metal, and related tooling.  It is not the current Blackhole task board and
does not define the active backend design.

Use these entries for Blackhole work:

- current status / active task / verification:
  `../tasks/progress.md`
- overall Blackhole backend design:
  `../tasks/dev_design/final_blackhole_backend_redesign.md`
- active design index:
  `../tasks/dev_design/README.md`
- TT-Sim environment entry:
  `ttsim_setup.md`

## Contents

| Directory | Role |
| --- | --- |
| `tilelang/` | TileLang source-analysis notes for the Python DSL, JIT, C++ core, examples, testing, and benchmarks. |
| `tt_metal/` | TT-Metal / TT-Metalium background notes and source-analysis references. |
| `skills/` | Agent / workflow analysis notes. |

## Maintenance

- Do not put current task state, blockers, or verification logs here.
- Do not use these background notes as protocol owner truth when a
  `tasks/dev_design/` contract exists.
- If a source-analysis note affects active Blackhole design, update the
  relevant task design or `tasks/progress.md` instead of relying on this
  directory as a hidden status channel.
