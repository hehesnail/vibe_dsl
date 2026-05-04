# GEMINI.md

本文件只保留 Gemini CLI 的入口说明。
仓库工作规范不要在这里复制维护。

## Read First

1. `AGENTS.md`
2. `tasks/dev_design/final_blackhole_backend_redesign.md`
3. `tasks/progress.md`
4. `tasks/dev_design/README.md`
5. Relevant active design docs listed by `tasks/dev_design/README.md`
6. `memory/general_dev.md` / `memory/bugs.md` when debugging or using TT-Sim

## Status

Do not infer current status from this file.
The only current status / blocker / next-task board is:

```text
tasks/progress.md
```

## Gemini-Specific Notes

- Follow `AGENTS.md` unless an explicit Gemini CLI safety rule conflicts.
- For complex non-local changes, form a design before editing.
- Do not stage or commit unless the user explicitly asks or the current
  workflow requires it.
- For TT-Sim, use the top-level environment entry:

```bash
source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh
export TILELANG_HOME=<current checkout>/tilelang_repo
cd <current checkout>/tilelang_repo
```

- Keep design docs as contracts, not progress logs.
