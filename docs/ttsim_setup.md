# TT-Sim Environment Entry

This file only defines the repository TT-Sim environment entry.
It does not maintain task-specific pytest selectors.

For task-specific runtime gates, read:

- `AGENTS.md`
- `CLAUDE.md`
- `GEMINI.md`
- `tasks/progress.md`
- `memory/general_dev.md`
- `memory/bugs.md`

## Entry

Use the top-level script:

```bash
source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh
```

Do not use a copied script from a worktree as the canonical entry.

## Rules

1. `setup_tt_sim.sh` and the following test command must run in the same shell.
2. After sourcing the script, set `TILELANG_HOME` to the checkout under test.
3. This file does not choose pytest cases; the current task/design/progress
   docs choose the selector.
4. For TT-Sim / runtime debugging, read:
   - `memory/general_dev.md`
   - `memory/bugs.md`

## Example

```bash
source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh
export TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo
cd /root/dev/vibe_dsl/tilelang_repo
pytest -q testing/python/target/blackhole/test_blackhole_copy_runtime.py
```

The command above is only an environment-entry smoke example, not the fixed
selector for the current task.
