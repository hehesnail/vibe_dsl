# TT-Sim 环境入口说明

本文件只说明 **当前仓库里 TT-Sim 环境从哪里进入**，不再维护具体 case 的测试命令模板。

如果你要跑哪条测试，应该看：

- `AGENTS.md`
- `CLAUDE.md`
- `GEMINI.md`
- `tasks/progress.md`
- `memory/general_dev.md`
- `memory/bugs.md`

## 当前正式入口

统一从顶层脚本进入：

```bash
source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh
```

不要把 worktree 里的副本脚本当成正式入口。  
当前正式入口是顶层仓库的 `scripts/setup_tt_sim.sh`。

## 使用规则

1. `setup_tt_sim.sh` 和后续测试必须在**同一个 shell**里执行。
2. 如果你在 worktree 里工作，source 完脚本后要把 `TILELANG_HOME` 指回当前 checkout/worktree。
3. 这个文件只负责“环境入口”，不负责任务级测试选择；具体跑哪个 `pytest`，由当前 task/design/progress 决定。
4. 遇到 TT-Sim / runtime 调试问题，先看：
   - `memory/general_dev.md`
   - `memory/bugs.md`

## 最小示例

```bash
source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh
export TILELANG_HOME=/root/dev/vibe_dsl/tilelang_repo
cd /root/dev/vibe_dsl/tilelang_repo
pytest -q testing/python/target/blackhole/test_blackhole_copy_runtime.py
```

上面只是确认环境入口可用的最小示例，不是“当前任务应该跑什么”的固定答案。
