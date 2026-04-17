# TileLang Blackhole Backend Progress

> 当前唯一总体设计文档:
> `tasks/dev_design/final_blackhole_backend_redesign.md`

> 当前 cleanup 执行总览:
> `tasks/dev_design/2026-04-16-blackhole-final-legacy-protocol-cleanup.md`

> 本文件只记录 repo HEAD 的当前状态、当前主任务和下一步。
> 设计边界看设计文档；实现细节看代码和对应 task 文档。

## 1. 当前状态

- **日期**: `2026-04-17`
- **总体状态**: legacy protocol cleanup 仍在进行中，尚未收口
- **当前主任务**: `Cleanup Task 1`
- **当前原则**:
  先完成 cleanup 主线，再恢复 support surface / workload payoff 扩展

## 2. repo HEAD 已完成

- `Cleanup Task 0` 已完成
- exact TT-Metal builtin selector 已接入主链
- helper/composite builtin 和 local pseudo builtin
  已退出 active compute surface
- `compute_epilogue_ops`
  已退出 `TTProgram / ExecutableSpec / codegen / runtime`
  这条主链

## 3. repo HEAD 仍未完成

下面这些 legacy residue 仍在 repo HEAD，需要继续清理：

- public analysis wrappers
- internal analysis / evidence bags
- `blackhole.lowering_requirements`
- `blackhole.copy_semantics`
- `blackhole.segment_kind`
- `blackhole.resource_plan`

结论保持不变：

- cleanup 主线还没做完
- 现在还不能把工作重心切回 support surface 扩展

## 4. 当前执行顺序

当前实际执行顺序固定为：

1. `Cleanup Task 1`
2. `Cleanup Task 2`
3. `Cleanup Task 3`
4. `Cleanup Task 4`
5. `Cleanup Task 5`
6. cleanup 完成后，再恢复 support surface / workload payoff 扩展

## 5. 当前下一步

当前下一步固定为：

1. 完成 `Cleanup Task 1`
   - 用 direct logical bridge capture
     取代 compute-region bag
2. 然后完成 `Cleanup Task 2`
   - 删除 public / internal legacy analysis bags
   - 收掉 `blackhole.lowering_requirements`
     这条旧语义通道
3. 再推进 `Cleanup Task 3 / 4`
   - 删除 `blackhole.copy_semantics`
   - 删除 `blackhole.segment_kind`

## 6. 文档职责

- `final_blackhole_backend_redesign.md`
  - 长期设计和表示层边界
- `task0/1/2/3`
  - 各层合同和完成判据
- `progress.md`
  - 只记录 repo HEAD 当前状态与下一步
