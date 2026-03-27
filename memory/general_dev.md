# 通用开发模式与当前有效经验

> 当前 Blackhole 后端唯一设计依据: `tasks/dev_design/final_blackhole_backend_redesign.md`
> 本文档只保留稳定、可复用的工程经验。

## 编译器后端开发

### codegen 职责

- codegen 只负责把已明确的 segment/spec 打印成源码，不要让它同时承担协议推断
- 如果 pass 只把规划结果写到 attrs 而不回写 IR body，后续 pass/codegen 必须从 attrs 恢复状态，形成"两套真源"维护负担。优先让 pass 同时完成 IR 回写
- 当 host TIR 中的 packed call 以表达式形式出现（如 `LetStmt(x, call_packed(...), ...)`），host codegen 必须同时支持：
  - 发出 packed runtime 调用语句
  - 把 `result` 容器重新打印成可用表达式
  否则最终 C 文本会退化成 `x = ;` 这类空表达式错误

### 类型与参数

- compile-time args 和 runtime args 必须严格区分
- compile-time ABI 一旦开始 formalize，就要把“匿名 `vector<uint32_t>` + host 侧猜位置”的约定收正成显式 schema（例如 `compile_time_arg_specs` / `launch_spec`）；兼容字段可以暂留，但不能继续当真源
- runtime 参数布局必须显式、可验证，不能依赖隐式猜测
- 统一 work descriptor 时，优先把 `start_id` / `num_tiles` 这类角色化字段写进 schema；不要继续让 host/runtime 从 `current_work_linear_id`、`tile_count` 之类的单值默认推导整套工作范围
- 对需要 `blockIdx` 重建的 Blackhole kernel，把 `work_linear_id` 作为独立字段保留；不要让 codegen 从 `a_tile_start_id` / `output_tile_start_id` 之类的 range 字段反推逻辑 work identity
- 64-bit 地址需明确拆分与重组规则

### 存储 scope

- Blackhole `shared` 的主映射是 CB/L1 资源，不是简单的 C 数组声明
- 当 target 硬件资源模型与 generic backend 不一致时，优先扩 IR 类型系统（如 StorageRank），不要给后段 pass 打豁免
- 判断"该不该扩 IR"的信号：同一根因在多个 generic pass 上以不同报错出现

### 测试策略

- 只做 codegen/reference compare 的脚本不算 true E2E
- 测试分层：结构层（lowered TIR/attrs）→ planner 层（cb_configs/bindings）→ runtime 层（direct path 真执行）

## TileLang 工程

- `3rdparty/` 和 `build/` 不应进入主仓库提交
- `pip install -e .` 可能重新触发构建并失败，用 `.pth` 指向本地构建产物
- C++ 改动后 pytest 前先确认 `libtilelang.so` 已重编，避免加载旧库假阴性

## TT-Metal / TT-Sim 环境

### 构建

- 系统依赖、clang 版本、RPATH、LD_LIBRARY_PATH 都影响编译和运行
- TileLang direct 模式需对齐 C++20、TT_METAL_HOME、tt_stl、hostdevcommon、umd
- 优先消费 TT-Metal local install tree（`find_package(tt-metalium CONFIG REQUIRED)`），不要把 `.cpmcache` 整片加进 include path

### TT-Sim

- `TT_METAL_SLOW_DISPATCH_MODE=1` 对 TT-Sim 很关键
- `scripts/setup_tt_sim.sh` 必须在执行测试的同一 shell 里 source
- 关键变量：`TT_METAL_RUNTIME_ROOT`、`TT_METAL_SIMULATOR`、`TT_METAL_SLOW_DISPATCH_MODE`、`LD_LIBRARY_PATH`
- direct path kernel 临时目录必须每次执行唯一化，避免 JIT 缓存串扰

### TT-Metal API

- 稳定 host-side 抽象：Program、CreateCircularBuffer、CreateKernel/CreateKernelFromString、SetRuntimeArgs
- CB 数据路径需要地址共享（get_write_ptr/get_read_ptr）**和** 同步原语（cb_reserve_back/cb_push_back/cb_wait_front/cb_pop_front）两者缺一不可
- 对 tile/dataflow target 至少分层：host tensor layout、device buffer/accessor、CB transport、compute ABI、work distribution
- 当 direct runtime 目前只支持某个 dtype 组合时，也要先把 tensor dtype / CB transport dtype / accumulator dtype 明确写进 schema，再由 runtime 显式校验“不支持的组合”；不要继续把 bring-up 时的硬编码假设当协议
- 对 TT-Metal GEMM，host layout conversion 是 correctness contract，不是后续优化：
  - host 侧必须明确谁负责 `transpose_B`
  - 上传前必须把 row-major tensor 转成设备期望的 tiled layout
  - 读回后必须做 untilize
  - copy E2E 通过不能证明 matmul contract 正确，因为 copy 只验证字节保持，不验证 tile 语义
- richer schema 先于更大支持面：如果 schema 已经能表达更多 range/stride 组合，但 direct runtime/codegen 还没正式支持，必须 `ICHECK` fail-fast，不能静默退回旧默认

## Blackhole 后端开发原则

1. 不把单个 kernel 字符串当成后端主产物
2. 新功能落到 `ExecutableSpec → BlackholeModule::ExecuteDirect()` 主路径
3. 三层分工：split 前保语义 → split 后提正式 plan → host side 只做 materialization
4. planner 的 identity 和 lifetime 必须分成两个独立维度（requirement_index ≠ lifetime）
5. 新 schema 先隔离到新场景，不卷入已稳定路径；稳定后再统一
6. 先修 bug（最小改动验证），再做 schema 工程（协议质量提升）

## 文档治理

- 阶段状态切换时，至少同步检查：progress.md 页首、任务表、活动设计文档列表
- 排障文档一旦问题已解决，要在文档头部把状态改成“已实施/历史记录”，并把最终根因补回去；不要让“初始假设”继续伪装成当前结论
- 设计文档分三类：当前活动、仍有效支撑、历史记录
- 历史文档不重写正文，在开头加头注标明历史性质
