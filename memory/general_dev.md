# 通用开发模式与当前有效经验

> 当前 Blackhole 后端唯一设计依据: `tasks/dev_design/final_blackhole_backend_redesign.md`
> 本文档只保留稳定、可复用的工程经验；不承担阶段状态播报。

## 1. 文档入口

- 当前活动文档只看：
  - `tasks/dev_design/final_blackhole_backend_redesign.md`
  - `tasks/dev_design/README.md`
  - `tasks/progress.md`
- `tasks/dev_design/archive/` 下的内容全部视为历史记录
- 设计边界、阶段状态、验证快照不要写回 `memory/`

## 2. 构建与测试卫生

- C++ 改动后，跑 pytest 前先确认 `libtilelang.so` 已重编；避免加载旧库
- 当前 `tilelang_repo/CMakeLists.txt` 用 `file(GLOB ...)` 收源码；新增 `.cc` 文件后，
  仅 `cmake --build` 不够，必须在 `tilelang_repo/build/` 里先重新执行一次 `cmake ..`
- 不要对同一个 `tilelang_repo/build/` 并行跑 `cmake --build` 和 pytest；
  否则很容易加载到旧/半更新的库
- `pip install -e .` 可能触发重建并失败；更稳的是用 `.pth` 指向本地构建产物
- `3rdparty/` 和 `build/` 不应进入主仓库提交

测试分层：

- 结构层：lowered TIR / attrs / companion IR
- planner 层：`ExecutableSpec` / `KernelSpec` / `cb_configs` / bindings
- runtime 层：direct path 真执行

只做 codegen 或 reference compare 不算 true E2E。

## 3. Layered IR 与 companion 纪律

- module-scope truth 放在 `IRModule.global_infos`，不要回退到单个 `PrimFunc.attrs`
- unsafe TIR mutation 必须整体 invalidate companion truth，不要只删单个 attr
- cross-pass schema 一律 handle-first；字符串只保留 display / debug / compatibility 角色
- semantic truth 属于 `Phase A`，spatial truth 属于 `Phase B`，TT target truth 属于 `Phase C`
- 下游 consumer 优先读最近的 typed IR，不要回头吃 legacy attrs 当 primary truth

当前稳定 companion 习惯：

- pre-lift seed / manifest / witness / program 分层存放
- workload noun 不进入长期 semantic schema
- evidence carrier 不是长期 truth owner
- `fragment_regions` 这类 attr 的删除顺序必须是：
  先踢 semantic consumer，再踢 lowering consumer，最后删 attr 本身

## 4. ABI、schema 与 runtime contract

- compile-time、common-runtime、per-work 三层 ABI 必须严格分开
- runtime 参数布局必须显式、可验证；不要依赖默认顺序、默认名字或 host 猜位置
- `common_runtime_args` 只放所有 core/work item 共享的 metadata：
  如 buffer address、semaphore id
- `work_linear_id`、tile range、logical core coord 这类值属于 per-work / per-core，
  不能塞进 common channel
- work descriptor 要用显式角色字段，例如 `start_id / num_tiles / work_linear_id`
- 64-bit 地址需要明确拆分 / 重组规则

共享参数与对象分组：

- arg identity 必须由 lowering / split 正式产出，不能留给 host 提取层猜
- dedup key 一律用 `identity + ":" + kind`
- 当多个字段共同表达一个对象时，应尽快上提成 schema object，
  不要长期只留在 runtime arg 列表里
- schema-only 路径一旦成立，派生物也必须能从 schema 重建，
  不能偷偷依赖 legacy `accessors`

materialization 原则：

- buffer materialization descriptor 必须显式进入 `ExecutableSpec`
- 当前未正式支持的 ABI / accessor / transport 组合要 build-time fail-fast
- 不要保留 `input0/output0` 这类默认 ABI fallback
- 空 work plan / 默认 core / 默认 packet 都不应由 runtime 补洞

## 5. 存储 scope、resource 与 lowering

- Blackhole `shared` 的主映射是 CB / L1 资源，不是普通 C 数组
- `local` 只是中间语义桶；进入后端后应继续分流成：
  - 真正的小标量临时
  - fragment / accumulator 对象
  - `local/accumulator -> shared(CB)` 这类显式 dataflow transfer
- 一旦 residual `local` 明显处在 fragment 结果写回 CB 的桥接位置，
  应尽快 lower 成正式 copy direction / builtin，不要让普通 `BufferStore` 漏到 codegen
- planner pass 的正式输入必须是上游显式 schema；不要一边吃 schema，
  一边从 `alloc_shared` 之类 IR 形态做 fallback inference
- target 硬件资源模型和 generic backend 不一致时，优先扩 IR / schema，
  不要给后段 pass 打豁免

## 6. analysis、lowering、planner、codegen 边界

- analysis 产出事实；lowering 消费事实；planner 只消费显式 requirement schema；
  codegen 只打印已确定 contract
- 如果 pass 只把规划结果写到 attrs 而不回写 IR body，就会制造两套真源；
  优先让 pass 同时完成 IR 回写
- unsupported subset gate 应在所有后端出口共享，
  不能只挂在 `ExecutableSpec` 或只挂在 device-only codegen 其中一边
- build-time gate 要按具体 unsupported op / contract 报错，
  不要用黑盒总括词长期挡在最前面
- 如果手工 pass 链能过、full `lower()` 过不了，优先检查：
  - optimized path 的 IR 形态是否改变
  - entry `PrimFunc` 是否在 pass 之前被错误过滤

host/codegen 侧额外注意：

- packed call 如果以表达式形式出现，host codegen 必须能同时打印调用语句和结果表达式
- device math builtin 不能只拦截理想 IR 形态；最终 emitted source 里的 `exp2f` 等外部调用也要兜住

## 7. TT-Metal / TT-Sim 环境

稳定环境入口：

```bash
source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh
export TILELANG_HOME=<当前 checkout 或 worktree>/tilelang_repo
cd <当前 checkout 或 worktree>/tilelang_repo
```

稳定经验：

- `scripts/setup_tt_sim.sh` 必须和测试命令在同一个 shell 里执行
- `TT_METAL_SLOW_DISPATCH_MODE=1` 对 TT-Sim 很关键
- direct path kernel 临时目录必须每次执行唯一化，避免 JIT 缓存串扰
- 优先消费 TT-Metal local install tree，不要把 `.cpmcache` 整片塞进 include path

稳定 host-side 抽象：

- `Program`
- `CreateCircularBuffer`
- `CreateKernel` / `CreateKernelFromString`
- `SetRuntimeArgs`

CB contract：

- 需要地址共享：`get_write_ptr / get_read_ptr`
- 也需要同步原语：`cb_reserve_back / cb_push_back / cb_wait_front / cb_pop_front`
- 两者缺一不可

semaphore / remote core：

- 正式 host API 用 `CreateSemaphore(program, core_ranges, initial_value)`
- host/runtime 正式下发的是 semaphore id；device kernel 再 `get_semaphore(id)`
- remote worker core 的 logical -> NOC 坐标转换应由 host materialize；
  不要让 device code 从 logical core 坐标自己猜

## 8. 常见调试线索

- compile / launch 已过但 enqueue 后 hang：
  优先查 CB ID 回写、CB 生命周期、semaphore / remote core runtime args
- direct runtime 只在整套测试里随机错：
  优先怀疑未初始化 device buffer、partial write、或 JIT 缓存串扰
- stick/page copy 报地址对齐错误：
  优先查 `page_bytes % 64 == 0`、page-aligned offset、global width 是否整除 transport width
- copy E2E 通过但 GEMM 数值不对：
  优先查 `transpose_B`、tilize / untilize、host layout conversion
- gate 从 spec 层消失，错误晚到 codegen 的 `undefined variable`：
  多半是另一条出口绕过了同一套 unsupported-op gate
- full `lower()` 比手工 pass 链更早/更怪地失败：
  先看 optimized path 的 TIR 形态变化，再看入口模块是否被提前过滤

Watcher 经验：

- TT-Metal execution hang 先开 `TT_METAL_WATCHER=2`
- 当前环境下 watcher 输出默认在 `generated/watcher/watcher.log`
- 稳定不变的 BRISC / NCRISC / TRISC 状态码组合，
  很适合判断问题是在移动边界，还是仍卡在原地
