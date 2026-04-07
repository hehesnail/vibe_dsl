# 通用开发模式与当前有效经验

> 当前 Blackhole 后端唯一设计依据: `tasks/dev_design/final_blackhole_backend_redesign.md`
> 本文档只保留长期可复用的工作模式；不承担阶段状态、bug 目录或验证快照职责。

## 1. 文档使用模式

- 当前活动入口固定为：
  - `tasks/dev_design/final_blackhole_backend_redesign.md`
  - `tasks/dev_design/README.md`
  - `tasks/progress.md`
- `tasks/dev_design/archive/` 只当历史记录，不再作为当前任务入口
- 设计边界写在设计文档，阶段状态写在 `progress.md`，`memory/` 只记长期经验

## 2. 构建模式

- C++ 改动后，先确认 `libtilelang.so` 已重编，再跑 pytest
- 当前 `tilelang_repo/CMakeLists.txt` 用 `file(GLOB ...)` 收源码；
  新增 `.cc` 文件后，要先在 `tilelang_repo/build/` 里重新执行一次 `cmake ..`
- 同一个 `tilelang_repo/build/` 不要并行跑 `cmake --build` 和 pytest
- `pip install -e .` 不是默认开发路径；更稳的是用 `.pth` 指向本地构建产物
- `3rdparty/` 和 `build/` 不进主仓库

## 3. 验证模式

测试层级固定分三层：

- 结构层：
  - lowered TIR、attrs、companion IR
- planner 层：
  - `ExecutableSpec`、`KernelSpec`、bindings、CB / core / semaphore 规划结果
- runtime 层：
  - direct path 真执行

经验上：

- 只做 codegen/reference compare 不算 true E2E
- 当前支持面和 fail-fast 边界都应该在更早层被看见，不要全部压到 runtime

## 4. Layered IR / companion 模式

- module-scope truth 放 `IRModule.global_infos`，不要回退到单个 `PrimFunc.attrs`
- unsafe TIR mutation 必须整体 invalidate companion truth，不要只删单个 attr
- cross-pass schema 一律 handle-first；字符串只保留 display / debug / compatibility 角色
- semantic truth 属于 `Phase A`，spatial truth 属于 `Phase B`，TT target truth 属于 `Phase C`
- 下游 consumer 优先读最近的 typed IR，不要回头把 legacy attrs 当 primary truth

当前稳定 companion 习惯：

- seed / manifest / witness / program 分层存放
- workload noun 不进入长期 semantic schema
- evidence carrier 不是 truth owner
- 兼容 attr 的删除顺序固定为：
  先移走 semantic consumer，再移走 lowering consumer，最后删 attr 本身

## 5. Schema / ABI 模式

- compile-time、common-runtime、per-work 三层 ABI 必须严格分开
- runtime 参数布局必须显式、可验证；不要依赖默认顺序、默认名字或 host 猜位置
- `common_runtime_args` 只放共享 metadata；per-work / per-core 值单独下发
- work descriptor 要用角色化字段，不用单值默认去反推整套语义
- 64-bit 地址需要明确拆分 / 重组规则

对象分组与派生规则：

- arg identity 必须由 lowering / split 正式产出
- dedup key 一律用 `identity + ":" + kind`
- 多个字段共同表达一个对象时，应尽快上提成 schema object
- schema-only 路径一旦成立，派生物也必须能从 schema 单独重建
- 未正式支持的 ABI / accessor / transport 组合，要 build-time fail-fast
- 不保留默认 ABI、默认 core、默认 packet 这类补洞

## 6. analysis / lowering / planner / codegen 模式

- analysis 产出事实
- lowering 消费事实并改写 IR
- planner 只消费显式 requirement schema
- codegen 只打印已确定 contract

稳定做法：

- 如果 pass 只把规划结果写到 attrs 而不回写 IR body，就会制造两套真源；
  优先让 pass 同时完成 IR 回写
- unsupported subset gate 应在所有后端出口共享
- gate 应按具体 contract / op family 报错，不要长期用黑盒总括词
- 需要的信息优先从 typed IR / schema 拿；拿不到就扩 IR / schema，
  不要把猜测沉淀成长期协议

## 7. Resource / storage 模式

- Blackhole `shared` 的主映射是 CB / L1 资源，不是普通 C 数组
- `local` 是中间语义桶；进入后端后要继续分流成：
  - 真正的小标量临时
  - fragment / accumulator
  - 显式 dataflow transfer
- 一旦 residual `local` 已经明显表示“结果写回 CB 的桥接语义”，
  就应尽快 lower 成正式 builtin / direction
- planner 的正式输入必须是上游显式 schema；
  不要一边吃 schema，一边从 IR 形态做 fallback inference
- target 资源模型与 generic backend 不一致时，优先扩 IR / schema，
  不要给后段 pass 打豁免

## 8. TT-Sim / TT-Metal 模式

统一入口：

```bash
source /root/dev/vibe_dsl/scripts/setup_tt_sim.sh
export TILELANG_HOME=<当前 checkout 或 worktree>/tilelang_repo
cd <当前 checkout 或 worktree>/tilelang_repo
```

稳定经验：

- `setup_tt_sim.sh` 和测试命令必须在同一个 shell
- `TT_METAL_SLOW_DISPATCH_MODE=1` 对 TT-Sim 很关键
- direct path kernel 临时目录必须每次执行唯一化
- 优先消费 TT-Metal local install tree，不要把 `.cpmcache` 整片塞进 include path

稳定 host-side 抽象：

- `Program`
- `CreateCircularBuffer`
- `CreateKernel` / `CreateKernelFromString`
- `SetRuntimeArgs`

稳定同步 / 资源模式：

- CB 既需要地址共享，也需要同步原语
- semaphore 在 host/runtime 侧的正式对象是 id，不是地址
- remote worker core 的 logical -> NOC 坐标转换由 host materialize

## 9. 调试模式

- 先判断问题落在哪一层：
  - 结构层
  - planner/spec 层
  - runtime 执行层
- 手工 pass 链和 full `lower()` 不一致时，先比对 optimized path 的 IR 形态，
  再检查入口 `PrimFunc` 是否被提前过滤
- compile / launch 已经通过时，优先查 CB 生命周期、同步协议和 runtime arg materialization
- copy 跑通只证明字节路径可用，不证明 matmul / tile layout contract 正确
- execution hang 优先配合 Watcher 看状态组合，而不是先堆日志
