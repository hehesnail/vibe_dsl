# 问题与 Bug 记录

> 本文档只保留仍有复用价值的问题模式。
> 阶段状态、总体 blocker 与完成判定以设计文档和 `tasks/progress.md` 为准。

## 1. 当前未解决

### `blackhole.acc` 混合语义会在 flash-attn runtime 上稳定制造 correctness mismatch

- **现象**:
  - flash-attention direct runtime 已不再 hang
  - 当前剩余失败主要表现为 softmax / accumulate 链上的数值错误、`nan` 或明显偏差
- **根因**:
  - `blackhole.acc` 在部分路径里仍同时承担两种不兼容语义：
    - TT compute 主路径里的 tile scratch / matmul destination
    - 线性 fragment helper 眼中的数组 scratch
- **当前结论**:
  - 这不是单个 emitter bug，而是分层未彻底收正后的结构性问题
  - 正式 payoff 归属 `Phase C2`

## 2. 已解决但值得记住的模式

### 2.1 ABI / schema

#### schema-only ABI 一旦成立，派生物也必须能从 schema 重建

- **症状**: strip 掉 legacy `accessors` 后，runtime 先报缺失 `buffer_materialization`
- **根因**: 物化信息仍只从 legacy accessor 路径推导
- **修法**: 从 `compile_time_arg_specs` 的 `buffer/layout/memory_space` 元数据恢复 materialization
- **教训**: schema 既然宣称自己是主路径，派生物也必须能从它单独重建

#### runtime / common-runtime arg 去重必须用 `identity:kind`

- **症状**: 同一 remote core 的 `logical_core_noc_x/y` 丢半边
- **根因**: 只按 `identity` 去重，把“同组对象的不同分量”合并掉了
- **修法**: dedup key 统一改成 `identity + ":" + kind`
- **教训**: `identity` 是分组标识，不是唯一字段

#### remote core 这种“多字段表达一个对象”的东西，应尽快上提成 schema object

- **症状**: runtime 侧长期从若干 runtime arg 手工重建 remote core
- **根因**: descriptor 没进 `KernelSpec`
- **修法**: 提升为 `KernelSpec.remote_core_descriptors`
- **教训**: 一旦多个字段共同表达一个对象，就别长期只留在 arg 列表里

#### synchronization schema 应在 spec / module build 边界校验，而不是留到执行期

- **症状**: `semaphore_binding` 缺失或 remote core x/y 不成对，只在 direct execution 时炸
- **根因**: semaphore 与 remote-core 解析散在多处 kind-switch，缺统一校验
- **修法**: 在 `ExecutableSpec` / `BlackholeModuleNode` 构造期统一校验
- **教训**: 只要已经进入正式 schema，对象合法性就应尽早 fail-fast

#### copy/dataflow 主路径不能退回默认 ABI

- **症状**: schema 缺失时仍然继续 build，到后段才报 buffer binding 缺失
- **根因**: 保留了 `input0/output0` 这类默认 runtime-arg fallback
- **修法**: 删除默认 fallback；schema 缺失 build-time 直接失败
- **教训**: 正式 ABI 不应该靠默认名字兜底

### 2.2 planner / runtime contract

#### partial-write output 必须先把 host 初值同步到 device

- **症状**: 单测单跑看似正确，整套顺序执行时 output 未覆盖区域读回脏数据
- **根因**: runtime 只初始化 input，不初始化 output device buffer
- **修法**: 执行前统一同步所有 host tensor 当前内容
- **教训**: 只要 schema 允许 partial write，output 初值就是 contract 的一部分

#### stick/page transport 需要显式 64B 对齐边界

- **症状**: TT-Metal NOC 报地址对齐错误
- **根因**: `transport_page_size`、offset 或全局宽度没有满足底层 page / alignment 约束
- **修法**: 把 `transport_page_size` 显式写进 schema，并在 lowering 阶段 fail-fast
- **教训**: transport 合法性要前移到 schema / lowering，不要留给 runtime

#### planner 缺 work plan 时，runtime 不能自动补默认 core / packet

- **症状**: planner/runtime contract break 被伪装成“还能跑”
- **根因**: spec 提取层和 runtime 都在补默认 work packet / fallback core
- **修法**: 删掉默认值；空 `work_packets` 直接 fail-fast
- **教训**: host/runtime 计划缺失时必须显式报错，不能补“最小可运行默认值”

#### logical core 坐标和 physical / NOC 坐标不能混用

- **症状**: core lookup 失败、range 越界、launch/core 映射错位
- **根因**: planner 产出旧 physical-style 坐标，runtime 消费 logical worker grid
- **修法**: planner/runtime 统一到 logical worker grid；logical -> NOC 由 host materialize
- **教训**: core descriptor 必须明确语义，不能让两端各自猜

### 2.3 CB / synchronization / compute lifecycle

#### GEMM output / writer bridge CB 去重不能只看 `Buffer` 对象或 `buffer->data`

- **症状**: single-core GEMM direct runtime 里 compute 发布到一个 CB，writer 却在另一个 CB 上 `cb_wait_front`，最终稳定挂死
- **根因**: `C_local` 在 GEMM extract 路径和 writer / decl-buffer 路径上出现成多个逻辑等价但对象身份不同的 `Buffer`；若 requirement 去重只看 `Buffer` 或 `buffer->data`，同一逻辑资源会被拆成两个 CB requirement
- **修法**: `AllocateRequirementIndex` 去重要覆盖稳定的 logical buffer identity，并在较晚看到更强 `input/output` 角色时把已建 requirement 从 `intermediate` 升级成正确角色
- **教训**: planner / lowering 的 dedupe key 不能只依赖对象身份；只要 logical resource 能跨 pass / canonicalization 漂移，就必须保留稳定 identity

#### 新 builtin 只要带 cb_id，就必须注册回写位置

- **症状**: compute kernel 写错 CB，consumer 永远等不到数据
- **根因**: `PlanBlackholeCB::GetCBArgPositions` 漏注册 cb_id 参数位置
- **修法**: 补注册，并加 post-condition guard
- **教训**: “新增 builtin -> 必须声明 cb_id 回写位置” 是正式协议，不是习惯

#### `blackhole.acc` 结果若会再喂 matmul，producer 侧发布页数必须按未来 consumer 算

- **症状**: 第二次 matmul 前挂在 `cb_wait_front` / `mm_init`
- **根因**: producer 只按当前 pointwise/cast 写入页数发布，没有按未来 matmul 需求 push_back
- **修法**: 预扫描 future matmul consumer，按其页数需求发布
- **教训**: scratch CB 的 producer 不只要“写进去”，还要按 future consumer 的协议正式发布

#### `blackhole.acc` GEMM 输出不能机械套 transport-CB reserve 模板

- **症状**: scratch CB 生命周期被破坏，compute hang 或错乱
- **根因**: matmul output path 无条件沿用 transport/output CB 的 reserve/push 模板
- **修法**: `blackhole.acc` 输出不再重复 reserve；按 scratch 生命周期处理
- **教训**: transport CB 和 scratch CB 不是同一类资源

#### 跨核 semaphore 握手必须下发真实 remote NOC 坐标

- **症状**: TT-Sim 在 enqueue 后挂死
- **根因**: device kernel 直接把 logical core 坐标塞给 `get_noc_addr`
- **修法**: host 用 `worker_core_from_logical_core(...)` 求真实 NOC 坐标后下发
- **教训**: remote route 信息必须 host-materialized，不能让 device 代码猜

### 2.4 analysis / lowering / gate

#### semantic-owned truth 缺失时，要回补 `Phase A`，不要让 `Phase B` 借旧 attrs 自救

- **症状**: `row_reduction.kind` 缺失后，`SemanticProgram` 丢 reduce update，
  `SpatialProgram` 退化成单 phase
- **根因**: formal device 主链缺 semantic-owned fact
- **修法**: 在 manifest / fragment analysis / semantic lift 把 truth 补齐
- **教训**: 缺的是 semantic truth，就回 `Phase A` 收；不要让 `Phase B` 临时绕回 raw attrs

#### `local/accumulator -> shared(CB)` bridge 应尽快变成正式 copy direction

- **症状**: compile-path 晚到 codegen 才报 residual shared store / undefined variable
- **根因**: fragment/local 结果写回 CB 的桥接语义仍以普通 `BufferStore` 漏到后段
- **修法**: 新增正式 copy direction / builtin，codegen 只消费 builtin
- **教训**: 对 Blackhole，`local` 只是中间态，不应长期作为最终资源语义

#### unsupported-op gate 不能只挂在一条出口

- **症状**: 一条路径按预期 fail-fast，另一条路径晚到 codegen 才炸 `undefined variable`
- **根因**: device-only codegen 绕过了 `ExecutableSpec` 路径上的 gate
- **修法**: spec 提取层和 codegen 入口共享同一套 gate
- **教训**: 只要仓库里有多条后端出口，shared lowering boundary 就要双边同时守住

#### fragment analysis 必须按结构 / 数据流识别，不能靠全局 op 扫描或名字匹配

- **症状**: copy/GEMM 被误伤成 `pointwise_chain`，或 MHA/GQA 的 row reduction / row broadcast 被漏掉
- **根因**:
  - 全局扫描 `tir.add/mul/div/max/...` 会把普通索引算术也算进去
  - 只识别 `CallNode`、只认 `floor_div`、或只认 split-after 某一种 IR 包装形态，都会漏真实 optimized path
- **修法**:
  - 只在 fragment/local region 自身的数据流里识别 pointwise
  - 同时识别 `AddNode/MaxNode/MulNode/DivNode` 等原生节点
  - 先剥掉无语义包装，再匹配 reduction / broadcast 形态
- **教训**: 对复杂 TIR，先看真实 IR 结构，再决定 matcher；不要把源码层直觉当 IR 协议

#### gate 应该按具体未支持子集收窄，而不是长期挡整类 blocker

- **症状**: `row_broadcast` / `pointwise_chain` 这种总括词掩盖哪些子集已可 lower
- **根因**: blocker 设计得太黑盒
- **修法**: 先吃掉稳定子集，再让 gate 随真实 lowering 一步步收窄
- **教训**: 细粒度 unsupported 集合比黑盒大类更有工程价值

### 2.5 低层基础设施

#### pass 拆分后，新 `.cc` 若没接进 `TILE_LANG_BLACKHOLE_SRCS`，会在 Python 导入时炸成共享库未定义符号

- **症状**: C++ 编译似乎通过，但 Python/pytest 一加载 `libtilelang.so` 就报
  `symbol lookup error: undefined symbol: BuildSpatialExecutionPlanForFunc(...)`
- **根因**: 新 split 出来的 translation unit 没被编进 `tilelang` 共享库，
  旧对象里只留下未解析引用
- **修法**: 把新文件显式加入 `tilelang_repo/CMakeLists.txt` 的
  `TILE_LANG_BLACKHOLE_SRCS`，重新 `cmake` + `cmake --build`
- **教训**: “文件已存在”不等于“目标已链接”；对 split pass，先用
  `nm -D libtilelang.so | c++filt` 确认符号真的进库

#### `TT_METAL_WATCHER` 改变症状时，先区分 direct runtime 回归还是 watcher 线程自己炸了

- **症状**: multicore GEMM direct call 在 `TT_METAL_WATCHER=10` 下于 `Dump #2` 前后 `SIGABRT`，或开 `TT_METAL_WATCHER_TEST_MODE=1` 后卡在同一 dump；但关闭 watcher 后 direct runtime baseline 仍能通过
- **根因**: native backtrace 落在 `tt::tt_metal::WatcherServer::Impl::poll_watcher_data()`，不是 `BlackholeModule` 主执行线程
- **修法**: 用 gdb / native bt 先确认 abort 源头；把 watcher-side failure 与 direct runtime regression 分开判断，正式 baseline 在 `TT_METAL_WATCHER` unset 的环境下跑
- **教训**: watcher 是调试器，不是真源。只要 watcher 改变了现象，先证明是 workload 坏了还是 watcher 自己坏了

#### 共享 protocol struct 必须只有一个定义

- **症状**: 改字段后随机崩溃、排序或字符串拷贝崩
- **根因**: 同 namespace 出现两份对象定义，布局漂移导致 ODR / ABI 错位
- **修法**: 共享协议 struct 集中到单一定义
- **教训**: 协议对象分叉定义迟早会炸成随机崩溃

#### `RemapBufferData` 之后，同源 Buffer 需要缓存，不能让 identity 漂掉

- **症状**: canonicalization 后下游去重或 `buffer_to_cb_` 查找失效
- **根因**: 对同一原始 buffer 多次 remap 产生多个不同对象
- **修法**: 在 remap helper 内缓存结果
- **教训**: 只要下游逻辑依赖 buffer identity，就必须保证 remap 后 identity 稳定

#### 不要对临时 `ObjectRef` 调 `CopyOnWrite()`

- **症状**: dangling pointer、随机崩溃
- **根因**: 临时 `ObjectRef` 析构后 COW 指针悬空
- **修法**: 不对临时对象做 COW；改为直接构造返回值
- **教训**: TVM object 生命周期问题会伪装成完全无关的崩溃

#### kernel 临时目录必须每次执行唯一

- **症状**: 同一 pytest 进程内 direct-call case 顺序相关、复用旧编译结果
- **根因**: TT-Metal JIT 复用固定临时路径
- **修法**: kernel 临时目录每次执行唯一化
- **教训**: JIT 缓存串扰首先要怀疑路径复用，而不是数值逻辑本身

## 3. 环境问题速查

| 问题 | 解决 |
|------|------|
| `pip install -e .` 失败 | 用 `.pth` 指向本地构建产物 |
| Python 加载旧库 | 统一使用 `tilelang_repo/build/` 单一构建目录，并确认已重编 |
| TT-Sim 报 `Root Directory is not set` | 设置 `TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME` |
| TT-Sim 报 `No chips detected` | 设置 `TT_METAL_SIMULATOR` 与 `TT_METAL_MOCK_CLUSTER_DESC_PATH` |
