# Stage 4 Phase A: Formalization Note

## 基本信息

- **文档角色**: `Phase A` 的并行理论化 / 证明化说明
- **当前状态**: `2026-04-07` research track；不阻塞当前工程主线
- **唯一总体设计**: `tasks/dev_design/final_blackhole_backend_redesign.md`
- **工程实现入口**: `tasks/dev_design/stage4_phase_a_semantic_ir.md`

## 1. 目的

这份文档只负责一件事：

- 把 `Phase A` 收成可证明、可验证、可对外表述的理论对象

它不负责：

- 充当第二份总体设计文档
- 指挥当前工程实施
- 取代 executable validator

## 2. Research Claim

这里不追求“万能语义恢复器”。
当前合理且诚实的研究命题是：

- 对满足 canonical evidence precondition 的程序，
  `Phase A` 产生的 `SemanticProgram`
  是该 evidence domain 上的 sound abstraction
- audited preserve / typed rebind pass 不破坏该抽象
- `Phase B` 只能对其做 organization-preserving refinement

也就是说：

- 不承诺 arbitrary workload completeness
- 不承诺从任意 lower 后 IR 恢复全部真实语义
- 只承诺 bounded semantic abstraction 的 soundness

## 3. 为什么 `Phase A` 必须存在

仓库里长期混在一起的 truth 至少有三类：

- algorithmic truth
- spatial program truth
- target/runtime truth

如果不先冻结 algorithmic truth，后段就只能：

- 从被压碎的 IR 里反向猜语义
- 或把本该属于 `Phase B / C` 的 truth 重新塞回语义层

因此 layered IR 不是偏好，而是问题结构直接推出的结果：

- `Phase A` 冻结 algorithmic truth
- `Phase B` 组织 spatial program truth
- `Phase C` 物化 target/runtime truth

## 4. 形式化对象

### 4.1 Canonical Evidence Domain `E`

`Phase A` 的 concrete 输入不定义为“任意 lower 后 TIR”，而定义为：

- `E = (E_seed, E_manifest, E_fallback, E_pre)`

其中：

- `E_seed`
  - pre-lift semantic seeds
- `E_manifest`
  - manifest-backed explicit-op 与 structural evidence
- `E_fallback`
  - 当前仍保留的 compatibility evidence
- `E_pre`
  - lift 合法的 precondition

关键点：

- `E` 是 `Phase A` 实际消费的 evidence domain
- `E` 不等于全部程序执行语义
- `E` 只包含 `Phase A` 应负责的 algorithmic evidence

### 4.2 Abstract Semantic Domain `A`

`Phase A` 的输出定义为：

- `A = (A_core, A_graph, A_contract)`

其中：

- `A_core`
  - `Domain`
  - `State`
  - `Update`
  - `UpdateLaw`
  - `AccessMap`
  - `SemanticSupplement`
- `A_graph`
  - `StateVersion`
  - `StateDef`
  - `StateUse`
  - `StateJoin`
- `A_contract`
  - `preserve`
  - `typed_rebind`
  - `invalidate`

### 4.3 Abstraction Function `alpha`

对当前仓库，`alpha` 最合适的定义不是单个纯函数，而是 pass composition：

- `alpha = BuildWitnesses ; LiftSemanticCore ; NormalizeStateEffect`

对应当前实现：

- `AnalyzeSemanticStructure`
- `LiftStatefulSemanticIR`
- `semantic_state_effect_graph`

`alpha(E)` 的结果就是当前 `tl.semantic_program`。

### 4.4 Refinement Predicate `R(E, A)`

`R(E, A)` 的含义是：

- `A` 不伪造与 `E` 矛盾的 semantic facts
- `A_graph` 是 `A_core` 的合法 normalization
- `A_contract` 与 companion lifecycle 一致

当前实现里，`ValidateSemanticRefinement` 是 `R` 的 executable approximation。

## 5. Obligation Families

把 `R(E, A)` 收成可执行 contract family 后，当前最重要的 obligation 是：

- `R_vocab`
  - witness vocabulary closure
- `R_anchor`
  - anchor / related-object closure
- `R_role`
  - state role soundness
- `R_law`
  - update law soundness
- `R_source`
  - source-set soundness
- `R_relation`
  - companion / carry relation soundness
- `R_graph`
  - graph normalization soundness
- `R_contract`
  - freeze contract legality
- `R_rebind`
  - typed rebind preservation
- `R_reject`
  - reject-rather-than-guess discipline

这些 obligation 的意义只有三个：

1. 给 `ValidateSemanticRefinement` 一个可枚举的扩展面
2. 给 `Phase B` refinement checker 一个稳定输入面
3. 明确 `Phase A` 允许 reject / invalidate / defer，而不是默认“总能猜出来”

## 6. `Phase A -> Phase B` 接口

`Phase B` 的理论输入接口可以写成：

- `I_B = (A_core, A_graph, A_contract)`

也就是说：

- `Phase B` 不应重读 raw witness
- `Phase B` 不应重读 raw fragment attrs 发明 semantic truth
- `Phase B` 应只消费已冻结的 algorithmic truth 与 normalized state/effect skeleton

因此正确关系是：

- `Phase B` 对 `Phase A` 做 refinement by organization

而不是：

- `Phase B` 再做一次 semantic recovery

## 7. 对应的 validator 方向

最自然的下一步是：

- `ValidateSpatialRefinement(SemanticProgram, SpatialProgram)`

它至少应检查：

- semantic coverage
- no semantic invention
- dependency preservation
- phase-boundary respect
- layout / partition non-interference
- fail-closed on missing structure

## 8. Failure Modes

这条理论线必须显式承认下面这些 failure mode：

- evidence 不足
- semantic core 不足
- truth 实际属于 `Phase B / C`
- companion contract 被破坏

因此这条线：

- 不证明 completeness
- 不保证没有 precision loss
- 不证明 `Phase C` target materialization correctness

## 9. 最小诚实外部表述

如果今天必须对外描述这项工作，最合理的版本是：

- 我们没有证明一个万能语义恢复器
- 我们定义并实现了一个针对 canonical evidence domain 的 bounded semantic abstraction layer
- 它带有 typed witness、normalized state/effect graph 与 executable refinement contract
- 它为后续 `SpatialProgram` refinement 提供了冻结的 algorithmic truth boundary

## 10. 当前仍有价值的研究交付物

最值得继续推进的 research artifact 只有四项：

1. `Phase A` formal semantics note
2. obligation matrix
3. `Phase B` refinement validator
4. 小范围 mechanized core

如果后续真的要 mechanize，推荐顺序是：

1. witness vocabulary / payload family
2. `SemanticProgram` core
3. state/effect graph normalization invariant
4. `ValidateSemanticRefinement` 对应 lemma

## 11. 选读参考

- Cousot and Cousot, *Abstract Interpretation* (POPL 1977)
- Darais and Van Horn, *Constructive Galois Connections* (ICFP 2016)
- Lattner et al., *MLIR* (2020)
- Lopes et al., *Alive2* (PLDI 2021)
