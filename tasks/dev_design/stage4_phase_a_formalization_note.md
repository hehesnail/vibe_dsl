# Stage 4 Phase A: Formalization Note

## Purpose

这份文档单独承接 `Phase A` 的理论化 / 证明化轨道。它不是第二份总体设计文档，也不是当前工程实现入口。

边界：

- [final_blackhole_backend_redesign.md](/root/dev/vibe_dsl/tasks/dev_design/final_blackhole_backend_redesign.md)
  仍是唯一总体设计
- [stage4_phase_a_semantic_ir.md](/root/dev/vibe_dsl/tasks/dev_design/stage4_phase_a_semantic_ir.md)
  仍是 `Phase A` 工程实现文档
- 本文档只负责把 `Phase A` 收成可证明、可验证、可写技术报告的理论对象

非 blocker 说明：

- 这条线是 `Phase B / C` 的并行 research track
- 它不阻塞当前主线实现
- 但它为后续 `Phase B` refinement validator 和 formal report 提供统一对象

## 1. Research Claim

这里不追求一个过强的“万能语义恢复器”命题。我们只追求下面这个更弱、也更正确的命题：

- 对满足 canonical evidence precondition 的程序，`Phase A` 产生的 `SemanticProgram`
  是该 evidence domain 上的 sound abstraction；后续 audited preserve/rebind pass 不破坏该抽象，
  `Phase B` 只能对其做 organization-preserving refinement。

换句话说：

- 不承诺 arbitrary workload completeness
- 不承诺从任意 lower 后 IR 恢复全部真实语义
- 只承诺 bounded semantic abstraction 的 soundness

## 2. Why Phase A Must Exist

`Phase A` 不是因为“想多加一层 IR”才存在，而是由当前仓库的混层问题直接推出的。

Blackhole 当前长期混在一起的 truth 至少有三类：

1. algorithmic truth
   - carry
   - selection companion
   - index flow
   - ordered recurrence
   - value/update dependency
2. spatial program truth
   - task
   - channel
   - layout
   - work partition
   - phase boundary
3. target/runtime truth
   - TT resource
   - transport / sync
   - ABI / common-runtime / per-work bindings

这三类 truth 混在同一层时，后段就会被迫：

- 从被压碎的 lower 后 IR 里反向猜算法语义
- 或者把本应属于 task/layout/resource 的事实重新塞回语义层

`blackhole.acc` 的历史问题正是这个混层的直接体现。

因此 layered IR 不是偏好，而是问题结构的自然结果：

- `Phase A` 冻结 algorithmic truth
- `Phase B` 组织 spatial program truth
- `Phase C` 物化 target/runtime truth

## 3. Repo-Driven Theoretical Definition

### 3.1 Phase A as an Abstraction Layer

一旦接受上面的分层，`Phase A` 就不能再被理解成“更聪明的 recovery pass”。

原因很简单：

- lower 后 IR 已经不稳定保留全部前端语义
- 但后续编译仍需要一个可消费、可冻结、可验证的 algorithmic truth layer

因此：

- `Phase A` 不是 concrete semantics 本身
- `Phase A` 是对 canonical evidence domain 的有界抽象

在当前实现中，这个定义落成：

- `AnalyzeSemanticStructure`
- `tl.semantic_witnesses`
- `LiftStatefulSemanticIR`
- `ValidateSemanticRefinement`

### 3.2 Why Phase A Must Stay Small

`Phase A` 的 vocabulary 必须保持 workload-agnostic；否则它会退化成 matcher registry。

适合进入 `Phase A` 的是：

- `carry`
- `reduction_accumulator`
- `selection_state`
- `index_state`
- `select`
- `recurrence`
- `source_set`
- `companion`
- `carried_from`

不应直接进入 `Phase A` vocabulary 的是：

- `flash_attn_state`
- `topk_selector`
- `paged_decode_route`
- `fusedmoe_dispatch`

前者是 semantic axis，后者是 workload noun。

### 3.3 Why Witness/Core/Validator Is Structurally Necessary

仓库约束已经明确：

- 不允许名字匹配恢复语义
- 不允许把关键信息继续留给后段猜

这直接推出三层结构：

1. witness layer
2. semantic core
3. refinement validator

缺一不可：

- 没有 witness：semantic core 只是 pass 产物
- 没有 core：analysis attrs 不能变成 compiler contract
- 没有 validator：semantic layer 只是缓存，不是可检验真相

### 3.4 Why State/Effect Graph Is Part of the Semantics

只靠 `State` 与 `Update`，不足以表达：

- version 来源
- state use
- carried/order relation
- recurrence join boundary

因此：

- `StateVersion`
- `StateDef`
- `StateUse`
- `StateJoin`

是 `Phase A` 抽象语义的一部分，而不是纯实现细节。

### 3.5 Why Preserve / Typed Rebind / Invalidate Is Not Optional

一旦 `Phase A` 被定义为冻结 truth，后续 pass 就不能在对象 identity 变化后继续沿用旧 truth。

于是得到 companion lifecycle contract：

- `preserve`
- `typed_rebind`
- `invalidate`

这不是工程偏好，而是 semantic layer 想长期存在的必要条件。

## 4. Formal Objects

### 4.1 Canonical Evidence Domain `E`

`Phase A` 的 concrete 输入不定义为“任意 lower 后 TIR”，而定义为：

- `E = (E_seed, E_frag, E_w, E_pre)`

其中：

- `E_seed`
  - pre-lift semantic seeds
- `E_frag`
  - selected fragment attrs
- `E_w`
  - canonicalized witness multiset，即 `tl.semantic_witnesses`
- `E_pre`
  - lift 合法的 precondition

关键点：

- `E` 是 `Phase A` 实际消费的 collecting domain
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

约束：

- `A_graph` 只允许表达 core 的 normalization
- `A_contract` 表示冻结 truth 的生命周期语义

### 4.3 Abstraction Function `alpha`

对当前仓库，`alpha` 最合适的定义不是单个文件里的纯函数，而是 pass composition：

- `alpha = BuildWitnesses ; LiftSemanticCore ; NormalizeStateEffect`

映射到当前实现：

- `AnalyzeSemanticStructure`
- `LiftStatefulSemanticIR`
- `semantic_state_effect_graph`

因此：

- `alpha(E)` 的结果就是当前 `tl.semantic_program`

但 `alpha` 只对满足 `E_pre` 的输入定义；否则应 reject / invalidate / defer。

### 4.4 Refinement Predicate `R(E, A)`

`R(E, A)` 的含义是：

- `A` 不伪造与 `E` 矛盾的 semantic facts
- `A_graph` 是 `A_core` 的合法 normalization
- `A_contract` 与 companion lifecycle 一致

当前实现里，`ValidateSemanticRefinement` 是 `R` 的 executable approximation。

## 5. Theorem Family

最值得追求的 theorem family 是：

- `T1` Evidence well-formedness
- `T2` Lift soundness
- `T3` Graph soundness
- `T4` Contract preservation
- `T5` Typed rebind preservation
- `T6` Invalidation safety
- `T7` `Phase A -> Phase B` refinement
- `T8` Rejection discipline

这些 theorem 的目标不是一开始全 mechanize，而是先和 executable validators 对齐。

## 6. Obligation Matrix for `R(E, A)`

可以把 `R(E, A)` 进一步拆成下面这些 obligation family：

- `R_vocab`
- `R_anchor`
- `R_role`
- `R_law`
- `R_source`
- `R_relation`
- `R_graph`
- `R_contract`
- `R_rebind`
- `R_reject`

它们分别对应：

- witness vocabulary closure
- anchor/related-object closure
- state role soundness
- update law soundness
- source-set soundness
- companion/carry relation soundness
- graph normalization soundness
- freeze contract legality
- typed rebind preservation obligation
- reject-rather-than-guess discipline

这一步的意义是：

- 把 `R(E, A)` 从一个抽象谓词收成可枚举 contract family
- 给 `ValidateSemanticRefinement` 提供明确扩展路线
- 给 `Phase B` refinement checker 提供稳定输入接口

## 7. Phase A to Phase B Interface

`Phase B` 的输入接口可以写成：

- `I_B = (A_core, A_graph, A_contract)`

也就是说：

- `Phase B` 不应该读 raw witness
- `Phase B` 不应该重新消费 raw fragment attrs 来发明 semantic truth
- `Phase B` 应该只消费已冻结的 algorithmic truth 及其 normalized state/effect skeleton

因此 `Phase A -> Phase B` 的正确关系是：

- `Phase B` 对 `Phase A` 做 **refinement by organization**

而不是：

- `Phase B` 再做一次 semantic recovery

## 8. Phase B Refinement Validator Skeleton

最自然的下一步是：

- `ValidateSpatialRefinement(SemanticProgram, SpatialProgram)`

它至少应检查：

- semantic coverage
- no semantic invention
- dependency preservation
- phase-boundary respect
- layout/partition non-interference
- fail-closed on missing structure

因此 `Phase B` 的两个关键组件应有清晰 contract：

- `lower_to_spatial_program`
  - 负责 `A -> S`
- `validate_spatial_program`
  - 负责检查 `S` 是否 refinement-preserve `A`

## 9. Failure Modes and Rejection Discipline

这条 formalization 轨道必须明确允许：

- reject
- invalidate
- defer

而不是默认“总能猜出来”。

主要 failure mode：

- evidence 不足
- semantic core 不足
- truth 属于 `Phase B/C`
- companion contract 被破坏

换句话说：

- 本工作不证明 completeness
- 本工作不保证没有 precision loss
- 本工作不证明 `Phase C` target materialization correctness

## 10. Minimal Honest External Claim

如果今天必须对外做一个学术上诚实的表述，最合理的版本是：

- 我们没有证明一个万能语义恢复器
- 我们定义并实现了一个针对 canonical evidence domain 的 bounded semantic abstraction layer
- 它带有 typed witness、normalized state/effect graph、以及 executable refinement contract
- 它为后续 `SpatialProgram` refinement 提供了冻结的 algorithmic truth boundary

## 11. Immediate Next Research Artifacts

最值得继续推进的 deliverable：

1. formal semantics note
2. obligation matrix
3. `Phase B` refinement validator
4. small mechanized core

如果要 mechanize，最小顺序建议是：

1. mechanize witness vocabulary/payload family
2. mechanize `SemanticProgram` core
3. mechanize state/effect graph normalization invariant
4. mechanize `ValidateSemanticRefinement` 对应 lemma

不要直接 mechanize 全部 TIR 或整条编译器。

## 12. References

- Cousot and Cousot, *Abstract Interpretation: A Unified Lattice Model for Static Analysis of Programs by Construction or Approximation of Fixpoints*, POPL 1977  
  https://www.di.ens.fr/~cousot/COUSOTpapers/POPL77.shtml
- Cousot, *Types as Abstract Interpretations*, POPL 1997  
  https://www.di.ens.fr/~cousot/COUSOTpapers/POPL97.shtml
- Cousot and Cousot, *Abstract Interpretation Algorithms and Proof Methods*, POPL 2014  
  https://www.di.ens.fr/~cousot/COUSOTpapers/POPL14.shtml
- Darais and Van Horn, *Constructive Galois Connections: Taming Proofs of Static Analyses*, ICFP 2016  
  https://arxiv.org/abs/1507.03559
- Lattner et al., *MLIR: Scaling Compiler Infrastructure for Domain Specific Computation*, 2020  
  https://arxiv.org/abs/2002.11054
- Wang et al., *Homeostasis: A Compiler Correctness Framework for Self-Regulating Program Analysis*, 2021  
  https://arxiv.org/abs/2106.01768
- Lopes et al., *Alive2: Bounded Translation Validation for LLVM*, PLDI 2021  
  https://pldi21.sigplan.org/details/pldi-2021-papers/5/Alive2-Bounded-Translation-Validation-for-LLVM
- Leroy, *Formal Verification of a Realistic Compiler*, CACM 2009  
  https://www.cs.cmu.edu/~15811/papers/compcert.pdf
- Lucassen and Gifford, *Polymorphic Effect Systems*, 1988  
  https://research.ibm.com/publications/polymorphic-effect-systems
- Tofte and Talpin, *Region-Based Memory Management*, 1997  
  https://www.sciencedirect.com/science/article/pii/S0890540196926139
