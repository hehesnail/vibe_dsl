/*!
 * \file semantic_state_effect_graph.h
 * \brief Build/query helpers for SemanticProgram internal state/effect graph.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_SEMANTIC_STATE_EFFECT_GRAPH_H_
#define TVM_TL_TRANSFORM_COMMON_SEMANTIC_STATE_EFFECT_GRAPH_H_

#include "semantic_program.h"

namespace tvm {
namespace tl {
namespace semantic {

struct BuiltStateEffectGraph {
  ffi::Array<StateVersion> state_versions;
  ffi::Array<StateDef> state_defs;
  ffi::Array<StateUse> state_uses;
  ffi::Array<StateJoin> state_joins;
};

TVM_DLL BuiltStateEffectGraph BuildStateEffectGraph(const ffi::Array<State>& states,
                                                    const ffi::Array<Update>& updates,
                                                    const ffi::Array<SemanticWitness>& witnesses);

}  // namespace semantic
}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_SEMANTIC_STATE_EFFECT_GRAPH_H_
