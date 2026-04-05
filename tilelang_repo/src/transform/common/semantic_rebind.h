/*!
 * \file semantic_rebind.h
 * \brief Typed rebind helpers for Blackhole semantic companion programs.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_SEMANTIC_REBIND_H_
#define TVM_TL_TRANSFORM_COMMON_SEMANTIC_REBIND_H_

#include <unordered_map>

#include "semantic_program.h"
#include "semantic_vocab.h"

namespace tvm {
namespace tl {
namespace semantic {

struct SemanticRebindPlan {
  RebindScope scope;
  std::string reason;
  ffi::Array<ffi::Any> trace;
  std::unordered_map<std::string, std::string> state_remap;
  std::unordered_map<std::string, std::string> update_remap;
};

TVM_DLL SemanticRebindPlan DecodeSemanticRebindPlan(const ffi::Map<ffi::String, ffi::Any>& plan);
TVM_DLL ffi::Map<ffi::String, ffi::Any> ApplySemanticRebindToStructure(
    const ffi::Map<ffi::String, ffi::Any>& structure, const SemanticRebindPlan& plan);
TVM_DLL ffi::Array<SemanticWitness> ApplySemanticRebindToWitnesses(
    const ffi::Array<SemanticWitness>& witnesses, const SemanticRebindPlan& plan);
TVM_DLL SemanticProgram ApplySemanticRebindToProgram(const SemanticProgram& program,
                                                     const ffi::Array<SemanticWitness>& witnesses,
                                                     const SemanticRebindPlan& plan);

}  // namespace semantic
}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_SEMANTIC_REBIND_H_
