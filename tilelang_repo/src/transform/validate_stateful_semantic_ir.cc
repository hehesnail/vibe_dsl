/*
 * \file validate_stateful_semantic_ir.cc
 * \brief Minimal structural validation for Stage 4 A1 SemanticProgram.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>
#include <tvm/tir/transform.h>

#include <unordered_set>
#include <unordered_map>

#include "common/blackhole_utils.h"
#include "common/semantic_program.h"
#include "common/semantic_vocab.h"

namespace tvm {
namespace tl {

using namespace tvm::tl::semantic;
using tvm::ffi::String;

tir::transform::Pass ValidateStatefulSemanticIR() {
  auto fpass = [](tir::PrimFunc func, IRModule, tir::transform::PassContext) -> tir::PrimFunc {
    if (!IsBlackholePrimFunc(func)) {
      return func;
    }
    auto maybe_program = func->GetAttr<SemanticProgram>(attr::kTLSemanticProgram);
    ICHECK(maybe_program) << "ValidateStatefulSemanticIR requires tl.semantic_program";
    const SemanticProgram& program = maybe_program.value();
    ICHECK(!program->domains.empty()) << "SemanticProgram must contain at least one Domain";
    for (const State& state : program->states) {
      ICHECK(state.defined());
      ICHECK(ParseStateRole(static_cast<std::string>(state->role)))
          << "Unsupported State.role in A2 validator: " << state->role;
    }
    for (const Update& update : program->updates) {
      ICHECK(update.defined());
      ICHECK(update->law.defined()) << "SemanticProgram update must carry UpdateLaw";
      ICHECK(ParseUpdateLawKind(static_cast<std::string>(update->law->kind)))
          << "Unsupported UpdateLaw.kind in A1 validator: " << update->law->kind;
    }
    std::unordered_set<std::string> state_names;
    std::unordered_set<std::string> update_names;
    for (const State& state : program->states) {
      state_names.insert(state->name);
    }
    for (const Update& update : program->updates) {
      if (!std::string(update->state_name).empty()) {
        ICHECK(state_names.count(update->state_name))
            << "Update references missing target state: " << update->state_name;
      }
    }
    for (const Update& update : program->updates) {
      update_names.insert(update->name);
    }
    std::unordered_map<std::string, std::string> version_state_by_name;
    for (const StateVersion& version : program->state_versions) {
      ICHECK(ParseStateVersionKind(static_cast<std::string>(version->kind)))
          << "Unsupported StateVersion.kind: " << version->kind;
      ICHECK(state_names.count(version->state_name))
          << "StateVersion references missing state: " << version->state_name;
      if (!std::string(version->producer_update).empty()) {
        ICHECK(update_names.count(version->producer_update))
            << "StateVersion references missing producer update: " << version->producer_update;
      }
      version_state_by_name[version->name] = version->state_name;
    }
    for (const StateDef& def : program->state_defs) {
      ICHECK(ParseStateDefKind(static_cast<std::string>(def->kind)))
          << "Unsupported StateDef.kind: " << def->kind;
      ICHECK(state_names.count(def->state_name))
          << "StateDef references missing state: " << def->state_name;
      ICHECK(version_state_by_name.count(def->version_name))
          << "StateDef references missing version: " << def->version_name;
      ICHECK_EQ(version_state_by_name.at(def->version_name), std::string(def->state_name))
          << "StateDef state_name does not match version state";
      if (!std::string(def->producer_update).empty()) {
        ICHECK(update_names.count(def->producer_update))
            << "StateDef references missing producer update: " << def->producer_update;
      }
    }
    for (const StateUse& use : program->state_uses) {
      ICHECK(ParseStateUseKind(static_cast<std::string>(use->kind)))
          << "Unsupported StateUse.kind: " << use->kind;
      ICHECK(update_names.count(use->consumer_update))
          << "StateUse references missing consumer update: " << use->consumer_update;
      ICHECK(state_names.count(use->state_name))
          << "StateUse references missing state: " << use->state_name;
      ICHECK(version_state_by_name.count(use->version_name))
          << "StateUse references missing version: " << use->version_name;
      ICHECK_EQ(version_state_by_name.at(use->version_name), std::string(use->state_name))
          << "StateUse state_name does not match version state";
    }
    for (const StateJoin& join : program->state_joins) {
      ICHECK(ParseStateJoinKind(static_cast<std::string>(join->kind)))
          << "Unsupported StateJoin.kind: " << join->kind;
      ICHECK(state_names.count(join->state_name))
          << "StateJoin references missing state: " << join->state_name;
      ICHECK(version_state_by_name.count(join->output_version))
          << "StateJoin references missing output version: " << join->output_version;
      ICHECK_EQ(version_state_by_name.at(join->output_version), std::string(join->state_name))
          << "StateJoin output_version does not match join state";
      for (const String& input_version : join->input_versions) {
        ICHECK(version_state_by_name.count(input_version))
            << "StateJoin references missing input version: " << input_version;
        ICHECK_EQ(version_state_by_name.at(input_version), std::string(join->state_name))
            << "StateJoin input_version does not match join state";
      }
    }
    return func;
  };
  return tir::transform::CreatePrimFuncPass(
      fpass, 0, "tl.transform.ValidateStatefulSemanticIR", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.ValidateStatefulSemanticIR", ValidateStatefulSemanticIR);
}

}  // namespace tl
}  // namespace tvm
