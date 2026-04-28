/*!
 * \file blackhole_tile_compute_covering.cc
 * \brief Local Blackhole tile compute pattern covering selection.
 */

#include "blackhole_tile_compute_covering.h"

#include "blackhole_tile_compute_legalizer.h"
#include "blackhole_tile_compute_patterns.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>

namespace tvm {
namespace tl {
namespace {

ffi::Array<ffi::String> EncodeStringVector(const std::vector<std::string>& values) {
  ffi::Array<ffi::String> encoded;
  for (const std::string& value : values) {
    encoded.push_back(ffi::String(value));
  }
  return encoded;
}

std::string MaterializationPolicyForPattern(const BlackholeTileComputePattern& pattern) {
  if (pattern.operation_name == "copy_tile" ||
      pattern.operation_name == "typecast_tile" ||
      pattern.operation_name == "pack_tile") {
    return "materialization_boundary_required_when_cross_phase";
  }
  if (pattern.side_effect_class == "tile_regs" || pattern.side_effect_class == "dst") {
    return "live_form_solver_required_for_cross_event_use";
  }
  return "none";
}

BlackholeTileComputeCoveringDecision RejectCovering(const std::string& operation_name,
                                                    const std::string& reason) {
  BlackholeTileComputeCoveringDecision decision;
  decision.selected = false;
  decision.selection_kind = "reject";
  decision.operation_name = operation_name;
  decision.reject_reason = reason;
  return decision;
}

}  // namespace

BlackholeTileComputeCoveringDecision SelectBlackholeTileComputeCovering(
    const std::string& operation_name) {
  const BlackholeTileComputePattern* pattern =
      FindBlackholeTileComputePattern(operation_name);
  if (pattern == nullptr) {
    return RejectCovering(operation_name, "no leaf pattern covers operation");
  }
  const BlackholeTileLegalizationDiagnostic legality =
      LegalizeBlackholeTileComputeSelection(pattern->result_kind,
                                            pattern->operation_name,
                                            pattern->operand_roles);
  if (!legality.IsLegal()) {
    return RejectCovering(operation_name, legality.reason);
  }
  BlackholeTileComputeCoveringDecision decision;
  decision.selected = true;
  decision.selection_kind = "selected_pattern";
  decision.pattern_name = pattern->name;
  decision.operation_name = pattern->operation_name;
  decision.result_kind = pattern->result_kind;
  decision.operand_roles = pattern->operand_roles;
  decision.selected_output = "tt_compute_op_plan";
  decision.source_emitter = pattern->source_emitter;
  decision.materialization_policy = MaterializationPolicyForPattern(*pattern);
  decision.cost = pattern->base_cost;
  return decision;
}

ffi::Map<ffi::String, ffi::Any> EncodeBlackholeTileComputeCoveringDecision(
    const BlackholeTileComputeCoveringDecision& decision) {
  ffi::Map<ffi::String, ffi::Any> encoded;
  encoded.Set(ffi::String("selected"), Bool(decision.selected));
  encoded.Set(ffi::String("selection_kind"), ffi::String(decision.selection_kind));
  encoded.Set(ffi::String("pattern_name"), ffi::String(decision.pattern_name));
  encoded.Set(ffi::String("operation_name"), ffi::String(decision.operation_name));
  encoded.Set(ffi::String("result_kind"), ffi::String(decision.result_kind));
  encoded.Set(ffi::String("operand_roles"), EncodeStringVector(decision.operand_roles));
  encoded.Set(ffi::String("selected_output"), ffi::String(decision.selected_output));
  encoded.Set(ffi::String("source_emitter"), ffi::String(decision.source_emitter));
  encoded.Set(ffi::String("materialization_policy"),
              ffi::String(decision.materialization_policy));
  encoded.Set(ffi::String("cost"), Integer(decision.cost));
  encoded.Set(ffi::String("reject_reason"), ffi::String(decision.reject_reason));
  return encoded;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.SelectBlackholeTileComputeCoveringDiagnostic",
                        [](ffi::String operation_name) {
                          return EncodeBlackholeTileComputeCoveringDecision(
                              SelectBlackholeTileComputeCovering(
                                  static_cast<std::string>(operation_name)));
                        });
}

}  // namespace tl
}  // namespace tvm
