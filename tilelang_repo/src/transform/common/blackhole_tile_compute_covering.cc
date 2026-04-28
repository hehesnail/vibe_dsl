/*!
 * \file blackhole_tile_compute_covering.cc
 * \brief Local Blackhole tile compute pattern covering selection.
 */

#include "blackhole_tile_compute_covering.h"

#include "blackhole_tile_compute_dag.h"
#include "blackhole_tile_compute_legalizer.h"
#include "blackhole_tile_compute_patterns.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

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
  if (pattern.operation == BlackholeTileComputeOperation::kCopyTile ||
      pattern.operation == BlackholeTileComputeOperation::kTypecastTile ||
      pattern.operation == BlackholeTileComputeOperation::kPackTile) {
    return "materialization_boundary_required_when_cross_phase";
  }
  if (pattern.side_effect_class == BlackholeTileComputeSideEffectClass::kTileRegs ||
      pattern.side_effect_class == BlackholeTileComputeSideEffectClass::kDst) {
    return "live_form_solver_required_for_cross_event_use";
  }
  return "none";
}

bool IsDAGOutputRole(const std::string& role) {
  return role == "output" || role == "c";
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

ffi::Map<ffi::String, ffi::Any> EncodeSelectedNodeDecision(
    const BlackholeTileComputeCoveringDecision& decision,
    const BlackholeTileComputeDAGNode& node, int64_t order_index) {
  ffi::Map<ffi::String, ffi::Any> encoded =
      EncodeBlackholeTileComputeCoveringDecision(decision);
  encoded.Set(ffi::String("node_id"), Integer(node.id));
  encoded.Set(ffi::String("selection_order_index"), Integer(order_index));
  encoded.Set(ffi::String("node_side_effect_class"),
              ffi::String(node.side_effect_class));
  encoded.Set(ffi::String("dp_state_key"),
              ffi::String(std::to_string(node.id) + ":" + decision.result_kind));
  return encoded;
}

ffi::Map<ffi::String, ffi::Any> EncodeMaterializationDecision(
    const BlackholeTileComputeCoveringDecision& decision,
    const BlackholeTileComputeDAGNode& node) {
  ffi::Map<ffi::String, ffi::Any> encoded;
  encoded.Set(ffi::String("node_id"), Integer(node.id));
  encoded.Set(ffi::String("operation_name"), ffi::String(decision.operation_name));
  encoded.Set(ffi::String("pattern_name"), ffi::String(decision.pattern_name));
  encoded.Set(ffi::String("policy"), ffi::String(decision.materialization_policy));
  encoded.Set(ffi::String("evidence"),
              ffi::String("selected_pattern:" + decision.pattern_name +
                          ";side_effect:" + node.side_effect_class));
  return encoded;
}

std::vector<ffi::Map<ffi::String, ffi::Any>> BuildFanoutDecisions(
    const BlackholeTileComputeDAG& dag) {
  std::unordered_map<int64_t, std::vector<const BlackholeTileComputeDAGEdge*>>
      uses_by_producer;
  for (const BlackholeTileComputeDAGEdge& edge : dag.edges) {
    if (edge.producer_node < 0 || IsDAGOutputRole(edge.value_role)) {
      continue;
    }
    uses_by_producer[edge.producer_node].push_back(&edge);
  }

  std::vector<ffi::Map<ffi::String, ffi::Any>> decisions;
  for (const auto& entry : uses_by_producer) {
    const int64_t producer_node = entry.first;
    const std::vector<const BlackholeTileComputeDAGEdge*>& uses = entry.second;
    if (uses.size() < 2U || producer_node < 0 ||
        producer_node >= static_cast<int64_t>(dag.nodes.size())) {
      continue;
    }
    const BlackholeTileComputeDAGNode& producer = dag.nodes[producer_node];
    const bool requires_materialization =
        producer.side_effect_class == "tile_regs" ||
        producer.side_effect_class == "dst" ||
        producer.side_effect_class == "pack";
    ffi::Array<Integer> consumer_nodes;
    for (const BlackholeTileComputeDAGEdge* use : uses) {
      consumer_nodes.push_back(Integer(use->consumer_node));
    }
    ffi::Map<ffi::String, ffi::Any> decision;
    decision.Set(ffi::String("producer_node"), Integer(producer_node));
    decision.Set(ffi::String("producer_operation"), ffi::String(producer.op_name));
    decision.Set(ffi::String("value_repr"), ffi::String(uses.front()->value_repr));
    decision.Set(ffi::String("use_count"), Integer(static_cast<int64_t>(uses.size())));
    decision.Set(ffi::String("consumer_nodes"), consumer_nodes);
    decision.Set(ffi::String("policy"),
                 ffi::String(requires_materialization
                                 ? "materialize_before_cross_event_use"
                                 : "share_live_value"));
    decision.Set(ffi::String("evidence"),
                 ffi::String("producer_use_count:" + std::to_string(uses.size()) +
                             ";producer_side_effect:" + producer.side_effect_class));
    decisions.push_back(decision);
  }
  std::sort(decisions.begin(), decisions.end(),
            [](const ffi::Map<ffi::String, ffi::Any>& lhs,
               const ffi::Map<ffi::String, ffi::Any>& rhs) {
              const int64_t lhs_node =
                  Downcast<Integer>(lhs.Get(ffi::String("producer_node")).value())->value;
              const int64_t rhs_node =
                  Downcast<Integer>(rhs.Get(ffi::String("producer_node")).value())->value;
              return lhs_node < rhs_node;
            });
  return decisions;
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
      LegalizeBlackholeTileComputeSelection(
          ToString(pattern->result_kind),
          ToString(pattern->operation),
          BlackholeTileComputeOperandRoleNames(pattern->operand_roles));
  if (!legality.IsLegal()) {
    return RejectCovering(operation_name, legality.reason);
  }
  BlackholeTileComputeCoveringDecision decision;
  decision.selected = true;
  decision.selection_kind = "selected_pattern";
  decision.pattern_name = pattern->name;
  decision.operation_name = ToString(pattern->operation);
  decision.result_kind = ToString(pattern->result_kind);
  decision.operand_roles = BlackholeTileComputeOperandRoleNames(pattern->operand_roles);
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
  encoded.Set(ffi::String("source_emitter"),
              ffi::String(decision.source_emitter
                              ? ToString(*decision.source_emitter)
                              : ""));
  encoded.Set(ffi::String("materialization_policy"),
              ffi::String(decision.materialization_policy));
  encoded.Set(ffi::String("cost"), Integer(decision.cost));
  encoded.Set(ffi::String("reject_reason"), ffi::String(decision.reject_reason));
  return encoded;
}

ffi::Map<ffi::String, ffi::Any> SelectBlackholeTileComputeDAGCoveringDiagnostic(
    const tir::PrimFunc& func) {
  const BlackholeTileComputeDAG dag = BuildBlackholeTileComputeDAG(func);
  ffi::Array<ffi::Map<ffi::String, ffi::Any>> selected_patterns;
  ffi::Array<ffi::Map<ffi::String, ffi::Any>> materialization_decisions;
  ffi::Array<ffi::String> unsupported_reasons;
  int64_t total_cost = 0;
  bool selected = true;

  for (size_t order_index = 0; order_index < dag.nodes.size(); ++order_index) {
    const BlackholeTileComputeDAGNode& node = dag.nodes[order_index];
    const BlackholeTileComputeCoveringDecision decision =
        SelectBlackholeTileComputeCovering(node.op_name);
    if (!decision.selected) {
      selected = false;
      unsupported_reasons.push_back(
          ffi::String("node " + std::to_string(node.id) + " operation " +
                      node.op_name + ": " + decision.reject_reason));
      continue;
    }
    total_cost += decision.cost;
    selected_patterns.push_back(
        EncodeSelectedNodeDecision(decision, node, static_cast<int64_t>(order_index)));
    if (decision.materialization_policy != "none") {
      materialization_decisions.push_back(
          EncodeMaterializationDecision(decision, node));
    }
  }

  const std::vector<ffi::Map<ffi::String, ffi::Any>> fanout =
      BuildFanoutDecisions(dag);
  ffi::Array<ffi::Map<ffi::String, ffi::Any>> fanout_decisions;
  for (const ffi::Map<ffi::String, ffi::Any>& decision : fanout) {
    fanout_decisions.push_back(decision);
    if (static_cast<std::string>(
            Downcast<ffi::String>(decision.Get(ffi::String("policy")).value())) ==
        "materialize_before_cross_event_use") {
      total_cost += 1;
    }
  }

  ffi::Map<ffi::String, ffi::Any> encoded;
  encoded.Set(ffi::String("selection_kind"), ffi::String("local_dag_dp"));
  encoded.Set(ffi::String("selection_status"),
              ffi::String(selected ? "selected" : "rejected"));
  encoded.Set(ffi::String("selection_order"), ffi::String("dependence_order"));
  encoded.Set(ffi::String("selected_patterns"), selected_patterns);
  encoded.Set(ffi::String("fanout_decisions"), fanout_decisions);
  encoded.Set(ffi::String("materialization_decisions"), materialization_decisions);
  encoded.Set(ffi::String("unsupported_reasons"), unsupported_reasons);
  encoded.Set(ffi::String("total_cost"), Integer(total_cost));
  encoded.Set(ffi::String("stale_fallback_policy"), ffi::String("reject"));
  return encoded;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.SelectBlackholeTileComputeCoveringDiagnostic",
                        [](ffi::String operation_name) {
                          return EncodeBlackholeTileComputeCoveringDecision(
                              SelectBlackholeTileComputeCovering(
                                  static_cast<std::string>(operation_name)));
                        })
      .def("tl.SelectBlackholeTileComputeDAGCoveringDiagnostic",
           SelectBlackholeTileComputeDAGCoveringDiagnostic);
}

}  // namespace tl
}  // namespace tvm
