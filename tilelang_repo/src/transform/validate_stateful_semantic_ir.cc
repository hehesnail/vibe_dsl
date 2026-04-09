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
using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
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
    for (const SemanticSupplement& supplement : program->supplements) {
      auto kind = ParseSupplementKind(static_cast<std::string>(supplement->kind));
      ICHECK(kind) << "Unsupported SemanticSupplement kind in A2 validator: "
                   << supplement->kind;
      if (*kind == SupplementKind::kFragmentLoweringStructure) {
        auto maybe_fragment_ops =
            supplement->payload.Get(String(schema_key::kFragmentOpKinds));
        ICHECK(maybe_fragment_ops)
            << "fragment_lowering_structure supplement must carry fragment_op_kinds";
        Array<Any> fragment_ops = tvm::Downcast<Array<Any>>(maybe_fragment_ops.value());
        ICHECK(!fragment_ops.empty())
            << "fragment_lowering_structure supplement must carry at least one fragment op";
        bool requires_pointwise_payload = false;
        bool requires_row_broadcast_payload = false;
        for (const Any& op_any : fragment_ops) {
          const std::string op_name = tvm::Downcast<String>(op_any);
          requires_pointwise_payload |= op_name == "pointwise_chain";
          requires_row_broadcast_payload |= op_name == "row_broadcast";
        }
        if (requires_pointwise_payload) {
          auto maybe_pointwise_ops =
              supplement->payload.Get(String(schema_key::kPointwiseOpKinds));
          ICHECK(maybe_pointwise_ops)
              << "fragment_lowering_structure pointwise_chain must carry pointwise_op_kinds";
          ICHECK(!tvm::Downcast<Array<Any>>(maybe_pointwise_ops.value()).empty())
              << "fragment_lowering_structure pointwise_op_kinds must be non-empty";
        }
        if (requires_row_broadcast_payload) {
          auto maybe_row_broadcast_sources =
              supplement->payload.Get(String(schema_key::kRowBroadcastSources));
          ICHECK(maybe_row_broadcast_sources)
              << "fragment_lowering_structure row_broadcast must carry row_broadcast_sources";
          ICHECK(!tvm::Downcast<Array<Any>>(maybe_row_broadcast_sources.value()).empty())
              << "fragment_lowering_structure row_broadcast_sources must be non-empty";
        }
        if (auto maybe_materialization_contracts =
                supplement->payload.Get(String(schema_key::kFragmentMaterializationContracts))) {
          Array<Any> contracts = tvm::Downcast<Array<Any>>(maybe_materialization_contracts.value());
          ICHECK(!contracts.empty())
              << "fragment_lowering_structure fragment_materialization_contracts must be non-empty";
          for (const Any& contract_any : contracts) {
            Map<String, Any> contract = tvm::Downcast<Map<String, Any>>(contract_any);
            ICHECK(contract.count(String(schema_key::kKind)))
                << "fragment_materialization_contract must carry kind";
            ICHECK(contract.count(String(schema_key::kTargetBuffer)))
                << "fragment_materialization_contract must carry target_buffer";
            ICHECK(contract.count(String(schema_key::kScope)))
                << "fragment_materialization_contract must carry scope";
            ICHECK(contract.count(String(schema_key::kMaterializationKind)))
                << "fragment_materialization_contract must carry materialization_kind";
            ICHECK(contract.count(String(schema_key::kValueRole)))
                << "fragment_materialization_contract must carry value_role";
            ICHECK(contract.count(String(schema_key::kMergeKind)))
                << "fragment_materialization_contract must carry merge_kind";
          }
        }
        if (auto maybe_flow_contracts =
                supplement->payload.Get(String(schema_key::kFragmentBufferFlowContracts))) {
          Array<Any> contracts = tvm::Downcast<Array<Any>>(maybe_flow_contracts.value());
          ICHECK(!contracts.empty())
              << "fragment_lowering_structure fragment_buffer_flow_contracts must be non-empty";
          for (const Any& contract_any : contracts) {
            Map<String, Any> contract = tvm::Downcast<Map<String, Any>>(contract_any);
            ICHECK(contract.count(String(schema_key::kBuffer)))
                << "fragment_buffer_flow_contract must carry buffer";
            ICHECK(contract.count(String(schema_key::kScope)))
                << "fragment_buffer_flow_contract must carry scope";
            ICHECK(contract.count(String(schema_key::kFlowClass)))
                << "fragment_buffer_flow_contract must carry flow_class";
            ICHECK(contract.count(String(schema_key::kGranuleKind)))
                << "fragment_buffer_flow_contract must carry granule_kind";
            ICHECK(contract.count(String(schema_key::kPublishGranule)))
                << "fragment_buffer_flow_contract must carry publish_granule";
            ICHECK(contract.count(String(schema_key::kConsumeGranule)))
                << "fragment_buffer_flow_contract must carry consume_granule";
            auto maybe_events = contract.Get(String(schema_key::kEvents));
            ICHECK(maybe_events)
                << "fragment_buffer_flow_contract must carry events";
            Array<Any> events = tvm::Downcast<Array<Any>>(maybe_events.value());
            ICHECK(!events.empty())
                << "fragment_buffer_flow_contract events must be non-empty";
            for (const Any& event_any : events) {
              Map<String, Any> event = tvm::Downcast<Map<String, Any>>(event_any);
              ICHECK(event.count(String(schema_key::kKind)))
                  << "fragment_buffer_flow_contract event must carry kind";
              ICHECK(event.count(String(schema_key::kOrderIndex)))
                  << "fragment_buffer_flow_contract event must carry order_index";
            }
          }
        }
      } else if (*kind == SupplementKind::kPipelineStructure) {
        auto maybe_pipeline_stages = supplement->payload.Get(String(schema_key::kPipelineStages));
        ICHECK(maybe_pipeline_stages)
            << "pipeline_structure supplement must carry pipeline_stages payload";
        Array<Any> pipeline_stages = tvm::Downcast<Array<Any>>(maybe_pipeline_stages.value());
        ICHECK(!pipeline_stages.empty())
            << "pipeline_structure supplement must carry at least one pipeline stage";
        for (const Any& stage_any : pipeline_stages) {
          auto stage = tvm::Downcast<Map<String, Any>>(stage_any);
          ICHECK(stage.count(String(schema_key::kLoopVar)))
              << "pipeline_structure stage must carry loop_var";
          ICHECK(stage.count(String(schema_key::kNumStages)))
            << "pipeline_structure stage must carry num_stages";
        }
      } else if (*kind == SupplementKind::kWorkDecompositionStructure) {
        auto maybe_loop_bounds =
            supplement->payload.Get(String(schema_key::kWorkDependentLoopBounds));
        ICHECK(maybe_loop_bounds)
            << "work_decomposition_structure supplement must carry work_dependent_loop_bounds";
        Array<Any> loop_bounds = tvm::Downcast<Array<Any>>(maybe_loop_bounds.value());
        ICHECK(!loop_bounds.empty())
            << "work_decomposition_structure supplement must carry at least one loop bound";
        for (const Any& bound_any : loop_bounds) {
          auto bound = tvm::Downcast<Map<String, Any>>(bound_any);
          ICHECK(bound.count(String(schema_key::kLoopVar)))
              << "work_decomposition_structure bound must carry loop_var";
        }
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
