/*!
 * \file validate_spatial_program.cc
 * \brief Validate minimal Phase B SpatialProgram invariants.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>

#include <string>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/blackhole_utils.h"
#include "common/semantic_program.h"
#include "common/semantic_vocab.h"

namespace tvm {
namespace tl {

using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::String;
using tvm::Integer;

namespace {

bool HasTrait(const Array<String>& traits, const char* expected) {
  for (const String& trait : traits) {
    if (static_cast<std::string>(trait) == expected) {
      return true;
    }
  }
  return false;
}

bool SameAxes(const Array<String>& lhs, const Array<String>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (int i = 0; i < lhs.size(); ++i) {
    if (static_cast<std::string>(lhs[i]) != static_cast<std::string>(rhs[i])) {
      return false;
    }
  }
  return true;
}

bool SameStringArray(const Array<String>& lhs, const Array<String>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (int i = 0; i < lhs.size(); ++i) {
    if (static_cast<std::string>(lhs[i]) != static_cast<std::string>(rhs[i])) {
      return false;
    }
  }
  return true;
}

bool SameIntegerAnyArray(const Array<Any>& lhs, const Array<Any>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (int i = 0; i < lhs.size(); ++i) {
    if (Downcast<Integer>(lhs[i])->value != Downcast<Integer>(rhs[i])->value) {
      return false;
    }
  }
  return true;
}

std::optional<int64_t> GetPayloadIndex(const Map<String, Any>& payload, const char* key);

bool SamePhaseSignature(const ProgramPhase& lhs, const ProgramPhase& rhs) {
  auto lhs_phase_index = GetPayloadIndex(lhs->payload, schema_key::kPhaseIndex);
  auto rhs_phase_index = GetPayloadIndex(rhs->payload, schema_key::kPhaseIndex);
  auto lhs_task_indices = lhs->payload.Get(String(schema_key::kTaskIndices));
  auto rhs_task_indices = rhs->payload.Get(String(schema_key::kTaskIndices));
  auto lhs_channel_indices = lhs->payload.Get(String(schema_key::kChannelIndices));
  auto rhs_channel_indices = rhs->payload.Get(String(schema_key::kChannelIndices));
  return static_cast<std::string>(lhs->name) == static_cast<std::string>(rhs->name) &&
         SameStringArray(lhs->task_names, rhs->task_names) &&
         SameStringArray(lhs->channel_names, rhs->channel_names) &&
         lhs_phase_index == rhs_phase_index &&
         lhs_task_indices.has_value() == rhs_task_indices.has_value() &&
         lhs_channel_indices.has_value() == rhs_channel_indices.has_value() &&
         (!lhs_task_indices.has_value() ||
          SameIntegerAnyArray(Downcast<Array<Any>>(lhs_task_indices.value()),
                              Downcast<Array<Any>>(rhs_task_indices.value()))) &&
         (!lhs_channel_indices.has_value() ||
          SameIntegerAnyArray(Downcast<Array<Any>>(lhs_channel_indices.value()),
                              Downcast<Array<Any>>(rhs_channel_indices.value())));
}

bool IsPipelineContractIntent(const ResourceIntent& intent) {
  return static_cast<std::string>(intent->kind) == "synchronization_support" &&
         HasTrait(intent->traits, "pipeline_contract");
}

bool IsFragmentContractIntent(const ResourceIntent& intent) {
  return static_cast<std::string>(intent->kind) == "lowering_support" &&
         HasTrait(intent->traits, "fragment_contract");
}

std::optional<int64_t> GetPayloadIndex(const Map<String, Any>& payload, const char* key) {
  if (auto value = payload.Get(String(key))) {
    return Downcast<Integer>(value.value())->value;
  }
  return std::nullopt;
}

std::optional<std::string> GetPayloadString(const Map<String, Any>& payload, const char* key) {
  if (auto value = payload.Get(String(key))) {
    return static_cast<std::string>(Downcast<String>(value.value()));
  }
  return std::nullopt;
}

std::optional<std::vector<int64_t>> GetPayloadIndices(const Map<String, Any>& payload, const char* key) {
  if (auto value = payload.Get(String(key))) {
    std::vector<int64_t> result;
    for (const Any& item : Downcast<Array<Any>>(value.value())) {
      result.push_back(Downcast<Integer>(item)->value);
    }
    return result;
  }
  return std::nullopt;
}

}  // namespace

tvm::transform::Pass ValidateSpatialProgram() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext) {
    std::unordered_map<std::string, Array<ProgramPhase>> phases_by_member_func;
    for (const auto& [gvar, base_func] : mod->functions) {
      auto func = base_func.as<tir::PrimFunc>();
      if (!func || !IsBlackholePrimFunc(func.value())) {
        continue;
      }
      auto maybe_program = func.value()->GetAttr<SpatialProgram>(attr::kTLSpatialProgram);
      if (!maybe_program) {
        continue;
      }
      SpatialProgram program = maybe_program.value();
      const std::string member_func =
          func.value()->GetAttr<String>(tvm::attr::kGlobalSymbol).value_or(gvar->name_hint);
      ICHECK_EQ(static_cast<std::string>(program->member_func), member_func)
          << "ValidateSpatialProgram requires SpatialProgram.member_func to match "
             "PrimFunc global_symbol";
      ICHECK(!program->phases.empty()) << "ValidateSpatialProgram requires at least one phase";
      ICHECK(!program->tasks.empty()) << "ValidateSpatialProgram requires at least one task";
      ICHECK(!program->layouts.empty())
          << "ValidateSpatialProgram requires at least one spatial layout";
      ICHECK(!program->work_partitions.empty())
          << "ValidateSpatialProgram requires at least one work partition";

      std::unordered_set<std::string> phase_names;
      std::vector<std::string> phase_name_by_index(program->phases.size());
      std::vector<std::vector<int64_t>> phase_task_indices_by_index(program->phases.size());
      std::vector<std::vector<int64_t>> phase_channel_indices_by_index(program->phases.size());
      for (const ProgramPhase& phase : program->phases) {
        const std::string phase_name = static_cast<std::string>(phase->name);
        ICHECK(phase_names.insert(phase_name).second)
            << "ValidateSpatialProgram found duplicate phase " << phase_name;
        auto maybe_phase_index = GetPayloadIndex(phase->payload, schema_key::kPhaseIndex);
        ICHECK(maybe_phase_index)
            << "ValidateSpatialProgram requires program phases to carry phase_index contract";
        ICHECK_GE(*maybe_phase_index, 0);
        ICHECK_LT(*maybe_phase_index, program->phases.size())
            << "ValidateSpatialProgram found program phase with invalid phase_index";
        auto maybe_task_indices = GetPayloadIndices(phase->payload, schema_key::kTaskIndices);
        ICHECK(maybe_task_indices)
            << "ValidateSpatialProgram requires program phases to carry task_indices contract";
        auto maybe_channel_indices = GetPayloadIndices(phase->payload, schema_key::kChannelIndices);
        ICHECK(maybe_channel_indices)
            << "ValidateSpatialProgram requires program phases to carry channel_indices contract";
        phase_name_by_index[*maybe_phase_index] = phase_name;
        phase_task_indices_by_index[*maybe_phase_index] = std::move(*maybe_task_indices);
        phase_channel_indices_by_index[*maybe_phase_index] = std::move(*maybe_channel_indices);
      }

      std::unordered_set<std::string> task_names;
      std::vector<std::string> task_name_by_index(program->tasks.size());
      std::vector<int64_t> task_phase_index_by_index(program->tasks.size(), -1);
      for (int task_index = 0; task_index < program->tasks.size(); ++task_index) {
        const Task& task = program->tasks[task_index];
        const std::string task_name = static_cast<std::string>(task->name);
        ICHECK(task_names.insert(task_name).second)
            << "ValidateSpatialProgram found duplicate task " << task_name;
        auto maybe_phase_index = GetPayloadIndex(task->payload, schema_key::kPhaseIndex);
        ICHECK(maybe_phase_index)
            << "ValidateSpatialProgram requires tasks to carry phase_index contract";
        ICHECK_GE(*maybe_phase_index, 0);
        ICHECK_LT(*maybe_phase_index, program->phases.size())
            << "ValidateSpatialProgram found task with invalid phase_index";
        const std::string phase_name = phase_name_by_index[*maybe_phase_index];
        ICHECK(!phase_name.empty())
            << "ValidateSpatialProgram found task referencing unresolved phase_index";
        ICHECK_EQ(static_cast<std::string>(task->phase_name), phase_name)
            << "ValidateSpatialProgram found task phase_name inconsistent with phase_index contract";
        task_name_by_index[task_index] = task_name;
        task_phase_index_by_index[task_index] = *maybe_phase_index;
      }

      std::unordered_set<std::string> channel_names;
      std::vector<std::string> channel_name_by_index(program->channels.size());
      std::vector<int64_t> channel_target_task_index_by_index(program->channels.size(), -1);
      for (int channel_index = 0; channel_index < program->channels.size(); ++channel_index) {
        const Channel& channel = program->channels[channel_index];
        const std::string channel_name = static_cast<std::string>(channel->name);
        ICHECK(channel_names.insert(channel_name).second)
            << "ValidateSpatialProgram found duplicate channel " << channel_name;
        auto maybe_source_task_index =
            GetPayloadIndex(channel->payload, schema_key::kSourceTaskIndex);
        auto maybe_target_task_index =
            GetPayloadIndex(channel->payload, schema_key::kTargetTaskIndex);
        ICHECK(maybe_source_task_index && maybe_target_task_index)
            << "ValidateSpatialProgram requires channels to carry "
               "source_task_index/target_task_index contract";
        ICHECK_GE(*maybe_source_task_index, 0);
        ICHECK_LT(*maybe_source_task_index, program->tasks.size())
            << "ValidateSpatialProgram found channel with invalid source_task_index";
        ICHECK_GE(*maybe_target_task_index, 0);
        ICHECK_LT(*maybe_target_task_index, program->tasks.size())
            << "ValidateSpatialProgram found channel with invalid target_task_index";
        ICHECK_EQ(static_cast<std::string>(channel->source_task),
                  task_name_by_index[*maybe_source_task_index])
            << "ValidateSpatialProgram found channel source_task inconsistent with source_task_index";
        ICHECK_EQ(static_cast<std::string>(channel->target_task),
                  task_name_by_index[*maybe_target_task_index])
            << "ValidateSpatialProgram found channel target_task inconsistent with target_task_index";
        channel_name_by_index[channel_index] = channel_name;
        channel_target_task_index_by_index[channel_index] = *maybe_target_task_index;
      }

      for (const Placement& placement : program->placements) {
        auto maybe_task_index = GetPayloadIndex(placement->payload, schema_key::kTaskIndex);
        ICHECK(maybe_task_index)
            << "ValidateSpatialProgram requires placements to carry task_index contract";
        ICHECK_GE(*maybe_task_index, 0);
        ICHECK_LT(*maybe_task_index, program->tasks.size())
            << "ValidateSpatialProgram found placement with invalid task_index";
        ICHECK_EQ(static_cast<std::string>(placement->task_name), task_name_by_index[*maybe_task_index])
            << "ValidateSpatialProgram found placement task_name inconsistent with task_index";
        ICHECK_EQ(static_cast<std::string>(placement->member_func), member_func)
            << "ValidateSpatialProgram requires placement.member_func to match "
               "SpatialProgram.member_func";
      }

      for (const SyncEdge& edge : program->sync_edges) {
        auto maybe_source_task_index = GetPayloadIndex(edge->payload, schema_key::kSourceTaskIndex);
        auto maybe_target_task_index = GetPayloadIndex(edge->payload, schema_key::kTargetTaskIndex);
        ICHECK(maybe_source_task_index && maybe_target_task_index)
            << "ValidateSpatialProgram requires sync edges to carry "
               "source_task_index/target_task_index contract";
        ICHECK_GE(*maybe_source_task_index, 0);
        ICHECK_LT(*maybe_source_task_index, program->tasks.size())
            << "ValidateSpatialProgram found sync edge with invalid source_task_index";
        ICHECK_GE(*maybe_target_task_index, 0);
        ICHECK_LT(*maybe_target_task_index, program->tasks.size())
            << "ValidateSpatialProgram found sync edge with invalid target_task_index";
        ICHECK_EQ(static_cast<std::string>(edge->source), task_name_by_index[*maybe_source_task_index])
            << "ValidateSpatialProgram found sync edge source inconsistent with source_task_index";
        ICHECK_EQ(static_cast<std::string>(edge->target), task_name_by_index[*maybe_target_task_index])
            << "ValidateSpatialProgram found sync edge target inconsistent with target_task_index";
      }

      for (int i = 0; i < program->phases.size(); ++i) {
        const ProgramPhase& phase = program->phases[i];
        const std::string phase_name = static_cast<std::string>(phase->name);
        auto maybe_phase_index = GetPayloadIndex(phase->payload, schema_key::kPhaseIndex);
        ICHECK(maybe_phase_index);
        ICHECK_EQ(*maybe_phase_index, i)
            << "ValidateSpatialProgram requires phase_index contract to follow phase order";
        const auto& task_indices = phase_task_indices_by_index[i];
        ICHECK_EQ(phase->task_names.size(), task_indices.size())
            << "ValidateSpatialProgram requires phase task_names to stay aligned with task_indices";
        for (int j = 0; j < task_indices.size(); ++j) {
          const int64_t task_index = task_indices[j];
          ICHECK_GE(task_index, 0);
          ICHECK_LT(task_index, program->tasks.size())
              << "ValidateSpatialProgram found phase with invalid task_indices entry";
          ICHECK_EQ(task_phase_index_by_index[task_index], i)
              << "ValidateSpatialProgram found phase task_indices entry assigned to a different phase";
          ICHECK_EQ(static_cast<std::string>(phase->task_names[j]), task_name_by_index[task_index])
              << "ValidateSpatialProgram found phase task_names inconsistent with task_indices";
        }
        const auto& channel_indices = phase_channel_indices_by_index[i];
        ICHECK_EQ(phase->channel_names.size(), channel_indices.size())
            << "ValidateSpatialProgram requires phase channel_names to stay aligned with channel_indices";
        for (int j = 0; j < channel_indices.size(); ++j) {
          const int64_t channel_index = channel_indices[j];
          ICHECK_GE(channel_index, 0);
          ICHECK_LT(channel_index, program->channels.size())
              << "ValidateSpatialProgram found phase with invalid channel_indices entry";
          ICHECK_EQ(static_cast<std::string>(phase->channel_names[j]), channel_name_by_index[channel_index])
              << "ValidateSpatialProgram found phase channel_names inconsistent with channel_indices";
          ICHECK_EQ(task_phase_index_by_index[channel_target_task_index_by_index[channel_index]], i)
              << "ValidateSpatialProgram requires phase channel contracts to target tasks in "
                 "the owning phase";
        }
        if (program->phases.size() > 1 && i > 0) {
          ICHECK(!channel_indices.empty())
              << "ValidateSpatialProgram requires downstream multi-phase programs to reference "
                 "at least one channel";
        }
      }

      auto maybe_semantic_program =
          func.value()->GetAttr<SemanticProgram>(attr::kTLSemanticProgram);
      bool semantic_requires_pipeline_contract = false;
      bool semantic_requires_fragment_contract = false;
      bool semantic_requires_work_dependent_payload = false;
      int semantic_stateful_state_count = 0;
      std::unordered_set<int> semantic_stateful_state_indices;
      if (maybe_semantic_program && !maybe_semantic_program.value()->domains.empty()) {
        const SemanticProgram& semantic_program = maybe_semantic_program.value();
        for (const SpatialLayout& layout : program->layouts) {
          auto maybe_domain_index = GetPayloadIndex(layout->payload, schema_key::kDomainIndex);
          ICHECK(maybe_domain_index)
              << "ValidateSpatialProgram requires spatial layouts to carry domain_index contract";
          ICHECK_GE(*maybe_domain_index, 0);
          ICHECK_LT(*maybe_domain_index, semantic_program->domains.size())
              << "ValidateSpatialProgram found layout with invalid semantic domain index";
          const Domain& domain = semantic_program->domains[*maybe_domain_index];
          ICHECK(SameAxes(layout->axes, domain->axes))
              << "ValidateSpatialProgram found layout axes inconsistent with semantic domain";
          const bool semantic_indexed = HasTrait(domain->traits, "derived_indices");
          const bool layout_indexed = static_cast<std::string>(layout->kind) == "indexed";
          ICHECK_EQ(layout_indexed, semantic_indexed)
              << "ValidateSpatialProgram found layout kind inconsistent with semantic domain "
                 "derived_indices trait";
        }
        for (const WorkPartition& partition : program->work_partitions) {
          auto maybe_domain_index = GetPayloadIndex(partition->payload, schema_key::kDomainIndex);
          ICHECK(maybe_domain_index) << "ValidateSpatialProgram requires work partitions to carry "
                                        "domain_index contract";
          ICHECK_GE(*maybe_domain_index, 0);
          ICHECK_LT(*maybe_domain_index, semantic_program->domains.size())
              << "ValidateSpatialProgram found work partition with invalid semantic domain index";
          const Domain& domain = semantic_program->domains[*maybe_domain_index];
          ICHECK(SameAxes(partition->axes, domain->axes))
              << "ValidateSpatialProgram found work partition axes inconsistent with semantic "
                 "domain";
        }
        const Domain& domain = semantic_program->domains[0];
        if (HasTrait(domain->traits, "work_dependent_bounds")) {
          semantic_requires_work_dependent_payload = true;
        }
        for (int i = 0; i < semantic_program->states.size(); ++i) {
          const State& state = semantic_program->states[i];
          auto role = semantic::ParseStateRole(static_cast<std::string>(state->role));
          if (role && (*role == semantic::StateRole::kCarry ||
                       *role == semantic::StateRole::kReductionAccumulator ||
                       *role == semantic::StateRole::kSelectionState ||
                       *role == semantic::StateRole::kIndexState)) {
            ++semantic_stateful_state_count;
            semantic_stateful_state_indices.insert(i);
          }
        }
        for (const SemanticSupplement& supplement : semantic_program->supplements) {
          const std::string supplement_kind = static_cast<std::string>(supplement->kind);
          if (supplement_kind ==
              semantic::ToString(semantic::SupplementKind::kFragmentLoweringStructure)) {
            auto maybe_fragment_ops =
                supplement->payload.Get(String(schema_key::kFragmentOpKinds));
            semantic_requires_fragment_contract =
                maybe_fragment_ops &&
                !Downcast<Array<Any>>(maybe_fragment_ops.value()).empty();
            continue;
          }
          if (supplement_kind !=
              semantic::ToString(semantic::SupplementKind::kPipelineStructure)) {
            if (supplement_kind ==
                semantic::ToString(semantic::SupplementKind::kWorkDecompositionStructure)) {
              auto maybe_loop_bounds =
                  supplement->payload.Get(String(schema_key::kWorkDependentLoopBounds));
              semantic_requires_work_dependent_payload =
                  maybe_loop_bounds && !Downcast<Array<Any>>(maybe_loop_bounds.value()).empty();
            }
            continue;
          }
          auto maybe_pipeline_stages =
              supplement->payload.Get(String(schema_key::kPipelineStages));
          semantic_requires_pipeline_contract =
              maybe_pipeline_stages && !Downcast<Array<Any>>(maybe_pipeline_stages.value()).empty();
          if (semantic_requires_pipeline_contract) {
            break;
          }
        }
      }

      if (semantic_requires_work_dependent_payload) {
        bool has_partition_payload = false;
        for (const WorkPartition& partition : program->work_partitions) {
          auto maybe_loop_bounds =
              partition->payload.Get(String(schema_key::kWorkDependentLoopBounds));
          if (!maybe_loop_bounds) {
            continue;
          }
          Array<Any> loop_bounds = Downcast<Array<Any>>(maybe_loop_bounds.value());
          ICHECK(!loop_bounds.empty())
              << "ValidateSpatialProgram requires work partition payload loop bounds to be "
                 "non-empty";
          has_partition_payload = true;
          break;
        }
        ICHECK(has_partition_payload)
            << "ValidateSpatialProgram requires work-dependent domains to materialize work "
               "partition payload";
      }

      std::unordered_set<std::string> resource_intent_kinds;
      bool has_fragment_contract = false;
      bool has_pipeline_contract = false;
      int state_residency_count = 0;
      int phase_boundary_materialization_count = 0;
      std::unordered_set<int> state_residency_state_indices;
      std::unordered_set<int> phase_boundary_state_indices;
      for (const ResourceIntent& intent : program->resource_intents) {
        const std::string intent_kind = static_cast<std::string>(intent->kind);
        resource_intent_kinds.insert(intent_kind);
        state_residency_count += intent_kind == "state_residency";
        phase_boundary_materialization_count += intent_kind == "phase_boundary_materialization";
        if (intent_kind == "state_residency" ||
            intent_kind == "phase_boundary_materialization") {
          auto maybe_target_kind = GetPayloadString(intent->payload, schema_key::kTargetKind);
          ICHECK(maybe_target_kind &&
                 *maybe_target_kind == spatial_contract::kSemanticStateTarget)
              << "ValidateSpatialProgram requires state materialization intents to carry "
                 "semantic_state target_kind contract";
          auto maybe_target_index = GetPayloadIndex(intent->payload, schema_key::kTargetIndex);
          ICHECK(maybe_target_index)
              << "ValidateSpatialProgram requires state materialization intents to carry "
                 "target_index contract";
          ICHECK(maybe_semantic_program)
              << "ValidateSpatialProgram requires SemanticProgram when validating state-targeted "
                 "resource intents";
          ICHECK_GE(*maybe_target_index, 0);
          ICHECK_LT(*maybe_target_index, maybe_semantic_program.value()->states.size())
              << "ValidateSpatialProgram found state-targeted intent with invalid target_index";
          if (intent_kind == "state_residency") {
            if (semantic_stateful_state_indices.count(*maybe_target_index)) {
              state_residency_state_indices.insert(*maybe_target_index);
            }
          } else {
            ICHECK(semantic_stateful_state_indices.count(*maybe_target_index))
                << "ValidateSpatialProgram requires phase-boundary intents to target "
                   "stateful semantic states";
            phase_boundary_state_indices.insert(*maybe_target_index);
          }
        }
        if (IsFragmentContractIntent(intent)) {
          has_fragment_contract = true;
          auto maybe_fragment_ops =
              intent->payload.Get(String(schema_key::kFragmentOpKinds));
          ICHECK(maybe_fragment_ops)
              << "ValidateSpatialProgram requires fragment contracts to carry fragment_op_kinds";
          Array<Any> fragment_ops = Downcast<Array<Any>>(maybe_fragment_ops.value());
          ICHECK(!fragment_ops.empty())
              << "ValidateSpatialProgram requires fragment contracts to carry at least one "
                 "fragment op";
          bool requires_pointwise_payload = false;
          bool requires_row_broadcast_payload = false;
          for (const Any& op_any : fragment_ops) {
            const std::string op_name = Downcast<String>(op_any);
            requires_pointwise_payload |= op_name == "pointwise_chain";
            requires_row_broadcast_payload |= op_name == "row_broadcast";
          }
          if (requires_pointwise_payload) {
            auto maybe_pointwise_ops =
                intent->payload.Get(String(schema_key::kPointwiseOpKinds));
            ICHECK(maybe_pointwise_ops)
                << "ValidateSpatialProgram requires fragment pointwise_chain contracts to "
                   "carry pointwise_op_kinds";
            ICHECK(!Downcast<Array<Any>>(maybe_pointwise_ops.value()).empty())
                << "ValidateSpatialProgram requires fragment pointwise_op_kinds to be non-empty";
          }
          if (requires_row_broadcast_payload) {
            auto maybe_row_broadcast_sources =
                intent->payload.Get(String(schema_key::kRowBroadcastSources));
            ICHECK(maybe_row_broadcast_sources)
                << "ValidateSpatialProgram requires fragment row_broadcast contracts to carry "
                   "row_broadcast_sources";
            ICHECK(!Downcast<Array<Any>>(maybe_row_broadcast_sources.value()).empty())
                << "ValidateSpatialProgram requires fragment row_broadcast_sources to be "
                   "non-empty";
          }
        }
        if (IsPipelineContractIntent(intent)) {
          has_pipeline_contract = true;
          auto maybe_pipeline_stages = intent->payload.Get(String(schema_key::kPipelineStages));
          ICHECK(maybe_pipeline_stages)
              << "ValidateSpatialProgram requires pipeline contracts to carry pipeline_stages";
          Array<Any> pipeline_stages = Downcast<Array<Any>>(maybe_pipeline_stages.value());
          ICHECK(!pipeline_stages.empty())
              << "ValidateSpatialProgram requires pipeline contracts to carry at least one "
                 "pipeline stage";
          for (const Any& stage_any : pipeline_stages) {
            auto stage = Downcast<Map<String, Any>>(stage_any);
            ICHECK(stage.count(String(schema_key::kLoopVar)))
                << "ValidateSpatialProgram requires pipeline stage entries to carry loop_var";
            ICHECK(stage.count(String(schema_key::kNumStages)))
                << "ValidateSpatialProgram requires pipeline stage entries to carry num_stages";
          }
        }
      }
      if (program->phases.size() > 1) {
        ICHECK(resource_intent_kinds.count("phase_boundary_materialization"))
            << "ValidateSpatialProgram requires multi-phase programs to materialize at least "
               "one phase-boundary resource intent";
        if (semantic_stateful_state_count > 0) {
          ICHECK_GE(phase_boundary_materialization_count, semantic_stateful_state_count)
              << "ValidateSpatialProgram requires multi-phase programs to materialize a "
                 "phase-boundary intent for each stateful semantic state";
          ICHECK_GE(phase_boundary_state_indices.size(), semantic_stateful_state_count)
              << "ValidateSpatialProgram requires multi-phase programs to materialize a "
                 "phase-boundary intent for each stateful semantic state";
        }
      }
      if (semantic_stateful_state_count > 0) {
        ICHECK_GE(state_residency_count, semantic_stateful_state_count)
            << "ValidateSpatialProgram requires stateful semantic states to materialize "
               "state-residency intents";
        ICHECK_GE(state_residency_state_indices.size(), semantic_stateful_state_count)
            << "ValidateSpatialProgram requires stateful semantic states to materialize "
               "state-residency intents";
      }
      if (semantic_requires_pipeline_contract) {
        ICHECK(has_pipeline_contract)
            << "ValidateSpatialProgram requires pipeline programs to materialize at least one "
               "pipeline contract";
      }
      if (semantic_requires_fragment_contract) {
        ICHECK(has_fragment_contract)
            << "ValidateSpatialProgram requires fragment programs to materialize at least one "
               "fragment contract";
      }

      phases_by_member_func[member_func] = program->phases;
    }

    if (auto maybe_registry = mod->global_infos.Get(attr::kTLDevicePrograms)) {
      for (const GlobalInfo& item : maybe_registry.value()) {
        auto info = Downcast<TLDeviceProgramInfo>(item);
        Array<ProgramPhase> expected_phases;
        for (const String& member_func : info->member_funcs) {
          auto it = phases_by_member_func.find(static_cast<std::string>(member_func));
          if (it == phases_by_member_func.end()) {
            continue;
          }
          for (const ProgramPhase& phase : it->second) {
            expected_phases.push_back(phase);
          }
        }
        if (expected_phases.empty() && info->member_funcs.size() == 1) {
          auto root_it = phases_by_member_func.find(static_cast<std::string>(info->root_symbol));
          if (root_it != phases_by_member_func.end()) {
            for (const ProgramPhase& phase : root_it->second) {
              expected_phases.push_back(phase);
            }
          }
        }
        ICHECK_EQ(info->phases.size(), expected_phases.size())
            << "ValidateSpatialProgram requires tl.device_programs to carry aggregated "
               "ProgramPhase truth";
        for (int i = 0; i < info->phases.size(); ++i) {
          ICHECK(SamePhaseSignature(info->phases[i], expected_phases[i]))
              << "ValidateSpatialProgram requires tl.device_programs aggregated ProgramPhase "
                 "truth to match member-local phase signatures";
        }
      }
    }

    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tl.transform.ValidateSpatialProgram", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.ValidateSpatialProgram", ValidateSpatialProgram);
}

}  // namespace tl
}  // namespace tvm
