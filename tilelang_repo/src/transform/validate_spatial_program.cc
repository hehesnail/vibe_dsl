/*!
 * \file validate_spatial_program.cc
 * \brief Validate Phase B SpatialProgram self-consistency.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/blackhole_utils.h"
#include "common/spatial_analysis.h"
#include "common/spatial_program.h"
#include "common/spatial_vocab.h"

namespace tvm {
namespace tl {

using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::String;
namespace sp = tvm::tl::spatial;

namespace {

sp::SpatialTaskKind RequireTaskKind(const Task& task) {
  auto parsed = sp::ParseSpatialTaskKind(task->kind);
  ICHECK(parsed) << "ValidateSpatialProgram found unknown task kind " << task->kind;
  return *parsed;
}

sp::SpatialChannelKind RequireChannelKind(const Channel& channel) {
  auto parsed = sp::ParseSpatialChannelKind(channel->kind);
  ICHECK(parsed) << "ValidateSpatialProgram found unknown channel kind " << channel->kind;
  return *parsed;
}

sp::SpatialChannelPayloadKind RequirePayloadKind(const Channel& channel) {
  auto parsed = sp::ParseSpatialChannelPayloadKind(channel->payload_kind);
  ICHECK(parsed) << "ValidateSpatialProgram found unknown channel payload kind "
                 << channel->payload_kind;
  return *parsed;
}

sp::SpatialChannelDeliveryKind RequireDeliveryKind(const Channel& channel) {
  auto parsed = sp::ParseSpatialChannelDeliveryKind(channel->delivery_kind);
  ICHECK(parsed) << "ValidateSpatialProgram found unknown channel delivery kind "
                 << channel->delivery_kind;
  return *parsed;
}

void ValidatePhases(const SpatialProgram& program) {
  std::unordered_set<std::string> phase_names;
  for (const ProgramPhase& phase : program->phases) {
    ICHECK(phase_names.insert(phase->name).second)
        << "ValidateSpatialProgram found duplicate phase " << phase->name;
    ICHECK_GE(phase->phase_index, 0)
        << "ValidateSpatialProgram requires phases to carry phase_index";
    auto maybe_task_indices = GetPayloadIndices(phase->payload, schema_key::kTaskIndices);
    auto maybe_channel_indices = GetPayloadIndices(phase->payload, schema_key::kChannelIndices);
    ICHECK(maybe_task_indices)
        << "ValidateSpatialProgram requires phases to carry task_indices";
    ICHECK(maybe_channel_indices)
        << "ValidateSpatialProgram requires phases to carry channel_indices";
    ICHECK(!phase->closure_basis.empty())
        << "ValidateSpatialProgram requires phases to carry closure_basis";
    for (const Integer& task_index : phase->task_indices) {
      ICHECK_LT(task_index->value, program->tasks.size())
          << "ValidateSpatialProgram found phase task_index out of range";
    }
    for (const Integer& channel_index : phase->channel_indices) {
      ICHECK_LT(channel_index->value, program->channels.size())
          << "ValidateSpatialProgram found phase channel_index out of range";
    }
  }
}

void ValidateTasks(const SpatialProgram& program) {
  std::unordered_set<std::string> task_names;
  for (const Task& task : program->tasks) {
    RequireTaskKind(task);
    ICHECK(task_names.insert(task->name).second)
        << "ValidateSpatialProgram found duplicate task " << task->name;
    ICHECK_GE(task->phase_index, 0)
        << "ValidateSpatialProgram requires tasks to carry phase_index";
    ICHECK_LT(task->phase_index, program->phases.size())
        << "ValidateSpatialProgram found task with invalid phase_index";
    ICHECK(!task->execution_role.empty())
        << "ValidateSpatialProgram requires tasks to carry execution_role";
    ICHECK(!task->formation_basis.empty())
        << "ValidateSpatialProgram requires tasks to carry formation_basis";
  }
}

void ValidateChannels(const SpatialProgram& program, std::unordered_set<std::string>* phase_boundary_subjects) {
  std::unordered_set<std::string> channel_names;
  for (const Channel& channel : program->channels) {
    RequireChannelKind(channel);
    RequirePayloadKind(channel);
    const sp::SpatialChannelDeliveryKind delivery_kind = RequireDeliveryKind(channel);
    ICHECK(channel_names.insert(channel->name).second)
        << "ValidateSpatialProgram found duplicate channel " << channel->name;
    ICHECK_GE(channel->source_task_index, 0)
        << "ValidateSpatialProgram requires channels to carry source_task_index";
    ICHECK_GE(channel->target_task_index, 0)
        << "ValidateSpatialProgram requires channels to carry target_task_index";
    ICHECK_LT(channel->source_task_index, program->tasks.size())
        << "ValidateSpatialProgram found channel with invalid source_task_index";
    ICHECK_LT(channel->target_task_index, program->tasks.size())
        << "ValidateSpatialProgram found channel with invalid target_task_index";
    const bool cross_phase =
        program->tasks[channel->source_task_index]->phase_index !=
        program->tasks[channel->target_task_index]->phase_index;
    if (cross_phase) {
      ICHECK(delivery_kind == sp::SpatialChannelDeliveryKind::kPhaseBoundaryMaterialized)
          << "ValidateSpatialProgram requires cross-phase channels to use "
             "phase_boundary_materialized delivery, but channel "
          << channel->name << " uses " << channel->delivery_kind;
      if (!channel->state_name.empty()) {
        phase_boundary_subjects->insert(channel->state_name);
      }
    }
  }
}

void ValidateLayoutsAndPartitions(const SpatialProgram& program) {
  for (const SpatialLayout& layout : program->layouts) {
    auto parsed = sp::ParseSpatialLayoutKind(layout->kind);
    ICHECK(parsed) << "ValidateSpatialProgram found unknown layout kind " << layout->kind;
    ICHECK_GE(layout->domain_index, 0)
        << "ValidateSpatialProgram requires layouts to carry domain_index";
    ICHECK(!layout->domain_transform_kind.empty())
        << "ValidateSpatialProgram requires layouts to carry domain_transform_kind";
  }
  for (const WorkPartition& partition : program->work_partitions) {
    auto parsed = sp::ParseSpatialPartitionKind(partition->kind);
    ICHECK(parsed) << "ValidateSpatialProgram found unknown work partition kind "
                   << partition->kind;
    ICHECK_GE(partition->domain_index, 0)
        << "ValidateSpatialProgram requires work partitions to carry domain_index";
    ICHECK(!partition->partition_family.empty())
        << "ValidateSpatialProgram requires work partitions to carry partition_family";
  }
}

void ValidatePlacementsAndSync(const SpatialProgram& program) {
  for (const Placement& placement : program->placements) {
    auto parsed = sp::ParseSpatialPlacementKind(placement->kind);
    ICHECK(parsed) << "ValidateSpatialProgram found unknown placement kind " << placement->kind;
    ICHECK_GE(placement->task_index, 0)
        << "ValidateSpatialProgram requires placements to carry task_index";
    ICHECK_LT(placement->task_index, program->tasks.size())
        << "ValidateSpatialProgram found placement with invalid task_index";
    ICHECK(!placement->affinity_kind.empty())
        << "ValidateSpatialProgram requires placements to carry affinity_kind";
    ICHECK(!placement->obligation_kind.empty())
        << "ValidateSpatialProgram requires placements to carry obligation_kind";
  }
  for (const SyncEdge& edge : program->sync_edges) {
    auto parsed = sp::ParseSpatialSyncKind(edge->kind);
    ICHECK(parsed) << "ValidateSpatialProgram found unknown sync kind " << edge->kind;
    ICHECK_GE(edge->source_task_index, 0)
        << "ValidateSpatialProgram requires sync edges to carry source_task_index";
    ICHECK_GE(edge->target_task_index, 0)
        << "ValidateSpatialProgram requires sync edges to carry target_task_index";
    ICHECK_LT(edge->source_task_index, program->tasks.size())
        << "ValidateSpatialProgram found sync edge with invalid source_task_index";
    ICHECK_LT(edge->target_task_index, program->tasks.size())
        << "ValidateSpatialProgram found sync edge with invalid target_task_index";
    ICHECK(!edge->ordering_kind.empty())
        << "ValidateSpatialProgram requires sync edges to carry ordering_kind";
    ICHECK(!edge->materialization_kind.empty())
        << "ValidateSpatialProgram requires sync edges to carry materialization_kind";
  }
}

void ValidateResourceIntents(const SpatialProgram& program,
                             const std::unordered_set<std::string>& phase_boundary_subjects) {
  std::unordered_set<std::string> seen_intents;
  std::unordered_set<std::string> seen_phase_boundary_targets;
  for (const ResourceIntent& intent : program->resource_intents) {
    auto parsed = sp::ParseSpatialResourceIntentKind(intent->kind);
    ICHECK(parsed) << "ValidateSpatialProgram found unknown resource intent kind "
                   << intent->kind;
    ICHECK(seen_intents.insert(intent->name).second)
        << "ValidateSpatialProgram found duplicate resource intent " << intent->name;
    if (*parsed == sp::SpatialResourceIntentKind::kPhaseBoundaryMaterialization) {
      ICHECK_EQ(intent->target_kind, spatial_contract::kBufferTarget)
          << "ValidateSpatialProgram requires phase_boundary_materialization intents to target buffers";
      ICHECK(!intent->target_name.empty())
          << "ValidateSpatialProgram requires phase_boundary_materialization intents to carry target_name";
      seen_phase_boundary_targets.insert(intent->target_name);
    }
    if (*parsed == sp::SpatialResourceIntentKind::kSynchronizationSupport &&
        HasTrait(intent->traits, "pipeline_contract")) {
      auto maybe_stages = intent->payload.Get(String(schema_key::kPipelineStages));
      ICHECK(maybe_stages)
          << "ValidateSpatialProgram requires pipeline_contract intents to carry pipeline_stages";
      ICHECK(!Downcast<Array<Any>>(maybe_stages.value()).empty())
          << "ValidateSpatialProgram requires pipeline_contract intents to be non-empty";
    }
    if (*parsed == sp::SpatialResourceIntentKind::kLoweringSupport &&
        HasTrait(intent->traits, "fragment_contract")) {
      const bool has_fragment_ops =
          intent->payload.Get(String(schema_key::kFragmentOpKinds)).has_value() ||
          intent->payload.Get(String(schema_key::kFragmentMaterializationContracts)).has_value() ||
          intent->payload.Get(String(schema_key::kFragmentBufferFlowContracts)).has_value();
      ICHECK(has_fragment_ops)
          << "ValidateSpatialProgram requires fragment_contract intents to carry lowering facts";
    }
  }
  for (const std::string& subject : phase_boundary_subjects) {
    ICHECK(seen_phase_boundary_targets.count(subject))
        << "ValidateSpatialProgram requires phase-boundary intents for cross-phase channel subject "
        << subject;
  }
}

}  // namespace

tvm::transform::Pass ValidateSpatialProgram() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext) {
    for (const auto& [_, base_func] : mod->functions) {
      auto func = base_func.as<tir::PrimFunc>();
      if (!func || !IsBlackholePrimFunc(func.value())) {
        continue;
      }
      auto maybe_program = func.value()->GetAttr<SpatialProgram>(attr::kTLSpatialProgram);
      if (!maybe_program) {
        continue;
      }
      const SpatialProgram& program = maybe_program.value();
      ValidatePhases(program);
      ValidateTasks(program);
      std::unordered_set<std::string> phase_boundary_subjects;
      ValidateChannels(program, &phase_boundary_subjects);
      ValidateLayoutsAndPartitions(program);
      ValidatePlacementsAndSync(program);
      ValidateResourceIntents(program, phase_boundary_subjects);
    }
    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 0,
                                          "tl.transform.ValidateSpatialProgram", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.ValidateSpatialProgram", ValidateSpatialProgram);
}

}  // namespace tl
}  // namespace tvm
