/*!
 * \file validate_spatial_program.cc
 * \brief Validate minimal Phase B SpatialProgram invariants.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "common/blackhole_utils.h"
#include "common/semantic_program.h"
#include "common/semantic_vocab.h"

namespace tvm {
namespace tl {

using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::String;

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

bool SamePhaseSignature(const ProgramPhase& lhs, const ProgramPhase& rhs) {
  return static_cast<std::string>(lhs->name) == static_cast<std::string>(rhs->name) &&
         SameStringArray(lhs->task_names, rhs->task_names) &&
         SameStringArray(lhs->channel_names, rhs->channel_names);
}

bool IsPipelineContractIntent(const ResourceIntent& intent) {
  return static_cast<std::string>(intent->kind) == "synchronization_support" &&
         HasTrait(intent->traits, "pipeline_contract");
}

bool IsFragmentContractIntent(const ResourceIntent& intent) {
  return static_cast<std::string>(intent->kind) == "lowering_support" &&
         HasTrait(intent->traits, "fragment_contract");
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
      for (const ProgramPhase& phase : program->phases) {
        const std::string phase_name = static_cast<std::string>(phase->name);
        ICHECK(phase_names.insert(phase_name).second)
            << "ValidateSpatialProgram found duplicate phase " << phase_name;
      }

      std::unordered_set<std::string> task_names;
      std::unordered_map<std::string, std::string> task_to_phase;
      for (const Task& task : program->tasks) {
        const std::string task_name = static_cast<std::string>(task->name);
        ICHECK(task_names.insert(task_name).second)
            << "ValidateSpatialProgram found duplicate task " << task_name;
        const std::string phase_name = static_cast<std::string>(task->phase_name);
        ICHECK(phase_names.count(phase_name))
            << "ValidateSpatialProgram found task assigned to unknown phase "
            << task->phase_name;
        task_to_phase[task_name] = phase_name;
      }

      std::unordered_set<std::string> channel_names;
      std::unordered_map<std::string, Channel> channels_by_name;
      for (const Channel& channel : program->channels) {
        const std::string channel_name = static_cast<std::string>(channel->name);
        ICHECK(channel_names.insert(channel_name).second)
            << "ValidateSpatialProgram found duplicate channel " << channel_name;
        if (!static_cast<std::string>(channel->source_task).empty()) {
          ICHECK(task_names.count(static_cast<std::string>(channel->source_task)))
              << "ValidateSpatialProgram found channel with unknown source task "
              << channel->source_task;
        }
        if (!static_cast<std::string>(channel->target_task).empty()) {
          ICHECK(task_names.count(static_cast<std::string>(channel->target_task)))
              << "ValidateSpatialProgram found channel with unknown target task "
              << channel->target_task;
        }
        channels_by_name[channel_name] = channel;
      }

      for (const Placement& placement : program->placements) {
        const std::string task_name = static_cast<std::string>(placement->task_name);
        ICHECK(task_names.count(task_name))
            << "ValidateSpatialProgram found placement referencing unknown task " << task_name;
        ICHECK_EQ(static_cast<std::string>(placement->member_func), member_func)
            << "ValidateSpatialProgram requires placement.member_func to match "
               "SpatialProgram.member_func";
      }

      for (const SyncEdge& edge : program->sync_edges) {
        const std::string source_task = static_cast<std::string>(edge->source);
        const std::string target_task = static_cast<std::string>(edge->target);
        ICHECK(task_names.count(source_task))
            << "ValidateSpatialProgram found sync edge with unknown source task " << source_task;
        ICHECK(task_names.count(target_task))
            << "ValidateSpatialProgram found sync edge with unknown target task " << target_task;
      }

      for (int i = 0; i < program->phases.size(); ++i) {
        const ProgramPhase& phase = program->phases[i];
        const std::string phase_name = static_cast<std::string>(phase->name);
        for (const String& task_name : phase->task_names) {
          const std::string task_name_str = static_cast<std::string>(task_name);
          ICHECK(task_names.count(task_name_str))
              << "ValidateSpatialProgram found phase referencing unknown task " << task_name;
          ICHECK_EQ(task_to_phase.at(task_name_str), phase_name)
              << "ValidateSpatialProgram found phase referencing task assigned to a different "
                 "phase";
        }
        for (const String& channel_name : phase->channel_names) {
          const std::string channel_name_str = static_cast<std::string>(channel_name);
          ICHECK(channel_names.count(channel_name_str))
              << "ValidateSpatialProgram found phase referencing unknown channel "
              << channel_name;
          const Channel& channel = channels_by_name.at(channel_name_str);
          if (!static_cast<std::string>(channel->target_task).empty()) {
            ICHECK_EQ(task_to_phase.at(static_cast<std::string>(channel->target_task)), phase_name)
                << "ValidateSpatialProgram requires phase channel contracts to target tasks in "
                   "the owning phase";
          }
        }
        if (program->phases.size() > 1 && i > 0) {
          ICHECK(!phase->channel_names.empty())
              << "ValidateSpatialProgram requires downstream multi-phase programs to reference "
                 "at least one channel";
        }
      }

      auto maybe_semantic_program =
          func.value()->GetAttr<SemanticProgram>(attr::kTLSemanticProgram);
      bool semantic_requires_pipeline_contract = false;
      bool semantic_requires_fragment_contract = false;
      bool semantic_requires_work_dependent_payload = false;
      if (maybe_semantic_program && !maybe_semantic_program.value()->domains.empty()) {
        const Domain& domain = maybe_semantic_program.value()->domains[0];
        for (const SpatialLayout& layout : program->layouts) {
          ICHECK(SameAxes(layout->axes, domain->axes))
              << "ValidateSpatialProgram found layout axes inconsistent with semantic domain";
          const bool semantic_indexed = HasTrait(domain->traits, "derived_indices");
          const bool layout_indexed = static_cast<std::string>(layout->kind) == "indexed";
          ICHECK_EQ(layout_indexed, semantic_indexed)
              << "ValidateSpatialProgram found layout kind inconsistent with semantic domain "
                 "derived_indices trait";
        }
        for (const WorkPartition& partition : program->work_partitions) {
          ICHECK(SameAxes(partition->axes, domain->axes))
              << "ValidateSpatialProgram found work partition axes inconsistent with semantic "
                 "domain";
        }
        if (HasTrait(domain->traits, "work_dependent_bounds")) {
          semantic_requires_work_dependent_payload = true;
        }
        for (const SemanticSupplement& supplement : maybe_semantic_program.value()->supplements) {
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
      for (const ResourceIntent& intent : program->resource_intents) {
        resource_intent_kinds.insert(static_cast<std::string>(intent->kind));
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
