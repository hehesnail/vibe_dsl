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

namespace tvm {
namespace tl {

using tvm::ffi::Any;
using tvm::ffi::Array;
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

      auto maybe_semantic_program = func.value()->GetAttr<SemanticProgram>(attr::kTLSemanticProgram);
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
      }

      std::unordered_set<std::string> resource_intent_kinds;
      for (const ResourceIntent& intent : program->resource_intents) {
        resource_intent_kinds.insert(static_cast<std::string>(intent->kind));
      }
      if (program->phases.size() > 1) {
        ICHECK(resource_intent_kinds.count("phase_boundary_materialization"))
            << "ValidateSpatialProgram requires multi-phase programs to materialize at least "
               "one phase-boundary resource intent";
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
