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

      std::unordered_set<std::string> phase_names;
      for (const ProgramPhase& phase : program->phases) {
        const std::string phase_name = static_cast<std::string>(phase->name);
        ICHECK(phase_names.insert(phase_name).second)
            << "ValidateSpatialProgram found duplicate phase " << phase_name;
      }

      std::unordered_set<std::string> task_names;
      for (const Task& task : program->tasks) {
        const std::string task_name = static_cast<std::string>(task->name);
        ICHECK(task_names.insert(task_name).second)
            << "ValidateSpatialProgram found duplicate task " << task_name;
        ICHECK(phase_names.count(static_cast<std::string>(task->phase_name)))
            << "ValidateSpatialProgram found task assigned to unknown phase "
            << task->phase_name;
      }

      std::unordered_set<std::string> channel_names;
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
      }

      for (const ProgramPhase& phase : program->phases) {
        for (const String& task_name : phase->task_names) {
          ICHECK(task_names.count(static_cast<std::string>(task_name)))
              << "ValidateSpatialProgram found phase referencing unknown task " << task_name;
        }
        for (const String& channel_name : phase->channel_names) {
          ICHECK(channel_names.count(static_cast<std::string>(channel_name)))
              << "ValidateSpatialProgram found phase referencing unknown channel "
              << channel_name;
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
