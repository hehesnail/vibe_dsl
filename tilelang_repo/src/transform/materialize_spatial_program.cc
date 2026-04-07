/*!
 * \file materialize_spatial_program.cc
 * \brief Materialize SpatialProgram from split Phase B domain/execution plans.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/transform.h>

#include <unordered_map>

#include "common/blackhole_utils.h"
#include "common/spatial_analysis.h"
#include "common/spatial_program.h"

namespace tvm {
namespace tl {

using tvm::DictAttrs;
using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::String;

tvm::transform::Pass MaterializeSpatialProgram() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext) {
    IRModule updates = IRModule(Map<GlobalVar, BaseFunc>({}));
    std::unordered_map<std::string, Array<ProgramPhase>> phases_by_member_func;

    for (const auto& [gvar, base_func] : mod->functions) {
      auto func = base_func.as<tir::PrimFunc>();
      if (!func || !IsBlackholePrimFunc(func.value())) {
        continue;
      }
      auto maybe_program = func.value()->GetAttr<SpatialProgram>(attr::kTLSpatialProgram);
      if (maybe_program) {
        phases_by_member_func[GetMemberFuncName(gvar, func.value())] = maybe_program.value()->phases;
        continue;
      }
      auto maybe_domain_plan = func.value()->GetAttr<SpatialDomainPlan>(attr::kTLSpatialDomainPlan);
      auto maybe_execution_plan =
          func.value()->GetAttr<SpatialExecutionPlan>(attr::kTLSpatialExecutionPlan);
      if (!maybe_domain_plan || !maybe_execution_plan) {
        continue;
      }
      const std::string member_func = GetMemberFuncName(gvar, func.value());
      SpatialProgram program(
          String(member_func), maybe_execution_plan.value()->phases, maybe_execution_plan.value()->tasks,
          maybe_execution_plan.value()->channels, maybe_domain_plan.value()->layouts,
          maybe_domain_plan.value()->work_partitions, maybe_execution_plan.value()->placements,
          maybe_execution_plan.value()->sync_edges, maybe_execution_plan.value()->resource_intents,
          MakeAnchors("spatial_program", member_func));
      phases_by_member_func[member_func] = program->phases;
      tir::PrimFunc updated_func = func.value();
      Map<String, Any> attrs = updated_func->attrs.defined() ? updated_func->attrs->dict
                                                             : Map<String, Any>();
      attrs.Set(attr::kTLSpatialProgram, program);
      updated_func.CopyOnWrite()->attrs = DictAttrs(attrs);
      updates->Add(gvar, updated_func);
    }

    mod->Update(updates);
    if (auto maybe_registry = mod->global_infos.Get(attr::kTLDevicePrograms)) {
      mod = mod->ShallowCopy();
      Array<GlobalInfo> rebuilt_registry;
      for (const GlobalInfo& info : maybe_registry.value()) {
        auto program_info = Downcast<TLDeviceProgramInfo>(info);
        Array<ProgramPhase> phases;
        for (const String& member_func : program_info->member_funcs) {
          auto it = phases_by_member_func.find(str(member_func));
          if (it == phases_by_member_func.end()) {
            continue;
          }
          for (const ProgramPhase& phase : it->second) {
            phases.push_back(phase);
          }
        }
        if (phases.empty() && program_info->member_funcs.size() == 1) {
          auto root_it = phases_by_member_func.find(str(program_info->root_symbol));
          if (root_it != phases_by_member_func.end()) {
            for (const ProgramPhase& phase : root_it->second) {
              phases.push_back(phase);
            }
          }
        }
        rebuilt_registry.push_back(
            TLDeviceProgramInfo(program_info->root_symbol, program_info->member_funcs, phases));
      }
      mod->UpdateGlobalInfo(attr::kTLDevicePrograms, rebuilt_registry);
    }
    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 0,
                                          "tl.transform.MaterializeSpatialProgram", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.MaterializeSpatialProgram",
                        MaterializeSpatialProgram);
}

}  // namespace tl
}  // namespace tvm
