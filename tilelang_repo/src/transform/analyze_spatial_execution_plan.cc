/*!
 * \file analyze_spatial_execution_plan.cc
 * \brief Analyze execution-bearing SpatialExecutionPlan from Phase B typed inputs.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/transform.h>
#include <tvm/target/target.h>

#include <optional>

#include "common/semantic_structure_decoder.h"
#include "common/spatial_analysis.h"
#include "common/spatial_program.h"
#include "common/blackhole_utils.h"
#include "common/tt_hardware_model.h"
#include "spatial_program_builder.h"

namespace tvm {
namespace tl {

using tvm::DictAttrs;
using tvm::ffi::Any;
using tvm::ffi::Map;
using tvm::ffi::String;
using namespace tvm::tl::semantic;

tvm::transform::Pass AnalyzeSpatialExecutionPlan() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext) {
    IRModule updates = IRModule(Map<GlobalVar, BaseFunc>({}));
    std::optional<SpatialCapabilityModel> capability_model = GetModuleSpatialCapabilityModel(mod);

    for (const auto& [gvar, base_func] : mod->functions) {
      auto func = base_func.as<tir::PrimFunc>();
      if (!func || !IsBlackholePrimFunc(func.value())) {
        continue;
      }
      auto maybe_semantic = MaybeDecodeSemanticProgramFromFunc(func.value());
      if (!maybe_semantic.has_value()) {
        continue;
      }
      ICHECK(func.value()->GetAttr<SpatialDomainPlan>(attr::kTLSpatialDomainPlan))
          << "AnalyzeSpatialExecutionPlan requires AnalyzeSpatialDomainPlan to run first";
      if (!capability_model) {
        auto maybe_target = func.value()->GetAttr<Target>(tvm::attr::kTarget);
        ICHECK(maybe_target)
            << "AnalyzeSpatialExecutionPlan requires blackhole PrimFunc target to derive capability";
        capability_model = DeriveSpatialCapabilityModel(BuildBlackholeTTHardwareModel(maybe_target.value()));
      }
      const std::string member_func = GetMemberFuncName(gvar, func.value());
      const SpatialExecutionPlan execution_plan =
          BuildSpatialExecutionPlanForFunc(member_func, maybe_semantic.value(), func.value(),
                                           capability_model.value());
      tir::PrimFunc updated_func = func.value();
      Map<String, Any> attrs = updated_func->attrs.defined() ? updated_func->attrs->dict
                                                             : Map<String, Any>();
      attrs.Set(attr::kTLSpatialExecutionPlan, execution_plan);
      updated_func.CopyOnWrite()->attrs = DictAttrs(attrs);
      updates->Add(gvar, updated_func);
    }

    mod->Update(updates);
    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 0,
                                          "tl.transform.AnalyzeSpatialExecutionPlan", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AnalyzeSpatialExecutionPlan",
                        AnalyzeSpatialExecutionPlan);
}

}  // namespace tl
}  // namespace tvm
