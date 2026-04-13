/*!
 * \file build_spatial_plan_companion.cc
 * \brief Freeze Task 1 SpatialPlan companion from analyzed spatial structure facts.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/transform.h>

#include "common/blackhole_utils.h"
#include "common/spatial_analysis.h"
#include "common/spatial_plan.h"

namespace tvm {
namespace tl {

using tvm::DictAttrs;
using tvm::ffi::Any;
using tvm::ffi::Map;
using tvm::ffi::String;

tvm::transform::Pass BuildSpatialPlanCompanion() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext) {
    IRModule updates = IRModule(Map<GlobalVar, BaseFunc>({}));
    for (const auto& [gvar, base_func] : mod->functions) {
      auto func = base_func.as<tir::PrimFunc>();
      if (!func || !IsBlackholePrimFunc(func.value())) {
        continue;
      }
      auto maybe_facts =
          func.value()->GetAttr<SpatialStructureFacts>(attr::kTLSpatialStructureFacts);
      if (!maybe_facts) {
        continue;
      }

      const std::string member_func = GetMemberFuncName(gvar, func.value());
      SpatialPlan plan(String(member_func), maybe_facts.value()->closure_candidates,
                       maybe_facts.value()->boundary_candidates,
                       maybe_facts.value()->validated_hints,
                       MakeAnchors("spatial_plan", member_func));

      tir::PrimFunc updated_func = func.value();
      Map<String, Any> attrs = updated_func->attrs.defined() ? updated_func->attrs->dict
                                                             : Map<String, Any>();
      attrs.Set(attr::kTLSpatialPlan, plan);
      updated_func.CopyOnWrite()->attrs = DictAttrs(attrs);
      updates->Add(gvar, updated_func);
    }
    mod->Update(updates);
    return mod;
  };
  return tvm::transform::CreateModulePass(
      pass_func, 0, "tl.transform.BuildSpatialPlanCompanion", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.BuildSpatialPlanCompanion",
                        BuildSpatialPlanCompanion);
}

}  // namespace tl
}  // namespace tvm
