/*!
 * \file analyze_spatial_domain_plan.cc
 * \brief Derive typed Phase B domain contracts from TIR + SpatialPlan-side facts.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/transform.h>
#include <tvm/target/target.h>

#include <optional>
#include <string>

#include "common/blackhole_utils.h"
#include "common/spatial_analysis.h"
#include "common/spatial_plan.h"
#include "common/spatial_program.h"
#include "common/spatial_vocab.h"
#include "common/tt_hardware_model.h"

namespace tvm {
namespace tl {

using tvm::DictAttrs;
using tvm::GlobalInfo;
using tvm::Integer;
using tvm::ffi::Any;
using tvm::ffi::Map;
using tvm::ffi::String;
namespace sp = tvm::tl::spatial;

namespace {

struct DomainContract {
  std::string transform_kind = "identity";
  std::string partition_family = "regular";
  sp::SpatialLayoutKind layout_kind = sp::SpatialLayoutKind::kRegular;
  sp::SpatialPartitionKind partition_kind = sp::SpatialPartitionKind::kReplicated;
};

Array<String> GetAxesFromWorkDecomposition(const tir::PrimFunc& func) {
  Array<String> axes;
  auto work = func->GetAttr<Map<String, Any>>("blackhole.work_decomposition");
  if (!work) {
    return axes;
  }
  auto maybe_axes = work.value().Get(String("axes"));
  if (!maybe_axes) {
    return axes;
  }
  for (const Any& axis_any : Downcast<Array<Any>>(maybe_axes.value())) {
    axes.push_back(Downcast<String>(axis_any));
  }
  return axes;
}

bool HasNonEmptyArrayField(const Map<String, Any>& payload, const char* key) {
  auto maybe_value = payload.Get(String(key));
  return maybe_value && !Downcast<Array<Any>>(maybe_value.value()).empty();
}

bool WorkDecompositionHasDerivedIndices(const tir::PrimFunc& func) {
  auto work = func->GetAttr<Map<String, Any>>("blackhole.work_decomposition");
  return work && HasNonEmptyArrayField(work.value(), "derived_index_exprs");
}

bool WorkDecompositionHasWorkDependentBounds(const tir::PrimFunc& func) {
  auto work = func->GetAttr<Map<String, Any>>("blackhole.work_decomposition");
  return work && HasNonEmptyArrayField(work.value(), "work_dependent_loop_bounds");
}

bool FragmentRegionsHaveSelectionTargets(const tir::PrimFunc& func) {
  auto regions = func->GetAttr<Array<Any>>("blackhole.fragment_regions");
  if (!regions) {
    return false;
  }
  for (const Any& region_any : regions.value()) {
    if (HasNonEmptyArrayField(Downcast<Map<String, Any>>(region_any),
                              manifest_key::kSelectionTargets)) {
      return true;
    }
  }
  return false;
}

Map<String, Any> BuildDomainPayload(int domain_index, const DomainContract& contract) {
  Map<String, Any> payload;
  payload.Set(String(schema_key::kDomainIndex), Integer(domain_index));
  payload.Set(String(schema_key::kDomainTransformKind), String(contract.transform_kind));
  return payload;
}

Map<String, Any> BuildWorkPartitionPayload(const tir::PrimFunc& func, int domain_index,
                                           const DomainContract& contract) {
  Map<String, Any> payload = BuildDomainPayload(domain_index, contract);
  payload.Set(String(schema_key::kPartitionFamily), String(contract.partition_family));
  auto work = func->GetAttr<Map<String, Any>>("blackhole.work_decomposition");
  if (work) {
    auto maybe_loop_bounds = work.value().Get(String("work_dependent_loop_bounds"));
    if (maybe_loop_bounds) {
      payload.Set(String(schema_key::kWorkDependentLoopBounds),
                  Downcast<Array<Any>>(maybe_loop_bounds.value()));
    }
  }
  return payload;
}

void RequireCapabilitySupport(const Array<String>& supported_kinds, const char* kind,
                              const char* contract_kind) {
  ICHECK(ContainsKind(supported_kinds, kind))
      << "AnalyzeSpatialDomainPlan requires SpatialCapabilityModel support for "
      << contract_kind << " kind " << kind;
}

const char* SelectLayoutKind(const SpatialCapabilityModel& capability_model,
                             const DomainContract& contract) {
  const char* kind = sp::ToString(contract.layout_kind);
  RequireCapabilitySupport(capability_model->supported_layout_kinds, kind, "layout");
  return kind;
}

const char* SelectPartitionKind(const SpatialCapabilityModel& capability_model,
                                const DomainContract& contract) {
  const char* kind = sp::ToString(contract.partition_kind);
  RequireCapabilitySupport(capability_model->supported_partition_kinds, kind, "work partition");
  return kind;
}

DomainContract DeriveDomainContract(const tir::PrimFunc& func, const Array<String>& axes) {
  const bool has_derived = WorkDecompositionHasDerivedIndices(func);
  const bool has_work_dependent_bounds = WorkDecompositionHasWorkDependentBounds(func);
  const bool has_selection_targets = FragmentRegionsHaveSelectionTargets(func);
  const bool multi_axis = axes.size() > 1;

  DomainContract contract;
  if (has_derived) {
    contract.transform_kind = has_work_dependent_bounds ? "paged" : "derived";
    contract.partition_family = has_work_dependent_bounds ? "paged" : "derived";
    contract.layout_kind = sp::SpatialLayoutKind::kIndexed;
    contract.partition_kind = sp::SpatialPartitionKind::kIndexed;
    return contract;
  }
  if (has_selection_targets) {
    contract.transform_kind = "filtered";
    contract.partition_family = "filtered";
    contract.partition_kind = sp::SpatialPartitionKind::kFiltered;
    return contract;
  }
  contract.partition_kind =
      multi_axis ? sp::SpatialPartitionKind::kBlocked : sp::SpatialPartitionKind::kReplicated;
  return contract;
}

SpatialDomainPlan BuildSpatialDomainPlan(const std::string& member_func, const tir::PrimFunc& func,
                                         const SpatialCapabilityModel& capability_model) {
  const Array<String> axes = GetAxesFromWorkDecomposition(func);
  const DomainContract contract = DeriveDomainContract(func, axes);
  Array<SpatialLayout> layouts{
      SpatialLayout(String("layout_" + member_func), String(SelectLayoutKind(capability_model, contract)),
                    String(member_func), axes, MakeTraits({"phase_b"}),
                    BuildDomainPayload(0, contract), MakeAnchors("spatial_layout", member_func))};
  Array<WorkPartition> work_partitions{
      WorkPartition(String("partition_" + member_func),
                    String(SelectPartitionKind(capability_model, contract)), String(member_func),
                    axes, MakeTraits({"phase_b"}),
                    BuildWorkPartitionPayload(func, 0, contract),
                    MakeAnchors("spatial_partition", member_func))};
  return SpatialDomainPlan(String(member_func), layouts, work_partitions,
                           MakeAnchors("spatial_domain_plan", member_func));
}

}  // namespace

tvm::transform::Pass AnalyzeSpatialDomainPlan() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext) {
    IRModule updates = IRModule(Map<GlobalVar, BaseFunc>({}));
    std::optional<TTHardwareModel> hardware_model;
    std::optional<SpatialCapabilityModel> capability_model;
    for (const auto& [gvar, base_func] : mod->functions) {
      auto func = base_func.as<tir::PrimFunc>();
      if (!func || !IsBlackholePrimFunc(func.value())) {
        continue;
      }
      if (!func.value()->GetAttr<SpatialPlan>(attr::kTLSpatialPlan)) {
        continue;
      }
      auto maybe_target = func.value()->GetAttr<Target>(tvm::attr::kTarget);
      ICHECK(maybe_target)
          << "AnalyzeSpatialDomainPlan requires blackhole PrimFunc target to derive capability";
      if (!hardware_model || !capability_model) {
        hardware_model = BuildBlackholeTTHardwareModel(maybe_target.value());
        capability_model = DeriveSpatialCapabilityModel(hardware_model.value());
      }
      const std::string member_func = GetMemberFuncName(gvar, func.value());
      const SpatialDomainPlan domain_plan =
          BuildSpatialDomainPlan(member_func, func.value(), capability_model.value());
      tir::PrimFunc updated_func = func.value();
      Map<String, Any> attrs =
          updated_func->attrs.defined() ? updated_func->attrs->dict : Map<String, Any>();
      attrs.Set(attr::kTLSpatialDomainPlan, domain_plan);
      updated_func.CopyOnWrite()->attrs = DictAttrs(attrs);
      updates->Add(gvar, updated_func);
    }
    mod->Update(updates);
    if (hardware_model || capability_model) {
      mod = mod->ShallowCopy();
    }
    if (hardware_model) {
      mod->UpdateGlobalInfo(attr::kTLTTHardwareModel, Array<GlobalInfo>{hardware_model.value()});
    }
    if (capability_model) {
      mod->UpdateGlobalInfo(attr::kTLSpatialCapabilityModel,
                            Array<GlobalInfo>{capability_model.value()});
    }
    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 0,
                                          "tl.transform.AnalyzeSpatialDomainPlan", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AnalyzeSpatialDomainPlan", AnalyzeSpatialDomainPlan);
}

}  // namespace tl
}  // namespace tvm
