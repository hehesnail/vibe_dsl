/*!
 * \file analyze_spatial_domain_plan.cc
 * \brief Derive typed Phase B domain contracts from semantic structure facts.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/transform.h>
#include <tvm/target/target.h>

#include <optional>
#include <string>
#include <vector>

#include "common/blackhole_utils.h"
#include "common/semantic_structure_decoder.h"
#include "common/spatial_analysis.h"
#include "common/spatial_program.h"
#include "common/spatial_vocab.h"
#include "common/tt_hardware_model.h"

namespace tvm {
namespace tl {

using tvm::DictAttrs;
using tvm::GlobalInfo;
using tvm::ffi::Any;
using tvm::ffi::Map;
using tvm::ffi::String;
using namespace tvm::tl::semantic;
namespace sp = tvm::tl::spatial;
using tvm::tl::str;

namespace {

Map<String, Any> BuildDomainPayload(int domain_index,
                                    const DomainRealizationContract& contract) {
  Map<String, Any> payload;
  payload.Set(String(schema_key::kDomainIndex), Integer(domain_index));
  payload.Set(String(schema_key::kDomainTransformKind), String(contract.domain_transform_kind));
  return payload;
}

Map<String, Any> BuildWorkPartitionPayload(const SemanticProgram& program, int domain_index,
                                           const DomainRealizationContract& contract) {
  Map<String, Any> payload = BuildDomainPayload(domain_index, contract);
  payload.Set(String(schema_key::kPartitionFamily), String(contract.partition_family));
  if (auto loop_bounds = GetWorkDependentLoopBoundsFromSupplements(program)) {
    payload.Set(String(schema_key::kWorkDependentLoopBounds), loop_bounds.value());
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
                             const DomainRealizationContract& contract) {
  const char* kind = sp::ToString(contract.layout_kind);
  RequireCapabilitySupport(capability_model->supported_layout_kinds, kind, "layout");
  return kind;
}

const char* SelectPartitionKind(const SpatialCapabilityModel& capability_model,
                                const DomainRealizationContract& contract) {
  const char* kind = sp::ToString(contract.partition_kind);
  RequireCapabilitySupport(capability_model->supported_partition_kinds, kind, "work partition");
  return kind;
}

SpatialDomainPlan BuildSpatialDomainPlan(const std::string& member_func,
                                         const SemanticProgram& semantic_program,
                                         const SpatialCapabilityModel& capability_model) {
  const std::vector<DomainRealizationContract> contracts =
      DeriveDomainRealizationContracts(semantic_program);
  Array<SpatialLayout> layouts;
  Array<WorkPartition> work_partitions;
  const bool multi_domain = semantic_program->domains.size() > 1;
  for (int domain_index = 0; domain_index < semantic_program->domains.size(); ++domain_index) {
    const Domain& domain = semantic_program->domains[domain_index];
    const DomainRealizationContract& contract = contracts[domain_index];
    const std::string domain_suffix = multi_domain ? "_" + str(domain->name) : std::string();
    layouts.push_back(SpatialLayout(
        String("layout_" + member_func + domain_suffix),
        String(SelectLayoutKind(capability_model, contract)), String(member_func), domain->axes,
        MakeTraits({"phase_b"}), BuildDomainPayload(domain_index, contract),
        MakeAnchors("spatial_layout", member_func + domain_suffix)));
    work_partitions.push_back(WorkPartition(
        String("partition_" + member_func + domain_suffix),
        String(SelectPartitionKind(capability_model, contract)), String(member_func), domain->axes,
        MakeTraits({"phase_b"}), BuildWorkPartitionPayload(semantic_program, domain_index, contract),
        MakeAnchors("spatial_partition", member_func + domain_suffix)));
  }
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
      auto maybe_semantic = MaybeDecodeSemanticProgramFromFunc(func.value());
      if (!maybe_semantic.has_value()) {
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
          BuildSpatialDomainPlan(member_func, maybe_semantic.value(), capability_model.value());
      tir::PrimFunc updated_func = func.value();
      Map<String, Any> attrs = updated_func->attrs.defined() ? updated_func->attrs->dict
                                                             : Map<String, Any>();
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
