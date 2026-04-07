/*!
 * \file spatial_analysis.cc
 * \brief Shared semantic-to-spatial analysis helpers and contracts.
 */

#include "spatial_analysis.h"

#include "blackhole_utils.h"
#include "companion_base.h"

namespace tvm {
namespace tl {

using tvm::tl::str;

namespace {

std::unordered_set<std::string> BuildDomainAxisNameSet(const Domain& domain) {
  std::unordered_set<std::string> axis_names;
  for (const String& axis : domain->axes) {
    axis_names.insert(str(axis));
  }
  return axis_names;
}

void PushBackUniqueProducerEdge(std::vector<ProducerVersionEdge>* edges,
                                const ProducerVersionEdge& candidate) {
  ICHECK(edges != nullptr);
  const bool duplicate =
      std::find_if(edges->begin(), edges->end(), [&](const ProducerVersionEdge& existing) {
        return existing.producer_update == candidate.producer_update &&
               existing.produced_version == candidate.produced_version;
      }) != edges->end();
  if (!duplicate) {
    edges->push_back(candidate);
  }
}

std::vector<ProducerVersionEdge> ResolveProducerEdgesForVersion(
    const std::string& version_name,
    const std::unordered_map<std::string, std::vector<ProducerVersionEdge>>& direct_edges_by_version,
    const std::unordered_map<std::string, StateJoin>& join_by_output_version,
    std::unordered_map<std::string, std::vector<ProducerVersionEdge>>* memoized_edges,
    std::unordered_set<std::string>* active_versions) {
  ICHECK(memoized_edges != nullptr);
  ICHECK(active_versions != nullptr);
  if (auto it = memoized_edges->find(version_name); it != memoized_edges->end()) {
    return it->second;
  }
  ICHECK(!active_versions->count(version_name))
      << "spatial analysis found cyclic semantic join producer resolution for version "
      << version_name;
  active_versions->insert(version_name);

  std::vector<ProducerVersionEdge> edges;
  if (auto it = direct_edges_by_version.find(version_name); it != direct_edges_by_version.end()) {
    edges = it->second;
  } else if (auto join_it = join_by_output_version.find(version_name);
             join_it != join_by_output_version.end()) {
    for (const String& input_version : join_it->second->input_versions) {
      for (const ProducerVersionEdge& edge :
           ResolveProducerEdgesForVersion(str(input_version), direct_edges_by_version,
                                          join_by_output_version, memoized_edges,
                                          active_versions)) {
        PushBackUniqueProducerEdge(&edges, edge);
      }
    }
  }

  active_versions->erase(version_name);
  (*memoized_edges)[version_name] = edges;
  return edges;
}

void SetDomainRealizationKinds(DomainRealizationContract* contract,
                               sp::SpatialLayoutKind layout_kind,
                               sp::SpatialPartitionKind partition_kind) {
  ICHECK(contract != nullptr);
  contract->layout_kind = layout_kind;
  contract->partition_kind = partition_kind;
}

}  // namespace

std::optional<int64_t> GetPayloadIndex(const Map<String, Any>& payload, const char* key) {
  if (auto value = payload.Get(String(key))) {
    return Downcast<Integer>(value.value())->value;
  }
  return std::nullopt;
}

std::optional<std::string> GetPayloadString(const Map<String, Any>& payload, const char* key) {
  if (auto value = payload.Get(String(key))) {
    return str(Downcast<String>(value.value()));
  }
  return std::nullopt;
}

std::optional<std::vector<int64_t>> GetPayloadIndices(const Map<String, Any>& payload,
                                                      const char* key) {
  if (auto value = payload.Get(String(key))) {
    std::vector<int64_t> result;
    for (const Any& item : Downcast<Array<Any>>(value.value())) {
      result.push_back(Downcast<Integer>(item)->value);
    }
    return result;
  }
  return std::nullopt;
}

Array<String> ToStringArray(const std::vector<std::string>& values) {
  Array<String> result;
  for (const auto& value : values) {
    result.push_back(String(value));
  }
  return result;
}

Array<String> MakeTraits(std::initializer_list<const char*> values) {
  Array<String> result;
  for (const char* value : values) {
    result.push_back(String(value));
  }
  return result;
}

bool HasTrait(const Array<String>& traits, const char* trait) {
  for (const String& current : traits) {
    if (str(current) == trait) {
      return true;
    }
  }
  return false;
}

bool SameStringArray(const Array<String>& lhs, const Array<String>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (int i = 0; i < lhs.size(); ++i) {
    if (str(lhs[i]) != str(rhs[i])) {
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

std::unordered_map<std::string, std::optional<StateRole>> BuildStateRoleByName(
    const SemanticProgram& program) {
  std::unordered_map<std::string, std::optional<StateRole>> state_role_by_name;
  for (const State& state : program->states) {
    state_role_by_name[str(state->name)] = ParseStateRole(str(state->role));
  }
  return state_role_by_name;
}

bool HasSupplementPayload(const SemanticProgram& program, SupplementKind kind,
                          const char* payload_key) {
  for (const SemanticSupplement& supplement : program->supplements) {
    if (str(supplement->kind) != ToString(kind)) {
      continue;
    }
    auto maybe_payload = supplement->payload.Get(String(payload_key));
    if (!maybe_payload) {
      continue;
    }
    return !Downcast<Array<Any>>(maybe_payload.value()).empty();
  }
  return false;
}

bool AccessMapTouchesDomain(const AccessMap& access_map, const Domain& domain) {
  if (access_map->indices.empty()) {
    return true;
  }
  const std::unordered_set<std::string> axis_names = BuildDomainAxisNameSet(domain);
  bool touches_domain = false;
  for (const PrimExpr& index : access_map->indices) {
    tir::PostOrderVisit(index, [&](const ObjectRef& node) {
      if (const auto* var = node.as<tir::VarNode>()) {
        if (axis_names.count(var->name_hint)) {
          touches_domain = true;
        }
      }
    });
    if (touches_domain) {
      return true;
    }
  }
  return false;
}

bool UpdateTouchesDomain(const Update& update, const Domain& domain) {
  if (update->law->access_maps.empty()) {
    return true;
  }
  for (const AccessMap& access_map : update->law->access_maps) {
    if (AccessMapTouchesDomain(access_map, domain)) {
      return true;
    }
  }
  return false;
}

bool DomainHasStateRole(
    const SemanticProgram& program, const Domain& domain,
    const std::unordered_map<std::string, std::optional<StateRole>>& state_role_by_name,
    StateRole role) {
  auto has_role = [&](const std::string& state_name) {
    auto it = state_role_by_name.find(state_name);
    return it != state_role_by_name.end() && it->second && *it->second == role;
  };
  for (const Update& update : program->updates) {
    if (!UpdateTouchesDomain(update, domain)) {
      continue;
    }
    if (has_role(str(update->state_name))) {
      return true;
    }
    for (const String& source_state : update->law->source_states) {
      if (has_role(str(source_state))) {
        return true;
      }
    }
  }
  return false;
}

bool DomainHasAccessTrait(const SemanticProgram& program, const Domain& domain, const char* trait) {
  for (const Update& update : program->updates) {
    for (const AccessMap& access_map : update->law->access_maps) {
      if (AccessMapTouchesDomain(access_map, domain) && HasTrait(access_map->traits, trait)) {
        return true;
      }
    }
  }
  return false;
}

std::optional<Array<Any>> GetPipelineStagesFromSupplements(const SemanticProgram& program) {
  for (const SemanticSupplement& supplement : program->supplements) {
    if (str(supplement->kind) != ToString(SupplementKind::kPipelineStructure)) {
      continue;
    }
    auto maybe_payload = supplement->payload.Get(String(schema_key::kPipelineStages));
    if (!maybe_payload) {
      continue;
    }
    return Downcast<Array<Any>>(maybe_payload.value());
  }
  return std::nullopt;
}

std::optional<Array<Any>> GetWorkDependentLoopBoundsFromSupplements(
    const SemanticProgram& program) {
  for (const SemanticSupplement& supplement : program->supplements) {
    if (str(supplement->kind) != ToString(SupplementKind::kWorkDecompositionStructure)) {
      continue;
    }
    auto maybe_payload = supplement->payload.Get(String(schema_key::kWorkDependentLoopBounds));
    if (!maybe_payload) {
      continue;
    }
    return Downcast<Array<Any>>(maybe_payload.value());
  }
  return std::nullopt;
}

std::optional<Map<String, Any>> GetFragmentLoweringPayloadFromSupplements(
    const SemanticProgram& program) {
  for (const SemanticSupplement& supplement : program->supplements) {
    if (str(supplement->kind) != ToString(SupplementKind::kFragmentLoweringStructure)) {
      continue;
    }
    auto maybe_fragment_ops = supplement->payload.Get(String(schema_key::kFragmentOpKinds));
    if (maybe_fragment_ops && !Downcast<Array<Any>>(maybe_fragment_ops.value()).empty()) {
      return supplement->payload;
    }
  }
  return std::nullopt;
}

DomainRealizationContract DeriveDomainRealizationContract(
    const SemanticProgram& program, const Domain& domain,
    const std::unordered_map<std::string, std::optional<StateRole>>& state_role_by_name) {
  const bool multi_axis_domain = domain->axes.size() > 1;
  const bool has_derived_indices = HasTrait(domain->traits, "derived_indices");
  const bool has_work_dependent_bounds = HasTrait(domain->traits, "work_dependent_bounds");
  const bool has_pipeline_contract =
      HasTrait(domain->traits, "pipeline") ||
      HasSupplementPayload(program, SupplementKind::kPipelineStructure, schema_key::kPipelineStages);
  const bool has_selection_state =
      DomainHasStateRole(program, domain, state_role_by_name, StateRole::kSelectionState);
  const bool has_index_state =
      DomainHasStateRole(program, domain, state_role_by_name, StateRole::kIndexState);
  const bool has_reduction_accumulator = DomainHasStateRole(
      program, domain, state_role_by_name, StateRole::kReductionAccumulator);
  const bool has_selected_access = DomainHasAccessTrait(program, domain, "selected");
  const bool has_indexed_access = DomainHasAccessTrait(program, domain, "indexed");

  DomainRealizationContract contract;
  SetDomainRealizationKinds(&contract, sp::SpatialLayoutKind::kRegular,
                            multi_axis_domain ? sp::SpatialPartitionKind::kBlocked
                                              : sp::SpatialPartitionKind::kReplicated);
  if (has_derived_indices) {
    contract.domain_transform_kind = "derived";
    contract.partition_family = "derived";
    SetDomainRealizationKinds(&contract, sp::SpatialLayoutKind::kIndexed,
                              sp::SpatialPartitionKind::kIndexed);
    if (has_work_dependent_bounds && has_pipeline_contract) {
      contract.domain_transform_kind = "paged";
      contract.partition_family = "paged";
      return contract;
    }
    if (has_selection_state && has_selected_access && has_indexed_access && !has_index_state &&
        !has_reduction_accumulator) {
      contract.domain_transform_kind = "routed";
      contract.partition_family = "routed";
      return contract;
    }
    return contract;
  }
  if (has_selection_state && has_selected_access && !has_indexed_access) {
    contract.domain_transform_kind = "filtered";
    contract.partition_family = "filtered";
    SetDomainRealizationKinds(&contract, sp::SpatialLayoutKind::kRegular,
                              sp::SpatialPartitionKind::kFiltered);
    return contract;
  }
  if (has_selection_state && has_selected_access && has_indexed_access) {
    contract.domain_transform_kind = !has_index_state && !has_reduction_accumulator
                                         ? "chunked"
                                         : "grouped";
    contract.partition_family = contract.domain_transform_kind;
    SetDomainRealizationKinds(&contract, sp::SpatialLayoutKind::kRegular,
                              sp::SpatialPartitionKind::kFiltered);
    return contract;
  }
  return contract;
}

std::vector<DomainRealizationContract> DeriveDomainRealizationContracts(
    const SemanticProgram& program) {
  ICHECK(!program->domains.empty())
      << "spatial analysis requires SemanticProgram to carry at least one domain";
  const auto state_role_by_name = BuildStateRoleByName(program);
  std::vector<DomainRealizationContract> contracts;
  contracts.reserve(program->domains.size());
  for (const Domain& domain : program->domains) {
    contracts.push_back(DeriveDomainRealizationContract(program, domain, state_role_by_name));
  }
  return contracts;
}

std::unordered_map<std::string, std::string> BuildUniqueUpdateResultVersionByUpdate(
    const Array<StateDef>& state_defs) {
  std::unordered_map<std::string, std::string> update_result_version_by_update;
  for (const StateDef& def : state_defs) {
    const std::string producer_update = str(def->producer_update);
    if (producer_update.empty()) {
      continue;
    }
    const auto def_kind = ParseStateDefKind(str(def->kind));
    if (def_kind && *def_kind != StateDefKind::kUpdateResult) {
      continue;
    }
    const std::string version_name = str(def->version_name);
    auto [it, inserted] = update_result_version_by_update.emplace(producer_update, version_name);
    ICHECK(inserted || it->second == version_name)
        << "spatial analysis requires a unique update-result version for update "
        << producer_update;
  }
  return update_result_version_by_update;
}

std::unordered_map<std::string, StateJoin> BuildStateJoinByOutputVersion(
    const Array<StateJoin>& state_joins) {
  std::unordered_map<std::string, StateJoin> join_by_output_version;
  for (const StateJoin& join : state_joins) {
    join_by_output_version[str(join->output_version)] = join;
  }
  return join_by_output_version;
}

std::unordered_map<std::string, int> BuildDistinctConsumerCountByVersion(
    const Array<StateUse>& state_uses) {
  std::unordered_map<std::string, std::unordered_set<std::string>> consumers_by_version;
  for (const StateUse& use : state_uses) {
    consumers_by_version[str(use->version_name)].insert(str(use->consumer_update));
  }
  std::unordered_map<std::string, int> counts;
  for (const auto& [version_name, consumers] : consumers_by_version) {
    counts[version_name] = static_cast<int>(consumers.size());
  }
  return counts;
}

std::unordered_set<std::string> CollectKnownUpdateNames(const SemanticProgram& program) {
  std::unordered_set<std::string> known_updates;
  for (const Update& update : program->updates) {
    known_updates.insert(str(update->name));
  }
  return known_updates;
}

std::unordered_map<std::string, std::vector<ProducerVersionEdge>> BuildVersionProducerEdges(
    const SemanticProgram& program) {
  std::unordered_map<std::string, std::vector<ProducerVersionEdge>> direct_edges_by_version;
  for (const StateDef& def : program->state_defs) {
    const std::string producer_update = str(def->producer_update);
    if (producer_update.empty()) {
      continue;
    }
    direct_edges_by_version[str(def->version_name)].push_back(
        ProducerVersionEdge{producer_update, str(def->version_name)});
  }
  const auto join_by_output_version = BuildStateJoinByOutputVersion(program->state_joins);
  std::unordered_map<std::string, std::vector<ProducerVersionEdge>> version_to_producer_edges;
  std::unordered_set<std::string> active_versions;
  for (const StateDef& def : program->state_defs) {
    const std::string version_name = str(def->version_name);
    version_to_producer_edges[version_name] =
        ResolveProducerEdgesForVersion(version_name, direct_edges_by_version,
                                       join_by_output_version, &version_to_producer_edges,
                                       &active_versions);
  }
  for (const StateJoin& join : program->state_joins) {
    const std::string output_version = str(join->output_version);
    version_to_producer_edges[output_version] =
        ResolveProducerEdgesForVersion(output_version, direct_edges_by_version,
                                       join_by_output_version, &version_to_producer_edges,
                                       &active_versions);
  }
  return version_to_producer_edges;
}

std::string DeriveOrderingKindForChannel(sp::SpatialChannelKind channel_kind,
                                         sp::SpatialChannelDeliveryKind delivery_kind) {
  switch (channel_kind) {
    case sp::SpatialChannelKind::kCarry:
      return "carry_handoff";
    case sp::SpatialChannelKind::kReduceMerge:
      return "reduction_completion";
    case sp::SpatialChannelKind::kGather:
    case sp::SpatialChannelKind::kScatter:
      return "selection_index_handoff";
    default:
      return delivery_kind == sp::SpatialChannelDeliveryKind::kPhaseBoundaryMaterialized
                 ? "phase_boundary_materialization"
                 : "must_happen_before";
  }
}

std::string DeriveMaterializationKindForChannel(sp::SpatialChannelKind channel_kind,
                                                sp::SpatialChannelDeliveryKind delivery_kind) {
  if (delivery_kind == sp::SpatialChannelDeliveryKind::kPhaseBoundaryMaterialized) {
    return "phase_boundary_materialization";
  }
  if (channel_kind == sp::SpatialChannelKind::kReduceMerge) {
    return "completion_visibility";
  }
  return "phase_boundary";
}

}  // namespace tl
}  // namespace tvm
