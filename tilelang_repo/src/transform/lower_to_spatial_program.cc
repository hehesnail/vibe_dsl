/*!
 * \file lower_to_spatial_program.cc
 * \brief Materialize typed SpatialProgram companion IR from frozen SemanticProgram.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/transform.h>
#include <tvm/target/target.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/blackhole_utils.h"
#include "common/spatial_program.h"
#include "common/spatial_vocab.h"
#include "common/semantic_program.h"
#include "common/semantic_vocab.h"
#include "common/tt_hardware_model.h"

namespace tvm {
namespace tl {

using tvm::DictAttrs;
using tvm::GlobalInfo;
using tvm::Integer;
using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::String;
using namespace tvm::tl::semantic;
namespace sp = tvm::tl::spatial;
using tvm::tl::str;

// Cross-pass entry points.
tvm::transform::Pass AnalyzeSpatialDomainPlan();
tvm::transform::Pass AnalyzeSpatialExecutionPlan();
tvm::transform::Pass MaterializeSpatialProgram();

namespace {

struct SpatialProgramBundle {
  SpatialProgram program;
  Array<ProgramPhase> phases;
};

struct TaskSynthesisRecord {
  std::string task_name;
  std::string task_kind;
  std::string phase_name;
  int phase_rank = 0;
  std::string execution_role;
  std::string formation_basis;
  std::vector<std::string> update_names;
  std::vector<std::string> task_traits;
};

struct ChannelSynthesisRecord {
  std::string channel_name;
  sp::SpatialChannelKind flow_kind;
  sp::SpatialChannelPayloadKind payload_kind;
  sp::SpatialChannelDeliveryKind delivery_kind;
  bool ordering_critical = false;
  int source_task_index;
  int target_task_index;
  std::optional<int> state_index;
  std::optional<std::string> source_version;
  std::optional<std::string> target_version;
};

struct UpdateTaskFacts {
  std::optional<UpdateLawKind> law_kind;
  std::optional<StateRole> target_state_role;
  std::string update_result_version;
  std::string access_requirement_signature = "direct";
  std::string domain_transform_kind = "identity";
  std::string partition_family = "regular";
  std::string placement_domain = "logical_worker_grid";
  bool consumes_join_output = false;
  bool has_carried_input = false;
  bool has_companion_input = false;
  bool has_multi_consumer_input = false;
  bool output_feeds_join = false;
};

struct ProducerVersionEdge {
  std::string producer_update;
  std::string produced_version;
};

struct TaskFormationCandidate {
  std::string update_name;
  std::string task_kind;
  std::string execution_role;
  std::string formation_basis;
  std::vector<std::string> task_traits;
  UpdateTaskFacts facts;
};

struct TaskFormationResult {
  std::vector<TaskSynthesisRecord> task_records;
};

struct UpdateGraph {
  std::vector<std::string> topo_order;
  std::unordered_map<std::string, std::vector<std::string>> predecessors;
  std::unordered_map<std::string, std::vector<std::string>> successors;
};

struct DomainRealizationContract {
  std::string domain_transform_kind = "identity";
  std::string partition_family = "regular";
  sp::SpatialLayoutKind layout_kind = sp::SpatialLayoutKind::kRegular;
  sp::SpatialPartitionKind partition_kind = sp::SpatialPartitionKind::kReplicated;
};

struct PhaseSynthesisResult {
  std::vector<std::string> phase_order;
  std::unordered_map<std::string, int> phase_index_by_name;
};

struct PhaseSyncSynthesisRecord {
  std::string source_phase_name;
  std::string target_phase_name;
  int source_task_index = -1;
  int target_task_index = -1;
  std::string ordering_kind = "must_happen_before";
  std::string materialization_kind = "phase_boundary";
};

std::string JoinSortedUniqueLabels(std::vector<std::string> labels,
                                   const char* empty_fallback) {
  if (labels.empty()) {
    return empty_fallback;
  }
  std::sort(labels.begin(), labels.end());
  labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
  if (labels.size() == 1) {
    return labels.front();
  }
  std::string summary = "multi(";
  for (int i = 0; i < static_cast<int>(labels.size()); ++i) {
    if (i != 0) {
      summary += ",";
    }
    summary += labels[i];
  }
  summary += ")";
  return summary;
}

std::string BuildDomainTransformSignature(
    const std::vector<DomainRealizationContract>& domain_contracts,
    const std::vector<int>& domain_indices) {
  std::vector<std::string> labels;
  labels.reserve(domain_indices.empty() ? domain_contracts.size() : domain_indices.size());
  if (domain_indices.empty()) {
    for (const DomainRealizationContract& contract : domain_contracts) {
      labels.push_back(contract.domain_transform_kind);
    }
  } else {
    for (int domain_index : domain_indices) {
      ICHECK_GE(domain_index, 0);
      ICHECK_LT(domain_index, static_cast<int>(domain_contracts.size()));
      labels.push_back(domain_contracts[domain_index].domain_transform_kind);
    }
  }
  return JoinSortedUniqueLabels(std::move(labels), "identity");
}

std::string BuildPartitionFamilySignature(
    const std::vector<DomainRealizationContract>& domain_contracts,
    const std::vector<int>& domain_indices) {
  std::vector<std::string> labels;
  labels.reserve(domain_indices.empty() ? domain_contracts.size() : domain_indices.size());
  if (domain_indices.empty()) {
    for (const DomainRealizationContract& contract : domain_contracts) {
      labels.push_back(contract.partition_family);
    }
  } else {
    for (int domain_index : domain_indices) {
      ICHECK_GE(domain_index, 0);
      ICHECK_LT(domain_index, static_cast<int>(domain_contracts.size()));
      labels.push_back(domain_contracts[domain_index].partition_family);
    }
  }
  return JoinSortedUniqueLabels(std::move(labels), "regular");
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

std::unordered_map<std::string, std::optional<StateRole>> BuildStateRoleByName(
    const SemanticProgram& program) {
  std::unordered_map<std::string, std::optional<StateRole>> state_role_by_name;
  for (const State& state : program->states) {
    state_role_by_name[str(state->name)] = ParseStateRole(str(state->role));
  }
  return state_role_by_name;
}

std::unordered_set<std::string> BuildDomainAxisNameSet(const Domain& domain) {
  std::unordered_set<std::string> axis_names;
  for (const String& axis : domain->axes) {
    axis_names.insert(str(axis));
  }
  return axis_names;
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

std::vector<int> BuildUpdateDomainIndices(const Update& update, const Array<Domain>& domains) {
  std::vector<int> domain_indices;
  domain_indices.reserve(domains.size());
  for (int domain_index = 0; domain_index < domains.size(); ++domain_index) {
    if (UpdateTouchesDomain(update, domains[domain_index])) {
      domain_indices.push_back(domain_index);
    }
  }
  if (domain_indices.empty()) {
    for (int domain_index = 0; domain_index < domains.size(); ++domain_index) {
      domain_indices.push_back(domain_index);
    }
  }
  return domain_indices;
}

template <typename T>
void PushBackUnique(std::vector<T>* values, const T& value) {
  ICHECK(values != nullptr);
  if (std::find(values->begin(), values->end(), value) == values->end()) {
    values->push_back(value);
  }
}

void PushProducerEdgeUnique(std::vector<ProducerVersionEdge>* edges,
                            const ProducerVersionEdge& edge) {
  ICHECK(edges != nullptr);
  for (const ProducerVersionEdge& existing : *edges) {
    if (existing.producer_update == edge.producer_update &&
        existing.produced_version == edge.produced_version) {
      return;
    }
  }
  edges->push_back(edge);
}

Map<String, Any> EmptyPayload() { return Map<String, Any>(); }

Array<Any> ToIntegerAnyArray(const std::vector<int>& values) {
  Array<Any> result;
  for (int value : values) {
    result.push_back(Integer(value));
  }
  return result;
}

bool ContainsKind(const Array<String>& supported_kinds, const char* kind) {
  for (const String& supported_kind : supported_kinds) {
    if (str(supported_kind) == kind) {
      return true;
    }
  }
  return false;
}

void RequireCapabilitySupport(const Array<String>& supported_kinds, const char* kind,
                              const char* contract_label) {
  ICHECK(ContainsKind(supported_kinds, kind))
      << "LowerToSpatialProgram requires SpatialCapabilityModel support for " << contract_label
      << " kind " << kind;
}

const char* SelectLayoutKind(const SpatialCapabilityModel& capability_model,
                             const DomainRealizationContract& contract) {
  const char* layout_kind = sp::ToString(contract.layout_kind);
  RequireCapabilitySupport(capability_model->supported_layout_kinds, layout_kind, "layout");
  return layout_kind;
}

const char* SelectPartitionKind(const SpatialCapabilityModel& capability_model,
                                const DomainRealizationContract& contract) {
  const char* partition_kind = sp::ToString(contract.partition_kind);
  RequireCapabilitySupport(capability_model->supported_partition_kinds, partition_kind,
                           "partition");
  return partition_kind;
}

const char* SelectChannelKind(const SpatialCapabilityModel& capability_model,
                              sp::SpatialChannelKind channel_kind) {
  const char* kind = sp::ToString(channel_kind);
  RequireCapabilitySupport(capability_model->supported_flow_kinds, kind, "channel flow");
  return kind;
}

const char* SelectChannelPayloadKind(const SpatialCapabilityModel& capability_model,
                                     sp::SpatialChannelPayloadKind payload_kind) {
  const char* kind = sp::ToString(payload_kind);
  RequireCapabilitySupport(capability_model->supported_payload_kinds, kind, "channel payload");
  return kind;
}

const char* SelectChannelDeliveryKind(const SpatialCapabilityModel& capability_model,
                                      sp::SpatialChannelDeliveryKind delivery_kind) {
  const char* kind = sp::ToString(delivery_kind);
  RequireCapabilitySupport(capability_model->supported_delivery_kinds, kind, "channel delivery");
  return kind;
}

const char* NeutralPlacementAffinityForSegmentKind(const std::string& segment_kind) {
  if (segment_kind == "reader") {
    return "ingress";
  }
  if (segment_kind == "compute") {
    return "compute";
  }
  if (segment_kind == "writer") {
    return "egress";
  }
  return "compute";
}

const char* NeutralPlacementAffinityForTask(const Task& task) {
  auto maybe_task_kind = sp::ParseSpatialTaskKind(str(task->kind));
  if (!maybe_task_kind) {
    return "compute";
  }
  switch (*maybe_task_kind) {
    case sp::SpatialTaskKind::kTransfer:
      return "ingress";
    case sp::SpatialTaskKind::kCompute:
    case sp::SpatialTaskKind::kCollective:
    case sp::SpatialTaskKind::kControl:
      return "compute";
  }
  return "compute";
}

std::string StateRoleLabel(const std::optional<StateRole>& maybe_role) {
  return maybe_role ? std::string(ToString(*maybe_role)) : std::string("stateless");
}

bool IsStatefulRole(const std::optional<StateRole>& maybe_role) {
  return maybe_role &&
         (*maybe_role == StateRole::kCarry ||
          *maybe_role == StateRole::kReductionAccumulator ||
          *maybe_role == StateRole::kSelectionState ||
          *maybe_role == StateRole::kIndexState);
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
    auto [it, inserted] =
        update_result_version_by_update.emplace(producer_update, version_name);
    ICHECK(inserted || it->second == version_name)
        << "LowerToSpatialProgram requires a unique update-result version for update "
        << producer_update;
  }
  return update_result_version_by_update;
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
      << "LowerToSpatialProgram found cyclic semantic join producer resolution for version "
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
        PushProducerEdgeUnique(&edges, edge);
      }
    }
  }

  active_versions->erase(version_name);
  (*memoized_edges)[version_name] = edges;
  return edges;
}

std::unordered_map<std::string, std::vector<ProducerVersionEdge>> BuildVersionProducerEdges(
    const SemanticProgram& program,
    const std::unordered_set<std::string>& known_updates) {
  std::unordered_map<std::string, std::vector<ProducerVersionEdge>> direct_edges_by_version;
  for (const StateDef& def : program->state_defs) {
    const std::string producer_update = str(def->producer_update);
    if (producer_update.empty() || !known_updates.count(producer_update)) {
      continue;
    }
    PushProducerEdgeUnique(&direct_edges_by_version[str(def->version_name)],
                           ProducerVersionEdge{producer_update, str(def->version_name)});
  }
  std::unordered_map<std::string, StateJoin> join_by_output_version;
  for (const StateJoin& join : program->state_joins) {
    join_by_output_version[str(join->output_version)] = join;
  }

  std::unordered_map<std::string, std::vector<ProducerVersionEdge>> version_to_producer_edges;
  std::unordered_set<std::string> active_versions;
  for (const StateDef& def : program->state_defs) {
    const std::string version_name = str(def->version_name);
    std::vector<ProducerVersionEdge> resolved_edges = ResolveProducerEdgesForVersion(
        version_name, direct_edges_by_version, join_by_output_version, &version_to_producer_edges,
        &active_versions);
    version_to_producer_edges[version_name] = std::move(resolved_edges);
  }
  for (const StateJoin& join : program->state_joins) {
    const std::string output_version = str(join->output_version);
    std::vector<ProducerVersionEdge> resolved_edges = ResolveProducerEdgesForVersion(
        output_version, direct_edges_by_version, join_by_output_version,
        &version_to_producer_edges, &active_versions);
    version_to_producer_edges[output_version] = std::move(resolved_edges);
  }
  return version_to_producer_edges;
}

std::unordered_set<std::string> CollectJoinOutputVersions(const Array<StateJoin>& state_joins) {
  std::unordered_set<std::string> join_output_versions;
  for (const StateJoin& join : state_joins) {
    join_output_versions.insert(str(join->output_version));
  }
  return join_output_versions;
}

std::unordered_map<std::string, StateJoin> BuildStateJoinByOutputVersion(
    const Array<StateJoin>& state_joins) {
  std::unordered_map<std::string, StateJoin> join_by_output_version;
  for (const StateJoin& join : state_joins) {
    join_by_output_version[str(join->output_version)] = join;
  }
  return join_by_output_version;
}

std::unordered_set<std::string> CollectJoinInputVersions(const Array<StateJoin>& state_joins) {
  std::unordered_set<std::string> join_input_versions;
  for (const StateJoin& join : state_joins) {
    for (const String& input_version : join->input_versions) {
      join_input_versions.insert(str(input_version));
    }
  }
  return join_input_versions;
}

std::unordered_map<std::string, int> BuildDistinctConsumerCountByVersion(
    const Array<StateUse>& state_uses) {
  std::unordered_map<std::string, std::unordered_set<std::string>> consumer_updates_by_version;
  for (const StateUse& use : state_uses) {
    consumer_updates_by_version[str(use->version_name)].insert(str(use->consumer_update));
  }

  std::unordered_map<std::string, int> consumer_count_by_version;
  for (const auto& [version_name, consumer_updates] : consumer_updates_by_version) {
    consumer_count_by_version[version_name] = static_cast<int>(consumer_updates.size());
  }
  return consumer_count_by_version;
}

std::string BuildAccessRequirementSignature(const Update& update) {
  std::vector<std::string> access_requirements;
  for (const AccessMap& access_map : update->law->access_maps) {
    std::vector<std::string> traits;
    for (const String& trait : access_map->traits) {
      traits.push_back(str(trait));
    }
    std::sort(traits.begin(), traits.end());
    std::string entry = str(access_map->kind);
    if (!traits.empty()) {
      entry += "(";
      for (int i = 0; i < static_cast<int>(traits.size()); ++i) {
        if (i != 0) {
          entry += ",";
        }
        entry += traits[i];
      }
      entry += ")";
    }
    access_requirements.push_back(std::move(entry));
  }
  if (access_requirements.empty()) {
    return "direct";
  }
  std::sort(access_requirements.begin(), access_requirements.end());
  std::string result;
  for (int i = 0; i < static_cast<int>(access_requirements.size()); ++i) {
    if (i != 0) {
      result += ";";
    }
    result += access_requirements[i];
  }
  return result;
}

std::unordered_map<std::string, UpdateTaskFacts> BuildUpdateTaskFacts(
    const SemanticProgram& program,
    const std::unordered_map<std::string, std::optional<StateRole>>& state_role_by_name,
    const std::vector<DomainRealizationContract>& domain_contracts,
    const SpatialCapabilityModel& capability_model) {
  const auto update_result_version_by_update =
      BuildUniqueUpdateResultVersionByUpdate(program->state_defs);
  const auto join_output_versions = CollectJoinOutputVersions(program->state_joins);
  const auto join_input_versions = CollectJoinInputVersions(program->state_joins);
  const auto consumer_count_by_version = BuildDistinctConsumerCountByVersion(program->state_uses);

  std::unordered_map<std::string, UpdateTaskFacts> facts_by_update;
  for (const Update& update : program->updates) {
    const std::string update_name = str(update->name);
    const std::vector<int> update_domain_indices =
        BuildUpdateDomainIndices(update, program->domains);
    UpdateTaskFacts facts;
    facts.law_kind = ParseUpdateLawKind(str(update->law->kind));
    const std::string state_name = str(update->state_name);
    auto state_role_it = state_role_by_name.find(state_name);
    if (state_role_it != state_role_by_name.end()) {
      facts.target_state_role = state_role_it->second;
    }
    facts.access_requirement_signature = BuildAccessRequirementSignature(update);
    facts.domain_transform_kind =
        BuildDomainTransformSignature(domain_contracts, update_domain_indices);
    facts.partition_family =
        BuildPartitionFamilySignature(domain_contracts, update_domain_indices);
    facts.placement_domain = str(capability_model->placement_domain);
    auto version_it = update_result_version_by_update.find(update_name);
    if (version_it != update_result_version_by_update.end()) {
      facts.update_result_version = version_it->second;
      facts.output_feeds_join = join_input_versions.count(version_it->second);
    }
    facts_by_update.emplace(update_name, std::move(facts));
  }

  for (const StateUse& use : program->state_uses) {
    const std::string consumer_update = str(use->consumer_update);
    auto fact_it = facts_by_update.find(consumer_update);
    if (fact_it == facts_by_update.end()) {
      continue;
    }
    fact_it->second.consumes_join_output |= join_output_versions.count(str(use->version_name));
    auto consumer_count_it = consumer_count_by_version.find(str(use->version_name));
    if (consumer_count_it != consumer_count_by_version.end() && consumer_count_it->second > 1) {
      fact_it->second.has_multi_consumer_input = true;
    }
    auto use_kind = ParseStateUseKind(str(use->kind));
    if (!use_kind) {
      continue;
    }
    fact_it->second.has_carried_input |= *use_kind == StateUseKind::kCarriedState;
    fact_it->second.has_companion_input |= *use_kind == StateUseKind::kCompanionState;
  }

  return facts_by_update;
}

std::unordered_set<std::string> CollectKnownUpdateNames(const SemanticProgram& program) {
  std::unordered_set<std::string> known_updates;
  for (int i = 0; i < program->updates.size(); ++i) {
    known_updates.insert(str(program->updates[i]->name));
  }
  return known_updates;
}

std::string DeriveExecutionRole(const UpdateTaskFacts& facts) {
  const std::string role_label = StateRoleLabel(facts.target_state_role);
  if (!facts.law_kind) {
    return role_label + "_compute";
  }
  switch (*facts.law_kind) {
    case UpdateLawKind::kMap:
      return facts.target_state_role ? role_label + "_materialize" : "bootstrap";
    case UpdateLawKind::kReduce:
      return facts.target_state_role &&
                     *facts.target_state_role == StateRole::kReductionAccumulator
                 ? "reduction"
                 : role_label + "_reduce";
    case UpdateLawKind::kSelect:
      if (facts.target_state_role &&
          *facts.target_state_role == StateRole::kSelectionState) {
        return "selection";
      }
      if (facts.target_state_role && *facts.target_state_role == StateRole::kIndexState) {
        return "index_selection";
      }
      return role_label + "_selection";
    case UpdateLawKind::kRecurrence:
      if (facts.target_state_role && *facts.target_state_role == StateRole::kCarry) {
        return "carry_refresh";
      }
      if (facts.target_state_role &&
          *facts.target_state_role == StateRole::kReductionAccumulator) {
        return "reduction_refresh";
      }
      if (facts.target_state_role &&
          *facts.target_state_role == StateRole::kSelectionState) {
        return "selection_refresh";
      }
      if (facts.target_state_role && *facts.target_state_role == StateRole::kIndexState) {
        return "index_refresh";
      }
      return "recurrence_refresh";
  }
  return role_label + "_compute";
}

std::string DeriveFormationBasis(const UpdateTaskFacts& facts) {
  const std::string law_label =
      facts.law_kind ? std::string(ToString(*facts.law_kind)) : std::string("unknown");
  const std::string role_label = StateRoleLabel(facts.target_state_role);
  std::string boundary_label = "direct_state_boundary";
  if (!facts.target_state_role) {
    boundary_label = "stateless_boundary";
  } else if (facts.consumes_join_output || facts.has_carried_input || facts.output_feeds_join) {
    boundary_label = "loop_carried_boundary";
  } else if (facts.has_multi_consumer_input) {
    boundary_label = "fanout_boundary";
  } else if (facts.law_kind && *facts.law_kind == UpdateLawKind::kReduce) {
    boundary_label = "reduction_boundary";
  } else if ((facts.law_kind && *facts.law_kind == UpdateLawKind::kSelect) ||
             (facts.target_state_role &&
              (*facts.target_state_role == StateRole::kSelectionState ||
               *facts.target_state_role == StateRole::kIndexState))) {
    boundary_label = facts.has_companion_input ? "selection_companion_boundary"
                                               : "selection_boundary";
  } else if (facts.target_state_role &&
             *facts.target_state_role == StateRole::kCarry) {
    boundary_label = "carry_boundary";
  }

  return "single_update|law=" + law_label + "|target_role=" + role_label +
         "|boundary=" + boundary_label +
         "|access=" + facts.access_requirement_signature +
         "|domain=" + facts.domain_transform_kind +
         "|partition=" + facts.partition_family;
}

std::string DeriveTaskKindFromFacts(const UpdateTaskFacts& facts) {
  if (!facts.law_kind) {
    return sp::ToString(sp::SpatialTaskKind::kCompute);
  }
  switch (*facts.law_kind) {
    case UpdateLawKind::kMap:
      return sp::ToString(sp::SpatialTaskKind::kCompute);
    case UpdateLawKind::kReduce:
      return sp::ToString(sp::SpatialTaskKind::kCollective);
    case UpdateLawKind::kSelect:
    case UpdateLawKind::kRecurrence:
      return sp::ToString(sp::SpatialTaskKind::kControl);
  }
  return sp::ToString(sp::SpatialTaskKind::kCompute);
}

bool HasMandatoryTaskBoundary(const UpdateTaskFacts& facts) {
  return facts.has_carried_input || facts.has_companion_input || facts.consumes_join_output ||
         facts.has_multi_consumer_input || facts.output_feeds_join ||
         (facts.law_kind &&
          (*facts.law_kind == UpdateLawKind::kReduce || *facts.law_kind == UpdateLawKind::kSelect ||
           *facts.law_kind == UpdateLawKind::kRecurrence)) ||
         (facts.target_state_role &&
          (*facts.target_state_role == StateRole::kCarry ||
           *facts.target_state_role == StateRole::kReductionAccumulator ||
           *facts.target_state_role == StateRole::kSelectionState ||
           *facts.target_state_role == StateRole::kIndexState));
}

int DeriveOrderingStageHint(const UpdateTaskFacts& facts) {
  if ((facts.law_kind && *facts.law_kind == UpdateLawKind::kRecurrence) ||
      facts.has_carried_input || facts.has_companion_input || facts.consumes_join_output) {
    return 1;
  }
  return 0;
}

TaskFormationCandidate BuildTaskFormationCandidate(const Update& update,
                                                   const UpdateTaskFacts& facts) {
  TaskFormationCandidate candidate;
  candidate.update_name = str(update->name);
  candidate.task_kind = DeriveTaskKindFromFacts(facts);
  candidate.execution_role = DeriveExecutionRole(facts);
  candidate.formation_basis = DeriveFormationBasis(facts);
  candidate.task_traits = {"phase_b", str(update->law->kind)};
  candidate.facts = facts;
  return candidate;
}

bool CanFuseTaskFormationCandidates(const TaskFormationCandidate& lhs,
                                    const TaskFormationCandidate& rhs) {
  if (lhs.task_kind != rhs.task_kind || lhs.execution_role != rhs.execution_role ||
      lhs.facts.law_kind != rhs.facts.law_kind ||
      lhs.facts.target_state_role != rhs.facts.target_state_role ||
      lhs.facts.access_requirement_signature != rhs.facts.access_requirement_signature ||
      lhs.facts.domain_transform_kind != rhs.facts.domain_transform_kind ||
      lhs.facts.partition_family != rhs.facts.partition_family ||
      lhs.facts.placement_domain != rhs.facts.placement_domain) {
    return false;
  }
  if (HasMandatoryTaskBoundary(lhs.facts) || HasMandatoryTaskBoundary(rhs.facts)) {
    return false;
  }
  return lhs.facts.law_kind && *lhs.facts.law_kind == UpdateLawKind::kMap &&
         !lhs.facts.target_state_role && !rhs.facts.target_state_role;
}

TaskSynthesisRecord MaterializeTaskRecord(const std::vector<TaskFormationCandidate>& group) {
  ICHECK(!group.empty());
  const TaskFormationCandidate& leader = group.front();
  TaskSynthesisRecord record;
  record.task_name = group.size() == 1 ? leader.update_name : "task_" + leader.update_name;
  record.task_kind = leader.task_kind;
  record.phase_rank = DeriveOrderingStageHint(leader.facts);
  record.execution_role =
      group.size() == 1 ? leader.execution_role : leader.execution_role + "_fused";
  record.formation_basis =
      group.size() == 1 ? leader.formation_basis
                        : "fused_updates|execution_role=" + leader.execution_role;
  record.task_traits = leader.task_traits;
  if (group.size() > 1) {
    record.task_traits.push_back("fused");
  }
  for (const TaskFormationCandidate& candidate : group) {
    record.update_names.push_back(candidate.update_name);
  }
  return record;
}

UpdateGraph BuildUpdateGraph(const SemanticProgram& program) {
  UpdateGraph graph;
  const auto known_updates = CollectKnownUpdateNames(program);
  const auto version_to_producer_edges = BuildVersionProducerEdges(program, known_updates);

  std::unordered_map<std::string, int> update_order_index;
  std::vector<std::string> update_names;
  update_names.reserve(program->updates.size());
  for (int i = 0; i < program->updates.size(); ++i) {
    const std::string update_name = str(program->updates[i]->name);
    update_order_index[update_name] = i;
    update_names.push_back(update_name);
    graph.predecessors[update_name] = {};
    graph.successors[update_name] = {};
  }

  auto add_edge = [&](const std::string& source_update, const std::string& target_update) {
    if (source_update == target_update || !known_updates.count(source_update) ||
        !known_updates.count(target_update)) {
      return;
    }
    PushBackUnique(&graph.successors[source_update], target_update);
    PushBackUnique(&graph.predecessors[target_update], source_update);
  };

  for (const StateUse& use : program->state_uses) {
    const std::string consumer_update = str(use->consumer_update);
    auto producer_edges_it = version_to_producer_edges.find(str(use->version_name));
    if (!known_updates.count(consumer_update) || producer_edges_it == version_to_producer_edges.end()) {
      continue;
    }
    for (const ProducerVersionEdge& edge : producer_edges_it->second) {
      add_edge(edge.producer_update, consumer_update);
    }
  }

  std::unordered_map<std::string, int> indegree;
  for (const std::string& update_name : update_names) {
    indegree[update_name] = static_cast<int>(graph.predecessors[update_name].size());
  }
  std::vector<std::string> ready;
  for (const std::string& update_name : update_names) {
    if (indegree[update_name] == 0) {
      ready.push_back(update_name);
    }
  }
  auto ready_less = [&](const std::string& lhs, const std::string& rhs) {
    return update_order_index.at(lhs) < update_order_index.at(rhs);
  };
  std::sort(ready.begin(), ready.end(), ready_less);

  while (!ready.empty()) {
    const std::string current = ready.front();
    ready.erase(ready.begin());
    graph.topo_order.push_back(current);
    for (const std::string& successor : graph.successors[current]) {
      int& successor_indegree = indegree[successor];
      --successor_indegree;
      if (successor_indegree == 0) {
        ready.push_back(successor);
        std::sort(ready.begin(), ready.end(), ready_less);
      }
    }
  }

  if (graph.topo_order.size() != update_names.size()) {
    std::unordered_set<std::string> emitted(graph.topo_order.begin(), graph.topo_order.end());
    std::vector<std::string> unresolved_updates;
    for (const std::string& update_name : update_names) {
      if (!emitted.count(update_name)) {
        unresolved_updates.push_back(update_name);
      }
    }
    std::sort(unresolved_updates.begin(), unresolved_updates.end(), ready_less);
    std::string unresolved_summary;
    for (int i = 0; i < static_cast<int>(unresolved_updates.size()); ++i) {
      if (i != 0) {
        unresolved_summary += ", ";
      }
      unresolved_summary += unresolved_updates[i];
    }
    ICHECK_EQ(graph.topo_order.size(), update_names.size())
        << "LowerToSpatialProgram requires semantic update graph to be a DAG; "
        << "unresolved updates: " << unresolved_summary;
  }

  return graph;
}

TaskFormationResult BuildTaskRecords(
    const SemanticProgram& program,
    const std::unordered_map<std::string, std::optional<StateRole>>& state_role_by_name,
    const std::vector<DomainRealizationContract>& domain_contracts,
    const SpatialCapabilityModel& capability_model) {
  const auto update_task_facts_by_name =
      BuildUpdateTaskFacts(program, state_role_by_name, domain_contracts, capability_model);
  const UpdateGraph update_graph = BuildUpdateGraph(program);
  ICHECK_EQ(update_graph.topo_order.size(), static_cast<size_t>(program->updates.size()))
      << "LowerToSpatialProgram requires task formation graph to cover all semantic updates";

  std::unordered_map<std::string, Update> update_by_name;
  for (int i = 0; i < program->updates.size(); ++i) {
    update_by_name[str(program->updates[i]->name)] = program->updates[i];
  }

  std::vector<TaskFormationCandidate> candidates;
  candidates.reserve(program->updates.size());
  std::unordered_map<std::string, TaskFormationCandidate> candidate_by_update;
  for (const std::string& update_name : update_graph.topo_order) {
    auto fact_it = update_task_facts_by_name.find(update_name);
    ICHECK(fact_it != update_task_facts_by_name.end())
        << "LowerToSpatialProgram missing task synthesis facts for update " << update_name;
    auto update_it = update_by_name.find(update_name);
    ICHECK(update_it != update_by_name.end())
        << "LowerToSpatialProgram missing semantic update definition for " << update_name;
    TaskFormationCandidate candidate =
        BuildTaskFormationCandidate(update_it->second, fact_it->second);
    candidate_by_update[update_name] = candidate;
    candidates.push_back(std::move(candidate));
  }

  TaskFormationResult result;
  std::unordered_set<std::string> grouped_updates;
  for (const std::string& update_name : update_graph.topo_order) {
    if (grouped_updates.count(update_name)) {
      continue;
    }
    std::vector<TaskFormationCandidate> group{candidate_by_update.at(update_name)};
    grouped_updates.insert(update_name);

    std::string current_update = update_name;
    while (true) {
      auto successor_it = update_graph.successors.find(current_update);
      if (successor_it == update_graph.successors.end() || successor_it->second.size() != 1) {
        break;
      }
      const std::string& next_update = successor_it->second.front();
      auto predecessor_it = update_graph.predecessors.find(next_update);
      if (predecessor_it == update_graph.predecessors.end() || predecessor_it->second.size() != 1 ||
          grouped_updates.count(next_update)) {
        break;
      }
      if (!CanFuseTaskFormationCandidates(candidate_by_update.at(current_update),
                                          candidate_by_update.at(next_update))) {
        break;
      }
      group.push_back(candidate_by_update.at(next_update));
      grouped_updates.insert(next_update);
      current_update = next_update;
    }

    TaskSynthesisRecord record = MaterializeTaskRecord(group);
    result.task_records.push_back(std::move(record));
  }
  return result;
}

sp::SpatialChannelKind DeriveStateVersionBaseFlowKind(const StateUse& use,
                                                      const std::optional<StateRole>& maybe_role,
                                                      const std::optional<StateJoinKind>& join_kind,
                                                      int version_consumer_count) {
  if (maybe_role) {
    switch (*maybe_role) {
      case StateRole::kCarry:
        return sp::SpatialChannelKind::kCarry;
      case StateRole::kReductionAccumulator:
        return sp::SpatialChannelKind::kReduceMerge;
      case StateRole::kSelectionState:
        return sp::SpatialChannelKind::kGather;
      case StateRole::kIndexState:
        return sp::SpatialChannelKind::kScatter;
      default:
        break;
    }
  }
  auto use_kind = ParseStateUseKind(str(use->kind));
  if (version_consumer_count > 1 && !join_kind.has_value() &&
      !(use_kind && *use_kind == StateUseKind::kCarriedState) &&
      !(use_kind && *use_kind == StateUseKind::kCompanionState)) {
    return sp::SpatialChannelKind::kBroadcast;
  }
  if ((join_kind && *join_kind == StateJoinKind::kLoopCarried) ||
      (use_kind && *use_kind == StateUseKind::kCarriedState)) {
    return sp::SpatialChannelKind::kCarry;
  }
  return sp::SpatialChannelKind::kPointToPoint;
}

sp::SpatialChannelDeliveryKind DeriveStateVersionDeliveryKind(sp::SpatialChannelKind flow_kind,
                                                              bool cross_phase) {
  if (cross_phase) {
    return sp::SpatialChannelDeliveryKind::kPhaseBoundaryMaterialized;
  }
  if (flow_kind == sp::SpatialChannelKind::kReduceMerge) {
    return sp::SpatialChannelDeliveryKind::kCompletionVisible;
  }
  return sp::SpatialChannelDeliveryKind::kOrdered;
}

bool IsOrderingCriticalChannel(const ChannelSynthesisRecord& record) {
  if (record.delivery_kind == sp::SpatialChannelDeliveryKind::kPhaseBoundaryMaterialized) {
    return true;
  }
  return record.flow_kind == sp::SpatialChannelKind::kCarry ||
         record.flow_kind == sp::SpatialChannelKind::kReduceMerge ||
         record.flow_kind == sp::SpatialChannelKind::kGather ||
         record.flow_kind == sp::SpatialChannelKind::kScatter;
}

bool IsOrderingCriticalFlowKind(sp::SpatialChannelKind flow_kind) {
  return flow_kind == sp::SpatialChannelKind::kCarry ||
         flow_kind == sp::SpatialChannelKind::kReduceMerge ||
         flow_kind == sp::SpatialChannelKind::kGather ||
         flow_kind == sp::SpatialChannelKind::kScatter;
}

std::string DeriveOrderingKindForChannelRecord(const ChannelSynthesisRecord& record) {
  switch (record.flow_kind) {
    case sp::SpatialChannelKind::kCarry:
      return "carry_handoff";
    case sp::SpatialChannelKind::kReduceMerge:
      return "reduction_completion";
    case sp::SpatialChannelKind::kGather:
    case sp::SpatialChannelKind::kScatter:
      return "selection_index_handoff";
    default:
      return record.delivery_kind == sp::SpatialChannelDeliveryKind::kPhaseBoundaryMaterialized
                 ? "phase_boundary_materialization"
                 : "must_happen_before";
  }
}

std::string DeriveMaterializationKindForChannelRecord(const ChannelSynthesisRecord& record) {
  if (record.delivery_kind == sp::SpatialChannelDeliveryKind::kPhaseBoundaryMaterialized) {
    return "phase_boundary_materialization";
  }
  if (record.flow_kind == sp::SpatialChannelKind::kReduceMerge) {
    return "completion_visibility";
  }
  return "phase_boundary";
}


Map<String, Any> BuildDomainPayload(int domain_index,
                                    const DomainRealizationContract& contract) {
  Map<String, Any> payload;
  payload.Set(String(schema_key::kDomainIndex), Integer(domain_index));
  payload.Set(String(schema_key::kDomainTransformKind), String(contract.domain_transform_kind));
  return payload;
}

Map<String, Any> BuildTargetPayload(const char* target_kind, int target_index) {
  Map<String, Any> payload;
  payload.Set(String(schema_key::kTargetKind), String(target_kind));
  payload.Set(String(schema_key::kTargetIndex), Integer(target_index));
  return payload;
}

Map<String, Any> BuildMemberFuncTargetPayload() {
  Map<String, Any> payload;
  payload.Set(String(schema_key::kTargetKind), String(spatial_contract::kMemberFuncTarget));
  return payload;
}

Map<String, Any> BuildTaskPayload(int phase_index) {
  Map<String, Any> payload;
  payload.Set(String(schema_key::kPhaseIndex), Integer(phase_index));
  return payload;
}

Map<String, Any> BuildTaskPayload(int phase_index, const TaskSynthesisRecord& record) {
  Map<String, Any> payload = BuildTaskPayload(phase_index);
  if (!record.execution_role.empty()) {
    payload.Set(String(schema_key::kExecutionRole), String(record.execution_role));
  }
  if (!record.formation_basis.empty()) {
    payload.Set(String(schema_key::kFormationBasis), String(record.formation_basis));
  }
  return payload;
}

Map<String, Any> BuildChannelPayload(int source_task_index, int target_task_index,
                                     const char* payload_kind, const char* delivery_kind,
                                     std::optional<int> state_index = std::nullopt) {
  Map<String, Any> payload;
  payload.Set(String(schema_key::kSourceTaskIndex), Integer(source_task_index));
  payload.Set(String(schema_key::kTargetTaskIndex), Integer(target_task_index));
  payload.Set(String(schema_key::kPayloadKind), String(payload_kind));
  payload.Set(String(schema_key::kDeliveryKind), String(delivery_kind));
  if (state_index.has_value()) {
    payload.Set(String(schema_key::kStateIndex), Integer(state_index.value()));
  }
  return payload;
}

Map<String, Any> BuildChannelPayload(const ChannelSynthesisRecord& record, const char* payload_kind,
                                     const char* delivery_kind) {
  Map<String, Any> payload =
      BuildChannelPayload(record.source_task_index, record.target_task_index, payload_kind,
                          delivery_kind, record.state_index);
  if (record.source_version.has_value()) {
    payload.Set(String(schema_key::kSourceVersion), String(record.source_version.value()));
  }
  if (record.target_version.has_value()) {
    payload.Set(String(schema_key::kTargetVersion), String(record.target_version.value()));
  }
  return payload;
}

Map<String, Any> BuildPlacementPayload(int task_index, const char* affinity_kind,
                                       const char* obligation_kind,
                                       const char* placement_domain) {
  Map<String, Any> payload;
  payload.Set(String(schema_key::kTaskIndex), Integer(task_index));
  payload.Set(String(schema_key::kAffinityKind), String(affinity_kind));
  payload.Set(String(schema_key::kObligationKind), String(obligation_kind));
  payload.Set(String(schema_key::kPlacementDomain), String(placement_domain));
  return payload;
}

Map<String, Any> BuildSyncEdgePayload(int source_task_index, int target_task_index,
                                      const std::string& ordering_kind,
                                      const std::string& materialization_kind) {
  Map<String, Any> payload;
  payload.Set(String(schema_key::kSourceTaskIndex), Integer(source_task_index));
  payload.Set(String(schema_key::kTargetTaskIndex), Integer(target_task_index));
  payload.Set(String(schema_key::kOrderingKind), String(ordering_kind));
  payload.Set(String(schema_key::kMaterializationKind), String(materialization_kind));
  return payload;
}

Map<String, Any> BuildProgramPhasePayload(int phase_index, const std::vector<int>& task_indices,
                                          const std::vector<int>& channel_indices,
                                          const std::string& closure_basis) {
  Map<String, Any> payload;
  payload.Set(String(schema_key::kPhaseIndex), Integer(phase_index));
  payload.Set(String(schema_key::kTaskIndices), ToIntegerAnyArray(task_indices));
  payload.Set(String(schema_key::kChannelIndices), ToIntegerAnyArray(channel_indices));
  payload.Set(String(schema_key::kClosureBasis), String(closure_basis));
  return payload;
}

Array<TIRAnchor> MakeAnchors(const std::string& kind, const std::string& value) {
  return Array<TIRAnchor>{TIRAnchor(String(kind), String(value))};
}

std::string GetMemberFuncName(const GlobalVar& gvar, const tir::PrimFunc& func) {
  return func->GetAttr<String>(tvm::attr::kGlobalSymbol).value_or(gvar->name_hint);
}

std::optional<Array<Any>> GetPipelineStagesFromSupplements(const SemanticProgram& program) {
  for (const SemanticSupplement& supplement : program->supplements) {
    if (str(supplement->kind) !=
        ToString(SupplementKind::kPipelineStructure)) {
      continue;
    }
    if (auto pipeline_stages = supplement->payload.Get(String(schema_key::kPipelineStages))) {
      return Downcast<Array<Any>>(pipeline_stages.value());
    }
  }
  return std::nullopt;
}

std::optional<Map<String, Any>> GetFragmentLoweringPayloadFromSupplements(
    const SemanticProgram& program) {
  for (const SemanticSupplement& supplement : program->supplements) {
    if (str(supplement->kind) !=
        ToString(SupplementKind::kFragmentLoweringStructure)) {
      continue;
    }
    auto maybe_fragment_ops =
        supplement->payload.Get(String(schema_key::kFragmentOpKinds));
    if (maybe_fragment_ops &&
        !Downcast<Array<Any>>(maybe_fragment_ops.value()).empty()) {
      return supplement->payload;
    }
  }
  return std::nullopt;
}

std::optional<Array<Any>> GetWorkDependentLoopBoundsFromSupplements(const SemanticProgram& program) {
  for (const SemanticSupplement& supplement : program->supplements) {
    if (str(supplement->kind) !=
        ToString(SupplementKind::kWorkDecompositionStructure)) {
      continue;
    }
    if (auto loop_bounds =
            supplement->payload.Get(String(schema_key::kWorkDependentLoopBounds))) {
      return Downcast<Array<Any>>(loop_bounds.value());
    }
  }
  return std::nullopt;
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

void SetDomainRealizationKinds(DomainRealizationContract* contract,
                               sp::SpatialLayoutKind layout_kind,
                               sp::SpatialPartitionKind partition_kind) {
  ICHECK(contract != nullptr);
  contract->layout_kind = layout_kind;
  contract->partition_kind = partition_kind;
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
      SetDomainRealizationKinds(&contract, sp::SpatialLayoutKind::kIndexed,
                                sp::SpatialPartitionKind::kIndexed);
      return contract;
    }
    if (has_selection_state && has_selected_access && has_indexed_access && !has_index_state &&
        !has_reduction_accumulator) {
      contract.domain_transform_kind = "routed";
      contract.partition_family = "routed";
      SetDomainRealizationKinds(&contract, sp::SpatialLayoutKind::kIndexed,
                                sp::SpatialPartitionKind::kIndexed);
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
    if (!has_index_state && !has_reduction_accumulator) {
      contract.domain_transform_kind = "chunked";
      contract.partition_family = "chunked";
      SetDomainRealizationKinds(&contract, sp::SpatialLayoutKind::kRegular,
                                sp::SpatialPartitionKind::kFiltered);
      return contract;
    }
    contract.domain_transform_kind = "grouped";
    contract.partition_family = "grouped";
    SetDomainRealizationKinds(&contract, sp::SpatialLayoutKind::kRegular,
                              sp::SpatialPartitionKind::kFiltered);
    return contract;
  }

  return contract;
}

std::vector<DomainRealizationContract> DeriveDomainRealizationContracts(
    const SemanticProgram& program) {
  ICHECK(!program->domains.empty())
      << "LowerToSpatialProgram requires SemanticProgram to carry at least one domain";
  const auto state_role_by_name = BuildStateRoleByName(program);
  std::vector<DomainRealizationContract> contracts;
  contracts.reserve(program->domains.size());
  for (const Domain& domain : program->domains) {
    contracts.push_back(DeriveDomainRealizationContract(program, domain, state_role_by_name));
  }
  return contracts;
}

void AppendPipelineResourceIntent(const std::string& member_func, const SemanticProgram& program,
                                  Array<ResourceIntent>* resource_intents) {
  auto pipeline_stages = GetPipelineStagesFromSupplements(program);
  if (!pipeline_stages.has_value() || pipeline_stages->empty()) {
    return;
  }
  Map<String, Any> payload = BuildMemberFuncTargetPayload();
  payload.Set(String(schema_key::kPipelineStages), pipeline_stages.value());
  resource_intents->push_back(ResourceIntent(
      String("pipeline_contract_" + member_func),
      String(sp::ToString(sp::SpatialResourceIntentKind::kSynchronizationSupport)),
      String(member_func), MakeTraits({"phase_b", "pipeline_contract"}), std::move(payload),
      MakeAnchors("spatial_resource_intent", "pipeline_contract_" + member_func)));
}

void AppendFragmentResourceIntent(const std::string& member_func, const SemanticProgram& program,
                                  Array<ResourceIntent>* resource_intents) {
  auto fragment_payload = GetFragmentLoweringPayloadFromSupplements(program);
  if (!fragment_payload.has_value()) {
    return;
  }
  Map<String, Any> payload = fragment_payload.value();
  payload.Set(String(schema_key::kTargetKind), String(spatial_contract::kMemberFuncTarget));
  resource_intents->push_back(ResourceIntent(
      String("fragment_contract_" + member_func),
      String(sp::ToString(sp::SpatialResourceIntentKind::kLoweringSupport)),
      String(member_func), MakeTraits({"phase_b", "fragment_contract"}), payload,
      MakeAnchors("spatial_resource_intent", "fragment_contract_" + member_func)));
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

std::vector<std::string> CollectSegmentKindsFromBody(const tir::Stmt& body) {
  class SegmentKindCollector : public tir::StmtVisitor {
   public:
    void VisitStmt_(const tir::AttrStmtNode* op) final {
      if (op->attr_key == "blackhole.segment_kind") {
        if (const auto* kind = op->value.as<tir::StringImmNode>()) {
          const std::string segment_kind = kind->value;
          if (seen_.insert(segment_kind).second) {
            segment_kinds_.push_back(segment_kind);
          }
        }
      }
      tir::StmtVisitor::VisitStmt_(op);
    }

    const std::vector<std::string>& segment_kinds() const { return segment_kinds_; }

   private:
    std::unordered_set<std::string> seen_;
    std::vector<std::string> segment_kinds_;
  };

  SegmentKindCollector collector;
  collector(body);
  return collector.segment_kinds();
}

bool HasSimpleSegmentKinds(const tir::PrimFunc& func, const std::vector<std::string>& expected_kinds) {
  return CollectSegmentKindsFromBody(func->body) == expected_kinds;
}

bool IsSimpleCopyFastPath(const SemanticProgram& program, const tir::PrimFunc& func) {
  if (HasSimpleSegmentKinds(func, {"reader", "compute", "writer"})) {
    return false;
  }
  if (!program->states.empty() || program->updates.size() != 1) {
    return false;
  }
  auto kind = ParseUpdateLawKind(str(program->updates[0]->law->kind));
  return kind && *kind == UpdateLawKind::kMap;
}

bool IsSimpleGemmFastPath(const SemanticProgram& program, const tir::PrimFunc& func) {
  if (!HasSimpleSegmentKinds(func, {"reader", "compute", "writer"})) {
    return false;
  }
  if (program->updates.size() != 1 || program->states.size() > 1) {
    return false;
  }
  if (program->states.size() == 1) {
    auto role = ParseStateRole(str(program->states[0]->role));
    if (!role || *role != StateRole::kTransient) {
      return false;
    }
  }
  return true;
}

void BuildCommonSpatialScaffolding(const std::string& member_func, const SemanticProgram& program,
                                   const SpatialCapabilityModel& capability_model,
                                   Array<SpatialLayout>* layouts,
                                   Array<WorkPartition>* work_partitions) {
  const std::vector<DomainRealizationContract> domain_contracts =
      DeriveDomainRealizationContracts(program);
  const bool multi_domain = program->domains.size() > 1;
  for (int domain_index = 0; domain_index < program->domains.size(); ++domain_index) {
    const Domain& domain = program->domains[domain_index];
    const DomainRealizationContract& domain_contract = domain_contracts[domain_index];
    const char* layout_kind = SelectLayoutKind(capability_model, domain_contract);
    const char* partition_kind = SelectPartitionKind(capability_model, domain_contract);
    const std::string domain_suffix =
        multi_domain ? "_" + str(domain->name) : std::string("");
    layouts->push_back(SpatialLayout(
        String("layout_" + member_func + domain_suffix), String(layout_kind), String(member_func),
        domain->axes, MakeTraits({"phase_b"}), BuildDomainPayload(domain_index, domain_contract),
        MakeAnchors("spatial_layout", member_func + domain_suffix)));
    work_partitions->push_back(WorkPartition(
        String("partition_" + member_func + domain_suffix), String(partition_kind),
        String(member_func), domain->axes, MakeTraits({"phase_b"}),
        BuildWorkPartitionPayload(program, domain_index, domain_contract),
        MakeAnchors("spatial_partition", member_func + domain_suffix)));
  }
}

std::string BuildPhaseClosureBasis(const std::string& phase_name,
                                   const std::vector<int>& task_indices,
                                   const std::vector<ChannelSynthesisRecord>& channel_records,
                                   const std::vector<TaskSynthesisRecord>& task_records) {
  int internal_edges = 0;
  int incoming_edges = 0;
  int outgoing_edges = 0;
  std::vector<std::string> closure_bases;
  for (const ChannelSynthesisRecord& record : channel_records) {
    const std::string& source_phase = task_records[record.source_task_index].phase_name;
    const std::string& target_phase = task_records[record.target_task_index].phase_name;
    if (source_phase == phase_name && target_phase == phase_name) {
      ++internal_edges;
    } else if (target_phase == phase_name) {
      ++incoming_edges;
    } else if (source_phase == phase_name) {
      ++outgoing_edges;
    }
    if (source_phase == phase_name || target_phase == phase_name) {
      PushBackUnique(&closure_bases, DeriveOrderingKindForChannelRecord(record));
    }
  }
  std::string basis_summary;
  for (int i = 0; i < static_cast<int>(closure_bases.size()); ++i) {
    if (i != 0) {
      basis_summary += ",";
    }
    basis_summary += closure_bases[i];
  }
  return "task_graph_closure|phase=" + phase_name + "|task_count=" +
         std::to_string(task_indices.size()) + "|internal_edges=" +
         std::to_string(internal_edges) + "|incoming_edges=" +
         std::to_string(incoming_edges) + "|outgoing_edges=" +
         std::to_string(outgoing_edges) + "|ordering_basis=" + basis_summary;
}

PhaseSynthesisResult SynthesizeProgramPhases(std::vector<TaskSynthesisRecord>* task_records,
                                             const std::vector<ChannelSynthesisRecord>& channel_records) {
  ICHECK(task_records != nullptr);
  std::vector<int> phase_rank;
  phase_rank.reserve(task_records->size());
  for (const TaskSynthesisRecord& record : *task_records) {
    phase_rank.push_back(record.phase_rank);
  }
  std::unordered_set<std::string> seen_ordering_edges;
  std::vector<std::vector<int>> successors(task_records->size());
  std::vector<int> indegree(task_records->size(), 0);
  for (const ChannelSynthesisRecord& record : channel_records) {
    if (!record.ordering_critical || record.source_task_index == record.target_task_index) {
      continue;
    }
    const std::string edge_key = std::to_string(record.source_task_index) + "->" +
                                 std::to_string(record.target_task_index);
    if (!seen_ordering_edges.insert(edge_key).second) {
      continue;
    }
    successors[record.source_task_index].push_back(record.target_task_index);
    ++indegree[record.target_task_index];
  }

  std::vector<int> ready;
  ready.reserve(task_records->size());
  for (int task_index = 0; task_index < static_cast<int>(task_records->size()); ++task_index) {
    if (indegree[task_index] == 0) {
      ready.push_back(task_index);
    }
  }
  std::sort(ready.begin(), ready.end());

  int processed = 0;
  while (!ready.empty()) {
    const int task_index = ready.front();
    ready.erase(ready.begin());
    ++processed;
    for (int successor : successors[task_index]) {
      phase_rank[successor] = std::max(phase_rank[successor], phase_rank[task_index] + 1);
      --indegree[successor];
      if (indegree[successor] == 0) {
        ready.push_back(successor);
        std::sort(ready.begin(), ready.end());
      }
    }
  }
  ICHECK_EQ(processed, static_cast<int>(task_records->size()))
      << "LowerToSpatialProgram requires ordering-critical task graph to be acyclic";

  std::unordered_map<int, int> compact_rank_by_rank;
  for (int rank : phase_rank) {
    if (!compact_rank_by_rank.count(rank)) {
      compact_rank_by_rank[rank] = static_cast<int>(compact_rank_by_rank.size());
    }
  }

  PhaseSynthesisResult result;
  for (int i = 0; i < static_cast<int>(task_records->size()); ++i) {
    TaskSynthesisRecord& record = (*task_records)[i];
    record.phase_rank = compact_rank_by_rank.at(phase_rank[i]);
    record.phase_name = "phase_" + std::to_string(record.phase_rank);
    if (!result.phase_index_by_name.count(record.phase_name)) {
      result.phase_index_by_name[record.phase_name] = record.phase_rank;
      result.phase_order.push_back(record.phase_name);
    }
  }
  std::stable_sort(result.phase_order.begin(), result.phase_order.end(),
                   [&](const std::string& lhs, const std::string& rhs) {
                     return result.phase_index_by_name.at(lhs) <
                            result.phase_index_by_name.at(rhs);
                   });
  return result;
}

std::vector<PhaseSyncSynthesisRecord> BuildPhaseSyncRecords(
    const std::unordered_map<std::string, int>& phase_index_by_name,
    const std::vector<TaskSynthesisRecord>& task_records,
    const std::vector<ChannelSynthesisRecord>& channel_records) {
  std::vector<PhaseSyncSynthesisRecord> records;
  std::unordered_set<std::string> seen_phase_edges;
  auto add_record = [&](const std::string& source_phase_name, const std::string& target_phase_name,
                        int source_task_index, int target_task_index,
                        const std::string& ordering_kind,
                        const std::string& materialization_kind) {
    if (source_phase_name == target_phase_name || source_task_index < 0 || target_task_index < 0) {
      return;
    }
    std::string key = source_phase_name + "->" + target_phase_name + "|" + ordering_kind + "|" +
                      materialization_kind;
    if (!seen_phase_edges.insert(key).second) {
      return;
    }
    records.push_back(PhaseSyncSynthesisRecord{
        source_phase_name,
        target_phase_name,
        source_task_index,
        target_task_index,
        ordering_kind,
        materialization_kind,
    });
  };

  for (const ChannelSynthesisRecord& record : channel_records) {
    if (!IsOrderingCriticalChannel(record)) {
      continue;
    }
    const std::string& source_phase_name = task_records[record.source_task_index].phase_name;
    const std::string& target_phase_name = task_records[record.target_task_index].phase_name;
    if (source_phase_name == target_phase_name) {
      continue;
    }
    add_record(source_phase_name, target_phase_name, record.source_task_index,
               record.target_task_index, DeriveOrderingKindForChannelRecord(record),
               DeriveMaterializationKindForChannelRecord(record));
  }

  std::stable_sort(records.begin(), records.end(),
                   [&](const PhaseSyncSynthesisRecord& lhs,
                       const PhaseSyncSynthesisRecord& rhs) {
                     if (lhs.source_phase_name != rhs.source_phase_name) {
                       return phase_index_by_name.at(lhs.source_phase_name) <
                              phase_index_by_name.at(rhs.source_phase_name);
                     }
                     return phase_index_by_name.at(lhs.target_phase_name) <
                            phase_index_by_name.at(rhs.target_phase_name);
                   });
  return records;
}

SpatialProgramBundle BuildCopyFastPath(const std::string& member_func,
                                       const SemanticProgram& program,
                                       const SpatialCapabilityModel& capability_model) {
  const char* task_kind = sp::ToString(sp::SpatialTaskKind::kTransfer);
  const char* channel_kind =
      SelectChannelKind(capability_model, sp::SpatialChannelKind::kPointToPoint);
  const char* payload_kind =
      SelectChannelPayloadKind(capability_model, sp::SpatialChannelPayloadKind::kTensor);
  const char* delivery_kind =
      SelectChannelDeliveryKind(capability_model, sp::SpatialChannelDeliveryKind::kBufferedAsync);
  TaskSynthesisRecord copy_record;
  copy_record.task_name = "copy";
  copy_record.task_kind = task_kind;
  copy_record.phase_name = "phase0_copy";
  copy_record.execution_role = "transfer_copy";
  copy_record.formation_basis = "fast_path|segment=copy|boundary=tensor_transfer";
  copy_record.update_names = {str(program->updates[0]->name)};
  copy_record.task_traits = {"fast_path", "copy"};
  Array<Task> tasks{
      Task(String("copy"), String(task_kind), String("phase0_copy"),
           Array<String>{String(program->updates[0]->name)}, ToStringArray(copy_record.task_traits),
           BuildTaskPayload(0, copy_record),
           MakeAnchors("spatial_task", "copy"))};
  Array<Channel> channels{
      Channel(String("copy_tensor"), String(channel_kind), String("copy"), String("copy"), String(),
              MakeTraits({"fast_path", "copy"}),
              BuildChannelPayload(0, 0, payload_kind, delivery_kind),
              MakeAnchors("spatial_channel", "copy_tensor"))};
  Array<ProgramPhase> phases{
      ProgramPhase(String("phase0_copy"), Array<String>{String("copy")},
                   Array<String>{String("copy_tensor")}, MakeTraits({"fast_path", "copy"}),
                   BuildProgramPhasePayload(0, {0}, {0},
                                            "single_phase_fast_path|task_graph_closure"),
                   MakeAnchors("spatial_phase", "phase0_copy"))};
  Array<SpatialLayout> layouts;
  Array<WorkPartition> work_partitions;
  Array<Placement> placements{
      Placement(String("place_copy"),
                String(sp::ToString(sp::SpatialPlacementKind::kExecution)), String("copy"),
                String(member_func), MakeTraits({"fast_path", "copy"}),
                BuildPlacementPayload(0, "ingress", "execution",
                                     str(capability_model->placement_domain).c_str()),
                MakeAnchors("spatial_placement", "copy"))};
  Array<SyncEdge> sync_edges;
  Array<ResourceIntent> resource_intents{
      ResourceIntent(String("copy_buffer"),
                     String(sp::ToString(sp::SpatialResourceIntentKind::kBuffer)), String("copy"),
                     MakeTraits({"fast_path", "copy"}), EmptyPayload(),
                     MakeAnchors("spatial_resource_intent", "copy_buffer"))};
  AppendFragmentResourceIntent(member_func, program, &resource_intents);
  AppendPipelineResourceIntent(member_func, program, &resource_intents);
  BuildCommonSpatialScaffolding(member_func, program, capability_model, &layouts,
                                &work_partitions);
  return {SpatialProgram(String(member_func), phases, tasks, channels, layouts,
                         work_partitions, placements, sync_edges, resource_intents,
                         MakeAnchors("spatial_program", member_func)),
          phases};
}

SpatialProgramBundle BuildGemmFastPath(const std::string& member_func,
                                       const SemanticProgram& program,
                                       const tir::PrimFunc& func,
                                       const SpatialCapabilityModel& capability_model) {
  Array<Task> tasks;
  Array<Placement> placements;
  Array<String> task_names;
  for (const std::string& segment_name : CollectSegmentKindsFromBody(func->body)) {
    const char* task_kind = segment_name == "compute"
                                ? sp::ToString(sp::SpatialTaskKind::kCompute)
                                : sp::ToString(sp::SpatialTaskKind::kTransfer);
    Array<String> update_names;
    if (!program->updates.empty()) {
      update_names.push_back(program->updates[0]->name);
    }
    const std::string execution_role =
        segment_name == "reader" ? "tile_ingress"
        : segment_name == "writer" ? "tile_egress"
                                   : "gemm_compute";
    const std::string formation_basis =
        segment_name == "reader" ? "fast_path|segment=reader|boundary=tensor_transfer"
        : segment_name == "writer" ? "fast_path|segment=writer|boundary=completion_handoff"
                                   : "fast_path|segment=compute|boundary=accumulator_update";
    TaskSynthesisRecord record;
    record.task_name = segment_name;
    record.task_kind = task_kind;
    record.phase_name = "phase0_gemm";
    record.execution_role = execution_role;
    record.formation_basis = formation_basis;
    record.update_names = program->updates.empty()
                              ? std::vector<std::string>{}
                              : std::vector<std::string>{str(program->updates[0]->name)};
    record.task_traits = {"fast_path", "gemm"};
    tasks.push_back(Task(String(segment_name), String(task_kind), String("phase0_gemm"),
                         update_names, ToStringArray(record.task_traits),
                         BuildTaskPayload(0, record),
                         MakeAnchors("spatial_task", segment_name)));
    task_names.push_back(String(segment_name));
    const char* placement_kind = sp::ToString(sp::SpatialPlacementKind::kExecution);
    placements.push_back(Placement(String("place_" + segment_name), String(placement_kind),
                                   String(segment_name), String(member_func),
                                   MakeTraits({"fast_path", "gemm"}),
                                   BuildPlacementPayload(
                                       task_names.size() - 1,
                                       NeutralPlacementAffinityForSegmentKind(segment_name),
                                       "execution",
                                       str(capability_model->placement_domain).c_str()),
                                   MakeAnchors("spatial_placement", segment_name)));
  }
  const char* tensor_flow =
      SelectChannelKind(capability_model, sp::SpatialChannelKind::kPointToPoint);
  const char* tensor_payload =
      SelectChannelPayloadKind(capability_model, sp::SpatialChannelPayloadKind::kTensor);
  const char* state_payload =
      SelectChannelPayloadKind(capability_model, sp::SpatialChannelPayloadKind::kStateVersion);
  const char* buffered_async =
      SelectChannelDeliveryKind(capability_model, sp::SpatialChannelDeliveryKind::kBufferedAsync);
  const char* completion_visible = SelectChannelDeliveryKind(
      capability_model, sp::SpatialChannelDeliveryKind::kCompletionVisible);
  const char* output_payload = program->states.empty() ? tensor_payload : state_payload;
  std::optional<std::string> output_version;
  if (!program->states.empty()) {
    const auto update_result_version_by_update =
        BuildUniqueUpdateResultVersionByUpdate(program->state_defs);
    if (!program->updates.empty()) {
      auto version_it = update_result_version_by_update.find(str(program->updates[0]->name));
      ICHECK(version_it != update_result_version_by_update.end())
          << "LowerToSpatialProgram requires GEMM fast path state payload to have an "
             "update-result version";
      output_version = version_it->second;
    }
  }
  Map<String, Any> c_tiles_payload =
      program->states.empty()
          ? BuildChannelPayload(1, 2, output_payload, completion_visible)
          : BuildChannelPayload(1, 2, output_payload, completion_visible, 0);
  if (output_version.has_value()) {
    c_tiles_payload.Set(String(schema_key::kSourceVersion), String(output_version.value()));
    c_tiles_payload.Set(String(schema_key::kTargetVersion), String(output_version.value()));
  }
  Array<Channel> channels{
      Channel(String("a_tiles"), String(tensor_flow), String("reader"), String("compute"),
              String("A"), MakeTraits({"fast_path", "gemm"}),
              BuildChannelPayload(0, 1, tensor_payload, buffered_async),
              MakeAnchors("spatial_channel", "a_tiles")),
      Channel(String("b_tiles"), String(tensor_flow), String("reader"), String("compute"),
              String("B"), MakeTraits({"fast_path", "gemm"}),
              BuildChannelPayload(0, 1, tensor_payload, buffered_async),
              MakeAnchors("spatial_channel", "b_tiles")),
      Channel(String("c_tiles"), String(tensor_flow), String("compute"), String("writer"),
              String(program->states.empty() ? "" : str(program->states[0]->name)),
              MakeTraits({"fast_path", "gemm"}),
              c_tiles_payload,
              MakeAnchors("spatial_channel", "c_tiles"))};
  Array<ProgramPhase> phases{
      ProgramPhase(String("phase0_gemm"), task_names,
                   Array<String>{String("a_tiles"), String("b_tiles"), String("c_tiles")},
                   MakeTraits({"fast_path", "gemm"}),
                   BuildProgramPhasePayload(0, {0, 1, 2}, {0, 1, 2},
                                            "segment_graph_closure|single_phase_fast_path"),
                   MakeAnchors("spatial_phase", "phase0_gemm"))};
  Array<SpatialLayout> layouts;
  Array<WorkPartition> work_partitions;
  const char* dep_kind = sp::ToString(sp::SpatialSyncKind::kDependency);
  Array<SyncEdge> sync_edges{
      SyncEdge(String("reader_to_compute"), String(dep_kind), String("reader"),
               String("compute"), MakeTraits({"fast_path", "gemm"}),
               BuildSyncEdgePayload(0, 1, "must_happen_before", "buffer_visibility"),
               MakeAnchors("spatial_sync", "reader_to_compute")),
      SyncEdge(String("compute_to_writer"), String(dep_kind), String("compute"),
               String("writer"), MakeTraits({"fast_path", "gemm"}),
               BuildSyncEdgePayload(1, 2, "must_happen_before", "completion_visibility"),
               MakeAnchors("spatial_sync", "compute_to_writer"))};
  Array<ResourceIntent> resource_intents{
      ResourceIntent(String("gemm_input_buffers"),
                     String(sp::ToString(sp::SpatialResourceIntentKind::kBuffer)),
                     String("reader"),
                     MakeTraits({"fast_path", "gemm"}), EmptyPayload(),
                     MakeAnchors("spatial_resource_intent", "gemm_input_buffers"))};
  if (!program->states.empty()) {
    resource_intents.push_back(ResourceIntent(
        String("gemm_accumulator"),
        String(sp::ToString(sp::SpatialResourceIntentKind::kStateResidency)),
        String(str(program->states[0]->name)), MakeTraits({"fast_path", "gemm"}),
        BuildTargetPayload(spatial_contract::kSemanticStateTarget, 0),
        MakeAnchors("spatial_resource_intent", "gemm_accumulator")));
  }
  AppendFragmentResourceIntent(member_func, program, &resource_intents);
  AppendPipelineResourceIntent(member_func, program, &resource_intents);
  BuildCommonSpatialScaffolding(member_func, program, capability_model, &layouts,
                                &work_partitions);
  return {SpatialProgram(String(member_func), phases, tasks, channels, layouts,
                         work_partitions, placements, sync_edges, resource_intents,
                         MakeAnchors("spatial_program", member_func)),
          phases};
}

SpatialProgramBundle BuildGenericSpatialProgram(const std::string& member_func,
                                                const SemanticProgram& program,
                                                const SpatialCapabilityModel& capability_model) {
  std::unordered_map<std::string, int> state_index_by_name;
  std::unordered_map<std::string, std::optional<StateRole>> state_role_by_name =
      BuildStateRoleByName(program);
  for (int i = 0; i < program->states.size(); ++i) {
    const std::string state_name = str(program->states[i]->name);
    state_index_by_name[state_name] = i;
  }

  const std::vector<DomainRealizationContract> domain_contracts =
      DeriveDomainRealizationContracts(program);
  TaskFormationResult task_formation =
      BuildTaskRecords(program, state_role_by_name, domain_contracts, capability_model);
  std::vector<TaskSynthesisRecord> task_records = task_formation.task_records;
  std::unordered_map<std::string, int> task_index_by_update_name;
  for (int i = 0; i < static_cast<int>(task_records.size()); ++i) {
    for (const std::string& record_update_name : task_records[i].update_names) {
      task_index_by_update_name[record_update_name] = i;
    }
  }

  std::vector<ChannelSynthesisRecord> channel_records;
  const std::unordered_set<std::string> known_updates = CollectKnownUpdateNames(program);
  const auto version_to_producer_edges = BuildVersionProducerEdges(program, known_updates);
  const auto join_by_output_version = BuildStateJoinByOutputVersion(program->state_joins);
  const auto update_result_version_by_update =
      BuildUniqueUpdateResultVersionByUpdate(program->state_defs);
  const auto version_consumer_count_by_version =
      BuildDistinctConsumerCountByVersion(program->state_uses);
  std::unordered_set<std::string> seen_channel_keys;
  for (const StateUse& use : program->state_uses) {
    const std::string consumer_update = str(use->consumer_update);
    const std::string version_name = str(use->version_name);
    if (!task_index_by_update_name.count(consumer_update)) {
      continue;
    }
    const std::string state_name = str(use->state_name);
    auto state_index_it = state_index_by_name.find(state_name);
    ICHECK(state_index_it != state_index_by_name.end())
        << "LowerToSpatialProgram requires StateUse.state_name to reference a known state";
    auto consumer_target_version_it = update_result_version_by_update.find(consumer_update);
    ICHECK(consumer_target_version_it != update_result_version_by_update.end())
        << "LowerToSpatialProgram could not derive a unique consumer target_version for update "
        << consumer_update;
    auto producer_edges_it = version_to_producer_edges.find(version_name);
    std::optional<StateJoinKind> join_kind;
    auto join_it = join_by_output_version.find(version_name);
    if (join_it != join_by_output_version.end()) {
      join_kind = ParseStateJoinKind(str(join_it->second->kind));
    }
    if (producer_edges_it == version_to_producer_edges.end()) {
      ICHECK(!join_kind.has_value())
          << "LowerToSpatialProgram missing producer task linkage for join output version "
          << version_name;
      continue;
    }
    const auto maybe_role = state_role_by_name.count(state_name) ? state_role_by_name.at(state_name)
                                                                 : std::optional<StateRole>();
    for (const ProducerVersionEdge& producer_edge : producer_edges_it->second) {
      if (!task_index_by_update_name.count(producer_edge.producer_update)) {
        continue;
      }
      std::string channel_key =
          producer_edge.producer_update + "|" + consumer_update + "|" + state_name + "|" +
          version_name + "|" + consumer_target_version_it->second + "|" + str(use->kind);
      if (!seen_channel_keys.insert(channel_key).second) {
        continue;
      }
      const int source_task_index = task_index_by_update_name.at(producer_edge.producer_update);
      const int target_task_index = task_index_by_update_name.at(consumer_update);
      const int version_consumer_count =
          version_consumer_count_by_version.count(version_name)
              ? version_consumer_count_by_version.at(version_name)
              : 0;
      const sp::SpatialChannelKind flow_kind =
          DeriveStateVersionBaseFlowKind(use, maybe_role, join_kind,
                                         version_consumer_count);
      const std::string channel_name =
          "channel_" + state_name + "_" + version_name + "_" + str(use->kind) + "_" +
          producer_edge.producer_update + "_to_" + consumer_update;
      channel_records.push_back(ChannelSynthesisRecord{
          channel_name,
          flow_kind,
          sp::SpatialChannelPayloadKind::kStateVersion,
          DeriveStateVersionDeliveryKind(flow_kind, false),
          IsOrderingCriticalFlowKind(flow_kind),
          source_task_index,
          target_task_index,
          state_index_it->second,
          version_name,
          consumer_target_version_it->second,
      });
    }
  }

  PhaseSynthesisResult phase_synthesis = SynthesizeProgramPhases(&task_records, channel_records);
  const std::vector<std::string>& phase_order = phase_synthesis.phase_order;
  ICHECK(!phase_order.empty())
      << "LowerToSpatialProgram requires at least one synthesized program phase";
  const bool multi_phase = phase_order.size() > 1;
  const std::unordered_map<std::string, int>& phase_index_by_name =
      phase_synthesis.phase_index_by_name;
  for (ChannelSynthesisRecord& record : channel_records) {
    const bool cross_phase =
        task_records[record.source_task_index].phase_name !=
        task_records[record.target_task_index].phase_name;
    record.delivery_kind = DeriveStateVersionDeliveryKind(record.flow_kind, cross_phase);
    record.ordering_critical =
        record.ordering_critical ||
        record.delivery_kind == sp::SpatialChannelDeliveryKind::kPhaseBoundaryMaterialized;
  }

  std::unordered_map<std::string, int> task_index_by_name;
  Array<Task> tasks;
  Array<Placement> placements;
  std::unordered_map<std::string, std::vector<std::string>> phase_to_tasks;
  for (int i = 0; i < static_cast<int>(task_records.size()); ++i) {
    const TaskSynthesisRecord& record = task_records[i];
    task_index_by_name[record.task_name] = i;
    Array<String> update_names = ToStringArray(record.update_names);
    Array<String> task_traits = ToStringArray(record.task_traits);
    Task task(String(record.task_name), String(record.task_kind), String(record.phase_name),
              update_names, task_traits,
              BuildTaskPayload(phase_index_by_name.at(record.phase_name), record),
              MakeAnchors("spatial_task", record.task_name));
    tasks.push_back(task);
    const char* placement_kind = sp::ToString(sp::SpatialPlacementKind::kExecution);
    placements.push_back(
        Placement(String("place_" + record.task_name), String(placement_kind),
                  String(record.task_name), String(member_func), MakeTraits({"phase_b"}),
                  BuildPlacementPayload(i, NeutralPlacementAffinityForTask(task), "execution",
                                       str(capability_model->placement_domain).c_str()),
                  MakeAnchors("spatial_placement", record.task_name)));
    phase_to_tasks[record.phase_name].push_back(record.task_name);
  }

  std::unordered_map<std::string, std::vector<std::string>> phase_to_channels;
  Array<Channel> channels;
  std::unordered_map<std::string, int> channel_index_by_name;
  for (const ChannelSynthesisRecord& record : channel_records) {
    const char* channel_kind = SelectChannelKind(capability_model, record.flow_kind);
    const char* payload_kind = SelectChannelPayloadKind(capability_model, record.payload_kind);
    const char* delivery_kind = SelectChannelDeliveryKind(capability_model, record.delivery_kind);
    const std::string& source_task_name =
        task_records[record.source_task_index].task_name;
    const std::string& target_task_name =
        task_records[record.target_task_index].task_name;
    const std::string state_name =
        record.state_index.has_value() ? str(program->states[record.state_index.value()]->name) : "";
    const bool cross_phase =
        task_records[record.source_task_index].phase_name !=
        task_records[record.target_task_index].phase_name;
    Array<String> channel_traits =
        cross_phase ? MakeTraits({"phase_b", "phase_boundary"}) : MakeTraits({"phase_b"});
    channels.push_back(Channel(
        String(record.channel_name), String(channel_kind), String(source_task_name),
        String(target_task_name), String(state_name), channel_traits,
        BuildChannelPayload(record, payload_kind, delivery_kind),
        MakeAnchors("spatial_channel", record.channel_name)));
    const int channel_index = static_cast<int>(channels.size()) - 1;
    channel_index_by_name[record.channel_name] = channel_index;
    phase_to_channels[task_records[record.target_task_index].phase_name].push_back(
        record.channel_name);
  }

  Array<ProgramPhase> phases;
  for (const auto& phase_name : phase_order) {
    auto task_it = phase_to_tasks.find(phase_name);
    if (task_it == phase_to_tasks.end() || task_it->second.empty()) {
      continue;
    }
    std::vector<int> task_indices;
    for (const auto& task_name : task_it->second) {
      task_indices.push_back(task_index_by_name.at(task_name));
    }
    auto channel_it = phase_to_channels.find(phase_name);
    std::vector<int> channel_indices;
    if (channel_it != phase_to_channels.end()) {
      for (const auto& channel_name : channel_it->second) {
        channel_indices.push_back(channel_index_by_name.at(channel_name));
      }
    }
    phases.push_back(ProgramPhase(String(phase_name), ToStringArray(task_it->second),
                                  channel_it == phase_to_channels.end()
                                      ? Array<String>{}
                                      : ToStringArray(channel_it->second),
                                  ToStringArray(multi_phase ? std::vector<std::string>{"phase_b", "multi_phase"}
                                                            : std::vector<std::string>{"phase_b"}),
                                  BuildProgramPhasePayload(phase_index_by_name.at(phase_name),
                                                          task_indices, channel_indices,
                                                          BuildPhaseClosureBasis(
                                                              phase_name, task_indices,
                                                              channel_records, task_records)),
                                  MakeAnchors("spatial_phase", phase_name)));
  }

  Array<SpatialLayout> layouts;
  Array<WorkPartition> work_partitions;
  BuildCommonSpatialScaffolding(member_func, program, capability_model, &layouts,
                                &work_partitions);

  Array<SyncEdge> sync_edges;
  if (multi_phase) {
    const std::vector<PhaseSyncSynthesisRecord> phase_sync_records = BuildPhaseSyncRecords(
        phase_index_by_name, task_records, channel_records);
    for (const PhaseSyncSynthesisRecord& record : phase_sync_records) {
      const std::string sync_name = "sync_" + record.source_phase_name + "_to_" +
                                    record.target_phase_name;
      sync_edges.push_back(SyncEdge(
          String(sync_name), String(sp::ToString(sp::SpatialSyncKind::kCompletion)),
          String(task_records[record.source_task_index].task_name),
          String(task_records[record.target_task_index].task_name),
          MakeTraits({"phase_boundary", "graph_ordered"}),
          BuildSyncEdgePayload(record.source_task_index, record.target_task_index,
                               record.ordering_kind, record.materialization_kind),
          MakeAnchors("spatial_sync", sync_name)));
    }
  }

  std::unordered_set<int> phase_boundary_state_indices;
  for (const ChannelSynthesisRecord& record : channel_records) {
    if (record.delivery_kind != sp::SpatialChannelDeliveryKind::kPhaseBoundaryMaterialized ||
        !record.state_index.has_value()) {
      continue;
    }
    const std::string state_name = str(program->states[record.state_index.value()]->name);
    auto role_it = state_role_by_name.find(state_name);
    if (role_it != state_role_by_name.end() && IsStatefulRole(role_it->second)) {
      phase_boundary_state_indices.insert(record.state_index.value());
    }
  }

  Array<ResourceIntent> resource_intents;
  for (const State& state : program->states) {
    const auto role = ParseStateRole(str(state->role));
    const std::string state_name = str(state->name);
    auto state_index_it = state_index_by_name.find(state_name);
    ICHECK(state_index_it != state_index_by_name.end())
        << "LowerToSpatialProgram requires semantic state indices for state-targeted intents";
    const bool is_stateful = role && (*role == StateRole::kCarry ||
                                      *role == StateRole::kReductionAccumulator ||
                                      *role == StateRole::kSelectionState ||
                                      *role == StateRole::kIndexState);
    const char* intent_kind = is_stateful
        ? sp::ToString(sp::SpatialResourceIntentKind::kStateResidency)
        : sp::ToString(sp::SpatialResourceIntentKind::kBuffer);
    resource_intents.push_back(ResourceIntent(
        String("intent_" + state_name), String(intent_kind), state->name,
        Array<String>{String(str(state->role)),
                      String(str(state->storage_scope))},
        BuildTargetPayload(spatial_contract::kSemanticStateTarget, state_index_it->second),
        MakeAnchors("spatial_resource_intent", state_name)));
    if (phase_boundary_state_indices.count(state_index_it->second)) {
      resource_intents.push_back(ResourceIntent(
          String("phase_boundary_" + state_name),
          String(sp::ToString(sp::SpatialResourceIntentKind::kPhaseBoundaryMaterialization)),
          state->name, MakeTraits({"phase_boundary"}),
          BuildTargetPayload(spatial_contract::kSemanticStateTarget, state_index_it->second),
          MakeAnchors("spatial_resource_intent", "phase_boundary_" + state_name)));
    }
  }
  AppendFragmentResourceIntent(member_func, program, &resource_intents);
  AppendPipelineResourceIntent(member_func, program, &resource_intents);

  return {SpatialProgram(String(member_func), phases, tasks, channels, layouts,
                         work_partitions, placements, sync_edges, resource_intents,
                         MakeAnchors("spatial_program", member_func)),
          phases};
}

SpatialProgramBundle BuildSpatialProgramForFunc(const std::string& member_func,
                                                const SemanticProgram& program,
                                                const tir::PrimFunc& func,
                                                const SpatialCapabilityModel& capability_model) {
  if (IsSimpleCopyFastPath(program, func)) {
    return BuildCopyFastPath(member_func, program, capability_model);
  }
  if (IsSimpleGemmFastPath(program, func)) {
    return BuildGemmFastPath(member_func, program, func, capability_model);
  }
  return BuildGenericSpatialProgram(member_func, program, capability_model);
}

SpatialExecutionPlan BuildSpatialExecutionPlanForFunc(const std::string& member_func,
                                                      const SemanticProgram& program,
                                                      const tir::PrimFunc& func,
                                                      const SpatialCapabilityModel& capability_model) {
  const SpatialProgramBundle bundle =
      BuildSpatialProgramForFunc(member_func, program, func, capability_model);
  return SpatialExecutionPlan(String(member_func), bundle.program->phases, bundle.program->tasks,
                              bundle.program->channels, bundle.program->placements,
                              bundle.program->sync_edges, bundle.program->resource_intents,
                              MakeAnchors("spatial_execution_plan", member_func));
}

}  // namespace

tvm::transform::Pass AnalyzeSpatialExecutionPlan() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext) {
    IRModule updates = IRModule(Map<GlobalVar, BaseFunc>({}));
    for (const auto& [gvar, base_func] : mod->functions) {
      auto func = base_func.as<tir::PrimFunc>();
      if (!func || !IsBlackholePrimFunc(func.value())) {
        continue;
      }
      auto maybe_semantic = func.value()->GetAttr<SemanticProgram>(attr::kTLSemanticProgram);
      if (!maybe_semantic) {
        continue;
      }
      ICHECK(func.value()->GetAttr<SpatialDomainPlan>(attr::kTLSpatialDomainPlan))
          << "AnalyzeSpatialExecutionPlan requires AnalyzeSpatialDomainPlan to run first";
      auto maybe_target = func.value()->GetAttr<Target>(tvm::attr::kTarget);
      ICHECK(maybe_target)
          << "AnalyzeSpatialExecutionPlan requires blackhole PrimFunc target to derive capability";
      const TTHardwareModel hardware_model = BuildBlackholeTTHardwareModel(maybe_target.value());
      const SpatialCapabilityModel capability_model = DeriveSpatialCapabilityModel(hardware_model);
      const std::string member_func = GetMemberFuncName(gvar, func.value());
      const SpatialExecutionPlan execution_plan =
          BuildSpatialExecutionPlanForFunc(member_func, maybe_semantic.value(), func.value(),
                                           capability_model);
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

tvm::transform::Pass MaterializeSpatialProgram() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext) {
    IRModule updates = IRModule(Map<GlobalVar, BaseFunc>({}));
    std::unordered_map<std::string, Array<ProgramPhase>> phases_by_member_func;
    std::optional<TTHardwareModel> hardware_model;
    std::optional<SpatialCapabilityModel> capability_model;

    auto ensure_models = [&](const tir::PrimFunc& func) {
      if (hardware_model && capability_model) {
        return;
      }
      auto maybe_target = func->GetAttr<Target>(tvm::attr::kTarget);
      ICHECK(maybe_target)
          << "MaterializeSpatialProgram requires blackhole PrimFunc target to derive "
             "TTHardwareModel/SpatialCapabilityModel";
      hardware_model = BuildBlackholeTTHardwareModel(maybe_target.value());
      capability_model = DeriveSpatialCapabilityModel(hardware_model.value());
    };

    for (const auto& [gvar, base_func] : mod->functions) {
      auto func = base_func.as<tir::PrimFunc>();
      if (!func || !IsBlackholePrimFunc(func.value())) {
        continue;
      }
      ensure_models(func.value());
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
    if (hardware_model || capability_model || mod->global_infos.Get(attr::kTLDevicePrograms)) {
      mod = mod->ShallowCopy();
    }
    if (hardware_model) {
      mod->UpdateGlobalInfo(attr::kTLTTHardwareModel, Array<GlobalInfo>{hardware_model.value()});
    }
    if (capability_model) {
      mod->UpdateGlobalInfo(attr::kTLSpatialCapabilityModel,
                            Array<GlobalInfo>{capability_model.value()});
    }
    if (auto maybe_registry = mod->global_infos.Get(attr::kTLDevicePrograms)) {
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

tvm::transform::Pass LowerToSpatialProgram() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext pass_ctx) {
    mod = AnalyzeSpatialDomainPlan()(mod);
    mod = AnalyzeSpatialExecutionPlan()(mod);
    mod = MaterializeSpatialProgram()(mod);
    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 0,
                                          "tl.transform.LowerToSpatialProgram", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AnalyzeSpatialExecutionPlan",
                        AnalyzeSpatialExecutionPlan);
  refl::GlobalDef().def("tl.transform.MaterializeSpatialProgram",
                        MaterializeSpatialProgram);
  refl::GlobalDef().def("tl.transform.LowerToSpatialProgram", LowerToSpatialProgram);
}

}  // namespace tl
}  // namespace tvm
