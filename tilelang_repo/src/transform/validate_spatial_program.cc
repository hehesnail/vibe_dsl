/*!
 * \file validate_spatial_program.cc
 * \brief Validate minimal Phase B SpatialProgram invariants.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <string>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/blackhole_utils.h"
#include "common/spatial_program.h"
#include "common/spatial_vocab.h"
#include "common/semantic_program.h"
#include "common/semantic_vocab.h"

namespace tvm {
namespace tl {

using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::Optional;
using tvm::ffi::String;
using tvm::Integer;
using namespace tvm::tl::semantic;
namespace sp = tvm::tl::spatial;
using tvm::tl::str;

namespace {

// ---------------------------------------------------------------------------
// Payload extraction helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Comparison helpers
// ---------------------------------------------------------------------------

bool HasTrait(const Array<String>& traits, const char* expected) {
  for (const String& trait : traits) {
    if (str(trait) == expected) {
      return true;
    }
  }
  return false;
}

bool IsKnownDomainTransformKind(const std::string& kind) {
  static const std::unordered_set<std::string> kKnownKinds = {
      "identity", "derived", "filtered", "grouped", "routed", "paged", "chunked"};
  return kKnownKinds.count(kind);
}

bool IsKnownPartitionFamily(const std::string& family) {
  static const std::unordered_set<std::string> kKnownFamilies = {
      "regular", "derived", "filtered", "grouped", "routed", "paged", "chunked"};
  return kKnownFamilies.count(family);
}

bool DomainTransformRequiresIndexedLayout(const std::string& kind) {
  return kind == "derived" || kind == "paged" || kind == "routed";
}

bool PartitionFamilyRequiresIndexedPartition(const std::string& family) {
  return family == "derived" || family == "paged" || family == "routed";
}

bool PartitionFamilyRequiresFilteredPartition(const std::string& family) {
  return family == "filtered" || family == "grouped" || family == "chunked";
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

struct ExpectedDomainContract {
  std::string domain_transform_kind = "identity";
  std::string partition_family = "regular";
};

ExpectedDomainContract DeriveExpectedDomainContract(
    const SemanticProgram& program, const Domain& domain,
    const std::unordered_map<std::string, std::optional<StateRole>>& state_role_by_name) {
  const bool has_derived_indices = HasTrait(domain->traits, "derived_indices");
  const bool has_work_dependent_bounds = HasTrait(domain->traits, "work_dependent_bounds");
  const bool has_pipeline_contract =
      HasTrait(domain->traits, "pipeline") ||
      HasSupplementPayload(program, SupplementKind::kPipelineStructure,
                           schema_key::kPipelineStages);
  const bool has_selection_state =
      DomainHasStateRole(program, domain, state_role_by_name, StateRole::kSelectionState);
  const bool has_index_state =
      DomainHasStateRole(program, domain, state_role_by_name, StateRole::kIndexState);
  const bool has_reduction_accumulator = DomainHasStateRole(
      program, domain, state_role_by_name, StateRole::kReductionAccumulator);
  const bool has_selected_access = DomainHasAccessTrait(program, domain, "selected");
  const bool has_indexed_access = DomainHasAccessTrait(program, domain, "indexed");

  ExpectedDomainContract contract;
  if (has_derived_indices) {
    contract.domain_transform_kind = "derived";
    contract.partition_family = "derived";
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
    return contract;
  }
  if (has_selection_state && has_selected_access && has_indexed_access) {
    if (!has_index_state && !has_reduction_accumulator) {
      contract.domain_transform_kind = "chunked";
      contract.partition_family = "chunked";
      return contract;
    }
    contract.domain_transform_kind = "grouped";
    contract.partition_family = "grouped";
    return contract;
  }
  return contract;
}

struct ProducerVersionEdge {
  std::string producer_update;
  std::string produced_version;
};

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
        << "ValidateSpatialProgram requires a unique update-result version for update "
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
      << "ValidateSpatialProgram found cyclic semantic join producer resolution for version "
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
        const bool duplicate = std::find_if(
            edges.begin(), edges.end(), [&](const ProducerVersionEdge& existing) {
              return existing.producer_update == edge.producer_update &&
                     existing.produced_version == edge.produced_version;
            }) != edges.end();
        if (!duplicate) {
          edges.push_back(edge);
        }
      }
    }
  }

  active_versions->erase(version_name);
  (*memoized_edges)[version_name] = edges;
  return edges;
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
  std::unordered_map<std::string, StateJoin> join_by_output_version;
  for (const StateJoin& join : program->state_joins) {
    join_by_output_version[str(join->output_version)] = join;
  }

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

bool SamePhaseSignature(const ProgramPhase& lhs, const ProgramPhase& rhs) {
  auto lhs_phase_index = GetPayloadIndex(lhs->payload, schema_key::kPhaseIndex);
  auto rhs_phase_index = GetPayloadIndex(rhs->payload, schema_key::kPhaseIndex);
  auto lhs_task_indices = lhs->payload.Get(String(schema_key::kTaskIndices));
  auto rhs_task_indices = rhs->payload.Get(String(schema_key::kTaskIndices));
  auto lhs_channel_indices = lhs->payload.Get(String(schema_key::kChannelIndices));
  auto rhs_channel_indices = rhs->payload.Get(String(schema_key::kChannelIndices));
  auto lhs_closure_basis = GetPayloadString(lhs->payload, schema_key::kClosureBasis);
  auto rhs_closure_basis = GetPayloadString(rhs->payload, schema_key::kClosureBasis);
  return str(lhs->name) == str(rhs->name) &&
         SameStringArray(lhs->task_names, rhs->task_names) &&
         SameStringArray(lhs->channel_names, rhs->channel_names) &&
         lhs_phase_index == rhs_phase_index &&
         lhs_closure_basis == rhs_closure_basis &&
         lhs_task_indices.has_value() == rhs_task_indices.has_value() &&
         lhs_channel_indices.has_value() == rhs_channel_indices.has_value() &&
         (!lhs_task_indices.has_value() ||
          SameIntegerAnyArray(Downcast<Array<Any>>(lhs_task_indices.value()),
                              Downcast<Array<Any>>(rhs_task_indices.value()))) &&
         (!lhs_channel_indices.has_value() ||
          SameIntegerAnyArray(Downcast<Array<Any>>(lhs_channel_indices.value()),
                              Downcast<Array<Any>>(rhs_channel_indices.value())));
}

// ---------------------------------------------------------------------------
// Intent classification
// ---------------------------------------------------------------------------

bool IsPipelineContractIntent(const ResourceIntent& intent) {
  auto parsed = sp::ParseSpatialResourceIntentKind(str(intent->kind));
  return parsed && *parsed == sp::SpatialResourceIntentKind::kSynchronizationSupport &&
         HasTrait(intent->traits, "pipeline_contract");
}

bool IsFragmentContractIntent(const ResourceIntent& intent) {
  auto parsed = sp::ParseSpatialResourceIntentKind(str(intent->kind));
  return parsed && *parsed == sp::SpatialResourceIntentKind::kLoweringSupport &&
         HasTrait(intent->traits, "fragment_contract");
}

bool IsStatefulRole(const std::string& role_str) {
  auto role = ParseStateRole(role_str);
  return role && (*role == StateRole::kCarry ||
                  *role == StateRole::kReductionAccumulator ||
                  *role == StateRole::kSelectionState ||
                  *role == StateRole::kIndexState);
}

sp::SpatialTaskKind RequireTaskKind(const Task& task) {
  auto parsed = sp::ParseSpatialTaskKind(str(task->kind));
  ICHECK(parsed) << "ValidateSpatialProgram found unknown task kind " << str(task->kind);
  return *parsed;
}

sp::SpatialChannelKind RequireChannelKind(const Channel& channel) {
  auto parsed = sp::ParseSpatialChannelKind(str(channel->kind));
  ICHECK(parsed) << "ValidateSpatialProgram found unknown channel kind " << str(channel->kind);
  return *parsed;
}

sp::SpatialLayoutKind RequireLayoutKind(const SpatialLayout& layout) {
  auto parsed = sp::ParseSpatialLayoutKind(str(layout->kind));
  ICHECK(parsed) << "ValidateSpatialProgram found unknown layout kind " << str(layout->kind);
  return *parsed;
}

sp::SpatialPartitionKind RequirePartitionKind(const WorkPartition& partition) {
  auto parsed = sp::ParseSpatialPartitionKind(str(partition->kind));
  ICHECK(parsed) << "ValidateSpatialProgram found unknown work partition kind "
                 << str(partition->kind);
  return *parsed;
}

sp::SpatialPlacementKind RequirePlacementKind(const Placement& placement) {
  auto parsed = sp::ParseSpatialPlacementKind(str(placement->kind));
  ICHECK(parsed) << "ValidateSpatialProgram found unknown placement kind " << str(placement->kind);
  return *parsed;
}

sp::SpatialSyncKind RequireSyncKind(const SyncEdge& edge) {
  auto parsed = sp::ParseSpatialSyncKind(str(edge->kind));
  ICHECK(parsed) << "ValidateSpatialProgram found unknown sync edge kind " << str(edge->kind);
  return *parsed;
}

sp::SpatialResourceIntentKind RequireResourceIntentKind(const ResourceIntent& intent) {
  auto parsed = sp::ParseSpatialResourceIntentKind(str(intent->kind));
  ICHECK(parsed) << "ValidateSpatialProgram found unknown resource intent kind "
                 << str(intent->kind);
  return *parsed;
}

sp::SpatialChannelPayloadKind RequireChannelPayloadKind(const Channel& channel) {
  auto maybe_payload_kind = GetPayloadString(channel->payload, schema_key::kPayloadKind);
  ICHECK(maybe_payload_kind)
      << "ValidateSpatialProgram requires channels to carry payload_kind/delivery_kind contract";
  auto parsed = sp::ParseSpatialChannelPayloadKind(*maybe_payload_kind);
  ICHECK(parsed) << "ValidateSpatialProgram found unknown channel payload kind "
                 << *maybe_payload_kind;
  return *parsed;
}

sp::SpatialChannelDeliveryKind RequireChannelDeliveryKind(const Channel& channel) {
  auto maybe_delivery_kind = GetPayloadString(channel->payload, schema_key::kDeliveryKind);
  ICHECK(maybe_delivery_kind)
      << "ValidateSpatialProgram requires channels to carry payload_kind/delivery_kind contract";
  auto parsed = sp::ParseSpatialChannelDeliveryKind(*maybe_delivery_kind);
  ICHECK(parsed) << "ValidateSpatialProgram found unknown channel delivery kind "
                 << *maybe_delivery_kind;
  return *parsed;
}

// ---------------------------------------------------------------------------
// Validation context: shared index maps built once per function
// ---------------------------------------------------------------------------

struct ValidationContext {
  std::vector<std::string> phase_name_by_index;
  std::vector<std::vector<int64_t>> phase_task_indices_by_index;
  std::vector<std::vector<int64_t>> phase_channel_indices_by_index;

  std::vector<std::string> task_name_by_index;
  std::vector<int64_t> task_phase_index_by_index;

  std::vector<std::string> channel_name_by_index;
  std::vector<int64_t> channel_source_task_index_by_index;
  std::vector<int64_t> channel_target_task_index_by_index;
  std::unordered_set<int64_t> cross_phase_materialized_state_indices;
};

// ---------------------------------------------------------------------------
// Sub-validators
// ---------------------------------------------------------------------------

void ValidatePhases(const SpatialProgram& program, ValidationContext* ctx) {
  ctx->phase_name_by_index.resize(program->phases.size());
  ctx->phase_task_indices_by_index.resize(program->phases.size());
  ctx->phase_channel_indices_by_index.resize(program->phases.size());

  std::unordered_set<std::string> phase_names;
  for (const ProgramPhase& phase : program->phases) {
    const std::string phase_name = str(phase->name);
    ICHECK(phase_names.insert(phase_name).second)
        << "ValidateSpatialProgram found duplicate phase " << phase_name;
    auto maybe_phase_index = GetPayloadIndex(phase->payload, schema_key::kPhaseIndex);
    ICHECK(maybe_phase_index)
        << "ValidateSpatialProgram requires program phases to carry phase_index contract";
    ICHECK_GE(*maybe_phase_index, 0);
    ICHECK_LT(*maybe_phase_index, program->phases.size())
        << "ValidateSpatialProgram found program phase with invalid phase_index";
    auto maybe_task_indices = GetPayloadIndices(phase->payload, schema_key::kTaskIndices);
    ICHECK(maybe_task_indices)
        << "ValidateSpatialProgram requires program phases to carry task_indices contract";
    auto maybe_channel_indices = GetPayloadIndices(phase->payload, schema_key::kChannelIndices);
    ICHECK(maybe_channel_indices)
        << "ValidateSpatialProgram requires program phases to carry channel_indices contract";
    auto maybe_closure_basis = GetPayloadString(phase->payload, schema_key::kClosureBasis);
    ICHECK(maybe_closure_basis && !maybe_closure_basis->empty())
        << "ValidateSpatialProgram requires program phases to carry closure_basis contract";
    ctx->phase_name_by_index[*maybe_phase_index] = phase_name;
    ctx->phase_task_indices_by_index[*maybe_phase_index] = std::move(*maybe_task_indices);
    ctx->phase_channel_indices_by_index[*maybe_phase_index] = std::move(*maybe_channel_indices);
  }
}

void ValidateTasks(const SpatialProgram& program, ValidationContext* ctx) {
  ctx->task_name_by_index.resize(program->tasks.size());
  ctx->task_phase_index_by_index.resize(program->tasks.size(), -1);

  std::unordered_set<std::string> task_names;
  for (int task_index = 0; task_index < program->tasks.size(); ++task_index) {
    const Task& task = program->tasks[task_index];
    RequireTaskKind(task);
    const std::string task_name = str(task->name);
    ICHECK(task_names.insert(task_name).second)
        << "ValidateSpatialProgram found duplicate task " << task_name;
    auto maybe_execution_role = GetPayloadString(task->payload, schema_key::kExecutionRole);
    ICHECK(maybe_execution_role && !maybe_execution_role->empty())
        << "ValidateSpatialProgram requires tasks to carry execution_role contract";
    auto maybe_formation_basis = GetPayloadString(task->payload, schema_key::kFormationBasis);
    ICHECK(maybe_formation_basis && !maybe_formation_basis->empty())
        << "ValidateSpatialProgram requires tasks to carry formation_basis contract";
    auto maybe_phase_index = GetPayloadIndex(task->payload, schema_key::kPhaseIndex);
    ICHECK(maybe_phase_index)
        << "ValidateSpatialProgram requires tasks to carry phase_index contract";
    ICHECK_GE(*maybe_phase_index, 0);
    ICHECK_LT(*maybe_phase_index, program->phases.size())
        << "ValidateSpatialProgram found task with invalid phase_index";
    const std::string& phase_name = ctx->phase_name_by_index[*maybe_phase_index];
    ICHECK(!phase_name.empty())
        << "ValidateSpatialProgram found task referencing unresolved phase_index";
    ICHECK_EQ(str(task->phase_name), phase_name)
        << "ValidateSpatialProgram found task phase_name inconsistent with phase_index contract";
    ctx->task_name_by_index[task_index] = task_name;
    ctx->task_phase_index_by_index[task_index] = *maybe_phase_index;
  }
}

void ValidateChannels(const SpatialProgram& program, const SemanticProgram& semantic_program,
                      ValidationContext* ctx) {
  ctx->channel_name_by_index.resize(program->channels.size());
  ctx->channel_source_task_index_by_index.resize(program->channels.size(), -1);
  ctx->channel_target_task_index_by_index.resize(program->channels.size(), -1);
  const auto update_result_version_by_update =
      BuildUniqueUpdateResultVersionByUpdate(semantic_program->state_defs);
  const auto version_to_producer_edges = BuildVersionProducerEdges(semantic_program);

  std::unordered_set<std::string> channel_names;
  for (int channel_index = 0; channel_index < program->channels.size(); ++channel_index) {
    const Channel& channel = program->channels[channel_index];
    const auto channel_kind = RequireChannelKind(channel);
    const std::string channel_name = str(channel->name);
    ICHECK(channel_names.insert(channel_name).second)
        << "ValidateSpatialProgram found duplicate channel " << channel_name;
    auto maybe_source_task_index =
        GetPayloadIndex(channel->payload, schema_key::kSourceTaskIndex);
    auto maybe_target_task_index =
        GetPayloadIndex(channel->payload, schema_key::kTargetTaskIndex);
    ICHECK(maybe_source_task_index && maybe_target_task_index)
        << "ValidateSpatialProgram requires channels to carry "
           "source_task_index/target_task_index contract";
    ICHECK_GE(*maybe_source_task_index, 0);
    ICHECK_LT(*maybe_source_task_index, program->tasks.size())
        << "ValidateSpatialProgram found channel with invalid source_task_index";
    ICHECK_GE(*maybe_target_task_index, 0);
    ICHECK_LT(*maybe_target_task_index, program->tasks.size())
        << "ValidateSpatialProgram found channel with invalid target_task_index";
    ICHECK_EQ(str(channel->source_task), ctx->task_name_by_index[*maybe_source_task_index])
        << "ValidateSpatialProgram found channel source_task inconsistent with source_task_index";
    ICHECK_EQ(str(channel->target_task), ctx->task_name_by_index[*maybe_target_task_index])
        << "ValidateSpatialProgram found channel target_task inconsistent with target_task_index";
    const auto payload_kind = RequireChannelPayloadKind(channel);
    const auto delivery_kind = RequireChannelDeliveryKind(channel);
    const int64_t source_phase_index = ctx->task_phase_index_by_index[*maybe_source_task_index];
    const int64_t target_phase_index = ctx->task_phase_index_by_index[*maybe_target_task_index];
    const bool cross_phase = source_phase_index != target_phase_index;
    if (payload_kind == sp::SpatialChannelPayloadKind::kStateVersion) {
      auto maybe_state_index = GetPayloadIndex(channel->payload, schema_key::kStateIndex);
      ICHECK(maybe_state_index)
          << "ValidateSpatialProgram requires state_version channels to carry state_index contract";
      ICHECK_GE(*maybe_state_index, 0);
      ICHECK_LT(*maybe_state_index, semantic_program->states.size())
          << "ValidateSpatialProgram found state_version channel with invalid state_index";
      ICHECK(!str(channel->state_name).empty())
          << "ValidateSpatialProgram requires state_version channels to carry state_name";
      ICHECK_EQ(str(semantic_program->states[*maybe_state_index]->name), str(channel->state_name))
          << "ValidateSpatialProgram found state_version channel with state_index/state_name mismatch";
      auto maybe_source_version =
          GetPayloadString(channel->payload, schema_key::kSourceVersion);
      ICHECK(maybe_source_version && !maybe_source_version->empty())
          << "ValidateSpatialProgram requires state_version channels to carry source_version contract";
      auto maybe_target_version =
          GetPayloadString(channel->payload, schema_key::kTargetVersion);
      ICHECK(maybe_target_version && !maybe_target_version->empty())
          << "ValidateSpatialProgram requires state_version channels to carry target_version contract";
      const Task& source_task = program->tasks[*maybe_source_task_index];
      const Task& target_task = program->tasks[*maybe_target_task_index];
      auto task_contains_update = [&](const Task& task, const std::string& update_name) {
        for (const String& task_update_name : task->update_names) {
          if (str(task_update_name) == update_name) {
            return true;
          }
        }
        return false;
      };
      auto producer_edges_it = version_to_producer_edges.find(*maybe_source_version);
      ICHECK(producer_edges_it != version_to_producer_edges.end())
          << "ValidateSpatialProgram found state_version channel with unresolved source_version";
      const bool source_matches_semantic_producer = std::any_of(
          producer_edges_it->second.begin(), producer_edges_it->second.end(),
          [&](const ProducerVersionEdge& edge) {
            return task_contains_update(source_task, edge.producer_update);
          });
      ICHECK(source_matches_semantic_producer)
          << "ValidateSpatialProgram found state_version channel source_version inconsistent "
             "with source task update membership";
      bool target_matches_semantic_use = false;
      for (const StateUse& use : semantic_program->state_uses) {
        const std::string consumer_update = str(use->consumer_update);
        if (str(use->state_name) != str(channel->state_name) ||
            str(use->version_name) != *maybe_source_version ||
            !task_contains_update(target_task, consumer_update)) {
          continue;
        }
        auto target_version_it = update_result_version_by_update.find(consumer_update);
        if (target_version_it != update_result_version_by_update.end() &&
            target_version_it->second == *maybe_target_version) {
          target_matches_semantic_use = true;
          break;
        }
      }
      bool target_matches_internal_handoff = false;
      if (*maybe_source_version == *maybe_target_version) {
        for (const String& task_update_name : target_task->update_names) {
          auto target_version_it = update_result_version_by_update.find(str(task_update_name));
          if (target_version_it == update_result_version_by_update.end() ||
              target_version_it->second != *maybe_target_version) {
            continue;
          }
          if (task_contains_update(source_task, str(task_update_name))) {
            target_matches_internal_handoff = true;
            break;
          }
        }
      }
      ICHECK(target_matches_semantic_use || target_matches_internal_handoff)
          << "ValidateSpatialProgram found state_version channel target_version inconsistent "
             "with semantic use/update-result linkage";
    }
    if (channel_kind == sp::SpatialChannelKind::kCarry ||
        channel_kind == sp::SpatialChannelKind::kReduceMerge) {
      ICHECK(payload_kind == sp::SpatialChannelPayloadKind::kStateVersion ||
             payload_kind == sp::SpatialChannelPayloadKind::kToken)
          << "ValidateSpatialProgram requires carry/reduce_merge channels to carry "
             "state_version or token payload";
    }
    if (delivery_kind == sp::SpatialChannelDeliveryKind::kPhaseBoundaryMaterialized) {
      ICHECK(cross_phase)
          << "ValidateSpatialProgram requires phase_boundary_materialized delivery to cross "
             "phase boundaries";
      ICHECK_LT(source_phase_index, target_phase_index)
          << "ValidateSpatialProgram requires phase_boundary_materialized delivery to advance "
             "phase order";
      if (payload_kind == sp::SpatialChannelPayloadKind::kStateVersion) {
        auto maybe_state_index = GetPayloadIndex(channel->payload, schema_key::kStateIndex);
        ICHECK(maybe_state_index)
            << "ValidateSpatialProgram requires phase-boundary state_version channels to carry "
               "state_index contract";
        ctx->cross_phase_materialized_state_indices.insert(*maybe_state_index);
      }
    } else if (cross_phase) {
      ICHECK(false)
          << "ValidateSpatialProgram requires cross-phase channels to carry "
             "phase_boundary_materialized delivery";
    }
    ctx->channel_name_by_index[channel_index] = channel_name;
    ctx->channel_source_task_index_by_index[channel_index] = *maybe_source_task_index;
    ctx->channel_target_task_index_by_index[channel_index] = *maybe_target_task_index;
  }
}

void ValidatePlacementsAndSyncEdges(const SpatialProgram& program,
                                    const std::string& member_func,
                                    const ValidationContext& ctx) {
  for (const Placement& placement : program->placements) {
    RequirePlacementKind(placement);
    auto maybe_task_index = GetPayloadIndex(placement->payload, schema_key::kTaskIndex);
    ICHECK(maybe_task_index)
        << "ValidateSpatialProgram requires placements to carry task_index contract";
    ICHECK_GE(*maybe_task_index, 0);
    ICHECK_LT(*maybe_task_index, program->tasks.size())
        << "ValidateSpatialProgram found placement with invalid task_index";
    ICHECK_EQ(str(placement->task_name), ctx.task_name_by_index[*maybe_task_index])
        << "ValidateSpatialProgram found placement task_name inconsistent with task_index";
    ICHECK_EQ(str(placement->member_func), member_func)
        << "ValidateSpatialProgram requires placement.member_func to match "
           "SpatialProgram.member_func";
    auto maybe_affinity_kind = GetPayloadString(placement->payload, schema_key::kAffinityKind);
    ICHECK(maybe_affinity_kind)
        << "ValidateSpatialProgram requires placements to carry affinity_kind contract";
    auto maybe_obligation_kind =
        GetPayloadString(placement->payload, schema_key::kObligationKind);
    ICHECK(maybe_obligation_kind && !maybe_obligation_kind->empty())
        << "ValidateSpatialProgram requires placements to carry obligation_kind contract";
    auto maybe_placement_domain =
        GetPayloadString(placement->payload, schema_key::kPlacementDomain);
    ICHECK(maybe_placement_domain && !maybe_placement_domain->empty())
        << "ValidateSpatialProgram requires placements to carry placement_domain contract";
  }

  std::unordered_set<std::string> covered_phase_pairs;
  for (const SyncEdge& edge : program->sync_edges) {
    RequireSyncKind(edge);
    auto maybe_source_task_index = GetPayloadIndex(edge->payload, schema_key::kSourceTaskIndex);
    auto maybe_target_task_index = GetPayloadIndex(edge->payload, schema_key::kTargetTaskIndex);
    ICHECK(maybe_source_task_index && maybe_target_task_index)
        << "ValidateSpatialProgram requires sync edges to carry "
           "source_task_index/target_task_index contract";
    ICHECK_GE(*maybe_source_task_index, 0);
    ICHECK_LT(*maybe_source_task_index, program->tasks.size())
        << "ValidateSpatialProgram found sync edge with invalid source_task_index";
    ICHECK_GE(*maybe_target_task_index, 0);
    ICHECK_LT(*maybe_target_task_index, program->tasks.size())
        << "ValidateSpatialProgram found sync edge with invalid target_task_index";
    ICHECK_EQ(str(edge->source), ctx.task_name_by_index[*maybe_source_task_index])
        << "ValidateSpatialProgram found sync edge source inconsistent with source_task_index";
    ICHECK_EQ(str(edge->target), ctx.task_name_by_index[*maybe_target_task_index])
        << "ValidateSpatialProgram found sync edge target inconsistent with target_task_index";
    auto maybe_ordering_kind = GetPayloadString(edge->payload, schema_key::kOrderingKind);
    ICHECK(maybe_ordering_kind && !maybe_ordering_kind->empty())
        << "ValidateSpatialProgram requires sync edges to carry ordering_kind contract";
    auto maybe_materialization_kind =
        GetPayloadString(edge->payload, schema_key::kMaterializationKind);
    ICHECK(maybe_materialization_kind && !maybe_materialization_kind->empty())
        << "ValidateSpatialProgram requires sync edges to carry materialization_kind contract";
    const int64_t source_phase_index = ctx.task_phase_index_by_index[*maybe_source_task_index];
    const int64_t target_phase_index = ctx.task_phase_index_by_index[*maybe_target_task_index];
    if (source_phase_index != target_phase_index) {
      ICHECK_LT(source_phase_index, target_phase_index)
          << "ValidateSpatialProgram requires cross-phase sync edges to advance phase order";
      covered_phase_pairs.insert(std::to_string(source_phase_index) + "->" +
                                 std::to_string(target_phase_index) + "|" +
                                 *maybe_ordering_kind + "|" + *maybe_materialization_kind);
    }
  }

  if (program->phases.size() <= 1) {
    return;
  }
  std::unordered_set<std::string> required_phase_pairs;
  for (int channel_index = 0; channel_index < program->channels.size(); ++channel_index) {
    const int64_t source_phase_index =
        ctx.task_phase_index_by_index[ctx.channel_source_task_index_by_index[channel_index]];
    const int64_t target_phase_index =
        ctx.task_phase_index_by_index[ctx.channel_target_task_index_by_index[channel_index]];
    if (source_phase_index == target_phase_index) {
      continue;
    }
    const Channel& channel = program->channels[channel_index];
    const auto channel_kind = RequireChannelKind(channel);
    const auto delivery_kind = RequireChannelDeliveryKind(channel);
    required_phase_pairs.insert(std::to_string(source_phase_index) + "->" +
                                std::to_string(target_phase_index) + "|" +
                                DeriveOrderingKindForChannel(channel_kind, delivery_kind) + "|" +
                                DeriveMaterializationKindForChannel(channel_kind, delivery_kind));
  }
  for (const std::string& required_pair : required_phase_pairs) {
    ICHECK(covered_phase_pairs.count(required_pair))
        << "ValidateSpatialProgram requires cross-phase channel coverage to materialize "
           "sync_edge contract";
  }
}

void ValidatePhaseTaskChannelCoherence(const SpatialProgram& program,
                                       const ValidationContext& ctx) {
  for (int i = 0; i < program->phases.size(); ++i) {
    const ProgramPhase& phase = program->phases[i];
    auto maybe_phase_index = GetPayloadIndex(phase->payload, schema_key::kPhaseIndex);
    ICHECK(maybe_phase_index);
    ICHECK_EQ(*maybe_phase_index, i)
        << "ValidateSpatialProgram requires phase_index contract to follow phase order";
    const auto& task_indices = ctx.phase_task_indices_by_index[i];
    ICHECK_EQ(phase->task_names.size(), task_indices.size())
        << "ValidateSpatialProgram requires phase task_names to stay aligned with task_indices";
    for (int j = 0; j < task_indices.size(); ++j) {
      const int64_t task_index = task_indices[j];
      ICHECK_GE(task_index, 0);
      ICHECK_LT(task_index, program->tasks.size())
          << "ValidateSpatialProgram found phase with invalid task_indices entry";
      ICHECK_EQ(ctx.task_phase_index_by_index[task_index], i)
          << "ValidateSpatialProgram found phase task_indices entry assigned to a different phase";
      ICHECK_EQ(str(phase->task_names[j]), ctx.task_name_by_index[task_index])
          << "ValidateSpatialProgram found phase task_names inconsistent with task_indices";
    }
    const auto& channel_indices = ctx.phase_channel_indices_by_index[i];
    ICHECK_EQ(phase->channel_names.size(), channel_indices.size())
        << "ValidateSpatialProgram requires phase channel_names to stay aligned with channel_indices";
    for (int j = 0; j < channel_indices.size(); ++j) {
      const int64_t channel_index = channel_indices[j];
      ICHECK_GE(channel_index, 0);
      ICHECK_LT(channel_index, program->channels.size())
          << "ValidateSpatialProgram found phase with invalid channel_indices entry";
      ICHECK_EQ(str(phase->channel_names[j]), ctx.channel_name_by_index[channel_index])
          << "ValidateSpatialProgram found phase channel_names inconsistent with channel_indices";
      ICHECK_EQ(ctx.task_phase_index_by_index[ctx.channel_target_task_index_by_index[channel_index]], i)
          << "ValidateSpatialProgram requires phase channel contracts to target tasks in "
             "the owning phase";
    }
    if (program->phases.size() > 1 && i > 0) {
      ICHECK(!channel_indices.empty())
          << "ValidateSpatialProgram requires downstream multi-phase programs to reference "
             "at least one channel";
    }
  }
}

struct SemanticRequirements {
  bool requires_pipeline_contract = false;
  bool requires_fragment_contract = false;
  bool requires_work_dependent_payload = false;
  int stateful_state_count = 0;
  std::unordered_set<int> stateful_state_indices;
};

void ValidateSemanticAlignment(const SpatialProgram& program,
                               const SemanticProgram& semantic_program,
                               SemanticRequirements* reqs) {
  const auto state_role_by_name = BuildStateRoleByName(semantic_program);
  std::unordered_set<int64_t> covered_layout_domain_indices;
  for (const SpatialLayout& layout : program->layouts) {
    const auto layout_kind = RequireLayoutKind(layout);
    auto maybe_domain_index = GetPayloadIndex(layout->payload, schema_key::kDomainIndex);
    ICHECK(maybe_domain_index)
        << "ValidateSpatialProgram requires spatial layouts to carry domain_index contract";
    auto maybe_domain_transform_kind =
        GetPayloadString(layout->payload, schema_key::kDomainTransformKind);
    ICHECK(maybe_domain_transform_kind && !maybe_domain_transform_kind->empty())
        << "ValidateSpatialProgram requires spatial layouts to carry domain_transform_kind contract";
    ICHECK(IsKnownDomainTransformKind(*maybe_domain_transform_kind))
        << "ValidateSpatialProgram found layout with unknown domain_transform_kind "
        << *maybe_domain_transform_kind;
    ICHECK_GE(*maybe_domain_index, 0);
    ICHECK_LT(*maybe_domain_index, semantic_program->domains.size())
        << "ValidateSpatialProgram found layout with invalid semantic domain index";
    ICHECK(covered_layout_domain_indices.insert(*maybe_domain_index).second)
        << "ValidateSpatialProgram found duplicate spatial layout domain_index";
    const Domain& domain = semantic_program->domains[*maybe_domain_index];
    const ExpectedDomainContract expected_contract =
        DeriveExpectedDomainContract(semantic_program, domain, state_role_by_name);
    ICHECK(SameStringArray(layout->axes, domain->axes))
        << "ValidateSpatialProgram found layout axes inconsistent with semantic domain";
    const bool semantic_indexed = HasTrait(domain->traits, "derived_indices");
    const bool layout_indexed = layout_kind == sp::SpatialLayoutKind::kIndexed;
    ICHECK_EQ(layout_indexed, semantic_indexed)
        << "ValidateSpatialProgram found layout kind inconsistent with semantic domain "
           "derived_indices trait";
    ICHECK_EQ(*maybe_domain_transform_kind, expected_contract.domain_transform_kind)
        << "ValidateSpatialProgram found domain_transform_kind inconsistent with semantic truth";
    if (DomainTransformRequiresIndexedLayout(*maybe_domain_transform_kind)) {
      ICHECK(layout_kind == sp::SpatialLayoutKind::kIndexed)
          << "ValidateSpatialProgram requires indexed layout kind for domain_transform_kind "
          << *maybe_domain_transform_kind;
    }
  }
  ICHECK_EQ(covered_layout_domain_indices.size(), semantic_program->domains.size())
      << "ValidateSpatialProgram requires spatial layout contract to cover every semantic "
         "domain index";
  for (int domain_index = 0; domain_index < semantic_program->domains.size(); ++domain_index) {
    ICHECK(covered_layout_domain_indices.count(domain_index))
        << "ValidateSpatialProgram requires spatial layout contract to cover every semantic "
           "domain index";
  }

  std::unordered_set<int64_t> covered_partition_domain_indices;
  for (const WorkPartition& partition : program->work_partitions) {
    const auto partition_kind = RequirePartitionKind(partition);
    auto maybe_domain_index = GetPayloadIndex(partition->payload, schema_key::kDomainIndex);
    ICHECK(maybe_domain_index) << "ValidateSpatialProgram requires work partitions to carry "
                                  "domain_index contract";
    auto maybe_partition_family =
        GetPayloadString(partition->payload, schema_key::kPartitionFamily);
    ICHECK(maybe_partition_family && !maybe_partition_family->empty())
        << "ValidateSpatialProgram requires work partitions to carry partition_family contract";
    ICHECK(IsKnownPartitionFamily(*maybe_partition_family))
        << "ValidateSpatialProgram found work partition with unknown partition_family "
        << *maybe_partition_family;
    ICHECK_GE(*maybe_domain_index, 0);
    ICHECK_LT(*maybe_domain_index, semantic_program->domains.size())
        << "ValidateSpatialProgram found work partition with invalid semantic domain index";
    ICHECK(covered_partition_domain_indices.insert(*maybe_domain_index).second)
        << "ValidateSpatialProgram found duplicate work partition domain_index";
    const Domain& domain = semantic_program->domains[*maybe_domain_index];
    const ExpectedDomainContract expected_contract =
        DeriveExpectedDomainContract(semantic_program, domain, state_role_by_name);
    ICHECK(SameStringArray(partition->axes, domain->axes))
        << "ValidateSpatialProgram found work partition axes inconsistent with semantic domain";
    ICHECK_EQ(*maybe_partition_family, expected_contract.partition_family)
        << "ValidateSpatialProgram found partition_family inconsistent with semantic truth";
    if (PartitionFamilyRequiresIndexedPartition(*maybe_partition_family)) {
      ICHECK(partition_kind == sp::SpatialPartitionKind::kIndexed)
          << "ValidateSpatialProgram requires indexed work partition kind for partition_family "
          << *maybe_partition_family;
    } else if (PartitionFamilyRequiresFilteredPartition(*maybe_partition_family)) {
      ICHECK(partition_kind == sp::SpatialPartitionKind::kFiltered)
          << "ValidateSpatialProgram requires filtered work partition kind for partition_family "
          << *maybe_partition_family;
    }
  }
  ICHECK_EQ(covered_partition_domain_indices.size(), semantic_program->domains.size())
      << "ValidateSpatialProgram requires work partition contract to cover every semantic "
         "domain index";
  for (int domain_index = 0; domain_index < semantic_program->domains.size(); ++domain_index) {
    ICHECK(covered_partition_domain_indices.count(domain_index))
        << "ValidateSpatialProgram requires work partition contract to cover every semantic "
           "domain index";
  }
  for (const Domain& domain : semantic_program->domains) {
    if (HasTrait(domain->traits, "work_dependent_bounds")) {
      reqs->requires_work_dependent_payload = true;
      break;
    }
  }
  for (int i = 0; i < semantic_program->states.size(); ++i) {
    if (IsStatefulRole(str(semantic_program->states[i]->role))) {
      ++reqs->stateful_state_count;
      reqs->stateful_state_indices.insert(i);
    }
  }
  for (const SemanticSupplement& supplement : semantic_program->supplements) {
    const std::string supplement_kind = str(supplement->kind);
    if (supplement_kind == ToString(SupplementKind::kFragmentLoweringStructure)) {
      auto maybe_fragment_ops = supplement->payload.Get(String(schema_key::kFragmentOpKinds));
      reqs->requires_fragment_contract =
          maybe_fragment_ops && !Downcast<Array<Any>>(maybe_fragment_ops.value()).empty();
      continue;
    }
    if (supplement_kind != ToString(SupplementKind::kPipelineStructure)) {
      if (supplement_kind == ToString(SupplementKind::kWorkDecompositionStructure)) {
        auto maybe_loop_bounds =
            supplement->payload.Get(String(schema_key::kWorkDependentLoopBounds));
        reqs->requires_work_dependent_payload =
            maybe_loop_bounds && !Downcast<Array<Any>>(maybe_loop_bounds.value()).empty();
      }
      continue;
    }
    auto maybe_pipeline_stages = supplement->payload.Get(String(schema_key::kPipelineStages));
    reqs->requires_pipeline_contract =
        maybe_pipeline_stages && !Downcast<Array<Any>>(maybe_pipeline_stages.value()).empty();
    if (reqs->requires_pipeline_contract) {
      break;
    }
  }
}

void ValidateWorkDependentPayload(const SpatialProgram& program) {
  bool has_partition_payload = false;
  for (const WorkPartition& partition : program->work_partitions) {
    auto maybe_loop_bounds =
        partition->payload.Get(String(schema_key::kWorkDependentLoopBounds));
    if (!maybe_loop_bounds) {
      continue;
    }
    Array<Any> loop_bounds = Downcast<Array<Any>>(maybe_loop_bounds.value());
    ICHECK(!loop_bounds.empty())
        << "ValidateSpatialProgram requires work partition payload loop bounds to be non-empty";
    has_partition_payload = true;
    break;
  }
  ICHECK(has_partition_payload)
      << "ValidateSpatialProgram requires work-dependent domains to materialize work "
         "partition payload";
}

void ValidateResourceIntents(const SpatialProgram& program,
                             const Optional<SemanticProgram>& maybe_semantic_program,
                             const SemanticRequirements& reqs,
                             const ValidationContext& ctx) {
  const char* phase_boundary_str =
      sp::ToString(sp::SpatialResourceIntentKind::kPhaseBoundaryMaterialization);

  std::unordered_set<std::string> resource_intent_kinds;
  bool has_fragment_contract = false;
  bool has_pipeline_contract = false;
  int state_residency_count = 0;
  std::unordered_set<int> state_residency_state_indices;
  std::unordered_set<int> phase_boundary_state_indices;
  std::unordered_set<int> expected_phase_boundary_state_indices;
  for (int64_t state_index : ctx.cross_phase_materialized_state_indices) {
    expected_phase_boundary_state_indices.insert(static_cast<int>(state_index));
  }

  for (const ResourceIntent& intent : program->resource_intents) {
    const sp::SpatialResourceIntentKind intent_kind = RequireResourceIntentKind(intent);
    const std::string intent_kind_str = str(intent->kind);
    resource_intent_kinds.insert(intent_kind_str);
    state_residency_count += (intent_kind == sp::SpatialResourceIntentKind::kStateResidency);

    if (intent_kind == sp::SpatialResourceIntentKind::kStateResidency ||
        intent_kind == sp::SpatialResourceIntentKind::kPhaseBoundaryMaterialization) {
      auto maybe_target_kind = GetPayloadString(intent->payload, schema_key::kTargetKind);
      ICHECK(maybe_target_kind &&
             *maybe_target_kind == spatial_contract::kSemanticStateTarget)
          << "ValidateSpatialProgram requires state materialization intents to carry "
             "semantic_state target_kind contract";
      auto maybe_target_index = GetPayloadIndex(intent->payload, schema_key::kTargetIndex);
      ICHECK(maybe_target_index)
          << "ValidateSpatialProgram requires state materialization intents to carry "
             "target_index contract";
      ICHECK(maybe_semantic_program)
          << "ValidateSpatialProgram requires SemanticProgram when validating state-targeted "
             "resource intents";
      ICHECK_GE(*maybe_target_index, 0);
      ICHECK_LT(*maybe_target_index, maybe_semantic_program.value()->states.size())
          << "ValidateSpatialProgram found state-targeted intent with invalid target_index";
      if (intent_kind == sp::SpatialResourceIntentKind::kStateResidency) {
        if (reqs.stateful_state_indices.count(*maybe_target_index)) {
          state_residency_state_indices.insert(*maybe_target_index);
        }
      } else {
        ICHECK(reqs.stateful_state_indices.count(*maybe_target_index))
            << "ValidateSpatialProgram requires phase-boundary intents to target "
               "stateful semantic states";
        phase_boundary_state_indices.insert(*maybe_target_index);
      }
    }
    if (IsFragmentContractIntent(intent)) {
      has_fragment_contract = true;
      auto maybe_fragment_ops = intent->payload.Get(String(schema_key::kFragmentOpKinds));
      ICHECK(maybe_fragment_ops)
          << "ValidateSpatialProgram requires fragment contracts to carry fragment_op_kinds";
      Array<Any> fragment_ops = Downcast<Array<Any>>(maybe_fragment_ops.value());
      ICHECK(!fragment_ops.empty())
          << "ValidateSpatialProgram requires fragment contracts to carry at least one fragment op";
      bool requires_pointwise_payload = false;
      bool requires_row_broadcast_payload = false;
      for (const Any& op_any : fragment_ops) {
        const std::string op_name = Downcast<String>(op_any);
        requires_pointwise_payload |= op_name == "pointwise_chain";
        requires_row_broadcast_payload |= op_name == "row_broadcast";
      }
      if (requires_pointwise_payload) {
        auto maybe_pointwise_ops = intent->payload.Get(String(schema_key::kPointwiseOpKinds));
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
            << "ValidateSpatialProgram requires fragment row_broadcast_sources to be non-empty";
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
    if (!expected_phase_boundary_state_indices.empty()) {
      ICHECK(resource_intent_kinds.count(phase_boundary_str))
          << "ValidateSpatialProgram requires multi-phase programs with cross-phase "
             "materialized state channels to materialize phase-boundary resource intents";
    }
  }
  ICHECK_EQ(phase_boundary_state_indices.size(), expected_phase_boundary_state_indices.size())
      << "ValidateSpatialProgram requires phase-boundary intents to cover every cross-phase "
         "materialized state index";
  for (int state_index : expected_phase_boundary_state_indices) {
    ICHECK(phase_boundary_state_indices.count(state_index))
        << "ValidateSpatialProgram requires phase-boundary intents to cover every cross-phase "
           "materialized state index";
  }
  if (reqs.stateful_state_count > 0) {
    ICHECK_GE(state_residency_count, reqs.stateful_state_count)
        << "ValidateSpatialProgram requires stateful semantic states to materialize "
           "state-residency intents";
    ICHECK_GE(state_residency_state_indices.size(), reqs.stateful_state_count)
        << "ValidateSpatialProgram requires stateful semantic states to materialize "
           "state-residency intents";
  }
  if (reqs.requires_pipeline_contract) {
    ICHECK(has_pipeline_contract)
        << "ValidateSpatialProgram requires pipeline programs to materialize at least one "
           "pipeline contract";
  }
  if (reqs.requires_fragment_contract) {
    ICHECK(has_fragment_contract)
        << "ValidateSpatialProgram requires fragment programs to materialize at least one "
           "fragment contract";
  }
}

void ValidateRegistry(const IRModule& mod,
                      const std::unordered_map<std::string, Array<ProgramPhase>>& phases_by_member_func) {
  auto maybe_registry = mod->global_infos.Get(attr::kTLDevicePrograms);
  if (!maybe_registry) return;

  for (const GlobalInfo& item : maybe_registry.value()) {
    auto info = Downcast<TLDeviceProgramInfo>(item);
    Array<ProgramPhase> expected_phases;
    for (const String& member_func : info->member_funcs) {
      auto it = phases_by_member_func.find(str(member_func));
      if (it == phases_by_member_func.end()) {
        continue;
      }
      for (const ProgramPhase& phase : it->second) {
        expected_phases.push_back(phase);
      }
    }
    if (expected_phases.empty() && info->member_funcs.size() == 1) {
      auto root_it = phases_by_member_func.find(str(info->root_symbol));
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

      // Core structural requirements
      ICHECK_EQ(str(program->member_func), member_func)
          << "ValidateSpatialProgram requires SpatialProgram.member_func to match "
             "PrimFunc global_symbol";
      ICHECK(!program->phases.empty()) << "ValidateSpatialProgram requires at least one phase";
      ICHECK(!program->tasks.empty()) << "ValidateSpatialProgram requires at least one task";
      ICHECK(!program->layouts.empty())
          << "ValidateSpatialProgram requires at least one spatial layout";
      ICHECK(!program->work_partitions.empty())
          << "ValidateSpatialProgram requires at least one work partition";

      // Build validation context
      auto maybe_semantic_program =
          func.value()->GetAttr<SemanticProgram>(attr::kTLSemanticProgram);
      ICHECK(maybe_semantic_program)
          << "ValidateSpatialProgram requires SpatialProgram-bearing PrimFuncs to carry "
             "SemanticProgram truth";
      ValidationContext ctx;
      ValidatePhases(program, &ctx);
      ValidateTasks(program, &ctx);
      ValidateChannels(program, maybe_semantic_program.value(), &ctx);
      ValidatePlacementsAndSyncEdges(program, member_func, ctx);
      ValidatePhaseTaskChannelCoherence(program, ctx);

      // Semantic alignment
      SemanticRequirements reqs;
      if (maybe_semantic_program && !maybe_semantic_program.value()->domains.empty()) {
        ValidateSemanticAlignment(program, maybe_semantic_program.value(), &reqs);
      }
      if (reqs.requires_work_dependent_payload) {
        ValidateWorkDependentPayload(program);
      }

      // Resource intents
      ValidateResourceIntents(program, maybe_semantic_program, reqs, ctx);

      phases_by_member_func[member_func] = program->phases;
    }

    ValidateRegistry(mod, phases_by_member_func);
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
