/*!
 * \file lower_to_spatial_program.cc
 * \brief Materialize typed SpatialProgram companion IR from frozen SemanticProgram.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/transform.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/blackhole_utils.h"
#include "common/semantic_program.h"
#include "common/semantic_vocab.h"

namespace tvm {
namespace tl {

using tvm::DictAttrs;
using tvm::GlobalInfo;
using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::String;
using namespace tvm::tl::semantic;

namespace {

struct SpatialProgramBundle {
  SpatialProgram program;
  Array<ProgramPhase> phases;
};

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

Map<String, Any> EmptyPayload() { return Map<String, Any>(); }

Array<TIRAnchor> MakeAnchors(const std::string& kind, const std::string& value) {
  return Array<TIRAnchor>{TIRAnchor(String(kind), String(value))};
}

std::string GetMemberFuncName(const GlobalVar& gvar, const tir::PrimFunc& func) {
  return func->GetAttr<String>(tvm::attr::kGlobalSymbol).value_or(gvar->name_hint);
}

Array<String> GetWorkAxes(const SemanticProgram& program, const tir::PrimFunc& func) {
  if (!program->domains.empty() && !program->domains[0]->axes.empty()) {
    return program->domains[0]->axes;
  }
  Array<String> axes;
  if (auto work_info = func->GetAttr<Map<String, Any>>("blackhole.work_decomposition")) {
    if (auto work_axes = work_info.value().Get("axes")) {
      for (const auto& axis_any : Downcast<Array<Any>>(work_axes.value())) {
        axes.push_back(Downcast<String>(axis_any));
      }
    }
  }
  return axes;
}

bool HasDerivedIndices(const SemanticProgram& program, const tir::PrimFunc& func) {
  if (!program->domains.empty()) {
    for (const String& trait : program->domains[0]->traits) {
      if (static_cast<std::string>(trait) == "derived_indices") {
        return true;
      }
    }
  }
  if (auto work_info = func->GetAttr<Map<String, Any>>("blackhole.work_decomposition")) {
    if (auto derived = work_info.value().Get("derived_index_exprs")) {
      return !Downcast<Array<Any>>(derived.value()).empty();
    }
  }
  return false;
}

std::optional<Array<Any>> GetPipelineStagesFromSupplements(const SemanticProgram& program) {
  for (const SemanticSupplement& supplement : program->supplements) {
    if (static_cast<std::string>(supplement->kind) !=
        ToString(SupplementKind::kPipelineStructure)) {
      continue;
    }
    if (auto pipeline_stages = supplement->payload.Get(String(schema_key::kPipelineStages))) {
      return Downcast<Array<Any>>(pipeline_stages.value());
    }
  }
  return std::nullopt;
}

void AppendPipelineResourceIntent(const std::string& member_func, const SemanticProgram& program,
                                  Array<ResourceIntent>* resource_intents) {
  auto pipeline_stages = GetPipelineStagesFromSupplements(program);
  if (!pipeline_stages.has_value() || pipeline_stages->empty()) {
    return;
  }
  Map<String, Any> payload;
  payload.Set(String(schema_key::kPipelineStages), pipeline_stages.value());
  resource_intents->push_back(ResourceIntent(
      String("pipeline_contract_" + member_func), String("synchronization_support"),
      String(member_func), MakeTraits({"phase_b", "pipeline_contract"}), std::move(payload),
      MakeAnchors("spatial_resource_intent", "pipeline_contract_" + member_func)));
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

std::string CoreTypeTraitForSegmentKind(const std::string& segment_kind) {
  if (segment_kind == "reader") {
    return "brisc";
  }
  if (segment_kind == "compute") {
    return "trisc";
  }
  if (segment_kind == "writer") {
    return "ncrisc";
  }
  return "";
}

bool IsSimpleCopyFastPath(const SemanticProgram& program, const tir::PrimFunc& func) {
  if (func->GetAttr<Array<Any>>("blackhole.segment_plan")) {
    return false;
  }
  if (!program->states.empty() || program->updates.size() != 1) {
    return false;
  }
  auto kind = ParseUpdateLawKind(static_cast<std::string>(program->updates[0]->law->kind));
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
    auto role = ParseStateRole(static_cast<std::string>(program->states[0]->role));
    if (!role || *role != StateRole::kTransient) {
      return false;
    }
  }
  return true;
}

bool NeedsMultiPhase(const SemanticProgram& program) {
  for (const Update& update : program->updates) {
    auto kind = ParseUpdateLawKind(static_cast<std::string>(update->law->kind));
    if (kind && *kind != UpdateLawKind::kMap) {
      return true;
    }
  }
  for (const State& state : program->states) {
    auto role = ParseStateRole(static_cast<std::string>(state->role));
    if (role && (*role == StateRole::kCarry || *role == StateRole::kReductionAccumulator ||
                 *role == StateRole::kSelectionState || *role == StateRole::kIndexState)) {
      return true;
    }
  }
  return false;
}

std::string GenericTaskKindForUpdate(const Update& update) {
  auto kind = ParseUpdateLawKind(static_cast<std::string>(update->law->kind));
  if (!kind) {
    return "compute";
  }
  switch (*kind) {
    case UpdateLawKind::kMap:
      return "compute";
    case UpdateLawKind::kReduce:
      return "collective";
    case UpdateLawKind::kSelect:
      return "control";
    case UpdateLawKind::kRecurrence:
      return "control";
  }
  return "compute";
}

std::string GenericPhaseNameForUpdate(const Update& update, bool multi_phase) {
  if (!multi_phase) {
    return "phase0_compute";
  }
  auto kind = ParseUpdateLawKind(static_cast<std::string>(update->law->kind));
  if (kind && *kind == UpdateLawKind::kRecurrence) {
    return "phase1_stateful";
  }
  return "phase0_compute";
}

void BuildCommonSpatialScaffolding(const std::string& member_func, const Array<String>& work_axes,
                                   bool has_derived_indices, Array<SpatialLayout>* layouts,
                                   Array<WorkPartition>* work_partitions) {
  const std::string layout_kind = has_derived_indices ? "indexed" : "regular";
  const std::string partition_kind = work_axes.size() > 1 ? "blocked" : "replicated";
  layouts->push_back(SpatialLayout(String("layout_" + member_func), String(layout_kind),
                                   String(member_func), work_axes,
                                   MakeTraits({"phase_b"}),
                                   MakeAnchors("spatial_layout", member_func)));
  work_partitions->push_back(WorkPartition(String("partition_" + member_func), String(partition_kind),
                                           String(member_func), work_axes,
                                           MakeTraits({"phase_b"}),
                                           MakeAnchors("spatial_partition", member_func)));
}

SpatialProgramBundle BuildCopyFastPath(const std::string& member_func,
                                       const SemanticProgram& program,
                                       const Array<String>& work_axes,
                                       bool has_derived_indices) {
  Array<Task> tasks{
      Task(String("copy"), String("transfer"), String("phase0_copy"),
           Array<String>{String(program->updates[0]->name)}, MakeTraits({"fast_path", "copy"}),
           MakeAnchors("spatial_task", "copy"))};
  Array<Channel> channels{
      Channel(String("copy_tensor"), String("tensor_flow"), String("copy"), String("copy"),
              String(), MakeTraits({"fast_path", "copy"}),
              MakeAnchors("spatial_channel", "copy_tensor"))};
  Array<ProgramPhase> phases{
      ProgramPhase(String("phase0_copy"), Array<String>{String("copy")},
                   Array<String>{String("copy_tensor")}, MakeTraits({"fast_path", "copy"}),
                   MakeAnchors("spatial_phase", "phase0_copy"))};
  Array<SpatialLayout> layouts;
  Array<WorkPartition> work_partitions;
  Array<Placement> placements{
      Placement(String("place_copy"), String("execution"), String("copy"), String(member_func),
                MakeTraits({"fast_path", "copy"}), MakeAnchors("spatial_placement", "copy"))};
  Array<SyncEdge> sync_edges;
  Array<ResourceIntent> resource_intents{
      ResourceIntent(String("copy_buffer"), String("buffer"), String("copy"),
                     MakeTraits({"fast_path", "copy"}), EmptyPayload(),
                     MakeAnchors("spatial_resource_intent", "copy_buffer"))};
  AppendPipelineResourceIntent(member_func, program, &resource_intents);
  BuildCommonSpatialScaffolding(member_func, work_axes, has_derived_indices, &layouts,
                                &work_partitions);
  return {SpatialProgram(String(member_func), phases, tasks, channels, layouts,
                         work_partitions, placements, sync_edges, resource_intents,
                         MakeAnchors("spatial_program", member_func)),
          phases};
}

SpatialProgramBundle BuildGemmFastPath(const std::string& member_func,
                                       const SemanticProgram& program,
                                       const tir::PrimFunc& func,
                                       const Array<String>& work_axes,
                                       bool has_derived_indices) {
  Array<Task> tasks;
  Array<Placement> placements;
  Array<String> task_names;
  for (const std::string& segment_name : CollectSegmentKindsFromBody(func->body)) {
    const std::string task_kind = segment_name == "compute" ? "compute" : "transfer";
    Array<String> update_names;
    if (!program->updates.empty()) {
      update_names.push_back(program->updates[0]->name);
    }
    tasks.push_back(Task(String(segment_name), String(task_kind), String("phase0_gemm"),
                         update_names, MakeTraits({"fast_path", "gemm"}),
                         MakeAnchors("spatial_task", segment_name)));
    task_names.push_back(String(segment_name));
    std::vector<std::string> placement_traits{"fast_path", "gemm"};
    const std::string core_type_trait = CoreTypeTraitForSegmentKind(segment_name);
    if (!core_type_trait.empty()) {
      placement_traits.push_back(core_type_trait);
    }
    placements.push_back(Placement(String("place_" + segment_name), String("execution"),
                                   String(segment_name), String(member_func),
                                   ToStringArray(placement_traits),
                                   MakeAnchors("spatial_placement", segment_name)));
  }
  Array<Channel> channels{
      Channel(String("a_tiles"), String("tensor_flow"), String("reader"), String("compute"),
              String("A"), MakeTraits({"fast_path", "gemm"}),
              MakeAnchors("spatial_channel", "a_tiles")),
      Channel(String("b_tiles"), String("tensor_flow"), String("reader"), String("compute"),
              String("B"), MakeTraits({"fast_path", "gemm"}),
              MakeAnchors("spatial_channel", "b_tiles")),
      Channel(String("c_tiles"), String("tensor_flow"), String("compute"), String("writer"),
              String(program->states.empty() ? "" : static_cast<std::string>(program->states[0]->name)),
              MakeTraits({"fast_path", "gemm"}),
              MakeAnchors("spatial_channel", "c_tiles"))};
  Array<ProgramPhase> phases{
      ProgramPhase(String("phase0_gemm"), task_names,
                   Array<String>{String("a_tiles"), String("b_tiles"), String("c_tiles")},
                   MakeTraits({"fast_path", "gemm"}),
                   MakeAnchors("spatial_phase", "phase0_gemm"))};
  Array<SpatialLayout> layouts;
  Array<WorkPartition> work_partitions;
  Array<SyncEdge> sync_edges{
      SyncEdge(String("reader_to_compute"), String("dependency"), String("reader"),
               String("compute"), MakeTraits({"fast_path", "gemm"}),
               MakeAnchors("spatial_sync", "reader_to_compute")),
      SyncEdge(String("compute_to_writer"), String("dependency"), String("compute"),
               String("writer"), MakeTraits({"fast_path", "gemm"}),
               MakeAnchors("spatial_sync", "compute_to_writer"))};
  Array<ResourceIntent> resource_intents{
      ResourceIntent(String("gemm_input_buffers"), String("buffer"), String("reader"),
                     MakeTraits({"fast_path", "gemm"}), EmptyPayload(),
                     MakeAnchors("spatial_resource_intent", "gemm_input_buffers")),
      ResourceIntent(String("gemm_accumulator"), String("state_residency"),
                     String(program->states.empty() ? "" : static_cast<std::string>(program->states[0]->name)),
                     MakeTraits({"fast_path", "gemm"}), EmptyPayload(),
                     MakeAnchors("spatial_resource_intent", "gemm_accumulator"))};
  AppendPipelineResourceIntent(member_func, program, &resource_intents);
  BuildCommonSpatialScaffolding(member_func, work_axes, has_derived_indices, &layouts,
                                &work_partitions);
  return {SpatialProgram(String(member_func), phases, tasks, channels, layouts,
                         work_partitions, placements, sync_edges, resource_intents,
                         MakeAnchors("spatial_program", member_func)),
          phases};
}

SpatialProgramBundle BuildGenericSpatialProgram(const std::string& member_func,
                                                const SemanticProgram& program,
                                                const Array<String>& work_axes,
                                                bool has_derived_indices) {
  const bool multi_phase = NeedsMultiPhase(program);
  const std::vector<std::string> phase_order =
      multi_phase ? std::vector<std::string>{"phase0_compute", "phase1_stateful"}
                  : std::vector<std::string>{"phase0_compute"};

  Array<Task> tasks;
  Array<Placement> placements;
  std::unordered_map<std::string, std::vector<std::string>> phase_to_tasks;
  std::unordered_map<std::string, Task> tasks_by_update;
  std::unordered_set<std::string> known_task_names;
  for (const Update& update : program->updates) {
    const std::string update_name = static_cast<std::string>(update->name);
    const std::string state_name = static_cast<std::string>(update->state_name);
    const auto law_kind = ParseUpdateLawKind(static_cast<std::string>(update->law->kind));
    if (update_name == "root_map" && state_name.empty() && law_kind &&
        *law_kind == UpdateLawKind::kMap && program->updates.size() > 1) {
      continue;
    }
    const std::string phase_name = GenericPhaseNameForUpdate(update, multi_phase);
    const std::string task_kind = GenericTaskKindForUpdate(update);
    Task task(String(update_name), String(task_kind), String(phase_name),
              Array<String>{update->name},
              Array<String>{String("phase_b"), String(static_cast<std::string>(update->law->kind))},
              MakeAnchors("spatial_task", update_name));
    tasks.push_back(task);
    placements.push_back(Placement(String("place_" + update_name), String("execution"),
                                   String(update_name), String(member_func),
                                   MakeTraits({"phase_b"}), MakeAnchors("spatial_placement", update_name)));
    phase_to_tasks[phase_name].push_back(update_name);
    tasks_by_update[update_name] = task;
    known_task_names.insert(update_name);
  }

  if (tasks.empty() && !program->updates.empty()) {
    const std::string update_name = static_cast<std::string>(program->updates[0]->name);
    Task task(String(update_name), String("compute"), String("phase0_compute"),
              Array<String>{program->updates[0]->name}, MakeTraits({"phase_b"}),
              MakeAnchors("spatial_task", update_name));
    tasks.push_back(task);
    placements.push_back(Placement(String("place_" + update_name), String("execution"),
                                   String(update_name), String(member_func),
                                   MakeTraits({"phase_b"}), MakeAnchors("spatial_placement", update_name)));
    phase_to_tasks["phase0_compute"].push_back(update_name);
    tasks_by_update[update_name] = task;
    known_task_names.insert(update_name);
  }

  Array<Channel> channels;
  std::unordered_map<std::string, std::vector<std::string>> phase_to_channels;
  std::unordered_map<std::string, std::string> version_to_producer_task;
  for (const StateDef& def : program->state_defs) {
    const std::string producer_update = static_cast<std::string>(def->producer_update);
    if (known_task_names.count(producer_update)) {
      version_to_producer_task[static_cast<std::string>(def->version_name)] = producer_update;
    }
  }
  std::unordered_set<std::string> seen_channel_keys;
  for (const StateUse& use : program->state_uses) {
    const std::string consumer_update = static_cast<std::string>(use->consumer_update);
    const std::string version_name = static_cast<std::string>(use->version_name);
    if (!known_task_names.count(consumer_update) || !version_to_producer_task.count(version_name)) {
      continue;
    }
    const std::string source_task = version_to_producer_task.at(version_name);
    const std::string state_name = static_cast<std::string>(use->state_name);
    const std::string channel_key = source_task + "->" + consumer_update + ":" + state_name;
    if (!seen_channel_keys.insert(channel_key).second) {
      continue;
    }
    const std::string channel_name = "channel_" + state_name + "_" + consumer_update;
    channels.push_back(Channel(String(channel_name), String("state_flow"), String(source_task),
                               String(consumer_update), String(state_name),
                               MakeTraits({"phase_b"}), MakeAnchors("spatial_channel", channel_name)));
    const std::string phase_name = static_cast<std::string>(tasks_by_update[consumer_update]->phase_name);
    phase_to_channels[phase_name].push_back(channel_name);
  }

  if (channels.empty() && multi_phase && !phase_to_tasks["phase0_compute"].empty() &&
      !phase_to_tasks["phase1_stateful"].empty()) {
    const std::string channel_name = "channel_phase_boundary";
    channels.push_back(Channel(String(channel_name), String("phase_boundary"),
                               String(phase_to_tasks["phase0_compute"].front()),
                               String(phase_to_tasks["phase1_stateful"].front()), String(),
                               MakeTraits({"phase_boundary"}),
                               MakeAnchors("spatial_channel", channel_name)));
    phase_to_channels["phase1_stateful"].push_back(channel_name);
  }

  Array<ProgramPhase> phases;
  for (const auto& phase_name : phase_order) {
    auto task_it = phase_to_tasks.find(phase_name);
    if (task_it == phase_to_tasks.end() || task_it->second.empty()) {
      continue;
    }
    const auto channel_it = phase_to_channels.find(phase_name);
    phases.push_back(ProgramPhase(String(phase_name), ToStringArray(task_it->second),
                                  channel_it == phase_to_channels.end()
                                      ? Array<String>{}
                                      : ToStringArray(channel_it->second),
                                  ToStringArray(multi_phase ? std::vector<std::string>{"phase_b", "multi_phase"}
                                                            : std::vector<std::string>{"phase_b"}),
                                  MakeAnchors("spatial_phase", phase_name)));
  }

  Array<SpatialLayout> layouts;
  Array<WorkPartition> work_partitions;
  BuildCommonSpatialScaffolding(member_func, work_axes, has_derived_indices, &layouts,
                                &work_partitions);

  Array<SyncEdge> sync_edges;
  if (multi_phase && phase_to_tasks.count("phase0_compute") &&
      phase_to_tasks.count("phase1_stateful") &&
      !phase_to_tasks["phase0_compute"].empty() &&
      !phase_to_tasks["phase1_stateful"].empty()) {
    sync_edges.push_back(SyncEdge(String("phase0_to_phase1"), String("completion"),
                                  String(phase_to_tasks["phase0_compute"].front()),
                                  String(phase_to_tasks["phase1_stateful"].front()),
                                  MakeTraits({"phase_boundary"}),
                                  MakeAnchors("spatial_sync", "phase0_to_phase1")));
  }

  Array<ResourceIntent> resource_intents;
  for (const State& state : program->states) {
    const auto role = ParseStateRole(static_cast<std::string>(state->role));
    const std::string state_name = static_cast<std::string>(state->name);
    const bool is_stateful = role && (*role == StateRole::kCarry ||
                                      *role == StateRole::kReductionAccumulator ||
                                      *role == StateRole::kSelectionState ||
                                      *role == StateRole::kIndexState);
    resource_intents.push_back(ResourceIntent(
        String("intent_" + state_name), String(is_stateful ? "state_residency" : "buffer"),
        state->name,
        Array<String>{String(static_cast<std::string>(state->role)),
                      String(static_cast<std::string>(state->storage_scope))},
        EmptyPayload(),
        MakeAnchors("spatial_resource_intent", state_name)));
    if (multi_phase && is_stateful) {
      resource_intents.push_back(ResourceIntent(
          String("phase_boundary_" + state_name), String("phase_boundary_materialization"),
          state->name, MakeTraits({"phase_boundary"}), EmptyPayload(),
          MakeAnchors("spatial_resource_intent", "phase_boundary_" + state_name)));
    }
  }
  AppendPipelineResourceIntent(member_func, program, &resource_intents);

  return {SpatialProgram(String(member_func), phases, tasks, channels, layouts,
                         work_partitions, placements, sync_edges, resource_intents,
                         MakeAnchors("spatial_program", member_func)),
          phases};
}

SpatialProgramBundle BuildSpatialProgramForFunc(const std::string& member_func,
                                                const SemanticProgram& program,
                                                const tir::PrimFunc& func) {
  Array<String> work_axes = GetWorkAxes(program, func);
  const bool has_derived_indices = HasDerivedIndices(program, func);
  if (IsSimpleCopyFastPath(program, func)) {
    return BuildCopyFastPath(member_func, program, work_axes, has_derived_indices);
  }
  if (IsSimpleGemmFastPath(program, func)) {
    return BuildGemmFastPath(member_func, program, func, work_axes, has_derived_indices);
  }
  return BuildGenericSpatialProgram(member_func, program, work_axes, has_derived_indices);
}

}  // namespace

tvm::transform::Pass LowerToSpatialProgram() {
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
      auto maybe_semantic = func.value()->GetAttr<SemanticProgram>(attr::kTLSemanticProgram);
      if (!maybe_semantic) {
        continue;
      }
      const std::string member_func = GetMemberFuncName(gvar, func.value());
      SpatialProgramBundle spatial = BuildSpatialProgramForFunc(member_func, maybe_semantic.value(),
                                                                func.value());
      phases_by_member_func[member_func] = spatial.phases;
      tir::PrimFunc updated_func = func.value();
      Map<String, Any> attrs = updated_func->attrs.defined() ? updated_func->attrs->dict
                                                             : Map<String, Any>();
      attrs.Set(attr::kTLSpatialProgram, spatial.program);
      updated_func.CopyOnWrite()->attrs = DictAttrs(attrs);
      updates->Add(gvar, updated_func);
    }

    mod->Update(updates);
    if (auto maybe_registry = mod->global_infos.Get(attr::kTLDevicePrograms)) {
      Array<GlobalInfo> rebuilt_registry;
      for (const GlobalInfo& info : maybe_registry.value()) {
        auto program_info = Downcast<TLDeviceProgramInfo>(info);
        Array<ProgramPhase> phases;
        for (const String& member_func : program_info->member_funcs) {
          auto it = phases_by_member_func.find(static_cast<std::string>(member_func));
          if (it == phases_by_member_func.end()) {
            continue;
          }
          for (const ProgramPhase& phase : it->second) {
            phases.push_back(phase);
          }
        }
        if (phases.empty() && program_info->member_funcs.size() == 1) {
          auto root_it = phases_by_member_func.find(static_cast<std::string>(program_info->root_symbol));
          if (root_it != phases_by_member_func.end()) {
            for (const ProgramPhase& phase : root_it->second) {
              phases.push_back(phase);
            }
          }
        }
        rebuilt_registry.push_back(
            TLDeviceProgramInfo(program_info->root_symbol, program_info->member_funcs, phases));
      }
      mod = mod->ShallowCopy();
      mod->UpdateGlobalInfo(attr::kTLDevicePrograms, rebuilt_registry);
    }
    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tl.transform.LowerToSpatialProgram", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerToSpatialProgram", LowerToSpatialProgram);
}

}  // namespace tl
}  // namespace tvm
