/*!
 * \file spatial_program.cc
 * \brief Stage 4 Phase B spatial companion objects and capability snapshot.
 */

#include "spatial_program.h"

namespace tvm {
namespace tl {

namespace {

int64_t GetIntFieldOrDefault(const ffi::Map<ffi::String, ffi::Any>& payload, const char* key,
                             int64_t default_value = -1) {
  if (auto value = payload.Get(ffi::String(key))) {
    return Downcast<Integer>(value.value())->value;
  }
  return default_value;
}

ffi::String GetStringFieldOrDefault(const ffi::Map<ffi::String, ffi::Any>& payload, const char* key,
                                    ffi::String default_value = ffi::String()) {
  if (auto value = payload.Get(ffi::String(key))) {
    return Downcast<ffi::String>(value.value());
  }
  return default_value;
}

ffi::Array<Integer> GetIntegerArrayFieldOrEmpty(const ffi::Map<ffi::String, ffi::Any>& payload,
                                                const char* key) {
  ffi::Array<Integer> result;
  if (auto value = payload.Get(ffi::String(key))) {
    for (const ffi::Any& item : Downcast<ffi::Array<ffi::Any>>(value.value())) {
      result.push_back(Downcast<Integer>(item));
    }
  }
  return result;
}

}  // namespace

Task::Task(ffi::String name, ffi::String kind, ffi::String phase_name,
           ffi::Array<ffi::String> update_names, ffi::Array<ffi::String> traits,
           ffi::Map<ffi::String, ffi::Any> payload, ffi::Array<TIRAnchor> anchors) {
  auto n = ffi::make_object<TaskNode>();
  n->name = std::move(name);
  n->kind = std::move(kind);
  n->phase_name = std::move(phase_name);
  n->phase_index = GetIntFieldOrDefault(payload, schema_key::kPhaseIndex);
  n->execution_role = GetStringFieldOrDefault(payload, schema_key::kExecutionRole);
  n->formation_basis = GetStringFieldOrDefault(payload, schema_key::kFormationBasis);
  n->update_names = std::move(update_names);
  n->traits = std::move(traits);
  n->payload = std::move(payload);
  n->anchors = std::move(anchors);
  data_ = std::move(n);
}

Channel::Channel(ffi::String name, ffi::String kind, ffi::String source_task,
                 ffi::String target_task, ffi::String state_name,
                 ffi::Array<ffi::String> traits, ffi::Map<ffi::String, ffi::Any> payload,
                 ffi::Array<TIRAnchor> anchors) {
  auto n = ffi::make_object<ChannelNode>();
  n->name = std::move(name);
  n->kind = std::move(kind);
  n->source_task = std::move(source_task);
  n->target_task = std::move(target_task);
  n->source_task_index = GetIntFieldOrDefault(payload, schema_key::kSourceTaskIndex);
  n->target_task_index = GetIntFieldOrDefault(payload, schema_key::kTargetTaskIndex);
  n->payload_kind = GetStringFieldOrDefault(payload, schema_key::kPayloadKind);
  n->delivery_kind = GetStringFieldOrDefault(payload, schema_key::kDeliveryKind);
  n->state_name = std::move(state_name);
  n->state_index = GetIntFieldOrDefault(payload, schema_key::kStateIndex);
  n->source_version = GetStringFieldOrDefault(payload, schema_key::kSourceVersion);
  n->target_version = GetStringFieldOrDefault(payload, schema_key::kTargetVersion);
  n->traits = std::move(traits);
  n->payload = std::move(payload);
  n->anchors = std::move(anchors);
  data_ = std::move(n);
}

SpatialLayout::SpatialLayout(ffi::String name, ffi::String kind, ffi::String target_name,
                             ffi::Array<ffi::String> axes, ffi::Array<ffi::String> traits,
                             ffi::Map<ffi::String, ffi::Any> payload,
                             ffi::Array<TIRAnchor> anchors) {
  auto n = ffi::make_object<SpatialLayoutNode>();
  n->name = std::move(name);
  n->kind = std::move(kind);
  n->target_name = std::move(target_name);
  n->domain_index = GetIntFieldOrDefault(payload, schema_key::kDomainIndex);
  n->domain_transform_kind = GetStringFieldOrDefault(payload, schema_key::kDomainTransformKind);
  n->axes = std::move(axes);
  n->traits = std::move(traits);
  n->payload = std::move(payload);
  n->anchors = std::move(anchors);
  data_ = std::move(n);
}

WorkPartition::WorkPartition(ffi::String name, ffi::String kind, ffi::String target_name,
                             ffi::Array<ffi::String> axes, ffi::Array<ffi::String> traits,
                             ffi::Map<ffi::String, ffi::Any> payload,
                             ffi::Array<TIRAnchor> anchors) {
  auto n = ffi::make_object<WorkPartitionNode>();
  n->name = std::move(name);
  n->kind = std::move(kind);
  n->target_name = std::move(target_name);
  n->domain_index = GetIntFieldOrDefault(payload, schema_key::kDomainIndex);
  n->partition_family = GetStringFieldOrDefault(payload, schema_key::kPartitionFamily);
  n->axes = std::move(axes);
  n->traits = std::move(traits);
  n->payload = std::move(payload);
  n->anchors = std::move(anchors);
  data_ = std::move(n);
}

Placement::Placement(ffi::String name, ffi::String kind, ffi::String task_name,
                     ffi::String member_func, ffi::Array<ffi::String> traits,
                     ffi::Map<ffi::String, ffi::Any> payload,
                     ffi::Array<TIRAnchor> anchors) {
  auto n = ffi::make_object<PlacementNode>();
  n->name = std::move(name);
  n->kind = std::move(kind);
  n->task_name = std::move(task_name);
  n->task_index = GetIntFieldOrDefault(payload, schema_key::kTaskIndex);
  n->member_func = std::move(member_func);
  n->affinity_kind = GetStringFieldOrDefault(payload, schema_key::kAffinityKind);
  n->obligation_kind = GetStringFieldOrDefault(payload, schema_key::kObligationKind);
  n->placement_domain = GetStringFieldOrDefault(payload, schema_key::kPlacementDomain);
  n->traits = std::move(traits);
  n->payload = std::move(payload);
  n->anchors = std::move(anchors);
  data_ = std::move(n);
}

SyncEdge::SyncEdge(ffi::String name, ffi::String kind, ffi::String source, ffi::String target,
                   ffi::Array<ffi::String> traits, ffi::Map<ffi::String, ffi::Any> payload,
                   ffi::Array<TIRAnchor> anchors) {
  auto n = ffi::make_object<SyncEdgeNode>();
  n->name = std::move(name);
  n->kind = std::move(kind);
  n->source = std::move(source);
  n->target = std::move(target);
  n->source_task_index = GetIntFieldOrDefault(payload, schema_key::kSourceTaskIndex);
  n->target_task_index = GetIntFieldOrDefault(payload, schema_key::kTargetTaskIndex);
  n->ordering_kind = GetStringFieldOrDefault(payload, schema_key::kOrderingKind);
  n->materialization_kind = GetStringFieldOrDefault(payload, schema_key::kMaterializationKind);
  n->traits = std::move(traits);
  n->payload = std::move(payload);
  n->anchors = std::move(anchors);
  data_ = std::move(n);
}

ResourceIntent::ResourceIntent(ffi::String name, ffi::String kind, ffi::String target_name,
                               ffi::Array<ffi::String> traits,
                               ffi::Map<ffi::String, ffi::Any> payload,
                               ffi::Array<TIRAnchor> anchors) {
  auto n = ffi::make_object<ResourceIntentNode>();
  n->name = std::move(name);
  n->kind = std::move(kind);
  n->target_name = std::move(target_name);
  n->target_kind = GetStringFieldOrDefault(payload, schema_key::kTargetKind);
  n->target_index = GetIntFieldOrDefault(payload, schema_key::kTargetIndex);
  n->traits = std::move(traits);
  n->payload = std::move(payload);
  n->anchors = std::move(anchors);
  data_ = std::move(n);
}

ProgramPhase::ProgramPhase(ffi::String name, ffi::Array<ffi::String> task_names,
                           ffi::Array<ffi::String> channel_names,
                           ffi::Array<ffi::String> traits,
                           ffi::Map<ffi::String, ffi::Any> payload,
                           ffi::Array<TIRAnchor> anchors) {
  auto n = ffi::make_object<ProgramPhaseNode>();
  n->name = std::move(name);
  n->phase_index = GetIntFieldOrDefault(payload, schema_key::kPhaseIndex);
  n->task_indices = GetIntegerArrayFieldOrEmpty(payload, schema_key::kTaskIndices);
  n->channel_indices = GetIntegerArrayFieldOrEmpty(payload, schema_key::kChannelIndices);
  n->closure_basis = GetStringFieldOrDefault(payload, schema_key::kClosureBasis);
  n->task_names = std::move(task_names);
  n->channel_names = std::move(channel_names);
  n->traits = std::move(traits);
  n->payload = std::move(payload);
  n->anchors = std::move(anchors);
  data_ = std::move(n);
}

SpatialDomainPlan::SpatialDomainPlan(ffi::String member_func, ffi::Array<SpatialLayout> layouts,
                                     ffi::Array<WorkPartition> work_partitions,
                                     ffi::Array<TIRAnchor> anchors) {
  auto n = ffi::make_object<SpatialDomainPlanNode>();
  n->member_func = std::move(member_func);
  n->layouts = std::move(layouts);
  n->work_partitions = std::move(work_partitions);
  n->anchors = std::move(anchors);
  data_ = std::move(n);
}

SpatialExecutionPlan::SpatialExecutionPlan(ffi::String member_func, ffi::Array<ProgramPhase> phases,
                                           ffi::Array<Task> tasks, ffi::Array<Channel> channels,
                                           ffi::Array<Placement> placements,
                                           ffi::Array<SyncEdge> sync_edges,
                                           ffi::Array<ResourceIntent> resource_intents,
                                           ffi::Array<TIRAnchor> anchors) {
  auto n = ffi::make_object<SpatialExecutionPlanNode>();
  n->member_func = std::move(member_func);
  n->phases = std::move(phases);
  n->tasks = std::move(tasks);
  n->channels = std::move(channels);
  n->placements = std::move(placements);
  n->sync_edges = std::move(sync_edges);
  n->resource_intents = std::move(resource_intents);
  n->anchors = std::move(anchors);
  data_ = std::move(n);
}

SpatialProgram::SpatialProgram(ffi::String member_func, ffi::Array<ProgramPhase> phases,
                               ffi::Array<Task> tasks, ffi::Array<Channel> channels,
                               ffi::Array<SpatialLayout> layouts,
                               ffi::Array<WorkPartition> work_partitions,
                               ffi::Array<Placement> placements,
                               ffi::Array<SyncEdge> sync_edges,
                               ffi::Array<ResourceIntent> resource_intents,
                               ffi::Array<TIRAnchor> anchors) {
  auto n = ffi::make_object<SpatialProgramNode>();
  n->member_func = std::move(member_func);
  n->phases = std::move(phases);
  n->tasks = std::move(tasks);
  n->channels = std::move(channels);
  n->layouts = std::move(layouts);
  n->work_partitions = std::move(work_partitions);
  n->placements = std::move(placements);
  n->sync_edges = std::move(sync_edges);
  n->resource_intents = std::move(resource_intents);
  n->anchors = std::move(anchors);
  data_ = std::move(n);
}

TLDeviceProgramInfo::TLDeviceProgramInfo(ffi::String root_symbol, ffi::Array<ffi::String> member_funcs,
                                         ffi::Array<ProgramPhase> phases) {
  auto n = ffi::make_object<TLDeviceProgramInfoNode>();
  n->root_symbol = std::move(root_symbol);
  n->member_funcs = std::move(member_funcs);
  n->phases = std::move(phases);
  data_ = std::move(n);
}

SpatialCapabilityModel::SpatialCapabilityModel(
    ffi::String arch_name, ffi::String topology_class, ffi::String placement_domain,
    int64_t logical_worker_grid_x, int64_t logical_worker_grid_y, int64_t functional_worker_count,
    int64_t router_only_count, int64_t dram_view_count, int64_t worker_l1_size,
    int64_t dram_view_size, ffi::Array<ffi::String> supported_flow_kinds,
    ffi::Array<ffi::String> supported_payload_kinds,
    ffi::Array<ffi::String> supported_delivery_kinds,
    ffi::Array<ffi::String> supported_sync_kinds,
    ffi::Array<ffi::String> supported_ordering_kinds,
    ffi::Array<ffi::String> supported_materialization_kinds,
    ffi::Array<ffi::String> supported_layout_kinds,
    ffi::Array<ffi::String> supported_partition_kinds,
    ffi::Array<ffi::String> supported_resource_intent_kinds) {
  auto n = ffi::make_object<SpatialCapabilityModelNode>();
  n->arch_name = std::move(arch_name);
  n->topology_class = std::move(topology_class);
  n->placement_domain = std::move(placement_domain);
  n->logical_worker_grid_x = logical_worker_grid_x;
  n->logical_worker_grid_y = logical_worker_grid_y;
  n->functional_worker_count = functional_worker_count;
  n->router_only_count = router_only_count;
  n->dram_view_count = dram_view_count;
  n->worker_l1_size = worker_l1_size;
  n->dram_view_size = dram_view_size;
  n->supported_flow_kinds = std::move(supported_flow_kinds);
  n->supported_payload_kinds = std::move(supported_payload_kinds);
  n->supported_delivery_kinds = std::move(supported_delivery_kinds);
  n->supported_sync_kinds = std::move(supported_sync_kinds);
  n->supported_ordering_kinds = std::move(supported_ordering_kinds);
  n->supported_materialization_kinds = std::move(supported_materialization_kinds);
  n->supported_layout_kinds = std::move(supported_layout_kinds);
  n->supported_partition_kinds = std::move(supported_partition_kinds);
  n->supported_resource_intent_kinds = std::move(supported_resource_intent_kinds);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  TaskNode::RegisterReflection();
  ChannelNode::RegisterReflection();
  SpatialLayoutNode::RegisterReflection();
  WorkPartitionNode::RegisterReflection();
  PlacementNode::RegisterReflection();
  SyncEdgeNode::RegisterReflection();
  ResourceIntentNode::RegisterReflection();
  ProgramPhaseNode::RegisterReflection();
  SpatialDomainPlanNode::RegisterReflection();
  SpatialExecutionPlanNode::RegisterReflection();
  SpatialProgramNode::RegisterReflection();
  TLDeviceProgramInfoNode::RegisterReflection();
  SpatialCapabilityModelNode::RegisterReflection();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.Task",
                        [](ffi::String name, ffi::String kind, ffi::String phase_name,
                           ffi::Array<ffi::String> update_names,
                           ffi::Array<ffi::String> traits,
                           ffi::Map<ffi::String, ffi::Any> payload,
                           ffi::Array<TIRAnchor> anchors) {
                          return Task(std::move(name), std::move(kind), std::move(phase_name),
                                      std::move(update_names), std::move(traits),
                                      std::move(payload), std::move(anchors));
                        });
  refl::GlobalDef().def("tl.Channel",
                        [](ffi::String name, ffi::String kind, ffi::String source_task,
                           ffi::String target_task, ffi::String state_name,
                           ffi::Array<ffi::String> traits,
                           ffi::Map<ffi::String, ffi::Any> payload,
                           ffi::Array<TIRAnchor> anchors) {
                          return Channel(std::move(name), std::move(kind),
                                         std::move(source_task), std::move(target_task),
                                         std::move(state_name), std::move(traits),
                                         std::move(payload), std::move(anchors));
                        });
  refl::GlobalDef().def("tl.SpatialLayout",
                        [](ffi::String name, ffi::String kind, ffi::String target_name,
                           ffi::Array<ffi::String> axes, ffi::Array<ffi::String> traits,
                           ffi::Map<ffi::String, ffi::Any> payload,
                           ffi::Array<TIRAnchor> anchors) {
                          return SpatialLayout(std::move(name), std::move(kind),
                                               std::move(target_name), std::move(axes),
                                               std::move(traits), std::move(payload),
                                               std::move(anchors));
                        });
  refl::GlobalDef().def("tl.WorkPartition",
                        [](ffi::String name, ffi::String kind, ffi::String target_name,
                           ffi::Array<ffi::String> axes, ffi::Array<ffi::String> traits,
                           ffi::Map<ffi::String, ffi::Any> payload,
                           ffi::Array<TIRAnchor> anchors) {
                          return WorkPartition(std::move(name), std::move(kind),
                                               std::move(target_name), std::move(axes),
                                               std::move(traits), std::move(payload),
                                               std::move(anchors));
                        });
  refl::GlobalDef().def("tl.Placement",
                        [](ffi::String name, ffi::String kind, ffi::String task_name,
                           ffi::String member_func, ffi::Array<ffi::String> traits,
                           ffi::Map<ffi::String, ffi::Any> payload,
                           ffi::Array<TIRAnchor> anchors) {
                          return Placement(std::move(name), std::move(kind),
                                           std::move(task_name), std::move(member_func),
                                           std::move(traits), std::move(payload),
                                           std::move(anchors));
                        });
  refl::GlobalDef().def("tl.SyncEdge",
                        [](ffi::String name, ffi::String kind, ffi::String source,
                           ffi::String target, ffi::Array<ffi::String> traits,
                           ffi::Map<ffi::String, ffi::Any> payload,
                           ffi::Array<TIRAnchor> anchors) {
                          return SyncEdge(std::move(name), std::move(kind), std::move(source),
                                          std::move(target), std::move(traits),
                                          std::move(payload), std::move(anchors));
                        });
  refl::GlobalDef().def("tl.ResourceIntent",
                        [](ffi::String name, ffi::String kind, ffi::String target_name,
                           ffi::Array<ffi::String> traits,
                           ffi::Map<ffi::String, ffi::Any> payload,
                           ffi::Array<TIRAnchor> anchors) {
                          return ResourceIntent(std::move(name), std::move(kind),
                                                std::move(target_name), std::move(traits),
                                                std::move(payload), std::move(anchors));
                        });
  refl::GlobalDef().def("tl.ProgramPhase",
                        [](ffi::String name, ffi::Array<ffi::String> task_names,
                           ffi::Array<ffi::String> channel_names,
                           ffi::Array<ffi::String> traits,
                           ffi::Map<ffi::String, ffi::Any> payload,
                           ffi::Array<TIRAnchor> anchors) {
                          return ProgramPhase(std::move(name), std::move(task_names),
                                              std::move(channel_names), std::move(traits),
                                              std::move(payload), std::move(anchors));
                        });
  refl::GlobalDef().def("tl.SpatialDomainPlan",
                        [](ffi::String member_func, ffi::Array<SpatialLayout> layouts,
                           ffi::Array<WorkPartition> work_partitions,
                           ffi::Array<TIRAnchor> anchors) {
                          return SpatialDomainPlan(std::move(member_func), std::move(layouts),
                                                   std::move(work_partitions),
                                                   std::move(anchors));
                        });
  refl::GlobalDef().def("tl.SpatialExecutionPlan",
                        [](ffi::String member_func, ffi::Array<ProgramPhase> phases,
                           ffi::Array<Task> tasks, ffi::Array<Channel> channels,
                           ffi::Array<Placement> placements,
                           ffi::Array<SyncEdge> sync_edges,
                           ffi::Array<ResourceIntent> resource_intents,
                           ffi::Array<TIRAnchor> anchors) {
                          return SpatialExecutionPlan(
                              std::move(member_func), std::move(phases), std::move(tasks),
                              std::move(channels), std::move(placements),
                              std::move(sync_edges), std::move(resource_intents),
                              std::move(anchors));
                        });
  refl::GlobalDef().def("tl.SpatialProgram",
                        [](ffi::String member_func, ffi::Array<ProgramPhase> phases,
                           ffi::Array<Task> tasks, ffi::Array<Channel> channels,
                           ffi::Array<SpatialLayout> layouts,
                           ffi::Array<WorkPartition> work_partitions,
                           ffi::Array<Placement> placements,
                           ffi::Array<SyncEdge> sync_edges,
                           ffi::Array<ResourceIntent> resource_intents,
                           ffi::Array<TIRAnchor> anchors) {
                          return SpatialProgram(std::move(member_func), std::move(phases),
                                                std::move(tasks), std::move(channels),
                                                std::move(layouts),
                                                std::move(work_partitions),
                                                std::move(placements),
                                                std::move(sync_edges),
                                                std::move(resource_intents),
                                                std::move(anchors));
                        });
  refl::GlobalDef().def("tl.TLDeviceProgramInfo",
                        [](ffi::String root_symbol, ffi::Array<ffi::String> member_funcs) {
                          return TLDeviceProgramInfo(std::move(root_symbol),
                                                     std::move(member_funcs),
                                                     ffi::Array<ProgramPhase>{});
                        });
  refl::GlobalDef().def("tl.SpatialCapabilityModel",
                        [](ffi::String arch_name, ffi::String topology_class,
                           ffi::String placement_domain, int64_t logical_worker_grid_x,
                           int64_t logical_worker_grid_y, int64_t functional_worker_count,
                           int64_t router_only_count, int64_t dram_view_count,
                           int64_t worker_l1_size, int64_t dram_view_size,
                           ffi::Array<ffi::String> supported_flow_kinds,
                           ffi::Array<ffi::String> supported_payload_kinds,
                           ffi::Array<ffi::String> supported_delivery_kinds,
                           ffi::Array<ffi::String> supported_sync_kinds,
                           ffi::Array<ffi::String> supported_ordering_kinds,
                           ffi::Array<ffi::String> supported_materialization_kinds,
                           ffi::Array<ffi::String> supported_layout_kinds,
                           ffi::Array<ffi::String> supported_partition_kinds,
                           ffi::Array<ffi::String> supported_resource_intent_kinds) {
                          return SpatialCapabilityModel(
                              std::move(arch_name), std::move(topology_class),
                              std::move(placement_domain), logical_worker_grid_x,
                              logical_worker_grid_y, functional_worker_count, router_only_count,
                              dram_view_count, worker_l1_size, dram_view_size,
                              std::move(supported_flow_kinds),
                              std::move(supported_payload_kinds),
                              std::move(supported_delivery_kinds),
                              std::move(supported_sync_kinds),
                              std::move(supported_ordering_kinds),
                              std::move(supported_materialization_kinds),
                              std::move(supported_layout_kinds),
                              std::move(supported_partition_kinds),
                              std::move(supported_resource_intent_kinds));
                        });
}

}  // namespace tl
}  // namespace tvm
