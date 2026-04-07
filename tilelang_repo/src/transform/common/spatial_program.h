/*!
 * \file spatial_program.h
 * \brief Stage 4 Phase B spatial companion objects and capability snapshot.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_SPATIAL_PROGRAM_H_
#define TVM_TL_TRANSFORM_COMMON_SPATIAL_PROGRAM_H_

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/global_info.h>

#include "companion_base.h"

namespace tvm {
namespace tl {

class TaskNode : public Object {
 public:
  ffi::String name;
  ffi::String kind;
  ffi::String phase_name;
  int64_t phase_index = -1;
  ffi::String execution_role;
  ffi::String formation_basis;
  ffi::Array<ffi::String> update_names;
  ffi::Array<ffi::String> traits;
  ffi::Map<ffi::String, ffi::Any> payload;
  ffi::Array<TIRAnchor> anchors;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TaskNode>()
        .def_ro("name", &TaskNode::name)
        .def_ro("kind", &TaskNode::kind)
        .def_ro("phase_name", &TaskNode::phase_name)
        .def_ro("phase_index", &TaskNode::phase_index)
        .def_ro("execution_role", &TaskNode::execution_role)
        .def_ro("formation_basis", &TaskNode::formation_basis)
        .def_ro("update_names", &TaskNode::update_names)
        .def_ro("traits", &TaskNode::traits)
        .def_ro("payload", &TaskNode::payload)
        .def_ro("anchors", &TaskNode::anchors);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.Task", TaskNode, Object);
};

class Task : public ObjectRef {
 public:
  TVM_DLL Task(ffi::String name, ffi::String kind, ffi::String phase_name,
               ffi::Array<ffi::String> update_names, ffi::Array<ffi::String> traits,
               ffi::Map<ffi::String, ffi::Any> payload, ffi::Array<TIRAnchor> anchors);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Task, ObjectRef, TaskNode);
};

class ChannelNode : public Object {
 public:
  ffi::String name;
  ffi::String kind;
  ffi::String source_task;
  ffi::String target_task;
  int64_t source_task_index = -1;
  int64_t target_task_index = -1;
  ffi::String payload_kind;
  ffi::String delivery_kind;
  ffi::String state_name;
  int64_t state_index = -1;
  ffi::String source_version;
  ffi::String target_version;
  ffi::Array<ffi::String> traits;
  ffi::Map<ffi::String, ffi::Any> payload;
  ffi::Array<TIRAnchor> anchors;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ChannelNode>()
        .def_ro("name", &ChannelNode::name)
        .def_ro("kind", &ChannelNode::kind)
        .def_ro("source_task", &ChannelNode::source_task)
        .def_ro("target_task", &ChannelNode::target_task)
        .def_ro("source_task_index", &ChannelNode::source_task_index)
        .def_ro("target_task_index", &ChannelNode::target_task_index)
        .def_ro("payload_kind", &ChannelNode::payload_kind)
        .def_ro("delivery_kind", &ChannelNode::delivery_kind)
        .def_ro("state_name", &ChannelNode::state_name)
        .def_ro("state_index", &ChannelNode::state_index)
        .def_ro("source_version", &ChannelNode::source_version)
        .def_ro("target_version", &ChannelNode::target_version)
        .def_ro("traits", &ChannelNode::traits)
        .def_ro("payload", &ChannelNode::payload)
        .def_ro("anchors", &ChannelNode::anchors);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.Channel", ChannelNode, Object);
};

class Channel : public ObjectRef {
 public:
  TVM_DLL Channel(ffi::String name, ffi::String kind, ffi::String source_task,
                  ffi::String target_task, ffi::String state_name,
                  ffi::Array<ffi::String> traits, ffi::Map<ffi::String, ffi::Any> payload,
                  ffi::Array<TIRAnchor> anchors);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Channel, ObjectRef, ChannelNode);
};

class SpatialLayoutNode : public Object {
 public:
  ffi::String name;
  ffi::String kind;
  ffi::String target_name;
  int64_t domain_index = -1;
  ffi::String domain_transform_kind;
  ffi::Array<ffi::String> axes;
  ffi::Array<ffi::String> traits;
  ffi::Map<ffi::String, ffi::Any> payload;
  ffi::Array<TIRAnchor> anchors;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SpatialLayoutNode>()
        .def_ro("name", &SpatialLayoutNode::name)
        .def_ro("kind", &SpatialLayoutNode::kind)
        .def_ro("target_name", &SpatialLayoutNode::target_name)
        .def_ro("domain_index", &SpatialLayoutNode::domain_index)
        .def_ro("domain_transform_kind", &SpatialLayoutNode::domain_transform_kind)
        .def_ro("axes", &SpatialLayoutNode::axes)
        .def_ro("traits", &SpatialLayoutNode::traits)
        .def_ro("payload", &SpatialLayoutNode::payload)
        .def_ro("anchors", &SpatialLayoutNode::anchors);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.SpatialLayout", SpatialLayoutNode, Object);
};

class SpatialLayout : public ObjectRef {
 public:
  TVM_DLL SpatialLayout(ffi::String name, ffi::String kind, ffi::String target_name,
                        ffi::Array<ffi::String> axes, ffi::Array<ffi::String> traits,
                        ffi::Map<ffi::String, ffi::Any> payload,
                        ffi::Array<TIRAnchor> anchors);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(SpatialLayout, ObjectRef, SpatialLayoutNode);
};

class WorkPartitionNode : public Object {
 public:
  ffi::String name;
  ffi::String kind;
  ffi::String target_name;
  int64_t domain_index = -1;
  ffi::String partition_family;
  ffi::Array<ffi::String> axes;
  ffi::Array<ffi::String> traits;
  ffi::Map<ffi::String, ffi::Any> payload;
  ffi::Array<TIRAnchor> anchors;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<WorkPartitionNode>()
        .def_ro("name", &WorkPartitionNode::name)
        .def_ro("kind", &WorkPartitionNode::kind)
        .def_ro("target_name", &WorkPartitionNode::target_name)
        .def_ro("domain_index", &WorkPartitionNode::domain_index)
        .def_ro("partition_family", &WorkPartitionNode::partition_family)
        .def_ro("axes", &WorkPartitionNode::axes)
        .def_ro("traits", &WorkPartitionNode::traits)
        .def_ro("payload", &WorkPartitionNode::payload)
        .def_ro("anchors", &WorkPartitionNode::anchors);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.WorkPartition", WorkPartitionNode, Object);
};

class WorkPartition : public ObjectRef {
 public:
  TVM_DLL WorkPartition(ffi::String name, ffi::String kind, ffi::String target_name,
                        ffi::Array<ffi::String> axes, ffi::Array<ffi::String> traits,
                        ffi::Map<ffi::String, ffi::Any> payload,
                        ffi::Array<TIRAnchor> anchors);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(WorkPartition, ObjectRef, WorkPartitionNode);
};

class PlacementNode : public Object {
 public:
  ffi::String name;
  ffi::String kind;
  ffi::String task_name;
  int64_t task_index = -1;
  ffi::String member_func;
  ffi::String affinity_kind;
  ffi::String obligation_kind;
  ffi::String placement_domain;
  ffi::Array<ffi::String> traits;
  ffi::Map<ffi::String, ffi::Any> payload;
  ffi::Array<TIRAnchor> anchors;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PlacementNode>()
        .def_ro("name", &PlacementNode::name)
        .def_ro("kind", &PlacementNode::kind)
        .def_ro("task_name", &PlacementNode::task_name)
        .def_ro("task_index", &PlacementNode::task_index)
        .def_ro("member_func", &PlacementNode::member_func)
        .def_ro("affinity_kind", &PlacementNode::affinity_kind)
        .def_ro("obligation_kind", &PlacementNode::obligation_kind)
        .def_ro("placement_domain", &PlacementNode::placement_domain)
        .def_ro("traits", &PlacementNode::traits)
        .def_ro("payload", &PlacementNode::payload)
        .def_ro("anchors", &PlacementNode::anchors);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.Placement", PlacementNode, Object);
};

class Placement : public ObjectRef {
 public:
  TVM_DLL Placement(ffi::String name, ffi::String kind, ffi::String task_name,
                    ffi::String member_func, ffi::Array<ffi::String> traits,
                    ffi::Map<ffi::String, ffi::Any> payload,
                    ffi::Array<TIRAnchor> anchors);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Placement, ObjectRef, PlacementNode);
};

class SyncEdgeNode : public Object {
 public:
  ffi::String name;
  ffi::String kind;
  ffi::String source;
  ffi::String target;
  int64_t source_task_index = -1;
  int64_t target_task_index = -1;
  ffi::String ordering_kind;
  ffi::String materialization_kind;
  ffi::Array<ffi::String> traits;
  ffi::Map<ffi::String, ffi::Any> payload;
  ffi::Array<TIRAnchor> anchors;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SyncEdgeNode>()
        .def_ro("name", &SyncEdgeNode::name)
        .def_ro("kind", &SyncEdgeNode::kind)
        .def_ro("source", &SyncEdgeNode::source)
        .def_ro("target", &SyncEdgeNode::target)
        .def_ro("source_task_index", &SyncEdgeNode::source_task_index)
        .def_ro("target_task_index", &SyncEdgeNode::target_task_index)
        .def_ro("ordering_kind", &SyncEdgeNode::ordering_kind)
        .def_ro("materialization_kind", &SyncEdgeNode::materialization_kind)
        .def_ro("traits", &SyncEdgeNode::traits)
        .def_ro("payload", &SyncEdgeNode::payload)
        .def_ro("anchors", &SyncEdgeNode::anchors);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.SyncEdge", SyncEdgeNode, Object);
};

class SyncEdge : public ObjectRef {
 public:
  TVM_DLL SyncEdge(ffi::String name, ffi::String kind, ffi::String source,
                   ffi::String target, ffi::Array<ffi::String> traits,
                   ffi::Map<ffi::String, ffi::Any> payload,
                   ffi::Array<TIRAnchor> anchors);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(SyncEdge, ObjectRef, SyncEdgeNode);
};

class ResourceIntentNode : public Object {
 public:
  ffi::String name;
  ffi::String kind;
  ffi::String target_name;
  ffi::String target_kind;
  int64_t target_index = -1;
  ffi::Array<ffi::String> traits;
  ffi::Map<ffi::String, ffi::Any> payload;
  ffi::Array<TIRAnchor> anchors;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ResourceIntentNode>()
        .def_ro("name", &ResourceIntentNode::name)
        .def_ro("kind", &ResourceIntentNode::kind)
        .def_ro("target_name", &ResourceIntentNode::target_name)
        .def_ro("target_kind", &ResourceIntentNode::target_kind)
        .def_ro("target_index", &ResourceIntentNode::target_index)
        .def_ro("traits", &ResourceIntentNode::traits)
        .def_ro("payload", &ResourceIntentNode::payload)
        .def_ro("anchors", &ResourceIntentNode::anchors);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.ResourceIntent", ResourceIntentNode, Object);
};

class ResourceIntent : public ObjectRef {
 public:
  TVM_DLL ResourceIntent(ffi::String name, ffi::String kind, ffi::String target_name,
                         ffi::Array<ffi::String> traits, ffi::Map<ffi::String, ffi::Any> payload,
                         ffi::Array<TIRAnchor> anchors);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ResourceIntent, ObjectRef, ResourceIntentNode);
};

class ProgramPhaseNode : public Object {
 public:
  ffi::String name;
  int64_t phase_index = -1;
  ffi::Array<Integer> task_indices;
  ffi::Array<Integer> channel_indices;
  ffi::String closure_basis;
  ffi::Array<ffi::String> task_names;
  ffi::Array<ffi::String> channel_names;
  ffi::Array<ffi::String> traits;
  ffi::Map<ffi::String, ffi::Any> payload;
  ffi::Array<TIRAnchor> anchors;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ProgramPhaseNode>()
        .def_ro("name", &ProgramPhaseNode::name)
        .def_ro("phase_index", &ProgramPhaseNode::phase_index)
        .def_ro("task_indices", &ProgramPhaseNode::task_indices)
        .def_ro("channel_indices", &ProgramPhaseNode::channel_indices)
        .def_ro("closure_basis", &ProgramPhaseNode::closure_basis)
        .def_ro("task_names", &ProgramPhaseNode::task_names)
        .def_ro("channel_names", &ProgramPhaseNode::channel_names)
        .def_ro("traits", &ProgramPhaseNode::traits)
        .def_ro("payload", &ProgramPhaseNode::payload)
        .def_ro("anchors", &ProgramPhaseNode::anchors);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.ProgramPhase", ProgramPhaseNode, Object);
};

class ProgramPhase : public ObjectRef {
 public:
  TVM_DLL ProgramPhase(ffi::String name, ffi::Array<ffi::String> task_names,
                       ffi::Array<ffi::String> channel_names, ffi::Array<ffi::String> traits,
                       ffi::Map<ffi::String, ffi::Any> payload,
                       ffi::Array<TIRAnchor> anchors);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ProgramPhase, ObjectRef, ProgramPhaseNode);
};

class SpatialDomainPlanNode : public Object {
 public:
  ffi::String member_func;
  ffi::Array<SpatialLayout> layouts;
  ffi::Array<WorkPartition> work_partitions;
  ffi::Array<TIRAnchor> anchors;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SpatialDomainPlanNode>()
        .def_ro("member_func", &SpatialDomainPlanNode::member_func)
        .def_ro("layouts", &SpatialDomainPlanNode::layouts)
        .def_ro("work_partitions", &SpatialDomainPlanNode::work_partitions)
        .def_ro("anchors", &SpatialDomainPlanNode::anchors);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.SpatialDomainPlan", SpatialDomainPlanNode, Object);
};

class SpatialDomainPlan : public ObjectRef {
 public:
  TVM_DLL SpatialDomainPlan(ffi::String member_func, ffi::Array<SpatialLayout> layouts,
                            ffi::Array<WorkPartition> work_partitions,
                            ffi::Array<TIRAnchor> anchors);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(SpatialDomainPlan, ObjectRef,
                                             SpatialDomainPlanNode);
};

class SpatialExecutionPlanNode : public Object {
 public:
  ffi::String member_func;
  ffi::Array<ProgramPhase> phases;
  ffi::Array<Task> tasks;
  ffi::Array<Channel> channels;
  ffi::Array<Placement> placements;
  ffi::Array<SyncEdge> sync_edges;
  ffi::Array<ResourceIntent> resource_intents;
  ffi::Array<TIRAnchor> anchors;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SpatialExecutionPlanNode>()
        .def_ro("member_func", &SpatialExecutionPlanNode::member_func)
        .def_ro("phases", &SpatialExecutionPlanNode::phases)
        .def_ro("tasks", &SpatialExecutionPlanNode::tasks)
        .def_ro("channels", &SpatialExecutionPlanNode::channels)
        .def_ro("placements", &SpatialExecutionPlanNode::placements)
        .def_ro("sync_edges", &SpatialExecutionPlanNode::sync_edges)
        .def_ro("resource_intents", &SpatialExecutionPlanNode::resource_intents)
        .def_ro("anchors", &SpatialExecutionPlanNode::anchors);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.SpatialExecutionPlan", SpatialExecutionPlanNode, Object);
};

class SpatialExecutionPlan : public ObjectRef {
 public:
  TVM_DLL SpatialExecutionPlan(ffi::String member_func, ffi::Array<ProgramPhase> phases,
                               ffi::Array<Task> tasks, ffi::Array<Channel> channels,
                               ffi::Array<Placement> placements,
                               ffi::Array<SyncEdge> sync_edges,
                               ffi::Array<ResourceIntent> resource_intents,
                               ffi::Array<TIRAnchor> anchors);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(SpatialExecutionPlan, ObjectRef,
                                             SpatialExecutionPlanNode);
};

class SpatialProgramNode : public Object {
 public:
  ffi::String member_func;
  ffi::Array<ProgramPhase> phases;
  ffi::Array<Task> tasks;
  ffi::Array<Channel> channels;
  ffi::Array<SpatialLayout> layouts;
  ffi::Array<WorkPartition> work_partitions;
  ffi::Array<Placement> placements;
  ffi::Array<SyncEdge> sync_edges;
  ffi::Array<ResourceIntent> resource_intents;
  ffi::Array<TIRAnchor> anchors;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SpatialProgramNode>()
        .def_ro("member_func", &SpatialProgramNode::member_func)
        .def_ro("phases", &SpatialProgramNode::phases)
        .def_ro("tasks", &SpatialProgramNode::tasks)
        .def_ro("channels", &SpatialProgramNode::channels)
        .def_ro("layouts", &SpatialProgramNode::layouts)
        .def_ro("work_partitions", &SpatialProgramNode::work_partitions)
        .def_ro("placements", &SpatialProgramNode::placements)
        .def_ro("sync_edges", &SpatialProgramNode::sync_edges)
        .def_ro("resource_intents", &SpatialProgramNode::resource_intents)
        .def_ro("anchors", &SpatialProgramNode::anchors);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.SpatialProgram", SpatialProgramNode, Object);
};

class SpatialProgram : public ObjectRef {
 public:
  TVM_DLL SpatialProgram(ffi::String member_func, ffi::Array<ProgramPhase> phases,
                         ffi::Array<Task> tasks, ffi::Array<Channel> channels,
                         ffi::Array<SpatialLayout> layouts,
                         ffi::Array<WorkPartition> work_partitions,
                         ffi::Array<Placement> placements,
                         ffi::Array<SyncEdge> sync_edges,
                         ffi::Array<ResourceIntent> resource_intents,
                         ffi::Array<TIRAnchor> anchors);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(SpatialProgram, ObjectRef, SpatialProgramNode);
};

class TLDeviceProgramInfoNode : public GlobalInfoNode {
 public:
  ffi::String root_symbol;
  ffi::Array<ffi::String> member_funcs;
  ffi::Array<ProgramPhase> phases;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TLDeviceProgramInfoNode>()
        .def_ro("root_symbol", &TLDeviceProgramInfoNode::root_symbol)
        .def_ro("member_funcs", &TLDeviceProgramInfoNode::member_funcs)
        .def_ro("phases", &TLDeviceProgramInfoNode::phases);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TLDeviceProgramInfo", TLDeviceProgramInfoNode,
                                    GlobalInfoNode);
};

class TLDeviceProgramInfo : public GlobalInfo {
 public:
  TVM_DLL TLDeviceProgramInfo(ffi::String root_symbol, ffi::Array<ffi::String> member_funcs,
                              ffi::Array<ProgramPhase> phases = ffi::Array<ProgramPhase>());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TLDeviceProgramInfo, GlobalInfo,
                                             TLDeviceProgramInfoNode);
};

class SpatialCapabilityModelNode : public GlobalInfoNode {
 public:
  ffi::String arch_name;
  ffi::String topology_class;
  ffi::String placement_domain;
  int64_t logical_worker_grid_x = 0;
  int64_t logical_worker_grid_y = 0;
  int64_t functional_worker_count = 0;
  int64_t router_only_count = 0;
  int64_t dram_view_count = 0;
  int64_t worker_l1_size = 0;
  int64_t dram_view_size = 0;
  ffi::Array<ffi::String> supported_flow_kinds;
  ffi::Array<ffi::String> supported_payload_kinds;
  ffi::Array<ffi::String> supported_delivery_kinds;
  ffi::Array<ffi::String> supported_sync_kinds;
  ffi::Array<ffi::String> supported_ordering_kinds;
  ffi::Array<ffi::String> supported_materialization_kinds;
  ffi::Array<ffi::String> supported_layout_kinds;
  ffi::Array<ffi::String> supported_partition_kinds;
  ffi::Array<ffi::String> supported_resource_intent_kinds;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SpatialCapabilityModelNode>()
        .def_ro("arch_name", &SpatialCapabilityModelNode::arch_name)
        .def_ro("topology_class", &SpatialCapabilityModelNode::topology_class)
        .def_ro("placement_domain", &SpatialCapabilityModelNode::placement_domain)
        .def_ro("logical_worker_grid_x", &SpatialCapabilityModelNode::logical_worker_grid_x)
        .def_ro("logical_worker_grid_y", &SpatialCapabilityModelNode::logical_worker_grid_y)
        .def_ro("functional_worker_count", &SpatialCapabilityModelNode::functional_worker_count)
        .def_ro("router_only_count", &SpatialCapabilityModelNode::router_only_count)
        .def_ro("dram_view_count", &SpatialCapabilityModelNode::dram_view_count)
        .def_ro("worker_l1_size", &SpatialCapabilityModelNode::worker_l1_size)
        .def_ro("dram_view_size", &SpatialCapabilityModelNode::dram_view_size)
        .def_ro("supported_flow_kinds", &SpatialCapabilityModelNode::supported_flow_kinds)
        .def_ro("supported_payload_kinds", &SpatialCapabilityModelNode::supported_payload_kinds)
        .def_ro("supported_delivery_kinds", &SpatialCapabilityModelNode::supported_delivery_kinds)
        .def_ro("supported_sync_kinds", &SpatialCapabilityModelNode::supported_sync_kinds)
        .def_ro("supported_ordering_kinds", &SpatialCapabilityModelNode::supported_ordering_kinds)
        .def_ro("supported_materialization_kinds",
                &SpatialCapabilityModelNode::supported_materialization_kinds)
        .def_ro("supported_layout_kinds", &SpatialCapabilityModelNode::supported_layout_kinds)
        .def_ro("supported_partition_kinds",
                &SpatialCapabilityModelNode::supported_partition_kinds)
        .def_ro("supported_resource_intent_kinds",
                &SpatialCapabilityModelNode::supported_resource_intent_kinds);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.SpatialCapabilityModel", SpatialCapabilityModelNode,
                                    GlobalInfoNode);
};

class SpatialCapabilityModel : public GlobalInfo {
 public:
  TVM_DLL SpatialCapabilityModel(ffi::String arch_name, ffi::String topology_class,
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
                                 ffi::Array<ffi::String> supported_resource_intent_kinds);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(SpatialCapabilityModel, GlobalInfo,
                                             SpatialCapabilityModelNode);
};

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_SPATIAL_PROGRAM_H_
