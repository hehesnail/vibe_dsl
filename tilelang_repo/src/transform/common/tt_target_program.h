/*!
 * \file tt_target_program.h
 * \brief Stage 4 Phase C TT target companion objects.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_TT_TARGET_PROGRAM_H_
#define TVM_TL_TRANSFORM_COMMON_TT_TARGET_PROGRAM_H_

#include <tvm/ffi/reflection/registry.h>

#include "companion_base.h"

namespace tvm {
namespace tl {

class TTMeshPlanNode : public Object {
 public:
  ffi::String name;
  ffi::String mesh_kind;
  ffi::Array<Integer> mesh_shape;
  ffi::Array<Integer> device_range_start;
  ffi::Array<Integer> device_range_shape;
  ffi::String system_mesh_ref;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTMeshPlan", TTMeshPlanNode, Object);
};

class TTMeshPlan : public ObjectRef {
 public:
  TVM_DLL TTMeshPlan(ffi::String name, ffi::String mesh_kind,
                     ffi::Array<Integer> mesh_shape,
                     ffi::Array<Integer> device_range_start,
                     ffi::Array<Integer> device_range_shape,
                     ffi::String system_mesh_ref);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TTMeshPlan, ObjectRef, TTMeshPlanNode);
};

class TTBufferDistributionPlanNode : public Object {
 public:
  ffi::String name;
  ffi::String buffer;
  ffi::String mesh_plan;
  int64_t mesh_plan_index = -1;
  ffi::String distribution_kind;
  ffi::String layout;
  ffi::String memory_space;
  int64_t page_size_bytes = 0;
  ffi::Array<Integer> shard_shape;
  ffi::String shard_orientation;
  ffi::String host_visibility;
  ffi::Array<PrimExpr> logical_shape;
  ffi::Array<PrimExpr> local_shape;
  PrimExpr thread_extent;
  PrimExpr replicate_extent;
  ffi::Array<PrimExpr> inverse_logical_index_vars;
  ffi::Array<PrimExpr> inverse_logical_index_exprs;
  ffi::String spatial_layout;
  ffi::String spatial_distribution_kind;
  ffi::String abi_layout;
  ffi::String abi_memory_space;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTBufferDistributionPlan",
                                    TTBufferDistributionPlanNode, Object);
};

class TTBufferDistributionPlan : public ObjectRef {
 public:
  TVM_DLL TTBufferDistributionPlan(ffi::String name, ffi::String buffer,
                                   ffi::String mesh_plan, int64_t mesh_plan_index,
                                   ffi::String distribution_kind, ffi::String layout,
                                   ffi::String memory_space, int64_t page_size_bytes,
                                   ffi::Array<Integer> shard_shape,
                                   ffi::String shard_orientation,
                                   ffi::String host_visibility,
                                   ffi::Array<PrimExpr> logical_shape,
                                   ffi::Array<PrimExpr> local_shape,
                                   PrimExpr thread_extent, PrimExpr replicate_extent,
                                   ffi::Array<PrimExpr> inverse_logical_index_vars,
                                   ffi::Array<PrimExpr> inverse_logical_index_exprs,
                                   ffi::String spatial_layout,
                                   ffi::String spatial_distribution_kind,
                                   ffi::String abi_layout,
                                   ffi::String abi_memory_space);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TTBufferDistributionPlan, ObjectRef,
                                             TTBufferDistributionPlanNode);
};

class TTComputeOperandBindingPlanNode : public Object {
 public:
  ffi::String role;
  ffi::String buffer;
  ffi::String host_buffer;
  ffi::String tensor_dtype;
  ffi::String cb_dtype;
  ffi::String transform_kind;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTComputeOperandBindingPlan",
                                    TTComputeOperandBindingPlanNode, Object);
};

class TTComputeOperandBindingPlan : public ObjectRef {
 public:
  TVM_DLL TTComputeOperandBindingPlan(ffi::String role, ffi::String buffer,
                                      ffi::String host_buffer, ffi::String tensor_dtype,
                                      ffi::String cb_dtype, ffi::String transform_kind);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TTComputeOperandBindingPlan, ObjectRef,
                                             TTComputeOperandBindingPlanNode);
};

class TTComputeOpPlanNode : public Object {
 public:
  ffi::String name;
  ffi::String kernel_name;
  int64_t kernel_plan_index = -1;
  ffi::String kind;
  bool enabled = true;
  ffi::Array<TTComputeOperandBindingPlan> operand_bindings;
  ffi::Array<ffi::String> problem_shape_axes;
  ffi::Array<Integer> problem_shape;
  ffi::Array<Integer> tile_shape;
  ffi::Array<Integer> block_shape;
  ffi::Array<Integer> subblock_shape;
  ffi::String accumulator_dtype;
  ffi::String mbarrier_buffer;
  ffi::String mbarrier_scope;
  ffi::Array<ffi::String> mbarrier_index_exprs;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTComputeOpPlan", TTComputeOpPlanNode, Object);
};

class TTComputeOpPlan : public ObjectRef {
 public:
  TVM_DLL TTComputeOpPlan(ffi::String name, ffi::String kernel_name,
                          int64_t kernel_plan_index, ffi::String kind, bool enabled,
                          ffi::Array<TTComputeOperandBindingPlan> operand_bindings,
                          ffi::Array<ffi::String> problem_shape_axes,
                          ffi::Array<Integer> problem_shape,
                          ffi::Array<Integer> tile_shape,
                          ffi::Array<Integer> block_shape,
                          ffi::Array<Integer> subblock_shape,
                          ffi::String accumulator_dtype,
                          ffi::String mbarrier_buffer, ffi::String mbarrier_scope,
                          ffi::Array<ffi::String> mbarrier_index_exprs);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TTComputeOpPlan, ObjectRef,
                                             TTComputeOpPlanNode);
};

class TTBlockPlanNode : public Object {
 public:
  ffi::String name;
  ffi::String placement_kind;
  ffi::Array<Integer> task_indices;
  ffi::String core_group;
  int64_t core_group_index = -1;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTBlockPlan", TTBlockPlanNode, Object);
};

class TTBlockPlan : public ObjectRef {
 public:
  TVM_DLL TTBlockPlan(ffi::String name, ffi::String placement_kind,
                      ffi::Array<Integer> task_indices,
                      ffi::String core_group, int64_t core_group_index);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TTBlockPlan, ObjectRef, TTBlockPlanNode);
};

class TTKernelPlanNode : public Object {
 public:
  ffi::String name;
  ffi::String kind;
  ffi::String core_type;
  int64_t block_plan_index = -1;
  int64_t abi_plan_index = -1;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTKernelPlan", TTKernelPlanNode, Object);
};

class TTKernelPlan : public ObjectRef {
 public:
  TVM_DLL TTKernelPlan(ffi::String name, ffi::String kind, ffi::String core_type,
                       int64_t block_plan_index, int64_t abi_plan_index);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TTKernelPlan, ObjectRef, TTKernelPlanNode);
};

class TTKernelLaunchSpecNode : public Object {
 public:
  ffi::String core_type;
  ffi::String processor;
  ffi::String noc;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTKernelLaunchSpec", TTKernelLaunchSpecNode, Object);
};

class TTKernelLaunchSpec : public ObjectRef {
 public:
  TVM_DLL TTKernelLaunchSpec(ffi::String core_type, ffi::String processor, ffi::String noc);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TTKernelLaunchSpec, ObjectRef,
                                             TTKernelLaunchSpecNode);
};

class TTKernelDefineNode : public Object {
 public:
  ffi::String name;
  ffi::String value;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTKernelDefine", TTKernelDefineNode, Object);
};

class TTKernelDefine : public ObjectRef {
 public:
  TVM_DLL TTKernelDefine(ffi::String name, ffi::String value);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TTKernelDefine, ObjectRef,
                                             TTKernelDefineNode);
};

class TTKernelNamedCompileArgNode : public Object {
 public:
  ffi::String name;
  int64_t value = 0;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTKernelNamedCompileArg",
                                    TTKernelNamedCompileArgNode, Object);
};

class TTKernelNamedCompileArg : public ObjectRef {
 public:
  TVM_DLL TTKernelNamedCompileArg(ffi::String name, int64_t value);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TTKernelNamedCompileArg, ObjectRef,
                                             TTKernelNamedCompileArgNode);
};

class TTKernelComputeConfigNode : public Object {
 public:
  ffi::String math_fidelity;
  bool fp32_dest_acc_en = false;
  bool dst_full_sync_en = false;
  bool math_approx_mode = false;
  ffi::Array<ffi::String> unpack_to_dest_mode;
  bool bfp8_pack_precise = false;
  ffi::Array<TTKernelDefine> defines;
  ffi::Array<TTKernelNamedCompileArg> named_compile_args;
  bool clear_accum = false;
  int64_t k_pack = 1;
  int64_t wg_wait = 0;
  int64_t policy_type = 0;
  ffi::String policy_name;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTKernelComputeConfig",
                                    TTKernelComputeConfigNode, Object);
};

class TTKernelComputeConfig : public ObjectRef {
 public:
  TVM_DLL TTKernelComputeConfig(ffi::String math_fidelity, bool fp32_dest_acc_en,
                                bool dst_full_sync_en, bool math_approx_mode,
                                ffi::Array<ffi::String> unpack_to_dest_mode,
                                bool bfp8_pack_precise,
                                ffi::Array<TTKernelDefine> defines,
                                ffi::Array<TTKernelNamedCompileArg> named_compile_args,
                                bool clear_accum, int64_t k_pack, int64_t wg_wait,
                                int64_t policy_type, ffi::String policy_name);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TTKernelComputeConfig, ObjectRef,
                                             TTKernelComputeConfigNode);
};

class TTPerWorkArgSpecNode : public Object {
 public:
  ffi::String arg_kind;
  ffi::String arg_identity;
  ffi::String buffer;
  ffi::String descriptor_kind;
  ffi::String value_kind;
  ffi::String value_source;
  int64_t constant_value = 0;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTPerWorkArgSpec", TTPerWorkArgSpecNode, Object);
};

class TTPerWorkArgSpec : public ObjectRef {
 public:
  TVM_DLL TTPerWorkArgSpec(ffi::String arg_kind, ffi::String arg_identity,
                           ffi::String buffer, ffi::String descriptor_kind,
                           ffi::String value_kind, ffi::String value_source,
                           int64_t constant_value);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TTPerWorkArgSpec, ObjectRef,
                                             TTPerWorkArgSpecNode);
};

class TTKernelNode : public Object {
 public:
  ffi::String name;
  ffi::String kind;
  ffi::String core_type;
  int64_t abi_plan_index = -1;
  TTKernelLaunchSpec launch_spec;
  TTKernelComputeConfig compute_config;
  ffi::Array<TTPerWorkArgSpec> per_work_arg_specs;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTKernel", TTKernelNode, Object);
};

class TTKernel : public ObjectRef {
 public:
  TVM_DLL TTKernel(ffi::String name, ffi::String kind, ffi::String core_type,
                   int64_t abi_plan_index,
                   TTKernelLaunchSpec launch_spec,
                   TTKernelComputeConfig compute_config,
                   ffi::Array<TTPerWorkArgSpec> per_work_arg_specs);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TTKernel, ObjectRef, TTKernelNode);
};

class TTCoreGroupNode : public Object {
 public:
  ffi::String name;
  int64_t logical_grid_x = 1;
  int64_t logical_grid_y = 1;
  ffi::String linearization;
  ffi::Array<ffi::Any> physical_cores;
  ffi::Array<ffi::Any> work_packets;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTCoreGroup", TTCoreGroupNode, Object);
};

class TTCoreGroup : public ObjectRef {
 public:
  TVM_DLL TTCoreGroup(ffi::String name, int64_t logical_grid_x, int64_t logical_grid_y,
                      ffi::String linearization, ffi::Array<ffi::Any> physical_cores,
                      ffi::Array<ffi::Any> work_packets);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TTCoreGroup, ObjectRef, TTCoreGroupNode);
};

class TTCBPlanNode : public Object {
 public:
  ffi::String name;
  int64_t cb_id = -1;
  ffi::String resource_class;
  int64_t num_pages = 0;
  int64_t page_size_bytes = 0;
  ffi::String data_format;
  int64_t initial_reserve_pages = 0;
  ffi::String flow_class;
  int64_t publish_pages_per_event = 0;
  int64_t consume_pages_per_event = 0;
  int64_t lifetime_begin = 0;
  int64_t lifetime_end = 0;
  ffi::Array<ffi::String> requirement_names;
  ffi::Array<Integer> requirement_indices;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTCBPlan", TTCBPlanNode, Object);
};

class TTCBPlan : public ObjectRef {
 public:
  TVM_DLL TTCBPlan(ffi::String name, int64_t cb_id, ffi::String resource_class,
                   int64_t num_pages, int64_t page_size_bytes, ffi::String data_format,
                   int64_t initial_reserve_pages, ffi::String flow_class,
                   int64_t publish_pages_per_event, int64_t consume_pages_per_event,
                   int64_t lifetime_begin, int64_t lifetime_end,
                   ffi::Array<ffi::String> requirement_names,
                   ffi::Array<Integer> requirement_indices);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TTCBPlan, ObjectRef, TTCBPlanNode);
};

class TTTransportPlanNode : public Object {
 public:
  ffi::String name;
  ffi::String kind;
  int64_t source_task_index = -1;
  int64_t target_task_index = -1;
  ffi::String value_kind;
  ffi::String delivery_kind;
  ffi::String subject;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTTransportPlan", TTTransportPlanNode, Object);
};

class TTTransportPlan : public ObjectRef {
 public:
  TVM_DLL TTTransportPlan(ffi::String name, ffi::String kind, int64_t source_task_index,
                          int64_t target_task_index, ffi::String value_kind,
                          ffi::String delivery_kind, ffi::String subject);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TTTransportPlan, ObjectRef, TTTransportPlanNode);
};

class TTSyncPlanNode : public Object {
 public:
  ffi::String name;
  ffi::String kind;
  int64_t source_task_index = -1;
  int64_t target_task_index = -1;
  ffi::String ordering_kind;
  ffi::String completion_kind;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTSyncPlan", TTSyncPlanNode, Object);
};

class TTSyncPlan : public ObjectRef {
 public:
  TVM_DLL TTSyncPlan(ffi::String name, ffi::String kind, int64_t source_task_index,
                     int64_t target_task_index, ffi::String ordering_kind,
                     ffi::String completion_kind);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TTSyncPlan, ObjectRef, TTSyncPlanNode);
};

class TTSemaphorePlanNode : public Object {
 public:
  ffi::String name;
  ffi::String kind;
  int64_t semaphore_id = -1;
  int64_t initial_value = 0;
  ffi::String core_type;
  int64_t source_task_index = -1;
  int64_t target_task_index = -1;
  ffi::Array<ffi::Any> core_ranges;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTSemaphorePlan", TTSemaphorePlanNode, Object);
};

class TTSemaphorePlan : public ObjectRef {
 public:
  TVM_DLL TTSemaphorePlan(ffi::String name, ffi::String kind, int64_t semaphore_id,
                          int64_t initial_value, ffi::String core_type,
                          int64_t source_task_index, int64_t target_task_index,
                          ffi::Array<ffi::Any> core_ranges);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TTSemaphorePlan, ObjectRef,
                                             TTSemaphorePlanNode);
};

class TTComputeSyncPlanNode : public Object {
 public:
  ffi::String name;
  ffi::String kind;
  int64_t source_task_index = -1;
  int64_t target_task_index = -1;
  ffi::String ordering_kind;
  ffi::String materialization_kind;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTComputeSyncPlan", TTComputeSyncPlanNode, Object);
};

class TTComputeSyncPlan : public ObjectRef {
 public:
  TVM_DLL TTComputeSyncPlan(ffi::String name, ffi::String kind, int64_t source_task_index,
                            int64_t target_task_index, ffi::String ordering_kind,
                            ffi::String materialization_kind);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TTComputeSyncPlan, ObjectRef,
                                             TTComputeSyncPlanNode);
};

class TTDstLayoutPlanNode : public Object {
 public:
  ffi::String name;
  ffi::String buffer;
  ffi::String layout;
  ffi::String memory_space;
  int64_t page_size_bytes = 0;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTDstLayoutPlan", TTDstLayoutPlanNode, Object);
};

class TTDstLayoutPlan : public ObjectRef {
 public:
  TVM_DLL TTDstLayoutPlan(ffi::String name, ffi::String buffer, ffi::String layout,
                          ffi::String memory_space, int64_t page_size_bytes);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TTDstLayoutPlan, ObjectRef,
                                             TTDstLayoutPlanNode);
};

class TTLiveFormPlanNode : public Object {
 public:
  ffi::String name;
  ffi::String logical_value;
  ffi::String spatial_live_value;
  int64_t spatial_live_value_index = -1;
  ffi::String producer_kernel;
  ffi::String physical_form;
  ffi::String execution_topology;
  int64_t physical_local_extent = 0;
  int64_t logical_element_count = 0;
  ffi::String ownership_kind;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTLiveFormPlan", TTLiveFormPlanNode, Object);
};

class TTLiveFormPlan : public ObjectRef {
 public:
  TVM_DLL TTLiveFormPlan(ffi::String name, ffi::String logical_value,
                         ffi::String spatial_live_value, int64_t spatial_live_value_index,
                         ffi::String producer_kernel, ffi::String physical_form,
                         ffi::String execution_topology, int64_t physical_local_extent,
                         int64_t logical_element_count, ffi::String ownership_kind);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TTLiveFormPlan, ObjectRef, TTLiveFormPlanNode);
};

class TTMaterializationPlanNode : public Object {
 public:
  ffi::String name;
  ffi::String source_live_form;
  ffi::String materialization_boundary;
  int64_t materialization_boundary_index = -1;
  ffi::String target_buffer;
  ffi::String host_buffer;
  ffi::String target_kernel;
  ffi::String bridge_kind;
  ffi::String materialization_kind;
  ffi::String materialization_protocol;
  ffi::String publication_protocol;
  ffi::Array<Integer> required_cb_plan_indices;
  ffi::Array<Integer> required_sync_plan_indices;
  ffi::String produced_live_form;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTMaterializationPlan", TTMaterializationPlanNode, Object);
};

class TTMaterializationPlan : public ObjectRef {
 public:
  TVM_DLL TTMaterializationPlan(ffi::String name, ffi::String source_live_form,
                                ffi::String materialization_boundary,
                                int64_t materialization_boundary_index,
                                ffi::String target_buffer, ffi::String host_buffer,
                                ffi::String target_kernel, ffi::String bridge_kind,
                                ffi::String materialization_kind,
                                ffi::String materialization_protocol,
                                ffi::String publication_protocol,
                                ffi::Array<Integer> required_cb_plan_indices,
                                ffi::Array<Integer> required_sync_plan_indices,
                                ffi::String produced_live_form);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TTMaterializationPlan, ObjectRef,
                                             TTMaterializationPlanNode);
};

class TTConsumerBindingPlanNode : public Object {
 public:
  ffi::String name;
  ffi::String consumer_kernel;
  ffi::String consumer_op_kind;
  ffi::String source_live_form;
  ffi::String live_value_edge;
  int64_t live_value_edge_index = -1;
  bool accepts_distributed_slice = false;
  bool requires_full_logical_tile = false;
  int64_t abi_plan_index = -1;
  ffi::String target_buffer;
  ffi::String materialization_plan;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTConsumerBindingPlan", TTConsumerBindingPlanNode,
                                    Object);
};

class TTConsumerBindingPlan : public ObjectRef {
 public:
  TVM_DLL TTConsumerBindingPlan(ffi::String name, ffi::String consumer_kernel,
                                ffi::String consumer_op_kind, ffi::String source_live_form,
                                ffi::String live_value_edge, int64_t live_value_edge_index,
                                bool accepts_distributed_slice,
                                bool requires_full_logical_tile, int64_t abi_plan_index,
                                ffi::String target_buffer, ffi::String materialization_plan);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TTConsumerBindingPlan, ObjectRef,
                                             TTConsumerBindingPlanNode);
};

class TTRuntimeArgSpecNode : public Object {
 public:
  ffi::String name;
  ffi::String kind;
  ffi::String dtype;
  ffi::String buffer;
  ffi::String identity;
  int64_t core_x = -1;
  int64_t core_y = -1;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTRuntimeArgSpec", TTRuntimeArgSpecNode, Object);
};

class TTRuntimeArgSpec : public ObjectRef {
 public:
  TVM_DLL TTRuntimeArgSpec(ffi::String name, ffi::String kind, ffi::String dtype,
                           ffi::String buffer, ffi::String identity,
                           int64_t core_x, int64_t core_y);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TTRuntimeArgSpec, ObjectRef,
                                             TTRuntimeArgSpecNode);
};

class TTCompileTimeArgSpecNode : public Object {
 public:
  ffi::String name;
  ffi::String kind;
  ffi::String dtype;
  int64_t offset = 0;
  int64_t count = 0;
  ffi::String buffer;
  ffi::String segment_role;
  ffi::Array<Integer> values;
  int64_t args_config_bits = 0;
  int64_t transport_page_size = 0;
  ffi::String layout;
  ffi::String memory_space;
  ffi::Array<Integer> host_axis_order;
  bool transpose_2d = false;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTCompileTimeArgSpec",
                                    TTCompileTimeArgSpecNode, Object);
};

class TTCompileTimeArgSpec : public ObjectRef {
 public:
  TVM_DLL TTCompileTimeArgSpec(ffi::String name, ffi::String kind, ffi::String dtype,
                               int64_t offset, int64_t count, ffi::String buffer,
                               ffi::String segment_role, ffi::Array<Integer> values,
                               int64_t args_config_bits, int64_t transport_page_size,
                               ffi::String layout, ffi::String memory_space,
                               ffi::Array<Integer> host_axis_order, bool transpose_2d);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TTCompileTimeArgSpec, ObjectRef,
                                             TTCompileTimeArgSpecNode);
};

class TTAccessorSpecNode : public Object {
 public:
  ffi::String buffer;
  int64_t compile_time_arg_offset = 0;
  int64_t compile_time_arg_count = 0;
  int64_t common_runtime_arg_offset = 0;
  int64_t common_runtime_arg_count = 0;
  int64_t args_config_bits = 0;
  int64_t transport_page_size = 0;
  ffi::String layout;
  ffi::String memory_space;
  ffi::Array<Integer> host_axis_order;
  bool transpose_2d = false;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTAccessorSpec", TTAccessorSpecNode, Object);
};

class TTAccessorSpec : public ObjectRef {
 public:
  TVM_DLL TTAccessorSpec(ffi::String buffer, int64_t compile_time_arg_offset,
                         int64_t compile_time_arg_count, int64_t common_runtime_arg_offset,
                         int64_t common_runtime_arg_count, int64_t args_config_bits,
                         int64_t transport_page_size, ffi::String layout,
                         ffi::String memory_space, ffi::Array<Integer> host_axis_order,
                         bool transpose_2d);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TTAccessorSpec, ObjectRef,
                                             TTAccessorSpecNode);
};

class TTSemaphoreBindingSpecNode : public Object {
 public:
  ffi::String name;
  int64_t semaphore_id = 0;
  ffi::String arg_kind;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTSemaphoreBindingSpec",
                                    TTSemaphoreBindingSpecNode, Object);
};

class TTSemaphoreBindingSpec : public ObjectRef {
 public:
  TVM_DLL TTSemaphoreBindingSpec(ffi::String name, int64_t semaphore_id,
                                 ffi::String arg_kind);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TTSemaphoreBindingSpec, ObjectRef,
                                             TTSemaphoreBindingSpecNode);
};

class TTABIPlanNode : public Object {
 public:
  ffi::String name;
  ffi::String kernel_name;
  ffi::Array<TTRuntimeArgSpec> runtime_args;
  ffi::Array<TTRuntimeArgSpec> common_runtime_args;
  ffi::Array<TTCompileTimeArgSpec> compile_time_arg_specs;
  ffi::Array<TTAccessorSpec> accessors;
  ffi::Array<TTSemaphoreBindingSpec> semaphore_bindings;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTABIPlan", TTABIPlanNode, Object);
};

class TTABIPlan : public ObjectRef {
 public:
  TVM_DLL TTABIPlan(ffi::String name, ffi::String kernel_name,
                    ffi::Array<TTRuntimeArgSpec> runtime_args,
                    ffi::Array<TTRuntimeArgSpec> common_runtime_args,
                    ffi::Array<TTCompileTimeArgSpec> compile_time_arg_specs,
                    ffi::Array<TTAccessorSpec> accessors,
                    ffi::Array<TTSemaphoreBindingSpec> semaphore_bindings);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TTABIPlan, ObjectRef, TTABIPlanNode);
};

class TTExecutionPlanNode : public Object {
 public:
  ffi::String name;
  ffi::Array<ffi::String> kernel_names;
  ffi::Array<Integer> phase_indices;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTExecutionPlan", TTExecutionPlanNode, Object);
};

class TTExecutionPlan : public ObjectRef {
 public:
  TVM_DLL TTExecutionPlan(ffi::String name, ffi::Array<ffi::String> kernel_names,
                          ffi::Array<Integer> phase_indices);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TTExecutionPlan, ObjectRef,
                                             TTExecutionPlanNode);
};

class TTProgramNode : public Object {
 public:
  ffi::String entry_name;
  ffi::String member_func;
  ffi::Array<TTMeshPlan> mesh_plans;
  ffi::Array<TTBufferDistributionPlan> buffer_distribution_plans;
  ffi::Array<TTBlockPlan> block_plans;
  ffi::Array<TTKernelPlan> kernel_plans;
  ffi::Array<TTComputeOpPlan> compute_op_plans;
  ffi::Array<TTTransportPlan> transport_plans;
  ffi::Array<TTSyncPlan> sync_plans;
  ffi::Array<TTABIPlan> abi_plans;
  ffi::Array<TTExecutionPlan> execution_plans;
  ffi::Array<TTKernel> kernels;
  ffi::Array<TTCoreGroup> core_groups;
  ffi::Array<TTCBPlan> cb_plans;
  ffi::Array<TTSemaphorePlan> semaphore_plans;
  ffi::Array<TTComputeSyncPlan> compute_sync_plans;
  ffi::Array<TTDstLayoutPlan> dst_layout_plans;
  ffi::Array<TTLiveFormPlan> live_form_plans;
  ffi::Array<TTMaterializationPlan> materialization_plans;
  ffi::Array<TTConsumerBindingPlan> consumer_binding_plans;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTProgram", TTProgramNode, Object);
};

class TTProgram : public ObjectRef {
 public:
  TVM_DLL TTProgram(ffi::String entry_name, ffi::String member_func,
                    ffi::Array<TTMeshPlan> mesh_plans,
                    ffi::Array<TTBufferDistributionPlan> buffer_distribution_plans,
                    ffi::Array<TTBlockPlan> block_plans,
                    ffi::Array<TTKernelPlan> kernel_plans,
                    ffi::Array<TTComputeOpPlan> compute_op_plans,
                    ffi::Array<TTTransportPlan> transport_plans,
                    ffi::Array<TTSyncPlan> sync_plans,
                    ffi::Array<TTABIPlan> abi_plans,
                    ffi::Array<TTExecutionPlan> execution_plans,
                    ffi::Array<TTKernel> kernels,
                    ffi::Array<TTCoreGroup> core_groups, ffi::Array<TTCBPlan> cb_plans,
                    ffi::Array<TTSemaphorePlan> semaphore_plans,
                    ffi::Array<TTComputeSyncPlan> compute_sync_plans,
                    ffi::Array<TTDstLayoutPlan> dst_layout_plans,
                    ffi::Array<TTLiveFormPlan> live_form_plans,
                    ffi::Array<TTMaterializationPlan> materialization_plans,
                    ffi::Array<TTConsumerBindingPlan> consumer_binding_plans);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TTProgram, ObjectRef, TTProgramNode);
};

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_TT_TARGET_PROGRAM_H_
