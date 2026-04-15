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

class TTBlockPlanNode : public Object {
 public:
  ffi::String name;
  ffi::String placement_kind;
  ffi::Array<Integer> task_indices;
  ffi::Map<ffi::String, ffi::Any> payload;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTBlockPlan", TTBlockPlanNode, Object);
};

class TTBlockPlan : public ObjectRef {
 public:
  TVM_DLL TTBlockPlan(ffi::String name, ffi::String placement_kind,
                      ffi::Array<Integer> task_indices,
                      ffi::Map<ffi::String, ffi::Any> payload);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TTBlockPlan, ObjectRef, TTBlockPlanNode);
};

class TTKernelPlanNode : public Object {
 public:
  ffi::String name;
  ffi::String kind;
  ffi::String core_type;
  int64_t block_plan_index = -1;
  int64_t abi_plan_index = -1;
  ffi::Map<ffi::String, ffi::Any> payload;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTKernelPlan", TTKernelPlanNode, Object);
};

class TTKernelPlan : public ObjectRef {
 public:
  TVM_DLL TTKernelPlan(ffi::String name, ffi::String kind, ffi::String core_type,
                       int64_t block_plan_index, int64_t abi_plan_index,
                       ffi::Map<ffi::String, ffi::Any> payload);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TTKernelPlan, ObjectRef, TTKernelPlanNode);
};

class TTKernelNode : public Object {
 public:
  ffi::String name;
  ffi::String kind;
  ffi::String core_type;
  int64_t abi_plan_index = -1;
  ffi::Map<ffi::String, ffi::Any> payload;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTKernel", TTKernelNode, Object);
};

class TTKernel : public ObjectRef {
 public:
  TVM_DLL TTKernel(ffi::String name, ffi::String kind, ffi::String core_type,
                   int64_t abi_plan_index, ffi::Map<ffi::String, ffi::Any> payload);
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
  ffi::Map<ffi::String, ffi::Any> payload;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTCoreGroup", TTCoreGroupNode, Object);
};

class TTCoreGroup : public ObjectRef {
 public:
  TVM_DLL TTCoreGroup(ffi::String name, int64_t logical_grid_x, int64_t logical_grid_y,
                      ffi::String linearization, ffi::Array<ffi::Any> physical_cores,
                      ffi::Array<ffi::Any> work_packets, ffi::Map<ffi::String, ffi::Any> payload);
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
  ffi::Map<ffi::String, ffi::Any> payload;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTCBPlan", TTCBPlanNode, Object);
};

class TTCBPlan : public ObjectRef {
 public:
  TVM_DLL TTCBPlan(ffi::String name, int64_t cb_id, ffi::String resource_class,
                   int64_t num_pages, int64_t page_size_bytes, ffi::String data_format,
                   ffi::Map<ffi::String, ffi::Any> payload);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TTCBPlan, ObjectRef, TTCBPlanNode);
};

class TTTransportPlanNode : public Object {
 public:
  ffi::String name;
  ffi::String kind;
  int64_t source_task_index = -1;
  int64_t target_task_index = -1;
  ffi::String payload_kind;
  ffi::String delivery_kind;
  ffi::Map<ffi::String, ffi::Any> payload;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTTransportPlan", TTTransportPlanNode, Object);
};

class TTTransportPlan : public ObjectRef {
 public:
  TVM_DLL TTTransportPlan(ffi::String name, ffi::String kind, int64_t source_task_index,
                          int64_t target_task_index, ffi::String payload_kind,
                          ffi::String delivery_kind, ffi::Map<ffi::String, ffi::Any> payload);
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
  ffi::Map<ffi::String, ffi::Any> payload;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTSyncPlan", TTSyncPlanNode, Object);
};

class TTSyncPlan : public ObjectRef {
 public:
  TVM_DLL TTSyncPlan(ffi::String name, ffi::String kind, int64_t source_task_index,
                     int64_t target_task_index, ffi::String ordering_kind,
                     ffi::String completion_kind,
                     ffi::Map<ffi::String, ffi::Any> payload);
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
  ffi::Map<ffi::String, ffi::Any> payload;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTSemaphorePlan", TTSemaphorePlanNode, Object);
};

class TTSemaphorePlan : public ObjectRef {
 public:
  TVM_DLL TTSemaphorePlan(ffi::String name, ffi::String kind, int64_t semaphore_id,
                          int64_t initial_value, ffi::String core_type,
                          int64_t source_task_index, int64_t target_task_index,
                          ffi::Array<ffi::Any> core_ranges,
                          ffi::Map<ffi::String, ffi::Any> payload);
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
  ffi::Map<ffi::String, ffi::Any> payload;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTComputeSyncPlan", TTComputeSyncPlanNode, Object);
};

class TTComputeSyncPlan : public ObjectRef {
 public:
  TVM_DLL TTComputeSyncPlan(ffi::String name, ffi::String kind, int64_t source_task_index,
                            int64_t target_task_index, ffi::String ordering_kind,
                            ffi::String materialization_kind,
                            ffi::Map<ffi::String, ffi::Any> payload);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TTComputeSyncPlan, ObjectRef,
                                             TTComputeSyncPlanNode);
};

class TTDstLayoutPlanNode : public Object {
 public:
  ffi::String name;
  ffi::String buffer;
  ffi::String layout;
  ffi::String memory_space;
  ffi::Map<ffi::String, ffi::Any> payload;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTDstLayoutPlan", TTDstLayoutPlanNode, Object);
};

class TTDstLayoutPlan : public ObjectRef {
 public:
  TVM_DLL TTDstLayoutPlan(ffi::String name, ffi::String buffer, ffi::String layout,
                          ffi::String memory_space, ffi::Map<ffi::String, ffi::Any> payload);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TTDstLayoutPlan, ObjectRef,
                                             TTDstLayoutPlanNode);
};

class TTABIPlanNode : public Object {
 public:
  ffi::String name;
  ffi::String kernel_name;
  ffi::Array<ffi::Any> runtime_args;
  ffi::Array<ffi::Any> common_runtime_args;
  ffi::Array<ffi::Any> compile_time_arg_specs;
  ffi::Array<ffi::Any> accessors;
  ffi::Array<ffi::Any> semaphore_bindings;
  ffi::Map<ffi::String, ffi::Any> payload;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTABIPlan", TTABIPlanNode, Object);
};

class TTABIPlan : public ObjectRef {
 public:
  TVM_DLL TTABIPlan(ffi::String name, ffi::String kernel_name, ffi::Array<ffi::Any> runtime_args,
                    ffi::Array<ffi::Any> common_runtime_args,
                    ffi::Array<ffi::Any> compile_time_arg_specs,
                    ffi::Array<ffi::Any> accessors, ffi::Array<ffi::Any> semaphore_bindings,
                    ffi::Map<ffi::String, ffi::Any> payload);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TTABIPlan, ObjectRef, TTABIPlanNode);
};

class TTExecutionPlanNode : public Object {
 public:
  ffi::String name;
  ffi::Array<ffi::String> kernel_names;
  ffi::Array<Integer> phase_indices;
  ffi::Map<ffi::String, ffi::Any> payload;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTExecutionPlan", TTExecutionPlanNode, Object);
};

class TTExecutionPlan : public ObjectRef {
 public:
  TVM_DLL TTExecutionPlan(ffi::String name, ffi::Array<ffi::String> kernel_names,
                          ffi::Array<Integer> phase_indices,
                          ffi::Map<ffi::String, ffi::Any> payload);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TTExecutionPlan, ObjectRef,
                                             TTExecutionPlanNode);
};

class TTProgramNode : public Object {
 public:
  ffi::String entry_name;
  ffi::String member_func;
  ffi::Array<TTBlockPlan> block_plans;
  ffi::Array<TTKernelPlan> kernel_plans;
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
  ffi::Map<ffi::String, ffi::Any> payload;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTProgram", TTProgramNode, Object);
};

class TTProgram : public ObjectRef {
 public:
  TVM_DLL TTProgram(ffi::String entry_name, ffi::String member_func,
                    ffi::Array<TTBlockPlan> block_plans,
                    ffi::Array<TTKernelPlan> kernel_plans,
                    ffi::Array<TTTransportPlan> transport_plans,
                    ffi::Array<TTSyncPlan> sync_plans,
                    ffi::Array<TTABIPlan> abi_plans,
                    ffi::Array<TTExecutionPlan> execution_plans,
                    ffi::Array<TTKernel> kernels,
                    ffi::Array<TTCoreGroup> core_groups, ffi::Array<TTCBPlan> cb_plans,
                    ffi::Array<TTSemaphorePlan> semaphore_plans,
                    ffi::Array<TTComputeSyncPlan> compute_sync_plans,
                    ffi::Array<TTDstLayoutPlan> dst_layout_plans,
                    ffi::Map<ffi::String, ffi::Any> payload);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TTProgram, ObjectRef, TTProgramNode);
};

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_TT_TARGET_PROGRAM_H_
