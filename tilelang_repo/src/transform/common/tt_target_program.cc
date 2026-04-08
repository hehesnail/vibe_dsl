/*!
 * \file tt_target_program.cc
 * \brief Stage 4 Phase C TT target companion objects.
 */

#include "tt_target_program.h"

namespace tvm {
namespace tl {

namespace {

template <typename NodeT>
void RegisterNodeReflection() {
  NodeT::RegisterReflection();
}

}  // namespace

void TTKernelNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTKernelNode>()
      .def_ro("name", &TTKernelNode::name)
      .def_ro("kind", &TTKernelNode::kind)
      .def_ro("core_type", &TTKernelNode::core_type)
      .def_ro("abi_plan_index", &TTKernelNode::abi_plan_index)
      .def_ro("payload", &TTKernelNode::payload);
}

TTKernel::TTKernel(ffi::String name, ffi::String kind, ffi::String core_type,
                   int64_t abi_plan_index, ffi::Map<ffi::String, ffi::Any> payload) {
  auto n = ffi::make_object<TTKernelNode>();
  n->name = std::move(name);
  n->kind = std::move(kind);
  n->core_type = std::move(core_type);
  n->abi_plan_index = abi_plan_index;
  n->payload = std::move(payload);
  data_ = std::move(n);
}

void TTCoreGroupNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTCoreGroupNode>()
      .def_ro("name", &TTCoreGroupNode::name)
      .def_ro("logical_grid_x", &TTCoreGroupNode::logical_grid_x)
      .def_ro("logical_grid_y", &TTCoreGroupNode::logical_grid_y)
      .def_ro("linearization", &TTCoreGroupNode::linearization)
      .def_ro("physical_cores", &TTCoreGroupNode::physical_cores)
      .def_ro("work_packets", &TTCoreGroupNode::work_packets)
      .def_ro("payload", &TTCoreGroupNode::payload);
}

TTCoreGroup::TTCoreGroup(ffi::String name, int64_t logical_grid_x, int64_t logical_grid_y,
                         ffi::String linearization, ffi::Array<ffi::Any> physical_cores,
                         ffi::Array<ffi::Any> work_packets,
                         ffi::Map<ffi::String, ffi::Any> payload) {
  auto n = ffi::make_object<TTCoreGroupNode>();
  n->name = std::move(name);
  n->logical_grid_x = logical_grid_x;
  n->logical_grid_y = logical_grid_y;
  n->linearization = std::move(linearization);
  n->physical_cores = std::move(physical_cores);
  n->work_packets = std::move(work_packets);
  n->payload = std::move(payload);
  data_ = std::move(n);
}

void TTCBPlanNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTCBPlanNode>()
      .def_ro("name", &TTCBPlanNode::name)
      .def_ro("cb_id", &TTCBPlanNode::cb_id)
      .def_ro("resource_class", &TTCBPlanNode::resource_class)
      .def_ro("num_pages", &TTCBPlanNode::num_pages)
      .def_ro("page_size_bytes", &TTCBPlanNode::page_size_bytes)
      .def_ro("data_format", &TTCBPlanNode::data_format)
      .def_ro("payload", &TTCBPlanNode::payload);
}

TTCBPlan::TTCBPlan(ffi::String name, int64_t cb_id, ffi::String resource_class, int64_t num_pages,
                   int64_t page_size_bytes, ffi::String data_format,
                   ffi::Map<ffi::String, ffi::Any> payload) {
  auto n = ffi::make_object<TTCBPlanNode>();
  n->name = std::move(name);
  n->cb_id = cb_id;
  n->resource_class = std::move(resource_class);
  n->num_pages = num_pages;
  n->page_size_bytes = page_size_bytes;
  n->data_format = std::move(data_format);
  n->payload = std::move(payload);
  data_ = std::move(n);
}

void TTTransportPlanNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTTransportPlanNode>()
      .def_ro("name", &TTTransportPlanNode::name)
      .def_ro("kind", &TTTransportPlanNode::kind)
      .def_ro("source_task_index", &TTTransportPlanNode::source_task_index)
      .def_ro("target_task_index", &TTTransportPlanNode::target_task_index)
      .def_ro("payload_kind", &TTTransportPlanNode::payload_kind)
      .def_ro("delivery_kind", &TTTransportPlanNode::delivery_kind)
      .def_ro("payload", &TTTransportPlanNode::payload);
}

TTTransportPlan::TTTransportPlan(ffi::String name, ffi::String kind, int64_t source_task_index,
                                 int64_t target_task_index, ffi::String payload_kind,
                                 ffi::String delivery_kind,
                                 ffi::Map<ffi::String, ffi::Any> payload) {
  auto n = ffi::make_object<TTTransportPlanNode>();
  n->name = std::move(name);
  n->kind = std::move(kind);
  n->source_task_index = source_task_index;
  n->target_task_index = target_task_index;
  n->payload_kind = std::move(payload_kind);
  n->delivery_kind = std::move(delivery_kind);
  n->payload = std::move(payload);
  data_ = std::move(n);
}

void TTSemaphorePlanNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTSemaphorePlanNode>()
      .def_ro("name", &TTSemaphorePlanNode::name)
      .def_ro("kind", &TTSemaphorePlanNode::kind)
      .def_ro("semaphore_id", &TTSemaphorePlanNode::semaphore_id)
      .def_ro("initial_value", &TTSemaphorePlanNode::initial_value)
      .def_ro("core_type", &TTSemaphorePlanNode::core_type)
      .def_ro("source_task_index", &TTSemaphorePlanNode::source_task_index)
      .def_ro("target_task_index", &TTSemaphorePlanNode::target_task_index)
      .def_ro("core_ranges", &TTSemaphorePlanNode::core_ranges)
      .def_ro("payload", &TTSemaphorePlanNode::payload);
}

TTSemaphorePlan::TTSemaphorePlan(ffi::String name, ffi::String kind, int64_t semaphore_id,
                                 int64_t initial_value, ffi::String core_type,
                                 int64_t source_task_index, int64_t target_task_index,
                                 ffi::Array<ffi::Any> core_ranges,
                                 ffi::Map<ffi::String, ffi::Any> payload) {
  auto n = ffi::make_object<TTSemaphorePlanNode>();
  n->name = std::move(name);
  n->kind = std::move(kind);
  n->semaphore_id = semaphore_id;
  n->initial_value = initial_value;
  n->core_type = std::move(core_type);
  n->source_task_index = source_task_index;
  n->target_task_index = target_task_index;
  n->core_ranges = std::move(core_ranges);
  n->payload = std::move(payload);
  data_ = std::move(n);
}

void TTComputeSyncPlanNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTComputeSyncPlanNode>()
      .def_ro("name", &TTComputeSyncPlanNode::name)
      .def_ro("kind", &TTComputeSyncPlanNode::kind)
      .def_ro("source_task_index", &TTComputeSyncPlanNode::source_task_index)
      .def_ro("target_task_index", &TTComputeSyncPlanNode::target_task_index)
      .def_ro("ordering_kind", &TTComputeSyncPlanNode::ordering_kind)
      .def_ro("materialization_kind", &TTComputeSyncPlanNode::materialization_kind)
      .def_ro("payload", &TTComputeSyncPlanNode::payload);
}

TTComputeSyncPlan::TTComputeSyncPlan(ffi::String name, ffi::String kind, int64_t source_task_index,
                                     int64_t target_task_index, ffi::String ordering_kind,
                                     ffi::String materialization_kind,
                                     ffi::Map<ffi::String, ffi::Any> payload) {
  auto n = ffi::make_object<TTComputeSyncPlanNode>();
  n->name = std::move(name);
  n->kind = std::move(kind);
  n->source_task_index = source_task_index;
  n->target_task_index = target_task_index;
  n->ordering_kind = std::move(ordering_kind);
  n->materialization_kind = std::move(materialization_kind);
  n->payload = std::move(payload);
  data_ = std::move(n);
}

void TTDstLayoutPlanNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTDstLayoutPlanNode>()
      .def_ro("name", &TTDstLayoutPlanNode::name)
      .def_ro("buffer", &TTDstLayoutPlanNode::buffer)
      .def_ro("layout", &TTDstLayoutPlanNode::layout)
      .def_ro("memory_space", &TTDstLayoutPlanNode::memory_space)
      .def_ro("payload", &TTDstLayoutPlanNode::payload);
}

TTDstLayoutPlan::TTDstLayoutPlan(ffi::String name, ffi::String buffer, ffi::String layout,
                                 ffi::String memory_space,
                                 ffi::Map<ffi::String, ffi::Any> payload) {
  auto n = ffi::make_object<TTDstLayoutPlanNode>();
  n->name = std::move(name);
  n->buffer = std::move(buffer);
  n->layout = std::move(layout);
  n->memory_space = std::move(memory_space);
  n->payload = std::move(payload);
  data_ = std::move(n);
}

void TTABIPlanNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTABIPlanNode>()
      .def_ro("name", &TTABIPlanNode::name)
      .def_ro("kernel_name", &TTABIPlanNode::kernel_name)
      .def_ro("runtime_args", &TTABIPlanNode::runtime_args)
      .def_ro("common_runtime_args", &TTABIPlanNode::common_runtime_args)
      .def_ro("compile_time_arg_specs", &TTABIPlanNode::compile_time_arg_specs)
      .def_ro("accessors", &TTABIPlanNode::accessors)
      .def_ro("semaphore_bindings", &TTABIPlanNode::semaphore_bindings)
      .def_ro("payload", &TTABIPlanNode::payload);
}

TTABIPlan::TTABIPlan(ffi::String name, ffi::String kernel_name, ffi::Array<ffi::Any> runtime_args,
                     ffi::Array<ffi::Any> common_runtime_args,
                     ffi::Array<ffi::Any> compile_time_arg_specs,
                     ffi::Array<ffi::Any> accessors, ffi::Array<ffi::Any> semaphore_bindings,
                     ffi::Map<ffi::String, ffi::Any> payload) {
  auto n = ffi::make_object<TTABIPlanNode>();
  n->name = std::move(name);
  n->kernel_name = std::move(kernel_name);
  n->runtime_args = std::move(runtime_args);
  n->common_runtime_args = std::move(common_runtime_args);
  n->compile_time_arg_specs = std::move(compile_time_arg_specs);
  n->accessors = std::move(accessors);
  n->semaphore_bindings = std::move(semaphore_bindings);
  n->payload = std::move(payload);
  data_ = std::move(n);
}

void TTExecutionPlanNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTExecutionPlanNode>()
      .def_ro("name", &TTExecutionPlanNode::name)
      .def_ro("kernel_names", &TTExecutionPlanNode::kernel_names)
      .def_ro("phase_indices", &TTExecutionPlanNode::phase_indices)
      .def_ro("payload", &TTExecutionPlanNode::payload);
}

TTExecutionPlan::TTExecutionPlan(ffi::String name, ffi::Array<ffi::String> kernel_names,
                                 ffi::Array<Integer> phase_indices,
                                 ffi::Map<ffi::String, ffi::Any> payload) {
  auto n = ffi::make_object<TTExecutionPlanNode>();
  n->name = std::move(name);
  n->kernel_names = std::move(kernel_names);
  n->phase_indices = std::move(phase_indices);
  n->payload = std::move(payload);
  data_ = std::move(n);
}

void TTProgramNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTProgramNode>()
      .def_ro("entry_name", &TTProgramNode::entry_name)
      .def_ro("member_func", &TTProgramNode::member_func)
      .def_ro("kernels", &TTProgramNode::kernels)
      .def_ro("core_groups", &TTProgramNode::core_groups)
      .def_ro("cb_plans", &TTProgramNode::cb_plans)
      .def_ro("transport_plans", &TTProgramNode::transport_plans)
      .def_ro("semaphore_plans", &TTProgramNode::semaphore_plans)
      .def_ro("compute_sync_plans", &TTProgramNode::compute_sync_plans)
      .def_ro("dst_layout_plans", &TTProgramNode::dst_layout_plans)
      .def_ro("abi_plans", &TTProgramNode::abi_plans)
      .def_ro("execution_plans", &TTProgramNode::execution_plans)
      .def_ro("payload", &TTProgramNode::payload);
}

TTProgram::TTProgram(ffi::String entry_name, ffi::String member_func, ffi::Array<TTKernel> kernels,
                     ffi::Array<TTCoreGroup> core_groups, ffi::Array<TTCBPlan> cb_plans,
                     ffi::Array<TTTransportPlan> transport_plans,
                     ffi::Array<TTSemaphorePlan> semaphore_plans,
                     ffi::Array<TTComputeSyncPlan> compute_sync_plans,
                     ffi::Array<TTDstLayoutPlan> dst_layout_plans,
                     ffi::Array<TTABIPlan> abi_plans,
                     ffi::Array<TTExecutionPlan> execution_plans,
                     ffi::Map<ffi::String, ffi::Any> payload) {
  auto n = ffi::make_object<TTProgramNode>();
  n->entry_name = std::move(entry_name);
  n->member_func = std::move(member_func);
  n->kernels = std::move(kernels);
  n->core_groups = std::move(core_groups);
  n->cb_plans = std::move(cb_plans);
  n->transport_plans = std::move(transport_plans);
  n->semaphore_plans = std::move(semaphore_plans);
  n->compute_sync_plans = std::move(compute_sync_plans);
  n->dst_layout_plans = std::move(dst_layout_plans);
  n->abi_plans = std::move(abi_plans);
  n->execution_plans = std::move(execution_plans);
  n->payload = std::move(payload);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  RegisterNodeReflection<TTKernelNode>();
  RegisterNodeReflection<TTCoreGroupNode>();
  RegisterNodeReflection<TTCBPlanNode>();
  RegisterNodeReflection<TTTransportPlanNode>();
  RegisterNodeReflection<TTSemaphorePlanNode>();
  RegisterNodeReflection<TTComputeSyncPlanNode>();
  RegisterNodeReflection<TTDstLayoutPlanNode>();
  RegisterNodeReflection<TTABIPlanNode>();
  RegisterNodeReflection<TTExecutionPlanNode>();
  RegisterNodeReflection<TTProgramNode>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tl.TTKernel",
      [](ffi::String name, ffi::String kind, ffi::String core_type, int64_t abi_plan_index,
         ffi::Map<ffi::String, ffi::Any> payload) {
        return TTKernel(std::move(name), std::move(kind), std::move(core_type), abi_plan_index,
                        std::move(payload));
      });
  refl::GlobalDef().def(
      "tl.TTCoreGroup",
      [](ffi::String name, int64_t logical_grid_x, int64_t logical_grid_y,
         ffi::String linearization, ffi::Array<ffi::Any> physical_cores,
         ffi::Array<ffi::Any> work_packets, ffi::Map<ffi::String, ffi::Any> payload) {
        return TTCoreGroup(std::move(name), logical_grid_x, logical_grid_y,
                           std::move(linearization), std::move(physical_cores),
                           std::move(work_packets), std::move(payload));
      });
  refl::GlobalDef().def(
      "tl.TTCBPlan",
      [](ffi::String name, int64_t cb_id, ffi::String resource_class, int64_t num_pages,
         int64_t page_size_bytes, ffi::String data_format,
         ffi::Map<ffi::String, ffi::Any> payload) {
        return TTCBPlan(std::move(name), cb_id, std::move(resource_class), num_pages,
                        page_size_bytes, std::move(data_format), std::move(payload));
      });
  refl::GlobalDef().def(
      "tl.TTTransportPlan",
      [](ffi::String name, ffi::String kind, int64_t source_task_index, int64_t target_task_index,
         ffi::String payload_kind, ffi::String delivery_kind,
         ffi::Map<ffi::String, ffi::Any> payload) {
        return TTTransportPlan(std::move(name), std::move(kind), source_task_index,
                               target_task_index, std::move(payload_kind),
                               std::move(delivery_kind), std::move(payload));
      });
  refl::GlobalDef().def(
      "tl.TTSemaphorePlan",
      [](ffi::String name, ffi::String kind, int64_t semaphore_id, int64_t initial_value,
         ffi::String core_type, int64_t source_task_index, int64_t target_task_index,
         ffi::Array<ffi::Any> core_ranges, ffi::Map<ffi::String, ffi::Any> payload) {
        return TTSemaphorePlan(std::move(name), std::move(kind), semaphore_id, initial_value,
                               std::move(core_type), source_task_index, target_task_index,
                               std::move(core_ranges), std::move(payload));
      });
  refl::GlobalDef().def(
      "tl.TTComputeSyncPlan",
      [](ffi::String name, ffi::String kind, int64_t source_task_index, int64_t target_task_index,
         ffi::String ordering_kind, ffi::String materialization_kind,
         ffi::Map<ffi::String, ffi::Any> payload) {
        return TTComputeSyncPlan(std::move(name), std::move(kind), source_task_index,
                                 target_task_index, std::move(ordering_kind),
                                 std::move(materialization_kind), std::move(payload));
      });
  refl::GlobalDef().def(
      "tl.TTDstLayoutPlan",
      [](ffi::String name, ffi::String buffer, ffi::String layout, ffi::String memory_space,
         ffi::Map<ffi::String, ffi::Any> payload) {
        return TTDstLayoutPlan(std::move(name), std::move(buffer), std::move(layout),
                               std::move(memory_space), std::move(payload));
      });
  refl::GlobalDef().def(
      "tl.TTABIPlan",
      [](ffi::String name, ffi::String kernel_name, ffi::Array<ffi::Any> runtime_args,
         ffi::Array<ffi::Any> common_runtime_args, ffi::Array<ffi::Any> compile_time_arg_specs,
         ffi::Array<ffi::Any> accessors, ffi::Array<ffi::Any> semaphore_bindings,
         ffi::Map<ffi::String, ffi::Any> payload) {
        return TTABIPlan(std::move(name), std::move(kernel_name), std::move(runtime_args),
                         std::move(common_runtime_args), std::move(compile_time_arg_specs),
                         std::move(accessors), std::move(semaphore_bindings),
                         std::move(payload));
      });
  refl::GlobalDef().def(
      "tl.TTExecutionPlan",
      [](ffi::String name, ffi::Array<ffi::String> kernel_names, ffi::Array<Integer> phase_indices,
         ffi::Map<ffi::String, ffi::Any> payload) {
        return TTExecutionPlan(std::move(name), std::move(kernel_names),
                               std::move(phase_indices), std::move(payload));
      });
  refl::GlobalDef().def(
      "tl.TTProgram",
      [](ffi::String entry_name, ffi::String member_func, ffi::Array<TTKernel> kernels,
         ffi::Array<TTCoreGroup> core_groups, ffi::Array<TTCBPlan> cb_plans,
         ffi::Array<TTTransportPlan> transport_plans, ffi::Array<TTSemaphorePlan> semaphore_plans,
         ffi::Array<TTComputeSyncPlan> compute_sync_plans,
         ffi::Array<TTDstLayoutPlan> dst_layout_plans, ffi::Array<TTABIPlan> abi_plans,
         ffi::Array<TTExecutionPlan> execution_plans, ffi::Map<ffi::String, ffi::Any> payload) {
        return TTProgram(std::move(entry_name), std::move(member_func), std::move(kernels),
                         std::move(core_groups), std::move(cb_plans),
                         std::move(transport_plans), std::move(semaphore_plans),
                         std::move(compute_sync_plans), std::move(dst_layout_plans),
                         std::move(abi_plans), std::move(execution_plans),
                         std::move(payload));
      });
}

}  // namespace tl
}  // namespace tvm
