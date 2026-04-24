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

void TTMeshPlanNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTMeshPlanNode>()
      .def_ro("name", &TTMeshPlanNode::name)
      .def_ro("mesh_kind", &TTMeshPlanNode::mesh_kind)
      .def_ro("mesh_shape", &TTMeshPlanNode::mesh_shape)
      .def_ro("device_range_start", &TTMeshPlanNode::device_range_start)
      .def_ro("device_range_shape", &TTMeshPlanNode::device_range_shape)
      .def_ro("system_mesh_ref", &TTMeshPlanNode::system_mesh_ref)
      .def_ro("payload", &TTMeshPlanNode::payload);
}

TTMeshPlan::TTMeshPlan(ffi::String name, ffi::String mesh_kind,
                       ffi::Array<Integer> mesh_shape,
                       ffi::Array<Integer> device_range_start,
                       ffi::Array<Integer> device_range_shape,
                       ffi::String system_mesh_ref,
                       ffi::Map<ffi::String, ffi::Any> payload) {
  auto n = ffi::make_object<TTMeshPlanNode>();
  n->name = std::move(name);
  n->mesh_kind = std::move(mesh_kind);
  n->mesh_shape = std::move(mesh_shape);
  n->device_range_start = std::move(device_range_start);
  n->device_range_shape = std::move(device_range_shape);
  n->system_mesh_ref = std::move(system_mesh_ref);
  n->payload = std::move(payload);
  data_ = std::move(n);
}

void TTBufferDistributionPlanNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTBufferDistributionPlanNode>()
      .def_ro("name", &TTBufferDistributionPlanNode::name)
      .def_ro("buffer", &TTBufferDistributionPlanNode::buffer)
      .def_ro("mesh_plan", &TTBufferDistributionPlanNode::mesh_plan)
      .def_ro("mesh_plan_index", &TTBufferDistributionPlanNode::mesh_plan_index)
      .def_ro("distribution_kind", &TTBufferDistributionPlanNode::distribution_kind)
      .def_ro("layout", &TTBufferDistributionPlanNode::layout)
      .def_ro("memory_space", &TTBufferDistributionPlanNode::memory_space)
      .def_ro("page_size_bytes", &TTBufferDistributionPlanNode::page_size_bytes)
      .def_ro("shard_shape", &TTBufferDistributionPlanNode::shard_shape)
      .def_ro("shard_orientation", &TTBufferDistributionPlanNode::shard_orientation)
      .def_ro("host_visibility", &TTBufferDistributionPlanNode::host_visibility)
      .def_ro("payload", &TTBufferDistributionPlanNode::payload);
}

TTBufferDistributionPlan::TTBufferDistributionPlan(
    ffi::String name, ffi::String buffer, ffi::String mesh_plan, int64_t mesh_plan_index,
    ffi::String distribution_kind, ffi::String layout, ffi::String memory_space,
    int64_t page_size_bytes, ffi::Array<Integer> shard_shape, ffi::String shard_orientation,
    ffi::String host_visibility, ffi::Map<ffi::String, ffi::Any> payload) {
  auto n = ffi::make_object<TTBufferDistributionPlanNode>();
  n->name = std::move(name);
  n->buffer = std::move(buffer);
  n->mesh_plan = std::move(mesh_plan);
  n->mesh_plan_index = mesh_plan_index;
  n->distribution_kind = std::move(distribution_kind);
  n->layout = std::move(layout);
  n->memory_space = std::move(memory_space);
  n->page_size_bytes = page_size_bytes;
  n->shard_shape = std::move(shard_shape);
  n->shard_orientation = std::move(shard_orientation);
  n->host_visibility = std::move(host_visibility);
  n->payload = std::move(payload);
  data_ = std::move(n);
}

void TTComputeOperandBindingPlanNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTComputeOperandBindingPlanNode>()
      .def_ro("role", &TTComputeOperandBindingPlanNode::role)
      .def_ro("buffer", &TTComputeOperandBindingPlanNode::buffer)
      .def_ro("host_buffer", &TTComputeOperandBindingPlanNode::host_buffer)
      .def_ro("tensor_dtype", &TTComputeOperandBindingPlanNode::tensor_dtype)
      .def_ro("cb_dtype", &TTComputeOperandBindingPlanNode::cb_dtype)
      .def_ro("transform_kind", &TTComputeOperandBindingPlanNode::transform_kind)
      .def_ro("payload", &TTComputeOperandBindingPlanNode::payload);
}

TTComputeOperandBindingPlan::TTComputeOperandBindingPlan(
    ffi::String role, ffi::String buffer, ffi::String host_buffer, ffi::String tensor_dtype,
    ffi::String cb_dtype, ffi::String transform_kind, ffi::Map<ffi::String, ffi::Any> payload) {
  auto n = ffi::make_object<TTComputeOperandBindingPlanNode>();
  n->role = std::move(role);
  n->buffer = std::move(buffer);
  n->host_buffer = std::move(host_buffer);
  n->tensor_dtype = std::move(tensor_dtype);
  n->cb_dtype = std::move(cb_dtype);
  n->transform_kind = std::move(transform_kind);
  n->payload = std::move(payload);
  data_ = std::move(n);
}

void TTComputeOpPlanNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTComputeOpPlanNode>()
      .def_ro("name", &TTComputeOpPlanNode::name)
      .def_ro("kernel_name", &TTComputeOpPlanNode::kernel_name)
      .def_ro("kernel_plan_index", &TTComputeOpPlanNode::kernel_plan_index)
      .def_ro("kind", &TTComputeOpPlanNode::kind)
      .def_ro("enabled", &TTComputeOpPlanNode::enabled)
      .def_ro("operand_bindings", &TTComputeOpPlanNode::operand_bindings)
      .def_ro("problem_shape_axes", &TTComputeOpPlanNode::problem_shape_axes)
      .def_ro("problem_shape", &TTComputeOpPlanNode::problem_shape)
      .def_ro("tile_shape", &TTComputeOpPlanNode::tile_shape)
      .def_ro("block_shape", &TTComputeOpPlanNode::block_shape)
      .def_ro("subblock_shape", &TTComputeOpPlanNode::subblock_shape)
      .def_ro("accumulator_dtype", &TTComputeOpPlanNode::accumulator_dtype)
      .def_ro("mbarrier_buffer", &TTComputeOpPlanNode::mbarrier_buffer)
      .def_ro("mbarrier_scope", &TTComputeOpPlanNode::mbarrier_scope)
      .def_ro("mbarrier_index_exprs", &TTComputeOpPlanNode::mbarrier_index_exprs)
      .def_ro("payload", &TTComputeOpPlanNode::payload);
}

TTComputeOpPlan::TTComputeOpPlan(
    ffi::String name, ffi::String kernel_name, int64_t kernel_plan_index, ffi::String kind,
    bool enabled, ffi::Array<TTComputeOperandBindingPlan> operand_bindings,
    ffi::Array<ffi::String> problem_shape_axes, ffi::Array<Integer> problem_shape,
    ffi::Array<Integer> tile_shape, ffi::Array<Integer> block_shape,
    ffi::Array<Integer> subblock_shape, ffi::String accumulator_dtype,
    ffi::String mbarrier_buffer, ffi::String mbarrier_scope,
    ffi::Array<ffi::String> mbarrier_index_exprs, ffi::Map<ffi::String, ffi::Any> payload) {
  auto n = ffi::make_object<TTComputeOpPlanNode>();
  n->name = std::move(name);
  n->kernel_name = std::move(kernel_name);
  n->kernel_plan_index = kernel_plan_index;
  n->kind = std::move(kind);
  n->enabled = enabled;
  n->operand_bindings = std::move(operand_bindings);
  n->problem_shape_axes = std::move(problem_shape_axes);
  n->problem_shape = std::move(problem_shape);
  n->tile_shape = std::move(tile_shape);
  n->block_shape = std::move(block_shape);
  n->subblock_shape = std::move(subblock_shape);
  n->accumulator_dtype = std::move(accumulator_dtype);
  n->mbarrier_buffer = std::move(mbarrier_buffer);
  n->mbarrier_scope = std::move(mbarrier_scope);
  n->mbarrier_index_exprs = std::move(mbarrier_index_exprs);
  n->payload = std::move(payload);
  data_ = std::move(n);
}

void TTBlockPlanNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTBlockPlanNode>()
      .def_ro("name", &TTBlockPlanNode::name)
      .def_ro("placement_kind", &TTBlockPlanNode::placement_kind)
      .def_ro("task_indices", &TTBlockPlanNode::task_indices)
      .def_ro("payload", &TTBlockPlanNode::payload);
}

TTBlockPlan::TTBlockPlan(ffi::String name, ffi::String placement_kind,
                         ffi::Array<Integer> task_indices,
                         ffi::Map<ffi::String, ffi::Any> payload) {
  auto n = ffi::make_object<TTBlockPlanNode>();
  n->name = std::move(name);
  n->placement_kind = std::move(placement_kind);
  n->task_indices = std::move(task_indices);
  n->payload = std::move(payload);
  data_ = std::move(n);
}

void TTKernelPlanNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTKernelPlanNode>()
      .def_ro("name", &TTKernelPlanNode::name)
      .def_ro("kind", &TTKernelPlanNode::kind)
      .def_ro("core_type", &TTKernelPlanNode::core_type)
      .def_ro("block_plan_index", &TTKernelPlanNode::block_plan_index)
      .def_ro("abi_plan_index", &TTKernelPlanNode::abi_plan_index)
      .def_ro("payload", &TTKernelPlanNode::payload);
}

TTKernelPlan::TTKernelPlan(ffi::String name, ffi::String kind, ffi::String core_type,
                           int64_t block_plan_index, int64_t abi_plan_index,
                           ffi::Map<ffi::String, ffi::Any> payload) {
  auto n = ffi::make_object<TTKernelPlanNode>();
  n->name = std::move(name);
  n->kind = std::move(kind);
  n->core_type = std::move(core_type);
  n->block_plan_index = block_plan_index;
  n->abi_plan_index = abi_plan_index;
  n->payload = std::move(payload);
  data_ = std::move(n);
}

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
      .def_ro("initial_reserve_pages", &TTCBPlanNode::initial_reserve_pages)
      .def_ro("flow_class", &TTCBPlanNode::flow_class)
      .def_ro("publish_pages_per_event", &TTCBPlanNode::publish_pages_per_event)
      .def_ro("consume_pages_per_event", &TTCBPlanNode::consume_pages_per_event)
      .def_ro("lifetime_begin", &TTCBPlanNode::lifetime_begin)
      .def_ro("lifetime_end", &TTCBPlanNode::lifetime_end)
      .def_ro("payload", &TTCBPlanNode::payload);
}

TTCBPlan::TTCBPlan(ffi::String name, int64_t cb_id, ffi::String resource_class, int64_t num_pages,
                   int64_t page_size_bytes, ffi::String data_format,
                   int64_t initial_reserve_pages, ffi::String flow_class,
                   int64_t publish_pages_per_event, int64_t consume_pages_per_event,
                   int64_t lifetime_begin, int64_t lifetime_end,
                   ffi::Map<ffi::String, ffi::Any> payload) {
  auto n = ffi::make_object<TTCBPlanNode>();
  n->name = std::move(name);
  n->cb_id = cb_id;
  n->resource_class = std::move(resource_class);
  n->num_pages = num_pages;
  n->page_size_bytes = page_size_bytes;
  n->data_format = std::move(data_format);
  n->initial_reserve_pages = initial_reserve_pages;
  n->flow_class = std::move(flow_class);
  n->publish_pages_per_event = publish_pages_per_event;
  n->consume_pages_per_event = consume_pages_per_event;
  n->lifetime_begin = lifetime_begin;
  n->lifetime_end = lifetime_end;
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

void TTSyncPlanNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTSyncPlanNode>()
      .def_ro("name", &TTSyncPlanNode::name)
      .def_ro("kind", &TTSyncPlanNode::kind)
      .def_ro("source_task_index", &TTSyncPlanNode::source_task_index)
      .def_ro("target_task_index", &TTSyncPlanNode::target_task_index)
      .def_ro("ordering_kind", &TTSyncPlanNode::ordering_kind)
      .def_ro("completion_kind", &TTSyncPlanNode::completion_kind)
      .def_ro("payload", &TTSyncPlanNode::payload);
}

TTSyncPlan::TTSyncPlan(ffi::String name, ffi::String kind, int64_t source_task_index,
                       int64_t target_task_index, ffi::String ordering_kind,
                       ffi::String completion_kind,
                       ffi::Map<ffi::String, ffi::Any> payload) {
  auto n = ffi::make_object<TTSyncPlanNode>();
  n->name = std::move(name);
  n->kind = std::move(kind);
  n->source_task_index = source_task_index;
  n->target_task_index = target_task_index;
  n->ordering_kind = std::move(ordering_kind);
  n->completion_kind = std::move(completion_kind);
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

void TTLiveFormPlanNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTLiveFormPlanNode>()
      .def_ro("name", &TTLiveFormPlanNode::name)
      .def_ro("logical_value", &TTLiveFormPlanNode::logical_value)
      .def_ro("spatial_live_value", &TTLiveFormPlanNode::spatial_live_value)
      .def_ro("spatial_live_value_index", &TTLiveFormPlanNode::spatial_live_value_index)
      .def_ro("producer_kernel", &TTLiveFormPlanNode::producer_kernel)
      .def_ro("physical_form", &TTLiveFormPlanNode::physical_form)
      .def_ro("execution_topology", &TTLiveFormPlanNode::execution_topology)
      .def_ro("physical_local_extent", &TTLiveFormPlanNode::physical_local_extent)
      .def_ro("logical_element_count", &TTLiveFormPlanNode::logical_element_count)
      .def_ro("ownership_kind", &TTLiveFormPlanNode::ownership_kind)
      .def_ro("payload", &TTLiveFormPlanNode::payload);
}

TTLiveFormPlan::TTLiveFormPlan(ffi::String name, ffi::String logical_value,
                               ffi::String spatial_live_value,
                               int64_t spatial_live_value_index,
                               ffi::String producer_kernel, ffi::String physical_form,
                               ffi::String execution_topology, int64_t physical_local_extent,
                               int64_t logical_element_count, ffi::String ownership_kind,
                               ffi::Map<ffi::String, ffi::Any> payload) {
  auto n = ffi::make_object<TTLiveFormPlanNode>();
  n->name = std::move(name);
  n->logical_value = std::move(logical_value);
  n->spatial_live_value = std::move(spatial_live_value);
  n->spatial_live_value_index = spatial_live_value_index;
  n->producer_kernel = std::move(producer_kernel);
  n->physical_form = std::move(physical_form);
  n->execution_topology = std::move(execution_topology);
  n->physical_local_extent = physical_local_extent;
  n->logical_element_count = logical_element_count;
  n->ownership_kind = std::move(ownership_kind);
  n->payload = std::move(payload);
  data_ = std::move(n);
}

void TTMaterializationPlanNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTMaterializationPlanNode>()
      .def_ro("name", &TTMaterializationPlanNode::name)
      .def_ro("source_live_form", &TTMaterializationPlanNode::source_live_form)
      .def_ro("materialization_boundary",
              &TTMaterializationPlanNode::materialization_boundary)
      .def_ro("materialization_boundary_index",
              &TTMaterializationPlanNode::materialization_boundary_index)
      .def_ro("target_buffer", &TTMaterializationPlanNode::target_buffer)
      .def_ro("host_buffer", &TTMaterializationPlanNode::host_buffer)
      .def_ro("target_kernel", &TTMaterializationPlanNode::target_kernel)
      .def_ro("materialization_protocol", &TTMaterializationPlanNode::materialization_protocol)
      .def_ro("publication_protocol", &TTMaterializationPlanNode::publication_protocol)
      .def_ro("required_cb_plan_indices",
              &TTMaterializationPlanNode::required_cb_plan_indices)
      .def_ro("required_sync_plan_indices",
              &TTMaterializationPlanNode::required_sync_plan_indices)
      .def_ro("produced_live_form", &TTMaterializationPlanNode::produced_live_form)
      .def_ro("payload", &TTMaterializationPlanNode::payload);
}

TTMaterializationPlan::TTMaterializationPlan(
    ffi::String name, ffi::String source_live_form, ffi::String materialization_boundary,
    int64_t materialization_boundary_index, ffi::String target_buffer,
    ffi::String host_buffer, ffi::String target_kernel, ffi::String materialization_protocol,
    ffi::String publication_protocol,
    ffi::Array<Integer> required_cb_plan_indices,
    ffi::Array<Integer> required_sync_plan_indices, ffi::String produced_live_form,
    ffi::Map<ffi::String, ffi::Any> payload) {
  auto n = ffi::make_object<TTMaterializationPlanNode>();
  n->name = std::move(name);
  n->source_live_form = std::move(source_live_form);
  n->materialization_boundary = std::move(materialization_boundary);
  n->materialization_boundary_index = materialization_boundary_index;
  n->target_buffer = std::move(target_buffer);
  n->host_buffer = std::move(host_buffer);
  n->target_kernel = std::move(target_kernel);
  n->materialization_protocol = std::move(materialization_protocol);
  n->publication_protocol = std::move(publication_protocol);
  n->required_cb_plan_indices = std::move(required_cb_plan_indices);
  n->required_sync_plan_indices = std::move(required_sync_plan_indices);
  n->produced_live_form = std::move(produced_live_form);
  n->payload = std::move(payload);
  data_ = std::move(n);
}

void TTConsumerBindingPlanNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TTConsumerBindingPlanNode>()
      .def_ro("name", &TTConsumerBindingPlanNode::name)
      .def_ro("consumer_kernel", &TTConsumerBindingPlanNode::consumer_kernel)
      .def_ro("consumer_op_kind", &TTConsumerBindingPlanNode::consumer_op_kind)
      .def_ro("source_live_form", &TTConsumerBindingPlanNode::source_live_form)
      .def_ro("live_value_edge", &TTConsumerBindingPlanNode::live_value_edge)
      .def_ro("live_value_edge_index", &TTConsumerBindingPlanNode::live_value_edge_index)
      .def_ro("accepts_distributed_slice",
              &TTConsumerBindingPlanNode::accepts_distributed_slice)
      .def_ro("requires_full_logical_tile",
              &TTConsumerBindingPlanNode::requires_full_logical_tile)
      .def_ro("abi_plan_index", &TTConsumerBindingPlanNode::abi_plan_index)
      .def_ro("payload", &TTConsumerBindingPlanNode::payload);
}

TTConsumerBindingPlan::TTConsumerBindingPlan(
    ffi::String name, ffi::String consumer_kernel, ffi::String consumer_op_kind,
    ffi::String source_live_form, ffi::String live_value_edge, int64_t live_value_edge_index,
    bool accepts_distributed_slice,
    bool requires_full_logical_tile, int64_t abi_plan_index,
    ffi::Map<ffi::String, ffi::Any> payload) {
  auto n = ffi::make_object<TTConsumerBindingPlanNode>();
  n->name = std::move(name);
  n->consumer_kernel = std::move(consumer_kernel);
  n->consumer_op_kind = std::move(consumer_op_kind);
  n->source_live_form = std::move(source_live_form);
  n->live_value_edge = std::move(live_value_edge);
  n->live_value_edge_index = live_value_edge_index;
  n->accepts_distributed_slice = accepts_distributed_slice;
  n->requires_full_logical_tile = requires_full_logical_tile;
  n->abi_plan_index = abi_plan_index;
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
      .def_ro("mesh_plans", &TTProgramNode::mesh_plans)
      .def_ro("buffer_distribution_plans", &TTProgramNode::buffer_distribution_plans)
      .def_ro("block_plans", &TTProgramNode::block_plans)
      .def_ro("kernel_plans", &TTProgramNode::kernel_plans)
      .def_ro("compute_op_plans", &TTProgramNode::compute_op_plans)
      .def_ro("transport_plans", &TTProgramNode::transport_plans)
      .def_ro("sync_plans", &TTProgramNode::sync_plans)
      .def_ro("abi_plans", &TTProgramNode::abi_plans)
      .def_ro("execution_plans", &TTProgramNode::execution_plans)
      .def_ro("kernels", &TTProgramNode::kernels)
      .def_ro("core_groups", &TTProgramNode::core_groups)
      .def_ro("cb_plans", &TTProgramNode::cb_plans)
      .def_ro("semaphore_plans", &TTProgramNode::semaphore_plans)
      .def_ro("compute_sync_plans", &TTProgramNode::compute_sync_plans)
      .def_ro("dst_layout_plans", &TTProgramNode::dst_layout_plans)
      .def_ro("live_form_plans", &TTProgramNode::live_form_plans)
      .def_ro("materialization_plans", &TTProgramNode::materialization_plans)
      .def_ro("consumer_binding_plans", &TTProgramNode::consumer_binding_plans)
      .def_ro("payload", &TTProgramNode::payload);
}

TTProgram::TTProgram(ffi::String entry_name, ffi::String member_func,
                     ffi::Array<TTMeshPlan> mesh_plans,
                     ffi::Array<TTBufferDistributionPlan> buffer_distribution_plans,
                     ffi::Array<TTBlockPlan> block_plans,
                     ffi::Array<TTKernelPlan> kernel_plans,
                     ffi::Array<TTComputeOpPlan> compute_op_plans,
                     ffi::Array<TTTransportPlan> transport_plans,
                     ffi::Array<TTSyncPlan> sync_plans,
                     ffi::Array<TTABIPlan> abi_plans,
                     ffi::Array<TTExecutionPlan> execution_plans,
                     ffi::Array<TTKernel> kernels, ffi::Array<TTCoreGroup> core_groups,
                     ffi::Array<TTCBPlan> cb_plans,
                     ffi::Array<TTSemaphorePlan> semaphore_plans,
                     ffi::Array<TTComputeSyncPlan> compute_sync_plans,
                     ffi::Array<TTDstLayoutPlan> dst_layout_plans,
                     ffi::Array<TTLiveFormPlan> live_form_plans,
                     ffi::Array<TTMaterializationPlan> materialization_plans,
                     ffi::Array<TTConsumerBindingPlan> consumer_binding_plans,
                     ffi::Map<ffi::String, ffi::Any> payload) {
  auto n = ffi::make_object<TTProgramNode>();
  n->entry_name = std::move(entry_name);
  n->member_func = std::move(member_func);
  n->mesh_plans = std::move(mesh_plans);
  n->buffer_distribution_plans = std::move(buffer_distribution_plans);
  n->block_plans = std::move(block_plans);
  n->kernel_plans = std::move(kernel_plans);
  n->compute_op_plans = std::move(compute_op_plans);
  n->transport_plans = std::move(transport_plans);
  n->sync_plans = std::move(sync_plans);
  n->abi_plans = std::move(abi_plans);
  n->execution_plans = std::move(execution_plans);
  n->kernels = std::move(kernels);
  n->core_groups = std::move(core_groups);
  n->cb_plans = std::move(cb_plans);
  n->semaphore_plans = std::move(semaphore_plans);
  n->compute_sync_plans = std::move(compute_sync_plans);
  n->dst_layout_plans = std::move(dst_layout_plans);
  n->live_form_plans = std::move(live_form_plans);
  n->materialization_plans = std::move(materialization_plans);
  n->consumer_binding_plans = std::move(consumer_binding_plans);
  n->payload = std::move(payload);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  RegisterNodeReflection<TTMeshPlanNode>();
  RegisterNodeReflection<TTBufferDistributionPlanNode>();
  RegisterNodeReflection<TTComputeOperandBindingPlanNode>();
  RegisterNodeReflection<TTComputeOpPlanNode>();
  RegisterNodeReflection<TTBlockPlanNode>();
  RegisterNodeReflection<TTKernelPlanNode>();
  RegisterNodeReflection<TTKernelNode>();
  RegisterNodeReflection<TTCoreGroupNode>();
  RegisterNodeReflection<TTCBPlanNode>();
  RegisterNodeReflection<TTTransportPlanNode>();
  RegisterNodeReflection<TTSyncPlanNode>();
  RegisterNodeReflection<TTSemaphorePlanNode>();
  RegisterNodeReflection<TTComputeSyncPlanNode>();
  RegisterNodeReflection<TTDstLayoutPlanNode>();
  RegisterNodeReflection<TTLiveFormPlanNode>();
  RegisterNodeReflection<TTMaterializationPlanNode>();
  RegisterNodeReflection<TTConsumerBindingPlanNode>();
  RegisterNodeReflection<TTABIPlanNode>();
  RegisterNodeReflection<TTExecutionPlanNode>();
  RegisterNodeReflection<TTProgramNode>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tl.TTMeshPlan",
      [](ffi::String name, ffi::String mesh_kind, ffi::Array<Integer> mesh_shape,
         ffi::Array<Integer> device_range_start, ffi::Array<Integer> device_range_shape,
         ffi::String system_mesh_ref, ffi::Map<ffi::String, ffi::Any> payload) {
        return TTMeshPlan(std::move(name), std::move(mesh_kind), std::move(mesh_shape),
                          std::move(device_range_start), std::move(device_range_shape),
                          std::move(system_mesh_ref), std::move(payload));
      });
  refl::GlobalDef().def(
      "tl.TTBufferDistributionPlan",
      [](ffi::String name, ffi::String buffer, ffi::String mesh_plan,
         int64_t mesh_plan_index, ffi::String distribution_kind, ffi::String layout,
         ffi::String memory_space, int64_t page_size_bytes, ffi::Array<Integer> shard_shape,
         ffi::String shard_orientation, ffi::String host_visibility,
         ffi::Map<ffi::String, ffi::Any> payload) {
        return TTBufferDistributionPlan(
            std::move(name), std::move(buffer), std::move(mesh_plan), mesh_plan_index,
            std::move(distribution_kind), std::move(layout), std::move(memory_space),
            page_size_bytes, std::move(shard_shape), std::move(shard_orientation),
            std::move(host_visibility), std::move(payload));
      });
  refl::GlobalDef().def(
      "tl.TTComputeOperandBindingPlan",
      [](ffi::String role, ffi::String buffer, ffi::String host_buffer,
         ffi::String tensor_dtype, ffi::String cb_dtype, ffi::String transform_kind,
         ffi::Map<ffi::String, ffi::Any> payload) {
        return TTComputeOperandBindingPlan(
            std::move(role), std::move(buffer), std::move(host_buffer),
            std::move(tensor_dtype), std::move(cb_dtype), std::move(transform_kind),
            std::move(payload));
      });
  refl::GlobalDef().def(
      "tl.TTComputeOpPlan",
      [](ffi::String name, ffi::String kernel_name, int64_t kernel_plan_index,
         ffi::String kind, bool enabled,
         ffi::Array<TTComputeOperandBindingPlan> operand_bindings,
         ffi::Array<ffi::String> problem_shape_axes, ffi::Array<Integer> problem_shape,
         ffi::Array<Integer> tile_shape, ffi::Array<Integer> block_shape,
         ffi::Array<Integer> subblock_shape, ffi::String accumulator_dtype,
         ffi::String mbarrier_buffer, ffi::String mbarrier_scope,
         ffi::Array<ffi::String> mbarrier_index_exprs,
         ffi::Map<ffi::String, ffi::Any> payload) {
        return TTComputeOpPlan(
            std::move(name), std::move(kernel_name), kernel_plan_index, std::move(kind),
            enabled, std::move(operand_bindings), std::move(problem_shape_axes),
            std::move(problem_shape), std::move(tile_shape), std::move(block_shape),
            std::move(subblock_shape), std::move(accumulator_dtype),
            std::move(mbarrier_buffer), std::move(mbarrier_scope),
            std::move(mbarrier_index_exprs), std::move(payload));
      });
  refl::GlobalDef().def(
      "tl.TTBlockPlan",
      [](ffi::String name, ffi::String placement_kind, ffi::Array<Integer> task_indices,
         ffi::Map<ffi::String, ffi::Any> payload) {
        return TTBlockPlan(std::move(name), std::move(placement_kind), std::move(task_indices),
                           std::move(payload));
      });
  refl::GlobalDef().def(
      "tl.TTKernelPlan",
      [](ffi::String name, ffi::String kind, ffi::String core_type, int64_t block_plan_index,
         int64_t abi_plan_index, ffi::Map<ffi::String, ffi::Any> payload) {
        return TTKernelPlan(std::move(name), std::move(kind), std::move(core_type),
                            block_plan_index, abi_plan_index, std::move(payload));
      });
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
         int64_t page_size_bytes, ffi::String data_format, int64_t initial_reserve_pages,
         ffi::String flow_class, int64_t publish_pages_per_event,
         int64_t consume_pages_per_event, int64_t lifetime_begin, int64_t lifetime_end,
         ffi::Map<ffi::String, ffi::Any> payload) {
        return TTCBPlan(std::move(name), cb_id, std::move(resource_class), num_pages,
                        page_size_bytes, std::move(data_format), initial_reserve_pages,
                        std::move(flow_class), publish_pages_per_event,
                        consume_pages_per_event, lifetime_begin, lifetime_end,
                        std::move(payload));
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
      "tl.TTSyncPlan",
      [](ffi::String name, ffi::String kind, int64_t source_task_index, int64_t target_task_index,
         ffi::String ordering_kind, ffi::String completion_kind,
         ffi::Map<ffi::String, ffi::Any> payload) {
        return TTSyncPlan(std::move(name), std::move(kind), source_task_index,
                          target_task_index, std::move(ordering_kind),
                          std::move(completion_kind), std::move(payload));
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
      "tl.TTLiveFormPlan",
      [](ffi::String name, ffi::String logical_value, ffi::String producer_kernel,
         ffi::String physical_form, ffi::String execution_topology,
         int64_t physical_local_extent, int64_t logical_element_count,
         ffi::String ownership_kind, ffi::Map<ffi::String, ffi::Any> payload) {
        ffi::String spatial_live_value;
        int64_t spatial_live_value_index = -1;
        if (auto value = payload.Get("spatial_live_value")) {
          spatial_live_value = Downcast<ffi::String>(value.value());
        }
        if (auto value = payload.Get("spatial_live_value_index")) {
          spatial_live_value_index = Downcast<Integer>(value.value())->value;
        }
        return TTLiveFormPlan(std::move(name), std::move(logical_value),
                              std::move(spatial_live_value), spatial_live_value_index,
                              std::move(producer_kernel), std::move(physical_form),
                              std::move(execution_topology), physical_local_extent,
                              logical_element_count, std::move(ownership_kind),
                              std::move(payload));
      });
  refl::GlobalDef().def(
      "tl.TTMaterializationPlan",
      [](ffi::String name, ffi::String source_live_form, ffi::String target_buffer,
         ffi::String target_kernel, ffi::String materialization_protocol,
         ffi::String publication_protocol,
         ffi::Array<Integer> required_cb_plan_indices,
         ffi::Array<Integer> required_sync_plan_indices, ffi::String produced_live_form,
         ffi::Map<ffi::String, ffi::Any> payload) {
        ffi::String materialization_boundary;
        ffi::String host_buffer;
        int64_t materialization_boundary_index = -1;
        if (auto value = payload.Get("materialization_boundary")) {
          materialization_boundary = Downcast<ffi::String>(value.value());
        }
        if (auto value = payload.Get("host_buffer")) {
          host_buffer = Downcast<ffi::String>(value.value());
        }
        if (auto value = payload.Get("materialization_boundary_index")) {
          materialization_boundary_index = Downcast<Integer>(value.value())->value;
        }
        return TTMaterializationPlan(
            std::move(name), std::move(source_live_form), std::move(materialization_boundary),
            materialization_boundary_index, std::move(target_buffer), std::move(host_buffer),
            std::move(target_kernel), std::move(materialization_protocol), std::move(publication_protocol),
            std::move(required_cb_plan_indices), std::move(required_sync_plan_indices),
            std::move(produced_live_form), std::move(payload));
      });
  refl::GlobalDef().def(
      "tl.TTConsumerBindingPlan",
      [](ffi::String name, ffi::String consumer_kernel, ffi::String consumer_op_kind,
         ffi::String source_live_form, bool accepts_distributed_slice,
         bool requires_full_logical_tile, int64_t abi_plan_index,
         ffi::Map<ffi::String, ffi::Any> payload) {
        ffi::String live_value_edge;
        int64_t live_value_edge_index = -1;
        if (auto value = payload.Get("live_value_edge")) {
          live_value_edge = Downcast<ffi::String>(value.value());
        }
        if (auto value = payload.Get("live_value_edge_index")) {
          live_value_edge_index = Downcast<Integer>(value.value())->value;
        }
        return TTConsumerBindingPlan(std::move(name), std::move(consumer_kernel),
                                     std::move(consumer_op_kind), std::move(source_live_form),
                                     std::move(live_value_edge), live_value_edge_index,
                                     accepts_distributed_slice, requires_full_logical_tile,
                                     abi_plan_index, std::move(payload));
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
      [](ffi::String entry_name, ffi::String member_func, ffi::Array<TTMeshPlan> mesh_plans,
         ffi::Array<TTBufferDistributionPlan> buffer_distribution_plans,
         ffi::Array<TTBlockPlan> block_plans,
         ffi::Array<TTKernelPlan> kernel_plans,
         ffi::Array<TTComputeOpPlan> compute_op_plans,
         ffi::Array<TTTransportPlan> transport_plans,
         ffi::Array<TTSyncPlan> sync_plans, ffi::Array<TTABIPlan> abi_plans,
         ffi::Array<TTExecutionPlan> execution_plans, ffi::Array<TTKernel> kernels,
         ffi::Array<TTCoreGroup> core_groups, ffi::Array<TTCBPlan> cb_plans,
         ffi::Array<TTSemaphorePlan> semaphore_plans,
         ffi::Array<TTComputeSyncPlan> compute_sync_plans,
         ffi::Array<TTDstLayoutPlan> dst_layout_plans,
         ffi::Array<TTLiveFormPlan> live_form_plans,
         ffi::Array<TTMaterializationPlan> materialization_plans,
         ffi::Array<TTConsumerBindingPlan> consumer_binding_plans,
         ffi::Map<ffi::String, ffi::Any> payload) {
        return TTProgram(std::move(entry_name), std::move(member_func),
                         std::move(mesh_plans), std::move(buffer_distribution_plans),
                         std::move(block_plans), std::move(kernel_plans),
                         std::move(compute_op_plans),
                         std::move(transport_plans), std::move(sync_plans),
                         std::move(abi_plans), std::move(execution_plans),
                         std::move(kernels), std::move(core_groups), std::move(cb_plans),
                         std::move(semaphore_plans), std::move(compute_sync_plans),
                         std::move(dst_layout_plans), std::move(live_form_plans),
                         std::move(materialization_plans),
                         std::move(consumer_binding_plans), std::move(payload));
      });
}

}  // namespace tl
}  // namespace tvm
